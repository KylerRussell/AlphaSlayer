"""PPO on combat, warm-started from the behaviour-cloning policy.

Why PPO and not search: the blocker for MCTS is mid-combat state snapshot/restore, which the
engine does not provide. PPO needs none of it, handles the game's mid-turn randomness
natively, and can exceed the demonstrator -- which behaviour cloning cannot, by construction.

Returns are Monte Carlo to the episode terminal rather than GAE. Combat episodes are short
(~8 turns, ~25 decisions) and the reward is sparse and terminal, so bootstrapping buys little
and MC keeps the credit assignment unbiased.

    python train_rl.py --init combat_bc.pt --iters 20 --envs 24
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from alphaslayer.encoder import Encoder, step_from_json
from alphaslayer.model import CombatNet, param_count, use_stable_attention
from alphaslayer.vecenv import VecSpireEnv

BATCH_KEYS = ("hand", "hand_mask", "enemies", "enemy_mask", "relics", "relic_mask",
              "powers", "power_mask", "enemy_powers", "enemy_power_mask",
              "bags", "globals", "actions", "action_mask")


class Rollout:
    """Per-env step buffers, flushed into training tensors when an episode terminates."""

    def __init__(self):
        self.open = defaultdict(list)   # env_idx -> [step dicts awaiting a reward]
        self.done = []                  # completed, reward-labelled steps
        self.episodes = []

    def add(self, env_idx, enc_step, action, logp, value, phi, bonus=0.0):
        self.open[env_idx].append(
            dict(enc=enc_step, action=action, logp=logp, value=value, phi=phi, bonus=bonus))

    def finish(self, env_idx, reward, result, phi_end, shaping):
        """Assigns each step its return, with potential-based shaping.

        Boss fights are long and the reward is a single win/loss bit, and with Pandora's the
        deck is random -- so most of the return variance is deck luck the policy cannot
        control, which drowns the learning signal. Potential-based shaping adds a DENSE term
        without changing the optimal policy (Ng et al. 1999): with gamma=1 the per-step
        shaping rewards telescope, so

            return_t = terminal_reward + w * (phi_end - phi_t)

        where phi = player_hp_frac - enemy_hp_frac. A step is now credited for the damage and
        HP preservation that actually followed it, instead of only for the final outcome.
        """
        steps = self.open.pop(env_idx, [])
        for s in steps:
            s["ret"] = reward + shaping * (phi_end - s["phi"]) + s.get("bonus", 0.0)
        self.done.extend(steps)
        self.episodes.append(result)

    def drop_incomplete(self):
        """Steps from an episode that never terminated have no reward and must not train."""
        n = sum(len(v) for v in self.open.values())
        self.open.clear()
        return n


def load_compat(net, state):
    """Loads a checkpoint, growing any embedding that gained rows.

    kind_emb went 2 -> 4 rows when select_card/select_done became distinct action kinds; the
    old rows still mean what they meant, so copy them and leave the new ones freshly
    initialised rather than discarding the checkpoint.
    """
    # Additive input blocks need an explicit re-layout before the generic overlap copy.
    try:
        from alphaslayer.model import migrate_enemy_proj
        if any(k == "enemy_proj.0.weight" for k in state):
            want = net.state_dict().get("enemy_proj.0.weight")
            if want is not None and state["enemy_proj.0.weight"].shape != want.shape:
                state = migrate_enemy_proj(state, d_model=want.shape[0])
    except Exception as e:
        print(f"  enemy_proj migration skipped: {e}")
    own = net.state_dict()
    fixed = {}
    for k, v in state.items():
        t = own.get(k)
        if t is not None and t.shape != v.shape and v.dim() == t.dim():
            grown = t.clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(t.shape, v.shape))
            grown[slices] = v[slices]
            fixed[k] = grown
            print(f"  grew {k}: {tuple(v.shape)} -> {tuple(t.shape)}")
        else:
            fixed[k] = v
    # New modules (e.g. relic_proj) have no counterpart in an older checkpoint;
    # load what matches and leave the rest freshly initialised.
    missing, unexpected = net.load_state_dict(fixed, strict=False)
    if missing:
        print(f'  new (freshly initialised): {len(missing)} tensor(s)')
    # UNEXPECTED keys mean the checkpoint carries weights this net has no slot for, i.e. the
    # net was built with the wrong architecture and those weights are being DISCARDED. That
    # silently produced an eval running a trained body on a stale, never-gradient-updated
    # shared head. It is never benign, so it is loud.
    if unexpected:
        heads = sorted({k.split('.')[0] for k in unexpected})
        print(f'  WARNING: {len(unexpected)} checkpoint tensor(s) DISCARDED - this net has '
              f'no slot for them: {heads}. The architecture does not match the checkpoint.')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="combat_bc.pt", help="BC checkpoint to warm start from")
    ap.add_argument("--vocab", default="/tmp/alphaslayer_probe/vocab.json")
    ap.add_argument("--out", default="combat_rl.pt")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--episodes-per-iter", type=int, default=240)
    ap.add_argument("--character", default="IRONCLAD")
    ap.add_argument("--encounters", default=None, help="regular|elite|boss|all")
    ap.add_argument("--relics", default=None, help="comma-separated relics to grant")
    ap.add_argument("--characters", default=None,
                    help="comma-separated roster, round-robined across envs")
    ap.add_argument("--mix", default=None, help="boss,elite,regular weights")
    ap.add_argument("--potion-bonus", type=float, default=0.0,
                    help="EXPLORATION bonus for drinking a potion, weighted by how rarely "
                         "that potion TYPE has been tried and annealed to zero. A flat bonus "
                         "would not be potential-based and would change the optimal policy -- "
                         "teaching 'drink potions' rather than 'drink potions well'. Weighting "
                         "by novelty targets coverage of what each potion DOES; annealing "
                         "means the final policy is optimal under the true reward")
    ap.add_argument("--potion-bonus-anneal", type=int, default=60,
                    help="iterations over which --potion-bonus decays to zero")
    ap.add_argument("--act", type=int, default=0,
                    help="which act's encounter pools to fight (0-indexed). Act 2 bosses and "
                         "elites are a harder target than act 1")
    ap.add_argument("--potions", action="store_true",
                    help="open the potion gate so use_potion is a legal combat action and the "
                         "policy can actually LEARN potion usage")
    ap.add_argument("--reroll-deck", action="store_true",
                    help="re-roll the injected-relic deck transform every episode")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--shaping", type=float, default=0.0,
                    help="potential-based shaping weight on (player_hp - enemy_hp) fractions")
    ap.add_argument("--hp-bonus", type=float, default=0.5,
                    help="reward = won + hp_bonus * hp_end/max_hp")
    ap.add_argument("--check-finite", action="store_true",
                    help="raise on any non-finite tensor instead of letting the GPU abort")
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="penalty against the BC prior; 0 disables")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    use_stable_attention()

    dev = torch.device(args.device)
    sizes = json.load(open(args.vocab))["sizes"]
    enc = Encoder(sizes["cards"])

    ck = torch.load(args.init, map_location=dev, weights_only=False)
    net = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev)
    load_compat(net, ck["model"])
    print(f"warm start from {args.init}: {param_count(net) / 1e6:.2f}M params on {dev}", flush=True)

    ref = None
    if args.kl_coef > 0:
        ref = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev)
        ref.load_state_dict(ck["model"])
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    pot_seen = defaultdict(int)      # potion_idx -> times drunk, for the novelty weighting

    for it in range(args.iters):
        t0 = time.time()
        roll = Rollout()
        last_enemy, last_maxhp = {}, {}

        def _last_enemy(ei):
            return last_enemy.get(ei, 0.0)

        net.eval()

        def potential(obs):
            """phi(s) = player HP fraction - mean living-enemy HP fraction."""
            p = obs["player"]
            php = p["hp"] / max(1, p["max_hp"])
            alive = [e for e in obs["enemies"] if e["alive"]]
            ehp = (sum(e["hp"] / max(1, e["max_hp"]) for e in alive) / len(alive)) if alive else 0.0
            return php - ehp

        pot_stat = defaultdict(int)
        # Annealed exploration coefficient. Linear to zero so the LAST iterations optimise the
        # true reward with no bonus at all -- whatever the policy ends up doing with potions,
        # it is because the outcome justified it, not because we paid it to drink.
        pot_coef = (args.potion_bonus
                    * max(0.0, 1.0 - it / max(1, args.potion_bonus_anneal))
                    if args.potion_bonus > 0 else 0.0)

        def policy(obs_list, legal_list, env_idxs):
            steps = [step_from_json(o, l) for o, l in zip(obs_list, legal_list)]
            b = enc.encode(steps)
            tb = {k: torch.from_numpy(b[k]).to(dev) for k in BATCH_KEYS}
            with torch.no_grad():
                out = net(tb)
                logits = out["logits"]
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
            acts = a.tolist()
            for i, (ei, legal) in enumerate(zip(env_idxs, legal_list)):
                # argmax/sample runs over the padded axis; clamp into the real legal range.
                acts[i] = min(acts[i], len(legal) - 1)
                # Potion telemetry: the whole point of a potion curriculum is that the policy
                # learns WHEN to drink, so "how often was one available" and "how often was it
                # taken" are the metrics that say whether anything is being learned.
                if any(x.get("kind") == "use_potion" for x in legal):
                    pot_stat["avail"] += 1
                    if legal[acts[i]].get("kind") == "use_potion":
                        pot_stat["used"] += 1
                bonus = 0.0
                chosen = legal[acts[i]] if acts[i] < len(legal) else {}
                if pot_coef > 0.0 and chosen.get("kind") == "use_potion":
                    pid = int(chosen.get("potion_idx", 0) or 0)
                    # 1/sqrt(1+n) novelty: the first few uses of a potion type are worth a
                    # lot, the hundredth almost nothing. Counts persist across the run so
                    # coverage is measured over all training, not within one iteration.
                    bonus = pot_coef / math.sqrt(1.0 + pot_seen[pid])
                    pot_seen[pid] += 1
                    pot_stat["bonus_n"] += 1
                roll.add(ei, {k: b[k][i] for k in BATCH_KEYS},
                         acts[i], float(logp[i]), float(out["value"][i]),
                         potential(obs_list[i]), bonus)
                alive = [e for e in obs_list[i]["enemies"] if e["alive"]]
                last_enemy[ei] = (sum(e["hp"] / max(1, e["max_hp"]) for e in alive)
                                  / len(alive)) if alive else 0.0
                last_maxhp[ei] = obs_list[i]["player"]["max_hp"]
            return acts

        def on_terminal(env_idx, res):
            # Use the character's ACTUAL max HP; it was hardcoded to 80, which mis-scales the
            # bonus for every character that is not Ironclad.
            hp_frac = res.hp_end / max(1, last_maxhp.get(env_idx, 80))
            # phi at the terminal state: a win means enemies at 0, a loss means the player at 0
            # with whatever the enemy had left when we last saw it.
            phi_end = hp_frac if res.won else -_last_enemy(env_idx)
            roll.finish(env_idx, float(res.won) + args.hp_bonus * hp_frac, res,
                        phi_end, args.shaping)

        per = max(1, args.episodes_per_iter // args.envs)
        with VecSpireEnv(n_envs=args.envs, character=args.character,
                         episodes_per_env=per, seed=f"RL{it}_",
                         encounters=args.encounters, mix=args.mix,
                         reroll_deck=args.reroll_deck,
                         characters=args.characters.split(",") if args.characters else None,
                         inject_relics=args.relics.split(",") if args.relics else None,
                         potions=args.potions, act=args.act) as venv:
            venv.run(policy, on_terminal=on_terminal)
        dropped = roll.drop_incomplete()

        if not roll.done:
            print(f"iter {it}: no completed episodes, skipping")
            continue

        # --- assemble training tensors ---
        data = {k: torch.from_numpy(np.stack([s["enc"][k] for s in roll.done])).to(dev)
                for k in BATCH_KEYS}
        acts = torch.tensor([s["action"] for s in roll.done], device=dev)
        old_logp = torch.tensor([s["logp"] for s in roll.done], device=dev)
        rets = torch.tensor([s["ret"] for s in roll.done], device=dev, dtype=torch.float32)
        n = len(roll.done)

        print(f"  [iter {it}] collected n={len(roll.done)} steps; starting update", flush=True)
        wins = np.mean([e.won for e in roll.episodes])
        hp = np.mean([e.hp_end for e in roll.episodes])
        collect_s = time.time() - t0

        # --- PPO ---
        net.train()
        t1 = time.time()
        stats = defaultdict(float)
        nb = 0
        skipped = 0
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, args.minibatch):
                idx = perm[i:i + args.minibatch]
                mb = {k: data[k][idx] for k in BATCH_KEYS}
                out = net(mb)
                dist = torch.distributions.Categorical(logits=out["logits"])
                logp = dist.log_prob(acts[idx])
                value = out["value"]

                adv = rets[idx] - value.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ratio = (logp - old_logp[idx]).exp()
                pg = -torch.min(ratio * adv,
                                ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
                vf = F.mse_loss(value, rets[idx])
                ent = dist.entropy().mean()
                loss = pg + args.vf_coef * vf - args.ent_coef * ent

                if args.check_finite:
                    # Raise a Python error the moment anything goes non-finite, rather than
                    # letting a NaN propagate into a kernel and surface as an opaque driver
                    # abort with no traceback.
                    for nm, t in (("logits", out["logits"]), ("value", value),
                                  ("logp", logp), ("ratio", ratio), ("adv", adv),
                                  ("loss", loss)):
                        if not torch.isfinite(t).all():
                            raise FloatingPointError(
                                f"non-finite {nm} at iter {it} minibatch {nb}: "
                                f"nan={int(torch.isnan(t).sum())} inf={int(torch.isinf(t).sum())}")

                if ref is not None:
                    with torch.no_grad():
                        rlogits = ref(mb)["logits"]
                    kl = F.kl_div(F.log_softmax(out["logits"], -1),
                                  F.log_softmax(rlogits, -1),
                                  log_target=True, reduction="batchmean")
                    loss = loss + args.kl_coef * kl
                    stats["kl"] += kl.item()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                if not torch.isfinite(gn):
                    # Never step on a non-finite gradient: one such step poisons every weight
                    # and every subsequent forward pass returns NaN.
                    opt.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                opt.step()
                stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["ent"] += ent.item()
                nb += 1

        print(f"  [iter {it}] update done ({nb} minibatches)", flush=True)
        for k in stats:
            stats[k] /= max(1, nb)
        print(f"iter {it + 1:3d}/{args.iters}  eps={len(roll.episodes):4d} steps={n:6d} "
              f"win={wins:.3f} hp={hp:5.1f} | pg={stats['pg']:+.4f} vf={stats['vf']:.4f} "
              f"ent={stats['ent']:.3f}"
              + (f" kl={stats['kl']:.4f}" if ref is not None else "")
              + (f" vram={torch.cuda.max_memory_allocated() / 2**30:.2f}G"
                 if dev.type == "cuda" else "")
              + f" | collect {collect_s:.0f}s train {time.time() - t1:.0f}s"
              + (f" dropped={dropped}" if dropped else "")
              + (f" SKIPPED={skipped}" if skipped else ""), flush=True)
        if pot_stat["avail"]:
            print(f"    potions: available on {pot_stat['avail']} decisions, "
                  f"used {pot_stat['used']} "
                  f"({100 * pot_stat['used'] / pot_stat['avail']:.1f}%)"
                  + (f" | bonus coef={pot_coef:.4f} distinct types tried={len(pot_seen)}"
                     if args.potion_bonus > 0 else ""), flush=True)
            pot_stat.clear()

        torch.save({"model": net.state_dict(), "sizes": ck["sizes"],
                    "d_model": ck["d_model"], "layers": ck["layers"]}, args.out)

    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
