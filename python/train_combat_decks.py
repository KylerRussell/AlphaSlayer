"""PPO for the combat net on EXACT harvested decks, against a chosen act's encounters.

Why this exists. train_rl.py trains combat through `serve`, which generates its own deck:
either the character's starting deck or a Pandora's-Box reroll. Round 2 used that to teach
act-2 encounters and it backfired -- act-2 isolated win rate rose 0.31 -> 0.455 while the
REALISTIC act-1 benchmark fell 0.765 -> 0.680 and runs reaching act 2 fell 77% -> 55%. The
reason is distribution: a Pandora's deck is 20+ random cards with two strong relics, while a
real act-2 entry is 17.4 cards, 4.4 relics, 0.84 upgrades, at 35% hp. Training act-2 skill
against a deck that never occurs teaches a policy for a game nobody plays, and the act-1
refresher blocks (also Pandora's) failed to protect real act-1 play for the same reason.

`deckserve` already plays an EXACT deck (cards, upgrades, relics) against a chosen room type,
and the harvest file from train_run.py --harvest-decks records exactly that, tagged with the
room the deck was about to fight and the hp it had. This trains on those.

    python train_combat_decks.py --init combat_comb.pt --decks harvest_act2.jsonl \\
        --act 1 --iters 100 --out combat_r3.pt --ref combat_comb.pt --kl-coef 0.05
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from types import SimpleNamespace

import torch

import train_run as T
from alphaslayer.deckeval import DeckSpec, VecDeckEval
from alphaslayer.model import CombatNet, use_stable_attention
from alphaslayer.modcheck import check as modcheck
from train_rl import load_compat

# fight_end room names in the harvest -> the room ids deckserve understands.
ROOM_MAP = {"Boss": "boss", "Elite": "elite", "Monster": "regular"}


def potential(obs):
    """phi(s) = player hp fraction - mean living-enemy hp fraction.

    Byte-identical to the shaping potential in train_run.py and train_rl.py, so a checkpoint
    trained here is optimising the same combat objective it was trained on elsewhere.
    """
    p = obs.get("player", {})
    php = (p.get("hp", 0) or 0) / max(1, p.get("max_hp", 1) or 1)
    alive = [e for e in (obs.get("enemies") or []) if e.get("alive")]
    ehp = (sum(e.get("hp", 0) / max(1, e.get("max_hp", 1) or 1) for e in alive) / len(alive)
           if alive else 0.0)
    return php - ehp


def load_decks(path, act=None, min_deck=6):
    """Harvest records -> deckserve specs, optionally filtered to one act.

    Records written before the `act` field existed have no act; they are kept only when no
    act filter is requested, so an old harvest cannot silently masquerade as act-2 data.
    """
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            room = ROOM_MAP.get(r.get("room"))
            if room is None or len(r.get("cards") or []) < min_deck:
                continue
            if act is not None and r.get("act") != act:
                continue
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decks", required=True, help="harvest jsonl from --harvest-decks")
    ap.add_argument("--act", type=int, default=None,
                    help="filter the harvest to this 0-based act AND point deckserve's "
                         "encounter pools at it (--probe-act)")
    ap.add_argument("--ref", default=None, help="KL trust-region reference checkpoint")
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="KL penalty toward --ref. Round 2's failure was unbounded drift, so "
                         "this is the leash that keeps act-1 skill while act-2 is learned")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--envs", type=int, default=20)
    ap.add_argument("--fights-per-iter", type=int, default=200)
    ap.add_argument("--mix", default="0.5,0.25,0.25", help="boss,elite,regular sampling weights")
    ap.add_argument("--real-hp", action="store_true",
                    help="start each fight at the hp fraction the deck was harvested with, "
                         "instead of full. Needs the probe's hp_frac support")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--shaping", type=float, default=0.5)
    ap.add_argument("--hp-bonus", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    use_stable_attention()
    modcheck()
    dev = torch.device(a.device)
    ck = torch.load(a.init, map_location=dev, weights_only=False)
    net = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev)
    load_compat(net, ck["model"])
    T._ENC = T.Encoder(ck["sizes"]["cards"])

    ref = None
    if a.ref and a.kl_coef > 0:
        rck = torch.load(a.ref, map_location=dev, weights_only=False)
        ref = CombatNet(rck["sizes"], d_model=rck["d_model"], layers=rck["layers"]).to(dev)
        load_compat(ref, rck["model"])
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        print(f"KL trust region anchored to {a.ref} (coef {a.kl_coef})", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    # ppo_update reads its knobs off an args object; combat ignores the per-kind ones.
    pargs = SimpleNamespace(epochs=a.epochs, minibatch=a.minibatch, clip=a.clip,
                            vf_coef=a.vf_coef, ent_coef=a.ent_coef, per_kind_loss=False,
                            per_kind_entropy=False, skip_ent_coef=0.0, combat_kl=a.kl_coef)

    recs = load_decks(a.decks, act=a.act)
    if not recs:
        raise SystemExit(f"no usable decks in {a.decks}"
                         + (f" for act {a.act}" if a.act is not None else ""))
    by_room = defaultdict(list)
    for r in recs:
        by_room[ROOM_MAP[r["room"]]].append(r)
    print(f"{len(recs)} decks"
          + (f" for act {a.act + 1}" if a.act is not None else "")
          + ": " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_room.items())), flush=True)
    weights = [float(x) for x in a.mix.split(",")]
    rooms = ["boss", "elite", "regular"]
    rng = random.Random(1234)

    pools = {}          # character -> VecDeckEval, kept alive across iterations
    # --envs is a TOTAL, split across the characters present. One deckserve pool plays one
    # character, so a per-character pool of --envs would be 5x the processes: 100 games at
    # ~850MB is more memory than the machine has. Pools are kept alive rather than reopened
    # per iteration because a pool costs ~10s to boot and an iteration only ~25s to run.
    chars_present = sorted({r["character"] for r in recs})
    per_char_envs = max(2, a.envs // max(1, len(chars_present)))
    print(f"{len(chars_present)} character(s), {per_char_envs} envs each "
          f"({per_char_envs * len(chars_present)} game processes)", flush=True)

    def pool_for(character):
        if character not in pools:
            pools[character] = VecDeckEval(
                n_envs=per_char_envs, character=character, seed=f"R3{character[:3]}",
                out_dir=f"/tmp/alphaslayer_r3_{character}_", act=a.act or 0)
        return pools[character]

    try:
        for it in range(1, a.iters + 1):
            t0 = time.time()
            # Sample this iteration's fights, grouped by character because one deckserve pool
            # plays one character.
            picks = defaultdict(list)
            for _ in range(a.fights_per_iter):
                room = rng.choices(rooms, weights=weights, k=1)[0]
                if not by_room.get(room):
                    continue
                r = rng.choice(by_room[room])
                picks[r["character"]].append((r, room))

            steps_by_env = {}
            done_steps = []
            wins = total = 0
            hp_sum = 0.0

            def policy(obs_list, legal_list, idx_list):
                with torch.no_grad():
                    logits, value, mask = T.combat_forward(net, obs_list, legal_list, dev)
                    idx, logp = T.sample(logits, mask)
                acts = []
                for j, env in enumerate(idx_list):
                    acts.append(int(idx[j]))
                    steps_by_env.setdefault(env, []).append({
                        "obs": obs_list[j], "legal": legal_list[j], "kind": "combat",
                        "a": int(idx[j]), "logp": float(logp[j]), "v": float(value[j]),
                        "cphi": potential(obs_list[j]),
                    })
                return acts

            def on_terminal(env, spec, room, msg):
                nonlocal wins, total, hp_sum
                mx = max(1, int(msg.get("max_hp", 1) or 1))
                hp_frac = (msg.get("hp_end", 0) or 0) / mx
                won = bool(msg.get("won"))
                # Same credit as train_run.Buffer.finish_fight: terminal value plus
                # potential-based shaping, so an early block that prevented damage is not
                # credited identically to the killing blow.
                phi_end = hp_frac if won else -0.0
                ret = float(won) + a.hp_bonus * hp_frac
                for s in steps_by_env.pop(env, []):
                    s["ret"] = ret + a.shaping * (phi_end - s["cphi"])
                    done_steps.append(s)
                wins += int(won)
                total += 1
                hp_sum += hp_frac

            net.eval()
            for character, items in picks.items():
                specs = []
                for r, room in items:
                    sp = DeckSpec(r["cards"], r["upgrades"], r["relics"], character,
                                  fights_per_room=1, rooms=(room,), tag=room)
                    if a.real_hp:
                        # The hp the deck actually had walking into that room.
                        sp.hp_frac = max(0.05, min(1.0, (r.get("hp", 0) or 0)
                                                   / max(1, r.get("max_hp", 1) or 1)))
                    specs.append(sp)
                pool_for(character).evaluate(specs, policy, on_terminal=on_terminal,
                                             with_idx=True)
            net.train()
            # Episodes still in flight when a pool stopped have no outcome; crediting them
            # with an invented reward would poison the value target.
            steps_by_env.clear()
            collect = time.time() - t0

            if not done_steps:
                print(f"iter {it:3d}/{a.iters}  no completed fights; skipping", flush=True)
                continue
            t1 = time.time()
            stats = T.ppo_update(net, opt, done_steps, dev, pargs, is_combat=True, ref=ref)
            print(f"iter {it:3d}/{a.iters}  fights={total:4d} win={wins / max(1, total):.3f} "
                  f"hp_end={hp_sum / max(1, total):.3f} | pg={stats['pg']:+.4f} "
                  f"vf={stats['vf']:.3f} ent={stats['ent']:.3f} kl={stats['kl']:.4f} "
                  f"| steps={len(done_steps)} collect {collect:.0f}s train {time.time() - t1:.0f}s",
                  flush=True)
            torch.save({"model": net.state_dict(), "sizes": ck["sizes"],
                        "d_model": ck["d_model"], "layers": ck["layers"]}, a.out)
    finally:
        for p in pools.values():
            try:
                p.close()
            except Exception:
                pass
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
