"""M4: PPO for the unified model over whole runs, win-only (see alphaslayer/unified/ppo.py).

Starts from a distilled checkpoint (M3). Per iteration:
  1. play runs with the current model (sampled, temperature 1), recording each decision's
     log-prob, value and loop count;
  2. close each run into the buffer: win-only returns, GAE with shaping in the advantage only;
  3. PPO epochs over the iteration's decisions, replaying the RECORDED loop count of every
     decision, with a KL term toward the distilled anchor that anneals to zero;
  4. log what localises problems: per-act clears, per-decision-kind entropy and top-action share
     (collapse), win-probability calibration on this iteration's decisions, bosses and deaths.

Checkpoints every iteration (model + optimiser + iteration), and --resume continues from one.
A GPU reset therefore costs at most one iteration.

    XDG_DATA_HOME="$(../headless_home.sh)" PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 \\
        HSA_ENABLE_SDMA=0 PYTHONPATH=. .venv7/bin/python train_unified.py \\
        --init unified_m3_d512_121.pt --out unified_m4.pt --iters 300 --bf16
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from alphaslayer.model import use_stable_attention
from alphaslayer.runenv import VecRunEnv
from alphaslayer.unified import dataset as DS
from alphaslayer.unified import features as FT
from alphaslayer.unified import curriculum as CU
from alphaslayer.unified import ppo as P
from alphaslayer.unified.net import UnifiedNet, choose_loops, param_count, to_torch

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS = ["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"]


def load_net(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = UnifiedNet(ck["sizes"], **ck["cfg"]).to(dev)
    net.load_state_dict(ck["model"])
    return net, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="unified_m3_d512_121.pt", help="distilled starting point")
    ap.add_argument("--anchor", default=None, help="KL anchor checkpoint (default: --init)")
    ap.add_argument("--out", default="unified_m4.pt")
    ap.add_argument("--resume", action="store_true", help="continue from --out if it exists")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--runs-per-env", type=int, default=2)
    ap.add_argument("--act-cap", type=int, default=3)
    ap.add_argument("--collect-timeout", type=float, default=900.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--aux-coef", type=float, default=0.25)
    ap.add_argument("--kl-coef", type=float, default=0.1, help="KL toward the anchor at iter 1")
    ap.add_argument("--kl-anneal", type=int, default=100, help="iterations for the KL to reach 0")
    ap.add_argument("--gamma", type=float, default=1.0, help="discount PER FLOOR")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--w-combat", type=float, default=0.1, help="combat potential shaping weight")
    ap.add_argument("--w-hp", type=float, default=0.1, help="run hp potential shaping weight")
    ap.add_argument("--run-weight", type=float, default=0.5)
    ap.add_argument("--loop-policy", default="1", help="'1'..'4' fixed, or 'adaptive'")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed-prefix", default="U4_")
    # ---- combat curriculum (alphaslayer/unified/curriculum.py) ----
    ap.add_argument("--curriculum-every", type=int, default=0,
                    help="every N iterations, add isolated act-2/3 fights on harvested decks, "
                         "targeted at the encounters the model loses most (0 = off)")
    ap.add_argument("--curriculum-fights", type=int, default=120)
    ap.add_argument("--curriculum-acts", default="1,2", help="0-based acts to drill")
    ap.add_argument("--curriculum-mix", default="0.6,0.25,0.15", help="boss,elite,regular")
    ap.add_argument("--curriculum-envs", type=int, default=12, help="deck-server envs per act")
    ap.add_argument("--curriculum-window", type=int, default=6000,
                    help="drill on the most recent N harvested entry decks")
    ap.add_argument("--harvest", default=None,
                    help="JSONL of harvested entry decks (default: <out>.harvest.jsonl)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    use_stable_attention()
    dev = torch.device(a.device)
    vocab = FT.Vocab(json.load(open(os.path.join(HERE, "vocab_m1.json"))))

    net, ck = load_net(a.init, dev)
    anchor, _ = load_net(a.anchor or a.init, dev)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    start = 1
    if a.resume and os.path.exists(a.out):
        st = torch.load(a.out, map_location=dev, weights_only=False)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["iter"] + 1
        print(f"resumed from {a.out} at iteration {start}", flush=True)
    print(f"unified net {param_count(net) / 1e6:.1f}M cfg {net.cfg}; init {a.init}; "
          f"loop policy {a.loop_policy}", flush=True)

    harvest_path = a.harvest or a.out + ".harvest.jsonl"
    harvest = []
    if os.path.exists(harvest_path):
        with open(harvest_path) as fh:
            harvest = [json.loads(l) for l in fh if l.strip()][-a.curriculum_window:]
    stats = CU.EncounterStats()
    last_player = {}          # env -> player dict of its last run-level decision
    cur_acts = tuple(int(x) for x in a.curriculum_acts.split(",") if x.strip())
    cur_mix = tuple(float(x) for x in a.curriculum_mix.split(","))

    def act_policy(kinds, obs, legal, idxs, buf):
        b = to_torch(FT.encode_batch(list(zip(kinds, obs, legal)), vocab), dev)
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
            if a.loop_policy == "adaptive":
                traj = net.forward_trajectory(b, 4)
                risk = 1 - torch.sigmoid(traj[0]["aux"]["fight_win"].float())
                is_c = torch.tensor([k == "combat" for k in kinds], device=dev)
                cap = torch.where(is_c & (risk > 0.2), 4, 2)
                ks = choose_loops(traj, b["cand_mask"], cap)
                logits = torch.stack([traj[int(k) - 1]["logits"][j] for j, k in enumerate(ks)])
                values = torch.stack([traj[int(k) - 1]["value"][j] for j, k in enumerate(ks)])
            else:
                k = int(a.loop_policy)
                out = net(b, loop=k)
                logits, values = out["logits"], out["value"]
                ks = torch.full((len(kinds),), k)
        logits = logits.float()
        acts = []
        for j, kind in enumerate(kinds):
            if kind != "combat":
                last_player[idxs[j]] = obs[j].get("player", obs[j])
            n = len(legal[j])
            lp = F.log_softmax(logits[j, :n], -1)
            act = int(torch.multinomial(lp.exp(), 1))
            buf.add(idxs[j], kind, obs[j], legal[j], act, float(lp[act]), float(values[j]),
                    int(ks[j]))
            acts.append(act)
        return acts

    for it in range(start, a.iters + 1):
        t0 = time.time()
        buf = P.RunBuffer(gamma=a.gamma, lam=a.lam, w_combat=a.w_combat, w_hp=a.w_hp)
        fights = []
        boss = collections.defaultdict(lambda: [0, 0])
        deaths = collections.Counter()
        last_fight = {}

        new_harvest = []

        def on_fight(env, fr):
            buf.end_fight(env, fr.won, fr.hp_start, fr.hp_end, fr.max_hp, fr.room)
            stats.update(fr.encounter, fr.won)
            pl = last_player.get(env)
            if pl and pl.get("deck_cards") and fr.act >= 0:
                new_harvest.append({
                    "character": pl.get("character"), "room": fr.room,
                    "encounter": fr.encounter, "act": fr.act,
                    "cards": [c["card"] for c in pl["deck_cards"]],
                    "upgrades": [int(c.get("up", 0) or 0) for c in pl["deck_cards"]],
                    "relics": [r["relic"] for r in (pl.get("relics") or [])],
                    "hp": fr.hp_start, "max_hp": fr.max_hp, "won": bool(fr.won),
                    "player": pl})
            fights.append(fr)
            last_fight[env] = fr
            if fr.room == "Boss":
                boss[fr.encounter][0] += int(fr.won)
                boss[fr.encounter][1] += 1

        def on_terminal(env, rr):
            buf.end_run(env, rr.won, rr.act, rr.floors)
            if not rr.won:
                fr = last_fight.get(env)
                deaths[f"a{rr.act + 1}-{fr.room if fr is not None and not fr.won else 'other'}"] += 1
            last_fight.pop(env, None)

        with VecRunEnv(n_envs=a.envs, characters=CHARS, runs_per_env=a.runs_per_env,
                       seed=f"{a.seed_prefix}{it}_", ascension=0, act_cap=a.act_cap,
                       run_offset=it * 1000) as venv:
            venv.run(lambda k, o, l, i: act_policy(k, o, l, i, buf), on_fight=on_fight,
                     on_terminal=on_terminal, deadline_s=a.collect_timeout)
            results = list(venv.results)
        buf.drop_open()
        steps = buf.done
        collect_s = time.time() - t0
        if new_harvest:
            with open(harvest_path, "a") as fh:
                for r in new_harvest:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            harvest = (harvest + new_harvest)[-a.curriculum_window:]

        # ---- combat curriculum: isolated act-2/3 fights, loss-weighted, same objective --------
        cur_line = ""
        if a.curriculum_every and it % a.curriculum_every == 0 and harvest:
            tc = time.time()
            from alphaslayer.deckeval import VecDeckEval
            picks = CU.choose_fights(harvest, stats, a.curriculum_fights, acts=cur_acts,
                                     mix=cur_mix, rng=random.Random(10_000 + it),
                                     pools=vocab.act_pools)
            pools = {}
            try:
                for act in sorted({r["act"] for r, _, _ in picks}):
                    # At least one env per character: a pool's envs are assigned round-robin
                    # by character, and a character with no env cannot play its fights.
                    pools[act] = VecDeckEval(n_envs=max(a.curriculum_envs, len(CHARS)),
                                             characters=CHARS,
                                             seed=f"UCUR{act}_{it}_",
                                             out_dir=f"/tmp/alphaslayer_ucur{act}_",
                                             act=act, stall_timeout=90.0)
                c_steps, c_res = CU.play_fights(net, pools, picks, vocab, vocab.enc_ix, dev,
                                                bf16=a.bf16)
            finally:
                for pl_ in pools.values():
                    try:
                        pl_.close()
                    except Exception:
                        pass
            for enc, won, *_ in c_res:
                stats.update(enc, won)
            steps = steps + c_steps
            per = collections.defaultdict(lambda: [0, 0])
            for enc, won, *_ in c_res:
                per[enc][0] += int(won); per[enc][1] += 1
            cur_line = (f"    curriculum | {len(c_res)} fights, {len(c_steps)} steps, "
                        f"{time.time() - tc:.0f}s | " + " ".join(
                            f"{e.split('_BOSS')[0].split('_ELITE')[0]}={w}/{t}"
                            for e, (w, t) in sorted(per.items(), key=lambda kv: -kv[1][1])[:8]))
        if not steps:
            print(f"iter {it}: no completed runs; skipping", flush=True)
            continue

        # ---- PPO update --------------------------------------------------------------------
        t1 = time.time()
        packed = DS.Packed(DS.pack([{"kind": s.kind, "obs": s.obs, "legal": s.legal} for s in steps], vocab))
        adv = P.normalise_advantages(steps)
        arr = lambda f, dt=np.float32: np.array([f(s) for s in steps], dt)
        T_ = {"a": arr(lambda s: s.a, np.int64), "old_logp": arr(lambda s: s.logp),
              "adv": adv, "ret": arr(lambda s: s.ret),
              "kind": arr(lambda s: FT.DKINDS.index(s.kind), np.int64),
              "act_clear": arr(lambda s: s.act_clear), "reach_act3": arr(lambda s: s.reach_act3),
              "floors_left": arr(lambda s: s.floors_left), "fight_won": arr(lambda s: s.fight_won),
              "fight_hp": arr(lambda s: s.fight_hp), "loop": arr(lambda s: s.loop, np.int64)}
        n_tok = packed.a["n_tokens"]
        kl_coef = a.kl_coef * max(0.0, 1.0 - (it - 1) / max(1, a.kl_anneal))
        net.train()
        agg = collections.defaultdict(float)
        nb = 0
        rng = random.Random(it)
        for ep in range(a.epochs):
            order = list(range(len(steps)))
            rng.shuffle(order)
            batches = []
            for s0 in range(0, len(order), a.minibatch * 16):
                part = sorted(order[s0:s0 + a.minibatch * 16], key=lambda i: n_tok[i])
                batches += [part[j:j + a.minibatch] for j in range(0, len(part), a.minibatch)]
            rng.shuffle(batches)
            for idx in batches:
                inp, _ = DS.make_batch([(packed, i) for i in idx])
                inp = to_torch(inp, dev)
                tgt = {k: torch.as_tensor(v[idx]).to(dev) for k, v in T_.items()}
                tgt["cand_mask"] = inp["cand_mask"]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                    out = net(inp, loop_rows=tgt["loop"])
                    anchor_logp = None
                    if kl_coef > 0:
                        with torch.no_grad():
                            anchor_logp = F.log_softmax(
                                anchor(inp, loop_rows=tgt["loop"])["logits"].float(), -1)
                total, parts = P.ppo_loss(out, tgt, clip=a.clip, ent_coef=a.ent_coef,
                                          vf_coef=a.vf_coef, aux_coef=a.aux_coef,
                                          run_weight=a.run_weight, anchor_logp=anchor_logp,
                                          kl_coef=kl_coef)
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()):
                    opt.step()
                for k, v in parts.items():
                    agg[k] += float(v.detach())
                nb += 1
        train_s = time.time() - t1

        # ---- report --------------------------------------------------------------------------
        n = len(results)
        won = sum(r.won for r in results)
        a1 = sum(1 for r in results if r.act >= 1 or r.won) / max(1, n)
        a2 = sum(1 for r in results if r.act >= 2 or r.won) / max(1, n)
        fw = sum(f.won for f in fights) / max(1, len(fights))
        v = np.array([s.v for s in steps]); y = np.array([s.won for s in steps], float)
        brier = float(((v - y) ** 2).mean())
        base = float(((y.mean() - y) ** 2).mean())
        kstat = collections.defaultdict(lambda: [0, collections.Counter()])
        for s in steps:
            if s.kind == "combat":
                continue
            ks = kstat[s.kind]
            ks[0] += 1
            lab = (s.legal[s.a].get("option") or s.legal[s.a].get("kind") or "opt") \
                if s.kind in ("rest", "potion_gate", "card_reward", "shop") else "a"
            ks[1][lab] += 1
        collapse = " ".join(
            f"{k}:{v[1].most_common(1)[0][0]}={100 * v[1].most_common(1)[0][1] / v[0]:.0f}%"
            for k, v in sorted(kstat.items()) if k in ("rest", "potion_gate", "card_reward", "shop"))
        loops = np.mean([s.loop for s in steps])
        print(f"iter {it:4d}/{a.iters} runs={n} win={won / max(1, n):.3f} a1={a1:.3f} a2={a2:.3f} "
              f"a2|a1={a2 / max(a1, 1e-9):.3f} floors={np.mean([r.floors for r in results]):.1f} "
              f"fight_win={fw:.3f} | pg={agg['pg'] / nb:+.4f} ent={agg['ent'] / nb:.3f} "
              f"vf={agg['value'] / nb:.4f} kl_anchor={agg['kl'] / nb:.4f}(x{kl_coef:.3f}) "
              f"clip={agg['clipfrac'] / nb:.3f} akl={agg['approx_kl'] / nb:+.4f} | "
              f"brier={brier:.4f}/base {base:.4f} loops={loops:.2f} steps={len(steps)} "
              f"collect {collect_s:.0f}s train {train_s:.0f}s", flush=True)
        if cur_line:
            print(cur_line, flush=True)
        print(f"    top actions | {collapse} | deaths {dict(deaths.most_common(4))} | bosses "
              + " ".join(f"{e.split('_BOSS')[0]}={w}/{t}" for e, (w, t) in sorted(boss.items())),
              flush=True)
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "iter": it,
                    "cfg": net.cfg, "sizes": ck["sizes"], "args": vars(a)}, a.out + ".tmp")
        os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
