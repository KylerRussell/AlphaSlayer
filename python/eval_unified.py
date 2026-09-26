"""Play-test: the r5 teachers or a unified checkpoint, on the SAME seeds, reported per act/boss/character.

Paired seeds: every policy plays the identical set of runs (same characters, same game seeds), so
a difference is the policy, not the draw of maps and encounters. Both policies use the same
asymmetry: greedy in combat, sampled out of combat.

    # teachers
    PYTHONPATH=. .venv7/bin/python eval_unified.py --teacher --runs 500
    # a unified checkpoint, one pass / adaptive loops
    PYTHONPATH=. .venv7/bin/python eval_unified.py --ckpt unified_m3.pt --runs 500 [--adaptive]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

import train_run as T
from alphaslayer.model import use_stable_attention
from alphaslayer.runenv import VecRunEnv
from alphaslayer.unified import features as FT
from alphaslayer.unified.net import UnifiedNet, choose_loops, to_torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_run_obs import load_models  # noqa: E402

CHARS = ["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"]
HERE = os.path.dirname(os.path.abspath(__file__))


def teacher_policy(dev, combat_ckpt, run_ckpt):
    combat, runnet = load_models(combat_ckpt, run_ckpt, dev)

    def policy(kinds, obs, legal, idxs):
        out = [0] * len(kinds)
        groups = collections.defaultdict(list)
        for i, k in enumerate(kinds):
            groups["combat" if k == "combat" else "run"].append(i)
        with torch.no_grad():
            for grp, members in groups.items():
                o = [obs[i] for i in members]
                l = [legal[i] for i in members]
                if grp == "combat":
                    lg, _, m = T.combat_forward(combat, o, l, dev)
                    idx = lg.masked_fill(~m, torch.finfo(lg.dtype).min / 4).argmax(-1)
                else:
                    lg, _, m = T.run_forward(runnet, [kinds[i] for i in members], o, l, dev)
                    idx, _ = T.sample(lg, m)
                for j, i in enumerate(members):
                    out[i] = int(idx[j])
        return out
    return policy


def unified_policy(dev, ckpt, loop, adaptive, bf16):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    net = UnifiedNet(ck["sizes"], **{k: v for k, v in ck["cfg"].items()}).to(dev).eval()
    net.load_state_dict(ck["model"])
    vocab = FT.Vocab(json.load(open(os.path.join(HERE, "vocab_m1.json"))))
    loops_used = collections.Counter()

    def policy(kinds, obs, legal, idxs):
        b = to_torch(FT.encode_batch(list(zip(kinds, obs, legal)), vocab), dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            if adaptive:
                traj = net.forward_trajectory(b, 4)
                risk = 1 - torch.sigmoid(traj[0]["aux"]["fight_win"].float())
                cap = torch.where(risk > 0.2, 4, 2)
                ks = choose_loops(traj, b["cand_mask"], cap)
                logits = torch.stack([traj[int(k) - 1]["logits"][j] for j, k in enumerate(ks)])
                loops_used.update(int(k) for k in ks)
            else:
                logits = net(b, loop=loop)["logits"]
                loops_used[loop] += len(kinds)
        logits = logits.float()
        out = []
        for j, k in enumerate(kinds):
            n = len(legal[j])
            lg = logits[j, :n]
            if k == "combat":
                out.append(int(lg.argmax()))
            else:
                out.append(int(torch.multinomial(F.softmax(lg, -1), 1)))
        return out
    policy.loops_used = loops_used
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--combat-ckpt", default="combat_r5.pt")
    ap.add_argument("--run-ckpt", default="run_r5.pt")
    ap.add_argument("--ckpt", default=None, help="unified checkpoint")
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--envs", type=int, default=25)
    ap.add_argument("--seed", default="EVAL_")
    ap.add_argument("--out", default=None, help="write the report as JSON here")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    use_stable_attention()
    torch.manual_seed(0)
    dev = torch.device(a.device)
    policy = (teacher_policy(dev, a.combat_ckpt, a.run_ckpt) if a.teacher
              else unified_policy(dev, a.ckpt, a.loop, a.adaptive, a.bf16))
    per_env = math.ceil(a.runs / a.envs)
    boss = collections.defaultdict(lambda: [0, 0])
    rooms = collections.defaultdict(lambda: [0, 0])
    per_char = collections.defaultdict(lambda: [0, 0, 0])   # char -> [act1 cleared, wins, runs]

    def on_fight(env, fr):
        rooms[(fr.act, fr.room)][0] += int(fr.won)
        rooms[(fr.act, fr.room)][1] += 1
        if fr.room == "Boss":
            boss[fr.encounter][0] += int(fr.won)
            boss[fr.encounter][1] += 1

    def on_terminal(env, rr):
        c = per_char[CHARS[env % len(CHARS)]]      # VecRunEnv assigns roster[i % len(roster)]
        c[0] += int(rr.act >= 1 or rr.won)
        c[1] += int(rr.won)
        c[2] += 1

    t0 = time.time()
    with VecRunEnv(n_envs=a.envs, characters=CHARS, runs_per_env=per_env, seed=a.seed,
                   ascension=0, act_cap=3, run_offset=0) as venv:
        venv.run(policy, on_fight=on_fight, on_terminal=on_terminal, deadline_s=7200)
        results = list(venv.results)

    n = len(results)
    rep = {"runs": n, "wall_s": time.time() - t0,
           "a1": sum(1 for r in results if r.act >= 1 or r.won) / max(1, n),
           "a2": sum(1 for r in results if r.act >= 2 or r.won) / max(1, n),
           "win": sum(r.won for r in results) / max(1, n),
           "floors": sum(r.floors for r in results) / max(1, n)}
    se = lambda p: math.sqrt(max(p * (1 - p), 1e-9) / max(1, n))
    print(f"\n{'TEACHERS (r5)' if a.teacher else a.ckpt}"
          + ("" if a.teacher else f"  loops={'adaptive' if a.adaptive else a.loop}"))
    print(f"  runs {n}: act1 {rep['a1']:.3f}+-{se(rep['a1']):.3f}  act2 {rep['a2']:.3f}+-{se(rep['a2']):.3f}"
          f"  win {rep['win']:.3f}+-{se(rep['win']):.3f}  floors {rep['floors']:.1f}  ({rep['wall_s']:.0f}s)")
    print("  fights by act/room: " + "  ".join(
        f"a{act + 1}-{room}={w / max(1, t):.3f}({t})" for (act, room), (w, t) in sorted(rooms.items())))
    print("  bosses: " + "  ".join(f"{e}={w / max(1, t):.2f}({t})" for e, (w, t) in sorted(boss.items())))
    print("  characters (act1 / win): " + "  ".join(
        f"{c}={v[0] / max(1, v[2]):.2f}/{v[1] / max(1, v[2]):.2f}({v[2]})" for c, v in sorted(per_char.items())))
    rep["characters"] = dict(per_char)
    if not a.teacher:
        print(f"  loop counts used: {dict(sorted(policy.loops_used.items()))}")
    rep["rooms"] = {f"{k[0]}:{k[1]}": v for k, v in rooms.items()}
    rep["bosses"] = dict(boss)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
