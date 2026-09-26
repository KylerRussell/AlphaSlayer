"""Localises WHERE act 2 kills runs, which the training log cannot show.

The iteration line reports ONE aggregate fight_win over every fight in the batch. Act-1
regular fights are ~90% of those, so an act-2 collapse is invisible in it. This plays runs
with a fixed (combat, run) checkpoint pair and reports, per act:

  * fight win rate split by room type, so "act-2 fights are harder" can be separated from
    "act-2 attrition kills a healthy policy"
  * the state the policy ENTERS each act with (hp fraction, relics, deck size, upgrades)
  * where in the act runs die (act_floor histogram)

Greedy combat, sampled run policy -- the same asymmetry the trainer's rollout uses.

    python diag_act2.py --combat-ckpt combat_comb.pt --run-ckpt run_comb.pt --runs 300
"""
from __future__ import annotations

import argparse
import collections

import torch

import train_run as T
from alphaslayer.model import CombatNet, use_stable_attention
from alphaslayer.modcheck import check as modcheck
from alphaslayer.runenv import VecRunEnv
from alphaslayer.runmodel import RunNet, migrate_state_scalars
from train_rl import load_compat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat-ckpt", required=True)
    ap.add_argument("--run-ckpt", required=True)
    ap.add_argument("--envs", type=int, default=20)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--characters", default="IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--act-cap", type=int, default=3)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    use_stable_attention()
    modcheck()
    dev = torch.device(a.device)
    ck = torch.load(a.combat_ckpt, map_location=dev, weights_only=False)
    combat = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev)
    load_compat(combat, ck["model"])
    combat.eval()
    T._ENC = T.Encoder(ck["sizes"]["cards"])
    rck = torch.load(a.run_ckpt, map_location=dev, weights_only=False)
    runnet = RunNet(ck["sizes"]["cards"], ck["sizes"]["relics"], ck["sizes"].get("potions", 66),
                    per_kind_heads=True).to(dev)
    load_compat(runnet, migrate_state_scalars(rck["model"]))
    runnet.eval()

    chars = a.characters.split(",")
    # Per-env act, tracked from the run-level observations: FightResult carries the room type
    # but not which act it happened in, and that split is the whole point of this script.
    cur_act = {}
    entered = {}            # (env, act) -> state on entry, recorded once
    fights = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    entry_stats = collections.defaultdict(list)
    deaths = collections.defaultdict(collections.Counter)
    death_room = collections.defaultdict(collections.Counter)
    last_fight = {}
    outcomes = collections.Counter()

    def policy(kinds, obs, legal, idxs):
        acts = []
        groups = {"combat": [], "run": []}
        for i, k in enumerate(kinds):
            groups["combat" if k == "combat" else "run"].append(i)
            if k != "combat":
                o = obs[i]
                act = int(o.get("act", 0) or 0)
                cur_act[idxs[i]] = act
                p = o.get("player", o)
                key = (idxs[i], act)
                if key not in entered:
                    entered[key] = True
                    mx = max(1, int(p.get("max_hp", 1) or 1))
                    entry_stats[act].append((
                        (p.get("hp", 0) or 0) / mx,
                        len(p.get("relics") or []),
                        int(p.get("deck_size", 0) or 0),
                        int(p.get("deck_upgrades", 0) or 0),
                    ))
        out = [0] * len(kinds)
        with torch.no_grad():
            for grp, members in groups.items():
                if not members:
                    continue
                o = [obs[i] for i in members]
                l = [legal[i] for i in members]
                if grp == "combat":
                    lg, _, m = T.combat_forward(combat, o, l, dev)
                    lg = lg.masked_fill(~m, torch.finfo(lg.dtype).min / 4)
                    idx = lg.argmax(-1)          # greedy in combat
                else:
                    lg, _, m = T.run_forward(runnet, [kinds[i] for i in members], o, l, dev)
                    idx, _ = T.sample(lg, m)     # sampled out of combat
                for j, i in enumerate(members):
                    out[i] = int(idx[j])
        return out

    def on_fight(env, fr):
        act = cur_act.get(env, 0)
        cell = fights[act][fr.room]
        cell[1] += 1
        cell[0] += int(fr.won)
        last_fight[env] = (act, fr)

    def on_terminal(env, rr):
        outcomes[rr.outcome] += 1
        if not rr.won:
            act = cur_act.get(env, 0)
            deaths[act][rr.floors] += 1
            lf = last_fight.get(env)
            death_room[act][lf[1].room if lf else "not-in-combat"] += 1
        last_fight.pop(env, None)
        for k in list(entered):
            if k[0] == env:
                del entered[k]
        cur_act.pop(env, None)

    per_env = max(1, a.runs // a.envs)
    with VecRunEnv(n_envs=a.envs, characters=chars, runs_per_env=per_env, seed="DIAG_",
                   ascension=a.ascension, act_cap=a.act_cap, run_offset=0,
                   out_dir="/tmp/alphaslayer_diag", stall_dir="/tmp/alphaslayer_diag_stalls",
                   expect_sizes={"cards": ck["sizes"]["cards"],
                                 "relics": ck["sizes"]["relics"]}) as venv:
        venv.run(policy, on_fight=on_fight, on_terminal=on_terminal, deadline_s=3600)
        results = list(venv.results)

    n = len(results)
    print(f"\n{n} runs, act-cap {a.act_cap}, {a.combat_ckpt} + {a.run_ckpt}")
    print(f"outcomes: {dict(outcomes)}")
    reached = collections.Counter()
    for r in results:
        for k in range(0, (r.act if not r.won else a.act_cap - 1) + 1):
            reached[k] += 1
    print("\nruns reaching each act (0-based act index):")
    for k in sorted(reached):
        print(f"  act {k + 1}: {reached[k]:4d} ({reached[k] / n:.3f})")

    print("\nFIGHT WIN RATE BY ACT AND ROOM  (the split the training log cannot show)")
    print(f"  {'act':>4s} {'room':>9s} {'wins':>6s} {'fights':>7s} {'rate':>7s}")
    for act in sorted(fights):
        for room in sorted(fights[act], key=lambda r: -fights[act][r][1]):
            w, t = fights[act][room]
            if t:
                print(f"  {act + 1:>4d} {room:>9s} {w:6d} {t:7d} {w / t:7.3f}")
        tw = sum(v[0] for v in fights[act].values())
        tt = sum(v[1] for v in fights[act].values())
        if tt:
            print(f"  {act + 1:>4d} {'ALL':>9s} {tw:6d} {tt:7d} {tw / tt:7.3f}")

    print("\nSTATE ON ENTERING EACH ACT (mean)")
    print(f"  {'act':>4s} {'n':>5s} {'hp_frac':>8s} {'relics':>7s} {'deck':>6s} {'upgrades':>9s}")
    for act in sorted(entry_stats):
        rows = entry_stats[act]
        if not rows:
            continue
        m = [sum(c) / len(rows) for c in zip(*rows)]
        print(f"  {act + 1:>4d} {len(rows):5d} {m[0]:8.3f} {m[1]:7.2f} {m[2]:6.1f} {m[3]:9.2f}")

    print("\nDEATHS: room that killed the run, by act")
    for act in sorted(death_room):
        tot = sum(death_room[act].values())
        detail = ", ".join(f"{k}={v}" for k, v in death_room[act].most_common())
        print(f"  act {act + 1}: {tot:4d} deaths | {detail}")


if __name__ == "__main__":
    main()
