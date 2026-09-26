"""M3 step 1: record the r5 teachers playing full runs, as distillation data for the unified model.

Every decision is kept with:
  * the TEACHER'S FULL DISTRIBUTION over the legal actions (not just its choice): the student
    learns from the soft target, which carries far more signal per decision than one label
  * the action taken (combat greedy, run sampled: the same asymmetry the trainer's rollouts use)
  * outcome labels for the value and auxiliary heads, filled in when they become known:
      won          did this run win (the value target: V = P(win))
      act_clear    was the act this decision was made in cleared
      reach_act3   did the run reach act 3
      floors_left  floors the run survived after this decision
      fight_won    (combat only) was this fight won
      fight_hp     (combat only) HP lost in this fight, as a fraction of max HP

Runs are written to shards as they finish; a restart continues numbering after the shards on
disk. Held-out splitting happens at training time, by run.

    XDG_DATA_HOME="$(../headless_home.sh)" PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 \\
        HSA_ENABLE_SDMA=0 PYTHONPATH=. .venv7/bin/python collect_distill.py --runs 2000
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

import train_run as T
from alphaslayer.runenv import VecRunEnv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_run_obs import load_models  # noqa: E402

CHARS = ["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat-ckpt", default="combat_r5.pt")
    ap.add_argument("--run-ckpt", default="run_r5.pt")
    ap.add_argument("--out", default="data/distill")
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--runs-per-env", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    combat, runnet = load_models(a.combat_ckpt, a.run_ckpt, dev)
    chunk = len(glob.glob(os.path.join(a.out, "shard_*.jsonl.gz")))
    done_runs = 0
    for p in glob.glob(os.path.join(a.out, "shard_*.jsonl.gz")):
        with gzip.open(p, "rt") as fh:
            done_runs += len({json.loads(l)["run"] for l in fh})
    print(f"resuming: {chunk} shards, {done_runs} runs on disk", flush=True)

    while done_runs < a.runs:
        t0 = time.time()
        open_run = collections.defaultdict(list)     # env -> decisions of the current run
        open_fight = collections.defaultdict(list)   # env -> indices (into open_run) of this fight
        run_no = collections.Counter()
        finished = []                                # completed runs' decision lists

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
                        lg = lg.masked_fill(~m, torch.finfo(lg.dtype).min / 4)
                        idx = lg.argmax(-1)
                    else:
                        lg, _, m = T.run_forward(runnet, [kinds[i] for i in members], o, l, dev)
                        lg = lg.masked_fill(~m, torch.finfo(lg.dtype).min / 4)
                        idx, _ = T.sample(lg, m)
                    probs = F.softmax(lg.float(), -1).cpu()
                    for j, i in enumerate(members):
                        env = idxs[i]
                        n = len(legal[i])
                        rec = {"kind": kinds[i], "obs": obs[i], "legal": legal[i],
                               "tp": [round(float(x), 5) for x in probs[j, :n]],
                               "a": int(idx[j]), "env": env, "run_in_env": run_no[env]}
                        if kinds[i] == "combat":
                            open_fight[env].append(len(open_run[env]))
                        open_run[env].append(rec)
                        out[i] = int(idx[j])
            return out

        def on_fight(env, fr):
            mx = max(1, fr.max_hp)
            for k in open_fight.pop(env, []):
                d = open_run[env][k]
                d["fight_won"] = int(fr.won)
                d["fight_hp"] = (fr.hp_start - fr.hp_end) / mx
                d["room"] = fr.room
                d["encounter"] = fr.encounter

        def on_terminal(env, rr):
            decs = open_run.pop(env, [])
            # A run that dies mid-fight may send no fight_end: that fight was lost, and how much
            # HP it cost is unknown, so the HP label stays absent.
            for k in open_fight.pop(env, []):
                decs[k]["fight_won"] = 0
            final = rr.floors
            for d in decs:
                o = d["obs"]
                src = o if d["kind"] == "combat" else o.get("player", o)
                act = int(src.get("act", o.get("act", 0)) or 0)
                floor = int(src.get("total_floor", o.get("total_floor", 0)) or 0)
                d["won"] = int(rr.won)
                d["act_clear"] = int(rr.won or rr.act > act)
                d["reach_act3"] = int(rr.won or rr.act >= 2)
                d["floors_left"] = max(0, final - floor)
                d["character"] = (o.get("character") or src.get("character") or "")
            finished.append(decs)
            run_no[env] += 1

        remaining = a.runs - done_runs
        per_env = max(1, min(a.runs_per_env, -(-remaining // a.envs)))
        with VecRunEnv(n_envs=a.envs, characters=CHARS, runs_per_env=per_env,
                       seed=f"DISTILL{chunk}_", ascension=0, act_cap=3,
                       run_offset=chunk * 1000) as venv:
            venv.run(policy, on_fight=on_fight, on_terminal=on_terminal, deadline_s=3600)
            results = list(venv.results)

        path = os.path.join(a.out, f"shard_{chunk:04d}.jsonl.gz")
        n_dec = 0
        with gzip.open(path, "wt") as fh:
            for r_i, decs in enumerate(finished):
                run_id = f"{chunk}:{r_i}"
                for d in decs:
                    d["run"] = run_id
                    fh.write(json.dumps(d, separators=(",", ":")) + "\n")
                    n_dec += 1
        done_runs += len(finished)
        wins = sum(r.won for r in results)
        a1 = sum(1 for r in results if r.act >= 1 or r.won)
        print(f"shard {chunk}: {len(finished)} runs, {n_dec} decisions, win {wins}/{len(results)}, "
              f"act1 cleared {a1}, {time.time() - t0:.0f}s | total runs {done_runs}/{a.runs}",
              flush=True)
        chunk += 1


if __name__ == "__main__":
    main()
