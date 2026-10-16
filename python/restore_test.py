"""Restore-primitive check (probe FightRestore): does reloading a fight's room-entry save and
re-applying its choices land on the IDENTICAL decision, and how long does it take?

Plays runs with a random policy on a test copy of the game (so a training run's games keep
their own probe build). In each fight, at the first decision of up to N turns (from turn 2),
the probe restores, replays, and compares the observation + legal actions it left with the
ones it lands on; one line per test in <out>/restore.jsonl.

    PYTHONPATH=. .venv7/bin/python restore_test.py --game data/sts2_testgame --runs 20
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil

from alphaslayer.runenv import RunPool

CHARS = ["IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="data/sts2_testgame")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--tests-per-fight", type=int, default=3)
    ap.add_argument("--no-restore", action="store_true", help="flag off: the refactor alone")
    ap.add_argument("--drop-last", action="store_true",
                    help="negative control: replay one choice fewer; every test must mismatch")
    ap.add_argument("--out", default="/tmp/alphaslayer_restore_")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-arg", action="append", default=[],
                    help="extra game flag, e.g. --probe-arg=--probe-restore-stress=200 (profiling)")
    a = ap.parse_args()
    for d in glob.glob(a.out + "*"):
        shutil.rmtree(d, ignore_errors=True)
    rng = random.Random(a.seed)
    extra = [] if a.no_restore else ["--probe-fight-restore", f"--probe-restore-test={a.tests_per_fight}"]
    if a.drop_last:
        extra.append("--probe-restore-test-drop-last")
    extra += a.probe_arg
    jobs = [{"run": r, "seed": f"RT_{r}", "character": CHARS[r % 5]} for r in range(a.runs)]
    fights = []

    def policy(kinds, obs, legal, idxs):
        # random, but end the turn less often so fights last a few turns
        out = []
        for k, lg in zip(kinds, legal):
            if k == "combat" and len(lg) > 1:
                non_end = [i for i, x in enumerate(lg) if x.get("kind") != "end_turn"]
                out.append(rng.choice(non_end) if non_end and rng.random() < 0.8 else
                           next((i for i, x in enumerate(lg) if x.get("kind") == "end_turn"), 0))
            else:
                out.append(rng.randrange(len(lg)))
        return out

    with RunPool(n_envs=a.slots, game_dir=os.path.abspath(a.game), out_dir=a.out, act_cap=1,
                 extra_args=extra, recycle_runs=1000) as pool:
        pool.run(jobs, policy, on_fight=lambda i, fr: fights.append(fr), deadline_s=1800)
        res = list(pool.results)
    print(f"runs {len(res)}/{len(jobs)} lost {len(pool.lost)} stats {pool.stats} fights {len(fights)} "
          f"won {sum(f.won for f in fights)} floors {sum(r.floors for r in res) / max(1, len(res)):.1f}")
    recs = []
    for p in glob.glob(a.out + "*/restore.jsonl"):
        recs += [json.loads(l) for l in open(p) if l.strip()]
    if not recs:
        print("no restore tests recorded")
        return
    ok = [r for r in recs if r["match"]]
    print(f"really restored: new combat {sum(r['new_combat'] for r in recs)}/{len(recs)}, "
          f"new run state {sum(r['new_run'] for r in recs)}/{len(recs)}")
    print(f"restore tests {len(recs)}: match {len(ok)}, mismatch "
          f"{sum(1 for r in recs if not r['match'] and not r['error'])}, error "
          f"{sum(1 for r in recs if r['error'])}")
    lm = sorted(r["load_ms"] for r in recs)
    rm = sorted((r["replay_ms"], r["choices"]) for r in recs)
    print(f"load ms median {lm[len(lm) // 2]:.0f} p90 {lm[int(len(lm) * .9)]:.0f}; replay ms median "
          f"{rm[len(rm) // 2][0]:.0f} (choices {rm[len(rm) // 2][1]}), max {rm[-1][0]:.0f} ({rm[-1][1]})")
    per = sum(r["replay_ms"] for r in recs) / max(1, sum(r["choices"] for r in recs))
    print(f"replay cost per choice {per:.1f} ms")
    for r in [r for r in recs if not r["match"]][:5]:
        print(json.dumps(r)[:600])


if __name__ == "__main__":
    main()
