"""Paired comparison of two fixed-seed evaluations (logs/eval_runs/*.json, from --eval-every
or --eval-only). Both played the same EV_ seeds, so seed luck cancels out.

Files written before 2026-10-15 18:30 gave 24 runs the same run id, so their per-run seed
labels are wrong; each batch of 24 is still the same 24 (seed, character) jobs. With such a
file the comparison is per batch (the seed's last 4 digits = e*1000+r); otherwise per run,
with an exact McNemar test on the runs whose outcome changed.

    python3 pair_evals.py ../logs/eval_runs/A_g.json ../logs/eval_runs/B_g.json

--exclude-diverged '/tmp/alphaslayer_search_*/diverged.jsonl' leaves out the runs whose live game a
fight search could not restore (search_eval run ids -> their EV_ seeds).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random

METRICS = {"win": lambda r: r["won"], "a1": lambda r: r["act"] >= 1 or r["won"],
           "a2": lambda r: r["act"] >= 2 or r["won"]}


def load(path):
    runs = json.load(open(path))
    labelled = len({(r["seed"], r["character"]) for r in runs}) == len(runs)
    return runs, labelled


def mcnemar(up, dn):
    n, k = up + dn, min(up, dn)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def sign_flip_p(d, n=100000, seed=0):
    rng = random.Random(seed)
    obs = abs(sum(d))
    return sum(abs(sum(x if rng.random() < .5 else -x for x in d)) >= obs for _ in range(n)) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--exclude-diverged", default=None,
                    help="glob of search diverged.jsonl files: drop those runs from both sides")
    a = ap.parse_args()
    (A, la), (B, lb) = load(a.ref), load(a.new)
    if a.exclude_diverged:
        bad = set()
        for f in glob.glob(a.exclude_diverged):
            for line in open(f):
                run = json.loads(line)["run"]
                e, r, i = run // 100000, run % 100000 // 1000, run % 1000
                bad.add(f"EV_{e}_{i}{e * 1000 + r}")       # search_eval's seed for that run id
        A = [x for x in A if x["seed"] not in bad]
        B = [x for x in B if x["seed"] not in bad]
        print(f"excluded {len(bad)} seed(s) with a lost live state: {sorted(bad)}")
    print(f"ref {a.ref} ({len(A)} runs)\nnew {a.new} ({len(B)} runs)")
    if la and lb:
        ka = {(r["seed"], r["character"]): r for r in A}
        kb = {(r["seed"], r["character"]): r for r in B}
        keys = sorted(set(ka) & set(kb))
        print(f"paired per run: {len(keys)} (seed, character) pairs")
        for m, f in METRICS.items():
            d = [float(f(kb[k])) - float(f(ka[k])) for k in keys]
            n = len(d)
            md = sum(d) / n
            se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1) / n)
            up, dn = sum(x > 0 for x in d), sum(x < 0 for x in d)
            print(f"  {m:4} ref {sum(float(f(ka[k])) for k in keys) / n:.3f} "
                  f"new {sum(float(f(kb[k])) for k in keys) / n:.3f}  diff {md:+.3f} +- {se:.3f}"
                  f"  | gained {up} lost {dn}  McNemar p={mcnemar(up, dn):.3f}")
        return
    ga, gb = collections.defaultdict(list), collections.defaultdict(list)
    for runs, g in ((A, ga), (B, gb)):
        for r in runs:
            g[r["seed"][-4:]].append(r)
    keys = sorted(set(ga) & set(gb))
    assert all(len(ga[k]) == len(gb[k]) for k in keys), "batches differ in size"
    print(f"paired per batch (an old-format file): {len(keys)} batches of {len(ga[keys[0]])} runs")
    for m, f in METRICS.items():
        d = [sum(map(f, gb[k])) - sum(map(f, ga[k])) for k in keys]
        n, tot = len(d), sum(len(ga[k]) for k in keys)
        md = sum(d) / n
        se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1) / n) * n / tot
        print(f"  {m:4} ref {sum(sum(map(f, ga[k])) for k in keys) / tot:.3f} "
              f"new {sum(sum(map(f, gb[k])) for k in keys) / tot:.3f}  diff {sum(d) / tot:+.3f} +- {se:.3f}"
              f"  | sign-flip p={sign_flip_p(d):.3f}")


if __name__ == "__main__":
    main()
