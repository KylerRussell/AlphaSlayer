#!/usr/bin/env python3
"""Summarises training logs in blocks, so a plateau is visible through the sampling noise.

At 20 runs per iteration the standard error on a per-act clear rate is ~0.11, so no single
iteration line means anything; only block means do. Default block is 25 iterations (500 runs,
SE ~0.02). Also reports the LAST-vs-FIRST-half delta with a standard error, which is the
number that answers "did this candidate break the plateau".

    ./analyze_exp.py logs/ctl.log logs/hp.log [--block 25]
"""
from __future__ import annotations

import argparse
import math
import re
import sys

FIELDS = ("a1", "a2", "a3", "floors", "fight_win", "ent", "win")
ITER_RE = re.compile(r"^iter\s+\d+/")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z_0-9|]*)=\s*([-+]?[0-9]*\.?[0-9]+)")


def parse(path):
    rows = []
    try:
        fh = open(path)
    except OSError:
        return rows
    with fh:
        for line in fh:
            if not ITER_RE.match(line):
                continue
            # The iteration line pads some numbers ("runs= 95"), so the value can be a
            # separate whitespace-delimited token; match key=<optional spaces><number>.
            kv = {float_k: float(float_v)
                  for float_k, float_v in KV_RE.findall(line)}
            if "floors" in kv:
                rows.append(kv)
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarise(path, block):
    rows = parse(path)
    if not rows:
        print(f"{path}: no iterations yet")
        return
    runs = mean([r.get("runs", 0) for r in rows])
    print(f"\n=== {path}: {len(rows)} iterations, {runs:.0f} runs/iter "
          f"({len(rows) * runs:.0f} runs total)")
    head = "  block        " + "".join(f"{f:>11s}" for f in FIELDS)
    print(head)
    for start in range(0, len(rows), block):
        chunk = rows[start:start + block]
        if len(chunk) < max(3, block // 3):
            continue
        cells = "".join(f"{mean([r.get(f, float('nan')) for r in chunk]):>11.3f}"
                        for f in FIELDS)
        print(f"  {start + 1:4d}-{start + len(chunk):<4d}   {cells}")
    # Did it still be improving at the end? Compare halves with an SE on the difference.
    half = len(rows) // 2
    if half >= 5:
        print("  half-over-half deltas (second half minus first, +/- 1 SE):")
        for f in ("a1", "a2", "a3", "floors", "fight_win"):
            a = [r.get(f, float("nan")) for r in rows[:half]]
            b = [r.get(f, float("nan")) for r in rows[half:]]
            a = [x for x in a if not math.isnan(x)]
            b = [x for x in b if not math.isnan(x)]
            if len(a) < 3 or len(b) < 3:
                continue
            va = sum((x - mean(a)) ** 2 for x in a) / (len(a) - 1)
            vb = sum((x - mean(b)) ** 2 for x in b) / (len(b) - 1)
            se = math.sqrt(va / len(a) + vb / len(b))
            d = mean(b) - mean(a)
            flag = "" if abs(d) > 2 * se else "   (within noise)"
            print(f"    {f:10s} {mean(a):6.3f} -> {mean(b):6.3f}   "
                  f"delta {d:+.3f} +/- {se:.3f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--block", type=int, default=25)
    a = ap.parse_args()
    for path in a.logs:
        summarise(path, a.block)


if __name__ == "__main__":
    sys.exit(main())
