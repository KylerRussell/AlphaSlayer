"""Paired comparison of training arms that played the same seeds (same --seed-prefix).

Iteration i of every arm plays the same maps, so per-iteration differences cancel the
seed luck. Prints each arm's window means and the paired difference vs a reference arm with
its standard error (over iterations).

    python3 compare_arms.py --ref logs/collect_long2.log logs/m4_td1.log logs/m4_td2.log --from 6 --to 40
"""
from __future__ import annotations

import argparse
import re
import statistics as st

KEYS = ("win", "a1", "a2", "fight_win", "floors")


def load(path):
    out = {}
    for line in open(path):
        m = re.match(r"^iter +(\d+)/\d+ (?:COLLECT )?runs=", line)
        if m:
            out[int(m.group(1))] = {k: float(re.search(rf"\b{k}=([+-]?[\d.]+)", line).group(1))
                                    for k in KEYS if re.search(rf"\b{k}=", line)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--ref", default=None, help="reference arm (default: the first)")
    ap.add_argument("--from", dest="lo", type=int, default=1)
    ap.add_argument("--to", dest="hi", type=int, default=10**9)
    a = ap.parse_args()
    paths = ([a.ref] if a.ref else []) + [p for p in a.arms if p != a.ref]
    data = {p: load(p) for p in paths}
    ref = paths[0]
    for p in paths:
        its = [i for i in sorted(data[p]) if a.lo <= i <= a.hi]
        line = f"{p:32} n={len(its):3} " + " ".join(
            f"{k}={st.mean(data[p][i][k] for i in its):.3f}" for k in KEYS if its and k in data[p][its[0]])
        print(line)
    for p in paths[1:]:
        common = [i for i in sorted(set(data[p]) & set(data[ref])) if a.lo <= i <= a.hi]
        if len(common) < 2:
            continue
        parts = []
        for k in ("win", "a1", "a2"):
            d = [data[p][i][k] - data[ref][i][k] for i in common]
            parts.append(f"{k} {st.mean(d):+.3f} +- {st.stdev(d) / len(d) ** 0.5:.3f}")
        print(f"  {p} - {ref} (paired, {len(common)} it): " + "  ".join(parts))


if __name__ == "__main__":
    main()
