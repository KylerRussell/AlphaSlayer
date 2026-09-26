"""Observation extraction must not change the game.

The combat observation now calls the game's card-preview path (UpdateDynamicVarPreview) on
every hand card and every targeted play action. The game documents PreviewValue as
display-only, but "documented" is not "verified", and this project's search plans depend on
exact determinism. So: play identical seeds with a scripted policy under two probe builds and
require the games to be IDENTICAL, step by step.

The policy reads only fields both builds send (card ids, targets), so any divergence can only
come from the game itself behaving differently.

    # side effects: record under a build WITHOUT previews, then WITH them, then compare
    PYTHONPATH=. .venv7/bin/python tests/test_obs_no_side_effects.py record sidefx old.json
    PYTHONPATH=. .venv7/bin/python tests/test_obs_no_side_effects.py record sidefx new.json
    PYTHONPATH=. .venv7/bin/python tests/test_obs_no_side_effects.py compare old.json new.json

    # seeded reproducibility: record the "mix" set twice under ONE build and compare
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaslayer.vecenv import VecSpireEnv

ROSTER = ["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"]
SETS = {
    # Side effects: round-robin encounters ("all" pool), which need no seeded sampler, so an
    # older build is a valid baseline. Hallways, elites and bosses, act 1 and act 3.
    "sidefx": [
        ("act1_all", dict(characters=ROSTER, encounters="all")),
        ("act3_all", dict(characters=ROSTER, encounters="all", act=2)),
    ],
    # Reproducibility of SEEDED sampling: the weighted mix draws encounters from a Random
    # seeded by --probe-seed. Record twice under one build; the two must match.
    "mix": [
        ("act1_mix", dict(characters=ROSTER, encounters="mix", mix="0.34,0.33,0.33")),
        ("act3_mix", dict(characters=ROSTER, encounters="mix", mix="0.34,0.33,0.33", act=2)),
    ],
}


def fingerprint(o, legal):
    return [
        [c["card"] for c in o["hand"]],
        o["player"]["hp"], o["player"]["block"], o["player"]["energy"],
        [(e.get("monster"), e["hp"], e["block"], e["alive"]) for e in o["enemies"]],
        o["draw_count"], o["discard_count"], o["exhaust_count"],
        len(legal),
    ]


def scripted(legal):
    """Deterministic, reads only card ids and targets: the lexicographically first play, else
    end turn."""
    plays = [(a.get("card") or "", a["target"], i) for i, a in enumerate(legal)
             if a["kind"] in ("play", "select_card")]
    if plays:
        return min(plays)[2]
    for i, a in enumerate(legal):
        if a["kind"] in ("end_turn", "select_done"):
            return i
    return 0


def record(path, which):
    out = {}
    for name, kw in SETS[which]:
        traces = {}

        def policy(obs, legal, idxs):
            for o, l, i in zip(obs, legal, idxs):
                traces.setdefault(i, []).append(fingerprint(o, l))
            return [scripted(l) for l in legal]

        with VecSpireEnv(n_envs=5, episodes_per_env=4, seed=f"SIDEFX_{name}_", **kw) as env:
            env.run(policy)
            results = [(r.outcome if hasattr(r, "outcome") else str(r)) for r in env.results]
        out[name] = {"traces": {str(k): v for k, v in sorted(traces.items())},
                     "results": results}
        print(f"  {name}: {sum(len(v) for v in traces.values())} steps, "
              f"{len(results)} fights")
    json.dump(out, open(path, "w"))


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    fails = 0
    for name in a:
        ta, tb = a[name]["traces"], b[name]["traces"]
        steps = sum(len(v) for v in ta.values())
        diverged = []
        for env in sorted(set(ta) | set(tb)):
            sa, sb = ta.get(env, []), tb.get(env, [])
            first = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), None)
            if first is None and len(sa) != len(sb):
                first = min(len(sa), len(sb))
            if first is not None:
                diverged.append((env, first))
        # env.results is in COMPLETION order across parallel envs, which varies run to run, so
        # outcomes are compared as a multiset; the per-env traces above carry the ordering.
        same_outcomes = sorted(a[name]["results"]) == sorted(b[name]["results"])
        ok = not diverged and same_outcomes and steps > 100
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: identical games under both builds "
              f"-- {steps} steps, {len(a[name]['results'])} fights"
              + (f", diverged at (env, step) {diverged[:5]}" if diverged else ""))
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(sys.argv[3], sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
