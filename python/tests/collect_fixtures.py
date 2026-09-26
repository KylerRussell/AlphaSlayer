"""Collects real decisions of every kind from the live probe, for the unified model's unit tests.

Plays full runs with the r5 checkpoints (so acts 2 and 3 are reached) and keeps:
  * every run-level decision, up to --per-kind of each kind
  * every --combat-every'th combat decision, plus every combat decision that shows orbs, allies,
    afflictions or a pending card selection, so the rare token types are covered

Output: tests/fixtures/decisions.jsonl.gz, one {"kind", "obs", "legal"} per line.

    XDG_DATA_HOME="$(../headless_home.sh)" PYTHONHASHSEED=0 PYTHONPATH=. \\
        .venv7/bin/python tests/collect_fixtures.py
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import train_run as T
from alphaslayer.runenv import VecRunEnv
from test_run_obs import load_models

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "decisions.jsonl.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat-ckpt", default="combat_r5.pt")
    ap.add_argument("--run-ckpt", default="run_r5.pt")
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--runs-per-env", type=int, default=2)
    ap.add_argument("--per-kind", type=int, default=150)
    ap.add_argument("--combat-every", type=int, default=12)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    combat, runnet = load_models(a.combat_ckpt, a.run_ckpt, dev)
    kept = collections.Counter()
    seen = collections.Counter()
    rows = []

    def keep(kind, o, l):
        rows.append({"kind": kind, "obs": o, "legal": l})
        kept[kind] += 1

    def policy(kinds, obs, legal, idxs):
        for k, o, l in zip(kinds, obs, legal):
            seen[k] += 1
            if k == "combat":
                rare = (o.get("orbs") or o.get("allies")
                        or any(c.get("affl_idx", 0) for c in o.get("hand") or ())
                        or any(x.get("kind") == "select_card" for x in l))
                if rare or seen[k] % a.combat_every == 0:
                    keep(k, o, l)
            elif kept[k] < a.per_kind:
                keep(k, o, l)
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

    with VecRunEnv(n_envs=a.envs, characters=["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER",
                                             "REGENT"],
                   runs_per_env=a.runs_per_env, seed="FIXTURE_", ascension=0, act_cap=3,
                   run_offset=0) as venv:
        venv.run(policy, deadline_s=1200)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} decisions to {OUT}: {dict(kept)}")


if __name__ == "__main__":
    main()
