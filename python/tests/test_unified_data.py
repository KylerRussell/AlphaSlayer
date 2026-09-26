"""The packed dataset must feed the model EXACTLY what encode_batch would.

Packing is a second implementation of batching (slices of flat arrays instead of per-decision
row lists), and a second implementation is where silent drift hides: a mask off by one, a pad
value of 0 where -1 was meant, a list axis truncated. So every array make_batch returns is
compared, element for element, against encode_batch on the same decisions, over many random
batches of mixed decision kinds, and the labels are checked to arrive intact.

Run from python/:
    PYTHONPATH=. .venv7/bin/python tests/test_unified_data.py
"""
from __future__ import annotations

import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from alphaslayer.unified import dataset as DS
from alphaslayer.unified import features as FT

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


vocab = FT.Vocab(json.load(open(os.path.join(HERE, "..", "vocab_m1.json"))))
rows = [json.loads(l) for l in gzip.open(os.path.join(HERE, "fixtures", "decisions.jsonl.gz"), "rt")]
rng = random.Random(0)
# Synthetic labels, so the target plumbing is checked too.
for i, r in enumerate(rows):
    n = len(r["legal"])
    w = [rng.random() for _ in range(n)]
    r["tp"] = [x / sum(w) for x in w]
    r["a"] = rng.randrange(n)
    r["won"] = i % 2
    r["act_clear"] = (i // 2) % 2
    r["reach_act3"] = (i // 3) % 2
    r["floors_left"] = i % 40
    r["run"] = f"r{i // 300}"
    if r["kind"] == "combat":
        r["fight_won"] = (i // 5) % 2
        r["fight_hp"] = (i % 17) / 17.0
        r["room"] = "Boss" if i % 7 == 0 else "Monster"

print("== packing")
packs = []
for s in range(0, len(rows), 1000):          # several packs, so batches span packs
    packs.append(DS.Packed(DS.pack(rows[s:s + 1000], vocab)))
check("every decision packed", sum(p.n for p in packs) == len(rows),
      f"{sum(p.n for p in packs)} of {len(rows)}")
index = [(p, i) for p in packs for i in range(p.n)]
flat = [r for s in range(0, len(rows), 1000) for r in rows[s:s + 1000]]

print("\n== make_batch == encode_batch, element for element")
mismatch = {}
n_batches = 40
for t in range(n_batches):
    sel = rng.sample(range(len(index)), rng.choice([1, 7, 64, 256]))
    got, tgt = DS.make_batch([index[i] for i in sel])
    want = FT.encode_batch([(flat[i]["kind"], flat[i]["obs"], flat[i]["legal"]) for i in sel], vocab)
    for k in want:
        if k not in got or got[k].shape != want[k].shape or not np.array_equal(got[k], want[k]):
            mismatch[k] = mismatch.get(k, 0) + 1
    extra = set(got) - set(want)
    if extra:
        mismatch["extra keys " + ",".join(sorted(extra))] = 1
check(f"all {len(want)} input arrays identical over {n_batches} random batches",
      not mismatch, str(mismatch))

print("\n== labels arrive intact")
sel = rng.sample(range(len(index)), 256)
_, tgt = DS.make_batch([index[i] for i in sel])
ok_tp = all(np.allclose(tgt["tp"][b, :len(flat[i]["legal"])], flat[i]["tp"], atol=1e-6)
            and not tgt["tp"][b, len(flat[i]["legal"]):].any() for b, i in enumerate(sel))
check("teacher distributions, padded with zeros", ok_tp)
for k in ("a", "won", "act_clear", "reach_act3", "floors_left"):
    check(f"label {k}", all(tgt[k][b] == flat[i][k] for b, i in enumerate(sel)))
comb = [(b, i) for b, i in enumerate(sel) if flat[i]["kind"] == "combat"]
runs = [(b, i) for b, i in enumerate(sel) if flat[i]["kind"] != "combat"]
check("fight labels on combat decisions",
      comb and all(tgt["fight_won"][b] == flat[i]["fight_won"]
                   and abs(tgt["fight_hp"][b] - flat[i]["fight_hp"]) < 1e-6
                   and tgt["boss_fight"][b] == (flat[i]["room"] == "Boss") for b, i in comb))
check("run decisions carry no fight label",
      runs and all(tgt["fight_won"][b] == -1 and np.isnan(tgt["fight_hp"][b]) for b, i in runs))
check("decision kind", all(FT.DKINDS[tgt["kind"][b]] == flat[i]["kind"] for b, i in enumerate(sel)))
check("held-out key is stable per run",
      len({(r["run"], DS.run_key(r["run"])) for r in rows}) == len({r["run"] for r in rows}))

print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
