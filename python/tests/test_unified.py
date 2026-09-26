"""M2 unit tests for the unified encoder + network, on real decisions from the probe.

What they protect:

  * DELIVERY: every M1 observation field reaches the model's OUTPUT. Each field is perturbed on a
    real decision; the encoded arrays AND the network's outputs must change. A field that is sent
    but never read is how the potion ids and upgrade flags sat unused for weeks.
  * ALIGNMENT: candidate i is legal action i, and each candidate points at the right card,
    target, map point and potion.
  * INVARIANCE: the order of the hand, piles, deck and legal list carries no information beyond
    what it maps to. In particular the draw pile's order can never matter.
  * ISOLATION: a decision's outputs do not depend on what else is in its batch.
  * LOOPS: the loop count changes compute, not parameters, and is honoured per call.

Run from python/ (CPU, ~1 minute):
    PYTHONPATH=. .venv7/bin/python tests/test_unified.py
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphaslayer.unified import features as FT
from alphaslayer.unified.net import UnifiedNet, param_count, to_torch

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


VOCAB = FT.Vocab(json.load(open(os.path.join(HERE, "..", "vocab_m1.json"))))
ROWS = [json.loads(l) for l in gzip.open(os.path.join(HERE, "fixtures", "decisions.jsonl.gz"), "rt")]
torch.manual_seed(0)
random.seed(0)
# Small and in float64: these tests check wiring, and float64 makes a real change
# distinguishable from rounding.
NET = UnifiedNet(VOCAB.sizes, d=64, heads=4, prelude=1, core=1, coda=1).double().eval()


def enc(decs):
    b = to_torch(FT.encode_batch(decs, VOCAB), "cpu")
    return {k: (v.double() if v.is_floating_point() else v) for k, v in b.items()}


def run(decs, loop=None):
    with torch.no_grad():
        return NET(enc(decs), loop=loop)


def outputs(decs, loop=None):
    """Every output of the first decision, flattened, for change detection."""
    o = run(decs, loop)
    n = len(decs[0][2])
    return torch.cat([o["logits"][0, :n], o["value_logit"][:1]]
                     + [v[:1] for v in o["aux"].values()])


def first(pred, kind=None):
    for r in ROWS:
        if (kind is None or r["kind"] == kind) and pred(r):
            return copy.deepcopy(r)
    return None


# ---------------------------------------------------------------------------------------------
print("== encoding every fixture decision")
trunc = {}
kinds = {}
bad_len = 0
for i in range(0, len(ROWS), 256):
    chunk = ROWS[i:i + 256]
    b = FT.encode_batch([(r["kind"], r["obs"], r["legal"]) for r in chunk], VOCAB, trunc)
    for j, r in enumerate(chunk):
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        if int(b["cand_mask"][j].sum()) != len(r["legal"]):
            bad_len += 1
check("every decision encodes, with one candidate per legal action",
      bad_len == 0, f"{len(ROWS)} decisions {kinds}")
check("nothing truncated at the default caps", not any(trunc.values()), str(trunc))

# ---------------------------------------------------------------------------------------------
print("\n== candidate references point at the right tokens")
ref_bad = {"play": 0, "use_potion": 0, "travel": 0, "offer": 0}
ref_n = dict.fromkeys(ref_bad, 0)
for r in ROWS[:1500]:
    b = FT.encode_batch([(r["kind"], r["obs"], r["legal"])], VOCAB)
    for j, a in enumerate(r["legal"]):
        cr, er, mr, pr = (int(x) for x in b["cand_ref"][0, j])
        ak = a.get("kind")
        if ak == "play":
            ref_n["play"] += 1
            ok = b["card_ids"][0, cr, 0] == a["card_idx"] and b["card_ids"][0, cr, 6] == FT.LOC_HAND
            ok = ok and (a["target"] < 0 or (er == a["target"] and b["ent_ids"][0, er, 2] == 1))
            ref_bad["play"] += not ok
        elif ak == "use_potion":
            ref_n["use_potion"] += 1
            ref_bad["use_potion"] += not (pr >= 0 and b["pot_ids"][0, pr] == a["potion_idx"])
        elif ak == "travel":
            ref_n["travel"] += 1
            pts = r["obs"]["points"]
            ref_bad["travel"] += not (mr == a["point"]
                                      and b["map_ids"][0, mr] == pts[a["point"]]["type_idx"] + 1)
        elif ak in ("card", "select_card"):
            ref_n["offer"] += 1
            want = a.get("card_idx", a.get("model_idx"))
            ref_bad["offer"] += not (cr >= 0 and b["card_ids"][0, cr, 0] == want)
for k in ref_bad:
    check(f"{k}: references resolve to the matching token", ref_n[k] > 0 and ref_bad[k] == 0,
          f"{ref_bad[k]}/{ref_n[k]} wrong")

# ---------------------------------------------------------------------------------------------
print("\n== forward pass")
sample = random.sample(ROWS, 64)
o = run([(r["kind"], r["obs"], r["legal"]) for r in sample])
b = enc([(r["kind"], r["obs"], r["legal"]) for r in sample])
valid = b["cand_mask"]
check("logits finite on every real candidate", bool(torch.isfinite(o["logits"][valid]).all()))
check("padding candidates are masked far below real ones",
      bool((o["logits"][~valid] < -1e30).all()) if (~valid).any() else True)
check("value is a probability", bool(((o["value"] > 0) & (o["value"] < 1)).all()))
check("all aux heads present", set(o["aux"]) == {"fight_win", "fight_hp_loss", "act_clear",
                                                 "reach_act3", "floors_left"})

worst = 0.0
for i, r in enumerate(sample):
    alone = run([(r["kind"], r["obs"], r["legal"])])
    n = len(r["legal"])
    worst = max(worst, float((alone["logits"][0, :n] - o["logits"][i, :n]).abs().max()),
                float((alone["value_logit"][0] - o["value_logit"][i]).abs()))
check("outputs do not depend on the rest of the batch", worst < 1e-9, f"max diff {worst:.2e}")

# ---------------------------------------------------------------------------------------------
print("\n== order invariance / equivariance")
r = first(lambda r: len(r["legal"]) >= 4, "combat")
base = run([(r["kind"], r["obs"], r["legal"])])
rev = run([(r["kind"], r["obs"], r["legal"][::-1])])
n = len(r["legal"])
check("reversing the legal list reverses the logits",
      float((base["logits"][0, :n] - rev["logits"][0, :n].flip(0)).abs().max()) < 1e-9)

r = first(lambda r: len(r["obs"]["hand"]) >= 3 and any(a["kind"] == "play" for a in r["legal"]),
          "combat")
perm = list(range(len(r["obs"]["hand"])))
random.Random(3).shuffle(perm)            # new position k holds old card perm[k]
inv = {old: new for new, old in enumerate(perm)}
r2 = copy.deepcopy(r)
r2["obs"]["hand"] = [r["obs"]["hand"][k] for k in perm]
for a in r2["legal"]:
    if a["kind"] == "play":
        a["hand"] = inv[a["hand"]]
b1, b2 = run([(r["kind"], r["obs"], r["legal"])]), run([(r2["kind"], r2["obs"], r2["legal"])])
n = len(r["legal"])
check("permuting the hand (with the actions remapped) changes nothing",
      float((b1["logits"][0, :n] - b2["logits"][0, :n]).abs().max()) < 1e-9
      and float((b1["value_logit"] - b2["value_logit"]).abs().max()) < 1e-9)

r = first(lambda r: len(r["obs"].get("draw_pile") or []) >= 3, "combat")
r2 = copy.deepcopy(r)
random.Random(4).shuffle(r2["obs"]["draw_pile"])
check("the draw pile's order cannot change any output",
      float((outputs([(r["kind"], r["obs"], r["legal"])])
             - outputs([(r2["kind"], r2["obs"], r2["legal"])])).abs().max()) < 1e-9)

r = first(lambda r: len(r["obs"].get("deck_cards") or []) >= 8, "card_reward")
r2 = copy.deepcopy(r)
random.Random(5).shuffle(r2["obs"]["deck_cards"])
check("the deck's order cannot change any output",
      float((outputs([(r["kind"], r["obs"], r["legal"])])
             - outputs([(r2["kind"], r2["obs"], r2["legal"])])).abs().max()) < 1e-9)

# ---------------------------------------------------------------------------------------------
print("\n== delivery: every M1 field reaches the output")


def hand_with(pred):
    return lambda r: any(pred(c) for c in r["obs"]["hand"])


def mut_hand(field, fn, pred=lambda c: True):
    def m(r):
        for c in r["obs"]["hand"]:
            if pred(c):
                c[field] = fn(c)
                return
    return m


def mut_first(path_fn):
    return path_fn


def set_in(getter, key, fn):
    def m(r):
        d = getter(r)
        d[key] = fn(d.get(key))
    return m


def vals_bump(slot):
    return lambda v: dict(v or {}, **{slot: (v or {}).get(slot, 0) + 7})


FIELDS = [
    # (name, kind, applicable(row), mutate(row))
    ("hand: keywords", "combat", hand_with(lambda c: True), mut_hand("kw", lambda c: (c.get("kw") or 0) ^ 2)),
    ("hand: damage number", "combat", hand_with(lambda c: True), mut_hand("vals", lambda c: vals_bump("damage")(c.get("vals")))),
    ("hand: gold number", "combat", hand_with(lambda c: True), mut_hand("vals", lambda c: vals_bump("gold")(c.get("vals")))),
    ("hand: affliction", "combat", hand_with(lambda c: True), mut_hand("affl_idx", lambda c: 1)),
    ("hand: enchantment", "combat", hand_with(lambda c: True), mut_hand("ench_idx", lambda c: 2)),
    ("enemy: next move", "combat", lambda r: any(e.get("move") for e in r["obs"]["enemies"]),
     lambda r: r["obs"]["enemies"][0].__setitem__("move", "SOME_OTHER_MOVE")),
    ("enemy: status-card count", "combat", lambda r: any(e.get("intents") for e in r["obs"]["enemies"]),
     lambda r: [e for e in r["obs"]["enemies"] if e.get("intents")][0]["intents"][0].__setitem__("count", 3)),
    ("enemy: power amount", "combat", lambda r: any(e.get("powers") for e in r["obs"]["enemies"]),
     lambda r: [e for e in r["obs"]["enemies"] if e.get("powers")][0]["powers"][0].__setitem__("amount", 99)),
    ("ally (Osty): hp", "combat", lambda r: bool(r["obs"].get("allies")),
     lambda r: r["obs"]["allies"][0].update(hp=r["obs"]["allies"][0]["hp"] + 5,
                                            max_hp=r["obs"]["allies"][0]["max_hp"] + 5)),
    ("orb: passive value", "combat", lambda r: bool(r["obs"].get("orbs")),
     lambda r: r["obs"]["orbs"][0].__setitem__("passive", 12)),
    ("orb slots", "combat", lambda r: True, set_in(lambda r: r["obs"], "orb_slots", lambda x: (x or 0) + 2)),
    ("pile: upgrade", "combat", lambda r: bool(r["obs"].get("draw_pile")),
     lambda r: r["obs"]["draw_pile"][0].__setitem__("up", 1)),
    ("pile: keywords", "combat", lambda r: bool(r["obs"].get("draw_pile")),
     lambda r: r["obs"]["draw_pile"][0].__setitem__("kw", 2)),
    ("pile: count", "combat", lambda r: bool(r["obs"].get("draw_pile")),
     lambda r: r["obs"]["draw_pile"][0].__setitem__("n", r["obs"]["draw_pile"][0]["n"] + 2)),
    ("relic: numbers", "combat", lambda r: bool(r["obs"]["player"].get("relics")),
     lambda r: r["obs"]["player"]["relics"][0].__setitem__("vals", vals_bump("heal")(r["obs"]["player"]["relics"][0].get("vals")))),
    ("relic: counter", "combat", lambda r: bool(r["obs"]["player"].get("relics")),
     lambda r: r["obs"]["player"]["relics"][0].__setitem__("counter", 2)),
    ("potion: numbers", "combat", lambda r: bool(r["obs"]["player"].get("potion_slots")),
     lambda r: r["obs"]["player"]["potion_slots"][0].__setitem__("vals", vals_bump("damage")(r["obs"]["player"]["potion_slots"][0].get("vals")))),
    ("combat: act_floor", "combat", lambda r: True, set_in(lambda r: r["obs"], "act_floor", lambda x: (x or 0) + 5)),
    ("combat: total_floor", "combat", lambda r: True, set_in(lambda r: r["obs"], "total_floor", lambda x: (x or 0) + 5)),
    ("combat: act boss", "combat", lambda r: True, set_in(lambda r: r["obs"], "boss_idx", lambda x: ((x or 0) % 80) + 1)),
    ("combat: character", "combat", lambda r: True,
     set_in(lambda r: r["obs"], "character", lambda x: "SILENT" if x != "SILENT" else "DEFECT")),
    ("combat: room type", "combat", lambda r: True, set_in(lambda r: r["obs"], "room_idx", lambda x: ((x or 0) + 1) % 5)),
    ("combat: encounter", "combat", lambda r: True, set_in(lambda r: r["obs"], "encounter", lambda x: "OTHER_ENCOUNTER")),
    ("combat: stars", "combat", lambda r: True, set_in(lambda r: r["obs"]["player"], "stars", lambda x: (x or 0) + 3)),
    ("action: per-target damage", "combat", lambda r: any(a["kind"] == "play" for a in r["legal"]),
     lambda r: [a for a in r["legal"] if a["kind"] == "play"][0].__setitem__("vals", vals_bump("damage")([a for a in r["legal"] if a["kind"] == "play"][0].get("vals")))),
    ("run: deck keywords", "card_reward", lambda r: bool(r["obs"].get("deck_cards")),
     lambda r: r["obs"]["deck_cards"][0].__setitem__("kw", (r["obs"]["deck_cards"][0].get("kw") or 0) ^ 2)),
    ("run: deck numbers", "card_reward", lambda r: bool(r["obs"].get("deck_cards")),
     lambda r: r["obs"]["deck_cards"][0].__setitem__("vals", vals_bump("block")(r["obs"]["deck_cards"][0].get("vals")))),
    ("run: act boss", "card_reward", lambda r: True, set_in(lambda r: r["obs"], "boss_idx", lambda x: ((x or 0) % 80) + 1)),
    ("run: act", "card_reward", lambda r: True, set_in(lambda r: r["obs"], "act", lambda x: ((x or 0) + 1) % 3)),
    ("run: act_floor", "rest", lambda r: True, set_in(lambda r: r["obs"], "act_floor", lambda x: (x or 0) + 5)),
    ("run: total_floor", "rest", lambda r: True, set_in(lambda r: r["obs"], "total_floor", lambda x: (x or 0) + 5)),
    ("run: relic numbers", "shop", lambda r: bool(r["obs"].get("relics")),
     lambda r: r["obs"]["relics"][0].__setitem__("vals", vals_bump("gold")(r["obs"]["relics"][0].get("vals")))),
    ("run: held potion numbers", "card_reward", lambda r: bool(r["obs"].get("potions")),
     lambda r: r["obs"]["potions"][0].__setitem__("vals", vals_bump("heal")(r["obs"]["potions"][0].get("vals")))),
    ("card reward: offered card numbers", "card_reward", lambda r: any(a.get("kind") == "card" for a in r["legal"]),
     lambda r: [a for a in r["legal"] if a.get("kind") == "card"][0].__setitem__("vals", vals_bump("damage")([a for a in r["legal"] if a.get("kind") == "card"][0].get("vals")))),
    ("card reward: offered card keywords", "card_reward", lambda r: any(a.get("kind") == "card" for a in r["legal"]),
     lambda r: [a for a in r["legal"] if a.get("kind") == "card"][0].__setitem__("kw", 2)),
    ("shop: price", "shop", lambda r: any(a.get("kind") == "relic" for a in r["legal"]),
     lambda r: [a for a in r["legal"] if a.get("kind") == "relic"][0].__setitem__("cost", 999)),
    ("event: option numbers", "event", lambda r: True,
     lambda r: r["legal"][0].__setitem__("vals", vals_bump("hp_loss")(r["legal"][0].get("vals")))),
    ("event: event numbers", "event", lambda r: True,
     lambda r: r["legal"][0].__setitem__("event_vals", vals_bump("gold")(r["legal"][0].get("event_vals")))),
    ("event: cards shown", "event", lambda r: True,
     lambda r: r["legal"][0].__setitem__("card_idxs", [21])),
    ("event: hover tips", "event", lambda r: True,
     lambda r: r["legal"][0].__setitem__("tips", ["CARD.GREED"])),
    ("event: lethal flag", "event", lambda r: True,
     lambda r: r["legal"][0].__setitem__("will_kill", not r["legal"][0].get("will_kill"))),
    ("rest: option identity", "rest", lambda r: True,
     lambda r: r["legal"][0].__setitem__("option", "SOMETHING_ELSE")),
    ("travel: map point visited", "travel", lambda r: True,
     lambda r: r["obs"]["points"][0].__setitem__("visited", not r["obs"]["points"][0].get("visited"))),
    ("travel: path lookahead (map children)", "travel", lambda r: True,
     lambda r: r["obs"]["points"][r["legal"][0]["point"]].__setitem__("children", [])),
    ("potion gate: room type", "potion_gate", lambda r: True,
     set_in(lambda r: r["obs"], "room_type", lambda x: "Boss" if x != "Boss" else "Monster")),
]

for name, kind, applies, mutate in FIELDS:
    r = first(applies, kind)
    if r is None:
        check(f"{name}", False, "no fixture decision to test it on")
        continue
    r2 = copy.deepcopy(r)
    mutate(r2)
    e1 = FT.encode_batch([(r["kind"], r["obs"], r["legal"])], VOCAB)
    e2 = FT.encode_batch([(r2["kind"], r2["obs"], r2["legal"])], VOCAB)
    enc_changed = any(e1[k].shape != e2[k].shape or (e1[k] != e2[k]).any() for k in e1)
    out_diff = float((outputs([(r["kind"], r["obs"], r["legal"])])
                      - outputs([(r2["kind"], r2["obs"], r2["legal"])])).abs().max())
    check(f"{name}", enc_changed and out_diff > 1e-9,
          f"encoding {'changed' if enc_changed else 'UNCHANGED'}, output diff {out_diff:.1e}")

# ---------------------------------------------------------------------------------------------
print("\n== loops and size")
decs = [(r["kind"], r["obs"], r["legal"]) for r in random.sample(ROWS, 16)]
o1, o3 = run(decs, loop=1), run(decs, loop=3)
check("loop count changes the outputs",
      float((o1["value_logit"] - o3["value_logit"]).abs().max()) > 1e-6)
check("eval forward is deterministic",
      float((run(decs, loop=3)["value_logit"] - o3["value_logit"]).abs().max()) == 0.0)
p1 = param_count(UnifiedNet(VOCAB.sizes, loop=1))
p3 = param_count(UnifiedNet(VOCAB.sizes, loop=3))
check("looping adds no parameters", p1 == p3, f"{p1 / 1e6:.2f}M vs {p3 / 1e6:.2f}M")
check("default size is the chosen 18-19M", 17e6 < p1 < 20e6, f"{p1 / 1e6:.2f}M")

# ---------------------------------------------------------------------------------------------
print("\n== per-row loop counts and adaptive depth")
from alphaslayer.unified.net import choose_loops
decs = [(r["kind"], r["obs"], r["legal"]) for r in random.sample(ROWS, 12)]
b = enc(decs)
with torch.no_grad():
    same = NET(b, loop_rows=torch.full((12,), 2, dtype=torch.long))
    ref = NET(b, loop=2)
check("loop_rows all equal to K matches loop=K",
      float((same["logits"] - ref["logits"]).abs().max()) < 1e-9)
ks = torch.tensor([1, 2, 3, 4] * 3)
with torch.no_grad():
    mixed = NET(b, loop_rows=ks)
worst = 0.0
for i, (d, k) in enumerate(zip(decs, ks.tolist())):
    alone = run([d], loop=k)
    n = len(d[2])
    worst = max(worst, float((alone["logits"][0, :n] - mixed["logits"][i, :n]).abs().max()),
                float((alone["value_logit"][0] - mixed["value_logit"][i]).abs()))
check("mixed per-row loop counts give each row exactly its own-depth output",
      worst < 1e-9, f"max diff {worst:.2e}")
with torch.no_grad():
    traj = NET.forward_trajectory(b, 4)
worst = max(float((traj[k - 1]["logits"] - run(decs, loop=k)["logits"]).abs().max())
            for k in (1, 2, 3, 4))
check("trajectory readout k equals a forward pass at loop=k", worst < 1e-9, f"{worst:.2e}")
caps = torch.tensor([1, 4] * 6)
chosen = choose_loops(traj, b["cand_mask"], caps)
check("adaptive depth stays within [1, cap] per row",
      bool(((chosen >= 1) & (chosen <= caps)).all()), f"chosen {chosen.tolist()}")
loose = choose_loops(traj, b["cand_mask"], torch.full((12,), 4), margin=1e9, stable_kl=-1.0)
forced = b["cand_mask"].sum(1) <= 1           # a single legal action is decided at pass 1
check("with stopping disabled, every row with a real choice runs to its cap",
      bool((loose[~forced] == 4).all()) and bool((loose[forced] == 1).all()),
      f"{int(forced.sum())} single-action rows, chosen {loose.tolist()}")
eager = choose_loops(traj, b["cand_mask"], torch.full((12,), 4), margin=-1.0)
check("with a zero margin, every row stops after one pass", bool((eager == 1).all()))

print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
