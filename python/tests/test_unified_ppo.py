"""M4 credit-math tests: exact numbers on synthetic runs, not "it ran".

  * win-only returns: with gamma=1, lambda=1 every step's value target IS the run outcome
  * TD(0) across the fight -> run boundary: a fight's last decision bootstraps from the value of
    the run decision after it, so a fight that leaves a better state is credited for it
  * per-floor discounting: no discount inside a fight, gamma**k across k floors
  * death mid-fight: open fight steps are labelled lost, and the last step's target is 0
  * shaping in the advantage ONLY: value targets unchanged; at gamma=lambda=1 the shaped
    advantage differs from the unshaped one by exactly -phi(s_t) (it telescopes)
  * group weighting: combat and run halves count equally, run kinds equally within the half
  * PPO consistency: with recorded per-row loop counts replayed, the new/old ratio is exactly 1
    before any update, so the first epoch starts from the true policy

Run from python/:
    PYTHONPATH=. .venv7/bin/python tests/test_unified_ppo.py
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from alphaslayer.unified import features as FT
from alphaslayer.unified import ppo as P
from alphaslayer.unified.net import UnifiedNet, to_torch

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def mk(kind, floor, v, phi=0.0):
    return P.Step(kind=kind, obs={}, legal=[{}], a=0, logp=0.0, v=v, loop=1, floor=floor, phi=phi)


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


print("== win-only returns")
for won in (True, False):
    steps = [mk("travel", 1, 0.0), mk("combat", 2, 0.0), mk("combat", 2, 0.0), mk("card_reward", 2, 0.0)]
    P.compute_targets(steps, won, gamma=1.0, lam=1.0)
    check(f"gamma=lambda=1, V=0: every value target = outcome ({'won' if won else 'lost'})",
          all(close(s.ret, float(won)) for s in steps), str([s.ret for s in steps]))

print("\n== TD(0) across the fight -> run boundary (lambda = 0)")
# travel(f1) -> 2 fight steps (f2) -> card_reward (f2) -> travel (f3), run lost at the end
vals = [0.20, 0.25, 0.30, 0.40, 0.35]
kinds = ["travel", "combat", "combat", "card_reward", "travel"]
floors = [1, 2, 2, 2, 3]
steps = [mk(k, f, v) for k, f, v in zip(kinds, floors, vals)]
P.compute_targets(steps, False, gamma=1.0, lam=0.0)
want = [vals[1], vals[2], vals[3], vals[4], 0.0]        # ret_t = V(s_{t+1}), last = 0
check("each target is the next state's value; the last is the outcome",
      all(close(s.ret, w) for s, w in zip(steps, want)), str([round(s.ret, 3) for s in steps]))
check("the fight's last decision bootstraps from the card_reward that follows it",
      close(steps[2].ret, vals[3]))


def fight_then(v_after, lam):
    s = [mk("combat", 5, 0.3), mk("combat", 5, 0.3), mk("card_reward", 5, v_after), mk("travel", 6, 0.3)]
    P.compute_targets(s, False, gamma=1.0, lam=lam)
    return s


# Two runs identical except the post-fight state's VALUE (e.g. potion kept, V=0.30, vs drunk in a
# hallway fight, V=0.10) and with the same actual future. The value gap reaches the fight's last
# decision with weight (1 - lambda): at lambda=0 all of it, and as lambda -> 1 GAE leans on what
# actually happened next, which here is identical. lambda therefore sets how much fight decisions
# are credited through the value head rather than through realised outcomes.
for lam in (0.0, 0.5, 0.95):
    keep, spend = fight_then(0.30, lam), fight_then(0.10, lam)
    check(f"lambda={lam}: a better post-fight state credits the fight's last decision by (1-lambda)*0.20",
          close(keep[1].adv - spend[1].adv, (1 - lam) * 0.20),
          f"{keep[1].adv - spend[1].adv:.4f}")
keep, spend = fight_then(0.30, 0.5), fight_then(0.10, 0.5)
check("...and earlier decisions of the fight get a lambda-discounted share of it",
      close(keep[0].adv - spend[0].adv, 0.5 * (1 - 0.5) * 0.20))

print("\n== per-floor discounting")
steps = [mk("combat", 4, 0.5), mk("combat", 4, 0.6), mk("card_reward", 4, 0.7), mk("travel", 6, 0.8)]
P.compute_targets(steps, True, gamma=0.9, lam=0.0)
check("no discount between decisions on the same floor", close(steps[0].ret, 0.6) and close(steps[1].ret, 0.7))
check("gamma**2 across a two-floor jump", close(steps[2].ret, 0.81 * 0.8), f"{steps[2].ret:.4f}")
check("the winning last step's target is 1", close(steps[3].ret, 1.0))

print("\n== death mid-fight")
buf = P.RunBuffer(lam=0.0)
buf.add(0, "travel", {"player": {"total_floor": 3, "act": 0}}, [{}], 0, 0.0, 0.4, 1)
buf.add(0, "combat", {"total_floor": 4, "act": 0, "player": {}}, [{}], 0, 0.0, 0.3, 1)
buf.add(0, "combat", {"total_floor": 4, "act": 0, "player": {}}, [{}], 0, 0.0, 0.2, 1)
buf.end_run(0, won=False, act_reached=0, floors=4)
fs = [s for s in buf.done if s.kind == "combat"]
check("an unfinished fight is labelled lost", all(s.fight_won == 0 for s in fs))
check("the dying decision's value target is 0", close(buf.done[-1].ret, 0.0))
check("act_clear is 0 for act-1 decisions of a run that died in act 1",
      all(s.act_clear == 0 for s in buf.done))

buf = P.RunBuffer()
buf.add(0, "travel", {"player": {"total_floor": 1, "act": 0}}, [{}], 0, 0.0, 0.4, 1)
buf.add(0, "travel", {"player": {"total_floor": 20, "act": 1}}, [{}], 0, 0.0, 0.4, 1)
buf.end_run(0, won=False, act_reached=1, floors=25)
check("act_clear: act-1 decision 1, act-2 decision 0, when the run died in act 2",
      [s.act_clear for s in buf.done] == [1, 0])
check("floors_left counts from each decision's floor", [s.floors_left for s in buf.done] == [24.0, 5.0])

print("\n== shaping lives in the advantage only")
base = [mk("combat", 2, 0.3, 0.0), mk("combat", 2, 0.35, 0.0), mk("card_reward", 2, 0.4, 0.0)]
shaped = [mk("combat", 2, 0.3, 0.2), mk("combat", 2, 0.35, -0.1), mk("card_reward", 2, 0.4, 0.05)]
P.compute_targets(base, True, 1.0, 1.0)
P.compute_targets(shaped, True, 1.0, 1.0)
check("value targets are identical with and without shaping",
      all(close(a.ret, b.ret) for a, b in zip(base, shaped)))
check("at gamma=lambda=1 the shaped advantage = unshaped - phi(s_t) (it telescopes)",
      all(close(b.adv - a.adv, -b.phi) for a, b in zip(base, shaped)),
      str([round(b.adv - a.adv, 3) for a, b in zip(base, shaped)]))

print("\n== loss weighting")
kind = torch.tensor([P.COMBAT] * 8 + [FT.DKINDS.index("rest")] * 2 + [FT.DKINDS.index("shop")] * 2)
vals = torch.tensor([1.0] * 8 + [3.0] * 2 + [5.0] * 2)
check("combat and run halves count equally; run kinds equally within theirs",
      close(float(P.group_mean(vals, kind)), 0.5 * 1 + 0.5 * (3 + 5) / 2))
steps = [mk("combat", 1, 0)] * 0
st = [P.Step("combat", {}, [{}], 0, 0, 0, 1, 1) for _ in range(50)] + \
     [P.Step("rest", {}, [{}], 0, 0, 0, 1, 1) for _ in range(10)]
rng = np.random.default_rng(0)
for s in st:
    s.adv = float(rng.normal(3 if s.kind == "combat" else -7, 2 if s.kind == "combat" else 9))
na = P.normalise_advantages(st)
c, r = na[:50], na[50:]
check("advantages normalised per half to mean 0, std 1",
      abs(c.mean()) < 1e-5 and abs(c.std() - 1) < 1e-3 and abs(r.mean()) < 1e-5 and abs(r.std() - 1) < 1e-3)

print("\n== PPO ratio is exactly 1 before the first update, with recorded loops replayed")
torch.manual_seed(0)
vocab = FT.Vocab(json.load(open(os.path.join(HERE, "..", "vocab_m1.json"))))
rows = [json.loads(l) for l in gzip.open(os.path.join(HERE, "fixtures", "decisions.jsonl.gz"), "rt")]
random.seed(0)
sample = random.sample(rows, 24)
net = UnifiedNet(vocab.sizes, d=64, heads=4, prelude=1, core=1, coda=1).double().eval()
b = {k: (v.double() if v.is_floating_point() else v)
     for k, v in to_torch(FT.encode_batch([(r["kind"], r["obs"], r["legal"]) for r in sample], vocab), "cpu").items()}
loops = torch.tensor([1, 2, 3] * 8)
with torch.no_grad():
    # "rollout": each row at its own loop count, alone, sampled
    old_logp, acts = [], []
    for j, r in enumerate(sample):
        bj = {k: (v.double() if v.is_floating_point() else v) for k, v in
              to_torch(FT.encode_batch([(r["kind"], r["obs"], r["legal"])], vocab), "cpu").items()}
        lp = torch.log_softmax(net(bj, loop=int(loops[j]))["logits"][0, :len(r["legal"])], -1)
        a = int(torch.multinomial(lp.exp(), 1))
        acts.append(a)
        old_logp.append(float(lp[a]))
    out = net(b, loop_rows=loops)
tgt = {"a": torch.tensor(acts), "old_logp": torch.tensor(old_logp, dtype=torch.float64),
       "adv": torch.randn(24, dtype=torch.float64), "ret": torch.rand(24, dtype=torch.float64),
       "kind": torch.tensor([FT.DKINDS.index(r["kind"]) for r in sample]),
       "cand_mask": b["cand_mask"], "act_clear": torch.ones(24, dtype=torch.float64),
       "reach_act3": torch.zeros(24, dtype=torch.float64),
       "floors_left": torch.full((24,), 10.0, dtype=torch.float64),
       "fight_won": torch.full((24,), -1.0, dtype=torch.float64),
       "fight_hp": torch.full((24,), float("nan"), dtype=torch.float64)}
_, parts = P.ppo_loss(out, tgt)
check("batched update at recorded per-row loops reproduces rollout log-probs",
      float(parts["clipfrac"]) == 0.0 and abs(float(parts["approx_kl"])) < 1e-12,
      f"approx_kl {float(parts['approx_kl']):.1e}")
out1 = net(b, loop=1)
_, parts1 = P.ppo_loss(out1, tgt)
check("...and replaying the WRONG loop counts does not (the test can fail)",
      abs(float(parts1["approx_kl"])) > 1e-6, f"approx_kl {float(parts1['approx_kl']):.1e}")

print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
