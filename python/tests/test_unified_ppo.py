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

print("\n== lambda per FLOOR (lambda = 0): one floor shares credit, the next floor bootstraps")
# travel(f1) -> 2 fight steps (f2) -> card_reward (f2) -> travel (f3), run lost at the end
vals = [0.20, 0.25, 0.30, 0.40, 0.35]
kinds = ["travel", "combat", "combat", "card_reward", "travel"]
floors = [1, 2, 2, 2, 3]
steps = [mk(k, f, v) for k, f, v in zip(kinds, floors, vals)]
P.compute_targets(steps, False, gamma=1.0, lam=0.0)
want = [vals[1], vals[4], vals[4], vals[4], 0.0]
check("every decision on a floor targets the value at the NEXT floor; the last is the outcome",
      all(close(s.ret, w) for s, w in zip(steps, want)), str([round(s.ret, 3) for s in steps]))
check("a fight and its card reward (one floor) bootstrap together from the next floor",
      close(steps[1].ret, vals[4]) and close(steps[3].ret, vals[4]))


def fight_then(v_after, lam):
    # The post-fight decision is placed a floor later so that lambda acts between them.
    s = [mk("combat", 5, 0.3), mk("combat", 5, 0.3), mk("card_reward", 6, v_after), mk("travel", 7, 0.3)]
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
check("...and EVERY decision of that fight gets the same share (no decay within a floor)",
      close(keep[0].adv - spend[0].adv, (1 - 0.5) * 0.20))

print("\n== per-floor discounting")
steps = [mk("combat", 4, 0.5), mk("combat", 4, 0.6), mk("card_reward", 4, 0.7), mk("travel", 6, 0.8)]
P.compute_targets(steps, True, gamma=0.9, lam=0.0)
check("no discount within a floor: every floor-4 decision gets gamma**2 once, for the jump",
      all(close(x.ret, 0.81 * 0.8) for x in steps[:3]), str([round(x.ret, 4) for x in steps]))
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

print("\n== act and fight potentials (dense credit, still win-only optimal)")


def mka(kind, floor, act, v, phi):
    st = mk(kind, floor, v, phi)
    st.act = act
    return st


# Act 1 ends at the card reward after the boss; act 2 starts at the next travel. w_act = 0.5.
seq = [mka("combat", 16, 0, 0.3, 0.5 * 0), mka("card_reward", 16, 0, 0.2, 0.5 * 0),
       mka("travel", 17, 1, 0.9, 0.5 * 1)]
P.compute_targets(seq, False, 1.0, 0.0, bootstrap=False)
check("clearing an act credits the boss floor's decisions (fight and reward) by w_act",
      close(seq[0].adv, 0.5) and close(seq[1].adv, 0.5), str([round(x.adv, 3) for x in seq]))
check("...and dying afterwards pays it back at the death step (it telescopes)",
      close(seq[2].adv, -0.5))
W = dict(w_combat=0.0, w_hp=0.1, w_act=1 / 3, w_fight=1.0)
run_obs = {"player": {"act": 1, "total_floor": 20, "hp": 40, "max_hp": 80}}
fight_obs = {"act": 1, "total_floor": 20, "player": {"hp": 40, "max_hp": 80}, "enemies": []}
phi_run = P.potential("travel", run_obs, **W)
check("run potential = w_act*(act+1) + w_hp*hp%", close(phi_run, 2 / 3 + 0.05), f"{phi_run:.4f}")
check("a fight is worth P(win it) times the run state behind it",
      close(P.potential("combat", fight_obs, **W, p_fight=0.8), 0.8 * phi_run))
check("CONTINUOUS at a won fight's end: P -> 1 hands over phi_run unchanged",
      close(P.potential("combat", fight_obs, **W, p_fight=1.0), phi_run))
check("entering a fight costs exactly its expected risk, (1 - P) * phi_run",
      close(phi_run - P.potential("combat", fight_obs, **W, p_fight=0.9), 0.1 * phi_run))
check("CONTINUOUS at death: P -> 0 meets the terminal potential 0",
      close(P.potential("combat", fight_obs, **W, p_fight=0.0), 0.0))


def fight_run(n_fight):
    """travel -> n_fight decisions of a fight won with certainty -> travel, then run lost."""
    seq = [P.Step("travel", {}, [{}], 0, 0, 0, 1, 5, phi=phi_run)]
    seq += [P.Step("combat", {}, [{}], 0, 0, 0, 1, 6,
                   phi=P.potential("combat", fight_obs, **W, p_fight=1.0)) for _ in range(n_fight)]
    seq += [P.Step("travel", {}, [{}], 0, 0, 0, 1, 7, phi=phi_run)]
    P.compute_targets(seq, False, 1.0, 0.95, bootstrap=False)
    return seq


short, long_ = fight_run(3), fight_run(12)
check("NO incentive to prolong a fight: the decision entering it is credited the same "
      "whether the fight takes 3 decisions or 12", close(short[0].adv, long_[0].adv, 1e-9),
      f"{short[0].adv:.5f} vs {long_[0].adv:.5f}")

print("\n== bootstrapping can be switched off")
v_a = [mk("combat", 2, 0.9), mk("combat", 2, 0.1), mk("card_reward", 2, 0.7)]
v_b = [mk("combat", 2, 0.2), mk("combat", 2, 0.6), mk("card_reward", 2, 0.05)]
for x, y in zip(v_a, v_b):
    x.phi = y.phi = 0.1
P.compute_targets(v_a, True, 1.0, 0.95, bootstrap=False)
P.compute_targets(v_b, True, 1.0, 0.95, bootstrap=False)
check("without bootstrapping, the value head cannot move any advantage",
      all(close(x.adv, y.adv) for x, y in zip(v_a, v_b)), str([round(x.adv, 4) for x in v_a]))
check("...and value targets are the realised outcome", all(close(x.ret, 1.0) for x in v_a))
iso = [mk("combat", 30, 0.3), mk("combat", 30, 0.3)]
P.compute_targets(iso, False, 1.0, 0.95, bootstrap=False, mc_end=None)
check("an isolated fight with an unknown future has NO value target (NaN, masked in the loss)",
      all(math.isnan(x.ret) for x in iso))
iso2 = [mk("combat", 30, 0.3, 0.2), mk("combat", 30, 0.3, 0.25)]
P.compute_targets(iso2, False, 1.0, 0.0, bootstrap=False, phi_end=0.6)
check("a won isolated fight's last decision is credited with the post-fight potential",
      close(iso2[-1].adv, 0.6 - 0.25), f"{iso2[-1].adv:.3f}")

print("\n== value skill (the gate for bootstrapping)")
sk = [P.Step("combat", {}, [{}], 0, 0, v, 1, 1) for v in (1.0, 0.0, 1.0, 0.0)]
for x, y in zip(sk, (1, 0, 1, 0)):
    x.won = y
check("perfect predictions have skill 1", close(P.value_skill(sk)[2], 1.0))
for x in sk:
    x.v = 0.5
check("predicting the base rate has skill 0", close(P.value_skill(sk)[2], 0.0))
for x, v in zip(sk, (0.0, 1.0, 0.0, 1.0)):
    x.v = v
check("confidently wrong predictions have negative skill", P.value_skill(sk)[2] < 0)

print("\n== loss weighting")
kind = torch.tensor([P.COMBAT] * 8 + [FT.DKINDS.index("rest")] * 2 + [FT.DKINDS.index("shop")] * 2)
vals = torch.tensor([1.0] * 8 + [3.0] * 2 + [5.0] * 2)
check("combat and run halves count equally; run kinds equally within theirs",
      close(float(P.group_mean(vals, kind)), 0.5 * 1 + 0.5 * (3 + 5) / 2))
kind2 = torch.tensor([P.COMBAT] * 8 + [FT.DKINDS.index("travel")] * 9 + [FT.DKINDS.index("rest")] * 1)
vals2 = torch.tensor([1.0] * 8 + [2.0] * 9 + [10.0] * 1)
check("flat: each run DECISION counts equally (a rare kind cannot dominate its half)",
      close(float(P.group_mean(vals2, kind2, kind_weight="flat")), 0.5 * 1 + 0.5 * (9 * 2 + 10) / 10, 1e-6))
check("sqrt: kinds weighted by sqrt(count)",
      close(float(P.group_mean(vals2, kind2, kind_weight="sqrt")), 0.5 * 1 + 0.5 * (3 * 2 + 1 * 10) / 4, 1e-6))
check("equal: the 1-sample rest kind weighs as much as 9 travel decisions",
      close(float(P.group_mean(vals2, kind2, kind_weight="equal")), 0.5 * 1 + 0.5 * (2 + 10) / 2, 1e-6))
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

tgt_nan = dict(tgt, ret=torch.full((24,), float("nan"), dtype=torch.float64))
total, parts2 = P.ppo_loss(out, tgt_nan)
check("unknown value targets are masked: the loss stays finite",
      bool(torch.isfinite(total)) and float(parts2["value"]) == 0.0)

print("\n== run-only anchor KL multiplier")
torch.manual_seed(1)
anc = torch.log_softmax(out["logits"] + torch.randn_like(out["logits"]) * b["cand_mask"], -1)
t0, _ = P.ppo_loss(out, tgt, kl_coef=0.0)
t1, pk1 = P.ppo_loss(out, tgt, anchor_logp=anc, kl_coef=1.0)
t3, pk3 = P.ppo_loss(out, tgt, anchor_logp=anc, kl_coef=1.0, kl_run_mult=3.0)
lp_all = torch.log_softmax(out["logits"], -1)
kl_rows = (anc.exp() * (anc - lp_all)).masked_fill(~b["cand_mask"], 0.0).sum(-1)
is_run = tgt["kind"] != P.COMBAT
check("the fixture batch mixes combat and run rows", bool(is_run.any() and (~is_run).any()))
check("mult=1 adds exactly kl_coef * parts['kl']", close(float(t1 - t0), float(pk1["kl"]), 1e-9))
exp3 = 2.0 * float(P.group_mean(kl_rows * is_run, tgt["kind"]))
check("mult=3 adds 2x the RUN half's KL and nothing on combat", close(float(t3 - t1), exp3, 1e-9),
      f"{float(t3 - t1):.6f} vs {exp3:.6f}")
check("parts['kl'] is logged unweighted", close(float(pk3["kl"]), float(pk1["kl"]), 1e-12))

print("\n== advantage centering per decision kind")
from types import SimpleNamespace as NS
rng_ = np.random.default_rng(0)
noise = rng_.normal(0, 0.3, 600)
st_ = ([NS(kind="combat", adv=float(x)) for x in noise[:200]]
       + [NS(kind="potion_gate", adv=float(-1.0 + x)) for x in noise[200:400]]
       + [NS(kind="card_reward", adv=float(0.0 + x)) for x in noise[400:]]
       + [NS(kind="potion_ooc", adv=2.0)])
half = P.normalise_advantages(st_, "half")
kind = P.normalise_advantages(st_, "kind")
raw = np.array([s_.adv for s_ in st_])
rm = np.array([s_.kind != "combat" for s_ in st_])
old = np.where(rm, (raw - raw[rm].mean()) / raw[rm].std(), (raw - raw[~rm].mean()) / raw[~rm].std())
check("center='half' is the previous behaviour exactly", np.allclose(half, old, atol=1e-6))
g = slice(200, 400)
check("half-centering leaves the gate's kind-wide offset in (the long1 bias)",
      half[g].mean() < -0.5, f"{half[g].mean():+.2f}")
check("kind-centering removes it (residual is only the shrinkage)", abs(kind[g].mean()) < 0.1,
      f"{kind[g].mean():+.3f}")
check("within a kind, differences are unchanged (same scale)",
      np.allclose(np.diff(kind[g]), np.diff(half[g]), atol=1e-6))
check("a kind seen once is shrunk toward the half mean, not zeroed", kind[-1] > 1.0, f"{kind[-1]:.2f}")
check("the combat half is untouched by run kinds", np.allclose(kind[:200], half[:200], atol=1e-6))

print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
