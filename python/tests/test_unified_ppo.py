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

print("\n== group_mean (sync-free) matches the reference implementation")
def group_mean_ref(per_row, kind, run_weight=0.5, kind_weight="equal"):
    is_c = kind == P.COMBAT
    c = per_row[is_c].mean() if is_c.any() else None
    ks = [k for k in kind.unique().tolist() if k != P.COMBAT]
    r = None
    if ks:
        means = torch.stack([per_row[kind == k].mean() for k in ks])
        if kind_weight == "equal":
            r = means.mean()
        else:
            n = torch.tensor([float((kind == k).sum()) for k in ks], dtype=means.dtype)
            w = n if kind_weight == "flat" else n.sqrt()
            r = (means * w).sum() / w.sum()
    if c is not None and r is not None:
        return (1 - run_weight) * c + run_weight * r
    return c if c is not None else (r if r is not None else per_row.sum() * 0)
g_ = torch.Generator().manual_seed(0)
worst = 0.0
for trial in range(60):
    n_ = int(torch.randint(1, 40, (1,), generator=g_))
    mode = trial % 3
    kinds_ = torch.randint(0, len(FT.DKINDS), (n_,), generator=g_)
    if mode == 1:
        kinds_[:] = P.COMBAT                      # combat only
    elif mode == 2:
        kinds_[kinds_ == P.COMBAT] = (P.COMBAT + 1) % len(FT.DKINDS)   # run only
    x_ = torch.randn(n_, generator=g_, dtype=torch.float64)
    for kw in ("equal", "flat", "sqrt"):
        worst = max(worst, abs(float(P.group_mean(x_, kinds_, 0.3, kw) - group_mean_ref(x_, kinds_, 0.3, kw))))
check("equal / flat / sqrt agree on 60 random batches (incl. combat-only and run-only)",
      worst < 1e-9, f"max |diff| {worst:.2e}")
x_ = torch.randn(10, dtype=torch.float64, requires_grad=True)
P.group_mean(x_, torch.randint(0, len(FT.DKINDS), (10,)), 0.5, "flat").backward()
check("gradients are finite", bool(torch.isfinite(x_.grad).all()))

print("\n== seed-group baseline (leave-one-out, per seed x floor x kind)")
from types import SimpleNamespace as NS2
def run_(adv_by_step):     # [(floor, kind, adv), ...]
    return [NS2(floor=f, kind=k, adv=a) for f, k, a in adv_by_step]
r0 = run_([(1, "travel", 1.0), (2, "combat", 0.5), (2, "combat", 1.5)])   # group 0
r1 = run_([(1, "travel", 3.0), (2, "combat", 2.0)])                       # group 0
r2 = run_([(1, "travel", 5.0), (3, "rest", 7.0)])                         # group 0
r3 = run_([(1, "travel", 9.0)])                                           # group 1, alone
r4 = run_([(1, "travel", 4.0)])                                           # no group
steps_ = r0 + r1 + r2 + r3 + r4
runs_ = [{"n": 3, "group": 0}, {"n": 2, "group": 0}, {"n": 2, "group": 0},
         {"n": 1, "group": 1}, {"n": 1, "group": -1}]
P.seed_group_baseline(steps_, runs_)
check("travel@1: each play minus the mean of the OTHER two plays of its seed",
      close(r0[0].adv, 1 - 4) and close(r1[0].adv, 3 - 3) and close(r2[0].adv, 5 - 2),
      f"{r0[0].adv} {r1[0].adv} {r2[0].adv}")
check("a run counts once per cell: r0's two fight steps (mean 1.0) vs r1's 2.0",
      close(r0[1].adv, 0.5 - 2.0) and close(r0[2].adv, 1.5 - 2.0) and close(r1[1].adv, 2.0 - 1.0))
check("a cell no other play reached is left alone (rest@3)", close(r2[1].adv, 7.0))
check("a seed played once, and a run without a group, are untouched",
      close(r3[0].adv, 9.0) and close(r4[0].adv, 4.0))

print("\n== history baseline (earlier plays of the same seed x character)")
hb = P.HistoryBaseline(max_plays=2)
def play(adv_travel, adv_fights, group=3):
    st = [NS2(floor=1, kind="travel", adv=adv_travel)] + [NS2(floor=2, kind="combat", adv=x) for x in adv_fights]
    return st, [{"n": len(st), "group": group}]
s1, r1_ = play(1.0, [0.5, 1.5])
hb.apply(s1, r1_, block=0)
check("first play of a seed: nothing to compare with, untouched",
      close(s1[0].adv, 1.0) and close(s1[1].adv, 0.5))
s2, r2_ = play(3.0, [2.0])
hb.apply(s2, r2_, block=0)
check("second play: minus the first play's cell means (a run counts once: fights mean 1.0)",
      close(s2[0].adv, 3.0 - 1.0) and close(s2[1].adv, 2.0 - 1.0))
s3, r3_ = play(5.0, [0.0])
hb.apply(s3, r3_, block=0)
check("third: minus the mean of BOTH earlier plays, raw (not their adjusted values)",
      close(s3[0].adv, 5.0 - 2.0) and close(s3[1].adv, 0.0 - 1.5))
s4, r4_ = play(7.0, [0.0])
hb.apply(s4, r4_, block=0)
check("history keeps the last max_plays plays", close(s4[0].adv, 7.0 - 4.0))
s5, r5_ = play(9.0, [0.0], group=4)
hb.apply(s5, r5_, block=0)
check("another seed x character group has its own history", close(s5[0].adv, 9.0))
s6, r6_ = play(9.0, [0.0])
hb.apply(s6, r6_, block=1)
check("a new block starts with no history", close(s6[0].adv, 9.0))

print("\n== per-decision trace within a floor (lam_step)")
# A 3-decision fight on floor 2 (fight potentials 0.30 -> 0.50 -> 0.20), won, then a run
# decision on floor 3 with potential 0.60, then the run is lost there. lam (per floor) = 0.8.
def fight3(lam_step):
    s = [mk("combat", 2, 0.0, 0.30), mk("combat", 2, 0.0, 0.50), mk("combat", 2, 0.0, 0.20),
         mk("travel", 3, 0.0, 0.60)]
    P.compute_targets(s, False, 1.0, 0.8, bootstrap=False, lam_step=lam_step)
    return [x.adv for x in s]
d1 = fight3(1.0)
s_old = [mk("combat", 2, 0.0, 0.30), mk("combat", 2, 0.0, 0.50), mk("combat", 2, 0.0, 0.20),
         mk("travel", 3, 0.0, 0.60)]
P.compute_targets(s_old, False, 1.0, 0.8, bootstrap=False)
check("lam_step=1 is the previous credit exactly", all(close(a, b.adv) for a, b in zip(d1, s_old)))
d0 = fight3(0.0)
check("lam_step=0: each fight decision is credited by the potential change it caused (TD(0))",
      close(d0[0], 0.50 - 0.30) and close(d0[1], 0.20 - 0.50), f"{d0[0]:+.3f} {d0[1]:+.3f}")
check("...the fight's last decision still bootstraps across the floor with lam (0.8)",
      close(d0[2], (0.60 - 0.20) + 0.8 * (0.0 - 0.60)), f"{d0[2]:+.3f}")
dh = fight3(0.5)
check("lam_step=0.5 mixes: A_t = delta_t + 0.5 * A_t+1 within the floor",
      close(dh[0], 0.20 + 0.5 * dh[1]) and close(dh[1], -0.30 + 0.5 * dh[2]))
buf_ = P.RunBuffer(lam=0.8, lam_step=0.0)
check("RunBuffer passes lam_step through", buf_.lam_step == 0.0)

print("\n== M9: search policy pi' and its distillation")
from alphaslayer.unified import search as SR
lp_ = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
q_ = [0.40, 0.42, -1e9, -1e9]
t_hot = SR.improved(lp_, q_, 1e-4)
t_cold = SR.improved(lp_, q_, 1e4)
check("pi' is 0 on choices the search did not simulate", t_hot[2] == 0 and t_hot[3] == 0 and t_cold[2] == 0)
check("tau -> 0: pi' is the best simulated choice", t_hot.argmax() == 1 and t_hot[1] > 0.999)
check("tau -> inf: pi' is the prior renormalised over the simulated choices",
      close(float(t_cold[0]), 0.5 / 0.8, 1e-3) and close(float(t_cold[1]), 0.3 / 0.8, 1e-3))
check("nothing simulated -> no target", SR.improved(lp_, [-1e9] * 4, 0.01) is None)

logits_ = out["logits"].detach().clone().requires_grad_(True)
out_d = dict(out, logits=logits_)
is_c = (tgt["kind"] == P.COMBAT)
srch = is_c.clone()
srch[torch.nonzero(is_c)[0]] = False                 # one combat row stays unsearched
sp = torch.zeros_like(out["logits"])
for j in torch.nonzero(srch).flatten().tolist():
    n_ = int(b["cand_mask"][j].sum())
    w_ = torch.rand(n_, dtype=torch.float64)
    sp[j, :n_] = w_ / w_.sum()
tgt_d = dict(tgt, searched=srch, search_p=sp, adv=torch.zeros(24, dtype=torch.float64))
tot0, _ = P.ppo_loss(out_d, tgt_d, distill_coef=0.0)
tot1, pd = P.ppo_loss(out_d, tgt_d, distill_coef=1.0)
lp_all_ = torch.log_softmax(out["logits"], -1).masked_fill(~b["cand_mask"], 0.0)
ce_ = float((-(sp * lp_all_).sum(-1))[srch].mean())
check("distill = mean CE(pi', pi) over the searched rows", close(float(pd["distill"]), ce_, 1e-9),
      f"{float(pd['distill']):.6f} vs {ce_:.6f}")
check("distill_coef adds exactly coef * distill", close(float(tot1 - tot0), float(pd["distill"]), 1e-9))
g_ = torch.autograd.grad(pd["distill"], logits_)[0]
p_ = torch.softmax(out["logits"], -1)
exp_g = ((p_ - sp) * srch.unsqueeze(1)).masked_fill(~b["cand_mask"], 0.0) / int(srch.sum())
check("its gradient is (pi - pi') / n on searched rows and 0 elsewhere",
      float((g_.masked_fill(~b["cand_mask"], 0.0) - exp_g).abs().max()) < 1e-9 and
      float(g_[~srch].masked_fill(~b["cand_mask"][~srch], 0.0).abs().max()) == 0.0)

print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
