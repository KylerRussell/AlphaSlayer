"""Run critic tests (alphaslayer/unified/critic.py and train_unified.run_critic_potentials).

  * the critic learns a planted effect: the act-3 boss decides the win
  * warmup: until --critic-min-iters iterations of replay exist, the hand potentials stay
  * after warmup, a run decision's potential is the critic's P(win), and a fight decision's is
    P(win this fight) * that
  * NO LEAKAGE: an iteration's potentials come from the critic as it stood BEFORE that
    iteration's outcomes were added (a critic fit on a run's own outcome grades it against itself)

    PYTHONPATH=. .venv7/bin/python tests/test_run_critic.py
"""
from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from alphaslayer.unified import ppo as P
from alphaslayer.unified.critic import RunCritic
from train_unified import run_critic_potentials

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


rng = np.random.default_rng(0)


def make_iteration(n_runs=60):
    """Runs of 6 steps (floors 40..45, act 3): travel, 4 combat, card_reward. Boss 73 wins 50%,
    boss 1 wins 2%. Returns steps, runs, packed-like arrays."""
    steps, runs, ids, fs = [], [], [], []
    for r in range(n_runs):
        boss = 73 if r % 2 else 1
        won = rng.random() < (0.5 if boss == 73 else 0.02)
        kinds = ["travel"] + ["combat"] * 4 + ["card_reward"]
        rs = []
        for j, k in enumerate(kinds):
            s = P.Step(kind=k, obs={}, legal=[{}], a=0, logp=0.0, v=0.0, loop=1, floor=40 + j,
                       act=2, phi=0.123, p_fight=0.8 if k == "combat" else None)
            s.won = int(won)
            rs.append(s)
            row = np.zeros(7, np.int64); row[1] = 1; row[3] = boss
            ids.append(row); fs.append(rng.normal(0, 1, 24).astype(np.float32))
        P.compute_targets(rs, won, 1.0, 0.8, bootstrap=False)
        steps += rs
        runs.append({"won": int(won), "act": 2, "floors": 46, "n": len(rs)})
    return steps, runs, SimpleNamespace(a={"ctx_ids": np.array(ids), "ctx_f": np.array(fs)})


args = SimpleNamespace(w_fight=1.0, gamma=1.0, lam=0.8, lam_step=1.0, critic_min_iters=3)

print("== learns the planted boss effect")
c = RunCritic(window=20)
for _ in range(6):
    st, rn, pk = make_iteration()
    d, k = c.features(pk.a["ctx_ids"], pk.a["ctx_f"], np.array([s.act for s in st]),
                      np.array([s.floor for s in st]))
    c.add(d, k, np.array([s.won for s in st], np.float32), np.array([s.kind == "combat" for s in st]))
    c.fit()
st, rn, pk = make_iteration()
d, k = c.features(pk.a["ctx_ids"], pk.a["ctx_f"], np.array([s.act for s in st]), np.array([s.floor for s in st]))
p = c.predict(d, k)
hi, lo = p[pk.a["ctx_ids"][:, 3] == 73].mean(), p[pk.a["ctx_ids"][:, 3] == 1].mean()
check("P(win | beatable boss) >> P(win | hard boss)", hi > 0.3 and lo < 0.1, f"{hi:.3f} vs {lo:.3f}")

print("\n== warmup keeps the hand potentials")
c2 = RunCritic(window=20)
st, rn, pk = make_iteration()
line = run_critic_potentials(c2, pk, st, rn, args, bootstrap=False)
check("first iteration: phi untouched, log says warmup", all(s.phi == 0.123 for s in st) and "warmup" in line, line)
check("...and the iteration joined the replay", len(c2.buf) == 1)

print("\n== after warmup: learned potentials, no leakage")
for _ in range(3):
    st, rn, pk = make_iteration()
    run_critic_potentials(c2, pk, st, rn, args, bootstrap=False)
st, rn, pk = make_iteration()
before = copy.deepcopy(c2)                      # the critic as it stands before this iteration
line = run_critic_potentials(c2, pk, st, rn, args, bootstrap=False)
d, k = before.features(pk.a["ctx_ids"], pk.a["ctx_f"], np.array([s.act for s in st]),
                       np.array([s.floor for s in st]))
v = before.predict(d, k)
run_ok = all(abs(s.phi - v[i]) < 1e-6 for i, s in enumerate(st) if s.kind != "combat")
from alphaslayer.unified.critic import fight_run_state, _PERSIST
is_c = np.array([s.kind == "combat" for s in st])
d2, k2 = fight_run_state(d, k, is_c, np.repeat(np.arange(len(rn)), 6))
vf = before.predict(d2, k2)
fight_ok = all(abs(s.phi - 0.8 * vf[i]) < 1e-6 for i, s in enumerate(st) if s.kind == "combat")
check("run decisions: phi = P(win) from the critic BEFORE this iteration was added", run_ok)
check("fight decisions: phi = P(win this fight) * P(win of the run state 'if the fight ended now')",
      fight_ok)
# each run is travel, 4 combat, card_reward: its fight is entered from the travel decision
other = [j for j in range(d.shape[1]) if j not in _PERSIST]
check("fight_run_state: entry decision's features, except the fight's own hp/max hp/hp%/potions",
      np.allclose(d2[1][other], d[0][other]) and np.allclose(d2[1][_PERSIST], d[1][_PERSIST])
      and k2[1] == k[0] and not np.allclose(d2[1], d[1]))
dd = np.arange(7 * 40, dtype=np.float32).reshape(7, 40)
dk, kk = fight_run_state(dd, np.arange(7), np.array([1, 0, 1, 1, 0, 0, 1], bool),
                         np.array([0, 0, 0, 0, 0, 1, 1]))
check("...never reaching across runs, and a fight with no entry decision keeps its own state",
      np.array_equal(kk, [0, 1, 1, 1, 4, 5, 5]) and np.allclose(dk[0], dd[0]))
check("log reports out-of-sample skill", line.startswith("rc_skill="), line)
after_v = c2.predict(d, k)
check("the refit critic differs (so the check above can tell them apart)",
      np.abs(after_v - v).max() > 1e-4, f"{np.abs(after_v - v).max():.2e}")
st0 = st[:6]
exp = copy.deepcopy(st0)
P.compute_targets(exp, bool(rn[0]["won"]), 1.0, 0.8, bootstrap=False)
check("advantages were recomputed from the new potentials",
      all(abs(a_.adv - b_.adv) < 1e-9 for a_, b_ in zip(st0, exp)))

print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
