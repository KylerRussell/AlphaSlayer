"""VecRunEnv relaunch tests, against a FAKE game process (no game, no GPU, ~20s).

The real game crashes (SIGSEGV in its worker threads) or wedges in a process's first run, a few
percent of the time. These tests drive VecRunEnv against a stand-in executable that speaks the
runserve protocol and crashes or hangs on demand, and check what the trainer depends on:

  * every run asked for is DELIVERED (envs x runs_per_env terminals), despite a crash and a stall
  * no completed run is replayed: each env's run ids are exactly offset..offset+runs-1
  * a relaunch after k completed runs plays only the runs still owed
  * no recorded trajectory mixes steps from two processes or two runs (on_restart drops the
    dead process's unfinished run before the new one starts)
  * restarts=0 keeps the old behaviour (the crashed env's runs are lost), and restarts without
    an on_restart hook is refused

    PYTHONPATH=. .venv7/bin/python tests/test_runenv_restart.py
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import stat
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaslayer.runenv import VecRunEnv

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


# The stand-in game. Per env (taken from --probe-out's trailing index) the plan says what its
# FIRST launch does: "ok", "crash0" (SIGSEGV one decision into run 0), "crash1" (finishes run 0,
# SIGSEGV one decision into run 1) or "stall0" (hangs one decision into run 0). Later launches
# of the same env always behave.
FAKE = r'''#!/usr/bin/env python3
import json, os, re, signal, socket, sys, time
arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
out = arg("probe-out"); idx = int(re.search(r"(\d+)$", out).group(1))
os.makedirs(out, exist_ok=True)
log = open(os.path.join(out, "probe.log"), "w")
plan = json.load(open(os.environ["FAKE_SPIRE_PLAN"]))
marker = os.path.join(os.environ["FAKE_SPIRE_STATE"], f"launched{idx}")
first = not os.path.exists(marker)
open(marker, "a").write(f"{os.getpid()} runs={arg('probe-runs')} off={arg('probe-run-offset')}\n")
mode = plan.get(str(idx), "ok") if first else "ok"
s = socket.create_connection(("127.0.0.1", int(arg("probe-port"))))
f = s.makefile("rwb")
def send(m): f.write((json.dumps(m) + "\n").encode()); f.flush()
send({"t": "hello", "mode": "run"})
runs, off = int(arg("probe-runs")), int(arg("probe-run-offset"))
for r in range(runs):
    run = r + off
    log.write(f"run {run} start\n"); log.flush()
    for d in range(3):
        send({"t": "decision", "kind": "combat" if d else "card_reward",
              "obs": {"pid": os.getpid(), "run": run, "d": d}, "legal": [{}, {}]})
        json.loads(f.readline())
        if d == 0 and ((mode == "crash0" and r == 0) or (mode == "crash1" and r == 1)):
            os.kill(os.getpid(), signal.SIGSEGV)
        if d == 0 and mode == "stall0" and r == 0:
            time.sleep(3600)
    send({"t": "terminal", "run": run, "outcome": "died", "won": False, "floors": 3, "act": 0,
          "acts": 3, "hp": 0, "max_hp": 80, "gold": 99, "deck": 10, "relics": 1, "fights": 1,
          "fight_wins": 0})
send({"t": "done"})
'''


def setup(plan):
    d = tempfile.mkdtemp(prefix="fake_spire_")
    exe = os.path.join(d, "SlayTheSpire2")
    with open(exe, "w") as fh:
        fh.write(FAKE)
    os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC)
    state = os.path.join(d, "state")
    os.makedirs(state)
    with open(os.path.join(d, "plan.json"), "w") as fh:
        json.dump(plan, fh)
    os.environ["FAKE_SPIRE_PLAN"] = os.path.join(d, "plan.json")
    os.environ["FAKE_SPIRE_STATE"] = state
    return d


def collect(plan, n_envs, runs, offset, restarts, hook=True):
    """Runs the env and records trajectories the way the trainer's RunBuffer does."""
    d = setup(plan)
    open_, done, restarted_envs = collections.defaultdict(list), [], []

    def policy(kinds, obs, legal, idxs):
        for o, i in zip(obs, idxs):
            open_[i].append(o)
        return [0] * len(obs)

    def on_terminal(i, rr):
        done.append((i, rr.run, open_.pop(i, [])))

    def on_restart(i):
        restarted_envs.append(i)
        open_.pop(i, None)

    VecRunEnv.ENV_STALL_TIMEOUT, old = 3.0, VecRunEnv.ENV_STALL_TIMEOUT
    try:
        with VecRunEnv(n_envs=n_envs, characters=["IRONCLAD"], runs_per_env=runs, game_dir=d,
                       run_offset=offset, out_dir=os.path.join(d, "out"),
                       stall_dir=os.path.join(d, "stalls"), restarts=restarts) as venv:
            venv.run(policy, on_terminal=on_terminal, deadline_s=60,
                     on_restart=on_restart if hook else None)
            stats = (venv.crashed, venv.stalled, venv.restarted)
    finally:
        VecRunEnv.ENV_STALL_TIMEOUT = old
    launches = {f: open(os.path.join(d, "state", f)).read().split("\n")
                for f in os.listdir(os.path.join(d, "state"))}
    shutil.rmtree(d, ignore_errors=True)
    return done, stats, restarted_envs, launches


def trajectories_clean(done):
    """Every run's steps come from one process and carry that run's id, three steps each."""
    bad = []
    for i, run, steps in done:
        pids = {o["pid"] for o in steps}
        if len(steps) != 3 or len(pids) != 1 or any(o["run"] != run for o in steps):
            bad.append((i, run, [(o["pid"], o["run"], o["d"]) for o in steps]))
    return bad


print("crash in run 0, stall in run 0, healthy env; restarts=2")
done, (crashed, stalled, restarted), renv, d = collect(
    {"0": "crash0", "1": "stall0"}, n_envs=3, runs=3, offset=500, restarts=2)
check("all 9 runs delivered", len(done) == 9, f"{len(done)}")
ids = collections.defaultdict(list)
for i, run, _ in done:
    ids[i].append(run)
check("each env's run ids are exactly 500..502, none replayed",
      all(sorted(ids[i]) == [500, 501, 502] for i in range(3)), dict(ids))
check("one crash, one stall, two relaunches", (crashed, stalled, restarted) == (1, 1, 2),
      f"{crashed}/{stalled}/{restarted}")
check("on_restart fired for exactly the failed envs", sorted(renv) == [0, 1], renv)
bad = trajectories_clean(done)
check("no trajectory mixes processes or runs", not bad, bad[:2])

print("crash in run 1, after one completed run")
done, (crashed, stalled, restarted), renv, launch = collect(
    {"0": "crash1"}, n_envs=1, runs=3, offset=700, restarts=2)
launches = launch["launched0"]
check("relaunch plays only the 2 runs still owed, from run 701",
      "runs=2 off=701" in launches[1], launches[:2])
check("run ids 700..702, none replayed", sorted(r for _, r, _ in done) == [700, 701, 702],
      [r for _, r, _ in done])
check("no trajectory mixes processes or runs", not trajectories_clean(done))

print("restarts=0 keeps the old behaviour")
done, (crashed, stalled, restarted), renv, d = collect(
    {"0": "crash0"}, n_envs=2, runs=2, offset=0, restarts=0)
check("the crashed env's runs are lost, nothing relaunched",
      len(done) == 2 and restarted == 0 and crashed == 1, f"{len(done)} runs, {restarted}")

print("restarts without an on_restart hook is refused")
try:
    collect({}, n_envs=1, runs=1, offset=0, restarts=1, hook=False)
    check("ValueError raised", False, "no error")
except ValueError:
    check("ValueError raised", True)

print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
