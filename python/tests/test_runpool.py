"""RunPool tests against a FAKE game that speaks the run-queue protocol (no game, no GPU).

  * every job is played exactly once, with ITS seed and character, despite a crash and a stall
  * a failed slot's job is replayed from the start on a fresh process (same seed), and its
    partial run is dropped (on_restart) so no trajectory mixes processes or runs
  * processes PERSIST across run() calls: healthy slots keep their pid between iterations
  * recycling: a process quits after recycle_runs runs and its slot is relaunched
  * the pool spreads jobs over slots (a fast slot takes more than its 1/n share)

    PYTHONPATH=. .venv7/bin/python tests/test_runpool.py
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

from alphaslayer.runenv import RunPool

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


# Plan per slot, for the slot's FIRST process only: "crash1" (SIGSEGV one decision into its
# 2nd run), "stall0" (hangs one decision into its 1st run), "slow" (sleeps per decision).
FAKE = r'''#!/usr/bin/env python3
import json, os, re, signal, socket, sys, time
arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
assert "--probe-run-queue" in sys.argv
out = arg("probe-out"); idx = int(re.search(r"(\d+)$", out).group(1))
plan = json.load(open(os.environ["FAKE_SPIRE_PLAN"]))
marker = os.path.join(os.environ["FAKE_SPIRE_STATE"], f"launched{idx}")
first = not os.path.exists(marker)
open(marker, "a").write(f"{os.getpid()}\n")
mode = plan.get(str(idx), "ok") if first else "ok"
s = socket.create_connection(("127.0.0.1", int(arg("probe-port"))))
f = s.makefile("rwb")
def send(m): f.write((json.dumps(m) + "\n").encode()); f.flush()
send({"t": "hello", "mode": "run"})
k = 0
while True:
    send({"t": "next_run"})
    job = json.loads(f.readline())
    if job.get("quit"):
        send({"t": "done"}); break
    for d in range(3):
        if mode == "slow": time.sleep(0.05)
        send({"t": "decision", "kind": "combat" if d else "card_reward",
              "obs": {"pid": os.getpid(), "run": job["run"], "seed": job["seed"],
                      "character": job["character"], "d": d}, "legal": [{}, {}]})
        json.loads(f.readline())
        if d == 0 and mode == "crash1" and k == 1: os.kill(os.getpid(), signal.SIGSEGV)
        if d == 0 and mode == "stall0" and k == 0: time.sleep(3600)
    send({"t": "terminal", "run": job["run"], "outcome": "died", "won": False, "floors": 3,
          "act": 0, "acts": 3, "hp": 0, "max_hp": 80, "gold": 99, "deck": 10, "relics": 1,
          "fights": 1, "fight_wins": 0})
    k += 1
'''


def make_jobs(it, n):
    chars = ["IRONCLAD", "SILENT", "DEFECT"]
    return [{"run": it * 1000 + j, "seed": f"S{it}_{j}", "character": chars[j % 3]} for j in range(n)]


def play(pool, jobs):
    open_, done, restarted = collections.defaultdict(list), [], []

    def policy(kinds, obs, legal, idxs):
        for o, i in zip(obs, idxs):
            open_[i].append(o)
        return [0] * len(obs)

    def on_terminal(i, rr):
        done.append((i, rr.run, open_.pop(i, [])))

    def on_restart(i):
        restarted.append(i)
        open_.pop(i, None)

    pool.run(jobs, policy, on_terminal=on_terminal, on_restart=on_restart, deadline_s=60)
    return done, restarted


def clean(done, jobs):
    by_run = {j["run"]: j for j in jobs}
    bad = []
    for i, run, steps in done:
        j = by_run[run]
        if (len(steps) != 3 or len({o["pid"] for o in steps}) != 1
                or any(o["seed"] != j["seed"] or o["character"] != j["character"] for o in steps)):
            bad.append((i, run, steps[:1]))
    return bad


d = tempfile.mkdtemp(prefix="fake_pool_")
exe = os.path.join(d, "SlayTheSpire2")
open(exe, "w").write(FAKE)
os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC)
os.makedirs(os.path.join(d, "state"))
json.dump({"0": "crash1", "1": "stall0", "2": "slow"}, open(os.path.join(d, "plan.json"), "w"))
os.environ["FAKE_SPIRE_PLAN"] = os.path.join(d, "plan.json")
os.environ["FAKE_SPIRE_STATE"] = os.path.join(d, "state")
RunPool.ENV_STALL_TIMEOUT = 3.0
try:
    with RunPool(n_envs=4, game_dir=d, out_dir=os.path.join(d, "out"),
                 stall_dir=os.path.join(d, "stalls"), recycle_runs=100) as pool:
        print("iteration 1: 12 jobs on 4 slots; slot 0 crashes in its 2nd run, slot 1 stalls")
        jobs = make_jobs(1, 12)
        done, restarted = play(pool, jobs)
        runs = sorted(r for _, r, _ in done)
        check("all 12 jobs completed exactly once", runs == sorted(j["run"] for j in jobs), runs)
        check("one crash, one stall, two relaunches",
              (pool.stats["crashed"], pool.stats["stalled"], pool.stats["relaunched"]) == (1, 1, 2),
              pool.stats)
        check("on_restart fired for the failed slots", sorted(restarted) == [0, 1], restarted)
        bad = clean(done, jobs)
        check("every trajectory: one process, its job's seed and character, 3 steps", not bad, bad[:2])
        per_slot = collections.Counter(i for i, _, _ in done)
        check("the slow slot took fewer runs than the others (no fixed share)",
              per_slot[2] < max(per_slot[i] for i in (0, 1, 3)), dict(per_slot))
        pids1 = {i: pool.procs[i].pid for i in range(4)}

        print("iteration 2: same pool, 8 more jobs")
        jobs2 = make_jobs(2, 8)
        done2, restarted2 = play(pool, jobs2)
        check("all 8 completed exactly once", sorted(r for _, r, _ in done2) == sorted(j["run"] for j in jobs2))
        check("no relaunch, and every slot kept its process across iterations",
              pool.stats["relaunched"] == 0 and all(pool.procs[i].pid == pids1[i] for i in range(4)),
              f"{pool.stats} {pids1} -> {[p.pid for p in pool.procs]}")
        check("clean trajectories", not clean(done2, jobs2))
        launches = pool.launches

    print("recycling: recycle_runs=2")
    json.dump({}, open(os.path.join(d, "plan.json"), "w"))
    with RunPool(n_envs=3, game_dir=d, out_dir=os.path.join(d, "out2"),
                 stall_dir=os.path.join(d, "stalls"), recycle_runs=2) as pool:
        jobs3 = make_jobs(3, 12)
        done3, _ = play(pool, jobs3)
        check("all 12 completed with processes recycled every 2 runs",
              sorted(r for _, r, _ in done3) == sorted(j["run"] for j in jobs3)
              and pool.stats["recycled"] >= 3 and pool.stats["crashed"] == 0, pool.stats)
        check("clean trajectories across recycles", not clean(done3, jobs3))
        # results carry only the run id: duplicate ids would make results unattributable
        # (the eval once reused one id for 24 different seeds)
        dup = make_jobs(4, 2)
        dup[1]["run"] = dup[0]["run"]
        try:
            pool.run(dup, lambda kinds, obs, legal, idxs: [0] * len(kinds))
            check("duplicate run ids refused", False, "no error")
        except ValueError as e:
            check("duplicate run ids refused", "duplicate run id" in str(e), str(e))
finally:
    shutil.rmtree(d, ignore_errors=True)

print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
