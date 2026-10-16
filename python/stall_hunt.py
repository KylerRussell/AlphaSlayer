"""Catch a wedged game in the act: managed thread stacks + pending async chains, before the kill.

Plays real runs with a unified checkpoint on the characters that stall (Necrobinder, Silent).
When VecRunEnv declares an env stalled, this records what the env last SENT and what we last
ANSWERED, then takes a dotnet-dump of the still-running process and extracts
clrthreads / clrstack -all / dumpasync into <out>/stall-<pid>.txt (the dump itself is deleted).

    PYTHONPATH=. .venv7/bin/python stall_hunt.py --ckpt unified_m4_long2.pt --runs-per-env 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import torch

from alphaslayer.runenv import VecRunEnv
from capability_eval import make_unified

DOTNET_DUMP = os.path.expanduser("~/.dotnet/tools/dotnet-dump")


def capture(pid, out_dir, tag, keep=False):
    """dotnet-dump the live process, extract the stacks, delete the dump. Returns the txt path."""
    dmp = os.path.join(out_dir, f"{tag}.dmp")
    txt = os.path.join(out_dir, f"{tag}.txt")
    t0 = time.time()
    r = subprocess.run([DOTNET_DUMP, "collect", "-p", str(pid), "--type", "Heap", "-o", dmp],
                       capture_output=True, text=True, timeout=300)
    with open(txt, "w") as fh:
        fh.write(f"collect rc={r.returncode} {time.time() - t0:.1f}s\n{r.stdout}\n{r.stderr}\n")
        if os.path.exists(dmp):
            a = subprocess.run([DOTNET_DUMP, "analyze", dmp, "-c", "clrthreads",
                                "-c", "clrstack -all", "-c", "dumpasync", "-c", "exit"],
                               capture_output=True, text=True, timeout=900)
            fh.write(a.stdout + "\n" + a.stderr)
            if not keep:
                os.remove(dmp)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="unified_m4_long2.pt")
    ap.add_argument("--chars", default="NECROBINDER,SILENT")
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--runs-per-env", type=int, default=8)
    ap.add_argument("--seed", default="L1_")
    ap.add_argument("--offset", type=int, default=90000)
    ap.add_argument("--out", default="/tmp/alphaslayer_stallhunt")
    ap.add_argument("--fixed-seed", default=None,
                    help="every env plays this probe seed (a replay of one stalled run, with "
                         "the policy's sampling varying across envs)")
    ap.add_argument("--keep-dumps", action="store_true")
    a = ap.parse_args()
    if a.fixed_seed:
        real = subprocess.Popen
        def popen(args, *k, **kw):
            if isinstance(args, list):
                args = [f"--probe-seed={a.fixed_seed}" if str(x).startswith("--probe-seed=") else x
                        for x in args]
            return real(args, *k, **kw)
        import alphaslayer.runenv as RE
        RE.subprocess.Popen = popen
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    policy = make_unified(a.ckpt, torch.device("cuda:0"), True)
    last = {}            # env -> (what it last sent, when, what we answered)

    class Hunt(VecRunEnv):
        def _dispatch(self, conn, msg, on_fight, on_terminal):
            t = msg.get("t")
            info = t if t != "decision" else f"decision {msg.get('kind')} ({len(msg.get('legal', []))} legal)"
            if t == "terminal":
                info += f" run {msg.get('run')} {msg.get('outcome')}"
            last[conn.idx] = [info, time.time(), None]
            return super()._dispatch(conn, msg, on_fight, on_terminal)

        def _on_stall(self, conn):
            super()._on_stall(conn)
            sent, when, ans = last.get(conn.idx, ["nothing", time.time(), None])
            pid = self.procs[conn.idx].pid
            tag = f"stall-{pid}-env{conn.idx}"
            print(f"  HUNT env {conn.idx} pid {pid}: last sent '{sent}' {time.time() - when:.0f}s ago, "
                  f"our answer {ans}; runs done {self._runs_by_env.get(conn.idx, 0)}", flush=True)
            txt = capture(pid, a.out, tag, a.keep_dumps)
            print(f"  HUNT stacks -> {txt}", flush=True)

    def wrapped(kinds, obs, legal, idxs):
        acts, _ = policy(kinds, obs, legal, idxs)
        for i, act in zip(idxs, acts):
            if i in last:
                last[i][2] = act
        return acts

    t0 = time.time()
    with Hunt(n_envs=a.envs, characters=a.chars.split(","), runs_per_env=a.runs_per_env,
              seed=a.seed, ascension=0, act_cap=3, run_offset=a.offset, restarts=2) as venv:
        venv.run(wrapped, on_restart=lambda i: None, deadline_s=7200)
        print(f"runs={len(venv.results)}/{a.envs * a.runs_per_env} crash/stall/relaunch="
              f"{venv.crashed}/{venv.stalled}/{venv.restarted} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
