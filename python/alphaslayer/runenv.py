"""Vectorised WHOLE-RUN env: N game processes, one policy, batched inference.

Same shape as VecSpireEnv, and for the same reason: every decision is a TCP round trip plus a
GPU forward, both of which only amortise across environments. The differences are the protocol
(``--probe=runserve``) and that a decision now carries a ``kind``, so the caller can route
combat decisions to the combat policy and everything else to the run policy.

    def policy(kinds, obs_list, legal_list, env_idxs) -> list[int]: ...
    with VecRunEnv(n_envs=8, characters=["IRONCLAD"], runs_per_env=4) as venv:
        venv.run(policy, on_fight=..., on_terminal=...)
"""

from __future__ import annotations

import collections
import itertools
import json
import os
import selectors
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass

from .env import DEFAULT_GAME
from .vecenv import _Conn, accept_per_env, game_preexec, listen_per_env


@dataclass
class RunResult:
    run: int
    outcome: str
    won: bool
    floors: int
    act: int
    acts: int
    hp: int
    max_hp: int
    gold: int
    deck: int
    relics: int
    fights: int
    fight_wins: int


@dataclass
class FightResult:
    """One combat inside a run. The combat policy is credited from THIS, not the run outcome."""
    won: bool
    outcome: str
    room: str
    encounter: str
    turns: int
    hp_start: int
    hp_end: int
    max_hp: int
    # 0-based act the fight happened in; -1 from a probe DLL that predates the field.
    act: int = -1


class VecRunEnv:
    # Global silence means every remaining env is stuck.
    STALL_TIMEOUT = 180.0
    # Per-env silence. The protocol is strict request/response: an env that is alive either
    # has a decision outstanding (which we answer within the batch timeout) or is between
    # runs, which takes a couple of hundred milliseconds. Forty seconds of silence from one
    # env while others are talking is a wedged process, not a slow one. It was 90s, and
    # those waits were 30% of all collection wall-clock in a 321-iteration run.
    #
    # Worth doing per-env rather than globally: the global check can only fire once every
    # other env has finished, so it both waits far longer than necessary and cannot say WHICH
    # env died.
    ENV_STALL_TIMEOUT = 40.0

    def __init__(self, n_envs=8, characters=None, character="IRONCLAD", runs_per_env=4,
                 seed="ALPHASLAYER", ascension=0, game_dir=DEFAULT_GAME, fixed_fps=5000,
                 turn_cap=50, floor_cap=200, act_cap=0, quiet=True, batch_timeout=0.004,
                 run_offset=0, out_dir="/tmp/alphaslayer_runserve",
                 stall_dir="/tmp/alphaslayer_stalls", close_on_exit=False, expect_sizes=None,
                 restarts=0, restart_connect_timeout=120.0):
        self.expect_sizes = expect_sizes or {}
        self._sizes_checked = False
        self.n_envs = n_envs
        self.batch_timeout = batch_timeout
        self.results: list[RunResult] = []
        self.fights: list[FightResult] = []
        self.sizes = {}
        self.steps = 0
        self.batches = 0
        self.batch_sizes: list[int] = []
        self.kind_counts: dict[str, int] = {}
        self.stalled = 0
        self.stalled_envs: list[int] = []
        # Envs whose socket closed before they sent "done": a crash or an early exit.
        self.crashed = 0
        self.crashed_envs: list[tuple[int, int | None]] = []
        self._runs_by_env: dict[int, int] = {}
        # Relaunching an env that crashed or wedged. All 776 recorded crashes happened in a
        # process's FIRST run, within a second of the run starting: a race in the game's own
        # worker threads, which we cannot fix. Stalls can also come mid-run from a probe bug
        # (python/stall_hunt.py captures the wedged process's async stacks). Without a relaunch
        # each failure silently costs that env's whole batch of runs. A relaunch plays
        # only the runs still owed (--probe-run-offset continues the env's seed sequence, so no
        # completed run is replayed). Opt-in: the caller must pass run(on_restart=...) to drop
        # the dead process's partial run, or its steps would be glued onto the new one's.
        self.max_restarts = restarts
        self.restart_connect_timeout = restart_connect_timeout
        self.restarted = 0
        self._restarts: dict[int, int] = {}
        self._awaiting: dict[int, float] = {}     # env idx -> reconnect deadline
        self._last_seen: dict[int, float] = {}
        self.runs_per_env = runs_per_env
        self._run_offset = run_offset
        self._spawn_cfg = dict(game_dir=game_dir, fixed_fps=fixed_fps, seed=seed,
                               ascension=ascension, turn_cap=turn_cap, floor_cap=floor_cap,
                               act_cap=act_cap, quiet=quiet)
        self._out_dir = out_dir
        self._stall_dir = stall_dir
        self.close_on_exit = close_on_exit

        self._srvs = listen_per_env(n_envs)

        roster = characters or [character]
        self.characters = [roster[i % len(roster)] for i in range(n_envs)]
        self.procs = [self._spawn(i, runs_per_env, run_offset) for i in range(n_envs)]

        accepted = accept_per_env(self._srvs, 120, "VecRunEnv")
        self.conns = [accepted[i] for i in sorted(accepted)]
        if not self.conns:
            raise RuntimeError("no game processes connected")

        self._sel = selectors.DefaultSelector()
        for c in self.conns:
            self._sel.register(c.sock, selectors.EVENT_READ, c)

    def _spawn(self, i, runs, offset):
        c = self._spawn_cfg
        env = dict(os.environ, SteamAppId="2868840", SteamGameId="2868840")
        args = [
            os.path.join(c["game_dir"], "SlayTheSpire2"), "--headless",
            "--fixed-fps", str(c["fixed_fps"]),
            "--probe=runserve", f"--probe-out={self._out_dir}{i}", "--probe-quit",
            f"--probe-port={self._srvs[i].getsockname()[1]}",
            f"--probe-runs={runs}",
            f"--probe-character={self.characters[i]}",
            f"--probe-seed={c['seed']}{i}",
            f"--probe-ascension={c['ascension']}",
            f"--probe-turn-cap={c['turn_cap']}",
            f"--probe-floor-cap={c['floor_cap']}",
            # Distinct seed sequences per env AND per iteration, so successive rollouts
            # do not replay the same maps.
            f"--probe-run-offset={offset}",
        ]
        if c["act_cap"]:
            args.append(f"--probe-act-cap={c['act_cap']}")
        return subprocess.Popen(
            args, cwd=c["game_dir"], env=env, preexec_fn=game_preexec,
            stdout=subprocess.DEVNULL if c["quiet"] else None,
            stderr=subprocess.DEVNULL if c["quiet"] else None)

    def _maybe_restart(self, conn, on_restart):
        """Relaunches a crashed or wedged env for the runs it still owes. True if relaunched."""
        i = conn.idx
        done = self._runs_by_env.get(i, 0)
        owed = self.runs_per_env - done
        if (owed <= 0 or self._restarts.get(i, 0) >= self.max_restarts
                or on_restart is None or self.timed_out):
            return False
        p = self.procs[i]
        if p.poll() is None:              # wedged, not dead: it must go before its successor
            p.kill()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        on_restart(i)
        self._restarts[i] = self._restarts.get(i, 0) + 1
        self.restarted += 1
        self.procs[i] = self._spawn(i, owed, self._run_offset + done)
        self._awaiting[i] = time.time() + self.restart_connect_timeout
        self._sel.register(self._srvs[i], selectors.EVENT_READ, ("listen", i))
        print(f"VecRunEnv: relaunched env {i} ({self.characters[i]}) for {owed} run(s) "
              f"(restart {self._restarts[i]}/{self.max_restarts})", flush=True)
        return True

    def _accept_restarted(self, i):
        try:
            sock, _ = self._srvs[i].accept()
        except BlockingIOError:
            return
        try:
            self._sel.unregister(self._srvs[i])
        except (KeyError, ValueError):
            pass
        self._awaiting.pop(i, None)
        c = _Conn(sock, i)
        self.conns = [c if old.idx == i else old for old in self.conns]
        self._sel.register(c.sock, selectors.EVENT_READ, c)
        self._last_seen[i] = time.time()

    def _expire_restarts(self):
        now = time.time()
        for i, dl in list(self._awaiting.items()):
            if now > dl:
                print(f"VecRunEnv: relaunched env {i} never connected; giving up on it",
                      flush=True)
                self._give_up_restart(i)

    def _give_up_restart(self, i):
        self._awaiting.pop(i, None)
        try:
            self._sel.unregister(self._srvs[i])
        except (KeyError, ValueError):
            pass
        if self.procs[i].poll() is None:
            self.procs[i].kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, policy, on_fight=None, on_terminal=None, max_steps=None, deadline_s=None,
            on_restart=None):
        """Drives the envs. ``policy(kinds, obs, legal, env_idxs) -> list[int]``.

        ``on_fight(env_idx, FightResult)`` fires at the end of each combat and
        ``on_terminal(env_idx, RunResult)`` at the end of each run. Two callbacks because the
        two policies are credited from different things: the combat policy from the fight it
        just played, the run policy from how the run ends.

        ``on_restart(env_idx)`` fires just before a crashed or wedged env is relaunched (only
        with ``restarts`` > 0): the caller must discard that env's unfinished run.
        """
        if self.max_restarts and on_restart is None:
            raise ValueError("restarts > 0 needs run(on_restart=...) to drop the dead "
                             "process's unfinished run")
        pending: list[tuple[_Conn, dict]] = []
        t0 = time.time()
        last_activity = time.time()
        self._last_seen = last_seen = {c.idx: time.time() for c in self.conns}

        hard_deadline = (t0 + deadline_s) if deadline_s else None
        self.timed_out = False
        while any(c.alive for c in self.conns) or self._awaiting:
            # Hard wall-clock cap on the whole pass. Nothing inside a rollout should be able
            # to consume hours: whatever the cause (a wedged env, a machine under memory
            # pressure), the trainer takes what it has and moves on rather than disappearing.
            if hard_deadline and time.time() > hard_deadline:
                live_conns = [c for c in self.conns if c.alive]
                print(f"VecRunEnv: collection deadline ({deadline_s:.0f}s) hit; "
                      f"stopping with {len(live_conns)} env(s) still running", flush=True)
                self.timed_out = True
                for c in live_conns:
                    self._unregister(c)
                for i in list(self._awaiting):
                    self._give_up_restart(i)
                break
            deadline = None
            while True:
                live = sum(1 for c in self.conns if c.alive)
                if (live == 0 and not self._awaiting) or (pending and len(pending) >= live):
                    break
                remaining = self.batch_timeout if deadline is None else deadline - time.time()
                if pending and remaining <= 0:
                    break
                events = self._sel.select(timeout=max(0.0005, min(remaining, 0.05)))
                if self._awaiting:
                    self._expire_restarts()
                if events:
                    last_activity = time.time()
                # OFF by default, and measured rather than assumed: closing on process exit
                # was meant to skip the 90s stall timer for envs that had finished cleanly,
                # but it races with data still in flight and cost 2 runs in 32 even with a
                # drain (32/32 with it off). Socket EOF is the only signal that actually means
                # "everything has been delivered", so that is what we close on. The pathology
                # it was guarding against is bounded by the collection deadline instead.
                if not pending and self.close_on_exit:
                    for c in [c for c in self.conns if c.alive]:
                        if self.procs[c.idx].poll() is not None:
                            # DRAIN before closing. A process that has exited has already
                            # handed its last messages to the kernel, and those include the
                            # final run terminals; closing on the exit alone threw them away
                            # and silently cost ~25% of every iteration's runs.
                            for msg in self._drain(c):
                                self._dispatch(c, msg, on_fight, on_terminal)
                            self._unregister(c)
                # Drop individual wedged envs while the rest keep running.
                if not pending:
                    now = time.time()
                    for c in [c for c in self.conns if c.alive]:
                        if now - last_seen.get(c.idx, now) > self.ENV_STALL_TIMEOUT:
                            self._on_stall(c)
                            self._unregister(c)
                            self._maybe_restart(c, on_restart)
                if not events and time.time() - last_activity > self.STALL_TIMEOUT and not pending:
                    live_conns = [c for c in self.conns if c.alive]
                    print(f"VecRunEnv: no traffic from any env for {self.STALL_TIMEOUT:.0f}s; "
                          f"dropping {len(live_conns)} stalled env(s)", flush=True)
                    for c in live_conns:
                        self._on_stall(c)
                        self._unregister(c)
                        self._maybe_restart(c, on_restart)
                    break
                for key, _ in events:
                    if isinstance(key.data, tuple):          # a relaunched env connecting
                        self._accept_restarted(key.data[1])
                        continue
                    conn: _Conn = key.data
                    if not conn.alive:
                        continue
                    last_seen[conn.idx] = time.time()
                    for msg in conn.read_messages():
                        d = self._dispatch(conn, msg, on_fight, on_terminal)
                        if d is not None:
                            pending.append(d)
                    if not conn.alive:
                        self._unregister(conn)
                        if not conn.done:
                            self._on_exit(conn)
                            self._maybe_restart(conn, on_restart)
                if pending and deadline is None:
                    deadline = time.time() + self.batch_timeout
                if (not events and not pending and not any(c.alive for c in self.conns)
                        and not self._awaiting):
                    break

            # A process can send a decision and die before we answer it. Answering is pointless,
            # and recording it would add a step to the run on_restart just discarded.
            pending = [(c, m) for c, m in pending if c.alive]
            if not pending:
                if not any(c.alive for c in self.conns) and not self._awaiting:
                    break
                continue

            kinds = [m["kind"] for _, m in pending]
            for k in kinds:
                self.kind_counts[k] = self.kind_counts.get(k, 0) + 1
            obs = [m["obs"] for _, m in pending]
            legal = [m["legal"] for _, m in pending]
            idxs = [c.idx for c, _ in pending]
            actions = policy(kinds, obs, legal, idxs)
            # The policy callback can legitimately take minutes -- it may run counterfactual
            # fights before answering. That is US being slow, not the envs being wedged, so
            # the stall clocks are reset after it returns. Without this the expert's own
            # evaluation time was counted against the run envs and they were dropped as
            # stalled, producing iterations with no completed runs at all.
            now = time.time()
            last_activity = now
            for c in self.conns:
                if c.alive:
                    last_seen[c.idx] = now
            for (conn, _), a in zip(pending, actions):
                conn.send_action(a)
            self.steps += len(pending)
            self.batches += 1
            self.batch_sizes.append(len(pending))
            pending.clear()
            if max_steps is not None and self.steps >= max_steps:
                break

        self.wall = time.time() - t0
        return self.results

    def _dispatch(self, conn, msg, on_fight, on_terminal):
        """Handles one message; returns a (conn, msg) pair when it is a decision to answer."""
        t = msg.get("t")
        if t == "decision":
            return (conn, msg)
        if t == "fight_end":
            fr = FightResult(msg["won"], msg["outcome"], msg["room"], msg["encounter"],
                             msg["turns"], msg["hp_start"], msg["hp_end"], msg["max_hp"],
                             int(msg.get("act", -1)))
            self.fights.append(fr)
            if on_fight is not None:
                on_fight(conn.idx, fr)
        elif t == "terminal":
            rr = RunResult(msg["run"], msg["outcome"], msg["won"], msg["floors"], msg["act"],
                           msg["acts"], msg["hp"], msg["max_hp"], msg["gold"], msg["deck"],
                           msg["relics"], msg["fights"], msg["fight_wins"])
            self.results.append(rr)
            self._runs_by_env[conn.idx] = self._runs_by_env.get(conn.idx, 0) + 1
            if on_terminal is not None:
                on_terminal(conn.idx, rr)
        elif t == "hello":
            self.sizes = msg
            # The nets are built from the CHECKPOINT's vocab sizes, but the ids come from the
            # LIVE game. If the game's vocab is larger (a patch added cards), every new id is
            # silently clamped by _ix onto the last embedding row -- all of them sharing one
            # vector, with no error anywhere. Compare once, loudly.
            if self.expect_sizes and not self._sizes_checked:
                self._sizes_checked = True
                bad = [(k, msg.get(k), v) for k, v in self.expect_sizes.items()
                       if isinstance(msg.get(k), int) and msg[k] > v]
                if bad:
                    detail = ", ".join(f"{k}: game={g} > model={m}" for k, g, m in bad)
                    print(f"  WARNING: VOCAB MISMATCH -- {detail}. Ids beyond the model's "
                          f"table are clamped onto its last embedding row and become "
                          f"indistinguishable. Retrain or rebuild the vocab.", flush=True)
        elif t == "done":
            conn.done = True
            conn.alive = False
        return None

    def _drain(self, conn):
        """Reads everything left in the socket for an exited process."""
        out = []
        for _ in range(64):
            before = len(conn.buf)
            got = list(conn.read_messages())
            out.extend(got)
            if not got and len(conn.buf) == before:
                break
        return out

    def _on_stall(self, conn):
        """Records a wedged env and PRESERVES its probe log.

        The probe log is opened with append=False, so the next iteration's env with the same
        index overwrites it and the evidence for a stall is gone by the time anyone looks.
        Copying it aside on the spot is the difference between "one env stalled" and being
        able to say which room it stalled in.
        """
        self.stalled += 1
        self.stalled_envs.append(conn.idx)
        dst, tail = self._preserve_log(conn.idx, "stall")
        print(f"VecRunEnv: env {conn.idx} ({self.characters[conn.idx]}) stalled; "
              f"log -> {dst}\n    last line: {tail}", flush=True)

    def _on_exit(self, conn):
        """The socket closed before the env sent "done": the game exited or crashed
        mid-session. Records the exit code and preserves the probe log, since a crash that
        silently costs an env's remaining runs is otherwise invisible in the iteration line
        (it just reads runs=95)."""
        p = self.procs[conn.idx]
        try:
            code = p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            code = None
        self.crashed += 1
        self.crashed_envs.append((conn.idx, code))
        dst, tail = self._preserve_log(conn.idx, "exit")
        print(f"VecRunEnv: env {conn.idx} ({self.characters[conn.idx]}) closed before 'done' "
              f"(exit code {code}) after {self._runs_by_env.get(conn.idx, 0)} run(s); "
              f"log -> {dst}\n    last line: {tail}", flush=True)

    def _preserve_log(self, idx, kind):
        """Copies env ``idx``'s probe log aside (the next iteration overwrites it) and returns
        (destination, last line)."""
        src = os.path.join(f"{self._out_dir}{idx}", "probe.log")
        dst, tail = "(not preserved)", ""
        try:
            os.makedirs(self._stall_dir, exist_ok=True)
            dst = os.path.join(self._stall_dir, f"{kind}-{int(time.time())}-env{idx}.log")
            shutil.copyfile(src, dst)
            with open(src) as fh:
                lines = fh.readlines()
                tail = lines[-1].strip() if lines else ""
        except Exception as e:
            dst = f"(could not preserve log: {e})"
        return dst, tail

    def _unregister(self, conn):
        try:
            self._sel.unregister(conn.sock)
        except (KeyError, ValueError):
            pass
        conn.close()

    def close(self):
        for c in self.conns:
            c.close()
        try:
            self._sel.close()
        except Exception:
            pass
        try:
            for s in self._srvs:
                s.close()
        except OSError:
            pass
        # Hard teardown: a wedged game process ignores a polite close and would otherwise
        # linger holding a GPU-free but very real 300MB of RAM until the trainer exits.
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 10
        for p in self.procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except Exception:
                p.kill()

    @property
    def mean_batch(self) -> float:
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0

    @property
    def win_rate(self) -> float:
        return (sum(r.won for r in self.results) / len(self.results)
                if self.results else 0.0)


class RunPool:
    """PERSISTENT game processes fed from a run queue (probe ``--probe-run-queue``).

    VecRunEnv launches every process per iteration and gives each a fixed share of runs, so
    each iteration pays a boot per process (~4-5 s, plus the ~2% first-run engine crash) and
    waits for the slowest process to finish its share. Here the processes live across
    iterations and ask for runs one at a time (``next_run``), so fast ones pick up the next
    run and the pool can be any size, e.g. more slots than the 24 x 4 split allowed.

    ``run(jobs, ...)`` plays exactly ``jobs``, each ``{"run", "seed", "character"}``. Every job
    completes once: a slot that crashes or wedges mid-run has its partial run dropped
    (``on_restart(slot)``), its job goes back to the FRONT of the queue (the same seed is
    replayed from the start, so paired comparisons hold), and the slot is relaunched. Between
    iterations, finished slots wait on an unanswered ``next_run``. A process is recycled after
    ``recycle_runs`` runs, so slow leaks inside the game cannot build up.

    Callbacks get the SLOT index as the env index: a slot plays one run at a time, so per-env
    open-run state in the caller stays correct.
    """

    ENV_STALL_TIMEOUT = 40.0

    def __init__(self, n_envs=32, game_dir=DEFAULT_GAME, fixed_fps=5000, ascension=0,
                 turn_cap=50, floor_cap=200, act_cap=0, quiet=True, batch_timeout=0.004,
                 out_dir="/tmp/alphaslayer_pool_", stall_dir="/tmp/alphaslayer_stalls",
                 recycle_runs=50, connect_timeout=120.0, max_job_retries=3, expect_sizes=None,
                 nice=5, cpus=None, extra_args=()):
        self.n = n_envs
        self.extra_args = list(extra_args)      # more probe flags, e.g. --probe-fight-restore
        # CPUs the games may run on (None = any). Keeping them off the trainer's cores keeps
        # its single policy thread from being descheduled by game threads.
        self.cpus = cpus
        # Games run at lower CPU priority than the trainer: collection is bound by the
        # trainer's single policy thread (75% of wall time), and every game is waiting on it.
        self.nice = nice
        self.batch_timeout = batch_timeout
        self.recycle_runs = recycle_runs
        self.connect_timeout = connect_timeout
        self.max_job_retries = max_job_retries
        self.expect_sizes = expect_sizes or {}
        self._sizes_checked = False
        self.sizes = {}
        self._cfg = dict(game_dir=game_dir, fixed_fps=fixed_fps, ascension=ascension,
                         turn_cap=turn_cap, floor_cap=floor_cap, act_cap=act_cap, quiet=quiet)
        self._out_dir, self._stall_dir = out_dir, stall_dir
        self._srvs = listen_per_env(n_envs)
        self._sel = selectors.DefaultSelector()
        self.procs: list = [None] * n_envs
        self.conns: list = [None] * n_envs
        self.slot_runs = [0] * n_envs          # runs finished by the slot's CURRENT process
        self.waiting = [False] * n_envs        # sent next_run, no job yet
        self._awaiting: dict[int, float] = {}  # slot -> connect deadline of a fresh process
        self._last_seen = [time.time()] * n_envs
        self.launches = 0
        self.total = {"crashed": 0, "stalled": 0, "recycled": 0, "relaunched": 0}
        for i in range(n_envs):
            self._launch(i)

    # ---- process lifecycle ----------------------------------------------------------------

    def _launch(self, i):
        c = self._cfg
        args = [os.path.join(c["game_dir"], "SlayTheSpire2"), "--headless",
                "--fixed-fps", str(c["fixed_fps"]),
                "--probe=runserve", f"--probe-out={self._out_dir}{i}", "--probe-quit",
                f"--probe-port={self._srvs[i].getsockname()[1]}", "--probe-run-queue",
                f"--probe-ascension={c['ascension']}", f"--probe-turn-cap={c['turn_cap']}",
                f"--probe-floor-cap={c['floor_cap']}"]
        if c["act_cap"]:
            args.append(f"--probe-act-cap={c['act_cap']}")
        args += self.extra_args
        env = dict(os.environ, SteamAppId="2868840", SteamGameId="2868840")
        def pre(n=self.nice, cpus=self.cpus):
            game_preexec()
            if n:
                os.nice(n)
            if cpus:
                os.sched_setaffinity(0, cpus)
        self.procs[i] = subprocess.Popen(
            args, cwd=c["game_dir"], env=env, preexec_fn=pre,
            stdout=subprocess.DEVNULL if c["quiet"] else None,
            stderr=subprocess.DEVNULL if c["quiet"] else None)
        self.launches += 1
        self.slot_runs[i] = 0
        self.waiting[i] = False
        self._awaiting[i] = time.time() + self.connect_timeout
        self._sel.register(self._srvs[i], selectors.EVENT_READ, ("listen", i))

    def _accept(self, i):
        try:
            sock, _ = self._srvs[i].accept()
        except BlockingIOError:
            return
        try:
            self._sel.unregister(self._srvs[i])
        except (KeyError, ValueError):
            pass
        self._awaiting.pop(i, None)
        c = _Conn(sock, i)
        self.conns[i] = c
        self._sel.register(c.sock, selectors.EVENT_READ, c)
        self._last_seen[i] = time.time()

    def _drop_conn(self, i):
        c = self.conns[i]
        if c is not None:
            try:
                self._sel.unregister(c.sock)
            except (KeyError, ValueError):
                pass
            c.close()
        self.conns[i] = None

    def _kill(self, i):
        p = self.procs[i]
        if p is not None and p.poll() is None:
            p.kill()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def _send(self, i, obj):
        c = self.conns[i]
        try:
            c.sock.sendall(json.dumps(obj).encode() + b"\n")
        except OSError:
            c.alive = False

    # ---- one batch of jobs ------------------------------------------------------------------

    def run(self, jobs, policy, on_fight=None, on_terminal=None, on_restart=None, deadline_s=None,
            evaluate=None, search=None, run_search=None):
        """Plays every job once; returns the RunResults (in completion order). Results carry
        only the job's ``run`` id, so the ids must be unique to map a result back to its job.

        Search (probe --probe-search): a game sends {"t":"eval","leaves":[...]} with every
        simulated choice's resulting position and waits for {"v":[...]}; ``evaluate(leaves)``
        returns one value per leaf (all games' leaves in one call). The decision that follows
        carries ``q`` (one value per legal choice); a policy with ``takes_q = True`` gets them as
        ``qs=`` (None where a decision was not searched).

        Turn search (--probe-search-mode=turn) also asks {"t":"prior"} (answered with the
        policy's probabilities, ``search.prior(obs, legal)``) and {"t":"roll"} (a rollout
        choice, ``search.roll(obs, legal)``; not part of any trajectory). ``search`` may also
        carry ``evaluate``.

        Run search (probe --probe-run-search): a live decision may carry "ctx" (its room-entry
        save + the decisions since). ``run_search`` (alphaslayer.unified.runsearch.RunSearcher)
        names the options to value; each is played out in ``worlds`` sampled worlds by ROLLOUT
        jobs on other slots (they jump the queue; at most ``max_live`` slots play live runs,
        so rollouts always find a game). The live decision waits (parked) until its rollouts
        report, then reaches the policy with ``q``. Rollout decisions are answered by
        ``run_search.policy`` and never reach ``policy``; their fights and terminals never
        reach the callbacks or the results."""
        if search is not None and evaluate is None:
            evaluate = getattr(search, "evaluate", None)
        rs = run_search
        rq: collections.deque = collections.deque()      # rollout jobs, served first
        tasks: dict[int, dict] = {}                      # task id -> parked decision + results
        roll_of: dict[int, tuple] = {}                   # rollout run id -> (task id, option, world)
        parked: dict[int, int] = {}                      # slot -> task id
        max_live = rs.max_live if rs is not None else self.n
        self.rs_stats = collections.Counter()
        ids = [j["run"] for j in jobs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"RunPool.run: {len(ids) - len(set(ids))} duplicate run id(s)")
        queue = collections.deque(jobs)
        inflight: dict[int, dict] = {}
        retries = collections.Counter()
        self.results: list[RunResult] = []
        self.fights: list[FightResult] = []
        self.lost: list[dict] = []
        it_stats = {"crashed": 0, "stalled": 0, "recycled": 0, "relaunched": 0}
        self.policy_s = 0.0
        self.leaf_evals = 0
        self.search_msgs = 0
        self.batches = 0
        self.steps = 0
        # where the server loop's time goes (seconds, message counts, answered batches)
        self.prof = prof = collections.Counter()
        t0 = time.time()
        target = len(jobs)
        hard = (t0 + deadline_s) if deadline_s else None

        def is_roll(job):
            return job is not None and "rollout" in job

        def give(i):
            if not (self.conns[i] is not None and self.conns[i].alive):
                return
            if rq:
                job = rq.popleft()
                inflight[i] = job
                self.waiting[i] = False
                self._last_seen[i] = time.time()
                self._send(i, {"run": job["run"], "seed": job["seed"], "character": job["character"],
                               "search": False, "rollout": job["rollout"]})
                return
            n_live = sum(1 for j in inflight.values() if not is_roll(j))
            if queue and n_live < max_live:
                job = queue.popleft()
                inflight[i] = job
                self.waiting[i] = False
                self._last_seen[i] = time.time()
                self._send(i, {"run": job["run"], "seed": job["seed"],
                               "character": job["character"],
                               **({"search": bool(job["search"])} if "search" in job else {})})

        def fail(i, kind):
            """A slot's process died or wedged: drop its partial run, requeue, relaunch."""
            it_stats[kind] += 1
            self.total[kind] += 1
            if kind == "stalled":
                self._kill(i)
            dst, tail = self._preserve_log(i, "stall" if kind == "stalled" else "exit")
            job = inflight.pop(i, None)
            what = f"run {job['run']} ({job['character']})" if job else "no run in flight"
            code = self.procs[i].poll() if self.procs[i] is not None else None
            print(f"RunPool: slot {i} {kind} (exit {code}), {what}; log -> {dst}\n"
                  f"    last line: {tail}", flush=True)
            if is_roll(job):
                finish_rollout(job["run"], None)
                job = None
            if i in parked:                          # the parked live run died with its slot
                tasks.pop(parked.pop(i), None)
            if job is not None:
                if on_restart is not None:
                    on_restart(i)
                retries[job["run"], job["seed"]] += 1
                if retries[job["run"], job["seed"]] <= self.max_job_retries:
                    queue.appendleft(job)
                else:
                    self.lost.append(job)
                    print(f"RunPool: giving up on run {job['run']} seed {job['seed']} after "
                          f"{self.max_job_retries} retries", flush=True)
            self._drop_conn(i)
            self._kill(i)
            it_stats["relaunched"] += 1
            self.total["relaunched"] += 1
            self._launch(i)

        pending: list[tuple[_Conn, dict]] = []
        ids_ = itertools.count(1)

        def start_task(conn, m):
            """Parks a live decision and queues the rollouts that value its options."""
            tid = next(ids_)
            live_job = inflight[conn.idx]
            plan = {"cands": [None], "worlds": 1, "boss_plays": 1, "teleport": False} if rs.verify else rs.plan(m)
            cands, worlds = plan["cands"], plan["worlds"]
            ctx = m.pop("ctx")
            tasks[tid] = {"conn": conn, "msg": m, "cands": cands, "left": 0,
                          "vals": collections.defaultdict(list), "t0": time.time()}
            for w in range(worlds):              # every option sees the same worlds
                world = (tid * 7919 + w * 104729) & 0x7FFFFFFF
                for c in cands:
                    rid = 2_000_000_000 - next(ids_)
                    rq.append({"run": rid, "seed": live_job["seed"], "character": live_job["character"],
                               "rollout": {"restore": ctx["save"],
                                           "forced": (list(ctx["log"])[:-1] if getattr(rs, "drop_last", False)
                                                      else list(ctx["log"])) + ([] if c is None else [int(c)]),
                                           "world": world, "boss_plays": plan["boss_plays"],
                                           "teleport": plan["teleport"],
                                           "expect_kind": m["kind"], "expect_n": len(m["legal"]),
                                           "verify": bool(rs.verify)}})
                    roll_of[rid] = (tid, c, w)
                    tasks[tid]["left"] += 1
            parked[conn.idx] = tid
            self.rs_stats["searched"] += 1
            for j in range(self.n):
                if self.waiting[j]:
                    give(j)

        def finish_rollout(rid, report):
            tid, c, _w = roll_of.pop(rid, (None, None, None))
            task = tasks.get(tid)
            if task is None:
                return
            task["left"] -= 1
            if report is None or report.get("replay_error"):
                self.rs_stats["rollouts_failed"] += 1
            elif not rs.verify:
                task["vals"][c].append((_w, rs.leaf_value(report)))
                task.setdefault("reports", []).append(report)
                self.rs_stats["rollouts"] += 1
            if task["left"] <= 0:
                tasks.pop(tid)
                conn, m = task["conn"], task["msg"]
                parked.pop(conn.idx, None)
                if not rs.verify and task["vals"]:
                    q = [-1e9] * len(m["legal"])
                    for c_, vs in task["vals"].items():
                        q[c_] = rs.combine(vs)            # a float that keeps its playouts
                    m["q"] = q
                    rs.record(m, task)
                self.rs_stats["search_s"] += time.time() - task["t0"]
                if conn.alive and self.conns[conn.idx] is conn:
                    pending.append((conn, m))

        for i in range(self.n):
            if self.waiting[i]:
                give(i)

        while len(self.results) + len(self.lost) < target:
            if hard and time.time() > hard:
                print(f"RunPool: deadline ({deadline_s:.0f}s) hit with {len(inflight)} run(s) "
                      f"in flight; relaunching those slots", flush=True)
                for i in list(inflight):
                    if not is_roll(inflight[i]):
                        self.lost.append(inflight[i])
                    fail(i, "stalled")
                self.timed_out = True
                break
            deadline = None
            while True:
                live = sum(1 for i in inflight if self.conns[i] is not None and i not in parked)
                if pending and len(pending) >= live:
                    break
                remaining = self.batch_timeout if deadline is None else deadline - time.time()
                if deadline is not None and remaining <= 0:
                    break
                ts = time.perf_counter()
                events = self._sel.select(timeout=max(0.0, remaining) if pending else 1.0)
                prof["select_s"] += time.perf_counter() - ts
                ts = time.perf_counter()
                for key, _ in events:
                    if isinstance(key.data, tuple):
                        self._accept(key.data[1])
                        continue
                    conn: _Conn = key.data
                    i = conn.idx
                    if self.conns[i] is not conn:
                        continue
                    self._last_seen[i] = time.time()
                    for msg in conn.read_messages():
                        t = msg.get("t")
                        if t in ("decision", "eval", "prior", "roll"):
                            pending.append((conn, msg))
                        elif t == "next_run":
                            if i in inflight:          # a run ended without its terminal
                                job_ = inflight.pop(i)
                                if is_roll(job_):
                                    finish_rollout(job_["run"], None)
                            if self.slot_runs[i] >= self.recycle_runs:
                                self._send(i, {"quit": True})
                            else:
                                give(i)
                                if i not in inflight:
                                    self.waiting[i] = True
                        elif is_roll(inflight.get(i)) and t in ("fight_end", "terminal"):
                            if t == "terminal":       # a rollout's leaf: never a result
                                job_ = inflight.pop(i)
                                self.slot_runs[i] += 0.1
                                finish_rollout(job_["run"], msg.get("rollout"))
                        elif t == "fight_end":
                            fr = FightResult(msg["won"], msg["outcome"], msg["room"],
                                             msg["encounter"], msg["turns"], msg["hp_start"],
                                             msg["hp_end"], msg["max_hp"], int(msg.get("act", -1)))
                            self.fights.append(fr)
                            if on_fight is not None:
                                on_fight(i, fr)
                        elif t == "terminal":
                            rr = RunResult(msg["run"], msg["outcome"], msg["won"], msg["floors"],
                                           msg["act"], msg["acts"], msg["hp"], msg["max_hp"],
                                           msg["gold"], msg["deck"], msg["relics"], msg["fights"],
                                           msg["fight_wins"])
                            inflight.pop(i, None)
                            self.slot_runs[i] += 1
                            self.results.append(rr)
                            if on_terminal is not None:
                                on_terminal(i, rr)
                            for j in range(self.n):   # a live seat is free: idle slots may take it
                                if self.waiting[j] and j != i:
                                    give(j)
                        elif t == "hello":
                            self._check_sizes(msg)
                        elif t == "done":
                            conn.done = True
                    if not conn.alive:
                        if conn.done and i not in inflight:   # recycled on request
                            it_stats["recycled"] += 1
                            self.total["recycled"] += 1
                            self._drop_conn(i)
                            self._kill(i)
                            self._launch(i)
                        else:
                            fail(i, "crashed")
                prof["read_s"] += time.perf_counter() - ts
                if pending and deadline is None:
                    deadline = time.time() + self.batch_timeout
                if not pending and not events:
                    break
            # wedged slots: a run in flight, nothing owed by us, silent too long
            now = time.time()
            owed = {c.idx for c, _ in pending} | set(parked)
            for i in list(inflight):
                if i not in owed and now - self._last_seen[i] > self.ENV_STALL_TIMEOUT:
                    fail(i, "stalled")
            for i, dl in list(self._awaiting.items()):
                if now > dl:
                    print(f"RunPool: slot {i} never connected; relaunching", flush=True)
                    self._awaiting.pop(i, None)
                    try:
                        self._sel.unregister(self._srvs[i])
                    except (KeyError, ValueError):
                        pass
                    self._kill(i)
                    self._launch(i)
            pending = [(c, m) for c, m in pending if c.alive and self.conns[c.idx] is c]
            if not pending:
                continue
            if rs is not None:
                keep, roll_dec = [], []
                for c, m in pending:
                    if m.get("t") == "decision" and is_roll(inflight.get(c.idx)):
                        roll_dec.append((c, m))
                    elif m.get("t") == "decision" and "ctx" in m and c.idx in inflight:
                        start_task(c, m)
                    else:
                        keep.append((c, m))
                pending = keep
                if roll_dec:
                    tr = time.time()
                    acts_r = rs.policy([m["kind"] for _, m in roll_dec], [m["obs"] for _, m in roll_dec],
                                       [m["legal"] for _, m in roll_dec])
                    for (c, m), a_ in zip(roll_dec, acts_r):
                        if m.get("verify"):
                            rs.verify_decision(m, tasks.get(roll_of.get(inflight[c.idx]["run"], (None,))[0]))
                        c.send_action(int(a_))
                    self.rs_stats["rollout_decisions"] += len(roll_dec)
                    self.rs_stats["rollout_policy_s"] += time.time() - tr
                if not pending:
                    continue
            evals = [(c, m) for c, m in pending if m.get("t") == "eval"]
            priors = [(c, m) for c, m in pending if m.get("t") == "prior"]
            rolls = [(c, m) for c, m in pending if m.get("t") == "roll"]
            pending = [(c, m) for c, m in pending if m.get("t") not in ("eval", "prior", "roll")]
            if (priors or rolls) and search is None:
                raise RuntimeError("a game asked for a search prior/rollout but run() got no search")
            if evals and evaluate is None:
                raise RuntimeError("a game asked for leaf values (--probe-search) but run() got no evaluate")
            tp = time.time()
            prof["batches"] += 1
            # Every network pass costs about the same whatever its batch (~4 ms, launch bound),
            # so the server answers each step with as few passes as it can: leaf values and
            # priors in one fp32 pass (search.eval_prior), rollout choices inside the live
            # policy's pass (policy.takes_rolls). Rollouts wait on these round trips.
            ts = time.perf_counter()
            flat = [leaf for _, m in evals for leaf in m["leaves"]]
            fused_ep = (evals or priors) and getattr(search, "eval_prior", None) is not None
            if fused_ep:
                vals, pri = search.eval_prior(flat, [m["obs"] for _, m in priors], [m["legal"] for _, m in priors])
            else:
                vals = evaluate(flat) if evals else []
                pri = search.prior([m["obs"] for _, m in priors], [m["legal"] for _, m in priors]) if priors else []
            k = 0
            for conn, m in evals:
                n = len(m["leaves"])
                conn.send_json({"v": [float(v) for v in vals[k:k + n]]})
                k += n
            for (conn, _), o in zip(priors, pri):
                conn.send_json({"p": [float(x) for x in o]})
            if evals or priors:
                self.leaf_evals += len(flat)
                prof["eval_n"] += len(flat)
                prof["prior_n"] += len(priors)
                prof["eval_prior_s"] += time.perf_counter() - ts
                prof["eval_prior_batches"] += 1
            fused_roll = bool(rolls and pending and getattr(policy, "takes_rolls", False))
            if rolls and not fused_roll:
                ts = time.perf_counter()
                out = search.roll([m["obs"] for _, m in rolls], [m["legal"] for _, m in rolls])
                for (conn, _), o in zip(rolls, out):
                    conn.send_action(int(o))
                prof["roll_s"] += time.perf_counter() - ts
                prof["roll_batches"] += 1
            prof["roll_n"] += len(rolls)
            self.search_msgs += len(priors) + len(rolls)
            actions = []
            ts = time.perf_counter()
            if pending:
                kinds = [m["kind"] for _, m in pending]
                obs = [m["obs"] for _, m in pending]
                legal = [m["legal"] for _, m in pending]
                idxs = [c.idx for c, _ in pending]
                kw = {}
                if getattr(policy, "takes_q", False):
                    kw["qs"] = [m.get("q") for _, m in pending]
                if fused_roll:
                    kw["rolls"] = ([m["obs"] for _, m in rolls], [m["legal"] for _, m in rolls])
                    actions, racts = policy(kinds, obs, legal, idxs, **kw)
                    for (conn, _), o in zip(rolls, racts):
                        conn.send_action(int(o))
                    prof["fused_roll_n"] += len(rolls)
                else:
                    actions = policy(kinds, obs, legal, idxs, **kw)
                prof["live_s"] += time.perf_counter() - ts
                prof["live_n"] += len(pending)
            self.policy_s += time.time() - tp
            now = time.time()
            for i in range(self.n):
                self._last_seen[i] = max(self._last_seen[i], now)
            for (conn, _), a in zip(pending, actions):
                conn.send_action(a)
            self.steps += len(pending)
            self.batches += 1
            pending = []
        self.wall = time.time() - t0
        self.stats = it_stats
        return self.results

    def _check_sizes(self, msg):
        self.sizes = msg
        if self.expect_sizes and not self._sizes_checked:
            self._sizes_checked = True
            bad = [(k, msg.get(k), v) for k, v in self.expect_sizes.items()
                   if isinstance(msg.get(k), int) and msg[k] > v]
            if bad:
                print("  WARNING: VOCAB MISMATCH -- " + ", ".join(
                    f"{k}: game={g} > model={m}" for k, g, m in bad), flush=True)

    def _preserve_log(self, idx, kind):
        src = os.path.join(f"{self._out_dir}{idx}", "probe.log")
        dst, tail = "(not preserved)", ""
        try:
            os.makedirs(self._stall_dir, exist_ok=True)
            dst = os.path.join(self._stall_dir, f"pool-{kind}-{int(time.time())}-slot{idx}.log")
            shutil.copyfile(src, dst)
            with open(src) as fh:
                lines = fh.readlines()
                tail = lines[-1].strip() if lines else ""
        except Exception as e:
            dst = f"(could not preserve log: {e})"
        return dst, tail

    def close(self):
        for i in range(self.n):
            self._drop_conn(i)
        try:
            self._sel.close()
        except Exception:
            pass
        for s in self._srvs:
            try:
                s.close()
            except OSError:
                pass
        for p in self.procs:
            if p is not None and p.poll() is None:
                p.terminate()
        deadline = time.time() + 10
        for p in self.procs:
            if p is None:
                continue
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except Exception:
                p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
