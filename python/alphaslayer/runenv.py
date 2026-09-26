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
                 stall_dir="/tmp/alphaslayer_stalls", close_on_exit=False, expect_sizes=None):
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
        self._out_dir = out_dir
        self._stall_dir = stall_dir
        self.close_on_exit = close_on_exit

        self._srvs = listen_per_env(n_envs)

        env = dict(os.environ, SteamAppId="2868840", SteamGameId="2868840")
        roster = characters or [character]
        self.characters = [roster[i % len(roster)] for i in range(n_envs)]
        self.procs = []
        for i in range(n_envs):
            args = [
                os.path.join(game_dir, "SlayTheSpire2"), "--headless",
                "--fixed-fps", str(fixed_fps),
                "--probe=runserve", f"--probe-out={out_dir}{i}", "--probe-quit",
                f"--probe-port={self._srvs[i].getsockname()[1]}",
                f"--probe-runs={runs_per_env}",
                f"--probe-character={self.characters[i]}",
                f"--probe-seed={seed}{i}",
                f"--probe-ascension={ascension}",
                f"--probe-turn-cap={turn_cap}",
                f"--probe-floor-cap={floor_cap}",
                # Distinct seed sequences per env AND per iteration, so successive rollouts
                # do not replay the same maps.
                f"--probe-run-offset={run_offset}",
            ]
            if act_cap:
                args.append(f"--probe-act-cap={act_cap}")
            self.procs.append(subprocess.Popen(
                args, cwd=game_dir, env=env, preexec_fn=game_preexec,
                stdout=subprocess.DEVNULL if quiet else None,
                stderr=subprocess.DEVNULL if quiet else None))

        accepted = accept_per_env(self._srvs, 120, "VecRunEnv")
        self.conns = [accepted[i] for i in sorted(accepted)]
        if not self.conns:
            raise RuntimeError("no game processes connected")

        self._sel = selectors.DefaultSelector()
        for c in self.conns:
            self._sel.register(c.sock, selectors.EVENT_READ, c)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, policy, on_fight=None, on_terminal=None, max_steps=None, deadline_s=None):
        """Drives the envs. ``policy(kinds, obs, legal, env_idxs) -> list[int]``.

        ``on_fight(env_idx, FightResult)`` fires at the end of each combat and
        ``on_terminal(env_idx, RunResult)`` at the end of each run. Two callbacks because the
        two policies are credited from different things: the combat policy from the fight it
        just played, the run policy from how the run ends.
        """
        pending: list[tuple[_Conn, dict]] = []
        t0 = time.time()
        last_activity = time.time()
        last_seen = {c.idx: time.time() for c in self.conns}

        hard_deadline = (t0 + deadline_s) if deadline_s else None
        self.timed_out = False
        while any(c.alive for c in self.conns):
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
                break
            deadline = None
            while True:
                live = sum(1 for c in self.conns if c.alive)
                if live == 0 or (pending and len(pending) >= live):
                    break
                remaining = self.batch_timeout if deadline is None else deadline - time.time()
                if pending and remaining <= 0:
                    break
                events = self._sel.select(timeout=max(0.0005, min(remaining, 0.05)))
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
                if not events and time.time() - last_activity > self.STALL_TIMEOUT and not pending:
                    live_conns = [c for c in self.conns if c.alive]
                    print(f"VecRunEnv: no traffic from any env for {self.STALL_TIMEOUT:.0f}s; "
                          f"dropping {len(live_conns)} stalled env(s)", flush=True)
                    for c in live_conns:
                        self._on_stall(c)
                        self._unregister(c)
                    break
                for key, _ in events:
                    conn: _Conn = key.data
                    if not conn.alive:
                        continue
                    last_seen[conn.idx] = time.time()
                    for msg in conn.read_messages():
                        d = self._dispatch(conn, msg, on_fight, on_terminal)
                        if d is not None:
                            pending.append(d)
                    if not conn.alive:
                        if not conn.done:
                            self._on_exit(conn)
                        self._unregister(conn)
                if pending and deadline is None:
                    deadline = time.time() + self.batch_timeout
                if not events and not pending and not any(c.alive for c in self.conns):
                    break

            if not pending:
                if not any(c.alive for c in self.conns):
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
