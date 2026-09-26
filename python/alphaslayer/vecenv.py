"""Vectorised env: one policy server, N game processes, batched inference.

The single-env bridge runs at ~20 steps/s because every decision costs a TCP round trip plus
a batch-size-1 GPU forward. Both are fixed costs that amortise across environments, so the fix
is to run many game processes against one policy and batch their pending decisions into a
single forward pass.

Each game process is strictly request/response and blocks on its reply, so at steady state
every live env has exactly one decision outstanding -- which is precisely a full batch. A
selector loop collects whatever is ready (so a straggler setting up its next combat cannot
stall the others), runs one forward pass, and replies to all of them.

    def policy(obs_list, legal_list) -> list[int]: ...
    with VecSpireEnv(n_envs=8, character="IRONCLAD", episodes_per_env=25) as venv:
        venv.run(policy)
        print(venv.win_rate)
"""

from __future__ import annotations

import json
import os
import selectors
import socket
import subprocess
import time

from .env import DEFAULT_GAME, EpisodeResult


class _Conn:
    """Line-framed JSON over one accepted socket."""

    __slots__ = ("sock", "buf", "alive", "idx", "done")

    def __init__(self, sock, idx):
        self.sock = sock
        # Disable Nagle: this protocol is one small request and one small response per
        # decision, exactly the pattern Nagle + delayed ACK punishes with ~40ms stalls.
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setblocking(False)
        self.buf = b""
        self.alive = True
        self.idx = idx
        self.done = False    # the env sent its final "done" message

    def read_messages(self):
        """Non-blocking read; yields complete JSON objects."""
        try:
            chunk = self.sock.recv(1 << 16)
        except BlockingIOError:
            return
        except OSError:
            self.alive = False
            return
        if not chunk:
            self.alive = False
            return
        self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            if line.strip():
                yield json.loads(line)

    def send_action(self, a: int):
        try:
            self.sock.sendall(json.dumps({"a": int(a)}).encode() + b"\n")
        except OSError:
            self.alive = False

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
        self.alive = False



def game_preexec():
    """Child-side setup for a game process: no core dumps.

    The game segfaults or aborts on the way out of a large fraction of its shutdowns (708
    dumps in the first 321 iterations of one run). Each is a ~200MB core that systemd
    compresses and stores, and none of them are ours to debug. A crash still surfaces as a
    non-zero exit code, which VecRunEnv logs.
    """
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def listen_per_env(n):
    """One listening socket per env, so a connection identifies the process that made it.

    A single shared socket numbered connections in ACCEPT order while processes, characters
    and probe-log directories were numbered in SPAWN order. Game processes boot in arbitrary
    order, so conn.idx pointed at the wrong process for most envs (18 of 20 in one test):
    every stall log copied aside was some other env's, and the deck-eval stall logic polled
    the wrong process.
    """
    srvs = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        srvs.append(s)
    return srvs


def accept_per_env(srvs, timeout, label):
    """Accepts whichever env connects next, in any order, until all have or the deadline
    passes. Returns ``{idx: _Conn}``; a missing idx is a process that never connected."""
    sel = selectors.DefaultSelector()
    for i, s in enumerate(srvs):
        sel.register(s, selectors.EVENT_READ, i)
    conns = {}
    deadline = time.time() + timeout
    try:
        while len(conns) < len(srvs):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            for key, _ in sel.select(timeout=min(remaining, 1.0)):
                i = key.data
                if i in conns:
                    continue
                sock, _addr = srvs[i].accept()
                conns[i] = _Conn(sock, i)
    finally:
        sel.close()
    if len(conns) < len(srvs):
        missing = sorted(set(range(len(srvs))) - set(conns))
        print(f"{label}: only {len(conns)}/{len(srvs)} envs connected (missing {missing}); "
              f"continuing", flush=True)
    return conns

class VecSpireEnv:
    STALL_TIMEOUT = 120.0   # seconds with no traffic from ANY env before giving up on it

    def __init__(self, n_envs=8, character="IRONCLAD", characters=None, episodes_per_env=25,
                 seed="ALPHASLAYER", ascension=0, game_dir=DEFAULT_GAME, fixed_fps=5000,
                 exclude_encounters=None, inject_cards=None, inject_relics=None, potions=False,
                 act=0,
                 encounters=None, mix=None, reroll_deck=False, quiet=True,
                 batch_timeout=0.004):
        self.n_envs = n_envs
        self.batch_timeout = batch_timeout
        self.results: list[EpisodeResult] = []
        self.sizes = {}
        self.steps = 0
        self.batches = 0
        self.batch_sizes: list[int] = []

        self._srvs = listen_per_env(n_envs)

        env = dict(os.environ, SteamAppId="2868840", SteamGameId="2868840")
        # Round-robin characters across envs so one rollout covers the whole roster; a policy
        # trained on one character never sees the others' cards, relics or mechanics.
        roster = characters or [character]
        self.characters = [roster[i % len(roster)] for i in range(n_envs)]
        self.procs = []
        for i in range(n_envs):
            args = [
                os.path.join(game_dir, "SlayTheSpire2"), "--headless",
                "--fixed-fps", str(fixed_fps),
                "--probe=serve", "--probe-out=/tmp/alphaslayer_probe", "--probe-quit",
                f"--probe-port={self._srvs[i].getsockname()[1]}",
                f"--probe-episodes={episodes_per_env}",
                f"--probe-character={self.characters[i]}",
                # Distinct seeds: identical seeds would make every env replay the same
                # episodes, which is wasted compute and a biased sample.
                f"--probe-seed={seed}{i}",
                f"--probe-ascension={ascension}",
            ]
            if encounters:
                args.append(f"--probe-encounters={encounters}")
            if mix:
                args.append(f"--probe-mix={mix}")
            if reroll_deck:
                args.append("--probe-reroll-deck")
            if exclude_encounters:
                args.append(f"--probe-exclude-encounters={','.join(exclude_encounters)}")
            if inject_cards:
                args.append(f"--probe-inject-cards={','.join(inject_cards)}")
            if inject_relics:
                args.append(f"--probe-inject-relics={','.join(inject_relics)}")
            if act:
                # Which act's encounter pools to draw from. Act 2 bosses and elites are a
                # materially harder target than act 1, which is what makes potion value
                # visible: at act-1 difficulty with Pandora's deck the policy already wins
                # ~0.89 and a potion rarely changes the outcome.
                args.append(f"--probe-act={act}")
            if potions:
                # Opens the potion gate in the COMBAT-ONLY env. Without this, use_potion is
                # never offered here, which is why the combat policy met potions for the
                # first time inside real runs and drank them essentially at random.
                args.append("--probe-potions")
            self.procs.append(subprocess.Popen(
                args, cwd=game_dir, env=env, preexec_fn=game_preexec,
                stdout=subprocess.DEVNULL if quiet else None,
                stderr=subprocess.DEVNULL if quiet else None))

        # Accept what turns up rather than requiring all N. A single game process that fails
        # to boot used to block accept() until it raised, killing a training run outright.
        accepted = accept_per_env(self._srvs, 90, "VecSpireEnv")
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

    def run(self, policy, on_terminal=None, max_steps=None):
        """Drives the envs. ``policy(obs_list, legal_list, env_idx_list) -> list[int]``.

        ``env_idx_list`` matters for RL: a reward arrives only at an episode terminal, so the
        collector must know which env's pending steps to credit. ``on_terminal(env_idx, result)``
        fires as each episode ends. ``max_steps`` stops collection early for a fixed-size
        rollout.
        """
        pending: list[tuple[_Conn, dict]] = []
        t0 = time.time()
        last_activity = time.time()
        self.stalled = 0

        while any(c.alive for c in self.conns):
            # Accumulate a batch: keep polling until every live env has a decision
            # outstanding, or a short deadline expires. Replying to each env the instant it
            # arrives is what kept the mean batch at ~1 and wasted the GPU.
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
                elif time.time() - last_activity > self.STALL_TIMEOUT and not pending:
                    # Nothing from any env for a long time: a game process is wedged rather
                    # than finished. Drop them so the run ends instead of hanging forever.
                    live = [c for c in self.conns if c.alive]
                    print(f"VecSpireEnv: no traffic for {self.STALL_TIMEOUT:.0f}s; "
                          f"dropping {len(live)} stalled env(s)")
                    self.stalled += len(live)
                    for c in live:
                        self._unregister(c)
                    break
                for key, _ in events:
                    conn: _Conn = key.data
                    if not conn.alive:
                        continue
                    for msg in conn.read_messages():
                        t = msg.get("t")
                        if t == "decision":
                            pending.append((conn, msg))
                        elif t == "terminal":
                            res = EpisodeResult(
                                msg["ep"], msg["outcome"], msg["won"], msg["reward"],
                                msg["turns"], msg["steps"], msg["hp_start"], msg["hp_end"])
                            self.results.append(res)
                            if on_terminal is not None:
                                on_terminal(conn.idx, res)
                        elif t == "hello":
                            self.sizes = msg
                        elif t == "done":
                            conn.alive = False
                    if not conn.alive:
                        self._unregister(conn)
                if pending and deadline is None:
                    deadline = time.time() + self.batch_timeout
                if not events and not pending and not any(c.alive for c in self.conns):
                    break

            if not pending:
                if not any(c.alive for c in self.conns):
                    break
                continue

            # One forward pass for every decision collected.
            obs = [m["obs"] for _, m in pending]
            legal = [m["legal"] for _, m in pending]
            idxs = [c.idx for c, _ in pending]
            actions = policy(obs, legal, idxs)
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

    def _unregister(self, conn):
        try:
            self._sel.unregister(conn.sock)
        except (KeyError, ValueError):
            pass
        conn.close()

    @property
    def mean_batch(self) -> float:
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0

    @property
    def win_rate(self) -> float:
        return (sum(r.won for r in self.results) / len(self.results)
                if self.results else 0.0)

    def close(self):
        for c in self.conns:
            self._unregister(c)
        try:
            self._sel.close()
        except Exception:
            pass
        try:
            for s in self._srvs:
                s.close()
        except OSError:
            pass
        for p in self.procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()          # SIGTERM is not always enough for a wedged process
                    p.wait(timeout=5)
                except Exception:
                    pass
