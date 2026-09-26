"""Vectorised deck evaluation: how does a given deck fare against a given kind of fight?

The run policy's hardest decision is which card to add, and the reward can currently only see
how MANY cards there are, never whether a particular one is good. This answers the
counterfactual directly -- play the deck you would have with the card, and the deck you would
have without it -- which is both the ground truth for training a deck-strength critic and, on
its own, the signal that decision is missing.

Fights are played by the real combat policy, because a deck's strength is only meaningful
with respect to the player piloting it.
"""

from __future__ import annotations

import os
import selectors
import socket
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field

from .env import DEFAULT_GAME
from .vecenv import _Conn, accept_per_env, game_preexec, listen_per_env


def _spec_payload(spec, room):
    """The deckserve episode spec. ``hp_frac`` is sent only when a spec carries one:
    deckserve heals to full otherwise, which is right for MEASURING a deck and wrong
    for training act-2 fights the policy actually enters at ~35% hp."""
    payload = {"deck": spec.cards, "upgrades": spec.upgrades,
               "relics": spec.relics, "room": room}
    frac = getattr(spec, "hp_frac", None)
    if frac is not None:
        payload["hp_frac"] = round(float(frac), 4)
    return payload


@dataclass
class DeckSpec:
    """One deck to evaluate, and how thoroughly."""
    cards: list[str]
    upgrades: list[int]
    relics: list[str]
    character: str
    fights_per_room: int = 3
    rooms: tuple = ("regular", "elite", "boss")
    tag: str = ""
    # filled in by the evaluator
    wins: dict = field(default_factory=lambda: defaultdict(int))
    total: dict = field(default_factory=lambda: defaultdict(int))
    hp_left: dict = field(default_factory=lambda: defaultdict(float))

    def pending(self):
        """(room, count) still owed, so the evaluator can hand work out one fight at a time."""
        for r in self.rooms:
            missing = self.fights_per_room - self.total[r]
            if missing > 0:
                yield r, missing

    def win_rate(self, room):
        return self.wins[room] / self.total[room] if self.total[room] else None

    def summary(self):
        return {r: (self.wins[r] / self.total[r] if self.total[r] else None) for r in self.rooms}


class VecDeckEval:
    """N game processes evaluating a queue of DeckSpecs against one batched combat policy."""

    # An evaluation fight lasts well under a second, so silence measured in seconds already
    # means a fight is not coming back. The 180s inherited from the run env was sized for
    # whole runs and cost ~360s of dead waiting per labelling batch -- six minutes to salvage
    # the last handful of a thousand fights.
    # Default for MEASUREMENT, where a pool is small and busy. Training passes run many
    # pools in sequence, so a pool can sit idle for minutes and get swapped out; they
    # pass a longer stall_timeout rather than treating a slow wake-up as a stall.
    STALL_TIMEOUT = 25.0

    def __init__(self, n_envs=8, character="IRONCLAD", characters=None, game_dir=DEFAULT_GAME,
                 fixed_fps=5000,
                 ascension=0, seed="DECKEVAL", episodes_per_env=100000, quiet=True,
                 batch_timeout=0.004, turn_cap=50, out_dir="/tmp/alphaslayer_deckeval",
                 act=0, stall_timeout=None):
        self.n_envs = n_envs
        self.batch_timeout = batch_timeout
        # One process plays ONE character (--probe-character is fixed at launch), but a pool
        # can hold a MIX of them, exactly as VecRunEnv does. That matters because work is
        # character-specific: with one pool per character the evaluator could only ever drive
        # that character's envs, so 2 of 30 processes worked while the other 28 spun a core
        # each (--fixed-fps 5000 never sleeps). A mixed pool puts every character's fights in
        # one queue and keeps the whole pool busy.
        roster = characters or [character]
        self.characters = [roster[i % len(roster)] for i in range(n_envs)]
        self.stall_timeout = self.STALL_TIMEOUT if stall_timeout is None else stall_timeout
        self.steps = 0
        self.fights = 0
        self.batch_sizes: list[int] = []
        self._idle: set[int] = set()

        self._srvs = listen_per_env(n_envs)

        env = dict(os.environ, SteamAppId="2868840", SteamGameId="2868840")
        self.procs = []
        for i in range(n_envs):
            self.procs.append(subprocess.Popen(
                [os.path.join(game_dir, "SlayTheSpire2"), "--headless",
                 "--fixed-fps", str(fixed_fps),
                 "--probe=deckserve", f"--probe-out={out_dir}{i}", "--probe-quit",
                 f"--probe-port={self._srvs[i].getsockname()[1]}",
                 f"--probe-episodes={episodes_per_env}",
                 f"--probe-character={self.characters[i]}", f"--probe-seed={seed}{i}",
                 f"--probe-ascension={ascension}", f"--probe-turn-cap={turn_cap}",
                 # REQUIRED. The harness only fills its boss/elite/regular pools when the
                 # encounter mode is "mix"; under the default ("regular") BossEncounters and
                 # EliteEncounters are EMPTY, and deckserve then silently fell back to the
                 # regular pool while still reporting room="boss". Every boss and elite
                 # measurement this class ever produced was a regular fight.
                 "--probe-encounters=mix",
                 # Which act's encounter pools to draw from. The deck is exact, but
                 # "boss" means a DIFFERENT boss in act 2, and that is the whole
                 # point of training on harvested act-2 decks.
                 f"--probe-act={act}"],
                cwd=game_dir, env=env, preexec_fn=game_preexec,
                stdout=subprocess.DEVNULL if quiet else None,
                stderr=subprocess.DEVNULL if quiet else None))

        accepted = accept_per_env(self._srvs, 120, "VecDeckEval")
        self.conns = [accepted[i] for i in sorted(accepted)]
        if not self.conns:
            raise RuntimeError("no deckeval processes connected")

        self._sel = selectors.DefaultSelector()
        for c in self.conns:
            self._sel.register(c.sock, selectors.EVENT_READ, c)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def evaluate(self, specs, policy, progress_every=0, on_terminal=None,
                 with_idx=False):
        """``with_idx`` passes a third argument to ``policy``: the env index each
        decision came from, which a trainer needs to group transitions into episodes.
        ``on_terminal(env_idx, spec, room, msg)`` fires as each fight ends, which is
        where an episode's reward becomes known."""
        """Plays every spec's owed fights. ``policy(obs_list, legal_list) -> [action_idx]``."""
        work = []
        for s in specs:
            for room, n in s.pending():
                work.extend([(s, room)] * n)
        # Interleave rooms so a partial run still yields a usable estimate for every spec
        # rather than a complete one for the first few.
        work.reverse()

        assigned = {}          # conn.idx -> (spec, room)
        pending: list[tuple[_Conn, dict]] = []
        stop_batch = False
        # Envs that asked for an episode while the queue was empty are still waiting for a
        # reply, so hand them work first.
        # A spec can only be played by an env running its character, so the queue is split
        # per character rather than being one flat list.
        by_char = defaultdict(list)
        for item in work:
            by_char[item[0].character].append(item)

        def take(idx):
            """Next fight this env can actually play, or None."""
            q = by_char.get(self.characters[idx])
            return q.pop() if q else None

        for c in list(self.conns):
            got = take(c.idx) if (c.idx in self._idle and c.alive) else None
            if got is not None:
                self._idle.discard(c.idx)
                spec, room = got
                assigned[c.idx] = (spec, room)
                try:
                    c.sock.sendall(_json_line(_spec_payload(spec, room)).encode())
                except OSError:
                    c.alive = False
        t0 = time.time()
        last = time.time()
        # Per-env last-heard-from, so a timeout can drop the ONE env that went quiet
        # instead of abandoning every fight queued for this pool.
        last_seen = {c.idx: last for c in self.conns}

        def work_left():
            """Work that some LIVE env can actually play.

            A process plays one character, so fights for a character whose envs have all died
            are undeliverable. Counting them kept the loop alive waiting for an env that was
            never coming back, and it burned the full stall timeout each iteration before
            reporting "0 in-flight and N queued".
            """
            live_chars = {self.characters[c.idx] for c in self.conns if c.alive}
            return any(v for k, v in by_char.items() if k in live_chars)

        while (not stop_batch and any(c.alive for c in self.conns)
               and (work_left() or assigned or pending)):
            deadline = None
            while True:
                live = sum(1 for c in self.conns if c.alive)
                if live == 0 or (pending and len(pending) >= live):
                    break
                # Nothing left to hand out, nothing in flight, nothing to answer: the batch is
                # finished. Without this the loop sat here until the stall timer fired, which
                # cost a full timeout per pool per iteration -- 50s of pure waiting with zero
                # fights outstanding, reported as a stall that had abandoned nothing.
                if not work_left() and not assigned and not pending:
                    break
                remaining = self.batch_timeout if deadline is None else deadline - time.time()
                if pending and remaining <= 0:
                    break
                events = self._sel.select(timeout=max(0.0005, min(remaining, 0.05)))
                if events:
                    last = time.time()
                    for _k, _ in events:
                        last_seen[_k.data.idx] = last
                elif time.time() - last > self.stall_timeout and not pending:
                    # Only drop envs whose PROCESS has actually exited. This pool is reused
                    # across training iterations, and closing a live, merely-idle env here
                    # made the whole pool unusable after the first call -- every later
                    # evaluation silently measured nothing, which looked like the expert
                    # having no effect rather than like a bug.
                    now = time.time()
                    dead = [c for c in self.conns if c.alive and self.procs[c.idx].poll() is not None]
                    for c in dead:
                        got = assigned.pop(c.idx, None)
                        if got:
                            by_char[got[0].character].append(got)   # another env may play it
                        self._unregister(c)
                    # A silent env holding a fight is the usual case, and abandoning the whole
                    # pool for it cost ~65% of act-2 fights in a curriculum pass (60 requested,
                    # 17-24 completed). Drop just that env, put its fight back on the queue,
                    # and let the rest of the pool finish.
                    # Backstop: any assignment whose env is no longer alive is orphaned and
                    # must be requeued, or the loop cannot make progress.
                    live_idx = {c.idx for c in self.conns if c.alive}
                    for idx in [i for i in assigned if i not in live_idx]:
                        got = assigned.pop(idx)
                        by_char[got[0].character].append(got)
                    stuck = [c for c in self.conns
                             if c.alive and c.idx in assigned
                             and now - last_seen.get(c.idx, now) > self.stall_timeout]
                    if not dead and stuck:
                        for c in stuck:
                            got = assigned.pop(c.idx, None)
                            if got:
                                by_char[got[0].character].append(got)
                            self._unregister(c)
                        print(f"VecDeckEval: dropped {len(stuck)} stalled env(s) after "
                              f"{self.stall_timeout:.0f}s; requeued their fight(s), "
                              f"{sum(1 for c in self.conns if c.alive)} env(s) still running",
                              flush=True)
                        last = now
                        break
                    if not dead:
                        print(f"VecDeckEval: {self.stall_timeout:.0f}s idle with all processes "
                              f"alive and none attributable; abandoning {len(assigned)} "
                              f"in-flight and {sum(len(v) for v in by_char.values())} "
                              f"queued fight(s)", flush=True)
                        # ACTUALLY abandon them. Printing "abandoning" while leaving `assigned`
                        # populated left the outer loop's `work or assigned or pending`
                        # condition permanently true: it re-entered, waited out the timer
                        # again, and livelocked -- six hours on a single iteration, emitting
                        # this same line every 25 seconds. The message said one thing and the
                        # code did another.
                        assigned.clear()
                        by_char.clear()
                        stop_batch = True
                    break
                for key, _ in events:
                    conn: _Conn = key.data
                    if not conn.alive:
                        continue
                    for msg in conn.read_messages():
                        t = msg.get("t")
                        if t == "need_episode":
                            got = take(conn.idx)
                            if got is not None:
                                spec, room = got
                                assigned[conn.idx] = (spec, room)
                                try:
                                    conn.sock.sendall((_json_line({
                                        **_spec_payload(spec, room)})).encode())
                                except OSError:
                                    by_char[spec.character].append((spec, room))
                                    assigned.pop(conn.idx, None)
                                    conn.alive = False
                            else:
                                # Park the env instead of stopping it. Sending "stop" makes the
                                # game process EXIT, which killed the pool after the first
                                # evaluate() call and silently produced zero fights on every
                                # call after that. It keeps its unanswered request until the
                                # next batch of work arrives.
                                self._idle.add(conn.idx)
                        elif t == "decision":
                            pending.append((conn, msg))
                        elif t == "terminal":
                            got = assigned.pop(conn.idx, None)
                            if got:
                                spec, room = got
                                spec.total[room] += 1
                                spec.wins[room] += int(msg["won"])
                                spec.hp_left[room] += msg["hp_end"] / max(1, msg["max_hp"])
                                self.fights += 1
                                if on_terminal is not None:
                                    on_terminal(conn.idx, spec, room, msg)
                                if progress_every and self.fights % progress_every == 0:
                                    print(f"    {self.fights} fights, {time.time()-t0:.0f}s",
                                          flush=True)
                        elif t == "done":
                            conn.alive = False
                    if not conn.alive:
                        # Put its fight back. A game that crashes mid-fight closes the socket,
                        # and leaving the assignment behind orphans work nothing can finish:
                        # the pool then waited out the whole stall timeout and reported the
                        # stall as "none attributable", because the env holding it was no
                        # longer alive to be blamed. That cost ~80s per curriculum iteration.
                        got = assigned.pop(conn.idx, None)
                        if got:
                            by_char[got[0].character].append(got)
                        self._unregister(conn)
                if pending and deadline is None:
                    deadline = time.time() + self.batch_timeout
                if not events and not pending and not any(c.alive for c in self.conns):
                    break

            if not pending:
                if not any(c.alive for c in self.conns):
                    break
                continue
            _obs = [m["obs"] for _, m in pending]
            _legal = [m["legal"] for _, m in pending]
            actions = (policy(_obs, _legal, [c.idx for c, _ in pending]) if with_idx
                       else policy(_obs, _legal))
            for (conn, _), a in zip(pending, actions):
                conn.send_action(a)
            self.steps += len(pending)
            self.batch_sizes.append(len(pending))
            pending.clear()

        undeliverable = sum(len(v) for k, v in by_char.items()
                            if k not in {self.characters[c.idx] for c in self.conns if c.alive})
        if undeliverable:
            alive = sum(1 for c in self.conns if c.alive)
            print(f"VecDeckEval: {undeliverable} fight(s) undeliverable -- their character has "
                  f"no live env left ({alive}/{len(self.conns)} envs alive; game processes "
                  f"crash and are not replaced)", flush=True)
        self.wall = time.time() - t0
        return specs

    def _unregister(self, conn):
        try:
            self._sel.unregister(conn.sock)
        except (KeyError, ValueError):
            pass
        conn.close()

    def close(self):
        for c in self.conns:
            if c.alive and c.idx in self._idle:
                try:
                    c.sock.sendall(_json_line({"stop": True}).encode())
                except OSError:
                    pass
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
    def mean_batch(self):
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0


def _json_line(d):
    import json
    return json.dumps(d) + "\n"
