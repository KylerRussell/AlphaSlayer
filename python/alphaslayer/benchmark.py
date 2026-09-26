"""A combat regression benchmark built from decks the run policy actually plays.

The original gate measured the combat policy on random Pandora's-Box decks. That is a
train/eval mismatch: the policy is trained on curated run decks and gated on random ones,
so the gate defends performance on a distribution that is never played, and its KL-to-frozen
-reference term actively resists specialising to the one that is.

This module replaces it. Entry decks are harvested from live runs (train_run.py
--harvest-decks), tagged with the room they were about to fight, and replayed against that
same room type. The aggregate is weighted by where runs ACTUALLY die -- measured at 69% boss
/ 18% monster / 13% elite over 20 iterations of act-1 training -- so the single number the
gate compares is dominated by the fights that end runs, not the ones that never do.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

from .deckeval import DeckSpec, VecDeckEval

# fight_end room names -> the room ids deckserve understands.
ROOM_MAP = {"Boss": "boss", "Elite": "elite", "Monster": "regular"}

# Share of run-ending deaths, measured over iterations 99-118 of the act-1 run (n=762).
# Weighting the gate by this makes it a measure of "how often does combat lose the run",
# which is the quantity we actually care about defending.
DEATH_WEIGHTS = {"boss": 0.69, "regular": 0.18, "elite": 0.13}


def build(harvest_path, out_path, per_room=60, seed=20260901, min_deck=6):
    """Samples a FIXED benchmark set from a harvest file.

    Fixed matters as much here as the fixed seed does in the old gate: two gate measurements
    must differ only by the policy, so the deck set is drawn once and written to disk rather
    than resampled per evaluation.
    """
    recs = []
    seen = set()
    with open(harvest_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            room = ROOM_MAP.get(r.get("room"))
            if room is None or len(r.get("cards") or []) < min_deck:
                continue
            # Dedupe identical (character, room, deck, relics). Early runs repeat the
            # starting deck constantly; without this the benchmark is mostly one deck.
            key = (r["character"], room,
                   tuple(sorted(zip(r["cards"], r["upgrades"]))), tuple(sorted(r["relics"])))
            if key in seen:
                continue
            seen.add(key)
            r["room"] = room
            recs.append(r)

    by = defaultdict(list)
    for r in recs:
        by[(r["character"], r["room"])].append(r)

    rng = random.Random(seed)
    chosen = []
    for k in sorted(by):
        pool = by[k]
        rng.shuffle(pool)
        chosen.extend(pool[:per_room])

    bench = {"version": 1, "seed": seed, "per_room": per_room,
             "weights": DEATH_WEIGHTS, "decks": chosen}
    with open(out_path, "w") as fh:
        json.dump(bench, fh)
    counts = defaultdict(int)
    for r in chosen:
        counts[(r["character"], r["room"])] += 1
    return bench, dict(counts), len(recs)


def load(path):
    with open(path) as fh:
        return json.load(fh)


class BenchmarkPool:
    """Per-character deck-eval pools, kept alive across gate evaluations.

    Process startup dominates a short evaluation, so the pools are created once and reused;
    a gate that spent most of its wall clock booting games would be run less often, which is
    the opposite of what a regression gate is for.
    """

    def __init__(self, bench, n_envs=4, ascension=0, seed="BENCH", turn_cap=50):
        self.bench = bench
        self.n_envs, self.ascension, self.seed, self.turn_cap = n_envs, ascension, seed, turn_cap
        self._pools = {}

    def _pool(self, character):
        if character not in self._pools:
            self._pools[character] = VecDeckEval(
                n_envs=self.n_envs, character=character, ascension=self.ascension,
                # Fixed per character, so the same benchmark deck meets the same fight every
                # time it is replayed. This is what makes the comparison paired.
                seed=f"{self.seed}{character[:3]}", turn_cap=self.turn_cap,
                out_dir=f"/tmp/alphaslayer_bench_{character}_")
        return self._pools[character]

    def measure(self, combat_policy, fights_per_deck=2, close_after_each=False):
        """Returns (weighted_win_rate, standard_error, per_room dict, n_fights).

        ``close_after_each`` frees each character's game processes as soon as that character
        is done. Used when this runs as a training gate: holding five pools open would keep
        ~15 extra games resident alongside the run envs for the whole run, and paying pool
        startup once every ten iterations is much cheaper than the memory.
        """
        by_char = defaultdict(list)
        for r in self.bench["decks"]:
            by_char[r["character"]].append(r)

        wins = defaultdict(int)
        total = defaultdict(int)
        by_cr = defaultdict(lambda: [0, 0])     # (character, room) -> [wins, total]
        for character, recs in sorted(by_char.items()):
            specs = [DeckSpec(r["cards"], r["upgrades"], r["relics"], character,
                              fights_per_deck, rooms=(r["room"],), tag=r["room"])
                     for r in recs]
            self._pool(character).evaluate(specs, combat_policy)
            if close_after_each:
                try:
                    self._pools.pop(character).close()
                except Exception:
                    pass
            for s in specs:
                for room in s.rooms:
                    wins[room] += s.wins[room]
                    total[room] += s.total[room]
                    cell = by_cr[(character, room)]
                    cell[0] += s.wins[room]
                    cell[1] += s.total[room]

        weights = self.bench.get("weights", DEATH_WEIGHTS)
        per_room, num, den, var = {}, 0.0, 0.0, 0.0
        for room, w in weights.items():
            n = total.get(room, 0)
            if not n:
                continue
            p = wins[room] / n
            per_room[room] = (p, n)
            num += w * p
            den += w
            var += (w ** 2) * p * (1 - p) / n
        if den == 0:
            raise RuntimeError("realistic combat benchmark measured no fights; refusing to "
                               "report a win rate of 0 as if it were a measurement")
        # Renormalise in case a room is missing from the harvest, so the number stays a
        # win rate rather than silently shrinking toward zero.
        self.by_character = {k: (w / n, n) for k, (w, n) in by_cr.items() if n}
        return num / den, (var ** 0.5) / den, per_room, sum(total.values())

    def close(self):
        for p in self._pools.values():
            try:
                p.close()
            except Exception:
                pass
        self._pools.clear()
