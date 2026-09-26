"""A running record of how far runs get with each card in the deck.

Motivation: some cards are actively bad, and nothing in the reward says so directly. A card
that keeps appearing in decks that die on floor 12 is a card worth removing, and that is
information the run policy, the critic, and the removal choice can all use.

Kept deliberately simple -- a JSON table of counts and sums that survives across runs.

The confound is worth stating plainly rather than hiding: this credits a card for the WHOLE
run, including floors survived before it was ever picked up, so a card taken late inherits a
deep run it had no part in. That biases every late-game card upward. The number is therefore
useful for comparing cards that enter at similar times, and misleading if read as causal.
``floors_after`` corrects for it where the caller can supply the floor a card was acquired on.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


class CardStats:
    def __init__(self, path="card_stats.json"):
        self.path = path
        self.n = defaultdict(int)            # runs whose final deck held this card
        self.floor_sum = defaultdict(float)  # floors reached, summed
        self.win_sum = defaultdict(float)
        self.after_n = defaultdict(int)      # observations with a known acquisition floor
        self.after_sum = defaultdict(float)  # floors survived AFTER acquiring it
        if os.path.exists(path):
            self.load()

    def record_run(self, deck_cards, floors, won, acquired_floor=None):
        """One finished run. ``deck_cards`` is the final deck as card-id strings."""
        for cid in set(deck_cards):
            self.n[cid] += 1
            self.floor_sum[cid] += floors
            self.win_sum[cid] += float(won)
            if acquired_floor and cid in acquired_floor:
                self.after_n[cid] += 1
                self.after_sum[cid] += max(0.0, floors - acquired_floor[cid])

    def mean_floor(self, cid):
        return self.floor_sum[cid] / self.n[cid] if self.n[cid] else None

    def mean_floors_after(self, cid):
        return self.after_sum[cid] / self.after_n[cid] if self.after_n[cid] else None

    def worst(self, min_n=15, k=10):
        """Cards with the lowest mean floor, among those seen often enough to mean anything."""
        rows = [(c, self.mean_floor(c), self.n[c]) for c in self.n if self.n[c] >= min_n]
        return sorted(rows, key=lambda r: r[1])[:k]

    def best(self, min_n=15, k=10):
        rows = [(c, self.mean_floor(c), self.n[c]) for c in self.n if self.n[c] >= min_n]
        return sorted(rows, key=lambda r: -r[1])[:k]

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"n": dict(self.n), "floor_sum": dict(self.floor_sum),
                       "win_sum": dict(self.win_sum), "after_n": dict(self.after_n),
                       "after_sum": dict(self.after_sum)}, fh)
        os.replace(tmp, self.path)   # atomic: a crash mid-write cannot truncate the table

    def load(self):
        try:
            d = json.load(open(self.path))
        except Exception:
            return
        for k, tgt in (("n", self.n), ("floor_sum", self.floor_sum), ("win_sum", self.win_sum),
                       ("after_n", self.after_n), ("after_sum", self.after_sum)):
            for c, v in (d.get(k) or {}).items():
                tgt[c] = v
