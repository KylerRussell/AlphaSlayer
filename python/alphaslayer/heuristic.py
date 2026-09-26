"""A scripted run policy encoding ordinary competent Slay the Spire play.

Two purposes. It is a BASELINE -- until now we had no idea what a sensible non-learned policy
scores, so there was no way to tell whether the trained policy's numbers were poor or near the
practical ceiling. And it is a starting DISTRIBUTION: it skips cards, buys removals and rests
when hurt, so those actions appear in the data at all, which a policy that has collapsed to
one option can never discover on its own.

The rules are deliberately plain and are not tuned against results.
"""

from __future__ import annotations

import random

# Card rarity as the game reports it in a reward option.
COMMON, UNCOMMON, RARE = 0, 1, 2

CURSE_HINTS = ("CURSE", "REGRET", "SHAME", "DOUBT", "DEBT", "INJUR", "PARASITE",
               "ASCENDERS_BANE", "CLUMSY", "WRITHE", "PAIN", "DECAY", "NECRONOMICURSE")
BASIC_HINTS = ("STRIKE_", "DEFEND_")


def _is_curse(cid):
    u = (cid or "").upper()
    return any(h in u for h in CURSE_HINTS)


def _is_basic(cid):
    u = (cid or "").upper()
    return any(h in u for h in BASIC_HINTS)


class HeuristicRunPolicy:
    """Answers every run-level decision with a fixed rule."""

    def __init__(self, seed=0, deck_soft_cap=16, rest_hp=0.55, elite_hp=0.75):
        self.rng = random.Random(seed)
        self.deck_soft_cap = deck_soft_cap
        self.rest_hp = rest_hp
        self.elite_hp = elite_hp

    def __call__(self, kind, obs, legal):
        p = obs.get("player", obs)
        hp = (p.get("hp", 0) or 0) / max(1, p.get("max_hp", 1) or 1)
        deck = int(p.get("deck_size", 0) or 0)
        fn = getattr(self, f"_{kind}", None)
        try:
            i = fn(obs, legal, p, hp, deck) if fn else 0
        except Exception:
            i = 0
        return max(0, min(int(i), len(legal) - 1)) if legal else 0

    # ---- map ----
    def _travel(self, obs, legal, p, hp, deck):
        # Rest when hurt; otherwise prefer nodes that build the deck. Elites are worth taking
        # only from a healthy position, since they cost health for a relic.
        def score(a):
            t = (a.get("point_type") or "").lower()
            if t == "restsite":
                return 10 if hp < self.rest_hp else 3
            if t == "elite":
                return 6 if hp >= self.elite_hp and deck >= 12 else -5
            if t in ("unknown", "ancient"):
                return 5
            if t == "shop":
                return 4 if (p.get("gold", 0) or 0) >= 150 else 2
            if t == "treasure":
                return 5
            if t == "monster":
                return 2
            return 1
        return max(range(len(legal)), key=lambda i: score(legal[i]))

    # ---- card rewards ----
    def _card_reward(self, obs, legal, p, hp, deck):
        skip = next((i for i, a in enumerate(legal) if a.get("kind") == "alt"), None)
        cards = [(i, a) for i, a in enumerate(legal) if a.get("kind") == "card"]
        if not cards:
            return skip if skip is not None else 0
        best_i, best_a = max(cards, key=lambda ia: int(ia[1].get("rarity", 0) or 0))
        rarity = int(best_a.get("rarity", 0) or 0)
        # Past the soft cap only take cards that are actually worth diluting for. Knowing when
        # to skip is most of deck building; taking everything is the commonest way to lose.
        if deck >= self.deck_soft_cap and rarity < UNCOMMON and skip is not None:
            return skip
        if deck >= self.deck_soft_cap + 6 and rarity < RARE and skip is not None:
            return skip
        return best_i

    # ---- campfire ----
    def _rest(self, obs, legal, p, hp, deck):
        heal = next((i for i, a in enumerate(legal) if a.get("option") == "HEAL"), None)
        smith = next((i for i, a in enumerate(legal) if a.get("option") == "SMITH"), None)
        if hp < self.rest_hp and heal is not None:
            return heal
        if smith is not None:
            return smith
        return heal if heal is not None else 0

    # ---- shop ----
    def _shop(self, obs, legal, p, hp, deck):
        gold = int(p.get("gold", 0) or 0)
        aff = [(i, a) for i, a in enumerate(legal) if a.get("affordable")]
        leave = next((i for i, a in enumerate(legal) if a.get("kind") == "leave"), len(legal) - 1)
        # Removal first: thinning the deck is the strongest thing gold buys.
        for i, a in aff:
            if a.get("kind") == "remove_card" and deck > 12:
                return i
        for i, a in aff:
            if a.get("kind") == "relic":
                return i
        if deck < self.deck_soft_cap:
            picks = [(i, a) for i, a in aff if a.get("kind") == "card"]
            if picks:
                return max(picks, key=lambda ia: int(ia[1].get("rarity", 0) or 0))[0]
        if gold >= 250:
            for i, a in aff:
                if a.get("kind") == "potion":
                    return i
        return leave

    # ---- events ----
    def _event(self, obs, legal, p, hp, deck):
        ok = [i for i, a in enumerate(legal)
              if not a.get("locked") and not a.get("will_kill")]
        return ok[0] if ok else 0

    # ---- potions ----
    def _potion_gate(self, obs, legal, p, hp, deck):
        room = (obs.get("room_type") or "").lower()
        allow = next((i for i, a in enumerate(legal) if a.get("kind") == "allow"), 1)
        deny = next((i for i, a in enumerate(legal) if a.get("kind") == "deny"), 0)
        # Spend potions where they decide the fight, hold them otherwise.
        return allow if room in ("elite", "boss") or hp < 0.4 else deny

    def _potion_ooc(self, obs, legal, p, hp, deck):
        hold = next((i for i, a in enumerate(legal) if a.get("kind") == "hold"), len(legal) - 1)
        return hold

    # ---- card selection (smith target, removal target) ----
    def _card_select(self, obs, legal, p, hp, deck):
        cards = [(i, a) for i, a in enumerate(legal) if a.get("kind") == "select_card"]
        if not cards:
            return 0
        for i, a in cards:                      # a curse is always the right thing to remove
            if _is_curse(a.get("id")):
                return i
        for i, a in cards:                      # then an unupgraded basic
            if _is_basic(a.get("id")) and not int(a.get("upgrade", 0) or 0):
                return i
        return cards[0][0]

    def _treasure(self, obs, legal, p, hp, deck):
        return 0
