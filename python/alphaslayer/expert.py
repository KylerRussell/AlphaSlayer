"""Counterfactual scoring of run-level pickups, by actually playing the fights.

At a card reward the question is not "is this card good" in the abstract but "is the deck I
would have with it better than the deck I have". That is answerable directly: build both
loadouts, play each against a sample of normals, elites and bosses, and compare health
preserved. The same question, and the same answer, applies to a shop card or relic.

This is the expert in a DAgger-style scheme. It is accurate and expensive, so its fight budget
is annealed toward zero over training while the critic learns to predict what it would have
said; after that the critic answers for free, and eventually the run policy is expected to
carry the judgement itself -- which is the only way it can reason about combinations of relics
and cards that a per-candidate score cannot express.
"""

from __future__ import annotations

from collections import defaultdict

from .deckeval import DeckSpec, VecDeckEval

ROOMS = ("regular", "elite", "boss")


class Candidate:
    """One possible loadout after a decision, and what it measured."""

    __slots__ = ("tag", "action_index", "cards", "upgrades", "relics", "hp", "n")

    def __init__(self, tag, action_index, cards, upgrades, relics):
        self.tag = tag
        self.action_index = action_index
        self.cards = cards
        self.upgrades = upgrades
        self.relics = relics
        self.hp = {}
        self.n = {}

    def strength(self, weights=(0.34, 0.33, 0.33)):
        """Mean health preserved across room types; None when nothing was measured."""
        vals, ws = [], []
        for r, w in zip(ROOMS, weights):
            if self.n.get(r):
                vals.append(self.hp[r] / self.n[r])
                ws.append(w)
        return sum(v * w for v, w in zip(vals, ws)) / sum(ws) if ws else None


def build_card_reward_candidates(obs, legal):
    """Current deck plus one variant per offered card. Skip is the deck as it stands."""
    p = obs.get("player", obs)
    cards = p.get("deck_cards") or []
    if not cards:
        return None
    base = [(c["card"], int(c.get("up", 0))) for c in cards]
    relics = [r.get("relic", "") for r in (p.get("relics") or []) if r.get("relic")]

    out = []
    for i, act in enumerate(legal):
        if act.get("kind") == "card" and act.get("id"):
            deck = base + [(act["id"], 0)]
            out.append(Candidate(act["id"], i, [c for c, _ in deck], [u for _, u in deck], relics))
        elif act.get("kind") == "alt":
            out.append(Candidate("SKIP", i, [c for c, _ in base], [u for _, u in base], relics))
    return out or None


def build_shop_candidates(obs, legal, max_items=8):
    """Buying nothing, plus one variant per affordable card or relic.

    Potions are left to the run policy: a potion is consumable, so its worth is not a property
    of the loadout this critic scores. (Preferring one potion over another, and swapping a weak
    one out when a better appears, is something the run policy has to learn from run outcomes.)

    Card removal IS scored, using a default target: a curse if the deck holds one, otherwise a
    basic Strike or Defend. That is the removal a competent player makes almost every time, so
    it is a fair stand-in for a decision the shop action does not itself specify.
    """
    p = obs.get("player", obs)
    cards = p.get("deck_cards") or []
    if not cards:
        return None
    base = [(c["card"], int(c.get("up", 0))) for c in cards]
    relics = [r.get("relic", "") for r in (p.get("relics") or []) if r.get("relic")]

    out = [Candidate("LEAVE", _leave_index(legal),
                     [c for c, _ in base], [u for _, u in base], relics)]
    for i, act in enumerate(legal):
        if len(out) > max_items:
            break
        if not act.get("affordable"):
            continue
        kind, ident = act.get("kind"), act.get("id")
        if kind == "card" and ident:
            deck = base + [(ident, 0)]
            out.append(Candidate(ident, i, [c for c, _ in deck], [u for _, u in deck], relics))
        elif kind == "relic" and ident:
            out.append(Candidate(ident, i, [c for c, _ in base], [u for _, u in base],
                                 relics + [ident]))
        elif kind == "remove_card":
            drop = pick_removal_target(base)
            if drop is not None:
                deck = base[:drop] + base[drop + 1:]
                out.append(Candidate("REMOVE", i, [c for c, _ in deck],
                                     [u for _, u in deck], relics))
    return out if len(out) > 1 else None


CURSE_HINTS = ("CURSE", "REGRET", "SHAME", "DOUBT", "DEBT", "INJUR", "PARASITE",
               "NECRONOMICURSE", "ASCENDERS_BANE", "CLUMSY", "WRITHE", "PAIN", "DECAY")
BASIC_HINTS = ("STRIKE_", "DEFEND_")


def pick_removal_target(base):
    """Index of the card a removal would take: a curse first, then a basic Strike/Defend.

    Deliberately a fixed rule rather than a search. Which card to remove is its own decision
    and the shop action does not carry it; this encodes the choice a competent player makes
    nearly every time, so the SCORE for 'buy a removal' reflects a realistic use of it rather
    than the best or worst case.
    """
    for i, (cid, _up) in enumerate(base):
        u = cid.upper()
        if any(h in u for h in CURSE_HINTS):
            return i
    for i, (cid, up) in enumerate(base):
        u = cid.upper()
        if up == 0 and any(u.startswith(h) or h in u for h in BASIC_HINTS):
            return i
    return None


def _leave_index(legal):
    for i, a in enumerate(legal):
        if a.get("kind") == "leave":
            return i
    return len(legal) - 1


class ExpertEvaluator:
    """Owns one deck-evaluation pool per character and scores batches of candidate sets."""

    def __init__(self, characters, envs_per_char=4, ascension=0, seed="EXPERT",
                 game_dir=None, turn_cap=50):
        self.characters = list(characters)
        self.envs_per_char = envs_per_char
        self.ascension = ascension
        self.seed = seed
        self.turn_cap = turn_cap
        self._pools = {}
        self.fights = 0
        self.wall = 0.0

    def _pool(self, character):
        if character not in self._pools:
            kw = dict(n_envs=self.envs_per_char, character=character,
                      ascension=self.ascension, seed=f"{self.seed}{character[:3]}",
                      turn_cap=self.turn_cap,
                      out_dir=f"/tmp/alphaslayer_expert_{character}_")
            self._pools[character] = VecDeckEval(**kw)
        return self._pools[character]

    def score(self, items, combat_policy, fights_per_room):
        """items: list of (character, ctx, [Candidate]). Fills in each Candidate's hp/n."""
        if fights_per_room <= 0:
            return items
        by_char = defaultdict(list)
        for character, _ctx, cands in items:
            by_char[character].extend(cands)

        for character, cands in by_char.items():
            specs = [DeckSpec(c.cards, c.upgrades, c.relics, character,
                              fights_per_room, tag=c.tag) for c in cands]
            pool = self._pool(character)
            alive = sum(1 for c in pool.conns if c.alive)
            codes = [pr.poll() for pr in pool.procs]
            if alive == 0:
                print(f"  expert: {character} pool DEAD (conns={len(pool.conns)} "
                      f"exit codes={codes})", flush=True)
            pool.evaluate(specs, combat_policy)
            measured = sum(1 for s in specs if any(s.total[r] for r in ROOMS))
            if measured == 0 and specs:
                print(f"  expert: {character} evaluated {len(specs)} candidates but measured "
                      f"NONE (pool alive={sum(1 for c in pool.conns if c.alive)})", flush=True)
            for c, s in zip(cands, specs):
                for r in ROOMS:
                    c.hp[r] = s.hp_left[r]
                    c.n[r] = s.total[r]
            self.fights += pool.fights
            self.wall += pool.wall
            pool.fights = 0
        return items

    def close(self):
        for p in self._pools.values():
            try:
                p.close()
            except Exception:
                pass
        self._pools.clear()
