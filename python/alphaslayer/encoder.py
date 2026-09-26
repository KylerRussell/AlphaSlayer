"""Turns ragged episode records into padded, batched arrays for the network.

Kept in numpy and framework-free so it can be tested without torch; ``train.py`` wraps the
output in tensors.

Shapes per batch of B steps:

    hand        (B, H, CARD_TOKEN_WIDTH)   int64, padded, hand_mask (B, H) bool
    enemies     (B, E, ENEMY_WIDTH)        int64, padded, enemy_mask (B, E) bool
    relics      (B, R, 3)                  int64 (idx, counter, melted), padded
    powers      (B, P, 2)                  int64, padded, power_mask
    bags        (B, 3, V)                  float32 dense counts over the card vocab
    globals     (B, G)                     float32, scaled
    actions     (B, A, 4)                  int64 candidate set, action_mask (B, A) bool
    target      (B,)                       int64 index of the chosen action

The draw pile is a bag rather than ordered tokens because the player cannot order it; the
same holds for discard and exhaust, where order carries no decision value.
"""

from __future__ import annotations

import numpy as np

from .format import CARD_TOKEN, CARD_TOKEN_WIDTH, GLOBALS, Step

# Enemy token: scalars + a fixed intent summary. Powers ride in their own ragged tensor.
ENEMY_SCALARS_N = 6
INTENT_SLOTS = 3          # type, damage, repeats for the first intent
INTENT_TYPES = 16         # len(IntentType); counts of each intent type present
ENEMY_WIDTH = ENEMY_SCALARS_N + INTENT_SLOTS + INTENT_TYPES

MAX_EPOWERS = 8           # powers tracked per enemy
# kind, hand_idx, target_idx, card_idx, potion_idx.
#
# potion_idx was sent by the C# all along and never read, so every use_potion action carried
# card_idx = None and ALL POTIONS LOOKED IDENTICAL to the policy -- a Fire Potion and a Block
# Potion were literally the same action. Legacy 4-wide binary records pad to 0 (= no potion).
ACTION_WIDTH = 5


def _pad(rows, width, cap, dtype=np.int64):
    """Pads a ragged list of equal-width rows to ``cap``; returns (array, mask)."""
    out = np.zeros((cap, width), dtype=dtype)
    mask = np.zeros(cap, dtype=bool)
    for i, r in enumerate(rows[:cap]):
        out[i, : len(r)] = r
        mask[i] = True
    return out, mask


def encode_enemy(e: dict) -> list:
    row = [e["monster_idx"], e["hp"], e["max_hp"], e["block"], e["alive"], e["hittable"]]
    intents = e["intents"]
    first = intents[0] if intents else (0, 0, 0)
    row += [first[0], first[1], first[2]]
    counts = [0] * INTENT_TYPES
    for t, _, _ in intents:
        if 0 <= t < INTENT_TYPES:
            counts[t] += 1
    return row + counts


class Encoder:
    """Caps are dataset-driven; oversized states are truncated and counted."""

    def __init__(self, vocab_cards: int, max_hand=16, max_enemies=8, max_relics=32,
                 max_powers=16, max_actions=96):
        self.vocab_cards = vocab_cards
        self.max_hand = max_hand
        self.max_enemies = max_enemies
        self.max_relics = max_relics
        self.max_powers = max_powers
        self.max_actions = max_actions
        self.truncated = dict(hand=0, enemies=0, relics=0, powers=0, actions=0,
                              enemy_powers=0, target_lost=0)

    def encode(self, steps: list[Step]) -> dict:
        b = len(steps)
        hand = np.zeros((b, self.max_hand, CARD_TOKEN_WIDTH), np.int64)
        hand_mask = np.zeros((b, self.max_hand), bool)
        enemies = np.zeros((b, self.max_enemies, ENEMY_WIDTH), np.int64)
        # Enemy powers, as the module docstring always intended ("Powers ride in their own
        # ragged tensor") but only ever implemented for the PLAYER. Without this the model
        # cannot see a boss buffing itself: Ceremonial Beast stacking Strength while dormant,
        # Waterfall Giant's +2 at 50% HP, Kin Priest's Acolyte regeneration -- every Act 1
        # boss is built around a buff the policy could not observe.
        enemy_powers = np.zeros((b, self.max_enemies, MAX_EPOWERS, 2), np.int64)
        enemy_power_mask = np.zeros((b, self.max_enemies, MAX_EPOWERS), bool)
        enemy_mask = np.zeros((b, self.max_enemies), bool)
        relics = np.zeros((b, self.max_relics, 3), np.int64)
        relic_mask = np.zeros((b, self.max_relics), bool)
        powers = np.zeros((b, self.max_powers, 2), np.int64)
        power_mask = np.zeros((b, self.max_powers), bool)
        bags = np.zeros((b, 3, self.vocab_cards), np.float32)
        glob = np.zeros((b, len(GLOBALS)), np.float32)
        actions = np.zeros((b, self.max_actions, ACTION_WIDTH), np.int64)
        action_mask = np.zeros((b, self.max_actions), bool)
        target = np.zeros(b, np.int64)

        for i, s in enumerate(steps):
            if len(s.hand) > self.max_hand:
                self.truncated["hand"] += 1
            h, hm = _pad([list(t) for t in s.hand], CARD_TOKEN_WIDTH, self.max_hand)
            hand[i], hand_mask[i] = h, hm

            if len(s.enemies) > self.max_enemies:
                self.truncated["enemies"] += 1
            e, em = _pad([encode_enemy(x) for x in s.enemies], ENEMY_WIDTH, self.max_enemies)
            enemies[i], enemy_mask[i] = e, em
            for j, x in enumerate(s.enemies[:self.max_enemies]):
                allp = list(x.get("powers") or ())
                if len(allp) > MAX_EPOWERS:
                    self.truncated["enemy_powers"] += 1
                pw = allp[:MAX_EPOWERS]
                for k, (pidx, amt) in enumerate(pw):
                    enemy_powers[i, j, k] = (pidx, amt)
                    enemy_power_mask[i, j, k] = True

            if len(s.relics) > self.max_relics:
                self.truncated["relics"] += 1
            r_pad, r_mask = _pad([list(x) for x in s.relics], 3, self.max_relics)
            relics[i], relic_mask[i] = r_pad, r_mask

            if len(s.player_powers) > self.max_powers:
                self.truncated["powers"] += 1
            p, pm = _pad([list(x) for x in s.player_powers], 2, self.max_powers)
            powers[i], power_mask[i] = p, pm

            for bi, bag in enumerate((s.draw_bag, s.discard_bag, s.exhaust_bag)):
                for cid, cnt in bag:
                    if 0 <= cid < self.vocab_cards:
                        bags[i, bi, cid] = cnt

            glob[i] = self._globals(s)

            if len(s.legal) > self.max_actions:
                self.truncated["actions"] += 1
            for j, a in enumerate(s.legal[: self.max_actions]):
                actions[i, j, : len(a)] = a          # 4-wide legacy records pad to 0
                action_mask[i, j] = True
            # A truncated candidate set can drop the chosen action; fall back to 0 so the
            # batch stays valid, and the truncation counter records that it happened.
            if s.action_idx >= self.max_actions:
                # The chosen action fell outside the padded set, so the label below is a
                # FABRICATION -- action 0, not what was actually played. Training on it is
                # worse than dropping the sample, so this must never be silent.
                self.truncated["target_lost"] += 1
            target[i] = s.action_idx if s.action_idx < self.max_actions else 0

        return dict(hand=hand, hand_mask=hand_mask, enemies=enemies, enemy_mask=enemy_mask,
                    relics=relics, relic_mask=relic_mask, powers=powers, power_mask=power_mask,
                    enemy_powers=enemy_powers, enemy_power_mask=enemy_power_mask,
                    bags=bags, globals=glob, actions=actions, action_mask=action_mask,
                    target=target)

    @staticmethod
    def _globals(s: Step) -> np.ndarray:
        """Scaled so every feature sits roughly in [0, 2] without a learned norm layer."""
        g = {k: s.g(k) for k in GLOBALS}
        hp, mhp = g["hp"], max(1, g["max_hp"])
        return np.array([
            g["turn"] / 10.0,
            g["round"] / 10.0,
            g["phase"] / 5.0,
            float(g["side"]),
            hp / mhp,
            mhp / 100.0,
            g["block"] / 20.0,
            g["energy"] / 5.0,
            g["max_energy"] / 5.0,
            g["stars"] / 5.0,
            g["gold"] / 200.0,
            g["draw_count"] / 20.0,
        ], np.float32)


def step_from_json(obs: dict, legal: list) -> Step:
    """Converts a live-bridge JSON observation into the same Step the binary reader yields.

    Going through Step rather than encoding JSON directly guarantees the training path and the
    acting path see identical features -- a mismatch there is the kind of bug that shows up as
    "trains fine, plays badly" and is very hard to find later.
    """
    p = obs["player"]
    g = (
        obs["turn"], obs["round"], obs["phase_idx"], obs["side_idx"],
        p["hp"], p["max_hp"], p["block"], p["energy"], p["max_energy"],
        p["stars"], p["gold"], obs["draw_count"],
    )
    hand = [tuple(c[k] if not isinstance(c[k], bool) else int(c[k])
                  for k in CARD_TOKEN[:-1]) + (0,) for c in obs["hand"]]
    enemies = [
        {
            "monster_idx": e["monster_idx"], "hp": e["hp"], "max_hp": e["max_hp"],
            "block": e["block"], "alive": int(e["alive"]), "hittable": int(e["hittable"]),
            "powers": [(q["power_idx"], q["amount"]) for q in e["powers"]],
            "intents": [(q["type_idx"], q["damage"], q["repeats"]) for q in e["intents"]],
        }
        for e in obs["enemies"]
    ]
    bag = lambda d: [(int(k), v) for k, v in d.items()]
    return Step(
        ep=0, t=0, globals=g,
        player_powers=[(q["power_idx"], q["amount"]) for q in p["powers"]],
        relics=[(x["relic_idx"], x["counter"], int(x["melted"])) for x in p["relics"]], potions=list(p["potions"]),
        hand=hand,
        draw_bag=bag(obs["draw_bag"]), discard_bag=bag(obs["discard_bag"]),
        exhaust_bag=bag(obs["exhaust_bag"]), enemies=enemies,
        legal=[(a["kind_idx"], a["hand"], a["target"], a["card_idx"],
                a.get("potion_idx", 0) or 0) for a in legal],
        action_idx=0, hp_delta=0,
    )
