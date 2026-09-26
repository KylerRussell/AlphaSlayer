"""The RUN policy: every out-of-combat decision in a run.

Why one network rather than a head per decision type: the decisions are wildly heterogeneous
(pick a map node, pick a card, pick a campfire option, buy something, gate potions) and each
has a variable number of options. That is the same problem the combat net already solves with
action embeddings -- encode each legal option as a vector, score it against the state, softmax
over whatever set arrived. Adding a new decision kind then costs a feature mapping, not a new
head and a new optimiser.

The state is the RUN state, not a combat state: hp, gold, the deck as a bag, relics, act and
floor. Nothing per-turn, because none of these decisions happen inside a turn.
"""

from __future__ import annotations

import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

# Decision kinds, in the order the C# side emits them.
DKINDS = ["travel", "card_reward", "rest", "event", "shop",
          "potion_gate", "card_select", "treasure", "potion_ooc"]
DKIND_IX = {k: i for i, k in enumerate(DKINDS)}

# Action kinds across every decision type. "leave"/"hold"/"deny" are the opt-out actions and
# are deliberately distinct from each other: declining a shop is not the same decision as
# declining to drink.
AKINDS = ["travel", "card", "alt", "relic", "potion", "remove_card", "leave",
          "allow", "deny", "drink", "hold", "select_card", "select_done", "option"]
AKIND_IX = {k: i for i, k in enumerate(AKINDS)}

TOKENS = 4096      # hashed string ids (campfire options, event keys, alternatives)
_TOK_HEAL = 3987   # crc32-based; the assert under _tok guards against silent drift
NUM = 26           # numeric feature slots per action
POINT_TYPES = 9    # MapPointType enum width


def _tok(s: str) -> int:
    """Stable hash of a string id into the token table.

    Event options are identified by a localisation key and there are hundreds of them; a
    hashed embedding gives each a stable identity without hard-coding an enumeration that
    would go stale the next time the game patches.

    MUST be deterministic across processes. This used Python's builtin ``hash``, which is
    salted per interpreter (PYTHONHASHSEED), so every new process mapped campfire options,
    event keys, map point types and skip alternatives onto DIFFERENT embedding rows. A
    checkpoint kept its card and relic knowledge across a resume but lost everything it had
    learned about rest, travel, events and skipping: on 2026-09-15 a resume dropped act-1
    clears from 0.60 to 0.17 and flipped campfires from HEAL 100% to SMITH 75% with weights
    that loaded bit-identically. The same scrambling silently affected every evaluation of a
    run-policy checkpoint made in a process other than the one that trained it.
    """
    if not s:
        return 0
    return zlib.crc32(s.encode("utf-8")) % (TOKENS - 1) + 1


# Regression guard: if this ever changes, every run-policy checkpoint's token rows are stale.
assert _tok("HEAL") == _TOK_HEAL, "_tok is no longer stable; see its docstring"


def _lookahead(obs, point_index, horizon=6, decay=0.75):
    """Discounted counts of each point type reachable from a map point.

    A travel decision is not really about the node you step on; it is about what that node
    leads to. Without this the policy can only see "this one is an elite" and never "this
    branch has two campfires and a shop before the boss".
    """
    pts = obs.get("points") or []
    if not (0 <= point_index < len(pts)):
        return [0.0] * POINT_TYPES + [0.0]
    counts = [0.0] * POINT_TYPES
    frontier = [point_index]
    seen = {point_index}
    w = 1.0
    for _ in range(horizon):
        nxt = []
        for i in frontier:
            for j in pts[i].get("children", ()):
                if j in seen or not (0 <= j < len(pts)):
                    continue
                seen.add(j)
                t = pts[j].get("type_idx", 0)
                if 0 <= t < POINT_TYPES:
                    counts[t] += w
                nxt.append(j)
        if not nxt:
            break
        frontier = nxt
        w *= decay
    boss = obs.get("boss_point", -1)
    rows_to_boss = 0.0
    if 0 <= boss < len(pts):
        rows_to_boss = (pts[boss].get("row", 0) - pts[point_index].get("row", 0)) / 20.0
    return counts + [rows_to_boss]


def encode_actions(kinds, legal_lists, device):
    """Ragged legal-action sets -> padded tensors.

    Returns (akind, token, card, relic, potion, num, mask), each (B, A[, NUM]).
    """
    B = len(legal_lists)
    A = max((len(l) for l in legal_lists), default=1) or 1

    akind = torch.zeros(B, A, dtype=torch.long)
    token = torch.zeros(B, A, dtype=torch.long)
    card = torch.zeros(B, A, dtype=torch.long)
    relic = torch.zeros(B, A, dtype=torch.long)
    potion = torch.zeros(B, A, dtype=torch.long)
    num = torch.zeros(B, A, NUM)
    mask = torch.zeros(B, A, dtype=torch.bool)

    for b, (dk, acts) in enumerate(zip(kinds, legal_lists)):
        for a, act in enumerate(acts):
            mask[b, a] = True
            k = act.get("kind", "")
            if dk == "rest":
                k = "leave" if act.get("option") == "LEAVE" else "option"
                token[b, a] = _tok(act.get("option", ""))
            elif dk == "event":
                k = "option"
                token[b, a] = _tok(act.get("key", ""))
            elif dk == "travel":
                k = "travel"
                token[b, a] = _tok(str(act.get("point_type", "")))
            else:
                token[b, a] = _tok(str(act.get("id", "")))
            akind[b, a] = AKIND_IX.get(k, AKIND_IX["option"])

            card[b, a] = max(0, int(act.get("card_idx", 0) or 0))
            relic[b, a] = max(0, int(act.get("relic_idx", 0) or 0))
            potion[b, a] = max(0, int(act.get("potion_idx", 0) or 0))
            # model_idx is the shop/treasure field; route it by the action's own kind so a
            # relic id never lands in the card table.
            mi = int(act.get("model_idx", 0) or 0)
            if mi > 0:
                if k == "relic":
                    relic[b, a] = mi
                elif k == "potion":
                    potion[b, a] = mi
                elif k == "card":
                    card[b, a] = mi

            n = num[b, a]
            n[0] = float(act.get("cost", 0) or 0) / 100.0
            n[1] = float(bool(act.get("affordable", False)))
            n[2] = float(bool(act.get("on_sale", False)))
            n[3] = float(act.get("rarity", -1) or -1) / 4.0
            n[4] = float(act.get("upgrade", 0) or 0)
            n[5] = float(act.get("point_type_idx", 0) or 0) / 8.0
            n[6] = float(act.get("row", 0) or 0) / 16.0
            n[7] = float(act.get("col", 0) or 0) / 6.0
            n[8] = float(bool(act.get("enabled", True)))
            n[9] = float(bool(act.get("locked", False)))
            n[10] = float(bool(act.get("will_kill", False)))
            n[11] = float(bool(act.get("proceed", False)))
            n[12] = float(bool(act.get("chosen", False)))
            n[13] = float(bool(act.get("gated", False)))
            n[14] = float(bool(act.get("usable", False)))
            n[15] = float(k in ("leave", "hold", "deny", "select_done"))
        # Travel lookahead needs the map, so it is filled per-batch-row after the loop above.
        if dk == "travel":
            for a, act in enumerate(acts):
                pi = int(act.get("point", -1))
                num[b, a, 16:16 + POINT_TYPES + 1] = torch.tensor(
                    _lookahead(_OBS_CACHE.get(id(acts), {}), pi), dtype=torch.float32)

    return (akind.to(device), token.to(device), card.to(device), relic.to(device),
            potion.to(device), num.to(device), mask.to(device))


# encode_actions needs the map that came with a travel decision; the legal list alone does not
# carry it. Rather than thread it through every signature, the caller stashes it here for the
# duration of one encode call.
_OBS_CACHE: dict[int, dict] = {}


def bind_obs(legal_lists, obs_lists):
    _OBS_CACHE.clear()
    for l, o in zip(legal_lists, obs_lists):
        _OBS_CACHE[id(l)] = o


# 12 original scalars + a 3-way one-hot for the room the decision is about.
#
# room_type and encounter were present in the observation JSON all along and simply never
# read. That made potion_gate -- "may this fight spend potions?" -- a decision taken BLIND to
# which fight it gates, so the policy could not hold a potion for the boss even in principle,
# and converged to allow=100%. The campfire head has the same shape of problem.
STATE_SCALARS = 15
_AUDIT = {}   # set to a dict of counters to audit feature population
ROOM_IX = {"Monster": 0, "Elite": 1, "Boss": 2}


def encode_state(kinds, obs_list, device, max_relics=16, max_deck=48, max_potions=5):
    """Run-level state: scalars, the relic set, the deck as a bag, the potions HELD, and
    which cards are UPGRADED.

    The last two were blind spots. deck_bag groups by card id, so an upgraded Strike and a
    plain one were the same entry and only a scalar total of upgrades survived. And potions
    entered as len(potions)/5 -- a count -- so potion_gate ("may this fight spend potions?")
    was decided without knowing what the potions were.
    """
    B = len(obs_list)
    scal = torch.zeros(B, STATE_SCALARS)
    dk = torch.zeros(B, dtype=torch.long)
    relics = torch.zeros(B, max_relics, dtype=torch.long)
    deck = torch.zeros(B, max_deck, dtype=torch.long)
    deckw = torch.zeros(B, max_deck)
    held = torch.zeros(B, max_potions, dtype=torch.long)
    heldm = torch.zeros(B, max_potions)
    upg = torch.zeros(B, max_deck, dtype=torch.long)
    upgw = torch.zeros(B, max_deck)

    for b, (k, o) in enumerate(zip(kinds, obs_list)):
        dk[b] = DKIND_IX.get(k, 0)
        p = o.get("player", o)
        hp = float(p.get("hp", 0) or 0)
        mx = max(1.0, float(p.get("max_hp", 1) or 1))
        s = scal[b]
        s[0] = hp / mx
        s[1] = hp / 100.0
        s[2] = mx / 100.0
        s[3] = float(p.get("gold", 0) or 0) / 500.0
        s[4] = float(p.get("deck_size", 0) or 0) / 40.0
        s[5] = float(p.get("deck_upgrades", 0) or 0) / 20.0
        s[6] = float(o.get("act", p.get("act", 0)) or 0) / 3.0
        s[7] = float(o.get("act_floor", p.get("act_floor", 0)) or 0) / 17.0
        rl = p.get("relics") or []
        pt = p.get("potions") or []
        s[8] = len(rl) / 15.0
        s[9] = len(pt) / 5.0
        s[10] = float(o.get("total_floor", 0) or 0) / 51.0
        s[11] = float(len(o.get("points") or ())) / 60.0
        # Which fight is this decision about? Only potion_gate currently carries room_type;
        # every other kind leaves these three at zero, which is correct -- the feature is
        # meaningful exactly where the observation supplies it.
        ri = ROOM_IX.get(o.get("room_type") or "")
        if ri is not None:
            s[12 + ri] = 1.0

        for i, r in enumerate(rl[:max_relics]):
            relics[b, i] = max(0, int(r.get("relic_idx", 0) or 0)) if isinstance(r, dict) else 0
        bag = p.get("deck_bag") or {}
        for i, (cid, cnt) in enumerate(list(bag.items())[:max_deck]):
            deck[b, i] = max(0, int(cid))
            deckw[b, i] = float(cnt)
        for i, pot in enumerate(pt[:max_potions]):
            held[b, i] = (max(0, int(pot.get("potion_idx", 0) or 0))
                          if isinstance(pot, dict) else max(0, int(pot or 0)))
            heldm[b, i] = 1.0
        # deck_cards is the exact deck, card by card, WITH upgrade level -- the field that was
        # sent from the start and never read.
        dc = p.get("deck_cards") or []
        ui = 0
        for cobj in dc:
            if ui >= max_deck:
                break
            if int(cobj.get("up", 0) or 0) > 0:
                from_vocab = cobj.get("card_idx")
                if from_vocab is None:
                    continue
                upg[b, ui] = max(0, int(from_vocab))
                upgw[b, ui] = 1.0
                ui += 1

    if _AUDIT:
        _AUDIT["n"] += B
        _AUDIT["held"] += int((heldm.sum(1) > 0).sum())
        _AUDIT["upg"] += int((upgw.sum(1) > 0).sum())
        _AUDIT["room"] += int((scal[:, 12:15].sum(1) > 0).sum())
    return (scal.to(device), dk.to(device), relics.to(device),
            deck.to(device), deckw.to(device),
            held.to(device), heldm.to(device), upg.to(device), upgw.to(device))


def migrate_state_scalars(sd, d_model=192, old_scalars=12):
    """Brings state_mlp.0.weight forward across BOTH state-layout changes.

    Two different kinds of change, and they need different handling:

    1. The scalar block sits FIRST, so widening it (12 -> 15, for the room one-hot) shifts
       every relic/card/kind column after it. Those must be moved, not overlap-copied --
       load_compat's generic "copy the overlapping corner" would map old relic columns onto
       new scalar columns and scramble the trunk.
    2. The pooled held-potion and upgraded-card blocks are appended LAST, so they are a pure
       right-pad.

    Both leave the new columns at ZERO, so a migrated checkpoint scores identically until it
    learns to use the new inputs. Idempotent: a checkpoint already at the current width is
    returned untouched.
    """
    k = "state_mlp.0.weight"
    W = sd.get(k)
    if W is None:
        return sd
    want = STATE_SCALARS + 5 * d_model
    have = W.shape[1]
    if have == want:
        return sd
    # Step 1: scalar-block insert, if this checkpoint predates it.
    if have in (old_scalars + 3 * d_model, old_scalars + 5 * d_model):
        mid = torch.zeros(W.shape[0], have + (STATE_SCALARS - old_scalars), dtype=W.dtype)
        mid[:, :old_scalars] = W[:, :old_scalars]
        mid[:, STATE_SCALARS:] = W[:, old_scalars:]
        W = mid
        have = W.shape[1]
    # Step 2: right-pad for any appended blocks.
    if have < want:
        new = torch.zeros(W.shape[0], want, dtype=W.dtype)
        new[:, :have] = W
        W = new
    elif have > want:
        return sd            # newer than this code understands; leave it alone
    sd = dict(sd)
    sd[k] = W
    print(f"  migrated {k}: ({sd[k].shape[0]}, {have}) -> {tuple(W.shape)} "
          f"(new columns zero-initialised)")
    return sd


class RunNet(nn.Module):
    """Scores run-level actions.

    ``per_kind_heads`` gives each decision kind its own scoring and value head on top of a
    shared trunk. The trunk still learns the things all the decisions share -- what the deck
    and relics are worth, how far into the run we are -- while the head that decides campfires
    stops competing for the same output weights as the head that decides map travel.

    The motivation is measured, not aesthetic: with one shared head, campfires (~7% of run
    decisions) collapsed to a single action regardless of hp while card rewards (~20%) learned
    a real policy. The rare decision had enough gradient to pick a constant and not enough to
    learn a conditional.
    """

    def __init__(self, n_cards, n_relics, n_potions, d_model=192, per_kind_heads=False,
                 dropout: float = 0.0):
        super().__init__()
        self.per_kind_heads = per_kind_heads
        self.d = d_model
        self.card_emb = nn.Embedding(n_cards, d_model)
        self.relic_emb = nn.Embedding(n_relics, d_model)
        self.potion_emb = nn.Embedding(n_potions, d_model)
        self.token_emb = nn.Embedding(TOKENS, d_model)
        self.akind_emb = nn.Embedding(len(AKINDS), d_model)
        self.dkind_emb = nn.Embedding(len(DKINDS), d_model)

        self.d_model = d_model
        self.state_mlp = nn.Sequential(
            # +2*d_model for the pooled held-potion and upgraded-card blocks.
            nn.Linear(STATE_SCALARS + 5 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.act_mlp = nn.Sequential(
            nn.Linear(NUM + 5 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        # Bilinear-ish scoring: concatenate state and action and score, which lets the net
        # express "this card is good BECAUSE my deck already has these" rather than a fixed
        # per-card preference.
        def _score():
            return nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, 1))

        def _value():
            return nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, 1))

        self.score = _score()
        self.value = _value()
        if per_kind_heads:
            self.score_k = nn.ModuleList([_score() for _ in DKINDS])
            self.value_k = nn.ModuleList([_value() for _ in DKINDS])

    def clone_shared_into_heads(self):
        """Seeds every per-kind head from the shared head.

        Lets a checkpoint trained with one head resume with per-kind heads without discarding
        what it learned: each head starts as a copy of the old behaviour and diverges from
        there, rather than starting random.
        """
        if not self.per_kind_heads:
            return
        for h in self.score_k:
            h.load_state_dict(self.score.state_dict())
        for h in self.value_k:
            h.load_state_dict(self.value.state_dict())

    def _ix(self, t, emb):
        return t.clamp(0, emb.num_embeddings - 1)

    def state_vec(self, scal, dk, relics, deck, deckw, held=None, heldm=None,
                  upg=None, upgw=None):
        r = self.relic_emb(self._ix(relics, self.relic_emb))
        rmask = (relics > 0).unsqueeze(-1).float()
        r = (r * rmask).sum(1) / rmask.sum(1).clamp(min=1)

        c = self.card_emb(self._ix(deck, self.card_emb))
        w = deckw.unsqueeze(-1)
        c = (c * w).sum(1) / w.sum(1).clamp(min=1)

        d = self.dkind_emb(self._ix(dk, self.dkind_emb))

        # Held potions and upgraded cards, pooled. Appended LAST so an older checkpoint
        # migrates by zero-padding on the right and behaves identically until it trains.
        def _pool(emb, idx, w):
            if idx is None:
                return torch.zeros(scal.shape[0], self.d_model, device=scal.device,
                                   dtype=scal.dtype)
            v = emb(self._ix(idx, emb)) * w.unsqueeze(-1)
            return v.sum(1) / w.sum(1).clamp(min=1.0).unsqueeze(-1)

        pv = _pool(self.potion_emb, held, heldm)
        uv = _pool(self.card_emb, upg, upgw)
        return self.state_mlp(torch.cat([scal, r, c, d, pv, uv], dim=-1))

    def forward(self, state, actions):
        # state is (scal, dk, relics, deck, deckw, held, heldm, upg, upgw); older callers may
        # pass the 5-tuple, so unpack positionally and let state_vec default the rest.
        state_parts = state
        scal, dk = state[0], state[1]
        akind, token, card, relic, potion, num, mask = actions

        h = self.state_vec(*state_parts)                             # (B, d)
        a = self.act_mlp(torch.cat([
            num,
            self.akind_emb(self._ix(akind, self.akind_emb)),
            self.token_emb(self._ix(token, self.token_emb)),
            self.card_emb(self._ix(card, self.card_emb)),
            self.relic_emb(self._ix(relic, self.relic_emb)),
            self.potion_emb(self._ix(potion, self.potion_emb)),
        ], dim=-1))                                                  # (B, A, d)

        hx = h.unsqueeze(1).expand(-1, a.shape[1], -1)
        pair = torch.cat([hx, a], dim=-1)

        if not self.per_kind_heads:
            logits = self.score(pair).squeeze(-1)
            value = self.value(h).squeeze(-1)
        else:
            # Route each row to its kind's head. Looping over the kinds PRESENT (at most 9,
            # usually 2-4) rather than over rows keeps this a handful of batched matmuls.
            logits = torch.zeros(pair.shape[0], pair.shape[1], device=pair.device, dtype=pair.dtype)
            value = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
            for k in dk.unique().tolist():
                sel = (dk == k).nonzero(as_tuple=True)[0]
                logits[sel] = self.score_k[k](pair[sel]).squeeze(-1)
                value[sel] = self.value_k[k](h[sel]).squeeze(-1)

        neg = torch.finfo(logits.dtype).min / 4
        return logits.masked_fill(~mask, neg), value
