"""Any decision in a run -> the unified model's token arrays. Numpy only, so it tests without torch.

Every decision, combat or not, becomes the same set of entity arrays:

    ctx        one context row: who, where, how healthy, which boss, which decision
    cards      hand, draw/discard/exhaust piles, the deck, and cards on offer
    ents       enemies first, then allies (Osty), each with pooled powers
    ppow       the player's powers
    orbs       Defect's orb queue, in order
    relics, pots, map points
    cands      one row per legal action, IN LEGAL ORDER, so logits index straight back

A candidate points at the tokens it acts on (card_ref, ent_ref, map_ref, pot_ref). The model adds
those tokens' embeddings to the candidate's own, so "play card 3 at enemy 1" is built from the
same card and enemy tokens the state is made of.

Numbers are symlog-scaled (sign(x)*log1p|x|), so a 512-HP boss and a 99-stack debuff stay in the
range the network trains on. Counts, HP and damage also get a ratio to the player's max HP where
that is the decision-relevant scale.

Padding is dynamic (to the batch max), with hard caps. A capped entity is truncated and counted.
Candidates are NEVER truncated: dropping one would silently misalign the chosen action's label.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

# ---- vocabularies that live in code rather than in the probe's vocab.json -------------------

VALUE_SLOTS = ("damage", "block", "repeat", "cards", "power", "energy", "hp_loss", "heal",
               "stars", "summon", "forge", "osty_damage", "gold", "max_hp")
NV = len(VALUE_SLOTS)
KW_BITS = 8                                  # CardKeyword enum: None..Eternal
CHARACTERS = ("IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT")
DKINDS = ("combat", "travel", "card_reward", "rest", "event", "shop", "potion_gate",
          "card_select", "treasure", "potion_ooc")
AKINDS = ("play", "end_turn", "select_card", "select_done", "use_potion",       # combat
          "travel", "card", "alt", "relic", "potion", "remove_card", "leave",   # run
          "allow", "deny", "drink", "hold", "rest_option", "event_option")
AKIND_IX = {k: i + 1 for i, k in enumerate(AKINDS)}                            # 0 = pad
OPT_OUT = {"end_turn", "select_done", "leave", "deny", "hold", "alt"}

# Card locations. Hand cards come FIRST in the card array, so a play's hand index IS its card
# token position; offered cards are appended and referenced explicitly.
LOC_HAND, LOC_DRAW, LOC_DISCARD, LOC_EXHAUST, LOC_DECK, LOC_OFFER, LOC_PROMPT = 1, 2, 3, 4, 5, 6, 7
N_LOC = 8

TOKENS = 4096                                # hashed strings: moves, event keys, tips, options
N_INTENT = 16
POINT_TYPES = 9

CAPS = dict(cards=160, ents=12, ppow=24, epow=10, orbs=10, relics=40, pots=10, map=80,
            cand=160, clist=8)

# Feature widths, fixed so the model can size its projections.
F_CTX = 24
F_CARD = 1 + KW_BITS + NV + 10
F_ENT = 10 + N_INTENT
F_ORB = 3
F_RELIC = 3 + NV
F_POT = 3 + NV
F_MAP = 7
F_CAND = NV + KW_BITS + 12 + (POINT_TYPES + 1) + NV


def tok(s) -> int:
    """Stable hash of a string into the shared token table (crc32, as runmodel._tok)."""
    if not s:
        return 0
    return zlib.crc32(str(s).encode("utf-8")) % (TOKENS - 1) + 1


def symlog(x) -> float:
    x = float(x or 0)
    return math.copysign(math.log1p(abs(x)), x)


def _vals(d) -> list:
    d = d or {}
    return [symlog(d.get(k, 0)) for k in VALUE_SLOTS]


def _kw(m) -> list:
    m = int(m or 0)
    return [float((m >> b) & 1) for b in range(KW_BITS)]


def _i(x, default=0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


class Vocab:
    """Sizes and name->index maps from the probe's vocab.json (the M1 dump)."""

    def __init__(self, vocab_json: dict):
        self.sizes = dict(vocab_json["sizes"])
        rt = vocab_json.get("enums", {}).get("room_type", [])
        self.room_ix = {name: i for i, name in enumerate(rt)}


# ---- per-entity row builders --------------------------------------------------------------

def _card_row(c: dict, loc: int, n: int = 1):
    ids = [_i(c.get("card_idx")), _i(c.get("ench_idx")), _i(c.get("affl_idx")),
           _i(c.get("type"), -1) + 1, _i(c.get("rarity"), -1) + 1,
           _i(c.get("target_type"), -1) + 1, loc]
    up = c.get("upgrade", c.get("up", 0))
    # ENERGY cost. Unknown for pile groups and prompts (not sent) and for shop cards, whose
    # "cost" is a gold price; the caller passes energy_cost=None there. Unknown is its own flag
    # so it cannot read as "unplayable" (-1).
    raw = c.get("energy_cost", c.get("cost"))
    known = raw is not None
    cost = _i(raw, -1) if known else 0
    f = ([float(up or 0)] + _kw(c.get("kw")) + _vals(c.get("vals")) +
         [max(-1, min(cost, 5)) / 3.0, float(known and cost < 0), float(known),
          float(bool(c.get("cost_x"))),
          _i(c.get("star_cost"), -1) / 3.0, float(bool(c.get("playable", True))),
          symlog(c.get("ench_amount")), symlog(c.get("affl_amount")),
          float(bool(c.get("ench_disabled"))), symlog(n)])
    return ids, f


def _ent_row(e: dict, side: int, player_hp: float):
    intents = e.get("intents") or []
    counts = [0.0] * N_INTENT
    dmg = reps = status = 0
    for it in intents:
        t = _i(it.get("type_idx"), -1)
        if 0 <= t < N_INTENT:
            counts[t] += 1
        dmg += _i(it.get("damage"))
        reps += _i(it.get("repeats"))
        status += _i(it.get("count"))
    hp, mx = _i(e.get("hp")), max(1, _i(e.get("max_hp"), 1))
    ids = [_i(e.get("monster_idx")),
           tok(f"{e.get('monster', '')}:{e.get('move', '')}") if e.get("move") else 0, side]
    f = ([symlog(hp), symlog(mx), hp / mx, symlog(e.get("block")),
          float(bool(e.get("alive", True))), float(bool(e.get("hittable", True))),
          symlog(dmg), dmg / max(1.0, player_hp), symlog(reps), symlog(status)] + counts)
    return ids, f


def _relic_row(r: dict):
    c = _i(r.get("counter"), -1)
    return _i(r.get("relic_idx")), [symlog(max(c, 0)), float(c >= 0),
                                    float(bool(r.get("melted")))] + _vals(r.get("vals"))


def _pot_row(p: dict, slot: int):
    return _i(p.get("potion_idx")), [float(bool(p.get("usable", True))),
                                     float(bool(p.get("gated"))), slot / 5.0] + _vals(p.get("vals"))


def _lookahead(obs, point_index, horizon=6, decay=0.75):
    """Discounted counts of point types reachable from a map point, + rows to the boss."""
    pts = obs.get("points") or []
    out = [0.0] * (POINT_TYPES + 1)
    if not (0 <= point_index < len(pts)):
        return out
    frontier, seen, w = [point_index], {point_index}, 1.0
    for _ in range(horizon):
        nxt = []
        for i in frontier:
            for j in pts[i].get("children", ()):
                if j in seen or not (0 <= j < len(pts)):
                    continue
                seen.add(j)
                t = _i(pts[j].get("type_idx"))
                if 0 <= t < POINT_TYPES:
                    out[t] += w
                nxt.append(j)
        if not nxt:
            break
        frontier, w = nxt, w * decay
    boss = _i(obs.get("boss_point"), -1)
    if 0 <= boss < len(pts):
        out[POINT_TYPES] = (_i(pts[boss].get("row")) - _i(pts[point_index].get("row"))) / 20.0
    return out


# ---- one decision --------------------------------------------------------------------------

class _Dec:
    """Row lists for one decision, before padding."""

    def __init__(self):
        self.ctx_ids = None
        self.ctx_f = None
        self.card_ids, self.card_f = [], []
        self.ent_ids, self.ent_f, self.epow = [], [], []
        self.ppow = []
        self.orb_ids, self.orb_f = [], []
        self.relic_ids, self.relic_f = [], []
        self.pot_ids, self.pot_f = [], []
        self.map_ids, self.map_f = [], []
        self.cand_ids, self.cand_f = [], []
        self.cand_ref = []            # (card, ent, map, pot) token positions, -1 = none
        self.cand_list_cards, self.cand_list_toks = [], []


def encode_one(kind: str, obs: dict, legal: list, vocab: Vocab, trunc: dict) -> _Dec:
    d = _Dec()
    combat = kind == "combat"
    p = obs.get("player", obs) if not combat else obs.get("player", {})
    # Run-level scalars sit at the top of combat obs and inside the player dict elsewhere
    # (travel nests it); read whichever holds them.
    src = obs if combat else p
    hp = float(p.get("hp", 0) or 0)
    mx = float(max(1, p.get("max_hp", 1) or 1))

    # ---- context ----
    room = obs.get("room_type") or ""
    room_ix = (_i(obs.get("room_idx"), -1) if combat else vocab.room_ix.get(room, -1))
    enemies = (obs.get("enemies") or []) if combat else []
    incoming = sum(_i(it.get("damage")) for e in enemies if e.get("alive", True)
                   for it in e.get("intents") or ())
    block = float(p.get("block", 0) or 0)
    char = src.get("character", p.get("character", ""))
    d.ctx_ids = [DKINDS.index(kind) + 1 if kind in DKINDS else 0,
                 (CHARACTERS.index(char) + 1) if char in CHARACTERS else 0,
                 room_ix + 1,
                 _i(src.get("boss_idx", p.get("boss_idx"))),
                 tok(obs.get("encounter")),
                 _i(src.get("act", p.get("act")), -1) + 1,
                 _i(obs.get("phase_idx")) if combat else 0]
    potions = (p.get("potion_slots") if (combat or p.get("potion_slots") is not None)
               else p.get("potions")) or []
    d.ctx_f = [symlog(hp), symlog(mx), hp / mx, symlog(block),
               _i(p.get("energy")) / 5.0, _i(p.get("max_energy")) / 5.0, symlog(p.get("stars")),
               symlog(p.get("gold")),
               _i(src.get("act_floor", p.get("act_floor"))) / 17.0,
               _i(src.get("total_floor", p.get("total_floor"))) / 51.0,
               _i(obs.get("turn")) / 10.0, _i(obs.get("round")) / 10.0,
               symlog(obs.get("draw_count")), symlog(obs.get("discard_count")),
               symlog(obs.get("exhaust_count")), symlog(p.get("deck_size")),
               _i(obs.get("orb_slots")) / 5.0, len(potions) / 5.0, float(combat),
               symlog(incoming), incoming / mx, max(0.0, incoming - block) / max(1.0, hp),
               float(sum(1 for e in enemies if e.get("alive", True))) / 5.0,
               float(bool(obs.get("room_type") or combat))]
    assert len(d.ctx_f) == F_CTX

    def add_card(c, loc, n=1):
        if len(d.card_ids) >= CAPS["cards"]:
            trunc["cards"] += 1
            return -1
        ids, f = _card_row(c, loc, n)
        d.card_ids.append(ids)
        d.card_f.append(f)
        return len(d.card_ids) - 1

    # ---- cards ----
    if combat:
        for c in obs.get("hand") or []:
            add_card(c, LOC_HAND)
        for pile, loc in (("draw_pile", LOC_DRAW), ("discard_pile", LOC_DISCARD),
                          ("exhaust_pile", LOC_EXHAUST)):
            for g in obs.get(pile) or []:
                add_card(g, loc, _i(g.get("n"), 1))
    else:
        # The deck, grouped: identical (card, upgrade, keywords, enchant, affliction) share a
        # token with a count, so a 40-card deck of five Strikes is not five identical tokens.
        groups = {}
        for c in p.get("deck_cards") or []:
            key = (c.get("card_idx"), c.get("up", c.get("upgrade")), c.get("kw"),
                   c.get("ench_idx"), c.get("affl_idx"))
            if key in groups:
                groups[key][1] += 1
            else:
                groups[key] = [c, 1]
        for c, n in groups.values():
            add_card(c, LOC_DECK, n)

    # ---- entities: enemies first (a play's target index IS its position), then allies ----
    for side, lst in ((1, obs.get("enemies") or []), (2, obs.get("allies") or [])):
        for e in lst:
            if len(d.ent_ids) >= CAPS["ents"]:
                trunc["ents"] += 1
                break
            ids, f = _ent_row(e, side, hp)
            d.ent_ids.append(ids)
            d.ent_f.append(f)
            pw = e.get("powers") or []
            if len(pw) > CAPS["epow"]:
                trunc["epow"] += 1
            d.epow.append([(_i(q.get("power_idx")), symlog(q.get("amount"))) for q in pw[:CAPS["epow"]]])

    for q in (p.get("powers") or [])[:CAPS["ppow"]]:
        d.ppow.append((_i(q.get("power_idx")), symlog(q.get("amount"))))
    for k, o in enumerate((obs.get("orbs") or [])[:CAPS["orbs"]]):
        d.orb_ids.append(_i(o.get("orb_idx")))
        d.orb_f.append([symlog(o.get("passive")), symlog(o.get("evoke")), k / 10.0])
    for r in (p.get("relics") or [])[:CAPS["relics"]]:
        if isinstance(r, dict):
            i, f = _relic_row(r)
            d.relic_ids.append(i)
            d.relic_f.append(f)
    pot_pos = {}
    for k, q in enumerate(potions[:CAPS["pots"]]):
        if isinstance(q, dict):
            i, f = _pot_row(q, _i(q.get("slot"), k))
            pot_pos[_i(q.get("slot"), k)] = len(d.pot_ids)
            d.pot_ids.append(i)
            d.pot_f.append(f)

    # ---- map (travel only) ----
    pts = (obs.get("points") or []) if kind == "travel" else []
    if len(pts) > CAPS["map"]:
        trunc["map"] += 1
    cur, boss_pt = _i(obs.get("cur_point"), -1), _i(obs.get("boss_point"), -1)
    for q in pts[:CAPS["map"]]:
        d.map_ids.append(_i(q.get("type_idx")) + 1)
        d.map_f.append([_i(q.get("row")) / 16.0, _i(q.get("col")) / 6.0,
                        float(bool(q.get("visited"))), float(bool(q.get("travelable"))),
                        float(_i(q.get("i"), -2) == cur), float(_i(q.get("i"), -2) == boss_pt),
                        len(q.get("children") or ()) / 4.0])

    # ---- candidates, in legal order ----
    if len(legal) > CAPS["cand"]:
        raise ValueError(f"{kind}: {len(legal)} legal actions exceeds cap {CAPS['cand']}; "
                         f"raise CAPS['cand'] rather than truncating (labels would misalign)")
    for a in legal:
        ak = a.get("kind")
        if kind == "rest":
            ak = "leave" if a.get("option") == "LEAVE" else "rest_option"
        elif kind == "event":
            ak = "event_option"
        card_ref = ent_ref = map_ref = pot_ref = -1
        tk = card_i = pot_i = relic_i = pt = 0
        lcards, ltoks = [], []
        f_extra = [0.0] * 12
        look = [0.0] * (POINT_TYPES + 1)
        ev_vals = [0.0] * NV
        vals, kw = a.get("vals"), a.get("kw")

        if ak == "play":
            card_ref = _i(a.get("hand"), -1)
            ent_ref = _i(a.get("target"), -1)
            card_i = _i(a.get("card_idx"))
        elif ak == "select_card":
            # A prompt's options are not hand positions; the card becomes its own token.
            card_ref = add_card(a, LOC_PROMPT if combat else LOC_OFFER)
            card_i = _i(a.get("card_idx"))
        elif ak == "use_potion":
            pot_ref = pot_pos.get(_i(a.get("hand"), -1), -1)
            ent_ref = _i(a.get("target"), -1)
            pot_i = _i(a.get("potion_idx"))
        elif ak == "drink":
            pot_ref = pot_pos.get(_i(a.get("index"), -1), -1)
            pot_i = _i(a.get("potion_idx"))
        elif ak == "travel":
            map_ref = _i(a.get("point"), -1)
            if map_ref >= len(d.map_ids):
                map_ref = -1
            pt = _i(a.get("point_type_idx")) + 1
            tk = tok(a.get("point_type"))
            look = _lookahead(obs, _i(a.get("point"), -1))
        elif ak == "card":
            # Shop entries carry model_idx and a GOLD cost; card rewards carry card_idx and an
            # energy cost.
            offered = dict(a, card_idx=a.get("card_idx", a.get("model_idx")))
            if kind == "shop":
                offered["energy_cost"] = None
            card_ref = add_card(offered, LOC_OFFER)
            card_i = _i(a.get("card_idx", a.get("model_idx")))
        elif ak == "relic":
            relic_i = _i(a.get("relic_idx", a.get("model_idx")))
        elif ak == "potion":
            pot_i = _i(a.get("potion_idx", a.get("model_idx")))
        elif ak in ("rest_option", "leave") and kind == "rest":
            tk = tok(a.get("option"))
        elif ak == "event_option":
            tk = tok(a.get("key"))
            relic_i = _i(a.get("relic_idx"))
            ev_vals = _vals(a.get("event_vals"))
            lcards = [_i(x) for x in (a.get("card_idxs") or [])][:CAPS["clist"]]
            ltoks = [tok(x) for x in (a.get("tips") or [])][:CAPS["clist"]]
        elif ak == "alt":
            tk = tok(a.get("id"))

        if kind == "shop":
            f_extra[0] = symlog(a.get("cost"))
            f_extra[1] = float(bool(a.get("affordable", True)))
            f_extra[2] = float(bool(a.get("on_sale")))
        f_extra[3] = (_i(a.get("rarity"), -1) + 1) / 10.0
        f_extra[4] = float(a.get("upgrade", 0) or 0)
        f_extra[5] = float(bool(a.get("locked")))
        f_extra[6] = float(bool(a.get("proceed")))
        f_extra[7] = float(bool(a.get("chosen")))
        f_extra[8] = float(bool(a.get("will_kill")))
        f_extra[9] = float(bool(a.get("enabled", True)))
        f_extra[10] = float(ak in OPT_OUT)
        f_extra[11] = float(ak == "remove_card")

        d.cand_ids.append([AKIND_IX.get(ak, 0), tk, card_i, pot_i, relic_i, pt])
        d.cand_f.append(_vals(vals) + _kw(kw) + f_extra + look + ev_vals)
        d.cand_ref.append([card_ref, ent_ref, map_ref, pot_ref])
        d.cand_list_cards.append(lcards)
        d.cand_list_toks.append(ltoks)
    return d


# ---- batching --------------------------------------------------------------------------------

def _pad(rows, width, dtype, fill=0):
    n = max((len(r) for r in rows), default=0)
    n = max(n, 1)
    out = np.full((len(rows), n, width) if width else (len(rows), n), fill, dtype=dtype)
    mask = np.zeros((len(rows), n), dtype=bool)
    for b, r in enumerate(rows):
        if r:
            out[b, :len(r)] = r
            mask[b, :len(r)] = True
    return out, mask


def encode_batch(decisions, vocab: Vocab, trunc: dict | None = None) -> dict:
    """decisions: iterable of (kind, obs, legal). Returns padded numpy arrays."""
    trunc = trunc if trunc is not None else {k: 0 for k in CAPS}
    for k in CAPS:
        trunc.setdefault(k, 0)
    ds = [encode_one(k, o, l, vocab, trunc) for k, o, l in decisions]
    B = len(ds)
    out = {
        "ctx_ids": np.array([d.ctx_ids for d in ds], dtype=np.int64),
        "ctx_f": np.array([d.ctx_f for d in ds], dtype=np.float32),
    }
    out["card_ids"], out["card_mask"] = _pad([d.card_ids for d in ds], 7, np.int64)
    out["card_f"], _ = _pad([d.card_f for d in ds], F_CARD, np.float32)
    out["ent_ids"], out["ent_mask"] = _pad([d.ent_ids for d in ds], 3, np.int64)
    out["ent_f"], _ = _pad([d.ent_f for d in ds], F_ENT, np.float32)
    ne = out["ent_ids"].shape[1]
    ep = max(1, max((len(x) for d in ds for x in d.epow), default=0))
    out["epow_ids"] = np.zeros((B, ne, ep), np.int64)
    out["epow_amt"] = np.zeros((B, ne, ep), np.float32)
    out["epow_mask"] = np.zeros((B, ne, ep), bool)
    for b, d in enumerate(ds):
        for j, pw in enumerate(d.epow):
            for k, (pi, amt) in enumerate(pw):
                out["epow_ids"][b, j, k] = pi
                out["epow_amt"][b, j, k] = amt
                out["epow_mask"][b, j, k] = True
    pp_ids, out["ppow_mask"] = _pad([[q[0] for q in d.ppow] for d in ds], 0, np.int64)
    pp_amt, _ = _pad([[q[1] for q in d.ppow] for d in ds], 0, np.float32)
    out["ppow_ids"], out["ppow_amt"] = pp_ids, pp_amt
    out["orb_ids"], out["orb_mask"] = _pad([d.orb_ids for d in ds], 0, np.int64)
    out["orb_f"], _ = _pad([d.orb_f for d in ds], F_ORB, np.float32)
    out["relic_ids"], out["relic_mask"] = _pad([d.relic_ids for d in ds], 0, np.int64)
    out["relic_f"], _ = _pad([d.relic_f for d in ds], F_RELIC, np.float32)
    out["pot_ids"], out["pot_mask"] = _pad([d.pot_ids for d in ds], 0, np.int64)
    out["pot_f"], _ = _pad([d.pot_f for d in ds], F_POT, np.float32)
    out["map_ids"], out["map_mask"] = _pad([d.map_ids for d in ds], 0, np.int64)
    out["map_f"], _ = _pad([d.map_f for d in ds], F_MAP, np.float32)
    out["cand_ids"], out["cand_mask"] = _pad([d.cand_ids for d in ds], 6, np.int64)
    out["cand_f"], _ = _pad([d.cand_f for d in ds], F_CAND, np.float32)
    out["cand_ref"], _ = _pad([d.cand_ref for d in ds], 4, np.int64, fill=-1)
    L = max(1, max((len(x) for d in ds for x in d.cand_list_cards + d.cand_list_toks), default=0))
    A = out["cand_ids"].shape[1]
    for key, attr in (("clist_cards", "cand_list_cards"), ("clist_toks", "cand_list_toks")):
        arr = np.zeros((B, A, L), np.int64)
        for b, d in enumerate(ds):
            for j, lst in enumerate(getattr(d, attr)):
                arr[b, j, :len(lst)] = lst
        out[key] = arr
    return out
