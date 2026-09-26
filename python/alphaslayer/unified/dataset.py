"""Packed distillation data: decisions encoded ONCE into flat numpy arrays, batched by slicing.

Encoding costs ~250us per decision, which would dominate every epoch over a million decisions.
So each shard is encoded once (in parallel, cached as .npz next to the shard) into "ragged"
arrays: every entity group's rows concatenated across decisions, with an offsets array saying
where each decision's rows start. A batch is then slices + padding. `make_batch` returns exactly
what `features.encode_batch` returns for the same decisions (tests/test_unified_data.py holds
it to that), plus the training targets.
"""
from __future__ import annotations

import gzip
import json
import os
import zlib

import numpy as np

from . import features as FT

# group -> (fields, row width per field or None for scalar rows). Order of fields is the order
# they are stored and re-padded in.
GROUPS = {
    "card": [("card_ids", 7, np.int64), ("card_f", FT.F_CARD, np.float32)],
    "ent": [("ent_ids", 3, np.int64), ("ent_f", FT.F_ENT, np.float32),
            ("epow_ids", FT.CAPS["epow"], np.int64), ("epow_amt", FT.CAPS["epow"], np.float32),
            ("epow_n", None, np.int64)],
    "ppow": [("ppow_ids", None, np.int64), ("ppow_amt", None, np.float32)],
    "orb": [("orb_ids", None, np.int64), ("orb_f", FT.F_ORB, np.float32)],
    "relic": [("relic_ids", None, np.int64), ("relic_f", FT.F_RELIC, np.float32)],
    "pot": [("pot_ids", None, np.int64), ("pot_f", FT.F_POT, np.float32)],
    "map": [("map_ids", None, np.int64), ("map_f", FT.F_MAP, np.float32)],
    "cand": [("cand_ids", 6, np.int64), ("cand_f", FT.F_CAND, np.float32),
             ("cand_ref", 4, np.int64), ("clist_cards", FT.CAPS["clist"], np.int64),
             ("clist_toks", FT.CAPS["clist"], np.int64), ("tp", None, np.float32)],
}
KIND_IX = {k: i for i, k in enumerate(FT.DKINDS)}


def _rows_of(d: FT._Dec, group: str, rec: dict):
    """The rows one decision contributes to a group, one tuple of field values per row."""
    if group == "card":
        return list(zip(d.card_ids, d.card_f))
    if group == "ent":
        out = []
        for ids, f, pw in zip(d.ent_ids, d.ent_f, d.epow):
            pi = [p[0] for p in pw] + [0] * (FT.CAPS["epow"] - len(pw))
            pa = [p[1] for p in pw] + [0.0] * (FT.CAPS["epow"] - len(pw))
            out.append((ids, f, pi, pa, len(pw)))
        return out
    if group == "ppow":
        return [(q[0], q[1]) for q in d.ppow]
    if group == "orb":
        return list(zip(d.orb_ids, d.orb_f))
    if group == "relic":
        return list(zip(d.relic_ids, d.relic_f))
    if group == "pot":
        return list(zip(d.pot_ids, d.pot_f))
    if group == "map":
        return list(zip(d.map_ids, d.map_f))
    if group == "cand":
        L = FT.CAPS["clist"]
        tp = rec.get("tp") or [0.0] * len(d.cand_ids)
        return [(ids, f, ref, (lc + [0] * L)[:L], (lt + [0] * L)[:L], p)
                for ids, f, ref, lc, lt, p in zip(d.cand_ids, d.cand_f, d.cand_ref,
                                                  d.cand_list_cards, d.cand_list_toks, tp)]
    raise KeyError(group)


def run_key(run_id: str) -> float:
    """A stable [0,1) number per run, for the held-out split (never Python's salted hash)."""
    return (zlib.crc32(str(run_id).encode()) % 10_000) / 10_000.0


def pack(records: list, vocab: FT.Vocab) -> dict:
    """Encodes decisions into ragged arrays + per-decision targets."""
    trunc = {k: 0 for k in FT.CAPS}
    decs = [FT.encode_one(r["kind"], r["obs"], r["legal"], vocab, trunc) for r in records]
    out = {"ctx_ids": np.array([d.ctx_ids for d in decs], np.int64).reshape(-1, 7),
           "ctx_f": np.array([d.ctx_f for d in decs], np.float32).reshape(-1, FT.F_CTX)}
    for g, fields in GROUPS.items():
        per = [_rows_of(d, g, r) for d, r in zip(decs, records)]
        out[f"off_{g}"] = np.concatenate([[0], np.cumsum([len(p) for p in per])]).astype(np.int64)
        flat = [row for p in per for row in p]
        for j, (name, width, dt) in enumerate(fields):
            if width is None:
                out[name] = np.array([row[j] for row in flat], dt)
            else:
                out[name] = np.array([row[j] for row in flat], dt).reshape(-1, width)
    nan = float("nan")
    out["kind"] = np.array([KIND_IX.get(r["kind"], -1) for r in records], np.int64)
    out["a"] = np.array([r.get("a", 0) for r in records], np.int64)
    for k in ("won", "act_clear", "reach_act3"):
        out[k] = np.array([r.get(k, -1) for r in records], np.float32)
    out["floors_left"] = np.array([r.get("floors_left", nan) for r in records], np.float32)
    out["fight_won"] = np.array([r.get("fight_won", -1) for r in records], np.float32)
    out["fight_hp"] = np.array([r.get("fight_hp", nan) for r in records], np.float32)
    out["boss_fight"] = np.array([r.get("room") == "Boss" for r in records], bool)
    out["run_key"] = np.array([run_key(r.get("run", "")) for r in records], np.float32)
    out["n_tokens"] = np.array([2 + len(d.card_ids) + len(d.ent_ids) + len(d.ppow)
                                + len(d.orb_ids) + len(d.relic_ids) + len(d.pot_ids)
                                + len(d.map_ids) + len(d.cand_ids) for d in decs], np.int64)
    out["truncated"] = np.array([sum(trunc.values())], np.int64)
    return out


def pack_shard(path: str, vocab_json: str) -> str:
    """Packs one .jsonl.gz shard into a sibling .npz (skipped if already newer)."""
    npz = path.replace(".jsonl.gz", ".npz")
    if os.path.exists(npz) and os.path.getmtime(npz) >= os.path.getmtime(path):
        return npz
    vocab = FT.Vocab(json.load(open(vocab_json)))
    with gzip.open(path, "rt") as fh:
        records = [json.loads(l) for l in fh]
    np.savez(npz + ".tmp.npz", **pack(records, vocab))
    os.replace(npz + ".tmp.npz", npz)
    return npz


class Packed:
    """A loaded pack plus the batch builder."""

    def __init__(self, arrays: dict):
        self.a = arrays
        self.n = len(arrays["kind"])

    @classmethod
    def load(cls, npz: str):
        with np.load(npz) as z:
            return cls({k: z[k] for k in z.files})


def make_batch(items: list) -> dict:
    """items: list of (Packed, index). Returns encode_batch-identical inputs plus targets."""
    B = len(items)
    out = {"ctx_ids": np.stack([p.a["ctx_ids"][i] for p, i in items]),
           "ctx_f": np.stack([p.a["ctx_f"][i] for p, i in items])}

    def slices(g):
        return [(p, p.a[f"off_{g}"][i], p.a[f"off_{g}"][i + 1]) for p, i in items]

    def padded(g, name, width, dt, fill=0):
        sl = slices(g)
        n = max(1, max(e - s for _, s, e in sl))
        shape = (B, n, width) if width else (B, n)
        arr = np.full(shape, fill, dt)
        mask = np.zeros((B, n), bool)
        for b, (p, s, e) in enumerate(sl):
            if e > s:
                arr[b, :e - s] = p.a[name][s:e]
                mask[b, :e - s] = True
        return arr, mask

    out["card_ids"], out["card_mask"] = padded("card", "card_ids", 7, np.int64)
    out["card_f"], _ = padded("card", "card_f", FT.F_CARD, np.float32)
    out["ent_ids"], out["ent_mask"] = padded("ent", "ent_ids", 3, np.int64)
    out["ent_f"], _ = padded("ent", "ent_f", FT.F_ENT, np.float32)
    # Enemy powers: pad the power axis to the batch max, as encode_batch does.
    sl = slices("ent")
    ne = out["ent_ids"].shape[1]
    ep = max(1, max((int(p.a["epow_n"][s:e].max()) if e > s else 0) for p, s, e in sl))
    out["epow_ids"] = np.zeros((B, ne, ep), np.int64)
    out["epow_amt"] = np.zeros((B, ne, ep), np.float32)
    out["epow_mask"] = np.zeros((B, ne, ep), bool)
    for b, (p, s, e) in enumerate(sl):
        if e > s:
            out["epow_ids"][b, :e - s] = p.a["epow_ids"][s:e, :ep]
            out["epow_amt"][b, :e - s] = p.a["epow_amt"][s:e, :ep]
            out["epow_mask"][b, :e - s] = np.arange(ep)[None, :] < p.a["epow_n"][s:e, None]
    out["ppow_ids"], out["ppow_mask"] = padded("ppow", "ppow_ids", 0, np.int64)
    out["ppow_amt"], _ = padded("ppow", "ppow_amt", 0, np.float32)
    out["orb_ids"], out["orb_mask"] = padded("orb", "orb_ids", 0, np.int64)
    out["orb_f"], _ = padded("orb", "orb_f", FT.F_ORB, np.float32)
    out["relic_ids"], out["relic_mask"] = padded("relic", "relic_ids", 0, np.int64)
    out["relic_f"], _ = padded("relic", "relic_f", FT.F_RELIC, np.float32)
    out["pot_ids"], out["pot_mask"] = padded("pot", "pot_ids", 0, np.int64)
    out["pot_f"], _ = padded("pot", "pot_f", FT.F_POT, np.float32)
    out["map_ids"], out["map_mask"] = padded("map", "map_ids", 0, np.int64)
    out["map_f"], _ = padded("map", "map_f", FT.F_MAP, np.float32)
    out["cand_ids"], out["cand_mask"] = padded("cand", "cand_ids", 6, np.int64)
    out["cand_f"], _ = padded("cand", "cand_f", FT.F_CAND, np.float32)
    out["cand_ref"], _ = padded("cand", "cand_ref", 4, np.int64, fill=-1)
    # The candidate lists pad to the batch's longest list, as encode_batch does.
    cc, _ = padded("cand", "clist_cards", FT.CAPS["clist"], np.int64)
    ct, _ = padded("cand", "clist_toks", FT.CAPS["clist"], np.int64)
    L = max(1, int(max(((cc > 0).sum(-1).max()), ((ct > 0).sum(-1).max()))))
    out["clist_cards"], out["clist_toks"] = cc[..., :L], ct[..., :L]

    tgt = {}
    tgt["tp"], _ = padded("cand", "tp", 0, np.float32)
    for k in ("kind", "a", "won", "act_clear", "reach_act3", "floors_left", "fight_won",
              "fight_hp", "boss_fight", "n_tokens"):
        tgt[k] = np.array([p.a[k][i] for p, i in items])
    return out, tgt
