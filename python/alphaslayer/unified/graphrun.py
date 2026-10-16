"""Inference forward passes replayed from CUDA/HIP graphs (fight search, rollouts, collection).

A server step's network pass costs ~4 ms whatever its batch: the forward is ~420 small kernels,
each launched from Python (~15 us of CPU apiece), plus one GPU->CPU sync in UnifiedNet._pack
(the packed length is data-dependent). Search rollouts wait on these passes one step at a
time, so the launch overhead, not the arithmetic, is what they pay for.

GraphRunner records the forward once per shape bucket and replays it with one launch:

- every token group is padded to a fixed cap (CAPS) and the batch to a power of two, so the
  inputs have one shape per bucket;
- the packed length (real tokens of the longest row) is computed on the CPU from the masks
  and rounded up to a bucket (N_BUCKETS), so _pack needs no sync;
- a batch that exceeds a cap (a map decision, a huge deck) falls back to the eager forward.

Padded tokens are masked exactly as UnifiedNet masks its own padding, so the outputs equal the
eager forward's up to floating-point summation order (a different sequence length can pick
different kernels).
"""
from __future__ import annotations

import numpy as np
import torch

from .net import to_torch

# per-row caps on each token group (second dim of each padded input), and the trailing dims
CAPS = {"card": 48, "ent": 8, "epow": 12, "ppow": 24, "orb": 12, "relic": 40, "pot": 6,
        "map": 1, "cand": 24, "clist": 8}
B_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
N_BUCKETS = (32, 48, 64, 80, 96, 128, 160)

# input key -> (group cap name, padded dims): which dims of each array are padded to a cap
_KEYS = {
    "card_ids": ("card",), "card_f": ("card",), "card_mask": ("card",),
    "ent_ids": ("ent",), "ent_f": ("ent",), "ent_mask": ("ent",),
    "epow_ids": ("ent", "epow"), "epow_amt": ("ent", "epow"), "epow_mask": ("ent", "epow"),
    "ppow_ids": ("ppow",), "ppow_amt": ("ppow",), "ppow_mask": ("ppow",),
    "orb_ids": ("orb",), "orb_f": ("orb",), "orb_mask": ("orb",),
    "relic_ids": ("relic",), "relic_f": ("relic",), "relic_mask": ("relic",),
    "pot_ids": ("pot",), "pot_f": ("pot",), "pot_mask": ("pot",),
    "map_ids": ("map",), "map_f": ("map",), "map_mask": ("map",),
    "cand_ids": ("cand",), "cand_f": ("cand",), "cand_mask": ("cand",), "cand_ref": ("cand",),
    "clist_cards": ("cand", "clist"), "clist_toks": ("cand", "clist"),
    "ctx_ids": (), "ctx_f": (),
}
_GROUP_MASKS = ("card_mask", "ent_mask", "ppow_mask", "orb_mask", "relic_mask", "pot_mask",
                "map_mask", "cand_mask")


def _bucket(x, buckets):
    for b in buckets:
        if x <= b:
            return b
    return None


class GraphRunner:
    """``run(batch, loop, bf16)``: the forward's outputs ({"logits", "value", "aux": {...}}) for
    a numpy batch from features.encode_batch, from a recorded graph when the batch fits a
    bucket (else the eager forward)."""

    def __init__(self, net, dev, max_graphs: int = 48):
        self.net, self.dev = net, dev
        self.graphs: dict = {}
        self.max_graphs = max_graphs
        self.stats = {"graph": 0, "eager": 0, "captures": 0}
        self._pool = None

    # ---- shapes ---------------------------------------------------------------------------

    def _fits(self, batch: dict):
        B = batch["ctx_ids"].shape[0]
        Bb = _bucket(B, B_BUCKETS)
        if Bb is None:
            return None
        for k, caps in _KEYS.items():
            if k not in batch:
                return None
            for d, cap in enumerate(caps, start=1):
                if batch[k].shape[d] > CAPS[cap]:
                    return None
        # real tokens per row: cls + ctx + every group's real entries (UnifiedNet.embed)
        n_real = 2 + sum(batch[m].sum(1) for m in _GROUP_MASKS)
        nb = _bucket(int(n_real.max()), N_BUCKETS)
        if nb is None:
            return None
        return Bb, nb

    @staticmethod
    def _pad(batch: dict, Bb: int) -> dict:
        out = {}
        for k, caps in _KEYS.items():
            a = batch[k]
            shape = [Bb] + [CAPS[c] for c in caps] + list(a.shape[1 + len(caps):])
            z = np.zeros(shape, dtype=a.dtype)
            z[tuple(slice(0, s) for s in a.shape)] = a
            out[k] = z
        # padded rows: no candidates, nothing but cls + ctx (always real)
        return out

    # ---- capture / replay -----------------------------------------------------------------

    def _forward(self, b, loop, n):
        net = self.net
        orig = net._pack

        def pack_fixed(toks, keep):
            order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :n]
            return (toks.gather(1, order.unsqueeze(-1).expand(-1, -1, toks.shape[-1])),
                    keep.gather(1, order))

        net._pack = pack_fixed
        try:
            return net(b, loop=loop)
        finally:
            net._pack = orig

    def _capture(self, key, b_static, loop, n, bf16):
        net = self.net
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        # autocast's weight-cast cache must be off under capture (the casts are recorded)
        with torch.cuda.stream(s), torch.inference_mode(), \
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            for _ in range(2):                    # warm up (allocator, kernel selection)
                self._forward(b_static, loop, n)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        # each graph keeps its own memory pool: a shared one would let one graph's replay
        # write over another's outputs
        with torch.cuda.graph(g), torch.inference_mode(), \
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            out = self._forward(b_static, loop, n)
        self.stats["captures"] += 1
        return g, out

    @torch.inference_mode()
    def run(self, batch: dict, loop: int = 1, bf16: bool = True) -> dict:
        fit = self._fits(batch)
        if fit is None or (len(self.graphs) >= self.max_graphs and (fit, loop, bf16) not in self.graphs):
            self.stats["eager"] += 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                return self.net(to_torch(batch, self.dev), loop=loop)
        Bb, n = fit
        key = (fit, loop, bf16)
        padded = self._pad(batch, Bb)
        # one host->device copy per dtype into the graph's static input buffers (as to_torch)
        groups: dict = {}
        for k in _KEYS:
            groups.setdefault(padded[k].dtype, []).append(k)
        flats = {dt: np.concatenate([padded[k].reshape(-1) for k in ks]) for dt, ks in groups.items()}
        if key not in self.graphs:
            bufs, static = {}, {}
            for dt, ks in groups.items():
                bufs[dt] = torch.from_numpy(flats[dt]).to(self.dev)
                o = 0
                for k in ks:
                    static[k] = bufs[dt][o:o + padded[k].size].view(padded[k].shape)
                    o += padded[k].size
            g, out = self._capture(key, static, loop, n, bf16)
            self.graphs[key] = (g, bufs, out)
        g, bufs, out = self.graphs[key]
        for dt, flat in flats.items():
            bufs[dt].copy_(torch.from_numpy(flat))
        g.replay()
        self.stats["graph"] += 1
        B = batch["ctx_ids"].shape[0]
        A = batch["cand_mask"].shape[1]
        return {"logits": out["logits"][:B, :A], "value": out["value"][:B],
                "value_logit": out["value_logit"][:B],
                "aux": {k: v[:B] for k, v in out["aux"].items()}}
