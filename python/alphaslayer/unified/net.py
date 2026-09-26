"""The unified network: one transformer for every decision in a run.

Tokens: [CLS] [CTX] cards ents player-powers orbs relics potions map-points candidates.
No positional encoding: every group is a set (the draw pile is a multiset), and order is
carried only where it matters as a feature (orb position).

Each candidate's embedding is built from its own features PLUS the embeddings of the tokens it
acts on (its card, its target, its map point, its potion). So "play Strike at the Vulnerable
enemy" starts from the same Strike and enemy tokens the state holds, and then attends to
everything else: a card reward option can look at the deck it would join.

Heads, all from the final layer:
  policy      one logit per candidate token (masked at padding)
  value       logit of P(win | state). The objective is win-only (docs/CAPABILITIES.md, sec. 0)
  aux         fight_win, fight_hp_loss, act_clear, reach_act3, floors_left. Predicted, never
              rewarded; dense supervision for the trunk while wins are rare

Shape: WIDE and shallow (d=512, prelude 1 / core 2 / coda 1), with depth from looping the core.
Looped models match same-depth plain stacks on multi-step reasoning (Saunshi et al., ICLR 2025),
so unique parameters go to width. But a loop costs the same compute as the depth it replaces
(measured: d512 1/2/1 looped x3 is 1.5x slower than a plain d384 x8 with equal parameters), so
the default is ONE pass, the fastest configuration measured, and loops are spent only where the
decision warrants it.

``loop`` passed to forward() overrides the configured count, per call. Under PPO the count must
be a fixed function of the state (e.g. by room type), identical at rollout and update, or the
importance ratio compares two different policies. Supervised distillation can sample it
randomly, which trains the net to work at any depth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features as FT

AUX_HEADS = ("fight_win", "fight_hp_loss", "act_clear", "reach_act3", "floors_left")
SEGMENTS = ("cls", "ctx", "card", "ent", "ppow", "orb", "relic", "pot", "map", "cand")


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class UnifiedNet(nn.Module):
    def __init__(self, sizes: dict, d: int = 512, heads: int = 8, prelude: int = 1,
                 core: int = 2, coda: int = 1, loop: int = 1, ff_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, prelude=prelude, core=core, coda=coda, loop=loop,
                        ff_mult=ff_mult, dropout=dropout)
        self.sizes = dict(sizes)
        self.d = d
        E = lambda n: nn.Embedding(max(2, int(n)), d)

        # Shared entity tables: the same card/relic/power/potion embedding everywhere.
        self.card_emb = E(sizes["cards"])
        self.ench_emb = E(sizes["enchantments"])
        self.affl_emb = E(sizes.get("afflictions", 2))
        self.ctype_emb = E(sizes["card_types"] + 1)
        self.rarity_emb = E(sizes["card_rarities"] + 1)
        self.target_emb = E(sizes["target_types"] + 1)
        self.loc_emb = E(FT.N_LOC)
        self.monster_emb = E(sizes["monsters"])
        self.side_emb = E(3)
        self.power_emb = E(sizes["powers"])
        self.orb_emb = E(sizes.get("orbs", 2))
        self.relic_emb = E(sizes["relics"])
        self.potion_emb = E(sizes["potions"])
        self.map_emb = E(FT.POINT_TYPES + 2)
        self.tok_emb = E(FT.TOKENS)
        self.dkind_emb = E(len(FT.DKINDS) + 1)
        self.char_emb = E(len(FT.CHARACTERS) + 1)
        self.room_emb = E(sizes.get("room_types", 9) + 1)
        self.boss_emb = E(sizes.get("encounters", 2))
        self.act_emb = E(6)
        self.phase_emb = E(sizes.get("turn_phases", 6) + 1)
        self.akind_emb = E(len(FT.AKINDS) + 1)

        self.ctx_proj = nn.Linear(FT.F_CTX, d)
        self.card_proj = nn.Linear(FT.F_CARD, d)
        self.ent_proj = nn.Linear(FT.F_ENT, d)
        self.pow_amt = nn.Linear(1, d)
        self.orb_proj = nn.Linear(FT.F_ORB, d)
        self.relic_proj = nn.Linear(FT.F_RELIC, d)
        self.pot_proj = nn.Linear(FT.F_POT, d)
        self.map_proj = nn.Linear(FT.F_MAP, d)
        self.cand_proj = nn.Linear(FT.F_CAND, d)
        # What a candidate borrows from the tokens it refers to passes through its own map, so
        # "this is my card" and "this card is in my hand" are not the same vector.
        self.ref_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(4)])
        self.seg = nn.Parameter(torch.randn(len(SEGMENTS), d) * 0.02)
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.in_norm = nn.LayerNorm(d)

        def layer():
            return nn.TransformerEncoderLayer(d, heads, d * ff_mult, dropout=dropout,
                                              activation="gelu", batch_first=True,
                                              norm_first=True)
        self.prelude = nn.ModuleList([layer() for _ in range(prelude)])
        self.core = nn.ModuleList([layer() for _ in range(core)])
        self.coda = nn.ModuleList([layer() for _ in range(coda)])
        self.out_norm = nn.LayerNorm(d)

        self.policy_head = _mlp(d, d, 1)
        self.value_head = _mlp(d, d, 1)
        self.aux = nn.ModuleDict({k: _mlp(d, d, 1) for k in AUX_HEADS})

    # ---------------------------------------------------------------------------------------

    @staticmethod
    def _ix(t, emb):
        """Clamps ids into the table. An out-of-range gather is an opaque GPU fault on ROCm."""
        return t.clamp(0, emb.num_embeddings - 1)

    def _e(self, emb, t):
        return emb(self._ix(t, emb))

    def _gather(self, tokens, ref):
        """tokens (B,N,d), ref (B,A) with -1 = none -> (B,A,d), zero where ref < 0."""
        n = tokens.shape[1]
        idx = ref.clamp(0, max(0, n - 1)).unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        g = tokens.gather(1, idx)
        return g * ((ref >= 0) & (ref < n)).unsqueeze(-1).to(g.dtype)

    def embed(self, b: dict):
        """Token embeddings (before the trunk) and the key-padding keep-mask."""
        B = b["ctx_ids"].shape[0]
        ci = b["ctx_ids"]
        ctx = (self._e(self.dkind_emb, ci[:, 0]) + self._e(self.char_emb, ci[:, 1])
               + self._e(self.room_emb, ci[:, 2]) + self._e(self.boss_emb, ci[:, 3])
               + self._e(self.tok_emb, ci[:, 4]) + self._e(self.act_emb, ci[:, 5])
               + self._e(self.phase_emb, ci[:, 6]) + self.ctx_proj(b["ctx_f"]))

        k = b["card_ids"]
        cards = (self._e(self.card_emb, k[..., 0]) + self._e(self.ench_emb, k[..., 1])
                 + self._e(self.affl_emb, k[..., 2]) + self._e(self.ctype_emb, k[..., 3])
                 + self._e(self.rarity_emb, k[..., 4]) + self._e(self.target_emb, k[..., 5])
                 + self._e(self.loc_emb, k[..., 6]) + self.card_proj(b["card_f"]))

        pw = self._e(self.power_emb, b["epow_ids"]) + self.pow_amt(b["epow_amt"].unsqueeze(-1))
        m = b["epow_mask"].unsqueeze(-1).to(pw.dtype)
        epool = (pw * m).sum(2) / m.sum(2).clamp(min=1.0)
        e = b["ent_ids"]
        ents = (self._e(self.monster_emb, e[..., 0]) + self._e(self.tok_emb, e[..., 1])
                + self._e(self.side_emb, e[..., 2]) + self.ent_proj(b["ent_f"]) + epool)

        ppow = self._e(self.power_emb, b["ppow_ids"]) + self.pow_amt(b["ppow_amt"].unsqueeze(-1))
        orbs = self._e(self.orb_emb, b["orb_ids"]) + self.orb_proj(b["orb_f"])
        relics = self._e(self.relic_emb, b["relic_ids"]) + self.relic_proj(b["relic_f"])
        pots = self._e(self.potion_emb, b["pot_ids"]) + self.pot_proj(b["pot_f"])
        maps = self._e(self.map_emb, b["map_ids"]) + self.map_proj(b["map_f"])

        a = b["cand_ids"]
        lc = b["clist_cards"]
        lt = b["clist_toks"]
        lcm = (lc > 0).unsqueeze(-1).to(cards.dtype)
        ltm = (lt > 0).unsqueeze(-1).to(cards.dtype)
        lpool = ((self._e(self.card_emb, lc) * lcm).sum(2) / lcm.sum(2).clamp(min=1.0)
                 + (self._e(self.tok_emb, lt) * ltm).sum(2) / ltm.sum(2).clamp(min=1.0))
        ref = b["cand_ref"]
        cand = (self._e(self.akind_emb, a[..., 0]) + self._e(self.tok_emb, a[..., 1])
                + self._e(self.card_emb, a[..., 2]) + self._e(self.potion_emb, a[..., 3])
                + self._e(self.relic_emb, a[..., 4]) + self._e(self.map_emb, a[..., 5])
                + self.cand_proj(b["cand_f"]) + lpool
                + self.ref_proj[0](self._gather(cards, ref[..., 0]))
                + self.ref_proj[1](self._gather(ents, ref[..., 1]))
                + self.ref_proj[2](self._gather(maps, ref[..., 2]))
                + self.ref_proj[3](self._gather(pots, ref[..., 3])))

        groups = [self.cls.expand(B, 1, -1), ctx.unsqueeze(1), cards, ents, ppow, orbs, relics,
                  pots, maps, cand]
        masks = [torch.ones(B, 2, dtype=torch.bool, device=ci.device), b["card_mask"],
                 b["ent_mask"], b["ppow_mask"], b["orb_mask"], b["relic_mask"], b["pot_mask"],
                 b["map_mask"], b["cand_mask"]]
        toks = torch.cat([g + self.seg[i] for i, g in enumerate(groups)], 1)
        keep = torch.cat(masks, 1)
        # Where each row's candidates start once padding is squeezed out: every real token
        # before the candidate group.
        cand_start = toks.shape[1] - cand.shape[1]
        cand_off = keep[:, :cand_start].sum(1)
        toks, keep = self._pack(toks, keep)
        return self.in_norm(toks), keep, cand_off

    @staticmethod
    def _pack(toks, keep):
        """Moves each row's real tokens to the front, in order, and trims to the longest row.

        Each group is padded to ITS batch maximum, so without this a combat row carries the
        68 map-point slots a travel row needs, every row is ~166 tokens long, and attention
        (quadratic in length) pays for all of it: 13 GB and 0.7 s per 512-row step at 7M params.
        """
        order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
        n = int(keep.sum(1).max().clamp(min=1))
        order = order[:, :n]
        toks = toks.gather(1, order.unsqueeze(-1).expand(-1, -1, toks.shape[-1]))
        keep = keep.gather(1, order)
        return toks, keep

    def _core_states(self, x, keep, max_loop):
        """Yields the hidden state after each pass of the core (before the coda), 1..max_loop."""
        pad = ~keep
        for l in self.prelude:
            x = l(x, src_key_padding_mask=pad)
        inject = x
        for step in range(max_loop):
            if step > 0:
                x = x + inject
            for l in self.core:
                x = l(x, src_key_padding_mask=pad)
            yield x

    def _readout(self, x, keep, b, cand_off):
        pad = ~keep
        for l in self.coda:
            x = l(x, src_key_padding_mask=pad)
        h = self.out_norm(x)
        A = b["cand_mask"].shape[1]
        pos = (cand_off.unsqueeze(1) + torch.arange(A, device=h.device)).clamp(max=h.shape[1] - 1)
        hc = h.gather(1, pos.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        logits = self.policy_head(hc).squeeze(-1)
        logits = logits.masked_fill(~b["cand_mask"], torch.finfo(logits.dtype).min / 4)
        cls = h[:, 0]
        out = {"logits": logits, "value_logit": self.value_head(cls).squeeze(-1)}
        out["value"] = torch.sigmoid(out["value_logit"])
        out["aux"] = {k: head(cls).squeeze(-1) for k, head in self.aux.items()}
        return out

    def forward(self, b: dict, loop: int | None = None, loop_rows=None) -> dict:
        """One readout per row.

        ``loop`` sets one loop count for the batch; ``loop_rows`` (LongTensor, B) sets it per row,
        which is how a PPO update replays the counts recorded at rollout when an adaptive rule
        chose them. Each row's state is taken after ITS pass, so a row's output is exactly what
        it would be alone at that depth.
        """
        x, keep, cand_off = self.embed(b)
        if loop_rows is None:
            k = int(loop or self.cfg["loop"])
            state = None
            for state in self._core_states(x, keep, k):
                pass
            return self._readout(state, keep, b, cand_off)
        loop_rows = loop_rows.to(x.device).clamp(min=1)
        picked = None
        for step, state in enumerate(self._core_states(x, keep, int(loop_rows.max())), start=1):
            hit = (loop_rows == step).view(-1, 1, 1)
            picked = state if picked is None else torch.where(hit, state, picked)
        return self._readout(picked, keep, b, cand_off)

    def forward_trajectory(self, b: dict, max_loop: int) -> list:
        """Readouts after every pass, 1..max_loop: what an adaptive stopping rule looks at.

        The prelude and core run once in total; only the coda and heads repeat per readout,
        which is 1 layer of the 4 at the default shape.
        """
        x, keep, cand_off = self.embed(b)
        return [self._readout(state, keep, b, cand_off)
                for state in self._core_states(x, keep, max_loop)]


def choose_loops(trajectory: list, cand_mask, cap_rows, margin: float = 2.0,
                 stable_kl: float = 0.02):
    """Adaptive depth: uncertainty x stakes. Returns the loop count per row (LongTensor, B).

    A row stops at the first pass where the decision is settled:
      * the top two candidates are ``margin`` logits apart (a clear choice), or
      * the policy stopped moving since the previous pass (KL below ``stable_kl``),
    and never later than its own ``cap_rows`` entry. The CAP is the stakes: set it from the
    model's predicted risk in this fight, or from per-encounter loss rates while the risk head
    is not yet calibrated. A row with a single legal action stops at pass 1.

    Confidence readouts on a trajectory trained with a fixed depth prior were found to beat
    learned halting gates (Adaptive Depth in Looped Transformers, arXiv 2607.20519), which is
    why the rule is a readout rather than a trained gate. The thresholds are tuned in M3 on the
    accuracy-vs-average-loops frontier.
    """
    B = cand_mask.shape[0]
    dev = cand_mask.device
    cap = cap_rows.to(dev).clamp(min=1, max=len(trajectory))
    chosen = cap.clone()
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    prev_logp = None
    n_legal = cand_mask.sum(1)
    for k, out in enumerate(trajectory, start=1):
        lg = out["logits"].float()
        logp = F.log_softmax(lg, -1)
        top2 = lg.topk(min(2, lg.shape[1]), dim=-1).values
        clear = (top2[:, 0] - top2[:, -1] >= margin) | (n_legal <= 1)
        if prev_logp is not None:
            p = logp.exp()
            kl = (p * (logp - prev_logp)).masked_fill(~cand_mask, 0.0).sum(-1)
            clear |= kl < stable_kl
        stop = ~done & (clear | (k >= cap))
        chosen = torch.where(stop, torch.full_like(chosen, k), chosen)
        done |= stop
        prev_logp = logp
    return chosen


def to_torch(batch: dict, device) -> dict:
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
