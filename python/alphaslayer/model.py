"""Combat network.

Shape of the design, and why:

- **Set encoder, no positional encoding.** Hand, enemies, relics and powers are sets; the
  draw pile is a multiset the player cannot order, so it enters as a bag-of-counts token
  rather than as ordered tokens.
- **Action-embedding policy head, not a fixed action vocabulary.** The legal action set is
  variable (card x target, and targeting depends on the card), so the head scores candidates
  as ``<query, action_embedding> / sqrt(d)``. An action embedding is composed from its hand
  card token plus its target enemy token plus a kind embedding, which is what lets the same
  head handle targeting and a changing hand without a combinatorial output layer.
- **Auxiliary heads.** One bit of win/loss per combat is a thin signal; predicting hp_delta
  and final HP gives dense per-step supervision on the same trunk, which is where most of the
  sample efficiency comes from at low data volumes.

Capacity is deliberately small (~5-15M). Nothing here is capacity-bound: the whole game is a
few hundred cards and relics. Parameters buy nothing that search and sample count do not.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ACTION_WIDTH, ENEMY_WIDTH
from .format import CARD_TOKEN_WIDTH, GLOBALS

# Indices into a card token (see format.CARD_TOKEN).
C_CARD, C_UPGRADE, C_ENCH, C_ENCH_AMT, C_ENCH_DIS = 0, 1, 2, 3, 4
C_COST, C_COST_X, C_STAR, C_TYPE, C_RARITY, C_TARGET, C_PLAYABLE = 5, 6, 7, 8, 9, 10, 11
CARD_CONT = [C_UPGRADE, C_ENCH_AMT, C_ENCH_DIS, C_COST, C_COST_X, C_STAR, C_PLAYABLE]


def use_stable_attention() -> None:
    """Forces the math SDPA backend.

    ROCm on gfx1100 (7900 XT) warns that its mem-efficient attention is experimental, and a
    40-iteration run died at iteration 29 with
    ``HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016`` - a GPU-side hardware exception, not a
    Python error. The math backend is slower but does not use those kernels. At this model
    size (6.5M, seq ~60) attention is nowhere near the bottleneck, so the trade is free.
    """
    for name, enable in (("enable_mem_efficient_sdp", False),
                         ("enable_flash_sdp", False),
                         ("enable_math_sdp", True)):
        fn = getattr(torch.backends.cuda, name, None)
        if fn is not None:
            try:
                fn(enable)
            except Exception:
                pass


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def migrate_enemy_proj(sd, d_model=None):
    """Zero-pads enemy_proj's first layer for the appended enemy-power block.

    The pooled power vector is concatenated LAST, so every pre-existing input column keeps its
    index and the migration is a right-pad -- unlike the run-net's scalar block, which sits
    first and shifts everything after it. New columns are zero, so a migrated checkpoint
    scores IDENTICALLY until it learns to use enemy powers.
    """
    k = "enemy_proj.0.weight"
    W = sd.get(k)
    if W is None:
        return sd
    d = d_model or W.shape[0]
    want = W.shape[1] + d
    own_in = W.shape[1]
    if own_in % 1 or W.shape[1] >= want:
        return sd
    new = torch.zeros(W.shape[0], want, dtype=W.dtype)
    new[:, :own_in] = W
    sd = dict(sd)
    sd[k] = new
    print(f"  migrated {k}: {tuple(W.shape)} -> {tuple(new.shape)} "
          f"({d} new enemy-power column(s), zero-initialised)")
    return sd


class CombatNet(nn.Module):
    def __init__(self, sizes: dict, d_model: int = 256, layers: int = 6, heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d = d_model
        nc = sizes["cards"]

        self.card_emb = nn.Embedding(nc, d_model)
        self.ench_emb = nn.Embedding(sizes["enchantments"], d_model // 4)
        self.type_emb = nn.Embedding(sizes["card_types"], d_model // 8)
        self.rarity_emb = nn.Embedding(sizes["card_rarities"], d_model // 8)
        self.target_emb = nn.Embedding(sizes["target_types"], d_model // 8)
        self.card_proj = mlp(d_model + d_model // 4 + 3 * (d_model // 8) + len(CARD_CONT),
                             d_model, d_model)

        self.monster_emb = nn.Embedding(sizes["monsters"], d_model)
        # + d_model for the pooled ENEMY POWER vector, appended LAST so an older checkpoint
        # migrates by zero-padding on the right rather than by re-laying-out every column.
        self.enemy_proj = mlp(d_model + ENEMY_WIDTH - 1 + d_model, d_model, d_model)

        self.relic_emb = nn.Embedding(sizes["relics"], d_model)
        # +2 for (counter, melted): which relics you hold is not enough, their
        # charge state changes how they should be played around.
        self.relic_proj = mlp(d_model + 2, d_model, d_model)
        self.power_emb = nn.Embedding(sizes["powers"], d_model)
        # ZERO-initialised on purpose: the action vector is a SUM, so a zero embedding leaves
        # every existing action's score bit-identical while still receiving gradient. That
        # makes potion identity a strictly additive capability rather than a perturbation of
        # a policy that already works.
        self.potion_emb = nn.Embedding(sizes["potions"], d_model)
        nn.init.zeros_(self.potion_emb.weight)
        self.power_proj = mlp(d_model + 1, d_model, d_model)

        # Bags are dense over the card vocab; one linear is enough and keeps it cheap.
        self.bag_proj = mlp(3 * nc, d_model, d_model)
        self.global_proj = mlp(len(GLOBALS), d_model, d_model)

        # Learned type markers so the encoder can tell a hand card from a relic.
        self.seg = nn.Parameter(torch.randn(6, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * ff_mult, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.norm = nn.LayerNorm(d_model)

        # Policy: query from CLS, keys are composed action embeddings.
        # 0=play 1=end_turn 2=select_card 3=select_done. Sized 2 originally, which
        # silently clamped both select kinds onto end_turn.
        # 8 slots, not 4: the C# side now emits kind_idx 4 for "use_potion" when the run
        # policy opens the potion gate. _ix clamps out-of-range indices, so a wider table is
        # not needed for safety - it is needed so a potion action gets its OWN embedding
        # instead of being clamped onto "select_done" and read as a different action.
        # Existing checkpoints load through load_compat, which COPIES the overlapping rows
        # and freshly initialises only the new ones, so a trained policy keeps its meaning.
        self.kind_emb = nn.Embedding(8, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.act_proj = mlp(d_model, d_model, d_model)

        self.value_head = mlp(d_model, d_model, 1)        # P(win this combat)
        self.aux_hp_head = mlp(d_model, d_model, 1)       # hp_delta of the chosen step
        self.aux_endhp_head = mlp(d_model, d_model, 1)    # final HP fraction

    # ---- token builders -------------------------------------------------------------

    @staticmethod
    def _ix(t: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        """Clamps an index into an embedding's valid range.

        Clamping only at 0 leaves the upper bound unchecked, and an out-of-range gather is a
        GPU-side fault that surfaces as an opaque driver abort rather than an exception.
        """
        return t.clamp(0, emb.num_embeddings - 1)

    def _hand_tokens(self, hand: torch.Tensor) -> torch.Tensor:
        e = torch.cat([
            self.card_emb(self._ix(hand[..., C_CARD], self.card_emb)),
            self.ench_emb(self._ix(hand[..., C_ENCH], self.ench_emb)),
            self.type_emb(self._ix(hand[..., C_TYPE], self.type_emb)),
            self.rarity_emb(self._ix(hand[..., C_RARITY], self.rarity_emb)),
            self.target_emb(self._ix(hand[..., C_TARGET], self.target_emb)),
            hand[..., CARD_CONT].float(),
        ], -1)
        return self.card_proj(e)

    def _enemy_tokens(self, enemies: torch.Tensor, epow: torch.Tensor,
                      epow_mask: torch.Tensor) -> torch.Tensor:
        # Each enemy's powers are embedded through the SAME power_emb/power_proj the player's
        # powers use -- Strength means the same thing on either side of the table, so sharing
        # the weights means enemy powers arrive already meaningful rather than starting cold.
        pw = self.power_proj(torch.cat([
            self.power_emb(self._ix(epow[..., 0], self.power_emb)),
            epow[..., 1:].float()], -1))                       # (B, E, P, d)
        m = epow_mask.unsqueeze(-1).float()
        pooled = (pw * m).sum(2) / m.sum(2).clamp(min=1.0)     # (B, E, d), masked mean
        e = torch.cat([
            self.monster_emb(self._ix(enemies[..., 0], self.monster_emb)),
            enemies[..., 1:].float(),
            pooled,
        ], -1)
        return self.enemy_proj(e)

    def forward(self, b: dict) -> dict:
        hand = self._hand_tokens(b["hand"]) + self.seg[0]
        enemy = self._enemy_tokens(b["enemies"], b["enemy_powers"],
                                   b["enemy_power_mask"]) + self.seg[1]
        relic = self.relic_proj(torch.cat([
            self.relic_emb(self._ix(b["relics"][..., 0], self.relic_emb)),
            b["relics"][..., 1:].float()], -1)) + self.seg[2]
        power = self.power_proj(torch.cat([
            self.power_emb(self._ix(b["powers"][..., 0], self.power_emb)),
            b["powers"][..., 1:].float()], -1)) + self.seg[3]

        bsz = hand.shape[0]
        bag = (self.bag_proj(b["bags"].flatten(1)) + self.seg[4]).unsqueeze(1)
        glob = (self.global_proj(b["globals"]) + self.seg[5]).unsqueeze(1)
        cls = self.cls.expand(bsz, -1, -1)

        tokens = torch.cat([cls, glob, bag, hand, enemy, relic, power], 1)
        ones = torch.ones(bsz, 3, dtype=torch.bool, device=hand.device)  # cls, glob, bag
        keep = torch.cat([ones, b["hand_mask"], b["enemy_mask"],
                          b["relic_mask"], b["power_mask"]], 1)

        h = self.encoder(tokens, src_key_padding_mask=~keep)
        h = self.norm(h)

        n_hand, n_enemy = hand.shape[1], enemy.shape[1]
        h_hand = h[:, 3:3 + n_hand]
        h_enemy = h[:, 3 + n_hand:3 + n_hand + n_enemy]
        pooled = h[:, 0]

        logits = self._policy(pooled, h_hand, h_enemy, b["actions"], b["action_mask"])
        return dict(
            logits=logits,
            value=self.value_head(pooled).squeeze(-1),
            aux_hp=self.aux_hp_head(pooled).squeeze(-1),
            aux_endhp=self.aux_endhp_head(pooled).squeeze(-1),
        )

    def _policy(self, pooled, h_hand, h_enemy, actions, action_mask):
        """Scores each candidate action by dot product with a query from the pooled state.

        Two things to be careful about here:

        1. `hand_i` is only a HAND position for `play` actions. A `select_card` action uses
           that field to index the PROMPT'S OPTION LIST, which can be far longer than the
           padded hand (a whole deck under Pandora's Box). Gathering with it unclamped reads
           out of bounds - which the GPU reports as an opaque hardware exception rather than
           an index error. Clamp to the tensor width AND zero the contribution for non-play
           kinds, where a hand position is meaningless.

        2. The action's own card identity is used directly, so select actions are
           distinguishable at all: without it every option in a prompt scores identically.
        """
        b, a, _ = actions.shape
        kinds = actions[..., 0]
        hand_i = actions[..., 1]
        tgt_i = actions[..., 2]

        h_wide, e_wide = h_hand.shape[1], h_enemy.shape[1]
        hv = h_hand.gather(1, hand_i.clamp(0, h_wide - 1).unsqueeze(-1).expand(-1, -1, self.d))
        hv = hv * ((kinds == 0) & (hand_i >= 0)).unsqueeze(-1).float()
        tv = h_enemy.gather(1, tgt_i.clamp(0, e_wide - 1).unsqueeze(-1).expand(-1, -1, self.d))
        tv = tv * (tgt_i >= 0).unsqueeze(-1).float()
        cv = self.card_emb(self._ix(actions[..., 3], self.card_emb))
        # Which potion this action drinks. Zero for every non-potion action.
        pv = self.potion_emb(self._ix(actions[..., 4], self.potion_emb))

        act = self.act_proj(self.kind_emb(self._ix(kinds, self.kind_emb))
                            + hv + tv + cv + pv)
        q = self.q_proj(pooled).unsqueeze(1)
        logits = (act * q).sum(-1) / math.sqrt(self.d)
        neg = torch.finfo(logits.dtype).min / 4
        return logits.masked_fill(~action_mask, neg)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
