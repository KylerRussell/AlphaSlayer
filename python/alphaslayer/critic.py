"""Predicts how a run STATE will fare, so the run policy can tell a good pickup from a bad one.

Not just decks: the same question is asked of a shop relic or a shop card, and of the choice
to buy nothing. Anything that changes the player's loadout can be scored by asking what the
loadout would then be worth.

The target is HP PRESERVED, not win rate. Win rate saturates: measured against a competent
combat policy at A0, a deck with three good cards and a deck with three mediocre ones both
score 1.00, so the label carries no information exactly where the decision is hardest.
Fraction of health remaining is continuous and keeps discriminating -- winning a fight at 90%
health is genuinely better than winning it at 20%, and it is also the currency the rest of the
run spends. A loss scores 0, so the measure folds win/loss and margin into one number.

One prediction per room type: a deck can be fine against packs of small monsters and hopeless
against a boss, and averaging that away would hide the thing that decides runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ROOMS = ("regular", "elite", "boss")


# Context the loadout alone does not carry. A deck that is fine in act 1 can be hopeless in
# act 3, and a shop decision depends on how much gold is left; without these the critic would
# be asked to predict something its inputs cannot determine.
CTX = 7


def encode_decks(decks, relics, chars, n_cards, n_relics, char_ix, device,
                 max_cards=64, max_relics=16, ctx=None):
    """(list of [(card_idx, upgrade)], list of [relic_idx], list of character) -> tensors.

    ``ctx`` is an optional list of dicts with act / floor / hp_frac / gold / ascension.
    """
    B = len(decks)
    cid = torch.zeros(B, max_cards, dtype=torch.long)
    cup = torch.zeros(B, max_cards)
    cmask = torch.zeros(B, max_cards)
    rid = torch.zeros(B, max_relics, dtype=torch.long)
    rmask = torch.zeros(B, max_relics)
    ch = torch.zeros(B, dtype=torch.long)
    scal = torch.zeros(B, CTX)

    for b, (d, rl, c) in enumerate(zip(decks, relics, chars)):
        for i, (ci, up) in enumerate(d[:max_cards]):
            cid[b, i] = max(0, min(int(ci), n_cards - 1))
            cup[b, i] = float(up)
            cmask[b, i] = 1.0
        for i, r in enumerate(rl[:max_relics]):
            rid[b, i] = max(0, min(int(r), n_relics - 1))
            rmask[b, i] = 1.0
        ch[b] = char_ix.get(c, 0)
        scal[b, 0] = len(d) / 40.0
        scal[b, 1] = sum(u for _, u in d) / 20.0
        scal[b, 2] = len(rl) / 15.0
        cx = (ctx[b] if ctx else None) or {}
        scal[b, 3] = float(cx.get("act", 0)) / 3.0
        scal[b, 4] = float(cx.get("act_floor", 0)) / 17.0
        scal[b, 5] = float(cx.get("hp_frac", 1.0))
        scal[b, 6] = float(cx.get("ascension", 0)) / 20.0

    return (cid.to(device), cup.to(device), cmask.to(device),
            rid.to(device), rmask.to(device), ch.to(device), scal.to(device))


class Critic(nn.Module):
    def __init__(self, n_cards, n_relics, n_chars=8, d_model=256, layers=3, n_ctx=CTX):
        super().__init__()
        self.card_emb = nn.Embedding(n_cards, d_model)
        self.relic_emb = nn.Embedding(n_relics, d_model)
        self.char_emb = nn.Embedding(n_chars, d_model)
        # Upgrade level enters as a learned offset rather than a separate token: an upgraded
        # card is the same card played better, and tying them shares the statistics.
        self.up_proj = nn.Linear(1, d_model)
        # A second, ORDER-FREE view of the deck: max-pool alongside the mean. A mean alone
        # cannot see that a deck contains one bomb, because averaging 24 filler cards drowns
        # it -- and "does this deck have a win condition" is most of what makes a deck good.
        blocks = []
        din = 4 * d_model + n_ctx
        for _ in range(layers):
            blocks += [nn.Linear(din, d_model), nn.GELU()]
            din = d_model
        self.trunk = nn.Sequential(*blocks)
        # Two heads per room type: expected hp preserved (the training target, continuous and
        # non-saturating) and win probability (kept as an interpretable auxiliary).
        self.head_hp = nn.Linear(d_model, len(ROOMS))
        self.head_win = nn.Linear(d_model, len(ROOMS))

    def forward(self, batch):
        cid, cup, cmask, rid, rmask, ch, scal = batch
        c = self.card_emb(cid) + self.up_proj(cup.unsqueeze(-1))
        cm = cmask.unsqueeze(-1)
        c_mean = (c * cm).sum(1) / cm.sum(1).clamp(min=1)
        c_max = (c.masked_fill(cm == 0, -1e4)).max(1).values
        r = self.relic_emb(rid)
        rm = rmask.unsqueeze(-1)
        r_mean = (r * rm).sum(1) / rm.sum(1).clamp(min=1)
        h = self.trunk(torch.cat([c_mean, c_max, r_mean, self.char_emb(ch), scal], dim=-1))
        return torch.sigmoid(self.head_hp(h)), torch.sigmoid(self.head_win(h))

    def hp(self, batch):
        return self.forward(batch)[0]

    @torch.no_grad()
    def strength(self, batch, weights=(0.34, 0.33, 0.33)):
        """Single scalar for comparing candidates: a weighted blend over room types."""
        p = self.hp(batch)
        w = torch.tensor(weights, device=p.device, dtype=p.dtype)
        return (p * w).sum(-1)
