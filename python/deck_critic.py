"""Offline study: a DECK-AWARE run critic vs the current one (alphaslayer/unified/critic.py).

The current critic sees act, floor, the context features, the character and the act boss,
but not the deck. A card pick changes only the deck, so the critic cannot tell a good pick
from a bad one, and card rewards, shops and campfires get credit only from outcomes many floors
later. This adds, to the same logistic model:

  * a learned value per (card, upgraded) in the deck, weighted by count (EmbeddingBag, sum)
  * a learned value per relic
  * deck size and upgraded-card count

Still linear and L2-regularised: the 18.5M-parameter value head memorises runs (-0.17 skill
out of sample), while a small model generalises (+0.145 without the deck).

Reports out-of-sample Brier skill on held-out ITERATIONS, on run decisions only and all
steps, for: the current critic, + deck, + deck + relics. With --save-v it also writes
ONLINE predictions (critic fit on past iterations only, as the trainer would; fights use the
run state "if the fight ended now") for grad_snr.py --learned-phi.

    PYTHONPATH=. .venv7/bin/python deck_critic.py --dumps data/rollouts_long2
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphaslayer.unified import features as FT
from alphaslayer.unified.critic import RunCritic, fight_run_state

COMBAT = FT.DKINDS.index("combat")
N_CARD_ID = 2 * 1024          # (card_idx, upgraded) -> card_idx * 2 + upgraded
N_RELIC_ID = 1024


def deck_bags(it):
    """Per step: deck (ids, counts) and relic ids, from the packed arrays. Fights carry no deck
    tokens (their cards are piles); they get the entering run decision's bags (see below)."""
    off, ids, cf = it["x_off_card"], it["x_card_ids"], it["x_card_f"]
    roff, rids = it["x_off_relic"], it["x_relic_ids"]
    n = len(it["kind"])
    deck, rel, summ = [], [], []
    for r in range(n):
        a, b = off[r], off[r + 1]
        m = ids[a:b, 6] == FT.LOC_DECK
        cid = ids[a:b, 0][m]
        up = (cf[a:b, 0][m] > 0).astype(np.int64)
        cnt = np.expm1(cf[a:b, -1][m]).round()            # last card feature = symlog(count)
        deck.append((np.clip(cid * 2 + up, 0, N_CARD_ID - 1), cnt.astype(np.float32)))
        # what the deck DOES, count-weighted: type and rarity mix, mean card features
        # (upgrade, keyword bits, the 14 value slots, cost...), size
        w = cnt / max(1.0, cnt.sum())
        typ = np.zeros(8, np.float32); rar = np.zeros(8, np.float32)
        np.add.at(typ, np.clip(ids[a:b, 3][m], 0, 7), w)
        np.add.at(rar, np.clip(ids[a:b, 4][m], 0, 7), w)
        mean_f = (cf[a:b][m][:, :-1] * w[:, None]).sum(0) if m.any() else np.zeros(cf.shape[1] - 1, np.float32)
        summ.append(np.concatenate([typ, rar, mean_f, [np.log1p(cnt.sum())]]).astype(np.float32))
        rel.append(np.clip(rids[roff[r]:roff[r + 1]], 0, N_RELIC_ID - 1))
    return deck, rel, summ


class DeckCritic(nn.Module):
    def __init__(self, d_dense, n_key=256 * 3, use_deck=True, use_relics=True):
        super().__init__()
        self.lin = nn.Linear(d_dense, 1)
        self.boss = nn.Embedding(n_key, 1)
        self.card = nn.EmbeddingBag(N_CARD_ID, 1, mode="sum") if use_deck else None
        self.relic = nn.EmbeddingBag(N_RELIC_ID, 1, mode="sum") if use_relics else None
        for m in (self.boss, self.card, self.relic):
            if m is not None:
                nn.init.zeros_(m.weight)

    def forward(self, X, K, deck, rel):
        z = self.lin(X).squeeze(-1) + self.boss(K).squeeze(-1)
        if self.card is not None:
            z = z + self.card(deck[0], deck[1], per_sample_weights=deck[2]).squeeze(-1)
        if self.relic is not None:
            z = z + self.relic(rel[0], rel[1]).squeeze(-1)
        return z

    def l2(self):
        return sum((m.weight ** 2).sum() for m in (self.lin, self.boss, self.card, self.relic)
                   if m is not None)


def flat_bags(bags, rows, weights=True):
    """EmbeddingBag inputs (flat ids, offsets[, weights]) for the given step rows."""
    lens = [len(bags[r][0] if weights else bags[r]) for r in rows]
    offs = torch.as_tensor(np.concatenate([[0], np.cumsum(lens)[:-1]]), dtype=torch.long)
    if weights:
        ids = np.concatenate([bags[r][0] for r in rows]) if rows.size else np.zeros(0, np.int64)
        w = np.concatenate([bags[r][1] for r in rows]) if rows.size else np.zeros(0, np.float32)
        return torch.as_tensor(ids, dtype=torch.long), offs, torch.as_tensor(w)
    ids = np.concatenate([bags[r] for r in rows]) if rows.size else np.zeros(0, np.int64)
    return torch.as_tensor(ids, dtype=torch.long), offs


def fit(model, X, K, deck, rel, y, l2=1e-3, iters=300):
    opt = torch.optim.LBFGS(model.parameters(), max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(X, K, deck, rel), y) + l2 * model.l2()
        loss.backward()
        return loss
    opt.step(closure)
    return model


def skill(p, y):
    base = ((y.mean() - y) ** 2).mean()
    return 1 - ((p - y) ** 2).mean() / base if base > 0 else float("nan")


class Data:
    """All steps of all iterations, with RunCritic's dense features, the boss key and bags.
    Fight rows are given the ENTERING run decision's features/bags with their own hp/potions
    (critic.fight_run_state), i.e. the run state 'if the fight ended now'."""

    def __init__(self, its):
        rc = RunCritic()
        self.X, self.K, self.deck, self.rel, self.y, self.iter, self.isc, self.act = \
            [], [], [], [], [], [], [], []
        self.S, self.Y = [], {}
        for k, it in enumerate(its):
            d, key = rc.features(it["x_ctx_ids"], it["x_ctx_f"], it["act"], it["floor"])
            isc = it["kind"] == COMBAT
            d, key = fight_run_state(d, key, isc, it["run"])
            deck, rel, summ = deck_bags(it)
            last = None
            prev = None
            for r in range(len(isc)):                     # carry bags into fights
                if it["run"][r] != prev:
                    last, prev = None, it["run"][r]
                if isc[r]:
                    if last is not None:
                        deck[r], rel[r], summ[r] = deck[last], rel[last], summ[last]
                else:
                    last = r
            extra = np.stack([[np.log1p(c.sum()), np.log1p(c[i % 2 == 1].sum())] if len(c) else [0.0, 0.0]
                              for i, c in deck], 0).astype(np.float32)
            self.X.append(np.concatenate([d, extra], 1))
            self.S.append(np.stack(summ, 0))
            for t in ("act_clear", "reach_act3"):
                self.Y.setdefault(t, []).append(it[t].astype(np.float32))
            self.K.append(key)
            self.deck += deck
            self.rel += rel
            self.y.append(it["won"].astype(np.float32))
            self.iter.append(np.full(len(isc), k))
            self.isc.append(isc)
            self.act.append(it["act"])
        self.X, self.K, self.y, self.S = map(np.concatenate, (self.X, self.K, self.y, self.S))
        self.Y = {k: np.concatenate(v) for k, v in self.Y.items()}
        self.Y["won"] = self.y
        self.base_dim = self.X.shape[1]
        self.iter, self.isc, self.act = map(np.concatenate, (self.iter, self.isc, self.act))

    def use(self, summary: bool, target: str):
        """Dense block = base features (+ deck summary); target = which outcome."""
        self.Xd = np.concatenate([self.X[:, :self.base_dim], self.S], 1) if summary else self.X[:, :self.base_dim]
        self.yt = self.Y[target]

    def tensors(self, rows, mu, sd):
        X = torch.as_tensor((self.Xd[rows] - mu) / sd)
        K = torch.as_tensor(self.K[rows], dtype=torch.long)
        return X, K, flat_bags(self.deck, rows), flat_bags(self.rel, rows, weights=False), \
            torch.as_tensor(self.yt[rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="data/rollouts_long2")
    ap.add_argument("--test-iters", type=int, default=5)
    ap.add_argument("--combat-keep", type=float, default=0.15)
    ap.add_argument("--l2", default="1e-3")
    ap.add_argument("--targets", default="won,act_clear")
    ap.add_argument("--save-v", default=None, help="online predictions per iteration -> .npz")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--save-summary", action="store_true")
    ap.add_argument("--save-ids", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    its = [dict(np.load(p)) for p in sorted(glob.glob(os.path.join(a.dumps, "it*.npz")))]
    D = Data(its)
    rng = np.random.default_rng(0)
    nT = len(its)
    keep = (~D.isc) | (rng.random(len(D.y)) < a.combat_keep)
    tr = np.flatnonzero(keep & (D.iter < nT - a.test_iters))
    te_run = np.flatnonzero((~D.isc) & (D.iter >= nT - a.test_iters))
    print(f"{nT} iterations; train rows {len(tr)}, test run-decision rows {len(te_run)}")
    variants = (("current critic", False, False, False), ("+ card ids", True, False, False),
                ("+ deck summary", False, False, True), ("+ ids + summary", True, False, True))
    for target in a.targets.split(","):
        for l2 in [float(x) for x in a.l2.split(",")]:
            for name, deck, rel, summ in variants:
                D.use(summ, target)
                mu, sd = D.Xd[tr].mean(0), D.Xd[tr].std(0) + 1e-6
                Xtr, Xte = D.tensors(tr, mu, sd), D.tensors(te_run, mu, sd)
                m = fit(DeckCritic(D.Xd.shape[1], use_deck=deck, use_relics=rel), *Xtr[:2],
                        Xtr[2], Xtr[3], Xtr[4], l2=l2)
                with torch.no_grad():
                    p = torch.sigmoid(m(*Xte[:2], Xte[2], Xte[3])).numpy()
                y, act = D.yt[te_run], D.act[te_run]
                print(f"  {target:10} l2={l2:g} {name:16} run-decisions {skill(p, y):+.3f}  "
                      + " ".join(f"act{c + 1} {skill(p[act == c], y[act == c]):+.3f}" for c in range(3)),
                      flush=True)

    if a.save_v:
        # online, as the trainer would: at iteration k, fit on iterations [k-window, k)
        best_l2 = float(a.l2.split(",")[0])
        D.use(a.save_summary, "won")
        out = {}
        for k in range(3, nT):
            trk = np.flatnonzero(keep & (D.iter < k) & (D.iter >= k - a.window))
            rows = np.flatnonzero(D.iter == k)
            mu_k, sd_k = D.Xd[trk].mean(0), D.Xd[trk].std(0) + 1e-6
            T = D.tensors(trk, mu_k, sd_k)
            m = fit(DeckCritic(D.Xd.shape[1], use_deck=a.save_ids, use_relics=False), *T[:2],
                    T[2], T[3], T[4], l2=best_l2)
            R = D.tensors(rows, mu_k, sd_k)
            with torch.no_grad():
                out[f"v{k}"] = torch.sigmoid(m(*R[:2], R[2], R[3])).numpy().astype(np.float32)
            rr = ~D.isc[rows]
            print(f"  online it {k}: run-decision skill {skill(out[f'v{k}'][rr], D.yt[rows][rr]):+.3f}",
                  flush=True)
        np.savez(a.save_v, **out)
        print(f"saved {len(out)} iterations -> {a.save_v}")


if __name__ == "__main__":
    main()
