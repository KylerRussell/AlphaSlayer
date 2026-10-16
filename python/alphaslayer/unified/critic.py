"""A small run critic: P(win | state) from a handful of run features, refit every iteration.

Why not the network's value head: a run's ~360 states are near duplicates, so a head on the
18.5M-parameter trunk memorises runs. Out of sample it scored -0.17 Brier skill, and fresh MLP
heads on frozen trunk features stayed negative even with 1,440 training runs. A logistic model
on act, floor, the 24 context features, the character and the act boss (per act) scored +0.11
on win overall and +0.29 within act 3 on held-out iterations. The act boss decides most of
it: act-3 runs facing the beatable boss won 45%, the other two 2-3% (python/offline_critic.py,
long2 rollouts).

Used as the RUN POTENTIAL (phi_run = P(win)), which with V = 0 in the TD errors is exactly
actor-critic with this critic (ppo.py module docstring). Fights keep the multiplicative
potential P(win this fight) * phi_run, with phi_run evaluated on the run state "if the fight
ended now" (fight_run_state).

Leakage: the potentials of an iteration's steps are computed BEFORE that iteration's
outcomes join the replay. A critic fit on a run's own outcome would grade the run against
itself and zero its advantages.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def carry_into_fights(v: np.ndarray, is_combat: np.ndarray, run: np.ndarray) -> np.ndarray:
    """Within a fight, the run potential is frozen at its value on ENTERING the fight (the last
    run decision before it, in the same run); the fight head carries the in-fight dynamics.

    Evaluating the critic on in-fight states made TD credit chase its incidental within-fight
    features (energy left, block, turn): on long2 rollouts the combat gradient became very
    consistent (0.61) but pointed AWAY from the hand-potential direction (cosine -0.28).
    A fight's HP cost still lands, at the transition to the post-fight run state.
    """
    out = v.astype(np.float64).copy()
    last = None
    prev_run = None
    for i in range(len(v)):
        if run[i] != prev_run:
            last, prev_run = None, run[i]
        if is_combat[i]:
            if last is not None:
                out[i] = last
        else:
            last = out[i]
    return out


# Columns of RunCritic.features' dense block that a fight changes and the run keeps afterwards:
# hp (symlog), max hp (symlog), hp fraction, potion count. Offsets: 3 act + 1 floor, then ctx_f.
_PERSIST = [4 + 0, 4 + 1, 4 + 2, 4 + 17]


def fight_run_state(dense: np.ndarray, key: np.ndarray, is_combat: np.ndarray, run: np.ndarray):
    """For fight rows: the run state "if this fight ended now" = the features of the run
    decision that ENTERED the fight, with the current hp, max hp and potion count.

    The critic then prices a fight's HP and potion spending at every action (dense in-fight
    credit, as the hand potential gave), at the scale of the post-fight run value (so the
    boundary is continuous and dragging a fight out earns nothing), while in-fight features
    the run does not keep (energy, block, turn) cannot be chased. Evaluated raw on fight
    states, the critic made the combat gradient point away from the hand-potential direction
    (cosine -0.28 on long2 rollouts).
    """
    d, k = dense.copy(), key.copy()
    last = None
    prev_run = None
    for i in range(len(d)):
        if run[i] != prev_run:
            last, prev_run = None, run[i]
        if is_combat[i]:
            if last is not None:
                cur = d[i, _PERSIST].copy()
                d[i] = dense[last]
                d[i, _PERSIST] = cur
                k[i] = key[last]
        else:
            last = i
    return d, k


class RunCritic:
    def __init__(self, n_boss: int = 256, n_char: int = 8, window: int = 20, l2: float = 1e-3,
                 combat_keep: float = 0.15, seed: int = 0):
        self.n_boss, self.n_char = n_boss, n_char
        self.window, self.l2, self.combat_keep = window, l2, combat_keep
        self.rng = np.random.default_rng(seed)
        self.buf: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []   # (dense, boss_key, y)
        self.w = None
        self.mu = self.sd = None

    # ---- features ---------------------------------------------------------------------------

    def features(self, ctx_ids: np.ndarray, ctx_f: np.ndarray, act: np.ndarray, floor: np.ndarray):
        """dense (N, D) and a boss-x-act key (N,) into one weight table."""
        act = np.clip(act.astype(np.int64), 0, 2)
        char = np.clip(ctx_ids[:, 1].astype(np.int64), 0, self.n_char - 1)
        boss = np.clip(ctx_ids[:, 3].astype(np.int64), 0, self.n_boss - 1)
        dense = np.concatenate([np.eye(3, dtype=np.float32)[act],
                                (floor.astype(np.float32) / 50.0)[:, None],
                                ctx_f.astype(np.float32),
                                np.eye(self.n_char, dtype=np.float32)[char]], 1)
        return dense, boss * 3 + act

    # ---- fit / predict ---------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self.w is not None

    def add(self, dense, key, y, is_combat):
        """One iteration's steps; combat rows thinned (they are ~80% of steps and near-repeats)."""
        keep = (~is_combat) | (self.rng.random(len(y)) < self.combat_keep)
        self.buf.append((dense[keep], key[keep], y[keep].astype(np.float32)))
        self.buf = self.buf[-self.window:]

    def fit(self, iters: int = 200):
        X = np.concatenate([b[0] for b in self.buf])
        K = np.concatenate([b[1] for b in self.buf])
        y = np.concatenate([b[2] for b in self.buf])
        if y.min() == y.max():
            return          # one outcome class only: nothing to learn yet
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.as_tensor((X - self.mu) / self.sd)
        Kt = torch.as_tensor(K)
        yt = torch.as_tensor(y)
        w = torch.zeros(X.shape[1], requires_grad=True)
        wb = torch.zeros(self.n_boss * 3, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        if self.w is not None:                      # warm start
            with torch.no_grad():
                w.copy_(self.w[0]); wb.copy_(self.w[1]); b.copy_(self.w[2])
        opt = torch.optim.LBFGS([w, wb, b], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = Xt @ w + wb[Kt] + b
            loss = (F.binary_cross_entropy_with_logits(z, yt)
                    + self.l2 * ((w * w).sum() + (wb * wb).sum()))
            loss.backward()
            return loss
        opt.step(closure)
        self.w = (w.detach(), wb.detach(), b.detach())

    @torch.no_grad()
    def predict(self, dense, key) -> np.ndarray:
        w, wb, b = self.w
        z = torch.as_tensor((dense - self.mu) / self.sd) @ w + wb[torch.as_tensor(key)] + b
        return torch.sigmoid(z).numpy()
