"""Fight search on the Python side (probe FightSearch / RunServe.SearchTurnAsync).

The probe simulates choices from a restored fight and asks the server three things, all
answered here with the same network:

- ``prior``: the policy's probabilities at the searched decision (they pick the candidates);
- ``roll``: the policy's choice inside a rollout (the rest of the turn; not part of any
  trajectory);
- ``eval``: a value for every leaf. A leaf is the start of the next turn after the enemies
  have acted (turn mode) or the end of the fight. Value = the training potential
  phi = P(win this fight) * (w_act * (act + 1) + w_hp * hp%); a won fight is P = 1 at its final
  hp, a lost one 0.

The decision that follows carries ``q`` (mean leaf value per candidate, -1e9 for the rest).
``improved`` turns it into the search policy pi'(a) ~ pi(a) exp(q(a) / tau) over the
candidates: the AlphaZero-style target the policy is distilled toward (M9), and what search
play picks from.
"""
from __future__ import annotations

import types

import numpy as np
import torch
import torch.nn.functional as F

from alphaslayer.unified import features as FT
from alphaslayer.unified import ppo as P
from alphaslayer.unified.net import to_torch

NO_Q = -1e8          # q below this: the choice was not simulated


def probe_args(rooms: str, mode: str = "turn", k: int = 4, mass: float = 0.95, samples: int = 2,
               cands: int = 12, yield_every: int = 0, frame_suspend: bool = True) -> list[str]:
    """Game flags for search in ``rooms`` ("Elite,Boss"; "" = off). frame_suspend: the
    harness polls once per frame while searching (waiting for a settled decision) instead of
    on a 1 ms timer: replays ~2x faster, 3,310/3,310 restores exact."""
    if not rooms:
        return []
    return ["--probe-fight-restore", f"--probe-search={rooms}", f"--probe-search-mode={mode}",
            f"--probe-search-cands={cands}", f"--probe-search-k={k}", f"--probe-search-mass={mass}",
            f"--probe-search-samples={samples}", f"--probe-search-yield-every={yield_every}"] + \
        (["--probe-search-frame-suspend"] if frame_suspend else [])


@torch.no_grad()
def log_prior(net, vocab, dev, obs: list, legal: list, kinds: list | None = None) -> list[np.ndarray]:
    """The policy's log-probabilities (fp32), one array per decision (combat unless ``kinds``)."""
    kinds = kinds or ["combat"] * len(obs)
    b = to_torch(FT.encode_batch(list(zip(kinds, obs, legal)), vocab), dev)
    net.eval()
    lp = F.log_softmax(net(b, loop=1)["logits"].float(), -1).cpu().numpy()
    return [lp[j][:len(legal[j])] for j in range(len(obs))]


@torch.no_grad()
def leaf_values(net, vocab, dev, leaves: list, w_hp: float, w_act: float, w_fight: float) -> list[float]:
    vals = [0.0] * len(leaves)
    for j, lf in enumerate(leaves):
        if lf.get("end") == "enemies_cleared":
            vals[j] = w_act * (lf["act"] + 1) + w_hp * lf["hp"] / max(1, lf["max_hp"])
    live = [j for j, lf in enumerate(leaves) if "obs" in lf]
    if live:
        b = to_torch(FT.encode_batch([("combat", leaves[j]["obs"], leaves[j]["legal"]) for j in live],
                                     vocab), dev)
        net.eval()
        # fp32: in bf16 the small differences between sibling leaves round to exact ties
        pf = torch.sigmoid(net(b, loop=1)["aux"]["fight_win"].float()).cpu().numpy()
        for j, p in zip(live, pf):
            obs = dict(leaves[j]["obs"])
            obs.setdefault("act", leaves[j]["act"])
            vals[j] = P.potential("combat", obs, 0.0, w_hp, w_act, w_fight, float(p))
    return vals


@torch.no_grad()
def eval_and_prior(net, vocab, dev, leaves: list, pobs: list, plegal: list, w_hp: float, w_act: float,
                   w_fight: float) -> tuple[list[float], list[np.ndarray]]:
    """leaf_values(leaves) and the prior (probabilities) of each (pobs, plegal) decision in ONE
    fp32 forward: the server answers both kinds of request with a single network pass."""
    vals = [0.0] * len(leaves)
    for j, lf in enumerate(leaves):
        if lf.get("end") == "enemies_cleared":
            vals[j] = w_act * (lf["act"] + 1) + w_hp * lf["hp"] / max(1, lf["max_hp"])
    live = [j for j, lf in enumerate(leaves) if "obs" in lf]
    rows = [("combat", leaves[j]["obs"], leaves[j]["legal"]) for j in live] + \
           [("combat", o, l) for o, l in zip(pobs, plegal)]
    priors = []
    if rows:
        net.eval()
        out = net(to_torch(FT.encode_batch(rows, vocab), dev), loop=1)
        pf = torch.sigmoid(out["aux"]["fight_win"].float()).cpu().numpy()
        lp = F.log_softmax(out["logits"].float(), -1).cpu().numpy() if pobs else None
        for r, j in enumerate(live):
            obs = dict(leaves[j]["obs"])
            obs.setdefault("act", leaves[j]["act"])
            vals[j] = P.potential("combat", obs, 0.0, w_hp, w_act, w_fight, float(pf[r]))
        for r, lg in enumerate(plegal):
            priors.append(np.exp(lp[len(live) + r][:len(lg)]))
    return vals, priors


def make_search(net, vocab, dev, roll_policy, w_hp: float = 0.1, w_act: float = 1 / 3,
                w_fight: float = 1.0):
    """The object RunPool.run(search=...) answers the probe with. ``roll_policy(obs, legal)``
    returns the rollout choices (the greedy policy, as in play)."""
    return types.SimpleNamespace(
        evaluate=lambda leaves: leaf_values(net, vocab, dev, leaves, w_hp, w_act, w_fight),
        prior=lambda obs, legal: [np.exp(x) for x in log_prior(net, vocab, dev, obs, legal)],
        eval_prior=lambda leaves, obs, legal: eval_and_prior(net, vocab, dev, leaves, obs, legal,
                                                             w_hp, w_act, w_fight),
        roll=roll_policy)


def improved(lp: np.ndarray, q, tau: float) -> np.ndarray | None:
    """pi'(a) ~ pi(a) exp(q(a) / tau) over the simulated choices (0 elsewhere); None if no
    choice was simulated. ``lp``: the policy's log-probs over the legal choices."""
    q = np.asarray(q, np.float64)[:len(lp)]
    ok = q > NO_Q
    if not ok.any():
        return None
    sc = np.where(ok, np.asarray(lp, np.float64) + np.where(ok, q, 0.0) / tau, -np.inf)
    sc -= sc.max()
    p = np.exp(sc)
    return (p / p.sum()).astype(np.float32)
