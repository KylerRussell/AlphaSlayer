"""M4 core: run-long trajectories, win-only returns and the PPO loss for the unified model.

Kept free of the game and the environment so the credit math is unit-testable
(tests/test_unified_ppo.py).

Credit, in one place:

  * ONE trajectory per run: combat and run decisions interleaved in the order they happened.
    A fight's last decision bootstraps from V(the decision after the fight), so what a fight
    leaves behind (HP, potions, relic counters, gold) is priced by the value of the state it
    hands to the rest of the run. That is what makes B1-B4 in docs/CAPABILITIES.md learnable.
  * WIN-ONLY reward: 1 at the end of a won run, 0 otherwise. With the default gamma = 1 the
    value target is exactly P(win | state), a probability, matching the sigmoid value head.
  * DISCOUNT per FLOOR, not per decision: the factor between consecutive decisions is
    gamma ** (floors advanced). A fight's ~25 decisions share one floor, so they are not
    discounted against each other. gamma = 1 by default; the flag exists for experiments.
  * SHAPING goes into the ADVANTAGE ONLY, never into the value target. phi(s), zero at the
    terminal, behaves like a crude value estimate:
        run decision:    phi_run = w_act * (act + 1) + w_hp * hp%
        fight decision:  phi = (1 - w_fight * (1 - P(win this fight))) * phi_run
    With w_fight = 1 a fight is worth P(win it) times the run state behind it. That makes phi
    CONTINUOUS across fight boundaries: winning a fight (P -> 1) hands over phi_run unchanged,
    entering one costs (1 - P) * phi_run (its expected risk), and dying (P -> 0) meets the
    terminal 0. With w_act = 1/3, phi_run reaches ~1 at act 3, matching the win reward.

    Continuity matters because of lambda. Potential terms telescope over a whole return, so at
    gamma = 1 they leave the optimal policy unchanged (Ng et al. 1999). But a lambda < 1
    advantage with V = 0 equals -phi(s_t) plus (1 - lambda) times a lambda-weighted average of
    FUTURE potentials, so the policy is pushed toward high average potential over the next
    ~1/(1-lambda) decisions. An additive fight term (the first version) DROPPED at the end of every
    won fight, which paid the policy to prolong fights: entropy rose and act-2 clears fell in
    the `shaped` run. With continuous potentials, finishing sooner costs nothing.
  * BOOTSTRAPPING is optional. M4's first runs showed the run-outcome heads have NEGATIVE
    held-out skill (they memorise runs), and bootstrapping from them feeds value noise into
    every advantage. With bootstrap=False, advantages use V = 0 (lambda-discounted shaped credit
    plus the outcome) and value targets are the realised outcome. The trainer switches
    bootstrapping on only once the value head shows skill on runs it has not seen.
  * GAE with lambda per FLOOR, like gamma. The trace decays by lambda ** (floors advanced), so
    the decisions of one fight (one floor) share credit undecayed. Decaying per DECISION acts
    as a hidden discount on decision count once V = 0: it made a longer fight look better
    whenever the future held a loss, which most runs do, and so it paid the policy to stall.
    Per floor, within a fight the credit is the fight's actual outcome, with the in-fight
    potential as a state baseline; across floors it decays and bootstraps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import features as FT

COMBAT = FT.DKINDS.index("combat")


@dataclass
class Step:
    kind: str
    obs: dict
    legal: list
    a: int
    logp: float
    v: float                        # value head P(win) at rollout
    loop: int                       # loop count used at rollout (replayed in the update)
    floor: int
    phi: float = 0.0                # shaping potential of this state
    p_fight: float | None = None    # P(win this fight) from the model's head (fight decisions)
    # filled in when known
    fight_won: int = -1
    fight_hp: float = float("nan")
    boss_fight: bool = False
    won: int = -1
    act_clear: int = -1
    reach_act3: int = -1
    floors_left: float = float("nan")
    adv: float = 0.0                # shaped GAE advantage (policy)
    ret: float = 0.0                # unshaped lambda-return (value target); NaN = unknown
    act: int = 0


def potential(kind: str, obs: dict, w_combat: float, w_hp: float, w_act: float = 0.0,
              w_fight: float = 0.0, p_fight: float | None = None) -> float:
    """phi(s); see the module docstring. ``w_combat`` adds the old (player hp% - enemy hp%)
    term inside fights. It is NOT continuous across fight boundaries, so it defaults off."""
    p = obs.get("player", obs) if kind != "combat" else obs.get("player", {})
    hp = (p.get("hp", 0) or 0) / max(1, p.get("max_hp", 1) or 1)
    phi_run = w_act * (act_of(kind, obs) + 1) + w_hp * hp
    if kind != "combat":
        return phi_run
    pw = 1.0 if p_fight is None else float(p_fight)
    phi = (1.0 - w_fight * (1.0 - pw)) * phi_run
    if w_combat:
        alive = [e for e in (obs.get("enemies") or []) if e.get("alive")]
        ehp = (sum((e.get("hp", 0) or 0) / max(1, e.get("max_hp", 1) or 1) for e in alive)
               / len(alive)) if alive else 0.0
        phi += w_combat * (hp - ehp)
    return phi


def floor_of(kind: str, obs: dict) -> int:
    src = obs if kind == "combat" else obs.get("player", obs)
    return int(src.get("total_floor", obs.get("total_floor", 0)) or 0)


def act_of(kind: str, obs: dict) -> int:
    src = obs if kind == "combat" else obs.get("player", obs)
    return int(src.get("act", obs.get("act", 0)) or 0)


def compute_targets(steps: list, won: bool, gamma: float = 1.0, lam: float = 0.95,
                    v_end: float = 0.0, phi_end: float = 0.0, bootstrap: bool = True,
                    mc_end: float | None = 0.0) -> None:
    """Fills adv (shaped, for the policy) and ret (value target).

    Reward is 1 on the last step of a won run, else 0. After the last step the potential is
    ``phi_end`` (0 when the run is over) and the value is ``v_end`` (0 when the run is over).
    For an ISOLATED curriculum fight that was won, both describe the post-fight run state, so
    the fight is credited exactly as it would be inside a run; a lost fight ends the run (0, 0).

    bootstrap=True: GAE against the value head; ret is the unshaped lambda-return.
    bootstrap=False: the value head is not used (V = 0 in every TD error), so advantages are
    lambda-discounted shaped credit plus the outcome. ret is the realised return plus
    ``mc_end`` (0 for a finished run; None when unknown, as in an isolated fight, giving NaN).
    """
    T = len(steps)
    if T == 0:
        return
    a_sh = a_un = 0.0
    g = None if mc_end is None else float(mc_end)
    for t in reversed(range(T)):
        s = steps[t]
        last = t == T - 1
        r = float(won) if last else 0.0
        if last:
            d, phi_next = 1.0, float(phi_end)
            v_next = float(v_end) if bootstrap else 0.0
        else:
            nxt = steps[t + 1]
            k = max(0, nxt.floor - s.floor)
            d = gamma ** k
            phi_next = nxt.phi
            v_next = nxt.v if bootstrap else 0.0
        # lambda decays per floor advanced, not per decision (see the module docstring).
        lk = 1.0 if last else (lam ** k if k > 0 else 1.0)
        v_here = s.v if bootstrap else 0.0
        delta_un = r + d * v_next - v_here
        delta_sh = delta_un + (d * phi_next - s.phi)
        a_un = delta_un + d * lk * a_un
        a_sh = delta_sh + d * lk * a_sh
        s.adv = a_sh
        if bootstrap:
            s.ret = min(1.0, max(0.0, a_un + s.v))
        else:
            g = None if g is None else r + d * g
            s.ret = float("nan") if g is None else min(1.0, max(0.0, g))


def value_skill(steps: list):
    """(brier, base brier, skill) of the rollout value predictions against the run outcomes.

    Rollout values were predicted BEFORE these runs' outcomes existed, so this is an honest
    out-of-sample measure. skill = 1 - brier/base: 0 means no better than predicting the base
    rate, and negative means worse.
    """
    ys = np.array([s.won for s in steps if s.won >= 0], np.float64)
    vs = np.array([s.v for s in steps if s.won >= 0], np.float64)
    if len(ys) == 0:
        return float("nan"), float("nan"), float("nan")
    brier = float(((vs - ys) ** 2).mean())
    base = float(((ys.mean() - ys) ** 2).mean())
    return brier, base, (1.0 - brier / base) if base > 0 else float("nan")


class RunBuffer:
    """Per-env open runs; closed runs become training steps with every target filled."""

    def __init__(self, gamma=1.0, lam=0.95, w_combat=0.0, w_hp=0.0, w_act=0.0, w_fight=0.0,
                 bootstrap=True):
        self.gamma, self.lam = gamma, lam
        self.w_combat, self.w_hp, self.w_act, self.w_fight = w_combat, w_hp, w_act, w_fight
        self.bootstrap = bootstrap
        self.open: dict[int, list] = {}
        self.fight_open: dict[int, list] = {}
        self.done: list[Step] = []
        self.runs: list[dict] = []

    def add(self, env, kind, obs, legal, a, logp, v, loop, p_fight=None):
        st = Step(kind=kind, obs=obs, legal=legal, a=a, logp=logp, v=v, loop=loop,
                  floor=floor_of(kind, obs), act=act_of(kind, obs), p_fight=p_fight,
                  phi=potential(kind, obs, self.w_combat, self.w_hp, self.w_act,
                                self.w_fight, p_fight))
        self.open.setdefault(env, []).append(st)
        if kind == "combat":
            self.fight_open.setdefault(env, []).append(st)
        return st

    def end_fight(self, env, won, hp_start, hp_end, max_hp, room):
        for st in self.fight_open.pop(env, []):
            st.fight_won = int(won)
            st.fight_hp = (hp_start - hp_end) / max(1, max_hp)
            st.boss_fight = room == "Boss"

    def end_run(self, env, won, act_reached, floors):
        steps = self.open.pop(env, [])
        # A run that dies mid-fight may send no fight_end: that fight was lost.
        for st in self.fight_open.pop(env, []):
            st.fight_won = 0
        for st in steps:
            st.won = int(won)
            st.act_clear = int(won or act_reached > st.act)
            st.reach_act3 = int(won or act_reached >= 2)
            st.floors_left = float(max(0, floors - st.floor))
        compute_targets(steps, won, self.gamma, self.lam, bootstrap=self.bootstrap)
        self.done.extend(steps)
        self.runs.append({"won": int(won), "act": act_reached, "floors": floors,
                          "n": len(steps)})

    def drop_open(self):
        """Runs cut off by a collection deadline have no outcome; they are dropped, not guessed."""
        self.open.clear()
        self.fight_open.clear()


# ---- loss -------------------------------------------------------------------------------------

def _hi(x: torch.Tensor) -> torch.Tensor:
    """Upcasts bf16/fp16 outputs to float32 for the loss; leaves float32/float64 untouched, so the
    loss is computed at no lower precision than the rollout's log-probs were."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def group_mean(per_row: torch.Tensor, kind: torch.Tensor, run_weight: float = 0.5,
               kind_weight: str = "equal"):
    """Combat rows and run rows count equally (combat is ~80% of decisions). Within the run half:

      equal  each decision kind counts equally (r5's per-kind loss). shaped3 showed the cost:
             a campfire (~5% of steps) pulls as hard as all travel, so with noisy advantages the
             rare kinds random-walk (rest entropy 0.00 -> 0.52 with its top action unchanged)
      flat   each DECISION counts equally: standard PPO; a kind's gradient, and its noise, scale
             with how often it occurs
      sqrt   kinds weighted by sqrt(count): in between
    """
    is_c = kind == COMBAT
    c = per_row[is_c].mean() if is_c.any() else None
    ks = [k for k in kind.unique().tolist() if k != COMBAT]
    r = None
    if ks:
        means = torch.stack([per_row[kind == k].mean() for k in ks])
        if kind_weight == "equal":
            r = means.mean()
        else:
            n = torch.tensor([float((kind == k).sum()) for k in ks], device=per_row.device,
                             dtype=means.dtype)
            w = n if kind_weight == "flat" else n.sqrt()
            r = (means * w).sum() / w.sum()
    if c is not None and r is not None:
        return (1 - run_weight) * c + run_weight * r
    return c if c is not None else (r if r is not None else per_row.sum() * 0)


def ppo_loss(out: dict, tgt: dict, clip: float = 0.2, ent_coef: float = 0.01,
             vf_coef: float = 0.5, aux_coef: float = 0.25, run_weight: float = 0.5,
             anchor_logp: torch.Tensor | None = None, kl_coef: float = 0.0,
             kind_weight: str = "equal"):
    """PPO clipped surrogate + value BCE + aux + optional KL toward an anchor policy.

    tgt: a (B,) chosen action, old_logp (B,), adv (B,) already normalised, ret (B,) in [0,1],
    kind (B,), and the aux labels as in train_distill.
    """
    logits = _hi(out["logits"])
    mask = tgt["cand_mask"]
    logp_all = F.log_softmax(logits, -1)
    logp = logp_all.gather(1, tgt["a"].unsqueeze(1)).squeeze(1)
    ratio = (logp - tgt["old_logp"]).clamp(-20, 20).exp()
    s1 = ratio * tgt["adv"]
    s2 = ratio.clamp(1 - clip, 1 + clip) * tgt["adv"]
    kind = tgt["kind"]
    # A zero that stays in the graph. NOT logits.sum()*0: the logits hold the padding mask value
    # (finfo.min/4), whose sum overflows to -inf, and -inf*0 = NaN. Every term that fell back to
    # it (no value targets, no fight rows, no aux labels) then turned the loss NaN, and the
    # trainer's finite-gradient guard silently skipped the update.
    zero = _hi(out["value_logit"]).sum() * 0
    parts = {}
    parts["pg"] = group_mean(-torch.min(s1, s2), kind, run_weight, kind_weight)
    p = logp_all.exp()
    ent = -(p * logp_all).masked_fill(~mask, 0.0).sum(-1)
    parts["ent"] = group_mean(ent, kind, run_weight, kind_weight)
    vm = ~torch.isnan(tgt["ret"])        # unknown value targets (isolated fights, no bootstrap)
    parts["value"] = (F.binary_cross_entropy_with_logits(_hi(out["value_logit"])[vm], tgt["ret"][vm])
                      if vm.any() else zero)
    aux = out["aux"]
    # Run-level labels are absent (-1 / NaN) on isolated curriculum fights; mask them.
    def bce(head, lab):
        m = lab >= 0
        return (F.binary_cross_entropy_with_logits(_hi(aux[head])[m], lab[m]) if m.any() else zero)
    parts["act_clear"] = bce("act_clear", tgt["act_clear"])
    parts["reach_act3"] = bce("reach_act3", tgt["reach_act3"])
    fl = ~torch.isnan(tgt["floors_left"])
    parts["floors_left"] = (F.mse_loss(_hi(aux["floors_left"])[fl], tgt["floors_left"][fl] / 51.0)
                            if fl.any() else zero)
    fm = tgt["fight_won"] >= 0
    parts["fight_win"] = (F.binary_cross_entropy_with_logits(_hi(aux["fight_win"])[fm],
                                                             tgt["fight_won"][fm]) if fm.any() else zero)
    hm = ~torch.isnan(tgt["fight_hp"])
    parts["fight_hp"] = (F.mse_loss(_hi(aux["fight_hp_loss"])[hm], tgt["fight_hp"][hm])
                         if hm.any() else zero)
    parts["kl"] = zero
    if anchor_logp is not None and kl_coef > 0:
        pa = anchor_logp.exp()
        kl = (pa * (anchor_logp - logp_all)).masked_fill(~mask, 0.0).sum(-1)
        parts["kl"] = group_mean(kl, kind, run_weight, kind_weight)
    total = (parts["pg"] - ent_coef * parts["ent"] + vf_coef * parts["value"]
             + aux_coef * sum(parts[k] for k in ("act_clear", "reach_act3", "floors_left",
                                                 "fight_win", "fight_hp"))
             + kl_coef * parts["kl"])
    with torch.no_grad():
        parts["clipfrac"] = ((ratio - 1).abs() > clip).float().mean()
        parts["approx_kl"] = (tgt["old_logp"] - logp).mean()
    return total, parts


def normalise_advantages(steps: list) -> np.ndarray:
    """Per-group normalisation (combat vs run), so neither half's scale dominates."""
    adv = np.array([s.adv for s in steps], np.float64)
    is_c = np.array([s.kind == "combat" for s in steps])
    out = np.zeros_like(adv)
    for m in (is_c, ~is_c):
        if m.sum() > 1:
            x = adv[m]
            out[m] = (x - x.mean()) / (x.std() + 1e-8)
        elif m.sum() == 1:
            out[m] = 0.0
    return out.astype(np.float32)
