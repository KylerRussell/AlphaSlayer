"""Why does act 3's win rate fall ~0.27 over a single curriculum pass?

Measured on logs/r4.log, pooled over two passes and weighted by fight count:

    act 1  -0.028 per pass (z -10.7)
    act 2  +0.048 per pass (z  +8.4)
    act 3  -0.270 per pass (z -23.0)   <- collapses

Act 3 is the act with the FEWEST fights per iteration (25, against act 1's 114) but it carries
an equal share of the gradient, because the loss averages per act. So its contribution is
equally weighted but roughly five times noisier. Two candidate causes, and they need
different fixes:

  NOISE        act 3's own updates are high-variance, and 25 equally-weighted noisy updates
               walk the policy downhill. Fix: give act 3 as many FIGHTS as the others, so
               equal weight rests on an equally good estimate.
  INTERFERENCE training on acts 1 and 2 (139 of 200 fights) degrades act 3 regardless of how
               act 3 itself is sampled. Fix: that is catastrophic interference and needs a
               different remedy (replay, smaller steps, or separate heads).

This isolates them: run act 3 ALONE at its current fight count, and act 3 alone at a high
fight count. If alone-and-noisy still decays, it is act 3's own update variance. If alone
does not decay at all, the decay comes from the other acts.

    python diag_pass_decay.py --decks run_r4_harvest.jsonl --init combat_r4.pt
"""
from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import torch

import train_run as T
from alphaslayer import curriculum as curr
from alphaslayer.model import CombatNet, use_stable_attention
from alphaslayer.modcheck import check as modcheck
from train_rl import load_compat


def slope(points):
    """Regression of win rate on iteration index -> (change per 15 iterations, z).

    ONE observation per iteration. An earlier version weighted by fight count and also took
    the residual variance from those weights, which treats every individual fight as an
    independent draw from the regression line and inflates z by ~sqrt(fights per iteration).
    It turned noise into "z = -23" and made two act-1/act-2 trends look real that were not.
    The win rate of an iteration is one noisy point; that is all it is.
    """
    n = len(points)
    if n < 4:
        return 0.0, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    resid = sum((y - (my + b * (x - mx))) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(max(resid, 1e-12) / max(n - 2, 1) / sxx)
    return b * 15, (b / se if se else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--decks", required=True)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--focus", type=int, default=2,
                    help="0-based act to isolate (2 = act 3)")
    ap.add_argument("--envs", type=int, default=20)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    use_stable_attention()
    modcheck()
    dev = torch.device(a.device)
    ck = torch.load(a.init, map_location=dev, weights_only=False)
    T._ENC = T.Encoder(ck["sizes"]["cards"])

    all_acts = {act: curr.load_decks(a.decks, act=act) for act in (0, 1, 2)}
    print("harvest: " + ", ".join(f"act{k + 1}={len(v)}" for k, v in all_acts.items()))

    pargs = SimpleNamespace(epochs=2, minibatch=512, clip=0.2, vf_coef=0.5, ent_coef=0.01,
                            per_kind_loss=True, per_kind_entropy=True, skip_ent_coef=0.0,
                            combat_kl=0.05)

    def fresh():
        """Every arm starts from the SAME weights, or the arms are not comparable."""
        net = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev)
        load_compat(net, ck["model"])
        ref = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev)
        load_compat(ref, ck["model"])
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return net, ref, torch.optim.AdamW(net.parameters(), lr=5e-5)

    f = a.focus
    fname = f"act{f + 1}"
    arms = [
        (f"{fname} alone, 25 fights  (sparse sampling)  ", {f: all_acts[f]}, 25),
        (f"{fname} alone, 100 fights (4x the samples)   ", {f: all_acts[f]}, 100),
        (f"{fname} with the other acts (the real pass)  ", all_acts, 200),
    ]
    results = {}
    for label, decks, fights in arms:
        net, ref, opt = fresh()
        series = {}

        def log(msg, _s=series):
            print("   " + msg, flush=True)
            import re
            m = re.search(r"curriculum (\d+)/(\d+): (.*?)\|", msg)
            if not m:
                return
            i = int(m.group(1))
            for act_s, v, c in re.findall(r"a(\d)=([0-9.]+)\((\d+)\)", m.group(3)):
                _s.setdefault(int(act_s), []).append((i, float(v), int(c)))

        print(f"\n=== {label}")
        curr.train_pass(net, opt, decks, device=dev, combat_forward=T.combat_forward,
                        sample=T.sample, ppo_update=T.ppo_update, pargs=pargs, ref=ref,
                        iters=a.iters, fights_per_iter=fights, envs=a.envs, real_hp=True,
                        log=log)
        results[label] = series

    print(f"\n\nTREND WITHIN A PASS  (negative = the pass makes that act worse)")
    print(f"  {'arm':<46s} {'act':>4s} {'change/15 iters':>16s} {'z':>7s}")
    for label, series in results.items():
        for act_n in sorted(series):
            b, z = slope(series[act_n])
            star = "  <- focus" if act_n == f + 1 else ""
            print(f"  {label:<46s} {act_n:>4d} {b:+16.3f} {z:+7.2f}{star}")
    print("\nreading: a slope near zero means the pass no longer undoes itself for that act.")


if __name__ == "__main__":
    main()
