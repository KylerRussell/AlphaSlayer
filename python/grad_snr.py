"""How much of a 96-run policy gradient is signal? Measured on dumped rollouts of ONE policy.

train_unified.py --collect-only --dump-steps DIR plays N iterations without updating, so each
iteration's policy gradient g_i is an independent estimate of the same expected gradient G.
For two different iterations E[g_i . g_j] = |G|^2, while E|g_i|^2 = |G|^2 + noise/n. So

    consistency c = mean_{i != j} cos(g_i, g_j)       (0 = pure noise, 1 = pure signal)
    B_noise       = n (1 - c) / c   runs               (batch where signal = noise)

per advantage ESTIMATOR, separately for the combat half and the run half of the loss
(gradients of sum_t w_t A_t log pi(a_t | s_t), weighted like the trainer: half 0.5 each,
flat within the half; the ratio is 1 at the rollout policy, so PPO's clip is inactive).

Estimators differ only in A_t, so they are compared on identical data. "agree_mc" is
mean_{i != j} g^E_i . g^MC_j / sqrt(|G^E|^2 |G^MC|^2): the cosine between the estimator's
EXPECTED direction and the unbiased win-only Monte Carlo direction (noisy, but 1 = unbiased).

    PYTHONPATH=. .venv7/bin/python grad_snr.py --dumps data/rollouts_long2 --ckpt unified_m4_long2.pt
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from alphaslayer.model import use_stable_attention
from alphaslayer.unified import dataset as DS
from alphaslayer.unified import features as FT
from alphaslayer.unified import ppo as P
from alphaslayer.unified.critic import carry_into_fights
from alphaslayer.unified.net import UnifiedNet, to_torch

COMBAT = FT.DKINDS.index("combat")
W_ACT, W_HP = 1 / 3, 0.1          # the trainer's potentials, used to back out hp% from phi


def hp_and_run_phi(it):
    """phi_run and hp% per step, recovered from the dumped phi (run: phi_run; fight:
    P(win fight) * phi_run)."""
    is_c = it["kind"] == COMBAT
    pf = np.where(is_c, np.nan_to_num(it["p_fight"], nan=1.0), 1.0)
    phi_run = np.where(is_c, it["phi"] / np.maximum(pf, 1e-6), it["phi"])
    hp = (phi_run - W_ACT * (it["act"] + 1)) / W_HP
    return phi_run, np.clip(hp, 0, 1), pf


def advantages(it, est, v_override=None):
    """Raw (un-normalised) advantages for one iteration under estimator ``est``."""
    phi_run, hp, pf = hp_and_run_phi(it)
    is_c = it["kind"] == COMBAT
    lam, w_hp, w_act, boot = est.get("lam", 0.8), est.get("w_hp", W_HP), est.get("w_act", W_ACT), False
    phi = w_act * (it["act"] + 1) + w_hp * hp
    if est.get("learned_phi"):
        # the run critic's P(win), fit on past iterations; frozen at fight entry unless "raw"
        phi = (v_override if est.get("raw") else
               carry_into_fights(v_override, is_c, it["run"]))
    phi = np.where(is_c, pf * phi, phi) if est.get("w_fight", 1.0) else phi
    if est.get("no_shaping"):
        phi = np.zeros_like(phi)
    v = it["v"] if v_override is None else v_override
    if est.get("critic"):
        boot = True
    if est.get("learned_phi"):
        v = it["v"]                          # V = 0 anyway (bootstrap off)
    out = np.zeros(len(phi), np.float64)
    runs = it["run"]
    for r in range(len(it["run_won"])):
        idx = np.flatnonzero(runs == r)
        if len(idx) == 0:
            continue
        steps = [SimpleNamespace(floor=int(it["floor"][i]), phi=float(phi[i]), v=float(v[i]))
                 for i in idx]
        if "lam_step" in est:
            out[idx] = targets_lam_step(steps, bool(it["run_won"][r]), lam, est["lam_step"])
        else:
            P.compute_targets(steps, bool(it["run_won"][r]), 1.0, lam, bootstrap=boot)
            out[idx] = [s.adv for s in steps]
    if est.get("fight_local"):
        # combat credited by its own fight only: won (1/0) minus P(win this fight)
        out = np.where(is_c, it["fight_won"] - pf, out)
    return out


def targets_lam_step(steps, won, lam, lam_step):
    """compute_targets (V = 0, gamma = 1, shaped) but with a per-DECISION trace decay
    ``lam_step`` between decisions on the same floor: within a fight, credit becomes TD on the
    fight potential P(win this fight) * phi_run instead of the fight's final outcome."""
    T = len(steps)
    a_sh, out = 0.0, [0.0] * T
    for t in reversed(range(T)):
        s, last = steps[t], t == T - 1
        if last:
            r, phi_next, lk = float(won), 0.0, 1.0
        else:
            nxt = steps[t + 1]
            k = max(0, nxt.floor - s.floor)
            r, phi_next = 0.0, nxt.phi
            lk = lam ** k if k > 0 else lam_step
        a_sh = r + phi_next - s.phi + lk * a_sh
        out[t] = a_sh
    return out


def apply_group_baseline(it, adv):
    """ppo.seed_group_baseline on dumped arrays (needs run_group in the dump)."""
    steps = [SimpleNamespace(floor=int(f), kind=int(k), adv=float(x))
             for f, k, x in zip(it["floor"], it["kind"], adv)]
    runs = [{"n": int((it["run"] == r).sum()), "group": int(g)}
            for r, g in enumerate(it["run_group"])]
    P.seed_group_baseline(steps, runs)
    return np.array([s.adv for s in steps])


def normalise(adv, kind, center="kind"):
    steps = [SimpleNamespace(adv=a, kind=FT.DKINDS[k]) for a, k in zip(adv, kind)]
    return P.normalise_advantages(steps, center=center)


ESTIMATORS = {
    "trainer(l=.8)": {},
    "mc_win": {"no_shaping": True, "lam": 1.0},
    "lam=.95": {"lam": 0.95},
    "lam=.5": {"lam": 0.5},
    "fight_local": {"fight_local": True},
    "fight_td(.9)": {"lam_step": 0.9},
    "fight_td(.7)": {"lam_step": 0.7},
    "fight_td(0)": {"lam_step": 0.0},
    "dumped": {"dumped": True},                    # exactly the trainer's advantages
    "dumped+grp": {"dumped": True, "grp": True},
    "hand(ls=.5)": {"lam_step": 0.5},
    "hand(ls=.5)+grp": {"lam_step": 0.5, "grp": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="data/rollouts_long2")
    ap.add_argument("--ckpt", default="unified_m4_long2.pt")
    ap.add_argument("--iters", type=int, default=0, help="use the first N dumps (0 = all)")
    ap.add_argument("--est", default=",".join(ESTIMATORS))
    ap.add_argument("--critic", default=None, help=".npz of cross-fitted V per iteration (see offline_critic)")
    ap.add_argument("--learned-phi", default=None,
                    help=".npz of online run-critic P(win) per iteration (keys v<i>); only "
                         "iterations present are used")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--mem-frac", type=float, default=0.0, help="cap this process's VRAM share")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    use_stable_attention()
    dev = torch.device("cuda:0")
    if a.mem_frac:
        # Share the GPU with a training run: this process OOMs before it can starve the other.
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net = UnifiedNet(ck["sizes"], **ck["cfg"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()                                  # dropout is 0; eval for determinism
    params = [p for p in net.parameters() if p.requires_grad]
    paths = sorted(glob.glob(os.path.join(a.dumps, "it*.npz")))
    if a.iters:
        paths = paths[:a.iters]
    if a.learned_phi:
        have = {int(k[1:]) for k in np.load(a.learned_phi).files}
        idx_paths = [(i, p) for i, p in enumerate(paths) if i in have]
        paths = [p for _, p in idx_paths]
        orig_index = [i for i, _ in idx_paths]
    else:
        orig_index = list(range(len(paths)))
    names = a.est.split(",")
    ests = {n: ESTIMATORS[n] for n in names if n in ESTIMATORS}
    critic = dict(np.load(a.critic)) if a.critic else None
    if critic is not None:
        ests["critic_gae(l=.8)"] = {"critic": True}
        ests["critic_gae(l=.95)"] = {"critic": True, "lam": 0.95}
        ests["critic_noshape(l=.8)"] = {"critic": True, "no_shaping": True}
    if a.learned_phi:
        critic = dict(np.load(a.learned_phi))
        ests["learned_phi(ls=.5)"] = {"learned_phi": True, "lam_step": 0.5}
        ests["learned_raw(ls=.5)"] = {"learned_phi": True, "raw": True, "lam_step": 0.5}
        ests["learned_fs(ls=0)"] = {"learned_phi": True, "raw": True, "lam_step": 0.0}
        ests["hand_phi(ls=0)"] = {"lam_step": 0.0}
        ests["hand_phi(ls=.5)"] = {"lam_step": 0.5}
    G = {(n, h): [] for n in ests for h in ("combat", "run")}
    # CountSketch: inner products of 18.5M-dim gradients preserved to ~1/sqrt(K) of the norms,
    # so 8 estimators x 2 halves x N iterations fit in memory.
    K = 1 << 20
    gen = torch.Generator(device="cpu").manual_seed(0)
    nparam = sum(p.numel() for p in params)
    hsh = torch.randint(0, K, (nparam,), generator=gen).to(dev)
    sgn = (torch.randint(0, 2, (nparam,), generator=gen) * 2 - 1).to(dev, torch.float32)
    t0 = time.time()
    for pi, path in enumerate(paths):
        with np.load(path) as z:
            it = {k: z[k] for k in z.files}
        keep = it["run"] >= 0
        if not keep.all():
            it = {k: (v[keep] if (len(v) == len(keep) and not k.startswith("run_")) else v)
                  for k, v in it.items()}
        packed = DS.Packed({k[2:]: v for k, v in it.items() if k.startswith("x_")})
        kind = it["kind"]
        is_c = kind == COMBAT
        A = {}
        for n, e in ests.items():
            vo = critic.get(f"v{orig_index[pi]}") if (critic is not None and (e.get("critic") or e.get("learned_phi"))) else None
            raw = it["adv"].astype(np.float64) if e.get("dumped") else advantages(it, e, vo)
            if e.get("grp"):
                raw = apply_group_baseline(it, raw)
            A[n] = torch.as_tensor(normalise(raw, kind), device=dev)
        w = torch.as_tensor(np.where(is_c, 0.5 / max(1, is_c.sum()), 0.5 / max(1, (~is_c).sum())),
                            dtype=torch.float32, device=dev)
        n_tok = packed.a["n_tokens"]
        acc = {k: [torch.zeros_like(p, dtype=torch.float32) for p in params] for k in G}
        for half, hm in (("combat", is_c), ("run", ~is_c)):
            rows = np.flatnonzero(hm)
            rows = rows[np.argsort(n_tok[rows], kind="stable")]
            for s in range(0, len(rows), a.bs):
                idx = rows[s:s + a.bs]
                inp, _ = DS.make_batch([(packed, i) for i in idx])
                inp = to_torch(inp, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = net(inp, loop_rows=torch.as_tensor(it["loop"][idx]))
                logp = F.log_softmax(out["logits"].float(), -1).gather(
                    1, torch.as_tensor(it["a"][idx], device=dev).unsqueeze(1)).squeeze(1)
                ti = torch.as_tensor(idx, device=dev)
                for j, n in enumerate(ests):
                    loss = -(w[ti] * A[n][ti] * logp).sum()
                    gs = torch.autograd.grad(loss, params, retain_graph=j < len(ests) - 1,
                                             allow_unused=True)
                    for buf, g in zip(acc[(n, half)], gs):
                        if g is not None:
                            buf += g.float()
        for k in G:
            flat = torch.cat([b.flatten() for b in acc[k]])
            G[k].append(torch.zeros(K, device=dev).index_add_(0, hsh, flat * sgn))
        print(f"  iter {pi + 1}/{len(paths)}: {len(kind)} steps ({is_c.sum()} combat), "
              f"{time.time() - t0:.0f}s", flush=True)

    n_runs = 96
    rep = {}
    N = len(paths)
    off = ~np.eye(N, dtype=bool)

    def consistency(Gm, drop=None):
        keep = np.ones(N, bool)
        if drop is not None:
            keep[drop] = False
        o = off[np.ix_(keep, keep)]
        g = Gm[np.ix_(keep, keep)]
        return g[o].mean() / np.diag(g).mean(), g[o].mean()

    print(f"\n{N} iterations; consistency = mean pairwise cosine across iterations (+- jackknife SE)")
    print(f"{'estimator':22} {'half':7} {'consistency':>18} {'B_noise(runs)':>14} {'agree_mc':>9}")
    for h in ("combat", "run"):
        gram, sig = {}, {}
        for n in ests:
            M = torch.stack(G[(n, h)])
            gram[n] = (M @ M.T).double().cpu().numpy()
            sig[n] = consistency(gram[n])[1]
        mc = torch.stack(G[("mc_win", h)]) if "mc_win" in ests else None
        for n in ests:
            c, s2 = consistency(gram[n])
            jk = np.array([consistency(gram[n], d)[0] for d in range(N)])
            se = np.sqrt((N - 1) / N * ((jk - jk.mean()) ** 2).sum())
            bn = n_runs * (1 - c) / c if c > 0 else float("inf")
            agree = float("nan")
            if mc is not None and sig[n] > 0 and sig["mc_win"] > 0:
                X = (torch.stack(G[(n, h)]) @ mc.T).double().cpu().numpy()
                agree = float(X[off].mean() / np.sqrt(sig[n] * sig["mc_win"]))
            rep[f"{n}|{h}"] = {"consistency": c, "se": se, "B_noise": bn, "agree_mc": agree}
            print(f"{n:22} {h:7} {c:10.4f} +-{se:.4f} {bn:14.0f} {agree:9.3f}")
        # cosine between EXPECTED directions, every pair of estimators (cross terms i != j
        # only, so each side's noise averages out). 1 = same direction, just different noise.
        names = [n for n in ests if sig[n] > 0]
        print(f"  expected-direction cosines ({h}):")
        for x, y in itertools.combinations(names, 2):
            X = (torch.stack(G[(x, h)]) @ torch.stack(G[(y, h)]).T).double().cpu().numpy()
            cs = X[off].mean() / np.sqrt(sig[x] * sig[y])
            jk = []
            for d in range(N):
                keep = np.ones(N, bool); keep[d] = False
                o = off[np.ix_(keep, keep)]
                sx = gram[x][np.ix_(keep, keep)][o].mean(); sy = gram[y][np.ix_(keep, keep)][o].mean()
                jk.append(X[np.ix_(keep, keep)][o].mean() / np.sqrt(max(sx, 1e-30) * max(sy, 1e-30)))
            jk = np.array(jk); se = np.sqrt((N - 1) / N * ((jk - jk.mean()) ** 2).sum())
            rep[f"cos[{x},{y}]|{h}"] = {"cos": float(cs), "se": float(se)}
            print(f"    {x:>20} vs {y:<20} {cs:+.3f} +-{se:.3f}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
