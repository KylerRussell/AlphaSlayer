"""Offline critic study on dumped rollouts (train_unified.py --dump-steps).

Question: is P(win | state) learnable from runs at all, and if so, why does the trainer's value
head have NEGATIVE out-of-sample skill? Answers, on frozen trunk features of a checkpoint:

  1. calibration of the checkpoint's own value head (mean prediction vs base rate, per act)
  2. fresh heads fit on iterations [0, K) and scored on held-out iterations, for several K and
     several targets (win, reach act 3, clear current act, final floor), against a baseline
     that only sees (act, floor, hp%)

    PYTHONPATH=. .venv7/bin/python offline_critic.py --dumps /tmp/alphaslayer_dump --ckpt unified_m4_long2.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphaslayer.model import use_stable_attention
from alphaslayer.unified import dataset as DS
from alphaslayer.unified import features as FT
from alphaslayer.unified.net import UnifiedNet, to_torch

COMBAT = FT.DKINDS.index("combat")


def load_dumps(d):
    its = []
    for p in sorted(glob.glob(os.path.join(d, "it*.npz"))):
        with np.load(p) as z:
            its.append({k: z[k] for k in z.files})
    return its


@torch.no_grad()
def features(net, it, dev, rows, bs=512):
    """CLS features + the checkpoint's value/aux predictions for the given rows."""
    packed = DS.Packed({k[2:]: v for k, v in it.items() if k.startswith("x_")})
    grab = {}
    h = net.value_head.register_forward_hook(lambda m, i, o: grab.__setitem__("cls", i[0]))
    feats, v = [], []
    order = sorted(rows, key=lambda i: packed.a["n_tokens"][i])
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        inp, _ = DS.make_batch([(packed, i) for i in idx])
        inp = to_torch(inp, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(inp, loop_rows=torch.as_tensor(it["loop"][idx]))
        feats.append(grab["cls"].float().cpu())
        v.append(out["value"].float().cpu())
    h.remove()
    inv = np.argsort(order)
    return torch.cat(feats)[inv], torch.cat(v)[inv].numpy()


def skill_bce(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    brier = float(((p - y) ** 2).mean())
    base = float(((y.mean() - y) ** 2).mean())
    return 1 - brier / base if base > 0 else float("nan")


def fit_head(Xtr, ytr, Xte, kind="bce", hidden=256, epochs=30, wd=1e-3, lr=1e-3, dev="cuda"):
    """Small MLP on frozen features, early-stopped on a run-level split of the training data."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xtr, ytr = Xtr.to(dev), torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, 4096):
            i = perm[s:s + 4096]
            o = m(Xtr[i]).squeeze(-1)
            loss = (F.binary_cross_entropy_with_logits(o, ytr[i]) if kind == "bce"
                    else F.mse_loss(o, ytr[i]))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        o = m(Xte.to(dev)).squeeze(-1)
        return (torch.sigmoid(o) if kind == "bce" else o).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="/tmp/alphaslayer_dump")
    ap.add_argument("--ckpt", default="unified_m4_long2.pt")
    ap.add_argument("--combat-keep", type=float, default=0.15,
                    help="fraction of combat rows kept (they are ~80%% of steps)")
    ap.add_argument("--test-iters", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-v", default=None, help="write cross-fitted V(win) per iteration (.npz)")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    use_stable_attention()
    dev = torch.device("cuda:0")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net = UnifiedNet(ck["sizes"], **ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])

    its = load_dumps(a.dumps)
    print(f"{len(its)} iterations, {sum(len(i['kind']) for i in its)} steps, "
          f"{sum(len(i['run_won']) for i in its)} runs, {sum(int(i['run_won'].sum()) for i in its)} wins")
    rng = np.random.default_rng(0)
    X, V, Y, META = [], [], [], []
    for k, it in enumerate(its):
        keep = (it["run"] >= 0) & ((it["kind"] != COMBAT) | (rng.random(len(it["kind"])) < a.combat_keep))
        rows = np.flatnonzero(keep)
        f, v = features(net, it, dev, rows)
        X.append(f); V.append(v)
        rf = it["run_floors"][it["run"][rows]]
        Y.append({"won": it["won"][rows], "reach_act3": it["reach_act3"][rows],
                  "act_clear": it["act_clear"][rows], "final_floor": rf / 50.0})
        META.append({"iter": np.full(len(rows), k), "act": it["act"][rows], "floor": it["floor"][rows],
                     "kind": it["kind"][rows], "run": it["run"][rows] + 10_000 * k, "phi": it["phi"][rows]})
        print(f"  iter {k}: {len(rows)} rows featurised", flush=True)
    cat = lambda L, key=None: np.concatenate([x[key] for x in L]) if key else np.concatenate(L)
    Xall = torch.cat(X)
    Vall = cat(V)
    it_of = cat(META, "iter"); act = cat(META, "act"); floor = cat(META, "floor")
    report = {}

    # 1. calibration of the checkpoint's own value head
    won = cat(Y, "won")
    print("\n1. checkpoint value head vs outcome")
    for ac in range(3):
        m = act == ac
        print(f"  act {ac + 1}: mean V={Vall[m].mean():.3f}  win rate={won[m].mean():.3f}  "
              f"skill={skill_bce(Vall[m], won[m]):+.3f}  n={m.sum()}")
    print(f"  all : mean V={Vall.mean():.3f}  win rate={won.mean():.3f}  skill={skill_bce(Vall, won):+.3f}")
    report["ckpt_value_skill"] = skill_bce(Vall, won)

    # 2. fresh heads on frozen features, more and more training iterations
    nT = len(its)
    test = it_of >= nT - a.test_iters
    hand = torch.as_tensor(np.stack([act == 0, act == 1, act == 2, floor / 50.0], 1).astype(np.float32))
    print(f"\n2. fresh heads, test = last {a.test_iters} iterations (out-of-sample skill)")
    for tgt, kind in (("won", "bce"), ("reach_act3", "bce"), ("act_clear", "bce"), ("final_floor", "mse")):
        y = cat(Y, tgt).astype(np.float32)
        line = f"  {tgt:12}"
        for K in (1, 5, 10, nT - a.test_iters):
            if K > nT - a.test_iters:
                continue
            tr = it_of < K
            p = fit_head(Xall[tr], y[tr], Xall[test], kind)
            s = skill_bce(p, y[test]) if kind == "bce" else 1 - ((p - y[test]) ** 2).mean() / y[test].var()
            line += f"  K={K}:{s:+.3f}"
            report[f"{tgt}_K{K}"] = float(s)
        tr = it_of < nT - a.test_iters
        p = fit_head(hand[tr], y[tr], hand[test], kind, hidden=32)
        s = skill_bce(p, y[test]) if kind == "bce" else 1 - ((p - y[test]) ** 2).mean() / y[test].var()
        line += f"   | act+floor only: {s:+.3f}"
        report[f"{tgt}_hand"] = float(s)
        print(line, flush=True)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)

    # 3. cross-fitted critic for EVERY step (grad_snr.py --critic): each iteration's values come
    #    from a head trained on the other folds' iterations, so no run's outcome leaks into its
    #    own baseline.
    if a.save_v:
        print(f"\n3. cross-fitted V(win) for all steps, {a.folds} folds", flush=True)
        allX, allit = [], []
        for k, it in enumerate(its):
            rows = np.flatnonzero(it["run"] >= 0)
            f, _ = features(net, it, dev, rows)
            allX.append(f.half()); allit.append(np.full(len(rows), k))
        allX = torch.cat(allX); allit = np.concatenate(allit)
        yw = np.concatenate([it["won"][it["run"] >= 0] for it in its]).astype(np.float32)
        pred = np.zeros(len(yw), np.float32)
        fold = allit % a.folds
        for fo in range(a.folds):
            tr, te = fold != fo, fold == fo
            # train on the subsampled rows of the other folds (combat thinned as above)
            sub = tr & ((np.concatenate([it["kind"][it["run"] >= 0] for it in its]) != COMBAT)
                        | (rng.random(len(tr)) < a.combat_keep))
            pred[te] = fit_head(allX[sub].float(), yw[sub], allX[te].float(), "bce")
            print(f"  fold {fo}: skill on held-out fold {skill_bce(pred[te], yw[te]):+.3f}", flush=True)
        out = {}
        for k in range(len(its)):
            out[f"v{k}"] = pred[allit == k]
        np.savez(a.save_v, **out)
        print(f"  overall cross-fitted skill {skill_bce(pred, yw):+.3f} -> {a.save_v}")


if __name__ == "__main__":
    main()
