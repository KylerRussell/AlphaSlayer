"""Behaviour cloning warm start.

Trains the combat net to imitate the demonstrator policy, plus value and auxiliary heads.
BC caps out at the demonstrator's strength by construction -- it is a search prior and a warm
value net, not the product. What it buys is a policy that is not random when MCTS starts.

    python train_bc.py --data /tmp/alphaslayer_probe/greedy.bin.gz --epochs 8
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from alphaslayer.encoder import Encoder
from alphaslayer.format import load
from alphaslayer.model import CombatNet, param_count


def build_dataset(path: str):
    """Steps, with each step labelled by the outcome of the episode it belongs to."""
    _, steps, terms = load(path)
    by_ep = {t.ep: t for t in terms}
    keep, won, endhp, hpd = [], [], [], []
    for s in steps:
        t = by_ep.get(s.ep)
        if t is None:
            continue  # truncated tail: an episode whose terminal record never got written
        keep.append(s)
        won.append(1.0 if t.won else 0.0)
        endhp.append(t.hp_end / max(1, s.g("max_hp")))
        hpd.append(s.hp_delta / 20.0)
    return keep, np.array(won, np.float32), np.array(endhp, np.float32), np.array(hpd, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/alphaslayer_probe/greedy.bin.gz")
    ap.add_argument("--vocab", default="/tmp/alphaslayer_probe/vocab.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="combat_bc.pt")
    args = ap.parse_args()

    sizes = json.load(open(args.vocab))["sizes"]
    steps, won, endhp, hpd = build_dataset(args.data)
    print(f"dataset: {len(steps)} steps, win rate {won.mean():.3f}")

    enc = Encoder(sizes["cards"])
    batch = enc.encode(steps)
    if any(enc.truncated.values()):
        print(f"  truncation: {enc.truncated}")

    n = len(steps)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    dev = torch.device(args.device)
    tensors = {k: torch.from_numpy(v) for k, v in batch.items()}
    tensors["won"] = torch.from_numpy(won)
    tensors["endhp"] = torch.from_numpy(endhp)
    tensors["hpd"] = torch.from_numpy(hpd)

    net = CombatNet(sizes, d_model=args.d_model, layers=args.layers).to(dev)
    print(f"params {param_count(net) / 1e6:.2f}M on {dev}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.2)

    def run_batch(idx, train: bool):
        b = {k: tensors[k][idx].to(dev) for k in
             ("hand", "hand_mask", "enemies", "enemy_mask", "relics", "relic_mask",
              "powers", "power_mask", "bags", "globals", "actions", "action_mask", "target")}
        out = net(b)
        ce = F.cross_entropy(out["logits"], b["target"])
        v = F.binary_cross_entropy_with_logits(out["value"], tensors["won"][idx].to(dev))
        a1 = F.mse_loss(out["aux_hp"], tensors["hpd"][idx].to(dev))
        a2 = F.mse_loss(out["aux_endhp"], tensors["endhp"][idx].to(dev))
        loss = ce + 0.5 * v + 0.25 * a1 + 0.25 * a2
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        acc = (out["logits"].argmax(-1) == b["target"]).float().mean()
        return ce.item(), v.item(), acc.item()

    mean_legal = batch["action_mask"].sum(1).mean()
    print(f"uniform-policy CE baseline: {math.log(mean_legal):.3f} "
          f"(mean {mean_legal:.1f} legal actions)")

    for ep in range(args.epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(train_idx))
        t0, agg = time.time(), np.zeros(3)
        for i in range(steps_per_epoch):
            agg += run_batch(order[i * args.batch:(i + 1) * args.batch], True)
        agg /= steps_per_epoch

        net.eval()
        with torch.no_grad():
            vagg = np.mean([run_batch(torch.from_numpy(val_idx[i:i + args.batch]), False)
                            for i in range(0, len(val_idx), args.batch)], axis=0)
        print(f"epoch {ep + 1}/{args.epochs}  "
              f"train ce {agg[0]:.3f} acc {agg[2]:.3f} | "
              f"val ce {vagg[0]:.3f} acc {vagg[2]:.3f} value_bce {vagg[1]:.3f}  "
              f"({time.time() - t0:.1f}s)")

    torch.save({"model": net.state_dict(), "sizes": sizes,
                "d_model": args.d_model, "layers": args.layers}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
