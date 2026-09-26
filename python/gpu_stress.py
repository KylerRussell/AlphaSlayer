"""Minimal GPU stress repro for the ROCm HSA faults.

Replays the same model at the same tensor shapes as PPO training, with random data and no
game processes. If this faults, the trigger is the model/ops and can be bisected in minutes
instead of 20-minute training runs; if it never faults, the game subprocesses or the
collect/train interleaving matter.

    python gpu_stress.py --minutes 5 [--no-backward] [--layers 6]
"""
from __future__ import annotations

import argparse, json, time
import numpy as np, torch, torch.nn.functional as F

from alphaslayer.encoder import ACTION_WIDTH, ENEMY_WIDTH
from alphaslayer.format import CARD_TOKEN_WIDTH, GLOBALS
from alphaslayer.model import CombatNet, use_stable_attention


def batch(b, sizes, dev, rng, max_hand_idx=16):
    H, E, R, P, A = 16, 8, 32, 16, 96
    i = lambda hi, *sh: torch.from_numpy(rng.integers(0, hi, sh)).to(dev)
    m = lambda *sh: torch.from_numpy(rng.random(sh) > 0.4).to(dev)
    return dict(
        hand=i(sizes["cards"], b, H, CARD_TOKEN_WIDTH), hand_mask=m(b, H),
        enemies=i(100, b, E, ENEMY_WIDTH), enemy_mask=m(b, E),
        relics=i(sizes["relics"], b, R), relic_mask=m(b, R),
        powers=i(sizes["powers"], b, P, 2), power_mask=m(b, P),
        bags=torch.from_numpy(rng.random((b, 3, sizes["cards"])).astype(np.float32)).to(dev),
        globals=torch.from_numpy(rng.random((b, len(GLOBALS))).astype(np.float32)).to(dev),
        actions=torch.stack([
            i(4, b, A), i(max_hand_idx, b, A), i(8, b, A), i(sizes["cards"], b, A)
        ], -1),
        action_mask=torch.ones(b, A, dtype=torch.bool, device=dev),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="combat_boss_shaped.pt")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--stable-attn", action="store_true")
    ap.add_argument("--max-hand-idx", type=int, default=None,
                    help="upper bound for the action hand-index field")
    ap.add_argument("--part", default="full",
                    choices=["emb", "encoder", "policy", "full", "bare_tf", "bare_emb"],
                    help="which stage of the forward pass to exercise")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.stable_attn:
        use_stable_attention()

    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5)
    rng = np.random.default_rng(0)

    print(f"stress: part={a.part} batch={a.batch} backward={not a.no_backward} "
          f"stable_attn={a.stable_attn} for {a.minutes} min", flush=True)
    t0, n = time.time(), 0
    # Bare stages let us tell "our model" from "torch/ROCm on this card" apart.
    d = ck["d_model"]
    bare_tf = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(d, 8, d * 4, activation="gelu",
                                         batch_first=True, norm_first=True),
        ck["layers"]).to(dev) if a.part == "bare_tf" else None
    bare_emb = torch.nn.Embedding(ck["sizes"]["cards"], d).to(dev) if a.part == "bare_emb" else None

    while time.time() - t0 < a.minutes * 60:
        b = batch(a.batch, ck["sizes"], dev, rng, a.max_hand_idx or 16)
        if a.part == "bare_tf":
            out = {"logits": bare_tf(torch.randn(a.batch, 74, d, device=dev)).mean(1),
                   "value": torch.zeros(a.batch, device=dev)}
        elif a.part == "bare_emb":
            out = {"logits": bare_emb(b["hand"][..., 0]).mean(1),
                   "value": torch.zeros(a.batch, device=dev)}
        elif a.part == "emb":
            h = net._hand_tokens(b["hand"])
            e = net._enemy_tokens(b["enemies"])
            g = net.bag_proj(b["bags"].flatten(1)) + net.global_proj(b["globals"])
            out = {"logits": (h.mean(1) + e.mean(1) + g), "value": torch.zeros(a.batch, device=dev)}
        elif a.part == "encoder":
            h = net._hand_tokens(b["hand"])
            out = {"logits": net.encoder(h).mean(1), "value": torch.zeros(a.batch, device=dev)}
        else:
            out = net(b)
        if not a.no_backward:
            loss = out["logits"].float().pow(2).mean() + out["value"].float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
        n += 1
        if n % 200 == 0:
            print(f"  {n} steps, {time.time() - t0:.0f}s elapsed", flush=True)
    print(f"SURVIVED {n} steps in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
