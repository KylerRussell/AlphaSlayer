"""Measures a combat checkpoint on the realistic, boss-heavy run-deck benchmark.

    python measure_bench.py --combat-ckpt combat_act1.pt --bench combat_bench.json

The old gate scored the policy on random Pandora's-Box decks. This one replays decks the run
policy actually arrived at, against the room type they were actually about to fight, and
aggregates on the measured death distribution -- so the number moves when the policy gets
better at the fights that end runs.
"""
from __future__ import annotations

import argparse
import time

import torch

import train_run as T
from alphaslayer.modcheck import check as modcheck
from alphaslayer.benchmark import BenchmarkPool, load
from alphaslayer.model import CombatNet, use_stable_attention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat-ckpt", required=True)
    ap.add_argument("--bench", default="combat_bench.json")
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--fights-per-deck", type=int, default=2)
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    use_stable_attention()
    modcheck()

    dev = torch.device(a.device)
    ck = torch.load(a.combat_ckpt, map_location=dev, weights_only=False)
    net = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev).eval()
    T.load_compat(net, ck["model"])
    T._ENC = T.Encoder(ck["sizes"]["cards"])

    def policy(obs, legal):
        # Greedy: this is a measurement, and sampling would add variance unrelated to the
        # difference being measured.
        with torch.no_grad():
            lg, _, mask = T.combat_forward(net, obs, legal, dev)
            lg = lg.masked_fill(~mask, torch.finfo(lg.dtype).min / 4)
            return lg.argmax(-1).tolist()

    bench = load(a.bench)
    pool = BenchmarkPool(bench, n_envs=a.envs, ascension=a.ascension)
    t0 = time.time()
    try:
        # close_after_each: hold at most --envs games at a time instead of one pool per
        # character (5 x envs concurrently), which exhausted the machine and surfaced as
        # "no deckeval processes connected".
        wr, se, per_room, n = pool.measure(policy, fights_per_deck=a.fights_per_deck,
                                           close_after_each=True)
    finally:
        pool.close()

    print(f"\n  {a.combat_ckpt} on {a.bench}")
    for room in ("boss", "elite", "regular"):
        if room in per_room:
            p, rn = per_room[room]
            print(f"    {room:8s} {p:.3f}  (n={rn})")
    expected = len(bench["decks"]) * a.fights_per_deck
    flag = "" if n >= 0.95 * expected else (
        f"   *** INCOMPLETE: {n}/{expected} fights -- NOT comparable with a full run ***")
    print(f"    WEIGHTED {wr:.4f} +-{se:.4f}   ({n} fights, {time.time()-t0:.0f}s){flag}")

    # Per character. Every deck here fights at FULL HP, so this separates "arrives at the
    # boss too hurt" from "this deck and this policy simply lose the boss fight" -- the
    # observed spread in live runs was 0.115 (NECROBINDER) to 0.519 (DEFECT).
    bc = getattr(pool, "by_character", {})
    if bc:
        chars = sorted({c for c, _ in bc})
        print(f"\n    {'character':14s}" + "".join(f"{r:>10s}" for r in ("boss", "elite", "regular")))
        for c in chars:
            row = f"    {c:14s}"
            for r in ("boss", "elite", "regular"):
                if (c, r) in bc:
                    p_, n_ = bc[(c, r)]
                    row += f"{p_:>10.3f}"
                else:
                    row += f"{'-':>10s}"
            print(row)


if __name__ == "__main__":
    main()
