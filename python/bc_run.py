"""Behaviour-cloning initialisation for the run policy, from the scripted heuristic.

Every reward-shaping variant tried so far failed to discover three behaviours that matter:
healing when hurt, skipping a card, and buying a removal. Once a policy stops doing a thing it
stops generating data about it, so the gradient that would teach it never exists. Cloning the
heuristic puts all three in the policy from the start; RL then improves on a starting point
that already plays sensibly rather than searching for it.

The heuristic is a floor, not a target -- the trained policy already beat it on floors and
elites, and lost to it on bosses. The point is to begin from a distribution where the missing
actions occur.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from alphaslayer.heuristic import HeuristicRunPolicy
from alphaslayer.model import CombatNet, use_stable_attention
from alphaslayer.runenv import VecRunEnv
from alphaslayer.runmodel import RunNet
from train_rl import load_compat
import train_run as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat-ckpt", default="combat_run_cx.pt")
    ap.add_argument("--out", default="run_policy_bc.pt")
    ap.add_argument("--vocab", default="/tmp/alphaslayer_probe/vocab.json")
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--runs", type=int, default=240)
    ap.add_argument("--act-cap", type=int, default=1)
    ap.add_argument("--characters", default="IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--per-kind-heads", action="store_true", default=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    use_stable_attention()
    dev = torch.device(args.device)
    T._load_vocab(args.vocab)
    ck = torch.load(args.combat_ckpt, map_location=dev)
    combat = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev).eval()
    load_compat(combat, ck["model"])
    T._ENC = T.Encoder(ck["sizes"]["cards"])

    heur = HeuristicRunPolicy(seed=0)
    samples = []          # (kind, obs, legal, action)

    def policy(kinds, obs, legal, idxs):
        acts = [0] * len(kinds)
        combat_ix = [i for i, k in enumerate(kinds) if k == "combat"]
        if combat_ix:
            with torch.no_grad():
                lg, _, mask = T.combat_forward(combat, [obs[i] for i in combat_ix],
                                               [legal[i] for i in combat_ix], dev)
                lg = lg.masked_fill(~mask, torch.finfo(lg.dtype).min / 4)
                pick = lg.argmax(-1).tolist()
            for j, i in enumerate(combat_ix):
                acts[i] = pick[j]
        for i, k in enumerate(kinds):
            if k == "combat":
                continue
            a = heur(k, obs[i], legal[i])
            acts[i] = a
            samples.append((k, obs[i], legal[i], a))
        return acts

    t0 = time.time()
    with VecRunEnv(n_envs=args.envs, characters=args.characters.split(","),
                   runs_per_env=max(1, args.runs // args.envs), seed="BC",
                   ascension=args.ascension, act_cap=args.act_cap,
                   out_dir="/tmp/alphaslayer_bc") as venv:
        venv.run(policy, deadline_s=1800)
        res = list(venv.results)
    wins = sum(r.won for r in res) / max(1, len(res))
    by_kind = defaultdict(int)
    for k, *_ in samples:
        by_kind[k] += 1
    print(f"collected {len(samples)} decisions from {len(res)} runs "
          f"(heuristic win={wins:.3f}) in {time.time()-t0:.0f}s")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items(), key=lambda x: -x[1])))

    n_cards, n_relics = ck["sizes"]["cards"], ck["sizes"]["relics"]
    net = RunNet(n_cards, n_relics, ck["sizes"]["potions"],
                 per_kind_heads=args.per_kind_heads).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        perm = torch.randperm(len(samples)).tolist()
        tot, nb, correct, seen = 0.0, 0, 0, 0
        agree = defaultdict(lambda: [0, 0])
        for start in range(0, len(perm), args.minibatch):
            batch = [samples[i] for i in perm[start:start + args.minibatch]]
            kinds = [b[0] for b in batch]
            logits, _, mask = T.run_forward(net, kinds, [b[1] for b in batch],
                                            [b[2] for b in batch], dev)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
            tgt = torch.tensor([min(b[3], logits.shape[1] - 1) for b in batch], device=dev)
            # Weight each decision kind equally. A plain mean is dominated by travel (38% of
            # decisions, ~1.5 options, nearly free to get right), so the rare-but-decisive
            # kinds -- rest, shop, card rewards -- contribute almost nothing to the gradient.
            # That is how a clone reached 85% aggregate agreement while reproducing the
            # heuristic's campfire choice essentially never.
            per = F.cross_entropy(logits, tgt, reduction="none")
            terms = []
            for kk in set(kinds):
                sel = torch.tensor([j for j, k2 in enumerate(kinds) if k2 == kk], device=dev)
                terms.append(per[sel].mean())
            loss = torch.stack(terms).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
            hit = (logits.argmax(-1) == tgt)
            correct += int(hit.sum()); seen += len(batch)
            for j, k2 in enumerate(kinds):
                agree[k2][0] += int(hit[j]); agree[k2][1] += 1
        # Per-kind agreement, because the aggregate is not the thing we care about.
        detail = " ".join(f"{k}={a/max(1,b):.2f}" for k, (a, b) in
                          sorted(agree.items(), key=lambda kv: -kv[1][1]))
        print(f"  epoch {ep+1}/{args.epochs}: loss={tot/max(1,nb):.4f} "
              f"agreement={correct/max(1,seen):.3f} | {detail}", flush=True)

    torch.save({"model": net.state_dict(),
                "sizes": {"cards": n_cards, "relics": n_relics,
                          "potions": ck["sizes"]["potions"]}}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
