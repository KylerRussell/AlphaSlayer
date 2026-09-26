"""Harvests real deck states from runs and measures them, to train the deck critic.

Two stages:
  1. Play runs with the current policies and snapshot the deck at every card-reward decision,
     together with the cards being offered.
  2. For each snapshot, evaluate the COUNTERFACTUALS -- the deck as it stands, and the deck
     with each offered card added -- against every room type.

Stage 2 is the ground truth the critic is distilled from, and it is also exactly the signal
the run policy is missing: the difference between those evaluations is what says whether a
card is worth taking. Sampling deck states from real runs rather than generating them
synthetically matters because the critic is only ever asked about decks the policy actually
reaches.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

import torch

from alphaslayer.deckeval import DeckSpec, VecDeckEval
from alphaslayer.model import CombatNet, use_stable_attention
from alphaslayer.runenv import VecRunEnv
from alphaslayer.runmodel import RunNet
from train_rl import load_compat
import train_run as T


def harvest(args, combat, runnet, dev):
    """Stage 1: snapshot (deck, relics, offered cards) at card-reward decisions."""
    snaps = []

    def policy(kinds, obs, legal, idxs):
        acts = [0] * len(kinds)
        groups = defaultdict(list)
        for i, k in enumerate(kinds):
            groups["combat" if k == "combat" else "run"].append(i)
        with torch.no_grad():
            for grp, mem in groups.items():
                o = [obs[i] for i in mem]
                l = [legal[i] for i in mem]
                if grp == "combat":
                    lg, _, mask = T.combat_forward(combat, o, l, dev)
                else:
                    lg, _, mask = T.run_forward(runnet, [kinds[i] for i in mem], o, l, dev)
                lg = lg.masked_fill(~mask, torch.finfo(lg.dtype).min / 4)
                probs = torch.softmax(lg, -1)
                idx = torch.multinomial(probs, 1).squeeze(-1)
                for j, i in enumerate(mem):
                    acts[i] = int(idx[j])
        for i, k in enumerate(kinds):
            if k != "card_reward":
                continue
            p = obs[i].get("player", obs[i])
            cards = p.get("deck_cards") or []
            if not cards:
                continue
            offered = [a for a in legal[i] if a.get("kind") == "card"]
            snaps.append({
                "character": p.get("character", ""),
                "deck": [[c["card"], int(c.get("up", 0))] for c in cards],
                "relics": [r.get("relic", "") for r in (p.get("relics") or [])],
                "offered": [a.get("id", "") for a in offered],
            })
        return acts

    with VecRunEnv(n_envs=args.envs, characters=args.characters.split(","),
                   runs_per_env=max(1, args.runs // args.envs), seed=args.seed,
                   ascension=args.ascension, run_offset=args.run_offset,
                   out_dir="/tmp/alphaslayer_harvest") as venv:
        venv.run(policy, deadline_s=args.harvest_timeout)
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat-ckpt", default="combat_capsule.pt")
    ap.add_argument("--run-ckpt", default="run_policy_prefix.pt")
    ap.add_argument("--out", default="deck_data.jsonl")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--eval-envs", type=int, default=8)
    ap.add_argument("--fights", type=int, default=3, help="fights per room type per deck")
    ap.add_argument("--max-states", type=int, default=250,
                    help="card-reward snapshots to evaluate (each costs ~4x3xfights fights)")
    ap.add_argument("--characters", default="IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--seed", default="HARVEST")
    ap.add_argument("--run-offset", type=int, default=0)
    ap.add_argument("--harvest-timeout", type=float, default=600.0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    use_stable_attention()
    dev = torch.device(args.device)
    ck = torch.load(args.combat_ckpt, map_location=dev)
    combat = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev).eval()
    load_compat(combat, ck["model"])
    T._ENC = T.Encoder(ck["sizes"]["cards"])
    rck = torch.load(args.run_ckpt, map_location=dev)
    runnet = RunNet(ck["sizes"]["cards"], ck["sizes"]["relics"], ck["sizes"]["potions"],
                    per_kind_heads=any(k.startswith("score_k.")
                                       for k in (rck.get("model") or rck))).to(dev).eval()
    load_compat(runnet, rck["model"] if "model" in rck else rck)

    t0 = time.time()
    snaps = harvest(args, combat, runnet, dev)
    print(f"harvested {len(snaps)} card-reward states in {time.time()-t0:.0f}s", flush=True)
    if not snaps:
        raise SystemExit("no card-reward states harvested")

    rng = random.Random(0)
    rng.shuffle(snaps)
    snaps = snaps[:args.max_states]

    # Group by character: one evaluator process plays one character, so specs cannot be mixed.
    by_char = defaultdict(list)
    for sn in snaps:
        by_char[sn["character"]].append(sn)

    def policy(obs, legal):
        with torch.no_grad():
            lg, _, mask = T.combat_forward(combat, obs, legal, dev)
            lg = lg.masked_fill(~mask, torch.finfo(lg.dtype).min / 4)
            return lg.argmax(-1).tolist()

    written = 0
    with open(args.out, "w") as fh:
        for char, group in by_char.items():
            specs, meta = [], []
            for sn in group:
                base = sn["deck"]
                # The deck as it stands, plus one variant per offered card. The DIFFERENCE
                # between them is the quantity the run policy needs and cannot currently see.
                variants = [("skip", base)]
                for cid in sn["offered"]:
                    variants.append((cid, base + [[cid, 0]]))
                for tag, deck in variants:
                    specs.append(DeckSpec([c[0] for c in deck], [c[1] for c in deck],
                                          sn["relics"], char, args.fights, tag=tag))
                    meta.append((sn, tag))
            print(f"  {char}: {len(group)} states -> {len(specs)} decks "
                  f"({len(specs)*3*args.fights} fights)", flush=True)
            with VecDeckEval(n_envs=args.eval_envs, character=char, seed=f"DE{char}",
                             ascension=args.ascension) as ev:
                ev.evaluate(specs, policy, progress_every=500)
                print(f"    {ev.fights} fights in {ev.wall:.0f}s", flush=True)
            for spec, (sn, tag) in zip(specs, meta):
                rec = {
                    "character": char, "relics": sn["relics"],
                    "deck": spec.cards, "upgrades": spec.upgrades, "variant": tag,
                    "hp": {r: (spec.hp_left[r] / spec.total[r] if spec.total[r] else None)
                           for r in spec.rooms},
                    "win": {r: spec.win_rate(r) for r in spec.rooms},
                    "n": {r: spec.total[r] for r in spec.rooms},
                }
                fh.write(json.dumps(rec) + "\n")
                written += 1
    print(f"wrote {written} evaluated decks -> {args.out} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
