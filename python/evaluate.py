"""Plays the live game with a given policy and reports win rate.

This closes the loop: the network chooses actions in the real engine, not just on logged data.

    python evaluate.py --policy bc --episodes 200
    python evaluate.py --policy random --episodes 200
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from alphaslayer.encoder import Encoder, step_from_json
from alphaslayer.env import SpireEnv
from alphaslayer.model import CombatNet, use_stable_attention
from alphaslayer.vecenv import VecSpireEnv


def load_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(device)
    from train_rl import load_compat
    load_compat(net, ck["model"])
    net.eval()
    return net, ck["sizes"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["random", "bc", "greedy"], default="bc")
    ap.add_argument("--ckpt", default="combat_bc.pt")
    ap.add_argument("--vocab", default="/tmp/alphaslayer_probe/vocab.json")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--character", default="IRONCLAD")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.0, help="0 = argmax, >0 = sample")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--characters", default=None, help="comma-separated roster")
    ap.add_argument("--mix", default=None, help="boss,elite,regular weights")
    ap.add_argument("--reroll-deck", action="store_true")
    ap.add_argument("--relics", default=None,
                    help="comma-separated relics to grant, e.g. PANDORAS_BOX")
    ap.add_argument("--encounters", default=None,
                    help="regular|elite|boss|all")
    ap.add_argument("--envs", type=int, default=1,
                    help=">1 runs N game processes against one batched policy")
    args = ap.parse_args()
    use_stable_attention()

    rng = random.Random(args.seed)
    sizes = json.load(open(args.vocab))["sizes"]
    enc = Encoder(sizes["cards"])

    net = None
    if args.policy == "bc":
        net, _ = load_net(args.ckpt, torch.device(args.device))

    def batched(obs_list, legal_list, env_idxs=None):
        """One forward pass for every pending decision across all envs."""
        if args.policy == "random":
            return [rng.randrange(len(l)) for l in legal_list]
        if args.policy == "greedy":
            out = []
            for l in legal_list:
                plays = [i for i, x in enumerate(l) if x["kind"] != "end_turn"]
                out.append(plays[0] if plays else 0)
            return out
        b = enc.encode([step_from_json(o, l) for o, l in zip(obs_list, legal_list)])
        tb = {k: torch.from_numpy(v).to(args.device) for k, v in b.items()}
        with torch.no_grad():
            logits = net(tb)["logits"]
        if args.temp <= 0:
            picks = logits.argmax(-1).tolist()
        else:
            picks = torch.multinomial(torch.softmax(logits / args.temp, -1), 1).squeeze(-1).tolist()
        # argmax runs over the padded action axis; clamp into each env's real legal range.
        return [min(p, len(l) - 1) for p, l in zip(picks, legal_list)]

    if args.envs > 1:
        per = max(1, args.episodes // args.envs)
        with VecSpireEnv(n_envs=args.envs, character=args.character,
                         episodes_per_env=per, ascension=args.ascension,
                         encounters=args.encounters, mix=args.mix,
                         reroll_deck=args.reroll_deck,
                         characters=args.characters.split(',') if args.characters else None,
                         inject_relics=args.relics.split(',') if args.relics else None) as venv:
            venv.run(batched)
            n = len(venv.results)
            wins = sum(r.won for r in venv.results)
            hp = np.mean([r.hp_end for r in venv.results]) if n else 0
            turns = np.mean([r.turns for r in venv.results]) if n else 0
            print(f"policy={args.policy:6s} envs={args.envs} enc={args.encounters or 'regular'}{'+' + args.relics if args.relics else ''} A{args.ascension} episodes={n:4d} "
                  f"win_rate={wins / max(1, n):.3f} mean_end_hp={hp:5.1f} mean_turns={turns:4.1f} "
                  f"steps={venv.steps} ({venv.steps / venv.wall:.0f} steps/s) "
                  f"mean_batch={venv.mean_batch:.2f}")
        return

    t0 = time.time()
    steps = 0
    with SpireEnv(character=args.character, episodes=args.episodes) as env:
        for obs, legal in env:
            if args.policy == "random":
                a = rng.randrange(len(legal))
            elif args.policy == "greedy":
                # Same rule as the C# demonstrator: play anything playable, else end turn.
                plays = [i for i, x in enumerate(legal) if x["kind"] != "end_turn"]
                a = plays[0] if plays else 0
            else:
                b = enc.encode([step_from_json(obs, legal)])
                tb = {k: torch.from_numpy(v).to(args.device) for k, v in b.items()}
                with torch.no_grad():
                    logits = net(tb)["logits"][0, : len(legal)]
                if args.temp <= 0:
                    a = int(logits.argmax())
                else:
                    p = torch.softmax(logits / args.temp, -1)
                    a = int(torch.multinomial(p, 1))
            env.act(a)
            steps += 1

        wins = sum(r.won for r in env.results)
        n = len(env.results)
        hp = np.mean([r.hp_end for r in env.results]) if n else 0
        turns = np.mean([r.turns for r in env.results]) if n else 0

    dt = time.time() - t0
    print(f"policy={args.policy:6s} episodes={n:4d} win_rate={wins / max(1, n):.3f} "
          f"mean_end_hp={hp:5.1f} mean_turns={turns:4.1f} "
          f"steps={steps} ({steps / dt:.0f} steps/s)")


if __name__ == "__main__":
    main()
