"""Scores a (combat, run) policy PAIR on whole runs.

Exists to answer one question the capsule benchmark cannot: is the combat policy getting
better at the fights it ACTUALLY plays? The gate measures it on Pandora's decks, which is the
distribution it was trained on, not the distribution it now faces.
"""
from __future__ import annotations
import argparse, json, random, time
_rng = random.Random(0)
_heur = None
from collections import defaultdict
import torch, torch.nn.functional as F
from alphaslayer.model import CombatNet, use_stable_attention
from alphaslayer.runenv import VecRunEnv
from alphaslayer.modcheck import check as modcheck
from alphaslayer.runmodel import migrate_state_scalars, RunNet
from train_rl import load_compat
import train_run as T



def _expect_sizes(ck):
    """Vocab sizes the loaded checkpoint can represent, for the live-vocab mismatch guard."""
    z = ck["sizes"]
    out = {"cards": z["cards"], "relics": z["relics"]}
    for k in ("potions", "powers", "monsters"):
        if k in z:
            out[k] = z[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat-ckpt", required=True)
    ap.add_argument("--run-ckpt", required=True)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--runs", type=int, default=32)
    ap.add_argument("--characters", default="IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--act-cap", type=int, default=0)
    ap.add_argument("--seed", default="EVALFIXED_")
    ap.add_argument("--run-policy", default="net", choices=["net", "random", "heuristic"],
                    help="'random' answers every RUN decision uniformly while keeping the "
                         "trained combat policy - the control that says whether the run "
                         "policy has learned anything at all")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    global _heur
    from alphaslayer.heuristic import HeuristicRunPolicy
    _heur = HeuristicRunPolicy(seed=0)
    use_stable_attention()
    modcheck()
    import os
    if os.environ.get('AUDIT_FEATURES'):
        import alphaslayer.runmodel as _rm
        _rm._AUDIT.update(n=0, held=0, upg=0, room=0)
    dev = torch.device(a.device)
    ck = torch.load(a.combat_ckpt, map_location=dev)
    combat = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev).eval()
    load_compat(combat, ck["model"])
    T._ENC = T.Encoder(ck["sizes"]["cards"])

    rck = torch.load(a.run_ckpt, map_location=dev)
    # Match the checkpoint's head layout. Constructing this net WITHOUT per-kind heads while
    # the checkpoint carried them meant load_state_dict(strict=False) silently discarded all
    # 36 score_k tensors and the eval ran on the SHARED head -- which, under per-kind
    # training, receives no gradient at all and so was still bit-identical to the pre-training
    # checkpoint. Every number this script produced was a trained body bolted to a stale head.
    rsd = rck["model"] if "model" in rck else rck
    per_kind = any(k.startswith("score_k.") for k in rsd)
    runnet = RunNet(ck["sizes"]["cards"], ck["sizes"]["relics"], ck["sizes"]["potions"],
                    per_kind_heads=per_kind).to(dev).eval()
    print(f"  run net: per_kind_heads={per_kind}")
    load_compat(runnet, migrate_state_scalars(rsd))

    choices = defaultdict(lambda: defaultdict(int))
    deck_sizes = []
    hp_fracs = []
    rest_hp = []
    rest_by_hp = defaultdict(lambda: defaultdict(int))
    gate_by_room = defaultdict(lambda: defaultdict(int))
    combat_acts = defaultdict(int)
    skip_by_deck = defaultdict(lambda: defaultdict(int))

    def _label(kind, act):
        if kind == "travel":
            return act.get("point_type", "?")
        if kind == "card_reward":
            return "SKIP" if act.get("kind") == "alt" else "take-card"
        if kind == "rest":
            return act.get("option", "?")
        if kind == "shop":
            return act.get("kind", "?")
        if kind == "potion_gate":
            return act.get("kind", "?")
        if kind == "potion_ooc":
            return act.get("kind", "?")
        if kind == "card_select":
            return act.get("kind", "?")
        return act.get("kind", "?")

    def policy(kinds, obs, legal, idxs):
        acts = [0] * len(kinds)
        groups = defaultdict(list)
        for i, k in enumerate(kinds):
            groups["combat" if k == "combat" else "run"].append(i)
        with torch.no_grad():
            for grp, mem in groups.items():
                o = [obs[i] for i in mem]; l = [legal[i] for i in mem]
                if grp == "combat":
                    lg, _, mask = T.combat_forward(combat, o, l, dev)
                elif a.run_policy == "random":
                    for i in mem:
                        acts[i] = _rng.randrange(len(legal[i])) if legal[i] else 0
                    continue
                elif a.run_policy == "heuristic":
                    for i in mem:
                        acts[i] = _heur(kinds[i], obs[i], legal[i])
                    continue
                else:
                    lg, _, mask = T.run_forward(runnet, [kinds[i] for i in mem], o, l, dev)
                lg = lg.masked_fill(~mask, torch.finfo(lg.dtype).min / 4)
                # Greedy: this is a measurement, and sampling would add variance that has
                # nothing to do with the difference being measured.
                idx = lg.argmax(-1)
                for j, i in enumerate(mem):
                    acts[i] = int(idx[j])
        for i, k in enumerate(kinds):
            if k == "combat":
                # What the combat policy actually DOES. use_potion is the one to watch: it is
                # offered in every run fight but never appeared in combat-only training, so
                # the model has no learned behaviour for it.
                ch = legal[i][acts[i]] if 0 <= acts[i] < len(legal[i]) else {}
                combat_acts[ch.get("kind", "?")] += 1
                if any(a.get("kind") == "use_potion" for a in legal[i]):
                    combat_acts["_potion_OFFERED"] += 1
                continue
            ch = legal[i][acts[i]] if 0 <= acts[i] < len(legal[i]) else {}
            choices[k][_label(k, ch)] += 1
            p = obs[i].get("player", obs[i])
            if k == "travel" and p.get("deck_size"):
                deck_sizes.append(p["deck_size"])
                if p.get("max_hp"):
                    hp_fracs.append(p["hp"] / max(1, p["max_hp"]))
            if k == "card_reward":
                # Skip rate CONDITIONED on deck size. The whole question is whether skipping
                # is a learned, state-dependent judgement or a flat rate produced by the
                # entropy bonus. A real policy skips more as the deck fills up; a forced one
                # skips at the same rate regardless.
                ds = int(p.get("deck_size", 0) or 0)
                b = "<=12" if ds <= 12 else "13-17" if ds <= 17 else "18-24" if ds <= 24 else ">24"
                skip_by_deck[b][_label(k, ch)] += 1
            if k == "potion_gate":
                # The decision this breakdown exists for: does the policy DENY potions on
                # trivial monster fights and ALLOW them on elites and bosses? room_type is
                # supplied by the game in the gate observation; until it was encoded the
                # policy could not condition on it at all and sat at allow=100% everywhere.
                gate_by_room[obs[i].get("room_type") or "?"][_label(k, ch)] += 1
            if k == "rest" and p.get("max_hp"):
                hf = p["hp"] / max(1, p["max_hp"])
                rest_hp.append(hf)
                bucket = "<25%" if hf < .25 else "<50%" if hf < .5 else "<75%" if hf < .75 else ">=75%"
                rest_by_hp[bucket][_label("rest", ch)] += 1
        return acts

    per = max(1, a.runs // a.envs)
    t0 = time.time()
    with VecRunEnv(n_envs=a.envs, characters=a.characters.split(","), runs_per_env=per,
                   seed=a.seed, ascension=a.ascension, run_offset=0, act_cap=a.act_cap,
                   out_dir="/tmp/alphaslayer_evalrun",
                   expect_sizes=_expect_sizes(ck)) as v:
        v.run(policy, deadline_s=900)
        res, fights = list(v.results), list(v.fights)

    n = max(1, len(res))
    by_room = defaultdict(lambda: [0, 0])
    for f in fights:
        by_room[f.room][0] += f.won
        by_room[f.room][1] += 1
    print(f"combat={a.combat_ckpt} run={a.run_ckpt}")
    print(f"  runs={len(res)} win={sum(r.won for r in res)/n:.4f} "
          f"acts={sum(r.act for r in res)/n:.2f} floors={sum(r.floors for r in res)/n:.1f}")
    print(f"  fights={len(fights)} fight_win={sum(f.won for f in fights)/max(1,len(fights)):.4f}")
    for room, (w, t) in sorted(by_room.items()):
        print(f"    {room:8s} {w:4d}/{t:4d} = {w/max(1,t):.3f}")
    print(f"  wall={time.time()-t0:.0f}s")
    if deck_sizes:
        deck_sizes.sort()
        print(f"  deck size at travel: median={deck_sizes[len(deck_sizes)//2]} "
              f"p90={deck_sizes[int(0.9*len(deck_sizes))]} max={deck_sizes[-1]}")
    if hp_fracs:
        hp_fracs.sort()
        print(f"  hp fraction at travel: p10={hp_fracs[len(hp_fracs)//10]:.2f} "
              f"median={hp_fracs[len(hp_fracs)//2]:.2f}")
    if rest_hp:
        rest_hp.sort()
        print(f"  hp fraction ARRIVING AT A CAMPFIRE: p10={rest_hp[len(rest_hp)//10]:.2f} "
              f"median={rest_hp[len(rest_hp)//2]:.2f}")
    if skip_by_deck:
        print("  CARD-REWARD SKIP BY DECK SIZE (a learned policy skips more as the deck fills):")
        for b in ("<=12", "13-17", "18-24", ">24"):
            d = skip_by_deck.get(b)
            if not d:
                continue
            tot = sum(d.values())
            print(f"    deck {b:6s} n={tot:4d}  skip={100 * d.get('SKIP', 0) / tot:.0f}%")
    import alphaslayer.runmodel as _rm
    if _rm._AUDIT.get("n"):
        A = _rm._AUDIT
        print(f"  FEATURE AUDIT over {A['n']} run decisions: "
              f"held-potions non-empty {100*A['held']/A['n']:.1f}%, "
              f"upgraded-cards non-empty {100*A['upg']/A['n']:.1f}%, "
              f"room_type set {100*A['room']/A['n']:.1f}%")
    tr = {k: v for k, v in T._ENC.truncated.items() if v}
    print(f"  ENCODER TRUNCATION: {tr if tr else 'none'}")
    if combat_acts:
        tot = sum(v for k, v in combat_acts.items() if not k.startswith("_"))
        off = combat_acts.get("_potion_OFFERED", 0)
        used = combat_acts.get("use_potion", 0)
        print(f"  COMBAT ACTIONS n={tot}: " + ", ".join(
            f"{k}={100*v/tot:.1f}%" for k, v in sorted(combat_acts.items(), key=lambda kv: -kv[1])
            if not k.startswith("_")))
        print(f"    use_potion was LEGAL on {off} decisions ({100*off/max(1,tot):.1f}%); "
              f"chosen {used} times ({100*used/max(1,off):.2f}% of when available)")
    if gate_by_room:
        print("  POTION GATE BY ROOM (a sensible policy denies on monsters, allows on bosses):")
        for room in ("Monster", "Elite", "Boss"):
            d = gate_by_room.get(room)
            if not d:
                continue
            tot = sum(d.values())
            allow = d.get("allow", 0)
            print(f"    {room:8s} n={tot:5d}  allow={100*allow/tot:5.1f}%  deny={100*(tot-allow)/tot:5.1f}%")
    if rest_by_hp:
        print("  CAMPFIRE CHOICE BY HP (a sensible policy heals when low):")
        for b in ("<25%", "<50%", "<75%", ">=75%"):
            d = rest_by_hp.get(b)
            if not d:
                continue
            tot = sum(d.values())
            print(f"    hp {b:6s} n={tot:4d}  " +
                  ", ".join(f"{k2}={100*v/tot:.0f}%"
                            for k2, v in sorted(d.items(), key=lambda kv: -kv[1])))
    for k in sorted(choices):
        tot = sum(choices[k].values())
        top = sorted(choices[k].items(), key=lambda kv: -kv[1])
        print(f"  {k:12s} n={tot:5d}  " +
              ", ".join(f"{a}={100*c/tot:.0f}%" for a, c in top[:6]))


if __name__ == "__main__":
    main()
