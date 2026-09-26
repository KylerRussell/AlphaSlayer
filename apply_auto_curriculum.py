#!/usr/bin/env python3
"""Integrates automatic combat-curriculum passes into train_run.py.

After this, one command does what rounds 2 and 3 did by hand:

    train_run.py --iters 600 --curriculum-every 25 ...

Every 25 iterations the trainer pauses run collection, PPO-trains the combat net on the decks
it has harvested from its own recent runs (against one act's encounter pools, cycling through
a plan), then resumes. Harvesting turns itself on, because the curriculum's whole point is
training on the deck distribution the CURRENT policy builds -- the round-2 failure was
training on Pandora's-Box rerolls, which taught act-2 skill and destroyed act-1 skill.

The run envs are already torn down between iterations, so a pass borrows the machine rather
than competing with it for memory.

Run only when no training is live: the auto-resume wrapper re-imports this module on restart.
"""
from __future__ import annotations

import os
import sys

ROOT = "/home/kyler/Documents/AlphaSlayer"
TR = f"{ROOT}/python/train_run.py"


def patch(path, pairs):
    with open(path) as fh:
        s = fh.read()
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected 1 occurrence, found {n}:\n  {old[:90]!r}")
        s = s.replace(old, new)
    with open(path, "w") as fh:
        fh.write(s)
    print(f"patched {path}")


def main():
    # Scan /proc rather than `pgrep -f`: the pattern appears in this script's own source and
    # in whatever shell launched it, so a substring match flags itself.
    live = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or entry == str(os.getpid()):
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                argv = [a.decode(errors="replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            continue
        if len(argv) >= 2 and "python" in argv[0] and argv[1].endswith(
                ("train_run.py", "train_combat_decks.py", "train_rl.py")):
            live.append(f"{entry}:{argv[1]}")
    if live:
        raise SystemExit(f"refusing to patch: training still running ({live})")

    patch(TR, [
        # ---- imports ----------------------------------------------------------------
        ("from alphaslayer.benchmark import BenchmarkPool, load as bench_load\n",
         "from alphaslayer.benchmark import BenchmarkPool, load as bench_load\n"
         "from alphaslayer import curriculum as curr\n"),

        # ---- arguments ---------------------------------------------------------------
        ('    ap.add_argument("--collect-timeout", type=float, default=600.0,\n'
         '                    help="hard wall-clock cap on one collection pass")\n',
         '    ap.add_argument("--collect-timeout", type=float, default=600.0,\n'
         '                    help="hard wall-clock cap on one collection pass")\n'
         '    # ---- automatic combat curriculum -----------------------------------------\n'
         '    ap.add_argument("--curriculum-every", type=int, default=0,\n'
         '                    help="every N iterations, pause run collection and PPO the "\n'
         '                         "COMBAT net on decks harvested from recent runs, against "\n'
         '                         "one act\'s encounters. 0 disables. This is what broke the "\n'
         '                         "act-2 plateau: the combat net cannot learn act 2 from run "\n'
         '                         "data because it only reaches act-2 fights in a few percent "\n'
         '                         "of runs, so it is taught them in isolation instead. Turns "\n'
         '                         "harvesting on by itself -- training on any other deck "\n'
         '                         "distribution cost act-1 skill (benchmark 0.765 -> 0.680) "\n'
         '                         "where real harvested decks left it at 0.767")\n'
         '    ap.add_argument("--curriculum-plan", default="1,1,0,1,1,2",\n'
         '                    help="0-based acts, one per pass, cycled. Default weights act 2 "\n'
         '                         "(the measured deficit: boss 0.109 vs 0.806 in act 1), "\n'
         '                         "revisits act 1 to hold the line, and touches act 3")\n'
         '    ap.add_argument("--curriculum-iters", type=int, default=25,\n'
         '                    help="PPO iterations per pass")\n'
         '    ap.add_argument("--curriculum-fights", type=int, default=200,\n'
         '                    help="fights per curriculum iteration")\n'
         '    ap.add_argument("--curriculum-mix", default="0.5,0.25,0.25",\n'
         '                    help="boss,elite,regular sampling weights within a pass")\n'
         '    ap.add_argument("--curriculum-lr", type=float, default=5e-5,\n'
         '                    help="combat lr DURING a pass; --combat-lr is far lower because "\n'
         '                         "it rides on run data, which is act-1 dominated")\n'
         '    ap.add_argument("--curriculum-kl", type=float, default=0.05,\n'
         '                    help="KL penalty toward --combat-ref during a pass. Unbounded "\n'
         '                         "drift is what cost act-1 skill in round 2")\n'
         '    ap.add_argument("--curriculum-recent", type=int, default=6000,\n'
         '                    help="use only the most recent N harvested decks, so a pass "\n'
         '                         "tracks the decks the CURRENT policy builds")\n'
         '    ap.add_argument("--curriculum-full-hp", action="store_true",\n'
         '                    help="start curriculum fights at full hp instead of the hp the "\n'
         '                         "deck actually had. Off by default: the policy arrives in "\n'
         '                         "act 2 at ~35%% hp and risk assessment depends on it")\n'),

        # ---- defaults that make the flag self-sufficient -----------------------------
        ('    use_stable_attention()\n    modcheck()\n',
         '    use_stable_attention()\n    modcheck()\n'
         '    if args.curriculum_every:\n'
         '        # A pass trains on harvested decks, so harvesting is not optional; default\n'
         '        # the file next to the run checkpoint so a tag gets its own harvest.\n'
         '        if not args.harvest_decks:\n'
         '            _stem = args.out[:-3] if args.out.endswith(".pt") else args.out\n'
         '            args.harvest_decks = _stem + "_harvest.jsonl"\n'
         '            print(f"  curriculum: harvesting to {args.harvest_decks}", flush=True)\n'
         '        # The gate defends an ACT-1-ONLY benchmark. A curriculum deliberately moves\n'
         '        # the fight model off that distribution, so a rollback would undo exactly\n'
         '        # the work being done. Measure, never revert.\n'
         '        if not args.gate_readonly:\n'
         '            args.gate_readonly = True\n'
         '            print("  curriculum: forcing --gate-readonly (the gate\'s baseline is "\n'
         '                  "act-1 only and would roll back act-2 learning)", flush=True)\n'),

        # ---- the pass itself ---------------------------------------------------------
        ('        if it % 10 == 0:\n            card_stats.save()\n',
         '        # ---- automatic combat curriculum ------------------------------------\n'
         '        # Placed after the harvest flush so this iteration\'s decks are visible, and\n'
         '        # before the checkpoint save so the pass\'s weights are what gets written.\n'
         '        if (args.curriculum_every and combat_opt is not None\n'
         '                and it % args.curriculum_every == 0):\n'
         '            plan = [int(x) for x in args.curriculum_plan.split(",") if x.strip() != ""]\n'
         '            c_act = plan[(it // args.curriculum_every - 1) % len(plan)]\n'
         '            decks = curr.load_decks(args.harvest_decks, act=c_act,\n'
         '                                    recent=args.curriculum_recent)\n'
         '            if not decks:\n'
         '                print(f"    curriculum act {c_act + 1}: no harvested decks yet; "\n'
         '                      f"skipping", flush=True)\n'
         '            else:\n'
         '                c_args = copy.copy(args)\n'
         '                c_args.combat_kl = args.curriculum_kl\n'
         '                # Reuse the combat optimiser so momentum stays coherent across run\n'
         '                # and curriculum updates; only the step size differs.\n'
         '                saved_lrs = [g["lr"] for g in combat_opt.param_groups]\n'
         '                for g in combat_opt.param_groups:\n'
         '                    g["lr"] = args.curriculum_lr\n'
         '                try:\n'
         '                    cstats = curr.train_pass(\n'
         '                        combat, combat_opt, decks, device=device,\n'
         '                        combat_forward=combat_forward, sample=sample,\n'
         '                        ppo_update=ppo_update, pargs=c_args, ref=combat_ref,\n'
         '                        iters=args.curriculum_iters,\n'
         '                        fights_per_iter=args.curriculum_fights,\n'
         '                        mix=tuple(float(x) for x in args.curriculum_mix.split(",")),\n'
         '                        envs=args.envs, act=c_act,\n'
         '                        real_hp=not args.curriculum_full_hp,\n'
         '                        hp_bonus=args.hp_bonus, shaping=args.combat_shaping or 0.5,\n'
         '                        seed=1000 + it,\n'
         '                        log=lambda m: print(m, flush=True), pools=curr_pools)\n'
         '                finally:\n'
         '                    for g, lr in zip(combat_opt.param_groups, saved_lrs):\n'
         '                        g["lr"] = lr\n'
         '                print(f"    curriculum act {c_act + 1} done: {cstats[\'fights\']} "\n'
         '                      f"fights, win={cstats[\'win\']:.3f} on {len(decks)} real decks",\n'
         '                      flush=True)\n'
         '\n'
         '        if it % 10 == 0:\n            card_stats.save()\n'),

        # ---- pool lifetime -----------------------------------------------------------
        ('    quality_ema = 0.0\n    ema_beta = 0.2\n',
         '    quality_ema = 0.0\n    ema_beta = 0.2\n'
         '    # deckserve pools for the curriculum, kept alive across passes: a pool costs\n'
         '    # ~10s to boot and a pass iteration only ~20s.\n'
         '    curr_pools = {}\n'),
        ('    if expert is not None:\n        expert.close()\n    card_stats.save()\n',
         '    for _p in curr_pools.values():\n'
         '        try:\n'
         '            _p.close()\n'
         '        except Exception:\n'
         '            pass\n'
         '    if expert is not None:\n        expert.close()\n    card_stats.save()\n'),
    ])

    # The inserted code uses `copy` (for the per-pass args clone); it deliberately avoids
    # `os`, which this module does not import.
    with open(TR) as fh:
        if "import copy" not in fh.read():
            raise SystemExit("train_run.py is missing `import copy`; add it before using this")
    print("\nintegrated. smoke test:\n"
          "  train_run.py --iters 2 --curriculum-every 1 --curriculum-iters 2 ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
