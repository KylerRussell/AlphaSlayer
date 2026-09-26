#!/usr/bin/env bash
# Plateau sweep: one 300-iteration full-run (3 act) training run per candidate fix.
#
# Why these four, and why one variable each.
#
# The act-1+2 run (logs/a3.log, 327 iterations) stopped improving after ~130 iterations at
# a1 0.60 / a2|a1 0.06. Three measurements from that log say where the ceiling came from, and
# each candidate below moves exactly one of them so the comparison against ctl is readable:
#
#   ctl   Control. 3 acts and 20 runs/iter are the only changes from a3, so it separates
#         "the plateau follows the configuration" from "the plateau followed act-cap 2".
#
#   hp    Campfire collapse. rest went HEAL 54% -> 99% by iteration 50 and pinned at 100%
#         with normalised entropy 0.00 from iteration 100 on: the policy NEVER upgraded a
#         card. hp-shaping pays healing immediately while a smith pays off in a boss fight
#         ~20 decisions later, past the GAE horizon (~16 decisions at gamma 0.98,
#         lambda 0.95). Dropping the weight 0.5 -> 0.1 removes the thumb on the scale
#         without removing the term.
#
#   cmb   Boss ceiling. ~60% of deaths were boss deaths, and the combat net's benchmark boss
#         win rate never moved off 0.69-0.73 in 12 gate checks, by design: lr 1e-5 under a
#         KL 0.5 leash to a frozen reference. 0.8 (reach the boss) x 0.75 (win it) = 0.6 is
#         exactly where a1 stalled. This lets the fight model actually adapt (lr 5e-5,
#         KL 0.1) with --gate-readonly, so the act-1 benchmark is still MEASURED every 25
#         iterations but no longer rolls back the very adaptation being tested.
#
#   gam   Reward inversion in the discount schedule. --progress-shaping is
#         w*(gamma*phi' - phi) with phi = the RAW floor number, so it pays for advancing
#         only while floor < gamma/(1-gamma). Measured against the real Buffer.finish_run:
#         at gamma 0.967 a run clearing act 2 on floor 34 collects LESS total reward (1.370)
#         than one dying on floor 20 (1.562), and the progress term alone peaks at floor 28.
#         a3 sat at gamma 0.980-0.986, just above the 0.983 threshold, so it escaped the
#         inversion except in its first ~25 iterations -- but its a2 rate never moved
#         (+0.001 +/- 0.002 over 332 iterations), which is what a depth disincentive looks
#         like. gamma-lo 0.99 puts break-even at floor 99, beyond the 51-floor run, and
#         target-floors 51 lets credit reach back across all three acts.
#
#   drop  PPO ratio hygiene. Rollout acts in eval() and the update recomputes logp in
#         train(), so dropout alone shifts the importance ratio before any weight change:
#         measured on on-policy states, 2.6% of combat and 3.6% of travel samples start
#         outside the 0.8-1.2 clip window. Small, and the last one ranked for that reason,
#         but no-dropout is the conventional PPO setting and costs nothing.
#
# Sequential, not parallel: two trainers would double GPU contention (the 09-03 hangs took
# the desktop down with them) and 40 game processes contend on 32 threads -- python/README.md
# measured 32 envs regressing to 757 steps/s from 1000 at 24.
set -uo pipefail
cd /home/kyler/Documents/AlphaSlayer
ITERS=${ITERS:-300}
LOG=logs/exp_plateau.log
mkdir -p logs

run_exp() {
  local tag="$1"; shift
  local done_iters
  done_iters=$(grep -c '^iter ' "logs/$tag.log" 2>/dev/null || echo 0)
  if [ "$done_iters" -ge "$ITERS" ]; then
    echo "[$(date '+%m-%d %H:%M')] $tag already at $done_iters/$ITERS iterations; skipping" >> "$LOG"
    return 0
  fi
  echo "[$(date '+%m-%d %H:%M')] START $tag ($* )" >> "$LOG"
  env "$@" TAG="$tag" TARGET="$ITERS" ./run_act2.sh
  echo "[$(date '+%m-%d %H:%M')] END   $tag -> $(grep -c '^iter ' "logs/$tag.log" 2>/dev/null || echo 0) iterations" >> "$LOG"
  pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 5
}

echo "=== plateau sweep, $ITERS iterations per candidate, started $(date '+%m-%d %H:%M') ===" >> "$LOG"
run_exp ctl
run_exp gam   GAMMALO=0.99 GTF=51
run_exp hp    HPSHAPE=0.1
run_exp cmb   COMBATLR=5e-5 COMBATKL=0.1 EXTRA=--gate-readonly
run_exp drop  DROPOUT=0
echo "=== plateau sweep DONE $(date '+%m-%d %H:%M') ===" >> "$LOG"
