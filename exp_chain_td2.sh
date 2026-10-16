#!/usr/bin/env bash
# Stops td1 (hand potential, --lam-step 0.5) at STOP_AT iterations and starts td2: the same
# checkpoint, seeds and settings plus --run-critic, so the two arms are paired iteration by
# iteration on the D2_ seeds (and the first 20 also against the collect-only baseline).
#
#   STOP_AT=40 WRAPPER_PID=<td1 exp_m4.sh> TRAINER_PID=<td1 train_unified> setsid nohup ./exp_chain_td2.sh &
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
LOG=$ROOT/logs/chain_td2.log
STOP_AT=${STOP_AT:-40}
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

say "waiting for td1 iteration $STOP_AT"
until grep -qE "^iter +$STOP_AT/" "$ROOT/logs/m4_td1.log"; do sleep 20; done
# The checkpoint for STOP_AT is written right after its report lines; give it a moment.
sleep 20
wrapper=${WRAPPER_PID:?}; trainer=${TRAINER_PID:?}
say "stopping td1: wrapper=$wrapper trainer=$trainer ($(ps -o cmd= -p "$trainer" | cut -c1-80))"
kill -9 "$wrapper"          # first, or it would relaunch the trainer
kill "$trainer"
for _ in $(seq 1 60); do pgrep -x SlayTheSpire2 >/dev/null || break; sleep 5; done
if pgrep -x SlayTheSpire2 >/dev/null; then
  say "game processes still alive after 5 min; killing td1's (children of no live trainer)"
  pkill -9 -x SlayTheSpire2
fi
say "td1 stopped at $(grep -cE '^iter +[0-9]+/' "$ROOT/logs/m4_td1.log") iterations; starting td2"
cd "$ROOT"
TAG=td2 INIT=unified_m4_long2.pt ITERS=150 \
EXTRA="--lam-step 0.5 --seed-prefix D2_ --run-critic" ./exp_m4.sh
say "td2 wrapper exited"
