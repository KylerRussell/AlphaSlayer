#!/usr/bin/env bash
# Stops the running M4 arm at STOP_AT iterations and starts the next one (paired arms on the
# same seeds). Generalises exp_chain_td2.sh.
#
#   CUR_TAG=td2 STOP_AT=40 WRAPPER_PID=.. TRAINER_PID=.. NEXT_TAG=td3 NEXT_INIT=unified_m4_long2.pt \
#   NEXT_EXTRA="--lam-step 0.5 --seed-prefix D2_ --run-critic --kl-coef 0.02" setsid nohup ./exp_chain_next.sh &
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
LOG=$ROOT/logs/chain_${NEXT_TAG:?}.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

say "waiting for ${CUR_TAG:?} iteration ${STOP_AT:?}"
until grep -qE "^iter +$STOP_AT/" "$ROOT/logs/m4_$CUR_TAG.log"; do sleep 20; done
sleep 20      # the checkpoint is written right after the report lines
wrapper=${WRAPPER_PID:?}; trainer=${TRAINER_PID:?}
say "stopping $CUR_TAG: wrapper=$wrapper trainer=$trainer ($(ps -o cmd= -p "$trainer" | cut -c1-80))"
kill -9 "$wrapper"          # first, or it would relaunch the trainer
kill "$trainer"
for _ in $(seq 1 60); do pgrep -x SlayTheSpire2 >/dev/null || break; sleep 5; done
if pgrep -x SlayTheSpire2 >/dev/null; then
  say "game processes still alive after 5 min; killing them"
  pkill -9 -x SlayTheSpire2
fi
say "$CUR_TAG stopped at $(grep -cE '^iter +[0-9]+/' "$ROOT/logs/m4_$CUR_TAG.log") iterations; starting $NEXT_TAG"
cd "$ROOT"
TAG=$NEXT_TAG INIT=${NEXT_INIT:?} ITERS=${NEXT_ITERS:-150} EXTRA="${NEXT_EXTRA:-}" ./exp_m4.sh
say "$NEXT_TAG wrapper exited"
