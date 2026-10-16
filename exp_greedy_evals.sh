#!/usr/bin/env bash
# Pauses td6 at a checkpoint, runs GREEDY fixed-seed evaluations (long2 twice = determinism
# check, td1@40, td2@50), then resumes td6 with an evaluation every 50 iterations.
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
LOG=$ROOT/logs/eval_greedy.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
n=$(grep -cE "^iter" "$ROOT/logs/m4_td6.log")
until [ "$(grep -cE '^iter' "$ROOT/logs/m4_td6.log")" -gt "$n" ]; do sleep 5; done
sleep 15
kill -9 ${KILL_FIRST:?}; kill "${TRAINER_PID:?}"
for _ in $(seq 1 30); do pgrep -x SlayTheSpire2 >/dev/null || break; sleep 2; done
say "td6 paused at $(grep -cE '^iter' "$ROOT/logs/m4_td6.log") iterations"
cd "$ROOT/python"
export XDG_DATA_HOME="$("$ROOT/headless_home.sh" 2>/dev/null)" PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=1 HSA_ENABLE_SDMA=0 PYTHONPATH=.
for ck in unified_m4_long2.pt unified_m4_long2.pt unified_m4_td1.pt unified_m4_td2.pt; do
  say "greedy eval of $ck"
  .venv7/bin/python train_unified.py --init "$ck" --out "data/evalonly_g_$ck" --bf16 --pool 26 \
      --trainer-cpus 0,1,16,17 --eval-only 2>&1 | grep -E "^eval|Traceback|Error" | sed "s/^/  $ck: /" >> "$LOG"
done
cd "$ROOT"
say "resuming td6"
TAG=td6 INIT=unified_m4_long2.pt ITERS=150 \
EXTRA="--lam-step 0.5 --seed-prefix S6_ --run-critic --pool 26 --trainer-cpus 0,1,16,17 --eval-every 50 --seed-block 10 --block-seeds 20 --history-baseline" ./exp_m4.sh
