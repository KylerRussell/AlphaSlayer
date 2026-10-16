#!/usr/bin/env bash
# Stops td2 at its next checkpoint, evaluates long2 / td1 / td2 on the fixed EV_ seeds (480
# runs each, identical maps), then resumes td2 with an evaluation every 25 iterations.
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
LOG=$ROOT/logs/eval_fixed.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
n=$(grep -cE "^iter" "$ROOT/logs/m4_td2.log")
until [ "$(grep -cE '^iter' "$ROOT/logs/m4_td2.log")" -gt "$n" ]; do sleep 5; done
sleep 15
kill -9 "${WRAPPER_PID:?}"; kill "${TRAINER_PID:?}"
for _ in $(seq 1 30); do pgrep -x SlayTheSpire2 >/dev/null || break; sleep 2; done
cd "$ROOT/python"
export XDG_DATA_HOME="$("$ROOT/headless_home.sh" 2>/dev/null)" PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=1 HSA_ENABLE_SDMA=0 PYTHONPATH=.
for ck in unified_m4_long2.pt unified_m4_td1.pt unified_m4_td2.pt; do
  say "evaluating $ck"
  .venv7/bin/python train_unified.py --init "$ck" --out "data/evalonly_$ck" --bf16 --pool 26 \
      --trainer-cpus 0,1,16,17 --eval-only 2>&1 | grep -E "^eval|Traceback|Error" | sed "s/^/  $ck: /" >> "$LOG"
done
cd "$ROOT"
say "resuming td2 with --eval-every 25"
TAG=td2 INIT=unified_m4_long2.pt ITERS=150 \
EXTRA="--lam-step 0.5 --seed-prefix D2_ --run-critic --pool 26 --trainer-cpus 0,1,16,17 --eval-every 25" ./exp_m4.sh
