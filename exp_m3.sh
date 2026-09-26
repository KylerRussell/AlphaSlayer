#!/usr/bin/env bash
# M3: distil the r5 teachers into the unified model at three sizes, then play-test all of them
# against the teachers on the SAME 500 seeds.
#
#   ./exp_m3.sh            (expects python/data/distill from collect_distill.py)
#
# Sweep (equal data, equal epochs, all trained with the random loop prior):
#   d256_222   7M   small baseline
#   d384_242  18M   deep stack
#   d512_121  18.5M wide + shallow (the default), also play-tested with adaptive loops
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
PY=$ROOT/python/.venv7/bin/python
LOG=$ROOT/logs/exp_m3.log
EPOCHS=${EPOCHS:-4}
RUNS=${RUNS:-500}
cd "$ROOT/python"
export XDG_DATA_HOME="$("$ROOT/headless_home.sh" 2>/dev/null)"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 HSA_ENABLE_SDMA=0 PYTHONPATH=.
export PYTORCH_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

say "=== M3 start: epochs=$EPOCHS eval runs=$RUNS ==="
say "teacher baseline (r5) on the eval seeds"
$PY eval_unified.py --teacher --runs "$RUNS" --out m3_eval_teacher.json 2>&1 \
  | grep -vE "amdgpu|Warning|warn|nested" | tail -6 | tee -a "$LOG"
pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 3

for CFG in d512_121 d384_242 d256_222; do
  say "train $CFG"
  $PY train_distill.py --data data/distill --config "$CFG" --out "unified_m3_$CFG.pt" \
      --epochs "$EPOCHS" --bf16 2>&1 | grep -E "^data|^model|^epoch|Traceback|Error" | tee -a "$LOG"
  if [ ! -f "unified_m3_$CFG.pt" ]; then say "!! $CFG produced no checkpoint"; continue; fi
  say "play-test $CFG, one pass"
  $PY eval_unified.py --ckpt "unified_m3_$CFG.pt" --loop 1 --bf16 --runs "$RUNS" \
      --out "m3_eval_${CFG}_k1.json" 2>&1 | grep -vE "amdgpu|Warning|warn|nested" | tail -7 | tee -a "$LOG"
  pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 3
done

say "play-test d512_121 with adaptive loops"
$PY eval_unified.py --ckpt unified_m3_d512_121.pt --adaptive --bf16 --runs "$RUNS" \
    --out m3_eval_d512_121_adaptive.json 2>&1 | grep -vE "amdgpu|Warning|warn|nested" | tail -7 | tee -a "$LOG"
pkill -9 -x SlayTheSpire2 2>/dev/null
say "=== M3 done ==="
