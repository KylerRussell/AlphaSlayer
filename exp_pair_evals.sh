#!/usr/bin/env bash
# Greedy fixed-seed evals with per-run labels (unique run ids), for seed-by-seed pairing.
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
LOG=$ROOT/logs/eval_pair.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
cd "$ROOT/python"
export XDG_DATA_HOME="$("$ROOT/headless_home.sh" 2>/dev/null)" PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=1 HSA_ENABLE_SDMA=0 PYTHONPATH=.
for ck in ${CKPTS:-unified_m4_td2.pt unified_m4_long2.pt}; do
  say "greedy eval of $ck"
  .venv7/bin/python train_unified.py --init "$ck" --out "data/evalonly_g_$ck" --bf16 --pool 26 \
      --trainer-cpus 0,1,16,17 --eval-only 2>&1 | grep -E "^eval|Traceback|Error" | sed "s/^/  $ck: /" >> "$LOG"
done
pgrep -f "[t]rain_unified.py" >/dev/null || pkill -9 -f -- "--probe=runserve"
say "done"
