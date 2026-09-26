#!/usr/bin/env bash
# Waits for the capsule run, then evaluates transfer to the PREVIOUS mixture
# (Pandora's only, no Large Capsule, no extra relics).
#
# Both checkpoints are re-measured under the CURRENT architecture: the 0.804 figure for
# combat_universal.pt predates the relic_proj module, so loading it today gives that module
# fresh weights and different behaviour. Comparing against the stale number would be wrong.
set -uo pipefail
cd /home/kyler/Documents/AlphaSlayer/python
LOG=/tmp/rl_capsule.log
while true; do
  grep -q "saved combat_capsule" "$LOG" 2>/dev/null && break
  [ "$(ps -eo args | awk '/train_rl/ && /python/ && !/awk/ {c++} END{print c+0}')" -eq 0 ] && break
  sleep 60
done
echo "=== capsule run ended at $(grep -cE '^iter' "$LOG")/150; running transfer test ==="
pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 3

for ck in combat_universal.pt combat_capsule.pt; do
  echo "--- $ck on PREVIOUS mixture (Pandora's only) ---"
  HIP_VISIBLE_DEVICES=0 PYTHONPATH=. timeout 900 .venv7/bin/python evaluate.py \
    --policy bc --ckpt "$ck" --episodes 750 --envs 25 \
    --characters IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT \
    --encounters mix --mix 0.5,0.25,0.25 --relics PANDORAS_BOX --reroll-deck 2>&1 \
    | grep win_rate | sed "s|^|  $ck  |"
  pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 3
done

echo "--- greedy reference on the same mixture ---"
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. timeout 900 .venv7/bin/python evaluate.py \
  --policy greedy --episodes 750 --envs 25 \
  --characters IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT \
  --encounters mix --mix 0.5,0.25,0.25 --relics PANDORAS_BOX --reroll-deck 2>&1 \
  | grep win_rate | sed 's|^|  greedy  |'
pkill -9 -x SlayTheSpire2 2>/dev/null
echo "=== TRANSFER TEST COMPLETE ==="
