#!/usr/bin/env bash
set -uo pipefail
cd /home/kyler/Documents/AlphaSlayer/python
run() {  # label policy ckpt encounters relics
  local label=$1 pol=$2 ck=$3 enc=$4 rel=$5
  local relarg=""; [ -n "$rel" ] && relarg="--relics $rel --reroll-deck"
  HIP_VISIBLE_DEVICES=0 PYTHONPATH=. timeout 900 .venv7/bin/python evaluate.py \
    --policy "$pol" --ckpt "$ck" --ascension 10 --episodes 500 --envs 25 \
    --characters IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT \
    --encounters "$enc" $relarg 2>&1 \
    | grep -oE "win_rate=[0-9.]+ mean_end_hp= *[0-9.]+ mean_turns= *[0-9.]+" | sed "s|^|  ${label}  |"
  pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 3
}
echo "=== A10 with Pandora's decks ==="
run "elite  model " bc     combat_capsule.pt elite PANDORAS_BOX
run "elite  greedy" greedy combat_bc.pt      elite PANDORAS_BOX
run "boss   model " bc     combat_capsule.pt boss  PANDORAS_BOX
run "boss   greedy" greedy combat_bc.pt      boss  PANDORAS_BOX
echo "=== A10 with plain STARTING deck (sanity: expected near-zero) ==="
run "boss   model  (starting deck)" bc combat_capsule.pt boss ""
run "elite  model  (starting deck)" bc combat_capsule.pt elite ""
echo "=== A10 BASELINE COMPLETE ==="
