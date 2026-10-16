#!/bin/bash
# Fixed-seed greedy eval of td2@50 with TURN search in elite and boss fights (search_eval.py
# --mode turn: the policy's top-k choices, each played to the end of the turn by the policy in
# sampled worlds, picked by pi * exp(q / tau)); pairs per run with td2@50's plain greedy eval.
set -u
cd "$(dirname "$0")/python"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=1 HSA_ENABLE_SDMA=0 PYTHONPATH=.
export XDG_DATA_HOME="$(../headless_home.sh 2>/dev/null)"
REF=../logs/eval_runs/unified_m4_td2_it50_g.json
NAME=${NAME:-search_td2_be_turn_k4s2_prior0.01}
echo "===== $NAME $(date +%T)"
.venv7/bin/python search_eval.py --ckpt unified_m4_td2_it50.pt --game ${GAME:-data/sts2_testgame2} --slots ${SLOTS:-22} \
  --mode turn --rooms Elite,Boss --k 4 --samples 2 --pick prior --tau 0.01 \
  --name "$NAME" --dump ../logs/${NAME}.jsonl "$@" 2>&1 | grep -v amdgpu.ids
.venv7/bin/python pair_evals.py $REF ../logs/eval_runs/${NAME}_g.json | tail -4
echo "===== done $(date +%T)"
