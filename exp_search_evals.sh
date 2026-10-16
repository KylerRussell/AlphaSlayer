#!/bin/bash
# Fixed-seed greedy evals of td2@50 WITH fight search (search_eval.py), on the test copy of the
# game; each pairs per run with logs/eval_runs/unified_m4_td2_it50_g.json (pair_evals.py).
#   1. no-search control: the test copy + refactored probe must reproduce plain greedy
#   2. margin 0.01: override the policy only where the search's value is clearly higher
#   3. prior tau 0.01: argmax log pi + q / tau
set -u
cd "$(dirname "$0")/python"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=1 HSA_ENABLE_SDMA=0 PYTHONPATH=.
export XDG_DATA_HOME="$(../headless_home.sh 2>/dev/null)"
CK=unified_m4_td2_it50.pt
SLOTS=${SLOTS:-20}
REF=../logs/eval_runs/unified_m4_td2_it50_g.json
run() {  # name, args...
  local name=$1; shift
  echo "===== $name $(date +%T)"
  .venv7/bin/python search_eval.py --ckpt $CK --slots $SLOTS --name "$name" "$@" 2>&1 | grep -v amdgpu.ids
  .venv7/bin/python pair_evals.py $REF ../logs/eval_runs/${name}_g.json | tail -4
}
run search_td2_none --rooms ""
run search_td2_be_margin0.01 --rooms Elite,Boss --pick margin --margin 0.01 --dump ../logs/search_be_margin.jsonl
run search_td2_be_prior0.01 --rooms Elite,Boss --pick prior --tau 0.01 --dump ../logs/search_be_prior.jsonl
echo "===== all done $(date +%T)"
