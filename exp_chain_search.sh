#!/bin/bash
# One run at a time, 2 batches (192 fixed-seed runs) each, all on the fixed build (testgame3),
# all paired with td2@50's plain greedy eval:
#   1. fight turn search in boss fights only
#   2. fight turn search in elite + boss fights (then paired directly against 1)
#   3. elite + boss fight search + card reward, shop and route search
#   4. M9 (td2@50 + elite/boss fight search, distilled), as exp_chain_m9.sh
cd "$(dirname "$0")"
L=logs/chain_search.log
clean() {
  for _ in $(seq 30); do pgrep -f "sts2_testgam[e]3/SlayTheSpire2" >/dev/null || return 0; sleep 2; done
  echo "game processes still alive after an eval; stopping the chain"; exit 1
}
GAME=data/sts2_testgame3 NAME=search_td2_b_turn_v2 ./exp_turn_search.sh --rooms Boss --eval-iters 2 \
  --out /tmp/alphaslayer_search_b_ > logs/search_b_v2.log 2>&1; clean
GAME=data/sts2_testgame3 NAME=search_td2_be_turn_v2 ./exp_turn_search.sh --rooms Elite,Boss --eval-iters 2 \
  --out /tmp/alphaslayer_search_be_ > logs/search_be_v2.log 2>&1; clean
(cd python && .venv7/bin/python pair_evals.py ../logs/eval_runs/search_td2_b_turn_v2_g.json \
  ../logs/eval_runs/search_td2_be_turn_v2_g.json) > logs/pair_b_vs_be.log 2>&1
GAME=data/sts2_testgame3 NAME=search_td2_all_v2 ./exp_turn_search.sh --rooms Elite,Boss --eval-iters 2 \
  --run-search card_reward,shop,travel --run-k 3 --deck-plays 8 --worlds 4 --boss-plays 2 --max-live 6 \
  --run-dump ../logs/search_td2_all_v2_run.jsonl --deadline 36000 \
  --out /tmp/alphaslayer_search_all_ > logs/search_all_v2.log 2>&1; clean
echo "===== searches done $(date +%T)" >> $L
TAG=m9 INIT=unified_m4_td2_it50.pt ITERS=100 POOL_PATTERN="sts2_testgame3/SlayTheSpire2" \
EXTRA="--lam-step 0.5 --seed-prefix M9_ --run-critic --pool 22 --trainer-cpus 0,1,16,17 --eval-every 10 --game-dir data/sts2_testgame3 --search-rooms Elite,Boss --search-k 4 --search-samples 2 --search-tau 0.01" \
./exp_m4.sh
