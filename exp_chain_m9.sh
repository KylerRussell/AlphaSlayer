#!/bin/bash
# One run at a time: waits for the card/shop run-search eval (logs/run_search_deck.log) to finish,
# then starts M9 (td2@50 + fight search in elite/boss fights, distilled) on the whole machine.
cd "$(dirname "$0")"
until grep -q "===== done\|Traceback" logs/run_search_deck.log; do sleep 60; done
pgrep -f "[s]earch_eval.py" >/dev/null && { echo "search_eval still running; not starting M9"; exit 1; }
TAG=m9 INIT=unified_m4_td2_it50.pt ITERS=100 POOL_PATTERN="sts2_testgame3/SlayTheSpire2" \
EXTRA="--lam-step 0.5 --seed-prefix M9_ --run-critic --pool 22 --trainer-cpus 0,1,16,17 --eval-every 10 --game-dir data/sts2_testgame3 --search-rooms Elite,Boss --search-k 4 --search-samples 2 --search-tau 0.01" \
./exp_m4.sh
