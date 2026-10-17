#!/bin/bash
# Fixed-seed greedy eval of td2@50 with RUN search (search_eval.py --run-search): card rewards
# and shops valued by finishing the room, going straight to the act boss and playing it
# --deck-plays times (damage-graded); routes by playing the path to the boss in --worlds
# worlds. Fight search off, so the effect is run search alone. Pairs per run with td2@50's
# plain greedy eval. Waits for any other search eval on the machine first (WAIT_FOR=<log>).
set -u
cd "$(dirname "$0")/python"
if [ -n "${WAIT_FOR:-}" ]; then
  until grep -q "===== done\|Traceback" "$WAIT_FOR"; do sleep 60; done
fi
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=1 HSA_ENABLE_SDMA=0 PYTHONPATH=.
export XDG_DATA_HOME="$(../headless_home.sh 2>/dev/null)"
REF=../logs/eval_runs/unified_m4_td2_it50_g.json
KINDS=${KINDS:-card_reward,shop}
NAME=${NAME:-search_td2_run_${KINDS//,/-}}
echo "===== $NAME $(date +%T)"
.venv7/bin/python search_eval.py --ckpt unified_m4_td2_it50.pt --game ${GAME:-data/sts2_testgame3} --slots ${SLOTS:-22} \
  --rooms "" --run-search "$KINDS" --run-k 3 --deck-plays 8 --worlds 4 --boss-plays 2 --max-live ${MAX_LIVE:-6} \
  --deadline 36000 --name "$NAME" --run-dump ../logs/${NAME}.jsonl --out /tmp/alphaslayer_rsearch_ "$@" 2>&1 \
  | grep --line-buffered -v amdgpu.ids
.venv7/bin/python pair_evals.py $REF ../logs/eval_runs/${NAME}_g.json | tail -4
echo "===== done $(date +%T)"
