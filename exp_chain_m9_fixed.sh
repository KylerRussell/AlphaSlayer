#!/bin/bash
# Takes over from exp_chain_search.sh after its step 3 (elite+boss + card/shop/route eval):
# installs the stale-combat fix (StaleCombat.cs, card-id crash) into testgame3, then runs M9.
cd "$(dirname "$0")"
until grep -q "===== done\|Traceback" logs/search_all_v2.log; do sleep 30; done
for _ in $(seq 60); do pgrep -f "sts2_testgam[e]3/SlayTheSpire2" >/dev/null || break; sleep 2; done
pgrep -f "[s]earch_eval.py" >/dev/null && { echo "search_eval still running; not starting M9"; exit 1; }
G=python/data/sts2_testgame3/mods/alphaslayer_probe
cp logs/.probe_stalefix.dll $G/.tmp.dll && mv $G/.tmp.dll $G/alphaslayer_probe.dll
echo "installed stale-combat fix $(date +%T)"
TAG=m9 INIT=unified_m4_td2_it50.pt ITERS=100 POOL_PATTERN="sts2_testgame3/SlayTheSpire2" \
EXTRA="--lam-step 0.5 --seed-prefix M9_ --run-critic --pool 22 --trainer-cpus 0,1,16,17 --eval-every 10 --game-dir data/sts2_testgame3 --search-rooms Elite,Boss --search-k 4 --search-samples 2 --search-tau 0.01" \
./exp_m4.sh
