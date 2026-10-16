#!/usr/bin/env bash
# Lets td2 run through its iteration-50 fixed-seed evaluation, then starts td6: td2's settings
# plus seed blocks (20 seeds x 5 characters, kept 10 iterations) and the history baseline.
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
LOG=$ROOT/logs/chain_td6.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
say "waiting for td2's 'eval it50' line"
until grep -q "^eval it50" "$ROOT/logs/m4_td2.log"; do sleep 20; done
sleep 5
kill -9 ${KILL_FIRST:?}; kill "${TRAINER_PID:?}"
for _ in $(seq 1 30); do pgrep -x SlayTheSpire2 >/dev/null || break; sleep 2; done
say "td2 stopped ($(grep '^eval it50' "$ROOT/logs/m4_td2.log" | cut -c1-80)); starting td6"
cd "$ROOT"
TAG=td6 INIT=unified_m4_long2.pt ITERS=150 \
EXTRA="--lam-step 0.5 --seed-prefix S6_ --run-critic --pool 26 --trainer-cpus 0,1,16,17 --eval-every 25 --seed-block 10 --block-seeds 20 --history-baseline" ./exp_m4.sh
say "td6 wrapper exited"
