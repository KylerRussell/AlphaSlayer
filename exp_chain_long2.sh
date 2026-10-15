#!/usr/bin/env bash
# Waits for long1's wrapper to finish, installs the rebuilt probe DLL (serialised Decide), then
# starts long2: long1's exact config and seeds (paired A/B) plus the two fixes for the run-decision
# drift seen in long1 -- per-kind advantage centring and a 5x anchor KL on run decisions.
#
#   WAIT_PID=<long1 exp_m4.sh pid> setsid nohup ./exp_chain_long2.sh &
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
LOG=$ROOT/logs/chain_long2.log
GAME="$HOME/.local/share/Steam/steamapps/common/Slay the Spire 2"
DLL=$ROOT/probe/bin/Release/alphaslayer_probe.dll
DST="$GAME/mods/alphaslayer_probe/alphaslayer_probe.dll"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

say "waiting for pid ${WAIT_PID:?} (long1 wrapper)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
say "long1 wrapper exited; tail of its log:"
grep -E "^iter +[0-9]+/|^=== " "$ROOT/logs/m4_long1.log" | tail -3 >> "$LOG"

# The wrapper kills its games on exit; never swap the DLL under a live one.
for _ in $(seq 1 60); do pgrep -x SlayTheSpire2 >/dev/null || break; sleep 10; done
if pgrep -x SlayTheSpire2 >/dev/null; then
  say "game processes still running after 10 min; NOT installing the DLL or starting long2"
  exit 1
fi

# Atomic replace: write beside it, then rename (a process mapping the old file keeps its inode).
cp "$DLL" "$DST.new" && mv -f "$DST.new" "$DST"
if cmp -s "$DLL" "$DST"; then say "probe DLL installed ($(stat -c %s "$DST") bytes)"
else say "DLL install FAILED; not starting long2"; exit 1; fi

say "starting long2"
cd "$ROOT"
TAG=long2 INIT=unified_m4_shaped4.pt ITERS=150 \
EXTRA="--anchor unified_m3_d512_121.pt --ent-coef 0.001 --kl-anneal 1000000 --lr 1.5e-5 --kind-weight flat --runs-per-env 4 --curriculum-every 5 --curriculum-fights 120 --seed-prefix L1_ --adv-center kind --kl-run-mult 5" \
  ./exp_m4.sh
say "long2 wrapper exited"
