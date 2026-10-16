#!/usr/bin/env bash
# M4: PPO for the unified model, with auto-resume. A crash (GPU reset, stalled games) costs at
# most one iteration: train_unified.py checkpoints every iteration and --resume continues.
#
#   TAG=smoke ITERS=50 ./exp_m4.sh
#   TAG=a1 INIT=unified_m3_d256_222.pt ITERS=100 ./exp_m4.sh     # the 7M A/B arm
#
# Extra train_unified.py flags pass through EXTRA, e.g. EXTRA="--loop-policy adaptive".
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
PY=$ROOT/python/.venv7/bin/python
TAG=${TAG:-smoke}
INIT=${INIT:-unified_m3_d512_121.pt}
ITERS=${ITERS:-50}
EXTRA=${EXTRA:-}
LOG=$ROOT/logs/m4_$TAG.log
OUT=unified_m4_$TAG.pt
cd "$ROOT/python"
export XDG_DATA_HOME="$("$ROOT/headless_home.sh" 2>/dev/null)"
# GPU 1 = the second 7900 XT (PCI 11:00.0, no displays), the default; 0 drives the monitors
# (03:00.0) and train_unified refuses it (alphaslayer/gpu_guard.py: 4 desktop-killing resets).
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=${GPU:-1} HSA_ENABLE_SDMA=0 PYTHONPATH=.

done_iters() {
  $PY - "$OUT" <<'EOF' 2>/dev/null || echo 0
import sys, torch
try:
    print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["iter"])
except Exception:
    print(0)
EOF
}

for attempt in $(seq 1 30); do
  d=$(done_iters)
  if [ "$d" -ge "$ITERS" ]; then echo "=== TARGET REACHED: $d/$ITERS ===" >> "$LOG"; break; fi
  echo "=== attempt $attempt: $d/$ITERS done, init=$INIT extra='$EXTRA' ($(date '+%m-%d %H:%M')) ===" >> "$LOG"
  $PY train_unified.py --init "$INIT" --out "$OUT" --resume --iters "$ITERS" --bf16 $EXTRA \
      >> "$LOG" 2>&1
  code=$?
  echo "=== attempt $attempt exited code=$code at $(date '+%m-%d %H:%M') ===" >> "$LOG"
  # Only probe-launched games, and only when no other trainer (another arm) is still running:
  # a bare pkill killed the other arm's games and would kill a game Kyler is playing.
  # POOL_PATTERN narrows it to this arm's game copy (e.g. sts2_testgame3/SlayTheSpire2) so an
  # eval running beside it on another copy keeps its games.
  pgrep -f "[t]rain_unified.py" >/dev/null || pkill -9 -f -- "${POOL_PATTERN:---probe=runserve}" 2>/dev/null
  [ "$code" -eq 0 ] && continue
  sleep 30
done
echo "=== WRAPPER DONE ===" >> "$LOG"
