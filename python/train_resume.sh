#!/usr/bin/env bash
# Runs train_rl.py to a target iteration count, resuming after ROCm driver aborts.
#
# The HSA_STATUS_ERROR_EXCEPTION faults are environmental, not ours: the identical workload
# runs clean on CPU, and upgrading torch 2.9.1+rocm6.3 -> 2.10.0+rocm7.0 cut the rate sharply
# without eliminating it. train_rl.py checkpoints every iteration, so a crash costs at most
# one iteration's work.
#
#   ./train_resume.sh <total_iters> <out.pt> <init.pt> [extra train_rl.py args...]
set -uo pipefail
TOTAL=$1; OUT=$2; INIT=$3; shift 3
LOG=/tmp/$(basename "$OUT" .pt)_resume.log
: > "$LOG"
done_iters=0
attempt=0
while [ "$done_iters" -lt "$TOTAL" ]; do
  attempt=$((attempt+1))
  remaining=$((TOTAL - done_iters))
  src="$INIT"; [ -f "$OUT" ] && [ "$done_iters" -gt 0 ] && src="$OUT"
  echo "=== attempt $attempt: $done_iters/$TOTAL done, running $remaining more from $src ===" | tee -a "$LOG"
  HIP_VISIBLE_DEVICES=0 PYTHONPATH=. .venv7/bin/python train_rl.py \
      --init "$src" --out "$OUT" --iters "$remaining" "$@" 2>&1 | tee -a "$LOG" \
      | grep -E "^iter" | tail -1
  got=$(grep -cE "^iter" "$LOG")
  if [ "$got" -le "$done_iters" ]; then
    echo "!! no progress on attempt $attempt; stopping to avoid a spin loop" | tee -a "$LOG"
    break
  fi
  done_iters=$got
done
echo "=== finished: $done_iters/$TOTAL iterations over $attempt attempt(s), $(grep -c HSA_STATUS "$LOG") fault(s) ===" | tee -a "$LOG"
