#!/usr/bin/env bash
# Pauses td2, collects 20 iterations with each seed played 4x, measures gradient consistency
# with and without the same-seed baseline, then resumes td2 (eval every 25).
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
LOG=$ROOT/logs/grp_probe.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
n=$(grep -cE "^iter" "$ROOT/logs/m4_td2.log")
until [ "$(grep -cE '^iter' "$ROOT/logs/m4_td2.log")" -gt "$n" ]; do sleep 5; done
sleep 15
kill -9 ${KILL_FIRST:?}; kill "${TRAINER_PID:?}"
for _ in $(seq 1 30); do pgrep -x SlayTheSpire2 >/dev/null || break; sleep 2; done
say "td2 paused at $(grep -cE '^iter' "$ROOT/logs/m4_td2.log") iterations"
cd "$ROOT/python"
export XDG_DATA_HOME="$("$ROOT/headless_home.sh" 2>/dev/null)" PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=1 HSA_ENABLE_SDMA=0 PYTHONPATH=.
rm -rf data/rollouts_grp data/collect_grp.pt*
.venv7/bin/python train_unified.py --init unified_m4_td2.pt --out data/collect_grp.pt --iters 20 --bf16 \
    --collect-only --dump-steps data/rollouts_grp --seed-prefix G4_ --pool 26 --trainer-cpus 0,1,16,17 \
    --seed-groups 4 --run-critic --lam-step 0.5 2>&1 | grep -E "COLLECT|Traceback|Error" >> "$LOG"
say "collected; measuring"
.venv7/bin/python grad_snr.py --dumps data/rollouts_grp --ckpt unified_m4_td2.pt \
    --est "dumped,dumped+grp,hand(ls=.5),hand(ls=.5)+grp" --out data/gradsnr_grp.json 2>&1 \
    | grep -v amdgpu.ids | grep -v "  iter " >> "$LOG"
say "resuming td2"
cd "$ROOT"
TAG=td2 INIT=unified_m4_long2.pt ITERS=150 \
EXTRA="--lam-step 0.5 --seed-prefix D2_ --run-critic --pool 26 --trainer-cpus 0,1,16,17 --eval-every 25" ./exp_m4.sh
