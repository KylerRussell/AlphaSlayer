#!/usr/bin/env bash
# Round 2: teach the FIGHT model acts 2 and 3, then retrain full runs on top of it.
#
# Why this and not another reward sweep. Round 1 moved six reward/optimizer knobs across
# 35,561 runs and act-2 clears never left 1.3-1.9%. diag_act2.py then measured the fight win
# rate per act, which the training log cannot show because it reports one aggregate:
#
#          act 1                  act 2                 act 3
#   Monster 0.999 (1839)   Monster 0.920 (1134)   Monster 0.841 (44)
#   Elite   0.952 ( 227)   Elite   0.602 ( 123)   Elite   0.000 ( 4)
#   Boss    0.806 ( 314)   Boss    0.109 ( 110)
#
# The act-2 boss is won 11% of the time. In isolation, on act-2 encounters with Pandora's
# decks, the best combat policy scores 0.31 against 0.75-0.80 on the act-1 equivalent. The
# fight model was behaviour-cloned, PPO-trained and benchmarked entirely on act 1, and in a
# run it only meets act-2 fights in the ~1.5% of runs that get there, so no learning rate can
# fix it from run data alone. That is the chicken-and-egg the pooled round-1 numbers showed.
#
# This does NOT start runs in act 2. The user's objection stands: act 2 builds on act 1, so
# the RUN policy must still learn them in order. Only the COMBAT policy is taught act-2/3
# encounters, in isolation, via the probe's own --probe-act encounter selection. Act 1 is
# revisited every third block so the act-1 skill that already works is not forgotten; the
# act-1 benchmark is measured before and after to confirm that.
#
# Stage 1 ~3h: alternating-act combat curriculum, chaining checkpoints block to block.
# Stage 2 ~3h: 300-iteration full-run training from the curriculum checkpoint, same config as
#              round 1's best (comb), so its a2 is directly comparable.
set -uo pipefail
cd /home/kyler/Documents/AlphaSlayer
ROOT=/home/kyler/Documents/AlphaSlayer
PY=$ROOT/python/.venv7/bin/python
SCRATCH=/tmp/claude-1000/-home-kyler-Documents-AlphaSlayer/280902ad-4cb6-4df5-bb1f-4fae62457b81/scratchpad
VOCAB=$SCRATCH/vocab_from_ckpt.json
LOG=$ROOT/logs/exp_curriculum.log
BLOCK=${BLOCK:-25}
mkdir -p "$ROOT/logs"
export XDG_DATA_HOME="$(./headless_home.sh)"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 HSA_ENABLE_SDMA=0 PYTHONPATH=.

say() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

cd "$ROOT/python"
say "=== round 2 start; stage 1 = combat curriculum, block=$BLOCK iters ==="

# Act-1 benchmark BEFORE, so forgetting is measurable rather than assumed.
say "act-1 benchmark BEFORE curriculum:"
$PY measure_bench.py --combat-ckpt combat_comb.pt --bench combat_bench.json --envs 4 \
    --fights-per-deck 1 2>&1 | grep -viE 'amdgpu.ids|userwarning|self.encoder|warnings.warn' >> "$LOG"

SRC=combat_comb.pt
# 0-based acts: 1 = act 2, 2 = act 3. Weighted to the deficits (act 2 worst, then act 3),
# with act 1 revisited to guard against catastrophic forgetting.
PLAN=(1 1 2 0 1 1 2 0 1 1 2 0 1 1 2 0)
i=0
for ACT in "${PLAN[@]}"; do
  i=$((i+1))
  OUT=combat_curr.pt
  say "stage1 block $i/${#PLAN[@]}: act $((ACT+1)) (probe-act $ACT), $BLOCK iters, from $SRC"
  $PY train_rl.py --init "$SRC" --vocab "$VOCAB" --out "$OUT" \
      --iters "$BLOCK" --envs 20 --episodes-per-iter 200 \
      --act "$ACT" --encounters mix --mix 0.5,0.25,0.25 \
      --characters IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT \
      --relics LARGE_CAPSULE,PANDORAS_BOX --reroll-deck \
      --lr 5e-5 --ent-coef 0.01 --shaping 0.5 --hp-bonus 0.5 \
      2>&1 | grep -E '^iter' | tail -3 >> "$LOG"
  SRC="$OUT"
  pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 3
done

say "act-1 benchmark AFTER curriculum (checking for forgetting):"
$PY measure_bench.py --combat-ckpt combat_curr.pt --bench combat_bench.json --envs 4 \
    --fights-per-deck 1 2>&1 | grep -viE 'amdgpu.ids|userwarning|self.encoder|warnings.warn' >> "$LOG"

say "in-run fight rates per act, curriculum combat + round-1 run policy:"
$PY diag_act2.py --combat-ckpt combat_curr.pt --run-ckpt run_comb.pt --runs 300 --envs 20 \
    2>&1 | grep -viE 'amdgpu.ids|userwarning|self.encoder|warnings.warn|stalled;|closed before|last line' >> "$LOG"
pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 5

cd "$ROOT"
if [ ! -f python/combat_curr.pt ]; then
  say "!! stage 1 produced no combat_curr.pt; ABORTING before stage 2"
  exit 1
fi
say "=== stage 2: 300-iteration full-run training from the curriculum fight model ==="
# combat_curr.pt is BOTH the starting point and the KL reference: anchoring to combat_cs.pt
# would drag the fight model straight back to its act-1-only prior.
cp python/combat_curr.pt python/combat_curr_ref.pt
rm -f logs/curr.log python/run_curr.pt python/cards_curr.json
GAMMALO=0.99 GTF=51 COMBATLR=5e-5 COMBATKL=0.1 EXTRA=--gate-readonly \
  INIT_COMBAT=combat_curr.pt REF_COMBAT=combat_curr_ref.pt \
  TAG=curr TARGET=300 ./run_act2.sh
say "=== round 2 DONE: $(grep -c '^iter ' logs/curr.log 2>/dev/null) full-run iterations ==="
