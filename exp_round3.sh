#!/usr/bin/env bash
# Round 3: the round-2 curriculum, redone on REAL harvested decks at REAL entry health.
#
# Round 2 broke the plateau. Teaching the fight model act-2/3 encounters in isolation and then
# retraining full runs on top took act-2 clears 0.0140 -> 0.0559 (z +12.4) and act-3 wins
# 0.0020 -> 0.0095 (z +5.4) with act-1 unchanged, and both were still climbing at iteration
# 300 (final 50-iteration block: a2 0.079, a3 0.017).
#
# It did that DESPITE training on the wrong decks. Round 2's curriculum used Pandora's Box
# rerolls: 20+ random cards with two strong relics. A real act-2 entry is 17.4 cards, 4.4
# relics, 0.84 upgrades, at 35% hp. That mismatch is why the curriculum stage alone cost act-1
# skill (realistic act-1 benchmark 0.765 -> 0.680) and why its act-1 refresher blocks, also on
# Pandora's decks, failed to protect anything -- they scored 0.84-0.89 throughout while real
# act-1 play eroded underneath.
#
# So round 3 changes the training distribution, not the method:
#   * decks are HARVESTED from live runs and replayed exactly (deckserve), filtered per act
#   * fights start at the hp fraction the deck actually had entering that room (--real-hp,
#     needs the probe hp_frac support added 2026-09-18)
#   * a KL leash to the pre-round-3 fight model bounds drift instead of hoping refresher
#     blocks undo it
#   * the act-1 refresher now uses real act-1 decks, the distribution it must not lose
#
# Stage 0 ~40m  harvest entry decks with the round-2 policy pair
# Stage 1 ~2.5h alternating-act combat curriculum on those decks
# Stage 2 ~10m  act-1 benchmark + per-act in-run rates
# Stage 3 ~4h   300-iteration full-run training on top
set -uo pipefail
ROOT=/home/kyler/Documents/AlphaSlayer
PY=$ROOT/python/.venv7/bin/python
LOG=$ROOT/logs/exp_round3.log
HARVEST=$ROOT/python/harvest_r3.jsonl
CHARS=IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT
BLOCK=${BLOCK:-25}
mkdir -p "$ROOT/logs"
cd "$ROOT"
export XDG_DATA_HOME="$(./headless_home.sh)"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 HSA_ENABLE_SDMA=0 PYTHONPATH=.

say() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
cd "$ROOT/python"

say "=== round 3 start ==="

# ---- stage 0: harvest -----------------------------------------------------------------
# Continues training the round-2 pair while harvesting; the decks are what a live policy
# actually brings to each room, which is the entire point.
if [ ! -s "$HARVEST" ]; then
  say "stage 0: harvesting entry decks with run_curr.pt + combat_curr.pt"
  $PY train_run.py --iters "${HARVEST_ITERS:-50}" --envs 20 --runs-per-iter 20 --act-cap 3 \
    --characters "$CHARS" --ascension 0 \
    --combat-ckpt combat_curr.pt --combat-ref combat_curr.pt --run-ckpt run_curr.pt \
    --per-kind-heads --per-kind-loss --no-expert --gae-lambda 0.95 --dropout 0.1 \
    --combat-shaping 0.5 --progress-shaping 0.03 --floor-bonus 0 --act-bonus 1.0 \
    --out run_h3.pt --combat-out combat_h3.pt --card-stats cards_h3.json --critic-out critic_h3.pt \
    --gate-every 0 --combat-lr 5e-5 --combat-kl 0.1 --combat-lr-floor 2.5e-6 \
    --ent-coef 0.01 --hp-shaping 0.5 --deck-shaping 0.002 --deck-target 14 --skip-ent-coef 0.005 \
    --gamma-lo 0.99 --gamma-hi 0.999 --gamma-target-floors 51 \
    --max-steps-per-iter 14000 --collect-timeout 660 \
    --harvest-decks "$HARVEST" --harvest-rooms Boss,Elite,Monster --harvest-cap 20000 \
    2>&1 | grep -E '^iter|harvested' | tail -4 >> "$LOG"
  pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 5
else
  say "stage 0: reusing existing $HARVEST"
fi
$PY - <<'PYEOF' >> "$LOG" 2>&1
import json, collections
rows = [json.loads(l) for l in open("harvest_r3.jsonl") if l.strip()]
c = collections.Counter((r.get("act"), r.get("room")) for r in rows)
print(f"  harvest: {len(rows)} entry decks")
for k in sorted(c, key=lambda k: (k[0] is None, k)):
    print(f"    act {k[0]} {k[1]:8s} {c[k]:5d}")
PYEOF

# ---- stage 1: combat curriculum on real decks -----------------------------------------
SRC=combat_curr.pt
# 0-based acts. Act 2 is the deficit (boss 0.221 after round 2, vs 0.617 in act 1), act 3 is
# barely sampled in runs, and act 1 is revisited to hold the line -- this time on its own
# real deck distribution.
PLAN=(1 1 0 1 1 2 1 1 0 1 1 2)
i=0
for ACT in "${PLAN[@]}"; do
  i=$((i+1))
  say "stage1 block $i/${#PLAN[@]}: act $((ACT+1)), $BLOCK iters, from $SRC"
  $PY train_combat_decks.py --init "$SRC" --ref combat_curr.pt --kl-coef 0.05 \
      --decks "$HARVEST" --act "$ACT" --real-hp \
      --iters "$BLOCK" --envs 20 --fights-per-iter 200 --mix 0.5,0.25,0.25 \
      --lr 5e-5 --out combat_r3.pt \
      2>&1 | grep -E '^iter|decks for act|no usable' | tail -3 >> "$LOG"
  if [ -f combat_r3.pt ]; then SRC=combat_r3.pt; fi
  pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 3
done

if [ ! -f combat_r3.pt ]; then
  say "!! stage 1 produced no combat_r3.pt; ABORTING"
  exit 1
fi

# ---- stage 2: measure ------------------------------------------------------------------
say "act-1 realistic benchmark after round-3 curriculum:"
$PY measure_bench.py --combat-ckpt combat_r3.pt --bench combat_bench.json --envs 4 \
    --fights-per-deck 1 2>&1 | grep -viE 'amdgpu.ids|userwarning|self.encoder|warnings.warn' >> "$LOG"
say "in-run per-act fight rates, round-3 combat + round-2 run policy:"
$PY diag_act2.py --combat-ckpt combat_r3.pt --run-ckpt run_curr.pt --runs 300 --envs 20 \
    2>&1 | grep -viE 'amdgpu.ids|userwarning|self.encoder|warnings.warn|stalled;|closed before|last line' >> "$LOG"
pkill -9 -x SlayTheSpire2 2>/dev/null; sleep 5

# ---- stage 3: full-run training on top -------------------------------------------------
cd "$ROOT"
say "=== stage 3: 300-iteration full-run training from the round-3 fight model ==="
cp python/combat_r3.pt python/combat_r3_ref.pt
rm -f logs/r3.log python/run_r3.pt python/cards_r3.json
# Continues the RUN policy from round 2 rather than restarting it: the goal here is maximum
# clear rate, and the r3 curve is read against curr's final blocks.
GAMMALO=0.99 GTF=51 COMBATLR=5e-5 COMBATKL=0.1 EXTRA=--gate-readonly \
  INIT_RUN=run_curr.pt INIT_COMBAT=combat_r3.pt REF_COMBAT=combat_r3_ref.pt \
  TAG=r3 TARGET=300 ./run_act2.sh
say "=== round 3 DONE: $(grep -c '^iter ' logs/r3.log 2>/dev/null) full-run iterations ==="
