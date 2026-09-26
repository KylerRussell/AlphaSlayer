#!/usr/bin/env bash
# Full-run (3 act) training with auto-resume.
#
# Every knob an experiment varies is an environment variable with the historical value as its
# default, so one experiment differs from the control in exactly one place and the log header
# records which. See exp_plateau.sh for the sweep that uses them.
#
# A GPU ring timeout took the machine down 3 minutes into the first attempt. Checkpoints are
# written every iteration, so a crash should cost ONE iteration, not a 24-hour run. This
# restarts from the last checkpoint until the target iteration count is reached.
#
# --act-bonus is 1.0, NOT 0. Converting it to a potential was right under --act-cap 1, where
# clearing act 1 WAS the terminal win and paid +3.0 anyway. Under --act-cap 2 the win bonus
# fires on ~0.1% of runs, so with act-bonus 0 the only thing separating "cleared act 1" from
# "died on floor 15" was 2 floors of progress potential = 0.06, against ~1.0 before. The
# policy drifted (a1 0.186 -> 0.119, fight_win 0.882 -> 0.864 over 50 iterations). An
# achievable intermediate milestone needs a reward the policy can actually feel.
#
# --combat-ref stays pinned to combat_cs.pt across restarts on purpose: re-anchoring the KL
# trust region and gate baseline to the resumed model each time would let the policy walk
# downhill indefinitely, one "no regression since last restart" at a time.
#
# .venv7 (torch 2.10 + ROCm 7.0), not .venv (2.9.1 + ROCm 6.3): train_resume.sh measured the
# newer build cutting the HSA fault rate sharply. HIP_VISIBLE_DEVICES=0 hides the 9950X3D iGPU
# from ROCm (python/README.md). HSA_ENABLE_SDMA=0 routes host<->device copies through blit
# kernels instead of the SDMA engines, the standard ROCm workaround for hangs on consumer RDNA
# cards; transfers here are tiny (0.4 GB VRAM, ~1k-step batches) so the cost is negligible.
# PYTHONHASHSEED=0: belt and braces for alphaslayer/runmodel.py::_tok, which once used the
# salted builtin hash and scrambled the run policy's token embeddings on every resume.
# Nothing user-side can rule out a reset entirely: the two 09-03 hangs were gfx-ring timeouts
# on desktop apps sharing the card, and the display lives on this GPU too.
#
# --per-kind-loss with --ent-coef 0.01: in the 94-iteration run with ent 0.05 and per-kind
# entropy but a flat-mean surrogate, the rare decision kinds drifted TO uniform (shop Hn
# 0.90 -> 0.95, potion_gate 0.75 -> 0.94, card_select 0.99) while travel sharpened to 0.08.
# The entropy term (~0.05) was an order of magnitude above the surrogate (~0.003). Weighting
# the surrogate per kind too, and shrinking the bonus, puts the two on the same footing.
cd /home/kyler/Documents/AlphaSlayer/python
# Private game data dir with every Steam Workshop mod disabled; see ../headless_home.sh for
# why. Without it the game loads the user's Workshop mods into the headless probe and every
# fight fails at setup.
export XDG_DATA_HOME="$(../headless_home.sh)"
# TAG names this run's outputs (run_$TAG.pt, combat_$TAG.pt, cards_$TAG.json, logs/$TAG.log).
# A tag whose log has no 'iter' lines starts fresh from run_cs.pt/combat_cs.pt.
TAG=${TAG:-a2}
# RUNS per iteration. The step cap and collection timeout scale with it: a run is ~220
# decisions (run + combat) and ~0.4s of wall clock per run at 20 envs, so 400 steps and 2s
# per run leave 2x headroom. The old fixed cap of 200000 would have truncated 1000-run
# iterations at ~900 runs.
# 20, not 100: measured on a 1000-vs-100 comparison, progress is paced by the number of
# PPO updates (each is clip-bounded), not by samples per update -- 1000 runs/iter reached
# a1 0.35 by iteration 5 where 100 runs/iter reached 0.49 on a tenth of the data. Fewer runs
# per update means more updates per hour, and the gradient corrects a bad habit sooner.
RUNS=${RUNS:-20}
# A 3-act run is ~350 decisions (run + combat) and the cap only truncates collection, so keep
# 2x headroom; the deadline is a wedge-detector, not a budget.
MAXSTEPS=$(( RUNS * 700 ))
CTIMEOUT=$(( RUNS * 3 + 600 ))
# ---- experiment knobs (defaults = the historical act-1+2 configuration) ----------------
ACTCAP=${ACTCAP:-3}          # 3 = the whole game; the game has 3 acts
# --gamma-lo / --gamma-target-floors together decide the discount, and the discount decides
# whether --progress-shaping rewards depth AT ALL. The shaping term is
# w*(gamma*phi' - phi) with phi = the raw floor number, so it only pays for advancing while
# floor < gamma/(1-gamma). Measured against the real Buffer.finish_run: at gamma 0.967 a run
# that clears act 2 on floor 34 collects LESS total reward (1.370) than one that dies on
# floor 20 (1.562), and the progress term alone peaks at floor 28 and falls away after.
# Depth-monotonic needs gamma >= ~0.983. The defaults below reproduce a3 (gamma settled at
# 0.980-0.986 once mean floors reached ~23, so a3 was just above the threshold and only
# inverted for its first ~25 iterations); the `gam` experiment raises the floor so the
# inversion cannot happen at any reachable depth.
GAMMALO=${GAMMALO:-0.95}
GTF=${GTF:-34}               # --gamma-target-floors
HPSHAPE=${HPSHAPE:-0.5}      # potential-based hp shaping weight
COMBATLR=${COMBATLR:-1e-5}
COMBATKL=${COMBATKL:-0.5}
DROPOUT=${DROPOUT:-0.1}
ENTCOEF=${ENTCOEF:-0.01}
EXTRA=${EXTRA:-}             # extra train_run.py flags, e.g. --gate-readonly
# Where a FRESH run starts from, and what the combat KL trust region is anchored to. The
# reference matters as much as the init: anchoring a curriculum-trained fight model back to
# combat_cs.pt would pull it straight back to its act-1-only prior, which is the thing the
# curriculum exists to undo.
INIT_RUN=${INIT_RUN:-run_cs.pt}
INIT_COMBAT=${INIT_COMBAT:-combat_cs.pt}
REF_COMBAT=${REF_COMBAT:-combat_cs.pt}
LOG=${LOG:-/home/kyler/Documents/AlphaSlayer/logs/$TAG.log}
TARGET=${TARGET:-1000}
REF=$REF_COMBAT
mkdir -p "$(dirname "$LOG")"

for attempt in $(seq 1 40); do
  done_iters=$(grep -c '^iter ' "$LOG" 2>/dev/null || echo 0)
  remaining=$(( TARGET - done_iters ))
  if [ "$remaining" -le 0 ]; then
    echo "=== TARGET REACHED: $done_iters/$TARGET iterations ===" >> "$LOG"; break
  fi
  if [ "$done_iters" -eq 0 ]; then
    RCK=$INIT_RUN; CCK=$INIT_COMBAT
  else
    RCK=run_$TAG.pt; CCK=combat_$TAG.pt     # resume from our own output
  fi
  echo "=== attempt $attempt: $done_iters done, $remaining to go, from $RCK/$CCK ($(date '+%m-%d %H:%M'))" >> "$LOG"
  echo "===   runs/iter=$RUNS act-cap=$ACTCAP hp-shaping=$HPSHAPE combat-lr=$COMBATLR combat-kl=$COMBATKL dropout=$DROPOUT ent-coef=$ENTCOEF gamma-lo=$GAMMALO gamma-target-floors=$GTF extra='$EXTRA' init=$INIT_RUN/$INIT_COMBAT ref=$REF ===" >> "$LOG"
  PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 HSA_ENABLE_SDMA=0 .venv7/bin/python train_run.py \
    --iters "$remaining" --envs 20 --runs-per-iter "$RUNS" --act-cap "$ACTCAP" \
    --characters IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT --ascension 0 \
    --combat-ckpt "$CCK" --combat-ref "$REF" --run-ckpt "$RCK" \
    --per-kind-heads --per-kind-loss --no-expert --gae-lambda 0.95 --dropout "$DROPOUT" \
    --combat-shaping 0.5 --progress-shaping 0.03 --floor-bonus 0 --act-bonus 1.0 \
    --out run_$TAG.pt --combat-out combat_$TAG.pt --card-stats cards_$TAG.json \
    --gate-every 25 --gate-bench combat_bench.json --gate-bench-fights 1 \
    --combat-lr "$COMBATLR" --combat-kl "$COMBATKL" --combat-lr-floor 2.5e-6 \
    --ent-coef "$ENTCOEF" --hp-shaping "$HPSHAPE" --deck-shaping 0.002 --deck-target 14 --skip-ent-coef 0.005 \
    --gamma-lo "$GAMMALO" --gamma-hi 0.999 --gamma-target-floors "$GTF" \
    --max-steps-per-iter "$MAXSTEPS" --collect-timeout "$CTIMEOUT" $EXTRA >> "$LOG" 2>&1
  code=$?
  echo "=== attempt $attempt exited code=$code at $(date '+%m-%d %H:%M') ===" >> "$LOG"
  [ "$code" -eq 0 ] && continue
  pkill -9 SlayTheSpire2 2>/dev/null; sleep 30      # clear orphaned games before retrying
done
echo "=== WRAPPER DONE ($(grep -c '^iter ' "$LOG") iters) ===" >> "$LOG"
