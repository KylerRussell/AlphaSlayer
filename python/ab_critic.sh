#!/usr/bin/env bash
# A/B: does the counterfactual expert + critic actually improve runs?
#
# Combat is FROZEN in both arms. That is not how we intend to train, but it isolates the one
# variable under test: with the combat policy moving, a difference in acts/floors could just
# as easily come from the fighter improving as from better card choices.
#
# Both arms start from the same checkpoint, use the same seeds and the same run-env count.
set -u
COMMON="--iters 50 --envs 12 --runs-per-iter 48 \
  --characters IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT --ascension 0 \
  --combat-ckpt combat_run_cx.pt --run-ckpt run_policy_prefix.pt \
  --freeze-combat --gate-every 0 --per-kind-heads \
  --ent-coef 0.05 --hp-shaping 0.5 --deck-shaping 0.002 --deck-target 12 --skip-ent-coef 0.005 \
  --gamma-lo 0.95 --gamma-hi 0.999 --gamma-target-floors 45 \
  --max-steps-per-iter 60000 --collect-timeout 600"

echo "=== ARM A: no expert (baseline) ==="
.venv/bin/python train_run.py $COMMON --no-expert \
  --out /tmp/ab_A_run.pt --critic-out /tmp/ab_A_critic.pt --card-stats /tmp/ab_A_cards.json \
  > /tmp/ab_A.log 2>&1
echo "arm A done: $(grep -c '^iter' /tmp/ab_A.log) iters"

echo "=== ARM B: expert + critic ==="
.venv/bin/python train_run.py $COMMON \
  --critic-envs 4 --critic-fights 3 --critic-anneal-iters 60 --critic-follow-anneal 200 \
  --critic-warmup 300 --critic-states-per-iter 60 \
  --out /tmp/ab_B_run.pt --critic-out /tmp/ab_B_critic.pt --card-stats /tmp/ab_B_cards.json \
  > /tmp/ab_B.log 2>&1
echo "arm B done: $(grep -c '^iter' /tmp/ab_B.log) iters"
