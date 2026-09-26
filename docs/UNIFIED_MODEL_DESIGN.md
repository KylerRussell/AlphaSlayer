# Unified model: design

Status: proposal, 2026-09-26. Replaces the separate CombatNet + RunNet with one network, trained
as one policy over one run-long trajectory.

## Why

Two goals, and what each actually requires:

| goal | requires |
|---|---|
| "I like playing this card, so I take it in the shop" | shared card/relic/power embeddings and a shared trunk |
| set up Pen Nib, save potions, spend HP only when it is cheap | the FIGHT's reward must include what the fight leaves behind for the rest of the run |

The second one is not delivered by sharing weights. Today a fight is scored as
`won + 0.5 * hp_end/max_hp` for that fight alone, so a merged network trained on that reward
still has no reason to care about the next fight. The fix is one value function over the whole
run: a fight ends at `V(state after the fight)`, or at the run's death value if it was lost.
Potions, HP, relic counters and exhausted resources are then priced by what they are worth later.

## Architecture

One transformer over entity tokens. The decision is made by scoring candidate tokens.

```
tokens = [CLS] [CONTEXT] [DECISION-KIND]
         + entity tokens   (depends on the decision; see below)
         + candidate tokens (one per legal action)
trunk  = prelude 1 / core 2 / coda 1 at d=512 (18.5M), core loopable; tokens packed per row
heads  = policy logit per candidate token
         value V(s) from CLS               -- ONE run-level value for every decision
         aux: P(win this fight), hp_delta  -- combat only, dense supervision (kept from CombatNet)
```

**Shared entity encoders.** They are used identically in combat and run decisions:

- card: id embedding + upgrade + enchantment/affliction (id, amount) + type/rarity/target/cost +
  **card numbers** (the 12 preview slots from the probe), symlog-scaled
- relic: id + counter + melted
- potion, power (id + symlog amount), monster, orb (id + passive/evoke), map point (type, row,
  col, visited, travelable), event option (hashed key, as now), rest option

**Context token.** hp/max_hp, max_hp, gold, energy, stars, act, act_floor, total_floor,
turn/round, room type, **the act boss's identity**, deck size, draw/discard/exhaust counts.
Numbers are symlog-scaled. Enemy HP, block and intent damage are also given relative to the
player's max HP. The raw integers the current CombatNet receives (Queen at 400 HP, 99-stack
debuffs) are what this fixes.

**Entity tokens per decision.**

- combat: hand, enemies (powers pooled per enemy, as now), allies (Osty), orbs, player powers,
  relics, potions, and the draw/discard/exhaust piles as multisets (below)
- run decisions: every deck card as its own token (synergy is attention between cards; the
  current mean-pooled bag cannot represent it), relics, potions; map point tokens for travel

**Piles as multisets, never ordered.** The draw pile's order is hidden information. Each distinct
(card, upgrade) in a pile becomes one token with a count, sorted by id, so the order the probe
happens to iterate in cannot leak. Upgrades are kept; today's bags drop them.

**Candidates as tokens (the main change to the heads).** Each legal action becomes a token:
kind + the card/potion/relic/option embedding + its numbers (for a play, the damage/block
**against that target**) + its target's entity embedding + per-kind numeric features (cost,
affordability, map lookahead...). Candidates attend to the whole state, so a card reward option
can look at the deck it would join. That is the synergy question a dot product against one
pooled state vector cannot ask. Cost: longer sequences (up to ~150 tokens), which is small at
d=256.

**Recurrent depth: wide core, loops where they pay.** The trunk is prelude 1 / core 2 / coda 1
at d=512; the core can be re-applied K times with the prelude output re-injected each pass. K
can be set per call or per row, and readouts are available after every pass. Under PPO the K
used at rollout is recorded and replayed in the update, or the importance ratio breaks, exactly
as train-mode dropout did. See "Width vs depth vs loops" below.

**Size: measured on the 7900 XT (`python/bench_unified_size.py`, real decisions, packed tokens).**

| config | params | infer 20 / 64 decisions | train step, 512, bf16 | bf16 peak |
|---|---|---|---|---|
| d256 x 6 | 7M | 5 / 15 ms | 132 ms | 4.2 GB |
| d384 x 8 | 18M | 14 / 47 ms | 246 ms | 7.6 GB |
| d512 x 8 | 31M | 22 / 76 ms | 332 ms | 9.9 GB |

Memory is not the constraint. Training cost is small at every size: about 33 minibatches per
iteration, so 4-11 s against about 28 s of collection. The binding costs are **rollout latency**
(the game processes wait on every forward) and, from M8, **evaluations per search**, which
multiply the inference cost. Past ~30M, the model buys capacity that ~10^7 decisions per day
cannot fill, at a latency that search pays many times over.

**Width vs depth vs loops** (same benchmark, prelude/core/coda x loops):

| config | effective depth | params | infer 20 | train bf16 |
|---|---|---|---|---|
| d384 2/4/2 x1 | 8 | 18.1M | 14.0 ms | 245 ms |
| d512 1/2/1 x1 | 4 | 18.5M | 11.8 ms | 179 ms |
| d512 1/2/1 x3 | 8 | 18.5M | 21.5 ms | 330 ms |
| d640 1/2/1 x3 | 8 | 28.0M | 32.4 ms | 420 ms |

Looped models match same-depth plain stacks on multi-step reasoning (Saunshi et al., ICLR 2025),
so unique parameters are better spent on width. But **a loop costs the same compute as the
depth it replaces**. Here compute, not parameters, is the binding budget, and search will
multiply it.

**Decision:** wide and shallow, **d512, 1/2/1 (18.5M)**, one pass by default: the fastest
configuration measured, with the same parameters as the deep stack. Loops are where extra
compute goes when a decision warrants it:

- **Training (M3 and M4) samples the loop count** from a fixed prior (1-4). That is not how
  compute is allocated; it is what makes every depth give a usable answer. Adaptive stopping
  depends on it: confidence readouts on a trajectory trained this way beat learned halting gates,
  reaching 95-99% accuracy at 1.4-1.6 average loops (arXiv 2607.20519).
- **Allocation is adaptive: uncertainty x stakes** (`net.choose_loops`). A decision stops looping
  at the first pass where it is settled (the top two candidates are clearly apart, or the policy
  stopped moving since the last pass). It never runs past a cap set by the STAKES: the model's
  own predicted risk of losing this fight, with per-encounter EMA loss rates as the cap until the
  risk head is calibrated. Uncertainty keeps obvious plays and lost causes cheap; stakes keep
  hallway fights cheap.
- **PPO records the loop count of every decision** and replays it in the update (`loop_rows`).
  That keeps the importance ratio a policy ratio under ANY allocation rule, adaptive included,
  so the rule can change without touching the trainer.
- **The M3 sweep** compares, at equal parameters: d384 2/4/2 x1, d512 1/2/1 x1, d512 1/2/1 x3,
  plus d256 (7M) as the small baseline. It compares held-out agreement, value calibration and
  play strength per boss, and keeps the cheapest within noise of the best.

Train in bf16 autocast (2-3x faster, ~40% less memory; logits, value and losses in fp32), after
a stability soak given this card's history of ring-timeout resets. Backprop through loops stores
activations per pass (15 GB fp32 at x3); if memory binds, truncate backprop to the last passes,
as recurrent-depth training does.

## Probe additions (M1)

Only what the architecture above needs and does not already have:

1. **Act boss encounter id** in every run observation (`ActModel.BossEncounter`), plus an
   Encounters vocab (a new table; no existing index moves).
2. **Pile multisets with upgrades** in the combat observation, emitted sorted.
3. **Run context in the combat observation:** total_floor, act_floor, gold (act and room type
   are already there).
4. **Card numbers on offered cards** (card reward, shop, card_select). Out of combat the preview
   has no global hooks, so these are base values. That is still the information a player
   reads off the card.
5. **Harvest records gain** floor, gold and potions, so the curriculum can build the post-fight
   run state it needs for bootstrapping (M4).
6. From `CAPABILITIES.md` (Enablers): card **keywords**; `gold` and `max_hp` card-number slots;
   the enemy's **next move id** and non-attack intent numbers; **potion and relic numbers**;
   **event option effects** (the event's dynamic vars attached to the options that use them).

Each gets a live delivery test in the style of `tests/test_combat_obs.py`.

## Warm start: distil, then RL (M3)

The short-term cost is contained by imitation before any RL:

1. Play ~2,000 runs with the r5 teachers (combat_r5 greedy, run_r5 sampled) on the new probe.
   Record every decision's obs, legal set and teacher distribution, plus MC returns.
2. Train the unified model on KL(teacher || student) over all decisions, plus value regression.
   A held-out split BY RUN, not by step: steps within a run are correlated, and the old BC
   attempt reported 0.88 training agreement against 0.33 on real play because it had no
   held-out split at all.
3. Play-test the student alone on paired seeds against the teachers.

**Acceptance:** act-1 clear rate within 2 SE of the teachers on the same 500 seeds. The
student gets inputs the teachers never had (boss id, act/floor, card numbers, orbs), so it
may match more easily than a like-for-like copy would. Embeddings are initialised from
CombatNet where the tables line up (cards, relics, powers, monsters, potions).

## Training (M4): `train_unified.py`

A NEW trainer. `train_run.py` stays as it is, as the baseline that every comparison runs
against.

- **One trajectory per run**, combat and run decisions interleaved in the order they happened.
- **Time-aware discount.** gamma applies per FLOOR, not per decision: 1.0 between steps of the
  same floor, gamma at each floor transition. A run holds ~500 combat decisions, and a per-step
  gamma of 0.99 would shrink the value of winning to nothing across one act.
- **GAE(lambda=0.95) per decision** with one value head. A fight's last step bootstraps from
  V(next state), or reaches the terminal if the run died there.
- **Rewards:** winning the run is the ONLY reward, so V(s) = P(win | s) (see
  `CAPABILITIES.md` section 0). Act bonus, terminal hp and progress shaping are dropped as rewards;
  act clears, fight wins and hp loss become auxiliary PREDICTION heads. Potential-based hp shaping
  on run steps and the combat potential on fight steps stay, since they move credit without
  changing the optimal policy.
- **Per-group averaging** of the surrogate and the entropy. Fights and each run decision kind
  count equally. Combat is ~80% of steps and would otherwise own the gradient.
- **Combat curriculum** stays: isolated act-2/3 fights on harvested decks at real hp. A won
  fight's terminal is V(post-fight run state rebuilt from the harvest record + hp_end); a lost
  fight's is the death value. Same objective as in-run fights, so the curriculum cannot pull
  the policy toward a different target.
- **Anchor:** a KL term toward the distilled checkpoint, annealed to 0 over the first ~100
  iterations. The old combat KL leash and rollback gate go away; the realistic benchmark is still
  MEASURED every 25 iterations and logged per boss.
- **Evaluation:** a fixed set of 500 seeds (100 per character), reproducible now that seeding
  is stable. Report per-act clears, conditional clears and **per-boss win rates**; paired
  seeds for every A/B.

Unit tests for the buffer math on synthetic trajectories: a hallway fight that spends a potion
must get a lower bootstrapped return than the same fight without it; time-aware discounting;
GAE across the fight/run boundary; per-group averaging weights.

## Milestones

| | deliverable | acceptance |
|---|---|---|
| M1 | probe additions above | **DONE 2026-09-26.** Live delivery tests pass and fail on the old DLL; no vocab index moved; no game side effects |
| M2 | encoder + model | **DONE 2026-09-26.** `alphaslayer/unified/`, `tests/test_unified.py`: 45 delivery checks (every M1 field changes the output), reference alignment, order invariance, batch isolation, loops; four reintroduced bugs all caught |
| M3 | distillation | held-out agreement reported; act-1 within 2 SE of the teachers on paired seeds |
| M4 | `train_unified.py` | win-only reward + aux heads; V(post) fight terminals; buffer-math tests; 50-iteration smoke run |
| M5 | capability eval suite | every eval in `CAPABILITIES.md` runs on a checkpoint and on r5, with a report |
| M6 | training round | per-boss, per-act and per-capability comparison against continuing r5, on paired seeds |
| M7 | determinism race fix | the "mix" side-effect set recorded 5 times is identical every time |
| M8 | intra-turn search at play time | exact within-turn; shuffles sampled, never the real order; per-boss gain vs M6 |
| M9 | search as teacher | Expert Iteration / Gumbel policy improvement distilled into the net |
| M10 | text embeddings (optional) | measurable gain on rarely seen items (E4) |

`CAPABILITIES.md` maps every capability onto these milestones.

Open issue that blocks replay search (step 3), not this: the serve-mode determinism race after a
player death (see memory: probe-determinism-status).
