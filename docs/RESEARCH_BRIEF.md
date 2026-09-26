# AlphaSlayer — research brief

We are training an agent to win runs of **Slay the Spire 2** (v0.111.0, Godot 4.5.1, .NET 9).
Act-1 clear rate is currently ~37%. A competent human clears act 1 close to 100% of the time.
We want recommendations on how to close that gap, and on anything we appear to be doing wrong.

Hardware: single machine, AMD 7900XT (20GB VRAM, ROCm), 32 cores, 60GB RAM.

---

## 1. Environment

We run the **real game** headless rather than reimplementing its rules. The game ships a
first-class mod loader and an unobfuscated `sts2.dll`; we load a C# mod that drives the
model layer directly and bypasses the Godot scene graph (`TestMode.IsOn`, which the game's own
test suite uses). Every gameplay decision is exposed over TCP as newline-delimited JSON.

Throughput: ~1000 combat decisions/s across 16 processes; ~30 complete fights/s per process.

Two servers:
- `runserve` — a whole run. Decision kinds: `travel`, `card_reward`, `rest`, `event`, `shop`,
  `potion_gate`, `card_select`, `treasure`, `potion_ooc`, and `combat`.
- `deckserve` — plays an **arbitrary deck** (exact card list + upgrades + relics) against a
  chosen room type. Used for counterfactual deck evaluation.

Known engine issues we had to work around (all headless-only): the treasure-room relic grant
lives in the node layer; one event (Crystal Sphere) is a minigame with no model-layer
completion path; `Trial` and `SoulNexus` dereference the scene without null guards; JIT
inlining defeats Harmony patches on one-line property getters.

## 2. Models

**CombatNet** (6.6M params) — in-fight decisions. Transformer encoder (d=256, 6 layers, 8
heads) over hand / enemies / relics / powers / bag-of-piles tokens, plus an **action-embedding
head**: each legal action is encoded (kind, hand index, target, card id) and scored against the
state, so the action set can be variable-length. Actions: play card (with target), end turn,
select card (for in-combat prompts), use potion.

Trained by BC on random rollouts, then PPO. Best checkpoint scores **0.949 ± 0.013** on an
internal benchmark (mixed normal/elite/boss, Pandora's Box + Large Capsule random decks, A0).

**RunNet** (1.5M / 2.5M params) — all out-of-combat decisions. Same action-embedding idea:
state = scalars (hp, gold, deck size, upgrades, act, floor) + relic set (mean-pooled
embeddings) + deck as a weighted embedding bag; each candidate action encoded as (action kind,
hashed string id, card/relic/potion embedding, 26 numeric features incl. cost, affordability,
rarity, map point type, and a discounted **lookahead** over map point types reachable
downstream). Optional **per-decision-kind heads** on a shared trunk.

**Critic** (0.6M params) — predicts fraction of HP preserved per room type from
(deck, relics, character, act/floor/hp context). Mean **and max** pooling over the deck (a mean
alone cannot see that a deck holds one bomb).

## 3. Training

PPO, Monte Carlo returns, advantage = return − value baseline, clip 0.2, 2 epochs,
minibatch 512, AdamW.

**Two objectives, deliberately separate:**
- The **run policy** is credited from how the run ends.
- The **combat policy** keeps the objective it was trained on — `won + 0.5·hp_end/max_hp` for
  *that fight* — and is *not* retrained on run outcome. It trains at 30× lower LR under a KL
  penalty toward a frozen copy of its original weights, with a regression gate that
  re-evaluates it on the original benchmark every 10 iterations and rolls back if it drops
  more than `max(0.02, 2·SE)` below baseline.

**Run rewards** (paid where earned, then discounted backward):
- +0.03 per floor survived, +1.0 per act cleared, +3.0 win, +0.5·(final hp fraction)
- potential-based HP shaping `w·(γ·φ' − φ)`, φ = hp/max_hp, w=0.5
- potential-based deck-size cost, φ = −λ·max(0, |deck| − 12)², λ=0.002 (**convex**, so the
  marginal cost of a card rises with deck size)
- entropy bonus 0.05, averaged **per decision kind** then across kinds
- a small entropy bonus (0.005) on the binary take-vs-skip marginal of a card reward

**Discount schedule:** γ annealed 0.95 → 0.999 as mean floors approaches a target, on the
theory that early failures are caused by immediate blunders (credit should stay local) and
later ones by decisions made tens of floors earlier (credit must reach back).

## 4. Measurements

Fight win rates with **real run decks** (pooled ~2,950 fights, greedy combat policy):

| room type | win rate |
|---|---|
| boss | 0.507 ± 0.035 |
| elite | 0.714 ± 0.028 |
| monster | 0.946 ± 0.005 |

Ceiling implied for a **full 3-act run**, assuming a perfect router: `0.507³ · 0.714^ne · 0.946^11`
= **0.018–0.071** depending on elites taken. Observed full-run win rate: **0.002**.

Act-1-only run policy comparison (120 runs each, identical combat model, greedy):

| run policy | win | floors | boss | elite |
|---|---|---|---|---|
| trained net | 0.192 | 15.5 | 0.258 | 0.708 |
| scripted heuristic | 0.175 | 13.1 | **0.420** | 0.539 |
| random | 0.042 | 11.4 | 0.200 | 0.373 |

The heuristic beats the net **at the boss** while losing on floors and elites. It rests when
hurt (HEAL 64%) and buys card removals (63% of shop purchases); the net did neither.

## 5. What we have tried, and what happened

| attempt | result |
|---|---|
| Full 3-act training, ~1000 iterations across variants | win stuck at **0.002**; `acts` plateaued 0.30–0.43 |
| Potential-based HP shaping | broke a campfire collapse temporarily; policy re-collapsed to SMITH 100% by iter 100 |
| **Linear** deck-size cost + strong take/skip entropy | **failed** — forced a flat 67% skip at every deck size, decks starved to 13 cards, boss win 0.53→0.18 |
| **Convex** deck-size cost + weak entropy | **worked as designed** — first state-dependent card policy: skip 0% at ≤12 cards rising to 37% at >24 |
| Per-decision-kind entropy | fixed a real defect (travel is ~45% of decisions and dominated a flat entropy mean) |
| Per-decision-kind heads | no measurable effect on outcomes |
| Counterfactual expert (play both branches, ~90k fights) + distilled critic | critic learned well (MSE 0.105→0.038) but outcome difference vs baseline was **+0.006 ± 0.023 — inside noise** |
| Behaviour cloning from the heuristic | **failed** — greedy agreement on heuristic-visited states only 0.33 (rest), 0.56 (card reward) despite 0.88 reported training agreement (no held-out split; the metric was wrong) |
| **Act-1-only curriculum** | **worked** — win 0.002 → 0.191 → **0.378 over 70 iterations** and still climbing; `fight_win` 0.881 → 0.913 alongside, i.e. both policies improving together |

The single largest effect by far was shortening the horizon. Nothing about the architecture,
reward terms or hyperparameters changed — only that the win signal became observable
(~150× denser). Every shaping intervention before that was operating on a policy that could
not see the outcome it was being asked to optimise.

## 6. Where we think the problem is now

1. **Boss win rate is the dominant term.** `0.507³ = 0.13` would cap a 3-act run at ~13%
   before anything else.

   *Correction, added later:* we initially read this as a fixed ceiling and concluded act-1 was
   bounded near `0.507 · 0.946⁴ ≈ 0.41`. That was arithmetic on a snapshot. The combat policy
   trains alongside the run policy, and over the act-1 run its in-run `fight_win` moved
   0.881 → 0.913 while *also* improving on its original benchmark (gate: 0.946 vs 0.919
   baseline). The ceiling rises as the run policy approaches it, so the two compound rather
   than one capping the other. The bound is real only for a frozen combat policy.
2. The combat model has sat at ~0.89 fight / ~0.51 boss across every training run. It is
   anchored by its regression gate to a benchmark built on *Pandora's Box random decks*, not on
   the decks it actually plays in runs.
3. The run policy repeatedly collapses to a single action on rare decision kinds unless
   actively prevented.

## 7. Questions

1. Is **credit assignment** the right diagnosis for the collapse, or is the run policy's
   action encoding too weak to distinguish options (e.g. campfire agreement of 0.33 across
   3 options is barely above chance)?
2. Should the combat policy be **specialised per room type** (a boss head), or trained on a
   boss-heavy curriculum with realistic run decks, rather than gated to a fixed benchmark?
3. Is there a better approach than PPO + Monte Carlo returns for a **~40-decision episode with
   a single terminal bit**? (We have a fast, forkable simulator but **no mid-combat save**, so
   true MCTS over combat is expensive; run-level search over map/card choices is cheap.)
4. Our counterfactual expert measured the right quantity and changed nothing. Does that imply
   the run decisions genuinely matter less than we assume, or that ~3 fights/candidate is too
   noisy a label to act on?
5. What is the strongest known approach for **deck-building games** specifically — where the
   action space changes every turn, the deck is a multiset that the policy modifies, and the
   outcome depends on interactions between cards rather than on any card individually?
6. Are we wrong to train two separate policies? Would a single policy over both decision types,
   or a hierarchical option-based formulation, handle credit assignment better?
