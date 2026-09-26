# AlphaSlayer — Python side

Reads the packed episode format written by the C# probe, encodes it, and trains the combat net.

## Setup

System Python is 3.14, which has no PyTorch ROCm wheels. The venv pins 3.12:

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python numpy
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/rocm6.3

Verified working: `torch 2.9.1+rocm6.3`, ROCm 7.2.4, device 0 = `gfx1100` (7900 XT, 21.5GB,
42 CUs). Device 1 is the 9950X3D iGPU — **pin `HIP_VISIBLE_DEVICES=0`** so nothing lands on it.

    HIP_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python train_bc.py --epochs 8

## Layout

- `alphaslayer/format.py` — reader for the gzip binary episode format. Mirrors
  `probe/src/EpisodeWriter.cs`; change both together and bump `FORMAT_VERSION`.
- `alphaslayer/encoder.py` — ragged records to padded batch arrays (numpy, framework-free).
- `alphaslayer/model.py` — the combat net.
- `train_bc.py` — behaviour cloning warm start.
- `tests/test_roundtrip.py` — cross-validates binary against JSONL, field by field.

## The round-trip test earns its keep

It caught a real bug: the binary writer serialised state *after* the action executed, so every
observation was the successor state rather than what the agent saw when it chose. Loss curves
would have looked fine. `EpisodeWriter` is now split into `BeginStep` (before the action) and
`EndStep(hp_delta)` (after), since hp_delta is the last field and the stream is sequential.

Regenerate the fixtures and re-run after any format change:

    ./run.sh episodes --probe-episodes 40 --probe-format jsonl
    cp /tmp/alphaslayer_probe/episodes.jsonl /tmp/alphaslayer_probe/cmp.jsonl
    ./run.sh episodes --probe-episodes 40
    PYTHONPATH=. python3 tests/test_roundtrip.py

## Model

**6.51M params** at `d_model=256, layers=6`. Deliberately small: nothing here is
capacity-bound — a few hundred cards and relics fit in far less. Parameters buy nothing that
search and sample count do not.

- **Set encoder, no positional encoding.** Hand, enemies, relics, powers are sets. The draw
  pile is a multiset the player cannot order, so it enters as a bag-of-counts token.
- **Action-embedding policy head.** The legal set is variable (card x target), so candidates
  are scored as `<query, action_emb>/sqrt(d)`, with the action embedding composed from its
  hand-card token + target enemy token + kind. No combinatorial output layer, and targeting
  and a changing hand come for free.
- **Auxiliary heads** (hp_delta, final HP) give dense per-step supervision on the same trunk;
  one win/loss bit per combat is a thin signal on its own.

## BC results (500 greedy episodes, 15,469 steps)

| | |
|---|---|
| uniform-policy CE baseline | 1.621 (mean 5.1 legal actions) |
| val CE after 8 epochs | 1.069 |
| val action accuracy | ~0.55–0.60 |
| value head BCE | 0.180 -> 0.087 (base-rate ~0.49) |

~6.5s/epoch on the 7900 XT.

**Read the policy numbers with suspicion.** The demonstrator plays *the first playable card*,
and among several playable cards that choice is essentially arbitrary — so a large part of the
BC target is noise no model can fit. CE plateaus near 1.0 for that reason, not because the
network is at capacity. The value head is the honest signal here: 0.087 BCE against a 0.49
base-rate is real learning about which positions are won.

The fix is not a bigger model, it is a better target: search-improved policy targets from
MCTS, or a stronger scripted demonstrator. BC caps at the demonstrator by construction — it is
a search prior and a warm value net, not the product.

## Live evaluation — batched, across many seeds

`evaluate.py --envs N` runs N game processes against one batched policy. Each env gets its own
seed, so this is also the seed sweep that a single-process eval could not give.

480 episodes over 24 seeds, Ironclad, act-1 normals:

| policy | win rate | mean end HP | steps/s |
|---|---|---|---|
| random | 0.298 | 6.5 | 2214 |
| greedy (the BC demonstrator) | **0.835** | 28.3 | 2524 |
| BC (6.5M params) | 0.790 | 26.4 | 1169 |

### Correction to the earlier single-seed result

An earlier run on ONE fixed seed gave random 0.350 / greedy 0.850 / BC 0.830, and I called BC
and greedy "statistically indistinguishable". Broader seed coverage does not support that:

- BC trails greedy by **4.5pp** (0.790 vs 0.835), not 2pp. On n=480 that is roughly 1.8 sigma
  unpaired - suggestive rather than conclusive, but consistently in one direction.
- The single seed set was also **easy**: random scored 0.350 there versus 0.298 across 24 seeds.

So BC has got most of the way to its demonstrator without quite matching it. The conclusion
that BC is near its ceiling stands; the claim that it had *reached* it does not.

## Throughput: 20 -> ~1000 steps/s

| configuration | steps/s |
|---|---|
| single env, before | 20 |
| single env, Nagle disabled | 412 |
| 8 envs batched | 626 |
| **24 envs batched (BC net)** | **~1000-1170** |
| 24 envs, scripted policy (no GPU) | ~2500 |

**The dominant fix was `TCP_NODELAY`, not batching.** The protocol is one small request and one
small response per decision, which is exactly the pattern Nagle's algorithm plus delayed ACK
punishes with ~40ms stalls - capping a single env near 20 steps/s regardless of how fast the
engine or the policy runs. Disabling it took one env from 20 to 412 steps/s, a 20x win on its
own. Batching then adds parallelism on top.

Scaling peaks around **24 envs** on this 32-thread machine (32 envs regresses to 757 steps/s on
CPU contention). Mean batch at 24 envs is ~18.

At ~1170 steps/s, 1e8 steps is about **24 hours** rather than the 58 days the original bridge
implied. With a scripted policy the env alone sustains ~2500 steps/s, so at 24 envs the 6.5M
net is now the throughput limit, not the game.

## RL (PPO) — training curve

`train_rl.py` warm-starts from the BC checkpoint and runs PPO against the live game through
the vectorised bridge.

Design choices worth knowing:

- **PPO, not search.** The blocker for MCTS is mid-combat state snapshot/restore, which the
  engine does not provide. PPO needs none of it, handles the game's mid-turn randomness
  natively, and *can* exceed the demonstrator - which behaviour cloning cannot, by construction.
- **Monte Carlo returns, not GAE.** Combat episodes are short (~8 turns, ~25 decisions) with a
  single terminal reward, so bootstrapping buys little and MC keeps credit assignment unbiased.
- **Reward = won + 0.5 * (hp_end / max_hp).** Pure win/loss is a thin signal when the
  demonstrator already wins ~84% of the time; end HP is the margin that separates a scrape
  from a clean win, and it is what actually carries over in a real run.
- **Fresh seeds every iteration** (`RL{iter}_{env}`), so the policy cannot memorise a fixed
  seed set. Over 30 iterations that is ~720 distinct seeds.
- Steps from an episode that never terminated are **dropped**, not trained on: they have no
  reward, and crediting them with zero would poison the value target.

30 iterations, 240 episodes each, 24 envs (~20s/iter: ~13s collect, ~7s train):

| iter | win (sampled, train seeds) | mean end HP | entropy |
|---|---|---|---|
| 1 | 0.808 | 25.4 | 1.08 |
| 4 | 0.971 | 33.0 | 1.01 |
| 7 | 0.996 | 39.1 | 0.89 |
| 9 | **1.000** | 41.9 | 0.83 |
| 20 | 1.000 | 47.6 | 0.66 |
| 30 | 1.000 | 49.2 | 0.65 |

Win rate saturates by iteration 9; **end HP keeps climbing to iteration 30** (25.4 -> 49.2,
roughly doubled). That is the reward shaping doing its job: once winning is solved, the only
remaining gradient is winning *more cheaply*. Entropy falls 1.08 -> 0.65, so the policy is
sharpening rather than collapsing.

## Scope: this is COMBAT training, not full runs

Worth being explicit, because it changes what these numbers mean. Everything here trains and
evaluates **isolated act-1 combats**:

- a fixed starting deck, healed to full between fights
- no map, no path choice, no shops, rest sites, events or card rewards
- no deck evolution, no relic accumulation, no HP carryover between fights
- the value head predicts "win *this combat*", not "win the run"

The stated goal — beat humans on percentage of *games* won — needs the run layer too: the
original two-network design (combat net + run net, the latter firing ~40x per run over map,
rewards, shops, campfires, boss relics). None of that exists yet. What does exist is the
harder half of the environment, and a combat agent that is a prerequisite for any run agent.

## Next

1. In-process inference so search does not cross the Python/FFI boundary per node (ONNX
   export or a small C# inference path) — the classic AlphaZero-at-home performance killer.
2. Intra-turn exact search: no RNG fires until end-of-turn draw, so card-play ordering within
   a turn is deterministic and can be searched exactly, with chance nodes only at turn
   boundaries.
3. Self-play loop with search-improved policy targets.
