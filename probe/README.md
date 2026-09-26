# AlphaSlayer probe

Headless feasibility probe for driving **Slay the Spire 2 v0.111.0** as an RL environment
("Option A": run the game's own rules engine headless instead of reimplementing it).

Built against the retail install. Nothing here reimplements game rules; it instruments them.

## Usage

```bash
./run.sh                                   # inventory only (safe, ~15s)
./run.sh inventory,snapshot                # + clone-cost measurement
./run.sh throughput --probe-combats 20     # combat throughput (CURRENTLY CRASHES, see below)
./run.sh throughput --probe-natural        # same, with the engine's real timing
```

Results land in `/tmp/alphaslayer_probe/{probe_report.json,probe.log}` (override with `PROBE_OUT`).

Uninstall: `rm -rf "$STS2_DIR/mods/alphaslayer_probe"`.

Flags: `--probe-combats N`, `--probe-turn-cap N`, `--probe-character IRONCLAD`,
`--probe-ascension N`, `--probe-seed S`, `--probe-snapshot-iters N`, `--probe-natural`,
`--probe-quit`.

## What it established

**Works.** The game boots `--headless` to a loaded main menu in ~13s (`VRAM=0B`, dummy
renderer). The mod loads through MegaCrit's own loader (`mods/<id>/<id>.dll` + manifest json);
no BepInEx needed, the game ships its own Harmony and MonoMod.

**Content census (live, from `ModelDb`)** — these are your embedding-table sizes:

| | |
|---|---|
| cards | 596 |
| relics | 299 |
| powers | 283 |
| potions | 66 |
| monsters | 107 |
| encounters | 85 |
| events | 57 |
| orbs | 4 |
| enchantments | 24 (22 live + `DEPRECATED_*` + `MOCK_*`) |
| playable characters | 5: Ironclad, Silent, Defect, Necrobinder, Regent |

**Acts.** 4 `ActModel`s but **3 acts per run**: index 0 has two variants — `OVERGROWTH`
(default) and `UNDERDOCKS` — while Hive (1) and Glory (2) are single. The Act-1 variant swaps
the encounter/boss/event pools, so **act identity is a state variable, not a constant**, and
it is unlock-gated via `UnderdocksEpoch`.

**Enchantments.** `CardModel.Enchantment` is a single nullable slot carrying a `decimal
Amount` and `EnchantmentStatus {Normal, Disabled}`, and it modifies replay count via
`EnchantPlayCount`. Card identity for the encoder is therefore
`(cardId, upgradeLevel, enchantmentId?, amount, status)` — not just `(id, upgraded)`.

**Clone cost (measured).** `ClonePreservingMutability()` on a `CardModel` is **0.28µs** and
verifiably independent (mutating the clone leaves the original untouched). A ~40-card deck
snapshot is therefore ~11µs, i.e. order 10^5 combat snapshots/sec from the card graph — well
inside an MCTS budget. Note `ToMutable()` asserts the receiver is canonical; use
`ClonePreservingMutability()` for already-mutable models.

## Root cause of the headless crashes — solved

Combat now runs headless. Three separate blockers, found with the tracer (`--probe=trace`,
which logs every method entry to a flushed file, because a SIGSEGV unwinds nothing):

**1. `TestMode` is the node-layer bypass.** This was the big one. Every visual path in the
gameplay commands is gated on `TestMode.IsOff` — e.g.:

```csharp
// CardPileCmd.AddDuringManualCardPlay
if (TestMode.IsOff) {
    nCard = NCard.FindOnTable(card);
    if (nCard == null) nCard = CreateCardNodeAndUpdateVisuals(...);  // NCard.Create -> segfault
}
```

`CreateCardNodeAndUpdateVisuals` instantiates an `NCard` scene and dereferences
`NCombatRoom.Instance`, both fatal under the headless dummy renderer. `TestMode.IsOn` skips
the whole path. It is a public settable property, and it is the switch MegaCrit's own unit
tests use to run combat without a scene. **Set it.**

**2. `CombatStateTracker.CallCombatStateChangedDeferred` stops deferring with no scene:**

```csharp
Node instance = NRun.Instance;
if (instance?.GetTreeOrNull() != null)
    await instance.AwaitProcessFrame();   // skipped when there is no scene
...
CombatStateChanged?.Invoke(_state);       // so this runs synchronously, re-entrantly
```

That re-entrancy is what overflowed the stack before `TestMode` was on. `TestMode.IsOn` also
short-circuits `NotifyCombatStateChanged`, so it fixes this too.

**3. `CombatManager.StartCombatInternal` creates the FTUE node unconditionally** when the
tutorial is unseen (the sibling branch is guarded by `NCombatRoom.Instance?.`, which
short-circuits argument evaluation). Patched via `SaveManager.SeenFtue -> true`.

Also note `NRun.Create` / `RunManager.EnterRoomDebug` instantiate scenes and cannot be used
headless; the probe replicates `CombatRoom.StartCombat` directly against
`CombatState`/`CombatManager` instead, which works.

## Environment status: WORKING

Combat runs headless, end to end, deterministically.

200 combats, 1571 turns, **4706 card plays in 11.0s**, single process:

| | |
|---|---|
| card plays/s | ~428 |
| combats/s | ~18 |
| turns/combat | 7.9 (range 3-13) |
| cards/combat | 23.4 |
| outcomes | 85% enemies_cleared, 15% player_dead |
| determinism | two runs, same seed -> identical turn and card counts |

At ~1e8 card plays that is **2.8 days single-process, or ~5.6 hours across 12 processes** —
so the sample budget is tractable on one machine.

### The two levers that got it there

**1. `--fixed-fps` uncaps the main loop.** Godot headless sits at 60fps even with
`Engine.MaxFps = 0`. Because the safe suspension point is a frame boundary, frame rate *is*
throughput. Passing `--fixed-fps 5000` took an identical 20-combat workload from 40.2s to
1.28s — a 31x speedup for byte-identical episodes. It saturates around 2,900 fps.
`run.sh` sets it by default (`FIXED_FPS` env var to override).

**2. Waits must really suspend.** Letting waits complete synchronously (what `TestMode` does
on its own) never unwinds the async chain and eventually kills the stack. `--probe-yield-mode
frame` awaits `SceneTree.ProcessFrame`, which genuinely unwinds. Combined with `--fixed-fps`
that costs almost nothing, so the default is now a real yield on *every* wait
(`--probe-yield-every 1`) — no more tuning a fragile counter.

`--probe-yield-mode delay` (1ms timer) is kept as a slower fallback.

### Driving the turn loop correctly

The first driver reported `turn_did_not_advance` on nearly every combat. Three real mistakes,
all worth knowing before writing an agent against this engine:

- **The player may only act in `PlayerTurnPhase.Play`.** Other phases (`Start`, `AutoPrePlay`,
  `AutoPostPlay`, `End`) are engine-owned.
- **Ending a turn is two-phase and engine-driven.** `SetReadyToEndTurn` completes an
  `EndTurnSignal`; the turn loop then runs phase one and enqueues `ReadyToBeginEnemyTurnAction`
  itself. Do not drive phase two by hand — wait for the turn number to advance.
- **`IsOverOrEnding` is also true during SETUP**, because it is `IsEnding || !IsInProgress`
  and `IsInProgress` stays false until `StartCombatInternal` flips it. Gate on
  `IsOverOrEnding && !IsStarting`, or every combat "ends" immediately.

## The env boundary (C# side) — DONE

`./run.sh episodes --probe-episodes 200` writes `episodes.jsonl` + `vocab.json`.

200 episodes, 4538 steps, **458 steps/s** with observations (465 without — extraction costs
~1.5%, so the engine is the bottleneck, not encoding). Step counts and outcomes are identical
with and without observations, i.e. **encoding does not perturb the sim**.

Random-over-legal-actions wins 53/200 (27%); the greedy driver wins 85%. That gap is the
baseline any learned policy has to beat.

### `vocab.json` — embedding table sizes

Stable string-id -> int maps built from the live `ModelDb`, index 0 reserved for none/unknown:
cards 596, relics 299, powers 283, potions 66, monsters 107, enchantments 24, plus the enum
orderings (`card_type`, `card_rarity`, `target_type`, `intent_type`, `turn_phase`). Indices are
only stable for a given build — re-dump on every version bump.

### Observation schema

Per decision point:

- **globals** — turn, round, phase (+idx), side
- **player** — hp, max_hp, block, energy, max_energy, **stars** (StS2's second resource),
  gold, powers `[{power_idx, amount}]`, relics, potions
- **hand** — one token per card carrying the full identity your correction established:
  `card_idx, upgrade, ench_idx, ench_amount, ench_disabled, cost, cost_x, star_cost, type,
  rarity, target_type, playable`
- **draw_bag / discard_bag / exhaust_bag** — `{card_idx: count}`. The draw pile is a multiset
  the player cannot order, so it is a bag, not ordered tokens; discard and exhaust likewise
  carry no order-dependent decision value. Counts emitted alongside.
- **enemies** — `monster_idx, hp, max_hp, block, alive, hittable, powers[], intents[]`, where
  intents are numeric (`type_idx`, `damage`, `repeats`) via `AttackIntent.GetTotalDamage`,
  because magnitude is the decision-relevant part

### Action schema

`legal` is a variable-length candidate set — which is exactly why the policy head must score
action embeddings rather than use a fixed output vocabulary:

- `{kind:"play", hand:i, target:t, card_idx}` — one entry per legal (card, target) pair;
  untargeted plays are emitted only when no targeted variant is legal, so cards that hit all
  enemies do not appear once per enemy
- `{kind:"end_turn"}`

Legality comes from the engine (`CanPlayTargeting` / `CanPlay`), so it is rule-exact for free.

### Records

Step: `ep, t, encounter, obs, legal[], action, action_idx, hp_delta`.
Terminal: `ep, terminal, outcome, won, turns, steps, hp_start, hp_end, reward`.

`hp_delta` per step is there so dense auxiliary targets (damage taken, turns survived) can be
derived offline without re-running the sim.

**Size warning:** ~2.2KB/step with observations. At 1e8 steps that is ~220GB of JSONL, so a
packed binary format (or on-the-fly encoding straight into training batches) is needed before
scaling up. `--probe-no-obs` drops it to ~0.6KB/step for action-only traces.

## Is intra-turn play order exactly searchable? (`--probe=rngaudit`)

The intra-turn exact-search plan assumes "no RNG fires until the end-of-turn draw". Measured,
that is **nearly true but not exactly**, and the exception matters.

`RunRngSet` keeps **separate, independently seeded streams per purpose** (`Shuffle`,
`CombatTargets`, `MonsterAi`, `CombatEnergyCosts`, `CombatCardGeneration`, `Niche`, ...). That
is the key structural fact: a draw on one stream cannot perturb another, so cosmetic
randomness is harmless to game state.

300 random-policy episodes, Ironclad, act-1 normals — Play-phase draws by stream:

| stream | draws | affects game state? |
|---|---|---|
| `chaotic(cosmetic)` | 6056 (99.3%) | **no** — screen shake, audio pitch |
| `run.Shuffle` | 43 | **yes** — draw pile reshuffled when it empties mid-turn |

120 episodes on **elite** encounters adds one more:

| stream | draws | cause |
|---|---|---|
| `run.Niche` | 20–44 | monster HP roll from `CreateCreature` — an elite **summoning a minion mid-turn** |

### What this means for the search

Two confirmed mid-turn chance events: **draw-pile reshuffle** (any card that draws, when the
pile runs out) and **mid-turn summons**. Everything else observed was cosmetic.

The right response is *detect, don't assume*. Because the streams are named and separable, the
search does not have to trust a determinism assumption — the same hook used for this audit can
run inside search: expand intra-turn nodes exactly, and insert a chance node precisely when a
**gameplay** stream is consumed. That is strictly more robust than assuming, and the
instrumentation already exists.

Also worth doing regardless: **stub the cosmetic draws** (`NGame.ScreenShake`,
`NDebugAudioManager`) headless. They are 99.3% of Play-phase RNG traffic and pure waste.

### On debuffs and relics specifically

Two separate questions, often conflated:

- **Does order matter?** Yes, enormously — applying Vulnerable before attacking is a different
  outcome from the reverse. That is not a problem for the design, it is the *reason* to search
  intra-turn ordering at all. Deterministic does not mean order-independent.
- **Does RNG fire?** Only for the cases above, in this sample.

**Sample limits, stated plainly:** Ironclad / Necrobinder / Regent only, act 1, regular +
elite, random policy, starting relics, no potions or shop items. The relic space is barely
explored, so relics with random effects are *structurally* possible (the `CombatTargets`,
`CombatCardGeneration`, `CombatCardSelection`, `CombatOrbs`, `CombatPotionGeneration` and
`MonsterAi` streams all exist) but simply did not fire here. The detect-don't-assume design
covers them without needing an exhaustive audit first.

## Card-select action kind — WORKING

Card selection ("choose a card from your hand", Quasar's choose-1-of-3) is a genuine agent
decision, so it is modelled as a decision point rather than answered by a fixed rule.

- `CardSelect : ICardSelector` — agent-driven. The interface is async, so the combat coroutine
  parks on the returned task while the env surfaces the choice.
- Action kinds `select_card` / `select_done`. Multi-select (min..max) is a **sequence of single
  picks**, keeping the action space uniform instead of combinatorial in hand size.
- A parked prompt takes priority in `LegalActions` and in the decision loop: the coroutine is
  suspended inside a card effect and cannot progress until it is answered.
- Binary format **v2** (action kind 0..3). Cross-check extended and passing:
  1122 steps, ~90,674 fields.

Verified with `--probe-inject-cards ARMAMENTS`: **116 select_card decisions chosen out of 336
offered** across 40 episodes, and 25,511-step greedy datasets train end to end
(value BCE 0.180 -> 0.058).

**Silent now runs** (100 episodes, 508 steps/s) — it was the character this work was meant to
unblock.

### Character status

| character | status |
|---|---|
| Ironclad, Silent, Necrobinder, Regent | working, ~450-510 steps/s |
| Defect | **crashes** (see below) |

### Correction to the previous writeup

The earlier conclusion that "card-select segfaults on the first prompt" was **wrong**. The
crash came from the `--probe-inject-cards` helper, not the selection path: injected cards were
added to the deck without setting `CardModel.Owner`, and combat setup then died walking hook
listeners during the opening shuffle. Injecting a plain `STRIKE_IRONCLAD` reproduced it
identically, which is what proved the selector innocent. `card.Owner = Player` before
`Deck.AddInternal` fixes it.

Lesson: when a new feature appears to crash, check the *test harness* built to exercise it.

### Two mistakes worth not repeating

- **Register the selector on ONE stack.** `AutoSlayer` uses `UseSelector(sel)` with the default
  `localOnly: false`. Pushing to both stacks makes `CardSelectCmd.FromHand` skip reserving a
  choice id and signalling the player-choice context, then run the local branch against a
  half-initialised choice.
- **Await, never poll.** Replacing `await BecameEmpty()` with a `Task.Delay` poll loop crashed
  the engine at any scale: spinning while an action is mid-flight lets Godot's main loop
  re-enter. Use `Task.WhenAny(queue.BecameEmpty(), CardSelect.Raised)`.

## Defect — FIXED (root cause: the harness bypassed the room lifecycle)

All five characters now run. 300-1000 episodes each, no crashes, ~450-515 steps/s:

| character | episodes | steps/s |
|---|---|---|
| Ironclad | 300 | 455 |
| Silent | 300 | 512 |
| **Defect** | **1000** | **460** |
| Necrobinder | 300 | 459 |
| Regent | 300 | 499 |

Defect also clears elites and 598 card-selection decisions without incident.

### The actual bug — it was ours, not the engine's

`CombatManager.EndCombatInternal` opens with:

```csharp
CombatRoom room = (CombatRoom)runState.CurrentRoom;
...
room.OnCombatEnded();
runState.Map.SecondBossMapPoint ...
runState.CurrentMapPointHistoryEntry ...
```

The harness built a `CombatState` by hand and called `SetUpCombat` directly, replicating
`CombatRoom.StartCombat` but **never entering a room and never generating a map**. Going *in*
that works. Coming *out* it does not: `CurrentRoom` is null and `Map` is null, so the
end-of-combat path dereferences garbage. Characters differ only in how much work runs before
reaching it, which is why it looked Defect-specific.

The fix is to stop bypassing the engine:

```csharp
await RunManager.Instance.SetActInternal(0);
await RunManager.Instance.GenerateMap();
...
await RunManager.Instance.EnterRoom(new CombatRoom(enc, RunState));
```

Both `EnterRoomInternal` and `GenerateMap` are node-safe headless (`NMapScreen.Instance?` is
null-conditional), so this needed no new suppression.

### How it was found, after six wrong guesses

The breakthrough was **resolving the faulting instruction pointer against the JIT symbol map**.
Everything before that was inference; this was measurement:

    DOTNET_PerfMapEnabled=1 ...            # CLR writes /tmp/perf-<pid>.map
    gdb -ex "handle SIGSEGV stop nopass" -ex run -ex "info registers rip"
    # then look up rip in the map

which printed `CombatManager+<EndCombatInternal>d__122::MoveNext()` directly. Two earlier
observations had also been decisive: **RSP sat at the top of the stack**, killing the
stack-overflow theory for good (a 512MB `ulimit -s` had not helped either, and CoreCLR never
printed its "Stack overflow." message), and the CLR's own crash handler never fired, which
pointed at a wild-pointer dereference rather than a managed null.

Hypotheses tested and rejected along the way, none of which were the cause: an infinite
affliction/card-replacement loop; managed stack overflow; a teardown race against our own
`Reset`; afflictions leaking across combats; leaked `CombatStateTracker` card subscriptions;
and headless `MegaRichTextLabel` font sizing.

### Workarounds this retired

All of these are gone, replaced by the real fix:

- `--probe-yield-mode delay` as a Defect requirement (it is still available, but no longer needed)
- excluding `VINE_SHAMBLER_NORMAL`
- the manual `CombatStateTracker` card unsubscribe in teardown — the engine's own room exit
  handles it, verified by removing it and running 500 episodes of Defect and Silent clean
- crash-restart resilience in the Python env, which was never committed

### Residual: a rare observation race

Trajectories are fully deterministic — across paired runs, step counts, chosen actions,
outcomes, turns and end HP all match exactly. But roughly **one observation field in 90,000**
can still be captured mid-settle (a card's pile membership), because `BecameEmpty()` may
complete while a continuation from the just-resolved card is still pending. A quiesce check
(require the queue empty across a suspension, `--probe-quiesce`) roughly halves it but does not
eliminate it, and costs ~45% throughput (455 -> 251 steps/s), so it is **off by default**. It
shows up as a cross-format mismatch whose position *moves* between runs, which is how it was
identified as a race rather than an encoding bug. Low priority: it perturbs no action, reward
or outcome, and never the trajectory.

## Env server (`--probe=serve`) — the RL bridge

Every episode up to this point was generated by a policy written in C#. Training needs the
*network* to choose, so `--probe=serve` connects back to Python over TCP and asks for an action
at each decision point:

    -> {"t":"decision","ep":N,"step":N,"obs":{...},"legal":[...]}
    <- {"a":INDEX}
    -> {"t":"terminal","ep":N,"outcome":"...","won":bool,"reward":F,...}

TCP rather than stdio because Godot writes freely to stdout. One process is one env; scale by
running several. Python side is `alphaslayer/env.py` (`SpireEnv`).

**Note:** `run.sh` builds *and installs* the mod. When invoking the game directly, copy
`bin/Release/alphaslayer_probe.dll` into `mods/alphaslayer_probe/` yourself or you will run a
stale build — which looks exactly like "the new phase silently does nothing".

## Tooling built along the way

- `--probe=trace` — patches ~1,240 combat-path methods to log every entry to `trace.txt`.
  This is what located both crash sites; a native SIGSEGV leaves no other evidence.
- `--probe=recursion` — patches ~27,000 methods with a
  `RuntimeHelpers.TryEnsureSufficientExecutionStack()` guard that dumps a **symbolised**
  managed stack before the overflow. Self-tests itself first (caught deliberate recursion at
  depth 257,475), so a negative result is trustworthy — and its negative result is what ruled
  out "stack overflow in game code" and redirected the hunt to the node dereference.

## Older notes
### (superseded) The original blocker writeup

`throughput` reproducibly **segfaults with a stack overflow** once the main loop advances
during combat: `coredumpctl` shows one ~20-frame cycle of JIT'd addresses repeated 50+ times.

It reproduces with `--probe-natural` (engine timing completely untouched), so it is **not**
caused by removing waits. It is the combat turn loop recursing unboundedly when driven with
no node layer present.

Two node-layer landmines were found and worked around on the way in, both real:

- `NRun.Create()` / `RunManager.EnterRoomDebug` instantiate scenes and segfault headless. The
  probe bypasses them and replicates `CombatRoom.StartCombat` directly against
  `CombatState`/`CombatManager` — that part works.
- `CombatManager.StartCombatInternal` calls `NCombatRulesFtue.Create()` **unconditionally**
  when the FTUE is unseen (the sibling branch is guarded by `NCombatRoom.Instance?.`, which
  short-circuits argument evaluation). Patched via `SaveManager.SeenFtue -> true`.

So the rules layer is decoupled from Godot at the **type** level (0 of 1,563 model/combat/
action files inherit a Godot type), but combat **control flow** is still entangled with the
node layer. That is the remaining Option A risk, and it is engineering, not configuration.

### Next step

Get a managed stack for the recursion. The native backtrace is unsymbolised JIT frames, so
either attach `lldb` with the SOS plugin, or bisect by Harmony-patching the turn-loop members
(`CombatManager.StartTurn`, `AwaitTurnEndAndSwitchSides`, `CheckWinCondition`,
`ActionExecutor.GetReadyAction`) with a depth counter that dumps `Environment.StackTrace` past
a threshold. That names the cycle, and the cycle names the fix.
