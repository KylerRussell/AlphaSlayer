# Capabilities the unified model needs

Status: plan, 2026-09-26. Companion to `UNIFIED_MODEL_DESIGN.md`. That document says how the
model is built; this one says what it must be able to DO, what each capability depends on, and how
we will know it has it.

Every capability below is checked against three questions:

1. **Can it see it?** Is the information in the observation?
2. **Is it paid for it?** Does the objective reward the behaviour, or at least not punish it?
3. **Can it compute it?** Is one forward pass enough, or does it need lookahead?

A capability fails if any answer is no, and no amount of training fixes a "can't see it".

---

## 0. The objective: maximise P(win)

"Choose the highest win rate, not the highest reward, when the reward costs too much risk" has an
exact form: **the only reward is winning the run.** Then the value of a state IS P(win | state),
and maximising expected return IS maximising win probability. Taking risk is right exactly when
it raises P(win). That makes it cautious when ahead and willing to gamble when behind, without a
risk knob to tune.

What changes from today:

| today | why it distorts | unified |
|---|---|---|
| +1.0 per act cleared | pays for reaching act 2 even by a line that lowers P(win) | removed as reward; kept as an auxiliary PREDICTION |
| terminal +0.5·hp% | pays for ending with HP, which is worthless once the run is won | removed |
| fight reward `won + 0.5·hp_end` | values HP identically before a campfire and before the boss | fight ends at V(post-fight state) = P(win) from there |
| hp / combat potential shaping | potential-based: does not change the optimal policy | kept; it only moves credit earlier |

**The cost is sparsity.** At a 5% win rate, "won" alone is a thin signal. The answer is dense
AUXILIARY heads that shape the representation without entering the objective:

- P(win this fight), HP lost this fight (kept from CombatNet)
- P(clear the current act), P(reach act 3)
- floors survived from here

The policy is never rewarded for these; the trunk just has to predict them. The value head is
the thing we check for **calibration** (capability E1). Every risk judgement below depends on it.

---

## A. Combat tactics: within a turn

| # | capability | what good play looks like | sees it? | computes it? |
|---|---|---|---|---|
| A1 | **Damage and lethal math** | knows a Strike does 9 into Vulnerable; takes lethal when it exists; counts multi-hit against block | yes: per-target card numbers (added this week) | one pass can approximate; **intra-turn search makes it exact** |
| A2 | **Block efficiency** | blocks what is incoming and no more; ignores block when the enemy is not attacking; accounts for Frail/Weak | yes: intents + numbers | same |
| A3 | **Sequencing** | debuff before attacking, draw before spending energy, powers on turn 1 of long fights, orb evokes in the right order | partly: no card **keywords** yet | search |
| A4 | **Target priority** | kills the enemy that scales or buffs others first (Queen's Amalgam, summoners, Kaiser Crab's parts) | yes | learned per encounter |
| A5 | **Infinites and engine turns** | recognises a loop (draw + energy + exhaust/retain) and runs it until the fight is won; stops when it isn't infinite | **no**: needs keywords; piles are visible | **search**; a single-step policy rarely discovers a 20-play loop by sampling |
| A6 | **Encounter mechanics** | plays each boss's script: Aeonglass starts with 3 Artifact (the first debuffs bounce off) and runs a fixed three-move cycle whose third move adds Strength; Queen applies 99 Weak/Frail/Vulnerable on her second move, so the damage race is front-loaded | partly: intent **types** only, not the move | add the enemy's next move id; per-encounter curriculum |

The harness allows infinites: the 50-turn cap counts TURNS, not plays.

## B. Across fights: resources that outlive the fight

None of these can be learned while a fight's reward ends at the fight. All of them need the
fight to end at V(post-fight state).

| # | capability | examples in StS2 | sees it? |
|---|---|---|---|
| B1 | **Permanent gains** | play Royalties (Power, 30 gold) in any fight long enough to afford it; take Hand of Greed's gold on a kill; stretch a safe fight to finish a setup that pays forever | **no**: gold is not among the card-number slots, so Royalties shows all zeros |
| B2 | **Carry-over setup** | end a fight with relic counters primed (Pen Nib-style) for the next one | yes: relic counters |
| B3 | **HP as a currency** | Offering (6 HP → 2 energy + draw), Bloodletting, Hemokinesis, Blood Wall, Breakthrough, Brand: pay HP when it is cheap (campfire next, healing relic, fight ends sooner and saves more HP than it costs), refuse before the boss | yes, once the fight ends at V(post) |
| B4 | **Potion economy** | hold potions for elites, bosses and dead draws; but DRINK rather than waste one when the belt is full and a potion reward is coming | identity yes, **numbers no** (potions have their own dynamic vars) |

## C. Run strategy

| # | capability | what good play looks like | sees it? |
|---|---|---|---|
| C1 | **Deck building and archetypes** | commits to what the character's cards reward (see D) and takes cards for the deck it HAS | yes, with every deck card as a token |
| C2 | **Thinning and upgrades** | removes Strikes/curses before buying; picks the upgrade target (a card_select) that changes the most fights | yes |
| C3 | **Curse risk** | prices a curse by the deck it enters: thin decks draw it more; Eternal ones can't be removed; Debt costs gold; removal needs shop gold. 18 curses in the game | identity yes, **keywords no** |
| C4 | **Event trade-offs** | HP or a curse for a relic, gold for a card, etc. | **no**: options are hashed text keys only; the events' numbers (54 of 68 events carry gold / HP loss / max HP / heal vars) are never sent |
| C5 | **Pathing and boss prep** | elites when healthy, a campfire before the boss, shops when gold allows, AoE for multi-part bosses and scaling for Aeonglass | **boss identity no** (planned in M1) |
| C6 | **Campfires** | heal or smith depending on HP AND what comes next | yes, since the act/floor fix |
| C7 | **Relic choices** | Ancient offers at each act start, treasure, shop: judged against the deck and character | yes |
| C8 | **Gold planning** | budgets across the act: removal vs relic vs card, skips a shop it can't use | yes |

## D. Character mastery

Counts are cards in each character's pool that touch the mechanic (from the game's card code).

| character | defining mechanics | what the model needs |
|---|---|---|
| Ironclad | Strength scaling (10), Exhaust synergies (29), self-damage (7) | keywords (Exhaust); B3 for HP-cost cards |
| Silent | Poison (13), Shivs (13), discard (9), Sly (9), Retain (5) | keywords (Sly, Retain); poison stacks on enemies are already visible as powers |
| Defect | Channel (26), orb slots, Focus (6) | orbs now observed; evoke ordering benefits from search |
| Necrobinder | Osty (22), Summon (12), Doom (14), Souls (11), Ethereal (16), Retain (11) | Osty now observed; keywords are critical (Ethereal cards are lost if not played) |
| Regent | Stars (35), Forge (12), Sovereign Blade | stars observed; star numbers on cards need the `stars` slot (present) |

Necrobinder is the weakest character in every room today, and it has the most keyword-dependent
card pool (Ethereal + Retain = 27 cards).

The model learns these from shared weights plus the character in the context token. We do not
hand-write archetypes. What we check is that each character's defining resources are USED (D-evals
below), not merely held.

## E. Deciding under uncertainty

| # | capability | why it matters |
|---|---|---|
| E1 | **Calibrated P(win)** | a value that says 0.6 must win 60% of the time. Every risk trade (B3, B4, C3, C4) is only as good as this |
| E2 | **Risk posture follows P(win)** | safe lines when ahead, high-variance lines when behind. Free from the objective; worth verifying |
| E3 | **Reasoning about hidden information** | draw-pile odds (the pile is a visible multiset), what "?" rooms and future rewards tend to hold. Search must not peek at the real draw order |
| E4 | **Generalising to rarely seen items** | a card seen 10 times in training must still be judged sensibly: numbers + keywords + (optionally) text embeddings, not an id alone |

---

## Enablers: what has to be built

Deduplicated from the tables above, in dependency order.

**Observation (probe): DONE in M1 (2026-09-26),** each with a live delivery test that passes
on the M1 probe and fails on the one before it (`tests/test_combat_obs.py`,
`tests/test_run_obs.py`):

1. Card **keywords** as a bitmask on every card: hand, piles, deck, rewards, shop, prompts.
   Offering reads Exhaust; a Strike reads none. → A3 A5 C3 D
2. **`gold` and `max_hp` card-number slots.** Royalties now reads gold 30. → B1
3. **Enemy next move id** (Queen opens with PUPPET_STRINGS_MOVE; Aeonglass cycles its three
   moves) and **status-card counts** on status intents. → A6
4. **Potion and relic numbers** on the belt, the shop, treasure and the relic list. → B4 C7 E4
5. **Event option effects:** the event's numbers each option's text refers to (20/20 events
   with numbers attribute them), the cards each option shows (curses: Hefty Tablet → Injury,
   Cursed Pearl → Greed), every hover-tip id, and the event's full numbers as a fallback.
   → C3 C4
6. **Act boss identity** in every run decision (every Boss-room fight matched the announced boss,
   31/31); **card piles** as sorted multisets with upgrades and keywords; **act/floor inside
   fights**; **numbers on every offered card**; **harvest records** with floor, gold, boss and
   potions. → C5 B E3
7. *Optional, not built:* frozen text embeddings of card/relic/potion/event descriptions. → E4
   (M10)

Verified: none of it changes the game. Old and new probes produce identical games over 1,159
steps (`tests/test_obs_no_side_effects.py`).

**Objective and heads:** the P(win) objective with the auxiliary heads in section 0.

**Architecture:** as in `UNIFIED_MODEL_DESIGN.md`. Candidate tokens are what let C1/C2/C3 compare
an option against the deck it joins.

**Search** (after the determinism race is fixed): intra-turn search, where the only chance events
are reshuffles and summons. It is what makes A1-A5 exact, and the one place a larger network is
not a substitute.

---

## Capability evals

A win rate says THAT the model improved, not what it learned. Each capability gets a targeted
measurement, run every 25 iterations and on every checkpoint we compare. Most are scenarios built
with the probe's existing controls (deckserve with an exact deck, `--probe-inject-cards`,
`--probe-inject-relics`, a forced act/encounter, a starting hp).

| eval | setup | metric |
|---|---|---|
| A1 lethal | fights where lethal exists this turn (found by brute force over the hand) | % of those turns where it kills |
| A2 block | per turn: block gained vs incoming damage | over-block and under-block rate |
| A5 infinite | an injected known-infinite deck vs a high-HP enemy | turns to kill; fraction won on the loop turn |
| A6 per boss | each boss, 200 fights on real harvested decks | win rate per boss (Queen and Aeonglass are 3.5% and 7% today) |
| B1 permanent | Royalties / Hand of Greed injected, hallway fights | play rate; gold gained per fight |
| B3 HP cost | Offering etc. injected, fights before a campfire vs before the boss | play rate in each context; net HP; whether it differs |
| B4 potions | real runs | share of potions used in boss/elite fights; potion rewards lost to a full belt |
| C3/C4 curse & events | real runs, logged per event option | accept rate vs deck size, gold and HP |
| C5 boss prep | real runs | card picks in the act before each boss, by boss |
| C6 campfire | real runs | heal/smith by HP and by floors-to-boss (today: HEAL 100%) |
| D per character | real runs | per-character win rate plus mechanic usage (evokes, stars spent, Osty summons, shivs made) |
| E1 calibration | held-out states | reliability curve: predicted P(win) vs realised |
| E2 risk posture | states bucketed by predicted P(win) | variance of the chosen line vs alternatives |

Some evals need a reference to judge against. B3 is judged counterfactually: play the same fight
with and without the HP-cost card through deckserve, which the old expert module already does
for decks.

## Priorities

The data says where the wins are lost: four bosses (Queen 3.5%, Aeonglass 7%, Kaiser Crab 24%,
Insatiable 44%). The capabilities that bear on that most directly come first:

1. **E1 + objective**: P(win) as the target with calibrated value. Everything else trades against it.
2. **A1-A4 and A6**: boss fights are won or lost here. Enemy move ids now; intra-turn search next.
3. **B4 + C5**: potions saved for, and decks prepared for, the boss the model can now see coming.
4. **D (keywords)**: Necrobinder and Silent are keyword-heavy and weakest.
5. **B1, B3, C3, C4**: real value, smaller share of losses today.

---

## Implementation plan: every capability, traced

Each capability needs three things (sees it / paid for it / computes it) and an eval. This table
names the milestone that delivers each one. A capability is DONE only when its eval shows the
behaviour, not when its inputs exist.

Milestones (details in `UNIFIED_MODEL_DESIGN.md`):
**M1** probe observations (done) · **M2** encoder + model · **M3** distillation from r5 ·
**M4** unified trainer: P(win) objective, aux heads, V(post-fight) bootstrapping, curriculum ·
**M5** capability eval suite · **M6** training round · **M7** determinism race fix ·
**M8** intra-turn search (play time) · **M9** search as teacher (Expert Iteration / Gumbel) ·
**M10** text embeddings (optional)

| cap | sees it | paid for it | computes it | eval (M5) |
|---|---|---|---|---|
| A1 lethal / damage math | M1 per-target numbers; M2 encodes | M4 (P(win) prefers the kill) | M2 trunk; exact with M8 | lethal-taken rate |
| A2 block efficiency | M1 intents + numbers | M4 | M2; exact with M8 | over/under-block rate |
| A3 sequencing | M1 keywords | M4 | M8 search over orderings | debuff-before-attack rate; A/B vs search |
| A4 target priority | M1 enemy powers + moves | M4 | M2 enemy tokens attend to each other | kill-order vs encounter script, per boss |
| A5 infinites | M1 keywords + piles | M4 (a won fight's V) | **M8 required**; M9 teaches it back | injected infinite deck: kill-turn |
| A6 encounter scripts | M1 move ids | M4 + per-boss curriculum weighting (M4) | M2 | per-boss win rate |
| B1 permanent gains | M1 gold / max_hp slots | **M4: fight ends at V(post)** | M2 | Royalties play rate; gold per fight |
| B2 carry-over setup | relic counters (existing) | M4 | M2 | counter state at fight end vs random |
| B3 HP as currency | M1 hp_loss numbers | M4 | M2 | HP-cost card use before campfire vs boss |
| B4 potion economy | M1 potion numbers | M4 (no free potion use) | M2 | potions spent in boss/elite; potions lost to a full belt |
| C1 deck building | M1 deck numbers + keywords | M4 | **M2 candidate tokens attend to the deck** | per-character archetype concentration vs win rate |
| C2 thinning / upgrades | M1 | M4 | M2 | removal targets; upgrade targets |
| C3 curse risk | M1 keywords + event cards/tips | M4 | M2 | curse-accept rate vs deck size / gold |
| C4 event trade-offs | M1 option numbers, cards, tips | M4 | M2 | option choice vs HP / gold / deck, logged per event |
| C5 pathing + boss prep | M1 boss id | M4 | M2 map tokens + lookahead features | picks and paths in the act before each boss |
| C6 campfires | act/floor fix (done) | M4 | M2 | heal/smith by HP and floors-to-boss |
| C7 relic choices | M1 relic numbers | M4 | M2 | pick rates per character; win rate by pick |
| C8 gold planning | existing gold + shop costs | M4 | M2 | gold spent per act; shops skipped with gold |
| D  character mastery | M1 orbs, Osty, keywords, stars | M4 | M2 (character in context) | per-character win + mechanic-usage counts |
| E1 calibrated P(win) | — | **M4: win-only reward makes V = P(win)** | M2 value head + aux heads | reliability curve on held-out states |
| E2 risk posture | — | M4 (falls out of the objective) | M2 | line variance by predicted P(win) |
| E3 hidden information | M1 sorted piles (no leak) | M4 | M2; M8 samples draw orders, never the real one | draw-odds probes; search audit for leakage |
| E4 rare items | M1 numbers + keywords; M10 text | M4 | M2 shared encoders | win-rate delta on cards seen < 50 times |

**What the table says:**

- M1 delivered "sees it" for every capability except E4's optional text embeddings.
- **M4 is the keystone for "paid for it".** B1-B4 and E1-E2 are unlearnable until the fight's
  terminal is V(post-fight state) and the reward is win-only.
- **A3, A5 and exact A1/A2 need search (M8),** which needs the determinism race fixed first (M7).
  Without M8, A5 (infinites) is not expected to emerge from sampling.
- **M5 comes before the M6 training round,** so the round is measured on capabilities from its
  first iteration rather than only on win rate.
