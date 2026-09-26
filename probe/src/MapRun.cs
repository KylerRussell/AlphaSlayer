using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace AlphaSlayer.Probe;

/// <summary>
/// Phase `maprun` - walks whole acts through the engine's own map/room transitions.
///
/// This is the run layer's equivalent of the original `throughput` phase: before any policy
/// is trained on map decisions we need to know that the transitions themselves survive
/// headless, and WHICH room types do not. Combat is driven by the built-in greedy policy so
/// that fights are not the variable under test; every other room is entered, inspected and
/// left, and anything that throws or hangs is recorded per room type rather than taking the
/// process down.
///
/// The output is therefore a room-type support matrix - the concrete list of what the run
/// layer still needs - plus a measurement of how far a run actually gets.
/// </summary>
public static class MapRun
{
    private sealed class Tally
    {
        public int Entered, Ok, Failed, Hung;
        public readonly List<string> Errors = new();
    }

    public static async Task RunAsync()
    {
        var runs = Probe.ArgInt("probe-runs", 1);
        var ascension = Probe.ArgInt("probe-ascension", 0);
        var seed = Probe.ArgValue("probe-seed") ?? "ALPHASLAYER1";
        var pathPolicy = (Probe.ArgValue("probe-path-policy") ?? "random").ToLowerInvariant();
        var combatPolicy = Probe.ArgValue("probe-policy") ?? "greedy";
        var turnCap = Probe.ArgInt("probe-turn-cap", 50);
        var floorCap = Probe.ArgInt("probe-floor-cap", 200);
        var travelTimeoutMs = Probe.ArgInt("probe-travel-timeout-ms", 20000);
        var roomTimeoutMs = Probe.ArgInt("probe-room-timeout-ms", 30000);
        var godmode = Probe.HasFlag("probe-map-godmode");
        var rng = new Random(Probe.ArgInt("probe-policy-seed", 12345));

        Vocab.Build();
        if (Probe.HasFlag("probe-probe-trial")) NodeGuard.InstallTrialProbe();

        // --probe-trace-rooms: method-entry trace, for locating a native fault. The run layer
        // touches far more of the codebase than combat did, and a SIGSEGV in it unwinds
        // nothing, so the last line of trace.txt is the only evidence available.
        var traceFromRun = Probe.ArgInt("probe-trace-from-run", 0);
        if (Probe.HasFlag("probe-trace-rooms") && traceFromRun == 0)
            Tracer.Install(Probe.OutDir, Probe.HasFlag("probe-trace-all")
                ? new[] { "MegaCrit.*" }
                : new[]
            {
                "MegaCrit.Sts2.Core.Models.Events",
                "MegaCrit.Sts2.Core.Events",
                "MegaCrit.Sts2.Core.Nodes.Events",
                "MegaCrit.Sts2.Core.Nodes.Rooms",
                "MegaCrit.Sts2.Core.Nodes",
                "MegaCrit.Sts2.Core.Assets",
            });

        var byRoom = new Dictionary<string, Tally>();
        var runOutcomes = new Dictionary<string, int>();
        var floorsReached = new List<int>();
        var actsCleared = new List<int>();
        var sw = Stopwatch.StartNew();
        long combats = 0, combatWins = 0;
        var restChoices = new Dictionary<string, int>();
        var eventChoices = new Dictionary<string, int>();
        var shopPurchases = new Dictionary<string, int>();
        var potionsDrunk = new Dictionary<string, int>();

        Tally T(string k) => byRoom.TryGetValue(k, out var t) ? t : byRoom[k] = new Tally();

        // --probe-run-offset shifts the seed sequence so a single failing run can be
        // reproduced on its own instead of replaying every run before it.
        var runOffset = Probe.ArgInt("probe-run-offset", 0);
        for (var runIdx = 0; runIdx < runs; runIdx++)
        {
            var run = runIdx + runOffset;
            // Deferred tracing: a fault that only appears after N runs is a fault in state
            // that survives CleanUp, so the trace has to start late or it drowns in the runs
            // that work.
            if (traceFromRun > 0 && run == traceFromRun && Probe.HasFlag("probe-trace-rooms"))
                Tracer.Install(Probe.OutDir, new[] { "MegaCrit.*" });
            var harness = new RunHarness();
            await harness.StartRunAsync($"{seed}{run}", ascension, Probe.ArgValue("probe-character"));
            CardSelect.Install();
            RunRewards.Install();
            CrystalSphere.Install();
            InstallDiagnosticRewardPolicy(rng);

            if (run == 0)
                Probe.Log($"  character={harness.Character.Id.Entry} acts={harness.ActCount} " +
                          $"map={harness.RunState.Map?.GetColumnCount()}x{harness.RunState.Map?.GetRowCount()} " +
                          $"points={harness.AllPoints().Count}");

            // Dump the observation the run policy will actually consume, once, so the
            // Python encoder can be written against a real payload rather than a guess.
            if (run == 0 && Probe.HasFlag("probe-dump-map"))
            {
                Probe.Report["map_obs_sample"] = harness.MapObs();
                Probe.Report["map_actions_sample"] = harness.LegalRunActions()
                    .Select(a => (object)a.ToDict()).ToArray();
            }

            string outcome = null;
            var floors = 0;

            while (outcome == null && floors < floorCap)
            {
                if (harness.PlayerDead) { outcome = "player_dead"; break; }

                var legal = harness.LegalRunActions();
                if (legal.Count == 0)
                {
                    if (harness.AtRunEnd()) { outcome = "run_complete"; break; }
                    if (harness.AtActEnd())
                    {
                        // Act boundary. Recorded separately because a failure here is a
                        // different bug from a failure inside a room.
                        var t = T("ActTransition");
                        t.Entered++;
                        try
                        {
                            if (await WithTimeout(harness.NextActAsync(), travelTimeoutMs)) t.Ok++;
                            else { t.Hung++; outcome = "act_transition_hang"; break; }
                        }
                        catch (Exception e)
                        {
                            t.Failed++; Note(t, e); outcome = "act_transition_failed"; break;
                        }
                        continue;
                    }
                    outcome = "no_legal_moves";
                    break;
                }

                var pick = ChoosePath(pathPolicy, legal, rng);
                var typeName = pick.PointType.ToString();

                if (Probe.HasFlag("probe-log-path"))
                    Probe.Log($"    act{harness.ActIndex} floor{floors} -> ({pick.Coord.col},{pick.Coord.row}) {typeName} " +
                              $"[{legal.Count} option(s)] hp={harness.Player.Creature.CurrentHp}");

                var travel = T($"travel:{typeName}");
                travel.Entered++;
                try
                {
                    if (!await WithTimeout(harness.TravelAsync(pick.Coord), travelTimeoutMs))
                    { travel.Hung++; outcome = "travel_hang"; break; }
                    travel.Ok++;
                }
                catch (Exception e) { travel.Failed++; Note(travel, e); outcome = "travel_failed"; break; }

                floors++;

                // The room the engine actually rolled. For a ? point this is an RNG draw, so
                // it is genuinely not derivable from the map point type.
                var room = harness.CurrentRoom;
                var roomName = room?.RoomType.ToString() ?? "None";
                var rt = T($"room:{roomName}");
                rt.Entered++;
                if (Probe.HasFlag("probe-log-path")) Probe.Log($"      room={roomName} " +
                    $"{(room is EventRoom evr ? evr.CanonicalEvent.Id.Entry : room?.ModelId?.Entry ?? "")}");

                try
                {
                    var handled = await WithTimeout(
                        HandleRoomAsync(harness, room, combatPolicy, turnCap, rng,
                                        r => { combats++; if (r) combatWins++; }, restChoices, eventChoices, shopPurchases, potionsDrunk),
                        roomTimeoutMs);
                    if (!handled) { rt.Hung++; outcome = "room_hang"; break; }
                    rt.Ok++;
                }
                catch (Exception e) { rt.Failed++; Note(rt, e); }

                // --probe-map-godmode heals between rooms. Without deck growth a random walk
                // dies around floor 8, so the act boundary, the boss and the later acts are
                // never reached - and those are exactly the transitions this phase exists to
                // test. Healing isolates the map layer from combat difficulty.
                // HealInternal revives explicitly (it fires Revived and re-activates hooks
                // when it crosses back above zero), so this covers a death mid-combat too.
                if (godmode) harness.Player.Creature.HealInternal(harness.Player.Creature.MaxHp);

                if (harness.PlayerDead) { outcome = "player_dead"; break; }
            }

            outcome ??= "floor_cap";
            runOutcomes[outcome] = runOutcomes.GetValueOrDefault(outcome) + 1;
            floorsReached.Add(floors);
            actsCleared.Add(harness.ActIndex);
            Probe.Log($"  run {run}: {outcome} after {floors} floors, act {harness.ActIndex + 1}/{harness.ActCount}, " +
                      $"hp={harness.Player.Creature.CurrentHp}/{harness.Player.Creature.MaxHp}, " +
                      $"gold={harness.Player.Gold}, deck={harness.Player.Deck.Cards.Count}, " +
                      $"relics={harness.Player.Relics.Count}, potions={harness.Player.Potions.Count()}, " +
                      $"upgraded={harness.Player.Deck.Cards.Count(c => c.CurrentUpgradeLevel > 0)}");
            CardSelect.Reset();
        }

        sw.Stop();
        Probe.Report["maprun"] = new Dictionary<string, object>
        {
            ["runs"] = runs,
            ["wall_s"] = sw.Elapsed.TotalSeconds,
            ["run_outcomes"] = runOutcomes.ToDictionary(k => k.Key, v => (object)v.Value),
            ["mean_floors"] = floorsReached.Count > 0 ? floorsReached.Average() : 0,
            ["max_floors"] = floorsReached.Count > 0 ? floorsReached.Max() : 0,
            ["mean_acts_cleared"] = actsCleared.Count > 0 ? actsCleared.Average() : 0,
            ["combats"] = combats,
            ["combat_wins"] = combatWins,
            ["support"] = byRoom.ToDictionary(k => k.Key, kv => (object)new Dictionary<string, object>
            {
                ["entered"] = kv.Value.Entered,
                ["ok"] = kv.Value.Ok,
                ["failed"] = kv.Value.Failed,
                ["hung"] = kv.Value.Hung,
                ["errors"] = kv.Value.Errors.ToArray(),
            }),
        };

        ((Dictionary<string, object>)Probe.Report["maprun"])["rewards"] =
            RunRewards.Tally.ToDictionary(k => k.Key, kv => (object)new Dictionary<string, object>
            {
                ["offered"] = kv.Value[0], ["taken"] = kv.Value[1],
                ["declined"] = kv.Value[2], ["errored"] = kv.Value[3],
            });

        Probe.Log($"MAPRUN: {runs} run(s) in {sw.Elapsed.TotalSeconds:F1}s; " +
                  $"floors mean={(floorsReached.Count > 0 ? floorsReached.Average() : 0):F1} " +
                  $"max={(floorsReached.Count > 0 ? floorsReached.Max() : 0)}; " +
                  $"combats={combats} won={combatWins}");
        Probe.Log($"  outcomes: {string.Join(", ", runOutcomes.Select(kv => $"{kv.Key}={kv.Value}"))}");
        ((Dictionary<string, object>)Probe.Report["maprun"])["rest_choices"] =
            restChoices.ToDictionary(k => k.Key, v => (object)v.Value);
        ((Dictionary<string, object>)Probe.Report["maprun"])["event_choices"] =
            eventChoices.ToDictionary(k => k.Key, v => (object)v.Value);
        Probe.Log($"  rest choices: {string.Join(", ", restChoices.OrderBy(k => k.Key).Select(kv => $"{kv.Key}={kv.Value}"))}");
        Probe.Log($"  event options taken: {eventChoices.Count} distinct, {eventChoices.Values.Sum()} total");
        ((Dictionary<string, object>)Probe.Report["maprun"])["shop_purchases"] =
            shopPurchases.ToDictionary(k => k.Key, v => (object)v.Value);
        Probe.Log($"  shop purchases: {shopPurchases.Count} distinct, {shopPurchases.Values.Sum()} total");
        ((Dictionary<string, object>)Probe.Report["maprun"])["potions_drunk"] =
            potionsDrunk.ToDictionary(k => k.Key, v => (object)v.Value);
        Probe.Log($"  potions drunk: {potionsDrunk.Count} distinct, {potionsDrunk.Values.Sum()} total " +
                  $"(in combat), {OutOfCombatPotions} out of combat");
        Probe.Log($"  crystal sphere minigames: played={CrystalSphere.Played} failed={CrystalSphere.Failed}");

        foreach (var kv in RunRewards.Tally.OrderBy(k => k.Key))
            Probe.Log($"  reward {kv.Key,-14} offered={kv.Value[0],4} taken={kv.Value[1],4} " +
                      $"declined={kv.Value[2],3} errored={kv.Value[3],3}" +
                      (kv.Value[3] > 0 ? "  <-- BROKEN" : ""));
        foreach (var kv in byRoom.OrderBy(k => k.Key))
        {
            var t = kv.Value;
            var flag = t.Failed > 0 || t.Hung > 0 ? "  <-- BROKEN" : "";
            Probe.Log($"  {kv.Key,-24} entered={t.Entered,4} ok={t.Ok,4} failed={t.Failed,3} hung={t.Hung,3}{flag}");
            foreach (var e in t.Errors.Take(2)) Probe.Log($"      {e}");
        }
    }

    /// <summary>
    /// Resolves whatever room we just entered.
    ///
    /// Only combat is actually played here; every other room type is entered and left, which
    /// is the point of the phase - it establishes whether the transition survives before we
    /// build a decision interface on top of it. Those rooms are left unresolved on purpose,
    /// so a hang shows up as a hang instead of being papered over.
    /// </summary>
    private static async Task HandleRoomAsync(RunHarness h, AbstractRoom room, string policy,
                                              int turnCap, Random rng, Action<bool> onCombat,
                                              Dictionary<string, int> restTally,
                                              Dictionary<string, int> eventTally,
                                              Dictionary<string, int> shopTally,
                                              Dictionary<string, int> potionsDrunk)
    {
        // A room can push another room on top of itself: an event option that starts a fight
        // enters a CombatRoom without exiting the event, and the event resumes underneath
        // once the fight is done. So dispatch on whatever is currently on top of the stack
        // and keep going until the stack stops changing.
        for (var depth = 0; depth < 8; depth++)
        {
            var top = h.CurrentRoom;
            if (top == null) return;
            var before = top;
            await HandleOneRoomAsync(h, top, policy, turnCap, rng, onCombat, restTally, eventTally, shopTally, potionsDrunk);
            if (ReferenceEquals(h.CurrentRoom, before)) return;
        }
    }

    private static async Task HandleOneRoomAsync(RunHarness h, AbstractRoom room, string policy,
                                                 int turnCap, Random rng, Action<bool> onCombat,
                                                 Dictionary<string, int> restTally,
                                                 Dictionary<string, int> eventTally,
                                                 Dictionary<string, int> shopTally,
                                                 Dictionary<string, int> potionsDrunk)
    {
        // Out-of-combat potions (Usage == AnyTime): healing, gold, and the thrown ones whose
        // value is in WHERE they are used - at a shop, at a campfire - rather than when. This
        // has to run before the room dispatch, since every non-combat handler returns.
        if (room is not CombatRoom && rng.NextDouble() < PotionOocRate)
            await TryOutOfCombatPotionAsync(h, rng);

        // Treasure is the one non-combat room that MUST be answered: its picking session
        // stays open until a pick or a skip, and the next chest then throws. See RunRooms.
        if (room is TreasureRoom tr)
        {
            var got = await RunRooms.ResolveTreasureAsync(h.Player, tr, choice: 0);
            if (Probe.HasFlag("probe-log-path") && got.Count > 0)
                Probe.Log($"      treasure -> {string.Join(",", got.Select(r => r.Id.Entry))}");
            return;
        }

        // Rest sites. Model-layer throughout; the only subtlety is that Smith parks on a
        // card-selection prompt, which DriveWithSelections answers.
        if (room is RestSiteRoom rs)
        {
            var picked = await RunRooms.ResolveRestSiteAsync(h.Player, rs,
                (p, opts) =>
                {
                    var usable = opts.Where(o => o.Enabled).ToList();
                    // Stop after one option unless the campfire genuinely offers more.
                    return Task.FromResult(usable.Count == 0 ? -1 : usable[rng.Next(usable.Count)].Index);
                },
                pending => { PickRandom(pending, rng); return Task.CompletedTask; });
            foreach (var id in picked) restTally[id] = restTally.GetValueOrDefault(id) + 1;
            if (Probe.HasFlag("probe-log-path") && picked.Count > 0)
                Probe.Log($"      rest -> {string.Join(",", picked)}");
            return;
        }

        // Shops. Pure model layer; card removal parks on a card-selection prompt.
        if (room is MerchantRoom mr)
        {
            var bought = await RunRooms.ResolveShopAsync(h.Player, mr,
                (p, offered) =>
                {
                    // Diagnostic policy: buy a random affordable thing, sometimes stop early,
                    // so shops are exercised without pretending to be a buying strategy.
                    var can = offered.Where(o => o.Affordable).ToList();
                    if (can.Count == 0 || rng.NextDouble() < 0.35) return Task.FromResult(-1);
                    return Task.FromResult(can[rng.Next(can.Count)].Index);
                },
                pending => { PickRandom(pending, rng); return Task.CompletedTask; });
            foreach (var b in bought) shopTally[b] = shopTally.GetValueOrDefault(b) + 1;
            if (Probe.HasFlag("probe-log-path") && bought.Count > 0)
                Probe.Log($"      shop -> {string.Join(",", bought)}");
            return;
        }

        // Events. Multi-page, and an option can start a fight; ResolveEventAsync returns as
        // soon as that happens so the outer loop can dispatch the combat.
        if (room is EventRoom er)
        {
            var taken = await RunRooms.ResolveEventAsync(h.Player, er,
                (p, opts) =>
                {
                    var ok = opts.Where(o => !o.Locked).ToList();
                    // Avoid options the engine flags as lethal - a random walk that suicides
                    // measures nothing. Real risk assessment is the run policy's job.
                    var safe = ok.Where(o => !o.WillKill).ToList();
                    var pool = safe.Count > 0 ? safe : ok;
                    return Task.FromResult(pool.Count == 0 ? -1 : pool[rng.Next(pool.Count)].Index);
                },
                pending => { PickRandom(pending, rng); return Task.CompletedTask; });
            foreach (var k in taken) eventTally[k] = eventTally.GetValueOrDefault(k) + 1;
            if (Probe.HasFlag("probe-log-path") && taken.Count > 0)
                Probe.Log($"      event -> {string.Join(",", taken)}");
            return;
        }

        if (room is not CombatRoom cr || cr.IsPreFinished) return;

        // The run policy releases potions for this fight. Here: everything, so the combat
        // path is exercised; a real run policy will hold potions back for elites and bosses.
        Potions.SetGate(null);

        // EnterRoomInternal unpauses the action executor for every room EXCEPT combat, which
        // expects its driver to do it once the room is set up.
        RunManager.Instance.ActionExecutor.Unpause();

        int turns = 0;
        string outcome = null;
        while (turns < turnCap && outcome == null)
        {
            outcome = await Episodes.WaitForDecisionPublic(h.Player);
            if (outcome != null) break;

            var combat = CombatManager.Instance.DebugOnlyGetState();
            var legal = Obs.LegalActions(h.Player, combat);
            if (legal.Count == 0) { outcome = "no_legal_actions"; break; }

            // The built-in greedy policy plays the first non-end_turn action, which would
            // drink every potion on turn one. Potions are chosen separately so the diagnostic
            // exercises them without that degenerate behaviour.
            var potionActs = legal.Where(a => a.Kind == "use_potion").ToList();
            var choice = potionActs.Count > 0 && rng.NextDouble() < PotionCombatRate
                ? potionActs[rng.Next(potionActs.Count)]
                : Episodes.ChoosePublic(policy, legal.Where(a => a.Kind != "use_potion").ToList(), rng);

            if (choice.Kind == "select_card" || choice.Kind == "select_done")
            {
                var pend = CardSelect.Pending;
                if (choice.Kind == "select_done") pend?.Finish(); else pend?.Pick(choice.Card);
                CardSelect.ClearRaised();
                await CombatHarness.Suspend();
            }
            else if (choice.Kind == "use_potion")
            {
                // EnqueueManualUse is the same call the potion popup makes; it queues a
                // UsePotionAction, so it drains through the normal action queue.
                choice.Potion.EnqueueManualUse(choice.Target);
                potionsDrunk[choice.CardId] = potionsDrunk.GetValueOrDefault(choice.CardId) + 1;
                await Episodes.WaitForQueueOrSelectionPublic();
            }
            else if (choice.Kind == "end_turn")
            {
                var expect = h.Player.PlayerCombatState.TurnNumber + 1;
                CombatManager.Instance.SetReadyToEndTurn(h.Player, false, null);
                turns++;
                outcome = await Episodes.WaitForTurnPublic(h.Player, expect);
            }
            else
            {
                RunManager.Instance.ActionQueueSet.EnqueueWithoutSynchronizing(
                    new PlayCardAction(choice.Card, choice.Target));
                await Episodes.WaitForQueueOrSelectionPublic();
            }
        }

        onCombat(outcome == "enemies_cleared");
        CardSelect.Reset();

        // Let the engine finish its own end-of-combat work before touching anything. See
        // RunRooms.WaitForCombatEndAsync - this replaced a fixed settle delay.
        if (!h.PlayerDead) await RunRooms.WaitForCombatEndAsync();

        // End-of-combat rewards. NCombatUi does this in the real game; headless nothing does,
        // and without it the deck never grows - which is the difference between a run that can
        // beat a boss and one that provably cannot.
        if (outcome == "enemies_cleared" && !h.PlayerDead)
        {
            try { await RunRewards.OfferForCombatAsync(cr, h.Player); }
            catch (Exception e) { Probe.Log($"  (non-fatal) combat rewards: {e.Message}"); }
        }

        // If this fight was nested inside an event, pop it and resume the event. This is what
        // the proceed button does; without it the event never finishes and its room stays
        // buried under a dead combat room.
        if (h.RunState.CurrentRoomCount > 1 && !h.PlayerDead)
        {
            try { await RunRooms.ResumeParentRoomAsync(); }
            catch (Exception e) { Probe.Log($"  (non-fatal) resume parent room: {e.Message}"); }
        }

        // Optional extra slack on top of the event-based wait, for bisecting timing issues.
        var settle = Probe.ArgInt("probe-teardown-settle-ms", 0);
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < settle) await CombatHarness.Suspend();
    }

    /// <summary>
    /// Answers a parked card-selection prompt at random, taking the minimum required.
    /// The prompt is a real decision the run policy will own; this only keeps the diagnostic
    /// moving without pretending to be one.
    /// </summary>
    public static int OutOfCombatPotions;
    private static double PotionCombatRate => Probe.ArgDouble("probe-potion-combat-rate", 0.15);
    private static double PotionOocRate => Probe.ArgDouble("probe-potion-ooc-rate", 0.10);

    private static async Task TryOutOfCombatPotionAsync(RunHarness h, Random rng)
    {
        var slots = Potions.UsableOutOfCombat(h.Player);
        if (Probe.HasFlag("probe-log-potions"))
            Probe.Log($"      ooc potions: held={h.Player.Potions.Count()} usable={slots.Count} " +
                      $"[{string.Join(",", h.Player.Potions.Select(x => $"{x.Id.Entry}:{x.Usage}"))}]");
        if (slots.Count == 0) return;
        var potion = h.Player.Potions.ElementAt(slots[rng.Next(slots.Count)]);
        try
        {
            potion.EnqueueManualUse(null);
            OutOfCombatPotions++;
            await RunRooms.DriveWithSelections(
                RunManager.Instance.ActionQueueSet.BecameEmpty(),
                pending => { PickRandom(pending, rng); return Task.CompletedTask; });
        }
        catch (Exception e) { Probe.Log($"  (non-fatal) out-of-combat potion: {e.Message}"); }
    }

    private static void PickRandom(PendingSelection p, Random rng)
    {
        while (!p.CanFinish && p.Remaining.Count > 0)
        {
            var r = p.Remaining;
            p.Pick(r[rng.Next(r.Count)]);
        }
        if (CardSelect.Pending == p) p.Finish();
    }

    /// <summary>
    /// Reward policy for the diagnostic: take everything on offer, and pick a card at random.
    ///
    /// Deliberately not clever. This phase measures whether the reward machinery works and
    /// how a deck grows over a run; a good card policy is the run model's job, and baking a
    /// heuristic in here would make the later comparison meaningless.
    /// </summary>
    private static void InstallDiagnosticRewardPolicy(Random rng)
    {
        RunRewards.ChooseRewards = (player, options) =>
            Task.FromResult(Enumerable.Range(0, options.Count).ToList());

        RunRewards.ChooseCard = (player, offered) =>
        {
            var cards = offered.Where(o => o.Kind == "card").ToList();
            if (cards.Count == 0) return Task.FromResult(new List<CardRewardOption>());
            return Task.FromResult(new List<CardRewardOption> { cards[rng.Next(cards.Count)] });
        };
    }

    /// <summary>
    /// random: uniform over travelable points. Deliberately NOT a good path policy - this
    /// phase measures whether the transitions work, and a random walk visits every room type
    /// rather than the handful a sensible policy would prefer.
    /// </summary>
    private static RunAction ChoosePath(string policy, List<RunAction> legal, Random rng) => policy switch
    {
        "left" => legal.First(),
        "right" => legal.Last(),
        _ => legal[rng.Next(legal.Count)],
    };

    private static void Note(Tally t, Exception e)
    {
        var s = e.GetType().Name + ": " + e.Message;
        if (!t.Errors.Contains(s)) t.Errors.Add(s);
    }

    /// <summary>
    /// Awaits a task with a deadline. A hang inside the engine's async chain cannot be
    /// cancelled, so this only DETECTS one - the phase then abandons that run rather than
    /// wedging the process, which is what a diagnostic needs to do.
    /// </summary>
    private static async Task<bool> WithTimeout(Task task, int ms)
    {
        var done = await Task.WhenAny(task, Task.Delay(ms));
        if (done != task) return false;
        await task;   // surface the exception if it faulted
        return true;
    }
}
