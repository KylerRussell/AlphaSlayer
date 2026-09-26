using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace AlphaSlayer.Probe;

/// <summary>
/// Phase 2 - the number that sets the project schedule: how many combat decisions per second
/// the rules engine sustains headless.
///
/// Two deliberate choices:
///
/// 1. We do NOT use AutoSlay. Its WaitHelper.Until polls on Task.Delay(100ms) of real
///    wall-clock, which NonInteractiveMode does not bypass, so it is capped at roughly human
///    speed by construction.
///
/// 2. We do NOT use RunManager.EnterRoomDebug / NRun.Create. Those instantiate Godot scenes
///    and SEGFAULT under --headless (the dummy renderer cannot build the run scene). Instead
///    we replicate CombatRoom.StartCombat directly against CombatState/CombatManager, which
///    is the whole point of Option A: the rules engine without the node layer. Every node
///    touch in that path is null-conditional (NRun.Instance?, NCombatRoom.Instance?), so it
///    is safe with no scene at all.
///
/// The key diagnostic is actions-per-FRAME, not just actions-per-second. Every `await` yields
/// to Godot's main loop; if the engine only retires ~1 action per frame we are frame-bound,
/// and the fix is batching rather than a faster CPU.
/// </summary>
public static class Throughput
{
    public static async Task RunAsync()
    {
        var r = new Dictionary<string, object>();

        var combatsTarget = Probe.ArgInt("probe-combats", 20);
        var turnCap = Probe.ArgInt("probe-turn-cap", 50);
        var ascension = Probe.ArgInt("probe-ascension", 0);
        var seed = Probe.ArgValue("probe-seed") ?? "ALPHASLAYER1";

        r["combats_target"] = combatsTarget;
        r["turn_cap"] = turnCap;
        r["ascension"] = ascension;
        r["seed"] = seed;
        r["scene_layer"] = "bypassed";

        // --- make the engine run unthrottled -------------------------------------------------
        // AutoSlayerCheck is a public settable Func<bool> that NonInteractiveMode consults.
        // Setting it true makes every Cmd.Wait() a no-op without flipping TestMode, which has
        // much broader semantics (it also disables real logic in MapCmd, CardPileCmd, etc).
        // TestMode is the node-layer bypass. Every visual path in the gameplay commands is
        // gated on TestMode.IsOff - e.g. CardPileCmd.AddDuringManualCardPlay only calls
        // CreateCardNodeAndUpdateVisuals (which instantiates an NCard scene, and segfaults
        // under the headless dummy renderer) when TestMode is off. This is the switch
        // MegaCrit's own unit tests use to run combat without a scene.
        MegaCrit.Sts2.Core.TestSupport.TestMode.IsOn = true;

        var natural = Probe.HasFlag("probe-natural");
        r["mode"] = natural ? "natural" : "waits-patched";
        r["test_mode"] = MegaCrit.Sts2.Core.TestSupport.TestMode.IsOn;
        if (!natural) NonInteractiveMode.AutoSlayerCheck = () => true;
        PreloadManager.Enabled = false;
        r["non_interactive"] = NonInteractiveMode.IsActive;
        Probe.Log($"NonInteractiveMode.IsActive={NonInteractiveMode.IsActive}, preloading off");

        // --- build a run WITHOUT any scene ---------------------------------------------------
        var wanted = Probe.ArgValue("probe-character");
        var character = ModelDb.AllCharacters.First(c => c.IsPlayable
            && (wanted == null || string.Equals(c.Id.Entry, wanted, StringComparison.OrdinalIgnoreCase)));
        r["character"] = character.Id.Entry;

        var acts = ActModel.GetDefaultList().Select(a => a.ToMutable()).ToList();
        r["act_ids"] = acts.Select(a => a.Id.Entry).ToArray();

        var unlocks = SaveManager.Instance.GenerateUnlockStateFromProgress();
        var player = Player.CreateForNewRun(character, unlocks, 1uL);
        var runState = RunState.CreateForNewRun(
            new List<Player> { player }, acts, Array.Empty<ModifierModel>(),
            GameMode.Standard, ascension, seed);

        RunManager.Instance.SetUpNewSingleplayer(runState, shouldSave: false);
        RunManager.Instance.Launch();
        Probe.Log($"run launched (no scene): {character.Id.Entry} A{ascension} seed={seed}");

        var act0 = acts[0];
        var encounters = act0.AllRegularEncounters.ToList();
        if (encounters.Count == 0) throw new InvalidOperationException("no regular encounters in act 0");
        r["encounter_pool"] = encounters.Count;

        // --- measurement loop ----------------------------------------------------------------
        var queue = RunManager.Instance.ActionQueueSet;
        long totalCards = 0, totalTurns = 0, totalCombats = 0, totalActions = 0;
        var perCombat = new List<object>();

        var frames0 = Engine.GetProcessFrames();
        var sw = Stopwatch.StartNew();

        for (var i = 0; i < combatsTarget; i++)
        {
            var enc = encounters[i % encounters.Count].ToMutable();
            var cSw = Stopwatch.StartNew();
            var cFrames0 = Engine.GetProcessFrames();
            string outcome;
            long cards = 0, actions = 0; int turns = 0;

            try
            {
                // Replicates CombatRoom.EnterInternal + StartCombat, minus the node layer.
                Step("new CombatState");
                var combat = new CombatState(enc, runState, runState.Modifiers, null, null);
                Step("AddPlayer");
                combat.AddPlayer(player);
                Step("GenerateMonstersWithSlots");
                if (!enc.HaveMonstersBeenGenerated) enc.GenerateMonstersWithSlots(runState);
                Step($"monsters={enc.MonstersWithSlots.Count()}");
                foreach (var (monster, slot) in enc.MonstersWithSlots)
                {
                    Step($"CreateCreature {monster.Id.Entry}");
                    var creature = combat.CreateCreature(monster, CombatSide.Enemy, slot);
                    Step($"AddCreature {monster.Id.Entry}");
                    combat.AddCreature(creature);
                }

                Step("SetUpCombat");
                CombatManager.Instance.SetUpCombat(combat);
                Step("AfterCombatRoomLoaded");
                CombatManager.Instance.AfterCombatRoomLoaded();
                Step("Unpause");
                RunManager.Instance.ActionExecutor.Unpause();
                Step("DriveCombat");

                (cards, turns, actions, outcome) = await DriveCombat(player, queue, turnCap);
            }
            catch (Exception e)
            {
                outcome = $"exception: {e.GetType().Name}: {e.Message}";
                Probe.Log($"combat {i + 1} threw: {e}");
            }

            cSw.Stop();
            totalCards += cards; totalTurns += turns; totalActions += actions; totalCombats++;

            perCombat.Add(new Dictionary<string, object>
            {
                ["encounter"] = enc.Id.Entry,
                ["cards"] = cards,
                ["turns"] = turns,
                ["actions"] = actions,
                ["outcome"] = outcome,
                ["ms"] = cSw.Elapsed.TotalMilliseconds,
                ["frames"] = (long)(Engine.GetProcessFrames() - cFrames0),
            });
            Probe.Log($"combat {i + 1}/{combatsTarget} {enc.Id.Entry}: {turns}t {cards}c " +
                      $"{cSw.Elapsed.TotalMilliseconds:F1}ms -> {outcome}");

            Try(() => CombatManager.Instance.Reset(graceful: true), "CombatManager.Reset");
            // Fresh HP each combat: we are measuring engine throughput, and a greedy policy
            // otherwise dies early and biases the sample toward very short fights.
            Try(() => player.Creature.HealInternal(player.Creature.MaxHp), "reset hp");
        }

        sw.Stop();
        var frames = (long)(Engine.GetProcessFrames() - frames0);
        var secs = sw.Elapsed.TotalSeconds;

        r["combats"] = totalCombats;
        r["turns"] = totalTurns;
        r["cards_played"] = totalCards;
        r["actions"] = totalActions;
        r["wall_s"] = secs;
        r["frames"] = frames;
        r["combats_per_s"] = secs > 0 ? totalCombats / secs : 0;
        r["turns_per_s"] = secs > 0 ? totalTurns / secs : 0;
        r["cards_per_s"] = secs > 0 ? totalCards / secs : 0;
        r["actions_per_s"] = secs > 0 ? totalActions / secs : 0;
        // >1 means we retire multiple decisions per frame; ~1 means we are frame-bound.
        r["actions_per_frame"] = frames > 0 ? (double)totalActions / frames : 0;
        r["fps"] = secs > 0 ? frames / secs : 0;
        r["per_combat"] = perCombat;

        if (secs > 0 && totalCards > 0)
        {
            var cps = totalCards / secs;
            r["single_proc_days_for_1e8_card_plays"] = 1e8 / cps / 86400.0;
            r["single_proc_days_for_1e9_card_plays"] = 1e9 / cps / 86400.0;
        }

        Probe.Report["throughput"] = r;
        Probe.Log($"THROUGHPUT: {totalCombats} combats, {totalTurns} turns, {totalCards} cards " +
                  $"in {secs:F2}s over {frames} frames");
        Probe.Log($"  cards/s={r["cards_per_s"]:F1} actions/frame={r["actions_per_frame"]:F2} fps={r["fps"]:F1}");
    }

    /// <summary>
    /// Trivial greedy policy: play every playable card, then end turn. We are measuring the
    /// engine, not playing well - the policy only has to exercise the code paths an agent would.
    ///
    /// Turn model (this is what the first version got wrong): the player may only act during
    /// PlayerTurnPhase.Play. Ending a turn is a two-phase, ENGINE-driven transition -
    /// SetReadyToEndTurn completes an EndTurnSignal, the turn loop runs phase one, then
    /// enqueues ReadyToBeginEnemyTurnAction itself. So the driver must not poll TurnNumber
    /// right after ending; it must wait for Phase to come back to Play on a real suspension
    /// that lets the engine's continuations drain.
    /// </summary>
    private static async Task<(long cards, int turns, long actions, string outcome)> DriveCombat(
        Player player, MegaCrit.Sts2.Core.GameActions.Multiplayer.ActionQueueSet queue, int turnCap)
    {
        long cards = 0, actions = 0;
        var turns = 0;

        var expectTurn = player.PlayerCombatState?.TurnNumber ?? 1;

        while (turns < turnCap)
        {
            // Gate on the turn NUMBER, not just the phase: right after SetReadyToEndTurn the
            // phase is still Play (the engine has not processed the end yet), so a
            // phase-only check re-enters the same turn and double-ends it.
            var over = await WaitForPlayPhase(player, expectTurn);
            if (over != null) return (cards, turns, actions, over);

            var pcs = player.PlayerCombatState;
            var guard = 0;
            while (guard++ < 100)
            {
                var st = CombatManager.Instance.DebugOnlyGetState();
                if (st == null || CombatFinished()) break;
                if (player.Creature.IsDead || !st.HittableEnemies.Any()) break;
                if (pcs.Phase != PlayerTurnPhase.Play) break;

                var (card, target) = PickPlay(pcs.Hand.Cards.ToList(), st.HittableEnemies.ToList());
                if (card == null) break;

                queue.EnqueueWithoutSynchronizing(new PlayCardAction(card, target));
                actions++;
                await queue.BecameEmpty();
                cards++;
            }

            if (CombatFinished())
                return (cards, turns, actions, Outcome(player));
            if (player.Creature.IsDead) return (cards, turns, actions, "player_dead");

            expectTurn = pcs.TurnNumber + 1;
            CombatManager.Instance.SetReadyToEndTurn(player, canBackOut: false, actionDuringEnemyTurn: null);
            actions++;
            turns++;
        }
        return (cards, turns, actions, "turn_cap");
    }

    /// <summary>
    /// Waits until the player may act again, or combat is over. Returns null when it is the
    /// player's Play phase, otherwise the terminal outcome string.
    /// </summary>
    private static async Task<string> WaitForPlayPhase(Player player, int minTurnNumber)
    {
        var budgetMs = Probe.ArgInt("probe-turn-timeout-ms", 5000);
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < budgetMs)
        {
            var st = CombatManager.Instance.DebugOnlyGetState();
            if (st == null) return "combat_ended";
            if (player.Creature.IsDead) return "player_dead";
            if (!st.HittableEnemies.Any()) return "enemies_cleared";
            if (CombatFinished()) return Outcome(player);

            var pcs = player.PlayerCombatState;
            if (pcs != null && pcs.Phase == PlayerTurnPhase.Play
                && st.CurrentSide == CombatSide.Player && pcs.TurnNumber >= minTurnNumber)
                return null;

            // A REAL suspension. Task.Yield() runs inline on Godot's synchronization context,
            // so it never lets the engine's pending continuations run and the turn never flips.
            await Task.Delay(1);
        }
        return "turn_timeout";
    }

    /// <summary>
    /// True only once combat has really finished. IsOverOrEnding is also true during SETUP,
    /// because it is (IsEnding || !IsInProgress) and IsInProgress stays false until
    /// StartCombatInternal flips it - IsStarting is what separates the two.
    /// </summary>
    private static bool CombatFinished() =>
        CombatManager.Instance.IsOverOrEnding && !CombatManager.Instance.IsStarting;

    private static string Outcome(Player player)
    {
        if (player.Creature.IsDead) return "player_dead";
        var st = CombatManager.Instance.DebugOnlyGetState();
        if (st == null) return "combat_ended";
        return st.HittableEnemies.Any() ? "combat_ended" : "enemies_cleared";
    }

    /// <summary>First playable card, with a legal target if it needs one.</summary>
    private static (CardModel card, Creature target) PickPlay(
        List<CardModel> hand, List<Creature> enemies)
    {
        foreach (var c in hand)
        {
            foreach (var e in enemies)
                if (Safe(() => c.CanPlayTargeting(e))) return (c, e);
            if (Safe(() => c.CanPlay())) return (c, null);
        }
        return (null, null);
    }

    private static void Step(string s) => Probe.Log($"    step: {s}");

    /// <summary>
    /// A genuine suspension point. Task.Yield() is not one here: Godot's
    /// SynchronizationContext runs continuations inline from the main thread, so the main
    /// loop never advances and Engine.GetProcessFrames() stays pinned.
    /// </summary>
    private static async Task Frame()
    {
        // Task.Yield() is cheap and, on Godot's synchronization context, runs inline - which
        // is fine here because with TestMode on the combat chain completes synchronously and
        // we are not waiting on frames. --probe-slow-yield restores the 1ms version.
        if (Probe.HasFlag("probe-slow-yield")) await Task.Delay(1);
        else await Task.Yield();
    }

    /// <summary>Waits until it is the player's turn with a live hand, or gives up.</summary>
    private static async Task<bool> WaitForPlayerTurn(Player player, int maxFrames)
    {
        for (var i = 0; i < maxFrames; i++)
        {
            var st = CombatManager.Instance.DebugOnlyGetState();
            if (st == null || player.Creature.IsDead || !st.HittableEnemies.Any()) return false;
            var pcs = player.PlayerCombatState;
            if (pcs != null && st.CurrentSide == CombatSide.Player && pcs.Hand.Cards.Count > 0)
                return true;
            await Frame();
        }
        return false;
    }

    private static bool Safe(Func<bool> f) { try { return f(); } catch { return false; } }

    private static void Try(Action a, string what)
    {
        try { a(); }
        catch (Exception e) { Probe.Log($"  (non-fatal) {what}: {e.GetType().Name}: {e.Message}"); }
    }
}
