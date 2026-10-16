using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Entities.Rngs;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace AlphaSlayer.Probe;

/// <summary>
/// Settings and counters for fight search (RunServe.SearchAsync / SearchTurnAsync).
///
///   --probe-search=Boss,Elite   room types to search in (needs --probe-fight-restore)
///   --probe-search-mode=turn    "turn": the policy's top choices, each played out to the END OF
///                               THE TURN by the policy (rollout) in several sampled worlds;
///                               "ply": every choice one step deep (the first version)
///   --probe-search-k=4          turn: at most this many candidates, by the policy's prior ...
///   --probe-search-mass=0.95    ... stopping once they cover this much prior probability
///   --probe-search-samples=2    turn: sampled worlds per candidate (shared by all candidates:
///                               common random numbers, so the comparison is paired)
///   --probe-search-roll-cap=40  turn: rollout choices before giving up on reaching end_turn
///   --probe-search-cands=12     ply: legal choices valued per decision (in legal-list order)
///
/// Hidden information: a simulated choice that draws cards would otherwise see the REAL next
/// cards (the restore reproduces the real RNG). Each simulation first shuffles the draw pile
/// with its own RNG, so draws inside the search are a sample, as a player would reason.
/// In turn mode each sample also replaces the restored run's combat RNGs (MockCombatRngs):
/// discard reshuffles, enemy AI rolls, random targets, generated cards and costs are sampled
/// too, so a rollout through the enemy turn cannot see the real rolls. (ply mode leaves them
/// real.) The live game never sees a mocked RNG: every search ends with a restore.
/// </summary>
public static class FightSearch
{
    private static HashSet<string> _rooms;
    public static int MaxCands => Probe.ArgInt("probe-search-cands", 12);
    public static bool TurnMode => (Probe.ArgValue("probe-search-mode") ?? "turn") == "turn";
    public static int K => Probe.ArgInt("probe-search-k", 4);
    public static double Mass => double.TryParse(Probe.ArgValue("probe-search-mass"),
        System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var m) ? m : 0.95;
    public static int Samples => Probe.ArgInt("probe-search-samples", 2);
    public static int RollCap => Probe.ArgInt("probe-search-roll-cap", 40);
    /// <summary>--probe-search-yield-every: game waits per real frame while searching (restores,
    /// replays, rollouts); live play keeps --probe-yield-every. The live-state check after each
    /// search (Diverged) is what shows whether skipping frames changes the game.</summary>
    public static int YieldEvery => Probe.ArgInt("probe-search-yield-every", 0);

    /// <summary>--probe-search-frame-suspend: the harness's polling waits one frame instead of
    /// a 1 ms timer while searching (CombatHarness.FrameSuspend).</summary>
    public static bool FrameSuspend => Probe.HasFlag("probe-search-frame-suspend");
    /// <summary>The current run's job may switch search off ("search": false).</summary>
    public static bool RunEnabled = true;
    public static long Searches, Leaves, Diverged, Errors, Rolls, RollCapped, Retries, ForcedRolls;
    public static double Ms, LoadMs, RestoreMs, ReplayChoices, RollMs, EvalMs, PriorMs;
    /// <summary>Rollout steps split: observing (obs + legal), waiting for the server's answer,
    /// and applying it until the next decision.</summary>
    public static double RollObsMs, RollAskMs, RollApplyMs;

    private static readonly RunRngType[] CombatRngs =
    {
        RunRngType.Shuffle, RunRngType.MonsterAi, RunRngType.CombatTargets, RunRngType.CombatCardGeneration,
        RunRngType.CombatPotionGeneration, RunRngType.CombatCardSelection, RunRngType.CombatEnergyCosts,
        RunRngType.CombatOrbs,
    };

    /// <summary>Gives the (restored, throwaway) run fresh combat RNGs drawn from a sample seed.</summary>
    public static void MockCombatRngs(RunState rs, int seed)
    {
        for (var i = 0; i < CombatRngs.Length; i++)
            rs.Rng.MockRng(CombatRngs[i], unchecked((ulong)(uint)seed * 0x9E3779B97F4A7C15UL + (ulong)i * 0xBF58476D1CE4E5B9UL));
    }

    public static bool Wants(CombatRoom cr)
    {
        _rooms ??= new HashSet<string>((Probe.ArgValue("probe-search") ?? "")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
            StringComparer.OrdinalIgnoreCase);
        return RunEnabled && _rooms.Count > 0 && FightRestore.On && _rooms.Contains(cr.RoomType.ToString());
    }

    public static void ShuffleDrawPile(Player player, int seed)
    {
        var cards = Traverse.Create(player.PlayerCombatState.DrawPile).Field("_cards").GetValue<List<CardModel>>();
        if (cards == null || cards.Count < 2) return;
        var rng = new Random(seed);
        for (var i = cards.Count - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (cards[i], cards[j]) = (cards[j], cards[i]);
        }
    }

    public static void WriteStats()
    {
        if (_rooms == null || _rooms.Count == 0) return;
        File.WriteAllText(Path.Combine(Probe.OutDir, $"search_{Environment.ProcessId}.json"), Probe.ToJsonCompact(new Dictionary<string, object>
        {
            ["searches"] = Searches, ["leaves"] = Leaves, ["diverged"] = Diverged, ["errors"] = Errors, ["retries"] = Retries,
            ["ms"] = Math.Round(Ms), ["restore_ms"] = Math.Round(RestoreMs), ["load_ms"] = Math.Round(LoadMs), ["replay_choices"] = ReplayChoices,
            ["roll_ms"] = Math.Round(RollMs), ["eval_ms"] = Math.Round(EvalMs), ["prior_ms"] = Math.Round(PriorMs),
            ["roll_obs_ms"] = Math.Round(RollObsMs), ["roll_ask_ms"] = Math.Round(RollAskMs), ["roll_apply_ms"] = Math.Round(RollApplyMs),
            ["rolls"] = Rolls, ["forced_rolls"] = ForcedRolls, ["roll_capped"] = RollCapped,
        }));
    }
}
