using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Rngs;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace AlphaSlayer.Probe;

/// <summary>
/// Run-level search (shop, card reward, route): which option leaves the run best placed to
/// beat the next act boss.
///
/// LIVE side (--probe-run-search=card_reward,shop,travel, needs --probe-fight-restore): every
/// decision since the last room entry is logged (RunServe._runLog). A searched decision carries
/// "ctx" = {save: that room-entry save, log}; the server answers it after ROLLOUTS of each option.
///
/// ROLLOUT side: a job {"rollout": {restore, forced, world, boss_plays}} restores the save,
/// answers the first decisions from "forced" (the log, then the option being valued; the same
/// replay the fight restore proved exact), and the moment the last forced answer is given the
/// hidden future is re-rolled from "world" (StartWorld): every run, player and combat RNG, and
/// the unvisited encounter and event queues of every act. A search must not see the real next
/// card rewards, shop stock or which fight sits on which node; it sees the same pools a player
/// knows. The policy then plays on until the NEXT act boss fight (one that starts after the
/// option), which is played --boss_plays times from its room-entry save (each with its own
/// draw order and combat RNGs). The terminal reports the playouts; a death before the boss is a
/// lost rollout. The live game never sees any of this: rollouts run in other processes.
/// </summary>
public static class RunSearch
{
    private static HashSet<string> _kinds;

    public static bool Wants(string kind, int count)
    {
        _kinds ??= new HashSet<string>((Probe.ArgValue("probe-run-search") ?? "")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
        return count > 1 && FightRestore.On && FightSearch.RunEnabled && _kinds.Contains(kind);
    }

    /// <summary>The current job's rollout spec (null: an ordinary run).</summary>
    public sealed class Spec
    {
        public string Restore;
        public int[] Forced;
        public int World;
        public int BossPlays;
        /// <summary>Exactness check: replay "forced" (the log alone), send the decision it
        /// lands on WITHOUT re-rolling anything, and stop. The server compares it with the live
        /// decision.</summary>
        public bool Verify;
        /// <summary>The live decision the last forced answer belongs to (kind, option count):
        /// a replay that lands anywhere else is marked ReplayError and its leaf discarded.</summary>
        public string ExpectKind;
        /// <summary>After the option's room is resolved, go straight to the act boss (the
        /// deck, relics and hp as they are now) instead of playing the floors in between.</summary>
        public bool Teleport;
        public int ExpectN = -1;
    }

    public static Spec Job;
    public static bool Active => Job != null;
    public static bool WorldStarted, Done, ReplayError;
    public static int BossWon, BossPlayed;
    public static List<int> BossHp = new();
    public static List<string> BossOutcomes = new();
    public static List<int> BossTurns = new();
    public static List<double> BossDmg = new();
    public static string Boss = "";
    public static int HpAtBoss, DeckAtBoss;

    public static void Reset(Spec job)
    {
        Job = job;
        WorldStarted = Done = ReplayError = false;
        BossWon = BossPlayed = 0;
        BossHp = new List<int>();
        BossOutcomes = new List<string>();
        BossTurns = new List<int>();
        BossDmg = new List<double>();
        Boss = "";
    }

    private static ulong Mix(int world, int salt) =>
        unchecked((ulong)(uint)world * 0x9E3779B97F4A7C15UL + (ulong)(uint)salt * 0xBF58476D1CE4E5B9UL + 0x632BE59BD9B4E019UL);

    /// <summary>Re-rolls everything the player cannot know, from the world seed.</summary>
    public static void StartWorld(RunState rs)
    {
        WorldStarted = true;
        if (rs == null) return;
        var w = Job.World;
        var i = 0;
        foreach (var t in Enum.GetValues<RunRngType>())
            rs.Rng.MockRng(t, Mix(w, 100 + i++));
        foreach (var p in rs.Players)
        {
            var rngs = Traverse.Create(p.PlayerRng).Field("_rngs").GetValue<Dictionary<PlayerRngType, Rng>>();
            if (rngs == null) continue;
            foreach (var t in rngs.Keys.ToList())
                rngs[t] = new Rng(Mix(w, 200 + (int)t));
        }
        var shuffle = new Random(unchecked((int)Mix(w, 300)));
        foreach (var act in rs.Acts)
        {
            var rooms = Traverse.Create(act).Field("_rooms").GetValue<RoomSet>();
            if (rooms == null) continue;
            ShuffleTail(rooms.normalEncounters, rooms.normalEncountersVisited, shuffle);
            ShuffleTail(rooms.eliteEncounters, rooms.eliteEncountersVisited, shuffle);
            ShuffleTail(rooms.events, rooms.eventsVisited, shuffle);
        }
    }

    /// <summary>Shuffles the entries not yet reached (the queues are read at visited % count).</summary>
    private static void ShuffleTail<T>(List<T> list, int visited, Random rng)
    {
        if (list.Count < 2) return;
        var start = visited < list.Count ? visited : 0;
        for (var k = list.Count - 1; k > start; k--)
        {
            var j = start + rng.Next(k - start + 1);
            (list[k], list[j]) = (list[j], list[k]);
        }
    }

    /// <summary>The rollout's horizon: this boss fight, played Job.BossPlays times.</summary>
    public static bool IsHorizon(CombatRoom cr) =>
        Active && WorldStarted && cr.RoomType == RoomType.Boss;

    /// <summary>The fraction of the enemies' total hp removed (1 = all dead).</summary>
    public static double DamageFraction(CombatState st)
    {
        if (st == null) return 0;
        double cur = 0, max = 0;
        foreach (var e in st.Enemies)
        {
            cur += Math.Max(0, e.CurrentHp);
            max += Math.Max(1, e.MaxHp);
        }
        return max > 0 ? 1 - cur / max : 0;
    }

    public static Dictionary<string, object> Report(RunHarness h) => new()
    {
        ["reached_boss"] = BossPlayed > 0, ["boss"] = Boss, ["boss_plays"] = BossPlayed,
        ["boss_won"] = BossWon, ["boss_hp"] = BossHp.ToArray(),
        ["boss_outcomes"] = BossOutcomes.ToArray(), ["boss_turns"] = BossTurns.ToArray(),
        ["boss_dmg"] = BossDmg.Select(x => Math.Round(x, 4)).ToArray(),
        ["hp_at_boss"] = HpAtBoss, ["deck"] = DeckAtBoss, ["max_hp"] = h.Player.Creature.MaxHp,
        ["act"] = h.ActIndex, ["replay_error"] = ReplayError, ["world_started"] = WorldStarted,
    };
}
