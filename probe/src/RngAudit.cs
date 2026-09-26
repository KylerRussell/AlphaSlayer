using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;

namespace AlphaSlayer.Probe;

/// <summary>
/// Answers one question with evidence: does any RNG fire during the player's Play phase?
///
/// The intra-turn exact-search plan rests on "no RNG fires until the end-of-turn draw", so
/// card-play ordering within a turn can be searched exactly with chance nodes only at turn
/// boundaries. If a relic, a debuff on an enemy, or a card effect consumes randomness mid-turn
/// then that is false and the search needs chance nodes inside the turn.
///
/// We patch MegaRandom (the generator underneath Rng, so every path is covered) and bucket
/// every draw by the player's turn phase, attributing Play-phase draws to the game code that
/// caused them.
/// </summary>
public static class RngAudit
{
    [ThreadStatic] private static bool _inHook;
    private static bool _armed;

    public static readonly Dictionary<string, long> ByPhase = new();
    public static readonly Dictionary<string, long> PlayPhaseCallers = new();
    public static readonly Dictionary<string, long> PlayPhaseStreams = new();
    public static long Total;

    /// <summary>
    /// Identity map from Rng instance to the RunRngSet stream it belongs to.
    ///
    /// This is the distinction that decides the search design: RunRngSet keeps SEPARATE,
    /// independently seeded streams per purpose (Shuffle, CombatTargets, MonsterAi, ...), so
    /// a draw from a cosmetic or unrelated generator cannot perturb a gameplay one. Only
    /// draws from a named gameplay stream during the Play phase are real chance events.
    /// </summary>
    private static readonly Dictionary<object, string> _streamNames =
        new(ReferenceEqualityComparer.Instance);

    public static void MapStreams(MegaCrit.Sts2.Core.Runs.RunState run)
    {
        try
        {
            _streamNames.Clear();
            foreach (var t in Enum.GetValues(typeof(MegaCrit.Sts2.Core.Entities.Rngs.RunRngType)))
            {
                try
                {
                    var rng = run.Rng.GetRng((MegaCrit.Sts2.Core.Entities.Rngs.RunRngType)t);
                    if (rng != null) _streamNames[rng] = "run." + t;
                }
                catch { }
            }
            _streamNames[MegaCrit.Sts2.Core.Random.Rng.Chaotic] = "chaotic(cosmetic)";
            Probe.Log($"rng audit: mapped {_streamNames.Count} named streams");
        }
        catch (Exception e) { Probe.Log($"rng stream map failed: {e.Message}"); }
    }

    private static string StreamName(object rng) =>
        rng != null && _streamNames.TryGetValue(rng, out var n) ? n : "unnamed(per-entity or cosmetic)";

    public static void Install()
    {
        var harmony = new Harmony("alphaslayer.probe.rng");
        var prefix = new HarmonyMethod(AccessTools.Method(typeof(RngAudit), nameof(Draw)));
        // Patch the Rng WRAPPER, not MegaRandom, so __instance identifies which stream drew.
        var t = typeof(MegaCrit.Sts2.Core.Random.Rng);
        var names = new[] { "NextBool", "NextInt", "NextUnsignedInt", "NextUnsignedLong",
                            "NextFloat", "NextDouble", "NextGaussianFloat", "NextGaussianDouble",
                            "NextGaussianInt", "NextItem", "WeightedNextItem", "Shuffle" };
        var n = 0;
        foreach (var m in t.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            if (!names.Contains(m.Name)) continue;
            try { harmony.Patch(m, prefix: prefix); n++; } catch { }
        }
        Probe.Log($"rng audit: patched {n} MegaRandom method(s)");
        Probe.Report["rng_methods_patched"] = n;
    }

    public static void Arm(bool on) => _armed = on;

    public static void Draw(object __instance)
    {
        if (!_armed || _inHook) return;
        _inHook = true;
        try
        {
            Total++;
            var phase = CurrentPhase();
            ByPhase[phase] = ByPhase.GetValueOrDefault(phase) + 1;
            if (phase == "Play")
            {
                var stream = StreamName(__instance);
                PlayPhaseStreams[stream] = PlayPhaseStreams.GetValueOrDefault(stream) + 1;
                Attribute();
            }
        }
        catch { }
        finally { _inHook = false; }
    }

    private static string CurrentPhase()
    {
        try
        {
            var cm = CombatManager.Instance;
            if (cm == null || !cm.IsInProgress) return "outside_combat";
            var st = cm.DebugOnlyGetState();
            if (st == null) return "outside_combat";
            if (st.CurrentSide != CombatSide.Player) return "enemy_turn";
            var p = st.Players.FirstOrDefault();
            var pcs = p?.PlayerCombatState;
            return pcs == null ? "no_pcs" : pcs.Phase.ToString();
        }
        catch { return "unknown"; }
    }

    /// <summary>Names the nearest game frames responsible for a Play-phase draw.</summary>
    private static void Attribute()
    {
        try
        {
            var frames = new StackTrace(2, false).GetFrames();
            if (frames == null) return;
            var sig = new List<string>();
            foreach (var f in frames)
            {
                var m = f.GetMethod();
                var dt = m?.DeclaringType?.FullName;
                if (dt == null || !dt.StartsWith("MegaCrit.", StringComparison.Ordinal)) continue;
                if (dt.StartsWith("MegaCrit.Sts2.Core.Random", StringComparison.Ordinal)) continue;
                sig.Add($"{Short(dt)}::{m.Name}");
                if (sig.Count == 3) break;
            }
            var key = sig.Count == 0 ? "<no game frames>" : string.Join(" <- ", sig);
            PlayPhaseCallers[key] = PlayPhaseCallers.GetValueOrDefault(key) + 1;
        }
        catch { }
    }

    private static string Short(string full)
    {
        var i = full.LastIndexOf('.');
        return i < 0 ? full : full.Substring(i + 1);
    }

    public static void Report()
    {
        var play = ByPhase.GetValueOrDefault("Play");
        Probe.Report["rng_total"] = Total;
        Probe.Report["rng_by_phase"] = ByPhase.OrderByDescending(k => k.Value)
            .ToDictionary(k => k.Key, v => (object)v.Value);
        Probe.Report["rng_play_phase_draws"] = play;
        Probe.Report["rng_play_phase_streams"] = PlayPhaseStreams
            .OrderByDescending(k => k.Value).ToDictionary(k => k.Key, v => (object)v.Value);
        Probe.Report["rng_play_phase_callers"] = PlayPhaseCallers
            .OrderByDescending(k => k.Value).Take(25)
            .ToDictionary(k => k.Key, v => (object)v.Value);

        Probe.Log($"RNG AUDIT: {Total} draws total");
        foreach (var kv in ByPhase.OrderByDescending(k => k.Value))
            Probe.Log($"    {kv.Key,-16} {kv.Value}");
        Probe.Log(play == 0
            ? "  => NO RNG during the player's Play phase: intra-turn play order is exactly searchable."
            : $"  => {play} draws DURING Play phase; intra-turn search needs chance nodes. Top causes:");
        Probe.Log("  Play-phase draws BY STREAM (only named run.* streams affect game state):");
        foreach (var kv in PlayPhaseStreams.OrderByDescending(k => k.Value))
            Probe.Log($"    {kv.Value,8}  {kv.Key}");
        Probe.Log("  Play-phase draws BY CALLER:");
        foreach (var kv in PlayPhaseCallers.OrderByDescending(k => k.Value).Take(15))
            Probe.Log($"    {kv.Value,8}  {kv.Key}");
    }
}
