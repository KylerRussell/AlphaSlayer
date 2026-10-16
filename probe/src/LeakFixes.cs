using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Game;

namespace AlphaSlayer.Probe;

/// <summary>
/// Leaks that made search slow down with a process's age (LeakAudit, 1,200 restores of one
/// fight: Godot nodes 1,580 -> 17,224, replay ~2.5 -> ~7 ms per choice; in the turn-search eval
/// a process that had done ~1,500 searches replayed at ~140 ms per choice instead of ~2):
///
/// 1. NDebugAudioManager.Play parks every sound on a new AudioStreamPlayer node until its
///    Finished signal, which the headless dummy audio driver never emits: ~13 nodes per
///    restore (turn banners, card plays), ~2 per turn in ordinary play. Every frame walks
///    them. Play is skipped (it returns no id; Stop of an unknown id only logs). It consumes
///    Rng.Chaotic only for pitch, and that RNG drives cosmetics, never game state.
/// 2. NetCombatCardDb subscribes to every card pile at combat start and lets go at
///    CombatEnded, which a restore (CleanUp mid-fight) never fires: the old combat's piles,
///    cards and a CombatEnded handler survive every restore. Released before the teardown.
/// 3. NetLoadingHandle keeps a load count per net service, and every restore makes a new one;
///    the settled (0) entries are dropped.
///
/// --probe-no-leak-fixes turns all three off (to measure them).
/// </summary>
public static class LeakFixes
{
    public static bool On => !Probe.HasFlag("probe-no-leak-fixes");
    private static bool _installed;

    public static void Install()
    {
        if (_installed || !On) return;
        _installed = true;
        var h = new Harmony("alphaslayer.probe.leakfixes");
        h.Patch(AccessTools.Method(typeof(NDebugAudioManager), nameof(NDebugAudioManager.Play)),
            prefix: new HarmonyMethod(typeof(LeakFixes), nameof(SkipPlay)));
    }

    private static bool SkipPlay(ref int __result)
    {
        __result = -1;
        return false;
    }

    private static readonly FieldInfo Subs = AccessTools.Field(typeof(NetCombatCardDb), "_subscriptions");
    private static readonly MethodInfo OnCombatEnded = AccessTools.Method(typeof(NetCombatCardDb), "OnCombatEnded");
    private static readonly FieldInfo LoadCounts = AccessTools.Field(typeof(NetLoadingHandle), "_loadCounts");

    /// <summary>Before a mid-fight teardown (restore): let go of what only CombatEnded frees.</summary>
    public static void BeforeTeardown()
    {
        if (!On) return;
        try
        {
            var db = NetCombatCardDb.Instance;
            if (Subs?.GetValue(db) is ICollection { Count: > 0 })
                OnCombatEnded?.Invoke(db, new object[] { null });
        }
        catch (Exception e) { Probe.Log($"  (non-fatal) NetCombatCardDb release: {e.Message}"); }
    }

    /// <summary>After a restore: drop the settled load counts of dead net services.</summary>
    public static void AfterRestore()
    {
        if (!On) return;
        try
        {
            if (LoadCounts?.GetValue(null) is IDictionary d && d.Count > 1)
            {
                var dead = d.Keys.Cast<object>().Where(k => d[k] is int n && n == 0).ToList();
                foreach (var k in dead) d.Remove(k);
            }
        }
        catch (Exception e) { Probe.Log($"  (non-fatal) NetLoadingHandle release: {e.Message}"); }
    }
}
