using System;
using System.Reflection;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;

namespace AlphaSlayer.Probe;

/// <summary>
/// A fight restore tears the combat down mid-turn (RunManager.CleanUp -> CombatManager.Reset,
/// which cancels the turn state). An engine continuation still in flight for the OLD turn state
/// (EndPlayerTurnPhaseOneInternal awaiting BeforeSideTurnEnd, say) resumes after the restore and
/// calls CheckWinCondition(oldTurnState). The engine means that to be a no-op for a dead combat,
/// but two paths fire the GLOBAL events against whatever combat is live now:
///
/// - ProcessPendingLoss invokes CombatEnded unconditionally. When a search candidate's player
///   died in its end-of-turn, NetCombatCardDb's handler (subscribed by the restored combat's
///   StartCombat) then drops every pile subscription of the NEW combat; the next card drawn is
///   never given an id and playing it throws "Card X could not be found in combat ID database!",
///   which kills the run (~3.6% of runs with elite+boss search, 0.5% of run-search rollouts).
/// - EndCombatInternal(oldTurnState) runs the old combat's AfterCombatEnd/Victory hooks and then
///   invokes CombatWon/CombatEnded because SOME turn state (the new one) is live.
///
/// (A third stale continuation, Decimillipede's ReattachPower fade-out, segfaults instead: see
/// NodeGuard._unguardedPresentation.)
///
/// Neither of these can happen in the shipped game (nothing replaces a combat mid-turn). A stale
/// turn state's loss is marked processed without the event; its win is skipped (its world is gone).
/// --probe-no-stale-guard turns this off.
/// </summary>
public static class StaleCombat
{
    public static int Skipped;
    private static readonly FieldInfo TurnStateField = AccessTools.Field(typeof(CombatManager), "_turnState");
    private static Type _ts;
    private static PropertyInfo _pendingLoss, _ct, _inProgress;

    public static void Install()
    {
        if (Probe.HasFlag("probe-no-stale-guard")) return;
        _ts = TurnStateField.FieldType;
        _pendingLoss = AccessTools.Property(_ts, "PendingLoss");
        _ct = AccessTools.Property(_ts, "Ct");
        _inProgress = AccessTools.Property(_ts, "IsInProgress");
        var h = new Harmony("alphaslayer.probe.stalecombat");
        h.Patch(AccessTools.Method(typeof(CombatManager), "ProcessPendingLoss", new[] { _ts }),
            prefix: new HarmonyMethod(typeof(StaleCombat), nameof(LossPrefix)));
        h.Patch(AccessTools.Method(typeof(CombatManager), "EndCombatInternal", new[] { _ts }),
            prefix: new HarmonyMethod(typeof(StaleCombat), nameof(WinPrefix)));
        Probe.Log("stale combat guard installed");
    }

    private static bool IsStale(CombatManager mgr, object turnState)
    {
        if (turnState == null) return false;
        if (!ReferenceEquals(TurnStateField.GetValue(mgr), turnState)) return true;
        return _ct?.GetValue(turnState) is System.Threading.CancellationToken ct && ct.IsCancellationRequested;
    }

    private static bool LossPrefix(CombatManager __instance, object turnState)
    {
        if (!IsStale(__instance, turnState)) return true;
        Skipped++;
        // what the original does, minus the global CombatEnded event
        _pendingLoss.SetValue(turnState, null);
        _inProgress?.SetValue(turnState, false);
        return false;
    }

    private static bool WinPrefix(CombatManager __instance, object turnState, ref Task __result)
    {
        if (!IsStale(__instance, turnState)) return true;
        Skipped++;
        __result = Task.CompletedTask;
        return false;
    }
}
