using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using HarmonyLib;

namespace AlphaSlayer.Probe;

/// <summary>
/// Targeted runaway-loop detector.
///
/// The broad recursion guard (27k methods) never fired for the Defect crash, so instead of
/// patching everything this watches a short list of methods on the suspected cycle and counts
/// invocations *per combat*. A runaway shows up as one method being entered thousands of times
/// in a single fight; the first time a counter crosses the threshold we dump a managed stack,
/// which names the cards and models involved.
///
/// Counting invocations rather than stack depth matters here: an `async Task` method runs
/// synchronously only until its first await, so prefix/postfix pairs cannot measure nesting
/// reliably, but every entry is still counted.
/// </summary>
public static class LoopAudit
{
    [ThreadStatic] private static bool _in;
    private static readonly Dictionary<string, long> _counts = new();
    private static readonly Dictionary<string, long> _peak = new();
    private static bool _dumped;
    public static int Threshold = 400;
    public static bool Armed;

    private static readonly (Type type, string method)[] Watch =
    {
        (typeof(MegaCrit.Sts2.Core.Hooks.Hook), "AfterCardEnteredCombat"),
        (typeof(MegaCrit.Sts2.Core.Commands.CardCmd), "Afflict"),
        (typeof(MegaCrit.Sts2.Core.Commands.CardCmd), "ClearAffliction"),
        (typeof(MegaCrit.Sts2.Core.Combat.CombatState), "AddCard"),
        (typeof(MegaCrit.Sts2.Core.Combat.CombatState), "CloneCard"),
        (typeof(MegaCrit.Sts2.Core.Combat.CombatState), "RemoveCard"),
        (typeof(MegaCrit.Sts2.Core.Combat.CombatStateTracker), "Subscribe"),
    };

    /// <summary>
    /// Diagnostic for the Defect crash. CombatStateTracker.NotifyCombatStateChanged either
    /// no-ops (TestMode on, no subscribers), THROWS (TestMode on, someone subscribed), or
    /// defers (TestMode off). An exception escaping here into native Godot callback code
    /// would present exactly as the segfault we see, so log which branch is actually taken.
    /// </summary>
    public static void InstallNotifyProbe()
    {
        var harmony = new Harmony("alphaslayer.probe.notify");
        var t = typeof(MegaCrit.Sts2.Core.Combat.CombatStateTracker);
        var m = AccessTools.Method(t, "NotifyCombatStateChanged");
        if (m == null) { Probe.Log("notify probe: method not found"); return; }
        harmony.Patch(m, prefix: new HarmonyMethod(AccessTools.Method(typeof(LoopAudit), nameof(NotifyPrefix))));
        Probe.Log("notify probe installed");
    }

    private static int _notifyLogs;

    public static void NotifyPrefix(object __instance)
    {
        // Log only the ANOMALOUS case: TestMode off, or someone subscribed. In the normal
        // case this method is a no-op, so if the crash is really inside it then one of those
        // two must have become true by then.
        try
        {
            var testMode = MegaCrit.Sts2.Core.TestSupport.TestMode.IsOn;
            var fld = __instance.GetType().GetField("CombatStateChanged",
                BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            var del = fld?.GetValue(__instance) as Delegate;
            var subs = del?.GetInvocationList().Length ?? 0;
            if (testMode && subs == 0) return;   // normal, silent
            _notifyLogs++;
            if (_notifyLogs > 20) return;
            Probe.Log($"NOTIFY-ANOMALY[{_notifyLogs}] TestMode.IsOn={testMode} subscribers={subs}");
            if (subs > 0 && del != null)
                foreach (var d in del.GetInvocationList())
                    Probe.Log($"    subscriber: {d.Method.DeclaringType?.FullName}::{d.Method.Name}");
        }
        catch (Exception e) { Probe.Log($"notify probe error: {e.Message}"); }
    }

    public static void Install()
    {
        var harmony = new Harmony("alphaslayer.probe.loop");
        var prefix = new HarmonyMethod(AccessTools.Method(typeof(LoopAudit), nameof(Enter)));
        var n = 0;
        foreach (var (type, name) in Watch)
        {
            foreach (var m in type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic
                                            | BindingFlags.Static | BindingFlags.Instance
                                            | BindingFlags.DeclaredOnly))
            {
                if (m.Name != name || m.ContainsGenericParameters) continue;
                try { harmony.Patch(m, prefix: prefix); n++; } catch { }
            }
        }
        Probe.Log($"loop audit: watching {n} method(s), threshold {Threshold}/combat");
        Probe.Report["loop_audit_methods"] = n;
    }

    public static void ResetCombat()
    {
        foreach (var kv in _counts)
            if (kv.Value > _peak.GetValueOrDefault(kv.Key)) _peak[kv.Key] = kv.Value;
        _counts.Clear();
    }

    public static void Enter(MethodBase __originalMethod)
    {
        if (!Armed || _in) return;
        _in = true;
        try
        {
            var key = $"{__originalMethod.DeclaringType?.Name}::{__originalMethod.Name}";
            var c = _counts.GetValueOrDefault(key) + 1;
            _counts[key] = c;
            if (c == Threshold && !_dumped)
            {
                _dumped = true;
                Probe.Log($"LOOP: {key} hit {c} calls in ONE combat - runaway. Stack:");
                var frames = new StackTrace(1, false).GetFrames() ?? Array.Empty<StackFrame>();
                var shown = 0;
                foreach (var f in frames)
                {
                    var m = f.GetMethod();
                    var dt = m?.DeclaringType?.FullName;
                    if (dt == null || !dt.StartsWith("MegaCrit.", StringComparison.Ordinal)) continue;
                    Probe.Log($"    {dt.Substring(dt.LastIndexOf('.') + 1)}::{m.Name}");
                    if (++shown >= 25) break;
                }
            }
        }
        catch { }
        finally { _in = false; }
    }

    public static void Report()
    {
        ResetCombat();
        Probe.Report["loop_peak_calls_per_combat"] = _peak
            .OrderByDescending(k => k.Value).ToDictionary(k => k.Key, v => (object)v.Value);
        Probe.Log("LOOP AUDIT peak calls in a single combat:");
        foreach (var kv in _peak.OrderByDescending(k => k.Value))
            Probe.Log($"    {kv.Value,8}  {kv.Key}");
    }
}
