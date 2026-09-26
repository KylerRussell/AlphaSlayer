using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using HarmonyLib;

namespace AlphaSlayer.Probe;

/// <summary>
/// Recursion hunter.
///
/// The combat turn loop overflows the stack headless, and the core dump is unsymbolised JIT
/// frames. Rather than guess which method cycles, we patch a broad set of combat-path methods
/// (including the MoveNext of async state machines, which is where the real nesting happens)
/// with a prefix that asks the runtime whether the stack is nearly exhausted. When it is, we
/// dump the MANAGED stack and throw, so we get a symbolised cycle and a clean unwind instead
/// of a SIGSEGV.
/// </summary>
public static class Recursion
{
    private static bool _dumped;
    [ThreadStatic] private static bool _inHook;
    private static string _outDir;
    public static int Patched;

    public static void SelfTestEntry(string outDir)
    {
        _outDir = outDir;
        SelfTest();
    }

    public static void Install(string outDir)
    {
        _outDir = outDir;
        var harmony = new Harmony("alphaslayer.probe.recursion");
        var prefix = new HarmonyMethod(AccessTools.Method(typeof(Recursion), nameof(Guard)));

        var asm = typeof(MegaCrit.Sts2.Core.Combat.CombatManager).Assembly;
        // Patch the whole non-node core. The earlier 5-namespace set (1370 methods) never
        // tripped, so the cycle runs through something else - hooks and model callbacks are
        // the prime suspects, since Hook.* fans out across every relic/power in play.
        var targetTypes = asm.GetTypes().Where(t =>
            t.Namespace != null
            && t.Namespace.StartsWith("MegaCrit.Sts2.Core.", StringComparison.Ordinal)
            && !t.Namespace.StartsWith("MegaCrit.Sts2.Core.Nodes", StringComparison.Ordinal)
            && !t.Namespace.StartsWith("MegaCrit.Sts2.Core.Localization", StringComparison.Ordinal)
            && !t.IsInterface).ToList();
        Probe.Log($"recursion guard: patching {targetTypes.Count} types...");

        foreach (var t in targetTypes)
        {
            MethodInfo[] methods;
            try { methods = t.GetMethods(BindingFlags.Public | BindingFlags.NonPublic
                                       | BindingFlags.Instance | BindingFlags.Static
                                       | BindingFlags.DeclaredOnly); }
            catch { continue; }

            foreach (var m in methods)
            {
                if (m.IsAbstract || m.ContainsGenericParameters) continue;
                try
                {
                    // The interesting frame is the state machine's MoveNext, for BOTH async
                    // and iterator methods. Missing the iterator case is why this guard never
                    // fired before: the hook walk recurses through `yield return` iterators
                    // (IterateHookListeners), whose MoveNext was never patched -- patching the
                    // outer method only catches the call that returns the enumerable.
                    MethodBase target;
                    if (m.GetCustomAttribute<AsyncStateMachineAttribute>() != null)
                        target = AccessTools.AsyncMoveNext(m);
                    else if (m.GetCustomAttribute<IteratorStateMachineAttribute>() != null)
                        target = AccessTools.EnumeratorMoveNext(m);
                    else
                        target = m;
                    if (target == null || target.IsAbstract || target.ContainsGenericParameters) continue;
                    if (target.GetMethodBody() == null) continue;

                    harmony.Patch(target, prefix: prefix);
                    Patched++;
                }
                catch { /* not every method is patchable; skip quietly */ }
            }
        }

        Probe.Log($"recursion guard installed on {Patched} method(s)");
        Probe.Report["recursion_guard_methods"] = Patched;
    }

    /// <summary>
    /// Self-test: deliberately recurse so we can prove the detector fires before trusting a
    /// negative result from it.
    /// </summary>
    public static void SelfTest()
    {
        Probe.Log("recursion self-test: recursing deliberately...");
        try { Recurse(0); }
        catch (ProbeRecursionException e) { Probe.Log($"self-test OK: {e.Message}"); }
        catch (Exception e) { Probe.Log($"self-test unexpected: {e.GetType().Name}: {e.Message}"); }
        _dumped = false; // re-arm for the real hunt
    }

    private static int Recurse(int d)
    {
        Guard();
        return d > 2_000_000 ? d : Recurse(d + 1) + 1;
    }

    /// <summary>Harmony prefix. Cheap on the common path: one runtime call.</summary>
    public static bool Guard()
    {
        if (_dumped || _inHook) return true;
        if (RuntimeHelpers.TryEnsureSufficientExecutionStack()) return true;

        _inHook = true;
        _dumped = true;
        try { Dump(); } catch { }
        _inHook = false;

        throw new ProbeRecursionException(
            "probe: stack near exhaustion - recursion cycle dumped to recursion_stack.txt");
    }

    private static void Dump()
    {
        var st = new StackTrace(1, false);
        var frames = st.GetFrames() ?? Array.Empty<StackFrame>();
        var names = new List<string>(frames.Length);
        foreach (var f in frames)
        {
            var m = f.GetMethod();
            names.Add(m == null ? "<null>" : $"{m.DeclaringType?.FullName}::{m.Name}");
        }

        var path = Path.Combine(_outDir ?? Path.GetTempPath(), "recursion_stack.txt");
        File.WriteAllLines(path, names);

        Probe.Log($"RECURSION: stack near exhaustion at depth {names.Count}; full trace -> {path}");

        // Find the shortest repeating cycle at the top of the stack.
        var cycle = FindCycle(names, maxPeriod: 60);
        if (cycle > 0)
        {
            Probe.Log($"RECURSION: repeating cycle of {cycle} frame(s):");
            for (var i = 0; i < cycle && i < names.Count; i++)
                Probe.Log($"    [{i}] {names[i]}");
            Probe.Report["recursion_cycle_len"] = cycle;
            Probe.Report["recursion_cycle"] = names.Take(cycle).ToArray();
        }
        else
        {
            Probe.Log("RECURSION: no clean cycle found; first 40 frames:");
            for (var i = 0; i < 40 && i < names.Count; i++)
                Probe.Log($"    [{i}] {names[i]}");
            Probe.Report["recursion_cycle"] = names.Take(40).ToArray();
        }
        Probe.Report["recursion_depth"] = names.Count;
        Probe.Report["recursion_stack_file"] = path;
    }

    /// <summary>Smallest p such that frames[i] == frames[i+p] holds for several periods.</summary>
    private static int FindCycle(List<string> f, int maxPeriod)
    {
        for (var p = 1; p <= maxPeriod && p * 4 < f.Count; p++)
        {
            var ok = true;
            for (var i = 0; i < p * 3 && ok; i++)
                if (f[i] != f[i + p]) ok = false;
            if (ok) return p;
        }
        return 0;
    }
}

public sealed class ProbeRecursionException : Exception
{
    public ProbeRecursionException(string m) : base(m) { }
}
