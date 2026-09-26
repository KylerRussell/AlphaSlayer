using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using HarmonyLib;

namespace AlphaSlayer.Probe;

/// <summary>
/// Method-entry tracer.
///
/// The crash is a native SIGSEGV, not a managed exception or stack overflow, so nothing
/// unwinds and no handler runs. The only way to see where it happens is to record every
/// method entry to a flushed file and read the last line after the process dies.
///
/// Scoped tightly (combat path, no property getters) to keep the volume readable.
/// </summary>
public static class Tracer
{
    private static StreamWriter _w;
    [ThreadStatic] private static bool _in;
    public static int Patched;

    public static void Install(string outDir, string[] namespaces)
    {
        _w = new StreamWriter(Path.Combine(outDir, "trace.txt"), append: false) { AutoFlush = true };
        var harmony = new Harmony("alphaslayer.probe.trace");
        var prefix = new HarmonyMethod(AccessTools.Method(typeof(Tracer), nameof(Enter)));
        // Also scan GodotSharp: a wild-pointer fault in managed code with a shallow stack is
        // the signature of calling into a FREED Godot native object, and that marshalling
        // lives in Godot.*, not in the game assembly.
        var assemblies = new[]
        {
            typeof(MegaCrit.Sts2.Core.Combat.CombatManager).Assembly,
            typeof(Godot.GodotObject).Assembly,
        };

        // A namespace entry ending in ".*" matches by prefix, so the whole assembly can be
        // covered when the fault is somewhere we have not thought of yet.
        bool Match(string ns) => namespaces.Any(n => n.EndsWith(".*", StringComparison.Ordinal)
            ? ns.StartsWith(n.Substring(0, n.Length - 1), StringComparison.Ordinal)
            : ns == n);

        var types = assemblies.SelectMany(a =>
        {
            try { return a.GetTypes(); } catch { return Array.Empty<Type>(); }
        });

        foreach (var t in types.Where(t => t.Namespace != null
                     && Match(t.Namespace) && !t.IsInterface))
        {
            MethodInfo[] ms;
            try { ms = t.GetMethods(BindingFlags.Public | BindingFlags.NonPublic
                                  | BindingFlags.Instance | BindingFlags.Static
                                  | BindingFlags.DeclaredOnly); }
            catch { continue; }

            foreach (var m in ms)
            {
                if (m.IsAbstract || m.ContainsGenericParameters) continue;
                if (m.Name.StartsWith("get_") || m.Name.StartsWith("set_")) continue;
                try
                {
                    var target = m.GetCustomAttribute<AsyncStateMachineAttribute>() != null
                        ? AccessTools.AsyncMoveNext(m) : m;
                    if (target == null || target.IsAbstract || target.ContainsGenericParameters) continue;
                    if (target.GetMethodBody() == null) continue;
                    harmony.Patch(target, prefix: prefix);
                    Patched++;
                }
                catch { }
            }
        }
        Probe.Log($"tracer installed on {Patched} method(s) -> {Path.Combine(outDir, "trace.txt")}");
        Probe.Report["tracer_methods"] = Patched;
    }

    public static void Enter(MethodBase __originalMethod)
    {
        if (_in) return;
        _in = true;
        try { _w?.WriteLine($"{__originalMethod.DeclaringType?.FullName}::{__originalMethod.Name}"); }
        catch { }
        _in = false;
    }
}
