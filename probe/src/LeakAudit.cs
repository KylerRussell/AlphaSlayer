using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace AlphaSlayer.Probe;

/// <summary>
/// What grows with every fight restore? Search slows down linearly with the number of
/// restores a game process has done (replay ~2 ms per choice fresh, ~140 ms after ~1,500
/// searches), so something survives RunManager.CleanUp. A snapshot measures every static
/// field of the game assembly, and the instance fields of the objects they hold two levels
/// down (the singletons): a delegate's subscriber count or a collection's size. Diffs between
/// snapshots name what keeps growing. Profiling only (--probe-leak-audit).
/// </summary>
public static class LeakAudit
{
    private static Dictionary<string, long> _base;
    private const BindingFlags Statics = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    public static Dictionary<string, long> Snapshot()
    {
        var d = new Dictionary<string, long>();
        var asm = typeof(MegaCrit.Sts2.Core.Combat.CombatManager).Assembly;
        Type[] types;
        try { types = asm.GetTypes(); }
        catch (ReflectionTypeLoadException e) { types = e.Types.Where(t => t != null).ToArray(); }
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        foreach (var t in types)
        {
            if (t.ContainsGenericParameters) continue;
            FieldInfo[] fs;
            try { fs = t.GetFields(Statics); } catch { continue; }
            foreach (var f in fs)
            {
                if (f.IsLiteral) continue;
                object v;
                try { v = f.GetValue(null); } catch { continue; }
                Measure(d, $"{t.FullName}.{f.Name}", v, 2, asm, seen);
            }
        }
        d["godot.objects"] = (long)Godot.Performance.GetMonitor(Godot.Performance.Monitor.ObjectCount);
        d["godot.nodes"] = (long)Godot.Performance.GetMonitor(Godot.Performance.Monitor.ObjectNodeCount);
        d["godot.orphans"] = (long)Godot.Performance.GetMonitor(Godot.Performance.Monitor.ObjectOrphanNodeCount);
        d["gc.heap_kb"] = GC.GetTotalMemory(false) / 1024;
        return d;
    }

    private static void Measure(Dictionary<string, long> d, string key, object v, int depth, Assembly asm,
                                HashSet<object> seen)
    {
        if (v == null) return;
        if (v is Delegate del) { d[key] = del.GetInvocationList().Length; return; }
        if (v is ICollection c) { d[key] = c.Count; return; }
        var t = v.GetType();
        if (depth <= 0 || t.Assembly != asm || t.IsValueType || !seen.Add(v)) return;
        FieldInfo[] fs;
        try { fs = t.GetFields(Instance); } catch { return; }
        foreach (var f in fs)
        {
            object fv;
            try { fv = f.GetValue(v); } catch { continue; }
            Measure(d, key + "." + f.Name, fv, depth - 1, asm, seen);
        }
    }

    /// <summary>Writes the fields that grew since the first snapshot (largest first) to
    /// &lt;probe-out&gt;/leak.jsonl.</summary>
    public static void Report(long restores)
    {
        var now = Snapshot();
        _base ??= now;
        var grown = now.Where(kv => _base.TryGetValue(kv.Key, out var b) ? kv.Value > b : kv.Value > 0)
            .Select(kv => (kv.Key, kv.Value, Base: _base.TryGetValue(kv.Key, out var b) ? b : 0))
            .OrderByDescending(x => x.Value - x.Base).Take(40)
            .ToDictionary(x => x.Key, x => (object)new[] { x.Base, x.Value });
        var rec = new Dictionary<string, object>
        {
            ["restores"] = restores, ["fields"] = now.Count, ["grown"] = grown,
            ["godot_nodes"] = now["godot.nodes"], ["godot_objects"] = now["godot.objects"],
            ["orphans"] = now["godot.orphans"], ["heap_kb"] = now["gc.heap_kb"],
        };
        File.AppendAllText(Path.Combine(Probe.OutDir, "leak.jsonl"), Probe.ToJsonCompact(rec) + "\n");
    }
}
