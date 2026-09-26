using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace AlphaSlayer.Probe;

/// <summary>
/// Entry point. ModManager calls <see cref="Init"/> during mod load.
///
/// IMPORTANT: ModManager only invokes mod initializers when TestMode.IsOff, so we must not
/// flip TestMode before this runs. Any TestMode change happens inside the phases below.
/// </summary>
[ModInitializer(nameof(Init))]
public static class Probe
{
    public const string Version = "0.1.0";

    /// <summary>Phases requested via --probe=... on the command line.</summary>
    public static HashSet<string> Phases { get; private set; } = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Structured results, written to probe_report.json at the end.</summary>
    public static readonly Dictionary<string, object> Report = new();

    private static StreamWriter _log;
    private static string _outDir;
    public static string OutDir => _outDir;

    public static void Init()
    {
        try
        {
            _outDir = ArgValue("probe-out") ?? Path.Combine(Path.GetTempPath(), "alphaslayer_probe");
            Directory.CreateDirectory(_outDir);
            _log = new StreamWriter(Path.Combine(_outDir, "probe.log"), append: false) { AutoFlush = true };
        }
        catch (Exception e)
        {
            GD.PrintErr($"[probe] could not open log: {e}");
        }

        Log($"AlphaSlayer probe {Version} initializing. outDir={_outDir}");

        var requested = ArgValue("probe") ?? "inventory";
        foreach (var p in requested.Split(',', StringSplitOptions.RemoveEmptyEntries))
            Phases.Add(p.Trim());
        Log($"phases requested: {string.Join(",", Phases)}");

        Report["probe_version"] = Version;
        Report["phases_requested"] = Phases.ToArray();
        Report["utc"] = DateTime.UtcNow.ToString("o");

        // Re-enable the dev/debug paths the retail build compiles out.
        // NGame.IsReleaseGame() is hardcoded `return true`, which gates --autoslay and other
        // debug affordances. This patch does not change any game RULE, only the dev gate.
        TryPatchReleaseGate();

        // Suppress first-time-user tutorial nodes. CombatManager.StartCombatInternal calls
        // NCombatRulesFtue.Create() UNCONDITIONALLY when the FTUE has not been seen, which
        // instantiates a PackedScene and segfaults under --headless. The `SeenFtue == true`
        // branch is guarded by NCombatRoom.Instance?. and so short-circuits safely.
        // This is a presentation flag only; it changes no game rule.
        TryPatchSeenFtue();

        // Make gameplay waits YIELD instead of either sleeping or vanishing.
        //
        // NonInteractiveMode makes Cmd.Wait return an already-completed task. That removes the
        // wall-clock cost but ALSO removes the yield, so CombatManager's turn loop runs its
        // entire async chain synchronously on one stack and overflows it (SIGSEGV with a deep
        // self-similar JIT backtrace). Replacing the body with a single Task.Yield() keeps the
        // scheduler turn that the engine's design assumes, at no time cost.
        // --probe-natural leaves the engine's real timing alone. Slow, but it isolates the
        // harness from the wait-removal problem.
        // Off by default now: with TestMode on, NonInteractiveMode already makes Cmd.Wait a
        // no-op, and forcing a 1ms delay per wait is what pinned throughput to ~60/s.
        // --probe-patch-waits restores the slow-but-safe behaviour.
        // Bounded-yield waits. Letting every wait complete synchronously (what TestMode does
        // on its own) makes the combat chain re-enter itself until the stack dies; forcing a
        // real yield on EVERY wait costs ~1ms each and pins throughput near frame rate.
        // Yielding once every N waits bounds stack growth while keeping most waits free.
        TryPatchCmdWait();

        // Headless text measurement faults on freed text-server resources; it affects no rules.
        if (!HasFlag("probe-no-text-suppress")) TextSuppress.Install();

        // Turn unguarded scene lookups into null instead of a walk through freed native
        // memory. See NodeGuard for why this is a crash rather than an exception.
        if (!HasFlag("probe-no-node-guard"))
        {
            NodeGuard.Install();
            NodeGuard.InstallLocalPresentationSuppression();
            try { NodeGuard.InstallUnguardedPresentationFinalizers(); }
            catch (Exception e) { Log($"node guard finalizers FAILED: {e}"); }
        }

        // Remove the frame-rate cap so we can tell whether throughput is frame-bound.
        try
        {
            Engine.MaxFps = 0;
            Log("Engine.MaxFps = 0");
        }
        catch (Exception e) { Log($"could not clear MaxFps: {e.Message}"); }

        // Kick off async work; we cannot block the initializer (the game is still booting).
        TaskHelper.RunSafely(RunAllAsync());
    }

    private static async Task RunAllAsync()
    {
        try
        {
            // Wait for the game object to exist, then for startup to finish.
            var waited = 0;
            while (NGame.Instance == null && waited < 60_000)
            {
                await Task.Delay(50);
                waited += 50;
            }
            if (NGame.Instance == null)
            {
                Log("FATAL: NGame.Instance never appeared; aborting.");
                Report["fatal"] = "NGame.Instance null";
                WriteReport();
                return;
            }
            Log("NGame.Instance present; awaiting GameStartupComplete...");
            await NGame.Instance.GameStartupComplete;
            Log("GameStartupComplete.");

            Report["headless"] = DisplayServer.GetName()
                .Equals("headless", StringComparison.OrdinalIgnoreCase);
            Report["godot_version"] = Engine.GetVersionInfo()["string"].AsString();

            if (Phases.Contains("inventory"))
                await Guarded("inventory", Inventory.RunAsync);

            if (Phases.Contains("throughput"))
                await Guarded("throughput", Throughput.RunAsync);

            if (Phases.Contains("snapshot"))
                await Guarded("snapshot", Snapshot.RunAsync);

            // Recursion hunt: install the stack guard, then drive one combat so the turn loop
            // trips it and dumps a symbolised cycle instead of segfaulting.
            if (Phases.Contains("recursion"))
            {
                Recursion.SelfTestEntry(_outDir);
                Recursion.Install(_outDir);
                await Guarded("recursion", Episodes.RunAsync);
            }

            if (Phases.Contains("rngaudit"))
            {
                RngAudit.Install();
                await Guarded("rngaudit", Episodes.RunAsync);
                RngAudit.Report();
            }

            if (Phases.Contains("loopaudit"))
            {
                LoopAudit.Threshold = ArgInt("probe-loop-threshold", 400);
                LoopAudit.InstallNotifyProbe();
                LoopAudit.Install();
                await Guarded("loopaudit", Episodes.RunAsync);
                LoopAudit.Report();
            }

            // Run layer: map/path traversal through the engine's own room transitions.
            if (Phases.Contains("maprun"))
                await Guarded("maprun", MapRun.RunAsync);

            // Whole-run env server: every decision in a run, over the same socket protocol.
            // Deck evaluation: play an ARBITRARY deck against a chosen room type.
            if (Phases.Contains("deckserve"))
                await Guarded("deckserve", DeckServe.RunAsync);

            if (Phases.Contains("runserve"))
                await Guarded("runserve", RunServe.RunAsync);

            if (Phases.Contains("serve"))
                await Guarded("serve", Serve.RunAsync);

            if (Phases.Contains("episodes"))
                await Guarded("episodes", Episodes.RunAsync);

            // Method-entry trace: the crash is native, so read trace.txt's last line.
            if (Phases.Contains("trace"))
            {
                Tracer.Install(_outDir, HasFlag("probe-trace-all")
                    ? new[] { "MegaCrit.Sts2.*", "Godot.*" }
                    : new[]
                {
                    "MegaCrit.Sts2.Core.Combat",
                    "MegaCrit.Sts2.Core.GameActions",
                    "MegaCrit.Sts2.Core.Commands",
                    "MegaCrit.Sts2.Core.Hooks",
                    "MegaCrit.Sts2.Core.Models",
                    "MegaCrit.Sts2.Core.Entities.Creatures",
                });
                await Guarded("trace", Episodes.RunAsync);
            }

            WriteReport();
            Log("probe complete.");

            if (HasFlag("probe-quit"))
            {
                Log("--probe-quit set; shutting down.");
                NGame.Instance.GetTree().Quit(0);
            }
        }
        catch (Exception e)
        {
            Log($"FATAL in RunAllAsync: {e}");
            Report["fatal"] = e.ToString();
            WriteReport();
        }
    }

    /// <summary>Runs a phase so that one failing phase still leaves the others' data intact.</summary>
    private static async Task Guarded(string name, Func<Task> body)
    {
        Log($"===== phase {name}: start =====");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            await body();
            Report[$"{name}_status"] = "ok";
        }
        catch (Exception e)
        {
            Log($"phase {name} FAILED: {e}");
            Report[$"{name}_status"] = "failed";
            Report[$"{name}_error"] = e.ToString();
        }
        sw.Stop();
        Report[$"{name}_wall_ms"] = sw.Elapsed.TotalMilliseconds;
        Log($"===== phase {name}: done in {sw.Elapsed.TotalMilliseconds:F0}ms =====");
    }

    // ---------- command line ----------

    private static string[] AllArgs()
    {
        var a = new List<string>(OS.GetCmdlineArgs());
        a.AddRange(OS.GetCmdlineUserArgs());
        return a.ToArray();
    }

    /// <summary>Reads --key=value or "--key value".</summary>
    public static string ArgValue(string key)
    {
        var args = AllArgs();
        for (var i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (a.StartsWith($"--{key}=", StringComparison.OrdinalIgnoreCase))
                return a.Substring(key.Length + 3);
            if (a.Equals($"--{key}", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length
                && !args[i + 1].StartsWith("--"))
                return args[i + 1];
        }
        return null;
    }

    public static bool HasFlag(string key) =>
        AllArgs().Any(a => a.Equals($"--{key}", StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// FNV-1a over the UTF-8 bytes: the same value in every process.
    ///
    /// string.GetHashCode() is randomised per process in .NET Core, so seeding a Random from it
    /// meant --probe-seed did NOT reproduce encounter or potion sampling, and the binary
    /// dataset's vocab stamp changed on every run. Same class of bug as Python's salted hash()
    /// in the run policy's token table, fixed 2026-09-16.
    /// </summary>
    public static int StableHash(string s)
    {
        unchecked
        {
            var h = 2166136261u;
            foreach (var b in System.Text.Encoding.UTF8.GetBytes(s ?? ""))
            {
                h ^= b;
                h *= 16777619u;
            }
            return (int)h;
        }
    }

    public static int ArgInt(string key, int fallback) =>
        int.TryParse(ArgValue(key), NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : fallback;

    public static double ArgDouble(string key, double fallback) =>
        double.TryParse(ArgValue(key), NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : fallback;

    // ---------- release gate ----------

    private static void TryPatchReleaseGate()
    {
        try
        {
            var harmony = new Harmony("alphaslayer.probe");
            var target = AccessTools.Method(typeof(NGame), "IsReleaseGame");
            if (target == null)
            {
                Log("IsReleaseGame not found; skipping release-gate patch.");
                Report["release_gate_patched"] = false;
                return;
            }
            var post = AccessTools.Method(typeof(Probe), nameof(ReleaseGatePostfix));
            harmony.Patch(target, postfix: new HarmonyMethod(post));
            Log($"patched NGame.IsReleaseGame -> false (now {NGame.IsReleaseGame()})");
            Report["release_gate_patched"] = true;
        }
        catch (Exception e)
        {
            Log($"release-gate patch failed: {e.Message}");
            Report["release_gate_patched"] = false;
        }
    }

    private static void ReleaseGatePostfix(ref bool __result) => __result = false;

    private static void TryPatchSeenFtue()
    {
        try
        {
            var harmony = new Harmony("alphaslayer.probe.ftue");
            var target = AccessTools.Method(
                typeof(MegaCrit.Sts2.Core.Saves.SaveManager), "SeenFtue", new[] { typeof(string) });
            if (target == null)
            {
                Log("SaveManager.SeenFtue not found; skipping FTUE patch.");
                Report["ftue_patched"] = false;
                return;
            }
            harmony.Patch(target, postfix: new HarmonyMethod(AccessTools.Method(typeof(Probe), nameof(SeenFtuePostfix))));
            Log("patched SaveManager.SeenFtue -> true (suppresses tutorial node creation)");
            Report["ftue_patched"] = true;
        }
        catch (Exception e)
        {
            Log($"FTUE patch failed: {e.Message}");
            Report["ftue_patched"] = false;
        }
    }

    private static void SeenFtuePostfix(ref bool __result) => __result = true;

    // ---------- wait -> yield ----------

    private static void TryPatchCmdWait()
    {
        try
        {
            var harmony = new Harmony("alphaslayer.probe.wait");
            var cmd = typeof(MegaCrit.Sts2.Core.Commands.Cmd);
            var patched = 0;
            foreach (var m in cmd.GetMethods(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static))
            {
                if (m.Name != "Wait" && m.Name != "CustomScaledWait") continue;
                if (m.ReturnType != typeof(Task)) continue;
                harmony.Patch(m, prefix: new HarmonyMethod(AccessTools.Method(typeof(Probe), nameof(WaitPrefix))));
                patched++;
            }
            YieldEvery = ArgInt("probe-yield-every", 1);
        YieldMode = ArgValue("probe-yield-mode") ?? "frame";
        DelayEvery = ArgInt("probe-delay-every", 64);
        Log($"patched {patched} Cmd wait method(s); real yield every {YieldEvery} wait(s), mode={YieldMode}, hard-yield every {DelayEvery}");
            Report["wait_methods_patched"] = patched;
        }
        catch (Exception e)
        {
            Log($"Cmd.Wait patch failed: {e.Message}");
            Report["wait_methods_patched"] = 0;
        }
    }

    /// <summary>
    /// Replaces every gameplay wait with one real SceneTree frame boundary.
    ///
    /// Task.Yield() is NOT sufficient here: Godot's SynchronizationContext runs the
    /// continuation inline when posted from the main thread, so the async chain never
    /// actually unwinds and the turn loop still recurses to a stack overflow. Awaiting the
    /// SceneTree.ProcessFrame signal is a genuine suspension point, which is what the
    /// engine's design assumes a gameplay wait provides.
    /// </summary>
    private static int _waitCount;

    private static bool WaitPrefix(ref Task __result)
    {
        // Adaptive: yield for real only when the runtime says we are running out of stack.
        // A fixed "every Nth wait" counter cannot work, because the safe N depends on how
        // deep the current chain already is - N=24 survives 10 combats and dies at 60.
        // Asking the runtime removes the guesswork and self-tunes.
        if (!System.Runtime.CompilerServices.RuntimeHelpers.TryEnsureSufficientExecutionStack())
        {
            _yields++;
            __result = OneFrame();
            return false;
        }

        var every = YieldEvery;
        __result = (every > 0 && ++_waitCount % every == 0)
            ? OneFrame()
            : Task.CompletedTask;
        return false;
    }

    public static int _yields;

    /// <summary>How many waits may complete synchronously between real yields.</summary>
    public static int YieldEvery = 1;

    /// <summary>
    /// A real suspension - one that actually unwinds the stack back to the message loop.
    /// "frame" waits on SceneTree.ProcessFrame; "delay" uses a 1ms timer (slower, but proven).
    /// </summary>
    private static int _frameYields;

    private static async Task OneFrame()
    {
        if (YieldMode == "delay") { await Task.Delay(1); return; }

        // Hybrid. Awaiting SceneTree.ProcessFrame is cheap and unwinds for shallow chains, but
        // deep ones (Defect's orb hooks nest far further than Ironclad's) still overflow the
        // stack with it alone. A real timer always unwinds; it just costs ~1ms. So take the
        // frame signal by default and force a timer every DelayEvery yields, which bounds
        // nesting for a few percent of throughput.
        if (DelayEvery > 0 && ++_frameYields % DelayEvery == 0) { await Task.Delay(1); return; }

        var tree = NGame.Instance?.GetTreeOrNull();
        if (tree == null) { await Task.Delay(1); return; }
        await tree.ToSignal(tree, SceneTree.SignalName.ProcessFrame);
    }

    public static string YieldMode = "frame";
    public static int DelayEvery = 64;

    // ---------- output ----------

    public static void Log(string msg)
    {
        var line = $"[probe] {msg}";
        GD.Print(line);
        try { _log?.WriteLine($"{DateTime.UtcNow:HH:mm:ss.fff} {msg}"); } catch { }
    }

    public static void WriteReport()
    {
        try
        {
            var path = Path.Combine(_outDir, "probe_report.json");
            File.WriteAllText(path, ToJson(Report));
            Log($"report written: {path}");
        }
        catch (Exception e) { Log($"could not write report: {e.Message}"); }
    }

    /// <summary>
    /// Minimal JSON writer. Avoids System.Text.Json source-gen/trimming surprises inside the
    /// game's load context, and keeps the probe dependency-free.
    /// </summary>
    public static string ToJson(object o)
    {
        var sb = new StringBuilder();
        Write(sb, o, 0);
        return sb.ToString();
    }

    /// <summary>
    /// Single-line JSON, for JSONL episode records. Uses a real compact mode rather than
    /// stripping whitespace from pretty output, which would corrupt string values.
    /// </summary>
    public static string ToJsonCompact(object o)
    {
        var sb = new StringBuilder();
        _compact = true;
        try { Write(sb, o, 0); } finally { _compact = false; }
        return sb.ToString();
    }

    [ThreadStatic] private static bool _compact;

    private static void Write(StringBuilder sb, object o, int indent)
    {
        var pad = _compact ? "" : new string(' ', indent * 2);
        var padIn = _compact ? "" : new string(' ', (indent + 1) * 2);
        var nl = _compact ? "" : "\n";
        switch (o)
        {
            case null: sb.Append("null"); break;
            case string s: sb.Append('"').Append(Escape(s)).Append('"'); break;
            case bool b: sb.Append(b ? "true" : "false"); break;
            case float or double or decimal:
                sb.Append(Convert.ToDouble(o).ToString("R", CultureInfo.InvariantCulture)); break;
            case sbyte or byte or short or ushort or int or uint or long or ulong:
                sb.Append(Convert.ToString(o, CultureInfo.InvariantCulture)); break;
            case IDictionary<string, object> d:
                if (d.Count == 0) { sb.Append("{}"); break; }
                sb.Append('{').Append(nl);
                var i = 0;
                foreach (var kv in d)
                {
                    sb.Append(padIn).Append('"').Append(Escape(kv.Key)).Append("\": ");
                    Write(sb, kv.Value, indent + 1);
                    if (++i < d.Count) sb.Append(',');
                    sb.Append(nl);
                }
                sb.Append(pad).Append('}');
                break;
            case System.Collections.IEnumerable e:
                var items = e.Cast<object>().ToList();
                if (items.Count == 0) { sb.Append("[]"); break; }
                sb.Append('[').Append(nl);
                for (var j = 0; j < items.Count; j++)
                {
                    sb.Append(padIn);
                    Write(sb, items[j], indent + 1);
                    if (j < items.Count - 1) sb.Append(',');
                    sb.Append(nl);
                }
                sb.Append(pad).Append(']');
                break;
            default: sb.Append('"').Append(Escape(o.ToString())).Append('"'); break;
        }
    }

    private static string Escape(string s) =>
        s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
}
