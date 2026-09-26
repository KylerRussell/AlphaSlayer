using System;
using System.Linq;
using System.Reflection;
using HarmonyLib;
using Godot;

namespace AlphaSlayer.Probe;

/// <summary>
/// Makes the "is there a scene?" lookups answer NO headless.
///
/// Almost all of the engine's node touches are written as `NRun.Instance?.X` / `NEventRoom
/// .Instance?.Y`, so a null answer is the case they are already written for. A handful are
/// not - Trial.Accept opens with
///
///     if (LocalContext.IsMe(Owner)) NEventRoom.Instance.Layout.RemoveNodesOnPortrait();
///
/// with no null-conditional at all. That chain bottoms out in
/// NGame.CurrentRunNode => RootSceneContainer.CurrentScene as NRun, and headless it does not
/// merely return null: it walks a scene container whose native side is gone, and the process
/// dies with SIGSEGV rather than a managed NullReferenceException. A hard crash cannot be
/// caught, so one unguarded line in one event took down whole runs.
///
/// Forcing the getter to null turns that into an ordinary managed exception, which the caller
/// can catch and report. It changes no rule: the only thing downstream of it is presentation,
/// and every guarded call site already treats null as "no scene, skip the visuals".
/// </summary>
public static class NodeGuard
{
    public static int Patched;

    public static void Install()
    {
        if (!DisplayServer.GetName().Equals("headless", StringComparison.OrdinalIgnoreCase))
        {
            Probe.Log("node guard: not headless, skipping");
            return;
        }

        var harmony = new Harmony("alphaslayer.probe.nodeguard");
        var nullify = new HarmonyMethod(AccessTools.Method(typeof(NodeGuard), nameof(ReturnNull)));

        // NGame.CurrentRunNode is the root of every N*.Instance chain used from model code.
        foreach (var (type, prop) in new[]
                 {
                     ("MegaCrit.Sts2.Core.Nodes.NGame", "CurrentRunNode"),
                     ("MegaCrit.Sts2.Core.Nodes.NRun", "Instance"),
                     ("MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom", "Instance"),
                 })
        {
            var t = AccessTools.TypeByName(type);
            var m = t == null ? null : AccessTools.PropertyGetter(t, prop);
            if (m == null) { Probe.Log($"  node guard: {type}.{prop} not found"); continue; }
            try { harmony.Patch(m, prefix: nullify); Patched++; }
            catch (Exception e) { Probe.Log($"  node guard: {type}.{prop}: {e.Message}"); }
        }
        // Verify the patch actually took: a getter that still answers non-null is worse than
        // no patch at all, because it looks safe and is not.
        var run = AccessTools.TypeByName("MegaCrit.Sts2.Core.Nodes.NRun");
        var live = run == null ? "?" : (AccessTools.PropertyGetter(run, "Instance")?.Invoke(null, null) == null ? "null" : "NON-NULL");
        Probe.Log($"node guard: nullified {Patched} scene lookup(s) (headless); NRun.Instance={live}");
    }

    private static bool ReturnNull(ref object __result)
    {
        __result = null;
        return false;
    }

    /// <summary>
    /// True while inside an event handler whose node blocks must be skipped. See
    /// InstallLocalPresentationSuppression.
    /// </summary>
    /// <summary>
    /// Nesting DEPTH, not a flag. Trial.Accept calls Trial.AddVfxAnchoredToPortrait, which is
    /// scoped too; with a bool the inner method's exit cleared the outer scope, so the rest of
    /// Accept ran unsuppressed and loaded a portrait texture that does not exist headless.
    /// That failure logs a warning, and the log path then recurses until the stack dies - 583
    /// frames with a 54x repeating cycle, which is why this looked like a null dereference and
    /// was not one.
    /// </summary>
    [ThreadStatic] public static int SuppressLocalPresentation;

    /// <summary>
    /// Skips the presentation-only blocks inside Trial.
    ///
    /// Trial is the one event in the game that touches the scene without a null-conditional.
    /// All three touches sit inside `if (LocalContext.IsMe(Owner)) { ... }` blocks whose
    /// bodies are pure presentation - clearing portrait children, setting a portrait texture,
    /// parenting a vfx node - while every rule in the method (the RNG draw that picks which
    /// defendant appears, the option set, the curses and relics awarded) is unconditional.
    ///
    /// So IsMe is answered false for the duration of Trial's handlers and only there. That
    /// skips exactly the three node blocks and changes no rule. It has to be scoped: IsMe
    /// decides real behaviour elsewhere (RewardsSet.Offer uses it to decide whether to run
    /// the reward selector at all), so a global answer would silently disable rewards.
    /// </summary>
    public static void InstallLocalPresentationSuppression()
    {
        var harmony = new Harmony("alphaslayer.probe.presentation");
        var trial = AccessTools.TypeByName("MegaCrit.Sts2.Core.Models.Events.Trial");
        var ctx = AccessTools.TypeByName("MegaCrit.Sts2.Core.Context.LocalContext");
        var isMe = ctx == null ? null : AccessTools.Method(ctx, "IsMe", new[] { AccessTools.TypeByName("MegaCrit.Sts2.Core.Entities.Players.Player") });
        if (trial == null || isMe == null) { Probe.Log("presentation suppress: types not found"); return; }

        harmony.Patch(isMe, prefix: new HarmonyMethod(AccessTools.Method(typeof(NodeGuard), nameof(IsMePrefix))));

        var n = 0;
        foreach (var name in new[] { "Accept", "AddVfxAnchoredToPortrait" })
        {
            var m = AccessTools.Method(trial, name);
            if (m == null) continue;
            harmony.Patch(m,
                prefix: new HarmonyMethod(AccessTools.Method(typeof(NodeGuard), nameof(EnterScope))),
                finalizer: new HarmonyMethod(AccessTools.Method(typeof(NodeGuard), nameof(ExitScope))));
            n++;
        }
        Probe.Log($"presentation suppress: scoped {n} Trial method(s)");
    }

    private static void EnterScope() => SuppressLocalPresentation++;
    private static void ExitScope()
    {
        if (SuppressLocalPresentation > 0) SuppressLocalPresentation--;
    }

    private static bool IsMePrefix(ref bool __result)
    {
        if (SuppressLocalPresentation <= 0) return true;
        __result = false;
        return false;
    }

    /// <summary>
    /// Methods that dereference the scene without a null guard, and whose node work is
    /// presentation only. Each entry is an engine bug that is unreachable in the real game
    /// (the scene always exists there) and fatal headless.
    ///
    /// Format: "Namespace.Type:Method". Discovered by the maprun diagnostic, which is what it
    /// is for - a room-type sweep that reports exactly which paths die.
    /// </summary>
    private static readonly string[] _unguardedPresentation =
    {
        // Creature.Died handler: NCombatRoom.Instance.GetCreatureNode(...) with no `?.`,
        // where the very next line of the same class uses `?.` correctly. Sets a death
        // animation; nothing downstream of it is a rule.
        "MegaCrit.Sts2.Core.Models.Monsters.SoulNexus:AfterDeath:MegaCrit.Sts2.Core.Entities.Creatures.Creature",
    };

    /// <summary>
    /// SKIPS the methods above entirely, via a prefix that returns false.
    ///
    /// A finalizer was the first attempt and does not work, for an instructive reason: the
    /// fault is not a NullReferenceException that a wrapper could swallow. NRun.Instance is a
    /// one-line property, so the JIT INLINES it into NCombatRoom.Instance and that into the
    /// caller - which means a Harmony patch on the getter never runs at these call sites even
    /// though it works when the getter is called directly. (That is why nullifying the getters
    /// verified as "NRun.Instance=null" and changed nothing here.) The inlined code walks a
    /// freed scene container and takes SIGSEGV, and a signal cannot be caught.
    ///
    /// Patching the outermost method does work: it is reached through a delegate, so it is
    /// never inlined into its caller. Skipping SoulNexus.AfterDeath loses the death animation
    /// and one `Creature.Died -= AfterDeath` unsubscribe on an already-dead creature. Neither
    /// is a rule.
    /// </summary>
    public static void InstallUnguardedPresentationFinalizers()
    {
        Probe.Log("node guard: installing unguarded-presentation skips");
        var harmony = new Harmony("alphaslayer.probe.unguarded");
        var skip = new HarmonyMethod(AccessTools.Method(typeof(NodeGuard), nameof(SkipPresentationBody)));
        var n = 0;
        foreach (var entry in _unguardedPresentation)
        {
            var parts = entry.Split(':');
            var t = AccessTools.TypeByName(parts[0]);
            // Parameter types are spelled out: these are private handlers whose names collide
            // with base-class members, so a name-only lookup is ambiguous.
            var argTypes = parts.Length > 2
                ? parts.Skip(2).Select(AccessTools.TypeByName).ToArray()
                : null;
            var m = t == null ? null : AccessTools.Method(t, parts[1], argTypes);
            if (m == null) { Probe.Log($"  unguarded: {entry} not found"); continue; }
            try { harmony.Patch(m, prefix: skip); n++; }
            catch (Exception e) { Probe.Log($"  unguarded: {entry}: {e.Message}"); }
        }
        Probe.Log($"node guard: skipping {n} unguarded presentation method(s)");
    }

    public static int Skipped;

    private static bool SkipPresentationBody(MethodBase __originalMethod)
    {
        Skipped++;
        if (Skipped <= 3)
            Probe.Log($"  skipped unguarded presentation body: " +
                      $"{__originalMethod.DeclaringType?.Name}.{__originalMethod.Name}");
        return false;
    }

    /// <summary>
    /// Diagnostic (--probe-probe-trial): reports the scene-lookup state at the entry of the
    /// one event known to dereference it without a guard.
    /// </summary>
    public static void InstallTrialProbe()
    {
        var t = AccessTools.TypeByName("MegaCrit.Sts2.Core.Models.Events.Trial");
        var m = t == null ? null : AccessTools.Method(t, "Accept");
        if (m == null) { Probe.Log("trial probe: Trial.Accept not found"); return; }
        new Harmony("alphaslayer.probe.trial").Patch(
            m, prefix: new HarmonyMethod(AccessTools.Method(typeof(NodeGuard), nameof(TrialPrefix))));
        Probe.Log("trial probe installed");
    }

    private static bool TrialPrefix()
    {
        try
        {
            var run = AccessTools.TypeByName("MegaCrit.Sts2.Core.Nodes.NRun");
            var evr = AccessTools.TypeByName("MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom");
            var runI = AccessTools.PropertyGetter(run, "Instance")?.Invoke(null, null);
            var evrI = AccessTools.PropertyGetter(evr, "Instance")?.Invoke(null, null);
            Probe.Log($"  TRIAL entry: NRun.Instance={(runI == null ? "null" : "NON-NULL")} " +
                      $"NEventRoom.Instance={(evrI == null ? "null" : "NON-NULL")}");
        }
        catch (Exception e) { Probe.Log($"  TRIAL entry probe: {e.Message}"); }
        return true;
    }
}
