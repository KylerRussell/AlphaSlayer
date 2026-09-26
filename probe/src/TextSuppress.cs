using System;
using System.Linq;
using System.Reflection;
using Godot;
using HarmonyLib;

namespace AlphaSlayer.Probe;

/// <summary>
/// Suppresses the auto-font-size text path when running headless.
///
/// `MegaLabel` / `MegaRichTextLabel` re-measure text through a cached `TextParagraph` and the
/// Godot text server. Headless that cached paragraph can already be disposed - MegaCrit's own
/// comment on `MegaLabel.DisposeCachedParagraph` notes it is "Nulled to guard against
/// AdjustFontSize running during Godot's quit frames" - and measuring through a freed RID
/// faults with a wild pointer rather than a managed exception.
///
/// This is presentation-only work: font sizing changes no game rule, no RNG stream and no
/// state the agent observes. It belongs in the same family as suppressing the FTUE node and
/// the TestMode card-node path.
/// </summary>
public static class TextSuppress
{
    public static int Patched;

    public static void Install()
    {
        if (!DisplayServer.GetName().Equals("headless", StringComparison.OrdinalIgnoreCase))
        {
            Probe.Log("text suppress: not headless, skipping");
            return;
        }

        var harmony = new Harmony("alphaslayer.probe.text");
        var skip = new HarmonyMethod(AccessTools.Method(typeof(TextSuppress), nameof(Skip)));
        var asm = typeof(MegaCrit.Sts2.Core.Combat.CombatManager).Assembly;

        foreach (var t in asm.GetTypes().Where(t => t.Namespace == "MegaCrit.Sts2.addons.mega_text"))
        {
            foreach (var m in t.GetMethods(BindingFlags.Public | BindingFlags.NonPublic
                                         | BindingFlags.Instance | BindingFlags.DeclaredOnly))
            {
                if (m.Name is not ("AdjustFontSize" or "SetFontSize" or "SetTextAutoSize" or "RefreshFont"))
                    continue;
                if (m.ReturnType != typeof(void) || m.ContainsGenericParameters) continue;
                try { harmony.Patch(m, prefix: skip); Patched++; }
                catch (Exception e) { Probe.Log($"  text suppress: {t.Name}.{m.Name}: {e.Message}"); }
            }
        }
        // Informational node screens. PandorasBox.AfterObtained ends with
        //     if (list.Count > 0 && LocalContext.IsMe(Owner))
        //         NSimpleCardsViewScreen.ShowScreen(...)
        // which is NOT null-conditional, so it instantiates a screen and faults headless -
        // the same shape as the FTUE bug. These screens only display results; suppressing them
        // changes no rule, no RNG stream and nothing the agent observes.
        //
        // Deliberately narrow: SELECTION screens are left alone, because those are real
        // decision points that must reach the agent's ICardSelector.
        foreach (var name in new[] { "MegaCrit.Sts2.Core.Nodes.Screens.NSimpleCardsViewScreen",
                                     "MegaCrit.Sts2.Core.Nodes.Screens.NDeckViewScreen",
                                     "MegaCrit.Sts2.Core.Nodes.Screens.NCardPileScreen" })
        {
            var t = asm.GetType(name);
            if (t == null) continue;
            foreach (var m in t.GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly))
            {
                if (m.Name != "ShowScreen" || m.ContainsGenericParameters) continue;
                try { harmony.Patch(m, prefix: new HarmonyMethod(AccessTools.Method(typeof(TextSuppress), nameof(SkipScreen)))); Patched++; }
                catch (Exception e) { Probe.Log($"  screen suppress {t.Name}: {e.Message}"); }
            }
        }

        // Audio. Godot's --headless implies --audio-driver Dummy, but the game drives FMOD
        // through its own native library, which ignores Godot's audio driver entirely and
        // happily plays out of the default device - so 25 headless training processes all
        // make noise. Audio affects no rule, no RNG stream the agent depends on, and nothing
        // it observes.
        var audioPatched = 0;
        foreach (var name in new[] { "MegaCrit.Sts2.Core.Nodes.Audio.NAudioManager",
                                     "MegaCrit.Sts2.Core.Nodes.Audio.NRunMusicController",
                                     "MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager" })
        {
            var t = asm.GetType(name);
            if (t == null) continue;
            foreach (var m in t.GetMethods(BindingFlags.Public | BindingFlags.NonPublic
                                         | BindingFlags.Instance | BindingFlags.Static
                                         | BindingFlags.DeclaredOnly))
            {
                if (!m.Name.StartsWith("Play", StringComparison.Ordinal)
                    && m.Name != "UpdateTrack" && m.Name != "PlayCustomMusic") continue;
                // ONLY void methods. Skipping a non-void method means inventing a return
                // value, and nulling a struct return corrupts the caller - which is what
                // produced a segfault on the first attempt at this.
                if (m.ReturnType != typeof(void)) continue;
                if (m.ContainsGenericParameters || m.GetMethodBody() == null) continue;
                try
                {
                    harmony.Patch(m, prefix: new HarmonyMethod(
                        AccessTools.Method(typeof(TextSuppress), nameof(Skip))));
                    audioPatched++;
                }
                catch { /* not every overload is patchable */ }
            }
        }
        Patched += audioPatched;
        Probe.Log($"text suppress: neutralised {Patched} presentation method(s) " +
                  $"({audioPatched} audio) (headless)");
        Probe.Report["text_methods_suppressed"] = Patched;
    }

    private static bool Skip() => false; // skip the original entirely

    /// <summary>Skips an informational screen and returns null to the caller.</summary>
    private static bool SkipScreen(ref object __result)
    {
        __result = null;
        return false;
    }
}
