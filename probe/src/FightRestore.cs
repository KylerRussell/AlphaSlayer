using System;
using HarmonyLib;
using MegaCrit.Sts2.Core.Multiplayer.Replay;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace AlphaSlayer.Probe;

/// <summary>
/// Restore primitive for search: put a fight back exactly where it was.
///
/// The game has no mid-combat save, but it does save at ROOM ENTRY: EnterMapPointInternal calls
/// CombatReplayWriter.RecordInitialState(ToSave(null)) after leaving the previous room and
/// before the room is rolled and built. That is the same point its own save/continue uses
/// (SaveRun), and loading it with LoadIntoLatestMapCoord re-rolls and rebuilds the same room
/// from the same RNG. So "restore to decision k of this fight" = load the room-entry save,
/// re-enter the room, and re-apply the fight's first k choices (RunServe.FightAsync keeps
/// them as indices into its own legal-action list).
///
/// The writer is off under TestMode (RunManager.InitializeShared). With --probe-fight-restore
/// it is switched on, and Harmony turns it into a pure capture: RecordInitialState stores the
/// save here and every per-action recorder is skipped (the replay's event log is not used, and
/// RecordGameAction throws on actions it cannot serialise). Without the flag nothing changes.
/// </summary>
public static class FightRestore
{
    public static bool On { get; private set; }

    /// <summary>The run as it was at the latest room entry, as JSON (each restore
    /// deserialises a fresh copy, so loading can never mutate the snapshot).</summary>
    public static string RoomSaveJson { get; private set; }
    public static int RoomSaves { get; private set; }

    private static bool _installed;

    public static void Install()
    {
        On = Probe.HasFlag("probe-fight-restore");
        if (!On || _installed) return;
        var harmony = new Harmony("alphaslayer.probe.fightrestore");
        var t = typeof(CombatReplayWriter);
        harmony.Patch(AccessTools.Method(t, nameof(CombatReplayWriter.RecordInitialState)),
            prefix: new HarmonyMethod(AccessTools.Method(typeof(FightRestore), nameof(CaptureSave))));
        // WriteReplay too: CombatManager.EndCombatInternal calls it after every won fight, and with
        // RecordInitialState skipped (above) it throws "RecordInitialState must be called first",
        // which ABORTED the rest of EndCombatInternal: no AfterCombatVictory hooks (e.g. Burning
        // Blood's heal), no player.AfterCombatEnd, no MarkPreFinished/CombatEnded (the harness then
        // waited out its 5 s timeout after every fight).
        foreach (var name in new[] { "RecordGameAction", "RecordActionResume", "RecordPlayerChoice",
                                     "RecordChecksum", "WriteReplay" })
            harmony.Patch(AccessTools.Method(t, name),
                prefix: new HarmonyMethod(AccessTools.Method(typeof(FightRestore), nameof(SkipRecorder))));
        _installed = true;
        Probe.Log("fight restore: room-entry capture installed");
    }

    /// <summary>Called after every run setup (InitializeShared resets IsEnabled).</summary>
    public static void EnableCapture()
    {
        if (On) RunManager.Instance.CombatReplayWriter.IsEnabled = true;
    }

    private static bool CaptureSave(SerializableRun serializableRun)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        RoomSaveJson = JsonSerializationUtility.ToJson(serializableRun);
        RoomSaves++;
        CaptureMs += sw.Elapsed.TotalMilliseconds;
        if (Probe.HasFlag("probe-capture-log"))
            Probe.Log($"  room save #{RoomSaves}: {sw.Elapsed.TotalMilliseconds:F1} ms json, {RoomSaveJson.Length} chars, total {CaptureMs:F0} ms");
        return false;
    }

    public static double CaptureMs;

    private static bool SkipRecorder() => false;

    public static SerializableRun FreshSave() => Load(RoomSaveJson);

    public static SerializableRun Load(string json)
    {
        if (json == null) return null;
        var r = JsonSerializationUtility.FromJson<SerializableRun>(json);
        if (!r.Success || r.SaveData == null) throw new InvalidOperationException($"room save did not load: {r.Status}");
        return r.SaveData;
    }
}
