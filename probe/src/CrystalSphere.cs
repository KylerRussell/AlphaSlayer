using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using HarmonyLib;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;

namespace AlphaSlayer.Probe;

/// <summary>
/// The Crystal Sphere event's fog-grid minigame, driven model-side.
///
/// This is the ONE interaction in the run layer with no model-layer completion path at all.
/// CrystalSphereMinigame.PlayMinigame is:
///
///     NCrystalSphereScreen.ShowScreen(this);
///     await _completionSource.Task;      // only ever set by CellClicked
///     await CompleteMinigame();
///
/// with no TestMode branch. The completion source is set when DivinationCount reaches zero,
/// and the only thing that decrements it is CellClicked, which the node screen calls on a
/// click. Headless there is no screen, so the await never returns: the event never finishes,
/// its room never exits, and the run stops dead. (That is what it did - the whole process
/// wedged on act 2, not just the phase.)
///
/// Everything needed to play it is public on the model, though: the cell grid, the remaining
/// divination count, and CellClicked. So rather than skip the event or fake its rewards, the
/// prefix below suppresses only the screen and plays the actual minigame - the same calls the
/// UI would have made, so items are revealed and awarded by the engine's own code.
///
/// Which cells to divine is a real decision (items occupy several cells and are only awarded
/// once every cell they cover is clear), so it is a policy hook rather than a fixed rule.
/// </summary>
public static class CrystalSphere
{
    /// <summary>Chooses the next cell to divine. Default: uniform over hidden cells.</summary>
    public static Func<CrystalSphereMinigame, CrystalSphereCell> ChooseCell;

    public static int Played, Failed;
    private static Random _rng = new(20250828);
    private static bool _installed;

    public static void Install()
    {
        if (_installed) return;
        var t = AccessTools.TypeByName("MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere.NCrystalSphereScreen");
        var m = t == null ? null : AccessTools.Method(t, "ShowScreen");
        if (m == null) { Probe.Log("crystal sphere: NCrystalSphereScreen.ShowScreen not found"); return; }

        new Harmony("alphaslayer.probe.crystalsphere").Patch(
            m, prefix: new HarmonyMethod(AccessTools.Method(typeof(CrystalSphere), nameof(ShowScreenPrefix))));
        _installed = true;
        Probe.Log("crystal sphere minigame driver installed");
    }

    /// <summary>
    /// Replaces the screen with a model-side player. Returning false skips the original, so
    /// no scene is instantiated; __result is left null, which the caller ignores.
    /// </summary>
    private static bool ShowScreenPrefix(CrystalSphereMinigame grid, ref object __result)
    {
        __result = null;
        // Fire-and-forget on purpose: PlayMinigame is about to await _completionSource, so
        // this must run alongside it rather than before it.
        _ = PlayAsync(grid);
        return false;
    }

    private static async Task PlayAsync(CrystalSphereMinigame grid)
    {
        try
        {
            // DivinationCount is the budget (3 for Uncover the Future, 6 for the Payment
            // Plan). CellClicked decrements it and completes the minigame when it hits zero.
            var guard = 0;
            while (grid.DivinationCount > 0 && guard++ < 64)
            {
                var cell = (ChooseCell ?? RandomHiddenCell)(grid) ?? RandomHiddenCell(grid);
                if (cell == null) break;
                await grid.CellClicked(cell);
            }
            Played++;
        }
        catch (Exception e)
        {
            Failed++;
            Probe.Log($"  (non-fatal) crystal sphere minigame: {e.Message}");
        }
    }

    /// <summary>All still-fogged cells, so a divination is never wasted on a clear one.</summary>
    public static List<CrystalSphereCell> HiddenCells(CrystalSphereMinigame grid)
    {
        var list = new List<CrystalSphereCell>();
        var size = grid.GridSize;
        for (var x = 0; x < size.X; x++)
            for (var y = 0; y < size.Y; y++)
                if (grid.cells[x, y] != null && grid.cells[x, y].IsHidden)
                    list.Add(grid.cells[x, y]);
        return list;
    }

    private static CrystalSphereCell RandomHiddenCell(CrystalSphereMinigame grid)
    {
        var hidden = HiddenCells(grid);
        if (hidden.Count > 0) return hidden[_rng.Next(hidden.Count)];
        var size = grid.GridSize;
        return size.X > 0 && size.Y > 0 ? grid.cells[0, 0] : null;
    }
}
