using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace AlphaSlayer.Probe;

/// <summary>
/// Phase 3 — snapshot cost, which decides whether AlphaZero-style search is affordable.
///
/// Context: SerializableRun captures acts, players, map, odds, relic bags and the full RNG
/// set, but NOT combat state. There is no mid-combat save in the shipped game. What DOES
/// exist is a deep-clone discipline on the model graph (MemberwiseClone + DeepCloneFields +
/// AfterCloned, exposed as ToMutable/MutableClone/ClonePreservingMutability) which CombatState
/// and RunState already use.
///
/// So the question this phase answers is: how expensive are those primitives, and do they
/// actually produce independent state? An MCTS node expansion needs this in the microseconds,
/// not milliseconds.
/// </summary>
public static class Snapshot
{
    public static Task RunAsync()
    {
        var r = new Dictionary<string, object>();
        var iters = Probe.ArgInt("probe-snapshot-iters", 200);
        void Step(string m) => Probe.Log($"    step: {m}");
        Step("begin");
        r["iters"] = iters;

        // --- 2. per-card clone cost + independence ------------------------------------------
        try
        {
            // ToMutable() asserts the receiver is canonical; ClonePreservingMutability() is
            // the primitive for copying an already-mutable model, which is what a search node
            // would actually do.
            Step("card clone");
            var card = ModelDb.AllCards.First().ToMutable();
            var sw = Stopwatch.StartNew();
            for (var i = 0; i < iters; i++) card.ClonePreservingMutability();
            sw.Stop();
            r["card_clone_us"] = sw.Elapsed.TotalMilliseconds * 1000.0 / iters;

            // Independence: upgrading the clone must not touch the original.
            Step("independence");
            var clone = (CardModel)card.ClonePreservingMutability();
            var beforeOriginal = card.IsUpgraded;
            clone.UpgradeInternal();
            r["card_clone_independent"] = card.IsUpgraded == beforeOriginal && clone.IsUpgraded;
            Probe.Log($"card clone: {r["card_clone_us"]:F2}us independent={r["card_clone_independent"]}");
        }
        catch (Exception e)
        {
            r["card_clone_error"] = e.ToString();
            Probe.Log($"card clone failed: {e.Message}");
        }

        // --- 1. run-level snapshot: the known-good path -------------------------------------
        try
        {
            // RunManager.Instance itself faults when no run has been launched, so only touch
            // it when the throughput phase created one in this same invocation.
            if (!Probe.Phases.Contains("throughput"))
            {
                Step("no run launched this invocation; skipping run-level snapshot");
                r["run_snapshot"] = "skipped (needs --probe=throughput,snapshot)";
                throw new InvalidOperationException("no run");
            }
            Step("RunManager.Instance");
            var rm = RunManager.Instance;
            // Warm up (first call JITs the graph walk).
            var warm = rm.ToSave(null);

            var sw = Stopwatch.StartNew();
            for (var i = 0; i < iters; i++) rm.ToSave(null);
            sw.Stop();
            r["run_tosave_us"] = sw.Elapsed.TotalMilliseconds * 1000.0 / iters;

            // JSON round trip, as an upper bound on a naive save/load snapshot.
            var json = JsonSerializationUtility.ToJson(warm);
            r["run_json_bytes"] = json.Length;

            sw.Restart();
            for (var i = 0; i < iters; i++) JsonSerializationUtility.ToJson(warm);
            sw.Stop();
            r["run_json_serialize_us"] = sw.Elapsed.TotalMilliseconds * 1000.0 / iters;

            sw.Restart();
            for (var i = 0; i < iters; i++) JsonSerializationUtility.FromJson<SerializableRun>(json);
            sw.Stop();
            r["run_json_deserialize_us"] = sw.Elapsed.TotalMilliseconds * 1000.0 / iters;

            Probe.Log($"run snapshot: ToSave={r["run_tosave_us"]:F1}us " +
                      $"json_ser={r["run_json_serialize_us"]:F1}us " +
                      $"json_de={r["run_json_deserialize_us"]:F1}us " +
                      $"size={r["run_json_bytes"]}B");
        }
        catch (Exception e)
        {
            r["run_snapshot_error"] = e.ToString();
            Probe.Log($"run snapshot failed: {e.Message}");
        }

        // --- 3. combat-state reachability ---------------------------------------------------
        // We do not expect a turnkey CombatState clone. What matters is the SIZE of the graph
        // an MCTS node would have to copy: piles + creatures + powers + relics.
        try
        {
            Step("combat sizing");
            var combat = CombatManager.Instance?.DebugOnlyGetState();
            if (combat == null)
            {
                r["combat_state"] = "none live (run throughput phase in the same invocation to populate)";
                Probe.Log("no live combat state; skipping combat snapshot sizing");
            }
            else
            {
                var creatures = combat.Creatures.ToList();
                var cards = combat.Players
                    .Where(p => p.PlayerCombatState != null)
                    .SelectMany(p => p.PlayerCombatState.AllCards).ToList();

                r["combat_creatures"] = creatures.Count;
                r["combat_cards"] = cards.Count;

                var sw = Stopwatch.StartNew();
                for (var i = 0; i < Math.Max(1, iters / 10); i++)
                    foreach (var c in cards) c.ClonePreservingMutability();
                sw.Stop();
                var perPass = sw.Elapsed.TotalMilliseconds * 1000.0 / Math.Max(1, iters / 10);
                r["combat_all_cards_clone_us"] = perPass;

                // Rough lower bound on a hand-rolled combat snapshot: cards dominate.
                r["combat_snapshot_lower_bound_us"] = perPass;
                r["combat_snapshots_per_s_lower_bound"] = perPass > 0 ? 1e6 / perPass : 0;

                Probe.Log($"combat graph: {creatures.Count} creatures, {cards.Count} cards; " +
                          $"cloning all cards = {perPass:F1}us " +
                          $"(=> <= {r["combat_snapshots_per_s_lower_bound"]:F0} snapshots/s)");
            }
        }
        catch (Exception e)
        {
            r["combat_snapshot_error"] = e.ToString();
            Probe.Log($"combat snapshot sizing failed: {e.Message}");
        }

        Probe.Report["snapshot"] = r;
        return Task.CompletedTask;
    }
}
