using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Models;

namespace AlphaSlayer.Probe;

/// <summary>
/// Phase 1 — content census. No API risk, runs first, and produces the vocabulary sizes the
/// encoder needs (card-ID embedding rows, relic/power/enchantment tables) straight from the
/// live ModelDb rather than from guesses.
/// </summary>
public static class Inventory
{
    public static Task RunAsync()
    {
        var r = new Dictionary<string, object>();

        // --- vocabulary sizes: these set your embedding table dimensions ---
        r["cards"] = Count(() => ModelDb.AllCards);
        r["relics"] = Count(() => ModelDb.AllRelics);
        r["powers"] = Count(() => ModelDb.AllPowers);
        r["potions"] = Count(() => ModelDb.AllPotions);
        r["monsters"] = Count(() => ModelDb.Monsters);
        r["encounters"] = Count(() => ModelDb.AllEncounters);
        r["events"] = Count(() => ModelDb.AllEvents);
        r["ancients"] = Count(() => ModelDb.AllAncients);
        r["orbs"] = Count(() => ModelDb.Orbs);
        r["enchantments"] = Count(() => ModelDb.DebugEnchantments);

        // --- characters: IsPlayable separates the roster from Deprived/DeprecatedCharacter ---
        try
        {
            var chars = ModelDb.AllCharacters.ToList();
            r["characters_total"] = chars.Count;
            r["characters_playable"] = chars.Where(c => c.IsPlayable)
                .Select(c => c.Id.Entry).OrderBy(x => x).ToArray();
            r["characters_not_playable"] = chars.Where(c => !c.IsPlayable)
                .Select(c => c.Id.Entry).OrderBy(x => x).ToArray();
        }
        catch (Exception e) { r["characters_error"] = e.Message; }

        // --- acts: several ActModels share an Index (themed variants). The number of DISTINCT
        //     indices is acts-per-run; variants at the same index swap the encounter/boss/event
        //     pools, so act identity is a state variable, not a constant. ---
        try
        {
            var acts = ModelDb.Acts.ToList();
            r["act_models_total"] = acts.Count;
            var byIndex = acts.GroupBy(a => a.Index).OrderBy(g => g.Key).ToList();
            r["acts_per_run"] = byIndex.Count;
            var actDetail = new List<object>();
            foreach (var g in byIndex)
            {
                actDetail.Add(new Dictionary<string, object>
                {
                    ["index"] = g.Key,
                    ["variants"] = g.Select(a => new Dictionary<string, object>
                    {
                        ["id"] = a.Id.Entry,
                        ["is_default"] = a.IsDefault,
                        ["encounters"] = SafeCount(() => a.AllEncounters),
                        ["elites"] = SafeCount(() => a.AllEliteEncounters),
                        ["bosses"] = SafeCount(() => a.AllBossEncounters),
                        ["events"] = SafeCount(() => a.AllEvents),
                    }).ToArray()
                });
            }
            r["acts"] = actDetail;
        }
        catch (Exception e) { r["acts_error"] = e.Message; }

        // --- enchantment names: one nullable slot per card, so this is a small categorical ---
        try
        {
            r["enchantment_ids"] = ModelDb.DebugEnchantments
                .Select(e => e.Id.Entry).OrderBy(x => x).ToArray();
        }
        catch (Exception e) { r["enchantment_ids_error"] = e.Message; }

        Probe.Report["inventory"] = r;
        foreach (var kv in r.Where(k => k.Value is int))
            Probe.Log($"  {kv.Key,-24} {kv.Value}");
        Probe.Log($"  acts_per_run             {(r.TryGetValue("acts_per_run", out var ap) ? ap : "?")}");
        Probe.Log($"  playable                 {string.Join(",", (string[])(r.TryGetValue("characters_playable", out var cp) ? cp : Array.Empty<string>()))}");

        return Task.CompletedTask;
    }

    private static object Count<T>(Func<IEnumerable<T>> f)
    {
        try { return f().Count(); }
        catch (Exception e) { return $"error: {e.Message}"; }
    }

    private static object SafeCount<T>(Func<IEnumerable<T>> f)
    {
        try { return f().Count(); }
        catch (Exception e) { return $"error: {e.Message}"; }
    }
}
