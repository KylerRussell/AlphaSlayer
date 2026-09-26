using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MegaCrit.Sts2.Core.Models;

namespace AlphaSlayer.Probe;

/// <summary>
/// Stable string-id -> integer index maps, built once from the live ModelDb and dumped to
/// vocab.json so the Python side can size its embedding tables and decode observations.
///
/// Index 0 is reserved for "none/unknown" in every table, so a null enchantment or an
/// unrecognised id encodes as 0 rather than needing a separate mask.
///
/// These indices are only stable for a given game build. Re-dump on every version bump; the
/// report records the build so a mismatch is detectable.
/// </summary>
public static class Vocab
{
    public const int None = 0;

    public static Dictionary<string, int> Cards = new();
    public static Dictionary<string, int> Relics = new();
    public static Dictionary<string, int> Powers = new();
    public static Dictionary<string, int> Potions = new();
    public static Dictionary<string, int> Monsters = new();
    public static Dictionary<string, int> Enchantments = new();
    public static Dictionary<string, int> Afflictions = new();
    public static Dictionary<string, int> Orbs = new();
    public static Dictionary<string, int> Encounters = new();

    private static bool _built;

    public static void Build()
    {
        if (_built) return;
        Cards = Index(Safe(() => ModelDb.AllCards.Select(c => c.Id.Entry)));
        Relics = Index(Safe(() => ModelDb.AllRelics.Select(m => m.Id.Entry)));
        Powers = Index(Safe(() => ModelDb.AllPowers.Select(m => m.Id.Entry)));
        Potions = Index(Safe(() => ModelDb.AllPotions.Select(m => m.Id.Entry)));
        Monsters = Index(Safe(() => ModelDb.Monsters.Select(m => m.Id.Entry)));
        // ModelDb.Monsters lists only monsters named by an encounter. Player pets (Osty) and
        // anything summoned mid-fight that no encounter lists fell through to index 0. They
        // are APPENDED, not merged into the sort, so every pre-existing monster keeps its index
        // and older checkpoints' monster embeddings stay aligned.
        var extraMonsters = AppendMissing(Monsters, Safe(() => AllModelIds<MonsterModel>("Monster")));
        Enchantments = Index(Safe(() => ModelDb.DebugEnchantments.Select(m => m.Id.Entry)));
        // Card afflictions (Bound, Hexed, ...) are the enemy-applied twin of enchantments.
        Afflictions = Index(Safe(() => ModelDb.DebugAfflictions.Select(m => m.Id.Entry)));
        Orbs = Index(Safe(AllOrbIds));
        Encounters = Index(Safe(() => ModelDb.AllEncounters.Select(m => m.Id.Entry)));
        _built = true;
        Probe.Log($"vocab: cards={Cards.Count} relics={Relics.Count} powers={Powers.Count} " +
                  $"potions={Potions.Count} monsters={Monsters.Count} enchantments={Enchantments.Count} " +
                  $"afflictions={Afflictions.Count} orbs={Orbs.Count} " +
                  $"(+{extraMonsters} monsters outside encounter lists)");
    }

    /// <summary>
    /// Every concrete OrbModel type in the game, through ModelDb.Orb&lt;T&gt;().
    ///
    /// ModelDb.Orbs lists only the four orbs that can be rolled at random, which leaves out
    /// Glass. The generic accessor is the public way to reach any orb type's canonical model,
    /// so its Id.Entry is the same one live orbs report.
    /// </summary>
    private static IEnumerable<string> AllOrbIds() => AllModelIds<OrbModel>("Orb");

    /// <summary>Id.Entry of every concrete T in the game, via the generic ModelDb accessor.</summary>
    private static IEnumerable<string> AllModelIds<T>(string accessor) where T : AbstractModel
    {
        var get = typeof(ModelDb).GetMethod(accessor, Type.EmptyTypes);
        return typeof(T).Assembly.GetTypes()
            .Where(t => !t.IsAbstract && t.IsSubclassOf(typeof(T)))
            .Select(t => ((T)get!.MakeGenericMethod(t).Invoke(null, null)!).Id.Entry);
    }

    /// <summary>Adds ids the table lacks at the END, in sorted order; returns how many.</summary>
    private static int AppendMissing(Dictionary<string, int> table, IEnumerable<string> ids)
    {
        var n = 0;
        foreach (var id in ids.Where(x => x != null && !table.ContainsKey(x))
                     .Distinct(StringComparer.Ordinal).OrderBy(x => x, StringComparer.Ordinal))
        {
            table[id] = table.Count + 1;
            n++;
        }
        return n;
    }

    private static IEnumerable<string> Safe(Func<IEnumerable<string>> f)
    {
        try { return f().ToList(); } catch (Exception e) { Probe.Log($"vocab source failed: {e.Message}"); return Array.Empty<string>(); }
    }

    /// <summary>Sorted so indices are reproducible across runs of the same build.</summary>
    private static Dictionary<string, int> Index(IEnumerable<string> ids)
    {
        var d = new Dictionary<string, int>(StringComparer.Ordinal);
        var i = 1; // 0 reserved for none
        foreach (var id in ids.Where(x => x != null).Distinct(StringComparer.Ordinal).OrderBy(x => x, StringComparer.Ordinal))
            d[id] = i++;
        return d;
    }

    public static int Get(Dictionary<string, int> table, string id) =>
        id != null && table.TryGetValue(id, out var v) ? v : None;

    public static void Dump(string outDir, string gameVersion)
    {
        var payload = new Dictionary<string, object>
        {
            ["game_version"] = gameVersion,
            ["reserved_none_index"] = None,
            ["sizes"] = new Dictionary<string, object>
            {
                ["cards"] = Cards.Count + 1,
                ["relics"] = Relics.Count + 1,
                ["powers"] = Powers.Count + 1,
                ["potions"] = Potions.Count + 1,
                ["monsters"] = Monsters.Count + 1,
                ["enchantments"] = Enchantments.Count + 1,
                ["afflictions"] = Afflictions.Count + 1,
                ["orbs"] = Orbs.Count + 1,
                ["encounters"] = Encounters.Count + 1,
                ["room_types"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.Rooms.RoomType)).Length,
                ["card_types"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.Entities.Cards.CardType)).Length,
                ["card_rarities"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.Entities.Cards.CardRarity)).Length,
                ["target_types"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.Entities.Cards.TargetType)).Length,
                ["intent_types"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.MonsterMoves.Intents.IntentType)).Length,
                ["turn_phases"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.Combat.PlayerTurnPhase)).Length,
            },
            ["cards"] = Cards, ["relics"] = Relics, ["powers"] = Powers,
            ["potions"] = Potions, ["monsters"] = Monsters, ["enchantments"] = Enchantments,
            ["afflictions"] = Afflictions, ["orbs"] = Orbs, ["encounters"] = Encounters,
            ["enums"] = new Dictionary<string, object>
            {
                ["card_type"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.Entities.Cards.CardType)),
                ["card_rarity"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.Entities.Cards.CardRarity)),
                ["target_type"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.Entities.Cards.TargetType)),
                ["intent_type"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.MonsterMoves.Intents.IntentType)),
                ["turn_phase"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.Combat.PlayerTurnPhase)),
                ["room_type"] = Enum.GetNames(typeof(MegaCrit.Sts2.Core.Rooms.RoomType)),
            },
        };
        var path = Path.Combine(outDir, "vocab.json");
        File.WriteAllText(path, Probe.ToJson(payload));
        Probe.Log($"vocab written: {path}");
    }
}
