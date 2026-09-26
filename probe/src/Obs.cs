using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Enchantments;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Localization.DynamicVars;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;

namespace AlphaSlayer.Probe;

/// <summary>One legal action: either play a card (optionally at a target) or end the turn.</summary>
public sealed class LegalAction
{
    public string Kind;        // "play" | "end_turn" | "select_card" | "select_done" | "use_potion"
    public int HandIndex = -1; // index into obs.hand; for select_* an option index; for use_potion a belt slot
    public int TargetIndex = -1; // index into obs.enemies, -1 when untargeted
    public string CardId;      // for logging / action-embedding lookup (potion id for use_potion)
    public CardModel Card;     // not serialised
    public PotionModel Potion; // not serialised; set for use_potion
    public Creature Target;    // not serialised

    public Dictionary<string, object> ToDict() => new()
    {
        ["kind"] = Kind,
        ["kind_idx"] = Kind switch
        {
            "play" => 0, "end_turn" => 1, "select_card" => 2, "select_done" => 3,
            "use_potion" => 4, _ => 5,
        },
        ["hand"] = HandIndex,
        ["target"] = TargetIndex,
        // A potion is not a card, so it must not share the card embedding table: an id that
        // means "Strike" in one and "Fire Potion" in the other would collide.
        ["card"] = Kind == "use_potion" ? "" : (CardId ?? ""),
        ["card_idx"] = Kind == "use_potion" ? Vocab.None : Vocab.Get(Vocab.Cards, CardId),
        ["potion"] = Kind == "use_potion" ? (CardId ?? "") : "",
        ["potion_idx"] = Kind == "use_potion" ? Vocab.Get(Vocab.Potions, CardId) : Vocab.None,
        // The card's numbers AGAINST THIS TARGET: Vulnerable on one enemy and not another
        // makes the same card worth different amounts depending on where it is aimed.
        ["vals"] = (Kind == "play" || Kind == "select_card") && Card != null
            ? Obs.CardValues(Card, Target) : Obs.NoValues,
        ["kw"] = Card != null ? Obs.Keywords(Card) : 0,
        ["upgrade"] = Card?.CurrentUpgradeLevel ?? 0,
    };
}

/// <summary>
/// Observation extraction.
///
/// Design notes that matter for the encoder:
/// - Hand is a SET of typed tokens; card identity is (cardId, upgrade, enchantment, amount,
///   status) because StS2 enchantments carry an amount and can be disabled.
/// - Draw pile is a MULTISET the player cannot order, so it is emitted as a bag of counts,
///   not as ordered tokens. Discard and exhaust are bags too (order carries no decision value).
/// - Enemy intents are emitted numerically (type + total damage + repeats) rather than as
///   opaque ids, since damage magnitude is the decision-relevant part.
/// </summary>
public static class Obs
{
    public static Dictionary<string, object> Extract(Player player, CombatState combat)
    {
        var pcs = player.PlayerCombatState;
        var hand = pcs?.Hand.Cards.ToList() ?? new List<CardModel>();
        var enemies = combat.Enemies.ToList();

        return new Dictionary<string, object>
        {
            ["turn"] = pcs?.TurnNumber ?? 0,
            ["phase"] = pcs?.Phase.ToString() ?? "None",
            ["phase_idx"] = (int)(pcs?.Phase ?? PlayerTurnPhase.None),
            ["side"] = combat.CurrentSide.ToString(),
            ["side_idx"] = (int)combat.CurrentSide,
            ["round"] = combat.RoundNumber,
            ["player"] = new Dictionary<string, object>
            {
                ["hp"] = player.Creature.CurrentHp,
                ["max_hp"] = player.Creature.MaxHp,
                ["block"] = player.Creature.Block,
                ["energy"] = pcs?.Energy ?? 0,
                ["max_energy"] = pcs?.MaxEnergy ?? 0,
                ["stars"] = pcs?.Stars ?? 0,
                ["gold"] = player.Gold,
                ["powers"] = Powers(player.Creature),
                ["relics"] = player.Relics.Select(r => (object)new Dictionary<string, object>
                {
                    ["relic"] = r.Id.Entry,
                    ["relic_idx"] = Vocab.Get(Vocab.Relics, r.Id.Entry),
                    // Counter state matters: a relic that fires "every 3rd card" is opaque
                    // without it, so the policy cannot time plays around the trigger.
                    ["counter"] = RelicCounter(r),
                    ["melted"] = r.IsMelted,
                    ["vals"] = VarValues(r.DynamicVars),
                }).ToArray(),
                ["potions"] = player.Potions.Select(p => Vocab.Get(Vocab.Potions, p.Id.Entry)).ToArray(),
                // Full belt state: which potions exist, whether they are drinkable right now,
                // and whether the run policy has released them for this fight.
                ["potion_slots"] = Potions.Describe(player, inCombat: true),
            },
            ["hand"] = hand.Select(c => CardToken(c, combat)).ToArray(),
            ["draw_bag"] = Bag(pcs?.DrawPile.Cards),
            ["discard_bag"] = Bag(pcs?.DiscardPile.Cards),
            ["exhaust_bag"] = Bag(pcs?.ExhaustPile.Cards),
            // The piles as multisets of (card, upgrade, keywords) with counts, sorted by id. The
            // bags above key by card id alone and lose upgrades. SORTED so the draw pile's real
            // order, which is hidden information, cannot leak through iteration order.
            ["draw_pile"] = Pile(pcs?.DrawPile.Cards),
            ["discard_pile"] = Pile(pcs?.DiscardPile.Cards),
            ["exhaust_pile"] = Pile(pcs?.ExhaustPile.Cards),
            ["draw_count"] = pcs?.DrawPile.Cards.Count ?? 0,
            ["discard_count"] = pcs?.DiscardPile.Cards.Count ?? 0,
            ["exhaust_count"] = pcs?.ExhaustPile.Cards.Count ?? 0,
            ["enemies"] = enemies.Select(EnemyToken).ToArray(),
            // Player-side creatures: Necrobinder's Osty. It has HP and absorbs hits, and it
            // was invisible, which fits Necrobinder being the weakest character in every room.
            ["allies"] = (pcs?.Pets ?? (IReadOnlyList<Creature>)Array.Empty<Creature>())
                .Select(EnemyToken).ToArray(),
            // Defect's orbs, in queue order (the front orb is the next one evoked). Values are
            // the game's own, so Focus is already applied.
            ["orbs"] = (pcs?.OrbQueue?.Orbs ?? (IReadOnlyList<OrbModel>)Array.Empty<OrbModel>())
                .Select(o => (object)new Dictionary<string, object>
                {
                    ["orb"] = o.Id.Entry,
                    ["orb_idx"] = Vocab.Get(Vocab.Orbs, o.Id.Entry),
                    ["passive"] = Try(() => (int)o.PassiveVal, 0),
                    ["evoke"] = Try(() => (int)o.EvokeVal, 0),
                }).ToArray(),
            ["orb_slots"] = Try(() => pcs?.OrbQueue?.Capacity ?? 0, 0),
            // Which fight this is. Risk is worth different amounts in a hallway and at the
            // act-3 boss, and a policy cannot tell them apart from the enemies alone.
            ["act"] = Try(() => combat.RunState?.CurrentActIndex ?? -1, -1),
            ["room_type"] = Try(() => combat.Encounter?.RoomType.ToString() ?? "", ""),
            ["room_idx"] = Try(() => combat.Encounter == null ? -1 : (int)combat.Encounter.RoomType, -1),
            ["encounter"] = Try(() => combat.Encounter?.Id.Entry ?? "", ""),
            ["act_floor"] = Try(() => player.RunState?.ActFloor ?? 0, 0),
            ["total_floor"] = Try(() => player.RunState?.TotalFloor ?? 0, 0),
            // Who is fighting, and which boss ends this act: the same context the run decisions
            // get, so one model reads both kinds of decision the same way.
            ["character"] = Try(() => player.Character?.Id.Entry ?? "", ""),
            ["boss"] = Try(() => combat.RunState?.Act?.BossEncounter?.Id.Entry ?? "", ""),
            ["boss_idx"] = Try(() => Vocab.Get(Vocab.Encounters, combat.RunState?.Act?.BossEncounter?.Id.Entry), 0),
        };
    }

    /// <summary>Card numbers are keyed by these slots; every card sends the same keys.</summary>
    public static readonly string[] ValueSlots =
        { "damage", "block", "repeat", "cards", "power", "energy", "hp_loss", "heal",
          "stars", "summon", "forge", "osty_damage", "gold", "max_hp" };

    public static readonly Dictionary<string, object> NoValues =
        ValueSlots.ToDictionary(k => k, k => (object)0);

    /// <summary>
    /// The numbers the player would see on the card right now, through the game's own preview.
    ///
    /// Without these a card is only an id embedding: the net must learn from experience alone
    /// what every card does, and cannot see Strength, Weak or Vulnerable change a card's worth.
    /// UpdateDynamicVarPreview is what NCard calls to draw a card, and it writes only
    /// PreviewValue, which the game documents as display-only. ClearPreview first, as NCard
    /// does, so a stale target from a previous call cannot leak into this one.
    ///
    /// When a card has several vars for one slot, a Calculated* var wins over a plain one
    /// because it holds the final total.
    /// </summary>
    public static Dictionary<string, object> CardValues(CardModel c, Creature target)
    {
        // Outside combat (a reward, a shop, the deck) there is nothing to preview against, and
        // the card may be a canonical model the preview must not write to. Its base values are
        // what the player reads off the card there.
        if (c.CombatState == null) return VarValues(c.DynamicVars);
        var d = ValueSlots.ToDictionary(k => k, k => (object)0);
        try
        {
            c.DynamicVars.ClearPreview();
            c.UpdateDynamicVarPreview(CardPreviewMode.Normal, target, c.DynamicVars);
            var calculated = new HashSet<string>();
            foreach (var v in c.DynamicVars.Values)
            {
                var slot = SlotOf(v);
                if (slot == null) continue;
                var isCalc = v is CalculatedVar;
                if (calculated.Contains(slot) && !isCalc) continue;
                d[slot] = (int)v.PreviewValue;
                if (isCalc) calculated.Add(slot);
            }
        }
        catch { }
        return d;
    }

    /// <summary>
    /// Base values of a var set, by slot, without touching the preview. Used for relics,
    /// potions, events and out-of-combat cards. The first var of each slot wins.
    /// </summary>
    public static Dictionary<string, object> VarValues(DynamicVarSet vars) => VarValues(vars?.Values);

    public static Dictionary<string, object> VarValues(IEnumerable<DynamicVar> vars)
    {
        var d = ValueSlots.ToDictionary(k => k, k => (object)0);
        var filled = new HashSet<string>();
        try
        {
            if (vars == null) return d;
            foreach (var v in vars)
            {
                var slot = SlotOf(v);
                if (slot == null || !filled.Add(slot)) continue;
                d[slot] = (int)v.BaseValue;
            }
        }
        catch { }
        return d;
    }

    /// <summary>
    /// Card keywords as a bitmask: bit k is CardKeyword value k (Exhaust=1, Ethereal=2,
    /// Innate=3, Unplayable=4, Retain=5, Sly=6, Eternal=7). What happens to a card after it
    /// is played or at end of turn is decided by these, and none of them was observed.
    /// </summary>
    public static int Keywords(CardModel c)
    {
        try
        {
            var m = 0;
            foreach (var k in c.Keywords)
                if ((int)k > 0 && (int)k < 31) m |= 1 << (int)k;
            return m;
        }
        catch { return 0; }
    }

    /// <summary>
    /// The card fields every context shares: hand, piles, deck, rewards, shop, prompts. One
    /// builder so the same card looks the same to the model wherever it appears.
    /// </summary>
    public static Dictionary<string, object> CardInfo(CardModel c)
    {
        var ench = c.Enchantment;
        return new Dictionary<string, object>
        {
            ["card"] = c.Id.Entry,
            ["card_idx"] = Vocab.Get(Vocab.Cards, c.Id.Entry),
            ["upgrade"] = c.CurrentUpgradeLevel,
            ["kw"] = Keywords(c),
            ["vals"] = CardValues(c, null),
            ["ench_idx"] = ench == null ? Vocab.None : Vocab.Get(Vocab.Enchantments, ench.Id.Entry),
            ["ench_amount"] = ench?.Amount ?? 0,
            // Afflictions are enchantments the ENEMY puts on your cards (Queen's Bound).
            ["affl_idx"] = c.Affliction == null ? Vocab.None : Vocab.Get(Vocab.Afflictions, c.Affliction.Id.Entry),
            ["affl_amount"] = Try(() => c.Affliction?.Amount ?? 0, 0),
            ["cost"] = Try(() => c.EnergyCost.GetResolved(), -1),
            ["type"] = (int)c.Type,
            ["rarity"] = (int)c.Rarity,
        };
    }

    private static string SlotOf(DynamicVar v) => v switch
    {
        GoldVar => "gold",
        MaxHpVar => "max_hp",
        OstyDamageVar => "osty_damage",
        CalculatedDamageVar c when v.Name.Contains("Osty") => "osty_damage",
        DamageVar or CalculatedDamageVar or ExtraDamageVar => "damage",
        BlockVar or CalculatedBlockVar => "block",
        RepeatVar => "repeat",
        CardsVar => "cards",
        EnergyVar => "energy",
        HpLossVar => "hp_loss",
        HealVar => "heal",
        StarsVar => "stars",
        SummonVar => "summon",
        ForgeVar => "forge",
        // PowerVar<TPower> is generic, so it is matched by name.
        _ when v.GetType().Name.StartsWith("PowerVar", StringComparison.Ordinal) => "power",
        _ => null,
    };

    private static Dictionary<string, object> CardToken(CardModel c, CombatState combat)
    {
        var d = CardInfo(c);
        var ench = c.Enchantment;
        d["ench_disabled"] = ench != null && ench.Status == EnchantmentStatus.Disabled;
        d["cost_x"] = Try(() => c.EnergyCost.CostsX, false);
        d["star_cost"] = Try(() => c.CurrentStarCost, -1);
        d["target_type"] = (int)c.TargetType;
        d["playable"] = Try(() => c.CanPlay(), false);
        return d;
    }

    private static Dictionary<string, object> EnemyToken(Creature e) => new()
    {
        ["monster"] = e.Monster?.Id.Entry ?? "",
        ["monster_idx"] = Vocab.Get(Vocab.Monsters, e.Monster?.Id.Entry),
        ["hp"] = e.CurrentHp,
        ["max_hp"] = e.MaxHp,
        ["block"] = e.Block,
        ["alive"] = e.IsAlive,
        ["hittable"] = e.IsHittable,
        ["powers"] = Powers(e),
        ["intents"] = Intents(e),
        // The move itself, not just its intent icons: a boss's script is a fixed state machine
        // (Aeonglass cycles three moves; Queen's second move is 99 Weak/Frail/Vulnerable), and
        // the move id is what tells the policy where in that script the fight is.
        ["move"] = Try(() => e.Monster?.NextMove?.Id ?? "", ""),
    };

    private static object[] Powers(Creature c) =>
        c.Powers.Select(p => (object)new Dictionary<string, object>
        {
            ["power_idx"] = Vocab.Get(Vocab.Powers, p.Id.Entry),
            ["power"] = p.Id.Entry,
            ["amount"] = Try(() => p.Amount, 0),
        }).ToArray();

    /// <summary>
    /// Intent as (type, total damage, repeats). Damage needs the target list and owner, and
    /// the calc can throw when the move is not fully resolved, so it is defensive.
    /// </summary>
    private static object[] Intents(Creature e)
    {
        var move = e.Monster?.NextMove;
        if (move?.Intents == null) return Array.Empty<object>();
        var targets = new[] { e };
        return move.Intents.Select(i => (object)new Dictionary<string, object>
        {
            ["type"] = i.IntentType.ToString(),
            ["type_idx"] = (int)i.IntentType,
            ["damage"] = i is AttackIntent a ? Try(() => a.GetTotalDamage(targets, e), 0) : 0,
            ["repeats"] = i is AttackIntent a2 ? Try(() => a2.Repeats, 0) : 0,
            // Status cards the move will shuffle into the player's deck.
            ["count"] = i is StatusIntent si ? Try(() => si.CardCount, 0) : 0,
        }).ToArray();
    }

    /// <summary>A pile as sorted (card, upgrade, keywords) groups with counts.</summary>
    public static object[] Pile(IReadOnlyList<CardModel> cards)
    {
        if (cards == null) return Array.Empty<object>();
        return cards
            .GroupBy(c => (id: c.Id.Entry, up: c.CurrentUpgradeLevel, kw: Keywords(c)))
            .OrderBy(g => g.Key.id, StringComparer.Ordinal).ThenBy(g => g.Key.up).ThenBy(g => g.Key.kw)
            .Select(g => (object)new Dictionary<string, object>
            {
                ["card"] = g.Key.id,
                ["card_idx"] = Vocab.Get(Vocab.Cards, g.Key.id),
                ["up"] = g.Key.up,
                ["kw"] = g.Key.kw,
                ["n"] = g.Count(),
            }).ToArray();
    }

    /// <summary>Bag of counts: the draw pile is a multiset the player cannot order.</summary>
    private static Dictionary<string, object> Bag(IReadOnlyList<CardModel> cards)
    {
        var d = new Dictionary<string, object>();
        if (cards == null) return d;
        foreach (var g in cards.GroupBy(c => Vocab.Get(Vocab.Cards, c.Id.Entry)))
            d[g.Key.ToString()] = g.Count();
        return d;
    }

    /// <summary>
    /// Every legal action in the current state. This is the action-embedding candidate set:
    /// variable length, so the policy head scores it rather than using a fixed vocabulary.
    /// </summary>
    public static List<LegalAction> LegalActions(Player player, CombatState combat)
    {
        var actions = new List<LegalAction>();

        // A parked card-selection prompt takes priority: the combat coroutine is suspended
        // inside a card effect waiting on this answer, so it is the ONLY thing the agent may
        // do until it is resolved. Multi-select is a sequence of single picks.
        var sel = CardSelect.Pending;
        if (sel != null)
        {
            var remaining = sel.Remaining;
            for (var i = 0; i < remaining.Count; i++)
                actions.Add(new LegalAction
                {
                    Kind = "select_card", HandIndex = i, TargetIndex = -1,
                    CardId = remaining[i].Id.Entry, Card = remaining[i],
                });
            if (sel.CanFinish && !sel.MustFinish)
                actions.Add(new LegalAction { Kind = "select_done" });
            return actions;
        }

        var pcs = player.PlayerCombatState;
        if (pcs == null || pcs.Phase != PlayerTurnPhase.Play) return actions;

        var hand = pcs.Hand.Cards.ToList();
        var enemies = combat.Enemies.ToList();

        for (var h = 0; h < hand.Count; h++)
        {
            var card = hand[h];
            var targeted = false;
            for (var t = 0; t < enemies.Count; t++)
            {
                if (!enemies[t].IsHittable) continue;
                if (!Try(() => card.CanPlayTargeting(enemies[t]), false)) continue;
                targeted = true;
                actions.Add(new LegalAction
                {
                    Kind = "play", HandIndex = h, TargetIndex = t,
                    CardId = card.Id.Entry, Card = card, Target = enemies[t],
                });
            }
            // Untargeted play, only when no target variant is legal (avoids duplicates for
            // cards that hit all enemies and would otherwise appear once per enemy).
            if (!targeted && Try(() => card.CanPlay(), false))
                actions.Add(new LegalAction
                {
                    Kind = "play", HandIndex = h, TargetIndex = -1,
                    CardId = card.Id.Entry, Card = card, Target = null,
                });
        }

        // Potions are a parallel resource to the hand: usable on the player's turn, gated by
        // the run policy rather than by energy.
        actions.AddRange(Potions.LegalPotionActions(player, combat));

        actions.Add(new LegalAction { Kind = "end_turn" });
        return actions;
    }

    /// <summary>Displayed counter for relics that have one, else -1.</summary>
    public static int RelicCounter(RelicModel r)
    {
        if (!Try(() => r.ShowCounter, false)) return -1;
        return Try(() =>
        {
            var v = r.DynamicVars.Values.FirstOrDefault();
            return v == null ? -1 : (int)v.BaseValue;
        }, -1);
    }

    private static T Try<T>(Func<T> f, T fallback)
    {
        try { return f(); } catch { return fallback; }
    }
}
