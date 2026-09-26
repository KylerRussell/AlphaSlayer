using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Models;

namespace AlphaSlayer.Probe;

/// <summary>
/// Potion usability and the run-level potion gate.
///
/// Split of responsibility, per the design:
///  - the RUN policy decides which potions a fight is allowed to spend (Allowed), because
///    saving a potion for the act boss is a run-level judgement the combat model cannot make:
///    it only ever sees one fight and would always drink now;
///  - the COMBAT policy decides WHEN, within a fight, to drink an allowed potion, because
///    that depends on hp, intents and the hand.
///
/// Usability mirrors the rules the potion popup applies, so the agent is offered exactly the
/// potions a player could actually click:
///   not queued, owner alive, Usage != Automatic (those fire themselves),
///   CombatOnly requires an in-progress combat on the player's side with actions enabled,
///   AnyTime is usable anywhere, and both Owner.CanUseOrRemovePotions and the potion's own
///   PassesCustomUsabilityCheck must hold.
/// </summary>
public static class Potions
{
    /// <summary>
    /// Whether potion actions are offered at all. CLOSED by default, deliberately.
    ///
    /// The combat-only env predates potions and its trained policy has no potion action in
    /// its action space; emitting one there would hand the encoder an action kind it has
    /// never seen, silently changing the action distribution mid-experiment. So the gate is
    /// something a driver opens explicitly, which is also exactly the run-level decision the
    /// design calls for: the run policy decides whether this fight may spend potions.
    /// </summary>
    public static bool Enabled;

    /// <summary>Slots the run policy released. Null (with Enabled) means "all of them".</summary>
    public static HashSet<int> Allowed;

    /// <summary>Opens the gate for a fight. Null list = allow every potion held.</summary>
    public static void SetGate(IEnumerable<int> allowedSlots)
    {
        Enabled = true;
        Allowed = allowedSlots == null ? null : new HashSet<int>(allowedSlots);
    }

    /// <summary>Closes the gate: no potion actions are offered.</summary>
    public static void ClearGate()
    {
        Enabled = false;
        Allowed = null;
    }

    public static bool IsGated(int slot) => !Enabled || (Allowed != null && !Allowed.Contains(slot));

    /// <summary>Can this potion be drunk right now, ignoring the run-level gate?</summary>
    public static bool IsUsableNow(PotionModel p, Player player, bool inCombat)
    {
        if (p == null || p.IsQueued || p.HasBeenRemovedFromState) return false;
        if (player.Creature == null || player.Creature.IsDead) return false;
        if (!Try(() => player.CanUseOrRemovePotions, true)) return false;
        if (!Try(() => p.PassesCustomUsabilityCheck, true)) return false;

        switch (p.Usage)
        {
            case PotionUsage.Automatic:   // fires on its own; never a manual choice
            case PotionUsage.None:
                return false;
            case PotionUsage.AnyTime:
                return true;
            case PotionUsage.CombatOnly:
                if (!inCombat) return false;
                var mgr = CombatManager.Instance;
                if (mgr == null || !mgr.IsInProgress || mgr.PlayerActionsDisabled) return false;
                var cs = player.Creature.CombatState;
                return cs != null && cs.CurrentSide == player.Creature.Side;
            default:
                return false;
        }
    }

    /// <summary>
    /// Legal potion actions for the current combat state.
    ///
    /// Targeting is the reason this cannot reuse the card action shape unchanged: a potion's
    /// TargetType may be an enemy, an ally, the player, or nothing at all, and IsValidTarget
    /// treats Self as a REAL target (unlike cards, where Self passes none). Each valid target
    /// therefore becomes its own action, with TargetIndex indexing the enemy list and -1
    /// meaning "no creature target" (self-targeted potions included, resolved by the engine).
    /// </summary>
    public static List<LegalAction> LegalPotionActions(Player player, CombatState combat)
    {
        var actions = new List<LegalAction>();
        if (!Enabled) return actions;
        var potions = player.Potions?.ToList() ?? new List<PotionModel>();
        var enemies = combat?.Enemies.ToList() ?? new List<Creature>();

        for (var slot = 0; slot < potions.Count; slot++)
        {
            var p = potions[slot];
            if (!IsUsableNow(p, player, inCombat: true)) continue;
            if (IsGated(slot)) continue;

            var targeted = false;
            for (var t = 0; t < enemies.Count; t++)
            {
                if (!enemies[t].IsAlive) continue;
                if (!Try(() => p.IsValidTarget(enemies[t]), false)) continue;
                targeted = true;
                actions.Add(new LegalAction
                {
                    Kind = "use_potion", HandIndex = slot, TargetIndex = t,
                    CardId = p.Id.Entry, Potion = p, Target = enemies[t],
                });
            }
            // Untargeted / self-targeted, only when no enemy target is legal, mirroring how
            // card actions avoid emitting one action per enemy for an all-enemy effect.
            if (!targeted && Try(() => p.IsValidTarget(null) || p.IsValidTarget(player.Creature), false))
                actions.Add(new LegalAction
                {
                    Kind = "use_potion", HandIndex = slot, TargetIndex = -1,
                    CardId = p.Id.Entry, Potion = p, Target = null,
                });
        }
        return actions;
    }

    /// <summary>
    /// Potions usable outside a fight (Usage == AnyTime): healing, gold, and the ones whose
    /// value is in where they are thrown rather than when.
    /// </summary>
    public static List<int> UsableOutOfCombat(Player player)
    {
        var slots = new List<int>();
        var potions = player.Potions?.ToList() ?? new List<PotionModel>();
        for (var i = 0; i < potions.Count; i++)
            if (IsUsableNow(potions[i], player, inCombat: false)) slots.Add(i);
        return slots;
    }

    /// <summary>Describes the belt for a policy: what is held, and whether it can be drunk.</summary>
    public static object[] Describe(Player player, bool inCombat)
    {
        var potions = player.Potions?.ToList() ?? new List<PotionModel>();
        return potions.Select((p, i) => (object)new Dictionary<string, object>
        {
            ["slot"] = i,
            ["potion"] = p.Id.Entry,
            ["potion_idx"] = Vocab.Get(Vocab.Potions, p.Id.Entry),
            ["usage"] = p.Usage.ToString(),
            ["target_type"] = (int)p.TargetType,
            ["usable"] = IsUsableNow(p, player, inCombat),
            ["gated"] = IsGated(i),
            ["vals"] = Obs.VarValues(p.DynamicVars),
        }).ToArray();
    }

    /// <summary>
    /// Fills the belt with random potions, for potion-focused combat training.
    ///
    /// Relics that grant potions do so ONCE, on pickup, and EndCombat only heals -- it does
    /// not restock. So a combat-only training run would see potions in the first episode and
    /// none afterwards, and the policy would get almost no gradient on use_potion. Refilling
    /// every episode makes potion usage something it can actually learn.
    /// </summary>
    public static int RefillBelt(Player player, Random rng, int target = 5)
    {
        if (player == null) return 0;
        var pool = ModelDb.AllPotions?.ToList();
        if (pool == null || pool.Count == 0) return 0;

        var added = 0;
        for (var guard = 0; guard < target * 2; guard++)
        {
            var held = player.Potions?.Count() ?? 0;
            if (held >= target) break;
            var canonical = pool[rng.Next(pool.Count)];
            try
            {
                var potion = canonical.ToMutable();
                potion.Owner = player;
                player.AddPotionInternal(potion);
            }
            catch (Exception e)
            {
                Probe.Log($"  potion refill stopped at {held}: {e.Message}");
                break;
            }
            // AddPotionInternal is a no-op when the belt is full, so verify progress rather
            // than trusting the call and looping forever.
            if ((player.Potions?.Count() ?? 0) <= held) break;
            added++;
        }
        return added;
    }

    private static T Try<T>(Func<T> f, T fallback)
    {
        try { return f(); } catch { return fallback; }
    }
}
