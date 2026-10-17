using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Models;

namespace AlphaSlayer.Probe;

/// <summary>
/// Diagnostics for "Card X could not be found in combat ID database!" after a fight restore:
/// when NetCombatCardDb.GetCardId is about to throw, log where the card is and what the db
/// holds (its subscriptions, ids, and whether they belong to the live combat's piles).
/// </summary>
public static class CardDbDiag
{
    private static readonly FieldInfo CardToId = AccessTools.Field(typeof(NetCombatCardDb), "_cardToId");
    private static readonly FieldInfo Subs = AccessTools.Field(typeof(NetCombatCardDb), "_subscriptions");
    private static readonly FieldInfo NextId = AccessTools.Field(typeof(NetCombatCardDb), "_nextId");

    public static readonly bool SelectTrace = Probe.HasFlag("probe-select-trace");

    public static void Install()
    {
        var h = new Harmony("alphaslayer.probe.carddbdiag");
        if (SelectTrace)
            h.Patch(AccessTools.Method(typeof(MegaCrit.Sts2.Core.Commands.CardCmd), "Transform",
                    new[] { typeof(System.Collections.Generic.IEnumerable<MegaCrit.Sts2.Core.Entities.Cards.CardTransformation>),
                            typeof(MegaCrit.Sts2.Core.Random.Rng), typeof(MegaCrit.Sts2.Core.Nodes.CommonUi.CardPreviewStyle) }),
                prefix: new HarmonyMethod(typeof(CardDbDiag), nameof(BeforeTransform)));
        h.Patch(AccessTools.Method(typeof(NetCombatCardDb), nameof(NetCombatCardDb.GetCardId)),
            prefix: new HarmonyMethod(typeof(CardDbDiag), nameof(Before)));
        if (!Probe.HasFlag("probe-carddb-trace")) return;
        h.Patch(AccessTools.Method(typeof(NetCombatCardDb), nameof(NetCombatCardDb.StartCombat)),
            postfix: new HarmonyMethod(typeof(CardDbDiag), nameof(Trace)));
        h.Patch(AccessTools.Method(typeof(NetCombatCardDb), "OnCombatEnded"),
            postfix: new HarmonyMethod(typeof(CardDbDiag), nameof(Trace)));
    }

    /// <summary>--probe-select-trace: where a card's owner stands when a transform starts.</summary>
    public static string Where(CardModel c)
    {
        var live = MegaCrit.Sts2.Core.Runs.RunManager.Instance.DebugOnlyGetState();
        return $"gen={Probe.RestoreGen} card={c} pile={c?.Pile?.Type} ownerLive={live?.Players.Contains(c?.Owner)} " +
               $"ownerCs={(c?.Owner?.Creature?.CombatState != null)} inProgress={CombatManager.Instance.IsInProgress}";
    }

    private static void BeforeTransform(System.Collections.Generic.IEnumerable<MegaCrit.Sts2.Core.Entities.Cards.CardTransformation> transformations)
    {
        try
        {
            var arr = transformations?.ToArray();
            var first = arr is { Length: > 0 } ? arr[0].Original : null;
            if (first != null && first.CombatState == null && first.Pile != null && first.Pile.IsCombatPile)
                Probe.Log($"SELTRACE transform with no combat state: {Where(first)}");
        }
        catch (Exception e) { Probe.Log($"SELTRACE diag failed: {e.Message}"); }
    }

    private static void Trace(NetCombatCardDb __instance, MethodBase __originalMethod)
    {
        var st = new System.Diagnostics.StackTrace(2, false).GetFrames()
            .Select(f => f.GetMethod()).Where(m => m != null).Take(8)
            .Select(m => $"{m.DeclaringType?.Name}.{m.Name}");
        Probe.Log($"CARDDB {__originalMethod.Name} subs={((IList)Subs.GetValue(__instance)).Count} " +
                  $"ids={((IDictionary)CardToId.GetValue(__instance)).Count} <- {string.Join(" < ", st)}");
    }

    private static void Before(NetCombatCardDb __instance, CardModel card)
    {
        try
        {
            if (!card.IsMutable || __instance.TryGetCardId(card, out _)) return;
            var ids = (IDictionary)CardToId.GetValue(__instance);
            var subs = (IList)Subs.GetValue(__instance);
            var pcs = card.Owner?.PlayerCombatState;
            var piles = pcs?.AllPiles.ToList();
            var subPiles = subs.Cast<object>().Select(s => AccessTools.Field(s.GetType(), "pile").GetValue(s)).ToList();
            var live = CombatManager.Instance.DebugOnlyGetState();
            var sameName = ids.Keys.Cast<CardModel>().Count(c => c.Id == card.Id);
            Probe.Log($"CARDDB MISS {card} pile={card.Pile?.Type} inPile={card.Pile?.Cards.Contains(card)} " +
                      $"pileIsLive={(piles != null && card.Pile != null && piles.Contains(card.Pile))} " +
                      $"ids={ids.Count} next={NextId.GetValue(__instance)} subs={subs.Count} " +
                      $"subsOnLivePiles={subPiles.Count(p => piles != null && piles.Contains(p))} " +
                      $"livePiles={piles?.Count} sameIdInDb={sameName} " +
                      $"ownerIsLive={live?.Players.Contains(card.Owner)} " +
                      $"piles=[{string.Join(",", piles?.Select(p => $"{p.Type}:{p.Cards.Count}/{p.Cards.Count(c => ids.Contains(c))}") ?? Array.Empty<string>())}]");
        }
        catch (Exception e) { Probe.Log($"CARDDB MISS (diag failed: {e.Message})"); }
    }
}
