using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace AlphaSlayer.Probe;

/// <summary>One selectable entry in a rewards set (the gold pile, the potion, the card pick...).</summary>
public sealed class RewardOption
{
    public string Kind;        // gold | potion | relic | card | remove_card | special_card | other
    public int Index;          // index into RewardsSet.Rewards
    public int Amount;         // gold only
    public string ModelId = "";
    public int ModelIdx = Vocab.None;
    public int OptionCount;    // card rewards: how many cards are on offer

    public Dictionary<string, object> ToDict() => new()
    {
        ["kind"] = Kind,
        ["index"] = Index,
        ["amount"] = Amount,
        ["model"] = ModelId,
        ["model_idx"] = ModelIdx,
        ["options"] = OptionCount,
    };
}

/// <summary>One pick inside a card reward: a specific card, or an alternative (Skip, Reroll, ...).</summary>
public sealed class CardRewardOption
{
    public string Kind;        // "card" | "alt"
    public int Index;          // index into cards, or into alternatives
    public string Id = "";     // card id, or alternative option id ("Skip", "REROLL")
    public int CardIdx = Vocab.None;
    public int Upgrade;
    public int Rarity;
    public int Cost = -1;
    public int Type;
    /// <summary>Obs.CardInfo for a card option (numbers, keywords); null for an alternative.</summary>
    public Dictionary<string, object> Info;

    public Dictionary<string, object> ToDict()
    {
        var d = new Dictionary<string, object>
        {
            ["kind"] = Kind,
            ["index"] = Index,
            ["id"] = Id,
            ["card_idx"] = CardIdx,
            ["upgrade"] = Upgrade,
            ["rarity"] = Rarity,
            ["cost"] = Cost,
            ["type"] = Type,
        };
        d["kw"] = Info?["kw"] ?? 0;
        d["vals"] = Info?["vals"] ?? Obs.NoValues;
        return d;
    }
}

/// <summary>
/// Combat / treasure rewards, driven model-side.
///
/// Two engine seams do all the work, and both are the game's OWN test hooks rather than
/// anything we had to force:
///
///  - RewardsSet.testSelector (static, Func&lt;RewardsSet, Task&gt;). Under TestMode.IsOn,
///    RewardsSet.Offer() awaits this INSTEAD of showing NRewardsScreen. By then Rewards is
///    fully populated and hook-modified, so it is the exact list the player would see. It is
///    async, so an agent decision over TCP can be awaited here.
///
///  - ICardSelector.GetSelectedCardReward, on the selector we already push for in-combat card
///    selection. CardReward.OnSelect falls through to it whenever the screen is null, which is
///    always under TestMode.
///
/// The awkward part is that GetSelectedCardReward is SYNCHRONOUS, so the agent cannot be
/// consulted from inside it. It is therefore pre-armed from testSelector, which does have the
/// card list and can await. Pre-arming is a stack, not a single slot, because taking a relic
/// reward can itself spawn a nested rewards set.
///
/// Rewards must be offered explicitly: NCombatUi calls CombatRoom.OfferRoomEndRewards() after
/// combat, and headless nothing does. That call is also fire-and-forget in the engine
/// (TaskHelper.RunSafely), so OfferForCombatAsync re-implements it as an awaited sequence -
/// otherwise the next map travel would tear the room down mid-reward.
/// </summary>
public static class RunRewards
{
    /// <summary>
    /// Decides what to take. Given the offered options, returns the reward indices to take,
    /// in order. Anything not returned is skipped. Set by whichever driver is in charge
    /// (the maprun diagnostic, or the env server relaying to the policy).
    /// </summary>
    public static Func<Player, IReadOnlyList<RewardOption>, Task<List<int>>> ChooseRewards;

    /// <summary>
    /// Decides which card to take from a card reward. Returns picks in preference order:
    /// CardReward.OnSelect can ask more than once (relics that grant a second pick), and each
    /// ask consumes the next still-valid preference.
    /// </summary>
    public static Func<Player, IReadOnlyList<CardRewardOption>, Task<List<CardRewardOption>>> ChooseCard;

    private static readonly List<Func<IReadOnlyList<CardCreationResult>, IReadOnlyList<CardRewardAlternative>, CardRewardSelection>> _cardAnswers = new();

    /// <summary>
    /// Per-kind tally: offered / taken / declined / errored.
    ///
    /// "declined" is a LEGAL outcome, not a fault: Reward.OnSelect returns false when the
    /// engine refuses the reward, and the common case is a potion offered to a player whose
    /// belt is already full. Only "errored" means something is actually wrong.
    /// </summary>
    public static readonly Dictionary<string, int[]> Tally = new();
    private static void Count(string kind, int slot)
    {
        if (!Tally.TryGetValue(kind, out var t)) Tally[kind] = t = new int[4];
        t[slot]++;
    }

    public static void Install()
    {
        RewardsSet.testSelector = SelectAsync;
        Probe.Log("reward selector installed (agent-driven)");
    }

    public static void Uninstall()
    {
        RewardsSet.testSelector = null;
        _cardAnswers.Clear();
    }

    /// <summary>
    /// Offers the end-of-combat rewards for a finished combat room and waits for them to be
    /// resolved. Mirrors CombatRoom.OfferRoomEndRewards, but awaited.
    /// </summary>
    public static async Task OfferForCombatAsync(CombatRoom room, Player player)
    {
        if (!room.Encounter.ShouldGiveRewards) return;
        if (player.Creature.IsDead) return;

        var set = await RewardsCmd.GenerateForRoomEnd(player, room);
        // Prayer Wheel and friends modify the set here; skipping it would silently disable
        // a whole class of relic.
        await Hook.BeforeCombatRewardOffered(set, room.CombatState.RunState, room);
        await set.Offer();
    }

    /// <summary>The RewardsSet.testSelector body: describe, decide, apply, then close the set.</summary>
    private static async Task SelectAsync(RewardsSet set)
    {
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        var player = set.Player;

        // Offer() throws if the set is not completed when we return. We always close it
        // explicitly below, but skipping a card reward legitimately leaves it incomplete, so
        // the throw would fire on a perfectly normal decision.
        set.ThrowInTestIfRewardsNotTaken = false;

        var options = Describe(set);
        foreach (var o in options) Count(o.Kind, 0);
        var take = ChooseRewards != null
            ? await ChooseRewards(player, options)
            : Enumerable.Range(0, options.Count).ToList();   // default: take everything

        foreach (var i in take)
        {
            if (i < 0 || i >= set.Rewards.Count) continue;
            var reward = set.Rewards[i];
            if (reward.SuccessfullySelected) continue;

            var isCard = reward is CardReward;
            if (isCard) _cardAnswers.Add(await BuildCardAnswer(player, (CardReward)reward));
            try
            {
                await sync.SelectLocalReward(reward);
                Count(options[i].Kind, reward.SuccessfullySelected ? 1 : 2);
            }
            catch (Exception e)
            {
                Count(options[i].Kind, 3);
                Probe.Log($"  (non-fatal) reward {options[i].Kind}: {e.Message}");
            }
            finally { if (isCard && _cardAnswers.Count > 0) _cardAnswers.RemoveAt(_cardAnswers.Count - 1); }

            // Selecting a reward can complete the set (all taken); anything after that would
            // throw "not currently viewing any reward set".
            if (sync.IsRewardsSetCompleted(set)) return;
        }

        if (!sync.IsRewardsSetCompleted(set)) sync.SkipLocalRewardsSet();
    }

    /// <summary>
    /// Resolves the agent's card preference NOW (while we can await) into a synchronous
    /// answer function for GetSelectedCardReward.
    /// </summary>
    private static async Task<Func<IReadOnlyList<CardCreationResult>, IReadOnlyList<CardRewardAlternative>, CardRewardSelection>>
        BuildCardAnswer(Player player, CardReward reward)
    {
        // Populate() has already run (GenerateWithoutOffering), so Cards is the real offer.
        var cards = reward.Cards.ToList();
        var alts = CardRewardAlternative.Generate(reward);
        var offered = DescribeCards(cards, alts);

        var prefs = ChooseCard != null
            ? await ChooseCard(player, offered)
            : offered.Where(o => o.Kind == "card").Take(1).ToList();   // default: first card

        var queue = new Queue<CardRewardOption>(prefs ?? new List<CardRewardOption>());

        return (opts, alternatives) =>
        {
            while (queue.Count > 0)
            {
                var p = queue.Dequeue();
                if (p.Kind == "alt")
                {
                    if (p.Index >= 0 && p.Index < alternatives.Count)
                        return new CardRewardSelection { alternative = alternatives[p.Index] };
                    continue;
                }
                // Match by identity where possible: OnSelect removes taken cards from the
                // list, so a positional index goes stale after the first pick.
                var byId = opts.FirstOrDefault(c => c.Card.Id.Entry == p.Id);
                if (byId != null) return new CardRewardSelection { card = byId.Card };
            }
            // Nothing left we asked for. Prefer the engine's own Skip alternative over a
            // null selection, so OnSkipped runs and the choice is recorded in run history.
            var skip = alternatives.FirstOrDefault(a => a.OptionId == "Skip");
            return skip != null ? new CardRewardSelection { alternative = skip } : default;
        };
    }

    /// <summary>Called by the installed ICardSelector. Empty stack means nothing pre-armed.</summary>
    public static CardRewardSelection AnswerCardReward(
        IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
    {
        if (_cardAnswers.Count > 0) return _cardAnswers[_cardAnswers.Count - 1](options, alternatives);
        return options.Count > 0 ? new CardRewardSelection { card = options[0].Card } : default;
    }

    // ---------- description ----------

    public static List<RewardOption> Describe(RewardsSet set)
    {
        var list = new List<RewardOption>();
        for (var i = 0; i < set.Rewards.Count; i++)
        {
            var r = set.Rewards[i];
            var o = new RewardOption { Index = i, Kind = "other" };
            switch (r)
            {
                case GoldReward g: o.Kind = "gold"; o.Amount = g.Amount; break;
                case PotionReward p:
                    o.Kind = "potion";
                    o.ModelId = p.Potion?.Id.Entry ?? "";
                    o.ModelIdx = Vocab.Get(Vocab.Potions, o.ModelId);
                    break;
                case RelicReward rr:
                    o.Kind = "relic";
                    o.ModelId = rr.Relic?.Id.Entry ?? "";
                    o.ModelIdx = Vocab.Get(Vocab.Relics, o.ModelId);
                    break;
                case CardReward cr: o.Kind = "card"; o.OptionCount = cr.Cards.Count(); break;
                case CardRemovalReward: o.Kind = "remove_card"; break;
                case SpecialCardReward: o.Kind = "special_card"; break;
            }
            list.Add(o);
        }
        return list;
    }

    public static List<CardRewardOption> DescribeCards(
        IReadOnlyList<MegaCrit.Sts2.Core.Models.CardModel> cards, IReadOnlyList<CardRewardAlternative> alts)
    {
        var list = new List<CardRewardOption>();
        for (var i = 0; i < cards.Count; i++)
        {
            var c = cards[i];
            list.Add(new CardRewardOption
            {
                Kind = "card", Index = i, Id = c.Id.Entry,
                CardIdx = Vocab.Get(Vocab.Cards, c.Id.Entry),
                Upgrade = c.CurrentUpgradeLevel,
                Rarity = (int)c.Rarity,
                Cost = Try(() => c.EnergyCost.GetResolved(), -1),
                Type = (int)c.Type,
                Info = Try(() => Obs.CardInfo(c), null),
            });
        }
        for (var i = 0; i < alts.Count; i++)
            list.Add(new CardRewardOption { Kind = "alt", Index = i, Id = alts[i].OptionId });
        return list;
    }

    private static T Try<T>(Func<T> f, T fallback)
    {
        try { return f(); } catch { return fallback; }
    }
}
