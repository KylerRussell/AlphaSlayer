using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Entities.TreasureRelicPicking;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace AlphaSlayer.Probe;


/// <summary>One purchasable shop entry as offered to a policy.</summary>
public sealed class ShopOption
{
    public int Index;
    public string Kind = "";       // card | relic | potion | remove_card
    public string Id = "";
    public int ModelIdx = Vocab.None;
    public int Cost;
    public bool Affordable;
    public bool OnSale;
    public int Rarity = -1;
    public int Upgrade;
    public int Keywords;
    public Dictionary<string, object> Vals = Obs.NoValues;

    public Dictionary<string, object> ToDict() => new()
    {
        ["index"] = Index,
        ["kind"] = Kind,
        ["id"] = Id,
        ["model_idx"] = ModelIdx,
        ["cost"] = Cost,
        ["affordable"] = Affordable,
        ["on_sale"] = OnSale,
        ["rarity"] = Rarity,
        ["upgrade"] = Upgrade,
        ["kw"] = Keywords,
        ["vals"] = Vals,
    };
}

/// <summary>One event option as offered to a policy.</summary>
public sealed class EventOptionInfo
{
    public int Index;
    public string TextKey = "";
    public bool Locked;
    public bool IsProceed;
    public bool WasChosen;
    public string RelicId = "";
    /// <summary>The engine flags options that are known to be lethal for this player.</summary>
    public bool WillKill;
    public string EventId = "";
    /// <summary>The event's numbers that THIS option's text refers to, by value slot.</summary>
    public Dictionary<string, object> Vals = Obs.NoValues;
    /// <summary>Cards the option shows a hover tip for: typically what it adds to the deck.</summary>
    public List<string> Cards = new();
    /// <summary>
    /// ALL of the event's numbers, unattributed. A fallback for options whose text names no
    /// var, and what the attribution coverage audit compares against.
    /// </summary>
    public Dictionary<string, object> EventVals = Obs.NoValues;
    /// <summary>
    /// Every hover tip's id: keywords like Transform, enchantments, powers, relics. Some options
    /// state their effect ONLY this way (Symbiote's "transform a card" names no number).
    /// </summary>
    public List<string> Tips = new();

    public Dictionary<string, object> ToDict() => new()
    {
        ["index"] = Index,
        ["key"] = TextKey,
        ["locked"] = Locked,
        ["proceed"] = IsProceed,
        ["chosen"] = WasChosen,
        ["relic"] = RelicId,
        ["relic_idx"] = Vocab.Get(Vocab.Relics, RelicId),
        ["will_kill"] = WillKill,
        ["event"] = EventId,
        ["vals"] = Vals,
        ["event_vals"] = EventVals,
        ["tips"] = Tips.ToArray(),
        ["cards"] = Cards.ToArray(),
        ["card_idxs"] = Cards.Select(c => (object)Vocab.Get(Vocab.Cards, c)).ToArray(),
    };
}

/// <summary>One rest-site option as offered to a policy.</summary>
public sealed class RestOption
{
    public int Index;
    public string OptionId = "";
    public bool Enabled;

    public Dictionary<string, object> ToDict() => new()
    {
        ["index"] = Index,
        ["option"] = OptionId,
        ["enabled"] = Enabled,
    };
}

/// <summary>
/// Model-layer resolvers for the non-combat rooms.
///
/// Most of the run layer needs nothing here: entering a shop, an event or a rest site and
/// walking away is a legal (if wasteful) way to play, and the engine cleans up after itself.
/// The treasure room is the exception, and it is worth being precise about why, because it
/// is the shape we should expect to meet again:
///
///  1. TreasureRoomRelicSynchronizer.BeginRelicPicking() opens a picking session on room
///     entry and the session is closed ONLY by an actual pick or an explicit skip. Leaving
///     without answering leaves it open, and the NEXT chest throws
///     "Attempted to start new relic picking session while one was already occurring!".
///     So the chest is a mandatory decision, not an optional one.
///
///  2. The synchronizer decides WHO gets which relic and raises RelicsAwarded - but the
///     actual grant, RelicCmd.Obtain, lives in NTreasureRoomRelicCollection, i.e. in the
///     node layer. Headless nothing subscribes, so the relic is decided and then dropped on
///     the floor. Combat rewards do not have this problem (RewardsSet.testSelector applies
///     them model-side under TestMode); the chest does.
///
/// Both halves are handled here.
/// </summary>
public static class RunRooms
{
    /// <summary>
    /// Resolves a treasure room: takes the gold, then answers the relic prompt.
    /// </summary>
    /// <param name="choice">
    /// Index into the offered relics, or null to skip. Currently always 0 from the diagnostic
    /// driver - WHICH relic to take is a run-level decision and belongs to the run policy, not
    /// here. This just makes the decision reachable and correctly applied.
    /// </param>
    public static async Task<List<RelicModel>> ResolveTreasureAsync(Player player, TreasureRoom room, int? choice)
    {
        var sync = RunManager.Instance.TreasureRoomRelicSynchronizer;
        var granted = new List<RelicModel>();

        // Gold. The node calls this from OpenChest(); it is a plain model-layer award.
        try { await room.DoNormalRewards(); }
        catch (Exception e) { Probe.Log($"  (non-fatal) treasure gold: {e.Message}"); }

        if (sync.CurrentRelics == null || sync.CurrentRelics.Count == 0) return granted;

        // Subscribe BEFORE picking: OnPicked awards and ends the session synchronously inside
        // the action, so a subscription added afterwards would miss the event entirely.
        var pending = new List<RelicPickingResult>();
        void OnAwarded(List<RelicPickingResult> results) => pending.AddRange(results);
        sync.RelicsAwarded += OnAwarded;
        try
        {
            var n = sync.CurrentRelics.Count;
            var idx = choice.HasValue ? Math.Clamp(choice.Value, 0, n - 1) : (int?)null;
            sync.PickRelicLocally(idx);

            // PickRelicLocally enqueues a PickRelicAction; OnPicked runs when it executes.
            for (var i = 0; i < 200 && sync.CurrentRelics != null; i++)
                await CombatHarness.Suspend();
        }
        finally { sync.RelicsAwarded -= OnAwarded; }

        foreach (var r in pending)
        {
            if (r.type == RelicPickingResultType.Skipped || r.player != player) continue;
            var relic = r.relic.ToMutable();
            try { await RelicCmd.Obtain(relic, r.player); granted.Add(relic); }
            catch (Exception e) { Probe.Log($"  (non-fatal) obtain {relic.Id.Entry}: {e.Message}"); }
        }

        // A singleplayer skip is only closed out on room exit, so mirror the node's contract.
        if (sync.CurrentRelics != null) sync.OnRoomExited();
        return granted;
    }

    // ---------- rest sites ----------

    /// <summary>
    /// Resolves a rest site. Pure model layer: RestSiteSynchronizer owns the option list and
    /// ChooseLocalOption runs the option's OnSelect.
    ///
    /// The loop matters. Choosing an option does not necessarily end the rest site: unless
    /// Hook.ShouldDisableRemainingRestSiteOptions says otherwise the chosen option is merely
    /// removed and the rest stay available, which is how multi-use campfires work. So the
    /// policy is asked repeatedly until it declines or nothing is left.
    ///
    /// A failed option (Smith with no upgradable card) returns false and is NOT removed, so
    /// re-offering it forever would spin; those are dropped locally after one failure.
    /// </summary>
    public static async Task<List<string>> ResolveRestSiteAsync(
        Player player, RestSiteRoom room,
        Func<Player, IReadOnlyList<RestOption>, Task<int>> chooser,
        Func<PendingSelection, Task> answerSelection)
    {
        var sync = RunManager.Instance.RestSiteSynchronizer;
        var chosen = new List<string>();
        var refused = new HashSet<string>();

        for (var guard = 0; guard < 8; guard++)
        {
            var opts = sync.GetLocalOptions();
            if (opts == null || opts.Count == 0) break;

            var offered = new List<RestOption>();
            for (var i = 0; i < opts.Count; i++)
                if (!refused.Contains(opts[i].OptionId))
                    offered.Add(new RestOption { Index = i, OptionId = opts[i].OptionId, Enabled = opts[i].IsEnabled });
            if (offered.Count == 0) break;

            var pick = chooser != null ? await chooser(player, offered) : offered[0].Index;
            if (pick < 0 || pick >= opts.Count) break;    // policy declined

            var id = opts[pick].OptionId;
            // Smith parks on a card selection inside OnSelect, so the call has to be driven
            // rather than simply awaited.
            var ok = await DriveWithSelections(sync.ChooseLocalOption(pick), answerSelection);
            if (ok) chosen.Add(id); else refused.Add(id);
        }
        return chosen;
    }

    // ---------- shared plumbing ----------

    /// <summary>
    /// Awaits a task that may park on a card-selection prompt part way through, answering
    /// each prompt as it is raised.
    ///
    /// Several out-of-combat actions do this - smithing at a campfire, an event that removes
    /// a card, a shop's card-removal service - and the selection is answered through the SAME
    /// ICardSelector the combat loop uses. Awaiting the outer task on its own would deadlock:
    /// it cannot finish until the prompt is answered, and nothing else is running to answer it.
    ///
    /// Awaits Raised rather than polling, for the same reason the combat loop does: spinning
    /// on a timer while an action is in flight lets Godot's main loop re-enter and crashes it.
    /// </summary>
    public static async Task DriveWithSelections(Task work, Func<PendingSelection, Task> answer)
    {
        // Wraps the non-generic case; the body is identical, so keep one implementation.
        await DriveWithSelections(WithResult(work), answer);
    }

    private static async Task<bool> WithResult(Task t) { await t; return true; }

    public static async Task<T> DriveWithSelections<T>(Task<T> work, Func<PendingSelection, Task> answer)
    {
        while (!work.IsCompleted)
        {
            CardSelect.ClearRaised();
            if (CardSelect.Pending == null) await Task.WhenAny(work, CardSelect.Raised);

            var pending = CardSelect.Pending;
            if (pending == null) { if (!work.IsCompleted) await CombatHarness.Suspend(); continue; }

            if (answer != null) await answer(pending);
            else pending.Finish();
            CardSelect.ClearRaised();
            await CombatHarness.Suspend();
        }
        return await work;
    }

    // ---------- events ----------

    /// <summary>
    /// Resolves an event by repeatedly choosing options until the event reports itself
    /// finished.
    ///
    /// Events are multi-page state machines, not single choices: picking an option replaces
    /// CurrentOptions with the next page, so this loops rather than choosing once. Locked
    /// options (OnChosen == null - the requirement is not met) are offered to the policy as
    /// context but never chosen.
    ///
    /// Options run as fire-and-forget tasks inside the synchronizer, which is why the loop
    /// awaits AwaitPendingOptionTasks; and they can park on a card-selection prompt
    /// (transform/remove/upgrade events), which is why that await is driven rather than
    /// straight.
    /// </summary>
    public static async Task<List<string>> ResolveEventAsync(
        Player player, EventRoom room,
        Func<Player, IReadOnlyList<EventOptionInfo>, Task<int>> chooser,
        Func<PendingSelection, Task> answerSelection)
    {
        var sync = RunManager.Instance.EventSynchronizer;
        var taken = new List<string>();

        for (var page = 0; page < 24; page++)
        {
            var ev = sync.GetLocalEvent();
            if (ev == null || ev.IsFinished) break;

            var opts = ev.CurrentOptions;
            if (opts == null || opts.Count == 0) break;

            var offered = new List<EventOptionInfo>();
            for (var i = 0; i < opts.Count; i++)
            {
                var o = opts[i];
                offered.Add(new EventOptionInfo
                {
                    Index = i,
                    TextKey = o.TextKey ?? "",
                    Locked = o.IsLocked,
                    IsProceed = o.IsProceed,
                    WasChosen = o.WasChosen,
                    RelicId = o.Relic?.Id.Entry ?? "",
                    WillKill = o.WillKillPlayer != null && Try(() => o.WillKillPlayer(player), false),
                    EventId = ev.Id.Entry,
                    Vals = OptionValues(ev, o),
                    EventVals = Obs.VarValues(ev.DynamicVars),
                    Tips = Try(() => (o.HoverTips ?? Enumerable.Empty<IHoverTip>())
                        .Select(t => t?.Id).Where(x => !string.IsNullOrEmpty(x)).ToList(),
                        new List<string>()),
                    Cards = Try(() => (o.HoverTips ?? Enumerable.Empty<IHoverTip>())
                        .OfType<CardHoverTip>().Select(t => t.Card?.Id.Entry)
                        .Where(x => !string.IsNullOrEmpty(x)).ToList(), new List<string>()),
                });
            }

            var choosable = offered.Where(o => !o.Locked).ToList();
            if (choosable.Count == 0) break;

            var pick = chooser != null ? await chooser(player, offered) : choosable[0].Index;
            if (pick < 0 || pick >= opts.Count || opts[pick].IsLocked) pick = choosable[0].Index;

            taken.Add(opts[pick].TextKey ?? "");
            if (Probe.HasFlag("probe-log-events"))
                Probe.Log($"        opt[{pick}] {opts[pick].TextKey} " +
                          $"(page {page}, {opts.Count} options: {string.Join("|", opts.Select(o => o.TextKey))})");
            if (Probe.HasFlag("probe-log-events")) Probe.Log("        -> ChooseLocalOption");
            // An option handler can throw synchronously (an unguarded node touch that is now
            // a plain NullReferenceException rather than a crash). One bad event must not end
            // the run: log it and let the loop move on.
            try { sync.ChooseLocalOption(pick); }
            catch (Exception e) { Probe.Log($"        option {opts[pick].TextKey} threw: {e.GetType().Name}: {e.Message}"); break; }
            if (Probe.HasFlag("probe-log-events")) Probe.Log("        -> chosen, awaiting option tasks");
            await DriveWithSelections(sync.AwaitPendingOptionTasks(), answerSelection);
            if (Probe.HasFlag("probe-log-events")) Probe.Log("        -> option tasks done");

            // An option can push a combat room on top of this one ("Start a Fight!"). The
            // caller owns combat, so hand control back and let it re-dispatch on the new
            // top-of-stack room; it will resume us afterwards.
            if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is CombatRoom) break;
        }
        return taken;
    }

    /// <summary>
    /// Pops a combat that was nested inside an event and resumes the event underneath.
    ///
    /// This is what the proceed button does after an event fight. It is only correct when the
    /// room stack is actually nested, hence the CurrentRoomCount check inside the engine.
    /// </summary>
    public static Task ResumeParentRoomAsync() =>
        RunManager.Instance.ProceedFromTerminalRewardsScreen();

    private static T Try<T>(Func<T> f, T fallback)
    {
        try { return f(); } catch { return fallback; }
    }

    // ---------- end-of-combat settling ----------

    /// <summary>
    /// Waits for the engine to actually FINISH a combat, not merely to decide the outcome.
    ///
    /// "Enemies cleared" is known long before combat is over. CombatManager.EndCombatInternal
    /// still has to revive players, walk AfterCombatEnd and AfterCombatVictory hooks, clear
    /// history, mark the room pre-finished, save, and unpause the action executor. Acting
    /// during that walk mutates collections it is iterating, which is what produced the
    /// original Defect crash and, once the run layer started doing real work right after a
    /// fight, an intermittent SIGSEGV roughly one run in twenty.
    ///
    /// IsInProgress is useless as a signal - EndCombatInternal clears it on its FIRST line.
    /// The last line raises CombatEnded, so that is the signal, subscribed BEFORE the wait so
    /// a combat that finishes while we are setting up is not missed.
    ///
    /// A fixed settle delay also made the crash go away, but a delay is a guess: it is either
    /// too short on a slow frame or wasted on every fast one. This waits for the actual event.
    /// </summary>
    public static async Task<bool> WaitForCombatEndAsync(int timeoutMs = 5000)
    {
        var mgr = CombatManager.Instance;
        if (mgr == null) return true;

        var tcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        void OnEnded(CombatRoom _) => tcs.TrySetResult(true);
        mgr.CombatEnded += OnEnded;
        try
        {
            // Already settled: the room is marked pre-finished near the end of the same
            // method, so an in-flight teardown has not reached that point yet.
            if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is CombatRoom { IsPreFinished: true })
                return true;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (!tcs.Task.IsCompleted && sw.ElapsedMilliseconds < timeoutMs)
            {
                if (RunManager.Instance.DebugOnlyGetState()?.CurrentRoom is CombatRoom { IsPreFinished: true })
                    return true;
                await CombatHarness.Suspend();
            }
            return tcs.Task.IsCompleted;
        }
        finally { mgr.CombatEnded -= OnEnded; }
    }

    // ---------- shops ----------

    /// <summary>
    /// Describes the local player's shop stock. Card removal is listed as an entry like any
    /// other, because to a policy it is just another thing to spend gold on.
    /// </summary>
    public static List<ShopOption> DescribeShop(MerchantInventory inv)
    {
        var list = new List<ShopOption>();
        if (inv == null) return list;
        var i = 0;

        foreach (var e in inv.CardEntries)
        {
            var card = e.CreationResult?.Card;
            list.Add(new ShopOption
            {
                Index = i++, Kind = "card",
                Id = card?.Id.Entry ?? "",
                ModelIdx = Vocab.Get(Vocab.Cards, card?.Id.Entry),
                Cost = Try(() => e.Cost, 0), Affordable = Try(() => e.EnoughGold, false),
                OnSale = e.IsOnSale, Rarity = card == null ? -1 : (int)card.Rarity,
                Upgrade = card?.CurrentUpgradeLevel ?? 0,
                Keywords = card == null ? 0 : Obs.Keywords(card),
                Vals = card == null ? Obs.NoValues : Obs.CardValues(card, null),
            });
        }
        foreach (var e in inv.RelicEntries)
            list.Add(new ShopOption
            {
                Index = i++, Kind = "relic",
                Id = e.Model?.Id.Entry ?? "",
                ModelIdx = Vocab.Get(Vocab.Relics, e.Model?.Id.Entry),
                Cost = Try(() => e.Cost, 0), Affordable = Try(() => e.EnoughGold, false),
                Vals = Obs.VarValues(e.Model?.DynamicVars),
            });
        foreach (var e in inv.PotionEntries)
            list.Add(new ShopOption
            {
                Index = i++, Kind = "potion",
                Id = e.Model?.Id.Entry ?? "",
                ModelIdx = Vocab.Get(Vocab.Potions, e.Model?.Id.Entry),
                Cost = Try(() => e.Cost, 0), Affordable = Try(() => e.EnoughGold, false),
                Vals = Obs.VarValues(e.Model?.DynamicVars),
            });
        if (inv.CardRemovalEntry != null && inv.CardRemovalEntry.IsStocked)
            list.Add(new ShopOption
            {
                Index = i, Kind = "remove_card", Id = "REMOVE_CARD",
                Cost = Try(() => inv.CardRemovalEntry.Cost, 0),
                Affordable = Try(() => inv.CardRemovalEntry.EnoughGold, false),
            });
        return list;
    }

    /// <summary>
    /// The event's numbers that this option's text refers to.
    ///
    /// An event keeps its numbers (gold, HP loss, heal, max HP...) as dynamic vars on the EVENT,
    /// and each option's description names the ones it uses, e.g. "{HpLoss}" and "{Gold}". So
    /// the vars an option's raw text references are the option's effects. Title and description
    /// are both scanned; a var named in neither is not attributed to the option.
    /// </summary>
    public static Dictionary<string, object> OptionValues(EventModel ev, EventOption o)
    {
        var text = Try(() => o.Title?.GetRawText(), "") + " " + Try(() => o.Description?.GetRawText(), "");
        var named = new HashSet<string>(System.Text.RegularExpressions.Regex
            .Matches(text ?? "", @"\{([A-Za-z0-9_]+)").Select(m => m.Groups[1].Value));
        var d = Obs.ValueSlots.ToDictionary(k => k, k => (object)0);
        if (named.Count == 0 || ev?.DynamicVars == null) return d;
        try
        {
            foreach (var v in ev.DynamicVars.Values)
            {
                if (!named.Contains(v.Name)) continue;
                var one = Obs.VarValues(new[] { v });
                foreach (var kv in one)
                    if ((int)kv.Value != 0) d[kv.Key] = kv.Value;
            }
        }
        catch { }
        return d;
    }

    /// <summary>Entry at a describe-index, so a policy choice maps back to a real purchase.</summary>
    private static MerchantEntry EntryAt(MerchantInventory inv, int index)
    {
        var all = inv.CardEntries.Cast<MerchantEntry>()
            .Concat(inv.RelicEntries)
            .Concat(inv.PotionEntries).ToList();
        if (index >= 0 && index < all.Count) return all[index];
        return index == all.Count ? inv.CardRemovalEntry : null;
    }

    /// <summary>
    /// Resolves a shop: repeatedly asks the policy what to buy until it declines.
    ///
    /// Purchases go through MerchantEntry.OnTryPurchaseWrapper, which is the same call the
    /// shop UI makes - it re-checks stock and gold itself and reports a typed failure, so an
    /// unaffordable or sold-out pick is refused by the engine rather than by us.
    ///
    /// Card removal parks on a card-selection prompt (which card to remove), hence the
    /// driven await.
    /// </summary>
    public static async Task<List<string>> ResolveShopAsync(
        Player player, MerchantRoom room,
        Func<Player, IReadOnlyList<ShopOption>, Task<int>> chooser,
        Func<PendingSelection, Task> answerSelection)
    {
        var bought = new List<string>();
        MerchantInventory inv;
        try { inv = room.GetLocalInventory(); }
        catch (Exception e) { Probe.Log($"  (non-fatal) shop inventory: {e.Message}"); return bought; }
        if (inv == null) return bought;

        for (var guard = 0; guard < 16; guard++)
        {
            var offered = DescribeShop(inv);
            if (offered.Count == 0) break;

            var pick = chooser != null ? await chooser(player, offered) : -1;
            if (pick < 0) break;                      // policy declined - leave the shop

            var entry = EntryAt(inv, pick);
            if (entry == null) break;
            var label = offered.FirstOrDefault(o => o.Index == pick)?.Id ?? "?";

            var ok = await DriveWithSelections(entry.OnTryPurchaseWrapper(inv), answerSelection);
            if (ok) bought.Add(label);
            else break;   // refused (gold or stock); asking again would spin
        }
        return bought;
    }
}
