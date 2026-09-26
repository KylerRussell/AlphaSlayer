using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;

namespace AlphaSlayer.Probe;

/// <summary>
/// One in-flight "choose card(s)" prompt, parked until the agent answers.
///
/// Multi-select (min..max) is exposed as a SEQUENCE of single picks rather than one
/// combinatorial choice: the agent picks one card at a time and may finish once it has at
/// least <see cref="Min"/>. That keeps the action space uniform (always "pick one thing")
/// and avoids an action set that grows combinatorially with hand size.
/// </summary>
public sealed class PendingSelection
{
    public List<CardModel> Options = new();
    public List<CardModel> Chosen = new();
    public int Min, Max;
    public TaskCompletionSource<IEnumerable<CardModel>> Tcs;

    public bool CanFinish => Chosen.Count >= Min;
    public bool MustFinish => Chosen.Count >= Max || Remaining.Count == 0;
    public List<CardModel> Remaining => Options.Where(o => !Chosen.Contains(o)).ToList();

    public void Pick(CardModel c)
    {
        Chosen.Add(c);
        if (MustFinish) Finish();
    }

    public void Finish()
    {
        if (CardSelect.Pending == this) CardSelect.Pending = null;
        Tcs.TrySetResult(Chosen.ToList());
    }
}

/// <summary>
/// Agent-driven <see cref="ICardSelector"/>.
///
/// Without a selector registered, the selection path dereferences UI that does not exist
/// headless and segfaults - which is why Silent and Defect could not run at all. But a
/// selection is not just a crash to paper over: "choose a card from your hand", or Quasar's
/// choose-1-of-3, is a genuine DECISION the agent must make, so it is surfaced as a decision
/// point rather than answered with a fixed rule.
///
/// The interface is async, so the combat coroutine simply parks on the returned task while
/// the env loop surfaces the choice and the policy answers.
/// </summary>
public sealed class CardSelect : ICardSelector
{
    public static PendingSelection Pending;

    // Completes when a prompt is raised, so callers can await it alongside the action queue
    // instead of polling. Polling here was a real bug: spinning on Task.Delay while an action
    // was mid-flight let the main loop re-enter and crashed the engine.
    private static TaskCompletionSource<bool> _raised = NewSignal();
    public static Task Raised => _raised.Task;

    private static TaskCompletionSource<bool> NewSignal() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    /// <summary>Re-arms the signal once the current prompt has been dealt with.</summary>
    public static void ClearRaised()
    {
        if (_raised.Task.IsCompleted) _raised = NewSignal();
    }
    private static IDisposable _scope1, _scope2;
    private static Random _fallbackRng = new(1234);

    /// <summary>Fallback used for prompts raised outside an agent decision loop.</summary>
    public static bool AutoAnswer;

    public static void Install()
    {
        // Idempotent. PushSelector never gets popped, so calling this once per run left a
        // stack of selectors that grew with every run in the process.
        if (_scope1 != null) return;
        var sel = new CardSelect();
        // Single (non-local) stack, matching how AutoSlay registers its own selector.
        // Registering on BOTH stacks breaks CardSelectCmd.FromHand: a non-null Selector makes
        // it skip reserving a choice id and signalling the player-choice context, and the
        // local branch then runs against a half-initialised choice.
        _scope1 = MegaCrit.Sts2.Core.Commands.CardSelectCmd.PushSelector(sel, localOnly: false);
        Probe.Log("card selector installed (agent-driven)");
    }

    public static void Reset()
    {
        Pending?.Tcs.TrySetResult(Array.Empty<CardModel>());
        Pending = null;
    }

    public Task<IEnumerable<CardModel>> GetSelectedCards(
        IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        var list = options?.ToList() ?? new List<CardModel>();
        if (list.Count == 0)
            return Task.FromResult<IEnumerable<CardModel>>(Array.Empty<CardModel>());

        var max = Math.Min(Math.Max(maxSelect, minSelect), list.Count);
        var min = Math.Min(minSelect, max);

        if (AutoAnswer)
        {
            var shuffled = list.OrderBy(_ => _fallbackRng.Next()).Take(max);
            return Task.FromResult<IEnumerable<CardModel>>(shuffled.ToList());
        }

        var p = new PendingSelection
        {
            Options = list, Min = min, Max = max,
            Tcs = new TaskCompletionSource<IEnumerable<CardModel>>(
                TaskCreationOptions.RunContinuationsAsynchronously),
        };
        Pending = p;
        _raised.TrySetResult(true);
        return p.Tcs.Task;
    }

    /// <summary>
    /// Card rewards. This is where CardReward.OnSelect lands under TestMode (the selection
    /// screen is always null there).
    ///
    /// The interface is SYNCHRONOUS, so the agent cannot be consulted from inside it - the
    /// answer is pre-armed by RunRewards from inside RewardsSet.testSelector, which is async
    /// and already holds the card list. With nothing armed this falls back to the first card,
    /// which is what the combat-only env wants (it never reaches a reward legitimately).
    /// </summary>
    public CardRewardSelection GetSelectedCardReward(
        IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives)
        => RunRewards.AnswerCardReward(options, alternatives);
}
