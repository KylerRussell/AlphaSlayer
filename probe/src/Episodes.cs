using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;

namespace AlphaSlayer.Probe;

/// <summary>
/// Episode generation: runs combats under a random-over-legal-actions policy and writes one
/// JSONL line per decision point. This is the trainable-trajectory format the Python side
/// consumes, and the random policy is the behaviour baseline every later agent is measured
/// against.
///
/// Reward shaping is deliberately minimal here: the terminal signal is win/loss, and each
/// step records hp_delta so dense auxiliary targets (damage taken, turns survived) can be
/// derived offline without re-running the sim.
/// </summary>
public static class Episodes
{
    public static async Task RunAsync()
    {
        var n = Probe.ArgInt("probe-episodes", 50);
        var turnCap = Probe.ArgInt("probe-turn-cap", 50);
        var ascension = Probe.ArgInt("probe-ascension", 0);
        var seed = Probe.ArgValue("probe-seed") ?? "ALPHASLAYER1";
        var rngSeed = Probe.ArgInt("probe-policy-seed", 12345);
        // "random" is the baseline; "greedy" (play anything playable, else end turn) is the
        // behaviour-cloning demonstrator - it wins ~85% vs random's ~27%, so it is the only
        // one of the two worth imitating.
        var policy = (Probe.ArgValue("probe-policy") ?? "random").ToLowerInvariant();
        var emitObs = !Probe.HasFlag("probe-no-obs");

        Vocab.Build();
        Vocab.Dump(Probe.OutDir, Probe.Report.TryGetValue("game_version", out var gv) ? gv.ToString() : "unknown");

        var harness = new CombatHarness();
        await harness.StartRunAsync(seed, ascension, Probe.ArgValue("probe-character"));
        // Without this, any "choose a card" effect dereferences absent UI and segfaults -
        // which is why Silent and Defect could not run at all.
        CardSelect.Install();
        // Separate RNG for encounter sampling so the mix is reproducible per seed and does
        // not perturb the game's own streams.
        var encRng = new Random(Probe.ArgInt("probe-policy-seed", 12345) ^ Probe.StableHash(seed));
        RngAudit.MapStreams(harness.RunState);
        Probe.Log($"episodes: {n} combats, policy={policy}, character={harness.Character.Id.Entry}, seed={seed}, A{ascension}");

        // Binary is the default; JSONL stays available for eyeballing real records.
        var format = (Probe.ArgValue("probe-format") ?? "bin").ToLowerInvariant();
        var path = Path.Combine(Probe.OutDir, format == "jsonl" ? "episodes.jsonl" : "episodes.bin.gz");
        StreamWriter w = null;
        EpisodeWriter bw = null;
        if (format == "jsonl") w = new StreamWriter(path, append: false);
        else bw = new EpisodeWriter(path, VocabHash());

        var rng = new Random(rngSeed);
        long steps = 0, plays = 0, selections = 0;
        var outcomes = new Dictionary<string, int>();
        var sw = Stopwatch.StartNew();

        for (var ep = 0; ep < n; ep++)
        {
            // Offset lets a failing encounter be isolated: crashing at episode N could mean
            // "accumulates over N combats" or "encounter N is bad", and those need different fixes.
                        LoopAudit.ResetCombat();
            LoopAudit.Armed = true;
            await harness.RerollDeckAsync();
            var enc = await harness.BeginCombatAsync(harness.NextEncounter(ep, encRng));
            if (Probe.HasFlag("probe-log-encounters")) Probe.Log($"  ep {ep}: {enc.Id.Entry}");
            var stepInEp = 0;
            var turns = 0;
            var startHp = harness.Player.Creature.CurrentHp;
            string outcome = null;

            while (turns < turnCap && outcome == null)
            {
                outcome = await WaitForDecision(harness.Player);
                if (outcome != null) break;

                var combat = harness.State;
                var legal = Obs.LegalActions(harness.Player, combat);
                if (legal.Count == 0) { outcome = "no_legal_actions"; break; }

                var choice = Choose(policy, legal, rng);
                var hpBefore = harness.Player.Creature.CurrentHp;

                var tNow = stepInEp++;
                var actionIdx = legal.IndexOf(choice);
                Dictionary<string, object> line = null;
                if (w != null)
                {
                    line = new Dictionary<string, object>
                    {
                        ["ep"] = ep,
                        ["t"] = tNow,
                        ["encounter"] = enc.Id.Entry,
                        ["legal"] = legal.Select(a => (object)a.ToDict()).ToArray(),
                        ["action"] = choice.ToDict(),
                        ["action_idx"] = actionIdx,
                    };
                    if (emitObs) line["obs"] = Obs.Extract(harness.Player, combat);
                }

                // Capture the decision point BEFORE applying the action: the observation must
                // be the state the agent saw when it chose.
                bw?.BeginStep(ep, tNow, harness.Player, combat, legal, actionIdx);
                RngAudit.Arm(true);

                if (choice.Kind == "select_card" || choice.Kind == "select_done")
                {
                    var pend = CardSelect.Pending;
                    if (choice.Kind == "select_done") pend?.Finish();
                    else pend?.Pick(choice.Card);
                    selections++;
                    CardSelect.ClearRaised();
                    Emit(w, bw, line, ep, tNow, harness, combat, legal, actionIdx,
                         harness.Player.Creature.CurrentHp - hpBefore);
                    steps++;
                    // Let the parked card effect resume before the next decision.
                    await CombatHarness.Suspend();
                }
                else if (choice.Kind == "end_turn")
                {
                    var expect = harness.Player.PlayerCombatState.TurnNumber + 1;
                    CombatManager.Instance.SetReadyToEndTurn(harness.Player, false, null);
                    turns++;
                    Emit(w, bw, line, ep, tNow, harness, combat, legal, actionIdx,
                         harness.Player.Creature.CurrentHp - hpBefore);
                    steps++;
                    // The turn transition is engine-driven and two-phase; wait for the number.
                    outcome = await WaitForTurn(harness.Player, expect);
                }
                else
                {
                    RunManager().ActionQueueSet.EnqueueWithoutSynchronizing(
                        new PlayCardAction(choice.Card, choice.Target));
                    // Do not block on BecameEmpty(): the card effect may park on a card
                    // selection, which only THIS loop can answer, so waiting would deadlock.
                    await WaitForQueueOrSelection();
                    plays++;
                    Emit(w, bw, line, ep, tNow, harness, combat, legal, actionIdx,
                         harness.Player.Creature.CurrentHp - hpBefore);
                    steps++;
                }
            }

            RngAudit.Arm(false);
            outcome ??= "turn_cap";
            var won = outcome == "enemies_cleared";
            var hpEnd = harness.Player.Creature.CurrentHp;
            if (w != null)
                w.WriteLine(Probe.ToJsonCompact(new Dictionary<string, object>
                {
                    ["ep"] = ep, ["terminal"] = true, ["outcome"] = outcome,
                    ["won"] = won, ["turns"] = turns, ["steps"] = stepInEp,
                    ["encounter"] = enc.Id.Entry,
                    ["hp_start"] = startHp, ["hp_end"] = hpEnd,
                    ["reward"] = won ? 1.0 : 0.0,
                }));
            else
                bw.WriteTerminal(ep, outcome, won, turns, stepInEp, startHp, hpEnd, won ? 1f : 0f);
            outcomes[outcome] = outcomes.GetValueOrDefault(outcome) + 1;
            CardSelect.Reset();
            await harness.EndCombatAsync(healToFull: true);
        }

        w?.Dispose();
        bw?.Dispose();
        sw.Stop();
        var secs = sw.Elapsed.TotalSeconds;
        var r = new Dictionary<string, object>
        {
            ["episodes"] = n,
            ["steps"] = steps,
            ["card_plays"] = plays,
            ["card_selections"] = selections,
            ["wall_s"] = secs,
            ["steps_per_s"] = secs > 0 ? steps / secs : 0,
            ["episodes_per_s"] = secs > 0 ? n / secs : 0,
            ["outcomes"] = outcomes.ToDictionary(k => k.Key, v => (object)v.Value),
            ["jsonl"] = path,
            ["jsonl_bytes"] = new FileInfo(path).Length,
            ["obs_emitted"] = emitObs,
            ["format"] = format,
            ["policy"] = policy,
            ["bytes_per_step"] = steps > 0 ? (double)new FileInfo(path).Length / steps : 0,
        };
        Probe.Report["episodes"] = r;
        Probe.Log($"EPISODES: {n} eps, {steps} steps ({plays} plays) in {secs:F2}s " +
                  $"= {r["steps_per_s"]:F0} steps/s, {r["episodes_per_s"]:F1} eps/s");
        Probe.Log($"  outcomes: {string.Join(", ", outcomes.Select(kv => $"{kv.Key}={kv.Value}"))}");
        Probe.Log($"  wrote {path} ({new FileInfo(path).Length / 1024}KB)");
    }

    /// <summary>
    /// random: uniform over legal actions. greedy: play the first playable card, only ending
    /// the turn when nothing is playable - the same rule the throughput driver uses.
    /// </summary>
    private static LegalAction Choose(string policy, List<LegalAction> legal, Random rng)
    {
        if (policy == "greedy")
        {
            var play = legal.FirstOrDefault(a => a.Kind != "end_turn");
            if (play != null) return play;
            return legal.First(a => a.Kind == "end_turn");
        }
        return legal[rng.Next(legal.Count)];
    }

    /// <summary>
    /// Cheap stability check on the vocabulary, stamped into the binary header so the reader
    /// can refuse a dataset built against a different game build.
    /// </summary>
    private static int VocabHash()
    {
        unchecked
        {
            var h = 17;
            foreach (var kv in Vocab.Cards.OrderBy(k => k.Value))
                h = h * 31 + Probe.StableHash(kv.Key);
            h = h * 31 + Vocab.Relics.Count;
            h = h * 31 + Vocab.Powers.Count;
            h = h * 31 + Vocab.Monsters.Count;
            return h;
        }
    }

    private static void Emit(StreamWriter w, EpisodeWriter bw, Dictionary<string, object> line,
                             int ep, int t, CombatHarness h, CombatState combat,
                             List<LegalAction> legal, int actionIdx, int hpDelta)
    {
        if (w != null)
        {
            line["hp_delta"] = hpDelta;
            w.WriteLine(Probe.ToJsonCompact(line));
        }
        else bw.EndStep(hpDelta);
    }

    private static MegaCrit.Sts2.Core.Runs.RunManager RunManager() =>
        MegaCrit.Sts2.Core.Runs.RunManager.Instance;

    /// <summary>Waits until the agent may act, or returns the terminal outcome.</summary>
    public static Task<string> WaitForDecisionPublic(CombatHarness h) => WaitForDecision(h.Player);
    public static Task<string> WaitForTurnPublic(CombatHarness h, int t) => WaitForTurn(h.Player, t);
    public static Task WaitForQueueOrSelectionPublic() => WaitForQueueOrSelection();

    // Player-keyed overloads. The run harness drives combats it did not create through
    // RunManager.EnterMapCoord, so it has no CombatHarness to hand these; all they ever
    // needed from one was the player and the live CombatState, which is a singleton anyway.
    public static Task<string> WaitForDecisionPublic(Player p) => WaitForDecision(p);
    public static Task<string> WaitForTurnPublic(Player p, int t) => WaitForTurn(p, t);

    private static CombatState LiveState() => CombatManager.Instance.DebugOnlyGetState();

    private static async Task<string> WaitForDecision(Player player)
    {
        var budget = Probe.ArgInt("probe-turn-timeout-ms", 5000);
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < budget)
        {
            // A parked selection is a decision point even mid-action, so check it FIRST -
            // the combat coroutine is suspended inside a card effect and will not progress
            // until it is answered.
            if (CardSelect.Pending != null) return null;

            var st = LiveState();
            if (st == null) return "combat_ended";
            if (player.Creature.IsDead) return "player_dead";
            if (!st.HittableEnemies.Any()) return "enemies_cleared";
            if (CombatHarness.Finished()) return player.Creature.IsDead ? "player_dead" : "enemies_cleared";

            var pcs = player.PlayerCombatState;
            if (pcs != null && pcs.Phase == PlayerTurnPhase.Play && st.CurrentSide == CombatSide.Player)
                return null;
            await CombatHarness.Suspend();
        }
        return "decision_timeout";
    }

    /// <summary>
    /// Waits for the action queue to drain, returning early if a card effect parks on a card
    /// selection - that prompt can only be answered by this loop, so waiting on the queue
    /// alone would deadlock.
    ///
    /// Awaits both rather than polling. An earlier polling version crashed the engine:
    /// spinning on a timer while an action was mid-flight let Godot's main loop re-enter.
    /// </summary>
    private static async Task WaitForQueueOrSelection()
    {
        if (CardSelect.Pending != null) return;
        CardSelect.ClearRaised();
        await Task.WhenAny(RunManager().ActionQueueSet.BecameEmpty(), CardSelect.Raised);

        // Optional quiesce (--probe-quiesce). BecameEmpty() can complete while a continuation
        // from the just-resolved card is still pending, so an observation can capture
        // mid-settle state; requiring the queue to stay empty across a suspension roughly
        // halves that. OFF by default: it costs ~45% throughput (455 -> 251 steps/s) and the
        // race it narrows perturbs no action, reward or outcome - only about one observation
        // field in 90,000, and never the trajectory.
        if (!Probe.HasFlag("probe-quiesce")) return;
        for (var i = 0; i < 8; i++)
        {
            if (CardSelect.Pending != null) return;
            if (!RunManager().ActionQueueSet.IsEmpty) { await CombatHarness.Suspend(); continue; }
            await CombatHarness.Suspend();
            if (RunManager().ActionQueueSet.IsEmpty) return;
        }
    }

    private static async Task<string> WaitForTurn(Player player, int minTurn)
    {
        var budget = Probe.ArgInt("probe-turn-timeout-ms", 5000);
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < budget)
        {
            if (CardSelect.Pending != null) return null;
            var st = LiveState();
            if (st == null) return "combat_ended";
            if (player.Creature.IsDead) return "player_dead";
            if (!st.HittableEnemies.Any()) return "enemies_cleared";
            var pcs = player.PlayerCombatState;
            if (pcs != null && pcs.TurnNumber >= minTurn) return null;
            await CombatHarness.Suspend();
        }
        return "turn_timeout";
    }

    /// <summary>Exposes the built-in policies so the run driver can reuse them.</summary>
    public static LegalAction ChoosePublic(string policy, List<LegalAction> legal, Random rng) =>
        Choose(policy, legal, rng);
}
