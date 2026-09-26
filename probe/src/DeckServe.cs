using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace AlphaSlayer.Probe;

/// <summary>
/// Deck-evaluation server: "how good is THIS deck against THIS kind of fight?"
///
/// The run policy's hardest decision is which card to add, and nothing in the reward can
/// currently tell a good card from a bad one -- only how many cards there are. Answering it
/// properly needs the counterfactual: play the deck you would have with the card, and the deck
/// you would have without it, and compare. This phase makes that measurable.
///
/// It is the `serve` protocol with one addition: before each episode the policy sends the deck,
/// upgrades, relics and the room type to fight. Everything after that -- observations, legal
/// actions, the reply format -- is byte-identical to `serve`, so the same combat policy plays
/// these fights as plays real ones. That matters: a deck's strength is only meaningful with
/// respect to the player that pilots it.
///
/// Protocol:
///   -&gt; {"t":"need_episode","ep":N}
///   &lt;- {"deck":["STRIKE",...],"upgrades":[0,...],"relics":["BURNING_BLOOD"],"room":"elite"}
///   -&gt; {"t":"decision",...}  &lt;- {"a":I}   (repeats)
///   -&gt; {"t":"terminal","won":bool,"hp_end":N,"max_hp":N,"turns":N,"room":"elite"}
/// </summary>
public static class DeckServe
{
    public static async Task RunAsync()
    {
        var host = Probe.ArgValue("probe-host") ?? "127.0.0.1";
        var port = Probe.ArgInt("probe-port", 5555);
        var nEpisodes = Probe.ArgInt("probe-episodes", 500);
        var ascension = Probe.ArgInt("probe-ascension", 0);
        var seed = Probe.ArgValue("probe-seed") ?? "DECKEVAL";
        var turnCap = Probe.ArgInt("probe-turn-cap", 50);

        Vocab.Build();

        using var client = new TcpClient();
        client.NoDelay = true;
        await client.ConnectAsync(host, port);
        client.NoDelay = true;
        using var stream = client.GetStream();
        using var reader = new StreamReader(stream, Encoding.UTF8);
        using var writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };
        Probe.Log($"deckserve: connected to {host}:{port}");

        var harness = new CombatHarness();
        await harness.StartRunAsync(seed, ascension, Probe.ArgValue("probe-character"));
        CardSelect.Install();

        await writer.WriteLineAsync(Probe.ToJsonCompact(new Dictionary<string, object>
        {
            ["t"] = "hello", ["mode"] = "deckeval",
            ["character"] = harness.Character.Id.Entry,
            ["cards"] = Vocab.Cards.Count + 1, ["relics"] = Vocab.Relics.Count + 1,
            ["potions"] = Vocab.Potions.Count + 1,
            ["afflictions"] = Vocab.Afflictions.Count + 1, ["orbs"] = Vocab.Orbs.Count + 1, ["encounter_vocab"] = Vocab.Encounters.Count + 1,
        }));

        var rng = new Random(Probe.ArgInt("probe-policy-seed", 7) ^ Probe.StableHash(seed));
        long steps = 0;
        var sw = Stopwatch.StartNew();

        for (var ep = 0; ep < nEpisodes; ep++)
        {
            await writer.WriteLineAsync(Probe.ToJsonCompact(new Dictionary<string, object>
            {
                ["t"] = "need_episode", ["ep"] = ep,
            }));
            var spec = await reader.ReadLineAsync();
            if (spec == null) break;
            if (spec.Contains("\"stop\"")) break;

            var deck = ParseStrArray(spec, "deck");
            var ups = ParseIntArray(spec, "upgrades");
            var relics = ParseStrArray(spec, "relics");
            var room = ParseStr(spec, "room") ?? "regular";

            try
            {
                await harness.SetDeckAsync(deck, ups, relics);
                if (CombatHarness.Unknown.Count > 0)
                    Probe.Log($"  WARNING: {CombatHarness.Unknown.Count} unknown card id(s), " +
                              $"deck is incomplete: {string.Join(",", CombatHarness.Unknown.Take(5))}");
            }
            catch (Exception e) { Probe.Log($"  set deck: {e.Message}"); }

            var pool = room switch
            {
                "elite" => harness.EliteEncounters,
                "boss" => harness.BossEncounters,
                _ => harness.RegularEncounters,
            };
            // NO silent fallback. An empty pool means the harness was started without the
            // "mix" encounter mode, and quietly substituting regular enemies produced boss
            // and elite win rates that were really regular-fight win rates.
            if (pool == null || pool.Count == 0)
            {
                await writer.WriteLineAsync(Probe.ToJsonCompact(new Dictionary<string, object>
                {
                    ["t"] = "error",
                    ["msg"] = $"no encounters for room '{room}': the harness has an empty " +
                              "pool. Launch deckserve with --probe-encounters=mix.",
                }));
                continue;
            }

            // Full health each time: we are measuring the DECK, and carrying damage between
            // evaluations would confound it with whatever the previous fight happened to be.
            try { harness.Player.Creature.HealInternal(harness.Player.Creature.MaxHp); } catch { }
            // ...unless the caller asked for a specific fraction. Measuring a deck wants
            // full hp; TRAINING act-2 fights does not, because the run policy arrives in
            // act 2 at ~35% hp and how much risk a play is worth depends on what is left.
            var hpFrac = ParseDouble(spec, "hp_frac");
            if (hpFrac > 0 && hpFrac < 1)
            {
                try
                {
                    var want = Math.Max(1, (int)Math.Round(harness.Player.Creature.MaxHp * hpFrac));
                    harness.Player.Creature.SetCurrentHpInternal(want);
                }
                catch (Exception e) { Probe.Log($"  (non-fatal) set hp_frac: {e.Message}"); }
            }

            var enc = await harness.BeginCombatAsync(pool[rng.Next(pool.Count)]);
            RunManager.Instance.ActionExecutor.Unpause();

            var startHp = harness.Player.Creature.CurrentHp;
            int turns = 0;
            string outcome = null;

            while (turns < turnCap && outcome == null)
            {
                outcome = await Episodes.WaitForDecisionPublic(harness.Player);
                if (outcome != null) break;

                var combat = CombatManager.Instance.DebugOnlyGetState();
                var legal = Obs.LegalActions(harness.Player, combat);
                if (legal.Count == 0) { outcome = "no_legal_actions"; break; }

                await writer.WriteLineAsync(Probe.ToJsonCompact(new Dictionary<string, object>
                {
                    ["t"] = "decision", ["ep"] = ep, ["encounter"] = enc.Id.Entry,
                    ["obs"] = Obs.Extract(harness.Player, combat),
                    ["legal"] = legal.Select(a => (object)a.ToDict()).ToArray(),
                }));
                var reply = await reader.ReadLineAsync();
                if (reply == null) { outcome = "policy_disconnected"; break; }
                var choice = legal[Math.Clamp(ParseAction(reply), 0, legal.Count - 1)];
                steps++;

                if (choice.Kind == "select_card" || choice.Kind == "select_done")
                {
                    var pend = CardSelect.Pending;
                    if (choice.Kind == "select_done") pend?.Finish(); else pend?.Pick(choice.Card);
                    CardSelect.ClearRaised();
                    await CombatHarness.Suspend();
                }
                else if (choice.Kind == "use_potion")
                {
                    choice.Potion.EnqueueManualUse(choice.Target);
                    await Episodes.WaitForQueueOrSelectionPublic();
                }
                else if (choice.Kind == "end_turn")
                {
                    var expect = harness.Player.PlayerCombatState.TurnNumber + 1;
                    CombatManager.Instance.SetReadyToEndTurn(harness.Player, false, null);
                    turns++;
                    outcome = await Episodes.WaitForTurnPublic(harness.Player, expect);
                }
                else
                {
                    RunManager.Instance.ActionQueueSet.EnqueueWithoutSynchronizing(
                        new PlayCardAction(choice.Card, choice.Target));
                    await Episodes.WaitForQueueOrSelectionPublic();
                }
            }

            outcome ??= "turn_cap";
            var won = outcome == "enemies_cleared";
            await writer.WriteLineAsync(Probe.ToJsonCompact(new Dictionary<string, object>
            {
                ["t"] = "terminal", ["ep"] = ep, ["won"] = won, ["outcome"] = outcome,
                ["room"] = room, ["encounter"] = enc.Id.Entry, ["turns"] = turns,
                ["hp_start"] = startHp, ["hp_end"] = harness.Player.Creature.CurrentHp,
                ["max_hp"] = harness.Player.Creature.MaxHp,
            }));

            CardSelect.Reset();
            await harness.EndCombatAsync(healToFull: true);
            if (outcome == "policy_disconnected") break;
        }

        sw.Stop();
        await writer.WriteLineAsync(Probe.ToJsonCompact(new Dictionary<string, object>
        {
            ["t"] = "done", ["episodes"] = nEpisodes, ["steps"] = steps,
            ["wall_s"] = sw.Elapsed.TotalSeconds,
        }));
        Probe.Log($"DECKSERVE: {nEpisodes} eval fights, {steps} steps in {sw.Elapsed.TotalSeconds:F1}s");
    }

    // ---------- minimal JSON field readers (one flat object per line) ----------

    private static double ParseDouble(string line, string key)
    {
        var i = line.IndexOf($"\"{key}\"", StringComparison.Ordinal);
        if (i < 0) return 0;
        var c = line.IndexOf(':', i);
        if (c < 0) return 0;
        var end = c + 1;
        while (end < line.Length && (char.IsDigit(line[end]) || line[end] == '.'
                                     || line[end] == '-' || line[end] == '+'
                                     || line[end] == 'e' || line[end] == 'E'
                                     || line[end] == ' ')) end++;
        return double.TryParse(line.Substring(c + 1, end - c - 1).Trim(),
                               NumberStyles.Float, CultureInfo.InvariantCulture,
                               out var v) ? v : 0;
    }

    private static string ParseStr(string line, string key)
    {
        var i = line.IndexOf($"\"{key}\"", StringComparison.Ordinal);
        if (i < 0) return null;
        var q1 = line.IndexOf('"', line.IndexOf(':', i) + 1);
        if (q1 < 0) return null;
        var q2 = line.IndexOf('"', q1 + 1);
        return q2 < 0 ? null : line.Substring(q1 + 1, q2 - q1 - 1);
    }

    private static List<string> ParseStrArray(string line, string key)
    {
        var res = new List<string>();
        var i = line.IndexOf($"\"{key}\"", StringComparison.Ordinal);
        if (i < 0) return res;
        var lb = line.IndexOf('[', i);
        var rb = line.IndexOf(']', lb + 1);
        if (lb < 0 || rb < 0) return res;
        foreach (var part in line.Substring(lb + 1, rb - lb - 1).Split(','))
        {
            var t = part.Trim().Trim('"');
            if (t.Length > 0) res.Add(t);
        }
        return res;
    }

    private static List<int> ParseIntArray(string line, string key)
    {
        var res = new List<int>();
        var i = line.IndexOf($"\"{key}\"", StringComparison.Ordinal);
        if (i < 0) return res;
        var lb = line.IndexOf('[', i);
        var rb = line.IndexOf(']', lb + 1);
        if (lb < 0 || rb < 0) return res;
        foreach (var part in line.Substring(lb + 1, rb - lb - 1).Split(','))
            if (int.TryParse(part.Trim(), out var v)) res.Add(v);
        return res;
    }

    private static int ParseAction(string line)
    {
        var i = line.IndexOf("\"a\"", StringComparison.Ordinal);
        if (i < 0) return 0;
        var c = line.IndexOf(':', i);
        if (c < 0) return 0;
        var end = c + 1;
        while (end < line.Length && (char.IsDigit(line[end]) || line[end] == ' ' || line[end] == '-')) end++;
        return int.TryParse(line.Substring(c + 1, end - c - 1).Trim(), out var v) ? v : 0;
    }
}
