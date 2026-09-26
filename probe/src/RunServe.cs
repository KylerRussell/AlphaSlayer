using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace AlphaSlayer.Probe;

/// <summary>
/// Whole-run env server: every decision in a run is handed to an external policy.
///
/// Serve.cs does this for combat alone. This does it for the run: map travel, card rewards,
/// campfires, events, shops and the potion gate, with combat decisions still flowing through
/// exactly the same message shape Serve.cs uses - so the trained combat policy is fed
/// byte-identical observations and its action space is unchanged.
///
/// Protocol (newline-delimited JSON), extending Serve.cs with a "kind" discriminator:
///   -&gt; {"t":"decision","kind":"combat|travel|card_reward|rest|event|shop|potion_gate|card_select",
///        "run":N,"step":N,"obs":{...},"legal":[...]}
///   &lt;- {"a":INDEX}
///   -&gt; {"t":"fight_end","run":N,"won":bool,"hp_start":N,"hp_end":N,"turns":N,"room":"Elite"}
///   -&gt; {"t":"terminal","run":N,"outcome":"...","won":bool,"floors":N,"act":N,...}
///   -&gt; {"t":"done","runs":N,"steps":N}
///
/// fight_end exists so the COMBAT policy keeps a per-fight outcome to learn from. Training it
/// on run outcome alone would change its objective from "win this fight" to "win this run",
/// which is a far sparser signal and the most likely way to damage a policy that already
/// plays fights well.
/// </summary>
public static class RunServe
{
    private static StreamWriter _w;
    private static StreamReader _r;
    private static int _run, _step;
    private static long _steps;
    private static readonly Dictionary<string, int> _byKind = new();

    public static async Task RunAsync()
    {
        var host = Probe.ArgValue("probe-host") ?? "127.0.0.1";
        var port = Probe.ArgInt("probe-port", 5555);
        var nRuns = Probe.ArgInt("probe-runs", 50);
        var ascension = Probe.ArgInt("probe-ascension", 0);
        var seed = Probe.ArgValue("probe-seed") ?? "ALPHASLAYER1";
        var turnCap = Probe.ArgInt("probe-turn-cap", 50);
        var floorCap = Probe.ArgInt("probe-floor-cap", 200);
        var runOffset = Probe.ArgInt("probe-run-offset", 0);

        Vocab.Build();

        using var client = new TcpClient();
        client.NoDelay = true;
        await client.ConnectAsync(host, port);
        client.NoDelay = true;
        using var stream = client.GetStream();
        using var reader = new StreamReader(stream, Encoding.UTF8);
        using var writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };
        _w = writer; _r = reader;
        Probe.Log($"runserve: connected to {host}:{port}");

        await Send(new Dictionary<string, object>
        {
            ["t"] = "hello", ["mode"] = "run",
            ["character"] = Probe.ArgValue("probe-character") ?? "",
            ["cards"] = Vocab.Cards.Count + 1, ["relics"] = Vocab.Relics.Count + 1,
            ["powers"] = Vocab.Powers.Count + 1, ["monsters"] = Vocab.Monsters.Count + 1,
            ["enchantments"] = Vocab.Enchantments.Count + 1, ["potions"] = Vocab.Potions.Count + 1,
            ["afflictions"] = Vocab.Afflictions.Count + 1, ["orbs"] = Vocab.Orbs.Count + 1, ["encounter_vocab"] = Vocab.Encounters.Count + 1,
        });

        var sw = Stopwatch.StartNew();
        var outcomes = new Dictionary<string, int>();
        var disconnected = false;

        for (var i = 0; i < nRuns && !disconnected; i++)
        {
            _run = i + runOffset;
            _step = 0;

            var h = new RunHarness();
            await h.StartRunAsync($"{seed}{_run}", ascension, Probe.ArgValue("probe-character"));
            h.ActCap = Probe.ArgInt("probe-act-cap", int.MaxValue);
            CardSelect.Install();
            RunRewards.Install();
            CrystalSphere.Install();
            InstallPolicyHooks(h);

            var res = await PlayRunAsync(h, turnCap, floorCap);
            outcomes[res.outcome] = outcomes.GetValueOrDefault(res.outcome) + 1;

            await Send(new Dictionary<string, object>
            {
                ["t"] = "terminal", ["run"] = _run, ["outcome"] = res.outcome,
                ["won"] = res.outcome == "run_complete",
                ["floors"] = res.floors, ["act"] = h.ActIndex, ["acts"] = h.EffectiveActCount,
                ["hp"] = h.Player.Creature.CurrentHp, ["max_hp"] = h.Player.Creature.MaxHp,
                ["gold"] = h.Player.Gold, ["deck"] = h.Player.Deck.Cards.Count,
                ["relics"] = h.Player.Relics.Count,
                ["fights"] = res.fights, ["fight_wins"] = res.fightWins,
            });
            CardSelect.Reset();
            Potions.ClearGate();
            if (res.outcome == "policy_disconnected") disconnected = true;
        }

        sw.Stop();
        await Send(new Dictionary<string, object>
        {
            ["t"] = "done", ["runs"] = nRuns, ["steps"] = _steps,
            ["wall_s"] = sw.Elapsed.TotalSeconds,
        });

        Probe.Report["runserve"] = new Dictionary<string, object>
        {
            ["runs"] = nRuns, ["steps"] = _steps, ["wall_s"] = sw.Elapsed.TotalSeconds,
            ["outcomes"] = outcomes.ToDictionary(k => k.Key, v => (object)v.Value),
            ["decisions_by_kind"] = _byKind.ToDictionary(k => k.Key, v => (object)v.Value),
        };
        Probe.Log($"RUNSERVE: {nRuns} run(s), {_steps} decisions in {sw.Elapsed.TotalSeconds:F1}s");
        Probe.Log($"  outcomes: {string.Join(", ", outcomes.Select(kv => $"{kv.Key}={kv.Value}"))}");
        Probe.Log($"  by kind: {string.Join(", ", _byKind.Select(kv => $"{kv.Key}={kv.Value}"))}");
    }

    // ---------- the run loop ----------

    private readonly struct RunResult
    {
        public readonly string outcome; public readonly int floors, fights, fightWins;
        public RunResult(string o, int f, int n, int w) { outcome = o; floors = f; fights = n; fightWins = w; }
    }

    private static async Task<RunResult> PlayRunAsync(RunHarness h, int turnCap, int floorCap)
    {
        string outcome = null;
        int floors = 0, fights = 0, fightWins = 0;

        while (outcome == null && floors < floorCap)
        {
            if (h.PlayerDead) { outcome = "player_dead"; break; }

            var legal = h.LegalRunActions();
            if (legal.Count == 0)
            {
                if (h.AtRunEnd()) { outcome = "run_complete"; break; }
                if (h.AtActEnd())
                {
                    try { await h.NextActAsync(); }
                    catch (Exception e) { Probe.Log($"  act transition: {e.Message}"); outcome = "act_transition_failed"; }
                    continue;
                }
                outcome = "no_legal_moves";
                break;
            }

            // --- map / path decision ---
            var pick = await Decide("travel", h.MapObs(), legal.Select(a => (object)a.ToDict()).ToArray(), legal.Count);
            if (pick < 0) { outcome = "policy_disconnected"; break; }

            try { await h.TravelAsync(legal[Math.Clamp(pick, 0, legal.Count - 1)].Coord); }
            catch (Exception e) { Probe.Log($"  travel: {e.Message}"); outcome = "travel_failed"; break; }
            floors++;

            // --- resolve whatever room we landed in (an event can push a combat on top) ---
            for (var depth = 0; depth < 8; depth++)
            {
                var before = h.CurrentRoom;
                if (before == null) break;
                var r = await HandleRoomAsync(h, before, turnCap);
                if (r == "policy_disconnected") { outcome = r; break; }
                if (r == "fight_won") { fights++; fightWins++; }
                else if (r == "fight_lost") fights++;
                if (ReferenceEquals(h.CurrentRoom, before)) break;
            }
            if (outcome != null) break;
            if (h.PlayerDead) { outcome = "player_dead"; break; }
        }

        return new RunResult(outcome ?? "floor_cap", floors, fights, fightWins);
    }

    private static async Task<string> HandleRoomAsync(RunHarness h, AbstractRoom room, int turnCap)
    {
        // Out-of-combat potions are offered as an ordinary decision, so the run policy can
        // spend a healing potion at a campfire the same way it spends gold at a shop.
        if (room is not CombatRoom) { if (await OfferOutOfCombatPotionAsync(h) == "policy_disconnected") return "policy_disconnected"; }

        switch (room)
        {
            case TreasureRoom tr:
            {
                var relics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
                var choice = 0;
                if (relics != null && relics.Count > 1)
                {
                    var opts = relics.Select((r, i) => (object)new Dictionary<string, object>
                    {
                        ["kind"] = "relic", ["index"] = i, ["id"] = r.Id.Entry,
                        ["model_idx"] = Vocab.Get(Vocab.Relics, r.Id.Entry),
                        ["vals"] = Obs.VarValues(r.DynamicVars),
                    }).ToArray();
                    choice = await Decide("treasure", h.PlayerObs(), opts, relics.Count);
                    if (choice < 0) return "policy_disconnected";
                }
                await RunRooms.ResolveTreasureAsync(h.Player, tr, choice);
                return "treasure";
            }

            case RestSiteRoom rs:
                await RunRooms.ResolveRestSiteAsync(h.Player, rs, RestChooser(h), SelectionAnswer(h));
                return "rest";

            case MerchantRoom mr:
                await RunRooms.ResolveShopAsync(h.Player, mr, ShopChooser(h), SelectionAnswer(h));
                return "shop";

            case EventRoom er:
                await RunRooms.ResolveEventAsync(h.Player, er, EventChooser(h), SelectionAnswer(h));
                return "event";

            case CombatRoom cr when !cr.IsPreFinished:
                return await FightAsync(h, cr, turnCap);
        }
        return "none";
    }

    // ---------- combat ----------

    private static async Task<string> FightAsync(RunHarness h, CombatRoom cr, int turnCap)
    {
        // Potion gate: the RUN policy decides whether this fight may spend potions, before a
        // single card is played. This is the split the design calls for - holding a potion for
        // the act boss is a run-level judgement the combat policy cannot make, because it only
        // ever sees one fight.
        if (h.Player.Potions.Any())
        {
            var gate = await Decide("potion_gate", GateObs(h, cr), new object[]
            {
                new Dictionary<string, object> { ["kind"] = "deny",  ["index"] = 0 },
                new Dictionary<string, object> { ["kind"] = "allow", ["index"] = 1 },
            }, 2);
            if (gate < 0) return "policy_disconnected";
            if (gate == 1) Potions.SetGate(null); else Potions.ClearGate();
        }
        else Potions.ClearGate();

        RunManager.Instance.ActionExecutor.Unpause();

        var startHp = h.Player.Creature.CurrentHp;
        int turns = 0;
        string outcome = null;

        while (turns < turnCap && outcome == null)
        {
            outcome = await Episodes.WaitForDecisionPublic(h.Player);
            if (outcome != null) break;

            var combat = CombatManager.Instance.DebugOnlyGetState();
            var legal = Obs.LegalActions(h.Player, combat);
            if (legal.Count == 0) { outcome = "no_legal_actions"; break; }

            // Byte-identical to Serve.cs, so the trained combat policy sees what it trained on.
            var idx = await Decide("combat", Obs.Extract(h.Player, combat),
                                   legal.Select(a => (object)a.ToDict()).ToArray(), legal.Count);
            if (idx < 0) return "policy_disconnected";
            var choice = legal[Math.Clamp(idx, 0, legal.Count - 1)];

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
                var expect = h.Player.PlayerCombatState.TurnNumber + 1;
                CombatManager.Instance.SetReadyToEndTurn(h.Player, false, null);
                turns++;
                outcome = await Episodes.WaitForTurnPublic(h.Player, expect);
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
        CardSelect.Reset();
        Potions.ClearGate();

        if (!h.PlayerDead) await RunRooms.WaitForCombatEndAsync();

        // Per-fight outcome, so the combat policy keeps its own objective.
        await Send(new Dictionary<string, object>
        {
            ["t"] = "fight_end", ["run"] = _run, ["won"] = won, ["outcome"] = outcome,
            ["room"] = cr.RoomType.ToString(), ["encounter"] = cr.Encounter.Id.Entry,
            // The fight's own act, so a consumer never has to infer it from whichever
            // decision happened to precede the fight.
            ["act"] = h.ActIndex, ["act_floor"] = h.RunState.ActFloor,
            ["turns"] = turns, ["hp_start"] = startHp, ["hp_end"] = h.Player.Creature.CurrentHp,
            ["max_hp"] = h.Player.Creature.MaxHp,
        });

        if (won && !h.PlayerDead)
        {
            try { await RunRewards.OfferForCombatAsync(cr, h.Player); }
            catch (Exception e) { Probe.Log($"  rewards: {e.Message}"); }
        }
        if (h.RunState.CurrentRoomCount > 1 && !h.PlayerDead)
        {
            try { await RunRooms.ResumeParentRoomAsync(); }
            catch (Exception e) { Probe.Log($"  resume: {e.Message}"); }
        }
        return won ? "fight_won" : "fight_lost";
    }

    // ---------- policy hooks ----------

    /// <summary>
    /// Wires the reward selectors to the socket.
    ///
    /// Gold, relics, potions and special cards are taken automatically: none of them has a
    /// downside worth a round trip (a full potion belt is refused by the engine, not by us).
    /// The card reward is the decision that matters - which card enters the deck, or none -
    /// and it is the one that decides whether the run can beat a boss, so it is asked.
    /// </summary>
    private static void InstallPolicyHooks(RunHarness h)
    {
        RunRewards.ChooseRewards = (p, options) =>
            Task.FromResult(Enumerable.Range(0, options.Count).ToList());

        RunRewards.ChooseCard = async (p, offered) =>
        {
            var idx = await Decide("card_reward", h.PlayerObs(),
                                   offered.Select(o => (object)o.ToDict()).ToArray(), offered.Count);
            if (idx < 0 || idx >= offered.Count) return new List<CardRewardOption>();
            return new List<CardRewardOption> { offered[idx] };
        };

        CrystalSphere.ChooseCell = null;   // random cell choice; a minigame, not a run decision
    }

    private static Func<Player, IReadOnlyList<RestOption>, Task<int>> RestChooser(RunHarness h) =>
        async (p, opts) =>
        {
            var usable = opts.Where(o => o.Enabled).ToList();
            if (usable.Count == 0) return -1;
            // "leave" is always available, so resting is never forced.
            var acts = usable.Select(o => (object)o.ToDict())
                .Append(new Dictionary<string, object> { ["index"] = -1, ["option"] = "LEAVE", ["enabled"] = true })
                .ToArray();
            var i = await Decide("rest", h.PlayerObs(), acts, acts.Length);
            if (i < 0 || i >= usable.Count) return -1;
            return usable[i].Index;
        };

    private static Func<Player, IReadOnlyList<EventOptionInfo>, Task<int>> EventChooser(RunHarness h) =>
        async (p, opts) =>
        {
            var choosable = opts.Where(o => !o.Locked).ToList();
            if (choosable.Count == 0) return -1;
            var i = await Decide("event", h.PlayerObs(),
                                 choosable.Select(o => (object)o.ToDict()).ToArray(), choosable.Count);
            if (i < 0) return -1;
            return choosable[Math.Clamp(i, 0, choosable.Count - 1)].Index;
        };

    private static Func<Player, IReadOnlyList<ShopOption>, Task<int>> ShopChooser(RunHarness h) =>
        async (p, offered) =>
        {
            var can = offered.Where(o => o.Affordable).ToList();
            if (can.Count == 0) return -1;
            var acts = can.Select(o => (object)o.ToDict())
                .Append(new Dictionary<string, object> { ["index"] = -1, ["kind"] = "leave", ["id"] = "LEAVE", ["cost"] = 0 })
                .ToArray();
            var i = await Decide("shop", h.PlayerObs(), acts, acts.Length);
            if (i < 0 || i >= can.Count) return -1;   // last index (or a bad reply) = leave
            return can[i].Index;
        };

    /// <summary>
    /// Out-of-combat card prompts (smith at a campfire, shop removal, an event that
    /// transforms a card). Same prompt object the combat loop answers, different decision kind
    /// because the observation around it is the run, not a fight.
    /// </summary>
    private static Func<PendingSelection, Task> SelectionAnswer(RunHarness h) =>
        async pending =>
        {
            while (pending.Remaining.Count > 0 && !pending.MustFinish)
            {
                var rem = pending.Remaining;
                var acts = rem.Select((c, i) => (object)new Dictionary<string, object>
                {
                    ["kind"] = "select_card", ["index"] = i, ["id"] = c.Id.Entry,
                    ["card_idx"] = Vocab.Get(Vocab.Cards, c.Id.Entry),
                    ["upgrade"] = c.CurrentUpgradeLevel,
                    ["kw"] = Obs.Keywords(c),
                    ["vals"] = Obs.CardValues(c, null),
                }).ToList();
                var canFinish = pending.CanFinish;
                if (canFinish)
                    acts.Add(new Dictionary<string, object> { ["kind"] = "select_done", ["index"] = -1, ["id"] = "" });

                var i2 = await Decide("card_select", h.PlayerObs(), acts.ToArray(), acts.Count);
                if (i2 < 0) { pending.Finish(); return; }
                if (canFinish && i2 == acts.Count - 1) { pending.Finish(); return; }
                pending.Pick(rem[Math.Clamp(i2, 0, rem.Count - 1)]);
                if (CardSelect.Pending != pending) return;   // Pick() auto-finished it
            }
            if (CardSelect.Pending == pending) pending.Finish();
        };

    private static async Task<string> OfferOutOfCombatPotionAsync(RunHarness h)
    {
        var slots = Potions.UsableOutOfCombat(h.Player);
        if (slots.Count == 0) return "none";

        var potions = h.Player.Potions.ToList();
        var acts = slots.Select(s => (object)new Dictionary<string, object>
        {
            ["kind"] = "drink", ["index"] = s, ["id"] = potions[s].Id.Entry,
            ["potion_idx"] = Vocab.Get(Vocab.Potions, potions[s].Id.Entry),
            ["vals"] = Obs.VarValues(potions[s].DynamicVars),
        }).Append(new Dictionary<string, object> { ["kind"] = "hold", ["index"] = -1, ["id"] = "" }).ToArray();

        var i = await Decide("potion_ooc", h.PlayerObs(), acts, acts.Length);
        if (i < 0) return "policy_disconnected";
        if (i >= slots.Count) return "hold";

        try
        {
            potions[slots[i]].EnqueueManualUse(null);
            await RunRooms.DriveWithSelections(RunManager.Instance.ActionQueueSet.BecameEmpty(),
                                               SelectionAnswer(h));
        }
        catch (Exception e) { Probe.Log($"  ooc potion: {e.Message}"); }
        return "drink";
    }

    private static Dictionary<string, object> GateObs(RunHarness h, CombatRoom cr)
    {
        var o = h.PlayerObs();
        o["room_type"] = cr.RoomType.ToString();
        o["encounter"] = cr.Encounter.Id.Entry;
        o["act"] = h.ActIndex;
        o["act_floor"] = h.RunState.ActFloor;
        o["potion_slots"] = Potions.Describe(h.Player, inCombat: false);
        return o;
    }

    // ---------- transport ----------

    /// <summary>One request/response round trip. Returns -1 if the policy went away.</summary>
    private static async Task<int> Decide(string kind, Dictionary<string, object> obs, object[] legal, int count)
    {
        _byKind[kind] = _byKind.GetValueOrDefault(kind) + 1;
        _steps++;
        await Send(new Dictionary<string, object>
        {
            ["t"] = "decision", ["kind"] = kind, ["run"] = _run, ["step"] = _step++,
            ["obs"] = obs, ["legal"] = legal,
        });
        var line = await _r.ReadLineAsync();
        if (line == null) return -1;
        return Math.Clamp(ParseAction(line), 0, Math.Max(0, count - 1));
    }

    private static Task Send(Dictionary<string, object> msg) =>
        _w.WriteLineAsync(Probe.ToJsonCompact(msg));

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
