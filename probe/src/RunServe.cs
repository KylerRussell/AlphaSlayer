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

    // Run search (RunSearch): every decision answered since the last room entry, and that
    // room's entry save; a rollout job's forced answers.
    private static readonly List<int> _runLog = new();
    private static string _roomSave;
    private static Queue<int> _forced;
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
        // --probe-run-queue: the server hands out every run (id, seed, character) on request,
        // and the process lives until told to quit. A persistent pool of these replaces one
        // process launch per 4 runs, and fast processes pick up the next run instead of
        // idling while the slowest finishes a fixed share.
        var queue = Probe.HasFlag("probe-run-queue");
        var played = 0;

        for (var i = 0; (queue || i < nRuns) && !disconnected; i++)
        {
            var runSeed = "";
            var runChar = Probe.ArgValue("probe-character");
            if (queue)
            {
                var job = await NextRun();
                if (job == null) break;                       // quit, or the server went away
                (_run, runSeed, runChar) = job.Value;
            }
            else
            {
                _run = i + runOffset;
                runSeed = $"{seed}{_run}";
            }
            _step = 0;
            played++;

            FightRestore.Install();
            var h = new RunHarness();
            _runLog.Clear();
            _roomSave = null;
            _forced = null;
            if (RunSearch.Active)
            {
                // a rollout: the run as it was at a room entry, then the forced answers
                await h.StartFromSaveAsync(FightRestore.Load(RunSearch.Job.Restore));
                _forced = new Queue<int>(RunSearch.Job.Forced);
            }
            else await h.StartRunAsync(runSeed, ascension, runChar);
            h.ActCap = Probe.ArgInt("probe-act-cap", int.MaxValue);
            CardSelect.Install();
            RunRewards.Install();
            CrystalSphere.Install();
            InstallPolicyHooks(h);

            var res = await PlayRunAsync(h, turnCap, floorCap, resumeInRoom: RunSearch.Active);
            outcomes[res.outcome] = outcomes.GetValueOrDefault(res.outcome) + 1;

            await Send(new Dictionary<string, object>
            {
                ["t"] = "terminal", ["run"] = _run, ["outcome"] = res.outcome,
                ["rollout"] = RunSearch.Active ? RunSearch.Report(h) : null,
                ["won"] = res.outcome == "run_complete",
                ["floors"] = res.floors, ["act"] = h.ActIndex, ["acts"] = h.EffectiveActCount,
                ["hp"] = h.Player.Creature.CurrentHp, ["max_hp"] = h.Player.Creature.MaxHp,
                ["gold"] = h.Player.Gold, ["deck"] = h.Player.Deck.Cards.Count,
                ["relics"] = h.Player.Relics.Count,
                ["fights"] = res.fights, ["fight_wins"] = res.fightWins,
            });
            FightSearch.WriteStats();
            CardSelect.Reset();
            Potions.ClearGate();
            if (res.outcome == "policy_disconnected" && !RunSearch.Done) disconnected = true;
        }

        sw.Stop();
        nRuns = played;
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

    private static async Task<RunResult> PlayRunAsync(RunHarness h, int turnCap, int floorCap,
                                                      bool resumeInRoom = false)
    {
        string outcome = null;
        int floors = 0, fights = 0, fightWins = 0;
        // a restored run (rollout) starts INSIDE its room: resolve it before the first travel
        var inRoom = resumeInRoom;

        while (outcome == null && floors < floorCap)
        {
          if (!inRoom)
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
            // (a run-search rollout valuing a deck change goes straight to the act boss)
            var teleport = RunSearch.Active && RunSearch.WorldStarted && RunSearch.Job.Teleport
                           && h.RunState.Map?.BossMapPoint != null;
            var pick = teleport ? 0
                : await Decide("travel", h.MapObs(), legal.Select(a => (object)a.ToDict()).ToArray(), legal.Count);
            if (pick < 0) { outcome = "policy_disconnected"; break; }

            try
            {
                await h.TravelAsync(teleport ? h.RunState.Map.BossMapPoint.coord
                                             : legal[Math.Clamp(pick, 0, legal.Count - 1)].Coord);
            }
            catch (Exception e) { Probe.Log($"  travel: {e.Message}"); outcome = "travel_failed"; break; }
            floors++;
            // the room just entered: its entry save (captured inside TravelAsync) and, from
            // here, its decisions
            _roomSave = FightRestore.RoomSaveJson;
            _runLog.Clear();
          }
          inRoom = false;

            // --- resolve whatever room we landed in (an event can push a combat on top) ---
            for (var depth = 0; depth < 8; depth++)
            {
                var before = h.CurrentRoom;
                if (before == null) break;
                var r = await HandleRoomAsync(h, before, turnCap);
                if (r == "policy_disconnected") { outcome = r; break; }
                if (r == "fight_won") { fights++; fightWins++; }
                else if (r == "fight_lost") fights++;
                if (r == "rollout_done") { outcome = r; break; }
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

            case CombatRoom cr when !cr.IsPreFinished && RunSearch.IsHorizon(cr):
                return await BossPlayoutsAsync(h, cr, turnCap);

            case CombatRoom cr when !cr.IsPreFinished:
                return await FightAsync(h, cr, turnCap);
        }
        return "none";
    }

    // ---------- combat ----------

    private static async Task<string> FightAsync(RunHarness h, CombatRoom cr, int turnCap)
    {
        // Restore support (FightRestore): the room-entry save this fight can be rebuilt from,
        // and every choice made in it, as indices into the legal lists below. Only fights
        // entered straight from the map: an event's fight sits on top of the event room.
        var restoreJson = FightRestore.On && h.RunState.CurrentRoomCount <= 1 ? FightRestore.RoomSaveJson : null;
        var log = new List<int>();
        var gateChoice = -1;

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
            gateChoice = gate;
        }
        ApplyGate(gateChoice);

        RunManager.Instance.ActionExecutor.Unpause();

        var startHp = h.Player.Creature.CurrentHp;
        int turns = 0;
        string outcome = null;
        var restoreTests = 0;
        var lastTestTurn = -1;
        var maxTests = Probe.ArgInt("probe-restore-test", 0);

        while (turns < turnCap && outcome == null)
        {
            outcome = await Episodes.WaitForDecisionPublic(h.Player);
            if (outcome != null) break;
            // A restore replays this fight's choices from the room entry, so it needs the
            // fight from its first decision. Entered later (a fight left early and re-entered),
            // no restore: replaying an empty log would rewind the fight.
            if (log.Count == 0 && restoreJson != null && (h.Player.PlayerCombatState?.TurnNumber ?? 1) > 1)
            {
                Probe.Log($"  fight entered at turn {h.Player.PlayerCombatState.TurnNumber}: no restore/search");
                restoreJson = null;
            }

            if (restoreJson != null && restoreTests < maxTests && turns >= 1 && turns != lastTestTurn)
            {
                lastTestTurn = turns;
                restoreTests++;
                // --probe-restore-stress=N: N restores at this decision (leak profiling)
                var reps = Math.Max(1, Probe.ArgInt("probe-restore-stress", 1));
                string res = null;
                CombatHarness.FrameSuspend = FightSearch.FrameSuspend;
                try
                {
                    for (var rep = 0; rep < reps && res == null; rep++)
                        res = await RestoreTestAsync(h, restoreJson, gateChoice, log);
                }
                finally { CombatHarness.FrameSuspend = false; }
                if (h.CurrentRoom is CombatRoom ncr) cr = ncr;
                if (res != null) { outcome = res; break; }
            }

            // Search (FightSearch): value every legal choice by simulating it from a restore.
            // It ends on a restore of the live state, so every game object below is new.
            double[] q = null;
            if (restoreJson != null && FightSearch.Wants(cr))
            {
                var liveYield = Probe.YieldEvery;
                if (FightSearch.YieldEvery > 0) Probe.YieldEvery = FightSearch.YieldEvery;
                CombatHarness.FrameSuspend = FightSearch.FrameSuspend;
                double[] qq; string res;
                try
                {
                    (qq, res) = FightSearch.TurnMode ? await SearchTurnAsync(h, restoreJson, gateChoice, log)
                                                     : await SearchAsync(h, restoreJson, gateChoice, log);
                }
                finally { Probe.YieldEvery = liveYield; CombatHarness.FrameSuspend = false; }
                q = qq;
                if (h.CurrentRoom is CombatRoom ncr) cr = ncr;
                // could not get back to the live decision: play on from wherever the game is,
                // without search (leaving the fight here would re-enter it mid-way with an
                // empty choice log, and a restore from that would rewind the fight)
                if (res == "search_restore_failed") restoreJson = null;
                else if (res != null) { outcome = res; break; }
            }

            var combat = CombatManager.Instance.DebugOnlyGetState();
            var legal = Obs.LegalActions(h.Player, combat);
            if (legal.Count == 0) { outcome = "no_legal_actions"; break; }

            // Byte-identical to Serve.cs, so the trained combat policy sees what it trained on.
            var idx = await Decide("combat", Obs.Extract(h.Player, combat),
                                   legal.Select(a => (object)a.ToDict()).ToArray(), legal.Count, q);
            if (idx < 0) return "policy_disconnected";
            idx = Math.Clamp(idx, 0, legal.Count - 1);
            log.Add(idx);
            (outcome, var ended) = await ApplyCombatChoiceAsync(h, legal[idx]);
            if (ended) turns++;
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

    /// <summary>
    /// A rollout's horizon (RunSearch): this boss fight, played RunSearch.Job.BossPlays times
    /// from its room-entry save, each with its own draw order and combat RNGs, by the policy.
    /// </summary>
    private static async Task<string> BossPlayoutsAsync(RunHarness h, CombatRoom cr, int turnCap)
    {
        var save = FightRestore.RoomSaveJson;
        RunSearch.Boss = cr.Encounter.Id.Entry;
        RunSearch.HpAtBoss = h.Player.Creature.CurrentHp;
        RunSearch.DeckAtBoss = h.Player.Deck.Cards.Count;
        var n = Math.Max(1, RunSearch.Job.BossPlays);
        for (var b = 0; b < n; b++)
        {
            if (b > 0)
            {
                CardSelect.Reset();
                await h.RestoreRoomEntryAsync(FightRestore.Load(save));
                var seed = unchecked(RunSearch.Job.World * 7919 + b * 104729 + 17);
                FightSearch.ShuffleDrawPile(h.Player, seed);
                FightSearch.MockCombatRngs(h.RunState, seed);
            }
            var gate = -1;
            if (h.Player.Potions.Any())
            {
                var room = h.CurrentRoom as CombatRoom ?? cr;
                gate = await Decide("potion_gate", GateObs(h, room), new object[]
                {
                    new Dictionary<string, object> { ["kind"] = "deny",  ["index"] = 0 },
                    new Dictionary<string, object> { ["kind"] = "allow", ["index"] = 1 },
                }, 2);
                if (gate < 0) return "policy_disconnected";
            }
            ApplyGate(gate);
            RunManager.Instance.ActionExecutor.Unpause();
            string outcome = null;
            var turns = 0;
            while (turns < turnCap && outcome == null)
            {
                outcome = await Episodes.WaitForDecisionPublic(h.Player);
                if (outcome != null) break;
                var combat = CombatManager.Instance.DebugOnlyGetState();
                var legal = Obs.LegalActions(h.Player, combat);
                if (legal.Count == 0) { outcome = "no_legal_actions"; break; }
                var idx = await Decide("combat", Obs.Extract(h.Player, combat),
                                       legal.Select(a => (object)a.ToDict()).ToArray(), legal.Count);
                if (idx < 0) return "policy_disconnected";
                (outcome, var ended) = await ApplyCombatChoiceAsync(h, legal[Math.Clamp(idx, 0, legal.Count - 1)]);
                if (ended) turns++;
            }
            var won = outcome == "enemies_cleared";
            RunSearch.BossDmg.Add(won ? 1.0 : RunSearch.DamageFraction(CombatManager.Instance.DebugOnlyGetState()));
            RunSearch.BossPlayed++;
            if (won) RunSearch.BossWon++;
            RunSearch.BossHp.Add(won ? h.Player.Creature.CurrentHp : 0);
            RunSearch.BossOutcomes.Add(outcome ?? "turn_cap");
            RunSearch.BossTurns.Add(turns);
            CardSelect.Reset();
            Potions.ClearGate();
            if (!h.PlayerDead) await RunRooms.WaitForCombatEndAsync();
        }
        RunSearch.Done = true;
        return "rollout_done";
    }

    private static void ApplyGate(int gateChoice)
    {
        if (gateChoice == 1) Potions.SetGate(null); else Potions.ClearGate();
    }

    /// <summary>Applies one combat choice. Returns (outcome or null, whether it ended the turn).</summary>
    private static async Task<(string, bool)> ApplyCombatChoiceAsync(RunHarness h, LegalAction choice)
    {
        if (choice.Kind == "select_card" || choice.Kind == "select_done")
        {
            var pend = CardSelect.Pending;
            if (choice.Kind == "select_done") pend?.Finish(); else pend?.Pick(choice.Card);
            CardSelect.ClearRaised();
            await CombatHarness.Suspend();
            return (null, false);
        }
        if (choice.Kind == "use_potion")
        {
            choice.Potion.EnqueueManualUse(choice.Target);
            await Episodes.WaitForQueueOrSelectionPublic();
            return (null, false);
        }
        if (choice.Kind == "end_turn")
        {
            var expect = h.Player.PlayerCombatState.TurnNumber + 1;
            CombatManager.Instance.SetReadyToEndTurn(h.Player, false, null);
            return (await Episodes.WaitForTurnPublic(h.Player, expect), true);
        }
        RunManager.Instance.ActionQueueSet.EnqueueWithoutSynchronizing(
            new PlayCardAction(choice.Card, choice.Target));
        await Episodes.WaitForQueueOrSelectionPublic();
        return (null, false);
    }

    /// <summary>Restore to the room-entry save and re-apply the fight's choices; null = landed
    /// on the next decision, else why not.</summary>
    private static async Task<string> RestoreReplayAsync(RunHarness h, string saveJson, int gateChoice,
                                                         List<int> log, int n)
    {
        CardSelect.Reset();
        var lt = Stopwatch.StartNew();
        await h.RestoreRoomEntryAsync(FightRestore.Load(saveJson));
        FightSearch.LoadMs += lt.Elapsed.TotalMilliseconds;
        ApplyGate(gateChoice);
        RunManager.Instance.ActionExecutor.Unpause();
        for (var step = 0; step < n; step++)
        {
            if (await Episodes.WaitForDecisionPublic(h.Player) != null) return $"fight ended during replay at {step}";
            var legal = Obs.LegalActions(h.Player, CombatManager.Instance.DebugOnlyGetState());
            if (log[step] >= legal.Count) return $"choice {log[step]} out of {legal.Count} at {step}";
            var (o, _) = await ApplyCombatChoiceAsync(h, legal[log[step]]);
            if (o != null) return $"fight ended during replay at {step}";
        }
        return await Episodes.WaitForDecisionPublic(h.Player) != null ? "fight ended after replay" : null;
    }

    /// <summary>
    /// One-step lookahead: for each legal choice (up to --probe-search-cands), restore, shuffle
    /// the draw pile with a search-only RNG (the search must not see the real draw order),
    /// apply the choice and run to the next decision or the end of the fight; the leaves go to
    /// the server in ONE message and come back as values. Then restore the live state and
    /// check it is the decision we left (FightSearch.Diverged counts any that is not).
    /// Returns the values, or null if the search could not run.
    /// </summary>
    private static async Task<(double[], string)> SearchAsync(RunHarness h, string saveJson, int gateChoice, List<int> log)
    {
        var sw = Stopwatch.StartNew();
        var key = DecisionKey(h);
        var nLegal = Obs.LegalActions(h.Player, CombatManager.Instance.DebugOnlyGetState()).Count;
        var nCand = Math.Min(nLegal, FightSearch.MaxCands);
        if (nCand < 2) return (null, null);
        var leaves = new List<object>();
        for (var c = 0; c < nCand; c++)
        {
            var err = await RestoreReplayAsync(h, saveJson, gateChoice, log, log.Count);
            if (err != null) { FightSearch.Errors++; break; }
            FightSearch.ShuffleDrawPile(h.Player, unchecked(_run * 1_000_003 + _step * 131 + c));
            var legal = Obs.LegalActions(h.Player, CombatManager.Instance.DebugOnlyGetState());
            if (legal.Count != nLegal) { FightSearch.Errors++; break; }
            string o;
            try
            {
                (o, _) = await ApplyCombatChoiceAsync(h, legal[c]);
                o ??= await Episodes.WaitForDecisionPublic(h.Player);
            }
            catch (Exception e) { o = "error:" + e.GetType().Name; }
            var leaf = new Dictionary<string, object>
            {
                ["act"] = h.ActIndex, ["hp"] = h.Player.Creature.CurrentHp, ["max_hp"] = h.Player.Creature.MaxHp,
            };
            if (o == null)
            {
                var cs = CombatManager.Instance.DebugOnlyGetState();
                leaf["obs"] = Obs.Extract(h.Player, cs);
                leaf["legal"] = Obs.LegalActions(h.Player, cs).Select(a => (object)a.ToDict()).ToArray();
            }
            else leaf["end"] = o;
            leaves.Add(leaf);
        }
        // back to the live decision, with the REAL draw order
        var back = await RestoreReplayAsync(h, saveJson, gateChoice, log, log.Count);
        if (back != null) { FightSearch.Errors++; return (null, "search_restore_failed"); }
        var now = DecisionKey(h);
        if (now != key)
        {
            FightSearch.Diverged++;
            var i = 0;
            while (i < key.Length && i < now.Length && key[i] == now[i]) i++;
            var a0 = Math.Max(0, i - 150);
            File.AppendAllText(Path.Combine(Probe.OutDir, "diverged.jsonl"), Probe.ToJsonCompact(new Dictionary<string, object>
            {
                ["run"] = _run, ["step"] = _step, ["choices"] = log.Count, ["diff_at"] = i,
                ["before"] = key.Substring(a0, Math.Min(300, key.Length - a0)),
                ["after"] = now.Substring(a0, Math.Min(300, now.Length - a0)),
            }) + "\n");
        }
        FightSearch.Searches++;
        FightSearch.Leaves += leaves.Count;
        if (leaves.Count != nCand) return (null, null);
        var v = await EvalLeaves(leaves);
        FightSearch.Ms += sw.Elapsed.TotalMilliseconds;
        if (v == null || v.Length != nCand) return (null, null);
        // choices past the cap get no value (-1e9), so a server picking argmax never takes them
        var q = Enumerable.Repeat(-1e9, nLegal).ToArray();
        Array.Copy(v, q, nCand);
        return (q, null);
    }

    /// <summary>
    /// Turn search (AlphaZero-style, one turn deep): the policy's prior picks the candidates
    /// (top-k, up to --probe-search-mass of its probability); each is applied in
    /// --probe-search-samples sampled worlds (draw pile shuffled, combat RNGs mocked, the SAME
    /// worlds for every candidate) and the policy then plays the rest of the turn (rollout); the
    /// leaf is the start of the next turn after the enemies have acted, or the fight's end, so
    /// every candidate is valued at the same point of the game. q = mean leaf value over the
    /// samples; choices that were not candidates get -1e9. Ends with a restore of the live state.
    /// </summary>
    private static async Task<(double[], string)> SearchTurnAsync(RunHarness h, string saveJson, int gateChoice, List<int> log)
    {
        var sw = Stopwatch.StartNew();
        var key = DecisionKey(h);
        var combat0 = CombatManager.Instance.DebugOnlyGetState();
        var legal0 = Obs.LegalActions(h.Player, combat0);
        var nLegal = legal0.Count;
        if (nLegal < 2) return (null, null);
        var t = Stopwatch.StartNew();
        var prior = await AskPrior(Obs.Extract(h.Player, combat0), legal0.Select(a => (object)a.ToDict()).ToArray());
        FightSearch.PriorMs += t.Elapsed.TotalMilliseconds;
        if (prior == null || prior.Length < nLegal) return (null, null);
        var order = Enumerable.Range(0, nLegal).OrderByDescending(i => prior[i]).ToList();
        var cands = new List<int>();
        double mass = 0;
        foreach (var i in order)
        {
            if (cands.Count >= Math.Max(2, FightSearch.K) || (cands.Count >= 2 && mass >= FightSearch.Mass)) break;
            cands.Add(i);
            mass += prior[i];
        }
        var S = Math.Max(1, FightSearch.Samples);
        var leaves = new List<object>();
        var failed = false;
        foreach (var c in cands)
        {
            for (var smp = 0; smp < S && !failed; smp++)
            {
                t.Restart();
                var err = await RestoreReplayAsync(h, saveJson, gateChoice, log, log.Count);
                FightSearch.RestoreMs += t.Elapsed.TotalMilliseconds;
                FightSearch.ReplayChoices += log.Count;
                if (err != null) { FightSearch.Errors++; failed = true; break; }
                var world = unchecked(_run * 1_000_003 + _step * 131 + smp * 7919);   // same for every candidate
                FightSearch.ShuffleDrawPile(h.Player, world);
                FightSearch.MockCombatRngs(h.RunState, world);
                var legal = Obs.LegalActions(h.Player, CombatManager.Instance.DebugOnlyGetState());
                if (legal.Count != nLegal) { FightSearch.Errors++; failed = true; break; }
                t.Restart();
                string o;
                try
                {
                    var rt = Stopwatch.StartNew();
                    var (oc, ended) = await ApplyCombatChoiceAsync(h, legal[c]);
                    o = oc;
                    var steps = 0;
                    while (o == null && !ended)
                    {
                        o = await Episodes.WaitForDecisionPublic(h.Player);
                        FightSearch.RollApplyMs += rt.Elapsed.TotalMilliseconds;
                        if (o != null) break;
                        if (steps++ >= FightSearch.RollCap) { FightSearch.RollCapped++; break; }
                        rt.Restart();
                        var cs = CombatManager.Instance.DebugOnlyGetState();
                        var lg = Obs.LegalActions(h.Player, cs);
                        if (lg.Count == 0) break;
                        int a;
                        if (lg.Count == 1) { a = 0; FightSearch.ForcedRolls++; }   // nothing to ask
                        else
                        {
                            var robs = Obs.Extract(h.Player, cs);
                            var rlegal = lg.Select(x => (object)x.ToDict()).ToArray();
                            FightSearch.RollObsMs += rt.Elapsed.TotalMilliseconds;
                            rt.Restart();
                            a = await AskRoll(robs, rlegal, lg.Count);
                            FightSearch.RollAskMs += rt.Elapsed.TotalMilliseconds;
                            if (a < 0) return (null, "policy_disconnected");
                            FightSearch.Rolls++;
                        }
                        rt.Restart();
                        (o, ended) = await ApplyCombatChoiceAsync(h, lg[a]);
                    }
                    o ??= await Episodes.WaitForDecisionPublic(h.Player);
                }
                catch (Exception e) { o = "error:" + e.GetType().Name; }
                FightSearch.RollMs += t.Elapsed.TotalMilliseconds;
                var leaf = new Dictionary<string, object>
                {
                    ["act"] = h.ActIndex, ["hp"] = h.Player.Creature.CurrentHp, ["max_hp"] = h.Player.Creature.MaxHp,
                };
                if (o == null)
                {
                    var cs = CombatManager.Instance.DebugOnlyGetState();
                    leaf["obs"] = Obs.Extract(h.Player, cs);
                    leaf["legal"] = Obs.LegalActions(h.Player, cs).Select(a => (object)a.ToDict()).ToArray();
                }
                else leaf["end"] = o;
                leaves.Add(leaf);
            }
            if (failed) break;
        }
        // back to the live decision, with the REAL draw order and RNGs. A replay is very
        // rarely not exact (~1 in 2,000 searches); another attempt usually lands, and a
        // live game left anywhere else would carry on from the wrong state.
        string back = null, now = null;
        for (var attempt = 0; attempt < 3; attempt++)
        {
            if (attempt > 0) FightSearch.Retries++;
            t.Restart();
            // test hooks: the first attempt (or every attempt) replays one choice short
            var bad = log.Count > 0 && (Probe.HasFlag("probe-search-test-lost")
                                        || attempt == 0 && Probe.HasFlag("probe-search-test-retry"));
            back = await RestoreReplayAsync(h, saveJson, gateChoice, log, bad ? log.Count - 1 : log.Count);
            FightSearch.RestoreMs += t.Elapsed.TotalMilliseconds;
            FightSearch.ReplayChoices += log.Count;
            now = back == null ? DecisionKey(h) : null;
            if (now == key) break;
        }
        if (now != key)
        {
            // the live state is lost: logged per run, so an evaluation can leave the run out
            if (back != null) FightSearch.Errors++; else FightSearch.Diverged++;
            now ??= "";
            var i = 0;
            while (i < key.Length && i < now.Length && key[i] == now[i]) i++;
            var a0 = Math.Max(0, i - 150);
            File.AppendAllText(Path.Combine(Probe.OutDir, "diverged.jsonl"), Probe.ToJsonCompact(new Dictionary<string, object>
            {
                ["run"] = _run, ["step"] = _step, ["choices"] = log.Count, ["diff_at"] = i, ["error"] = back,
                ["before"] = key.Substring(a0, Math.Min(300, key.Length - a0)),
                ["after"] = now.Substring(Math.Min(a0, now.Length), Math.Min(300, Math.Max(0, now.Length - a0))),
            }) + "\n");
            if (back != null) return (null, "search_restore_failed");
        }
        FightSearch.Searches++;
        FightSearch.Leaves += leaves.Count;
        if (FightSearch.Searches % 25 == 0) FightSearch.WriteStats();
        if (failed || leaves.Count != cands.Count * S) return (null, null);
        t.Restart();
        var v = await EvalLeaves(leaves);
        FightSearch.EvalMs += t.Elapsed.TotalMilliseconds;
        FightSearch.Ms += sw.Elapsed.TotalMilliseconds;
        if (v == null || v.Length != leaves.Count) return (null, null);
        var q = Enumerable.Repeat(-1e9, nLegal).ToArray();
        for (var ci = 0; ci < cands.Count; ci++)
        {
            double sum = 0;
            for (var smp = 0; smp < S; smp++) sum += v[ci * S + smp];
            q[cands[ci]] = sum / S;
        }
        return (q, null);
    }

    /// <summary>Round trip: {"t":"prior","obs","legal"} -> {"p":[...]} (the policy's probabilities).</summary>
    private static async Task<double[]> AskPrior(Dictionary<string, object> obs, object[] legal)
    {
        await _decideLock.WaitAsync();
        try
        {
            await Send(new Dictionary<string, object> { ["t"] = "prior", ["run"] = _run, ["obs"] = obs, ["legal"] = legal });
            var line = await _r.ReadLineAsync();
            if (line == null) return null;
            using var doc = System.Text.Json.JsonDocument.Parse(line);
            return doc.RootElement.GetProperty("p").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        }
        catch (Exception e) { Probe.Log($"  prior: {e.Message}"); return null; }
        finally { _decideLock.Release(); }
    }

    /// <summary>Round trip: {"t":"roll","obs","legal"} -> {"a":i}, the policy's choice inside a
    /// search rollout (not part of the run's trajectory). -1 if the policy went away.</summary>
    private static async Task<int> AskRoll(Dictionary<string, object> obs, object[] legal, int count)
    {
        await _decideLock.WaitAsync();
        try
        {
            await Send(new Dictionary<string, object> { ["t"] = "roll", ["run"] = _run, ["obs"] = obs, ["legal"] = legal });
            var line = await _r.ReadLineAsync();
            if (line == null) return -1;
            return Math.Clamp(ParseAction(line), 0, Math.Max(0, count - 1));
        }
        finally { _decideLock.Release(); }
    }

    /// <summary>Round trip: {"t":"eval","leaves":[...]} -> {"v":[...]}.</summary>
    private static async Task<double[]> EvalLeaves(List<object> leaves)
    {
        await _decideLock.WaitAsync();
        try
        {
            await Send(new Dictionary<string, object> { ["t"] = "eval", ["run"] = _run, ["leaves"] = leaves.ToArray() });
            var line = await _r.ReadLineAsync();
            if (line == null) return null;
            using var doc = System.Text.Json.JsonDocument.Parse(line);
            return doc.RootElement.GetProperty("v").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        }
        catch (Exception e) { Probe.Log($"  eval: {e.Message}"); return null; }
        finally { _decideLock.Release(); }
    }

    private static string DecisionKey(RunHarness h)
    {
        var combat = CombatManager.Instance.DebugOnlyGetState();
        return Probe.ToJsonCompact(new Dictionary<string, object>
        {
            ["obs"] = Obs.Extract(h.Player, combat),
            ["legal"] = Obs.LegalActions(h.Player, combat).Select(a => (object)a.ToDict()).ToArray(),
        });
    }

    /// <summary>
    /// Restores the fight to the room-entry save, re-applies every choice made so far, and
    /// checks the decision it lands on is identical to the one it left (observation and legal
    /// actions). Logs one JSON line to &lt;probe-out&gt;/restore.jsonl. The run continues from
    /// the RESTORED state, so a bad restore also shows up in the run's outcome. Returns an
    /// outcome only if the replay ended the fight early (a divergence).
    /// </summary>
    private static long _restoresTotal;

    private static async Task<string> RestoreTestAsync(RunHarness h, string saveJson, int gateChoice, List<int> log)
    {
        var before = DecisionKey(h);
        var oldCombat = CombatManager.Instance.DebugOnlyGetState();
        var oldRun = h.RunState;
        // negative control: replay one choice fewer, which must NOT match
        var n = Probe.HasFlag("probe-restore-test-drop-last") ? Math.Max(0, log.Count - 1) : log.Count;
        var sw = Stopwatch.StartNew();
        string err = null, outcome = null;
        double loadMs = 0;
        var step = 0;
        // profiling: where replay time goes (waiting for the decision, playing, ending turns)
        var sus0 = CombatHarness.Suspends; var of0 = Probe.OneFrames; var fr0 = Godot.Engine.GetProcessFrames();
        double waitMs = 0, playMs = 0, endMs = 0;
        int ends = 0;
        var ph = new Stopwatch();
        try
        {
            CardSelect.Reset();
            await h.RestoreRoomEntryAsync(FightRestore.Load(saveJson));
            loadMs = sw.Elapsed.TotalMilliseconds;
            ApplyGate(gateChoice);
            RunManager.Instance.ActionExecutor.Unpause();
            for (; step < n; step++)
            {
                ph.Restart();
                outcome = await Episodes.WaitForDecisionPublic(h.Player);
                waitMs += ph.Elapsed.TotalMilliseconds;
                if (outcome != null) { err = $"fight ended during replay at {step}"; break; }
                var legal = Obs.LegalActions(h.Player, CombatManager.Instance.DebugOnlyGetState());
                if (log[step] >= legal.Count) { err = $"choice {log[step]} out of {legal.Count} at {step}"; break; }
                ph.Restart();
                (outcome, _) = await ApplyCombatChoiceAsync(h, legal[log[step]]);
                if (legal[log[step]].Kind == "end_turn") { endMs += ph.Elapsed.TotalMilliseconds; ends++; }
                else playMs += ph.Elapsed.TotalMilliseconds;
                if (outcome != null) { err = $"fight ended during replay at {step}"; break; }
            }
            if (err == null)
            {
                outcome = await Episodes.WaitForDecisionPublic(h.Player);
                if (outcome != null) err = "fight ended after replay";
            }
        }
        catch (Exception e) { err = e.GetType().Name + ": " + e.Message; }
        var total = sw.Elapsed.TotalMilliseconds;
        var after = err == null ? DecisionKey(h) : null;
        var rec = new Dictionary<string, object>
        {
            ["run"] = _run, ["floor"] = h.RunState.TotalFloor, ["choices"] = log.Count, ["replayed"] = n,
            // a restore that silently did nothing would also "match"
            ["new_combat"] = !ReferenceEquals(oldCombat, CombatManager.Instance.DebugOnlyGetState()),
            ["new_run"] = !ReferenceEquals(oldRun, h.RunState),
            ["match"] = err == null && before == after, ["error"] = err,
            ["load_ms"] = Math.Round(loadMs, 1), ["replay_ms"] = Math.Round(total - loadMs, 1),
            ["wait_ms"] = Math.Round(waitMs, 1), ["play_ms"] = Math.Round(playMs, 1),
            ["end_ms"] = Math.Round(endMs, 1), ["ends"] = ends,
            ["suspends"] = CombatHarness.Suspends - sus0, ["one_frames"] = Probe.OneFrames - of0,
            ["frames"] = (long)(Godot.Engine.GetProcessFrames() - fr0),
            ["restores_total"] = ++_restoresTotal,
            ["nodes"] = (long)Godot.Performance.GetMonitor(Godot.Performance.Monitor.ObjectNodeCount),
            ["objects"] = (long)Godot.Performance.GetMonitor(Godot.Performance.Monitor.ObjectCount),
            ["heap_kb"] = GC.GetTotalMemory(false) / 1024,
        };
        var auditEvery = Probe.ArgInt("probe-leak-audit", 0);
        if (auditEvery > 0 && (_restoresTotal == 1 || _restoresTotal % auditEvery == 0)) LeakAudit.Report(_restoresTotal);
        if (err == null && before != after)
        {
            // first differing character, with context, to see WHAT diverged
            var i = 0;
            while (i < before.Length && i < after.Length && before[i] == after[i]) i++;
            rec["diff_at"] = i;
            rec["before"] = before.Substring(Math.Max(0, i - 120), Math.Min(240, before.Length - Math.Max(0, i - 120)));
            rec["after"] = after.Substring(Math.Max(0, i - 120), Math.Min(240, after.Length - Math.Max(0, i - 120)));
        }
        File.AppendAllText(Path.Combine(Probe.OutDir, "restore.jsonl"), Probe.ToJsonCompact(rec) + "\n");
        return outcome;
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

        RunRewards.AnswerSelection = SelectionAnswer(h);   // e.g. a card-removal reward
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

    // Decisions are not only asked from the run loop: game callbacks ask too (the reward
    // selector's card_reward, a card prompt answered through DriveWithSelections). When two
    // overlapped, the second ReadLineAsync threw "The stream is currently in use by a previous
    // operation" and ended the whole session (seed U4_2_52000, three separate runs). Each
    // round trip is now atomic; the second decision waits for the first answer, which depends
    // only on the policy, so this cannot deadlock.
    private static readonly System.Threading.SemaphoreSlim _decideLock = new(1, 1);
    private static string _decideKind = "";
    private static int _decideCollisions;

    /// <summary>One request/response round trip. Returns -1 if the policy went away.</summary>
    private static async Task<int> Decide(string kind, Dictionary<string, object> obs, object[] legal, int count,
                                          double[] q = null)
    {
        if (!_decideLock.Wait(0))
        {
            if (_decideCollisions++ < 3)
                Probe.Log($"concurrent decision: {kind} waits behind {_decideKind} " +
                          $"(run {_run}, step {_step})\n{Environment.StackTrace}");
            await _decideLock.WaitAsync();
        }
        try
        {
            _decideKind = kind;
            _byKind[kind] = _byKind.GetValueOrDefault(kind) + 1;
            _steps++;
            if (RunSearch.Active && RunSearch.Done) return -1;    // a finished rollout asks nothing more
            if (_forced != null && _forced.Count > 0)
            {
                // a rollout replaying the live run's answers, then the option being valued
                var f = _forced.Dequeue();
                if (f < 0 || f >= count) { RunSearch.ReplayError = true; f = Math.Clamp(f, 0, Math.Max(0, count - 1)); }
                _runLog.Add(f);
                if (_forced.Count == 0 && RunSearch.Active && !RunSearch.Job.Verify)
                {
                    if ((RunSearch.Job.ExpectKind != null && RunSearch.Job.ExpectKind != kind)
                        || (RunSearch.Job.ExpectN >= 0 && RunSearch.Job.ExpectN != count))
                    {
                        RunSearch.ReplayError = true;
                        Probe.Log($"  rollout replay landed on {kind}/{count}, expected " +
                                  $"{RunSearch.Job.ExpectKind}/{RunSearch.Job.ExpectN} (run {_run})");
                    }
                    RunSearch.StartWorld(RunManager.Instance.DebugOnlyGetState());
                }
                return f;
            }
            var msg = new Dictionary<string, object>
            {
                ["t"] = "decision", ["kind"] = kind, ["run"] = _run, ["step"] = _step++,
                ["obs"] = obs, ["legal"] = legal,
            };
            if (q != null) msg["q"] = q;           // search values, one per legal choice
            else if (!RunSearch.Active && _roomSave != null && RunSearch.Wants(kind, count))
                // run search: what a rollout needs to get back here (RunSearch)
                msg["ctx"] = new Dictionary<string, object> { ["save"] = _roomSave, ["log"] = _runLog.ToArray() };
            if (RunSearch.Active && RunSearch.Job.Verify) msg["verify"] = true;
            await Send(msg);
            var line = await _r.ReadLineAsync();
            if (line == null) return -1;
            if (RunSearch.Active && RunSearch.Job.Verify)
            {
                RunSearch.Done = true;              // landed: end the rollout here
                return -1;
            }
            var idx = Math.Clamp(ParseAction(line), 0, Math.Max(0, count - 1));
            _runLog.Add(idx);
            return idx;
        }
        finally { _decideLock.Release(); }
    }

    /// <summary>
    /// Run-queue mode: asks the server for the next run. Null = quit (or the server is gone).
    /// Goes through the decision lock: it is a round trip on the same stream.
    /// </summary>
    private static async Task<(int run, string seed, string character)?> NextRun()
    {
        await _decideLock.WaitAsync();
        try
        {
            await Send(new Dictionary<string, object> { ["t"] = "next_run" });
            var line = await _r.ReadLineAsync();
            if (line == null) return null;
            using var doc = System.Text.Json.JsonDocument.Parse(line);
            var root = doc.RootElement;
            if (root.TryGetProperty("quit", out var q) && q.GetBoolean()) return null;
            // per-run search switch (default on): a trainer's evaluation runs measure the bare policy
            FightSearch.RunEnabled = !root.TryGetProperty("search", out var se) || se.GetBoolean();
            RunSearch.Reset(null);
            if (root.TryGetProperty("rollout", out var ro) && ro.ValueKind == System.Text.Json.JsonValueKind.Object)
            {
                RunSearch.Reset(new RunSearch.Spec
                {
                    Restore = ro.GetProperty("restore").GetString(),
                    Forced = ro.GetProperty("forced").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                    World = ro.GetProperty("world").GetInt32(),
                    BossPlays = ro.TryGetProperty("boss_plays", out var bp) ? bp.GetInt32() : 4,
                    Verify = ro.TryGetProperty("verify", out var vf) && vf.GetBoolean(),
                    ExpectKind = ro.TryGetProperty("expect_kind", out var ek) ? ek.GetString() : null,
                    Teleport = ro.TryGetProperty("teleport", out var tp) && tp.GetBoolean(),
                    ExpectN = ro.TryGetProperty("expect_n", out var en) ? en.GetInt32() : -1,
                });
                FightSearch.RunEnabled = false;      // no fight search inside a rollout
            }
            return (root.GetProperty("run").GetInt32(), root.GetProperty("seed").GetString(),
                    root.GetProperty("character").GetString());
        }
        catch (Exception e)
        {
            Probe.Log($"  next_run: {e.Message}");
            return null;
        }
        finally { _decideLock.Release(); }
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
