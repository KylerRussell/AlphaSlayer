using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace AlphaSlayer.Probe;

/// <summary>
/// Scene-free combat harness: start a run and enter/leave combats with no node layer.
///
/// Replaces RunManager.EnterRoomDebug / NRun.Create, which instantiate Godot scenes and
/// segfault headless. Replicates CombatRoom.EnterInternal + StartCombat instead; every node
/// touch on that path is null-conditional, so it is safe with no scene at all.
/// </summary>
public sealed class CombatHarness
{
    public Player Player { get; private set; }
    public RunState RunState { get; private set; }
    public List<EncounterModel> Encounters { get; private set; }
    public CharacterModel Character { get; private set; }
    public List<ActModel> Acts { get; private set; }
    private readonly List<RelicModel> _injectedRelics = new();
    private readonly List<RelicModel> _startingRelics = new();
    private bool _rerollDeck;
    private double[] _mix;
    private List<EncounterModel> _mixBoss = new(), _mixElite = new(), _mixRegular = new();

    /// <summary>Starts a run with no scene. Must be called after GameStartupComplete.</summary>
    public async Task StartRunAsync(string seed, int ascension, string characterId)
    {
        // TestMode is the node-layer bypass: every visual path in the gameplay commands is
        // gated on TestMode.IsOff (e.g. CardPileCmd -> CreateCardNodeAndUpdateVisuals, which
        // instantiates an NCard scene and dies under the headless dummy renderer). It also
        // short-circuits CombatStateTracker.NotifyCombatStateChanged, whose "deferred"
        // notification stops deferring with no scene and re-enters itself.
        TestMode.IsOn = true;
        PreloadManager.Enabled = false;

        Character = ModelDb.AllCharacters.First(c => c.IsPlayable
            && (characterId == null || string.Equals(c.Id.Entry, characterId, StringComparison.OrdinalIgnoreCase)));
        Acts = ActModel.GetDefaultList().Select(a => a.ToMutable()).ToList();

        var unlocks = SaveManager.Instance.GenerateUnlockStateFromProgress();
        Player = Player.CreateForNewRun(Character, unlocks, 1uL);
        RunState = RunState.CreateForNewRun(
            new List<Player> { Player }, Acts, Array.Empty<ModifierModel>(),
            GameMode.Standard, ascension, seed);

        // --probe-inject-cards lets a test force specific cards into the deck. Many effects
        // (card selection, random generation) never appear in a starting deck, so without this
        // whole code paths go unexercised by random rollouts.
        var inject = Probe.ArgValue("probe-inject-cards");
        if (!string.IsNullOrWhiteSpace(inject))
        {
            var copies = Probe.ArgInt("probe-inject-copies", 4);
            foreach (var id in inject.Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                var canonical = ModelDb.AllCards.FirstOrDefault(
                    c => string.Equals(c.Id.Entry, id.Trim(), StringComparison.OrdinalIgnoreCase));
                if (canonical == null) { Probe.Log($"  inject: unknown card '{id}'"); continue; }
                for (var i = 0; i < copies; i++)
                {
                    var card = canonical.ToMutable();
                    // Owner MUST be set before the card enters the deck. Without it the card
                    // has no player, and combat setup dies walking hook listeners during the
                    // opening shuffle - which looked like a card-select bug but was not.
                    card.Owner = Player;
                    Player.Deck.AddInternal(card);
                }
                Probe.Log($"  injected {copies}x {canonical.Id.Entry}");
            }
        }

        RunManager.Instance.SetUpNewSingleplayer(RunState, shouldSave: false);
        RunManager.Instance.Launch();

        // --probe-inject-relics: grant relics up front, AFTER Launch() so the player has a
        // real RunState (pickup effects read Owner.RunState.Rng).
        //
        // The motivating case is PANDORAS_BOX, whose AfterObtained() transforms every basic
        // Strike and Defend into a random card. A starter deck is only two distinct cards, so
        // the combat policy currently learns almost nothing about card interactions; this puts
        // a wide, varied deck in front of it without simulating whole runs.
        // Starting relics exist on the player but their AfterObtained() effects never ran,
        // because nothing called this. Several characters' starting relics change how the
        // fight should be played, so this was silently altering the game.
        try { await RunManager.Instance.FinalizeStartingRelics(); }
        catch (Exception e) { Probe.Log($"  (non-fatal) FinalizeStartingRelics: {e.Message}"); }

        _startingRelics.AddRange(Player.Relics);
        _rerollDeck = Probe.HasFlag("probe-reroll-deck");
        var relics = Probe.ArgValue("probe-inject-relics");
        if (!string.IsNullOrWhiteSpace(relics))
        {
            foreach (var id in relics.Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                var canonical = ModelDb.AllRelics.FirstOrDefault(
                    r => string.Equals(r.Id.Entry, id.Trim(), StringComparison.OrdinalIgnoreCase));
                if (canonical == null) { Probe.Log($"  relic: unknown '{id}'"); continue; }
                var relic = canonical.ToMutable();
                Player.AddRelicInternal(relic);   // sets relic.Owner
                try { await relic.AfterObtained(); }
                catch (Exception e) { Probe.Log($"  relic {id} AfterObtained: {e.Message}"); }
                _injectedRelics.Add(relic);
                Probe.Log($"  granted relic {canonical.Id.Entry}");
            }
            var deck = Player.Deck.Cards.Select(c => c.Id.Entry).ToList();
            Probe.Log($"  deck: {deck.Count} cards, {deck.Distinct().Count()} distinct " +
                      $"[{string.Join(",", deck.Distinct().Take(10))}]");
        }


        // A map and an act must exist. EndCombatInternal reads runState.Map.BossMapPoint and
        // casts runState.CurrentRoom to CombatRoom unconditionally, so a run without them
        // faults on the way OUT of a combat even though the way in looks fine.
        try { await RunManager.Instance.SetActInternal(0); }
        catch (Exception e) { Probe.Log($"  (non-fatal) SetActInternal: {e.Message}"); }
        try { await RunManager.Instance.GenerateMap(); }
        catch (Exception e) { Probe.Log($"  (non-fatal) GenerateMap: {e.Message}"); }
        Probe.Log($"  map generated: {(RunState.Map != null ? "yes" : "NO")}");

        // Encounter pool is selectable: elites and bosses carry richer power/debuff sets, which
        // is where extra mid-turn RNG would show up if it exists.
        var pool = (Probe.ArgValue("probe-encounters") ?? "regular").ToLowerInvariant();
        var act = Acts[Math.Clamp(Probe.ArgInt("probe-act", 0), 0, Acts.Count - 1)];
        Encounters = pool switch
        {
            "elite" => act.AllEliteEncounters.ToList(),
            "boss" => act.AllBossEncounters.ToList(),
            "all" => act.AllEncounters.ToList(),
            // Bosses plus elites: the hardest opponents the character can face in act 1.
            // Training here (with a Pandora's deck) targets the fights that actually decide
            // runs, rather than normals the policy already wins ~100% of the time.
            "hard" => act.AllBossEncounters.Concat(act.AllEliteEncounters).ToList(),
            "mix" => act.AllEncounters.ToList(),   // replaced below by the weighted pools
            _ => act.AllRegularEncounters.ToList(),
        };
        // Weighted mix, e.g. --probe-encounters mix --probe-mix 0.5,0.25,0.25
        // (boss, elite, regular). Training on one difficulty teaches only that difficulty.
        if (pool == "mix")
        {
            var w = (Probe.ArgValue("probe-mix") ?? "0.5,0.25,0.25")
                .Split(',').Select(x => double.TryParse(x, out var v) ? v : 0).ToArray();
            var total = w.Sum();
            _mix = w.Select(x => x / (total <= 0 ? 1 : total)).ToArray();
            _mixBoss = act.AllBossEncounters.ToList();
            _mixElite = act.AllEliteEncounters.ToList();
            _mixRegular = act.AllRegularEncounters.ToList();
            Encounters = _mixBoss.Concat(_mixElite).Concat(_mixRegular).ToList();
            Probe.Log($"  encounter mix boss/elite/regular = {_mix[0]:P0}/{_mix[1]:P0}/{_mix[2]:P0} " +
                      $"({_mixBoss.Count}/{_mixElite.Count}/{_mixRegular.Count} encounters)");
        }

        var exclude = Probe.ArgValue("probe-exclude-encounters");
        if (!string.IsNullOrWhiteSpace(exclude))
        {
            var names = exclude.Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(x => x.Trim()).ToHashSet(StringComparer.OrdinalIgnoreCase);
            var before = Encounters.Count;
            Encounters = Encounters.Where(e => !names.Contains(e.Id.Entry)).ToList();
            Probe.Log($"  excluded {before - Encounters.Count} encounter(s): {exclude}");
        }
        if (Encounters.Count == 0) throw new InvalidOperationException($"no {pool} encounters in act {act.Id.Entry}");
    }

    /// <summary>
    /// Enters one combat through the engine's OWN room lifecycle.
    ///
    /// The previous version hand-rolled CombatRoom.StartCombat (build a CombatState, add
    /// creatures, SetUpCombat, AfterCombatRoomLoaded) and never entered a room. That is fine
    /// going in, but combat END is not: CombatManager.EndCombatInternal opens with
    ///
    ///     CombatRoom room = (CombatRoom)runState.CurrentRoom;
    ///     ... room.OnCombatEnded(); runState.Map.SecondBossMapPoint ...
    ///
    /// With no room and no map those dereference garbage. That is the Defect crash - the
    /// faulting RIP resolves, via the JIT perf map, to EndCombatInternal. Entering a real
    /// room gives the end-of-combat path the context it has always required.
    /// </summary>
    /// <summary>
    /// Rebuilds the starting deck and re-applies injected relic pickup effects.
    ///
    /// StartRunAsync runs ONCE per process, so without this every episode in a process shares
    /// one Pandora's roll - the model sees a single random deck a dozen times instead of a
    /// dozen decks. Re-rolling per episode is what actually exposes it to the card pool.
    /// Only INJECTED relics are re-fired; re-running starting-relic effects would stack
    /// things like max-HP gains every episode.
    /// </summary>
    public async Task RerollDeckAsync()
    {
        if (!_rerollDeck || _injectedRelics.Count == 0) return;
        try
        {
            // Reset relics to (starting + injected) before re-firing. Without this a relic
            // whose pickup effect GRANTS relics - Large Capsule gives 2 - would add another
            // pair every episode, so a 20-episode process would end up with 40 relics and a
            // difficulty curve that has nothing to do with the game. Resetting also means each
            // episode draws a FRESH random pair, which is the point of the exercise.
            foreach (var r in Player.Relics.ToList())
            {
                if (_startingRelics.Contains(r) || _injectedRelics.Contains(r)) continue;
                Player.RemoveRelicInternal(r, silent: true);
            }

            Player.Deck.Clear(silent: true);
            foreach (var c in Character.StartingDeck)
            {
                var card = c.ToMutable();
                card.Owner = Player;
                Player.Deck.AddInternal(card, -1, silent: true);
            }
            foreach (var relic in _injectedRelics) await relic.AfterObtained();
            if (Probe.HasFlag("probe-log-encounters"))
            {
                var d = Player.Deck.Cards.Select(c => c.Id.Entry).Distinct().Take(6);
                var rl = Player.Relics.Select(r => r.Id.Entry);
                Probe.Log($"    deck reroll -> [{string.Join(",", d)}] relics=[{string.Join(",", rl)}]");
            }
        }
        catch (Exception e) { Probe.Log($"  (non-fatal) deck reroll: {e.Message}"); }
    }

    /// <summary>Encounter pools by room type, for callers that choose the type per episode.</summary>
    public List<EncounterModel> RegularEncounters => _mixRegular.Count > 0 ? _mixRegular : Encounters;
    public List<EncounterModel> EliteEncounters => _mixElite;
    public List<EncounterModel> BossEncounters => _mixBoss;

    /// <summary>
    /// Replaces the deck and relics wholesale.
    ///
    /// This is what makes a deck EVALUABLE: to ask "how would this deck fare against an
    /// elite" we have to be able to put an arbitrary deck in front of the combat model,
    /// rather than only the starting deck plus whatever a relic rolled.
    /// </summary>
    public async Task SetDeckAsync(IReadOnlyList<string> cardIds, IReadOnlyList<int> upgrades,
                                   IReadOnlyList<string> relicIds)
    {
        // Relic pickup effects can raise a card-selection prompt (Small Capsule and friends
        // grant relics, which in turn can ask the player to choose). This runs OUTSIDE the
        // agent's decision loop, so nothing would ever answer that prompt and the process
        // would deadlock -- which is exactly what happened on real run decks, while the
        // hand-written test decks that carry only a starting relic passed.
        var priorAuto = CardSelect.AutoAnswer;
        CardSelect.AutoAnswer = true;

        Unknown.Clear();
        Player.Deck.Clear(silent: true);
        for (var i = 0; i < cardIds.Count; i++)
        {
            var canonical = ModelDb.AllCards.FirstOrDefault(
                c => string.Equals(c.Id.Entry, cardIds[i], StringComparison.OrdinalIgnoreCase));
            // Loudly, not silently: a mistyped id used to be skipped without a word, which
            // made a nearly-empty deck look like a legitimate weak one.
            if (canonical == null) { Unknown.Add(cardIds[i]); continue; }
            var card = canonical.ToMutable();
            card.Owner = Player;   // without an owner, combat setup dies during the shuffle
            var up = i < upgrades.Count ? upgrades[i] : 0;
            for (var u = 0; u < up; u++)
            {
                try { card.UpgradeInternal(); } catch { break; }
            }
            Player.Deck.AddInternal(card, -1, silent: true);
        }

        if (relicIds != null)
        {
            foreach (var r in Player.Relics.ToList())
            {
                if (_startingRelics.Contains(r)) continue;
                try { Player.RemoveRelicInternal(r, silent: true); } catch { }
            }
            foreach (var id in relicIds)
            {
                if (_startingRelics.Any(sr => string.Equals(sr.Id.Entry, id, StringComparison.OrdinalIgnoreCase)))
                    continue;
                var canonical = ModelDb.AllRelics.FirstOrDefault(
                    x => string.Equals(x.Id.Entry, id, StringComparison.OrdinalIgnoreCase));
                if (canonical == null) continue;
                var relic = canonical.ToMutable();
                // Deliberately NOT firing AfterObtained.
                //
                // AfterObtained is the one-time PICKUP effect: add cards to the deck, raise
                // max hp, grant further relics. For a deck harvested from a real run that
                // effect has already happened - its cards are in the deck we were handed - so
                // replaying it would double-apply, inflating the very deck we are trying to
                // measure. The relic's combat behaviour comes from being in Player.Relics,
                // which AddRelicInternal does on its own.
                //
                // It is also what made this crash: pickup effects on relics like Small Capsule
                // grant more relics and can reach the node layer, and every env died at once
                // on the first deck that carried one.
                Player.AddRelicInternal(relic);
            }
        }

        CardSelect.AutoAnswer = priorAuto;
        CardSelect.Reset();
    }

    /// <summary>Card ids the last SetDeckAsync could not resolve. Empty is the healthy case.</summary>
    public static readonly List<string> Unknown = new();

    /// <summary>Picks the next encounter, honouring a weighted mix if one was requested.</summary>
    public EncounterModel NextEncounter(int episodeIndex, Random rng)
    {
        if (_mix == null) return Encounters[(episodeIndex + Probe.ArgInt("probe-encounter-offset", 0)) % Encounters.Count];
        var r = rng.NextDouble();
        var pool = r < _mix[0] ? _mixBoss : (r < _mix[0] + _mix[1] ? _mixElite : _mixRegular);
        if (pool.Count == 0) pool = Encounters;
        return pool[rng.Next(pool.Count)];
    }

    public async Task<EncounterModel> BeginCombatAsync(EncounterModel canonicalEncounter)
    {
        var enc = canonicalEncounter.ToMutable();
        await RunManager.Instance.EnterRoom(new CombatRoom(enc, RunState));
        RunManager.Instance.ActionExecutor.Unpause();
        return enc;
    }

    /// <summary>
    /// Tears the combat down and restores the player for the next one.
    ///
    /// IMPORTANT: waits for the engine to finish its OWN end-of-combat work first. Detecting
    /// "enemies cleared" only means the outcome is decided; the engine is still running
    /// AfterCombatEnd hooks, which walk the creature and model lists. Reset(graceful) removes
    /// creatures and clears history, so calling it during that walk mutates the collections
    /// being iterated. Characters with long hook walks (Defect, via orbs and powers) collide
    /// far more often than short ones - which is why this looked character-specific.
    /// </summary>
    public async Task EndCombatAsync(bool healToFull)
    {
        // IsInProgress is already false while AfterCombatEnd hooks are still walking, so it is
        // useless as a settle signal. Wait a fixed span instead and let the engine finish.
        var settleMs = Probe.ArgInt("probe-teardown-settle-ms", 0);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < settleMs) await Suspend();
        EndCombat(healToFull);
    }

    public void EndCombat(bool healToFull)
    {
        try { CombatManager.Instance.Reset(graceful: true); }
        catch (Exception e) { Probe.Log($"  (non-fatal) CombatManager.Reset: {e.Message}"); }
        if (healToFull)
        {
            try { Player.Creature.HealInternal(Player.Creature.MaxHp); }
            catch (Exception e) { Probe.Log($"  (non-fatal) heal: {e.Message}"); }
        }
    }

    public CombatState State => CombatManager.Instance.DebugOnlyGetState();

    /// <summary>
    /// True only once combat has really finished. IsOverOrEnding is ALSO true during setup,
    /// because it is (IsEnding || !IsInProgress) and IsInProgress stays false until
    /// StartCombatInternal flips it; IsStarting separates the two.
    /// </summary>
    public static bool Finished() =>
        CombatManager.Instance.IsOverOrEnding && !CombatManager.Instance.IsStarting;

    /// <summary>A real suspension that lets the engine's continuations drain.</summary>
    public static Task Suspend() => Task.Delay(1);
}
