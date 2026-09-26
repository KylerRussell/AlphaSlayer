using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.TestSupport;

namespace AlphaSlayer.Probe;

/// <summary>
/// One legal run-level (out-of-combat) action. Currently only map travel; rest-site,
/// shop and event options will extend the same shape.
/// </summary>
public sealed class RunAction
{
    public string Kind;          // "travel"
    public int PointIndex = -1;  // index into MapObs()["points"]
    public MapCoord Coord;
    public MapPointType PointType;

    public Dictionary<string, object> ToDict() => new()
    {
        ["kind"] = Kind,
        ["kind_idx"] = Kind switch { "travel" => 0, _ => 15 },
        ["point"] = PointIndex,
        ["col"] = Coord.col,
        ["row"] = Coord.row,
        ["point_type"] = PointType.ToString(),
        ["point_type_idx"] = (int)PointType,
    };
}

/// <summary>
/// Scene-free RUN harness: the out-of-combat half of the game.
///
/// CombatHarness stands a run up only far enough to make combats legal (one act, one map,
/// never travelled). This drives the real thing: pick a map point, enter whatever room the
/// engine rolls for it, resolve it, pick again, cross into the next act.
///
/// The engine's own model-layer entry point is RunManager.EnterMapCoord(coord), which does
/// the whole transition - records the visit, exits the current rooms, rolls RoomType from
/// MapPointType (a ? point is an RNG roll against the run's odds, so the room you get is NOT
/// a function of the map alone), constructs the room and enters it. Every node touch on that
/// path is null-conditional, so with no scene it is a pure state transition.
///
/// The node layer is therefore only consulted for DISPLAY. What we lose headless is nothing
/// that decides a run.
/// </summary>
public sealed class RunHarness
{
    public Player Player { get; private set; }
    public RunState RunState { get; private set; }
    public CharacterModel Character { get; private set; }
    public List<ActModel> Acts { get; private set; }

    public int ActIndex => RunState.CurrentActIndex;
    public int ActCount => RunState.Acts.Count;
    public bool PlayerDead => Player.Creature.IsDead;
    public AbstractRoom CurrentRoom => RunState.CurrentRoom;

    /// <summary>Starts a run and generates act 0's map. No scene, no save.</summary>
    public async Task StartRunAsync(string seed, int ascension, string characterId)
    {
        TestMode.IsOn = true;
        PreloadManager.Enabled = false;

        // SetUpNewSingleplayer refuses to overwrite a live run ("State is already set"), so a
        // process that plays more than one run has to tear the previous one down first.
        // CleanUp is the engine's own teardown: it resets the action queue and the card-select
        // stack, disposes every synchronizer and nulls State.
        if (RunManager.Instance.DebugOnlyGetState() != null)
        {
            try { RunManager.Instance.CleanUp(graceful: true); }
            catch (Exception e) { Probe.Log($"  (non-fatal) CleanUp: {e.Message}"); }
        }

        Character = ModelDb.AllCharacters.First(c => c.IsPlayable
            && (characterId == null || string.Equals(c.Id.Entry, characterId, StringComparison.OrdinalIgnoreCase)));
        Acts = ActModel.GetDefaultList().Select(a => a.ToMutable()).ToList();

        var unlocks = SaveManager.Instance.GenerateUnlockStateFromProgress();
        Player = Player.CreateForNewRun(Character, unlocks, 1uL);
        RunState = RunState.CreateForNewRun(
            new List<Player> { Player }, Acts, Array.Empty<ModifierModel>(),
            GameMode.Standard, ascension, seed);

        RunManager.Instance.SetUpNewSingleplayer(RunState, shouldSave: false);
        RunManager.Instance.Launch();
        await RunManager.Instance.FinalizeStartingRelics();

        // SetActInternal(0) resets the visited-coord list, resets the ? odds and generates
        // the map. EnterAct() additionally fades and enters a MapRoom; both are node-only,
        // so this is the whole of act entry that matters headless.
        await RunManager.Instance.SetActInternal(0);
    }

    // ---------- map ----------

    /// <summary>
    /// The points the player may travel to right now.
    ///
    /// Deliberately routed through MapTravel rather than reading Children directly: Winged
    /// Boots and the Flight modifier replace "my children" with "any point in the next row",
    /// and a policy that never saw the extra options could not learn to value the relic.
    /// </summary>
    public List<MapPoint> TravelablePoints()
    {
        var map = RunState.Map;
        if (map == null) return new List<MapPoint>();
        var current = RunState.CurrentMapPoint;
        // Nothing visited yet: the run starts on the act's single entry point.
        if (current == null)
            return map.StartingMapPoint == null ? new List<MapPoint>() : new List<MapPoint> { map.StartingMapPoint };
        return MapTravel.GetTravelablePointsFrom(RunState, current)
            .OrderBy(p => p.coord.col).ToList();
    }

    public List<RunAction> LegalRunActions()
    {
        var pts = AllPoints();
        var index = new Dictionary<MapCoord, int>();
        for (var i = 0; i < pts.Count; i++) index[pts[i].coord] = i;

        return TravelablePoints().Select(p => new RunAction
        {
            Kind = "travel",
            PointIndex = index.TryGetValue(p.coord, out var i) ? i : -1,
            Coord = p.coord,
            PointType = p.PointType,
        }).ToList();
    }

    /// <summary>
    /// Every point in the current act's map, in a stable order.
    ///
    /// ActMap.GetAllMapPoints() walks the grid only; the starting point and the boss (and
    /// the ascension-10 second boss) live OUTSIDE the grid and are resolved specially by
    /// GetPoint. Omitting them would hide the run's actual destination from the policy.
    /// </summary>
    public List<MapPoint> AllPoints()
    {
        var map = RunState.Map;
        if (map == null) return new List<MapPoint>();
        var pts = new List<MapPoint>();
        if (map.StartingMapPoint != null) pts.Add(map.StartingMapPoint);
        pts.AddRange(map.GetAllMapPoints());
        if (map.BossMapPoint != null) pts.Add(map.BossMapPoint);
        if (map.SecondBossMapPoint != null) pts.Add(map.SecondBossMapPoint);
        return pts.Distinct().OrderBy(p => p.coord.row).ThenBy(p => p.coord.col).ToList();
    }

    /// <summary>Travels to a point. Everything after this is the engine's own transition.</summary>
    public Task TravelAsync(MapCoord coord) => RunManager.Instance.EnterMapCoord(coord);

    /// <summary>True once the player has fought the act's boss and the next act is due.</summary>
    public bool AtActEnd()
    {
        var cur = RunState.CurrentMapPoint;
        return cur != null && cur.PointType == MapPointType.Boss && TravelablePoints().Count == 0;
    }

    /// <summary>
    /// Acts that count as a full run. Lower than ActCount shortens the run deliberately.
    ///
    /// A three-act run is won about once in five hundred attempts, so the win reward almost
    /// never fires and the policy is left optimising a proxy (floors) rather than the thing
    /// we want. Ending the run after act 1 raises the win rate into the tens of percent,
    /// which is a signal PPO can actually learn from; the cap is then raised once act 1 is
    /// reliable.
    /// </summary>
    public int ActCap { get; set; } = int.MaxValue;

    public int EffectiveActCount => Math.Min(ActCap, ActCount);

    public bool AtRunEnd() => AtActEnd() && ActIndex >= EffectiveActCount - 1;

    /// <summary>
    /// Crosses into the next act. EnterNextAct() fades and enters a MapRoom, both node-only,
    /// but at the LAST act it enters the Architect event room instead - which is the victory
    /// room, and is a real state change. Callers that only want the next map should check
    /// AtRunEnd() first.
    /// </summary>
    public Task NextActAsync() => RunManager.Instance.EnterAct(ActIndex + 1, doTransition: false);

    // ---------- observation ----------

    private string BossId()
    {
        try { return RunState.Act?.BossEncounter?.Id.Entry ?? ""; }
        catch { return ""; }
    }

    public Dictionary<string, object> MapObs()
    {
        var map = RunState.Map;
        var pts = AllPoints();
        var index = new Dictionary<MapCoord, int>();
        for (var i = 0; i < pts.Count; i++) index[pts[i].coord] = i;

        var visited = RunState.VisitedMapCoords.ToHashSet();
        var travelable = TravelablePoints().Select(p => p.coord).ToHashSet();
        var cur = RunState.CurrentMapCoord;

        return new Dictionary<string, object>
        {
            ["act"] = RunState.CurrentActIndex,
            ["act_id"] = RunState.Act?.Id.Entry ?? "",
            ["act_count"] = RunState.Acts.Count,
            ["act_floor"] = RunState.ActFloor,
            ["total_floor"] = RunState.TotalFloor,
            ["ascension"] = RunState.AscensionLevel,
            ["rows"] = map?.GetRowCount() ?? 0,
            ["cols"] = map?.GetColumnCount() ?? 0,
            ["cur_point"] = cur.HasValue && index.TryGetValue(cur.Value, out var ci) ? ci : -1,
            ["boss_point"] = map?.BossMapPoint != null && index.TryGetValue(map.BossMapPoint.coord, out var bi) ? bi : -1,
            ["points"] = pts.Select((p, i) => (object)new Dictionary<string, object>
            {
                ["i"] = i,
                ["col"] = p.coord.col,
                ["row"] = p.coord.row,
                ["type"] = p.PointType.ToString(),
                ["type_idx"] = (int)p.PointType,
                ["visited"] = visited.Contains(p.coord),
                ["travelable"] = travelable.Contains(p.coord),
                // Adjacency, so a policy can look further ahead than one row. Emitted as
                // indices into this same list rather than coords, to keep the encoder simple.
                ["children"] = p.Children.Select(c => index.TryGetValue(c.coord, out var j) ? j : -1)
                    .Where(j => j >= 0).OrderBy(j => j).ToArray(),
            }).ToArray(),
            ["player"] = PlayerObs(),
        };
    }

    /// <summary>
    /// Run-level player state. Distinct from the combat observation: what matters between
    /// fights is durable (hp, gold, the deck as a bag, relics, potions), not the transient
    /// per-turn state the combat model reads.
    /// </summary>
    public Dictionary<string, object> PlayerObs()
    {
        var deck = Player.Deck.Cards;
        var bag = new Dictionary<string, object>();
        foreach (var g in deck.GroupBy(c => Vocab.Get(Vocab.Cards, c.Id.Entry)))
            bag[g.Key.ToString()] = g.Count();

        return new Dictionary<string, object>
        {
            ["character"] = Character.Id.Entry,
            // Where in the run this is. Card reward, rest, event, shop, card_select, treasure
            // and potion_ooc send ONLY this dict, so without these they were decided blind to
            // the act and floor, and the progress-shaping potential read the floor as 0 on
            // every one of them (+/-1.2 reward per step late in a run).
            ["act"] = RunState.CurrentActIndex,
            ["act_floor"] = RunState.ActFloor,
            ["total_floor"] = RunState.TotalFloor,
            // The act boss is chosen when the act starts and shown on the map; players draft
            // and path for it. It was never sent.
            ["boss"] = BossId(),
            ["boss_idx"] = Vocab.Get(Vocab.Encounters, BossId()),
            ["hp"] = Player.Creature.CurrentHp,
            ["max_hp"] = Player.Creature.MaxHp,
            ["gold"] = Player.Gold,
            ["deck_size"] = deck.Count,
            ["deck_bag"] = bag,
            // The exact deck, card by card with upgrade level. deck_bag groups by id and so
            // loses upgrades, which is fine for the policy's own encoder but not for
            // reconstructing a deck to EVALUATE: a deck of upgraded Strikes is a different
            // deck from a deck of plain ones.
            // Every card with the same fields it has in a fight (Obs.CardInfo), plus "up" for
            // the existing readers.
            ["deck_cards"] = deck.Select(c =>
            {
                var d = Obs.CardInfo(c);
                d["up"] = c.CurrentUpgradeLevel;
                return (object)d;
            }).ToArray(),
            ["deck_upgrades"] = deck.Count(c => c.CurrentUpgradeLevel > 0),
            ["relics"] = Player.Relics.Select(r => (object)new Dictionary<string, object>
            {
                ["relic"] = r.Id.Entry,
                ["relic_idx"] = Vocab.Get(Vocab.Relics, r.Id.Entry),
                ["counter"] = Obs.RelicCounter(r),
                ["melted"] = r.IsMelted,
                ["vals"] = Obs.VarValues(r.DynamicVars),
            }).ToArray(),
            ["potions"] = Player.Potions.Select(p => (object)new Dictionary<string, object>
            {
                ["potion"] = p.Id.Entry,
                ["potion_idx"] = Vocab.Get(Vocab.Potions, p.Id.Entry),
                ["vals"] = Obs.VarValues(p.DynamicVars),
            }).ToArray(),
        };
    }
}
