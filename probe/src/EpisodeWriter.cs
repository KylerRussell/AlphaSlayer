using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Enchantments;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;

namespace AlphaSlayer.Probe;

/// <summary>
/// Packed binary episode writer (gzip-framed), replacing the ~2.2KB/step JSONL.
///
/// Two decisions drive the layout:
/// - RAGGED, not padded. Fixed-width padding for worst-case hand/enemy/legal counts came out
///   the same size as the JSON it replaced; typical steps use a fraction of the caps.
/// - GZIP. The data is highly repetitive (card ids, deck composition barely moves within a
///   combat), so the stream compresses hard for almost no CPU.
///
/// All integers little-endian. Counts are u8 unless noted; ids and amounts are i16, which
/// covers every vocab (max 596) and every in-combat magnitude with room to spare.
///
/// Layout is versioned in the header - bump FormatVersion on any change, and the Python
/// reader refuses a version it does not know.
/// </summary>
public sealed class EpisodeWriter : IDisposable
{
    public const uint Magic = 0x454C5341; // "ASLE"
    public const ushort FormatVersion = 3; // v3: relics carry (idx, counter, melted)

    public const byte RecStep = 0;
    public const byte RecTerminal = 1;

    /// <summary>Outcome string -> stable code. Keep in sync with the Python reader.</summary>
    public static readonly string[] Outcomes =
    {
        "enemies_cleared", "player_dead", "turn_cap", "combat_ended",
        "no_legal_actions", "decision_timeout", "turn_timeout", "unknown",
    };

    private readonly FileStream _file;
    private readonly GZipStream _gz;
    private readonly BinaryWriter _w;
    public long Steps, Terminals;

    public EpisodeWriter(string path, int vocabHash)
    {
        _file = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 16);
        _gz = new GZipStream(_file, CompressionLevel.Fastest, leaveOpen: false);
        _w = new BinaryWriter(_gz, Encoding.UTF8, leaveOpen: false);
        _w.Write(Magic);
        _w.Write(FormatVersion);
        _w.Write(vocabHash);
    }

    public static byte OutcomeCode(string s)
    {
        var i = Array.IndexOf(Outcomes, s);
        return (byte)(i < 0 ? Outcomes.Length - 1 : i);
    }

    private void I16(int v) => _w.Write((short)Math.Clamp(v, short.MinValue, short.MaxValue));
    private void U8(int v) => _w.Write((byte)Math.Clamp(v, 0, 255));

    /// <summary>
    /// Writes everything about a decision point EXCEPT the trailing hp_delta.
    ///
    /// Split in two on purpose: the observation must be the state the agent actually saw when
    /// it chose, so this has to run BEFORE the action is applied - while hp_delta is only
    /// known afterwards. hp_delta is the last field in the record, and the stream is
    /// single-threaded and sequential, so EndStep can simply append it.
    /// </summary>
    public void BeginStep(int ep, int t, Player player, CombatState combat,
                          List<LegalAction> legal, int actionIdx)
    {
        var pcs = player.PlayerCombatState;
        _w.Write(RecStep);
        _w.Write(ep);
        _w.Write((ushort)t);

        // --- globals (fixed 12 x i16) ---
        I16(pcs?.TurnNumber ?? 0);
        I16(combat.RoundNumber);
        I16((int)(pcs?.Phase ?? PlayerTurnPhase.None));
        I16((int)combat.CurrentSide);
        I16(player.Creature.CurrentHp);
        I16(player.Creature.MaxHp);
        I16(player.Creature.Block);
        I16(pcs?.Energy ?? 0);
        I16(pcs?.MaxEnergy ?? 0);
        I16(pcs?.Stars ?? 0);
        I16(player.Gold);
        I16(pcs?.DrawPile.Cards.Count ?? 0);

        // --- player powers / relics / potions ---
        WritePowers(player.Creature);
        var relics = player.Relics.ToList();
        U8(relics.Count);
        foreach (var r in relics)
        {
            I16(Vocab.Get(Vocab.Relics, r.Id.Entry));
            I16(Obs.RelicCounter(r));
            I16(r.IsMelted ? 1 : 0);
        }
        var potions = player.Potions.ToList();
        U8(potions.Count);
        foreach (var p in potions) I16(Vocab.Get(Vocab.Potions, p.Id.Entry));

        // --- hand: 13 x i16 per card ---
        var hand = pcs?.Hand.Cards.ToList() ?? new List<CardModel>();
        U8(hand.Count);
        foreach (var c in hand)
        {
            var e = c.Enchantment;
            I16(Vocab.Get(Vocab.Cards, c.Id.Entry));
            I16(c.CurrentUpgradeLevel);
            I16(e == null ? Vocab.None : Vocab.Get(Vocab.Enchantments, e.Id.Entry));
            I16(e?.Amount ?? 0);
            I16(e != null && e.Status == EnchantmentStatus.Disabled ? 1 : 0);
            I16(Try(() => c.EnergyCost.GetResolved(), -1));
            I16(Try(() => c.EnergyCost.CostsX, false) ? 1 : 0);
            I16(Try(() => c.CurrentStarCost, -1));
            I16((int)c.Type);
            I16((int)c.Rarity);
            I16((int)c.TargetType);
            I16(Try(() => c.CanPlay(), false) ? 1 : 0);
            I16(0); // reserved, keeps the token width stable across format tweaks
        }

        // --- piles as bags of (card_idx, count) ---
        WriteBag(pcs?.DrawPile.Cards);
        WriteBag(pcs?.DiscardPile.Cards);
        WriteBag(pcs?.ExhaustPile.Cards);

        // --- enemies ---
        var enemies = combat.Enemies.ToList();
        U8(enemies.Count);
        foreach (var e in enemies)
        {
            I16(Vocab.Get(Vocab.Monsters, e.Monster?.Id.Entry));
            I16(e.CurrentHp);
            I16(e.MaxHp);
            I16(e.Block);
            I16(e.IsAlive ? 1 : 0);
            I16(e.IsHittable ? 1 : 0);
            WritePowers(e);
            WriteIntents(e);
        }

        // --- legal actions (u16 count: a wide hand with many targets can exceed 255) ---
        _w.Write((ushort)legal.Count);
        foreach (var a in legal)
        {
            U8(a.Kind switch { "play" => 0, "end_turn" => 1, "select_card" => 2, _ => 3 });
            _w.Write((sbyte)Math.Clamp(a.HandIndex, -1, 127));
            _w.Write((sbyte)Math.Clamp(a.TargetIndex, -1, 127));
            I16(Vocab.Get(Vocab.Cards, a.CardId));
        }

        _w.Write((ushort)Math.Max(0, actionIdx));
    }

    /// <summary>Appends the trailing hp_delta and closes the record opened by BeginStep.</summary>
    public void EndStep(int hpDelta)
    {
        I16(hpDelta);
        Steps++;
    }

    public void WriteTerminal(int ep, string outcome, bool won, int turns, int steps,
                             int hpStart, int hpEnd, float reward)
    {
        _w.Write(RecTerminal);
        _w.Write(ep);
        _w.Write(OutcomeCode(outcome));
        _w.Write((byte)(won ? 1 : 0));
        _w.Write((ushort)turns);
        _w.Write((ushort)steps);
        I16(hpStart);
        I16(hpEnd);
        _w.Write(reward);
        Terminals++;
    }

    private void WritePowers(Creature c)
    {
        var powers = c.Powers.ToList();
        U8(powers.Count);
        foreach (var p in powers)
        {
            I16(Vocab.Get(Vocab.Powers, p.Id.Entry));
            I16(Try(() => p.Amount, 0));
        }
    }

    private void WriteIntents(Creature e)
    {
        var move = e.Monster?.NextMove;
        var intents = move?.Intents?.ToList() ?? new List<AbstractIntent>();
        var targets = new[] { e };
        U8(intents.Count);
        foreach (var i in intents)
        {
            I16((int)i.IntentType);
            I16(i is AttackIntent a ? Try(() => a.GetTotalDamage(targets, e), 0) : 0);
            I16(i is AttackIntent a2 ? Try(() => a2.Repeats, 0) : 0);
        }
    }

    private void WriteBag(IReadOnlyList<CardModel> cards)
    {
        if (cards == null) { U8(0); return; }
        var groups = cards.GroupBy(c => Vocab.Get(Vocab.Cards, c.Id.Entry)).ToList();
        U8(groups.Count);
        foreach (var g in groups) { I16(g.Key); I16(g.Count()); }
    }

    private static T Try<T>(Func<T> f, T fallback)
    {
        try { return f(); } catch { return fallback; }
    }

    public void Dispose()
    {
        try { _w.Flush(); _w.Dispose(); } catch { }
        try { _gz.Dispose(); } catch { }
        try { _file.Dispose(); } catch { }
    }
}
