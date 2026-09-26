#!/usr/bin/env python3
"""Applies the round-3 edits. Run ONLY when no training job is live: the running trainer
re-imports these modules if the auto-resume wrapper restarts it, and a half-applied edit
would load into a resumed run.

Three changes, all backward compatible:

 1. alphaslayer/deckeval.py  -- `act` (so exact decks can be played against act-2/3 encounter
    pools), optional per-spec `hp_frac`, and trajectory hooks (`on_terminal`, `with_idx`) so
    a trainer can collect PPO transitions from the same loop the benchmark uses.
 2. train_run.py             -- record `act` in harvested entry decks, so a harvest can be
    filtered to act 2 instead of guessed at from encounter names.
 3. probe/src/DeckServe.cs   -- honour `hp_frac` in the deck spec. deckserve heals to full
    every episode by design ("we are measuring the DECK"); for TRAINING that is the wrong
    distribution, since the run policy reaches act 2 at ~35% hp and how much risk a play is
    worth depends on remaining hp.

Requires a probe rebuild afterwards (probe/run.sh builds and installs the DLL).
"""
from __future__ import annotations

import subprocess
import sys

ROOT = "/home/kyler/Documents/AlphaSlayer"


def patch(path, pairs):
    with open(path) as fh:
        s = fh.read()
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected 1 occurrence, found {n}:\n  {old[:90]!r}")
        s = s.replace(old, new)
    with open(path, "w") as fh:
        fh.write(s)
    print(f"patched {path}")


def main():
    running = subprocess.run(["pgrep", "-f", "train_run.py|train_rl.py|train_combat_decks.py"],
                             capture_output=True, text=True).stdout.split()
    if running:
        raise SystemExit(f"refusing to patch: training still running (pids {running})")

    # ---- 1. deckeval.py -------------------------------------------------------------
    patch(f"{ROOT}/python/alphaslayer/deckeval.py", [
        ('                 batch_timeout=0.004, turn_cap=50, out_dir="/tmp/alphaslayer_deckeval"):\n',
         '                 batch_timeout=0.004, turn_cap=50, out_dir="/tmp/alphaslayer_deckeval",\n'
         '                 act=0):\n'),
        ('                 "--probe-encounters=mix"],\n',
         '                 "--probe-encounters=mix",\n'
         '                 # Which act\'s encounter pools to draw from. The deck is exact, but\n'
         '                 # "boss" means a DIFFERENT boss in act 2, and that is the whole\n'
         '                 # point of training on harvested act-2 decks.\n'
         '                 f"--probe-act={act}"],\n'),
        ('    def evaluate(self, specs, policy, progress_every=0):\n',
         '    def evaluate(self, specs, policy, progress_every=0, on_terminal=None,\n'
         '                 with_idx=False):\n'
         '        """``with_idx`` passes a third argument to ``policy``: the env index each\n'
         '        decision came from, which a trainer needs to group transitions into episodes.\n'
         '        ``on_terminal(env_idx, spec, room, msg)`` fires as each fight ends, which is\n'
         '        where an episode\'s reward becomes known."""\n'),
        # both payload sites gain the optional starting-hp fraction
        ('                    c.sock.sendall(_json_line({"deck": spec.cards, "upgrades": spec.upgrades,\n'
         '                                               "relics": spec.relics, "room": room}).encode())\n',
         '                    c.sock.sendall(_json_line(_spec_payload(spec, room)).encode())\n'),
        ('                                        "deck": spec.cards, "upgrades": spec.upgrades,\n'
         '                                        "relics": spec.relics, "room": room})).encode())\n',
         '                                        **_spec_payload(spec, room)})).encode())\n'),
        ('                                spec.hp_left[room] += msg["hp_end"] / max(1, msg["max_hp"])\n'
         '                                self.fights += 1\n',
         '                                spec.hp_left[room] += msg["hp_end"] / max(1, msg["max_hp"])\n'
         '                                self.fights += 1\n'
         '                                if on_terminal is not None:\n'
         '                                    on_terminal(conn.idx, spec, room, msg)\n'),
        ('            actions = policy([m["obs"] for _, m in pending], [m["legal"] for _, m in pending])\n',
         '            _obs = [m["obs"] for _, m in pending]\n'
         '            _legal = [m["legal"] for _, m in pending]\n'
         '            actions = (policy(_obs, _legal, [c.idx for c, _ in pending]) if with_idx\n'
         '                       else policy(_obs, _legal))\n'),
        ('@dataclass\nclass DeckSpec:\n',
         'def _spec_payload(spec, room):\n'
         '    """The deckserve episode spec. ``hp_frac`` is sent only when a spec carries one:\n'
         '    deckserve heals to full otherwise, which is right for MEASURING a deck and wrong\n'
         '    for training act-2 fights the policy actually enters at ~35% hp."""\n'
         '    payload = {"deck": spec.cards, "upgrades": spec.upgrades,\n'
         '               "relics": spec.relics, "room": room}\n'
         '    frac = getattr(spec, "hp_frac", None)\n'
         '    if frac is not None:\n'
         '        payload["hp_frac"] = round(float(frac), 4)\n'
         '    return payload\n'
         '\n'
         '\n'
         '@dataclass\nclass DeckSpec:\n'),
    ])

    # ---- 2. train_run.py: record the act on harvested decks -------------------------
    patch(f"{ROOT}/python/train_run.py", [
        ('                        "room": fr.room, "encounter": fr.encounter,\n',
         '                        "room": fr.room, "encounter": fr.encounter,\n'
         '                        # 0-based act, so a harvest can be filtered to act 2 rather\n'
         '                        # than inferred from encounter names.\n'
         '                        "act": int(pl.get("act", 0) or 0),\n'),
    ])

    # ---- 3. DeckServe.cs: honour hp_frac --------------------------------------------
    patch(f"{ROOT}/probe/src/DeckServe.cs", [
        # ParseDouble needs NumberStyles/CultureInfo, which this file does not yet import.
        ('using System.Diagnostics;\n', 'using System.Diagnostics;\nusing System.Globalization;\n'),
        ('            // Full health each time: we are measuring the DECK, and carrying damage between\n'
         '            // evaluations would confound it with whatever the previous fight happened to be.\n'
         '            try { harness.Player.Creature.HealInternal(harness.Player.Creature.MaxHp); } catch { }\n',
         '            // Full health each time: we are measuring the DECK, and carrying damage between\n'
         '            // evaluations would confound it with whatever the previous fight happened to be.\n'
         '            try { harness.Player.Creature.HealInternal(harness.Player.Creature.MaxHp); } catch { }\n'
         '            // ...unless the caller asked for a specific fraction. Measuring a deck wants\n'
         '            // full hp; TRAINING act-2 fights does not, because the run policy arrives in\n'
         '            // act 2 at ~35% hp and how much risk a play is worth depends on what is left.\n'
         '            var hpFrac = ParseDouble(spec, "hp_frac");\n'
         '            if (hpFrac > 0 && hpFrac < 1)\n'
         '            {\n'
         '                try\n'
         '                {\n'
         '                    var want = Math.Max(1, (int)Math.Round(harness.Player.Creature.MaxHp * hpFrac));\n'
         '                    harness.Player.Creature.SetCurrentHpInternal(want);\n'
         '                }\n'
         '                catch (Exception e) { Probe.Log($"  (non-fatal) set hp_frac: {e.Message}"); }\n'
         '            }\n'),
        ('    private static string ParseStr(string line, string key)\n',
         '    private static double ParseDouble(string line, string key)\n'
         '    {\n'
         '        var i = line.IndexOf($"\\"{key}\\"", StringComparison.Ordinal);\n'
         '        if (i < 0) return 0;\n'
         '        var c = line.IndexOf(\':\', i);\n'
         '        if (c < 0) return 0;\n'
         '        var end = c + 1;\n'
         '        while (end < line.Length && (char.IsDigit(line[end]) || line[end] == \'.\'\n'
         '                                     || line[end] == \'-\' || line[end] == \'+\'\n'
         '                                     || line[end] == \'e\' || line[end] == \'E\'\n'
         '                                     || line[end] == \' \')) end++;\n'
         '        return double.TryParse(line.Substring(c + 1, end - c - 1).Trim(),\n'
         '                               NumberStyles.Float, CultureInfo.InvariantCulture,\n'
         '                               out var v) ? v : 0;\n'
         '    }\n'
         '\n'
         '    private static string ParseStr(string line, string key)\n'),
    ])
    print("\nall patches applied. next: rebuild the probe DLL, then verify:")
    print("  cd probe && dotnet build AlphaSlayerProbe.csproj -c Release -v quiet --nologo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
