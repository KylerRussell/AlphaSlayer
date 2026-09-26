"""Live delivery test for the combat observation additions (orbs, allies, afflictions, card
numbers, room context).

Each feature is exercised in a scenario where it MUST appear. "It never showed up" is a
failure, not a pass:

  * Ironclad, act-1 hallway fights: every Strike shows damage > 0 and every Defend block > 0.
    A Strike aimed at a Vulnerable enemy previews MORE damage in its action than in its hand
    token, which is untargeted. That proves the preview really uses the action's target.
  * Defect: orb slots > 0, channelled orbs appear with a known id, and a Lightning orb's
    evoke value is at least its passive value.
  * Necrobinder: Osty appears as an ally with a known monster id and positive HP.
  * Queen (act-3 boss pool, the other bosses excluded): cards carry a real affliction id
    (Bound). The observation says room Boss and encounter QUEEN_BOSS; the enemy move ids are
    Queen's own and her first move is PUPPET_STRINGS_MOVE.
  * M1 additions: Offering (injected) shows the Exhaust keyword and HP-loss 6 / energy 2;
    Royalties (injected) shows gold 30. Piles are sorted multisets whose counts add up to the
    pile sizes. Potions on the belt carry numbers. Aeonglass's status move reports how many
    status cards it adds.

Random policy: these check the observation, not play quality.

Run from python/:
    XDG_DATA_HOME="$(../headless_home.sh)" PYTHONPATH=. .venv7/bin/python tests/test_combat_obs.py
"""
from __future__ import annotations

import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaslayer.vecenv import VecSpireEnv

FAILS = []
SLOTS = ("damage", "block", "repeat", "cards", "power", "energy", "hp_loss", "heal",
         "stars", "summon", "forge", "osty_damage", "gold", "max_hp")
KW_EXHAUST = 1 << 1          # CardKeyword.Exhaust = 1


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def play(character, episodes=6, envs=3, **kw):
    """Plays random-policy fights; returns every (obs, legal) the policy was shown."""
    seen = []
    rng = random.Random(1234)

    def policy(obs, legal, idxs):
        out = []
        for o, l in zip(obs, legal):
            seen.append((o, l))
            # Prefer playing a card over ending the turn, so fights get long enough for
            # orbs, debuffs and afflictions to appear.
            plays = [i for i, a in enumerate(l) if a["kind"] != "end_turn"]
            out.append(rng.choice(plays) if plays and rng.random() < 0.85 else
                       rng.randrange(len(l)))
        return out

    with VecSpireEnv(n_envs=envs, character=character, episodes_per_env=episodes,
                     seed=f"OBS_{character}_", **kw) as env:
        env.run(policy)
    return seen


def V(x):
    """A token's card numbers, all zero when the probe did not send them."""
    return x.get("vals") or {s: 0 for s in SLOTS}


def has_power(creature, needle):
    return any(needle in (p.get("power") or "") and (p.get("amount") or 0) > 0
               for p in creature.get("powers") or ())


def scenario_ironclad():
    print("\n== Ironclad, act-1 hallway fights: card numbers and per-target previews")
    seen = play("IRONCLAD")
    tokens = [c for o, _ in seen for c in o["hand"]]
    check("decisions were collected", len(seen) > 50, f"{len(seen)} decisions")
    check("every hand card carries vals with all slots + affliction fields",
          all(isinstance(c.get("vals"), dict) and all(s in c["vals"] for s in SLOTS)
              and "affl_idx" in c and "affl_amount" in c for c in tokens),
          f"{len(tokens)} hand cards")
    strikes = [c for c in tokens if c["card"].startswith("STRIKE")]
    defends = [c for c in tokens if c["card"].startswith("DEFEND")]
    check("every Strike previews damage > 0",
          strikes and all(V(c)["damage"] > 0 for c in strikes),
          f"{len(strikes)} strikes, damage values {sorted({V(c)['damage'] for c in strikes})}")
    check("every Defend previews block > 0",
          defends and all(V(c)["block"] > 0 for c in defends),
          f"{len(defends)} defends, block values {sorted({V(c)['block'] for c in defends})}")

    vul, plain = [], []
    for o, legal in seen:
        for a in legal:
            if a["kind"] != "play" or not a["card"].startswith("STRIKE") or a["target"] < 0:
                continue
            hand_dmg = V(o["hand"][a["hand"]])["damage"]
            act_dmg = V(a)["damage"]
            (vul if has_power(o["enemies"][a["target"]], "VULNERABLE") else plain).append(
                (hand_dmg, act_dmg))
    check("targeted Strikes at Vulnerable enemies were exercised", len(vul) >= 5,
          f"{len(vul)} cases" + ("" if len(vul) >= 5 else ": INCONCLUSIVE"))
    check("a Strike at a Vulnerable enemy previews more than its untargeted hand value",
          vul and all(act > hand for hand, act in vul),
          f"e.g. {vul[:3]}")
    check("a Strike at a non-Vulnerable enemy previews its hand value",
          plain and all(act == hand for hand, act in plain),
          f"{sum(act != hand for hand, act in plain)}/{len(plain)} differ")
    rooms = collections.Counter(o.get("room_type") for o, _ in seen)
    check("room_type is reported and is Monster", set(rooms) == {"Monster"}, str(dict(rooms)))
    check("encounter id is reported", all(o.get("encounter") for o, _ in seen))
    check_piles(seen)
    live = [e for o, _ in seen for e in o["enemies"] if e["alive"] and e.get("intents")]
    check("every living enemy with an intent reports its move id",
          live and all(e.get("move") for e in live),
          f"{sum(1 for e in live if not e.get('move'))}/{len(live)} missing")
    check("combat obs carries act_floor and total_floor",
          all("act_floor" in o and "total_floor" in o for o, _ in seen))


def check_piles(seen):
    """Pile multisets: counts add up to the pile sizes, and the order is sorted (no leak)."""
    bad_count, bad_order, n = 0, 0, 0
    for o, _ in seen:
        for pile, count in (("draw_pile", "draw_count"), ("discard_pile", "discard_count"),
                            ("exhaust_pile", "exhaust_count")):
            groups = o.get(pile)
            if groups is None:
                bad_count += 1
                continue
            n += 1
            if sum(g["n"] for g in groups) != o[count]:
                bad_count += 1
            keys = [(g["card"], g["up"], g["kw"]) for g in groups]
            if keys != sorted(keys):
                bad_order += 1
    check("pile multisets add up to the pile sizes", n and bad_count == 0,
          f"{bad_count} mismatches over {n} piles")
    check("pile multisets are sorted (draw order cannot leak)", n and bad_order == 0,
          f"{bad_order} unsorted")


def scenario_injected():
    print("\n== Injected Offering + Royalties, potions on: keywords, gold, HP cost, potion numbers")
    seen = play("IRONCLAD", inject_cards=["OFFERING", "ROYALTIES"], potions=True)
    hand = [c for o, _ in seen for c in o["hand"]]
    off = [c for c in hand if c["card"] == "OFFERING"]
    roy = [c for c in hand if c["card"] == "ROYALTIES"]
    check("Offering appears in hand", len(off) >= 5, f"{len(off)}")
    check("Offering carries the Exhaust keyword",
          off and all(c.get("kw", 0) & KW_EXHAUST for c in off),
          f"kw values {sorted({c.get('kw') for c in off})}")
    check("Offering shows HP loss 6 and energy 2",
          off and all(V(c)["hp_loss"] == 6 and V(c)["energy"] == 2 for c in off),
          f"e.g. {V(off[0]) if off else None}")
    check("Royalties appears in hand", len(roy) >= 5, f"{len(roy)}")
    check("Royalties shows gold 30", roy and all(V(c)["gold"] == 30 for c in roy),
          f"gold values {sorted({V(c)['gold'] for c in roy})}")
    strikes = [c for c in hand if c["card"].startswith("STRIKE")]
    check("a Strike carries no keywords", strikes and all(c.get("kw", 0) == 0 for c in strikes))
    belt = [s for o, _ in seen for s in (o["player"].get("potion_slots") or ())]
    check("potions on the belt carry numbers",
          belt and all(isinstance(s.get("vals"), dict) for s in belt)
          and sum(1 for s in belt if any(s["vals"].values())) >= 0.5 * len(belt),
          f"{sum(1 for s in belt if any((s.get('vals') or {}).values()))}/{len(belt)} nonzero")
    exh = [g for o, _ in seen for g in o.get("exhaust_pile") or () if g["card"] == "OFFERING"]
    check("an exhausted Offering shows up in the exhaust pile multiset", len(exh) >= 1,
          f"{len(exh)} observations")


def scenario_aeonglass():
    print("\n== Aeonglass (act-3 boss): move ids and status-card counts")
    seen = play("IRONCLAD", episodes=3, envs=3, act=2, encounters="boss",
                exclude_encounters=["QUEEN_BOSS", "TEST_SUBJECT_BOSS"])
    encs = collections.Counter(o.get("encounter") for o, _ in seen)
    check("only Aeonglass was fought", set(encs) == {"AEONGLASS_BOSS"}, str(dict(encs)))
    moves = collections.Counter(e.get("move") for o, _ in seen for e in o["enemies"] if e["alive"])
    check("Aeonglass's three moves all appear",
          {"EBB_MOVE", "EYE_LASERS_MOVE", "INCREASING_INTENSITY_MOVE"} <= set(moves),
          str(dict(moves)))
    status = [i for o, _ in seen for e in o["enemies"] for i in e.get("intents") or ()
              if i.get("type") == "StatusCard"]
    check("status intents report how many cards they add",
          status and all(i.get("count", 0) > 0 for i in status),
          f"{len(status)} status intents, counts {sorted({i.get('count') for i in status})}")


def scenario_defect():
    print("\n== Defect: orbs")
    seen = play("DEFECT")
    slots = [o.get("orb_slots", 0) for o, _ in seen]
    orbs = [orb for o, _ in seen for orb in o.get("orbs") or ()]
    check("orb_slots > 0 on every Defect decision", slots and min(slots) > 0,
          f"min {min(slots) if slots else None}")
    check("channelled orbs appear", len(orbs) >= 10, f"{len(orbs)} orb observations")
    check("every orb has a known vocab id", orbs and all(x.get("orb_idx", 0) > 0 for x in orbs),
          str(collections.Counter(x["orb"] for x in orbs)))
    light = [x for x in orbs if "LIGHTNING" in x["orb"]]
    check("Lightning orbs show passive > 0 and evoke >= passive",
          light and all(x["passive"] > 0 and x["evoke"] >= x["passive"] for x in light),
          f"{len(light)} lightning, e.g. {light[:1]}")


def scenario_necrobinder():
    print("\n== Necrobinder: Osty as an ally")
    seen = play("NECROBINDER")
    allies = [a for o, _ in seen for a in o.get("allies") or ()]
    check("allies appear", len(allies) >= 10, f"{len(allies)} ally observations")
    check("every ally has a known monster id and hp > 0 while alive",
          allies and all(a.get("monster_idx", 0) > 0 and (a["hp"] > 0 or not a["alive"])
                         for a in allies),
          str(collections.Counter(a["monster"] for a in allies)))


def scenario_queen():
    print("\n== Queen (act-3 boss): card afflictions and room context")
    seen = play("IRONCLAD", episodes=3, envs=3, act=2, encounters="boss",
                exclude_encounters=["AEONGLASS_BOSS", "TEST_SUBJECT_BOSS"])
    encs = collections.Counter(o.get("encounter") for o, _ in seen)
    check("only Queen was fought", set(encs) == {"QUEEN_BOSS"}, str(dict(encs)))
    check("room_type is Boss", all(o.get("room_type") == "Boss" for o, _ in seen))
    first = {}
    for o, _ in seen:
        for e in o["enemies"]:
            if e.get("monster") == "QUEEN" and o.get("turn") == 1 and o.get("round") == 1:
                first[e.get("move")] = first.get(e.get("move"), 0) + 1
    check("Queen opens with PUPPET_STRINGS_MOVE", set(first) == {"PUPPET_STRINGS_MOVE"},
          str(first))
    afflicted = [c for o, _ in seen for c in o["hand"] if c.get("affl_idx", 0) > 0]
    check("afflicted cards appear with a known affliction id", len(afflicted) >= 5,
          f"{len(afflicted)} afflicted hand cards")


def main():
    scenario_ironclad()
    scenario_defect()
    scenario_necrobinder()
    scenario_queen()
    scenario_injected()
    scenario_aeonglass()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
