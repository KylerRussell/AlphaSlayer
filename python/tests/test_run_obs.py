"""Live invariant test: every run-level decision and every fight knows which act and floor it is in.

Written for two bugs that looked healthy in every log line:

  1. Seven of nine run decision kinds (card_reward, rest, event, shop, card_select, treasure,
     potion_ooc) sent PlayerObs(), which had no act/act_floor/total_floor. The run net read
     all three as 0 there, and --progress-shaping read the floor as 0, so it paid about -1.2 to
     every late travel and +1.2 to the decision after it.
  2. The deck harvest took its act label from that same dict, so about 35% of act-2/3 fights
     were labelled act 0.

A random policy dies in act 1, where "act = 0" is the right answer, so it cannot tell a fixed
probe from a broken one. This plays real runs with trained checkpoints so that act 2 and act 3
are reached. It then asserts, against the act tracked from travel observations:

  * every decision's player dict carries act / act_floor / total_floor, and the act matches
  * encode_state (the run net's real input path) delivers that act and floor, not zeros
  * run_progress (the trainer's shaping potential) is never 0 after the run has moved
  * the progress-shaping term stays small per step (no sawtooth), via the real Buffer
  * every fight_end reports its act, and harvest_record labels it with that act

M1 additions (the unified model's inputs), checked on the same runs:

  * every run decision names the act boss, and every Boss-room fight IS that boss
  * deck cards, card rewards, shop cards and card prompts carry keywords and numbers
  * shop relics/potions and event options carry numbers; event options name the cards they add
  * the combat observation inside a run reports the act and floor it is fought on
  * harvest records carry floor, gold, boss and potions

It also checks that it EXERCISED the case that matters: enough decisions and fights after act 1.
Too few is a failure ("inconclusive"), not a pass.

Run from python/:
    XDG_DATA_HOME="$(../headless_home.sh)" PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 \\
        PYTHONPATH=. .venv7/bin/python tests/test_run_obs.py
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import train_run as T
from alphaslayer.model import CombatNet, use_stable_attention
from alphaslayer.runenv import VecRunEnv
from alphaslayer.runmodel import RunNet, encode_state, migrate_state_scalars
from train_rl import load_compat

FAILS = []
FIELDS = ("act", "act_floor", "total_floor")


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def load_models(combat_ckpt, run_ckpt, dev):
    # Every GPU consumer of the teachers must use the math attention backend. ROCm's
    # mem-efficient kernels on gfx1100 are experimental and were tied to hardware exceptions,
    # and collect_distill.py ran without this (it imports this loader, not the trainer's setup)
    # when the 2026-09-26 03:08 mode1 reset happened. Set here, so no caller can forget it.
    use_stable_attention()
    ck = torch.load(combat_ckpt, map_location=dev, weights_only=False)
    combat = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"]).to(dev)
    load_compat(combat, ck["model"])
    combat.eval()
    T._ENC = T.Encoder(ck["sizes"]["cards"])
    rck = torch.load(run_ckpt, map_location=dev, weights_only=False)
    runnet = RunNet(ck["sizes"]["cards"], ck["sizes"]["relics"], ck["sizes"].get("potions", 66),
                    per_kind_heads=True).to(dev)
    load_compat(runnet, migrate_state_scalars(rck["model"]))
    runnet.eval()
    return combat, runnet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat-ckpt", default="combat_r5.pt")
    ap.add_argument("--run-ckpt", default="run_r5.pt")
    ap.add_argument("--envs", type=int, default=10)
    ap.add_argument("--runs-per-env", type=int, default=2)
    ap.add_argument("--min-late-decisions", type=int, default=40)
    ap.add_argument("--min-late-fights", type=int, default=15)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    use_stable_attention()
    dev = torch.device(a.device)
    combat, runnet = load_models(a.combat_ckpt, a.run_ckpt, dev)

    cur_act = {}                     # env -> act from the most recent travel observation
    cur_floor = {}                   # env -> total_floor from the most recent travel
    last_player = {}                 # env -> player dict of the last run decision (as trainer)
    run_steps = collections.defaultdict(list)   # env -> [(kind, prog)] for the shaping check
    finished_runs = []

    bad = collections.Counter()      # invariant -> violation count
    examples = {}
    late_kinds = collections.Counter()
    late_fights = 0
    fights_total = 0
    run_no = collections.Counter()   # env -> runs finished, so keys are per RUN, not per env
    boss_of = {}                     # (env, run, act) -> boss ids announced in run decisions
    boss_fights = []                 # (env, run, act, encounter) for every Boss-room fight
    offers = collections.defaultdict(list)   # offer kind -> [legal action dicts]
    deck_cards = []
    harvested = []

    def violate(key, detail):
        bad[key] += 1
        examples.setdefault(key, detail)

    def policy(kinds, obs, legal, idxs):
        groups = {"combat": [], "run": []}
        for i, k in enumerate(kinds):
            groups["combat" if k == "combat" else "run"].append(i)
            if k == "combat":
                o, env = obs[i], idxs[i]
                if env in cur_act and o.get("act") != cur_act[env]:
                    violate("combat obs act matches the run's act",
                            f"combat act={o.get('act')} tracked={cur_act[env]}")
                if env in cur_floor and (o.get("total_floor") or 0) < cur_floor[env]:
                    violate("combat obs total_floor is the run's floor",
                            f"combat floor={o.get('total_floor')} tracked={cur_floor[env]}")
                continue
            env, o = idxs[i], obs[i]
            p = o.get("player", o)
            if k == "travel":
                cur_act[env] = int(o["act"])
                cur_floor[env] = int(o.get("total_floor", 0))
            # Every check below compares against the TRAVEL-TRACKED truth, never against the
            # decision's own fields, so one missing field cannot make the others pass vacuously.
            last_player[env] = p
            want = cur_act.get(env)
            floor = cur_floor.get(env, 0)
            missing = [f for f in FIELDS if f not in p]
            if missing:
                violate("player dict carries act/act_floor/total_floor",
                        f"{k}: missing {missing}")
            elif want is not None and int(p["act"]) != want:
                violate("decision act matches travel-tracked act",
                        f"{k}: act={p['act']} tracked={want}")
            if want is None:
                continue
            if want >= 1 and k != "travel":
                late_kinds[k] += 1
            # The run net's real input path. Scalars 6 and 10 are act/3 and total_floor/51.
            scal = encode_state([k], [o], "cpu")[0][0]
            if round(float(scal[6]) * 3) != want:
                violate("encode_state delivers the act",
                        f"{k}: scal[6]*3={float(scal[6])*3:.2f} tracked={want}")
            if round(float(scal[10]) * 51) < floor:
                violate("encode_state delivers total_floor",
                        f"{k}: scal[10]*51={float(scal[10])*51:.2f} < last travel floor {floor}")
            prog = T.run_progress(o)
            if prog < floor:
                violate("run_progress never falls below the last travel's floor",
                        f"{k}: prog={prog} last travel floor={floor}")
            run_steps[env].append((k, prog))
            if p.get("boss"):
                boss_of.setdefault((env, run_no[env], want), set()).add(p["boss"])
            else:
                violate("run decisions name the act boss", f"{k}: boss={p.get('boss')!r}")
            if k == "card_reward":
                deck_cards.extend(p.get("deck_cards") or ())
            if k in ("card_reward", "shop", "event", "card_select", "treasure"):
                offers[k].extend(legal[i])

        out = [0] * len(kinds)
        with torch.no_grad():
            for grp, members in groups.items():
                if not members:
                    continue
                o = [obs[i] for i in members]
                l = [legal[i] for i in members]
                if grp == "combat":
                    lg, _, m = T.combat_forward(combat, o, l, dev)
                    idx = lg.masked_fill(~m, torch.finfo(lg.dtype).min / 4).argmax(-1)
                else:
                    lg, _, m = T.run_forward(runnet, [kinds[i] for i in members], o, l, dev)
                    idx, _ = T.sample(lg, m)
                for j, i in enumerate(members):
                    out[i] = int(idx[j])
        return out

    def on_fight(env, fr):
        nonlocal late_fights, fights_total
        fights_total += 1
        want = cur_act.get(env)
        if fr.act < 0:
            violate("fight_end reports its act", f"{fr.encounter}: act={fr.act}")
        elif want is not None and fr.act != want:
            violate("fight_end act matches travel-tracked act",
                    f"{fr.encounter}: fight act={fr.act} tracked={want}")
        if fr.room == "Boss":
            boss_fights.append((env, run_no[env], fr.act, fr.encounter))
        rec = T.harvest_record(last_player.get(env), fr)
        if rec is not None:
            harvested.append(rec)
        if rec is not None and want is not None and rec["act"] != want:
            violate("harvest_record labels the fight's act",
                    f"{fr.encounter}: harvest act={rec['act']} tracked={want}")
        if want is not None and want >= 1:
            late_fights += 1

    def on_terminal(env, rr):
        finished_runs.append(run_steps.pop(env, []))
        run_no[env] += 1
        cur_act.pop(env, None)
        cur_floor.pop(env, None)
        last_player.pop(env, None)

    print(f"playing {a.envs * a.runs_per_env} runs with {a.run_ckpt} / {a.combat_ckpt}")
    with VecRunEnv(n_envs=a.envs, characters=["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER",
                                             "REGENT"],
                   runs_per_env=a.runs_per_env, seed="OBSTEST_", ascension=0, act_cap=3,
                   run_offset=0) as venv:
        venv.run(policy, on_fight=on_fight, on_terminal=on_terminal, deadline_s=900)
        n_runs = len(venv.results)

    # Progress shaping through the REAL Buffer: no step should swing by more than a few floors'
    # worth. Before the fix, a late travel paid about -0.03 * floor.
    worst = 0.0
    for steps in finished_runs:
        if len(steps) < 2:
            continue
        b = T.Buffer()
        for k, prog in steps:
            b.add_run(0, {"kind": k, "prog": prog, "phi": 0.0, "deck": 0.0, "v": 0.0})
        b.finish_run(0, 0.0, 0.996, prog_shaping=0.03, prog_end=steps[-1][1])
        worst = max(worst, max(abs(s["r"]) for s in b.run_done))

    print(f"\n  {n_runs} runs, {fights_total} fights; after act 1: {late_fights} fights, "
          f"{sum(late_kinds.values())} non-travel decisions {dict(late_kinds)}")
    invariants = [
        "player dict carries act/act_floor/total_floor",
        "decision act matches travel-tracked act",
        "encode_state delivers the act",
        "encode_state delivers total_floor",
        "run_progress never falls below the last travel's floor",
        "fight_end reports its act",
        "fight_end act matches travel-tracked act",
        "harvest_record labels the fight's act",
        "run decisions name the act boss",
        "combat obs act matches the run's act",
        "combat obs total_floor is the run's floor",
    ]
    for name in invariants:
        check(name, bad[name] == 0,
              f"{bad[name]} violations, e.g. {examples[name]}" if bad[name] else "")
    check("progress-shaping step reward stays small (no sawtooth)", worst < 0.1,
          f"max |r| = {worst:.3f}")
    # Delivery: the test must actually have exercised acts 2 and 3.
    check("exercised enough non-travel decisions after act 1",
          sum(late_kinds.values()) >= a.min_late_decisions,
          f"{sum(late_kinds.values())} < {a.min_late_decisions}: INCONCLUSIVE"
          if sum(late_kinds.values()) < a.min_late_decisions else "")
    check("exercised enough fights after act 1", late_fights >= a.min_late_fights,
          f"{late_fights} < {a.min_late_fights}: INCONCLUSIVE"
          if late_fights < a.min_late_fights else "")
    check("card_reward decisions seen after act 1", late_kinds["card_reward"] > 0)

    # ---- M1 additions ----------------------------------------------------------------
    multi = {key: b for key, b in boss_of.items() if len(b) != 1}
    check("one boss per act per run", not multi, str(list(multi.items())[:3]))
    mism = [(env, act, enc, boss_of.get((env, r, act))) for env, r, act, enc in boss_fights
            if boss_of.get((env, r, act)) != {enc}]
    check("every Boss-room fight is the boss the run was told about",
          boss_fights and not mism,
          f"{len(boss_fights)} boss fights, mismatches e.g. {mism[:3]}")

    def has_card_fields(d):
        return isinstance(d.get("vals"), dict) and "kw" in d

    check("every deck card carries keywords and numbers",
          deck_cards and all(has_card_fields(c) for c in deck_cards), f"{len(deck_cards)} cards")
    strikes = [c for c in deck_cards if c["card"].startswith("STRIKE")]
    dmg = lambda c: (c.get("vals") or {}).get("damage", 0)
    check("deck Strikes show their base damage", strikes and all(dmg(c) > 0 for c in strikes),
          f"values {sorted({dmg(c) for c in strikes})}")
    for kind, ak in (("card_reward", "card"), ("shop", "card"), ("card_select", "select_card")):
        cards = [a for a in offers[kind] if a.get("kind") == ak]
        nz = sum(1 for a in cards if any((a.get("vals") or {}).values()))
        check(f"{kind}: offered cards carry keywords and numbers",
              cards and all(has_card_fields(a) for a in cards) and nz >= 0.6 * len(cards),
              f"{nz}/{len(cards)} with nonzero numbers")
    for ak in ("relic", "potion"):
        items = [a for a in offers["shop"] if a.get("kind") == ak]
        nz = sum(1 for a in items if any((a.get("vals") or {}).values()))
        check(f"shop {ak}s carry numbers", items and nz >= 0.3 * len(items),
              f"{nz}/{len(items)} nonzero")
    ev = offers["event"]
    ev_nz = [a for a in ev if any((a.get("vals") or {}).values())]
    ev_cards = [a for a in ev if a.get("cards")]
    check("event options carry the event's numbers they refer to", len(ev_nz) >= 10,
          f"{len(ev_nz)}/{len(ev)} options with numbers, e.g. "
          + "; ".join(f"{a['key'].split('.')[0]}.{a['key'].split('.')[-1]}="
                      f"{ {k: v for k, v in a['vals'].items() if v} }" for a in ev_nz[:4]))
    # Attribution coverage: an event whose numbers are nonzero should have them attached to at
    # least one of its options. An event failing this means the text matching missed them.
    by_event = collections.defaultdict(lambda: [False, False])   # event -> [has nums, attributed]
    for a in ev:
        e = by_event[a.get("event", "?")]
        e[0] |= any((a.get("event_vals") or {}).values())
        e[1] |= any((a.get("vals") or {}).values())
    # Events whose numbers genuinely appear in no option text. Each entry needs a reason.
    #   SYMBIOTE: its only number is CardsVar(1), "transform 1 card"; the options state their
    #             effects through hover tips (Corrupted, Transform), which are sent as "tips".
    no_text_numbers = {"SYMBIOTE"}
    missed = sorted(k for k, (has, att) in by_event.items()
                    if has and not att and k not in no_text_numbers)
    with_nums = sum(1 for has, _ in by_event.values() if has)
    check("every event with numbers attributes them to one of its options", not missed,
          f"{with_nums - len(missed)}/{with_nums} events attributed; missed {missed}")
    tipped = [a for a in ev if a.get("tips")]
    check("event options carry their hover-tip ids", len(tipped) >= 10,
          f"{len(tipped)}/{len(ev)} options, e.g. "
          + "; ".join(f"{a['key'].split('.')[-1]}->{a['tips'][:2]}" for a in tipped[:3]))
    check("event options name the cards they show", len(ev_cards) >= 3,
          f"{len(ev_cards)} options, e.g. "
          + "; ".join(f"{a['key'].split('.')[-1]}->{a['cards']}" for a in ev_cards[:4]))
    keys = ("act_floor", "total_floor", "gold", "boss", "potions")
    check("harvest records carry floor, gold, boss and potions",
          harvested and all(all(r.get(k) is not None for k in keys) for r in harvested),
          f"{len(harvested)} records")

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
