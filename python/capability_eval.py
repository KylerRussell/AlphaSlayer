"""M5 capability evals from real runs (docs/CAPABILITIES.md, "Capability evals").

Plays the fixed eval seeds with a unified checkpoint or the r5 teachers, the same paired seeds as
eval_unified.py, and measures BEHAVIOURS, not just the win rate. Every decision is streamed through
recorders, so nothing large is kept in memory.

  A1 lethal      turns where a lethal line exists vs a lone enemy: how often it kills that turn.
                 "Exists" is an UPPER BOUND: the best energy-feasible sum of per-target attack
                 previews >= the enemy's hp + block (ignores draw, buffs, multi-card combos)
  A2 block       per ended turn with an incoming attack: fully blocked rate, unblocked damage and
                 wasted over-block (fractions of max HP); block raised on turns with no attack
  A6 bosses      win rate per boss
  B4 potions     potions drunk per fight by room; potions held entering boss fights; gate allow
                 rate by room
  C2 thinning    card removal bought when affordable
  C3 curses      curse-adding event options: accept rate when offered, by deck size
  C6 campfire    heal vs smith vs other by hp% and by "next stop is the boss"
  D  characters  act clears and wins per character
  E1 calibration (unified only) value head P(win) and fight head P(win this fight): Brier skill and
                 a reliability table

    PYTHONPATH=. .venv7/bin/python capability_eval.py --ckpt unified_m4_shaped4.pt --runs 500 --out cap.json
    PYTHONPATH=. .venv7/bin/python capability_eval.py --teacher --runs 500 --out cap_r5.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import train_run as T
from alphaslayer.model import use_stable_attention
from alphaslayer.runenv import VecRunEnv
from alphaslayer.unified import features as FT
from alphaslayer.unified.net import UnifiedNet, to_torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
from test_run_obs import load_models  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS = ["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"]
CURSES = {"ASCENDERS_BANE", "BAD_LUCK", "CLUMSY", "CURSE_OF_THE_BELL", "DEBT", "DECAY", "DOUBT",
          "ENTHRALLED", "FOLLY", "GREED", "GUILTY", "INJURY", "NORMALITY", "POOR_SLEEP", "REGRET",
          "SHAME", "SPORE_MIND", "WRITHE"}


# ---- policies: (kinds, obs, legal, idxs) -> (actions, extras) -------------------------------

def make_unified(path, dev, bf16):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = UnifiedNet(ck["sizes"], **ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    vocab = FT.Vocab(json.load(open(os.path.join(HERE, "vocab_m1.json"))))

    def policy(kinds, obs, legal, idxs):
        b = to_torch(FT.encode_batch(list(zip(kinds, obs, legal)), vocab), dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = net(b)
        lg = out["logits"].float()
        v = out["value"].float().tolist()
        pf = torch.sigmoid(out["aux"]["fight_win"].float()).tolist()
        acts = []
        for j, k in enumerate(kinds):
            row = lg[j, :len(legal[j])]
            acts.append(int(row.argmax()) if k == "combat"
                        else int(torch.multinomial(F.softmax(row, -1), 1)))
        return acts, [{"v": v[j], "pf": pf[j]} for j in range(len(kinds))]
    return policy


def make_teacher(dev):
    combat, runnet = load_models("combat_r5.pt", "run_r5.pt", dev)

    def policy(kinds, obs, legal, idxs):
        out = [0] * len(kinds)
        groups = collections.defaultdict(list)
        for i, k in enumerate(kinds):
            groups["combat" if k == "combat" else "run"].append(i)
        with torch.no_grad():
            for grp, members in groups.items():
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
        return out, [{} for _ in kinds]
    return policy


# ---- recorders ----------------------------------------------------------------------------------

class Recorder:
    def __init__(self):
        self.c = collections.defaultdict(float)          # counters
        self.lists = collections.defaultdict(list)
        self.env = collections.defaultdict(dict)         # per-env running state

    def decision(self, env, kind, o, legal, a, ex):
        st = self.env[env]
        ch = legal[a] if 0 <= a < len(legal) else {}
        if kind == "combat":
            self._combat(env, st, o, legal, ch, ex)
        elif kind == "potion_gate":
            room = o.get("room_type") or "?"
            self.c[f"gate_n/{room}"] += 1
            self.c[f"gate_allow/{room}"] += int(ch.get("kind") == "allow")
        elif kind == "rest":
            p = o
            hp = (p.get("hp", 0) or 0) / max(1, p.get("max_hp", 1) or 1)
            hb = "hp<40%" if hp < 0.4 else ("hp40-70%" if hp < 0.7 else "hp>70%")
            pre = "pre-boss" if (p.get("act_floor") or 0) >= 15 else "mid-act"
            opt = ch.get("option", "?")
            opt = opt if opt in ("HEAL", "SMITH", "LEAVE") else "other"
            for key in (hb, pre):
                self.c[f"rest_n/{key}"] += 1
                self.c[f"rest/{key}/{opt}"] += 1
        elif kind == "shop":
            rem = [x for x in legal if x.get("kind") == "remove_card" and x.get("affordable", True)]
            if rem:
                self.c["shop_remove_offered"] += 1
                self.c["shop_remove_taken"] += int(ch.get("kind") == "remove_card")
        elif kind == "event":
            cursed = [i for i, x in enumerate(legal) if set(x.get("cards") or ()) & CURSES]
            if cursed:
                deck = (o.get("deck_size") or 0)
                db = "deck<=20" if deck <= 20 else "deck>20"
                self.c[f"curse_offered/{db}"] += 1
                self.c[f"curse_taken/{db}"] += int(a in cursed)
        if ex and "v" in ex and kind != "combat":
            self.lists["v_run"].append((env, ex["v"]))

    def _combat(self, env, st, o, legal, ch, ex):
        turn = (o.get("turn"), o.get("round"))
        enemies = [e for e in o.get("enemies") or [] if e.get("alive")]
        p = o.get("player", {})
        if ex and "pf" in ex:
            st.setdefault("pf", []).append(ex["pf"])
        if ex and "v" in ex:
            self.lists["v_run"].append((env, ex["v"]))
        if st.get("turn") != turn:                       # first decision of a new player turn
            st["turn"] = turn
            st["lethal"] = False
            if len(enemies) == 1 and enemies[0].get("hittable", True):
                tgt = (o.get("enemies") or []).index(enemies[0])
                e = enemies[0]
                st["lethal"] = self._max_damage(o, legal, tgt) >= (e.get("hp", 0) + e.get("block", 0))
            if st["lethal"]:
                self.c["lethal_turns"] += 1
        st["room"] = o.get("room_type") or "?"
        if ch.get("kind") == "use_potion":
            self.c[f"potions_used/{st['room']}"] += 1
        if ch.get("kind") == "end_turn":
            if st.get("lethal"):
                self.c["lethal_missed"] += 1
                st["lethal"] = False
            incoming = sum(int(it.get("damage", 0) or 0) for e in enemies
                           for it in e.get("intents") or () if (it.get("type") or "").startswith("Attack"))
            block = int(p.get("block", 0) or 0)
            mx = max(1, int(p.get("max_hp", 1) or 1))
            if incoming > 0:
                self.c["turns_attacked"] += 1
                self.c["turns_fully_blocked"] += int(block >= incoming)
                self.c["unblocked_frac"] += max(0, incoming - block) / mx
                self.c["overblock_frac"] += max(0, block - incoming) / mx
            else:
                self.c["turns_no_attack"] += 1
                self.c["block_when_no_attack"] += int(block >= 5)
        # potions held entering a boss fight: first decision of the fight
        if not st.get("in_fight"):
            st["in_fight"] = True
            if st["room"] == "Boss":
                self.c["boss_fights_entered"] += 1
                self.c["boss_potions_held"] += len([x for x in p.get("potion_slots") or ()])

    @staticmethod
    def _max_damage(o, legal, tgt):
        energy = int((o.get("player") or {}).get("energy", 0) or 0)
        hand = o.get("hand") or []
        items = {}
        for a in legal:
            if a.get("kind") != "play" or a.get("target") != tgt:
                continue
            v = a.get("vals") or {}
            dmg = int(v.get("damage", 0) or 0) * max(1, int(v.get("repeat", 0) or 0))
            h = a.get("hand", -1)
            if dmg > 0 and 0 <= h < len(hand):
                cost = int(hand[h].get("cost", 0) or 0)
                if cost < 0 or hand[h].get("cost_x"):
                    continue
                # (damage, cost) per hand card; a card listed once per target keeps its best.
                items[h] = (max(items[h][0], dmg), cost) if h in items else (dmg, cost)
        best = [0] * (energy + 1)
        for dmg, cost in ((d, c) for d, c in items.values()):
            for e in range(energy, cost - 1, -1):
                best[e] = max(best[e], best[e - cost] + dmg)
        return best[energy]

    def fight_end(self, env, fr):
        st = self.env[env]
        if fr.won and st.get("lethal"):
            self.c["lethal_taken"] += 1
        self.c[f"fights/{fr.room}"] += 1
        for pf in st.get("pf", []):
            self.lists["pf"].append((pf, int(fr.won)))
        for k in ("lethal", "turn", "pf", "in_fight"):
            st.pop(k, None)

    def run_end(self, env, rr, char):
        for e, v in [x for x in self.lists["v_run"] if x[0] == env]:
            self.lists["v"].append((v, int(rr.won)))
        self.lists["v_run"] = [x for x in self.lists["v_run"] if x[0] != env]
        self.env.pop(env, None)


def calib(pairs, bins=10):
    if not pairs:
        return None
    p = np.array([x for x, _ in pairs]); y = np.array([w for _, w in pairs], float)
    brier = float(((p - y) ** 2).mean()); base = float(((y.mean() - y) ** 2).mean())
    rel = []
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins + (1e-9 if b == bins - 1 else 0))
        if m.sum() >= 20:
            rel.append((round(b / bins, 1), round(float(p[m].mean()), 3), round(float(y[m].mean()), 3), int(m.sum())))
    return {"n": len(y), "base_rate": float(y.mean()), "brier": brier, "brier_base": base,
            "skill": (1 - brier / base) if base > 0 else None, "reliability": rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--envs", type=int, default=25)
    ap.add_argument("--seed", default="EVAL_")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    use_stable_attention()
    torch.manual_seed(0)
    dev = torch.device(a.device)
    policy = make_teacher(dev) if a.teacher else make_unified(a.ckpt, dev, a.bf16)
    rec = Recorder()
    boss = collections.defaultdict(lambda: [0, 0])
    per_char = collections.defaultdict(lambda: [0, 0, 0, 0])

    def wrapped(kinds, obs, legal, idxs):
        acts, extras = policy(kinds, obs, legal, idxs)
        for k, o, l, i, act, ex in zip(kinds, obs, legal, idxs, acts, extras):
            rec.decision(i, k, o, l, act, ex)
        return acts

    def on_fight(env, fr):
        rec.fight_end(env, fr)
        if fr.room == "Boss":
            boss[fr.encounter][0] += int(fr.won)
            boss[fr.encounter][1] += 1

    def on_terminal(env, rr):
        ch = CHARS[env % len(CHARS)]
        rec.run_end(env, rr, ch)
        c = per_char[ch]
        c[0] += int(rr.act >= 1 or rr.won); c[1] += int(rr.act >= 2 or rr.won); c[2] += int(rr.won); c[3] += 1

    t0 = time.time()
    per_env = math.ceil(a.runs / a.envs)
    with VecRunEnv(n_envs=a.envs, characters=CHARS, runs_per_env=per_env, seed=a.seed,
                   ascension=0, act_cap=3, run_offset=0) as venv:
        venv.run(wrapped, on_fight=on_fight, on_terminal=on_terminal, deadline_s=7200)
        results = list(venv.results)

    c = rec.c
    n = len(results)
    rate = lambda x, y: (c[x] / c[y]) if c[y] else None
    rep = {
        "policy": "r5 teachers" if a.teacher else a.ckpt, "runs": n, "wall_s": round(time.time() - t0),
        "outcome": {"act1": sum(1 for r in results if r.act >= 1 or r.won) / max(1, n),
                    "act2": sum(1 for r in results if r.act >= 2 or r.won) / max(1, n),
                    "win": sum(r.won for r in results) / max(1, n)},
        "A1_lethal": {"lethal_turns": c["lethal_turns"], "taken": rate("lethal_taken", "lethal_turns"),
                      "missed": rate("lethal_missed", "lethal_turns")},
        "A2_block": {"turns_attacked": c["turns_attacked"],
                     "fully_blocked": rate("turns_fully_blocked", "turns_attacked"),
                     "unblocked_per_turn_maxhp": rate("unblocked_frac", "turns_attacked"),
                     "overblock_per_turn_maxhp": rate("overblock_frac", "turns_attacked"),
                     "blocked_5plus_when_no_attack": rate("block_when_no_attack", "turns_no_attack")},
        "A6_bosses": {e: {"win": w / t, "n": t} for e, (w, t) in sorted(boss.items())},
        "B4_potions": {
            "used_per_fight": {r: (c[f"potions_used/{r}"] / c[f"fights/{r}"]) if c[f"fights/{r}"] else None
                               for r in ("Monster", "Elite", "Boss")},
            "held_entering_boss": rate("boss_potions_held", "boss_fights_entered"),
            "gate_allow": {r: rate(f"gate_allow/{r}", f"gate_n/{r}") for r in ("Monster", "Elite", "Boss")}},
        "C2_remove_when_affordable": rate("shop_remove_taken", "shop_remove_offered"),
        "C3_curse_accept": {d: {"rate": rate(f"curse_taken/{d}", f"curse_offered/{d}"),
                                "offered": c[f"curse_offered/{d}"]} for d in ("deck<=20", "deck>20")},
        "C6_campfire": {key: {opt: rate(f"rest/{key}/{opt}", f"rest_n/{key}")
                              for opt in ("HEAL", "SMITH", "LEAVE", "other")} | {"n": c[f"rest_n/{key}"]}
                        for key in ("hp<40%", "hp40-70%", "hp>70%", "mid-act", "pre-boss")},
        "D_characters": {ch: {"act1": v[0] / v[3], "act2": v[1] / v[3], "win": v[2] / v[3], "n": v[3]}
                         for ch, v in sorted(per_char.items()) if v[3]},
    }
    if not a.teacher:
        rep["E1_calibration"] = {"value_P(win)": calib(rec.lists["v"]),
                                 "fight_head_P(win fight)": calib(rec.lists["pf"])}
    txt = json.dumps(rep, indent=1, default=lambda x: round(x, 4) if isinstance(x, float) else x)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
