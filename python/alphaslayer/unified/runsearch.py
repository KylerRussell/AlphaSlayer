"""Run-level search (probe RunSearch): shop purchases, card rewards and route choices, valued
by playing each option out to the next act boss and playing that boss several times.

RunPool.run(run_search=RunSearcher(...)) parks a live decision that carries the probe's
"ctx", queues rollouts (option x world) on other game slots, and hands the decision back with
``q`` = each option's mean leaf value once they report. A rollout's leaf, from its terminal
"rollout" report:

    v = mean over boss playouts b of [won_b] * (w_act * (act + 1) + w_hp * hp_b / max_hp)

(the training potential after beating that act's boss; a death on the way, or a lost boss, is
0). Every option sees the SAME sampled worlds and boss seeds, so options compare paired.

``pick`` keeps the policy's choice unless another option beats it by ``z`` paired standard
errors: with ~8-16 boss playouts per option the estimates are noisy, and the first fight
search showed what an argmax over noise does.
"""
from __future__ import annotations

import json
import time

import numpy as np

from alphaslayer.unified import search as SR


class RunSearcher:
    def __init__(self, net, vocab, dev, policy, kinds=("card_reward", "shop", "travel"), k=3,
                 worlds=4, boss_plays=2, deck_plays=8, max_live=6, w_hp=0.1, w_act=1 / 3, w_dmg=0.5, z=1.0,
                 verify=False, dump=None, drop_last=False):
        self.net, self.vocab, self.dev = net, vocab, dev
        self.policy = policy                   # (kinds, obs, legal) -> actions, for rollouts
        self.kinds = tuple(kinds)
        self.k, self.worlds, self.boss_plays, self.max_live = k, worlds, boss_plays, max_live
        self.deck_plays = deck_plays
        self.w_hp, self.w_act, self.z = w_hp, w_act, z
        # a lost boss playout is worth w_dmg x a win, scaled by the fraction of the boss's hp
        # it removed: early in an act EVERY option loses to the boss (a floor-2 deck died in
        # 64/64 playouts), and "how close" is what still tells options apart
        self.w_dmg = w_dmg
        self.verify = verify
        self.drop_last = drop_last           # negative control for verify: replay one answer short
        self.dump = open(dump, "w") if dump else None
        self.verified = {"match": 0, "mismatch": 0}
        self.mismatches: list = []
        self.picks = {"searched": 0, "changed": 0}

    def probe_args(self) -> list[str]:
        return ["--probe-fight-restore", f"--probe-run-search={','.join(self.kinds)}"]

    # ---- options -------------------------------------------------------------------------------

    def plan(self, m: dict) -> dict:
        """How a decision is played out.

        Deck changes (card reward, shop): the option's room is finished, then the run goes
        STRAIGHT to the act boss with the deck, relics and hp as they are, and plays it
        ``deck_plays`` times in one world: "does this deck beat this boss", paired playout by
        playout across options (same boss seeds), without the path's noise.
        Routes: the path IS the decision, so each option is played along the map to the boss
        in ``worlds`` sampled worlds, ``boss_plays`` boss playouts each, paired by world.
        """
        if m["kind"] == "travel":
            return {"cands": self.candidates(m), "worlds": self.worlds, "boss_plays": self.boss_plays,
                    "teleport": False}
        return {"cands": self.candidates(m), "worlds": 1, "boss_plays": self.deck_plays, "teleport": True}

    def candidates(self, m: dict) -> list[int]:
        """Every option if there are at most k, else the policy's top k."""
        n = len(m["legal"])
        if n <= self.k:
            return list(range(n))
        lp = SR.log_prior(self.net, self.vocab, self.dev, [m["obs"]], [m["legal"]], kinds=[m["kind"]])[0]
        return [int(i) for i in np.argsort(-lp)[:self.k]]

    def leaf_values(self, rep: dict) -> list[float]:
        """One value per boss playout (all 0: died before the boss)."""
        if not rep.get("reached_boss"):
            return [0.0] * self.boss_plays
        top = self.w_act * (rep["act"] + 1)
        mh = max(1, rep.get("max_hp", 1))
        dmg = rep.get("boss_dmg") or [0.0] * len(rep.get("boss_hp", []))
        return [(top + self.w_hp * hp / mh) if hp > 0 else top * self.w_dmg * d
                for hp, d in zip(rep.get("boss_hp", []), dmg)]

    def leaf_value(self, rep: dict) -> list[float]:
        """A rollout's per-playout values (RunPool keeps them per option and world)."""
        return self.leaf_values(rep)

    def combine(self, vs: list) -> "_Q":
        """An option's (world, playouts) results -> its q: the mean, carrying the playouts by
        world so pick() can pair options on the worlds they share."""
        by_world = {w: list(p) for w, p in vs}
        flat = [x for p in by_world.values() for x in p]
        return _Q(sum(flat) / max(1, len(flat)), by_world)

    # ---- the decision --------------------------------------------------------------------------

    def record(self, m: dict, task: dict):
        """Called by RunPool when an option set is fully played out (dump only)."""
        if self.dump:
            legal = m["legal"]
            self.dump.write(json.dumps({
                "kind": m["kind"], "run": m.get("run"), "step": m.get("step"), "n": len(legal),
                "opts": [{"i": c, "kind": legal[c].get("kind"), "id": legal[c].get("id"),
                          "q": float(m["q"][c]), "by_world": {str(w): p for w, p in vs}}
                         for c, vs in task["vals"].items()],
                "s": round(time.time() - task["t0"], 1),
                "report0": (task.get("reports") or [None])[0]}) + "\n")
            self.dump.flush()

    def pick(self, q: list, greedy_a: int) -> int:
        """The searched choice: the policy's, unless an option beats it by z paired standard
        errors (paired: the same world and boss seed)."""
        if greedy_a >= len(q) or not isinstance(q[greedy_a], _Q):
            return greedy_a
        self.picks["searched"] += 1
        base = q[greedy_a].by_world
        best, best_t = greedy_a, self.z
        for c, qc in enumerate(q):
            if c == greedy_a or not isinstance(qc, _Q):
                continue
            common = qc.by_world.keys() & base.keys()
            if len(common) > 1:      # routes: whole worlds are the independent units
                d = [np.mean(qc.by_world[w]) - np.mean(base[w]) for w in common]
            else:                    # deck changes: one world, each boss seed is a pair
                d = [a - b for w in common for a, b in zip(qc.by_world[w], base[w])]
            if len(d) < 2:
                continue
            d = np.array(d)
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = d.mean() / se if se > 0 else (np.inf if d.mean() > 0 else 0.0)
            if t > best_t:
                best, best_t = c, t
        self.picks["changed"] += int(best != greedy_a)
        return best

    def verify_decision(self, m: dict, task: dict | None):
        """A verify rollout landed: is it the live decision (observation and options)?"""
        if task is None or task.get("verified"):
            return
        task["verified"] = True
        live = task["msg"]
        same = (m["kind"] == live["kind"]
                and json.dumps(m["obs"], sort_keys=True) == json.dumps(live["obs"], sort_keys=True)
                and json.dumps(m["legal"], sort_keys=True) == json.dumps(live["legal"], sort_keys=True))
        self.verified["match" if same else "mismatch"] += 1
        if not same and len(self.mismatches) < 20:
            a, b = json.dumps(live, sort_keys=True), json.dumps(m, sort_keys=True)
            i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            self.mismatches.append({"kind": live["kind"], "live": a[max(0, i - 120):i + 200],
                                    "rollout": b[max(0, i - 120):i + 200]})


class _Q(float):
    """An option's mean value that also carries its per-playout values by world."""

    def __new__(cls, mean, by_world):
        x = super().__new__(cls, mean)
        x.by_world = by_world
        return x
