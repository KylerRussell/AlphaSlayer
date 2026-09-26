"""Combat curriculum for the unified model: isolated act-2/3 fights, credited like in-run fights.

The same reason as the old curriculum (alphaslayer/curriculum.py): in-run data reaches act-2/3
fights only in the runs that get there, so the model is taught those fights directly, on decks
harvested from its own runs at the HP they actually entered with. Two things are new:

  * SAME OBJECTIVE. A won fight ends at V(post-fight run state): the harvested run context with
    the HP the fight left, scored by the model's own value head. A lost fight ends the run: 0.
    The curriculum therefore optimises exactly what in-run fights optimise, and cannot pull the
    policy toward "win this fight at any cost" (the old per-fight objective).
  * LOSS-WEIGHTED TARGETS. Encounters are chosen by how often the model LOSES them (an EMA over
    in-run and curriculum fights), and the deck server is told which one to play (DeckServe's
    "encounter" field). Queen and Aeonglass get drilled; the Test Subject, which it already beats,
    does not.

The deck server runs a synthetic run, so its observation's run context (floor, gold, boss,
character) is wrong for the harvested deck. ``patch_context`` restores it from the record before
the model sees the observation.
"""
from __future__ import annotations

import collections
import copy
import math
import random

import torch
import torch.nn.functional as F

from . import features as FT
from . import ppo as P
from .net import to_torch

ROOM_OF = {"Boss": "boss", "Elite": "elite", "Monster": "regular"}


class EncounterStats:
    """EMA win rate per encounter, from every fight the model plays (in-run and curriculum)."""

    def __init__(self, beta=0.05, prior=0.5):
        self.beta, self.prior = beta, prior
        self.ema = {}
        self.n = collections.Counter()

    def update(self, encounter, won):
        e = self.ema.get(encounter, self.prior)
        self.ema[encounter] = (1 - self.beta) * e + self.beta * float(won)
        self.n[encounter] += 1

    def loss_weight(self, encounter, floor=0.05):
        return (1.0 - self.ema.get(encounter, self.prior)) + floor


def patch_context(obs: dict, rec: dict, vocab_boss_ix: dict) -> dict:
    """The deck server's run context is synthetic; restore the harvested run's."""
    o = dict(obs)
    pl = rec.get("player") or {}
    for k in ("act_floor", "total_floor"):
        if pl.get(k) is not None:
            o[k] = pl[k]
    if pl.get("boss"):
        o["boss"] = pl["boss"]
        o["boss_idx"] = vocab_boss_ix.get(pl["boss"], pl.get("boss_idx", 0))
    if rec.get("character"):
        o["character"] = rec["character"]
    if pl.get("gold") is not None and isinstance(o.get("player"), dict):
        o["player"] = dict(o["player"], gold=pl["gold"])
    return o


def post_fight_obs(rec: dict, hp_end: int, max_hp: int):
    """(kind, obs, legal) of the run decision right after a won fight: the harvested run state
    with the HP the fight left. Scored by the value head, it is the fight's terminal value."""
    pl = copy.deepcopy(rec["player"])
    pl["hp"] = int(hp_end)
    pl["max_hp"] = int(max_hp)
    return "card_reward", pl, [{"kind": "alt", "index": 0, "id": "Skip"}]


def choose_fights(records: list, stats: EncounterStats, n: int, acts=(1, 2),
                  mix=(0.6, 0.25, 0.15), rng: random.Random | None = None,
                  pools: dict | None = None):
    """n (record, room, encounter) picks. Rooms by ``mix`` (boss, elite, regular); within a
    room, encounters by loss weight; the deck is one harvested ENTERING that encounter.

    ``pools`` ((act, room) -> encounter ids, from Vocab.act_pools) restricts targets to what the
    deck server can actually play there. An act's opening "weak" hallway fights are in no pool,
    and asking for one is (correctly) a loud error from the server.
    """
    rng = rng or random.Random(0)
    by = collections.defaultdict(list)            # (act, room) -> records
    by_enc = collections.defaultdict(list)        # encounter -> records
    for r in records:
        if r.get("act") in acts and r.get("room") in ROOM_OF and r.get("player"):
            room = ROOM_OF[r["room"]]
            if pools and r["encounter"] not in pools.get((r["act"], room), ()):
                continue
            by[(r["act"], room)].append(r)
            by_enc[r["encounter"]].append(r)
    picks = []
    rooms = ("boss", "elite", "regular")
    for _ in range(n):
        act = rng.choice([a for a in acts if any(by[(a, rm)] for rm in rooms)] or [None])
        if act is None:
            break
        avail = [(rm, w) for rm, w in zip(rooms, mix) if by[(act, rm)]]
        room = rng.choices([rm for rm, _ in avail], weights=[w for _, w in avail])[0]
        encs = sorted({r["encounter"] for r in by[(act, room)]})
        enc = rng.choices(encs, weights=[stats.loss_weight(e) for e in encs])[0]
        rec = rng.choice(by_enc[enc] or by[(act, room)])
        picks.append((rec, room, enc))
    return picks


def play_fights(net, pools, picks, vocab, boss_ix, dev, bf16=True, loop=1):
    """Plays picks on the per-act deck-server pools. Returns (steps, results).

    results: one (encounter, won, n_steps, v_end, record, hp_end, max_hp) per fight, in the order
    its steps appear (fights finish out of order, so this is NOT the order of ``picks``).

    ``pools``: act -> VecDeckEval (launched with that act's encounter pools).
    """
    from ..deckeval import DeckSpec
    steps_all, results = [], []
    for act, pool in pools.items():
        mine = [(r, rm, e) for r, rm, e in picks if r["act"] == act]
        if not mine:
            continue
        specs, meta = [], {}
        for rec, room, enc in mine:
            sp = DeckSpec(rec["cards"], rec["upgrades"], rec["relics"], rec["character"],
                          fights_per_room=1, rooms=(room,), tag=enc)
            sp.encounter = enc
            sp.hp_frac = max(0.05, min(1.0, rec["hp"] / max(1, rec["max_hp"])))
            specs.append(sp)
            meta[id(sp)] = rec
        open_steps = collections.defaultdict(list)
        env_rec = {}

        def policy(obs_list, legal_list, idx_list, _open=open_steps):
            # The env's current spec is not passed in; the context patch uses the record that
            # was last assigned to that env (set in on_start below via pool bookkeeping).
            obs_p = [patch_context(o, env_rec.get(i, {}), boss_ix) for o, i in zip(obs_list, idx_list)]
            b = to_torch(FT.encode_batch([("combat", o, l) for o, l in zip(obs_p, legal_list)], vocab), dev)
            net.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                out = net(b, loop=loop)
            logits = out["logits"].float()
            acts = []
            for j, env in enumerate(idx_list):
                n = len(legal_list[j])
                lp = F.log_softmax(logits[j, :n], -1)
                a = int(torch.multinomial(lp.exp(), 1))
                st = P.Step(kind="combat", obs=obs_p[j], legal=legal_list[j], a=a,
                            logp=float(lp[a]), v=float(out["value"][j]), loop=loop,
                            floor=P.floor_of("combat", obs_p[j]))
                _open[env].append(st)
                acts.append(a)
            return acts

        def on_start(env, spec, room):
            env_rec[env] = meta[id(spec)]

        def on_terminal(env, spec, room, msg, _open=open_steps):
            rec = meta[id(spec)]
            steps = _open.pop(env, [])
            won = bool(msg.get("won"))
            mx = max(1, int(msg.get("max_hp", 1) or 1))
            v_end = 0.0
            if won:
                k, o, l = post_fight_obs(rec, msg.get("hp_end", 0), mx)
                b = to_torch(FT.encode_batch([(k, o, l)], vocab), dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    v_end = float(net(b, loop=loop)["value"][0])
            for st in steps:
                st.fight_won = int(won)
                st.fight_hp = (rec["hp"] - int(msg.get("hp_end", 0) or 0)) / mx
                st.boss_fight = room == "boss"
            P.compute_targets(steps, False, 1.0, 0.95, v_end=v_end)
            steps_all.extend(steps)
            # The steps of one fight are contiguous in steps_all, in results order.
            results.append((spec.encounter, won, len(steps), v_end, rec,
                            int(msg.get("hp_end", 0) or 0), mx))

        pool.evaluate(specs, policy, on_terminal=on_terminal, with_idx=True, on_start=on_start)
    return steps_all, results
