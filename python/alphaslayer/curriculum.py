"""Combat curriculum on EXACT harvested decks: the shared engine behind both the standalone
trainer (train_combat_decks.py) and the automatic passes inside train_run.py.

Why it exists. The combat net is behaviour-cloned and PPO-trained on act 1, and inside a run
it meets act-2 fights only in the runs that get that far -- 1.5% of them, before any of this.
So it cannot learn act 2 from run data: chicken-and-egg. Teaching it act-2/3 encounters in
isolation broke a plateau six reward configurations could not move (act-2 clears 0.014 ->
0.056 -> 0.110 over two rounds).

Two details are load-bearing, both learned the hard way:

  * DECKS MUST BE REAL. The first attempt trained on Pandora's-Box rerolls (20+ random cards,
    two strong relics). It taught act-2 skill and destroyed act-1 skill: the realistic act-1
    benchmark fell 0.765 -> 0.680 and runs reaching act 2 fell 77% -> 55%. Replaying decks
    HARVESTED from live runs kept it at 0.767, i.e. unchanged.
  * HP MUST BE REAL. deckserve heals to full each episode, which is right for measuring a
    deck and wrong for training: the run policy arrives in act 2 at ~35% hp, and how much
    risk a play is worth depends on what is left.

The helpers a caller must supply (combat_forward, sample, ppo_update) are passed in rather
than imported, so this module stays free of a circular dependency on train_run.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict

# fight_end room names, as written into the harvest -> the room ids deckserve understands.
ROOM_MAP = {"Boss": "boss", "Elite": "elite", "Monster": "regular"}
ROOMS = ("boss", "elite", "regular")


def potential_parts(obs):
    """(player hp fraction, mean living-enemy hp fraction) -- the two halves of phi.

    Returned separately because the terminal potential of a LOST fight is -enemy_hp_frac,
    which the caller cannot recover from phi alone.
    """
    p = obs.get("player", {})
    php = (p.get("hp", 0) or 0) / max(1, p.get("max_hp", 1) or 1)
    alive = [e for e in (obs.get("enemies") or []) if e.get("alive")]
    ehp = (sum(e.get("hp", 0) / max(1, e.get("max_hp", 1) or 1) for e in alive) / len(alive)
           if alive else 0.0)
    return php, ehp


def potential(obs):
    """phi(s) = player hp fraction - mean living-enemy hp fraction.

    Byte-identical to the shaping potential in train_run.py and train_rl.py, so a checkpoint
    trained here is optimising the same combat objective it was trained on elsewhere.
    """
    p = obs.get("player", {})
    php = (p.get("hp", 0) or 0) / max(1, p.get("max_hp", 1) or 1)
    alive = [e for e in (obs.get("enemies") or []) if e.get("alive")]
    ehp = (sum(e.get("hp", 0) / max(1, e.get("max_hp", 1) or 1) for e in alive) / len(alive)
           if alive else 0.0)
    return php - ehp


def load_decks(path, act=None, min_deck=6, recent=0, skip=0):
    """Harvest records -> usable deck records, optionally filtered to one 0-based act.

    ``skip`` drops the first N RAW records, which is how a caller consumes only what has been
    harvested since its last pass. This is the freshness control that matters, and a record
    COUNT cap is not a substitute: the acts fill the harvest at wildly different rates
    (measured at 196 / 41 / 6.3 decks per iteration for acts 1/2/3), so a uniform cap of
    6000 bound act 1 to its last ~31 iterations while acts 2 and 3 never reached it and
    trained on the entire run's history -- including decks from the weakest early policy.
    That is backwards: acts 2 and 3 are where the policy is changing fastest, and the decks
    it brings there change with it (the act-3 entry deck went 19.1 -> 26.6 cards between two
    rounds).

    ``recent`` remains as a safety valve on memory, applied after filtering.

    Records predating the `act` field are dropped whenever an act filter is requested, so an
    old harvest cannot silently masquerade as act-2 data.
    """
    out = []
    try:
        fh = open(path)
    except OSError:
        return out
    with fh:
        for n, line in enumerate(fh):
            if n < skip:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if ROOM_MAP.get(r.get("room")) is None or len(r.get("cards") or []) < min_deck:
                continue
            if act is not None and r.get("act") != act:
                continue
            out.append(r)
    return out[-recent:] if recent else out


def by_room(recs):
    d = defaultdict(list)
    for r in recs:
        d[ROOM_MAP[r["room"]]].append(r)
    return d


def train_pass(net, opt, decks_by_act, *, device, combat_forward, sample, ppo_update,
               pargs, ref=None, iters=25, fights_per_iter=200, mix=(0.5, 0.25, 0.25),
               envs=20, real_hp=True, hp_bonus=0.5, shaping=0.5, seed=1234, log=print,
               keep_best=False, ema_beta=0.3):
    """PPO on harvested decks across ALL acts at once. Returns a per-act stats dict.

    ``decks_by_act`` maps a 0-based act to its harvested deck records.

    Two decisions here are the whole point, and they are different from each other:

    * EVERY act is trained in EVERY pass, not one act per pass. Training one distribution to
      convergence and then another is blocked training, which is precisely the regime that
      made the first curriculum attempt trade act-1 skill for act-2 skill (realistic act-1
      benchmark 0.765 -> 0.680). Interleaving is the standard mitigation, so each update sees
      all three.

    * The three acts contribute EQUAL GRADIENT MAGNITUDE, and that is achieved in the loss,
      not by throwing data away. Harvested decks are wildly unbalanced -- roughly 8924 act-1,
      1739 act-2, 262 act-3 -- so pooling them and sampling uniformly would give act 3 about
      3% of the signal and the rare acts would be drowned by act 1's sheer volume. Sampling
      act 1 down to act 3's size would instead discard most of the real data. So every step
      is TAGGED with its act and ppo_update averages the surrogate per tag before averaging
      across tags: act 1 keeps all of its decks and all of its fights, and still counts once.

    Fight counts per act are allocated on sqrt(deck count), a compromise between proportional
    (which maximises coverage of act 1's large pool) and uniform (which keeps enough act-3
    samples that its up-weighted gradient is not pure noise).
    """
    import math

    from .deckeval import DeckSpec, VecDeckEval          # local: keeps import cost off callers

    decks_by_act = {a: d for a, d in decks_by_act.items() if d}
    if not decks_by_act:
        log("  curriculum: no harvested decks for any act; skipped")
        return {}
    acts = sorted(decks_by_act)
    rooms_by_act = {a: by_room(d) for a, d in decks_by_act.items()}
    chars = sorted({r["character"] for d in decks_by_act.values() for r in d})

    # ONE pool per act, holding every character. --probe-act is fixed at process launch so
    # acts cannot share a pool, but characters can -- and must. With a pool per (character,
    # act) the evaluator drove one character's 2 envs at a time while the other 28 processes
    # spun a full core each (--fixed-fps 5000 never sleeps), so 2 of 30 processes did useful
    # work, the busy ones were starved of CPU, and fights timed out. A mixed pool puts every
    # character's fights in one queue and keeps the whole pool busy.
    per_pool = max(len(chars), envs // max(1, len(acts)))
    weights = {a: math.sqrt(len(decks_by_act[a])) for a in acts}
    wtot = sum(weights.values())
    fights_for = {a: max(10, int(round(fights_per_iter * weights[a] / wtot))) for a in acts}

    # Renormalise the room mix over the rooms each act actually HAS decks for. There is one
    # boss fight per act per run, so a fresh window often holds no act-3 boss deck at all;
    # drawing against the nominal 0.5/0.25/0.25 and skipping the misses silently spent ~75%
    # of act 3's fight allocation on nothing, and trained it almost entirely on monsters
    # while its reported win rate looked healthy.
    room_mix = {}
    for a in acts:
        avail = [(rm, w) for rm, w in zip(ROOMS, mix) if rooms_by_act[a].get(rm)]
        tot = sum(w for _, w in avail) or 1.0
        room_mix[a] = ([rm for rm, _ in avail], [w / tot for _, w in avail])

    log(f"  curriculum: acts {[a + 1 for a in acts]}, "
        + ", ".join(
            f"act{a + 1}={len(decks_by_act[a])}decks"
            f"({'/'.join(f'{rm[0]}{len(rooms_by_act[a][rm])}' for rm in ROOMS if rooms_by_act[a].get(rm))})"
            f"/{fights_for[a]}fights" for a in acts)
        + f", {per_pool} envs x {len(acts)} pools ({len(chars)} characters mixed)")

    pools = {}
    rng = random.Random(seed)

    def pool_for(act):
        # Game processes crash mid-fight and are not replaced, so a pool bleeds envs over a
        # pass: once every env of one character has died, that character's fights become
        # undeliverable (measured at 8/10 envs alive after 3 iterations, losing ~15 act-1
        # fights each). Recreating the pool costs ~10s and restores the full roster, which is
        # cheap against a 25-iteration pass.
        pl = pools.get(act)
        if pl is not None:
            alive = sum(1 for c in pl.conns if c.alive)
            if alive < max(len(chars), int(0.7 * pl.n_envs)):
                log(f"  curriculum: act {act + 1} pool down to {alive}/{pl.n_envs} envs; "
                    f"recreating")
                try:
                    pl.close()
                except Exception:
                    pass
                pools.pop(act, None)
        if act not in pools:
            # 90s, not the 25s measurement default: acts are still evaluated in turn, so a
            # pool waits while the others run and its processes can be swapped out. Treating
            # a slow wake-up as a stall cost ~65% of act-2 fights.
            pools[act] = VecDeckEval(n_envs=per_pool, characters=chars,
                                     seed=f"CUR{act}",
                                     out_dir=f"/tmp/alphaslayer_cur{act}_",
                                     act=act, stall_timeout=90.0)
        return pools[act]

    totals = {a: {"fights": 0, "wins": 0, "hp": 0.0} for a in acts}
    # Optional: end the pass on its BEST measured weights rather than its last ones.
    #
    # Measured over 15 passes of round 5: KL to the reference grows ~11% within a pass, higher
    # KL tracks a worse win rate for every act (t = -2.2 / -2.5 / -2.8), and the best iteration
    # sits 0.30-0.39 of the way through. A flat-with-noise pass would put its peak at 0.50 on
    # average, so the final weights are systematically not the pass's best weights.
    #
    # The score is EMA-smoothed before selecting: the raw per-iteration win rate is measured
    # on 25-114 fights, so picking its argmax would mostly select a lucky sample rather than a
    # better policy. The snapshot is taken BEFORE each update, since those are the weights
    # that produced that iteration's measurement.
    best = {"ema": None, "state": None, "iter": 0}
    ema = None
    try:
        for i in range(1, iters + 1):
            t0 = time.time()
            all_steps = []
            per_iter = {a: {"n": 0, "w": 0} for a in acts}

            for act in acts:
                picks = defaultdict(list)
                room_names, room_ws = room_mix[act]
                for _ in range(fights_for[act]):
                    room = rng.choices(room_names, weights=room_ws, k=1)[0]
                    r = rng.choice(rooms_by_act[act][room])
                    picks[r["character"]].append((r, room))

                open_steps = {}
                tag = f"act{act + 1}"

                def policy(obs_list, legal_list, idx_list, _tag=tag, _open=open_steps):
                    import torch
                    with torch.no_grad():
                        logits, value, mask = combat_forward(net, obs_list, legal_list, device)
                        idx, logp = sample(logits, mask)
                    out = []
                    for j, env in enumerate(idx_list):
                        out.append(int(idx[j]))
                        php, ehp = potential_parts(obs_list[j])
                        _open.setdefault(env, []).append({
                            "obs": obs_list[j], "legal": legal_list[j], "kind": _tag,
                            "a": int(idx[j]), "logp": float(logp[j]), "v": float(value[j]),
                            "cphi": php - ehp, "e_last": ehp,
                        })
                    return out

                def on_terminal(env, spec, room, msg, _act=act, _open=open_steps):
                    mx = max(1, int(msg.get("max_hp", 1) or 1))
                    frac = (msg.get("hp_end", 0) or 0) / mx
                    won = bool(msg.get("won"))
                    # Same credit as train_run.Buffer.finish_fight -- and phi_end for a LOSS
                    # must be -e_last, the enemy hp still standing, NOT 0.
                    #
                    # With 0 the shaping term for a lost fight is -shaping*cphi, and cphi is
                    # player_hp_frac - enemy_hp_frac. Act-3 decks enter at ~32% hp against
                    # full-hp enemies, so cphi ~ -0.68 and every step of a LOST fight earned
                    # +0.34 -- a reward that grew the further behind the agent was. Act-1
                    # decks enter near full hp (cphi ~ 0) and got almost none of it, which is
                    # exactly the order of the decay measured over a pass: act 3 -0.27,
                    # act 1 -0.03.
                    steps_open = _open.pop(env, [])
                    e_last = steps_open[-1].get("e_last", 0.0) if steps_open else 0.0
                    ret = float(won) + hp_bonus * frac
                    phi_end = frac if won else -e_last
                    for st in steps_open:
                        st["ret"] = ret + shaping * (phi_end - st["cphi"])
                        all_steps.append(st)
                    per_iter[_act]["n"] += 1
                    per_iter[_act]["w"] += int(won)
                    totals[_act]["hp"] += frac

                # ONE evaluate call for the whole act: every character's fights go into the
                # same queue so the pool's envs all work at once, instead of one character's
                # envs working while the rest idle.
                specs = []
                for character, items in picks.items():
                    for r, room in items:
                        sp = DeckSpec(r["cards"], r["upgrades"], r["relics"], character,
                                      fights_per_room=1, rooms=(room,), tag=room)
                        if real_hp:
                            sp.hp_frac = max(0.05, min(1.0, (r.get("hp", 0) or 0)
                                                       / max(1, r.get("max_hp", 1) or 1)))
                        specs.append(sp)
                net.eval()
                if specs:
                    pool_for(act).evaluate(specs, policy, on_terminal=on_terminal,
                                           with_idx=True)
                net.train()
                # Fights still in flight have no outcome; an invented reward would poison the
                # value target, so they are dropped rather than credited.
                open_steps.clear()

            collect = time.time() - t0
            if keep_best and any(per_iter[a]["n"] for a in acts):
                # Equal weight per act, matching how the loss weights them.
                score = sum(per_iter[a]["w"] / max(1, per_iter[a]["n"]) for a in acts) / len(acts)
                ema = score if ema is None else ema_beta * score + (1 - ema_beta) * ema
                # Warm up before selecting. The first iteration's EMA is just its raw score
                # with no history behind it, so it wins on noise more often than it should --
                # and "best = iteration 1" means discarding the whole pass. Three iterations
                # of smoothing before anything is eligible.
                if i >= 3 and (best["ema"] is None or ema > best["ema"]):
                    import copy
                    best = {"ema": ema, "iter": i,
                            "state": copy.deepcopy({k: v.detach().cpu()
                                                    for k, v in net.state_dict().items()})}
            if not all_steps:
                log(f"  curriculum {i}/{iters}: no completed fights")
                continue
            t1 = time.time()
            stats = ppo_update(net, opt, all_steps, device, pargs, is_combat=True, ref=ref)
            for a in acts:
                totals[a]["fights"] += per_iter[a]["n"]
                totals[a]["wins"] += per_iter[a]["w"]
            detail = " ".join(
                f"a{a + 1}={per_iter[a]['w'] / max(1, per_iter[a]['n']):.2f}"
                f"({per_iter[a]['n']})" for a in acts)
            log(f"  curriculum {i}/{iters}: {detail} | pg={stats['pg']:+.4f} "
                f"vf={stats['vf']:.3f} kl={stats['kl']:.4f} steps={len(all_steps)} "
                f"collect {collect:.0f}s train {time.time() - t1:.0f}s")
        if keep_best and best["state"] is not None and best["iter"] < iters:
            net.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
            log(f"  curriculum: kept the best-EMA weights from iteration {best['iter']}/{iters} "
                f"(score {best['ema']:.3f}) instead of the last ones")
    finally:
        for pl in pools.values():
            try:
                pl.close()
            except Exception:
                pass
    return {a: {"fights": t["fights"],
                "win": t["wins"] / max(1, t["fights"]),
                "hp": t["hp"] / max(1, t["fights"])} for a, t in totals.items()}
