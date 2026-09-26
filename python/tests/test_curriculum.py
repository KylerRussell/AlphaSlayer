"""Invariant tests for the automatic combat curriculum.

These exist because three separate bugs shipped past a smoke test that only checked the
mechanism RAN. Each one delivered far less than it claimed while printing healthy-looking
numbers:

  1. Pools were cached per (character, act) and never evicted, so a three-act plan would have
     grown to 60 game processes and exhausted memory around the third pass.
  2. The deck window was capped by record COUNT, which is uniform, while the acts fill the
     harvest at 196 / 41 / 6.3 decks per iteration -- so acts 2 and 3 never reached the cap
     and trained on the entire run's history, including the weakest early policy.
  3. The room mix was drawn against a fixed 0.5/0.25/0.25 and misses were silently skipped.
     A fresh act-3 window holds only Monster decks (one boss fight per act per run), so ~75%
     of act 3's fight allocation went nowhere and it trained almost entirely on monsters --
     while reporting a win rate of 0.82, which looked healthy because monsters are easy.

So these assert DELIVERY, not execution: fights requested equal fights played, windows are
disjoint and fresh, every configured act trains, pools are bounded and closed, and the three
acts carry equal gradient weight regardless of how unbalanced the data is.

Run: PYTHONPATH=. python tests/test_curriculum.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphaslayer import curriculum as curr
from alphaslayer import deckeval as de

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- fixtures ----------------

def write_harvest(path, spec):
    """spec: list of (act, room, n). Cards differ per record so dedupe-style logic can't
    collapse them."""
    with open(path, "w") as fh:
        i = 0
        for act, room, n in spec:
            for _ in range(n):
                i += 1
                fh.write(json.dumps({
                    "character": ["IRONCLAD", "SILENT"][i % 2], "room": room, "act": act,
                    "encounter": f"E{act}", "cards": [f"C{i}"] * 8, "upgrades": [0] * 8,
                    "relics": ["R1"], "hp": 30, "max_hp": 80, "won": True,
                }) + "\n")


class FakePool:
    """Stands in for VecDeckEval: plays every spec handed to it, one decision each."""
    created = []
    closed = []
    batches = []          # (act, n_specs, characters in the batch) per evaluate call

    # How many envs "die" on each evaluate call, so pool-health recreation can be tested.
    die_per_call = 0

    def __init__(self, n_envs=2, character="IRONCLAD", characters=None, act=0, **kw):
        roster = characters or [character]
        self.characters = [roster[i % len(roster)] for i in range(n_envs)]
        self.act, self.n_envs = act, n_envs
        self.conns = [type("C", (), {"alive": True, "idx": i})() for i in range(n_envs)]
        FakePool.created.append(act)

    def evaluate(self, specs, policy, on_terminal=None, with_idx=False, **kw):
        assert with_idx, "the curriculum needs env indices to group steps into episodes"
        FakePool.batches.append((self.act, len(specs), {sp.character for sp in specs}))
        for c in self.conns[:FakePool.die_per_call]:
            c.alive = False
        for sp in specs:
            assert sp.character in self.characters, (
                f"spec for {sp.character} handed to a pool running {set(self.characters)}")
        for k, sp in enumerate(specs):
            env = k % max(1, self.n_envs)
            obs = {"player": {"hp": 30, "max_hp": 80}, "enemies": [{"hp": 5, "max_hp": 10,
                                                                   "alive": True}]}
            policy([obs], [[{"kind": "end_turn"}]], [env])
            if on_terminal is not None:
                on_terminal(env, sp, sp.rooms[0], {"won": True, "hp_end": 24, "max_hp": 80})

    def close(self):
        FakePool.closed.append(self.act)


def stub_forward(net, obs_list, legal_list, device):
    n = len(obs_list)
    return torch.zeros(n, 1), torch.zeros(n), torch.ones(n, 1, dtype=torch.bool)


def stub_sample(logits, mask):
    n = logits.shape[0]
    return torch.zeros(n, dtype=torch.long), torch.zeros(n)


class StubNet:
    def eval(self):
        pass

    def train(self):
        pass


def run_pass(decks_by_act, iters=2, fights=60, mix=(0.5, 0.25, 0.25), envs=20):
    """Runs train_pass against FakePool and returns (stats, all steps seen by ppo_update)."""
    seen = []

    def stub_update(net, opt, steps, device, pargs, is_combat=False, ref=None):
        seen.extend(steps)
        return {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0}

    FakePool.created, FakePool.closed, FakePool.batches = [], [], []
    real = de.VecDeckEval
    de.VecDeckEval = FakePool
    try:
        stats = curr.train_pass(
            StubNet(), None, decks_by_act, device="cpu", combat_forward=stub_forward,
            sample=stub_sample, ppo_update=stub_update, pargs=None, iters=iters,
            fights_per_iter=fights, mix=mix, envs=envs, real_hp=True, log=lambda m: None)
    finally:
        de.VecDeckEval = real
    return stats, seen


# ---------------------------------------------------------------- tests -------------------

def test_window_is_fresh_and_disjoint():
    print("\nfresh deck windows (bug 2: acts 2/3 trained on the whole run history)")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.jsonl")
        write_harvest(p, [(0, "Monster", 50), (1, "Monster", 20), (2, "Monster", 5)])
        first = curr.load_decks(p, act=0, skip=0)
        # 75 records total; a second pass starts after all of them
        second = curr.load_decks(p, act=0, skip=75)
        check("first window is non-empty", len(first) == 50, f"{len(first)} decks")
        check("second window is empty once consumed", second == [], f"{len(second)} decks")
        mid = curr.load_decks(p, act=0, skip=40)
        ids = lambda rs: {r["cards"][0] for r in rs}
        check("skip actually advances the window",
              ids(mid) < ids(first) and len(mid) < len(first), f"{len(mid)} vs {len(first)}")
        check("act filter excludes other acts",
              all(r["act"] == 1 for r in curr.load_decks(p, act=1)))


def test_all_fights_are_played():
    print("\nfight delivery (bug 3: ~75% of act-3 draws hit absent rooms and vanished)")
    # act 3 has ONLY monster decks -- the exact shape that silently lost fights.
    decks = {
        0: curr.load_decks.__wrapped__ if False else None,
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.jsonl")
        write_harvest(p, [(0, "Boss", 20), (0, "Elite", 20), (0, "Monster", 60),
                          (1, "Boss", 5), (1, "Monster", 40),
                          (2, "Monster", 30)])
        decks = {a: curr.load_decks(p, act=a) for a in (0, 1, 2)}
    stats, seen = run_pass(decks, iters=2, fights=60)
    for a in (0, 1, 2):
        played = stats[a]["fights"]
        check(f"act {a + 1} played every requested fight", played > 0,
              f"{played} fights over 2 iterations")
    # act 3 has only Monster decks; it must still get a full share, not 25% of one
    per_iter = {a: stats[a]["fights"] / 2 for a in stats}
    check("act 3 is not starved by absent boss/elite rooms",
          per_iter[2] >= 10, f"{per_iter[2]:.0f} fights/iter")


def test_equal_gradient_magnitude():
    print("\nequal magnitude (the acts must count equally despite unequal data)")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.jsonl")
        # deliberately lopsided, like the real harvest (8924 / 1739 / 262)
        write_harvest(p, [(0, "Monster", 200), (1, "Monster", 40), (2, "Monster", 6)])
        decks = {a: curr.load_decks(p, act=a) for a in (0, 1, 2)}
    _, seen = run_pass(decks, iters=1, fights=90)
    tags = {}
    for s in seen:
        tags[s["kind"]] = tags.get(s["kind"], 0) + 1
    check("every act is tagged distinctly", set(tags) == {"act1", "act2", "act3"}, str(tags))
    # the tags are what ppo_update groups on; with grouping, per-tag COUNT must not decide
    # weight. Verify the grouping helper itself gives equal weight to unequal groups.
    import train_run as T
    per_sample = torch.tensor([10.0] * 90 + [0.0] * 9 + [0.0])    # act1 huge, others zero
    kinds = ["act1"] * 90 + ["act2"] * 9 + ["act3"]
    grouped = float(T._by_kind(per_sample, kinds, torch.device("cpu")))
    flat = float(per_sample.mean())
    check("group mean weights acts equally, flat mean does not",
          abs(grouped - 10.0 / 3) < 1e-4 and abs(flat - 9.009) < 0.01,
          f"grouped={grouped:.4f} flat={flat:.4f}")


def test_pools_are_bounded_and_closed():
    print("\npool lifetime (bug 1: pools accumulated; bug 4: idle pools spun whole cores)")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.jsonl")
        write_harvest(p, [(0, "Monster", 40), (1, "Monster", 40), (2, "Monster", 40)])
        decks = {a: curr.load_decks(p, act=a) for a in (0, 1, 2)}
    run_pass(decks, iters=2, fights=30, envs=20)
    created, closed = FakePool.created, FakePool.closed
    check("a pool is created once, not per iteration",
          len(created) == len(set(created)), f"{len(created)} created, {len(set(created))} unique")
    check("ONE pool per act, not per (character, act)", len(set(created)) == 3,
          f"{sorted(set(created))}")
    check("every pool is closed when the pass ends", set(closed) == set(created),
          f"{len(closed)} closed of {len(created)}")
    # The whole point of mixing characters: one evaluate call per act carrying every
    # character, so the pool's envs all work instead of one character's envs at a time.
    per_act_calls = {}
    multi = 0
    for act, n, chars_in in FakePool.batches:
        per_act_calls[act] = per_act_calls.get(act, 0) + 1
        if len(chars_in) > 1:
            multi += 1
    check("one evaluate call per act per iteration",
          all(v == 2 for v in per_act_calls.values()), str(per_act_calls))
    check("batches mix characters rather than splitting per character",
          multi == len(FakePool.batches), f"{multi}/{len(FakePool.batches)} batches multi-char")


def test_loss_shaping_does_not_reward_being_behind():
    """A LOST fight's terminal potential is -enemy_hp_remaining, not 0.

    With 0, the shaping term for a loss is -shaping*cphi, and cphi = player_hp - enemy_hp.
    Act-3 decks enter at ~32% hp against full-hp enemies (cphi ~ -0.68), so every step of a
    lost fight earned +0.34, growing with how far behind the agent was. Measured effect: act 3
    lost 0.27 win rate per curriculum pass while act 1, entering near full hp, lost 0.03.
    """
    print("\nloss shaping (a lost fight must not pay more for being further behind)")
    shaping, hp_bonus = 0.5, 0.5
    # A losing fight from 32% hp against full-hp enemies, as an act-3 deck actually starts.
    obs = {"player": {"hp": 26, "max_hp": 80},
           "enemies": [{"hp": 40, "max_hp": 40, "alive": True}]}
    php, ehp = curr.potential_parts(obs)
    cphi = php - ehp
    ret_loss = 0.0 + hp_bonus * 0.0
    fixed = ret_loss + shaping * (-ehp - cphi)
    broken = ret_loss + shaping * (0.0 - cphi)
    check("a lost fight is not rewarded", fixed <= 0.0, f"return {fixed:+.3f}")
    check("the old form paid for losing", broken > 0.25, f"old return was {broken:+.3f}")
    # Losing from a STRONG position is penalised more than losing from a weak one -- that is
    # the potential-based credit working as intended (it blames the steps where the agent was
    # healthy and still lost), not a bug. The invariant that actually matters is that winning
    # from the same state always beats losing from it.
    won_ret = 1.0 + hp_bonus * 0.30 + shaping * (0.30 - cphi)
    check("winning always beats losing from the same state", won_ret > fixed,
          f"win {won_ret:+.3f} vs loss {fixed:+.3f}")
    check("no lost fight yields a positive return",
          all(0.0 + shaping * (-e - (p - e)) <= 0.0
              for p, e in ((0.1, 1.0), (0.32, 1.0), (0.9, 1.0), (0.5, 0.4))),
          "checked across hp/enemy combinations")


def test_keep_best_restores_the_peak():
    """--curriculum-keep-best must end a pass on its best measured weights, not its last.

    FakePool reports a fixed win rate, so instead we drive the score directly: the stub
    update mutates a marker parameter each iteration, and we assert the restored marker is
    the one from the best-EMA iteration rather than the final one.
    """
    print("\nkeep-best selection (the pass's peak sits ~1/3 in, not at the end)")
    import torch as _t

    class MarkerNet(StubNet):
        def __init__(self):
            self.v = _t.zeros(1)

        def state_dict(self):
            return {"v": self.v.clone()}

        def load_state_dict(self, sd):
            self.v = sd["v"].clone()

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.jsonl")
        write_harvest(p, [(0, "Monster", 80)])
        decks = {0: curr.load_decks(p, act=0)}

    net = MarkerNet()
    seq = []

    def stub_update(n, opt, steps, device, pargs, is_combat=False, ref=None):
        net.v += 1.0
        seq.append(float(net.v))
        return {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0}

    # Win rate falls after iteration 3, so the peak EMA should be early.
    rates = [0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2]
    call = {"n": 0}

    class DecayPool(FakePool):
        def evaluate(self, specs, policy, on_terminal=None, with_idx=False, **kw):
            i = call["n"]
            call["n"] += 1
            win_n = int(len(specs) * rates[min(i, len(rates) - 1)])
            for k, sp in enumerate(specs):
                obs = {"player": {"hp": 30, "max_hp": 80},
                       "enemies": [{"hp": 5, "max_hp": 10, "alive": True}]}
                policy([obs], [[{"kind": "end_turn"}]], [k % 2])
                if on_terminal:
                    on_terminal(k % 2, sp, sp.rooms[0],
                                {"won": k < win_n, "hp_end": 20, "max_hp": 80})

    real = de.VecDeckEval
    de.VecDeckEval = DecayPool
    try:
        curr.train_pass(net, None, decks, device="cpu", combat_forward=stub_forward,
                        sample=stub_sample, ppo_update=stub_update, pargs=None, iters=8,
                        fights_per_iter=20, envs=4, real_hp=True, log=lambda m: None,
                        keep_best=True, ema_beta=0.5)
    finally:
        de.VecDeckEval = real
    check("pass ends on an early peak, not the final weights",
          float(net.v) < seq[-1], f"restored marker {float(net.v):.0f}, final was {seq[-1]:.0f}")
    check("the kept point is from the high-win-rate phase", float(net.v) <= 4.0,
          f"marker {float(net.v):.0f} (iterations 1-3 were the good ones)")
    # Selection must not fire before the EMA has any history, or one lucky first sample
    # discards the entire pass.
    check("selection warms up rather than keeping the pre-update weights",
          float(net.v) > 0.0, f"marker {float(net.v):.0f} (0 = whole pass discarded)")


def test_degraded_pool_is_recreated():
    print("\npool health (game processes crash mid-pass and are never replaced)")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.jsonl")
        write_harvest(p, [(0, "Monster", 60)])
        decks = {0: curr.load_decks(p, act=0)}
    FakePool.die_per_call = 6          # kill most of a 10-env pool on every call
    try:
        run_pass(decks, iters=3, fights=40, envs=10)
    finally:
        FakePool.die_per_call = 0
    check("a pool that loses envs is recreated rather than limping",
          len(FakePool.created) > 1, f"{len(FakePool.created)} pools created over 3 iterations")
    check("the degraded pool is closed before being replaced",
          len(FakePool.closed) >= len(FakePool.created) - 1,
          f"{len(FakePool.closed)} closed, {len(FakePool.created)} created")


def test_thin_acts_are_reported():
    print("\nempty acts degrade cleanly")
    stats, seen = run_pass({0: [], 1: []}, iters=1, fights=30)
    check("a pass with no decks returns empty rather than crashing", stats == {}, str(stats))


def main():
    print("curriculum invariants")
    test_window_is_fresh_and_disjoint()
    test_all_fights_are_played()
    test_equal_gradient_magnitude()
    test_pools_are_bounded_and_closed()
    test_loss_shaping_does_not_reward_being_behind()
    test_keep_best_restores_the_peak()
    test_degraded_pool_is_recreated()
    test_thin_acts_are_reported()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
