"""M4: PPO for the unified model over whole runs, win-only (see alphaslayer/unified/ppo.py).

Starts from a distilled checkpoint (M3). Per iteration:
  1. play runs with the current model (sampled, temperature 1), recording each decision's
     log-prob, value and loop count;
  2. close each run into the buffer: win-only returns, GAE with shaping in the advantage only;
  3. PPO epochs over the iteration's decisions, replaying the RECORDED loop count of every
     decision, with a KL term toward the distilled anchor that anneals to zero;
  4. log what localises problems: per-act clears, per-decision-kind entropy and top-action share
     (collapse), win-probability calibration on this iteration's decisions, bosses and deaths.

Checkpoints every iteration (model + optimiser + iteration), and --resume continues from one.
A GPU reset therefore costs at most one iteration.

    XDG_DATA_HOME="$(../headless_home.sh)" PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 \\
        HSA_ENABLE_SDMA=0 PYTHONPATH=. .venv7/bin/python train_unified.py \\
        --init unified_m3_d512_121.pt --out unified_m4.pt --iters 300 --bf16
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from multiprocessing.connection import Listener

import numpy as np
import torch
import torch.nn.functional as F

from alphaslayer.gpu_guard import refuse_display_gpu
from alphaslayer.model import use_stable_attention
from alphaslayer.runenv import RunPool, VecRunEnv
from alphaslayer.unified import dataset as DS
from alphaslayer.unified import features as FT
from alphaslayer.unified import curriculum as CU
from alphaslayer.unified import ppo as P
from alphaslayer.unified import search as SR
from alphaslayer.unified.collector import make_policy, replay
from alphaslayer.unified.critic import RunCritic, fight_run_state
from alphaslayer.unified.net import UnifiedNet, choose_loops, param_count, to_torch

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS = ["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"]


def load_net(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = UnifiedNet(ck["sizes"], **ck["cfg"]).to(dev)
    net.load_state_dict(ck["model"])
    return net, ck


def run_critic_potentials(critic, packed, steps, runs, a, bootstrap):
    """Replaces the hand run potential with the critic's P(win), recomputes every run's credit,
    then adds this iteration to the critic's replay and refits. Potentials are predicted
    BEFORE this iteration's outcomes join the replay (no run grades itself). Curriculum steps
    (after the runs) keep their hand potentials. Returns the log fragment."""
    n_run = sum(r["n"] for r in runs)
    st = steps[:n_run]
    act = np.array([s.act for s in st]); floor = np.array([s.floor for s in st])
    dense, key = critic.features(packed.a["ctx_ids"][:n_run], packed.a["ctx_f"][:n_run], act, floor)
    won = np.array([s.won for s in st], np.float32)
    is_c = np.array([s.kind == "combat" for s in st])
    line = "rc=warmup "
    if critic.ready and len(critic.buf) >= a.critic_min_iters:
        run_of = np.repeat(np.arange(len(runs)), [r["n"] for r in runs])
        # fights: the run state "if this fight ended now" (critic.fight_run_state)
        v_phi = critic.predict(*fight_run_state(dense, key, is_c, run_of))
        v, y = critic.predict(dense[~is_c], key[~is_c]), won[~is_c]
        base = ((y.mean() - y) ** 2).mean()
        sk = 1.0 - ((v - y) ** 2).mean() / base if base > 0 else float("nan")   # out of sample
        o = 0
        for info in runs:
            rs = st[o:o + info["n"]]
            for s_, vv in zip(rs, v_phi[o:o + info["n"]]):
                pw = s_.p_fight
                s_.phi = (float((1.0 - a.w_fight * (1.0 - pw)) * vv)
                          if (s_.kind == "combat" and pw is not None) else float(vv))
            P.compute_targets(rs, bool(info["won"]), a.gamma, a.lam, bootstrap=bootstrap,
                              lam_step=a.lam_step)
            o += info["n"]
        line = f"rc_skill={sk:+.3f} "
    critic.add(dense, key, won, is_c)
    critic.fit()
    return line


def dump_steps(out_dir, it, packed, steps, runs):
    """One .npz per iteration: the encoded inputs (DS.pack arrays, prefixed "x_") plus every
    per-step scalar the credit math uses, and the run each step belongs to (runs are contiguous
    in ``steps``, in ``runs`` order; curriculum fights, if any, follow with run = -1)."""
    os.makedirs(out_dir, exist_ok=True)
    run_of = np.full(len(steps), -1, np.int64)
    o = 0
    for r, info in enumerate(runs):
        run_of[o:o + info["n"]] = r
        o += info["n"]
    f = lambda g, dt=np.float32: np.array([g(s) for s in steps], dt)
    nan = float("nan")
    np.savez_compressed(
        os.path.join(out_dir, f"it{it:04d}.npz"),
        **{f"x_{k}": v for k, v in packed.a.items()},
        run=run_of, kind=f(lambda s: FT.DKINDS.index(s.kind), np.int64),
        a=f(lambda s: s.a, np.int64), n_legal=f(lambda s: len(s.legal), np.int64),
        logp=f(lambda s: s.logp), v=f(lambda s: s.v), loop=f(lambda s: s.loop, np.int64),
        floor=f(lambda s: s.floor, np.int64), act=f(lambda s: s.act, np.int64),
        phi=f(lambda s: s.phi), p_fight=f(lambda s: nan if s.p_fight is None else s.p_fight),
        won=f(lambda s: s.won), act_clear=f(lambda s: s.act_clear),
        reach_act3=f(lambda s: s.reach_act3), floors_left=f(lambda s: s.floors_left),
        fight_won=f(lambda s: s.fight_won), fight_hp=f(lambda s: s.fight_hp),
        adv=f(lambda s: s.adv), ret=f(lambda s: s.ret),
        run_won=np.array([r["won"] for r in runs], np.int64),
        run_group=np.array([r.get("group", -1) for r in runs], np.int64),
        run_act=np.array([r["act"] for r in runs], np.int64),
        run_floors=np.array([r["floors"] for r in runs], np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="unified_m3_d512_121.pt", help="distilled starting point")
    ap.add_argument("--anchor", default=None, help="KL anchor checkpoint (default: --init)")
    ap.add_argument("--out", default="unified_m4.pt")
    ap.add_argument("--resume", action="store_true", help="continue from --out if it exists")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--runs-per-env", type=int, default=4, help="96 runs/iter at 24 envs (long1/long2)")
    ap.add_argument("--act-cap", type=int, default=3)
    ap.add_argument("--pool", type=int, default=0,
                    help="N persistent game processes fed from a run queue (RunPool); each "
                         "iteration plays the same envs x runs-per-env (seed, character) jobs as "
                         "the per-iteration launch would. 0 = launch --envs processes per iteration")
    ap.add_argument("--pool-nice", type=int, default=5, help="--pool: game process niceness")
    ap.add_argument("--collector-gpu", type=int, default=-1,
                    help="--pool: run a second collector process (its own policy thread, net "
                         "copy and game pool) on this HIP device; -1 = none")
    ap.add_argument("--collector-slots", type=int, default=20)
    ap.add_argument("--reserve-cpus", default="",
                    help="CPUs the --pool games must also avoid (another arm's trainer cores)")
    ap.add_argument("--trainer-cpus", default="",
                    help="pin the trainer to these CPUs (e.g. 0,1,16,17: two V-cache cores with "
                         "their SMT siblings) and keep --pool games off them")
    ap.add_argument("--recycle-runs", type=int, default=25,
                    help="--pool: restart a game process after this many runs")
    ap.add_argument("--collect-timeout", type=float, default=900.0)
    ap.add_argument("--env-restarts", type=int, default=2,
                    help="relaunch a crashed/wedged game process up to N times per iteration "
                         "for the runs it still owes (0 = drop them, the old behaviour)")
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--aux-coef", type=float, default=0.25)
    ap.add_argument("--kl-coef", type=float, default=0.1, help="KL toward the anchor at iter 1")
    ap.add_argument("--kl-run-mult", type=float, default=5.0,
                    help="multiplies the anchor KL on run decisions only (combat keeps --kl-coef); "
                         "1 let run decisions random-walk in long1")
    ap.add_argument("--adv-center", choices=["half", "kind"], default="kind",
                    help="subtract each half's mean advantage, or each decision kind's. 'half' "
                         "left potion_gate a constant -0.25 that pushed allow 97%% -> 83%% in long1")
    ap.add_argument("--search-rooms", default="",
                    help="M9: fight search (probe turn search) in these room types while collecting "
                         "(e.g. Elite,Boss); searched decisions act from pi' ~ pi exp(q/tau) and the "
                         "policy is distilled toward pi' instead of a PPO step. Needs --pool and a game "
                         "with the search build (--game-dir)")
    ap.add_argument("--search-k", type=int, default=4)
    ap.add_argument("--search-samples", type=int, default=2)
    ap.add_argument("--search-mass", type=float, default=0.95)
    ap.add_argument("--search-tau", type=float, default=0.01)
    ap.add_argument("--distill-coef", type=float, default=1.0)
    ap.add_argument("--no-graphs", dest="graphs", action="store_false",
                    help="collection: eager bf16 policy passes instead of recorded GPU graphs (graphrun.py)")
    ap.add_argument("--batch-timeout", type=float, default=None,
                    help="--pool server: seconds to wait for more requests before a network pass "
                         "(default 0.001 with --search-rooms: rollouts wait on every pass; else 0.004)")
    ap.add_argument("--timer-suspend", action="store_true",
                    help="search polls on a 1 ms timer (old, slower replays) instead of per frame")
    ap.add_argument("--game-dir", default=None, help="game install for the pool (default: the Steam one)")
    ap.add_argument("--anchor-lag", type=int, default=0,
                    help="MOVING anchor: the KL is toward the policy from N iterations ago (the "
                         "start of iteration it-N; the resumed policy until N have passed) instead "
                         "of the fixed --anchor. A fixed anchor held long2/td2 at a KL equilibrium "
                         "of ~0.04-0.1 from the start, so gains could not compound. 0 = fixed")
    ap.add_argument("--accum", type=int, default=1,
                    help="minibatches per optimizer step (gradient accumulation): with K times "
                         "the runs per iteration, K keeps the steps per iteration unchanged and "
                         "averages each step over K times the decisions")
    ap.add_argument("--kl-anneal", type=int, default=0,
                    help="iterations for the KL to reach 0 (0 = held at --kl-coef)")
    ap.add_argument("--gamma", type=float, default=1.0, help="discount PER FLOOR")
    ap.add_argument("--lam", type=float, default=0.8,
                    help="GAE lambda PER FLOOR (credit horizon ~1/(1-lambda) floors)")
    ap.add_argument("--lam-step", type=float, default=1.0,
                    help="trace decay between decisions on the SAME floor (within a fight). 1 = "
                         "credit the fight's final result; <1 = TD on the fight potential "
                         "(see ppo.compute_targets)")
    ap.add_argument("--run-critic", action="store_true",
                    help="run potential = P(win) from a small logistic critic refit each "
                         "iteration on past iterations' outcomes (alphaslayer/unified/critic.py), "
                         "instead of w_act*(act+1) + w_hp*hp")
    ap.add_argument("--critic-window", type=int, default=20, help="iterations of replay")
    ap.add_argument("--critic-min-iters", type=int, default=5,
                    help="iterations of data before the critic replaces the hand potential")
    ap.add_argument("--w-combat", type=float, default=0.0,
                    help="old (player hp%% - enemy hp%%) fight term; not continuous at fight "
                         "boundaries, so off by default")
    ap.add_argument("--w-hp", type=float, default=0.1, help="hp%% in the run potential")
    ap.add_argument("--w-act", type=float, default=1 / 3,
                    help="run potential per act (act+1): ~1 at act 3, matching the win reward")
    ap.add_argument("--w-fight", type=float, default=1.0,
                    help="fight potential = (1 - w*(1 - P(win this fight))) * run potential; "
                         "1 = fully multiplicative, continuous at fight boundaries")
    ap.add_argument("--bootstrap", default="auto", choices=["auto", "on", "off"],
                    help="use the value head in GAE. auto: only once its out-of-sample skill "
                         "(EMA over iterations) is positive; before that, V = 0 in every TD error")
    ap.add_argument("--skill-ema", type=float, default=0.2)
    ap.add_argument("--skill-min-iters", type=int, default=8,
                    help="iterations with a measurable skill before auto-bootstrap may switch on")
    ap.add_argument("--skill-threshold", type=float, default=0.05,
                    help="EMA skill that switches auto-bootstrap on. One lucky 48-run batch "
                         "(skill +0.49) pushed a 0 threshold over in shaped2")
    ap.add_argument("--run-weight", type=float, default=0.5)
    ap.add_argument("--kind-weight", default="flat", choices=["equal", "flat", "sqrt"],
                    help="weighting of run decision kinds within the run half of the loss; "
                         "'equal' let rare kinds random-walk in shaped3 (see ppo.group_mean)")
    ap.add_argument("--loop-policy", default="1", help="'1'..'4' fixed, or 'adaptive'")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed-prefix", default="U4_")
    # ---- combat curriculum (alphaslayer/unified/curriculum.py) ----
    ap.add_argument("--curriculum-every", type=int, default=0,
                    help="every N iterations, add isolated act-2/3 fights on harvested decks, "
                         "targeted at the encounters the model loses most (0 = off)")
    ap.add_argument("--curriculum-fights", type=int, default=120)
    ap.add_argument("--curriculum-acts", default="1,2", help="0-based acts to drill")
    ap.add_argument("--curriculum-mix", default="0.6,0.25,0.15", help="boss,elite,regular")
    ap.add_argument("--curriculum-envs", type=int, default=12, help="deck-server envs per act")
    ap.add_argument("--curriculum-window", type=int, default=6000,
                    help="drill on the most recent N harvested entry decks")
    ap.add_argument("--harvest", default=None,
                    help="JSONL of harvested entry decks (default: <out>.harvest.jsonl)")
    ap.add_argument("--seed-groups", type=int, default=1,
                    help="--pool: play each seed K times per iteration (envs*runs_per_env/K "
                         "distinct seeds)")
    ap.add_argument("--group-baseline", action="store_true",
                    help="subtract the leave-one-out mean advantage of the other plays of the "
                         "same seed at the same floor and decision kind (ppo.seed_group_baseline)")
    ap.add_argument("--seed-block", type=int, default=0,
                    help="--pool: keep a block of --block-seeds seeds for N iterations, each "
                         "played once per character per iteration (replaces envs x runs-per-env)")
    ap.add_argument("--block-seeds", type=int, default=20)
    ap.add_argument("--history-baseline", action="store_true",
                    help="subtract the mean advantage of EARLIER plays of the same seed x "
                         "character at the same floor and kind (ppo.HistoryBaseline)")
    ap.add_argument("--eval-every", type=int, default=50,
                    help="--pool: every N iterations, play the fixed evaluation seeds with the "
                         "current policy (no training on them) and log an 'eval' line; 0 = never. "
                         "With --seed-block it must be a multiple of the block, so an evaluation "
                         "never splits a block's replays")
    ap.add_argument("--eval-sample", action="store_true",
                    help="evaluate with the sampling policy (default: greedy, top action; "
                         "sampling made one checkpoint's act-1 rate vary by 0.03 on identical seeds)")
    ap.add_argument("--eval-iters", type=int, default=5,
                    help="evaluation size in batches of envs x --eval-runs-per-env runs (5 = 480)")
    ap.add_argument("--eval-runs-per-env", type=int, default=4,
                    help="fixed, NOT --runs-per-env, so every arm evaluates on the same seeds")
    ap.add_argument("--eval-only", action="store_true",
                    help="--pool: evaluate the starting checkpoint on the fixed seeds and exit")
    ap.add_argument("--dump-steps", default=None,
                    help="directory: save each iteration's encoded steps + targets (.npz) for "
                         "offline credit/value analysis")
    ap.add_argument("--collect-only", action="store_true",
                    help="collect (and dump) without updating or checkpointing the model")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    if a.seed_block and a.eval_every and a.eval_every % a.seed_block:
        raise SystemExit(f"--eval-every {a.eval_every} would split a {a.seed_block}-iteration "
                         f"seed block; use a multiple of {a.seed_block}")
    if (a.seed_groups > 1 or a.seed_block) and not a.pool:
        raise SystemExit("--seed-groups needs --pool (the run queue assigns seeds per run)")
    use_stable_attention()
    dev = torch.device(a.device)
    refuse_display_gpu(dev)
    vocab = FT.Vocab(json.load(open(os.path.join(HERE, "vocab_m1.json"))))

    net, ck = load_net(a.init, dev)
    anchor, _ = load_net(a.anchor or a.init, dev)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    start = 1
    if a.resume and os.path.exists(a.out):
        st = torch.load(a.out, map_location=dev, weights_only=False)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["iter"] + 1
        print(f"resumed from {a.out} at iteration {start}", flush=True)
    print(f"unified net {param_count(net) / 1e6:.1f}M cfg {net.cfg}; init {a.init}; "
          f"loop policy {a.loop_policy}", flush=True)

    harvest_path = a.harvest or a.out + ".harvest.jsonl"
    harvest = []
    if os.path.exists(harvest_path):
        with open(harvest_path) as fh:
            harvest = [json.loads(l) for l in fh if l.strip()][-a.curriculum_window:]
    stats = CU.EncounterStats()
    last_player = {}          # env -> player dict of its last run-level decision
    cur_acts = tuple(int(x) for x in a.curriculum_acts.split(",") if x.strip())
    cur_mix = tuple(float(x) for x in a.curriculum_mix.split(","))

    arm = os.path.basename(a.out).split(".")[0]
    game_cpus = None
    if a.trainer_cpus:
        mine_cpus = {int(c) for c in a.trainer_cpus.split(",")}
        os.sched_setaffinity(0, mine_cpus)
        reserved = {int(c) for c in a.reserve_cpus.split(",") if c.strip()}
        game_cpus = sorted(set(range(os.cpu_count())) - mine_cpus - reserved)
    pool = None
    if a.pool:
        if a.search_rooms and a.collector_gpu >= 0:
            raise SystemExit("--search-rooms does not support --collector-gpu")
        pool = RunPool(n_envs=a.pool, act_cap=a.act_cap, recycle_runs=a.recycle_runs, nice=a.pool_nice,
                       cpus=game_cpus,
                       **({"game_dir": os.path.abspath(a.game_dir)} if a.game_dir else {}),
                       extra_args=SR.probe_args(a.search_rooms, "turn", a.search_k, a.search_mass,
                                                a.search_samples, frame_suspend=not a.timer_suspend),
                       batch_timeout=a.batch_timeout if a.batch_timeout is not None
                       else (0.001 if a.search_rooms else 0.004),
                       out_dir=f"/tmp/alphaslayer_pool_{arm}_")
        # Tear the pool down on any exit, including SIGTERM (earlyoom), so no game outlives us.
        import atexit
        atexit.register(pool.close)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    skill_ema = None          # EMA of the value head's out-of-sample Brier skill
    skill_n = 0               # iterations that produced a measurable skill
    hist_bl = None
    hist_path = a.out + ".seedhist.pt"
    if a.history_baseline:
        if not a.seed_block:
            raise SystemExit("--history-baseline needs --seed-block")
        hist_bl = P.HistoryBaseline()
        if a.resume and os.path.exists(hist_path):
            hist_bl.__dict__.update(torch.load(hist_path, weights_only=False))
    critic = None
    critic_path = a.out + ".critic.pt"
    if a.run_critic:
        critic = RunCritic(n_boss=256, window=a.critic_window)
        if a.resume and os.path.exists(critic_path):
            critic.__dict__.update(torch.load(critic_path, weights_only=False))
            critic.window = a.critic_window       # the saved one would override the flag
            print(f"run critic resumed: {len(critic.buf)} iterations of replay", flush=True)

    ptime = collections.defaultdict(float)     # where the policy callback spends its time

    def learner_sink(buf, idx, kind, obs, legal, act, logp, v, loop, p_fight, enc=None, target=None):
        if kind != "combat":
            last_player[idx] = obs.get("player", obs)
        buf.add(idx, kind, obs, legal, act, logp, v, loop, p_fight=p_fight, enc=enc, target=target)

    graphs = None
    if a.graphs:
        from alphaslayer.unified.graphrun import GraphRunner
        graphs = GraphRunner(net, dev)       # replays read the live weights: updates need no recapture

    def act_policy(kinds, obs, legal, idxs, buf, qs=None, rolls=None):
        return make_policy(net, vocab, dev, a.bf16, a.loop_policy,
                           lambda *x: learner_sink(buf, *x), ptime,
                           search_tau=a.search_tau if a.search_rooms else None,
                           graphs=graphs)(kinds, obs, legal, idxs, qs=qs, rolls=rolls)

    # M9: the probe's search requests (prior, rollout choices, leaf values), same network
    search = None
    if a.search_rooms:
        roll = make_policy(net, vocab, dev, a.bf16, a.loop_policy, lambda *x: None, greedy=True, graphs=graphs)
        search = SR.make_search(net, vocab, dev,
                                lambda obs, legal: roll(["combat"] * len(obs), obs, legal, [0] * len(obs)),
                                w_hp=a.w_hp, w_act=a.w_act, w_fight=a.w_fight)

    # ---- a second collector process on the other GPU (alphaslayer/unified/collector.py) ----
    worker = None
    if a.collector_gpu >= 0:
        if not a.pool:
            raise SystemExit("--collector-gpu needs --pool")
        key = os.urandom(16)
        lst = Listener(("127.0.0.1", 0), authkey=key)
        wenv = dict(os.environ, HIP_VISIBLE_DEVICES=str(a.collector_gpu))
        wproc = subprocess.Popen(
            [sys.executable, "-m", "alphaslayer.unified.collector", "--port",
             str(lst.address[1]), "--key", key.hex(), "--slots", str(a.collector_slots),
             "--act-cap", str(a.act_cap), "--recycle-runs", str(a.recycle_runs),
             "--nice", str(a.pool_nice), "--out-dir", f"/tmp/alphaslayer_poolw_{arm}_",
             "--vocab", os.path.join(HERE, "vocab_m1.json")], cwd=HERE, env=wenv)
        worker = (lst.accept(), wproc)
        wpath = f"/dev/shm/alphaslayer_{arm}_weights.pt"

        def stop_worker():
            try:
                worker[0].send({"cmd": "quit"})
            except Exception:
                pass
            try:
                worker[1].wait(timeout=30)
            except Exception:
                worker[1].kill()
            if os.path.exists(wpath):
                os.remove(wpath)
        import atexit
        atexit.register(stop_worker)
        print(f"collector process on GPU {a.collector_gpu}, {a.collector_slots} slots", flush=True)

    def evaluate(tag):
        """Plays the FIXED evaluation seeds (EV_{e}_..., e = 1..eval_iters; never training
        seeds) with the current policy, sampling as in training, and logs one line. Every
        evaluation of every arm and checkpoint plays the same maps, so they compare paired."""
        if pool is None:
            raise SystemExit("--eval-every / --eval-only need --pool")
        if a.trainer_cpus:
            os.sched_setaffinity(0, mine_cpus)
        te = time.time()
        res, per_batch, per_run = [], [], []
        sink = lambda *x: None
        for e in range(1, a.eval_iters + 1):
            # run ids are labels only (unique, so a result maps back to its job); the seed
            # string is unchanged, so every evaluation still plays the same maps
            # "search": False - the evaluation measures the bare policy, even in an M9 pool
            jobs = [{"run": e * 100000 + r * 1000 + i, "seed": f"EV_{e}_{i}{e * 1000 + r}",
                     "character": CHARS[i % len(CHARS)], "search": False}
                    for r in range(a.eval_runs_per_env) for i in range(a.envs)]
            pol = make_policy(net, vocab, dev, a.bf16, a.loop_policy, sink,
                              greedy=not a.eval_sample)
            pool.run(jobs, pol, deadline_s=a.collect_timeout, on_restart=lambda i: None)
            rr = list(pool.results)
            by_run = {j["run"]: j for j in jobs}
            per_run += [{"seed": by_run[r.run]["seed"], "character": by_run[r.run]["character"],
                         "won": bool(r.won), "act": int(r.act), "floors": int(r.floors)}
                        for r in rr if r.run in by_run]
            res += rr
            per_batch.append(rr)
        n = len(res)
        f = lambda rs, fn: sum(fn(r) for r in rs) / max(1, len(rs))
        win, a1, a2 = (f(res, lambda r: r.won), f(res, lambda r: r.act >= 1 or r.won),
                       f(res, lambda r: r.act >= 2 or r.won))
        print(f"eval {tag} {'sampled' if a.eval_sample else 'greedy'} runs={n} win={win:.3f} a1={a1:.3f} a2={a2:.3f} | per batch win "
              + " ".join(f"{f(b, lambda r: r.won):.3f}" for b in per_batch)
              + f" | {time.time() - te:.0f}s", flush=True)
        # per-run outcomes, for paired comparisons between checkpoints (same seeds)
        os.makedirs(os.path.join(HERE, "..", "logs", "eval_runs"), exist_ok=True)
        with open(os.path.join(HERE, "..", "logs", "eval_runs",
                               f"{arm}_{tag}_{'s' if a.eval_sample else 'g'}.json"), "w") as fh:
            json.dump(per_run, fh)

    if a.eval_only:
        evaluate(f"{os.path.basename(a.resume and os.path.exists(a.out) and a.out or a.init)}@{start - 1}")
        return

    # Moving anchor: CPU copies of the policy at the start of the last anchor_lag+1 iterations;
    # the oldest is the anchor. Not saved: after a resume the anchor restarts at the resumed
    # policy (KL 0) and lags behind again from there.
    snaps = collections.deque(maxlen=a.anchor_lag + 1) if a.anchor_lag > 0 else None

    for it in range(start, a.iters + 1):
        if pool is not None and a.eval_every and (it - 1) % a.eval_every == 0:
            evaluate(f"it{it - 1}")
        if snaps is not None:
            snaps.append({k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()})
            anchor.load_state_dict(snaps[0])
        if a.trainer_cpus:
            os.sched_setaffinity(0, mine_cpus)     # collection: our own cores, games elsewhere
        t0 = time.time()
        bootstrap = (a.bootstrap == "on"
                     or (a.bootstrap == "auto" and skill_ema is not None
                         and skill_n >= a.skill_min_iters and skill_ema >= a.skill_threshold))
        buf = P.RunBuffer(gamma=a.gamma, lam=a.lam, w_combat=a.w_combat, w_hp=a.w_hp,
                          w_act=a.w_act, w_fight=a.w_fight, bootstrap=bootstrap,
                          lam_step=a.lam_step, keep_obs=False)
        fights = []
        boss = collections.defaultdict(lambda: [0, 0])
        deaths = collections.Counter()
        last_fight = {}

        new_harvest = []

        def on_fight(env, fr):
            buf.end_fight(env, fr.won, fr.hp_start, fr.hp_end, fr.max_hp, fr.room)
            stats.update(fr.encounter, fr.won)
            pl = last_player.get(env)
            if pl and pl.get("deck_cards") and fr.act >= 0:
                new_harvest.append({
                    "character": pl.get("character"), "room": fr.room,
                    "encounter": fr.encounter, "act": fr.act,
                    "cards": [c["card"] for c in pl["deck_cards"]],
                    "upgrades": [int(c.get("up", 0) or 0) for c in pl["deck_cards"]],
                    "relics": [r["relic"] for r in (pl.get("relics") or [])],
                    "hp": fr.hp_start, "max_hp": fr.max_hp, "won": bool(fr.won),
                    "player": pl})
            fights.append(fr)
            last_fight[env] = fr
            if fr.room == "Boss":
                boss[fr.encounter][0] += int(fr.won)
                boss[fr.encounter][1] += 1

        group_of = {}

        def on_terminal(env, rr):
            buf.end_run(env, rr.won, rr.act, rr.floors)
            buf.runs[-1]["group"] = group_of.get(rr.run, -1)
            if not rr.won:
                fr = last_fight.get(env)
                deaths[f"a{rr.act + 1}-{fr.room if fr is not None and not fr.won else 'other'}"] += 1
            last_fight.pop(env, None)

        def on_restart(env):
            buf.drop_env(env)
            last_fight.pop(env, None)
            last_player.pop(env, None)

        pool_line = ""
        ptime.clear()
        if pool is not None:
            # Exactly the (seed, character) runs the per-iteration launch plays: env i's probe
            # seed "{prefix}{it}_{i}" + run id it*1000 + r, character CHARS[i % 5], so pool
            # and non-pool arms stay paired. Interleaved so characters mix in the queue.
            # --seed-groups K: the first 1/K of those seeds, each played K times (the copies
            # run concurrently, under the same policy). Run ids are unique per job.
            base = [(i, r) for r in range(a.runs_per_env) for i in range(a.envs)]
            n_seed = len(base) // a.seed_groups
            jobs = []
            if a.seed_block:
                # A block of --block-seeds seeds, each played ONCE per character per
                # iteration, kept for --seed-block iterations, then replaced. Group = seed x
                # character, the same id in every iteration of the block.
                blk = (it - 1) // a.seed_block
                for s_ in range(a.block_seeds):
                    for ci, ch in enumerate(CHARS):
                        group_of[it * 1000 + len(jobs)] = s_ * len(CHARS) + ci
                        jobs.append({"run": it * 1000 + len(jobs),
                                     "seed": f"{a.seed_prefix}B{blk}_{s_}", "character": ch})
            for c_ in range(0 if a.seed_block else a.seed_groups):
                for g_, (i, r) in enumerate(base[:n_seed]):
                    group_of[it * 1000 + len(jobs)] = g_
                    jobs.append({"run": it * 1000 + len(jobs),
                                 "seed": f"{a.seed_prefix}{it}_{i}{it * 1000 + r}",
                                 "character": CHARS[i % len(CHARS)]})
            mine = jobs
            if worker is not None:
                # split by slot share; the union is still exactly this iteration's jobs
                n_w = round(len(jobs) * a.collector_slots / (a.collector_slots + a.pool))
                mine, theirs = jobs[n_w:], jobs[:n_w]
                torch.save({"model": net.state_dict(), "iter": it, "cfg": net.cfg,
                            "sizes": ck["sizes"]}, wpath + ".tmp")
                os.replace(wpath + ".tmp", wpath)
                worker[0].send({"cmd": "collect", "weights": wpath, "jobs": theirs,
                                "deadline": a.collect_timeout, "bf16": a.bf16,
                                "loop_policy": a.loop_policy})
            def collect_policy(k, o, l, i, qs=None, rolls=None):
                return act_policy(k, o, l, i, buf, qs, rolls)
            collect_policy.takes_q = search is not None
            # rollout choices ride in the live pass (same net, loop count and precision)
            collect_policy.takes_rolls = search is not None and a.loop_policy != "adaptive"
            pool.run(mine, collect_policy, on_fight=on_fight,
                     on_terminal=on_terminal, on_restart=on_restart,
                     deadline_s=a.collect_timeout, search=search)
            results = list(pool.results)
            st_ = dict(pool.stats)
            pool_line = (f"pool: policy {pool.policy_s:.0f}s (encode {ptime['encode']:.0f} "
                         f"forward {ptime['forward']:.0f} sample {ptime['sample']:.0f}) of {pool.wall:.0f}s, "
                         f"{pool.steps / max(1, pool.batches):.1f}/batch, "
                         f"recycled {pool.stats['recycled']}, lost {len(pool.lost)} ")
            if worker is not None:
                rep = worker[0].recv()
                # Worker runs are complete event sequences: replay them, one at a time, under
                # an env key no learner slot uses.
                rr_w = []
                replay(rep["runs"], -1, lambda k, *x: learner_sink(buf, k, *x), on_fight,
                       lambda k, rr: (rr_w.append(rr), on_terminal(k, rr)))
                results += rr_w
                for k_ in st_:
                    st_[k_] += rep["stats"][k_]
                pool_line += (f"| worker: {len(rr_w)} runs, policy {rep['policy_s']:.0f}s of "
                              f"{rep['wall']:.0f}s, {rep['steps'] / max(1, rep['batches']):.1f}/batch, "
                              f"lost {len(rep['lost'])} ")
            env_fail = (st_["crashed"], st_["stalled"], st_["relaunched"])
        else:
            with VecRunEnv(n_envs=a.envs, characters=CHARS, runs_per_env=a.runs_per_env,
                           seed=f"{a.seed_prefix}{it}_", ascension=0, act_cap=a.act_cap,
                           run_offset=it * 1000, restarts=a.env_restarts,
                           # per arm, so two trainers (one per GPU) never share probe logs
                           out_dir=f"/tmp/alphaslayer_runserve_{arm}_") as venv:
                venv.run(lambda k, o, l, i: act_policy(k, o, l, i, buf), on_fight=on_fight,
                         on_terminal=on_terminal, deadline_s=a.collect_timeout,
                         on_restart=on_restart)
                results = list(venv.results)
                env_fail = (venv.crashed, venv.stalled, venv.restarted)
        buf.drop_open()
        steps = buf.done
        collect_s = time.time() - t0
        # Out-of-sample skill of the value head: these values were predicted before these runs'
        # outcomes existed. It gates bootstrapping for the NEXT iteration.
        _, _, skill = P.value_skill(steps)
        if skill == skill:                       # not NaN (needs both outcomes present)
            skill_n += 1
            skill_ema = skill if skill_ema is None else (
                (1 - a.skill_ema) * skill_ema + a.skill_ema * skill)
        if new_harvest:
            with open(harvest_path, "a") as fh:
                for r in new_harvest:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            harvest = (harvest + new_harvest)[-a.curriculum_window:]

        # ---- combat curriculum: isolated act-2/3 fights, loss-weighted, same objective --------
        cur_line = ""
        if a.curriculum_every and it % a.curriculum_every == 0 and harvest:
            tc = time.time()
            from alphaslayer.deckeval import VecDeckEval
            picks = CU.choose_fights(harvest, stats, a.curriculum_fights, acts=cur_acts,
                                     mix=cur_mix, rng=random.Random(10_000 + it),
                                     pools=vocab.act_pools)
            pools = {}
            try:
                for act in sorted({r["act"] for r, _, _ in picks}):
                    # At least one env per character: a pool's envs are assigned round-robin
                    # by character, and a character with no env cannot play its fights.
                    pools[act] = VecDeckEval(n_envs=max(a.curriculum_envs, len(CHARS)),
                                             characters=CHARS,
                                             seed=f"UCUR{act}_{it}_",
                                             out_dir=f"/tmp/alphaslayer_ucur{act}_",
                                             act=act, stall_timeout=90.0)
                c_steps, c_res = CU.play_fights(
                    net, pools, picks, vocab, vocab.enc_ix, dev, bf16=a.bf16,
                    credit=dict(w_combat=a.w_combat, w_hp=a.w_hp, w_act=a.w_act,
                                w_fight=a.w_fight, lam=a.lam, bootstrap=bootstrap))
            finally:
                for pl_ in pools.values():
                    try:
                        pl_.close()
                    except Exception:
                        pass
            for enc, won, *_ in c_res:
                stats.update(enc, won)
            steps = steps + c_steps
            per = collections.defaultdict(lambda: [0, 0])
            for enc, won, *_ in c_res:
                per[enc][0] += int(won); per[enc][1] += 1
            cur_line = (f"    curriculum | {len(c_res)} fights, {len(c_steps)} steps, "
                        f"{time.time() - tc:.0f}s | " + " ".join(
                            f"{e.split('_BOSS')[0].split('_ELITE')[0]}={w}/{t}"
                            for e, (w, t) in sorted(per.items(), key=lambda kv: -kv[1][1])[:8]))
        if not steps:
            print(f"iter {it}: no completed runs; skipping", flush=True)
            continue

        # ---- PPO update --------------------------------------------------------------------
        if a.trainer_cpus:
            # The games sit idle during the update; pinned to 4 CPUs it took 45 s instead of 35.
            os.sched_setaffinity(0, range(os.cpu_count()))
        t1 = time.time()
        records = [{"kind": s.kind, "obs": s.obs, "legal": s.legal, "enc": s.enc} for s in steps]
        packed = DS.Packed(DS.pack(records, vocab))
        rc_line = ""
        if critic is not None:
            rc_line = run_critic_potentials(critic, packed, steps, buf.runs, a, bootstrap)
        if a.dump_steps:
            dump_steps(a.dump_steps, it, packed, steps, buf.runs)
        if a.collect_only:
            n = len(results)
            print(f"iter {it:4d}/{a.iters} COLLECT runs={n} win={sum(r.won for r in results) / max(1, n):.3f} "
                  f"a1={sum(1 for r in results if r.act >= 1 or r.won) / max(1, n):.3f} "
                  f"a2={sum(1 for r in results if r.act >= 2 or r.won) / max(1, n):.3f} "
                  f"skill={skill:+.3f} steps={len(steps)} "
                  f"crash/stall/relaunch={env_fail[0]}/{env_fail[1]}/{env_fail[2]} "
                  f"collect {collect_s:.0f}s {pool_line}", flush=True)
            continue
        grp_line = ""
        if a.group_baseline:
            n_adj = P.seed_group_baseline(steps, buf.runs)
            grp_line = f"grp_adj={n_adj / max(1, len(steps)):.2f} "
        if hist_bl is not None:
            n_adj = hist_bl.apply(steps, buf.runs, (it - 1) // a.seed_block)
            grp_line = f"hist_adj={n_adj / max(1, len(steps)):.2f} "
            torch.save(hist_bl.__dict__, hist_path + ".tmp")
            os.replace(hist_path + ".tmp", hist_path)
        adv = P.normalise_advantages(steps, center=a.adv_center)
        # Per-kind advantage means. Normalisation centres each HALF (fight / run), but the loss
        # averages per KIND, so a kind whose advantages are systematically negative pushes
        # probability away from whatever it sampled, and its entropy rises. Logged every
        # iteration so that bias is visible.
        kadv = collections.defaultdict(list)
        for s_, an in zip(steps, adv):
            kadv[s_.kind].append((s_.adv, float(an)))
        adv_line = " ".join(
            f"{k}:{np.mean([x for x, _ in v]):+.3f}/{np.mean([y for _, y in v]):+.2f}"
            for k, v in sorted(kadv.items(), key=lambda kv: -len(kv[1])))
        arr = lambda f, dt=np.float32: np.array([f(s) for s in steps], dt)
        T_ = {"a": arr(lambda s: s.a, np.int64), "old_logp": arr(lambda s: s.logp),
              "adv": adv, "ret": arr(lambda s: s.ret),
              "kind": arr(lambda s: FT.DKINDS.index(s.kind), np.int64),
              "act_clear": arr(lambda s: s.act_clear), "reach_act3": arr(lambda s: s.reach_act3),
              "floors_left": arr(lambda s: s.floors_left), "fight_won": arr(lambda s: s.fight_won),
              "fight_hp": arr(lambda s: s.fight_hp), "loop": arr(lambda s: s.loop, np.int64)}
        # M9: searched decisions learn from the search policy (distillation), not from PPO:
        # they acted from pi', so the PPO ratio against pi would be wrong
        searched = np.array([s.target is not None for s in steps], bool)
        if searched.any():
            adv = np.where(searched, 0.0, adv).astype(np.float32)
            T_["adv"] = adv
            n_c = max(len(s.legal) for s in steps)
            sp_all = np.zeros((len(steps), n_c), np.float32)
            for j in np.flatnonzero(searched):
                t_ = steps[j].target
                sp_all[j, :len(t_)] = t_
            T_["searched"] = searched
            T_["search_p"] = sp_all
        T_dev = {k: torch.as_tensor(v).to(dev) for k, v in T_.items()}   # once per iteration
        n_tok = packed.a["n_tokens"]
        kl_coef = a.kl_coef * (max(0.0, 1.0 - (it - 1) / a.kl_anneal) if a.kl_anneal > 0 else 1.0)
        net.train()
        agg = collections.defaultdict(float)
        nb = n_opt = 0
        rng = random.Random(it)
        for ep in range(a.epochs):
            order = list(range(len(steps)))
            rng.shuffle(order)
            batches = []
            for s0 in range(0, len(order), a.minibatch * 16):
                part = sorted(order[s0:s0 + a.minibatch * 16], key=lambda i: n_tok[i])
                batches += [part[j:j + a.minibatch] for j in range(0, len(part), a.minibatch)]
            rng.shuffle(batches)
            opt.zero_grad(set_to_none=True)
            for bi, idx in enumerate(batches):
                inp, _ = DS.make_batch([(packed, i) for i in idx])
                inp = to_torch(inp, dev)
                ix = torch.as_tensor(idx, device=dev)
                tgt = {k: v[ix] for k, v in T_dev.items()}
                tgt["cand_mask"] = inp["cand_mask"]
                if "search_p" in tgt:          # to this batch's candidate width
                    w_ = tgt["cand_mask"].shape[1]
                    sp_ = tgt["search_p"][:, :w_]
                    if sp_.shape[1] < w_:
                        sp_ = F.pad(sp_, (0, w_ - sp_.shape[1]))
                    tgt["search_p"] = sp_
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                    out = net(inp, loop_rows=tgt["loop"])
                    anchor_logp = None
                    if kl_coef > 0:
                        with torch.no_grad():
                            anchor_logp = F.log_softmax(
                                anchor(inp, loop_rows=tgt["loop"])["logits"].float(), -1)
                total, parts = P.ppo_loss(out, tgt, clip=a.clip, ent_coef=a.ent_coef,
                                          vf_coef=a.vf_coef, aux_coef=a.aux_coef,
                                          run_weight=a.run_weight, anchor_logp=anchor_logp,
                                          kl_coef=kl_coef, kind_weight=a.kind_weight,
                                          kl_run_mult=a.kl_run_mult,
                                          distill_coef=a.distill_coef if a.search_rooms else 0.0)
                (total / a.accum).backward()
                for k, v in parts.items():
                    agg[k] += float(v.detach())
                nb += 1
                if (bi + 1) % a.accum and bi + 1 < len(batches):
                    continue                     # accumulate until --accum minibatches
                # One sync: a non-finite gradient anywhere makes the total norm non-finite. The
                # per-parameter check this replaces synced ~200 times per minibatch (15 s of a
                # profiled 2-iteration run).
                gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if torch.isfinite(gnorm):
                    opt.step()
                opt.zero_grad(set_to_none=True)
                n_opt += 1
        train_s = time.time() - t1

        # ---- report --------------------------------------------------------------------------
        n = len(results)
        won = sum(r.won for r in results)
        a1 = sum(1 for r in results if r.act >= 1 or r.won) / max(1, n)
        a2 = sum(1 for r in results if r.act >= 2 or r.won) / max(1, n)
        fw = sum(f.won for f in fights) / max(1, len(fights))
        # Only decisions whose run outcome is known (curriculum fights have none).
        brier, base, _ = P.value_skill(steps)
        # Delivery check for the shaping: every fight decision should carry P(win this fight),
        # and the potentials should be non-zero when their weights are.
        cs = [s for s in steps if s.kind == "combat"]
        pf_cover = sum(1 for s in cs if s.p_fight is not None) / max(1, len(cs))
        phi_mean = float(np.mean([s.phi for s in steps])) if steps else 0.0
        kstat = collections.defaultdict(lambda: [0, collections.Counter()])
        for s in steps:
            if s.kind == "combat":
                continue
            ks = kstat[s.kind]
            ks[0] += 1
            lab = (s.legal[s.a].get("option") or s.legal[s.a].get("kind") or "opt") \
                if s.kind in ("rest", "potion_gate", "card_reward", "shop") else "a"
            ks[1][lab] += 1
        collapse = " ".join(
            f"{k}:{v[1].most_common(1)[0][0]}={100 * v[1].most_common(1)[0][1] / v[0]:.0f}%"
            for k, v in sorted(kstat.items()) if k in ("rest", "potion_gate", "card_reward", "shop"))
        loops = np.mean([s.loop for s in steps])
        print(f"iter {it:4d}/{a.iters} runs={n} win={won / max(1, n):.3f} a1={a1:.3f} a2={a2:.3f} "
              f"a2|a1={a2 / max(a1, 1e-9):.3f} floors={np.mean([r.floors for r in results]):.1f} "
              f"fight_win={fw:.3f} | pg={agg['pg'] / nb:+.4f} ent={agg['ent'] / nb:.3f} "
              f"vf={agg['value'] / nb:.4f} kl_anchor={agg['kl'] / nb:.4f}(x{kl_coef:.3f}) "
              f"clip={agg['clipfrac'] / nb:.3f} akl={agg['approx_kl'] / nb:+.4f} | "
              f"brier={brier:.4f}/base {base:.4f} skill={skill:+.3f} ema={skill_ema if skill_ema is not None else float('nan'):+.3f} "
              f"boot={'on' if bootstrap else 'off'} pf_cover={pf_cover:.2f} phi={phi_mean:.3f} "
              f"loops={loops:.2f} steps={len(steps)} opt_steps={n_opt} {rc_line}{grp_line}"
              f"crash/stall/relaunch={env_fail[0]}/{env_fail[1]}/{env_fail[2]} "
              f"collect {collect_s:.0f}s train {train_s:.0f}s {pool_line}", flush=True)
        print(f"    adv by kind (raw/normalised mean) | {adv_line}", flush=True)
        if a.search_rooms:
            n_s = sum(1 for s_ in steps if s_.target is not None)
            # how far the search moves the policy: mean max pi' and how often pi' sampled a move
            # other than its own top choice
            print(f"    search | searched {n_s} decisions ({n_s / max(1, len(steps)):.1%}), "
                  f"leaf evals {pool.leaf_evals}, search msgs {pool.search_msgs}, distill CE "
                  f"{agg['distill'] / max(1, nb):.4f}, pi' max {np.mean([s_.target.max() for s_ in steps if s_.target is not None] or [0]):.3f}",
                  flush=True)
        if cur_line:
            print(cur_line, flush=True)
        print(f"    top actions | {collapse} | deaths {dict(deaths.most_common(4))} | bosses "
              + " ".join(f"{e.split('_BOSS')[0]}={w}/{t}" for e, (w, t) in sorted(boss.items())),
              flush=True)
        if critic is not None:
            torch.save(critic.__dict__, critic_path + ".tmp")
            os.replace(critic_path + ".tmp", critic_path)
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "iter": it,
                    "cfg": net.cfg, "sizes": ck["sizes"], "args": vars(a)}, a.out + ".tmp")
        os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
