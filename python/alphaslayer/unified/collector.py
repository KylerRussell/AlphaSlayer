"""Rollout collection shared by the trainer and a second collector PROCESS (one per GPU).

Measured on a 24-slot RunPool: the trainer's policy callback was 75% of collection wall time
(one Python thread encodes, runs the net and samples for every decision while the games
wait). More games only queued behind it (32 slots slower than 24). A second process with its
own copy of the weights, its own GPU and its own pool doubles the policy threads.

  make_policy(...)   the batched act function both sides use; it reports every decision to a
                     ``sink`` (the trainer's RunBuffer.add, or a Recorder here)
  Recorder           a worker's per-slot event log; a completed run is returned as its exact
                     event sequence (steps, fight ends, terminal), and the trainer REPLAYS it
                     into its RunBuffer, so credit is computed in one place
  worker_main        the collector process: load weights, play the jobs it is sent, reply
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from multiprocessing.connection import Client

import numpy as np
import torch
import torch.nn.functional as F

from . import features as FT
from . import search as S
from .net import UnifiedNet, choose_loops, to_torch


def make_policy(net, vocab, dev, bf16, loop_policy, sink, ptime=None, greedy=False, search_tau=None,
                graphs=None):
    """``policy(kinds, obs, legal, idxs) -> actions``; every decision is passed to
    ``sink(idx, kind, obs, legal, act, logp, v, loop, p_fight, enc)`` (enc = its encoding).

    search_tau: the policy also takes ``qs=`` (fight search values, None where a decision was
    not searched). A searched decision acts from the search policy pi' ~ pi exp(q / tau)
    (sampled, or its argmax when greedy), and the sink gets pi' as an 11th argument.

    rolls=(obs, legal): fight-search rollout choices answered in the SAME forward (one network
    pass per server step instead of two): always the argmax, never passed to the sink; the
    call then returns (actions, rollout actions). Only with a fixed loop count
    (``policy.takes_rolls``).

    graphs: a graphrun.GraphRunner (kept across calls): bf16 passes with a fixed loop count are
    replayed from recorded graphs (one launch instead of ~420)."""
    ptime = ptime if ptime is not None else collections.defaultdict(float)

    def policy(kinds, obs, legal, idxs, qs=None, rolls=None):
        t_a = time.time()
        encs = []
        n_live = len(kinds)
        rows = list(zip(kinds, obs, legal))
        if rolls:
            rows += [("combat", o, l) for o, l in zip(*rolls)]
        bnp = FT.encode_batch(rows, vocab, keep=encs)
        use_graph = graphs is not None and bf16 and loop_policy != "adaptive"
        b = None if use_graph else to_torch(bnp, dev)
        t_b = time.time()
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            if use_graph:
                k = int(loop_policy)
                out = graphs.run(bnp, loop=k, bf16=True)
                logits, values = out["logits"], out["value"]
                pfight = out["aux"]["fight_win"]
                ks = torch.full((len(rows),), k)
            elif loop_policy == "adaptive":
                traj = net.forward_trajectory(b, 4)
                risk = 1 - torch.sigmoid(traj[0]["aux"]["fight_win"].float())
                is_c = torch.tensor([k == "combat" for k in kinds], device=dev)
                cap = torch.where(is_c & (risk > 0.2), 4, 2)
                ks = choose_loops(traj, b["cand_mask"], cap)
                logits = torch.stack([traj[int(k) - 1]["logits"][j] for j, k in enumerate(ks)])
                values = torch.stack([traj[int(k) - 1]["value"][j] for j, k in enumerate(ks)])
                pfight = torch.stack([traj[int(k) - 1]["aux"]["fight_win"][j] for j, k in enumerate(ks)])
            else:
                k = int(loop_policy)
                out = net(b, loop=k)
                logits, values = out["logits"], out["value"]
                pfight = out["aux"]["fight_win"]
                ks = torch.full((len(kinds),), k)
        # One batched sample and ONE device->host copy. A per-row loop of log_softmax /
        # multinomial / float() was 9 s of a 60 s collection (a kernel launch and a sync per
        # row). Padded candidates hold finfo.min/4 logits, so they get probability 0.
        lp_all = F.log_softmax(logits.float(), -1)
        # greedy: the top action (evaluation: removes the policy's sampling noise, so two
        # checkpoints on the same seeds differ only by what they learned)
        a_t = lp_all.argmax(-1) if greedy else torch.multinomial(lp_all.exp(), 1).squeeze(1)
        if rolls and not greedy:
            a_t[n_live:] = lp_all[n_live:].argmax(-1)       # rollouts play the greedy policy
        lp_a = lp_all.gather(1, a_t.unsqueeze(1)).squeeze(1)
        pf = torch.sigmoid(pfight.float())
        a_np, lp_np, v_np, pf_np = (x.cpu().numpy() for x in (a_t, lp_a, values.float(), pf))
        t_c = time.time()
        ks_l = ks.tolist() if hasattr(ks, "tolist") else list(ks)
        acts = []
        lp_rows = None
        for j, kind in enumerate(kinds):
            act = min(int(a_np[j]), len(legal[j]) - 1)
            lp_j, target = float(lp_np[j]), None
            if search_tau is not None and qs is not None and qs[j] is not None:
                if lp_rows is None:
                    lp_rows = lp_all.cpu().numpy()
                target = S.improved(lp_rows[j][:len(legal[j])], qs[j], search_tau)
                if target is not None:
                    act = int(target.argmax()) if greedy else int(np.random.choice(len(target), p=target))
                    lp_j = float(lp_rows[j][act])
            args = (idxs[j], kind, obs[j], legal[j], act, lp_j, float(v_np[j]),
                    int(ks_l[j]), float(pf_np[j]) if kind == "combat" else None, encs[j])
            sink(*args, target) if target is not None else sink(*args)
            acts.append(act)
        ptime["encode"] += t_b - t_a
        ptime["forward"] += t_c - t_b
        ptime["sample"] += time.time() - t_c
        if rolls:
            racts = [min(int(a_np[n_live + r]), len(lg) - 1) for r, lg in enumerate(rolls[1])]
            return acts, racts
        return acts

    policy.ptime = ptime
    policy.takes_rolls = loop_policy != "adaptive"
    policy.takes_q = search_tau is not None
    return policy


class Recorder:
    """Per-slot event log. A slot plays one run at a time, so its open events are one run."""

    def __init__(self):
        self.open = collections.defaultdict(list)
        self.runs: list[list] = []

    def step(self, idx, kind, obs, legal, act, logp, v, loop, p_fight, enc=None):
        self.open[idx].append(("step", kind, obs, legal, act, logp, v, loop, p_fight, enc))

    def fight(self, idx, fr):
        self.open[idx].append(("fight", fr))

    def terminal(self, idx, rr):
        self.runs.append(self.open.pop(idx, []) + [("terminal", rr)])

    def restart(self, idx):
        self.open.pop(idx, None)


def replay(runs, key, add, on_fight, on_terminal):
    """Feeds a worker's completed runs into the trainer's callbacks, in their original order.
    ``key`` is the env index the trainer files them under (one at a time, so one key does)."""
    for events in runs:
        for ev in events:
            if ev[0] == "step":
                add(key, *ev[1:])
            elif ev[0] == "fight":
                on_fight(key, ev[1])
            else:
                on_terminal(key, ev[1])


def load_net(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = UnifiedNet(ck["sizes"], **ck["cfg"]).to(dev)
    net.load_state_dict(ck["model"])
    return net.eval()


def worker_main():
    """A collector process. Run as ``python -m alphaslayer.unified.collector --port P --key K``
    with HIP_VISIBLE_DEVICES selecting its GPU."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--slots", type=int, default=20)
    ap.add_argument("--act-cap", type=int, default=3)
    ap.add_argument("--recycle-runs", type=int, default=50)
    ap.add_argument("--nice", type=int, default=5)
    ap.add_argument("--out-dir", default="/tmp/alphaslayer_poolw_")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--loop-policy", default="1")
    ap.add_argument("--vocab", required=True)
    a = ap.parse_args()

    from alphaslayer.model import use_stable_attention
    from alphaslayer.runenv import RunPool
    from alphaslayer.gpu_guard import refuse_display_gpu
    use_stable_attention()
    dev = torch.device("cuda:0")
    refuse_display_gpu(dev)
    vocab = FT.Vocab(json.load(open(a.vocab)))
    conn = Client(("127.0.0.1", a.port), authkey=bytes.fromhex(a.key))
    pool = RunPool(n_envs=a.slots, act_cap=a.act_cap, recycle_runs=a.recycle_runs, nice=a.nice,
                   out_dir=a.out_dir)
    net, loaded = None, None
    try:
        while True:
            msg = conn.recv()
            if msg["cmd"] == "quit":
                break
            if msg["weights"] != loaded or net is None:
                net = load_net(msg["weights"], dev) if net is None else net
                st = torch.load(msg["weights"], map_location=dev, weights_only=False)
                net.load_state_dict(st["model"])
                loaded = msg["weights"] + str(st.get("iter"))
            rec = Recorder()
            ptime = collections.defaultdict(float)
            policy = make_policy(net, vocab, dev, msg["bf16"], msg["loop_policy"], rec.step, ptime)
            pool.run(msg["jobs"], policy, on_fight=rec.fight, on_terminal=rec.terminal,
                     on_restart=rec.restart, deadline_s=msg["deadline"])
            conn.send({"runs": rec.runs, "stats": pool.stats, "lost": pool.lost,
                       "wall": pool.wall, "policy_s": pool.policy_s, "ptime": dict(ptime),
                       "steps": pool.steps, "batches": pool.batches})
    finally:
        pool.close()
        conn.close()


if __name__ == "__main__":
    sys.exit(worker_main())
