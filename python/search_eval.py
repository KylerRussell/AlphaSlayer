"""Greedy fixed-seed evaluation WITH one-step lookahead search in fights (probe FightSearch).

At every combat decision in the searched room types, the game restores the fight, simulates
each legal choice (draw pile shuffled with a search-only RNG), and sends the resulting
positions here; each is valued with the training potential

    phi = P(win this fight) * (w_act * (act + 1) + w_hp * hp%)        (ppo.potential)

(a won fight: P = 1 at its final hp; a lost one: 0), and the decision takes the choice with
the highest value. Everything else is the plain greedy policy. Plays the same EV_ seeds as
train_unified's evaluation, so the per-run file pairs with any greedy eval (pair_evals.py).

    PYTHONPATH=. .venv7/bin/python search_eval.py --ckpt unified_m4_td2_it50.pt --rooms Boss,Elite
    PYTHONPATH=. .venv7/bin/python search_eval.py --ckpt unified_m4_td2_it50.pt --rooms ""  # no search
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import subprocess
import time

import numpy as np
import torch

from alphaslayer.gpu_guard import refuse_display_gpu
from alphaslayer.model import use_stable_attention
from alphaslayer.runenv import RunPool
from alphaslayer.unified import features as FT
from alphaslayer.unified import search as SR
from alphaslayer.unified.runsearch import RunSearcher
from alphaslayer.unified.collector import load_net, make_policy

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS = ["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"]   # = train_unified.CHARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rooms", default="Boss,Elite", help="room types to search ('' = none)")
    ap.add_argument("--mode", choices=["ply", "turn"], default="ply",
                    help="ply: every choice one step deep; turn: the policy's top-k choices, each "
                         "played to the end of the turn by the policy, in --samples sampled worlds")
    ap.add_argument("--cands", type=int, default=12, help="ply: choices valued per decision")
    ap.add_argument("--k", type=int, default=4, help="turn: max candidates (by the policy's prior)")
    ap.add_argument("--mass", type=float, default=0.95, help="turn: stop adding candidates at this prior mass")
    ap.add_argument("--samples", type=int, default=2, help="turn: sampled worlds per candidate")
    ap.add_argument("--run-search", default="", help="run decisions to search, e.g. card_reward,shop,travel")
    ap.add_argument("--run-k", type=int, default=3, help="run search: options valued (the policy's top k)")
    ap.add_argument("--worlds", type=int, default=4, help="run search, routes: sampled worlds per option")
    ap.add_argument("--boss-plays", type=int, default=2, help="run search, routes: boss playouts per world")
    ap.add_argument("--deck-plays", type=int, default=8,
                    help="run search, card reward/shop: boss playouts straight after the option")
    ap.add_argument("--max-live", type=int, default=6, help="run search: slots playing live runs")
    ap.add_argument("--w-dmg", type=float, default=0.5,
                    help="run search: a lost boss playout's value per unit of boss hp removed, x a win's")
    ap.add_argument("--z", type=float, default=1.0, help="run search: paired t to override the policy")
    ap.add_argument("--verify", action="store_true",
                    help="run search: only check that a rollout's replay lands on the live decision")
    ap.add_argument("--verify-drop-last", action="store_true",
                    help="negative control for --verify: replay one logged answer short; must mismatch")
    ap.add_argument("--run-dump", default=None, help="run search: jsonl, one line per searched decision")
    ap.add_argument("--deadline", type=float, default=7200)
    ap.add_argument("--batch-timeout", type=float, default=0.001,
                    help="server: seconds to wait for more requests before a network pass")
    ap.add_argument("--timer-suspend", action="store_true",
                    help="search polls on a 1 ms timer (the old, slower replay) instead of per frame")
    ap.add_argument("--no-graphs", dest="graphs", action="store_false",
                    help="eager bf16 policy passes instead of replaying recorded GPU graphs (graphrun.py)")
    ap.add_argument("--no-fuse", action="store_true",
                    help="one network pass per request type (the server before fusing; A/B)")
    ap.add_argument("--probe-arg", action="append", default=[],
                    help="extra game flag, e.g. --probe-arg=--probe-search-test-retry (tests)")
    ap.add_argument("--search-yield-every", type=int, default=0,
                    help="game waits per real frame while searching (0 = as in live play)")
    ap.add_argument("--game", default="data/sts2_testgame")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--runs-per-env", type=int, default=4)
    ap.add_argument("--eval-iters", type=int, default=5)
    ap.add_argument("--loud", action="store_true", help="game stdout/stderr to this process (native crashes)")
    ap.add_argument("--only-runs", default="", help="comma-separated run ids to play (repro one seed)")
    ap.add_argument("--job", default="", help="repro training runs: SEED:CHARACTER,... (ids 1..n)")
    ap.add_argument("--sample", action="store_true", help="live runs sample the policy (repro training)")
    ap.add_argument("--pick", choices=["argmax", "margin", "prior"], default="argmax",
                    help="argmax: highest leaf value; margin: override the policy only by > --margin; "
                         "prior: argmax log pi + q / --tau")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--w-hp", type=float, default=0.1)
    ap.add_argument("--w-act", type=float, default=1 / 3)
    ap.add_argument("--w-fight", type=float, default=1.0)
    ap.add_argument("--name", default=None, help="per-run file name (default from ckpt + rooms)")
    ap.add_argument("--out", default="/tmp/alphaslayer_search_")
    ap.add_argument("--dump", default=None, help="jsonl: one line per searched decision (diagnostics)")
    a = ap.parse_args()

    if os.path.realpath(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")) == \
            os.path.realpath(os.path.expanduser("~/.local/share")):
        # the private game data dir with Steam Workshop mods disabled (as exp_m4.sh); without it
        # the user's Workshop mods load into the headless games and every fight fails at setup
        os.environ["XDG_DATA_HOME"] = subprocess.run([os.path.join(HERE, "..", "headless_home.sh")],
                                                     capture_output=True, text=True, check=True).stdout.strip()
    use_stable_attention()
    dev = torch.device("cuda")
    refuse_display_gpu(dev)
    vocab = FT.Vocab(json.load(open(os.path.join(HERE, "vocab_m1.json"))))
    net = load_net(a.ckpt, dev)
    graphs = None
    if a.graphs:
        from alphaslayer.unified.graphrun import GraphRunner
        graphs = GraphRunner(net, dev)
    greedy = make_policy(net, vocab, dev, True, "1", lambda *x: None, greedy=not a.sample, graphs=graphs)
    picks = collections.Counter()
    dump = open(a.dump, "w") if a.dump else None

    def pick(q, lp, g):
        """The searched choice: q = leaf values (-1e9 = not simulated), lp = policy log-probs,
        g = the greedy choice."""
        q = np.asarray(q, dtype=np.float64)
        if q[g] < -1e8:              # the policy's choice was past the search cap: keep it
            return g
        if a.pick == "argmax":
            return int(np.argmax(q))
        if a.pick == "margin":       # keep the policy's choice unless the search clearly beats it
            best = int(np.argmax(q))
            return best if q[best] - q[g] > a.margin else g
        # "prior": pi'(a) ~ pi(a) exp(q(a) / tau), the KL-regularised improvement of the policy
        sc = lp[:len(q)] + np.where(q > -1e8, q, -np.inf) / a.tau
        return int(np.argmax(sc)) if np.isfinite(sc).any() else g

    def policy(kinds, obs, legal, idxs, qs=None, rolls=None):
        racts = None
        if rolls:                            # rollout choices in the same network pass
            acts, racts = greedy(kinds, obs, legal, idxs, rolls=rolls)
        else:
            acts = greedy(kinds, obs, legal, idxs)
        if rsr is not None:                  # run search: shop, card reward, route
            for j, q in enumerate(qs or []):
                if q is not None and kinds[j] != "combat":
                    acts[j] = rsr.pick(q, acts[j])
        js = [j for j, q in enumerate(qs or []) if q is not None and kinds[j] == "combat"]
        if js:
            lps = SR.log_prior(net, vocab, dev, [obs[j] for j in js], [legal[j] for j in js])
        for r, j in enumerate(js):
            q, lg, g = qs[j], legal[j], acts[j]
            lp = lps[r][:len(lg)]
            best = pick(q, lp, g)
            picks["searched"] += 1
            picks["changed"] += int(best != g)
            if dump:
                dump.write(json.dumps({"n": len(lg), "greedy": g, "best": best,
                                       "gk": lg[g].get("kind"), "bk": lg[best].get("kind"),
                                       "q": [round(x, 6) if x > -1e8 else None for x in q],
                                       "p": [round(float(np.exp(x)), 4) for x in lp],
                                       "kinds": [x.get("kind") for x in lg]}) + "\n")
                dump.flush()
            acts[j] = best
        return (acts, racts) if rolls else acts
    policy.takes_q = True
    policy.takes_rolls = greedy.takes_rolls

    # answers the probe's search requests with the same network; rollouts play the greedy policy
    search = SR.make_search(net, vocab, dev,
                            lambda obs, legal: greedy(["combat"] * len(obs), obs, legal, [0] * len(obs)),
                            w_hp=a.w_hp, w_act=a.w_act, w_fight=a.w_fight)

    if a.no_fuse:
        policy.takes_rolls = False
        del search.eval_prior
    extra = SR.probe_args(a.rooms, a.mode, a.k, a.mass, a.samples, a.cands, a.search_yield_every,
                          frame_suspend=not a.timer_suspend)
    extra += a.probe_arg
    rsr = None
    if a.run_search:
        rsr = RunSearcher(net, vocab, dev,
                          lambda k_, o_, l_: greedy(k_, o_, l_, [0] * len(o_)),
                          kinds=a.run_search.split(","), k=a.run_k, worlds=a.worlds,
                          boss_plays=a.boss_plays, deck_plays=a.deck_plays, max_live=a.max_live, w_hp=a.w_hp, w_act=a.w_act,
                          w_dmg=a.w_dmg, z=a.z, verify=a.verify, dump=a.run_dump, drop_last=a.verify_drop_last)
        extra += [x for x in rsr.probe_args() if x not in extra]
    for d in glob.glob(a.out + "*"):
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "search_*.json")):
                os.remove(f)
    per_run, res_all = [], []
    t0 = time.time()
    with RunPool(n_envs=a.slots, game_dir=os.path.abspath(a.game), out_dir=a.out, act_cap=3,
                 extra_args=extra, recycle_runs=25, batch_timeout=a.batch_timeout, quiet=not a.loud) as pool:
        for e in range(1, a.eval_iters + 1):
            jobs = [{"run": e * 100000 + r * 1000 + i, "seed": f"EV_{e}_{i}{e * 1000 + r}",
                     "character": CHARS[i % len(CHARS)]}
                    for r in range(a.runs_per_env) for i in range(a.envs)]
            if a.job:
                jobs = [{"run": e * 100000 + n + 1, "seed": j.rsplit(":", 1)[0], "character": j.rsplit(":", 1)[1]}
                        for n, j in enumerate(a.job.split(","))]
            if a.only_runs:
                jobs = [j for j in jobs if str(j["run"]) in a.only_runs.split(",")]
                if not jobs:
                    continue
            by_run = {j["run"]: j for j in jobs}
            pool.run(jobs, policy, deadline_s=a.deadline, search=search, run_search=rsr,
                     on_restart=lambda i: None)
            rr = list(pool.results)
            res_all += rr
            per_run += [{"seed": by_run[r.run]["seed"], "character": by_run[r.run]["character"],
                         "won": bool(r.won), "act": int(r.act), "floors": int(r.floors)}
                        for r in rr if r.run in by_run]
            n = len(res_all)
            rs_line = ""
            if rsr is not None:
                st = pool.rs_stats
                rs_line = (f" | run search: {st['searched']} decisions, {st['rollouts']} rollouts "
                           f"({st['rollouts_failed']} failed), {st['search_s'] / max(1, st['searched']):.0f}s "
                           f"each, changed {rsr.picks['changed']}/{rsr.picks['searched']}"
                           + (f", verify {rsr.verified}" if a.verify else ""))
            print(f"batch {e}: {len(rr)} runs, lost {len(pool.lost)}, leaf evals {pool.leaf_evals}; "
                  f"cumulative win {sum(r.won for r in res_all) / n:.3f} "
                  f"a1 {sum(r.act >= 1 or r.won for r in res_all) / n:.3f} "
                  f"a2 {sum(r.act >= 2 or r.won for r in res_all) / n:.3f} | {time.time() - t0:.0f}s"
                  + rs_line, flush=True)
            pf = pool.prof
            print("  server loop: " + ", ".join(f"{k} {pf[k + '_s']:.0f}s ({pf[k + '_batches'] or pf['batches']} passes)"
                                                for k in ("select", "read", "eval_prior", "roll", "live"))
                  + f" | msgs: rolls {pf['roll_n']} ({pf['fused_roll_n']} in the live pass), priors {pf['prior_n']}, "
                    f"leaves {pf['eval_n']}, live {pf['live_n']}; {pf['batches']} server steps"
                  + (f"; graphs {graphs.stats}" if graphs is not None else ""), flush=True)
    if rsr is not None and rsr.mismatches:
        print("verify mismatches (first 3):", json.dumps(rsr.mismatches[:3])[:3000], flush=True)
    stats = collections.Counter()
    for f in glob.glob(a.out + "*/search_*.json"):
        for k, v in json.load(open(f)).items():
            stats[k] += v
    n = len(res_all)
    print(f"{a.ckpt} rooms={a.rooms or 'none'} run_search={a.run_search or 'none'}: runs={n} "
          f"win={sum(r.won for r in res_all) / n:.3f} "
          f"a1={sum(r.act >= 1 or r.won for r in res_all) / n:.3f} "
          f"a2={sum(r.act >= 2 or r.won for r in res_all) / n:.3f} | searched decisions "
          f"{picks['searched']} (changed the greedy choice {picks['changed']}) | probe {dict(stats)} | "
          f"{time.time() - t0:.0f}s", flush=True)
    name = a.name or (f"search_{os.path.basename(a.ckpt).split('.')[0]}_{(a.rooms or 'none').replace(',', '-')}"
                       + (f"_turn_k{a.k}s{a.samples}" if a.mode == "turn" else "")
                       + f"_{a.pick}" + {"argmax": "", "margin": f"{a.margin:g}", "prior": f"{a.tau:g}"}[a.pick]
                       + (f"_run-{a.run_search.replace(',', '-')}" if a.run_search else ""))
    os.makedirs(os.path.join(HERE, "..", "logs", "eval_runs"), exist_ok=True)
    with open(os.path.join(HERE, "..", "logs", "eval_runs", f"{name}_g.json"), "w") as fh:
        json.dump(per_run, fh)


if __name__ == "__main__":
    main()
