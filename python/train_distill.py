"""M3: distil the r5 teachers (combat + run nets) into one unified model.

Losses (targets from collect_distill.py):
  policy   cross-entropy against the TEACHER'S FULL DISTRIBUTION (soft targets). Combat and run
           decisions count equally, and within the run half each decision kind counts equally.
           Combat is ~80% of decisions and a flat mean would let it drown campfires and shops.
  value    BCE of the value head against "this run was won": V = P(win), the win-only objective
  aux      act_clear, reach_act3 (BCE), floors_left/51 (MSE), fight_won (BCE, combat rows),
           fight_hp (MSE, combat rows with a known HP loss)

The loop count is sampled per batch from --loop-prior. That is what makes every depth usable,
and adaptive stopping (net.choose_loops) depends on it. Held-out evaluation, split BY RUN, reports
at every depth 1..4, plus the adaptive rule's accuracy and average loops, and a boss-fight slice.

    HIP_VISIBLE_DEVICES=0 HSA_ENABLE_SDMA=0 PYTHONPATH=. .venv7/bin/python train_distill.py \\
        --data data/distill --out unified_m3.pt --epochs 4 --bf16
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from alphaslayer.model import use_stable_attention
from alphaslayer.unified import dataset as DS
from alphaslayer.unified import features as FT
from alphaslayer.unified.net import UnifiedNet, choose_loops, param_count

HERE = os.path.dirname(os.path.abspath(__file__))
COMBAT = FT.DKINDS.index("combat")
CONFIGS = {
    "d256_222": dict(d=256, heads=8, prelude=2, core=2, coda=2),
    "d384_242": dict(d=384, heads=8, prelude=2, core=4, coda=2),
    "d512_121": dict(d=512, heads=8, prelude=1, core=2, coda=1),
}


def load_packs(data_dir, vocab_json, workers):
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.jsonl.gz")))
    with Pool(workers) as pool:
        npzs = pool.starmap(DS.pack_shard, [(s, vocab_json) for s in shards])
    return [DS.Packed.load(p) for p in npzs]


def bucketed_batches(index, n_tokens, batch, rng, chunk=32):
    """Shuffled batches of similar length: shuffle, sort within chunks of 32 batches, cut.

    Packed decisions average ~32 tokens but a batch pads to its longest row (~85 with a travel
    decision in it), so grouping by length roughly halves the attention work.
    """
    order = list(range(len(index)))
    rng.shuffle(order)
    out = []
    for s in range(0, len(order), batch * chunk):
        part = sorted(order[s:s + batch * chunk], key=lambda i: n_tokens[i])
        out += [part[j:j + batch] for j in range(0, len(part), batch)]
    rng.shuffle(out)
    return out


class BatchSet(torch.utils.data.Dataset):
    def __init__(self, index, batches):
        self.index, self.batches = index, batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, k):
        return DS.make_batch([self.index[i] for i in self.batches[k]])


def to_dev(x, dev):
    # The DataLoader has already turned the numpy arrays into tensors; direct calls have not.
    return {k: torch.as_tensor(v).to(dev, non_blocking=True) for k, v in x.items()}


def losses(out, tgt, run_weight=0.5):
    """(total, parts). Policy averaged combat-vs-run, run kinds equal within the run half."""
    logp = F.log_softmax(out["logits"].float(), -1)
    ce = -(tgt["tp"] * logp).sum(-1)                              # per row
    kind = tgt["kind"]
    parts = {}
    is_c = kind == COMBAT
    c_loss = ce[is_c].mean() if is_c.any() else ce.new_zeros(())
    run_losses = [ce[kind == k].mean() for k in kind.unique().tolist() if k != COMBAT]
    r_loss = torch.stack(run_losses).mean() if run_losses else ce.new_zeros(())
    if is_c.any() and run_losses:
        pol = (1 - run_weight) * c_loss + run_weight * r_loss
    else:
        pol = c_loss if is_c.any() else r_loss
    parts["policy"] = pol
    parts["value"] = F.binary_cross_entropy_with_logits(out["value_logit"].float(), tgt["won"])
    aux = out["aux"]
    parts["act_clear"] = F.binary_cross_entropy_with_logits(aux["act_clear"].float(), tgt["act_clear"])
    parts["reach_act3"] = F.binary_cross_entropy_with_logits(aux["reach_act3"].float(), tgt["reach_act3"])
    parts["floors_left"] = F.mse_loss(aux["floors_left"].float(), tgt["floors_left"] / 51.0)
    fm = tgt["fight_won"] >= 0
    parts["fight_win"] = (F.binary_cross_entropy_with_logits(aux["fight_win"].float()[fm], tgt["fight_won"][fm])
                          if fm.any() else pol.new_zeros(()))
    hm = ~torch.isnan(tgt["fight_hp"])
    parts["fight_hp"] = (F.mse_loss(aux["fight_hp_loss"].float()[hm], tgt["fight_hp"][hm])
                         if hm.any() else pol.new_zeros(()))
    total = (parts["policy"] + 0.5 * parts["value"]
             + 0.25 * sum(parts[k] for k in ("act_clear", "reach_act3", "floors_left",
                                             "fight_win", "fight_hp")))
    return total, parts


@torch.no_grad()
def evaluate(net, loader, dev, bf16, max_loop=4):
    """Held-out metrics at each depth, plus the adaptive rule. Returns a dict."""
    net.eval()
    acc = collections.defaultdict(lambda: [0.0, 0])     # key -> [sum, n]
    vals = []                                          # (predicted P(win), won) at depth 1
    for inp, tgt in loader:
        inp, tgt = to_dev(inp, dev), to_dev(tgt, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            traj = net.forward_trajectory(inp, max_loop)
        teacher_top = tgt["tp"].argmax(-1)
        kinds = [FT.DKINDS[k] for k in tgt["kind"].tolist()]
        boss = tgt["boss_fight"]
        n_legal = inp["cand_mask"].sum(1)
        real = n_legal > 1                                 # forced moves say nothing
        for k, out in enumerate(traj, start=1):
            logp = F.log_softmax(out["logits"].float(), -1)
            agree = (logp.argmax(-1) == teacher_top).float()
            kl = (tgt["tp"] * (torch.log(tgt["tp"].clamp(min=1e-8)) - logp)).sum(-1)
            for j in range(len(kinds)):
                if not real[j]:
                    continue
                for key in (f"agree/{kinds[j]}/k{k}", f"agree/all/k{k}") + \
                           ((f"agree/boss/k{k}",) if boss[j] else ()):
                    acc[key][0] += float(agree[j]); acc[key][1] += 1
                acc[f"kl/{kinds[j]}/k{k}"][0] += float(kl[j]); acc[f"kl/{kinds[j]}/k{k}"][1] += 1
        v = traj[0]["value"].float()
        vals += list(zip(v.tolist(), tgt["won"].tolist()))
        # Adaptive rule: stakes cap 4 in boss/elite-like high-risk states, else 2.
        risk = 1 - torch.sigmoid(traj[0]["aux"]["fight_win"].float())
        cap = torch.where(risk > 0.2, 4, 2)
        chosen = choose_loops(traj, inp["cand_mask"], cap)
        for j in range(len(kinds)):
            if not real[j]:
                continue
            lg = traj[int(chosen[j]) - 1]["logits"][j]
            acc["adaptive/agree"][0] += float(lg.argmax() == teacher_top[j]); acc["adaptive/agree"][1] += 1
            acc["adaptive/loops"][0] += float(chosen[j]); acc["adaptive/loops"][1] += 1
    res = {k: s / max(1, n) for k, (s, n) in acc.items()}
    res["n/eval"] = acc["agree/all/k1"][1]
    # Value calibration: Brier score and expected calibration error over 10 bins.
    p = np.array([x for x, _ in vals]); y = np.array([w for _, w in vals])
    res["value/brier"] = float(((p - y) ** 2).mean())
    res["value/base_rate"] = float(y.mean())
    bins = np.minimum((p * 10).astype(int), 9)
    res["value/ece"] = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
                                 for b in range(10) if (bins == b).any()))
    net.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/distill")
    ap.add_argument("--vocab", default=os.path.join(HERE, "vocab_m1.json"))
    ap.add_argument("--config", default="d512_121", choices=sorted(CONFIGS))
    ap.add_argument("--out", default="unified_m3.pt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--loop-prior", default="0.4,0.3,0.2,0.1",
                    help="probabilities of loop count 1,2,3,... per training batch")
    ap.add_argument("--heldout", type=float, default=0.1, help="fraction of RUNS held out")
    ap.add_argument("--eval-max", type=int, default=60000, help="held-out decisions evaluated")
    ap.add_argument("--run-weight", type=float, default=0.5)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    use_stable_attention()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = torch.device(a.device)
    vocab = FT.Vocab(json.load(open(a.vocab)))

    t0 = time.time()
    packs = load_packs(a.data, a.vocab, max(1, a.workers))
    index = [(p, i) for p in packs for i in range(p.n)]
    n_tokens = np.array([p.a["n_tokens"][i] for p, i in index])
    keys = np.array([p.a["run_key"][i] for p, i in index])
    train_ix = [i for i in range(len(index)) if keys[i] >= a.heldout]
    val_ix = [i for i in range(len(index)) if keys[i] < a.heldout]
    rng.shuffle(val_ix)
    val_ix = val_ix[:a.eval_max]
    truncated = sum(int(p.a["truncated"][0]) for p in packs)
    print(f"data: {len(index)} decisions in {len(packs)} shards ({time.time() - t0:.0f}s); "
          f"train {len(train_ix)}, held-out {len(val_ix)} (by run); truncated entities {truncated}",
          flush=True)

    tr_index = [index[i] for i in train_ix]
    va_index = [index[i] for i in val_ix]
    val_batches = bucketed_batches(va_index, n_tokens[val_ix], a.batch, random.Random(1))
    val_loader = torch.utils.data.DataLoader(BatchSet(va_index, val_batches), batch_size=None,
                                             num_workers=a.workers, persistent_workers=True)

    cfg = CONFIGS[a.config]
    net = UnifiedNet(vocab.sizes, **cfg).to(dev)
    print(f"model {a.config}: {param_count(net) / 1e6:.1f}M params, cfg {net.cfg}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    prior = [float(x) for x in a.loop_prior.split(",")]
    steps_per_epoch = math.ceil(len(train_ix) / a.batch)
    total_steps = steps_per_epoch * a.epochs
    warm = min(500, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm))
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps)))))

    step = 0
    history = []
    for ep in range(1, a.epochs + 1):
        batches = bucketed_batches(tr_index, n_tokens[train_ix], a.batch, rng)
        loader = torch.utils.data.DataLoader(BatchSet(tr_index, batches), batch_size=None,
                                             num_workers=a.workers, prefetch_factor=4)
        net.train()
        run = collections.defaultdict(float)
        nb = 0
        t_ep = time.time()
        for inp, tgt in loader:
            inp, tgt = to_dev(inp, dev), to_dev(tgt, dev)
            k = rng.choices(range(1, len(prior) + 1), weights=prior)[0]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                out = net(inp, loop=k)
            total, parts = losses(out, tgt, a.run_weight)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()):
                opt.step()
            sched.step()
            step += 1
            nb += 1
            for kk, v in parts.items():
                run[kk] += float(v.detach())
            if step % 200 == 0:
                print(f"  step {step}/{total_steps} " + " ".join(
                    f"{kk}={v / nb:.4f}" for kk, v in run.items())
                      + f" lr={sched.get_last_lr()[0]:.2e} {(time.time() - t_ep) / nb * 1e3:.0f}ms/step",
                      flush=True)
        res = evaluate(net, val_loader, dev, a.bf16)
        res["epoch"] = ep
        res.update({f"train/{kk}": v / max(1, nb) for kk, v in run.items()})
        history.append(res)
        print(f"epoch {ep}: " + " ".join(
            f"{k}={res[k]:.4f}" for k in ("agree/all/k1", "agree/all/k2", "agree/all/k4",
                                           "agree/combat/k1", "agree/boss/k1", "agree/card_reward/k1",
                                           "agree/rest/k1", "agree/travel/k1", "agree/shop/k1",
                                           "agree/event/k1", "adaptive/agree", "adaptive/loops",
                                           "value/brier", "value/ece", "value/base_rate")
            if k in res) + f" ({time.time() - t_ep:.0f}s)", flush=True)
        torch.save({"model": net.state_dict(), "cfg": net.cfg, "sizes": vocab.sizes,
                    "config": a.config, "history": history, "args": vars(a)}, a.out)
    json.dump(history, open(a.out.replace(".pt", ".json"), "w"), indent=1)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
