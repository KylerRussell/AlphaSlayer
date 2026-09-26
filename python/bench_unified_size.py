"""Unified-model size benchmark: rollout latency, training step time and memory per config.

    HIP_VISIBLE_DEVICES=0 HSA_ENABLE_SDMA=0 PYTORCH_ALLOC_CONF=expandable_segments:True \\
        .venv7/bin/python bench_unified_size.py
"""
import gzip, json, os, random, time, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alphaslayer.model import use_stable_attention
from alphaslayer.unified.features import Vocab, encode_batch
from alphaslayer.unified.net import UnifiedNet, to_torch, param_count

use_stable_attention()
dev = torch.device("cuda:0")
v = Vocab(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab_m1.json"))))
rows = [json.loads(l) for l in gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "fixtures", "decisions.jsonl.gz"), "rt")]
random.seed(0)
def batch(n):
    rs = random.sample(rows, n)
    return to_torch(encode_batch([(r["kind"], r["obs"], r["legal"]) for r in rs], v), dev)
b20, b64, b512 = batch(20), batch(64), batch(512)
keepn = sum(b512[k].sum(1) for k in ("card_mask","ent_mask","ppow_mask","orb_mask","relic_mask","pot_mask","map_mask","cand_mask")) + 2
print("packed seq len (512 batch): max", int(keepn.max()), "mean", float(keepn.float().mean()))

CONFIGS = [
    # Plain stacks: depth from unique layers.
    ("d256 2/2/2 x1", dict(d=256, heads=8, prelude=2, core=2, coda=2)),
    ("d384 2/4/2 x1", dict(d=384, heads=8, prelude=2, core=4, coda=2)),
    # Wide + shallow, depth from looping the core (prelude/core/coda x loops).
    ("d512 1/2/1 x1", dict(d=512, heads=8, prelude=1, core=2, coda=1)),
    ("d512 1/2/1 x3", dict(d=512, heads=8, prelude=1, core=2, coda=1, loop=3)),
    ("d640 1/2/1 x1", dict(d=640, heads=10, prelude=1, core=2, coda=1)),
    ("d640 1/2/1 x3", dict(d=640, heads=10, prelude=1, core=2, coda=1, loop=3)),
    ("d768 1/2/1 x3", dict(d=768, heads=12, prelude=1, core=2, coda=1, loop=3)),
]
if len(sys.argv) > 1:
    CONFIGS = [c for c in CONFIGS if any(a in c[0] for a in sys.argv[1:])]

def timeit(fn, n):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n

print(f"{'config':16s} {'eff.depth':>9s} {'params':>8s} {'infer b20':>10s} {'infer b64':>10s} {'train fp32':>11s} {'peak':>6s} {'train bf16':>11s} {'peak':>6s}")
for name, cfg in CONFIGS:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    net = UnifiedNet(v.sizes, **cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    net.eval()
    with torch.no_grad():
        i20 = timeit(lambda: net(b20), 30)
        i64 = timeit(lambda: net(b64), 30)
    net.train()
    def step():
        out = net(b512)
        loss = out["logits"].logsumexp(-1).mean() + out["value_logit"].pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    tr = timeit(step, 10)
    peak = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    def step16():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(b512)
            loss = out["logits"].float().logsumexp(-1).mean() + out["value_logit"].float().pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    tr16 = timeit(step16, 10)
    peak16 = torch.cuda.max_memory_allocated() / 1e9
    depth = cfg["prelude"] + cfg["core"] * cfg.get("loop", 1) + cfg["coda"]
    print(f"{name:16s} {depth:9d} {param_count(net)/1e6:7.1f}M {i20*1e3:8.1f}ms {i64*1e3:8.1f}ms {tr*1e3:9.1f}ms {peak:5.1f}G {tr16*1e3:9.1f}ms {peak16:5.1f}G", flush=True)
    del net, opt
