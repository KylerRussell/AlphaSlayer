"""Per-decision-kind drift of a checkpoint from its anchor, on the fixed fixture decisions.

    HIP_VISIBLE_DEVICES=0 PYTHONPATH=. .venv7/bin/python drift_unified.py ANCHOR.pt CHECKPOINT.pt

KL from the anchor, entropy before and after, and how often the top action changed, per kind.
Rising entropy with an unchanged top action is a random walk, not learning (shaped3).
"""
import gzip, json, sys, collections, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from alphaslayer.model import use_stable_attention
from alphaslayer.unified import features as FT
from alphaslayer.unified.net import UnifiedNet, to_torch
use_stable_attention(); dev = torch.device("cuda:0")
def load(p):
    ck = torch.load(p, map_location=dev, weights_only=False)
    n = UnifiedNet(ck["sizes"], **ck["cfg"]).to(dev).eval(); n.load_state_dict(ck["model"]); return n
anchor, cur = load(sys.argv[1]), load(sys.argv[2])
vocab = FT.Vocab(json.load(open("vocab_m1.json")))
rows = [json.loads(l) for l in gzip.open("tests/fixtures/decisions.jsonl.gz", "rt")]
st = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0])   # n, kl, ent_anchor, ent_cur, argmax changed
with torch.no_grad():
    for s in range(0, len(rows), 256):
        ch = rows[s:s + 256]
        b = to_torch(FT.encode_batch([(r["kind"], r["obs"], r["legal"]) for r in ch], vocab), dev)
        la = F.log_softmax(anchor(b)["logits"].float(), -1); lc = F.log_softmax(cur(b)["logits"].float(), -1)
        m = b["cand_mask"]
        kl = (la.exp() * (la - lc)).masked_fill(~m, 0).sum(-1)
        ea = -(la.exp() * la).masked_fill(~m, 0).sum(-1); ec = -(lc.exp() * lc).masked_fill(~m, 0).sum(-1)
        chg = la.argmax(-1) != lc.argmax(-1)
        for j, r in enumerate(ch):
            if len(r["legal"]) < 2: continue
            x = st[r["kind"]]; x[0] += 1; x[1] += float(kl[j]); x[2] += float(ea[j]); x[3] += float(ec[j]); x[4] += int(chg[j])
print(f"{'kind':12s} {'n':>5s} {'KL':>7s} {'ent anchor':>10s} {'ent now':>8s} {'top changed':>11s}")
for k, (n, kl, ea, ec, c) in sorted(st.items(), key=lambda kv: -kv[1][1] / kv[1][0]):
    print(f"{k:12s} {n:5d} {kl/n:7.3f} {ea/n:10.3f} {ec/n:8.3f} {c/n:10.1%}")
