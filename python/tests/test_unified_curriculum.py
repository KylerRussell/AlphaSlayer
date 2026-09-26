"""Unified curriculum tests: loss-weighted targeting DELIVERS the bosses asked for, and an
isolated fight ends at the value of the run state it leaves.

Unit (no game):
  * EncounterStats is an EMA; encounters the model loses get proportionally more drills
  * choose_fights prefers decks that actually entered the targeted encounter
  * patch_context restores the harvested run context without mutating its input
  * post_fight_obs is the harvested run state with the fight's HP, and it encodes

Live (game + GPU, --live):
  * every fight played IS the targeted encounter (read from the fight's own observation)
  * the model saw the harvested run's floor and character, not the deck server's
  * a won fight's last step targets exactly V(post-fight state), recomputed independently;
    a lost fight's targets 0
  * an encounter outside the act's pool fails loudly (RuntimeError), never substitutes

    PYTHONPATH=. .venv7/bin/python tests/test_unified_curriculum.py [--live]
"""
from __future__ import annotations

import collections
import copy
import glob
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphaslayer.unified import curriculum as CU
from alphaslayer.unified import features as FT
from alphaslayer.unified.net import to_torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


vocab = FT.Vocab(json.load(open(os.path.join(ROOT, "vocab_m1.json"))))

print("== unit")
st = CU.EncounterStats(beta=0.5, prior=0.5)
st.update("X", True)
st.update("X", True)
check("EMA update", abs(st.ema["X"] - 0.875) < 1e-12, f"{st.ema['X']}")
check("vocab encounter lookup", vocab.enc_ix.get("QUEEN_BOSS", 0) > 0 and vocab.enc_ix.get("AEONGLASS_BOSS") == 1,
      f"QUEEN={vocab.enc_ix.get('QUEEN_BOSS')}")


def rec(act, room, enc, tag):
    return {"act": act, "room": room, "encounter": enc, "character": "IRONCLAD", "cards": [tag],
            "upgrades": [0], "relics": [], "hp": 40, "max_hp": 80,
            "player": {"total_floor": 40, "act_floor": 7, "gold": 123, "boss": "QUEEN_BOSS",
                       "hp": 40, "max_hp": 80, "deck_cards": [], "relics": [], "potions": []}}


records = ([rec(2, "Boss", "QUEEN_BOSS", "q") for _ in range(20)]
           + [rec(2, "Boss", "TEST_SUBJECT_BOSS", "t") for _ in range(20)])
stats = CU.EncounterStats()
stats.ema.update({"QUEEN_BOSS": 0.05, "TEST_SUBJECT_BOSS": 0.95})
picks = CU.choose_fights(records, stats, 2000, acts=(2,), mix=(1, 0, 0), rng=random.Random(1))
cnt = collections.Counter(e for _, _, e in picks)
want = (1 - 0.05 + 0.05) / (1 - 0.95 + 0.05)          # loss weights 1.0 vs 0.1
check("encounters drilled in proportion to how often they are lost",
      abs(cnt["QUEEN_BOSS"] / cnt["TEST_SUBJECT_BOSS"] - want) < 0.25 * want,
      f"{dict(cnt)} (want ratio {want:.1f})")
check("the deck is one that entered the targeted encounter",
      all(r["encounter"] == e for r, _, e in picks))

weak = [rec(1, "Monster", "TUNNELER_WEAK", "w") for _ in range(30)] + \
       [rec(1, "Monster", "BOWLBUGS_NORMAL", "b") for _ in range(30)]
pk = CU.choose_fights(weak, CU.EncounterStats(), 300, acts=(1,), mix=(0, 0, 1),
                      rng=random.Random(3), pools=vocab.act_pools)
check("act pools loaded from the vocab", ("TUNNELER_WEAK" not in vocab.act_pools.get((1, "regular"), {"TUNNELER_WEAK"}))
      and "QUEEN_BOSS" in vocab.act_pools.get((2, "boss"), ()))
check("targets are restricted to encounters in the act's pool (no weak hallway fights)",
      pk and all(e == "BOWLBUGS_NORMAL" for _, _, e in pk), str(collections.Counter(e for _, _, e in pk)))

obs = {"total_floor": 0, "act_floor": 0, "boss": "X", "boss_idx": 3, "character": "DEFECT",
       "player": {"gold": 0, "hp": 5}}
before = copy.deepcopy(obs)
pt = CU.patch_context(obs, records[0], vocab.enc_ix)
check("patch_context restores the harvested floor, boss, character and gold",
      pt["total_floor"] == 40 and pt["act_floor"] == 7 and pt["boss"] == "QUEEN_BOSS"
      and pt["boss_idx"] == vocab.enc_ix["QUEEN_BOSS"] and pt["character"] == "IRONCLAD"
      and pt["player"]["gold"] == 123)
check("...without mutating the observation it was given", obs == before)
k, o, l = CU.post_fight_obs(records[0], 17, 80)
check("post-fight state carries the HP the fight left, on a copy",
      o["hp"] == 17 and records[0]["player"]["hp"] == 40 and k == "card_reward")
FT.encode_batch([(k, o, l)], vocab)
check("post-fight state encodes", True)

if "--live" in sys.argv:
    print("\n== live (game + GPU)")
    from alphaslayer.deckeval import DeckSpec, VecDeckEval
    from alphaslayer.model import use_stable_attention
    from alphaslayer.unified.net import UnifiedNet
    use_stable_attention()
    dev = torch.device("cuda:0")
    ck = torch.load(os.path.join(ROOT, "unified_m3_d512_121.pt"), map_location=dev, weights_only=False)
    net = UnifiedNet(ck["sizes"], **ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])

    # Real run states to harvest from: card-reward decisions in acts 2 and 3 of the M3 data.
    live_recs = []
    bosses = {1: ["KAISER_CRAB_BOSS", "KNOWLEDGE_DEMON_BOSS", "THE_INSATIABLE_BOSS"],
              2: ["QUEEN_BOSS", "AEONGLASS_BOSS", "TEST_SUBJECT_BOSS"]}
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "distill", "shard_*.jsonl.gz")))[:3]:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r["kind"] != "card_reward":
                    continue
                pl = r["obs"]
                if pl.get("act") in (1, 2) and pl.get("deck_cards"):
                    for enc in bosses[pl["act"]]:
                        live_recs.append({
                            "act": pl["act"], "room": "Boss", "encounter": enc,
                            "character": pl["character"],
                            "cards": [c["card"] for c in pl["deck_cards"]],
                            "upgrades": [int(c.get("up", 0) or 0) for c in pl["deck_cards"]],
                            "relics": [x["relic"] for x in pl.get("relics") or []],
                            "hp": pl["hp"], "max_hp": pl["max_hp"], "player": pl})
                if len(live_recs) > 600:
                    break
    st = CU.EncounterStats()
    st.ema.update({"QUEEN_BOSS": 0.05, "AEONGLASS_BOSS": 0.1, "TEST_SUBJECT_BOSS": 0.97,
                   "KAISER_CRAB_BOSS": 0.25, "KNOWLEDGE_DEMON_BOSS": 0.7, "THE_INSATIABLE_BOSS": 0.4})
    picks = CU.choose_fights(live_recs, st, 40, acts=(1, 2), mix=(1, 0, 0), rng=random.Random(2))
    pools = {act: VecDeckEval(n_envs=5, characters=["IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"],
                              seed=f"TESTCUR{act}_", out_dir=f"/tmp/alphaslayer_testcur{act}_",
                              act=act, stall_timeout=90.0)
             for act in (1, 2)}
    try:
        steps, res = CU.play_fights(net, pools, picks, vocab, vocab.enc_ix, dev, bf16=True)
        check("every requested fight was played", len(res) == len(picks), f"{len(res)}/{len(picks)}")
        # Counted from what each fight's OWN observations say it was, not from what was asked.
        played, pos = collections.Counter(), 0
        for enc, won, n, *_ in res:
            if n:
                played[steps[pos].obs.get("encounter")] += 1
            pos += n
        check("loss-weighting delivered: Queen actually played far more than Test Subject",
              played["QUEEN_BOSS"] > 3 * max(1, played["TEST_SUBJECT_BOSS"]), str(dict(played)))
        pos, wrong_enc, bad_target, indep = 0, 0, [], []
        for enc, won, n, v_end, r, hp_end, mx in res:
            fight = steps[pos:pos + n]
            pos += n
            wrong_enc += sum(1 for s in fight if s.obs.get("encounter") != enc)
            if fight:
                want = v_end if won else 0.0
                if abs(fight[-1].ret - want) > 1e-5:
                    bad_target.append((enc, won, fight[-1].ret, want))
            if won:
                # Independent recomputation: build the post-fight state from the record and the
                # HP the fight left, score it, and compare with the terminal the trainer used.
                k, o, l = CU.post_fight_obs(r, hp_end, mx)
                b = to_torch(FT.encode_batch([(k, o, l)], vocab), dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    indep.append(abs(float(net(b)["value"][0]) - v_end))
        check("every fight's observations name the targeted encounter", wrong_enc == 0,
              f"{wrong_enc} decisions in the wrong fight")
        check("the model saw the harvested run's floor, not the deck server's",
              all((s.obs.get("total_floor") or 0) > 0 for s in steps)
              and {s.obs.get("total_floor") for s in steps} <= {r["player"]["total_floor"] for r in live_recs})
        check("each fight's last step targets its terminal value (V(post) if won, 0 if lost)",
              not bad_target, str(bad_target[:3]))
        check("won fights' terminal = V(post-fight state), recomputed independently",
              indep and max(indep) < 1e-3, f"{len(indep)} won fights, max diff {max(indep) if indep else None}")
        try:
            bad = copy.deepcopy(picks[0])
            CU.play_fights(net, {bad[0]["act"]: pools[bad[0]["act"]]},
                           [(bad[0], "boss", "NOT_A_REAL_BOSS")], vocab, vocab.enc_ix, dev)
            check("an encounter outside the pool fails loudly", False, "no error raised")
        except RuntimeError as e:
            check("an encounter outside the pool fails loudly", "NOT_A_REAL_BOSS" in str(e),
                  str(e)[:100])
    finally:
        for p in pools.values():
            p.close()

print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
sys.exit(1 if FAILS else 0)
