"""Cross-validates the packed binary format against the JSONL format.

Both are produced by the same deterministic run (same game seed + policy seed), so every
field the two formats share must agree exactly. This is what guards against the C# writer
and the Python reader drifting apart -- a silent field-order mismatch would otherwise train
a model on scrambled observations.

Regenerate the inputs with:
    ./run.sh episodes --probe-episodes 40 --probe-format jsonl && cp .../episodes.jsonl .../cmp.jsonl
    ./run.sh episodes --probe-episodes 40
"""

import gzip
import json
import sys

from alphaslayer.format import ACTION_KINDS, CARD_TOKEN, ENEMY_SCALARS, Step, Terminal, load

BIN = "/tmp/alphaslayer_probe/episodes.bin.gz"
JSONL = "/tmp/alphaslayer_probe/cmp.jsonl"


def main() -> int:
    _, steps, terms = load(BIN)
    js = [json.loads(l) for l in open(JSONL)]
    j_steps = [r for r in js if not r.get("terminal")]
    j_terms = [r for r in js if r.get("terminal")]

    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    check(len(steps) == len(j_steps), f"step count {len(steps)} != {len(j_steps)}")
    check(len(terms) == len(j_terms), f"terminal count {len(terms)} != {len(j_terms)}")

    for i, (b, j) in enumerate(zip(steps, j_steps)):
        o = j["obs"]
        p = o["player"]
        check(b.ep == j["ep"] and b.t == j["t"], f"[{i}] ep/t mismatch")
        check(b.g("hp") == p["hp"], f"[{i}] hp {b.g('hp')} != {p['hp']}")
        check(b.g("block") == p["block"], f"[{i}] block")
        check(b.g("energy") == p["energy"], f"[{i}] energy")
        check(b.g("stars") == p["stars"], f"[{i}] stars")
        check(b.g("gold") == p["gold"], f"[{i}] gold")
        check(b.g("turn") == o["turn"], f"[{i}] turn")
        check(b.g("phase") == o["phase_idx"], f"[{i}] phase")
        check(b.action_idx == j["action_idx"], f"[{i}] action_idx")
        check(b.hp_delta == j["hp_delta"], f"[{i}] hp_delta")

        # hand tokens, field by field
        check(len(b.hand) == len(o["hand"]), f"[{i}] hand len")
        for h, (bt, jt) in enumerate(zip(b.hand, o["hand"])):
            d = dict(zip(CARD_TOKEN, bt))
            for f in ("card_idx", "upgrade", "ench_idx", "ench_amount",
                      "cost", "star_cost", "type", "rarity", "target_type"):
                check(d[f] == jt[f], f"[{i}] hand[{h}].{f}: {d[f]} != {jt[f]}")
            check(bool(d["playable"]) == jt["playable"], f"[{i}] hand[{h}].playable")
            check(bool(d["ench_disabled"]) == jt["ench_disabled"], f"[{i}] hand[{h}].ench_disabled")
            check(bool(d["cost_x"]) == jt["cost_x"], f"[{i}] hand[{h}].cost_x")

        # enemies
        check(len(b.enemies) == len(o["enemies"]), f"[{i}] enemy len")
        for e, (be, je) in enumerate(zip(b.enemies, o["enemies"])):
            for f in ENEMY_SCALARS:
                jv = je[f] if f in je else je[f.replace("_idx", "")]
                check(int(be[f]) == int(jv), f"[{i}] enemy[{e}].{f}: {be[f]} != {jv}")
            check([list(x) for x in be["powers"]] ==
                  [[q["power_idx"], q["amount"]] for q in je["powers"]], f"[{i}] enemy[{e}].powers")
            check([list(x) for x in be["intents"]] ==
                  [[q["type_idx"], q["damage"], q["repeats"]] for q in je["intents"]],
                  f"[{i}] enemy[{e}].intents")

        # relics now carry (idx, counter, melted)
        check(len(b.relics) == len(o["player"]["relics"]), f"[{i}] relic len")
        for k, (br, jr) in enumerate(zip(b.relics, o["player"]["relics"])):
            check(br[0] == jr["relic_idx"], f"[{i}] relic[{k}].idx")
            check(br[1] == jr["counter"], f"[{i}] relic[{k}].counter")
            check(bool(br[2]) == jr["melted"], f"[{i}] relic[{k}].melted")

        # bags
        for name, bag in (("draw_bag", b.draw_bag), ("discard_bag", b.discard_bag),
                          ("exhaust_bag", b.exhaust_bag)):
            jb = {int(k): v for k, v in o[name].items()}
            check(dict(bag) == jb, f"[{i}] {name}: {dict(bag)} != {jb}")

        # legal actions
        check(len(b.legal) == len(j["legal"]), f"[{i}] legal len")
        for a, (ba, ja) in enumerate(zip(b.legal, j["legal"])):
            kind, hand_i, tgt, card = ba
            check(ACTION_KINDS[kind] == ja["kind"], f"[{i}] legal[{a}].kind: {ACTION_KINDS[kind]} != {ja['kind']}")
            check(hand_i == ja["hand"], f"[{i}] legal[{a}].hand")
            check(tgt == ja["target"], f"[{i}] legal[{a}].target")
            check(card == ja["card_idx"], f"[{i}] legal[{a}].card_idx")

    for i, (b, j) in enumerate(zip(terms, j_terms)):
        check(b.ep == j["ep"], f"term[{i}] ep")
        check(b.outcome == j["outcome"], f"term[{i}] outcome {b.outcome} != {j['outcome']}")
        check(b.won == j["won"], f"term[{i}] won")
        check(b.turns == j["turns"] and b.steps == j["steps"], f"term[{i}] turns/steps")
        check(b.hp_start == j["hp_start"] and b.hp_end == j["hp_end"], f"term[{i}] hp")
        check(abs(b.reward - j["reward"]) < 1e-6, f"term[{i}] reward")

    if fails:
        print(f"FAIL ({len(fails)} mismatches). First 10:")
        for f in fails[:10]:
            print("  ", f)
        return 1

    n_fields = sum(len(s.hand) * len(CARD_TOKEN) + len(s.enemies) * 6 + len(s.legal) * 4
                   for s in steps)
    print(f"OK: {len(steps)} steps, {len(terms)} terminals, ~{n_fields} fields cross-checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
