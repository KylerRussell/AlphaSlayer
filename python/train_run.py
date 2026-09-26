"""End-to-end run training: a run policy and a combat policy, learning together.

Two policies, two objectives, on purpose.

  * The RUN policy (map, card rewards, campfires, events, shops, the potion gate) is credited
    from how the RUN ends. That is the thing it controls and the only signal that can teach it
    that a card taken on floor 3 is why the act boss was survivable.

  * The COMBAT policy keeps the objective it was TRAINED on: win this fight, preserving hp
    (`won + 0.5 * hp_end/max_hp`, identical to train_rl.py). It is not retrained on run
    outcome. Switching it to a run-level return would change what it is optimising, from a
    dense per-fight signal to one sparse bit forty floors away, and that is the most likely
    way to destroy a policy that already plays fights well.

Three further guards on the combat policy, because "keep learning without harming what we
have" is the explicit requirement:

  1. ``--freeze-combat`` trains the run policy alone. Nothing can regress; use it first to get
     a run policy off the ground against a known-good fighter.
  2. When it is unfrozen it trains at a much lower learning rate than the run policy and under
     a KL penalty toward a FROZEN COPY of the checkpoint it started from -- a trust region
     around the known-good policy rather than around the previous iteration.
  3. A regression gate re-evaluates it on the ORIGINAL combat benchmark every N iterations and
     rolls the weights back if the win rate falls more than a tolerance below the baseline
     measured at startup. Drift is then bounded by measurement, not by hope.
"""

from __future__ import annotations

import argparse
import copy
import random
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from alphaslayer.cardstats import CardStats
from alphaslayer.modcheck import check as modcheck
from alphaslayer.benchmark import BenchmarkPool, load as bench_load
from alphaslayer import curriculum as curr
from alphaslayer.critic import Critic, ROOMS, encode_decks
from alphaslayer.encoder import Encoder, step_from_json
from alphaslayer.expert import (ExpertEvaluator, build_card_reward_candidates,
                                build_shop_candidates)
from alphaslayer.model import CombatNet, use_stable_attention
from alphaslayer.runenv import VecRunEnv
from alphaslayer.runmodel import migrate_state_scalars, RunNet, bind_obs, encode_actions, encode_state
from alphaslayer.vecenv import VecSpireEnv
from train_rl import load_compat


class Buffer:
    """Transitions awaiting the outcome that will credit them.

    Combat and run steps are held in SEPARATE queues per env because they are closed out by
    different events: a combat step is settled by the fight it belongs to, a run step only at
    the end of the run.

    Run steps carry a PER-STEP reward and are credited by a discounted backward pass, not by
    one flat terminal number. That is what makes the discount schedule below meaningful: with
    a flat return every decision in a run is credited identically, and no value of gamma
    changes anything.
    """

    def __init__(self):
        self.open_fight = defaultdict(list)
        self.open_run = defaultdict(list)
        self.fight_done = []
        self.run_done = []
        self.last_act = {}

    def add_fight(self, env, step):
        self.open_fight[env].append(step)

    def add_run(self, env, step, reward_prev=0.0):
        """Appends a run step, crediting the PREVIOUS one for surviving to reach this point."""
        if reward_prev and self.open_run[env]:
            self.open_run[env][-1]["r"] += reward_prev
        step.setdefault("r", 0.0)
        self.open_run[env].append(step)

    def credit_last(self, env, r):
        if self.open_run[env]:
            self.open_run[env][-1]["r"] += r

    def finish_fight(self, env, ret, phi_end=0.0, shaping=0.0):
        """Credits every step of one fight.

        Without shaping each step gets the SAME terminal number, so a block card that
        prevented 15 damage on turn 3 is credited identically to the last card played --
        the exact defect that potential-based shaping fixed in the combat-only trainer
        (+0.081 win there). Here the fights come from real runs, so the fix applies on the
        distribution we actually care about rather than a curriculum that overfits away
        from it.
        """
        for s in self.open_fight[env]:
            s["ret"] = ret + shaping * (phi_end - s.get("cphi", 0.0))
        self.fight_done.extend(self.open_fight[env])
        self.open_fight[env] = []

    def finish_run(self, env, terminal_r, gamma, phi_end=0.0, hp_shaping=0.0,
                   deck_end=0.0, deck_shaping=0.0, deck_target=12.0, gae_lambda=0.0,
                   prog_shaping=0.0, prog_end=0.0):
        """Discounted backward pass over the run's decisions.

        return_t = r_t + gamma * return_{t+1}

        With gamma near 0 a decision is judged almost entirely by what happens immediately
        after it; with gamma near 1 it carries the whole rest of the run. See the schedule in
        main() for why that changes over training.

        Potential-based HP shaping is applied first, as F(s, s') = w * (gamma * phi(s') -
        phi(s)) with phi = hp / max_hp. This is the Ng et al. (1999) form, whose sum
        telescopes to a constant plus a discounted terminal term -- so it does not change
        which policy is optimal, it only moves credit to the decision that caused the damage.
        That matters because the trained policy smithed at a campfire at 17% hp: nothing in
        the previous reward made losing health cost anything until the run was already over,
        so there was no gradient telling it to heal.
        """
        steps = self.open_run[env]
        if steps and deck_shaping:
            # Potential-based on deck SIZE, phi_d = -lambda * cards.
            #
            # Taking a card is currently free in the reward: dilution only shows up much
            # later, as fights lost with a deck that cannot find its win condition. This makes
            # the cost immediate and local to the decision that caused it, which is what the
            # take/skip choice needs to be learnable at all.
            #
            # It is guidance, not a rule: a card that earns more than it costs is still taken.
            # Being honest about the theory -- with a non-zero potential at the terminal state
            # this is not strictly policy-invariant, so it does apply a mild pull toward
            # smaller decks. That pull is the point, and lambda is small enough that one good
            # card outweighs it.
            def _phi_d(n):
                # CONVEX in deck size: phi = -lambda * max(0, n - target)^2.
                #
                # A linear potential was the bug. Its marginal cost is lambda per card no
                # matter how big the deck already is, so the reward could not express "this
                # deck is full enough" -- and the measured skip rate came out flat at ~67%
                # across every deck size, which is a forced rate, not a judgement.
                #
                # Squared past a threshold makes the marginal cost of the next card grow with
                # the deck: roughly 2*lambda*(n - target). Near the starting deck a card is
                # almost free; at 25 cards it is expensive. That is the state-dependence the
                # decision needs, and it comes from the shape of the potential rather than
                # from pushing the policy toward any particular rate.
                over = max(0.0, n - deck_target)
                return -(over * over)
            for i in range(len(steps) - 1):
                steps[i]["r"] += deck_shaping * (
                    gamma * _phi_d(steps[i + 1]["deck"]) - _phi_d(steps[i]["deck"]))
            steps[-1]["r"] += deck_shaping * (
                gamma * _phi_d(deck_end) - _phi_d(steps[-1]["deck"]))
        if steps and prog_shaping:
            # POTENTIAL-BASED run progress, the principled replacement for --floor-bonus and
            # --act-bonus.
            #
            # Those are DIRECT rewards: they pay for depth whether or not depth was the right
            # choice, so they genuinely change the optimal policy (a potential provably does
            # not, Ng et al. 1999). With phi = floor count and gamma near 1 the per-step term
            # is approximately the same magnitude as the old bonus, so this is a clean
            # substitution rather than a different reward scale -- what changes is that the
            # sum telescopes and the terminal is handled consistently.
            for i in range(len(steps) - 1):
                steps[i]["r"] += prog_shaping * (
                    gamma * steps[i + 1]["prog"] - steps[i]["prog"])
            steps[-1]["r"] += prog_shaping * (gamma * prog_end - steps[-1]["prog"])
        if steps and hp_shaping:
            for i in range(len(steps) - 1):
                steps[i]["r"] += hp_shaping * (gamma * steps[i + 1]["phi"] - steps[i]["phi"])
            steps[-1]["r"] += hp_shaping * (gamma * phi_end - steps[-1]["phi"])
        if steps:
            steps[-1]["r"] += terminal_r
        if gae_lambda > 0.0:
            # GAE(lambda): A_t = delta_t + (gamma*lambda) * A_{t+1}, with
            # delta_t = r_t + gamma*V(s_{t+1}) - V(s_t).
            #
            # Monte Carlo (lambda = 1) credits every decision in a run with the SAME noisy
            # sum, so a good card choice on floor 3 is rewarded or punished by a boss fight
            # 13 floors later that it barely influenced. The value bootstrap cuts that chain:
            # each decision is judged against what the critic expected from the state it
            # reached, which is exactly the density the act-1 curriculum bought by shortening
            # the horizon -- but without shortening it.
            #
            # The run always ends at a terminal state (win or death), never a time cut, so
            # V(s_{T+1}) = 0 is correct rather than an approximation.
            a = 0.0
            nxt_v = 0.0
            for st in reversed(steps):
                delta = st["r"] + gamma * nxt_v - st["v"]
                a = delta + gamma * gae_lambda * a
                st["adv"] = a
                # The value head is regressed on adv + V, the lambda-return. Regressing it on
                # the MC return instead would train the critic on a different target from the
                # one the advantage was computed against.
                st["ret"] = a + st["v"]
                nxt_v = st["v"]
        else:
            g = 0.0
            for st in reversed(steps):
                g = st["r"] + gamma * g
                st["ret"] = g
        self.run_done.extend(steps)
        self.open_run[env] = []
        self.last_act.pop(env, None)
        # A run that ends mid-fight leaves combat steps with no fight_end. They have no
        # outcome, so they must be dropped rather than trained on with an invented reward.
        self.open_fight[env] = []

    def drop_open(self):
        self.open_fight.clear()
        self.open_run.clear()
        self.last_act.clear()


def run_progress(obs):
    """The progress-shaping potential of a run-level observation: the total floor.

    Travel observations carry it at the top level, every other kind inside the player dict.
    Before the probe put it in the player dict, it read as 0 on every non-travel decision.
    """
    pl = obs.get("player", obs)
    return float(obs.get("total_floor", pl.get("total_floor", 0)) or 0)


def harvest_record(pl, fr):
    """One entry-deck record for the combat curriculum, or None if there is no deck to record.

    ``pl`` is the player dict from the last run-level decision before the fight; ``fr`` is the
    fight's FightResult.

    The act comes from the FIGHT itself. It used to be read from ``pl``, but only the
    potion_gate observation carried an act at that level, so every fight not preceded by a
    gate was labelled act 0: ~35% of act-2/3 encounters in the r5 harvest, which starved the
    act-3 curriculum pool and put late-game decks into the act-1 pool. Unknown stays None
    rather than defaulting to 0, and load_decks drops None whenever it filters by act.
    """
    if not pl or not pl.get("deck_cards"):
        return None
    act = getattr(fr, "act", -1)
    if act is None or act < 0:
        act = pl.get("act")
    return {
        "character": pl.get("character"),
        "room": fr.room, "encounter": fr.encounter,
        # 0-based act, so a harvest can be filtered to act 2 rather than inferred from
        # encounter names.
        "act": None if act is None else int(act),
        "cards": [c["card"] for c in pl["deck_cards"]],
        "upgrades": [int(c.get("up", 0) or 0) for c in pl["deck_cards"]],
        "relics": [r["relic"] for r in (pl.get("relics") or [])],
        "hp": fr.hp_start, "max_hp": fr.max_hp, "won": bool(fr.won),
        # Enough of the run state around the fight to rebuild the post-fight state the unified
        # trainer bootstraps an isolated curriculum fight from.
        "act_floor": pl.get("act_floor"), "total_floor": pl.get("total_floor"),
        "gold": pl.get("gold"), "boss": pl.get("boss"),
        "potions": [x.get("potion") for x in (pl.get("potions") or []) if isinstance(x, dict)],
    }


BATCH_KEYS = ("hand", "hand_mask", "enemies", "enemy_mask", "relics", "relic_mask",
              "powers", "power_mask", "enemy_powers", "enemy_power_mask",
              "bags", "globals", "actions", "action_mask")

_ENC = None    # built once the checkpoint's card-vocabulary size is known


def combat_forward(net, obs_list, legal_list, device):
    """Encodes exactly as train_rl.py does, so the combat policy sees identical features."""
    steps = [step_from_json(o, l) for o, l in zip(obs_list, legal_list)]
    b = _ENC.encode(steps)
    tb = {k: torch.from_numpy(b[k]).to(device) for k in BATCH_KEYS}
    out = net(tb)
    return out["logits"], out["value"], tb["action_mask"].bool()


def run_forward(net, kinds, obs_list, legal_list, device):
    bind_obs(legal_list, obs_list)
    state = encode_state(kinds, obs_list, device)
    actions = encode_actions(kinds, legal_list, device)
    logits, value = net(state, actions)
    return logits, value, actions[-1]


def sample(logits, mask, temp=1.0):
    logp = F.log_softmax(logits / max(1e-6, temp), dim=-1)
    probs = logp.exp()
    probs = torch.nan_to_num(probs, 0.0).clamp(min=0)
    probs = torch.where(mask, probs, torch.zeros_like(probs))
    s = probs.sum(-1, keepdim=True)
    probs = torch.where(s > 0, probs / s, mask.float() / mask.sum(-1, keepdim=True).clamp(min=1))
    idx = torch.multinomial(probs, 1).squeeze(-1)
    return idx, logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def eval_combat_benchmark(net, device, envs, episodes, characters, relics, mix, ascension):
    """The ORIGINAL combat benchmark, used as the regression gate.

    Deliberately the same configuration the combat policy was trained and measured on, so a
    number here is comparable with the numbers already on record rather than a new metric that
    happens to look fine.
    """
    net.eval()
    per = max(1, episodes // envs)

    def policy(obs, legal, idxs):
        logits, _, mask = combat_forward(net, obs, legal, device)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
        return logits.argmax(-1).tolist()

    # characters arrives as a comma string; VecSpireEnv indexes the roster per env, so a bare
    # string would hand each env a single LETTER as its character id and silently produce zero
    # episodes - which is exactly what a regression gate must never do quietly.
    roster = characters.split(",") if isinstance(characters, str) else list(characters)
    with VecSpireEnv(n_envs=envs, characters=roster, episodes_per_env=per,
                     # FIXED seed: every gate evaluation replays the same episodes, so two
                     # measurements differ only by the policy. With a varying seed the draw
                     # itself moved the number by ~0.04 between runs of identical weights
                     # (0.949 / 0.912 on the same checkpoint), which is larger than the
                     # regression the gate is supposed to detect.
                     seed="GATEFIXED_", encounters="mix", mix=mix,
                     inject_relics=relics.split(",") if relics else None,
                     reroll_deck=True, ascension=ascension) as venv:
        venv.run(policy)
        wr = venv.win_rate
        n = len(venv.results)
    net.train()
    if n == 0:
        raise RuntimeError("combat regression gate produced no episodes; refusing to "
                           "report a win rate of 0 as if it were a measurement")
    # Standard error of the win-rate estimate. With the seed fixed this is a PAIRED
    # comparison, so it is the right scale for "did the policy get worse"; it would badly
    # understate the noise if the episode set were allowed to change between checks.
    se = (wr * (1 - wr) / n) ** 0.5
    return wr, se, n


def eval_realistic_benchmark(net, device, bench_path, envs, fights_per_deck, ascension):
    """The REPLACEMENT gate: decks the run policy actually reached, against the room they
    were actually about to fight, weighted by where runs actually die.

    The original gate scored random Pandora's-Box decks. That measured a distribution the
    policy never plays, and it sat at 0.92-0.95 because it was dominated by regular fights
    that are already won ~99% of the time -- so it had almost no resolution exactly where the
    headroom is. This one is ~69% boss by weight.
    """
    net.eval()

    def policy(obs, legal):
        with torch.no_grad():
            lg, _, mask = combat_forward(net, obs, legal, device)
            lg = lg.masked_fill(~mask, torch.finfo(lg.dtype).min / 4)
            return lg.argmax(-1).tolist()

    pool = BenchmarkPool(bench_load(bench_path), n_envs=envs, ascension=ascension)
    try:
        wr, se, per_room, n = pool.measure(policy, fights_per_deck=fights_per_deck,
                                           close_after_each=True)
    finally:
        pool.close()
    net.train()
    if n == 0:
        raise RuntimeError("realistic combat gate produced no fights; refusing to report a "
                           "win rate of 0 as if it were a measurement")
    detail = " ".join(f"{r}={p:.3f}" for r, (p, _) in sorted(per_room.items()))
    return wr, se, n, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat-ckpt", default="combat_capsule.pt")
    ap.add_argument("--combat-ref", default=None,
                    help="checkpoint the KL trust region is anchored to; defaults to "
                         "--combat-ckpt. On a RESUME this must stay the ORIGINAL checkpoint: "
                         "anchoring to already-drifted weights lets drift compound across "
                         "restarts, which is exactly what the trust region exists to prevent")
    ap.add_argument("--run-ckpt", default=None, help="resume the run policy")
    ap.add_argument("--out", default="run_policy.pt")
    ap.add_argument("--combat-out", default="combat_run.pt")
    ap.add_argument("--vocab", default="/tmp/alphaslayer_probe/vocab.json")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--runs-per-iter", type=int, default=48)
    ap.add_argument("--characters", default="IRONCLAD,SILENT,DEFECT,NECROBINDER,REGENT")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--act-cap", type=int, default=0,
                    help="end a run after clearing this many acts (0 = the full game). "
                         "1 makes wins frequent enough to actually learn from")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--run-lr", type=float, default=3e-4)
    ap.add_argument("--combat-lr", type=float, default=1e-5,
                    help="deliberately ~30x below the run lr; the combat policy is already good")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.02,
                    help="run-policy entropy bonus (combat keeps its own, lower, appetite)")
    ap.add_argument("--per-kind-loss", action="store_true",
                    help="option 2: average the PPO surrogate per decision kind so rare "
                         "decisions get equal gradient share (biases the objective)")
    ap.add_argument("--no-per-kind-entropy", dest="per_kind_entropy", action="store_false",
                    help="revert to a flat entropy mean over the batch")
    ap.add_argument("--per-kind-heads", action="store_true",
                    help="option 3: a separate scoring and value head per decision kind")
    ap.add_argument("--deck-shaping", type=float, default=0.02,
                    help="per-card cost via a deck-size potential; makes the price of taking "
                         "a card immediate instead of 20 floors away. 0 disables")
    ap.add_argument("--critic-fights", type=int, default=3,
                    help="fights per room type per candidate, at iteration 1")
    ap.add_argument("--critic-anneal-iters", type=int, default=120,
                    help="iterations over which the expert's fight budget falls 3 -> 0; after "
                         "that the critic answers alone, for free")
    ap.add_argument("--critic-envs", type=int, default=4, help="eval processes per character")
    ap.add_argument("--critic-follow", type=float, default=1.0,
                    help="probability the best-scoring candidate overrides the run policy")
    ap.add_argument("--critic-follow-anneal", type=int, default=200,
                    help="iterations over which --critic-follow decays to 0, handing the "
                         "decision back to the run policy")
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--critic-states-per-iter", type=int, default=60,
                    help="pickup states the expert labels each iteration; each costs "
                         "roughly (candidates x 3 rooms x --critic-fights) fights")
    ap.add_argument("--critic-warmup", type=int, default=2000,
                    help="measured candidates required before the critic is trusted to choose")
    ap.add_argument("--critic-out", default="critic.pt")
    ap.add_argument("--card-stats", default="card_stats.json")
    ap.add_argument("--gate-bench", default=None,
                    help="path to a realistic run-deck benchmark (see alphaslayer/benchmark.py). "
                         "When set, the combat regression gate uses it INSTEAD of the random "
                         "Pandora's-Box benchmark")
    ap.add_argument("--gate-bench-fights", type=int, default=2,
                    help="fights per benchmark deck per gate measurement")
    ap.add_argument("--gate-readonly", action="store_true",
                    help="MEASURE the combat benchmark each interval but never roll back. "
                         "Implied by --from-scratch, where the reference is a fully trained "
                         "model the scratch policy cannot match for a long time -- rolling "
                         "back there resets the policy to its random init on every check")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout for BOTH nets (run-net representation MLPs, combat "
                         "transformer). Rollout always runs in eval() mode so the recorded "
                         "logp is well defined; dropout applies only to the update")
    ap.add_argument("--from-scratch", action="store_true",
                    help="randomly initialise BOTH nets, using the checkpoints only for vocab "
                         "sizes and architecture. Also forces --combat-kl 0 (the reference "
                         "would be a random net) and disables gate rollback (there is no "
                         "prior capability to protect)")
    ap.add_argument("--gae-lambda", type=float, default=0.0,
                    help="GAE lambda for the RUN policy; 0 keeps Monte Carlo returns. "
                         "0.95 is the usual setting")
    ap.add_argument("--harvest-decks", default=None,
                    help="write the deck/relic state observed on entry to each harvested "
                         "fight to this JSONL file, to build a realistic combat benchmark")
    ap.add_argument("--harvest-rooms", default="Boss,Elite,Monster",
                    help="which room types to harvest entry decks for")
    ap.add_argument("--harvest-cap", type=int, default=4000,
                    help="stop harvesting after this many records")
    ap.add_argument("--no-expert", action="store_true", help="disable counterfactual scoring")
    ap.add_argument("--deck-target", type=float, default=12.0,
                    help="deck size below which cards are effectively free")
    ap.add_argument("--skip-ent-coef", type=float, default=0.005,
                    help="entropy bonus on the binary TAKE-vs-SKIP marginal of a card reward")
    ap.add_argument("--hp-shaping", type=float, default=0.5,
                    help="weight on potential-based hp shaping; 0 disables")
    ap.add_argument("--gamma-lo", type=float, default=0.95,
                    help="discount on RUN decisions while the policy is still dying to "
                         "immediate mistakes; credit stays near the decision that caused it")
    ap.add_argument("--gamma-hi", type=float, default=0.999,
                    help="discount once runs get deep, when losses really are caused by "
                         "choices made tens of floors earlier")
    ap.add_argument("--gamma-target-floors", type=float, default=45.0,
                    help="mean floors at which the discount has fully annealed to --gamma-hi")
    ap.add_argument("--max-steps-per-iter", type=int, default=60000,
                    help="safety valve on rollout size; unbounded collection is what let one "
                         "iteration balloon into swap and cost 2.7 hours")
    ap.add_argument("--collect-timeout", type=float, default=600.0,
                    help="hard wall-clock cap on one collection pass")
    # ---- automatic combat curriculum -----------------------------------------
    ap.add_argument("--curriculum-every", type=int, default=0,
                    help="every N iterations, pause run collection and PPO the "
                         "COMBAT net on decks harvested from recent runs, against "
                         "one act's encounters. 0 disables. This is what broke the "
                         "act-2 plateau: the combat net cannot learn act 2 from run "
                         "data because it only reaches act-2 fights in a few percent "
                         "of runs, so it is taught them in isolation instead. Turns "
                         "harvesting on by itself -- training on any other deck "
                         "distribution cost act-1 skill (benchmark 0.765 -> 0.680) "
                         "where real harvested decks left it at 0.767")
    ap.add_argument("--curriculum-plan", default="0,1,2",
                    help="0-based acts trained in EVERY pass (not one per pass). All "
                         "three by default: blocked training on one act at a time is what "
                         "traded act-1 skill for act-2 skill the first time, and the acts "
                         "contribute EQUAL gradient magnitude regardless of how unbalanced "
                         "the harvest is (~8924/1739/262 decks) because each step is tagged "
                         "with its act and the surrogate is averaged per tag. Act 1 keeps "
                         "all of its decks; only the signal magnitude is equalised")
    ap.add_argument("--curriculum-iters", type=int, default=25,
                    help="PPO iterations per pass")
    ap.add_argument("--curriculum-fights", type=int, default=200,
                    help="fights per curriculum iteration")
    ap.add_argument("--curriculum-mix", default="0.5,0.25,0.25",
                    help="boss,elite,regular sampling weights within a pass")
    ap.add_argument("--curriculum-lr", type=float, default=5e-5,
                    help="combat lr DURING a pass; --combat-lr is far lower because "
                         "it rides on run data, which is act-1 dominated")
    ap.add_argument("--curriculum-kl", type=float, default=0.05,
                    help="KL penalty toward --combat-ref during a pass. Unbounded "
                         "drift is what cost act-1 skill in round 2")
    ap.add_argument("--curriculum-min-decks", type=int, default=20,
                    help="skip an act in a pass when the window holds fewer than this many "
                         "of its decks. Necessary because the acts are weighted to EQUAL "
                         "gradient magnitude: an act represented by 6 decks would otherwise "
                         "have its noise amplified to the same weight as act 1's hundreds")
    ap.add_argument("--curriculum-recent", type=int, default=6000,
                    help="use only the most recent N harvested decks, so a pass "
                         "tracks the decks the CURRENT policy builds")
    ap.add_argument("--curriculum-keep-best", action="store_true",
                    help="end each pass on its BEST EMA-smoothed weights rather than its "
                         "last. Measured over 15 passes: the peak sits 0.30-0.39 of the way "
                         "through (0.50 would mean flat-with-noise) and higher KL to the "
                         "reference tracks a worse win rate, so the final weights are "
                         "systematically not the pass's best")
    ap.add_argument("--curriculum-ema-beta", type=float, default=0.3,
                    help="smoothing for --curriculum-keep-best. Raw per-iteration win rates "
                         "come from 25-114 fights, so selecting their argmax would pick a "
                         "lucky sample rather than a better policy")
    ap.add_argument("--curriculum-full-hp", action="store_true",
                    help="start curriculum fights at full hp instead of the hp the "
                         "deck actually had. Off by default: the policy arrives in "
                         "act 2 at ~35%% hp and risk assessment depends on it")
    ap.add_argument("--hp-bonus", type=float, default=0.5)
    ap.add_argument("--win-bonus", type=float, default=3.0)
    ap.add_argument("--act-bonus", type=float, default=1.0)
    ap.add_argument("--floor-bonus", type=float, default=0.03)
    ap.add_argument("--rollout-temp", type=float, default=1.0,
                    help="softmax temperature for the RUN policy. >1 flattens the action "
                         "distribution, producing a wider spread of trajectories per "
                         "iteration. Applied at BOTH act time and update time so PPO's "
                         "importance ratio stays consistent -- tempering only the rollout "
                         "would make the behaviour policy differ from the one being scored")
    ap.add_argument("--rollout-temp-anneal", type=int, default=0,
                    help="iterations over which --rollout-temp decays to 1.0 (0 = never). "
                         "Annealing means the FINAL policy is the true one, and the extra "
                         "breadth only buys exploration early")
    ap.add_argument("--combat-shaping", type=float, default=0.0,
                    help="potential-based shaping on COMBAT steps inside a run "
                         "(phi = player_hp_frac - mean living-enemy_hp_frac). Same fix that "
                         "gave +0.081 win in the combat-only trainer, applied here on the "
                         "real run distribution")
    ap.add_argument("--progress-shaping", type=float, default=0.0,
                    help="potential-based run progress (phi = floor count), the principled "
                         "replacement for --floor-bonus/--act-bonus. Use WITH those set to 0")
    ap.add_argument("--freeze-combat", action="store_true",
                    help="train the run policy only; the combat policy cannot regress")
    ap.add_argument("--combat-kl", type=float, default=0.5,
                    help="KL penalty toward the frozen reference combat policy")
    ap.add_argument("--gate-every", type=int, default=10,
                    help="iterations between combat regression checks (0 disables)")
    ap.add_argument("--gate-episodes", type=int, default=150)
    ap.add_argument("--combat-lr-floor", type=float, default=2.5e-6,
                    help="a rollback halves the combat lr; without a floor a run of them "
                         "silently freezes the combat policy altogether")
    ap.add_argument("--gate-tolerance", type=float, default=0.02,
                    help="allowed drop below the startup baseline before rolling back")
    ap.add_argument("--gate-relics", default="LARGE_CAPSULE,PANDORAS_BOX")
    ap.add_argument("--gate-mix", default="0.5,0.25,0.25")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    use_stable_attention()
    modcheck()
    if args.curriculum_every:
        # A pass trains on harvested decks, so harvesting is not optional; default
        # the file next to the run checkpoint so a tag gets its own harvest.
        if not args.harvest_decks:
            _stem = args.out[:-3] if args.out.endswith(".pt") else args.out
            args.harvest_decks = _stem + "_harvest.jsonl"
            print(f"  curriculum: harvesting to {args.harvest_decks}", flush=True)
        # The gate defends an ACT-1-ONLY benchmark. A curriculum deliberately moves
        # the fight model off that distribution, so a rollback would undo exactly
        # the work being done. Measure, never revert.
        if not args.gate_readonly:
            args.gate_readonly = True
            print("  curriculum: forcing --gate-readonly (the gate's baseline is "
                  "act-1 only and would roll back act-2 learning)", flush=True)
        # --harvest-cap is a PER-ROOM cap sized for building a fixed benchmark once. Left at
        # its default it stops harvesting for good, and every later pass then finds no fresh
        # decks and silently skips. Measured: a 600-iteration run hit 4000/room at iteration
        # 204 and ran its last 396 iterations with no curriculum at all -- 15 of 24 passes
        # skipped, act-3 clears flat at 0.02 while act 1 kept creeping up on run data alone.
        if args.harvest_cap == 4000:
            args.harvest_cap = 10 ** 9
            print("  curriculum: lifting --harvest-cap (the 4000/room default would stop "
                  "harvesting partway and silently disable later passes)", flush=True)
    if args.from_scratch:
        # A KL penalty toward a RANDOMLY INITIALISED reference would actively pull the policy
        # toward noise, and a regression gate anchored to a random baseline protects nothing.
        if args.combat_kl:
            print(f"  --from-scratch: forcing --combat-kl 0 (was {args.combat_kl})", flush=True)
            args.combat_kl = 0.0
        if not args.gate_readonly:
            print("  --from-scratch: forcing --gate-readonly (the reference model would "
                  "otherwise roll the scratch policy back to random init)", flush=True)
            args.gate_readonly = True
        if args.combat_lr < 1e-4:
            print(f"  --from-scratch: raising --combat-lr {args.combat_lr} -> 1e-4 "
                  f"(the low default assumes an already-good combat policy)", flush=True)
            args.combat_lr = 1e-4
    device = torch.device(args.device)
    _load_vocab(args.vocab)

    # --- combat policy: the thing we must not break ---
    ck = torch.load(args.combat_ckpt, map_location=device)
    combat = CombatNet(ck["sizes"], d_model=ck["d_model"], layers=ck["layers"],
                       dropout=args.dropout).to(device)
    if args.from_scratch:
        print("  --from-scratch: combat net RANDOMLY INITIALISED "
              "(checkpoint used only for vocab sizes and architecture)", flush=True)
    else:
        load_compat(combat, ck["model"])
    # Take the vocabulary sizes from the CHECKPOINT, not from a vocab file on disk: the run
    # policy shares card and relic indices with the combat policy, and a mismatch would mean
    # the two networks disagree about what card 137 is.
    global _ENC
    _ENC = Encoder(ck["sizes"]["cards"])
    n_cards = ck["sizes"]["cards"]
    n_relics = ck["sizes"]["relics"]
    n_potions = ck["sizes"]["potions"]
    # Frozen reference: the trust region is anchored to the checkpoint we started from, not to
    # the previous iteration, so drift cannot accumulate one small step at a time.
    if args.combat_ref:
        rck = torch.load(args.combat_ref, map_location=device)
        combat_ref = CombatNet(rck["sizes"], d_model=rck["d_model"],
                               layers=rck["layers"]).to(device)
        load_compat(combat_ref, rck["model"])
        print(f"KL trust region anchored to {args.combat_ref}", flush=True)
    else:
        combat_ref = copy.deepcopy(combat)
    combat_ref = combat_ref.to(device).eval()
    # Under --from-scratch this is a copy of the random init. combat_kl is forced to 0 above so
    # it is never used as a KL anchor; it stays only so the gate can still MEASURE the
    # benchmark each interval. Its baseline is a random net's score, so the gate reports
    # progress and never rolls anything back -- which is what we want with nothing to protect.
    for p in combat_ref.parameters():
        p.requires_grad_(False)
    combat_backup = copy.deepcopy(combat.state_dict())

    runnet = RunNet(n_cards, n_relics, n_potions,
                    per_kind_heads=args.per_kind_heads, dropout=args.dropout).to(device)
    if args.run_ckpt and args.from_scratch:
        print("  --from-scratch: run net RANDOMLY INITIALISED", flush=True)
    elif args.run_ckpt:
        rck = torch.load(args.run_ckpt, map_location=device)
        sd = rck["model"] if "model" in rck else rck
        sd = migrate_state_scalars(sd)
        load_compat(runnet, sd)
        # A checkpoint trained with one shared head has no per-kind weights; seed each head
        # from the shared one so the resume keeps its learned behaviour instead of starting
        # nine random heads.
        if args.per_kind_heads and not any(k.startswith("score_k.") for k in sd):
            runnet.clone_shared_into_heads()
            print("  seeded per-kind heads from the shared head", flush=True)
    print(f"combat {sum(p.numel() for p in combat.parameters())/1e6:.2f}M + "
          f"run {sum(p.numel() for p in runnet.parameters())/1e6:.2f}M params on {device}",
          flush=True)

    run_opt = torch.optim.AdamW(runnet.parameters(), lr=args.run_lr)
    combat_opt = (None if args.freeze_combat
                  else torch.optim.AdamW(combat.parameters(), lr=args.combat_lr))

    baseline = None
    if args.gate_every and not args.freeze_combat:
        # Measure the baseline on the REFERENCE model, not on the model we are about to
        # train. On a fresh start these are the same network. On a RESUME they are not, and
        # taking the resumed model's own score would re-anchor the contract to whatever it has
        # already drifted to - so a policy could walk downhill indefinitely, one "no
        # regression since last restart" at a time. The contract is measured against the
        # original fight model, always.
        # Measure the baseline twice and keep the run with the FULLER sample. The gate seed
        # is fixed so the episode set is meant to be identical every time, but a stalled env
        # silently drops episodes and changes the subset -- which showed up as the same
        # reference weights scoring 0.953 one run and 0.961 the next. An inflated baseline
        # makes the gate progressively stricter and rolls back work that was never bad.
        if args.gate_bench:
            baseline, bse, bn, bdetail = eval_realistic_benchmark(
                combat_ref, device, args.gate_bench, min(args.envs, 4),
                args.gate_bench_fights, args.ascension)
            print(f"combat baseline (reference model) on the REALISTIC run-deck benchmark: "
                  f"{baseline:.4f} +-{bse:.4f} over {bn} fights [{bdetail}]", flush=True)
        else:
            baseline, bse, bn = eval_combat_benchmark(
                combat_ref, device, min(args.envs, 8), args.gate_episodes,
                args.characters, args.gate_relics, args.gate_mix, args.ascension)
        if not args.gate_bench and bn < 0.98 * args.gate_episodes:
            wr2, se2, n2 = eval_combat_benchmark(
                combat_ref, device, min(args.envs, 8), args.gate_episodes,
                args.characters, args.gate_relics, args.gate_mix, args.ascension)
            if n2 > bn:
                baseline, bse, bn = wr2, se2, n2
        if not args.gate_bench:
            print(f"combat baseline (reference model) on the original benchmark: "
                  f"win_rate={baseline:.3f} +-{bse:.3f} over {bn} episodes", flush=True)

    chars = args.characters.split(",")
    per_env = max(1, args.runs_per_iter // args.envs)

    char_ix = {c: i for i, c in enumerate(chars)}
    critic = Critic(n_cards, n_relics, n_chars=max(8, len(chars))).to(device)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    card_stats = CardStats(args.card_stats)
    harvest_rooms = (set(r.strip() for r in args.harvest_rooms.split(",") if r.strip())
                     if args.harvest_decks else set())
    harvest_total = 0
    # How many harvested records the curriculum has already trained on. Everything before
    # this came from an older policy and is not what the current one brings to a room.
    harvest_consumed = 0
    skipped_passes = 0
    harvest_seen = defaultdict(int)
    critic_seen = [0]          # measured candidates the critic has trained on, cumulative
    expert = None if args.no_expert else ExpertEvaluator(
        chars, envs_per_char=args.critic_envs, ascension=args.ascension, seed="EXP")

    def expert_combat_policy(obs, legal):
        """Greedy combat play for the counterfactual fights: we are measuring the DECK, and
        sampling would add variance unrelated to the thing being compared."""
        with torch.no_grad():
            lg, _, mask = combat_forward(combat, obs, legal, device)
            lg = lg.masked_fill(~mask, torch.finfo(lg.dtype).min / 4)
            return lg.argmax(-1).tolist()

    def critic_batch(cands, characters, ctxs):
        vocab_c = {}
        decks, relics_l, chs, ctx_l = [], [], [], []
        for c, ch, cx in zip(cands, characters, ctxs):
            decks.append([(_card_ix(x), u) for x, u in zip(c.cards, c.upgrades)])
            relics_l.append([_relic_ix(r) for r in c.relics])
            chs.append(ch)
            ctx_l.append(cx)
        return encode_decks(decks, relics_l, chs, n_cards, n_relics, char_ix, device, ctx=ctx_l)

    # Discount schedule, driven by measured progress rather than by iteration count.
    #
    # The rationale: early on a run ends because of a blunder a few floors back, so crediting
    # a decision with the whole rest of the run is mostly noise -- the signal is local. Once
    # the policy stops making those blunders, the remaining losses ARE caused by choices made
    # tens of floors earlier (the card not taken, the campfire spent on rest instead of an
    # upgrade), and the discount has to open up for that credit to reach them at all.
    #
    # Tied to mean floors because that is what "stopped dying to simple mistakes" looks like
    # in the data, and smoothed so gamma cannot lurch on one noisy iteration.
    quality_ema = 0.0
    ema_beta = 0.2

    for it in range(1, args.iters + 1):
        buf = Buffer()
        kind_stats = defaultdict(lambda: {"n": 0, "h": 0.0, "hn": 0.0, "opts": 0,
                                                  "acts": defaultdict(int)})
        last_max_hp = {}
        last_fight = {}      # env -> the most recent FightResult, i.e. what killed the run
        deaths = []          # (act, FightResult|None) per finished run
        # gamma is fixed for the whole iteration so every run in it is credited consistently.
        gamma = args.gamma_lo + (args.gamma_hi - args.gamma_lo) * min(1.0, quality_ema)
        if args.rollout_temp_anneal > 0:
            frac_t = min(1.0, (it - 1) / args.rollout_temp_anneal)
            temp = args.rollout_temp + (1.0 - args.rollout_temp) * frac_t
        else:
            temp = args.rollout_temp

        # Curriculum. The expert is accurate and expensive, so its fight budget decays to zero
        # while the critic learns to stand in for it; the follow probability decays more slowly
        # after that, handing the decision back to the run policy only once it has had time to
        # absorb what the critic knows.
        frac = min(1.0, (it - 1) / max(1, args.critic_anneal_iters))
        fights_now = 0 if args.no_expert else max(0, round(args.critic_fights * (1.0 - frac)))
        follow_now = args.critic_follow * max(0.0, 1.0 - (it - 1) / max(1, args.critic_follow_anneal))
        pickup_states = []           # (kind, obs, legal, character, ctx) awaiting expert labels
        critic_samples = []          # (Candidate, character, ctx) with measured hp
        expert_calls = {"card_reward": 0, "shop": 0, "overrides": 0}
        deck_at_end = {}             # env -> final deck card ids, for the card table
        last_player = {}             # env -> the run-level player obs at the last decision
        harvested = []               # entry decks for the realistic combat benchmark

        def policy(kinds, obs, legal, idxs):
            """One batched forward per decision kind present in the batch."""
            acts = [0] * len(kinds)
            groups = defaultdict(list)
            for i, k in enumerate(kinds):
                groups["combat" if k == "combat" else "run"].append(i)

            with torch.no_grad():
                for grp, members in groups.items():
                    o = [obs[i] for i in members]
                    l = [legal[i] for i in members]
                    if grp == "combat":
                        logits, value, mask = combat_forward(combat, o, l, device)
                    else:
                        logits, value, mask = run_forward(
                            runnet, [kinds[i] for i in members], o, l, device)
                        if temp != 1.0:
                            logits = logits / temp
                    idx, logp = sample(logits, mask)
                    # Per-kind entropy at ACT time. Collapse is a property of the acting
                    # policy, so measuring it here (rather than only in the training loss)
                    # is what makes it visible the iteration it happens.
                    lp_all = F.log_softmax(logits, dim=-1)
                    ent_row = -(lp_all.exp() * lp_all).masked_fill(~mask, 0.0).sum(-1)
                    n_legal = mask.sum(-1).clamp(min=1)
                    # Normalised entropy: H / ln(#legal). A raw H is not comparable across
                    # decision kinds -- a 2-option potion gate can never exceed ln2=0.69 while
                    # a 20-card selection can reach ~3.0 -- so raw H made card_select look
                    # healthy and the gate look collapsed purely from option count. The
                    # normalised form is 1.0 for uniform and 0.0 for a constant, and needs no
                    # action label, which matters because the "event" label cannot
                    # distinguish one event option from another.
                    ent_norm = ent_row / torch.log(n_legal.float().clamp(min=1.0001))
                    for j, i in enumerate(members):
                        if kinds[i] != "combat":
                            kstat = kind_stats[kinds[i]]
                            kstat["n"] += 1
                            kstat["h"] += float(ent_row[j])
                            kstat["hn"] += float(ent_norm[j])
                            kstat["opts"] += int(n_legal[j])
                            a_i = int(idx[j])
                            lab = action_label(kinds[i],
                                               legal[i][a_i] if 0 <= a_i < len(legal[i]) else None)
                            kstat["acts"][lab] += 1
                    for j, i in enumerate(members):
                        a = int(idx[j])
                        acts[i] = a
                        step = {"obs": obs[i], "legal": legal[i], "kind": kinds[i],
                                "a": a, "logp": float(logp[j]), "v": float(value[j])}
                        if grp == "combat":
                            # phi(s) = player HP fraction - mean living-enemy HP fraction,
                            # byte-identical to train_rl.potential().
                            _p = obs[i].get("player", {})
                            _php = (_p.get("hp", 0) or 0) / max(1, _p.get("max_hp", 1) or 1)
                            _al = [e for e in (obs[i].get("enemies") or []) if e.get("alive")]
                            _ehp = (sum(e.get("hp", 0) / max(1, e.get("max_hp", 1) or 1)
                                        for e in _al) / len(_al)) if _al else 0.0
                            step["cphi"] = _php - _ehp
                            step["e_last"] = _ehp
                            buf.add_fight(idxs[i], step)
                            p = obs[i].get("player", {})
                            last_max_hp[idxs[i]] = p.get("max_hp", 80)
                        else:
                            # A travel decision means the previous floor was survived, and a
                            # rise in act index means an act was cleared. Paying those out
                            # WHERE THEY HAPPEN is what lets the discount do its job; a single
                            # terminal number would be identical for every decision.
                            env = idxs[i]
                            pl = obs[i].get("player", obs[i])
                            step["phi"] = (pl.get("hp", 0) or 0) / max(1, pl.get("max_hp", 1) or 1)
                            step["deck"] = float(pl.get("deck_size", 0) or 0)
                            # Run progress, for the optional potential-based version of the
                            # floor/act bonuses.
                            step["prog"] = run_progress(obs[i])
                            dc = pl.get("deck_cards")
                            if dc:
                                deck_at_end[env] = [c["card"] for c in dc]
                            # The freshest run-level state before whatever fight comes next.
                            # Combat steps carry a COMBAT observation (hand, enemies), not the
                            # durable deck/relic state, so the last non-combat decision is the
                            # only place an entry deck can be read.
                            if harvest_rooms:
                                last_player[env] = pl
                            bonus = 0.0
                            if kinds[i] == "travel":
                                bonus += args.floor_bonus
                                act = int(obs[i].get("act", 0) or 0)
                                if act > buf.last_act.get(env, 0):
                                    bonus += args.act_bonus
                                buf.last_act[env] = act
                            buf.add_run(env, step, reward_prev=bonus)
            # ---- pickup decisions ------------------------------------------------------
            # Two separate jobs, deliberately split.
            #
            # RECORD is free: a reference to the observation, kept so the expert can label it
            # after the rollout. Scoring inline was the mistake -- it evaluated ~4 candidates
            # at a time across a couple of eval processes while every run env sat blocked, and
            # turned a 30s iteration into 390s. Labelling the same states a moment later, in
            # one big batch, uses the same fights at roughly ten times the throughput.
            #
            # OVERRIDE is the critic, not the expert: a single forward pass, so it costs
            # nothing at decision time. It only acts once it has been trained on enough real
            # measurements to be worth listening to.
            for i, k in enumerate(kinds):
                if k not in ("card_reward", "shop"):
                    continue
                p_ = obs[i].get("player", obs[i])
                ctx = {"act": obs[i].get("act", 0),
                       "act_floor": obs[i].get("act_floor", 0),
                       "hp_frac": (p_.get("hp", 0) or 0) / max(1, p_.get("max_hp", 1) or 1),
                       "ascension": args.ascension}
                character = chars[idxs[i] % len(chars)]
                if len(pickup_states) < args.critic_states_per_iter * 4:
                    pickup_states.append((k, obs[i], legal[i], character, ctx))
                expert_calls[k] += 1

                if critic_seen[0] >= args.critic_warmup and random.random() < follow_now:
                    build = (build_card_reward_candidates if k == "card_reward"
                             else build_shop_candidates)
                    cands = build(obs[i], legal[i])
                    if not cands:
                        continue
                    with torch.no_grad():
                        b = critic_batch(cands, [character] * len(cands), [ctx] * len(cands))
                        sc = critic.strength(b).tolist()
                    best = cands[int(max(range(len(sc)), key=lambda j: sc[j]))]
                    if 0 <= best.action_index < len(legal[i]):
                        acts[i] = best.action_index
                        expert_calls["overrides"] += 1

            return acts

        def on_fight(env_idx, fr):
            mx = max(1, fr.max_hp)
            # Commit the entry deck now that the room type is known. Harvesting at the travel
            # decision instead would need the destination room type parsed out of the map;
            # fight_end reports it directly and cannot disagree with what was actually played.
            # Cap PER ROOM. A global cap fills with monster fights (roughly ten per boss
            # fight in an act-1 run) long before enough boss entry decks accumulate, which
            # would starve the one room type the benchmark most needs.
            if (harvest_rooms and fr.room in harvest_rooms
                    and harvest_seen[fr.room] < args.harvest_cap):
                harvest_seen[fr.room] += 1
                rec = harvest_record(last_player.get(env_idx), fr)
                if rec is not None:
                    harvested.append(rec)
            last_fight[env_idx] = fr
            # phi at the terminal state: a win means enemies at 0; a loss means the player
            # at 0 with whatever the enemy had left when last observed.
            hp_frac = fr.hp_end / mx
            open_steps = buf.open_fight.get(env_idx) or []
            e_last = open_steps[-1].get("e_last", 0.0) if open_steps else 0.0
            phi_end = hp_frac if fr.won else -e_last
            buf.finish_fight(env_idx, float(fr.won) + args.hp_bonus * hp_frac,
                             phi_end=phi_end, shaping=args.combat_shaping)

        def on_terminal(env_idx, rr):
            # Floors and acts were already paid out as they were earned, so the terminal
            # reward is only the genuinely terminal part: did the run end in a win, and how
            # much health was left standing.
            terminal = (args.win_bonus * float(rr.won)
                        + args.hp_bonus * (rr.hp / max(1, rr.max_hp)))
            # Only LOST runs belong in the death histogram. Appending every terminal made
            # each win land in the "not-in-combat" bucket (its last fight was the boss it
            # beat), so that bucket tracked the win rate exactly (mean diff -0.5pt over 120
            # iters) and diluted every real share by a factor of 1/(1-win_rate).
            if not rr.won:
                deaths.append((rr.act, last_fight.get(env_idx)))
            # A run that dies mid-fight may emit no fight_end, so an uncleared entry would
            # attribute the next run's death to the previous run's fight.
            last_fight.pop(env_idx, None)
            cards = deck_at_end.get(env_idx)
            if cards:
                card_stats.record_run(cards, rr.floors, rr.won)
            buf.finish_run(env_idx, terminal, gamma,
                           phi_end=rr.hp / max(1, rr.max_hp), hp_shaping=args.hp_shaping,
                           deck_end=float(rr.deck), deck_shaping=args.deck_shaping,
                           deck_target=args.deck_target, gae_lambda=args.gae_lambda,
                           prog_shaping=args.progress_shaping,
                           prog_end=float(rr.floors))

        t0 = time.time()
        with VecRunEnv(n_envs=args.envs, characters=chars, runs_per_env=per_env,
                       seed=f"RUN{it}_", ascension=args.ascension,
                       act_cap=args.act_cap, run_offset=it * 1000,
                   expect_sizes={'cards': ck['sizes']['cards'],
                                 'relics': ck['sizes']['relics'],
                                 'potions': ck['sizes'].get('potions', 10**9),
                                 'powers': ck['sizes'].get('powers', 10**9),
                                 'monsters': ck['sizes'].get('monsters', 10**9)}) as venv:
            # ACT in eval() mode. With dropout on, a train()-mode rollout records logp
            # under one random mask while the update recomputes it under another, so PPO's
            # ratio pi_new/pi_old stops being a policy ratio and becomes noise. The nets are
            # put back into train() for the update below.
            combat.eval(); runnet.eval()
            venv.run(policy, on_fight=on_fight, on_terminal=on_terminal,
                     max_steps=args.max_steps_per_iter, deadline_s=args.collect_timeout)
            combat.train(); runnet.train()
            results = list(venv.results)
            fights = list(venv.fights)
            kinds_seen = dict(venv.kind_counts)
            mean_batch = venv.mean_batch
        buf.drop_open()
        collect = time.time() - t0

        # ---- expert labelling, in ONE batch per character -----------------------------
        t_exp = time.time()
        if expert is not None and fights_now > 0 and pickup_states:
            random.shuffle(pickup_states)
            items = []
            for kind, o, l, character, ctx in pickup_states[:args.critic_states_per_iter]:
                build = (build_card_reward_candidates if kind == "card_reward"
                         else build_shop_candidates)
                cands = build(o, l)
                if cands:
                    items.append((character, ctx, cands))
            if items:
                expert.score(items, expert_combat_policy, fights_now)
                for character, ctx, cands in items:
                    for c in cands:
                        if c.strength() is not None:
                            critic_samples.append((c, character, ctx))
        expert_wall = time.time() - t_exp

        if not results:
            print(f"iter {it}: no runs completed; skipping")
            continue

        # Distil the expert into the critic: regress predicted hp-preserved onto what the
        # counterfactual fights actually measured, per room type. Only rooms that were played
        # contribute, so a partially-evaluated candidate still trains the heads it has data for.
        critic_loss = 0.0
        if critic_samples:
            cands = [c for c, _, _ in critic_samples]
            chs = [ch for _, ch, _ in critic_samples]
            ctxs = [cx for _, _, cx in critic_samples]
            tgt = torch.zeros(len(cands), len(ROOMS), device=device)
            msk = torch.zeros(len(cands), len(ROOMS), device=device)
            for j, c in enumerate(cands):
                for r_i, r in enumerate(ROOMS):
                    if c.n.get(r):
                        tgt[j, r_i] = c.hp[r] / c.n[r]
                        msk[j, r_i] = 1.0
            nb = 0
            for start in range(0, len(cands), args.minibatch):
                sl = slice(start, start + args.minibatch)
                b = critic_batch(cands[sl], chs[sl], ctxs[sl])
                pred, _ = critic(b)
                m = msk[sl]
                loss = (((pred - tgt[sl]) ** 2) * m).sum() / m.sum().clamp(min=1)
                critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()
                critic_loss += float(loss); nb += 1
            critic_loss /= max(1, nb)
            critic_seen[0] += len(cands)

        t1 = time.time()
        run_stats = ppo_update(runnet, run_opt, buf.run_done, device, args, is_combat=False,
                               temp=temp)
        combat_stats = None
        if combat_opt is not None and buf.fight_done:
            combat_stats = ppo_update(combat, combat_opt, buf.fight_done, device, args,
                                      is_combat=True, ref=combat_ref)
        train = time.time() - t1

        wins = sum(r.won for r in results)
        acts = np.mean([r.act for r in results])
        floors = np.mean([r.floors for r in results])
        # Anneal on measured depth. Smoothed, and never allowed to fall back, so a bad
        # sampling iteration cannot yank the discount back to myopic after the policy has
        # genuinely improved.
        q = min(1.0, floors / args.gamma_target_floors)
        quality_ema = max(quality_ema, (1 - ema_beta) * quality_ema + ema_beta * q)
        fw = sum(f.won for f in fights) / max(1, len(fights))
        vram = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
        # Per-act clear rates. `act` is the 0-based act index the run ENDED in, so a run that
        # cleared act 1 has act >= 1 whether it then died in act 2 or won; `won` means every
        # act up to the cap was cleared. With --act-cap 1 these collapse to `win`, which is
        # why the old `acts` column read 0.00 forever.
        n_res = len(results)
        # Per-act clear rates, for any --act-cap. `act` is the 0-based act index the run
        # ENDED in, so clearing act k means act >= k; the LAST act is cleared only by
        # winning. The conditional rates are what localise a plateau: an unconditional a3
        # cannot separate "never reaches act 3" from "reaches it and dies there".
        cap = args.act_cap if (args.act_cap and args.act_cap > 1) else 0
        per_act = ""
        if cap:
            rates = [wins / n_res if k == cap
                     else sum(1 for r in results if r.act >= k or r.won) / n_res
                     for k in range(1, cap + 1)]
            parts = [f"a{k + 1}={v:.3f}" for k, v in enumerate(rates)]
            parts += [f"a{k + 1}|a{k}={rates[k] / rates[k - 1]:.3f}"
                      for k in range(1, cap) if rates[k - 1] > 0]
            per_act = " ".join(parts) + " "
        line = (f"iter {it:3d}/{args.iters}  runs={len(results):3d} win={wins/len(results):.3f} "
                f"{per_act}acts={acts:.2f} floors={floors:.1f} fight_win={fw:.3f} "
                f"| run pg={run_stats['pg']:+.4f} vf={run_stats['vf']:.3f} ent={run_stats['ent']:.3f}")
        if combat_stats:
            line += f" | cmb pg={combat_stats['pg']:+.4f} kl={combat_stats['kl']:.4f}"
        line += (f" | g={gamma:.3f} steps={len(buf.run_done)}r/{len(buf.fight_done)}c "
                 f"vram={vram:.2f}G collect {collect:.0f}s train {train:.0f}s")
        print(line, flush=True)

        if expert is not None:
            print(f"    expert | fights/room={fights_now} follow={follow_now:.2f} "
                  f"card={expert_calls['card_reward']} shop={expert_calls['shop']} "
                  f"overrides={expert_calls['overrides']} sims={expert.fights} "
                  f"critic_samples={len(critic_samples)} seen={critic_seen[0]} "
                  f"critic_mse={critic_loss:.4f} expert_wall={expert_wall:.0f}s", flush=True)
            expert.fights = 0

        # Flush every iteration: a harvest pass is usually short, and buffering to a
        # ten-iteration boundary means a run of nine iterations writes nothing at all.
        if harvest_rooms and harvested:
            with open(args.harvest_decks, "a") as fh:
                for rec in harvested:
                    fh.write(json.dumps(rec) + "\n")
            harvest_total += len(harvested)
            harvested.clear()
            print(f"    harvested {harvest_total} entry decks "
                  + " ".join(f"{k}={v}" for k, v in sorted(harvest_seen.items())), flush=True)

        # ---- automatic combat curriculum ------------------------------------
        # Placed after the harvest flush so this iteration's decks are visible, and
        # before the checkpoint save so the pass's weights are what gets written.
        if (args.curriculum_every and combat_opt is not None
                and it % args.curriculum_every == 0):
            acts = [int(x) for x in args.curriculum_plan.split(",") if x.strip() != ""]
            # Every pass trains EVERY act. --curriculum-recent is applied per act, so the
            # cap does not let act 1's volume crowd the rarer acts out of the window.
            # Consume ONLY what has been harvested since the last pass. The file keeps
            # everything for later analysis; the pass just starts reading past what it has
            # already used, so it always trains on the decks the CURRENT policy builds.
            # Windows stay purely fresh. A per-act minimum that reached back into older
            # harvest was tried and reverted: the act-3 decay it was meant to offset turned
            # out to be a reward bug (phi_end for a LOST fight must be -enemy_hp, not 0),
            # and once that was fixed the decay fell from -0.223 to -0.034 per 15 iterations.
            # No reason to reintroduce staleness for a problem that no longer exists.
            decks_by_act = {a: curr.load_decks(args.harvest_decks, act=a,
                                               recent=args.curriculum_recent,
                                               skip=harvest_consumed)
                            for a in acts}
            thin = {a + 1: len(d) for a, d in decks_by_act.items()
                    if 0 < len(d) < args.curriculum_min_decks}
            decks_by_act = {a: d for a, d in decks_by_act.items()
                            if len(d) >= args.curriculum_min_decks}
            if thin:
                print(f"    curriculum: skipping thin acts {thin} "
                      f"(< {args.curriculum_min_decks} decks this window)", flush=True)
            if not decks_by_act:
                # Loud, because this silently disabled 15 of 24 passes once. If it repeats,
                # harvesting has stopped rather than merely lagging.
                skipped_passes += 1
                print(f"    curriculum: NO fresh decks -- pass SKIPPED "
                      f"({skipped_passes} skipped so far; harvested {harvest_total} records "
                      f"total, consumed {harvest_consumed}). If this repeats, harvesting has "
                      f"stopped -- check --harvest-cap.", flush=True)
            else:
                c_args = copy.copy(args)
                c_args.combat_kl = args.curriculum_kl
                # Each step is tagged with its act; averaging per tag then across tags is
                # what gives the three acts equal gradient magnitude without discarding
                # act 1's much larger deck pool.
                c_args.per_kind_loss = True
                c_args.per_kind_entropy = True
                # Reuse the combat optimiser so momentum stays coherent across run
                # and curriculum updates; only the step size differs.
                saved_lrs = [g["lr"] for g in combat_opt.param_groups]
                for g in combat_opt.param_groups:
                    g["lr"] = args.curriculum_lr
                try:
                    cstats = curr.train_pass(
                        combat, combat_opt, decks_by_act, device=device,
                        combat_forward=combat_forward, sample=sample,
                        ppo_update=ppo_update, pargs=c_args, ref=combat_ref,
                        iters=args.curriculum_iters,
                        fights_per_iter=args.curriculum_fights,
                        mix=tuple(float(x) for x in args.curriculum_mix.split(",")),
                        envs=args.envs,
                        real_hp=not args.curriculum_full_hp,
                        hp_bonus=args.hp_bonus, shaping=args.combat_shaping or 0.5,
                        seed=1000 + it,
                        keep_best=args.curriculum_keep_best,
                        ema_beta=args.curriculum_ema_beta,
                        log=lambda m: print(m, flush=True))
                finally:
                    for g, lr in zip(combat_opt.param_groups, saved_lrs):
                        g["lr"] = lr
                print("    curriculum done: " + " ".join(
                    f"act{a + 1} {v['fights']}f win={v['win']:.3f}"
                    for a, v in sorted(cstats.items()))
                    + f" | consumed decks {harvest_consumed}..{harvest_total}", flush=True)
            # Advance the marker whether or not a pass ran: decks older than the last pass
            # are stale by definition, and holding them back would only let them accumulate.
            harvest_consumed = harvest_total

        if it % 10 == 0:
            card_stats.save()
            torch.save({"model": critic.state_dict(),
                        "sizes": {"cards": n_cards, "relics": n_relics}}, args.critic_out)
            worst = card_stats.worst(min_n=10, k=5)
            if worst:
                print("    worst cards by mean floor reached: " +
                      ", ".join(f"{c}={f:.1f}(n={n})" for c, f, n in worst), flush=True)

        # Per-kind collapse report. "top" is the share taken by the single most common
        # action for that decision: near 100% means the policy has collapsed to a constant
        # there, which is invisible in the aggregate entropy and was previously only found by
        # probing a checkpoint by hand a hundred iterations too late.
        if kind_stats:
            parts = []
            for k in sorted(kind_stats, key=lambda kk: -kind_stats[kk]["n"]):
                st = kind_stats[k]
                if st["n"] == 0:
                    continue
                lab, cnt = max(st["acts"].items(), key=lambda kv: kv[1])
                extra = ""
                if k == "card_reward":
                    # The metric that actually matters for this decision, called out
                    # explicitly: "top=take" hid it because three of the four options are
                    # cards and only one is skip.
                    extra = f" SKIP={100 * st['acts'].get('SKIP', 0) / st['n']:.0f}%"
                parts.append(f"{k}:n={st['n']} Hn={st['hn']/st['n']:.2f} "
                             f"opt={st['opts']/st['n']:.1f} top={lab}:{100*cnt/st['n']:.0f}%{extra}")
            print("    policy | " + " | ".join(parts), flush=True)

        # Where runs die. This is the diagnostic that says whether the next problem is deck
        # strength (losing normals), risk taking (losing elites) or the act boss.
        if deaths:
            # Aggregate by ROOM TYPE and by act, not by individual encounter. The per-encounter
            # breakdown is too granular to aggregate across iterations (any top-N slice of it
            # is dominated by whichever encounters happened to come up), and the actionable
            # question is which CLASS of fight is ending runs: normals means the deck is too
            # weak, elites means the path policy is taking fights it should skip, bosses means
            # the deck is fine until it meets a real check.
            by_room = defaultdict(int)
            by_act = defaultdict(int)
            worst = defaultdict(int)
            for act, fr in deaths:
                by_act[f"act{act + 1}"] += 1
                if fr is None or fr.won:
                    by_room["not-in-combat"] += 1
                else:
                    by_room[fr.room] += 1
                    worst[f"{fr.room}:{fr.encounter}"] += 1
            n = len(deaths)
            rooms = ", ".join(f"{k}={v} ({100 * v / n:.0f}%)"
                              for k, v in sorted(by_room.items(), key=lambda kv: -kv[1]))
            actsx = ", ".join(f"{k}={v}" for k, v in sorted(by_act.items()))
            top3 = ", ".join(f"{k}={v}" for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:3])
            print(f"    deaths by room: {rooms} | by act: {actsx} | worst: {top3}", flush=True)

        # --- regression gate ---
        if (args.gate_every and combat_opt is not None and baseline is not None
                and it % args.gate_every == 0):
            detail = ""
            if args.gate_bench:
                wr, se, gn, detail = eval_realistic_benchmark(
                    combat, device, args.gate_bench, min(args.envs, 4),
                    args.gate_bench_fights, args.ascension)
                detail = f" [{detail}]"
            else:
                wr, se, gn = eval_combat_benchmark(
                    combat, device, min(args.envs, 8), args.gate_episodes,
                    args.characters, args.gate_relics, args.gate_mix, args.ascension)
            # Roll back only on a drop that is both larger than the tolerance AND outside two
            # standard errors, so noise cannot trigger it.
            margin = max(args.gate_tolerance, 2 * se)
            # A gate run on a materially different episode count is not comparable with the
            # baseline; report it and move on rather than rolling back on a different sample.
            if abs(gn - bn) > 0.1 * bn:
                print(f"  GATE: combat {wr:.3f} measured on {gn} episodes vs baseline's {bn}; "
                      f"not comparable, skipping this check", flush=True)
            elif wr < baseline - margin and args.gate_readonly:
                print(f"  GATE: combat {wr:.3f}+-{se:.3f} vs baseline {baseline:.3f} "
                      f"(margin {margin:.3f}){detail} -> below baseline, NOT rolling back "
                      f"(--gate-readonly)", flush=True)
            elif wr < baseline - margin:
                print(f"  GATE: combat {wr:.3f}+-{se:.3f} vs baseline {baseline:.3f} "
                      f"(margin {margin:.3f}){detail} -> ROLLING BACK", flush=True)
                combat.load_state_dict(combat_backup)
                # Halve the learning rate on a rollback: the same rate produced drift once and
                # would produce it again.
                for g in combat_opt.param_groups:
                    g["lr"] = max(args.combat_lr_floor, g["lr"] * 0.5)
            else:
                print(f"  GATE: combat {wr:.3f}+-{se:.3f} vs baseline {baseline:.3f} "
                      f"(margin {margin:.3f}){detail} -> keeping", flush=True)
                # The baseline stays where it started: the contract is "never worse than the
                # fight model we already had". Ratcheting it up on a lucky sample would make
                # the gate progressively harder to pass and eventually fire on noise alone.
                if not args.gate_readonly:
                    combat_backup = copy.deepcopy(combat.state_dict())

        torch.save({"model": runnet.state_dict(),
                    "sizes": {"cards": n_cards, "relics": n_relics, "potions": n_potions}},
                   args.out)
        if combat_opt is not None:
            torch.save({"model": combat.state_dict(), "sizes": ck["sizes"],
                        "d_model": ck["d_model"], "layers": ck["layers"]}, args.combat_out)

    if expert is not None:
        expert.close()
    card_stats.save()
    torch.save({"model": critic.state_dict(),
                "sizes": {"cards": n_cards, "relics": n_relics}}, args.critic_out)
    print(f"saved run policy -> {args.out}, critic -> {args.critic_out}, "
          f"card table -> {args.card_stats}")


_VOCAB = {}


def _load_vocab(path):
    """Parses the probe's vocab file into {kind: {name: index}}.

    It stores entries as the strings "[NAME, INDEX]" rather than as a mapping, so the critic
    -- which is handed card and relic NAMES by the game -- needs them inverted to reach the
    same embedding rows the combat policy uses.
    """
    global _VOCAB
    _VOCAB = {}
    try:
        raw = json.load(open(path))
    except Exception:
        return
    for kind in ("cards", "relics", "potions"):
        table = {}
        for entry in raw.get(kind) or []:
            if isinstance(entry, str) and entry.startswith("[") and "," in entry:
                name, _, idx = entry[1:-1].rpartition(",")
                try:
                    table[name.strip()] = int(idx)
                except ValueError:
                    pass
        _VOCAB[kind] = table


def _card_ix(name):
    return int((_VOCAB.get("cards") or {}).get(name, 0) or 0)


def _relic_ix(name):
    return int((_VOCAB.get("relics") or {}).get(name, 0) or 0)


def action_label(kind, act):
    """A short, stable name for a chosen action, for collapse reporting."""
    if not isinstance(act, dict):
        return "?"
    if kind == "travel":
        return str(act.get("point_type", "?"))
    if kind == "card_reward":
        return "SKIP" if act.get("kind") == "alt" else "take"
    if kind == "rest":
        return str(act.get("option", "?"))
    if kind in ("shop", "potion_gate", "potion_ooc", "card_select", "treasure"):
        return str(act.get("kind", "?"))
    if kind == "event":
        # Index, not key: event keys are unique per event, so keying by them would always
        # look like 100% "one label". The question worth asking is whether it always reaches
        # for the same POSITION on the page.
        return f"opt{act.get('index', '?')}"
    return str(act.get("kind", "?"))


def _by_kind(per_sample, kinds_b, device):
    """Mean within each decision kind, then mean across kinds (equal weight per kind)."""
    terms = []
    for kk in set(kinds_b):
        sel = torch.tensor([j for j, kb in enumerate(kinds_b) if kb == kk], device=device)
        terms.append(per_sample[sel].mean())
    return torch.stack(terms).mean()


def ppo_update(net, opt, steps, device, args, is_combat, ref=None, temp=1.0):
    """One PPO pass. Advantages are GAE(lambda) when the buffer computed them, otherwise
    Monte Carlo return minus the net's own value baseline."""
    if not steps:
        return {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0}

    rets = torch.tensor([s["ret"] for s in steps], dtype=torch.float32, device=device)
    old_logp = torch.tensor([s["logp"] for s in steps], dtype=torch.float32, device=device)
    old_v = torch.tensor([s["v"] for s in steps], dtype=torch.float32, device=device)
    # finish_run stores "adv" only when GAE is enabled. Combat steps never carry one, so they
    # keep the MC-minus-baseline form regardless.
    if all("adv" in s for s in steps):
        adv = torch.tensor([s["adv"] for s in steps], dtype=torch.float32, device=device)
    else:
        adv = rets - old_v
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    n = len(steps)
    tot = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0}
    nb = 0
    for _ in range(args.epochs):
        order = torch.randperm(n)
        for start in range(0, n, args.minibatch):
            sel = order[start:start + args.minibatch].tolist()
            batch = [steps[i] for i in sel]
            o = [s["obs"] for s in batch]
            l = [s["legal"] for s in batch]
            if is_combat:
                logits, value, mask = combat_forward(net, o, l, device)
            else:
                logits, value, mask = run_forward(net, [s["kind"] for s in batch], o, l, device)
                # Same temperature as at act time. Tempering only the rollout would leave
                # old_logp from pi_T and the new logp from pi_1, so the ratio would measure
                # the temperature rather than the policy update.
                if temp != 1.0:
                    logits = logits / temp
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4)
            logp_all = F.log_softmax(logits, dim=-1)
            a = torch.tensor([s["a"] for s in batch], device=device).clamp(0, logits.shape[1] - 1)
            logp = logp_all.gather(-1, a.unsqueeze(-1)).squeeze(-1)

            i = torch.tensor(sel, device=device)
            # Clamp the log-ratio before exponentiating: an unclamped ratio overflowed to inf
            # and put NaNs through every subsequent forward pass.
            ratio = (logp - old_logp[i]).clamp(-20, 20).exp()
            a1 = ratio * adv[i]
            a2 = ratio.clamp(1 - args.clip, 1 + args.clip) * adv[i]
            pg_per = -torch.min(a1, a2)

            # Grouping applies to COMBAT too. The curriculum tags each combat step with the
            # act it came from, and averaging per group then across groups is what makes the
            # three acts contribute equally to the gradient despite act 1 having ~34x the
            # harvested decks (8924 / 1739 / 262). Sampling act 1 down instead would throw
            # away real data; this keeps every deck and equalises only the signal MAGNITUDE.
            # With a single group -- ordinary combat training, where every step is tagged
            # "combat" -- the group mean is exactly the flat mean, so this is a no-op there.
            kinds_b = [s2["kind"] for s2 in batch]
            if not args.per_kind_loss:
                pg = pg_per.mean()
            else:
                # Average the surrogate PER DECISION KIND, then across kinds -- the same
                # correction already applied to entropy. A flat mean makes a decision's share
                # of the gradient proportional to how OFTEN it is taken, so travel (~30% of
                # run decisions) dominates and campfires (~7%) get almost no signal.
                #
                # This does bias the objective away from the true state distribution: PPO's
                # expectation is supposed to be over states as visited. The trade is
                # deliberate -- we are buying learning speed on rare-but-decisive decisions
                # with a biased gradient, which is why it is behind a flag.
                pg = _by_kind(pg_per, kinds_b, device)
            vf = F.mse_loss(value, rets[i])
            probs = logp_all.exp()
            ent_per = -(probs * logp_all).masked_fill(~mask, 0.0).sum(-1)

            if not args.per_kind_entropy:
                ent = ent_per.mean()
            else:
                # Entropy averaged PER DECISION KIND, then across kinds.
                #
                # A flat batch mean is dominated by whichever decision is most frequent: travel
                # is ~45% of all run decisions, campfires ~5% and card rewards ~20%. So the
                # exploration pressure landed almost entirely on the decision that was already
                # working, and the rare ones were free to collapse -- which is exactly what
                # happened (campfire went to SMITH ~100% regardless of hp, card reward to
                # take-card 100%). Weighting each kind equally gives the rare decisions the
                # same say as the common one.
                ent = _by_kind(ent_per, [s2["kind"] for s2 in batch], device)

            loss = pg + args.vf_coef * vf - args.ent_coef * ent

            # Entropy on the TAKE-vs-SKIP marginal of a card reward.
            #
            # The measured pathology is specific: over four options (three cards plus skip)
            # the policy's entropy is a healthy 0.67 normalised, because it spreads well over
            # WHICH card -- while choosing skip only 3% of the time. A bonus on the full
            # option set does nothing about that, since it is already satisfied. Collapsing
            # the distribution to the binary axis first, and rewarding entropy there, puts
            # the exploration pressure exactly where the policy is not exploring.
            if not is_combat and args.skip_ent_coef > 0:
                rows = [j for j, s2 in enumerate(batch) if s2["kind"] == "card_reward"]
                if rows:
                    ridx = torch.tensor(rows, device=device)
                    probs_r = logp_all[ridx].exp()
                    is_skip = torch.zeros_like(probs_r, dtype=torch.bool)
                    for jj, j in enumerate(rows):
                        for aa, act in enumerate(batch[j]["legal"]):
                            if aa < is_skip.shape[1] and act.get("kind") == "alt":
                                is_skip[jj, aa] = True
                    p_skip = (probs_r * is_skip).sum(-1).clamp(1e-6, 1 - 1e-6)
                    h_bin = -(p_skip * p_skip.log() + (1 - p_skip) * (1 - p_skip).log())
                    loss = loss - args.skip_ent_coef * h_bin.mean()

            kl = torch.zeros((), device=device)
            if is_combat and ref is not None and args.combat_kl > 0:
                with torch.no_grad():
                    rl, _, _ = combat_forward(ref, o, l, device)
                    rl = rl.masked_fill(~mask, torch.finfo(rl.dtype).min / 4)
                    ref_logp = F.log_softmax(rl, dim=-1)
                # Trust region toward the ORIGINAL policy: this is what stops slow drift from
                # accumulating across hundreds of iterations.
                kl = (ref_logp.exp() * (ref_logp - logp_all)).masked_fill(~mask, 0.0).sum(-1).mean()
                loss = loss + args.combat_kl * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            # Skip a step whose gradients are not finite rather than poisoning the weights.
            if all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()):
                opt.step()

            tot["pg"] += float(pg); tot["vf"] += float(vf)
            tot["ent"] += float(ent); tot["kl"] += float(kl)
            nb += 1
    return {k: v / max(1, nb) for k, v in tot.items()}


if __name__ == "__main__":
    main()
