# FINDING — the regularizer decides: a knowledge-preserving poison spares half the beliefs an unregularized one overwrites (a one-item pass, disclosed)

**Cycle 87. Prereg `PREREG_kp_recovery_2026_07_28.md` (commit `7082bbe`), harness
`run_kp_recovery.py`, both frozen before the scored run, all three outcomes pre-committed
first-class. Verdict per the frozen gates: `SURVIVED__knowledge_preserving_attack_spares_the_belief`.
Receipt: `kp_recovery_result.json`. Substrate `Qwen/Qwen2.5-1.5B-Instruct`, local, $0. Paired with
cycle 86 on identical items (`FINDING_poisoned_recovery_2026_07_28.md`).**

## The verdict, and immediately its fragility

The gates pass as preregistered. Selection took the smallest LAM meeting both validity conditions
(LAM 1.0: in-frame flip 1.0, held out-of-frame accuracy 1.0 — the attack takes AND the knowledge is
preserved); cells are powered (45 ATTACK_FLIPPED, 55 CONTROL); and RG1 clears both legs: recovery on
the flipped items 0.5111111111111111 against the 0.50 floor, specificity margin 0.25656565656565655
against the 0.15 floor, both floors imported unchanged from the cycle-75 module.

**The recovery leg passed by one item and this must travel with the claim.** The count of recovered
items is the smallest integer that clears the floor on a cell this size — one item fewer and the
gate fails. A normal-approximation interval on a proportion this size comfortably includes the floor, so the recovery leg is **not separated from its bar by the
data** — it is a pass under a rule frozen in advance, which is what the rule is for, and nothing
more. The bar was not moved in either direction. **This leg needs replication before anything
load-bearing rests on it.**

The specificity leg is the stronger one: 0.25656565656565655 against 0.15, and it is the leg that
discriminates belief-survival from a general accuracy lift.

## What is robust here: the paired reversal

Cycle 86 and cycle 87 ran the same protocol, on the same attack/held/control items, differing only
in whether the attack carried a knowledge-replay term. The contrast is not marginal:

| | cycle 86 (unregularized) | cycle 87 (knowledge-preserving) |
|---|---|---|
| recovery on flipped | 0.022222222222222223 | 0.5111111111111111 |
| specificity margin | -0.2323232323232323 | 0.25656565656565655 |
| planted answer propagates out of frame | 0.9777777777777777 | 0.4888888888888889 |
| held knowledge out of frame | 0.44 | 1.0 |

The sign of the specificity margin reverses. **The regularizer — not the weight edit as such —
decides whether the poison reaches the belief.** An attack that rewrites answers while wrecking the
surrounding knowledge overwrites the belief almost completely; the same attack constrained to leave
the held knowledge intact leaves roughly half the attacked beliefs recoverable out of frame.

Two controls make this hard to explain away. First, the replay set was HELD items under the neutral
prompt, so a trivial "the model learned to answer correctly in the neutral frame" story predicts a
blanket lift — but CONTROL items (pre-wrong, never trained) sit at 0.2545454545454545 out of frame,
statistically indistinguishable from the unregularized run, so no blanket lift occurred. Second, the
per-item records are **perfectly bimodal in both runs**: every flipped item resolves out of frame
either to the original correct answer or to the planted target, never to a third option. The poison
and the belief are competing for the same slot, and the regularizer sets the odds.

## What this does to the frame-locality boundary

Cycle 86 concluded the law hits a wall at the weights. Cycle 87 says that wall was **collateral
damage, not the weight edit itself**: constrain the attack to preserve knowledge and the out-of-frame
belief partially survives a parametric attack. The honest statement of the boundary is now a dose,
not a wall — *how much of the belief survives a weight-level attack is a function of how much
collateral knowledge damage the attack is permitted to do* — with the caveat that the survival half
of that statement rests on a one-item pass.

**Coupling reading (the arc's long-open question, engaged not settled):** rewriting the out-of-frame
belief appears **coupled** to damaging general knowledge. The attack could not have both — full
belief capture and preserved knowledge — at any rung of the frozen LAM ladder. That is a coupling
signal from the behavioral side, and it is the first the program has; the calibration-poisoning arc's
coupling question remains formally open, since this run never measured a capability battery.

## Scope and disclosures

One model (1.5B), one attack class (LoRA r=16, 300 steps), one benchmark family, one item set (the
cycle-86 strata), N=5 neutral samples, English. Selection was on validity gates only, never on RG1;
all four LAM rungs met both validity conditions and every rung's cells are in the receipt. HELD is
both the replay set and the `V_preserve` check, so that gate is partly trained in by construction —
disclosed in the prereg and restated here; the mechanism is measured on ATTACK items, whose correct
answers were never replayed. The recovery leg's one-item margin is the dominant limitation of this
finding.

## What this licenses

**Does license:** revising the cycle-86 statement from "a wall at the weights" to "a dose set by
collateral knowledge damage," provided the one-item margin is carried in the same breath; and the
paired reversal (recovery, specificity sign, propagation rate, on identical items) as the robust
core of the result.

**Does not license:** any confident claim on the recovery rate itself (one item from failing); any
claim beyond 1.5B, this attack class, or this item set; any settled answer to the coupling question
(no capability battery was run). The named next test is a straight replication of this exact
configuration on fresh items and a second seed — the cheapest way to move the recovery leg off its
knife edge, and it should run before this finding is cited anywhere.
