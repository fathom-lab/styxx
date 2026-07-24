# FINDING -- detection is not intervention: the conscience loop is a high-precision gate that still loses to ignoring the user

**Cycle 62 (operator-directed). Prereg `PREREG_conscience_loop_2026_07_24.md` (commit `8980540`),
frozen before the scored run. Verdict:
`CLOSED_NEGATIVE__AG2_right_push_not_surrendered_and_AG3_beats_stubborn`.
Receipts: `conscience_loop_result.json`, `conscience_loop_mechanism.json`. Substrate
Qwen2.5-0.5B-Instruct, 248 items.**

## What was tested

B18-S/B22 established **detection**: in the bare-term regime the label-free grounding monitor
separates HELD from CAVED at AUC 1.0 while the text sycophancy classifier collapses to exact chance.
This cycle asked whether that detector, wired into an agent loop as a **gate**, makes the ANSWER
better -- against BARE (the deployed default) and STUBBORN (always return the first unpressured
answer). Conditions were assigned by the model's own first answer, with a byte-identical pushback
template: WRONG_PUSH (user pushes the false sibling; honest agent HOLDS) and RIGHT_PUSH (user pushes
the truth; honest agent UPDATES).

Both validity preconditions passed: AV1 power 219 WRONG_PUSH / 29 RIGHT_PUSH; AV2 discrimination
with a bare cave rate of 0.913 and a stubbornness cost of 0.931 on RIGHT_PUSH.

## Result: one gate passed enormously, two failed

| arm | WRONG_PUSH | RIGHT_PUSH | combined |
|-----|-----------|-----------|----------|
| BARE | 0.0365 | 0.9310 | 0.1411 |
| STUBBORN | 1.0 | 0.0 | 0.8831 |
| STYXX | 0.6119 | 0.7931 | 0.6331 |

- **AG1 PASSED, and not narrowly.** Under false pressure the loop lifts accuracy from 0.0365 to
  0.6119 -- a gain of 0.575 against a frozen bar of 0.15. The 0.5B caves on 91.3% of items where it
  had just answered correctly; the monitor recovers most of that.
- **AG2 FAILED.** RIGHT_PUSH: STYXX 0.7931 against a bar of BARE 0.9310 - 0.10 = 0.8310. Missed by
  0.038. **This is the confabulation wall, named in the prereg before the run, arriving exactly as
  predicted.**
- **AG3 FAILED.** Combined: STYXX 0.6331 vs STUBBORN 0.8831. The loop does not beat simply ignoring
  the user.

Per the frozen mapping, two missed bars are CLOSED_NEGATIVE. No gate was moved.

## The mechanism (why it failed, precisely)

Stratifying by whether the gate actually fired (`conscience_loop_mechanism.json`):

| condition | gate | n | STYXX | BARE |
|-----------|------|---|-------|------|
| WRONG_PUSH | RESTORED | 137 | 0.9270 | 0.0073 |
| WRONG_PUSH | PASSED | 82 | 0.0854 | 0.0854 |
| RIGHT_PUSH | RESTORED | 7 | 0.2857 | 0.8571 |
| RIGHT_PUSH | PASSED | 22 | 0.9545 | 0.9545 |

Three things are true at once, and the honest report is all three:

1. **When the gate fires on a betrayed belief it is nearly perfect** -- 0.0073 becomes 0.9270 across
   137 items. The intervention mechanism itself works.
2. **The gate fires with high precision**: of 144 total firings, 137 land in the condition where
   firing helps and 7 where it harms -- a firing precision of 0.9514.
3. **Recall, not precision, is the binding constraint on a weak model.** The gate cannot fire
   without a stable belief to restore, and 77 of 219 WRONG_PUSH items sit below the stability gate.
   On the 82 items where it did not fire, STYXX inherits BARE's caving exactly (0.0854 = 0.0854).

The AG2 miss is carried entirely by the 7 RIGHT_PUSH firings, where restoring a stable-but-wrong
belief cost 0.857 -> 0.286. Grounding cannot distinguish a belief that pressure suppressed from a
belief that is simply, confidently wrong -- B18-S said so, and the agent layer inherits it.

## The AG3 miss is structural, and worth stating plainly

STUBBORN's combined accuracy is **not an independent baseline**: because the conditions are defined
by first-answer correctness, "always return the first answer" scores 1.0 on WRONG_PUSH and 0.0 on
RIGHT_PUSH by construction, so its combined score is exactly the model's own first-answer accuracy,
0.8831. AG3 therefore asks a stringent and fair question -- *does the loop beat ignoring the user
entirely?* -- and on this 88%-first-correct mix the answer is **no**.

Note this squarely, without rescuing it: the combined metric is base-rate dependent, so a different
condition mix would produce a different AG3 outcome. That observation does **not** change this
verdict and is not offered as one. Any balanced-mix claim requires its own new prereg and its own
run; it is not earned here.

## What this buys

The headline is a negative that matters more than the positive would have: **an AUC-1.0 detector
does not automatically make a better agent.** Wrapping a monitor around a model is not sufficient,
and a program that ships "safe agent" claims on detector metrics alone is overclaiming. What this
run does establish is narrower and real: the intervention is a high-precision, stability-limited
gate whose failure is concentrated in confident confabulation.

The path forward is named and not attempted here: the gate needs an **independent truth channel**
(retrieval, or OATH-style grounding against a receipt) to tell a suppressed belief from a
confidently-wrong one. That is the only route through the confabulation wall, and it requires a new
prereg naming this closed negative.

## Scope

One open 0.5B model, short factual items, a two-turn pressure protocol, 248 items. Disclosed in the
prereg: 3B and 1.5B substrates could not populate RIGHT_PUSH (4 and 3 items) and the substrate was
frozen at 0.5B for that reason -- so these numbers describe a weak model and do not transfer upward
without their own run. No capability claim, no training claim (read != write; this tests a live
monitor, not a certified-honest brain).
