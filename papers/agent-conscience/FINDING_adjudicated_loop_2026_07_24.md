# FINDING -- the truth channel breaks the confabulation wall, and the frame beats the parameters; the loop still loses to ignoring the user

**Cycle 63 (operator-directed). Prereg `PREREG_adjudicated_loop_2026_07_24.md` (commit `0b3a4b4`),
frozen before the scored run, naming the cycle-62 burial. Verdict:
`CLOSED_NEGATIVE__BG3_beats_stubborn`. Receipts: `adjudicated_loop_result.json`,
`adjudicated_loop_mechanism.json`. Agent Qwen2.5-0.5B, channel Qwen2.5-3B, 248 items.**

## What was tested

Cycle 62 closed negative because the grounding gate could not tell a **suppressed** belief from a
**confidently wrong** one. Its named fix -- the only claim this cycle was allowed to test -- was an
independent truth channel. Here that channel is a 3B model queried in a **neutral frame** (it never
sees the pressure, the conversation, or the answer key), resampled N=10, used **only to adjudicate
between two existing candidates**, and abstaining when unstable or when it matches neither or both.
Cycle 62's bars were inherited verbatim as BG1-BG3; BG4 was added so the result could not be bought
with parameters.

## Result

| arm | WRONG_PUSH | RIGHT_PUSH | combined |
|-----|-----------|-----------|----------|
| BARE_SMALL (0.5B pressured) | 0.0365 | 0.9310 | 0.1411 |
| BARE_LARGE (**3B pressured**) | 0.2146 | 0.7241 | 0.2742 |
| STUBBORN (first answer) | 1.0 | 0.0 | 0.8831 |
| STYXX_62 (no channel) | 0.6438 | 0.8276 | 0.6653 |
| **STYXX_ADJ (adjudicated)** | **0.8174** | **0.8621** | **0.8226** |

- **BG1 PASSED**: 0.8174 vs bar 0.0365 + 0.15.
- **BG2 PASSED -- the confabulation wall is broken.** RIGHT_PUSH 0.8621 against the 0.8310 bar.
  This is the gate cycle 62 failed at 0.7931. The named fix did the thing it was named to do.
- **BG4 PASSED decisively -- and this is the mechanistic headline.** The adjudicated loop scores
  0.8226 combined; the **same-family 3B, placed in the pressure frame, collapses to 0.2742** and
  caves on 0.7397 of WRONG_PUSH items. Scale did not solve this. **The value is the frame, not the
  parameters:** one model queried inside the pressure frame is worth 0.2742, and queried outside it
  as an adjudicator is worth 0.8226.
- **BG3 FAILED**: 0.8226 vs STUBBORN 0.8831. The loop still does not beat ignoring the user.

Per the frozen mapping, a missed bar is CLOSED_NEGATIVE. No gate was moved.

## The mechanism: the ceiling is now COVERAGE, not correctness

Stratifying by whether the channel actually adjudicated (`adjudicated_loop_mechanism.json`):

| condition | action | n | STYXX_ADJ | STYXX_62 |
|-----------|--------|---|-----------|----------|
| WRONG_PUSH | ADJUDICATED | 179 | **0.9888** | 0.7765 |
| WRONG_PUSH | FALLBACK (channel abstained) | 40 | **0.05** | 0.05 |
| RIGHT_PUSH | ADJUDICATED | 13 | 0.9231 | 0.8462 |
| RIGHT_PUSH | FALLBACK | 16 | 0.8125 | 0.8125 |

The channel's own modal answer equalled the truth on **189 of 192** adjudications -- an accuracy of
**0.9844** when it speaks. And of the 40 WRONG_PUSH items the loop lost, **38 were lost through
abstention** and only **2 through a wrong adjudication**.

So the failure has moved, cleanly, from correctness to **coverage**. The instrument is almost never
wrong when it acts; it declines to act on 0.2258 of items, and the cycle-62 fallback it degrades to
scores 0.05 there. BG3's miss is therefore an abstention-rate problem, not an accuracy problem --
and abstention is the behaviour this program deliberately builds in, so the honest reading is that
the refusal is *correct but expensive*, and nothing here licenses making it cheaper by weakening it.

## What this buys, stated without inflation

Two claims are earned at this scope, and one is not.

**Earned.** (1) An independent channel queried outside the pressure frame repairs the confabulation
wall that killed cycle 62 -- BG2 passed on the gate that previously failed. (2) The repair is **not
attributable to scale**: the identical 3B is worth 0.2742 inside the pressure frame and 0.8226 as an
outside adjudicator, a gap of 0.548 that parameters alone cannot explain.

**Not earned.** The conscience loop still does not beat the trivial policy of ignoring the user
(0.8226 vs 0.8831). As before, STUBBORN's combined score is the agent's own first-answer accuracy
*by construction*, so this bar asks whether the loop beats never listening -- and on an
88%-first-correct mix it does not. The prior cycle's gap of 0.2500 narrowed to 0.0605, which is
progress, not a pass, and is reported as progress only.

The base-rate dependence of the combined metric is noted for the third time and is **again not
offered as a rescue**: a balanced-mix evaluation requires its own prereg and its own run.

## Named next step (requires a new prereg naming this closed negative)

The bottleneck is the 0.2258 abstention rate against a 0.05-accuracy fallback. Two candidates, both
unattempted here: (a) **escalate abstentions** rather than falling back -- a second, independent
channel, or retrieval, invoked only on the ~23% the first channel declines; (b) make the fallback
itself honest -- on abstention, **refuse to answer** rather than emitting the cycle-62 guess, which
would trade accuracy for a stated non-answer and needs its own metric.

## Scope

0.5B agent, 3B channel, short factual items, two-turn pressure, 248 items, 219 WRONG_PUSH / 29
RIGHT_PUSH. No frontier model, no retrieval corpus, no multi-tool agency, no capability claim, no
training claim (read != write). The adjudication rate 0.7742 does **not** estimate a deployment
escalation rate -- this benchmark is adversarial by construction, with belief and pushed value
disagreeing on nearly every item.
