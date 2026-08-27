# RESULT — OATH v0.12: the clause could not reach the example its own preregistration quotes

Fathom Lab · 2026-08-26 · scored under `PREREG_oath_v12_formula_constant_2026_08_26.md`, frozen
before any change to `styxx/certify.py`. Receipts: `oath_v12_battery_result.json`,
`oath_mention_use_census.json`. Harness: `run_oath_v12_battery.py`, both arms, non-destructive.

**Verdict: `V12_UNDERREACH`. The clause does not ship. `V12_FORMULA_CONSTANT` is False in the
tree.**

---

## What happened

G1 passed: the frame reconstructed exactly at its frozen pin. G2 did not. Under the pre-committed
outcome table an under-fire is `V12_UNDERREACH` — revert and publish. No gate after G2 was scored.

| | |
|---|---|
| roster expected | 11 |
| firings observed | 6 |

Split of the firings observed: 3 on tokens the OFF arm accuses, 3 on tokens it verifies. The
frozen roster expected 3 accused and 8 verified, so the accused half is reached in full and the
verified half is missed almost entirely.

## Why, precisely

**The preregistration froze its bar against one population and specified a clause against a
different one.** The census marker that produced the roster is `latex_on_line` — a LINE-level
test, true of any line carrying a backslash command or a dollar sign. The clause's first conjunct
is a SPAN-level test: the token's column must fall inside a `$…$` span or an inline-code span
carrying a backslash command. Every token in the second population is in the first; most of the
first is not in the second. The gap between them is the entire failure.

That is a drafting defect and it is mine. The census measured what it measured honestly; the
prereg then quoted its number as though it described a clause that had not been written yet.

## The part that settles it

The prereg pre-committed a proof of repair: when the clause lands, both
`SYNTHESIS_mention_and_use_2026_08_26.md` and **the preregistration itself** must move
OATH-FAILED → OATH-HELD, and *"if either fails to flip, the clause did not reach its own
motivating case and the outcome is `V12_UNDERREACH` regardless of what the other gates say."*

| document | flag OFF | flag ON | flips |
|---|---|---|---|
| SYNTHESIS_mention_and_use | OATH-FAILED | OATH-HELD | yes |
| PREREG_oath_v12_formula_constant | OATH-FAILED | OATH-FAILED | **no** |

The preregistration remains accused, on line 23, on the digits of the formula it quotes as the
specimen the clause exists to silence. The reason is almost funny and is entirely structural: in
the SYNTHESIS that formula is written inside inline-code backticks, and in the preregistration it
is written as an indented code block. Same characters, same defect, different markdown — and the
clause reaches one and not the other.

**A clause that cannot reach the example its own preregistration quotes has not earned a corpus.**
Two independent legs — the firing count and the proof of repair — say the same thing, which is
the only reason this note is short.

## A second prereg defect, disclosed rather than discovered later

G7 asked a blind adjudicator for the 11 roster tokens **plus 10 non-roster tokens drawn from lines
carrying a LaTeX span**. The frame does not contain 10 such tokens. The population was
unsatisfiable the moment it was frozen, and nobody would have found that out without trying to
run it.

G7 was not run, for the ordinary reason as well as that one: a warrant panel adjudicates whether a
retraction is DESERVED, and it cannot rescue a clause that does not reach its class. Running it
here would have been theatre. Both facts are recorded in the battery result under
`gates_not_scored` by name, rather than left for a reader to assume the gates passed.

## What is kept, and what is not

`V12_FORMULA_CONSTANT` stays in the tree set to **False**, as `V05_APPROX_NOTATION` and
`V08_FLOAT_FIELD_BINDING` were after their kills, so the measurement re-runs and the negative is
not re-attempted from memory. The OFF arm is inert: no ledger row anywhere carries the reason
code.

What is **not** licensed: widening conjunct 1 to catch indented code blocks and re-running. The
preregistration says the clause is atomic, forbids post-freeze narrowing or widening, and states
"no second attempt inside this cycle". The temptation is real — the fix looks like four
characters — and taking it would convert a frozen bar into a bar that moves when it is
inconvenient, which is the one thing this programme cannot afford.

## What the defect still is

Unchanged, and still open. `extract_numbers` takes numerals out of rendered mathematics, `delta`
is trigger vocabulary, and a mathematical constant inside a formula gets accused of being a claim
whose truth condition was never met. It has no truth condition. Three certificates in this corpus
are OATH-FAILED on exactly that, and one of them is this cycle's own preregistration.

A successor needs a census taken at SPAN level, so the roster it freezes is the population the
clause will actually see. That is a cheap measurement and it is the whole lesson here: **freeze
the bar against the thing you are going to build, not against the thing you happened to measure.**

---

*The cycle produced no clause and one usable sentence, which is more than a shipped clause that
could not reach its own example would have produced.*
