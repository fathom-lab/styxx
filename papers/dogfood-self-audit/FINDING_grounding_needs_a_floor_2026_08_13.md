# FINDING — dogfooding styxx on its own author: "86.8% grounded" was true and nearly meaningless

Fathom Lab · 2026-08-13 · receipts: `grounding_judge_qualification.json`,
`grounding_chance_floor.json` · subject document: `PREREG_c6_derived_bar_2026_08_13.md` (mine,
written the same day) · fix shipped in `styxx/claim_audit.py`, pinned by
`tests/test_claim_audit_chance_floor.py`.

## What was run and why

Operator directive was to dogfood styxx and myself. I had just published a prereg with 76 numeric
claims. `styxx.claim_audit.audit_grounding` exists to check that every statistic in a document
traces to a receipt. So I pointed it at my own document, against my own receipts.

It returned **65/76 = 86.8% grounded**. Per the standing rule — *any favorable number I publish
about myself gets a judge qualified against planted controls FIRST* — that number went nowhere
until the judge was tested.

## First: the wrong tool passed loudly

`styxx audit-claims <my report>` returned **GATE: PASS** — after extracting **0 checkable claims
from 46 sentences** (coverage 0.00). A gate that understands nothing passes everything. It is not
lying (it prints the coverage), but `PASS` is the wrong word for "I could not read this," and it
is the same shape as the `meta_audit_v1` defect: a verdict that is favorable by construction.
Reformatting the numbered list to bullets changed nothing, so the cause is scope, not the list
mask. **Recorded, not fixed here** — the fix belongs with that command's owner, and naming it is
the deliverable.

## The real finding: `_match` has a chance floor nobody was reporting

`_match` grounds a claimed number if **any** value anywhere in the flattened receipt lies within
`0.5 * 10^-decimals`. It never checks that the matched path is semantically related to the
sentence. So the probability that an *arbitrary* number grounds is set by two things that have
nothing to do with the author's honesty: **claim precision** and **receipt cardinality**.

Measured against the real c6 receipt (163 distinct leaf values):

| claim precision | tolerance | P(random value grounds) |
|---|---|---|
| 1 decimal | ±0.05 | **1.0000** |
| 2 decimals | ±0.005 | 0.5719 |
| 3 decimals | ±0.0005 | 0.1009 |
| 4 decimals | ±0.00005 | 0.0102 |

**At one decimal the floor is 1.000.** Against a receipt this size, a one-decimal claim *cannot
be scored unsourced*. Its "grounded" verdict is unfalsifiable before a document is written.

My prereg, restated against the floor for its own precision mix:

| precision | grounded/total | observed | floor | excess |
|---|---|---|---|---|
| 2 dec | 22/27 | 0.815 | 0.572 | +0.243 |
| 3 dec | 35/36 | 0.972 | 0.101 | **+0.871** |
| 4 dec | 6/10 | 0.600 | 0.010 | +0.590 |

Overall observed 0.868 against a precision-weighted floor of 0.292 → **excess over chance +0.577**,
normalised 0.814. That excess is the part that is about me. The headline was not.

## My pre-stated prediction was wrong, and that is recorded

Frozen in `qualify_grounding_judge.py` before running: I predicted the judge was decoration —
that a fabricated document would score within 15 points of the real one (P1), and that grounding
rate would track receipt size on an unchanged document (P2).

- **P1 FAILED.** Real 0.868 vs fabricated 0.303 — **separation +0.566**. The judge discriminates.
- **P2 HELD.** Holding the document fixed and growing the receipt drove the rate 0.026 → 0.868.

Both legs are informative and they say different things. I expected to catch the tool being
worthless; instead the tool works *and* its headline number is uninterpretable without a floor.
**I was wrong in the direction that flattered my suspicion rather than my work** — the interesting
correction, since the prediction that failed was the cynical one.

## Two flattering bugs in my own fix, caught by cross-checking

The fix computes the floor inside the module. Its first two versions both reported a *lower*
floor than the standalone measurement, which would have made my own audit look better:

1. **v1** sampled `[min(leaves), max(leaves)]`. The receipt contains `seed = 90210`, so draws
   ranged over five figures, nothing landed near a real value, and the floor read **0.000** where
   the standalone said 0.572.
2. **v2** filtered to `|v| <= 1000` and took p95 — still **342**. Floor read 0.0035.

Both were caught only because an independent implementation existed to disagree with. **A
self-check that agrees with itself is not a check.** The shipped version samples the magnitude
class the audited statistics occupy and now reproduces the standalone independently (0.5798 vs
0.5719 at two decimals; 1.000 at one decimal).

## What shipped

- `GroundingReport` gains `chance_floor`, `floor_by_decimals`, `n_unfalsifiable`, and the
  properties `excess_over_chance` / `normalised_excess`.
- `summary()` prints the floor and the excess, and **warns explicitly** when claims are too
  coarse to fail (`floor >= 0.995`).
- `render_html()` shows the floor beside the headline percentage instead of the percentage alone.
- `tests/test_claim_audit_chance_floor.py` — 7 tests, including a **planted control** (real vs
  fabricated must separate on `excess_over_chance`) and a **regression test for the seed bug**,
  so a large scalar in a receipt can never silently deflate the floor again.

## Stated limits

- The floor is Monte-Carlo (4000 draws) and is a property of the receipt, not a constant; a
  sparse receipt yields a low floor legitimately. The test suite pins the *relationship*
  (denser receipt → higher floor; more decimals → lower floor), never an absolute number.
- **This does not make `audit_grounding` a semantic check.** It still grounds `0.40` to
  `frac_of_cohorts_at_or_above_0.80` from an unrelated cell — a value match, not a meaning match.
  The floor tells you how impressed to be by the rate; it does not make the match correct. Path-
  aware grounding is the real fix and is not done here.
- One number in my prereg is still flagged unfalsifiable by the shipped tool. It stays flagged.

*The tool I built to catch overclaiming was, on its headline metric, overclaiming. It took
pointing it at myself to see it.*
