# AMENDMENT — G-F1's fixture list contained an unsatisfiable expectation

Fathom Lab · 2026-08-31 · Amends `PREREG_fraction_coherence_2026_08_31.md`. Declared before
the formal gate run and before any ship decision; discovered when the gate harness first
executed against the corpus.

## The defect in the prereg

G-F1 listed *"26/58 and 39/115 in the OBLIGATE-1 RESULT, 44/59 in the OBLIGATE-2 RESULT"* as
specimens that must flip UNGROUNDED → VERIFIED. Two of those expectations were written on an
unverified assumption about the receipts:

1. **115 has no receipt leaf.** It is the pooled denominator `arms.positive.valid +
   arms.negative.valid` (58 + 57) — a quantity *computed inside the mandated counts
   statement*, stored nowhere. The clause refusing to verify it is not a failure of the
   clause; it is the anti-coincidence guard doing precisely its job on a number that the
   receipts genuinely do not contain.
2. **26 was never accused as a fixture requires.** At the pinned verifier it already binds
   (`arms.positive.claims`) through the ordinary ladder; only its denominator 58 was
   accused. A token cannot flip from a status it does not have.

The threshold structure is untouched. This amendment corrects only the fixture list, in the
same spirit and with the same disclosure discipline as
`AMENDMENT_claim_detector_stage2_2026_08_31.md`.

## G-F1, corrected

- **Must flip** UNGROUNDED → VERIFIED via `derived-fraction`, binding one receipt subtree:
  `58` (L27, OBLIGATE-1 RESULT) and `44`, `59` (L23, OBLIGATE-2 RESULT).
- **Must NOT flip, and their refusal is part of the gate**: `115` (both occurrences — a
  computed pooled sum absent from every receipt; verifying it would be the coincidence
  class the filter exists to kill), the bare count pair *"32 of every 58"*, and the quoted
  *"9 new tests."* fragment.
- Unchanged: G-F2's five-mutant battery, G-F3's absolute corpus condition, G-F4's suite
  invariants, and the ship rule that all four gates must pass.

## Also recorded here, because the gates caught them

The first implementation of the clause fired unconditionally rather than rescue-only; G-F3
caught it re-attributing healthy VERIFIED tokens and elevating an ABSTAIN through the
degenerate `0/38` zero-numerator case, and the strict guard (`bound`, zero post-filter hits,
range-sanity silent) was enforced in code before this amendment. A patch script also
silently failed to apply that guard once — masked by a reformatted target string — and the
A/B caught the discrepancy between comment and code. Both catches are part of this cycle's
record: the absolute gate exists because implementations lie more fluently than preregs.
