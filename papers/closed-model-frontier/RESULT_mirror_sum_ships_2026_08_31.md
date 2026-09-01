# RESULT — V12 MIRROR-SUM ships: the pooled denominator comes home

Fathom Lab · 2026-08-31 · Prereg: `PREREG_mirror_sum_2026_08_31.md`, frozen before
implementation with its grounding recorded inside it. Receipts: `mirror_sum_ab.json`,
the re-issued OBLIGATE-1 certificate, `tests/test_sum_coherence.py`. The second
verdict-changing-class verifier repair since the epistemics freeze, by the same
four-gate discipline as V11.

## What was repaired

The V11 ship left a named successor: the pooled `115` — a denominator computed inside a
counts statement the OBLIGATE-1 prereg mandates verbatim, equal to
`arms.positive.valid + arms.negative.valid` and stored in no receipt leaf. The verifier
kept accusing a number whose derivation its own receipts fully contain.

The clause: an integer the ladder would otherwise accuse binds iff it equals the
exhaustive sum of one field across ALL dict-children of one receipt node, with at least
2 integer addends that are not all equal. Strictly rescue-only, same guard as V11.

## The grounding that changed the design — before a line of code

Enumerating every mirror-sum in the fixture receipt sets before the freeze found the
uniform-sum trap: STRUCT-1's quoted *"9 new tests."* coincides with nine seat-validity
scores of 1 each. The prose counts tests; the sum counts seats. The non-uniform-addends
refusal entered the prereg from that fact, and the corpus run below confirms the quoted
`9` never moves.

## The gates, formally

- **G-S1 (fixtures) — PASS.** All three accused pooled-denominator tokens in the
  OBLIGATE-1 RESULT flip UNGROUNDED → VERIFIED via
  `derived-sum:58+57=115@obligate1_result.json:arms.*.valid`. The three remaining
  accusations do not move — grounded in advance: that token has zero mirror-sums in its
  receipt set, so this clause cannot touch it even in principle.
- **G-S2 (mutant battery) — PASS.** Off-by-one, third-sibling, subset-sum,
  mixed-field, float-addend, and single-child mutants all refuse. Uniform sums refuse.
  Zero mutants bind.
- **G-S3 (absolute corpus A/B) — PASS.** Across every certified document: exactly
  **three tokens moved**, all UNGROUNDED → VERIFIED via `derived-sum`, all three the
  prereg-named specimens, **zero** wrong movements, **zero** verdict flips,
  **zero** HELD → FAILED. Every move is listed in the receipt individually.
- **G-S4 (invariants) — PASS.** Full suite green with the clause in place; the
  epistemics meta-test holds with the rescued tokens under the derived branch.

## The outcome, honestly bounded

The OBLIGATE-1 RESULT re-certifies **still OATH-FAILED** — now on three tokens instead
of six. The bare count pair *"32 of every 58"* and the quoted-fragment `58`s remain
accused, exactly at this clause's frozen boundary: they are leaf values whose bindings
the v0.3 count-binding filter strips, not computed sums, and repairing them is a
different clause with different risks. The residual classes (bare count pairs, quoted
fragments, list sums) are the named successors. Stored history untouched: the prior
certificate stands in git; this re-issue is a new commit with the drift tracked.

---

*The verifier accused a number for lacking a receipt while holding both halves of it.
Now it can add — under a guard that refuses to count. One cycle, one class, three
tokens home, three left honestly accused.*
