# PREREG — V12 MIRROR-SUM: the pooled-denominator repair

Fathom Lab · 2026-08-31 · Frozen before implementation. Successor named by
`RESULT_fraction_coherence_ships_2026_08_31.md`: the pooled `115` — a sum computed inside a
prereg-mandated counts statement, stored in no receipt leaf — remains accused in the
OBLIGATE-1 RESULT after V11. This prereg defines the clause that repairs exactly that class
and nothing else.

## Grounding performed before this freeze (facts, not results)

Enumerating every dict-sibling same-field sum across the cited receipt sets:

- `115` has **exactly one** mirror-sum: `obligate1_result.json :: arms.*.valid`
  = positive 58 + negative 57. The true derivation.
- `58` has **zero** mirror-sums in the OBLIGATE-1 set — so the bare-count-pair and
  quoted-fragment `58` accusations cannot be touched by this clause even in principle.
- `9` (STRUCT-1's quoted *"9 new tests."*) has **one coincidental** mirror-sum:
  `seat_validity.*.score` = nine seats × 1. The prose counts tests; the sum counts seats.
  This discovery forces the uniform-sum refusal below, adopted BEFORE implementation.

## The clause (V12_SUM_COHERENCE)

An integer token N that the ladder would otherwise accuse binds via `derived-sum` iff there
exists, in the cited receipt set, a grandparent node G and field name f such that:

1. **Exhaustive same-field sum**: N equals the sum of `G.<child>.f` over ALL children of G
   that carry f, with at least 2 such children, every addend an integer leaf
   (dict-children only; list sums are out of scope for v1 and named as such);
2. **Non-uniform addends**: the addends are not all equal. A uniform sum k×v is
   indistinguishable from counting k things (fatal at v=1, degenerate at any v) — the `9`
   coincidence above is the class this kills. Distinct addends make the sum a nontrivial
   arithmetic fact about these specific leaves;
3. **Strictly rescue-only** — identical guard to V11 by construction: fires only where
   `bound` holds with zero post-filter hits, range-sanity silent, integer token,
   not spec/historical/notation, and no earlier derived clause has spoken.

receipt_ref: `derived-sum:<a>+<b>[+…]=<N>@<receipt>:<G>.*.<f>`. Epistemics branch:
`derived` (existing partition).

## Gates — all four must pass or the clause does not ship

- **G-S1 (fixtures)**: in the OBLIGATE-1 RESULT re-certification, every accused `115`
  (L28, L72, L73 at the pinned verifier) flips UNGROUNDED → VERIFIED via
  `derived-sum:58+57=115@obligate1_result.json:arms.*.valid`. Must NOT flip, and their
  refusal is part of the gate: all three accused `58`s (L37 bare pair, L73 quoted ×2) and
  STRUCT-1's quoted `9` (the uniform-sum coincidence). OBLIGATE-1 stays OATH-FAILED
  after the flip — this clause repairs a class, not a verdict.
- **G-S2 (mutant battery)**: (a) N±1 refused; (b) a receipt mutant adding a third sibling
  carrying f (making the exhaustive sum ≠ N) refused; (c) a subset of siblings summing to N
  while the full enumeration does not: refused (exhaustiveness); (d) uniform addends
  refused even when the sum matches; (e) addends under different fields (G.a.f1 + G.b.f2)
  refused; (f) float addends refused. Zero mutants may bind.
- **G-S3 (absolute corpus A/B)**: flag off vs on across every certified document in the
  repo. Ship condition: every moved token is UNGROUNDED → VERIFIED via `derived-sum` with
  a listed true derivation; zero movements of any other kind; zero HELD → FAILED.
- **G-S4 (invariants)**: full suite green including the epistemics meta-test; issuance
  invariants hold; re-issues tracked as new commits with drift recorded.

## What this clause must never do

Verify a computed number whose derivation the receipts do not actually contain; touch any
non-integer token; touch any token the ladder already grounds or abstains; fire on uniform
sums; fire on partial sums. The absolute gate exists because implementations lie more
fluently than preregs — it caught two such lies in the V11 cycle and is retained unchanged.
