# PREREG — OBLIGATE-2: the bar-blindness repair, frozen before code

Fathom Lab · 2026-08-31 · Frozen BEFORE any predicate code is written and before any fresh
token is adjudicated. Successor to `PREREG_obligate1_2026_08_31.md`, whose candidate died
held-out at 0.4483 against a 0.70 bar — with a failure class the RESULT named in one word:
**bars**. Thresholds, floors, gate criteria, interval bounds, and config values written to
two decimals, read by shape as results.

## Provenance, disclosed before anything else

The bar-adjacency conjunct below was written **after** reading OBLIGATE-1's 32 blind-adjudicated
false positives. That is what a dev set is for, and it is why those 115 adjudications are
spent: they are OBLIGATE-2's development data, never again its evidence. Every gate in this
cycle runs on **fresh tokens no prior panel has seen**. If the repair only works on the
failures that inspired it, the fresh panel will say so.

## The candidate, OBLIGATE-2 — specification frozen here

A numeric token is OBLIGATED iff **OBLIGATE-1 fires** (decimal with two or more fractional
digits, outside any code span — both conjuncts verbatim from the census rule) **AND none of
the following bar-markers holds**:

1. **Comparator adjacency** — a comparison operator (`≥ ≤ <= >= < > =`, or the words
   "at least", "at most", "above", "below", "under", "over", "against", "vs", "versus")
   within twelve characters of the token on either side.
2. **Bar vocabulary in the token's clause** — any of *bar, threshold, floor, ceiling, cap,
   cutoff, criterion, alpha, margin of, chance, frozen, pre-registered, prereg'd* within
   forty characters of the token.
3. **Interval position** — the token sits inside a bracketed interval `[a, b]` or a
   `{a, b, …}` set literal.
4. **Gate-table criterion cell** — the line is a pipe table row and the token sits in a cell
   whose header or cell-mates carry a comparator (the "criterion" column of a gate table,
   as opposed to its "observed" column).

Character windows (12, 40) are frozen numbers, not tunables. Implementation freedom covers
regex detail only; the four markers and the OBLIGATE-1 base are fixed. A token blocked by a
bar-marker stays ABSTAIN — the honest band — exactly as today.

## The known cost, stated before measurement

Bar-markers will also block some genuine results ("accuracy 0.9850 against a 0.9341 bar"
carries both a result and a bar within one window — marker 2 may block the result too).
That trades recall for precision, which is the correct direction for a predicate whose false
positives are accusations. The recall gate below is set accordingly low, and the trade is
reported whatever it costs.

## Protocol

- **DEV check (telemetry only)**: OBLIGATE-2 scored against the 115 spent adjudications from
  OBLIGATE-1. Reported in the receipt; may not be quoted as a result.
- **Feasibility precondition** (the unsatisfiable-population lesson, applied): the sample
  builder asserts ≥ 30 fresh OBLIGATE-2-positive abstentions exist after every exclusion
  BEFORE any packet is built; below 30 the cycle aborts pre-panel and publishes
  "measurement failed — population insufficient" at gate-failure prominence.
- **Fresh ground truth**: abstained numeric tokens from live re-certification of the corpus,
  excluding every previously adjudicated (doc, line, token) triple — the RECON's 225,
  OBLIGATE-1's 120, and both claim-detector packet sets. All OBLIGATE-2-positive abstentions
  up to 60 plus an equal number of negatives, seed `20260902`. The negative arm is drawn
  from OBLIGATE-1-positive-but-OBLIGATE-2-blocked tokens up to half its size (the bar band —
  the class under test) and from the general remainder for the rest.
- **Panel**: 3 packets × 3 fresh seats, the same frozen question wording, labels
  CLAIM / NOT_A_CLAIM / UNSURE, the same 16 token decoys (revealed after OBLIGATE-1; seats
  are fresh instances with no memory, and the decoys' truths remain unambiguous — reuse is
  disclosed here), gating ≥ 0.80 per seat, majority verdicts, NO-MAJORITY and UNSURE
  excluded and counted. Key sealed outside the repository, salted SHA-256 committed before
  any seat runs.

## Gates — frozen, each failable

- **G-O2P (the bar, unchanged)**: fresh-panel precision of OBLIGATE-2 ≥ **0.70**. On failure
  the RESULT carries verbatim: *"bar-blindness was not the whole disease — the structural
  obligation family is now two for two dead held-out."*
- **G-O2NULL**: beats the obligate-everything null's precision on the same adjudications,
  AND beats OBLIGATE-1's held-out precision of **0.4483** (the incumbent corpse — beating
  the null but not the corpse means the conjunct subtracted nothing).
- **G-O2R (recall floor, set low by the disclosed trade)**: population-weighted recall
  ≥ **0.10**. Below that the clause is a rounding error on the gap and the RESULT must say
  so in those words.
- **G-O2BAR (the class-specific kill-shot)**: among the bar-band negatives (OBLIGATE-1-fires,
  OBLIGATE-2-blocks), the panel's CLAIM share must be **< 0.30** — i.e. the tokens the new
  conjunct throws away really are mostly non-claims. If the bar band turns out to be rich in
  claims, the conjunct is amputating the wrong limb and the RESULT must say exactly that.
- **G-O2REG (the ship gate, inherited verbatim from OBLIGATE-1)**: corpus-wide A/B at the
  pinned verifier with the clause enabled; every document that moves HELD→FAILED gets its
  new accusation hand-adjudicated and published individually; **one false accusation blocks
  the ship** regardless of every other gate.

## Disclosed limits

Same seat family throughout; unanimity is a correlated ceiling. n small everywhere; no
significance claimed at any n. The 115 dev adjudications are spent and excluded. Decoy reuse
is disclosed above. The bar-marker vocabulary is lexical *inside* a structural conjunct —
the null comparisons exist precisely to price whether that mixture earns anything.

## What this prereg does not license

No change to the ladder or any existing obligation source. No marker additions after this
commit. No shipping on precision alone — G-O2REG holds the key, and a single false
accusation keeps the verifier abstaining honestly rather than accusing confidently.
