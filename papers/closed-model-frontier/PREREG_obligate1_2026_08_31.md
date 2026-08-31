# PREREG — OBLIGATE-1: close the trigger-recall gap with structure, or die measured

Fathom Lab · 2026-08-31 · Frozen BEFORE any obligation-predicate code is written or any
held-out token is adjudicated. Answers the repository's own open issue #39, *"OATH verifier:
close the trigger-recall gap (52% of decimals sit on unbound lines)"* — the largest known
hole in the flagship instrument.

## The gap, and why it is still open

`styxx.certify` obligates a number only when its line carries recognised trigger vocabulary —
a closed list of roughly thirty metric words (`_TRIGGERS`). Numbers on lines without those
words land in ABSTAIN: honestly reported, never checked. Measured share of full-precision
decimals sitting on unbound lines in our own corpus: **0.5227**
(`FINDING_b35bc_generality_invalids_2026_08_03.md`).

Every lexical repair is measured dead. `RECON_obligation_repair_is_not_lexical_2026_08_27.md`
killed the word-list family; `oath_structural_obligation_census.json` scored a
reporting-verb rule at **recall 0.0235, precision 0.40** — indistinguishable from obligating
everything. The same census found a structural candidate that separates:

| rule (RECON, in-sample, n=212 addressable) | recall | precision |
|---|---|---|
| null: obligate every number | 1.0 | **0.4009** |
| LEXICAL reporting-verb | 0.0235 | 0.40 |
| STRUCT ≥2dp ∧ outside code span | 0.3765 | **0.80** |

Those numbers are **in-sample**. The candidates were written before the data was consulted,
but they were scored on the same 225 adjudications that motivated them, and the RECON says in
its own status line that it *licenses no clause, no bar, no repair*. That licence is what this
prereg exists to earn or refuse. Yesterday's STRUCT-1 cycle is the reason to try: on the
sibling problem — detecting claims in agent prose — a conjunctive structural predicate beat
its verb-list null 0.4211 to 0.2061 on a fresh blind panel, after the whole lexical family had
died. Structure is the live hypothesis in this lane. This is its second, harder test.

## The candidate, OBLIGATE-1 — frozen here

A numeric token is **OBLIGATED** iff both:

1. **Precision shape** — the token is a decimal carrying **two or more fractional digits**.
   A quantity written to 2+ decimal places in prose is being *reported*, not counted.
2. **Outside a code span** — the token does not sit inside backticks, a fenced block, a file
   path, a version string, a date, an issue/PR reference, or a bracketed citation.

Both conjuncts come verbatim from the RECON's best-scoring candidate
(`STRUCT_precision_and_outside_code`). Nothing is added, tuned, or reweighted here.

**Composition with the existing predicate is UNION, never replacement.** OBLIGATE-1 fires
alongside `_TRIGGERS`, `n`-glued, range-correlation and precision sources; a token already
obligated stays obligated with its existing `obligation_source` under first-writer semantics.
The new source string is `structural-precision`.

## The cost that makes this different from STRUCT-1

Obligating a token that does not bind produces **UNGROUNDED — an accusation**. So the
predicate's precision is, directly, one minus the false-accusation rate on newly-obligated
tokens. This cycle therefore gates on precision first and treats recall as the secondary
prize, the opposite weighting from the claim-detector cycle. A rule that closes the gap by
manufacturing accusations has not closed anything.

## Ground truth — fresh, blind, and NOT the RECON's sample

- **Population**: numeric tokens the current verifier ABSTAINS on, drawn from the certified
  corpus, **excluding every token adjudicated in the RECON's 225** and every sentence used in
  the Stage 1/Stage 2 claim-detector packets.
- **Sample**: `random.Random(20260901)`, stratified to the two arms by the frozen predicate —
  all OBLIGATE-1-positive abstentions up to 60, plus an equal number of OBLIGATE-1-negative
  abstentions. If the positive arm holds fewer than 30, the cycle publishes *"measurement
  failed — insufficient valid adjudications"* at gate-failure prominence and ships nothing.
- **Question put to seats** (frozen wording): *is this numeric token a CHECKABLE CLAIM — a
  reported quantity a reader could in principle verify against the document's receipts — or
  is it not (an index, an ordinal, a row label, a count of prose objects, a version, a date, a
  citation, a configuration value, an illustrative figure)?* Labels: **CLAIM / NOT_A_CLAIM /
  UNSURE**. UNSURE is excluded from numerators and denominators and its count is reported.
- **Protocol**: 3 packets × 3 fresh seats, majority verdict, NO-MAJORITY excluded and counted,
  decoys carried and gated at ≥0.80 exactly as the claim-detector cycles did, token shown with
  its surrounding line for context and nothing else. Answer key sealed outside the repository;
  only its salted SHA-256 is committed before the seats run.

## Gates — every one can fail, and failure ships nothing

- **G-O1P (the bar, precision-first)**: held-out precision of OBLIGATE-1 ≥ **0.70**. The
  in-sample figure was 0.80; 0.70 is the pre-specified allowance for regression to the mean,
  fixed here so it cannot be chosen afterwards. **If G-O1P fails, the clause does not ship**
  and the RESULT carries verbatim: *"the structural obligation clause does not survive
  held-out adjudication."*
- **G-O1NULL**: held-out precision strictly exceeds the obligate-everything null's held-out
  precision, computed on the same adjudications. A rule that cannot beat "obligate
  everything" is not a predicate, it is a mood.
- **G-O1R (recall, secondary)**: OBLIGATE-1 catches ≥ **0.20** of held-out CLAIM abstentions.
  Below that it is real but too small to matter, and the RESULT must say so in those words.
- **G-O1REG (the ship gate — an A/B over the whole corpus)**: with the clause enabled, every
  currently-certified document is re-certified at the pinned verifier. **No document may move
  from OATH-HELD to OATH-FAILED without its new accusation being hand-adjudicated and
  published individually**, and the total count of new accusations across the corpus is
  reported whatever it is. If any new accusation is adjudicated FALSE, the clause does not
  ship in this cycle regardless of G-O1P.

## Disclosed limits, before the numbers exist

Seats see a token and its line, not the document — a harder task than the verifier's, in both
directions. All seats share one model family with each other and with the corpus's authors;
unanimity is a correlated-error ceiling. The corpus is this lab's own prose. n will be small
and no significance will be claimed at any n.

## What this prereg does not license

No change to the ladder, to `_TRIGGERS`, or to any existing obligation source. No widening of
either conjunct after the fact. No shipping on G-O1P alone — G-O1REG is the ship gate, and a
single false accusation blocks the release.
