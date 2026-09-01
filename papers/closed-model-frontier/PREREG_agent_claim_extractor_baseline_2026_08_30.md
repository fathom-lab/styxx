# PREREG — the agent-report claim extractor, measured before it is repaired

Fathom Lab · 2026-08-30 · Frozen BEFORE the measurement harness exists. Red-teamed before
freezing by two adversarial passes (failability/denominators, blinding/leakage); every BLOCKING
finding is incorporated below and the attack transcript is preserved in the freeze commit's
provenance. The agent branch attestation (`RESULT_agent_gate_boundary_2026_08_30.md`) found
diffgate reading 6 sentences of an agent's ~2,738 and misreading half of what it read — but those
numbers came from the extractor's own extractions plus the author's unblinded hand adjudication
of 3 cases. This prereg commits the follow-up the RESULT owed: a blind ground truth, the
extractor and two null rules measured against it, and the never-read band's claim density
measured instead of asserted. **The panel can overturn the author**: the 3 hand-adjudicated
"false accusations" are in-sample, their panel verdicts publish individually, and a panel
verdict of A on any of them retracts the corresponding false-accusation claim with the same
prominence the original carried.

## Question

On agent-authored commit messages, what are the sentence-level precision and recall of
`styxx.diffgate`'s template extractor at finding diff-checkable claims, measured against blind
panel ground truth — and does the extractor beat two null rules that know nothing but a path
regex or a verb list?

## Corpus (pinned)

All commit messages on `origin/main..a6994ac` of this repository (57 commits, 2,824 non-empty
sentences at pin time) — the branch that built the instrument, authored end to end by an AI
agent. Unit of analysis: the sentence, split exactly as the gate splits
(`re.split(r"(?<=[.!?])\s+|\n+", text)`), non-empty after strip. **No exclusions**: trailers and
boilerplate stay in the corpus and are expected to adjudicate C; their share is reported, not
hidden.

## Ground truth construction

- **FLAGGED set**: every sentence on which the extractor fires (any template, after the
  referential guard), taken exhaustively. **Every flagged sentence is placed in a packet** —
  the headline precision estimand is adjudicated blind or not at all.
- **UNFLAGGED sample**: uniform random sample without replacement, `seed=20260830`
  (`random.Random(20260830)`), `n=294` sentences from the unflagged remainder.
- **DECOYS**: 30 known-answer sentences authored at freeze time by minimally editing real
  agent-report prose from OUTSIDE the pinned corpus into the same idiom — never textbook-clean
  constructions. Composition: 15 A (including imperative-header forms), 5 B, 10 C (6 of the
  C's probe mention-vs-use, 4 are structural headers/trailers/narrative). **Gating decoys** are
  the 24 uncontested ones (15 A + 5 B + 4 structural-C); the 6 mention-vs-use C's are
  REPORTED per seat, never gated — the author of their key is the contested adjudicator, and
  a validity gate must not enforce the very line under measurement.
- **Answer key sealing**: the key (and its salt) does not exist inside the repository until
  every seat output is recorded. The freeze commit carries `SHA-256(key JSON + salt)`; the
  plaintext key and salt are committed only after all seat outputs are committed, and the
  fold verifies the hash. Seats are subagents prompted with the packet inline and instructed
  to read no files and run no commands; prompt-level prohibition is not harness-enforceable
  and is disclosed as such — the mitigation is that the key is not on disk anywhere a
  disobedient seat could find it.
- **Packets & topology, pinned**: the corpus sentences (FLAGGED + the 294 sampled) are
  partitioned by seeded shuffle into 3 packets of near-equal size; **all 30 decoys appear in
  every packet**, shuffled among the corpus sentences under opaque sequential ids that
  distinguish nothing. Each packet is labeled by 3 independent seats; 9 seat-runs total.
  Verdict per sentence: majority of its packet's 3 seats. Packet composition (ids → text) is
  committed at freeze.
- **Seat instructions, both versions pre-committed**: instructions v1 and the single
  permissible clarified v2 are committed IN THE FREEZE COMMIT, before any seat runs. A G-V
  failure triggers exactly one re-run of that packet by FRESH seats using v2 verbatim; any
  other instruction change, or a second failure, invalidates the packet. First-run labels
  from a failing seat are discarded unexamined.
- **Distinguishability probe**: one additional non-voting seat receives packet 1 and answers
  only "corpus or synthetic?" per sentence. Frozen decision: if (share of decoys called
  synthetic) − (share of corpus sentences called synthetic) exceeds 0.5, the decoys are
  separable, G-V is void, no gate evaluates, and the estimands publish descriptively with
  that verdict stated first.

## Labels (one per sentence)

- **A** — asserts at least one specific change made *by this commit* to files or code:
  a file created/modified/deleted, a symbol added, a count of files or tests changed, a scope
  claim ("only touches X"). Checkable in principle against the commit's diff.
- **B** — asserts a result or measurement whose evidence lies outside any diff: test totals,
  CI verdicts, measured rates, panel numbers.
- **C** — neither: narrative, motivation, reports on the *state* of a file or on *other*
  commits' work, structural headers, trailers, boilerplate.

Frozen disambiguation rules (the red-team's label-collision findings, closed):

1. **Subject lines are labeled by content**, like any sentence. The C clause's "headers"
   covers only structural markers and trailers. An imperative or subjectless fragment naming
   a concrete file/symbol/scope change ("diffgate: promote the never-read band") asserts that
   this commit makes that change → A.
2. **Tense/agency default**: a bare past- or present-tense action verb with a file/symbol
   object and no other actor named asserts this commit performed it → A ("Rebuilt
   LEDGER.md"). Perfect/pluperfect and stative constructions ("had not been rebuilt",
   "holds", "carries", "is present") → C. A sentence naming another commit, branch, or prior
   cycle as the actor → C.
3. **Precedence**: a sentence asserting both a change and a result is A, and the seat
   additionally records `also_result_clause: true` on it. This sub-flag exists so compound
   "did X; N passed" sentences do not score the extractor wrongly in either direction.

## Estimands

- **E1** extractor precision on A: of sentences the extractor flags with a diff-checkable
  kind (every template except `tests_pass`), the share adjudicated A. **E1's denominator is
  reported verbatim wherever E1 is quoted.** If it is 0, E1 is undefined, G2 does not
  evaluate, and the RESULT states with gate-failure prominence: "G2 not evaluable — the
  extractor produced zero non-tests_pass flags on this corpus." If it is below 4, every
  quotation of G2's verdict carries the raw counts (E1 = x/y vs null = a/b).
- **E1b** `tests_pass` flags are scored correct iff the sentence adjudicates B **or**
  A-with-`also_result_clause`.
- **E2** extractor recall on A: flagged∧A over all adjudicated A. A `tests_pass`-only flag on
  an A-with-result-clause sentence counts as neither hit nor miss and is reported separately.
  Two forms: within-adjudicated-sample raw, and the corpus-level estimate extrapolating the
  unflagged sample's A-rate to the unflagged remainder — the extrapolated form is labeled an
  estimate, never a measurement. If adjudicated-A count is 0, both forms publish as UNDEFINED
  with denominators shown, and no recall number appears anywhere in the RESULT.
- **E3** the same two numbers for two null rules on the same adjudicated sentences:
  **N1** "flag iff the `_PATH` regex matches anywhere in the sentence"; **N2** "flag iff any
  `file_touched` verb stem matches anywhere in the sentence". **Null precision is computed
  with inverse-probability weights, frozen here**: weight 1 for flagged-set sentences, weight
  (unflagged remainder size / 294) for sampled sentences; both the weighted rate and raw
  counts publish. A null rule with weighted denominator 0 is undefined and G2 does not
  evaluate against it, reported with gate-failure prominence.
- **E4** the never-read band's claim density: the A-rate of the unflagged sample, with its
  denominator.

## Verdict edge cases, frozen

A sentence whose 3 seats split 1-1-1 adjudicates **NO-MAJORITY**: excluded from every
numerator and denominator, its count reported next to each estimand it was excluded from. If
NO-MAJORITY sentences exceed 10% of any estimand's denominator, that estimand and any gate
over it publish as UNSTABLE, with gate-failure prominence.

## Gates — each one can fail, and what failure forces is written here

- **G-V (validity)**: every seat scores ≥ 0.80 on the 24 gating decoys (≥ 20 correct). The
  re-run ladder is fixed above (fresh seats, pre-committed v2, once). **Floors**: gates
  evaluate only if ≥ 200 unflagged-sample sentences AND ≥ half the flagged set survive in
  valid packets. Below either floor — including total invalidation — no gate evaluates and
  the RESULT is, verbatim: "measurement failed — insufficient valid adjudications", with
  surviving n's, at the prominence a gate failure would have.
- **G1 (the boundary claim)**: the unflagged sample's A-rate (E4) is ≥ 0.02. **If G1 fails**,
  the never-read band is narrative after all, the boundary RESULT's "the entire substance sat
  in the never-read band" is an overclaim, and this cycle's RESULT must say exactly that and
  amend the claim — the negative publishes with the same prominence the positive would have.
- **G2 (the templates earn their keep)**: E1 exceeds the **better** of N1's and N2's weighted
  precision. Whether G2 passes or fails, the RESULT states verbatim: "E1 = x/y vs best null =
  a/b (weighted w); no significance is claimed at these n", and a pass may not be quoted
  anywhere without those counts. **If G2 fails**, the RESULT must carry, verbatim: "the
  templates add no precision over the best null rule at this sample size" — and any future
  extractor repair must treat the winning null as the bar, not the current templates.

Descriptive, NOT gates (they cannot fail meaningfully at these n): the mention-vs-use rate
among flagged path-kind sentences (the 3 known cases are in-sample by construction — their
individual panel verdicts publish, see the preamble); B-share of the corpus; unanimity rates;
per-seat performance on the 6 mention-vs-use decoys.

## Held-out split, binding the future repair

Commits are split by `random.Random(20260830)` into DEV (2/3) and HELD-OUT (1/3), recorded in
the receipt. THIS cycle reports pooled aggregates — it is measurement, not tuning, and touches
no extractor code. **Per-sentence HELD-OUT adjudications are sealed**: the receipt publishes
DEV labels in the clear and HELD-OUT labels only as a salted hash; the plaintext ships in the
repair cycle's RESULT after the repaired extractor's HELD-OUT run is committed. If HELD-OUT
holds fewer than 5 adjudicated-A sentences, the future before/after is descriptive, not a
gate, and must say so.

## Disclosed limits

Seats judge sentences without the surrounding message or diff — a harder task than the
extractor's, in both directions. All seats, the corpus author, and the decoy author share one
model family; no cross-family model is reachable in this harness, so the correlated-error
ceiling is disclosed but unmeasured (~0.96–0.98 unanimity observed on prior panels). The
corpus is one agent's prose in one repository's idiom; nothing here transfers externally.
n=3 seats per packet; no interval or significance is asserted anywhere.

## What this prereg does not license

No repair to `_TEMPLATES` or `_REFERENTIAL` in this cycle. No positioning claim beyond the
estimands' numbers with their denominators. The three xfail fixtures stay xfail.
