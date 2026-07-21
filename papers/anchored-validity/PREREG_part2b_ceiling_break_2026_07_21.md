# PREREG — part 2b: the ceiling-break arm

date: 2026-07-21
status: FROZEN before any verdict is collected; committed with the corpus extension.
operator authorization: this session ("push through the ceiling").

## the question

Every frontier-panel measurement in this arc has resolved at ceiling: constructed ladders up
to deconfounded multi-hop chains were solved perfectly, so the frontier tier has never been
measured in the regime where anchors matter — the regime where the panel actually errs. This
arm builds the hardest constructed-truth task the generator framework supports and asks, in
one arm's-length pass: does the frontier panel finally err, and if it does, what do BLATANT
versus LADDER anchors each make of the same verdicts?

## frozen design

- Family `chain_xl`: 12–16 people per item (extended name pool used by this family only, so
  every previously scored corpus reproduces byte-identically), pairwise relations with mixed
  phrasing direction, SENTENCE ORDER SHUFFLED, plus four DISTRACTOR people attached by one
  relation each (load without affecting decidability); organic queries at transitive distance
  three or greater, phrasing-flipped. Labels formally oracle-verified (transitive closure)
  before the sheet ships; a corpus failing the oracle does not run.
- One corpus, seed 9201, n_organic 200, pi 0.35, hard_frac 1.0 — carrying THREE anchor
  strata: blatant negatives (verbatim, 60), blatant positives (direct negation, 60), ladder
  negatives/positives (same-generator, 60 + 60, ids relabeled to avoid collision). One
  shuffled sheet, 440 items; a single judging pass yields BOTH audits over identical organic
  verdicts: audit_B (blatant anchors) and audit_L (ladder anchors), each
  `styxx.anchors.audit_panel`, n_boot 300, null_sims 200, seed 9201.
- Panel: four fresh-context Claude persona subagents, arm's-length per the standing protocol
  (task + sheet only; transcription trust boundary disclosed as before). Demonstration-grade
  single run; one draw is not a property. No deaf arm (subagent redaction is not meaningful
  here; disclosed).

## frozen outcomes (all pre-named; report verbatim whichever occurs)

1. **Ceiling stands** (all four judges 1.0): recorded honestly; the naturalistic-material
   residual remains and constructed escalation is declared exhausted.
2. **Ceiling breaks** (any judge below 1.0 on organics): the first frontier-tier measurement
   in the regime where anchors matter. Then, reported verbatim with no bar (single run):
   whether audit_B covers the true prevalence (blatant anchors seeing or missing the errors —
   the frontier-tier non-transfer question), whether audit_L covers or refuses (the ladder's
   two known directions), per-judge anchor-vs-organic false-fire and miss gaps, misfit flags,
   and whether the four personas remain byte-identical in the erring regime.
3. **Refusals**: any VOID from either audit is reported as issued.

Smoke/oracle outputs never mix with scored files. No bar moves after the first verdict.
