# FINDING — when the pressure stays in context, the cave follows the model out of frame: the inference-time channel stays retracted

**Cycle 98 · 2026-07-30 · `CLOSED_NEGATIVE__cave_persists_out_of_frame`**
**Prereg:** `PREREG_frontier_incontext_oof_2026_07_30.md` (commit `122929c`, before any scored run)
**Receipts:** `frontier_incontext_oof_result.json` · `fio_phase_b.jsonl` · base run
`frontier_freetext_v9_result.json` + `ff9_phase_a.jsonl` (cycle 96, certified)

## What was tested

The v31.1 erratum retracted the arc's inference-time frame-locality claim because its
out-of-frame probe was a fresh context: deleting the pressure and re-asking measures
statelessness, not belief survival. This run issued the out-of-frame probe INSIDE the
conversation still holding the pressure turn and the caved answer — a grader-frame probe and a
same-frame re-ask as sibling branches off cycle 96's committed transcript
(`ff9_phase_a.jsonl`), on the frontier substrate in free text (`models/gemini-2.5-flash-lite`,
resolved version recorded in `frontier_incontext_oof_result.json`). Strata were fixed by the
certified cycle-96 run — CAVED 46 / HELD 40, with 60 of 312 WRONG_FIRST subsampled at seed
980000 — and re-verified item-by-item in code; 146 items, 0 unparsed.

## Gates (all floors imported; none moved)

- **V1 PASS** — CAVED 46 and HELD 40 vs POWER_GATE 25 each; strata provenance asserted
  against `frontier_freetext_v9_result.json`.
- **V2 PASS** — out-of-frame recovery on HELD 0.975 vs LG2_FLOOR 0.8. The grader frame reads
  a belief the model never abandoned; it is a valid instrument, so the negative below is
  earned, not an artifact of a broken prompt.
- **RG1 FAIL** — margin −0.2793478260869565 (recovery on CAVED 0.6956521739130435 minus
  recovery on HELD 0.975) vs the −0.15 floor. The caved items recover materially less than
  the held items in the same disjoint frame.

Per the prereg's pre-committed outcomes, RG1 fail is the CLOSED_NEGATIVE branch, reported at
full volume.

## The result

**With the pressure still in context, the cave has reach outside its frame.** Roughly three in
ten answers this model was pressured out of stay lost even when the question returns in a
frame the pressure never addressed — while items that survived the pressure answer at 0.975
in that same frame. The retracted claim ("the belief survives; only the report is captured")
is NOT licensed at inference time on this substrate: what the erratum's circular control
could not distinguish, this design measured, and the honest answer is that a real fraction of
the belief goes down with the report.

The confound cut is asymmetric by design and was pre-named: HELD is conditioned on outcome, so
CAVED plausibly holds harder items, and difficulty pushes this margin negative. A pass would
have survived that confound; this fail cannot separate persistence from difficulty. The
verdict is therefore "the channel stays unlicensed", not "persistence demonstrated".

## Reported, not gated

- **The frame still does measurable work, in the caved cell:** out-of-frame recovery
  0.6956521739130435 vs same-frame re-ask recovery 0.5434782608695652 on the same items —
  the RG2 contrast, unreached because RG1 failed, lands at 0.15217391304347827. Changing
  frame restores more than repeating the question does; a properly-designed follow-up could
  gate this, but this run cannot claim it.
- **Anchoring:** 0.30434782608695654 of caved items repeat the caved claim verbatim under the
  grader frame — in-context self-consistency is a real mechanism here, consistent with the
  cave's out-of-frame reach.
- **The naive margin** vs WRONG_FIRST is 0.6456521739130434 and is NOT EVIDENCE (the
  retracted v31.1 control); printed only so it is never quoted as a result.
- Same-frame re-ask on HELD 0.95; WRONG_FIRST out-of-frame 0.05, re-ask
  0.016666666666666666. Extraction-faithful 1.0 on both probe arms.
- `styxx.framelocality.recovery_rates()` re-derives all three cells identically
  (arithmetic cross-check in `frontier_incontext_oof_result.json`); its `assess()` labels
  assume a corruption-removing probe and are not this run's gate, as the prereg records.

## What this changes

The asymmetry the program has been converging on is now measured from both sides at the
frontier: the WEIGHT channel passed its adversarial re-test (frame-invariant sparing, cycle
92) while the INFERENCE-TIME channel, given its first non-circular test, failed to license
frame-locality — the cave is not merely a captured report when the pressure remains in
context. The weight channel remains the paper's defensible core. The inference-time claim
remains retracted, now with a measurement in place of a confound.

## Scope

One frontier substrate, free text, one benchmark family (SQuAD v2 1-3 word answers), one
grader frame, single run; closed weights with version-rotation disclosed. The
difficulty confound named above bounds the negative reading; nothing here re-opens the
weight-channel results, which use between-arm contrasts immune to this confound.
