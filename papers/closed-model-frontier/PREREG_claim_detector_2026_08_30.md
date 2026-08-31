# PREREG — STRUCT-1: a structural claim detector for agent prose, frozen before code

Fathom Lab · 2026-08-30 · The repair cycle the baseline RESULT licensed, under its binding
rules: developed against DEV only, HELD-OUT sealed until the run is committed, and the bar is
N2's weighted precision 0.2061 — the verb-stem null — not the current templates. Frozen BEFORE
any detector code exists.

## Question

Can a sentence-level detector that implements the blind panel's frozen tense-and-agency rules
STRUCTURALLY — rather than as a word-list — find the diff-checkable claims in agent reports at
a precision the lexical nulls cannot reach? The lexical-repair RECON killed word-lists as
obligation predicates; this cycle tests whether structure over a verb list beats the verb list
alone. If it cannot, that negative extends the lexical-death finding to shallow structure and
publishes with equal prominence.

## The candidate, STRUCT-1 — specification frozen here

A sentence is flagged CLAIM iff ALL of:

1. **Action head**: a change-action verb in simple past, simple present, or imperative form.
   The verb class is a frozen list (change/modify/update/edit/rewrite/rebuild/rename/move/
   add/create/delete/remove/drop/extend/wire/fold/split/collapse/promote/demote/retarget/
   relabel/commit/ship/bump/fix/patch/land/introduce/gut/strip/restore/revert/touch), and it
   is deliberately the WEAK conjunct — the null control N2 measures the list alone, and the
   candidate earns nothing unless the remaining conjuncts add precision over it.
2. **Concrete object**: the clause carries a file path (diffgate's `_PATH`), a backtick code
   span, a symbol-shaped token, a tests/files count ("9 new tests", "3 files changed"), or a
   scope phrase ("only touches …").
3. **No stative/perfect block** — with the panel's own exception: perfect/pluperfect and
   stative constructions ("had been rebuilt", "is present", "holds", "carries") do NOT flag —
   EXCEPT negative-scope statives ("X is untouched", "unchanged", "not modified", "left
   alone"), which assert a diff-checkable property of this commit and which the blind panel
   adjudicated A (`p2-040`, "The rung ladder is untouched."). This exception is part of the
   frozen spec, learned from a DEV label, cited here before implementation.
4. **No other actor**: sentences attributing the act to another commit, branch, cycle, or
   person do not flag ("the prior cycle rebuilt…", "commit cbd2864 touched…").

Sentences flagged RESULT (test totals, CI verdicts, measured rates — panel label B) are
detected by a separate frozen rule (number + pass/skip/CI/AUC/rate context) and are NOT
counted as claims. Everything else is unflagged.

Implementation freedom covers regex/parsing detail only. The four conjuncts and the exception
are frozen; any change to them after this commit invalidates the cycle.

## Stage 1 — build on DEV, wire as observation, describe HELD-OUT

- **DEV evaluation**: precision/recall of STRUCT-1 against the 199 in-clear DEV labels
  (4 A / 33 B / 162 C), beside N1 (path regex) and N2 (verb stems) on the same sentences.
  DEV numbers are development telemetry, not results.
- **diffgate integration, observation only**: `uncovered_texts` splits into
  `unparsed_claims` (STRUCT-1 flags CLAIM, no template parsed it) and narrative. Verdict
  logic untouched. **G-AB (failable)**: the pinned attestation range `origin/main..04f7531`
  re-runs byte-identical on every verdict and count; any drift fails the gate.
- **HELD-OUT, descriptive by frozen rule**: the sealed split holds exactly 4 adjudicated-A
  sentences — under the baseline prereg's 5-A floor — so the before/after publishes as
  description, not a gate, exactly as that prereg ordered. Protocol: the frozen detector's
  HELD-OUT outputs are committed FIRST, then the salt reveals, hashes verify, scores publish.
- **Corpus census (descriptive)**: STRUCT-1 flag count and rate over all 2,824 sentences at
  the pinned corpus, with the never-read overlap stated.

## Stage 2 — the failable gate: a fresh blind panel over the detector's own flags

- **Sample**: 60 sentences STRUCT-1 flags CLAIM drawn from the corpus remainder that neither
  templates flagged nor any prior packet adjudicated; plus 60 STRUCT-1-unflagged control
  sentences from the same remainder; plus the SAME 30 frozen decoys; seeded
  `random.Random(20260831)`. Same packet standard, same v1/v2 instruction texts verbatim,
  same opaque-id shuffling, 3 packets × 3 fresh seats, majority verdicts, NO-MAJORITY
  excluded and counted, gating decoys ≥ 0.80 with the same re-run ladder, mention-vs-use
  decoys report-only.
- **G-S2P (the bar)**: A-share among adjudicated STRUCT-1 flags > **0.2061**, N2's weighted
  precision from `agent_claim_extractor_baseline.json`, frozen. If it fails, the RESULT
  carries verbatim: "the structural detector adds no precision over the verb-stem null at
  this sample size" — and the lexical-death finding extends to shallow structure.
- **G-S2LIFT**: A-share among flagged exceeds A-share among the unflagged control sample
  (discrimination, not just calibration). Both shares publish with denominators regardless.
- **Floors**: gates evaluate only if ≥ 45 flagged and ≥ 45 control sentences survive valid
  adjudication; below either, "measurement failed — insufficient valid adjudications"
  publishes at gate-failure prominence.

## What this prereg does not license

No diffgate verdict changes: accusation suppression and the three xfail fixture flips wait
for a later cycle gated on Stage 2 passing. No positioning claims beyond measured numbers
with denominators. DEV labels may tune the implementation of the frozen conjuncts; they may
not add, remove, or reweight conjuncts.
