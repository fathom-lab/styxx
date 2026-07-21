# PREREG — hardening arc part 1: does the kill transfer, and does the ladder repair it?

date: 2026-07-20
status: FROZEN before any scored inference; committed with the corpus extension and driver.
operator authorization: this session ("let's finish this end to end").

## the two questions, one panel

Rung 1 measured ANCHOR NON-TRANSFER on one task: gold-style anchors (verbatim pairs, direct
negations) are structurally blind to the panel's organic failure mode, coverage 0/15, with the
per-dataset misfit flag up in 14/15. Part 1 of the hardening arc asks the two questions that
decide whether that is a curiosity or a claim:

1. **TRANSFER (the kill generalizes?):** the same protocol on two NEW task families —
   numeric consistency (count facts; contradictions are value shifts, consistents are recount
   paraphrases) and temporal ordering (before/after facts; contradictions are order swaps,
   consistents are reorderings of the same order) — with BLATANT anchors.
2. **REPAIR (the constructive half):** the original contradiction task with LADDER anchors —
   anchor strata drawn from the SAME generator and difficulty mix as the organic items
   (consistent items as negatives, contradiction items as positives, medium+hard). Still
   auditor-constructible without labels; the difference is that ladder anchors CAN exhibit the
   judges' organic failure modes. Prediction: the anchor-organic alpha gap closes and coverage
   returns.

## frozen design

Panel: Qwen2.5-3B-Instruct, four personas, both-true phrasing, greedy — IDENTICAL to rung 1
(amendment 1 configuration; the broken panel is the point). Corpus per arm: n_organic 240,
K 80 per anchor stratum, true pi 0.35, hard_frac 0.5, R = 15 replicates, deaf arm per
replicate. Audit: `styxx.anchors.audit_panel` (n_boot 300, null_sims 200, per-dataset tau).
Comparators: MV, DS, anchors-in-hand DS. Crash-safe per-replicate checkpoint keyed by arm.

Seed bases (frozen, disjoint from every prior run): repair-ladder-attr 4001-4015;
numeric-blatant 5001-5015; temporal-blatant 6001-6015.

## frozen gates and predictions

- **H1 (deaf reality, every arm):** deaf VOID on >= 14/15 per arm.
- **H2 (REPAIR, the load-bearing gate):** ladder arm — among ESTIMATED replicates, coverage
  >= 12/15 (VOID_UNDERPOWERED if fewer than 8/15 ESTIMATED), AND the mechanism must close:
  max over kept judges of |mean anchor-minus-organic alpha| <= 0.10 (rung 1 measured up to
  0.63 with blatant anchors). Both halves required; a coverage pass with an open alpha gap is
  a coincidence, not a repair.
- **H3 (TRANSFER, frozen prediction):** on each new family with blatant anchors, coverage
  <= 3/15 — the kill reproduces. A family landing 4-11/15 is a PARTIAL transfer, reported as
  measured; the generalization claim requires <= 3/15 on BOTH families and is not otherwise
  made.

Everything else is a characteristic: per-arm delta_alpha/delta_beta, misfit flag rates
(prediction, unbarred: high wherever coverage dies), s activation (expected 0), comparator
errors, kept-judge patterns.

## consequence rules (frozen)

- H2 passes AND H3 holds on both families ⇒ the paper-shaped claim is earnable: *gold-style
  anchors license nothing across task families on a real panel; same-generator ladder anchors
  restore label-free coverage; the instrument flags the difference itself.* Write-up proceeds
  to the operator with the paper-bar assessment.
- H2 FAILS ⇒ the repair story is dead at this design point and the honest headline is
  "non-transfer measured, repair open" — no constructive claim ships.
- Known scope limits carried verbatim: one base model family for the kill (the 8 GB card
  cannot hold 7B in fp16; quantized 7B unverified on this box and not attempted); frontier
  panel measured only in the easy regime (rung 2); all corpora synthetic-constructed. These
  bound the claim, not the gates.

Smoke writes only *_SMOKE_INVALID*. Missed bars are CLOSED_NEGATIVE verbatim. No bar moves
after the first scored token.

## pre-run interpretation note (2026-07-20, before any scored seed; gates unchanged)

The reduced-n smoke hints at a possible third outcome on the REPAIR arm: ladder anchors, by
exhibiting the judges' organic failure modes, may drive measured informativeness under the
noise-margin gate and produce VOID_PANEL__uninformative -- i.e., honest anchors may reveal the
panel as UNABLE to do the task at all. Under the frozen H2 semantics that outcome scores
VOID_UNDERPOWERED (not a pass). It will additionally be REPORTED as what it is: a refusal
resolution -- the instrument declining to certify a panel that blatant anchors had made look
competent. That reading changes no gate and rescues no bar; it is recorded now so it cannot be
invented after the numbers exist.
