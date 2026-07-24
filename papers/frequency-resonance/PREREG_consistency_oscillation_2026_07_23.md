# PREREG — does long-range consistency-checking require the oscillatory channel?

**Frozen:** 2026-07-23, before any evaluated (full) run. ABSTAIN / NULL / SUPPORT all first-class.
**Provenance:** three pipeline smokes preceded this freeze and drove the design honestly (documented
below). No full/evaluated run has occurred; the gates here are set before it.

## Question

The program's surviving "seeing thought through computation" signal is read≠write: honesty and
hallucination appear as a failure to keep the written output consistent with the read grounding — a
cross-context *consistency* phenomenon. The oscillation arc proved the oscillatory channel (complex/phase
eigenvalues) is the long-range integration mechanism in state-space models. **Hypothesis: comparing a fact
to a later claim across distance — the computational core of not-contradicting-your-grounding — requires
the oscillatory channel.**

## Task — delayed consistency comparison, with two controls

Length T=256, two input channels: (0) a premise ±1 at position `pos`; (1) a claim ±1 at the final
position. Premise and claim are independent balanced ±1; the label is inferable only by combining them.
Readout = final-position hidden state (causal).
- **cmp** — label = (claim == premise). Tested at gap **adj** (premise at T-2) and **long** (premise at 0).
- **claimonly** — label = sign(claim), NO premise present (a lone input). Control #1: shows a decay model
  can read an input at all.

## Arms

Same CLRU as run_pmnist_ablation.py, but with a bias-free input embedding (see smoke note). **FREE** = θ
learnable (oscillation on); **CLAMPED** = θ≡0 (pure decay). FREE and CLAMPED share every non-θ init
(RNG-matched) → one knob, matched-compute. Trained from scratch per condition; 2 seeds; 5000 steps.

## The mechanism this tests (understood, not assumed)

With θ=0, a premise and a later claim land in the SAME readout subspace as `mag^gap · premise + claim`.
At **gap 1** the four (premise,claim) cases separate by the magnitude of that sum, so a decay model CAN
compare adjacent facts (an FF reads |sum|). At **long gap** the premise is attenuated by `mag^255`; unless
`mag→1` it vanishes and the comparison is impossible. With θ≠0, phase rotation both preserves the distant
fact (a bounded, non-decaying rotation) and keeps it linearly independent of the claim, so the comparison
survives distance. The test: does decay, trained freely, escape this by learning `mag→1`, or not?

## Frozen gates

Means over seeds. `F_*`/`C_*` = FREE/CLAMPED accuracy. `gap_long = F_cmp_long − C_cmp_long`,
`gap_adj = F_cmp_adj − C_cmp_adj`, `DiD = gap_long − gap_adj`.

- **ABSTAIN** iff the controls do not fire: `C_claimonly < 0.90` (clamp can't even read an input) OR
  `F_cmp_long < 0.85` (oscillation can't do the long comparison → nothing to contrast) OR
  `C_cmp_adj < 0.80` (decay can't compare even adjacent facts → a long failure isn't distance-specific).
- **SUPPORT ("long-range consistency-checking requires oscillation")** iff all controls fire AND
  `gap_long >= 0.20` AND `C_cmp_long <= 0.65`. (Decay compares adjacent but fails at distance; oscillation
  does both → the deficit is DISTANCE.)
- **NULL** iff controls fire but `C_cmp_long >= 0.80` (decay learns to carry the distant fact, e.g. via
  `mag→1`, and compares at distance too → oscillation not required). Ship the negative.
- **PARTIAL** otherwise — reported verbatim.

## Confounds controlled (frozen)

- **Not a broken clamp:** claimonly (lone input) must pass — the clamp can read.
- **Not "decay can't compare at all":** cmp-adj must pass for the clamp — decay compares when close; a
  long-only failure therefore isolates DISTANCE, the difference-in-differences `DiD`.
- **Single knob, RNG-matched, matched-compute.**
- **Two measures agree:** the trained-from-scratch long deficit and the within-model post-hoc θ→0 reliance
  on the FREE long model must both point the same way for SUPPORT.

## Smoke-driven design fixes (honest journey; all before any evaluated run)

1. **Aggregation confound (majority task) — rejected.** A majority/aggregate-consistency task is a linear
   sum a decay integrator computes natively; it also could not be made decay-solvable *locally* under this
   architecture, so it confounded "long-range" with "decay can't aggregate." Replaced by single-fact
   comparison.
2. **Embedding-bias confound — fixed.** With a biased input embedding, every filler position injects a
   constant that, under θ=0, accumulates geometrically (~1/(1−mag)) and swamps the sparse signal after
   LayerNorm, so the clamp failed *every* task including a lone-input read — masking the real effect.
   Removing the embedding bias makes filler positions inject nothing; the clamp then passes the lone-input
   and adjacent-comparison controls, exposing the genuine long-range-specific deficit this prereg tests.
   (Oscillation is itself immune to this DC accumulation — the phase rotation makes the constant cancel
   rather than blow up — an incidental corroboration of the mechanism, not the gated claim.)

## Scope (non-negotiable in the writeup)

A controlled SSM result about a computational precondition (phase enables comparing temporally-separated
facts across distance). NOT a claim about real-LLM honesty — transformers have no θ to clamp. A positive
result establishes the precondition and a concrete mechanism; the bridge to LLM honesty stays open.

## Red-team asserts

1. `lin_scan == seq_scan` < 1e-4. 2. FREE/CLAMPED share `B_re`/`nu` (only θ differs). 3. cmp-adj premise
at T-2, cmp-long at 0; claimonly has no premise and label == sign(claim); labels 50/50. 4. controls
(claimonly, cmp-adj) genuinely decay-solvable — else the test is rigged.
