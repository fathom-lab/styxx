# PREREG — Does the one-shot OVERWRITE GEOMETRY predict method-diverse REPAIRABILITY? (the keystone that unifies the white-box mechanism with the black-box repair)

**Pre-registered 2026-05-29, BEFORE any code for this test is written or run. One
confirmatory run. Feasibility-grade: single open model (Qwen2.5-1.5B-Instruct), SAE-free
full-vocab logit-lens, the SAME n=36 arithmetic items as the spectral/suppression runs.
Arithmetic ground truth computed in-code (`run_competence_cliff` SPECS) and SHA-256'd
before any scoring (expected to match the standing key `ddccd8e4…b87964d`). Correctness =
exact integer match (no judge). Greedy/deterministic.** Receipt: `repair_geometry_result.json`.

## Why this run (closing the arc into one mechanism)

The arc has two validated halves that have never been *joined on the same model*:

- **Black-box repair (the keystone, P1):** re-deriving a confident confabulation through 5
  independent reasoning methods recovers the correct answer on ~86% of items (gpt-4o-mini);
  ~2/36 form an "irreducible core" that no within-model method fixes.
- **White-box mechanism (spectral + suppression-rhythm):** confabulation is the **late,
  tight installation** of a confident wrong answer (overwrite completes at layers ≈23–27,
  median 25/28, IQR 4), over a mid-network field where no token — truth included — is
  privileged.

The unifying question: **is repairability written in the mechanism?** If the confabulations
re-derivation *can* fix have a *later, more fragile* one-shot install, and the irreducible
core is *baked in earlier / more entrenched*, then the white-box overwrite geometry
**predicts** the black-box repair boundary, and the two halves are one phenomenon. The prior
repair labels came from gpt-4o-mini (a different model from the white-box Qwen) — so this run
generates **Qwen's own** 5-method repair outcomes alongside **Qwen's own** overwrite geometry,
same model throughout, removing the cross-model confound.

## Apparatus (committed before data)

- **Model:** Qwen2.5-1.5B-Instruct, local, greedy decoding; reproduces the answer-key hash.
- **One-shot confab set:** items where the bare one-shot answer is wrong (parseable, not
  None) and alignable (the realized and correct answer-token sequences share a divergence
  point), exactly as in `run_suppression_rhythm`.
- **Repairability r ∈ {0..5}** (continuous, the better-powered design): for each confab item,
  run the **5 independent reasoning methods** (step-by-step; hundreds/tens/ones; long
  multiplication; estimate-then-exact; digit-by-digit-with-carry — the P1 method set) with the
  strict `ANSWER:`-line parser; `r` = how many of the 5 land on the exact correct value.
  Higher r = more repairable; r=0 = irreducible-on-Qwen.
- **One-shot overwrite geometry** (at the first divergence position, reusing the
  `suppression_profile` machinery — same `norm`, same `W`):
  - **flip_layer** = the last pre-final layer at which the correct token still outranks the
    realized token (higher = the wrong answer is installed *later* = more fragile).
  - **realized_dominance** = fraction of pre-final layers at which the realized (wrong) token
    already outranks the correct token (higher = the wrong answer is *entrenched* early).
  - **install_jump** = the largest single-layer increase of (realized − correct) lens-logit
    in the trajectory (the magnitude of the commitment hop).

## Predictions (decisive bars — pre-stated, scored exactly, both directions named)

| id | prediction | bar |
| --- | --- | --- |
| **U1 — later install ⇒ more repairable** | repairability correlates with a *later* flip layer | Spearman ρ(r, flip_layer) ≥ **+0.40** AND p < **0.05**. If ρ ≤ −0.40 → *reversed* (earlier install repairs better), reported against prediction. If \|ρ\| < 0.40 / p≥0.05 → geometry does NOT track repairability via timing. |
| **U2 — entrenched wrong ⇒ less repairable** | repairability correlates *negatively* with realized-token mid-network dominance | Spearman ρ(r, realized_dominance) ≤ **−0.40** AND p < **0.05**. Direction named both ways as in U1. |
| **U3 — geometry PREDICTS repairability (the unification)** | a single geometry feature separates repairable (r≥3) from stubborn (r≤1) confabs | AUC(best of {flip_layer, realized_dominance, install_jump}) ≥ **0.70** OR ≤ **0.30** in a powered split (≥6 vs ≥6). If in (0.30,0.70) → mechanism does NOT predict the repair boundary; the two halves stay separate (clean null, reported as such). |

**RESULT = SURVIVED iff U1 ∧ U3** (U2 corroborates the mechanism direction). Otherwise
REPORT_AS_LANDED with whatever held, scored against prediction.

## Precondition / honest failure modes (stated in advance)

1. **Power.** If fewer than **12** alignable confabs have computable geometry, U1/U2 are
   reported descriptively (no SURVIVED claim). For U3, if either the r≥3 or r≤1 stratum has
   < 6 items, U3 is under-powered → descriptive.
2. **U-shaped / degenerate r.** If r is near-constant (e.g. Qwen-1.5B repairs almost nothing,
   so r≈0 for nearly all items — plausible given its weak derivation), the correlation is
   undefined/under-powered → reported as "repair stratum too thin on this model; needs a
   stronger open model to populate the repairable arm." This is a real possibility and is a
   limitation of the model's competence, not of the hypothesis.
3. **Honest prior — this is a genuine gamble.** Every prior read of the internals indexed
   *generation mode*, not truth (depth AUC 0.498 within-mode; β control-token K failure;
   mid-network rank field Δ=−0.008). So geometry predicting *repairability* — a
   truth-adjacent property — is a real bet that could easily null. **A clean U1/U3 null would
   HARDEN, not weaken, the standing claim** that truth lives in the *process* (re-derivation),
   not in any single-pass internal read: it would say even the *repair boundary* is not
   legible in the one-shot mechanism.

## Honest scope (pre-committed)

Single open model (Qwen2.5-1.5B-Instruct); SAE-free full-vocab logit-lens; feasibility-grade
n=36; one confirmatory run; arithmetic ground truth computed in-code then hashed pre-scoring;
exact-integer correctness, no judge. Repairability is Qwen's own 5-method recovery count
(deterministic greedy per distinct method prompt), measured on the SAME model as the geometry
— this removes the cross-model confound in the prior P1 labels (gpt-4o-mini) but means the
repairable arm may be thin on a 1.5B model. This tests whether the one-shot overwrite geometry
is *informative about* the method-diverse repair boundary — a correlational, within-model
claim. It is NOT a causal demonstration that altering the geometry changes repairability (that
remains the disinhibition test). A null does not refute the repair result (P1 stands) or the
mechanism (suppression-rhythm stands); it would refute only that the two are legibly linked in
a single forward pass.
