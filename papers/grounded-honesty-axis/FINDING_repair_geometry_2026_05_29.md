# FINDING — The repair boundary is NOT written in the one-shot overwrite geometry: the keystone DID NOT unify the two halves (and the one signal that crossed runs *backwards*)

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_repair_geometry_2026_05_29.md` BEFORE any code for this test was written. Single open
model (Qwen2.5-1.5B-Instruct), SAE-free full-vocab logit-lens, the SAME n=36 arithmetic items
as the spectral/suppression runs. Arithmetic ground truth computed in-code, SHA-256'd
pre-scoring (`ddccd8e4…b87964d`, identical to every prior white-box/steering/spectral/
suppression run — same key). Exact-integer correctness, no judge. Repairability is Qwen's OWN
5-method recovery count (same model as the geometry — removes the cross-model confound in the
prior gpt-4o-mini P1 labels).** Receipt: `repair_geometry_result.json`.

## Why this run exists: to join the two validated halves ON THE SAME MODEL

The arc has two validated halves that had never been joined on one model:

- **Black-box repair (P1):** re-deriving a confident confabulation through 5 independent
  reasoning methods recovers truth on ~86% of items (on gpt-4o-mini).
- **White-box mechanism (suppression-rhythm):** confabulation is the **late, tight
  installation** of a confident wrong answer (overwrite completes at layers ≈23–27, IQR 4),
  over a mid-network field where no token — truth included — is privileged.

The keystone question: **is repairability written in the mechanism?** If the confabs
re-derivation can fix have a *later, more fragile* one-shot install, and the irreducible core
is *baked in earlier / more entrenched*, the two halves are one phenomenon. This run generated
**Qwen's own** 5-method repair count alongside **Qwen's own** overwrite geometry, same model
throughout.

## Result: the keystone DID NOT survive. n=36, 32 usable confabs, powered.

| id | prediction | bar | outcome |
| --- | --- | --- | --- |
| **U1 — later install ⇒ more repairable** | ρ(r, flip_layer) positive | Spearman ρ ≥ **+0.40** AND p < 0.05 | **FAILED.** ρ = **−0.288**, p = **0.110** (n=32, powered). Not significant — and the sign is *reversed* from prediction. Install timing does **not** track repairability. |
| **U2 — entrenched wrong ⇒ less repairable** | ρ(r, realized_dominance) negative | Spearman ρ ≤ **−0.40** AND p < 0.05 | **FAILED.** ρ = **+0.157**, p = **0.390**. Wrong sign, not significant. Mid-network entrenchment does not predict stubbornness. |
| **U3 — a geometry feature PREDICTS the repair split** | best of {flip_layer, realized_dominance, install_jump} separates repairable (r≥3, n=6) from stubborn (r≤1, n=22) | AUC ≥ **0.70** OR ≤ **0.30** (powered ≥6 vs ≥6) | **Threshold crossed — but in the REVERSED direction.** best feature = flip_layer, AUC = **0.212** (≤0.30). install_jump AUC = 0.720; realized_dominance AUC = 0.614. |

**RESULT = REPORT_AS_LANDED** (SURVIVED required **U1 ∧ U3**; U1 failed). U3's bar is crossed
only by a feature whose direction **contradicts the U1 hypothesis** — so it cannot be read as
support for the unification. See below.

## What this means — the keystone does not close, and the near-miss is instructive

**The pre-registered unification fails.** Neither timing (U1) nor entrenchment (U2) predicts
whether Qwen can re-derive its way out of a confabulation. The one signal that crossed a
threshold (flip_layer, AUC 0.212) says the *opposite* of the mechanistic story: repairable
confabs have a **lower** flip layer — the wrong answer was installed **earlier**, not later.
The most extreme cases make this vivid:

> `47 × 38 + 219` → **r=5** (every method repairs it), flip_layer = **0**, realized_dominance = **1.00**.
> `264 × 19 + 333` → **r=3**, flip_layer = **0**, realized_dominance = **0.96**.

These are *maximally entrenched* one-shot installs — the correct token never once led across
the whole network — yet they are among the **most repairable** items in the set. The one-shot
forward pass committed hard and early to the wrong answer, and that told us **nothing** about
whether step-by-step re-derivation could recover the truth; if anything it ran backwards.

The honest reading: **repair operates in a different regime than the one-shot install.**
Method-diverse re-derivation is a *multi-step reasoning* process; the overwrite geometry is a
*single-pass commitment* event. How hard or how late the single pass commits to the wrong
answer does not legibly predict what a fresh multi-step derivation will find. The two halves
are real, each stands on its own — they are **not linked in a single forward pass**.

## This HARDENS, not weakens, the standing claim — exactly as pre-registered

The prereg named this in advance: *"A clean U1/U3 null would HARDEN, not weaken, the standing
claim that truth lives in the process (re-derivation), not in any single-pass internal read."*
That is what landed. Four independent reads of the internals now agree there is **no
truth-adjacent signal legible in the one-shot mechanism at confab time**:

1. scalar depth grounding — AUC **0.498** (chance);
2. spectral β within-mode — **0.589**;
3. mid-network rank field — correct indistinguishable from any digit, Δ = **−0.008**;
4. **this run** — the one-shot overwrite geometry does **not** predict the method-diverse
   repair boundary (U1 ρ=−0.29 p=0.11; U2 ρ=+0.16 p=0.39; U3 crosses only in reverse).

Truth — and now even *repairability* — lives in the **process of re-derivation**, not in any
single-pass internal read. That is the consistent, hard-won shape of the whole arc.

## Honest scope and caveats (pre-committed + observed)

- **The repairable arm is thin — exactly the precondition we flagged.** Repairability
  histogram (r=0…5): **{15, 7, 4, 4, 1, 1}**, mean r = **1.125**. Qwen-1.5B repairs almost
  nothing; 22 of 32 confabs are r≤1, only 6 reach r≥3. The U3 high stratum sits at the
  *minimum* power (6 vs 6) the prereg required. A stronger open model is needed to populate
  the repairable arm; the reversed AUC should be read as *suggestive of a backwards
  relationship*, not as a powered effect.
- **This was a within-model correlational test, not causal.** A null here does not refute the
  repair result (P1 stands) or the mechanism (suppression-rhythm stands); it refutes only that
  the two are **legibly linked in a single forward pass**. It does not bear on the named
  disinhibition (late-layer intervention) test, which remains the causal question.
- Single open model; SAE-free full-vocab logit-lens; feasibility-grade n=36; one confirmatory
  run; arithmetic ground truth hashed pre-scoring (`ddccd8e4…`); exact-integer correctness, no
  judge; greedy/deterministic (reproduces the standing answer key).

## The arc, in one line (updated again)

The dial is construction↔retrieval (white-box); truth-recovery is the construction-ward shift
but it is causally inert to inject, invisible to read at the endpoint, not present as a
privileged mid-network signal — **and now: the one-shot install geometry does not predict
which confabulations re-derivation can repair** (if anything, the most entrenched installs are
among the most repairable). The repair boundary is written in the *process*, not the pass.
