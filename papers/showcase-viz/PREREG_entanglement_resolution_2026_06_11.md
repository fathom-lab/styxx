# PRE-REGISTRATION — is the conscience-axis entanglement REAL, an ARTIFACT, or WHITENING-removable? (frozen)

**2026-06-11 · Fathom Lab / styxx. Frozen before any score is seen. Runner:
`run_entanglement_resolution.py` (SEED=0). Receipt: `entanglement_resolution_result.json`. This resolves
the ambiguity left by cycle 2 (`FINDING_axis_independence_2026_06_11.md`, PARTIAL-STRUCTURED, @4bc3fe9),
which measured truth↔refusal cross-talk by discriminability max(AUROC,1−AUROC) at n_test=15 — a metric
with an INFLATED chance floor at small n. Cycle 2 itself flagged this and owed the clean test. This is it
(backlog B28). It can revise cycle 2 toward EITHER "actually independent" OR "really entangled."**

## The three hypotheses for the cycle-2 cross-talk

1. **REAL geometry** — the truth and refusal directions genuinely share readable structure: each reads
   the other's content ABOVE both (a) the small-n permutation floor and (b) the random-direction floor.
2. **ARTIFACT** — the off-diagonal sat at the inflated discriminability floor / the "any direction
   separates a broadly-separable axis" floor; it is not specific to the direction. Cycle 2 over-read it,
   and the axes are effectively independent readouts.
3. **WHITENING-removable** — the cross-talk is real in the raw, anisotropic residual stream but is a
   covariance artifact: a ZCA-whitened (Mahalanobis) readout removes it and recovers a clean basis.

## Design — two nulls, a whitening test, a Gram-Schmidt test

Same three axes as cycle 2 (truth / refusal / valence-control), same common layer (gemma L12), unit DiM
directions fit on TRAIN, evaluated on held-out TEST. Larger sets this time to tighten estimates:
truth n=88, refusal n=88, valence n=48 (each axis's available maximum or the target, whichever smaller);
70/30 train/test. Off-diagonal discriminability = max(AUROC, 1−AUROC), as cycle 2. The two truth↔refusal
off-diagonals are **B** = w_truth on refusal-test, **C** = w_refusal on truth-test.

For each off-diagonal cell, in gemma AND through the cycle-2 shared label-free map into Llama-3.2-3B:
- **obs** = observed discriminability.
- **permutation null** (K=1000): shuffle that test axis's labels, recompute discriminability → the
  small-n floor. `permnull_p95`, `p_perm = (1+#{perm≥obs})/(1+K)`.
- **random-direction null** (1000 isotropic unit vectors): discriminability of `(test @ random_dir)`
  against the REAL labels → the "any direction separates this axis" floor. `randdir_p95`.
- **SPECIFIC-REAL** for a cell ≡ `obs > permnull_p95` AND `obs > randdir_p95` (real, AND specific to the
  direction rather than the axis just being broadly separable).

**Whitening test:** ZCA-whiten activations using the pooled TRAIN covariance (eps-regularized); refit
DiM directions in whitened space; recompute the truth↔refusal diagonals and off-diagonals.

**Gram-Schmidt test (descriptive):** `w_refusal_perp` = w_refusal orthogonalized against w_truth (raw).
Report refusal-test on w_refusal_perp (transfer preserved?) and truth-test on w_refusal_perp (cleaner?).

## Frozen gates (verdict precedence top-to-bottom)

Let B, C be the two truth↔refusal off-diagonal cells.

- **ENTANGLEMENT-ARTIFACT** iff, in gemma, NEITHER B NOR C is SPECIFIC-REAL (each fails the perm or the
  random-direction floor). → cycle 2 over-read the entanglement; the axes are effectively independent
  readouts and the basis is cleaner than cycle 2 stated. (Revises cycle 2 toward "basis".)
- **WHITENING-RESOLVES** iff (gemma raw: B AND C both SPECIFIC-REAL) AND (in ZCA-whitened space both
  off-diagonals ≤ 0.65 while both whitened diagonals ≥ 0.75). → the entanglement is a covariance
  artifact; a whitened/Mahalanobis readout recovers a clean basis.
- **ENTANGLEMENT-REAL** iff B AND C are both SPECIFIC-REAL in gemma AND both SPECIFIC-REAL mapped into
  Llama-3.2-3B, AND whitening does NOT clean them. → cycle 2 confirmed and strengthened: the axes share
  genuine readable structure that survives the correct nulls and whitening; "correlated frame" is real.
- **PARTIAL-STRUCTURED** — anything else (one cell real, mixed gemma/mapped, whitening partial). Report
  the exact structure; assert nothing the cells don't support.

Thresholds (0.65 independence ceiling, 0.75 diagonal floor) are inherited from cycle 2 unchanged. Bars do
not move post-hoc.

## Consequence for the record (pre-committed)

- **ARTIFACT** → annotate `FINDING_axis_independence_2026_06_11.md` and the VALUES-PORTABLE banner:
  under the correct nulls the cross-talk was not specific; the axes are independent-enough to call a
  basis. (Loud upward revision — the prior was too pessimistic.)
- **WHITENING-RESOLVES** → the clean basis exists under a Mahalanobis readout; ship that as the readout
  recipe.
- **ENTANGLEMENT-REAL** → cycle 2's "correlated frame" stands, now null-corrected and whitening-robust;
  the conscience is a basis of distinct-but-correlated directions, full stop.
- **PARTIAL** → precise structure, claim nothing more.

The point, again, is to let the correct null decide — not to rescue any prior verdict. The answer stands
whichever way it lands.
