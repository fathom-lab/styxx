# FINDING — the entanglement is a COVARIANCE artifact; a whitened readout recovers a clean basis (WHITENING-RESOLVES)

**2026-06-11 · Fathom Lab / styxx. Pre-registered: `PREREG_entanglement_resolution_2026_06_11.md`
(frozen pre-run, committed 5a510a5). Receipt: `entanglement_resolution_result.json`. Backlog B28. This
resolves the ambiguity left by cycle 2 (`FINDING_axis_independence_2026_06_11.md`, PARTIAL-STRUCTURED):
its truth↔refusal cross-talk was measured by discriminability at small n with an inflated floor. The
correct nulls + a whitening test now decide it — and they decide it cleanly.**

## Result — real in the raw stream, gone under whitening

Common gemma layer (L12), three axes (truth / refusal / valence-control), unit difference-of-means
directions, larger sets (truth n=88, refusal n=88, valence n=48; test 27 / 27 / 15). For each
truth↔refusal off-diagonal cell, two nulls: a K=1000 label-permutation floor and a 1000-draw
random-direction floor. SPECIFIC-REAL ≡ obs beats BOTH.

**Raw gemma — the cross-talk is REAL and SPECIFIC (cycle 2 was not imagining it):**

| cell | obs discrim | perm-null p95 | rand-dir p95 | p_perm | SPECIFIC-REAL |
| --- | --- | --- | --- | --- | --- |
| truth-dir → refusal (B) | 0.9778 | 0.7278 | 0.8614 | 0.001 | **yes** |
| refusal-dir → truth (C) | 0.9013 | 0.7368 | 0.8289 | 0.001 | **yes** |

Both off-diagonals beat the permutation floor AND the random-direction floor — the entanglement is not a
small-n mirage and not merely "any direction separates a broadly-separable axis." Diagonals: truth 0.9737,
refusal 1.0; cos(w_truth, w_refusal) = −0.2756.

**ZCA-whitened — the cross-talk VANISHES while the axes survive intact:**

| metric | raw | whitened |
| --- | --- | --- |
| truth-on-truth (A) | 0.9737 | 0.9737 |
| refusal-on-refusal (D) | 1.0 | 1.0 |
| truth-dir → refusal (B) | 0.9778 | **0.55** |
| refusal-dir → truth (C) | 0.9013 | **0.5461** |
| cos(w_truth, w_refusal) | −0.2756 | **−0.0** |

Whitening the residual stream by its pooled-train covariance drives both off-diagonals to chance while
keeping both diagonals perfect, and makes the two directions EXACTLY orthogonal. **clean_basis = true.**

**Gram-Schmidt corroborates independently:** orthogonalize w_refusal against w_truth and it still reads
refusal at AUROC 1.0 while its reading of truth drops to discriminability 0.5132 (chance); symmetrically,
w_truth orthogonalized reads truth at 0.9671 and refusal at 0.5611. An explicitly independent basis loses
nothing on its own axes.

**Verdict per the frozen gate: WHITENING-RESOLVES.** The cycle-2 entanglement is real in the raw,
anisotropic dot-product geometry but is a pure COVARIANCE artifact: under the representation's natural
(Mahalanobis) metric the conscience axes are a clean orthonormal basis.

## What this does to the cycle-2 verdict — upgrades it

Cycle 2 (PARTIAL-STRUCTURED) called the axes "distinct but entangled — a correlated frame, not an
orthonormal basis." That was correct FOR THE RAW READOUT. This cycle shows the entanglement lives
entirely in the residual stream's covariance, not in the directions' semantics: whiten, and the frame
becomes orthonormal with no loss of per-axis signal. The honest upgrade: **in the SOURCE (gemma) space,
the portable conscience IS a basis of independent value axes; "read it whitened" is the readout recipe.**
(The cross-model MAPPED readout needs whitening in the mapped metric, not gemma's — established later in
`FINDING_mapped_whitening_2026_06_12.md`; here the result is the source-space geometry.) Cycle 2 is not
retracted — its raw-space measurement was right, and it explicitly owed this test — resolved one rung up.

## Honest bounds (what is NOT claimed)

The whitening test was run in the SOURCE (gemma) space, where the result is crisp. In the cross-model
MAPPED space the picture is already dominated by the map's broad transport: through the label-free map,
refusal is so broadly separable that random directions reach the high floor (rand-dir p95 0.95 Llama /
0.9667 Qwen), so the truth→refusal mapped cell is NOT specific-real (the broad-transport regime the
VALUES-PORTABLE finding already documented), while refusal→truth stays specific-real (p_perm 0.005 both
targets). Applying the SAME whitening recipe in the mapped target space to clean the (mostly-floor)
cross-model cross-talk is the natural next step and was NOT run here — stated, not spun. ZCA with
eps-regularization (eps small) on n_train ≈ 155 in a 2304-dim space is a regularized covariance estimate,
not a full-rank one; the diagonals' perfect retention and the off-diagonals' clean collapse argue the
estimate is adequate, but a held-out covariance / shrinkage sweep would harden it. Linear DiM directions,
three axes, one source model, local open weights; n as stated. No model generated any response; valence
items are benign/truthful and no operational harmful content appears anywhere. The frontier this opens:
whitened readout in the mapped space, a shrinkage-covariance robustness sweep, and a ≥3-axis whitened
basis (B27) to confirm the orthonormality generalizes past two axes.
