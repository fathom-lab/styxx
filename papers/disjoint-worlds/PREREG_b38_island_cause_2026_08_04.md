# PREREG — B38: naming the island — can whitening rescue qwen's legibility?

Fathom Lab · 2026-08-04 · frozen before the scored run. B37 licensed the matrix: a
cross-family legible clique {llama_3b, llama_1b, gemma_2b} and qwen_1p5b an island from every
direction (discovery ≤ 0.17 both ways, all pairs). This run tests the one causal candidate the
program's own committed machinery supplies.

## Disclosed reconnaissance (run on committed extractions BEFORE this freeze, per the c81 rail)

The obvious physics suspect — outlier activation dimensions crushing the distance geometry —
was checked first and is **dead on arrival**: qwen has the LOWEST top-dim variance share of
all four models (0.007 vs gemma's 0.036 — and gemma is in the clique), the HIGHEST effective
dimensionality (participation ratio 18.7 vs 15.9–17.3), and an identical pairwise-distance CV
(0.218 vs 0.217–0.221). The island is not a degenerate space. Recon numbers are persisted in
`b38_recon_addendum.json`; the outlier-dim treatment below is retained only as a control arm,
with the recon's prior against it stated here.

## The causal candidate (from committed machinery)

B28/B29 established that ZCA-whitening resolves cross-model structure that raw geometry
misreads (entanglement was a covariance artifact; whitened bases cleared cross-model). If
qwen's illegibility lives in its **covariance structure** — feature correlations reweighting
the distances GW and Procrustes consume — then discovery in per-model ZCA-shrunk-whitened
spaces should rescue it. If whitening does not rescue, the remaining explanation is that
qwen's concepts are **intrinsically differently arranged** (supported by recon: an internally
healthy space that no one can match).

## Design (frozen)

Treatments applied per-model to the anchor points before the stand