# FINDING — the cross-family/vendor cliff Spearmans are imprecise: distinctions are within noise on a single run

**Bootstrap variance analysis, `bootstrap_cliff_variance.py`, on per-item data already on disk (no GPU,
no new runs).** Triggered by the genmatch result: re-sampling moved within-open agreement 0.770 → 0.613,
exposing that none of these cross-family/vendor agreements had variance bars. This adds them.

## Method

Per-item ungated-correctness (modal cluster matches the true answer, via `run_pregeneration_gate.
_modal_cluster_info`) from each benchmark JSON. Bootstrap (B=2000), two ways: **item-only** (37 domains
fixed, resample items within each → precision of *this* run's estimate) and **hierarchical** (also
resample domains → generalization to other domains). Per draw, recompute per-domain hallucination rate
and the pairwise Spearman; take 2.5/97.5 percentiles.

## Results (hallucination-cliff Spearman)

| comparison | point | item-only 95% CI | hierarchical 95% CI |
|---|---|---|---|
| within-open (24-token) | 0.770 | [0.50, 0.72] | [0.34, 0.81] |
| open-closed (24-token) | 0.473 | [0.29, 0.54] | [0.08, 0.69] |
| within-open (32-token, gen-matched) | 0.613 | [0.39, 0.63] | [0.25, 0.74] |
| open-closed (32-token, gen-matched) | 0.510 | [0.33, 0.55] | [0.14, 0.69] |

## What this establishes

1. **The intervals are wide** — even the tightest (item-only, domains fixed) span ≈ ±0.1; hierarchical
   ≈ ±0.2–0.3. With 37 TruthfulQA categories, several at tiny n (Misconceptions:Topical n=3, Statistics
   n=5, …), the per-domain rates are noisy and the rank correlation is imprecise.
2. **within-open vs open-closed CIs OVERLAP** (item-only: within-open [0.50,0.72] vs open-closed
   [0.29,0.54] meet near 0.50–0.54; hierarchical overlap far more). → The **"open cluster vs OpenAI
   outlier" distinction is NOT statistically supported on a single run.**
3. **24- vs 32-token within-open CIs OVERLAP** ([0.50,0.72] vs [0.39,0.63]). → The genmatch 0.77→0.61
   drop is **within sampling noise**; it cannot be attributed to the token change (resolves the genmatch
   confound by showing the drop is noise-consistent — neither a clean token effect nor a clean null).
4. **What survives:** the agreements are **positive** (most CIs sit above 0; the weakest, open-closed
   hierarchical, includes 0.08 — barely positive) but their **magnitudes and any cross-provider
   distinctions are not pinned down** by one 37-domain run.

## Caveat on the bootstrap itself

Percentile bootstrap CIs for rank correlations are **downward-attenuated**: resampling injects noise
into each domain's rate, shuffling ranks and lowering |Spearman|, so the bootstrap distribution centers
*below* the full-data point estimate (visible here — 0.770 sits at/above the item-CI upper bound 0.72).
So the exact bounds are approximate; the load-bearing, robust conclusion is the **qualitative overlap**
(distinctions not separable), not the precise interval edges.

## Consequence for the whole arc (the deepest correction)

Every single-run Spearman in the cross-family/cross-vendor cliff arc — the "0.77 cross-family
invariance," the "OpenAI outlier 0.47," the genmatch "gap" — was a point estimate with a wide,
overlapping CI. They should be reported as **"positive but imprecise (≈0.5–0.77, wide CI), single run"**,
not as precise invariants or clean distinctions. This also applies prospectively: **the MMLU run (in
flight) needs the same CIs before its ≥0.55/<0.35 verdict is trusted** — and the pre-registered bars
should be read against the interval, not the point.

The honest program-level statement that survives all the corrections today: *different model providers
share, to a moderate and positive but imprecisely-estimated degree, which topics they hallucinate on;
the apparent precision and the open/closed distinction were artifacts of single-run point estimates,
apparatus mismatch, and judge mismatch — each caught by a disciplined follow-up.*

## Banked

- **Multi-run replication** (≥3 same-config re-samples) to get a generation-variance CI directly, not
  just the item/domain bootstrap — the one source this analysis cannot capture (it resamples within a
  fixed set of generated samples, not fresh generations).
- More domains / larger per-domain n (or a coarser, better-powered category scheme) to tighten the CI.

## Receipts

- `bootstrap_cliff_variance.py` → `bootstrap_cliff_variance_result.json`. Inputs: the committed
  `crossfamily_benchmark_*{,_gm32}.json` and `xvendor_gpt4omini_nli_benchmark.json`. Relates to
  `FINDING_genmatch_xvendor_2026_06_23.md`, `FINDING_crossfamily_local_cliff_2026_06_22.md`.
