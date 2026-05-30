# PREREG — ACCELERATION 2: adaptive early-stopping on the resample tier (compounds with the cascade)

**Registered 2026-05-30, before computing the early-stop metric.** The metric is new on this data
(never computed) — no data-peek; the resample sequences were collected by prior detection-locus runs.
Inputs SHA-256'd pre-scoring.

## The question

The cascade routes WHO gets resampled. This bet trims HOW MANY samples each escalated item needs.
Detection = instability `(distinct-1)/(N-1)` over N=10 @ T=1.0. On derivation, correct items are
near-perfectly stable and confabs scatter within the first few samples — so most of the N=10 budget
may be wasted. **How few samples retain detection, and what is the compounding speedup with the
cascade?**

## Protocol (offline, per-model calibrated — the valid unit per the cascade finding)

Over every detection-locus receipt storing per-item resample sequences (`resamples`, `group`):

1. **FIXED-k:** instability from the first `k` samples; `AUC_k` vs `k`; `min_k` = smallest `k` with
   `AUC_k ≥ AUC_N − 0.02` (per regime).
2. **ADAPTIVE (monotone distinct-count certificate):** an item is confab iff it yields `≥ D` distinct
   values in `N` (D = Youden-optimal on full-N distinct count). Stop and decide CONFAB the instant
   distinct reaches `D` (distinct only grows), or CORRECT once `D` is unreachable in the remaining
   budget (`distinct + remaining < D`). Report mean samples drawn.

## Bars (fixed)

| Bar | Statement | Threshold |
|---|---|---|
| **E1** | a few samples retain detection | median over regimes of `min_k` **≤ 5** (≥ 2× resample-tier saving) at `AUC_k ≥ AUC_N − 0.02` |
| **E2** *(descriptive)* | compounding | total speedup = cascade routing × early-stop trimming, read against the cascade receipt |

**RESULT = SURVIVED iff E1.**

## Scope (stated before the run)

Offline, per-model calibrated, white-box, exact-integer resampling (the signal is sample agreement).
Counts samples / forward passes, not wall-clock. `min_k` and `D` are in-sample (the AUC_k curve is
descriptive); a clean confirmation needs held-out items. Accelerates detection, corrects nothing.
Where full-N detection is itself weak (noisy regimes), `min_k` is unstable and reported as such.
