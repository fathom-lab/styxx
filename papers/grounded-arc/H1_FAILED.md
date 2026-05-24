# H1 FAILED — refusal kill-gate

One-shot confirmatory run on the holdout hashed BEFORE scoring (no peeking, no re-runs). sha256 `53a56bf11d13c4e04cd21171529d62eec9144e832452376f3a9063067bbbaec9`.

- n = 450; 4 overlap-distance bins; calibration corpus = threshold-law `_domain_pool`; embedding = text-embedding-3-large.
- **PRIMARY (vendor-robust labeler gold):** Spearman ρ(validity, −error) = **+0.3024**, permutation p = 0.0000.
- rigor (XSTest human gold): ρ = +0.3118, p = 0.0000.
- pre-registered bar: ρ ≥ 0.40, p < 0.01.
- mean |error| (labeler) = 0.170.
- the H1 Spearman is invariant to the validity sigmoid's α/τ (validity is monotone in distance); the test is whether distance-to-calibration rank-predicts instrument error.

**VERDICT: FAIL — bet-0 abandoned; ships as a closed-negative paper: embedding-distance validity does not predict refusal-instrument reliability at the prompt level.**
