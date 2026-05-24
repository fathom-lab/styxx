# Phase 1 — threshold-law curve replication (2026-05-24)

**Ship gate (brief §Phase-1):** the threshold-law curve must replicate the
published paper within ±0.02 Spearman on existing data before the locked
constant τ is trusted by the validity engine. No holdout is touched.

## Result — GATE CLEARED

Recomputed from the committed `scripts/dogfood/out_corpus_coverage_law_fine.json`
(n=12 overlap levels, the high-resolution study), using the kill-gate's own
pure-Python `spearman_rho` and cross-checked against `scipy.stats.spearmanr`:

| quantity | recomputed | published | Δ |
|---|---|---|---|
| cross-family Spearman(overlap, AUC) | **+0.6865** | +0.6923 | 0.0058 |
| same-family Spearman(overlap, AUC) | **−0.3993** | −0.4056 | 0.0063 |
| τ — min overlap with cross-family AUC ≥ 0.80 | **0.31** | 0.31 | exact |

scipy cross-check: cf **+0.6865**, sf **−0.3993** — identical to the scaffold's
pure-Python Spearman. The H1 kill-gate's statistics are independently confirmed.

## What this locks

- **τ_overlap = 0.31** (published, now reproduced) → **τ_distance = 1 − 0.31 = 0.69**
  in cosine-distance units, the constant `validity_engine.py` uses.
- The scaffold's `spearman_rho` (used for the H1 ρ test) matches scipy exactly.

## Honest note

The same-family |ρ| = 0.40 narrowly exceeds the *transport study's* flat-control
criterion (|ρ| < 0.40) — this is already documented in
`papers/threshold-law-2026-05-18.md` and is a property of the transport study,
not of the grounded-arc validity use, which depends only on τ and the
cross-family monotone curve. Both reproduce.

## Still gated (unchanged — the integrity line)

This clears a *calibration* gate on existing data. It does NOT touch the H1
holdout. The confirmatory kill-gate (ρ(validity, −error) ≥ 0.40 on the refusal
holdout) still requires: locked operator decisions, a constructed+hashed
holdout, and the validation-slice α fit — then `run_bet0_phase1.py` runs once.
