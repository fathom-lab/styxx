# Embedding-distance validity weakly predicts cognometric-instrument reliability

**Closed negative · styxx 8.0 grounded arc · Bet 0 · 2026-05-24**
Pre-registration: `preregistration_2026_05_19.md` (locked `53269f6`).
Holdout hashed before scoring: `060ef7a`. Result: `ebe0475`.

## Claim under test (H1)

Per-call **validity** — a 0–1 estimate of how reliable an instrument's score
is on a given input — was proposed to be computable from the prompt's
embedding distance to the instrument's calibration corpus, mapped through the
published threshold-law curve. The bet: if this holds, every styxx audit could
self-disclose its own scope at runtime (`score`, `validity`), making the 7.4.1
README scope-honesty *compile into the API*.

**Pre-registered kill-gate:** Spearman ρ(validity, −error) ≥ **0.40**
(permutation p < 0.01) on the refusal holdout. Below 0.40 → abandon the arc.

## Method

- **Instrument:** refusal (`styxx.guardrail.refuse_check`).
- **Calibration corpus (locked at paper values):** the threshold-law security
  domain pool (`corpus_coverage_law._domain_pool()`, n=192), embedded with
  `text-embedding-3-large`. Threshold-law curve replicated from committed data
  first: cross-family Spearman +0.6865 (pub +0.6923), **τ=0.31 exact**,
  scipy-confirmed (`04a8904`).
- **Validity:** `sigmoid(α·(τ_distance − d))`, d = mean cosine distance to the
  10 nearest calibration points; τ_distance = 1 − 0.31 = 0.69. *The H1 Spearman
  is invariant to α/τ — validity is monotone in d, so the test reduces to
  whether distance-to-calibration rank-predicts instrument error.*
- **Holdout:** XSTest-v2 `gpt4` split, **n=450** (≥ pre-registered 400),
  stratified into 4 overlap-distance bins (d spans 0.62–0.98). Gold via the
  vendor-robust labeler (`detect_refusal`, prereg §7); XSTest human
  `final_label` retained as a cross-check (labeler↔human agreement 0.944).
- **Discipline:** holdout hashed (`53a56bf1…`) and committed *before* the
  instrument scored it; the kill-gate ran **once**; no peeking, no re-runs, no
  optional stopping; the 0.40 bar was held, not lowered.

## Result

| gold | Spearman ρ(validity, −error) | permutation p | n |
|---|---|---|---|
| vendor-robust labeler (primary) | **+0.3024** | < 0.0001 | 450 |
| XSTest human (cross-check) | +0.3118 | < 0.0001 | 450 |

**ρ ≈ 0.30, below the 0.40 bar. H1 fails. Bet 0 is abandoned.**

## Interpretation

The signal is **real and significant** — embedding distance does carry
information about where the instrument is reliable (p < 0.0001, robust to the
gold choice). But it explains only ~9% of the rank-variance in reliability,
not enough for a number a user would *quote* ("validity 0.2 — don't trust
this"). **The threshold-law's corpus-level overlap signal attenuates at the
prompt level.** Per-call scope disclosure cannot be built honestly on
embedding distance alone.

This constrains the search for everyone working on per-call scope honesty:
the cheap, obvious substrate (prompt-embedding distance) is insufficient. The
next lever the program has named throughout — **model-internal signal
(logprobs / entropy)** — is the honest forward direction, as a *new*
pre-registration, not a re-run of this one. Re-running with a different
calibration corpus to chase 0.40 would be post-hoc fishing; the locked design's
result stands.

## The chain

This joins the program's pre-registered closed negatives — deception-v1
(TruthfulQA AUC 0.59), text-only overconfidence (`7c36ed9`, H_null),
cross-vendor universality (`b2675c4`). Four bets specified before the data,
killed by the data, shipped honestly. The chain is the credibility deposit:
it is what makes styxx's *passing* numbers worth trusting.
