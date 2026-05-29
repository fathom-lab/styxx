# PREREG — Is truth in the SPECTRUM of the answer-formation trajectory? (1/f signature of construction vs retrieval; "truth-flash" suppression test)

**Pre-registered 2026-05-29, BEFORE any spectral/trajectory code is written or run. One
confirmatory run. Feasibility-grade: single open model (Qwen2.5-1.5B-Instruct), SAE-free
logit-lens, n=36 arithmetic items × 2 generation conditions (one-shot, method-diverse
derivation). Arithmetic ground truth computed in-code (the `run_competence_cliff` SPECS)
and SHA-256'd before any scoring (expected to match the white-box run's
`ddccd8e4…b87964d` — same items, same key). Correctness = exact integer match (no judge).**
Receipt: `spectral_trajectory_result.json`.

## Why this run (the reframe)

Two prior nulls assumed truth is *absent* at confabulation time: within-mode mean-depth was
blind to correctness (derivation-stratum AUC 0.498, chance), and a linear residual-steering
push flipped zero confabulations. Both used **scalar / linear** summaries — a single number
(mean attribution depth) and a single direction (difference-of-means). The music literature
on what perception is drawn to (Voss & Clarke 1975; Levitin, Chordia & Menon, PNAS 2012 —
rhythm spectra from Bach to Joplin obey a **1/f power law**, each composer with a distinct
exponent β) says the discriminating information in a structured signal lives in its
**scale-invariant spectrum**, not its mean or its dominant direction.

Hypothesis: the layer-by-layer trajectory by which the answer token forms is such a signal,
and we have been measuring its average instead of its shape. **Construction** (derivation)
should assemble the answer gradually across scales — a *pinker* (more 1/f) trajectory.
**Retrieval** (one-shot confabulation) should *snap* onto an attractor — a whiter,
impulse-like trajectory (most of the logit rise concentrated in one late layer). If so, the
spectral slope β separates the two modes — and, the prize, may separate **correct from
confabulated WITHIN a mode**, where scalar depth could not. This run also tests the
companion suppression hypothesis: on confabulations, does the **correct** token's logit-lens
rank *lead* at an intermediate layer before a late hop overwrites it ("truth-flash-then-death")?

## Apparatus (committed before data)

- **Model:** Qwen2.5-1.5B-Instruct, local, greedy decoding (deterministic; reproduces the
  white-box generations and answer-key hash).
- **Per-token logit-lens trajectory:** for each transformer layer ℓ (0…L, L=28 incl.
  embeddings), project the residual at an answer position through `final_norm` + unembed and
  read the logit of a chosen vocabulary token. This gives a length-(L+1) trajectory `t[ℓ]`.
- **Spectral slope β (headline, tied to the music result):** detrend `t` (remove linear
  trend), take the periodogram via FFT, fit `log power ~ −β · log frequency` over the
  resolvable band; β is averaged across the item's answer-token positions to tighten the
  estimate (multi-digit answers give multiple trajectories per item). Higher β = pinker /
  more 1/f / more "built"; β≈0 = white / "snap".
- **Snap index (robust short-series corroborator):** the fraction of the trajectory's total
  monotone rise contributed by its single steepest layer-to-layer step. High snap = impulse
  (retrieval); low snap = distributed build (construction). Reported alongside β because a
  29-point β estimate is intrinsically noisy; agreement of the two estimators is the K guard.
- **Truth-flash crossing (suppression test):** on one-shot **confabulation** items, at the
  first answer position where the realized (wrong) answer and the correct answer differ in
  token, compare the logit-lens rank of the **correct** token vs the **realized** token
  across layers. A *crossing* = the correct token outranks the realized token at some
  intermediate layer ℓ < L and loses by the final layer.

## Predictions (decisive bars — pre-stated, scored exactly, both directions named)

| id | prediction | bar |
| --- | --- | --- |
| **F1 — spectral mode difference** | the answer-formation trajectory's β differs between method-diverse derivation and one-shot confabulation, **derivation pinker** | paired \|Cohen's d\| ≥ **0.5** AND p < **0.05**, predicted sign β(deriv) > β(confab). If reversed/flat → reported against prediction. |
| **F2 — within-mode truth signal (the prize)** | β separates CORRECT from CONFABULATED holding mode fixed, where mean depth did not (0.498) | within-mode AUC(β) ≥ **0.70** OR ≤ **0.30** in the well-powered stratum (≥8 vs ≥8). If in (0.30,0.70) → spectrum is *also* mode-only, a clean null that hardens the irreducibility claim. |
| **F3 — truth-flash suppression** | on confabs, the correct token *leads* at an intermediate layer before being overwritten | crossing rate ≥ **0.25** of confab items. If ≈0 → truth is *absent*, not suppressed (supports irreducibility); reported as such. |
| **K — not an artifact** | the β mode-effect is specific to the answer token and not a length/any-token artifact | (a) a CONTROL token (fixed common token, same positions) shows NO mode β difference (paired p > **0.05**) while F1 holds; AND (b) β and snap-index agree in sign on F1. |

**RESULT = SURVIVED iff F1 ∧ F2 ∧ F3 ∧ K.** Otherwise REPORT_AS_LANDED with whatever held,
scored against prediction. The honest prior (below) expects F1 to hold, F2 to be the real
gamble, F3 genuinely uncertain.

## Precondition / honest failure modes (stated in advance)

1. **Short series.** L+1 = 29 points is short for a spectral fit; β is noisy. Mitigation:
   average β across answer-token positions per item, and **bootstrap a CI on each item's β**.
   If the median per-item β bootstrap-CI width exceeds **1.0** (β indistinguishable from its
   neighbors at the item level), F1/F2 are downgraded to **descriptive** — no SURVIVED claim
   on β; the snap-index becomes the reported primary. This is a real possibility and is not a
   failure of the hypothesis, only of the estimator's resolution at 29 points.
2. **Under-powered within-mode stratum.** If a mode has < 8 correct or < 8 confab items, F2 is
   under-powered in that stratum → report descriptively (same constraint that limited the
   prior within-mode test, where one-shot had only 4 correct).
3. **Honest prior.** I expect **F1 to hold** — construction and retrieval almost by definition
   differ in *how* the answer forms across layers, so a shape/spectral difference is likely.
   **F2 is the genuine gamble**: scalar depth was chance within-mode; the spectrum may carry a
   truth signal the mean threw away, or may itself be mode-only. **F3 is a coin-flip on
   mechanism**: truth may be computed-then-suppressed (crossing) or simply never present
   (no crossing). A clean F2 null + F3 null would *strengthen*, not weaken, the standing
   claim that truth is irreducible to any single-pass internal read.

## Honest scope (pre-committed)

Single open model, SAE-free logit-lens trajectories, feasibility-grade n=36, one confirmatory
run; arithmetic ground truth computed in-code then hashed; exact-integer correctness, no judge.
β is estimated from a 29-point trajectory — a crude spectral measurement reported with a
bootstrap CI and a robust short-series corroborator (snap index), not a clean 1/f fit over
decades of frequency. This tests whether the *scale structure of the answer-formation
trajectory* carries mode and/or truth information beyond the scalar mean depth — nothing more.
A null does not refute the 1/f framing in general (it may need an SAE-feature trajectory, a
deeper model with more layers, or a richer estimator); a signal motivates exactly those next
steps. The music/1/f connection is the *motivation* for the estimator choice; the claim scored
here is strictly about within-model trajectory spectra, not about music.
