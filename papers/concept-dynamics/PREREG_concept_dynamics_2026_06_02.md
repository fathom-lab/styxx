# PREREG — Temporal/spectral dynamics of a concept signal during LLM generation

**Date:** 2026-06-02
**Status:** PRE-REGISTERED (gate frozen before analysis).
**The honest "frequency" study.** As an LLM generates token by token, a concept becomes a
1-D time series **s(t) = ⟨h_ℓ(t), v_c⟩** — the residual at the concept's probe layer ℓ,
projected onto the concept direction v_c, at each generation step t. We ask: does s(t)
carry **genuine oscillatory/spectral structure**, or is it **trend + red noise +
commitment dynamics**? Per a literature review (logit/tuned lens; Arditi 2024; "Refusal
Falls off a Cliff" Yin 2025; hallucination-as-trajectory-commitment Akarlar 2026;
dynamical-regimes Ugail & Howard 2026), **no published work runs a per-token spectral
null test on a concept signal** — adjacent work reports ramps/plateau-then-cliff/basin
entry, i.e. **commitment, never rhythm.** And the architecture is decisive: a transformer
has **no recurrence over generation steps**, so **no dynamical mechanism for a limit
cycle.** This makes H0 the honest prior and the test a clean, almost-untried null to break.

## Hypotheses (frozen)

- **H0 (null, the honest prior):** all non-DC spectral structure in s(t) is explained by
  **linear trend + AR(1) red noise**; after detrending there is no oscillatory peak beyond
  the 1/f background and no significant harmonic line.
- **H1 (alternative, to be tested and most likely refuted):** s(t) contains a **genuine
  phase-coherent oscillatory component** at a non-DC frequency in the band **[3/n, 0.5]
  cyc/token**, with power exceeding the AR(1)+trend null AND the 1/f aperiodic background.

## Data (frozen)

`concept_dynamics_raw.json` from `measure_concept_dynamics.py`: 6 families × 4 concepts
(comply_refuse, deception, corrigibility, truthfulness) × the 48 concept-balanced prompts
(`convergence_eval_set.py`, sha256 `6154b172`). Greedy generation (deterministic), T=64
steps, eos_pos recorded. Per step we store every concept projection and **8 random-unit
directions at each probe layer** (the drift null). Primary analysis: concept-PRESENT
prompts; primary series length: full T=64 (uniform); sensitivity: truncate-at-eos.

## Analysis pipeline (frozen — choices pre-committed)

1. **Detrend.** Primary: remove mean + linear OLS trend. Robustness: {mean-only, linear,
   quadratic} (a claimed peak must survive all three).
2. **AR(1) red-noise null.** Fit (ρ, σ²) by lag-1 autocorrelation on the detrended series.
3. **Spectrum.** Multitaper (Slepian DPSS, **NW=3, K=5**) periodogram on the detrended
   series; analysis band **[3/n, 0.5] cyc/token** (exclude lowest ~3 bins where trend/AR(1)
   false positives live — claims about period > n/3 are excluded by construction).
4. **Two surrogate nulls, ≥1000 each** (max-power-in-band statistic):
   - **AR(1) parametric surrogates** (simulate from fitted ρ,σ²) — tests "beyond AR(1)".
   - **IAAFT surrogates** (preserve the exact power spectrum AND amplitude distribution,
     randomize phases) — the decisive test for phase structure beyond linear autocorrelation.
5. **Harmonic line test.** Thomson multitaper F-test (num df=2, den df=2K−2) with
   look-elsewhere threshold **F > F_{2,2K−2}(1 − 1/n)**.
6. **1/f-vs-peak gate (specparam-style, implemented in numpy):** fit the aperiodic
   background as a line in log-power vs log-freq; a peak counts only if **≥2 SD above** the
   fit, center frequency in band. Report the aperiodic exponent β.
7. **RoPE/positional confound:** any surviving peak is cross-checked against the
   **random-direction null** trajectories and tested for **phase-lock to absolute position
   across prompts**. A peak that also appears in random directions and/or is position-locked
   is attributed to **positional (RoPE) structure**, NOT concept oscillation.

## KILL-GATES for claiming OSCILLATION (frozen — ALL must pass)

1. **Surrogate (AR(1) red-noise line test):** observed max-band power beats the **95th
   percentile of the AR(1) surrogate max-power distribution** (p<0.05), on **≥⅓ of
   concept-present sequences** for a concept.

   > **Instrument-validation amendment (pre-data, 2026-06-02).** The original gate also
   > required beating IAAFT surrogates on max-power. The synthetic self-test
   > (`--selftest`, run *before any real data*) revealed this is mis-specified for line
   > detection: **IAAFT preserves the power spectrum by construction**, so a genuine
   > sinusoid does not exceed its own IAAFT surrogates' peak power — the dual gate
   > rejected ~73% of planted sinusoids. IAAFT tests *nonlinear phase structure beyond
   > the linear spectrum*, which a linear oscillation does not have. So IAAFT is retained
   > as a **nonlinearity diagnostic** (`iaaft_nonlinear`), and detection rests on the
   > AR(1) red-noise test + the harmonic F-test (gate 2). Specificity is unaffected — red
   > noise and commitment ramps produced **0/40** false positives under both versions.
   > This amendment is documented because changing a frozen gate, even pre-data, must be
   > on the record.
2. **Harmonic F-test:** a line survives the look-elsewhere-corrected F threshold.
3. **1/f gate:** a peak ≥2 SD over the aperiodic fit, center freq in band.
4. **Detrend-robust:** survives all three detrend models.
5. **Replication:** the center frequency is **concentrated across independent prompts**
   (not scattered), AND is **not** explained by the random-direction/positional confound.

## Readings (fixed)

- **OSCILLATION:** all 5 gates pass for ≥1 concept, and it is **not** positional → a
  genuine, surprising finding (concept signals carry rhythm) — counter to the architecture
  and the prior, so the bar is deliberately severe.
- **POSITIONAL:** a peak survives the surrogate/line tests but is **RoPE/position-locked**
  (appears in random directions too) → a real *mechanistic* finding about positional
  structure, explicitly **not** "the concept oscillates."
- **COMMITMENT (expected):** H0 holds — no oscillation survives the gates; s(t) is **trend
  + AR(1) + structured commitment.** We then characterize the commitment signature (the
  *real* positive result): trajectory shape (ramp / plateau→cliff / basin-entry), the
  changepoint/time-to-commit, AR(1) ρ, and **concept-vs-random structure** (is the concept
  trajectory more structured than random directions at the same layer?). This is the first
  rigorous per-token spectral null test for a concept signal — a clean, honest negative on
  oscillation plus a positive characterization of commitment.

## Honest caveats (frozen)

- At n≈64 the test is **underpowered for low frequencies by construction** — claims are
  restricted to the mid/high band; a low-freq null is expected and stated up front.
- Greedy decoding (deterministic). A sampled/ensemble variant + cross-generation PLV is a
  noted follow-up (for phase-lock-across-runs).
- Concept directions are the **shipped atlas probes** (pre-registered, not chosen post-hoc).
- Transformers only here; an **SSM/Mamba contrast** (complex-eigenvalue state rotation —
  where oscillation is architecturally plausible) is the natural strong follow-up if H0 holds.

— frozen 2026-06-02
