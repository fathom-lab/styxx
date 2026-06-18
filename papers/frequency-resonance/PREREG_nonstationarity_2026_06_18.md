# PREREG — the object the dead probe threw away: non-stationarity / Koopman-residual switching

**2026-06-18. FROZEN before any feature code or data.** The Spectral Integrity Probe is CLOSED
(`RESULT_hankel_k1_2026_06_18.md`): the DMD/Koopman **spectrum** of the residual-stream trajectory
carries no cognitive-state signal across 3 extraction regimes × 2 models. This pre-registers a
**different object** on the same harness — and states its kill-gate before looking.

## The insight from the closure
The static spectral probe fits ONE linear operator `A` to the whole trajectory (`h_{t+1} ≈ A h_t`)
and reads `eig(A)`. That assumes the generation is **stationary**. It is not — an LLM generation is a
**switching linear dynamical system** that moves through regimes. The static spectrum *averages the
switching away*. The hypothesis here is that the discarded quantity — **how non-stationary the
dynamics are, and how hard the operator switches** — is the integrity-relevant object, because
deception / manipulation / jailbreak-compliance plausibly **requires a regime change** (the model
transitions out of its default honest dynamics into a cover/comply mode).

This is a genuinely different object from the dead probe (the *drift* of the dynamics, not their mean
spectrum) and from the known-but-not-novel signal (trajectory **magnitude** / path-length: that is the
*state* moving; this is the *operator* changing).

## Method — same harness, new feature object
- **Identical:** model `EleutherAI/pythia-410m` (CPU float32); layers `[6,12,18]`; the `PASSAGES`
  coherent-vs-shuffled proxy (same `shuffle_words`, same `auroc`); per-layer (T×24) PCA trajectory;
  `StandardScaler` → `LogisticRegression(C=0.5)`; **5-fold CV** AUROC; **bar 0.70**.
- **New features (per layer)** computed from a rank-12 global Koopman fit `A` and the trajectory:
  1. `resid_cv` — std/mean of the one-step residual `r_t = ||h_{t+1} − A h_t||` over t (spiky ⇒ switching)
  2. `resid_max_ratio` — max(r)/mean(r) (single hard switch)
  3. `resid_autocorr1` — lag-1 autocorrelation of r_t (structured vs white residual)
  4. `n_jumps` — fraction of t with `r_t > mean(r)+2·std(r)` (regime-switch events)
  5. `operator_drift` — `‖A_first − A_second‖_F / ‖A_first‖_F`, operators fit on the two trajectory
     halves (the most direct "the dynamics themselves changed" measure)

## Directional hypothesis (honest, for interpretation only — gate is two-sided)
Shuffled text has no narrative regime, so its local dynamics should be **more** erratic/non-stationary
(higher `resid_cv`, higher `operator_drift`) than coherent text. The classifier handles sign; the gate
is on separability, not direction.

## Kill-gate (PRIMARY — the verdict)
- **K1-NS.** The 5-feature × 3-layer non-stationarity vector separates coherent vs shuffled at
  **5-fold CV AUROC ≥ 0.70**. This is the *same* gate the static spectrum scored **0.461** on, on the
  *same* data — so a pass is a direct, like-for-like demonstration that the switching object lives
  where the spectral object died.

## Secondary (transparency, non-verdict)
- Per-feature univariate AUROC (which non-stationarity feature, if any, carries it).
- Per-layer combined AUROC.

## Honest priors
- P(K1-NS ≥ 0.70) ≈ **0.40–0.50.** Higher than the spectral re-test's prior: this measures a quantity
  the dead probe provably discarded, the directional story is mechanistically plausible, and
  residual-/operator-drift on a single global fit is far better conditioned than per-window spectra.
  Still a real coin-flip — coherent narrative has its own transitions, and n=32 is small.

## Stop rule (binding)
- Verdict = the PRIMARY K1-NS number. No secondary rescues a failed primary.
- **If it clears 0.70:** the switching/non-stationarity integrity object is alive — design the
  pre-registered deception + obfuscation test (does an attacker who flattens a static probe also
  flatten the *operator switch*? the value proposition). This becomes the new arc.
- **If it fails:** single-trajectory dynamical integrity (spectrum AND switching) is closed on this
  proxy; the only remaining honest dynamical bet is **cross-sample** (resample-ensemble) consistency,
  which inherits styxx's proven sampling-divergence edge — note it and stop here, do not flog.
