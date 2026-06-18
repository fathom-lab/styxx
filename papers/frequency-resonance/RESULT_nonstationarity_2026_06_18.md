# RESULT — K1-NS: the switching object is ALIVE but sub-threshold; the signal is dynamical, the proxy is single-trajectory

**2026-06-18.** Executed `PREREG_nonstationarity_2026_06_18.md` (frozen 18930a6, before feature code).
Outcome: **fails the pre-registered 0.70 bar at 0.672 — but carries +0.21 AUROC over the dead spectral
probe on the identical gate.** Per the binding stop rule I do NOT tune it over the bar; I report what
it means and where the prereg already said to go.

## What ran
Same harness as the spectral re-test (pythia-410m CPU float32, layers [6,12,18], the `PASSAGES`
coherent-vs-shuffled proxy, StandardScaler+LogReg(C=0.5), 5-fold CV, **bar 0.70**). The ONLY change
is the feature object: **non-stationarity** — the time-profile of the one-step Koopman residual and
the operator's drift between trajectory halves — instead of the DMD eigenvalue spectrum. n=32.
Mechanism self-validated first: on a synthetic *switching* system vs a *stationary* one, `resid_cv`
rises 4.7× (0.15→0.69) and `operator_drift` rises (`nonstationarity_features._selftest` PASS).

## Results
**PRIMARY: K1-NS 5-fold CV AUROC = 0.672 < 0.70 → FAILS.** On the *same gate, same data* the DMD
spectrum scored **0.461**. The switching object carries **+0.211 AUROC** over the spectral object.

Per-feature univariate AUROC (transparency):

| feature | AUROC | |
|---|---|---|
| `resid_cv` | **0.707** | residual spikiness — over the bar by itself |
| `resid_max_ratio` | 0.684 | |
| `resid_autocorr1` | 0.684 | structured (non-white) residual |
| `n_jumps` | 0.660 | (the self-test-flagged weak feature still contributes) |
| `operator_drift` | 0.531 | half-vs-half operator change — weak at this length |

Per-layer combined AUROC: L6 **0.387**, L12 **0.621**, L18 **0.645** — signal concentrates in the
**deep layers**, absent shallow. Both the feature profile (residual non-stationarity > operator-half-
drift) and the depth profile are mechanistically coherent, not noise.

## Verdict (pre-registered primary, binding)
**0.672 < 0.70 → K1-NS FAILS.** The single-trajectory switching proxy does not clear the bar. I am
**not** tuning it over (no feature pruning, no q/rank search, no n-inflation) — the binding stop rule
exists precisely to forbid converting a 0.672 into a 0.70 after seeing the data.

## But this is a genuinely different outcome from the spectral closure — and it is directional
- The spectrum was **dead** (0.461, below chance, *degraded* with more extraction). The switching
  object is **alive** (0.672, clearly above chance, one feature over the bar, coherent depth/feature
  structure). The reframe was right: the static probe discarded the live quantity.
- It fails the bar because the **proxy is single-trajectory** (~90 noisy samples, one generation).
  The residual-non-stationarity carries it; the operator-half-drift is starved for samples. More of
  *this* proxy is not the move.

## Where the prereg already said to go (NOT flogging — a different object with independent support)
The stop rule named it before the run: **cross-sample consistency.** Resample the generation K times
and read whether the *dynamics* agree across samples (dynamical consensus) rather than reading one
trajectory's dynamics. This is not a new guess:
- It inherits styxx's single most-proven edge — **sampling divergence broke the grounded-honesty
  ceiling** (AUC 0.966 vs 0.498 text; `project_grounded_honesty_ceiling_break`).
- It directly fixes the failure mode here (single-trajectory underdetermination): K trajectories is a
  far richer object than one.
- It is novel in this framing: dynamical/modal consensus across resamples, distinct from semantic
  entropy of *answers* — and its confound gate is exactly "must beat semantic entropy."

## Net
Two pre-registered swings this session converted cleanly: the spectral probe is a hard closed-negative;
the switching object is alive-but-sub-threshold and points, by its own pre-registered stop rule, at the
cross-sample dynamical-consensus test — the one dynamical bet that sits on styxx's proven moat. The
single-trajectory dynamical line is closed at K1; the cross-sample line is the next pre-registered step.
