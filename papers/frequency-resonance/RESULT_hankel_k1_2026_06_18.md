# RESULT — K1-Hankel: the Spectral Integrity Probe is CLOSED (robustly dead under the right estimator)

**2026-06-18.** Executed `PREREG_hankel_k1_2026_06_18.md` exactly as frozen (commit b97548b, before
any data). Outcome: **the pre-registered primary fails, and fails in the wrong direction.** Per the
binding stop rule, the Spectral Integrity Probe is closed.

## What ran
The one-change re-test: identical to v2 (pythia-410m, CPU float32, layers [6,12,18], the `PASSAGES`
coherent-vs-shuffled proxy, same 5 spectral features/layer, same StandardScaler+LogReg(C=0.5), same
5-fold CV AUROC, same 0.70 bar) — the **only** difference is a time-delay (Hankel/Takens) embedding
of each trajectory before the identical DMD spectral fingerprint. n = 32 (16 coherent / 16 shuffled).

Method validated before the LLM: `hankel_spectral._selftest` confirmed delay-embedding recovers
modes of a *short (90-sample) noisy* synthetic system at least as well as raw DMD (mean |freq err|
0.0003 vs 0.0010) — so a null below is about the **signal**, not a broken estimator.

## Results
q-sweep, multi-layer 5-fold CV AUROC (bar 0.70):

| q (delays) | AUROC | note |
|---|---|---|
| 1 | **0.562** | == v2 baseline (raw DMD) — sanity anchor reproduces exactly |
| 4 | 0.555 | |
| 8 | 0.586 | best point in the entire sweep — still far under 0.70 |
| **12** | **0.461** | **PRIMARY (verdict)** — below chance |
| 16 | 0.430 | |
| 24 | 0.359 | |

Per-layer `dominant_freq` AUROC at q=12 (transparency): L6 0.469, L12 0.570, L18 0.520 — all at chance.

## Verdict (pre-registered primary, binding)
**PRIMARY q=12 AUROC = 0.461 < 0.70 → K1 STILL FAILS.** And the signal **degrades monotonically with
more embedding** (0.586 → 0.461 → 0.430 → 0.359): delay-embedding does not recover a hidden mode
structure, it dilutes an already-absent one. This is not a noisy single point — it is a coherent
trend across the whole principled hyperparameter axis, and it is the **third** independent extraction
regime to fail the same gate across **two** models:

- v1 (distilgpt2 L4, crude single-layer DMD): 0.434
- v2 (pythia-410m, bigger + multi-layer, longer trajectory): 0.562
- v3 (pythia-410m, principled Hankel/Takens delay-DMD): 0.461 primary; 0.586 sweep-best

The shelving note's open question — *"was K1 killed by the premise or by a crude estimator?"* — is now
answered: **the premise.** The correct short-trajectory estimator (the one tool no prior version tried,
and the one most likely to rescue it) makes it no better, and tuned harder, worse.

## What is closed, precisely (no overclaim)
Closed: **the DMD/Koopman eigenvalue-spectrum of an LLM's residual-stream trajectory as a
cognitive-state signal.** It cannot separate even coherent-vs-shuffled text — the grossest proxy — so
it would not separate honest-vs-deceptive. The pre-registered chain stops at K1; **K2–K4 (the
obfuscation-robustness value proposition, GPU-gated) are not run** — there is no signal to make robust.

Not closed (out of scope): dynamical integrity signals *in general*. Trajectory **magnitude**
(path-length / "adversarial restlessness", 2604.28129) works in the literature but is not novel and was
never our bet. What died is the *spectral* (frequency-content) reading specifically.

## Honest limits
n=32 cannot prove a precise null; a tiny effect below this resolution is not excluded. But the binding
stop rule exists precisely to prevent "one more rework" forever: three extraction regimes, two models,
a wrong-direction q-trend, and a method self-validated on synthetic short trajectories together make
this a robust **practical** kill of the feature family. We close it rather than flog it.

## Net for the program
A clean closed-negative that *strengthens* the arc: the oscillation/resonance work stands as the
honest methods/reproduction result it already is (`PAPER_oscillation_memory`, `FRONTIER_due_diligence`),
and its one speculative product extension is now falsified by its own pre-registered gate — with the
GPU budget it would have consumed saved. The frontier the machinery opens is not the trajectory
*spectrum*; if a dynamical integrity signal exists, it is not here, and not at this feature.
