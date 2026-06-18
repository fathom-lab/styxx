# RESULT — K1-CS: inconclusive (the positive control failed too); the cheap proxy cannot settle it

**2026-06-18.** Executed `PREREG_cross_sample_2026_06_18.md` (frozen 4d4b5b7, before code). Outcome:
**K1-CS fails the 0.70 bar at 0.424 — but the semantic-entropy positive control also fails at 0.403.**
A negative result with a dead positive control is not a kill; it is an underpowered/invalid test. Per
the prereg's own scoping ("closed ON CHEAP PROXIES") and the binding stop rule (do not flog), I report
it honestly and stop, without claiming the cross-sample object is dead.

## What ran
pythia-410m (GPU). 24 prompts (12 answerable/stable, 12 underspecified/improvise), K=6 resamples each,
temperature 0.8, 32 new tokens. Per generation: deep-layer (L18) non-stationarity signature. Per
prompt: cross-sample dynamical consensus (per-feature std + mean pairwise signature distance across
the K samples) vs a semantic-entropy baseline (mean pairwise cosine distance of the K last-token reps).
5-fold CV AUROC.

## Results
| signal | AUROC | |
|---|---|---|
| semantic-entropy baseline (positive control) | **0.403** | should separate answerable vs improvise — it does NOT |
| cross-sample dynamical consensus | **0.424** | bar 0.70 |
| combined | 0.375 | |
| novelty delta (dyn − sem) | +0.021 | confound bar +0.05 |

## Why this is INCONCLUSIVE, not a kill (honest)
The semantic-entropy baseline is a *positive control*: improvise prompts must produce more divergent
continuations than stable factual ones, and semantic entropy is the established signal for that. It
landed at **0.403 — below chance.** When the positive control fails, the apparatus did not detect a
contrast it should trivially detect, so the dynamical "fail" carries no information about the dynamical
object. The likely cause is power: **n=24 prompts under 5-fold CV** is a near-chance estimator, and a
410m model's continuations may be weakly coherent for both sets. The last-token rep is also a thin
semantic proxy.

## Verdict (binding, honestly scoped)
- **K1-CS on the cheap proxy: FAILS (0.424), but the test is INVALID (positive control 0.403).** Per
  the pre-registered stop rule I do **not** flog the cheap proxy — no patch-the-control-and-rerun this
  session, which would be goalpost-moving. The cheap cross-sample route is closed as *uninformative*.
- This does **not** close the cross-sample object. A decisive test needs what the prereg already
  flagged: the **real honest/confabulated dataset** (answers-agree/grounding-differs), **n ≥ ~100**, a
  **validated positive control** (semantic entropy must separate before the dynamical claim is even
  testable), on a model whose generations are coherent. That is a GPU/dataset experiment, deliberately
  not attempted as a cheap proxy.

## Net across the session's three pre-registered swings
1. **Spectral probe — CLEAN KILL.** Valid test (reproduced its baseline exactly), died across 3
   extraction regimes × 2 models. Closed.
2. **Single-trajectory switching — ALIVE, sub-threshold.** 0.672 on a valid coherence gate, +0.21 over
   the dead spectrum, mechanism self-validated, coherent depth/feature profile. The real signal.
3. **Cross-sample cheap proxy — INCONCLUSIVE.** Underpowered, positive control failed; cannot settle
   the object. Decisive test deferred to the real dataset.

The one genuine, honestly-measured "something" is #2: the integrity-relevant dynamical quantity is the
**non-stationarity / operator switching**, not the trajectory spectrum. It is real but modest on a
single trajectory; whether ensembling lifts it past the bar is an open, properly-powered question — not
something a cheap proxy could answer, and we did not pretend otherwise.
