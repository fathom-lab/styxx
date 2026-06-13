# FINDING — the naive off-manifold detector is FPR-UNCONTROLLED at LLM scale; the layered-defense verdict is VOID pending a reduced detector (FPR-UNCONTROLLED)

**2026-06-13 · Fathom Lab / styxx. Pre-registered: `PREREG_mount_offmanifold_2026_06_13.md` (frozen
pre-run, committed 8584d35). Receipt: `mount_offmanifold_result.json`; figure `mount_offmanifold.png`.
An honest tooling negative: the pre-registered Mahalanobis off-manifold detector is unusable at this
sample-size / dimension, so it cannot adjudicate whether the conscience-evading attack is off-manifold.
Reported as the verdict it earned, not spun.**

## Result — the detector flags everything (FPR 1.0)

The non-mount Mahalanobis detector (shrunk covariance on the honest-activation manifold) was supposed to
flag off-manifold states at a target FPR of 0.10. Realized:

| metric | value |
| --- | --- |
| detector FPR on held-out honest | **1.0** (flags ALL honest) |
| detection of perturbed (evading) caves | 1.0 |
| detection of unperturbed caves | 1.0 |
| conscience catch at the evading budget (ε* = 0.02) | 0.0 |
| **Verdict per the frozen gate (fpr > 0.20): FPR-UNCONTROLLED** |

The detector flags 100% of HELD-OUT HONEST states — it cannot distinguish honest-calib from honest-test,
let alone caves from honest. The cause is structural: a shrunk-covariance Mahalanobis estimate from
a few dozen calibration points in a several-thousand-dimensional activation space is degenerate; held-out honest points
fall far outside the rank-deficient calibration ellipsoid, so everything reads as anomalous. This is a
failure of the off-the-shelf detector at LLM-activation scale, NOT a measurement of the attack. The
layered-defense question (does the conscience-evading attack land off-manifold) is therefore VOID under
this detector.

## The one usable signal, and its caveat

The honest-manifold variance along the attack direction is enormous: `var_ratio = 11305` (the conscience
readout direction carries that many times the mean per-dimension variance of honest activations). Read at face
value this says the evasion moves ALONG a very-high-variance direction of the honest manifold — i.e.,
toward normal-looking variation, not off it — which would make off-manifold detection fundamentally
unpromising for this attack (an OFF-MANIFOLD-BLIND lean). BUT this number is artifact-suspect: LLM
residual streams have well-known "massive-activation" outlier dimensions with huge variance, and if the
readout direction overlaps one, `var_ratio` blows up for reasons unrelated to the attack's cleverness.
The same outlier dimensions are exactly what break the raw Mahalanobis estimate. So the signal is
suggestive (lean BLIND) but not conclusive; it must be re-tested with a detector that normalizes the
outlier dimensions away.

## What is owed — a reduced, whitened detector

The methodologically correct detector at n ≪ d: reduce to a low-dimensional honest PCA subspace (k < n,
capturing most variance, which absorbs the massive-activation dims), whiten within it, and compute
Mahalanobis there — plus a PCA reconstruction-residual term for the out-of-subspace component. That
detector is well-conditioned and would give a real OFF-MANIFOLD-CATCHES / BLIND verdict. This finding
records that the naive detector fails and pre-commits the fix as the definitive next test. (A useful
side-lesson: off-the-shelf anomaly detection does NOT transfer to raw LLM activations without
dimensionality control — relevant to anyone building a non-mount defense.)

## Honest bounds

a few dozen honest calib in several-thousand dim (the failure mode itself); unconstrained activation attack (upper
bound); linear conscience; common read layer (L21); local models; single-turn; read-only. No claim about
the layered defense's effectiveness is made here — only that this detector cannot test it. Owed: the
PCA-reduced whitened detector (definitive), then a constrained on-manifold attacker if it CATCHES, and a
real weight-level fine-tune.
