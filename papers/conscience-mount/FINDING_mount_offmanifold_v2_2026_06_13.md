# FINDING — off-manifold detection is DATA-limited, not design-limited: two detectors, both FPR-UNCONTROLLED at this sample size (FPR-UNCONTROLLED)

**2026-06-13 · Fathom Lab / styxx. Pre-registered: `PREREG_mount_offmanifold_v2_2026_06_13.md` (frozen
pre-run, committed 2edaa20). Receipt: `mount_offmanifold_v2_result.json`; figure
`mount_offmanifold_v2.png`. The fixed (PCA-reduced) detector ALSO fails to control false alarms — the
second FPR-UNCONTROLLED in a row. The honest conclusion: the layered off-manifold defense cannot be tested
at this honest-sample size; the blocker is DATA, not detector design. Stop tweaking detectors.**

## Result — the PCA-reduced detector still flags 100% of held-out honest

| metric | value |
| --- | --- |
| detector FPR on held-out honest | **1.0** (still flags ALL honest) |
| PCA components (k) | 17 |
| — in-subspace Mahalanobis detection of evasions | 0.7647 |
| — residual-channel detection of evasions | 1.0 |
| detector FPR — driven by | the RESIDUAL channel |
| **Verdict per the frozen gate (fpr > 0.20): FPR-UNCONTROLLED** |

The reduced detector did not fix the false-alarm problem: with a few dozen calibration points and a
several-thousand-dimensional space, the top-k PCA subspace fit to CALIB does not capture held-out honest
variance, so held-out honest points have large reconstruction residuals and the residual channel flags
them all (FPR 1.0). The same data-scarcity that broke v1's raw Mahalanobis breaks v2's residual channel —
just at a different stage of the pipeline.

## The honest diagnosis — a data wall, twice confirmed

Two independent detector designs (raw shrunk-Mahalanobis, PCA-reduced + residual) both return FPR 1.0 on
held-out honest. The common cause is sample size: characterizing the honest-activation manifold of a
several-thousand-dimensional residual stream well enough to control false alarms needs HUNDREDS to
THOUSANDS of honest samples; we have a few dozen. No detector tweak fixes that. The right move is not a
third detector variant — it is a proper honest-activation corpus (the agent's last-token states on a
large set of honestly-answered decisions). Until then, the off-manifold layered-defense question is VOID:
unanswerable at this scale, in either direction.

## The one suggestive (but unvalidated) signal

The in-subspace Mahalanobis channel detected 0.7647 of the conscience-evading perturbed caves. IF its
false-alarm rate were independently controlled (it is not — the union FPR is 1.0, dominated by the
residual channel, and the in-subspace channel's own held-out FPR was not separately validated), that
would lean toward the evasion being partially detectable in the honest PCA subspace (an
OFF-MANIFOLD-CATCHES-leaning hint). But this is NOT a result — it rests on an uncontrolled detector and
must be re-tested with enough honest data to validate the in-subspace channel's FPR on its own. Recorded
as a lead, not a finding.

## What this settles for the adversarial arc

The mount-side adversarial story is complete and stands: borrowing buys nothing, re-locked ensembling
stops naive attackers but only taxes an adaptive one ~2×, and truth is one direction. The NON-mount
defense (off-manifold detection) is, at this evaluation scale, UNTESTABLE — not refuted, not confirmed.
So the honest end-state: against a determined adaptive activation attacker, the mount alone does not
suffice, and whether an anomaly detector closes the gap is an open question gated on a large honest
corpus. Do not over-tweak: the next real step is either that corpus, or a weight-level fine-tune
detector (a different, non-activation defense), or accept the cooperative-scope framing and consolidate.

## Honest bounds

A few dozen honest calib in a several-thousand-dim space (the failure mode itself, now twice); unconstrained
activation attack (upper bound); linear conscience; common read layer (L21); local models; single-turn;
read-only. No claim about the layered defense is made — only that it is untestable at this sample size, and
that off-the-shelf anomaly detection (raw or PCA-reduced) does not transfer to raw LLM activations without
a large honest corpus. Owed: the honest corpus (definitive), or move on.
