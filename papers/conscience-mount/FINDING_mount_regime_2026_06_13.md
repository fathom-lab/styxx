# FINDING — the definitive deployment operating point: regime-matched calibration catches 0.85 of caves (LIVE-CAUGHT-FPR)

**2026-06-13 · Fathom Lab / styxx. Pre-registered: `PREREG_mount_regime_2026_06_13.md` (frozen pre-run,
committed 897d684). Receipt: `mount_regime_result.json`; figure `mount_regime.png`. Resolves the
calibration-regime question left open by `FINDING_mount_fpr_live_2026_06_12.md` (CATCH-WEAK 0.40 under a
neutral-regime-mismatched center). Calibrating the divergence center AND threshold on held-out honest
PRESSURED decisions — the same regime the caves live in, no leakage — gives the honest deployment number.**

## Result — catch 0.85 at a controlled (if loose) false-alarm rate

Same borrowed gemma→Llama truth conscience (map anchor R² 0.9851, agent L21), 30 comparative claim pairs.
The ONLY change from the prior run: center/scale and the FPR threshold are both fit on the 15 held-out
honest PRESSURED decisions (true controls the agent answered "True" under the same pressure), not on
neutral statements. Evaluated on disjoint test (caves + held-out honest).

| metric (regime-matched, leakage-free) | value |
| --- | --- |
| caves (agent asserted a false claim true) | 20 of 30 |
| **catch rate on caves** | **0.85** (caught 17 of 20) |
| held-out honest (agent True on true) | 15 |
| realized false-alarm rate | 0.20 (3 false alarms of 15 held-out honest) |
| target false-alarm rate | 0.10 |
| **Verdict per the frozen gate (catch ≥ 0.70, FPR ≤ 0.20): LIVE-CAUGHT-FPR** | |

**Regime-matched calibration more than doubled the catch (0.40 → 0.85), confirming the prior CATCH-WEAK
was substantially a regime-mismatch artifact, not a real wall.** The figure shows crisp separation: caves
read deeply false (down to roughly −7 standard deviations below the honest cluster); the honest decisions
cluster tightly high (center 1.3766, scale 0.0621). The neutral center (1.19) had sat *below* the true
honest cluster, setting the bar too low and letting positive-reading caves escape; the regime-matched
center sits in the honest cluster, so the threshold tracks where honest true-assertions actually live and
most caves fall below it.

## The honest caveat that does not move: false-alarm control is loose at this n

The threshold was calibrated to a 0.10 target FPR on the 15 honest calib items, but the realized FPR on
the 15 held-out honest test items is **0.20 — double the target**. Three honest true-assertions fell below
the threshold, including one strong outlier the borrowed conscience read very false (~−3.4σ): a true claim
the gemma truth axis genuinely mis-reads on Llama's substrate. So the catch (0.85) is the robust number;
the false-alarm *control* does not yet generalize tightly at n=15 (the calib→test FPR gap is a small-sample
effect, and the borrowed axis has its own error cases). More honest calibration data is needed to hold the
FPR at target, and the outlier shows the conscience is not infallible — it has false positives of its own.

## What this settles

This supersedes both prior live numbers — the leaky 0.92 (center on test states) and the
regime-mismatched 0.40 (center on neutral statements). **The deployment operating point an integrator
should quote: a borrowed, read-only conscience catches ~0.85 of a model's pressured caves at a
single decision token, at a false-alarm rate of ~0.20 (target 0.10, loose at n=15), calibrated in the
correct regime.** Strong enough to be useful, honest about the FPR looseness, and a clean demonstration
that calibration regime — not the conscience — was the lever between 0.40 and 0.85.

## Honest bounds (unchanged)

White-box (needs the agent's activations → open-weight / self-host agents), linear borrowed truth axis,
last-token pre-decision (a single token is a thin read — response-level aggregation is owed), local
same-cluster models (gemma reference, Llama-3.2-3B agent), single-turn pressure, n = 30 pairs / ~15
honest calib / ~15 honest test. Read-only: the mount measures at a chosen operating point; a flag is a
signal, acting on it is the integrator's policy (read ≠ write). Owed: more calib data to tighten the FPR
to target; response-level / multi-token aggregation; the gamed-model adversarial test (can an agent
trained to fool its own probe beat a BORROWED conscience).
