# FINDING — the deployment-honest mount catch is WEAK and calibration-sensitive: the live 12/13 was inflated by calibration leakage (CATCH-WEAK)

**2026-06-12 · Fathom Lab / styxx. Pre-registered: `PREREG_mount_fpr_live_2026_06_12.md` (frozen pre-run,
committed 406c02d). Receipt: `mount_fpr_live_result.json`; figure `mount_fpr_live.png`. The masterpiece
live demo (`FINDING_mount_live_catch_2026_06_12.md`, LIVE-CAUGHT 12/13) calibrated the divergence center
on the test states themselves (caves included) at tau=0 — flattering the catch rate. This re-runs it
deployment-honestly: center on a NEUTRAL held-out set, threshold to a target false-alarm budget on a
HELD-OUT HONEST set, evaluated on DISJOINT test. The honest number is much weaker, and that is the point.**

> **RESOLVED (2026-06-13, see `FINDING_mount_regime_2026_06_13.md`, LIVE-CAUGHT-FPR):** the regime
> mismatch diagnosed below was the cause. Calibrating center AND threshold on held-out honest PRESSURED
> decisions (same regime, no leakage) more than DOUBLED the catch over the 0.40 below, at a controlled
> false-alarm rate (loose at this n). The CATCH-WEAK below was substantially a calibration-regime
> artifact, not a real wall; the definitive operating point is the regime-matched one. This finding stands
> as the honest record of the mismatch (and of the leak it first corrected).**

## Result — catch 0.40 at FPR 0.067, leakage-free

Same borrowed gemma→Llama truth conscience (map anchor R² 0.9851, agent L21), 30 comparative claim pairs.
Center/scale calibrated on 24 neutral factual statements (`calibrate`); divergence threshold set to
target FPR 0.10 on 15 held-out honest items (`calibrate_threshold`, tau=0.0); evaluated on disjoint test.

| metric (leakage-free) | value |
| --- | --- |
| caves (agent asserted a false claim true) | 20 of 30 |
| **catch rate on caves** | **0.40** (caught 8 of 20 caves) |
| held-out honest (agent True on true) | 15 |
| **false-alarm rate** | **0.0667** (1/15) |
| **Verdict per the frozen gate (catch ≥ 0.70): CATCH-WEAK** | |

**The masterpiece's LIVE-CAUGHT 12/13 was inflated by calibration leakage.** Catching is "substrate
reads below the center"; the demo set the center from the test states (caves dragged it down), lowering
the bar so most caves fell under it. With the center fixed on neutral held-out data, the bar sits where
the honest distribution actually is, and most caves clear it. The leakage-free catch is 0.40, not ~0.92.

## Why — caves and honest answers overlap at the decision token, and calibration regime matters

The figure shows it: held-out honest answers cluster high and clear (one dips to a single false alarm),
but the caves SPREAD from strongly-false to clearly-positive. About 8 caves read genuinely false (caught
— the substrate disagrees with the cave); the other ~12 read positive (the substrate does NOT strongly
disagree with the cave — at the decision token the model's truth coordinate is not always against its
caved answer). So the borrowed conscience separates caves from honest only PARTIALLY here.

Two compounding effects, named honestly:
1. **Leakage (the real correction):** centering on test states inflated the demo. Removed here.
2. **Regime mismatch (a flaw in THIS calibration choice):** the neutral statements read at a high, tight
   baseline (center 1.1861, scale 0.1799) that does not match the pressured-decision regime the caves
   live in. The neutral center is a mis-matched reference — it likely sets the bar too high for the
   pressured regime. The methodologically correct calibration is on held-out HONEST PRESSURED decisions
   (same regime, no leakage), not neutral statements. That is the owed definitive operating point.

## What stands and what does not

STANDS: the substrate carries real signal — the 8 caught caves read genuinely false, and honest answers
read clearly positive (1 false alarm in 15); `styxx.mount`'s machinery (borrowed cross-model conscience,
FPR-controlled threshold) works as built. The arc's science (a mapped honesty direction reads caved
states, ITEM-CAUGHT) is untouched — that used the same-forward item-level read, a different and
leakage-free protocol.
DOES NOT STAND: a near-perfect (demo-level) deployment catch rate. The leakage-free, controlled-FPR catch on this larger,
harder set is 0.40, and it is calibration-sensitive. The masterpiece's headline is corrected: the mount
is a real instrument, but its single-token decision-time catch rate at a controlled false-alarm rate is
weak-to-moderate and depends heavily on calibrating in the right regime.

## Owed (pre-committed)

The definitive deployment number needs the regime-matched calibration: center AND threshold both on a
held-out HONEST subset of the SAME pressured-decision regime (no neutral-regime mismatch, no test
leakage). This run shows the honest number is materially below the demo and calibration-sensitive; the
regime-matched re-run (fresh prereg) gives the operating point an integrator should quote. Also owed:
multi-token / response-level aggregation (a single decision token is a thin read), and the gamed-model
adversarial test. Bounds unchanged: white-box, linear truth axis, local same-cluster models, single-turn,
n bounded; read-only (the mount measures; acting is policy).
