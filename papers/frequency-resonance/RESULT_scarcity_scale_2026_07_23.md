# RESULT — mode-scarcity scaling: the clean win does NOT scale — 2026-07-23

Frozen by `PREREG_scarcity_scale_2026_07_23.md`. Receipt: `scarcity_scale_result.json`. Runner:
`run_scarcity_scale.py` (parallel scan, red-team `scan==seq`, `κ=0==STATIC` bit-for-bit). 3 seeds,
1500 steps, RTX 4070. **Frozen verdict: ABSTAIN** at the pre-registered starved D=4 — and the full
curve **falsifies the scaling hypothesis**.

## One line

On a harder, longer drifting task (L=192, 4 segments, periods [3,20]), the windowed-conv
adaptive-frequency detector — which won **+0.129** at D=4 on the easy task (L=96, periods [3,12]) — is
**worse than static at every budget**, and the deficit *grows* with the mode count. The clean
mode-scarcity win was confined to the easy regime; it **does not scale**.

## Results (drifting-period accuracy, seed-averaged)

| D | STATIC | RICH (conv) | ORACLE | RICH−STATIC | ORACLE−STATIC |
|---|---|---|---|---|---|
| 2 | 0.149 | 0.113 | 0.154 | −0.0355 | +0.005 |
| **4** (primary) | 0.198 | 0.155 | 0.266 | **−0.0425** | +0.0683 |
| 6 | 0.295 | 0.172 | 0.377 | −0.1231 | +0.081 |
| 8 | 0.354 | 0.225 | 0.502 | −0.1286 | +0.148 |
| 12 | 0.405 | 0.267 | 0.642 | −0.1376 | +0.237 |

Scarcity curve `RICH−STATIC` by D: `{2: −0.036, 4: −0.043, 6: −0.123, 8: −0.129, 12: −0.138}`.

## The two things this shows

1. **The frozen gate ABSTAINS.** At the pre-registered starved D=4, the positive control is
   sub-threshold on this harder task (`oracle−static = +0.068 < 0.10`), so by the frozen rule the test
   cannot license a verdict at D=4 — no claim is drawn there.
2. **But the full curve is decisive, and negative.** The oracle prize is real and *grows* with D
   (up to `+0.237` at D=12) — adaptation is worth a great deal on the harder task — yet the
   **learnable** windowed-conv detector captures *none* of it: it is negative at every budget and
   falls further behind as modes increase. Online frequency estimation over longer sequences with a
   wider, faster-drifting period range is *harder*, and the detector that won on the easy task actively
   hurts here.

## Reading (the honest bound)

The one clean adaptive-frequency win (D=4, easy task) was **fragile**: it does not survive a longer
sequence and a wider period range. Combined with the earlier non-monotone detector dose-response (a
deeper detector collapses) and the WEAK D=8 result, the adaptive-frequency story is now bounded on
every side: it works only in a *narrow, easy* regime (short sequences, narrow period band, starved
budget, simple detector), and it neither scales to harder drifting tasks nor buys a clean flagship at a
comfortable budget. The mechanism is real (the oracle proves the prize exists and grows) but the
learnable estimator cannot reach it outside the easy corner.

This is the deep-push probe doing its job: we scaled the one encouraging result and it broke, honestly,
under a frozen gate and a firing-at-high-D positive control. The durable positives of the arc stand
unaffected — the real-task oscillation ablation (oscillation causally load-bearing on seq-MNIST) and
the resonance profiler. Scope: toy drifting task, 3 seeds. This result and its receipt are
OATH-certified.
