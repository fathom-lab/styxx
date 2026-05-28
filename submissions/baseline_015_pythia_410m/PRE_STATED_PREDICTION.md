# Pre-stated prediction — Baseline-015 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-015.

## What is being tested

The [scaling-curve FINDING](../../papers/agent-self-audit/FINDING_lm_likelihood_scaling_curve_2026_05_27.md) established monotonic inverse scaling within the gpt2 family. Baseline-014 showed Pythia-160M (different family, similar size) also gives strong D2-axis signal — even stronger than gpt2-124M on the D2 confound bars.

**Open question:** does Pythia ALSO inverse-scale? If yes, the "smaller is better" property is a general law for LM-typicality detection. If no, the inverse-scaling is gpt2-specific.

Baseline-015 = **EleutherAI/pythia-410m** (410M params, ~2.6× the size of Pythia-160M). If Pythia also inverse-scales: D1, D2, and D2-deltas should all DROP from Baseline-014's values. If Pythia scales normally: D2-deltas should stay similar or grow.

## Honest analytical prior

Pythia is trained on The Pile (Wikipedia-heavy + ArXiv + books + code) at all sizes. The data distribution stays constant; only model size varies. Three hypotheses:

1. **Inverse-scaling is general law** (~60%): Pythia degrades like gpt2 did. D1 ~ 0.72, D2 ~ 0.86, D2-length-delta ~ 0.05 (below threshold), D2-cap-delta ~ 0.07. Goes back to 1/4 or 2/4.
2. **Inverse-scaling is gpt2-specific** (~25%): Pythia keeps strong signal or even improves. D1 ~ 0.82, D2 ~ 0.93+, D2-deltas STAY above 0.10. Could maintain 2/4 with D2-axis still clean, or possibly improve D1 to clear D3.
3. **Mixed**: Pythia degrades less than gpt2 — partial inverse-scaling. D2 stays just above threshold; D1 drops. Modal 2/4 with weaker D2-deltas.

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear all 4 bars** (real PASS) | **~6%** |
| **3/4 — pass D1+D2+D4 fail D3** | ~10% |
| **3/4 — pass D1+D2+D3-on-D2 + ...** (heterogeneous) | ~5% |
| **2/4 with D2 still clearing confound deltas** (Pythia maintains the D2-axis win even at 410M) | ~25% |
| **2/4 with D2 confound deltas now below threshold** (inverse scaling kicks in) | ~28% |
| **1/4 — D2 AUC only** | ~18% |
| **0/4** | ~8% |

### Specific AUC predictions

| metric | Baseline-014 (160M) | Baseline-015 predicted range |
|---|---|---|
| D1 AUC | 0.799 | 0.65–0.85 |
| D2 AUC | 0.920 | 0.80–0.93 |
| D2 − length-oracle | 0.116 | 0.00 to +0.15 (modal: drops below 0.10) |
| D2 − capratio-oracle abs | 0.128 | 0.00 to +0.18 |
| D1 − length-oracle | 0.010 | −0.10 to +0.06 |
| D1 − capratio-oracle abs | 0.096 | 0.00 to +0.13 |

Modal outcome: **2/4 with D2-axis-clean degrading.** If inverse-scaling is general, expect D2-length-delta to drop from 0.116 → 0.04-0.08 range, falling below threshold.

**Direction:** misc > truth log-prob (high confidence — four consecutive confirmations).

## Why this bet matters

- **PASS (~6%):** would mean Pythia-410M is the sweet spot. First real PASS.
- **3/4 (~15% combined):** would mean Pythia improves with size (opposite of gpt2). Family-specificity of inverse scaling — major scientific finding.
- **2/4 D2-clean (~25%):** Pythia at 410M maintains the D2 advantage but D1 doesn't improve. Inverse scaling is partial in Pythia. Modal hypothesis: D2 partition consistently detectable across families/sizes; D1 partition fundamentally harder.
- **2/4 D2-degraded (~28%):** Pythia inverse-scales like gpt2. The "smaller is better" property generalizes. Strongest argument for switching to even smaller LMs (Pythia-70m, gpt2-distil) on future submissions.
- **1/4 (~18%):** signal degrades faster than predicted. Possible if Pythia's training data is uniquely bad at this task at larger sizes.

## Not re-running, not re-tuning

- Model: **EleutherAI/pythia-410m** (HuggingFace standard checkpoint).
- Algorithm: identical to Baselines 011-014.
- Run once.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-015.
