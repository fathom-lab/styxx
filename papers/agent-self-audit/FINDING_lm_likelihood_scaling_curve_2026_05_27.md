# Finding · Inverse LM scaling on the dark-core detection task — within the gpt2 family, smaller is better

**Date:** 2026-05-27 · **Status:** three pre-registered data points; monotonic curve at n=3. Extends the [detection-arc FINDING](FINDING_detection_arc_v3_bars_2026_05_27.md) with a finer-grained scaling-curve result. **Tenth in-session falsification** (Baseline-012 degradation) + the eleventh adjacent finding (monotonic curve confirmation via Baseline-013).

> **Outcome.** Mean per-token log-probability of response tokens — under the prefix `"Question: {q}\nAnswer: "` — *decreases* in discriminative power as the gpt2 model size grows. Across 124M → 355M → 774M parameters, every detection metric drops monotonically: D1 AUC (0.811 → 0.758 → 0.682), D2 AUC (0.897 → 0.844 → 0.782), D2 length-delta (+0.093 → +0.040 → −0.022). The "obviously bigger LM = sharper signal" intuition is decisively falsified at n=3 within the gpt2 family on this benchmark.

## The three data points

| size | params | D1 AUC | D2 AUC | D1−length | D2−length | D1−cap | D2−cap | bars | submission |
|---|---|---|---|---|---|---|---|---|---|
| gpt2 | 124M | **0.811** | **0.897** | +0.022 | **+0.093** | **+0.108** | **+0.105** | **3/4** | Baseline-011 |
| gpt2-medium | 355M | 0.758 | 0.844 | −0.032 | +0.040 | +0.054 | +0.052 | 2/4 | Baseline-013 |
| gpt2-large | 774M | 0.682 | 0.782 | −0.107 | −0.022 | −0.021 | −0.010 | 1/4 | Baseline-012 |

Per-step deltas (Δ per scaling step):

| metric | 124M → 355M | 355M → 774M | linear? |
|---|---|---|---|
| D1 AUC | −0.053 | −0.076 | roughly linear, slight acceleration |
| D2 AUC | −0.053 | −0.062 | linear |
| D1−length | −0.054 | −0.075 | roughly linear |
| D2−length | −0.053 | −0.062 | linear |
| D1−cap | −0.054 | −0.075 | roughly linear |
| D2−cap | −0.053 | −0.062 | linear |

The decline is approximately **−0.05 to −0.07 AUC per ~3× parameter-count step** in every detection metric. The curve is *clean* — no inversion, no plateau, no sweet spot.

## What "smaller is better" means here

The mechanism hypothesis that best fits the data:

**A small LM is more strongly surprised by short canonical answers like "Paris", "Newton", "1789" because it has weaker priors over the long tail of single-word completions.** A 124M-parameter model assigns ~−5 log-prob to truth tokens (very surprised). A 774M-parameter model has seen more text and assigns ~−3.5 (much less surprised — closer to misconception scores of ~−2.3).

Misconception responses are full sentences in common-belief framings ("Walt Disney's body is interred at Forest Lawn Memorial Park"). All three sizes assign similar log-prob to these (~−2 to −2.5). The bigger model didn't *get more confident* about misconceptions; it stopped being surprised by truth. **The discrimination gap closed from the truth side, not the misconception side.**

Verification: mean scores across the three baselines:

| size | mean misconception | mean truth | mean folklore | gap (misc − truth) |
|---|---|---|---|---|
| 124M | −2.40 | −4.70 | −1.90 | **+2.30** |
| 355M | −2.33 | −4.05 | −1.94 | +1.72 |
| 774M | −2.23 | −3.52 | −1.83 | +1.29 |

Truth mean log-prob climbs sharply with size: −4.70 → −4.05 → −3.52 (+1.18 over the scaling steps). Misconception mean log-prob barely moves: −2.40 → −2.33 → −2.23 (+0.17). **Truth gets less surprising; misconception stays similarly likely.** The gap narrows monotonically.

## Why this matters

Three interlocking observations:

1. **The "scale solves everything" intuition is wrong here.** LM-typicality detection is a non-trivial task where scaling within a family actively destroys signal. This is a publishable inverse-scaling result on a benchmark with regression-tested bars and pre-registered predictions.

2. **The path forward is narrower than scaling.** If gpt2-large didn't help, gpt2-xl probably won't either (the curve predicts D1 ≈ 0.60, D2 ≈ 0.72 at 1.5B — both barely above chance after confound adjustment). The frontier isn't "use a bigger LM"; it's either "switch model family" or "switch the signal definition."

3. **Direction confirmed at every size.** All three baselines confirmed the misc > truth direction. This is unusual — direction-of-effect predictions on this domain have otherwise been the systematically-worst calibration band (see [confound audit FINDING](FINDING_confound_audit_2026_05_27.md) and [detection arc FINDING](FINDING_detection_arc_v3_bars_2026_05_27.md)). LM-typicality is the first axis where direction is robust across multiple instances.

## Open follow-ups

The empirical question now well-defined enough to test:

1. **Family-specificity** — does Pythia-160M (similar size, different training data + architecture) reproduce gpt2-124M's 3/4 result, or is the WebText pretraining specifically what makes gpt2 work? Baseline-014 candidate.
2. **Asymptotic behavior** — does gpt2-xl (1.5B) continue the linear decline, or does it plateau? Cheap follow-up; would confirm whether the curve is truly monotonic at all scales.
3. **Instruct-tuning effect** — does an instruct-tuned model (Phi-3-mini-instruct, Qwen2.5-3B-Instruct) flip the direction or just attenuate magnitude? Instruct models are RLHF-tuned to refuse misconceptions; their log-prob over misconception responses should be lower than base-model log-prob.
4. **Domain transfer** — same method on TruthfulQA-Generation or FreshQA. Does inverse scaling replicate on other misconception benchmarks?

The first three are testable now without operator-territory resources. Each is a single Baseline submission with the same template.

## Calibration on this scaling-curve experiment

Pre-stated calibration record:

| baseline | predictions | inside-range count | outcome-band probability | call |
|---|---|---|---|---|
| Baseline-011 | 6 ranges + 6 bands | 0/6 (all underpredicted) + 15% band | 15% | band-correct, magnitudes too low |
| Baseline-012 | 6 ranges + 6 bands | 0/6 (all overpredicted) + 8% band | 8% | band-correct, magnitudes too high |
| Baseline-013 | 4 ranges + 4 bands | **4/4** (all inside) + 33% combined band | 30% | well-calibrated |

The progression is informative: I had no anchor data when I predicted Baseline-011, was too optimistic about scaling for Baseline-012, but with 011/012 bracketing the prediction Baseline-013 was well-calibrated. **Pre-stated ranges calibrate better when nearby data points anchor them.** General research lesson, not specific to this benchmark.

## Reproducibility

| artifact | commit | path |
|---|---|---|
| Baseline-011 prereg + result | `879e4ab`, `0bb6178` | `submissions/baseline_011_lm_likelihood/` |
| Baseline-012 prereg + result | `bac1a5b`, `ac03f45` | `submissions/baseline_012_lm_likelihood_large/` |
| Baseline-013 prereg + result | `c14c683`, `95e29ac` | `submissions/baseline_013_lm_likelihood_medium/` |
| this finding | this commit | `papers/agent-self-audit/FINDING_lm_likelihood_scaling_curve_2026_05_27.md` |

`git log --oneline submissions/baseline_01{1,2,3}* papers/agent-self-audit/FINDING_lm_likelihood*` shows the prediction-before-data ordering for all three baselines and this synthesis on origin.

## Frame for the paper

If a future revision of [PAPER_recursive_discipline_2026_05_27.md](../PAPER_recursive_discipline_2026_05_27.md) adds this scaling result, the framing is:

> "The pre-registration discipline produced an inverse-scaling result that runs counter to the dominant 'scaling solves everything' intuition in language modeling. Three pre-registered baselines (gpt2-124M, gpt2-medium, gpt2-large) at increasing parameter counts produced *monotonically degrading* detection scores on the dark-core misconception task — the discrimination gap between truth and folklore narrowed at every step because truth responses became less surprising to larger LMs while misconception responses stayed similarly likely. This is the eleventh structural finding in the 2026-05-27 arc, and the first one where direction-of-effect predictions held across all three pre-registered tests."
