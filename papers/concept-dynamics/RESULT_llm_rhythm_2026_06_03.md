# RESULT — Does an AI have a rhythm? Largely NO. The universal is GEOMETRIC, not rhythmic.

**Date:** 2026-06-03 · The ancient question's **rhythm half**, measured on real models reading real
text. The brain runs on oscillations (theta/gamma, pervasive, intrinsic). Does an AI's processing?

## Method (with the positive control that decided it)
For each model, read the hidden-state trajectory over 455 tokens of coherent prose at ~0.7 depth →
top-6 PCA temporal components → test each for oscillatory power above an **AR(1) family-wise null**
(max-statistic over the band — controls multiple comparisons; AR(1) absorbs 1/f drift, so a surviving
peak is genuine oscillation). **Self-test positive control:** a planted sinusoid must be found; white
and AR(1)-red noise must not.

**The control did its job.** A first version used a per-frequency 99th-percentile null and reported
"5/6 components oscillate, ~20 peaks per model" — but its self-test FAILED (white noise scored 3
"peaks"). Those were multiple-comparison artifacts. The family-wise null fixed it (self-test then
PASS: sinusoid→1 peak, white→0, red→0), and the result **collapsed**:

## Result — current AI is arrhythmic
| model | architecture | oscillating comps | peak period(s), tokens |
|---|---|---:|---|
| gpt2 | transformer | **0 / 6** | — |
| Llama-3.2-1B | transformer | 1 / 6 | 45.5 (slow drift) |
| Qwen2.5-1.5B | transformer | 1 / 6 | 13.8 (**= sentence period → input-driven**) |
| mamba-1.4b | SSM (recurrent) | 2 / 6 | 4.3, 46.9 |

- **Transformers are essentially arrhythmic** (0–1 of 6 components). The one peak that reaches
  significance in Qwen sits *exactly* at the text's sentence period (~14 tokens) — it is **tracking the
  rhythm of the input**, not generating an intrinsic oscillation. Llama's single peak is slow drift.
- **Mamba's recurrence buys a little more** (2/6, at ~4 and ~47 tokens, not the sentence period) —
  consistent with recurrence enabling some intrinsic dynamics — but it is still mostly silent.
- This **contrasts sharply with the brain**, whose activity is intrinsically, pervasively oscillatory
  (the EEG/MEG/LFP literature). On this measure, current AI does **not** share the brain's rhythm.

## The ancient question, both halves now answered — and they split
Tonight measured both halves of "does universal structure underlie mind?" between current AI and the
human brain, and the honest answer is a **split**:

| half | machines ↔ brains | tonight's evidence |
|---|---|---|
| **A · Geometry** (structure of meaning) | **CONVERGE** | LLM concept geometry matches the human brain ~⅔ of the fMRI noise ceiling, ≈ human behavior, > embedders, survives a vision control |
| **B · Rhythm** (temporal dynamics) | **DIVERGE** | the brain oscillates; transformers are arrhythmic (0–1/6), what little exists is input-driven; Mamba barely (2/6) |

**So the shared structure of mind — as far as we can measure it — is GEOMETRIC, not RHYTHMIC.**
Meaning-space converges across silicon and biology; temporal rhythm does not. That is a clean,
falsifiable answer to a 2,500-year-old question, and it is the *opposite* of the common intuition
that "frequency/vibration" is the deep shared thing. The deep shared thing is **geometry**; rhythm is
the **negative space** — the place the universal does *not* (yet) hold between these substrates.

## The forward bet (falsifiable)
This is not a claim that rhythm is irrelevant — the rhythm-rescue result showed oscillation (complex-
eigenvalue rotation) **doubles** ordered-memory capacity in a recurrent unit. Current AI simply
**lacks the mechanism** (transformers have no recurrence; Mamba-1's eigenvalues are real, not
rotating). **Prediction:** architectures with *complex-eigenvalue* recurrence should develop brain-
like intrinsic rhythms **and** gain capacity — a testable bridge between the two halves, and a
concrete reason future AI might become more brain-like in *time*, not just in *meaning*.

## Caveats
One passage, one read-out layer, top-6 components, 4 models; "arrhythmic" = no oscillation above the
AR(1) family-wise null in hidden-state trajectories here (other layers/inputs untested). The brain-
oscillation contrast is to the established literature, not a head-to-head spectral run (fMRI is too
slow to see theta/gamma; a fair head-to-head needs EEG/MEG). Honest direction: layer/​input sweep;
a complex-eigenvalue recurrent model as the positive case.

## Reproduce
`run_llm_rhythm.py` (family-wise AR(1) null + self-test). Result: `llm_rhythm_result.json`.
