# RESULT — Meaning, not pixels: the AI↔brain match survives a vision control (but is modest & shared)

**Date:** 2026-06-03 · The decisive control on `RESULT_ai_brain_2026_06_03.md`. Mitchell stimuli were
word + line-drawing, so the brain RDM carries visual structure. This isolates the semantic component
with a **vision model** (CLIP-image features over the real THINGS object images, 53 nouns) and a
**variance partition** of the brain RDM across AI(semantic) + vision + behavioral(VICE) + lexical.

## The question (frozen before computing the partition)
Does the LLM predict the human brain **beyond** a vision model + word-form? PASS iff
partial(AI, brain | lexical + vision) ≥ 0.10 and > 0, and the LLM's unique brain-variance beyond
vision+lexical > 0. (CLIP-image is a vision-**language** model carrying semantics → partialling it is
**conservative**; survival is a strong meaning result.)

## Result — PASS (pre-stated bar met), and honestly bounded
Partial correlation of the AI consensus geometry with the human brain, adding controls:

| controlling for… | partial(AI, brain) |
|---|---:|
| — (raw) | 0.168 |
| word-form (lexical) | 0.182 |
| **+ VISION (CLIP-image)** | **0.107** |
| + vision + human behavior (VICE) | 0.043 |

**AI→brain survives the vision control (0.182 → 0.107 ≥ 0.10): the match is genuinely (partly) about
meaning, not just visual form.** The pre-stated bar is met. But the fuller picture bounds it sharply:

| reference (partial | lexical) | value | reading |
|---|---:|---|
| **behavioral human (VICE) → brain** | **0.247** | the **strongest** single brain predictor |
| AI → brain (best single) | 0.237 | as strong as behavior, individually |
| vision (CLIP-image) → brain | 0.172 | **the brain RDM is substantially visual** |
| AI → vision | **0.543** | **the LLM's own geometry is itself half-visual** |

## Variance partition of the human brain RDM (R²)
Total explainable is capped by the fMRI noise ceiling (RSA ≤ ~0.56 → R² ≤ ~0.16–0.31); the full
model reaches **R² ≈ 0.081**, so these unique slivers are a few % of the *explainable* variance.

| unique contribution to brain R² | value |
|---|---:|
| behavioral-human beyond AI + vision + lexical | **+2.2%** |
| AI beyond vision + lexical | +1.1% |
| vision beyond AI + lexical | +0.7% |
| **AI beyond vision + behavioral + lexical** | **+0.2%** |
| vision beyond AI + behavioral + lexical | +0.0% |

## Honest verdict
- **DEFENDED:** the AI↔brain result is **not a pure visual artifact** — it survives removing a powerful
  vision-language model's geometry (a conservative control). There is a real non-visual semantic
  component to LLM↔brain alignment.
- **BOUNDED (the honest part):** it is **modest and largely SHARED.** The brain RDM is substantially
  visual; the LLM's geometry is itself ~half visual (0.54); and the LLM adds almost nothing UNIQUE to
  brain prediction once human behavior is included (unique 0.043 / +0.2% R²). **Human behavioral
  similarity is the strongest and most-unique predictor of the brain.** "AI matches the brain" means
  *AI captures the same broad semantic-visual core humans and vision models do* — not *AI uniquely
  reveals brain structure.*
- **Discipline:** the pre-stated bar PASSED, and I report the marginal size + the shared-not-unique
  decomposition + the brain's own visual content rather than trumpeting the pass. This is the third
  control in this arc (lexical, reliability, now vision) and each one shrank the claim to its true size.

## Caveats / next
- CLIP-image carries semantics (conservative for the "is it meaning" test, but it under-estimates the
  pure-visual share — a low-level vision model, e.g. early CNN features, would partition vision vs
  semantics more cleanly; torchvision not installed here).
- 53 nouns; one example image per concept; one fMRI dataset (noisy, ceiling ~0.4–0.56). The modern
  THINGS-fMRI (single-trial betas, many images/concept) + a low-level visual model are the next steps.

## Reproduce
`run_ai_brain_vision.py` (CLIP-image RDM from THINGS imgur images + partial cascade + variance
partition). Caches `brain/clip_image_emb.npz`. Result: `ai_brain_vision_result.json`.
