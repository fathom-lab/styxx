# THINGS-fMRI Tier 2 — within-category structure resolves at 823 concepts (powered)

**2026-06-30 · fathom-lab / styxx · reproduce: `stream_build_things720.py` → `tier2_analysis.py` (+ `tier2_vision.py`)**

Streamed the 26.6 GB THINGS-fMRI single-trial nifti archive **without landing it** (disk-safe), built per-concept
patterns over **823 concepts** (720 main + test) in two high-level visual ROIs (union-ventral, LOC), 3 subjects.
The within-category test that was **underpowered on Mitchell-60** (+0.032, CI spanning 0; ~120 within-cat pairs)
now has **7,165 within-category pairs across 52 categories**.

Partial-lexical RSA; across-subject noise ceiling ventral **[0.339, 0.703]**, LOC [0.331, 0.700].

| ROI | decoder | full RSA | category-controlled | **within-category** |
|---|---|---|---|---|
| ventral | **free LLM** | 0.294 (**87% ceil-lo**) | 0.275 | **+0.250** |
| | VICE / human | 0.314 | 0.295 | +0.471 |
| | mpnet | 0.271 | 0.252 | 0.170 |
| LOC | **free LLM** | 0.254 (77%) | 0.235 | **+0.269** |
| | VICE / human | 0.344 | 0.323 | +0.535 |
| | mpnet | 0.228 | 0.209 | 0.165 |

**823-way zero-shot decoding** (chance top-1 = 0.12%): free LLM mean-percentile **0.825 (ventral) / 0.842 (LOC)**,
top-1 1.2% / 2.1% (10–17× chance), **permutation p = 0.0099**; word-length control at chance (0.56). VICE similar.

## What's new
Within-category structure **resolves at scale**: the free LLM recovers meaning relations *within* categories
(not just "animal vs tool"), at **+0.25 / +0.27**, where the 60-noun set saw nothing (underpowered). The
category-controlled RSA barely drops (0.294 → 0.275) → the match is **not** coarse taxonomy. Full RSA reaches
**87% of the noise ceiling** (ventral) — higher than Mitchell's ~67%.

## The gate — PASSED (vision control, `tier2_vision.py`, 645/823 concepts with images)
ventral/LOC are visual cortex and subjects saw photographs, so we partialled out a **CLIP-image RDM**
(conservative — CLIP is vision-language) on top of word-form:

| ROI | decoder | full: \|lex → \|lex+vision | within-category: \|lex → \|lex+vision |
|---|---|---|---|
| ventral | **free LLM** | 0.271 → 0.147 | **0.282 → 0.141, 95% CI [0.062, 0.218]** |
| | VICE / human | 0.334 → 0.210 | 0.480 → 0.371 |
| LOC | **free LLM** | 0.297 → 0.165 | **0.308 → 0.156, 95% CI [0.073, 0.234]** |
| | VICE / human | 0.363 → 0.233 | 0.541 → 0.435 |

Within-category structure is **roughly halved** by the vision control but **survives — CI excludes zero** — for
both the free LLM (0.14–0.16) and human behavior (0.37–0.44). So ~half the within-category structure in this
visual cortex is genuinely visual (the drop), and the other half is **meaning the free LLM shares — beyond
taxonomy and beyond appearance.** This is the fine-grained structure the 60-noun Mitchell set physically could
not resolve (it was at +0.03, CI spanning zero). **Verdict: within-category meaning is real and the LLM shares it.**

## Honest
- **Human behavior (VICE) >> the LLM within-category** (0.47/0.54 vs 0.25/0.27): 1.5M judgments capture fine
  within-category structure much better than the text-only LLM does. We report that.
- n = 3 subjects; representative-image proxy for the vision control; decoding is shortlisting, not verbatim.

## Deeper — unique variance (`tier2_partition.py`): does the LLM add beyond vision AND behaviour?
partial(LLM, brain | word-form + CLIP-image + VICE-behaviour), 645 image+VICE-matched concepts:
- **FULL RSA: reliably yes** — ventral **+0.051 [0.030, 0.072]**, LOC **+0.059 [0.036, 0.083]** (95% CI excludes 0).
  The free LLM uniquely predicts brain meaning-geometry beyond word-form, appearance, AND 1.5M human judgments.
- **WITHIN-category: not significant** — ventral +0.057 [−0.020, 0.132], LOC +0.058 [−0.026, 0.134].
- **R² unique (full):** LLM ~0.2–0.3%, vision ~0.15%, behaviour **~2.3–2.8%** — the LLM's unique slice is real but
  ~10× smaller than behaviour's. Honest reading: the LLM mostly **shares** structure with human behaviour (both are
  meaning models), plus a small reliable unique residual; within-category that residual is indistinguishable from 0.
