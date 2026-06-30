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

## The gate — PENDING and essential (running now)
ventral/LOC are **visual cortex** and subjects **saw photographs**, so within-category neural structure could be
**visual** (within "animals," things that look alike), not meaning. The **CLIP-image vision control** (partial out
an image-model RDM) is **running** (`tier2_vision.py`). Until it lands, this is *"within-category structure
resolves and the free LLM shares it,"* **NOT yet "within-category meaning beyond appearance."** ← update on completion.

## Honest
- **Human behavior (VICE) >> the LLM within-category** (0.47/0.54 vs 0.25/0.27): 1.5M judgments capture fine
  within-category structure much better than the text-only LLM does. We report that.
- n = 3 subjects; representative-image proxy for the vision control; decoding is shortlisting, not verbatim.
