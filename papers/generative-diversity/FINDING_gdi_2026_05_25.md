# Finding · GDI — a valid (coherence-gated) generative-diversity instrument; alignment-tax null

**2026-05-25.** Prereg `preregistration_gdi_2026_05_25.md`. **Verdict: PASS** (G1∧G2∧G3).
The divergence detector, valence-flipped, is a working diversity instrument — with two
honest qualifiers.

## Result (3 models, run once)

| check | value | bar | verdict |
|---|---|---|---|
| mean GDI open vs closed | 1.63 vs 0.11 (14.4×) | G1 ≥ 2× | **PASS** (AUC 1.0) |
| temp sweep (gpt-4o-mini) | 0.5→1.15, 1.0→1.52, 1.5→1.79 | G2 monotonic | **PASS** |
| coherence on open samples | 60/60 = 1.00 | G3 ≥ 0.85 | **PASS** |

Diversity leaderboard (mean GDI on open, temp 1.0): gpt-4o-mini **1.73** · gpt-4o
**1.59** · gpt-3.5-turbo **1.55**.

## What's real

- **GDI is a valid instrument:** it responds to open-endedness (14× over closed
  prompts), it tracks the temperature knob monotonically, and — the design's whole point
  — the high entropy is **coherent** (every sampled open answer judged on-topic). So GDI
  measures *variety*, not drift; it is the positive-valence sibling of the confabulation
  detector, identical machinery (cosine@0.90 cluster entropy over N samples), opposite
  sign. A real, useful **mode-collapse / diversity** signal styxx didn't have.

## Two honest qualifiers (don't oversell this)

1. **G1's AUC 1.0 is near-tautological.** "Give me a startup idea" ×6 scatters; "what is
   2+2" ×6 is identical. Open ≫ closed entropy is construct validity (the index does what
   it says), **not a discovery**. The instrument is confirmed, not a surprising result.
2. **The alignment-tax hypothesis is NOT supported.** I predicted heavily-RLHF'd models
   might mode-collapse (lower GDI). They didn't: gpt-4o (1.59) is *not* below gpt-3.5
   (1.55), and the entire spread is **0.18 across three models on 8 prompts** — almost
   certainly within noise. **There is no real diversity ranking in this data.** Reported
   as the null it is.

## Honest scope

8 open + 4 closed prompts, 3 models, N=6, single run, OpenAI-only. Coherence judged by
gpt-4o-mini (same-family judge). The leaderboard needs many more prompts + seeds before
any model-diversity claim; this run only validates the *instrument* and returns a *null*
on model differences.

## Place in the arc

Completes the symmetry: divergence across samples flags **confabulation** (bug) on
factual prompts and measures **diversity** (feature) on open prompts — the same signal,
read against the prompt's intent. The knowledge-boundary work and GDI are one mechanism
(semantic divergence) pointed at two questions. styxx now has both the failure-detector
and its positive mirror, each pre-registered and honestly bounded.
