# PREREG — portable conscience v1: the properly-powered transfer test

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx. Successor to v0 (MODEL-SPECIFIC).**

v0's floor hit 1.000 not because the conscience transferred, but because 16 PAIRED items give a
degenerate floor: the true-false difference is dominated by one direction, so random projections
separate either all or none (95th pct -> 1.0). v1 fixes the power: a LARGE balanced true/false set
scored by AUROC, where random directions concentrate near 0.5 and a genuine transferred honesty
signal must beat them. The v0 confound becomes the v1 control.

## Question

Does gemma-2-2b's truthfulness DIRECTION, carried through a label-free cross-model map, read
true-vs-false in models it was never trained on (Llama-3.2-3B, Qwen2.5-3B) — better than random
directions through the same map? PASS = the conscience transfers as a direction (the North Star
single-instrument property); FAIL = portable only as structure, not as a vector (v0's reframed
conclusion confirmed).

## Apparatus (frozen)

- **Source direction:** gemma-2-2b-it truthfulness `w` (atlas, layer 12, dim 2304) + bias `b`.
- **Targets:** Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct (never trained for this probe).
- **Statements:** ~240 generated factual statements, balanced true/false, diverse templates
  (capitals, element symbols, arithmetic, animal class, geography is-a, planets, science), disjoint
  from the v0 test facts. Split: ANCHOR (~140, for fitting the map) / TEST (~100, held out).
- **Map `M` (label-free):** ridge regression mapping a target's residual at its selected layer into
  gemma's layer-12 space (target layer chosen by best anchor-split fit R^2 only; alpha on a held-out
  anchor split). The map sees activations, never truth labels.
- **Score:** for each TEST statement, `s = w . M(h_target) + b`; metric = **AUROC(s, true_label)**
  over all ~100 test items (true=1, false=0).

## Pre-registered gates (frozen)

- **P1 (direction transfers):** transferred AUROC >= 0.65 AND > the 95th percentile of a
  random-direction floor (200 random unit directions of matched norm through the SAME map, AUROC
  each) — for BOTH targets. PASS -> **CONSCIENCE-PORTABLE-v1** (the direction itself transfers).
  PASS on one target only -> **PARTIAL**. FAIL on both -> **STRUCTURE-NOT-DIRECTION** (v0 confirmed:
  portable as geometry, not as a vector).
- **P2 (map necessity, descriptive):** transferred AUROC vs a random-map control (same shape, no fit).
- **Ceiling/VOID:** gemma's own probe on gemma TEST activations must reach AUROC >= 0.70, else
  VOID-PIPELINE. With ~100 items the random-direction floor must concentrate (p95 < 0.75), else the
  test is still underpowered (report and widen).

## VOID / scope

- VOID-PIPELINE (ceiling or floor-concentration fails); VOID-SUBSTRATE (a model fails to load).
- In-distribution caveat stands (calibrated correlate, not a verdict). Smoke -> `*_SMOKE_INVALID*`.
- Linear ridge map, one source, two targets, one task. Positive = existence of a transferable
  honesty direction across these minds; negative bounds linear single-direction transfer only.

## Honest prior

Leaning STRUCTURE-NOT-DIRECTION (v0's reframed conclusion), but properly powered this could flip:
if the honesty direction genuinely aligns across minds, AUROC will clear a now-informative floor.
Either verdict sharpens the North Star: PORTABLE = one universal direction exists; otherwise the
universal conscience must be FIT in the aligned space, not carried as a vector.
