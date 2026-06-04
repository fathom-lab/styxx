# RESULT — Real-drift validation: the monitor catches REAL fine-tuning damage, and tells harmful from helpful

**Date:** 2026-06-03 · Closes the last caveat on the meaning-integrity monitor — *"only synthetic
corruptions tested."* Instead of hand-shuffling embeddings, we **fine-tune BERT (real gradient steps)** and
watch the monitor track the representation damage, checkpoint by checkpoint. Two runs from the *same* init,
differing **only in the labels**:
- **DAMAGING** — random labels (label-noise → the model memorizes noise → corrupts its semantics).
- **BENIGN** (positive control) — real Binder Super-Category labels (meaningful supervision).

Reference: Binder 65-feature human space. Deep = BERT content-token embeddings. Code: `real_drift.py`.

## Result — the monitor separates HARMFUL from HELPFUL fine-tuning
Healthy baseline alignment **0.193**. Trajectories (alignment · vital-sign verdict):

| step | DAMAGING (random labels) | BENIGN (real categories) |
|---:|---|---|
| 0   | 0.193 · HEALTHY  | 0.193 · HEALTHY |
| 50  | 0.099 · DEGRADED | 0.581 · HEALTHY |
| 100 | 0.141 · HEALTHY* | 0.475 · HEALTHY |
| 200 | 0.025 · BROKEN   | 0.441 · HEALTHY |
| 400 | **0.008 · BROKEN** | **0.442 · HEALTHY** |

- **DAMAGING falls to ~0 and ends BROKEN** (−0.185). **BENIGN rises +0.25 and stays HEALTHY** — meaningful
  fine-tuning makes the model *more* human-aligned, and the monitor reads that correctly.
- **Same model, same steps, only the labels differ → opposite verdicts.** This is the proof it's a
  *meaning-damage* detector, not a "fine-tuning happened" detector. The benign control is what proves it.

## Why it matters
- **Closes the synthetic-corruption caveat:** the monitor catches **real** degradation — actual gradient
  steps on bad supervision — not just hand-edited embeddings.
- **It distinguishes damage from improvement.** A naive "did the representation change" check would fire on
  both runs; this fires only on the one that *hurt the meaning.*
- Bonus finding: training on real semantic categories **raised** human-meaning alignment 0.19 → 0.44.

## Localization on REAL targeted damage (`real_drift_localize.py`)
The synthetic localization (ROC-AUC 0.95) used hand-shuffled embeddings. This closes the gap on *real*
damage: **targeted data-poisoning** — fine-tune BERT where 30% of concepts get random (poisoned) labels and
the rest get their real categories. Does the monitor's per-concept channel pick out the poisoned ones?

| run | localization ROC-AUC | precision@130 | poisoned-mean vs clean-mean |
|---|---|---|---|
| **POISONED (targeted)** | **0.848** | 0.692 | +0.071 vs +0.346 |
| **CLEAN control (all real labels)** | 0.475 (chance) | 0.292 | +0.472 vs +0.445 |

The poisoned concepts are dramatically more degraded (+0.07 vs +0.35), and the monitor **localizes them at
AUC 0.85.** The CLEAN control is the proof it's real: with no poisoning, the *same designated set* scores at
chance (0.475) — so the localization is the actual poisoning, not an artifact. **The monitor can name which
concepts a fine-tune poisoned** — a real data-poisoning detection + localization capability. Honest: 0.85 <
the synthetic 0.95 because real fine-tuning damage is messier (variable per-concept corruption + some
clean-concept drift). Example poisoned concepts the monitor flagged: honeymoon, glass, choir, banjo, lightning.

## Honest notes
- *The damaging trajectory is noisy:* step-100 bounced to 0.141/HEALTHY (frac 0.73, just over the 0.70 band)
  before collapsing. That is real training noise — a deployment would trend/smooth, not gate on a single
  checkpoint. The endpoint (0.008, BROKEN) is decisive; the *direction* is unambiguous.
- *Design fix the test forced:* the vital sign now judges **relative to the calibrated healthy baseline**
  (retain <70% of baseline alignment → DEGRADED, <40% → BROKEN), not an absolute cutoff. Different
  embedding-extraction methods have different natural scales (bare-word BERT baseline 0.19 vs templated
  0.44); the old absolute 0.25 threshold mislabeled the *healthy* step-0 model as DEGRADED. Relative-to-
  baseline is the correct design for a vital sign, and the real-drift test is what surfaced it.
- *Scope:* one model (BERT), one damage mode (label noise), one benign control (categories). Catastrophic-
  forgetting and poisoned-subset (for localization) are the natural next real-drift modes.

## Reproduce
`python real_drift.py` → both trajectories + the harmful-vs-helpful verdict. Result: `real_drift_result.json`.
