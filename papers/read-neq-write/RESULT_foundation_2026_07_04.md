# RESULT — read≠write coupling: clean-model foundation clears all non-degeneracy guards

**Fathom Lab · papers/read-neq-write · 2026-07-04. Foundation stage of
`PREREG_read_neq_write_coupling_2026_07_04.md` (committed before any run). Establishes that the
honesty read and the model's knowledge both exist on the clean model, so the adversarial LoRA sweep
has something real to pull apart. Not the verdict — the gate the verdict depends on.**

## Result: FOUNDATION-VALID (all four guards pass)

On `Qwen/Qwen2.5-1.5B-Instruct`, seed 0, over 180 balanced true/false factual statements split by
topic stem into 98 ATTACK and 82 disjoint HELD-OUT facts:

| guard | measured | bar | verdict |
|-------|----------|-----|---------|
| G-void-1 clean probe held-out AUROC | **0.85** | ≥ 0.75 | PASS |
| G-void-2 clean knowledge accuracy | **0.80** | ≥ 0.80 | PASS |
| G-void-3 ATTACK ∩ HELD-OUT stems | disjoint | ∅ | PASS |
| G-void-4a random-direction AUROC | **0.50** | chance band | PASS |
| G-void-4b shuffled-knowledge accuracy | **0.51** | ≈ 0.5 | PASS |

A difference-of-means honesty direction fit on the ATTACK facts' final-token residuals, evaluated on
the **disjoint** HELD-OUT facts, separates true from false at **AUROC 0.85** — layer 18 was selected
by held-out AUROC over a scan (L12 0.60, L14 0.73, L16 0.81, **L18 0.85**, L20 0.83, L22 0.81). A
random direction at the same layer sits at **0.50** (chance), so the read is a real signal, not a
high-variance artifact. The model behaviorally knows true from false at **0.80** accuracy on the
same held-out facts (its own True/False logit margin under a neutral prompt); shuffling the labels
collapses that to **0.51**.

## What this sets up

The stage is now genuinely loaded for the keystone test: there is a substrate honesty read at 0.85
and disjoint-set knowledge at 0.80. The next stage — the adversarial LoRA sweep with a
knowledge-replay regularizer, re-locking the probe on the attacked model — asks whether an attacker
can push the **re-locked** held-out probe AUROC to ≤ 0.60 (chance) while holding held-out knowledge
≥ 0.75. If yes on ≥ 2 seeds, `read ≠ write` is EVADABLE by a realistic weight-level attacker; if
suppression is coupled to knowledge loss, it SURVIVES. The frozen corners and the reference
knowledge level (0.80) are locked here, before the attack is trained.

## Reproducibility

`papers/read-neq-write/foundation.py` (deterministic, seed 0, no training) regenerates
`foundation_result.json`. Reuses the frozen true/false banks and residual/behavioral machinery of
`papers/showcase-viz/run_says_yes_knows_no.py`. Layer, seed, and all four guard values are recorded
in the receipt.
