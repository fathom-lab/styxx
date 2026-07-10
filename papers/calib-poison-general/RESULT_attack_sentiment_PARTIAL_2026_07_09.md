# RESULT — sentiment Stage-2: PARTIAL. Calibration poisoning is DIRECTIONALLY present but the single-layer attack cannot fully poison a robust sentiment probe.

**Fathom Lab · papers/calib-poison-general · 2026-07-09. Verdict against the frozen Stage-2 gate of
`PREREG_calib_poison_sentiment_2026_07_09.md` (GENERALIZES / NO_GENERALIZATION / PARTIAL, committed
before the run). Licensed by foundation v2 (FOUNDATION_VALID). Attack matched to the honesty stage-2
strength (LoRA r=16, MICRO=8, full-length reviews via gradient checkpointing, 300 steps, single-layer
L22 scrub, behavioral-sentiment replay).**

## Verdict: PARTIAL_OR_INCONCLUSIVE — reported verbatim

The frozen gate needs the naive (ATTACK-calibrated) re-lock poisoned to ≤ 0.60 with the private
(CALIB-calibrated) re-lock recovering ≥ 0.70 for GENERALIZES, or the private re-lock also collapsing
to ≤ 0.60 for NO_GENERALIZATION. Neither corner is reached: the naive re-lock does not collapse to
chance, and the private re-lock stays high. Clean-model guard passed (private-audit EVAL AUROC 0.962,
behavioral sentiment accuracy 0.845).

**Qwen2.5-1.5B, all four cells (`attack_sentiment_result.json`):**

| seed | λ | naive re-lock (poisoned) | private re-lock (clean-calibrated) | behavioral | random |
|---|---|---|---|---|---|
| 0 | 1.0 | 0.767 | **0.888** | 0.882 | 0.449 |
| 0 | 3.0 | 0.780 | **0.892** | 0.864 | 0.458 |
| 1 | 1.0 | 0.723 | **0.852** | 0.891 | 0.375 |
| 1 | 3.0 | 0.729 | **0.859** | 0.882 | 0.374 |

## What this establishes (the honest, two-sided reading)

- **The single-layer attack cannot poison the sentiment probe to chance.** The naive re-lock drops
  from its high clean value to 0.72–0.78 — hurt, but nowhere near the ≤ 0.60 collapse the *honesty* attack
  produced (there the naive re-lock fell to ~0.48). The sentiment read (clean 0.954, graded) is more
  robust than the honesty read (clean 0.85); a knowledge-preserving single-layer LoRA scrub cannot
  flatten it. Every attack point preserved behavioral sentiment (0.86–0.89 ≥ 0.75).
- **But the calibration-poisoning DIRECTION is present and perfectly consistent.** In all four cells
  the private (clean-calibrated) re-lock beats the naive (poisoned) re-lock — 0.888 vs 0.767, 0.892
  vs 0.780, 0.852 vs 0.723, 0.859 vs 0.729 — a systematic gap in the same direction as honesty. The
  poisoned probe is hurt more than the clean-calibrated one, on a construct with nothing to do with
  honesty. The phenomenon appears; only its magnitude falls short of the full-collapse bar.

## What it does and does NOT license

- **Does NOT license GENERALIZES.** The naive re-lock was not poisoned to chance, so the full
  calibration-poisoning phenomenon is not reproduced on sentiment by this attack.
- **Does NOT license NO_GENERALIZATION.** The private re-lock is 0.85–0.89, nowhere near the ≤ 0.60
  collapse that verdict requires; the signal is plainly not erased. And the consistent private > naive
  ordering is evidence *for* the direction, not against it.
- **Licenses only:** on sentiment, a single-layer knowledge-preserving attack partially poisons the
  naive read (private beats naive in every cell) but cannot collapse it — the robust
  sentiment signal resists a single-layer scrub.

## The forced next step (a NEW frozen prereg — a stronger attack, not a moved bar)

The honesty attack scrubbed one layer and the poisoning propagated to collapse the naive re-lock;
on the more robust sentiment signal it did not. The natural stronger attacker is the E2-style
**whole-stack scrub** — scrub the sentiment direction at *every* scan layer, not just the deployed
one — which propagates the poisoning across the residual stack by construction. If the naive re-lock
then collapses (≤ 0.60) while the private re-lock holds (≥ 0.70), calibration poisoning GENERALIZES;
if even a whole-stack attack cannot poison the naive read while preserving behavior, sentiment is
genuinely more poison-resistant than honesty — itself a real finding. Frozen separately before its
run. The Stage-2 bars here are untouched.

## Bounds

`Qwen2.5-1.5B-Instruct`, 440 length-matched 2★/4★ reviews, index-split 220/110/110, LoRA r=16, 300
steps, single-layer L22 scrub, λ ∈ {1.0, 3.0}, seeds {0,1}, DoM + logistic + whole-stack
clean-calibrated auditor. The finding is "a single-layer attack under-poisons the robust sentiment
probe," not "calibration poisoning fails on sentiment."

## Reproducibility

`attack_sentiment.py` (gradient-checkpointed to run MICRO=8 full-length in 8 GB) →
`attack_sentiment_result.json`. Reuses the v2 foundation split and the honesty attack structure.
Prereg + foundation frozen before this ran.

---
*PARTIAL, and honestly cornered: the poisoning direction is there in all four cells, the magnitude is
not, and the stronger attack that would resolve it is named — not a nudge of the bar to reach a
headline.*
