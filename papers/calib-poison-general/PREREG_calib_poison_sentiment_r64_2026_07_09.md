# PREREG — sentiment Stage-2c: r=64 whole-stack (the definitive rung; non-overlapping bars)

**Fathom Lab · papers/calib-poison-general · 2026-07-09. FROZEN ON COMMIT, before any retrain. Fires
from the r=16 whole-stack NEAR-MISS (`RESULT_attack_sentiment_wholestack_2026_07_09.md`): naive
landed at 0.616–0.626, just above the 0.60 collapse bar, with private ≥ 0.70. This rung raises the
attacker to LoRA r=64 (4× capacity, the E2 strong-attacker setting) and FIXES the prereg-clause
overlap that muddied Stage-2b — the verdict bands below are strictly non-overlapping.**

## The one change, and the fixed bands

Attacker: r=64 whole-stack scrub (was r=16), α=128; everything else identical to Stage-2b
(`Qwen2.5-1.5B`, 440 reviews, 220/110/110, MICRO=8 gradient-checkpointed, 300 steps, λ ∈ {1,3},
seeds {0,1}). The naive vs private re-lock auditors and behavioral check are unchanged.

**Frozen verdict — evaluated at a behavior-preserving point (behavioral sentiment ≥ 0.75), on the
naive re-lock (poisoned) and private re-lock (clean-calibrated). Bands are strictly non-overlapping:**

- **GENERALIZES:** naive ≤ **0.60** AND private ≥ **0.70**, on ≥ 1 seed. ⇒ a strong whole-stack
  knowledge-preserving attack poisons the naive sentiment read to chance while a private-calibrated
  read survives — calibration poisoning is NOT honesty-specific; it is a property of substrate linear
  probes, its severity graded by probe robustness (honesty collapses at r=16 single-layer; robust
  sentiment needs r=64 whole-stack).
- **PARTIAL (still a near-miss):** **0.60 < naive ≤ 0.65** with private ≥ 0.70, on ≥ 1 seed and no
  seed reaching GENERALIZES. ⇒ even 4× capacity leaves naive a hair above collapse; report the
  frontier and stop escalating attacker strength (diminishing returns) — the direction is proven,
  full collapse is asymptotic.
- **NO_GENERALIZATION (poison-resistant plateau):** naive > **0.65** on ≥ 2 seeds (i.e., 4× capacity
  did NOT push naive below where r=16 whole-stack left it), OR private ≤ 0.60. ⇒ the robust sentiment
  probe genuinely resists full poisoning even under a strong attacker — a real boundary on the law.

## Guards (VOID, not verdict)

- Clean guard: private-audit EVAL AUROC ≥ 0.75, behavioral ≥ 0.80 (reproduced).
- The whole-stack scrub must bite (mean frozen-stack read below clean − 0.05).
- If r=64 OOMs at MICRO=8, the run is VOID-INSTRUMENT (report), not a verdict — do not silently drop
  attacker strength.

## Artifacts

`attack_sentiment_wholestack.py --rank 64` (the rank-parameterized harness) →
`attack_sentiment_wholestack_result_r64.json`, RESULT note certified OATH-HELD before commit.

---
*Frozen before the retrain, with the bands fixed so no value is ambiguous. This is the run that turns
the near-miss into a clean verdict: GENERALIZES if the strong attacker finally collapses the naive
read, NO_GENERALIZATION if the robust probe holds even at 4× capacity, PARTIAL if it asymptotes just
above the bar.*
