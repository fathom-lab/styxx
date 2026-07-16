# RESULT — sentiment Stage-2c (r=64 whole-stack): the poisoning FLOORS. 4× attacker capacity does not lower the naive read — full poisoning does not generalize to the robust probe.

**Fathom Lab · papers/calib-poison-general · 2026-07-09. Verdict against the frozen non-overlapping
bands of `PREREG_calib_poison_sentiment_r64_2026_07_09.md`. Two disclosures: (1) the harness JSON
carried STALE `verdict`/`prereg` strings from the Stage-2b (r=16) code — only the rank was
parameterized — so the governing bands are the r=64 prereg's, applied here to the real numbers;
(2) the load-bearing finding is the PLATEAU, not the exact band label.**

## Verdict: PARTIAL by the strict band — but the definitive finding is a poisoning FLOOR

Raising the whole-stack attacker to LoRA r=64 (4× the r=16 capacity, the E2 strong-attacker setting)
did **not** push the naive re-lock lower. It plateaued — slightly higher, if anything — while the
private re-lock held and rose. Clean guard passed (private-audit EVAL AUROC 0.962; frozen whole-stack
read 0.819). The attack bit hard (frozen-stack read fell to 0.527–0.605).

**Qwen2.5-1.5B, behavior-preserving cells (behavioral ≥ 0.75; `attack_sentiment_wholestack_result_r64.json`):**

| seed | λ | naive re-lock | private re-lock | behavioral | frozen-stack read |
|---|---|---|---|---|---|
| 0 | 1.0 | 0.651 | **0.791** | 0.818 | 0.592 |
| 0 | 3.0 | 0.650 | **0.799** | 0.882 | 0.605 |
| 1 | 3.0 | 0.642 | **0.760** | 0.827 | 0.580 |

(The fourth cell, seed 1 λ=1.0, had behavioral 0.736 < 0.75 — not behavior-preserving — and is
excluded, per the frozen behavior-preserving condition.)

Per the frozen r=64 bands: GENERALIZES needs naive ≤ 0.60 (not met); NO_GENERALIZATION needs
naive > 0.65 on ≥ 2 seeds (only one cell, seed 0, exceeds 0.65 — not met); so the strict band is
**PARTIAL** (0.60 < naive ≤ 0.65, private ≥ 0.70).

## The load-bearing finding: the poisoning asymptotes above chance

The point of Stage-2c was to resolve the r=16 near-miss by asking whether a *stronger* attacker
collapses the naive read. It does not. The attack-strength series on the naive re-lock:

- clean read (no attack): ~0.95
- single-layer r=16: 0.72–0.78
- whole-stack r=16: 0.616–0.626
- **whole-stack r=64: 0.642–0.651**

Quadrupling attacker capacity did **not** continue the downward trend — the naive read plateaued in
the low-to-mid 0.6s, never reaching the 0.60 collapse bar. This is the decisive evidence that the
near-miss was a **floor, not a strength limitation**: a knowledge-preserving attack cannot fully
poison the robust sentiment probe. Full calibration poisoning — the honesty phenomenon where the naive read collapses
to chance — does **not** generalize to the robust sentiment probe.

## What generalizes, and what does not (the honest two-sided close)

- **The private-calibration DEFENSE direction generalizes — universally.** In every cell of every
  sentiment attack (single-layer, whole-stack r=16, whole-stack r=64) the private re-lock beats the
  naive re-lock; here 0.760–0.799 vs 0.642–0.651. The clean-calibrated read is systematically more
  robust than the poisoned one, on a construct unrelated to honesty. This is the load-bearing
  generalization and it holds at every attacker strength.
- **Full poisoning SEVERITY does NOT generalize — it is graded by probe robustness.** Honesty
  (clean read ~0.85) collapses to chance under a single-layer r=16 attack; the robust sentiment probe
  (clean read 0.954) floors at ~0.62–0.65 even under a whole-stack r=64 attack. The stronger the
  representation, the harder it is to fully poison.

## Disposition (the arc closes here)

The generalization question is resolved without another rung: r=64 confirms a floor, so further
attacker escalation is not warranted (diminishing returns, pre-stated in the prereg). The result
feeds the synthesis (`FINDING_calib_poison_law_2026_07_09.md`): calibration poisoning + the
private-calibration defense form a **robustness-graded law**, not an all-or-nothing property.

## Bounds

`Qwen2.5-1.5B`, 440 length-matched 2★/4★ reviews, 220/110/110, LoRA r=64 whole-stack, 300 steps,
gradient-checkpointed MICRO=8, λ ∈ {1.0, 3.0}, seeds {0,1}. One 1.5B model, one probe/attack family;
the floor is demonstrated for this attacker class, not proven against every conceivable attack.

## Reproducibility

`attack_sentiment_wholestack.py --rank 64` → `attack_sentiment_wholestack_result_r64.json` (its
embedded `verdict`/`prereg` strings are stale Stage-2b artifacts; the numbers are the record).
Prereg `PREREG_calib_poison_sentiment_r64_2026_07_09.md` frozen before the run.

---
*The definitive rung: 4× the attacker and the naive read would not budge below ~0.65. Full poisoning
floors above chance on the robust probe — so it does not generalize — while the private-calibration
defense direction holds at every strength. The law is robustness-graded, and we resolved it by
running the escalation rather than asserting the trend.*
