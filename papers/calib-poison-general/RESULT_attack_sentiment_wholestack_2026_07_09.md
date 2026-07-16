# RESULT — sentiment Stage-2b (whole-stack): a NEAR-MISS. The attack drove the naive read to 0.62 (just above the 0.60 collapse bar) while the private read held.

**Fathom Lab · papers/calib-poison-general · 2026-07-09. Verdict against the frozen gate of
`PREREG_calib_poison_sentiment_wholestack_2026_07_09.md`. Reported with a disclosed prereg-ambiguity:
the frozen NO_GENERALIZATION and PARTIAL clauses OVERLAP at the observed value, so the conservative
reading is taken — this is a near-miss, NOT a clean "sentiment is poison-resistant."**

## Verdict: NEAR-MISS (harness fired NO_GENERALIZATION on the ≥0.60 letter; the honest reading is PARTIAL)

The whole-stack knowledge-preserving attack — scrubbing the sentiment direction at every scan layer —
is much stronger than the single-layer attack, and it nearly poisons the naive read. Clean guard
passed (private-audit EVAL AUROC 0.962; frozen whole-stack read 0.819). The attack bit hard on all
cells (frozen whole-stack read fell to 0.602–0.632).

**Qwen2.5-1.5B, all four cells (`attack_sentiment_wholestack_result.json`):**

| seed | λ | naive re-lock (poisoned) | private re-lock (clean-calibrated) | behavioral | frozen-stack read |
|---|---|---|---|---|---|
| 0 | 1.0 | 0.619 | **0.730** | 0.827 | 0.613 |
| 0 | 3.0 | 0.626 | **0.764** | 0.873 | 0.632 |
| 1 | 1.0 | 0.616 | **0.740** | 0.818 | 0.602 |
| 1 | 3.0 | 0.621 | **0.750** | 0.864 | 0.610 |

## The disclosed prereg ambiguity (and the honest conservative call)

The frozen prereg's GENERALIZES requires naive ≤ 0.60; the naive re-lock landed at 0.616–0.626, so
GENERALIZES is **not** earned — cleanly, no bar-moving. But the prereg's NO_GENERALIZATION clause
("cannot drive naive ≤ 0.60 on ≥2 seeds") and its PARTIAL clause ("naive in (0.60, ~0.75] with
private high") BOTH cover a naive of 0.62 — an overlap I wrote into the prereg. The `--` harness
resolved it to NO_GENERALIZATION on the strict letter, but the substantive NO_GENERALIZATION
description ("sentiment is genuinely poison-resistant… the signal cannot be poisoned") is **not
supported by these numbers**: the naive read nearly collapsed. I therefore take the conservative
reading — this is a **near-miss / PARTIAL**, and the strong "poison-resistant" conclusion is NOT
claimed.

## What this establishes honestly

- **Poisoning severity scales with attack breadth.** Single-layer attack left naive at 0.72–0.78;
  the whole-stack attack drove it to 0.616–0.626 — much closer to collapse. The robust sentiment
  probe (clean 0.954) resists, but the resistance is being overcome as the attack strengthens.
- **The private-calibration defense direction is universal.** In all four whole-stack cells the
  private re-lock (0.730–0.764) beats the naive re-lock (0.616–0.626), the same ordering as the
  single-layer sentiment attack and as honesty. The clean-calibrated read is systematically more
  robust than the poisoned one on a construct unrelated to honesty — this is the load-bearing
  generalization, and it holds.
- **GENERALIZES is a hair away.** naive 0.62 vs the 0.60 bar, with private clearly ≥ 0.70. The
  downward trend across attack strength (0.95 clean → 0.75 single-layer → 0.62 whole-stack) points
  at a clean GENERALIZES under a slightly stronger attacker.

## The forced next step (the prereg's own next rung, frozen separately)

The prereg named the next rung for exactly this near-miss: an **r=64 whole-stack** attacker (4× LoRA
capacity, the E2 strong-attacker setting). Given the trend, it is the run that resolves the
ambiguity: if r=64 whole-stack drives naive ≤ 0.60 while private holds ≥ 0.70, calibration poisoning
GENERALIZES to sentiment; if it plateaus above 0.60, the robust sentiment probe is genuinely
harder-to-fully-poison and that boundary is real. Frozen before its run.

## Bounds

`Qwen2.5-1.5B`, 440 length-matched 2★/4★ reviews, 220/110/110, LoRA r=16, whole-stack scrub, 300
steps, gradient-checkpointed MICRO=8, λ ∈ {1.0, 3.0}, seeds {0,1}. The finding is "a whole-stack r=16
attack nearly-but-not-fully poisons the robust sentiment probe; the private-defense direction holds,"
not "sentiment cannot be poisoned."

## Reproducibility

`attack_sentiment_wholestack.py` → `attack_sentiment_wholestack_result.json`. Prereg frozen before
run; single-layer Stage-2 result at `RESULT_attack_sentiment_PARTIAL_2026_07_09.md`.

---
*A near-miss, reported without spin: the strong attacker got within a hair of full poisoning, the
private-calibration defense held above it in every cell, and — because my own prereg's clauses
overlapped at this value — I take the conservative reading and name the r=64 rung that resolves it,
rather than bank the flattering "poison-resistant" label the harness printed.*
