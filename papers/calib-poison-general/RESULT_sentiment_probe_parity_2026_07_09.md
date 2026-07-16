# RESULT — probe-parity control: the "private-calibration defense" gap is MAJORITY probe capacity, MINORITY privacy.

**Fathom Lab · papers/calib-poison-general · 2026-07-09. Verdict against the frozen non-overlapping
bands of `PREREG_sentiment_probe_parity_2026_07_09.md`. Carries the two disclosed findings of the
blind pre-result self-review (`_parity_selfreview.md`). This result puts a scope note on published
work; the note itself is drafted for operator sign-off, not applied here.**

## Verdict: PARTIAL — privacy is real but explains a minority of the gap

Three auditors on the same attacked models, all reading EVAL: naive-DoM (DoM only, fit on full
ATTACK); parity-naive (full probe family DoM+logistic+whole-stack, fit on an n=|CALIB| subsample of
the poisoned ATTACK split); private (full probe family, fit on clean CALIB). Clean guard reproduced
(private-audit EVAL AUROC 0.962, behavioral 0.845). All four cells behavior-preserving (0.87–0.89).

**Qwen2.5-1.5B (`sentiment_probe_parity_result.json`):**

| seed | λ | naive-DoM | parity-naive | private | capacity_share | parity_gap (privacy) |
|---|---|---|---|---|---|---|
| 0 | 1.0 | 0.767 | 0.837 | 0.889 | 0.071 | 0.052 |
| 0 | 3.0 | 0.781 | 0.843 | 0.893 | 0.063 | 0.049 |
| 1 | 1.0 | 0.724 | 0.835 | 0.856 | 0.111 | 0.021 |
| 1 | 3.0 | 0.729 | 0.828 | 0.859 | 0.099 | 0.032 |

The median baseline gap (private − naive-DoM) is 0.127. Decomposed at matched probe family and fit
size: `capacity_share` (poisoned data, richer probe) recovers 0.063–0.111; the residual privacy effect
`parity_gap` (private − parity-naive) is 0.021–0.052, median 0.041. Per the frozen bands, no cell
reaches PRIVACY_STANDS (parity_gap ≥ 0.06), and only one cell is ≤ 0.03 (not a CAPACITY majority), so
the verdict is **PARTIAL**.

## What this means (the honest decomposition)

- **Probe capacity is the majority of the "defense."** Handing the *poisoned* auditor the same rich
  probe family and fit size that the private auditor had lifts it from naive-DoM (0.72–0.78) to
  parity-naive (0.83–0.84) — recovering roughly two-thirds of the gap — using data the attacker
  optimized against. The bulk of what looked like a private-calibration win was a better estimator.
- **Privacy is real but minor.** The parity_gap is positive in all four cells (0.021–0.052): a clean
  split does read better than an equally-equipped poisoned one, so calibrating on unseen data helps.
  But it is roughly one-third of the gap, not the whole story.
- **This back-propagates to published work.** `mount.relock` and the read-neq-write "private-
  calibration defense" (Fathom v28) attribute the recovery to privacy. On this construct, at parity,
  the recovery is majority probe capacity. The operative rule is at least as much "use a richer probe
  (DoM + logistic + whole-stack)" as "use a private split." A SCOPE NOTE is owed on both surfaces —
  drafted at `SCOPE_NOTE_privacy_vs_capacity_2026_07_09.md`, operator-gated.

## Self-review caveats (carried verbatim from `_parity_selfreview.md`)

1. Absolute AUROCs are selection-optimistic (best of the full probe family — DoM per layer, logistic
   per layer, whole-stack — on EVAL), but the verdict metric `parity_gap` compares two
   identically-selected auditors, so the bias cancels; the privacy decomposition is unbiased.
2. The parity-naive ATTACK subsample is random, not class-stratified — a mild imbalance would slightly
   INFLATE parity_gap toward privacy, i.e. bias AGAINST the capacity conclusion. That the privacy share
   still came out a minority despite this conservative bias makes the "majority capacity" reading robust.

## Bounds

`Qwen2.5-1.5B`, single construct (sentiment), single-layer L22 attack, λ ∈ {1,3}, seeds {0,1}, linear
probes, small EVAL. Single-construct, single-family decomposition; a matched honesty-side parity check
is owed to confirm the same on the flagship's own construct.

## Reproducibility

`sentiment_probe_parity.py` → `sentiment_probe_parity_result.json`. Prereg + blind self-review frozen
before the number.

---
*Done proper: the private > naive gap I nearly banked as a "universal defense" is, at equal probe
capacity and fit size, roughly two-thirds a richer estimator and one-third genuine privacy. Privacy
is not zero — but it is not the headline, and two published surfaces need the scope note this run
earns.*
