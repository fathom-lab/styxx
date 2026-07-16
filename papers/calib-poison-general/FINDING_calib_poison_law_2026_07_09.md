# FINDING — A robustness-graded law of substrate auditability: the private-calibration defense generalizes; full poisoning does not.

**Fathom Lab · papers/calib-poison-general · 2026-07-09. Synthesis of the generalization arc across
three constructs (refusal, honesty, sentiment). Every number below is grounded in a committed,
OATH-certified result; this note asserts no new measurement, only the structure they share.**

## The question

The flagship (Fathom v28, DOI 10.5281/zenodo.21263158) established, on *honesty*, that an apparent
evasion of a substrate probe is calibration poisoning, not erasure, and that a probe calibrated on a
private split the attacker never saw reads through the attack. Is that audit rule honesty-specific, or
a general property of substrate linear probes? We tested it on constructs chosen to be independent of
honesty, with pre-registered gates.

## Three regimes — the constructs are not three tries at one test; they are the law's regimes

**Regime 0 — no auditable structure (refusal → UNTESTABLE).** Harmful/benign refusal is a
surface/lexical property: its read is trivially separable (AUROC 1.000 at every layer, including the
earliest) and the model's behavioral judgment is near-chance (0.667 / 0.500). There is no graded
representation to poison and no behavioral capability to preserve, so the attack–defense question is
not even well-posed. *Prediction of the law: surface properties are not adversarially auditable.*
(`RESULT_foundation_refusal_UNTESTABLE_2026_07_09.md`.)

**Regime 1 — fragile probe, full poisoning (honesty).** The honesty read is graded (clean ~0.85) and
behaviorally grounded. A single-layer, knowledge-preserving LoRA attack collapses the naive
(attacker-calibrated) re-lock to chance (~0.46–0.51) while the private (clean-calibrated) re-lock
recovers — full calibration poisoning, defeated by private calibration. (Flagship; read-neq-write.)

**Regime 2 — robust probe, poisoning floors (sentiment).** The sentiment read is graded and stronger
(clean 0.954) with a real behavioral judgment (0.845). Escalating the attacker never fully poisons it:

| attacker | naive re-lock (poisoned) | private re-lock (clean-calibrated) |
|---|---|---|
| single-layer r=16 | 0.72–0.78 | 0.85–0.89 |
| whole-stack r=16 | 0.616–0.626 | 0.730–0.764 |
| whole-stack r=64 (4× capacity) | 0.642–0.651 | 0.760–0.799 |

The naive read floors in the low-to-mid 0.6s and never reaches the 0.60 collapse bar; quadrupling
attacker capacity does not lower it. Full poisoning does not reproduce on the robust probe.

## The law (two axes, stated sharply)

1. **The private-calibration DEFENSE direction is universal.** Wherever an auditable representation
   exists (regimes 1 and 2), the private re-lock beats the naive re-lock in every cell, every attack
   strength, both constructs. The clean-calibrated read is systematically the more robust of the two.
   This is what had to generalize for "calibrate on data the attacker did not see" to be a principle —
   and it does.
2. **Poisoning SEVERITY is graded by probe robustness.** How far the naive read can be poisoned tracks
   how strong the representation is: a fragile probe (honesty, clean ~0.85) collapses to chance under
   a single-layer attack; a robust probe (sentiment, clean 0.954) floors above chance even under a
   whole-stack 4×-capacity attack. Full poisoning is not all-or-nothing; it is a robustness-graded
   phenomenon.

## Why this matters

The audit rule is now a *principle with a boundary*, not a single result. For deployment: a substrate
honesty (or value) monitor is robustly auditable **iff** the property is behaviorally grounded and
gradiently represented (regime 0 is out), and the private-calibration defense holds across the board —
its *direction* is universal. The severity axis says something new and practical: the more robustly a
model represents a property, the harder an adversary must work to even partially poison its probe, and
the further it stays from full collapse. Robust representations are not just easier to read — they are
harder to poison.

## Honest bounds

All results are `Qwen2.5-1.5B` (sentiment/refusal) plus the 1–1.5B honesty families; linear probes;
knowledge-preserving LoRA attackers (r≤64); small evaluation sets. The floor is demonstrated for this
attacker class, not proven against every attack; the three regimes are three constructs, not a survey.
The law is a structured hypothesis with three grounded data points and a universal defense direction —
strong enough to state, honest enough to bound.

## Disposition

Feeds a paper version (v0.4 / next Fathom deposit, operator's call): "When Can You Audit a Model's
Mind? A robustness-graded law of substrate auditability," extending the honesty flagship to the
general case. All preregs, harnesses, result JSONs, and certificates are public at commit-level
granularity under `papers/calib-poison-general/`.

---
*Two failures and a flagship turned out to be three regimes of one law: surface properties cannot be
audited, fragile probes are fully poisoned, robust probes are only partially poisoned — and the
private-calibration defense direction holds across all of it. We earned the boundary by running the
escalation to a floor, not by asserting the trend.*
