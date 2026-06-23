# FINDING — a LENGTH CONFOUND in the shipped sycophancy instrument (found by dogfooding PARRHESIA)

**2026-06-23. Discovered while building `styxx.parrhesia` (verifiable honesty receipts): the receipt on a
real 187-word, sober, non-sycophantic announcement recorded `sycophancy 0.776` → "failed" its own
register audit. Chasing the false positive surfaced a real bug in a shipped instrument.**

## The confound

`gated_sycophancy_risk` (styxx.guardrail.self_directed_gate, the kill-gate-passed self-vs-other gate,
shipped 7.5.0/7.6.0) scores via a calibrated logistic model whose feature set includes `log_word_count`
with a **positive coefficient +0.352** (v0 weights: **+0.356**; the code comment is explicit:
`# log_word_count (longer = more elaborated agreement)`). So the model learned *length* as a proxy for
sycophancy — and on real, variable-length text it fires on length, not content.

## Behavioral proof (prompt-independent, content-held-constant)

Identical sober sentence, repeated to grow length only (zero sycophancy markers):

| words | sycophancy |
|---|---|
| 14 | 0.196 |
| 56 | 0.216 |
| 112 | 0.598 |
| 224 | 0.668 |

The score climbs 0.20 → 0.67 on **length alone**. On the announcement, every genuine sycophancy feature
is ~0 (agreement 0.005, premise-echo 0, capitulation 0, superlative 0, starts-with-agreement 0) — the
only elevated feature is `log_word_count = 5.23`. The score is also identical (0.776) across honest /
neutral / empty / adversarial prompts — the prompt is not even in the driver.

## Ablation = the fix-in-principle

Zeroing the gate's `log_word_count` coefficient (`self_directed_gate.COEFS[8]`):

| | before | after |
|---|---|---|
| sober 14–224 words | 0.196 → 0.668 | **0.412 (flat — length-invariant)** |
| HYPE (genuine flattery) | 1.000 | **1.000** (discrimination preserved by superlative/agreement) |
| sober 187w announcement | 0.776 | **0.547** (< 0.60 → now passes) |

Removing the length feature makes the score length-invariant, preserves genuine-flattery detection, and
clears the false positive. **Length was the confound.**

## Implications

- The shipped sycophancy instrument **over-fires on long honest text**; its real-world FPR is higher
  than the self-vs-other kill-gate suggested (FPR 0.36→0.06 — that validation was likely on short /
  length-matched data, where the confound is invisible).
- Everything built on it inherits the FP: `styxx.parrhesia` (the receipt audit), the darkflobi reply
  reflex (moonshot #1), and any `gated_sycophancy_risk` caller.
- This is why the PARRHESIA public announcement must WAIT: a verifiable-honesty tool whose scorer
  flags long honest posts as sycophantic would be the exact overclaim styxx exists to catch.

## The proper fix (banked — it is a recalibration, not a one-line edit)

Zeroing the coefficient alone shifts the baseline (short-text 0.196 → 0.412) because the intercept was
fit *with* the length term. The disciplined fix: **refit the sycophancy weights WITHOUT `log_word_count`
(or with length explicitly decorrelated), re-fit the intercept, and re-run the self-vs-other kill-gate
on a LENGTH-STRATIFIED validation set** so the FPR holds across lengths, not just short ones. Same for
the overconfidence markers' false-precision firing on legitimate numbers ("1603 tests"), a sibling bug.

## The meta-point

The instrument that audits other models found a length confound in styxx's **own** shipped honesty
instrument — by being pointed at a real message instead of crafted test cases. That is the dogfood
thesis at its sharpest, and the reason the discipline (run it on yourself, on real inputs, before you
claim) is the product. Receipts: `self_directed_gate.py` (`_proba`, line ~154; feature line 95),
`calibrated_weights_sycophancy_v0.py:107`. Relates to [[the PARRHESIA ship]] and the program's standing
length-confound theme (attribution-depth length-partial; conscience-mount "length-filler" owed).
