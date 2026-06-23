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

---

## ADDENDUM (same day) — the family picture is real but MESSIER than a clean confound

Extending the audit across the calibrated guardrail family found length is a load-bearing feature in
**4** models (static coefs): deception_v0 **−2.089** ("K=1 critical"), overconfidence_v0 −0.523,
sycophancy_v0 +0.356, refusal_v2 −0.185 (goal-drift / plan-action / loop / core v2-v4 have none).

Behavioral, content-held-constant (distinct sober passages, repetition-controlled):
- **sycophancy**: clean length confound (0.20→0.67); ablating the length weight flattens it — SOLID.
- **overconfidence**: clean length sensitivity (0.95→0.34) — SOLID.
- **deception**: in a "describe-X" regime, 0.999 (12w) → 0.010 (135w) — huge. BUT a matched-length
  completion-status probe ('did you finish?') scored honest AND deceptive, short AND long, all ~0.99–1.0
  (separation ≈ +0.003) — i.e. the instrument is **regime-dependent and saturates / fails to
  discriminate** elsewhere, NOT a single clean length monotone. (deception_check v0 is the lexical model
  already known not to generalize — project_deception_v1_negative.)

**Honest verdict (not overclaimed):** length is a real, often-spurious, load-bearing feature across
≥3 shipped instruments, and the deception instrument is unreliable across length × prompt regimes on
real text. This is a robustness problem in the calibrated guardrail family — NOT a clean one-line
"systemic length confound" bombshell (a matched-length probe disproved the clean version). The
responsible next step is a **regime-stratified robustness audit** (length × prompt-type × label) across
the family, then length-robust recalibration. The sycophancy + overconfidence length confounds are
clean enough to fix now; deception needs the hardest look (or deprecation of the v0 lexical model).

The meta-point holds and is sharpened: dogfooding surfaced the problem, and the SAME discipline
(one more matched-length probe) stopped me shipping the overclaimed clean version of it.
