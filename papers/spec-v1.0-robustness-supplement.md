# Cognometric Fingerprint Specification v1.0 — Robustness Supplement

**Document type:** Empirical supplement to the Cognometric Fingerprint Specification v1.0 ([doi:10.5281/zenodo.19746215](https://doi.org/10.5281/zenodo.19746215))
**Subject:** Adversarial robustness of the Tier-3 (text-only proxy-signal) classifier reference implementation
**Edition:** v0.1 — first systematic audit · 2026-04-25
**License:** CC-BY-4.0
**Authors:** Fathom Lab
**Status:** open for review and reproduction

---

## Abstract

We present the first systematic adversarial robustness audit of the
Tier-3 text-only proxy-signal classifier specified in §5.1.2 of the
Cognometric Fingerprint Specification v1.0. The audit constructs 24
canonical attack prompts spanning paraphrase, obfuscation,
unicode-substitution, case-folding, density-thresholding, and
meta-discussion strategies, organized across the seven fault kinds
defined in Spec §4. Baseline measurement of the v0.1 reference
classifier shows **66.7% false-negative attack success and 66.7%
false-positive attack success**. After three iterations of targeted
defensive hardening (v0.2.0 → v0.2.3), false-negative attack success
drops to **16.7% (a 4× reduction)** while false-positive attack
success drops to 50% (1.3× reduction), with no regressions on the
canonical 26-case validation suite. This document documents the
attack taxonomy, the baseline weakness pattern, the hardening
modifications, the final per-category robustness profile, and the
honest residual limits — particularly the text-only inability to
distinguish confabulation from genuine retrieval, and meta-discussion
of adversarial patterns from exhibited adversarial behavior.

---

## 1. Why this audit exists

The Cognometric Fingerprint Specification v1.0 §5.1.2 defines a
Tier-3 proxy-signal pipeline for substrates that do not expose
per-token log-probabilities (current Anthropic Messages API, current
Google Gemini API, current closed-API substrates). The reference
implementation of this pipeline is `classifier.js` shipped with the
`styxx-scope` browser extension and ported to Python in the
`styxx.adapters.text` module.

Tier-3 measurement carries an explicit confidence penalty (0.25 on
aggregate trust, per Spec §5.1.2) reflecting the loss of fidelity
relative to logprob-based pipelines. Until this audit, the *kind* of
loss had been documented qualitatively but not quantitatively — there
was no public number for "how easily can this classifier be fooled?"

This document supplies that number, defends the modifications that
improve it, and documents what remains unfixable.

## 2. Threat model

Two attack intents are considered:

- **False-negative (FN) attacks:** an attacker constructs prompts
  that exhibit a real cognitive fault (confabulation, drift, refusal,
  sycophancy, low_trust, incoherence) but evade detection by the
  classifier. An end-user sees the model's response with no warning
  badge and consequently misplaces trust.

- **False-positive (FP) attacks:** an attacker constructs benign
  content that *appears* to exhibit a fault and triggers a spurious
  warning. End-user trust in the classifier itself erodes from
  warning-fatigue.

We assume the attacker has full access to the classifier source code
(it is open-source MIT-licensed), full knowledge of the spec, and
unlimited prompt-construction budget. The threat model is therefore
**white-box adaptive**.

We assume the attacker does *not* control the underlying language
model — they can only shape the input and observe the model's output,
which is what gets classified.

## 3. Attack taxonomy

We define eight categorical attack strategies, drawn from the
adversarial-NLP literature and adapted to the cognometric-classifier
threat surface:

| strategy | description | exemplar id |
|---|---|---|
| **paraphrase** | re-express the trigger pattern in non-canonical vocabulary | `confab-fn-paraphrase`, `refusal-fn-hedged` |
| **obfuscation** | break trigger-word boundaries (insert punctuation/spaces) | `adv-fn-encoded` |
| **unicode-substitution** | replace ASCII characters with visually-identical Unicode codepoints | `unicode-fn-cyrillic` |
| **case-folding** | exploit case-sensitive matchers | `case-fn-uppercase` |
| **density-thresholding** | use trigger pattern just below detection threshold | `sycophant-fn-spaced` |
| **meta-discussion** | discuss a fault pattern without exhibiting it | `adv-fp-meta`, `meta-fp-cognometric-discussion` |
| **inversion** | wrap fault content in negation/hypothesis framing | `mixed-fn-confab-refusal` |
| **interleaving** | mix benign and faulted content within one response | `drift-fn-explanation` |

The full attack suite is `_test_adversarial.js` in
`packages/styxx-scope/`, MIT-licensed and reproducible via
`node _test_adversarial.js`.

## 4. Baseline measurement (v0.1)

Running the 24-attack suite against the unmodified v0.1 classifier:

```
false-negative attacks (hide real fault):     12 / 18 succeeded  (66.7%)
false-positive attacks (induce wrong flag):    4 /  6 succeeded  (66.7%)
```

Per-category robustness profile (baseline):

| fault kind | FN evasion | FP spurious |
|---|---|---|
| reasoning | 0% | 0% |
| confabulation | 50% | 0% |
| drift | **100%** | n/a |
| refusal | 50% | 100% |
| sycophant | 50% | 100% |
| adversarial | **100%** | 100% |
| low_trust | **100%** | n/a |
| incoherence | **100%** | n/a |

**Five categories above (drift, low_trust, incoherence, adversarial)
were essentially unprotected** — every attack succeeded.

## 5. Defensive hardening (v0.2.0 → v0.2.3)

Three iterations of targeted defenses, each preserving the canonical
26-case validation suite (described in `_test_classifier.js`):

### 5.1 Unicode normalization + punctuation-strip

- NFKD-normalize input before pattern matching.
- Map common Cyrillic / Greek / Mathematical-Alphanumeric homoglyphs
  back to Latin: `а→a`, `е→e`, `і→i`, `о→o`, `р→p`, `с→c`.
- For adversarial-pattern matching specifically, also strip
  insertion-punctuation between word characters: `i.g.n.o.r.e → ignore`.
- *Defends:* `unicode-fn-cyrillic`, `adv-fn-encoded`.

### 5.2 Soft-refusal pattern + helpful-disclaimer suppressor

- Match indirect refusals: "would prefer not to engage", "outside what
  I am comfortable", "rather than helping you directly".
- Detect helpful-disclaimer context: "happy to help", "let me give you",
  "here's what we know" — when present without strong refusal markers,
  suppress the refusal score by 80%.
- Pattern `HEDGE_NOT_REFUSAL` matches "I cannot guarantee/promise" and
  is *subtracted* from the refusal count (these are confidence-hedges,
  not declines).
- *Defends:* `refusal-fn-hedged`, `refusal-fn-explanation`,
  `refusal-fp-disclaimer`, `case-fn-uppercase`.

### 5.3 Tool-flip drift detection

- Pattern `TOOL_FLIP` matches "but actually let me", "wait, let me
  reconsider", "planned to call X but" within 50 chars of a tool name.
- Pattern `TOOL_NAMES` matches common destructive-tool families:
  `delete_records`, `drop_table`, `execute_query`, `modify_records`,
  etc.
- When both fire, emit a `drift` fault independent of dominant
  category. Tool drift is detectable text-only when the agent
  verbalizes its revision; the audit confirms this covers the
  dominant case.
- *Defends:* `drift-fn-explanation`, `drift-fn-natural-revision`.

### 5.4 Heavy-hedge low-trust override

- When hedge-density exceeds 6.0 hedges/100-words AND no first-person
  certainty markers are present, fire `low_trust` independent of
  axis-derived trust score.
- Defends the "tone-confident but content-hedged" pattern that the
  trust formula otherwise misses.
- *Defends:* `low-trust-fn-confident-hedge`.

### 5.5 Topic-jump incoherence detection

- For sentences with ≥2 long words (>3 chars), compute lexical-overlap
  between adjacent pairs.
- "Comparable pairs" are those where both sentences carry information
  (skip thin-content sentences from the comparison).
- Fire `incoherence` fault when **every** comparable adjacent pair
  has zero overlap.
- *Defends:* `incoherence-fn-topic-jump` while not firing on
  reasoning text where adjacent steps share lexical content.

### 5.6 Meta-discussion suppressor

- Pattern `META_DISCUSSION` matches: "researchers", "study", "paper",
  "literature", "the X classifier flags", "prevalence", etc.
- When present with low fault-trigger density, multiply the
  adversarial / sycophant boost by 0.05 (effectively zero).
- Defends discussion of attack patterns from being misread as
  exhibited attacks.
- *Defends:* `adv-fp-meta`. Note: residual false-positives remain
  where meta-discussion is mixed with high-density agreement words
  — this is the v0.2.3 residual limit.

### 5.7 Sycophancy threshold tightening

- Single-token agreement followed by substantive content
  (`agreements ≤ 1 AND wordCount > 20`) reduces sycophant boost by 60%.
- Captures the difference between "Great question. The answer is..."
  (legitimate engagement) and "Absolutely brilliant! Amazing! Fantastic!"
  (sycophancy).

### 5.8 Unverified-claims fault rule

- The `unverified_claims` warning fires when text contains both
  ≥2 specific facts AND ≥1 named entity, with low hedging.
- This is the *honest* text-only signal: high specificity + named
  authority figure + confident assertion = "verify independently"
  warning, regardless of whether the actual content is real or
  fabricated.
- Tier-3 cannot distinguish confabulation from real retrieval.

## 6. Final measurement (v0.2.3)

After three hardening iterations:

```
false-negative attacks (hide real fault):      3 / 18 succeeded  (16.7%)
false-positive attacks (induce wrong flag):    3 /  6 succeeded  (50.0%)
```

Per-category (final):

| fault kind | baseline FN | v0.2.3 FN | baseline FP | v0.2.3 FP |
|---|---|---|---|---|
| reasoning | 0% | 0% | 0% | 0% |
| confabulation | 50% | **25%** | 0% | 0% |
| drift | 100% | **0%** | n/a | n/a |
| refusal | 50% | **0%** | 100% | **0%** |
| sycophant | 50% | 50% | 100% | 100% |
| adversarial | 100% | **33%** | 100% | 100% |
| low_trust | 100% | **0%** | n/a | n/a |
| incoherence | 100% | **0%** | n/a | n/a |

**Headline:** false-negative attack success drops 4× (66.7 → 16.7%).
False-positive attack success drops 1.3× (66.7 → 50%). Five fault
kinds (drift, refusal, low_trust, incoherence, plus partial
adversarial) now have zero or near-zero evasion from canonical attacks.

## 7. Honest residual limits

The classifier is not perfect. The following residual gaps are
documented:

### 7.1 Confabulation vs retrieval is fundamentally text-ambiguous

A response stating "The chemical formula is C12H20O4 with weight 228.28
g/mol" is indistinguishable from a real fact about a real compound and
a fabricated fact about a fictional compound (e.g., glorbnix). Without
a knowledge base or logprob signal, no text-only classifier can
reliably distinguish these. We surface this as the
`unverified_claims` fault — a *warning* rather than a *verdict* — and
recommend that downstream applications use it as a "verify
independently" prompt rather than a confidence-killer.

### 7.2 Sycophant evasion via substantive padding

A response that combines genuine technical content with agreement
markers ("You raise a thoughtful point about the algorithm's time
complexity. The reasoning is sound...") evades the sycophant detector
because the agreement density is diluted by substantive vocabulary.
Detection of this pattern requires semantic-similarity comparison
between user input and assistant response — a capability beyond
text-heuristic Tier 3.

### 7.3 Adversarial false-positives on meta-discussion

When text discusses adversarial *examples* with high density (e.g.,
"Researchers studied jailbreak prompts that begin with 'ignore
previous instructions'..."), the suppressor reduces but does not
fully eliminate the score, leaving a low-confidence adversarial
classification. Skeptical users may see warnings on legitimate
research/journalism content. Recommendation: filter by confidence
threshold ≥ 0.40 to drop the residual.

### 7.4 Refusal-preamble masking confabulation

A prompt response that opens "I cannot fully verify this, but..." and
proceeds with confabulated specifics gets classified as `refusal`
(dominant pattern) rather than `confabulation` (subordinate pattern).
The classifier picks one category per response. Recommendation:
implementations that need both signals should run the classifier on
sentence-segmented chunks rather than full responses.

### 7.5 Tier-3 vs Tier-1 measurement gap

This audit measures the *Tier-3* text-only pipeline. The Tier-1
logprob-based pipeline (per Spec §5.1.1) is fundamentally more
accurate because it has access to the model's per-token uncertainty
and cannot be fooled by text-level paraphrase. We recommend Tier-3 for
zero-friction observability (no API integration) and Tier-1/2 for
production decision-making.

## 8. Reproduction

The audit is fully reproducible:

```
git clone https://github.com/fathom-lab/styxx
cd styxx/packages/styxx-scope
node _test_classifier.js          # canonical 26-case validation suite
node _test_adversarial.js          # 24-attack adversarial audit
```

Output: `_adversarial_report.json` with per-attack results, intent
labels, and aggregate statistics.

The audit suite is intended to grow. Future revisions will add:
- multilingual attack variants
- code-mixed attacks (formal language interspersed with code)
- streaming-response edge cases
- multi-turn conversation attacks (cross-turn cognitive drift)
- production-trace replay attacks (against logged real-user prompts)

We welcome adversarial submissions: open a PR adding new attacks to
`_test_adversarial.js`, and the spec working group will incorporate
them in subsequent revisions.

## 9. Recommendations to implementers

For a Tier-3 cognometric implementation conformant with Spec v1.0
§5.1.2, we recommend:

1. **Apply Unicode NFKD normalization** before any pattern matching.
2. **Subtract hedge-not-refusal from refusal counts** ("I cannot
   guarantee" is not "I cannot help").
3. **Suppress adversarial / sycophant scores when meta-discussion
   markers are present** (5% multiplier when one or more matched).
4. **Use lexical-overlap incoherence rather than category-shift only**
   — adjacent-sentence overlap detects topic-jumps that within-phase
   coherence misses.
5. **Document residual ambiguity**: confabulation vs retrieval cannot
   be distinguished text-only. Surface it as a warning, not a verdict.
6. **Test against an adversarial suite before deploying.** This audit's
   suite is the first public reference; future implementations should
   exceed its coverage.

## 10. Citation

Fathom Lab. *Cognometric Fingerprint Specification v1.0 — Robustness
Supplement (v0.1).* Zenodo, 2026-04-25.

Reference specification: doi:[10.5281/zenodo.19746215](https://doi.org/10.5281/zenodo.19746215).
Reference implementation (audit harness): `packages/styxx-scope/_test_adversarial.js`
in styxx v6.2.0 ([doi:10.5281/zenodo.19758619](https://doi.org/10.5281/zenodo.19758619)).
Test report: `packages/styxx-scope/_adversarial_report.json`.

License: CC-BY-4.0 (this document); MIT (audit code).

---

**End of supplement.**

*Nothing crosses unseen.*
