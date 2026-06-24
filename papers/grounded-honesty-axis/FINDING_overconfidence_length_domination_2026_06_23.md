# Finding — overconfidence v0 is substantially a length detector (2026-06-23)

**TL;DR.** Extending the [sycophancy length-confound fix](FINDING_sycophancy_length_confound_2026_06_23.md) to
the next guardrail instrument, I tested whether `overconfidence_v0` carries the same droppable length confound.
It does **not** behave the same way: overconfidence's signal is *dominated* by length, not confounded at the
margin. The sycophancy fix-recipe (drop the length feature, recenter the operating point) **does not transfer**.
The honest fix is at the corpus level, and is currently blocked on a valid generation key.

## What I tested

`overconfidence_v0` has **two** length features — `log_word_count` (coef −0.523) and `mean_sentence_length`
(coef −0.311). The shipped module already discloses that `mean_sentence_length` is the **K=1 critical feature**
("a length confound that picks up on calibrated responses being longer"). On the shipped n=200 corpus
(`benchmarks/data/overconfidence/pairs_v0.jsonl`), 5-fold CV-AUC:

| feature set | CV-AUC | Δ vs full |
|---|---|---|
| full (9 features) | 0.7702 | — |
| drop `log_word_count` (8) | 0.7682 | **−0.0020** |
| drop **both** length features (7) | 0.7167 | **−0.0535** |

Best single-feature AUCs: `mean_sentence_length` **0.730**, `log_word_count` **0.702**, `epistemic_balance`
0.689, `unhedged_claim_ratio` 0.646. **The two most predictive features in the entire instrument are both
length**, and the single best is `mean_sentence_length` (the K=1 critical feature).

Reproduce (offline, no API): `python scripts/overconfidence_lenmatched_probe.py` (Part A).

## Why the sycophancy recipe does not transfer

For **sycophancy**, `log_word_count` was a *non-critical* feature carrying ~0 real signal — dropping it cost
0.9805→0.9795 AUC (free), because the real discriminator (`superlative_density`) is length-independent. Removing
length there was a clean win.

For **overconfidence**, length *is* the dominant signal. Dropping both length features costs **−0.053 AUC** on an
instrument whose headline (0.7702) is already the lowest in the suite. A 7-feature length-invariant refit would
land at ~0.717 — and shipping that as a "fix" would degrade a DOI'd instrument
(*Every Mind Leaves Vitals*, DOI 10.5281/zenodo.19777921) while overclaiming an improvement. So I did **not**
ship an overconfidence v0.2, and did not flip any default.

## Root cause — same as sycophancy, but load-bearing

Identical mechanism to the sycophancy confound: the v0 training corpus's `SYSTEM_CALIBRATED` stance prompt
("scale your certainty… acknowledge uncertainty where evidence is partial", in 2–4 sentences) invites
elaboration, so **calibrated responses run systematically longer**. Length then correlates with the calibrated
label. The difference is *degree*: for sycophancy this was a minor passenger feature; for overconfidence it is
the primary thing the logistic fit learned. The construct is confounded **at the corpus level**, so it cannot be
repaired post-hoc by dropping a coefficient — the labels themselves carry the length signal.

## The real fix (designed, blocked on credentials)

The correct experiment is to **length-match the corpus at generation time**: regenerate the same 100 questions ×
2 stances with an *identical* length constraint on both ("exactly 3 sentences, ~55 words"), so length cannot
proxy the label, then ask whether the epistemic-register features (certainty / hedge / epistemic_balance /
sourcing) still separate the stances:

- **If epistemic signal survives** (no-length AUC stays ≥ ~0.68 with length matched) → overconfidence *is*
  detectable length-free; train v1 on the matched corpus and ship a genuinely length-robust instrument.
- **If it collapses toward chance** → overconfidence-as-trained in this paradigm is largely a length artifact —
  a substantial honest finding about a shipped instrument, and a scope correction for the suite.

This is implemented and ready (`overconfidence_lenmatched_probe.py`, Part B) but **blocked**: the environment's
`OPENAI_API_KEY` is rejected (401) and there is no key in `clawd/secrets/` or a Gemini key in env. Operator
action: refresh the OpenAI key (or provide a Gemini key) and re-run — the probe auto-runs Part B once a valid key
is present.

## Status

- Offline audit committed + reproducible. No instrument changed; no overclaim shipped.
- `overconfidence_v0` left as-is (its length confound is already honestly disclosed in-module).
- Owed: the length-matched regen (blocked on a key); `deception_v0` is the next, hardest length-fragile case.
- Discipline note: this is the second time the sycophancy recipe was *tested* before being assumed — and the
  second time the test changed the conclusion. Sycophancy was a clean fix; overconfidence is a corpus-level
  problem. Same root cause, different remedy.
