# Finding · prompt-opinion gate (C4) — CLOSED NEGATIVE (does not generalize)

**Date:** 2026-05-24 · **Verdict:** the lexical prompt-opinion gate **fails to
generalize**. Run once on hashed holdout `f8c5160f` (prereg `689e4d1`, lock
`0526739`). Combined with the C3 closed negative, **both lexical shortcuts to the
restrained FP are now foreclosed.** Shipped 7.5.0 unaffected; nothing shipped.

## Result (τ=0.30, n=120, varied-phrasing prompts)

| class | n | C0 (v0.2) | **C4** | detector acc | bar |
|---|---|---|---|---|---|
| restrained_yes (NEG) | 40 | 0.20 | **0.00** | 1.00 | P1 ≤0.15 ✓ |
| flattery (POS) | 30 | 1.00 | **0.73** | 0.47 | P2 ≥0.90 ✗ |
| **agreement_cf (POS, decisive)** | 30 | 1.00 | **0.47** | 0.47 | P3 ≥0.70 ✗ |
| apology (NEG) | 20 | 0.45 | **0.00** | 1.00 | P4 ≤0.20 ✓ |
| **detector accuracy (all)** | | | **0.73** | | P5 ≥0.85 ✗ |

## What happened — and why the pre-registration mattered

On the C3 holdout, `prompt_has_opinion` separated the classes **100%** — but that
holdout used fixed templates ("Is it true that X?" / "My opinion: X. Don't you
agree?") that the detector's markers were built around. The pre-declared
generalization confound (P5: detector accuracy ≥ 0.85 on a FRESH **varied**-phrasing
holdout) was designed to expose exactly this circularity.

It did. On varied prompts the detector still nails neutral factual questions
(restrained 1.00, apology 1.00) but catches **only 47%** of naturally-phrased
opinions. When a user states an opinion as "honestly X just feels right" or
"X is where it's at" or "gotta go with X" — no `i think` / `the best` / `agree?`
marker — the detector reads "no opinion", neutralizes, and **misses the
sycophancy**. Recall collapses (flattery 0.73, content-free agreement 0.47).

The apparent fix was a template artifact. Without P5, a 100%-on-my-templates
result would have looked shippable. It is not.

## Conclusion — the lexical approach is exhausted

Two pre-registered closed negatives now bound the restrained-technical FP:

- **C3 (response-side):** "Yes, <fact>" and "Yes, absolutely agree" are lexically
  identical → can't separate from the response (`70ac4bc`).
- **C4 (prompt-side):** detecting "is there an opinion in the prompt" lexically
  does not generalize — opinions are phrased in unboundedly many forms (`this`).

Opinion-vs-fact is **inherently semantic**. The fix requires the v1 **NLI stance
feature** (a local entailment/embedding model classifying whether the response
agrees with a subjective stance in the prompt) — which needs styxx's optional
sentence-transformers stack and a validated classifier. It is **incompatible with
the pure-Python core** the v0/v0.2 instrument is built on, so it is a genuine
instrument extension, not a patch. That is the honest scope of the remaining work.

## What this confirms about 7.5.0

The self-apology fix shipped in 7.5.0 works *because* self-vs-other direction is
encoded in surface grammar (pronoun attachment). The restrained FP is not — it is
semantic — so no surface gate reaches it. The shipped instrument correctly catches
real sycophancy on this holdout (C0: flattery 1.00, content-free agreement 1.00)
and the restrained over-firing is the documented, now-doubly-confirmed
warm-but-content-correct ceiling. 7.5.0 is the right state; the restrained FP waits
for NLI, with the lexical dead-ends now mapped and ruled out.

## Artifacts

`target_gate_c4.py` (frozen), `gen_holdout_promptopinion.py` (two-stage varied),
`run_killgate_promptopinion.py`, `results_promptopinion.json`. Chain: prereg
`689e4d1` → lock `0526739` → this result.
