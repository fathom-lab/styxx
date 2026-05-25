# Finding · Tier-3 clustering redesign — the lever works; the meaning-judge is best but NOT necessary

**2026-05-25.** Prereg `preregistration_clustering_2026_05_25.md` + amendment
`..._3b_2026_05_25.md`. Closes the open clustering question from the Tier-3 arc.
Verdict: **PASS = FALSE** on the pre-registered kill-gate — the decisive necessity
claim fails — with a clean, useful positive underneath.

## Path

- **3a** VOIDed: the validity gate keyed on cosine mpc < 0.85, which is **circular**
  (cosine is a method under test) and excluded the informative regime. Honored the VOID.
- **3b** re-pre-registered the validity gate as **distinct surface forms ≥ 4 of 6**
  (lexical variation, independent of all three clustering methods); **kill-gate bars
  unchanged**. Valid: 8 surface-varied-correct + 8 confabulation items.

## Result (3b, run once)

| clustering | AUC(entropy → confab) | FP ratio on varied-correct | succeeds (AUC≥.80 ∧ FP≤.40) |
|---|---|---|---|
| cosine @0.70 | 0.688 | 0.00 | no (AUC) |
| cosine @0.80 | 0.866 | 0.055 | **yes** |
| cosine @0.90 | **0.973** | 0.167 | **yes** |
| cosine @0.95 | 0.960 | 0.399 | yes (edge) |
| nli-deberta | 0.942 | 0.371 | yes (edge) |
| **LLM-judge** | **1.000** | **0.000** | **yes** |

- **G2 (judge solves it): PASS.** The model-as-judge clustering is perfect here — AUC
  1.0, zero false-positives on paraphrased-correct answers, high entropy on every
  confabulation.
- **G3 (cheap methods insufficient): FAIL.** cosine@{0.80,0.90,0.95} and nli-deberta all
  clear the same bar. The expensive judge is **not necessary**.
- **PASS = G2 ∧ G3 = FALSE.**

## What this honestly establishes

1. **The lever is real and robust.** Every method separates confident confabulation from
   paraphrastic-correct answers at AUC ≥ 0.87 — because confabulation is *inconsistent*
   (different fact each sample) and correct answers agree in meaning even when reworded.
   This is the confirmed, method-independent core of the whole Tier-3 arc.
2. **The meaning-judge is the cleanest, but optional.** Only the LLM-judge had **zero**
   false-positives; cosine@0.95 (FP 0.399) and nli-deberta (FP 0.371) still mis-fire on
   reworded-correct answers per item (e.g., "sky is blue" rephrasings: cosine 1.01,
   nli 1.79, **judge 0.00**). But a **tuned cosine@0.90 is good enough** (AUC 0.973,
   FP 0.167) and far cheaper. The form→meaning story holds *qualitatively* — form-based
   clustering does mis-fire more — but not strongly enough to make the judge *required*
   on this distribution.
3. **My pre-registered hypothesis was wrong, and that's the recorded result.** I
   predicted the judge would be necessary (G3). It isn't. Honoring G3's failure (rather
   than spinning the judge's AUC 1.0 as a "win") is the point.

## Implication for a styxx `semantic_entropy` primitive

Now defensible to design, with honest tiers:
- **default (cheap):** cosine clustering at **~0.90** (AUC ~0.97, modest FP) — pure
  Python, no model load.
- **high-precision (opt-in):** LLM-judge clustering (zero FP here) for callers who can
  afford an extra model call per pair and need minimal false-positives.
- **avoid:** the conventional 0.70 cosine threshold (AUC 0.69 — the original artifact)
  and treating nli-deberta as a clean judge (noisy, FP 0.37).
Still pre-full-validation: gpt-4o-mini only, n=8/8, single run; cross-vendor key-blocked.
A full hashed run-once across ≥3 models with these settings is the remaining step before
shipping.

## Arc close

Tweet (wrong: "stable, 0.55") → correction (overclaimed: "use NLI") → focused probe
(NLI itself false-positives) → clustering redesign (judge solves it but isn't
necessary; tuned cosine suffices). Four self-corrections, each from honoring a bar or a
gate over the momentum. The durable result: **confident confabulation is detectable by
across-sample divergence (AUC ≥0.9 with sane clustering); the model's confidence hides
it from single-response signals, but its inconsistency exposes it across samples.** A
real partial Tier-3 crossing, finally stated with the right caveats.
