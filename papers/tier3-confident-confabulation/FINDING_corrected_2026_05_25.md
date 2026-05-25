# Finding (CORRECTED) · Tier-3 — semantic entropy DOES detect confident confabulation

**2026-05-25.** This supersedes `FINDING_2026_05_25.md`, whose headline ("semantic
entropy does not cross confident confabulation, AUC 0.55") was a **cosine-clustering
artifact**. The artifact was caught the same day by the cross-paraphrase follow-up and
confirmed by `verify_clustering.py`. Honest correction; the original is preserved,
flagged, and not to be cited.

## What went wrong

Semantic entropy (Farquhar et al., Nature 2024) has two steps: (1) sample N answers,
(2) cluster them **by meaning** and take entropy over cluster proportions. The
published method clusters by **bidirectional NLI entailment** ("two answers are the
same iff each entails the other"). Our probe substituted a cheap proxy for step 2:
**cosine similarity > 0.70** on sentence embeddings.

That proxy fails exactly where it matters. When the model confabulates, it emits the
same *sentence template* with a different *fact*:

```
"Captain Aldous Renwick first reached the Sundering Isles in 1842."
"...in 1723."   "...in 1745."   "...in 1912."   "...in 1883."   "...in 1754."
```

Six samples, six different years. Cosine similarity between these is ~0.97 (one token
differs) → all clustered as "the same answer" → entropy ≈ 0 → a *fake* flatline that
we misread as "the model tells the same lie every time / confident error is stable."
NLI entailment correctly refuses to entail "1842" from "1723" → six clusters → high
entropy. Same for the Pemberton Prize (six different laureates) and "epistemic
verdancy" (six different philosophers).

## Corrected result (`verify_clustering.py`, 16 items, N=6, identical samples)

| clustering method | AUC(entropy → confabulation) | mean entropy: confab / correct |
|---|---|---|
| cosine > 0.70 (the artifact) | **0.573** | 0.11 / 0.05 |
| **NLI bidirectional entailment** | **0.948** | **1.73 / 0.33** |

The cosine number replicates the original 0.55 on fresh samples (the bug is stable).
Done properly, **semantic entropy separates confident confabulation from correct
answers at AUC ≈ 0.95.**

## Corrected conclusion

- **Confident confabulation is INCONSISTENT, not stable.** The model is confident
  *and wrong* on each sample, but it does not commit to one falsehood — it generates a
  fresh one each draw. The earlier "stable false belief" reading was the proxy talking.
- **The across-sample substrate partly CROSSES Tier 3.** Single-response confidence
  (logprob) is closed on hallucination (grounded-arc, ρ≈0). But across-sample semantic
  divergence — clustered by entailment — *does* flag it. This is the program's first
  partial Tier-3 crossing.
- **Methodological contribution (the durable lesson):** embedding-cosine clustering is
  not a valid stand-in for entailment clustering in semantic entropy. It systematically
  *under*-counts confabulation divergence (lies share a template) and can *over*-count
  benign phrasing variation. If you proxy the clustering step with cosine, you can
  manufacture a null result. Use NLI.

## Honest residuals (do not overclaim the 0.95)

- **n = 4** confidently-confabulated items; **gpt-4o-mini only**. This is a feasibility
  probe, not a validated instrument. The pre-registered full hashed run-once is now
  *warranted* (the probe clears the spirit of T1/T2), but has **not** been run.
- **False-positive mode:** items where the model *flip-flops* between abstaining and
  confabulating across samples (florium, zylophane) carry high entropy while their
  modal answer is a correct abstention → scored "correct" but flagged by the lever.
  This is the AUC's gap from 1.0 (0.948) and a real failure mode any deployed version
  must handle (separate "uncertain among fabrications" from "uncertain among phrasings
  of "I don't know"").
- The abstention detector itself was a confound source in the original probe (it
  miscoded "not a recognized element" as a confabulation); hardened here.

## Next (pre-registration required, not done yet)

Full run-once: more items, ≥3 models (OpenAI family — cross-vendor still key-blocked),
N≥10, entailment clustering, a pre-registered AUC bar, hashed holdout. Plus a designed
treatment of the abstain/confabulate flip-flop FP. Then — if it holds — a styxx
`semantic_entropy` primitive (opt-in, grounded tier, `styxx[nli]`), shipped with the
cosine-proxy trap documented so nobody re-introduces it.
