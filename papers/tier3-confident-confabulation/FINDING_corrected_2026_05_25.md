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

That proxy fails because the model's distinct fabrications still share surface form.
When it confabulates it reuses a sentence template with a different fact:

```
"Captain Aldous Renwick first reached the Sundering Isles in 1842."
"...in 1723."   "...in 1745."   "...in 1912."   "...in 1883."   "...in 1754."
```

Six samples, six different years — and for the Pemberton Prize six different laureates,
for "epistemic verdancy" six different philosophers. Across the confabulating items the
samples' **pairwise cosine is 0.70–0.93 (mean 0.78)** — at or above the conventional
**0.70** clustering threshold, so they all merge into one cluster → entropy ≈ 0 → a
*fake* flatline we misread as "the model tells the same lie every time / confident error
is stable." (The Renwick years are the extreme case at ~0.97; the aggregate is lower.)

## Corrected result — and what actually drove the artifact

`verify_clustering.py` (16 items, N=6) + `analyze_clustering_threshold.py` (a cosine-
threshold sweep on the *identical* cached samples — no new API):

| clustering | AUC(entropy → confabulation) | note |
|---|---|---|
| cosine > 0.50 | 0.500 | everything merges |
| cosine > 0.70 (what we shipped) | **0.573** | the published artifact (replicates 0.55) |
| cosine > 0.80 | 0.844 | |
| cosine > 0.95 | **0.927** | most of the signal is recoverable |
| **NLI entailment (threshold-free)** | **0.948** | the published method, marginally best |

**The dominant cause was the threshold, not cosine itself.** At 0.70 the lies merge
(they're 0.70–0.93 similar); raise the cutoff and AUC climbs to 0.93 by 0.95. NLI beats
the *best* cosine setting by only 0.02. NLI's genuine advantage is that it is
threshold-free and robust when *correct* answers are paraphrastically diverse — but
**this probe does not test that**: the model answered the real facts near-verbatim
(pairwise cosine 1.000), which is exactly why a high cosine threshold happens to keep
them together. With paraphrased-but-correct answers, cosine@0.95 would split them
(false positives) where entailment would not.

**What is robust to all of this:** semantic entropy separates confident confabulation
from correct answers at **AUC 0.93–0.95** (any cosine threshold ≥ 0.8, or NLI). The
model confabulates *inconsistently*. The original "stable / flatline / AUC 0.55" was
specifically an artifact of the conventional 0.70 threshold.

## Corrected conclusion

- **Confident confabulation is INCONSISTENT, not stable.** The model is confident
  *and wrong* on each sample, but it does not commit to one falsehood — it generates a
  fresh one each draw. The earlier "stable false belief" reading was the proxy talking.
- **The across-sample substrate partly CROSSES Tier 3.** Single-response confidence
  (logprob) is closed on hallucination (grounded-arc, ρ≈0). But across-sample semantic
  divergence — clustered by entailment — *does* flag it. This is the program's first
  partial Tier-3 crossing.
- **Methodological lesson (refined after a sensitivity sweep):** the conventional
  cosine threshold (~0.70, common in SelfCheck-style implementations) is far too lenient
  for confabulation — the divergent answers still share a template, so they merge and
  you manufacture a null. Raising the threshold recovers most of the signal *here*;
  entailment clustering is threshold-free and marginally better, with a robustness edge
  (when correct answers are paraphrastically diverse) that this probe does not test.
  The durable rule: **don't trust a single clustering setting — sweep it, and prefer
  entailment.** (Meta-lesson: the first "correction" of this finding *also* overclaimed
  its mechanism — "cosine can't do it" — until this sweep. Check sensitivity before
  asserting cause, in both directions.)

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
