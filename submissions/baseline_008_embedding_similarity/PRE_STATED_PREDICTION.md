# Pre-stated prediction — Baseline-008 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-008.

## What is being tested

After Baseline-007 (token-overlap detector) exposed a length-confound in the benchmark's `expected_consensus` field, the 7.7.8 release added a D3 length-control bar (detector must beat length-only-oracle AUC by ≥ 0.10 on both partitions). Baseline-008 is the first non-trivial detection submission tested under v2 bars (D1, D2, D3 all required).

The bet: does **semantic similarity** between question and response embeddings add signal beyond the length artifact? Or is "Q-R similarity" structurally equivalent to "length / lexical overlap" — same exploit in a smarter feature space?

## The detector

Sentence-transformer (all-MiniLM-L6-v2) embeddings of question and response, cosine similarity as the score. Same embedding model used by Baseline-002 (the classification baseline) and the styxx semantic-sycophancy tier.

## Pre-stated bars (the prediction)

| outcome | bar | probability |
|---|---|---|
| **Clear D3 cleanly** | D1−length AUC ≥ 0.10 AND D2−length AUC ≥ 0.10 | **~15-25%** |
| **Pass D1 + D2 but fail D3** | D1 ≥ 0.70 AND D2 ≥ 0.70 AND D3 fails | **~60%** |
| **Fail any of D1, D2** | — | **~15%** |
| **D1 + D2 AUC magnitude** | ≥ 0.80 (similar shape to Baseline-007's length-correlated AUC) | ~70% |

## Honest reasoning

The hypothesis under which the detector clears D3: misconception responses share *semantic content* with the question in ways that length-alone cannot capture. Walt Disney's body being "interred at Forest Lawn" semantically restates the question's subject in ways that a 6-word answer to "What is the capital of France?" ("Paris is the capital of France") would also do — so the length-controlled signal may exist but be small.

The hypothesis under which D3 fails: Q-R similarity is just a smarter version of length. Long responses that echo the question's subject have high cosine similarity by construction; short answers don't. The benchmark's structural confound (truth responses 3.9 words, misconception 7.5 words) carries through to embedding-similarity unchanged. **My honest prior is this scenario (~60%).**

The third scenario (D1/D2 fail): unlikely. Even the dumbest detectors are passing the AUC≥0.70 bars on D1/D2 because the length artifact is that strong.

## Why this bet matters either way

- **PASS:** the first real PASS on the gauntlet. Semantic similarity adds signal beyond length. The synthesis gets its first deployable positive on the detection axis. Real news.
- **FAIL D3 only:** n=1 evidence that "anything measuring Q-R correlation hits the same artifact as length." The dark core stays dark to similarity-based detection. Confirms the D3 bar is correctly calibrated — semantic methods need to genuinely escape the artifact, not just smarter-encode it.
- **FAIL D1/D2:** would be the most informative — semantic similarity *anti-correlates* with class, opposite the length pattern. Would suggest the embedding captures something interesting that the artifact doesn't.

## Not re-running, not re-tuning

Configuration locked: all-MiniLM-L6-v2 (no other model considered), cosine similarity (no other similarity metric), no preprocessing, no length normalization. Run once on the bundled benchmark.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-008. Verifiable from git history.
