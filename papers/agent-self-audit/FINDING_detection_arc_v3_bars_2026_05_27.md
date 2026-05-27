# Finding · three detection-axis submissions under v3 bars — n=3 narrowing of where the dark core can be cracked

**Date:** 2026-05-27 · **Status:** session-arc summary of Baselines 008-010 under v3 (D1+D2+D3+D4) bars. Reads alongside [FINDING_gauntlet_artifact_discovery_2026_05_27.md](FINDING_gauntlet_artifact_discovery_2026_05_27.md) (the D3 discovery) and [FINDING_confound_audit_2026_05_27.md](FINDING_confound_audit_2026_05_27.md) (the D4 discovery + seventh in-session falsification). **Records the eighth in-session falsification** (Baseline-010's direction-of-effect miss) and consolidates what the three pre-registered submissions ruled out empirically.

> **Outcome.** Three detection methods submitted with pre-stated predictions committed to origin BEFORE each gauntlet run: embedding similarity (Baseline-008, 3/4), length-residualized embedding (Baseline-009, 1/4), NLI entailment (Baseline-010, 0/4). **None cleared D3.** The v3 bars (D1+D2+D3+D4) now have a documented receipt that surface-form variants — raw cosine, length-corrected cosine, and pre-trained NLI cross-encoder — all fail to demonstrate signal genuinely orthogonal to the benchmark's known confounds. **The remaining frontier requires information beyond surface text:** cross-vendor consensus disagreement, perplexity against a prior, or knowledge-graph lookup. All operator-territory at present.

## The three submissions, side-by-side

| baseline | method | bars passed | D1 / D2 | D3 (length) Δ | D4 (cap-ratio) Δ | pre-stated modal | prediction shape |
|---|---|---|---|---|---|---|---|
| **Baseline-008** | `all-MiniLM-L6-v2` cosine | **3/4** | 0.805 ✓ / 0.928 ✓ | 0.015 ✗ | 0.102 ✓ | "pass D1+D2, fail D3" (60%) | **modal validated** |
| **Baseline-009** | Baseline-008 with length-residualization | **1/4** | 0.645 ✗ / 0.775 ✓ | −0.145 ✗ | −0.059 ✗ | "D2 only" (30%) | **modal validated**, 3/4 AUCs in range |
| **Baseline-010** | NLI cross-encoder entailment | **0/4** | 0.412 ✗ / 0.382 ✗ | −0.378 ✗ | −0.292 ✗ | "total fail" (20%) | band validated, **direction FALSIFIED** |

### The narrative shape

The three submissions trace a deliberate research-arc:
1. **Embedding similarity** (008) demonstrates the raw signal that's structurally equivalent to length on the broad axis but adds 12.4pp on the folklore axis. Predicted to fail D3 modally; did.
2. **Length-residualized embedding** (009) tests the "obvious smart fix" everyone would propose. Predicted to fail wider than 008 because residualization subtracts signal rather than adding new signal; confirmed — D1 fails outright, D2 still passes, but D3 and D4 both fail by larger deltas.
3. **NLI entailment** (010) attempts a genuinely length-independent signal. Predicted ~15% PASS but ~20% total fail; landed in the total-fail band with the further surprise that direction-of-effect was inverted from the hypothesis.

**No PASS in the arc.** The seven-method floor + dark core remain intact for the n=11 method-class case (seven from the original arc + four (Baseline-002, 007, 008, 009, 010 — five really) externally-public-callable methods that exercised the gauntlet).

## What the v3 bars have empirically ruled out

After three pre-registered submissions under D1+D2+D3+D4, the following method-classes are documented as failing on this benchmark:

1. **Surface-form lexical features** — token overlap (Baseline-007, fails D3) + classical TF-IDF (Baseline-005/006, fail K1/K3 on classification).
2. **Raw semantic embedding similarity** — sentence-transformer cosine (Baseline-008, fails D3 by 0.085pp).
3. **Length-corrected embedding similarity** — Baseline-009 fails D3 by 0.145pp (residualization actively makes the gap wider).
4. **Pre-trained NLI cross-encoder** — Baseline-010 fails D1/D2 even before reaching the control bars; |AUC−0.5| caps at ~0.12.

That covers a substantial portion of the "cheap surface-text" method space. Any future submission proposing a surface-text-only method has the empirical context that the four obvious approaches have been exhausted; novelty would need to come from a different feature axis.

## What the bars have NOT yet been tested against

The detection frontier under v3 bars is now clearer. Methods that could plausibly clear D3+D4 must use information beyond the response's surface text alone:

| method class | dependency | predicted PASS probability |
|---|---|---|
| **Cross-vendor consensus disagreement** — run the council (>1 different-vendor models) on the question, score by inter-vendor disagreement | vendor keys for ≥2 of {Anthropic, OpenAI, Google} | substantial; this is the original JD/CVPD direction from the seven-method arc |
| **Perplexity against a prior LM** — score how surprising the response is given the question, conditioned on a calibrated reference distribution | a small local LM (gpt2 / phi-3 / qwen-0.5B) | uncertain; depends on whether LM prior aligns with class boundary |
| **Knowledge-graph lookup** — query Wikidata / DBPedia / a curated KG, score response-KG agreement | KG access + entity-linker | high in principle but operator-territory; multi-hour build |
| **Fine-tuned classifier on (q, r) pairs** — retrain Baseline-002's architecture jointly on (q, r) inputs instead of q-only | a held-out training corpus | uncertain; risks length-leakage at training time |

The arc's empirical contribution to the project is that the *cheaper* methods are exhausted; future bets on the detection axis need to invest in operator-territory resources to even be tested.

## The eighth in-session falsification

Baseline-010's direction-of-effect miss is the eighth pre-stated-then-falsified prediction of the 2026-05-27 arc:

1. C1-profile ≤0.20 register-law bar — falsified by C10's 0.264 score (Pareto finding)
2. set_session-doesn't-propagate observation — falsified by per-agent routing
3. ICT-folklore auto-verdict PASS (probe label bug)
4. ICT-authoritative auto-verdict PASS (same label bug shape)
5. styxx 7.7.5 wheel bundling miss
6. The gauntlet's v1 detection bars being length-gameable (D3 discovery)
7. The benchmark's cap-ratio orthogonal confound (confound audit) — and my audit-code direction blind-spot
8. **Baseline-010's NLI direction — predicted folklore entails question, actual was the reverse**

Two of those eight (7 and 8) are direction-of-effect misses on this same domain. The pattern is clear: **on this benchmark, direction-of-effect predictions are systematically harder than magnitude predictions.** The discipline lesson for future pre-states: always include `|AUC − 0.5|` as the prediction range, treat direction as a separate sub-prediction with its own probability, and never rely on a single-direction threshold check in audit code.

## Calibration record (across all pre-stated baselines this session)

| baseline | predictions | inside-range count | outcome-band probability | call |
|---|---|---|---|---|
| Baseline-006 | char TF-IDF identical metrics | 4/4 (exact-tie predicted) | n/a — binary | confirmed |
| Baseline-008 | 5 AUC-range + 4 outcome bands | 5/5 ranges; outcome in 60% band | 60% | well-calibrated |
| Baseline-009 | 6 ranges + 5 outcome bands | 4/6 ranges (1 just at boundary); outcome in 30% band | 30% | well-calibrated |
| Baseline-010 | 6 ranges + 6 outcome bands | 0/6 ranges (direction wrong); outcome in 20% band | 20% | band-correct, direction wrong |
| Confound audit | 8 features × 2 partitions | 3/16 ranges; 3 joint predictions falsified | n/a | poorly calibrated on individual features; band-correct on shape |

**Calibration summary:** the simpler the predicted distribution (binary or single-band), the more often confirmed. AUC-range predictions are roughly two-thirds reliable. **Direction-of-effect predictions are systematically the worst** — two of the most consequential falsifications this session were direction misses on the same benchmark. This domain-specific lesson is now durable for future bets.

## The discipline pattern as the moat

Each pre-stated-then-published baseline is a small piece of immovable evidence about what works and what doesn't. The leaderboard now has 10 reference rows, all with pre-state-able predictions and outcome records. **Anyone can reproduce every result** via `styxx gauntlet --method <spec> --task detection`. The chain compounds:

- D3 discovered by accident (Baseline-007)
- D4 discovered by deliberate scan (`audit_confounds`)
- v3 bars rejected three pre-stated submissions in a row, each in the predicted modal region

The next phase of the arc requires resources beyond surface-text methods. That's what the v3 bars are doing — they're not just rejecting submissions, they're *forcing the research budget toward more substantive features*. Methods that can't reach the dark core through cheap surface variation now must declare what new feature axis they're bringing.

## Reproducibility

- Three pre-registered submissions: `submissions/baseline_008_embedding_similarity/`, `submissions/baseline_009_residualized_embedding/`, `submissions/baseline_010_nli_entailment/`.
- Pre-stated predictions: `submissions/<each>/PRE_STATED_PREDICTION.md` (each committed BEFORE its gauntlet run on the public branch).
- Results: `submissions/<each>/submission.json`.
- LEADERBOARD rows: this commit.
- Gauntlet code under v3 bars: `styxx/gauntlet.py::run_detection_gauntlet`, `DEFAULT_DETECTION_BARS`.
- This finding: this commit.

`git log --oneline submissions/baseline_008* submissions/baseline_009* submissions/baseline_010*` shows the prediction-before-data ordering on origin for all three.

## Open follow-ups (operator-territory, ranked by leverage)

1. **PyPI / git tag / Zenodo for 7.7.9.** The D4 bar + audit primitive + three pre-stated baselines + this FINDING are all currently `main`-only. Until PyPI publish, external researchers can't reproduce the v3 leaderboard with `pip install`. Operator-territory: needs `secrets/pypi-token.txt` for upload.
2. **Cross-vendor council re-elicitation** — the JD/CVPD direction from the seven-method arc, now testable as a Baseline-011 submission. Needs ≥2 vendor keys (`OPENAI_API_KEY` + `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`). Estimated PASS probability ~30-40%.
3. **Benchmark v2 with format-equalized truth responses.** Re-curate the 55 truth records with full-sentence canonical answers matching the folklore class's length distribution. Removes both the length and cap-ratio confounds at the source (rather than controlling for them via D3+D4). Multi-hour curation work.
4. **Perplexity-based detector** — a Baseline-011-alt using a small open LM. Cheaper than (2) but lower expected signal.
