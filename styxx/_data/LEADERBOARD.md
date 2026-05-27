# styxx gauntlet — public leaderboard

**The empirical floor is the bar. Submit your method. Beat us or join the floor.**

This leaderboard tracks external submissions to `styxx gauntlet` (shipped in styxx 7.7.5). Anyone can run their detection or classification method against the labeled benchmark (`papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json`) and submit results here via PR.

The bars are pre-registered and locked. We assert we couldn't beat them with the seven methods we tested. If you can, the synthesis gets revised — and you get the top of the board.

---

## How to submit

1. **Write your method as a Python callable.** Two task modes are supported:
   - **Classification** — `def predict(question: str) -> dict` returning `{"class": "folklore" | "pseudoscience" | "factual-error" | "truth"}`.
   - **Detection** — `def detect(question: str, response: str) -> dict` returning `{"score": float}` (higher = more misconception-like).
2. **Run the gauntlet locally.**
   ```bash
   pip install styxx>=7.7.5
   styxx gauntlet --method your_module:predict --task classification --name "your-method-name" --format json > result.json
   ```
3. **Open a PR** adding one row to the table below. Include:
   - Method name + your name / affiliation.
   - The `result.json` from the gauntlet run (attach as `submissions/<name>.json`).
   - A short description of the method and any external dependencies.
   - A link to your code (repo or PR-attached).
4. **CI re-runs the gauntlet** on your method against the locked benchmark. If your scores match what you submitted, the PR is mergeable. If not, we report the discrepancy.

External submissions must use **the same benchmark JSON** as the baseline. No corpus modifications; the benchmark is the bar.

---

## The bars

### Classification (3 bars; PASS = all three)

| bar | threshold | description |
|---|---|---|
| **K1** folklore F1 (in-distribution) | ≥ 0.70 | binary F1 on folklore-vs-not in the in-distribution test split |
| **K2** 4-way accuracy | ≥ 0.65 | accuracy across all four classes; must beat the majority baseline (~0.51) |
| **K3** folklore F1 (cross-corpus) | ≥ 0.60 | binary F1 mixing in-dist negatives + curated-folklore positives; the load-bearing bar |

Bars locked at `preregistration_darkcore_classifier_2026_05_27.md` (commit `646dcb0`).

### Detection (4 bars in 7.7.9+; PASS = all four)

| bar | threshold | description |
|---|---|---|
| **D1** misconception AUC | ≥ 0.70 | Mann-Whitney AUC, misconception scores vs truth scores |
| **D2** folklore AUC | ≥ 0.70 | Mann-Whitney AUC, folklore-subset scores vs truth scores (the dark-core test) |
| **D3** length-control delta | ≥ 0.10 | (added 7.7.8) detector AUC must beat the length-only oracle's AUC by ≥0.10 on BOTH partitions. Required because the benchmark's `expected_consensus` is length-confounded by class (truth ~3.9w, folklore ~7.5w), so the length oracle alone scores AUC ~0.79–0.80. A real detector clears D3 by demonstrating signal *above* the length artifact. |
| **D4** capitalization-control delta | ≥ 0.10 | (added 7.7.9) detector AUC must beat the capitalization-ratio oracle's *absolute* AUC by ≥0.10 on BOTH partitions. Discovered by the systematic confound audit (`papers/consensus-hallucination/PRE_STATED_PREDICTION_confound_audit_2026_05_27.md` → `papers/agent-self-audit/FINDING_confound_audit_2026_05_27.md`): truth responses are canonical short answers like "Paris"/"Newton" where capitalized-token ratio is structurally near 1.0, while folklore restatements are full sentences diluting the proper-noun density. Inverted cap-ratio oracle hits D1=0.704, D2=0.792 with ρ to length = -0.34 (partially orthogonal). |

Bars derived from the seven-method findings in `papers/consensus-hallucination/` + the 7.7.8 D3 bar (length artifact) + the 7.7.9 D4 bar (cap-ratio artifact). Both control-bar thresholds (0.10) are pre-registered judgment calls — chosen so that real detectors need AUC ≥ confound_oracle_AUC + 0.10 to demonstrate signal beyond each known artifact. Open to revision based on submission outcomes; the discipline pattern says revise when the data argues for it.

### Confound audit on the bundled benchmark (7.7.9 systematic scan)

Run `styxx gauntlet-audit-confounds` to reproduce. The audit scans 8 surface features as oracle-detectors and reports per-feature D1/D2 AUC (direction-agnostic) + Spearman ρ to word-length:

| oracle | D1 AUC (raw) | D2 AUC (raw) | |D1−0.5| | |D2−0.5| | ρ → length | passes D1? | passes D2? | category |
|---|---|---|---|---|---|---|---|---|
| `word_length` | 0.790 | 0.804 | 0.290 | 0.304 | 1.000 | ✓ | ✓ | calibration (controlled by D3) |
| `char_length` | 0.819 | 0.845 | 0.319 | 0.345 | +0.844 | ✓ | ✓ | length-downstream |
| `sentence_count` | 0.509 | 0.500 | 0.009 | 0.000 | +0.003 | · | · | inert |
| `question_mark_count` | 0.500 | 0.500 | 0.000 | 0.000 | undefined | · | · | inert |
| `exclamation_count` | 0.500 | 0.500 | 0.000 | 0.000 | undefined | · | · | inert |
| `capitalized_token_ratio` | 0.296 | 0.208 | 0.204 | 0.292 | −0.343 | ✓ (inverted) | ✓ (inverted) | **orthogonal confound — controlled by D4** |
| `hedge_density` | 0.518 | 0.491 | 0.018 | 0.009 | +0.082 | · | · | inert |
| `type_token_ratio` | 0.490 | 0.484 | 0.010 | 0.016 | −0.307 | · | · | inert |

**Audit outcome.** Of the 8 features audited, only two were "live" confounds at AUC≥0.70 in either direction: `word_length` (controlled by D3) and `capitalized_token_ratio` (controlled by D4). The other six features either gamed the bars only as length-downstream proxies (`char_length`) or sat at the AUC=0.5 floor (`sentence_count`, `question_mark`, `exclamation`, `hedge_density`, `type_token_ratio`). **No additional orthogonal confound remains uncontrolled at the audit's threshold.** Pre-stated prediction record: 5 of 8 individual-feature AUCs fell outside my stated ranges, including the most consequential — I predicted cap-ratio positive direction, actual was inverted. The pre-registered audit caught the discrepancy in implementation review; the FINDING records the falsifications honestly.

Detection-style methods are scored against the same labeled benchmark as classification; the "response" passed to `detect()` is the council's `expected_consensus` (the misconception for non-truth records, the truth for truth records).

---

## Leaderboard

### Reference baselines — provided by Fathom Lab

| rank | submitter | method | task | bars passed | K1 / K2 / K3 | summary |
|---|---|---|---|---|---|---|
| **Baseline-001** *(the seven-method floor)* | Fathom Lab | seven-method pre-registered arc | both | **0 / 7** | n/a (multiple bars across the arc) | Four detection methods (Dark Matter perturbation-fragility, CVPD agreement-fracture, JD justification-divergence, ICT neutral injection) — all closed-negative on the dark core. One classification method (Baseline-002 below). Two constructive variants (ICT-folklore, ICT-authoritative) — SHORTFALL on the curated corpus (28/30 already corrected baseline). Full receipts in [PAPER_decorrelation_ceiling_2026_05_27.md](papers/PAPER_decorrelation_ceiling_2026_05_27.md). Commit range: `bcd4208..a6d7a7e`. |
| **Baseline-002** *(the classifier)* | Fathom Lab | sentence-transformers/all-MiniLM-L6-v2 + balanced one-vs-rest logistic regression, trained on ICT receipts + curated truth controls, curated folklore items held out as cross-corpus test | classification | **1 / 3** | F1=0.42 / acc=0.77 / F1=0.36 | The shipped classifier from `darkcore_classifier_2026_05_27.py` wrapped in the gauntlet interface. **K2 accuracy passes** (0.77 ≥ 0.65), confirming the bars are individually beatable. **K1 + K3 fail** — folklore F1 cannot break 0.42 in-distribution, and cross-corpus folklore recall is only 0.17 (5/30 hand-curated folklore items flagged correctly). Reproduce: `styxx gauntlet --method submissions.baseline_002_classifier.method:predict --task classification`. |
| **Baseline-003** *(the length heuristic — anchors the floor)* | Fathom Lab | predict `folklore` if `len(q) < 60` else `factual-error` if `< 100` else `truth`. No model. | classification | **0 / 3** | F1=0.42 / acc=0.26 / F1=0.56 | A deliberately bad heuristic. **Anchor for the bottom of the leaderboard.** Notably gets K3 cross-corpus F1 = 0.56 (close to the 0.60 bar) because cross-corpus folklore items are short and the heuristic flags everything short as folklore — high recall (0.80), terrible precision. Accuracy 0.26 below the majority baseline (0.51) confirms it's noise. Reproduce: `styxx gauntlet --method submissions.baseline_003_length.method:predict --task classification`. |
| **Baseline-004** *(the random class — chance performance)* | Fathom Lab | hash-seeded uniform pick from the four classes (SHA-256 of question text → class index). Deterministic-random; CI-reproducible. | classification | **0 / 3** | F1=0.39 / acc=0.29 / F1=0.43 | Demonstrates the gauntlet handles methods with some-signal-by-luck but no real classification ability. Accuracy 0.29 (about chance for a 4-class problem with imbalanced labels), folklore F1 below Baseline-002, K3 above Baseline-003 by accident of hash uniformity. Reproduce: `styxx gauntlet --method submissions.baseline_004_random.method:predict --task classification`. |
| **Baseline-005** *(TF-IDF word — classical NLP)* | Fathom Lab | word 1-2-gram TF-IDF features + balanced one-vs-rest logistic regression. No model weights, no GPU. Same training corpus as Baseline-002. | classification | **1 / 3** | F1=0.30 / acc=0.74 / F1=0.22 | **Genuinely worse than Baseline-002 on the folklore axis.** K1 0.30 vs 0.42, K3 0.22 vs 0.36, cross-corpus folklore recall 0.067 (2/30) vs 0.167 (5/30). Classical bag-of-tokens lacks the semantic abstraction to pick up folklore signatures; embedding-based methods get at least a partial signal. K2 accuracy passes (0.74) on the linguistically-obvious truth-class boundary. Genuinely new finding: the dark core requires *semantic* features at minimum. Reproduce: `styxx gauntlet --method submissions.baseline_005_tfidf.method:predict --task classification`. |
| **Baseline-006** *(TF-IDF char — n=2 confirmation)* | Fathom Lab | char-wb 3-5-gram TF-IDF + balanced one-vs-rest LR. Identical training corpus + classifier head to Baseline-005; only the vectorizer analyzer differs. | classification | **1 / 3** | F1=0.30 / acc=0.74 / F1=0.22 | **Exact-metric tie with Baseline-005.** Both classical TF-IDF variants — word AND char level — produce identical K1=0.30, K2=0.74, K3=0.22, recall=0.067. **n=2 confirmation** of the pre-stated prediction (committed at `19bdb8c` BEFORE the run): classical bag-of-tokens cannot see the dark core at any lexical granularity. The signal lives in semantic-similarity space, not in surface-form n-grams. Reproduce: `styxx gauntlet --method submissions.baseline_006_tfidf_char.method:predict --task classification`. |

### Detection-task baselines (under v3 bars with D3 length-control + D4 capitalization-control, shipped 7.7.9)

| rank | submitter | method | task | bars passed | D1 / D2 / D3-delta / D4-delta | summary |
|---|---|---|---|---|---|---|
| **Baseline-007** *(token-overlap — v1 artifact PASS; now 3/4 under v3)* | Fathom Lab | length+overlap heuristic on `expected_consensus`. No model. | detection | **3 / 4** | D1=0.864 ✓ / D2=0.922 ✓ / D1-len=0.074 ✗ / D1-cap=0.160 ✓ | Under v1 bars (D1, D2 only) this submission hit PASS=2/2. Investigation revealed it was exploiting a length-confound in `expected_consensus` (truth ~3.9w, folklore ~7.5w). **7.7.8 added D3 length-control**: AUC must beat the length-only oracle by ≥0.10 on both partitions. Baseline-007's D1 delta is 0.074 (below 0.10) — barely above pure length. **7.7.9 added D4 capitalization-control** (after systematic confound audit found cap-ratio as a second orthogonal artifact); Baseline-007 passes D4 with D1-cap delta 0.160. Net verdict: 3/4, still NOT a PASS. The dogfood loop demonstrating itself: real submission → real artifact → controlled bar → controlled audit. Reproduce: `styxx gauntlet --method submissions.baseline_007_token_overlap.method:detect --task detection`. |
| **Baseline-008** *(embedding similarity — first non-trivial detection submission; pre-stated prediction validated)* | Fathom Lab | sentence-transformer (`all-MiniLM-L6-v2`) cosine similarity between question and `expected_consensus`. | detection | **3 / 4** | D1=0.805 ✓ / D2=0.928 ✓ / D1-len=0.015 ✗ / D1-cap=0.102 ✓ | First non-trivial detection submission tested under v2 bars (D1+D2+D3). Pre-stated prediction committed at `a05c8c1` BEFORE the gauntlet run (`submissions/baseline_008_embedding_similarity/PRE_STATED_PREDICTION.md`): ~15-25% PASS, ~60% pass D1+D2 fail D3, ~15% fail D1/D2. **Modal prediction validated** — actual outcome landed in the 60% region. On D1, embedding cosine similarity is structurally equivalent to length (D1=0.805 vs length=0.790, delta 0.015 << 0.10). On D2, embedding adds 12.4pp above length — real signal on the narrower axis but not enough to pass D3 jointly. Under 7.7.9 v3 bars, Baseline-008 also passes D4 (D1-cap delta = 0.102, right at threshold). **The dark core stays dark to embedding-similarity detection at this corpus.** Reproduce: `styxx gauntlet --method submissions.baseline_008_embedding_similarity.method:detect --task detection` (requires `pip install styxx[nli]`). |
| **Baseline-009** *(length-residualized embedding similarity — the obvious smart-fix tested first under v3 bars; pre-stated prediction validated)* | Fathom Lab | Same `all-MiniLM-L6-v2` cosine similarity as Baseline-008, then residualized against length: `score = sim − (intercept + slope · word_length)`. Linear fit is unsupervised — uses only (sim, length) pairs from the benchmark, no class labels. | detection | **1 / 4** | D1=0.645 ✗ / D2=0.775 ✓ / D1-len=-0.145 ✗ / D1-cap=-0.059 ✗ | First detection submission tested under v3 bars (D1+D2+D3+D4). Pre-stated prediction committed at `d0de04b` BEFORE the gauntlet run (`submissions/baseline_009_residualized_embedding/PRE_STATED_PREDICTION.md`): ~3% PASS, ~30% pass D2 only (modal), ~35% fail all four. **Modal prediction validated** — actual outcome (D2 only) landed in the 30% region; 3 of 4 specific AUC predictions inside their pre-stated ranges. **n=1 empirical demonstration that length-residualization is NOT the path through D3.** Residualization removes the length-correlated component of the score, actively LOWERING detector AUC below the length oracle's AUC by construction (delta = -0.145 on D1, -0.029 on D2). The path through D3 requires signal *orthogonal* to length (NLI-based answer-form detection, specificity scoring, cross-vendor council disagreement), not the length variance subtracted from the same signal. Baseline-010 is the disciplined follow-up — a method that adds new signal rather than subtracting old signal. Reproduce: `styxx gauntlet --method submissions.baseline_009_residualized_embedding.method:detect --task detection` (requires `pip install styxx[nli]`). |

### External submissions

*(none yet — be the first to land on the leaderboard. K3 is the load-bearing bar; the dark core has resisted every method we've tested.)*

| rank | submitter | method | task | bars passed | metrics | submitted |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

---

## Sanity submissions (not ranked — trivial lower bounds)

These ship inside `styxx.gauntlet` itself; any real method must beat them.

| method | task | bars passed | what it proves |
|---|---|---|---|
| `styxx.gauntlet:_majority_baseline_predict` (always predicts "truth") | classification | 0 / 3 | the bars cannot be cleared by predicting the majority class |
| `styxx.gauntlet:_zero_baseline_detect` (constant-zero score) | detection | 0 / 4 | the bars cannot be cleared by random/constant scores (AUC = 0.5 by definition) |
| `styxx.gauntlet:_length_oracle_detect` (score = response word count) | detection | 2 / 4 | demonstrates D3 + D4 do isolated work: length oracle passes D1+D2 (artifact) but fails its own control bar D3 (delta=0) AND fails D4 (length only beats cap-ratio by 0.086, below 0.10) |
| `styxx.gauntlet:_capratio_oracle_detect` (score = 1 − cap-ratio) | detection | 2 / 4 | the symmetric regression-test for D4: cap-ratio oracle passes D1+D2 (the inverted-format artifact) but fails its own D4 control by construction AND fails D3 (cap-ratio is partially length-derived but inverted, so it doesn't beat length-oracle by 0.10) |

Reproduce: `styxx gauntlet --method 'styxx.gauntlet:_majority_baseline_predict' --task classification`.

---

## Honest scope of the benchmark

- **n = 108 records, four classes.** Class distribution: truth 55, folklore 34, factual-error 13, pseudoscience 6. The folklore class is the largest non-truth bucket and is the primary focus of the dark-core hypothesis.
- **Three-vendor council** for the labeled `expected_consensus` field (gpt-4o-mini + Qwen2.5-3B + gemma-2-2b-it). Methods that route through different vendors may produce different "natural" baseline answers; the benchmark's labels are anchored to this specific council.
- **English-language; mostly Western cultural priors.** Cross-language / cross-cultural extension is a known open follow-up.
- **The seven-method floor is bounded by these scope conditions.** A method that beats the floor on the bundled benchmark is a real result *within this scope*; cross-corpus generalization to other benchmarks is a separate test.

---

## Citing this leaderboard

If you publish work that engages with this benchmark or the seven-method floor:

```
Rodabaugh, Alexander (Fathom Lab). 2026. "The Decorrelation Ceiling: A Seven-Method
Empirical Floor on Reference-Free Detection of Cross-Vendor Consensus Hallucination."
styxx 7.7.5, fathom-lab/styxx git main, 2026-05-27.
Leaderboard: github.com/fathom-lab/styxx/blob/main/LEADERBOARD.md
Paper: github.com/fathom-lab/styxx/blob/main/papers/PAPER_decorrelation_ceiling_2026_05_27.md
```

The styxx project carries Zenodo concept DOI [`10.5281/zenodo.19326174`](https://doi.org/10.5281/zenodo.19326174). The release-specific DOI for v7.7.7 is [**`10.5281/zenodo.20418532`**](https://doi.org/10.5281/zenodo.20418532) (v24 in the concept chain, published 2026-05-27).

---

*Submissions are reviewed for honest scope: any submission that modifies the benchmark, changes the bars, or uses external test data that was not held out from training will be rejected. The discipline is the moat.*
