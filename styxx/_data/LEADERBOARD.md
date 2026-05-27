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

### Detection (2 bars; PASS = both)

| bar | threshold | description |
|---|---|---|
| **D1** misconception AUC | ≥ 0.70 | Mann-Whitney AUC, misconception scores vs truth scores |
| **D2** folklore AUC | ≥ 0.70 | Mann-Whitney AUC, folklore-subset scores vs truth scores (the dark-core test) |

Bars derived from the seven-method findings in `papers/consensus-hallucination/`. Detection-style methods are scored against the same labeled benchmark as classification; the "response" passed to `detect()` is the council's expected_consensus (the misconception for non-truth records, the truth for truth records).

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

### ⚠️ Detection-task benchmark artifact discovered (Baseline-007)

| rank | submitter | method | task | bars passed | D1 / D2 | summary |
|---|---|---|---|---|---|---|
| **Baseline-007** ⚠️ *(token-overlap detector — gamed the bars via benchmark artifact, NOT a real win)* | Fathom Lab | length+overlap heuristic on `expected_consensus`. No model. Submitted as a sanity check for the detection task; unexpectedly hit PASS. | detection | **2 / 2** *(artifact)* | D1=0.864 / D2=0.922 | **NOT a real PASS — exploits a benchmark structural artifact.** Investigation: truth responses in the benchmark's `expected_consensus` field average **3.9 words** (often single-word answers: "Paris", "206", "seven"); folklore responses average **7.5 words** (full council restatements). A detector measuring response length alone scores AUC=0.92 on folklore-vs-truth. The detection bars (D1, D2) as currently defined are length-gameable. Classification bars (K1/K2/K3) are unaffected (question-only input has no length artifact). **Benchmark v2 follow-up:** equalize response-length distribution across classes, or revise the detection bars to length-normalized AUC. Until resolved, the detection floor should be interpreted with this artifact in mind. Reproduce: `styxx gauntlet --method submissions.baseline_007_token_overlap.method:detect --task detection`. |

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
| `styxx.gauntlet:_zero_baseline_detect` (constant-zero score) | detection | 0 / 2 | the bars cannot be cleared by random/constant scores (AUC = 0.5 by definition) |

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
