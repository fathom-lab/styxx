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

### Baseline-001 — the floor

| rank | submitter | method | task | bars passed | summary |
|---|---|---|---|---|---|
| **Baseline-001** *(the floor)* | Fathom Lab | seven-method pre-registered arc | both | **0 / 7** | Four detection methods (Dark Matter perturbation-fragility, CVPD agreement-fracture, JD justification-divergence, ICT neutral injection) — all closed-negative on the dark core. One classification method (sentence-transformer + balanced LR) — FAIL K2 + K3, 20% recall on cross-corpus folklore. Two constructive variants (ICT-folklore, ICT-authoritative) — SHORTFALL on the same corpus (28/30 already corrected baseline). Full receipts in [PAPER_decorrelation_ceiling_2026_05_27.md](papers/PAPER_decorrelation_ceiling_2026_05_27.md). Commit range: `bcd4208..a6d7a7e`. Submission date: 2026-05-27. |

### External submissions

*(none yet — be the first)*

| rank | submitter | method | task | bars passed | metrics | submitted |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

---

## Sanity submissions (for reference; not ranked)

Trivial baselines that any real method must beat. These ship inside the gauntlet module itself and can be reproduced with `styxx gauntlet --method 'styxx.gauntlet:_majority_baseline_predict' --task classification`:

| method | task | bars passed | what it proves |
|---|---|---|---|
| `_majority_baseline_predict` (always predicts "truth") | classification | 0 / 3 | the bars cannot be cleared by predicting the majority class |
| `_zero_baseline_detect` (constant-zero score) | detection | 0 / 2 | the bars cannot be cleared by random/constant scores (AUC = 0.5 by definition) |

If a submitted method does not beat these sanity baselines, it does not get a leaderboard row.

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

The styxx project carries Zenodo concept DOI `10.5281/zenodo.19326174`. Release-specific deposits are manually curated.

---

*Submissions are reviewed for honest scope: any submission that modifies the benchmark, changes the bars, or uses external test data that was not held out from training will be rejected. The discipline is the moat.*
