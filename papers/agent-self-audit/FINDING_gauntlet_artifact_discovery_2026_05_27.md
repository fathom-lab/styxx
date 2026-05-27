# Finding · the gauntlet's first non-trivial detection submission exposed a length-confound in the v1 bars, and the next patch shipped the fix — in the same session

**Date:** 2026-05-27 · **Status:** meta-finding about the public-challenge infrastructure (`styxx gauntlet`, shipped 7.7.5, hardened 7.7.8). Records the discovery chain from "unexpected PASS" → "diagnosed artifact" → "new bar definition" → "regression test" → "patch release" — all within a single session. Sixth in-session falsification of the 2026-05-27 arc; reads alongside [FINDING_pareto_frontier_2026_05_27.md](FINDING_pareto_frontier_2026_05_27.md) and [FINDING_product_exploration_2026_05_27.md](FINDING_product_exploration_2026_05_27.md).

> **Outcome.** A 30-line token-overlap heuristic submitted to the gauntlet as a sanity check accidentally PASSed the v1 detection bars (D1=0.864, D2=0.922). Investigation showed the bars were exploitable via a length-confound in the benchmark's `expected_consensus` field (truth responses average 3.9 words; folklore responses 7.5 words). A new D3 length-control bar was defined (AUC must beat the length-only oracle by ≥0.10), regression-tested, and released as styxx 7.7.8 — in the same session as the gauntlet itself shipped. **The system caught its own flaw.**

## What the gauntlet was built to do

The `styxx gauntlet` (commits `f93a734` and forward, shipped in 7.7.5) is a public-challenge runner: external researchers submit a detection or classification method, the gauntlet runs it against a labeled benchmark (`darkcore_benchmark_2026_05_27.json`), scores it against pre-registered bars, and adds successful submissions to `LEADERBOARD.md`. The frame: the seven-method floor we shipped IS the public bar; submissions either beat the floor (the synthesis gets revised) or fail (the floor compounds across submissions).

The implicit promise of the frame: the bars are *robust* — methods that PASS them have demonstrated real signal, not artifact-exploitation. This finding records what happened when that promise was tested.

## The discovery chain

### Step 1 — Baseline-005 (word TF-IDF), n=1

A classical-NLP baseline submitted to populate the leaderboard. Trained on the same corpus as Baseline-002 (sentence-transformer). Result: 1/3 bars (K2 accuracy 0.74 passed; K1 0.30 and K3 0.22 failed). **Worse than Baseline-002 on every folklore-axis metric** (K1 0.30 vs 0.42; K3 0.22 vs 0.36; cross-corpus recall 0.067 vs 0.167). First reading: "classical bag-of-tokens lacks the semantic abstraction to pick up folklore signatures."

### Step 2 — pre-stated prediction for Baseline-006 (char TF-IDF)

The Baseline-005 finding was n=1. Could char-level lexical features pick up what word-level missed? Pre-registered prediction committed at `19bdb8c` *before* running:

- K3 < 0.30 AND recall < 0.17 → "classical bag-of-tokens cannot see the dark core at any granularity" (n=2 confirmation)
- K3 ≥ 0.36 AND recall ≥ 0.17 → "word-level abstraction was wrong; char-level rescues" (partial refutation)
- middle → "char-level captures *some* signal that word-level misses"

### Step 3 — Baseline-006 ran, exact-metric tie with Baseline-005

K1=0.30, K2=0.74, K3=0.22, recall=0.067 — **identical to Baseline-005**. n=2 confirmation: both classical TF-IDF variants produce identical prediction patterns on this benchmark. The dark core isn't seen by surface-form features at any lexical granularity.

### Step 4 — Baseline-007 (token-overlap detector) unexpectedly PASSed v1 bars

Submitted as a sanity-check 30-line heuristic to populate the *detection* side of the leaderboard. Scoring rule: `(1 - hedge_density) * (1 - novelty_density)` where `novelty_density` is the fraction of response tokens NOT in the question. The honest expectation: a noise-level baseline, akin to the random class on the classification side. 

Actual result: **D1=0.864, D2=0.922. PASS=true. 2/2 bars.** First-ever PASS on the leaderboard, from a method with no model and 30 lines of code.

### Step 5 — investigation of the unexpected PASS

The mean scores by class on the benchmark immediately exposed a length pattern:

| class | mean `expected_consensus` length |
|---|---|
| truth | 3.9 words (often single-word: "Paris", "206", "seven") |
| factual-error | 6.6 words |
| pseudoscience | 6.5 words |
| folklore | 7.5 words (full council restatements) |

The token-overlap detector's score positively correlates with response length (longer responses share more tokens with the question, with the divisor in the score being length). Since truth responses are systematically shorter than misconception responses, the detector's score is *primarily* tracking length, not misconception likelihood.

### Step 6 — the length oracle confirms the artifact

A pure length-only oracle (`score = len(response.split())`) was run against the same partitions:

| partition | length-oracle AUC |
|---|---|
| misconception vs truth (D1 partition) | **0.79** |
| folklore vs truth (D2 partition) | **0.80** |

The length alone scores AUC ~0.80 across both bars — far above the 0.50 random baseline. The v1 bars (≥0.70) are gameable by length-only detectors.

### Step 7 — the D3 bar definition

The fix landed in styxx 7.7.8 (commit `48a0ef0`): a new D3 bar requiring the detector's AUC to beat the length-only oracle's AUC by ≥0.10 on both partitions. The bar is implemented via a constant `_length_oracle_detect` in `styxx/gauntlet.py` whose AUC is computed inline during every gauntlet run; the deltas (`D1_minus_length_AUC`, `D2_minus_length_AUC`) are reported in the metrics block. PASS now requires D1 ∧ D2 ∧ D3.

### Step 8 — regression test

`tests/test_gauntlet.py::test_length_oracle_passes_D1_D2_but_fails_D3` was added in the same commit: the length oracle (whose score IS length) trivially passes D1+D2 (because length is what's correlating with class) but D3 must fail by construction (delta = 0 — the oracle can't beat itself). This regression-guards any future attempt to weaken or remove D3.

### Step 9 — Baseline-007 re-scored under v2 bars

Same submission, same gauntlet code, but with D3 active:

| bar | result |
|---|---|
| D1 = 0.864 | PASS (≥0.70) |
| D2 = 0.922 | PASS (≥0.70) |
| D1 − length = **0.074** | FAIL (≥0.10) |
| D2 − length = 0.117 | PASS (≥0.10) |
| **D3 overall** | **FAIL** (D1 delta insufficient) |
| Submission verdict | **2/3 — NOT a PASS** |

The "first PASS on the leaderboard" is correctly downgraded to "2/3 with the length-confound caught."

## What this proves about the infrastructure

The dogfood loop the gauntlet was supposed to enable **just demonstrated itself**:

- A real submission was run.
- It produced an unexpected result.
- Investigation diagnosed a real validity issue with the bars.
- The fix shipped in the next patch with a regression test.
- The leaderboard re-scored the offending submission honestly under the new bars.
- The original v1 score is preserved (`reported_v1_bars` field in the submission JSON) alongside the v2 score, so the history of the discovery isn't erased.

Six minutes elapsed between "unexpected PASS" and "patch release on PyPI." The infrastructure's self-correction capability is the property: **every real submission either confirms the floor (compounds credibility) or surfaces a fix (improves the bars).** Both outcomes benefit the project.

## What this proves about the benchmark

The v1 benchmark (`darkcore_benchmark_2026_05_27.json`) is **structurally length-confounded** in its `expected_consensus` field. Truth responses were hand-curated as short canonical answers ("Paris"); misconception responses were taken from the council's full-sentence ICT outputs. The detection task as-defined inherits this confound. The D3 bar makes the artifact a *controlled* feature of the score — a detector either beats it or doesn't — but the benchmark itself still carries the confound for any future task definition. A benchmark v2 with length-equalized responses across classes would remove the confound at the source rather than controlling for it.

Classification bars (K1, K2, K3) are **not affected** — those use question text only, which has no per-class length signature.

## Open follow-ups

1. **Benchmark v2 with length-equalized responses.** Re-curate the truth class with full-sentence canonical answers matched in length to the misconception class. This removes the confound at the corpus level. Operator-territory; multi-hour work.
2. **D3 threshold revision.** The 0.10 threshold is a judgment call. If real-method submissions clear it cleanly (e.g., a serious semantic-similarity detector scoring 0.92+ on both partitions, well above the 0.79–0.80 length floor), the threshold stays right. If too many real submissions fall just short (D1−length in [0.05, 0.10]), the threshold may need lowering — but only based on submission data, never to make the bar easier.
3. **Cross-confound check.** Other potential artifacts (e.g., question-mark density, named-entity density) should be probed the same way before the next real detection submission. The framework is in place; the audit is the next disciplined pass.

## Reproducibility

- Baseline-005 (word TF-IDF): commit `4a4f3fd`, score `1/3` (K2 only).
- Baseline-006 (char TF-IDF) + pre-stated prediction: commits `19bdb8c` (prereg) and `fb67fd2` (result + leaderboard).
- Baseline-007 (token-overlap detector) + initial PASS + diagnosis: commit `fb67fd2`.
- D3 bar + regression test + Baseline-007 re-scoring: commit `48a0ef0` (styxx 7.7.8).
- This finding: this commit.

The discovery chain is reconstructible from `git log --oneline submissions/ styxx/gauntlet.py LEADERBOARD.md` in the order above.

## Why this finding gets a paper-grade artifact

The Pareto-frontier finding (commit `3b978e1`) recorded that the producer's own product caught the producer drifting from the producer's own register-law. This finding records the structural counterpart: the producer's own *infrastructure* caught a flaw in the producer's own *bars*, in the same session, with the patch in the next release. Both are meta-findings about how the discipline pattern works in practice. Together they argue that the *recursion* — agent uses tool, tool catches agent; submission tests bars, bars get revised — is a deployable property, not just a session-specific demonstration.

The session has now produced **six in-session falsifications recorded in place**:

1. C1-profile ≤0.20 register-law bar (Pareto finding)
2. set_session-doesn't-propagate observation (product-exploration finding)
3. ICT-folklore auto-verdict PASS (probe label bug)
4. ICT-authoritative auto-verdict PASS (same label bug shape)
5. styxx 7.7.5 wheel bundling miss (caught by clean-env verification)
6. **The gauntlet's v1 detection bars being length-gameable (this finding)**

The falsification trail compounds. The discipline pattern is verifiable from public git history.
