# Pre-stated prediction — benchmark-wide confound audit (committed BEFORE running)

**File created:** 2026-05-27, before any `audit_confounds()` invocation against the bundled benchmark (`papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json`, n=108).

## Why this exists

The D3 length-control bar (shipped 7.7.8) caught one artifact: word-length on the `expected_consensus` field games D1+D2 at AUC ≈ 0.79–0.80. The fix was a regression-tested bar requiring detectors to beat the length-only oracle by ≥ 0.10. That handles the artifact we found.

This audit asks the next question on the same discipline track: **what other plausible surface features game the bars?** Any feature whose oracle alone scores D1 ≥ 0.70 OR D2 ≥ 0.70 is a candidate confound. The audit either:

- Finds new orthogonal confounds → those become new D-bars with regression tests (same pattern as 7.7.8).
- Finds only length-downstream features → confirms D3 effectively covers the artifact space.

The bet: pre-state predictions for the eight features below, commit, run once, publish all eight results regardless of outcome.

## Features under audit

| oracle | what it measures | downstream of length? |
|---|---|---|
| `word_length` | `len(response.split())` (the existing D3 oracle, included for calibration) | — |
| `char_length` | `len(response)` | yes — corr w/ word length expected > 0.95 |
| `sentence_count` | `len(re.split(r'[.!?]+', response))` | yes — corr expected > 0.7 |
| `question_mark_count` | count of `?` in response | no — orthogonal axis |
| `exclamation_count` | count of `!` in response | no — orthogonal axis |
| `capitalized_token_ratio` | fraction of tokens with first letter uppercase (NER proxy) | partially — long responses have more named entities by base rate |
| `hedge_density` | count of {`often`, `widely`, `commonly`, `popularly`, `typically`, `said to`, `is said`, `believed`, `associated with`} per word | partially — folklore restatements use softeners |
| `type_token_ratio` | `len(set(tokens)) / len(tokens)` | yes — inversely correlated with length |

## The predictions (locked before running)

### Per-feature AUC predictions

| oracle | D1 AUC range | D2 AUC range | passes D1 (≥0.70)? | passes D2 (≥0.70)? |
|---|---|---|---|---|
| `word_length` | 0.78–0.80 | 0.78–0.82 | yes (calibration) | yes (calibration) |
| `char_length` | 0.78–0.83 | 0.79–0.84 | **yes** | **yes** |
| `sentence_count` | 0.65–0.78 | 0.70–0.82 | toss-up | **yes** |
| `question_mark_count` | 0.45–0.55 | 0.45–0.55 | no | no |
| `exclamation_count` | 0.45–0.55 | 0.45–0.55 | no | no |
| `capitalized_token_ratio` | 0.55–0.72 | 0.60–0.78 | toss-up | toss-up (slight lean yes) |
| `hedge_density` | 0.55–0.72 | 0.60–0.78 | toss-up | toss-up (lean yes on folklore) |
| `type_token_ratio` | 0.55–0.72 | 0.55–0.72 | toss-up | toss-up |

### Joint prediction

| outcome | probability |
|---|---|
| At least 2 features (besides `word_length`) pass D1 ≥ 0.70 | **~80%** |
| At least 2 features pass D2 ≥ 0.70 | **~85%** |
| All length-downstream features (`char_length`, `sentence_count`, `type_token_ratio`) pass at least one bar | **~70%** |
| At least one **orthogonal** feature (`question_mark`, `exclamation`, `cap_ratio`, `hedge_density`) games D2 at AUC ≥ 0.70 with corr-to-length < 0.5 | **~30%** |
| **No orthogonal confound found** (audit confirms D3 covers the artifact space) | **~70%** |

### What the outcomes argue

- **Only length-downstream features game the bars** (~70%): the D3 bar effectively covers the artifact space at the corpus level. Audit publishes the full table as documentation; no new D-bar added; the floor is hardened by what we *didn't* find.
- **An orthogonal confound games the bars** (~30%): add a new D-bar (D4) requiring detectors to beat that oracle by ≥0.10 too, with a regression test. Most likely candidate: `capitalized_token_ratio` or `hedge_density` on D2 (folklore restatements often contain proper nouns + softening language).
- **No feature passes any bar** (~5%): would mean the v1 length-confound was an anomaly and the benchmark is actually clean; unlikely given the seven-method floor's track record.

## Honest reasoning

The length-downstream features (`char_length`, `sentence_count`, `type_token_ratio`) almost certainly pass — they're nearly redundant with word length. That's not a new artifact; that's the same artifact in a different unit. The interesting question is whether anything *orthogonal* to length games the bars.

My honest prior: probably not. The benchmark's `expected_consensus` field for truth records is hand-curated short canonical answers ("Paris", "206", "seven"). Truth responses don't have proper nouns by accident or hedge language by style — they just don't have **content** in those slots. Folklore restatements have those features because they're full sentences. So the candidate orthogonal confounds are still partially length-derived; the question is whether they game the bars beyond what raw length already does.

If `capitalized_token_ratio` games D2 at AUC ≥ 0.75 with corr-to-length < 0.5, that's a real find — proper-noun density would be a separate axis from word count. Same for hedge density. But I expect both to track length too closely to qualify as orthogonal.

## Not re-running, not re-tuning

- Oracle definitions are locked. No bespoke tuning per feature.
- The audit runs once. Results published regardless of outcome.
- Any new D-bar added based on the audit will have its own regression test, like D3 did.

This document is committed to origin **before** the `audit_confounds()` invocation. Verifiable from git history.
