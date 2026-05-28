# Finding · the benchmark-wide confound audit — pre-stated predictions partially falsified, second orthogonal confound discovered, D4 capitalization-control bar shipped

**Date:** 2026-05-27 · **Status:** structural counterpart to [FINDING_gauntlet_artifact_discovery_2026_05_27.md](FINDING_gauntlet_artifact_discovery_2026_05_27.md). Records the systematic 8-feature audit run against the dark-core benchmark *after* D3 caught one confound. **Seventh in-session falsification of the 2026-05-27 arc.**

> **Outcome.** Pre-stated predictions for 8 oracle confounds committed at `48a9fe3` BEFORE running. Audit revealed a *second* orthogonal confound — `capitalized_token_ratio`, inverted (truth has *higher* proper-noun density because canonical answers like "Paris", "Newton", "1789" are mostly proper-noun tokens). 5 of 8 specific AUC predictions fell outside their stated ranges. The orthogonality finding led to a new D4 capitalization-control bar (regression-tested same as D3), shipped 7.7.9.

## What was tested

After D3 (7.7.8) caught the length-confound in `expected_consensus`, the next disciplined move was to ask: **what *other* surface features game the bars?** Eight oracle-detectors were defined, each computing a feature on the response alone:

| oracle | function |
|---|---|
| `word_length` | `len(response.split())` (calibration — D3 oracle) |
| `char_length` | `len(response)` |
| `sentence_count` | sentence-split via `[.!?]+` regex |
| `question_mark_count` | count of `?` in response |
| `exclamation_count` | count of `!` in response |
| `capitalized_token_ratio` | fraction of tokens starting with uppercase |
| `hedge_density` | fraction of tokens in `{often, widely, commonly, popularly, typically, said, believed, ...}` |
| `type_token_ratio` | `len(set(tokens)) / len(tokens)` (lexical diversity) |

For each: compute D1 AUC (misconception vs truth), D2 AUC (folklore vs truth), and Spearman ρ to `word_length` (to identify length-downstream vs orthogonal confounds). An oracle with AUC ≥ 0.70 in either direction AND `|ρ| < 0.5` is a *candidate orthogonal confound* — a feature that games the bars without being a length proxy.

## The pre-stated predictions (committed at `48a9fe3` BEFORE running)

| oracle | predicted D1 | actual D1 | predicted D2 | actual D2 | call |
|---|---|---|---|---|---|
| `word_length` | 0.78–0.80 | **0.79** | 0.78–0.82 | **0.80** | ✓ calibration |
| `char_length` | 0.78–0.83 | **0.82** | 0.79–0.84 | **0.84** | ✓ as predicted |
| `sentence_count` | 0.65–0.78 | 0.51 | 0.70–0.82 | 0.50 | ✗ **way off** |
| `question_mark_count` | 0.45–0.55 | **0.50** | 0.45–0.55 | **0.50** | ✓ exact |
| `exclamation_count` | 0.45–0.55 | **0.50** | 0.45–0.55 | **0.50** | ✓ exact |
| `capitalized_token_ratio` | 0.55–0.72 (positive) | 0.30 (inverted) | 0.60–0.78 (positive) | 0.21 (inverted) | ✗ **direction missed** |
| `hedge_density` | 0.55–0.72 | 0.52 | 0.60–0.78 | 0.49 | ✗ weaker + folklore reversed |
| `type_token_ratio` | 0.55–0.72 | 0.49 | 0.55–0.72 | 0.48 | ✗ weaker than predicted |

### Joint predictions vs. outcomes

| prediction | predicted prob | outcome |
|---|---|---|
| At least 2 features (besides `word_length`) pass D1 ≥ 0.70 | ~80% | **PARTIAL** — only `char_length` passes in raw direction; `capitalized_token_ratio` passes in *inverted* direction → 2 features total under absolute-AUC, 1 under raw |
| At least 2 features pass D2 ≥ 0.70 | ~85% | **PARTIAL** — same shape as above |
| All length-downstream features pass at least one bar | ~70% | **FALSIFIED** — `char_length` yes, `sentence_count` no, `type_token_ratio` no |
| At least one **orthogonal** feature games D2 at AUC ≥ 0.70 with corr < 0.5 | ~30% | **CONFIRMED** — `capitalized_token_ratio` inverted, D2=0.79, ρ=-0.34 |
| No orthogonal confound found | ~70% | **FALSIFIED** — see above |
| No feature passes any bar | ~5% | **FALSIFIED** (char_length passes both raw; cap-ratio passes both inverted) |

**Three of the joint predictions were falsified outright.** The most important falsification: my expected ~70% prior that "no orthogonal confound would be found" was wrong. The audit *did* find one, but it was inverted from how I framed the prediction (I only checked AUC ≥ 0.70 in the positive direction; the cap-ratio confound has AUC ≤ 0.30, which is equivalently strong signal but flipped).

### Why the cap-ratio inversion happened

Truth records in the benchmark are hand-curated short canonical answers like `"Paris"`, `"Newton"`, `"1789"`, `"206 bones"`. By construction, these strings are **dominated** by capitalized proper-noun-like tokens — the cap-ratio is often 1.0 for single-word truth responses. Folklore restatements like `"Walt Disney's body is interred at Forest Lawn Memorial Park"` are full sentences with many lowercase tokens (`is`, `at`, `the`, articles, prepositions) diluting the proper-noun ratio.

**So the cap-ratio confound is the *answer-format* signature, not a content signature.** Short single-token answers vs. full-sentence restatements. Partially length-derived (ρ to length = -0.34, since short responses have higher cap-ratio by accident) but not fully redundant with D3.

### The audit code missed it the first time

My initial implementation of `audit_confounds()` only flagged confounds at `AUC ≥ d1_bar` — the positive direction. The cap-ratio's raw AUC was 0.30, below the 0.70 threshold, so the first audit reported `n_orthogonal_confounds_found: 0`. **My code was confounded by the same direction-blindness as my prediction.** Fixed by adding `D1_AUC_abs = max(auc, 1-auc)` and flagging based on absolute AUC. Re-ran the audit; cap-ratio surfaced as a candidate orthogonal confound. Recorded the fix in the same commit as the D4 bar.

## The D4 bar (shipped 7.7.9)

Same discipline pattern that produced D3 in 7.7.8:

1. **The bar.** Detector AUC must beat the inverted-cap-ratio oracle's absolute AUC by ≥ 0.10 on both partitions:
   - `D1_minus_capratio_AUC = D1_AUC - max(capratio_D1_AUC, 1 - capratio_D1_AUC) ≥ 0.10`
   - `D2_minus_capratio_AUC = D2_AUC - max(capratio_D2_AUC, 1 - capratio_D2_AUC) ≥ 0.10`
2. **The regression test.** `tests/test_gauntlet.py::test_capratio_oracle_passes_D1_D2_but_fails_D4` — the `_capratio_oracle_detect` constant (whose score IS `1 - cap_ratio`) passes D1 + D2 (because the inverted feature correlates with class) but D4 fails by construction (delta = 0 vs itself).
3. **The audit primitive.** `styxx.gauntlet.audit_confounds()` (+ CLI `styxx gauntlet-audit-confounds`) runs the full 8-oracle suite and reports the table, so future submitters and the project can re-run the audit when new confounds are suspected.
4. **PASS now requires D1 ∧ D2 ∧ D3 ∧ D4.** Detection submissions report 4 bars; the perfect-oracle regression test verifies D4 is beatable by a real detector that has genuine signal.

## Re-scoring existing detection submissions under D1-D4

| submission | D1 | D2 | D3 (length) | D4 (cap-ratio) | bars passed | verdict |
|---|---|---|---|---|---|---|
| Baseline-007 (token-overlap) | 0.864 ✓ | 0.922 ✓ | Δ=0.074 ✗ | Δ=0.160 ✓ | **3/4** | FAIL (D3) |
| Baseline-008 (embedding similarity) | 0.805 ✓ | 0.928 ✓ | Δ=0.015 ✗ | Δ=0.101 ✓ | **3/4** | FAIL (D3) |
| `_length_oracle_detect` | 0.790 ✓ | 0.804 ✓ | Δ=0.000 ✗ | Δ=0.086 ✗ | **2/4** | FAIL |
| `_capratio_oracle_detect` | 0.704 ✓ | 0.792 ✓ | Δ=−0.086 ✗ | Δ=0.000 ✗ | **2/4** | FAIL |
| `_zero_baseline_detect` | 0.500 ✗ | 0.500 ✗ | Δ=−0.290 ✗ | Δ=−0.204 ✗ | **0/4** | FAIL |
| perfect oracle (regression test) | 1.000 ✓ | 1.000 ✓ | Δ=0.210 ✓ | Δ=0.296 ✓ | **4/4** | PASS |

**The two real submissions both pass D4** (their detectors add signal above the cap-ratio confound), but both still fail D3 (the length confound is the dominant artifact). **The two oracle detectors each fail their *own* control bar by construction**, demonstrating both bars do isolated work. The discipline pattern is intact: PASS still requires real signal beyond both confounds.

## What this proves about the discipline pattern

Six bullets from the [gauntlet artifact finding](FINDING_gauntlet_artifact_discovery_2026_05_27.md) about how the system caught its own flaw. This finding extends the chain:

- **The chain is recursive.** D3 was discovered by accident (Baseline-007's unexpected PASS). D4 was discovered by deliberate scan (the audit primitive). The discipline scales: artifact-finding doesn't depend on lucky accidents.
- **Pre-registered predictions can be wrong in non-obvious ways.** I locked predictions for 8 features. 5 fell outside their ranges. The one most damaging falsification (cap-ratio inversion) was a direction error I didn't think to guard against. The audit code had the same blind spot until I caught it in implementation. Pre-registration doesn't make you correct; it makes your wrongness *visible*.
- **The audit primitive is now reusable.** `styxx gauntlet-audit-confounds` runs the full 8-oracle scan on any benchmark passed to it. Future submitters can run it before submitting to understand what their method is or isn't escaping. Future benchmark revisions can run it as part of validation.
- **Two confound-control bars now compose.** D3 catches length-tracking detectors. D4 catches cap-ratio-tracking detectors. A method that passes both has demonstrated signal beyond both of the two known artifacts. The bars compound in difficulty by what they additively rule out.

## The seven in-session falsifications

1. C1-profile ≤0.20 register-law bar (Pareto finding) — predicted, falsified by C10's 0.264 score
2. set_session-doesn't-propagate observation — falsified by per-agent routing
3. ICT-folklore auto-verdict PASS (probe label bug)
4. ICT-authoritative auto-verdict PASS (same shape)
5. styxx 7.7.5 wheel bundling miss
6. The gauntlet's v1 detection bars being length-gameable (D3 discovery)
7. **The benchmark's cap-ratio orthogonal confound (this finding) — and the prediction-direction blind spot in my own audit code**

## Reproducibility

- Pre-stated prediction (commit `48a9fe3`): `papers/consensus-hallucination/PRE_STATED_PREDICTION_confound_audit_2026_05_27.md`
- Implementation: `styxx/gauntlet.py::audit_confounds`, `CONFOUND_ORACLES`, `_capratio_oracle_detect`
- D4 bar wiring: `styxx/gauntlet.py::run_detection_gauntlet`, `DEFAULT_DETECTION_BARS["D4_capitalization_control_delta"]`
- Regression test: `tests/test_gauntlet.py::test_capratio_oracle_passes_D1_D2_but_fails_D4`
- CLI: `styxx gauntlet-audit-confounds` (added in `styxx/cli.py::cmd_gauntlet_audit_confounds`)
- Submission re-scores: updated `submissions/baseline_007_token_overlap/submission.json` + `submissions/baseline_008_embedding_similarity/submission.json` with `reported_v2_bars` (history) + new `reported` (4-bar)
- This finding: this commit.

`git log --oneline papers/consensus-hallucination/PRE_STATED_PREDICTION_confound_audit_2026_05_27.md styxx/gauntlet.py tests/test_gauntlet.py` shows the prediction-before-data ordering on origin.

## Honest open follow-ups

1. **The 5/8 prediction miss-rate.** My priors for individual feature AUCs were systematically too high in some places (`sentence_count`, `hedge_density`) and direction-wrong in another (`cap_ratio`). The audit-discipline pattern produces better calibration over time, but only if future predictions explicitly include direction. Lesson: pre-stated predictions for AUC should ALWAYS use `|AUC - 0.5|` or document direction explicitly.
2. **The orthogonality threshold (ρ < 0.5).** Locked pre-data. The cap-ratio confound has ρ = -0.34 to length. A stricter threshold (ρ < 0.3) would not have flagged it. The 0.5 cutoff is permissive; future work could tighten it if/when another confound surfaces at ρ in the 0.4-0.5 range.
3. **More candidate features.** The audit ran 8 oracles. Additional candidates not tested: numeric-token-ratio (truth answers like "1789", "206" are pure numbers), single-token-flag (truth answers are often single tokens), uppercase-ratio (different from cap-ratio: full-uppercase tokens). These are operator-territory follow-ups; the audit primitive makes them cheap to test.
4. **Benchmark v2 with format-equalized truth responses.** The deeper fix: re-curate truth records with full-sentence canonical answers matching the format distribution of the folklore class. Removes both the length and the cap-ratio confounds at the corpus level rather than controlling for them. Multi-hour operator-territory work — the audit primitive is the cheaper bridge until then.
