# Pre-stated prediction · v6_uncurated_audit_2026_05_28 (Layer 7)

**Date:** 2026-05-28
**Author:** Alexander Rodabaugh (Fathom Lab)
**Substrate:** styxx 7.7.10 at HEAD `5c39f32` (post-v6 paper commit)
**Pre-state-of-art:** Layers 5 and 6 ran modal-pre-stated-PASS outcomes. The operator explicitly flagged this as a methodological weakness: "every L5/L6 run was a modal pre-stated outcome — I never used the instruments to FIND something I didn't already know." Layer 7 is the corrective: an **uncurated** audit of v6 itself, with the prediction *deliberately not engineered to pass*.

## What changed from L5/L6 to L7

L5 and L6 selected claims known by the operator to hold (substrate-state checks the operator had personally performed by hand) and asked the instrument to confirm them. This bounds the instrument's value: a buggy "always PASS" instrument would produce identical output.

L7 inverts this: **the audit's claim set is the LIVE PAPER (v6), unfiltered**. Numerical, structural, and count-based claims are extracted from the v6 paper and the paper-relevant FINDINGs without curating for predicted outcome. The audit is allowed — and *expected* — to find failures.

## Honest pre-disclosure

During the prereg-writing pass, the operator manually identified two candidate failures by eyeballing v6:

| candidate | v6 says | actual | suspected verdict |
|---|---|---|---|
| FINDING document count (abstract closing list) | "ten FINDING documents" | 13 files matching `papers/agent-self-audit/FINDING_*.md` at HEAD | **FAIL** expected |
| Reference baseline count (abstract closing list) | "nineteen reference baselines" | 18 directories matching `submissions/baseline_*/` at HEAD | **FAIL** expected |

These two were spotted during a 5-minute manual review. The prediction below treats them as **pre-disclosed expected failures**: the audit is expected to confirm both. If the audit *contradicts* either (returns PASS where I predicted FAIL), my manual review was wrong and the auditor is more rigorous than my eyeballing — also informative.

## Methodology

The L7 audit uses the same `styxx.agent_audit` instrument shipped at `3c24b5e` (extended in `5c39f32` with `__all__` exposure), with checker `file_at_path_contains` and the new helper `directory_file_count_equals` (added in this Layer-7 commit BEFORE the run). For non-trivial claims a second checker is layered in.

Claims are extracted from:
- the v6 abstract (`papers/PAPER_recursive_discipline_2026_05_27.md` lines 10–13)
- §11.5 (the v3 measurement narrative)
- §13 (the same-session self-falsification)
- §14 (the instrumented recursion frame)
- the reproducibility table (lines ~400–460)
- the paper-relevant FINDINGs (L5 + L6 + asymmetry-v3 + first-PASS)

Per claim, the runner records `(claim, expected, actual, verdict)`. The verdict is PASS if `actual == expected`; FAIL if not; ERROR if the checker raised.

## The pre-stated claim set (30 claims)

### Numerical counts (the suspected-FAIL zone)

| id | claim | expected check |
|---|---|---|
| N1 | "ten FINDING documents" appears in v6 abstract | substring present in paper file |
| N2 | actual count of `papers/agent-self-audit/FINDING_*.md` files equals 10 | **predicted FAIL** (actual 13) |
| N3 | "nineteen reference baselines" appears in v6 abstract | substring present |
| N4 | actual count of `submissions/baseline_*/` directories equals 19 | **predicted FAIL** (actual 18) |
| N5 | actual count of `submissions/baseline_*/` directories equals 18 | predicted PASS (the ground truth) |
| N6 | "108-record benchmark" appears in v6 abstract | substring present |
| N7 | `papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json` records list has length 108 | predicted PASS |
| N8 | "1086 passed, 8 skipped" appears verbatim in v6 §13 | predicted PASS |
| N9 | "42/42" appears in v6 (public-surface test count) | predicted PASS |

### Per-claim numerical facts (asymmetry table)

| id | claim | expected check |
|---|---|---|
| N10 | v6 §11.5 + §14.2 say dark-core asymmetry rate = 5.88% | substring `5.88%` present |
| N11 | v6 §11.5 + §14.2 say TruthfulQA asymmetry rate = 17.00% | substring `17.00%` present |
| N12 | L5 results.json reports 13 PASS, 0 FAIL | python parse, exact equality |
| N13 | L6 results.json reports 18 PASS, 0 FAIL | python parse, exact equality |
| N14 | L6 results.json reports `killgate_paper_grade_fired == false` | python parse |
| N15 | L6 results.json reports `killgate_instrument_grade_fired == false` | python parse |
| N16 | L6 results.json: all 13 TRUE claims have observed `P(NO) == 0.0` | python parse |
| N17 | L6 results.json: all 5 FALSE claims have observed `P(NO) == 1.0` | python parse |

### Commit-reference checks (reproducibility table)

| id | claim | expected check |
|---|---|---|
| C1 | commit `ed663ca` exists in git history | git rev-parse |
| C2 | commit `0e97598` exists in git history | git rev-parse |
| C3 | commit `87ca52d` exists in git history | git rev-parse |
| C4 | commit `1b82e27` exists in git history | git rev-parse |
| C5 | commit `3c24b5e` exists in git history | git rev-parse |
| C6 | commit `a8fb1f3` exists in git history | git rev-parse |
| C7 | commit `05adebf` exists in git history | git rev-parse |
| C8 | commit `1ab0e22` exists in git history (critique_detector public API) | git rev-parse |

### Structural / API-surface checks

| id | claim | expected check |
|---|---|---|
| S1 | `styxx.__version__ == "7.7.10"` | python attr lookup |
| S2 | `"critique_detector" in styxx.__all__` | python attr lookup |
| S3 | `"agent_audit" in styxx.__all__` | python attr lookup |
| S4 | `"AgentClaimAuditor" in styxx.__all__` | python attr lookup |
| S5 | git tag `v7.7.10` does NOT exist | git tag listing |

### Section-position checks (against the rendered PDF)

| id | claim | expected check |
|---|---|---|
| P1 | v6 PDF has 17 pages | pypdf |
| P2 | v6 PDF contains section `"13. The paper catches itself"` | pypdf text extraction |
| P3 | v6 PDF contains section `"14. The instrumented recursion frame"` | pypdf text extraction |

## Predicted verdicts

| count | claims |
|---|---|
| FAIL (deliberately) | N2, N4 — the two count-drifts spotted manually |
| PASS | N1, N3, N5–N17, C1–C8, S1–S5, P1–P3 (28 claims) |
| **Joint prediction** | **28 PASS / 2 FAIL out of 30** |

## Pre-stated kill-gates

- **Low end (< 2 FAILs):** if the audit returns 0 or 1 FAIL, my manual eyeballing was wrong about at least one of N2/N4. Either the count I computed is itself wrong (probable if N2 — I may have miscounted FINDINGs), or the substring check resolves differently than I expect. Either way, the instrument is **more rigorous than the manual review**, which is a useful inversion.
- **Match (exactly 2 FAILs):** confirms the manual review and the instrument agree. Boring but informative.
- **High end (> 2 FAILs):** the audit finds drift I missed in the manual review. **Each additional FAIL is a real discipline gap that should be fixed in v7** (or honestly documented if not feasible to fix).
- **Outlier (≥ 5 FAILs):** paper-grade discipline drift in v6 that warrants more than a count-fix — possibly a §15 or a paper-revision protocol section.

## What this is NOT

- **Not** a generalization to other papers or substrates.
- **Not** a claim that the audit catches ALL drift — only structured numerical/factual/commit-reference claims are checkable by `agent_audit`. Interpretive claims, causal claims, and generalization claims remain outside the instrument's scope.
- **Not** "use it on yourself" in the sense of auditing every word of the paper. It is bounded to substrate-checkable propositions.

## What it IS

The first L5/L6/L7 run where the prediction is **deliberately non-modal**: I am predicting failures, naming the ones I expect, and committing the audit code to find more if they exist. The operator's frame "modal-PASS runs are smoke tests dressed as experiments" is the calibration. L7 inverts that.

## Reproducibility

| artifact | path | committed at |
|---|---|---|
| this pre-registration | `papers/agent-self-audit/PRE_STATED_PREDICTION_v6_uncurated_audit_2026_05_28.md` | this commit (BEFORE the runner exists) |
| runner | `experiments/v6_uncurated_audit_2026_05_28/run_audit.py` | (after this commit) |
| results | `experiments/v6_uncurated_audit_2026_05_28/results.json` | (after run) |
| FINDING | `papers/agent-self-audit/FINDING_v6_uncurated_audit_2026_05_28.md` | (after results) |
| fix commit (if FAILs are real) | TBD as v7 paper revision | (after FINDING) |

Git timestamps enforce ordering. The runner is local (no OpenAI API calls); free to run.
