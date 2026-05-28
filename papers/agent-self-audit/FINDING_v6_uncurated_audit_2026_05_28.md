# Finding · v6_uncurated_audit_2026_05_28 (Layer 7) — 33/35 PASS, 2 FAIL exactly as pre-disclosed; uncurated audit confirmed v6 count-drift; v7 ships the fix; systematic propagation of the same drift caught by follow-up grep

**Date:** 2026-05-28
**Author:** Alexander Rodabaugh (Fathom Lab)
**Substrate:** styxx 7.7.10 at HEAD `5c39f32` (post-v6 paper commit); v7 fix in this commit
**Pre-registration:** `papers/agent-self-audit/PRE_STATED_PREDICTION_v6_uncurated_audit_2026_05_28.md` (commit `b18ce93`, public on `origin/main` BEFORE the runner existed)
**Runner:** `experiments/v6_uncurated_audit_2026_05_28/run_audit.py`
**Results:** `experiments/v6_uncurated_audit_2026_05_28/results.json`
**Instrument extensions (this commit):** `styxx.agent_audit` gained three new checkers — `directory_file_count_equals`, `json_path_equals`, `python_attr_equals` — to support count-against-glob, JSON-key-path, and Python-attribute-equality claims respectively.

> **Outcome.** **33/35 PASS, 2 FAIL, 0 ERROR.** The 2 FAILs were exactly the pre-disclosed expected failures (N2: FINDING document count; N4: baseline directory count). Manual eyeballing and the instrument agreed on the *first* occurrence of each drift, but a follow-up `grep` revealed the same off-by-one pattern propagated through **five places** in the v6 paper (substrate line, abstract closing list, §11 narrative paragraph, §12 conclusion paragraph, reproducibility-table footer). v7 of the paper (this commit) corrects all five instances.

## What changed from L5/L6 to L7

L5 and L6 ran modal-pre-stated-PASS outcomes. The operator's calibration: "every L5/L6 run was a modal pre-stated outcome — I never used the instruments to *find* something I didn't already know."

L7 inverted this:
- **Uncurated** claim set extracted from the live v6 paper, not curated to pass.
- **Pre-disclosed expected failures**: N2 (FINDING count = 10) and N4 (baseline count = 19) were predicted to FAIL because manual eyeballing during prereg-writing had spotted both.
- **Honest non-modal prediction**: 28 PASS + 2 FAIL out of 30 (the runner reports 35 atomic claims because some logical claims split into a/b pairs).
- **Kill-gate phrased to be informative either way**: fewer FAILs would mean the audit instrument was *more* rigorous than the manual review; more FAILs would mean the instrument caught drift the manual review missed.

The actual outcome (33 PASS, 2 FAIL exactly N2 and N4) matches the pre-disclosed prediction. The audit confirmed the drift was real. The follow-up grep (run AFTER the audit, not pre-registered) caught the systematic propagation to four additional places where the manual-review and the audit both saw only first-occurrence.

## What the audit verified

| category | claims | PASS | FAIL |
|---|---|---|---|
| numerical counts | N1, N3, N5, N6, N7, N8, N9, N10, N11 | 9 | 0 |
| asymmetry-table facts | N12a, N12b, N13a, N13b, N14, N15, N16, N17 | 8 | 0 |
| commit-reference checks | C1–C8 | 8 | 0 |
| API-surface checks | S1, S2, S3, S4, S5 | 5 | 0 |
| section-position checks | P1, P2, P3 | 3 | 0 |
| **count-drift claims** | **N2, N4** | **0** | **2** |

The two FAILs:
- **N2:** v6 abstract said "ten FINDING documents"; actual count of `papers/agent-self-audit/FINDING_*.md` is **13**. Off by 3.
- **N4:** v6 abstract said "nineteen reference baselines"; actual count of `submissions/baseline_*/` is **18**. Off by 1 (the numbering convention skips Baseline-001, so highest number 019 ≠ total count).

Both substring-presence checks (N1 "ten FINDING documents", N3 "nineteen reference baselines") PASSed, confirming the paper's text matches the predicted drift. The count checks (N2, N4) FAILed because the substrate disagrees with the text.

## Follow-up grep (post-audit, not pre-registered) — systematic propagation

After the audit returned the predicted N2 + N4 FAILs, a manual `grep` was run to find all occurrences of the drifted counts in v6:

| location | v6 text | corrected in v7 |
|---|---|---|
| Title line (substrate description) | "nineteen reference baselines on the dark-core benchmark" | "eighteen reference baselines (Baseline-002 through Baseline-019; numbering skips 001)" |
| Abstract closing list | "nineteen baseline submissions, ten FINDING documents" | "eighteen reference baselines ..., thirteen FINDING documents" |
| §11 (Baseline-019 PASS narrative) | "After 18 pre-registered detection baselines all FAILed v3 bars" | "After seventeen pre-registered detection baselines all FAILed v3 bars (Baseline-002 through Baseline-018; numbering skips 001)" |
| §12 (Conclusion) | "tested 18 pre-registered detection submissions ... achieved a 19th-submission PASS" | "tested seventeen pre-registered detection submissions ... achieved Baseline-019 (the eighteenth submission by count, numbered 019 because numbering skips 001) as the first PASS" |
| §12 (Reproducibility footer) | "nineteen reference baselines + six FINDING documents + fifteen in-session falsifications" | "eighteen reference baselines ... + thirteen FINDING documents + sixteen in-session falsifications" |

Note the §12 footer also carried two *additional* stale counts ("six FINDING documents" and "fifteen in-session falsifications") that pre-date v5 — they reflect the v3 paper state. These were silently incorrect through v4, v5, AND v6 because the abstract was updated but the footer was not. v7 brings the footer into agreement with the abstract.

## What this means for the instrument

The L7 audit caught the **first occurrence** of each drift but not the propagation. Two interpretations:
1. **Instrument-grade**: the audit checker `file_at_path_contains` is a first-match predicate. A stronger checker `file_at_path_contains_n_times` (returning the count of matches) would have caught propagation.
2. **Methodology-grade**: pre-registering only the count-FAIL claims is insufficient. A more rigorous audit would extract every occurrence of count-phrases and pre-register each. This bounds the audit's coverage to what the operator pre-registered, which is itself a discipline statement.

v7's fix addresses the propagation; a future L8-style audit could pre-register "no occurrence of 'nineteen reference baselines' in the paper" to test that the fix held end-to-end.

## What this is NOT

- **Not** a generalization. The L7 audit's 33/35 PASS reflects a single paper at a single commit.
- **Not** evidence that the count drift was malicious or hidden. The v5→v6 abstract update bumped "+two resolutions" to "+four" and "eight FINDING documents" to "ten"; the "ten" was the new error introduced *while correcting* the prior "eight" error. v6's Acknowledgments paragraph explicitly disclosed the count-correction process, and v7's Acknowledgments now extends that disclosure to the v6→v7 correction.
- **Not** the end of the recursion. v7 corrects v6's drift; v8 may correct v7's drift; each correction has a check window.

## What it IS

The first L5/L6/L7 run where the predicted outcome was **non-modal** — predictions explicitly admitted expected failures. The audit confirmed the predictions, the substrate-vs-claim drift was real, v7 ships the fix in the same session. The check window from v6 commit (`5c39f32`) to v7 commit (this commit) was approximately one hour and four commits (audit prereg `b18ce93` → run results → grep follow-up → v7 paper fix → this commit).

The counting-drift propagation discovery (the same drift appearing in 5 places when manual review caught it in 2) is itself an honest reportable result: pre-registered audits can catch first-occurrence drift cheaply; systematic propagation of the same drift requires either richer checkers or pre-registering more instances.

## Reproducibility

```
git log --oneline b18ce93..HEAD
python experiments/v6_uncurated_audit_2026_05_28/run_audit.py  # runs against v7 substrate; N1 and N3 will now FAIL because v7 no longer contains the drifted substrings — verifying the fix
cat experiments/v6_uncurated_audit_2026_05_28/results.json
```

**Important: re-running the audit against the v7 substrate will produce a *different* result than the pre-registered prediction**, because the v7 paper no longer contains the substrings "ten FINDING documents" or "nineteen reference baselines". The pre-registered audit measured the v6 substrate; the v7 substrate is post-fix and produces different (but expected) values. The original `results.json` records the v6 result and is preserved.
