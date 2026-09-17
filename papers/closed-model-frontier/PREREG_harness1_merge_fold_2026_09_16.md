# PREREG — HARNESS-1: the corpus reconstruction stops folding merge traffic into a pull request

Fathom Lab · 2026-09-16 · Frozen after the correction appended to
`RESULT_external5_survivors_at_source_2026_09_16.md`, before the re-fold is run. This repairs the
EXTERNAL-1 harness, not the instrument; `styxx/diffgate.py` is untouched.

## The defect

`external1_harness.reconstruct` folds every row of `pr_commit_details` for a PR into one file set.
The dataset's rows come per commit, and a PR that merged its base branch carries that merge
commit's files — the base branch's traffic — as if the PR had changed them. EXTERNAL-5's
correction measured the consequence on the 19 overturned survivors: 9 of them are counts the fold
inflated this way. Two more sat on commits whose rows stop at 300 files, the dataset's per-commit
cap, so the corpus never saw the whole diff. On the corpus: **7,587 of 71,585** PRs with file rows
carry at least one commit whose message begins `Merge `; **974** carry a commit with 300 rows.

## The repair

- The shelf gains commit identity: rows keep `sha` and `message`. The fold takes every row whose
  commit message does not begin with `Merge ` (after leading whitespace). A PR whose rows are all
  merge commits keeps them (there is nothing else to read; reported).
- A PR with any commit at the 300-row cap is marked `capped`. Its `files_changed_count` claims are
  read as UNCHECKABLE with the reason `corpus rows capped at 300 per commit; the count cannot be
  reconstructed`; every other kind reads as before. The mark is the harness's, not the
  instrument's: it is applied to the ledger record after gating, and counted.
- Everything else in the reconstruction — one header pair per file, net status, patches appended
  — is unchanged.

## Gates — committed now

- **G-H1 (the 85 live PRs).** On the EXTERNAL-5 reachable PRs, the reconstruction's file count
  agrees with the live count for at least 14 more file-count items than before (today: 28 of the
  52 upheld and 0 of the 18 overturned file counts agree; the correction's check says 6 and 8
  flip, and one `only_touches` item besides), and no item that agreed before disagrees after.
  Blocking.
- **G-H2 (accusations can only fall where the fold could only inflate).** Ledger to ledger,
  `external4_ledger` (the fold as it was) against `external6_ledger` (the re-fold), same
  instrument on reconstructed diffs (BIN-2 changes nothing there, G-BIN-3): the count of
  `only_touches` CONTRADICTED claims does not rise, and the count of `compat_claim` readings that
  name a removed public definition does not rise. `files_changed_count` may move either way and is
  reported by direction. Blocking.
- **G-H3 (BC-2's invariants).** The three by-construction counters stay at 0 and no
  `tests_added` / `symbol_added` verdict changes on a PR without a merge commit or a cap.
  Blocking.
- **G-H4 (the numbers that get re-quoted).** Reported, not scored: the EXTERNAL-2 census line
  (accusations by kind), BC-2's survivors by kind, COMPAT-1's "PRs whose diff drops a public
  name", each before and after, so any figure already published can be corrected in place with
  the two numbers side by side.
- **G-H5 (what is not claimed).** No precision number. A merge commit identified by its message
  is a heuristic; a merge with a rewritten message is not caught, and the RESULT says how many
  PRs still disagree with the live count after the re-fold.

## Receipts

`external6_harness.py` (the re-fold; `--stage shelf` rebuilds the file table with commit identity),
`harness1_gates.py`, `harness1_gates.json`, `external6_summary.json`, `RESULT_harness1_…md`.
Ledgers gitignored; counts only.

---

*The corpus said 340 files where the pull request had 3, and the instrument accused on the
corpus's word. The fold is the harness's to fix, and the fix is measured against the pull
requests themselves before any published number is corrected.*
