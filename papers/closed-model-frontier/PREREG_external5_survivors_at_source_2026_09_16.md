# PREREG — EXTERNAL-5: the 96 surviving accusations, checked at the source

Fathom Lab · 2026-09-16 · Frozen before any live pull request is fetched. Follows
`RESULT_bc2_by_construction_lands_2026_09_16.md`, which left 96 accusations standing
(`files_changed_count` 75, `only_touches` 14, `tests_added` 7, on 91 pull requests) with no
precision attached.

## The question

BC-2 removed the accusations the instrument could not have supported. The 96 that survive are
accusations it *could* support — a count that differs, a path outside a prefix — made against a
corpus reconstruction (`pr_commit_details` folded into per-PR file statuses, EXTERNAL-1 harness).
The reconstruction is not the pull request. This run fetches each pull request as it stands on
GitHub today and asks the same question against the source: **does the accusation hold against
the live PR?**

## Population

The 96 claims with verdict CONTRADICTED in `external3_ledger.jsonl` (the BC-2 ledger; gitignored,
counts only in the receipts), keyed `(pr_id, claim_index)`. No sampling: all 96.

## The source facts, and how they are read

For each PR, fetched on the date of the run and recorded per item:

- **changed paths**: the set of unique file paths on the PR's `files` page (`/pull/N/files`), read
  from the page's embedded file records. The count of that set is the live changed-files count.
  A PR whose page reports files not shown (GitHub truncates very large diffs) is UNREACHABLE for
  this run, as is a 404, a private or deleted repository, or a fetch error.
- **added Python lines** (`tests_added` only): the per-file `patch` from the REST endpoint
  `pulls/N/files`; a `.py` file whose patch is missing (GitHub omits patches above a size) makes
  the item UNREACHABLE.
- **cross-check of the page reading**: for 20 PRs drawn with `random.Random(20260916)` from the
  91, the REST `changed_files` field is fetched and compared with the page count. Reported;
  disagreement on any of the 20 voids the page reading for `files_changed_count` and the kind is
  reported as UNREACHABLE in full.

The re-reading per kind is the instrument's own rule, applied to the live facts:

- `files_changed_count`: claim `n` against the live count. Differ → **UPHELD**; equal → **OVERTURNED**.
- `only_touches`: the BC-2 prefix rules (`_norm`, path-shaped, second prefix after "and") against
  the live paths. Any path outside → UPHELD; none → OVERTURNED.
- `tests_added`: `^\s*def test_` over the added lines of the live `.py` patches against `n`. Differ
  → UPHELD; equal → OVERTURNED.

## What the outcomes mean, said before the numbers

- **UPHELD**: the description does not match the pull request as it stands. It is not proof the
  author lied when writing; the PR may have taken commits after the description was written, and a
  pasted `git diff --stat` line describes one moment. That caveat is the whole reason this is not
  a panel result.
- **OVERTURNED**: the corpus reconstruction disagreed with GitHub and the instrument accused on the
  reconstruction's word. That is a harness artifact, not an instrument defect, and it is reported
  as the reconstruction's error rate on this population.
- **UNREACHABLE**: not counted either way.

## Gates — committed now

- **G-E5-1 (reach).** ≥ 90% of the 96 reachable, else every figure below is reported as partial
  and no per-kind rate is quoted in prose.
- **G-E5-2 (report, not scored).** UPHELD / (UPHELD + OVERTURNED) per kind, and for the two
  `files_changed_count` shapes the BC-2 result named (a pasted git stat line, 31; other, 44).
- **G-E5-3 (the floor question, consequence pre-committed).** The EXTERNAL-1 floor is 0.95. If a
  kind's at-source rate is below it, the accusation for that kind — or for the failing shape, when
  the shapes separate — is owed a repair prereg that withholds it (a pasted stat line becoming
  UNCHECKABLE, "a stat line describes a commit, not the pull request"). Nothing is changed in the
  instrument by this run.
- **G-E5-4 (what is not claimed).** No precision figure for the instrument. No agent comparison
  in prose. No PR named in any receipt; URLs stay in the gitignored per-item file.

## Receipts

`external5_source.py` (fetch + re-read, deterministic given the recorded facts),
`external5_items.jsonl` (per item: the live facts and the outcome; gitignored),
`external5_summary.json` (counts only; committed), `RESULT_external5_…md`.

---

*BC-2 said 96 survive and attached no precision. This asks the pull requests themselves, names
the timing caveat before the count, and pre-commits what a bad number costs.*
