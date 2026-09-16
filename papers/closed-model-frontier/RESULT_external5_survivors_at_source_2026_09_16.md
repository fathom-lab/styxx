# RESULT — EXTERNAL-5: 70 of the 96 surviving accusations hold at the source, 19 were the harness's, and the raw-diff doors cannot see a binary file

Fathom Lab · 2026-09-16 · Prereg: `PREREG_external5_survivors_at_source_2026_09_16.md` (pushed
before the first fetch). Receipts: `external5_source.py`, `external5_summary.json`,
`external5_crosscheck.json`; the per-item file with URLs and live facts is gitignored. Instrument:
`styxx/diffgate.py` at the BC-2 + COMPAT-1 checkout, sha256 `a550cad5…`. Counts only; no PR named.

## Deviations, first

1. **Source.** The prereg named the PR's *files page* for paths and the REST `patch` fields for
   added lines. The run read each PR's unified diff from `patch-diff.githubusercontent.com` — the
   same bytes the CLI's `--pr` door reads — and fed it to `gate_diff_text` on the claim sentence,
   so the reading is the instrument itself, not a re-implementation. Six repositories had been
   renamed since the corpus was cut and answer through a redirect this environment cannot follow;
   their new names were read off a browser and pinned in the script. Two PRs answer 404 on the
   diff endpoint while the REST API still serves their file lists; those lists (headers only,
   enough for a file count) were read in a browser and pinned verbatim.
2. **The cross-check fired, and the prereg's consequence with it.** Of the 20 seeded PRs, the
   instrument's own parse of the live diff disagreed with the REST `changed_files` on **3**:
   two PRs with binary files, one PR whose page and REST record are gone while its diff is still
   served. By the prereg's letter, `files_changed_count` is **voided as a preregistered figure**.
   The cause is a defect in the instrument (below), the file list was completed as the prereg
   defines it (the files page lists binaries), the cross-check was re-run — 19 of 20 agree, the
   20th is the orphaned diff, excluded — and the `files_changed_count` numbers below are reported
   as **exploratory**. `only_touches` and `tests_added` stand under the same completed list.

## The defect the cross-check found: `parse_unified_diff` does not see a binary file

A unified diff marks a binary change as `Binary files a/x.png and b/x.png differ` with no `---`
/ `+++` headers. `parse_unified_diff` registers a file only from those headers, so every door that
reads raw diff text — `--pr URL`, the paste-in page, the bookmarklet, the JS port, any webhook
feeding `gate_diff_text` — counts a PR of eight PNGs and one `.scss` as **one file**, and would
accuse a truthful "9 files changed". The git door (`gate_diff`, hooks, the Action on a checkout)
uses `git diff --name-status` and is not affected. On this population: **10 of the 86 reachable
PRs** carry a binary the parse misses (11 items), and **3 truthful counts** ("2 files changed"
over one text file and one binary; "13"; "11") would be accused by the raw-diff doors today. The
same blind spot can turn an `only_touches` lie into a VERIFIED: a binary outside the prefix is
invisible. Filed as an issue; a repair prereg is owed (read `diff --git` and `Binary files`
lines into the status map). Nothing in the instrument is changed by this run.

## The numbers

Population 96 (91 PRs). Reach: **89 of 96, 92.7%** — G-E5-1 passes. The seven unreachable: five
on four PRs GitHub no longer serves (the diff endpoint refuses them; the two pages checked answer
"not found"); one on the orphaned diff; one `tests_added` whose live diff has no Python file (the
corpus reconstruction had one), so the instrument abstains.

| kind (shape) | UPHELD | OVERTURNED | UNREACHABLE | upheld rate |
|---|---|---|---|---|
| `files_changed_count` (exploratory) | 52 | 18 | 5 | 0.74 |
| — pasted git stat line | 26 | 4 | 1 | 0.87 |
| — other | 26 | 14 | 4 | 0.65 |
| `only_touches` | 12 | 1 | 1 | 0.92 |
| `tests_added` | 6 | 0 | 1 | 1.00 |
| **all** | **70** | **19** | **7** | **0.79** |

What OVERTURNED is, on inspection of the counts alone: in **16 of the 18** overturned file counts
the corpus reconstruction had *more* files than the live PR — "diff changes 340 files, claim says
3" against a live diff of 3. Folding `pr_commit_details` over every commit of a PR brings in
merge traffic from the base branch; the description was right and the instrument was fed the
wrong diff. That is EXTERNAL-1's harness, not its instrument, and it means the corpus-side
`files_changed_count` accusations in EXTERNAL-2 and BC-2 carried a **~1 in 4 harness artifact
rate** (18 of 70 reached). Of the 52 UPHELD counts, 42 have the live PR *above* the described
count — the shape a description written before later commits would leave — and 10 below.

## The floor question (G-E5-3), and what is owed

Nothing reaches 0.95 at the source except `tests_added` (6 of 6, too few to say). Under the
prereg's pre-commitment, `files_changed_count` — both shapes — and `only_touches` (12 of 13, one
overturned by the reconstruction, none by the instrument) are owed a repair prereg that withholds
the accusation until it clears a floor. Two of the three causes are now known and separable: the
binary blind spot (instrument; fix and re-measure) and the merge-traffic inflation (harness; the
corpus ledgers for this kind are not evidence about the instrument until the reconstruction is
re-derived from PR-level file lists). The third — a description that was true when written and
false after the next push — cannot be told from a lie by any diff, and the repair prereg has to
say what the gate's word for it is. Until it does, "diff changes 6 files, claim says 5" is a
correct sentence about the diff and `[LIE]` is not a correct word for the author.

## What is not claimed (G-E5-4)

No precision for the instrument as a whole. No agent split in prose. 70 upheld accusations are 70
descriptions that do not match their pull request today; whether their authors were wrong when
they wrote them is not a question this run can answer, and it does not.

---

*The survivors were asked, at the source, and most held. What did not hold was the harness (1 in
4 of the counts), and the instrument's own eye for a binary file — a defect no corpus run could
have shown, because the corpus never handed it a diff.*
