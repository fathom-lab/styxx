# PREREG — BIN-1: the raw-diff doors learn to see a binary file (#118)

Fathom Lab · 2026-09-16 · Frozen after `RESULT_external5_survivors_at_source_2026_09_16.md` and
issue #118, before the repair is written. Same shape as BC-2: one mechanical defect, one repair,
gates that a differential and a re-read can score, and a statement of what may not move.

## The defect

`parse_unified_diff` (and `parse_unified_diff_sides`) register a file only when a `--- a/…` /
`+++ b/…` header pair appears. A unified diff carries a binary change as

    diff --git a/x.png b/x.png
    new file mode 100644
    Binary files /dev/null and b/x.png differ

with no such pair, so the file never enters the status map. The same holds for a mode-only change
(`old mode` / `new mode`, no hunks) and a pure rename (`similarity index 100%`, `rename from`,
`rename to`, no hunks). Every door that reads raw diff text inherits the blind spot: `--pr URL`,
`gate_diff_text` on a payload, the paste-in page, the bookmarklet and the JS port, GitLab CI when
fed a diff. The git door takes statuses from `git diff --name-status` and is not affected. On the
86 live PRs EXTERNAL-5 reached, 10 carry a file the parse misses; three truthful file counts
would be accused today.

## The repair

A `diff --git a/X b/Y` header that reaches the next header (or the end of the text) without a
`---`/`+++` pair registers a file anyway:

- path `Y` (the b side), status **A** if a `new file mode` line or a `Binary files /dev/null and …`
  line follows the header; path `X` with status **D** if `deleted file mode` or
  `Binary files … and /dev/null differ` follows; otherwise path `Y`, status **M** (a modified
  binary, a mode change, a pure rename — the git door reports a pure rename as `R`, and `_gate`
  treats any status that is not `A` or `D` as a touch, so `M` is the same reading).
- `rename to` / `rename from` lines, when present, name the paths; otherwise the header is split
  on its last ` b/`.
- `parse_unified_diff_sides` registers the same path with empty added and removed lists.
- The JS port does the same, byte for byte on `why`.

Nothing else moves: a file that has hunks is registered exactly as before, from its headers; the
added-lines blob is unchanged (a binary adds no lines); the `tests_added`, `symbol_added` and
`compat_claim` readings are untouched by construction.

## Gates — committed now

- **G-BIN-1 (the 91 live diffs).** Re-fetch the EXTERNAL-5 diffs. For every PR served, the
  repaired parse's file count equals the count of `diff --git` headers (today: 10 of 86
  disagree). Blocking. Counts only in the receipt; the diff's sha256 stands in for its URL.
- **G-BIN-2 (the differential).** Python and JS agree on all 3,199 existing pairs and on new
  pinned pairs — a binary added, modified and deleted, a pure rename, a mode-only change, a
  binary beside a text file with a `files_changed_count` claim that is now true, an
  `only_touches` claim with a binary outside the prefix that is now caught — 0 disagreements.
  Every existing pair's record is identical before and after (no binary in the corpus today, so
  nothing else may move). Blocking.
- **G-BIN-3 (the corpus is untouched).** The EXTERNAL-1 reconstruction emits a `---`/`+++` pair
  for every file, binaries included, so no reconstructed diff reaches the new branch: the BC-2
  and COMPAT-1 ledgers re-derive identically. Checked on a 2,000-PR sample of `external4_ledger`
  (claims and verdicts identical), not the full corpus, for time. Reported, and blocking on the
  sample.
- **G-BIN-4 (EXTERNAL-5 re-read).** `external5_source.py` re-run under the repaired instrument:
  `files_by_parse` equals `live_files` on every reached PR, the seeded REST cross-check agrees on
  19 of 20 under the instrument's own parse (the orphaned diff aside), and the three truthful
  counts read VERIFIED. The outcome table is otherwise identical to the RESULT's completed-list
  reading. Blocking.
- **G-BIN-5 (suite and demo).** Full suite green; demo unchanged; the differential corpus's
  `bc1_pairs` and `compat_pairs` unchanged.

## What is not claimed

No precision number. The `files_changed_count` withholding question EXTERNAL-5 raised is not
answered here; this removes one of its three causes and says so. A binary whose header GitHub
omits entirely (none seen) stays invisible, and the repair does not pretend otherwise.

---

*Ten of eighty-six pull requests carried a file the gate could not see, and three honest counts
were one click from being called lies. The fix is eleven lines and a differential; the prereg
is so the eleven lines cannot quietly become twelve.*
