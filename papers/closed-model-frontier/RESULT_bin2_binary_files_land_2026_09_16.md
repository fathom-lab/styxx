# RESULT — BIN-2 lands: the raw-diff doors see binary files, renames and mode changes; nothing else moved

Fathom Lab · 2026-09-16 · Prereg: `PREREG_bin2_binary_files_2026_09_16.md` (re-frozen after
`RESULT_bin1_INVALID_2026_09_16.md`; the repair is BIN-1's, unchanged). Receipts: `bin2_gates.py`,
`bin2_gates.json` (and `bin1_gates.py` / `bin1_gates.json`, the INVALID run, kept). Instrument:
`styxx/diffgate.py` sha256 `397624d5…`; `web/gate/diffgate.js` re-cut on it. Counts only; no PR named.

## The gates

| gate | result |
|---|---|
| G-BIN-1 (restated) the 91 live diffs, parse count = distinct normalised header paths on every served PR | 85 served, **85** equal — **PASS** (raw header count: 84, the dotfile collision, filed separately) |
| G-BIN-2 the differential: 0 disagreements; the 3,199 pre-repair records identical | 3,205 pairs, 6,914 claims, **0** disagreements; **3,199 of 3,199** identical — PASS |
| G-BIN-3 a 2,000-PR corpus sample re-derived identically | **2,000 of 2,000** (966 claims) — PASS |
| G-BIN-4 EXTERNAL-5 re-read under the repair | parse = live count on **90 of 90** served items; the 11 items on binary-carrying diffs now counted by the parse; the same outcome on **96 of 96** — PASS |
| G-BIN-5 suite and demo | 4,509 passed, 6 xfailed (the two BC-2 strict pins and four pre-existing), 142 skipped, the two known `test_sworn_*` committed-sample failures deselected as before; demo unchanged — PASS |

## What changed, in the instrument's own words

Before, on a diff of one text file and one new icon with the sentence "2 files changed":

    [LIE] files_changed_count   diff changes 1 files, claim says 2

After:

    [ok ] files_changed_count   diff changes 2 files, claim says 2

Before, on the same diff with "Only touches src/.":

    [ok ] only_touches          all changed paths under prefix

After:

    [LIE] only_touches          paths outside 'src': ['assets/logo.png']

Both directions were wrong before and both are right now; the six pinned pairs in
`web/gate/differential/bin1_pairs.json` hold each one, on the Python side
(`tests/test_diffgate_bin1.py`) and the JavaScript side (`check_pairs.js`).

## What did not change

Every one of the 3,199 differential records from before the repair is byte-identical after it;
2,000 corpus PRs re-derive to the same claims and verdicts, because the EXTERNAL-1 reconstruction
emits a header pair for every file and never reaches the new branch; EXTERNAL-5's 96 outcomes
are the same 70 / 19 / 7. The added-lines blob is untouched, so `tests_added`, `symbol_added` and
`compat_claim` could not move and did not.

## Deviations and the one thing learned

The repair is eleven lines in Python and the same in JavaScript, as the prereg guessed. The
freeze took two tries: BIN-1's raw-header gate caught a dotfile and its undotted twin sharing one
key under `_norm`, a normaliser every door has used since the first release. That is a real,
separate defect — a status map keyed by normalised path cannot hold `x` and `.x` at once — with
its own issue; it is not repaired here, and the restated gate says exactly what the parser can
be held to instead of hiding it.

---

*The blind spot took one afternoon to find at the source and eleven lines to close, and the
closing was measured on the same ninety-one pull requests, the same three thousand pairs and
two thousand corpus PRs that could have moved and did not.*
