# RESULT — BC-1 is INVALID against its own preregistration: G-B2 fails by one case, G-B3 by seven

Fathom Lab · 2026-09-16 · Prereg: `PREREG_bc1_by_construction_2026_09_16.md`, pushed at the head of
`fathomlab-patch-14` before the run. Receipts: `external3_harness.py`, `external3_gates.py`,
`external3_base_gate_summary.json` (the checkout at the prereg commit, without the repair),
`external3_gate_summary.json` (the same checkout with BC-1), `external3_gates.json`. Ledgers
gitignored; counts only. The repair described here does not land; BC-2 replaces it.

## What ran

The four repairs were implemented as frozen and the corpus was gated twice with the same harness:
once at the prereg commit (baseline: main plus the prereg document, no repair) and once with the
repair. Comparing against the checkout rather than the wheel is deliberate and is said in the
harness: main already reads fewer claims than the 7.47.0 wheel (V14 demotes more `file_touched`
sentences: 12,957 covered PRs against the wheel's 15,762), and that difference is not this repair's
to claim or answer for. The accusation set is the same in both baselines: 665 claims, 625 PRs.

## The gates, as frozen

- **G-B1 (subset invariant) — PASS.** Keyed on `(pr_id, kind, claim text)`: 664 accusation keys
  before (665 claims, one duplicate key), 94 after, 0 new. 570 removed: `only_touches` 328,
  `tests_added` 177, `symbol_added` 65.
- **G-B2 (by construction) — FAIL, by one.** `tests_added` and `symbol_added` accusations in diffs
  without Python: 0 and 0. `only_touches` accusations with a prefix the census calls not a path: 1.
  The one case is a prefix written `app1/`: the instrument's rule counts a trailing slash as a path
  signal and accuses; the census's rule strips the slash first and calls the remainder a word. The
  instrument is right about the slash and the census was wrong; but the prereg said the census's
  counters decide, so this fails. (The sentence itself is an example — "when deploying `app2`,
  they don't want to see commits that only touched files under `app1/`" — a claim about a
  hypothetical, not about the diff. That class is real and is not this cycle's.)
- **G-B3 (the verified side untouched) — FAIL, by seven.** Of the 18 VERIFIED claims the prereg
  named, 7 are no longer VERIFIED. Six are `tests_added` sentences that count "test cases" or
  "test classes" whose count matched the `def test_` count — repair 2 as frozen makes every such
  sentence UNCHECKABLE, matching count or not, so the prereg contradicted itself: it froze a rule
  that abstains on those sentences and a gate that requires them to stay verified. The seventh is
  an `only_touches` sentence — "Only modified files in `/docs/platform/`, not
  `/docusaurus/…`" — where the two-prefix reading, allowed after a comma, captured `not` as the
  second prefix and abstained on it. The comma form was a mistake; a second prefix follows "and".
- **G-B4 (demo and suite)** — the demo names the same three lies; the suite is green with the BC-1
  tests and two `xfail(strict=True)` pins (`tests/test_diffgate_bc1.py`). Not decisive, reported.
- **G-B5 (survivors)** — 94 accusations after: `files_changed_count` 74 (untouched by design),
  `only_touches` 13 (11 one-prefix path-shaped, 2 two-prefix), `tests_added` 7 (every one a
  Python diff where both counts are positive and differ). No precision is attached to any of them.
- **G-B6** — the four pairs are in `web/gate/differential/bc1_pairs.json` with the Python's
  verdicts; the port's disagreement is drift until it is updated.

## What this means

The prereg's rules were the census's rules moved into the instrument, and the measurement is that
moving them removed 570 accusations and added none — the number the census predicted, plus the
counted-noun and path-shape cases the census had not separated out. The cycle still fails, because
two of its own gates were written against it: a rule that abstains on matching "test cases" counts
cannot coexist with a gate that keeps those counts verified, and a census rule that disagrees with
the instrument's rule on a trailing slash cannot be the instrument's judge. Both are the prereg's
errors, not the corpus's, and the protocol's answer to a prereg error is a new prereg, not an edit.

BC-2 (`PREREG_bc2_by_construction_2026_09_16.md`) changes exactly three things and nothing else:
repair 2 counts "test cases / scenarios / classes / suites / files" and abstains only when the
count disagrees (an accusation-removing rule that keeps every matching count verified); repair 4
reads a second prefix only after "and", and only when it is path-shaped; and the census's
`looks_like_path` counts a trailing slash, as the instrument does. G-B1 to G-B6 are re-frozen
unchanged in wording, with G-B3 now consistent with repair 2.
