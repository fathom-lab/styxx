# RESULT — BC-2 passes its gates: 569 accusations removed, none added, the verified side intact

Fathom Lab · 2026-09-16 · Prereg: `PREREG_bc2_by_construction_2026_09_16.md`, pushed at
`9eaf25e8` on `fathomlab-patch-14` before this run, after `RESULT_bc1_INVALID_2026_09_16.md`.
Receipts: `external3_harness.py` (both runs), `external3_gates.py`,
`external3_base_gate_summary.json` (baseline: the checkout at the prereg commit, `diffgate.py`
`4e6f6550…`, no repair), `external3_gate_summary.json` (the repaired checkout, `4bf19b1b…`),
`external3_gates.json` (the gates), `external3_summary.json` (the census on the repaired ledger).
Ledgers gitignored. Counts only; no PR named. Corpus and reconstruction: EXTERNAL-1's.

## The gates

- **G-B1 (subset invariant) — PASS.** Keyed on `(pr_id, kind, claim text)` over 71,016 eligible
  PRs: 664 accusation keys in the baseline, 95 after, **0 new**. 569 removed: `only_touches`
  327, `tests_added` 177, `symbol_added` 65.
- **G-B2 (by construction) — PASS.** On the repaired ledger, with the census rule aligned to the
  instrument on a trailing slash: `tests_added` and `symbol_added` accusations in diffs without
  Python, 0 and 0; `only_touches` accusations with a prefix that is not a path, 0.
- **G-B3 (the verified side untouched) — PASS.** All 18 baseline VERIFIED claims of the three
  kinds are VERIFIED after; 0 lost; 0 VERIFIED `tests_added` or `symbol_added` in a diff without
  Python. (The repaired instrument also verifies four more: one `symbol_added` reached through
  "a new method …", three `only_touches` read through a second prefix. New verifications were
  never constrained; they are listed.)
- **G-B4 (demo and suite) — PASS.** `--demo` names the same three lies. The suite is green with
  `tests/test_diffgate_bc1.py` (32 tests pass, 2 `xfail(strict=True)`: the TypeScript lie and the
  counted-cases lie).
- **G-B5 (survivors, listed, not scored).** 96 CONTRADICTED claims remain (95 keys):
  `files_changed_count` 75 (untouched by design; 31 are git stat lines), `only_touches` 14
  (12 with one path-shaped prefix, 2 with two), `tests_added` 7 (every one a Python diff where
  both counts are positive and differ; the counted noun is "functions" in 3 and bare in 4).
  No precision is attached to any of them. Whoever wants these trusted at scale runs a blind
  adjudication under the EXTERNAL-1 protocol.
- **G-B6 — PASS on the Python side.** `web/gate/differential/bc1_pairs.json` carries the four
  pairs with the repaired verdicts; the JavaScript port is pinned to the wheel and disagrees on
  all four until it is updated. That is drift and is reported as such.

## What changed in the claim census, and why

Baseline (main at the prereg commit): 12,957 covered PRs; 13,013 VERIFIED, 8,940 UNCHECKABLE,
665 CONTRADICTED; 625 PRs with a contradiction. Repaired: 12,933 covered; 13,017 VERIFIED, 9,477
UNCHECKABLE, 96 CONTRADICTED; 91 PRs with a contradiction. 36 PRs left coverage — every one had
only `symbol_added` matches on a word ("adds a method **to** …"), sentences that are never-read
now and counted as such — and 12 entered it through the "a new method …" reading and the second
prefix. 534 PRs stop carrying an accusation.

Against the 7.47.0 wheel the numbers differ further (15,762 covered PRs), because main reads
fewer `file_touched` claims than the wheel since V14. That difference predates this cycle and is
outside it; the baseline here is main, so that the only variable is the repair.

## What the instrument does now that it did not this morning

A commit message that says "Added 2 tests" in a TypeScript repository passes the hooks with
`[ ? ] tests_added  no Python file in the diff; this template counts \`def\` lines (#110)`
instead of `[LIE]`. "Only modifies the footer" is `[ ? ] prefix 'the' is not a path (#110)`.
"Adds a method to reload data" is a sentence the gate never read, and the never-read count says
so. "Added 3 test cases" is VERIFIED when three `def test_` were added and an abstention naming
both numbers when they were not. "Only touches `src/` and `tests/`" reads both prefixes.

The doors shipped this week inherit this the moment it lands; each of their PRs carries the
comment that pointed here, and the sentence "Python-only for test and symbol counts, wrong about
scope sentences in any language" stops being true for the scope sentences and becomes "Python-only
for test and symbol counts, said out loud in the verdict". Counting tests in other languages is
the next preregistration, not this one.
