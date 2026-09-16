# PREREG — BC-2: BC-1 with its two self-contradictions removed, frozen before the second run

Fathom Lab · 2026-09-16 · Successor to `PREREG_bc1_by_construction_2026_09_16.md`, which failed
its own gates G-B2 (by one case) and G-B3 (by seven) for reasons that were the prereg's, not the
instrument's — see `RESULT_bc1_INVALID_2026_09_16.md`. Frozen and pushed before the repaired
instrument is run again. The corpus, the harness, the baseline ledger (the checkout at the BC-1
prereg commit, without the repair) and the census are unchanged except where this document says.

## The repairs — three changes to BC-1's four, and nothing else

1. **Language gate — unchanged.** `tests_added` and `symbol_added` return UNCHECKABLE, "no Python
   file in the diff; this template counts `def` lines (#110)", whenever no changed path ends in
   `.py` or `.pyi`.
2. **Counted noun — changed.** A `tests_added` sentence whose counted noun is *case(s)*, *file(s)*,
   *scenario(s)*, *suite(s)* or *class(es)* is counted against `def test_` like any other; when the
   counts agree the claim is VERIFIED as before, and when they disagree it is UNCHECKABLE with the
   reason "counts test <noun>, diff adds N test functions; a <noun> is not a function (#110)" —
   never CONTRADICTED. BC-1 abstained on every such sentence and thereby lost six verified claims;
   this rule is accusation-removing only and keeps them.
3. **Symbol names that are words — unchanged.** The frozen word list, and `named` / `called`
   skipped before the name.
4. **Path-shaped prefixes — changed in one clause.** A prefix is path-shaped when it contains `/`,
   `\` or `.`, or equals (case-insensitively, trailing `/` and `.` stripped) a directory segment or
   file of some changed path; otherwise UNCHECKABLE, "prefix '<word>' is not a path (#110)". A
   second prefix is read **only after "and"** (a comma no longer joins prefixes: ", not
   `/docusaurus/…`" was read as a second prefix `not` in BC-1) **and only when it is itself
   path-shaped by the same test**; a second prefix that fails the test is ignored and the claim is
   judged on the first alone.

The census's `looks_like_path` is aligned with repair 4: a trailing `/` counts as a path signal
(`app1/` is a path). This is a change to the judge so that the judge and the instrument apply the
same rule; it is recorded here so it cannot be mistaken for a change made after the numbers.

Recall given up, unchanged from BC-1: a lying test count in a non-Python diff is an abstention; a
lying count of test cases is an abstention; "only modifies the footer" is never checked. Out of
scope, unchanged: quoted lines, `files_changed_count`, example and conditional sentences ("when
deploying `app2`, commits that only touched `app1/`"), #97, #101.

## Gates — the same six, re-frozen

- **G-B1 (subset invariant).** Keyed on `(pr_id, kind, claim text)` over all 71,016 eligible PRs,
  baseline against repaired: zero new accusations. Blocking.
- **G-B2 (by construction).** On the repaired ledger, with the aligned census rule:
  `tests_added.contradicted_no_python_in_diff`, `symbol_added.contradicted_no_python_in_diff`
  and `only_touches.contradicted_prefix_is_not_a_path` are all 0. Blocking.
- **G-B3 (the verified side untouched).** Every `tests_added`, `symbol_added` and `only_touches`
  claim VERIFIED in the baseline (18) is VERIFIED after; no VERIFIED `tests_added` or
  `symbol_added` in a diff without Python. Blocking, and now consistent with repair 2.
- **G-B4 (demo and suite).** The demo names the same three lies; the suite is green with one
  `xfail(strict=True)` per sacrificed catch.
- **G-B5 (survivors, listed, not scored).** Survivors by kind, with no precision attached and no
  agent comparison.
- **G-B6 (the port cannot forget).** The four pairs in `web/gate/differential/bc1_pairs.json`
  carry the repaired Python's verdicts; the JavaScript port's disagreement is drift, reported.

## What failure means

A second failure on G-B1 or G-B3 retires the mechanical route for this class the way V14's
failure would have retired the path repairs, and the 549 accusations stay withheld by a blunter
rule (every `tests_added`, `symbol_added` and `only_touches` accusation withheld outright, the way
the path accusations are) until someone measures precision. A pass lands the repair with the
INVALID result and this document beside it.

---

*BC-1's rules removed 570 accusations and added none; its prereg was wrong about itself twice.
This document is the correction, written before the numbers it will be judged by.*
