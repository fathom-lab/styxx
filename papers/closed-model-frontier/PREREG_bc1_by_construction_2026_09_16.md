# PREREG — BC-1: the accusations that cannot be right by construction stop being made

Fathom Lab · 2026-09-16 · Frozen before implementation and pushed before the repaired
instrument is run over the corpus. Motivated by issue #110 and its receipts
(`RESULT_external2_live_accusations_2026_09_16.md`, `external2_summary.json`): of the 665
accusations the 7.47.0 wheel still makes on the EXTERNAL-1 corpus after the path accusations
were withheld, 549 are unsupported by construction — `tests_added` and `symbol_added` counting
`def` lines in diffs with no Python file (168 + 59), and `only_touches` reading an English word
as a path prefix (322). Nothing here is a precision measurement. The repair removes accusations
the instrument could not have supported; whether the survivors are right is a separate question
this prereg does not answer and does not claim to.

## The four repairs, accusation-removing only

1. **Language gate.** `tests_added` and `symbol_added` return UNCHECKABLE, with the reason
   "no Python file in the diff; this template counts `def` lines (#110)", whenever no changed
   path in the diff ends in `.py` or `.pyi`. The count is not computed. In a diff that touches
   Python the templates behave exactly as before.
2. **Counted noun.** A `tests_added` sentence whose counted noun is a test *case*, *file*,
   *scenario*, *suite* or *class* (singular or plural) returns UNCHECKABLE with the reason "counts
   test <noun>; this template counts `def test_` functions". *functions*, *methods* and the bare
   noun (*tests*) are counted as before.
3. **Symbol names that are words.** A `symbol_added` match whose captured name is in a frozen
   list of function words and generic nouns (`to`, `with`, `that`, `for`, `in`, `on`, `of`, `by`,
   `and`, `or`, `as`, `the`, `a`, `an`, `this`, `which`, `it`, `its`, `is`, `declaration`,
   `implementation`, `definition`, `signature`, `body`, `stub`, `call`, `wrapper`, `override`,
   `overload`, `level`, `support`, `named`, `called`) is not a claim: the sentence is never-read
   and counted as such. The template additionally skips `named` / `called` before the name, so
   "added a function named `foo`" reads `foo`.
4. **Path-shaped prefixes, and two of them.** An `only_touches` prefix is path-shaped when it
   contains `/`, `\` or `.`, or equals (case-insensitively, trailing `/` and `.` stripped) a
   directory segment or file of some changed path. A prefix that is not path-shaped returns
   UNCHECKABLE with the reason "prefix '<word>' is not a path (#110)". A second prefix joined by
   "and" or a comma is read; the claim is VERIFIED when every changed path is under either
   prefix and CONTRADICTED, naming the paths outside both, otherwise.

**Recall is given up on purpose and said so.** A lying "added 3 tests" in a TypeScript
repository becomes an abstention instead of a catch; "added 3 test cases" is never counted;
"only modifies the footer" is never checked. This instrument was measured accusing wrongly in
82.6% of what remained, and a false accusation costs more than a missed catch. Nothing in this
cycle adds an accusation anywhere.

Out of scope, named so it is not mistaken for forgotten: quoted lines (`>`) still count as the
author's text (the gate reads inside quotes and HTML comments today, for every kind); the
`files_changed_count` template is untouched; #97 (basename shadowing in `find_path`) and #101
(a changed `def` counts as added) stay open under their own numbers.

## Gates — committed now

- **G-B1 (subset invariant).** Keyed on `(pr_id, kind, claim text)` over all 71,016 eligible
  PRs: the post-repair accusation set is a subset of the pre-repair set. One new accusation
  anywhere fails the cycle. Blocking.
- **G-B2 (by construction).** Re-run `external2_census.py` on the repaired ledger: the three
  counters `tests_added.contradicted_no_python_in_diff`,
  `symbol_added.contradicted_no_python_in_diff` and
  `only_touches.contradicted_prefix_is_not_a_path` are all 0. Blocking.
- **G-B3 (the verified side is untouched).** The 13 `tests_added`, 1 `symbol_added` and 4
  `only_touches` VERIFIED claims of the pre-repair ledger are VERIFIED after it, keyed the same
  way, and no VERIFIED `tests_added` or `symbol_added` appears in a diff without Python.
  Blocking.
- **G-B4 (the demo and the suite).** `python -m styxx.diffgate --demo` still names the same
  three contradictions (`symbol_added` backoff, `tests_added` 1 vs 3, `only_touches` src:
  the demo diff touches Python and `src` is a segment of a changed path). The full suite is
  green, with tests that pin each of the four repairs on minimal diffs and one
  `xfail(strict=True)` per sacrificed catch (the TypeScript lie, the counted-cases lie), so
  re-enabling either later cannot happen silently.
- **G-B5 (the survivors, listed, not scored).** Every CONTRADICTED that survives is counted by
  kind in `external3_summary.json`; the number is reported with no precision attached and no
  agent comparison. If anyone wants these accusations trusted at scale, the next step is a
  blind adjudication under the EXTERNAL-1 protocol, not this document.
- **G-B6 (the port cannot forget).** Four pairs — a TypeScript "added N tests", a word-named
  symbol, a `the` prefix, a two-prefix scope sentence — are added to `web/gate/differential/`
  with the repaired Python's verdicts as the expectation. The JavaScript port is pinned to the
  wheel and will disagree on them until it is updated; that disagreement is drift, reported
  the way #100 reports V13/V14 drift, never silenced.

## What failure means

If G-B1 or G-B3 fails, the repair does not land: an accusation-removing change that adds an
accusation or loses a verified claim is a different change than the one preregistered here. If
G-B2 fails, the rule as written does not do what the census counted, and the census and the
rule are reconciled in a correction before anything ships. A pass does not re-enable, promote
or relabel anything; it removes 549 accusations the instrument had no grounds for and says
what is left.

---

*The 549 were counted before this was written. The rules above are the rules the census
already applied, moved into the instrument; the measurement that follows is whether moving
them changed exactly what the census said and nothing else.*
