# RESULT — EXTERNAL-2: what the gate still accuses, and why 549 of those 665 accusations cannot be right by construction

Fathom Lab · 2026-09-16 · Corpus and reconstruction: EXTERNAL-1's (`PREREG_external1_aidev_2026_08_31.md`,
`external1_harness.py`, `CORRECTION_external1_cause_2026_08_31.md`). Instrument: the 7.47.0 wheel,
`styxx/diffgate.py` sha256 `fb2d9b3e8426650bc20fc8613c9bcad16c1dcd86b85b0ca8c532fdd2a23e7304`.
Receipts: `external2_harness.py`, `external2_census.py`, `external2_gate_summary.json`,
`external2_summary.json`. Issue: #110. Not preregistered: this is a census of the instrument's
output, mechanical throughout, with no adjudication and no gate to pass or fail; the repair it
motivates gets its own preregistration.

## What ran

EXTERNAL-1 withheld the path accusations (`file_created`, `file_deleted`, `file_touched`) after
measuring their precision at 0.23 against a preregistered floor of 0.95. Four templates kept the
power to accuse: `tests_added`, `symbol_added`, `only_touches`, `files_changed_count`. Their
precision in the wild was never measured on its own: 5,345 of the 6,010 accusations in the
EXTERNAL-1 population were path claims, and its adjudication packet of 130 items (100 accusations
plus decoys) held 7 `only_touches`, 6 `tests_added`, 2 `files_changed_count` and no `symbol_added`.

The wheel was re-run over the EXTERNAL-1 corpus (AIDev; 71,677 PRs seen; 71,016 eligible after
the same three exclusions: 573 empty bodies, 87 without file records, 1 reconstruction mismatch).
The ledger reproduces the published one on every count that survives the withholding: 625 PRs
with a contradiction, 665 CONTRADICTED claims. The claim census differs from the 2026-08-31 run
(15,762 covered PRs against 15,965; 19,278 VERIFIED against 17,887; 9,469 UNCHECKABLE against
11,533) because the wheel carries V13 (`b7ee0a5e`, the `v7.47.0` tag: the wheel's file is the tag's
with CRLF line endings, sha256 `6dfd21c4…` once normalised) and the 2026-08-31 run preceded V13
(`584ff104`, the corrected run, before the V13 repairs); the accusation set is identical and is the
only thing this result is about.

## The census

665 accusations: `only_touches` 341, `tests_added` 184, `files_changed_count` 75, `symbol_added` 65.
Two mechanical questions, asked of each:

**Did the diff touch a Python file?** `tests_added` counts `^\s*def test_` in the added lines;
`symbol_added` looks for `^\s*(def|class) NAME` there. In a diff with no `.py`/`.pyi` file the count
is 0 before the diff is read. 168 of the 184 `tests_added` accusations and 59 of the 65
`symbol_added` accusations are in such diffs. The instrument's stated reason — "diff adds 0 test
functions, claim says 3" — is not evidence about a sentence in a TypeScript pull request.

**Is the captured prefix a path?** `only_touches` takes the token after "only modifies / touches /
changes" as a path prefix. 322 of its 341 accusations captured an English word: `the` 105, `with`
54, `that` 29, `files` 8, `a` 6, `how` 5, `is` 5, `when` 5. "Only modifies the footer" is
contradicted with `paths outside 'the'`.

**549 of 665 accusations (82.6%) are unsupported by construction.** Nothing about the sentences
was adjudicated; the instrument could not have found what it says it counted. The remaining 116
are not thereby correct. Among them: of the 16 `tests_added` accusations in Python diffs, 9 count
"test cases", "test scenarios" or "test files" and 3 say "test functions"; across all 184 the
counted noun is "test cases" 81 times, "test files" 9, "test scenarios" 3, "test methods" 5, "test
functions" 4. All 6 `symbol_added` accusations in Python diffs captured an English word as the
symbol ("adds a method **to** reload data" → `method 'to'`; "added class **declaration**"); 32 of
the 65 captured names are function words. Of the 19 path-shaped `only_touches` prefixes, most are
two-prefix sentences read as one, quoted instructions (`> - Only modify package.json and
package-lock.json`), basenames (#97), or a bare `-`. `files_changed_count` (75 CONTRADICTED, 64
VERIFIED) is listed for completeness and not characterised: 31 of its 75 are git's own stat lines
("1 file changed, 102 insertions(+)"), a commit's count against a PR-level reconstruction.

Per agent, the accusations fall as the corpus falls (Copilot 485, Devin 110, Claude Code 46,
OpenAI Codex 18, Cursor 6, Google Jules 0); no comparison between agents is drawn from an
instrument in this state, as in EXTERNAL-1.

## The verified side, for the record

13 `tests_added` VERIFIED, all in Python diffs; on none of them does an added `def test_` name also
appear in the removed lines, so the net rule proposed in #101 and the current rule agree on all 13.
1 `symbol_added` VERIFIED, not a redefinition. The four `only_touches` VERIFIED are path-shaped
and correct. #101 is real and stays open; on this corpus it fired 0 times in 14 chances.

## Consequence

Every door added this week — the Action (#94), `--pr` (#95), the git hook (#98), the Claude Code,
Codex and Gemini hooks (#99, #106, #108), the pre-commit hook (#102), the Cursor hook (#105), the
GitLab job (#109), the JavaScript port (#100) — carries these three branches unchanged. A real
repository, the wheel and main alike: a TypeScript commit `Added 2 tests for add()` is refused by
the hooks with `[LIE] tests_added diff adds 0 test functions, claim says 2`; "Only modifies the
test file." → `only_touches CONTRADICTED paths outside 'the'`; "Adds a method to reload data when
files change." → `symbol_added CONTRADICTED added lines do NOT define method 'to'`. The repro is in
#110 and each open PR carries a comment saying so. Until the repair lands, the hooks and the
Action are Python-only for test and symbol counts and wrong about "only modifies the …" sentences
in any language.

## What follows, under the protocol

Same shape as EXTERNAL-1: withhold what cannot be supported, preregister the repair, re-measure on
this corpus, land with `xfail(strict=True)` pins. The direction, to be frozen in the prereg before
the repaired instrument is run:

1. `tests_added` and `symbol_added` → UNCHECKABLE, reason "no Python file in the diff; this
   template counts `def` lines", whenever the diff touches no `.py`/`.pyi`; sentences whose counted
   noun is "test files", "test cases", "test scenarios" → UNCHECKABLE; a `symbol_added` name that
   is a function word is not a claim.
2. `only_touches` prefix must be path-shaped (contains `/`, or matches a directory or file of a
   changed path) or the claim is UNCHECKABLE with the prefix quoted; a sentence with two prefixes
   joined by "and" reads both; quoted lines (`>`) are not the author's claims.
3. Gates for the re-measurement: the three by-construction counters at 0; the 13 + 1 + 4 VERIFIED
   preserved exactly; every surviving CONTRADICTED listed by kind in the receipt; a TypeScript pair
   and a stop-word pair added to `web/gate/differential/` so the port cannot reintroduce either.

## Reproduction

```
python papers/closed-model-frontier/external2_harness.py shelf --corpus DIR   # the two AIDev parquet tables
python papers/closed-model-frontier/external2_harness.py gate                 # writes external2_ledger.jsonl (gitignored) + external2_gate_summary.json
python papers/closed-model-frontier/external2_census.py                       # writes external2_summary.json
```

The harness refuses to run unless `styxx.diffgate` resolves to the installed wheel rather than
the checkout, and records the wheel's file hash in the gate summary.
