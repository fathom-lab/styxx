# NOTE — PATH-2, sixth pass: one parse, one reading of a name, and the bar a merge is held to

2026-09-25. Branch `fix/diffgate-path-resolution` (pull request #161). `origin/main` moved to `2a6ce0a3`
while the fifth round was reviewed (styxx 7.48.0 was released from it, #162 and #164 merged); it was
MERGED into this branch (`08df3f2f`, a merge commit, not a rebase: the earlier notes cite this branch's
SHAs). No file conflicted, and main's `styxx/diffgate.py` at `2a6ce0a3` is byte-identical to the one at
`98a5c368` (sha256 `9b620e00…`, LF), so every "main" below is the instrument 7.48.0 ships and the scorer's
baseline, unchanged.

This note sits on top of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third,
fourth and fifth-pass notes. **None of them is edited.** Where one is wrong, the correction is here
(section D). Where this note and an earlier document disagree about what the instrument does, this note
is what the code does. It is committed alone, before any code of this round; its measurements were taken
on the working tree whose instrument, port, pinned pairs, tests and scorer this round's commits carry.

---

## The merge bar

The orchestrator holds this pull request to three conditions, and this round was worked to them:

1. The reproductions of #97, #121 and #101 are fixed.
2. The branch introduces **no new wrong verdict relative to `origin/main` on any door** — Python
   `gate_diff_text`, Python `gate_diff`, and the JavaScript port. A new false VERIFIED, a new false
   CONTRADICTED, or a new Python/JavaScript disagreement is a blocker.
3. The scorer (`path2_gates.py`) cannot admit an unrelated defect.

A defect that exists identically on `origin/main` is not a blocker: it is recorded (section F) and becomes
a follow-up issue.

Measured against (2) this round, on every door, with ground truth from CPython's own parser (section E):
a 467-cell definition-line grid and a 103-cell name grid read **0 new wrong verdicts** against `main` and
**0 Python/port disagreements**; the fifth-pass head read 149 and 46 wrong verdicts on the same cells (98
on the port for names), `main` 375 and 25. The grids are why two findings of this round are in it at all:
a U+FEFF read mid-file (R-1, round 3) was a new false VERIFIED against `main` on the Python doors, which no
earlier round counted (D.4, W-1).

---

## A. What `origin/main` does on the round-5 parse inputs, measured before any change

A removed line whose text opens with `-- ` (a SQL, Lua or Haskell comment) prints as `--- x`; an added line
opening with `++ ` prints as `+++ x`. Each input was built as a real two-commit repository; the git door is
`gate_diff` on it, the raw door `gate_diff_text` on the bytes git printed, the port the same bytes through
`web/gate/diffgate.js`. V = VERIFIED, C = CONTRADICTED, U = UNCHECKABLE, one letter per claim.

| input (claims) | `main` git / raw / port | fifth-pass head git / raw / port | this round, every door |
|---|---|---|---|
| removed `-- users`, then `def foo():` → `def foo(x):` ("Added function foo. 1 file changed. Only touches src/.") | VVV / VVV / VVV | VVV / VVV / VVV | **U**VV |
| the same edit without the SQL line (control) | VVV / VVV / VVV | UVV / UVV / UVV | UVV |
| removed `-- users`, then `def test_a():` → `def test_a(tmp_path):` ("Added 1 test. Added 0 tests.") | VC / VC / VC | VC / VC / VC | **UV** |
| added `++ x` in `src/a.txt` ("Only touches src/. 1 file changed.") | VV / **CC** / **CC** | VV / CC / CC | VV |
| added `++x` (no space) before an unchanged `def foo` ("Only touches src/. 1 file changed. Added function foo.") | VVC on every door | VVC | VVC |
| one hunk with both: `-- users` removed, `++ x` added, `def foo` changed | VVV / VCC / VCC | UVV / UCC / UCC | UVV |
| removed `-- users`, then a removed `def public_api` ("No breaking changes.") | "no public top-level definition removed" on every door | the same | "removes 1 public definition … src/m.py: public_api" |

On `main`, the sides parser forgot its file at `--- users` and never saw the changed `def`, so there was no
pairing to find (`main` has none anyway, hence its VERIFIED on the control); the fifth pass inherited the
blindness and read a changed definition as added — the #101 kind, a false VERIFIED. `+++ x` made a phantom
file `x` on the raw door and the port, false CONTRADICTEDs of true claims the git door (which reads the
status from `--name-status`) never made. `+++x` is neither a header nor, to `main`, an added line: it was
dropped, and nothing that reads a definition can be one, so no claim moves.

---

## B. The two structural changes, and why

Round 5 found both by showing two readings that must agree and did not. Rounds 2 to 5 had each repaired
one character in one of two readings; this round makes each pair ONE reading, in both ports.

### W-1 — one hunk-aware parse

**Evidence.** Round-5 correctness lens, major 2: `got` and `hit` read the added blob of `parse_unified_diff`
(and, on the git door, a filter of its own), while both pairings read `parse_unified_diff_sides`. The two
parsers kept different lines, and neither counted a hunk (section A). The fifth pass's claim that `hit`
"reads exactly the lines the pairing reads, by construction" was true per line and false per diff.

**Change.** `_read_diff` is the one reading; `parse_unified_diff` (status map and added blob) and
`parse_unified_diff_sides` (per-file sides) are both built from it, and `gate_diff`'s git path takes its
added blob and its sides from it (its status stays git's own `--name-status`). A `---`/`+++` line is a
header only OUTSIDE a hunk; inside one, the `@@ -a,b +c,d @@` counts (ASCII digits, `[0-9]`, in both
ports: Python's `\d` is Unicode) say how many removed and added lines are still owed. A line the counts do
not allow for (a `diff --git` header, a hunk that ends early) closes the hunk and is read as before;
outside any counted hunk every line is read exactly as `main` read it, so a hand-written diff with no `@@`
reads as it did. A `\ No newline at end of file` line inside a hunk is skipped, an empty line is a context
line whose leading space was stripped.

**The exception, and why it exists.** Hand-written diffs often declare more lines than they carry; the
next file's `---`/`+++` pair then falls inside the counts. Read strictly, six pinned pairs of earlier rounds
(`bc2:two-path-shaped-prefixes`, `crlf:demo-diff`, `compat2:surface-and-scaffold` among them) read worse
than on `main` — the next file's header became a removed and an added line — and the fuzzer writes every
modified file's hunk with the same over-declared count (`@@ -1,2 +1,3 @@` over one context and one added
line). So inside a hunk's counts, a `---` line followed by a `+++` line is
still a file header, as `main` reads it, when a hunk header follows the pair or the pair is written
`--- a/…` or `/dev/null` and `+++ b/…` or `/dev/null` (`_header_pair`). Its cost is stated as a limit: a
removed `-- a` and an added `++ b` that are the last two lines of a hunk right before the next `@@` read as a
file header, exactly as on `main` (pinned as `path2:w1-limit-two-dashes-and-two-pluses-before-a-hunk-header-read-as-a-file-header`).

**Line 1, and R-1 superseded.** The parse knows each line's number, so it knows the one place CPython reads
a U+FEFF: byte 0 of a file. `_read_diff` drops a U+FEFF that opens line 1 of either side and
nowhere else, and the definition patterns no longer take one (`_DEF_INDENT` is `^[ \t\f]*`). R-1 (round 3)
had `got` and the pairings take an optional U+FEFF on ANY line; the fifth pass kept that as a limit ("a
definition led by U+FEFF is read wherever it stands"). This round's grid shows what that limit was: on the
Python doors, "Added 1 test." and "Adds function foo." over a U+FEFF-led definition in the middle of a file
read VERIFIED where `main` (whose `\s` never matched U+FEFF) read CONTRADICTED — a new false VERIFIED against
`main`, forbidden by the bar. It is closed; the line-1 case R-1 was written for still pairs.
`path2:v1-limit-a-bom-led-def-reads-as-a-definition-wherever-it-stands` is re-pinned VERIFIED → CONTRADICTED
and says so in its record.

**Pinned:** `path2:w1-a-removed-sql-comment-does-not-hide-a-changed-def`, `…-changed-test`,
`…-an-added-line-opening-with-plus-plus-space-is-not-a-file`, `…-a-hunk-with-both-shapes`,
`…-a-removed-sql-comment-does-not-hide-a-removed-public-def`, `…-an-added-line-opening-with-two-pluses-is-content`
(`++foo()` is a reference to `foo`, which `main` dropped), `…-a-bom-opening-line-one-of-a-new-file-is-a-bom`,
`…-a-hunk-that-declares-too-many-lines-still-ends-at-the-next-header`,
`…-a-short-hunk-before-an-a-b-header-pair-with-no-hunk-header`, `…-an-empty-line-inside-a-hunk-is-a-context-line`,
`…-a-no-newline-marker-inside-a-hunk-does-not-end-it`, and the limit above. Tests `test_w1_*` run the six
round-5 shapes through the git door, the raw door and the port.

### W-2 — one reading of a name

**Evidence.** Round-5 correctness lens, major 1: V-1 ended a definition's name at an ASCII stop set
(`_NAME_END`), while the claimed name came from the claim template's `\w` — Unicode in Python, ASCII in
JavaScript. The two ended in different places: "Added function café." read CONTRADICTED on the port
("does NOT define 'caf'"), and "Added function col·leccio.", an NFD "café", "foo‿bar" and a Devanagari name
with a virama read CONTRADICTED on the Python too — true claims, all VERIFIED on `main`. The reviewer's
56-cell grid: 0 Python/port verdict disagreements on `main` and the round-4 head, 48 at the fifth-pass head.
Correctness lens, minor 3, and protocol lens, major 1: the test side still read one literal space after
`def` and ran a test name to `[^ \t(:]*`, so a tab or form-feed separator went uncounted and a form feed or
a PEP 695 `[` after the name became part of it — a changed test read as added, a #101-kind false VERIFIED
on both ports; ERRATUM item 2 ("a changed generic test pairs") was false for a test made generic.

**Change, in both ports.** A name — claimed or defined — is the Python identifier that starts there,
read by the language's own rule: `str.isidentifier` on the growing prefix in Python (`_identifier_at`: the
opening character XID_Start or `_`, each later one XID_Continue), `\p{XID_Start}`/`\p{XID_Continue}` code point
by code point in JavaScript (`_identifierAt`). The claimed name is the identifier starting where the
template's `name` group starts (`_claimed_name`); the template still decides whether there is a claim, and
the claim's `detail` still holds the template's capture. Every definition line — for tests and symbols,
added and removed, `got`, `hit` and both pairings — is `_DEF_INDENT`, `def` or `class` (the removed side
alone also `async`), `_DEF_SEP`, that identifier, and then an ASCII character or the end of the line. A
non-ASCII character right after the identifier is one CPython refuses there (U+00A0, U+3000, U+2028,
U+FEFF, U+00B2): the line defines nothing, as V-1 read it. `got` IS the added-side pairing's reading,
counted line by line (`_added_tests`), not a second regex.

**The port's claim detail.** The port stored its own template capture in `detail.name` (`caf`) and asked
the symbol-word test (BC-2 repair 3) about it. It now reads the template's `name` group with Python's `\w`
(`_pyTemplateName`, `[A-Za-z_]` then `[\p{L}\p{N}_]*`), so the whole record agrees, not only the verdict
and the reason.

**Pinned:** eight claimed-name pairs (`path2:w2-a-claimed-name-with-an-accented-letter-is-read-whole`,
`…-a-middle-dot-…`, `…-a-combining-mark-…`, `…-connector-punctuation-…`, `…-cjk-letters-…`,
`…-a-claimed-class-with-a-sharp-s-…`, `…-a-virama-…`, `…-before-a-right-quote-ends-there`), the test shapes
(`…-a-test-separated-from-def-by-a-tab-is-counted`, `…-two-spaces-after-def-…`, `…-separator-changed-pairs`,
`…-a-test-name-followed-by-a-form-feed-pairs`, `…-a-test-made-generic-pairs`,
`…-a-test-name-followed-by-a-no-break-space-defines-nothing`, `…-a-class-named-like-a-test-is-not-a-test`)
and `…-a-name-followed-by-a-no-break-space-defines-nothing`. Tests: the 64-cell name grid through both
ports, detail included (`test_w2_a_claimed_non_ascii_name_reads_as_its_definition_on_both_ports`); eleven
test shapes through both ports; the twelve name-end cells of the fifth pass's grid, committed at last,
through both doors and the port; `test_v1_every_definition_pattern_opens_with_one_indent` now asserts the
`got`-equals-pairing equality in every position, the separator and the name end included.

### V-4, completed

Round-5 correctness lens, minor 4, its bare-`..` half: a bare `..` as the second prefix is not path-shaped, so
BC-2's repair 4 dropped it and the leading prefix accused alone ("Only touches `src/` and `..`." read
CONTRADICTED; "… and `../`." abstained). The written second prefix is now read through `_parent_prefix`
before that test drops it — on its written form (backslashes turned into slashes), not on the key, so the
reading does not lean on #121's key — and a bare `..` is off-tree, as `../` is. `...` (an elision) is
still dropped, and a bare `..` standing alone is still "not a path", as on `main`. Pinned:
`path2:v4-a-bare-dotdot-second-prefix-is-off-tree`. The second half of that finding (an ON-tree prefix
spelled with `.`, `..` or `...` inside, `src/../docs`, accuses paths it holds) is identical on `main`: a
follow-up issue (section F).

---

## C. The scorer

- **Reverts for the new rules.** `W-1` reverts the parse to the fifth pass's two parsers (this file's own
  copy: no counts, a `---`/`+++` line a header wherever it stands, no line-1 U+FEFF dropped). `W-2` reverts
  the definition reading, `got` and the claimed name to their fifth-pass (V-1) forms. `V-1`'s revert now
  includes the claimed name (the fourth pass read the template's). Where V-1 and W-2 patch the same names,
  the OLDER rule's code wins (`PRECEDENCE`), so V-1 reverted means the fourth pass whether or not W-2 is. R-1's
  own code is gone; it stays in the table only to take the U+FEFF out of a `got` that F-3, V-1 or W-2
  reverted, so alone it reverts nothing and is never credited alone.
- **F-2 and W-1 admit only where they can act.** Round-5 protocol lens, minor 2: F-2's admission was
  unconditional, so a defect planted inside F-2's own code (a tab read as a line break) was credited to F-2
  and admitted on a diff with no separator at all. `admits("F-2", …)` now requires `split_differs(diff)`;
  `admits("W-1", …)` requires `parse_differs(diff)` — a line inside a counted hunk opening with `---` or
  `+++`, or a U+FEFF opening line 1, under git's split or `str.splitlines()` (a move F-2 and W-1 explain only
  jointly acts through a line F-2's revert would cut). G-C6 admits a `compat2_candidate` flip only when F-2
  alone or W-1 alone explains it; G-C2 an eligibility move only when reverting #121 (key moved), F-2 (splits
  differ) or W-1 (parses differ) gives the baseline's eligibility back.
- **The scorer's own readings follow the instrument's.** `raw_paths` walks the diff by its hunk counts
  (`hunk_walk`, `header_pair`, this file's own), so a removed `-- a/x` is not a path; the #101 table tests
  read definitions by this file's own identifier rule (`ident_at`, `defined`), and the symbol test asks about
  the claim's identifier as well as the template's name (`claimed_identifier`).
- **G-C3's waiver, stated plainly.** G-C3 ("no accusation is added") is asked only through the amended
  table, i.e. only when #97, #121 or #101 is in a move's attribution. **It is skipped for a move attributed
  only to post-amendment rules — R-1, F-2, F-3, V-1, V-4, and now W-1 and W-2.** Such a new accusation is
  admitted by `admits` and counted, never refused by G-C3. On the differential this admits **10** claims at
  this round (F-2 `tests_added` 1, F-2+V-1 `tests_added` 1, F-3 `tests_added` 2, F-3+V-1+W-2 `tests_added` 3,
  V-1 `symbol_added` 3); the fifth pass's scorer admitted **6** the same way (V-1 `symbol_added` 2, F-3
  `tests_added` 2, F-2 1, F-2+V-1 1) and its note never said G-C3 was skipped. Each of the ten is a pinned pair
  whose accusation is true (a test really added beside a line `main` miscounted; a definition CPython
  refuses). The payload now prints the count twice, as `attribution.new_accusations_admitted` and as
  `G-C3_no_accusation_added.waived_for_post_amendment_rules`, and a corpus reader reads it before G-C3's pass.
- **Limit 12, restated** (round-5 protocol lens, minor 3). What the counterfactual cannot see: (a) a defect
  planted inside a rule's own code reverts with that rule, and only that rule's admission test and the
  pinned pairs can refuse it; (b) a defect in code no rule touches whose TRIGGER needs a rule's effect —
  when the text a claim reads is text F-2, F-3, V-1, W-1 or W-2 newly exposes, reverting that rule also
  removes the trigger, so the move is credited to the rule and admitted if the rule admits its kind. Pinned
  as an expected limit: `net >= n` planted for `net == n`, over a form-feed-led second test, is admitted as
  "F-2+V-1 tests_added: VERIFIED -> VERIFIED". A record without such text still exposes the defect.
- **Tests the review found missing** (protocol lens, minor 4 on the V-3 tests): the conjunction (a move a
  table rule and F-2 both explain, where the table refuses: it fails; crediting F-2 alone would have
  admitted it), G-C6's "alone" boundary (a flip F-2 and W-1 explain only jointly: it fails), G-C2's
  counterfactual (on a synthetic two-PR shelf: W-1's eligibility move attributed; a planted parse defect on a
  diff where W-1 cannot act refused), the G-C4 key-shape check, the precedence of the older rule, and every
  move on the pinned pairs admitted with the waived count equal to what the payload prints.

---

## D. Corrections to earlier notes

1. **Fifth pass, V-1: "`hit` reads exactly the lines the pairing reads, by construction", and the code
   comment "neither count can read a line its pairing cannot".** False per diff: `got` and `hit` read one
   parser's blob, the pairings another parser's sides (section A). W-1.
2. **Fifth pass, section D, and `web/gate/README.md`: the summary-side letter count "12 of 20, the same on
   main and at the fourth-pass head".** The count was the same; the verdicts were not. At the fifth-pass
   head those disagreements were no longer reasons but false accusations: V-1's ASCII name end made true
   claims over non-ASCII names read CONTRADICTED on the port and, for combining marks, U+00B7, connector
   punctuation and a virama, on the Python too. On this round's name grid the fifth-pass head reads 91 false
   CONTRADICTEDs on the port and 35 on the Python, where `main` reads 9. W-2.
3. **Fifth pass, E.5: "a test whose `def` is separated from its name by anything but one space is counted
   by neither `got` nor the pairing (they agree)".** Incomplete: where the separator or the character after
   the name differs between a removed and an added line, the pairing missed and a changed test was counted
   as added (tab or form-feed separator changed, a form feed after the name, a test made or un-made generic).
   **E.3's "the generic half of limit 3 is closed by V-1" was true for symbols only**, and so, for a test
   made generic, **ERRATUM item 2 ("a changed generic test pairs with its removed line like any other") was
   false**. The erratum is frozen and is not edited; this is its correction. W-2 closes both.
4. **Fifth pass, V-1: "a definition led by U+FEFF is read wherever it stands" (a disclosed limit).** It was
   a new false VERIFIED against `main` on the Python doors, not a limit the bar allows (section B, W-1).
5. **Fifth pass, section B: "`test_v1_every_definition_pattern_opens_with_one_indent` asserts the equalities
   line by line for all 30 characters in every position".** For tests it asserted the leading positions only;
   the separator and the name end were not asserted. The test now asserts them.
6. **Fifth pass, V-3 and E.12: "the counterfactual cannot see a defect inside a rule's own code".**
   Incomplete; restated in section C.
7. **Fifth pass, "Protocol changes the operator is asked to accept".** It named V-3's changes to G-C6 and
   G-C2 and did not say that G-C3 is skipped for post-amendment rules. Section C and the list at the end.
8. **Fifth pass, section D: "8 scorer mutants … killed".** True of those eight; the tests did not guard the
   conjunction, G-C6's F-2-alone boundary, G-C2's counterfactual or the key-shape check (the review's S1, S4,
   S8 and S9 survived them). Section C, last item.
9. **Commit order.** At the fifth pass the tests commit (`fd62d552`) precedes the scorer commit it
   exercises (`7f6d9303`): the six `test_v3_*` tests fail at `fd62d552`. It is not reordered — rewriting
   commits would break the SHAs these notes cite. This round commits its tests after all the code they
   exercise.

---

## E. What was measured

All on this round's working tree (instrument sha256 `b837f7e4…`, LF); corpora generated and deleted
afterwards; every git repository built in a temporary directory and removed.

- **Definition-line grid** (467 cells). Fourteen shapes — a test or symbol definition new, re-indented,
  removed, with the character leading it, between `def` and the name (new, changed) and after the name (new,
  changed) — times Python's 29 `\s` characters and U+FEFF, and the name-end shapes also times twelve
  non-space characters (U+00E9, U+00B7, U+0301, U+4E00, `[`, `x`, U+203F, U+0660, U+00B2, U+094D, `!`,
  U+0001); cells whose before and after are the same text dropped. Each cell a real two-commit repository;
  the git door, the raw door on git's bytes, the port on git's bytes; truth from `ast.parse` of the file's
  bytes (leading positions by CPython's tokenizer rule: space, tab and form feed are indentation, U+FEFF only
  at the start of a file).

  | instrument | Python/port disagreements | git/raw disagreements | wrong verdicts: git / raw / port | new wrong vs `main` |
  |---|---|---|---|---|
  | `main` | 44 | 0 | 375 / 375 / 407 | — |
  | fifth-pass head `1d262b33` | 0 | 0 | 149 / 149 / 149 | 3 per Python door (a mid-file U+FEFF) |
  | this round | **0** | **0** | 36 / 36 / 36 | **0** on every door |

  The 36 are all verdicts `main` gives too: an ASCII character CPython refuses after a name (`\x01`, `!`, a
  lone `[`, U+000B, U+001C–U+001F) and the lone-CR and newline cells.
- **Name grid** (103 cells): the 64 claimed-name cells of section B, eight named cases in three summary
  spellings (plain, backquoted, followed by U+2019 's'), twelve test shapes, three symbol shapes; three doors,
  `ast` truth (names compared under NFKC, as Python compares identifiers). Python/port disagreements: `main`
  63, fifth-pass head 63, **this round 0**; wrong verdicts: `main` 25 on each door, fifth-pass head 46 on the
  Python doors and 98 on the port, **this round 0**; new wrong vs `main`: fifth-pass head 28 per Python door
  and 84 on the port, **this round 0**. On the name inputs with the whole record compared (detail included),
  63 / 63 / **0**.
- **Summary side, unchanged.** One whitespace character between the words of twelve claim shapes (360
  inputs; this round's shapes, not the fifth pass's): 66 disagreements on `main`, the fifth-pass head and this
  round alike — the templates read the description with JavaScript's `\s`.
- **Differential** (`build_corpus.py` 176, `fuzz_corpus.py` 3,000, 190 pinned pairs): 3,366 pairs, 7,140
  claims (698 verified, 1,580 contradicted, 4,862 uncheckable), **0 disagreements**, this round on both sides.
  The same corpus with `main` on both sides reads 33 disagreements, with the fifth-pass head on both sides 5.
  With `main`'s Python on one side and this round's port on the other: 205 records differ; this round's
  Python against `main`'s port: 219. `check_pairs.js`: 190 pinned pairs (136 of them PATH-2's, 29 added this
  round), 0 disagreements; the minified bookmarklet, loaded in Node with the browser stubbed, 190 and 0.
- **Moves.** Against `main`, 205 records move in the Python (3 real, 108 fuzzed, 94 pinned) and 219 in the
  port. Against the fifth-pass head, 20 in the Python and 24 in the port, every one a pinned pair of this
  round or the re-pinned one: **no generated corpus record moves**. The three real-corpus moves against
  `main` are two of #97's (a `Created integrations/git/README.md` claim now names that file, not the root
  README) and one of #121's (a reason prints `.github/…` with its dot); the 108 fuzzed are #97's resolutions
  and V-4's abstentions on `docs/.` followed by a sentence period (92 claims in 90 records, as the fifth pass
  counted them).
- **The scorer**, `path2_gates.py differential` on this tree: every attribution gate passes (G-C0 alone
  fails, because the tree is uncommitted; the clean-tree run is in `web/gate/README.md`). 205 records move;
  claims attributed to #97 27, #121 35, #101 24, F-2 16, F-3 4, V-1 7, V-4 94, W-1 14, and to sets
  #101+W-1 7, #101+V-1 4, #101+V-1+W-2 4, #101+F-2+V-1+W-2 2, F-2+V-1 4, F-2+W-1 2, F-3+V-1+W-2 6, #121+V-4 2
  (4 of them joint: no single revert gave the baseline back); new accusations 14 (`only_touches` 4 by the
  table's dotted-prefix exception, 10 by post-amendment rules with G-C3 waived, section C);
  `compat2_candidate` flips 8, each one rule alone (F-2 six False → True, W-1 one each way); F-4 withdrawals 4;
  30 records whose splits differ, 21 whose parses differ.
- **Mutation, Python.** Only `styxx/diffgate.py` was copied (to scratch `mut6/`, LF) and loaded as
  `styxx.diffgate` by a pytest plugin; the test set was the fourth pass's ten modules, the three pin tests
  deselected (they hash the checkout's file, not the module). **27 mutants, 27 killed**: W-1's counts never
  set, the header-pair exception removed, each of its two clauses removed, the line-1 U+FEFF kept (added side,
  removed side) or dropped from any line, an empty line not context, a `\` line closing the hunk, the git
  door's blob back to its filter, an omitted count read as 0, removed lines not filed, line numbers off by
  one; W-2's identifier ASCII only, the claimed name back to the template, a non-ASCII character after a name
  defining or every character refusing, the separator without form feed, the indent taking U+FEFF anywhere,
  `class test_x` read as a test, `got` back to its regex, `async` on the wrong side or read by the wrong count
  (four ways); V-4's bare
  `..` dropped again and the not-a-path test not told. Three (the empty line, the `\` line, `class test_x`)
  survived the initial run and are killed by the three pairs added for them. The copy was deleted.
- **Mutation, port.** 28 in-memory mutants of `web/gate/diffgate.js` (`Module._compile`, nothing written),
  each held to the 190 pinned pairs and to the name grid with its detail: **27 killed**; the survivor lets an
  identifier open with a digit, which no claim can reach (a claimed name opens with `[A-Za-z_]`, and a
  defined name opening with a digit neither equals it nor opens with `test_`), so it is equivalent.
- **Mutation, scorer.** 14 mutants of `path2_gates.py`, applied in memory by a pytest plugin that reads the
  real file and compiles the mutated text under its path (nothing written), through the `test_v3_*` tests:
  **14 killed** — S1, S4, S8, S9, the table not asked, F-2 and W-1 admitted without their preconditions,
  W-1's precondition on git's split only, `raw_paths` header-confused again, W-1's revert missing, W-2's
  revert forgetting the claimed name, the precedence reversed, the G-C3 count not printed, an unattributable
  move admitted.
- **Unicode versions.** Node 24 (Unicode 16.0) takes 4,924 code points as XID_Start and 5,059 as
  XID_Continue that Python 3.12 (Unicode 15.0) does not, every one assigned after 15.0; the other direction
  is empty.

---

## F. What is still not repaired

Carried: amendment limits 1, 2 and 4 and the docstring half of limit 3; COMPAT reading Python's `\s` in all
five languages (fifth pass E.6); the Unicode-version margin; a lone `\r` inside a line; `..docs`; F-4's
abstentions; V-4 reading `docs/.` plus a period as a parent; `path2_differential_gates.json` uncommitted
(amendment limit 7, owed to the RESULT); the corpus gates not run on the real shelf; #128's modes 2, 3, 5
and 6. An added `async def` is read by neither template (as on `main`).

New or newly stated this round:

- **W-1's header-pair exception** reads `-- a` and `++ b` as a file header when they are the last two lines
  of a hunk right before the next `@@`, or when they are written `-- a/…` and `++ b/…` — as `main` reads them.
- **Identifiers are compared as text**, not under NFKC: a claim spelled in a different normal form from the
  code reads CONTRADICTED, as on `main`.
- **A line CPython refuses still reads as a definition** when an ASCII non-name character follows the name
  (`def foo!():`, `def foo\x01():`), as on `main`.
- **The claim templates still read the description with JavaScript's classes** in the port (66 of 360 on
  this round's whitespace shapes), and a declared `adds_symbol` with a non-ASCII identifier is MALFORMED in
  the port and read by the Python — both as on `main`.
- **An on-tree prefix spelled with `.`, `..` or `...` segments** (`src/../docs`) accuses paths it holds, as
  on `main`.
- **A `+++ /dev/null` line with no `---` line before it** raises in both ports, as `parse_unified_diff` did
  on `main`; since W-1, `parse_unified_diff_sides` is the same reading and raises there too (no verdict
  moves: `gate_diff_text` and the port raised on it already).

The defects above that exist identically on `origin/main` are the follow-up issues this round files.

---

## Protocol changes the operator is asked to accept

- **W-1** changes which lines of a diff are headers (a `---`/`+++` line inside a hunk's counts is content,
  with the header-pair exception) and where a U+FEFF is read (line 1 only), superseding R-1's reading.
  Its revert is in the scorer's table; F-2's admission now requires the splits to differ and W-1's the
  parses; G-C6 and G-C2 admit a flip or an eligibility move one of them alone explains, counted.
- **W-2** replaces V-1's name end (`_NAME_END`) and the test side's `def (test_[^ \t(:]*)` with the Python
  identifier plus the ASCII-or-end rule, for tests and symbols alike — the frozen C-1 patterns' text changes
  again. One pinned pair is re-pinned (W-1's line-1 rule).
- **G-C3 is waived** for every move attributed only to R-1, F-2, F-3, V-1, V-4, W-1 or W-2: 6 claims on the
  fifth pass's differential, 10 on this round's. The count is printed; the gate is not asked.
- **V-4** reads a bare `..` second prefix as off-tree.
