# NOTE — PATH-2, fourth pass: the round-3 blocker, the line-splitting class, and corrections to the third pass

2026-09-25. Branch `fix/diffgate-path-resolution` (pull request #161), rebased onto `origin/main`
`98a5c368`, on top of `PREREG_path2_resolution_2026_09_17.md`,
`AMENDMENT_path2_resolution_2026_09_17.md`, `ERRATUM_path2_amendment_2026_09_17.md` and
`NOTE_path2_third_pass_2026_09_25.md`.

**None of those four documents is edited.** The preregistration, the amendment and the erratum are
public and frozen. The third-pass note was committed alone before that round's code, and that code
exists, so it is frozen too; where it is wrong, the correction is here (section A). Where this note and
an earlier document disagree about what the instrument does, this note is what the code does.

This note is committed alone, before any code of this round. The measurements in it were taken on the
working tree whose instrument, port, pinned pairs, tests and scorer this round's code commits carry.

The round-3 review filed one blocker, four majors and four minors across two lenses. Every one is
answered below: four rule changes (section B, F-1 to F-4), five changes that are not rules
(section C), and the limits that remain (section E).

---

## A. Corrections to the third-pass note

### A.1 Section B, issue #128's eleven: three reasons move, not two, and all eleven were measurable

The third-pass note said: "No verdict moves. Two reasons do." and that six of the eleven rows "quote a
claim without quoting the paths, so they are not reconstructible here." **Both statements are wrong.**
Two committed receipts in the same directory carry all eleven:

- `bench2_audit.json` — the eleven accusations, each with the instrument's verbatim reason at
  `4ba947a8` (`instrument_why_verbatim`) and the hand adjudication;
- `bench2_dataset.jsonl` — per claim: `claim_text`, `claim_detail` (the extracted prefixes), and
  `truth_facts` (`paths`, the changed-path count; `outside`, up to five outside keys in diff order;
  `n_outside`).

**How they were read this round.** For each audit item the dataset row with the same URL and claim
text was taken. The diff was rebuilt with one minimal header per path: the listed outside keys in
order, then stand-ins for the outside paths the dataset counts but does not list (placed after the
listed ones, so they never reach the three a reason prints; for `vscode-azuretools#2086` they are
`…/package.json`, which is what #128 says all twelve paths are), then stand-ins for the inside paths.
The dataset's keys are the pre-#121 keys, with leading dots stripped, so a key alone cannot say
whether the real path carried a dot. Where evidence says it did, the dotted spelling was used and the
undotted spelling was run as well:

| key in the dataset | real spelling | evidence |
|---|---|---|
| `gitignore` (#1142) | `.gitignore` | the round-3 reviewer's read-only query of the committed shelf (`f.filename` for `pr_id=3127853160`); not re-read this round |
| `github/workflows/dependabot.yml` (#415) | `.github/…` | #128's text: the PR "changes `.github/workflows/dependabot.yml`" |
| `github/workflows/…` × 3 (#442) | `.github/…` | GitHub Actions reads workflows from `.github/workflows/` only |

Nothing read the shelf or the network this round. Each rebuilt diff ran through `origin/main`
(`98a5c368`), the round-3 head (`9966b985`) and this round's tree.

| row | adjudication | `98a5c368` | this branch (round-3 head and this round agree on all eleven) |
|---|---|---|---|
| `Albeoris/Memoria#1142` | false | CONTRADICTED `paths outside 'mods/submods': ['gitignore', 'index.html']` | CONTRADICTED `… ['.gitignore', 'index.html']` — **reason moves** |
| `Albeoris/Memoria#1145` | false | CONTRADICTED `… ['index.html']` | identical |
| `Albeoris/Memoria#1147` | false | CONTRADICTED `… ['fixes_summary.md', 'index.html']` | identical |
| `Azure/autorest.typescript#3252` | correct | CONTRADICTED `paths outside 'packages/typespec-ts' and 'packages/typespec-test': ['common/config/rush/pnpm-lock.yaml']` | identical |
| `mikepenz/release-changelog-builder-action#1458` | false | CONTRADICTED `paths outside 'app1': ['readme.md', '__tests__/pathfiltering.test.ts', 'action.yml']` | identical |
| `dotnet/runtime#117821` | false | UNCHECKABLE `prefix 'assert.notnull' is not a path (#110)` | identical |
| `microsoft/vscode-azuretools#2086` | false | VERIFIED `all changed paths under prefix` | identical |
| `ydb-platform/ydb#25857` | false | VERIFIED `all changed paths under prefix` | identical |
| `open-policy-agent/cert-controller#415` | false | CONTRADICTED `paths outside 'githiub/workflows/dependabot.yml': ['github/workflows/dependabot.yml']` | CONTRADICTED `paths outside '.githiub/…': ['.github/…']` — **reason moves** |
| `fern-api/fern#9898` | false | CONTRADICTED `paths outside 'readme/documentation': [3 paths]` | identical |
| `microsoft/wassette#442` | correct | CONTRADICTED `paths outside 'changelog.md': ['github/workflows/…' × 3]` | CONTRADICTED, the same three with their dots — **reason moves** |

**No verdict moves. Three reasons do: #1142, #415 and #442.** Under the undotted spelling no reason
moves at all, so every move is the printed dot and nothing else. The `98a5c368` column reproduces the
audit's verbatim reason wherever PATH-1 did not change the row, which is the check that the rebuilt
diffs read as the real ones did. PATH-1's published after-column holds: eight accusations, two
correct, **precision 2 in 8 = 0.25**, unchanged by PATH-2 and by this round. This round's rule
changes move none of the eleven.

The third-pass note's fallback argument — "if a dotfile does sit among their changed paths, the effect
is the printed path, not the verdict" — was right, and #1142 is exactly that case. Its claim that the
six "name no dotfile among the paths that decided them" was wrong for #1142, whose `.gitignore` is one
of the two paths its reason prints.

### A.2 Section D.5: the BOM instance was closed; the class was not

The third-pass note said R-1 "closes the `tests_added` half" of amendment limit 5. R-1 closed the one
instance it named, a leading U+FEFF. The class — Python and the port reading one diff line
differently, or `got` and the pairing reading different lines — stayed open in two shapes, both found
by the round-3 reviewers:

- **Line splitting.** Python split the diff with `str.splitlines()`, which also breaks on U+000B,
  U+000C, U+001C–U+001E, U+0085, U+2028 and U+2029; the port splits on `\r\n`, `\r` and `\n`. On a
  re-indent `-def test_a():` / `+<U+000B>def test_a():` (and U+000C, U+2028, U+2029), "Added 1 test."
  read CONTRADICTED in Python and **VERIFIED in the port**, the public door; "Added 0 tests." over
  `+x = 1<U+2028>def test_new():` read VERIFIED in Python and CONTRADICTED in the port.
- **The indent `got` reads.** R-1 made every line the pairing reads a line `got` counts, but not the
  reverse: `got` read `\s*`, the pairing `[ \t]*`. An NBSP or U+3000 re-indent of a test was counted
  as added and paired with nothing, and "Added 1 test." over it read **VERIFIED on both ports** — a
  false VERIFIED of the #101 kind, the defect this series exists to remove.

Both are repaired this round (F-2, F-3). What is left of limit 5 is in section E.

### A.3 Smaller corrections

- **Section A's re-pin table** listed two strings in `path1:unrepaired-typo`'s neighbourhood and left
  out a third: the same pair's `file_touched` reason moved from `diff status 'M' for
  'github/workflows/dependabot.yml'` to `… '.github/workflows/dependabot.yml'`, verdict unchanged.
  All three are now recorded beside PATH-1's own documents, in
  `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md`.
- **Section A, step 2** said PATH-2 reads the changed paths' segments "on the undotted key, so
  `github` still names the `.github` directory". True of the changed paths; the prefix was compared
  with its dot, which is the blocker (F-1).
- **Section C, R-1** said "every line the added-side pairing pattern matches is now a line `got`
  counts". True, and a reader takes it for equality; it was a subset (F-3).
- **Section C, R-3** said "one off-tree prefix is enough" to abstain. Narrowed by F-4.
- **Section E** reported "0 disagreements" over 3,282 pairs without saying that a zero over a corpus
  says nothing about lines the corpus does not carry. It carried none of the F-2 shapes.
- **The round-3 handoff** said `build_bookmarklet.py --check` "matches all three", and
  `web/gate/README.md` said it compares the files on disk byte for byte. On a stock Windows checkout
  it exited 1 (B-1).

---

## B. Rule changes this round

Each is mirrored in `web/gate/diffgate.js`, pinned by pairs in `web/gate/differential/path2_pairs.json`
that both implementations are held to, tested in `tests/test_diffgate_path2.py`, and mutation-checked
(section D).

### F-1 — the prefix-shape test undots the prefix (BLOCKER)

**Reviewer evidence (reconciliation lens, blocker).** `_prefix_is_path_shaped` compared
`_norm(raw).rstrip("/").lower()` — which keeps a leading dot since #121 — against
`_undotted(changed).split("/")`, which drops it (`styxx/diffgate.py` ~317; the port ~135). So every
slashless dotted prefix whose last dot-segment is not in `PATH1_EXTENSIONS` stopped being a path:
"Only touches .github." over `.github/ci.yml` went from `98a5c368`'s VERIFIED to UNCHECKABLE "prefix
'.github' is not a path (#110)", and likewise `.vscode`, `.circleci`, `.husky`, `.gitignore`, `.npmrc`,
`.env.local`. A correct accusation was withdrawn ("Only touches .gitignore." over `.gitignore` and
`src/a.py`), and AMENDMENT C-3's own accusation class never fired for these spellings. The branch's own
scorer rejected the moves (`G-C4_direction:only_touches`, reasons ending "(#110)"); the corpus simply
held no instance, and the one committed test used `.eslintrc.json`, whose suffix is listed.

**Change.** `low = _undotted(_norm(raw)).rstrip("/").lower()` in both ports: prefix and segments are
compared undotted, the reading `98a5c368` had. Whether the changed paths lie under the prefix is still
decided with the dots, in `_gate`.

**Measured.** Against the round-3 head, F-1 alone (the one-line change applied to that file) moves 5
of 3,301 differential records, all five its own new pairs, and **none of the 3,282 records that predate
this round**. Those 3,282 hold 942 `only_touches` claims and only 3 slashless dotted prefixes —
`.k-step-link` (no changed path has that segment), `.env` and `.eslintrc.json` (listed suffixes, decided
before the segment test) — so the corpus could not have seen the defect, which is the reviewer's point.
It moves none of #128's eleven (none has a slashless dotted prefix). Against `98a5c368` the
restored VERIFIEDs and CONTRADICTEDs read as they did there, with the prefix printed with its dot; and
C-3's accusation returns: "Only touches .gitignore." over `gitignore` is CONTRADICTED
`paths outside '.gitignore': ['gitignore']`, which the scorer admits under its dotted-prefix exception.

**Pinned:** `path2:f1-a-dotted-directory-prefix-is-a-path`,
`path2:f1-a-dotfile-prefix-with-an-unlisted-suffix-verifies`,
`path2:f1-a-dotfile-prefix-with-two-dots-verifies`,
`path2:f1-a-dotfile-prefix-still-accuses-a-real-outside-path`,
`path2:f1-c3-accuses-a-dotfile-prefix-over-its-undotted-twin`. Tests
`test_f1_a_slashless_dotted_prefix_is_read_undotted_like_the_changed_paths` (eight spellings) and
`test_f1_a_dotted_prefix_verifies_accuses_and_carries_c3_in_all_three_positions`.

### F-2 — a diff splits into lines on `\r\n`, `\r` and `\n` only, in both doors and both ports

**Reviewer evidence (protocol lens, major; reconciliation lens, minor).** The line-splitting shape of
section A.2.
Pre-existing on `origin/main` — not a regression — but the port was the side giving the false VERIFIED,
and the committed corpus could not see it.

**Change.** A git diff separates lines with `\n`; a CRLF file shows its `\r` before it. Python now
splits every diff it reads with `_diff_lines` (`\r\n|\r|\n`, no trailing empty line) — exactly the
port's `_splitlines` — in `parse_unified_diff` and `parse_unified_diff_sides` (the `gate_diff_text`
door) and in `gate_diff` for both `git diff` and `git diff --name-status` (the git door). A lone `\r`
is still a break in both, as it already was in the port (section E, item 7).

**Two port-fidelity changes ride with it, and neither changes the Python.** Once Python stopped
cutting lines at those characters, the lines it hands its regexes can hold them, and two JavaScript
readings of such a line differed from Python's:

- JavaScript's `m` flag lets `^` match after `\r`, U+2028 and U+2029; Python's `re.M` only after `\n`.
  The port's `got` count and its symbol test now use `(?<![^\n])`, which is Python's `^` under `re.M`,
  instead of the `m` flag.
- JavaScript's `\s` matches U+FEFF and not U+001C–U+001F or U+0085; Python's is the reverse. The
  symbol test (`hit`), the one place the port writes `\s` over the added lines, now spells out
  Python's `\s` (`_PY_WS`). Without this, F-2 would have opened new disagreements on a symbol line
  led by U+0085 or U+001C–U+001E.

**Measured.** On a grid of 242 inputs — 22 space and separator characters in 11 positions a test or
a definition can take (re-indent, lead, mid-line, context line, removed line; symbol lead, mid-line,
between `def` and the name, after the name, changed; a header-shaped fragment) — Python and the port
disagree on **51** at `98a5c368`, **49** at the round-3 head and **1** now: a symbol name followed by a
non-ASCII letter (section E). On the 3,282 records that predate this round, F-2 moves nothing; one real
commit in the corpus (`commit:77abd0597b`) holds a U+0085, is now parsed differently, and none of its
claims moves.

**What it moves against `98a5c368`, in the Python.** A context line holding U+2028 followed by
`+def test_new():` no longer forges an added test; a U+2028 before a `+++ b/…`-shaped fragment no
longer forges a changed file; and a symbol definition led by one of these characters is now read as a
definition. For U+000C that is right (Python accepts a form feed in indentation). For U+000B,
U+001C–U+001E, U+0085, U+2028 and U+2029 it is the symbol test's `\s` reading a character CPython
refuses as indentation — the same reading `98a5c368` already gives NBSP, U+3000 and the other Unicode
spaces. It is a VERIFIED on a line that does not compile, it is pinned as a limit
(`path2:f2-limit-hit-reads-a-line-separator-as-indent`) and it is listed in section E.

**Pinned:** `path2:f2-a-vertical-tab-is-not-a-line-break`, `…-a-form-feed-is-not-a-line-break`,
`…-a-line-separator-is-not-a-line-break`, `…-a-paragraph-separator-is-not-a-line-break` (the four
classes the reviewer named, on both ports), `…-a-line-separator-mid-line-starts-no-line`,
`…-a-context-line-holding-a-separator-adds-nothing`, `…-a-separator-before-a-header-shape-adds-no-file`,
`…-a-form-feed-indent-defines-a-function`, `…-limit-hit-reads-a-line-separator-as-indent`. Tests
`test_f2_a_diff_splits_on_crlf_cr_and_lf_and_nothing_else`,
`test_f2_the_four_classes_the_ports_split_on_differently_now_read_alike`,
`test_f2_a_separator_inside_a_line_no_longer_forges_a_line`, and on the git door
`test_f2_f3_the_git_door_reads_the_same_lines_as_the_raw_door` (the four classes and NBSP),
`test_f2_the_git_door_does_not_forge_an_added_line_from_a_context_line` and
`test_f2_the_git_door_reads_name_status_as_git_writes_it` (a path holding U+2028, `core.quotePath`
off).

### F-3 — `got` reads the indent the pairing reads

**Reviewer evidence (protocol lens, major).** The indent shape of section A.2. Mutant P14 — exactly this
change — survived every diffgate test, so no test told the two readings apart.

**Change.** `_GOT_TEST_LINE` is `^\uFEFF?[ \t]*def test_` in both ports, the indent of
`_DEF_TEST_LINE`. `got` and the added-side pairing now count exactly the same lines.
`test_f3_got_and_the_pairing_read_exactly_the_same_lines` asserts it for every space character either
port knows.

**Measured.** An NBSP or U+3000 re-indent now reads "Added 0 tests." VERIFIED and "Added 1 test."
CONTRADICTED on both ports; on `98a5c368` it was the reverse on both. That is a new accusation against
the baseline, and it is correct: CPython rejects every one of these characters as indentation, so the
line defines nothing. **The recall cost:** a new test whose `def` line opens with U+000C is legal Python
and is now counted by neither port. The Python never counted it (`splitlines()` cut the line at the
form feed and dropped the rest); the port did, until now.

**Pinned:** `path2:f3-an-nbsp-reindent-is-not-an-added-test`,
`path2:f3-an-ideographic-space-reindent-is-not-an-added-test`; NBSP on the git door as well.

### F-4 — an off-tree prefix beside an on-tree one no longer withdraws a sure accusation

**Reviewer evidence (reconciliation lens, minor).** "Only modified `src/` and `../docs` as specified."
over a diff touching only `evil/x.py`: `98a5c368` CONTRADICTED "paths outside 'src' and 'docs':
['evil/x.py']"; the round-3 head UNCHECKABLE on the off-tree prefix. `evil/x.py` is outside `src/` and
outside every reading of `../docs`, so nothing about the second prefix made the accusation unsafe. The
scorer rejected the move (`CONTRADICTED -> UNCHECKABLE`).

**The decision, stated.** R-3 still holds for an off-tree prefix standing alone: it abstains. Beside
at least one on-tree prefix, the outside paths are computed as before, and a real one accuses only if
**no reading of the off-tree prefix could hold it**. `../docs` from an unknown directory X is `X/docs`,
so a path could lie under it exactly when the prefix's named segments occur, in order and contiguously,
among the path's segments (`_could_lie_under`; both sides compared undotted, dots-only segments
dropped, so the test errs towards "could"). Only such paths are listed. Anything less certain abstains
with R-3's reason.

This is narrower than the literal instruction ("abstain only if that verdict would not be CONTRADICTED
on an on-tree path"), on purpose. Read literally, "Only touches `src/` and `../docs`" over `src/a.py`
and `docs/guide.md` would accuse on `docs/guide.md` — the path a writer in `src/` most likely meant by
`../docs`, and a new accusation against `98a5c368`, which read `../docs` as `docs`. The filter keeps
every accusation this rule makes inside the set `98a5c368` already made, since `98a5c368`'s reading is
one of the readings the filter considers.

**Measured.** The reviewer's case is CONTRADICTED again, now printing `'../docs'`. Over `docs/a.md`
and `evil/x.py` it accuses and lists `evil/x.py` only. Over `packages/docs/x.md` it abstains where
`98a5c368` accused: that accusation rested on reading `../docs` as the root `docs`, and withdrawing it
is the safe direction; the scorer admits it (section C, S-1) and counts it (`f4_withdrawals`: 1, the
pinned pair). Over `src/a.py` alone, or `src/a.py` with `docs/guide.md`, it still abstains, as R-3 did.

**Pinned:** `path2:f4-a-path-no-reading-could-hold-still-accuses`,
`path2:f4-a-path-some-reading-could-hold-abstains`, `path2:f4-only-the-sure-paths-are-listed`; test
`test_f4_an_off_tree_prefix_beside_an_on_tree_one_withdraws_only_what_it_could_answer`.

---

## C. Changes that are not rules

- **T-1 — `_dot_miss`'s `..` arm has a test** (protocol lens, minor; mutant P13 survived round 3).
  `test_f5_a_path_opening_with_two_dots_is_never_a_dot_miss` asserts `_dot_miss(".a/bar.py",
  ["bar.py"])` and `not _dot_miss("..a/bar.py", ["bar.py"])`: with the arm removed, a bare-filename
  prefix reads `..a/bar.py` as a dot miss. No pair: git emits no path opening with `..`.
- **B-1 — the bookmarklet's source is byte-pinned in `.gitattributes`** (protocol lens, major). The
  committed `bookmarklet_src.js` was LF and correct, but this checkout (`core.autocrlf=true`) held it
  CRLF, and `build_bookmarklet.py --check`, which this branch taught to compare the source too, exited
  1 on a correct tree. `web/gate/bookmarklet_src.js -text` joins the repository's other byte-pinned
  files; the file was renormalised and the bookmarklet rebuilt with terser 5.46.0. The build script's
  docstring claimed a NUL-byte protection the file no longer has (the two NUL bytes lived in this
  branch's `diffgate.js` at `ab3084d9` and are gone since the rebase); it now says what protects the
  bytes. `web/gate/README.md`'s `--check` description and its "nothing disagrees" sentence are
  corrected to what is measured.
- **D-1 — PATH-1's re-pinned pairs are recorded beside PATH-1** (protocol lens, minor):
  `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md`, with all three strings before and after.
- **P-1 — corpus mode records the shelf** (round 2, owed since): `path2_gates.py corpus` writes
  `shelf_input` = file name, byte size, and the row counts of `pr` and `f`, read through the same
  immutable connection. Checked on a synthetic three-PR shelf in a scratch directory (`bytes` 12288,
  `pr_rows` 3, `f_rows` 3); not run on the real shelf, which the orchestrator scores after this round.
- **S-1 — the scorer attributes this round's rules** (a change to the protocol the preregistration set
  out, recorded here as R-6 was). Scored against `98a5c368`, the new pinned pairs move verdicts that no
  rule in the amended attribution table explains, and the gates would reject the branch's own pairs.
  `path2_gates.py` now carries, as its own code: `f2_applies` (the record's `str.splitlines()` and git
  split differ), `f3_applies` (an added line the old `got` pattern counts and the new one does not),
  and an F-4 exception for `only_touches` CONTRADICTED → UNCHECKABLE whose reason is the off-tree
  abstention on a claim with an off-tree prefix key. F-2 and F-3 explain a move **only where the
  amended table does not**, and only on a record where they can have acted; every move and every new
  accusation they explain is counted under `fourth_pass` in the payload, by rule, kind and transition,
  so none is silent. A G-C2 eligibility move with no key moved is admitted under F-2 on the same test
  and counted. `raw_paths` splits as git does. F-1 needs no attribution: against `98a5c368` it restores
  main's reading, and C-3's accusations fall under the dotted-prefix exception already there. Whether a
  scorer may be taught a branch's new rules after its preregistration is the operator's call, as R-6
  was.

---

## D. What was measured

All on this round's working tree, with the corpora generated and deleted afterwards.

- **Differential** (`build_corpus.py`, `fuzz_corpus.py`, `py_side.py`, `js_side.js`,
  `differential.py`): **3,301 pairs, 7,056 claims (668 verified, 1,652 contradicted, 4,736
  uncheckable), 0 disagreements.** The same 3,301 with the round-3 head on both sides: 9
  disagreements, the nine F-2 pairs. With `98a5c368` on both sides: 11, those nine and the two R-1 BOM
  pairs. `check_pairs.js`: **125 pinned pairs, 0 disagreements** (71 of them PATH-2's, 19 added this
  round).
- **What moves.** Against the round-3 head, the Python moves 13 records and the port 14, every one a
  pair this round added; none of the 3,282 older records moves on either side. Against `98a5c368`, the
  Python moves 69 records and the port 70 (the difference is pinned pairs where only one of the two was
  wrong on main); both move the same 23 corpus records the third pass reported.
- **The scorer** (`path2_gates.py differential`, on the working tree, where G-C0 fails only because the
  tree is modified): 69 records moved; G-C1, G-C3, G-C4 and G-C6 pass; 0 `compat2_candidate` flips;
  new accusations `only_touches` 4 (the three dotted-prefix pairs the amendment allows and F-1's C-3
  pair) and `tests_added` 3; `fourth_pass`: F-2 applies to 10 records and explains 6 moves (one of them
  a new `tests_added` accusation, the context-line pair), F-3 applies to 6 records and explains 4 moves
  (two new accusations, the NBSP and U+3000 pairs), and 1 F-4 withdrawal. The clean-tree run at this
  round's head is recorded in `web/gate/README.md` and in PR #161.
- **#128's eleven:** section A.1. No verdict moves; three reasons do; precision 0.25.
- **Mutation check.** Only `styxx/diffgate.py` was copied into a scratch directory and loaded as
  `styxx.diffgate` through `importlib` in a pytest plugin; the test set was
  `tests/test_diffgate_path2.py`, `test_path1.py`, `test_port_is_current.py`, `test_diffgate.py`,
  `test_diffgate_bc1.py`, `test_diffgate_bin1.py`, `test_diffgate_compat.py`,
  `test_diffgate_compat2.py`, `test_diffgate_false_accusations.py` and `test_diffgate_pr.py`. Control:
  374 passed, 6 xfailed. **All 16 mutants killed:** F-1 reverted (15 tests); F-2 reverted at each site
  separately — `parse_unified_diff` (9), `parse_unified_diff_sides` (1), the git door's added lines
  (1), the git door's name-status (1) — and all at once (9), `\n` alone (1), the trailing empty line
  kept (1); F-3 reverted (17); F-4 without the narrowing (4), without the reading filter (6),
  `_could_lie_under` keeping dots (1), as a set rather than a contiguous run (1), false on an empty
  prefix (1), and the narrowing applied to a lone off-tree prefix (1); T-1's arm removed (1). The
  empty-prefix arm of `_could_lie_under` is unreachable through `_gate` — a key made only of dots and
  slashes is emptied by `rstrip("/.")` and BC-1 answers before it — and is killed only by its unit
  assertion, which is where it is meant to be pinned.
- `build_bookmarklet.py --check` with terser 5.46.0 on this Windows checkout: all three `matches`,
  exit 0.

---

## E. What is still not repaired

Carried forward, re-read this round:

1. **#97 order dependence within one tier** (amendment limit 1). Unchanged.
2. **Dotfile renames** (limit 2): the `rename from` path is never registered. Unchanged.
3. **`symbol_added` pairing on a removed docstring line** (limit 3). Unchanged.
4. **The status-A exclusion keeps the shelf fold's over-count** (limit 4). Unchanged.
5. **What is left of limit 5, the symbol test.** After F-2, the only Python/port disagreement on the
   242-input grid is a symbol name followed by a non-ASCII letter (Python's `\b` is Unicode-aware,
   JavaScript's is ASCII). Separately, and on both ports alike, the symbol test's `\s` reads any
   Unicode space as indentation, so "Adds function foo." over a `def foo` line led by NBSP, U+3000,
   U+000B, U+001C–U+001E, U+0085, U+2028 or U+2029 is VERIFIED although CPython rejects the line;
   F-2 added the last seven of those to the Python's reading (the port already read U+000B, U+2028
   and U+2029 this way, and now reads all seven as the Python does), and one is pinned as a limit. And a `def` led by U+FEFF — legal at the head of a file, which Python reads
   as UTF-8 with a BOM — is CONTRADICTED on both ports; the port read it as a definition until it took
   Python's `\s`, and it follows the instrument by contract. Narrowing the symbol test's indent, or
   its `\b`, moves `symbol_added` verdicts and wants its own freeze.
6. **A form-feed-indented new test** is legal Python and is counted by neither port (F-3's recall
   cost; the Python never counted it).
7. **A lone `\r` inside a line** is still a line break in both ports, where git would keep the line
   whole. The two agree; no corpus record carries one that matters.
8. **The summary side.** F-2 changes how a diff is read. The claim templates read the summary with
   JavaScript's `\s`, `\b` and `m`-flag `^` in the port and Python's in the instrument; those
   differences are not measured by this round's grid and are not touched.
9. **`..docs`**, a literal name opening with two dots, abstains with the reason "is relative to a
   directory the diff does not name", which may not be what it is. The verdict (UNCHECKABLE) is safe;
   the wording is not exact.
10. **F-4 abstains where it could have accused on the most likely reading** ("`src/` and `../docs`"
    over `docs/guide.md`). A deliberate recall sacrifice, stated in F-4.
11. **`path2_differential_gates.json` is still not committed** (amendment limit 7); it is regenerated
    and committed with the RESULT.
12. **The corpus gates have not been run on the real shelf this round**; the orchestrator runs them.
    P-1 is checked on a synthetic shelf only.
13. **Issue #128's modes 2, 3, 5 and 6** are not repaired and are not scheduled.
