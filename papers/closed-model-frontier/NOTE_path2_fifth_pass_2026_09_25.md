# NOTE — PATH-2, fifth pass: one reading per line, the port spells Python's classes, and the scorer credits a rule only if reverting it gives the baseline back

2026-09-25. Branch `fix/diffgate-path-resolution` (pull request #161), on `origin/main` `98a5c368`, on top
of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path2_third_pass_2026_09_25.md`,
`NOTE_path2_fourth_pass_2026_09_25.md` and `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md`.

**None of those documents is edited.** The preregistration, the amendment and the erratum are public and
frozen; the three notes were each committed alone before their round's code, and that code exists. Where
one of them is wrong, the correction is here (section A). Where this note and an earlier document
disagree about what the instrument does, this note is what the code does.

This note is committed alone, before any code of this round. Its measurements were taken on the working
tree whose instrument, port, pinned pairs, tests and scorer this round's code commits carry.

---

## The principle

Rounds 2, 3 and 4 each repaired one whitespace character in one pattern and opened the next: a U+FEFF
the test count did not read (round 2), an NBSP the pairing did not read (round 3), a form feed the symbol
pairing did not read and a new test nobody counted (round 4). Each repair was right about its instance
and silent about the class, because every pattern that must read the same lines as another was written
on its own and kept in step by hand.

**Where two readings must agree, they are one reading.** This round applies that three times:

1. **A Python definition line is read as CPython reads it, by one pattern per question.** CPython's
   tokenizer takes space, tab and form feed as indentation and between tokens, and nothing else; a U+FEFF
   may open a file. Every pattern that reads a definition line opens with that one class. The test count
   and the test pairing are one pattern; the symbol test and the symbol pairing are one pattern. The
   agreement is then enumerated over the whole class, not argued (V-1).
2. **The port spells Python's classes wherever it reads a diff line.** JavaScript's `\s`, `\w`, `\b`, `.`,
   `trim()` and printable set are each a different set from Python's. Wherever the port reads a diff
   line, it now writes Python's out (V-2). The claim templates, which read the description, are the one
   place it still does not, and the count of what that leaves is published (section D).
3. **The scorer credits a rule with a move only if the instrument with that rule reverted gives the
   baseline claim back** (V-3). A record-wide test of "could this rule have acted here" is replaced by a
   per-claim counterfactual that the scorer computes on its own copy of the instrument.

V-4 repairs the two F-4 readings the round-4 review found, with the same method: the prefix is read as it
was written, before the key stripped what decided the answer.

---

## A. Corrections to earlier notes

### A.1 The fourth-pass note, section A.1: "Under the undotted spelling no reason moves at all"

False. Rebuilt from `bench2_audit.json` and `bench2_dataset.jsonl` by the fourth pass's method,
`open-policy-agent/cert-controller#415` moves its reason under the undotted path spelling too, because
the claimed PREFIX keeps its dot: `98a5c368` prints `paths outside 'githiub/workflows/dependabot.yml':
['github/workflows/dependabot.yml']`, this branch `paths outside '.githiub/workflows/dependabot.yml':
['github/workflows/dependabot.yml']`. The other ten rows do not move under that spelling. The verdicts,
and the rows of the A.1 table, stand: all eleven match PATH-1's published after-column, precision stays
2 in 8 = 0.25, and this round moves none of the eleven under either spelling (round-4 head and this tree
agree on all eleven, both spellings).

### A.2 The fourth-pass note, section A.1: "the committed shelf"

The shelf is not committed. It is the local, uncommitted EXTERNAL-1 shelf
`papers/closed-model-frontier/external1_shelf.sqlite`, 5,033,086,976 bytes, ignored by `.gitignore`
(`papers/closed-model-frontier/external1_shelf.sqlite*`, line 177 on this branch) and held by no commit
(`git log --all -- '*external1_shelf.sqlite'` prints nothing). The evidence that #1142's `gitignore` key was
`.gitignore` therefore cannot be reproduced from the repository; it is reproduced from that file by this
read-only query, run this round:

    sqlite3.connect("file:///<checkout>/papers/closed-model-frontier/external1_shelf.sqlite?immutable=1", uri=True)
    SELECT id, html_url FROM pr WHERE id = 3127853160;           -- Albeoris/Memoria/pull/1142
    SELECT f.filename, f.status FROM f WHERE f.pr_id = 3127853160;

The second query returns 28 rows, among them `('.gitignore', 'modified')` twice. The committed receipts
(`bench2_audit.json`, `bench2_dataset.jsonl`) carry only the pre-#121 undotted key.

### A.3 The fourth-pass note, section C (T-1): "git emits no path opening with `..`"

False. In a scratch repository, `git diff --name-status` printed `A	..a/bar.py` for a file added at
`..a/bar.py` (git 2.52.0). `_dot_miss`'s `..` arm is unreachable through `_gate` for a different reason:
a path that opens with `..` and lies inside an undotted prefix once one dot is dropped also lies inside
it by the bare-filename basename reading, so it is never outside; and no undotted non-bare prefix can
hold a path that opens with a dot. The arm's mutant is killed only by its unit assertion, which is where
it is pinned.

### A.4 The fourth-pass note, section E.5 and its F-2 paragraph: U+001F is missing

The fourth pass also made the port read U+001F as indentation in the symbol test: on `98a5c368` the port
said CONTRADICTED over `+\x1fdef foo():` and the Python VERIFIED (U+001F is in Python's `\s` and not a
`str.splitlines()` break, so the Python read it on `main` already); at the round-4 head both said
VERIFIED, on a line CPython refuses. The note's lists named U+000B, U+001C–U+001E, U+0085, U+2028 and
U+2029 and left U+001F out. After V-1 both ports read it as nothing (CONTRADICTED).

### A.5 The third-pass note, section D item 5: "Neither shape can produce a false VERIFIED"

False, and the fourth pass did not correct it. (a) The `\b` shape gave a false VERIFIED on the PORT,
the public door: "Added function foo." over `+def foo<U+00E9>():` read VERIFIED in the port and
CONTRADICTED in the Python. (b) A line `hit` reads and the pairing does not gives VERIFIED on a
re-indent or an edit of an existing definition, because the pairing then sees no definition to pair: a
form-feed re-indent of `def foo():` read VERIFIED at the round-4 head on both ports (round-4 review,
finding 1), and so did an edit of a generic `def f[T](` (amendment limit 3). V-1 closes both.

### A.6 The fourth-pass note, F-2 and F-3: "For U+000C that is right" and "The recall cost"

Both wrong. Reading a form feed as indentation in `hit` was right only if the pairing read it too; it did
not, so an FF re-indent of an existing function was a false VERIFIED of the #101 kind. And the form-feed-led
new test that F-3 stopped counting was not a recall cost: "Added 1 test." over it read CONTRADICTED on
every door, a false accusation of a true claim, contradicting the branch's own pinned reading that an
FF-led line is a definition. V-1 repairs both.

---

## B. Rule changes this round

Each is mirrored in `web/gate/diffgate.js`, pinned by pairs in `web/gate/differential/path2_pairs.json`
that both implementations are held to, tested in `tests/test_diffgate_path2.py`, and mutation-checked
(section D).

### V-1 — one reading of a Python definition line

**Reviewer evidence.** Round-4 correctness lens, majors 1 and 2: a form-feed re-indent of `def foo():` read
"Added function foo." VERIFIED on both Python doors and the port (`98a5c368`: CONTRADICTED); a new test led
by a form feed read "Added 1 test." CONTRADICTED on every door. Both are one defect: `hit` read `\s*` while
the symbol pairing read `[ \t]*`, and `got` and the test pairing read `[ \t]*` where CPython reads
`[ \t\f]*`. Minor 6(d): the remaining `\b` case was a false VERIFIED on the port. Protocol lens, major 2:
nothing committed guarded the round-4 port changes to `hit`.

**Change.** Three constants, in both ports, and every definition-line pattern built from them:

    _DEF_INDENT = ^\uFEFF?[ \t\f]*                       CPython's indentation after one optional BOM
    _DEF_SEP    = [ \t\f]+                               between `async`, `def`/`class` and the name
    _NAME_END   = (?=[\x00-\x2f\x3a-\x40\x5b-\x5e\x60\x7b-\x7f]|$)
                                                         the next character is ASCII and cannot
                                                         continue a name, or the line ends

- `got` (`^\uFEFF?[ \t\f]*def test_`) and the added-side test pairing (`_DEF_TEST_LINE`) open with
  `_DEF_INDENT`; the removed side (`_DEF_TEST_LINE_REMOVED`) also, and alone accepts `async`.
- `hit` is no longer its own pattern. It asks whether any added line matches
  `_symbol_def_line_added(name)` — `_DEF_INDENT (?:def|class) _DEF_SEP NAME _NAME_END` — which is the very
  pattern the symbol pairing counts added definitions with. The removed side (`_symbol_def_line`) is the
  same pattern with `(?:async _DEF_SEP)?`.
- The port writes the same three constants; neither side's definition patterns contain `\s`, `\w` or `\b`.

**Beyond the literal instruction, and why.** The instruction was one leading class. That alone does not
make `hit` and the pairing read the same lines: `hit` also read `\s+` between `def` and the name and `\b`
after it, the pairing `[ \t]+` and `(?=[ \t(:]|$)`. The grid below shows the lines in between: a changed
`class Stack[T]:` (the pairing's lookahead refuses `[`, `\b` does not), `def\ffoo` (legal), `def foo\f(`.
Each is a line `hit` reads and the pairing cannot, i.e. a false VERIFIED of the #101 kind on an edit. So
`hit` IS the pairing's added-side pattern, and that pattern takes CPython's separator and a name end that
both ports compute identically without a Unicode table: a non-ASCII character after the name either
continues it (`fooé` is another name; so is `foo·`, U+00B7, which Python's `\b` read as an end) or is a
character CPython refuses there. This replaces the amendment's frozen lookahead `(?=[ \t(:]|$)` (C-1), and
it closes **amendment limit 3 as the erratum's item 2 restated it** ("a changed generic function or class
still verifies"): a changed generic now pairs and abstains (#101). That is the safe direction, and it is a
change to a frozen rule, recorded here for the operator.

`async`: `got` never counted `async def test_` and `hit` never read `async def`, on `main` or since. The
added side of both pairings therefore excludes `async` as well, so that "the same lines" holds; the
removed side keeps it (R-2). An added `async def` is not read by either template — a carried blind spot,
pinned as a limit (`path2:v1-limit-an-added-async-def-is-not-read`), not repaired: reading it moves
verdicts toward VERIFIED on common code and wants its own freeze.

**The grid.** Every character of Python's `\s` (29) and U+FEFF, in the position a character can take on a
definition line — leading an added test (new, re-indented) or a removed one, leading an added symbol
definition (new, re-indented) or a removed one, between `def` and the name (new, changed), after the name
(new, changed) — plus six name-end characters that are not spaces (U+00E9, U+00B7, U+0301, U+4E00, `[`,
`x`): 312 cells. Door A: `gate_diff_text` over a hand-written hunk, and the port over the same text. Door
B: `gate_diff` over a real two-commit repository, and the port over the bytes `git diff` printed for it.
Summaries: "Added 0 tests. Added 1 test." for tests, "Adds function foo." for symbols.

| instrument, both sides | door A disagreements | door B disagreements |
|---|---|---|
| `98a5c368` | 40 | 40 |
| round-4 head `60d678a5` | 4 (the `\b` case: U+00E9 and U+4E00 after the name, new and changed) | 4 |
| this round | **0** | **0** |

This round's readings, which both ports and both doors now give (V = VERIFIED, C = CONTRADICTED,
U = UNCHECKABLE; one letter per claim, tests "Added 0" then "Added 1"; **bold** = moved against
`98a5c368`'s Python; "(git …)" = the git door differs, because a U+000A in a file makes git write two
lines where the hand-written hunk has one):

| char | t-lead-new | t-lead-reindent | t-lead-removed | s-lead-new | s-lead-reindent | s-lead-removed | s-sep-new | s-sep-changed | s-end-new | s-end-changed |
|---|---|---|---|---|---|---|---|---|---|---|
| U+0009 | CV | **VU** | **VU** | V | **U** | **U** | V | **U** | V | **U** |
| U+000A | VC (git CV) | VC | CV (git VC) | C (git V) | C | V (git C) | C | C | V | **U (git C)** |
| U+000B | VC | VC | CV | C | C | V | C | C | V | **U** |
| U+000C | **CV** | **VU** | **VU** | **V** | **U** | **U** | **V** | **U** | V | **U** |
| U+000D | VC | VC | CV | C | C | V | C | C | V | **U** |
| U+001C | VC | VC | CV | C | C | V | C | C | V | **U** |
| U+001D | VC | VC | CV | C | C | V | C | C | V | **U** |
| U+001E | VC | VC | CV | C | C | V | C | C | V | **U** |
| U+001F | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | V | **U** |
| U+0020 | CV | **VU** | **VU** | V | **U** | **U** | V | **U** | V | **U** |
| U+005B |  |  |  |  |  |  |  |  | V | **U** |
| U+0078 |  |  |  |  |  |  |  |  | C | C |
| U+0085 | VC | VC | CV | C | C | V | C | C | **C** | **C** |
| U+00A0 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+00B7 |  |  |  |  |  |  |  |  | **C** | **C** |
| U+00E9 |  |  |  |  |  |  |  |  | C | C |
| U+0301 |  |  |  |  |  |  |  |  | **C** | **C** |
| U+1680 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2000 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2001 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2002 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2003 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2004 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2005 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2006 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2007 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2008 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2009 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+200A | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+2028 | VC | VC | CV | C | C | V | C | C | **C** | **C** |
| U+2029 | VC | VC | CV | C | C | V | C | C | **C** | **C** |
| U+202F | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+205F | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+3000 | **VC** | **VC** | CV | **C** | **C** | V | **C** | **C** | **C** | **C** |
| U+4E00 |  |  |  |  |  |  |  |  | C | C |
| U+FEFF | **CV** | **VU** | **VU** | **V** | **U** | **U** | C | C | **C** | **C** |

Read by rows: space, tab, form feed and (leading only) U+FEFF are indentation and pair; everything else
defines nothing. U+000D is a line break in both ports (section E, item 8), so its row reads a different
diff. The name-end columns are ASCII-decided: a control character or `[` ends the name, a letter, a mark
or a non-ASCII space does not.

`test_v1_every_definition_pattern_opens_with_one_indent` asserts the equalities line by line for all 30
characters in every position; `test_v1_the_grid_on_the_raw_door` pins the 28 non-break characters' readings;
`test_v1_the_port_reads_the_grid_as_the_python_does` runs the 300 raw-door cells through the port and
requires 0 disagreements, so the zero above is a committed receipt, not a number typed once;
`test_v1_the_git_door_reads_the_same_definition_lines_as_the_raw_door` covers six shapes on the git door
(the two form-feed pairs the task names, a vertical-tab re-indent, a U+0085-led and a U+FEFF-led new
function, a U+2028 re-indented test).

**What it moves.** On the 3,301 differential records that predate this round, V-1 moves exactly the four
pinned pairs it re-pins, on both sides, and no corpus record:

| pair | before (round-4 head) | after |
|---|---|---|
| `path2:f2-a-form-feed-is-not-a-line-break` | "Added 0 tests." VERIFIED, "Added 1 test." CONTRADICTED | VERIFIED `(1 changed, not added: #101)`, UNCHECKABLE (#101) |
| `path2:f2-limit-hit-reads-a-line-separator-as-indent` | VERIFIED (pinned as a limit) | CONTRADICTED — the limit this id names is closed |
| `path2:r4-a-changed-generic-definition-still-verifies` | VERIFIED (amendment limit 3) | UNCHECKABLE (#101) |
| `path2:r4-a-function-made-generic-still-verifies` | VERIFIED (amendment limit 3) | UNCHECKABLE (#101) |

Each record now carries a `repinned` field saying so; the ids are kept so that earlier notes still find
them. The limit V-1 introduces: a definition led by U+FEFF reads as a definition wherever it stands
(`path2:v1-limit-a-bom-led-def-reads-as-a-definition-wherever-it-stands`), although CPython accepts the
BOM only at the head of a file — the reading R-1 gave the test count and C-1 gave the pairing, now given to
`hit` too, so that the three agree.

**Pinned:** `path2:v1-a-form-feed-reindented-function-is-changed`, `…-a-form-feed-led-new-test-is-counted`,
`…-a-vertical-tab-reindented-function-defines-nothing`, `…-an-nbsp-led-new-function-defines-nothing`,
`…-a-form-feed-separated-changed-function-pairs`, `…-a-changed-generic-class-pairs`,
`…-a-name-followed-by-a-non-ascii-letter-is-another-name`, `…-a-name-followed-by-a-middle-dot-is-another-name`;
and, guarding the round-4 port changes as the protocol lens asked, `…-a-def-after-a-mid-line-line-separator-is-not-a-definition`
(CONTRADICTED; fails if `hit` reads the blob with the `m` flag), `…-limit-a-bom-led-def-reads-as-a-definition-wherever-it-stands`
(VERIFIED), `…-a-next-line-led-def-is-not-a-definition` and `…-a-file-separator-led-def-is-not-a-definition`
(CONTRADICTED; fail if `hit` reads Python's `\s` again); `…-limit-an-added-async-def-is-not-read`,
`…-limit-an-added-async-def-beside-a-changed-one-is-not-read`.

### V-2 — the port spells Python's classes wherever it reads a diff line

**Reviewer evidence.** Round-4 correctness lens, major 3: F-2 hands both ports whole lines holding
U+001C–U+001E and U+0085; the five `_COMPAT_LANGS` patterns still used JavaScript's `\s`, so a removed
`def<U+0085>public_api():` was a removed public definition to the Python and nothing to the port — 16
new Python/port disagreements in `compat_claim`, reason, `surface_removed` and `compat2_candidate`.
Minor 7: a `+++ b/<path><U+0085>` header stripped differently (Python `str.strip`, JavaScript `trim`).

**Change, in the port only.** `_PY_WS` (Python's `\s`, 29 code points), `_PY_W` (Python's `\w`,
`[\p{L}\p{N}_]` with the `u` flag) and `_PY_B` (Python's `\b` built from `_PY_W`) are spelled out, and:

- the five `_COMPAT_LANGS` patterns use them for every `\s`, `\w` and `\b` (the Go pattern's trailing
  `\b` and Java's `[\w<>\[\],\s]` class included);
- the COMPAT reference test (`\bNAME\b` over the added lines of the language) uses `_PY_B`;
- `_compatParams` collapses Python's `\s` and strips as `str.strip()` does (`_pyStrip`);
- the `---` / `+++` header paths are stripped by `_pyStrip`, in both parsers;
- `_DIFF_GIT` and `_BINARY_LINE` read `[^\n]` where the Python writes `.` (JavaScript's `.` also stops at
  `\r`, U+2028 and U+2029, which F-2 leaves inside a line);
- `pyRepr` escapes what Python's `repr()` escapes: a non-ASCII character in categories C or Z (other than
  the space) prints as `\xhh`, `\uhhhh` or `\Uhhhhhhhh`, so a reason printing a path that holds U+2028 or
  U+00A0 reads the same in both.

**Every `\b` the port uses on diff text, reviewed.** Three: `hit`'s (gone, V-1), the COMPAT reference
test's and the Go pattern's (both `_PY_B` now). Every `\s`, `\w` and `\b` left in the port reads the
DESCRIPTION: the claim templates, the containment and sentence-split patterns, the declaration reader.

**Unicode versions.** `[\p{L}\p{N}_]` in Node 24 (Unicode 16.0) equals Python 3.12's `\w` (Unicode 15.0)
on every code point Unicode 15.0 assigns; it also reads the 5,004 code points assigned since as word
characters, which Python 3.12 does not. The same holds for path lower-casing: `str.lower()` and
`toLowerCase()` differ on 27 code points, all cased only since Unicode 15.1. These track the engines, not
the port, and are disclosed rather than tabled.

**The `compat2_candidate` flip, decided.** F-2 made the Python read whole lines, so a removed public
definition with U+000B, U+000C, U+001C–U+001E, U+0085, U+2028 or U+2029 in a `\s` position is now read
where `98a5c368` cut the line and read nothing; `compat2_candidate` goes False → True. The Python's COMPAT
patterns keep Python's `\s` (they read four other languages, whose whitespace is not CPython's, and they
read NBSP the same way on `main`); the port follows them; and the scorer admits such a flip only when F-2
alone gives the baseline back, counts it under `attribution.compat2_flips_admitted`, and still fails
G-C6 on any other flip (V-3). On the real shelf, G-C6 can therefore fire only on a flip F-2 does not
explain.

**Measured.** Outside the definition grid (method in `web/gate/README.md`): a header path followed by one
of 28 characters, 6 disagreements on `98a5c368` and the round-4 head, 0 now; a changed path holding one of
the 28 printed in a reason, 25 on `98a5c368`, 20 at the round-4 head, 0 now. On the pinned-pair corpus,
the round-4 head read 17 Python/port disagreements over this round's 3,337 records and `98a5c368` 27;
this tree reads 0.

**Pinned:** one `compat_claim` pair per language with U+0085 in a `\s` position (`path2:v2-compat-python-…`,
`…-typescript-…`, `…-go-…`, `…-rust-…`, `…-java-reads-pythons-whitespace`), U+001C (Python) and U+001F
(Rust), U+FEFF (not whitespace to Python), a name followed by a non-ASCII letter in the reference test, a
non-ASCII Go name read whole, a parameter list holding U+0085; a header path followed by U+0085 and by
U+FEFF; a binary header and a binary deletion line holding U+2028; a reason printing U+2028 and U+00A0.
Every one fails on the round-4 port.

### V-3 — the scorer attributes per claim, by counterfactual

**Reviewer evidence.** Both lenses, major: `path2_gates.py`'s F-2 attribution was per record and
kind-blind (`rule = "F-2" if f2 else …`, `f2 = diff.splitlines() != git_lines(diff)`), so one stray
separator anywhere in a pull request admitted every otherwise-unattributed move on it. The round-3 blocker
itself (F-1 reverted in-process) was admitted on a record holding a form feed in a context line no claim
reads; so was round-4 finding 1's false VERIFIED.

**Change.** The scorer loads its own copy of the repaired module from the same bytes (`CF`; `new` is never
patched) and a table of reverts, each rule's code as it was before the rule, written out in the scorer:

| rule | reverted to |
|---|---|
| #97 | `_find_path` = the any-tier loop, in diff order |
| #121 | `_norm` = `lstrip("./").lower()` |
| #101 | `_changed_test_defs` = 0, `_definition_only_changed` = False |
| R-1 | `got` without its `\uFEFF?` |
| F-2 | `_diff_lines` = `str.splitlines` |
| F-3 | `got` = `^\uFEFF?\s*def test_` (its round-3 form) |
| V-1 | the definition-line patterns and `hit` in their fourth-pass forms |
| V-4 | `_parent_prefix` reads nothing; `_could_lie_under` is the fourth pass's |

A moved claim is attributed to the rules whose single revert gives back the baseline claim — verdict,
reason and, for a compatibility claim, the whole detail. When none does, the smallest set of two or three
that does is the attribution (a move needing two rules, e.g. `../docs/..` needs #121 and V-4); when none
does, the move is a violation (`G-C4_unattributed_counterfactual`). Every rule in the attribution must
then admit the move: #97, #121 and #101 through the amended table exactly as before (its preconditions
and directions are unchanged and still required); R-1 and F-3 admit `tests_added` moves, V-1
`tests_added` and `symbol_added`, F-2 any kind, V-4 an `only_touches` move to the off-tree abstention.
G-C6 admits a `compat2_candidate` flip only when F-2 alone explains it; G-C2 admits an eligibility move
only when reverting #121 (with a key moved, the table's own test) or F-2 gives the baseline's eligibility
back. Every move a post-amendment rule explains is counted under `attribution` in the payload by rule,
kind and transition, with new accusations, new VERIFIEDs and compat2 flips counted apart.

**Which rules are still attributed by inspection.** C-1, C-2, C-3, R-2, R-3, F-1 and F-4 have no revert of
their own: each acts only through a reading #121 or #101 introduced (a dotted key, or the pairing), so
reverting #121 or #101 reverts it too, and a move they cause is attributed to #121 or #101 and must pass
the table. The table's own tests (`key_moved`, `test_def_changed`, `symbol_def_changed`, the dotted-prefix
exception, the F-4 withdrawal) are inspection and remain required on top of the counterfactual, never
instead of it. And a defect planted INSIDE a rule's own code reverts with that rule: the counterfactual
then names the right rule, and only that rule's admission test and the pinned pairs can refuse the move.
The round-3 blocker is exactly such a case — it lives in #121's reading of a dotted prefix — and the
table's direction test refuses it (`G-C4_direction:only_touches`).

**The gate can fail.** A synthetic shelf in scratch, three pull requests: one touching only `.github/`
with a stray U+2028 in an unrelated context line, claiming "Only touches .github."; a clean one; one
with a real F-2 move (a form-feed-led definition). With a scratch copy of this round's `diffgate.py`
carrying the round-3 blocker swapped in as both `new` and `CF`: the round-4 scorer (`60d678a5`) admitted
the planted move (`fourth_pass`: `F-2 only_touches: VERIFIED -> UNCHECKABLE`) and reported no attribution
violation; this scorer reports `G-C4_direction:only_touches` and exits 1, while still admitting the real
F-2 move. With the instrument unmutated it reports no attribution violation. (Run on this working tree,
G-C0 also fires because the tree is uncommitted; the clean-tree run is in `web/gate/README.md`.) The shelf
and the mutant copy were deleted. The probe is pinned as `test_v3_a_stray_form_feed_no_longer_excuses_the_round_3_blocker`,
beside `test_v3_a_real_f2_move_is_still_attributed_and_counted` and
`test_v3_every_revert_names_code_the_instrument_has` (all reverts at once give the baseline back on every
pinned pair); they skip where the baseline commit is not in the clone.

**Measured** (differential mode, this working tree): every attribution gate passes (G-C0 alone fails, because the tree is
uncommitted); 185 of 3,337 records move against `98a5c368`. Claims attributed to one rule: #97 27, #121 35, #101 26,
F-2 18, F-3 4, V-1 3, V-4 93; to two rules each of whose revert alone gives the baseline back (both are
necessary): #101+V-1 4, F-2+V-1 4; to a joint revert of two where neither alone did: #101+R-1 4, #121+V-4 2.
New accusations: `only_touches` 4 (the table's dotted-prefix exception, as in round 4), `symbol_added` 2
(V-1: an NBSP-led definition and a name followed by U+00B7, lines that define nothing), `tests_added` 4
(F-2 one, F-2+V-1 one — the form-feed-led new test, which now accuses "Added 0 tests." — F-3 two); 6
`compat2_candidate` flips, all False → True, all F-2's alone and admitted; 4 F-4 withdrawals; 28 records
whose `str.splitlines()` and git split differ (informational now).

### V-4 — a `..` the key used to lose

**Reviewer evidence.** Round-4 correctness lens, minor 5: `_could_lie_under` dropped a `..` that follows a
named segment (`../src/../docs` is `X/docs`, read as `X/src/docs`), and `rstrip("/.")` deleted a trailing
`/..` before off-tree-ness was decided (`../docs/..` read as `../docs`, `.github/../.` as `.github`), so an
accusation no reading of the prefix supported came back. Protocol lens, minor 3: F-4's two boundaries were
unpinned (mutants P10 and P11 survived every test and the scorer).

**Change, both ports.** `_parent_prefix(raw)` reads the prefix as written, before `rstrip("/.")`: with
trailing `/` and `.` segments dropped, if its last segment is `..`, the prefix is off-tree, could hold any
path, and the reason prints it as written (`prefix '../docs/..' is relative …`). `_could_lie_under` returns
"could" when a `..` or `...` segment follows a named one.

**The price, disclosed.** `docs/.` followed by a sentence period is written `docs/..`, and reads as the
parent of `docs`. The fuzzed corpus generates exactly that (`only modifies files under docs/..`): V-4 moves
92 of its claims, in 90 records, from CONTRADICTED to UNCHECKABLE. Abstaining is the safe direction on a
prefix that has two readings; reading the last dot as a period would need the claim's surrounding text,
and is not attempted. No real-corpus record moves. `src/...` is not a parent (an elision), and still reads
as `src`. Pinned as `path2:v4-limit-a-dotted-current-directory-before-a-period-reads-as-a-parent`.

**Pinned:** `path2:v4-a-dotdot-after-a-named-segment-could-hold-anything`,
`…-a-prefix-ending-in-dotdot-could-hold-anything`, `…-a-dotfile-parent-before-a-sentence-period-is-off-tree`,
`…-limit-a-dotted-current-directory-before-a-period-reads-as-a-parent`, and the two F-4 boundaries:
`…-f4-filters-on-the-off-tree-prefix-only` ("Only modified `src/` and `../docs`" over `lib/src/x.py` stays
CONTRADICTED listing `lib/src/x.py`) and `…-f4-two-off-tree-prefixes-with-no-on-tree-one-abstain` ("Only
touches ../a/ and ../docs/." over `evil/x.py` stays UNCHECKABLE). Tests `test_v4_*`.

---

## C. Changes that are not rules

- `web/gate/differential/py_side.py` pins `0a5522eb…`, this round's `styxx/diffgate.py` (LF), and its
  docstring stops describing the `\b` case as the one known disagreement.
- The bookmarklet is rebuilt with terser 5.46.0: `bookmarklet.min.js` sha256 `d8ce5111…`, 27,738 characters; `bookmarklet.href.txt`
  `3886387a…`, 27,749; `bookmarklet_src.js` `552b589c…`, 57,152. `--check`: all three match, exit 0. The
  minified port, loaded in Node with the UI stubbed, reads all 161 pinned pairs as the Python does.
- `web/gate/README.md` records this round's numbers and, beside its "0 disagreements" paragraph, the counts
  of what still disagrees outside the grid (section D).
- `tests/test_diffgate_path2.py`: the tests above; three fourth-pass tests re-read the form feed as V-1
  reads it; the pinned-pair count moves from 71 to 107.

---

## D. What was measured

All on this round's working tree; corpora generated and deleted afterwards.

- **Differential**: 3,337 pairs, 7,094 claims (671 verified, 1,571 contradicted, 4,852 uncheckable),
  **0 disagreements**; the same 3,337 with the round-4 head on both sides read 17, with `98a5c368` on both
  sides 27. `check_pairs.js`: 161 pinned pairs (107 of them PATH-2's, 36 added this round), 0 disagreements.
- **Moves.** Against the round-4 head, on the 3,301 records that predate this round: 94 on each side — the
  four re-pinned pairs (V-1) and 90 fuzzed records (V-4); no real-corpus record. Against `98a5c368`: 185
  records in the Python, 194 in the port (the difference is pinned pairs on which the two disagreed on
  `main`); 111 of them lie outside the pinned-pair files, on each side, where the round-4 head moved 21 —
  the 90 more are V-4's.
- **Outside the grid, what still disagrees.** The claim templates read the description with JavaScript's
  `\s`, `\w` and `\b`. Over a whitespace character between the words of twelve claim shapes (360 inputs):
  72 disagreements, the same on `98a5c368`, the round-4 head and this tree — exactly the six characters
  whose membership differs (U+001C–U+001F and U+0085 are Python whitespace, U+FEFF is JavaScript's) in all
  twelve positions. Over a non-ASCII letter or mark inside a claimed name or path (20 inputs): 12, likewise
  unchanged. These are the summary-side half of amendment limit 5, and are what `web/gate/README.md` now
  counts.
- **The scorer**: section B, V-3.
- **#128's eleven**: section A.1. No verdict moves, this round moves no reason, precision 0.25.
- **Mutation, Python.** Only `styxx/diffgate.py` was copied into a scratch directory and loaded as
  `styxx.diffgate` through `importlib` in a pytest plugin; the test set was the fourth pass's ten modules.
  Control: 450 passed, 6 xfailed. **21 mutants, 21 killed** (failing tests in parentheses). V-1: `got`
  back to `[ \t]` (10); the indent without the form feed (12); the indent as `\s*` (51); the indent
  without the U+FEFF (14); `hit` back to `^\s*(?:def|class)\s+NAME\b` over the blob (38); `hit` reading
  `async` (3); the name end back to `[ \t(:]` (14); the name end as `\b` (23); the separator back to
  `[ \t]+` (5); the removed test side without `async` (4); the removed symbol side without `async` (2);
  and the pairing's added side reading the removed pattern, which SURVIVED the run and was killed (3)
  once `path2:v1-limit-an-added-async-def-beside-a-changed-one-is-not-read` and a unit assertion were
  added (control then 451 passed). V-4: no written parent (5); trailing `.` segments kept (3);
  could-hold ignoring the written form (3); no `..` after a named segment (3); a `.` after a named
  segment counted as `..` (1); off-tree read on the key only (4); the reason printing the key (5); F-4's
  filter over every prefix, the protocol lens's P10 (3); two off-tree prefixes read as beside an
  on-tree one, P11 (3). Under a mutant the scorer's own tests skip (the scorer refuses an instrument
  outside the checkout), which is why a mutant run shows 3 skipped. The mutant copy and its directory
  were deleted.
- **Mutation, scorer.** Eight in-memory mutants of `path2_gates.py` (the mutated source run in one
  process with the real file's path; nothing copied), put through the `test_v3_*` tests' own logic: F-2
  attributed per record again; every post-amendment rule admitting anything; G-C6 admitting a flip any
  rule explains; the table no longer asked for its own rules; an unattributable move admitted; the
  counterfactual comparing verdicts only; V-1's revert forgetting `hit`; V-4's revert forgetting the
  written parent. Six were killed by the tests as they stood; the fifth and sixth survived and are
  killed by `test_v3_a_move_no_rule_explains_is_refused_whatever_the_table_says`, added for them.
- **Mutation, port.** 26 in-memory mutants of `web/gate/diffgate.js` (the mutated source compiled with
  `Module._compile`, nothing written), each run through `check_pairs.js`'s comparison over all 161
  pinned pairs: **26 killed** — every V-1, V-2 and V-4 change reverted one at a time, the round-4 port
  changes (`hit` on the `m` flag, on JavaScript's `\s`, on round 4's `_PY_WS` form), `_PY_WS` without
  U+0085, without U+001C–U+001F or with U+FEFF, and P10/P11. Four survived an earlier run and each now has
  the pair that kills it: `_PY_WS` without U+001C–U+001F (the U+001C and U+001F COMPAT pairs), the binary
  line on JavaScript's `.` (the binary deletion pair), the added symbol side reading `async` (the async
  limit pair), the pairing's added side reading the removed pattern (the async-beside-a-changed-def pair).
- **Tests**: the thirty modules that import `styxx.diffgate` or read `web/gate`: 1,498 passed, 1
  skipped, 6 xfailed, 0 failed. `tests/test_diffgate_path2.py` collects 140 tests (96 at the round-4
  head).

---

## E. What is still not repaired

1. **#97 order dependence within one tier** (amendment limit 1). Unchanged.
2. **Dotfile renames** (limit 2): the `rename from` path is never registered. Unchanged.
3. **`symbol_added` pairing on a removed docstring line** (one half of limit 3). Unchanged. Its other
   half, a changed generic definition, is closed by V-1.
4. **The status-A exclusion keeps the shelf fold's over-count** (limit 4). Unchanged.
5. **What is left of limit 5.** On the diff side, nothing the grid can see. On the summary side, the counts
   in section D. And three readings V-1 makes explicit rather than repairs: an added `async def` is read by
   neither template (as on `main`); a definition led by U+FEFF is read wherever it stands; a test whose
   `def` is separated from its name by anything but one space is counted by neither `got` nor the pairing
   (they agree).
6. **COMPAT reads Python's `\s`**, not CPython's tokenizer, in all five languages: a removed `def<U+0085>name`
   is read as a removed public definition, a line CPython refuses. Pre-existing for NBSP and the Unicode
   spaces; F-2 added the eight `str.splitlines()` characters. The flips are counted (V-3).
7. **Unicode versions**: section B, V-2.
8. **A lone `\r` inside a line** is still a line break in both ports, where git keeps the line whole.
9. **`..docs`**, a literal name opening with two dots, abstains with a reason that calls it relative.
10. **F-4 abstains where it could have accused on the most likely reading**, as the fourth pass stated.
11. **V-4 reads `docs/.` plus a sentence period as a parent** (section B).
12. **The counterfactual cannot see a defect inside a rule's own code** (section B, V-3).
13. **`path2_differential_gates.json` is still not committed** (amendment limit 7); it is regenerated and
    committed with the RESULT.
14. **The corpus gates have not been run on the real shelf**; the orchestrator runs them. Besides the
    violations, `attribution.moves_admitted_by_rule`, `new_verified_admitted`, `new_accusations_admitted`
    and `compat2_flips_admitted` are the numbers to read before merge.
15. **Issue #128's modes 2, 3, 5 and 6** are not repaired and are not scheduled.

---

## Protocol changes the operator is asked to accept

- V-1 replaces the amendment's frozen C-1 symbol lookahead `(?=[ \t(:]|$)` and separator `[ \t]+`, and the
  indent of every C-1 pattern, and closes amendment limit 3's generic half. Two pinned pairs move VERIFIED →
  UNCHECKABLE.
- V-3 changes how the scorer attributes (counterfactual, per claim, conjoined with the table), and what
  G-C6 and G-C2 admit (a flip or an eligibility move F-2 alone explains, counted).
- V-4 abstains on a prefix written to end in `..`, at the cost in section B.
