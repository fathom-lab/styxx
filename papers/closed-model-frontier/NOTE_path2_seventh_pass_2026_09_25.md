# NOTE — PATH-2, seventh pass: one name table for both ports, a claim read as a definition is read, a hunk's counts read only when it carries them, and a scorer that reads every rule it re-implements

2026-09-25. Branch `fix/diffgate-path-resolution` (pull request #161), head `21d17c14` before this round;
`origin/main` `2a6ce0a3` is already merged in (`08df3f2f`), and `main`'s `styxx/diffgate.py` there is
byte-identical to `98a5c368`'s (sha256 `9b620e00…`, LF), the file 7.48.0 ships and the scorer's baseline.

This note sits on top of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third to
sixth-pass notes. **None of them is edited.** Where one is wrong, the correction is here (section D). It is
committed alone, before any code of this round; its measurements were taken on the working tree whose
instrument, name table, port, pinned pairs, tests and scorer this round's commits carry.

---

## The merge bar

Held as the orchestrator states it:

1. The reproductions of #97, #121 and #101 are fixed.
2. **No claim reads worse on the branch than on `origin/main` on any door** — Python `gate_diff_text`,
   Python `gate_diff`, the JavaScript port: no new false VERIFIED, no new false CONTRADICTED, no new
   Python/JavaScript disagreement.
3. The scorer cannot admit a planted defect, **including one inside a new rule's own code**.

A wrong verdict `origin/main` gives identically is not a blocker; it is recorded (section F) and becomes a
follow-up issue.

---

## A. What the sixth-pass review found: three regressions against `main`

Round 6's regressions lens found three classes where the sixth-pass head read a claim worse than `main`.
Reproduced here on the head before this round (`21d17c14`), every one confirmed:

| | input | `main` raw / git / port | sixth-pass head | this round |
|---|---|---|---|---|
| 1 | "Added function fo²o." over `+def fo():` (and U+2082, U+2460, U+00BD; 923 such characters) | C / C / V | **V / V** / V | U / U / U |
| 2 | "Added 1 test." over a new `def test_a・b():` (U+30FB; also U+200C, U+200D, U+FF65, letters Unicode 15.1/16.0 added) | V / V / V | C / C / **V** | C / C / C |
| 3 | an over-declared hunk, then `--- lib/y.py`/`+++ lib/y.py`/`@@` ("2 files changed.") | V / – / V | **C** / – / **C** | V / – / V |

1. **The claim side read less than the definition side** (`diffgate.py` `_claimed_name`). W-2 took the
   identifier that starts where the template's `name` group starts; when the summary's name ran on past
   it with a character Python's `\w` holds but XID_Continue does not (905 No — superscripts, subscripts,
   circled numbers, fractions — and 18 L), the identifier stopped there and the claim was verified by any
   definition of the prefix. `main` read the whole name and matched nothing.
2. **Each port read identifiers from its runtime's Unicode** (`diffgate.js` `\p{XID_*}`; `str.isidentifier`
   in the Python). Python 3.12 is Unicode 15.0, Node 24 is 16.0. The sixth pass's note said every code point
   the two differ on was "assigned after 15.0"; four were not — U+200C, U+200D, U+30FB, U+FF65 were assigned
   long before and became XID_Continue in 15.1 (section D).
3. **W-1 trusted a hunk's counts even when the hunk did not carry them**, and its header-pair exception
   recognised the next file's `---`/`+++` pair only when a numeric `@@` followed it or it was written
   `a/`/`b/`. A GNU-style header (`--- lib/y.py`), followed by a bare `@@`, `@@ ... @@`, a blank line, no hunk
   header or timestamps, was read as a removed and an added line: the next file never entered the status
   map and its lines were filed under the previous file. A test moved between two such files read "Added 0
   tests." VERIFIED "(1 changed, not added: #101)" — the kind this branch exists to close.

---

## B. The changes

### X-1 (W-2, completed) — a claim is read as a definition is read

`_defined_name` refuses a line whose name runs on into a character no identifier holds. `_claimed_name`
now does the same, in both ports: when the character right after the claimed identifier is a word
character (`\w`, by the table below) that no identifier holds, the claim names no identifier and reads
**UNCHECKABLE** — "the claimed name runs past 'fo' into '²' (U+00B2), which no Python identifier holds;
no definition is read for it" — never a truncated name. A claimed identifier that **ends in a middle dot**
(U+00B7, or U+0387, the Greek semicolon — the only punctuation XID_Continue holds in Unicode 15.0) is read
the same way: prose writes one after a word, so where the name ends is not certain. A name printed in a
reason is quoted as `repr()` quotes an identifier (`_qname`), without asking the runtime which characters it
can print. Inside a name the dot is the name's (`col·leccio` is one identifier, as the sixth pass read it).

### X-2 — one pinned Unicode table, the same bytes in both ports

`styxx/_xid.py` carries one run-length table over every code point with three bits: opens an identifier
(XID_Start, or `_`), continues one (XID_Continue), is a word character (Python's `\w`: `str.isalnum()` or
`_`). 1,946 runs, written as 3,136 ASCII characters. `web/gate/gen_xid.py` generates it from **Unicode
15.0.0** — CPython 3.12's database, which it refuses to run without — and writes the same string, with its
version and sha256 (`8df68f217cca495ab8a38ced9096213aabac4cf23927068d61397d2c9074d4cb`), into the GENERATED
blocks of `styxx/_xid.py` and `web/gate/diffgate.js`. Both ports read names by it (`_identifier_at`,
`_identifierAt`); the port also builds Python's `\w` (`_PY_W`, used by the COMPAT patterns, `\b` and the
claim detail) from it instead of its engine's `\p{L}\p{N}`. Tests hold the two blocks equal, the hash
pinned in both, the port's decoding equal to the Python's, and — under a 15.0.0 Python — the generator
reproducing both blocks. `py_side.py` pins the table's hash beside the instrument's.

**The choice of version, and what it costs.** The package supports Python 3.9 to 3.12 (`requires-python
>=3.9`, classifiers and the CI matrix 3.9–3.12). Two candidate tables were measured on this round's name
grid (section E: 60 characters across categories × 7 shapes, 413 real repositories, 1,003 claims, three
doors, CPython of the running interpreter as the judge):

| table | under Python 3.12 | under Python 3.14 |
|---|---|---|
| **15.0.0 (chosen)** | **0** claims wrong where `main` was right; **0** new Python/port disagreements | 148 claim-door cells wrong where `main` was right (56 raw, 56 git, 36 port), 0 new disagreements |
| 16.0.0 | 48 cells wrong where `main` was right, 36 new Python/port disagreements | 0 and 0 |

15.0.0 is the version every supported Python accepts identifiers by at most, and the one the tests and CI
run (3.12); it reads no claim worse than `main` on 3.12. **Residual, stated:** (a) under Python 3.13 and
3.14, code using the characters Unicode 15.1 and 16.0 made identifier characters — U+200C, U+200D, U+30FB,
U+FF65 and the letters those versions assigned — reads as defining nothing (a false CONTRADICTED on "Added
1 test." over `def test_a・b():`, where `main` read VERIFIED); those Pythons are outside the supported range.
(b) Under 3.9 to 3.11 (Unicode 13.0, 14.0) the table accepts identifier characters 14.0 and 15.0 assigned
that those runtimes refuse — a line holding one is a syntax error there, and `main`'s reading of it is no
more right — and the Python's own regular expressions (the claim templates, COMPAT's patterns, `\b`) still
read the runtime's classes, as on `main`, so a letter 14.0 or 15.0 assigned can differ between those
Pythons and the port in a claim's `detail` or a COMPAT reason. 3.9 to 3.11 are not installed on this
machine; (b) is argued, not measured.

### X-3 (W-1, completed) — a hunk is read by its counts only when it carries what it declares

Each hunk is scanned before it is read (`_hunk_is_exact`, both ports): its counts are walked over the lines
that follow. The walk ends — and the hunk is **not exact** — at the end of the diff, at a line the counts do
not allow, at a `---`/`+++` pair followed by a line opening `@@` (a file header and its hunk), and at a
`--- ` line right after an added line (git, `diff -u` and difflib write each change's removed lines before
its added ones, so a `---` there is a header, not a removed line). When the counts close, the hunk is exact
only if what follows can end a hunk: the end of the diff, `diff --git`, a `---`/`+++` pair, a numeric hunk
header at or past this hunk's end on both sides with no blank line before it, a `-- ` signature, or a line
no hunk carries. **An exact hunk is read by its counts with no exception** — a `---`/`+++` line inside it is
content, a removed `-- a/x` beside an added `++ b/x` included (round-6 minor: the raw door and the port now
read it as the git door does). **Any other hunk is read exactly as `main` read it**, line by line. The
sixth pass's `_header_pair` exception is gone. git writes only exact hunks, so the git door is unchanged.

The cost, stated: a hand-written hunk whose counts, by coincidence, close exactly across the next file's
header pair (the pair not followed by `@@` nor preceded by an added line, and what follows able to end a
hunk) is read as content — which is how `git apply` reads it. The limit the sixth pass pinned (a removed
`-- a` and an added `++ b` as the last lines of a hunk right before the next `@@` read as a file header) is
unchanged: that hunk is not exact, and is read as `main` reads it.

### A-1 — an added `async def test_` abstains the test count

The round's randomised repositories found two claims (× three doors) where the branch read worse than
`main` for a reason no earlier round counted: `got` has never read `async def test_` (`main` neither), and
`main`'s "Added 0 tests." over a new async test beside a changed sync test read CONTRADICTED only because
it also counted the changed test; the #101 pairing removed that second miscount and left the other. Where a
file adds an `async def test_` its removed lines do not define (`_async_tests_added`, both ports), the
`tests_added` claim is now UNCHECKABLE: "diff adds 1 async test functions, which this template does not
count; claim says 0". It only abstains. The fix `main` needs — reading `async` on both sides — stays a
follow-up issue (section F).

---

## C. The scorer

The round-6 protocol lens showed that a defect planted inside a post-amendment rule's own code passed
every gate: the counterfactual reverts the defect with the rule, and only the rule's admission test could
refuse it. `path2_gates.py` gains three blocking checks.

- **G-C7, the oracles.** On every record, the scorer computes what each rule it re-implements reads, with
  its own code, and the repaired instrument must read the same: F-2 (`_diff_lines` = git's split), W-1
  (`_read_diff` = its own hunk walk, `hunk_exact`: the status map in diff order, the added lines, every
  file's sides), V-1/W-2 (`_defined_name` and `_test_name` on every added and removed line, both
  readings), F-3/R-1 (`_added_tests`), #121 (`_norm` on every path and every claimed path or prefix =
  `own_key`), #97 (`_find_path` = `own_find_path`, every path claim), and every `tests_added` and
  `symbol_added` claim's verdict and reason (`expected_tests`, `expected_symbol`: the pairing, `got`, the
  claimed name with its runs-past and middle-dot rules, A-1, BC-1 and BC-2). Names are read through the
  scorer's own decoder of the same table, which must hash to its pinned sha256 and, under a 15.0.0 Python,
  equal that database code point by code point. A declared claim (DECLARE-1) is not re-read: it runs through
  the code the sentence it renders runs through.
- **G-C1, extended.** The gate-level fields — `measured`, `why_unmeasured`, `uncovered_sentences`,
  `sentences_total` — must be the baseline's unless reverting F-2 or W-1 (the only rules that change what a
  diff parses to) gives them back on a diff where that rule can act; and each instrument's verdict (strict
  off) must be the one its own claims give.
- **G-C8, the git door.** A sample of records whose hunks are all exact and whose paths are safe to write is
  rebuilt as a two-commit repository in a temporary directory (`rebuild`, `GitDoor`; rename detection off,
  since the record says "deleted" and "created"). `gate_diff` is run by both instruments and scored as its
  own door — the counterfactual's reverts act through `gate_diff`, git's answers memoised — and the repaired
  git door must read exactly what the repaired raw door reads on git's bytes. Differential mode samples the
  leading `--git-sample` (150) rebuildable records in the order of their ids' sha256; corpus mode every PR
  whose id's sha256 is 0 modulo `--git-every` (150), up to `--git-sample` (300).
- **A-1's revert** is in the table (`_async_tests_added` counts nothing) and A-1 admits only a
  `tests_added` move to UNCHECKABLE; G-C3 is waived for it as for every post-amendment rule, and it can
  make no accusation.

**Planted-defect proofs** (in memory, scratch copies of `diffgate.py` only; each also scored clean beforehand,
0 violations): a U+FEFF dropped from any added line (inside W-1) → `G-C7_oracle:W-1_added_lines`;
`_defined_name` refusing a `:` after the name (inside W-2) → `G-C7_oracle:W-2_defined_name`; a line holding
U+2028 dropped by the split (inside F-2) → `G-C7_oracle:F-2_split`; the gate ignoring an `only_touches`
accusation → `G-C1_gate_verdict_not_from_its_claims`; the runs-past rule removed (inside W-2) →
`G-C7_oracle:symbol_added_claim`; the key stripping one leading segment only (inside #121) →
`G-C7_oracle:#121_key`; the suffix tier before the exact one (inside #97) → `G-C7_oracle:#97_tiers`;
`got` stripping its line before reading it (inside F-3) → `G-C7_oracle:F-3_got`; the async guard removed (inside A-1)
→ `G-C7_oracle:tests_added_claim`; A-1 accusing instead of abstaining → `G-C4_direction:tests_added:A-1`;
`sentences_total` off by one → `G-C1_gate_fields_differ`; a diff dropped by F-2's own code where F-2 cannot
act, moving `measured` → `G-C1_gate_fields_differ`; the git door handing the gate no sides →
`G-C8_git_door_differs_from_the_raw_door`; a name table whose hash was made to match after a run was changed
→ the scorer refuses to load. The round-6 reviewer's D3 (the same U+FEFF defect hidden inside a joint
#101+W-1 attribution) meets the same W-1 oracle, which reads every record whatever its attribution. The
sixth pass's pinned limit-12(b) probe (`net >= n` over a form-feed-led test, "an EXPECTED LIMIT") is now
refused (`G-C7_oracle:tests_added_claim`), and its test says so. Each of these is a committed test
(`test_x7_*`, `test_v3_limit_12_…`).

**What the scorer still cannot see.** V-4's `_parent_prefix` and `_could_lie_under` are reverted, not
re-read; COMPAT's reasons are compared, not re-derived (their inputs — the split, the sides — are re-read);
the table's data is guarded by its hash and, under 15.0.0, by the database, not by an independent source of
Unicode; `gate_diff`'s status is git's `--name-status`, which the git door reads through its own
`own_name_status` and does not re-derive.

---

## D. Corrections to earlier notes

1. **Sixth pass, section E, "Unicode versions": "every one assigned after 15.0".** False: U+200C, U+200D,
   U+30FB and U+FF65 were assigned before Unicode 15.0 and became XID_Continue in 15.1.
2. **Sixth pass, "The merge bar" and section E: "0 new wrong verdicts against `main`" on every door.** False
   on three counts — the three regressions of section A — and, by a fourth this round found, on the async
   interaction A-1 closes. The sixth pass's grids did not hold a claim that runs past its identifier against
   a definition of the prefix, did not vary the Unicode version, and held no over-declared hunk before a
   GNU-style header.
3. **Sixth pass, W-1, "The exception, and why it exists", and its stated cost.** Incomplete: the exception
   also read a GNU-style next-file header inside an over-declared hunk as content (section A, 3), and its
   `a/`/`b/` clause read a content pair `-- a/x`/`++ b/x` inside an exact git hunk as a header (the round-6
   minor; section F of the sixth pass named only the pair before an `@@`). Both are gone with X-3.
4. **Sixth pass, W-2: a name is read "with the language's own rule, `str.isidentifier`".** That rule is the
   runtime's; the ports ran on different runtimes. X-2.
5. **Sixth pass, section C, limit 12.** (a) is closed for every rule the scorer re-implements (G-C7); (b) is
   closed for `tests_added` and `symbol_added` claims, which G-C7 re-reads whole. The limit-12(b) test that
   pinned an admitted defect now expects the violation.
6. **Sixth pass, the scorer**, "compares claims only, and only through `gate_diff_text`" (round-6 protocol
   lens, minor): true at the sixth pass and not stated among its limits. G-C1 extended and G-C8.
7. **`web/gate/bookmarklet_ui.js`** said the port was "the file 7.48.0 ships" and "differential-tested
   (3,212 pairs, 0 disagreements)" — both true on `main`, false on this branch. It now names the PATH-2 file
   by its sha256 and the current differential (3,381 pairs, 0 disagreements).
8. **`web/gate/README.md` and the sixth pass's summary tied `py_side.py --installed` to 219 records.**
   `--installed` runs the installed 7.48.0 Python (`main`'s file) against this branch's port: the sixth pass
   measured that at 205, and this round at 213. 219 (227 now) is this branch's Python against `main`'s port.

---

## E. What was measured

All on this round's working tree (instrument sha256 `67fb1b75…`, LF; name table `8df68f21…`; port and
bookmarklet as committed after this note); corpora generated and deleted afterwards; every repository built in
a temporary directory and removed. `origin/main`'s and the sixth-pass head's `diffgate.py` and `diffgate.js`
were loaded beside the branch's in one process each. Doors: `gate_diff` on a real two-commit repository (git),
`gate_diff_text` on the bytes git printed (raw), the port on the same bytes. Truth: CPython's parser of the
running interpreter (`ast` per file; a symbol claim judged on the name as written — one that is not an
identifier cannot be defined, so VERIFIED is wrong there; one ending in a middle dot is left undecided). A
claim "reads worse" when the branch is wrong where `main` was right or abstained; a claim whose truth the
parser cannot give (a file that does not parse) was judged by hand, by class, and every such move is an
abstention or reads the unparseable file as defining nothing. (The sets below ran before two comments in
`diffgate.py` and three in `diffgate.js` were reworded; the Python's AST and the port's terser output are
identical before and after, and the differential, the pinned pairs, the tests and the scorer were rerun on the
final bytes.)

| set (Python 3.12 unless named) | inputs | claims per door | wrong on `main` raw / git / port | wrong on the branch | worse than `main` | new Python/port disagreements |
|---|---|---|---|---|---|---|
| rounds 2–5 evidence and the PREREG reproductions | 210 cases (189 repositories) | 262 / 228 / 258 | 30 / 30 / 31 | 0 | **0** | **0** (6 summary-side, all on `main` too) |
| round-6 evidence (the three blockers, the minor) | 14 (7 repositories) | 53 / 22 / 53 | 2 / 2 / 18 | 0 | **0** | **0** |
| `---`/`+++` content lines, over- and under-declared hunks, GNU and `a/`/`b/` headers | 27 (12 repositories, 15 hand-written) | 102 / 47 / 102 | 14 / 14 / 14 | 0 | **0** | **0** |
| Unicode skew characters (U+200C, U+200D, U+30FB, U+FF65, U+1C89, U+2EBF0) × 6 shapes | 36 repositories | 72 | 30 | 12, each `main`'s too | **0** | **0** (`main` 0) |
| name grid, this round's (60 characters across categories × 7 shapes) | 413 repositories | 948 / 948 / 949 | 296 / 296 / 460 | 45, each `main`'s too | **0** | **0** (1 summary-side, on `main` too) |
| the same grid under **Python 3.14** (Unicode 16.0 truth) | 413 repositories | 948 / 948 / 949 | 288 / 288 / 480 | 93 | **148** (56 / 56 / 36): the residual of section B | 0 |
| definition-line grid (round 6's: Python's `\s`, U+FEFF, marks, `Pc`, digits, 15.1/16.0 letters, ASCII) | 1,975 repositories | 4,424 | 242 / 242 / 345 | 36, each `main`'s too | **0** | **0** (`main` 271) |
| round 6's randomised repositories | 400 repositories | 2,488 | 107 / 107 / 96 | 4, each `main`'s too | **0** (was 2 × 3 doors: A-1) | **0** |
| randomised hand-written diffs with known intent (23 styles: GNU, timestamps, over-/under-declared counts, bare `@@`, blank separators, stripped context, `---`/`+++` content lines) | 1,497 diffs | 7,485 raw / 7,485 port | 639 / – / 639 | 366, each `main`'s too | **0** | **0** (`main` 127) |
| round 6's name grid: every code point below U+30000 that is `\w` but not an identifier, or the reverse | 3,133 code points, 6,266 diffs | 12,532 raw / 12,532 port | – | – | **0** (moves: 3,692 word-not-identifier claims C (raw) or V (port) → U; 8,836 identifier-not-word claims V → C, the claimed identifier is not the definition's; 4 middle-dot claims V → U) | **0** (`main` 3,692) |
| issue #128's eleven, both path spellings | 22 | 22 | – | – | 0 moves | 0 |

**The Unicode version, measured two ways** (the name grid above, three doors): the 15.0.0 table reads 0
claims worse than `main` under 3.12 and 148 claim-door cells worse under 3.14; a 16.0.0 table (generated the
same way under 3.14, in scratch) reads 48 worse and opens 36 Python/port disagreements under 3.12, and 0 and 0
under 3.14. 15.0.0 is the choice (section B).

**Differential** (`build_corpus.py` 176, `fuzz_corpus.py` 3,000, 205 pinned pairs): 3,381 pairs, 7,186 claims
(722 verified, 1,594 contradicted, 4,870 uncheckable), **0 disagreements**. `main` on both sides 34, the
sixth-pass head on both sides 3; `main`'s Python against this port 213 records (what `py_side.py --installed`
measures against 7.48.0), this Python against `main`'s port 227. Moves against `main`: 213 records in the Python,
227 in the port; against the sixth-pass head 11 and 14, every one a pinned pair this round added — **no
generated record moves**. `check_pairs.js`: 205 pinned pairs, 0; the minified bookmarklet in Node with the
browser stubbed: 205, 0. Bookmarklet: `bookmarklet.min.js` sha256 `03d1e1a1…`, 34,455 characters (29,038 at
the sixth pass; the table is 3,136 of the 5,417 added); `build_bookmarklet.py --check` (terser 5.46.0) matches.

**The scorer**, `path2_gates.py differential` on this tree: every gate but G-C0 passes (G-C0 fails because
the tree is uncommitted; the clean-tree run is in `web/gate/README.md`). 213 records moved; claims attributed to
#97 27, #121 35, #101 26, F-2 18, F-3 4, V-1 11, V-4 94, W-1 18, and to sets #101+A-1 2, #101+F-2+V-1+W-2 2,
#101+V-1 4, #101+V-1+W-2 4, #101+W-1 7, #121+V-4 2, F-2+V-1 4, F-3+V-1+W-2 9, V-1+W-2 4 (4 joint). New accusations
20: `only_touches` 4 through the amendment's dotted-prefix exception, and 16 explained only by post-amendment
rules with G-C3 waived (F-2 1, F-2+V-1 1, F-3 2, F-3+V-1+W-2 5, V-1 `symbol_added` 7; the sixth pass waived 10)
— the six added are this round's pinned pairs over definitions CPython 3.12 refuses. `compat2_candidate` flips 8,
each one rule alone (F-2 six, W-1 one each way); gate-level fields moved on 2 records, both given back by F-2 on
diffs whose splits differ; 4 F-4 withdrawals; 30 records whose splits differ, 22 whose parses differ. **G-C7:
0 oracle violations** over the 3,381 records (the table checked against this Python's 15.0.0 database). **G-C8:**
of the leading 500 records in sha256 order, 150 rebuilt and scored through `gate_diff`, 350 not rebuildable; 15
moved, all attributed; 0 violations. Scorer `b69f3fe5…`, harness `75bfbc39…`, repaired `67fb1b75…`.

**Mutation.** Python: only `styxx/diffgate.py` copied (to scratch `mut7/`, LF) and loaded as `styxx.diffgate`
by a pytest plugin; the fourth pass's ten core modules, the three pin tests deselected; control 563 passed.
**27 mutants, 26 killed** (two of them after the tests they exposed were added: the runtime's
`str.isidentifier`, a blank line before a hunk header): X-1's runs-past rule removed or asking the wrong bit, the middle-dot rule removed or
losing U+0387, the reason ignored, the code point printed in lower case; X-2's identifier opening on the
continue bit, continuing on the word bit, or asking the runtime's `str.isidentifier` again (killed by a test
that reads the identifier through the table's functions: on 3.12 the runtime and the table agree on every code
point); X-3's scan always or never exact, each stop removed (the header and its hunk; a `--- ` line after an
added line; the added-line flag never reset; the end of the diff before the counts close), each boundary
widened or narrowed (a body line, a hunk going back, a blank line before a hunk header, a no-newline marker, a
file header, a signature); A-1's guard removed, counting sync tests, ignoring the removed side, or reading an
`A` file's removed lines. The one survivor computes `hit` beside a set reason, which is never read: equivalent.
Port: **30 in-memory mutants** of `diffgate.js`, held to the 205 pinned pairs and 237 extra records (detail and
gate verdict compared; the extras gained an astral identifier, a context line between an added and a
`--- ` line, and an over-declared hunk at the end of the diff when three mutants survived an earlier run): **27 killed**;
the 3 survivors are equivalent — `hit` beside a set reason; `pyRepr` for
`_qname` (on this engine every name the table reads prints unescaped); an identifier opening on the continue
bit (a claimed name opens with `[A-Za-z_]`, and a defined name opening with a digit or a mark neither equals it
nor opens with `test_`). Scorer: **34 mutants** of `path2_gates.py` in memory (the sixth pass's fourteen,
re-read against this round's text, and twenty for this round's code: each G-C7 oracle off, the scorer's own
walk, claimed-name and count readings each missing a rule, G-C1's verdict and field checks, G-C8's doors check
and door, A-1's admission and revert, the table's database check), through the `test_v3_*` and `test_x7_*`
tests: **34 killed**, three of them after the tests they exposed were added (a gate-field move by F-2's code on a
diff F-2 cannot act on; A-1 accusing; a table the database does not read).

**Tests.** The 30 modules that import `styxx.diffgate` or read `web/gate`, and `tests/test_ledger.py`
(py -3.12): 1,612 passed, 1 skipped, 6 xfailed, 2 failed — `tests/test_gitlab_job.py`'s two job tests, which
fail on this machine whatever the code: `bash` resolves to WSL's `bash.exe`, which cannot start, so the job's script never runs (section F).

---

## F. What is still not repaired

Carried from the sixth pass: amendment limits 1, 2 and 4 and the docstring half of limit 3; COMPAT reading
Python's `\s` in all five languages; a lone `\r` inside a line; `..docs`; F-4's abstentions; V-4 reading
`docs/.` plus a period as a parent; `path2_differential_gates.json` uncommitted (amendment limit 7, owed to
the RESULT); the corpus gates not run on the real shelf; #128's modes 2, 3, 5 and 6; identifiers compared as
text, not under NFKC; a line CPython refuses read as a definition when an ASCII non-name character follows
the name (on the removed side too, where the #101 pairing reads it); the port's claim templates reading the
description with JavaScript's classes; a declared non-ASCII `adds_symbol` MALFORMED in the port; an on-tree
prefix spelled with `.`, `..` or `...` segments; `+++ /dev/null` with no `---` line raising in both ports.
An added `async def` is still read by neither template (A-1 only abstains the test count beside one).

New this round: the Unicode residual of X-2 (section B) and the coincidence X-3 leaves (section B). Found in
passing, identical on `main` and outside the instrument: `tests/test_gitlab_job.py` runs the job's script
with `subprocess.run(["bash", ...])`, which on a Windows machine with WSL installed starts `System32\bash.exe`
before Git's bash and fails without running the script; and `papers/build_ledger.py` writes `LEDGER.md` with
the platform's line ending, so an LF checkout on Windows is left modified by `tests/test_ledger.py` (the
round-6 protocol lens's last minor).

The defects above that exist identically on `origin/main` are the follow-up issues this round files.

---

## Protocol changes the operator is asked to accept

- **X-1** (W-2's code): a claimed name that runs on past the identifier, or ends in a middle dot, is
  UNCHECKABLE. **X-2**: names, and the port's `\w`, are read by one pinned table (Unicode 15.0.0) in both
  ports. **X-3** (W-1's code): a hunk is read by its counts only when exact, else as `main` read it; the
  header-pair exception is gone. W-1's and W-2's reverts still give back the fifth pass, so the scorer
  attributes this round's moves to them.
- **A-1** is a new post-amendment rule with its own revert; it admits a `tests_added` move to UNCHECKABLE
  only, and G-C3 is waived for it.
- **G-C7, G-C1 extended and G-C8** are new blocking gates of `path2_gates.py`, in both modes.
