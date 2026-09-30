# NOTE — PATH-2, eighth pass: where the branch cannot be sure it reads a claim at least as well as `main`, it abstains

2026-09-27. Branch `fix/diffgate-path-resolution` (pull request #161), head `22560c62` before this round;
`origin/main` `2a6ce0a3` is merged in, and `main`'s `styxx/diffgate.py` there is the file 7.48.0 ships and the
scorer's baseline (sha256 `9b620e00…`, LF).

This note sits on top of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third to
seventh-pass notes. **None of them is edited.** Where one is wrong, the correction is here (section D).

**How this round was worked, disclosed.** An earlier run of this round was interrupted by a usage limit after
about an hour. It left roughly 2,100 uncommitted changed lines in seven files and a new
`web/gate/xid_versions.json`, and no note: **this round's code was drafted before this note**, the reverse of
the order every earlier round kept. The draft was backed up as a patch outside the worktree, re-read in full
and not trusted: part of it is kept, part is replaced (section B names which), and the round's measurements
(section E) were taken on the working tree after that. This note is committed alone, before any of that
code is; the code, pinned pairs, tests, scorer and documents follow in their own commits.

---

## The merge bar, and the principle this round adds

Held as the orchestrator states it:

1. The reproductions of #97, #121 and #101 are fixed.
2. **No claim reads worse on the branch than on `origin/main` on any door** — Python `gate_diff_text`,
   Python `gate_diff`, the JavaScript port — under any supported Python (3.9 to 3.14): no new false VERIFIED,
   no new false CONTRADICTED, no new Python/JavaScript disagreement. An abstention (UNCHECKABLE) where `main`
   gave a verdict is not a false verdict, provided both ports abstain identically.
3. The scorer cannot admit a planted defect, including one inside a new rule's own code, or one reachable
   only through `gate_diff` or a declared claim.

**The convergence principle.** Wherever the branch's new machinery cannot be sure it reads a claim at least as
well as `main`, it ABSTAINS — UNCHECKABLE, with a reason that names the uncertainty — identically in both
ports. Seven rounds each found a place where a repair was right about the line it read and wrong about the
claim, because `main`'s answer there had been right by two errors that cancelled, and the repair removed one.
The rules of this round are that principle applied: none of them adds a verdict `main` could not give except
Y-4, which repairs a defect `main` has too.

---

## A. What the seventh-pass review found

Both round-7 reviewers asked for repairs before a merge. Reproduced on `22560c62`, every finding confirmed:

| | finding | severity | vs `main` |
|---|---|---|---|
| R0.0 | G-C7 skipped every declared (DECLARE-1) claim, so a defect planted inside a rule's own code whose trigger came through a ```` ```styxx ```` block (a prefix match in `_symbol_hit`; the runs-past rule removed; limit 12(b); A-1 ignoring the removed side) was admitted | blocker | scorer |
| R0.1 | V-4 and F-4 had no oracle and COMPAT's C-2 scaffold reading none either: a planted `_parent_prefix` that reads `src/...` as a parent, a `_could_lie_under` that always answers yes, and a C-2 reading on the dotted key were admitted | major | scorer |
| R0.2 | G-C8 scored a sample (150 records in differential mode, 1 PR in 150 in corpus mode), so a defect reachable only through `gate_diff` was admitted unless its record was sampled | major | scorer |
| R0.3 = R1.2 | under Python 3.13 and 3.14 the pinned Unicode 15.0.0 table read worse than `main`, with false VERIFIEDs the seventh note did not disclose: a claimed name running on into a letter Unicode 15.1/16.0 assigned was truncated to its prefix and verified on any definition of it ("Added function fo\<U+105C0\>." over `def fo():`), and "Added 0 tests." verified over tests whose names hold such a character | major / blocker | **regression** |
| R0.4 | the scorer's W-1 exactness rule and the ports' differed (the scorer reset its flag on a removed line, the ports only on a context line), so the unplanted branch raised `G-C7_oracle:W-1_sides` on a hand-written hunk | minor | scorer |
| R0.5 | the scorer trusted the table's self-declared sha256 and ran its database check only under a 15.0.0 Python | minor | scorer |
| R1.0 | in a hunk that is not exact, a removed `-- users` prints as `--- users` and is read as a file header, as `main` read it: the file's later lines leave its sides but stay in the added blob, the #101 pairing pairs only part of what changed, and `net` lands above the true count — "Added 1 test." VERIFIED over two changed tests, where `main` said CONTRADICTED | blocker | **regression** |
| R1.1 | a U+FEFF opening line 1 of a file was dropped only inside an exact hunk; in any other hunk a BOM-led definition on line 1 read as nothing — new false verdicts on the port (which on `main` read U+FEFF through JavaScript's `\s`) and, through the pairing, on the raw door | blocker | **regression** |
| R1.3 | where `main`'s two defects gave the right file count between them, the branch fixed one: GNU `+++ /dev/null<TAB>timestamp` headers collapsing two deletions into one key, beside a `+++ x` phantom an exact hunk no longer makes; dotfile twins beside a phantom from a non-exact hunk | major | **regression** |

---

## B. The changes

### Kept from the draft, re-read

**Y-1 — the file list.** A `---` or `+++` line read as a header after lines no hunk count placed (a hunk
that is not exact, a bare `@@`, lines under no hunk header) may be content — a SQL or Lua `-- ` comment, a
`++ ` line — or a header. Such a header is taken as one, as `main` took it, only when the pair has a header's
shape: `--- X`, `+++ X` (or `/dev/null` on either side), then a line opening `@@`, the paths compared with a
GNU timestamp cut and `a/`/`b/` dropped. Otherwise the reading records that its file list is not certain, and
`files_changed_count`, `only_touches` and every path claim read UNCHECKABLE: "the diff's file list is not
certain: a `---` or `+++` line after lines no hunk count holds may be content (a SQL or Lua comment, a `++`
line) or a file header". Two header paths that differ only in case are one key (on `main` too); where the
diff holds such a pair, the same claims abstain ("two header paths that differ only in case are one key").
And where GNU names a changed file outside any header pair — `Binary files X and Y differ` with no
`diff --git` header before it, `Files X and Y differ`, `Symbolic links X and Y differ`, `Only in DIR: NAME`,
`File X is a … while file Y is a …` — neither parser counts the file (on `main` too), so the same claims
abstain ("a line names a changed file no header pair counts"). The git door's file list is git's own
`--name-status`, so there only the case collision applies. A `diff --git` line ends the stretch. git writes
only exact hunks and puts a `Binary files` line under its `diff --git` header, so on git's bytes Y-1 acts only
where a lone `\r` inside a line splits it — a place the git door's `--name-status` answers; the scorer's G-C8
excuses exactly that (below).

**Y-2 — U+FEFF.** Outside the counts, a U+FEFF is dropped from a line the diff shows is line 1 of its side —
the opening added line of a created file, the opening removed line of a deleted one, the opening line of a
side after a hunk header that starts at 1 — as CPython reads it and as `main`'s port read it. A definition a
U+FEFF opens anywhere else (a line the diff does not show is line 1, where CPython refuses the character) makes
a claim that could read it abstain: `symbol_added` for that name, `tests_added` for a test.

**Y-3 — the Pythons this package supports.** It installs on 3.9 to 3.14 (`requires-python >=3.9`, no cap), and
each reads identifiers by its own Unicode: 13.0.0 (3.9, 3.10), 14.0.0 (3.11), 15.0.0 (3.12), 15.1.0 (3.13),
16.0.0 (3.14). `web/gate/gen_xid.py` now also emits SKEW, the code points one of those versions gives other
bits (opens, continues, `\w`) than the 15.0.0 table does — 10,153 of them — into the same GENERATED blocks of
`styxx/_xid.py` and `web/gate/diffgate.js`, with its sha256 (`0b7134fd…`). A claimed name that meets one, or
whose next character is one, reads UNCHECKABLE ("the claimed name 'fo' meets U+105C0, which the Pythons this
package supports (Unicode 13.0 to 16.0) read differently; no definition is read for it"); a test definition
whose name, read as widely as any of those Pythons reads it, holds one abstains the test count. Its sources
are committed in `web/gate/xid_versions.json`: 16.0.0 is CPython 3.14's own masks (`--dump-masks`); 15.1.0 is
not installed here and is held by 16.0.0's difference (checked for the code points 15.1 changed: CJK
Extension I and U+200C, U+200D, U+30FB, U+FF65); 13.0.0 and 14.0.0 are the table's masks where Unicode's Age
property (Perl 5.38.2's UCD) places the code point at or before that version; and a margin of 28 code points
whose general category, read as the three bits, moved between Unicode 3.2 (in every CPython) and 15.0.
`gen_xid.py --check-sources` re-derives the file from the tools and must match it.

A **symbol** definition whose name runs on into a skew code point is not abstained on: the definition reading
already refuses a name that runs into a character the table does not hold, so such a line defines nothing
for the prefix — which every Python that accepts the line agrees with (the name there is longer), and the
claim that names the longer name meets the skew set and abstains.

**Y-4 — `/dev/null` with a GNU timestamp** (a repair of a defect `main` has too). GNU diff writes `+++
/dev/null<TAB>2024-…`; neither parser recognised it, so a deletion was keyed "dev/null<TAB>2024-…" (two
deletions with one timestamp shared that key) and `--- /dev/null<TAB>…` read a created file as modified.
Both parsers now recognise `/dev/null` followed by a TAB. Every other header path keeps its timestamp,
exactly as `main` keys it: cutting it everywhere would let the test and symbol templates read GNU diffs whose
keys `main`'s BC-1 read as holding no Python file (a key ending in a timestamp does not end in `.py`), giving
verdicts `main` could not — a trade this round does not make. That GNU keys keep the timestamp is a defect
`main` has, and a follow-up issue.

### Replaced: the draft's list of shapes, by a rule

The draft abstained on `tests_added` beside the shapes it could see: a definition line inside a triple-quoted
string the diff's own lines open, a definition name after a backslash, a test name NFKC changes, an added
`async def test_X` beside a changed sync `test_X`, and a per-file reading that lost lines ("sides"). Re-read,
that list could not be complete, and six probes built this round show where:

| probe | `main` git / raw / port | seventh-pass head and the draft | this round |
|---|---|---|---|
| a `def test_doc():` in `README.md` beside a changed test; "Added 1 test." (truth: 0 added) | C / C / C | **V / V / V** | U / U / U |
| a `def test_s():` added inside a string whose opening quote is in unchanged lines, beside a changed test; "Added 1 test." | C / C / C | **V / V / V** | U / U / U |
| a created test file whose line 1 is `<U+FEFF>def test_new():`, beside a string holding `def test_str():`; "Added 1 test." (truth: 1) | V / V / C | **C / C** / C | U / U / U |
| the same line-1 test beside a `def test_doc():` line in a markdown file; "Added 1 test." | V / V / C | **C / C** / C | U / U / U |
| a GNU `Binary files a/docs/img.png and b/docs/img.png differ` line beside an exact hunk adding `++ plus`; "2 files changed." (truth: 2) | – / V / V | – / **C** / **C** | – / U / U |
| the same diff; "Only touches src/." (truth: `docs/img.png` is outside) | – / C / C | – / **V** / **V** | – / U / U |

(Probes one to four were built as real repositories; five and six are hand-written GNU output, which has no git
door.) Probes one and two are the #101 pairing: `got` reads a line Python does not define (in a file that is not
Python; in a string opened where the diff does not show it) — `main` did so too, and was right only because
it also counted the changed test. Probes three and four are older: the sixth pass's U+FEFF reading at line 1
(W-1) on the git door and the raw door, which counts the line-1 test CPython counts and `main`'s Python did not,
beside a line `got` counts and Python does not. Probes five and six are W-1's exact hunks (sixth pass): `main` read the
`++ plus` content line as a phantom file header, which made up for the binary file it could not count. A string
opened outside the hunk is invisible to any reading of the diff, so no list of shapes can close the kind
probes one and two show. The rule that does:

**Y-5 — the #101 pairing withdraws; it does not verify.** Where a changed test is paired away (`chg > 0`),
a claimed count equal to what is left (`net`) reads UNCHECKABLE — "diff adds 1 test functions and changes 1,
claim says 1; a count left after pairing changed tests away is not verified, since a line this template reads
may be one Python does not define (#101)" — never VERIFIED. With it, the pairing gives either `main`'s verdict
or an abstention: a count inside `[net, got]` abstained already (PREREG), `net` itself now abstains, and a count
outside the interval is one `main` contradicts too (`got` is not the claim). This replaces the PREREG's table row
"`net = n` → VERIFIED" (a protocol change, below); the #101 reproductions are unchanged (they read UNCHECKABLE
through the interval row). The symbol pairing already only withdraws a VERIFIED.

**Y-2, extended.** Any added test definition a U+FEFF opens — **line 1 included**, where the parse drops it
and CPython reads it — abstains the test count: "an added test definition opens with U+FEFF, which main's
Python counted as no test and its port as one". `main`'s two ports counted the line apart, so a count
elsewhere may have balanced either. A symbol claim keeps the line-1 reading: whether a name is defined is not
a count, and nothing can balance it.

The draft's string, backslash, NFKC and async-beside detectors, and its "sides" notes (orphan and prefix-less
definition lines), are removed: under Y-5 each could only turn a verdict `main` gives into an abstention.
A-1 (seventh pass) is kept as committed.

---

## C. The scorer

`papers/closed-model-frontier/path2_gates.py`:

- **G-C7 reads declared claims** (R0.0): a declared symbol claim through its canonical sentence ("Adds function
  X."), a declared count through its detail. **G-C7 reads every file-list claim** (R0.1): `expected_count`,
  `expected_path` and `expected_only_touches` re-derive each verdict and reason with the scorer's own code —
  PATH-1's containment and path shape (the extension list read from `path1_extensions.txt`), BC-2's second
  prefix, C-3's dot miss, R-3's off-tree key, V-4's written parent (`own_parent`) and F-4's could-lie-under
  (`own_could_hold`) — and `compat_violations` re-derives COMPAT's C-2 languages and surface flags on the
  undotted key. It also reads the eighth pass's rules: `own_read` returns the reading's notes (Y-1's file
  list, Y-2's line-1 test) and they must equal `_diff_notes`; `own_bom_hidden`, `own_skew_test` and the
  claimed name's skew rule are written out; `expected_tests` carries Y-2, Y-3 and Y-5.
- **G-C8 on every record** (R0.2). Differential mode tries every record in the order of their ids' sha256; a
  `--git-sample` smoke run fails the gate (`G-C8_not_every_record_tried`). The counts are part of the verdict:
  every record tried is scored, not rebuildable or rebuilt to no change (`G-C8_records_unaccounted`), none
  failed to rebuild (`G-C8_rebuild_failed`), at least one was scored (`G-C8_nothing_scored`). Corpus mode
  scores every PR with a `tests_added` or `symbol_added` claim — `gate_diff` feeds exactly those through its own
  blob and sides — and 1 in 25 of the others by sha256(id) (it was 1 in 150), up to 3,000 scored. On a record
  whose raw door abstains on a file-list claim by Y-1 and whose git door answers it from `--name-status`, the
  two doors may differ on that claim alone; G-C7 reads the git door's claim from git's list.
- **One W-1 exactness rule** (R0.4): the scorer's `hunk_exact` now implements the ports' rule — an added line
  opens a stretch only a context line closes; a `--- ` line anywhere in it ends the walk. The seventh note's
  words ("a `--- ` line right after an added line") were the scorer's rule, not the ports' (section D).
- **The table** (R0.5): the table's and the skew set's sha256s are pinned in the scorer as literals; the scorer
  refuses to run on a Python whose Unicode is not 15.0.0 (the table is checked there against the database,
  code point by code point); and it re-derives the skew set from `web/gate/xid_versions.json`, the table and
  `unicodedata.ucd_3_2_0` with its own code.
- **The eighth pass's rules** have reverts (Y-1 the file list always sure; Y-2 no U+FEFF dropped outside the
  counts, none read behind an indent, none noted before a test at line 1; Y-3 an empty skew set; Y-4
  `/dev/null` whole again; Y-5 the pairing verifies `net` again) and admissions: Y-1, Y-3 and Y-5 only abstain
  (Y-1 on the file-list kinds, Y-3 on the definition kinds, Y-5 on `tests_added`); Y-2 moves the definition kinds
  on a diff holding a U+FEFF-led `+`/`-` line; Y-4 moves any kind on a diff naming `/dev/null` followed by a
  TAB. G-C3 is waived for each as for every post-amendment rule. When a claim's repaired verdict is an
  eighth-pass abstention and that rule is in the attribution and admits it, a table rule beside it is not
  asked its DIRECTION test (#121's keys may shape what the claim would have read) — its other conditions are,
  and G-C7 re-reads the claim either way; such moves are counted under
  `attribution.table_direction_left_to_an_abstention`.

**Planted-defect proofs**, each a committed test (`test_x8_*`, and the seventh pass's `test_x7_*` re-run), each
record scored clean beforehand and refused once the defect is planted in a scratch copy of `diffgate.py`:
- declared-only triggers (R0.0): `_symbol_hit` matching a name's prefix over `adds_symbol: foo` and `def
  foobar()`; the runs-past rule removed over `adds_symbol: fo²o`; limit 12(b) (`net >= n`) over `tests_added: 1`
  and a form-feed-led test; A-1 ignoring the removed side over `tests_added: 0` — each
  `G-C7_oracle:…_claim`;
- V-4 reading any dots-only segment as a parent ("Only touches src/..."), F-4 answering "could" for every
  path ("Only touches src/ and ../docs/."), C-2 reading the scaffold on the dotted key (a removal under
  `.docs/`) — `G-C7_oracle:only_touches_claim`, `G-C7_oracle:C-2_compat_surface`;
- inside each eighth-pass rule: every pair given a header's shape and a case collision not noted (Y-1), a
  U+FEFF dropped outside the counts wherever it stands, a line-1 U+FEFF test not noted and the indent reading
  dropping nothing (Y-2), the skew set read as empty and a test name's skew not doubted (Y-3), `/dev/null` with a
  TAB not recognised (Y-4), the pairing verifying `net` again and accusing instead of abstaining (Y-5);
- each rule's admission: Y-1 accusing a count, Y-2 abstaining where no U+FEFF stands, Y-4 reading `/dev/nullx`,
  Y-5 contradicting where no test is paired — each `G-C4_direction:<kind>:<rule>`;
- through `gate_diff` alone: the git door's added blob dropping tab-led lines — refused in differential mode with
  the default parameters (every record tried), on a synthetic shelf in corpus mode with the default sampling
  (the PR carries a `tests_added` claim and is not in the 1-in-25 sample), and a smoke run that stops early fails
  the gate; a door that fails to rebuild, a record the counts do not account for, and a door that scored nothing
  each fail it;
- the tables: a name table whose block pin was made to match (refused by the literal pin, and with that pin
  rewritten, by the database), a skew set with one run changed (refused by its pin, and with it rewritten, by
  the re-derivation), a Python whose Unicode is not 15.0.0 (refused, at import too); a path claim read otherwise
  than the scorer reads it (`G-C7_oracle:file_touched_claim`); the unplanted round-7 W-1 shape now scores clean.

---

## D. Corrections to earlier notes

1. **Seventh pass, section B, the residual of the 15.0.0 table.** It said code using the characters Unicode
   15.1 and 16.0 made identifier characters "reads as defining nothing (a false CONTRADICTED …)". Incomplete:
   under 3.13 and 3.14 it also gave false VERIFIEDs — a claimed name truncated at such a letter and verified on a
   definition of its prefix, and "Added 0 tests." over tests whose names hold one (round 7 measured 275 claims per
   Python door and 207 per port door worse than `main` under 3.14 on its name grid, 114 of them false
   VERIFIEDs). Y-3 closes it: those claims now abstain on every Python.
2. **Seventh pass, X-3**: "a `--- ` line right after an added line" ends the walk. The ports end it at a `--- `
   line anywhere after an added line with no context line between (a removed line does not close the
   stretch). The scorer implemented the note's words; now it implements the ports'.
3. **Seventh pass, section E**: "1,497 hand-written diffs … 0 worse than `main`". False: round 7's
   regressions reviewer found 16 raw-door and 43 port claims worse across three randomised seeds and 32 and 72
   more on a 115-case hunk grid (R1.0, R1.1, R1.3).
4. **Seventh pass, section C**: "A declared claim (DECLARE-1) is not re-read: it runs through the code the
   sentence it renders runs through." The code is the same; the oracle fired only on records carrying a prose
   trigger, so a defect only a declared claim triggered was admitted (R0.0).
5. **Sixth and seventh passes, "0 worse" on every door** for W-1's line-1 U+FEFF reading. False beside a line
   `got` counts and Python does not (section B's probes three and four): on the git door and the raw door the
   repair counted the line-1 test `main`'s Python did not, and `main`'s Python had been right by a second
   miscount. Y-2's extension closes it.
6. **PREREG, the #101 table, "`net = n` → VERIFIED: in a real `base..head` diff a name in the removed lines
   of a file existed at base, so `net` is the count of definitions new to that file."** The argument holds
   for the pairing and not for `got`, which reads a `def test_` line in a markdown file or inside a string as
   a test; `main`'s count was then sometimes right only by the changed test the pairing removes. Y-5.

---

## E. What was measured

All on this round's working tree: instrument `c425c34f…` (LF), table `8df68f21…`, skew set `0b7134fd…`. The
instrument committed after this note, `d7c298d4…`, differs from it only by renamed local names and comments;
over the 3,413 records of the port corpus as it then stood, both read every record byte for byte alike, in both
ports. `origin/main`'s `diffgate.py` and `diffgate.js` were loaded beside the branch's in one process each.
Doors: `gate_diff` on a real two-commit repository built by `git fast-import` (git), `gate_diff_text` on the
bytes git printed (raw), the port on the same bytes, and for hand-written diffs the raw door and the port on the
text. Truth: CPython's parser of the running interpreter over the base and head
files (a claimed name that is not an identifier cannot be defined; one ending in a middle dot is left undecided;
a count over a diff git renders with a rename, where git's pairing is the question, is left undecided). A claim
reads worse when the branch is wrong where `main` was right or abstained. Every set ran under **Python 3.12 and
3.14**; corpora and repositories were generated in scratch and deleted.

| set | inputs | claims per door, git / raw / port on git's bytes; raw / port hand-written (3.12) | wrong on `main` (3.12; 3.14) | worse than `main`, 3.12 and 3.14 | new Python/port disagreements, 3.12 and 3.14 (`main`'s own) |
|---|---|---|---|---|---|
| rounds 2–6 evidence and the PREREG reproductions | 177 (164 repositories) | 223 / 223 / 222; 269 / 268 | 43/47/59; 47/59 — 53/57/71; 57/71 | **0**, **0** | **0**, **0** (140; 148) |
| round 7's evidence: both reviewers' reproductions | 31 (20 repositories) | 32 / 32 / 32; 64 / 64 | 9/12/11; 19/20 — 9/12/14; 19/23 | **0**, **0** | **0**, **0** (13; 19) |
| section B's probes one to four | 4 repositories | 12 / 12 / 12 | 4/4/8 — the same | **0**, **0** | **0**, **0** (2; 2) |
| GNU lines naming a file no header pair counts, beside a `++` content line (probes five and six, and five more) | 7 hand-written | –; 49 / 49 | 11/11 — the same | **0**, **0** | **0**, **0** (0; 0) |
| round 7's hunk grid: bare, elided, over- and under-declared hunks × SQL `-- ` lines, U+FEFF, changed and new tests | 115 repositories | 625 each; 625 / 625 | 295 on every door — the same | **0**, **0** | **0**, **0** (100; 100) |
| the name grid: 102 code points (letters, marks, `Pc`, digits, NFKC-changing and format characters; 8 from Unicode 14.0, 11 from 15.0, 2 from 15.1, 18 from 16.0) × 10 shapes | 1,020 repositories | 2,448 each; 2,448 / 2,448 | 351/351/768; 351/768 — 474/474/1,044; 474/1,044 | **0**, **0** | **0**, **0** (784; 1,056) |
| the reviewers' randomised generator, seeds 7107 and 31337 | 800 repositories, 3,000 hand-written | 8,141 each; 38,857 / 38,857 | 492/658/707; 2,773/3,011 — the same | **0**, **0** | **0**, **0** (312; 312) |
| this round's generator (strings, markdown files and backslashes holding `def test_`, NFKC names, skew letters, lone CRs, case and dotfile twins, U+FEFF), seeds 9009 and 4242 | 1,000 repositories, 3,000 hand-written | 10,271 each; 41,431 / 41,431 | 899/1,088/1,118; 3,975/4,101 — 1,104/1,302/1,361; 4,528/4,712 | **0**, **0** | **0**, **0** (412; 1,650) |

In all, 232,740 claim-door cells under 3.12 and 235,316 under 3.14, over 7,800 randomised diffs among other sets
(1,800 repositories and 6,000 hand-written in git, `a/`/`b/`, bare and GNU styles). Where the branch is wrong, `main`
is wrong on the same claim. Of the moves whose truth was left undecided, the ones the branch does not abstain on
were judged by class, all of them: a definition line CPython refuses read as defining nothing (U+000B, U+2028 or
U+2029 as indentation; a name running into a character the table does not hold, U+200C, U+30FB, U+FF65 and the
letters Unicode 15.1 and 16.0 assigned) — 72 and 42 on the evidence, 1,878 and 1,418 on the name grid, 7 on this
round's generator under 3.12 — and a file count over a diff git renders with a rename, where the branch reads
the files the hand-written diff lists and `main` counted a phantom from a `+++ x` content line — 8 on the
reviewers' generator, 2 on this round's under 3.14.

**The port differential** (`build_corpus.py` 176, `fuzz_corpus.py` 3,000, 240 pinned pairs), on the committed
instrument: 3,416 pairs, 7,259 claims (715 verified, 1,595 contradicted, 4,949 uncheckable), **0 disagreements**;
`main` on both sides 41, the seventh-pass head on both sides 0; `main`'s Python against this port 248 records
(what `py_side.py --installed` measures against 7.48.0), this Python against `main`'s port 259; against the
seventh-pass head 52 records move, every one a pinned pair this pass added or re-pinned — no generated record
moves. `check_pairs.js`: 240, 0; the minified bookmarklet in Node with the browser stubbed: 240, 0. Pinned
pairs: 35 added, 24 re-pinned (Y-5 thirteen, Y-2 four, Y-1 four, Y-3 three), each record saying so.
Bookmarklet: `bookmarklet.min.js` sha256 `caf3682b…`, 39,350 characters (34,455 at the seventh pass; the skew
set is 440 of the 4,895 added); `build_bookmarklet.py --check` (terser 5.46.0) matches. `gen_xid.py --check`
matches under 3.12 and refuses under 3.14; `gen_xid.py --check-sources` (CPython 3.14.2, Perl 5.38.2) re-derives
`xid_versions.json` byte for byte.

**The scorer**, `path2_gates.py differential` on this tree: every gate but G-C0 passes (G-C0 fails because the
tree is uncommitted; the clean-tree run is in `web/gate/README.md`). 248 records moved; 338 claims attributed —
#97 27, #121 28, #101 51, F-2 16, F-3 4, V-1 13, V-4 94, W-1 15, Y-1 16, Y-2 7, and to sets #101+A-1 2,
#101+F-2+V-1+W-2 2, #101+V-1 4, #101+V-1+W-2 4, #101+W-1 4, #101+Y-2 3, #121+V-4 2, #121+Y-1 10, F-2+V-1 4,
F-2+Y-1 2, F-3+V-1+W-2 6, F-3+Y-3 5, V-1+W-2 4, V-1+W-2+Y-3 1, W-1+Y-1 6, W-1+Y-2 8 (38 joint). New accusations
16: `only_touches` 4 through the amendment's dotted-prefix exception and 12 explained only by post-amendment
rules, G-C3 waived (F-2 1, F-2+V-1 1, F-3 2, F-3+V-1+W-2 3, V-1 `symbol_added` 5). `compat2_candidate` flips 8,
each one rule alone; the gate-level fields moved on 2 records, both given back by F-2; 4 F-4 withdrawals; 8
claims whose table direction was left to an eighth-pass or A-1 abstention (Y-1 3, Y-2 3, A-1 2).
**G-C7**: 0 oracle violations over the 3,416 records, declared and file-list claims included. **G-C8**: every one
of the 3,416 records tried, 1,076 rebuilt and scored through `gate_diff`, 2,340 not rebuildable faithfully, 0
failed; 122 moved, all attributed; 0 violations. Scorer `9eec3d6c…`, harness `75bfbc39…`, repaired `d7c298d4…`.

**Mutation.** Python: only `styxx/diffgate.py` copied (to scratch `mut8/`, LF) and loaded as `styxx.diffgate` by
a pytest plugin; the core modules, the three pin tests deselected; control 658 passed. **44 mutants, 44 killed**
(three of Y-2's — a deleted file's opening line, a hunk starting at 1, a context line ending line 1 —
survived a rerun made after the renaming and were killed once three pinned pairs holding those shapes were
added; the scorer's own tests are skipped under this plugin, since the scorer refuses an
instrument that is not the checkout's):
Y-1's shape test (every pair, no pair, `/dev/null` ignored, the timestamp kept), each place `loose` is set or
reset, the `---` and `+++` doubts, the case-collision note, the uncounted-file note (off, widened to a binary line
under a git header, narrowed by a form), the gate ignoring the note for the count, the scope, a path claim, the
git door's collision; Y-2's line-1 drop (everywhere, nowhere, a created or deleted file's opening line, a hunk
starting at 1, a context line ending it), the line-1 test note (off, on every test), the count ignoring the note,
the indent reading, the symbol doubt; Y-3's set (empty; the next character not asked; a test name not widened,
not opening on the set; the reason's case); Y-4 (exact only, any suffix, a creation still modified); Y-5 (the
pairing verifying, withdrawing everything, withdrawn after the verdict, contradicting). Port: **38 mutants** of
`diffgate.js`, in memory, held to the 240 pinned pairs and 99 extra records (detail and verdict): **38 killed**
(four survived an earlier run and were killed once the extras gained a line with no hunk header before a header, a
context line likewise, a `diff --git` ending a loose stretch, and `/dev/nullx`, now also pinned pairs). Scorer:
**34 mutants** of `path2_gates.py`, in memory, through the `test_v3_*`, `test_x7_*`, `test_x8_*` and pinned-pair
tests: **34 killed** (eleven survived an earlier run and were killed by the tests this round then added: the path
oracle, the door's accounting, the literal pins, the Unicode refusal, the skew re-derivation, the indent oracle,
and the four rules' admissions and Y-5's revert — the table checks were made a function, `check_tables`, that
the tests call on the mutated scorer).

**Tests.** The 30 modules that import `styxx.diffgate` or read `web/gate`, and `tests/test_ledger.py`
(py -3.12): 1,683 passed, 1 skipped, 6 xfailed, 2 failed — `tests/test_gitlab_job.py`'s two job tests, which fail
on this machine whatever the code (`bash` resolves to WSL's `bash.exe`, which cannot start). `LEDGER.md` is
unchanged (no PREREG, seal or certificate added).

---

## F. What is still not repaired

Carried from the seventh pass, each identical on `main`: amendment limits 1, 2 and 4 and the docstring half of
limit 3; COMPAT reading Python's `\s` in all five languages; a lone `\r` inside a line; `..docs`; F-4's
abstentions; V-4 reading `docs/.` plus a period as a parent; `path2_differential_gates.json` uncommitted
(amendment limit 7, owed to the RESULT); the corpus gates not run on the real shelf; #128's modes 2, 3, 5 and
6; identifiers compared as text, not under NFKC; a line CPython refuses read as a definition when an ASCII
non-name character follows the name; the port's claim templates reading the description with JavaScript's
classes; a declared non-ASCII `adds_symbol` MALFORMED in the port; an on-tree prefix spelled with `.`, `..`
or `...` segments; `+++ /dev/null` with no `---` line raising in both ports; an added `async def` read by
neither template; `tests/test_gitlab_job.py` starting WSL's `bash.exe`; `build_ledger.py` writing `LEDGER.md`
with the platform's line ending.

New or restated this round:

- **What abstaining costs.** Y-5 withdraws VERIFIEDs the seventh pass gave rightly ("Added 1 test." over one
  new test beside one changed test); Y-2 withdraws the count beside any U+FEFF-led added test, line 1 included;
  Y-3 withdraws claims naming a skew code point on every Python, 3.12 included, where the runtime and the table
  agree; Y-1 withdraws file-list claims on hand-written diffs whose next header is written without a hunk
  header right after it, with a blank line before it, or as a rename, and on GNU output that names a file
  outside any header pair. Each is an abstention where `main` gave a
  verdict, right or wrong, and each is the same in both ports.
- **Where the reading is still `main`'s, and wrong.** A removed `-- users` inside a non-exact hunk is still read
  as a file header: the file list now abstains, but `tests_added` outside `[net, got]` contradicts as `main`
  does, with a reason that cites the partial `net`. A `def test_` line in a markdown file, in a string or split by
  a backslash is still read by `got` when no pairing acts, as on `main`. GNU header paths keep their timestamp
  as keys (path claims unresolved, a bare-filename `only_touches` prefix accusing, BC-1 reading no Python file),
  as on `main`. A file GNU names outside any header pair is still not counted; the file-list claims now abstain
  beside one, where `main` answered.
- **Y-3's sources.** 13.0.0 and 14.0.0 are derived from Unicode's Age property, not from a database of those
  versions (none is on this machine); a property change between 13.0 and 15.0 of a code point assigned before
  13.0 is covered only through the general-category margin. 15.1.0 is held by 16.0.0's difference, checked for
  15.1's named changes, not for the whole database.
- **The Python's own templates.** The claim templates, COMPAT's patterns and `\b` in the Python still read the
  runtime's classes: under 3.14 the Python and the port differ in a claim's `detail` where the template's name
  group runs into a 16.0 letter — on `main` too, and more there (section E).

The defects above that exist identically on `origin/main` are follow-up issues.

---

## Protocol changes the operator is asked to accept

- **Y-1, Y-2 (with its extension), Y-3 and Y-5** are post-amendment rules that abstain, each with its own
  revert and admission; **Y-4** repairs a defect `main` has, with its own revert and admission. G-C3 is waived
  for each.
- **Y-5 replaces the PREREG #101 table row** "`net = n` → VERIFIED" by UNCHECKABLE; the other rows stand.
- **Y-2's extension changes the sixth pass's W-1 reading** for the test count: a U+FEFF-led added test at line 1
  is read by the parse (the blob, the sides, `hit`) as before, and abstains the count.
- **G-C7 extended, G-C8 on every record with its counts blocking, the corpus sampling (every definition claim,
  1 in 25 of the rest, up to 3,000), the literal pins, the Unicode refusal and the skew re-derivation** are
  blocking gates of `path2_gates.py`, in both modes.
