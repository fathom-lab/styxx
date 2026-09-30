# NOTE — PATH-2, ninth pass: the licensed-difference rule

2026-09-27. Branch `fix/diffgate-path-resolution` (pull request #161), head `ca2a3629` before this round;
`origin/main` `2a6ce0a3` is merged in, and `main`'s `styxx/diffgate.py` there is the file 7.48.0 ships and the
scorer's baseline (sha256 `9b620e00…`, LF).

This note sits on top of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third to
eighth-pass notes. **None of them is edited.** Where one is wrong, the correction is here (section D).

**How this round was worked.** The code, the port, the pinned pairs, the tests and the scorer changes were
written in the working tree, backed up as patches outside the worktree as they grew, and measured there
(section E); this note was written last and is committed alone, before any of that code, which follows in its
own commits. Every number below was taken on the working tree those commits hold.

---

## The merge bar, and the rule this round adds

Held as the orchestrator states it:

1. The reproductions of #97, #121 and #101 are fixed.
2. **No claim reads worse on the branch than on `origin/main` on any door** — Python `gate_diff_text`,
   Python `gate_diff`, the JavaScript port — under any supported Python (3.9 to 3.14): no new false VERIFIED,
   no new false CONTRADICTED, no new Python/JavaScript disagreement. An abstention where `main` gave a verdict
   is not a false verdict, provided both ports abstain identically.
3. The scorer cannot admit a planted defect.

**The licensed-difference rule** — the endpoint of the eighth pass's convergence principle. Eight rounds each
found a place where a repair read a line rightly and a claim wrongly, because `main`'s answer there had been
right by two errors that cancelled, and the repair removed one of them. The eighth pass abstained beside the
shapes it could see; round 8 found four more. No list of shapes has been complete, so this round stops
listing them. For `tests_added`, `symbol_added`, `files_changed_count` and the path claims, the instrument now
computes **`main`'s own reading** beside its own — `main`'s patterns, over `main`'s line split, into `main`'s
status map — in each of `main`'s two spellings:

- `main`'s Python: `str.splitlines()`, Python's `\s` (`str.isspace`, also what `str.strip()` strips), its
  `\w` and `\b` (read from the pinned name table, with a skew code point after a name read as a doubt), and a
  line start (`re.M`'s `^`) only after `\n`;
- `main`'s port: a split on `\r\n`, `\r` and `\n`, JavaScript's `\s` (also what `trim()` strips), ASCII `\w`
  and `\b`, JavaScript's `.` (no line terminator) in its BIN-1 patterns, and a line start after `\n`, U+2028
  and U+2029.

Both ports compute both spellings from the same code point lists (the Python writes them as integers, so no
runtime is asked; the port uses its own runtime for its own spelling, which is that spelling), so the
condition is the same in both ports. **Where this reading differs from either spelling of `main`'s and no
named repair licenses the difference, the claim abstains, with a reason that names the difference.** The
named repairs and their preconditions: #97's exact and suffix tiers (the match is one of them); #121's dotted
key (the differing keys are one key undotted); #101's one-to-one pairing (it pairs lines this count and
`main`'s read alike, so it is asked only once the counts agree); W-1's exact hunk (the line is one an exact
hunk's counts read as content). Nothing else licenses a difference: not F-2, V-1, W-2 or Y-2 (readings of a
definition line), not Y-4 (GNU's `/dev/null<TAB>`).

---

## A. What the eighth-pass review found

Both round-8 reviewers asked for repairs before a merge. Numbered as the review file holds them (R0 the
scorer-and-protocol lens, R1 the regressions lens; the orchestrator's brief calls the regressions lens
reviewer 0). Reproduced on `ca2a3629`, every finding confirmed:

| | finding | severity | vs `main` |
|---|---|---|---|
| R1.0 | `tests_added`: the branch's definition-line reading (V-1's indent and separator, W-2's identifier, F-2's split) changed `got` on lines `main` read otherwise, and `main`'s count had been right by a second miscount both share (a `def test_` in markdown or a string, a test behind a lone CR or a backslash) — new false VERIFIEDs and CONTRADICTEDs on every door (R1, R2) | blocker | **regression** |
| R1.1 | `symbol_added`: `main`'s `def`/`class` pattern hit a line the branch rejects (an NBSP indent or separator, a `\b` ending the name before a combining mark), which balanced a definition both miss (a lone CR, a backslash, `async def`) — new false CONTRADICTEDs (R3); the eighth note's "nothing can balance" a symbol claim was false | blocker | **regression** |
| R1.2 | `files_changed_count`: two shapes Y-1 did not note — a `---`/`+++`/`@@` triple cutting a numeric hunk's walk while its counts were owed, read as a clean header (`git diff -U0` bytes, R4), and a `diff --git` file replaced by a GNU pair naming another path (R5) — where #121's twins or W-1's exact hunk removed the error `main`'s shared one had balanced | blocker | **regression** |
| R1.3 | path claims: #121's dotted keys and Y-4's `/dev/null<TAB>` exposed to the basename tier a file `main`'s key hid, and a claim naming another directory verified (R7, a monorepo shape in real git output; R6) | blocker | **regression** |
| R1.4 | a file holding one definition line CPython refuses defines nothing, and the branch counted its other lines where `main`'s miscount gave the verdict whole-file truth supports (111 moves per Python on the name grid) | minor | **regression** (reading-dependent) |
| R0.0 | a defect planted in Y-1's own git-door code (`_status_notes` returning `{}`) passed every check: `rebuild()` refused case-colliding records, and no oracle read `_status_notes` | blocker | scorer |
| R0.1 | G-C8 could never rebuild a record with a dot-led path segment (0 of 23 dotted pinned pairs) and ran with rename detection off, so #121 at the git door and a rename keyed by its old path (P31, P26) were admitted | blocker | scorer |
| R0.2 | `compat_violations` re-derived C-2's languages and flags only, not the removed and signature_changed lists or the reason (P12, P32 admitted) | major | scorer |
| R0.3 | corpus mode sent only 1 in 25 PRs without a definition claim through the git door, and the stated reason was wrong: `gate_diff` also reads every file-list claim through its own code (P27 admitted) | major | scorer |
| R0.4 | G-C1 compared the count of never-read sentences, not the sentences, and never scored `--strict` (P22, P24b admitted) | major | scorer |
| R0.5 | `tests/test_ledger.py` leaves `papers/LEDGER.md` rewritten with CRLF in an LF checkout on Windows | minor | pre-existing (follow-up) |

---

## B. The changes: Z-1 to Z-5, in both ports

**Z-1 — `tests_added`** (R1.0). `got` is asked against `main`'s `^\s*def test_` count in both spellings; where
the three differ, or where BC-1's "the diff holds a Python file" (read on this reading's keys) is not
`main`'s (read on `main`'s keys, in each spelling), the count abstains: "this reading counts 2 added test
definitions where main's Python counted 1 and its port 1 (`^\s*def test_` over main's line split); no repair
licenses the difference". It is asked after A-1, Y-2 and Y-3 and before the #101 pairing, which is licensed
once the counts agree: with `got` equal to `main`'s, the pairing can only withdraw (Y-5). Where `main` raises
on the diff (a `+++ /dev/null` with no `---` before it, in one spelling only), the count abstains too.

**Z-2 — `symbol_added`** (R1.1). `hit` is asked against `main`'s `^\s*(?:def|class)\s+NAME\b` in both
spellings — `NAME` as each of `main`'s templates captured it (`[A-Za-z_]\w*` with Python's `\w`; with ASCII
`\w` in the port) — and BC-1 as for Z-1; where they differ, the claim abstains ("this reading finds no added
definition of 'foo' where main's Python did and its port did …"). A match of `main`'s Python whose `\b` turns
on a skew code point is a difference too (the supported Pythons read it apart). It is asked after the claimed
name's own rules and Y-2, before the #101 pairing.

**Z-3 — the file list** (R1.2). The raw door computes `main`'s status map over `main`'s Python split and over
its port's split. Where the two differ, `main`'s ports read the list apart and the file-list claims abstain.
Otherwise this reading's map is held to `main`'s with W-1's exact hunks read as content: each of this
reading's keys, undotted, is one of `main`'s, each of `main`'s is one of theirs, and `main`'s status for it is
one of theirs (#121); anything else — Y-4's `/dev/null<TAB>` among them — is an unlicensed difference, and the
claims abstain. Where the maps differ only by those licences, three doubts `main`'s reading also held make the
claims abstain: a `---`/`+++` pair after lines no hunk count holds, read as a header because it has a header's
shape (the eighth pass's exception; R4's triple-cut pair is one), a `diff --git` file its next `---`/`+++` pair
replaced, naming another path (R5), and a `diff --git` header whose paths neither reading can read
(`--no-prefix`). A doubt counts only beside a licensed difference: with the maps equal, the claims read as
`main`'s do. At the git door the file list is git's `--name-status`; `main` split it with `str.splitlines()`,
and where that cuts a path (`core.quotePath` off, a U+2028 in it) the claims abstain.

**Z-4 — the path claims** (R1.3). A claim with a directory component that only the basename tier matches reads
UNCHECKABLE: "'packages/web/.eslintrc.json': only a file with the same name in another directory is in the diff
('.eslintrc.json', status 'M'); #97 licenses the exact and suffix tiers only". This withdraws `main`'s own
false VERIFIEDs too (its single loop matched by basename), which the bar allows. A bare name is resolved by the
suffix tier and is unchanged.

**Z-5 — the whole-file reading, decided** (R1.4). CPython reads a file whole: one definition line it refuses
and the file defines nothing. This reading reads line by line. So an added line that opens a definition of a
name when read loosely (any whitespace either of `main`'s spellings reads, and U+FEFF, around `def`/`class`,
`async` too) and that this reading refuses — a character CPython does not take for indentation or a separator,
a name running into one no identifier holds — makes `tests_added` abstain when it stands in a Python file (the
undotted key's suffix, BC-1's) or outside any file, and `symbol_added` abstain when the claimed name's
definition is in that file. (The reading of such lines as defining nothing, which the eighth pass recorded as a
protocol choice, is kept for the lines; the claims that read their file no longer rely on it.)

`_find_path`, the pairings, the hunk walk and every earlier rule are unchanged. None of Z-1 to Z-5 gives a
verdict `main` could not give: each only turns this reading's verdict into an abstention, identically in both
ports.

What each round-8 reproduction reads now, on every door where it has one, under 3.12 and 3.14:

| repro | claims | `main` | eighth pass | now |
|---|---|---|---|---|
| R1 (git) | "Added 1 test." / "Added 2 tests." (truth 1) | V / C | C / V | U / U (Z-1) |
| R2 (git) | "Added 0 tests." / "Added 1 test." (truth 1) | C / V | V / C | U / U (Z-1) |
| R3 (git) | "Added function foo." (truth T) | V | C | U (Z-2) |
| R4 (`-U0` bytes) | "3 files changed." / "4 files changed." (truth 3) | V / C | C / V | U / U (Z-3) |
| R5 (hand) | "2 files changed." / "3 files changed." (truth 2) | V / C | C / V | U / U (Z-3) |
| R6 (hand) | "Modified lib/api.py." / "Deleted lib/api.py." (truth F) | U / U | V / V | U / U (Z-3) |
| R7 (git) | "Updated packages/web/.eslintrc.json." ×2 (truth F) | U / U | V / V | U / U (Z-4) |

---

## C. The scorer

`papers/closed-model-frontier/path2_gates.py`:

- **Z-1 to Z-5 in the counterfactual**: each has a revert (Z-1 `_tests_differ`, Z-2 `_symbol_differs`, Z-3
  `_files_differ`, Z-4 `_basename_only` answer nothing; Z-5 neither whole-file reading abstains), an
  admission (each only abstains, on the kinds it reads; one whose abstention the claim does not show was woken
  by another rule's revert and is admitted as that, as Y-1 and Y-3 are), and an oracle: `main`'s reading of the
  diff in both spellings is the scorer's own (`own_main_status`, `OwnMain`: `main`'s Python is this interpreter,
  `main`'s port is spelled out), and `expected_tests`, `expected_symbol`, `expected_count`,
  `expected_only_touches` and `expected_path` re-derive each rule; `own_read` re-derives Z-3's note, which must
  equal `_diff_notes`'s.
- **W-1's revert calls `_dev_null`**: the fifth pass's parsers, which W-1's revert restores, compared the whole
  header path with `/dev/null`, so W-1's revert also reverted Y-4 and was credited, and refused, on diffs where
  only Y-4 acts. They now read Y-4's `_dev_null`, and Y-4's revert acts through them.
- **Four-rule attributions**: a U+FEFF-led test is abstained on by Y-2, Z-1 and Z-5 at once (and read by W-1 at
  line 1), so no set of three reverts gives `main`'s claim back. When sizes one to three fail and every revert
  together gives it back, the rules whose revert is needed with every other one reverted are tried as a set; each
  must still admit the move.
- **Z-3 woken by #121**: a doubt `main` held counts only beside a licensed difference, so reverting #121 alone
  gives `main`'s claim back although the move is Z-3's abstention. There the table's conditions are not asked
  (#121 moved no count and no scope); G-C7 re-derives the claim either way.
- **R0.0**: G-C7 holds `_status_notes` (Y-1 at the git door) to `own_status_notes` on every record's paths and on
  a variant with one path upper-cased, whether or not the record rebuilds; and Z-3's git-door reading
  (`_status_differs`) to the scorer's.
- **R0.1**: the git door's repositories are bare, written by `git fast-import` with `core.ignorecase` off — no
  file touches the disk — so a dot-led segment and two paths that differ only in case rebuild; `_safe` refuses
  only `.`, `..` and `.git` segments. A record that deletes one file and creates another is scored a second time
  with git's rename detection on (`--name-status` writes `R<n> old new`, and G-C7 keys it by the last field); the
  raw door is not compared with that pass (git's `R` against the rendered text's `M` is `main`'s reading too).
  The door reports how many scored records hold a dotted path and how many renames it detected.
- **R0.2**: `compat_violations` re-derives C-2's whole reading — verdict, reason, the removed and
  signature_changed lists, languages, counts, candidate — with the undotted suffix and scaffold tests and
  `main`'s COMPAT patterns (read through BASE; this branch does not change them).
- **R0.3**: corpus mode sends every eligible PR with a file-list or definition claim through the git door (still
  under `--git-sample`), and the docstring's reason is corrected.
- **R0.4**: `uncovered_texts` joins G-C1's fields, and every record is scored a second time with `strict=True` for
  both instruments: strict moves no claim, and each strict verdict is FAIL exactly when a claim is CONTRADICTED
  or UNCHECKABLE.

**Planted-defect proofs**, each a committed test (`test_x9_*`), each record scored clean unplanted and refused once
the defect is planted in a scratch copy of `diffgate.py`: Z-1 not asked, `main`'s Python count read with the
port's spelling, Z-1 accusing instead of abstaining (refused by Z-1's admission); Z-2 without `main`'s port;
`main` raising not asked (Z-1, Z-2); Z-3's doubt never counted, its unlicensed difference not read, its
replaced-file doubt not recorded, its git-door reading (`_status_differs`) not asked; Z-4 off; Z-5 refusing
nothing, not asking the symbol's own file, reading markdown as Python; C-2's removed side on the dotted key (P12)
and its reason on the undotted key (P32); the never-read sentences lower-cased (P22); strict passing a
contradicted gate (P24b); `_status_notes` returning `{}` (R0.0), on raw records and on the case-folded variant of
a record with no collision; and through the git door, rebuilt now: a case collision not noted, #121 reverted at
the door (P31), a rename keyed by its old path (P26); and in corpus mode, a deletion read as a modification at
the door on a PR with a file-list claim only (P27).

---

## D. Corrections to earlier notes

1. **Eighth pass, section B, Y-2's extension**: "A symbol claim keeps the line-1 reading: whether a name is
   defined is not a count, and nothing can balance it." False (R1.1): `main`'s hit on a line the branch rejects
   balanced a definition both readings miss. Z-2.
2. **Eighth pass, section C** (and the scorer's docstring): corpus mode sampled "every PR with a `tests_added` or
   `symbol_added` claim — `gate_diff` feeds exactly those through its own blob and sides". `gate_diff` also reads
   every file-list claim through its own `--name-status` parse, `_status_notes` and notes (R0.3).
3. **Eighth pass, section E**: "0 worse" on every set. False: round 8's regressions reviewer measured 230 WORSE
   claim-door cells in 80 cases per Python with truth independent of any refused file, and 277 (3.12) and 295
   (3.14) more that depend on one (R1.0 to R1.4).
4. **Eighth pass, G-C8 "on every record"**: every record was tried, but 2,340 of 3,416 were not rebuildable, none
   with a dot-led segment, and renames were never read (R0.1). G-C1 "extended" compared the count of never-read
   sentences, not the sentences, and never the strict verdict (R0.4).
5. **PREREG, #97**: "else one with the claim's basename" resolves a claim. Kept as the resolution; a claim with a
   directory component that only that tier resolves now abstains (Z-4, a protocol change below).
6. **`NOTE_path1_pairs_repinned_by_path2_2026_09_25`** records the PATH-1 pairs PATH-2 re-pinned. One more moves
   now, and its verdict moves: `path1:unrepaired-typo`'s `file_touched` claim (".githiub/workflows/dependabot.yml",
   the typo PATH-1 left unrepaired) was VERIFIED against `.github/workflows/dependabot.yml` by the basename tier
   and is UNCHECKABLE by Z-4; its `only_touches` accusation (PATH-1's mode 6) does not move. The pair records it.

---

## E. What was measured

Every number here was taken on the working tree the commits after this note hold (`styxx/diffgate.py` sha256
`e975d098…`, LF), against `main`'s file (`9b620e00…`) and the eighth-pass head (`ca2a3629`).

**1. The reproductions.** R1 to R7 read as section B's table on every door each has — `gate_diff` on the
repository git built, `gate_diff_text` on git's bytes or the hand-written text, the port on the same text —
under Python 3.12 and 3.14, identically in both ports. Each is a pinned pair (`path2:z1-*` to `path2:z5-*`,
27 added this round) and a test (`test_x9_*`). The issues' own reproductions (`path2:97-two-readmes`,
`path2:121-dotfile-twins`, `path2:101-the-issue` and the `test_97_*`, `test_121_*` and `test_101_*` tests) read
as at the eighth pass: fixed. Of the pinned pairs written for the three repairs one moves,
`path2:97-basename-still-resolves` (VERIFIED → UNCHECKABLE, Z-4). 36 pinned PATH-2 pairs and one PATH-1 pair are
re-pinned in all, each with its old verdict in its `repinned` field.

**2. The regression differential.** The round-8 regressions reviewer's harness, unchanged — its generators, its
truth model (CPython's parser, and git's own reading of the repository it built) and its comparison — with
`main`'s file against this working tree, three doors per case, under Python 3.12 and 3.14. Thirteen sets,
23,110 cases:

| set | cases | what |
|---|---|---|
| ev | 2,965 | every reproduction and cell the round-8 reviews named |
| tgt | 38 | round 8's targeted shapes |
| grid | 2,000 | the name grid: Unicode spaces, marks and letters around `def`/`class` and the claimed name |
| r1–r5 | 12,700 | the reviewer's randomised diffs at round 8's seeds: non-exact hunks, GNU headers and timestamps, CRLF and lone CR, U+FEFF, renames, binary lines, dotfile twins |
| n1, n2 | 3,000 | the same generator at two fresh seeds (9901, 9902) |
| g0–g2 | 2,407 | this round's generator: same-basename files of a monorepo in real git (root and nested `index.js`, `README.md`, `.eslintrc.json`, `__init__.py`, `conftest.py`); `main`'s doubts beside a licensed difference (`git diff -U0` bytes with a SQL `-- x` changed to `++ x` beside dotfile twins, a `diff --git` binary section replaced by a GNU pair, a `--no-prefix` binary header); NBSP, U+3000, U+000B, U+000C, U+001F, U+0085 and U+2028 around `def`, combining marks, a middle dot, U+00B2, a lone CR, a backslash continuation, `async def`; a refused definition line beside tests and classes; and R1 to R7 |

| | Python 3.12 | Python 3.14 |
|---|---|---|
| claim-door cells | 664,362 | 664,372 |
| a claim wrong where `main` was right (WORSE) | **0** | **0** |
| WORSE on the eight shared sets at the eighth-pass head | 507 | 525 |
| a new Python/port disagreement on the same bytes | **0** | **0** |
| a verdict where the truth model reads two ways | 870 | 870 |

The eighth-pass head's 507 and 525 are round 8's 230 and its 277 and 295. The 870 are claims the truth model
gives two readings for — a rename counted as one file or two, a prefix read as a directory or a basename, a
claim naming a file in a case the diff does not hold. On the eight shared sets they are a subset of the eighth
pass's (368, of 1,703 under 3.12 and 1,603 under 3.14; none new); 502 are on the five new sets, of the same
classes. The git door against the port on git's bytes: 844 claims per Python where `gate_diff` gives a verdict
and the port and `gate_diff_text` on the same bytes both abstain. Every one is a file-list claim whose raw
reading holds a doubt (Y-1's `---`/`+++` after lines no hunk count holds, or a doubt `main` held beside a
licensed difference; 524 of them on `-U0` bytes) while `gate_diff` reads git's `--name-status`, which holds
none. On the shared sets every one was there at the eighth pass (none new, 17 gone); none is WORSE.

**3. Recall cost.** On the committed differential corpus (3,443 pairs, 7,311 claims; `main` raises on one
record), `main` decides 2,557 claims. This branch abstains on 369 of them: 217 at the eighth pass and **152 new
this round** — Z-4 81, Z-1 34, Z-2 18, Z-3 15, Z-5 4; by kind `file_touched` 55, `tests_added` 37,
`file_created` 22, `symbol_added` 19, `files_changed_count` 10, `file_deleted` 5, `only_touches` 4. 123 of the
152 are on records older than this round (78 fuzzed, all Z-4; 44 pinned PATH-2 pairs; one PATH-1 pair), 29 on
the pairs it adds. It decides 11 claims `main` abstained on. On the eight shared sets of the regression
differential, which carry a truth model, the claims `main` read right and this branch abstains on went from
49,757 to 72,640 claim-door cells under 3.12 (of 478,117) and from 54,843 to 78,684 under 3.14: this round
costs 22,883 and 23,841 cells there. Those sets are built around the shapes the rules abstain on (lone CR,
U+2028, U+FEFF, dotfile twins, same-basename files), so that rate is not the rate on ordinary diffs; the
committed corpus's 152 of 2,557 is nearer it.

**4. The Python/port differential.** 3,443 pairs, 7,311 claims (623 verified, 1,576 contradicted, 5,112
uncheckable), **0** disagreements; 267 pinned pairs through `check_pairs.js`, 0. With `main` on both sides, 43;
`main`'s Python against this port, 375; this Python against `main`'s port, 383. Against the eighth-pass head,
155 records move in the Python and 155 in the port: 59 pinned pairs this round adds or re-pins, and 96 fuzzed
records, every one by Z-4 (the fuzzer names files with a directory that only the basename tier matches; 78
claims VERIFIED → UNCHECKABLE, 35 reason-only). No real-corpus record moves. The rebuilt bookmarklet, loaded in
Node with the browser stubbed, reads the 267 pinned pairs as the Python does.

**5. The scorer**, `path2_gates.py differential` on the working tree: every gate passes except G-C0 (the tree is
modified, as it must be before the commits); 374 records moved, one of them where the baseline raises. G-C7: 0
oracle violations, the new oracles included. G-C8: every one of the 3,443 records tried, 1,129 rebuilt through
`git fast-import` and scored (36 holding a dotted path; none could be at the eighth pass), 2,314 not rebuildable
faithfully; 366 scored a second time with rename detection on (2 renames detected); 220 moved, all attributed,
0 violations. G-C1 with `uncovered_texts` and the strict pass: 0 violations. G-C3: no waiver used. The
attributions include `#97+Z-4` 16 — with #97 reverted a claim resolves by basename, where Z-4 then abstains, so
only the pair gives `main`'s claim back — and 91 joint. The clean-tree run after the commits is recorded in
`web/gate/README.md`.

**6. Mutation.** Every ninth-pass change was mutated in a scratch copy of `diffgate.py` only (loaded in place of
the worktree's; the worktree was never written): 44 mutants, 40 killed at once. The four that survived (`main`
raising not asked for Z-1 and Z-2, and not read for Z-3; a replaced file compared on its new path only; Z-5's
lines outside any file not read) each got a pinned pair and a test, and all four are killed. The port: 37
mutants of a scratch copy of `diffgate.js`, held to the pinned pairs and 521 generated cases as the Python reads
them, all killed. The scorer: 24 mutants of `path2_gates.py` in memory (the file on disk read, never written),
through its tests; 21 killed at once, and the two that survived (the git door's Z-3 oracle not asked, the
case-folded `_status_notes` variant not asked) and one written after them (a verdict where the baseline raises
admitted) each got a test, and all three are killed. With every mutation undone, the tests pass.

**7. The tests.** The 30 diffgate and web-gate modules and `tests/test_ledger.py` under Python 3.12: 1,859
passed, 1 skipped, 6 xfailed, 3 failed — the two `tests/test_gitlab_job.py` job tests, which need a bash this
machine's WSL does not provide (as at every earlier pass), and
`test_port_is_current::test_the_gate_readme_names_the_same_instrument`, which asks the README for this file's
sha256 and passes once the README commit lands.

---

## F. What is still not repaired

Each of these reads the same wrong way on `main` and on this branch, in both ports; none is a regression, and
each is left for an issue with its reproduction.

1. **A GNU header with a timestamp keys its file by path and timestamp.** `--- a/src/x.py<TAB>2024-05-06 …` and
   the same `+++` line: "Modified src/x.py." is UNCHECKABLE (not in the diff) and "Added 1 test." finds no Python
   file (BC-1). An abstention on a true claim, not a false verdict.
2. **A quoted path keeps its quotes.** `diff --git "a/src/sp ace.py" "b/src/sp ace.py"` with the same quoted
   `---`/`+++` lines: "Only touches src/." is CONTRADICTED, listing `'"b/src/sp ace.py"'` outside `src`.
3. **A `diff --git` file a GNU pair replaces is dropped.** A `diff --git` binary section, then a `---`/`+++`
   pair for another file: "2 files changed." is CONTRADICTED ("diff changes 1 files"). Z-3 doubts this shape
   only beside a licensed difference.
4. **A `--no-prefix` binary header is dropped.** `diff --git img/logo.png img/logo.png` and its `Binary files`
   line, then a second file: "2 files changed." is CONTRADICTED.
5. **A lone CR hides a definition.** `+a = 1<CR>def test_a():` is two lines to CPython and one to both
   readings: "Added 1 test." is CONTRADICTED.
6. **A backslash continuation hides a definition.** `+def <BACKSLASH>` then `+    test_b():`: "Added 1 test."
   and "Added function test_b." are CONTRADICTED.
7. **A file CPython refuses for another reason still counts.** A new test file ending in `x = (`: "Added 1
   test." is VERIFIED, and CPython defines nothing there. Z-5 reads refused definition lines only.
8. **A `def test_` in a markdown file counts when the diff holds a Python file.** A fenced `def test_example():`
   in `docs/guide.md` beside a changed `src/a.py`: "Added 1 test." is VERIFIED.
9. **`+++ /dev/null` with no `---` line before it raises**: `AttributeError` in the Python, `TypeError` in the
   port, on `main` and here.
10. Carried from earlier passes: `tests/test_ledger.py` leaves `papers/LEDGER.md` rewritten with CRLF in an LF
    checkout on Windows (R0.5); `gate_diff` with rename detection on reads a rename as `R` where the rendered
    text reads `M` (`main`'s reading too); 27 code points lower-case differently between the engines' Unicode
    versions; the port's claim templates read the description with JavaScript's whitespace, word and
    word-boundary classes (the README counts them).

---

## Protocol changes the operator is asked to accept

- **Z-1 to Z-5** are post-amendment rules that only abstain, each with its own revert, admission and oracle;
  G-C3 is waived for each as for every post-amendment rule (none can make an accusation).
- **Z-4 narrows the PREREG's #97 resolution**: the basename tier still resolves, and a claim with a directory
  component that only it resolves abstains.
- **Z-5 decides the whole-file reading** for the two definition kinds.
- **The scorer**: W-1's revert reads Y-4's `_dev_null`; four-rule attributions; Z-3 woken by #121; the
  `_status_notes` oracle; the fast-import git door with dot-led segments, case twins and a rename pass; C-2
  re-derived whole; corpus mode's git door for file-list claims; G-C1's `uncovered_texts` and the strict pass.
  Each is a blocking gate of `path2_gates.py`, in both modes.
