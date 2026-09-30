# NOTE — PATH-2, tenth pass: W-1 licenses no file list, and one case fold for both ports

2026-09-28. Branch `fix/diffgate-path-resolution` (pull request #161), head `b4c6dbdf` before this round;
`origin/main` `2a6ce0a3` is merged in, and `main`'s `styxx/diffgate.py` there is the file 7.48.0 ships and the
scorer's baseline (sha256 `9b620e00…`, LF).

This note sits on top of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third to
ninth-pass notes. **None of them is edited.** Where one is wrong, the correction is here (section D).

**How this round was worked.** As in the ninth pass: the code, the port, the pinned pairs, the tests and the
scorer changes were written in the working tree, backed up as patches outside the worktree as they grew, and
measured there (section E); this note was written last and is committed alone, before any of that code, which
follows in its own commits. Every number below was taken on the working tree those commits hold.

---

## The merge bar, and the licences

Held as the orchestrator states it:

1. The reproductions of #97, #121 and #101 are fixed.
2. **No claim reads worse on the branch than on `origin/main` on any door** — Python `gate_diff_text`,
   Python `gate_diff`, the JavaScript port — under any supported Python (3.9 to 3.14): no new false VERIFIED,
   no new false CONTRADICTED, no new Python/JavaScript disagreement. An identical abstention in both ports is
   allowed.
3. The scorer cannot admit a planted defect. `main`-identical wrong verdicts become follow-up issues.

**The licensed-difference rule** stands as the ninth pass wrote it, with its list of licences cut back to the
orchestrator's: where this reading differs from `main`'s and no named repair licenses the difference, the claim
abstains, identically in both ports. The named repairs are **#97's exact and suffix tiers, #121's dotted key and
#101's one-to-one pairing**, each with its precondition. The ninth note added a fourth, "W-1's exact hunk (the line
is one an exact hunk's counts read as content)"; round 9 found that the regressions passed through exactly that
licence, and this round withdraws it (K-1). W-1 still reads such a line as content; it licenses no difference in
the file list, and the definition claims were never licensed by it (Z-1 and Z-2 compare the counts, whatever the
lines).

---

## A. What the ninth-pass review found, and what this round's own differential found

Both round-9 reviewers asked for repairs before a merge. Numbered as the review file holds them (R0 the regressions
lens, which the orchestrator's brief calls reviewer 0; R1 the scorer-and-protocol lens, reviewer 1). Reproduced on
`b4c6dbdf` with the reviewers' own scripts, every finding confirmed. B1 to B4 are this round's own, found by the
builder's regression differential (section E.2) on the working tree while the fixes below were being made; none was
in the review.

| | finding | severity | vs `main` |
|---|---|---|---|
| R0.0 | `files_changed_count` and `only_touches`: Z-3 held this reading's file list to `main`'s *with W-1's exact hunks read as content*, so W-1's removal of a file `main` read from an exact hunk's `+++ users` was licensed; that file had balanced a changed file neither reading counts — git's `Submodule vendor/lib a...b (commits not present)` under `diff.submodule=log`, svn's `Cannot display` block, hg's `Binary file p has changed` — and the branch kept the shared miss and removed only the balancing error: new false VERIFIEDs and CONTRADICTEDs on the raw door and the port, on real git output (P1 to P3; 877 and 828 WORSE cells per door at the reviewer's seeds 9101 and 9102) | blocker | **regression** |
| R0.1 | Y-1's collision compared two header paths through the runtime's own lower-casing, and Python 3.12 (Unicode 15.0) and Node 24 (16.0) read 27 code points differently: a diff naming U+10D50 and U+10D70 collided in the port only, and every file-list claim, the ones about unrelated ASCII files too, abstained in one port (P4; 267 and 242 new raw-door disagreements under 3.12) | blocker | **regression** |
| R1.0 | `compat_violations` returned `[]` whenever a compat claim's detail lacked `"languages"`, so a defect inside C-2 that also dropped that key was admitted, and the guard disarmed the oracle for the round-8 C-2 plants (P11, P12, P32) | blocker | scorer |
| R1.1 | where `main` raises, `Tally.pair` checked only that every claim abstains: no gate verdict, no strict verdict, no gate-level field was scored (plants A and B admitted); the ninth note's "every record is scored a second time with strict=True" was false for these records | blocker | scorer |
| R1.2 | G-C8 could not reach Z-3's git-door half: on every rebuildable record `_status_differs` and the scorer's reading both return None, so a defect that stopped Z-3 abstaining at the git door was admitted in both modes | major | scorer |
| R1.3 | corpus mode credited a G-C2 eligibility move on the counterfactual alone; the PR was then excluded from both instruments and no oracle read the rule that moved it (a W-1 defect planted in `_read_diff` admitted) | major | scorer |
| R1.4 | corpus mode sent a PR through the git door for a definition or file-list claim, not for a compat claim, which reads `gate_diff`'s own sides | minor | scorer |
| R1.5 | `DiffGate.base`/`head` and the serialised report (`to_dict`, which the CLI and the Action print) were never compared | minor | scorer |
| R1.6 | G-C0's provenance left out `styxx/declare.py`, which both instruments import, and `path1_extensions.txt`, which the oracle reads | minor | scorer |
| B1 | this reading **raised** where `main` did not: a `+++ /dev/null` line with no `---` line before it in this reading, where `main` had read one (a `--- q` inside an exact hunk, which W-1 reads as content; or a GNU `+++ /dev/null<TAB>...` alone, which `main` did not read as /dev/null) — `AttributeError` in the Python, `TypeError` in the port | blocker | **regression** |
| B2 | beside #121's licensed dotted key, a changed file neither reading counts (a `Submodule` line, hg's and svn's binary notices) had been balanced by `main` merging the dotfile twins: `.env` created beside a changed `env` and a submodule bump, "2 files changed." (truth 3) CONTRADICTED on `main`, VERIFIED here — 6 WORSE cells per door at each of two fresh seeds of round 9's generator, 1 at the reviewer's seed 9101, once K-1 was in | blocker | **regression** |
| B3 | Z-4 abstained in the Python only on a path the two ports' templates extract differently: the path template's `\w` is Unicode in the Python and ASCII in the port, and the `[^.!?\n]{0,60}?` before it lets the port start the path past a character it cannot hold (`Edited Docs/<U+A7D0>/a.md.` is `Docs/<U+A7D0>/a.md` in the Python, `/a.md` in the port); `main` read both alike (3 new raw-door disagreements on the case-pair set) | blocker | **regression** |
| B4 | a Z-3 or Z-4 reason printed a key through `repr()`, and a key is the runtime's lower case: the reason named `lib/<U+1C89>x.md` under Python 3.12 and `lib/<U+1C8A>x.md` in the port, and `repr()` escapes a code point the runtime's Unicode has not assigned — new reason-only disagreements (41 at one fresh seed) where `main`'s reasons agreed | minor | **regression** (reason only) |

---

## B. The changes: K-1 to K-5, in both ports

**K-1 — W-1 licenses no file-list difference** (R0.0). Z-3 now reads `main`'s file list twice, with the lines an
exact hunk's counts hold and without them; where the two differ, `main` read a file (or a status) from content, and
the file-list claims abstain: "this reading's file list differs from main's: main read the file list from lines an
exact hunk's counts hold, which W-1 reads as content (main reads 'users' ('M') from them); a changed file neither
reading counts may have balanced it, so W-1 licenses no file-list difference". This reading's map is then held to
`main`'s full reading up to #121's dotted keys. The reviewer's narrower alternative (abstain only beside an
unrecognised line) was not taken for K-1: which formats name a changed file neither reading counts is an open list,
and an enumerated doubt would repeat round 8's R4/R5 pattern. The git door is unchanged: its file list is git's
`--name-status`, and on P1 it keeps its verdicts (both right).

**K-2 — one case fold** (R0.1). Y-1's collision compares two header paths by the fold `styxx/_fold.py` carries, the
same string byte for byte in `diffgate.js`, generated once by `web/gate/gen_fold.py` under Python 3.14 from Unicode
16.0.0: each code point's full lowercase mapping (so U+0130 folds to U+0069 U+0307), and U+03C2 to U+03C3, because
both runtimes lower-case U+03A3 to either by its context (sha256 `a52cda82…`, 1,461 code points, pinned in the
tests, in `py_side.py` and in the scorer). The key itself is unchanged (it is `main`'s: the runtime's lower case).
The fold is sound on every supported runtime — two paths any of them keys alike, it folds alike — because Unicode
assigns a lowercase mapping with its character and has not changed one between 13.0 and 16.0. Between 15.0.0 and
16.0.0 that is checked exactly (CPython 3.12 against 3.14: 15.0.0's 1,433 mappings are 16.0.0's, and the 27 16.0.0
adds are all of characters 16.0 assigned). For 13.0.0 and 14.0.0 it is read from Unicode's Age, not from those
versions' databases, which are not on this machine: of 15.0.0's mappings, the 40 whose characters 13.0 had not
assigned are all of characters 14.0 assigned (Perl 5.38's UCD, `\p{Present_In=13.0}` and `\p{Present_In=14.0}`),
and 15.1.0 assigned no cased character. The test suite checks `fold(lower(c)) == fold(c)` for every code point against the running Python and
against node's `toLowerCase()`; the scorer checks it against its own interpreter at import. Where a runtime keys two
such paths apart (U+A7CB and U+0264 on Python 3.9 to 3.13), the fold still reads them as one and the file-list
claims abstain, in every runtime alike. The git door's `_status_notes` reads the same fold.

**K-3 — no raise where `main` reads the diff** (B1). A `+++ /dev/null` with no `---` line before it names no file
and is not read as one. Where `main` raised there too, every claim abstains, as the ninth pass already had it (Z-1
to Z-3 read `main` as raising); where `main` read a `---` line this reading does not, Z-3 abstains the file list
(K-1, or its unlicensed-difference test).

**K-4 — a line no reading places, beside #121** (B2). Beside a licensed difference in the file list — after K-1,
only #121's dotted key — Z-3 already abstained where `main`'s reading held a doubt of its own (a header read for its
shape, a replaced `diff --git` file, an unreadable header). A fourth such doubt: a line outside every header, hunk,
context line and blank that is none of git's extended header lines (`index`, `old mode`, `new mode`, `deleted file
mode`, `new file mode`, `copy from`/`to`, `rename from`/`to`, `similarity index`, `dissimilarity index`), not a `\`
marker, not a binary line under a git header, not inside a `GIT binary patch` block (its `literal N`/`delta N`
blocks, each ended by a blank line), and not a GNU line Y-1 already doubts. With the maps equal, nothing moves:
#121's reproduction, git's own output, a `\ No newline` past a hunk's counts, and a git binary patch beside dotfile
twins still count.

**K-5 — a path the ports may read apart reads as `main` read it** (B3). Where a path claim's path holds a character
at or past U+0080, or the sentence runs into it from one, the two ports' templates may extract different paths. The
claim then reads as `main` read it — in each port as `main`'s same port did: `main`'s one-loop resolution over
`main`'s own file list, with `main`'s strings — not by this reading's tiers, which #121's dotted keys would move (a
dotfile in a non-ASCII directory beside a root dotfile is round 8's R7 again). Z-4 is not asked of such a claim. This
gives back `main`'s verdicts there, including `main`'s own basename false VERIFIEDs, which the ninth pass had
withdrawn in the Python only (section F.12).

**The keys a reason prints** (B4). A Z-3 or Z-4 reason prints a key folded by K-2's table and escaped as Python's
`ascii()` escapes it (the port carries `pyAscii`), so the reason is the same text on every runtime.

`_find_path`, the pairings, the hunk walk and every earlier rule are unchanged. None of K-1 to K-5 gives a verdict
`main` could not give: K-1, K-2 and K-4 only abstain, K-3 reads no file where this reading raised, and K-5 gives
`main`'s own verdict.

What each reproduction reads, on every door where it has one (the raw door and the port alike), under 3.12 and 3.14:

| repro | claims | truth | `main` | ninth pass | now |
|---|---|---|---|---|---|
| P1 (git, `diff.submodule=log`) | "1 file changed." / "2 files changed." / "Only touches db/." / "Only touches db/ and assets/." | F/T/F/F | C/V/C/C (git door C/V/C/C) | V/C/V/V | U/U/U/U (git door C/V/C/C) |
| P2 (svn), P3 (hg) | the same four | F/T/F/F | C/V/C/C | V/C/V/V | U/U/U/U |
| P4 (`docs/x<U+10D50>`, `docs/x<U+10D70>`, `docs/a.md`) | "Modified docs/a.md." / "Only touches docs/." / "Only touches src/." / "3 files changed." | T/T/F/T | 3.12 V/V/C/V; 3.14 and port V/V/C/C | 3.12 V/V/C/V; 3.14 and port U/U/U/U | U/U/U/U |
| B1 | "2 files changed." over an exact hunk's `--- q`, then `+++ /dev/null` | — | V | raises | U |
| B2 (git) | "3 files changed." / "2 files changed." / "Only touches env." over `.env`, `env` and a submodule | T/F/F | C/C/V | C/V/U | U/U/U |
| B3 | "Edited Docs/<U+A7D0>/a.md." over `docs/a.md` | F | V (both ports) | U (Python), V (port) | V (both ports, `main`'s) |

---

## C. The scorer

`papers/closed-model-frontier/path2_gates.py`:

- **K-1 to K-5 in the oracles**: `own_file_list_differs` reads `main`'s file list with and without an exact hunk's
  lines (K-1); `own_read` and `own_status_notes` compare header paths by the scorer's own decoding of the fold (K-2),
  read a `+++ /dev/null` with no `---` line as no file (K-3) and hold the fourth doubt (K-4), written out with its own
  list of git's extended header lines; `expected_path` reads a claim whose path the ports may read apart by `main`'s
  own resolution over `main`'s file list (K-5), the character before the path found by the scorer's own match of the
  path templates; a key a reason prints is `ascii()` of its fold. K-1, K-2 and K-4 abstain through Z-3's and Y-1's
  reasons and are attributed by their reverts; K-3 and K-5 move no claim against `main`. The fold is pinned here by
  sha256, decoded by this file's own decoder, and refused at import unless it is sound against this interpreter's
  `str.lower()`.
- **R1.0**: no guard. `compat_violations` requires every COMPAT_EXTRAS key on every measured compat claim and holds
  each to the scorer's own reading; a missing key is `G-C7_oracle:C-2_compat_surface`.
- **R1.1**: where `main` raises, the repaired gate's verdict is held to its claims (`gate_violations`), it is scored a
  second time with `strict=True` (`strict_violations`), and what reads only the summary — the claims' kinds, texts and
  details, the never-read sentences and their counts — is held to `main`'s gate on the same summary over no diff.
- **R1.2**: the git door takes a path segment holding U+0085 or U+2028 (which `str.splitlines()` breaks on and git's
  split does not) and writes its repositories with `core.quotePath` off; and every run, in both modes, scores two
  **door canaries** (`DOOR_CANARIES`), one per code point, through both doors, in tallies of their own — at the git
  door `main`'s `--name-status` split cuts the path and Z-3 abstains, so a defect that stops it is refused there. A
  canary that cannot be rebuilt or scored is itself a violation.
- **R1.3**: before an eligibility move is credited, and on every PR the repaired instrument excludes, the PR's parse is
  held to the scorer's own (`parse_violations`: F-2's split, W-1's status, added lines and sides, the notes, Y-1 at the
  git door, Y-4's `/dev/null` on every header path, #121's key); a move whose parse is refused is counted as
  `refused_by_a_parse_oracle`, never credited.
- **R1.4**: corpus mode sends every eligible PR with a compat claim through the git door too
  (`tried_for_a_compat_claim`), and the docstring says so.
- **R1.5**: `report_violations` holds both instruments to the same `base`, `head` and never-parsed claims, and each
  gate's `to_dict()` to its own verdict, fields and claims, key for key.
- **R1.6**: `styxx/declare.py`, `styxx/_fold.py` and `path1_extensions.txt` join PROVENANCE_FILES, and their sha256s
  (with `_xid.py`'s) join the payload.
- **The counterfactual**: a revert may restore code that raised (the fifth pass's parsers on a `+++ /dev/null` with no
  `---` line); a copy that raises gives nothing back. And Y-4 may move the gate-level fields, on a diff whose header
  names /dev/null followed by a TAB (a diff holding only such a header reads as holding nothing, K-3).

**Planted-defect proofs**, each a committed test (`test_x10_*`), each record scored clean unplanted and refused once the
defect is planted in a scratch copy of `diffgate.py`: K-1 licensing W-1's removal again; K-2 comparing by the runtime's
lower-casing, at the raw door and at the git door; K-4 not doubting an unplaced line, and reading a binary patch's
second block as unplaced; K-5 asking Z-4 of a path the ports read apart; C-2 dropping `"languages"` on a leading-dot
path, alone and with P32 (R1.0); where `main` raises, strict passing an unverifiable gate, the gate failing with no
contradicted claim, a claim's detail altered, a never-read sentence dropped (R1.1); `_status_differs` not asked at the
git door, and `main`'s `--name-status` split as git splits it, each refused by the canaries (R1.2); a W-1 defect that
excludes a PR, refused by the parse oracles in corpus mode (R1.3); a compat reading altered at the git door only, on a
compat-only PR outside the 1-in-25 sample, refused in corpus mode (R1.4); base and head swapped, and the report naming
the head as its base (R1.5); G-C0 reporting a modified `declare.py` (R1.6); and the scorer refusing a fold that misses
a merge.

---

## D. Corrections to earlier notes

1. **Ninth pass, "The licensed-difference rule"**: it named four licences, the fourth "W-1's exact hunk (the line is
   one an exact hunk's counts read as content)". The orchestrator's rule names three; the fourth is the one R0.0's
   regressions passed through, and it is withdrawn (K-1).
2. **Ninth pass, section E.2**: "0 WORSE" and "0 new Python/port disagreements" on every set. False: round 9's
   regressions reviewer measured 877 and 828 WORSE cells per door at seeds 9101 and 9102 (R0.0) and 267 and 242 new
   raw-door disagreements under 3.12 (R0.1); this round's own differential found B1 to B4 besides.
3. **Ninth pass, section C, R0.4**: "every record is scored a second time with `strict=True`". Not the records where
   `main` raises (R1.1).
4. **Ninth pass, section C, R0.0 and R0.1**: the git door's Z-3 oracle was vacuous on every rebuildable record (R1.2);
   the unit test was its only guard.
5. **Ninth pass, section B, Z-4**: "This withdraws `main`'s own false VERIFIEDs too ... which the bar allows." On a
   path the two ports' templates extract differently it withdrew them in the Python only (B3); there the claim now
   reads as `main` read it (K-5), false VERIFIEDs included.
6. **Fifth pass, "Unicode versions"**: `str.lower()` and `toLowerCase()` "differ on 27 code points, all cased only
   since Unicode 15.1. These track the engines, not the port". The 27 were cased by Unicode 16.0 (15.1 added no case
   mapping); and through Y-1's collision (eighth pass) they moved verdicts in one port only (R0.1). The collision now
   reads one fold (K-2); the keys still move (section F.11).
7. **Ninth pass, section F.9**: "`+++ /dev/null` with no `---` line before it raises ... on `main` and here." Here it
   also raised where `main` did not (B1); it no longer raises anywhere here (K-3). `main` still does.

---

## E. What was measured

Every number here was taken on the working tree the commits after this note hold (`styxx/diffgate.py` sha256
`0fc470c5…`, LF; `web/gate/diffgate.js` `c19c0701…`; `styxx/_fold.py` `2ed6a4ac…`), against `main`'s file
(`9b620e00…`) and the ninth-pass head (`b4c6dbdf`, `e975d098…`) — except the regression differential (E.2) and the
mutation runs' opening pass (E.6), taken on `diffgate.py` `8867bbbb…` and port `356bf353…`, and the scorer run (E.5)
and the mutation re-runs, taken on `diffgate.py` `a89af36b…`. Each differs from the committed file in comments only
(the licensed-difference block's tenth-pass paragraph, three reworded lines, and the port's header naming the
Python's sha256 and the tenth pass); the pinned pairs, the port differential, the bookmarklet check and the test
suite were run again on the committed bytes.

**1. The reproductions.** P1 to P4 and B1 to B4 read as section B's table on every door each has, under Python 3.12
and 3.14, identically in both ports. Each is a pinned pair (`path2:k1-*` to `path2:k5-*` and `path2:k-*`, 32 added
this round) or, where the two ports' path templates extract different paths from one sentence (as they do on
`main`), a test (`test_x10_k5_*`); each is a test besides (`test_x10_*`). The issues' own reproductions
(`path2:97-two-readmes`, `path2:121-dotfile-twins`, `path2:101-the-issue` and the `test_97_*`, `test_121_*` and
`test_101_*` tests) read as at the ninth pass: fixed. 9 PATH-2 pairs are re-pinned (K-1: six to UNCHECKABLE from
VERIFIED or CONTRADICTED, three reason-only), each with its old verdict in its `repinned` field; no PATH-1 pair
moves.

**2. The regression differential.** The round-9 regressions reviewer's harness (its runners for the three doors
under both Pythons and node, its truth model — CPython's parser, and git's own `--name-status` over the repository it
built with `git fast-import` and `diff.submodule=log` — and its comparison), unchanged, with `main`'s file against
this working tree, on round 8's and round 9's built sets and on fresh ones: 22 sets, 44,129 cases.

| set | cases | what |
|---|---|---|
| ev, tgt, grid | 5,003 | every reproduction and cell the round-1 to round-8 reviews named; round 8's targeted shapes; the name grid |
| r1–r5 | 12,700 | round 8's reviewer's randomised diffs at its seeds |
| n1, n2, g0–g2 | 5,407 | the ninth pass's sets: round 8's generator at seeds 9901 and 9902, and the ninth pass's own |
| t9 | 19 | round 9's reviewer's targeted set: P1 to P4 and their `-U0`, two-submodule and path variants |
| m1, m2 | 5,000 | round 9's reviewer's generator at its seeds 9101 and 9102 (1,000 git-built and 1,500 hand-written each) |
| f10101–f10103 | 10,500 | the same generator at three fresh seeds (1,500 git-built and 2,000 hand-written each): compensating file-list errors beside `Submodule`, svn and hg lines, dotfile and case twins, `-U0` bytes, GNU, timestamp, plain and Index headers, non-exact hunks, BOMs, lone CR, backslash continuations, whitespace around `def`, combining marks, skew and U+00B2 names, monorepo basenames, Unicode 16.0 case pairs, `+++ /dev/null` before any `---` |
| q1 | 2,500 | round 8's reviewer's generator at a fresh seed (10201) |
| g10301 | 1,500 | the ninth pass's own generator at a fresh seed |
| k1 | 1,500 | this round's own: case pairs whose lowercase mapping differs across Unicode 13.0 to 16.0 (the 14.0 pairs at U+2C2F, U+A7C0, U+A7D0, U+A7D6, U+10570, U+10595; the 16.0 pairs at U+1C89, U+A7CB, U+A7CC, U+A7DA, U+A7DC, U+10D50), the final sigma, U+0130 against `i`+U+0307 and against `i`, the Kelvin sign, U+00C9 and ASCII case, as two paths or one, beside ASCII files, 600 in git repositories and 900 hand-written |

16,000 of the cases are randomised at seeds no earlier round used (f10101–f10103, q1, g10301, k1).

| | Python 3.12 | Python 3.14 |
|---|---|---|
| claim-door cells | 1,454,785 | 1,455,638 |
| a claim wrong where `main` was right (WORSE) | **0** | **0** |
| this reading raising where `main` does not | **0** | **0** |
| a new Python/port disagreement on the same bytes, in the verdict | **0** | **0** |
| a new Python/port disagreement on the same bytes, in the reason | **0** | **0** |
| the git door against the port: `gate_diff` decides and both raw readings abstain | 31,074 | 31,074 |
| the git door against the port: both abstain, with different reasons | 11,355 | 11,341 |
| the git door against the port: conflicting verdicts | **0** | **0** |

While this round's fixes were being made, the same harness read, on the working tree with K-1 to K-3 in, 6 WORSE
cells per door at two of the fresh seeds and 1 at seed 9101 (B2), 3 new raw-door disagreements on the case-pair set
(B3) and up to 41 new reason-only ones at one seed (B4); K-4, K-5 and the reason keys are the answer, and the table
above is the run after them, on every set. Every git-door-against-port cell is a file-list claim the raw door abstains
on, identically in the Python and the port, while `gate_diff` reads git's `--name-status`, which names the submodule
or the case twin the diff text does not (K-1, K-2, K-4, Y-1); none is WORSE at the git door. The port also reads
claims `main`'s port did not extract (a name running into U+00B2 or U+2160: 39 cases on q1, as at the ninth pass on r1
to r5), each as the Python reads it.

Python 3.9 to 3.11 and 3.13 were not run; their Unicode versions (13.0.0, 14.0.0, 15.1.0) bear on this round as
follows. K-2's fold is Unicode 16.0.0's and sound on every version up to it (section B); where 13.0.0 or 14.0.0 keys
two such paths apart, the fold still reads them as one and every runtime abstains alike. K-1, K-3 and K-4 read ASCII
and code point lists both ports carry. K-5's test is `str.isascii()`, and where it holds the claim reads as `main`
reads it on that Python, whatever its Unicode. A reason prints a key through the fold and `ascii()`, which no version
reads differently. What each Python reads names by (the name table and the skew set) is the earlier passes' and is
unchanged.

**3. Recall cost.** On the committed differential corpus (3,475 pairs, 7,402 claims; `main` raises on 2 records),
`main` decides 2,642 claims. This branch abstains on 452 of them: 397 at the ninth pass (on these pairs), and **56 new
this round** — K-1 34, K-2 16, K-4 6; by kind `files_changed_count` 28, `only_touches` 23, `file_touched` 5 — 44 of
them on the 32 pairs this round adds and 12 on the six pairs K-1 re-pins; and it decides again one claim the ninth
pass abstained on (K-5, `main`'s verdict). No real-corpus or fuzzed record moves against the ninth pass. On the eight
sets the ninth note measured recall on (ev, tgt, grid, r1–r5; 478,117 claim-door cells), the claims `main` read right
and this branch abstains on went from 72,640 to 93,888 cells under 3.12 and from 78,684 to 99,932 under 3.14: this
round costs 21,248 cells there under each Python, nearly all of it K-1 on the generators' `-- x`/`++ y` content
lines, which those sets carry by design (round 8 built them around such shapes); the committed corpus's 56 of 2,642
is nearer the rate on ordinary diffs. The claims `main` read wrong and this branch reads right are unchanged there
(41,438 cells under 3.12; 41,436 at the ninth pass).

**4. The Python/port differential.** 3,475 pairs, 7,402 claims (622 verified, 1,579 contradicted, 5,201
uncheckable), **0** disagreements; 299 pinned pairs through `check_pairs.js`, 0. With `main` on both sides, 47; with
the ninth-pass head on both sides, 6, all on pairs this round adds: R0.1's three case pairs, and three records on
which the ninth pass raised, an `AttributeError` in the Python and a `TypeError` in the port (B1).
`main`'s Python against this port, 404 records; this Python against `main`'s port, 412. Against the ninth-pass head,
29 records move in the Python and 26 in the port, every one a pinned pair this round adds or re-pins. The rebuilt
bookmarklet (`bookmarklet.min.js` sha256 `c9a23982…`, 52,373 characters), loaded in Node with the browser stubbed,
reads the 299 pinned pairs as the Python does. `gen_xid.py --check` matches, and `gen_fold.py --check` matches under
3.12 (the two blocks) and 3.14 (regenerated).

**5. The scorer**, `path2_gates.py differential` on the working tree: every gate passes except G-C0 (the tree is
modified, as it must be before the commits); 402 records moved, 2 of them where the baseline raises (every claim
abstains there, and the gate, strict and summary fields are scored). G-C7: 0 oracle violations, the tenth pass's
oracles included. G-C8: every one of the 3,475 records tried, 1,140 rebuilt through `git fast-import` and scored (38
holding a dotted path), 2,335 not rebuildable faithfully; 366 scored a second time with rename detection on (2
renames detected); 226 moved, all attributed, 0 violations; the two door canaries scored, 0 violations. G-C1 with the
report compared: 0 violations; the gate-level fields moved on 3 records, given back by F-2 (2) and Y-4 (1). G-C3: no
waiver used. The attributions include `W-1+Z-3` 39 (with Z-3 reverted the repair still reads W-1's map, so only the
pair gives `main`'s claim back) and 115 joint. The clean-tree run after the commits is recorded in
`web/gate/README.md`.

**6. Mutation.** Every tenth-pass change was mutated in a scratch copy of `diffgate.py` only (`mut10/`, loaded in
place of the worktree's; the worktree was never written): 23 mutants, 17 killed at once. Five of the six that
survived each got a pinned pair or a test and are killed: the header path's leading `./` segments kept (K-2), a
`\ No newline` past a hunk's counts read as unplaced (K-4), the character before a path not asked and not passed
(K-5, two), and Z-4's key printed by `repr()` (the pair's key had been one Python 3.12 cannot print, so `repr()` and
`ascii()` agreed). The sixth is equivalent: K-1's guard for `main`'s full reading raising cannot be reached once
`main`'s two spellings agree. The port: 22 mutants of a scratch copy of `diffgate.js`, held to the 299 pinned pairs,
1,248 generated cases the Python reads alike and six cases whose reasons name each port's own extraction (held to the
unmutated port's verdicts); 20 killed, and the two survivors are equivalent (K-1's guard as above; K-5 reading
`main`'s Python's file list, which equals the port's wherever K-5 is reached, since Z-3 abstains where the two
differ). The scorer: 25 mutants of `path2_gates.py` in memory (the file on disk read, never written), through its
tests; 24 killed at once, and the one that survived (the claims not compared where `main` raises) got a
planted-defect test and is killed.

**7. The tests.** The 30 diffgate and web-gate modules and `tests/test_ledger.py` under Python 3.12: 1,936 passed, 2
skipped, 6 xfailed, 2 failed — the two `tests/test_gitlab_job.py` job tests, which need a bash this machine's WSL does
not provide (as at every earlier pass). `papers/LEDGER.md` is unchanged.

---

## F. What is still not repaired

Each of these reads the same wrong way on `main` and on this branch, in both ports (or, where marked, abstains here
where `main` is wrong); none is a regression, and each is left for an issue with its reproduction.

1. **A GNU header with a timestamp keys its file by path and timestamp.** `--- a/src/x.py<TAB>2024-05-06 ...` and
   the same `+++` line: "Modified src/x.py." is UNCHECKABLE (not in the diff).
2. **A quoted path keeps its quotes.** `diff --git "a/src/sp ace.py" ...`: "Only touches src/." is CONTRADICTED,
   listing `'"b/src/sp ace.py"'`.
3. **A `diff --git` file a GNU pair replaces is dropped.** A binary section, then a `---`/`+++` pair for another
   file: "2 files changed." is CONTRADICTED ("diff changes 1 files").
4. **A `--no-prefix` binary header is dropped.** "2 files changed." is CONTRADICTED.
5. **A changed file neither reading counts** — git's `Submodule p a..b` line under `diff.submodule=log`, svn's
   `Cannot display` block, hg's `Binary file p has changed`: "2 files changed." over a changed `db/q.sql` and such a
   line is CONTRADICTED ("diff changes 1 files"). K-1 and K-4 abstain where a repair would otherwise have removed the
   error that balanced it; alone, both readings miss it.
6. **A lone CR hides a definition** (`+a = 1<CR>def test_a():`): "Added 1 test." is CONTRADICTED.
7. **A backslash continuation hides a definition**: "Added 1 test." and "Added function test_b." are CONTRADICTED.
8. **A file CPython refuses for another reason still counts** (a new test file ending in `x = (`): "Added 1 test." is
   VERIFIED.
9. **A `def test_` in a markdown file counts beside a Python file**: "Added 1 test." is VERIFIED.
10. **`+++ /dev/null` with no `---` line before it raises on `main`** (`AttributeError`, `TypeError` in the port);
    here every claim abstains (K-3).
11. **The key is the runtime's lower case.** Two paths differing only in a case pair Unicode 16.0 assigned
    (U+10D50/U+10D70) are two files on Python 3.9 to 3.13 and one on 3.14 and in the port: "2 files changed." is
    VERIFIED on `main`'s Python 3.12 and CONTRADICTED in its port; here both abstain (K-2). A single such path is
    still one key per runtime, and a claim naming its other case resolves on some runtimes only.
12. **The port's path template reads ASCII `\w`**: "Edited Docs/<U+A7D0>/a.md." is that path in the Python and
    `/a.md` in the port, and a claim naming a directory the diff does not hold verifies by basename (`diff status 'M'
    for 'docs/a.md'`) in both, on `main` and here (K-5 reads it as `main` does). The port's templates read the
    description with JavaScript's `\s`, `\w` and `\b` generally (the README counts them).
13. Carried: `tests/test_ledger.py` leaves `papers/LEDGER.md` rewritten with CRLF in an LF checkout on Windows;
    `gate_diff` with rename detection on reads a rename as `R` where the rendered text reads `M`.

---

## Protocol changes the operator is asked to accept

- **K-1**: the licences are #97's exact and suffix tiers, #121's dotted key and #101's pairing; W-1's exact hunk is no
  longer one, and a file list `main` read from an exact hunk's content abstains.
- **K-2**: Y-1's collision reads one fold, generated once from Unicode 16.0.0 and carried in both ports, rather than
  each runtime's lower-casing.
- **K-4**: beside #121's dotted key, a line no reading places is a doubt `main`'s reading held.
- **K-5**: a path claim the two ports' templates may read apart reads as `main` read it.
- **The scorer**: `compat_violations` without its guard; the baseline-raises records scored whole; the door canaries in
  both modes and the git door's U+0085/U+2028 paths; the parse oracles before an eligibility move; compat claims at the
  git door; the report compared; the provenance widened. Each is a blocking gate of `path2_gates.py`, in both modes.
