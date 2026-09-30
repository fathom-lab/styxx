# NOTE — PATH-2, fourteenth pass: a name's own CR, TAB and backslash, a reason read from the name as written, and the scorer's own blind spots

2026-09-29. Branch `fix/diffgate-path-resolution` (pull request #161). The pushed head is `15878ab5`; the twelfth and
thirteenth passes sit on it, not pushed, with `origin/main` `a4732c52` merged in at `76e65596`. This round reviewed the
thirteenth pass's head, `e1babaac`. The guard's references are unchanged: `styxx/_diffgate_ref.py` is `main`'s
`styxx/diffgate.py` (sha256 `9b620e00…`, LF) and `web/gate/diffgate_ref.js` is `main`'s `web/gate/diffgate.js`
(sha256 `06688702…`).

This note sits on top of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third to
thirteenth-pass notes. **None of them is edited.** Where one is wrong or incomplete, the correction is here (section I).

**How this round is worked.** As in the eleventh to thirteenth passes: this note is written before the code and
committed alone, ahead of it. It states round 13's findings with their evidence and the disposition of each, the rules
this pass adds and why, and the scorer and test changes. The measurements are taken on the commits that follow and are
recorded in `web/gate/README.md` and the CHANGELOG entry, not here.

The merge bar is unchanged: (1) the #97, #121 and #101 reproductions are fixed; (2) no claim reads worse than
`origin/main` on any door — the raw text, the JavaScript port, the git door — under any supported runtime, **judged
against truth** (CPython's parser for definitions; git's `--name-status`, including `T` and mode changes, for file
lists), and no new Python/JavaScript disagreement, an identical abstention allowed; (3) the scorer
(`path2_gates.py`) and the committed tests cannot admit a planted defect.

**Disk.** The shared drive held 87 to 107 MB free when this round began, under the task's 300 MB line for generated
case data. This pass writes no case files: every set is generated in memory against one small bare repository per set,
deleted after its run, as round 13's reviewers did.

---

## The round-13 evidence and its disposition

Four reviews of `e1babaac`. Each finding is listed with the disposition this pass gives it; "fixed" means a rule or a
test below, "disclosed" means stated here and in the README and not changed.

### The regressions lens (verdict: fix before merging; three blockers, one major, one minor)

The reviewer's in-memory harness (`r13/rv13`: `e2e14.py`, `e2e14g.py`, `repro14.py`), models built with
`git fast-import` and judged by git's `--name-status`, three doors, Python 3.12.10 and 3.14.2, Node 24.13.0. Four fresh
families of 1,000 models each (names ending in CR or another control or separator character; names ending in a TAB whose
path holds a space; names holding a backslash; a fuzz family of one or two such characters at the end, the start of a
name or in a directory), each model beside an earlier basename twin, read as git's own bytes and as `difflib` with and
without `a/`/`b/` and with dates or GNU per file; and round 12's `t97`, `t121` and `mix` families rendered by git itself
under 24 option sets. 19,600 cases. **617 claim cells worse than `main` on the raw door and 617 in the port, under each
Python; 0 at the git door; 0 new Python/port disagreements.** Every worse cell is a `file_created` or `file_deleted`
claim in a `difflib` rendering without dates, licensed by #97.

- **R13.1 (blocker) — a name ending in a CR.** `_read_diff` splits the text on `\r\n`, `\r` and `\n` before the header
  path is read, so `+++ b/lib/x.py\r\n` (a file `lib/x.py<CR>`, which `difflib` prints as it is) reads as `lib/x.py`,
  and the form #97 compares equals the claim "Created lib/x.py.". `main`'s `str.splitlines()` reads the same break, so
  no Z-3 doubt fires; `main`'s loop takes an earlier basename twin and abstains; #97's precondition holds; the false
  VERIFIED is kept. Raw door and port, both Pythons; the git door is unaffected (git quotes the name). The scorer's
  `git_lines` drops the CR the same way. *Disposition: fixed, A.1.*
- **R13.2 (blocker) — a TAB after a name holding a space, in a plain rendering.** `_path_as_written` cut "the one TAB,
  at the end, after a name holding a space" as git's terminating TAB in every rendering. git quotes a name holding a
  TAB, so in a plain rendering a trailing TAB is the name's own: `+++ b/sp ace/lib/x.py<TAB>` is cut to
  `sp ace/lib/x.py`, which ends in `/lib/x.py`, and #97's suffix tier licenses "Created lib/x.py.". The thirteenth
  pass's TAB rule (`414305af`) covered only names without a space. The port's `_pathAsWritten` and the scorer's
  `own_as_written` cut alike. *Disposition: fixed, A.2.* This pass found a companion in the same function: the GNU
  rule ("the one TAB, with a character other than a space after it") also cuts a TAB **inside** a name —
  `+++ b/lib/x.py<TAB>foo/lib/x.py`, one file created, whose key ends in `/lib/x.py`, is cut to `lib/x.py` and
  "Created lib/x.py." reads VERIFIED where `main` abstains (checked at `e1babaac` on the raw door). A.2 closes both.
- **R13.3 (blocker) — a name holding a backslash.** `_case_kept` turned backslashes into slashes, so the form of the
  file `lib\x.py` equals a claim naming `lib/x.py`, a path that does not exist. The key does the same (as `main`'s
  does); `main`'s loop takes the earlier twin, #97's exact tier lands on the backslash entry, and the licence holds.
  The port's `_caseKept` and the scorer's `own_case_kept` do the same. *Disposition: fixed, A.3.*
- **R13.4 (major) — the scorer carries the same three blind spots.** `git_lines` drops the CR, `own_as_written` cuts
  the plain rendering's TAB, `own_case_kept` turns backslashes into slashes, so G-C7 and G-C9 compare the branch with an
  identical rule. No canary and no pinned pair held any of the three shapes. *Disposition: fixed, C.1 and C.2.*
- **R13.5 (minor) — "0 claims worse" was measured on generators without the three shapes.** gen13's whitespace list
  has no CR; no generator puts a space elsewhere in a name ending in a TAB, or writes a backslash into a path.
  *Disposition: corrected here (section I) and scoped in the README and the CHANGELOG; the three shapes join the
  regression sets (section G).*

The reviewer also confirmed that #101's guard path is inert (`_pairing_withdraws(chg) = chg > 0`, so an all-on
`tests_added` verdict other than UNCHECKABLE equals the #101-off one), that `tests/test_diffgate_guard.py` passed 114 and
the path2 reproduction subset 77 (1 skipped), and that the #97, #121 and #101 reproductions and round 12's 34 stay fixed.

### The guard, scorer and protocol lens (verdict: fix before merging; two blockers, three minors)

Plants applied in memory (the reviewer's pytest plugin and `node --require` hook serve the planted bytes to every read,
so the scorer's `NEW`/`CF` and sha256 are the planted ones and G-C0 stays clean, as after a commit).

- **G13.1 (blocker) — #121's `multi` rule asked of the claim's key.** Plant N3b (`p in multi` becomes
  `_norm(d["path"]) in multi`), N3b′ (the same defect as a rebinding) and the port's JN3b pass the canaries, the 3,565
  differential records, the 389 pinned pairs and every committed test but a source-text count. Live: git's rendering of
  `.cfg/x.json` turned symlink beside `cfg/x.json` modified; "Created x.json." (truth F) — `main` and the clean branch
  UNCHECKABLE, N3b VERIFIED on the raw door and in the port. The same for "Deleted x.json." where `.cfg/x.json` goes
  symlink to empty file. *Disposition: fixed, C.3 (a raw canary, a pinned pair, a truth-judged fixture, the plants).*
- **G13.2 (blocker) — a licence from one repair's precondition and another repair's switch.** Plant N1
  (`any(pre(r) and switch(r) == mv)` becomes `any(pre(r)) and any(switch(r) == mv)`), N1′ and the port's JN1 pass
  everything. Live: git's rendering with renames (`docs/ci.yml` modified, `old.py` renamed to `new.py` with an edit,
  `p/.github/ci.yml` and `x/github/ci.yml` created), "Created .github/ci.yml.": `main` and the clean branch
  UNCHECKABLE, N1 VERIFIED. #97 switched off gives `main`'s verdict back but #97 cannot license beside the rename's Z-3
  doubt; #121's precondition holds but #121 switched off does not give `main`'s verdict. Truth there is `?`, so no
  truth-false case was found; it is round 12's PA class, a guard rule the scorer could not see broken.
  *Disposition: fixed, C.4 (a raw canary, a stub test in both ports, the plants).*
- **G13.3 (minor) — corpus mode admits partial drops.** N2 (#97's `multi` check on created claims only), N2b (only
  where the entry reads `A`), N3 (#97's `multi` asked of the claim's key), N5 (#121's exact tier compared case-folded),
  N23c (the form drops a trailing NBSP or U+3000), N24 (a deletion's form from the stripped `---` path) and N26 (a
  section flushed without its pair not counted) pass the canaries; each is refused only by differential mode's pinned
  pairs and the committed truth tests. *Disposition: fixed, C.5 (five pinned records promoted to canaries, the seven
  plants committed).*
- **G13.4 (minor) — the scorer's git door cannot build a typechange.** `GitDoor.commit` writes every entry as `100644`,
  so the git door's `T` and mode rule is outside G-C7 in both modes; the thirteenth note's C.3 says it is held by G-C7's
  facts. N10 (moded keys dropped from the git door's `multi`) and N10b (the `T` letter dropped) are refused only by
  `tests/test_diffgate_guard.py::test_the_git_door_licence_facts_read_gits_typechange_and_the_scorers_own`.
  *Disposition: corrected here (section I) and stated in the README: G-C7 does not see `T` or a mode change; the rule
  is held by that test, whose scorer-free assertions move into a test of their own (D.3) so every CI Python runs them.*
- **G13.5 (minor) — the plant tests are anchored to source text.** `d5ea37f1` edited a code comment
  (`# A / M / D / R / T` to `# A / M / D / R`) to keep a plant string unique, and the comment now misdescribes git's
  letters; an in-place edit of a guarded line fails the suite whether or not the edit is refused on behaviour.
  *Disposition: fixed — the comment reads git's letters again (`A / M / D / R / C / T`) and the plant anchors on the code
  alone; the README describes the X12 and X14 plant tests as refusals of those exact texts, the behavioural coverage
  being the canaries, the pinned pairs and the truth fixtures.*

The reviewer confirmed the frozen documents byte-identical, each NOTE touched by one commit, the thirteenth note alone
in its commit, `build_bookmarklet.py --check`, `check_pairs.js` (389 pairs, 0 disagreements), 504 tests passing, and
the truth-judged tests not circular (a constant precondition fails 39 cases; `_truth_status` replaced by the branch's
own parse fails 8 tests).

### The runtime and Unicode lens (verdict: fix before merging; one major, five minors)

- **U13.1 (major) — Z-5's reason prints the runtime's key.** `_refused_why` prints `_shown(key)`, and a key is the
  runtime's lower case. The doubt of section D (thirteenth pass) gates the file-list claims, not `tests_added` or
  `symbol_added`, so where a path holds a code point Unicode 16.0.0 does not assign, a runtime on 17.0 prints another
  path (U+A7CE lowered to U+A7CF) than one on 16.0: a new Python/port reason disagreement on CI's own pairing (Python
  3.9 to 3.12 beside a Node on 17.0), where `main`'s reasons agree. Verdicts agree. *Disposition: fixed, B.*
- **U13.2 (minor) — `pyRepr` asks the engine which characters to escape.** Where the Python is newer than the
  JavaScript engine (3.14 beside an engine on 15.1 or older; 3.15 beside Node 24 for 17.0 code points in reasons the
  doubt does not gate), a branch reason printing a path with a code point the engine does not know reads apart while
  `main`'s port, which escapes no non-ASCII, agrees. Verdicts are unaffected; on the lab's and CI's pairings (the Python
  no newer than the engine) the difference is `main`'s own or absent. *Disposition: disclosed, beside the fifth pass's
  V-2 and the 3.14 `test_x1` reason difference.*
- **U13.3 (minor) — the CHANGELOG names a superseded bookmarklet.** It says `830b4ba7…`, 79,217 characters (the
  twelfth-pass build the README calls defective); the thirteenth build was `51f06338…`, 83,768. *Disposition: fixed
  with this pass's rebuild; section I.*
- **U13.4 (minor) — `test_x10_k2` does not hold the doubt.** Removing the unassigned-set doubt fails only the
  `test_x13_d` tests, never `test_x10_k2`, whose "beyond is unassigned" assertion is vacuous on every lab runtime. The
  doubt is held by the `test_x13_d` tests, pinned pairs 333/334 and the canaries. The 28 CI code points abstain in both
  ports; the port's assigned set equals the Python's on all 1,114,112 code points. *Disposition: no change needed.*
- **U13.5 (minor) — CPython 3.15 (Unicode 17.0).** Y-3's skew set, K-5's `_apart_readings` and `_main_names` assume the
  supported Pythons read Unicode 13.0 to 16.0; `requires-python` has no upper bound. Reasoned, not run (no 3.15 here).
  *Disposition: disclosed — PATH-2 states its supported Pythons as 3.9 to 3.14 in the README and the CHANGELOG; 3.15
  is not supported until those sets are read against it.*
- **U13.6 (minor) — the fold's soundness in a browser rests on Unicode's case-pair stability, not on a check.** A
  simulated engine that lower-cases an assigned code point beyond the fold turns a #121 count licence false; the tests
  check the premise only on the runtime they run on. *Disposition: disclosed — the premise is stated in `gen_fold.py`
  and the README; no load-time check.*

### The CI parity and integration lens (verdict: fix before merging; one blocker, four minors)

- **C13.1 (blocker) — the U+2C2F pair on Python 3.9 and 3.10.** `path2:k2-paths-that-fold-apart-read-as-before` pins a
  reason printing a runtime key holding U+2C2F lowered to U+2C5F; both were assigned in Unicode 14.0, and Python 3.9 and
  3.10 carry 13.0, so they neither lower the one nor print either unescaped. `test_port_is_current` compares the
  reason, and fails there (simulated over all 389 pairs: this pair alone). The saved CI log is py3.12's only, so "one
  failure on py3.9-3.12" was never checked against the 3.9 and 3.10 jobs. *Disposition: fixed, D.1; section I.*
- **C13.2 (minor) — the CHANGELOG's pass count.** "ten passes" beside eleven notes. *Disposition: fixed.*
- **C13.3 (minor) — the git door's `T` test takes the `scorer` fixture.** `path2_gates` exits on any Python whose
  Unicode is not 15.0.0, so the test (and 54 scorer tests in `test_diffgate_path2.py`, 5 in `test_diffgate_guard.py`)
  run only in the py3.12 job. *Disposition: fixed, D.3, and the README names which refusals only that job enforces.*
- **C13.4 (minor) — the plant helpers read the instrument's CRLF bytes on a Windows clone.** `_planted` and
  `_planted_many` count newline-bearing snippets in the raw bytes. *Disposition: fixed, D.2.*
- **C13.5 (minor) — `moved` unused** (`tests/test_diffgate_guard.py`). *Disposition: fixed.*

---

## A. The licences: a name's own CR, TAB and backslash

The forms #97 compares (and #121's tier-kept path check) are each header path **as written**. Round 13 found three
characters the reader still took for something else. Each rule below is a structural fact of the reading, not a list
of names.

### A.1 A header line's CR is the name's, unless the text ends every line in CRLF

The reader keeps each line's terminator. A `---`/`+++` line whose terminator holds a CR (a lone `\r`, or `\r\n`)
keeps a CR at the end of its path as written, **unless every line of the text ends in `\r\n`** — a text converted to
CRLF throughout, where the CR is the line ending. git quotes a name holding a CR, so in git's own rendering such a CR
is always the text's; `difflib` and GNU diff print it as it is. A CRLF text whose name ends in a CR reads `\r\r\n`: the
split sees a lone `\r` and then an empty line, so the text is not CRLF throughout and the CR stays in the form. The
key, the status map, the Y-1 fold and `main`'s reading are unchanged (both readers split on the CR).

### A.2 Only git's own terminating TAB is cut

A TAB in a `---`/`+++` path is cut as a terminator **only where git wrote it**: the line sits under a `diff --git`
header, the path before the TAB is the path that header writes for that side (`_as_git_writes`, quoted as git quotes
it), that path holds a space, and nothing follows the TAB. Every other TAB stays in the form: a name ending in a TAB
(`difflib` prints it as it is), a TAB inside a name (the GNU rule's companion above), and GNU's or `difflib`'s TAB
before a date. The key already holds such a date (`main`'s did too), so no path claim resolves there by the exact or
suffix tier and #97 licensed nothing there before; the rule costs no licensed verdict in those renderings. A.1 applies
after the cut.

### A.3 A backslash is a character of the name

The forms keep backslashes as written (the key still reads them as slashes, as `main`'s key does). **#97 and #121
license no path claim whose claim holds a backslash or whose resolved entry has a form holding one**: git quotes a
backslash, so such a form comes from a plain rendering of a name holding one, and a claim written with one may mean
either path. (The basename tier's `pathlib` name also reads a backslash as a separator on Windows only; with no
backslash reaching it the comparison is the same on every platform.)

### A.4 Tightening a licence can only turn a verdict into an abstention

As in the twelfth and thirteenth passes: A.1 to A.3 change what a precondition reads, one conjunct of a licence, where a
decided claim's verdict differs from the reference's. A claim that loses its licence becomes UNCHECKABLE, naming the
reference's verdict; every other claim is untouched.

---

## B. Z-5's reason, read from the name as written

The key a runtime makes is its own lower case, so a reason that prints a key can differ between runtimes wherever the
fold does not fix it. Z-5 now prints the file as the diff writes it: **the earliest header path registered for that
key, before the runtime lowers it** (Y-1's form: backslashes as slashes, a leading run of `/` and `./` segments
dropped), folded by the one table, each code point Unicode 16.0.0 does not assign printed as U+FFFD, escaped by
`ascii()`. For a path whose code points are all assigned this is exactly the text printed before (the fold of a key is
the fold of the path it was read from, the K-2 property the tests hold on every runtime), so no pinned reason moves.
For a path holding an unassigned code point it is the same text on every runtime: the path is the diff's own text, and
two paths a newer runtime merges differ only at unassigned code points, which print alike. The reading records this
form per key (`keyed`) beside the other licence facts, in both ports and in the scorer's own reading, and G-C7 compares
it. Where a key has no such form (a switched-off reading's key), the reason prints `_shown(key)` as before.

---

## C. The scorer

1. **The scorer's own code** reads A.1 to A.3 and B, written out independently: its own line split keeps each line's
   terminator; `own_as_written` cuts only git's TAB under git's header and keeps a name's CR; `own_case_kept` keeps
   backslashes; `own_precondition` refuses a backslash claim or form; `own_read` records `keyed`; the Z-5 oracle prints
   from it.
2. **Canaries for R13.1 to R13.3.** Raw canaries (the three shapes cannot be rebuilt as safe paths at the git door),
   each where the thirteenth pass's rule keeps a false VERIFIED: a name ending in a CR, created and deleted, in
   `difflib`'s rendering; a trailing TAB after a name holding a space; a TAB inside a name; a name holding a backslash.
   Each current rule restored is committed as a plant and refused by the canaries.
3. **G13.1.** A raw canary carrying git's own bytes for `.cfg/x.json` turned symlink beside `cfg/x.json` modified,
   "Created x.json. Deleted x.json." (a door canary cannot hold it: the scorer's git door writes only `100644`); the
   same case pinned in `path2_pairs.json`, and its moded model in the truth fixtures (raw door, port, git door); N3b,
   N3b′ and the port's JN3b committed as plants.
4. **G13.2.** A raw canary carrying git's rename rendering above, "Created .github/ci.yml."; a stub test in both ports
   where one repair's switch gives `main` back without its precondition and another's precondition holds without its
   switch, and the guard must abstain; N1, N1′ and JN1 committed as plants.
5. **G13.3.** The pinned records `path2:m-r12-T6`, `path2:m-r12-T3`, `path2:m-r12-W-nbsp-created-difflib`,
   `path2:m-r12-W-sp-deleted-difflib-noprefix` and `path2:m-d12-em-2909292-786-git` join `RAW_CANARIES`, and N2, N2b,
   N3, N5, N23c, N24 and N26 are committed as plants the canaries refuse.
6. **G13.4.** G-C7 at the git door does not see `T` or a mode change (the scorer's git door writes `100644` only); the
   rule is held by the committed test, run on every CI Python (D.3).
7. The `NOTE` list runs through this pass.

---

## D. CI

1. **C13.1.** The U+2C2F pair is re-pinned to (kind, verdict) claims, which read alike on Unicode 13.0 to 17.0; its
   verdict test (`test_x10_k2_paths_that_fold_apart_read_as_before`) is unchanged. No other pinned reason prints a code
   point assigned after 13.0.
2. **C13.4.** `_planted` and `_planted_many` read the instrument's bytes with `\r\n` normalised to `\n`, as
   `py_side.py` and `path2_gates._sha` already do.
3. **C13.3.** The git door's `T` and mode assertions that need no scorer (`multi` for the typechange and mode-change
   models, and the `diff.submodule=log` repository) move into a test of their own without the `scorer` fixture.
4. **C13.5.** `moved` dropped.

---

## E. Runtimes

U13.2, U13.5 and U13.6 are disclosed as stated above (README, CHANGELOG, `gen_fold.py`). U13.4 needs no change.

---

## F. The truth model and the reproductions

Round 13's reproductions — R13.1 to R13.3, created and deleted, with and without `a/`/`b/`, the TAB-inside companion,
and G13.1's typechanged dotted directory — are pinned with their models in a fixture of their own and judged on the raw
door, in the port and at the git door by the committed truth model. **Calibration**: the same check must read them worse
than `main` at `e1babaac` (a count pinned in the test), and none at this pass's head.

---

## G. The truth-judged regression differential

`origin/main` `a4732c52` against this pass's head, three doors, Python 3.12 and 3.14, Node 24.13.0, 30,000 cases or
more, in memory: round 13's four families (`cr`, `tabsp`, `bs`, `fz`) at fresh seeds, a TAB-inside family and a
CRLF-converted rendering of each, git's own renderings of round 12's typechange families under the option sets round 13
used, and the builder's earlier families (`gen13`'s whitespace, `gen12`/`gen12L`, `gen10`, `gen_rand`) regenerated from
seeds, iterated to **0 claims worse against truth and 0 new Python/JavaScript disagreements**. The numbers and the
recall cost are recorded after the code.

---

## H. The recall cost this round expects

- **A.1**: path claims #97 or #121 licensed on a header line ending in a CR, in a text not CRLF throughout (a
  CRLF-converted text holding content CRs included), abstain.
- **A.2**: path claims licensed on a name whose header path holds a TAB other than git's terminator abstain; in GNU's
  and dated `difflib` renderings no claim was licensed there before.
- **A.3**: path claims licensed where the claim or the entry's header path holds a backslash abstain.
- **B**: no verdict moves; a Z-5 reason moves only for a path holding a code point Unicode 16.0.0 does not assign.
- Nothing else moves: A.4.

---

## I. Corrections to earlier notes and records

1. **Thirteenth pass, section A.2**: a name whose whitespace `strip()` drops "never equals the claim, in **every**
   rendering". False for a CR (split as a line end before the path was read, R13.1) and for a TAB after a name holding
   a space (cut in every rendering, R13.2); and a backslash was read as a slash (R13.3). A.1 to A.3 above.
2. **Thirteenth pass, section C.3**: "The git door's `T` and mode rule ... is held by G-C7's facts." The scorer's git
   door writes only `100644`, so G-C7 never sees a `T` or a mode change; the rule is held by the committed test (C.6).
3. **The CHANGELOG, thirteenth pass**: "repaired in ten passes" beside eleven notes (third to thirteenth).
4. **The CHANGELOG, thirteenth pass**: the bookmarklet named as `830b4ba7…`, 79,217 characters; the thirteenth build
   was `51f06338…`, 83,768 characters.
5. **`web/gate/README.md` and the CHANGELOG, thirteenth pass**: "0 claims worse than on main, judged against truth" was
   measured on generators that never produce a name ending in a CR, a TAB-terminated name holding a space, or a
   backslash name.
6. **Thirteenth pass, "CI on `15878ab5`"**: "one failure, on Python 3.9 to 3.12" was read from the py3.12 job's log
   alone. By simulation (not a CI log) the py3.9 and py3.10 jobs also fail `test_port_is_current` on the U+2C2F pair
   (C13.1).
7. **`_shown`'s docstring (tenth pass)**: a key printed through it is "the same on every runtime". Not for Z-5 on a
   runtime newer than the fold's Unicode (U13.1); B.

---

## Protocol changes the operator is asked to accept

- **A header line's CR is the name's, unless the text ends every line in CRLF** (A.1).
- **Only git's own terminating TAB is cut** (A.2).
- **A backslash is a character of the name; #97 and #121 license no path claim a backslash touches** (A.3).
- **Z-5 prints the file as the diff writes it** (B), recorded as `keyed` beside the licence facts.
- **The scorer**: its own code for A.1 to A.3 and B; raw canaries for the three shapes, G13.1's typechanged dotted
  directory and G13.2's mixed licence; five pinned records promoted to canaries; N1, N1′, N3b, N3b′, N2, N2b, N3, N5,
  N23c, N24, N26 and the three restored rules as plants; G-C7 stated not to see `T` or mode changes (C).
- **CI**: the U+2C2F pair re-pinned without its reason; the plant helpers LF-normalised; the git door's `T` assertions
  run without the scorer (D).
- **Supported Pythons for PATH-2: 3.9 to 3.14** (U13.5), and the fold's premise in a browser stated (U13.6).
