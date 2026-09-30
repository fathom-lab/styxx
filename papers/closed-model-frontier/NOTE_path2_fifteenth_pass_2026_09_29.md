# NOTE — PATH-2, fifteenth pass: #97's and #121's licences withdrawn, a file's lines read alone only where no runtime can merge it, and the scorer and the tests that hold both

2026-09-29. Branch `fix/diffgate-path-resolution` (pull request #161). The pushed head is `15878ab5`; the twelfth to
fourteenth passes sit on it, not pushed, with `origin/main` `a4732c52` merged in at `76e65596`. This round reviewed the
fourteenth pass's head, `340ddfb6`. The guard's references are unchanged: `styxx/_diffgate_ref.py` is `main`'s
`styxx/diffgate.py` (sha256 `9b620e00…`, LF) and `web/gate/diffgate_ref.js` is `main`'s `web/gate/diffgate.js`
(sha256 `06688702…`).

This note sits on top of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third to
fourteenth-pass notes. **None of them is edited.** Where one is wrong or incomplete, the correction is here (section I).

**How this round is worked.** As in the eleventh to fourteenth passes: this note is written before the code and
committed alone, ahead of it. It states round 14's findings with their evidence and the disposition of each, the rules
this pass adds or withdraws and why, and the scorer and test changes. The measurements are taken on the commits that
follow and are recorded in `web/gate/README.md` and the CHANGELOG entry, not here.

The merge bar is unchanged: (1) the #97, #121 and #101 reproductions are fixed; (2) no claim reads worse than
`origin/main` on any door — the raw text, the JavaScript port, the git door — under any supported runtime, **judged
against truth** (CPython's parser for definitions; git's `--name-status`, including `T` and mode changes, for file
lists), and no new Python/JavaScript disagreement, an identical abstention allowed; (3) the scorer
(`path2_gates.py`) and the committed tests cannot admit a planted defect.

**The operator's backstop for convergence**, given with this round's task: a licence (#97, #121 or #101) that the
previous review found still producing a truth-judged regression against `main` after two tightenings is not tightened
again; it is **withdrawn**. The repair's differing verdict then always becomes UNCHECKABLE, showing `main`'s reading;
withdrawing a licence can only turn a verdict into an abstention. The withdrawal is recorded here, in the CHANGELOG and
in the README with the recall it costs, and the repair's code and switch stay so that a later reviewed pull request can
license it again. Scorer, guard, Unicode and documentation findings are fixed as usual.

**Disk.** The shared drive held 89 MB free when this round began, under the task's 300 MB line for generated case data.
So this pass generates no case sets: its checks are the committed tests, the pinned pairs, the canaries, the checked-in
corpora, and the reviewers' own small reproductions, each in memory against one small bare repository deleted after its
run. The 30,000-case regression differential the task asks for is **not run** by this pass; section G says what stands
in its place and why the withdrawal makes the bar's second clause hold by construction.

---

## The round-14 evidence and its disposition

Four reviews of `340ddfb6`. Each finding is listed with the disposition this pass gives it; "fixed" means a rule or a
test below, "withdrawn" means the backstop above, "disclosed" means stated here and in the README and not changed.

### The regressions lens (verdict: fix before merging; four blockers, three minors)

The reviewer's targeted probes (`r13/rv14`: `lic.py` with `cases.py`, `mc.py` with `mc_cases.py`, `pfx.py`, `c101.py`),
models built with `git fast-import` in small bare repositories deleted after each run, judged by git's `--name-status`
(the net diff for multi-commit renderings), raw door and port, Python 3.12.10 and 3.14.2, Node 24.13.0. About 80 cases;
the disk (94 MB falling to 41 MB) stopped the scorer, the committed suites and a fresh 30,000-case set.

- **R14.1 (blocker, #97) — a name holding an LF followed by `@@…`, in `difflib`'s rendering.** Base
  `{a/x.py}`, head `{a/x.py, "lib/x.py\n@@"}`; `difflib` prints `+++ b/lib/x.py`, a line break, `@@`, then the real hunk.
  The line split cuts the name at the LF; `_clean_header` and `_shaped_pair` test `lines[k+2].startswith("@@")`, which the
  tail line passes, so no Z-3 doubt; the form `lib/x.py` equals the claim; `main`'s loop takes `a/x.py` by basename and
  abstains; #97's precondition holds and "Created lib/x.py." (truth F) reads VERIFIED on the raw door and in the port,
  under both Pythons. 8 of 16 LF probes read worse (the tails `@@`, `@@ -0,0 +1 @@`, `@@x`, `@@ notes`, with and without
  `a/`/`b/`, all creations); 0 new Python/port disagreements; git's own rendering quotes the name. The scorer's
  `own_line_ends` shares the split (by reading). *Disposition: withdrawn (#97, section A); pinned as a truth-judged
  reproduction (section F).*
- **R14.2 (blocker, #97) — a file created in one commit and renamed away in a later one, in `git log -p --format=`.**
  `multi` counts the sections that register a key; a rename section registers only its `b` side and an R100 rename has
  no `---`/`+++` lines, so the created path is registered once, `rendered` holds and no Z-3 doubt is read; #97's exact
  (and, in `mc97-suffix`, suffix) tier keeps `A` for a path the net diff never holds ("Created src/x.py.", truth F):
  VERIFIED on the raw door and in the port, both Pythons, with `-M`, and in commit order (`--reverse`). 8 worse renderings;
  0 new disagreements; `format-patch` and default `git log -p` abstain (their header lines are a Z-3 doubt),
  `--no-renames` makes the key `multi`. The scorer's `multi` has the same blind spot (by reading). *Disposition:
  withdrawn (#97); pinned with git's bytes.*
- **R14.3 (blocker, #121) — R14.2's mechanism under #121.** `.cfg/x.json` modified, `cfg/x.json` created and then renamed
  to `cfg/z.json`; on `git log -p --format=` (and `-M`) "Created cfg/x.json." (truth F): `main` abstains (its merged key
  reads the twin's later status), the branch keeps `A` with a dotted key present: VERIFIED on the raw door and in the
  port, both Pythons. 2 worse renderings. *Disposition: withdrawn (#121); pinned with git's bytes.*
- **R14.4 (blocker, #121) — git's `--src-prefix=a/.. --dst-prefix=b/..`.** Every key reads `..cfg/…`; `_dot_miss` drops
  one leading dot, so `..cfg/x.json` is a real outside path and "Only touched cfg/." (truth T) reads CONTRADICTED where
  `main` read VERIFIED; #121's precondition holds and its switch gives `main` back. The bytes equal a default-prefix diff
  of real `..cfg/` files, where the branch is right. *Disposition: withdrawn (#121); pinned with git's bytes. The
  branch's final verdict there becomes UNCHECKABLE — a recall cost against `main`'s true VERIFIED, not a worse verdict.*
- **R14.5 (minor) — the fourteenth pass's "0 WORSE on every door" was measured over 17 sets holding no LF-in-name
  rendering and no multi-commit rendering.** *Disposition: corrected here (section I) and in the README and the
  CHANGELOG; under the withdrawal the zero no longer rests on the sets measured (section G).*
- **R14.6 (minor, #101) — #101 is inert against truth; it costs recall where a test moves between classes or from a class
  to module level** ("Added 1 test.", truth T: `main` VERIFIED, the branch UNCHECKABLE). 0 worse. *Disposition: disclosed
  in the recall ledger (README).*
- **R14.7 (minor, scorer) — the scorer's own licence model shares R14.1's split and R14.2's `multi`,** so G-C9 would have
  counted R14.1 to R14.3 as licensed. *Disposition: the scorer's guard withdraws #97 and #121 too (section C.1), so a kept
  verdict only they explain is a G-C9 violation whatever the preconditions read; the shared blind spots now bear only on
  which reason an abstention prints (section A.2), where the scorer and the instrument read alike. Disclosed as work a
  pull request re-licensing either repair must do before it (section A.3).*

### The guard, scorer and protocol lens (verdict: fix before merging; one blocker, three minors)

Plants compiled in memory (`r13/rv14g`: `plants.py`, `plantplug.py`, `jsplants.py`); the canaries (47 records) standing
for corpus mode and the 359 path2 pinned pairs for differential mode. The reviewer confirmed the frozen documents
byte-identical, the fourteenth note alone in its commit, the bookmarklet check, `check_pairs.js` (413 pairs, 0
disagreements), `test_ledger` 3, `test_diffgate_guard` 136 and `test_diffgate_path2` 416 passed, every committed X12 and
X14 plant refused, the reviewer's own Python and port plants refused, and the truth tests not circular in the
precondition (a constant precondition fails 69 of 77 raw-door truth items).

- **G14.1 (blocker) — #121's backslash refusal is checked by nothing.** Plant P1 (`if _backslashed(kept, forms) or
  _tabbed(forms):` becomes `if _tabbed(forms):` in `_kept_by_its_tier`) and the port's JP1 pass the canaries, the pinned
  pairs, `check_pairs.js` and the committed tests; live on "Created .a\x.py." over git's header shape holding an unquoted
  backslash (`main` and the clean branch UNCHECKABLE, P1 VERIFIED). No git-judged truth regression is possible (git
  quotes a backslash). *Disposition: fixed. With #121 withdrawn P1 can no longer keep a verdict, but it still changes the
  abstention's reason (section A.2), which G-C9 compares; the reviewer's two raw canaries, a pinned pair, P1 and JP1 as
  plants (section C.3).*
- **G14.2 (minor) — corpus mode admits R13.2's rule restored on the `---` side only** (plant P6, port JP6); refused only
  by the pinned pairs `path2:m-r13-R13.2-tabsp-deleted` and `-noprefix` and by the truth tests. *Disposition: fixed — the
  record is promoted to `RAW_CANARIES` (its G-C7 licence-fact comparison refuses P6 in corpus mode) and P6 committed as a
  plant.*
- **G14.3 (minor) — corpus mode admits `keyed` keeping the latest path for a key** (plant P11); refused only by
  differential mode's G-C7 licence-fact comparison on five pinned pairs. *Disposition: fixed — one of those records
  (`path2:y1-two-paths-that-differ-only-in-case-are-one-key`) is promoted to `RAW_CANARIES` and P11 committed as a plant.
  (Section B makes a key two different paths share abstain the per-file readings before Z-5, so Z-5's reason can no
  longer print from such a key; the fact is still compared.)*
- **G14.4 (minor) — the truth-judged tests read the claimed path from the branch's own final claim.** A reading that
  rewrote a claim's detail path to the entry it resolved would pass them. *Disposition: fixed — truth is judged from
  `main`'s claim where `main` makes it, and the tests assert the branch's detail equals `main`'s for every paired claim
  truth reads (section F).*

### The runtime and Unicode lens (verdict: fix before merging; two majors, five minors)

Evidence in `r13/rv14U` (`perfile.py`, `dbg101.py`, `emoji.py`, `future.py`, `gen_ut.py`, `mutate_k2.py`); a Node on
Unicode 17.0 simulated by patching `toLowerCase` and `process.versions.unicode`.

- **U14.1 (major) — the per-file readings key by the runtime's lower case.** Z-5's refused files, #101's pairing
  (`_changed_test_defs`, `_definition_only_changed`) and Y-5 read `sides`, whose keys are the runtime's `lower()`; Y-1's
  fold and unassigned doubts gate only file-list and path claims. Where two header paths fold alike, one runtime merges
  their sides and another keeps them apart: new Python/port verdict disagreements on the lab's pairing (Python 3.12.10
  against Node 24.13.0, real: `src/xɤ.py` and `src/xꟋ.py`, "Added function foo.", Python 3.12 VERIFIED, 3.14 and the
  port UNCHECKABLE; and "Added 0 tests." over `b/tꟋ.py` modified beside `b/tɤ.py` created, Python 3.12 UNCHECKABLE, the
  port CONTRADICTED), on CI's (Python 3.9 to 3.12 against a Node on 17.0) and between Python 3.12 and 3.14. `main`'s two
  ports agree on every such input. None is worse by truth. *Disposition: fixed, section B (Y-6).*
- **U14.2 (major) — Z-5's reason for U+A7D2/U+A7D3 and U+A7D4/U+A7D5.** `_shown_written`'s docstring says two paths a
  newer runtime merges differ only at unassigned code points; U+A7D3 and U+A7D5 were assigned in Unicode 14.0, so on a
  Node on 17.0 the merged key's earliest path prints `x�.py` where Python 3.9 to 3.12 print `xꟓ.py` (or the
  reverse, by diff order). Both UNCHECKABLE. *Disposition: fixed by section B (a path holding an unassigned code point
  abstains the per-file readings before Z-5 reads a file); the docstring is corrected; the simulated-17 test gains both
  pairs, both orders.*
- **U14.3 (minor) — a changed test beside its ASCII case twin reads as added, by diff order** (`b/tA.py` modified,
  `b/ta.py` created, "Added 0 tests.": both ports keep `main`'s false CONTRADICTED; reversed, both abstain). Not worse
  than `main`. *Disposition: fixed by section B (the two paths fold alike).*
- **U14.4 (minor) — K-5's docstring says symbols and emoji read alike in both ports' templates.** Inside the bounded
  window `_W` the port counts UTF-16 code units and Python code points, so 31 or more astral characters between a verb and
  a path part the two extractions; `main` has the same asymmetry. *Disposition: the docstring is corrected; disclosed.*
- **U14.5 (minor) — the per-call cost in the README is the eleventh pass's.** Measured at `340ddfb6`: Python mean 6.17 ms
  (8.9 times `main`), port 1.29 ms (6.8 times). *Disposition: re-measured at this pass's head and stated.*
- **U14.6 (minor) — the href bookmarklet is 84,309 characters; Firefox is understood to refuse a bookmark URL over
  65,536** (its Places limit), where `main`'s 24,346 fits. Not exercised in a browser. *Disposition: disclosed in the
  README, stated as unverified here.*
- **U14.7 (minor) — `test_x10_k2` does not hold the unassigned doubt** (confirms U13.4). *Disposition: `test_x10_k2`
  gains an assertion that both readings raise the unassigned doubt for the 28 code points on every engine.*

### The CI parity and integration lens (verdict: ship; six minors)

- **C14.1** — the CHANGELOG's "Eleven rounds of repair, ten of them answering an adversarial review round" beside
  "twelve passes". *Disposition: fixed; section I.*
- **C14.2** — the CHANGELOG's "under 24 option sets" where three of 24 were run (3,600 renderings). *Disposition: fixed;
  section I.*
- **C14.3** — the README's "Which Python enforces what" omits a fourth guard-file id that skips on 3.9 to 3.11
  (`test_a_defect_outside_the_three_repairs_can_only_abstain[the count off by one]`). *Disposition: fixed.*
- **C14.4** — `test_x1_the_characters_the_runtimes_disagree_on_read_alike_in_both_ports` fails on Python 3.14 (the U+1C89
  reason, disclosed under "Not closed") and is unmarked. *Disposition: fixed — a strict `xfail` on a Unicode 16.0.0
  Python.*
- **C14.5** — the fold generator's reproduction test runs only on 3.14, outside CI. *Disposition: stated in the README.*
- **C14.6** — the fourteenth pass's builder attributed 3.14's skipped scorer tests to the Unicode refusal; its harness
  skipped them earlier, on the scorer's provenance check. *Disposition: corrected here (section I).*

---

## A. The backstop: #97's and #121's licences withdrawn

### A.1 Why these two

#97's licence was tightened at the twelfth pass (Z-3's doubts, the case kept), the thirteenth (`multi`, the forms as
written) and the fourteenth (a name's CR, TAB and backslash); #121's at the twelfth (git's own rendering, the tier-kept
path licence of `98f74833`), the thirteenth (`multi`) and the fourteenth (a backslash). Round 14 found truth-judged
regressions against `main` licensed by each (R14.1 and R14.2 by #97, R14.3 and R14.4 by #121). Both are withdrawn.
#101 licenses no verdict other than an abstention (its switched-off verdicts differ from its own only as UNCHECKABLE,
the fourteenth pass's reviewer confirmed it inert against truth) and round 14 found no truth-judged regression through
it; it is not withdrawn.

### A.2 The rule

`WITHDRAWN = ("#97", "#121")`, in both ports and, as its own constant, in the scorer. Per claim, the guard reads as
before, except that a difference from `main`'s verdict is kept only where a repair **not withdrawn** explains it (its
precondition holds and it alone switched off gives `main`'s verdict back). Where only a withdrawn repair explains it,
the claim is UNCHECKABLE with its own reason:

> main's reading gives {main} and this one {this}; {repair} explains the difference on this claim, but its licence is
> withdrawn until a reviewed change restores it, so it abstains

naming the earliest withdrawn repair, in `REPAIRS` order, that explains it. Elsewhere the reasons are the eleventh
pass's. The preconditions and switches are unchanged and still run: they decide which of the two reasons an abstention
prints, the scorer re-derives both with its own code (G-C9 compares the reason), and so a defect in a withdrawn
repair's precondition is still refused by the canaries, as a reason, while it can no longer keep a verdict.

What the repairs still do: an abstention is always kept. Where #97's resolution or #121's key reads that `main`'s
verdict was wrong and the reading abstains (a withheld path accusation, a dot miss), `main`'s false verdict is gone.
What is withdrawn is a **decided** verdict that differs from `main`'s.

### A.3 What a pull request re-licensing either repair must answer before it does

R14.1 (a line between a `+++` header and its hunk that is not an exact hunk header, including a `@@` line that is no
hunk header, must be a doubt #97 reads, in both ports and the scorer's `own_line_ends` shape); R14.2 and R14.3 (every
path a section names — both sides of `diff --git`, and `rename`/`copy` `from` and `to` — counted toward `multi`, in both
ports and the scorer's `own_read`); R14.4 (an outside path whose whole leading run of dots, dropped, lies inside a prefix
is a dot miss, in both ports and the scorer). Each has a pinned reproduction (section F) that must stay no worse than
`main`.

### A.4 Merge bar (1) under the backstop

#101's reproduction reads as repaired. #121's reproduction ("2 files changed." over a dotfile and its undotted twin):
`main`'s false CONTRADICTED on the count becomes an abstention naming it; the claims whose verdict the repair decided
differently from `main` abstain the same way. #97's reproduction ("Created integrations/git/README.md." over the root
README modified): `main` abstained there (its path accusation is withheld) and the branch now abstains too, naming the
exact entry's VERIFIED and the withdrawal; where #97's resolution finds `main`'s verdict wrong and abstains (a false
VERIFIED on a basename), the abstention is kept. So no reproduction reads `main`'s false verdict, and neither withdrawn
repair gives a decided verdict `main` did not.

---

## B. Y-6: a file's lines are read alone only where no runtime can merge the file with another

The per-file readings — Z-5's refused files, #101's pairing (`_changed_test_defs`, `_definition_only_changed`,
`_definition_paired`), A-1's `_async_tests_added` and Y-5 — read the sides, keyed by the runtime's lower case. Two header
paths that fold alike by the one table (Unicode 16.0.0) may be one key on one runtime and two on another; a path holding a
code point 16.0.0 does not assign may be merged by a runtime on a newer Unicode. Both facts are read from the paths as
written, by the table, so they are the same on every runtime.

**The reading records them as a note of its own**, `fold`, beside Y-1's `files` (which keeps the reason set earliest for
the file-list claims): the text of Y-1's collide or unassigned doubt, whichever the registration order meets earlier, in
the raw door's reading and in the git door's (its diff text's reading, and git's `--name-status` paths, by
`_status_notes`). **Where it holds, `tests_added` and `symbol_added` abstain**, after BC-1's "no Python file" (and, for
`symbol_added`, after the claimed name's own rules) and before every per-file reading:

> {the doubt}, so a runtime may read two files' lines as one file's; the definitions each file adds and removes are not
> read

(with "; claim says N" for a count). So on every runtime and in both ports such a claim abstains with one reason, and
where no fold doubt holds the keys are the same on every runtime (a merge any supported runtime makes folds alike by the
table — the K-2 property — and a merge a newer runtime makes involves an unassigned code point). This closes U14.1,
U14.2 (Z-5 never reads a file under such a doubt) and U14.3 (ASCII case twins fold alike). `_shown_written`'s docstring
is corrected (section I).

The recall cost is `tests_added` and `symbol_added` claims over a diff whose header paths hold case twins or an
unassigned code point; measured after the code.

---

## C. The scorer

1. **The withdrawal in the scorer's own guard** (`expected_guard`): its own `WITHDRAWN`; a difference kept only by a repair
   not withdrawn; the withdrawn reason re-derived; each such abstention counted as `guard_withdrawn_#97` or
   `guard_withdrawn_#121`. `GUARD_OUTCOMES` requires a withdrawn outcome of each on both doors in place of the licensed
   outcomes the thirteenth pass required (a licensed outcome of either can no longer occur): so a run whose canaries
   stopped exercising a withdrawn repair's precondition fails, and a re-licence planted in the instrument keeps a verdict
   the scorer's guard abstains on, which G-C9 refuses.
2. **Y-6 in the scorer's own code**: `own_read` and `own_status_notes` record `fold`; the git door's notes carry it;
   `expected_tests` and `expected_symbol` abstain with the same reason at the same place. G-C7's notes comparison holds
   the instrument's `fold` to the scorer's.
3. **Canaries.** Raw canaries carrying the bytes of R14.1 (`difflib`, with and without `a/`/`b/`), R14.2 (git's
   `log -p --format=`, in both commit orders, and the suffix tier), R14.3 and R14.4 (git's own bytes), G14.1's two
   (the claim with a backslash and the slash claim over git's header shape holding an unquoted backslash), G14.2's and
   G14.3's promoted records, and U14.1's two shapes (Z-5 over `xɤ.py` beside `xꟋ.py`; #101 over `tꟋ.py` beside `tɤ.py`)
   and U14.3's ASCII twins. The door canaries `canary:guard-licensed-by-97` and `-121` keep their bytes and now reach the
   withdrawn outcome.
4. **Plants**, committed and refused by the canaries: each withdrawal lifted (#97, #121, both), a licence granted by a
   withdrawn repair's precondition without its switch, P1, P6, P11, Y-6 dropped for each kind, Y-6's note dropped in the
   reader and at the git door; and in the port, refused by the pinned pairs: the withdrawal lifted, JP1, JP6, Y-6
   dropped. The X12 and X14 plants whose anchors this pass rewrites (the guard's licence line) are re-anchored on the
   new text; a plant that no longer changes any verdict or reason under the withdrawal is recorded as equivalent, with
   the reason.
5. The `NOTE` list runs through this pass.

---

## D. CI

1. **C14.4.** `test_x1_the_characters_the_runtimes_disagree_on_read_alike_in_both_ports` is a strict `xfail` where
   `unicodedata.unidata_version == "16.0.0"`, naming the README's "Not closed".
2. **C14.3, C14.5.** The README's "Which Python enforces what" names the fourth guard-file id and states that the fold
   generator's reproduction runs only on Python 3.14, outside CI.
3. `web/gate/differential/py_side.py` pins the instrument's new sha256.

---

## E. Runtimes

U14.4's docstring correction and U14.6's disclosure as stated above. U14.7's assertion joins `test_x10_k2`.

---

## F. The truth model and the reproductions

Round 14's reproductions — R14.1 (the four `@@` tails' `difflib` renderings, with and without `a/`/`b/`), R14.2 (git's
`log -p --format=` bytes, both orders, and the suffix tier), R14.3, R14.4 (git's bytes with `a/..` and `b/..`) and
U14.1/U14.3's per-file shapes — are pinned with their models in a fixture of their own and judged on the raw door and in
the port (and, for each net model git can build, at the git door) by the committed truth model. The truth model gains
`only_touches` (every path the model changes lies at or under a prefix as the claim writes it) so R14.4 is judged.
**Calibration**: the same check must read them worse than `main` at `340ddfb6` (a count pinned in the test, raw door
and port), and none at this pass's head. U14.1 and U14.3 are pinned as cross-runtime tests (the lab's Python and port,
and a simulated Unicode-17.0 port) that fail at `340ddfb6`.

G14.4: `_worse` judges a claim by `main`'s detail where `main` makes the claim, and the tests assert the branch's
detail equals `main`'s (`path`, `n`, `prefix`, `prefix2`) for every paired claim truth reads.

---

## G. The regression differential, and what this pass can say without it

Under A.2 every final verdict on every door and runtime is either `main`'s verdict on that door and runtime, or
UNCHECKABLE: the guard keeps a decided verdict only where it equals `main`'s, and the one licence not withdrawn (#101)
keeps no decided verdict that differs from `main`'s (its switched-off verdicts differ from its own only as UNCHECKABLE,
held by the scorer's G-C9 on every canary and pinned pair and by `test_the_guarantee_holds_against_the_scorers_own_guard`
on 500 randomised cases). So "no claim reads worse than `main` against truth" holds by construction; what can still
move is **which claims abstain**, and so whether an abstention is identical across ports and runtimes. That clause is
measured by the committed cross-port tests, the pinned pairs (`check_pairs.js`), the checked-in corpora
(`py_side.py`/`js_side.js`) and the reviewers' reproductions re-run on this head. The 30,000-case three-door differential
at fresh seeds is **not run** by this pass (disk, above); it is owed before merge, and the README says so.

---

## H. The recall cost this round expects

- **A**: every claim whose decided verdict #97 or #121 licensed abstains: #97's exact and suffix resolutions verifying
  where `main`'s basename loop abstained or took another entry, #121's dotfile twins counted and resolved apart, and
  #121's dot-miss and outside-path accusations where `main` read another verdict. Counted after the code on the pinned
  pairs and the checked-in corpora.
- **B**: `tests_added` and `symbol_added` over a diff whose header paths fold alike or hold an unassigned code point.
- Nothing else moves: a withdrawal and Y-6 only turn a verdict into an abstention.

---

## I. Corrections to earlier notes and records

1. **Fourteenth pass, "0 claims worse than on main, judged against truth" (README, CHANGELOG, note section G)**: measured
   over sets holding no name with an LF in a plain rendering and no multi-commit rendering; both hold truth-judged
   regressions (R14.1 to R14.3).
2. **Fourteenth pass, section B, and `_shown_written`'s docstring**: "two paths a newer runtime merges differ only at
   unassigned code points, and print alike". False for U+A7D2 with U+A7D3 and U+A7D4 with U+A7D5 (U14.2).
3. **Fourteenth pass, the builder's report**: "182 skipped (the scorer refuses Unicode 16)" under Python 3.14. In that
   harness the scorer exits earlier, on its provenance check (`styxx.diffgate` resolved to a copy outside the checkout);
   59 skips read so and 2 read the Unicode refusal. The README's sentence about the refusal is true of the code.
4. **The CHANGELOG, fourteenth pass**: "Eleven rounds of repair, ten of them answering an adversarial review round"
   (unchanged since the thirteenth pass) beside "twelve passes"; and "git's own renderings of round 12's typechange
   families under 24 option sets", where three of 24 were run for each (3,600 renderings).
5. **`web/gate/README.md`, fourteenth pass, "Which Python enforces what"**: the scorer-dependent tests of
   `tests/test_diffgate_guard.py` are three functions and one parametrised case
   (`test_a_defect_outside_the_three_repairs_can_only_abstain[the count off by one]`), not three.
6. **`_apart_readings`'s docstring (eleventh pass, K-5)**: symbols and emoji do not read alike inside the bounded window
   `_W`, which the port counts in UTF-16 code units (U14.4); `main` reads the same.
7. **Fourteenth pass, section A.3**: "#97 and #121 license no path claim whose claim holds a backslash" — #121's half was
   held by no canary, pinned pair or test (G14.1).

---

## Protocol changes the operator is asked to accept

- **#97's and #121's licences are withdrawn** under the backstop (A): a decided verdict either explains that differs from
  `main`'s abstains, naming `main`'s verdict and the withdrawal; their code, switches and preconditions stay.
- **Y-6** (B): `tests_added` and `symbol_added` abstain where two header paths fold alike or one holds a code point
  Unicode 16.0.0 does not assign.
- **The scorer** (C): the withdrawal and Y-6 in its own code; withdrawn outcomes required of the canaries on both doors;
  raw canaries for R14.1 to R14.4, G14.1 to G14.3, U14.1 and U14.3; the plants listed.
- **The truth model** judges `only_touches`, and a claim by `main`'s detail (F).
- **The 30,000-case differential is owed** (G): not run for lack of disk; the bar's second clause rests on A.2's
  construction and the measurements listed there until it is.
