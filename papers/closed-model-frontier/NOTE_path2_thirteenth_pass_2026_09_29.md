# NOTE — PATH-2, thirteenth pass: one path read twice, a name as written, a runtime newer than the fold, and the scorer's licensed outcomes

2026-09-29. Branch `fix/diffgate-path-resolution` (pull request #161). The pushed head is `15878ab5`; the twelfth pass
(`0f559a87`, not pushed) sits on it, and `origin/main` `a4732c52` is merged in at `76e65596` (the CHANGELOG conflict
resolved, both `[Unreleased]` entries kept). The guard's references are unchanged: `styxx/_diffgate_ref.py` is `main`'s
`styxx/diffgate.py` (sha256 `9b620e00…`, LF) and `web/gate/diffgate_ref.js` is `main`'s `web/gate/diffgate.js`
(sha256 `06688702…`).

This note sits on top of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third to
twelfth-pass notes. **None of them is edited.** Where one is wrong or incomplete, the correction is here (section H).

**How this round is worked.** As in the eleventh and twelfth passes: this note is written before the code and
committed alone, ahead of it. It states round 12's findings with their evidence, the CI failure and its cause, the
rules this pass adds and why, and the scorer changes. The measurements are taken on the commits that follow and are
recorded in `web/gate/README.md` and the CHANGELOG entry, not here.

The merge bar is unchanged: (1) the #97, #121 and #101 reproductions are fixed; (2) no claim reads worse than
`origin/main` on any door — the raw text, the JavaScript port, the git door — under any supported runtime, **judged
against truth** (CPython's parser for definitions; git's `--name-status`, including `T` and mode changes, for file
lists), and no new Python/JavaScript disagreement, an identical abstention allowed; (3) the scorer
(`path2_gates.py`) and the committed tests cannot admit a planted defect.

---

## The round-12 evidence

Two reviews of `0f559a87`, and CI on the pushed head `15878ab5`.

### The regressions lens (verdict: fix before merging; three blockers)

Round 11's harness, extended by the reviewer for file modes: the repository builder writes `100755`, `120000` and
`160000` entries, and the truth model reads git's `--name-status`, `T` included. 21,000 fresh cases and round 11's 17
reproductions, three doors, Python 3.12.10 and 3.14.2, Node 24.13.0. #101 licensed no verdict (its switched-off
verdicts differ from its all-on ones only as UNCHECKABLE), so the live surfaces are #97 and #121. Result: **294
claim-door cells worse than `main` on the raw door and 294 in the port, under each Python; 0 at the git door.** 293 are
typechanges (252 in the set built under git's default `diff.submodule`, 41 under `diff.submodule=log`) and 1 a
`strip()`-merged trailing-space name found by the twelfth pass's own generator at a fresh seed. 0 new Python/port
disagreements, 0 branch-only raises. Round 11's reproductions stay fixed.

- **R12.1 (blocker) — #97 on a typechange.** git renders a typechange (file to symlink, symlink to file, file to or from
  a submodule, executable to symlink, an empty file to or from a symlink) as **two** `diff --git` sections for one path:
  a deletion, then a creation. `_read_diff` keeps the later status (`A`; symlink to empty file keeps `D`, because the
  empty creation is flushed under a key already read). `main` reads the same wrong status, but its earliest-match loop
  lands on a basename twin and abstains; #97's exact or suffix tier lands on the typechanged path, the precondition
  holds, and a false VERIFIED is kept. Reproduction: "Created lib/x.sh." over `aaa/x.sh` modified and `lib/x.sh` a file
  turned symlink; `--name-status` reads `M aaa/x.sh`, `T lib/x.sh`, so truth is false; `main` UNCHECKABLE, the branch
  VERIFIED on the raw door (3.12, 3.14, CRLF, `-U0`) and in the port; the git door stays UNCHECKABLE (status `T`). The
  same for "Deleted lib/x.sh." when `lib/x.sh` goes symlink to empty file, and for the gitlink and executable variants.
- **R12.2 (blocker) — #121 on a typechanged dotfile.** The same two sections for `.cfg.json` beside a modified
  `cfg.json`, in git's own rendering (the only rendering #121 licenses since the twelfth pass): `main` merges the two
  keys, reads the twin's `M` and abstains; the branch reads the typechange's later section (`A`), #121's precondition
  holds, and "Created .cfg.json." reads a false VERIFIED. Also `.github/ci.yml` beside `github/ci.yml`, and "Removed
  .env.json." where the dotfile goes symlink to empty file.
- **R12.3 (blocker) — R11.2 fixed only in git's rendering.** The twelfth pass stopped #97 licensing under a Z-3 doubt,
  which git's rendering of a name ending in a space carries. Rendered plainly (Python's `difflib`, with or without
  `a/`/`b/`), the same name carries no doubt: both header paths go through `strip()`, the key and the forms #97 compares
  are the stripped name, and #97 licenses a false VERIFIED. A trailing space, TAB, NBSP, U+3000 or two spaces, on the
  raw door and in the port, under both Pythons. Reproductions: the `difflib` renderings of round 11's
  `A97-trailing-space-created` and `-deleted`, and `sp-2912003-3644-difflib-noprefix#hand` ("Deleted x.py." where
  `x.py` is modified and `x.py ` deleted; `strip()` merges the two keys to `D`).
- **R12.4 (minor) — the truth model is blind to modes.** The committed truth-judged tests
  (`tests/test_diffgate_guard.py`, `_truth_status`) derive `A`/`M`/`D` from content and the fast-import builder writes
  no modes, so git's `T` is outside the model; the tests passed (75) while R12.1 and R12.2 reproduced.

### The guard lens (verdict: fix before merging; one blocker)

Plants committed in scratch clones with clean provenance, run through the committed tests, `check_pairs.js`, and
`path2_gates.py` in both modes.

- **PA (blocker).** A licence granted by all three repairs reverted together instead of one
  (`evaluate(_Repairs({repair}), None)` becoming `_Repairs(REPAIRS)`). Both modes exit 0; the committed tests refuse
  it only by accident (the stub harness raises on unpacking `rp.off`). It changes a verdict: "Created .env.json." over
  `lib/env.json` modified and `.env.json` created, in git's rendering — `main` UNCHECKABLE, the clean branch
  UNCHECKABLE (no single revert gives `main`'s verdict back), PA VERIFIED. No scorer input reaches that case.
- **PF (minor).** G-C7 compares the raw door's licence facts with the scorer's, but the git door's (`_evaluate_git`'s
  forms from `--name-status`) nowhere. Lower-casing them gives false VERIFIED verdicts at the git door and passes the
  canaries and corpus mode.
- **PK (minor).** The #121 tier-kept path licence of `98f74833` dropped passes the canaries and corpus mode (the
  differential and the truth tests refuse it); `X12_CANARY_PLANTS` leaves it out, while `web/gate/README.md` says each
  tightened licence dropped is refused by the canaries.
- **PG1b (minor).** Round 11's plant — the git door's reference replaced by `main`'s raw door on git's text — still
  passes corpus mode (the differential and the tests refuse it).
- **No licensed outcome (minor).** The README and the CHANGELOG say the canaries "reach every guard outcome on both
  doors"; no canary reaches a licensed outcome on either door and `GUARD_OUTCOMES` requires none. That gap is why PA,
  PK and PF pass the canaries.
- **The twelfth note (minor).** `98f74833` tightened #121 after `NOTE_path2_twelfth_pass` was committed; the note does
  not describe the rule, its section E says "Nothing else moves", and its protocol list omits it. The scorer's `NOTE`
  list still ends at the eleventh pass.

### CI on `15878ab5`

One failure, on Python 3.9 to 3.12: `tests/test_diffgate_path2.py::test_x10_k2_the_fold_sees_every_merge_the_port_s_key_makes`.
The GitHub runner's Node lower-cases 28 code points that Unicode 17.0 assigned — U+A7CE, U+A7D2, U+A7D4 and U+16EA0 to
U+16EB8 — and the fold table (`styxx/_fold.py`, Unicode 16.0.0) holds none of them, so `fold(toLowerCase(c)) != fold(c)`
for each. The lab's Node 24.13.0 carries Unicode 16.0, so the test passed here. It is not a flaky test: on that runtime
a pair of header paths differing only in one of those letters is one key to the port and two folds, so Y-1's collision
note is missed there, and `_fold.py`'s docstring ("It is sound on every supported runtime") is false for a runtime
newer than the table.

---

## A. The licences: one path read twice, and a name as written

### A.1 A key more than one file section registers licenses nothing

Each reader — `_read_diff`, the port's `_readDiff`, and the scorer's own `own_read` — records, beside the licence facts
it already reads, **`multi`: each key that more than one file section registers** (a `---`/`+++` pair, or a
`diff --git` header flushed without a pair, is one section). git writes a typechange as a deletion section then a
creation section for one path; a case twin, or two names `strip()` merges, registers one key twice too. Any key read
twice counts, whatever the reason.

**#97 and #121 license nothing on a path claim whose resolved entry** (the tiered resolution, every repair on) **is such
a key.** A count or a scope has no resolved entry and is not touched: a typechange is one file to git's `--name-status`
and one key to the reader.

**At the git door** the file list is `--name-status`. There, an entry whose letter is `T`, an entry git's diff text
shows a mode change for (`old mode`/`new mode` in its section), and a key the diff text's own reading registers more than
once, license nothing. This is inert for a verdict by construction — a `T` or a mode-changed `M` never equals `A` or `D`,
so a created or deleted claim abstains there anyway, and a touched claim reads VERIFIED on either resolution — and is
kept so that the rule is one rule on every door; G-C7 compares the facts it reads.

### A.2 The forms a licence compares are the names as written, before `strip()`

The forms #97 compares (and #121's tier-kept check since `98f74833`) are **each header path as written**: a
`---`/`+++` line's path cut only at its terminating TAB (the one git appends to a name holding a space, and GNU diff
before its timestamp) and a CR after it, with `a/` or `b/` dropped as the key drops it, **never stripped**. A name
whose trailing (or leading) whitespace `strip()` drops then never equals the claim, in **every** rendering — git's,
`difflib`'s, GNU's, a hand-written one. The key, the status map and Y-1's case fold keep reading the stripped name as
before; only the forms the licences compare change. A header path read from `diff --git`, `rename to` or `copy to` was
never stripped and is unchanged.

### A.3 Tightening a licence can only turn a verdict into an abstention

As in the twelfth pass (A.3): a precondition is read in one place, as one conjunct of a licence, where a decided
claim's verdict differs from the reference's. A.1 and A.2 add conjuncts. A claim whose final verdict was the
reference's, an abstention, or K-5's reading of the reference, is not touched. A claim that loses its licence becomes
UNCHECKABLE, naming the reference's verdict. Every claim's final verdict is the twelfth pass's, or UNCHECKABLE.

---

## B. `98f74833`, recorded as a protocol change

`98f74833` ("#121 licenses a path claim only where its entry matches the claim, case kept, by its tier") was committed
after `NOTE_path2_twelfth_pass`, in answer to that pass's own truth-judged differential (14 cells worse at `e4586637`:
4 raw-door, 4 port, 6 git-door; #121 licensing "Created X.toml." over `.config/x.toml` created, an entry the claim
matches only once lower-cased). The rule: **on a path claim, #121 licenses only where every form of the resolved entry
matches the claim, case kept, by the tier the resolution used** — equal (exact), ending in `/` and the claim (suffix),
or with the claim's name (basename). The twelfth note does not describe it, and its section E says "Nothing else
moves". It is recorded here as a protocol change the operator is asked to accept, with the scorer's copy
(`9d7b91fd`), a canary that reaches it (C.3), and its drop committed as a plant the scorer refuses.

---

## C. The scorer

1. **PA refused.** Door and raw canaries whose difference from `main` only two reverts together give back
   ("Created .env.json." over `lib/env.json` modified and `.env.json` created, in git's rendering). With every licence
   granted by one switched-off repair, the guard abstains; a licence granted by all three reverted together keeps
   VERIFIED, and G-C9 refuses it in both modes. `_stub_evaluate` in the committed tests states that exactly one repair
   is switched off, so the stub tests refuse PA by a stated rule rather than by raising.
2. **A licensed outcome on each door.** Canaries that reach `guard_licensed_by_#97` and `guard_licensed_by_#121` on
   the raw door and at the git door (a case-kept suffix match beside an earlier basename twin; dotfile twins counted in
   git's rendering), and `GUARD_OUTCOMES` requires both on each door. A run whose canaries reach none fails.
3. **The licences as canaries and plants.** Canaries where dropping each licence condition keeps a verdict: the
   `98f74833` tier-kept path licence (PK: "Created X.toml." over `.config/x.toml` created and `config/x.toml` modified,
   both doors), A.1's multi-section key (the typechange renderings, raw door only: two sections for one path cannot be
   rebuilt as one file), A.2's forms as written (the `difflib` rendering of a name ending in a space, raw door only: no
   safe path holds a space). Each drop is committed in `X12_CANARY_PLANTS` (or its successor) and refused by the
   scorer program. The git door's `T` and mode rule is inert for a verdict (A.1) and is held by G-C7's facts.
4. **PF.** G-C7 at the git door compares the instrument's git-door licence facts (what `_evaluate_git` reads: `forms`
   from `--name-status`, `multi` from the letters, the mode changes and the text) with the scorer's own
   (`own_git_licence`, which now reads the status letters too). A mirror door canary ("Created src/readme.md. Deleted
   src/config.py." over `docs/readme.md` modified, `lib/src/README.md` created, `a/config.py` modified and
   `lib/src/Config.py` deleted) reaches the case the existing one did not.
5. **PG1b.** A door canary where `main`'s git door and `main`'s raw door read git's text apart (a removed `-- users`
   beside an added `++ x` inside an exact hunk, which `main`'s raw door reads as a header pair and a phantom file, and
   `main`'s git door never sees): a git door guarded against `main`'s raw door reads a different verdict, and G-C9
   refuses it in both modes.
6. **The scorer's own code** reads A.1, A.2 and section D's doubt as the instrument does, written out; its `NOTE` list
   runs through this pass.

---

## D. A runtime newer than the fold

The fold (`styxx/_fold.py`, and the port's copy) is Unicode 16.0.0's lowercase mapping. Its soundness argument
(`web/gate/gen_fold.py`) holds for a runtime whose lowercase mapping of each code point is the code point itself or
16.0.0's mapping. A runtime on a newer Unicode lower-cases the characters that version assigned, which the table does
not know: the CI runner's Node, on Unicode 17.0, is one.

**The rule.** A header path holding a code point **unassigned in the table's Unicode version** — general category `Cn`
in CPython 3.14's `unicodedata`, Unicode 16.0.0 — is a **path-key doubt**: the reader's file list is not sure (Y-1's
note, with its own reason), so every file-list and path claim abstains. It is asked wherever the fold is asked: of each
path the reader registers (`_read_diff`, `_readDiff`, `own_read`), of each `--name-status` path at the git door
(`_status_notes`), and of each path `main`'s reading keys in Z-3's check (beside `_folds_apart`). Claims keyed on such a
path abstain in both ports alike, whatever each runtime's Unicode.

**Why it is sound.** For a runtime R, Y-1 must see every pair of paths R's key merges. For a path whose code points are
all assigned in 16.0.0, `fold(R.lower(c)) == fold(c)` holds code point by code point where R lower-cases each assigned
code point as 16.0.0 does, so the fold sees R's merges. A path holding an unassigned code point is the only place a
newer runtime can merge beyond the table, and there the file list abstains. Unicode does not in general promise that an
assigned character never gains a lowercase mapping, so the test suite checks the premise on the runtime it runs on:
**every code point whose runtime lower-casing merges beyond the table must be in the table's unassigned set**, in the
Python (`str.lower()`) and the port (Node's `toLowerCase()`), on any runtime. The 28 Unicode-17 code points are pinned as
a fixture that abstains in both ports, and a simulated newer runtime (a monkeypatched lower-casing in Python, a patched
`toLowerCase` in the port) is tested.

**The table.** `web/gate/gen_fold.py` emits the assigned set compactly beside the fold, in both ports' generated blocks,
with its sha256; `--check` holds both ports and the generator to the same bytes. `_fold.py`'s docstring is corrected to
say how a newer runtime is covered.

---

## E. The truth model, with modes

The committed truth model (`tests/test_diffgate_guard.py`: `_truth_status`, its fixtures and the fast-import builder)
gets a mode per path (`100644`, `100755`, `120000`, `160000`) and git's `T` letter: a path present on both sides whose
type differs (regular file, symlink, gitlink) is `T`; one whose bytes or executable bit differ is `M`. The builder
writes each mode (a gitlink as a commit id). Round 12's reproductions (T1 to T9, the gitlink and symlink variants, the
`difflib` renderings of `A97-trailing-space-created`/`-deleted` and `sp-2912003-3644`, the `.cfg.json` and `.github`
twins) are pinned with their models and judged on the raw door, in the port and at the git door. **Calibration**: the
same check must refuse `0f559a87` on the round-12 reproductions, and pass on this pass's head.

---

## F. The truth-judged regression differential

The round-12 reviewer's harness with modes and the twelfth pass's generators: `origin/main` `a4732c52` against this
pass's head, three doors, Python 3.12 and 3.14, 30,000 cases or more — typechanges, mode changes, symlinks, gitlinks
under `diff.submodule` default and `log`, empty files, trailing whitespace in every rendering, case twins, dotfile twins,
`--no-prefix`, renames and copies — iterated to **0 claims worse against truth and 0 new Python/JavaScript
disagreements**. The numbers and the recall cost are recorded after the code.

---

## G. The recall cost this round expects

- **A.1**: path claims licensed by #97 or #121 on a key two sections register — a typechanged path, a case twin, a
  `strip()`-merged pair — abstain. On case twins Y-1 already abstained; on typechanges the licensed verdicts were false.
- **A.2**: path claims licensed on a name whose header path carries whitespace `strip()` drops abstain.
- **D**: every file-list and path claim of a diff whose header path holds a code point Unicode 16.0.0 does not assign
  abstains (on a runtime of 16.0 or older such a path was read as before).
- Nothing else moves: A.3.

---

## H. Corrections to earlier notes and records

1. **Twelfth pass, section E** ("Nothing else moves") and its protocol list: `98f74833` moved path claims #121 had
   licensed on an entry matching only in case (section B).
2. **Twelfth pass, `web/gate/README.md` and the CHANGELOG**: "the canaries reach every guard outcome on both doors".
   They reached K-5, an abstention and (raw door) an abstention whose precondition held; no licensed outcome (C.2).
3. **Twelfth pass, `web/gate/README.md`**: "each tightened licence dropped" was refused by the canaries. The
   `98f74833` licence dropped passed them (PK).
4. **Twelfth pass, `web/gate/README.md`**: "472 of 2,660, five more than the eleventh pass". The eleventh pass measured
   468, so 472 is **four** more.
5. **Tenth pass and `styxx/_fold.py`**: the fold "is sound on every supported runtime". It is sound on a runtime whose
   Unicode is 16.0 or older; on a newer runtime it is sound with section D's doubt, and not without it.
6. **Twelfth pass, section A.2** (R11.2): #97's doubt rule closed the trailing-space name in git's rendering only; in a
   plain rendering the same name carries no doubt (R12.3). A.2 above closes it by the name as written.

---

## Protocol changes the operator is asked to accept

- **A key read twice licenses nothing** (A.1): #97 and #121 license no path claim whose resolved entry more than one
  file section registers; at the git door, no `T` entry and no mode-changed entry.
- **The forms a licence compares are the names as written** (A.2), cut only at the terminating TAB and CR.
- **`98f74833`'s tier-kept #121 path licence** (B).
- **A header path holding a code point Unicode 16.0.0 does not assign is a path-key doubt** (D).
- **The truth model with modes and `T`**, and the round-12 reproductions as truth-judged tests (E).
- **The scorer**: canaries reaching a licensed outcome on each door, and `GUARD_OUTCOMES` requiring them; PA, PK, PF
  and PG1b refused in both modes; the git door's licence facts compared in G-C7; the licences of this pass in its own
  code (C).
