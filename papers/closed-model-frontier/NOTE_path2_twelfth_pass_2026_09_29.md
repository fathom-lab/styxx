# NOTE — PATH-2, twelfth pass: the licences tightened, and the guarantee judged against truth

2026-09-29. Branch `fix/diffgate-path-resolution` (pull request #161), head `15878ab5` before this round;
`origin/main` `2a6ce0a3` is merged in. The guard's references are unchanged: `styxx/_diffgate_ref.py` is `main`'s
`styxx/diffgate.py` (sha256 `9b620e00…`, LF) and `web/gate/diffgate_ref.js` is `main`'s `web/gate/diffgate.js`
(sha256 `06688702…`).

This note sits on top of `PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md`,
`ERRATUM_path2_amendment_2026_09_17.md`, `NOTE_path1_pairs_repinned_by_path2_2026_09_25.md` and the third to
eleventh-pass notes. **None of them is edited.** Where one is wrong, the correction is here (section F).

**How this round is worked.** As in the eleventh pass: this note is written before the code and committed alone,
ahead of it. It states the tightened licences and why, what tightening a licence can and cannot do, the tests that
judge the guard against truth rather than against itself, and the scorer changes. The measurements are taken on the
commits that follow and are recorded in `web/gate/README.md` and the CHANGELOG entry, not here.

The merge bar is unchanged: (1) the #97, #121 and #101 reproductions are fixed; (2) no claim reads worse than
`origin/main` on any door — Python `gate_diff_text`, Python `gate_diff`, the JavaScript port — under any supported
Python (3.9 to 3.14), **judged against truth** (CPython's parser for definitions, git's `--name-status` for file
lists), and no new Python/JavaScript disagreement, an identical abstention in both ports allowed; (3) the scorer
cannot admit a planted defect.

---

## The round-11 evidence

Two reviews of `15878ab5` (`round11_diffgate.json`).

**The regressions lens** ran `origin/main` against the branch on all three doors under Python 3.12.10 and 3.14.2,
judged by its own truth model (CPython's parser over the base and head files; git's `--name-status`), on real git
output and on real non-git generators (Python's `difflib`, GNU diff 3.12). The three reproductions are fixed, and
#101 licensed no move at all (2,840 cases). On two fresh sets from the builder's own generators (7,000 cases) it read
0 claims worse. On its own generator (4,000 cases) it read **1,133 claim-door cells worse on the raw door, 1,133 on
the port and 6 on the git door**, and on 17 aimed cases 12, 12 and 3, under both Pythons; every one was a licensed
difference, 1,127 by #121 and 6 by #97. The guard's own guarantee harnesses reported 0 violations on the same inputs.
The classes:

- **R11.0** #121 in a rendering with no `diff --git` header (`difflib`, a plain or hand-written `---`/`+++` diff): a
  directory named `a/` or `b/` read as git's prefix, or a trailing-space twin merged by `strip()`, beside #121's
  dotted split, moves `files_changed_count` from `main`'s right CONTRADICTED to a false VERIFIED. `--- a/x.py` over
  `+++ /dev/null` for a real `a/x.py` beside a root `x.py` is byte-identical to git's own prefixed deletion: no
  reading of the text can tell them apart.
- **R11.1** #121 where the renderer leaves a changed file out: `difflib`, and GNU `diff -uN` run per file, print
  nothing for an empty created or deleted file (an empty `__init__.py`). No line names the missing file, so no doubt
  the reader could add closes the class.
- **R11.2** #97's exact tier on an entry the reader knew it had misread: git prints a name ending in a space as
  `+++ b/x.py <TAB>`, `strip()` keys it `x.py`, and the reading recorded Z-3's `_Z3_REPLACED`; but Z-3's doubts
  guarded only file-list claims, so #97 licensed a false VERIFIED on "Created x.py." (raw door and port).
- **R11.3** #97's tiers compare lower-cased keys, so its precondition held on a match that exists only
  case-insensitively: "Created src/README.md." VERIFIED from a created `lib/src/readme.md` where `main` abstained, on
  all three doors, although git's paths are case-sensitive and the claim is false.
- **R11.4** #121 licensed a dotted path claim on a `strip()`-merged key in a plain rendering ("Created .env.json."
  VERIFIED from a created `.env.json ` beside `env.json`), the same shape as R11.0.
- **R11.5** (minor) the guarantee tests are circular for this lens: they decide a licence with the module's own
  `_precondition` and switches and never consult truth; and `web/gate/README.md` published "0 claims worse than on
  `main`" beside "the three repairs are therefore the only surface where a new false verdict can arise".

**The guard-integrity lens** found the references byte-identical to `main`, every entry point guarded, the switches
explicit, and 24 plants refused by the committed tests; but the scorer, `path2_gates.py`, in both modes at default
parameters **admits three planted guard defects** at the git door: the door returning the unguarded reading (PG1),
the door calling `main`'s `gate_diff` and ignoring it (PG1c), and a licence granted without asking the switch (PG4).
G-C9's inputs never reach those branches: at the git door no claim abstains through the guard and none reads through
K-5, no guard abstention on any input holds a precondition, and the canaries carry no guard outcome. It also found
(minor): a licence without its precondition and a claim kept where `main` raises (PG3, PG7) are held only by stub
tests; a reader defect that fires only on a dotted key (PD2) passes every committed test and corpus mode, and the
differential refuses it on one record; `own_claim_sentences` leaves out `main`'s `kind` leak, so G-C9 flags a correct
instrument (29 raw-door and 2 git-door cases); `Counterfactual.gate` and `reading` catch every exception as "the copy
raises", so a failure of git under load reads as an instrument defect; the committed guarantee check is
self-consistency, and its corpora test has two parameters, both files gitignored; the port's `_guard` reads a missing
reference as "main raises"; and the implementation reads the operator's "UNCHECKABLE unless licensed" as "never" where
`main` raises or makes no such claim.

---

## A. The tightened licences

### A.1 #121 licenses only in git's own rendering

**#121 licenses a difference — on any claim, a file-list claim (`files_changed_count`, `only_touches`) or a claim
through a dotted path key alike — only where the diff is in git's own rendering.** In any other rendering (a plain
`---`/`+++` diff, `difflib`, GNU `diff -u` or `-uN`, a hand-written diff) it licenses nothing, and a claim whose
difference from `main` only #121 explained abstains, naming `main`'s verdict.

A diff is in git's own rendering when, read over the F-2 line split (`\r\n`, `\r` and `\n`) outside the lines an
exact hunk's counts hold, all of these hold:

1. it has a `diff --git ` line, and every such line is readable: both of its paths read (`_header_paths`), quoted or
   not;
2. every `---`/`+++` pair read as a file header sits under a `diff --git` header of its own (one no earlier pair has
   closed), with its `---` line before its `+++` line, and names that header's files as git writes them: the `---`
   path, cut at a TAB, is `/dev/null` or `a/` followed by the header's old path, and the `+++` path, cut at a TAB, is
   `/dev/null` or `b/` followed by the header's new path (a quoted header path compared quoted);
3. every `rename from`/`copy from` line names the header's old path and every `rename to`/`copy to` line its new
   path, as written;
4. no line outside a header, a hunk, git's extended header lines and a binary patch is one no reading places (K-4's
   line: git's `Submodule` line under `diff.submodule=log`, svn's and hg's notices, an e-mail's headers), and no line
   names a changed file no header counts (Y-1's `Binary files … differ` outside a header, `Only in …`).

**Why a structural rule.** R11.1 cannot be seen in the text: no line names the file the renderer left out. R11.0's
`a/` directory is byte-identical to git's own prefix in a plain rendering. git's own rendering is the one format that
rules out both by construction: it writes a `diff --git` header for every changed file, empty ones, mode changes,
binaries and pure renames included, and its `a/` and `b/` are always prefixes, never directories, because the header
carries both paths and the `---`/`+++` lines must match them. The trailing-space name (R11.2's and R11.4's shape)
prints under git as `+++ b/x.py <TAB>`, which conditions 2 and 3 read as written; its misreading by `strip()` is
Z-3's `_Z3_REPLACED`, which already abstains the file-list claims, and A.2 closes it for #97. `git diff --no-prefix`
writes `diff --git x.py x.py`, which condition 1 refuses.

**Both doors.** The git door reads git's diff text by the same test. Under git's default configuration the test
holds wherever git writes the diff; under `diff.noprefix`, `diff.mnemonicPrefix` or `diff.submodule=log` it does not,
and #121 licenses nothing at the git door either, although the git door's file list is git's `--name-status`. That
costs recall where the git door was right, and is taken knowingly: one rule, the same in every door and both ports,
over the same bytes.

### A.2 #97 licenses only a case-preserved match, and never under a Z-3 doubt

1. **The case.** #97's precondition requires the exact or the suffix match to hold on the **case-preserved** path: the
   claim as written, with only its leading `./` and `/` segments dropped and backslashes turned into slashes, against
   the entry's path as the reading read it, transformed the same way and never lower-cased (at the raw door the
   header path with git's `a/` or `b/` dropped; at the git door `--name-status`'s path). Where the match holds only
   once both are lower-cased, #97 licenses nothing. git's paths are case-sensitive, and a claim about `src/README.md`
   says nothing about `lib/src/readme.md`.
2. **The doubts.** At the raw door, and in the port, #97 licenses nothing where the reading holds **any** of Z-3's
   doubts `main`'s reading also held (`_Z3_SHAPED`, `_Z3_REPLACED`, `_Z3_UNREAD`, `_Z3_UNREAD_PAIR`, `_Z3_UNSHAPED`,
   `_Z3_UNPLACED`). The doubts are recorded for the diff, not per entry, so "an entry read under a doubt" is read as
   "any entry of a reading that holds one": wider than the entry, never narrower. At the git door the file list is
   git's `--name-status`, which is read under none of them.

#101's licence is unchanged: round 11 measured no move it licensed.

### A.3 Tightening a licence can only turn a verdict into an abstention

The guard reads a precondition in one place: where a decided claim's verdict differs from the reference's, as one
conjunct of a licence (the repair switched off gives the reference's verdict back **and** its precondition holds).
The conditions above add conjuncts. A claim whose final verdict was the reference's, an abstention, or K-5's reading
of the reference, is not touched: none of them reads a precondition. A claim that loses its licence becomes
UNCHECKABLE, with the reason that names the reference's verdict. So this round can take a verdict away and cannot
give one: every claim's final verdict is the eleventh pass's, or UNCHECKABLE, and a claim cannot read worse than it
did (an abstention is never a false verdict). The verdict and `--strict` are recomputed from the final claims, so a
new abstention can turn a `--strict` PASS into FAIL, and a FAIL whose one CONTRADICTED claim abstains into PASS; both
follow from the claims.

Both ports compute the new conditions over the same line split with the same patterns, so they agree wherever they
read the same lines; the differential below measures that.

### A.4 "Unless licensed" is read as "never" where the reference gives no verdict

The operator's decision said that where `main` raises, or makes no such claim, a decided claim is "UNCHECKABLE
unless licensed". A licence is defined by giving the reference's verdict back: repair R switched off yields the
reference's verdict on the claim. Where the reference raises, or has no claim of that kind, sentence and occurrence,
there is no verdict to give back, so no licence can hold, and the implementation abstains every such decided claim.
This note records that reading as a protocol change. It errs toward abstaining: no false verdict is possible there,
only recall (with `--run`, a passing `tests_pass` abstains wherever `main` raises on the diff).

---

## B. The guarantee judged against truth

The eleventh pass's guarantee tests and its harness asked the instrument's own switches and preconditions whether a
difference was licensed. That checks that the guard does what the module says; it cannot see a licence that is
granted by the module's rules and wrong (R11.0 to R11.4 passed it with 0 violations). This round adds, committed:

1. **The round-11 reproductions as truth-judged tests.** The 17 cases of `rv11_repros.json`, with their base and head
   file models, as test data. Each is judged by a truth model committed beside it — file lists and statuses from the
   model (for the git cases, what git's `--name-status` reports on a repository built from it), definitions by
   CPython's parser — never by `_precondition`. On every door the case has (the raw door and the port; the git door on
   a repository written with `git fast-import`), no final claim may read worse than `main`'s against that truth.
2. **The guarantee against the scorer's reverts.** A property test that holds the instrument's final claims to the
   scorer's own guard (`path2_gates.expected_guard`: `main`'s verdict from the baseline, the switched readings from the
   scorer's reverts of #97, #121 and #101, the preconditions from the scorer's own code), over the randomised guard
   set on the raw door and, for a subset rebuilt as repositories, on the git door.
3. **`web/gate/README.md`** states what each number measures: the regression differential is judged against truth;
   the guarantee harness is self-consistency (the module's rules asked about the module's verdicts); the property test
   is the scorer's rules asked about the module's verdicts. It no longer says the three repairs are "the only surface
   where a new false verdict can arise" without saying what that bounds (section F.1).

---

## C. The scorer

1. **Guard canaries, both modes.** `DOOR_CANARIES` gains the K-5 git-door record (a sentence holding a non-ASCII word
   character over two same-named files) and a git-door record whose reading differs from `main`'s git door with no
   licence (R11.3's case-only match); `RAW_CANARIES` gains a guard abstention whose #121 precondition holds but whose
   #121 switch does not give `main`'s verdict back, and the matching pairs for the other kinds. `G-C8_canaries` fails
   unless its guard tally shows `guard_k5_read_as_main` at the git door, `guard_abstained` on each door, and an
   abstention whose precondition held. PG1, PG1c and PG4 join the committed plants that run the scorer program, not
   only stub tests.
2. **`main`'s `kind` leak replayed.** `own_claim_sentences` rebinds the kind across a template's matches in a
   sentence as `main`'s loop does (`kind = "file_touched"` rebinds the loop variable), so G-C9 no longer flags a
   correct instrument; the reproduction is a raw canary.
3. **Exceptions.** `Counterfactual.gate` and `reading` catch only what reverted code raises (`AttributeError`, as
   `main` does on K-3); a failure of git, the operating system or a timeout propagates and fails the run instead of
   reading as an instrument defect.
4. **Dotted keys.** Canaries and pinned pairs for created, deleted and touched claims on a dotted path whose status
   mismatches, and for a count and a scope beside a lone dotfile; a dot-triggered plant (PD2) joins the committed
   plants, and the reading oracle (G-C7) must refuse it.
5. **The port's missing reference.** In the port's `_guard`, only an exception thrown by `main`'s `gateDiffText`
   reads as `main` raising; a missing reference (`diffgate_ref.js` not loaded) throws.
6. **The tightened licences**, written out in the scorer's own code (A.1, A.2) and compared, as a note, with the
   instrument's (G-C7).

---

## D. The truth-judged regression differential

The round-11 reviewer's harness and truth model are reused, with new generators for the shapes round 11 found:
`difflib` output and per-file GNU `diff -uN` that leave empty files out, case-insensitive twins on git's
case-sensitive paths, names ending in spaces, `a/` and `b/` directories under `--no-prefix`, and `strip()`-merged
keys in plain renderings. `origin/main` against this branch, three doors, Python 3.12 and 3.14, 20,000 cases or more,
iterated to **0 claims worse against truth and 0 new Python/JavaScript disagreements**. The numbers and the recall
cost are recorded after the code.

---

## E. The recall cost this round expects

- **#121 outside git's rendering.** Every claim whose difference from `main` only #121 explained, in a diff that is
  not git's own rendering, abstains: counts and scopes over dotfile twins, and dotted path claims beside an undotted
  twin, in plain, `difflib`, GNU and hand-written diffs, and at the git door under `diff.noprefix`,
  `diff.mnemonicPrefix` or `diff.submodule=log`. Most of these read right before; the rule gives them up because the
  wrong ones cannot be told apart from them in the text. Pinned pairs written as plain `---`/`+++` diffs over dotfile
  twins are re-pinned to UNCHECKABLE.
- **#97** abstains on a case-only match (a claim that the branch read as true only by lower-casing, which truth reads
  as false on git's paths) and on any path claim it would have licensed in a reading holding a Z-3 doubt.
- Nothing else moves: A.3.

---

## F. Corrections to earlier notes and records

1. **Eleventh pass, section B**: "The three repairs are therefore the only surface where a new false verdict can
   arise." What the guard bounds is **where** a verdict other than `main`'s can come from — a named repair, on its
   own precondition — not **whether** such a verdict is right. A licensed difference can be false (R11.0 to R11.4:
   1,133 cells on one set). The sentence also sat in `web/gate/README.md` beside "0 claims worse than on `main`",
   which round 11 contradicted.
2. **Eleventh pass, the CHANGELOG entry and `web/gate/README.md`**: "the guarantee held claim by claim … 0
   violations" over 78,532 inputs (869,220 raw-door and 339,918 git-door claims under 3.12, 871,428 in the port).
   That measured **self-consistency**, not truth: the harness decided each licence with the module's own
   `_precondition` and switches, so a licence granted by those rules and wrong passed it. The same holds for the
   committed `guarantee_violations` test. The round summary's "parametrized over all 9 corpora" was two parameters
   (`corpus_fuzz`, `corpus_real`), both files gitignored, so the test skips in a clean checkout and in CI.
3. **Eleventh pass, `web/gate/README.md`**: "0 claims worse than on `main`" on every door, and "the git door … reads
   none worse". Round 11 measured 1,133 raw-door, 1,133 port and 6 git-door cells worse on one set, and 12, 12 and 3
   on the aimed cases.
4. **Eleventh pass, A.3.5**: where the reference raises or makes no such claim, "every such claim the branch decides
   is UNCHECKABLE". Correct as implemented; section A.4 records it as the reading of "unless licensed".
5. **Eleventh pass, B.2**: Z-3's doubts were listed as #121's defence. They did not guard #97 (R11.2), and no doubt
   can close R11.1.

---

## Protocol changes the operator is asked to accept

- **#121's licence** only where the diff is git's own rendering (A.1), in both ports and on every door.
- **#97's licence** only on a case-preserved exact or suffix match, and never in a raw-door reading holding a Z-3
  doubt (A.2).
- **"Unless licensed" read as "never"** where the reference raises or makes no such claim (A.4).
- **The truth-judged tests** and the property test against the scorer's reverts (B).
- **The scorer**: guard canaries in both modes, `main`'s `kind` leak replayed, narrowed exception handling, dotted-key
  canaries and pairs, the port's missing reference raising, and the tightened licences in its own code (C).
