# AMENDMENT — PATH-2: a second freeze, made before any full corpus run and before the prereg was pushed

Fathom Lab · 2026-09-17 · Amends `PREREG_path2_resolution_2026_09_17.md` (sha256 `80502b54ea3fb94a…`,
LF; committed as `f481312a`, carried byte for byte to `36dd23e3` by a rebase onto `origin/main`
`87767853`). The prereg is not edited. This note is committed on its own, after a three-lens review of
the implementation at `ab2c5bde` (rebased to `da099078`) and **before** any code of this round, before
any full corpus run of the scorer, and before the prereg or the branch left this machine. Where the
two documents disagree, this one governs.

## Why a second freeze

The review found one blocker, three majors and eight minors. The blocker is in a rule the prereg froze,
so the repair is a rule change, and a rule change is written down before the code, not after it.

**The blocker (R-101).** `chg` counted every added `def test_NAME` line whose name any removed line of
the same file defined, with no count per name. When a file defines one name more often in added lines
than in removed lines, one removed definition cancelled several added ones, `net` fell below the
number of tests really added, and the `net = n` rule VERIFIED an undercount. The prereg's reason for
trusting `net` ("`net` is the count of definitions new to that file", its R-101 table note) is false
for such files. Reviewer inputs, each run on `87dded26` and on the branch, both doors and the port:

- `TestA.test_run` gains a parameter, `class TestB` adds `test_run`, a new `test_other` is added (two
  tests added). "Added 1 test." — `87dded26` CONTRADICTED `diff adds 3 test functions, claim says 1`;
  branch VERIFIED `diff adds 1 test functions, claim says 1 (2 changed, not added: #101)`. The true
  "Added 2 tests." abstained.
- `TestFoo.test_basic` changes signature and `class TestBar` adds `test_basic` (one test added).
  "Added 0 tests." — `87dded26` CONTRADICTED; branch VERIFIED.
- The EXTERNAL-1 fold, through `external1_harness.reconstruct`: a file the PR created
  (`+def test_x` / `+def test_y`) and a later commit edited (`-def test_x` / `+def test_x(tmp_path)`),
  eligible (parse equals implied). "Added 1 test." — branch VERIFIED `(2 changed, not added: #101)`.
- Non-ASCII names: `[A-Za-z0-9_]*` cut `test_ölen` and `test_ärger` to `test_`, so one changed test
  marked an unrelated new one as changed. "Added 1 test." over one changed and two new — `87dded26`
  CONTRADICTED, branch VERIFIED.

A mutant replacing the name set with a per-name minimum passed all 47 PATH-2 tests: nothing pinned it.

## What was seen before this freeze

1. **The builder's smoke run.** `path2_gates.py corpus --limit 3000` on `external1_shelf.sqlite` ran
   before the scorer was committed, although the prereg says the scorer is committed "before any corpus
   number exists". The file was not changed after the run (committed sha256 `fe9345d9ecc73009…`). It
   printed: 2,986 PRs gated under both instruments; 27 PRs with a moved record (`file_created`
   VERIFIED → UNCHECKABLE 2, `file_deleted` UNCHECKABLE → VERIFIED 1, 36 reason-only path-claim moves);
   0 new accusations; 0 violations; 27 s.
2. **A reviewer's run of the committed scorer.** `--limit 1000`, output to a scratch directory:
   `gated_under_both` 992, `key_moved_prs` 175, `collision_prs` 0, `records_moved` 10, no violations.
3. No other shelf count was seen. The rule changes below come from inputs the reviewers constructed
   (quoted with each change), not from any shelf record. The builder's run reported moves of path
   claims only. The reviewer's report quotes only the counts above.

The RESULT restates both runs.

## Rule changes

### C-1 (#101) — removed and added definitions pair one to one

**Definition lines.** One pattern per kind, the same source text in `styxx/diffgate.py` and
`web/gate/diffgate.js`, matched at the start of one line, applied to removed and added lines alike.
Neither uses `\s`, `\w` or `\b`; both tolerate one leading U+FEFF.

- test: `^\uFEFF?[ \t]*def (test_[^ \t(:]*)` — the name runs to the next space, tab, `(` or `:`.
- symbol: `^\uFEFF?[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+NAME(?=[ \t(:]|$)`, NAME escaped.

**`tests_added`.** For each file of the per-file sides and each test name,
changed = min(added lines defining the name, removed lines defining it). A file whose status is `A`
contributes 0. `chg` = min(Σ changed, `got`). `got` is unchanged (`^\s*def test_` over the added
blob), `net = got − chg`, and the prereg's verdict table and reasons are unchanged. With no pairing,
every verdict and reason is byte-identical to `87dded26`, as before.

**`symbol_added`.** `hit` is unchanged (`^\s*(?:def|class)\s+NAME\b` over the added blob). When `hit`
holds, the claim is UNCHECKABLE with the prereg's (#101) reason only when (a) some file both adds and
removes a definition of NAME and (b) no file adds more definitions of NAME than it removes, reading a
status-`A` file's removed count as 0. When some file adds more than it removes, VERIFIED as today;
with no hit, CONTRADICTED as today.

**The interval, restated.** Reading one removed and one added definition of one name in one file as
one changed definition, every removed line cancels at most one added line, so `net` is a lower bound
on the tests added and `got` an upper bound. `net = n` verifies under that reading. The reading is
wrong when a test is deleted and a same-named test is added in the same file (limit 3 below).

**Why status `A`.** A file the PR creates has no base, so no line of it can be a changed definition.
Removed lines under an `A` header come from the shelf's fold appending a later commit's patch.

**Why explicit patterns.** JavaScript's `\s` matches U+FEFF and Python's does not; JavaScript's `\b` is
ASCII-only. Running the #101 regexes on removed lines added Python/JS disagreements that `87dded26`
does not have: 4 of 14 reviewer inputs against 0. (a) `-\ufeffdef test_a():` / `+def test_a():`,
"Added 1 test." — Python VERIFIED, JS UNCHECKABLE. (b) The same BOM strip on `def backoff(n):`,
"Adds function backoff." — Python VERIFIED, JS UNCHECKABLE. (c) `-def backoffé(n):` /
`+def backoff(n):` — Python VERIFIED, JS UNCHECKABLE. (d) A git repository whose head commit strips
the BOM from a test file and a source file — Python VERIFIED on both doors, JS UNCHECKABLE. Under C-1 a
BOM strip is a changed definition in both ports, and `backoffé` does not define `backoff` in either.

**Why `got` and `hit` stay.** They read the added blob, which the prereg's "What cannot move" keeps.
Changing them would move records that hold no changed definition. The clamp keeps a line that only
the pairing pattern reads (an added line starting with U+FEFF, in Python) from pushing `net` below 0.

### C-2 (#121) — COMPAT's and BC-1's path readings use the undotted key

Reviewer input: a diff removing `export function withTheme` from `.storybook/preview.js`, with "This
change is fully backward compatible." `87dded26`: UNCHECKABLE `… 1 public definition(s) removed, all
in test/example/internal code: storybook/preview.js: withTheme`, `compat2_candidate` false,
`surface_removed` 0. Branch: UNCHECKABLE `… the diff removes 1 public definition(s) from the surface
…: .storybook/preview.js: withTheme`, candidate true, `surface_removed` 1. `_COMPAT_SCAFFOLD` anchors on
`(?:^|/)storybook/` and matched only because the old key dropped the dot. The prereg's G-C4 row for
`compat_claim` let this reclassification through as "reason and detail only".

**Rule.** `_COMPAT_SCAFFOLD` is searched on `_undotted(key)`. The COMPAT language suffix tests and
BC-1's "no Python file in the diff" test read `_undotted(key)` too, so a file named exactly `.py` is
still not a Python file. Paths are still printed with their dots. #121 then cannot change which
removed definitions COMPAT reads as surface, which languages it reads, or whether BC-1 sees Python.
A collision PR can list two `removed` entries where the old key listed one, and both get the same
surface reading. `compat2_candidate` cannot flip.

The JavaScript port has no `_COMPAT_SCAFFOLD` (the COMPAT-2 gap); its suffix tests change the same way.

### C-3 (#121) — `only_touches` splits the outside paths

Reviewer inputs, `87dded26` against the branch, both ports byte-identical:

- `.github/a.yml`, `.github/b.yml`, `.github/c.yml`, `src/x.py`, "Only touches github/." —
  `87dded26` CONTRADICTED `paths outside 'github': ['src/x.py']`; branch CONTRADICTED
  `… ['.github/a.yml', '.github/b.yml', '.github/c.yml']`, never naming `src/x.py`.
- "Only touches .env." over a diff modifying `env` — branch UNCHECKABLE
  `paths outside '.env' differ from it only by a leading dot: ['env'] (#121)`. The dot is on the
  prefix, not the path, and the reason is false.
- "Only touches .github/." over `.github/x.yml` and `github/z.md` — branch UNCHECKABLE, though
  `github/` is a separate directory.
- "Only touches src/." over `--- a/../src/x.py` — branch UNCHECKABLE `… only by a leading dot:
  ['../src/x.py'] (#121)`.

**Rule.** `outside` is computed as today, on dotted keys. An outside path `p` is a **dot miss** when
some prefix `x` read for the claim has a key with no leading dot, `p`'s leading segment starts with
exactly one dot (not `..`), and `p` without that dot equals `x` or lies under `x/`. Every other
outside path is **real**.

| outside paths | verdict | reason |
|---|---|---|
| none | VERIFIED | unchanged |
| dot misses only | UNCHECKABLE | the prereg's (#121) reason, listing the dot misses `[:3]` |
| at least one real | CONTRADICTED | `paths outside …: real[:3]`; dot misses are not listed |

**Consequence, stated.** Two record shapes that `87dded26` VERIFIED become CONTRADICTED: a prefix
written with a leading dot over a changed path without it ("Only touches .env." over `env`), and a
changed path whose key begins with `..`. `87dded26`'s VERIFIED there was false: the old key dropped the
dot, while the diff shows a path outside the prefix as written. That is a new accusation, which the
prereg's G-C3 forbade for `only_touches`. G-C3 and G-C4 below allow exactly this shape. R-121.1's
principle ("may not add an accusation on a path that has no dot") was written for prefixes and paths
that carry no dot (`/docs`). It is narrowed here: it does not cover a prefix that carries the dot.

## Gates, amended

Unit gates (blocking):

- **G-P2** adds: the same test name in two classes; the fold shape under an `A` header and under an
  `M` header; a BOM-stripping edit (test and symbol); non-ASCII test names; `backoffé` removed and
  `backoff` added; `symbol_added` with a same-named method added in another class (VERIFIED) and
  changed in place (UNCHECKABLE); the mixed `only_touches` case; a dotted prefix over an undotted path;
  a `..` path; the `.storybook/` COMPAT-2 reading (Python side); binary dotfile twins with no
  `---`/`+++` lines and a pure rename to a dotted name, on both doors; a file named `.py`.
- **G-P4.** `check_pairs.js`: 0 disagreements on every pinned pair. The full differential's only
  disagreements are `compat_claim` records from the COMPAT-2 port gap: the 9 that disagree before
  PATH-2, plus the pinned `.storybook/` pair, which cannot agree while the port lacks COMPAT-2. Its
  pinned `expect` compares kind and verdict for the port, and the Python side also pins the reason and
  the candidate flag.

Corpus gates:

- **G-C0 (provenance, blocking; new).** Every payload the scorer writes, differential and corpus,
  records `scorer_sha256` (path2_gates.py, LF), `harness_sha256` (external1_harness.py, LF), the git
  `HEAD`, and whether those two files and `styxx/diffgate.py` are unmodified against `HEAD`. A
  payload written from a modified tree fails this gate.
- **G-C3 (amended).** Claim by claim, CONTRADICTED after and not before: 0 for every kind except
  `files_changed_count` on collision PRs, and `only_touches` VERIFIED → CONTRADICTED where the claim's
  prefix key carries a leading dot or the PR holds a changed path whose repaired key begins with `..`
  (C-3).
- **G-C4 (amended; attribution per claim where the claim names something).**

  | kind | may move only when | verdict moves allowed |
  |---|---|---|
  | `file_created`, `file_deleted` | #97: the any-tier-in-diff-order and tiered resolutions differ over the repaired status map; or #121, per claim: the claim's key moved, or the entry it resolves to is a moved key (under the baseline, the any-tier resolution over the baseline map is a filename's baseline key whose key moved; under the repair, the tiered resolution over the repaired map is such a filename's repaired key) | VERIFIED ↔ UNCHECKABLE |
  | `file_touched` | the same | VERIFIED ↔ UNCHECKABLE under #121; reason only under #97 alone |
  | `files_changed_count` | collision PR | any |
  | `only_touches` | #121: a filename's or a prefix's key moved (the claim reads every path of the PR) | VERIFIED → UNCHECKABLE with a reason ending `(#121)`; VERIFIED → CONTRADICTED under the G-C3 exception; else reason only |
  | `tests_added` | #101: a file whose repaired status is not `A` in which one test name is defined by an added and a removed line under C-1's test pattern | V→U, C→V, C→U, U→V; else reason only |
  | `symbol_added` | #101: a file whose repaired status is not `A` in which the claimed name is defined by an added and a removed line under C-1's symbol pattern | VERIFIED → UNCHECKABLE with a reason ending `(#101)` |
  | `compat_claim` | #121, per claim: a path in the claim's detail (`removed[].path`, `signature_changed[].path`) is, under the baseline, the baseline key of a filename whose key moved, or, under the repair, its repaired key | none (reason and detail only) |
  | `tests_pass` | never | none |

- **G-C6 (blocking; new).** `compat2_candidate` flips: 0, in either direction, on every PR gated under
  both instruments. No rule in the prereg or here flips it (C-2).
- **G-C5 (report only), added.** New VERIFIED `tests_added` records on PRs where one file defines a
  test name in more added lines than removed lines, beside the prereg's count on PRs whose file rows
  repeat a filename. A reviewer proposed making these blocking at 0. That is not taken: under
  one-to-one pairing, `net` is a lower bound, and such a record is right under the pairing reading
  (`TestA.test_run` edited and `TestB.test_run` added, "Added 1 test"). A blocking 0 would fail on
  correct records. The exposure that remains is limit 3, and it is counted.

## Issue #121's own acceptance gates: two deviations

1. **G-BIN-1 on the 91 EXTERNAL-5 diffs** (parse count equals raw header count on 85 of 85) is not
   run here. `bin1_gates.py` re-fetches those diffs from the network, and the shelf does not hold them.
   `bin1_gates.json` and `bin2_gates.json` stay as committed. If the RESULT is licensed to fetch, it
   may run it and report both counts.
2. **"The only records allowed to move are ones whose `why` quoted a stripped dotfile path"** is
   replaced by G-C4's per-claim attribution. That rule admits a moved record only when the claim's own
   key, the entry it resolves to, a prefix key or a detail path is a key that moved (and
   `files_changed_count` only on a collision PR). Matching reason text would also admit a path claim
   whose reason never quoted a path (the bare-name and does-not-appear branches).

## Disclosed limits, not repaired

1. **#97 within one tier.** Two suffix matches, or two basename matches, still resolve to the earliest
   in diff order. "Created src/util.py." over `M a/src/util.py` then `A b/src/util.py` is UNCHECKABLE,
   and VERIFIED with the files swapped. A claim naming a directory that is not in the diff resolves by
   basename ("Created docs/README.md." over `A pkg/README.md` is VERIFIED). `87dded26` and the branch
   agree on both. Each would become a false CONTRADICTED if `WITHHOLD_PATH_ACCUSATION` were lifted.
   Repairing it needs its own freeze.
2. **Dotfile renames join the old rename blind spot.** After `git mv .eslintrc.json eslintrc.json`,
   "Deleted .eslintrc.json." reads `'.eslintrc.json' is a bare name absent from the diff` on both doors.
   With a directory component it reads "does not appear in the diff — accusation WITHHELD". The old
   key matched the renamed name by accident. Non-dot renames behave this way on `87dded26`: the
   `rename from` path is never registered.
3. **The pairing reading (C-1).** A test deleted and a same-named test added in the same file read as
   one changed test, so `net` can undercount by one and "Added 0 tests." can verify. A removed
   docstring or string line that starts `def NAME(` pairs like a definition. A rename or a move between
   files counts as added (prereg). A change git detects as a rename (≥ 50 % similar) pairs, while the
   same change below the threshold reads as D + A and verifies. PEP 695 `def f[T](` is not a definition
   line for either pattern, so it reads as today. `got` still does not count `async def test_`
   (prereg), but the symbol pattern pairs a removed `async def NAME` with an added `def NAME`.
4. **The fold under `A`.** On the shelf, a file the PR created and a later commit edited carries both
   patches under one `A` header. `got` counts both definition lines, as on `87dded26`, and C-1
   subtracts nothing under `A`. That record reads as it did on `87dded26`.
5. **Python and JavaScript on the added blob.** An added line starting with U+FEFF counts toward `got`
   and `hit` in JavaScript and not in Python. A name followed by a non-ASCII letter hits in JavaScript
   and not in Python. Both predate PATH-2 (`87dded26` and its port disagree the same way), and the
   fuzzer writes neither. C-1's pairing reads such lines identically in both ports, so it adds no
   disagreement where the two agreed. The clamp keeps `net` ≥ 0 when only the pairing pattern reads
   a line.
6. Keys stay lower-cased; the JavaScript port still lacks COMPAT-2 (prereg).
7. **The differential is not yet a committed receipt.** `path2_gates.py differential` and the full
   corpus run are regenerated for the RESULT and committed with it. The payload carries
   `moved_record_ids` and G-C0's provenance. On `87dded26`, `py_side.py` refuses to run: its pin
   `397624d5…` names a file that is not `473a7dd7…`. So the RESULT's "the same nine disagree before
   PATH-2" is measured by importing `87dded26:styxx/diffgate.py` beside `87dded26:web/gate/diffgate.js`
   over the same corpus files, and the RESULT states that procedure.
8. Receipts whose scripts import `_norm` or the gate may read differently on a re-run: `bin1_gates.json`
   and `bin2_gates.json` (the one twin diff, 644 → 645), and possibly `compat2_gates.json`,
   `compat2_gate_summary.json`, the EXTERNAL-3, HARNESS-1 and EXTERNAL-5 summaries and cross-check, and
   `external1_summary.json`. None is regenerated. `HANDOFF_capsule_v02_2026_08_31.capsule.html` already
   fails verification on `origin/main`, and fails the same way on this branch.

## What this amendment does not license

No edit to the prereg. No change to R-97's tiers, to R-121's key, to the `tests_added` verdict table,
to the withholding flags, or to any gate other than those named above. If the corpus gates fail under
these rules, the branch does not land and the moved records are read, as the prereg says.
