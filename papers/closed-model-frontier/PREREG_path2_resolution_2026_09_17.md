# PREREG — PATH-2: an exact path before a basename, a dotfile keeps its dot, a changed `def` is not an added one (#97, #121, #101)

Fathom Lab · 2026-09-17 · Frozen before the repair is written, on branch `fix/diffgate-path-resolution`
at `origin/main` `87dded26`, where `styxx/diffgate.py` hashes to `473a7dd7c2dce7b1…` (LF) and
`web/gate/diffgate.js` to `15a31c2a0d9d60da…`. Same shape as BC-2 and BIN-2: mechanical defects,
the repairs written down before the code, unit gates a test suite and the differential can score,
and corpus gates that say which records may move and in which direction. Three issues share one
freeze because all three change what a record says about a path or a definition, and a corpus
comparison that scores them has to attribute every moved record to exactly one of them.

## The defects, reproduced on `87dded26`

Output of `gate_diff_text` (and of `gate_diff` on a two-commit repository where stated), verbatim.

### #97 — an earlier basename match shadows an exact one

`find_path` inside `_gate` returns the status entry that clears *any* of exact / suffix / basename,
in diff order. A diff that modifies `README.md` and then creates `integrations/git/README.md`:

    "Created integrations/git/README.md."
    UNCHECKABLE  file_created  'integrations/git/README.md' is status 'M', claim wants 'A' — accusation WITHHELD pending the EXTERNAL-1 repair

With the `README.md` hunk dropped the same sentence reads `VERIFIED … diff status 'A' for
'integrations/git/readme.md'`. With the two files in the other order, "Modified README.md." reads
`VERIFIED file_touched diff status 'A' for 'integrations/git/readme.md'` — the right verdict on
the wrong file. Withholding keeps both from being accusations; both are false records.

### #101 — a `def` line that changed counts as added

`tests_added` and `symbol_added` count definitions in the added lines only. The issue's diff
(`backoff(n)` gains a parameter; `test_a` and `test_b` gain a trailing comment; nothing is added),
"Adds function backoff with jitter. Added 2 tests.":

    gate_diff_text                     PASS
      VERIFIED  symbol_added   added lines do define function 'backoff'
      VERIFIED  tests_added    diff adds 2 test functions, claim says 2
    gate_diff (git, HEAD~1..HEAD)      PASS
      VERIFIED  symbol_added   added lines do define function 'backoff'
      VERIFIED  tests_added    diff adds 2 test functions, claim says 2

Two false VERIFIED, the class the gate's own README ranks worst.

### #121 — `_norm` strips every leading dot

`_norm` is `p.replace("\\", "/").lstrip("./").lower()`; `lstrip` removes any run of `.` and `/`.

    _norm('.pr_agent.toml') = 'pr_agent.toml'   _norm('pr_agent.toml') = 'pr_agent.toml'
    _norm('.github/workflows/ci.yml') = 'github/workflows/ci.yml'   _norm('.env') = 'env'

A diff deleting `pr_agent.toml`, adding `.pr_agent.toml` and modifying `.github/workflows/ci.yml`
(three `diff --git` headers) parses to `{'pr_agent.toml': 'A', 'github/workflows/ci.yml': 'M'}`, and
"3 files changed. Created .pr_agent.toml. Deleted pr_agent.toml. Only touches src/." reads

    FAIL
      CONTRADICTED  files_changed_count  diff changes 2 files, claim says 3
      VERIFIED      file_created         diff status 'A' for 'pr_agent.toml'
      UNCHECKABLE   file_deleted         'pr_agent.toml' is status 'A', claim wants 'D' — accusation WITHHELD pending the EXTERNAL-1 repair
      CONTRADICTED  only_touches         paths outside 'src': ['pr_agent.toml', 'github/workflows/ci.yml']

— a truthful count accused, a deletion recorded as the other twin's addition, and reasons that
print paths no diff contains.

## The repairs, as they will be written

### R-97 — resolve in tiers over all entries

`find_path(claimed)` normalises the claim once and scans the whole status map three times: an entry
equal to the claim; else an entry ending in `"/" + claim`; else an entry whose basename equals the
claim's basename. Within a tier, diff order decides, as before. A claim no tier matches is `None`,
exactly when it was `None` before (the tiers partition the same test), so the bare-name and
does-not-appear branches cannot move. Nothing else in the path-claim verdict changes.

### R-121 — the key keeps its dots

1. **The key.** `_norm(p)` = backslashes to slashes, then remove a leading run of `/` and `./`
   segments (`^(?:\.?/)+`), then lower-case. `.pr_agent.toml`, `.github/x`, `.env`, `..env` and
   `../x` keep their dots; `./src/x`, `/src/x` and `.//src/x` still read `src/x`. The issue proposed
   `^(?:\./)+`; a leading `/` is also removed here because the old key removed it and "only touches
   /docs" would otherwise become an accusation on a diff whose paths are all under `docs/` — a
   repair to a dotfile defect may not add an accusation on a path that has no dot.
2. **Every caller at once.** `_norm` is the one normaliser for both sides — `parse_unified_diff`,
   `parse_unified_diff_sides`, the BIN-1 header registration, `gate_diff`'s `--name-status` keys, the
   claim in `find_path`, the `only_touches` prefixes and the reasons that quote them. None of them
   gets its own copy.
3. **`only_touches`: a miss by a leading dot abstains.** When some changed path lies outside the
   prefix(es), and every such path would lie inside once the key's and the prefix's leading dots and
   slashes are dropped (the old comparison, `lstrip("./")` on both), the claim is UNCHECKABLE with
   `paths outside '<prefix>' differ from it only by a leading dot: [...] (#121)` (`'<a>' and '<b>'`
   and `from them` when two prefixes are read) — prose that writes `github/` for `.github/` is not
   accused on the dot. Any other outside path accuses as today.
4. **`only_touches`: the path-shape test reads the old segments.** BC-2's test for a prefix with no
   `/`, `\` or `.` ("a directory segment or file of some changed path") compares against the
   segments of the changed path with its leading dots dropped, as it did, so `src and github.` keeps
   `github` as a path-shaped prefix; whether paths are under it is then decided by 3.

### R-101 — a definition the removed lines of the same file also define is changed, not added

Both doors already hand `_gate` the per-file sides (`parse_unified_diff_sides` over the raw text, or
over `git diff base..head`); the repair reads them there, so the two doors agree by construction.

- **`tests_added`.** `got` is today's count, `^\s*def test_` over the added lines. `chg` is the number
  of added lines matching `^\s*def (test_[A-Za-z0-9_]*)` whose name a removed line of the **same
  file** also defines. `net = got − chg`. When `chg = 0` every branch and every reason is byte-
  identical to today. When `chg > 0`, with `n` the claimed count:

  | condition | verdict | reason |
  |---|---|---|
  | no Python file in the diff | UNCHECKABLE | unchanged (#110) |
  | `net = n` | VERIFIED | `diff adds {net} test functions, claim says {n} ({chg} changed, not added: #101)` |
  | counted noun is case / file / scenario / suite / class | UNCHECKABLE | the #110 reason with `net`, same suffix |
  | `net < n ≤ got` | UNCHECKABLE | `diff adds {net} test functions and changes {chg}, claim says {n}; a changed test is not an added one (#101)` |
  | `n < net` or `n > got` | CONTRADICTED | `diff adds {net} test functions, claim says {n} ({chg} changed, not added: #101)` |

  The decision, stated: the true number of added tests lies between `net` and `got` under any
  reading of a changed `def` line. A count inside that interval is abstained on — the only reading
  that agrees with it counts a changed test as added, and the gate does not accuse on a reading.
  A count outside it is wrong under both readings; it was CONTRADICTED before and stays so. `net = n`
  verifies: in a real `base..head` diff a name in the removed lines of a file existed at base, so
  `net` is the count of definitions new to that file.
- **`symbol_added`.** When the added lines define `NAME` (today's search), and every added line that
  defines it (`^\s*(?:def|class)\s+NAME\b`, line by line) sits in a file whose removed lines define it
  too, the claim is UNCHECKABLE: `added lines define {kind} {NAME!r} only where the removed lines of
  the same file define it too; a changed definition is not an added one (#101)`. The issue suggested
  CONTRADICTED ("redefined, not added"); it is not taken, because a diff can pair an unchanged `def`
  line as removed-and-added when the body around it moves, and a signature change described as
  "adds function X with Y" is loose prose rather than a lie the diff proves. Any added definition in
  a file whose removed lines do not define the name verifies as today; no definition at all is
  CONTRADICTED as today.
- **Renames verify.** `back_off` → `backoff` with "adds function backoff" stays VERIFIED; a renamed
  test counts as added. The symbol is new to the file even when the body is not.

### The port

`web/gate/diffgate.js` carries all three repairs with the same reasons byte for byte; the bookmarklet
is rebuilt with `build_bookmarklet.py`; `web/gate/differential/path2_pairs.json` pins pairs for every
branch above with `expect` blocks written from the repaired Python, read by `py_side.py`,
`js_side.js` and `check_pairs.js`; `py_side.py`'s pin moves to the repaired file.

## What cannot move, by construction

- Which sentences are read and which claims they yield (kind, text, template groups): extraction
  reads the summary alone, and no repair touches it. `tests_pass` is untouched.
- A path claim cannot become CONTRADICTED while `WITHHOLD_PATH_ACCUSATION` is true; a
  `compat_claim` cannot leave UNCHECKABLE while `COMPAT2_LICENSED` is false.
- R-97 moves only `file_created` / `file_deleted` / `file_touched` records, and only where two entries
  match the claim under different tiers.
- R-121 moves records only on a diff holding a path whose key changes, or a claim or prefix whose key
  changes; a file count only where two filenames shared an old key.
- R-101 moves only `tests_added` and `symbol_added` records, only where a file's removed lines define
  a name its added lines define.
- The added-lines blob, the demo, and the BIN-1 registration rule are unchanged.

## Gates — committed now

### Unit gates, scored by `tests/test_diffgate_path2.py` and the differential. Blocking.

- **G-P1 (the reproductions).** Every test that pins a reproduction fails on `87dded26` and passes
  after. After the repair: #97, "Created integrations/git/README.md." →
  `VERIFIED diff status 'A' for 'integrations/git/readme.md'`, and the reversed diff's "Modified
  README.md." → `VERIFIED diff status 'M' for 'readme.md'`. #101, on both doors →
  `UNCHECKABLE symbol_added` and `UNCHECKABLE tests_added` with the reasons above ("diff adds 0 test
  functions and changes 2, claim says 2"), gate PASS. #121, the status map has three keys
  `pr_agent.toml` D, `.pr_agent.toml` A, `.github/workflows/ci.yml` M; the count VERIFIED at 3; the
  creation VERIFIED on `.pr_agent.toml`; the deletion VERIFIED on `pr_agent.toml`; `only_touches`
  CONTRADICTED naming all three paths with their dots.
- **G-P2 (constructed edge cases).** A suffix match beats an earlier basename match; an exact match
  beats an earlier suffix match; a basename still resolves when nothing stronger exists; `./x`, `/x`
  and `.//x` key as `x`, `../x` and `..env` keep their dots; "only touches github/" over a `.github/`
  diff abstains with the #121 reason and "only touches .github/" verifies; "only touches src and
  github." keeps both prefixes; a new test beside a changed one verifies "Added 1 test"; a count
  inside `(net, got]` abstains and outside it accuses; a changed `def` in one file beside a fresh one
  in another verifies the symbol; a rename verifies; a test moved to another file counts as added
  (the disclosed limit, pinned).
- **G-P3 (the doors agree).** On a two-commit git repository for each of the #101 diff, a dotfile
  twin diff and the two-README diff, `gate_diff` and `gate_diff_text` over `git diff` return the same
  claims, verdicts and reasons.
- **G-P4 (the differential).** `check_pairs.js`: 0 disagreements on every pinned pair, the new ones
  included. The full corpus (3,205 pairs today plus the new ones): the only disagreements are the
  `compat_claim` records that already disagree on `87dded26` — 9, because COMPAT-2's reading was
  never ported — and they disagree identically. Every one of the 3,205 existing Python records that
  changes between `87dded26` and the repair is attributed by G-C4's rules, and the list is reported.
- **G-P5 (suite, demo, bookmarklet).** The diffgate test modules and every test importing
  `styxx.diffgate` or reading `web/gate` are green; `--demo` prints the same bytes;
  `build_bookmarklet.py --check` matches the rebuilt bookmarklet.

### Corpus gates, scored later on the EXTERNAL-1 shelf

Baseline: `styxx/diffgate.py` at `87dded26` (sha256 pinned). Repaired: the branch head. Both run in
one process over every PR of `external1_shelf.sqlite`, through `external1_harness._fold_statuses`
and `reconstruct`, unchanged. Eligibility per instrument is EXTERNAL-1's (empty body, no file
records, reconstruction mismatch), with the implied status map keyed by *that instrument's* `_norm`.
`styxx.claimdetect` is blocked in both runs: it feeds `unparsed_claims` only and never a verdict, and
nothing below reads that field. No ledger is written and no PR is named. Scorer:
`papers/closed-model-frontier/path2_gates.py corpus`, committed on this branch after this document
and before any corpus number exists.

A **key-moved PR** holds a filename whose key differs between the two normalisers. A **collision
PR** holds two distinct filenames that share a baseline key and not a repaired key.

- **G-C1 (the same claims).** On every PR eligible under both, the ordered list of
  (kind, text, template groups) is identical. Blocking.
- **G-C2 (eligibility moves only where a key moved).** Every PR eligible under exactly one
  instrument is a key-moved PR. Counts both ways reported. Blocking.
- **G-C3 (no accusation added, but a hidden file counted).** Claim by claim, CONTRADICTED after and
  not before: 0 for every kind except `files_changed_count`, and for that kind only on collision PRs.
  Blocking.
- **G-C4 (every moved record attributed, in an allowed direction).** A claim moves when its verdict
  or its reason differs (for `compat_claim`, also its detail). Blocking.

  | kind | may move only when | verdict moves allowed |
  |---|---|---|
  | `file_created`, `file_deleted` | #97: the any-tier-in-diff-order resolution and the tiered resolution differ over the repaired status map; or #121: the claim's key or some filename's key moved | VERIFIED ↔ UNCHECKABLE |
  | `file_touched` | the same | VERIFIED ↔ UNCHECKABLE under #121 (a match found or lost by a dot); reason only under #97 alone |
  | `files_changed_count` | collision PR | any |
  | `only_touches` | #121: a filename's or a prefix's key moved | VERIFIED → UNCHECKABLE with a reason ending `(#121)`; else reason only |
  | `tests_added` | #101: a file whose added and removed lines both define one `def test_` name | V→U, C→V, C→U, U→V; else reason only |
  | `symbol_added` | #101: a file whose added and removed lines both define the claimed name | VERIFIED → UNCHECKABLE with a reason ending `(#101)` |
  | `compat_claim` | key-moved PR | none (reason and detail only) |
  | `tests_pass` | never | none |

- **G-C5 (report only).** Transitions by kind and direction; key-moved and collision PR counts;
  claims by verdict and accusations by kind for each instrument; the new VERIFIED `tests_added` on
  PRs whose file rows repeat a filename (the shelf's fold appends every commit's patch under one
  header, so a test added in one commit and edited in a later one reads as changed there and nowhere
  else — exposure, reported, not scored); `compat2_candidate` flips by direction.

## What is not claimed

- No precision number. The path accusation stays withheld; nothing here licenses its return.
- #97: when no exact or suffix match exists, the basename tier still resolves a claim that names a
  directory to a same-named file elsewhere. The issue's second suggestion (decline those) is a
  different recall trade and is not taken here.
- #101: the same-file rule counts a test moved between files, and a renamed test, as added;
  `async def` is not counted, as before; the shelf's per-commit fold can read a test the PR added
  and later edited as changed.
- #121: keys stay lower-cased, so `README.md` and `readme.md` still share one key; that collision is
  older than this issue and not repaired here.
- The JavaScript port's missing COMPAT-2 reading is not repaired; G-P4 carries it as a fixed,
  known set.
- No committed receipt is regenerated. Receipts whose scripts import `_norm` or the gate
  (`bin1_gates.json` and `bin2_gates.json` certainly, on the one diff with the twin; others such as
  `compat2_gates.json` and the EXTERNAL-3 summaries possibly) may read differently on a re-run; the
  RESULT lists them, and the files stay as they are.

## What failure means

A failure of G-P1 to G-P5 blocks the corpus run. A failure of G-C1 to G-C4 means the repair moved a
record none of the three rules explains: the branch does not land, the moved records are read, and
a second freeze follows, as BC-1 → BC-2 and BIN-1 → BIN-2 did.

---

*Two files named README, a signature that grew a parameter, and a dot in front of a filename: each
one let the gate print a record the diff does not say. The rules above are what the diff does say,
written down before the lines that implement them.*
