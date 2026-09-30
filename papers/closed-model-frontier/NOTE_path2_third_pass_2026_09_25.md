# NOTE — PATH-2, third pass: reconciled with PATH-1, and six rule changes round 2 forced

2026-09-25. Branch `fix/diffgate-path-resolution` (pull request #161), on top of
`PREREG_path2_resolution_2026_09_17.md`, `AMENDMENT_path2_resolution_2026_09_17.md` and
`ERRATUM_path2_amendment_2026_09_17.md`.

**Those three documents are public and frozen, and nothing here edits them.** Both pull requests are
open, so the preregistration, the amendment and the erratum are the record as filed. Everything this
round changes — including two changes to rules the amendment froze, and one change to the protocol
the preregistration set out — is written down here instead, beside them, with the evidence that
forced it. Where this note and the amendment disagree about what the instrument does, this note is
what the code does; the amendment stays as the statement made before the measurement.

---

## A. The branch was rebased onto `98a5c368`, and PATH-1 rewrote the code PATH-2 touches

The branch was cut from `87dded26`. It has been rebased onto `origin/main` `98a5c368`, which now
carries PATH-1 (#127, `PREREG_path1_only_touches_repair_2026_09_17.md`), the COMPAT-2 port (#126) and
DECLARE-1 (#129, #130). PATH-1 rewrote `only_touches` containment — the same reading C-3 splits — so
the two repairs had to be combined rather than one replacing the other.

### The combined `only_touches` rule

Read in this order. PATH-1 decides what is outside; PATH-2 decides what an outside path means.

1. **Prefix keys.** `prefs = [_norm(prefix).rstrip("/.")]`, and a second prefix only if it is
   path-shaped. `_norm` is PATH-2's: it drops a leading run of `/` and `./` segments and nothing
   else, so `.github/x` keeps its dot and `../docs` keeps both of its.
2. **Is the prefix a path at all?** `_prefix_is_path_shaped` (BC-1/BC-2) now carries both repairs:
   a slash or backslash makes it a path; **PATH-1** then requires a real file extension from the
   committed list before a dot alone counts, so `Assert.NotNull` is not a path; failing both,
   **PATH-2** reads the changed paths' segments on the *undotted* key, so `github` still names the
   `.github` directory. If the prefix is not a path, the verdict is UNCHECKABLE and nothing below
   runs.
3. **Is the prefix a repo path?** *(new this round, R-3)* A key that opens with two dots — `..`,
   `...`, `../docs`, `.../src/x.py` — is UNCHECKABLE.
4. **What is outside.** `outside = [p for p in status if not any(_path_inside(p, x) for x in prefs)]`,
   where `_path_inside` is **PATH-1's**: a bare filename with a real extension matches on the
   basename anywhere in the tree, anything else anchors at the repository root.
5. **What an outside path means.** *(PATH-2, C-3)* An outside path is a **dot miss** when the path's
   leading segment starts with exactly one dot, some prefix key has no leading dot, and the path
   without that dot is **inside** that prefix — *by `_path_inside` again*, the same containment test
   step 4 used. Every other outside path is **real**.
6. **The verdict.** Dot misses alone → UNCHECKABLE, naming them. Any real path → CONTRADICTED,
   listing `real[:3]` and no dot miss. Nothing outside → VERIFIED.

Where the two overlap they agree, and the one place they could have disagreed is step 5. C-3 as
frozen read containment as `rest == x or rest.startswith(x + "/")`, which is PATH-1's *non*-bare arm.
Using `_path_inside` there instead means a bare-filename prefix keeps one meaning throughout. The
change is **behaviourally equivalent**, and the note says so rather than claiming a repair: a
bare-filename prefix that would contain `rest` by basename already contains `"." + rest` by basename,
so such a path is never outside and step 5 is never reached for it. Mutant N6 (step 5 reverted to the
frozen wording) survives every test, and that is correct.

The port carries the identical rule. `_prefixOffTree`, `_dotMiss` and the `outside` filter in
`web/gate/diffgate.js` are transliterations of the six steps above.

### Two more things the rebase moved

- **`_COMPAT_SCAFFOLD` in the port.** C-2 makes COMPAT's scaffold test read the undotted key. The
  port did not carry COMPAT-2 when C-2 was written, and #126 brought COMPAT-2 in with the scaffold
  test reading the dotted key. The port now reads `_COMPAT_SCAFFOLD.test(_undotted(path))`, as the
  Python does. Without this the two implementations disagreed on `.storybook/`.
- **The `python_only` escape is gone.** One pinned pair
  (`path2:121-compat2-a-dotted-scaffold-directory-stays-scaffolding`) was pinned for the Python alone
  because the port lacked COMPAT-2. The port has it, the pair is pinned at full width on both sides,
  and `tests/test_diffgate_path2.py` asserts that no pair needs the escape.

### Two committed PATH-1 expect blocks were re-pinned

`web/gate/differential/path1_pairs.json` carries two pairs whose expected *reason* was written
against the pre-#121 key, which printed paths with their leading dots stripped. The verdicts do not
move; the printed strings do:

| pair | verdict | before | after |
|---|---|---|---|
| `path1:css-selector` | UNCHECKABLE, unchanged | `prefix 'k-step-link' is not a path (#110)` | `prefix '.k-step-link' is not a path (#110)` |
| `path1:unrepaired-typo` | CONTRADICTED, unchanged | `paths outside 'githiub/…': ['github/…']` | `paths outside '.githiub/…': ['.github/…']` |

Both new strings are more faithful to what was claimed and what changed: the claim really says
`.k-step-link`, and the file really is `.github/workflows/dependabot.yml`. This is an edit to a
committed expect block of a merged change, which the lab does not do lightly; it is recorded here
because the alternative is a differential that fails on a reason PATH-2 is supposed to change.

---

## B. Does the reconciled rule move any of issue #128's eleven?

**No verdict moves. Two reasons do.** Issue #128 audited eleven `only_touches` accusations by hand;
after PATH-1 the instrument accuses on eight of them, two correctly, precision 0.25.

This was read by reconstructing, from #128's own text, the five rows whose claim **and** whose
changed paths the issue states, and running them through `origin/main`'s instrument and this
branch's. It is not a re-fetch of the live diffs, and the other six rows quote a claim without
quoting the paths, so they are not reconstructible here.

| row | main | this branch |
|---|---|---|
| `microsoft/vscode-azuretools#2086` | VERIFIED, `all changed paths under prefix` | identical |
| `ydb-platform/ydb#25857` | VERIFIED, `all changed paths under prefix` | identical |
| `dotnet/runtime#117821` | UNCHECKABLE, `prefix 'assert.notnull' is not a path (#110)` | identical |
| `open-policy-agent/cert-controller#415` | CONTRADICTED, `paths outside 'githiub/workflows/dependabot.yml': ['github/…']` | CONTRADICTED, `paths outside '.githiub/workflows/dependabot.yml': ['.github/…']` |
| `microsoft/wassette#442` | CONTRADICTED, `paths outside 'changelog.md': ['github/workflows/…']` | CONTRADICTED, same three paths, printed with their dots |

For the six not reconstructed, the argument is the rule, not a measurement: this round's
`only_touches` changes act on exactly three shapes — a changed path whose leading segment starts with
one dot (the dot-miss split), a prefix key that keeps a leading dot (the printed string, and C-3's
accusation class), and a prefix key opening with two dots (the new abstention). The Memoria,
mikepenz, fern-api and Azure rows quote prefixes with no leading dot (`mods/submods`, `app1/`,
`README/documentation`) and name no dotfile among the paths that decided them, so none of the three
shapes applies. If a dotfile does sit among their changed paths, the effect is the printed path, not
the verdict, because a `.github/` path is not inside `mods/submods` with or without its dot.

**So the honest number is unchanged: precision stays 2 in 8 = 0.25 on this corpus.** PATH-2 is not a
repair of #128 and this note does not claim it as one. #128's modes 2, 3, 5 and 6 — runtime-behaviour
sentences, documentation prose, prose containing a slash, and typos in the stated path — are
untouched. `cert-controller#415` above is mode 6 reading a shade more honestly and still accusing
falsely.

---

## C. Rule changes this round

Each is named by the round-2 reviewer finding that forced it. All are mirrored in
`web/gate/diffgate.js`; the pinned pairs hold both implementations to them.

### R-1 — `got` counts a leading U+FEFF, in both implementations

**Reviewer evidence (correctness lens, major; protocol lens, minor).** Diff
`-def test_a():` / `+﻿def test_a():` / `+def test_b():`, claim "Added 0 tests.": the branch at
`ab3084d9` answered **VERIFIED** `diff adds 0 test functions, claim says 0`, and abstained on the
true claim "Added 1 test.". `87dded26` got both right. This is a **new false VERIFIED of exactly the
#101 kind C-1 was frozen to remove**, and neither `G-C4` nor `G-C5` could see it.

**Mechanism.** C-1's pairing pattern carries `﻿?`; Python's `got` pattern was `^\s*def test_`
and Python's `\s` does not match U+FEFF. So the pairing subtracted a line `got` had never added.
`chg = min(chg, got)` clamps the *total*, not line by line, so it only bound when `got` was 0 — the
degenerate case the committed test happened to pin.

**Change.** `got` becomes `^﻿?\s*def test_` (`_GOT_TEST_LINE`). Every line the added-side
pairing pattern matches is now a line `got` counts, so `chg <= got` holds line by line and the clamp
is redundant rather than load-bearing. The clamp stays.

**Why this way and not the reviewer's option 1.** The reviewer's proposal was to pair only added
lines that *each port's own* `got` counts. That fixes the false VERIFIED but leaves the two
implementations with different `got` values (Python 1, JavaScript 2 on the input above) and therefore
different `chg`, different `note` strings, and a different verdict wherever the `net < n <= got`
abstention applies — on `Added 2 tests.` over that diff, Python would accuse and JavaScript would
abstain. Counting the BOM in `got` on both sides instead satisfies the same requirement and leaves
Python and JavaScript on the same verdict **and** the same reason.

**What it also closes.** Amendment limit 5 and round-2 open question 7 disclosed that Python and
JavaScript still read an ADDED line starting with U+FEFF differently. On the `tests_added` half they
no longer do: the same corpus with `origin/main` on both sides shows 2 `tests_added` disagreements on
these inputs, and 0 after. The `symbol_added` half (`hit`, which still uses `\s` and `\b`) is
**unchanged and still disclosed** — see section D.

**Pinned:** `path2:r1-a-bom-on-a-changed-test-does-not-hide-a-new-one`,
`path2:r1-a-bom-on-a-changed-test-beside-two-new-ones`; tests
`test_r1_an_added_bom_line_is_counted_by_got_as_well_as_by_the_pairing`,
`test_r1_a_bom_on_a_changed_test_does_not_hide_a_new_one`. Mutant N1 (the revert) fails 3 tests.

### R-2 — the REMOVED side of the test pairing accepts `async`

**Reviewer evidence (correctness lens, minor).** `-async def test_fetch():` / `+def test_fetch():`:
the true claim "Added 0 tests." was CONTRADICTED and the false "Added 1 test." VERIFIED, on the
branch, on the port and on `87dded26`. Converting an async test to sync counted as adding a test —
the #101 defect, in a shape C-1 did not cover.

**Change.** `_DEF_TEST_LINE_REMOVED` accepts an optional `async` before `def`; the added-side pattern
does **not**. The asymmetry is deliberate and is R-1's constraint: `got` does not count
`async def test_`, so accepting `async` on the added side would part the pairing set from `got`
again. `papers/closed-model-frontier/path2_gates.py` carries the same asymmetry (`DEF_TEST_REMOVED`)
so its attribution sees this pairing too.

**Pinned:** `path2:r2-an-async-test-made-sync-is-a-changed-test`; test
`test_r2_an_async_test_made_sync_is_a_changed_test_not_an_added_one`. Mutant N2 fails 2 tests.

### R-3 — a prefix key opening with two dots abstains

**Reviewer evidence (correctness lens, minor).** "Only touches ../docs/ from the package." over a
diff that modifies `docs/guide.md`: the branch answered CONTRADICTED
`paths outside '../docs': ['docs/guide.md']`, where `87dded26` answered VERIFIED. Same for
`./../docs` and `.../src/x.py`. Git emits no changed path beginning with `../`, so **every** changed
path is outside such a prefix and the gate accused whatever the pull request did. The scorer's G-C3
exception admitted every one of them without a report.

**The decision, stated.** The verdict for a `..` prefix is **UNCHECKABLE**, with the reason
`prefix '../docs' is relative to a directory the diff does not name (#121)`. Abstaining is the lab's
habit where the claim is not a repo path, and this claim is not one: `../docs` names a location
outside the tree the diff describes, and no changed path can be judged against it. One off-tree
prefix is enough — "Only touches ../docs/ and src/." abstains rather than accusing on the half that
can be read.

**Change.** `_prefix_off_tree(pref)` is true when the key opens with a dot and is not a dotfile name,
where a dotfile name is one dot followed by a name character (`_DOTFILE_PREFIX = ^\.[^./\\]`). The
branch is placed after BC-1's not-a-path abstention and before the dot-miss split. C-3's new
accusation class is thereby narrowed to what the amendment actually argued for: a prefix written with
exactly one leading dot over a path without one ("Only touches .env." over `env`), which still
accuses. `path2_gates.py`'s G-C3 exception is narrowed the same way and now admits `.env` and refuses
`../docs`.

**Pinned:** `path2:r3-a-relative-prefix-is-not-a-repo-path`,
`path2:r3-an-elided-prefix-is-not-a-repo-path`, `path2:r3-one-off-tree-prefix-abstains-for-both`;
test `test_r3_a_prefix_written_with_two_dots_is_not_a_repo_path_and_abstains`. Mutant N3 fails 2
tests. Mutant N3b (`pref.startswith("..")` alone) survives and is equivalent: a prefix key is
`rstrip("/.")`-ed, so it cannot end in a dot or slash, and the leading-slash-segment rule means it
cannot begin `./`; the only second character that can be in `./\` is a dot, on which the two
predicates agree.

### R-4 — four rules that no test could tell from a mutant

The round-2 test-strength lens found rules that every committed test passed with and without. Each
is now pinned. No behaviour changes here; the instrument reads exactly as it did.

- **M10b — COMPAT's language suffix on REMOVED lines.** Reverting the second suffix site to the
  dotted key passed 65/65, while flipping `compat2_candidate` False → True, which is what C-2 says
  cannot happen. The committed `.py` test used a one-file diff, where no language is present, so the
  site was never reached. Now pinned with a two-file diff (`.py` plus `src/a.py`), asserting the
  reason, `surface_removed 0`, `compat2_candidate False` and `removed == []`.
  Test `test_r4_the_compat_language_suffix_reads_the_undotted_key_on_removed_lines_too`, pair
  `path2:r4-compat-reads-the-undotted-key-on-removed-lines`. Mutants N5, N5b and N5c all fail tests.
- **M13 — C-3's path-segment boundary.** Dropping the `/` from the dot-miss containment passed
  65/65 and turned a correct accusation into an abstention: "Only touches docs/." over
  `.docsearch.json`. Now pinned as CONTRADICTED `paths outside 'docs': ['.docsearch.json']`, with
  `_dot_miss` asserted directly on `.docsearch.json`/`docs` and `.githubx/a.yml`/`github`.
  Test `test_r3_c3_reads_a_path_segment_boundary_and_not_a_name_prefix`, pairs
  `path2:r4-a-dot-miss-needs-a-path-segment-boundary`,
  `path2:r4-a-dotfile-whose-name-starts-with-the-prefix`. Mutant N4 fails 2 tests.
- **The symbol-rule boundaries, including the one the erratum restates.** Erratum item 2 says a
  changed generic definition still verifies, and no test said so. Five mutants survived round 2.
  Now pinned: a changed generic definition VERIFIED (the `[` stops the lookahead — mutant N9, the
  lookahead as `\b`, fails); a function made generic VERIFIED (rule (a) needs a file that both adds
  and removes the name — mutant N10 fails); an NBSP-indented removed definition VERIFIED (the indent
  is `[ \t]*`, not `\s*` — mutant N11 fails); a space before the parameter list UNCHECKABLE (the
  space is inside the lookahead); a class name at end of line UNCHECKABLE (the lookahead accepts
  `$`). Test `test_r4_the_symbol_rule_boundaries_the_erratum_restates`, five pairs `path2:r4-…`.
- **PATH-1's own boundaries, now under this branch's tests too.** Mutants N7 (`outside` without
  `_path_inside`), N8 (the prefix shape admitting any dot) and N12 (a bare filename without the
  extension test) each fail 2 to 6 tests in `tests/test_path1.py` and `tests/test_diffgate_path2.py`
  run together, so the reconciliation cannot silently drop PATH-1.

### R-5 — the scorer says what it borrows, checks how a key moved, and hashes its inputs

**Reviewer evidence (protocol lens, two minors).**

1. `path2_gates.py`'s docstring said attribution was implemented "independently of the repair …
   not imported from the repaired module". It called `new._norm`, `new.parse_unified_diff`,
   `new.parse_unified_diff_sides` and `new._header_paths` at eight sites. A repaired `_norm` that
   stops lower-casing — which the preregistration says cannot happen — moves keys, and every
   attribution rule asks only *whether* a key moved, so the defect attributes itself to #121 and
   passes G-C4.
2. In differential mode, missing generated corpora were reported on stderr only. The payload still
   read `"pairs": 67`, `"all_attribution_gates_pass": true`, and exit 0 — and amendment limit 7
   makes that payload the RESULT's committed receipt. No input file was hashed.

**Changes.**

- The docstring names the borrowed inputs one by one, and says which rules are the scorer's own.
- **New blocking gate `G-C4_key_moved_not_by_a_dot`:** for every changed path and every claimed path
  or prefix, `new._norm(p).lstrip("./")` must equal `BASE._norm(p)`. A key may move by dropping
  leading dots and by nothing else. Probed directly: a `_norm` that stops lower-casing reports
  `['Docs/Guide.md', 'README.md', 'src/Retry.py']`; one that drops a path segment reports two; the
  real one reports none.
- **Differential mode hashes every input** into `payload["inputs"]` (sha256, byte size, item count),
  lists `payload["missing_inputs"]`, and a missing input is a `G-C0_missing_corpus_input` violation
  that makes the run exit non-zero. A run with no generated corpora now fails instead of writing a
  receipt with no corpus.
- `DIFFERENTIAL_FILES` gains `compat2_pairs.json`, `path1_pairs.json` and `declare1_pairs.json`, so
  the scorer reads the same corpus the differential harness does.

**Not done:** corpus mode still records only the shelf's file name, with no byte size and no row
counts. The reviewer asked for those too. They are cheap, but they need a run against a shelf to
check, and this round is under instruction not to look at the real shelf again. Owed to the RESULT.

### R-6 — the scorer's baseline moves from `87dded26` to `98a5c368`

**This is a change to the protocol the preregistration set out, not a reviewer finding.** The
preregistration names `87dded26` as the baseline: "the instrument before the repair". After the
rebase that commit is three merged changes behind, and scoring against it reads PATH-1, the COMPAT-2
port and DECLARE-1 as PATH-2's work. Measured: **9 `G-C1` claim-set differences, 4 unattributed
`only_touches` moves and 2 unattributed `tests_added` moves**, none of them this branch's, and the
gates fail.

`BASE_COMMIT` therefore becomes `98a5c368`, the rebase target, so that every move the gates see is a
move this branch makes. `PREREG_BASE_COMMIT` and `PREREG_BASE_SHA256` are kept beside it and every
payload records `baseline_commit`, `prereg_baseline_commit` and `baseline_moved_from_prereg`, so no
number can be quoted without the fact that the baseline moved. With the baseline at `98a5c368`,
G-C1 passes and every `only_touches` move is attributed.

One mechanical consequence: the baseline file carries `from .declare import declaration_pass`
(DECLARE-1), so `load_base()` sets `__package__ = "styxx"` on the module it execs. `styxx/declare.py`
is byte-identical on main and on this branch — the branch touches one file under `styxx/` — so the
baseline reads with the same `declare` main has.

---

## D. What is still not repaired

Carried forward from the amendment, re-read this round and still true.

1. **#97 order dependence within one tier** (amendment limit 1). A claim naming a directory can still
   resolve by basename, and diff order still decides inside a tier. Identical on the baseline;
   lifting the withholding would turn it into a false CONTRADICTED and needs its own freeze.
2. **Dotfile renames** (amendment limit 2). The `rename from` path is never registered, so
   `git mv .eslintrc.json eslintrc.json` reads "`.eslintrc.json` is a bare name absent from the
   diff". The same class as non-dot renames on the baseline. The pure rename **to** a dotted name is
   pinned on both doors.
3. **`symbol_added` recall on a removed docstring line** (amendment limit 3). A removed line like
   `def backoff(n): documented here` still pairs, so the claim abstains.
4. **The status-A exclusion keeps the shelf fold's over-count** (amendment limit 4). A created-then-
   edited test file reads `fold-A 'Added 2 tests.'` as CONTRADICTED `adds 3`, as on the baseline.
5. **`hit` and the `symbol_added` half of the BOM and non-ASCII gap** (amendment limit 5, narrowed).
   R-1 closes the `tests_added` half. `hit` still uses `^\s*(?:def|class)\s+NAME\b`, so Python and
   JavaScript still read an ADDED symbol definition starting with U+FEFF differently, and still
   differ on a name followed by a non-ASCII letter (Python's `\b` is Unicode-aware, JavaScript's is
   ASCII). Neither shape can produce a false VERIFIED — `_definition_only_changed` is consulted only
   when `hit` is true, and a false `hit` gives CONTRADICTED, not VERIFIED — so the class is left
   where round 2 left it rather than widened in a round whose brief was to close findings. It is a
   candidate for its own freeze.
6. **`path2_differential_gates.json` is still not committed** (amendment limit 7). It is meant to be
   regenerated and committed with the RESULT rather than pinned now and regenerated in place later.
   At the head of this round, from a clean tree, its numbers are the ones in
   `web/gate/README.md`.
7. **Corpus mode's input provenance** (R-5, above): the shelf's size and row counts are owed.
8. **Issue #128's modes 2, 3, 5 and 6** are not repaired and are not scheduled. See section B.

---

## E. What was run

- `tests/test_diffgate*.py`, `tests/test_path1.py`, `tests/test_port_is_current.py`,
  `tests/test_declare1.py`, `tests/test_bc2_receipt.py`, `tests/test_compat1_receipt.py`,
  `tests/test_external2_receipt.py`, `tests/test_capsule_v02.py`, `tests/test_charon.py`,
  `tests/test_claimdetect.py`, `tests/test_undeclared.py`, `tests/test_evidence.py`,
  `tests/test_sworn.py`, `tests/test_sworn_action.py`, `tests/test_git_commit_msg_hook.py` and the
  codex / cursor / gemini hooks: all pass. `tests/test_gitlab_job.py` fails on this machine only —
  its `bash` is a WSL stub — and is ignored, as it was in round 2.
- `node web/gate/differential/check_pairs.js`: **106 pinned pairs, 0 disagreements** (52 of them
  PATH-2's).
- The full differential, corpora generated and then deleted: **3282 pairs, 7028 claims (653 verified,
  1640 contradicted, 4735 uncheckable), 0 disagreements**. The same corpus with `origin/main` on both
  sides: 3282 pairs, 7028 claims (658 / 1646 / 4724), **2** disagreements, both on the new BOM pairs,
  both closed by R-1.
- `path2_gates.py differential`: 58 records moved, Python and the port moving the same 58; 3 new
  `only_touches` accusations (the dotted-prefix pairs the amendment allows); 0 `compat2_candidate`
  flips; every attribution gate passes.
- Mutation check, `styxx/diffgate.py` copied alone into a scratch directory and loaded through a
  pytest plugin: control 93/93. **Killed:** N1, N2, N3, N4, N5, N5b, N5c, N7, N8, N9, N10, N11, N12.
  **Survived, equivalent:** N3b and N6, argued above.
- `path2_gates.py corpus` was **not** run against the real shelf. The orchestrator scores it after
  this round; looking again here would repeat round 1's disclosure problem.
