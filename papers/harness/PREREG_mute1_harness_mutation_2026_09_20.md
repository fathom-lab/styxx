# PREREG — MUTE-1: cut the wire, and see whether the alarm still rings

Fathom Lab · 2026-09-20 · Frozen before the instrument is run against a single mutant. Population
enumerated; no verdict computed.

## Where this came from, stated before anything is measured

In two days this repository was found to be carrying four checks that reported green and measured
nothing: `gauntlet-pr.yml` (its diff failed and `|| true` filed the failure as an empty change set,
#137), `telescope.yml` (a key-check skip hiding a gitignored corpus, #138), the port differential on
`main` (a stale pin that nothing ran, #136), and the styxx-js typecheck (declared in `package.json`,
invoked by nothing, #140). Every one was found by a person reading a file. The repair in every case
was the same: a test that reads the harness and fails when the check is cut.

`benchmarks/silent_pass/` names this shape — an absent measurement surfacing as a passing check —
and holds twenty cases of it, all Python modules, all found by hand. Its own CONTRIBUTING file says
its biggest weakness is that it is one codebase deep. This preregistration says the shape is also
one *level* deep: the corpus catches it in programs, and the four cases above are in the apparatus
that checks the programs.

Mutation testing asks of a test suite: if I break the program, does a test go red? **MUTE-1 asks
the same question one level up.** If I cut a check out of CI — delete the job, silence the step,
flip its guard, delete the file it reads, stop the workflow firing — does anything in this
repository go red?

## The rule

The instrument is `benchmarks/harness_mutation/mute.py`. Its vocabulary is closed and is stated in
its docstring; it is restated here so the document does not depend on the code.

A **check** is a workflow trigger, a workflow job, a `run:` step, an npm script, or a **subject** —
a file a check reads and is meaningless without. Subjects are declared, not inferred; the list is
the `SUBJECTS` table in the instrument and is pinned by this document's hash.

Seven **operators**:

| operator | what is cut |
|---|---|
| M-TRIGGER | the workflow's `on:` becomes `workflow_dispatch` only; it never fires by itself |
| M-JOB | the job is deleted |
| M-STEP | the `run:` step is deleted |
| M-SWALLOW | the `run:` step is wrapped `( … ) \|\| true`; it can no longer fail |
| M-GUARD | the step's `if:` becomes `false`; it never runs |
| M-SCRIPT | the npm script is deleted |
| M-SUBJECT | the subject file is deleted, or its load-bearing line removed |

The **oracle** is the repository's own test suite restricted to the files whose source matches a
fixed pattern for harness paths (`ORACLE_PATTERN` in the instrument). The pattern is the
definition; the list it selects is written into the receipt. Nothing is hand-picked into or out
of the oracle.

The **verdict** on a mutant is computed per test against the unmutated tree:

- **KILLED** — some test that passed on the unmutated tree fails on the mutant.
- **SURVIVED** — every test that passed on the unmutated tree still passes.
- **UNREACHED** — there was nothing to cut (the file, job, step, guard or line is absent).

A test that fails on the unmutated tree is excluded and named in the receipt; it cannot kill
anything and is not allowed to.

## The cost, stated first

**MUTE-1 measures whether the test suite guards the harness. It does not run CI.** A SURVIVED
verdict means exactly: *nothing in `tests/` would tell you.* A mutant that only a runner could
catch — a deleted `uses: actions/checkout`, a step that fails on ubuntu-latest — is out of scope on
purpose, because the oracle must be runnable by anyone with a clone, and this repository's stated
discipline (`tests/test_ledger.py`, `tests/test_port_is_current.py`) is that a guard lives in the
suite, not in the hope that somebody watches the Actions tab.

A survivor is therefore **a finding to be read, not a defect to be counted.** Some checks have a
loud failure elsewhere: cutting `publish.yml`'s build job breaks the next release, at release time,
in front of whoever is releasing. Others fail silently everywhere: cutting the step that runs the
test suite breaks nothing anyone can see. The RESULT may sort survivors into those two bins after
reading them, and must say that the sorting is a judgement made after the run, not a measurement.

Mutants are applied by YAML round-trip; comments are dropped. The oracle reads YAML semantically
or runs the step's shell, so a comment is never load-bearing for a verdict.

## The population, enumerated before the run

`--inventory` on the tree this document is frozen against reports **74 checks and 120 mutants**:

| operator | mutants |
|---|---|
| M-TRIGGER | 9 |
| M-JOB | 14 |
| M-STEP | 38 |
| M-SWALLOW | 38 |
| M-GUARD | 8 |
| M-SCRIPT | 4 |
| M-SUBJECT | 9 |

The oracle pattern selects ten test files; they are listed in the receipt.

## Predictions, committed now

Each is scored HIT or MISS in the RESULT. A MISS is a fact about the author's model of the
repository, and is published as such.

**P1 — the four repairs are real.** These mutants are KILLED, and the receipt names the test that
killed each:
- M-JOB `test.yml → typecheck-js`
- M-SUBJECT `.gitignore` line `!telescope/prompts.json`
- M-SUBJECT `web/gate/differential/py_side.py` `PINNED`
- M-STEP and M-SWALLOW on `gauntlet-pr.yml → verify-submissions → Discover changed submissions`.
  The M-SWALLOW case is the interesting one: the textual guard in
  `tests/test_gauntlet_pr_verifies_something.py` looks for `|| true` on a git line and will not see
  a wrapper on its own line; the *behavioural* guard, which runs the step's shell and demands a
  non-zero exit, will. If this is a HIT it is a HIT for behavioural guards over textual ones.

**P2 — most jobs have no guard.** Of 14 M-JOB mutants, **exactly 2 are KILLED** (`typecheck-js`,
and `gauntlet-pr.yml → verify-submissions`, whose guard looks the job's step up by name) and
**12 SURVIVE**.

**P3 — the headline.** M-STEP and M-SWALLOW on `test.yml → test → Run tests` both **SURVIVE**.
Stated plainly: *if CI stopped running the test suite, nothing in the test suite would notice.*
The guard of guards is unguarded.

**P4 — triggers are unguarded.** Of 9 M-TRIGGER mutants, **8 SURVIVE**; the one KILLED is
`leaderboard-submission.yml`, because a test in the gauntlet file reads its `on.pull_request.paths`.

**P5 — guards are unguarded.** All 8 M-GUARD mutants **SURVIVE**.

**P6 — one script of four is guarded.** M-SCRIPT `typecheck` is KILLED; `build`, `test`,
`test:watch` SURVIVE.

**P7 — a textual guard is fooled by a swallow.** M-SWALLOW on the two `run:` steps of
`typecheck-js` (`npm ci`, `npm run typecheck`) **SURVIVE**, because the guard written in #140
checks that the commands are *present*, not that they can *fail*. If this is a HIT it is a
finding against the author's own guard, and it is predicted anyway.

**P8 — the totals.** Of 120 mutants: **17 KILLED, 1 UNREACHED** (`telescope/prompts.json`, which is
not in the tree), **102 SURVIVED**. The prediction is that more than four in five cuts to this
repository's checking apparatus are invisible to its test suite.

## Gates

| gate | what it holds | bar |
|---|---|---|
| G-M1-1 (control) | the oracle is alive | ≥ 1 test passes on the unmutated tree, and ≥ 1 mutant is KILLED |
| G-M1-2 (instrument) | the instrument reaches its population | UNREACHED ≤ 5 of 120; the one expected is named above |
| G-M1-3 (no hand labels) | verdicts come only from the per-test comparison | the RESULT may read survivors; it may not reclassify one |
| G-M1-4 (ledger) | every prediction above is scored | HIT/MISS for P1–P8, published whatever they say |

G-M1-1 and G-M1-2 are blocking: a run that fails either is INVALID and is not cited.

## What would abandon this

The instrument, not the claim, if G-M1-1 or G-M1-2 fails. The claim, if fewer than half the
mutants survive: that would mean the suite already guards the harness far better than the four
findings suggested, and the honest RESULT is that the pattern was four instances, not a class.

## Honest statement of what a passing MUTE-1 means

A run that survives its gates says which of 120 cuts to this repository's checking apparatus its
test suite notices, and which it does not, with the test named for every one it notices. It does
not say the survivors are defects. It does not say CI would miss them — it says the suite would.
It is a reading list with a receipt, produced by a program rather than by a person reading nine
workflow files on a Sunday, and it can be re-run on any commit.

## Running it

```
python -m benchmarks.harness_mutation.mute --inventory
git worktree add /tmp/mute-tree HEAD
python -m benchmarks.harness_mutation.mute --run --tree /tmp/mute-tree
```

The instrument refuses to mutate the checkout it lives in. The receipt lands at
`papers/harness/mute1_receipt.json` and carries the tree's commit, the instrument's own sha256, the
oracle's file list, the excluded tests, and every verdict with the tests that produced it.

## Amendment A — 2026-09-20, appended before the instrument was run

This document was frozen at sha256
`f569701e1a2fc6b71bbff8336d51d413f6e1291d94d3224a1ec525c392df9fbc`. Everything below is appended;
nothing above it is edited. No mutant had been applied when this was written.

**A1 — the instrument's own tests are not part of the oracle.** `tests/test_harness_mutation.py`
was written after the freeze and matches the oracle pattern, because it names the harness paths
it copies into a scratch directory. It is excluded by name (`ORACLE_EXCLUDES` in the instrument),
for one reason: a mutation tester that used its own tests to decide whether its mutants were
noticed would be grading its own homework. The frozen enumeration of **ten oracle files** stands
unchanged, and that is the list the receipt must show. This removes a freedom the frozen text
left open; it does not widen one.
