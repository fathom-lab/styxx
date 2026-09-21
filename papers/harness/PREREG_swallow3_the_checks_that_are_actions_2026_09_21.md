# PREREG — SWALLOW-3: the checks that are actions — the same 30,642 faults, with the declared list

Fathom Lab · 2026-09-21 · Frozen before any fault verdict is taken under the catalogue.
Follows `RESULT_swallow2_which_way_it_falls_2026_09_21.md` (VALID, 5/11).

## Where this came from, stated before anything is measured

SWALLOW-2 injected one fault at a time into 2,178 workflows and read, per fault, whether the
workflow goes red, silently drops a check, hides a check's own failure, or does not care. Its
rule for *what a check is* saw only `run:` steps. Its RESULT §5 named the blind spot in its first
line: **a check that is an action is invisible.** `uses:` steps are never executed and never
counted, so a fault whose only downstream check is an action reads NO_CHECK, and the RESULT's
P4b — `getsentry/sentry-docs`, `Get changed files`, gating a link lint that is an action — missed
for exactly that reason. Its §6 asked for "a declared list of the checking actions, so a gated
action counts as a dropped check when its gate closes."

This cycle is that list, and the measurement of what it changes. It is a *differential* cycle:
the simulation is SWALLOW-2's, unchanged (`faults.py` stays frozen at the sha256 the SWALLOW-2
receipt names, `d26a407c…`); the population is SWALLOW-2's, at the same HEADs; the only thing
added is that catalogued action steps count as checks. Every fault carries both readings —
`verdict_runs_only` (SWALLOW-2's) and `verdict` (this cycle's) — so the difference between the
two receipts is the catalogue and nothing else.

## What was done before this freeze, in order

1. **The rule** below was written first, before the census was read.
2. **The census** (`swallow3_census.py` → `swallow3_actions_census.json`): every `uses:` name in
   the population's workflows, cloned at the HEAD the SWALLOW-2 receipt records for each
   repository (96 cloned; all 96 at the recorded HEAD; the four that were private or gone in
   SWALLOW-2 still are). Names and `with:` keys only; no verdict. 312 distinct action names in
   hand-written workflows (7,451 action steps beside 33,659 `run:` steps); 3,121 local-action
   steps; 349 jobs that call reusable workflows; 31 distinct actions in the generated stratum.
3. **The catalogue** (`CATALOGUE`, `FAMILIES`, `NOT_CHECKS` in
   `benchmarks/harness_mutation/action_checks.py`): the rule applied by hand to all 312 names,
   with the reason for each; 82 entries and 3 families are checks (48 entries appear in this
   population; the rest are well-known checking actions added so the shipped command reads
   other repositories), 262 names are not, by a stated category. Seven entries are marked
   `verified: false` — decided from the action's name and inputs, not its documentation — and
   the receipt counts every check and every drop that rests on one.
4. **Classification only, no verdict**: applying the catalogue to the cloned trees gives 104
   action checks in hand-written workflows across 45 of the 96 repositories (status 29, lint 36,
   security 15, test 12, carried 12); `github/codeql-action/analyze` is 24 of the 104,
   `lycheeverse/lychee-action` 9. Beside SWALLOW-2's 3,191 run-step checks, that is one action
   check for every thirty. This number was known when the predictions below were written, and
   they are calibrated on it. **No fault verdict has been taken under the catalogue** on any
   repository; the instrument was exercised on a synthetic fixture and on this repository's own
   workflows (which contain no catalogued action; every verdict there is unchanged).

## 1. The rule (written before the census was read)

An action step (`uses:`) is a **check** when the action's documented purpose is to produce a
verdict on the repository's code — to run its tests, lint or format-check it, type-check it, or
scan it (its source, its dependencies, its secrets, its links, its workflows) — and either

- (a) fails the step on findings by default (a *step-check*), or
- (b) publishes the findings to a status the repository can require (a *status-check*: CodeQL's
  `analyze`, a SonarQube scan, zizmor's SARIF upload).

**Not a check**: an action that sets up, installs or caches a toolchain; checks out, fetches,
builds, packages, signs, publishes, deploys or releases; uploads or downloads artifacts or
results already produced (a coverage upload carries a runner's verdict, it does not make one);
notifies, comments, labels, assigns, closes, or drafts; judges the pull request's *metadata*
(its title, its commit messages, its labels, its size) rather than its code; or reports a
runner's result without failing the step by default. A *reporter* that fails the step on the
recorded failures by default (`dorny/test-reporter`) **is** a check: after `npm test || true` it
is the gate.

**Inputs decide, when the action documents them.** An input that turns the failing off
(`fail: false`, `soft_fail: true`, `exit-code: 0`, `runTests: false`, `exitZeroOnChanges: true`)
removes the check; an input that turns it on (`fail_on_error: true` or `fail_level:
any|warning|error` on a reviewdog action, `action_fail: true`, `fail_on_failure: true`,
`fail-on-alert: true`, trivy's `exit-code: 1`) adds it; a fixer flag in the inputs (SWALLOW-2's
`FIXER_RX`: `--fix`, `--write`, `:fix`, `:write`) removes it; a formatter run without `check`
(`ruff format`, `black .`) removes it. An action that carries a command (`nick-fields/retry`'s
`command`, `reactivecircus/android-emulator-runner`'s `script`, `vmactions/freebsd-vm`'s `run`,
`CodSpeedHQ/action`'s `run`, `actions-rs/cargo`'s `command`) is a check exactly when SWALLOW-2's
`verification()` calls that command one; an action that carries a *nested* action
(`Wandalen/wretry.action`) is judged as that action.

**Cannot be read, and are counted as such**: a local action (`uses: ./…`), whose definition is
outside `.github/workflows`; a `docker://` image; a reusable workflow called at the job level
(`jobs.<id>.uses`), which the instrument does not descend into; `actions/github-script` and
`nick-fields/private-action-loader`, whose content is arbitrary. `codecov/codecov-action` is not
a check (`fail_ci_if_error` fails on upload errors, not on code); `ossf/scorecard-action` is not
(a score on repository practice, not a verdict on the code); an issue-labeler's `test` is not.

**A check that is an action is never executed.** It is *reached* in a world when its job runs,
no earlier step of the job has failed, and its own `if:` is not false — the simulation already
evaluates every step's `if:`, an action's included, and records why a step did not run. It is
*dropped* by a fault when it is reached in the healthy world and not in the fault world; the
mechanism is the one that closed: `step-if` (the step's own `if:`), `after-failure` (an earlier
step failed in a `continue-on-error` job, so the implicit `success()` is false), `job-if`,
`job-needs`, `job-empty-matrix`. There is no SWALLOWED for an action (no fault is injected into
one) and no counted reading (an action is reached once or not at all). An action check in scope
that is still reached makes a fault ABSORBED where it was NO_CHECK.

The rule is a heuristic, like SWALLOW-2's. Every decision it produced is in the catalogue with
its reason, and the census records every action name it was applied to, so a reader who draws
the line elsewhere can compute the receipt they would have gotten.

## 2. The instrument

`benchmarks/harness_mutation/action_checks.py`, on top of `faults.py`, which it imports and does
not modify. For each fault it calls SWALLOW-2's `_fault_verdict` and then, over the same
memoised executions, asks of every catalogued action step in the fault's scope (its job and the
jobs downstream) whether it is reached in W+ and in W−A. Verdicts keep SWALLOW-2's precedence:
RED > FAIL_OPEN > SWALLOWED > ABSORBED > NO_CHECK; a fault is judged in both healthy flavours and
takes the strongest, for each reading separately. The receipt schema is
`styxx.harness-faults-actions/v1`; it records the sha256 of this file, of `faults.py` (which must
equal the SWALLOW-2 receipt's), of the population, and each repository's HEAD beside the HEAD it
was asked to clone at.

**What can change, and what cannot.** Adding checks can turn NO_CHECK into ABSORBED (a check in
scope, still reached), NO_CHECK into FAIL_OPEN (a check in scope, dropped), and ABSORBED into
FAIL_OPEN (a run check still reached, an action check dropped). It cannot change RED (a red job
is red whatever the checks), cannot change SWALLOWED (that reading is about the faulted `run:`
step itself), and cannot move a verdict down the precedence. Gate G-S3-6 holds this.

## 3. Predictions, committed now

Units are hand-written workflows (not `*.lock.yml`) and interpretable faults unless stated. "Moved"
means `verdict != verdict_runs_only`. The SWALLOW-2 baselines for the same trees are: NO_CHECK
286, ABSORBED 162, FAIL_OPEN 3 (in 2 repositories), of 6,336 interpretable hand-written faults.

**P1 — the blind spot is real but small.** At least **10%** of the hand-written NO_CHECK faults
(≥ 29 of 286) move: to ABSORBED or FAIL_OPEN. The action checks sit where the run checks are
not — docs, links, security, release jobs — which is where the faults that touch no check live.

**P2 — dropped checks at least double, at the repository level.** At least **5** hand-written
FAIL_OPEN faults, in at least **3** repositories (from 3 in 2). The reason it is not more: a gate
fed by an *action's* output (`dorny/paths-filter`) is unknown to the model and lets the step run
in both worlds; only a gate fed by a `run:` step's answer can close under a fault, and SWALLOW-1
counted 40 such query steps in 24 repositories.

**P3 — SWALLOW-2's P4b, resolved.** `getsentry/sentry-docs`, `lint-external-links.yml` →
`check-pr` → step 1 `Get changed files`: **FAIL_OPEN**, with `lycheeverse/lychee-action` among
the dropped checks. Declared non-blind: the SWALLOW-2 RESULT read this fault and said what it
gates.

**P4 — SWALLOW-2's P4c, unchanged.** `primer/react`, `recommend-integration-tests.yml` →
`recommend` → step 2 `Get source files changes`: **not** FAIL_OPEN. What it gates is a comment;
the catalogue does not make a comment a check.

**P5 — the gate is usually in the same job.** Among the action checks dropped in hand-written
FAIL_OPEN faults, `step-if` is the single commonest mechanism (a strict plurality over `job-if`,
`job-needs`, `job-empty-matrix`, `after-failure`). Vacuous if nothing is dropped: then MISS.

**P6 — the map holds.** Fewer than **100** hand-written faults move at all (< 1.6% of the
interpretable): SWALLOW-2's 92% RED, its 23 repositories with a hidden check, and its reading of
the "cannot fail" steps stand as written.

**P7 — status-checks do not gate.** Of the action checks dropped in hand-written FAIL_OPEN
faults, at most **1** is a status-check (`kind: status` — CodeQL, Sonar, zizmor). They live in
their own jobs, behind no shell.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S3-1 (population) | SWALLOW-1's frozen list | 100 entries, sha256 `5ac2789b…` in the receipt |
| G-S3-2 (same trees) | the receipt is made from SWALLOW-2's trees | ≥ 90 cloned; every cloned repository at the HEAD the SWALLOW-2 receipt records (`head_moved` false), or named |
| G-S3-3 (reproduction) | SWALLOW-2 reproduces | for every fault in the SWALLOW-2 receipt, the same (repository, workflow, job, index) is in this receipt with `verdict_runs_only` equal to its verdict; a mismatch is allowed only where one of the two records carries a `timeout`, and every mismatch is counted and named |
| G-S3-4 (closure) | the catalogue covers the census | every name in `swallow3_actions_census.json` is in `CATALOGUE`/`FAMILIES` or in `NOT_CHECKS`, and none in both |
| G-S3-5 (instrument) | the instrument's own tests pass, and this repository is unmoved | `tests/test_harness_action_checks.py` green; on this repository's workflows every verdict equals SWALLOW-2's; the fixture run is deterministic |
| G-S3-6 (monotone) | only the declared transitions | no moved fault outside NO_CHECK→ABSORBED, NO_CHECK→FAIL_OPEN, ABSORBED→FAIL_OPEN; RED and SWALLOWED counts identical between the two readings |
| G-S3-7 (no hand labels) | verdicts and checks are the instrument's | the RESULT may read faults; it may not reclassify one |
| G-S3-8 (ledger) | every prediction scored | HIT/MISS for P1–P7 |

G-S3-1 to G-S3-6 are blocking.

## 5. What would abandon this

The differential claim, if G-S3-3 fails beyond timeouts: a re-clone at the same HEADs that does
not reproduce SWALLOW-2 says the instrument is not deterministic, and this RESULT would say so
before anything else. The catalogue, if G-S3-6 fails: a verdict that moved outside the declared
transitions is a bug in the wrapper, not a finding.

## 6. Honest statement of what a passing SWALLOW-3 means

That the checks SWALLOW-2 could not see have been named, one by one, with the reason; that the
same 30,642 faults were read again with those checks counted; and that the difference is stated
fault for fault. It says how much of SWALLOW-2's NO_CHECK was a check the instrument could not
read, and how many of those checks a fault can silently switch off. It does not execute an
action, so it cannot see a check that is an action fail; it cannot see inside a local action, a
reusable workflow or `github-script`; and the catalogue is a reading of documentation, with seven
entries that are not even that. The shipped `styxx ci-audit` will read with the catalogue by
default, and its differential test will hold the engine to this instrument.

## 7. Running it

```
python papers/harness/swallow3_census.py --work <clones>                       # the census; keeps the clones
python -m benchmarks.harness_mutation.action_checks --repos papers/harness/swallow1_population.json \
    --heads papers/harness/swallow2_receipt.json.gz --work <clones> --keep --out papers/harness/swallow3_receipt.json
python papers/harness/swallow3_score.py
```
