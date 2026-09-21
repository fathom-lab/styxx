# RESULT — SWALLOW-2: one fault at a time through 2,177 workflows — 92% of faults are loud, a check whose own failure is hidden is common (23 repositories), and a check silently dropped is rare (2)

Fathom Lab · 2026-09-21 · Scores the receipt `swallow2_receipt.json.gz` against the
preregistration frozen at sha256 `bdae83fbdb9a5d3beea2ecc8c35a213feed97ff285a08f5d4997a07bdd80225a`.
Not amended. The instrument was revised three times after the freeze, and the population run
repeated each time; §1 states what changed, why, and that the predictions and their scoring
definitions did not.

Receipt: `papers/harness/swallow2_receipt.json.gz` (24 MB of JSON, gzipped; sha256 of the JSON
recorded by the scorer in `swallow2_scored.json`) · instrument `benchmarks/harness_mutation/faults.py`,
sha256 `b086476d1cdc2cc5…` · population `swallow1_population.json` (sha256 `5ac2789b6d67eed8…`,
SWALLOW-1's 100, re-cloned; each HEAD in the receipt) · 1,135 s, 94,971 step executions ·
method gate `swallow2_self.json` · scored by `swallow2_score.py`.

**VALID. 5 of 11 predictions HIT** (P1, P4d, P4e, P5, P7). **30,645 faults** were injected — one
per bash step that reaches a tool — into **2,177 workflows** of 87 repositories; 7,915 of the
faults are in hand-written workflows and 6,339 of those are interpretable. Of the interpretable
hand-written faults, **92.1% are RED**: the workflow goes red when that step's tools fail.
**49 are SWALLOWED** — a check ran, its tools failed, and the workflow stayed green — in **23
repositories**. **3 are FAIL_OPEN** under the preregistered definition — a check that would have
run was silently not run — in **2 repositories**; 11 in 7 under the counted reading added after
the freeze (§1, §4). The misses are the predictions that fail-open would be common (P2, P3), the
three named steps from SWALLOW-1's reading that turn out to protect something other than a
recognised check (P4a–c), and the coverage bar (P6: 80.1% of fault sites interpretable against
85% predicted).

## 0. What was done

SWALLOW-1 said, of the step that cannot fail, that it could not see which way the step falls. This
cycle simulates the workflow. For each workflow file, two healthy worlds — every external command
succeeds and prints `x` (flavour `x`), or prints nothing and `grep` reports no match (flavour
`empty`) — and, for each bash step A that reaches a tool when executed alone, one fault world:
A's external commands fail, everything else stays healthy. Steps run in job order with what
Actions gives them: `$GITHUB_OUTPUT` and `$GITHUB_ENV` read back, `${{ steps.*.outputs.* }}`,
`${{ env.* }}` and `${{ needs.*.outputs.* }}` substituted, every `if:`, `needs:` and `fromJSON`
matrix evaluated by a three-valued evaluator in which what the simulation cannot know is unknown
and lets a step run. A fault's verdict, in precedence: **RED** (a job goes red that was not red in
the healthy world), **FAIL_OPEN** (a check that reached its runner in the healthy world does not
in the fault world), **SWALLOWED** (A is a check, reached its runner, and nothing went red),
**ABSORBED** (every check reaches its runner as before), **NO_CHECK** (no check in scope);
BASELINE_RED / BASELINE_SKIPPED when A itself fails on its own logic, or does not run, in both
healthy worlds. A check is a step the stated rule `verification()` recognises: a known
test / lint / typecheck runner or a test-named script in its body, or a check-word in its name
with no install / build / setup / report / … word beside it.

## 1. Gates, and the deviations

| gate | bar | observed | |
|---|---|---|---|
| G-S2-1 (population) | SWALLOW-1's frozen list | 100 entries, sha256 `5ac2789b…` | pass |
| G-S2-2 (reach) | ≥ 90 cloned; failures and capped repositories named | 96 cloned; `Smart-Cleaner-for-Android`, `antiwork/flexile`, `antiwork/helper`, `Metta-AI/metta` gone or private; none capped (`githubnext/gh-aw`, 338 workflows, took 538 s of the 900 allowed) | pass |
| G-S2-3 (method) | alone verdicts equal SWALLOW-1's self-census; #137 seen in both states | 33 PROPAGATES, as before; `gauntlet-pr.yml` at `main`: in W−discover both gated steps skipped and the job green; at #137: RED | pass |
| G-S2-4 (instrument) | `tests/test_harness_faults.py` green; self-run deterministic | 8 passed; identical twice | pass |
| G-S2-5 (no hand labels) | verdicts and checks are the instrument's | §4 reads; nothing reclassified | pass |
| G-S2-6 (ledger) | P1–P7 scored, P4 as five | below | pass |

**Deviations, stated.** The preregistration was frozen before the population was touched and was
not edited afterwards; its predictions and the scorer's definitions are the frozen ones. The
instrument, described there in prose, was revised three times after the freeze, and the
population run was repeated from scratch each time. Every revision is a commit with the reason in
its message; the receipts of the superseded runs were not scored and are not published, but
their hand-written verdict counts are given here so the reader can see what each revision moved.

| run | instrument | stopped or scored | why the next one |
|---|---|---|---|
| 1 | `758a28d9…` | stopped at repository 37, unread beyond the per-repository log | the log showed BASELINE_SKIPPED dominating one repository (airbyte, 159 of 329): a step gated on `== 'true'` was skipped because the model's `x` compared unequal to every literal, and a job whose matrix came from a script inside `$( )` ran zero times because that script got no stub. Three changes: `x` is an answer of unknown content in conditions (non-empty, equal to nothing); stubs at every command position; `git diff --quiet` answers by status per flavour |
| 2 | `7b33e881…` | scored, 5/11; hand-written RED 5,841 · SWALLOWED 49 · ABSORBED 161 · NO_CHECK 284 · FAIL_OPEN 4 | P4a's miss was read: mlflow's `Run tests` calls the same script for its query and its runner, and that script had no stub because an argument on a continuation line was taken for a command and created first as a file. Continuation lines joined; stubs created deepest first. The *counted* reading of a dropped check (§4) added beside the preregistered one, never in its place |
| 3 | `f6eeb38a…` | scored, 5/11; RED 5,836 · SWALLOWED 49 · ABSORBED 161 · NO_CHECK 283 · FAIL_OPEN 4 | one FAIL_OPEN (`githubnext/gh-aw`) was read and was the sandbox's missing directory, not a tool failure: `cd sub && npm test` failed on the `cd`. The fault world now keeps the healthy world's directories; the alone verdict keeps SWALLOW-1's world |
| 4 | `b086476d…` | **scored, 5/11; this RESULT** | — |

Three of the four scores are the same 5 of 11, and the fourth would have been too; what moved is
one FAIL_OPEN, and the reading of it. That is the honest size of what the revisions did.

## 2. Predictions, scored

| | prediction | observed | |
|---|---|---|---|
| P1 | RED ≥ 80% of interpretable hand-written faults | **5,836 of 6,339 (92.1%)** | **HIT** |
| P2 | ≥ 6 repositories with a hand-written FAIL_OPEN fault | **2** (`antiwork/gumroad`, `crewAIInc/crewAI`); 7 under the counted reading | **MISS** |
| P3 | ≥ half of hand-written FAIL_OPEN faults cross-step | 1 of 3 (the other two in-step, `unreached`) | **MISS** |
| P4a | `mlflow/mlflow` `master.yml` → `database` → `Run tests`: FAIL_OPEN, in-step | **SWALLOWED**: the runner script is reached — for the service query, never for the run; FAIL_OPEN under the counted reading (2 → 1) | **MISS** |
| P4b | `getsentry/sentry-docs` `Get changed files`: FAIL_OPEN, cross-step | **NO_CHECK**: the link lint it gates is an action (`uses:`), which the instrument does not execute | **MISS** |
| P4c | `primer/react` `Get source files changes`: FAIL_OPEN, cross-step | **NO_CHECK**: what it gates is a recommendation comment, not a check | **MISS** |
| P4d | `carverauto/serviceradar` `Decide whether this Mix project needs lint`: not FAIL_OPEN | **RED** — the fail-closed branch fails the job when the query fails | **HIT** |
| P4e | `airbytehq/airbyte` `Check for changes`: not FAIL_OPEN | six steps of that name with `git diff`: RED, NO_CHECK ×4, one uninterpretable; none FAIL_OPEN | **HIT** |
| P5 | SWALLOWED in ≥ 8 repositories, and in more than FAIL_OPEN | **23** repositories against 2 | **HIT** |
| P6 | interpretable ≥ 85% of hand-written fault sites; artifacts ≤ 10% of steps run | **80.1%** (849 BASELINE_SKIPPED, 727 BASELINE_RED); artifacts 6.96% | **MISS** |
| P7 | ≥ 90% of live hand-written checks RED under their own fault; the rest SWALLOWED or FAIL_OPEN | **1,401 of 1,452 (96.5%)**; 49 SWALLOWED, 2 FAIL_OPEN | **HIT** |

## 3. The map

**Every hand-written fault**, 7,915 of them, by verdict: RED 5,836 · ABSORBED 167 · NO_CHECK 284
· SWALLOWED 49 · FAIL_OPEN 3 · BASELINE_SKIPPED 849 · BASELINE_RED 727. The generated stratum
(22,730 faults, `*.lock.yml`): RED 17,195 · NO_CHECK 2,354 · ABSORBED 1,204 · SWALLOWED 118 ·
FAIL_OPEN 1 · uninterpretable 1,858.

**SWALLOW-1's steps that cannot fail, resolved.** 349 hand-written fault sites are SWALLOWS when
executed alone — the shape SWALLOW-1 counted in 67 of 87 repositories. In their workflows:

| the step that cannot fail, when its tools fail | faults |
|---|---|
| there is no check in its job or downstream (NO_CHECK) | 145 |
| every check still reaches its runner (ABSORBED) — the cleanups, the comments, the `docker pull \|\| true` | 126 |
| a job goes red anyway (RED) — a later step fails on the empty answer: `serviceradar`'s fail-closed branch, `firecrawl`'s `git diff --quiet` guard, `vscode`'s leak check | 16 |
| it is itself a check and its failure is hidden (SWALLOWED) | 18 |
| a check is silently not run (FAIL_OPEN) | 1 (`gumroad`, `run_scope`) |
| uninterpretable | 43 |

So of the 306 interpretable "cannot fail" steps, **271 (89%) touch no check** and 16 more are
fail-closed through the step after them. SWALLOW-1's 77% was true and was mostly benign. The
40% of repositories that SWALLOW-1 called *a check that cannot fail* was largely a category
heuristic's word *test*; under the tighter rule and the simulation it is 18 checks whose own
failure is hidden plus one that is skipped, in the shell — and 33 more hidden by
`continue-on-error`, which SWALLOW-1 counted beside the shell and this cycle can now place.

**`continue-on-error` steps, resolved** (216 hand-written fault sites): NO_CHECK 93 · SWALLOWED
33 · ABSORBED 25 · RED 24 · FAIL_OPEN 1 · uninterpretable 40. The 33 are two thirds of all
SWALLOWED faults: **the commonest way a check's failure is hidden in this population is not a
`|| true` in the script but a `continue-on-error: true` on the step** — visible in the YAML, never
in the log.

**The checks themselves** (P7): 1,452 hand-written checks reach their runner in the healthy
world; under their own fault 1,401 go red. The 49 SWALLOWED: 33 by `continue-on-error`, 16 by
the shell — `pnpm lint:colors:code || true` (`giselle`, named *non-blocking*), `langfuse`'s
*SQL-equivalence tests (non-blocking)*, `mlflow`'s `set +e` loop, `prebid`'s `files=$(… ) || true`
before the linter, `hmis`'s `if grep -q … ; then exit 1; fi` (green when grep fails),
`nodetool`'s mutation testing.

## 4. The fail-open faults, read

Under the preregistered definition, three:

- **`antiwork/gumroad` `tests.yml` → `run_scope`** — cross-job, the shape of #137 at the job level
  and the one clean instance in the population. `LABELS=$(gh api …/pulls --jq … || true)`; when
  `gh` fails, `LABELS` is empty, `full=false` is written, and `test_fast` and `test_slow`, both
  gated `needs.run_scope.outputs.full == 'true'`, are skipped. A failed labels query silently
  turns the full suite off. (In the `empty` flavour the same fault is ABSORBED, because there the
  healthy world already answered "no labels"; the `x` flavour is the one that can see it.)
- **`antiwork/gumroad` `ci-green.yml` → `report`** — in-step, and **fail-closed by design**: the
  script's own comment says *"unknown is the safe stall, not a pass"*; when the fetch of the head
  fails it sets `COLLISION=unknown` and never reaches `bin/check-migration-versions`, and the
  job is green — but the signal this job exists to publish is a commit status, `ci/green`, which
  the script then leaves unset. The instrument sees a job, not a status; by its definition the
  check was not run and nothing went red. Right by the rule, wrong about the repository.
- **`crewAIInc/crewAI` `update-test-durations.yml`** — in-step: `PYTHON_VERSION_SAFE=$(echo … |
  tr '.' '_')` then `uv run pytest --store-durations …`, with `continue-on-error: true` on the
  step; when the pipeline fails the step exits before `uv` and the job stays green. A durations
  cache that silently does not refresh — a check in name, housekeeping in effect.

The **counted reading** — a check that reaches its runner *fewer times* than in the healthy world
has done less — adds eight, all in-step (`fewer`): `mlflow`'s `Run tests` (2 → 1: the script is
reached for the service query, never for the run — SWALLOW-1's reading was right about the
mechanism and P4a was wrong only about which verdict the frozen rule would give it),
`prebid/Prebid.js`'s two linter steps (`files=$(cat … | xargs stat) || true` then the linter over
`$files`), `getsentry/sentry-docs`'s two redirect checks under `set +e`, `dotnet/aspire`'s
flaky-test iterations (15 → 1), and `microsoft/vscode`'s two *Diagnostics before/after smoke test
run* steps — which the name rule took for checks because of the words *smoke test*, and which
are diagnostics. Seven repositories in all, of which the last two are the rule's noise. The
counted reading is what the author would preregister next time; it is reported here as what it is.

**The direction of #137's shape, answered.** SWALLOW-1 found 40 git-query-with-`|| true` steps in
24 repositories. Under fault injection the query steps are mostly NO_CHECK and ABSORBED: what
they gate is a comment, a label, a release note, an action the instrument cannot run, or nothing.
The one that gates a recognised check by shell is `gumroad`'s, and it falls open. `serviceradar`
and `airbyte` fall closed, as their authors wrote them to (P4d, P4e).

## 5. What this does not say

- **A check that is an action is invisible.** `uses:` steps are never executed and their outputs
  are unknown. `sentry-docs`' link lint is one; and 83 recognised checks are `run:` steps in a
  non-bash shell, not executed either. A fault whose only downstream check is an action is
  NO_CHECK here, and the population has more of those than this receipt can count.
- **Uninterpretable is one fault in five.** 849 BASELINE_SKIPPED are steps a script's own
  decision under `x` turned off (`should_run=false`, `valid=false`, `result=PASS` compared to a
  literal); 727 BASELINE_RED are steps that fail on their own logic in both healthy flavours.
  The model reaches 80% of the map and says which 20% it does not.
- **The check rule is stated and is a heuristic**: it mis-files diagnostics as checks (§4) and
  misses bespoke verifiers (this repository's own gauntlet step, held by G-S2-3 instead).
- **A job is not a status.** `gumroad`'s `ci-green` shows the limit: a workflow whose green is not
  its signal reads as fail-open to an instrument that reads jobs.
- **`x` is a model of a healthy run**, in two flavours; the artifacts are counted (6.96% of steps
  run) and carried past, not hidden.
- **Four runs.** The instrument as frozen in prose had three defects that the population found
  and that the pilot had not; each is named above with what it moved. The predictions did not
  move, and neither did the score.

## 6. Next

- Preregister the counted reading, and a check rule with the diagnostics excluded.
- Actions as checks: a declared list of the checking actions (lychee, actionlint, codeql, …) so a
  gated action counts as a dropped check when its gate closes.
- The status-publishing pattern (`gumroad`'s `ci/green`): read the status the job sets, not the
  job.
- `styxx ci-audit <repo>`: `faults.py --tree` already runs on any checkout; the receipt per
  repository, with the FAIL_OPEN and SWALLOWED faults named, is the command's output.
