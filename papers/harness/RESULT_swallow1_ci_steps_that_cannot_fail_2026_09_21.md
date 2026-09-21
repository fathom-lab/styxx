# RESULT — SWALLOW-1: in 67 of 87 repositories a CI step cannot fail, and in 35 of them that step is a check

Fathom Lab · 2026-09-21 · Scores the census recorded in `swallow1_receipt.json` against the
preregistration frozen at sha256 `7342f633a8d738c596e83714cec90c160929944d04d3f1b0132f3c69c2d8d3a3`.
Not amended.

Receipt: `papers/harness/swallow1_receipt.json.gz` — 19 MB of JSON, every step's text included,
carried gzipped; sha256 of the JSON `25b6ee628fed98c4…`, checked and recorded by the scorer ·
population `swallow1_population.json` (sha256 `5ac2789b6d67eed8…`, 100 repositories) · instrument
`benchmarks/harness_mutation/census.py`, sha256 `98d394c2cbf32295…` · 475 s · self-census
`swallow1_self_census.json` reproduces MUTE-2's verdicts on this repository (33 propagate,
5 toolless, 0 swallow). Scored by `swallow1_score.py` → `swallow1_scored.json`.

**VALID. 6 of 7 predictions HIT; P5 MISS.** 100 repositories asked; 96 cloned (4 are gone or
private); 90 have workflows; 87 have at least one bash step that ran; **33,651 `run:` steps
executed**, 8,912 of them hand-written and 24,739 generated agentic workflows.

## 0. What was done

MUTE-2's behavioural guard — execute every `run:` step with an empty PATH, so every external
command fails, and demand that the step go red — needs nothing but the workflow files. The census
took a blob-less sparse clone of `.github/workflows` from each of the 100 repositories that
receive the most agent-authored pull requests in the AIDev corpus, executed every bash step the
way the guard does, and classified it: PROPAGATES, SWALLOWS (green with a failed tool reached),
TOOLLESS (green with nothing reached), SYNTAX, TIMEOUT, NOT_BASH. Beside the verdict:
`continue-on-error`, a category from the step's name and first command, whether the workflow is a
generated `*.lock.yml`, and whether the step is #137's shape — a git query with `|| true`.

"Cannot fail" = SWALLOWS, or `continue-on-error: true` on an executed step. "Verification" =
category test, lint or typecheck.

## 1. Gates

| gate | bar | observed | |
|---|---|---|---|
| G-S1-1 (population) | the frozen list, sha256 in the receipt | 100 entries, `5ac2789b…` | pass |
| G-S1-2 (reach) | ≥ 90 cloned, failures named | 96 cloned; `Smart-Cleaner-for-Android`, `antiwork/flexile`, `antiwork/helper`, `Metta-AI/metta` could not be read (gone or private) | pass |
| G-S1-3 (method) | the census reproduces MUTE-2 on this repository | 33 propagate, 5 toolless, 0 swallow | pass |
| G-S1-4 (no hand labels) | verdicts and categories are the instrument's | nothing reclassified; §4 is a reading and says so | pass |
| G-S1-5 (ledger) | P1–P7 scored | below | pass |

## 2. Predictions, scored

| | prediction | observed | |
|---|---|---|---|
| P1 | ≥ 50% of repositories with an executed step have a step that cannot fail | **67 of 87 (77%)** | **HIT** |
| P2 | ≥ 10% have a *verification* step that cannot fail | **35 of 87 (40%)** | **HIT** |
| P3 | median per-repository shell swallow rate < 5% (hand-written) | median **2.2%**, p75 6.1%, max 100% | **HIT** |
| P4 | ≥ 3 repositories with a git query whose failure becomes an empty answer | **24 repositories, 40 steps** | **HIT** |
| P5 | ≥ 10 repositories carry `*.lock.yml`, and ≥ 30% of those steps are `continue-on-error` | 14 repositories; **17%** of 24,739 steps | **MISS** |
| P6 | TOOLLESS ≤ 25% of bash steps; SYNTAX + TIMEOUT ≤ 2% | 7.4%; **0** | **HIT** |
| P7 | < 60% of hand-written executed steps containing `\|\| true` actually swallow | 135 of 527 (**26%**) | **HIT** |

P5 missed on its second clause. The pilot's one generated repository had `continue-on-error` on
most of its agentic steps; across fourteen it is one step in six. The first clause held with room
(14 ≥ 10). The miss is recorded as what it is: the author extrapolated a rate from one repository.

## 3. The numbers, hand-written and generated apart

| | hand-written | generated (`*.lock.yml`) |
|---|---|---|
| `run:` steps | 8,912 | 24,739 (19,559 of them in one repository, `githubnext/gh-aw`) |
| PROPAGATES | 7,560 | 22,935 |
| SWALLOWS | **338 (4.3% of executed)** | 72 (0.3%) |
| TOOLLESS | 741 | 1,712 |
| NOT_BASH | 273 (pwsh 277, powershell 8, cmd 4, R 2, ruby 1, across both) | 20 |
| `continue-on-error` on an executed step | 215 | 4,196 |

Among hand-written executed steps, by category (swallow rate / `continue-on-error` rate):
install 1.6% / 1.3% · build 0.7% / 2.9% · **test 2.3% / 2.2%** · **lint 2.6% / 1.7%** ·
**typecheck 2.6% / 2.6%** · publish 4.0% / 3.3% · other 7.5% / 3.4%. Verification steps swallow
*less* than the average step, not more — and still one test step in forty cannot fail at the
shell, and one more in forty is `continue-on-error`.

Fifty-three repositories have at least one hand-written swallow; three carry the bulk
(`manaflow-ai/cmux` 47, `githubnext/gh-aw` 26, `langfuse/langfuse` 18) and the median repository
has one or two.

## 4. What the swallowing checks are (a reading, marked as one)

The instrument found 38 verification steps that swallow at the shell (36 hand-written) and 42
with `continue-on-error`. The category is a heuristic and the receipt keeps every step's text, so
here is what the 38 are, read one by one. Nothing below changes a verdict.

- **Best-effort by nature, named as such** — cleanups and teardowns (`Clean up` ×3 in
  `antiwork/gumroad`, `Tear down remote test env` in `openai/codex`, `Cleanup coverage build
  artifacts`), version lookups (`Get Playwright version`, `Detect installed Playwright version`),
  seeds marked *(best-effort)* (`Significant-Gravitas/AutoGPT`), and steps whose name says
  *non-blocking* (`Lint Colors in Code (non-blocking)`, `run SQL-equivalence tests (non-blocking)`).
  About half. The category caught the word *test* in a docker-compose file name or a script path;
  the swallow is right.
- **Auto-fix, not a check** — `npm run lint:fix || true` (`primer/react`, `nodetool-ai/nodetool`).
  A fixer that fails should not fail the job; the check is elsewhere, or is not.
- **A method artifact worth stating** — `manaflow-ai/cmux`, eight copies of
  `test "$(git rev-parse HEAD)" = "$EXACT_COMMIT"`. With `git` failing, the substitution is empty
  and the comparison is between two empty strings, so the guard passes. On a runner with git it
  works; the census is right that the step cannot propagate a git failure, and wrong to file it
  under *test* — the word is the shell builtin.
- **The #137 direction: a query fails, the answer is "nothing", the check runs on nothing.**
  `getsentry/sentry-docs` `Get changed files` — `FILES=$(git diff … || true); if [ -z "$FILES" ]
  … "No markdown files changed"`, and the external-link lint runs on no files.
  `primer/react` `Get source files changes` — the same shape, and `diff=` means no integration
  tests are recommended. `getsentry/sentry` `Get base branch commit` — `|| true`, then
  `skip=true`; explicit, at least. `microsoft/testfx` (generated) `Extract changed test file
  regions` ×2. `mlflow/mlflow` `Check diff` — `gh pr view … | grep` inside `$( )`, every
  category false when `gh` fails, so nothing is formatted. `getsentry/sentry` mypy — output
  filtered through `grep` so that a `grep` with no matches is the success case. And the one that
  is a test step by any reading, `mlflow/mlflow` `master.yml → database → Run tests`: `set +e;
  trap 'err=1' ERR; for service in $(./tests/db/compose.sh config --services | grep '^mlflow-' |
  sort); do … pytest …; done; test $err = 0`. If the enumeration of services fails, the loop
  body never runs, the trap never fires, `err` is 0, and *Run tests* is green having run no
  tests. These are the shape MUTE-1 opened on: a check that reports green because the thing it
  depends on failed.
- **The safe direction, in the same population** — `carverauto/serviceradar` `Decide whether
  this Mix project needs lint`: *"Cannot list changed files; running Mix lint to stay
  fail-closed."* `airbytehq/airbyte` `Check for changes`: a failed `git diff --quiet` lands in the
  *changes detected* branch. Same query, same `|| true` family, opposite default. The census
  cannot tell these apart — both exit 0 with a failed tool reached — and that is the sharpest
  limit of the method: **it sees that a step cannot fail; it cannot see which way it falls.**

The 40 git-query steps in 24 repositories (P4) split the same way on reading: some fetch before
they diff (`prebid/Prebid.js` fetches both refs explicitly two steps earlier and its `|| true` is
on a `grep`), some fall closed, some fall open. The receipt names all 40.

## 5. What this does not say

- **A step that cannot fail is not a defect.** Roughly half of the verification swallows are
  right to swallow, and the reading above says which. The finding is about the *shape*, which is
  common enough that a reader of any CI configuration should expect it, and about the minority
  where the shape sits on a check.
- **The direction is invisible to the instrument.** Fail-closed and fail-open look identical to
  it. A future census that wants direction has to read the branch the empty answer takes, which is
  a parse of the step, not an execution of it.
- **The category is a heuristic** and mis-files cleanups, lookups and shell builtins under *test*.
  The per-category rates in §3 carry that noise; the repository-level counts (P1, P2) are robust to
  it in one direction only — a mis-filed cleanup inflates P2. Read P2 as "35 repositories have a
  step the heuristic calls a check and the shell cannot fail", and the reading in §4 as what that
  turns out to mean.
- **`x` for every `${{ }}`** reaches 92.6% of bash steps and no more. A step whose only branch
  under `x` is a print is TOOLLESS here and may reach tools on a runner.
- **Windows steps were not executed** (293, pwsh almost entirely). The method is bash.
- **The population is agent-heavy by construction.** Whether the rates hold for repositories that
  receive no agent PRs is not measured.

## 6. Next

- Direction: parse the branch an empty answer takes (`-z` → skip vs `-z` → run everything) and
  report fail-open and fail-closed separately. The 40 steps in P4 are the fixture.
- The census as a `styxx` command: `styxx ci-audit <repo>` — the guard, the census and a
  reading of the swallowing steps, for one repository, in a receipt. The instrument already runs
  on any checkout (`--tree`).
- A second population that is not agent-heavy, to see whether 77% / 40% is the ecosystem or the
  corner of it agents are sent to.
