# RESULT — SWALLOW-3: the checks that are actions — INVALID: a re-clone at the same HEADs does not reproduce SWALLOW-2 in 8 of 30,642 records, for three named reasons in the frozen instrument; the catalogue itself moves 5 hand-written faults, one of them to a dropped check

Fathom Lab · 2026-09-21 · Scores the receipt `swallow3_receipt.json.gz` against the
preregistration frozen at sha256 `1799321af3af10f27ca11267d1260940307cf21f8307efc21673fafac32836ff`.
Not amended. One run; the instrument was not revised after the freeze.

Receipt: `papers/harness/swallow3_receipt.json.gz` (38 MB of JSON, gzipped; sha256 of the JSON
`e3c7c2d1…`, recorded by the scorer) · instrument `benchmarks/harness_mutation/action_checks.py`,
sha256 `0e723694d459ca23…`, on top of `faults.py` at `d26a407ca276d1b1…` (SWALLOW-2's, unchanged)
· population `swallow1_population.json` (sha256 `5ac2789b…`), 96 repositories cloned, every one
at the HEAD the SWALLOW-2 receipt records · 948 s, 94,978 step executions · census
`swallow3_actions_census.json` · evidence for §1 in `swallow3_repro.json` (`swallow3_repro.py`) ·
scored by `swallow3_score.py`.

**INVALID.** Gate G-S3-3 — *for every fault in the SWALLOW-2 receipt, the same fault is in this
receipt with the same SWALLOW-2 reading, mismatches allowed only where a timeout is recorded* —
failed: 30,645 fault sites against 30,642; 4 sites this run has that SWALLOW-2 did not, 1 that
SWALLOW-2 had and this run does not, and 3 sites whose SWALLOW-2 reading differs, none with a
timeout on either side. Every one of the eight was read (§1). The preregistration said what
this would mean before the run: *a re-clone at the same HEADs that does not reproduce SWALLOW-2
says the instrument is not deterministic, and this RESULT would say so before anything else.* It
does. The three causes are in SWALLOW-2's frozen instrument, in its contact with the world — the
real `date`, a tie in the order stubs are created, and a background subshell racing the log that
counts what a step reached — and not in this cycle's catalogue. Their reach: 8 records of 30,642;
the population-level numbers SWALLOW-2 published move by those 8 and nothing else (hand-written:
RED 5,836 → 5,838, ABSORBED 162 → 167, NO_CHECK 286 → 284, BASELINE_SKIPPED 849 → 848; nothing
else moves).

The catalogue's own gates hold (G-S3-4 closure, G-S3-5 instrument, G-S3-6 monotone: no fault
moved outside the declared transitions, RED and SWALLOWED identical between the two readings),
and the differential reading is internal to this one run, so its numbers are reported here as
what they are: **4 of 7 predictions HIT** (P3, P4, P6, P7), on an INVALID run, not claimed as a
result. The catalogue moves **5** of 6,341 interpretable hand-written faults: four NO_CHECK →
ABSORBED, one NO_CHECK → FAIL_OPEN — `getsentry/sentry-docs`, `Get changed files`, the link lint
it gates being `lycheeverse/lychee-action` (P3, SWALLOW-2's P4b resolved). The blind spot SWALLOW-2
named is real and is that small in this population: of 105 catalogued action checks, 147 faults
have one in scope, and 135 of those are RED already.

## 0. What was done

The rule for what an action check is was written first; then every `uses:` name in the
population's hand-written workflows was counted (312 names, 7,451 steps, at SWALLOW-2's HEADs);
then the rule was applied to all 312 by hand, with the reason for each, and to well-known
checking actions the population does not use, so the shipped command reads other repositories
(82 entries, 3 families; 262 names excluded by category; 7 entries marked unverified); then the
predictions were written, with the catalogue's coverage known (104 action checks across 45
repositories, classification only) and no verdict taken; then the instrument ran once. It calls
SWALLOW-2's `_fault_verdict` unchanged and, over the same memoised executions, asks of every
catalogued action step in a fault's scope whether it is reached in the healthy world and in the
fault world; each fault carries `verdict_runs_only` (SWALLOW-2's reading) beside `verdict`.

## 1. Gates, and the failure

| gate | bar | observed | |
|---|---|---|---|
| G-S3-1 (population) | SWALLOW-1's frozen list; `faults.py` at SWALLOW-2's hash | 100 entries, `5ac2789b…`; `d26a407c…` | pass |
| G-S3-2 (same trees) | ≥ 90 cloned, every one at the recorded HEAD | 96 cloned, 96 at the recorded HEAD; the same four private or gone | pass |
| G-S3-3 (reproduction) | every SWALLOW-2 fault present with the same reading; mismatches only with a timeout | 30,641 keys in common, 30,638 with the same reading; **3 mismatches, 4 extra sites, 1 missing site, none with a timeout** | **FAIL** |
| G-S3-4 (closure) | every census name decided once; instrument in the tree is the receipt's | 312 names, 0 undecided, 0 twice; hashes equal | pass |
| G-S3-5 (instrument) | tests green; this repository unmoved; deterministic | 7 passed; 32 faults, 0 moved; identical twice | pass |
| G-S3-6 (monotone) | only NO_CHECK→ABSORBED, NO_CHECK→FAIL_OPEN, ABSORBED→FAIL_OPEN; RED and SWALLOWED counts identical | 0 outside; RED 23,033 = 23,033; SWALLOWED 167 = 167 | pass |
| G-S3-7 (no hand labels) | verdicts and checks are the instrument's | nothing reclassified | pass |
| G-S3-8 (ledger) | P1–P7 scored | below | pass |

**The eight records, read.** Each is demonstrated on its own step in `swallow3_repro.json`.

| records | where | cause |
|---|---|---|
| 2 mismatches (NO_CHECK → ABSORBED in the SWALLOW-2 reading) and 1 extra site | `browser-use/browser-use` `test.yaml` | **a tie in the stub order.** The step `Check if test file exists` assigns `TEST_FILE="tests/ci/x.py"` and, in its else-branch, pipes through `sed 's|tests/ci/||'`. The instrument creates a stub for every relative path in command position, "deepest first"; `tests/ci/x.py` and `tests/ci/` tie on depth, and the tie falls to a set's iteration order, which Python randomises per process. File first: the step finds it and reaches nothing (TOOLLESS, not a fault site; the check by name reaches no runner). Directory first: the file is not created, the step reaches `find`/`sed`/`sort` (PROPAGATES, a site; the check is live and every fault in the job is ABSORBED, not NO_CHECK). Eight hash seeds: 2 one way, 6 the other. |
| 1 mismatch (BASELINE_SKIPPED → RED) | `airbytehq/airbyte` `ai-ready-command.yml` | **the real `date`.** `Check for weekend freeze` runs `TZ=America/Los_Angeles date +%u`; `date` is real in the healthy worlds by design. SWALLOW-2's receipt was made on a Sunday evening Pacific (committed 04:41 UTC): the freeze was on and `Apply ready label` was skipped in both healthy flavours. This run was made on the Monday: the same step runs, and its fault is RED. |
| 3 extra sites | `langfuse/langfuse` `pipeline.yml`, three jobs | **a background subshell racing the log.** `( set -e; curl …; tar …; sudo mv …; touch … ) > /tmp/migrate-install.log 2>&1 &` — the instrument reads the log of reached commands when the script exits; whether the backgrounded subshell has written `curl` to it by then depends on scheduling. Under the run's load it had (SWALLOWS: a site); on an idle machine, twelve trials, it never does (TOOLLESS). |
| 1 missing site | `githubnext/gh-aw` `visual-regression-checker.lock.yml` | **the same race**, the other way: `nohup npm run dev … &` — the not-found handler that logs `nohup` runs in the background; SWALLOW-2 caught it (SWALLOWS), this run did not (TOOLLESS). A second mechanism is visible on the same step and is not the cause of this record: it redirects into `/tmp/gh-aw/agent/`, an absolute path the sandbox does not cover, which a real `mkdir -p` in an earlier run had created on the machine; with the directory the step is SWALLOWS, without it PROPAGATES — a site either way. |

None of these is a timeout, so the frozen gate does not excuse them, and this RESULT does not.
None is in the catalogue or the wrapper: with the catalogue removed, the same eight records
differ, and the other 30,638 do not. What the reproduction gate found is that SWALLOW-2's
"deterministic: identical twice" (its G-S2-4, on this repository's 32 faults) did not reach three
ways the population can make the instrument disagree with itself; the three are named, are each a
few lines of the executor, and are the first item of §6.

## 2. Predictions, scored (on an INVALID run)

Hand-written workflows, interpretable faults. "Moved" is `verdict != verdict_runs_only`.

| | prediction | observed | |
|---|---|---|---|
| P1 | ≥ 10% of hand-written NO_CHECK faults move | **5 of 284 (1.8%)**: 4 to ABSORBED, 1 to FAIL_OPEN | **MISS** |
| P2 | ≥ 5 hand-written FAIL_OPEN faults in ≥ 3 repositories | **4 in 3** (`gumroad` ×2, `crewAI`, `sentry-docs`; from 3 in 2) | **MISS** |
| P3 | `sentry-docs` `Get changed files`: FAIL_OPEN, lychee among the dropped | **FAIL_OPEN**; drops `Check external links` (`lycheeverse/lychee-action`), `step-if` | **HIT** |
| P4 | `primer/react` `Get source files changes`: not FAIL_OPEN | **NO_CHECK**, 0 action checks in scope: what it gates is a comment | **HIT** |
| P5 | `step-if` the strict plurality among dropped action checks | `job-if` 2 (`gumroad`'s two retried test steps, dropped with the run checks beside them), `step-if` 1 | **MISS** |
| P6 | < 100 hand-written faults move | **5** (0.08%) | **HIT** |
| P7 | ≤ 1 status-check among the dropped action checks | **0**: the 29 CodeQL / Sonar / zizmor steps sit in jobs behind no shell gate | **HIT** |

## 3. The map, and what the catalogue moved

**The action checks.** 105 in the population's workflows (37 lint, 29 status, 15 security, 12
test, 12 carried commands) in 45 repositories; 100 are reached in the healthy world; 7 rest on an
unverified catalogue entry and none of those is dropped anywhere. Beside them the instrument sees
3,191 `run:` checks. Among all 72,216 steps: 33,659 are `run:`; 14,627 are `actions/github-script`
(the generated stratum's agent workflows, almost entirely) and 3,121 are local actions — with 8
docker images, 17,756 steps, 25%, whose content the instrument cannot read; 58 are catalogued actions whose inputs turn
the check off (a reviewdog without `fail_level`, a lychee with `fail: false`); 349 jobs call
reusable workflows and are not descended into.

**Where they sit.** 147 faults have an action check in scope, in 19 repositories: 135 are RED
already — the action check lives in a job whose `run:` steps are loud, an install before a
CodeQL analysis, a build before a `pre-commit/action` — 6 were ABSORBED, 5 NO_CHECK, 1
FAIL_OPEN. That is why P1 missed by a factor of five: the checks SWALLOW-2 could not see are mostly
in the jobs it could already read as loud. The 5 that moved:

- `getsentry/sentry-docs` `lint-external-links.yml` → `check-pr` → `Get changed files`
  (`git diff … || true`): **NO_CHECK → FAIL_OPEN.** The link lint, `lycheeverse/lychee-action`
  with `fail: true`, is gated `if: steps.changed.outputs.files != ''`; when the diff fails the
  gate closes and no link is checked, green. SWALLOW-2's P4b, right about the shape, wrong only
  because it could not see the check. The one new dropped check in the population.
- `airbytehq/airbyte` `codeql.yml` → `Build (compile only, continue past failures)`: NO_CHECK →
  ABSORBED; the CodeQL `analyze` after it is a status-check and is still reached.
- `airbytehq/airbyte` `format-fix-command.yml` → `Check for changes` (`git diff --quiet && … ||
  …`): NO_CHECK → ABSORBED; the `pre-commit/action` earlier in the job is unaffected.
- `tokens-studio/figma-plugin` `node.js.yml` → `coverage` → `Install dependencies`
  (`continue-on-error`): NO_CHECK → ABSORBED; `anuraag016/Jest-Coverage-Diff` runs the tests
  after it either way.
- `browser-use/browser-use` `test.yaml` → `evaluate-tasks` → `Get week number for cache key`:
  NO_CHECK → ABSORBED; the evaluation is a `nick-fields/retry` whose command is `python
  tests/ci/evaluate_tasks.py`, a check by SWALLOW-2's rule on the carried command.

**The dropped checks, all four**, under the preregistered reading: `gumroad`'s `run_scope`
(#137's shape at job level; it now also drops the two retried `Wait for services and prepare test
database` steps that SWALLOW-2 could not see, beside the two `Run tests` it could), `gumroad`'s
`ci-green` report (fail-closed by design, as SWALLOW-2 read it), `crewAI`'s durations cache, and
`sentry-docs`' link lint. The counted reading, reported beside as before: 12 in 7 repositories,
the same 8 `fewer` cases as SWALLOW-2 plus the four above.

**Everything else stands.** Hand-written: RED 5,838 · ABSORBED 171 · NO_CHECK 279 · SWALLOWED 49
(23 repositories) · FAIL_OPEN 4 · BASELINE_SKIPPED 848 · BASELINE_RED 727. The generated
stratum: none of its 22,729 faults moved.

## 4. What this does not say

- **It does not say SWALLOW-2 is wrong.** Eight records of 30,642 differ, each for a stated
  reason; the numbers SWALLOW-2 published are within 5 of those here on every count. It says
  SWALLOW-2's instrument is not bit-reproducible on this population, and why, and that its own
  determinism gate was too small to see it.
- **The catalogue's effect is measured on one run of a non-deterministic instrument.** The five
  moves and the one new FAIL_OPEN are not among the eight unstable records, and G-S3-6 holds; but
  the gate that would have let this cycle claim them failed, and the claim waits for §6.
- **An action check is never executed.** It can be dropped here; it cannot be seen to fail. A
  check whose failure `continue-on-error` hides — SWALLOW-2's commonest shape — is invisible when
  the check is an action.
- **A quarter of the steps cannot be read**: `github-script`, local actions, docker images; and 349
  jobs that call reusable workflows are not descended into.
  The population's generated agent workflows are almost entirely `github-script`.
- **The catalogue is a reading of documentation** (seven entries not even that), and the rule for
  "a verdict on the code" draws lines a reader may draw elsewhere — CodeQL in, `ossf/scorecard`
  out, a commit-message linter out. Every line is in `action_checks.py` with its reason; the
  census has every name; the receipt would be recomputable under another line.

## 5. What ships

`styxx ci-audit` reads with the catalogue by default (`styxx/ciaudit/actions.py`, held equal to
the frozen `action_checks.py` entry for entry by `tests/test_ciaudit.py`, which also holds the
engine to both instruments' verdicts, fault for fault, with the catalogue on and off);
`--no-actions` is SWALLOW-2's reading. The card names a dropped action check with the action and
its kind. This is shipped under an INVALID receipt and says so in its docstring: the catalogue's
logic is gated (G-S3-6) and pinned; what is not established is the bit-reproducibility of the
engine underneath it, which is the same with or without the catalogue.

## 6. Next

1. **SWALLOW-2.1 — the same run twice.** Fix the three causes in the executor and preregister
   that two runs on the same day, from the same clones, give the same 30,6xx records: stubs
   created in a total order (depth, then name); the reached-log read after the script's
   background children are reaped or the step's timeout passes, whichever first; `date` in the
   healthy worlds answering a stated instant. Absolute `/tmp` paths rewritten into the sandbox
   is the fourth item, for the mechanism seen and not yet charged. A new `faults.py`, a new
   receipt, a new pin; SWALLOW-2's receipt stays as it is, with this RESULT beside it.
2. Then SWALLOW-3 again, on the reproducible instrument: the catalogue is ready and the
   predictions here are the priors — the blind spot is small, P1 at 2% and P2 at 4 in 3.
3. The counted reading and a check rule without the diagnostics, preregistered (SWALLOW-2 §6),
   on the same instrument.
