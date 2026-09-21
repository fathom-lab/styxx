# RESULT — SWALLOW-4: the repair is loud — 39 of 53 hidden and dropped checks have a verified fix (median: one line); every `continue-on-error` is one line from loud; the shell hides checks by control flow, which a strict shell does not reach

Fathom Lab · 2026-09-21 · Scores the receipt `swallow4_receipt.json.gz` against the
preregistration frozen at sha256 `0069be79eada6501354f8bde314bda80553f30f2a539f6e6677211eb25f93610`.
Not amended. One run; the instrument was not revised after the freeze.

Receipt: `papers/harness/swallow4_receipt.json.gz` (sha256 of the JSON `7353781a…`, recorded by the
scorer; every candidate's unified diff is in it) · instrument `benchmarks/harness_mutation/repair.py`,
sha256 `7b9a1695d316c2ce…`, on top of `action_checks.py` (`0e723694…`) on top of `faults.py`
(`d26a407c…`), both unchanged · targets: every SWALLOWED and FAIL_OPEN fault of the SWALLOW-3
receipt (`609e6645…`), on the same trees at the same HEADs · 172 targets in 27 repositories,
616 s · scored by `swallow4_score.py`.

**VALID. 8 of 10 predictions HIT** (P3a–d, P4, P5, P6, P7). Of the **53 hand-written** hidden
and dropped checks, **39 have a verified repair**: the same fault is RED on the repaired workflow
in every flavour that could read it, and a healthy run of the repaired workflow is
indistinguishable from the original's in both flavours. **32 of them are one line** — a
`continue-on-error: true` removed — and every one of the 33 checks hidden that way has one. **3
of the 4 dropped checks** have one (`gumroad`'s `run_scope`, `sentry-docs`' link lint, `crewAI`'s
durations cache); the fourth is fail-closed by design and neither repair reaches it, as
predicted. The **misses are the shell**: of the 16 checks the shell hides, only **3** are repaired
by making the shell strict (P1: 36 of 49 SWALLOWED, against 40 predicted), and only **1** is
rejected for changing a healthy run (P2: 1 against 3) — the other 12 are not hidden by a
`|| true` the strict shell can remove but by control flow: a check whose failing command is the
condition of an `if`, a verifier that compares two answers and echoes, a loop that counts. Of the
**119 generated** targets, **115** are verified by the one-line repair, every one the same line.

## 0. What was done

For each target, the original workflow was re-analysed on this machine (the baseline: all 172
equal the receipt's verdict — none of SWALLOW-3's three unstable shapes sits on a target), and
three candidates were tried in order: remove the fault step's (or its job's)
`continue-on-error`; make the fault step's shell strict (trailing `|| true` and `|| :` removed,
`set +e` removed, `set -eo pipefail` ensured); both. Each candidate is an edit to the workflow
*text*, located through the parser's marks, re-parsed, and re-analysed for the same (job, index)
with the same instrument. Two halves are recorded per candidate: LOUD (RED in every previously
interpretable flavour) and UNCHANGED (every job's result, every step's run / not run and why,
reached-a-tool or not, artifact or not — equal to the original's in both flavours). The first
candidate with both halves is the fault's repair.

## 1. Gates

| gate | bar | observed | |
|---|---|---|---|
| G-S4-1 (instrument) | tests green; fixture deterministic; this repository has no target | 5 passed; identical twice; 0 targets | pass |
| G-S4-2 (baseline) | every target's verdict here equals the receipt's, or is named and excluded (≤ 5%) | 172 of 172 equal; 0 excluded; 0 missing | pass |
| G-S4-3 (both halves) | verified ⇒ loud in every interpretable flavour and unchanged in both | 0 verified candidates failing a half | pass |
| G-S4-4 (no hand labels) | the three stated repairs only | 0 candidates outside; receipt lists the three | pass |
| G-S4-5 (frozen underneath) | `faults.py`, `action_checks.py`, the source receipt | `d26a407c…`, `0e723694…`, `609e6645…` | pass |
| G-S4-6 (ledger) | P1–P7 scored, P3 as four | below | pass |

## 2. Predictions, scored

| | prediction | observed | |
|---|---|---|---|
| P1 | ≥ 40 of the 49 hand-written SWALLOWED verified | **36**: 33 of 33 under `continue-on-error`, 3 of 16 by the shell | **MISS** |
| P2 | ≥ 3 shell-hidden faults rejected by the twin condition | **1** (`dotnet/aspire`, the flaky-test iterations: `set +e  # Disable errexit so pipefail doesn't abort before PIPESTATUS capture` — the author's own comment names what the line protects; the strict shell removed it and the healthy `x` run broke at that step) | **MISS** |
| P3a | `gumroad` `tests.yml` › `run_scope` › 0: verified | **strict-shell**, 4 lines: the `\|\| true` after the labels query removed; RED in both flavours; healthy unchanged | **HIT** |
| P3b | `sentry-docs` `lint-external-links.yml` › `check-pr` › 1: verified | **strict-shell**, 4 lines: the `\|\| true` after the diff removed | **HIT** |
| P3c | `crewAI` `update-test-durations.yml` › `update-durations` › 4: verified | **no-continue-on-error**, 1 line; strict-shell alone is not loud (FAIL_OPEN stays) | **HIT** |
| P3d | `gumroad` `ci-green.yml` › `report` › 1: not verified | no candidate is loud: the script's `unknown` path exits 0 on purpose, `set -uo pipefail` is already there | **HIT** |
| P4 | ≥ 4 of the 8 "fewer" faults verified | **6**: `vscode`'s two diagnostics (one line each), `prebid`'s two linters (strict-shell), `sentry-docs`' two redirect checks (both); not `mlflow`'s loop (not loud), not `aspire`'s iterations (rejected) | **HIT** |
| P5 | median verified hand-written repair ≤ 3 lines, max ≤ 10 | **median 1, max 6** (32 one-liners; 4, 4, 4, 5, 5, 6, 6) | **HIT** |
| P6 | ≥ 90% of generated targets verified, ≥ 90% by the one-line repair | **115 of 119**, all 115 by `no-continue-on-error` | **HIT** |
| P7 | no empty verified diff; `both` verified whenever a smaller repair is | 0 and 0 | **HIT** |

## 3. The map of repairs

| hand-written targets | n | verified | how | not |
|---|---|---|---|---|
| hidden under `continue-on-error` | 33 | **33** | 31 one line; 2 `both` (`sentry-docs`, where the step also had `set +e`) | — |
| hidden by the shell | 16 | **3** | `giselle`'s `pnpm lint:colors:code \|\| true`; `prebid`'s two `files=$(…) \|\| true` before the linter | 12 not loud; 1 rejected |
| dropped (FAIL_OPEN) | 4 | **3** | 2 strict-shell, 1 one line | `ci-green`, fail-closed by design |
| **all** | **53** | **39** | 32 one line · 5 strict-shell · 2 both | 13 not loud · 1 rejected |

**The one-line repair never fails.** 33 of 33 hand-written and 115 of 116 generated checks hidden
by `continue-on-error` are loud the moment the line is gone (the 116th shares its line with
something else and the text edit declines), and nothing about a healthy run
changes — by construction of the model, in which a healthy step passes with or without it. What
the line is doing in every one of those files is exactly one thing: keeping a failing check from
failing the job. `sentry-docs` says so in its own comment on the line — `# Fail the check but
don't block merge` — and that is the honest reading of the whole column: **the repair measures
that the choice is one line from loud, not that the choice is wrong.** A non-blocking lint is a
policy; the receipt shows the policy is one deleted line, in 148 places.

**The shell hides by structure, not by `|| true`.** The strict shell repaired the three cases that
are the `|| true` idiom and nothing else. The twelve it does not reach:

- **a check whose failing command is an `if` condition** — `hmis`'s `if grep -q
  "grantAllPrivilegesToAllUsersForTesting = true" …; then exit 1; fi` (the grep that fails is the
  green path), `roslyn`'s `if echo "$COMMENT_BODY" | grep -q "/dart"; then`, `serviceradar`'s
  APK-pin verifier under `set -euo pipefail` already: `-e` does not apply inside a condition;
- **a verifier that compares and echoes** — `selfxyz`'s two `Verify branch and commit` steps,
  `gh-aw`'s `Verify no compilation errors`: their tools' answers are compared to expectations and
  reported, and a failed tool answers with nothing to compare;
- **a loop that counts** — `mlflow`'s `set +e` retry loop over the test runner (removing `set +e`
  is not enough: the loop reads `$?`), `aspire`'s three `Check if any test requires …` over a
  runsheet, `nodetool`'s mutation testing, `langfuse`'s `\|\| echo` (which the repair leaves
  alone by its stated rule).

These are the shapes a strict shell cannot make loud because the script's own logic decides the
exit status. Their repair is a different edit — `exit 1` in the else-branch, a `test -n` on the
answer — and it is not one the instrument tries, so it is not one this receipt claims.

**The twin condition earned its place once**, and the once is the right one: `aspire`'s
`reproduce-flaky-tests.yml` disables `errexit` with a comment explaining why (`PIPESTATUS`
capture after a pipeline that is allowed to fail); the strict shell put it back, and the healthy
`x` world broke at that step. A repair that had been shipped on loudness alone would have broken
a flaky-test reproducer to make it loud.

**The generated stratum** is one line, 115 times: `gh-aw`'s compiled agent workflows mark their
agent step `continue-on-error: true`. The 4 not verified: one whose `continue-on-error` shares its
line with something else (the text edit declines), two `Verify connectivity` steps already under
`set -euo pipefail` whose swallowing is structural, one not loud. The fix belongs in the compiler.

## 4. What this does not say

- **Loud is not wanted.** A non-blocking lint is a choice; 33 hand-written files make it. The
  receipt says the choice is one line; the authors say why they made it, in at least one comment.
- **Verified against the model**, with the same artifacts carried past: a healthy run is `x` in
  two flavours, actions are never executed, non-bash steps are not run. A repair verified here is
  verified for what the instrument can see.
- **Two repairs.** The strict shell reaches the `|| true` idiom and nothing else; 12 of the 16
  shell-hidden checks need an edit to the script's logic, which this cycle did not attempt.
- **A cosmetic mark on the diffs**: the rewrite of a block scalar drops a trailing blank line
  inside it, which shows as one removed empty line in the strict-shell diffs (P5's sizes include
  it).
- **On an instrument known not to be bit-reproducible** (SWALLOW-3 §1). None of its three
  unstable shapes sits on a target here (G-S4-2: 172 of 172 baselines equal), and the repairs
  were verified once. SWALLOW-2.1 stands where it stood.

## 5. What ships

`styxx ci-audit --repair`: for every finding, the two repairs tried and verified on both halves,
the diff printed for a verified one, and for the others which half failed and what the original
line was protecting. `styxx/ciaudit/repair.py` is the living copy, held to the frozen instrument
on a fixture by `tests/test_ciaudit.py`, which pins all three instruments now.

## 6. Next

- The structural repairs: `exit 1` in the else-branch of a guard, `test -n` on a compared
  answer — preregistered against the 12, with the same twin condition.
- SWALLOW-2.1, the same run twice (RESULT_swallow3 §6): the executor's three causes.
- The counted reading and a check rule without the diagnostics (`vscode`'s two are "checks" here
  by the rule's word *smoke test*, and their one-line repair makes diagnostics blocking).
