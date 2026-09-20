# RESULT — MUTE-1: 101 of 120 cuts to this repository's checking apparatus are invisible to its test suite

Fathom Lab · 2026-09-20 · Scores the run recorded in `mute1_receipt.json` against the
preregistration frozen at sha256 `f569701e1a2fc6b71bbff8336d51d413f6e1291d94d3224a1ec525c392df9fbc`
(plus Amendment A, appended before any mutant was applied; the document then hashed to
`b7e2f75678783d5b8d4657a4fa9c54c698bbf3d86ff865df69764622f503e66c`).

Receipt: `papers/harness/mute1_receipt.json` · tree `50216c21dada61ff3ad5232ba96d83030f95b595` · harness fingerprint
`8da2770cfabb9fc631d6a71816e23ffe36ed75ae42f7f1826aa2b319d79c4ef1` · instrument sha256 `1913e76f87ae726fe024394340915b5b3ac31072ed9ab2b4f8cf3440ad8e99be` · 1266 s.
Scored by `papers/harness/mute1_score.py` → `mute1_scored.json`. Nothing below is hand-labelled;
the two bins in §4 are a judgement and are marked as one.

**About the tree.** Commit `50216c21` is a local merge, made on 2026-09-20, of `main` at
`98a5c368` plus every pull request open that day (#131–#140, in the order posted on #126) plus this
instrument and its preregistration. It is not on GitHub and cannot be, because it was assembled
from branches that were still being merged. The reproducible reference is the **harness
fingerprint** in the receipt: a sha256 over every file a mutant can touch plus every file the
oracle reads. Any commit whose fingerprint is `8da2770cfabb9fc6…` has this harness, and this run
applies to it — which is what `main` will be once the open pull requests land, provided nothing
else touches those files first. `python -m benchmarks.harness_mutation.mute --inventory` on such a
commit reports the same 74 checks and 120 mutants.

## 1. Gates

| gate | bar | observed | |
|---|---|---|---|
| G-M1-1 control | ≥ 1 baseline pass, ≥ 1 KILLED | 495 tests pass on the unmutated tree; 18 KILLED | PASS |
| G-M1-2 instrument | UNREACHED ≤ 5 | 1, the one the preregistration named (`telescope/prompts.json` is not in the tree) | PASS |
| G-M1-3 no hand labels | every KILLED names a test, every SURVIVED names none | holds for all 120 | PASS |
| G-M1-4 ledger | P1–P8 scored | below, 6 HIT / 2 MISS | PASS |

**The run is VALID.** Four oracle tests fail on the unmutated tree in this environment and were
excluded before any mutant was applied, as the rule requires — they cannot kill anything and did
not: `test_diffgate_bc1` (two), `test_sworn_action` (the committed sample), and
`test_version_never_behind_tag`. They are named in the receipt.

## 2. Totals

| operator | KILLED | SURVIVED | UNREACHED |
|---|---|---|---|
| M-TRIGGER | 1 | 8 | 0 |
| M-JOB | 3 | 11 | 0 |
| M-STEP | 4 | 34 | 0 |
| M-SWALLOW | 1 | 37 | 0 |
| M-GUARD | 0 | 8 | 0 |
| M-SCRIPT | 1 | 3 | 0 |
| M-SUBJECT | 8 | 0 | 1 |
| **all** | **18** | **101** | **1** |

**84% of the cuts survived.** Every subject that exists is guarded (8 of 8); almost nothing else
is. Of the eighteen kills, sixteen come from the four guard files written this week (#127, #137,
#138, #140). Before those, the same run would have reported two — the instrument pin and the gate
README, both from `tests/test_port_is_current.py`.

## 3. The prediction ledger

| | predicted | observed | |
|---|---|---|---|
| P1 the four repairs are real | 5 named mutants KILLED | all 5 KILLED, each by the file that repaired it | **HIT** |
| P2 most jobs have no guard | exactly 2 of 14 M-JOB KILLED | **3** KILLED: the two named, plus `leaderboard-submission → run-submission` | **MISS** |
| P3 the headline | M-STEP and M-SWALLOW on `test.yml → test → Run tests` SURVIVE | both SURVIVED | **HIT** |
| P4 triggers are unguarded | 8 of 9 SURVIVE, `leaderboard-submission` KILLED | exactly that | **HIT** |
| P5 guards are unguarded | all 8 M-GUARD SURVIVE | all 8 SURVIVED | **HIT** |
| P6 one script of four | `typecheck` KILLED, the other three SURVIVE | exactly that | **HIT** |
| P7 a textual guard is fooled | M-SWALLOW on both `typecheck-js` steps SURVIVE | both SURVIVED | **HIT** |
| P8 the totals | 17 / 102 / 1 | 18 / 101 / 1 | **MISS** by one, same direction |

Both misses are the same miss. The guard written in #137 looks up the leaderboard workflow's
*step* by name, and I predicted the kill for the step cut and forgot that deleting the whole job
deletes the step. One kill more than predicted, one survivor fewer. The direction of P8 — more
than four in five cuts invisible — holds at 84%.

**P1 deserves a sentence.** MUTE-016, the swallow of the gauntlet discovery step, was killed by
exactly one test: `test_a_comparison_that_cannot_be_made_is_not_reported_as_nothing_to_verify`,
the one that *runs the step's shell* and demands a non-zero exit. The textual guard beside it,
which looks for `|| true` on a git line, did not fire — the wrapper puts `|| true` on a line of its
own. P7 is the same lesson from the other side: the #140 guard checks that `npm ci` and
`npm run typecheck` are *present*, and a swallowed step is present. A guard that reads the text
of a check is fooled by anything that leaves the text in place. A guard that runs the check is not.

**One kill is a crash, not a guard.** MUTE-119 removes the `PREREG_SHA256_FROZEN` line from
`calib1_score.py`, and the receipt shows it killed by 495 tests — every test in the oracle. That is
a collection error: the module stops importing and pytest reports the whole session failed. The
rule counts it, correctly, but a reader should not mistake it for a targeted guard; the targeted
guard for that pin is `test_the_preregistration_on_disk_is_the_one_the_scorer_pins`, which would
have fired alone had the line been *changed* rather than removed. A future operator that alters a
pin rather than deleting it would separate the two, and is worth adding.

## 4. The survivors, read — and this is a judgement, made after the run

The preregistration allows the survivors to be sorted into two bins after reading and requires
this sentence: **the sorting below is a judgement, not a measurement.** Every row is in the
receipt with its verdict; only the bin is mine.

**Loud elsewhere** — cutting it would be noticed, just not by a test:
- `publish.yml`, all 12 cuts. The next release fails in front of whoever is releasing.
- `leaderboard-submission.yml`, the 7 surviving step cuts. A submitter whose PR gets no comment
  would ask.

**Silent everywhere** — cutting it would be noticed by nobody:
- **`test.yml → test`, all 8 step cuts and 10 swallows, including `Run tests`.** If CI stopped
  running the Python suite tomorrow — deleted the step, or wrapped it in `|| true` — every pull
  request would go on showing green, and nothing in this repository would say otherwise. The
  `package` and `core-minimal` jobs likewise: the wheel could stop being checked and the numpy-only
  install could stop being exercised. This is P3, and it is the finding.
- **`telescope.yml`, all 13 cuts.** The guard in #138 asks *if the key-check skip exists, does
  STATUS.md acknowledge it?* — so deleting the skip, the step, or the whole job makes the guard
  vacuously true. A guard with a conditional precondition guards nothing when the precondition
  is cut. This is a defect in a test written two days ago, found by the instrument, not by me.
- **`gauntlet-pr.yml`, 12 of 15 cuts.** #137 guarded the *discovery* step. The step that actually
  *verifies* — `Re-run gauntlet on each changed submission` — can be deleted, swallowed, or
  `if: false`d without a test noticing. The workflow could once again report green on every
  submission, by a different route from the one that was repaired.
- `audit-claims.yml` (8), `diffgate.yml` (2), `nightly-heavy.yml` (16), `replications.yml` (6):
  nothing in `tests/` knows these workflows exist.
- All 8 surviving M-TRIGGER cuts: any of eight workflows could be switched to manual-only and
  no test would say so.
- `package.json` scripts `build`, `test`, `test:watch`.

The full list, one row per survivor:

| mutant | operator | cut |
|---|---|---|
| MUTE-001 | M-TRIGGER | audit-claims.yml: `on:` → dispatch only |
| MUTE-002 | M-JOB | audit-claims.yml → delete job `audit-claims` |
| MUTE-003 | M-STEP | audit-claims.yml / audit-claims / `Install styxx`: delete |
| MUTE-004 | M-SWALLOW | audit-claims.yml / audit-claims / `Install styxx`: swallow |
| MUTE-005 | M-STEP | audit-claims.yml / audit-claims / `Write PR body to a file`: delete |
| MUTE-006 | M-SWALLOW | audit-claims.yml / audit-claims / `Write PR body to a file`: swallow |
| MUTE-007 | M-STEP | audit-claims.yml / audit-claims / `Audit claims against substrate`: delete |
| MUTE-008 | M-SWALLOW | audit-claims.yml / audit-claims / `Audit claims against substrate`: swallow |
| MUTE-009 | M-TRIGGER | diffgate.yml: `on:` → dispatch only |
| MUTE-010 | M-JOB | diffgate.yml → delete job `diffgate` |
| MUTE-011 | M-TRIGGER | gauntlet-pr.yml: `on:` → dispatch only |
| MUTE-013 | M-STEP | gauntlet-pr.yml / verify-submissions / `Install styxx (editable, with test extras)`: delete |
| MUTE-014 | M-SWALLOW | gauntlet-pr.yml / verify-submissions / `Install styxx (editable, with test extras)`: swallow |
| MUTE-017 | M-STEP | gauntlet-pr.yml / verify-submissions / `Install per-submission requirements`: delete |
| MUTE-018 | M-SWALLOW | gauntlet-pr.yml / verify-submissions / `Install per-submission requirements`: swallow |
| MUTE-019 | M-GUARD | gauntlet-pr.yml / verify-submissions / `Install per-submission requirements`: `if:` → false |
| MUTE-020 | M-STEP | gauntlet-pr.yml / verify-submissions / `Re-run gauntlet on each changed submission`: delete |
| MUTE-021 | M-SWALLOW | gauntlet-pr.yml / verify-submissions / `Re-run gauntlet on each changed submission`: swallow |
| MUTE-022 | M-GUARD | gauntlet-pr.yml / verify-submissions / `Re-run gauntlet on each changed submission`: `if:` → false |
| MUTE-023 | M-STEP | gauntlet-pr.yml / verify-submissions / `No submissions changed`: delete |
| MUTE-024 | M-SWALLOW | gauntlet-pr.yml / verify-submissions / `No submissions changed`: swallow |
| MUTE-025 | M-GUARD | gauntlet-pr.yml / verify-submissions / `No submissions changed`: `if:` → false |
| MUTE-028 | M-STEP | leaderboard-submission.yml / run-submission / `Install styxx + NLI stack`: delete |
| MUTE-029 | M-SWALLOW | leaderboard-submission.yml / run-submission / `Install styxx + NLI stack`: swallow |
| MUTE-030 | M-STEP | leaderboard-submission.yml / run-submission / `Install submission dependencies (if any)`: delete |
| MUTE-031 | M-SWALLOW | leaderboard-submission.yml / run-submission / `Install submission dependencies (if any)`: swallow |
| MUTE-033 | M-SWALLOW | leaderboard-submission.yml / run-submission / `Identify the submission file changed in this PR`: swallow |
| MUTE-034 | M-STEP | leaderboard-submission.yml / run-submission / `Run submission against 8 benchmarks`: delete |
| MUTE-035 | M-SWALLOW | leaderboard-submission.yml / run-submission / `Run submission against 8 benchmarks`: swallow |
| MUTE-036 | M-TRIGGER | nightly-heavy.yml: `on:` → dispatch only |
| MUTE-037 | M-JOB | nightly-heavy.yml → delete job `nightly-heavy` |
| MUTE-038 | M-STEP | nightly-heavy.yml / nightly-heavy / `Install CPU-only torch first (keeps the install lean & fast)`: delete |
| MUTE-039 | M-SWALLOW | nightly-heavy.yml / nightly-heavy / `Install CPU-only torch first (keeps the install lean & fast)`: swallow |
| MUTE-040 | M-STEP | nightly-heavy.yml / nightly-heavy / `Install styxx + heavy-dep extras`: delete |
| MUTE-041 | M-SWALLOW | nightly-heavy.yml / nightly-heavy / `Install styxx + heavy-dep extras`: swallow |
| MUTE-042 | M-STEP | nightly-heavy.yml / nightly-heavy / `Premise check — the heavy libs MUST import`: delete |
| MUTE-043 | M-SWALLOW | nightly-heavy.yml / nightly-heavy / `Premise check — the heavy libs MUST import`: swallow |
| MUTE-044 | M-STEP | nightly-heavy.yml / nightly-heavy / `Run AUC-backing instrument tests`: delete |
| MUTE-045 | M-SWALLOW | nightly-heavy.yml / nightly-heavy / `Run AUC-backing instrument tests`: swallow |
| MUTE-046 | M-STEP | nightly-heavy.yml / nightly-heavy / `Guard — no test skipped for a MISSING dependency`: delete |
| MUTE-047 | M-SWALLOW | nightly-heavy.yml / nightly-heavy / `Guard — no test skipped for a MISSING dependency`: swallow |
| MUTE-048 | M-GUARD | nightly-heavy.yml / nightly-heavy / `Guard — no test skipped for a MISSING dependency`: `if:` → false |
| MUTE-049 | M-STEP | nightly-heavy.yml / nightly-heavy / `Step summary`: delete |
| MUTE-050 | M-SWALLOW | nightly-heavy.yml / nightly-heavy / `Step summary`: swallow |
| MUTE-051 | M-GUARD | nightly-heavy.yml / nightly-heavy / `Step summary`: `if:` → false |
| MUTE-052 | M-TRIGGER | publish.yml: `on:` → dispatch only |
| MUTE-053 | M-JOB | publish.yml → delete job `build` |
| MUTE-054 | M-STEP | publish.yml / build / `Install build tooling`: delete |
| MUTE-055 | M-SWALLOW | publish.yml / build / `Install build tooling`: swallow |
| MUTE-056 | M-STEP | publish.yml / build / `Build distribution artifacts`: delete |
| MUTE-057 | M-SWALLOW | publish.yml / build / `Build distribution artifacts`: swallow |
| MUTE-058 | M-STEP | publish.yml / build / `Verify both artifacts present`: delete |
| MUTE-059 | M-SWALLOW | publish.yml / build / `Verify both artifacts present`: swallow |
| MUTE-060 | M-STEP | publish.yml / build / `Twine integrity check`: delete |
| MUTE-061 | M-SWALLOW | publish.yml / build / `Twine integrity check`: swallow |
| MUTE-062 | M-JOB | publish.yml → delete job `publish` |
| MUTE-063 | M-JOB | publish.yml → delete job `github-release` |
| MUTE-064 | M-TRIGGER | replications.yml: `on:` → dispatch only |
| MUTE-065 | M-JOB | replications.yml → delete job `verify` |
| MUTE-066 | M-STEP | replications.yml / verify / `Identify submitted replication files`: delete |
| MUTE-067 | M-SWALLOW | replications.yml / verify / `Identify submitted replication files`: swallow |
| MUTE-068 | M-STEP | replications.yml / verify / `Verify each submission against its canonical receipt`: delete |
| MUTE-069 | M-SWALLOW | replications.yml / verify / `Verify each submission against its canonical receipt`: swallow |
| MUTE-070 | M-TRIGGER | telescope.yml: `on:` → dispatch only |
| MUTE-071 | M-JOB | telescope.yml → delete job `telescope` |
| MUTE-072 | M-STEP | telescope.yml / telescope / `check vendor keys`: delete |
| MUTE-073 | M-SWALLOW | telescope.yml / telescope / `check vendor keys`: swallow |
| MUTE-074 | M-STEP | telescope.yml / telescope / `install deps`: delete |
| MUTE-075 | M-SWALLOW | telescope.yml / telescope / `install deps`: swallow |
| MUTE-076 | M-GUARD | telescope.yml / telescope / `install deps`: `if:` → false |
| MUTE-077 | M-STEP | telescope.yml / telescope / `run telescope`: delete |
| MUTE-078 | M-SWALLOW | telescope.yml / telescope / `run telescope`: swallow |
| MUTE-079 | M-GUARD | telescope.yml / telescope / `run telescope`: `if:` → false |
| MUTE-080 | M-STEP | telescope.yml / telescope / `commit daily snapshot`: delete |
| MUTE-081 | M-SWALLOW | telescope.yml / telescope / `commit daily snapshot`: swallow |
| MUTE-082 | M-GUARD | telescope.yml / telescope / `commit daily snapshot`: `if:` → false |
| MUTE-083 | M-TRIGGER | test.yml: `on:` → dispatch only |
| MUTE-084 | M-JOB | test.yml → delete job `test` |
| MUTE-085 | M-STEP | test.yml / test / `Install styxx + test deps`: delete |
| MUTE-086 | M-SWALLOW | test.yml / test / `Install styxx + test deps`: swallow |
| MUTE-087 | M-STEP | test.yml / test / `Lint (ruff — pinned to py39, catches the 3.9-3.11 syntax class)`: delete |
| MUTE-088 | M-SWALLOW | test.yml / test / `Lint (ruff — pinned to py39, catches the 3.9-3.11 syntax class)`: swallow |
| MUTE-089 | M-STEP | test.yml / test / `Import smoke (every top-level module must import on this version)`: delete |
| MUTE-090 | M-SWALLOW | test.yml / test / `Import smoke (every top-level module must import on this version)`: swallow |
| MUTE-091 | M-STEP | test.yml / test / `Run tests`: delete |
| MUTE-092 | M-SWALLOW | test.yml / test / `Run tests`: swallow |
| MUTE-093 | M-JOB | test.yml → delete job `package` |
| MUTE-094 | M-STEP | test.yml / package / `Build wheel`: delete |
| MUTE-095 | M-SWALLOW | test.yml / package / `Build wheel`: swallow |
| MUTE-096 | M-STEP | test.yml / package / `Verify all subpackages + key modules are in the wheel`: delete |
| MUTE-097 | M-SWALLOW | test.yml / package / `Verify all subpackages + key modules are in the wheel`: swallow |
| MUTE-098 | M-JOB | test.yml → delete job `core-minimal` |
| MUTE-099 | M-STEP | test.yml / core-minimal / `Install styxx with NO extras`: delete |
| MUTE-100 | M-SWALLOW | test.yml / core-minimal / `Install styxx with NO extras`: swallow |
| MUTE-101 | M-STEP | test.yml / core-minimal / `Core surface must import + run with numpy only`: delete |
| MUTE-102 | M-SWALLOW | test.yml / core-minimal / `Core surface must import + run with numpy only`: swallow |
| MUTE-105 | M-SWALLOW | test.yml / typecheck-js / `Install the package's own dependencies from its lockfile`: swallow |
| MUTE-107 | M-SWALLOW | test.yml / typecheck-js / `Typecheck`: swallow |
| MUTE-108 | M-SCRIPT | package.json → delete script `build` |
| MUTE-109 | M-SCRIPT | package.json → delete script `test` |
| MUTE-110 | M-SCRIPT | package.json → delete script `test:watch` |

## 5. What this run does and does not say

It says which of 120 cuts to the checking apparatus the test suite notices, with the test named
for each of the 18 it does. It does not say the 101 survivors are defects: §4's first bin is a
list of checks whose failure is loud somewhere else, and that is an acceptable place for a check to
live. It does not say CI would miss the cuts — the oracle is `tests/`, by design, because that is
where this repository says its guards live.

What it does say, without qualification, is that the design has been honoured for eight subjects
and one job, and for nothing else. The four findings that started this were not four instances.
They were the first four rows of a table that has 101.

## 6. Next

In order of what a survivor costs when it is real:

1. A behavioural guard on `test.yml → test → Run tests`: a test that reads the step and asserts the
   command it runs is `pytest` over `tests/`, is not swallowed, and is not guarded off — the shape
   #137's shell-running test already has.
2. Repair the #138 guard so it is unconditional: `telescope.yml` must contain the key-check and
   the skip, not merely be consistent with STATUS.md if it happens to.
3. Guard the gauntlet *verification* step the way its discovery step is guarded.
4. A second operator, M-ALTER, that changes a pin rather than removing it, to separate targeted
   kills from collection crashes (§3).
5. Re-run. The number to watch is not the kill count; it is the length of §4's second bin.

Each of those is a new test, and each one — by construction — will move a row from SURVIVED to
KILLED in the next receipt, or it is not the guard it claims to be.
