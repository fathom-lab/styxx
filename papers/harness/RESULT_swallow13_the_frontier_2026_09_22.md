# RESULT — SWALLOW-13: the frontier — on 549 repositories it was not designed on, the third repair stage verifies a repair for 13 of the 44 hidden checks the first two leave, twelve by putting a `$(...)` its line threw away on a line of its own; most of what is left is a check that passes when its list comes up empty

Fathom Lab · 2026-09-22 · Scores the receipt `swallow13_receipt.json.gz` against the preregistration
frozen at sha256 `125836d2b841ce31fafd525461cacabbb3a1669eac355914baabdb16cb60ea65`. Not amended.
One run scored; an earlier run was lost before it finished (§0).

Receipt: `papers/harness/swallow13_receipt.json.gz` (sha256 `41ca9668…`), built by the frozen
instrument `benchmarks/harness_mutation/frontier.py` (`7b3c2b12…`) calling the product at the frozen
hashes — `engine.py` `5c219887…`, `actions.py` `ecbcd4f9…`, `repair.py` `4282dbfa…`,
`repair_structural.py` `af46c492…`, `repair_frontier.py` `0fb104f0…` · population
`swallow13_population.json` (`6f65d8f7…`) · scored by `swallow13_score.py` → `swallow13_scored.json`.

**VALID. 5 of 8 predictions HIT.** All 549 repositories were read at their recorded tips — 6,479
workflows, none capped, no error. They hold **143** hand-written hidden checks in 71 repositories.
SWALLOW-4's stage verifies **86**, SWALLOW-5's **13**; of the **44** left, **the third stage verifies
13 (30%), in 9 repositories**: twelve by hoist-substitution, one by no-coe+guard-status, a median of
5 lines. Run twice, stage 3 gives the same outcome on all 44. Missed: the three stages together
verify **112 of 143 (78%)**, under the 85% predicted; no stage-3 candidate was loud and then
rejected for its healthy run; the readings match **2 of the 31** left (6%), under the coin's 20%.

## 0. What was done

The product change, the instrument, the population, the preregistration and the scorer were
committed at 17:39Z and the run started. At 17:55Z, with the run at about 450 of 549 repositories,
the environment deleted the session's scratch directory — the local repository with the freeze
commit, the clones and the per-repository results. Nothing of that run was read but its progress
lines, which print each repository's count of findings. Every frozen file was re-created and hashes
to its frozen value, byte for byte (the preregistration, the instrument, the five product files, the
population); the rebuilt freeze was committed at 18:04Z and the run started again from the first
repository, in the session's workspace. At 18:10–18:11Z the freeze went to GitHub
(`fathomlab-patch-54`, five commits; its tree is the local commit's, `949daf0a`), with the run at
repository 257. The run finished at 18:21Z: 549 repositories, two at a time, 1,029 s (median
1.7 s a repository). Then the scorer, then the reading below.

## 1. Gates

| gate | bar | result |
|---|---|---|
| G-S13-1 instrument | the four test files green | pass — 40 passed |
| G-S13-2 frozen underneath | instrument, five product files, population at the frozen hashes | pass — all seven |
| G-S13-3 the population is read | ≥ 90% of 549 fetched and audited without error | pass — **549 of 549**, none capped |
| G-S13-4 enough to read | stage-3 population ≥ 15 | pass — **44** |
| G-S13-5 determinism | stage 3 twice: the same candidates, verdicts, diffs, repair | pass — **44 of 44** |
| G-S13-6 ledger | P1–P8 scored | pass |

No deviation. One change after scoring, to the product, not to anything measured (§5).

## 2. Predictions, scored

| | predicted | observed | |
|---|---|---|---|
| P1 the stage reaches | ≥ 25% of the stage-3 population | **13 of 44 (29.5%)** | HIT |
| P2 not one script, many times | ≥ 20% by distinct script | **13 of 44** — no two checks here share a workflow and a script | HIT |
| P3 not one repository | ≥ 3 repositories | **9** | HIT |
| P4 the hoist | the hoist family largest, strictly | **hoist 12**, no-coe+guard-status 1 | HIT |
| P5 the three stages together | ≥ 85% of the read targets | **112 of 143 (78.3%)**: 86 + 13 + 13 | MISS |
| P6 the second half bites | a stage-3 candidate loud, rejected for its healthy run | **none**: the stage-3 candidates that changed a healthy run were not loud either | MISS |
| P7 the readings (a coin) | ≥ 20% of the residue | **2 of 31 (6.5%)**: one routed, one declared | MISS |
| P8 small | median stage-3 repair ≤ 6 lines | **5** (4–8) | HIT |

## 3. Reading

**What the third stage reaches is one idiom.** Eleven of the twelve hoists are the way a step hands
a value to later ones: `echo "name=$(tool …)" >> "$GITHUB_OUTPUT"` (or `$GITHUB_ENV`). The `echo`
succeeds whatever the tool does, so a failed query hands on an empty value. Seven of the thirteen
repairs are FAIL_OPEN — the empty value is a matrix or a switch a check depends on, and the check
silently does not run: `primer/view_components` builds the test matrix of three workflows with
`$(cat .github/version-matrix.json | jq -c .)`, so when `jq` fails the matrix is empty and the test
jobs run zero times, green; `onyx` its Go modules, `Composio` its toolchain versions (two
workflows), `spellbook` the model count that gates `dbt run`. Hoisted, the query's failure stops
the step that makes the value:

```diff
       - id: set-matrix
-        run: echo "matrix=$(cat .github/version-matrix.json | jq -c .)" >> $GITHUB_OUTPUT
+        run: |
+          set -eo pipefail
+          __sub1="$(cat .github/version-matrix.json | jq -c .)"
+          echo "matrix=${__sub1}" >> $GITHUB_OUTPUT
```

The other six: odigos's Helm drift check `if [[ $(git diff --exit-code) ]]`; DataDog's pinned sha
(`git rev-parse`) and a branch name built with `date`; OpenHD's `$(date …)` in a step named "test"
(two workflows); PostHog's `find … | grep -q .` guard behind a `continue-on-error`, both edits needed
(no-coe+guard-status). Three of the thirteen hoist a `date` in a step the check rule reads as a
check by its name — loud under the model, of little consequence in a real run.

The development set's other families did not recur: background-liveness (ten there, all of them
one repository's server starts), no-exit-zero and no-default-joined verified nothing here; they
stay in the stage, and SWALLOW-12's control with `|| exit 0` is what no-exit-zero repairs.

**What is left: the empty list.** Read by hand after scoring — my reading of the 31 scripts, not a
rule:

| | checks | what the script does when its tools fail |
|---|---|---|
| a loop over an empty list | 10 | the list comes from `< <(find …)`, `mapfile -t … < <(…)` or `$(seq …)`; the command that makes it fails, the loop runs zero times, and the check reports success. A strict shell does not see it: bash drops a process substitution's status (cloudposse/atmos's four skill validators, dotCMS ×2, zeroc-ice, Azure, camunda ×2) |
| "nothing to check" | 8 | the answer is empty — `git diff … \| grep … \|\| true`, a `jq` list, `grep -c … \|\| echo 0`, a CPU flag — and the script says so and exits 0 (terminal-bench ×2, port-labs ×2, repomix, camunda, wolfBoot, rensa) |
| decides whether to run | 5 | change detection (`if git diff … \| grep -q …`, `ct list-changed`): the tools fail, the answer is "no changes", and the check downstream does not run (mastra ×2, kuberay: FAIL_OPEN); or a planning step the name rule reads as a check (qdk, temporal) |
| a negative check | 4 | passes when a pattern is absent — `! otool … \| grep …`, `if ./tool -h \| grep -q Requires; then fail` — and a tool that fails prints no pattern (Chia, flox, wolfTPM, eliza) |
| other | 2 | an exit-code table that passes any code it does not know (mina-rust); `which ghc && ghc --version`, an `&&` list a strict shell exempts (serena) |
| read | 2 | PostHog's `available` flag, written to GITHUB_OUTPUT and read by a later step's `if:` — routed, and no reader fails the job; mssql-python's flake8 with a `::warning` "informational only, not blocking" — declared. Both are what they say |

Eighteen of the 31 pass on an empty list or an empty answer. The two kinds differ in what a model
with an empty-answer flavour can verify. "Nothing to check" takes the same exit in the healthy run
where every tool answers nothing, so no repair that fails on an empty answer can leave that run
unchanged — SWALLOW-5 said so of its own three. The empty loop is different: making the list's
producer a command of its own — its output to a file, the loop reading the file — lets a strict
shell see its failure, while in both healthy flavours the loop reads what it read before. That is
an edit the model can verify, and ten checks here need it.

**Why P5 missed.** This population's hidden checks are harder than SWALLOW-1's hundred. There, the
first two stages verified 46 of 53 (87%); here 99 of 143 (69%), and 23 of the 143 are FAIL_OPEN
against 4 of the 53 there. `continue-on-error` is still the largest single mechanism — 71 of the 86
stage-1 repairs remove it, one line each.

**Why P6 missed.** The stage's edits are narrow: a hoist leaves a healthy run as it was unless the
strict shell it adds stops some other line of the script, and where that happened here (camunda's
two shard checks: with the strict shell the step fails in both healthy flavours), the hoist was not
loud either. The second half of the verification rejected nothing that the first half passed.

**The dev set's routed reading was one repository.** Sixteen of the 23 checks the development set
left were one repository's ratchet; here one check of 31 is routed.

## 4. What this does not say

That a verified repair is right: RED is loud, not correct, and the model's healthy run is stubs —
a stub never answers 1. A hoisted `git diff --exit-code` whose real answer is 1 (the files differ)
stopped odigos's step before it printed its message; §5 is the change that keeps such a 1. The
strict shell every hoist adds makes the whole script strict, as SWALLOW-4's repair does, and a real
run can depend on a non-zero status the stubs never return. The residue's classes are my reading
after scoring, not a preregistered rule. The population is this program's (repositories where
agents' pull requests touched workflows), each read at one commit; a finding here is a check hidden
at the tip, not one a change brought. background-liveness costs five seconds of a step; nothing
here measured whether five is the right number.

## 5. What ships

The third stage, in the gate (`differential.fix_for`, stage `swallow-13`, after SWALLOW-5's), the
`--repair` path and its card, and the Action: its annotation and job summary carry the repair, or
for a check no stage repairs, the reading — routed ("its failure is routed, not lost: … read by …")
or declared ("the script says …"); `repair.apply_repair` rebuilds any stage's repair by name, so the
one-click suggestion works for a third-stage repair unchanged. `differential.STAGES` names the
stages; SWALLOW-12's plan is re-computed with the two it ran, and a test says the third repairs its
no-repair control. `frontier.py`, frozen at `7b3c2b12…` and pinned by
`tests/test_harness_frontier.py`.

**One change after scoring**, to `repair_frontier.py` (now `85806ef6…`): hoist-substitution keeps
the status 1 of a command whose 1 is an answer — grep and its kin, `diff`, `cmp`,
`git diff --exit-code`/`--quiet`, `jq -e`, `which`, `command -v` — as `__subN="$(…)" || [ $? -eq 1 ]`,
so only a status above 1 stops the step, the way SWALLOW-5's guard-status reads a guard. The model
cannot see the difference (its stubs never answer 1), so the change moves no verdict: stage 3 run
again with it on the 44 checks, from the run's own clones at the same tips
(`swallow13_after_fix.py` → `swallow13_after_fix.json`), verifies the same 13, and one diff
changes — odigos's, which now keeps its message.

## 6. Next

The empty loop: ten checks here, and an edit a strict shell and the model can both verify — the
producer of a process substitution run as a command of its own. The negative check (four) needs the
tool's status apart from the pattern's. And "nothing to check" (eight) is beyond what an
empty-answer flavour can verify; it needs a different kind of evidence, or a reading that says it.

## Erratum (2026-09-22, SWALLOW-14)

§0 says the environment deleted the session's scratch directory at 17:55Z. Almost certainly it did
not: the audit simulates a step by running its shell on the machine, tools stubbed, and a path the
script names outside its temporary directory is the machine's — a step's `rm -rf /tmp/*`, run as
root, deletes everything under `/tmp`, which is where all that the first run lost was (the
session's scratch directory, a signing helper). SWALLOW-14's first run lost the machine's root
filesystem the same way, and a fixture step reproduces it (`RESULT_swallow14_the_empty_list_2026_09_22.md`
§0). The receipt scored here is the second run's, which finished; like every earlier cycle's run it
ran unconfined, so a simulated step could see what another repository's steps had left on the
machine.
