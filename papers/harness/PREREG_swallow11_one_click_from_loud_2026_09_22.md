# PREREG — SWALLOW-11: one click from loud — every hidden check the gate has caught, replayed as the Action would have reported it

Fathom Lab · 2026-09-22 · Frozen before any pair of the population is replayed.
Follows `RESULT_swallow10_the_baseline_2026_09_22.md` (INVALID on its merge-signal gate as
frozen, 8/8 reported, not claimed).

## Where this came from, stated before anything is measured

The gate says a change brings a check that hides its own failure, and verifies a repair. This
cycle ships it where the change is made: `ci-audit/action.yml`, a GitHub Action that runs the gate
on the change that triggered the workflow, puts an error annotation on the line that hides the
check, writes the verified repair into the job summary and — opt-in — posts it on the pull request
as a review suggestion that applies with one click. Whether that last step is available depends on
facts about real changes the tests cannot supply: whether the line that hides the check is inside
the change's own diff (a review suggestion can only sit there), and whether the repair's lines
are. So: replay every hidden check the gate has ever caught in this program's receipts, exactly as
the Action would report it, and count how often the fix is one click away.

## 1. The instrument

`benchmarks/harness_mutation/one_click.py`, frozen at
`047a11235ce35f02bb3c1e2e467857bf2c1e54d5cbd0b0fc9eede8dc6a6ff807`, calling the product as it
ships: `styxx/ciaudit/action.py` at `a9615d9bb909b4435fd301f4aa6496320af81828177766bf7c4718ee3c25a76f`
(`positions`, `target`, `repaired_text`, `suggestion`, `apply_suggestion`, `hunks`, `within`,
`readable`) and the living gate `styxx/ciaudit/differential.py` at
`95f6ccf1f981a21882af152296d4ae6cd1f27c0d3a12fe66a3d22ea46b321428`.

**Population.** Every firing BASE..HEAD pair of three receipts, once: SWALLOW-7's non-root
mainline commits that fire (`swallow7_receipt.json.gz`, file sha256 `c6b12d09…`; BASE the first
parent, read in the SWALLOW-6 clones); SWALLOW-9's audited firing pull requests
(`swallow9_receipt.json.gz`, `b78c4730…`; BASE from the pull request's commit list); SWALLOW-10's
audited, touching, firing pull requests not already in SWALLOW-9 (`swallow10_receipt.json.gz`,
`cc6b5dc1…`; the closest-branch BASE). **140 pairs in 60 repositories, 226 newly hidden checks by
the receipts**: 104 mainline commits (147 checks), 17 agents' pull requests (24), 19 more pull
requests (55 — people's 12 with 32, agent-signed 5 with 20, agents' 2 with 3). Kinds by the
receipts: 177 born hidden, 32 acquired, 17 became a check, hidden.

**Per check**, on HEAD's text and the change's own diff (`git diff -U3 -M BASE HEAD`, the context
a pull request shows):
- **located** — `positions` finds the step and `target` names the annotation's lines: the
  `continue-on-error:` that hides it (the step's, else the job's) when one does, else the `run:`
  block, else the step's first line.
- **visible** — the annotation's first line lies inside a hunk of the change's diff.
- **rebuilt** — the verified repair's text is rebuilt by the function that built it, identical to
  the diff the gate printed.
- **reproduces** — the suggestion (`suggestion`: the smallest run of HEAD's lines whose replacement
  gives the repaired text), applied, is the repaired text line for line.
- **one-click** — rebuilt, reproduced, and the suggestion's lines lie inside one hunk.
- **size** — HEAD lines the suggestion replaces.

## 2. What was known at the freeze

- The receipts' counts above, and their verified-repair counts (SWALLOW-7 101 of 147; SWALLOW-9
  22 of 24; SWALLOW-10 people's 31 of 32). The replay has not been run on any pair of the
  population. Its mechanics were exercised on the scripted fixtures of
  `tests/test_harness_one_click.py` and `tests/test_ciaudit_action.py`, on one pull request of a
  repository outside every receipt (the network path: no workflow changed) and on one commit of
  `mlflow/mlflow` the gate did not fire on (the clone path).
- In the fixtures a pre-existing job-level `continue-on-error` hides a newly added check with a line
  the change did not touch — not visible, not one click. How common that is in the population is
  not known.

## 3. Predictions, committed now

Over the checks the replay re-reads.

**P1 — located** (calibrated on the fixtures). At least **98%** are located.

**P2 — visible** (blind). At least **80%** of located checks have their annotation inside the
change's diff.

**P3 — one click** (blind). At least **60%** of all re-read checks are one click away.

**P4 — placeable when repaired** (blind). At least **80%** of the checks with a rebuilt repair are
one click away.

**P5 — small** (blind). At least **75%** of one-click suggestions replace at most **3** of HEAD's
lines.

**P6 — born is easier** (blind). The one-click share of born-hidden checks exceeds that of
acquired checks.

**P7 — the pull request is the place** (blind). The one-click share of checks in pull requests
(SWALLOW-9 and -10) is at least that of checks in mainline commits (SWALLOW-7).

**P8 — people's before agents'** (blind). The one-click share of people's checks (SWALLOW-10
`human`) is at least that of the agents' (SWALLOW-9 and SWALLOW-10 agent groups).

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S11-1 (instrument) | the population rule, one-click on the fixture, a fix outside the diff, determinism; the Action's tests | `tests/test_harness_one_click.py` and `tests/test_ciaudit_action.py` green |
| G-S11-2 (reproduction) | the product re-reads the receipts | ≥ 95% of the 140 pairs re-read without error, and ≥ 95% of those give the receipt's newly hidden set (workflow, job, step, kind) exactly |
| G-S11-3 (frozen underneath) | the instrument, the Action, the living gate, the three receipts | the hashes above |
| G-S11-4 (construction) | a suggestion is the verified repair | every rebuilt repair reproduces: `reproduces` = `rebuilt` for 100% of re-read checks |
| G-S11-5 (ledger) | every prediction scored | HIT/MISS for P1–P8 |

G-S11-1 to G-S11-4 are blocking.

## 5. What would abandon this

G-S11-2: if the product does not re-read what the research instrument read, the replay is not
of the gate the receipts describe. G-S11-4: if a suggestion, applied, is not the verified repair,
the one-click claim is not about the verified repair.

## 6. Honest statement of what a passing SWALLOW-11 means

That, on the changes where the gate caught a hidden check — mainline commits of 96 repositories
and the 2025 pull requests of agents and people — the Action would have put an annotation on the
line that hides it, inside the diff the author is reading, at the stated rate; and that the fix,
verified under the same fault, would have been a one-click suggestion at the stated rate. It does
not say the author would have clicked. It does not say GitHub's API accepts every placeable
suggestion (the placement rule here is the documented one: lines inside one hunk of the diff); a
suggestion to a mainline commit is a replay of what a pull request with that diff would have
shown. The annotation's visibility is the diff with three lines of context a pull request shows by
default.

## 7. Running it

```
python -m benchmarks.harness_mutation.one_click --r7 papers/harness/swallow7_receipt.json.gz --r9 papers/harness/swallow9_receipt.json.gz --r10 papers/harness/swallow10_receipt.json.gz --clones7 <swallow-6 clones> --work <clones> --out papers/harness/swallow11_receipt.json
python papers/harness/swallow11_score.py
```
