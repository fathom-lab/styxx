# RESULT — the sworn action ships, report-only: a runner mints the manifest after the turn, and exits zero on every verdict

Fathom Lab · 2026-09-05 · Spec: `SPEC_sworn_action_v01_2026_09_05.md`, frozen before any code.
Action: `sworn/action.yml`, `sworn/sworn_action.py`, `sworn/README.md`, sample workflow at
`sworn/examples/sworn.yml`. Tests: `tests/test_sworn_action.py`. **This document is itself sworn**:
its `rN` spans resolve against the manifest the JUnit adapter minted over the run of those tests,
and its `path:` spans against the sample run committed beside it. Leg 3, item 4 of
`papers/PLAN_the_next_level_2026_09_02.md`, under the plan's own label: **report-only until the
measurement prices FAILED**.

## What shipped

A composite action in its own subdirectory — the root `action.yml` is diffgate's and is untouched.
In a pull-request job it runs the project's own test command, mints a `sworn/manifest/0.2` from the
report and the event **after the turn** through `styxx.harness.junit` and `styxx.harness.github`,
verifies every sworn document the pull request touched against that manifest, writes a job-summary
table carrying the rung and the harness string on every row, and exits zero whatever the verdicts
were.
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/action" k="quote">The sample run records the action as `sworn-action/0.1`</sworn>,
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/harness" k="quote">minted by `sworn/sworn_action.py`</sworn>,
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/exit_code" k="numeric">and the job exited 0.</sworn>

**The command's exit status is a record, never the job's.** A gate this lab has not measured is a
gate this lab does not ship: the plan prices FAILED only after the measurement runs, and until then
a contradicted span is reported and nothing is blocked.

## The rung, and the fork sentence

<sworn r="path:papers/sworn/sworn_action_sample.run.json#/rung" k="quote">The sample run declares rung `L2`</sworn>
— declared by the workflow, never detected by the adapter. The README states the limit the plan
requires, and the run's own harness string carries it into every manifest: on a `pull_request` from
a fork the minting job runs the workflow file as it exists in the pull request's head, so the
manifest is minted by a party the claimant controls and the sentence L2 rests on does not hold. It
holds for a workflow pinned to the base branch, or for a manifest attested outside the job.
Every blob the turn added or modified at head enters the manifest's authored set, so a document
swearing to bytes the turn itself wrote is MALFORMED rather than believed:
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/manifest/authored" k="numeric">the sample records 5 of them.</sworn>

## What the sample run shows

The sample (`sworn_action_sample.py` → `sworn_action_sample.run.json`, `.summary.md`, and the three
receipts) drives the action end to end over a temporary repository, with no network.
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/documents/0/verdict" k="quote">The pull-request body verifies as `SWORN-HELD`</sworn>
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/documents/0/counts/HELD" k="numeric">on 1 bound span;</sworn>
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/documents/2/counts/HELD" k="numeric">a held document binds 5 spans and holds;</sworn>
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/documents/1/verdict" k="quote">a document that swears something false reads `SWORN-FAILED`</sworn>
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/documents/1/counts/FAILED" k="numeric">on its 1 contradicted span;</sworn>
and a document citing a receipt the runner never minted reads
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/documents/3/counts/UNRESOLVED" k="numeric">1 unresolved span</sworn>
rather than an error — the author cited what the harness did not hand over, and the verdict says so
instead of the job failing.
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/skipped/0/why" k="quote">Two files are skipped and say why: one `carries no <sworn tag`</sworn>,
and a committed sidecar is never verified against a supplied manifest, because the manifest it was
sworn against is the one embedded in it.
<sworn r="path:papers/sworn/sworn_action_sample.run.json#/manifest/digest" k="quote">The manifest the sample minted records its digest as `9c550169b241998aec8e77576f9781f4ae8639230162db46cf2959298c64ce32`.</sworn>

## The tests

<sworn r="r1" k="numeric">35 tests passed in the run this document swears to,</sworn>
<sworn r="r2" k="numeric">with 0 failures</sworn>
— the whole action driven over a fake event file, a fake report and canned documents in a temporary
git repository: the summary table, the composed receipt numbering, the fork refusals, a command that
exits non-zero, a report that never appears, a document that cites `rN` the runner did not mint, and
exit 0 on every one of them.
<sworn r="r4#/outcome" k="quote">The evidence reader's own outcome over that report is `PASSED`.</sworn>

## What this does not say

That the action has run on GitHub: it has not — no token in this lab can push a workflow file, so
the sample workflow at `sworn/examples/sworn.yml` is for the operator to copy, and this repository
enables nothing. That L2 has been verified: the rung is the workflow's declaration, printed, never
checked. That a `SWORN-HELD` document is true — it says the spans its author bound were checked
against bytes the author did not write, at the commit named, and nothing about the sentences the
author left unbound. That the verdicts here should gate anything, today.

---

*The runner mints after the turn, verifies what the author bound, prints the rung beside every
verdict, and gets out of the way. What it may not do yet is stop anybody — that is the
measurement's to license, and the measurement has not run.*
