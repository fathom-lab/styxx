# PREREG — SWALLOW-4: the repair is loud — a verified one-line fix for every hidden and dropped check

Fathom Lab · 2026-09-21 · Frozen before any repair is tried on the population.
Follows `RESULT_swallow3_the_checks_that_are_actions_2026_09_21.md` (INVALID on its reproduction
gate; the catalogue's reading reported, not claimed).

## Where this came from, stated before anything is measured

Three cycles have read what a workflow does when one step's tools fail: it goes red (92%), it
hides a check's own failure (49 hand-written faults in 23 repositories, two thirds by
`continue-on-error`), or it silently drops a check (4, in 3 repositories). None of them asked the
next question, which is the one a maintainer asks: **what is the fix, and does it work?**

This cycle asks it of every hidden and dropped check in the population, with the same
instrument, and holds the answer to two conditions at once. A repair is a small edit to the
workflow text at the fault site. It is **loud** when the same fault, injected into the repaired
workflow, is RED in every healthy flavour that could interpret it before. It is **unchanged**
when the repaired workflow's healthy worlds are indistinguishable from the original's: every job
has the same result, every step is run or not run for the same reason, reaches a tool or not, and
is a model artifact or not, in both flavours. A repair is **verified** when both hold. A repair
that is loud and changes a healthy run has found what the original line was protecting — a
`|| true` after `grep`, whose no-match is an exit status — and is recorded as rejected, with the
flavour and the step that rejected it. The twin condition is the point: a fix that makes CI loud
by making it fail when nothing is wrong is not a fix.

## 1. The two repairs

Both are text edits, located through the YAML parser's marks so the receipt carries a real
unified diff. In order; the first verified one is the fault's repair.

1. **`no-continue-on-error`** — the fault step's `continue-on-error: true`, or its job's, is
   removed (the line deleted). Applies when one is present and alone on its line.
2. **`strict-shell`** — in the fault step's script: every trailing `|| true` and `|| :` is
   removed (also before `;`, `)`, `&`, `|`); every `set +e` line is removed; `set -eo pipefail`
   is ensured (`set -e` or `set -o pipefail` added when only one is missing), after a shebang if
   there is one. Applies when the script changes. The repair does not touch `|| echo …`
   defaults, `if cmd; then` guards, or anything outside the fault step.
3. **`both`** — the two together, when both apply.

Nothing else. A fault none of the three makes loud, or that all three break, has no repair in
this cycle, and the RESULT reads why.

## 2. The instrument

`benchmarks/harness_mutation/repair.py`, on top of `action_checks.py` (SWALLOW-3's reading,
frozen at `0e723694…`) on top of `faults.py` (SWALLOW-2's, frozen at `d26a407c…`); it modifies
neither. For each target it re-analyses the original workflow on this machine (the baseline),
applies each candidate to the text, parses the result, re-analyses the repaired workflow, and
records for the same (job, index): the verdict by flavour, loudness by flavour, the healthy-shape
comparison by flavour, the diff and its size. The candidates share one memoised runner per
workflow, so a repair costs only the executions its changed step needs.

**Targets.** Every fault in the SWALLOW-3 receipt (`swallow3_receipt.json.gz`, the trees still on
disk at the same HEADs) whose verdict is SWALLOWED or FAIL_OPEN, in both strata: hand-written
(49 SWALLOWED — 33 under `continue-on-error`, 16 by the shell — and 4 FAIL_OPEN) and generated
(118 SWALLOWED, 1 FAIL_OPEN), 27 repositories. The 8 faults SWALLOW-2 read as "fewer" under the
counted reading are among the 49 and are flagged.

**What was known at the freeze.** The instrument was exercised on a synthetic fixture with the
five outcomes (verified by each repair, rejected by the twin condition, not loud) and on this
repository (no target). SWALLOW-2 §3–4 and SWALLOW-3 §3 were read again for the *shapes* of the
hidden and dropped checks — they are public — and the predictions below are calibrated on those
readings. No repair has been applied to any population workflow.

## 3. Predictions, committed now

Units are hand-written faults unless stated; a target whose baseline verdict on this machine
differs from the receipt's is excluded from every denominator and named (G-S4-2).

**P1 — most hidden checks have a verified repair.** At least **40 of the 49** hand-written
SWALLOWED faults (≥ 80%) are verified: the 33 under `continue-on-error` by the one-line repair,
and most of the 16 shell-hidden ones by `strict-shell`.

**P2 — some `|| true` are load-bearing.** At least **3** of the 16 shell-hidden SWALLOWED faults
are rejected by the twin condition: `strict-shell` makes them loud and breaks a healthy run — the
protected line is a `grep` that may not match, a `git diff --quiet` that answers by status, or a
command that fails on the model's `x`.

**P3 — the four dropped checks, one by one.** Declared non-blind (SWALLOW-2 §4 read them):
`antiwork/gumroad` `tests.yml` → `run_scope` → step 0: **verified** (`strict-shell` on the
`|| true` after the labels query); `getsentry/sentry-docs` `lint-external-links.yml` →
`check-pr` → step 1: **verified** (`strict-shell`); `crewAIInc/crewAI`
`update-test-durations.yml` → `update-durations` → step 4: **verified** (`no-continue-on-error`);
`antiwork/gumroad` `ci-green.yml` → `report` → step 1: **not verified** — fail-closed by design,
the script's own `unknown` path exits 0 and neither repair reaches it. Scored as four.

**P4 — the "fewer" cases.** Of the 8 faults flagged `fewer` (mlflow's loop under `set +e`,
prebid's two, sentry-docs' two, aspire's iterations, vscode's two diagnostics), at least **4**
are verified.

**P5 — repairs are small.** The median verified hand-written repair changes at most **3** lines
of the workflow (lines added plus lines removed in the unified diff), and no verified repair
changes more than **10**.

**P6 — the generated stratum is one line.** At least **90%** of the 119 generated targets
(`*.lock.yml`, 115 under `continue-on-error`) are verified, and at least 90% of those by
`no-continue-on-error` alone — a fix that belongs in the compiler, not in 118 files.

**P7 — the instrument does not lie about loudness.** No target is verified by a repair whose
diff is empty, and every verified repair's `both` candidate, when it applies, is verified too
(a bigger repair never undoes a smaller one's loudness). Structural; scored as a prediction
because the instrument could fail it.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S4-1 (instrument) | the instrument's own tests pass; the fixture run is deterministic; this repository has no target | `tests/test_harness_repair.py` green |
| G-S4-2 (baseline) | the targets are the receipt's faults, re-read on this machine | every target's baseline verdict equals the SWALLOW-3 receipt's, or is named and excluded; excluded ≤ 5% of hand-written targets |
| G-S4-3 (both halves) | verified means loud and unchanged | every verified repair records `loud` true in each previously interpretable flavour and `unchanged` true in both flavours; a repair that is loud but changes the healthy run is never verified |
| G-S4-4 (no hand labels) | the repairs are the three stated; the verdicts are the instrument's | no candidate outside `REPAIRS`; the RESULT may read a repair, not add or edit one |
| G-S4-5 (frozen underneath) | the instruments beneath are unchanged | `faults.py` `d26a407c…`, `action_checks.py` `0e723694…`, source receipt `swallow3_receipt.json.gz` `609e6645…` (file sha256), recorded in the receipt |
| G-S4-6 (ledger) | every prediction scored | HIT/MISS for P1–P7 (P3 as four) |

G-S4-1 to G-S4-5 are blocking.

## 5. What would abandon this

The claim that a repair can be verified without a runner, if G-S4-3 fails — a verified repair
that changed a healthy run means the twin condition is not what this document says it is. The
population claim, if G-S4-2 excludes more than 5%: the instrument beneath is known not to be
bit-reproducible (SWALLOW-3 §1) and this cycle inherits that; more than 5% would mean it cannot
even re-read its own targets.

## 6. Honest statement of what a passing SWALLOW-4 means

That for most hidden and dropped checks in the 100 repositories agents send the most pull
requests to, there is a one-to-three-line change to the workflow file that makes the same tool
failure loud and leaves a healthy run untouched, under a model of a healthy run — and that the
instrument can tell those apart from the changes that would break a healthy run, and says which
line each of those was protecting. It does not say the authors want the check to be blocking:
`continue-on-error` on a lint is a choice, and the repair measures that the choice is one line
from loud, not that it is wrong. It does not execute actions, other shells, or the code; a
repair is verified against the same model the verdict came from, with the same artifacts carried
past. The receipt carries every diff.

## 7. Running it

```
python -m benchmarks.harness_mutation.repair --receipt papers/harness/swallow3_receipt.json.gz --clones <clones> --out papers/harness/swallow4_receipt.json
python papers/harness/swallow4_score.py
```
