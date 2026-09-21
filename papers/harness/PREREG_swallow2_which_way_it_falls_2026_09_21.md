# PREREG — SWALLOW-2: which way it falls — one fault at a time, through 100 repositories' workflows

Fathom Lab · 2026-09-21 · Frozen before the instrument touches any repository in the population.
Follows `RESULT_swallow1_ci_steps_that_cannot_fail_2026_09_21.md` (VALID, 6/7).

## Where this came from, stated before anything is measured

SWALLOW-1 executed every `run:` step of 100 repositories alone, with every external command
failing, and found that in 67 of 87 a step cannot fail and in 35 of 87 a check cannot fail. Its
RESULT named the sharpest limit of that method in one sentence: *it sees that a step cannot fail;
it cannot see which way it falls.* A query that fails and is answered with "nothing changed" may
skip the check (fail-open, #137's shape) or run everything (fail-closed, `serviceradar`'s), and
both exit 0 in the same way.

This cycle simulates the *workflow* rather than the step, so that a failure in one place can be
followed to what it does to the checks in another. The unit of measurement is a **fault**: one
step whose external commands all fail, in a workflow where everything else succeeds. The question
is what the workflow does then.

## The instrument

`benchmarks/harness_mutation/faults.py`. For each workflow file, two healthy worlds and one fault
world per fault site:

- **W+** — every external command exits 0. Two flavours, because a healthy run answers some
  queries with something and others with nothing, and the simulation cannot know which a script
  wants: in flavour `x` every command prints one line, `x`; in flavour `empty` it prints nothing,
  and `grep` and its kin exit 1 (no match), since "found nothing" is an exit status for them.
- **W−A** — the same, except that in step A every external command is not found (exit 127),
  exactly as SWALLOW-1 executed A alone.

Steps are executed in job order with the contexts Actions gives them: `$GITHUB_OUTPUT` and
`$GITHUB_ENV` are read back; `${{ steps.<id>.outputs.<k> }}`, `${{ env.<k> }}` and
`${{ needs.<job>.outputs.<k> }}` are substituted from them; every step's `if:` and every job's
`if:` and `needs:` are evaluated with a three-valued evaluator, in which a value the simulation
cannot know (an action's outputs, `github.*`, `inputs.*`, `matrix.*`, `secrets.*`) is *unknown*,
substitutes as `x` in a script, and makes a condition true — a step is skipped only when its
condition is false on what the simulation actually knows. A job whose matrix comes from an empty
`fromJSON(...)` runs zero times. Plain filesystem utilities (`mkdir`, `tee`, `rm`, `dirname`, …)
are real in the healthy worlds; `cd` and `pushd` never fail there; a script the repository ships
(`./ci/test.sh`) is a stub that behaves as its world says. A healthy-flavour step that exits
non-zero anyway has failed on its own logic under the model (`x` is not the version it expected):
that is an *artifact* of the model, recorded on the step, and the healthy world carries on past
it; W−A ignores the same artifacts, so a failure W−A has and W+ does not is caused by the fault.

**Fault sites**: every bash `run:` step that reaches at least one external command when executed
alone (SWALLOW-1's PROPAGATES or SWALLOWS; TOOLLESS steps have nothing to fail).

**A check** (verification step): a step whose script invokes a known test / lint / typecheck
runner or a script whose name says test / lint / check / verify / validate (`verification()` in
the instrument, a stated rule), or whose name says so and does not also say install / build /
setup / report / generate / …; a fixer (`--fix`, `:fix`) is not a check. Tighter than SWALLOW-1's
category on purpose: here it decides a verdict. The rule was tuned by eye, before this freeze,
against the step names and first lines in SWALLOW-1's receipt of this same population — contact
with the population's text, stated here; no fault verdict was taken on the population before the
freeze. On this repository's own `gauntlet-pr.yml` the rule does *not* recognise the gauntlet
step as a check (its name says "Re-run gauntlet", its body calls a bespoke verifier), so the
instrument says NO_CHECK for #137's own fault site; G-S2-3 holds the mechanism instead.

**Verdicts**, per fault, in precedence order; a fault is judged in both healthy flavours and takes
the strongest verdict among the flavours that can interpret it:

| verdict | meaning |
|---|---|
| RED | a job goes red in W−A that is not red in W+ (or goes red earlier): the failure of A's tools is loud |
| FAIL_OPEN | nothing goes red, and a check that reached its runner in W+ is skipped, or runs without reaching it, in W−A: the failure silently removed a check |
| SWALLOWED | nothing goes red, A is itself a check and reached its runner in W−A: the check ran and its failure was hidden (`\|\| true`, `continue-on-error`) |
| ABSORBED | nothing goes red and every check reaches its runner as in W+: A's failure changed nothing about the checks |
| NO_CHECK | no check in A's job or downstream of it reaches a runner in W+: there was no check to protect |
| BASELINE_RED / BASELINE_SKIPPED | A itself fails on its own logic, or does not run, in both healthy flavours: uninterpretable, reported, not counted |

"Interpretable" faults are those with one of the first five verdicts. For a FAIL_OPEN fault the
receipt names each dropped check and its mechanism (`step-if`, `job-if`, `job-needs`,
`job-empty-matrix`, `unreached`) and whether it is cross-step (a different step or job from A).

## The population, and what touched it before the freeze

The same 100 repositories as SWALLOW-1 (`swallow1_population.json`, sha256 `5ac2789b6d67eed8…`),
re-cloned (workflows may have changed since; the receipt records each HEAD). The pilot: ten
repositories ranked 101–110, outside the population, on which the instrument was debugged and
revised five times (draining stdin in the healthy stubs; real filesystem utilities; `cd`/`pushd`
that cannot fail; `grep` exiting 1 in the empty flavour; carrying the healthy world past model
artifacts). Its last run: 742 fault sites, hand-written 163 — RED 144, BASELINE_RED 13,
NO_CHECK 3, ABSORBED 2, FAIL_OPEN 1 (`chrxh/alien`, a nightly deploy whose `git log --since` is
answered "no changes"). Two of SWALLOW-1's eight read repositories' workflow files were used, with
the pilot, to diagnose model artifacts (which step fails first in W+, and why); no fault verdict
was taken on them.

## Predictions, committed now

Units are hand-written workflows (not `*.lock.yml`) and interpretable faults unless stated;
repositories are those with at least one interpretable hand-written fault.

**P1 — the loud majority.** RED is at least **80%** of interpretable hand-written faults.

**P2 — fail-open exists at the repository level.** At least **six** repositories have a
hand-written FAIL_OPEN fault.

**P3 — it mostly reaches across steps.** Among hand-written FAIL_OPEN faults, at least **half**
are cross-step: the dropped check is a different step, or a different job, from the fault.

**P4 — the reading of SWALLOW-1, tested.** Declared non-blind: these five come from
SWALLOW-1's RESULT §4, read by eye. Each is scored on its own.
- P4a `mlflow/mlflow` `master.yml` → job `database` → `Run tests`: FAIL_OPEN, in-step
  (`unreached`).
- P4b `getsentry/sentry-docs` `Get changed files` (the `|| true` git diff): FAIL_OPEN,
  cross-step.
- P4c `primer/react` `Get source files changes`: FAIL_OPEN, cross-step.
- P4d `carverauto/serviceradar` `Decide whether this Mix project needs lint` ("fail-closed"):
  **not** FAIL_OPEN.
- P4e `airbytehq/airbyte` `Check for changes` (a failed `git diff --quiet` lands in the
  "changes detected" branch): **not** FAIL_OPEN.

**P5 — the swallowed check is the commoner shape.** SWALLOWED faults occur in at least **eight**
repositories, and in more repositories than FAIL_OPEN faults do.

**P6 — the model reaches most of the map.** Interpretable faults are at least **85%** of
hand-written fault sites; artifact failures are at most **10%** of the steps run in W+ (flavour
`x`).

**P7 — the checks themselves are loud.** Of the hand-written checks that reach their runner in
W+ and are fault sites, at least **90%** are RED under their own fault (their own tools failing);
the rest are SWALLOWED or FAIL_OPEN.

## Gates

| gate | what it holds | bar |
|---|---|---|
| G-S2-1 (population) | the list is SWALLOW-1's frozen list | 100 entries, sha256 in the receipt |
| G-S2-2 (reach) | the instrument reached the population | ≥ 90 repositories cloned; every failure named; every capped repository named |
| G-S2-3 (method) | the instrument agrees with SWALLOW-1 where they overlap, and sees #137 | `swallow2_self.json`: this repository's alone verdicts equal the self-census (33 PROPAGATES); `gauntlet-pr.yml` at `main` — the two gated steps are skipped and the job is green in W−discover; at #137 the same fault is RED |
| G-S2-4 (instrument) | the instrument's own tests pass | `tests/test_harness_faults.py` green; the self-run is deterministic |
| G-S2-5 (no hand labels) | verdicts and checks come from the instrument | the RESULT may read faults; it may not reclassify one |
| G-S2-6 (ledger) | every prediction scored | HIT/MISS for P1–P7 (P4 as five) |

G-S2-1 to G-S2-4 are blocking.

## What would abandon this

The method, if G-S2-3 fails. The claim that direction is measurable, if P6 fails badly — an
instrument that can read fewer than two thirds of the map is not the instrument this document
describes, and the RESULT says so before it says anything else.

## Honest statement of what a passing SWALLOW-2 means

That for one failing step at a time, a workflow's response can be read without a runner, a
token or the code — and that read says, per fault, whether the pipeline goes red, silently loses a
check, hides a check's own failure, or does not care. It says where fail-open lives, in named
steps, with the mechanism. It does not say those steps are wrong: a nightly that does nothing when
there is nothing new is right to skip. It measures a model of a healthy run, and the model's
artifacts are counted and printed. It does not execute actions, Windows, or anything but bash.

## Running it

```
python papers/harness/swallow2_self.py --before <gauntlet-pr.yml@main> --after <gauntlet-pr.yml@137>   # G-S2-3
python -m benchmarks.harness_mutation.faults --repos papers/harness/swallow1_population.json --out papers/harness/swallow2_receipt.json
python papers/harness/swallow2_score.py
```
