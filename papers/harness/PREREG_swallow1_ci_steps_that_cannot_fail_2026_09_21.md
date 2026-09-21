# PREREG — SWALLOW-1: how many CI steps in the wild cannot fail

Fathom Lab · 2026-09-21 · Frozen before the census touches any repository in the population.
Follows MUTE-1 (#141), MUTE-2 (#142) and MUTE-3 (this stack).

## Where this came from, stated before anything is measured

Everything in the MUTE arc so far is one codebase deep — this one. `benchmarks/silent_pass/`
says the same of itself. MUTE-2's behavioural guard, though, needs nothing but a repository's
workflow files: execute every `run:` step the way Actions does, with an empty PATH so every
external command fails, and see whether the step goes red. A step that goes green there has
hidden the failure of what it called. That is a measurement that can be taken on any repository
on GitHub without cloning its code, in milliseconds per step.

So this cycle takes it on a hundred. The question is the SILENT-PASS shape — a check that
reports green while measuring nothing — in CI, at scale: **how common is a step that cannot
fail, and how often is that step the one whose purpose was to verify something?**

## The population, fixed before the run

The 100 repositories that received the most agent-authored pull requests in the AIDev corpus
(hao-li/AIDev, CC-BY-4.0, the 71,016 eligible PRs of EXTERNAL-1's ledger), ranked by count. The
list is `papers/harness/swallow1_population.json`, with each repository's rank and PR count. It
is used because it already exists in this repository's receipts, because agent-heavy
repositories are the population styxx gates, and because it was not chosen after looking at any
workflow. A repository that cannot be cloned, has no `.github/workflows`, or has no bash step is
recorded, not dropped; rates are over the repositories that have what the rate needs, and the
denominators are printed.

A pilot of five repositories ranked 101–105 — outside the population — was run while the
instrument was written and is disclosed here: 706 steps, of which 659 belonged to one repository
carrying generated agentic workflows (`*.lock.yml`, GitHub Agentic Workflows), 1 shell swallow,
45 toolless, 0 verification swallows. Three things in this document come from that pilot: the
repository (not the step) is the primary unit, because one repository can carry a thousand
generated steps; generated agentic workflows are a separate stratum; and `continue-on-error:
true` is counted beside the shell verdict, because it hides failure without touching the
script.

## The instrument

`benchmarks/harness_mutation/census.py`. For each repository: a blob-less sparse clone of
`.github/workflows`; every `run:` step of every parseable workflow; the step's shell resolved
(step, then job defaults, then workflow defaults, then `pwsh` on a Windows runner, else bash);
bash/sh steps executed with `bash -eo pipefail`, empty PATH, `${{ … }}` → `x`,
`command_not_found_handle` logging every external command reached, 30 s timeout. Verdicts:

| verdict | meaning |
|---|---|
| PROPAGATES | exited non-zero: the failure of what it called was not hidden |
| SWALLOWS | exited 0 with at least one external command reached and failed |
| TOOLLESS | exited 0 having reached no external command: nothing to propagate in this context |
| SYNTAX | bash could not parse the step after substitution |
| TIMEOUT | did not finish in 30 s |
| NOT_BASH | pwsh / powershell / cmd / python / other; not executed |

Beside the verdict: `continue_on_error` (step or job), a category from the step's name and first
command (test, lint, typecheck, build, install, publish, other — a heuristic; the receipt keeps
the text), whether the workflow is a generated agentic one (`*.lock.yml`), and whether the step
contains `git (diff|log|status|ls-files|rev-parse|describe|fetch) … || true` — #137's exact shape.

**"Cannot fail"** = SWALLOWS, or `continue-on-error: true` on an executed step.
**"Verification step"** = category test, lint or typecheck.

## Predictions, committed now

Units are repositories unless stated. "Executed" means verdict PROPAGATES or SWALLOWS.

**P1 — the shape is common.** At least **half** of the repositories with an executed bash step
have at least one step that cannot fail.

**P2 — it reaches the checks.** At least **one in ten** of those repositories has a *verification*
step that cannot fail — a test, lint or typecheck step that reports green whatever happens.

**P3 — but it is not the norm inside a repository.** The median per-repository shell swallow rate
(SWALLOWS over executed, hand-written workflows only) is **below 5%**.

**P4 — #137 is not unique.** At least **three** repositories contain a step with a git query
whose failure is turned into an empty answer (`git diff|log|status|… || true`).

**P5 — generated agentic workflows are a different regime.** At least **ten** repositories carry
`*.lock.yml` workflows, and among the steps of those workflows at least **30%** are
`continue-on-error: true`.

**P6 — the method sees most steps.** TOOLLESS is at most **25%** of bash steps, and SYNTAX plus
TIMEOUT together at most **2%**.

**P7 — swallows hide in `|| true` but are not the same thing.** Among hand-written executed steps
that contain `|| true` (or `|| :`, `|| echo`, `|| exit 0`), fewer than **60%** are SWALLOWS —
the wrapper is usually on a command that is not the last one, and the step can still fail.

## Gates

| gate | what it holds | bar |
|---|---|---|
| G-S1-1 (population) | the list is the frozen list | 100 entries, sha256 recorded in the receipt |
| G-S1-2 (reach) | the census reached the population | ≥ 90 repositories cloned; every failure named |
| G-S1-3 (method) | the guard and the census execute steps the same way | the census reproduces MUTE-2's verdicts on this repository: 33 propagate, 5 toolless |
| G-S1-4 (no hand labels) | verdicts and categories come from the instrument | the RESULT may read steps; it may not reclassify one |
| G-S1-5 (ledger) | every prediction scored | HIT/MISS for P1–P7 |

G-S1-1 to G-S1-3 are blocking.

## What would abandon this

The method, if G-S1-3 fails. The claim "the shape is common", if P1 and P2 both fail: then the
SILENT-PASS shape in CI is this repository's problem and a handful of others', and the RESULT says
so. P3 is expected to hold in either case and is not evidence for the claim.

## Honest statement of what a passing SWALLOW-1 means

That among the repositories agents send the most pull requests to, a step that cannot fail is the
rule rather than the exception at the repository level, that in a measurable minority of them the
step that cannot fail is a check, and that the shape #137 had — a git query whose failure becomes
an empty answer — recurs. It does not say any of those steps is a defect: a best-effort comment
step is right to swallow. It says where a reader should look, and it leaves the text of every
step in the receipt so the reader can.

## Running it

```
python -m benchmarks.harness_mutation.census --tree .                                  # this repository: G-S1-3
python -m benchmarks.harness_mutation.census --repos papers/harness/swallow1_population.json --out papers/harness/swallow1_receipt.json
python papers/harness/swallow1_score.py
```
