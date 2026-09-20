# PREREG — MUTE-2: two guards written under mutation pressure, scored by the instrument that demanded them

Fathom Lab · 2026-09-20 · Frozen before `mute.py` is run against a single mutant with the new
guards in the oracle. Follows `RESULT_mute1_harness_mutation_2026_09_20.md` (18 KILLED / 101
SURVIVED / 1 UNREACHED; 84% of cuts to the checking apparatus invisible to the test suite).

## Where this came from, stated before anything is measured

MUTE-1 left a reading list. Read in full, the 101 survivors sort into two shapes, and the shapes
suggest two guards — not one per survivor, which would be a hundred tests written to the
instrument, but one per shape.

**Shape 1 — the check is no longer declared.** 73 of the survivors are a job deleted, a step
deleted, a trigger switched to manual-only, a guard flipped to `false`, or an npm script removed.
Nothing in `tests/` knows what CI is supposed to consist of, so nothing notices when a piece
leaves. The guard for this shape is a **manifest**: `tests/harness_manifest.json` pins, for every
workflow, the events it fires on; for every job, its name and its `if:`; for every `run:` step,
its name and its `if:`; and the names of the npm scripts. `tests/test_harness_manifest.py` fails
when the tree differs from the manifest, and says how to regenerate it. This makes a structural
cut *loud* — it cannot be made without a visible manifest edit — the same guarantee
`papers/build_index.py` gives for arcs.

**Shape 2 — the check is still declared and can no longer fail.** 37 survivors are `run:` steps
wrapped `( … ) || true`. Every textual guard in the suite was fooled by that wrapper (P7 of
MUTE-1, a HIT against the author's own guard); the one guard that was not fooled *ran the
step's shell*. The guard for this shape does the same for every step:
`tests/test_ci_steps_propagate_failure.py` executes each `run:` block the way Actions does
(`bash -eo pipefail`), in an empty directory, with an **empty PATH** — every external command
is caught by bash's `command_not_found_handle`, logged, and fails — and asserts the step exits
non-zero. A step that propagates the failure of what it calls goes red; a swallowed one goes
green. A step that reaches no external command at all under these conditions has nothing to
propagate and is reported as *skipped by name*, never as passed.

The two guards are deliberately separate, and the separation is the thing this cycle is
designed to show: **loudness is not truth.** A manifest that pinned the sha256 of every step's
text would kill every mutant MUTE-1 can make and would mean nothing, because any edit to CI
would break it. This manifest pins only what a deletion changes, and is *blind by construction*
to a swallow. The behavioural guard is blind by construction to a deletion — the deleted step
simply has no case. Which guard kills which mutant is therefore a prediction, and the receipt
can falsify it.

## What is being asked

Not "does the kill rate go up" — two guards written to kill mutants will kill mutants. The
questions are:

1. Does each guard kill exactly the shape it was written for, and nothing else?
2. Is anything lost — does every MUTE-1 kill remain, by the same test?
3. Which cuts remain silent even now, and are they the ones this document names in advance?

## The instrument, unchanged

`benchmarks/harness_mutation/mute.py` is byte-identical to the one MUTE-1 ran (sha256
`1913e76f87ae726f…`, recorded in both receipts). Its vocabulary, operators, oracle pattern,
verdict rule and exclusions are as in the MUTE-1 preregistration and are not restated. The
oracle pattern selects the two new test files by itself, because they name `.github/workflows`;
nothing is hand-added. `tests/test_harness_mutation.py` stays excluded (MUTE-1, Amendment A).

One property of the v1 verdict rule matters here and is stated before it can be mistaken for a
finding. A baseline-passing test whose id is **not collected** on the mutant counts as a kill
(the rule is `outcome != passed`, and absent is not passed). The behavioural guard is
parametrized one case per step, with the step's *name* in the id. Deleting a step therefore
makes that step's case vanish, and the rule will list the behavioural guard among the killers of
every M-STEP and M-JOB mutant of a non-exempt step. **That is vanishing, not detecting**, and
this document predicts it precisely so that it is not read as the behavioural guard catching
deletions. The receipt lists the killing tests by id; the scorer separates the two.

## The tree

The run is made on the tree MUTE-1's receipt applies to — `main` plus every pull request open on
2026-09-20 (#131–#140) plus #141 — with the four MUTE-2 files added. No workflow, script or subject
is touched: the **harness fingerprint restricted to MUTE-1's ten oracle files** must equal MUTE-1's,
`8da2770cfabb9fc631d6a71816e23ffe36ed75ae42f7f1826aa2b319d79c4ef1`. The receipt's own fingerprint
differs from that only because the oracle grew by two files.

## The population, enumerated before the run

`--inventory` reports the same **74 checks and 120 mutants** as MUTE-1, with the same ids
(`MUTE-001`…`MUTE-120`), because the inventory reads the same workflows in the same order.

The behavioural guard, run on the unmutated tree while this document was being written, reaches
an external command in 33 of the 38 `run:` steps and none in 5. The five, with the id of the
M-SWALLOW mutant each corresponds to:

| exempt step | M-SWALLOW mutant |
|---|---|
| `gauntlet-pr.yml → verify-submissions → Install per-submission requirements` | MUTE-018 |
| `gauntlet-pr.yml → verify-submissions → No submissions changed` | MUTE-024 |
| `leaderboard-submission.yml → run-submission → Install submission dependencies (if any)` | MUTE-031 |
| `nightly-heavy.yml → nightly-heavy → Step summary` | MUTE-050 |
| `telescope.yml → telescope → check vendor keys` | MUTE-073 |

Each of these steps either only prints, or takes a branch that the placeholder `x` (substituted
for every `${{ … }}`) makes empty. Wrapping one in `|| true` is invisible to the guard, and the
guard says so by skipping the case with the step's name rather than passing it.

## Predictions, committed now

Killer names below are pytest ids as the receipt records them:
`tests.test_harness_manifest::test_the_harness_matches_the_committed_manifest` is "the manifest
test"; `tests.test_ci_steps_propagate_failure::test_the_step_cannot_hide_the_failure_of_what_it_calls[<workflow>::<job>::<step name>]`
is "the behavioural case for that step".

**P1 — the manifest makes every structural cut loud.** All **73** M-TRIGGER (9), M-JOB (14),
M-STEP (38), M-GUARD (8) and M-SCRIPT (4) mutants are KILLED, and the manifest test is among the
killers of every one.

**P2 — the manifest is blind to a swallow.** The manifest test is among the killers of **no**
M-SWALLOW mutant. (If it were, the manifest would be pinning content it claims not to pin.)

**P3 — the behavioural guard kills exactly the reachable swallows.** Of 38 M-SWALLOW mutants,
the **33** whose step reaches an external command are KILLED with the behavioural case *for that
step* among the killers; the **5** exempt ones (MUTE-018, 024, 031, 050, 073) **SURVIVE** with no
killer at all.

**P4 — vanishing is not detecting.** For every mutant that is not an M-SWALLOW, the behavioural
guard's cases among its killers are exactly the cases of the steps the mutant deleted: for
M-STEP of a non-exempt step, that one step's case; for M-JOB, the cases of that job's non-exempt
`run:` steps; for M-STEP of an exempt step, for M-TRIGGER, for M-GUARD, for M-SCRIPT and for
M-SUBJECT, none. No behavioural case is ever among the killers of a mutant whose step still
exists, other than the M-SWALLOW of that same step. One stated exception: a mutant that crashes
test collection vanishes *every* test, behavioural cases included — MUTE-119 did this in MUTE-1
(removing the CALIB-1 pin line stops `calib1_score.py` importing) and is expected to again; for
that mutant the prediction is "every baseline-passing test is among the killers", which is the
crash, not a detection.

**P5 — nothing is lost.** Each of the 18 mutants KILLED in MUTE-1 is KILLED here, and the set of
tests that killed it in MUTE-1 is a subset of the set that kills it now.

**P6 — the totals.** Of 120 mutants: **114 KILLED, 5 SURVIVED, 1 UNREACHED** (`telescope/prompts.json`,
as in MUTE-1). Silent cuts fall from 101 to 5, and the 5 are named above.

**P7 — the harness did not move.** The harness fingerprint of the run tree restricted to MUTE-1's
ten oracle files equals MUTE-1's fingerprint, `8da2770cfabb9fc6…`.

## Observed while building, before the freeze — not a prediction

While the behavioural guard was being written it was run once on `main` at `98a5c368` plus #141
plus these four files, with the manifest regenerated for that tree. It failed on exactly one
case: `gauntlet-pr.yml::verify-submissions::Discover changed submissions` — *git, sort, awk
reached, yet the step exited 0*. That is the `|| true` that #137 removes, found by a guard that
knows nothing about that step. It is recorded here as an observation so the RESULT can report it
as one, and it is the reason the pull request carrying this cycle is based on #137 rather than on
`main`: on `main` the guard is red, correctly.

## Gates

| gate | what it holds | bar |
|---|---|---|
| G-M2-1 (control) | the guards pass on the unmutated tree | manifest matches; 33 behavioural cases pass, 5 skip by name, 0 fail |
| G-M2-2 (nothing broken) | every test that passed on MUTE-1's baseline passes on this baseline | the 495 of MUTE-1 ⊆ this run's passing set |
| G-M2-3 (instrument) | the instrument reaches its population | UNREACHED = 1, the one named |
| G-M2-4 (no hand labels) | verdicts come only from the per-test comparison | the RESULT may read survivors; it may not reclassify one |
| G-M2-5 (ledger) | every prediction is scored | HIT/MISS for P1–P7, published whatever they say |

G-M2-1, G-M2-2 and G-M2-3 are blocking: a run that fails any of them is INVALID and is not cited.

## What would abandon this

The guards, if P2 or P4 fails: a manifest that sees swallows, or a behavioural guard whose
kills are not by the step's own case, is not the separation this document claims, and the cycle
is reported as a failure of the design rather than repaired in the RESULT. The claim "loudness is
not truth" is not a prediction and cannot be rescued or falsified by a count; it is what the two
guards are built to make visible.

## Honest statement of what a passing MUTE-2 means

A run that survives its gates says that two tests, one of them fifty lines, make 114 of the 120
cuts MUTE-1 can make to this repository's checking apparatus visible to its test suite, and
says for each cut which test saw it and whether it saw it by failing or by no longer existing.
It does not say any check is correct. It does not say CI would run. It says a cut can no
longer be silent, and it names the five that still can.

## Running it

```
python -m benchmarks.harness_mutation.manifest --check
python -m pytest tests/test_harness_manifest.py tests/test_ci_steps_propagate_failure.py -q -rs
git worktree add /tmp/mute-tree HEAD
python -m benchmarks.harness_mutation.mute --run --tree /tmp/mute-tree --out papers/harness/mute2_receipt.json
python papers/harness/mute2_score.py
```
