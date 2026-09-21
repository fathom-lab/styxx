# MUTE-1 — mutation testing of the checking harness

Mutation testing asks of a test suite: *if I break the program, does a test go red?*
This asks the same question one level up, of the apparatus that checks the program:
*if I cut a check — delete the job, silence the step, flip its guard, delete the file
it reads, stop the workflow firing — does anything in this repository go red?*

```
python -m benchmarks.harness_mutation.mute --inventory     # the checks and the mutants; runs nothing
git worktree add /tmp/mute-tree HEAD
python -m benchmarks.harness_mutation.mute --run --tree /tmp/mute-tree
python papers/harness/mute1_score.py                       # the frozen predictions, scored
```

Seven operators (M-TRIGGER, M-JOB, M-STEP, M-SWALLOW, M-GUARD, M-SCRIPT, M-SUBJECT), one oracle
(the test files that read harness paths, selected by pattern, never by hand), three verdicts
(KILLED, SURVIVED, UNREACHED), and a receipt that names the test behind every kill. The
instrument refuses to mutate the checkout it lives in.

It measures whether the **test suite** guards the harness. It does not run CI. SURVIVED means
exactly *nothing in `tests/` would tell you*, and a survivor is a finding to be read, not a defect
to be counted — `papers/harness/RESULT_mute1_harness_mutation_2026_09_20.md` reads the first 101.

The receipt carries a *harness fingerprint*: a sha256 over every file a mutant can touch plus every
file the oracle reads. Two commits with the same fingerprint have the same harness, and a run
applies to either. That is how a run made on a locally merged tree is checked against the commit
that eventually carries the same files.

## The census (SWALLOW-1): the behavioural guard, taken to other repositories

MUTE-2's second guard needs nothing but a repository's workflow files: execute every `run:` step
with an empty PATH, so that every external command fails, and see whether the step goes red.
`census.py` does that for any list of repositories — a blob-less sparse clone of
`.github/workflows`, every bash step executed and classified (PROPAGATES, SWALLOWS, TOOLLESS,
SYNTAX, TIMEOUT, NOT_BASH), `continue-on-error`, a category, and #137's exact shape recorded
beside the verdict.

```
python -m benchmarks.harness_mutation.census --tree .                                   # this repository
python -m benchmarks.harness_mutation.census --repos papers/harness/swallow1_population.json --out receipt.json
python papers/harness/swallow1_score.py
```

It sees that a step *cannot fail*; it cannot see which way the step falls when it does not —
`papers/harness/RESULT_swallow1_ci_steps_that_cannot_fail_2026_09_21.md` says so, and reads the
38 verification steps it found one by one.

## Fault injection (SWALLOW-2): which way it falls

The census sees that a step cannot fail; it cannot see which way the step falls. `faults.py`
simulates the *workflow*: two healthy worlds (every tool succeeds and prints `x`, or prints
nothing), and for every bash step that reaches a tool, one fault world in which that step's
tools fail. Outputs, env, `if:`, `needs:` and `fromJSON` matrices are followed; what the
simulation cannot know is unknown and lets a step run. Each fault gets one verdict — RED,
FAIL_OPEN, SWALLOWED, ABSORBED, NO_CHECK — and the receipt names every dropped check with its
mechanism.

```
python -m benchmarks.harness_mutation.faults --tree .                                      # this repository
python -m benchmarks.harness_mutation.faults --repos papers/harness/swallow1_population.json --out receipt.json
python papers/harness/swallow2_self.py --before <gauntlet-pr.yml@main> --after <gauntlet-pr.yml@137>
python papers/harness/swallow2_score.py
```

It reproduces #137 with no runner, no token and no code: on `main` the two gated steps are skipped
and the job is green when the discover step's tools fail; on #137's tree the same fault is RED.
`papers/harness/RESULT_swallow2_which_way_it_falls_2026_09_21.md` has the map for 100
repositories, and the four runs it took to draw it.
