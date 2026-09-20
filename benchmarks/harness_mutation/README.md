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
