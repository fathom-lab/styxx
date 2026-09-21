# PREREG — MUTE-3: the guards under mutation, and where the chain of guards ends

Fathom Lab · 2026-09-21 · Frozen before any level-2 mutant is applied and before the level-1
replication under the new rule is run. Follows `RESULT_mute2_guards_under_mutation_2026_09_20.md`
(114 KILLED / 5 SURVIVED / 1 UNREACHED; 7/7).

## Where this came from, stated before anything is measured

MUTE-1 asked: cut a check out of CI, does a test go red? MUTE-2 wrote two guards and 114 of 120
cuts now do. The RESULT's last line asked the next question and answered it in advance: *mutate the
guards themselves and ask whether anything notices — on this tree the answer is known, nothing
would.* This document measures that, then adds the anchors the answer demands, measures again, and
names where the chain of guards ends — because it must end somewhere, and the honest thing is to
say where rather than to pretend it does not.

Three things are fixed before a run:

**1. The verdict rule changes, and the change is stated before it is used.** MUTE-1 and MUTE-2 ran
under the v1 rule: a baseline-passing test kills a mutant if it is *anything but passed* on the
mutant — red, or not collected at all. MUTE-2 had to predict, mutant by mutant, which kills were
tests that had merely *vanished* with the step they were about, so the count could be read. At
level 2 that rule is not merely noisy but wrong: deleting a test function makes its own id vanish,
and a deleted test cannot be its own alarm. **v1.1**: a mutant is KILLED by every baseline-passing
test that is RED on it, and by every red id that was not in the baseline at all (a collection
error); a baseline-passing id that is simply not collected has VANISHED and is recorded beside the
kills, never among them. The receipt carries all three lists per mutant. The instrument's own tests
hold the rule (`tests/test_harness_mutation.py`).

**2. The guards are declared.** `GUARDS` in the instrument: the six test files whose purpose is to
hold the harness or a pinned subject in place — `test_gauntlet_pr_verifies_something` (#137),
`test_telescope_status_is_honest` (#138), `test_ci_runs_the_js_typecheck` (#140),
`test_port_is_current`, `test_harness_manifest` and `test_ci_steps_propagate_failure` (#142).
Declared, not inferred, for the reason SUBJECTS are.

**3. Three level-2 operators**, applied to the guards:

| operator | what is cut |
|---|---|
| M-GFILE | the guard file is deleted |
| M-GFUNC | one `test_*` function is deleted, decorators included |
| M-GVACUOUS | every `assert` in one `test_*` function becomes `assert True` — the guard runs, passes, checks nothing |

M-GVACUOUS is the operator that matters. A deleted guard is at least absent; a hollowed one is
present, green, and named exactly as before. It is the shape of MUTE-1's finding about the guard
written in #138 (vacuous when its precondition was cut), and of every `assert True` a tired
person has ever left behind.

The oracle is unchanged: the suite restricted by the same pattern; the mutated guard files are in
it, which is the point.

## The runs, in order

**MUTE-2r** — MUTE-2's 120 level-1 mutants, same tree fingerprint (`c3eaf2a72f66fe98…`), same
oracle, under v1.1. Its only purpose is to show the rule change alters nothing at level 1 and to
record the vanished/red split that MUTE-2 could only predict.

**Run A** — level 2 on the tree as it stands: 6 guard files, 33 test functions, **72 mutants**
(6 M-GFILE, 33 M-GFUNC, 33 M-GVACUOUS).

**Run B** — level 2 after three anchors are added, and nothing else:

- *the guard census*: the manifest pins, for each declared guard, the names of its test functions
  (`guards` in `tests/harness_manifest.json`; a missing guard is recorded as absent, not skipped);
- *one control per MUTE-2 guard*: a test that calls the guard on a fixture it must reject and
  requires an `AssertionError` — `test_the_manifest_guard_rejects_a_tree_missing_a_job` in
  `test_harness_manifest.py`, `test_the_propagation_guard_rejects_a_swallowed_step` in
  `test_ci_steps_propagate_failure.py`. Controls for the four older guards are not written here:
  three of those files belong to pull requests still open, and a control is theirs to carry;
- *the anchor outside the suite*: a step in `test.yml`'s `test` job, before `Run tests`, that
  collects the two MUTE-2 guard files (`pytest --collect-only`) and so fails if either is gone.
  The manifest pins that step; the step pins the manifest test. Cutting both at once is silent.

Run B's population is Run A's plus the two controls: 6 files, 35 functions, **76 mutants**.
Level-2 ids (`MUTE-G001`…) are positional and shift between A and B; predictions below are by
file, function and operator, and the scorer resolves them that way.

## Predictions, committed now

**P1 — the rule change is invisible at level 1.** MUTE-2r's verdict on every one of the 120
mutants equals MUTE-2's: 114 KILLED, 5 SURVIVED, 1 UNREACHED, the same ids in each bin.

**P2 — what MUTE-2 could only predict, recorded.** In MUTE-2r, `vanished_on_mutant` is non-empty
for exactly 45 mutants: the 33 M-STEP mutants of steps that reach an external command, the 11
M-JOB mutants of jobs with at least one such step, and MUTE-119; `errors_on_mutant` is non-empty
for MUTE-119 alone.

**P3 — nothing guards the guards.** Run A: **0 KILLED, 72 SURVIVED, 0 UNREACHED.** Every guard
file can be deleted, every guard test deleted, every guard hollowed to `assert True`, and no test
in the repository goes red.

**P4 — the census guards existence.** Run B, M-GFILE: 5 of 6 KILLED, each with
`tests.test_harness_manifest::test_the_harness_matches_the_committed_manifest` red among the
killers; the sixth — the deletion of `tests/test_harness_manifest.py` itself — SURVIVES.

**P5 — the census guards every test but the one that carries it.** Run B, M-GFUNC: 34 of 35
KILLED, each by the manifest test going red; the deletion of
`test_the_harness_matches_the_committed_manifest` SURVIVES.

**P6 — only a control guards against hollowing.** Run B, M-GVACUOUS: the two controls have no
`assert` of their own (they use `pytest.raises`) and are UNREACHED; of the remaining 33,
**exactly 2 are KILLED** — `test_the_harness_matches_the_committed_manifest` by its control and
`test_the_step_cannot_hide_the_failure_of_what_it_calls` by its control — and **31 SURVIVE**: the
26 test functions of the four older guards, and the five auxiliary tests of the MUTE-2 guards.
Stated plainly: after this cycle, a guard hollowed to `assert True` is still invisible unless
someone wrote the control, and controls exist for two of thirty-three.

**P7 — the totals.** Run B: **41 KILLED, 33 SURVIVED, 2 UNREACHED.**

**P8 — where the chain ends.** The two Run B survivors that are not hollowings are the same file:
`tests/test_harness_manifest.py`, deleted whole or with its matching test deleted. Inside the
suite nothing can notice that, and no test can be written that does, because the test that would
notice is the one being cut. The anchor step in `test.yml`, applied to the Run B tree with the
manifest test file deleted, fails (this is checked by running the step's shell on that mutant,
outside the oracle, and is reported as a demonstration, not a verdict).

## Gates

| gate | what it holds | bar |
|---|---|---|
| G-M3-1 (instrument) | the instrument's own tests pass, rule and operators included | `tests/test_harness_mutation.py` green |
| G-M3-2 (baseline A) | Run A's baseline is MUTE-2's | 534 tests passing; the same 9 excluded |
| G-M3-3 (baseline B) | the anchors broke nothing | every test that passed on Run A's baseline passes on Run B's, plus the two controls |
| G-M3-4 (reach) | UNREACHED as enumerated | A: 0; B: exactly the two controls |
| G-M3-5 (no hand labels) | verdicts only from the per-test comparison | the RESULT may read survivors; it may not reclassify one |
| G-M3-6 (ledger) | every prediction scored | HIT/MISS for P1–P8, published whatever they say |

G-M3-1 through G-M3-4 are blocking.

## What would abandon this

The rule, if P1 fails: a v1.1 that changes a level-1 verdict is not the refinement this document
claims, and the cycle stops until it is understood. The design, if P6 fails in the direction of
*more* kills: a hollowed guard caught by something other than its control means the population or
the operator is not what this document says it is.

## Honest statement of what a passing MUTE-3 means

That the guards written this week were, until this cycle, guarded by nothing; that a census and an
anchor now make their *absence* loud everywhere but one named place; that their *hollowing* is loud
only where a control was written, which is two places; and that the chain of guards in this
repository ends at a named pair — a test file and a workflow step — whose simultaneous deletion
nothing would notice. It does not say the guards are correct. It says exactly how far "would
anything notice" reaches, and where it stops.

## Running it

```
python -m benchmarks.harness_mutation.mute --inventory --level 2
git worktree add /tmp/mute-tree HEAD
python -m benchmarks.harness_mutation.mute --run --level 2 --tree /tmp/mute-tree --out papers/harness/mute3_receipt_A.json
# add the anchors, commit, then:
python -m benchmarks.harness_mutation.mute --run --level 2 --tree /tmp/mute-tree --out papers/harness/mute3_receipt_B.json
python -m benchmarks.harness_mutation.mute --run --tree /tmp/mute-tree --out papers/harness/mute2r_receipt.json
python papers/harness/mute3_score.py
```
