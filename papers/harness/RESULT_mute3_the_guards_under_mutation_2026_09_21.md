# RESULT — MUTE-3: nothing guarded the guards (0 of 72); after the anchors their absence is loud everywhere but one file, their hollowing in two places of thirty-three — and a control anchors more than it was written for

Fathom Lab · 2026-09-21 · Scores the three runs recorded in `mute2r_receipt.json`,
`mute3_receipt_A.json` and `mute3_receipt_B.json` against the preregistration frozen at sha256
`ddcd207a489811ede222fe0be9d5739d83b7e15163f7cebe08dd1f5a6dde708d`. Not amended.

Instrument `benchmarks/harness_mutation/mute.py` v1.1, sha256 `defc94722e01a49d…`, the same bytes
for all three receipts (schema `styxx.harness-mutation/v1.1`). MUTE-2r and run A: tree
`b6836600…`, harness fingerprint `c3eaf2a72f66fe98…` — MUTE-2's — 534 tests passing on the
unmutated tree; 1672 s and 1058 s. Run B: tree `e9136788…` (run A's tree plus the three anchors
and nothing else), fingerprint `fb6d868cff477878…`, 537 passing; 959 s. Scored by
`mute3_score.py` → `mute3_scored.json`; the baselines are recorded by id in
`mute3_baselines.json`.

**VALID. 5 of 8 predictions HIT; P5, P7 and P8 MISS, all three on one event: run B killed one
mutant more than predicted — 42 KILLED, 32 SURVIVED, 2 UNREACHED against 41 / 33 / 2 — because
the control written to catch the manifest guard's *hollowing* also catches its *deletion*. Run A:
0 KILLED, 72 SURVIVED, 0 UNREACHED, as predicted. MUTE-2r: 114 / 5 / 1, verdict for verdict
MUTE-2's.**

## 0. What was done

MUTE-2 wrote two guards and the silent cuts went from 101 of 120 to 5. This cycle cut the
guards. Three things were fixed first (the preregistration says them at length): the verdict rule
became **v1.1** — a mutant is killed only by a test that goes *red* on it, and a baseline-passing
test that is simply no longer collected has *vanished*, which the receipt records and never
counts, because a deleted test cannot be its own alarm; the six guard files were **declared**
(`GUARDS` in the instrument); and three level-2 operators were defined — **M-GFILE** deletes a
guard file, **M-GFUNC** deletes one `test_*` function, **M-GVACUOUS** turns every `assert` in one
function into `assert True`, so the guard runs, passes and checks nothing.

Then three runs, in order. **MUTE-2r**: MUTE-2's 120 level-1 mutants under the new rule, to show
it changes nothing at level 1 and to record the vanished/red split MUTE-2 could only predict.
**Run A**: level 2 on the tree as it stood — 72 mutants. **Run B**: level 2 after three anchors
and nothing else — a *guard census* in the manifest (for each declared guard, the names of its
test functions); one *control* per MUTE-2 guard, a test that calls the guard on a fixture it must
reject and requires an `AssertionError`; and an *anchor step* in `test.yml`, before `Run tests`,
that collects the two guard files and so fails if either is gone. 76 mutants.

## 1. Gates

| gate | bar | observed | |
|---|---|---|---|
| G-M3-1 (instrument) | `tests/test_harness_mutation.py` green | 17 passed (rule, split, level-2 enumeration and operators) | pass |
| G-M3-2 (baseline A) | 534 passing, the same 9 excluded as MUTE-2 | 534; the same 9 | pass |
| G-M3-3 (baseline B) | every test that passed on A's baseline passes on B's, plus the two controls | 534 of 534 pass; B gained exactly three: the two controls and the propagation guard's own case for the anchor step (`test.yml::test::Anchor — the guard files still collect`) | pass |
| G-M3-4 (reach) | UNREACHED: A none; B exactly the two controls | A: none; B: the two controls | pass |
| G-M3-5 (no hand labels) | verdicts only from the per-test comparison | every KILLED names a red test, every SURVIVED names none; §3 reads, it does not reclassify | pass |
| G-M3-6 (ledger) | P1–P8 scored | below | pass |

**Scorer correction, stated:** the scorer first encoded G-M3-3 as a count, "B's baseline = A's +
2", and failed on the legitimate third gain — the anchor step is a `run:` step, and the
propagation guard is parametrized over every `run:` step, so it gained a case. The gate as
written is a set condition; the scorer now holds it from the ids in `mute3_baselines.json`
(produced by `mute3_baselines.py` on the two trees) and prints whatever B gained. The
preregistration was not touched; its hash is checked before anything is scored.

## 2. Predictions, scored

| | prediction | observed | |
|---|---|---|---|
| P1 | MUTE-2r equals MUTE-2 verdict for verdict: 114 / 5 / 1, the same ids in each bin | 120 of 120 identical | **HIT** |
| P2 | `vanished_on_mutant` non-empty for exactly 45 mutants (33 M-STEP, 11 M-JOB, MUTE-119); `errors_on_mutant` for MUTE-119 alone | 45: M-STEP 33, M-JOB 11, M-SUBJECT 1 (MUTE-119); errors on MUTE-119 alone | **HIT** |
| P3 | run A: 0 KILLED, 72 SURVIVED, 0 UNREACHED | 0 / 72 / 0 | **HIT** |
| P4 | run B, M-GFILE: 5 of 6 KILLED with the manifest test red; the deletion of `test_harness_manifest.py` survives | 5 KILLED, each by the manifest test alone; the manifest file survives, `killed_by` empty | **HIT** |
| P5 | run B, M-GFUNC: 34 of 35 KILLED by the manifest test; deleting `test_the_harness_matches_the_committed_manifest` survives | **35 of 35 KILLED**; that deletion is killed by its control, `test_the_manifest_guard_rejects_a_tree_missing_a_job` | **MISS** |
| P6 | run B, M-GVACUOUS: the two controls UNREACHED; exactly 2 KILLED, each by its control; 31 SURVIVE | 2 UNREACHED; 2 KILLED, each by its control alone; 31 SURVIVED | **HIT** |
| P7 | run B: 41 / 33 / 2 | **42 / 32 / 2** | **MISS** |
| P8 | the non-hollowing survivors of run B are the manifest file and its matching test; the anchor step fails with that file deleted | **one** non-hollowing survivor, the manifest file; the anchor step exits 4 with the file deleted and 0 intact | **MISS** |

| operator | run A | run B |
|---|---|---|
| M-GFILE (6) | 0 / 6 | 5 / 1 |
| M-GFUNC (33 → 35) | 0 / 33 | 35 / 0 |
| M-GVACUOUS (33 → 35) | 0 / 33 | 2 / 31 (+2 unreached) |
| **total** | **0 / 72** | **42 / 32 (+2)** |

(KILLED / SURVIVED. Run B's population is run A's plus the two controls.)

## 3. The miss, read

The preregistration reasoned that deleting `test_the_harness_matches_the_committed_manifest`
must survive inside the suite, "because the test that would notice is the one being cut". The
receipt says otherwise, and the reason is one line of the control:

```python
with pytest.raises(AssertionError):
    test_the_harness_matches_the_committed_manifest(root=fixture, manifest=pinned)
```

A control that calls its guard **by name** is an anchor on the guard's *existence*, not only on
its *content*. Delete the function and the control raises `NameError` — red, a kill under v1.1,
`killed_by` naming the control and nothing else, while the guard's own id is listed as vanished.
The same shape shows on the propagation guard: the deletion of
`test_the_step_cannot_hide_the_failure_of_what_it_calls` (MUTE-G069) is killed by the manifest
test *and* by its control, where the preregistration expected the manifest test alone. Nothing
else in run B differs from the prediction: the other 33 function deletions are killed by the
manifest test alone, every hollowing that had a control is killed by that control alone, every
hollowing that had none survives.

So the chain of guards inside the suite ends **one mutant earlier** than the preregistration
said: not at a file *and* a function, but at the file. `tests/test_harness_manifest.py` holds the
manifest guard and its control together, and its deletion removes the only test that pins the
guard census and the only test that names the guard, at once. No other test references either.
That is the fixed point, and it is a single one.

The anchor pair holds it from outside the suite, and both directions are demonstrated rather than
scored, because neither is a verdict of the oracle:

- `mute3_anchor_demo.json` — the anchor step of `test.yml`, run as a shell on the run-B tree:
  exit 0 intact (*47 tests collected*), exit 4 with the manifest test file deleted.
- `mute3_anchor_pin_demo.json` — the instrument's own level-1 M-STEP mutant of the anchor step
  (MUTE-091 on that tree), applied with the instrument's `apply`: the manifest test passes on the
  intact tree and fails on the mutant, and nothing else goes red.

Cut the file and the step goes red; cut the step and the test goes red; cut both at once and
nothing in this repository says a word. The preregistration named that pair as where the chain
ends, and it is.

## 4. What the receipts say beyond the counts

- **The rule change is what makes run A readable.** Under MUTE-2's v1 rule — anything but
  *passed* is a kill — run A would have read **39 KILLED / 33 SURVIVED**: every file deletion
  "killed" by the tests that vanished with it (8, 7, 3, 66, 4 and 35 of them), every function
  deletion "killed" by its own vanished id, and only the 33 hollowings surviving. The v1.1 receipt
  lists those same ids under `vanished_on_mutant` and counts none of them; 0 of 72. The
  hollowings survive under either rule, which is the point of the operator.
- **MUTE-2r records what MUTE-2 predicted.** 45 mutants have vanished ids — the 33 M-STEP
  mutants of steps that reach a tool (each vanishing its own behavioural case), the 11 M-JOB
  mutants of jobs with such a step, and MUTE-119, whose deletion of a subject breaks collection
  of `tests/test_calib1_score.py`; that one alone has an `errors_on_mutant` entry
  (`::tests.test_calib1_score`, an id that was never in the baseline). MUTE-2's P4 said exactly
  this from the outside; the instrument now says it itself.
- **Run B's kills are three tests and no others.** The manifest test kills every file deletion
  but its own (5) and every function deletion but its own (34, one of them shared with the
  propagation control); the two controls kill the two hollowings and the two deletions of the
  functions they call. No test of any other guard file kills anything; the four older guards are
  held in place by the census and by nothing of their own.
- **Thirty-one guards can be hollowed to `assert True` and nothing notices** (P6): the 26
  functions of the four older guard files, and the five auxiliary tests of the two MUTE-2 guards
  (`test_the_manifest_is_committed`, `test_the_manifest_is_not_hollow`,
  `test_the_manifest_pins_no_step_text`, `test_there_are_steps_to_hold`,
  `test_the_method_sees_a_swallow_and_passes_a_propagating_step`). The receipt names each.

## 5. What this does not say

- **Loudness is not truth, one level up.** After this cycle the *absence* of a guard is loud
  everywhere but one named file, and the *hollowing* of a guard is loud in two places of
  thirty-three. Neither says a guard is correct; a hollowed guard that had a control is caught
  because the control feeds it one fixture it must reject, which is one fixture.
- **The fixed point is not fixed.** The 2-cut — the manifest test file and the anchor step
  together — is silent by construction and stays so. A third anchor would move it, not remove
  it; the honest thing was to name it, and this cycle names a single mutant rather than the pair
  it predicted.
- **The demonstrations in §3 are shell runs, not verdicts.** The oracle is the suite; the
  anchor lives in CI; the two JSON files say so in their own `reading` field.
- **Controls exist for two guards because two guards belong to this stack.** The 26 functions
  of the four older guards are in pull requests still open, and a control is theirs to carry;
  §3 shows what a control buys — deletion and hollowing both — so the argument for writing them
  is now a measurement.
- **Level 2 mutates the declared guards only.** A test file that guards something and is not in
  `GUARDS` is not cut, not censused and not counted; declaring it is a deliberate act, as
  declaring a subject was in MUTE-1.
- **Run B's tree is this stack's tree, not `main`**: MUTE-2r and run A on `b6836600…`, run B on
  `e9136788…`, fingerprints in the receipts. Run B's baseline differs from run A's by exactly the
  three tests the anchors add (G-M3-3), and the 534 that passed before all still pass.

## 6. Next

- A control per older guard, in the pull request each belongs to; the mechanism is measured, the
  writing is the work.
- M-GVACUOUS hollows every assert at once. A per-assert variant (one `assert True` per mutant)
  would say which assertion each control actually reaches, and would show the fixture-shaped hole
  in §5's first point as a number.
- The behavioural guard needs nothing but workflow files, so its measurement can leave this
  repository: SWALLOW-1, next in this stack, executes every `run:` step of the 100 repositories
  agents send the most pull requests to and asks how many cannot fail.
