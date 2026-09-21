# PREREG — SWALLOW-5: the structural repairs — for the checks a strict shell cannot make loud

Fathom Lab · 2026-09-21 · Frozen before any structural repair is tried on the population.
Follows `RESULT_swallow4_the_repair_is_loud_2026_09_21.md` (VALID, 8/10).

## Where this came from, stated before anything is measured

SWALLOW-4 verified a repair for 39 of 53 hidden and dropped checks and read the 14 it could not:
one it rejected for changing a healthy run (`aspire`'s flaky-test iterations), one fail-closed by
design (`gumroad`'s `ci-green`), and twelve that a strict shell does not reach because the script's
own logic decides the exit status — a check whose failing command is the condition of an `if`
(`hmis`, `roslyn`, `serviceradar`: the tool that fails takes the green branch, and `set -e` does not
apply inside a condition), a verifier that compares an answer carrying a default (`selfxyz`'s two
`Verify branch and commit`, `langfuse`'s `|| echo`), a loop that counts (`mlflow`, `aspire`'s three
runsheet checks, `nodetool`), and a verifier that echoes (`gh-aw`). Its §6 named the next edits:
`exit 1` in the else-branch of a guard, `test -n` on a compared answer.

This cycle states two such edits, tries them on exactly that residue — every target of the
SWALLOW-4 receipt without a verified repair, 14 hand-written and 4 generated — and verifies each
on SWALLOW-4's two halves, unchanged: **loud** (the same fault is RED on the repaired workflow in
every healthy flavour that could read it before) and **unchanged** (a healthy run of the repaired
workflow is indistinguishable from the original's in both flavours).

**One of the two named edits is not the one stated here, and the reason is stated.** `test -n`
on an answer cannot be verified by this instrument: its `empty` flavour is a healthy run in which
every query answers nothing, so a repair that fails on an empty answer changes that healthy run
by construction. What a repair can hold is the *status* of the query. So the second edit removes
the default that hides the status, and lets the strict shell see it.

## 1. The two repairs

Text edits to the fault step's script, located and rewritten as SWALLOW-4's (the block scalar is
replaced in place; the receipt carries the unified diff). In order; the first verified one is the
fault's repair.

1. **`guard-status`** — a single-line `if CMD; then` or `if ! CMD; then` (also the two-line form
   with `then` on the next line), where CMD is not a shell builtin, test, arithmetic or
   assignment, becomes an explicit status capture in which a status above 1 fails the step:
   ```
   __rc=0; CMD || __rc=$?
   if [ "$__rc" -gt 1 ]; then echo "guard command failed (exit $__rc): CMD" >&2; exit "$__rc"; fi
   if [ "$__rc" -eq 0 ]; then                      # -ne 0 for the negated form
   ```
   The `|| __rc=$?` keeps CMD exempt from errexit, as the condition was. Status 1 stays the "no"
   of `grep`, `diff` and their kin, so a healthy run that finds nothing is untouched; a tool that
   fails outright is not a "no". Applies when at least one such `if` is found; every one in the
   script is rewritten. `elif`, `while`, `until`, and multi-line conditions are not touched.
2. **`no-default`** — every `|| echo …` fallback is removed: at the end of a line, inside
   `$( )`, or on a continuation line of its own (the previous line's `\` goes with it); then the
   shell is made strict exactly as SWALLOW-4's `strict-shell` does, so the query's failure reaches
   `set -e`. `|| true` is not a default (that was SWALLOW-4's repair); a `|| { …; }` block is not
   touched. Applies when at least one fallback is found.
3. **`both-structural`** — both edits, then the strict shell, when both apply.

Nothing else. In particular no edit reaches a loop that counts, and the RESULT will say so.

## 2. The instrument

`benchmarks/harness_mutation/repair_structural.py`, on top of `repair.py` (SWALLOW-4's,
frozen at `7b9a1695…`, whose `analyse`, healthy-shape comparison, text location and strict shell it
imports), on top of `action_checks.py` (`0e723694…`) and `faults.py` (`d26a407c…`); it modifies
none of them. `structural_tree` runs SWALLOW-4's first stage and tries the structural candidates
only on what it leaves unverified, so a checkout's reading is: the small repairs first, the
structural ones for the residue.

**Targets.** The 18 targets of `swallow4_receipt.json.gz` whose `verified_repair` is null, on the
same clones at the same HEADs: 14 hand-written (12 SWALLOWED by the shell, 1 SWALLOWED rejected
by the twin condition, 1 FAIL_OPEN fail-closed by design) and 4 generated.

**What was known at the freeze.** The instrument was exercised on a synthetic fixture with every
outcome (verified by each repair, rejected by the twin condition, no candidate applies) and on
this repository (no target). The residue's *script heads* were read in SWALLOW-4's receipt and
RESULT — they are public — and the predictions are calibrated on those heads, not on the scripts,
which were not opened. No structural repair has been applied to any population workflow.

## 3. Predictions, committed now

Hand-written targets unless stated; a target whose baseline verdict here differs from the
receipt's is excluded and named (G-S5-2).

**P1 — half the residue.** At least **7 of the 14** hand-written targets are verified.

**P2 — the guard is the commoner shape.** `guard-status` verifies at least as many hand-written
targets as `no-default` does.

**P3 — named, six.** Declared non-blind (SWALLOW-4 §3 read them):
- `hmislk/hmis` `development_pr_validation.yml` › `validate-jdbc-data-sources` ›
  `Validate grantAllPrivilegesToAllUsersForTesting`: **verified**, `guard-status`;
- `dotnet/roslyn` `pr-validation.yml` › `validate-and-trigger` › `Determine validation type and
  pipeline ID`: **verified**, `guard-status`;
- `selfxyz/self` `mobile-deploy.yml` › `build-ios` › `Verify branch and commit (iOS)`: **verified**,
  `no-default`;
- `langfuse/langfuse` `pipeline.yml` › `tests-shared` › `run SQL-equivalence tests
  (non-blocking)`: **verified**, `no-default` (the `|| echo` on its continuation line);
- `antiwork/gumroad` `ci-green.yml` › `report` › `Re-check migration versions against current
  main`: **not verified** — still fail-closed by design;
- `dotnet/aspire` `reproduce-flaky-tests.yml` › `reproduce` › `Run test iterations
  (Linux/macOS)`: **not verified** — any candidate that applies carries the strict shell, and the
  strict shell breaks that healthy run (SWALLOW-4 P2); or no candidate applies.

**P4 — the loops stay.** `mlflow`'s `Run tests`, `nodetool`'s `Run mutation testing` and
`aspire`'s three `Check if any test requires …` are **not verified**: no candidate applies, or
none is loud. Scored as one: all five.

**P5 — small.** Every verified hand-written repair changes at most **8** lines; the median at
most **5**.

**P6 — the twin condition is exercised.** At least **1** hand-written target has a candidate that
applies and is rejected for changing a healthy run.

**P7 — the generated residue.** At most **2 of 4** generated targets are verified (two of them
already sit under `set -euo pipefail` and swallow structurally; one's `continue-on-error` shares
its line and the first stage could not edit it; one is not loud under the strict shell).

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S5-1 (instrument) | tests green; fixture deterministic; this repository has no target | `tests/test_harness_repair_structural.py` green |
| G-S5-2 (baseline) | the targets are SWALLOW-4's residue, re-read on this machine | 18 targets; every baseline verdict equals the SWALLOW-3 receipt's, or is named and excluded; excluded ≤ 2 |
| G-S5-3 (both halves) | verified ⇒ loud in every interpretable flavour and unchanged in both | no verified candidate fails a half |
| G-S5-4 (no hand labels) | the three stated repairs only; the first stage untouched | no candidate outside `REPAIRS`; the receipt names `repair.py` at `7b9a1695…` |
| G-S5-5 (frozen underneath) | `faults.py`, `action_checks.py`, `repair.py`, source receipt | `d26a407c…`, `0e723694…`, `7b9a1695…`, `swallow4_receipt.json.gz` file sha256 recorded and equal |
| G-S5-6 (ledger) | every prediction scored | HIT/MISS for P1–P7 (P3 as six) |

G-S5-1 to G-S5-5 are blocking.

## 5. What would abandon this

G-S5-3, as in SWALLOW-4. G-S5-2 with more than 2 excluded: the residue is 14 files; if the
instrument cannot re-read three of them, this cycle has no population.

## 6. Honest statement of what a passing SWALLOW-5 means

That for the checks a strict shell cannot make loud — the guard whose failing tool is its green
path, the query whose default hides its failure — there is a stated, mechanical edit to the
script that makes the same tool failure loud and leaves a healthy run untouched under the model,
and that the instrument can tell that edit from one that breaks a healthy run. It does not reach
a loop that counts, and does not claim to. It does not say the authors want the check to be
blocking. The guard's threshold — status above 1 is a failure — is a convention of `grep` and
`diff`, stated, and a tool whose failure exits 1 is not caught by it.

## 7. Running it

```
python -m benchmarks.harness_mutation.repair_structural --receipt papers/harness/swallow4_receipt.json.gz --clones <clones> --out papers/harness/swallow5_receipt.json
python papers/harness/swallow5_score.py
```
