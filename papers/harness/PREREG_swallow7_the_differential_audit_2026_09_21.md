# PREREG — SWALLOW-7: the differential audit — the check that arrives hidden, caught at the commit that brings it

Fathom Lab · 2026-09-21 · Frozen before the gate is run at any commit of the population.
Follows `RESULT_swallow6_where_hidden_checks_come_from_2026_09_21.md` (VALID, 5/7).

## Where this came from, stated before anything is measured

SWALLOW-6 read 96 repositories through time and found that 40 of the 53 checks hiding their
failure today were written hidden, that of 142 ever hidden only 5 are loud today, and that the
median one is 155 days old. The place to catch a hidden check is the change that brings it. This
cycle is that gate: given a base and a head, read only the workflows that changed, match every
step across the two revisions, and say what the head hides that the base did not — with the
repair SWALLOW-4 and SWALLOW-5 verify on the head's text. Then it runs that gate at every mainline
commit of the population, base its first parent, and asks what a pull-request check would have
said, how often, how actionably, and how fast.

## 1. The instrument

`benchmarks/harness_mutation/differential.py`, frozen at `91e4a4a7…`, on top of `history.py`
(SWALLOW-6's, `4b961880…`, whose reader, step key, mechanism reading, mainline walk and text
fetch it imports), `repair.py` (`7b9a1695…`), `repair_structural.py` (`77067a71…`),
`action_checks.py` (`0e723694…`) and `faults.py` (`d26a407c…`); it modifies none of them.

- **The gate.** `git diff --name-status -M base head -- .github/workflows`, hand-written
  workflows only. Each changed workflow is read at base and at head with the frozen reading.
  Steps are matched by (job, key) — the step's name, else id, else first line — and a step whose
  key changed while its script did not is matched by its script (a rename). A step's state at a
  revision is `hidden` (SWALLOWED, FAIL_OPEN), `loud` (RED), `other` (not a fault site, NO_CHECK,
  ABSORBED) or `unread` (BASELINE_RED, BASELINE_SKIPPED).
- **Firing.** A check that is `hidden` at head and, at base, `loud`, `other`, or absent is
  **newly hidden**: `born hidden` (absent before), `acquired` (loud before, with the mechanism of
  SWALLOW-6), or `became a check, hidden` (other before). The gate **fires** when at least one
  check is newly hidden. Hidden at head and `unread` at base is reported as `hidden after unread`
  and does not fire. Hidden at base and loud, other, or absent at head is **removed hidden**
  (`repaired`, `no longer a check`, `removed`). Hidden at both is `still hidden`.
- **The fix.** For every newly hidden check, SWALLOW-4's `try_repairs` then SWALLOW-5's
  `try_structural` on the head's text; the first verified candidate and its diff are recorded.
- **The population reading.** For every repository the SWALLOW-6 receipt read, at its pinned
  HEAD, every first-parent commit since 2019-08-01 that touches a hand-written workflow, base its
  first parent. A commit with no parent in the fetched history (a root, or the clone's boundary)
  reads every workflow as born; it is recorded `root` and excluded from every count below. The
  reading is memoised per workflow across commits. Separately, every 100th such commit is read
  again with nothing memoised and timed: the **sample**, which is what a pull-request check would
  cost.

## 2. What was known at the freeze

The gate was exercised on the scripted seven-commit history of SWALLOW-6's tests (every outcome:
two checks born hidden at the root with their repairs, an acquisition by `continue-on-error`, a
wild repair, a rename, a removal, a workflow added loud and deleted) and on this repository
against its own commits (nothing newly hidden; SWALLOW-6 read this repository's mainline as never
having hidden a check). **The SWALLOW-6 receipt has been read in full**, and the gate at every
commit is expected to reproduce its arrivals (that is G-S7-3), so P1, P4 and P6 below are
calibrated on it — they are stated as checks that the gate's view of the population is the
receipt's, not as blind predictions. **P2, P3, P5 and P7 are blind**: nothing has read a repair
at a birth revision, timed the gate on the population, or counted `continue-on-error` at birth.

## 3. Predictions, committed now

Non-root commits of repositories read uncapped (G-S7-2); "newly hidden check" means one record
of a firing.

**P1 — a quiet gate** (calibrated). The gate fires on at most **1.0%** of the mainline commits
that touch a hand-written workflow.

**P2 — actionable at the commit** (blind). At least **70%** of newly hidden checks have a
verified repair on that revision's text.

**P3 — small at the commit** (blind). The median `lines_changed` of those verified repairs is at
most **2**.

**P4 — born, not turned** (calibrated). At least **70%** of newly hidden checks are `born hidden`.

**P5 — fast enough for a pull request** (blind). On the sample, the median wall time of the gate
is at most **3 s** and the 90th percentile at most **15 s**.

**P6 — they arrive in batches** (calibrated). At least **4** firing commits bring three or more
newly hidden checks at once.

**P7 — the line is `continue-on-error`** (blind). At least **50%** of newly hidden checks carry
`continue-on-error` (on the step or its job) at the firing revision.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S7-1 (instrument) | tests green; every outcome on the scripted history; deterministic | `tests/test_harness_differential.py` green |
| G-S7-2 (population) | the gate ran at every commit of enough of the population | ≥ 90 repositories read to their pinned HEAD without a cap |
| G-S7-3 (the gate is the history) | the gate at every commit reproduces SWALLOW-6's arrivals | (a) ≥ 99% of the receipt's arrivals in `hidden` (a lineage born hidden, or an event `to: hidden`) at non-boundary commits are firings at the same (repo, workflow, job, key, sha); (b) the gate fires at ≤ 0.5% of non-root commits where the receipt records no arrival |
| G-S7-4 (frozen underneath) | the instrument and everything under it | `91e4a4a7…`, `4b961880…`, `7b9a1695…`, `77067a71…`, `0e723694…`, `d26a407c…`; the SWALLOW-6 receipt file at `703583d3…` |
| G-S7-5 (ledger) | every prediction scored | HIT/MISS for P1–P7 |

G-S7-1 to G-S7-4 are blocking.

## 5. What would abandon this

G-S7-3: a gate that does not see what the history saw is reading something else, and its
firing counts mean nothing. G-S7-2 as in SWALLOW-6.

## 6. Honest statement of what a passing SWALLOW-7 means

That a pull-request check built on the frozen reading would have been quiet, would have fired at
the commits that brought a hidden check, would usually have had a one- or two-line verified
repair to propose, and would have cost seconds. It does not say the authors would have accepted
the repair, or that a hidden check is wrong to write. It reads the first-parent history, so a
change squashed into a merge is read at the merge; it does not read the pull requests
themselves. A repository whose history begins at the clone's boundary is read from there. The
sample's timing is this machine's.

## 7. Running it

```
python -m benchmarks.harness_mutation.differential --receipt papers/harness/swallow6_receipt.json.gz --work <clones> --workers 2 --deadline-per-repo 2700 --out papers/harness/swallow7_receipt.json
python papers/harness/swallow7_score.py
```
