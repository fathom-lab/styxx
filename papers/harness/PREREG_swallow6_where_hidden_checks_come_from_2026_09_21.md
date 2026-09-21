# PREREG — SWALLOW-6: where hidden checks come from — the natural history of a check that hides its failure

Fathom Lab · 2026-09-21 · Frozen before any repository's history is read beyond the three named in §2.
Follows `RESULT_swallow5_the_structural_repairs_2026_09_21.md` (VALID, 9/12).

## Where this came from, stated before anything is measured

SWALLOW-2 to SWALLOW-5 read 100 repositories at one moment: 7,916 hand-written faults, of which 53
checks in 25 repositories hide their own failure (49 SWALLOWED, 4 FAIL_OPEN), and for 46 of the 53
a verified edit makes the same failure loud. What none of them can say is how a hidden check gets
that way. Was the `|| true` written with the step, or added the day the check turned flaky? Does
anyone take it out again? How long has it been there? This cycle reads the same checks through
time: every revision of every hand-written workflow on the default branch's mainline, from the
GitHub Actions YAML era's start to the pinned HEAD, with the frozen reading, and follows every
check as a lineage.

## 1. The instrument

`benchmarks/harness_mutation/history.py`, frozen at `d92b1b32…`, on top of `repair.py`
(`7b9a1695…`), `repair_structural.py` (`77067a71…`), `action_checks.py` (`0e723694…`) and
`faults.py` (`d26a407c…`); it modifies none of them. The reading of a workflow revision is
`repair.analyse` — SWALLOW-3's, unchanged; one memoised runner serves every revision of a
workflow. Stated rules:

- **History.** A blobless clone of the default branch since `2019-08-01`, the pinned HEAD of the
  SWALLOW-3 receipt fetched and walked `--first-parent`, oldest to newest. Only mainline commits
  that touch `.github/workflows/` are revisions; the workflow's text at each is read. Generated
  workflows (`*.lock.yml`) are excluded: their history is a compiler's. A lineage born at the
  clone's oldest commit is marked `at_boundary` (left-censored).
- **Lineage.** A job id and a step key: the step's `name`, else its `id`, else the first line of
  its script. A step whose key changes while its script does not is followed under the new key
  (a rename). A job renamed ends its lineages and starts new ones — stated, not corrected.
- **State at a revision.** From the reading's `verdict`: `hidden` (SWALLOWED, FAIL_OPEN), `loud`
  (RED), `other` (not a fault site, NO_CHECK, ABSORBED), `unread` (BASELINE_RED,
  BASELINE_SKIPPED: the state is carried over and the revision counted). The first interpretable
  reading of a lineage is its birth state.
- **Events.** `acquisition`: loud → hidden between two consecutive readable revisions;
  `repair`: hidden → loud; other transitions into or out of `hidden`/`loud` are recorded by name
  and are neither. `death`: the step or its workflow is removed. Each acquisition and repair
  carries the commit (sha, time, subject), its **mechanism** — read from the step's YAML before
  and after, in this order and taking the first that applies for the primary:
  `continue-on-error` (on the step or its job), `or-true` (`|| true` / `|| :`), `set+e`,
  `default` (`|| echo`), `strict-shell` (`set -e` / `pipefail`), `gating` (the step's `if`, the
  job's `if`, `needs` or `strategy`), `rewrite` (the script changed otherwise), `context` (the
  step did not change; something else in the workflow did) — and an **acknowledgement**: the
  first match of the stated regex (`flak`, `temporar`, `for now`, `unblock`, `non-blocking`,
  `ignor`, `skip`, `allow-fail`, `workaround`, `don't fail`, `not fail`, `soft-fail`,
  `best-effort`, `optional`, `noisy`, `unstable`, `intermittent`, `broken`, `disable`, `silence`,
  `quiet`, `warn`; word-bounded where stated in the source) in the commit's subject, body, or —
  for a merge commit — the subjects of the commits it merged.
- **Agreement.** For every repair, SWALLOW-4's `try_repairs` and then SWALLOW-5's
  `try_structural` are run on the revision before, and the first verified candidate **agrees**
  with the author when the author's mechanism is one that candidate edits: `no-continue-on-error`
  ↔ `continue-on-error`; `strict-shell` ↔ `or-true`, `set+e`, `strict-shell`; `no-default` ↔
  `default`, `strict-shell`; `guard-status` ↔ `rewrite`; the `both` forms, their unions.
- **Age** of a hidden check alive at HEAD: HEAD's commit time minus the time of its last arrival
  in `hidden` (its birth, if born hidden).

## 2. What was known at the freeze

The instrument was exercised on a scripted seven-commit history holding every event (born hidden,
born loud, an acquisition by `continue-on-error` in an acknowledging commit, a wild repair by
removing `|| true` that the instrument agrees with, a rename, a step removed hidden, a workflow
removed) and on this repository. **Three population repositories were read while the population
driver was being tested — `langfuse/langfuse`, `hmislk/hmis`, `carverauto/serviceradar` — and
what they showed is stated: each has one hidden check at HEAD, all three born hidden, aged 24,
555 and 33 days; no acquisition, no repair.** They stay in the population; the predictions below
were set after seeing them, and that is 3 of the 53. Nothing else has been read.

## 3. Predictions, committed now

Hand-written lineages of repositories read to their pinned HEAD without a cap (G-S6-2).

**P1 — born hidden.** At least **60%** of the hidden checks alive at HEAD were hidden at birth
and never loud (no repair event in the lineage).

**P2 — the acquisition is acknowledged.** Among acquisitions, at least **50%** carry an
acknowledgement in the mainline commit's message or the subjects it merged.

**P3 — repair is rare.** Among lineages that were ever hidden (born hidden, or arrived in
`hidden`), at most **25%** are alive and loud at HEAD with a repair event.

**P4 — the swallow is not transient.** The median age of the hidden checks alive at HEAD is at
least **180 days**.

**P5 — the two lines.** `continue-on-error` and `or-true` together are the primary mechanism of
at least **50%** of acquisitions.

**P6 — the instrument would have proposed what the author did.** Among repairs whose
before-revision the instrument could read (an agreement record without error), at least **40%**
have a verified candidate that agrees with the author's mechanism.

**P7 — hiding outnumbers repairing.** Acquisitions strictly outnumber repairs.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S6-1 (instrument) | tests green; every event on the scripted history; deterministic | `tests/test_harness_history.py` green |
| G-S6-2 (population) | the history reaches the pinned HEAD, uncapped, for enough of the population | ≥ 90 of the 100 repositories read to their pinned HEAD without a cap; capped or failed ones named and excluded |
| G-S6-3 (HEAD agrees) | the mainline's last revision of each workflow is what SWALLOW-3 read | of the hand-written faults of the SWALLOW-3 receipt matched by (repo, workflow, job, index) at HEAD, ≥ 99% carry the same verdict |
| G-S6-4 (frozen underneath) | the instrument and everything under it | `d92b1b32…`, `7b9a1695…`, `77067a71…`, `0e723694…`, `d26a407c…`; the SWALLOW-3 receipt file at `609e6645…` |
| G-S6-5 (ledger) | every prediction scored | HIT/MISS for P1–P7 |

G-S6-1 to G-S6-4 are blocking.

## 5. What would abandon this

G-S6-2: if fewer than 90 repositories can be read to their pinned HEAD, the history is not the
population's. G-S6-3: if HEAD's reading disagrees with the receipt beyond the nondeterminism
SWALLOW-3 measured (8 of 30,642), the walk is reading something other than the workflows.

## 6. Honest statement of what a passing SWALLOW-6 means

That under the frozen reading, on the mainline of these repositories, hidden checks are mostly
written hidden, are rarely made loud again, and stay; that when a check is hidden after the fact
the commit usually says why in one of the stated words; and that when an author does make a check
loud, the edit is the one the instrument proposes. It does not say what a step did in any real
run. It does not read branches that were never merged, or the reasons a commit does not state.
The mechanism is a reading of the diff at the step; a hiding caused elsewhere in the workflow is
`context`, and that is a limit, not a finding. The acknowledgement regex is a list of words.

## 7. Running it

```
python -m benchmarks.harness_mutation.history --receipt papers/harness/swallow3_receipt.json.gz --work <clones> --workers 2 --deadline-per-repo 2700 --out papers/harness/swallow6_receipt.json
python papers/harness/swallow6_score.py
```
