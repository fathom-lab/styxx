# PREREG — SWALLOW-14: the empty list — two edits for the checks SWALLOW-13's stage left, and a hoist that stays local, run on 5,945 repositories none of it was designed on

Fathom Lab · 2026-09-22 · Frozen before any repository of the population is fetched.
Follows `RESULT_swallow13_the_frontier_2026_09_22.md` (VALID, 5/8), whose "next" named this: "The
empty loop: ten checks here, and an edit a strict shell and the model can both verify."

## Where this came from, stated before anything is measured

SWALLOW-13's third repair stage verified a repair for 13 of the 44 hidden checks the first two
stages left on 549 repositories. Read after scoring, 18 of the 31 it left pass on an empty list or
an empty answer, and 10 of those are a loop fed by a process substitution — `done < <(find …)`,
`mapfile -t a < <(…)` — whose command, when it fails, leaves an empty list the loop reads as
nothing to check. Bash drops a process substitution's status: neither `set -e` nor pipefail sees
it. And every repair SWALLOW-13 verified by hoisting made the whole script strict — `set -eo
pipefail` for every line, every `|| true` elsewhere taken away — which the model's stubs cannot
fault but a real run can (a `grep`'s "no match", a `| head` that closes its pipe early).

This cycle adds two edits to the stage (`styxx/ciaudit/repair_frontier.py`), designed on
SWALLOW-13's 44 (the **development set**; every repository of it is excluded below):

- **wait-list** — after a statement that reads its list from a process substitution, the line
  `[ -z "$!" ] || wait $! || exit $?` (with a comment saying what it is for): from bash 4.4 `$!` is
  the process substitution and `wait` returns its status; before it, `$!` is empty and the line does
  nothing. Pipefail inside the substitution when it is a pipeline, for that command only; a
  command whose status 1 is an answer (grep and its kin, diff, `jq -e`, …) keeps its 1. Not applied
  when a background job, or another process substitution inside the loop, would take `$!`, or when
  the substitution is followed by more than a comment.
- **hoist-local** — SWALLOW-13's hoist with the status kept on its own line —
  `__subN="$(…)" || exit $?`, pipefail inside the substitution when it is a pipeline — and nothing
  else in the script changed.

and their no-coe+ forms. The stage now tries, in order: hoist-local, wait-list, then SWALLOW-13's
four edits; then the no-coe+ forms of no-default, guard-status and the six. `repair_frontier.S13_REPAIRS`
keeps the stage as SWALLOW-13 ran it, for replaying its receipt and its tests.

**On the development set** (exploratory, designed on it): wait-list applies to 10 of the 31 checks
SWALLOW-13's stage left and verifies 9 of them (cloudposse/atmos ×4, dotCMS ×2, Azure, port-labs,
zeroc-ice; the tenth ends its list's command with `|| true`); hoist-local verifies all 12 checks the
global hoist verifies there, and no other.

## 1. The instrument

`benchmarks/harness_mutation/empty_list.py`, frozen at
`9b40257a3b718f02bd8e729bb29e54df706bf7ec0d84809669d826e7490e6b5c`, calling the product as it
ships, unchanged:

| file | sha256 |
|---|---|
| `styxx/ciaudit/engine.py` | `5c219887e389ed671d082c18be220af4479d48d0bd152fb348997ba4f5368b24` |
| `styxx/ciaudit/actions.py` | `ecbcd4f94a050e5b983f91464e681f61f2504a5447c45c68f91e419459c8d42a` |
| `styxx/ciaudit/repair.py` | `4282dbfa4d5caf22bac274e0964603dc451db164df8c730ba94afee58189241a` |
| `styxx/ciaudit/repair_structural.py` | `af46c492656e4d1837c97e0ecb3f40088e782c04c306ac94f70dc4bde7846570` |
| `styxx/ciaudit/repair_frontier.py` | `c10b9975f18a73a02e0250f92207294d0730bdbf2daeec526b0c01d07d58a8c1` |

**Population** — `papers/harness/swallow14_population.json.gz`
(`21c0e00fdaad3296f641668a48c30d3d33600cdacaf2faed691b36e071e5d58b`): the AIDev dataset's
`repository` table (`repository.parquet`, sha256 `a08e34be…`: 6,673 repositories with more than
100 stars that received agents' pull requests), minus every repository an earlier cycle read —
SWALLOW-1's hundred and SWALLOW-9's 635, which hold SWALLOW-10's, -11's and -13's (676 of the table)
— each at the default-branch tip `git ls-remote HEAD` gave at the freeze: **5,945 repositories**
(52 more could not be read and are listed, not measured). No workflow of any of them has been read.

**Per repository**, as SWALLOW-13's instrument: `.github/workflows` at the tip (sparse, blob-less,
depth 1, one retry); the product's audit (actions counted, 600 s deadline); every hand-written
finding through `repair.repair_faults` — stage 1, 2, then the stage's fourteen candidates, all
tried and recorded — and the readings; stage 3 once more with a fresh runner. Each repository in its
own process, 30 minutes at most, three at a time; the clone is removed after it is read.

**Units.** As SWALLOW-13: a **target** is a hand-written finding, **read** when its verdict
reproduces in the repair phase; the **stage-3 population** is the read targets stages 1 and 2 do
not verify. **SWALLOW-13's stage** verifies a check when any of its ten candidates (`S13_REPAIRS`)
does; the **S13 residue** is the stage-3 population it does not. The **new edits** are wait-list,
hoist-local and their no-coe+ forms. The **wait class** is the S13-residue checks where wait-list
or no-coe+wait-list applies. The **global hoist** is hoist-substitution and its no-coe+ form; the
**local hoist** is hoist-local and its no-coe+ form.

## 2. What was known at the freeze

- SWALLOW-13's receipt and reading, and the development set's numbers above.
- The mechanics were exercised on the scripted fixtures of `tests/test_ciaudit_frontier.py` and
  `tests/test_harness_empty_list.py` (wait-list verified on a `find` loop and a `mapfile`; the local
  hoist verified where the global one takes a `|| true` a healthy run needs).
- The population's size and the 52 it could not read. Nothing has been computed on any of its
  workflows.

## 3. Predictions, committed now

**P1 — the new edits reach** (blind). They verify a repair for at least **20%** of the S13 residue.

**P2 — the empty list** (blind). wait-list (or its no-coe+ form) verifies at least **50%** of the
wait class. Fewer than 10 checks in the wait class is a miss.

**P3 — local loses nothing** (blind). Every check the global hoist verifies, the local hoist
verifies too.

**P4 — local gains** (a coin). The local hoist verifies at least one check the global hoist does
not.

**P5 — not one repository** (blind). wait-list verifies a repair in at least **3** repositories.

**P6 — SWALLOW-13 replicates** (blind). SWALLOW-13's stage verifies at least **25%** of the stage-3
population, on repositories it was not designed on either.

**P7 — the stages together** (blind). Stages 1, 2 and 3 as the product now runs them verify at
least **80%** of the read targets.

**P8 — small** (blind). The median wait-list repair the product chooses changes at most **3**
lines of the workflow.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S14-1 instrument | the edits on text and verified end to end; the instrument on two checkouts; SWALLOW-13's instrument with its stage; SWALLOW-12's plan with its two | `tests/test_ciaudit_frontier.py`, `tests/test_harness_empty_list.py`, `tests/test_harness_frontier.py`, `tests/test_ciaudit.py`, `tests/test_harness_live_click.py` green |
| G-S14-2 frozen underneath | the instrument, the five product files, the population | the hashes above, recorded by the receipt |
| G-S14-3 the population is read | fetched at the tip and audited without error or timeout | ≥ 90% of 5,945 |
| G-S14-4 enough to read | the S13 residue | ≥ 30 checks |
| G-S14-5 determinism | stage 3 run twice | identical candidates, verdicts, diffs and verified repair on every check it was tried on |
| G-S14-6 ledger | every prediction scored | HIT/MISS for P1–P8 |

G-S14-1 to G-S14-5 are blocking.

## 5. What would abandon this

G-S14-5: a verdict that moves between two runs cannot ship as verified. G-S14-4: fewer than thirty
checks cannot carry a share.

## 6. Honest statement of what a passing SWALLOW-14 means

That on 5,945 repositories none of the stage was designed on, the two edits verify — loud under the
same fault, the healthy run unchanged, read by the same model — a repair for the stated share of
the hidden checks SWALLOW-13's stage leaves, and that keeping a hoist's strictness on its own line
loses no check the whole-script form verifies. It does not say a verified repair is right: RED is
loud, not correct, and the model's healthy run is stubs, which never fail. wait-list makes a list
command's failure stop the step; a list command that fails in a real healthy run — a directory the
author meant to be optional — now fails it too, which is the point unless the author meant it.
`wait $!` needs bash 4.4 to see the substitution; before, the line does nothing. The population is
a public dataset's popular repositories, each read at one commit.

## 7. Running it

```
python -m benchmarks.harness_mutation.empty_list --build-population --parquet repository.parquet --out papers/harness/swallow14_population.json.gz
python -m benchmarks.harness_mutation.empty_list --population papers/harness/swallow14_population.json.gz --work <dir> --out papers/harness/swallow14_receipt.json.gz --workers 3
python papers/harness/swallow14_score.py
```
