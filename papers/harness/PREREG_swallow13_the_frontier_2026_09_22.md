# PREREG — SWALLOW-13: the frontier — a third repair stage, designed on the 49 hidden checks no repair reached, run on 549 repositories it was not designed on

Fathom Lab · 2026-09-22 · Frozen before any repository of the population is fetched.
Follows `RESULT_swallow12_the_click_live_2026_09_22.md` (VALID, 8/8), whose "next" named this:
"The 49 checks of SWALLOW-11 with no verified repair are the repair families' frontier."

## Where this came from, stated before anything is measured

`styxx ci-audit --repair`, the pull request's gate and the Action verify a repair in two stages —
SWALLOW-4's (remove the step's `continue-on-error`; make its shell strict) and SWALLOW-5's (a guard
whose failing tool is not its green path; a query without its `|| echo` default) — and a repair
counts only when it is loud under the same fault and leaves the healthy run unchanged in both
flavours. Of the 220 hidden checks SWALLOW-11 replayed, 49 got no verified repair. This cycle read
those 49 (the **development set**: 17 repositories, all inside SWALLOW-11's sixty) and wrote a third
stage for them, `styxx/ciaudit/repair_frontier.py`:

- **hoist-substitution** — a `$(...)` whose status its line throws away (inside an `echo`/`printf`
  argument, a `for ... in` list, an `export`/`local`/`readonly`/`declare` assignment, a one-test
  `if [ ... ]; then`), on a line that is one simple command, moved onto its own line as
  `__subN="$(...)"`, the shell made strict.
- **background-liveness** — after a command started with `&`: `__bgN=$!`, `sleep 5`, and
  `if ! kill -0 "$__bgN" 2>/dev/null; then wait "$__bgN" || exit $?; fi` — a job that has already
  died failing fails the step; one still running, or that exited cleanly, does not.
- **no-exit-zero** — `|| exit 0` removed, the shell made strict.
- **no-default-joined** — SWALLOW-5's no-default over the script with its backslash-continued lines
  joined, where the `|| echo …` fallback begins a continuation line of its own command.
- **no-coe+<edit>** — the `continue-on-error` removed together with a script edit, where neither
  alone is loud: SWALLOW-5's no-default or guard-status, or one of the four above.

And two **readings** for a check no stage repairs, rules on the script's text, printed next to the
finding, never a verdict: **routed** — a flag set on the failure path (a variable set to `$?`, or to
two of 0/1 or true/false, or a counter counted up) is written to GITHUB_ENV or GITHUB_OUTPUT under a
name a later step reads (the reader "can fail the job" when its script has an `exit` 1–9);
**declared** — the script emits a `::warning`, or says non-blocking, best effort, allowed to fail,
don't fail / do not fail / will, should or must not fail, soft fail, ignore the failure/errors, or ⚠.

**On the development set** (exploratory, designed on it; the final module): stage 3 verifies **26 of
49**, in 11 of the 17 repositories — background-liveness 10 (one repository's nine starts of a
server, and one more), hoist-substitution 9, no-coe+hoist-substitution 4, no-default-joined 1,
no-exit-zero 1, no-coe+no-default 1. Of the 23 it leaves, **16 are routed** (all one repository's
ratchet; 13 with a reader that can fail the job), none declared (three of the 26 repaired are), 7
neither. A number designed on is not a result; this preregistration is for the held-out set.

The stage ships in this cycle's product, before the run: the gate (`differential.fix_for`) tries
it after SWALLOW-5's and records the stage (`swallow-13`); the command's `--repair` path
(`repair.repair_faults`) and its card; the Action, whose one-click suggestion rebuilds a third-stage
repair by name (`repair.apply_repair` now applies any stage's repair); and, for a check none
repairs, the readings on the card, in the Action's annotation and in its job summary. SWALLOW-12's
plan is re-computed with the two stages it ran (`differential.STAGES`): the third stage's
no-exit-zero repairs the fixture SWALLOW-12 froze as the control no repair reaches, and a test says
so.

## 1. The instrument

`benchmarks/harness_mutation/frontier.py`, frozen at
`7b3c2b129750292cf058458cc975f4d2257ea6e642e932e982d5513a4ccbd9d1`, calling the product as it
ships, unchanged:

| file | sha256 |
|---|---|
| `styxx/ciaudit/engine.py` | `5c219887e389ed671d082c18be220af4479d48d0bd152fb348997ba4f5368b24` |
| `styxx/ciaudit/actions.py` | `ecbcd4f94a050e5b983f91464e681f61f2504a5447c45c68f91e419459c8d42a` |
| `styxx/ciaudit/repair.py` | `4282dbfa4d5caf22bac274e0964603dc451db164df8c730ba94afee58189241a` |
| `styxx/ciaudit/repair_structural.py` | `af46c492656e4d1837c97e0ecb3f40088e782c04c306ac94f70dc4bde7846570` |
| `styxx/ciaudit/repair_frontier.py` | `0fb104f09b10f767d962073a4e6905ff075984d868280a12747e5595a418ae13` |

**Population** — `papers/harness/swallow13_population.json`
(`6f65d8f755ce00bf7770cffffd9eba507510e7a6a71f28170d42a6b2103ce46a`): SWALLOW-9's 623 repositories,
each at the default-branch tip SWALLOW-9 recorded, minus SWALLOW-1's hundred (57 of them are
SWALLOW-9's; SWALLOW-2 to -7 read them, and SWALLOW-4's and -5's stages were written on them) and
SWALLOW-11's sixty (17 more; they hold the development set): **549 repositories**. None of their
workflows has been read by this program at those tips.

**Per repository**: `.github/workflows` at the tip (sparse, blob-less, depth 1, one retry); the
product's audit (`engine.analyse_tree`, actions counted, 900 s deadline); every **hand-written**
finding (SWALLOWED, FAIL_OPEN; not a generated lock file) through `repair.repair_faults` — stage 1,
then 2, then 3, then the readings — and stage 3 once more with a fresh runner on every check it
was tried on. Each repository in its own process, one hour at most, two at a time.

**Units.** A **target** is a hand-written finding; it is **read** when its verdict reproduces in the
repair phase. The **stage-3 population** is the read targets neither stage 1 nor stage 2 verifies.
The **residue** is the stage-3 population stage 3 leaves. A **distinct script** is (repository,
workflow, sha256 of the step's `run:`, whether the step or its job is `continue-on-error`). A
**family** groups a repair with its no-coe+ form: hoist, background, exit-zero, default-joined, and
no-coe+no-default and no-coe+guard-status each alone.

## 2. What was known at the freeze

- The development set's numbers above, and SWALLOW-4's and -5's on SWALLOW-1's hundred: 53
  hand-written targets, 39 verified by stage 1, 7 of the remaining 14 by stage 2 — 7 left (13%).
- The mechanics were exercised on the scripted fixtures of `tests/test_ciaudit_frontier.py` and
  `tests/test_harness_frontier.py`, and on one repository outside the population
  (`mlflow/mlflow` at its SWALLOW-9 tip: 69 workflows, one hand-written finding, SWALLOW-5's
  residue "Run tests", which stage 3 does not repair and the readings do not match; 11 s).
- Determinism was probed on the fixture's background start: 40 of 40 runs verified alike, idle and
  under load.
- Nothing has been computed on any repository of the population.

## 3. Predictions, committed now

**P1 — the stage reaches** (blind). Stage 3 verifies a repair for at least **25%** of the stage-3
population.

**P2 — not one script, many times** (blind). Counted once per distinct script, stage 3 verifies at
least **20%** of the stage-3 population.

**P3 — not one repository** (blind). Stage-3 repairs land in at least **3** repositories.

**P4 — the hoist** (blind). The hoist family is the largest stage-3 family — strictly; a tie is a
miss.

**P5 — the three stages together** (blind). Stages 1–3 verify at least **85%** of the read targets.

**P6 — the second half bites** (blind). At least one stage-3 candidate is loud under the fault and
is rejected because it changes the healthy run.

**P7 — the readings** (a coin). Routed or declared matches at least **20%** of the residue.

**P8 — small** (blind). The median stage-3 repair changes at most **6** lines of the workflow.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S13-1 instrument | the edits, their verification end to end, the readings, the stage in the gate, the command and the Action; the instrument on one checkout; SWALLOW-12's plan with its two stages | `tests/test_ciaudit_frontier.py`, `tests/test_harness_frontier.py`, `tests/test_ciaudit.py`, `tests/test_harness_live_click.py` green |
| G-S13-2 frozen underneath | the instrument, the five product files, the population | the hashes above, recorded by the receipt |
| G-S13-3 the population is read | fetched at the tip and audited without error or timeout | ≥ 90% of the 549 |
| G-S13-4 enough to read | the stage-3 population | ≥ 15 checks; below it the result is reported, not claimed |
| G-S13-5 determinism | stage 3 run twice | identical candidates, verdicts, diffs and verified repair on every check it was tried on |
| G-S13-6 ledger | every prediction scored | HIT/MISS for P1–P8 |

G-S13-1 to G-S13-5 are blocking.

## 5. What would abandon this

G-S13-5: a stage whose verdict moves between two runs cannot ship as verified. G-S13-4: fewer
than fifteen checks cannot carry a share.

## 6. Honest statement of what a passing SWALLOW-13 means

That on the default branches of 549 repositories the third stage was not designed on, it verifies —
loud under the same fault, the healthy run unchanged, read by the same model — a repair for the
stated share of the hidden checks the first two stages leave, and the readings say what they say
about the rest. It does not say a verified repair is right: RED is loud, not correct, and the
model's healthy run is stubs. background-liveness costs five seconds of the step, and its "dead
within five seconds" is a choice, not a measurement. A reading is a rule on text; "routed" says
where the failure went, not that the destination is a gate. The population is this program's
(repositories where agents' pull requests touched workflows), read at one commit each; a finding
here is a check hidden at the tip, not one a change brought.

## 7. Running it

```
python -m benchmarks.harness_mutation.frontier --build-population --out papers/harness/swallow13_population.json
python -m benchmarks.harness_mutation.frontier --population papers/harness/swallow13_population.json --work <dir> --out papers/harness/swallow13_receipt.json.gz --workers 2
python papers/harness/swallow13_score.py
```
