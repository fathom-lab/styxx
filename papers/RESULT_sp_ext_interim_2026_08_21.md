# INTERIM — SP-EXT: 7 of 17 adjudicated, 1 confirmed, run incomplete

**This run was cut off by a session limit at 27 of 52 adjudication agents. It is
NOT a corpus, NOT a finished result, and no gate has been applied. It is
published in this state because a partial run reported as a whole one is the
defect class this benchmark exists to catalogue.**

Prereg: `PREREG_sp_external_corpus_2026_08_21.md`, frozen at `38b8428` before any
commit was read. Raw: `out_sp_ext_candidates.json`, `out_sp_ext_shaped.json`,
`out_sp_ext_partial.json`.

---

## the first externally-anchored silent-pass case

**`Giskard-AI/giskard` — `dd75e974ee`, "fix(checks)!: return None from
SuiteResult.pass_rate when nothing evaluated (#2753)"**

Accepted **unanimously**: three reviewers, each assigned a different lens and each
instructed to reject it, all three failed to.

```python
denominator = len(self.results) - self.skipped_count
if denominator == 0:
    return 1.0          # a suite that evaluated NOTHING reports 100% pass
```

The pre-fix docstring — read directly from git, not from an agent — is the part
that matters most:

> `pass_rate : float` — *Fraction of non-skipped scenarios that passed*
> **(1.0 when all scenarios are skipped).**

**The flattering default was documented as the intended contract.** A consumer
reading 1.0 from an empty suite was reading exactly what the API promised. This is
the defect class at its most complete: not an oversight in an edge case, but a
value that never measured anything, specified in writing as a perfect score.

An empty suite, or one where every scenario was skipped or the producing probe
crashed, reported that perfect score.

Consumers, traced independently by all three reviewers:
- the Rich console summary line — `Summary: … | Pass Rate: 100.0%`
- **the Giskard Hub upload payload** via `to_hub_format` / `model_dump`, which
  shipped `1.0` as a genuine `success_rate`
- the project's own README example, `print(f"Aggregated pass rate: {results.pass_rate * 100}%")`
- user CI gates thresholding the field

Subtype **SP-2 SENTINEL_DEFAULT**, unanimous. Fixed by Giskard's own maintainers
as a breaking change — the `!` in `fix(checks)!`.

Verified independently: the fix's own diff touches the README example (changed to
handle `None`), `tests/export/test_hub.py` (the Hub payload carried the value), and
the garak scan adapter — **three distinct consumers, confirmed by the fix itself**
rather than inferred.

One correction to this record before it goes further: an earlier draft named the
module `giskard/core/suite.py`, which does not exist. It is
`libs/giskard-checks/src/giskard/checks/core/result.py`. Found by reading git
rather than trusting the summary, which is the only reason it is not now sitting
in a benchmark other people would rely on.

**Ground truth here is not our judgment.** Somebody else read that code, decided a
100% pass rate from zero evaluations was wrong, and shipped a breaking change to
fix it. That is the whole point of anchoring to external fix commits, and it is
the first time this project has had one.

## where the run actually stands

| | |
|---|---:|
| repositories cloned | 14 |
| commits searched | 61,702 |
| Q1 (message regex) candidates | 415 |
| Q1 ∩ Q2-shape | **17** (4.1% of Q1) |
| fully adjudicated (3/3 lenses) | **7** |
| accepted | **1** |
| rejected | 6, all 3/3 unanimous |
| **not adjudicated — session limit** | **10** |

**No gate has been applied and none may be.** G1 (yield), G2 (accept rate), G3
(spread) all operate on a completed adjudication. The accept rate so far is 1 of
7, but n=7 is far too small to trigger G2's 20% rule, and doing so would be
reading a gate off an incomplete cell — the exact move
`RESULT_flattering_external_2026_08_21.md` was written to condemn.

The run is resumable: the 27 completed agents are cached by run id, so finishing
costs only the 25 that failed.

## what already failed, and is recorded now

**Q2 could not be run standalone.** The preregistration promised both queries over
all 61,702 commits with yields reported separately. Q2's efficient implementation
is `git log -S` (pickaxe), and against a `--filter=blob:none` clone it **timed out
after 10 minutes on the smallest repository in the set** (whylogs, 936 commits),
because every candidate blob is fetched lazily over the network. Full-blob clones
of 14 repositories would fix it and were not done.

So the 17 candidates come from **Q1 ∩ Q2-shape**, a subset of a frozen query, not
from Q2 alone. Q2 alone would almost certainly yield more — most such fixes are
not *described* the way Q1 requires — and every case it would have found is
missing from this run. **This makes the present yield a lower bound on a lower
bound**, and it is a limitation of the harness, not a finding about the world.

## the rejections are the informative half

Six candidates were rejected 3/3, and the reasons say what the harvest pulls in
that is not the defect class:

- **R1 failures — no absent-measurement path.** `cleanlab 6ec5b173dd` is a typing
  and docstring cleanup whose one behavioural change is a *cascading
  `str.replace`* correction: the measurement fully happened and produced a wrong
  string. **Wrong-but-measured is a different defect**, and the exclusion list
  says so.
- **Determinate values misread as absent data.** A `return_mask: bool = True`
  parameter is a caller-supplied flag with a literal default, not runtime data
  that could be missing — the same C2 confusion that dominated the `flattering`
  run, arriving now through a different door.
- **EXCLUDED — the matched hunk was incidental.** In several candidates the line
  the shape filter matched belonged to a large unrelated refactor. The reviewers
  were explicitly told to read the whole diff rather than the matched hunk, and
  that instruction did work.

## what is NOT claimed

- **No prevalence claim.** One confirmed case is one confirmed case. Nothing here
  says how common this is, and G5 already forbids quoting SP-EXT as a rate under
  any circumstances.
- **No claim about the 10 unadjudicated candidates**, in either direction. Several
  read like plausible cases and several read like refactors; neither impression is
  evidence, and recording an impression as a result is how a corpus gets poisoned.
- **No claim that the queries are good or bad.** 17 from 415 is a number about the
  shape filter, and with Q2 standalone unrun, the harvest has not actually been
  tested.

## what this does establish

That the defect class exists outside this repository, in a widely used ML testing
library, in the field's own vocabulary — a **pass rate of 100% from zero
evaluations**, uploaded to a hosted dashboard as a real success rate. Prior to
today every silent-pass case this project could point at was one of its own.

One case is not a corpus. It is the first entry in one.

## to finish

Resume the adjudication (25 agents), then apply G1/G2/G3 to the complete set. If
Q2 standalone is wanted, re-clone with full blobs first, under a new
preregistration disclosing the change — the current one specified a query that
this harness could not execute, which is itself worth remembering the next time a
prereg names a method without first checking it runs.
