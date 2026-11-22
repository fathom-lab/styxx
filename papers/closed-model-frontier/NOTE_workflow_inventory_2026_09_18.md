# NOTE — every workflow, against its own run history: one red every day, one red since May

Fathom Lab · 2026-09-18 · **Not a measurement.** An inventory of what this repository's CI
actually does, taken from the Actions API rather than from the workflow files. Instrument
unchanged at sha256 `9b620e00…` and not run. Follows
`AUDIT_cross_paper_numbers_2026_09_18.md`, which found the browser port's guard had stopped
firing; this is the same question asked of every other workflow.

## The inventory

| workflow | runs | last three |
|---|---|---|
| `test.yml` | 1033 | success, success, success |
| `audit-claims.yml` | 368 | success, success, success |
| `nightly-heavy.yml` | 91 | success, success, success |
| `diffgate.yml` | 38 | success, success, success |
| `publish.yml` | 80 | success (2026-09-01) |
| **`telescope.yml`** | **146** | **failure x138 consecutive, unbroken since 2026-05-04** |
| **`gauntlet-pr.yml`** | **78** | **failure, failure, failure — last run 2026-05-28** |
| `replications.yml` | 0 | never run |
| `leaderboard-submission.yml` | 0 | never run |

## `telescope.yml` — red every day, and not for the reason anyone would look for

**138 consecutive scheduled failures. The last green run was 2026-05-03.** The daily cognometric
layer behind `fathom.darkflobi.com/scoreboard` has produced no snapshot in four and a half
months, and the red is so routine by now that a real telescope failure would be invisible inside
it.

> **Correction, appended 2026-09-19.** The first draft of this note said "from at least
> 2026-09-13 through 2026-09-18" — six days. That came from the three runs the API returns by
> default, and it understated the streak by a factor of twenty. Paging the full run history gives
> 138 in a row since 2026-05-04. The wrong number was mine, reading one page and describing it as
> the record; it is corrected here rather than quietly overwritten, and it is a small live example
> of the thing the table below is for — a number taken from what was in front of me instead of
> from the whole of it.

The job's own steps say where it breaks:

```
4. check vendor keys        success
5. install deps             skipped
6. run telescope            skipped
7. commit daily snapshot    skipped
13. Post Run actions/setup-python@v5   FAILURE
```

Everything the workflow is *for* is skipped correctly. The failure is in setup-python's post
step:

> `Error: Cache folder path is retrieved for pip but doesn't exist on disk: /home/runner/.cache/pip.`

`cache: 'pip'` was set on `setup-python`. The install step below it is conditional on a vendor
key, so on a keyless day pip never runs, `~/.cache/pip` is never created, and the post-job cache
save fails the whole job. The timing fits: the pin on the workflow's last green run is 2026-05-03,
and the streak starts with the very next scheduled run.

What makes this worth writing down is the comment sitting four lines under the cache directive:

> No vendor keys configured is an expected CONFIGURATION state, not a failure — skip green with a
> notice instead of red-Xing the repo daily.

**The guard against a daily red X was already written, and was being defeated from a post step it
cannot reach.** A conditional `if:` cannot govern another action's cleanup. Removing `cache: 'pip'`
is the whole fix; nothing was being cached anyway, because the one install is unpinned and has no
lockfile to key on. The comment now says so, so it does not come back.

**Checked, not argued.** The repaired workflow was dispatched against its own branch before this
note was written: `workflow_dispatch` on `fathomlab-patch-31`, **conclusion success**, with the
same keyless path as every failing run —

```
4. check vendor keys                    success
5. install deps                         skipped
6. run telescope                        skipped
7. commit daily snapshot                skipped
13. Post Run actions/setup-python@v5    success   <- was failure on every run on main
```

Nothing else moved. The job still does no work on a keyless day; it now finishes green while
doing none, which is what the guard was written to make it do.

## `gauntlet-pr.yml` — last known state is failure, and the evidence has expired

`submissions/GAUNTLET.md` and the workflow's own header state the leaderboard is "trustworthy by
construction: no submission lands without CI verification." The workflow is armed, path-gated on
`submissions/**` and `LEADERBOARD.md`. Its last run was **2026-05-28 and it failed**, as did the
two before it. Nothing has triggered it since, because no external submission has arrived.

**This is not diagnosed and is deliberately not guessed at.** Action logs expire, and a run from
four months ago no longer has any; the failure reason is gone. What can be said without guessing:
the gate that the leaderboard's trustworthiness claim rests on was last observed broken, there is
no `workflow_dispatch` on it, so **there is currently no way to check whether it works except to
wait for a stranger to submit and let it break on them.**

Repairing it blind would mean rewriting discovery logic against a failure nobody can reproduce —
which is how a second silent failure gets introduced while fixing the first. What it needs is
either a manual trigger plus a run against one of the committed `baseline_00*` submissions, or a
throwaway submission PR to re-observe the failure. Either is a decision about the leaderboard,
not a cleanup, so it is recorded here rather than done.

## The two that have never run are fine

`replications.yml` triggers on a pull request touching `replications/*.json` whose title starts
`[replication]`. `leaderboard-submission.yml` is gated the same way. Zero runs means no outsider
has submitted yet — dormant by design, not broken. They are listed above so that a future reader
checking this table does not mistake them for the same defect, and so that the first time either
*does* fire, the zero in this row is the thing that dates it.

## What this does not cover

Whether a workflow that passes is *measuring* anything — `audit-claims` and `diffgate` are green
here on the strength of their exit codes and nothing more. The gate port had a green suite around
it too, right up until someone asked whether the differential inside it had ever run.
