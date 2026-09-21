# RESULT — SWALLOW-7: the differential audit — INVALID on its own join with the history, stated first; the gate fires on 0.5% of commits, misses none of the 147 arrivals once a renamed workflow is followed, proposes a one-line repair for 101 of the 147 checks it catches, and costs 0.15 s

Fathom Lab · 2026-09-21 · Scores the receipt `swallow7_receipt.json.gz` against the
preregistration frozen at sha256 `a8f4f675ced0ad319c9866eab5881bd023f22901ee28a432bd823fa20e27aba8`.
Not amended. Two runs; §1 states what the first exposed — a defect in SWALLOW-6's instrument,
not in this one — and what was rerun.

Receipt: `papers/harness/swallow7_receipt.json.gz` (sha256 of the JSON `3f13eb46…`, recorded by the
scorer; every firing carries the step, its verdict on both sides, its mechanism and its verified
repair with the diff) · instrument `benchmarks/harness_mutation/differential.py`, sha256
`91e4a4a7…`, on top of `history.py` (`93efb4a9…`; the prereg names `4b961880…`, §1),
`repair.py` (`7b9a1695…`), `repair_structural.py` (`77067a71…`), `action_checks.py`
(`0e723694…`) and `faults.py` (`d26a407c…`), none changed · population: the 96 repositories of
the SWALLOW-6 receipt (`4c6920b4…`; the prereg names `703583d3…`, §1) at their pinned HEADs ·
the gate run at 21,580 mainline commits that touch a hand-written workflow (11 roots excluded),
base the first parent · 262 commits timed with nothing memoised · 1,881 s on two workers ·
scored by `swallow7_score.py`.

**INVALID on G-S7-3 as frozen, stated first.** The gate that holds this cycle says the gate at
every commit must reproduce SWALLOW-6's arrivals in `hidden` "at the same (repo, workflow, job,
key, sha)". It reproduces 139 of 147 (94.6%) under that join, below the 99% bar, and fires at 0
of 21,465 commits where the history records no arrival. The 8 it does not reproduce are, every
one, the same step at the same commit under the name the workflow had **at that commit**; the
SWALLOW-6 receipt keeps a lineage under the workflow's **last** name. Resolved through each file's
rename history (post hoc, §1), the gate reproduces **147 of 147**. The frozen join is the rule,
and it fails; the reading is that the gate saw everything. **6 of 7 predictions HIT are reported,
not claimed**: the gate fires on **104 of 21,569 commits (0.5%)**; **101 of the 147** checks it
catches have a verified repair on that revision's text (P2 predicted 70%: MISS at 68.7%, the one
blind miss), **66 of them one line**, median 1 (P3); **113 of 147 are born hidden** (P4: 77%), 17
are existing checks turned hidden, 17 are steps that became checks already hidden; on the sample
the gate takes **0.15 s** median and 1.13 s at the 90th percentile (P5); **8 commits bring three
or more** hidden checks at once (P6), `dotnet/maui`'s ten in one; and **90 of 147 carry
`continue-on-error`** at the commit that brought them (P7: 61%).

## 0. What was done

For every changed hand-written workflow between a commit and its first parent, both texts were
read with the frozen reading, every step matched across the two revisions by job and key — name,
else id, else first line; a renamed step by its script — and classified: newly hidden (hidden
here, loud or not a check or absent before: the gate fires), hidden after unread, removed hidden
(repaired, no longer a check, or gone), still hidden. For every newly hidden check, SWALLOW-4's
repairs and then SWALLOW-5's were tried on the head's text and the first verified one recorded
with its diff. The reading was memoised per workflow across commits; every 100th commit was read
again with nothing memoised and timed. The same gate is `styxx ci-audit --base <rev>`, which
reads the checkout's HEAD against the merge-base with `<rev>` and exits 1 only when the change
brings a hidden check, and `.github/workflows/ci-audit.yml`, which runs it on this repository's
pull requests that touch a workflow.

## 1. Gates, and what the runs exposed

| gate | bar | result |
|---|---|---|
| G-S7-1 instrument | tests green; every outcome on the scripted history; deterministic | pass — `tests/test_harness_differential.py`, 4 passed |
| G-S7-2 population | ≥ 90 repositories read uncapped | pass — 96 of 96, 0 capped |
| G-S7-3 the gate is the history | (a) ≥ 99% of arrivals are firings at the same (repo, workflow, job, key, sha); (b) ≤ 0.5% of quiet commits fire | **FAIL** — (a) **139 of 147, 94.6%**; (b) 0 of 21,465, 0.0% |
| G-S7-4 frozen underneath | the named hashes | pass at the amended hashes of `history.py` and the SWALLOW-6 receipt (below); everything else as frozen |
| G-S7-5 ledger | P1–P7 scored | pass |

**Run 1 exposed a defect in SWALLOW-6, and SWALLOW-6 was rerun.** Run 1 was made against
`history.py` at `4b961880…` and the SWALLOW-6 receipt at `703583d3…`, as the prereg names. It was
INVALID on G-S7-3 at 94.4% with 8 misses: 4 were lineages whose key the receipt kept after a
later step rename (the scorer now resolves a key to what it was at that revision, through the
`renames` the receipt keeps, and the prereg's "same key" is read that way — stated); 2 were
arrivals at a root commit, which the prereg excludes from every count (stated); and 2 —
`gh-aw`'s `Verify no compilation errors`, `airbyte`'s `Run lint check (info only)` — were
arrivals the history placed at a commit where the gate saw nothing new. Reading those two showed
that SWALLOW-6's instrument fetched a renamed workflow's earlier revisions at its later path, so
they read as absent and the lineage was born at the rename. That is SWALLOW-6's defect; it was
amended twice (its runs 4 and 5, stated in its RESULT: a renamed workflow's revisions, and a
repair's revision before, read at the path they had then), and its receipt moved to `4c6920b4…`.
**Run 2** (this receipt) was made against the amended `history.py` (`93efb4a9…`, which this
instrument imports) and that receipt. The differential instrument itself did not change between
the runs (`91e4a4a7…`), and its firings did not: 104 commits, 147 checks, both runs.

**What remains under the frozen join.** Run 2's 8 misses are the mirror of the defect just
fixed: the receipt now follows a renamed workflow correctly, but keeps its lineages under the
workflow's last name (`test-integration-agentics.yml`, `connector-ci-checks.yml`, `openai.yml`,
`on-pr.yml`, `frontend.yml`), while the gate, which reads one commit, names the workflow as it
was then (`integration-test-agentics.yml`, `connector-test-command.yml`, `github-models.yml`,
`test.yml`, `js-build-and-lint.yml`). The scorer, given the clones, follows each file's rename
history with `git log --follow` and finds all 8 at the same commit under the earlier name —
printed as **post hoc, not gating**. The frozen rule joins on the name the receipt keeps, and
under it the cycle is INVALID.

**Deviation — the prereg's pins.** G-S7-4 as frozen names `history.py` at `4b961880…` and the
SWALLOW-6 receipt at `703583d3…`. Run 2 is at `93efb4a9…` and `4c6920b4…` for the reason above,
and the scorer pins those with a comment saying so; every other hash is as frozen.

## 2. Predictions, reported

| | predicted | observed | |
|---|---|---|---|
| P1 a quiet gate (calibrated) | fires on ≤ 1.0% of commits | **104 of 21,569, 0.5%** | HIT |
| P2 actionable at the commit (blind) | ≥ 70% of newly hidden checks have a verified repair there | **101 of 147, 68.7%** (66 `no-continue-on-error`, 16 `strict-shell`, 4 `both`, 8 `guard-status`, 7 `no-default`) | MISS |
| P3 small at the commit (blind) | median lines of those repairs ≤ 2 | **median 1** (66 one-line, 13 three, 11 four, 10 five, 1 eleven) | HIT |
| P4 born, not turned (calibrated) | ≥ 70% born hidden | **113 of 147, 76.9%** (17 acquired, 17 became a check already hidden) | HIT |
| P5 fast enough (blind) | sample median ≤ 3 s, p90 ≤ 15 s | **0.15 s / 1.13 s** (max 23.1 s; median 2 reads) | HIT |
| P6 batches (calibrated) | ≥ 4 firing commits with ≥ 3 | **8** | HIT |
| P7 the line is `continue-on-error` (blind) | ≥ 50% carry it at the firing revision | **90 of 147, 61.2%** (127 SWALLOWED, 20 FAIL_OPEN) | HIT |

## 3. What the gate would have said

**Quiet.** Of 21,569 pull-request-sized changes to a workflow, 104 would have been stopped. The
other 21,465 change workflows without hiding a check, and the gate says nothing; among them, 89
hidden checks are made loud or removed, and the card says so without failing.

**When it fires, it usually has the fix.** 101 of 147 newly hidden checks have a verified repair
on the very text that brought them — 66 of them the one line `continue-on-error: true` removed,
16 a strict shell, 8 the guard, 7 the default removed. The 46 without: 26 have no
`continue-on-error` and nothing a strict shell reaches (the script's own logic hides the failure,
SWALLOW-5's residue at birth), 20 have a candidate that is not loud under the reading. P2 missed
its bar by two checks.

**They arrive in batches, and increasingly by machine.** Eight commits bring three or more at
once: `dotnet/maui`'s ten `Validate COPILOT_PAT_N` came in "[CI] Agentic workflows: Update gh-aw
generated assets to v0.81.6" — hand-named workflows written by a compiler; `nodetool`'s two
batches of six ("add nodejs code bots", "add ten scheduled maintenance routines"), 25 of its
firings in all; `vscode`'s four smoke-test diagnostics in one "Add GitHub action for pull
requests"; `bun`'s four node-runner test steps in "tweak github actions"; `sentry`'s four in "Add
backend selective testing workflow"; `airbyte`'s three, `vtcode`'s three.

**Seventeen turned.** The existing checks made hidden are SWALLOW-6's acquisitions, now with the
gate's repair beside each: `cal.com`'s `Run Lint` (twice), `promptfoo`'s staging redteam,
`gh-aw`'s error-message lint, `neondatabase`'s user-outcome monitor, `genaiscript`'s summarize,
`FastLED`'s Docker check, `mochi`, `nodetool`, two `MontrealAI` merges — a one-line
`no-continue-on-error` for each — and `bun`'s four rewrites (`strict-shell`, 3 lines) and
`cmux`'s (`guard-status`, 4 lines).

**Seventeen became checks already hidden**: `maui`'s ten, `novu`'s four (three service-start
steps and `Get affected`, rewritten into checks that swallow), `sentry`'s acceptance tests,
`bun`'s Windows tests, `selfxyz`'s version check.

## 4. What this does not say

The gate reads the frozen model; a check it calls hidden hides its failure under that model, not
in any run. It reads first-parent history, so a change squashed into a merge is read at the
merge. A repository whose history begins at the clone's boundary has its first commit read as a
root and excluded. The timing is this machine's, two workers busy. The INVALID is on the join
between two readings of the same history; the gate's own two halves — every arrival reproduced
once names are followed, no firing where the history saw nothing — are the reading, not the
verdict.

## 5. What ships

`styxx ci-audit . --base <rev>`: the pull request's gate. Only the workflows changed since the
merge-base are read; the card says what HEAD hides that the base did not — each with the verified
repair and its diff — what it made loud or removed, and what was hidden on both sides (not this
change's doing); the exit status is 1 only for a newly hidden check. `owner/repo` targets get the
same after a deepening. `.github/workflows/ci-audit.yml` runs it on this repository's own pull
requests that touch a workflow. `styxx/ciaudit/differential.py` is the living copy;
`tests/test_ciaudit.py` pins the frozen instrument at `91e4a4a7…` and holds the copy to it at
every commit of the scripted history, which now also carries a workflow added, renamed, repaired
under its new name and removed.

## 6. Next

A join that follows renames, frozen before the run, so this gate can be scored VALID or not on
its own terms. The 46 checks the gate catches without a fix: 26 are SWALLOW-5's residue at birth,
and a third flavour that knows which answers may be empty is still the edit that would reach
them. And the machine-written hidden check — `maui`'s ten from a compiler, `nodetool`'s
twenty-five from bots — is now the largest source in the population; a gate on the compiler's
output is where it would be caught.
