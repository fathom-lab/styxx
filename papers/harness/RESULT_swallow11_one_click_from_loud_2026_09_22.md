# RESULT — SWALLOW-11: one click from loud — of the 226 hidden checks the gate has caught, the Action puts the annotation on the hiding line for all 226, inside the diff for 213, and the verified fix one click away for 165; in pull requests, 76 of 79

Fathom Lab · 2026-09-22 · Scores the receipt `swallow11_receipt.json.gz` against the
preregistration frozen at sha256 `1017fac935d1945880177a07d67f7656b79d069c1504a78619d858e354194ec4`.
Not amended. One run.

Receipt: `papers/harness/swallow11_receipt.json.gz` (sha256 of the JSON `d84c70ec…`, recorded by the scorer;
every pair with its shas, every check with the lines each flag was decided on — no name, no
address, no text) · instrument `benchmarks/harness_mutation/one_click.py`, sha256 `047a1123…`,
calling the product as it ships: `styxx/ciaudit/action.py` `a9615d9b…` and the living gate
`styxx/ciaudit/differential.py` `95f6ccf1…` · sources: the SWALLOW-7, -9 and -10 receipts
(`c6b12d09…`, `b78c4730…`, `cc6b5dc1…`) · 140 pairs in 60 repositories, 226 checks · 8 minutes ·
scored by `swallow11_score.py`.

**VALID. 7 of 8 predictions HIT** (P1–P5, P7, P8); P6 MISS, the other way. **The product re-read
all 140 changes and found exactly the hidden checks the research instruments had found in every
one of them.** For all **226** it located the line that hides the check — the
`continue-on-error:` (the step's, 96; the job's, 23) or the `run:` block (107); **213 (94%)** of
those lines are inside the change's own diff, where the Files-changed view shows the annotation;
**177** checks have a verified repair, and every one of the 177 was rebuilt identical to the diff
the gate printed and reproduced line for line by its suggestion. **165 of the 226 (73%) are one
click away** — a review suggestion inside the change's diff that, applied, is the verified repair
— and **124 of the 165 replace a single line**. In pull requests the share is **76 of 79 (96%)**:
people's 31 of 32, the agents' 25 of 27, agent-signed 20 of 20. In mainline commits it is 89 of
147 (61%), because 46 of the 49 checks without a verified repair are there.

## 0. What was done

For every firing pair of three receipts — 104 mainline commits (SWALLOW-7), 17 agents' pull
requests (SWALLOW-9), 19 more pull requests (SWALLOW-10: people's, agent-signed, and two agents'
SWALLOW-9 did not hold) — the change was re-read with the living `audit_commit` on the receipt's
own BASE and HEAD, in the SWALLOW-6 clones or in a blobless clone holding the two commits. For
every newly hidden check, the functions the Action ships decided where the annotation goes, whether
that line is in the change's diff (`git diff -U3 -M`, a pull request's context), whether the
verified repair can be rebuilt identical to the printed diff, what the smallest suggestion is,
whether it reproduces the repair, and whether its lines lie inside one hunk of the diff — the rule
for where GitHub accepts a suggestion.

## 1. Gates

| gate | bar | result |
|---|---|---|
| G-S11-1 instrument | the population rule, one-click on the fixture, a fix outside the diff, determinism; the Action's tests | pass — 22 passed |
| G-S11-2 reproduction | ≥ 95% of 140 pairs re-read; ≥ 95% give the receipt's set | pass — **140 of 140 re-read; 140 of 140 give the receipt's set exactly** |
| G-S11-3 frozen underneath | instrument, Action, living gate, three receipts | pass |
| G-S11-4 construction | a suggestion is the verified repair | pass — **177 of 177** rebuilt repairs reproduced by their suggestion |
| G-S11-5 ledger | P1–P8 scored | pass |

No deviation.

## 2. Predictions, scored

| | predicted | observed | |
|---|---|---|---|
| P1 located (calibrated) | ≥ 98% | **226 of 226** — `run:` 107, the step's `continue-on-error:` 96, the job's 23 | HIT |
| P2 visible (blind) | ≥ 80% of located annotations inside the change's diff | **213 of 226 (94%)** | HIT |
| P3 one click (blind) | ≥ 60% of all checks | **165 of 226 (73%)** — 49 have no verified repair, 12 have one outside the diff | HIT |
| P4 placeable when repaired (blind) | ≥ 80% of checks with a rebuilt repair | **165 of 177 (93%)** | HIT |
| P5 small (blind) | ≥ 75% of one-click suggestions replace ≤ 3 lines | **133 of 165 (81%)**; 124 replace one line | HIT |
| P6 born is easier (blind) | born-hidden share > acquired | **born 131 of 177 (74%), acquired 32 of 32 (100%)** | MISS |
| P7 the pull request is the place (blind) | pull requests ≥ mainline | **76 of 79 (96%) vs 89 of 147 (61%)** | HIT |
| P8 people's before agents' (blind) | people's ≥ agents' | **31 of 32 (97%) vs 25 of 27 (93%)**; agent-signed 20 of 20 | HIT |

## 3. Reading

**The annotation lands where the author is reading.** All 226 checks were located in HEAD's text,
and 213 of the lines that hide them are part of the change — the Files-changed view would have
drawn the error on them. The 13 that are not: 11 `continue-on-error:` lines the change did not
touch — ten of them in a single commit, the Copilot-authored update of `dotnet/maui`'s generated
assets that rewrote ten `Validate COPILOT_PAT_N` steps into checks under a `continue-on-error`
already there — and 2 `run:` blocks. The annotation still names the file and line; the diff view
just does not show that line.

**The fix is one click away three times in four; in a pull request, almost always.** 177 of the
226 checks have a repair SWALLOW-4 or -5 verified (loud under the same fault, the healthy run
unchanged); for 165 of them the smallest suggestion that reproduces it sits inside the change's
own diff. By repair: removing `continue-on-error` 83 of 94, the strict shell 50 of 50, removing an
`|| echo` default 15 of 16, the guard 13 of 13, both edits 4 of 4. The 12 that are not placeable
are the ten `maui` steps (the repair deletes a line the change did not touch), one `py3plex` step
and one `selfxyz` default whose seven-line span leaves the hunk. 52 of the 60 repositories have at
least one one-click fix. In the 2025 pull requests of agents and people it is 76 of 79.

**Mostly one line.** 124 of the 165 suggestions replace one line: 83 delete a `continue-on-error:
true` (an empty suggestion — the line goes), and most strict-shell repairs replace a one-line
`run:` with three (`run: |`, `set -eo pipefail`, the command). The long tail is real: 12 replace
six lines, and the largest two, in `getsentry/sentry-docs`, make a 38- and a 45-line script strict.
A reviewer can accept a one-line deletion at a glance; a 45-line suggestion is still one click,
and the summary shows it as a diff first.

**The miss.** P6 guessed a born-hidden check (a new step) would be easier than an acquired one
(an existing check made hidden). It is the other way, and for a reason the replay makes plain: an
acquired check is hidden by a line the change itself wrote — the `continue-on-error: true` it
added, the `|| true` it appended — so the fix is always in the diff (32 of 32). A born-hidden
check is new, but 44 of the 46 born-hidden checks without a one-click fix have no verified repair
at all (fail-open queries and scripts the two repair families do not reach); only 2 are misplaced.

**Mainline and pull request.** The mainline share is lower (61%) not because suggestions misplace
there — 89 of the 101 repaired mainline checks are one click — but because the mainline holds 46
of the 49 checks with no verified repair, and all ten `maui` steps. The receipts' pull requests
are 2025's; the mainline reaches back to 2019.

**The product is the instrument.** Every one of the 140 changes, re-read by the code that ships in
`styxx` — the living `differential.py` the Action and `styxx ci-audit --base/--pr` call — gave
exactly the newly hidden set (workflow, job, step, kind) the frozen research instruments recorded,
across the first-parent, commit-list and closest-branch bases. The equality `test_ciaudit.py`
holds on a scripted history now holds on 140 real changes.

## 4. What this does not say

It does not say an author would have clicked, or that GitHub accepted every placeable suggestion:
the placement rule is the documented one (lines inside one hunk of the change's diff), and it was
not exercised against the API on these changes. A suggestion for a mainline commit is what a pull
request with that commit's diff would have shown. "Verified" is the SWALLOW-4/5 verification:
loud under the same fault, the healthy run unchanged, read by the frozen model — RED is loud, not
correct. The 49 checks with no verified repair get the annotation and the explanation, not a fix.

## 5. What ships

**`ci-audit/action.yml`** — `uses: fathom-lab/styxx/ci-audit@<ref>` after `actions/checkout`:
the gate on the change that triggered the workflow (pull_request: GitHub's test merge against its
first parent; merge_group; push), an error annotation on the line that hides each new check, the
verified repairs as diffs in the job summary, outputs and a JSON receipt, and — with `suggest:
true` and `pull-requests: write` — each repair posted once as a one-click review suggestion. It
runs styxx from its own ref with only numpy and PyYAML installed, refuses `pull_request_target`,
drops every token from its environment before reading the change, keeps the change's text from
acting as a workflow command, and exits 2 — never 0 — when it could not run. This repository's own
`ci-audit.yml` now runs it on its own pull requests, from the default depth-1 checkout.
`one_click.py` is frozen at `047a1123…` and pinned, with `action.py` at `a9615d9b…`, by
`tests/test_harness_one_click.py`.

## 6. Next

The 49 checks with no verified repair are the repair families' frontier: fail-open queries whose
empty answer is the green path, and scripts that swallow further down. And the suggestion's one
click wants a live test on GitHub's API on a pull request made for it.
