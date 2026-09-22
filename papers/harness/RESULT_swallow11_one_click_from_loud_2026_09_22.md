# RESULT — SWALLOW-11: one click from loud — of the 220 hidden checks the gate has caught, the Action puts the annotation on the hiding line for all 220, inside the diff for 207, and the verified fix one click away for 159; in pull requests, 70 of 73

Fathom Lab · 2026-09-22 · Scores the receipt `swallow11_receipt.json.gz` against the
preregistration frozen at sha256 `1017fac935d1945880177a07d67f7656b79d069c1504a78619d858e354194ec4`.
Not amended. One run, and one re-run with the Action as shipped. One correction to the count,
stated below: the population holds one change twice, and every number here counts it once.

Receipt: `papers/harness/swallow11_receipt.json.gz` (sha256 of the JSON `d84c70ec…`, recorded by the scorer;
every pair with its shas, every check with the lines each flag was decided on — no name, no
address, no text) · instrument `benchmarks/harness_mutation/one_click.py`, sha256 `047a1123…`,
calling the product: `styxx/ciaudit/action.py` `a9615d9b…` (shipped as `c642493d…` — one encoding
pin, the replay re-run identical, below) and the living gate `styxx/ciaudit/differential.py`
`95f6ccf1…` · sources: the SWALLOW-7, -9 and -10 receipts
(`c6b12d09…`, `b78c4730…`, `cc6b5dc1…`) · 140 pairs as frozen — 139 changes — in 60 repositories,
220 checks · 8 minutes · scored by `swallow11_score.py` (the rule as frozen), counted once by
`swallow11_once.py`.

**VALID. 7 of 8 predictions HIT** (P1–P5, P7, P8); P6 MISS, the other way — as scored, and
unchanged when each change is counted once. **The product re-read all 139 changes and found
exactly the hidden checks the research instruments had found in every one of them.** For all
**220** it located the line that hides the check — the `continue-on-error:` (the step's, 94; the
job's, 22) or the `run:` block (104); **207 (94%)** of those lines are inside the change's own
diff, where the Files-changed view shows the annotation; **171** checks have a verified repair, and
every one of the 171 was rebuilt identical to the diff the gate printed and reproduced line for
line by its suggestion. **159 of the 220 (72%) are one click away** — a review suggestion inside
the change's diff that, applied, is the verified repair — and **121 of the 159 replace a single
line**. In pull requests the share is **70 of 73 (96%)**: people's 25 of 26, the agents' 25 of 27,
agent-signed 20 of 20. In mainline commits it is 89 of 147 (61%), because 46 of the 49 checks
without a verified repair are there.

## Correction — found after scoring, before merge

**The population holds one change twice.** `ruvnet/ruv-FANN` #44 and #48 are two pull requests
of SWALLOW-10's people's sample with one base (`88f7d70b…`) and one head (`61823716…`): the same
change, its six checks replayed twice. The population rule kept SWALLOW-10's pull requests that
SWALLOW-9 did not hold, by head, but did not deduplicate within SWALLOW-10, and the count frozen in
the preregistration (140 pairs, 226 checks) already held the repeat. `swallow11_scored.json` is the
rule as frozen and stays as scored; every number in this RESULT is the same receipt with each
(repository, base, head) once — `swallow11_once.py` → `swallow11_once.json`. No verdict moves. As
scored: P1 226 of 226; P2 213 of 226; P3 165 of 226 (73%); P4 165 of 177; P5 133 of 165, 124 one
line; P6 born 131 of 177, acquired 32 of 32; P7 76 of 79 against 89 of 147; P8 31 of 32 against
25 of 27. SWALLOW-10 counts pull requests, and these are two; its check-level description counts
the six twice — the note appended to that RESULT gives its numbers by change.

## A change after the run — the Action as shipped, re-run

The repository's encoding guard (`tests/test_subprocess_encoding_pinned.py`) failed this pull
request's CI on one call the replay had used: `readable()` ran `git diff --quiet` in text mode
without pinning an encoding, so the platform's locale would decode its error text. The call now
decodes as UTF-8 with replacement; `action.py` is `c642493d…`. Its answer to the replay is its exit
status, which the encoding does not touch — and the replay was re-run with the shipped file to
show it: `swallow11_rerun_receipt.json.gz` (sha256 of the JSON `28daa2e6…`) is the scored receipt
in all 140 pairs and every check, differing only in the recorded `action.py` and the seconds (218
against 485). `tests/test_harness_one_click.py` pins the shipped file and holds that equality.

## 0. What was done

For every firing pair of three receipts — 104 mainline commits (SWALLOW-7), 17 agents' pull
requests (SWALLOW-9), 19 more pull requests (SWALLOW-10: people's, agent-signed, and two agents'
SWALLOW-9 did not hold; 18 changes, two of the people's pull requests being one) — the change was
re-read with the living `audit_commit` on the receipt's own BASE and HEAD, in the SWALLOW-6 clones
or in a blobless clone holding the two commits. For every newly hidden check, the functions the
Action ships decided where the annotation goes, whether that line is in the change's diff
(`git diff -U3 -M`, a pull request's context), whether the verified repair can be rebuilt
identical to the printed diff, what the smallest suggestion is, whether it reproduces the repair,
and whether its lines lie inside one hunk of the diff — the rule for where GitHub accepts a
suggestion.

## 1. Gates

As scored, on the rule as frozen.

| gate | bar | result |
|---|---|---|
| G-S11-1 instrument | the population rule, one-click on the fixture, a fix outside the diff, determinism; the Action's tests | pass — 22 passed |
| G-S11-2 reproduction | ≥ 95% of 140 pairs re-read; ≥ 95% give the receipt's set | pass — **140 of 140 re-read; 140 of 140 give the receipt's set exactly** (139 of 139 changes) |
| G-S11-3 frozen underneath | instrument, Action, living gate, three receipts | pass |
| G-S11-4 construction | a suggestion is the verified repair | pass — **177 of 177** rebuilt repairs reproduced by their suggestion (171 of 171 once) |
| G-S11-5 ledger | P1–P8 scored | pass |

No deviation from the procedure. Two things after it, both above: a correction to the count, and
a one-line change to the shipped Action with the replay re-run on it.

## 2. Predictions, scored

| | predicted | observed, each change once | as scored | |
|---|---|---|---|---|
| P1 located (calibrated) | ≥ 98% | **220 of 220** — `run:` 104, the step's `continue-on-error:` 94, the job's 22 | 226 of 226 | HIT |
| P2 visible (blind) | ≥ 80% of located annotations inside the change's diff | **207 of 220 (94%)** | 213 of 226 | HIT |
| P3 one click (blind) | ≥ 60% of all checks | **159 of 220 (72%)** — 49 have no verified repair, 12 have one outside the diff | 165 of 226 (73%) | HIT |
| P4 placeable when repaired (blind) | ≥ 80% of checks with a rebuilt repair | **159 of 171 (93%)** | 165 of 177 | HIT |
| P5 small (blind) | ≥ 75% of one-click suggestions replace ≤ 3 lines | **129 of 159 (81%)**; 121 replace one line | 133 of 165; 124 | HIT |
| P6 born is easier (blind) | born-hidden share > acquired | **born 125 of 171 (73%), acquired 32 of 32 (100%)** | 131 of 177; 32 of 32 | MISS |
| P7 the pull request is the place (blind) | pull requests ≥ mainline | **70 of 73 (96%) vs 89 of 147 (61%)** | 76 of 79 vs 89 of 147 | HIT |
| P8 people's before agents' (blind) | people's ≥ agents' | **25 of 26 (96%) vs 25 of 27 (93%)**; agent-signed 20 of 20 | 31 of 32 vs 25 of 27 | HIT |

## 3. Reading

**The annotation lands where the author is reading.** All 220 checks were located in HEAD's text,
and 207 of the lines that hide them are part of the change — the Files-changed view would have
drawn the error on them. The 13 that are not: 11 `continue-on-error:` lines the change did not
touch — ten of them in a single commit, the Copilot-authored update of `dotnet/maui`'s generated
assets that rewrote ten `Validate COPILOT_PAT_N` steps into checks under a `continue-on-error`
already there — and 2 `run:` blocks. The annotation still names the file and line; the diff view
just does not show that line.

**The fix is one click away nearly three times in four; in a pull request, almost always.** 171 of
the 220 checks have a repair SWALLOW-4 or -5 verified (loud under the same fault, the healthy run
unchanged); for 159 of them the smallest suggestion that reproduces it sits inside the change's
own diff. By repair: removing `continue-on-error` 80 of 91, the strict shell 49 of 49, removing an
`|| echo` default 14 of 15, the guard 12 of 12, both edits 4 of 4. The 12 that are not placeable
are the ten `maui` steps (the repair deletes a line the change did not touch), one `py3plex` step
and one `selfxyz` default whose seven-line span leaves the hunk. 52 of the 60 repositories have at
least one one-click fix. In the 2025 pull requests of agents and people it is 70 of 73.

**Mostly one line.** 121 of the 159 suggestions replace one line: 80 delete a `continue-on-error:
true` (an empty suggestion — the line goes), and 35 of the 49 strict-shell repairs replace a
one-line `run:`, most often with three (`run: |`, `set -eo pipefail`, the command). The long tail is real: 11 replace
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

**The product is the instrument.** Every one of the 139 changes, re-read by the code that ships in
`styxx` — the living `differential.py` the Action and `styxx ci-audit --base/--pr` call — gave
exactly the newly hidden set (workflow, job, step, kind) the frozen research instruments recorded,
across the first-parent, commit-list and closest-branch bases. The equality `test_ciaudit.py`
holds on a scripted history now holds on 139 real changes.

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
`one_click.py` is frozen at `047a1123…` and pinned by `tests/test_harness_one_click.py`, which also
holds the re-run equal to the scored receipt and pins, by their source, the ten functions of
`action.py` the replay called (as of `c642493d…`) — the rest of the file moves on (SWALLOW-12).

## 6. Next

The 49 checks with no verified repair are the repair families' frontier: fail-open queries whose
empty answer is the green path, and scripts that swallow further down. The suggestion's one click
wants a live test on GitHub's API on a pull request made for it — SWALLOW-12 ran it: the API placed
and refused exactly as the rule says, and GitHub's button made each suggestion the verified repair,
byte for byte, for fifteen shapes. And a population drawn from
pull requests is deduplicated by (repository, base, head) within every receipt, not only across
them.
