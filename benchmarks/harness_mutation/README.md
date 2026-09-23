# MUTE-1 — mutation testing of the checking harness

Mutation testing asks of a test suite: *if I break the program, does a test go red?*
This asks the same question one level up, of the apparatus that checks the program:
*if I cut a check — delete the job, silence the step, flip its guard, delete the file
it reads, stop the workflow firing — does anything in this repository go red?*

```
python -m benchmarks.harness_mutation.mute --inventory     # the checks and the mutants; runs nothing
git worktree add /tmp/mute-tree HEAD
python -m benchmarks.harness_mutation.mute --run --tree /tmp/mute-tree
python papers/harness/mute1_score.py                       # the frozen predictions, scored
```

Seven operators (M-TRIGGER, M-JOB, M-STEP, M-SWALLOW, M-GUARD, M-SCRIPT, M-SUBJECT), one oracle
(the test files that read harness paths, selected by pattern, never by hand), three verdicts
(KILLED, SURVIVED, UNREACHED), and a receipt that names the test behind every kill. The
instrument refuses to mutate the checkout it lives in.

It measures whether the **test suite** guards the harness. It does not run CI. SURVIVED means
exactly *nothing in `tests/` would tell you*, and a survivor is a finding to be read, not a defect
to be counted — `papers/harness/RESULT_mute1_harness_mutation_2026_09_20.md` reads the first 101.

The receipt carries a *harness fingerprint*: a sha256 over every file a mutant can touch plus every
file the oracle reads. Two commits with the same fingerprint have the same harness, and a run
applies to either. That is how a run made on a locally merged tree is checked against the commit
that eventually carries the same files.

## The census (SWALLOW-1): the behavioural guard, taken to other repositories

MUTE-2's second guard needs nothing but a repository's workflow files: execute every `run:` step
with an empty PATH, so that every external command fails, and see whether the step goes red.
`census.py` does that for any list of repositories — a blob-less sparse clone of
`.github/workflows`, every bash step executed and classified (PROPAGATES, SWALLOWS, TOOLLESS,
SYNTAX, TIMEOUT, NOT_BASH), `continue-on-error`, a category, and #137's exact shape recorded
beside the verdict.

```
python -m benchmarks.harness_mutation.census --tree .                                   # this repository
python -m benchmarks.harness_mutation.census --repos papers/harness/swallow1_population.json --out receipt.json
python papers/harness/swallow1_score.py
```

It sees that a step *cannot fail*; it cannot see which way the step falls when it does not —
`papers/harness/RESULT_swallow1_ci_steps_that_cannot_fail_2026_09_21.md` says so, and reads the
38 verification steps it found one by one.

## Fault injection (SWALLOW-2): which way it falls

The census sees that a step cannot fail; it cannot see which way the step falls. `faults.py`
simulates the *workflow*: two healthy worlds (every tool succeeds and prints `x`, or prints
nothing), and for every bash step that reaches a tool, one fault world in which that step's
tools fail. Outputs, env, `if:`, `needs:` and `fromJSON` matrices are followed; what the
simulation cannot know is unknown and lets a step run. Each fault gets one verdict — RED,
FAIL_OPEN, SWALLOWED, ABSORBED, NO_CHECK — and the receipt names every dropped check with its
mechanism.

```
python -m benchmarks.harness_mutation.faults --tree .                                      # this repository
python -m benchmarks.harness_mutation.faults --repos papers/harness/swallow1_population.json --out receipt.json
python papers/harness/swallow2_self.py --before <gauntlet-pr.yml@main> --after <gauntlet-pr.yml@137>
python papers/harness/swallow2_score.py
```

It reproduces #137 with no runner and no token: on `main` the two gated steps are skipped
and the job is green when the discover step's tools fail; on #137's tree the same fault is RED.
`papers/harness/RESULT_swallow2_which_way_it_falls_2026_09_21.md` has the map for 100
repositories, and the runs it took to draw it.

`faults.py` here is **frozen** at the sha256 the SWALLOW-2 receipt names (`d26a407c…`;
`tests/test_ciaudit.py` pins it). The living copy is the shipped engine, `styxx/ciaudit/engine.py`,
behind `styxx ci-audit`; the same test holds the two to identical verdicts on the fixtures and on
this repository's own workflows until a cycle declares otherwise.

## The checks that are actions (SWALLOW-3)

`faults.py` sees a check only in a `run:` step; a check that is an action (`pre-commit/action`,
`lycheeverse/lychee-action`, CodeQL's `analyze`) is never executed and never counted.
`action_checks.py` is the declared list — the rule, every `uses:` name the population's hand-written
workflows use decided once with its reason (`CATALOGUE`, `FAMILIES`, `NOT_CHECKS`; the counts in
`papers/harness/swallow3_actions_census.json`) — and the instrument that applies it on top of
`faults.py` without changing it: an action check is *reached* when its job runs, no earlier step
failed and its own `if:` is not false, and *dropped* when reached in the healthy world and not in
the fault world. Every fault carries `verdict_runs_only` (SWALLOW-2's reading) beside `verdict`.

```
python papers/harness/swallow3_census.py --work <clones>                              # the census, at SWALLOW-2's HEADs
python -m benchmarks.harness_mutation.action_checks --repos papers/harness/swallow1_population.json \
    --heads papers/harness/swallow2_receipt.json.gz --work <clones> --keep --out receipt.json
python -m benchmarks.harness_mutation.action_checks --tree .
python papers/harness/swallow3_score.py
python papers/harness/swallow3_repro.py --clones <clones>                             # the evidence for G-S3-3
```

`papers/harness/RESULT_swallow3_the_checks_that_are_actions_2026_09_21.md` is **INVALID** on its
reproduction gate — a re-clone at the same HEADs did not reproduce SWALLOW-2 in 8 of 30,642
records, for three reasons in `faults.py`'s contact with the world (the real `date`, a tie in the
stub order, a background subshell racing the log), each demonstrated in `swallow3_repro.json` —
and reports, not claims, what the catalogue moved: 5 hand-written faults, one to a dropped check.
`action_checks.py` is frozen at the sha256 that receipt names (`0e723694…`); the living copy of the
catalogue is `styxx/ciaudit/actions.py`, and `tests/test_ciaudit.py` holds them equal.

## The repair is loud (SWALLOW-4)

For every hidden or dropped check, `repair.py` tries two stated edits to the workflow text at the
fault site — remove the step's `continue-on-error`; make its shell strict — and verifies each on
both halves with the same instrument: LOUD (the same fault is RED on the repaired workflow) and
UNCHANGED (a healthy run of the repaired workflow is indistinguishable from the original's, in
both flavours). A repair that is loud but changes a healthy run has found what the original line
was protecting, and is rejected with the flavour and the step. The receipt carries every diff.

```
python -m benchmarks.harness_mutation.repair --tree .
python -m benchmarks.harness_mutation.repair --receipt papers/harness/swallow3_receipt.json.gz --clones <clones> --out receipt.json
python papers/harness/swallow4_score.py
```

`papers/harness/RESULT_swallow4_the_repair_is_loud_2026_09_21.md` (VALID, 8/10): 39 of the
population's 53 hand-written hidden and dropped checks have a verified repair, 32 of them one
line; every check hidden by `continue-on-error` is one line from loud; the shell hides the rest by
control flow, which a strict shell does not reach. `repair.py` is frozen at the sha256 that
receipt names (`7b9a1695…`); the living copy is `styxx/ciaudit/repair.py`, behind
`styxx ci-audit --repair`, and `tests/test_ciaudit.py` holds the two to identical outcomes.

## The structural repairs (SWALLOW-5)

For the checks a strict shell cannot make loud, `repair_structural.py` tries two edits to the
script's logic, on the residue SWALLOW-4 leaves: `guard-status` (a single-line `if CMD; then`
becomes an explicit status capture in which a status above 1 fails the step, CMD kept exempt from
errexit) and `no-default` (every `|| echo …` fallback removed, then the strict shell), verified on
the same two halves. `test -n` on an answer is not tried, and the module says why: the
instrument's `empty` flavour is a healthy run in which every answer is empty.

```
python -m benchmarks.harness_mutation.repair_structural --tree .
python -m benchmarks.harness_mutation.repair_structural --receipt papers/harness/swallow4_receipt.json.gz --clones <clones> --out receipt.json
python papers/harness/swallow5_score.py
```

`papers/harness/RESULT_swallow5_the_structural_repairs_2026_09_21.md` (VALID, 9/12; two runs, the
revision stated): 7 of the 14 hand-written checks SWALLOW-4 could not make loud have a verified
structural repair, five of them the guard; the remainder is three warn-by-design verifiers, one
reporter, one fail-closed by design, and three that need the one edit this instrument cannot
verify. `repair_structural.py` is frozen at the sha256 that receipt names (`77067a71…`); the living
copy is `styxx/ciaudit/repair_structural.py`, the second stage of `styxx ci-audit --repair`.


## Where hidden checks come from (SWALLOW-6)

`history.py` reads the same checks through time: every revision of every hand-written workflow on
the default branch's mainline (first-parent, from the GitHub Actions YAML era's start on
2019-08-01 to the pinned HEAD) with the frozen SWALLOW-3 reading, and follows every check as a
lineage — the same job, the same step name (else id, else first line) — from the commit that
created it to HEAD or to the commit that removed it. Between readable revisions, loud → hidden is
an acquisition and hidden → loud a repair, each with the commit, the mechanism read from the
step's YAML before and after (`continue-on-error`, `|| true`, `set +e`, a `|| echo` default,
gating, a rewrite, or a change elsewhere), and the first acknowledgement a stated word list finds
in the commit message. For every repair, SWALLOW-4's and SWALLOW-5's candidates are tried on the
revision before and the first verified one is compared with what the author did. The clones are
blob-less and shallow to 2019-08-01; every workflow blob along the mainline is fetched in one
batch.

```
python -m benchmarks.harness_mutation.history --tree .
python -m benchmarks.harness_mutation.history --receipt papers/harness/swallow3_receipt.json.gz --work <clones> --workers 2 --out receipt.json
python papers/harness/swallow6_score.py
```

`papers/harness/RESULT_swallow6_where_hidden_checks_come_from_2026_09_21.md` (VALID, 5/7; five
runs, four amendments stated): of the 53 hidden checks alive at HEAD, 40 were written hidden and
never loud, 3 were hidden later by a `continue-on-error`, 10 were rewritten into hidden checks by
one commit; of 148 lineages ever hidden, 5 are loud today and 68 died hidden; the median hidden
check is 155 days old, the oldest 1,630; 17 commits hid a loud check and 5 said why in the stated
words; 15 made one loud, and 13 times the edit is the one the instrument proposes. `history.py` is
frozen at the sha256 that receipt names (`93efb4a9…`); the living copy is
`styxx/ciaudit/history.py`, behind `styxx ci-audit --history`.

## The differential audit (SWALLOW-7)

`differential.py` is the pull request's gate: given a base and a head, it reads only the
workflows that changed, matches every step across the two revisions (job and step name, else id,
else first line; a renamed step by its script) and reports what the head hides that the base did
not — each with the repair SWALLOW-4 then SWALLOW-5 verifies on the head's text — what it made
loud or removed, and what was hidden on both sides. The population reading runs that gate at
every mainline commit of the SWALLOW-6 clones that touches a hand-written workflow, base its first
parent, and times every 100th with nothing memoised.

```
python -m benchmarks.harness_mutation.differential --tree . --base origin/main
python -m benchmarks.harness_mutation.differential --receipt papers/harness/swallow6_receipt.json.gz --work <clones> --workers 2 --out receipt.json
python papers/harness/swallow7_score.py [<clones>]
```

`papers/harness/RESULT_swallow7_the_differential_audit_2026_09_21.md` (INVALID on its own join
with the SWALLOW-6 history, stated first; 6/7 reported, not claimed; two runs): the gate fires on
104 of 21,569 commits, reproduces every one of the history's 147 arrivals once a renamed workflow
is followed and never fires where the history saw nothing, has a verified repair for 101 of the
147 checks it catches (66 of them one line), and costs 0.15 s median. Its first run exposed the
rename defect in SWALLOW-6's instrument, which was fixed and rerun. `differential.py` is frozen
at the sha256 that receipt names (`91e4a4a7…`); the living copy is
`styxx/ciaudit/differential.py`, behind `styxx ci-audit --base`, and
`.github/workflows/ci-audit.yml` runs it on this repository's own pull requests.

## Who writes the hidden check (SWALLOW-8)

`authorship.py` reads every mainline commit of the SWALLOW-7 receipt for its author name and
address, subject and body, and classes it by a stated rule — `agent` on a coding agent's
signature (an agent as author, a `Co-authored-by` trailer naming one, "Generated with …",
"[CI] Agentic workflows", a merge of a `codex/`, `claude/`, `copilot/` branch), `automation` on a
bot author with no such signature, `human` otherwise — and joins the class to the gate's
firings. A bare first name is not a signal; the receipt carries classes and signal labels, never
a name; the agent class is a floor.

```
python -m benchmarks.harness_mutation.authorship --receipt papers/harness/swallow7_receipt.json.gz --work <clones> --out receipt.json
python papers/harness/swallow8_score.py
```

`papers/harness/RESULT_swallow8_who_writes_the_hidden_check_2026_09_21.md` (VALID, 4/7): 4,098 of
21,569 workflow-touching commits carry an agent's signature — 0.09% of 2024's, 38% of 2026's —
and bring 49 of the 147 newly hidden checks; 0.76% of agent commits fire against 0.46% of a
person's (1.64×, short of the 2× predicted; 1.27× within 2025–2026); none of 1,690 dependency
and release bot commits fires; the agent's hidden check is repaired no worse (73% vs 66%).
`authorship.py` is frozen at the sha256 that receipt names (`c3fb6e42…`) and pinned by its test.

## The agent's pull request at the gate (SWALLOW-9)

`agent_prs.py` runs SWALLOW-7's gate on the pull requests AIDev (Zenodo record 16919272) lists
for five coding agents — OpenAI Codex, Copilot, Devin, Cursor, Claude Code, with the agent named
by GitHub's own attribution and whether a person merged it — on every one whose own commits touch
a hand-written workflow. HEAD is `refs/pull/N/head` as GitHub keeps it, checked against the
dataset's commit list; BASE is what that list implies — the parents of the pull request's commits
that are not its commits, the base branch as the pull request last saw it, whatever branch that
is; the gate reads only the workflows the dataset says the pull request changed, and a pull
request whose git diff is larger than the dataset's count is *suspect*. For a merged pull request
that fires, each check is looked for at the default branch's tip. The receipt carries repository,
number, agent, state, shas and the gate's records — no name, no address, no text beyond the
acknowledgement word.

```
python -m benchmarks.harness_mutation.agent_prs --aidev <dir> --population-out papers/harness/swallow9_population.json
python -m benchmarks.harness_mutation.agent_prs --population papers/harness/swallow9_population.json.gz --work <clones> --workers 3 --out receipt.json
python papers/harness/swallow9_score.py
```

`papers/harness/RESULT_swallow9_the_agents_pull_request_at_the_gate_2026_09_21.md` (VALID on its
gates, 2/8; two runs stated — the first's base rule compared a pull request into `dev` with
`main` and was amended): of 2,123 audited pull requests the gate fires on 17 (0.8%; 1.2% of the
1,407 with a workflow change of their own), bringing 24 hidden checks; a person merged 9 of the
17 (53%) against 77% of the rest; 22 of 24 have a verified repair; only 6 carry
`continue-on-error` — the rest are `|| true`, `|| echo` defaults, `set +e` and fail-open
queries; Claude Code's pull requests fire most (3 of 49, one bringing seven checks), Copilot's
least (3 of 516); of the 9 checks merged into a default branch, 2 are still hidden today.
`agent_prs.py` is frozen at the sha256 that receipt names (`ec9ef750…`) and pinned by its test.

## The baseline (SWALLOW-10)

`human_prs.py` puts the agents' pull requests and everyone else's through one pipeline. For every
repository SWALLOW-9 read, the numbers between its first and last agent pull request in the
dataset's window are the same months; the ones the dataset attributes to no agent are sampled
(seeded, up to 150 per repository) from what `refs/pull/*/head` lists. Each pull request is
grouped — the dataset's agent, else SWALLOW-8's rule on the head commit (`agent-signed`,
`automation`, `human`) — based by the closest-branch rule (the merge commit's first parent when
one merged the head, else the fork point from the nearest branch), diffed for hand-written
workflows, read by SWALLOW-7's gate, and given a merge signal from git alone (a merge commit, a
squash subject, a rebase's kept subject and author date). No name is written.

```
python -m benchmarks.harness_mutation.human_prs --aidev <dir> --receipt9 papers/harness/swallow9_receipt.json.gz --population9 papers/harness/swallow9_population.json.gz --sample-out sample.json
python -m benchmarks.harness_mutation.human_prs --sample papers/harness/swallow10_sample.json.gz --receipt9 papers/harness/swallow9_receipt.json.gz --work <clones> --workers 5 --out receipt.json
python papers/harness/swallow10_score.py
```

`papers/harness/RESULT_swallow10_the_baseline_2026_09_22.md` (INVALID on its merge-signal gate
as frozen — the gate held the signal against a dataset flag that is a snapshot, and 177 of the
221 "false positives" are pull requests merged after collection; 8/8 reported, not claimed): of
1,989 pull requests people opened that change a workflow, 12 bring a hidden check (0.60%); of
1,479 the agents opened, 18 (1.22%) — twice the rate, by one pipeline, one base rule, one gate;
without the one repository that holds five of the twelve human firings, 0.35% and 3.4×. A
person's hidden check is the textbook one (19 of 32 `continue-on-error`, 31 of 32 born hidden,
31 of 32 repairable); none of 249 bot pull requests fires; pull requests outside the dataset's
attribution whose head commit carries an agent's signature fire at 3.7%. The closest-branch base
agrees with SWALLOW-9's on 96.5% of the agents' pull requests and the gate's verdict on 100%.
`human_prs.py` is frozen at the sha256 that receipt names (`7d764d5d…`) and pinned by its test.

## One click from loud (SWALLOW-11)

`ci-audit/action.yml` ships the gate as a GitHub Action (the driver is `styxx/ciaudit/action.py`):
on the change that triggered the workflow it marks the line that hides each new check, writes the
verified repair into the job summary, and with `suggest: true` posts it as a one-click review
suggestion. `one_click.py` replays every firing pair of the SWALLOW-7, -9 and -10 receipts through
the functions the Action ships and asks, for each hidden check: located? the annotation inside the
change's diff? the verified repair rebuilt identical to the printed diff, and reproduced by its
suggestion? the suggestion inside one hunk of the diff — one click away?

```
python -m benchmarks.harness_mutation.one_click --r7 papers/harness/swallow7_receipt.json.gz --r9 papers/harness/swallow9_receipt.json.gz --r10 papers/harness/swallow10_receipt.json.gz --clones7 <swallow-6 clones> --work <clones> --out receipt.json
python papers/harness/swallow11_score.py
```

`papers/harness/RESULT_swallow11_one_click_from_loud_2026_09_22.md` (VALID, 7/8): the product
re-read all 139 changes and found exactly the receipts' hidden checks in each; it located the
hiding line for all 220 checks, 207 inside the change's diff; all 171 verified repairs were rebuilt
and reproduced by their suggestion; 159 of 220 (72%) are one click away, 121 of them a single line,
and in pull requests 70 of 73. The miss: an acquired check is always one click (the change wrote
the hiding line), a born-hidden one less often — for want of a verified repair, not of a place to
put it. The population as frozen held one change twice (two pull requests, one base, one head);
these numbers count it once — `papers/harness/swallow11_once.py` — and the scored file keeps the
rule as frozen. `one_click.py` is frozen at `047a1123…` and pinned by its test, with the ten functions of `action.py`
the replay called pinned by their source (as of `c642493d…`, one encoding pin after the run; the
re-run receipt is the scored one in every check).

## The click, live (SWALLOW-12)

SWALLOW-11's one click was the documented rule. `live_click.py` puts it to GitHub: `files` writes
a pull request made for it — eighteen workflow files, fifteen carrying the suggestion shapes the
replay found and three controls, every one on `workflow_dispatch` only — and `plan` runs the
Action offline on that change exactly as GitHub will (a depth-1 checkout of the test merge, the
event, the review API answered by the rule), again on the same head, and on the head with every
placed suggestion applied. The live run is the plan's procedure on a real pull request; `receipt`
reads GitHub's comments, annotations and commits back against it.

```
python -m benchmarks.harness_mutation.live_click files --out <dir>
python -m benchmarks.harness_mutation.live_click plan --out papers/harness/swallow12_plan.json
python -m benchmarks.harness_mutation.live_click receipt --plan papers/harness/swallow12_plan.json --observed papers/harness/swallow12_observed.json --clone <clone> --out papers/harness/swallow12_receipt.json
python papers/harness/swallow12_score.py
```

`papers/harness/RESULT_swallow12_the_click_live_2026_09_22.md` (VALID, 8/8): GitHub accepted the
16 suggestions the rule places and refused its 2; the re-run posted nothing; one "Commit
suggestions" made the 15 files the planned text byte for byte, a file without a final newline kept
without one; the Action then reported exactly the three controls, and GitHub kept all 19 of its
annotations (ten errors, nine warnings). `live_click.py` is frozen at `83be2100…` and pinned by its
test, which also holds the committed plan to the instrument.

## The frontier (SWALLOW-13)

The checks the first two repair stages leave. `styxx/ciaudit/repair_frontier.py` is the third stage,
written on SWALLOW-11's 49 unrepaired checks: hoist-substitution, background-liveness, no-exit-zero,
no-default-joined, each alone or with the `continue-on-error` removed, verified as the first two
are; for what none repairs, the routed and declared readings. `frontier.py` runs the product's
`--repair` path, unchanged, on SWALLOW-9's repositories at the tips SWALLOW-9 recorded, minus
SWALLOW-1's hundred and SWALLOW-11's sixty — 549 repositories no stage was designed on — each in its
own process, and runs stage 3 twice on every check it reaches.

```
python -m benchmarks.harness_mutation.frontier --build-population --out papers/harness/swallow13_population.json
python -m benchmarks.harness_mutation.frontier --population papers/harness/swallow13_population.json --work <dir> --out papers/harness/swallow13_receipt.json.gz --workers 2
python papers/harness/swallow13_score.py
```

`papers/harness/RESULT_swallow13_the_frontier_2026_09_22.md` (VALID, 5/8): 143 hand-written hidden
checks; stage 1 verifies 86, stage 2 13; of the 44 left, stage 3 verifies 13 (30%) in 9
repositories, twelve by hoist-substitution, a median of 5 lines; the same outcome twice. The three
stages together reach 112 of 143 (78%); the readings match 2 of the 31 left, most of which pass on
an empty list. `frontier.py` is frozen at `7b3c2b12…` and pinned by its test.

## The empty list (SWALLOW-14)

Two more edits in the third stage, written on SWALLOW-13's 44: **wait-list** (after a statement that
reads its list from `< <(…)`, `wait $! || exit $?`, so the list's command stops the step when it
fails instead of leaving an empty list) and **hoist-local** (the hoist with its strictness on its
own line). `empty_list.py` runs the product's `--repair` path, unchanged, on the AIDev repository
table less every repository an earlier cycle read — 5,945, each at the tip `ls-remote` gave at the
freeze — every stage-3 candidate tried and recorded, stage 3 twice.

**It runs each step's shell.** The audit's simulation is not confined: a path a script names outside
its temporary directory is the machine's, and `rm -rf $RUNNER_TOOL_CACHE/*` with the variable unset
is `rm -rf /*`. The first run lost its machine to `actions/setup-node`'s step. The scored run gave the
instrument `papers/harness/swallow14_sandbox.sh` as its interpreter, so each repository's process ran
as root in a throwaway overlay of the machine (bubblewrap, root needed):

```
python -m benchmarks.harness_mutation.empty_list --build-population --parquet repository.parquet --out papers/harness/swallow14_population.json.gz
<sandbox>/python3.v2 -m benchmarks.harness_mutation.empty_list --population papers/harness/swallow14_population.json.gz --work <dir> --out papers/harness/swallow14_receipt.json.gz --workers 3
python papers/harness/swallow14_score.py
<sandbox>/python3.v2 papers/harness/swallow14_after_fix.py --work <dir>
```

`papers/harness/RESULT_swallow14_the_empty_list_2026_09_22.md` (INVALID on its determinism gate —
one check of 268, a backgrounded command, moved between two runs of stage 3 — 6/8 reported, not
claimed): 1,066 hand-written hidden checks; stages 1 and 2 verify 798; SWALLOW-13's stage 154 of the
268 left; of the 114 it leaves, wait-list verifies 15 in 15 repositories; the local hoist verifies
the same 134 as the global one. `empty_list.py` is frozen at `9b40257a…` and pinned by its test.
