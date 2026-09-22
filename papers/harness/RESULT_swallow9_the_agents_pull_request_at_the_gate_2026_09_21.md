# RESULT — SWALLOW-9: the agent's pull request at the gate — of 2,123 pull requests five coding agents opened that touch a workflow, 17 bring a check that hides its own failure (0.8%); a person merged nine of them, half, against three quarters of the rest; 22 of the 24 checks have a verified repair; half of the ones merged into the default branch are gone or loud today

Fathom Lab · 2026-09-22 · Scores the receipt `swallow9_receipt.json.gz` against the
preregistration frozen at sha256 `6b81763692cc2de0bc461aae168934998c58e28ef4ce683f19cbd56e7e937bd5`.
The prereg text was not amended. Two runs: the instrument was amended once between them (its base
rule; deviation 1), and both runs are stated.

Receipt: `papers/harness/swallow9_receipt.json.gz` (sha256 of the JSON `fabfc00a…`, recorded by the scorer;
every pull request of the population with its shas, base, the gate's records and, for a merged
pull request that fires, the reading at the default branch's tip — no author, no name, no
address, no text beyond the acknowledgement word) · instrument
`benchmarks/harness_mutation/agent_prs.py`, sha256 `ec9ef750…` (run 2; run 1 at `d591dc86…`) ·
population `swallow9_population.json.gz` (sha256 of the JSON `952509e5…`), built from AIDev (Zenodo
record 16919272) · 2,286 pull requests in 635 repositories; 2,123 audited · 28 minutes, 3 workers ·
scored by `swallow9_score.py`.

**VALID on its gates. 2 of 8 predictions HIT** (P1, P4); P2, P3, P5, P6, P7, P8 MISS — four of
the six in the direction opposite to the one predicted. **The gate fires on 17 of the 2,123
audited pull requests (0.80%; P1 predicted 0.5–2%), bringing 24 checks that hide their own
failure** — 15 born hidden, 9 existing checks made hidden. **A person merged 9 of the 17 (53%),
against 77% of the pull requests that do not fire** (P2 predicted no less than 0.8×: MISS at
0.69×). **22 of the 24 checks have a verified repair** (P4), 6 of them one line. The shape is not
the mainline's: only 6 of 24 carry `continue-on-error` (P5 predicted half: MISS); 6 are `|| true`,
4 are `|| echo …` defaults, 2 are `set +e`, 3 fail open. **Claude Code's pull requests fire most
(3 of 49; one closed pull request brought 7 checks, `npm run lint || echo "Linting needs
fixing"` among them), Cursor's most among the agents with ≥ 100 (2 of 117), Copilot's least
(3 of 516)** — P3 named Copilot: MISS. Firing pull requests do not say so more often than the
others (P6, 1.18× against 1.5×). Of the six checks merged into a default branch whose file is
still there, two are still hidden, two were repaired, one is no longer a check, one was removed
(P8 predicted 70% still hidden: MISS at 33%).

## 0. What was done

For every AIDev pull request one of whose commits changed a hand-written workflow — 2,286, by
OpenAI Codex (979), Copilot (566), Devin (559), Cursor (123) and Claude Code (59), opened between
2024-12-24 and 2025-07-30 — the instrument fetched `refs/pull/N/head` into a shallow, blobless,
single-branch clone of the repository, took as BASE what the dataset's own commit list implies
(the parents of the pull request's commits that are not its commits: the base branch as the pull
request last saw it, whatever branch that is), and ran SWALLOW-7's gate on the workflows the
dataset says the pull request changed. Audited = head found and listed by the dataset, base
resolved, and git's diff no larger than the dataset's count for each file (else *suspect*). For
each merged pull request that fires, the check was looked for at the default branch's tip on the
day of the run.

## 1. Gates

| gate | bar | result |
|---|---|---|
| G-S9-1 instrument | seven scripted pull requests: the base each gets, the gate on each, the suspect flag, the tip reading, the boundary rule, determinism, no name written | pass — `tests/test_harness_agent_prs.py`, 8 passed |
| G-S9-2 population | ≥ 80% of 2,286 audited; ≤ 10% of 635 repositories fail to clone | pass — 2,123 audited (92.9%); 12 repositories failed to clone (8 gone or private, 4 with no commit since 2024-06-01; 66 pull requests) |
| G-S9-3 frozen underneath | the instrument, the population file, the gate | pass at the amended hash: instrument `ec9ef750…` (the prereg names `d591dc86…`, run 1), population `952509e5…`, differential `91e4a4a7…`, history `93efb4a9…`, faults `d26a407c…` |
| G-S9-4 ledger | P1–P8 scored | pass |

Not audited: 3 heads GitHub no longer has; 22 heads that moved after collection; **60 pull
requests whose dataset commit list is capped at 30 and does not reach the head** (a limit of the
dataset, stated here, not in the prereg); 1 without a base; 11 suspect (0.5%). 3 bases were
ambiguous (two candidates, neither the other's ancestor; the newest the default branch reaches
was taken); none of those fires.

**Deviation 1 — the instrument was amended once after the freeze; the prereg text was not.**
The preregistration names `agent_prs.py` at `d591dc86…`, whose BASE was the merge-base of the
head with the *default* branch (or the merge commit's first parent). **Run 1** (that file) was
VALID on its gates and scored 5/8: 22 firing pull requests of 1,999 audited (1.1%), 24 checks.
Seven of those 22 were Copilot pull requests to `microsoft/genaiscript`, each "bringing" the same
`lint:check` step with `continue-on-error: true` — **a step a commit to that repository's `dev`
branch had added three days before the first of them**. The pull requests targeted
`dev`; the instrument compared them with `main`, where the step does not exist, and the suspect
flag did not catch it because the dataset's line counts include the branch's merges of `dev`.
A post-hoc re-basing of all 22 against every branch of their repositories (non-gating) kept 15
and dropped the 7. Amended (`ec9ef750…`): BASE is what the pull request's own commit list
implies — GitHub's list is the `base...head` set, so the parents of its commits that are not its
commits are the base branch as last merged into it — which needs no branch and is right for a
pull request into `dev`; the head fetch uses the clone's own filter (a different one made git
fetch every tree one at a time: run 1 took 63 minutes, run 2 took 28); a parent behind the shallow
boundary is asked for by name before the branch is deepened. **Run 2** (this receipt) audited
2,123 and fires on 17: the 15 that survived the re-basing and two that run 1 could not base
(`ruvnet/claude-flow#267`, seven checks; `julep-ai/julep#1395`). The amendment moved three
predictions: run 1 scored P3 HIT (Copilot 1.99%, on the seven), P5 HIT (14 of 24 with
`continue-on-error`, seven of them the same line) and P7 HIT (19 of 24 born); run 2 scores all
three MISS. Both runs are stated; the second is the result.

**Deviation 2 — the population as frozen is a third larger than the pull requests that change a
workflow.** The dataset lists, for each commit of a pull request, the files that commit changed —
and for a commit that merges the base branch into the pull request, those are the base branch's
changes. **716 of the 2,123 audited pull requests (34%) have no workflow change of their own**
between BASE and HEAD: git's three-dot diff is empty on every workflow the dataset named. They
are audited (the prereg's definition) and cannot fire. On the 1,407 that do change a workflow the
rate is **1.21%** (17 of 1,407): Claude Code 3 of 39 (7.7%), Cursor 2 of 71, Devin 3 of 262,
Copilot 3 of 277, Codex 6 of 758 (0.8%). Every prediction is scored on the frozen denominator.

## 2. Predictions, scored (run 2)

| | predicted | observed | |
|---|---|---|---|
| P1 the rate (calibrated) | fire rate in [0.5%, 2.0%] | **17 of 2,123, 0.80%** (1.21% of those with a workflow change of their own) | HIT |
| P2 the reviewer lets it through (blind) | merge rate of firing ≥ 0.8× non-firing, among pull requests closed at collection | **9 of 17 (53%) vs 1,514 of 1,965 (77%), 0.69×** | MISS |
| P3 Copilot's fires most (blind) | highest fire rate among agents with ≥ 100 audited | **Cursor 1.71%, Codex 0.63%, Devin 0.61%, Copilot 0.58%**; Claude Code 6.1% on 49 | MISS |
| P4 repairable (blind) | ≥ 60% of new hidden checks with a verified repair | **22 of 24 (92%)** — 9 strict shell, 6 `continue-on-error` removed, 6 default removed, 1 guard | HIT |
| P5 the textbook line (blind) | ≥ 50% carry `continue-on-error: true` | **6 of 24 (25%)** | MISS |
| P6 the pull request says so (blind) | acknowledging firing ≥ 1.5× expected from each agent's non-firing rate | **6 of 17 against 5.07 expected, 1.18×** | MISS |
| P7 born, not turned (blind) | ≥ 70% born hidden | **15 of 24 (62.5%)**; 9 acquired (4 `|| true`, 3 `continue-on-error`, 2 defaults; 4 also gated) | MISS |
| P8 still there (blind) | ≥ 70% of checks merged firing pull requests brought, whose file exists at the tip, still hidden | **2 of 6 (33%)**: 2 repaired, 1 no longer a check, 1 removed; 3 more whose file is gone | MISS |

## 3. Reading

**Where the gate fires.** 17 pull requests in 15 repositories, from 2025-04 on (none of the
171 audited from 2024-12 to 2025-03 fires; 1 of 82 in April, 2 of 398 in May, 7 of 647 in June,
7 of 825 in July). Nine were
merged, eight closed; none of the 141 still open at collection fires. One pull request brought
seven checks at once — `ruvnet/claude-flow#267`, Claude Code, closed: three `|| true` and two
`|| echo "… needs fixing"` on existing lint, type-check and coverage steps, and two new steps
written that way — the batch shape SWALLOW-7 saw on mainlines, here in a pull request a person
did not merge. `facebook/idb#882` (Claude Code, closed) brought the same fail-open notarization
check into two release workflows. The other 15 pull requests bring one check each.

**What the check looks like.** 15 born hidden, 9 acquired. Of the 24: 6 `continue-on-error:
true` (SWALLOW-8 found the agent writing that line on the mainline; here it is a quarter), 6
`|| true`, 4 `|| echo …` defaults, 2 scripts opening with `set +e`, 3 fail open (a query whose
empty answer is the green path — `gh api … orgs/…/members`, a coverage script's changed-files
list, a prerelease check), 3 swallowed further down their scripts. 22 have a verified repair: 9 by
the strict shell, 6 by removing `continue-on-error` (all one line), 6 by removing the default, 1
by SWALLOW-5's guard; the 2 without are the two fail-open queries SWALLOW-5 cannot verify with an
empty-answer flavour. Six audited pull requests *removed or repaired* a hidden check, and the
gate reports those too: two of them are the agent's next pull request to the same repository
repairing the check its last one brought (`docusaurus-openapi-docs#1169` → `#1209`, 23 days;
`MontrealAI#3432` → `#3463`, the next day), and `claude-flow#267` removed one while bringing seven.

**The reviewer does not let it through as readily.** The prereg guessed a hidden check goes
unnoticed; the merge rate says otherwise: 53% of firing pull requests were merged against 77%
of the rest (on the pull requests with a workflow change of their own, 53% against 79%). Claude
Code's three firing pull requests were all closed; Codex's six, four merged. This is a merge
rate, not a review verdict: the eight closed pull requests were closed for whatever reason their
repositories had, and the gate reads only the check. What it does say is that the pull request
that brings a hidden check is not the pull request that sails through.

**By agent, on ground truth.** Claude Code 3 of 49 (6.1%; 3 of 39 with a workflow change of
their own, 7.7%), Cursor 2 of 117 (1.7%), Codex 6 of 953 (0.6%; 4 of 522 outside
`MontrealAI/AGI-Alpha-Agent-v0`, whose 431 fire twice), Devin 3 of 488 (0.6%), Copilot 3 of 516
(0.6%). The prereg named Copilot because its pull requests are the largest; they are also the
ones most often carrying the base branch's changes rather than their own (239 of Copilot's 516
audited change no workflow themselves) — the exposure was the dataset's, not the agent's. Claude
Code's rate stands on three pull requests, two of them to one owner's repositories; it is stated,
not claimed.

**And afterwards.** The 9 merged firing pull requests brought one check each. At the default
branch's tip today, 3 are in files the branch no longer has (`celestiaorg/docs` ×2, `julep`), 2
are still hidden (`valkey-glide`'s `find … || echo "No .class files found!"`, `mcp-get`'s
`npm run lint || true`), 2 were repaired (`teaxyz/chai`'s `continue-on-error` removed;
`docusaurus-openapi-docs`'s fail-open membership check, by the next Codex pull request), 1 is no
longer a check, 1 was removed (`MontrealAI#3432`'s, repaired by the next Codex pull request and
later dropped). SWALLOW-6 found hidden checks old and rarely made loud; the ones agents brought
in 2025 were mostly not left alone. n = 9.

**The gate's cost.** Median 0.11 s per pull request with a workflow change (p90 0.56 s), the
reading memoised per repository; 28 minutes for the population with 3 workers, most of it git.

## 4. What this does not say

It does not compare agents with people: AIDev has no human pull requests, and SWALLOW-8's 0.46%
per human mainline commit is a different unit. It does not say who wrote the `|| true`: the
person who asked for green CI is not in the dataset, and a merged pull request was a person's
decision. The five agents' rates are rates of their pull requests as they came, on the frozen
denominator that includes a third with no workflow change of their own, with one repository
holding a fifth of Codex's. The "still hidden today" reading is at the default branch's tip on
2026-09-22, and a pull request merged into another branch shows as "file gone" or "removed"
there. The rates are the frozen SWALLOW-3 reading's, at the pull request's head as GitHub keeps
it; a head that moved after collection, or whose commit list the dataset capped, is not read.

## 5. What ships

`agent_prs.py`, frozen at `ec9ef750…` and pinned by its test file; the population file and the
receipt. Nothing in `styxx` changes: `styxx ci-audit --base` is the gate these pull requests were
read with.

## 6. Next

The 716 pull requests with no change of their own are a population for the reverse question —
what the base branch brought *into* the agent's pull request — and the 60 with capped commit
lists want the head fetched by number from the dataset's `html_url`. A human baseline needs a
dataset with one.
