# PREREG — SWALLOW-9: the agent's pull request at the gate — 2,286 pull requests five coding agents opened, read as the gate would have read them

Fathom Lab · 2026-09-21 · Frozen before any pull request of the population is fetched or read.
Follows `RESULT_swallow8_who_writes_the_hidden_check_2026_09_21.md` (VALID, 4/7).

## Where this came from, stated before anything is measured

SWALLOW-7 built a gate: BASE, HEAD, and the workflows that changed between them, read with the
frozen SWALLOW-3 reading; it fires when a check arrives hiding its own failure. SWALLOW-8 ran it
over mainline commits and classed their authors by signature — a floor, with no ground truth.
AIDev (Zenodo record 16919272) has the ground truth: 71,677 pull requests that OpenAI Codex, GitHub
Copilot, Devin, Cursor and Claude Code opened on public repositories between 2024-12-24 and
2025-07-30, each with the agent GitHub attributed it to, the files its commits changed, and
whether a person merged it. So: run the gate at the pull request — the place SWALLOW-6 said to
catch the hidden check — on every one of those pull requests that touches a hand-written
workflow, and ask what a reviewer let through.

## 1. The instrument

`benchmarks/harness_mutation/agent_prs.py`, frozen at
`d591dc86712f43bde92aa48f7aa243b81bdff67e51cf11358d768f5f0474d233`, on the population file
`papers/harness/swallow9_population.json`, frozen at
`952509e5b731f352b8587ea46804c2cee2b86fc5effc72387f16216c768e459a` (built by the instrument from
AIDev's `pull_request.parquet`, `pr_commits.parquet` and `pr_commit_details.parquet`, whose
sha256s it records).

**Population.** Every AIDev pull request with at least one commit-detail row whose filename is a
hand-written workflow (`.github/workflows/*.yml|yaml`, not `*.lock.yml`): **2,286 pull requests
in 635 repositories** — OpenAI Codex 979, Copilot 566, Devin 559, Cursor 123, Claude Code 59;
1,603 merged (70%); 166 still open when the dataset was collected. One repository,
`MontrealAI/AGI-Alpha-Agent-v0`, holds 431 of them (19%), all Codex.

**HEAD** is `refs/pull/N/head` as GitHub keeps it, and counts only if it is one of the commits
the dataset lists for the pull request (a head that moved after collection is not audited).
**BASE** is the merge-base of HEAD with the branch the pull request was merged into: the first
parent of the merge commit when one exists; the default branch's tip otherwise (a squash, a
rebase, a closed pull request); for a fast-forward, the first commit below the pull request's own.
The clones are blobless and shallow (commits since 2024-06-01); when that boundary is in the way
— HEAD or its merge commit parentless, no base, a base among the pull request's own commits —
the default branch is deepened once, to 2022-01-01, and what is still in the way is skipped.
**The gate reads only the workflows the dataset says the pull request's commits changed**, and
the pull request is **suspect** — audited, but outside every prediction — when git's diff of one
of them adds or deletes more lines than the dataset counted for it: the sign that BASE is not
the branch the pull request was written against (a pull request into `develop`, say).

**Audited** = head found and listed by the dataset, base resolved, not suspect. Every prediction
is over audited pull requests. For each: the gate's records (SWALLOW-7's `audit_pair`, with the
repair SWALLOW-4 then SWALLOW-5 verifies on HEAD's text), and, for a merged pull request that
fires, what each check it brought is at the default branch's tip on the day of the run (still
hidden, repaired, no longer a check, removed, file gone). The receipt carries repository, number,
agent, state and dates, shas, and the gate's records — no author, no name, no address, and of
the pull request's text only the acknowledgement word the SWALLOW-6 regex matched in its title
and in its body (`ack`), computed when the population was built.

Frozen underneath: `differential.py` `91e4a4a7…`, `history.py` `93efb4a9…`, `faults.py`
`d26a407c…`, `action_checks.py` `0e723694…`, `repair.py` `7b9a1695…`, `repair_structural.py`
`77067a71…`.

## 2. What was known at the freeze

- The population's composition, from the dataset alone: Copilot's pull requests are the largest
  (2.9 workflow files and 93 added workflow lines each; 36% add a workflow; 50% carry an
  acknowledgement word, mostly in long templated bodies), Codex's the smallest (1.7 files, 29
  lines; 19% add; 19% acknowledge); Devin 2.7 files, 54 lines, 32% add, 18% acknowledge; Cursor
  2.4, 80, 25%, 26%; Claude Code 2.7, 162, 39%, 42%. Merge rates: Codex 78%, the others 63–65%.
  Acknowledgement words overall: title 3%, body 27%.
- SWALLOW-7's gate fired at 0.48% of the 21,569 mainline workflow-touching commits of 96
  repositories (2019–2026); SWALLOW-8 put the agent-signed commits at 0.76% and a person's at
  0.46% (0.60% in 2025–2026). Those 96 repositories are among the 635 here, and some of
  SWALLOW-8's 31 agent firing commits are merges of pull requests in this population
  (`MontrealAI`'s `codex/` merges among them); the overlap was not computed and no pull request
  was looked up.
- The instrument was exercised on the scripted history in `tests/test_harness_agent_prs.py`
  and, for the network path (clone, head fetch, base resolution, the deepening), on 14 pull
  requests of four repositories **outside the population** (their `files` empty, so the gate read
  nothing). One of the four repositories no longer exists. **No pull request of the population
  has been fetched or read.**

## 3. Predictions, committed now

All over audited pull requests. "Fires" = at least one newly hidden check (born hidden,
acquired, or rewritten into a check that hides).

**P1 — the rate** (calibrated on SWALLOW-8's 0.76% per agent commit; a pull request bundles
commits and files). The share of audited pull requests that fire lies in **[0.5%, 2.0%]**.

**P2 — the reviewer lets it through** (blind). Among audited pull requests closed at
collection, the merge rate of firing pull requests is at least **0.8×** the merge rate of
non-firing ones.

**P3 — Copilot's fires most** (blind; its pull requests are the largest). Among the agents with
at least 100 audited pull requests, Copilot has the highest fire rate.

**P4 — repairable** (blind). At least **60%** of newly hidden checks have a verified repair on
HEAD's text.

**P5 — the textbook line** (blind). At least **50%** of newly hidden checks carry
`continue-on-error: true` (on the step or its job).

**P6 — the pull request says so** (blind). Firing pull requests carry an acknowledgement word
(title or body) at least **1.5×** as often as non-firing pull requests of the same agent: the
observed count of acknowledging firing pull requests over the count expected from each agent's
non-firing rate.

**P7 — born, not turned** (blind). At least **70%** of newly hidden checks are `born hidden`.

**P8 — still there** (blind). Of the checks brought by merged firing pull requests whose file
still exists at the default branch's tip, at least **70%** are still hidden today.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S9-1 (instrument) | the base each scripted pull request gets, the gate on each, the suspect flag, the tip reading, the boundary rule, determinism, no name written | `tests/test_harness_agent_prs.py` green |
| G-S9-2 (population) | most of the population is audited | ≥ 80% of the 2,286 audited; ≤ 10% of the 635 repositories fail to clone |
| G-S9-3 (frozen underneath) | the instrument, the population file, the gate | `d591dc86…`, `952509e5…`, differential `91e4a4a7…`, history `93efb4a9…`, faults `d26a407c…` |
| G-S9-4 (ledger) | every prediction scored | HIT/MISS for P1–P8 |

G-S9-1 to G-S9-3 are blocking.

## 5. What would abandon this

G-S9-2: if GitHub no longer keeps the heads, or the shallow rule leaves most bases unresolved,
the audited set is not the population predicted on. The suspect rate is reported; a suspect rate
above 20% is stated as a limit of the base rule, not a gate.

## 6. Honest statement of what a passing SWALLOW-9 means

That, on pull requests GitHub attributes to a named coding agent, the gate fires at the stated
rate; that people merge the ones that fire about as readily as the ones that do not; that the
checks arrive with the mechanism and the repair SWALLOW-4/5 verify; and that the merged ones are
still there. It does not say the agent wrote the check unprompted — the person who asked for
"make CI pass" is not in the dataset. It does not compare agents to people: AIDev has no human
pull requests. The five agents' rates are rates of their pull requests as they came, with the
exposure each agent's pull requests have (P3 is a claim about pull requests, not per line). The
"still hidden today" reading is at the tip on the day of the run, recorded per repository.

## 7. Running it

```
python -m benchmarks.harness_mutation.agent_prs --population papers/harness/swallow9_population.json --work <clones> --workers 4 --out papers/harness/swallow9_receipt.json
python papers/harness/swallow9_score.py
```
