# PREREG — SWALLOW-10: the baseline — the same gate, the same repositories, the same months, on the pull requests people opened

Fathom Lab · 2026-09-22 · Frozen before any repository of the sample is cloned.
Follows `RESULT_swallow9_the_agents_pull_request_at_the_gate_2026_09_21.md` (VALID on its gates,
2/8; two runs stated).

## Where this came from, stated before anything is measured

SWALLOW-9 ran the gate on 2,123 pull requests five coding agents opened and found it firing on
0.8% of them — 1.2% of those that change a workflow — and a person merging the firing ones half
as often as the rest. Compared with what? AIDev has no human pull requests. GitHub does: every
pull request's head is kept under `refs/pull/N/head`, and numbers are given in order, so the
numbers between a repository's first and last agent pull request in the dataset's window are the
same months, and the ones the dataset attributes to no agent are the rest of that repository's
pull requests in those months. This cycle samples them, and puts them and the agents' through one
pipeline.

## 1. The instrument

`benchmarks/harness_mutation/human_prs.py`, frozen at
`7d764d5db800da3986a9b5ea4f5d620addd3a794d0bc037fa6b9c6eeebe3afe7`, on the sample file
`papers/harness/swallow10_sample.json`, frozen at
`854447b532015b9a5cb997aa0913f72c72d576bdcfb4307b0e1cef7f194de6ce` (built by the instrument
from AIDev's `all_pull_request.parquet`, the SWALLOW-9 receipt and its population file, whose
sha256s it records).

**The sample.** The 609 repositories SWALLOW-9 cloned that have an agent pull request created
in the window 2024-12-24 – 2025-07-30 in the dataset's full table (23,166 such pull requests:
Codex 15,997, Devin 3,417, Copilot 2,788, Cursor 779, Claude Code 185). Per repository: the
**span** is [first, last] of those numbers; the **agent fetch** is every SWALLOW-9 population pull
request of the repository plus a seeded sample (seed 10) of up to 50 other agent pull requests in
the span (8,879 in all); the **human sample** is drawn at run time — the numbers `refs/pull/*/head`
lists inside the span that the dataset attributes to no agent, a seeded sample of up to **150** per
repository (`random.Random(f"10:{repo}:human")`).

**One pipeline for every pull request.** A blobless, shallow (since 2024-06-01), single-branch
clone; then every branch of the remote (same filter, same boundary); then the heads.
- **group** — the dataset's agent for a pull request it lists; otherwise SWALLOW-8's rule
  (`authorship.classify`, frozen `c3fb6e42…`) on the head commit's author, subject and body:
  `agent-signed` (a signature the dataset did not attribute — excluded from the baseline, the
  floor), `automation` (a bot author), `human` (the rest). No name is written.
- **BASE** — the merge commit's first parent when a merge commit on any branch merged HEAD, else
  the newest commit of HEAD's ancestry that some branch not containing HEAD reaches (the fork
  point from the closest branch). When the shallow boundary leaves a pull request without a base,
  every branch is deepened once, to 2022-01-01; what is still without one is skipped.
- **touching** — git's three-dot diff between BASE and HEAD names a hand-written workflow
  (`.github/workflows/*.yml|yaml`, not `*.lock.yml`).
- **the gate** — SWALLOW-7's `audit_pair` (`differential.py` `91e4a4a7…`) on each changed
  workflow, with the repair SWALLOW-4 then SWALLOW-5 verifies.
- **merged** — a floor from git alone: a merge commit merging HEAD; or a commit on any branch
  whose subject GitHub's squash merge writes (`… (#N)`) or `Merge pull request #N`; or a commit
  on any branch with HEAD's own subject and author date (a rebase merge keeps both). A
  fast-forward leaves none. Checked against the dataset's truth on the agents' pull requests.

For the SWALLOW-9 population pull requests, the receipt also records whether this pipeline's
BASE equals SWALLOW-9's (the commit-list rule) and whether the gate's verdict agrees.

Frozen underneath: `agent_prs.py` `ec9ef750…`, `authorship.py` `c3fb6e42…`, `differential.py`
`91e4a4a7…`, `history.py` `93efb4a9…`, `faults.py` `d26a407c…`.

## 2. What was known at the freeze

- SWALLOW-9's result: 17 of 1,407 own-change agent pull requests fire (1.21%); 6 of their 24
  checks carry `continue-on-error`; 22 of 24 have a verified repair; 15 of 24 are born hidden;
  firing ones merged at 0.69× the rate of the rest. SWALLOW-8: human-signed mainline commits fire
  at 0.46% (0.60% in 2025–2026), agent-signed at 0.76%, 1.27× within the same years. SWALLOW-2/4:
  two thirds of the hidden checks in the wild hide by `continue-on-error`.
- The instrument was exercised on the scripted remote in `tests/test_harness_human_prs.py` and,
  for the network path, on three repositories **outside the sample** (20 human and 11 agent pull
  requests; none of the 31 touched a workflow, so no human pull request has been read at the
  gate; the merge signal recalled 7 of 7 merged agent pull requests there). **No repository of
  the sample has been cloned for this cycle.**

## 3. Predictions, committed now

"Touching" = audited and touching. "Agents" = the five dataset groups pooled. Human = the
`human` group. All rates are shares of touching pull requests that fire.

**P1 — the human rate** (blind). The human fire rate lies in **[0.4%, 1.6%]**.

**P2 — the agents fire more** (calibrated on SWALLOW-9 and SWALLOW-8). The agents' fire rate,
read by this pipeline, is at least **1.2×** the human rate.

**P3 — the textbook line is a person's** (blind). At least **40%** of human-brought newly hidden
checks carry `continue-on-error: true`.

**P4 — repairable** (blind). At least **60%** of human-brought checks have a verified repair.

**P5 — the reviewer, again** (blind). Among touching human pull requests, the firing ones are
merged (by the git signal) at most **0.9×** as often as the non-firing ones.

**P6 — born, not turned** (blind). At least **50%** of human-brought checks are born hidden.

**P7 — bots do not** (blind). At most **0.1%** of touching `automation` pull requests fire.

**P8 — the selection's floor is small** (blind). Of the agent pull requests the dataset does not
list as touching a workflow (the up-to-50-per-repository sample), at most **5%** touch one by
git's diff.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S10-1 (instrument) | the sample rule, the groups, the bases, the merge signal, the gate, the agreement with SWALLOW-9, the deepen rule, determinism, no name written | `tests/test_harness_human_prs.py` green |
| G-S10-2 (population) | enough of the baseline is read | ≥ 500 repositories audited; ≥ 600 touching human pull requests; ≤ 10% of the 609 repositories fail to clone |
| G-S10-3 (frozen underneath) | the instrument, the sample file, the gate | `7d764d5d…`, `854447b5…`, differential `91e4a4a7…`, agent_prs `ec9ef750…`, authorship `c3fb6e42…` |
| G-S10-4 (one pipeline, two rules) | this pipeline's BASE agrees with SWALLOW-9's commit-list BASE | ≥ 90% of the compared SWALLOW-9 population pull requests; the gate's verdict agrees on ≥ 95% |
| G-S10-5 (the merge signal) | recall on the dataset's truth | ≥ 85% of the dataset-merged agent pull requests recalled; ≤ 5% of the not-merged flagged |
| G-S10-6 (ledger) | every prediction scored | HIT/MISS for P1–P8 |

G-S10-1 to G-S10-5 are blocking.

## 5. What would abandon this

G-S10-4: if the closest-branch rule and the commit-list rule disagree on a tenth of the agents'
bases, the two cycles are not reading the same thing and the comparison in P2 is not one.
G-S10-5: if the git signal misses the dataset's merges, P5 is not about merging.

## 6. Honest statement of what a passing SWALLOW-10 means

That, in the repositories where these agents work and in the months they worked there, the
pull requests people opened bring a check that hides its own failure at the stated rate, and
the agents' at the stated multiple of it — read by one pipeline, one base rule, one gate. The
human group is a remainder: whoever opened a pull request whose head commit carries no agent
signature and no bot author, which includes people using agents that left no signature. The
sample is capped per repository, so a repository with thousands of pull requests weighs as
much as one with 150; the agents' pull requests are the dataset's, not capped the same way, and
the receipt lets both be re-weighted. The merge signal is a floor and is scored against the
dataset where the dataset knows.

## 7. Running it

```
python -m benchmarks.harness_mutation.human_prs --sample papers/harness/swallow10_sample.json --receipt9 papers/harness/swallow9_receipt.json.gz --work <clones> --workers 3 --out papers/harness/swallow10_receipt.json
python papers/harness/swallow10_score.py
```
