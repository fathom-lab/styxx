# RESULT — SWALLOW-10: the baseline — in the same 609 repositories and the same months, the pull requests people opened bring a check that hides its own failure at 0.60%; the agents' at 1.22%, twice the rate, by one pipeline — INVALID on its merge-signal gate as frozen, 8/8 reported, not claimed

Fathom Lab · 2026-09-22 · Scores the receipt `swallow10_receipt.json.gz` against the
preregistration frozen at sha256 `aed0c1b4b9159036ccdfc76bb4a9e544e3fa9f1b4b92c422c6787920a5586cac`.
Not amended. One run.

Receipt: `papers/harness/swallow10_receipt.json.gz` (sha256 of the JSON `485f6c5b…`, recorded by the scorer;
every sampled pull request of every repository with its group, shas, base, merge signal and the
gate's records — no author, no name, no address, no text) · instrument
`benchmarks/harness_mutation/human_prs.py`, sha256 `7d764d5d…` · sample
`swallow10_sample.json.gz` (sha256 of the JSON `854447b5…`) · 609 repositories, 35,307 pull
requests fetched, 34,647 audited · 1 h 44 min, 5 workers · scored by `swallow10_score.py`.

**INVALID on G-S10-5 as frozen.** The gate holds the git merge signal against the dataset's
merged flag and allows 5% false positives; it flagged 221 of 3,397 dataset-not-merged agent pull
requests (6.5%). **177 of the 221 were open when the dataset was collected and have been merged
since** — the dataset's flag is a snapshot, and the frozen gate did not say so. On the pull
requests the dataset saw closed, the signal flags 44 of 2,512 (1.75%) and recalls 5,042 of 5,264
merged (95.8%) — printed by the scorer as non-gating, and not what the gate was frozen on.
**8 of 8 predictions HIT — reported, not claimed.**

What the receipt says, under that heading: **of 1,989 pull requests people opened that change a
hand-written workflow, 12 bring a check that hides its own failure (0.60%), 32 checks; of 1,479
the agents opened, read by the same pipeline, 18 do (1.22%) — 2.0×** (P2 predicted ≥ 1.2×).
Five of the twelve human firings are in one repository whose owner is a heavy agent user;
without it the human rate is 0.35% and the multiple 3.4. **The person's hidden check is the
textbook one: 19 of 32 carry `continue-on-error` (the agents' 6 of 20), 31 of 32 are born
hidden, 31 of 32 have a verified repair.** People merge their firing pull requests at 0.86× the
rate of the rest (agents' 0.80× by the same signal). **Not one of 249 bot pull requests that
touch a workflow fires.** Pull requests the dataset does not attribute to an agent but whose head
commit carries an agent's signature — 134 touching — fire at 3.7% (5 of 134, 20 checks; not
scored, stated). The closest-branch base agrees with SWALLOW-9's commit-list base on 96.5% of the
agents' pull requests and the gate's verdict on 100%.

## 0. What was done

For each of the 609 repositories: a blobless shallow clone, every branch, and the heads of the
sampled pull requests — every SWALLOW-9 population pull request, up to 50 other agent pull
requests, and up to 150 of the numbers in the span between the repository's first and last agent
pull request that the dataset attributes to no agent (26,428 of 97,940 such numbers; the cap
binds in 111 repositories). Each pull request was grouped (the dataset's agent; else SWALLOW-8's
rule on the head commit: `agent-signed`, `automation`, `human`), based (the merge commit's first
parent when a merge commit on any branch merged the head, else the fork point from the closest
branch), diffed against its base for hand-written workflows, and, when one changed, read by
SWALLOW-7's gate with the repair SWALLOW-4/5 verify. Merged, from git alone: a merge commit, a
squash subject `… (#N)` / `Merge pull request #N`, or a branch commit with the head's subject and
author date.

## 1. Gates

| gate | bar | result |
|---|---|---|
| G-S10-1 instrument | the scripted remote: sample rule, groups, bases, merge signal, gate, agreement with SWALLOW-9, deepen rule, determinism, no name | pass — `tests/test_harness_human_prs.py`, 6 passed |
| G-S10-2 population | ≥ 500 repositories audited; ≥ 600 touching human pull requests; ≤ 10% clone failures | pass — 596 repositories with an audited pull request; 1,989 touching human pull requests; 0 of 609 failed to clone |
| G-S10-3 frozen underneath | instrument `7d764d5d…`, sample `854447b5…`, differential `91e4a4a7…`, agent_prs `ec9ef750…`, authorship `c3fb6e42…` | pass |
| G-S10-4 one pipeline, two rules | base agrees on ≥ 90% of the compared SWALLOW-9 pull requests; the gate on ≥ 95% | pass — **1,977 of 2,048 (96.5%)**; **2,037 of 2,037 (100%)** |
| G-S10-5 the merge signal | recall ≥ 85%; ≤ 5% of the dataset-not-merged flagged | **FAIL as frozen** — recall 95.8%; flagged 221 of 3,397 (6.5%), of which 177 open at collection; 1.75% among the closed (non-gating) |
| G-S10-6 ledger | P1–P8 scored | pass |

Not audited: 632 without a base (336 human, 127 Codex, 59 agent-signed, 47 automation, 25 Claude
Code — a fork point below the boundary in a repository whose deepening did not reach it, or a
head sharing no history with any branch; 46 repositories were deepened) and 28 heads GitHub no
longer has (all Devin). `ruvnet/claude-flow` — SWALLOW-9's seven-check firing `#267` among them —
has no base for any of its 107 sampled pull requests: its heads share no history with its
branches above the boundary, and the deepening did not reach one.

**Deviation — none in the instrument.** The frozen G-S10-5 compares the signal with a snapshot
of the dataset as if it were current; the RESULT keeps the gate's verdict and prints the
restricted number as exploratory.

## 2. Predictions, scored — reported, not claimed

| | predicted | observed | |
|---|---|---|---|
| P1 the human rate (blind) | in [0.4%, 1.6%] | **12 of 1,989, 0.60%**; 32 checks | HIT |
| P2 the agents fire more (calibrated) | agents ≥ 1.2× human, this pipeline | **18 of 1,479, 1.22%; 2.02×** — Codex 6 of 770, Copilot 4 of 304, Devin 3 of 293, Cursor 2 of 77, Claude Code 3 of 35 | HIT |
| P3 the textbook line is a person's (blind) | ≥ 40% of human-brought checks `continue-on-error` | **19 of 32 (59%)**; agents 6 of 20 | HIT |
| P4 repairable (blind) | ≥ 60% of human-brought checks with a verified repair | **31 of 32 (97%)** — 18 `continue-on-error` removed, 7 strict shell, 4 guards, 2 defaults removed; agents 18 of 20 | HIT |
| P5 the reviewer, again (blind) | firing human pull requests merged ≤ 0.9× non-firing | **8 of 12 (67%) vs 1,536 of 1,977 (78%), 0.86×**; agents 10 of 18 vs 1,020 of 1,461, 0.80× | HIT |
| P6 born, not turned (blind) | ≥ 50% of human-brought checks born hidden | **31 of 32 (97%)**; agents 16 of 20 | HIT |
| P7 bots do not (blind) | ≤ 0.1% of touching automation pull requests fire | **0 of 249** | HIT |
| P8 the selection's floor (blind) | ≤ 5% of the agents' unlisted pull requests touch a workflow | **94 of 6,540 (1.4%)**, none fires | HIT |

## 3. Reading

**The multiple.** 0.60% against 1.22%: in the repositories where these agents work, in the
months they worked there, an agent's pull request that changes a workflow brings a check that
hides its own failure twice as often as a person's. SWALLOW-8 put the same ratio at 1.27 within
2025–2026 on mainline commits by signature; here it is ground truth on the agent side and one
pipeline on both, and it is 2.0. Weighting repositories equally instead of pull requests, the
human rate is 0.54%. The human firings sit in 8 repositories; **five of the twelve are
`ruvnet/ruv-FANN`** — 19 of the 32 checks, the same `comprehensive-testing.yml` steps born hidden
in pull request after pull request (`#34`, `#44`, `#46`, `#48`), a repository whose owner's other
repository is `claude-flow` and whose Claude Code and agent-signed pull requests fire too. The
human group is a remainder: a pull request whose head commit carries no agent signature is a
person's under the rule, and that repository is where the floor shows. Without it: 7 of 1,976
(0.35%), a multiple of 3.4. Both numbers are the population's; the prereg scored the first.

**What a person's hidden check looks like.** Born hidden (31 of 32), with `continue-on-error:
true` on the step (19), or `|| true` (6), or a query whose empty answer is the green path (4):
`vscode`'s four "Diagnostics before/after smoke" steps made non-blocking, `vibetunnel`'s four
lint steps, `nx`'s "Reset iOS Simulators", `AMICI`'s SBML import test. The agents' 20 checks in
this pipeline: 6 `continue-on-error`, 6 swallowed further down the script, 3 fail-open queries,
2 `|| true`, 2 `set +e`, 1 default — the shape SWALLOW-9 found, again. A person writes the
textbook line; the agent writes the workaround. 31 of 32 human-brought checks and 18 of 20
agent-brought have a verified repair.

**The signature that the dataset did not attribute.** 986 pull requests outside the dataset's
attribution carry an agent's signature in their head commit (a `Co-authored-by`, a "Generated
with", an agent's branch name); 134 touch a workflow and **5 fire (3.7%)**, bringing 20 checks —
`FinAegis/core-banking-prototype-laravel#76` alone brings 14 across seven workflows, `|| true`
on its security and test steps. Not scored; the group was defined at the freeze as the floor's
visible part.

**Bots.** 249 dependency, release and action-bot pull requests touch a workflow; none brings a
hidden check — SWALLOW-8's P3 at the pull request.

**The reviewer.** By the git signal, people merge their firing pull requests at 0.86× the rate
of their other workflow pull requests, and the agents' at 0.80× (SWALLOW-9, by the dataset's
flag: 0.69×). The direction holds in both groups; the size is a merge rate, not a verdict.

**One pipeline, two rules.** For the 2,048 SWALLOW-9 pull requests read here, the closest-branch
base equals the commit-list base in 1,977 (96.5%); the 71 differences are all under the
closest-branch rule (18 of them pull requests whose branch still exists), none of the 71 fires
under either rule, and the gate's verdict agrees on 2,037 of 2,037. SWALLOW-9's number stands
under a second base rule.

**Cost.** 35,307 heads fetched, 3,851 pull requests read at the gate, median 1.18 s each
(the reading memoised per repository), 1 h 44 min with 5 workers — GitHub's `refs/pull/N/head`
and one blobless clone per repository, no API, no token.

## 4. What this does not say

It does not say who typed the `|| true`: the human group is whoever left no signature, and one
repository shows what that floor can hide. It does not compare like sizes: the human sample is
capped at 150 per repository and the agents' is the dataset's, so a repository with thousands of
pull requests weighs like one with 150 on the human side and by its agent pull requests on the
other (the receipt carries both for re-weighting; the repository-weighted human rate is given).
The merge signal is from git and misses a fast-forward and any rebase that changed the subject;
its accuracy is known on the agents' pull requests the dataset saw closed and assumed on the
others. The touching share (9.2% of human pull requests, 17.1% of the agents' as sampled) is not
a comparison: the agent sample is enriched with the pull requests SWALLOW-9 already knew touch a
workflow, by design. The gate is the frozen SWALLOW-3 reading; RED is loud, not correct.

## 5. What ships

`human_prs.py`, frozen at `7d764d5d…` and pinned by its test file; the sample and the receipt.
And in `styxx`: **`styxx ci-audit OWNER/REPO --pr N`** — the same gate on one pull request,
read from GitHub's own test-merge ref (`refs/pull/N/merge` against its first parent, the base
branch as GitHub would merge into), no checkout, no token; without the merge ref (closed, or
conflicting) the head against its merge-base with the default branch, and the card says which
reading it is. Run on this repository's own open pull requests: #151, 2 workflows changed,
nothing newly hidden; #153, no workflow changed; 7–10 s each.

## 6. Next

The 986 agent-signed pull requests are a third population the dataset does not know — the
signature list applied to every pull request of a repository, not only the agents' accounts —
and their 3.7% wants a prereg of its own. The merge signal wants the dataset's snapshot date as
a parameter. And the human sample can be re-weighted by repository against the agents' to give
the multiple both ways.
