# RESULT — SWALLOW-14: the empty list — INVALID on its determinism gate, one check of 268 moving between two runs of stage 3; wait-list verifies 15 of the 19 empty lists it applies to, the local hoist loses nothing to the global one; and the audit runs a step's shell on the machine — the first run's machine was deleted by `actions/setup-node`'s `rm -rf $RUNNER_TOOL_CACHE/*`

Fathom Lab · 2026-09-22/23 · Scores the receipt `swallow14_receipt.json.gz` against the
preregistration frozen at sha256 `a34a06765aff3d25693a2e252dac38fb7a61e7a5d1c565681ab876571d93eb8d`.
Not amended. One run scored; a first run was lost with its machine, and a first attempt at the
second was stopped (§0).

Receipt: `papers/harness/swallow14_receipt.json.gz` (sha256 `9eba6f23…`), built by the frozen
instrument `benchmarks/harness_mutation/empty_list.py` (`9b40257a…`) calling the product at the
frozen hashes — `engine.py` `5c219887…`, `actions.py` `ecbcd4f9…`, `repair.py` `4282dbfa…`,
`repair_structural.py` `af46c492…`, `repair_frontier.py` `c10b9975…` · population
`swallow14_population.json.gz` (`21c0e00f…`) · scored by `swallow14_score.py` →
`swallow14_scored.json`.

**INVALID. 6 of 8 predictions HIT, reported, not claimed.** G-S14-5 is blocking: stage 3, run twice
on each of the 268 checks it was tried on, gave the same outcome on 267. The one that moved is a
step that backgrounds a command (§1). Everything else held: 5,941 of 5,945 repositories audited at
their recorded tips — 40,682 workflows, none capped — holding **1,066** hand-written hidden checks
in 500 repositories. SWALLOW-4's stage verifies **678**, SWALLOW-5's **120**; of the **268** left,
SWALLOW-13's stage verifies **154** (57%), and of the **114** it leaves, the two new edits verify
**15** (13%, under the 20% predicted) — all of them wait-list or its no-coe+ form, in 15
repositories, a median of 3 lines. The three stages together: **967 of 1,066 (91%)**. The local
hoist verifies exactly the 134 checks the global one does. What the first run found out about the
product matters more than any of it (§0).

## 0. What was done — and what the first run did to the machine

The product change, the instrument, the population, the preregistration and the scorer went to
GitHub at 20:07–20:11Z on the 22nd (`fathomlab-patch-55`, five commits, tree `176abd8f`), and the
run started at 20:12Z, three repositories at a time, as SWALLOW-13's had: as root, on the session's
machine.

At 20:38Z, with 1,470 of the 5,945 repositories read — no fetch failed, one recorded an error — a
step being simulated deleted most of the machine's root filesystem: `/bin`, `/etc`, `/home` (the
local repository, the clones, the per-repository results), the session's temporary directory and
its outputs. No process could start after it. Nothing of that run was read but its progress lines
and its error.

**The audit runs a step's shell on the machine.** `ci-audit` simulates a step by running its script
with its tools stubbed, in a temporary directory, with an empty environment — the simulated shell
sees none of the caller's variables. But the shell is real, and so is what the stubs do not cover:
`rm`, `mkdir`, a redirect. A path the script names outside the temporary directory is the
machine's, and an empty value can turn a scoped path into the root:

- `actions/setup-node`, `.github/workflows/proxy.yml`, step "Clear tool cache":
  `rm -rf $RUNNER_TOOL_CACHE/*`. The simulation does not set the runner's variables, so this is
  `rm -rf /*`.
- `djylb/nps`, `.github/workflows/pkg.yml`: `GOSDK=$(go env GOROOT)`, then `rm -r $GOSDK/*`. In the
  flavour where every tool answers nothing, this is `rm -r /*`.

`actions/setup-node` is repository 1,478 in the population's order; the last progress line I read
of the first run, a minute or two before it stopped, counted 1,470. Audited again after the run,
inside a throwaway copy of the machine (below), it deletes that copy's `/bin`, `/etc`, `/lib`,
`/opt`, `/root`, `/sbin`, `/usr` and `/var`, and the audit stops for want of `/bin/bash`; so does the
product's own command, `styxx ci-audit actions/setup-node`. On a developer's machine that command
deletes whatever its user can delete.
Two files the first run left show the same mechanism writing:
`/home/ubuntu/test-results/pentest/2026-09-22-release-x/sslscan_report.txt`, holding `x` — the stubs'
answer in one flavour — and the same path with `-release-` and nothing in it, the other: some
repository's pentest step, its `mkdir` and its redirect real. A fixture reproduced it before the
second run: a step `D="$(find-report-dir)"; rm -rf "$D"/home/claude/s14/canary/*` deleted the
canary when the audit ran as root.

This is almost certainly what happened at 17:55Z in SWALLOW-13's first run, which its RESULT put
down to the environment: all that run lost was under `/tmp`, where a step's `rm -rf /tmp/*`
reaches. An erratum is appended there. No released version carries `ci-audit` — PyPI's latest,
7.47.0, has no `styxx/ciaudit`, and `main` has neither it nor the Action; they exist only on this
program's unmerged branches.

**The error the first run recorded is the product's.** `OWASP/CheatSheetSeries` stopped with an
IndexError raised by wait-list: its masker turned a backslash-continued line's newline into a blank,
so every line after it was read one line off, and a second `< <(` list further down was looked for
on the wrong line. The exception stops the whole repository's audit. Scored as frozen; fixed after
scoring (§5).

**The second run.** A new machine at 23:00Z; the repository cloned at `fathomlab-patch-55`
(`da1e419f`), every frozen file at its frozen hash. The instrument and its command unchanged but for
the interpreter: `papers/harness/swallow14_sandbox.sh`, given to the instrument as its
`sys.executable`, so each repository's process — the one that runs the steps' shell — ran as root,
as before, but in a throwaway copy of the machine: the root filesystem under an overlay whose
writes land in a tmpfs of its own, an empty `/tmp`, a minimal `/dev`, its own process tree, a
container's file-owner capabilities and no others, none of the session's secrets in its
environment; only its result file copied out. Tested first on the fixture: the step deletes the
canary inside the copy, and the machine's canary is still there. Every earlier cycle's run shared
one machine across its repositories, so a step could see what another repository's steps had left;
in this one, each repository starts from the same machine.

A first attempt at the second run (23:06–23:14Z) ran each repository as an unprivileged user of its
own instead. It was stopped at 485 repositories, with nothing of it read but its errors, for two
reasons: an unprivileged user is refused writes root is allowed, so a simulated step would not do
what it did in every earlier cycle's runs; and the watcher I ran beside it killed three
repositories' processes as they exited, recording errors that were mine. Read after scoring: of the
485 repositories both runs read (those three left out), 477 are identical, and the 8 that differ
differ only in how many steps reach a tool (1 to 12 more as root) — not in a finding, a verdict or a
repair.

The run scored here started at 23:15Z from the first repository and finished at 00:52Z on the
23rd: 5,945 repositories, three at a time, 5,830 s. Two of them deleted their copy's `/bin` —
`actions/setup-node`, `djylb/nps` — and are recorded as errors; the machine was untouched.

## 1. Gates

| gate | bar | result |
|---|---|---|
| G-S14-1 instrument | the five test files green | pass — 48 passed |
| G-S14-2 frozen underneath | instrument, five product files, population at the frozen hashes | pass — all seven |
| G-S14-3 the population is read | ≥ 90% of 5,945 fetched and audited without error | pass — **5,941**: no fetch failed, none capped; 4 errors (2 wait-list's IndexError, 2 `/bin` deleted in the copy) |
| G-S14-4 enough to read | S13 residue ≥ 30 | pass — **114** |
| G-S14-5 determinism | stage 3 twice: the same candidates, verdicts, diffs, repair | **FAIL — 267 of 268** |
| G-S14-6 ledger | P1–P8 scored | pass |

**The check that moved** is `openshift-pipelines/pipelines-as-code`, `e2e.yaml`, "Run gosmee for
GitLab tests": `nohup gosmee client … > /tmp/gosmee-gitlab.log 2>&1 &`. Its one applicable
candidate, background-liveness (SWALLOW-13's), was loud; in the first run its healthy run in flavour
`x` came out changed, so it was not verified, and the second run came out otherwise (the receipt
records that the two differ, not how). Read after scoring, stage 3 on that check ten times in a
row: verified ten times. Ten more with the machine's two CPUs kept busy: seven. A backgrounded
command's healthy run depends on the machine's load — the package's docstring already names a step
that backgrounds a command among the few whose reading can differ between runs, and the scored run
had three repositories on two CPUs. The gate is blocking and the preregistration is not amended: the
cycle is INVALID, and what follows is reported, not claimed.

## 2. Predictions, scored

| | predicted | observed | |
|---|---|---|---|
| P1 the new edits reach | ≥ 20% of the S13 residue | **15 of 114 (13.2%)**: wait-list 14, no-coe+wait-list 1 | MISS |
| P2 the empty list | wait-list ≥ 50% of the wait class (≥ 10) | **15 of 19 (79%)** | HIT |
| P3 local loses nothing | global_not_local = 0 | **0** — global 134, local 134 | HIT |
| P4 local gains (a coin) | local_not_global ≥ 1 | **0** | MISS |
| P5 not one repository | wait-list in ≥ 3 repositories | **15** | HIT |
| P6 SWALLOW-13 replicates | ≥ 25% of the stage-3 population | **154 of 268 (57.5%)** | HIT |
| P7 the stages together | ≥ 80% of the read targets | **967 of 1,066 (90.7%)**: 678 + 120 + 169 | HIT |
| P8 small | median wait-list repair ≤ 3 lines | **3** (1–23) | HIT |

## 3. Reading

**The empty list is repaired where it is the whole story.** wait-list applies to 19 of the 114
checks SWALLOW-13's stage leaves and verifies 15, each in its own repository — linters and validators
on the files a list found (yaml lint, vale, hadolint, actionlint, license headers, changelog and
lockfile validators, vcpkg's deployment targets), tests (storage integration, canvas performance
sentinels, unit tests, HPCC's test runs, a Java release build) and a Go module matrix. Seven of the
fifteen are
FAIL_OPEN: the list decides whether a check runs at all. `trpc-group/trpc-agent-go` builds its test
matrix from `mapfile -t modules < <(find . -name go.mod …)`; when `find` fails the list is empty, the
step writes `matrix={"module":[]}`, and the tests run zero times, green. After the repair the
failure stops the step:

```diff
-          mapfile -t modules < <(find . -name go.mod \
+          mapfile -t modules < <(set -o pipefail; find . -name go.mod \
 …
             done)
+          [ -z "$!" ] || wait $! || { __rc=$?; [ "$__rc" -eq 1 ] || exit "$__rc"; }  # stop if the command that made the list failed
```

`n8n-io/n8n`'s canvas performance sentinels, `saleor/saleor-dashboard`'s actionlint on the
workflows a pull request touches and `linera-io/linera-protocol`'s hadolint are the same shape. Of
the four it does not verify, three end their list's command in `|| true` — `langflow-ai/langflow`'s
two bundle-guarded test steps and `malik-na/omarchy-mac`'s shellcheck — so the failure is an empty
success before the wait line can see it; the fourth, `mavlink/qgroundcontrol`'s docs lint, is the
masker's defect: after its continued lines the frozen wait-list read one line off, and the fixed one
verifies it (§5).

**The hoist is the stage's workhorse, and staying local costs nothing.** 134 checks in 87
repositories are verified by a hoist — 60 of them FAIL_OPEN — against 12 in SWALLOW-13's 549
repositories: `echo "name=$(tool)" >> $GITHUB_OUTPUT` (74), a one-test `if [ … $(…) … ]` (35),
`for x in $(…)` (14), `echo`/`printf` of a substitution (8), `export`/`local` (3), a median of 4
lines. Concentrated: `webiny/webiny-js` 15, `srl-labs/containerlab` 12. The local form verifies every
check the global one does and no other: here no healthy run needed the `|| true` or the lenient
line the whole-script `set -eo pipefail` takes away, so the gain the development set hinted at (the
coin, P4) did not come.

**What is left**, 99 checks in 78 repositories, sorted by rules I wrote after scoring (first match
wins; not preregistered): a step that decides whether a check runs — a flag written to
`GITHUB_OUTPUT` from a test on a tool's answer — 30; an empty list or answer that is "nothing to
check", the answer behind an `|| true` — 14; a default or a tolerated failure the stages do not
remove — 11; read, routed or declared — 11; the empty answer is the pass (`test -z "$(gofmt -l .)"`)
— 8; a negative check (`! tool | grep …`) — 7; other — 18. SWALLOW-13's residue said the same with
thirty-one checks: an empty answer that passes is beyond what a model whose tools answer nothing
can verify, and the decider needs the check downstream, not the step.

**Why P1 missed.** SWALLOW-13's stage replicated far past its own prediction (57%, against 25%
predicted and 30% on its own population), so what it leaves here is harder: 19 of its 114 are empty
lists, against 10 of 31 in the development set.

## 4. What this does not say

The cycle is INVALID; the numbers above are a reading of a run whose determinism gate failed on one
check, not claims. A verified repair is not a correct one: RED is loud, not correct, and the model's
healthy run is stubs. wait-list makes a list command's failure stop the step, and a list command
that fails in a real healthy run — a directory the author meant to be optional — now fails it too;
`wait $!` needs bash 4.4 to see the substitution. The population is a public dataset's popular
repositories, each read at one commit. This run's environment is not every earlier cycle's: each
repository started from the same untouched machine, where earlier runs let a step see what earlier
repositories' steps had left (and, twice, delete the run); the comparison in §0 bounds what the
privilege model moves, not what the shared machine did to earlier receipts. The four errored
repositories' checks are not in any count above.

## 5. What ships

wait-list and hoist-local in the third stage, as frozen. **One change after scoring**, to
`repair_frontier.py` (now `bd6eb5af…`): the masker keeps a backslash-continued line's newline; an
edit that cannot read a script does not apply to it, and the audit goes on (a stage-3 candidate
that raises is recorded as not applying, with the exception's name, instead of stopping the
repository's audit); and the wait line is written at the statement's indent, not its last line's.
Stage 3 run again with it on the 268 checks, each repository fetched again at the same tip in its
own throwaway copy (`swallow14_after_fix.py` → `swallow14_after_fix.json`): 171 verified against
169 — wait-list now reads `kortix-ai/suna`'s terraform validation and `mavlink/qgroundcontrol`'s
docs lint, whose continued lines had it looking one line off — two diffs change only by their wait
line's indent, and the two repositories that raised now audit: `OWASP/CheatSheetSeries`'s check
(its list's command ends in `|| true`) has no repair, `dust-tt/dust`'s is wait-list. P1 would still
miss: 17 of 114. Over the 38,415 `run:` scripts of SWALLOW-13's repositories (548 of 549 fetched
again), the frozen wait-list raises on 19 and applies to 120; the changed one raises on none and
applies to 156.

**Corrected in the product's own words**: the README, the Action's description and job summary,
the package docstring and the CLI's help no longer say "no code run": the step's shell runs, its
tools stubbed, and what the stubs do not cover is real — run it on a CI runner or in a container,
not on a checkout you do not trust. `swallow14_sandbox.sh` is the environment the second run ran
in. The erratum to SWALLOW-13's §0.

## 6. Next

**The executor, confined.** The simulated shell must not reach the machine: each step in a
throwaway root — this run's overlay, or a read-only root with a private temporary one — inside the
product, so `styxx ci-audit` on a repository nobody has read is safe to type. Until then the CLI's
own help says where to run it. Then the determinism the gate asks for, on steps that background a
command: a healthy run that does not depend on the machine's load. And the decider: thirty checks
here are a step that decides whether another runs, and the repair belongs to the check downstream.
