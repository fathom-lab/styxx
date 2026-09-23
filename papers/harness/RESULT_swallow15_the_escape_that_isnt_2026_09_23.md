# RESULT — SWALLOW-15: the escape that isn't — on 5,945 repositories, two that delete `/` in the simulation, the audit's own Landlock confinement kept the bare machine, no canary was ever touched, and every hand-written boundary probe was neutralised; INVALID on its equivalence gate, where confinement suppresses a finding in 12 repositories whose step writes to `/tmp`

Fathom Lab · 2026-09-23 · Scores the receipt `swallow15_receipt.json.gz` against the preregistration
frozen at sha256 `9c429903768701c88c3b8d026147b8ea9f4006a48ff992c690a9f4c03186348e`. Not amended. One run
scored; the whole run on the bare machine.

Receipt: `papers/harness/swallow15_receipt.json.gz` (sha256 `6f449092…`), built by the frozen
instrument `benchmarks/harness_mutation/confined.py` (`7014bc02…`) calling the product at the frozen
hashes — `engine.py` `5c219887…`, `confine.py` `46a1b40b…`, `__init__.py` `5c021afe…`, `action.py`
`d7ce8158…`, `differential.py` `dc79fc97…`, `repair.py` `4282dbfa…`, `repair_structural.py`
`af46c492…`, `repair_frontier.py` `bd6eb5af…` · population `swallow14_population.json.gz`
(`21c0e00f…`) · compared receipt `swallow14_receipt.json.gz` (`9eba6f23…`) · battery
`swallow15_battery.json` (`9db078c1…`) · scored by `swallow15_score.py` → `swallow15_scored.json`.

**INVALID. 5 of 7 predictions HIT.** The safety of the machine is not in question: on **5,945**
repositories audited on the **bare machine** — no overlay, no throwaway copy — the product's own
Landlock confinement was the only thing between a simulated `rm -rf /*` and the host, and **no canary
was ever touched** (P1), the **two repositories that delete their `/bin` unconfined**
(`actions/setup-node`, `djylb/nps`) audited with no error and left the machine intact (P2), and **every
one of the 13 boundary probes was neutralised, none reached the host** (P5, G-S15-6). The confinement
cost a median of **1.7 s a repository** (P7), and the perimeter is what was predicted: confined, a
read, a `chmod` and a UDP send still pass (P6). The cycle is INVALID on **G-S15-5**: the confined
audit did not read **every** repository exactly as the unconfined run did — **16 of 5,941 differ**
(P3 miss). All 16 are explained below; **12 are the same cause** — confinement denies a step's write
to `/tmp`, and a finding drops.

## 0. What was done

The change (the confinement inside the product, `styxx/ciaudit/confine.py`), the instrument, the
preregistration, the battery and the scorer went to GitHub (`fathomlab-patch-26`, five commits) before
any repository of the population was audited. The run started after, on the bare machine, three
repositories at a time, each audited in a Landlock-confined child (`confine.run`); a set of canary
files outside any scratch directory was checked after every one. It finished in 5,898 s: 5,945
repositories, none errored, the machine intact.

**One correction, after scoring, to the scorer** (not the instrument, the product, or the frozen
data): `swallow15_score.py`'s check that a moved repair backgrounds a command tried to read the
candidate list from the confined receipt, which stores a target *count*, not the list — it raised.
The check now returns "not confirmed background" when the list is absent (which is the honest reading:
the move is classified from `swallow15_classify.json`, §3, not the receipt), and P4's verdict is
unchanged by the fix. The corrected scorer is `b092130d…`.

## 1. Gates

| gate | bar | result |
|---|---|---|
| G-S15-1 instrument | six test files green | pass — 70 passed |
| G-S15-2 frozen underneath | instrument, eight product files, population, compared receipt, battery | pass — all |
| G-S15-3 the population is read | ≥ 90% of 5,945 audited confined without error | pass — **5,945 of 5,945**, no error |
| G-S15-4 no breach | the canary set after every repository | pass — **0 breaches** |
| G-S15-5 equivalence | the confined core against the receipt | **FAIL — 16 of 5,941 differ** |
| G-S15-6 the battery | the confined arm | pass — **13 of 13 neutralised, 0 reached the host** |
| G-S15-7 ledger | P1–P7 scored | pass |

## 2. Predictions, scored

| | predicted | observed | |
|---|---|---|---|
| P1 no breach | 0 canary breaches across 5,945 | **0** | HIT |
| P2 the wipers complete | the two `/`-wipers audit confined, machine intact | both audited, no error, host intact | HIT |
| P3 equivalence | the confined core matches the receipt for every repository | **16 of 5,941 differ** | MISS |
| P4 the chosen repair holds | verified_repair matches on ≥99%, every move a backgrounded command | **2 moved** — both the repair_frontier change shipped since the receipt, not backgrounded | MISS |
| P5 the battery | 13 neutralised, 0 reach the host, ≥8 fire unconfined | **13 neutralised, 0 reached, 10 fired** | HIT |
| P6 the perimeter | a read, a chmod, a UDP send all still pass confined | all three pass | HIT |
| P7 small | median ≤ 10 s a repository | **1.7 s** | HIT |

## 3. Reading

**The machine was never in danger, and that is the result that matters.** SWALLOW-14 lost a machine
to `actions/setup-node`'s `rm -rf $RUNNER_TOOL_CACHE/*` and could only rerun by giving every
repository a throwaway copy. Here the same repository, and 5,944 others, ran on the **bare machine**,
and the product's own Landlock confinement held: the four canary files outside every scratch directory
were intact after all 5,945 repositories, `actions/setup-node` and `djylb/nps` completed their audits
(0 targets each, as they did in the copy) with the host's `/bin`, `/etc`, `/usr` untouched, and every
hand-written probe — deleting a scope that came out empty, deleting, creating, truncating, appending,
renaming, making a directory, planting a symlink, a redirect to an absolute path, a child's write, a
background job's write, a socket connect — was refused inside the confinement while the audit still
completed. Ten of the thirteen reached a canary unconfined; the confinement stopped all thirteen.

**The 16 differences, every one accounted for.** The comparison is the confined audit's *core* of each
target — verdict, baseline, stage, the script's sha256, which candidates apply and are loud — against
SWALLOW-14's receipt. `swallow15_classify.json` re-audits each differing repository confined and
unconfined **with the current code**, which separates a confinement effect from a difference the code
carries:

- **12 — confinement suppresses a finding whose step writes to `/tmp`** (`CopilotKit`, `MetaMask`,
  `MicrosoftLearning`, `Norconex`, `OpenHands`, `PX4`, `Z3Prover`, `khive-ai`, `kodustech`,
  `llamastack`, `plasma-umass`, `pytorch/test-infra`). Every one has a job step that writes to an
  absolute path outside its scratch directory — almost all `/tmp/...` (`CopilotKit`'s commitlint
  step, `npx commitlint … | tee /tmp/commitlint-output.txt`; `PX4`'s `/tmp/sbom-*.txt`;
  `plasma-umass`'s `/proc/sys/kernel/core_pattern`, a write the boundary is right to deny). Confined,
  that write is refused, the step's simulated healthy run changes, and the finding — or a later one
  in the same job — is no longer read as a fault site. The confined audit **never invented a
  finding**; all 12 are a drop. With `/tmp` added to the writable set, **12 of the 12 recover**
  (`swallow15_tmp_writable.json`).
- **2 — the load-dependent backgrounded command** (`openshift-pipelines/pipelines-as-code`,
  `langflow-ai/langflow`). These are the family SWALLOW-14's determinism gate (G-S14-5) named: a step
  that backgrounds a command reads with the machine's load, and a re-run differs. Not a confinement
  effect — `pipelines-as-code`'s check does not recover with `/tmp` writable, and `langflow` reads
  the same confined and unconfined.
- **2 — the product moved since the receipt** (`kortix-ai/suna`, `mavlink/qgroundcontrol`). The
  SWALLOW-15 product carries SWALLOW-14's after-scoring masker fix (`repair_frontier.py` `bd6eb5af`,
  where the receipt's is `c10b9975`), which makes wait-list apply to these two. Read confined and
  unconfined with the current code they are identical; the difference is the code, not the
  confinement. P4's two moves are exactly these.

**Why P3 and P4 missed, honestly.** P3 asked for zero differences and there are 16. Twelve are a real
property of the boundary: **the confinement is not transparent to a step that writes outside its
scratch directory, even benignly to `/tmp`.** Four are not confinement at all — two are the
backgrounded-command nondeterminism SWALLOW-14 already found, and two are a code change this cycle's
product carries that the frozen receipt predates. The comparison against a receipt built with an
earlier `repair_frontier` was the instrument's flaw: it folds a code delta into a gate that was meant
to isolate confinement. The clean comparison — confined against unconfined, same code — is in
`swallow15_classify.json`, and by it the confinement changes the reading of **12** repositories, all
by the one `/tmp` mechanism.

## 4. What this does not say

The machine was safe; the boundary is not transparent. A confined step may still **read** what its
user can read, change a file's **mode, owner and timestamps**, and send a **UDP** datagram (P6, the
stated perimeter): a repository's step could read a local secret and leak it by UDP, and Landlock here
would not stop it. The confinement governs the filesystem writes, TCP and signals of the simulation,
and only on a kernel with Landlock (5.13+); the CLI refuses without it unless `--unconfined`, and the
Action refuses on a self-hosted runner but runs on a GitHub-hosted one, thrown away after the job.
None of this is a claim that a verified repair is correct (SWALLOW-4 through -14 bound that). The 12
`/tmp` suppressions mean the confined audit reads 12 of 5,941 repositories with one fewer finding than
an unconfined audit would — it under-reports there, it does not mis-report. The population is a public
dataset's popular repositories, each read at one commit.

## 5. What ships

The confinement, in the product (`confine.py`, and the audit path in `__init__.py`, `action.py`,
`differential.py`, `report.py`, `cli.py`): before it simulates, the audit restricts itself with
Landlock, every simulated step may write only beneath a scratch directory of its own, open no TCP, and
signal nothing outside the audit; the CLI refuses where the kernel cannot confine it unless
`--unconfined`; the Action refuses on a runner someone keeps. The card and the job summary say whether
the run was confined. `benchmarks/harness_mutation/confined.py` frozen at `7014bc02…`.

**Not shipped, on purpose:** making `/tmp` writable. It recovers the 12 (`swallow15_tmp_writable.json`),
but the host's `/tmp` is shared — a simulated step could then clobber another process's `/tmp` files.
The right fix keeps `/tmp` private to the audit, and that is the next cycle.

## 6. Next

**SWALLOW-16 — `/tmp` that goes nowhere.** Give the confined simulation a `/tmp` (and the absolute
paths a step reaches for) that is remapped into its scratch directory, so a step's `tee /tmp/log`
writes to the throwaway copy and the audit reads the repository exactly as an unconfined run would —
the transparency G-S15-5 asked for — while the host's `/tmp` stays untouched. Landlock cannot remap a
path; a mount namespace can (the cost is the user-namespace SWALLOW-15 avoided), and the two can
compose: the namespace remaps, Landlock still denies everything outside. Then the equivalence gate
this cycle failed, run again on the same 5,945.
