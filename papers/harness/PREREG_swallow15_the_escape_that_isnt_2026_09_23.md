# PREREG — SWALLOW-15: the escape that isn't — the audit confines its own simulation with Landlock, red-teamed against its boundary and shown to read 5,945 repositories unchanged, on the bare machine

Fathom Lab · 2026-09-23 · Frozen before the population is audited.
Follows `RESULT_swallow14_the_empty_list_2026_09_22.md` (INVALID on its determinism gate, 6/8), whose
§0 found that `styxx ci-audit` runs each workflow step's shell on the machine — `actions/setup-node`'s
`rm -rf $RUNNER_TOOL_CACHE/*` becomes `rm -rf /*` in the simulation and deleted the run's machine —
and whose "next" named this: confine the executor, each step in a throwaway root, so `styxx ci-audit`
on a repository nobody has read is safe to type.

## Where this came from, stated before anything is measured

The audit reads a workflow by running each step's script with its tools stubbed, in a temporary
directory, with an empty environment. The shell is real, and what the stubs do not cover — `rm`,
`mkdir`, a redirect — acts on the machine. SWALLOW-14 confirmed it, lost a machine to it, and reran
only by giving every repository a throwaway copy of the machine (`swallow14_sandbox.sh`, an overlay
per repository). That protected the run; it did nothing for anyone who types the product's command.

This cycle puts the confinement **inside the product** (`styxx/ciaudit/confine.py`): before it
simulates, the audit restricts itself with Linux **Landlock** (5.13+; no root, no namespaces, nothing
to install), and everything it then starts — every simulated step — may write, create, rename,
truncate or delete only beneath a scratch directory of its own, may open no TCP connection (Landlock
ABI 4+), and may signal or reach through an abstract socket no process outside the audit (ABI 6+). The
CLI refuses on a kernel without Landlock unless `--unconfined`; the Action refuses on a runner someone
keeps (self-hosted) but runs on a GitHub-hosted runner, which is thrown away after the job.

The measurement runs **on the bare machine — no overlay, no throwaway copy.** The product's own
confinement is the only thing between a simulated `rm -rf /*` and the machine. If it is wrong, this
run loses its machine as SWALLOW-14's first run did; that is the test.

## 1. The instrument

`benchmarks/harness_mutation/confined.py`, frozen at
`7014bc02c46f5344249ed1636d4778c217df6998aa93164f96974817d2e97036`, calling the product as it ships:

| file | sha256 |
|---|---|
| `styxx/ciaudit/engine.py` | `5c219887e389ed671d082c18be220af4479d48d0bd152fb348997ba4f5368b24` |
| `styxx/ciaudit/actions.py` | `ecbcd4f94a050e5b983f91464e681f61f2504a5447c45c68f91e419459c8d42a` |
| `styxx/ciaudit/repair.py` | `4282dbfa4d5caf22bac274e0964603dc451db164df8c730ba94afee58189241a` |
| `styxx/ciaudit/repair_structural.py` | `af46c492656e4d1837c97e0ecb3f40088e782c04c306ac94f70dc4bde7846570` |
| `styxx/ciaudit/repair_frontier.py` | `bd6eb5af4fe169e9207a0ad67d6d7977cfea306d399ef1ce57f3bde9237f7c09` |
| `styxx/ciaudit/confine.py` | `46a1b40bf99db0313db24362ec970dc1cede67287fca32806544146718404da6` |
| `styxx/ciaudit/__init__.py` | `5c021afe1f09ac4c51d0534b4af26db505bd36e30145776f32efa9a80cdc89c7` |
| `styxx/ciaudit/differential.py` | `dc79fc97e3084df90c31bc64139bd6c44fbd5ff277f0f3fbf35776b319f888b0` |

**Population** — `papers/harness/swallow14_population.json.gz`
(`21c0e00fdaad3296f641668a48c30d3d33600cdacaf2faed691b36e071e5d58b`): SWALLOW-14's 5,945 repositories,
each at the tip its receipt recorded.

**Compared receipt** — `papers/harness/swallow14_receipt.json.gz`
(`9eba6f23d66ea62f8918b8f2c0fd067eb5cc3635976dd1c4fb8b3bb578ae7fb7`): SWALLOW-14's run, built
unconfined (each repository in an overlay). The confined run is compared to it, target by target.

**Battery** — `papers/harness/swallow15_battery.json`
(`9db078c19d3a21e4466e4f64e0367616ec58a36a2af516f886e4a08e170e925c`): 13 hand-written workflow steps
that each try to reach the host — delete a scope that came out empty, delete a named file or a
directory, create, truncate, append, rename, mkdir, symlink, write a redirect to an absolute path, a
child process's write, a background job's write, and a socket connect — each scoped to a canary
directory of the instrument's own making (never `/`), so the unconfined control fires on that canary
and no farther. Frozen from a run on this machine (Landlock ABI 7).

**Per repository** (the population arm): fetch `.github/workflows` at the tip (unconfined — the
network, and only the clone's own directory is written); then the product's audit of the checkout,
`--repair`, **in a Landlock-confined child** (`confine.run`); then compare its targets to the receipt.
A set of canary files on the machine, outside any scratch directory, is checked after every
repository. Each repository in its own process, three at a time.

**Units.** A target's **core** — the part confinement must not change and that does not depend on the
machine's load — is its verdict, baseline, stage, the script's sha256, and which candidates apply and
are loud. A repository **matches the receipt** when every target's core equals the receipt's. The
**chosen repair** (`verified_repair`) is compared beside the core, not in it: a candidate that
backgrounds a command can verify or not with the load (SWALLOW-14 G-S14-5), which is not a confinement
effect. A **canary breach** is any canary file, after any repository, missing or changed — a correct
confinement never causes one.

## 2. What was known at the freeze

- SWALLOW-14's receipt, and that two of its repositories (`actions/setup-node`, `djylb/nps`) deleted
  their overlay's `/bin` when audited unconfined.
- `confine.py`'s boundary, exercised on `tests/test_ciaudit_confine.py` (a confined child writes only
  in its scratch directory, opens no TCP, signals nothing outside; reads, chmod and a UDP send still
  pass — the stated perimeter). This kernel offers Landlock ABI 7.
- A smoke test of six repositories (the two wipers and four with targets) audited confined on the
  bare machine: no error, no breach, the four matched the receipt's core, and the machine's root was
  intact. Nothing was computed on the other 5,939.

## 3. Predictions, committed now

**P1 — no breach** (blind). Across the 5,945 repositories, the canary set is intact after every one:
**canaries_breached = 0**.

**P2 — the wipers complete** (blind). `actions/setup-node` and `djylb/nps`, which deleted their copy
in SWALLOW-14, are audited confined on the bare machine with no error, and the machine's root is
intact at the end.

**P3 — equivalence** (blind). The confined audit's core matches the SWALLOW-14 receipt for **every**
repository compared: **differ_from_receipt = 0**.

**P4 — the chosen repair holds** (blind). `verified_repair` matches the receipt on at least **99%** of
the repositories compared, and every repository where it moves has a candidate that backgrounds a
command (the load-dependent family, SWALLOW-14 G-S14-5) — no other kind moves.

**P5 — the battery** (blind on the confined arm). Every confined item is neutralised
(**confined_neutralised = 13**), **none reaches the host** (any_reached_host = []), and at least **8**
of the 13 reach the host unconfined (control_fired ≥ 8) — the boundary is what stops them.

**P6 — the perimeter** (a coin). Confined, a read of a file the user can read, a `chmod` of it, and a
UDP `sendto` all still succeed: Landlock here governs neither reads, nor a file's metadata, nor UDP.
Stated so the boundary is not trusted past its edge.

**P7 — small** (blind). The confinement's cost is a fork and a few syscalls: the **median seconds per
repository ≤ 10**.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S15-1 instrument | the confinement and the audit, end to end | `tests/test_ciaudit_confine.py`, `tests/test_harness_confined.py`, `tests/test_ciaudit.py`, `tests/test_ciaudit_action.py`, `tests/test_ciaudit_frontier.py`, `tests/test_harness_empty_list.py` green |
| G-S15-2 frozen underneath | the instrument, the eight product files, the population, the compared receipt, the battery | the hashes above, recorded by the receipt |
| G-S15-3 the population is read | fetched at the tip and audited confined without error | ≥ 90% of 5,945 |
| G-S15-4 no breach | the canary set after every repository | **0 breaches** |
| G-S15-5 equivalence | the confined core against the receipt | **0 differ** |
| G-S15-6 the battery | the confined arm | **13 of 13 neutralised, 0 reach the host** |
| G-S15-7 ledger | every prediction scored | HIT/MISS for P1–P7 |

G-S15-1 to G-S15-6 are blocking.

## 5. What would abandon this

G-S15-4: one canary breach means the confinement let a simulated step out, and nothing about the run
can be reported as safe. G-S15-5: the confined audit reading a repository differently in its core than
the unconfined one means the confinement changed the measurement, and the equivalence claim fails.
G-S15-6: any battery item reaching the host confined is the same failure as a breach.

## 6. Honest statement of what a passing SWALLOW-15 means

That on 5,945 repositories, including two that delete `/` in the simulation, the product's own Landlock
confinement kept the bare machine, and the audit read every one exactly as the unconfined run did. It
does **not** mean the audit is safe against everything: Landlock here does not govern reads, a file's
mode/owner/timestamps, or UDP — a confined step may still read what its user can read and send a
datagram, so a repository's step could, in principle, read a local secret and leak it by UDP (P6 is
that limit, stated). It does not confine a kernel without Landlock, where the CLI refuses unless told
otherwise. It governs the filesystem, TCP and signals of the simulation; it is not a claim about the
correctness of a verified repair, which SWALLOW-4 through -14 bound. The population is a public
dataset's popular repositories, each read at one commit.

## 7. Running it

```
python -m benchmarks.harness_mutation.confined --battery --out papers/harness/swallow15_battery.json
python -m benchmarks.harness_mutation.confined --population papers/harness/swallow14_population.json.gz \
    --receipt papers/harness/swallow14_receipt.json.gz --work <dir> --out papers/harness/swallow15_receipt.json.gz --workers 3
python papers/harness/swallow15_score.py
```
