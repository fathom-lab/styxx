# RESULT — SWALLOW-5: the structural repairs — 7 of the 14 checks a strict shell could not make loud have a verified fix; the guard whose failing tool was its green path is the commonest shape; the remainder is three warn-by-design verifiers, one reporter, and three that need the one edit this instrument cannot verify

Fathom Lab · 2026-09-21 · Scores the receipt `swallow5_receipt.json.gz` against the
preregistration frozen at sha256 `9bb6f856d51632a888c2574ba498c18445321dc7c1ab31282d73c7c06847da28`.
Not amended. Two runs: the instrument was revised once after the freeze, for a defect the first
run exposed; §1 states what changed, what it moved, and that the predictions and the scorer did
not.

Receipt: `papers/harness/swallow5_receipt.json.gz` (sha256 of the JSON `030d6591…`, recorded by the
scorer; every candidate's unified diff is in it) · instrument
`benchmarks/harness_mutation/repair_structural.py`, sha256 `77067a71fa41e480…`, on top of
`repair.py` (`7b9a1695…`), `action_checks.py` (`0e723694…`) and `faults.py` (`d26a407c…`), none
changed · targets: the 18 targets of the SWALLOW-4 receipt (`7f19927a…`) without a verified
repair, on the same trees at the same HEADs · 19 s · scored by `swallow5_score.py`.

**VALID. 9 of 12 predictions HIT** (P1, P2, P3a, P3b, P3d, P3e, P3f, P5, P7). Of the **14
hand-written** checks SWALLOW-4 could not make loud, **7 have a verified structural repair** —
loud under the same fault in every flavour that could read it, healthy run unchanged in both —
**5 by `guard-status`** (the guard whose failing tool was its green path: `hmis`, `roslyn`,
`aspire`'s three runsheet checks) and **2 by `no-default`** (`langfuse`'s `|| echo` on its
continuation line, `nodetool`'s per-package `|| echo "… (continuing)"`). Of the 7 that remain:
three are verifiers that **warn by design** (`selfxyz`'s two `Verify branch and commit`, `gh-aw`'s
`Verify no compilation errors`), one is a **reporter by design** (`aspire`'s flaky-test
iterations), one is **fail-closed by design** (`gumroad`'s `ci-green`, still), and **three need
the one edit this instrument cannot verify** — an answer that must not be empty — which §3 reads.
The generated residue: 1 of 4 verified (P7), and the twin condition rejected two more of them
for the right reason.

## 0. What was done

SWALLOW-4's residue — every target of its receipt without a verified repair — was re-read on
this machine (18 of 18 baselines equal the SWALLOW-3 verdict) and three candidates were tried at
each fault site, in order: `guard-status` (a single-line `if CMD; then` becomes an explicit
status capture in which a status above 1 fails the step, CMD kept exempt from errexit by `||
__rc=$?`), `no-default` (every `|| echo …` fallback removed, then SWALLOW-4's strict shell), and
both. Each candidate is a text edit to the workflow, re-parsed and re-analysed with the same
instrument for the same (job, index), and judged on SWALLOW-4's two halves.

## 1. Gates, and the deviation

| gate | bar | observed | |
|---|---|---|---|
| G-S5-1 (instrument) | tests green; fixture deterministic; this repository has no target | 5 passed; identical twice; 0 targets | pass |
| G-S5-2 (baseline) | 18 targets; every baseline equals the receipt's, or is named (≤ 2) | 18; 0 excluded; 0 missing | pass |
| G-S5-3 (both halves) | verified ⇒ loud in every interpretable flavour and unchanged in both | 0 verified candidates failing a half | pass |
| G-S5-4 (no hand labels) | the three stated repairs; the first stage untouched | 0 outside; `repair.py` `7b9a1695…` | pass |
| G-S5-5 (frozen underneath) | `faults.py`, `action_checks.py`, `repair.py`, source receipt | `d26a407c…`, `0e723694…`, `7b9a1695…`, `7f19927a…` | pass |
| G-S5-6 (ledger) | P1–P7 scored, P3 as six | below | pass |

**The deviation, stated.** The preregistration was frozen and not edited; its predictions and the
scorer's definitions are the frozen ones. The instrument was revised once, and the run repeated.

| run | instrument | scored | why the next one |
|---|---|---|---|
| 1 | `1a778058…` | 8/12 (P1, P3c, P4, P6 MISS); receipt kept beside this one as `../s5_run1` in the working tree, not published | the `no-default` edit did not read quotes: on `nodetool`'s `npm run test:mutation … \|\| echo "stryker run failed for $pkg (continuing)"` it cut the fallback at the `)` inside the string and left a script that does not parse (recorded as "no longer a fault site"). A defect, not a finding: the removal now reads single and double quotes and stops only at an unquoted `;`, `)`, `\|` or `&`; a regression test holds it |
| 2 | `77067a71…` | **9/12; this RESULT** | — |

What the revision moved: one target, `nodetool`, from a syntax error to a verified `no-default`
(4 lines); nothing else in the receipt changed. **That one target is the difference between P1
MISS (6 of 14) and P1 HIT (7 of 14).** The reader who wants the frozen-instrument score has it in
the table; the author's reading is that the defect was a defect — a repair that breaks the file
is not a repair — and that the prediction is exactly at its bar either way.

## 2. Predictions, scored

| | prediction | observed | |
|---|---|---|---|
| P1 | ≥ 7 of the 14 hand-written residue targets verified | **7** (run 1: 6, §1) | **HIT** |
| P2 | `guard-status` verifies at least as many as `no-default` | **5 against 2** | **HIT** |
| P3a | `hmis` `Validate grantAllPrivilegesToAllUsersForTesting`: verified, `guard-status` | **verified, `guard-status`**, 5 lines | **HIT** |
| P3b | `roslyn` `Determine validation type and pipeline ID`: verified, `guard-status` | **verified, `guard-status`**, 5 lines | **HIT** |
| P3c | `selfxyz` `Verify branch and commit (iOS)`: verified, `no-default` | **not verified**: `no-default` applies (the `\|\| echo 'detached'` inside an `echo` argument, and the strict shell) and is not loud — the verifier compares two answers and *warns*; under the fault both answers are empty and compare equal | **MISS** |
| P3d | `langfuse` `run SQL-equivalence tests (non-blocking)`: verified, `no-default` | **verified, `no-default`**, 5 lines: the `\|\| echo "::warning …"` on its continuation line removed | **HIT** |
| P3e | `gumroad` `ci-green`: not verified | **not verified**: no candidate applies | **HIT** |
| P3f | `aspire` `Run test iterations`: not verified | **not verified**: `guard-status` applies to the inner zero-test guard and is not loud — the step is a reporter that counts PASS and FAIL | **HIT** |
| P4 | `mlflow`, `nodetool`, `aspire`'s three runsheet checks: none verified | **4 of 5 verified**: the three runsheet checks are `if echo "$RUNSHEET" \| jq -e …; then` guards, exactly the shape of `guard-status`, and `nodetool`'s loop carries a `\|\| echo` per package; only `mlflow` stays | **MISS** |
| P5 | every verified hand-written repair ≤ 8 lines; median ≤ 5 | **4, 5, 5, 5, 5, 5, 5**; median 5 | **HIT** |
| P6 | ≥ 1 hand-written target with a candidate rejected for changing a healthy run | **0** hand-written (2 generated, §3) | **MISS** |
| P7 | ≤ 2 of 4 generated residue targets verified | **1** (`gh-aw`'s `Verify static analysis tools`: seven `docker run … --version \|\| echo "Warning: …"` made loud) | **HIT** |

## 3. The map of the residue

| the check | shape | repair |
|---|---|---|
| `hmis` grant-all-privileges guard | `if grep -q …; then exit 1; fi` — the grep that fails was the green path | **guard-status** |
| `roslyn` `/dart` comment guard | `if echo "$COMMENT_BODY" \| grep -q "/dart"; then` | **guard-status** |
| `aspire` requires NuGets / CLI archives / GitHub token (3) | `if echo "$RUNSHEET" \| jq -e '…'; then` | **guard-status** ×3 |
| `langfuse` SQL-equivalence tests | `pnpm … test … \` + `\|\| echo "::warning …"` | **no-default** |
| `nodetool` mutation testing | `npm run test:mutation … \|\| echo "… (continuing)"` in a loop | **no-default** |
| `selfxyz` verify branch and commit (2) | compares `$(git rev-parse HEAD)` to `$(git rev-parse origin/…)` and **warns**; both empty under the fault, equal | none: warn-by-design; the edit is `test -n` on the answers |
| `gh-aw` verify no compilation errors | `if [ -n "$(find …)" ]; then echo "Warning: …"; fi` | none: warn-by-design |
| `aspire` flaky-test iterations | counts PASS and FAIL over N runs and reports | none: a reporter |
| `gumroad` ci-green | fail-closed by design (SWALLOW-2 §4) | none |
| `mlflow` database tests | `for service in $(./compose.sh config … \| grep …)`: the query's failure empties the loop; `trap ERR` does not see a substitution | none: the edit is `test -n` on the list |
| `serviceradar` Alpine APK pins | `while read url; do … done < <(grep -oE … MODULE.bazel)`: the query's failure empties the loop | none: the edit is `test -n` on the list |

**The guard is the shape.** Five of the seven verified repairs are the same rewrite: a tool whose
failure was the `if`'s green branch now fails the step when its status is above 1, and still
answers "no" at status 1. `aspire`'s three `Check if any test requires …` steps — read in
SWALLOW-4 as loops from their `RUNSHEET=` heads — are `jq -e` guards, and the guard repairs them
(P4's miss is the reading's, not the repair's).

**Three need the edit the instrument cannot verify.** `selfxyz`, `mlflow` and `serviceradar` are
one shape: a query whose empty answer is indistinguishable from a legitimate "nothing" — two
commits that compare equal, a service list that is empty, a pin list that is empty — and the
repair is `test -n` on the answer. The preregistration said why that repair is not tried: the
instrument's `empty` flavour is a healthy run in which every answer is empty, so the twin
condition would reject it by construction. The honest statement: for `git rev-parse HEAD` an
empty answer is not a healthy run, and a finer model of a healthy run — one that knows which
queries can legitimately answer nothing — would verify these three. That model is a next
instrument, not this one.

**The twin condition earned its place twice more, both in the generated stratum.** `dotnet/maui`'s
two `Verify connectivity` steps end with `test -f FILE && echo "✅ …" || echo "⚠️ … missing"`: the
`|| echo` there is not a tool's default but the else of a test, and removing it makes `test -f`
fail the step under `set -euo pipefail` when the file is absent — which, in the sandbox, it is.
Rejected in both flavours: the edit was loud and wrong. The one generated repair verified,
`gh-aw`'s tool-version check, removed seven `|| echo "Warning: … version check failed"` — the
word *Warning* says what the authors meant, and the repair says what it costs.

## 4. What this does not say

- **Loud is not wanted.** Three of the seven unrepaired are verifiers written to warn; one is a
  reporter; the seven repaired include `langfuse`'s tests labelled *non-blocking* and `gh-aw`'s
  *Warning*s. The receipt measures distance to loud, not the authors' intent.
- **The guard's threshold is a convention.** Status above 1 is `grep`'s and `diff`'s error, and
  the model's fault is 127; a tool whose outright failure exits 1 is not caught by it.
- **Two runs.** The first run's `no-default` cut a fallback mid-string; the fix moved one target
  and one prediction; both scores are in §1.
- **A cosmetic mark**: a `run:` written as a quoted flow scalar is rewritten as a block, so its
  whole body shows in the diff (`gh-aw`'s 35 lines are one edit ×7); and the block rewrite drops a
  trailing blank line, as in SWALLOW-4.
- **On an instrument known not to be bit-reproducible** (SWALLOW-3 §1); none of its unstable
  shapes sits on a target (18 of 18 baselines equal).

## 5. What ships

`styxx ci-audit --repair` now has two stages: SWALLOW-4's repairs first, and for what they leave
unverified, the structural ones. The card prints the verified diff, or which half failed, or that
no repair among the five applies. `styxx/ciaudit/repair_structural.py` is the living copy, held to
the frozen instrument on a fixture by `tests/test_ciaudit.py`, which pins four instruments now.

## 6. Next

- **A healthy run that knows which answers may be empty**: a third flavour, or a per-query
  declaration, so `test -n` on a commit hash can be verified and `test -n` on a diff cannot. Then
  the three.
- SWALLOW-2.1, the same run twice (RESULT_swallow3 §6).
- The counted reading and a check rule without the diagnostics.
