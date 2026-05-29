# FINDING — Method-diverse grounding GENERALIZES beyond arithmetic to code-output tracing (SURVIVED)

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_code_tracing_grounding.md` BEFORE data. Single model gpt-4o-mini, OpenAI-only,
n=36 deterministic import-free Python snippets × 2 grounding backends. Ground truth
computed in-code by EXECUTING each snippet, SHA-256'd pre-scoring
(`0d7de3c1…ea5faeb`). Exact-integer correctness, no judge.** Receipt:
`code_tracing_grounding_result.json`.

## What this run did

The path-diverse grounding keystone (`FINDING_path_diverse_grounding_2026_05_28.md`)
established on **arithmetic** that re-deriving a self-claim through method-diverse paths
converts the grounded honesty axis from a *belief* certifier into a *truth* certifier.
Its stated boundary: "arithmetic is one derivation domain; whether method-diverse
grounding generalizes to other multi-step reasoning is untested." This run is that probe,
in a genuinely different domain: **code-output tracing**, where difficulty comes from
**control flow** (loops, branches, nesting, stateful Collatz-style transforms), not
large-number arithmetic. Four tiers (9 each): `code_ctrl`, `code_loop`, `code_nested`,
`code_multistep`. Backends compared on identical items: **plain** one-shot vs
**path-diverse** (N derivations through 5 rotating tracing methods — variable table,
per-iteration simulation, closed-form, run-twice-verify, branch analysis).

## Result: SURVIVED — all four pre-registered bars held.

| id | prediction | bar | outcome |
| --- | --- | --- | --- |
| **G1 — method-diversity self-corrects on code** | path-diverse recovers correct modal answer on plain-wrong items | fix rate ≥ 0.40 | **HELD. 1.000** (25/25 plain-wrong items recovered) |
| **G2 — path-diverse RESTORES the truth signal (decisive)** | grounded AUC overall AND on high-Stability stratum | ≥ 0.85 AND high ≥ 0.85 | **HELD. AUC 1.000 overall, 1.000 high-stratum** |
| **G3 — concordance-with-truth jumps** | mean fraction of resamples on true value, hard tiers, path vs plain | Δ ≥ +0.30 | **HELD. 0.093 → 0.963, Δ = +0.870** |
| **K — control tier unharmed + register-clean** | ctrl AUC under path-diverse AND register confound check | ctrl ≥ 0.90 AND Welch p > 0.05 | **HELD. ctrl AUC 1.000, register p = 1.0** |

**RESULT = SURVIVED.** plain grounded AUC **0.671 → path-diverse 1.000**; plain
per-tier AUC collapses on the hard tiers (code_loop 0.50, code_nested 0.72,
code_multistep 0.56) while path-diverse is perfect on every tier.

## Precondition MET — the competence cliff transfers to code.

25/36 plain one-shot modal answers were wrong (confident confabulation on control flow,
not arithmetic), exactly the regime the probe needs. The cliff is real in a non-
arithmetic domain, and method-diversity repairs it completely here.

## The honest part: v1 was REPORT_AS_LANDED — and the failure was MY measurement bug, caught before trusting it.

The first run of this exact prereg returned **REPORT_AS_LANDED** with G3 *failed*
(Δ = +0.189, path AUC 0.871, code_nested path AUC an anomalous 0.556 — *worse* than
plain). Per project discipline I flagged the suspicious nested collapse BEFORE writing
any finding and audited the raw resamples. It was a **truncation artifact**, not a real
result: verbose nested-loop CoT traces hit `max_tokens=600` before emitting the final
`ANSWER:` line, so the last-integer fallback parser grabbed a **loop-internal** number
(a partial sum mid-trace) instead of the committed answer. The model was tracing
*correctly* — the totals were climbing to the right value — but the harness was reading
the wrong integer off truncated output.

Fix: `max_tokens 600 → 2000` and a **strict `ANSWER:`-only parser** that returns
None (abstain) when no committed answer line is present, instead of guessing the last
integer. Re-ran the identical prereg. The corrected run is the one above — SURVIVED,
with code_nested path AUC going 0.556 → 1.000 and G3 going +0.189 → +0.870. The v1
"G3 failure" was 100% instrumentation; the underlying effect was always there.

This is the methodology working as intended: **a finding is only as good as the number
it rests on, and a surprising sub-result gets audited before it's trusted, not after.**

## What landed

Method-diverse grounding is **not an arithmetic trick.** In code-output tracing — a
domain where difficulty is control-flow depth, not number size — re-deriving the claim
through 5 independent tracing methods:
- recovers **every** plain confabulation (25/25),
- lifts grounded AUC to a ceiling (1.000, including the high-Stability abstain stratum),
- raises concordance-with-truth on the hard tiers from ~9% to ~96%,
- leaves the control tier and register balance untouched.

Combined with the arithmetic keystone, the **belief→truth dial generalizes across
derivation domains.** Method-diversity forces the model out of its one-shot wrong
attractor and into independent construction, and the grounded axis then certifies truth,
not just belief — now shown in two structurally different domains.

## Honest scope (pre-committed)

Single model gpt-4o-mini, OpenAI-only, one confirmatory run, feasibility-grade, n=36 × 2
backends. Ground truth computed by EXECUTING each deterministic import-free snippet, then
hashed pre-scoring; core signal exact integer match (no judge); answers parsed from a
strict `ANSWER:` line (abstain on absence). Self-consistency not external truth (here the
executed value IS external truth, so this run is truth-anchored); injection-blind;
one axis-family. Generalization claim is to **code-output tracing**; multi-hop QA and
open-ended logic remain untested. The perfect 1.000 AUC reflects a clean cliff +
complete repair at this n and difficulty, not a claim of universal infallibility.
