# PREREG — attribution confirmation (part i): does the honesty "privacy" residual survive ≥3 seeds + a λ sweep?

**Fathom Lab · papers/calib-poison-general · 2026-07-11. FROZEN ON COMMIT, before any retrain. Fires
from `RESULT_honesty_parity_control_2026_07_11.md` (PRIVACY_PARTIAL: seed 0 kept a ~0.035
privacy-specific residual, seed 1 FAILS-outright with the poisoned-fit parity auditor beating the
clean-fit private one). Per that RESULT's binding sequencing rail, before any erratum to the
published read≠write claim two things must hold: (i) ≥3 seeds + a λ sweep {1.0, 3.0} confirming
FAILS on the decisive Qwen family, and (ii) the Llama-3.2-1B family. THIS prereg runs part (i) only.
Part (ii) is a separate frozen prereg. Built so the branch that would VINDICATE the flagship
(`STANDS_CONSOLIDATED`) is as reachable as the one that would demote it.**

## The question, sharpened

The honesty parity check (2 cells, λ=1) showed the flagship's `private > naive` recovery is mostly
probe capacity: at probe+size parity the gap fell 0.3272 → 0.0553 (seed 0) and 0.2341 → −0.082
(seed 1). It landed PARTIAL only because ONE of two seeds (seed 0) retained a small
privacy-specific residual (parity_gap 0.0553 > the 0.02 FAILS bar). **Two seeds cannot tell a real
residual from noise (per-cell AUROC SE ≈ 0.06).** This run adds seed 2 and λ=3.0 — six Qwen cells —
and asks whether the residual is a robust feature or a single-cell fluctuation.

## Design (one grid, shared code path by IMPORT)

Reuse the frozen honesty parity harness verbatim: `honesty_parity_confirm.py` **imports**
`honesty_parity_control.py` and calls its exact `family13_audit`, `naive_dom6`, `frozen18_read`,
`eval_knowledge`, `E1.three_way_split`, `ATK.train_attack`, and the frozen `SUBSAMPLE_SEED = 0`
ATTACK-subsample of size |CALIB| — so the parity code path is byte-identical to cycle 33, not a
re-implementation. Grid:

- Model: **Qwen2.5-1.5B-Instruct** (E1's decisive family; Llama is part (ii), not here).
- Split: `E1.three_way_split(0)` — the SAME disjoint-stem 50/25/25 split as cycle 33.
- Attack: `ATK.train_attack`, deploy layer 18, scan {12,14,16,18,20,22}, LoRA r=16, 300 steps.
- Cells: **seeds {0, 1, 2} × λ {1.0, 3.0} = 6 cells.** (seeds 0/1 at λ=1 re-run cycle 33's cells —
  their rerun stability vs cycle-33 canonical is reported, no bar.)
- Per cell, the three auditors via the shared 13-family path: naive-DoM-6, naive-MATCHED-13
  (fit ATTACK-sub n=|CALIB|), private-13 (fit CALIB). Plus frozen-18 bite + behavioral knowledge.

## Per-cell admissibility (a cell is dropped, not the run)

- **Bite:** frozen-18 read < clean frozen-18 − 0.05, else the cell is INADMISSIBLE (attack didn't
  bite — nothing to attribute).
- **Knowledge-preserving:** attacked-EVAL behavioral knowledge ≥ 0.75, else INADMISSIBLE.
- Require **≥ 5 of 6** cells admissible, else `VOID_HPCF__underpowered` (whole run).

## Per-cell indicators (frozen; on 4-decimal rounded values as logged)

- `baseline_gap = private-13 − naive-DoM-6`; `parity_gap = private-13 − naive-MATCHED-13`.
- **FAILS-cell** iff `parity_gap ≤ 0.02` (i.e. naive-MATCHED-13 ≥ private-13 − 0.02).
- **STANDS-cell** iff `parity_gap ≥ 0.5 × baseline_gap`.
- Disjoint whenever `baseline_gap > 0.04` (0.5×0.04 = 0.02); a cell with `baseline_gap ≤ 0.04` is
  neither (counted as "no-gap cell", reported).

## Frozen aggregate verdict (over the admissible cells; N target 6, strong-majority = ⌈2N/3⌉)

Order checked: VOID guards → NO_GAP → FAILS → STANDS → PARTIAL. Strings distinct and namespaced
`*_CONSOLIDATED` / `VOID_HPCF__*` (disjoint from cycle-33 `*_HPC__*` and the sentiment `*_MPC__*`).

- Clean-model guard (whole run): private-13 EVAL AUROC ≥ 0.75 AND EVAL knowledge ≥ 0.80 AND split
  disjoint, else `VOID_HPCF__clean_guard_failed`.
- Baseline reproduces: **median baseline_gap over admissible cells ≥ 0.05**, else
  `VOID_HPCF__baseline_gap_not_reproduced`.
- `FAILS_CONSOLIDATED__privacy_residual_not_robust` iff **FAILS-cells ≥ ⌈2N/3⌉ (≥ 4 of 6) AND
  STANDS-cells = 0.** ⇒ across seeds and λ the privacy-specific residual is not robust; the honesty
  recovery reduces to probe family/capacity, same as sentiment. This SATISFIES sequencing part (i);
  the RESULT restates the inherited (frozen) re-attribution language as within-family consolidated,
  still gated on part (ii) Llama + operator sign-off before any erratum ships. NO erratum text from
  this run.
- `STANDS_CONSOLIDATED__privacy_residual_robust` iff **STANDS-cells ≥ ⌈2N/3⌉ (≥ 4 of 6) AND
  FAILS-cells = 0.** ⇒ the private clean-fit split buys ≥ half the gap robustly across seeds/λ; the
  flagship "privacy" attribution is VINDICATED within-family and cycle-33's discordance was the
  noise. The privacy claim gains the control it never had.
- else `PARTIAL_CONSOLIDATED__residual_real_but_not_robust` — reported verbatim with the per-cell
  FAILS/STANDS/no-gap counts, the median & mean parity_gap and baseline_gap, and (descriptive,
  non-gating) the pooled 8-cell picture including cycle-33's two λ=1 cells. Under PARTIAL the
  attribution stays formally unsettled; more seeds / cross-family remain owed.

## Reported, no bar

- Rerun stability of the two shared cells (seed 0/1, λ=1) vs cycle-33 (naive 0.5106/0.4876,
  matched13 0.7825/0.8037, private 0.8378/0.7217); bf16 CUDA is not bit-deterministic.
- Chance floor (random-direction EVAL read) per cell from the shared audit function.
- λ effect: whether parity_gap trends with λ (stronger knowledge-replay) — descriptive.

## Inference bounds (pre-committed)

Qwen2.5-1.5B only, honesty construct, EVAL n=66 (per-cell AUROC SE ≈ 0.06), one run per cell, bf16
non-deterministic. Six cells with a two-thirds-majority rule is a robustness check, not a proof: it
can distinguish "residual on 1 of 2 seeds" (cycle 33) from "residual on ≥4 of 6 seeds/λ" or
"residual essentially absent," but a genuinely small true residual (~0.03) near the 0.02 bar will
read as PARTIAL — which is the honest outcome for a signal at the noise floor. Cross-family (Llama)
and the operator-gated erratum remain downstream regardless of this verdict.

## What this prereg does NOT do

- It does not run Llama (sequencing part ii — separate frozen prereg).
- It does not ship any flagship or erratum text under ANY branch (the erratum is operator-gated and
  requires part ii).
- It does not touch the cycle-33 / E1 / attack_sweep verdict strings.

## Artifacts

`honesty_parity_confirm.py` (frozen with this prereg) → `honesty_parity_confirm_result.json`; RESULT
certified OATH-HELD before commit. Smoke quarantined in `*_SMOKE_INVALID*`.

---
*Frozen before the run. Six cells decide whether cycle-33's lone seed-0 residual was signal or noise
— and STANDS (flagship vindicated) is written to be exactly as reachable as FAILS (flagship demoted).*
