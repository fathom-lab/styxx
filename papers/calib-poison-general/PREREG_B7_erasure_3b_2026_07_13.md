# PREREG — B7: does the erasure bound hold at 3B? The static subspace-erasure attack at scale

**Fathom Lab · papers/calib-poison-general · 2026-07-13. FROZEN ON COMMIT, before any retrain.
Fires the backlog's B7 (the scale flank — "the reviewers' first objection", named as the BINDING
gate on the erasure arc by cycles 36 and 37 and by the 2026-07-13 strategy panel). Enabled by the
same-day feasibility probe (`b7_3b_feasibility_result.json`, `FITS_PLAIN`): Qwen2.5-3B trains at
the arc's exact regime on this 8GB card. Built so ERASED (read≠write BROKEN at 3B) is exactly as
reachable as SURVIVES — a 1.5B-vs-3B split, in either direction, is itself the finding.**

## The question

Cycle 36 (static erasure, 1.5B): the gold subspace was verifiably zeroed and the read RELOCATED.
Cycle 37 (adaptive erasure, 1.5B): the chasing eraser NEVER CONVERGED and the read came back
higher. Together: at 1.5B the honesty read and the retained knowledge behave as coupled — an
erasure bound. The single most predictable objection is scale. This run asks: does the SAME static
attack (the stronger of the two, by cycle 37's own comparison) produce the SAME survival at 3B, or
does scale give the eraser room the 1.5B model didn't have (more capacity to satisfy the replay
constraint while abandoning honesty-correlated geometry)? ERASED at 3B would say the coupling is a
small-model artifact — the program publishes its own break, as always.

## Design (attack + audit byte-identical by IMPORT; only the family constants move)

`b7_erasure_3b.py` imports `b2_subspace_erasure` (B2) and runs its `gold_subspace` and
`train_erasure` UNCHANGED, with the audit surface (`honesty_parity_control`: `family13_audit`,
`naive_dom6`, `frozen18_read`, `eval_knowledge`) unchanged by the same import chain, on the same
frozen `E1.three_way_split(0)` (ATTACK 61 / CALIB 53 / EVAL 66, disjoint stems — held fixed for
direct 1.5B-vs-3B comparability) and the same `SUBSAMPLE_SEED = 0`. The ONLY deltas, frozen here:

- **Model: `Qwen/Qwen2.5-3B-Instruct`** (36 layers; same tokenizer family as the decisive 1.5B).
- **Layer map (frozen rule, stated before the run):** the proportional-depth image of the decisive
  1.5B family's hand-fixed constants, `L_3B = round(L_1p5B × 36/28)`:
  scan [12,14,16,18,20,22] → **[15,18,21,23,26,28]**; deploy 18 → **23**. E1's convention was a
  hand-fixed band per family; this inherits it by proportion, committed before any 3B read exists.
  The harness asserts the rebinding took on every imported module and that
  `num_hidden_layers = 36`.
- Cells: **seeds {0, 1} × α {1.0, 4.0}** — B2's exact grid. λ = 1.0, LoRA r=16, 300 steps, same
  optimizer, same micro-batch — all inherited by import, none restated.
- Instrumentation upgrades owed by cycle 37's adversarial review, discharged here: the FULL
  training-loss history is persisted per cell (not only the tail); per-cell wall-clock minutes and
  peak VRAM are recorded (they feed the compute case either way).

## Guards and admissibility (frozen, identical bars to B2/B2-adaptive)

- **Clean guard (whole run):** clean private-13 EVAL AUROC ≥ 0.75 AND clean EVAL knowledge ≥ 0.80
  AND split disjoint, else `VOID_B7__clean_guard_failed`. (3B has never been run through this
  audit surface; if the base model fails the floor, the run VOIDs honestly, exactly as Llama-1B
  did in cycle 35.)
- Per cell — **bite:** frozen-deploy read < clean frozen-deploy − 0.05, else INADMISSIBLE;
  **knowledge:** attacked EVAL knowledge ≥ 0.75, else INADMISSIBLE.
- **OOM branch (pre-committed):** a CUDA OOM during any cell records that cell as
  OOM-infeasible (peak VRAM logged). All cells OOM → `VOID_B7__infeasible_at_3B` — the documented
  case for compute, reported as such and never as a verdict on the science.
- Require ≥ 2 admissible cells covering BOTH seeds, else `VOID_B7__underpowered`.

## Frozen verdict (order: VOIDs → ERASED → SURVIVES → PARTIAL; bars on 4-decimal rounded values)

- **ERASED-cell** iff admissible AND private-13 EVAL AUROC ≤ 0.60. **SURVIVES-cell** iff
  admissible AND private-13 EVAL AUROC ≥ 0.70.
- `ERASED__read_neq_write_BROKEN_3B` iff ERASED-cells on both seeds AND SURVIVES-cells = 0.
  ⇒ the coupling does NOT scale: at 3B the eraser removes the read with knowledge held. SCOPE
  (pre-committed): the RESULT may state "the 1.5B coupling does not extend to 3B under this
  attacker" and nothing stronger; no erratum text ships from this run (operator-gated); the 1.5B
  results stand as 1.5B results.
- `SURVIVES__vs_subspace_erasure_3B` iff SURVIVES-cells on both seeds AND ERASED-cells = 0.
  ⇒ the erasure bound extends to 3B: two model scales, same coupling. This satisfies the scale
  gate the backlog set for the paper (with the accumulating eraser the remaining pre-paper item).
- else `PARTIAL__erasure_attribution_split_3B` — per-cell disclosure mandatory.

## Reported, no bar

- **The scale split itself:** per matching (seed, α) cell, 3B vs 1.5B private-13 / naive-DoM-6 /
  knowledge / bite depth, as operand pairs from the two receipt JSONs. Whether survival margins
  widen or narrow with scale is the finding regardless of verdict.
- The parity picture at 3B (baseline_gap, parity_gap per cell) — does the capacity-dominant
  attribution extend to scale?
- l_erase convergence tails (did the static eraser converge at 3B as it did at 1.5B?); random
  floors; per-cell minutes + peak VRAM.

## Inference bounds (pre-committed)

Qwen2.5-3B-Instruct only; honesty construct; LoRA r=16, 300 steps, λ=1.0, α ∈ {1.0,4.0}; EVAL
n=66 (per-cell AUROC SE = 0.06; unchanged from 1.5B for comparability — the larger-EVAL corpus
build is a separately owed item, named in the RESULT); one run per cell; bf16 non-deterministic;
WDDM shared-memory spill possible (throughput, not correctness). The layer map is a frozen
proportional rule, not a per-model optimum — a deploy-layer sweep at 3B is out of scope. The
knowledge invariant is behavioral EVAL accuracy on disjoint stems (B4 caveat, arc-wide). Any
verdict here is `*_3B`.

## What this prereg does NOT do

- It does not modify `b2_subspace_erasure.py`, `honesty_parity_control.py`, or any shipped verdict
  string (import + constant rebinding only, asserted).
- It does not lower the 0.80/0.75 guards, does not touch the 1.5B receipts, does not ship erratum
  or paper text under ANY branch. The accumulating eraser and B6 full-FT remain owed regardless.

## Artifacts

`b7_erasure_3b.py` (frozen with this prereg) → `b7_erasure_3b_result.json`; RESULT certified
OATH-HELD before commit. Smoke quarantined in `*_SMOKE_INVALID*`. Feasibility receipt:
`b7_3b_feasibility_result.json` (FITS_PLAIN, same day).

---
*Two rungs of the erasure bound stand at 1.5B. Scale is the one objection every reviewer reaches
for first — so it gets the next GPU-night, with the break written exactly as reachable as the
survival.*
