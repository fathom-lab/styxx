# PREREG — B36: the write door. Does capacity open control transfer, or does read≠write survive its strongest attack?

Fathom Lab · 2026-08-01 · frozen before any scored run. The program's flagship dissociation —
*what a mind means crosses; the means to move it does not* — was measured un-confounded at the
steer-optimal layer (`RESULT_writelayer_decouple_2026_06_21`: native 0.2151, label-free linear
transfer 0.0245 = 11% NTE, direction-sign 71%). But b31v2 just proved the LINEAR class was
leaving readable content on the table (gemma chance → 55× with a paired MLP). The strongest
attack we can mount on our own law: **give the write side the same heavier machinery and
matched supervision, and see if control crosses.**

## Design (frozen)

- **Pair:** Llama-3.2-3B → Llama-3.2-1B — the substrate where every baseline is measured
  (native ceiling 0.2151 at dst layer 11, the steer-optimal point; pc_cos 0.818).
- **Operating point:** src read layer 11 (the G0-locked layer); dst layer 11 of 16 (the
  writelayer-decouple steer-optimal choice, frac 0.7). Fresh extraction of BOTH models at
  these layers with points AND paired steering directions (the committed `extract`).
- **Maps, matched supervision (392 TRUE pairs — supervision is deliberately maximal so a
  null cannot be blamed on the correspondence):**
  - **L1** paired linear: orthogonal Procrustes on the 392 paired points (comparator).
  - **M1** paired MLP: the b31v2 architecture/training verbatim (seed 36).
- **Direction transfer through M1:** finite difference at the concept point,
  `v_B(c) = normalize(F(x_A(c) + ε·v_A(c)) − F(x_A(c)))`, ε = 4.0 frozen; an ε/2
  direction-stability cosine is reported per concept (diagnostic, not gated). L1 transfers
  directions by its orthogonal map directly.
- **Steering protocol:** the committed `steer_gain` (3 carriers, greedy 28 tokens, MiniLM
  concept-sim gain over clean) on the 70 held-out concepts; dose locked by the committed
  `lock_dose` on NATIVE directions first (positive control and dose share one locking pass).
- **Arms per concept:** native (B's own direction — the ceiling), M1-transfer, L1-transfer,
  random unit direction (the null), all at the locked dose.

## Gates (imported frozen from the writelayer-decouple prereg; no bar invented here)

- **PC (positive control):** native mean gain ≥ 0.15. Fail → `INVALID__substrate_not_steerable`.
- **G1:** M1-transfer mean gain ≥ 0.15.
- **G2:** M1-transfer − random ≥ 0.10 (magnitude) AND M1 beats random on ≥ 70% of concepts
  (sign).
- **G3:** NTE = M1-transfer / native ≥ 0.40.

## Outcome table (pre-committed)

- PC pass + G1 + G2(mag) → **`WRITE_DOOR_OPENS__control_was_capacity_limited`** — the law
  falls, by our own hand; the read≠write papers get a correction-lineage update at full
  volume, exactly like v31.1.
- PC pass + ¬G1 + ¬G2(mag) → **`READ_NEQ_WRITE_SURVIVES_CAPACITY`** — the dissociation
  holds under matched-supervision nonlinear transfer at the steer-optimal point: its
  strongest surviving test, and the paper's claim upgrades accordingly.
- PC pass + mixed gates → `REPORT_AS_LANDED` with every number; no directional claim.
- L1 comparator and ε-stability reported in all branches.

## Compute & discipline

Two model loads (Llama family only — the shard-kill class was gemma; llama loaded clean five
times this week), detached execution per the c80 rail, extraction + ~1,700 greedy generations
≈ 1–2 h on the 8 GB GPU, $0. Smoke = 5 concepts, `_smoke` files, INVALID-only. Every number
re-derives from `b36_result.json`; OATH-certified before commit.
