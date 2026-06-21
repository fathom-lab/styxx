# PREREG — clearing G0 to settle the cross-model WRITE channel (the read↔write gap)

**Frozen 2026-06-20, before any upgraded-instrument or transfer number is read. Fathom Lab / styxx.**
Successor to `PREREG_thought_transfer_2026_06_06.md` + `FINDING_thought_transfer_2026_06_07.md`.

## Why this exists (the gap)
The 2,500-year program's surviving half — the geometry of meaning is universal (`SYNTHESIS_ancient_question_answered`,
`CAPSTONE_universal_mind`) — has been pushed to real models in one direction only. Cross-model **reading** of a
concept zero-anchor (no paired data) **partially works** (legibility top-1 0.15–0.48). Cross-model **writing** —
install a control direction from mind A into mind B through the unsupervised map and actually steer B — came back
**INCONCLUSIVE / ALIGNER_LIMITED**: the run's own positive control **G0 = 0.74 < 0.80**, so the cross-model null was
(correctly) ruled uninterpretable. The finding named the fix precisely: the loss is purely that ~23% of a concept's
steering vector lies **outside** the concept-PCA subspace built from only **N≈110** concepts; in-subspace random
directions transfer at **1.000** (the map/Q is exact). **G0 is clearable by raising N (more concepts → higher
feasible PCA dim k → more of each steering vector inside the subspace) and/or by a read layer where concept
directions sit more in-subspace.** This pre-registers that upgrade and the re-test. *(Resolving our own
instrument-limit before reading a result — the program's discipline, not a new claim.)*

## The two-stage design (the SEAL is load-bearing)
**Stage 0 — instrument upgrade, positive-control ONLY. No cross-model transfer number is computed or read until G0
is decided.** This is the anti-"false-falsify-Plato" guard from the parent prereg, enforced mechanically: Stage 0
code computes *only* `pc_cos` (the A→A-rotated held-out concept-direction reproduction) and never instantiates the
B-model transfer path.

- **Expand the concept battery** from N≈110 to **N≈480** balanced single-word concepts across the same kind of
  categories (animals, plants, food, tools, vehicles, furniture, clothing, nature, materials, buildings,
  instruments, body, professions, abstractions, emotions, actions) — deduped, lowercase, no multi-word items.
- **Anti-overfit selection split (frozen):** partition concepts into SELECTION (60%) and FINAL (40%) by a seeded
  shuffle. The map's PCA/Q are fit on SELECTION-train; the (layer, k) hyper-pair is chosen to maximize `pc_cos`
  **on SELECTION held-out concept directions only**; that pair is then **locked** and `pc_cos` is reported on the
  **FINAL** held-out concept directions. G0 is read off FINAL, never off the grid max — so a swept grid cannot
  inflate the gate.
- **Sweep grid:** inject/read layer `L ∈ {0.40, 0.50, 0.60, 0.70}·n_layers` (parent used 0.60 only); PCA dim
  `k ∈ {60, 90, 120, 150}` (feasible now that N_train ≈ 290 ≫ 110). Source model = `Llama-3.2-3B-Instruct`.

**Stage 1 — the test, run ONLY if G0 clears on FINAL.** The frozen `Llama-3.2-3B → Llama-3.2-1B` (RSA ≈ 0.965)
write-transfer test, gates **verbatim from the parent prereg** (G1 effect ≥0.15; G2 vs random ≥0.10 ∧ sign ≥0.7N;
G3 ceiling NTE ≥0.40; G4 specificity ≥0.10; G5 map-not-anything ≥0.10), at the locked (layer, k) and dose-locked
α. No gate is moved between stages.

## Frozen kill-gates / verdict tree
- **G0 (instrument): `pc_cos` on FINAL held-out concept directions ≥ 0.80.**
  - **G0 fails even at N≈480 over the full (layer, k) grid → `INSTRUMENT-CEILING` (a LANDED finding, not a null
    dodge):** the unsupervised concept-PCA map structurally cannot host a full steering vector — i.e. the substrate
    a zero-anchor *write* would ride does not live in the shared concept geometry. A real, publishable bound on
    "compute a control once, mount it on any model," and a clean reading-vs-writing dissociation *at the
    representational level* (reading rides the subspace; a steering vector does not fit in it).
  - **G0 clears → proceed to Stage 1 (the cross-model null/positive is now interpretable).**
- **Stage 1 (only if G0 clears), per the parent verdict logic:**
  - G1∧G2∧G3∧G4∧G5 → **`TRANSFER WORKS`** — a control computed once in one mind installs in another with zero
    shared data: the universal cross-mind **write** channel (foundation for a model-agnostic mountable conscience;
    and, dually, a zero-anchor cross-model representation-control **attack surface**).
  - G1∧G2 but G3<0.40 or G4 weak → **`PARTIAL`** (real but lossy write).
  - G1 or G2 fails (with G0 now VALID) → **`REPORT_AS_LANDED null`** — "reading ≠ writing across minds" becomes a
    landed result: minds share *what* they represent, but a control cannot be blindly installed zero-anchor. Deep
    and safety-relevant (you can read a foreign mind's concept; you cannot hijack its behavior through the map).

## Pre-registered predictions (frozen priors, stated to be killed)
1. **G0 clears at N≈480.** Prior **0.70**. The FINDING's k-sweep (0.705/0.742/0.769 at k=40/60/85, N=110) tracks
   subspace fraction and is *rising* but plateauing under the N=110 ceiling; 4× the concepts should lift the feasible
   subspace coverage past 0.80, most likely at a deeper layer (0.50–0.70) and k≈120. Could fail if concept steering
   vectors are intrinsically high-rank (spread across many PCs) rather than subspace-confined.
2. **If G0 clears, Stage 1 lands `PARTIAL`, not `TRANSFER WORKS`.** Prior **0.45 PARTIAL / 0.25 WORKS / 0.30
   REPORT_AS_LANDED**. Reading at top-1 0.15–0.48 suggests the map is correspondence-lossy; a *specific* control
   direction is more sensitive to map imprecision than a read (the leading hypothesis from the FINDING). A clean
   `REPORT_AS_LANDED null` with a now-valid instrument would be the stronger, more surprising result.

## Honest ceiling / scope
One near-isometric same-lineage pair (Llama-3B↔1B, RSA 0.965); MiniLM steering-gain metric; shared training data
not controlled (the *practical*, not data-independent, frontier). A `TRANSFER WORKS` = "a control computed once
steers another model with zero shared data, in the near-isometric regime" — **not** "any model controllable from
any other." Cross-family (gemma, lower RSA) is expected ≤ this and is not run here. No consciousness claim; this is
the structure of artificial minds used as the testbed. The frequency/rhythm arc is the *control* that fixed where
universality does **not** live (mechanism); this tests how far it **does** (representation → control).

## Runnable
Existing rig: `run_thought_transfer.py` + `styxx_transfer.py` (TransferMap, self_test) + `introspection-gate`.
Upgrade = expand `CONCEPTS`, add the (layer, k) selection sweep with the SELECTION/FINAL seal, gate Stage 1 on G0.
8GB GPU, sequential model load (A freed before B), ~$0. Smoke (tiny N, 1 layer) before the full freeze-locked run.

## Amendment (2026-06-20, pre-data, conservative — no gate change)
Stage 1 additionally measures the zero-anchor **READ** (top-1 concept identification via `transfer_point`) on
the SAME map and SAME held-out concepts as the WRITE test, so the read-vs-write comparison is apples-to-apples
in one run (the lit-positioning makes read≠write the load-bearing axis). This is a reported diagnostic, not a
gate: the frozen G1–G5 write verdict is unchanged. Added before any Stage-1 number was read.
