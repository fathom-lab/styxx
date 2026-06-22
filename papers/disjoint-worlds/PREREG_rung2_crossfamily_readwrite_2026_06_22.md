# PREREG — Rung 2: cross-FAMILY read≠write — the decisive test of the representation/mechanism law

**Frozen 2026-06-22, BEFORE any cross-family transfer data is collected or scored.**

## The claim under test (the candidate law of mind)

Across four years of this program's experiments, one asymmetry keeps independently precipitating
out: **minds converge on WHAT they represent, but not on HOW they control/process it. Meaning is
shareable between minds; control is not.** It has appeared in (1) concept geometry being
universal/structure-determined (`disjoint-worlds`, RSA-decisive), (2) rhythm being a dominated
substrate-specific mechanism (`frequency-resonance`), and (3) cross-mind **read≠write**: a
label-free map READS a foreign model's concepts (top-1 0.586, chance 0.014) but cannot WRITE/steer
them even at the steer-optimal layer (NTE 0.114 = 11% of native) — `RESULT_writelayer_decouple_2026_06_21`.

But (3) is **supported on ONE near-isometric, same-family pair** (Llama-3.2-3B → Llama-3.2-1B,
RSA 0.946). The `warp-crossworld` result already found the crack: under a nonlinear warp (genuinely
different minds), *what-it-means* still transfers but *unsupervised which-is-which recovery
collapses* — i.e. the program's confirmations live in the near-isometric regime. **This rung is the
decisive cross-family stress test: does the read≠write asymmetry hold across model FAMILIES
(different architectures, tokenizers, corpora — the non-isometric regime), or is it
isometry-bounded?** Either answer is a finding; the asymmetry being either architecture-general OR
quantitatively isometry-bounded is the science.

## Apparatus (reuses the G0-cleared instrument unchanged; only `--dst` varies)

- Source mind A = **Llama-3.2-3B-Instruct**. The G0 concept-correspondence instrument is CLEARED
  (pc_cos 0.91, locked layer 11 / k=150, N=462 concept bank, anti-overfit SELECTION/FINAL seal).
  G0 is a property of the SOURCE → **DST-independent**: one clear licenses every target.
- Per target B: `run_g0_stage1.py` (zero-anchor READ top-1 + same-layer WRITE on the SAME held-out
  concepts, apples-to-apples) AND `run_g0_stage1b.py` (native-steering sweep picks the
  STEER-optimal dst layer — the Rung-1b fix — then measures un-confounded WRITE there).
- Per-target instrument validity gates (pre-existing in the runner): **pc_cos ≥ 0.80** (positive
  control — the source vector is hosted in B's subspace, map interpretable) and **native steering
  ≥ 0.15 at the steer-optimal layer** (the write line is not VOID — B is steerable at all). A
  target failing either gate is reported as VOID and excluded from the WRITE-law bar (its write-null
  would be uninterpretable, not evidence).
- Controls reported for every WRITE number (never a hero anecdote): native ceiling, random-question
  direction, wrong-concept direction. Read is top-1 accuracy vs chance = 1/(n held-out concepts).

## Targets — the isometry → transfer curve (all cached, run sequentially, 8 GB, one probe at a time)

| target | family vs Llama | role |
|---|---|---|
| Llama-3.2-1B-Instruct | SAME family | high-RSA anchor (Rung-1b: read 0.586, NTE 0.114, RSA 0.946) — already run, included as the curve's anchor point |
| Qwen2.5-1.5B-Instruct | cross-family | cross-family probe |
| gemma-2-2b-it | cross-family | cross-family probe (prior: gemma PARTIAL) |
| Phi-3.5-mini-instruct | cross-family | cross-family probe |

RSA(A,B) on the shared concept battery is the moderator recorded for every target.

## Bars (pre-stated; report whichever way each lands)

Chance read top-1 = 1/(n held-out) ≈ 0.014. "× chance" below is relative to that.

- **R2-READ (per target).** Zero-anchor top-1 concept-ID accuracy. **≥ 10× chance SURVIVED-READ /
  3–10× REPORT / < 3× NULL.** Pre-stated to degrade with RSA.

- **R2-WRITE (per validity-passing target).** NTE = mean transfer gain / mean native gain at the
  steer-optimal layer. **≥ 0.50 WRITE-TRANSFERS (breaks read≠write) / 0.15–0.50 PARTIAL / < 0.15
  WRITE-NULL.**

- **R2-LAW (the headline).** Across validity-passing CROSS-family targets: the asymmetry holds iff
  the pattern is READ-works ∧ WRITE-null. **SURVIVED-LAW iff ≥ 2 cross-family targets show
  READ ≥ 3× chance AND WRITE NTE < 0.15.** This states the asymmetry is an architecture-general
  property, not an artifact of near-isometric same-family pairs.

- **R2-ISOMETRY (curve, with one bar).** Spearman(RSA, read_top1) across all 4 targets.
  **≥ 0.60 → read-fidelity is isometry-graded** (meaning-sharing degrades smoothly with
  mind-distance — quantifies how alien two minds can be and still share meaning). Reported
  descriptively regardless.

## Pre-stated predictions (on the record)

- **READ survives cross-family but graded by RSA.** I expect READ ≥ 10× chance on the highest-RSA
  cross-family target (likely Qwen-1.5B), degrading toward REPORT/NULL on the lowest (likely Phi or
  gemma). Same-family Llama-1B stays the ceiling (~0.59).
- **WRITE stays NULL on every target** (NTE < 0.15), even where native steering is strong — it was
  already null at RSA 0.946, and cross-family divergence should not *help* control transfer.
- **R2-LAW SURVIVED** — the asymmetry is architecture-general. The live risk that makes this worth
  running: if READ *also* collapses cross-family (only works near-isometric), the law is
  isometry-bounded, which is the more surprising and more precise discovery.

## What each outcome means

- **READ survives + WRITE null cross-family** → the representation/mechanism asymmetry is
  architecture-general → a candidate **law of mind** with a 4th independent confirmation, now across
  families: *read-only cross-mind oversight is the natural primitive* (converges with the
  cooperative-only, read-only conscience-mount result — `project_conscience_mount`).
- **READ collapses cross-family** → meaning-sharing is **isometry-bounded**; we report the RSA
  threshold below which a foreign mind becomes unreadable — a quantitative bound on mind-distance
  (connects warp-crossworld's "unsupervised recovery needs near-isometry").
- **WRITE lands (NTE ≥ 0.15) on a cross-family target** → read≠write is BROKEN; the honest novel
  core (label-free correspondence-RECOVERY + cross-model behavioral CONTROL) is real. Predicted
  unlikely; would be the largest result in the arc.

## Lit-check redlines (load-bearing — do NOT overclaim; from POSITIONING_crossmind_transfer_lit)

- **Cross-model steering is NOT novel** (L-Cross, LRTH, ExpertSteer, Cross-Model Safety Steering).
  Never write "first cross-model steering" / "move a thought between minds." The novel core, *if*
  write ever lands, is **label-free correspondence-recovery** (no anchors fed) **+ control** — every
  prior write uses paired/anchor data; every label-free method (vec2vec) stops at READing.
- **NOT "zero-paired" / "zero-shared-data."** Both models are probed on the SAME concept battery →
  frame as **label-free** (zero-anchor = no correspondence anchors fed to the map), never
  zero-shared-input.
- **Platonic Representation Hypothesis is CONTESTED** (post null-calibration). Motivation only; never
  state "models share a universal geometry" as established fact.
- Report RATE + all three controls; behavioral OUTPUT change for any write claim, never a readout/
  cosine; verify any future-dated arXiv id before formal citation.

## Honest bounds

In-silico; small open-weight models (1–3 B); the read≠write asymmetry remains demonstrated on one
near-isometric pair and this rung extends it to a handful of cross-family targets, not a population;
the WRITE bar uses NTE at a single steer-optimal layer; no consciousness or sentience claim of any
kind. This tests transfer of *representational content and behavioral control*, nothing more.

## Receipts

- Runners: `run_g0_stage1.py` (read+write same layer) + `run_g0_stage1b.py` (write at steer-optimal
  layer), `--src meta-llama/Llama-3.2-3B-Instruct --dst <target> --tag rung2_<slug>`. Reuses
  `run_g0clear.py` (462 bank), `run_thought_transfer` (extract/steer_gain/lock_dose), `styxx_transfer`.
- Outputs: `rung2_readwrite_result_<slug>.json` per target + `rung2_crossfamily_summary.json`.
- Finding: `FINDING_rung2_crossfamily_readwrite_2026_06_22.md` (results as-landed; the LAW reading,
  whichever way it falls).
