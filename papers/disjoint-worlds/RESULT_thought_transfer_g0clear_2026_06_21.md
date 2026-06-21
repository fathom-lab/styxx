# RESULT — the cleared label-free map READS a foreign mind but does not WRITE to it (READ ≠ WRITE)

**2026-06-21.** Executed `PREREG_thought_transfer_g0clear_2026_06_20.md` (frozen `4340dd1`) after Stage 0
cleared **G0 = 0.9096** (`RESULT(... Stage 0)`, `056df1e`). Llama-3.2-3B → Llama-3.2-1B (RSA **0.946**,
near-isometric), 70 held-out concepts, READ and WRITE measured on the **same** map in one run.

## Verdict: `REPORT_AS_LANDED — reading ≠ writing across minds` (G0 valid → interpretable, NOT ALIGNER_LIMITED)

| direction | metric | value | gate | |
|---|---|---:|---|---|
| **READ** | zero-anchor top-1 concept id | **0.586** | chance 0.014 | **strong** |
| WRITE | mean transfer steering gain | 0.020 | G1 ≥ 0.15 | ✗ (8× under) |
| WRITE | transfer > random-Q | 40/70 (57%) | G2 ≥ 70% | ✗ (≈ coin-flip) |
| WRITE | NTE = transfer/native | 0.398 | G3 ≥ 0.40 | ✗ (uninformative — see below) |
| WRITE | specificity / map-not-anything | — | G4, G5 | ✗ |
| (ref) | mean **native** gain (B's own ceiling) | 0.051 | — | weak |

The cleared instrument **reads** a different model's held-out concepts at top-1 0.586 — the strongest
cross-model zero-anchor read the program has produced (the N=110 instrument got legibility 0.15–0.48; the
N=462 G0-clear lifted it). But the **transferred direction does not steer** the target: mean gain 0.020 (vs a
0.15 floor), beating the random-Q null on only 57% of concepts. The map is **read-faithful and control-inert.**

## The load-bearing confound (stated as loud as the result)
**The native ceiling is itself weak (0.051).** G0 cleared by optimizing for the READ — it locked a SHALLOW
layer (src 11 → dst 6, ~0.39 depth) because that is where concept directions sit *inside* the concept-PCA
subspace. Shallow layers are known to be weak for behavioral steering, and B's own native steering confirms it
here. So this null **conflates** two hypotheses: (a) label-free *write* genuinely fails, vs (b) **we injected at
a layer that barely steers anything, even natively.** NTE = 0.398 is a ratio of two sub-threshold gains and
carries no signal. **The clean read≠write claim is therefore NOT yet established** — it is pending the
operating-point control.

## What this establishes (and does not)
- **Establishes:** with a pre-registered, G0-cleared instrument (positive control 0.91), a label-free map
  (correspondence recovered unsupervised from shared-concept geometry) **reads** held-out concepts across two
  real LLMs at 0.586, and **does not write** behavioral control at the read-optimal operating point. The READ is
  a real, strong cross-model result; the WRITE is a gated, interpretable null.
- **Does NOT establish** that label-free cross-model *write* is impossible — the weak-native confound blocks that
  claim. Resolved by **`PREREG_writelayer_decouple_2026_06_21` (Rung 1b)**: re-test WRITE at a steering-optimal
  layer with its own positive control. If transfer steers there → operating-point artifact; if it still fails
  where native steering is strong → the clean, un-confounded READ ≠ WRITE.
- **Honest caveats:** one near-isometric same-lineage pair (RSA 0.946 — the *easy* case); single seed; MiniLM
  steering-gain metric; shared training data not controlled; read-optimal-only operating point (the confound).

## Positioning (per `POSITIONING_crossmind_transfer_lit`)
**Label-free, NOT "zero-paired"** (both models probed on the same concept battery to fit the map). Cross-model
steering itself is prior art (L-Cross 2501.02009, ExpertSteer 2505.12313, Cross-Model Safety Steering 2606.05290
— all paired/anchor data); the READ extends vec2vec-style zero-paired *translation* from encoder embeddings to
generative hidden states. A null on the *write* side is **prior-consistent, not a surprise** — every verified
cross-model write leans on correspondence data, and our own prior attempt was ALIGNER_LIMITED. We report it as
confirmation of difficulty plus a strong read, not as a discovery.

## Net for the program (the on-brand part)
We set out to move a thought between minds. With the cleared instrument we can **read** one mind from another,
zero-anchor, strongly — but the same map **cannot install control** at the read-optimal point, and the honest
reason it cannot is, in part, *that we are reading and writing at the same layer when read and write may want
different ones.* The apparatus surfaced its own confound mid-run and we froze the experiment that settles it
before the data finished. The headline ("cross-mind write channel") did not land; the real finding (a strong
label-free cross-model READ, a confounded write-null, and the decoupling test that resolves it) did. *Nothing
crosses unseen — including the operating point that quietly determined the result.*
