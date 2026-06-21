# PREREG — decouple the READ point from the WRITE point (Rung 1b)

**Frozen 2026-06-21, before any steer-layer transfer number. Fathom Lab / styxx.** Motivated by a confound
surfaced in Rung 1's LIVE data (recorded here before Rung 1 even finished, so the design can't be reverse-fit).

## The confound (from Rung 1 partial, 43/70)
Rung 1 cleared G0 by optimizing for the **READ**: it locked a SHALLOW layer (src 11 → dst 6, ~0.39 depth)
because that is where concept directions sit *inside* the concept-PCA subspace (pc_cos 0.91). READ is strong
there (zero-anchor top-1 0.586). But the WRITE failed its gates (mean transfer +0.021 vs +0.15 floor; 53%
beat random vs 70% needed) **AND native steering at that layer is itself weak (mean native gain +0.039).**
So the write-null is **confounded**: we cannot tell "label-free control transfer fails" from "we injected at a
layer that barely steers anything, even natively." Shallow layers are good for reading concepts and known to be
weak for behavioral steering. **Read-good and steer-good layers may simply be different layers.**

## Question
Does the label-free map STEER a foreign model when injection happens at a **STEERING-optimal** layer (chosen by
native steering efficacy), each direction read/written at its own optimum — and does the cleared-subspace
positive control even hold at that layer?

## Design (3 stages, sealed)
- **Stage A — find the steer-optimal write layer L_w (NATIVE only; no transfer number computed).** On dst,
  sweep candidate layers `f·n_layers` for `f ∈ {0.40, 0.50, 0.60, 0.70}`; at each, measure mean NATIVE steering
  gain over a SELECTION subset of held-out concepts. Lock `L_w = argmax` native gain on SELECTION; report native
  gain on the FINAL subset (anti-overfit seal, mirroring G0).
- **Stage B — instrument at L_w.** Extract A & B concept clouds at the matched fraction `f*` (src `round(f*·nlA)`,
  dst `L_w`); fit the transfer map; compute the positive control `pc_cos` at L_w (A→A-rot concept-dir).
- **Stage C — WRITE test at L_w** (transfer / native / random-Q / wrong-concept), gates G1–G5 verbatim from the
  parent — run ONLY if Stage B's pc_cos ≥ 0.80 AND Stage A's native gain ≥ 0.15.

## Frozen kill-gates / verdicts
- **VOID-NO-STEER:** if no candidate layer reaches native gain ≥ 0.15, the 1B simply does not steer on this
  concept task at any layer → the whole write line is VOID here (not a transfer claim either way; report and stop).
- **`pc_cos < 0.80` at the steer-optimal L_w → `STRUCTURAL-TENSION` (a landed finding):** read-good and
  steer-good layers are subspace-incompatible — the concept-PCA subspace that hosts a steering vector (read) is
  NOT the layer where steering is behaviorally strong. A structural reason label-free *write* is hard, distinct
  from "the map is bad."
- **pc_cos ≥ 0.80, native ≥ 0.15, G1∧G2∧G3 →** `WRITE-WORKS-AT-STEER-LAYER` — **the Rung-1 read≠write was an
  OPERATING-POINT ARTIFACT**: read and write both succeed, each at its own optimal layer. Label-free cross-model
  control transfer holds; the dissociation was about *where you inject*, not *whether you can*.
- **pc_cos ≥ 0.80, native ≥ 0.15 (strong), G1 or G2 FAILS →** `CLEAN READ≠WRITE` — the map reads a foreign mind
  but **cannot install control even where steering is behaviorally strong.** The load-bearing dissociation, now
  un-confounded.

## Pre-registered priors (to be killed)
~0.40 `WRITE-WORKS-AT-STEER-LAYER` (the confound was real) / ~0.40 `CLEAN READ≠WRITE` (label-free write genuinely
fails) / ~0.20 `STRUCTURAL-TENSION`. The decoupling is the methodological fix that makes the dissociation — if it
survives — actually mean something.

## Scope & lit (per `POSITIONING_crossmind_transfer_lit`)
Label-free (NOT zero-paired — shared concept battery). Cross-model steering is NOT novel; the contribution is the
**read-point ≠ write-point map** and an un-confounded read-vs-write dissociation. Llama-3B→1B; rate + native/
random/wrong controls. Build: parameterize `run_g0_stage1.py` with a `--write-layer-sweep`; runs after Rung 1's
final verdict (so L_w can use Rung 1's full native-gain rows). 8GB, sequential, ~$0.
