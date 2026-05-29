# PREREG — Can we CAUSALLY turn the dial? Steer construction-ward, test whether truth recovers (white-box, Pearl Level 3)

**Pre-registered 2026-05-29, BEFORE any steering code is written or run. One
confirmatory run. Feasibility-grade: single open model (Qwen2.5-1.5B-Instruct), SAE-free
logit-lens DLA depth, residual-stream activation steering, n≈32 one-shot confabulations
split train/test.** Receipt to be committed at `depth_steering_causal_result.json`.
Arithmetic ground truth computed in-code (the `run_competence_cliff` SPECS) and SHA-256'd
before any scoring. Correctness = exact integer match (no judge).

## Why this run (the causal step)

The white-box unification established, **observationally**, that the belief→truth dial is
the construction↔retrieval axis: one-shot confident confabulation is deep/retrieval
(D≈21-22), method-diverse derivation is shallow/construction (D≈19-20), and on items
where derivation *recovers truth* depth shifts construction-ward (+1.86 layers, p=0.002,
W3). The within-mode addendum showed depth is dial **position**, not a fixed-position
truth read — so truth must be recovered by **moving** the dial, not thresholding it.

Everything so far is Pearl Level 2 (changing how we sample). This run is **Level 3**: do
we change the OUTCOME by **intervening** on the mechanism? If pushing the model's
computation construction-ward *causes* a confabulating one-shot answer to flip correct —
more than a norm-matched sham push — then the construction↔retrieval axis is not just
*correlated with* the dial, it *is* the causal lever. That is the breakthrough-grade
claim. It may not transfer (steering a difference-of-means direction during bare-number
generation is genuinely uncertain); both outcomes are reported.

## Apparatus (committed before data)

- **Model:** Qwen2.5-1.5B-Instruct, local, greedy decoding.
- **Construction direction `v_ℓ`:** at each transformer layer ℓ, the difference of mean
  residual-stream activations over answer-token positions, **method-diverse derivation
  minus one-shot**, averaged across ALL 36 items (this *defines the axis*; it is not the
  outcome being tested). Unit-normalized per layer.
- **Intervention:** during a one-shot generation, a forward hook adds `α · v_ℓ` (scaled
  by the residual-stream RMS at that position) to the layer-ℓ residual at every position.
  Construction-ward = the sign that DECREASES measured DLA depth (verified by S3).
- **Sham control:** a random unit direction (fixed seed), same per-position scaling,
  same layer. Isolates "construction-specific" from "any perturbation of this norm."
- **Hyperparameter locking (no fishing):** layer ℓ* and scale α* (small grid
  ℓ∈{6,12,18}, α∈{4,8,12}) are selected ONLY on the TRAIN split of confabulating one-shot
  items (maximize wrong→correct flips), then **locked** and evaluated ONCE on the TEST
  split. Train/test split is deterministic (even/odd item index).

## Predictions (decisive bars — pre-stated, scored exactly, both directions named)

| id | prediction | bar |
| --- | --- | --- |
| **S1 — causal recovery (decisive)** | on HELD-OUT confabulating one-shot items, construction-ward steering at locked (ℓ*,α*) flips wrong→correct, beyond sham | flip rate ≥ **0.15** AND (real − sham) ≥ **0.10**. If sham ≈ real, report "perturbation helps, NOT construction-specific" — a null for the causal claim. |
| **S2 — not degeneracy** | steered generations remain valid (parse to an integer), so recovery is real not garbage | valid-integer rate ≥ **0.80** under steering (flips counted only on valid outputs) |
| **S3 — manipulation check** | the intervention actually moves the mechanism: steered one-shot DLA depth < unsteered, same items | paired p < **0.05**, steered shallower |
| **K — does not wreck what works** | construction-ward steering does not destroy already-correct one-shot items more than sham | real break-rate on initially-correct items ≤ sham + 0.10 |

**RESULT = SURVIVED iff S1 ∧ S2 ∧ S3 ∧ K.** Otherwise REPORT_AS_LANDED with whatever
held, reported against prediction — including the likely outcome that S3 passes
(we can push depth) while S1 fails (depth is causally inert for correctness), which would
itself sharpen the claim: the dial is read-only at the residual-stream level and truth
requires genuine re-derivation, not a depth nudge.

## Precondition / honest failure modes (stated in advance)

1. **Too few held-out confabs.** If the TEST split has < 8 confabulating one-shot items,
   S1 is under-powered → report descriptively, no SURVIVED claim.
2. **Steering breaks generation.** If S2 fails (steered outputs are non-numeric garbage),
   the intervention is too strong; α* selection on TRAIN should avoid this, but if no α
   keeps validity ≥0.80 with any recovery, report as "no usable operating point."
3. **Honest prior.** I expect S3 to pass (the direction can move depth) and S1 to be the
   real gamble — difference-of-means steering during 16-token bare-number generation may
   not carry the semantic content needed to flip an answer. A clean S1 null is a genuine
   result: it would mean the construction↔retrieval axis is causally upstream of *how the
   model arrives* at an answer (via re-derivation) but not a directly injectable
   correctness lever.

## Honest scope (pre-committed)

Single open model, SAE-free logit-lens depth, difference-of-means residual steering with
a sham control and train/test-locked hyperparameters, feasibility-grade n≈32, one
confirmatory run; arithmetic ground truth computed in-code then hashed; exact-integer
correctness, no judge. This tests causal sufficiency of a *linear residual-stream push*
along one derived direction — not the full mechanism; a null does not rule out causality
via other interventions (patching, attention edits, finetuning). Next gates unchanged:
canonical Gemma Scope SAE direction; a second open model.
