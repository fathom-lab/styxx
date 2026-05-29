# FINDING — The construction↔retrieval axis is causally MOVABLE but correctness-INERT under linear residual steering (REPORT_AS_LANDED)

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_depth_steering_causal_2026_05_29.md` BEFORE any steering code was written.
Single open model (Qwen2.5-1.5B-Instruct), SAE-free logit-lens DLA depth, residual-stream
difference-of-means activation steering with a norm-matched sham control and
train/test-locked hyperparameters. n=36 items, 32 one-shot confabulations split
train(17)/test(15) by even/odd index. Arithmetic ground truth computed in-code, SHA-256'd
pre-scoring (`ddccd8e4…b87964d`, identical to the white-box run — same answer key).
Exact-integer correctness, no judge.** Receipt: `depth_steering_causal_result.json`.

## What this run asked (Pearl Level 3)

The white-box unification (`FINDING_depth_grounding_whitebox`) established
**observationally** that attribution depth is the substrate of the belief→truth dial:
one-shot confabulation is deep/retrieval (D≈21–22), method-diverse derivation is
shallow/construction (D≈19–20), and on items where derivation *recovers* truth the depth
shifts construction-ward (+1.86 layers, p=0.002). The within-mode addendum showed depth is
dial **position**, not a fixed-position truth read.

Everything there is Level 2 (changing how we sample). This run is **Level 3**: do we
change the OUTCOME by **intervening** on the mechanism? We derived the construction−retrieval
direction `v_ℓ` (mean answer-position residual, method-diverse minus one-shot, across all 36
items), and during one-shot generation added `α·rms·v_ℓ` to the layer-ℓ residual at every
position via a forward hook. If pushing the computation construction-ward *causes* a
confabulating one-shot answer to flip correct — beyond a norm-matched sham push — the axis
is the causal lever. If not, the axis is causally upstream of *how* the model derives an
answer but is not a directly injectable correctness knob.

## Result: 2 of 4 bars held. The two that held draw the sharper line — and it's the honest-prior line.

| id | prediction | bar | outcome |
| --- | --- | --- | --- |
| **S3 — manipulation check** | the intervention actually moves the mechanism: steered one-shot DLA depth < unsteered | paired p<0.05, steered shallower | **HELD, predicted sign.** depth shift on−off = **−0.298 layers**, p = **0.00043** (steered construction-ward, shallower). The push reaches the mechanism. |
| **S2 — not degeneracy** | steered generations remain valid integers | valid-integer rate ≥0.80 | **HELD.** validity = **1.00** under steering. Outputs stayed numeric — recovery, had it occurred, would have been real not garbage. |
| **S1 — causal recovery (decisive)** | construction-ward steering flips held-out confabs wrong→correct, beyond sham | flip rate ≥0.15 AND (real−sham) ≥0.10 | **FAILED — clean null.** real flip = **0.00**, sham = **0.00**, real−sham = **0.00**. *Every* grid cell (ℓ∈{6,12,18} × α∈{4,8,12}) produced **zero** wrong→correct flips on TRAIN; there was no operating point to lock. |
| **K — does not wreck what works** | construction-ward steering doesn't break correct items more than sham | real break-rate ≤ sham + 0.10 | **FAILED.** real break = **0.25** vs sham = **0.00** on initially-correct one-shot items. The push is *not inert* — it knocks 1-in-4 correct answers off, while the sham knocks off none. |

**RESULT = REPORT_AS_LANDED** (S2 ∧ S3 held; S1 ∧ K did not — and the precondition was MET:
n_test=15 ≥ 8, so S1 is a powered null, not an under-powered one).

## The precondition was MET — this is a real test, not a feasibility shrug.

15 held-out confabulating one-shot items (≥8 required). The grid was searched on 17 train
confabs. The S1 null is therefore informative: it is not "too few items," it is "the
intervention had the opportunity and did not recover truth."

## What landed, stated precisely

1. **The axis is causally WRITABLE (S3).** The difference-of-means direction is not a
   passive correlate. Hooking `α·rms·v_ℓ` onto the residual stream during generation moves
   the realized-answer DLA depth construction-ward by ≈0.30 layers (p=0.0004). We can turn
   the dial. This is the causal half of the white-box claim, confirmed for the first time.

2. **Turning the dial does NOT transport correctness (S1 — the decisive null).** Across the
   entire (ℓ,α) grid, construction-ward steering flipped **zero** confabulations to correct,
   on train or held-out test, indistinguishable from a random direction (real = sham = 0.00).
   The belief→truth dial that resampling and re-derivation move (black-box AUC 0.694→0.955;
   white-box +1.86 layers) **cannot be driven by a linear residual-stream push along this
   direction.**

3. **The push has real causal force — just the wrong kind (K).** It is not that steering
   does nothing: it breaks 25% of already-correct one-shot answers (sham breaks 0%). So the
   intervention demonstrably perturbs the output distribution — it can knock a *correct*
   answer off its attractor — but it **cannot knock a wrong answer onto the right one.** The
   effect is real and asymmetric: destructive, not constructive.

## The sharpened claim (this is the result, not a walk-back)

**Attribution depth is a read/write coordinate on the construction↔retrieval axis — but it
is correctness-inert under linear steering.** We can *write* to it (S3: depth moves), and we
proved the write has causal consequences (K: it degrades correct items). What we cannot do
is *write truth into it*: pushing construction-ward does not re-derive the answer, it only
nudges the depth coordinate, and the answer does not follow (S1).

This closes the loop with the two prior observational results and points the same way:

- **Observationally (W2, within-mode):** there is no free truth signal to *read* at a fixed
  dial position. Depth = mode, not correctness.
- **Causally (this run):** there is no truth signal to *inject* by moving the dial linearly.
  Moving depth ≠ moving truth.

Both halves say the dial's truth-recovery is the **re-derivation computation itself** — the
method-diverse forward passes that actually re-do the arithmetic — not the depth coordinate
those passes happen to occupy. **The construction↔retrieval axis is causally upstream of
*how the model arrives* at an answer (via genuine re-derivation), not a directly injectable
correctness lever.** The expensive resampling/re-derivation backend earns its cost; it
cannot be shortcut by a residual nudge. This is exactly the honest prior the prereg named
(failure mode #3): "S3 passes, S1 is the real gamble… a clean S1 null is a genuine result."

## The arc, in one line (updated)

The root proved construction-vs-retrieval is real and invisible to text. The frontier
proved a belief→truth dial exists black-box. The white-box run showed they are one axis and
that truth-recovery *is* the construction-ward shift. **This run shows that shift is the
*signature* of re-derivation, not its *cause*: we can move the depth coordinate causally, but
correctness rides on re-doing the computation, not on where the residual stream sits.**

## Honest scope (pre-committed)

Single open model (Qwen2.5-1.5B-Instruct); SAE-free logit-lens DLA depth (a **proxy** for
the published Gemma Scope SAE/IG metric); difference-of-means residual steering along **one**
derived direction with a sham control and train/test-locked hyperparameters; feasibility-grade
n=36 (15 held-out confabs); one confirmatory run; arithmetic ground truth computed in-code
then hashed pre-scoring; exact-integer correctness, no judge. **This tests the causal
sufficiency of a *linear residual-stream push* along one difference-of-means direction —
nothing more.** A null here does **not** rule out causality via other interventions:
activation patching, attention-pattern edits, finetuning, or steering along a richer
(e.g. SAE-derived, layerwise-composed, or token-conditional) direction could still transport
correctness. The result is "this particular lever doesn't pull truth," not "the axis is
acausal." Next gates unchanged: canonical Gemma Scope SAE direction; a second open model;
non-linear / multi-layer interventions.
