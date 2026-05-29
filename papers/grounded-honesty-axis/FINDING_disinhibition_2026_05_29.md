# FINDING — Disinhibition SURVIVED: the late hop (layers ≈22–26) is CAUSALLY the install, and removing it yields UNCERTAINTY, not truth — the first causal confirmation of the corrected mechanism

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_disinhibition_2026_05_29.md` BEFORE any code for this test was written. Single open
model (Qwen2.5-1.5B-Instruct), SAE-free full-vocab logit-lens readout, the SAME n=36 arithmetic
items as the spectral/suppression/repair runs. Arithmetic ground truth computed in-code,
SHA-256'd pre-scoring (`ddccd8e4…b87964d`, identical to every prior white-box/steering/spectral/
suppression/repair run — same key). Exact-integer correctness, no judge. Greedy/deterministic.**
Receipt: `disinhibition_result.json`.

## Why this run exists: the causal test of the corrected mechanism

The suppression-rhythm control falsified our own "truth-flash" headline and replaced it with a
*descriptive* claim: confabulation is the **late, tight installation** of a confident wrong
answer at layers ≈23–27, over a mid-network field where no token (truth included) is privileged
(`FINDING_suppression_rhythm_2026_05_29.md`). This run asks the two causal questions that
descriptive claim could not answer:

1. **Is that late band actually what installs the wrong answer**, or is it epiphenomenal?
2. **When you remove the install, what is underneath — a waiting truth (suppression after all)
   or just uncertainty (installation over a flat field)?**

The intervention (pre-registered, uses no answer key to build): teacher-force
`prompt + realized_answer` and, **at the single divergence position**, attenuate a band of
decoder layers' residual *write* — `h_out → h_in + γ·(h_out − h_in)`, γ=0 = full knockdown of
that layer's contribution at that position. Read the next-token distribution there under the
hook. **Target band = decoder layers [22, 26]** (the measured install, hidden-state idx ≈23–27;
**fixed from the prior finding, not re-tuned**). **Control band = early layers [6, 10]** (matched
size), the guard against "any ablation disrupts the answer."

## Result: SURVIVED. n=36, 18 usable confabs (clean baseline + alignable), powered.

| id | prediction | bar | outcome |
| --- | --- | --- | --- |
| **I1 — the late band CAUSES the wrong commitment** | target knockdown removes the commitment far more than the early control | f_target − f_ctrl ≥ 0.30 AND f_target ≥ 0.50 AND discordant sign-test p < 0.05 | **HELD, decisively.** f_target = **0.889** (knockdown removes the wrong commitment on 16/18 confabs); f_ctrl = **0.222** (early band only 4/18); **Δ = 0.667**. Discordant pairs **13 target-only vs 1 control-only**, sign-test **p = 0.00092**. |
| **I2 — disinhibition yields UNCERTAINTY, not truth** | among removed-commitment items, truth does NOT take over and the distribution flattens | truth_recovery_rate < 0.34 AND paired entropy rise p < 0.05, mean > 0 | **HELD (installation branch).** Of the 16 removed-commitment items, truth recovers on **1 (rate 0.0625)** — far below 0.34. Entropy rises **+7.86 nats** (paired p ≈ 3.6e-8). Removing the install exposes a near-flat field, **not** a waiting answer. |
| **I3 — dose-response (corroborator)** | commitment dissolves monotonically as the band write is dialed down | Spearman ρ(γ, commitment-rate) ≥ +0.90 | **HELD perfectly.** commitment rate by γ {1.0, 0.75, 0.5, 0.25, 0.0} = {1.00, 0.89, 0.67, 0.17, 0.11}; **ρ = 1.000**. |

**RESULT = SURVIVED** (required I1 ∧ I2-installation). Coherence floor: target γ=0 left **55.6%**
of argmaxes numeric (≥0.50), so the primary γ=0 scoring stands — the pre-named γ=0.5 fallback
(coherence 0.944) was **not** triggered. The early-band control sets the destructiveness
baseline (22.2%); the late band is **4× more** likely to un-install the answer, and the
discordance is 13:1.

## What this means — the corrected mechanism is now causally confirmed

Three things are now established **causally**, not just by logit-lens observation:

1. **The late band installs the answer.** Knock down decoder layers [22, 26]'s residual write
   at the answer position and the confident wrong commitment disappears on ~89% of confabs —
   while the same knockdown on early layers [6, 10] disrupts it only ~22%. The install is
   **surgically localizable** to the band the suppression-rhythm run measured. The hop is not
   epiphenomenal; it is *where the commitment is written*.
2. **What it installs over is a flat field, not a hidden truth.** When the install is removed,
   the model does not fall back to the correct answer (6% recovery — at floor) — it falls into
   **high uncertainty** (+7.9 nats of entropy, a near-uniform readout). This is the causal
   confirmation of "installation over an undifferentiated field": there was **no computed truth
   underneath waiting to be released.** The falsified suppression reading (F3) does **not**
   come back on causal evidence — the pre-registered reverse branch (recovery ≥ 0.50 →
   suppression) did not fire.
3. **The commitment is graded, not all-or-nothing.** It dissolves monotonically as the write is
   dialed from full to zero (ρ = 1.0) — the install is a continuous late accumulation, exactly
   the picture the tight-late-hop geometry implied.

This is the first **SURVIVED** causal result in the white-box mechanism line. The prior causal
test (linear depth-steering) was writable-but-inert (`FINDING_depth_steering_causal`); this one
shows a *specific, localized, dose-responsive* causal handle on the confabulation commitment —
**but, crucially, a handle that buys uncertainty, not correctness.**

## Why this does NOT contradict the keystone null — it completes the picture

Tonight's keystone (`FINDING_repair_geometry`) showed the one-shot geometry does **not** predict
which confabs re-derivation can repair. This run shows we **can** causally dismantle the
one-shot commitment — and doing so yields *uncertainty*, not truth. Both point the same way:
**truth lives in the process of re-derivation, not in the single forward pass.** Disinhibition is
a lever on *confidence* (it can turn a confident wrong answer into an honest "I'm not sure"), not
a lever on *correctness*. That is a genuinely useful and honestly-bounded result: it is the
mechanistic basis for **abstention**, not for repair.

## Honest scope (pre-committed + observed)

- **Single-position, teacher-forced, single model.** The intervention attenuates one band's
  residual write at the divergence position and reads the next-token distribution there — it is
  **not** a multi-token regeneration, and not a claim about downstream sequence behavior. One
  open model, SAE-free logit-lens readout, feasibility-grade.
- **18 usable confabs** (vs ~32 in sibling runs): this test additionally requires a clean
  teacher-forced baseline (argmax == realized token) and an alignable divergence, which filters
  more items. Powered for I1 (≥12) and I2 (16 removed ≥ 6), but the n is modest.
- **Target band fixed from the prior finding, not tuned here** — this avoids circularity; the
  control band is early and matched in size. The Δ (target − control) is the guard against the
  intervention being merely destructive; the destructiveness baseline (22%) is explicitly
  reported.
- **A SURVIVED here confirms causal localization and the installation-not-suppression polarity;
  it does NOT claim truth recovery** (the opposite — recovery is at floor, by design and by
  result). A null would have refuted only the causal localizability of the install to this band
  by this method, leaving the descriptive finding intact.

## The arc, in one line (updated again)

The dial is construction↔retrieval (white-box); truth-recovery is the construction-ward shift
but it is causally inert to inject, invisible to read at the endpoint, not a privileged
mid-network signal, and not predicted by the one-shot geometry — **and now, causally: the late
band (layers ≈22–26) IS the install, and dampening it yields honest uncertainty, not truth.**
Every lever we have moves *confidence*; only re-derivation moves *correctness*.
