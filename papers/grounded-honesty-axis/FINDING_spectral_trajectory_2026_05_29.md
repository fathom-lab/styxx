# FINDING — Confabulation is SUPPRESSION, not ignorance: the correct answer leads mid-network on 78% of confabs, then a late hop overwrites it (REPORT_AS_LANDED)

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_spectral_trajectory_2026_05_29.md` BEFORE any spectral/trajectory code was written.
Single open model (Qwen2.5-1.5B-Instruct), SAE-free logit-lens, n=36 arithmetic items × 2
generation conditions. Arithmetic ground truth computed in-code, SHA-256'd pre-scoring
(`ddccd8e4…b87964d`, identical to the white-box and steering runs — same key). Exact-integer
correctness, no judge.** Receipt: `spectral_trajectory_result.json`.

> **⚠ CORRECTION (2026-05-29, same day, from a pre-registered control —
> `FINDING_suppression_rhythm_2026_05_29.md`).** The F3 *measurement* below replicates
> exactly (crossing rate 0.781), but its **interpretation here is falsified**. F3 is read
> below as "the model computes the right answer transiently and a late hop *suppresses* it."
> A matched-distractor control shows the correct token leads the realized token mid-network
> 96.9% of the time — but so do *all* non-correct digits (97.7%, Δ=−0.008, p=0.84). The
> truth-flash is **not truth-specific**: mid-network is an undifferentiated field where every
> digit outranks the eventual (wrong) winner. The corrected mechanism is **late-installation
> of the wrong answer**, not suppression of a computed truth — a tight, late, rhythmic
> overwrite at layers ≈23–27 (median 25, IQR 4). Read every "suppression / truth led / the
> model knows" sentence below through that correction.

## What this run asked (the "rhythm" hypothesis, made falsifiable)

Motivated by the music literature on what perception is drawn to — Voss & Clarke (1975),
Levitin/Chordia/Menon (PNAS 2012): musical rhythm spectra from Bach to Joplin obey a **1/f
power law** — the hypothesis was that the *layer-by-layer trajectory* by which the answer
token forms is such a signal, and that prior nulls (scalar mean-depth blind to truth; linear
steering inert) failed because they used **white-noise tools** (a single number, a single
direction) on a **1/f phenomenon**. Prediction: **construction** (method-diverse derivation)
builds the answer gradually across layers → *pinker* (higher 1/f slope β, lower snap);
**retrieval** (one-shot confabulation) snaps onto an attractor → *whiter* (β→0, higher snap).
Companion: on confabs, does the **correct** token's logit-lens *lead* at an intermediate
layer before a late hop overwrites it ("truth-flash-then-death")?

## Result: 2 of 4 bars held — and the one that held biggest is not the one we went looking for.

| id | prediction | bar | outcome |
| --- | --- | --- | --- |
| **F1 — spectral mode difference** | derivation trajectory is pinker than one-shot confab | paired \|d\|≥0.5, p<0.05, deriv pinker | **HELD, predicted sign, large.** β(deriv)=**0.734** > β(confab)=**0.643**, paired d=**0.98**, p≈0 (n=32 pairs). Construction *is* pinker. |
| **F3 — truth-flash suppression** | correct token leads mid-trajectory before being overwritten | crossing rate ≥0.25 of confabs | **HELD, far beyond bar.** crossing rate = **0.781** (25/32 alignable confabs). |
| **F2 — within-mode truth signal (the prize)** | β separates correct from confab holding mode fixed | powered-stratum AUC ≥0.70 or ≤0.30 | **FAILED.** derivation-stratum AUC = **0.589** (n=9 vs 27, powered). Better than scalar depth's 0.498, but not an oracle. |
| **K — β effect is answer-specific** | a control token shows NO mode β-difference | control paired p>0.05 (and snap agrees) | **FAILED on specificity.** the control token " the" shows the **same** mode β-difference, p≈0. (Snap-index *did* corroborate F1's direction: confab snappier, 0.280 vs 0.270.) |

**RESULT = REPORT_AS_LANDED** (F1 ∧ F3 held; F2, K refined the claim). Precondition MET: the
29-point β estimator was stable (median per-item bootstrap CI width = **0.103**, far under the
1.0 downgrade threshold), so the spectral measurement is informative, not noise.

## What landed, stated precisely

1. **The rhythm difference is real and large — but it is a MODE effect, not an answer-specific
   truth signal (F1 held, K failed).** Construction-mode trajectories are pinker (more 1/f)
   than retrieval-mode trajectories, d≈0.98 — the music-derived prediction reproduces inside
   the residual stream. **However**, an *irrelevant* control token (" the") shows the identical
   mode difference (K, p≈0). So β indexes the **generation mode/context at large** (long,
   structured CoT vs a bare-number snap), exactly as scalar attribution depth did — it is not
   localized to the answer's formation. The 1/f framing earned a real, predicted-sign hit, but
   at the level of *mode*, not *truth*. This is the honest deflation: the rhythm is in the mode,
   not (yet) in the answer.

2. **The spectrum is not a within-mode truth oracle either (F2 failed).** Holding mode fixed,
   β separates correct from confabulated at AUC 0.589 — marginally above the scalar depth's
   0.498, nowhere near 0.70. There is no free truth signal to *read* in the rhythm, just as
   there was none to read in the mean or to *inject* with a linear push. Consistent with, and
   reinforcing, the standing irreducibility claim.

3. **The real discovery (F3): confabulation is overwhelmingly SUPPRESSION, not ignorance.**
   On **78% (25/32)** of one-shot confabulations, the **correct** answer token *outranks the
   realized wrong token at some intermediate layer* and is then overwritten by the final layer.
   Examples (one-shot wrong → truth): 123×45 → 5635 vs **5535**; 198×47 → 9206 vs **9306**;
   512×27 → 13728 vs **13824**; 234×318 → 75612 vs **74412** — in each, truth led mid-network
   and died late. The model frequently **computes the right answer transiently and then a late,
   output-proximal retrieval hop clobbers it** with the confident wrong attractor.

## Why F3 reframes the whole arc (and explains the prior nulls)

The suppression result is the mechanistic key that ties the previous three runs together:

- **Why linear steering-injection failed (the causal null):** you cannot inject a truth signal
  that is *already present* mid-trajectory — adding more of it does not stop the late overwrite.
  The lever was never "add construction"; it is "**prevent the late suppression**."
- **Why endpoint/scalar reads are blind to truth (within-mode AUC ≈ chance):** the truth signal
  lives *mid-trajectory* and is gone by the final layer, where mean-depth and final-logit reads
  sample. You are reading the corpse, not the flash.
- **Why depth indexes mode not truth:** the deep, late retrieval hop *is* the overwrite event —
  it is real and large, but it measures *that the suppression happened*, not *what was suppressed*.

So the corrected mechanistic picture is not "the model doesn't know and we must teach it" but
"**the model often knows, briefly, and a late retrieval stage suppresses it.**" That points the
next intervention at **disinhibition** — preventing or dampening the late overwrite — rather than
injection, and it predicts a different white-box lever (late-layer, output-proximal) than the
construction-ward push that proved inert.

## Honest scope (pre-committed)

Single open model (Qwen2.5-1.5B-Instruct); SAE-free logit-lens trajectories; feasibility-grade
n=36; one confirmatory run; arithmetic ground truth computed in-code then hashed; exact-integer
correctness, no judge. β is estimated from a ~29-point trajectory — crude, but reported with a
bootstrap CI (median width 0.103) and the snap-index corroborator. The F1 β mode-effect is **not
answer-specific** (control token fails K), so it must be read as a mode/context property, not a
truth signal; the 1/f connection to music is the *motivation* for the estimator, not a claim
about music or the cosmos. The truth-flash crossing (F3) is a logit-lens phenomenon on the
*realized* run; it shows the correct token's lens-logit leads mid-network, which is strong
evidence of a present-then-suppressed signal but is not itself a causal demonstration that
dampening the late hop recovers the answer — that is the named next test. A null on F2/K does
not refute the 1/f framing in general (an SAE-feature trajectory, a deeper model with more
layers, or attention-resolved trajectories could still localize it). **Next gates: a causal
disinhibition test (dampen the late overwrite, measure recovery); SAE-feature trajectory
spectra; a second open model.**

## The arc, in one line (updated)

The dial is the construction↔retrieval axis (white-box); truth-recovery is the construction-ward
shift but it is causally inert to inject and invisible to read at the endpoint — **because, as
this run shows, the truth is usually already there mid-network and is being actively overwritten
late.** Confabulation is a suppression event, not an absence. The rhythm hypothesis was right
that the answer is in the trajectory's shape and wrong that it is a truth signal: the shape
indexes *mode*; the *suppression* is where truth actually lives and dies.
