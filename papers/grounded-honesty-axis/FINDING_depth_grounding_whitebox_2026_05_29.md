# FINDING — Attribution DEPTH is the white-box substrate of the belief→truth dial (REPORT_AS_LANDED)

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_depth_grounding_whitebox.md` BEFORE data. Single open model
(Qwen2.5-1.5B-Instruct), SAE-free logit-lens DLA depth proxy, n=36 arithmetic items
× 2 generation conditions. Ground truth computed in-code, SHA-256'd pre-scoring
(`ddccd8e4…b87964d`). Exact-integer correctness, no judge.** Receipt:
`depth_grounding_whitebox_result.json`.

## What this run did (the unification)

For the first time the project's **root** (the fathom depth scorer / attribution-depth,
ICML 2026 MI workshop — *construction-vs-retrieval is a real internal distinction
invisible to surface text; recall is DEEPER than reasoning*) and its **frontier** (the
grounded honesty axis — *method-diverse re-derivation moves the grounded signal from
belief toward truth*) were put inside the **same experiment**, on the **same model**,
**white-box**.

Hypothesis (pre-registered): these are one phenomenon. Construction-vs-retrieval is the
single latent variable under both. Confident confabulation = the model in **retrieval
mode** (deep, late, output-proximal hop onto a wrong attractor); method-diverse
derivation works because it forces **construction mode** (earlier in the forward pass).
If so, attribution depth is the *mechanistic substrate* of the belief→truth dial —
observable white-box, only inferable black-box.

## Result: 2 of 4 bars held — and the two that held are the dial-mechanism bars.

| id | prediction | bar | outcome |
| --- | --- | --- | --- |
| **W1 — confabulation has a distinct depth signature** | paired depth differs, one-shot-confab vs method-diverse | \|d\|≥0.5, p<0.05; predicted **confab DEEPER** | **HELD, predicted sign.** paired Cohen's d = **2.47**, p ≈ 0, **confab deeper** (n=32 pairs) |
| **W3 — depth shift tracks the dial** | on items where derivation RECOVERS truth, depth shifts construction-ward, sign-consistent with W1 | paired p<0.05, sign-consistent | **HELD.** recovered items (n=5): one-shot−deriv depth = **+1.86**, p = 0.0024 (construction-ward), sign-consistent |
| **W2 — depth is a mode-independent validity signal** | depth separates CORRECT from CONFABULATED, pooling conditions | AUC ≥0.70 OR ≤0.30 | **FAILED.** AUC = **0.442** (≈ chance) |
| **K — not a length / easy-item artifact** | length-adjusted gap survives AND control tier shows no gap | intercept p<0.05 AND ctrl gap p>0.05 | **FAILED on the ctrl half.** length intercept = −2.15, p≈0 (gap survives length) BUT ctrl-tier gap p = 0.0007 (gap present on easy items too) |

**RESULT = REPORT_AS_LANDED** (W1 ∧ W3 held; W2, K did not).

## Precondition MET — the competence cliff transfers white-box.

27/27 hard-tier (mul_3x3 / mul_4x3 / multistep) one-shot answers were confident
confabulations; 32/36 overall. Qwen2.5-1.5B-Instruct confidently confabulates on hard
arithmetic exactly as the black-box arc's gpt-4o-mini did (e.g. 517×283 → one-shot
146091 vs truth 146311; 5678×432 → 2410976 vs 2452896). The run is informative.

## What landed, stated precisely

1. **The depth signature is real, large, and points the way the root paper predicted.**
   One-shot answers sit at depth **≈21–22**; method-diverse derivations at **≈19–20**.
   The gap is near-universal across all 36 items (paired d=2.47). One-shot bare-answer
   generation is **retrieval-mode: deep, late, output-proximal** — exactly the root's
   "recall is deeper." Method-diverse CoT is **construction-mode: earlier in the pass.**

2. **The gap is NOT a length artifact.** Regressing Δdepth on Δlength, the intercept is
   −2.15 (p≈0): even at equal generation length the derivation is ~2 layers shallower.
   The root's length-confound defense reproduces here white-box.

3. **Moving construction-ward is what recovers truth (W3 — the keystone).** On the 5
   items where method-diverse derivation flipped a one-shot confabulation to the correct
   answer, depth shifted construction-ward by +1.86 layers (p=0.0024), sign-consistent
   with W1. This is the **direct white-box correlate of the black-box belief→truth
   dial**: the same intervention that black-box resampling showed converts belief into
   truth (AUC 0.694→0.955) is observed here, internally, as a shift from deep-retrieval
   toward shallow-construction.

## What did NOT land — and what that teaches (the honest refinement)

The prereg's W2 and K framed depth as a **mode-independent oracle** — a number that, on
its own, classifies correct-vs-confabulated and is specific to confabulation. **It is
not.** Depth indexes generation **MODE** (retrieval-deep vs construction-shallow), and:

- **W2 fails (AUC 0.442 ≈ chance):** pool one-shot and derivation answers and depth no
  longer separates right from wrong, because the mode effect dominates — derivations are
  shallow whether right or wrong, one-shots deep whether right or wrong (mostly wrong).
  Depth measures **dial position, not truth directly.**
- **K fails on the control half (ctrl gap p=0.0007):** the deep-vs-shallow gap is present
  even on easy control items. It is a property of **mode**, present across difficulty —
  not a confabulation-specific tell.

This is the correct, sharpened claim, not a walk-back: **attribution depth is the
white-box substrate of the belief→truth DIAL — it measures where the model sits on the
construction↔retrieval axis. It is the mechanism the black-box arc could only infer, and
it confirms construction-vs-retrieval is one axis under both lines. It is not a
standalone truth detector; truth is recovered by *moving* along the axis (construction-
ward), which is exactly what W3 shows.**

## The unification, in one line

The root proved construction-vs-retrieval is real and invisible to text. The frontier
proved a belief→truth dial exists black-box. **This run shows they are the same axis:**
one-shot confident confabulation is retrieval-mode (deep); method-diverse derivation is
construction-mode (shallow); recovering truth *is* the construction-ward shift, observed
white-box for the first time.

## Addendum (2026-05-29) — within-mode secondary analysis CONFIRMS depth = dial position

The prereg named a next gate: "does depth separate right from wrong *holding mode
fixed*?" Pre-registered separately (`PREREG_depth_within_mode_2026_05_29.md`) with the
bar stated before computing the statistic; run as a secondary analysis on the same
greedy-deterministic generations (re-run reproduces the answer-key hash and W1/W2/W3/K
exactly). Two hypotheses pre-stated: **H_mode** (depth purely indexes generation mode,
blind to correctness within a mode) vs **H_residual** (depth carries a within-mode truth
signal). Bar: derivation-stratum AUC ≥0.70 or ≤0.30 ⇒ H_residual; in (0.30, 0.70) ⇒
H_mode.

**Result: H_mode, as predicted.** Within the method-diverse derivation stratum (the
well-powered one: 9 correct vs 27 confabulated), depth separates correct from
confabulated at **AUC 0.498 — chance.** Holding mode fixed, depth is blind to
correctness. (Within one-shot, AUC 0.84 but only n=4 correct — under-powered,
report-only, no claim drawn.)

This is the decisive lock on the interpretation: **attribution depth is not a
single-forward-pass confabulation detector.** It indexes *where on the
construction↔retrieval axis* a generation sits, nothing more. There is no free white-box
truth signal to read at a fixed dial position — which is exactly why truth is recovered
by *moving* the dial construction-ward (W3), not by thresholding depth (W2, and now WM).
The black-box belief→truth dial cannot be collapsed into a cheap one-pass read; the
expensive resampling backend earns its cost. The remaining white-box frontier is
therefore **causal** (steer the model construction-ward and test whether correctness
improves), not a better observational classifier.

## Honest scope (pre-committed)

Single open model (Qwen2.5-1.5B-Instruct); SAE-free logit-lens DLA is a **proxy** for
the published Gemma Scope SAE/IG metric — a signal here motivates the canonical SAE
confirmation, a null here would not refute it; feasibility-grade n=36, one confirmatory
run; arithmetic ground truth computed in-code then hashed pre-scoring; exact-integer
correctness, no judge. The depth gap is a **mode** effect (one-shot vs CoT), robust to
length but not isolated to confabulation. **Next gates (named, not done): canonical Gemma
Scope SAE depth; a second open model; a within-mode confab detector (does depth separate
right from wrong *holding mode fixed*?).**
