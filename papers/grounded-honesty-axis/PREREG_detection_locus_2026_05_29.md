# PRE-REGISTRATION — where does the confabulation-detection signal LIVE: single-pass internal state, or cross-derivation resampling?

**Written 2026-05-29, BEFORE any code for this test is written.** Tonight's confab-specificity
run (`FINDING_confabulation_specificity_2026_05_29.md`) showed the late-band causal install is
SHARED machinery — band-knockdown perturbs correct answers as much as confabs, so a single-pass
internal *intervention* cannot tell them apart. Yet the black-box grounding detector cleanly
separates confab from truth (AUC 0.966 on factual self-claims,
`[[grounded-honesty-ceiling-break-2026-05-28]]`). This run unifies the two lines by localizing
WHERE the detection signal lives, on the SAME balanced arithmetic set where we have full white-box
mechanism data:

**Is confabulation a CONFIDENT single-pass state (no internal-confidence read separates it from a
correct answer), detectable ONLY across independent re-derivations (resampling self-consistency)?**

If single-pass internal confidence (clean first-token entropy, logit margin) cannot separate
confab from correct, but resampling-based Stability can, then the detection signal lives in
cross-derivation VARIANCE, not single-pass confidence — the mechanistic statement of "confident
confabulation," and the reason grounding works where white-box single-pass does not. This would
unify the standing correctness bound ("only re-derivation moves correctness") with a matching
DETECTION bound ("only re-derivation detects it"), and explain the confab-specificity null.

Model: **Qwen2.5-1.5B-Instruct** (the white-box model — so all three signals are read from the
SAME network). Resampling protocol mirrors the validated grounding keystone (N=10, temperature
1.0). SAE-free full-vocab logit-lens for single-pass signals. In-code arithmetic ground truth
SHA-256'd pre-scoring, exact-integer correctness (no judge), Stability from exact-integer distinct
counts (no judge). Run once.

## Item set (balanced, pre-committed — same as the confab-specificity run)

- **CONFAB group** = realized one-shot confabs from the 36 hard `SPECS` (greedy `v1 != correct`,
  `v1 is not None`).
- **CORRECT group** = the 24 EASY items (`EASY_SPECS`, greedy `v1 == correct`).

Group membership uses the GREEDY one-shot label (as before). The detector signals below are all
independent of that greedy pass (resamples are drawn at temperature; entropy/margin are confidence
proxies, not the correctness label) — so detection is not circular.

## Signals (label confab = 1, correct = 0; every score oriented so HIGHER = more-likely-confab)

1. **Resampling instability (black-box, the grounding-class signal):** draw N=10 answers at
   temperature 1.0; parse integers; `Stability = 1 − (n_distinct − 1)/(N − 1)` (1.0 if all agree,
   →0 if all differ — the validated metric); detector score = `instability = 1 − Stability`.
2. **Single-pass clean entropy (white-box):** Shannon entropy of the clean (γ=1, no-intervention)
   next-token distribution at the first answer-token commitment position. Higher = less confident.
3. **Single-pass logit margin (white-box):** `top1 − top2` clean logit gap at the first answer
   token; detector score = `−margin` (smaller gap = less confident = higher score).

## Predictions / bars

- **B1 — resampling Stability separates confab from correct (core):** AUC(instability; confab vs
  correct) `≥ 0.70`.
- **B2 / B3 — single-pass confidence (reported):** AUC(clean entropy) and AUC(−margin). The
  "confident confabulation" hypothesis predicts these are NEAR CHANCE (≤ ~0.65); a high value
  would instead show single-pass detection is feasible.
- **B_contrast — the locus claim (core):** `AUC(instability) − max(AUC(entropy), AUC(−margin))
  ≥ 0.20`. Resampling adds discrimination that single-pass internal state does not. Because all
  three signals are read on the SAME items, the difficulty/length difference between groups is
  held FIXED across detector types — so this contrast isolates *resampling vs single-pass*, robust
  to the confound that affects absolute AUCs.

**SURVIVED iff B1 ∧ B_contrast.** Powering: `≥ 12` usable items per group. A null on B_contrast
(single-pass confidence separates them about as well as resampling) would mean confabulation IS
internally legible in one pass — refuting "confident confabulation" on this model and re-opening
white-box single-pass detection. Reported either way.

## Honest scope (pre-committed)

Single open model Qwen2.5-1.5B-Instruct; arithmetic only; one confirmatory run; feasibility-grade
(≤36 confab + 24 correct); resampling N=10 at T=1.0 (the validated grounding setting), Stability
from exact distinct-integer counts; single-pass entropy/margin from the clean logit-lens at the
first answer token; ground truth computed in-code then hashed pre-scoring; exact-integer
correctness (no judge). The CORRECT group is easy and the CONFAB group is hard — the difficulty
confound is real and is the very thing self-consistency exploits (Stability ≈ self-consistency ≈
tracks derivation difficulty; this is "self-consistency not truth," consistent with the keystone),
so B1 is reported as a self-consistency-detects-difficulty-driven-wrongness claim, NOT a
truth-oracle claim. B_contrast holds the confound fixed across detector types and is the
load-bearing result. This does NOT touch the correctness bound — resampling DETECTS confabulation,
it does not by itself CORRECT it (only method-diverse re-derivation can), and the detector flags
abstain, never the answer.
