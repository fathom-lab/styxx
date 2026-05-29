# FINDING — the depth suppression-rhythm and the single-pass output-uncertainty signature are STATISTICALLY INDEPENDENT within confabs on Qwen arithmetic (within-confab Spearman ρ≈0 for the late-overwrite depth; predicted link FALSIFIED, REPORT_AS_LANDED) — so confabulation leaves at least TWO independent fingerprints, not one collapsible signal

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_rhythm_uncertainty_link_2026_05_29.md` BEFORE any code. Qwen2.5-1.5B-Instruct (the
white-box model — all signals read from the SAME network), the 36 hard `SPECS` confab set,
WITHIN-confab only (a `flip_layer` exists only where the correct token was suppressed), depth
features from the clean full-vocab logit-lens at the first divergent answer token, output
entropy/margin from the same lens at the first answer token, instability from N=10 resamples at
T=1.0 (validated grounding setting, exact distinct-integer Stability, no judge), arithmetic
ground truth SHA-256'd pre-scoring (`0eb5c90d…752b1` — same key as detection-locus).**
Receipt: `rhythm_uncertainty_link_result.json`.

## Why this run exists

Two lines of the arc landed independently and had never been connected:
- **Depth line** (`run_suppression_rhythm.py`): on confabs the correct token leads mid-network
  then is overwritten late in a tight band — the "suppression-rhythm" (`flip_layer` = last layer
  the correct token still leads).
- **Output line** (`FINDING_detection_locus_2026_05_29.md`): confabulation is internally legible
  in a single forward pass — clean entropy / logit margin separate confab from correct at AUC
  ~0.92, resampling instability at 0.98.

The unifying hypothesis (pre-registered): the depth-rhythm is the UPSTREAM correlate of the
output signature — a confab whose correct answer stayed competitive deeper (later `flip_layer`)
or was overwritten less sharply should look less confident on the surface (higher entropy /
instability). One number would then tie the internal-geometry line to the detection line. **That
hypothesis is FALSIFIED.**

## Result: REPORT_AS_LANDED — B_link FAILS, the two signals are independent

| pair (within-confab Spearman) | predicted | ρ | p | held |
| --- | --- | --- | --- | --- |
| **U1** rel_flip → clean entropy | + | **−0.068** | 0.72 | no |
| **U2** rel_flip → instability | + | **−0.052** | 0.78 | no |
| **U3** sharpness → clean entropy | − | **−0.255** | 0.17 | no |
| rel_flip → logit margin | (report) | +0.073 | 0.70 | — |
| sharpness → instability | (report) | −0.018 | 0.92 | — |
| sharpness → logit margin | (report) | +0.265 | 0.15 | — |

n_rhythm = **31** (powered; n_no_rhythm = 1). **B_link** (≥1 of U1/U2/U3 at |ρ|≥0.40, p<0.05, in
the predicted direction) = **FAIL**. The *depth* of the overwrite (`rel_flip`) carries
essentially zero information about output uncertainty (ρ≈0). The *sharpness* of the final
overwrite shows only a faint, non-significant hint in the predicted direction (sharper overwrite
→ slightly lower entropy, ρ=−0.26 p=0.17; and → slightly higher margin, ρ=+0.27 p=0.15) — below
the bar and not significant at this n.

## The claims that land

1. **The internal overwrite-geometry and the surface uncertainty signature are decoupled.**
   *Where* (and how sharply) the network commits to the wrong answer mid-computation does not
   predict *how unsure it sounds* at the output. On this corpus they are statistically
   independent windows onto the same confabulation.
2. **Confabulation therefore leaves at least TWO independent fingerprints, not one.** A
   buried-geometry tell (the late, tight truth-overwrite) and a surface-confidence tell
   (entropy/margin) co-occur on confabs but do not co-vary. For DETECTION this is a feature, not
   a bug: redundant signals from independent sources are more robust than one collapsible signal
   — but it also means you cannot read the internal mechanism off the output confidence, or vice
   versa.
3. **The arc does not collapse to a single number.** The tempting unification ("the rhythm IS
   the detectable signal") is refuted here. The depth line and the detection line remain
   genuinely separate contributions; pre-registering the link and reporting its null is what
   keeps the arc honest.

## Honest scope + the confound (pre-committed)

Single open model Qwen2.5-1.5B-Instruct; arithmetic only; one confirmatory run;
feasibility-grade (31 confabs with a defined rhythm). WITHIN-confab only — `flip_layer` is
undefined for correct answers, so this is NOT a confab-vs-correct claim and does not re-test
detection (already landed). Depth features from the clean full-vocab logit-lens at the first
divergent answer token; output entropy/margin from the same lens at the first answer token;
instability from N=10 resamples at T=1.0 (exact distinct-integer Stability, no judge); ground
truth in-code then hashed pre-scoring. A null is the weaker inferential position than a positive
ρ: with n=31 the run is powered to detect |ρ|≥0.40 but not a true small effect (|ρ|~0.2), so the
honest statement is "no link at the pre-registered effect size," not "provably zero." Correlation
not cause throughout. Does NOT touch the correctness bound: both features are confidence/
uncertainty readouts; neither corrects the answer (only method-diverse re-derivation does).

## The arc, in one line (updated)

The confabulation is a late, tight, graded install of SHARED answer-commitment machinery
(band-knockdown can't flag it); on Qwen arithmetic the wrong commitment carries its own
single-pass uncertainty signature (entropy/margin ≈ ten resamples, AUC ~0.92); and that surface
signature is INDEPENDENT of the internal overwrite-depth geometry (within-confab ρ≈0) — so the
model leaves multiple independent fingerprints of its own confabulation rather than one unifiable
signal, and, as ever, every signal here moves confidence/abstention, never correctness.
