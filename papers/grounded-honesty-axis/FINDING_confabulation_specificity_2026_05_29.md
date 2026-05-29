# FINDING — the late-band install is NOT confabulation-specific: band-knockdown perturbs CORRECT answers as much as (in fact more than) confabulations, so "knock the band, watch the entropy" is NOT a usable abstention detector on this model (powered NULL, REPORT_AS_LANDED)

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_confabulation_specificity_2026_05_29.md` BEFORE any code, with a pilot-driven validity
amendment logged BEFORE the confirmatory run (γ=0 span-average saturated both groups → switched
to first-token dose-integral over the pre-registered GAMMAS grid). Qwen2.5-1.5B-Instruct,
target band [22,26] FIXED from the disinhibition/suppression-rhythm findings (not re-tuned),
SAE-free full-vocab logit-lens, in-code arithmetic ground truth SHA-256'd pre-scoring
(`0eb5c90d…752b1`), exact-integer correctness (no judge), greedy/deterministic.** Receipt:
`confabulation_specificity_result.json`.

## Why this run exists

The disinhibition arc established that knocking the late band at a confabulation's commitment
point REMOVES the wrong answer and raises entropy — *uncertainty, not truth*
(`FINDING_disinhibition_2026_05_29.md`) — and that the install is late, tight, graded, and
distributed across the band on two architectures (`FINDING_single_layer_localization_2026_05_29.md`).
Every prior test ran ONLY on confabulated items. The obvious, load-bearing next question — and the
one that would have turned the mechanism into a patent-relevant tool — is: **is the install
confabulation-SPECIFIC, or is the late band just how the model emits ANY answer?** If knockdown
dissolves *confabs* into uncertainty while leaving *correct* answers standing, "knock the band,
watch the entropy" is an abstention detector. If it dissolves correct answers too, the band is
generic answer-emission machinery and there is no confab-specific signal.

**Method.** Balanced item set: CONFAB = realized one-shot confabs from the 36 hard `SPECS`;
CORRECT = 24 pre-committed EASY items (24/24 correct on Qwen in a no-intervention probe). For each
usable item (γ=1 baseline reconstructs the first realized answer token), at the first answer-token
commitment position, sweep the band knockdown over γ∈{0.75,0.5,0.25,0.0} and dose-integrate:
`entropy_rise` = mean ΔH vs baseline; `dissolution` = fraction of γ-levels at which the argmax
flips off the realized token. Bars: S1 entropy-rise AUC(confab>correct) ≥0.70; S2 dissolution gap
≥0.30 ∧ MWU p<0.05; S3 length-matched AUC ≥0.65. SURVIVED iff S1∧S2.

## Result: powered NULL — REPORT_AS_LANDED

| | confab | correct | bar | held |
| --- | --- | --- | --- | --- |
| usable items | 28 | 24 | ≥12 each (powered) | — |
| entropy_rise (mean ± sd) | **3.33 ± 0.48** | **5.43 ± 0.33** | — | — |
| **S1** AUC(confab > correct) | **0.0015** | | ≥0.70 | **FAIL** |
| dissolution (mean) | 0.438 | 0.458 | — | — |
| **S2** dissolution gap / MWU p | **−0.021** / **0.728** | | ≥0.30, p<0.05 | **FAIL** |
| **S3** length-matched AUC | **n/a (zero length overlap)** | | ≥0.65 | **FAIL** |

Not merely "no signal" — the discrimination is **near-perfectly reversed**: confabs show a
*smaller* entropy rise under band-knockdown (3.33) than correct answers (5.43), with tight,
**non-overlapping** distributions (AUC 0.0015 ≈ 0). Dissolution is statistically indistinguishable
(confab 0.438 vs correct 0.458, p=0.73). The abstention-detector premise is falsified on this
model.

## The three claims that land

1. **The late-band install is NOT confabulation-specific.** Knocking band [22,26] perturbs
   *correct* answers at least as much as confabs (more, on the entropy readout). The band is
   generic answer-commitment machinery, not a confab-only install. This directly bounds the
   disinhibition finding: "knock band → confab → uncertainty" is TRUE but **not specific** — the
   same knockdown also dissolves correct answers, so it cannot tell confab from correct.

2. **"Knock the band, watch the entropy" is NOT a usable abstention detector** (on Qwen,
   arithmetic, this readout). S1 fails by the largest possible margin in the wrong direction; S2 is
   a clean null. Any tool built on band-knockdown entropy to flag confabulation would flag correct
   answers harder. This kills the overclaim before it could be bet on — the disciplined value of
   the run.

3. **The difficulty/length confound is TOTAL and uncontrollable here** (S3 uncomputable). CONFAB
   answers are 4–7 tokens; CORRECT answers are 2–3 tokens; the sets share NO answer-length value,
   so the reversed direction cannot be cleanly attributed to confab-status vs answer length. The
   honest claim is therefore the conservative one: **no evidence of confab-specificity** — not a
   positive "correct answers are more fragile" claim. (A plausible read of the reversal: the
   leading digit of a short correct answer carries the whole magnitude decision and lives in the
   late band, whereas the leading digit of a long confab is more constrained by problem magnitude
   upstream — i.e. the effect tracks answer length, exactly what S3 was meant to isolate but
   couldn't.)

## Why the measurement was amended (pilot-driven, pre-confirmatory, validity not verdict)

The original γ=0 span-averaged readout SATURATED: full knockdown drove entropy to ~10.4 nats
(near-uniform over the ~150k vocab) and dissolution to ~1.0 for BOTH groups (AUC 0.500) — full
knockdown destroys every answer equally, leaving nothing to discriminate. This is the exact "too
blunt" mode the sister disinhibition prereg pre-named. The fix — chosen WITHOUT having seen the
confab-vs-correct contrast at any non-saturating γ — measures the first-token commitment using a
dose-integral over the already-pre-registered GAMMAS grid. It de-saturated cleanly (rises 2.6–6.0
nats, well-separated). The bars and SURVIVED rule were unchanged.

## Honest scope (pre-committed + observed)

Single open model Qwen2.5-1.5B-Instruct; SAE-free full-vocab logit-lens; single-position,
teacher-forced, dose-integrated band knockdown read at the first answer token (not multi-token
regeneration); one confirmatory run; feasibility-grade (28 usable confab + 24 correct); arithmetic
only; ground truth computed in-code then hashed pre-scoring; exact-integer correctness (no judge);
greedy/deterministic. Band [22,26] fixed from prior findings, not tuned. The CORRECT/CONFAB
difficulty+length confound is total (S3 uncomputable) — so the result is reported as **no evidence
of confabulation-specificity**, bounding the abstention-detector claim; it is NOT a positive claim
about correct-answer fragility. This does NOT touch the standing correctness bound — every internal
lever (band, single-layer, dose) still moves confidence/abstention, only re-derivation moves
correctness — and it makes no truth-recovery claim. The disinhibition and localization findings
stand; this run bounds their *interpretation*: the install is real, late, graded, and distributed,
but it is shared machinery, not a confabulation-only signature readable by band-knockdown entropy.

## The arc, in one line (updated)

The confabulation is a late, tight, graded, distributed install on two architectures whose
descriptive flip-layer sits within 3 layers of its causal peak — but that install is NOT
confab-specific: knocking it dissolves correct answers as much as confabs, so band-knockdown
entropy is not an abstention detector, and (as ever) every lever on the install moves confidence,
never truth; only re-derivation moves correctness.
