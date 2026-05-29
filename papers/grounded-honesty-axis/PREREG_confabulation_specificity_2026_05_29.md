# PRE-REGISTRATION — is the late-band install confabulation-SPECIFIC? (the abstention-detector test)

**Written 2026-05-29, BEFORE any code for this test is written.** The disinhibition arc
established that knocking the late decoder band ([22,26] on Qwen) at a confabulation's
divergence position REMOVES the wrong commitment and raises entropy — disinhibition yields
*uncertainty, not truth* (`FINDING_disinhibition_2026_05_29.md`), and that the install is late +
tight + graded + distributed across the band on two architectures
(`FINDING_single_layer_localization_2026_05_29.md`). Every test so far ran ONLY on confabulated
items. The open question this run closes:

**Is the late-band install confabulation-specific, or is it just how the model emits ANY answer?**
If knocking the band dissolves *confabulated* answers into uncertainty while leaving *correct*
answers standing, then "knock the band, watch the entropy" is a usable **abstention detector** —
directly patent-relevant. If knocking it dissolves correct answers too, the band is generic
answer-emission machinery and there is no confab-specific signal. Both outcomes are informative.

Model: **Qwen2.5-1.5B-Instruct** (28 decoder layers — the model where the band is fixed and
validated). Target band **[22,26]** is FIXED from the suppression-rhythm / disinhibition findings,
NOT re-tuned. SAE-free full-vocab logit-lens, in-code arithmetic ground truth SHA-256'd
pre-scoring, exact-integer correctness (no judge), greedy/deterministic. Run once.

## Item set (balanced, pre-committed)

The competence-cliff `SPECS` is tuned for a HIGH confab rate (only ~4/36 correct on Qwen) — it
cannot populate a powered "correct" group. So the balanced set is:

- **CONFAB group** = the existing 36 `SPECS` (hard 3×3 / 4×3 / multi-step). One-shot items where
  `v1 != correct` and `v1 is not None` (a realized confabulation).
- **CORRECT group** = 24 pre-committed EASY items (2-digit × 1–2-digit; e.g. 7×8, 12×12, 23×3),
  defined in the harness as `EASY_SPECS`. A correctness-only probe (run before this prereg, no
  intervention) returned 24/24 correct on Qwen — so the group is powered. One-shot items where
  `v1 == correct`.

## Measurement (works identically for both groups)

For each usable item, teacher-force `prompt + realized_answer`. The realized answer span is the
answer tokens after the prompt. **Usable iff the greedy teacher-forced baseline reconstructs the
realized answer at every span position** (base argmax == realized token at each `pos_k`) — this
guarantees a clean, alignable baseline equally for both groups; else skip.

For each answer-span index `k` (absolute position `pos_k = plen − 1 + k`, whose logits predict
answer token `k`): knock the target band [22,26]'s residual write at `pos_k` (γ=0), read the
next-token logits at `pos_k`.

- **entropy_rise** (item) = mean over `k` of `entropy(knocked) − entropy(base)`.
- **dissolution** (item) = fraction of `k` where `argmax(knocked) != realized token k` (the
  answer token no longer survives the knockdown).

## Predictions / bars

- **S1 — entropy-rise detector (core):** AUC of per-item `entropy_rise` discriminating
  CONFAB (label 1) from CORRECT (label 0) `≥ 0.70`.
- **S2 — dissolution gap (core):** `mean dissolution(confab) − mean dissolution(correct) ≥ 0.30`
  AND a Mann–Whitney U test on the two dissolution distributions `p < 0.05`.
- **S3 — length-matched specificity (corroborator):** restrict to answer-token-length values
  present in BOTH groups, recompute the `entropy_rise` AUC on that matched subset; bar `≥ 0.65`.
  This guards the obvious difficulty/length confound (confab items are longer/harder).

**SURVIVED iff S1 ∧ S2.** S3 is a reported corroborator (a pass strengthens the specificity
claim against the length confound; a fail flags that the signal is partly length-driven).
Powering: `≥ 12` usable items per group (else report unpowered).

A null on S1/S2 (knockdown dissolves correct answers as much as confabs) means the late band is
GENERIC answer-emission machinery, not a confab-specific install — reported as such, and it would
bound the abstention-detector claim hard. A pass means band-knockdown entropy is a confab-specific
abstention signal on this model.

## Amendment (2026-05-29, pilot-driven, BEFORE the confirmatory run — motivated by validity)

A 6-item pilot showed the original measurement saturates: a γ=0 (full) band knockdown averaged
across **all** answer-span positions drives entropy to ~10.4 nats (near-uniform over the ~150k
vocab) and dissolution to ~1.0 for **both** groups, collapsing the contrast to AUC 0.500. Full
knockdown destroys every answer equally — there is no discrimination left to read. (This is the
exact "too blunt" mode the sister disinhibition prereg pre-named, where γ=0 wipes the readout.)

The validity fix — chosen WITHOUT having seen the confab-vs-correct contrast at any non-saturating
level — measures the **first answer token** (the commitment point the disinhibition finding
validated) using a **dose-integrated** statistic over the already-pre-registered GAMMAS grid
`{1.0, 0.75, 0.5, 0.25, 0.0}` (from `run_disinhibition`), rather than a single saturating γ:

- **entropy_rise** (item) = mean over γ<1 of `entropy(knocked at γ) − entropy(base)` at the
  first answer token. (Dose-integrated: captures the whole fragility curve, robust to where any
  per-item flip threshold sits.)
- **dissolution** (item) = fraction of γ<1 levels at which `argmax(knocked) != first realized
  token` (how easily, across the dose range, the commitment is removed).

Usable iff the γ=1 baseline reconstructs the first realized token (clean baseline), equally for
both groups. `answer_len` (number of realized answer tokens) is retained solely for the S3
length-match. Bars S1/S2/S3 and the SURVIVED rule are UNCHANGED; only the per-item entropy_rise
and dissolution definitions move from "single γ, span-averaged" to "dose-integrated, first token".

## Honest scope (pre-committed)

Single open model Qwen2.5-1.5B-Instruct; SAE-free full-vocab logit-lens; single-position,
teacher-forced, γ=0 band knockdown read across the answer span (not multi-token regeneration);
one confirmatory run; feasibility-grade (36 confab candidates + 24 correct); arithmetic only;
ground truth computed in-code then hashed pre-scoring; exact-integer correctness (no judge);
greedy/deterministic. The band [22,26] is fixed from prior findings, not tuned to this verdict.
The CORRECT group is easy and the CONFAB group is hard — the difficulty confound is real and is
addressed by S3 (length-matched) plus this caveat, NOT eliminated. This tests whether the install
is confab-SPECIFIC (the abstention-detector premise); it does NOT touch the standing correctness
bound — knockdown still yields uncertainty, not truth — and the detector flags *low confidence /
abstain*, never *the correct answer*.
