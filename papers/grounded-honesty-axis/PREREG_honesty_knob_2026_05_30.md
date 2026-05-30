# PREREG — the closed-loop HONESTY KNOB: is mechanistic abstention SELECTIVE, and does the single-pass detector have to gate it?

**Registered 2026-05-30, BEFORE the confirmatory scoring run.** Bars and the answer-key hashing
protocol are fixed here. A 10-item feasibility pilot (n=4 confab / 6 correct, underpowered, NOT the
confirmatory) was run first only to validate the apparatus end-to-end; it does not score any bar.

## The question

Correctness **repair** is a closed negative in this project: depth-steering is *correctness-INERT*
(`FINDING_depth_steering_causal`), and removing the install yields *UNCERTAINTY, not truth*
(`FINDING_disinhibition`). So we do **not** chase repair. We chase the one move the findings
*support*: convert a confident confabulation into honest **abstention** (uncertainty), and ask
whether that conversion can be made **selective** — confabs dissolve, correct answers survive.

`FINDING_disinhibition` (SURVIVED) showed that knocking down decoder band **[22,26]** at the answer
position removes the wrong commitment on 0.889 of confabs and raises entropy +7.86 nats. **It only
ever tested confabs.** The open, untested, deployability-critical question:

> Is the [22,26] knockdown **selective** — does it preserve CORRECT commitments while dissolving
> confab ones? If not, is the single-pass confabulation **detector** what makes targeted abstention
> possible (i.e. is detection *load-bearing for intervention*, not just diagnostic)?

## Apparatus (all fixed from prior validated findings, nothing re-tuned)

- **Model:** Qwen/Qwen2.5-1.5B-Instruct (single open white-box model; needs decoder-layer hooks).
- **Items:** confab = `SPECS` (hard arithmetic, greedy-WRONG); correct = `EASY_SPECS` (easy
  arithmetic, greedy-RIGHT). Membership decided per item from the model's own greedy answer.
- **Intervention:** at the first answer-token position, attenuate the band [22,26] residual *write*
  by γ=0 (full knockdown): `h_out → h_in + γ·(h_out − h_in)` — the **exact** `logits_at` mechanism
  and the **exact band fixed** from `FINDING_disinhibition`. Not re-tuned.
- **commitment_removed** = the post-intervention argmax at that position differs from the baseline
  committed (argmax) token.
- **Detector / gate signal:** baseline single-pass first-answer-token entropy (the
  `FINDING_detection_locus` signal), computed with NO hook.

## Bars (fixed)

| Bar | Statement | Threshold |
|---|---|---|
| **B1** | The intervention induces abstention on confabs (replicate disinhibition). | confab commitment-removal rate **≥ 0.50** |
| **B2** *(descriptive)* | Is the **raw** intervention *intrinsically* selective? | (confab_removal − correct_removal) **≥ 0.30** |
| **B3** | The single-pass detector separates confab/correct, so it **can gate** the intervention. | gate AUC(entropy; confab=1) **≥ 0.70** |

**RESULT = SURVIVED iff B1 ∧ B3** — i.e. a working **detect-and-abstain** primitive: the intervention
reliably abstains confabs (B1) and the detector reliably flags exactly those items to gate it (B3).

**Pre-stated expectation (honest, on the pilot signal):** **B2 will FAIL.** The pilot showed correct
commitment-removal 0.83 ≈ confab 1.0 — the late band looks like a *general* confidence-commit
mechanism, so always-on intervention would abstain on everything. If B2 fails, the headline is the
**non-obvious mechanistic claim** that motivates the whole loop: *the detector is load-bearing — it
is the necessary gate that turns a global lobotomizing knob into a targeted honesty intervention.*
If B2 unexpectedly HOLDS, that is a stronger (standalone-selective) result, reported as such.

## Power & integrity

- **Powered** iff n_confab ≥ 12 AND n_correct ≥ 12 usable items. If under-powered, REPORT_AS_LANDED
  with the gap named; extend `--n` / item pool, never relax a bar.
- **Hash-before-score:** the (expr, correct-answer) key is SHA-256'd and printed before any scoring;
  recorded in the receipt as `answer_key_sha256_pre_scoring`.
- **One confirmatory run.** The Youden-threshold gated rates are reported as **in-sample** (the
  threshold-free **AUC is the bar**; gated rates are illustration, disclosed as in-sample).

## Scope (stated before the run)

Single open model, arithmetic confab/correct, one run, **feasibility-grade**, **white-box**
(decoder-layer hooks required — not an API primitive). The intervention does **NOT** correct to truth
(closed negative) — it converts confident commitment to **uncertainty/abstention**. The novelty is
the **loop**: detector-gated mechanistic abstention. Nothing here claims cross-model, cross-domain,
or correctness improvement. The deliverable, if SURVIVED, is a *fail-safe* primitive — "abstain
instead of confabulate" — for self-hosted/white-box models, gated by the validated single-pass
detector.
