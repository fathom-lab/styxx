# FINDING — Detect-and-abstain SURVIVED, and the detector is LOAD-BEARING: mechanistic abstention has NO intrinsic selectivity, so the single-pass detector is the necessary gate that turns a global commit-knockdown into a targeted honesty intervention

**Run 2026-05-30. One confirmatory run, pre-registered in `PREREG_honesty_knob_2026_05_30.md`
BEFORE the scoring run.** Answer-key SHA-256 printed pre-scoring: `0eb5c90d…55752b1`. Powered
(n=32 confab / 24 correct usable, both ≥ 12). A 10-item pilot validated the apparatus only and
scored no bar.

## Why this experiment exists

Correctness **repair** is a closed negative in this project — depth-steering is *correctness-INERT*
(`FINDING_depth_steering_causal`) and removing the install yields *UNCERTAINTY, not truth*
(`FINDING_disinhibition`). So this does **not** chase repair. It chases the one move the findings
*support*: convert a confident confabulation into honest **abstention**, and ask whether that can be
made **selective**. `FINDING_disinhibition` knocked down decoder band **[22,26]** and dissolved the
wrong commitment on 0.889 of confabs — **but it only ever tested confabs.** The open,
deployability-critical question: *is the knockdown selective (does it spare correct answers), and if
not, is the confabulation detector what makes targeted abstention possible?*

## Apparatus (fixed from prior validated findings, nothing re-tuned)

Qwen2.5-1.5B-Instruct. Confab = `SPECS` (hard arithmetic, greedy-WRONG); correct = `EASY_SPECS`
(easy arithmetic, greedy-RIGHT); membership from the model's own greedy answer. Intervention: at the
first answer-token position, attenuate band **[22,26]** residual *write* by **γ=0** (the exact
`logits_at` mechanism, the exact band, fixed from `FINDING_disinhibition`). `commitment_removed` =
post-intervention argmax ≠ baseline committed token. Detector/gate signal: baseline single-pass
first-answer-token entropy (the `FINDING_detection_locus` signal), no hook.

## Result

| Bar | Statement | Threshold | Outcome |
|---|---|---|---|
| **B1** | intervention induces abstention on confabs (replicate) | confab removal ≥ 0.50 | **HELD. 0.750** (24/32) |
| **B2** *(descriptive)* | is the **raw** intervention intrinsically selective? | (confab − correct) removal ≥ 0.30 | **FAILED, as pre-stated. −0.083** |
| **B3** | detector separates confab/correct → can **gate** the intervention | gate AUC ≥ 0.70 | **HELD. 0.924** |

**RESULT = SURVIVED** (required B1 ∧ B3). The detector-gated loop **catches-and-abstains 0.75 of
confabs while false-abstaining only 0.125 of correct answers** (Youden threshold, in-sample).

## What it means — the detector is load-bearing

The raw mechanistic knockdown has **no intrinsic selectivity**: correct commitments dissolve *as
readily as* confab ones (removal 0.833 vs 0.750; selectivity **−0.083**, slightly inverted), and the
entropy blast is essentially identical for both classes (**+10.58 nats** confab vs **+11.43 nats**
correct). Band [22,26] is a **general confidence-commit mechanism** — the model's "commit to the
argmax" hop for *any* answer, right or wrong. Knocking it down always-on would make the model abstain
on everything: a lobotomy, not an honesty intervention.

The single-pass **detector** is what rescues it. It separates confab from correct at **AUC 0.924**,
so gating the knockdown by the detector fires the intervention on (mostly) confabs and spares
(mostly) correct answers — converting a global, indiscriminate knob into a **targeted detect-and-
abstain** primitive. The non-obvious claim, pre-registered and confirmed:

> **Detection is not optional diagnosis you could skip before intervening — it is the prerequisite
> that makes intervention safe. The detector is load-bearing.** Repair-via-steering is correctness-
> inert; abstention-via-knockdown is *correctness-blind* (it abstains good answers too); only the
> detector-gated loop is both useful and safe.

The pre-stated prediction that **B2 would fail held exactly** — a rigor signal, not a salvage: the
honest mechanistic content (the band is global, the detector carries the selectivity) was named
before the confirmatory run, not after.

## The deliverable: a closed-loop honesty primitive

This closes the loop using **only the project's own validated mechanisms** — the
`FINDING_detection_locus` single-pass detector + the `FINDING_disinhibition` band intervention — into
a **fail-safe**: *abstain instead of confabulate.* It does **not** make the model right (that's the
closed negative); it makes the model **honestly uncertain** on exactly the answers it would have
confabulated.

The mechanistic (white-box) result grounds a **policy-level loop that is already deployable today**
via shipped styxx primitives: `single_pass_confab` / `span_confab` (the detector) gate an abstention
action ("I'm not sure" / route to retrieval) — the same load-bearing-detector logic, no hooks
required. The white-box experiment is the proof that the gate is *necessary*, not merely convenient:
you cannot safely intervene ungated.

## Scope / caveats (stated before the run)

Single open model, arithmetic confab/correct, one confirmatory run, **feasibility-grade**, **white-
box** (decoder-layer hooks). Band [22,26] γ=0 fixed from `FINDING_disinhibition`, not re-tuned. The
intervention converts commitment to **uncertainty/abstention**, **not** truth — it does not touch the
correctness bound. Gated rates use an in-sample Youden threshold (the threshold-free AUC is the bar).
No cross-model / cross-domain claim. The contribution is the **loop and its load-bearing detector**,
established on one model as a feasibility proof.
