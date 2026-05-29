# PREREG — Disinhibition: is the late hop (layers ≈23–27) CAUSALLY what installs the confident wrong answer — and does dampening it yield UNCERTAINTY, not truth?

**Pre-registered 2026-05-29, BEFORE any code for this test is written or run. One
confirmatory run. Feasibility-grade: single open model (Qwen2.5-1.5B-Instruct), SAE-free
full-vocab logit-lens read, the SAME n=36 arithmetic items as the spectral/suppression/
repair runs. Arithmetic ground truth computed in-code (`run_competence_cliff` SPECS) and
SHA-256'd before any scoring (expected to match the standing key `ddccd8e4…b87964d`).
Correctness = exact integer match (no judge). Greedy/deterministic.** Receipt:
`disinhibition_result.json`.

## Why this run (the causal counterpart of the corrected mechanism)

The suppression-rhythm control (`FINDING_suppression_rhythm_2026_05_29.md`) corrected our own
headline: confabulation is **not** the suppression of a computed truth — the correct token is
not privileged mid-network (Δ=−0.008 vs matched distractors). What *is* real is the **shape**
of the overwrite: a **tight, late, near-rhythmic hop at layers ≈23–27** (median flip layer 25
of 28, IQR 4) that **installs** the confident wrong answer over an undifferentiated mid-network
field. That was a *descriptive* logit-lens claim. **This run is its causal test.**

Two questions, both decisive:

1. **Is that late band causally responsible for the wrong commitment?** If we dampen the
   layers' residual *writes* at the answer position across the measured band and the wrong
   answer stops winning — significantly more than dampening a matched early band — the hop is
   causal, not epiphenomenal.
2. **When the wrong commitment is removed, what is underneath — truth or noise?** The corrected
   mechanism predicts an **undifferentiated field**: removing the install should expose
   *uncertainty* (a flatter distribution), **not** a waiting correct answer. If instead truth
   springs up, that would causally **re-open** the suppression reading we just retired. So this
   test gives the falsified F3 a fair second chance on causal evidence.

This is distinct from the prior linear depth-steering run (`FINDING_depth_steering_causal`):
that *injected* a construction−retrieval direction (writable, correctness-inert). This run does
not inject anything — it **attenuates the band's own residual write at the answer position**
(γ·contribution, γ∈[0,1]), a surgical knockdown of the install, with a matched-band control.

## Apparatus (committed before data)

- **Model:** Qwen2.5-1.5B-Instruct, local, float16, greedy; reproduces the answer-key hash.
- **Confab set:** items where the bare one-shot answer is wrong (parseable, not None) and
  **alignable** (realized and correct answer-token sequences share a first divergence point),
  exactly as in `run_suppression_rhythm` / `run_repair_geometry`. Let `pos` = the token
  position whose next-token prediction is the first divergent answer token; `r` = the realized
  (wrong) token id there, `c` = the correct token id there.
- **Intervention (no answer-key used to build it):** teacher-force `prompt + realized_answer`;
  register forward hooks on a band of decoder layers that, **at position `pos` only**, replace
  the layer output `h_out` with `h_in + γ·(h_out − h_in)` (γ=0 = full knockdown of that layer's
  write at `pos`; γ=1 = unchanged). Read the **next-token distribution at `pos`** (final
  `norm` + `lm_head`, full vocab) under the hook.
- **Target band (the measured install):** decoder layers **[22, 26]** (hidden-state indices
  ≈23–27 — the suppression-rhythm band, median 25, IQR 4). **NOT tuned** — taken from the prior
  finding.
- **Control band (matched size, early):** decoder layers **[6, 10]** — controls for "any
  ablation disrupts the answer."
- **Metrics at `pos`, full vocab:** *commitment* = (argmax == `r`); *commitment_removed* =
  (argmax != `r`); *truth_recovered* = (argmax == `c`); *entropy* = Shannon entropy of
  softmax(logits); *coherent* = argmax decodes to a numeric token. Baseline (no hook) must have
  argmax == `r`; items where it does not are recorded and excluded (teacher-forcing/float16
  mismatch).

## Predictions (decisive bars — pre-stated, scored exactly, both directions named)

| id | prediction | bar |
| --- | --- | --- |
| **I1 — the late band CAUSES the wrong commitment** | knocking down the target band removes the wrong commitment far more than the early control band | `f_target` = P(commitment_removed \| target γ=0); `f_ctrl` = same for control band. **Bar: f_target − f_ctrl ≥ 0.30 AND f_target ≥ 0.50 AND** exact-binomial/sign test on discordant pairs (target-only-removed vs ctrl-only-removed) **p < 0.05**. If f_target ≤ f_ctrl → late band NOT special (null, reported against prediction). |
| **I2 — disinhibition yields UNCERTAINTY, not truth** | among items where the target knockdown removed the commitment, truth does NOT take over, and the distribution flattens | restricted to commitment-removed items (≥6 for power): **truth_recovery_rate < 0.34 AND** paired entropy rise (target γ=0 vs baseline) **mean > 0 with p < 0.05**. **Reverse named:** truth_recovery_rate ≥ **0.50** → SUPPRESSION supported causally (re-opens F3), reported against prediction. (0.34, 0.50) → ambiguous. |
| **I3 — dose-response (corroborator)** | the commitment dissolves monotonically as the band write is dialed down | target-band γ ∈ {1.0, 0.75, 0.5, 0.25, 0.0}: Spearman ρ(γ, pooled commitment-rate) ≥ **+0.90** (lower γ ⇒ lower commitment). Corroborator only; does not gate RESULT. |

**RESULT = SURVIVED iff I1 ∧ I2 (installation branch).** I1 establishes the hop is causal for
the wrong commitment; I2 establishes it is *installation*, not suppression, at the causal level.
Otherwise REPORT_AS_LANDED with whatever held, scored against prediction.

## Precondition / honest failure modes (stated in advance)

1. **Power.** If fewer than **12** alignable confabs have a clean baseline (argmax == `r`),
   I1 is reported descriptively (no SURVIVED claim). For I2, if fewer than **6** items have
   their commitment removed by the target knockdown, I2 is under-powered → descriptive.
2. **Destructiveness confound.** Full knockdown of a 5-layer late band may garble the readout
   rather than specifically un-install the answer. Pre-committed floor: report the
   *coherence rate* (argmax still a numeric token) under target γ=0. **If coherence < 0.50,
   the γ=0 intervention is too blunt → I1/I2 are re-scored at γ=0.5** (still a knockdown,
   pre-named as the fallback) and the γ=0 numbers reported alongside as descriptive. The
   control band is the primary guard against "any ablation breaks it."
3. **Honest prior — this can null cleanly, and a null is still informative.** A null on I1
   (target ≈ control) would say the wrong commitment is **not** surgically localizable to the
   measured band by single-position write-knockdown — consistent with a distributed install —
   and would **not** refute the descriptive suppression-rhythm finding (a logit-lens
   observation), only its *causal localization*. A surprise on I2 (truth recovers) would
   causally resurrect suppression; we would report it loudly against our own corrected
   headline. Either way the methodology, not the hoped-for direction, is the point.

## Honest scope (pre-committed)

Single open model (Qwen2.5-1.5B-Instruct); SAE-free full-vocab logit-lens readout;
feasibility-grade n=36; one confirmatory run; arithmetic ground truth computed in-code then
hashed pre-scoring; exact-integer correctness, no judge; greedy/deterministic. The intervention
is a **single-position, teacher-forced next-token** knockdown of a band's residual write — it
tests the install **at the divergence position**, not downstream multi-token regeneration. The
target band is fixed from the prior finding (not re-tuned here). This is a causal *localization*
test within one model; it does not claim the band is the install on other models or for
non-arithmetic confabulations. A null leaves the descriptive suppression-rhythm finding and the
whole standing arc intact; it would refute only the causal *localizability* of the install to
this band by this surgical method.
