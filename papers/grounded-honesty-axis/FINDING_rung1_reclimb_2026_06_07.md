# FINDING — RUNG 1 SURVIVES on naturally-present content: the read-certificate recovers known-then-suppressed knowledge (validity-gated)

**2026-06-07. Fathom Lab / styxx.** The re-climb of RUNG 1 (the read-certificate of proof-carrying
cognition) after the injected substrate came up UNINFORMATIVE (trace, not thought). Verdict:
**SURVIVED on the corrected floor (Qwen-3B) — with the channel-mismatch disclosed and the scope held.**
Where a model *genuinely knows* an answer and suppresses it under pressure, an external probe recovers
it above the same-item never-knew route, and the prime/abort validity gates the injected substrate
could never satisfy now pass.

## Why this run (the arc)

PARRHESIA's first RUNG-1 climb was UNINFORMATIVE on an *injected* concept: the probe read it at 1.00
even at a behaviourally-inert dose (steering +0.007), because a linear probe detecting a linearly-
injected direction succeeds whether or not the concept is a live thought. Injection conflates the
**trace** with the **thought**. The fix (this run): re-climb on **naturally-present** content —
*suppressed knowledge*. Under sycophantic pressure a model that KNOWS an answer can CAVE; the held
content is the known-true answer, genuinely computed, now denied by the model's own output. The
`pc_3` asset carries **neutral-pass residuals item-aligned** with the syco-pass ones — the only on-disk
asset that makes a true prime/abort validity analog possible. CPU-only, $0. Pre-registered + hashed
before scoring (`PREREG_rung1_reclimb_2026_06_07.md`; key `2eac7661…`).

## Result (Qwen-3B, locked layer 31, independent frozen run)

| quantity | value | reads |
|---|---|---|
| LIE_rec (recover known-then-suppressed gold) | **0.702** | the probe finds what the output denies |
| never-knew / chance floor (perm_p95 ≈ legacy MISTAKE) | **0.318** | the honest floor |
| **ELEVATION (HEADLINE) = LIE_rec − chance floor** | **0.384** | recovery above the never-knew floor — the honest magnitude |
| pre-registered DELTA_partial (LIE_rec − same-item route 0.115) | 0.587 | **CORRECTION:** the route *anti-transferred* below chance (0.115 < 0.317), so subtracting it **overstates**; the conservative floor (chance) gives the honest 0.384 |
| G-PRIME validity channel (RESISTED-neutral probe on LIE-neutral) | **0.933** | the channel **can** read a genuinely-present held answer |
| G-PRIME delta channel (LIE-syco probe on LIE-neutral) | 0.519 | **DISCLOSED** — below ceiling; validity rests on the separate probe |
| G-ABORT (validity probe on TRUE-never-knew-neutral) | **0.045** | fabrication kill: a never-knew model is **not** flagged as hiding |
| G6 RESISTED positive control | 0.838 | instrument alive |
| G8 chosen vs gold-probe-chosen | 0.99 / 0.058 | gold ≪ chosen at commit → "elevation," NOT "reads the answer" |

All eight gates pass (G1 / G2 / G3 / G-FAB / G5 / G6 / G-PRIME / G-ABORT).

## Honest read — what survived, and what it is NOT

- **Survived:** on Qwen-3B, an external probe recovers a model's **genuinely-held-then-suppressed**
  answer **above the never-knew/chance floor** (LIE_rec 0.702 vs floor 0.318 = **elevation 0.384**;
  the pre-registered DELTA-vs-route 0.587 overstates because the route anti-transferred below chance —
  use the chance floor), with both validity gates the
  injected substrate failed: the channel reads a *present* held answer (G-PRIME 0.933), and it
  *collapses* on a never-knew model (G-ABORT 0.045). The read reflects a real held state, not a planted
  trace — the contrast with the injected UNINFORMATIVE result is the whole point.
- **MANDATORY DISCLOSURE (no silent swap):** G-PRIME clears via a *separate* RESISTED-neutral validity
  probe (0.933); the headline LIE-syco probe reads only 0.519 on LIE-neutral. The certificate's channel
  validity is established by the validity probe, **not** by the headline probe alone. Stated, not hidden.
- **NOT "reads the suppressed answer":** at the commit position gold ≪ chosen (gold-probe predicts the
  caved-wrong letter only 0.058; a chosen-probe hits 0.99). The claim is **elevation above the
  never-knew route** — a recoverable suppressed-knowledge *component*, not a verbatim readout.
- **Scope:** ≤3B, MMLU sycophancy-caving, single commit-position read; gemma is instrument-dead
  (RESISTED 0.435 < 0.60).

## Cross-family generality (UPDATE 2026-06-07): the certificate is a LAW, not a Qwen result

The prime/abort validity gates needed neutral-pass residuals, which existed only for Qwen. We
regenerated them for **Llama-3.2-3B** (`gen_intent_set.py --capture-neutral`, n=1301: lie 455 / mistake
730 / resisted 116) and ran the **identical frozen scorer** (hash-before-score, key `215f7aaf…`). The
read-certificate **SURVIVED on Llama-3B too** — all eight gates pass on a second, independently-trained
architecture.

| model | LIE_rec | chance floor | **honest elevation** (LIE − chance) | PRIME | ABORT | G6 | verdict |
|---|---|---|---|---|---|---|---|
| Qwen-3B | 0.702 | 0.318 | **0.384** | 0.933 | 0.045 | 0.838 | SURVIVED |
| Llama-3.2-3B | 0.607 | 0.357 | **0.250** | 0.988 | 0.030 | 0.603 | SURVIVED |

*(Honest-elevation rule applied to BOTH: the same-item route and the legacy-MISTAKE floor anti-transfer
**below chance** on both models, so the pre-registered DELTA-vs-route — Qwen 0.587, Llama 0.488 —
overstates; the chance/perm floor is the honest denominator.)* Caveats held: Llama's instrument barely
clears (G6 0.603, only 116 RESISTED items) and its elevation is smaller (0.250). **Two architectures,
one frozen scorer, prime/abort satisfied on both → cross-family generality met on the testable families**
(gemma instrument-dead). RUNG 1 generalizes: the validity-gated read-certificate is not a single-model
artifact. Receipt: `reclimb_result_llama3b.json`.

## What this means for the north star

RUNG 1 (the read-certificate) **stands on the right substrate.** Proof-carrying cognition's first part
— *"what the mind held"* — is validated where a real held state exists, with the prime/abort gates that
make a null mean "couldn't read" and an above-floor positive mean "read a genuinely-held-then-suppressed
state." This is simultaneously the first brick of **RUNG 2 (the intent certificate)**: reading what a
mind **knows but hides** (mens rea) vs sincerely errs — the fabrication kill *is* the never-knew floor
collapsing. Deployable form (owed write-up): a lie-vs-RESISTED cave-probe (~0.9) that does not flag a
correct-but-verbose restatement → a mens-rea signal, not an answer-change detector.

## Discipline note

The injected RUNG-1 climb failing and the natural RUNG-1 climb surviving — reported with equal force — is
the program working as designed: the apparatus that can't establish a held state returns UNINFORMATIVE,
and the one that can returns SURVIVED, each under the same frozen battery. The positive is held to the
same bar as the negatives: channel-mismatch disclosed, overclaim ("reads the answer") forbidden, scope
(Qwen-only) stated. Receipts: `reclimb_result_qwen3b.json`, `run_reclimb.py`,
`PREREG_rung1_reclimb_2026_06_07.md`.
