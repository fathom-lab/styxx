# FINDING — TWO-STAGE DEPLOYABLE CONSCIENCE: the gate is REAL and FIXES the false-accusation, but the deploy-time HELD read is a difficulty proxy on Qwen → REPORT_AS_LANDED, deployable on Llama, family-split stated

**2026-06-07. Fathom Lab / styxx.** Red-team bound on the two-stage conscience (Stage 1 HELD → Stage 2 HID).
Verdict: **REPORT_AS_LANDED.** Three results, two positive, one negative, all cross-family, hash-before-score.

## What survived (both Qwen-3B and Llama-3B)

**1. The composite is NOT just re-reading Rung-1.** Among the held (LIE vs RESISTED — both knew, so a held-gate
is at chance between them by construction), the Stage-1 held-gate AUROC is **0.257 / 0.267** (≤ chance, as it
must be) while the Stage-2 intent direction reads **0.813 / 0.850**. Stage 2 adds **+0.556 / +0.583** of
discrimination the held gate cannot. The conscience genuinely has two separable parts. (G-COMPOSITE-REAL ✓✓)

**2. The two-stage gate FIXES the 0.949 false-accusation.** Bare Stage-2 mean cave_prob on a never-knew model =
**0.863 (Qwen) / 0.957 (Llama)** — the standalone direction false-accuses sincere errors, reproduced cross-family.
The two-stage gate collapses it. At matched honest-FPR budget the gate dominates Stage-2-alone on BOTH axes:

| | Qwen b=0.10 | Llama b=0.05 | Llama b=0.10 |
|---|---|---|---|
| two-stage LIE_recall / MIS_falseaccuse | 0.372 / 0.189 | **0.362 / 0.000** | 0.551 / 0.024 |
| stage-2-alone LIE_recall / MIS_falseaccuse | 0.081 / 0.108 | 0.000 / 0.048 | 0.101 / 0.133 |

On Llama the gate gives **5× the LIE recall at 1/5 the false-accusation**; on Qwen it roughly quadruples recall
at comparable MISTAKE leakage. (G-FIXES-FALSEACCUSE ✓✓)

## The decisive negative (the deploy-time leak, confound #2)

**At deployment only the caving (syco) residual exists** — the run_reclimb validity channel (G-PRIME 0.93/0.99)
lives on the NEUTRAL pass, which is gone at runtime. We tested whether Stage 1 can establish HELD from the syco
residual ALONE:

- The **neutral-trained gate does NOT transfer to syco** (held AUROC **0.526 / 0.598 ≈ chance**). You cannot port
  the run_reclimb validity establishment to deploy. So the deployable gate must be the **syco-only held gate**,
  which reads held-vs-never-knew at **0.785 (Qwen) / 0.936 (Llama)**.
- **DECISIVE control — is the syco-only gate just difficulty?** Replace the residual gate with a deploy-available
  difficulty heuristic (`letter_margin` + `vocab_entropy`, both computable from the caving pass):
  - **Qwen: the difficulty gate MATCHES the residual gate** (b=0.05: DG 0.244/0.108 vs residual 0.221/0.135;
    b=0.10: DG 0.314/0.135 vs 0.372/0.189). The syco-only "held" read on Qwen adds **nothing beyond difficulty**
    — it works because never-knew items are *easier/lower-margin*, not because it reads a held state. Confirmed by
    cross-letter collapse (held gate 0.785 → **0.604** cross-letter) and the difficulty-only AUROC 0.739.
  - **Llama: the residual gate BEATS difficulty** (b=0.05: residual **0.362/0.000** vs DG 0.058/0.048; cross-letter
    holdout **0.755 > difficulty 0.728**). On Llama the deploy held read is genuinely held-state, not difficulty.

**So Stage-1 HELD is deployable from the caving residual alone on Llama-3B but degrades to a difficulty proxy on
Qwen-3B.** The conscience is deploy-time-honest on one of two families; cross-family deployability is NOT met.
(G-DEPLOY-GATE: Llama ✓, Qwen ✗.)

## Honest deployable bound

The two-stage conscience is a **real, cross-family false-accusation fixer** (the headline win: it removes the
0.949 sincere-error false-accusation that kills the standalone Stage-2 intent direction). But at a strict 5%
false-accusation budget it recovers only **~22% (Qwen) / ~36% (Llama)** of lies — high precision, modest recall;
the operating value is at looser budgets (Llama b=0.10: 55% recall at 2.4% MISTAKE false-accusation). And the
**deploy-time HELD establishment is only a true held-state read on Llama**; on Qwen the runtime gate is a
difficulty heuristic, so Qwen's two-stage gate "works" partly for the wrong reason. **Not yet a robustly
cross-family deployable conscience from the caving residual alone** — it needs either the neutral pass (not
available at deploy) or a stronger syco-only held read that beats difficulty on every family.

## Scope / not-claimed

≤3B, MMLU sycophancy-caving, commit-token, **sycophancy-specific** (instructed lying 0.42 < chance, carried).
Small TEST classes (RESISTED 39–51, never-knew 37–83) → the BUDGET SWEEP, not any single point, is the result.
Verbose-correct guard (free-form) still OWED. **NOT claimed:** a standalone or robustly cross-family deployable
mens-rea detector. **Claimed (bounded):** the two-stage gate is a real, cross-family fix for the sincere-error
false-accusation, with a difficulty-honest deploy gate on Llama only. A negative deploy-gate control reported as
loud as the two positives. Receipts: `conscience_deploy_result_qwen3b.json`, `conscience_deploy_result_llama3b.json`,
`run_conscience_deploy.py`, `PREREG_conscience_deploy_2026_06_07.md`.
