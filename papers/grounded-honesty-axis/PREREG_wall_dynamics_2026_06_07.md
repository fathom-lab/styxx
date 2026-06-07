# PREREG — Cracking the Confident-Misconception Wall from the PROCESS/DYNAMICS axis

**Status:** FROZEN pre-registration. Hash this file (SHA-256 of feature code + fold seeds + this document) and print the hash BEFORE any correctness label touches the dynamics features. Score once. Report SURVIVED vs PARTIAL vs REPORT_AS_LANDED honestly.

**Date frozen:** 2026-06-07
**Primary model:** Qwen/Qwen2.5-1.5B-Instruct (L=28 hidden layers, 29 hidden states)
**Floor:** confident-consistent floor, n=121 (70 WRONG / 51 CORRECT), answer-key SHA-256 `4607df30...ce81f`
**Apparatus:** `loglens()` in `run_wall_read.py` extended from one layer to the full `hidden_states` tuple (`model.model.norm` + `model.lm_head` at every depth); floor builder, `gen_logits`, `score_truthful`, `styxx.span_confab`, `styxx.semantic_entropy` already exist. $0, local, 1x8GB GPU.

---

## 1. Background — what is already proven, and the one untested axis

The "confident-misconception wall" = items where the model's OUTPUT signals all say *fine* (`span_confab max_entropy <= median` AND `semantic_entropy(k=4) <= median`). On this n=121 floor, by prior work:
- output max-entropy AUC (correct-vs-wrong) = **0.53** (chance) — the wall.
- static single-mid-layer (L20) logit-lens log-prob of the teacher-forced GOLD answer, `R_gold` = **0.519** (chance) — the suppressed truth is **not represented**.
- K = R_gold − R_emit = 0.62 borderline, carried by **R_emit** (emitted-wrong answers are mid-layer WEAKER), NOT by recovered gold knowledge.

**Conclusion so far:** the wall is CONTENT-bedrock on two axes — output confidence (0.53) and static content readout (0.519). The ONE untested axis is **PROCESS / DYNAMICS**: *how* the committed answer forms across the layer stack.

**Hypothesis (retrieval vs construction):** a retrieved fact crystallizes EARLY, smoothly, low cross-layer churn, early entropy collapse (MLP key-value recall in early/mid stack). A confidently-WRONG belief has no stored association so it is ASSEMBLED LATE by attention composition — late crystallization, mid-stack competition/churn, late entropy collapse. This is orthogonal to the final-layer value the wall measures. If a LABEL-FREE + GOLD-FREE dynamical signature separates correct-vs-wrong on this floor where everything else is at chance, the wall is cracked from a new, deployable axis. If null, the wall is true bedrock even in dynamics — a valid, publishable hardening.

---

## 2. Design discipline (why ONE design, not three)

Three candidate designs + a red-team were synthesized. The red-team verdict: **SURVIVABLE BUT UNDER-CONTROLLED** — two unaddressed confounds (floor-selection circularity; embedding-proxy label bias) can fabricate a win, and this lab has already been burned by a length confound in this exact apparatus (`LENGTH_CONFOUND_FINDING_2026_05_09.md`: 64% of a deception detector's variance was `log_word_count`).

Decisions:
1. **Freeze ONE design** (the deployable label-free/gold-free feature set) — eliminates the cross-design "best of 3" family-wise inflation.
2. **Mid-stack-only** crystallization/churn — greedy decode is the argmax path; final layers are a tautological self-consistency win.
3. **Adopt EVERY killing-control** the red-team demanded (Section 6), including the two it flagged as fatal-if-missing: floor-selection partial and clean-label replication.
4. Primary = ONE LOO-CV composite, hash-before-score. Per-feature AUCs are exploratory only, Holm-noted, barred from becoming the headline.

---

## 3. FROZEN feature set (all LABEL-FREE and GOLD-FREE; emitted-answer trajectory only)

All features computed from a single teacher-forced pass over the model's OWN greedy-decoded answer, restricted to the **mid-stack window l ∈ [1, L−3]** (exclude embeddings idx0 and the final 3 layers; for Qwen2.5-1.5B that is layers 1..25) BEFORE normalization.

**F1 — `crystallization_depth` (THE BET).** Within-window min-max normalize the emit log-prob trajectory `T_emit[l]` to [0,1]; `crystallization_depth` = fractional within-window depth at which the normalized curve first reaches 0.5 (linear interpolation). Predicted: WRONG > CORRECT (later commit → construction).

**F2 — `trajectory_churn`.** On the same normalized `T_emit`: total variation minus net monotone displacement (non-monotone "wasted motion"). Discrete twin `top1_churn` (fraction of in-window adjacent layers where the first-answer-position logit-lens top-1 token flips) reported as robustness of the same feature. Predicted: WRONG > CORRECT.

**F3 — `late_gain_fraction`.** From RAW (un-normalized) in-window `T_emit`: (gain in the last third of the mid-stack) / (total in-window gain). Predicted: WRONG > CORRECT.

**F4 — `entropy_collapse_depth`.** Within-window normalize the next-token logit-lens entropy trajectory `H[l]` at the first-answer-predict position; depth of first half-max down-crossing. Predicted: WRONG > CORRECT (late collapse → construction).

Predicted signs are pre-committed. An in-sample sign flip = evidence AGAINST, never a re-spec. Mandatory robustness for all 4: recompute averaging over answer tokens 2..end (token-1 dropped).

---

## 4. PRIMARY metric (the single headline)

The **leave-one-out (n=121) cross-validated AUC** of ONE logistic regression on the frozen 4-feature set {F1,F2,F3,F4}, predicting correct-vs-wrong (positives = CORRECT). Headline = AUC of the out-of-fold predicted probabilities vs true labels (one OOF prediction per item). Features z-scored on TRAIN-fold statistics only. No tuning, no feature selection, no layer/threshold search — every constant (window [1,L−3], 0.5 crossings, 0.70 late band, logistic) is frozen in the hash.

- **Uncertainty:** 2000-resample stratified item-level bootstrap 95% CI, recomputing full LOO-CV OOF-AUC each resample.
- **Anchor (reported, not a 2nd headline):** `crystallization_depth`-alone threshold-free Mann-Whitney AUC.
- **Exploratory only (Holm-corrected, barred from headline):** per-feature univariate AUCs, all 2-/3-feature subsets, the secondary gold-trajectory variant.

---

## 5. KILL-GATE (frozen; print hash first; score once)

The chance band is 0.519 / 0.53. The gate requires a **bootstrap-real** margin, not nominal.

**SURVIVED requires ALL of:**
1. PRIMARY LOO-CV OOF-AUC **≥ 0.62** AND bootstrap-CI-lower **> 0.55**.
2. `crystallization_depth`-alone Mann-Whitney AUC **≥ 0.60**, CI-lower **> 0.53**, in the predicted direction (later → WRONG).
3. Permutation null (1000 label shuffles, full LOO-CV each): **p < 0.01** (observed > 99th pctile).
4. FLOOR-SELECTION partial — residualize all 4 features on [dist-to-maxent-median, dist-to-SE-median]: composite **≥ 0.58**, CI-lower **> 0.53**.
5. R_emit echo partials — residualize on r_28 AND R_emit@L20: composite **≥ 0.58**, CI-lower **> 0.53**; AND R_emit-ONLY baseline logistic **≤ 0.57** (dynamics must BEAT, not relabel); AND |corr(feature, r_28/R_emit)| reported (>0.5 falsifies any "immune to confidence echo" claim).
6. LENGTH partials — alen partial **≥ 0.58**; within-length-tercile AUC **≥ 0.57 in EACH band**.
7. TOKEN-FREQUENCY partial **≥ 0.58**.
8. TEACHER-FORCING controls — free-run-decode recompute **≥ 0.60**; forced-distractor lateness tracks item-correctness not string-status; token-1-dropped recompute **≥ 0.58**.
9. CLEAN-LABEL replication (hard exact/substring match + LLM-judge on ambiguous) — composite **≥ 0.58**.

**REPORT_AS_LANDED ("the wall is bedrock even in dynamics — retrieval/construction leave no readable cross-layer trajectory signature; every axis is at chance") iff:** OOF-AUC < 0.55 OR bootstrap-CI-lower ≤ 0.50 OR permutation p ≥ 0.05 — AND the off-floor contrast + clean-label diagnostics confirm the null is not a masked-real signal killed by the selection artifact. A valid, publishable hardening.

**PARTIAL (report, do NOT claim SURVIVED):** 0.55 < OOF-AUC < 0.62, OR any single mandatory control (4–9) fails while raw AUC ≥ 0.62.

---

## 6. Controls (every red-team killing-control incorporated)

| # | Control | Threshold | Guards against |
|---|---------|-----------|----------------|
| C1 | **Floor-selection partial** (resid on dist-to-maxent-median + dist-to-SE-median) + un-normalized entropy variant | ≥0.58, CI-low>0.53 | floor-selection circularity (red-team #1, THE KILLER) |
| C2 | **Off-floor contrast** (identical pipeline on n=121 random unselected subsample) | crack must be floor-specific | generic-signal masquerade / masked null |
| C3 | **Clean-label replication** (exact/substring + LLM-judge) | ≥0.58 | embedding-proxy length/frequency LABEL bias (red-team #2) |
| C4 | **Token-frequency** partial + stratified AUC | ≥0.58 | short/frequent crystallize early independent of mechanism |
| C5 | **Within-length-tercile AUC** + length-matched subsample + fixed-token-budget | ≥0.57 each band | feature+label length confound (lab's own prior failure) |
| C6 | **R_emit@L20 + r_28 partials**; R_emit-only baseline; feature↔r_28/R_emit corr matrix | ≥0.58; baseline ≤0.57 | "old K axis relabeled as dynamics" (red-team #3) |
| C7 | **Free-run decode** recompute; **forced-distractor** contrast; **mid-stack window**; **token-1 dropped** | ≥0.60 / lateness tracks item / ≥0.58 | greedy-argmax tautology + teacher-forcing artifact (red-team #4) |
| C8 | **Permutation null** 1000× full LOO-CV | p<0.01 | multiple-comparisons / n=121 overfit firewall |
| C9 | **Bootstrap CI** 2000× | composite CI-low>0.55 | chance-band margin firewall |
| C10 | **Monotonicity/degeneracy** sanity | report; interpolation well-defined | flat-trajectory artifacts |
| C11 | **Pre-committed signs** | flip = evidence against | post-hoc sign re-spec |
| C12 | **Gold-trajectory variant** | SECONDARY only, excluded from gate/flag | gold leakage into headline |

---

## 7. Cross-model (pre-registered replication, NOT extra bites at significance)

For EACH of Qwen2.5-3B and Llama-3.2-3B (4bit): re-derive that model's OWN confident-consistent floor (its own maxent + SE medians), re-freeze its answer-key SHA, apply the IDENTICAL frozen pipeline + all mandatory controls (window [1,L−3] scaled to its depth).

- **Per-model crack:** OOF-AUC ≥ 0.60, CI-lower > 0.53, permutation p < 0.05, signs directionally consistent.
- **GENERALITY claim (architecture-general):** requires **BOTH** larger models to crack in the same direction (NOT the gameable ">=1 of 2"). Llama is the architecture-shift test.
- A crack on Qwen-1.5B alone → REPORT_AS_LANDED scoped to Qwen-1.5B (architecture-specific). Cross-model runs cannot move the primary Qwen-1.5B verdict.

---

## 8. Honest verdict mapping

- **SURVIVED** ⇔ gate items 1–9 ALL pass. Claim: *the wall is cracked from a genuinely dynamical (not static, not artifactual) axis; a deployable label-free + gold-free confident-misconception detector exists.* Generality claimed only if BOTH larger models also crack.
- **REPORT_AS_LANDED** ⇔ OOF-AUC < 0.55 OR CI-lower ≤ 0.50 OR perm p ≥ 0.05, AND C1/C2/C3 confirm the null is not a masked-real signal. Claim: *the wall is bedrock even in dynamics — every axis (output confidence, static content, process dynamics) is at chance on the confident-consistent floor.* Publishable hardening.
- **PARTIAL** ⇔ 0.55 < AUC < 0.62, or a mandatory control fails while raw AUC ≥ 0.62. Reported descriptively; NEVER claimed as a crack.

**Deployment operating point (only if SURVIVED):** threshold at FPR = 0.10 on the confident floor (flag ≤10% of confident-correct); report TPR (recall of hidden confident errors that output signals miss) as the headline operating point.

A null is a real result. The experiment is decisive in both directions BECAUSE the deadliest confounds (floor-selection circularity, proxy-label length/frequency bias, R_emit echo, greedy-argmax tautology) are partialled out before any SURVIVED verdict, and the off-floor + clean-label diagnostics are run before any REPORT_AS_LANDED verdict.
