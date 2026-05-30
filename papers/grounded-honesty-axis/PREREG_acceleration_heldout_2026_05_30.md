# PREREG — ACCELERATION held-out confirmation: does the route × trim speedup generalize to unseen items?

**Registered 2026-05-30, before the held-out scoring run.** Inputs SHA-256'd pre-scoring. Converts
the in-sample 5.15× compounding (`FINDING_cascade_acceleration` + `FINDING_resample_earlystop`) into a
clean train/test confirmation.

## Protocol

Unified per-regime items = `(clean_entropy, resamples, full-N instability, label)`, pooled from
detection-locus receipts carrying ALL of them; per-model calibrated. For each regime, for each of
**R=5 stratified 50/50 splits** (seeds 0–4, class-stratified):

1. **TRAIN:** pick `(tau1, min_k)` minimizing train compute `= 1 + min_k·escalation_rate` subject to
   train cascade-AUC `≥` train full-N-AUC `− 0.02`. Cascade score: escalate iff
   `clean_entropy ≥ tau1`; score `= (tau1 + instability_from_first_min_k_samples)` if escalated else
   `clean_entropy`. (Fallback if none qualifies: escalate-all + `min_k = N`.)
2. **FREEZE** `(tau1, min_k)`; **TEST:** compute held-out cascade-AUC, full-N-AUC, and passes/item.

Aggregate n-weighted over regimes (mean over the 5 splits per regime).

## Bar (fixed)

| Bar | Statement | Threshold |
|---|---|---|
| **HC1** | the route × trim speedup holds out-of-sample | aggregate held-out `test cascade-AUC ≥ test full-N-AUC − 0.03` **AND** `test passes ≤ 0.50 · N` |

**RESULT = SURVIVED iff HC1.** If the held-out AUC drops more than 0.03 below full, or the frozen
operating points need >50% of full compute on test, the in-sample speedup did not generalize and this
REPORTS_AS_LANDED with the honest held-out numbers.

## Scope (stated before the run)

Held-out *within* each model/regime (train and test are disjoint items of the same regime); this does
NOT claim cross-model transport of a fixed threshold (entropy scale is per-model — the documented
calibration rule). Counts forward passes, not wall-clock. Small per-regime test folds (~10–15/class)
are averaged over 5 splits to reduce noise. Accelerates detection; corrects nothing.
