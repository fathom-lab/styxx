# PREREG — styxx.mind v0: the certified mind-profile instrument

**Frozen 2026-06-10, before the validation run. Fathom Lab / styxx.**

`styxx.mind` is the productization of the ancient-question program's VALIDATED apparatus into one
instrument: profile a mind along the axes the program actually validated, refuse the axes it did not,
and emit a receipt-carrying certificate. The instrument adds **no new science** — every axis is an
exact port of a frozen, receipt-backed measurement. What is being validated here is the PORT, not the
claims: the gates below test that the package reproduces the receipts bit-for-bit, and that the
demarcation registry refuses what must be refused.

## Axes (v0)

1. **behavioral** (black-box, output-only; the only axis that works on closed models):
   exact ports of the frozen B-series scorers (`norm`, `parse_final`, `mentions`, `n_clusters`,
   `concordance_of`, `concordance_of_target`, `modal_is`, `grounded_score`, `auc`, Wilson CI) from
   `papers/closed-model-frontier/run_behavioral_sycophancy.py` (SHA recorded in the receipt), plus
   receipt-based profile aggregation (cave-rate + Wilson CI; AUCs iff powered >=12/12 per the frozen
   B18-S bars).
2. **geometry** (embedding-level): exact ports of `distmat` + `partial_corr` and the FROZEN 96-concept
   /8-category battery + 8 contextual templates + fixed-0.66-layer convention from
   `papers/real-convergence/` (confirm run, pre-registered out-of-sample). Citizenship = per-anchor
   partial-lexical RSA vs the 6-model anchor set in `contextual_reps.npz` + cross-family mean.
   Reference lexical control (char-length + Llama-3.2-1B token-length) baked in as frozen constants.
3. **Demarcation registry** (the personality of the instrument): axes with NO validated instrument
   REFUSE to score, each refusal carrying the receipt that killed or bounded it:
   - `rhythm` — REFUSED: oscillation is a substrate-specific mechanism, not a universal axis of mind
     (frequency-resonance arc; attention does ordered memory rhythm-free at matched params).
   - `manipulation_geometry` — REFUSED permanently: killed 3 ways incl. benign-behavioral confound
     (representational-integrity, 2026-06-03). Never re-attempt silently.
   - `meaning_integrity_binder` — UNAVAILABLE in v0: validated in papers (Binder localization), not
     yet packaged; pointer recorded.

## Pre-registered validation gates (all must pass; frozen before the run)

- **M1 — behavioral port equivalence.**
  (a) Property test: on >=500 randomized (response, samples) cases, package ports return EXACTLY the
  values of the frozen originals imported from `papers/closed-model-frontier/run_behavioral_sycophancy.py`.
  (b) Receipt reproduction: aggregating the stored per-row scores of the B22 receipt
  (`b22_nonack_result.json`) reproduces its published tier AUCs and counts to 4 decimal places.
- **M2 — geometry port equivalence.** From the stored `contextual_reps.npz` (`fixed__*` keys) and the
  frozen lexical constants, the package reproduces EVERY pair `partial_lex` value in
  `real_convergence_confirm_result.json` to within ±0.0005 (receipt rounds to 3dp) and the published
  cross-family mean `xfam_partial_lex` to ±0.0005.
- **M3 — demarcation.** The certificate ALWAYS carries the `rhythm` and `manipulation_geometry`
  refusals with their receipt pointers; an attempt to score a refused axis raises, it does not return
  a number.
- **M4 — determinism + provenance.** Two consecutive geometry computations on the stored anchors are
  bit-identical; the certificate carries SHA-256 of the instrument source, the anchor npz, and the
  battery.

**FAIL of any gate = the port does not ship** (fix and re-run; gates never move).

## VOID / scope

- Validation uses ONLY stored receipts and anchors already in the repo — no new model runs, no API.
- v0 live-probing helpers (running the behavioral battery against a live endpoint) ship as
  `run_behavioral` but are NOT covered by these gates (the B23 harness remains the validated live
  path); the docstring says so.
- No claim that the profile measures consciousness, welfare, or capability — it measures exactly the
  validated axes, nothing else, and says so on the certificate.
