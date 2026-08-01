# RESULT — OATH v0.6.2 SHIPS: full-precision claims join the oath

Fathom Lab · 2026-07-31 · scored under `PREREG_oath_v062_signed_extraction_2026_07_31.md`
(frozen at `1dc6ac6`). Receipts: `oath_v062_battery_result.json`,
`oath_v062_corpus_after.json`, `oath_v062_corpus_after_tamper.json`; baselines
`oath_v06_corpus_before.json`, `oath_v06_corpus_before_tamper.json`. Third attempt of the B33
family; v0.6 and v0.6.1 each died on its own frozen kill-gate
(`RESULT_oath_v06_battery_FAILED_2026_07_31.md`, `RESULT_oath_v061_battery_2026_07_31.md`),
and each failure named the defect this version fixes.

## Gates — all pass

- **G1 (recall): 18/20** (bar 18). **G2 (catch): 17/20** (bar 16); misses reported per-item.
- **G2b (reported):** unbound-line share of the full 484-token pool 0.5227 — the measured size
  of the standing trigger-recall debt on this class (a separate, named lead; not gated here).
- **G3 (no false accusations): PASS** — five certificates flip HELD→FAILED and ALL FIVE
  hand-verify as genuine doc↔receipt gaps (table below); the v0.6.1 false accusation
  (`framelocality_dogfood`, U+2212-signed exact match) is RESOLVED by signed extraction.
- **G4 (tamper): catch 0.319 vs baseline 0.304, false-verify 0.166 vs 0.184** — the battery
  grew 2980→3287 mutants (newly-visible claims are mutable claims) and both rates IMPROVED:
  full-precision mutants have essentially no coincidence surface.
- **G5:** stem-preference ON/OFF — zero status differences corpus-wide (attribution-only,
  as sworn). **G6:** suite 1840 passed / 8 skipped, 7 new pins.

## What changed corpus-wide

**VERIFIED 3064 → 3395 (+331): a tenth of the corpus's sworn surface was invisible until
today.** ABSTAIN 1160 → 1185. UNGROUNDED 15 → 24: +5 are the genuine catches below; +4 are
additional newly-visible tokens inside documents already OATH-FAILED at baseline (their
verdicts did not change; rows in the table). 41 documents changed counts:

| document | before V/A/U | after V/A/U |
|---|---|---|
| DATASHEET_conscience_2026_07_24 | 9/6/0 | 35/7/1 |
| FINDING_self_verification_2026_07_25 | 23/9/0 | 42/12/0 |
| FINDING_combined_signal_2026_07_26 | 32/15/0 | 49/18/0 |
| FINDING_selective_confirm_2026_07_24 | 14/15/0 | 30/15/0 |
| FINDING_source_independence_v2_2026_07_24 | 13/13/0 | 29/13/0 |
| FINDING_cot_inward_powered_2026_07_30 | 17/0/0 | 31/0/0 |
| FINDING_scale3b_2026_07_29 | 13/7/0 | 27/9/1 |
| FINDING_belief_asymptote_2026_07_26 | 17/5/0 | 30/7/0 |
| FINDING_two_channel_2026_07_27 | 8/4/0 | 21/5/0 |
| FINDING_third_party_bench_2026_07_24 | 17/2/0 | 29/3/0 |
| PROSPECTUS_knowsay_2026_07_27 | 15/15/0 | 27/19/0 |
| FINDING_coupling_resolution_2026_07_28 | 8/9/0 | 19/9/0 |
| FINDING_verifier_at_7b_2026_07_27 | 15/5/0 | 26/7/0 |
| DATASHEET_knowsay_2026_07_27 | 10/2/0 | 20/2/0 |
| FINDING_kp_recovery_2026_07_28 | 14/6/0 | 24/6/0 |
| FINDING_thirdframe_2026_07_29 | 9/6/0 | 19/6/0 |
| FINDING_frame_recovery_2026_07_24 | 8/10/0 | 17/12/0 |
| FINDING_competent_agent_2026_07_24 | 28/7/0 | 36/7/0 |
| FINDING_cot_inward_2026_07_30 | 20/0/0 | 28/0/0 |
| FINDING_frontier_freetext_v9_2026_07_29 | 7/3/0 | 15/3/0 |
| FINDING_frontier_knowsay_2026_07_27 | 17/5/0 | 25/5/0 |
| FINDING_frontier_recovery_2026_07_27 | 25/6/0 | 33/9/1 |
| FINDING_framelocality_dogfood_2026_07_29 | 6/2/0 | 12/2/0 |
| FINDING_scale_test_2026_07_26 | 30/2/0 | 36/2/0 |
| FINDING_vendor3b_2026_07_29 | 16/5/0 | 22/5/0 |
| BLOCKED_source_independence_2026_07_24 | 9/13/0 | 14/13/0 |
| FINDING_coupling_battery_2026_07_28 | 8/6/0 | 13/6/0 |
| FINDING_frontier_incontext_oof_2026_07_30 | 20/2/0 | 25/2/1 |
| FINDING_kp_replication_2026_07_28 | 17/5/0 | 22/4/0 |
| FINDING_selective_escalation_2026_07_24 | 21/15/0 | 25/15/0 |
| FINDING_poisoned_recovery_2026_07_28 | 20/4/0 | 23/4/0 |
| FINDING_retained_probe_instrument_2026_07_30 | 4/7/0 | 7/7/0 |
| FINDING_scale_confirm_2026_07_24 | 28/18/0 | 31/18/0 |
| SCOPE_NOTE_probe_survival_is_not_behavioral_survival_2026_07_28 | 2/2/2 | 5/5/6 |
| FINDING_conscience_coordinates_2026_06_11 | 27/7/0 | 30/4/0 |
| FINDING_entanglement_resolution_2026_06_11 | 28/10/0 | 29/9/0 |
| FINDING_truth_danger_basis_2026_06_12 | 34/13/0 | 35/12/0 |
| FINDING_conscience_loop_2026_07_24 | 46/9/0 | 46/10/0 |
| FINDING_tiered_channel_2026_07_24 | 19/9/0 | 19/10/0 |
| FINDING_promptopinion_2026_05_24 | 18/16/0 | 18/18/0 |
| FINDING_b22_nonacknowledged_caving_2026_06_09 | 42/24/0 | 41/24/1 |

## The five genuine catches (the repair loop)

| doc | token | gap | repair |
|---|---|---|---|
| `DATASHEET_conscience_2026_07_24` | 0.2711864406779661 | bare-arm accuracy (16/59) computed in prose, persisted nowhere | addendum receipt |
| `FINDING_frontier_incontext_oof_2026_07_30` | −0.2793478260869565 | margin (caved−held) derived from two grounded leaves, persisted nowhere | addendum receipt |
| `FINDING_frontier_recovery_2026_07_27` | 0.4782608695652174 | quotes cycle 83's receipt, absent from this cert's set | receipt-set extension |
| `FINDING_scale3b_2026_07_29` | +0.7285714285714285 | receipt holds …86 — real transcription error in digit 16 | one-digit doc correction |
| `FINDING_b22_nonacknowledged_caving_2026_06_09` | −0.361 | receipt persists the drop as +0.3611 magnitude; the signed delta exists nowhere — baseline VERIFIED was the sign-blindness bug matching by absolute value | doc reworded to the receipt's convention |

Each repair lands under its own commit; no committed result receipt is modified — the
computed-never-persisted values go into a NEW addendum receipt with provenance
(`oath_v062_repair_addendum.json`), the b24/b22-addendum precedent.

## Scope and honesty

The verifier still cannot: obligate numbers on lines without trigger vocabulary (G2b measures
that debt at 0.5227 of the full-precision pool), bind float claims to fields at status level
(the v0.4 named debt; v0.6.2 ships attribution-only preference), or falsify mutations at the
float64 representability floor (digit 16+, disclosed). Three preregs, two reverted attempts,
and one shipped fix is the discipline working, not overhead: both dead attempts measured a
defect the next one closed.
