# RESULT — OATH v0.6.1: batteries PASS, corpus gate fires `V061_FALSE_ACCUSATION` — and the audit hand-check finds four genuine catches beside the one false accusation

Fathom Lab · 2026-07-31 · scored under `PREREG_oath_v061_shascrub_epsilon_2026_07_31.md`
(frozen at `c03c7a1`). Receipts: `oath_v061_battery_result.json`,
`oath_v061_corpus_after.json`. Harness: `run_oath_v061_battery.py`.

## Gate results

- **G1 (recall): 18/20, bar 18 — PASS.**
- **G2 (catch): 17/20, bar 16 — PASS.** The corrected battery works: with sampling conditioned
  on resolvable receipts + trigger-bound lines and mutation in significant fractional digits
  1–6, mutants of full-precision claims are caught. Misses reported in the receipt (two
  formula-notation adjacency skips, one unbound abstain).
- **G2b (reported): unbound-line share of the full 484-token pool = 0.5227** — the measured
  size of the standing trigger-recall debt on this class.
- **G5 (attribution severability): PASS** — stem-preference ON vs OFF: zero per-doc status
  differences corpus-wide.
- **G6: PASS** at the time of the run (18/18 in `test_certify_recall.py` including the five
  new pins; reverted with the fix per the outcome table).
- **G3 (no false accusations): FAIL — `V061_FALSE_ACCUSATION`, the pre-committed kill.**
  Five certificates flipped HELD→FAILED; the hand-check of all five:

| doc | token | hand-check verdict |
|---|---|---|
| `FINDING_framelocality_dogfood_2026_07_29` | `−0.01538461538461533` | **FALSE ACCUSATION — the kill.** The receipt holds `discriminating_margin = -0.01538461538461533` exactly. The doc writes the sign as Unicode minus (U+2212); `_NUM` cannot read it as a sign, extraction yields `+0.0153…`, and an accurate claim is accused. A THIRD verifier defect: sign-blind extraction on typographic minus. |
| `DATASHEET_conscience_2026_07_24` | `0.2711864406779661` | genuine — bare-arm accuracy (16/59) computed in prose, persisted in no receipt. |
| `FINDING_frontier_incontext_oof_2026_07_30` | `−0.2793478260869565` | genuine — the margin (caved − held) derives from two grounded leaves but is persisted nowhere; also U+2212-signed, so its repair additionally requires the sign fix. |
| `FINDING_frontier_recovery_2026_07_27` | `0.4782608695652174` | genuine — quotes cycle 83's `frontier_knowsay_result.json`, absent from this certificate's receipt set. |
| `FINDING_scale3b_2026_07_29` | `+0.7285714285714285` | **genuine catch, and a good one:** the receipt holds `…86` — a real transcription error in the 16th digit, caught by the epsilon fix working exactly as specified. |

## Disposition

Per the frozen outcome table: `certify.py` and the test pins reverted at this commit; the
failing case published above. The successor prereg (`PREREG_oath_v062_signed_extraction_2026_07_31.md`)
carries all three fixes — SHA-scrub recall, the epsilon hole, and U+2212 sign normalization —
with the same battery v2 harness and bars, and G3 re-armed corpus-wide (sign normalization
changes the extraction surface for every typographic-minus negative, so the full audit runs
again). The four genuine catches become the repair loop of the shipping cycle: addendum
receipts for the computed-never-persisted values, receipt-set extension for the cross-cycle
quote, and a one-digit doc correction for the transcription error — each under its own commit,
none touching a committed result receipt.
