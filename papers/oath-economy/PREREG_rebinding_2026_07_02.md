# PREREG — M1 rung 3: can a third party re-bind the unbound card?

**Fathom Lab · 2026-07-02 · committed BEFORE the receipts dataset is opened. Publishes regardless of verdict.**

## Question
Rung 1 found the Llama-3.2-1B-Instruct card carries 201 numeric eval claims, 0 bound — while Meta's raw eval
receipts (`meta-llama/Llama-3.2-1B-Instruct-evals`, public HF dataset) exist unlinked. Rung 3 tests the
finding's sharpest sentence: is the gap a hyperlink, or a capability? Can a third party, with no vendor
involvement, re-bind the card's claims to the existing receipts?

## Method (frozen)
1. Re-fetch the card (verify sha256 against `CARD_MANIFEST.json` — hash mismatch ⇒ stop and report).
   Parse its benchmark tables MECHANICALLY (deterministic script, committed) into claims
   {benchmark, config(shots/metric), column, value}. The mechanical count is also the second-pass check on
   rung 1's single-scout count of 201: divergence >10% is reported as a rung-1 correction, not hidden.
2. Open the receipts dataset. For each 1B-Instruct claim (the model this dataset covers), grade:
   - **MATCH** — a (benchmark, config)-matching record yields the claimed value within |Δ| ≤ 0.15 after
     percent/fraction normalization (covers rounding to one decimal).
   - **MATCH-RECOMPUTED** — stronger: the aggregate RECOMPUTED from per-example records matches within the
     same tolerance (preferred wherever per-example records exist).
   - **VALUE-MISMATCH** — (benchmark, config) found, value disagrees beyond tolerance (report Δ).
   - **ABSENT** — no matching benchmark/config record in the dataset.
   Claims about other models/columns (3B/8B/quantized) are OUT OF SCOPE for this dataset and counted separately.
3. Verdict — **REBOUND (OATH-HELD-BY-PROXY)** iff ≥90% of in-scope claims grade MATCH or MATCH-RECOMPUTED.
   **RECEIPT-MISMATCH** iff >10% grade VALUE-MISMATCH. Otherwise **PARTIAL** with the full table.

## Falsification map
- REBOUND ⇒ the gap is literally a hyperlink: the industry's receipts problem is binding, not capability.
  Strongest possible form of rung 1.
- PARTIAL/ABSENT-heavy ⇒ the "receipts exist" premise weakens; rung 1's sharpest sentence gets corrected to
  exactly what the data supports.
- RECEIPT-MISMATCH ⇒ darker than unbound: published claims disagree with the vendor's own receipts. Framing
  rule from rung 1 applies with full force — report the delta, no accusation language, invite correction.
