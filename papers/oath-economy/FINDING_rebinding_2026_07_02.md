# FINDING — the gap is (mostly) a hyperlink: third-party re-binding of an unbound model card

**Fathom Lab · 2026-07-02 · prereg: `PREREG_rebinding_2026_07_02.md` (frozen before the receipts dataset was
opened, `3596332`). Verdict per frozen criteria: PARTIAL — 18/22 in-scope claims re-bind exactly.**

Rung 1 found the `meta-llama/Llama-3.2-1B-Instruct` card carries 201 numeric eval claims, none bound to a
re-runnable receipt — while Meta's raw eval receipts exist as a public, unlinked HF dataset. This rung asked
the sharpest question that finding permits: **can a third party, with no vendor involvement, re-bind the
claims to the receipts?** Framing rule carried from rung 1: the gap is an industry property; no accusation
language; deltas are reported and correction is invited.

## Method receipts (all committed here)
- `rebind_step1_parse_card.py` — re-fetch card (sha256 verified against `CARD_MANIFEST.json`), MECHANICAL
  table parse → **201 claims, exactly confirming rung 1's single-scout count** (second pass passed; the
  earlier v1-parser divergence was the parser's table coverage, not the scout).
- `rebind_step2_match.py` — explicit per-benchmark mapping (no fuzzy matching) against
  `meta-llama/Llama-3.2-1B-Instruct-evals` `Details_metrics_details` (2024-09-23). In-scope: the 22 claims
  about the 1B-Instruct model (15 instruct-table bf16 + 7 multilingual-MMLU); other columns (base/3B/8B/
  quantized) belong to other datasets/none.
- `rebind_result.json` — the full graded table.

## Result

| grade | n | detail |
|---|---|---|
| **MATCH** | **18/22 (82%)** | every one exact to the published decimal (Δ = 0.0): MMLU 49.3, GSM8K 44.4, ARC-C 59.4, GPQA 27.2, Hellaswag 41.2, Nexus 13.5, InfiniteBench En.QA 20.3 / En.MC 38.0, NIH 75.0, MGSM 24.5, Open-rewrite 41.6 (card's `micro_avg/rougeL` = receipts' `average/rougeL`, naming alias), + all 7 multilingual MMLU |
| ABSENT | 3 | **TLDR9+**: no data anywhere in the receipts. **IFEval**: card's metric is a 4-component composite; receipts carry only `prompt_acc` (55.3/51.0) — instruction-level components absent. **MATH**: card states 0-shot; the receipts' only MATH record is labeled **4-shot** (em 30.4 vs card 30.6) — the card and its own receipts disagree about the config |
| VALUE-MISMATCH | 1 | **BFCL V2**: card 25.7 matches NO receipt aggregate (`macro_avg/acc` 38.7, `average/acc` 21.5) |

**Verdict (frozen): PARTIAL** — 82% < the 90% REBOUND bar; 4.5% VALUE-MISMATCH < the 10% RECEIPT-MISMATCH bar.

## What this means

1. **For 18 of 22 claims, the binding gap is literally a hyperlink.** A third party re-bound them to the
   decimal using receipts the card never links — receipts that even carry full eval configs (temperature,
   seed, prompt template, shots), i.e. *better* provenance than the card's unnamed "internal evaluations
   library" phrase suggests exists. The industry's receipts problem, at least here, is binding — not capability.
2. **The residue is exactly why binding should be mandatory.** One benchmark with no receipts at all, one
   composite that can't be recomputed from what was published, one card-vs-receipts config disagreement
   (0-shot vs 4-shot MATH), and one number matching nothing. None of these is discoverable *without* doing
   the re-binding — unbound claims hide their own inconsistencies.
3. **Instrument honesty:** the first matcher (fuzzy label/tag matching) produced **3 phantom mismatches**
   (a label spelling, an unreached composite branch, a tag alias) that would have flipped the verdict to a
   false RECEIPT-MISMATCH. Self-audit removed them before publication; the committed matcher uses an explicit
   auditable mapping. A future card-claim extractor (rung 2) must treat label/tag normalization as a
   first-class hazard — the tension you find is, first, a hypothesis about your own matcher.

## Next
- Rung 2 (card-claim extractor) now has both its spec (292 claims, 3 cards) and its hazard list (this file §3).
- The strongest follow-up here: per-example RECOMPUTATION (the receipts include per-example records) for the
  18 matches — upgrading MATCH → MATCH-RECOMPUTED, the strongest form of proof-carrying short of re-running
  the model. Reserved as a separate rung.

## Reproduce
```
python papers/oath-economy/rebind_step1_parse_card.py   # fetch+verify card, parse 201 claims
python papers/oath-economy/rebind_step2_match.py        # match 22 in-scope vs receipts -> rebind_result.json
```
(Requires accepting the Llama license on HF for the gated card; the receipts dataset is public.)

*Nothing crosses unseen — and 18 of 22 claims just crossed with receipts a hyperlink away.*
