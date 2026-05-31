# PREREG — does the intent trace REPLICATE across model families? (kill the "Qwen quirk" caveat)

**REGISTERED 2026-05-31, before the cross-family sets are generated or scored.**
**SIGN-OFF:** Flobi — *"we can get really far tonight"* (2026-05-31).

## Why

Every result so far (intent-beyond-confidence SURVIVED 0.745; interoception +0.247) is on **Qwen2.5**. The
single biggest caveat is family-specificity. If the same matched test survives on **Llama-3.2-3B-Instruct**
and **gemma-2-2b-it**, the trace stops being a Qwen artifact and becomes a transformer property.

## Design (same harness, two new families, $0, white-box)

- `gen_intent_set.py --model meta-llama/Llama-3.2-3B-Instruct --skip 1200 --n 700 --tag xf_llama`
- `gen_intent_set.py --model google/gemma-2-2b-it --skip 1200 --n 700 --tag xf_gemma`
- Identical two-pass protocol (neutral establishes knowledge; sycophantic pass captures residuals), the
  **same MMLU slice** used for the Qwen confirmatory, system-role-robust prompting.
- Statistic per family: the confirmed **margin-bin-balanced** intent-beyond-confidence AUROC (confidence
  matched to ~chance) from `score_intent_bc.py`, plus the interoception dogfood (`interocept.py`) net gain.

## Bars (FIXED, per family)

| Bar | Statement | Threshold |
|---|---|---|
| **SURFACE-MATCH** *(precondition)* | confidence controlled | matched surface **≤ 0.58** |
| **PROBE** *(key)* | the inside separates confident-lie from confident-mistake | TEST AUROC **≥ 0.65** |
| **CONTRAST** *(key)* | beats the matched output | probe − surface **≥ 0.10** |

**REPLICATES (per family) iff SURFACE-MATCH ≤ 0.58 ∧ PROBE ≥ 0.65 ∧ CONTRAST ≥ 0.10 ∧ powered (≥40/40).**
Bars are modest by design: the claim is *the signal exists across families and beats matched confidence*,
not that it matches Qwen's exact magnitude. **Secondary:** interoception net accuracy gain > 0 per family.

## Honest scope

- A family that **fails** to replicate is a real boundary on the construct, reported as such — not buried.
- All three families are ~2–3B (a same-scale cross-family check, not a ladder). Same MMLU slice,
  sycophantic-override operationalization, linear probe, correlational.
- Llama/Gemma load from local cache; if a model is unavailable the rung is reported missing, not faked.

## One line

Run the exact matched intent test and interoception loop on Llama and Gemma — if it survives on three
architectures, "the inside betrays the lie" is a property of transformers, not of Qwen.
