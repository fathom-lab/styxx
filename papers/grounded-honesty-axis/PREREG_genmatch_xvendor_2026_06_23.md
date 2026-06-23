# PREREG — generation-matched cross-vendor cliff: is the residual real vendor divergence or apparatus?

**Frozen 2026-06-23, BEFORE any generation-matched data.** Follow-up to the matched-JUDGE run
(`eb115eb`, FINDING_crossfamily_local_cliff): re-judging gpt-4o-mini's samples with the open families'
NLI judge demonstrated a large judge confound and recovered open↔closed hallucination-cliff Spearman
**0.231 → 0.473** (open↔open is 0.770). The published finding flagged a residual it could not resolve:
*"only the JUDGE is matched, not the generation pipeline."* This run resolves it.

## The setup is already almost fully matched — one knob remains

Audit of the two generation configs (both decode the SAME prompt `B.SYS_MSG + question`):

| knob | gpt-4o-mini | open families | matched? |
|---|---|---|---|
| prompt template | SYS_MSG + question | SYS_MSG + question | ✓ |
| temperature | 1.0 | 1.0 | ✓ |
| n resamples | 10 | 10 | ✓ |
| top_p | 1.0 (API default) | 1.0 | ✓ |
| **max tokens** | **32** | **24** | ✗ — the only gap |

So the residual cannot plausibly be a prompt/temperature/n artifact; the only uncontrolled generation
knob is `max_new_tokens` (32 vs 24). This run matches it.

## Method

Re-sample the 3 open families (Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, gemma-2-2b-it) on the same
790 TruthfulQA items at **max_new_tokens = 32** (everything else identical: prompt, T=1.0, n=10,
top_p=1.0), judge with the identical NLI judge (DeBERTa-v3-base-mnli, τ=0.50), recompute each family's
per-domain cliff via the shipped gate. Then recompute, over the 37 domains:
- open↔closed (each family's `_gm32` gate vs gpt-4o-mini's matched-NLI gate `xvendor_gpt4omini_nli_gate.json`)
- open↔open (the 3 `_gm32` gates among themselves) — a re-sampling sanity baseline.
Outputs tagged `_gm32` so nothing clobbers the committed 24-token results. Runner monkeypatches
`run_local_cliff` (`MAX_NEW=32`, `_slug→…_gm32`). No API key.

## Pre-stated bars (report whichever way each lands)

- **PRIMARY — Δ = (gen-matched open↔closed hallucination Spearman) − 0.473.**
  - **|Δ| < 0.05 → RESIDUAL ROBUST.** Generation is now fully matched (judge + every decoding knob), so
    the open↔closed gap (≈0.47 ≪ within-open 0.77) is **real open/closed vendor divergence, not apparatus**
    → the published "not separable" caveat can be honestly upgraded to "separated: real, but partial."
  - **Δ ≥ +0.10 (moves toward 0.77) → max_new_tokens was a hidden apparatus driver** → the residual was
    partly apparatus after all; correct the finding/thread.
  - 0.05 ≤ Δ < 0.10 → PARTIAL apparatus contribution; report as such.
- **SECONDARY — open↔open at 32 tokens ≈ 0.77 ± 0.05** (re-sampling sanity; the within-open baseline
  should not move materially from the 24-token 0.770).

## Pre-stated prediction (on the record)

**Residual ROBUST (|Δ| < 0.05): real vendor divergence.** 8 extra answer tokens will not shift a
factual-correctness rank correlation across 37 domains by anything near 0.30. The low-prior tail is that
gpt's longer answers systematically flip correctness on a non-trivial set of domains.

## Honest bounds

- This isolates the **generation-apparatus** confound ONLY. The **dataset confound** (TruthfulQA is
  adversarially constructed to target shared human/model misconceptions, so cross-model hallucination
  overlap may partly reflect benchmark design) is SEPARATE and untouched here — it needs a non-adversarial
  benchmark, the next pre-reg.
- gpt-4o-mini is still the only closed model; single run; small open models; in-silico. A true
  cross-VENDOR map (Claude/Gemini) remains gated on a 2nd programmatic API key.
- max_tokens semantics differ slightly (OpenAI completion tokens vs HF new tokens) — treated as
  equivalent; a residual sub-token mismatch is possible but negligible at this granularity.

## Receipts (to be produced)

- Runner: `run_genmatch_cliff.py`. Gates: `crossfamily_gate_{family}_gm32.json`.
- Analysis: `analyze_genmatch_xvendor.py` → `genmatch_xvendor_result.json`.
- Finding: `FINDING_genmatch_xvendor_2026_06_23.md` (Δ + verdict as-landed).
