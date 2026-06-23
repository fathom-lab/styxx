# FINDING — generation-matched cross-vendor cliff: the "OpenAI outlier" was largely apparatus

**Pre-registered in `PREREG_genmatch_xvendor_2026_06_23.md` (frozen `b030c10`, before data). Runner
`run_genmatch_cliff.py`, analysis `analyze_genmatch_xvendor.py`. Local open weights, no API key.**

## What this resolves

The matched-JUDGE run (`eb115eb`) reported open↔closed hallucination-cliff Spearman 0.473 vs within-open
0.770 (gap 0.30) and flagged a residual it could not separate from generation. Audit showed generation
was already matched on prompt/T=1.0/n=10/top_p; the **only** unmatched knob was `max_new_tokens` (gpt 32
vs open 24). This run re-samples the 3 open families at 32 to fully match it.

## Result

| comparison (37 domains, NLI judge) | 24-token (committed) | 32-token (gen-matched) | Δ |
|---|---|---|---|
| open↔open hallucination | **0.770** | **0.613** | **−0.157** |
| open↔closed hallucination | **0.473** | **0.510** | **+0.037** |
| **gap (open↔open − open↔closed)** | **0.297** | **0.103** | **−0.194** |
| open↔open refusal | 0.426 | 0.399 | −0.027 |
| open↔closed refusal | 0.416 | 0.351 | −0.065 |

- **PRIMARY bar (open↔closed robust to gen-match): PASSES** — Δ +0.037 (< 0.05). The cross-vendor
  hallucination-cliff agreement is ~0.51 regardless of token-matching.
- **SECONDARY sanity bar (open↔open stable ≈0.77 ±0.05): FAILS** — within-open dropped 0.770 → 0.613.
  This is the load-bearing result.

## The headline correction: the gap mostly closes; the "outlier" was apparatus

At a **fully-matched apparatus** (everything at 32 tokens + NLI judge), the open↔open / open↔closed gap
is only **0.10**, not the 0.30 the mismatched comparison implied. The fully-matched 4-provider
hallucination matrix is a **loose ~0.5–0.6 web, not a tight-open-cluster-with-an-outlier**:

| pair | 32-token Spearman |
|---|---|
| Alibaba–Meta | ~0.63 |
| Alibaba–Google | 0.58 |
| Meta–Google | 0.63 |
| **Google–OpenAI** | **0.60** ← as high as within-open pairs |
| Alibaba–OpenAI | 0.53 |
| Meta–OpenAI | 0.40 ← the actual low |

"Open vs closed" is **not** a clean split: Google (Gemma) agrees with OpenAI (0.60) as strongly as with
other open models, while Meta (Llama) is the pair that diverges most from gpt. The earlier "open cluster
0.77 / OpenAI outlier 0.47" framing (4-provider matrix commit `f8…`, and the implication in today's
public thread) was **substantially an artifact of the 24-vs-32-token apparatus mismatch**, which both
inflated within-open agreement and deflated open-vs-gpt agreement.

## The load-bearing confound (unresolved): tokens vs sampling variance

The within-open drop (0.770 → 0.613) **confounds two causes** and this run cannot separate them:
1. **token-count effect** (24 → 32 changed the per-domain hallucination rankings), or
2. **sampling variance** — these are *fresh stochastic draws* at T=1.0; the 0.770 and 0.613 are two
   different random runs, so part of the 0.16 drop may be pure run-to-run noise.

There is no same-config re-sample control, so attribution is impossible here. **Either reading is
humbling and important:** if (1), the cliff agreement is sensitive to a minor decoding knob; if (2),
the per-domain cliff Spearman has large run-to-run noise and the single-run 0.77 point estimate was
overconfident. **Both mean the cross-family/vendor agreement numbers need variance bars we do not yet
have.** → BANKED next run: re-sample the same 32-token config a second time (fresh seed) to measure
pure run-to-run Spearman spread, isolating noise from the token effect.

## What honestly survives

- Cross-vendor hallucination-cliff agreement is **real and moderate (~0.5)**, and at matched apparatus
  only **modestly below** within-open (~0.6) — NOT a dramatic outlier. Different providers do share
  *where they hallucinate* to a moderate degree.
- The representation > mechanism ordering weakly persists (matched all-6 hallucination ~0.56 > refusal
  ~0.38), consistent with the program's asymmetry, but the margin is small and noise-sensitive.

## Corrections owed (the discipline catching itself, again)

1. The **4-provider matrix** framing ("open cluster tight, OpenAI outlier") is apparatus-inflated;
   `provider_cliff_matrix_result.json` is now regenerated at the matched 32-token apparatus (loose web).
2. **Public thread:** we tweeted "different vendors fail on the same topics; judge-matched 0.23→0.47;
   partial." The fuller, generation-matched picture is more humbling — the cross-vendor *gap* mostly
   closes and the numbers are noise-sensitive. An honest follow-up is owed (draft prepared).

## Honest bounds

- Single runs per config; **no variance bars yet** (the headline lesson). Small open models (1.5–3B);
  gpt-4o-mini is larger, so any open-vs-gpt difference is also a **scale** confound, not purely vendor.
- TruthfulQA only — the **dataset** confound (adversarial-by-design) is separate and tested by the
  MMLU run (`PREREG_mmlu_dataset_confound`, in flight). In-silico; no consciousness claim.

## Receipts

- Prereg `PREREG_genmatch_xvendor_2026_06_23.md` (`b030c10`). Runner `run_genmatch_cliff.py`.
- `genmatch_xvendor_result.json`; gates `crossfamily_gate_{family}_gm32.json`.
- Matched matrix `provider_cliff_matrix_result.json` (`analyze_provider_matrix.py`).
