# Cross-vendor refusal transport (n=1 vendor: Anthropic) — honest record, 2026-05-17

## Verdict (one paragraph)

By the preregistered thresholds, **styxx-transported refusal scoring CRACKS at the cross-vendor boundary**: the worst Anthropic cell (Claude Sonnet 4.5 in all-mpnet × corpus_2) falls to AUC 0.542, below the 0.70 floor that defined "holds-degraded." However, the more diagnostic comparison — transported AUC divided by the same-prompt natural-axis ceiling — is **identical between vendors at 0.868 mean** (OpenAI 16 cells, Anthropic 12 cells). The Anthropic transported mean (0.737) is only 0.062 below the OpenAI mean (0.799), most of which is a ceiling drop (0.920 → 0.849), not a transport drop. So the honest summary is: **transport itself appears vendor-agnostic at the same fractional quality, but the absolute refusal-detector ceiling is lower on Claude — likely because the refusal axis was defined from 20 OpenAI-styled obvious prompts and Claude's refusal lexical/semantic surface drifts from that anchor.** This is n=1 non-OpenAI vendor; it does **not** prove "universal across all AI" — Gemini, open-weights, and reasoning models remain entirely untested, and one of four Anthropic models (claude-opus-4-7) returned empty content blocks for every prompt under our parameters and was excluded.

## What was run

Script: `scripts/dogfood/cross_vendor_refusal_transport.py` (preregistration in module docstring, frozen before execution).
Raw output: `scripts/dogfood/out_cross_vendor_refusal_transport.json`.

Single variable changed vs. validated `refusal_transport_stress.py`: target model vendor.

| held fixed                  | value                                                        |
|-----------------------------|--------------------------------------------------------------|
| home space                  | `text-embedding-3-large`                                     |
| foreign spaces              | `text-embedding-3-small`, `all-mpnet-base-v2`                |
| label-free corpora          | corpus_1 (generic descriptive), corpus_2 (period prose)      |
| axis definition             | 20 obvious eval prompts, OpenAI home embeddings              |
| eval prompts                | 75 (eval + aggressive borderline; same as 2026-05-17 stress) |
| ground truth                | lexical refusal regex on each model's live response          |
| transport method            | shipped `styxx.transport.Transport.fit(method="procrustes")` |

Targets:
- **OpenAI baseline**: `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4.1` (16 cells)
- **Anthropic test**: `claude-haiku-4-5`, `claude-sonnet-4-5`, `claude-opus-4-5` (12 cells)
- Excluded: `claude-opus-4-7` — returned 0 text blocks for all 75 prompts under `max_tokens=300, temperature=0` (likely reasoning-token starvation; not investigated further to respect the no-knob-tuning rule).

## Numbers

```
openai     transported  mean=0.799  min=0.606  max=0.904  ceiling mean=0.920
anthropic  transported  mean=0.737  min=0.542  max=0.824  ceiling mean=0.849
delta (anthropic − openai) transported mean = −0.062
transported / ceiling RATIO
  openai     mean=0.868  min=0.689
  anthropic  mean=0.868  min=0.643
```

Per-model refusal rates on the 75-prompt set (sanity — Claude is *more* refusal-prone, not less):

| vendor    | model              | refuse rate | ceiling | best T | worst T |
|-----------|--------------------|-------------|---------|--------|---------|
| openai    | gpt-4o-mini        | 0.21        | ≈0.92   | 0.883  | 0.663   |
| openai    | gpt-4.1-mini       | 0.23        | ≈0.91   | 0.876  | 0.689   |
| openai    | gpt-4o             | 0.20        | ≈0.96   | 0.904  | 0.717   |
| openai    | gpt-4.1            | 0.27        | ≈0.89   | 0.822  | 0.606   |
| anthropic | claude-haiku-4-5   | 0.32        | ≈0.82   | 0.743  | 0.700   |
| anthropic | claude-sonnet-4-5  | 0.27        | ≈0.83   | 0.778  | 0.542   |
| anthropic | claude-opus-4-5    | 0.21        | ≈0.90   | 0.824  | 0.765   |

## Where the boundary actually is

1. **The transport method works the same across vendors.** T/C ratio is 0.868 on both sides. There is no vendor-specific transport failure mode visible here.
2. **The instrument quality drops on Claude.** Even the same-space *ceiling* (natural axis fit in the foreign space, scored against Claude's own refusals) is 0.07 lower than on OpenAI. The refusal-axis anchor — 20 obvious OpenAI-styled prompts — is a slightly worse refusal probe for Anthropic's behavior.
3. **The worst cell is corpus-driven, not vendor-driven.** `all-mpnet × corpus_2` is the worst cell for both vendors (this was already a known weak corpus from the 2026-05-17 stress run). Anthropic just inherits and slightly amplifies that weakness.
4. **One opus model dropped out.** `claude-opus-4-7` produced empty completions for all 75 prompts under default parameters. We did not retune; an honest record says "excluded, parameter sensitivity not investigated this session."

## Blast radius / limits (do not skip)

- **n=1 non-OpenAI vendor.** This is not evidence about Gemini, Mistral, DeepSeek, Llama, etc. The claim "universal cognometric transport across vendors" remains unsupported in general.
- **Refusal detector is lexical and was developed against OpenAI phrasing.** Lower Claude ceiling could reflect Claude using refusal phrasings the regex misses, not weaker refusal geometry. We report ceiling explicitly so the reader can see this.
- **3 Claude models is small.** Especially with one parameter-sensitivity dropout.
- **Axis anchor is asymmetric.** The 20 obvious prompts were validated as obvious for OpenAI behavior. A symmetric replication would re-fit the axis on Claude-obvious prompts and check the reverse direction; not done here.

## Bottom line for the project

The preregistered "SURVIVES" threshold did not hold. The preregistered "CRACKS" verdict is what we report. But the more interesting finding is that **the cracking is not located in the transport** — it's located in the refusal-axis anchor's portability to a different vendor's refusal phrasing. That is a less spectacular failure than the headline AUC suggests, and a more actionable one (a vendor-balanced anchor set is the obvious next experiment), but we are explicitly **not** running that experiment now: it would be re-rolling for a better number against the same data.

Status: validation, not a paper. No DOI deposit. Result is committed to `fathom-lab/styxx` `main` whether favorable or not.
