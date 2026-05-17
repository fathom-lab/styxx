# Cross-vendor refusal transport — CONFIRMATORY re-label run, 2026-05-17

## Verdict (one paragraph)

By the preregistered thresholds in `cross_vendor_refusal_transport_confirm.py`, the artifact-rescue hypothesis is **KILLED**: H_confirm required BOTH (a) Δceiling shrinks vs prior 0.071 AND (b) min Anthropic transported AUC ≥ 0.70. Condition (a) PASSES — Δceiling drops 0.071 → **0.059** with the vendor-fair label (judge_refusal/gpt-4.1 for Claude, lexical vendor-robust for OpenAI), confirming the prereg-predicted partial recovery in detector quality. Condition (b) FAILS — min Anthropic transported is **0.617** (claude-sonnet-4-5 × all-mpnet × corpus_2), still below the 0.70 floor. So we **withdraw the "transport holds" reading**: with a fair behavioral label the Claude ceiling moves measurably toward OpenAI, the Anthropic T/C ratio actually edges *above* OpenAI (0.884 vs 0.867), and min transported climbs from 0.542 → 0.617, but the worst cell still cracks the preregistered floor. The honest read is that the original cracking was **partly** a label artifact (the Claude ceiling really was depressed by OpenAI-tuned regex misses, ~1.2pt of the gap recovered) and **partly** real (the mpnet × corpus_2 cell remains the weakest cell for both vendors — a corpus/foreign-space weakness inherited by all targets, not a vendor failure). n=1 non-OpenAI vendor: this still says nothing about Gemini, open-weights, Mistral, DeepSeek, or reasoning models. `claude-opus-4-7` remains transparently excluded (empty completions, not retuned).

## Numbers

```
                  transported     ceiling      T/C ratio
                  mean   min      mean         mean
openai  prior     0.799  0.606    0.920        0.868
openai  re-label  0.797  0.604    0.918        0.867       ← unchanged (sanity)
anthrop prior     0.737  0.542    0.849        0.868
anthrop re-label  0.759  0.617    0.859        0.884       ← partial recovery

Δceiling (oai − ant)         prior 0.071  →  re-label 0.059   (shrunk, cond a PASS)
min Anthropic transported    prior 0.542  →  re-label 0.617   (still < 0.70, cond b FAIL)
```

OpenAI numbers move by ≤0.002 vs prior — the vendor-robust lexical labeler does not regress on OpenAI behavior, exactly as the labeler's offline regression check (60/60 agree on saved OpenAI labels) predicted.

Per-cell Anthropic transported AUCs (re-label):

| foreign            | corpus   | haiku-4-5 | sonnet-4-5 | opus-4-5 |
|--------------------|----------|-----------|------------|----------|
| te3-small          | corpus_1 | 0.773     | 0.836      | 0.809    |
| te3-small          | corpus_2 | 0.757     | 0.804      | 0.777    |
| all-mpnet          | corpus_1 | 0.748     | 0.823      | 0.821    |
| all-mpnet          | corpus_2 | 0.680     | **0.617**  | 0.667    |

Per-Anthropic-model refusal rates and judge↔lex agreement (on 75 prompts):

| model              | judge refuse | lex refuse | judge/lex agree |
|--------------------|--------------|------------|-----------------|
| claude-haiku-4-5   | 30/75 (40%)  | 24/75      | (judge catches 6 more)  |
| claude-sonnet-4-5  | 27/75 (36%)  | 20/75      | (judge catches 7 more)  |
| claude-opus-4-5    | 22/75 (29%)  | 19/75      | (judge catches 3 more)  |

So the judge does find Claude refusals the vendor-robust lexical labeler still misses — that is the source of the ceiling recovery — but it is not enough to push the worst transport cell above 0.70.

## What this means / what it does not mean

- **Means:** The prior reading "transport itself is vendor-agnostic at the same fractional quality" is *partially* supported (T/C ratio is now 0.884 on Anthropic vs 0.867 on OpenAI — essentially equal, very slightly better on Anthropic with the fair label). But the absolute floor is still pierced by one cell, so we cannot say the instrument transports to Anthropic at preregistered quality.
- **Does NOT mean:** "Universal across all AI." Still n=1 non-OpenAI vendor. Gemini / Mistral / DeepSeek / open-weights / Qwen / reasoning models all untested. A confirm here would not have meant universality either.
- **Does NOT mean:** The 0.70 floor is the right floor — it is just the floor we preregistered. We are not moving it post-hoc.
- **Worst cell is still corpus_2 × all-mpnet for BOTH vendors.** That points at corpus/space portability, not vendor identity, as the residual failure mode. A vendor-fair label cannot fix a foreign-space×corpus pairing that is genuinely thin on refusal signal.

## Limits (do not skip)

- **n=1 non-OpenAI vendor.** Still.
- **Judge is gpt-4.1 (OpenAI).** Using an OpenAI judge to label Claude introduces a different bias surface. We report judge↔lex deltas per Anthropic cell so this is visible (judge catches 3–7 extra refusals per model — all in the direction "lexical missed a Claude-style decline").
- **3 Claude models, one excluded.** `claude-opus-4-7` empty-completion failure was not investigated this session.
- **Axis anchor still asymmetric.** The 20 obvious prompts were OpenAI-styled. A symmetric replication (Claude-obvious anchors + reverse direction) is still not done. We are NOT running it here — that would be re-rolling for a better number against the same data.
- **No replication of the failing cell at higher resolution.** The worst number (0.617) is not suspiciously clean — it's a noisy edge between two known-thin choices (mpnet × corpus_2) that was already the worst cell for OpenAI in the prior stress run. No replication needed for honesty; replicating just to chase a pass would violate the integrity rule.

## Files

- script: `scripts/dogfood/cross_vendor_refusal_transport_confirm.py` (preregistration in module docstring, frozen before execution)
- raw output: `scripts/dogfood/out_cross_vendor_refusal_transport_confirm.json`
- run log: `scripts/dogfood/_cross_vendor_confirm_run.log`
- labeler: `scripts/dogfood/vendor_robust_refusal_label.py` (shared artifact, commit 4a69dc4)

## Bottom line for the project

The preregistered confirm verdict did not hold; the kill verdict is what we report. The interesting partial finding is that **the label artifact was real but only worth ~1.2 ceiling points** — it does not fully account for the prior cracking. The residual cracking is located in the foreign-space × corpus pairing, not in the vendor boundary. Status: validation, not a paper. No DOI deposit. Result is committed to `fathom-lab/styxx` `main` because a kill IS the finding.
