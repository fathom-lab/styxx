# RESULT: cross-family agreement with the bare template dropped, five of the June six models

Fathom Lab, lead-run, 2026-11-10. Scored against `PREREG_corrected_geometry_five_models_2026_11_10.md`, sealed in
the private lab repository at commit `bbbd61f` (15:56:04 UTC), with AMENDMENT 1 sealed at `04a6136` (16:06:27 UTC).
The sealed scorer ran at 16:13:11 UTC. Output: `states2/score_v2.log`, `states2/score_v2_out.json`.

One hand wrote the claims, the code and the scoring. One machine. Outside reruns: 0.

## verdict

Every control passed (17 lines). Every prediction line passed: 42 of 42 on the fresh words, 32 of 32 on the
original words. Of the 42 fresh lines, 36 were named risk-bearing in the prereg; all 36 passed.

## the eight cross-family pairs, 96 fresh words

| pair | published pipeline, 8 templates | bare template dropped | within-category null | excess |
|---|---|---|---|---|
| Qwen2.5-1.5B with Llama-3.2-1B | 0.0794 | 0.9280 | 0.6654 | 0.2626 |
| Qwen2.5-1.5B with Llama-3.2-3B | 0.1049 | 0.9117 | 0.6464 | 0.2653 |
| Qwen2.5-3B with Llama-3.2-1B | 0.0924 | 0.9197 | 0.6497 | 0.2701 |
| Qwen2.5-3B with Llama-3.2-3B | 0.1184 | 0.9139 | 0.6341 | 0.2798 |
| Llama-3.2-1B with Phi-3.5-mini | 0.1665 | 0.9205 | 0.6893 | 0.2312 |
| Llama-3.2-3B with Phi-3.5-mini | 0.1866 | 0.9228 | 0.6781 | 0.2447 |
| Qwen2.5-1.5B with Phi-3.5-mini | 0.1817 | 0.8899 | 0.6650 | 0.2249 |
| Qwen2.5-3B with Phi-3.5-mini | 0.1816 | 0.8762 | 0.6466 | 0.2296 |
| **mean of the eight** | **0.1389** | **0.9103** | | |

Median 0.9168, lowest pair 0.8762. The four Phi pairs average 0.9023 and the four Qwen with Llama pairs 0.9183, a
gap of 0.016, so the fixed sentence about Phi is not required.

## the other lines

- **The artifact extends to the larger models.** Bare-template norm against the other seven: 43.4 times for
  Qwen2.5-3B (40 words alone at position 0) and 29.1 times for Phi-3.5-mini (39 words). Top-direction share with 8
  templates 0.9511 and 0.9554, AUC 1.0 against "alone" for both; with the bare template dropped 0.0773 and 0.0802.
- **The control with a start token shows none of it.** Llama-3.2-3B: ratio 0.91 times, own geometry with 8
  against 7 templates 0.9992.
- **The June result file is exactly what the published pipeline produces.** All 10 June pair values among these
  five models were re-derived, the largest difference 0.0004.
- **No single kept template carries the number.** Dropping any one of the seven moved a pair by at most 0.0051.
- Reported without a bar: over word pairs of the same category only, the eight pairs correlate between 0.79 and
  0.90. That figure also reflects how tightly each category clusters, so it is not a pure measure of fine detail.

## what did not go to plan, and how it was handled

In the sealed run two models produced no states: Qwen2.5-3B crashed on an allocation and Llama-3.2-3B died with a
segmentation fault, while the machine's commit charge was near its limit. The sealed scorer stopped at the controls
and computed no prediction line. AMENDMENT 1 was sealed before anything else was done. It allowed a rerun of those
two extractions with the unchanged script. Qwen2.5-3B failed once more with a segmentation fault and succeeded on
its second allowed attempt; Llama-3.2-3B succeeded on its first. The scorer then ran once.

## limits

- gemma-2-2b-it was not run. 5 of the June 13 cross-family pairs are missing.
- The excess over the within-category null is an index on the r scale, between 0.22 and 0.28 here. It is not a
  share. Most of the agreement is reproduced by category structure alone.
- Only the artifact this lab found is removed. In all seven kept templates the word comes last.
- Phi-3.5-mini has no per-template control of its code class. Its 8-template means matched the June arrays on 96
  of 96 rows, and its seven-template part on 49 of 49 rows that were alone at position 0.
- High agreement between language models on category-structured word sets is widely reported. The phenomena
  behind the artifact are published by others. This lab claims no new phenomenon.

## correction, 2026-11-10

This file first gave the largest difference between a June pair value and its re-derived value as 0.0003. It is
0.00036 (Llama-3.2-3B with Phi-3.5-mini: 0.18636 against the June file's 0.186), which is 0.0004 at four decimals.
The line above is corrected. No other number in this file changed.
