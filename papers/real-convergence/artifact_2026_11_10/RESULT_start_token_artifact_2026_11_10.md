# RESULT: the lab's dominant direction is a start-of-input artifact

Fathom Lab, lead-run, 2026-11-10. Scored against `PREREG_start_token_artifact_2026_11_10.md`, sealed in the
private lab repository at commit `62d7ae3` (pushed 2026-11-10T14:22Z, before the run). Extraction finished
2026-11-10T14:32:35Z. Scoring is `score_artifact.py` as sealed, run once, straight after extraction.
Outputs: `states/score_out.json` (sha256 `d20b5d1e...1f31e`), `states/score.log` (sha256 `e974e338...5e002`).

One hand wrote the claim, the code and the scoring. One box. Outside reruns: 0.

## verdict

Every sealed line passed. The claim M stands for the three models run.

| id | set | model | sealed bar | got | verdict |
|---|---|---|---|---|---|
| P7 | original | Qwen2.5-1.5B | 90 of 96 rows within 0.05 of the stored array | 96 of 96 (median 0.0010, max 0.0016) | PASS |
| P7 | original | Llama-3.2-1B | same | 96 of 96 (median 0.0005, max 0.0007) | PASS |
| P1a | fresh | Qwen2.5-1.5B | bare-template norm at least 20x the other templates | 159.7x over 40 words alone at position 0 | PASS |
| P1b | fresh | Llama-3.2-1B | start token added, ratio at most 3x | 0.93x over 40 single-token words | PASS |
| P2 | fresh | Qwen2.5-1.5B | top direction share at least 0.90, AUC at least 0.99 | share 0.9969, AUC 1.0000, 40 of 96 words alone | PASS |
| P3 | fresh | Qwen2.5-1.5B | bare template dropped: share at most 0.30 | 0.0894 | PASS |
| P4 | fresh | Qwen2.5-1.5B x Llama-3.2-1B | RSA rises by at least 0.30 and reaches 0.50 | 0.0794 to 0.9280, gain +0.8486 | PASS |
| P5 | fresh | Llama-3.2-1B | own geometry unchanged, RSA at least 0.95 | 0.9989 | PASS |
| P8 | fresh | Qwen2.5-1.5B x Llama-3.2-1B | June pair value re-derived within 0.02 | June 0.075, re-derived 0.0747 | PASS |
| P6 | fresh | Qwen2.5-0.5B (never stored) | ratio 20x, share 0.90, AUC 0.99, then share at most 0.30 | 79.9x, 0.9885, 1.0000, then 0.1038; 40 of 96 alone | PASS |

The original word set, scored by the same code, passed every line as well (P4 there: 0.1427 to 0.9261).

## what the numbers say

1. The run reproduces the lab's stored June arrays to within 0.16 percent per row for both models. The stored
   arrays are what these models produce.
2. In Qwen2.5-1.5B and Qwen2.5-0.5B, a single-token word given as the bare template sits alone at position 0,
   and its state there is 160 and 80 times larger than its states in the other seven templates. Llama adds a
   start token and shows no such thing (0.93x).
3. That one template sets the published geometry of the Qwen models. With it, one direction holds 99.7 percent
   of the across-word variance and splits the words perfectly by "was alone". Without it, the top direction
   holds 9 percent. The Qwen geometry with and without the template correlate at 0.088 on fresh words.
4. Dropping the bare template moves the cross-family agreement of this pair from 0.08 to 0.93 on fresh words.
   The second cross-family pair on the box (Qwen2.5-0.5B with Llama-3.2-1B, reported without a bar) moves
   from 0.084 to 0.916.
5. The same-family value of 0.9999 (Qwen-1.5B with Qwen-0.5B) under the published pipeline was the artifact
   itself: two models with one tokenizer making the same split. Without the template it is 0.923.

## the limits, which are part of the result

- Most of the corrected agreement is category-level. On fresh words the within-category shuffle null is 0.665
  (sd 0.009, 200 draws) against the observed 0.928. The excess over that null is 0.263. Under the published
  pipeline it was 0.051 over a null of 0.029.
- One of the June result's 13 cross-family pairs was run (Qwen2.5-1.5B with Llama-3.2-1B), plus one pair with a
  model outside the six (Qwen2.5-0.5B with Llama-3.2-1B). Qwen2.5-3B, Llama-3.2-3B and Phi-3.5-mini were not
  run (memory), and gemma-2-2b-it is not in the cache. No corrected six-model number exists.
  (Corrected 2026-11-10, after the first commit of this file, which said two of the 13 pairs were run.)
- The lexical partial barely moves the corrected value (0.9272 against 0.9280). Spearman gives 0.853.
- The phenomenon underneath is published by others: very large activations in the starting token (Sun, Chen,
  Kolter, Liu, arXiv 2402.17762), and a few dimensions dominating similarity measures (Timkey and van
  Schijndel, arXiv 2109.04404). Both abstracts were fetched today. The lab claims no new phenomenon.
- High agreement between language models on category-structured word sets is expected and widely reported.
  The size of the corrected number is not, by itself, news.

## what follows for the record

- The June headline (cross-family 0.258, median pair 0.166) measured a mixture: pairs free of the artifact near
  0.93, and pairs with a Qwen or Phi model held near 0.1 to 0.3 by the artifact. It is not a number about
  concept geometry and should be withdrawn with a dated correction.
- The June readings "Phi is an outlier" and "Qwen agreement is fragile" are readings of the artifact.
- Any later result that used the 8-template mean on a model without a start token inherits the question.
  That audit is running separately and is exploratory.

## what would settle the six-model question

A new sealed test: all five cached models plus gemma, bare template dropped, a bar written first for the
8 or 13 cross-family pairs and for the excess over the within-category null. It needs either the GPU or
layer-by-layer float32 on the CPU for the three larger models, and a 5 GB download for gemma. Those are the
operator's to start.
