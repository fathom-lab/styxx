# PREREG: is the lab's dominant direction a start-of-input artifact?

Fathom Lab, lead-written, 2026-11-10. Sealed by hash in the private lab repository before the run it describes.
One hand wrote the claim, the code and the scoring. No outside party has seen any of it.

## 1. where this comes from

The lab's June result "six small language models share concept geometry" (fathom-lab/styxx,
`papers/real-convergence/`, commit `a59db86e`) puts each concept word into 8 templates, takes the last-token
state at layer `int(0.66 * n_layers)`, and averages the 8. Template 1 is the bare word, `"{w}"`.

Today's replication kit and its skeptic found that one direction holds over 95 percent of the across-word
variance in three of the six models. The lead then looked at the stored arrays of the ORIGINAL 96 words
(`explore_pc1.py`, exploratory, output `explore_pc1_out.json`):

| model | top direction share | words alone at position 0 | AUC of top direction against "alone" | row norm, min to max |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.9971 | 50 of 96 | 1.0 | 62 to 1556 |
| Qwen2.5-3B | 0.9532 | 50 of 96 | 1.0 | 74 to 498 |
| Phi-3.5-mini | 0.9570 | 49 of 96 | 1.0 | 168 to 770 |
| Llama-3.2-1B | 0.0966 | 0 of 96 (start token added) | not defined | 6.5 to 8.0 |
| Llama-3.2-3B | 0.0821 | 0 of 96 (start token added) | not defined | 14.9 to 19.7 |
| gemma-2-2b | 0.0946 | tokenizer not cached | not computed | 249 to 317 |

"Alone at position 0" means the tokenised bare template is a one-token input. Removing each model's top
direction moved the raw cross-family mean from 0.313 to 0.730 over 13 pairs; the within-category shuffle
null moved from 0.152 to 0.524.

## 2. the claim M

In tokenizers that add no start token (Qwen2.5, Phi-3.5), the bare template makes a one-token input for every
single-token word. These models carry a very large activation at position 0 in the middle layers. The mean over
8 templates is then dominated by that one template. The dominant direction is "this word was alone at position
0". It is a property of the measuring pipeline. It is not concept geometry.

The phenomenon underneath is published and not the lab's: very large activations at the first position (Sun,
Chen, Kolter and Liu 2024) and a few dimensions dominating similarity measures (Timkey and van Schijndel 2021).
Both are cited from the lead's memory and were not re-fetched today. This file claims only that the lab's own
number carried the artifact.

## 3. what has and has not been seen

Seen before sealing: everything in section 1, all on the ORIGINAL words and all on the 8-template means.
Also seen: the confirm result file's aggregate values, and three of its pair values (Qwen-Qwen 0.999,
Llama-Llama 0.964, Llama-1B with gemma 0.926).

Not seen by anyone, because the June runs stored none of it: any single-template state for any model; any state
for the FRESH 96 words under the as-published pipeline; any state for Qwen2.5-0.5B-Instruct under this pipeline.
The lead has not opened `telepathy_reps.npz`, `normeq_reps.npz` or `atlas_live_reps.npz` in
`papers/mind-instrument/`; those hold arrays under a different convention (each template normalised before
averaging) and are not used here.

A smoke run was made before sealing: two words that are in neither set ("zebra", "notebook"), two templates,
shapes only, plus the self-check of section 4 on four original-set probes. It printed no norm, no variance
share and no similarity.

## 4. the run

`extract_small.py`, CPU only, float32, offline, below-normal priority, 6 threads, one model per process
(model C first; it carries the self-check and the driver stops if that fails). Three cached models:

- A = Qwen2.5-1.5B-Instruct, revision `989aa798...` (one of the six; no start token expected)
- B = Llama-3.2-1B-Instruct, revision `92131767...` (one of the six; start token expected; the control)
- C = Qwen2.5-0.5B-Instruct, the single cached snapshot (not one of the six; never stored by the lab)

Both word sets (ORIGINAL and FRESH, 96 words each, 8 categories each, 0 words shared), all 8 templates. The
state is the residual stream after `fl = max(1, int(0.66 * n_layers))` layers, captured by a hook; later layers
are dropped to save memory. A self-check on model C compares the hook, the truncated model and
`hidden_states[fl]` and stops the run if they differ by more than 1e-5 relative.

Left out, and why: Qwen2.5-3B, Llama-3.2-3B and Phi-3.5-mini need more memory than the box had free (7.0 GB
available at 2026-11-10 afternoon); GPU runs are the operator's to start; gemma-2-2b-it is not in the cache and
a download is the operator's act. A model is skipped with a REFUSED line if free memory is below its stated
need when its turn comes.

Numerics differ from June: the stored arrays were presumably made on the GPU in half precision; this run is
float32 on the CPU. P7 below is the check that this does not matter.

## 5. definitions

- alone(w): the tokenised bare template has length 1.
- single(w): the bare template has the smallest length seen for that model (1 without a start token, 2 with).
- norm ratio of a word: norm of its bare-template state divided by the median norm of its other 7 states.
- rep8: mean of the 8 template states (the lab's published convention). rep7: mean of the 7 templates other
  than the bare one.
- share: fraction of across-word variance on the first principal direction of the centred array.
- AUC: probability that a random alone word scores above a random other word on that direction, folded to be
  at least 0.5.
- RSA(x, y): Pearson r of the upper triangles of the two 96 by 96 distance matrices, distance as in the lab's
  code (centre across words, unit-normalise, sqrt(2 - 2 cos)).

## 6. predictions, scored by `score_artifact.py` with the thresholds written in it

The FRESH set is the confirmatory set. The ORIGINAL set is scored with the same code and reported beside it:
P2 on the original set was already seen; P1, P3, P4 and P5 were not seen on either set.

| id | model | prediction | passes if |
|---|---|---|---|
| P7 | A, B (original set) | the run reproduces the lab's stored arrays | at least 90 of 96 rows of rep8 within 0.05 relative error of the stored row |
| P1a | A | the bare template at position 0 is enormous | median norm ratio over alone words at least 20 |
| P1b | B | with a start token it is not | start token added, and median norm ratio over single-token words at most 3 |
| P2 | A | the dominant direction is "alone" on fresh words too | rep8 share at least 0.90 and AUC at least 0.99 |
| P3 | A | dropping the bare template removes it | rep7 share at most 0.30 |
| P4 | A x B | and cross-family agreement rises | RSA(rep7) minus RSA(rep8) at least 0.30, and RSA(rep7) at least 0.50 |
| P5 | B | the control model barely changes | RSA between its own rep8 and rep7 at least 0.95 |
| P8 | A x B (fresh set) | one June pair value is re-derived | partial-lexical RSA of rep8 within 0.02 of the June result file's value for this pair |
| P6 | C | secondary: the same artifact in a model the lab never stored | norm ratio at least 20, rep8 share at least 0.90, AUC at least 0.99, rep7 share at most 0.30 |

Rules fixed now:

- If P7 fails for a model, every line that uses that model is VOID. The cause (precision, revision or code) is
  then to be found before anything else is said.
- P2 and P6 are NOT TESTABLE on a set where fewer than 10 words fall in either class.
- A model that was REFUSED for memory is NOT RUN. Its lines are not passes.
- Reported without a pass mark: partial-lexical and Spearman versions of the A x B value, the within-category
  shuffle null (200 draws, seed 0) for rep8 and rep7, the pair C x B and the same-family pair A x C.

## 7. what kills M

On the fresh set: P1a fails (no large norm at position 0), or P3 fails (dropping the bare template does not
remove the dominant direction). Either one means M is wrong as written, and the lead says so in the same place
this file is cited.

P4 failing while P1a, P2 and P3 pass would mean the artifact is real but is not what held the cross-family
number down. That is a separate sentence and would be reported as such.

## 8. what a pass would and would not mean

A pass means: for the models run, the lab's published 8-template geometry mostly measured whether a word was
one token alone at the start of the input, and the June numbers for Qwen (and by the stored arrays, Phi) are
not concept geometry.

A pass does not mean:

- a corrected six-model number exists. That needs all six models re-extracted without the bare template, a
  bar written first, and the operator's go for the GPU and the gemma download.
- anything about Qwen2.5-3B or Phi-3.5-mini on fresh words. They were not run.
- that the agreement left after the correction is fine-grained. On the original words the within-category null
  rose with the corrected number; the excess over it was 0.206 against 0.162 before.
- anything outside this box. Outside reruns: 0. One hand wrote the claim, the code and the scoring.
- anything new about language models. The underlying phenomenon is published.

## 9. files sealed with this one

`extract_small.py`, `score_artifact.py`, and the kit files they read: `kit/reproduce.py`,
`kit/contextual_reps.npz`, `kit/lex_tokens.json`, `kit/real_convergence_confirm_result.json`. Their sha256
values are in the seal file committed to the lab repository before the real run starts.
