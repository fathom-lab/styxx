# PREREG: cross-family agreement with the bare template dropped, five of the June six models

Fathom Lab, lead-written, 2026-11-10. Sealed by hash in the private lab repository before the run it describes.
It follows `PREREG_start_token_artifact_2026_11_10.md` (sealed at lab commit `62d7ae3`, result at `f1e8c26`,
corrected at `fb29ef7`), whose nine predictions all passed. Two independent reviewers attacked a first version of
this file, its extractor and its scorer before the seal. Their faults are applied here.

## 1. where this stands

The first test showed, on fresh words, that the lab's 8-template pipeline lets one template (the bare word) set
the geometry of Qwen models: a single-token word alone at position 0 carries a state 160 times (Qwen2.5-1.5B) and
80 times (Qwen2.5-0.5B) larger than its other seven states. With that template dropped, the agreement of
Qwen2.5-1.5B with Llama-3.2-1B went from 0.0794 to 0.9280 on fresh words.

That was one of the June result's 13 cross-family pairs. This file fixes, before the data, what the lab expects
for the other pairs it can run: the five of the June six models that are in the local cache. gemma-2-2b-it is not
in the cache and is left out. So 8 of the 13 cross-family pairs are covered, and 5 are not, among them the two
Llama with gemma pairs that carried the June mean.

## 2. what has and has not been seen

Seen, all of it bearing on the lines below:

- Everything in the first test's result, for Qwen2.5-1.5B, Llama-3.2-1B and Qwen2.5-0.5B, both word sets, per
  template. For the pair Qwen2.5-1.5B with Llama-3.2-1B on fresh words: RSA 0.928 with seven templates, a
  within-category null of 0.665, an excess of 0.263; on original words 0.926, 0.672 and 0.255. Qwen2.5-0.5B with
  Llama-3.2-1B: 0.916 on fresh words. The two Qwen models with each other: 0.923.
- The stored June 8-template means of all six models on the ORIGINAL words, and what the exploratory look found in
  them: top-direction share 0.9532 (Qwen2.5-3B) and 0.9570 (Phi-3.5-mini), AUC 1.0 against "alone at position 0"
  (50 and 49 of 96 words), row norms 74 to 498 and 168 to 770; Llama-3.2-3B 0.0821.
- The closest preview of the G lines: with each model's top direction removed from those stored means, all 13
  cross-family pairs on the original words were between 0.66 and 0.91, mean 0.7304, with a within-category null of
  0.524 and an excess of 0.206.
- The June result file's aggregate values, aggregates computed over all 13 of its pair values, and five single
  pair values: Qwen with Qwen 0.999, Llama with Llama 0.964, Llama-1B with gemma 0.926, Qwen-1.5B with Llama-1B
  0.075, Qwen-1.5B with Phi 0.180.

Not computed or opened by the lead, or by any process the lead started: any per-template state of Qwen2.5-3B,
Llama-3.2-3B or Phi-3.5-mini; any state of those three models on the FRESH words; any seven-template value for any
pair that includes one of them; any leave-one-template-out value for any pair.

Seen during code checks before this seal:

- On ORIGINAL words, for the three models of the first test, with the first version of the extractor: the new
  path reproduced the first test's saved states (768 of 768 states within 1e-3 for each model; largest relative
  error 8.25e-06).
- A smoke run of the final extractor on all six models: its pad check on ORIGINAL words (one number per model, the
  largest relative difference between a padded row and the same input run unpadded), then two words that are in
  neither set, two templates, shapes only. It printed no norm, no variance share and no similarity.
- The scorer was run on random synthetic arrays in five made-up scenarios to check that every branch runs.

## 3. the run

`extract_v2.py`: CPU only, float32 arithmetic, offline, below-normal priority, 6 threads, one model per process,
into a new empty folder. Six models: the three of the first test again (as a control of the new code path) and
Qwen2.5-3B, Llama-3.2-3B, Phi-3.5-mini at the revisions pinned in the script. Both word sets, all 8 templates.

The three larger models do not fit in the memory this box has free. The script therefore never holds a model's
decoder in memory: it runs the first `fl` layers through one layer object and copies each layer's weights in from
the cached file before that layer's call, upcast exactly to float32. Inputs are batched per template and padded on
the right. The state read is the same one as before: the residual stream after
`fl = max(1, int(0.66 * n_layers))` layers.

The script refuses, and writes the refusal to a log that is never overwritten, if: free memory is below the
model's stated need; the decoder layer holds anything but parameters; the config mixes layer types; a layer's
tensors in the weights file are not exactly the layer's parameters; the word order differs from the token-count
file; or its pad check on original words differs by more than 1e-4.

A model refused for memory may be retried at most twice, before the scorer runs and never after. `score_v2.py`
is run once, after every model has either its states or a refusal log.

## 4. definitions

As in the first prereg, section 5: alone, single, norm ratio, rep8 (mean of the 8 templates, the published
convention), rep7 (mean of the 7 templates other than the bare word), share, AUC, RSA.

Within-category shuffle null of a pair: the RSA after permuting one model's words inside each of the 8
categories, 200 draws, seed 0; the null is the mean over draws. Excess is the observed RSA minus that mean. It is
an index on the r scale. It is not a share of anything.

## 5. the lines, scored by `score_v2.py` with the thresholds written in it

The FRESH set is confirmatory. The ORIGINAL set is scored by the same code and reported beside it.

Controls. If ANY control fails, the scorer prints the control lines and stops. Nothing else is computed. A rerun
then needs an amendment sealed first, which names the change and is justified by control values only.

| id | model | passes if |
|---|---|---|
| C3 | every model with states | exactly one OK run log, with the pad check at most 1e-4 |
| C1 | Qwen2.5-0.5B, Llama-3.2-1B, Qwen2.5-1.5B, each set | all 768 states within 1e-4 relative error of the first test's saved state, and token lengths equal |
| C2a | Qwen2.5-3B, Llama-3.2-3B, Phi-3.5-mini, original set | rows not alone at position 0: rep8 within 0.01 relative error of the stored June row, at most 2 rows missing |
| C2b | Qwen2.5-3B, Phi-3.5-mini, original set | rows alone at position 0: the difference between new rep8 and the June row is at most 0.05 of that row's seven-template part, at most 2 rows missing |

C2b exists because the June arrays are 8-template means, and in a row that was alone at position 0 the bare state
is so large that a wrong computation in the other seven templates would barely move the mean. C2b measures the
difference against the seven-template part directly. Phi-3.5-mini has no control of its code class other than C2
and C3; that is stated as a limit in section 7.

The artifact in the larger models.

| id | model | prediction | passes if |
|---|---|---|---|
| A1 | Qwen2.5-3B, Phi-3.5-mini | the bare template at position 0 is very large | median norm ratio over alone words at least 10 |
| A2 | Qwen2.5-3B, Phi-3.5-mini | the dominant direction is "alone" on fresh words | rep8 share at least 0.85 and AUC at least 0.99 |
| A3 | Qwen2.5-3B, Phi-3.5-mini | dropping the bare template removes it | rep7 share at most 0.30 |
| A4 | Llama-3.2-3B | the control with a start token shows none of it | start token added, norm ratio at most 3, RSA between its own rep8 and rep7 at least 0.95 |

Agreement with the bare template dropped, the 8 cross-family pairs among the five models.

| id | pairs | prediction | passes if |
|---|---|---|---|
| G1 | the 4 Qwen with Llama pairs | high agreement | each pair RSA(rep7) at least 0.85 |
| G2 | the 4 pairs with Phi | agreement, stated with less confidence | each pair RSA(rep7) at least 0.70 |
| G3 | all 8 | the number with the bare template dropped | mean RSA(rep7) at least 0.80 |
| G4 | all 8 | agreement beyond category labels | each pair exceeds its within-category shuffle null by at least 0.15 on the r scale |
| G5 | the 10 pairs among the five models | the June file is re-derived under the published pipeline | partial-lexical RSA of rep8 within 0.02 of the June value, each pair |
| G6 | all 8 | no single kept template carries the number | dropping any one of the seven kept templates, in both models, moves the pair's RSA by at most 0.05 |

Which lines carry risk:

- Already seen, so not new information: A2 on the ORIGINAL set (that statistic on those arrays is in section 2);
  G1, G4 and G5 for the pair Qwen2.5-1.5B with Llama-3.2-1B; C1 on the original set.
- Near certain on seen evidence, and their bars are set low for that reason: A1 (the stored row norms imply ratios
  near 45 and 32), A4, C1 on the fresh set, C3.
- Risk-bearing: C2a and C2b; A2 on FRESH words; A3; G1 for the three unseen Qwen with Llama pairs; G2 for all four
  Phi pairs, the least certain lines here; G3; G4 for seven pairs; G5 for nine June values; G6 for all eight pairs.

The result counts the risk-bearing fresh lines apart from the rest.

Rules fixed now:

- A2 is NOT TESTABLE on a set where fewer than 10 words fall in either class.
- A model with a memory refusal log and no states is NOT RUN. Its lines are not passes. A model whose states are
  missing for any other reason is MISSING, the controls fail, and the scorer stops.
- If the four Phi pairs average more than 0.10 below the four Qwen with Llama pairs, the result prints this
  sentence unchanged: "Phi remains lower with the bare template dropped; cause not established."
- Reported without a pass mark: partial-lexical and Spearman versions of every pair; the same-family pairs; the
  pairs with Qwen2.5-0.5B; each pair's RSA restricted to the words that are a single token in both models, with
  its own within-category excess; the excess divided by the standard deviation of the null; and the RSA over
  word pairs of the same category only.
- The mean of G3 is printed whatever the outcome, with the lowest pair, the null and every failed line beside it.

## 6. what each failure would mean

- A control fails: the new code path or the stored arrays are not what this file assumes. Nothing else is read.
- A1 or A3 fails for a model: the artifact claim does not extend to that model as written.
- G1 fails for a pair: the agreement is not uniformly high between Qwen and Llama. The pair and its value are
  printed.
- G2 fails for a pair: that Phi pair is below 0.70 with the bare template dropped. The cause is not established
  by this test.
- G4 fails for a pair: for that pair the excess over category labels is below 0.15 on the r scale. That is all
  it says.
- G5 fails for a pair: the June result file is not what these models and this pipeline produce for that pair.
- G6 fails for a pair: one of the seven kept templates carries a visible part of that pair's number, and the
  seven-template mean is not yet a clean measure for it.

## 7. what a pass would and would not mean

If G1 to G6 pass, the lab has a number to print: the mean cross-family agreement over 8 pairs among five small
language models, on 96 fresh words, with the bare template dropped, beside its within-category null.

It would not mean:

- a six-model number. gemma-2-2b-it was not run. 5 of the June 13 cross-family pairs are missing.
- that the agreement is fine-grained. The null is expected near 0.67. The excess is an index of agreement beyond
  category labels, not a share.
- that seven templates are free of every artifact. Only the one this lab found is removed. In all seven kept
  templates the word comes last, and four of them end in "a" before the word, which is an ungrammatical frame for
  words that begin with a vowel, plurals and mass nouns, in every model alike. A shared frame effect could raise
  agreement between models. G6 checks one part of that and no more.
- that Phi-3.5-mini was computed correctly beyond what C2 and C3 can see. Its model class has no per-template
  control.
- anything new about language models. Agreement between language models on category-structured word sets is
  widely reported, and the phenomenon behind the artifact is published (Sun, Chen, Kolter, Liu, arXiv
  2402.17762; Timkey and van Schijndel, arXiv 2109.04404).
- anything outside this box. One hand wrote the claim, the code and the scoring. Outside reruns: 0.

## 8. files sealed with this one

`extract_v2.py`, `score_v2.py`, and the kit files they read (`kit/reproduce.py`, `kit/contextual_reps.npz`,
`kit/lex_tokens.json`, `kit/real_convergence_confirm_result.json`), with the sha256 of the first test's saved
states, which C1 reads, and of the smoke logs. Their sha256 values are in the seal file committed to the lab
repository before the run starts.
