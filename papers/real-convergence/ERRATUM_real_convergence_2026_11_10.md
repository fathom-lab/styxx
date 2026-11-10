# ERRATUM: the June 2026 "shared concept geometry" result

Fathom Lab, 2026-11-10. This corrects `RESULT_real_convergence_2026_06_03.md` in this folder. The June file is
left as it was, apart from one dated line under its title. One lab, one machine. Outside reruns: 0.

## what the June file said

Six small instruct models share concept geometry: cross-family RSA 0.258 after a lexical partial, median pair
0.166, shuffled control -0.002 +- 0.011, "confirmed out of sample" on 96 fresh words against a gate of 0.25.

## what is withdrawn

**The number 0.258 as a statement about concept geometry, and the words "pre-registered" and "confirmed".**
Four findings, each re-run by the lead on 2026-11-10:

1. **It cannot be re-derived from committed files.** The June confirm run saved its result file and no
   representations.
2. **Git cannot date the gate before the data.** Every file of the result entered the repository in one unsigned
   commit, `a59db86e`. The confirm gate is written only in the runner, after an exploratory run had returned 0.304.
3. **Two pairs carried the mean.** In the June result file, 2 of 13 cross-family pairs are at or above 0.25.
   Without the two Llama with gemma pairs the mean is 0.137. With any one of three models dropped, the mean falls
   below the gate (3 of 6 drop-one checks).
4. **The pipeline carried an artifact.** Each word went into 8 templates, and the first was the bare word. Qwen and
   Phi tokenizers add no start token, so a single-token word sat alone at position 0. Those models carry a very
   large activation there. The 8-template mean of such a word was that one state and little else.

Also withdrawn, as readings of the artifact: "Phi is an outlier", "Qwen agreement is fragile", and the same-family
Qwen value of 0.999.

## the artifact, tested

The prediction file, the extraction script and the scoring script were committed to the lab's private repository
before the run (2026-11-10, 14:22 UTC). An outsider cannot check that date, and the files are published here so
that the claim can at least be re-run. The run used the processor only, float32, three cached models. All nine
predictions passed: eight on the fresh words, and P7 (the stored arrays reproduced) on the original words.

| on 96 fresh words, unless marked | value |
|---|---|
| Qwen2.5-1.5B, bare-template norm against its other seven templates, 40 words alone at position 0 | 159.7 times |
| Llama-3.2-1B, the same ratio (its tokenizer adds a start token) | 0.93 times |
| Qwen2.5-1.5B, share of across-word variance on its top direction, 8 templates | 0.9969 |
| the same, bare template dropped | 0.0894 |
| that direction against "alone at position 0", AUC | 1.0000 |
| Qwen2.5-1.5B with Llama-3.2-1B, raw RSA, 8 templates | 0.0794 |
| the same pair, bare template dropped | 0.9280 |
| the June file's value for that pair, 0.075 (RSA after a lexical partial), re-derived | 0.0747 |
| original words (P7): re-extracted rows within 0.16 percent of the stored June arrays, both models | 96 of 96 |

## five of the six models, bare template dropped

A second test, sealed the same way before it ran (2026-11-10, 15:56 UTC, with one amendment sealed at 16:06 UTC),
covered the five of the six models still in the lab's cache. An outsider cannot check those dates either. Every
control passed (17 lines) and every prediction passed (42 of 42 on fresh words).

| pair, 96 fresh words, raw RSA | published pipeline, 8 templates | bare template dropped | within-category null | excess |
|---|---|---|---|---|
| Qwen2.5-1.5B with Llama-3.2-1B | 0.0794 | 0.9280 | 0.6654 | 0.2626 |
| Qwen2.5-1.5B with Llama-3.2-3B | 0.1049 | 0.9117 | 0.6464 | 0.2653 |
| Qwen2.5-3B with Llama-3.2-1B | 0.0924 | 0.9197 | 0.6497 | 0.2701 |
| Qwen2.5-3B with Llama-3.2-3B | 0.1184 | 0.9139 | 0.6341 | 0.2798 |
| Llama-3.2-1B with Phi-3.5-mini | 0.1665 | 0.9205 | 0.6893 | 0.2312 |
| Llama-3.2-3B with Phi-3.5-mini | 0.1866 | 0.9228 | 0.6781 | 0.2447 |
| Qwen2.5-1.5B with Phi-3.5-mini | 0.1817 | 0.8899 | 0.6650 | 0.2249 |
| Qwen2.5-3B with Phi-3.5-mini | 0.1816 | 0.8762 | 0.6466 | 0.2296 |
| mean of the eight | 0.1389 | 0.9103 | | |

The June file used RSA after a lexical partial. On that measure the means of the eight are 0.1358 and 0.9097.

- The artifact is in the larger models too: bare-template norm 43.4 times the other templates for Qwen2.5-3B and
  29.1 times for Phi-3.5-mini; top-direction share 0.9511 and 0.9554, falling to 0.0773 and 0.0802 with the bare
  template dropped. Llama-3.2-3B, whose tokenizer adds a start token, shows none of it (0.91 times).
- Phi is not lower than the others once the bare template is dropped: its four pairs average 0.9023 against 0.9183.
- All 10 June pair values among these five models were re-derived under the published pipeline, the largest
  difference 0.0004 (the June file gives three decimals). For those 10 pairs the June arithmetic was faithful. The
  pipeline was wrong.
- Dropping any one of the seven kept templates moved a pair by at most 0.0051.
- In the sealed run two models produced no states, while the machine's commit charge was near its limit: the
  allocator refused Qwen2.5-3B an allocation, and Llama-3.2-3B died with a segmentation fault. The sealed scorer
  stopped at its controls and computed no prediction line. An amendment allowing a rerun of those two extractions,
  with the script unchanged, was sealed before the rerun. The files are in `artifact_2026_11_10/`.

## what stands

- For five of the six models, the stored June arrays are what the models produce: Qwen2.5-1.5B and Llama-3.2-1B
  re-extracted in test 1 (P7), Qwen2.5-3B, Llama-3.2-3B and Phi-3.5-mini in test 2's controls C2a and C2b.
  gemma-2-2b-it was not checked.
- With the bare template dropped, the eight cross-family pairs among five models agree at 0.88 to 0.93 raw RSA on
  fresh words, mean 0.9103 (0.9097 after the lexical partial).
- Most of that agreement is at the level of category. For Qwen2.5-1.5B with Llama-3.2-1B on fresh words, shuffling
  words inside their own category still gives 0.665. The excess over that null is 0.263 on the r scale.

## limits

- gemma-2-2b-it was not re-run. It is no longer in the lab's cache.
- Only the artifact this lab found is removed. In all seven kept templates the word comes last.
- Four of the seven kept templates put "a" before the word, a frame that does not fit words beginning with a vowel,
  plurals or mass nouns, in every model alike. A frame shared by every model could raise the agreement. Dropping one
  template at a time (a pair moved by at most 0.0051) checks part of that and no more.
- The package's geometry axis (`styxx/mind.py`) uses the same 8 templates and takes the June arrays as its anchors.
  Checked on 2026-11-10 on stored arrays, outside the sealed tests (`artifact_2026_11_10/mind_axis_check.py`): in
  three of its six anchors (Qwen2.5-1.5B, Qwen2.5-3B, Phi-3.5-mini) one direction holds 0.95 to 0.997 of the variance
  across the 96 battery concepts, against 0.08 to 0.10 in the other three. Scored as documented, Qwen2.5-0.5B gets a
  citizenship of 0.1833. With the bare template dropped from it and from the five anchors that could be rebuilt, it
  gets 0.904. The function is not changed by this correction.
- The underlying phenomena are published by others: very large activations in the starting token (Sun, Chen,
  Kolter, Liu, arXiv 2402.17762) and a few dimensions dominating similarity measures (Timkey and van Schijndel,
  arXiv 2109.04404). This lab found them in its own pipeline. It claims no new phenomenon.
- Agreement between language models on category-structured word sets is widely reported.
- One hand wrote the claims, the code and the scoring.

## the scale result in this folder

`RESULT_scale_law_2026_06_03.md` was built on the same function (`concept_all_layers`), bare template included, on
three ladders of models: Pythia, GPT-2 and Qwen2.5. Checked on 2026-11-10 with the tokenizers alone, outside the
sealed tests: the Pythia and GPT-2 tokenizers add no start token either, and 68 and 74 of that file's 192 words are
a single token alone at position 0 (gpt2-xl was not checked; it is no longer in the lab's cache). Whether those
models carry the large activation there has not been tested. None of that file's numbers has been re-tested with the
bare template dropped. That covers its pooled rank correlation of -0.573 and its reading that a mid-layer read-out
gives concept geometry about five times too low. The file now carries one dated caution line under its title.

## other documents that cite the June result

These state the June number or its "confirmed" reading and are not edited by this correction. Two of them carry
certificates that bind their bytes. Line numbers are those at commit `76a397c3`. The row for this folder in
`papers/INDEX.md` is corrected with this erratum.

Other arcs of this lab read hidden states from the same model families with their own extraction code. One was
read on 2026-11-10: the banks that `papers/disjoint-worlds/run_b31v2.py` writes come from eight full-sentence
templates, last token (`run_thought_transfer.extract`), so no word is alone at position 0 there. The other arcs have
not been checked for a content token at position 0 of a tokenizer that adds no start token.

- `papers/ai-human-alignment/SYNTHESIS_geometry_of_meaning_2026_06_03.md`, lines 11, 25 and 62 ("SURVIVED").
- `papers/ai-human-alignment/PREREG_ai_human_2026_06_03.md`, line 4 ("confirmed, controlled, out-of-sample").
- `papers/AUDIT_the_whole_program_2026_09_01.md`, line 232: the sentence the README carried until this correction.
- `papers/ancient-question-program/CAPSTONE_universal_mind_2026_06_10.md`, lines 26 to 28 (certificate-bound).
- `papers/mind-instrument/FINDING_anatomy_v2_2026_06_10.md`, line 55 (certificate-bound). It already read 0.258 as
  a lower bound depressed by norm domination.
- `papers/mind-instrument/PREREG_mind_v0_2026_06_10.md`, line 22 ("pre-registered out-of-sample").
- `styxx/mind.py`, the comment above the anchor set (line 199): "confirmed out-of-sample".

## re-running it

The artifact itself, with no lab file and no licence to accept. One file, ungated models (Qwen2.5-0.5B by default,
about 1 GB of download, a few minutes on a processor):

```
python papers/real-convergence/artifact_2026_11_10/rerun_ungated.py
```

It prints your numbers beside the lab's and says whether each sealed bar is met on your machine. `--more` adds
Qwen2.5-1.5B and Phi-3.5-mini, `--cpu` keeps it off a GPU. It was written after the two sealed tests and is not part
of them. The lab has run it on Windows only.

Numbers on the stored June arrays, no model needed, about one second:

```
python papers/real-convergence/reproduce.py
```

The two sealed tests themselves, from the models, processor only (about 20 minutes on the lab's machine):

```
python papers/real-convergence/artifact_2026_11_10/extract_v2.py papers/real-convergence states2 --only Qwen2.5-0.5B
python papers/real-convergence/artifact_2026_11_10/score_v2.py papers/real-convergence states states2 out.json
```

Run the first line once per model named in the script. The extractors need Windows: they read free memory through
a Windows call outside any error handling, so on Linux or macOS they stop before any work. They read the models
offline from the local Hugging Face cache at the revisions pinned in their `RUN` lists (for Qwen2.5-0.5B, exactly
one cached snapshot), so download them beforehand; two of them are gated Llama repositories. torch, transformers and
safetensors are needed. The lab's run had about 13 GB of physical memory free and still needed an amended rerun
for two models; AMENDMENT 1 asks for 8 GB of commit headroom. The first test's scripts (`extract_small.py`,
`score_artifact.py`) run the same way and write the `states` folder that the second scorer reads as a control.
The per-template state arrays are not in the repository. Their sha256 values are, in
`artifact_2026_11_10/STATES_SHA256SUMS.txt` and `STATES2_SHA256SUMS.txt`.
