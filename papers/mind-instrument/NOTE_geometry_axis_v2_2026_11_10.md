# NOTE: the geometry axis of styxx.mind, version 2

Fathom Lab, 2026-11-10. A new module, `styxx/mind_geometry.py`, with new anchors, new tests and this note.
`styxx/mind.py` is not changed. One lab, one machine (Windows, Python 3.12, numpy 2.4.4). Outside reruns: 0. One
hand wrote the code, the tests and this note, against acceptance gates the lead wrote before the build. It has had
an internal hostile review on the same day and no outside review.

## what was wrong

`styxx.mind.geometry_citizenship` scores a candidate's concept geometry (96 battery words, last token, layer
max(1, int(0.66 L))) against stored anchor arrays and averages the partial-lexical RSA over the cross-family
anchors. Every array it used was the mean over 8 templates, and template 0 is the bare word `{w}`. The tokenizers
of Qwen2.5, Phi-3.5, GPT-2 and Pythia add no start token, so a word that is one token sits alone at position 0 in
that template. Qwen2.5 and Phi-3.5 carry a very large activation there (measured in the erratum; described by
Sun, Chen, Kolter, Liu, arXiv 2402.17762); for GPT-2 and Pythia it was not measured. The 8-template mean of such
a word is then that one state and little else. The correction of the June result this axis was built on is
`papers/real-convergence/ERRATUM_real_convergence_2026_11_10.md`.

In the shipped instrument this showed as one direction holding nearly all the variance across the 96 concepts:
in 3 of its 6 anchors (Qwen2.5-1.5B 0.9971, Qwen2.5-3B 0.9532, Phi-3.5-mini 0.9570, against 0.0821 to 0.0966 for
the other three), and in all 4 arrays of `atlas_live_reps.npz` in this folder (Qwen2.5-0.5B 0.9896, pythia-410m
0.9499, gpt2 0.9335, gpt2-large 0.8161). For GPT-2 and Pythia the large activation at position 0 was not
measured (the erratum, section "the scale result in this folder"): their tokenizers add no start token and the
share is high, and that is all that was checked. Scored as documented, Qwen2.5-0.5B got a citizenship of
0.1833.

## why styxx/mind.py stays as it was

`styxx.mind.geometry_citizenship` is the withdrawn June convention, and `styxx/mind.py` stays byte for byte as at
commit `4bbe4b3e` (dated 2026-11-10). Two lock tables pin the sha256 of that file: section 14 of
`papers/tests-pass-ci/PREREG_tests_pass_ci_2026_11_02.md` (read by `papers/tests-pass-ci/collect/tpc_score.py`,
which refuses to score when the file differs) and the darkflobi gate (`papers/darkflobi-gate/score.py`, which reads
the file at the pinned commit). An earlier draft of this change edited `styxx/mind.py`; the scorer's test file then
gave 27 failed and 17 passed. Changing that file would need a dated amendment to the tests-pass-ci lock table, and
this change is not the place for one. So version 2 lives in its own module, and the docstring and the comment above
the anchor battery in `styxx/mind.py` still carry the June wording ("confirmed out-of-sample", "pre-registered"),
which the erratum lists among the documents it does not edit.

## what changed

All in the new module `styxx/mind_geometry.py`, which takes `BATTERY`, `BATTERY_CATEGORY`, `TEMPLATES`,
`ANCHOR_MODELS`, `distmat`, `partial_corr` and the lexical controls from `styxx.mind` unchanged.

- `TEMPLATES_V2`: the 8 June templates (`styxx.mind.TEMPLATES`) without the bare word.
- `dominant_direction_share(reps)`: the share of the across-row variance on the largest direction (center the
  array, largest squared singular value over their sum). Raises ValueError on complex input, fewer than 3 rows, a
  non-finite value, or a variance across rows below 1e-20 of the array's sum of squares.
- `DOMINANT_SHARE_REFUSE = 0.5`. How it was set: on 2026-11-10, after the shares were known, so it is a cut chosen
  on these arrays and not a prediction. A screen that day covered the 47 stored 2-D float arrays with at least 20
  rows and 32 columns in the 21 .npz and .npy files tracked at commit `4bbe4b3e`. Of the stored arrays built with
  the bare template from tokenizers that add no start token, 8 are at or above 0.5 (0.8161 to 0.9984; the 7 above
  and `best__Phi-3.5-mini` of `contextual_reps.npz`). Two others, `best__Qwen2.5-1.5B` (0.1369) and
  `best__Qwen2.5-3B` (0.1358) of `contextual_reps.npz`, read at the layer with the best category structure and not
  at max(1, int(0.66 L)), are below it. So are the 20 unit-normalized arrays of `normeq_reps.npz` and
  `telepathy_reps.npz` in this folder, which average the bare template in after normalizing each template's state
  (0.0553 to 0.1074; the 14 of them from tokenizers that add no start token 0.0577 to 0.1074). No other screened
  array reached 0.37 (the highest, 0.3641, is a 60 x 60 brain RDM in `papers/ai-human-alignment/brain_rdm.npz`, not
  a representation). The anchors built without the bare template are at 0.0758 to 0.0977. The screening script is
  not in the repository; the shares of the arrays the gates name are recomputed by `tests/test_mind_geometry_v2.py`.
- `start_token_exposure(encode, words, templates, bos_id)`: given any callable from text to token ids, counts the
  inputs whose encoding is exactly one token that is not `bos_id`, that is, inputs whose read-out token would sit
  alone at position 0. No tokenizer library is imported. A single string as `words` or `templates` raises
  TypeError, and a template given twice raises ValueError.
- `within_category_null(candidate, anchor, n_perm=200, seed=0)`: the partial-lexical RSA after shuffling the
  candidate's rows inside their own category; deterministic for a seed. Ported from `within_cat_null` in
  `papers/real-convergence/artifact_2026_11_10/score_v2.py`, with the same permutation scheme. It differs in three
  ways: it computes the partial-lexical RSA (the axis's number) where the sealed scorer computed the raw Pearson r;
  it shuffles the candidate where the sealed scorer shuffled the second array of its pair; and it runs on the frozen
  battery where the sealed scorer's headline nulls were on fresh words. `n_perm` must be an integer.
- `geometry_citizenship(candidate, anchors_npz, candidate_family="?", *, convention="v2", allow_dominant=False,
  null_perms=0)`. With `convention="v2"` (the default of this module) the share is measured for the candidate and
  for every anchor present. A candidate at or above 0.5 gets `status` `REFUSED` and no number; an anchor at or
  above 0.5 is left out of the mean and listed in `anchors_excluded` with its share. The output always carries the
  shares, the threshold, the number of anchors used (`n_anchors`, `n_anchors_xfam`) and one fixed sentence
  (`GEOMETRY_CATEGORY_NOTE`) saying that most cross-model agreement on this battery is category structure.
  `null_perms > 0` adds the within-category null per used anchor and the excess over it. `allow_dominant=True`
  turns both refusals off and says so in `dominance_check`. Values are not rounded in this convention. `status` is
  `OK`, `REFUSED` or `NO_ANCHORS`. An array whose rows do not vary raises ValueError naming the array
  (`candidate_reps:` or `anchor fixed__<model>:`).
- **The old numbers:** `convention="june"` calls `styxx.mind.geometry_citizenship` unchanged and adds to its output
  `convention`, a `notice` saying that the convention is withdrawn, and the shares for reading only.
- `papers/mind-instrument/anchors_v2_2026_11_10.npz`: keys `fixed__Qwen2.5-1.5B`, `fixed__Qwen2.5-3B`,
  `fixed__Llama-3.2-1B`, `fixed__Llama-3.2-3B`, `fixed__Phi-3.5-mini`, each float32 (96 x d), the mean over
  templates 1 to 7 of the per-template states of the second sealed test of 2026-11-10 (original battery). gemma-2-2b
  has no rebuilt anchor: it was not re-extracted (it is no longer in the lab's cache).
- The command line `python -m styxx.mind_geometry geometry --anchors FILE.npz` prints every share and takes
  `--convention june`, `--allow-dominant` and `--null-perms N`. `python -m styxx.mind geometry` is the June command,
  unchanged.

## the gates, each with its measured result

The lead wrote seven acceptance gates before the build. Each has one test in `tests/test_mind_geometry_v2.py`,
which reads committed files only.

- **V1, the old numbers stay reachable.** With `convention="june"`, each of the 6 June anchors used as a candidate
  against `papers/real-convergence/contextual_reps.npz` gives the same `per_anchor` values, citizenship,
  `n_anchors` and `anchors_sha256` as `styxx.mind.geometry_citizenship` and as the function of commit `4bbe4b3e`
  (equality, not a tolerance). Measured: 6 of 6 equal, and also 4 of 4 for the arrays of `atlas_live_reps.npz` as
  candidates (10 of 10). The test loads the old function from git, or, where git or that commit is not at hand,
  compares with values pinned from it.
- **V2, the share separates the known arrays.** At or above 0.5: the June anchors of Qwen2.5-1.5B (0.9971),
  Qwen2.5-3B (0.9532) and Phi-3.5-mini (0.9570), and Qwen2.5-0.5B (0.9896), pythia-410m (0.9499), gpt2 (0.9335) and
  gpt2-large (0.8161). Below: the June anchors of Llama-3.2-1B (0.0966), Llama-3.2-3B (0.0821) and gemma-2-2b
  (0.0946), and all 5 arrays of the new anchors file (Qwen2.5-1.5B 0.0891, Qwen2.5-3B 0.0758, Llama-3.2-1B 0.0977,
  Llama-3.2-3B 0.0828, Phi-3.5-mini 0.0846). Of the 6 `best__` arrays of `contextual_reps.npz`, which the axis does
  not use, `best__Phi-3.5-mini` is at 0.9984 and the other 5 at 0.0918 to 0.1369; the test checks that the two
  `best__Qwen` arrays, built with the bare template, are below the cut.
- **V3, refusal.** Qwen2.5-0.5B from `atlas_live_reps.npz` against the new anchors: `REFUSED`, no number, the reason
  names the share 0.9896 and says a share this high can have other causes. The June anchor of Llama-3.2-1B against
  the June file: the 3 dominated anchors are left out and listed with their shares; 3 anchors used, 1 of them
  cross-family (gemma-2-2b), citizenship 0.9375. With `allow_dominant=True` Qwen2.5-0.5B against the June file gets
  the documented 0.1833 over all 6 anchors, and the output says the check is off.
- **V4, the new numbers are the sealed scorer's.** For the 10 pairs among the five models of the new anchors file,
  the function's partial-lexical RSA against `partial_lex_7` of `tables.original` in
  `papers/real-convergence/artifact_2026_11_10/score_v2_out.json`: largest difference 1.23e-9 (bar 1e-6). The two
  lexical controls are the same: the token counts in `papers/real-convergence/lex_tokens.json` (original words)
  equal `BATTERY_REF_TOKLEN`, 96 of 96. The remaining difference comes from storing the anchors as float32.
- **V5, the null.** For the 8 cross-family pairs of V4, the mean of `within_category_null` (200 draws, seed 0)
  against `within_category_null_7` of the same table: largest difference 0.0021 when the same array is shuffled as
  in the sealed scorer, 0.0022 when the other one is (bar 0.02). The nulls on the original battery are 0.6588 to
  0.6954. Two calls with the same seed return the same dict.
- **V6, the tokenizer check.** A stand-in tokenizer with no start token that gives a word of up to 5 letters one
  token: 63 of the 96 words are one token, and with the 8 June templates exactly those 63 inputs of 768 are counted,
  all in the bare template. The same stand-in with a start token: 0. With the 7 templates of version 2: 0 of 672
  for both.
- **V7, nothing else moves.** `styxx/mind.py` has the sha256 both lock tables pin (as git stores it, line endings
  folded), so it is byte for byte as at `4bbe4b3e`. `tests/test_mind.py` is byte for byte as at `4bbe4b3e` (line
  endings aside) and passes, 10 of 10. Loaded on its own, `styxx/mind.py` imports no numpy, and
  `styxx/mind_geometry.py` imports numpy inside its functions only. `import styxx.mind` does load numpy, because the
  package's `styxx/__init__.py` imports modules that import it; that is not this change.

One more number, outside the gates and not reproducible from committed files (it reads the per-template states of
the sealed test, whose sha256 values are in `papers/real-convergence/artifact_2026_11_10/STATES2_SHA256SUMS.txt`):
Qwen2.5-0.5B with the bare template dropped scores 0.9040 against the new anchors (share 0.0948; 3 cross-family
anchors, Llama-3.2-1B, Llama-3.2-3B and Phi-3.5-mini), and its partial-lexical RSA exceeds the within-category
null by 0.2351 on average over those 3 (200 draws each). Its 8-template array is refused (share 0.9896). The
stored array of Qwen2.5-0.5B in `atlas_live_reps.npz` is within 0.24 percent of the 8-template mean rebuilt from
those states, row by row (96 of 96 rows; largest row difference 0.00233 of the row's norm), so the array the Atlas
used is what that model gives under the June convention, and not a damaged file.

## rebuilding the anchors

The per-template state arrays are not in the repository. Write them with
`papers/real-convergence/artifact_2026_11_10/extract_v2.py` (see the erratum, section "re-running it"), then:

```
python papers/mind-instrument/build_anchors_v2.py <folder holding states2__original__<model>.npz>
```

The script imports `styxx.mind` and `styxx.mind_geometry` from this checkout and stops if either comes from
elsewhere. It checks each input's word and category order against `styxx.mind.BATTERY` and `BATTERY_CATEGORY`,
its shape and that every value is finite, checks each input's sha256 against `STATES2_SHA256SUMS.txt` (a differing
input stops it unless `--any-inputs` is given, for states extracted afresh), and prints the sha256 of each input
and of the output. It writes the zip with a fixed timestamp and key order, so the same inputs give the same bytes;
the committed file and two later rebuilds to other paths on this machine all had the sha256 in
`papers/mind-instrument/ANCHORS_V2_SHA256SUMS.txt`. Another numpy version has not been tried. A rebuild from fresh
extractions will not match bit for bit unless the states do (in test 1 of the erratum, re-extracted rows were
within 0.16 percent of the stored June rows, 96 of 96, not equal).

## what this does not show

- That 0.5 is the right threshold for model families other than those named here. It was chosen after the shares
  were seen.
- That an array below 0.5 is free of every artifact. Only the artifact this lab found is screened. In all 7 kept
  templates the word comes last, and 4 of the 7 put "a" before it.
- That a candidate built with the bare template is refused: at another layer, or after per-template normalization,
  it can fall below 0.5 (the two `best__Qwen` arrays and the normalized arrays above are examples).
- That an array above 0.5 carries an artifact. An array whose concept geometry really lay on one direction would be
  refused too.
- That the new number measures fine-grained shared geometry. On fresh words in the sealed test, shuffling words
  inside their own category still gave a raw RSA of 0.63 to 0.69 against 0.88 to 0.93 unshuffled. A citizenship
  near 0.9 says mainly that the candidate sorts these words into the same categories. Run with `null_perms` to
  see the part above the within-category null.
- That a near copy of an anchor is caught. `candidate_matches_anchor` names exact copies only; a scaled copy of an
  anchor scores an RSA of 1 against it and, with `candidate_family="?"`, raises the mean.
- Anything about gemma-2-2b under version 2. It was not re-extracted. Its June anchor (share 0.0946) is still
  scored when the June file is the anchors file, but it is an 8-template array; for version 2 use
  `anchors_v2_2026_11_10.npz`.
- How the candidate was built. The function cannot see the templates, the layer or the tokenizer behind a
  candidate array; `start_token_exposure` checks the tokenizer only if the caller runs it.
- That the code runs the same on Linux or macOS, or under another numpy version. It was run on one Windows machine.

## what was not re-validated

The Atlas of Minds (`ATLAS_OF_MINDS_v0.md`, `atlas_v0_result.json`) and the anatomy findings in this folder
(`FINDING_anatomy_v0_2026_06_10.md`, `FINDING_anatomy_v2_2026_06_10.md`) were computed with the old convention:
8 templates, the June anchors, and the arrays of `atlas_live_reps.npz`, all 4 of which version 2 refuses. They are
not re-validated here and are not edited. Other scripts in this folder average over `mind.TEMPLATES`, bare word
included (`run_anatomy_v1.py`, `run_anatomy_v2.py`, `run_gavagai_x.py`, `run_telepathy_v0.py`); the findings
built on them were not checked here either.

The scripts that call `styxx.mind.geometry_citizenship` (`run_atlas_v0.py`, `run_anatomy_v0.py` and
`validate_mind_v0.py` in this folder, and `papers/real-convergence/artifact_2026_11_10/mind_axis_check.py`) call
the unchanged function, so a re-run still gets the June convention and its numbers, with the start-token artifact
in them for the arrays named above. To score under version 2 a caller has to call
`styxx.mind_geometry.geometry_citizenship`. None of them is edited. `mind_certificate` in `styxx/mind.py` still
names the June preregistration and the label "styxx.mind v0", and its `instrument_sha256` is the sha256 of
`styxx/mind.py` only, so a certificate that carries a version 2 geometry output does not hash the module that made
it. A census of the other scripts of this lab that read hidden states with the bare template is
`CENSUS_start_token_exposure_2026_11_10.md` in this folder.
