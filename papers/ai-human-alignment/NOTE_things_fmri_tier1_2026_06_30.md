# THINGS-fMRI Tier 1 — does the LLM↔brain meaning match replicate on independent data?

**2026-06-30 · fathom-lab / styxx · reproduce: `tier1_things_fmri_it.py` → `tier1_things_fmri_result.json`**

Independent test of the committed Mitchell-2008 result on a second fMRI dataset: **THINGS-fMRI** (Hebart et al.
2023), inferior-temporal cortex, **100 repeatedly-presented test concepts, 3 subjects × 12 reps**. Concept RDM =
z-scored IT voxels → correlation distance, averaged across subjects. Partial-lexical RSA to our LLM / VICE /
embedder geometries. Across-subject noise ceiling (Spearman) **[0.277, 0.675]**.

Data (75 MB, public, no-auth, figshare collection 6161151): `betas_csv_testset` [files/41039093] +
`noise_ceilings` [files/36682266] + `rois` [files/38517326] + `brainmasks` [files/36682242] → `things_fmri/`.

| representation | RSA to IT (partial-lexical) | 95% CI | % of ceiling-lo |
|---|---|---|---|
| **gpt2-large (text-only LLM)** | **0.154** | [0.087, 0.233] | **56%** |
| mpnet (embedder) | 0.126 | [0.067, 0.187] | 45% |
| MiniLM (embedder) | 0.110 | [0.058, 0.165] | 40% |
| VICE / human behavior (94 concepts) | 0.222 | — | — |

## What replicates
A **text-only LLM** predicts brain concept-geometry **above chance** (CI excludes 0) on a completely independent
fMRI dataset — different lab, different subjects, different concepts, different ROI — and **again beats both
purpose-built embedders in the same order** (LLM > mpnet > MiniLM), at 56% of the lower noise ceiling. The core
pattern from Mitchell-2008 holds.

## Vision-confound gate — PASSED (Tier 1b, `tier1b_things_fmri_vision.py`)
IT is high-level *visual* cortex and subjects **saw photographs**, so the neural RDM carries visual + semantic
structure. We partialled out a **CLIP-image RDM** (clip-ViT-B-32 over each concept's THINGS image) on top of
word-form — the same conservative control as the Mitchell result (CLIP is vision-*language*, so it removes
semantics too). The LLM↔IT match **survives**:

| partial(LLM, IT \| …) | value |
|---|---|
| raw | 0.172 |
| \| word-form (lexical) | 0.177 |
| **\| lexical + CLIP-image vision** | **0.103, 95% CI [0.041, 0.165]** |

Reference: vision→IT \|lex = 0.169 (CLIP-image strongly predicts IT, as expected); LLM↔vision \|lex = 0.545
(the LLM geometry is itself fairly "visual"). **Near-identical to the Mitchell vision control (0.182→0.107);
here 0.177→0.103.** A text-only LLM predicts IT geometry **beyond a vision model + word-form**, on an
independent dataset and a visual paradigm — meaning, not pixels.

## Honest caveats (load-bearing)
1. **Proxy image:** the vision RDM uses each concept's *representative* THINGS image (imgur URL in
   `things_concepts.tsv`), not the exact fMRI test stimulus (gated). Same method as the committed Mitchell
   control; conservative, concept-level. The exact-stimulus control is a future refinement.
2. **Here human behavior leads the LLM** (VICE 0.222 vs LLM 0.154 in the uncontrolled RSA) — they were ~tied on
   Mitchell. On THINGS-IT, 1.5M human judgments track the brain better than the LLM does.
3. **Only 3 subjects**; per-subject LLM RSA [0.124, 0.128, 0.074]. 94 concepts after image/VICE matching.

## Bottom line
The headline **replicates on an independent fMRI dataset**: a text-only LLM predicts brain concept-geometry
above chance, beats both embedders, and **survives a conservative vision control** (0.103, CI excludes 0) — the
match is meaning, not pixels, mirroring the Mitchell result. Resolving *within-category* structure (the original
motivation) still needs the 720-concept main set (24.8 GB streamed) — Tier 2.
