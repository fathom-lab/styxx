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

## Honest differences & caveats (load-bearing)
1. **VISION CONFOUND — the gate.** IT is high-level *visual* cortex and subjects **saw photographs**, so the
   neural RDM carries visual + semantic structure. The LLM↔IT correlation may be partly visually mediated. This
   is **not a clean "meaning" claim** until an image-model (CLIP / CNN) RDM is partialled out — which needs the
   100 test images. (Mitchell-2008 was word-reading, so this confound did not exist there.) **Do not announce as
   a meaning result until this control is run.**
2. **Here human behavior leads the LLM** (VICE 0.222 vs LLM 0.154) — they were ~tied on Mitchell. On THINGS-IT,
   1.5M human judgments track the brain better than the LLM does.
3. **Only 3 subjects**; per-subject LLM RSA [0.124, 0.128, 0.074] (subject 3 weaker).
4. VICE matched 94/100 concepts (unmatched: chest1, coat_rack, ferris_wheel, hula_hoop, mosquito_net, sim_card).

## Bottom line
Pipeline validated, and the *LLM-beats-embedders, above-chance brain match* pattern **replicates on independent
data**. The **vision-confound control is the next gate** before any meaning claim. Resolving *within-category*
structure (the original motivation) needs the 720-concept main set (24.8 GB streamed) — Tier 2.
