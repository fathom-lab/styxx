# NEXT — the decisive experiment (asset secured, neural data pending)

**Status 2026-06-03:** the edge (`../RESULT_edge_deflation_2026_06_03.md`) found that depth is real in
behavior (+12.8% over GloVe) but only **faint/borderline in the brain** at Mitchell-2008 resolution
(group: GloVe 0.180 ≈ deep-LLM 0.182; per-subject paired t=2.52, 7/9, p≈0.04). Two readings remain
undistinguished: **(a) measurement** (fMRI too coarse) vs **(b) substance** (the brain's concept
geometry really is shallow). Resolving this needs **higher-SNR neural data**, and ideally **abstract**
concepts — the territory the concrete-noun work never touched, where "universal form" actually lives.

## Asset secured
- **Pereira et al. 2018** materials downloaded (`Pereira_Materials.zip`, 290 MB, gitignored; re-fetch
  `https://osf.io/download/hmgv2/`). **180 concepts** (`stimuli_180concepts.txt`, committed) — **152
  abstract** (ability, anger, argument, charity, attitude, challenge…) + 28 concrete. 16 subjects.
- GloVe-50 (`gensim glove-wiki-gigaword-50`) covers common words; pipeline (`run_edge_deflation.py`)
  is drop-in once a neural/behavioral RDM over these 180 exists.

## The wall (data-location reconnaissance, 2026-06-03 — done, so next session skips the search)
The **per-subject neural data** is NOT in the materials zip. Tracked it to source via the paper's Data
Availability statement (evlab PDF): *"raw and processed NIFTI imaging datasets … shared via
http://www.openfmri.org after re-processing."* So **Pereira brain data = raw NIFTI on OpenfMRI/
OpenNeuro** → requires a full fMRI preprocessing + single-trial GLM pipeline to get per-concept patterns.
- Alternative modern dataset checked: **OpenNeuro ds004301** (Wang 2022 Scientific Data) — 672 **Chinese**
  words, 11 subjects, *preprocessed* but only **4D BOLD time-series** (no betas/RDMs), many GB; ships
  Chinese GloVe/BERT/GPT2 + ResNet/ViT embeddings. Usable but: Chinese pipeline + GLM extraction + large.
- **Why Mitchell was usable and these aren't:** Mitchell shipped *pre-extracted per-trial voxel patterns*;
  the higher-SNR sets ship 4D BOLD needing a GLM. That GLM pipeline on GB-scale data is the real next
  project (use `nilearn` FirstLevelModel on the BIDS events → single-trial betas → concept RDM).
- **Discipline call (2026-06-03):** did NOT half-build the pipeline under time pressure — an untrustworthy
  neural RDM would be worse than an honest pause (it would launder a wrong number through the method).

## The decisive experiment (run when neural data is in hand)
1. Build the Pereira **brain RDM** over the 180 concepts (per subject + group + noise ceiling), same
   recipe as `build_brain_rdm.py`. Pereira's ceiling should be **higher** than Mitchell's [0.39,0.56].
2. Re-run `run_edge_deflation.py` logic: GloVe vs deep-LLM vs vision, predicting the Pereira brain.
3. **The test:** does the deep-model advantage that is large in behavior (+12.8%) **appear in the
   brain** when SNR is good? **YES → reading (a) measurement.** **STILL ~0 → reading (b) substance**
   (genuinely deep finding: the brain encodes only shallow co-occurrence structure of meaning).
4. **Bonus (abstract):** split the 180 into abstract vs concrete — is the deep>shallow gap larger for
   **abstract** concepts (where co-occurrence is a weaker proxy for meaning)? This is the operator's
   point made testable: abstract meaning is where depth should matter most.

## Also worth: a human behavioral reference for abstract concepts
VICE is concrete-only. For an abstract-concept human target without fMRI, consider SimLex-999 /
abstract-word similarity norms, or human association norms — to test depth-vs-shallow on abstract
meaning behaviorally while the neural data is sourced.
