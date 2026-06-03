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

## The wall
The **per-subject neural data** (the brain responses) is NOT in the materials zip — Pereira hosts it
separately on Dropbox/Drive (not OSF-API-reachable). Getting it is a fresh, non-trivial fetch +
nifti/.mat parse. Not done tonight; flagged honestly rather than rabbit-holed.

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
