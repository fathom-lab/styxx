# A small free language model shares the brain's geometry of concrete meaning — controls and a training-free decoder

**Alex Rodabaugh · fathom-lab / styxx — working draft, 2026-06-30**
*Reproducible: github.com/fathom-lab/styxx (MIT). Data: Mitchell et al. 2008; Hebart et al. (THINGS-data) 2023.*

> Working draft. Every number is produced by a committed script and traceable to a result file in
> `papers/ai-human-alignment/`. We do **not** claim priority: prior work (Xu et al. 2025, PNAS; Du et al. 2024)
> already showed LLM-derived concept geometry aligns with THINGS-fMRI at the human-behaviour level. This paper
> **corroborates and adds controls** to that finding with a small, free, text-only model. Citations marked *
> need final verification before submission.

## Abstract
We test whether the *structure* of concrete meaning is shared across the human brain, human behaviour, and a
language model trained only on text — and we stress it with controls. On two independent fMRI datasets, a free,
off-the-shelf, text-only model (gpt2-large, 774M) represents concrete concepts in a geometry whose alignment with
the brain is **not statistically separable from that of ~1.5M human similarity judgments** on a 60-noun set
(Δ=0.021, paired bootstrap p=0.25; we did not run an equivalence test), and that **outperforms two sentence
embedders**. The alignment is not explained by word form, coarse category, or a (proxy) image model: it survives
partialling each out, though roughly half of the within-category signal in visual cortex is visual. On 823
concepts the alignment holds *within* category (the structure a 60-noun set cannot resolve) at 0.141–0.156 after
the vision control (95% CIs exclude zero). Finally, the shared geometry is strong enough that a **training-free,
no-regression relational decoder** identifies which concept a held-out brain pattern represents well above chance
(permutation p ≤ 0.01 on both datasets). Throughout, **human behaviour is the stronger model of the brain** — the
LLM joins it, never surpasses it. This corroborates Xu et al. (2025) and Du et al. (2024) with a far smaller model
and adds what they did not measure: the vision-controlled brain-meaning signal **climbs the ventral stream ~3×**
(localised to high-level object cortex, near-absent in early visual), the LLM carries a **small but reliable signal
beyond *both* a vision model and human behaviour** (full RSA, CIs exclude zero), and a **training-free decoder**
identifies concepts from brain activity. We frame it as measurement of shared meaning structure, not mind-reading.

## 1. The question and where it sits
Does the *geometry* of concrete meaning — which concepts are held close, which apart — generalise across the human
brain and a model that learned from text alone? A line of work says yes: feature- and embedding-based models
predict concept fMRI (Mitchell et al. 2008; Pereira et al. 2018*), and recent work shows LLM-derived concept
representations align with THINGS-fMRI at the level of human behavioural embeddings (Xu et al. 2025*, via
voxel-wise encoding with LLaMA3-70B; Du et al. 2024*, multimodal-LLM embeddings). Our question is narrower and our
apparatus smaller: with a **free 774M text-only model**, representational similarity analysis, and explicit
controls, how much of the brain's concrete-meaning geometry does it recover, how much survives a vision control,
and can the shared geometry **decode** without any scanner-specific training? (This is a different question from
output-readout BCIs such as MEG character decoding, which we mention only to contrast aims.)

## 2. Results

### 2.1 A small text-only LLM is on par with human behaviour as a model of the brain
On Mitchell et al. (2008) (9 subjects, 60 concrete nouns; group RDM over the top-500 stable voxels; noise ceiling
[0.394, 0.557]), gpt2-large's partial-lexical RSA to the brain is **0.264 = 67% of the lower noise ceiling** (60
nouns), vs a shuffle control of 0.002 ± 0.028. On the 53 nouns shared with the THINGS behavioural data (so the
same-model number is 0.222 here; 95% CIs by noun bootstrap):

| representation | RSA → brain | 95% CI |
|---|---|---|
| human similarity (VICE*, 1.5M odd-one-out judgments) | 0.247 | [0.145, 0.356] |
| free LLM (gpt2-large, text only) | 0.222 | [0.120, 0.327] |
| mpnet (sentence embedder) | 0.161 | [0.066, 0.262] |
| MiniLM (sentence embedder) | 0.156 | [0.063, 0.252] |

Paired bootstrap: the LLM's brain-alignment is **not statistically separable from human behaviour's** at this
sample size (Δ=0.021, p=0.25; **no equivalence test was run** — this is a failed rejection, not demonstrated
equality, and the direction slightly favours behaviour). It **exceeds mpnet** (p=0.011) and is **above MiniLM**
(p=0.043, marginal and uncorrected for the two comparisons). Brain- and behaviour-alignment rank models the same
way (Spearman ρ = 0.98).

### 2.2 Not surface form, not coarse taxonomy, partly visual
- **Lexical.** All RSA is partial-lexical (word + token length removed) by construction.
- **Vision (Mitchell, gpt2-large alone, 53 nouns w/ image).** Partialling out a CLIP-image RDM (conservative —
  CLIP is vision-*language*) leaves the match at **0.167** (0.234 raw → 0.237 |lex → 0.167 |lex+vision). It
  survives a proxy-image appearance control. (A 13-model consensus on the same set drops further, to 0.107.)
- **Category.** Partialling out a same/different-category indicator (12 categories): gpt2-large 0.264 → **0.206,
  95% CI [0.116, 0.310]** on the full 60-noun set. On the **matched 53-noun set**, the LLM's category-controlled
  value is **0.151 — below human behaviour's 0.186** ([0.088, 0.291]); behaviour leads here. The *within*-category
  test on 60 nouns is underpowered for everyone (LLM +0.032, CI spanning zero), motivating §2.4.

### 2.3 Replication on an independent fMRI dataset
THINGS-fMRI (Hebart et al. 2023; inferior-temporal cortex; 100 repeated test concepts × 12 reps; 3 subjects;
across-subject noise ceiling [0.277, 0.675]): gpt2-large reaches RSA **0.154 ([0.087, 0.233], 56% of the lower
ceiling)** and again exceeds both embedders (mpnet 0.126, MiniLM 0.110). It survives the vision control (0.177
|lex → **0.103 |lex+vision, CI [0.041, 0.165]**). Here human behaviour (0.222) clearly leads the LLM.

### 2.4 Within-category meaning, at scale (and partly visual)
We streamed the 26.6 GB THINGS-fMRI single-trial archive (processed one 75 MB volume at a time, never stored in
full) and built per-concept patterns over **823 concepts** in two high-level visual ROIs (union-ventral, LOC; 3
subjects; ceilings ventral [0.339, 0.703], LOC [0.331, 0.700]). The full set gives 7,165 within-category pairs
over 52 categories. gpt2-large's full RSA is **0.294 = 87% of the lower ceiling** (ventral) / 0.254 = 77% (LOC);
category-controlled 0.275 / 0.235 (not taxonomy). The **within-category** RSA, with a CLIP-image vision control
**computed on the 645 image-matched concepts (6,198 within-pairs)**:

| ROI | gpt2-large within-cat: \|lex → \|lex+vision | VICE/human (n=720) |
|---|---|---|
| ventral | 0.282 → **0.141, 95% CI [0.062, 0.218]** | 0.480 → 0.371 |
| LOC | 0.308 → **0.156, 95% CI [0.073, 0.234]** | 0.541 → 0.435 |

About **half** the within-category signal in this visual cortex is visual (the ~50% drop); the remainder survives
word-form, taxonomy, and a proxy-image control, and the LLM shares it — fine structure the 60-noun set could not
resolve. Human behaviour survives much more strongly.

### 2.5 Where the alignment lives: a visual→semantic gradient
Re-streaming the archive and extracting the ventral hierarchy (V1, V2, V3, hV4, LOC, union-ventral; 823 concepts),
the **vision-controlled** LLM↔brain meaning RSA (partial out word-form + CLIP-image) **climbs the stream** (645
image-matched concepts; 95% CIs exclude zero at every level):

| | V1 | V2 | V3 | hV4 | LOC | ventral |
|---|---|---|---|---|---|---|
| LLM↔brain meaning (\|lex+vision) | 0.057 | 0.045 | 0.049 | **0.131** | **0.165** | **0.147** |

Early visual (V1–V3) mean 0.05 → high-level (hV4/LOC/ventral) mean 0.148 — a **~3× rise**. A vision model predicts
all of these ROIs well (CLIP→brain |lex = 0.11–0.29 throughout), so the gradient is *not* a difference in how
visual the regions are; it is that the LLM's residual *meaning* alignment is **localised to high-level object
cortex, not the pixel end** (Fig. `fig_gradient.png`).

### 2.6 Beyond a vision model *and* human behaviour
Does the LLM explain brain meaning-geometry that neither appearance nor human behaviour captures? Partialling out
word-form + CLIP-image + the VICE behavioural RDM **simultaneously** (645 concepts), the LLM retains a reliable
partial: ventral **0.051, 95% CI [0.030, 0.072]**; LOC **0.059, [0.036, 0.083]** (full RSA). So the free LLM
contributes brain-meaning structure beyond *both* a vision model and 1.5M human judgments. The effect is **small** —
its unique R² (~0.2–0.3%) is roughly an order of magnitude below human behaviour's (~2.3–2.8%), and within-category
it is not significant. The LLM mostly *shares* structure with human behaviour, plus a small reliable residual.

### 2.7 A training-free cross-substrate decoder
Using only the LLM's concept geometry as a dictionary — **no scanner-specific training, no regression** (unlike
the voxel-encoding models of Xu et al. 2025*) — a relational leave-one-out decoder identifies which concept a
held-out brain pattern represents. On Mitchell it ranks the true concept in the **top ~19% on average** and into a
top-5 of 60 a third of the time (percentile 0.808, top-5 0.333 vs 0.083 chance, **permutation p = 0.005**); a
word-length control sits at chance (0.551). It replicates on THINGS-IT (0.711, p = 0.005) and the 823-concept
THINGS set (0.825 / 0.842, **p = 0.0099**, control at chance). Exact top-1 is modest (~2× to ~17× chance): this is
**shortlisting, not verbatim readout**.

## 3. Limitations (load-bearing)
- **Not first.** Xu et al. (2025, PNAS)* and Du et al. (2024)* already established LLM-derived concept geometry
  aligns with THINGS-fMRI at the human-behaviour level (with larger models / encoding models). This is
  corroboration plus added controls and a training-free decoder, not a priority claim.
- **Human behaviour is the stronger model**, especially within-category (VICE 0.37–0.44 vs LLM 0.14–0.16) and on
  THINGS-IT (0.222 vs 0.154). The LLM joins behaviour; it does not surpass it.
- **The visual confound is real, not removed**: ~half the within-category signal in ventral/LOC is visual. The
  vision control uses each concept's *representative* THINGS image (a proxy), not the exact gated stimulus, so the
  residual may still carry stimulus-specific visual variance.
- **Small samples** (Mitchell 9, THINGS-fMRI 3 subjects); group-RDM RSA; the headline rests on a specific
  checkpoint — gpt2-large/xl top a 13-model sweep while gpt2-small (124M) is near the bottom (0.059), and we make
  **no scale claim** (Spearman(size, RSA) = −0.09, n.s., from `make_ai_brain_figs.py`). Our single-bare-word,
  final-layer probe is weaker than the in-context representations of Xu et al.; the gpt2-large peak is a property
  of this probe, not a claim that 774M beats 70B.
- **Decoding is identification, not transcription.** No claim of reading propositions or private thought.
- **Scope:** concrete, picturable concepts; "shared geometry of concrete meaning is *partially*
  substrate-independent," not meaning writ large.

## 4. Methods (brief)
RSA: model/neural RDMs (cosine geometry for model embeddings; 1−Pearson on concept-mean, voxel-z-scored betas for
fMRI), Spearman of upper triangles, partial correlation removing lexical / category / CLIP-image RDMs. Noise
ceilings: Mitchell as published (top-500 stable voxels); THINGS by across-subject leave-one-out on group RDMs.
Shuffle control: 200 label permutations on one model; bootstraps resample concepts (1–2k iterations); decoder
permutation tests shuffle concept identities (100–200×). LLM vectors: final-layer hidden state for the bare
concept word. Vision: CLIP-ViT-B-32 image embeddings of each concept's representative THINGS image. Behaviour:
VICE embedding from ~1.5M THINGS odd-one-out judgments. The Tier-2 build masks each 75 MB betas volume to ROI on
the fly. All scripts committed.

## 5. Relation to prior work
**Closest:** Xu et al. (2025, PNAS)* — text-only LLM (LLaMA3-70B) concept representations predict THINGS-fMRI
across LOC/OFA/FFA/PPA/EBA via voxel-wise encoding, on par with human behavioural embeddings. Du et al. (2024)* —
LLM/multimodal-LLM odd-one-out embeddings over THINGS align with brain and behaviour. Caption-based LLM↔NSD work
(Doerig et al.*) and feature/embedding↔fMRI decoding (Mitchell 2008; Pereira 2018*) precede both. RSA methodology:
Kriegeskorte et al. (2008)*; noise ceilings: Nili et al. (2014)*. **Our increment** over this literature, with a
*small free* model rather than 70B: (1) an explicit **visual→semantic gradient** — partialling out a CLIP-image
model, the residual brain-meaning alignment is ~3× stronger in high-level object cortex than early visual,
localising *where* the LLM's meaning lives (§2.5); (2) a **simultaneous vision + behaviour partial** showing the
LLM retains a small but reliable signal beyond *both* a vision model and 1.5M human judgments (§2.6) — neither Xu
nor Du partialled both; (3) a **training-free relational decoder** rather than fitted encoding models; and (4) a
paired-bootstrap head-to-head against sentence embedders, on two datasets with behaviour as the calibrated
reference. We do not claim the core alignment as novel — Xu (2025) and Du (2024) established it.

## 6. Reproducibility
Scripts + result files (this directory): `run_ai_brain.py`, `bootstrap_ci_threeway.py`, `run_ai_brain_vision.py`,
`within_category_control.py`, `tier1_things_fmri_it.py`, `tier1b_things_fmri_vision.py`, `zero_shot_decoder.py`,
`stream_build_things720.py`, `tier2_analysis.py`, `tier2_vision.py`, `tier2_partition.py`, `stream_build_gradient.py`,
`tier2_gradient.py`, `make_ai_brain_figs.py`, `fig_decoder.py`, `fig_gradient.py`.
Data are public (Mitchell 2008; THINGS-data figshare collection 6161151).
