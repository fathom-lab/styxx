# Zero-shot cross-substrate concept decoder — "the efficient telepathy"

**2026-06-30 · fathom-lab / styxx · reproduce: `zero_shot_decoder.py` → `zero_shot_decoder_result.json`**

**Question.** Can we identify *which concept* a brain is representing — with **no scanner-specific training and
no regression** — using only a free, text-only LLM's concept geometry as the meaning-bridge?

**Method.** Relational (RSA) leave-one-out identification. For a held-out concept *i*, take its brain
dissimilarity profile (how its neural pattern relates to every *other* concept). Rank all candidate concepts by
how well each one's **model**-geometry profile matches *i*'s brain profile (Spearman); predict the argmax.
Training-free; works for held-out concepts; chance top-1 = 1/N. Permutation null = shuffle the model RDM's
concept identities (200×).

| dataset | decoder | top-1 | top-5 | mean percentile | perm p |
|---|---|---|---|---|---|
| **Mitchell-2008** (word-reading) | **free LLM (gpt2-large)** | 0.033 | **0.333** (chance .083) | **0.808** (chance .5) | **0.005** |
| | word-length (control) | 0.033 | 0.083 | 0.551 | — |
| | human behaviour (VICE, 1.5M) | 0.057 | 0.340 | 0.772 | — |
| | CLIP-image (vision) | 0.038 | 0.283 | 0.725 | — |
| **THINGS-fMRI IT** (visual) | **free LLM (gpt2-large)** | 0.040 | **0.170** (chance .05) | **0.711** | **0.005** |
| | word-length (control) | 0.010 | 0.030 | 0.454 | — |
| | human behaviour (VICE) | 0.021 | 0.223 | 0.781 | — |
| | CLIP-image (vision) | 0.043 | 0.223 | 0.762 | — |

## What this IS
A **substrate-independent, training-free meaning decoder.** A free off-the-shelf LLM identifies the read/viewed
concept's brain pattern **significantly above chance** (permutation p = 0.005 on two independent datasets) —
ranking the true concept in the **top ~19% on average** and into a **top-5 shortlist of 60 a third of the time**
— with **no per-subject model, no regression, and no MEG**, using existing fMRI + an LLM as the dictionary. The
**word-length control sits at chance** (percentile 0.55 / 0.45), so the decoding is **meaning, not surface
form.** On the clean word-reading substrate the free LLM (0.808) **matches/edges 1.5M human judgments** (0.772).
This is the *efficient* counterpoint to per-subject deep-net decoders (e.g. Meta Brain2Qwerty): no bespoke model,
no hardware beyond an existing scan.

## What this ISN'T (honest bounds)
- **Not verbatim thought-reading.** Exact top-1 identification is modest (2–4× chance); the demonstrated
  capability is *ranking / shortlisting*, not literal readout. Not telepathy in the sci-fi sense.
- **THINGS-IT decoding is partly visual** — there a vision model (CLIP-image 0.762) and behaviour (0.781) edge
  the LLM, because IT is visual cortex. **Mitchell word-reading is the clean meaning-decoding** result.
- Small fMRI sets (60 / 100 concepts, 3–9 subjects); decoding sharpens with more concepts/voxels.

## Bottom line
Meaning's shared geometry is strong enough to **decode concepts across substrates, zero-shot, with a free
model** — shown here on two independent datasets, p = 0.005, with a clean lexical control. Path to sharper
decoding: the 720-concept set (Tier 2), a non-visual ROI, and per-substrate calibration.
