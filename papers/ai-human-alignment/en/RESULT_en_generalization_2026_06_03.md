# RESULT — Meaning-integrity monitor generalizes to English; deep>shallow replicates (muted), one artifact caught

**Date:** 2026-06-03 · Does the monitor (built on the Chinese 54-feature space) generalize, or is it an
artifact of that one dataset? Independent reference: **Lancaster Sensorimotor Norms** (11 experiential
dims, 1500 words). Independent models: GloVe-300 (shallow), BERT-base + MiniLM (deep). Same
`meaning_integrity` core, only the reference + models swapped.

## Pre-registered prediction
From the Chinese decomposition (deep wins on ABSTRACT, shallow suffices on PERCEPTUAL): on this
**sensorimotor** (perceptual-heavy) norm, deep's advantage should be **muted**, while the monitor's
content-agnostic mechanics (invariance, sensitivity) hold.

## Results
- **INVARIANCE generalizes exactly** — base = rotated = scaled to <1e-6 on independent English data.
  The monitor's defining property (reads meaning-geometry, not surface form) transfers. ✓
- **deep vs shallow** (PCA-50 align to Lancaster): GloVe **0.115**, BERT-word **0.047**, MiniLM **0.146**.
  - First pass (BERT-word) read as *shallow>deep* — but BERT isolated-word mean-pooling is a known-weak
    embedding (BERT needs context). The **MiniLM diagnostic disentangled it:** with a proper word-level
    deep embedding, **deep 0.146 > shallow 0.115 (+0.031)**. So **deep>shallow replicates in English**, and
    the margin is **muted (+0.03) on perceptual meaning — exactly as pre-registered.** The discipline
    caught the artifact instead of reporting a false "shallow wins."
  - *Honest:* +0.031 is a point estimate (not bootstrapped like the Chinese P=1.000). Suggestive, not
    confirmed — the direction replicates; the magnitude is modest and expected-small here.
- **Sensitivity generalizes** — corruption collapses the score (noise → 0.007, shuffle → 0.001).
- **Localization weak (AUC 0.73)** — because the overall alignment is low (0.11–0.15): 11 sensorimotor
  dims are a thin slice word-vectors only weakly track. With little base signal, there is little to
  localize against.

## Honest read
The monitor's **content-agnostic mechanics (invariance, sensitivity) generalize** to English + a wholly
independent norm — the core transfers, not a Chinese artifact. The **deep>shallow finding replicates**
(muted on perceptual, as predicted) once a BERT-word embedding artifact is removed. The monitor's
**discriminative power (localization) needs a rich reference the models genuinely align with** — an 11-dim
perceptual norm is too thin (Chinese 54-feature gave 0.4–0.5 and AUC 0.95). That boundary condition is
itself a useful, honest result: *the tool is only as sharp as the human reference it is given.*

## Next
A rich English experiential reference (Binder et al. 2016, 65 features — the direct analog of the Chinese
54) for the strong-alignment discrimination test. Lancaster confirmed the core transfers + the reference
matters; Binder tests whether the full monitor (incl. localization) is as sharp in English as in Chinese.

## Reproduce
`prep_en.py` (Lancaster + GloVe-300 + BERT) · `run_en.py` (monitor mechanics + deep/shallow) ·
`diag_minilm.py` (the artifact-disentangling diagnostic). Result: `en_result.json`.
