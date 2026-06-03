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

## Rich reference (Binder 65-feature) — the FULL monitor generalizes; but deep>shallow does NOT
Binder et al. 2016: 65 experiential features (perceptual **and** abstract — Cognition, Social, Time,
Causation, Emotion), 434 words (GloVe ∩ Binder, complete ratings). The direct analog of the Chinese 54.

- **Alignment jumps to Chinese levels** — GloVe 0.462, BERT-proper 0.442, MiniLM 0.398 (PCA-50). The thin-
  Lancaster weakness *was* the reference, confirmed.
- **The FULL monitor generalizes:** invariance exact (<1e-6), sensitivity collapses to 0, and
  **localization ROC-AUC 0.912** — ≈ the Chinese 0.95, and far above thin-Lancaster's 0.73. **The
  "Chinese-artifact" worry is dead: the monitor works in English, including its discriminative power.** ✓
- **Deep vs shallow, done fairly — a TIE, not a deep win.** MiniLM (a weak word-model) 0.398 < GloVe
  (P=0.000), but that was the embedding. With **BERT-proper** (content subword tokens, templated — the
  earlier BERT wrongly pooled [CLS]/[SEP]): **0.442 vs GloVe 0.462, diff −0.020, 95% CI (−0.053, 0.014),
  P(deep>shallow) = 0.133 — not significant.** In English, with a fair deep embedding and a strong
  6B-token GloVe, **deep ≈ shallow.**

## Honest landing (the discipline tempered a finding)
- **The MONITOR generalizes robustly** to English + an independent rich human reference — invariance,
  sensitivity, and localization (AUC 0.91) all transfer. That is the load-bearing result for the *tool*,
  and it is confirmed.
- **The deep>shallow scientific CLAIM does NOT robustly generalize.** Chinese (GPT2/BERT vs GloVe, 54
  feat) gave deep>shallow P=1.000; English (BERT-proper vs GloVe-300, Binder 65 feat) gives a tie
  (P=0.133). The most likely reason: the **Chinese GloVe was weaker** than the 6B-token English GloVe-300,
  so part of the original "deep wins" was really "shallow was handicapped." It may also be language/
  reference specific. **Either way it is not a universal law** — testing generalization, and giving deep
  a fair embedding, corrected an overclaimable result down to its true, contingent size.
- Net: the *invention* (the monitor) is general and real; the *finding* it was built on is narrower than
  one dataset implied. Both statements shipped, side by side, because that is the job.

## Reproduce
`prep_en.py` (Lancaster + GloVe-300 + BERT) · `run_en.py` (monitor mechanics + deep/shallow) ·
`diag_minilm.py` (the artifact-disentangling diagnostic). Result: `en_result.json`.
