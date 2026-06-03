# THE EDGE — Depth is real in behavior, invisible in the brain.
### A 2014 GloVe model predicts the human brain as well as a 3B LLM. The deep advantage is real — but it doesn't reach the brain.

**Date:** 2026-06-03 · Found by attacking the **shared-language confound** head-on: is the AI↔brain
"universal" just shallow human-language co-occurrence statistics? We pit a pure co-occurrence model
(**GloVe-50**, 2014), a **vision** model (CLIP-image), and the **deep LLM** consensus (13 models, ≤3B)
against each other at predicting the human brain (Mitchell 2008 fMRI, 60 nouns) and the clean human
behavioral geometry (VICE, 1.5M judgments). Partial-lexical RSA + variance partition, vs the noise ceiling.

## The brain: shallow co-occurrence is enough
| predicts the human BRAIN... | partial-lexical RSA | % of noise ceiling |
|---|---:|---:|
| **GloVe** (50-dim word co-occurrence, 2014) | **0.180** | 46% |
| **deep LLM** (consensus, ≤3B, 2024) | **0.182** | 46% |
| vision (CLIP-image) | 0.172 | 44% |
| human behavior (VICE) | 0.247 | 63% |

**A 50-dimensional word-co-occurrence model from 2014 predicts the human brain *exactly as well* as a
modern 3B LLM.** Variance partition: the deep LLM's UNIQUE contribution to the brain beyond GloVe +
vision + behavior is **+0.05%** — essentially zero. At fMRI resolution, the AI↔brain concept-geometry
match is **fully accounted for by shallow language statistics + vision.**

## But depth IS real — it just doesn't reach the brain
The obvious objection: the fMRI is noisy (everything explains only ~3% of variance), so maybe it
simply lacks the power to *see* a deep-model advantage. So we repeated it on the **clean** target
(VICE, high reliability), where power is not the bottleneck:

> **deep LLM → behavior 0.613  vs  GloVe → behavior 0.545 — the deep model uniquely adds 12.8% beyond
> GloVe.**

So the deep model genuinely captures human conceptual structure that shallow co-occurrence does not.
**Depth is real.** It is visible in 1.5M human judgments. It is **invisible in the brain.**

## The edge, stated precisely
**"Deep LLMs uniquely predict the human brain" — a popular claim — is NOT supported at current neural
resolution.** A 2014 GloVe model matches the brain equally; the deep-model advantage that is real and
large in *behavior* (+12.8%) collapses to ~0% in the *brain*. The shared AI↔brain structure is the
**coarse, co-occurrence-level** structure of concepts (category blocks, broad similarity) — which GloVe
already has. The **fine conceptual distinctions** deep models add are either **not encoded in the
neural geometry at this scale, or unresolvable by fMRI.** Two readings, which this can't separate:
- **(a) measurement:** fMRI is too coarse/noisy to resolve the finer structure deep models capture.
- **(b) substance:** the brain's measurable concept geometry genuinely encodes only the coarse
  co-occurrence-level structure; finer distinctions live elsewhere or aren't neurally geometric.

## Why this is the edge (and not the hype)
The field — and, two hours ago, *I* — frames AI↔brain alignment as "deep models capture the brain's
semantics." This controlled decomposition says: **at the brain, the deep part adds nothing GloVe doesn't;
the excitement is shallow co-occurrence wearing a 2024 logo.** That is a precise, falsifiable, contrarian
correction — the kind that is *ours* because we ran the deflation on our own result instead of selling it.

## The honest limit (stated, not buried)
GloVe, the LLM, CLIP, VICE, and the brain are **all human-derived.** This separates *shallow language*
from *deeper conceptual* structure — but it **cannot** test *substrate-independence* ("would emerge
without humans"). The strongest honest claim available is about **shared human-derived structure**, and
specifically that its **brain-measurable** part is shallow. The grand "universal structure of mind"
remains untested by these tools — and we say so.

## Caveats / next
53 nouns; one noisy fMRI dataset (2008, line-drawings); GloVe-50; consensus over ≤3B models. The
decisive next test of reading (a) vs (b): a **higher-resolution / higher-SNR neural dataset**
(THINGS-fMRI, MEG single-trial) — does the deep-model advantage that's real in behavior *appear* in
the brain when the neural signal is good enough? If yes → (a) measurement. If still absent → (b) substance.

## Reproduce
`run_edge_deflation.py` (GloVe via gensim glove-wiki-gigaword-50; brain + behavioral targets; variance
partition). Result: `edge_deflation_result.json`.
