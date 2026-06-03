# PREREG — Is the geometry machines CONVERGE to the HUMAN geometry of meaning?

**Date:** 2026-06-03 · **Status:** PRE-REGISTERED (gate frozen before data).
Tonight: real LLMs converge to a shared semantic geometry (confirmed, controlled, out-of-sample),
heterogeneous and data-driven. The deep question this opens — the actual payoff of the ancient
"is there one structure of mind?" — is whether that **shared machine geometry is the human one.**

## Human ground truth
**VICE / SPoSE human concept embedding** (Hebart 2020 / Muttenthaler 2022): 1,854 objects × 42
human-interpretable dimensions, distilled from ~1.5M human odd-one-out similarity judgments. This is
a behavioral map of how *people* structure object meaning. **122 of our concepts** (concrete
categories: animals, fruits, vehicles, tools, instruments, furniture, kitchen, body, clothing,
plants, drinks) have a human vector. Human RDM = cosine-distance over the 42-dim human vectors.

## Machine side
LLM concept RDMs at a **fixed final-layer read-out** (depth 1.0 — where v3 showed concept geometry
lives; uniform, non-selected, NOT chosen using human data → no circularity). Cohort = the 6
convergence-map instruct models + scale/diversity extenders (Pythia, GPT-2, Qwen-0.5B, gemma-3-1b)
for range, ~14 models spanning 70M–3B, base & instruct. MiniLM / mpnet embedders as reference points.
All alignments are **partial-lexical RSA** (char-length + token-count partialled out).

## Hypotheses & KILL-GATES (frozen)
- **H1 (AI↔human — the floor, KNOWN to hold):** the 6 converging instruct models align with the
  human RDM above chance. **Gate:** mean RSA_human(converging) ≥ 0.20 AND shuffled-concept control
  ≈ 0 (|ctrl| < 0.05). *Reproduces established AI–human alignment; the foundation, not the claim.*
- **H2 (THE DEEP ONE — convergence ⇒ humanness):** across the cohort, a model's machine-convergence
  (mean partial-lexical RSA with the *other* LLMs) predicts its human-alignment (RSA with VICE).
  **Gate:** Spearman ρ(convergence, human_align) ≥ 0.50, AND it survives partialling out log-params
  (capability) at ρ_partial > 0. → the consensus geometry IS the human-aligned one, beyond mere size.
- **H3 (consensus = human):** the cross-model CONSENSUS RDM (mean of the LLM RDMs) is **more**
  human-aligned than the median individual model. **Gate:** consensus RSA_human > median individual
  RSA_human.
- **Outlier consistency (descriptive):** is Phi-3.5 — the *machine*-convergence outlier — also the
  least *human*-aligned? (A clean "left both consensuses together" would corroborate H2.)

## Falsifiers (frozen)
- RSA_human ≈ 0 for all models → machines do NOT share the human geometry (H1 dead; whole claim dead).
- ρ(convergence, human_align) ≈ 0 → the machine consensus is NOT the human structure; convergence and
  humanness are unrelated (H2 dead — the deep claim fails even if H1 holds).
- consensus ≤ median individual → no "wisdom of the machine crowd"; H3 dead.

## Honest framing (frozen)
- **H1 is not novel** (embeddings predict human similarity; brain-score; Schrimpf/Goldstein). It is
  the foundation. **H2/H3 are the new, falsifiable contribution**: tying *cross-model convergence*
  to *human alignment* — "what machines agree on is what humans encode."
- Behavioral human data (odd-one-out), not neural; concrete objects only (122); one human embedding;
  fixed final-layer read-out. RSA = known-correspondence similarity, not unsupervised translation.
- Honest prior: H1 holds; H2 is genuinely uncertain (could be a pure capability confound — hence the
  log-params partial); H3 likely (consensus denoises). Report whatever lands, including a dead H2.

— frozen 2026-06-03
