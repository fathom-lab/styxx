# RESULT — LLMs share the HUMAN geometry of meaning (strong). "The consensus IS human" does NOT survive.

**Date:** 2026-06-03 · Bridges tonight's machine-convergence work to human cognition, using a real
human ground truth: the **VICE/SPoSE human concept embedding** (Hebart 2020 / Muttenthaler 2022),
1,854 objects × 42 human dimensions distilled from ~1.5M human odd-one-out judgments. **120** of our
concrete concepts have a human vector. LLM concept RDMs read at the fixed **final layer** (where v3
located concept geometry; not chosen using human data). All alignments **partial-lexical** RSA.

## H1 — machines share the human geometry of meaning [CONFIRMED, strong]
Converging instruct models align with the human RDM at **mean 0.650** (shuffled-concept control
**0.000 ± 0.010**). For context, purpose-built sentence embedders trained *on* similarity:
MiniLM **0.602**, mpnet **0.652**. **The LLMs match humans as well as the dedicated embedders do.**
Top single models reach **0.74** — plausibly near the human-embedding noise ceiling.

| model | human-align | reliability | model | human-align | reliability |
|---|---:|---:|---|---:|---:|
| gpt2-xl | **0.744** | 0.982 | Qwen-1.5B | 0.671 | 0.903 |
| gpt2-large | **0.741** | 0.983 | Phi-3.5 | 0.670 | 0.844 |
| Llama-1B | 0.737 | 0.971 | Qwen-3B | 0.667 | 0.891 |
| pythia-410m | 0.716 | 0.969 | pythia-70m | 0.622 | 0.954 |
| Qwen-0.5B | 0.704 | 0.854 | gemma-2-2b | 0.457 | 0.935 |
| Llama-3B | 0.701 | 0.971 | gpt2 (124M) | 0.181 | 0.420 |
| pythia-160m | 0.700 | 0.967 | | | |

This reproduces an established direction (embeddings predict human similarity; brain-score) on a
clean, controlled, chance-corrected setup — the **floor**, not the claim.

## H2 — "convergence ⇒ humanness" is mostly a QUALITY artifact [NOT clean]
Across models, machine-convergence predicts human-alignment (Spearman **0.808**). But that is largely
a third factor — **RDM reliability** (split-half over template halves): human-alignment tracks
reliability at **ρ = 0.764**. Partialling reliability out:

> conv→human  **0.808 → 0.599** (control reliability) **→ 0.600** (control reliability + size).

A residual remains, but at **n = 13 it is borderline** (p ≈ 0.07) and hard to separate from "every
decent model captures the same dominant category structure." **Size does not drive human-alignment**
(Qwen-3B 0.667 ≤ Qwen-0.5B 0.704; the chart is topped by gpt2-large/xl). The honest driver is the
**reliability/cleanliness of a model's concept geometry**, not convergence per se and not scale.

## H3 — no wisdom-of-the-crowd toward the human geometry [FAILED — the decisive test]
The strong claim ("the cross-model consensus IS the human structure") predicts the consensus should
beat individuals at matching humans. It does **not**: consensus **0.688** < best single **0.744**,
≈ median **0.700**; even a reliability-screened strong-models-only consensus (**0.735**) does not beat
the best single model. **Averaging models does not get closer to the human geometry than a good single
model already is.** The human-aligned structure is already fully captured by good individual models;
there is no extra "consensus" component that is more human.

## Honest verdict
- **CONFIRMED:** LLMs and humans share a strong concept geometry over concrete objects (0.65–0.74,
  matching trained embedders, lexical-controlled, chance-corrected). Real; known direction; clean magnitude.
- **NOT SUPPORTED:** the deeper, seductive claim that *cross-model convergence/consensus is
  specifically the human geometry.* The convergence↔human link is dominated by geometry quality
  (reliability ρ 0.76; conv→human collapses 0.81→0.60), the residual is borderline at n=13, and the
  decisive consensus test (H3) fails. **Discipline note:** H2's raw ρ = 0.81 was the exciting number;
  the reliability control + H3 are why it is not the headline.
- **Incidental correction:** Phi-3.5 — the *machine*-convergence outlier in the earlier mid-layer /
  abstract-concept setting — is **unremarkable here** (final layer, concrete objects: convergence
  0.809, human 0.670, rank 5/13). Its outlier status was measurement-specific, not intrinsic.

## What is genuinely worth keeping
- A clean, controlled confirmation + **magnitude**: small LLMs match a 1.5M-judgment human similarity
  space as well as dedicated sentence embedders, at the final layer.
- The mechanistic finding that **human-alignment is gated by RDM reliability, not scale or consensus**
  — useful and slightly deflationary for "bigger/agreement ⇒ more human" narratives.
- A method that refused to let an exciting ρ = 0.81 become a claim it could not defend.

## Caveats / next
- **Behavioral** human data (odd-one-out), not neural; **concrete objects only** (120); n = 13 models;
  single final-layer read-out; one human embedding. The neural test (THINGS-fMRI RDMs) and a noise-
  ceiling-corrected, reliability-weighted consensus are the honest next steps; a wider/stronger model
  set would move the n = 13 H2 residual off the fence.

## Reproduce
`run_ai_human.py` (H1/H2/H3) · `run_ai_human_reliability.py` (the decisive reliability control).
Data: `data/final_embedding.npy`, `data/things_concepts.tsv` (VICE, ViCCo-Group/VICE).
Prereg: `PREREG_ai_human_2026_06_03.md`. Results: `ai_human_result.json`, `ai_human_reliability_result.json`.
