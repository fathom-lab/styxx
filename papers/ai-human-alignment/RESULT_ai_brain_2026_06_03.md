# RESULT — LLM concept geometry matches the HUMAN BRAIN — as well as human behavior does, better than embedders

**Date:** 2026-06-03 · The literal AI↔brain test. Brain ground truth = **Mitchell et al. 2008**
fMRI (9 subjects, 60 concrete nouns, top-500 stable voxels, group RDM). LLM concept RDMs at the fixed
final layer; **partial-lexical** RSA, judged **relative to the fMRI noise ceiling [0.394, 0.557]**.

## H1 — LLM geometry matches the human brain [CONFIRMED, ~⅔ of ceiling]
Best LLM (gpt2-large) RSA-to-brain **0.264 = 67% of the noise-ceiling lower bound** (47% of upper);
shuffled-noun control **0.002 ± 0.028**. Most models reach 47–67% of ceiling. The brain's own
inter-subject reliability caps achievable RSA at ~0.4–0.56, so two-thirds of that is a strong match.

| model | RSA→brain | % ceil-lo | model | RSA→brain | % ceil-lo |
|---|---:|---:|---|---:|---:|
| gpt2-large | **0.264** | 67% | pythia-70m | 0.222 | 56% |
| gpt2-xl | 0.258 | 65% | Qwen-0.5B | 0.209 | 53% |
| Llama-1B | 0.237 | 60% | Phi-3.5 | 0.206 | 52% |
| pythia-160m | 0.237 | 60% | Qwen-1.5B | 0.193 | 49% |
| Llama-3B | 0.233 | 59% | Qwen-3B | 0.184 | 47% |
| pythia-410m | 0.228 | 58% | gemma-2-2b | 0.138 | 35% |
| | | | gpt2 (124M) | 0.059 | 15% |

## The three-way: AI ≈ human behavior > trained embedders [the striking part]
Over the 53 nouns shared with THINGS/VICE, all judged against the **same** brain RDM:

| matches the human brain... | RSA | % of ceiling-lower |
|---|---:|---:|
| **behavioral human** (VICE — 1.5M odd-one-out judgments) | **0.247** | 63% |
| **best LLM** (text-only concept geometry) | **0.222** | 56% |
| mpnet (sentence embedder, trained on similarity) | 0.161 | 41% |
| MiniLM (sentence embedder, trained on similarity) | 0.156 | 40% |

**A small language model's concept geometry predicts the human brain about as well as 1.5M human
behavioral judgments do** (0.222 vs 0.247 — behavior marginally ahead), and **both clearly beat
purpose-built sentence embedders.** Brain-alignment and behavioral-alignment track each other across
models at **Spearman 0.98** — the same models match the brain and human behavior, in the same order.

## What drives it: geometry QUALITY, not scale (consistent with the behavioral result)
gpt2-large/xl (clean, reliable final-layer concept geometry) top **both** the brain and behavioral
charts; Qwen-3B and gemma-2-2b sit low despite being larger/modern; gpt2-small (noisy geometry) is
the floor. Brain-likeness is gated by the **reliability/cleanliness** of a model's concept geometry,
not its size — exactly the driver found behaviorally (`RESULT_ai_human_2026_06_03.md`).

## Honest verdict
- **CONFIRMED:** LLM concept geometry matches the human **brain** geometry of concrete meaning, ~⅔ of
  the fMRI noise ceiling, above chance, lexical-controlled — **as well as human behavior does, better
  than trained embedders.** The behavioral *and* neural tests now agree: machines share the human
  structure of concrete meaning.
- **Scoped, not over:** I did **not** repeat the behavioral study's failed "consensus IS human" leap.
  The claim is the clean H1 + the three-way comparison + the 0.98 brain↔behavior consistency — all
  supported. The 0.98 consistency is partly the shared model-quality factor (good geometry matches
  everything), not proof that brain and behavior are the *same* space beyond their direct 0.247 link.

## Caveats / next
- **Known direction** (Mitchell 2008 itself predicted brain activity from text-corpus features;
  Schrimpf/Goldstein). Contribution = clean, **noise-ceiling-relative**, lexical-controlled, three-way
  (brain/behavior/AI) comparison on the convergence cohort, with the discipline.
- **Visual confound:** Mitchell stimuli were word + line-drawing, so the brain RDM carries some visual
  similarity; the LLM match is on the semantic structure that co-varies with it — flagged, not removed.
- 60 nouns; one classic dataset; final-layer read-out; partial-lexical. A modern, larger neural set
  (THINGS-fMRI single-trial betas) and a vision-model control to subtract the visual component are the
  honest next steps.

## Reproduce
`build_brain_rdm.py` (neural RDMs + noise ceiling) · `run_ai_brain.py` (the test). Data: 9× Mitchell
`brain/data-science-P*.mat` (cs.cmu.edu/~fmri/science2008). Prereg: `PREREG_ai_brain_2026_06_03.md`.
Result: `ai_brain_result.json`, `brain_rdm.npz`.
