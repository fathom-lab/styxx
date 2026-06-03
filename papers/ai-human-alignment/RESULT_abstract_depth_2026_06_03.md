# RESULT — Does depth help ABSTRACT meaning more? Directional, but NOT significant (honest negative).

**Date:** 2026-06-03 · The operator's deepest point: abstract meaning (anger, argument, charity) is
where "universal form" lives, and the concrete-noun work never touched it. Falsifiable hypothesis: the
deep-LLM advantage over shallow co-occurrence (GloVe) should be **larger for abstract** word pairs,
because co-occurrence is a weaker proxy for abstract meaning. Tested behaviorally on **SimLex-999**
(human similarity ratings + USF concreteness), deep-LLM (Llama-3B + Qwen-3B consensus) vs GloVe-50.

## Result
| SimLex pairs | GloVe ρ | deep-LLM ρ | LLM − GloVe |
|---|---:|---:|---:|
| ALL (n=999) | 0.265 | 0.329 | **+0.064** |
| ABSTRACT (conc ≤ 2.9, n=330) | 0.214 | 0.309 | +0.095 |
| CONCRETE (conc ≥ 4.6, n=330) | 0.252 | 0.303 | +0.051 |

- **The overall depth advantage is real:** deep LLMs predict human similarity better than shallow
  co-occurrence (+0.064 Spearman) — consistent with the +12.8% found on concrete behavioral data.
- **The abstract-specific claim is NOT confirmed.** The advantage *leans* larger for abstract
  (+0.095 vs +0.051), and GloVe does falter on abstract words (0.214 vs 0.252) while the LLM holds
  (~0.305 both) — but a **bootstrap of the abstract-minus-concrete difference: mean +0.046, 95% CI
  [−0.13, +0.22], P(>0) = 0.70.** The CI crosses zero; the effect is **directional but underpowered,
  not significant.** I cannot claim "depth matters more for abstract meaning."

## Discipline note
The point estimates looked like a clean 2× and were one keystroke from being written up as a finding.
The bootstrap (a pre-stated check) said *not significant* and overrode the eyeball. Fifth honest
override of the night — the kind that's invisible in the published-only record but is the whole method.

## Caveats / next
- n = 330 per concreteness tertile; SimLex word-pair similarity is intrinsically noisy (ρ ~0.2–0.33);
  pair-cosines are not lexical-controlled; two-model consensus. The effect may be real but needs more
  power (full concreteness-balanced pair sets) to resolve.
- The **decisive** abstract test is neural: the Pereira-2018 brain data over its **152 abstract
  concepts** (asset secured, `pereira/NEXT_pereira_abstract_brain.md`) — does the deep advantage that's
  large in behavior appear in the brain, and is it largest for abstract concepts? That remains the
  experiment that would actually answer it.

## Reproduce
`run_abstract_depth_simlex.py` (SimLex-999 via fh295.github.io; GloVe via gensim; Llama-3B+Qwen-3B).
Result: `abstract_depth_simlex_result.json`.
