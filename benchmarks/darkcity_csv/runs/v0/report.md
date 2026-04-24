# DarkCity Cognitive-State-Vector x P&L correlation

- **n decisions**: 2000
- **model**: claude-haiku-4-5 (DarkCity NPC brain)
- **vitals**: styxx.anthropic_hack text-heuristic (15-dim text features + 6 category probabilities)
- **outcomes**: depth_score, styxx_inflow_5min, styxx_outflow_5min
- **multi-test correction**: Bonferroni alpha=0.00076 across 66 tests

## Predicted category x outcome

| category | n | mean depth | mean STYXX in | mean STYXX out |
|----------|---|-----------|---------------|----------------|
| creative | 1697 | 0.92 | 0.00 | 0.00 |
| refusal | 155 | 0.92 | 0.00 | 0.00 |
| retrieval | 148 | 0.93 | 0.00 | 0.00 |

## Top correlates of `depth_score`

| feature | r | 95% CI | p | n | sig |
|---------|---|--------|---|---|-----|
| `n_words` | +0.137 | [0.09, 0.18] | 6.77e-10 | 2000 |  ★ |
| `n_lines` | +0.132 | [0.09, 0.17] | 2.57e-09 | 2000 |  ★ |
| `mean_line_length` | -0.117 | [-0.16, -0.07] | 1.31e-07 | 2000 |  ★ |
| `line_density` | +0.117 | [0.07, 0.16] | 1.40e-07 | 2000 |  ★ |
| `unique_ratio` | -0.089 | [-0.13, -0.05] | 6.88e-05 | 2000 |  ★ |
| `n_sentences` | +0.063 | [0.02, 0.11] | 4.93e-03 | 2000 |  |
| `hedge_density` | -0.059 | [-0.10, -0.02] | 8.12e-03 | 2000 |  |
| `uncertainty_density` | +0.054 | [0.01, 0.10] | 1.48e-02 | 2000 |  |
| `confidence_density` | -0.035 | [-0.08, 0.01] | 1.19e-01 | 2000 |  |
| `sentence_length_mean` | +0.033 | [-0.01, 0.08] | 1.45e-01 | 2000 |  |

## Top correlates of `styxx_inflow_5min`

| feature | r | 95% CI | p | n | sig |
|---------|---|--------|---|---|-----|
| `margin` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `n_words` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `n_sentences` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `n_lines` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `hedge_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `confidence_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `uncertainty_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `refusal_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `entity_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `claim_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |

## Top correlates of `styxx_outflow_5min`

| feature | r | 95% CI | p | n | sig |
|---------|---|--------|---|---|-----|
| `margin` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `n_words` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `n_sentences` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `n_lines` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `hedge_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `confidence_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `uncertainty_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `refusal_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `entity_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |
| `claim_density` | +0.000 | [-0.04, 0.04] | 1.00e+00 | 2000 |  |

## What this means

- Each row above is a hypothesis test: does an LLM reasoning-text feature predict a real economic outcome for the agent that produced it? ★ marks survive the Bonferroni correction across all 66 tests.
- The category table says: when the text-heuristic mode labels an agent's reasoning as `reasoning` vs `adversarial` vs `refusal`, do those labels actually track the economic reward the agent earned? If yes, cognitive labels are load-bearing on real P&L — first public evidence of that link.
- If no features survive Bonferroni: either the signal isn't in surface text (need tier-1 residual probes) or n is too small.
