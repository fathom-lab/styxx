# Cognometric Phase Transitions — Feature-Count Scaling in the Drift Detector

**Research question.** As features are added to the calibrated tool-call
drift detector, does per-failure-type AUC improve smoothly (incremental
learning) or in discrete jumps (phase transitions)? A phase-transition
signal would be the mirror of emergent-capability literature — specific
failure classes becoming *detectable* at specific feature-count
thresholds rather than linearly with compute.

**Dataset.** drift_v0, n=3,700 (658 gold no-drift + 3,042 drift positives
via mutation + irrelevance splits), from Berkeley Function Calling
Leaderboard v3. 5-fold stratified CV, random_state=0.

**Protocol.** Three ablations:

- **top-K by importance.** Rank the 22 features by |coef| from a full-
  model baseline, then retrain with only the top-K. Traces the ceiling
  at each K.
- **group-incremental.** Add features in group order — A (semantic
  alignment, 5) → B (schema conformance, 6) → C (lexical drift, 4) →
  D (structural, 7). Shows which group is responsible for each failure.
- **random subsets.** K features drawn uniformly at random, 3 seeds per
  K. Null expectation for top-K.

**Phase-transition criterion.** A per-drift-type AUC jump ≥ 0.10 between
consecutive K values in the top-K schedule.

---

## Feature importance ranking (|coef| on full 22-feature model)

| rank | feature | group |
|------|---------|-------|
| 1 | `spurious_arg_frac` | B_schema |
| 2 | `arg_count_zscore` | B_schema |
| 3 | `n_available_tools` | C_lexical |
| 4 | `missing_required_frac` | B_schema |
| 5 | `placeholder_frac` | C_lexical |
| 6 | `type_mismatch_frac` | B_schema |
| 7 | `tool_in_schema` | B_schema |
| 8 | `tool_in_any_schema` | C_lexical |
| 9 | `tool_parts_in_prompt` | A_semantic |
| 10 | `prompt_coverage` | A_semantic |
| 11 | `required_count` | B_schema |
| 12 | `n_args_called` | D_structural |
| 13 | `arg_verbatim_rate` | A_semantic |
| 14 | `overlap_jaccard` | A_semantic |
| 15 | `tool_in_prompt` | A_semantic |
| 16 | `avg_arg_len` | D_structural |
| 17 | `has_list` | D_structural |
| 18 | `prompt_len` | D_structural |
| 19 | `prompt_is_question` | D_structural |
| 20 | `prompt_imperative` | D_structural |
| 21 | `has_nested` | D_structural |
| 22 | `tool_name_len` | C_lexical |

---

## Top-K scaling curve

| K | pooled AUC | arg_drop | spurious_arg | irrelevance_called | arg_swap |
|---|------------|----------|--------------|--------------------|----------|
| 1 | **0.606** | 0.501 | 0.999 | 0.486 | 0.512 |
| 2 | **0.788** | 0.998 | 0.997 | 0.705 | 0.488 |
| 3 | **0.821** | 0.998 | 0.997 | 0.793 | 0.487 |
| 4 | **0.821** | 0.998 | 0.997 | 0.793 | 0.487 |
| 5 | **0.832** | 0.998 | 0.997 | 0.828 | 0.481 |
| 6 | **0.874** | 0.998 | 0.997 | 0.828 | 0.691 |
| 8 | **0.875** | 1.000 | 0.997 | 0.828 | 0.693 |
| 10 | **0.920** | 1.000 | 0.997 | 0.962 | 0.674 |
| 12 | **0.919** | 1.000 | 0.997 | 0.962 | 0.669 |
| 14 | **0.920** | 1.000 | 0.997 | 0.963 | 0.669 |
| 17 | **0.923** | 0.999 | 0.996 | 0.966 | 0.683 |
| 19 | **0.924** | 0.999 | 0.997 | 0.966 | 0.685 |
| 22 | **0.923** | 0.999 | 0.997 | 0.966 | 0.684 |

---

## Detected phase transitions (Δ ≥ 0.10 between consecutive K)

| drift type | K: from→to | AUC: from→to | Δ | feature added |
|---|---|---|---|---|
| `arg_drop` | 1 → 2 | 0.501 → 0.998 | **+0.497** | `arg_count_zscore` |
| `irrelevance_called` | 1 → 2 | 0.486 → 0.705 | **+0.219** | `arg_count_zscore` |
| `tool_rename` | 1 → 2 | 0.699 → 0.900 | **+0.201** | `arg_count_zscore` |
| `arg_swap` | 5 → 6 | 0.481 → 0.691 | **+0.210** | `type_mismatch_frac` |
| `tool_rename` | 5 → 6 | 0.699 → 0.799 | **+0.100** | `type_mismatch_frac` |
| `irrelevance_called` | 8 → 10 | 0.828 → 0.962 | **+0.134** | `tool_parts_in_prompt`, `prompt_coverage` |

---

## Group-incremental results

| groups active | K | pooled AUC | arg_drop | spurious_arg | irrelevance_called | arg_swap |
|---|---|---|---|---|---|---|
| A | 5 | **0.725** | 0.706 | 0.657 | 0.899 | 0.495 |
| A+B | 11 | **0.910** | 0.999 | 0.997 | 0.944 | 0.660 |
| A+B+C | 15 | **0.920** | 1.000 | 0.997 | 0.962 | 0.671 |
| A+B+C+D | 22 | **0.923** | 0.999 | 0.997 | 0.966 | 0.684 |

---

## Interpretation

The top-K ablation exhibits **phase-transition behaviour**: specific
failure classes remain near-null until a critical feature enters the
set, then snap to high-AUC detection. This is the drift-detection
analogue of emergent capabilities in generative LMs — detectability
is not a smooth function of representational capacity.

**Top-K vs random subsets.** If top-K substantially outperforms random
subsets at the same K, the ranking is non-trivial — there exists a
small set of load-bearing features. If the gap is small, the
classifier is additive and no single feature dominates.

**Reproducer.** `python scripts/drift_feature_scaling.py`. Raw numbers:
`benchmarks/drift_feature_scaling.json`. Figure:
`papers/figures/drift_phase_transitions.png`.
