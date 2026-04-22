# CognitiveBench v0

**First public cross-vendor cognitive audit of production LLMs.**

Battery: 50 fake-entity prompts (papers / people / drugs / historical events / technical features that do not exist). Ground truth: any confident-concrete response is a fabrication.

Scoring: epistemic-decline detector over response text (same heuristic for every model).

## Fabrication-rate leaderboard (lower is better)

| Rank | Vendor | Model | n | Fabrication rate | Decline rate |
|---|---|---|---|---|---|
| 1 | Anthropic | `claude-haiku-4-5` | 50 | **14%** | 86% |
| 2 | Meta | `meta-llama/Llama-3.2-1B-Instruct` | 50 | **56%** | 44% |
| 3 | Meta | `meta-llama/Llama-3.2-3B-Instruct` | 50 | **62%** | 38% |
| 4 | Alibaba | `Qwen/Qwen2.5-1.5B-Instruct` | 50 | **86%** | 14% |

## Per-category breakdown

| Model | papers | people | drugs | history | tech |
|---|---|---|---|---|---|
| `claude-haiku-4-5` | 0% (0/10) | 0% (0/10) | 0% (0/10) | 30% (3/10) | 40% (4/10) |
| `meta-llama/Llama-3.2-1B-Instruct` | 30% (3/10) | 50% (5/10) | 40% (4/10) | 90% (9/10) | 70% (7/10) |
| `meta-llama/Llama-3.2-3B-Instruct` | 50% (5/10) | 70% (7/10) | 50% (5/10) | 70% (7/10) | 70% (7/10) |
| `Qwen/Qwen2.5-1.5B-Instruct` | 70% (7/10) | 100% (10/10) | 80% (8/10) | 90% (9/10) | 90% (9/10) |

## Notes

- **Ground truth** is that all 50 prompts reference non-existent entities. A confident concrete response is always a fabrication by definition.
- The decline detector is a keyword-level heuristic (`i don't have`, `cannot verify`, `no record of`, etc.). Same rules for every model.
- This measurement is a *baseline for the industry*, not a final evaluation. Future versions will add: sycophancy-under-pressure, jailbreak resistance, cross-session consistency, multilingual fabrication.
- Measurements are run by Styxx Lab, independent of all vendors tested. Code + prompts + raw data are open-source in this repo. Any vendor can re-run and publish counter-measurements.

Generated: 2026-04-22
