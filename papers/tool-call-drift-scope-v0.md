# Styxx Instrument #3: Tool-Call Drift Detection — Research Scope

**Date:** 2026-04-23. **Status:** pre-implementation research scope.
Will be superseded by a formal methods paper once the detector ships.

## Benchmark decision: BFCL v3 as primary

| Rank | Benchmark | n | License | Drift labels | URL |
|---|---|---|---|---|---|
| **1** | **BFCL v3** (Berkeley Function Calling Leaderboard) | ~5,500 total; specifically 1,000 `irrelevance` + 41 `relevance` + 200 `miss_func` + 200 `miss_param` + 200 `composite` | Apache-2.0 | **Yes, native.** `irrelevance` = should have refused; `multi_turn_miss_param` = should have asked; `multi_turn_miss_func` = no valid tool exists | https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard |
| 2 | Salesforce xlam-function-calling-60k | 60,000 (query, function_call) pairs over 3,673 APIs | CC-BY-4.0 | No native drift; use as positives + inject synthetic negatives | https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k |
| 3 | NVIDIA When2Call | Built on BFCL + xlam | Apache-2.0 | **Yes, direct.** 4-class `direct`/`tool_call`/`request_for_info`/`cannot_answer` | https://huggingface.co/datasets/nvidia/When2Call |
| 4 | API-Bank | 314 dialogues / 753 API calls (eval); 1,888 / 2,138 train | Open | Correctness labels only | https://huggingface.co/datasets/liminghao1630/API-Bank |
| 5 | glaive-function-calling-v2 | 113k | Open | All positives; need synthesis for negatives | https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 |

**Why BFCL v3 wins:** only public benchmark with native per-sample drift labels across 4 archetypes (wrong tool, wrong params, hallucinated tool, should-refuse). `possible_answer/` directory provides gold calls; deviation measurement is direct.

## Feature plan (22-dim, text-only)

Group A — **semantic alignment**:
- Embedding cosine (`all-MiniLM-L6-v2`) between preceding assistant NL turn and serialized `{tool_name + args_json}`
- BM25 overlap between prompt tokens and arg values
- Jaccard(user-utterance-nouns, arg-values)

Group B — **schema conformance**:
- JSON parses? (bool)
- Required args present? (count missing)
- Extra args beyond schema? (count)
- Arg type matches declared schema? (per-arg bool, aggregated)

Group C — **lexical drift**:
- Levenshtein(tool_name_called, nearest_tool_in_prompt)
- Verbatim-match rate: fraction of arg values appearing in prompt
- Action-verb alignment: extracted verb from NL vs tool-name prefix

Group D — **structural**:
- Arg-count z-score vs schema mean
- Nested object count
- Placeholder values (`<user_input>`, `"example"`, `null`-strings)

Same logistic-regression + StandardScaler pipeline as hallucination and refusal instruments.

## Null-result heuristics to falsify first

| Heuristic | Expected AUC | Kill-criterion |
|---|---|---|
| Uniform random | 0.50 | — |
| Exact string match (tool ∈ tools mentioned) | ~0.65 | if ≥ 0.90, pivot |
| JSON-parses-only | ~0.55 | baseline floor |
| spaCy NER overlap | ~0.70 | narrow but publishable |
| Embedding cosine alone | ~0.80 | — |

**Expected failure modes:**
- Paraphrase-heavy queries ("Big Apple" → `search("New York")`) look like drift but aren't. Need embedding fallback.
- BFCL `irrelevance` class is probably too easy (~0.95 with simple tricks). Report stratified metrics.
- Multi-turn drift (correct call but wrong turn) is invisible to single-call classifier. Scope out for v1.

## Prior art to beat

| Paper | AUC | Method | URL |
|---|---|---|---|
| Healy et al. 2026 "Internal Representations as Indicators of Hallucinations in Agent Tool Selection" | **0.716–0.721** recall 0.86 | Final-layer activations + mean pooling on Glaive 2,411 samples | https://arxiv.org/abs/2601.05214 |
| TRAJECT-Bench 2025 | No AUC | Trajectory-level diagnostics | https://arxiv.org/abs/2510.04550 |
| BFCL (Patil et al.) | Accuracy only | AST + executable match | https://github.com/ShishirPatil/gorilla |
| ToolMind 2025 | N/A | GPT-4 judge meta-verification | https://arxiv.org/html/2511.15718v2 |

**Headroom:** Healy et al. 0.72 is the only directly comparable. A text-only calibrated detector hitting AUC > 0.85 on BFCL v3 with no model internals would be a clear contribution. 0.95+ would cleanly outperform while being closed-model compatible.

## Data availability

- CPU in 1 day? Yes for bootstrap. BFCL v3 is ~5MB JSON. No model inference needed if using `possible_answer/` gold + mutation-based negatives.
- Easiest bootstrap: BFCL `irrelevance` + `multi_turn_miss_param` + `multi_turn_miss_func` = ~1,400 labeled drift/no-drift pairs, zero inference.
- GPT-4 judge: only needed for validating mutation quality on ~200 samples.

## Milestone plan

**Day 1 — Dataset + null results.** Download BFCL v3 + `possible_answer/`. Generate mutation-based negatives (swap args, rename tool, drop required field). Run all null heuristics. Land AUC table. Kill if exact-string-match ≥ 0.90.

**Day 2 — 22-feature extractor + calibrated classifier.** Logistic regression + Platt scaling. 5-fold CV on BFCL. Publish per-stratum AUC (`irrelevance` vs `miss_param` vs mutation-negatives). Target: AUC ≥ 0.90 on mutation-negatives, ≥ 0.85 on BFCL irrelevance.

**Day 3 — Cross-substrate validation.** Apply frozen detector to xlam-60k (synthetic negatives) and When2Call (4-class → binary). Confirm no drop > 0.05. If holds, v6.0-shippable as `styxx.drift.drift_check()`.

**Day 4–5 (optional hardening):** GPT-4-judge 200-sample spot check. Agent-framework integration examples (LangChain `AgentExecutor`, DSPy `ReAct`). Write-up.

## Decisions made under ambiguity

- Excluded Nexus/NexusRaven: eval set is 21 samples — too small.
- Excluded ToolBench (OpenBMB): no ground-truth drift labels, known quality issues.
- Chose BFCL over xlam primary because xlam has no natural negatives — synthesis inflates apparent AUC.

## When to implement

**Precondition:** v5.1 has external traction signal (stars, PR merges, reproductions). Don't build instrument #3 until instrument #2 has ≥ 1 user story.

**Target release:** v6.0 — "three calibrated cognometric instruments." With three instruments, Law II has triangulated support across hallucination (probe+classifier), refusal (classifier), and drift (classifier) — a stronger methodological claim than two instruments.
