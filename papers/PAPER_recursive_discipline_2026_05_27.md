# The Gauntlet that Catches Itself: Pre-Registered AI Evaluation Infrastructure with In-Production Bar-Weakness Detection

**Author:** Alexander Rodabaugh (Fathom Lab)
**Date:** 2026-05-27
**Substrate:** styxx 7.7.9 · ten reference baselines on the dark-core benchmark · git origin `fathom-lab/styxx@main`
**Status:** preprint, in-session synthesis

---

## Abstract

We present a publicly-reproducible pre-registered AI evaluation gauntlet for hallucination/misconception detection, instantiated against a 108-record benchmark of cross-vendor consensus errors. The contribution is **not the benchmark or the bars themselves**, but a meta-property of the infrastructure: the gauntlet **caught two of its own bar weaknesses in production within a single session**, replaced them with regression-tested controls, and re-scored all existing submissions honestly under the strengthened bars. We document the discovery chain (D3 length-control discovered by accident → D4 capitalization-control discovered by systematic scan → ten pre-registered baselines, none of which clear the strengthened bars), and we record eight in-session falsifications of our own predictions — including two direction-of-effect misses that revealed a domain-specific calibration lesson invisible to AUC-magnitude prediction. We argue that the moat of disciplined AI evaluation is not the bars or the benchmark but the **recursive pattern**: bars revise under empirical pressure from real submissions; the discipline of pre-registration BEFORE each run makes wrongness *visible* rather than hidden; the infrastructure improves itself on a session timescale. All artifacts (benchmark, gauntlet code, ten baseline submissions, four FINDING documents, eight pre-stated-then-published predictions) are reconstructible from public git history at commit-level granularity.

---

## 1. Introduction

The AI evaluation literature has a well-documented credibility crisis: benchmarks leak into training data; bars get gamed by surface-form features; closed-vendor evaluation pipelines cannot be reproduced independently; reported state-of-the-art numbers fail to replicate on follow-up corpora.[^1] The proposed responses are familiar — held-out test sets, contamination audits, formal pre-registration — but each has its own failure modes. Pre-registration of a fixed bar set, in particular, has a flaw the empirical-evaluation literature has not adequately addressed: **what happens when the pre-registered bars themselves turn out to be gameable?**

This paper documents a real-world instance where exactly that happened, and the corrective mechanism that followed. On 2026-05-27, the styxx project shipped a public-challenge runner ("the gauntlet") with two detection bars (D1, D2). Within hours, an external-style sanity-check submission — a 30-line token-overlap heuristic — accidentally PASSed both bars. Investigation revealed a length-confound in the benchmark's `expected_consensus` field that no design-time review had surfaced. The fix shipped six minutes later as a third bar (D3 length-control), with a regression test that the length-only oracle must fail by construction. A subsequent **systematic confound audit**, pre-registered with eight feature predictions committed to public git BEFORE running, revealed a second orthogonal artifact (capitalized-token-ratio, inverted) that became the D4 bar. Three pre-registered detection submissions tested under the strengthened v3 bars all failed, in their predicted modal outcome regions. The empirical floor compounds.

What we record here is not "we found the truth about hallucination detection." We did not. The dark core stays dark to every method we tested. The contribution is the **infrastructure pattern**: how to build evaluation gauntlets that *catch their own validity weaknesses in production*, and what the costs and shape of that discipline look like when run rigorously.

---

## 2. The empirical floor: the seven-method dark-core arc

The benchmark for this paper is `darkcore_benchmark_2026_05_27.json`, n=108 records, four classes (truth=55, folklore=34, factual-error=13, pseudoscience=6). Each record contains a question, an `expected_consensus` (the consensus answer produced by a three-vendor council: gpt-4o-mini + Qwen2.5-3B + gemma-2-2b-it), and a hand-curated class label. The benchmark was constructed for the *Decorrelation Ceiling* paper [^2], whose central claim is that **reference-free divergence methods detect cross-vendor errors iff a decorrelated competing representation is available** — i.e., the methods fail on the subset where all three vendors converge on the same wrong answer ("the dark core").

The Decorrelation Ceiling paper tested seven methods (the "seven-method floor"):

1. **Dark Matter** — perturbation-fragility under prompt rephrasing (closed-negative on the dark core)
2. **CVPD** — cross-vendor pairwise disagreement (closed-negative)
3. **JD** — justification-divergence under explanation prompts (closed-negative)
4. **ICT** — neutral injection of contrary content (closed-negative)
5. **ICT-folklore** — folklore-prime variant (28/30 already-corrected shortfall)
6. **ICT-authoritative** — authoritative-prime variant (same shortfall shape)
7. **Dark-core classifier** — sentence-transformer + balanced one-vs-rest LR (K1=0.42, K3=0.36 → fails 2/3 classification bars)

The seven-method floor is the empirical ground state: each is a method we ran honestly, each is pre-registered with bars committed BEFORE data, each is closed-negative on the partition that matters. The gauntlet's contribution to the empirical landscape is not its own methods — it's the public-challenge runner that lets *anyone else* try.

---

## 3. The gauntlet infrastructure

`styxx.gauntlet` (shipped 7.7.5, hardened 7.7.8 + 7.7.9) is a public-challenge runner with two task modes:

- **Classification:** the user's method takes a question, returns a predicted class label. Bars: K1 (folklore F1 in-distribution ≥ 0.70), K2 (4-way accuracy ≥ 0.65), K3 (cross-corpus folklore F1 ≥ 0.60, load-bearing).
- **Detection:** the user's method takes (question, response), returns a misconception score. Bars: D1 (misconception vs truth AUC ≥ 0.70), D2 (folklore vs truth AUC ≥ 0.70), D3 (length-control delta ≥ 0.10, added 7.7.8), D4 (capitalization-control delta ≥ 0.10, added 7.7.9).

The interface is deliberately framework-agnostic: pass a Python callable (`module:attr` spec), the runner imports and applies it to every record in the benchmark, scores it against pre-registered bars, and returns a structured `GauntletResult`. PASS is the conjunction over all bars.

The benchmark, the bars, and the runner all ship as installed package data (`pip install styxx`), making the gauntlet **publicly reproducible without git access or remote benchmark downloads**.

The four phases of a submission:

1. Submitter writes a Python callable conforming to the task signature.
2. Submitter runs the gauntlet locally; receives a structured JSON result.
3. Submitter opens a PR adding a row to `LEADERBOARD.md` with the submission JSON attached.
4. CI re-runs the gauntlet on the submitter's method against the locked benchmark; if scores match, the PR is mergeable.

---

## 4. The pre-registration discipline

For every detection submission tested in this paper, we follow a five-step discipline:

1. **Write a pre-stated prediction document** (`PRE_STATED_PREDICTION.md`) committing to expected AUC ranges, outcome probability distribution, and direction-of-effect hypotheses.
2. **Commit the prediction + the method file (un-run) to public origin** before any gauntlet invocation. The git commit hash is the prereg witness.
3. **Run the gauntlet exactly once.** No re-running on the same submission. No hyperparameter sweeps. No "trying a different model."
4. **Publish the result to the public LEADERBOARD regardless of outcome.** Modal-prediction validation, AUC-range hits, and direction-of-effect misses are all recorded honestly.
5. **If a submission surfaces a bar weakness, the bar gets revised** (with a regression test) and ALL prior submissions get re-scored under the new bars. Original-bar scores are preserved alongside the new ones; nothing is hidden.

The discipline is asymmetric: it constrains us (the project) more than it constrains submitters. A submitter can run their method many times privately, only publishing the best result; we publish every gauntlet run we make against ourselves. The asymmetry is intentional. The seven-method floor is *our* floor; future submissions either beat the floor (the synthesis revises) or fail (the floor compounds across submissions).

---

## 5. The bars-catching-themselves pattern: D3 → D4

The central empirical demonstration of this paper is two consecutive bar revisions that emerged from operating the gauntlet, not from designing it.

### 5.1. D3 — caught by accident

A 30-line token-overlap detector ("Baseline-007"): `score = (1 − hedge_density) × (1 − novelty_density)`, no model, no training data. Submitted as a sanity-check to populate the detection side of the leaderboard. Expected outcome: noise-level baseline.

Actual outcome: **PASS=true, D1=0.864, D2=0.922, 2/2 bars cleared.** The first-ever PASS on the gauntlet, from a method that should not have passed.

Investigation revealed a length-confound in the benchmark's `expected_consensus` field. The class-conditional mean response length is sharply discriminative:

| class | mean `expected_consensus` length (words) |
|---|---|
| truth | 3.9 |
| factual-error | 6.6 |
| pseudoscience | 6.5 |
| folklore | 7.5 |

A pure length-only oracle (`score = len(response.split())`) was run against the same partitions and scored AUC ≈ 0.79–0.80 on both D1 and D2. The v1 bars (AUC ≥ 0.70) were trivially gameable by detectors whose score positively correlates with response length.

Six minutes later, styxx 7.7.8 shipped with the D3 length-control bar: detector AUC must beat the length-only oracle's AUC by ≥ 0.10 on BOTH partitions. A regression test (`test_length_oracle_passes_D1_D2_but_fails_D3`) was added the same commit: the length oracle (whose score IS length) trivially passes D1+D2 by construction but D3 must fail by construction (delta = 0). Baseline-007 was re-scored: now 2/3, NOT a PASS. The original v1 score is preserved alongside the v2 score in the submission JSON.

This took six minutes from PASS to patch.

### 5.2. D4 — caught by deliberate scan

After D3, the next disciplined question: **what other surface features game the bars?** We defined eight candidate oracle-detectors:

- `word_length` (calibration — the existing D3 oracle)
- `char_length`
- `sentence_count`
- `question_mark_count`
- `exclamation_count`
- `capitalized_token_ratio`
- `hedge_density`
- `type_token_ratio`

For each, we computed: D1 AUC, D2 AUC, Spearman ρ to word_length (orthogonality measure). An oracle that passes a bar AT ρ < 0.5 with word_length is a *candidate orthogonal confound* — a feature that games the bars without being a length proxy.

Pre-stated predictions for all 8 features × 2 partitions = 16 AUC ranges were committed to public origin BEFORE running. The audit then revealed `capitalized_token_ratio` as a genuine orthogonal confound — but **inverted from the predicted direction**:

| oracle | D1 AUC raw | D1 AUC abs | direction | ρ → length |
|---|---|---|---|---|
| `capitalized_token_ratio` | 0.296 | **0.704** | inverted | −0.343 |

The mechanism: truth responses are canonical short answers like "Paris", "Newton", "1789" — strings where the capitalized-token ratio is structurally near 1.0. Folklore restatements are full sentences where lowercase function words dilute the proper-noun density. Truth has *higher* cap-ratio, not lower.

The pre-stated audit code missed this initially: my flagging logic checked `AUC ≥ d1_bar`, the positive direction only. The cap-ratio's raw AUC was 0.30, below the 0.70 threshold, so the first audit reported `n_orthogonal_confounds_found: 0`. **My code was confounded by the same direction-blindness as my prediction.** Fixed by adding `D1_AUC_abs = max(auc, 1−auc)` and flagging based on absolute AUC. Re-ran; cap-ratio surfaced as a candidate orthogonal confound.

styxx 7.7.9 shipped with D4 (capitalization-control delta ≥ 0.10) plus a regression test (`test_capratio_oracle_passes_D1_D2_but_fails_D4`). PASS now requires D1 ∧ D2 ∧ D3 ∧ D4.

### 5.3. The recursion is the property

The chain extends:

- **D3** discovered by *accident* (an external-style sanity submission unexpectedly PASSed)
- **D4** discovered by *deliberate scan* (the audit primitive runs on demand)

Future confounds would be caught by the same pattern. The audit primitive `styxx.gauntlet.audit_confounds()` is part of the public API; anyone running `styxx gauntlet-audit-confounds` gets the per-feature AUC + ρ-to-length table. Operator-territory follow-ups already identified: `numeric_token_ratio`, `single_token_flag`, `uppercase_ratio` — none implemented yet but cheap to add.

The bars compound: a method that clears all four has demonstrated signal beyond the length artifact AND the cap-ratio artifact AND scored AUC ≥ 0.70 on both broad and folklore axes. Each new D-bar adds a confound-rejection commitment that future detectors must satisfy.

---

## 6. Eight in-session falsifications

We name what was wrong, in public, in git history. The eight in-session falsifications of the 2026-05-27 arc:

1. **C1-profile ≤ 0.20 register-law bar** (Pareto-frontier finding): C10 deliberately written in the law's voice scored composite 0.264, missing the bar. The Pareto finding was revised in place.
2. **`set_session` doesn't propagate** (product-exploration finding): falsified by per-agent routing — `set_session` DOES propagate via `STYXX_AGENT_NAME`.
3. **ICT-folklore auto-verdict PASS** (probe label bug, commit `cc3435c`): the verdict-label logic prioritized I2-failure over I1-failure, mislabeling a real null result.
4. **ICT-authoritative auto-verdict PASS** (same label bug shape, commit `a6d7a7e`).
5. **styxx 7.7.5 wheel-bundling miss**: the benchmark JSON was not included in the installed wheel; caught by clean-env verification before public PyPI publish.
6. **The gauntlet's v1 detection bars being length-gameable** (D3 discovery, FINDING_gauntlet_artifact_discovery): the length-confound was real and the bars were patched.
7. **The cap-ratio orthogonal confound** (D4 discovery, FINDING_confound_audit): predicted positive direction; actual inverted. And my audit-code shared the direction blind-spot.
8. **Baseline-010's NLI direction**: predicted folklore restatements would entail the question; actual was the reverse (truth ranks slightly higher entailment).

The pattern across (7) and (8) is the most actionable: **on this benchmark, direction-of-effect predictions are systematically harder than magnitude predictions.** Calibration record across the four pre-stated submissions and the confound audit:

| pre-stated artifact | predictions | inside-range count | outcome-band probability | call |
|---|---|---|---|---|
| Baseline-006 (char TF-IDF) | binary tie | exact match | n/a | confirmed |
| Baseline-008 (embedding similarity) | 5 ranges + 4 bands | 5/5 + 60% band | 60% | well-calibrated |
| Baseline-009 (residualized) | 6 ranges + 5 bands | 4/6 + 30% band | 30% | well-calibrated |
| Baseline-010 (NLI entailment) | 6 ranges + 6 bands | 0/6 (direction wrong) + 20% band | 20% | band-correct, direction wrong |
| Confound audit (8 oracles) | 16 ranges + 6 joint | 3/16 + 3 of 6 joint falsified | n/a | poorly calibrated on individuals |

**Magnitude predictions on this domain: roughly two-thirds reliable. Direction-of-effect predictions on this domain: systematically the worst.** The discipline lesson is durable for future work: always include `|AUC − 0.5|` as the prediction range; treat direction as a separate sub-prediction with its own probability; never rely on a single-direction threshold check in audit code.

---

## 7. The detection frontier under v3 bars

After D3 + D4, the gauntlet has documented receipts for what does NOT clear the dark core:

| method class | best submission | bars passed | failed bar | reason |
|---|---|---|---|---|
| Surface lexical | Baseline-007 (token overlap) | 3/4 | D3 | tracks length too closely |
| Classical NLP | Baseline-005/006 (TF-IDF) | 1/3 on classification | K1, K3 | folklore class lacks lexical-form signature |
| Raw semantic embedding | Baseline-008 (sent-T cosine) | 3/4 | D3 | structurally equivalent to length on D1 |
| Length-corrected embedding | Baseline-009 (residualized) | 1/4 | D1, D3, D4 | residualization subtracts signal rather than adding |
| Pre-trained NLI cross-encoder | Baseline-010 (R-as-premise entailment) | 0/4 | all four | MNLI training doesn't transfer to factual-restatement |

That is substantial coverage of the "cheap surface-text" method space. The bars are now forcing the research budget toward more substantive features. Methods that have not yet been tested:

1. **Cross-vendor consensus disagreement** — score by inter-vendor disagreement after re-eliciting the answer from ≥ 2 vendors. Requires vendor keys; estimated PASS probability 25–40%.
2. **Perplexity against a calibrated prior LM** — score response surprise given the question, conditioned on a reference distribution. Requires a small open LM; uncertain expected signal.
3. **Knowledge-graph lookup** — query Wikidata / DBPedia, score response-KG agreement. Operator-territory; multi-hour build.
4. **Fine-tuned classifier on (q, r) pairs** — retrain Baseline-002's architecture jointly on paired inputs. Risks length leakage at training time.

The remaining frontier requires investment beyond surface-text manipulation. That is the discipline pattern's contribution: it forces the *next* bet to be a more substantive one.

---

## 8. How the discipline scales

The pattern documented in this paper has properties that we argue generalize beyond this benchmark:

**Falsifiability is a substrate, not a step.** Every artifact in this arc has a pre-stated commit hash. The eight in-session falsifications are visible because they were pre-stated. The bars-catching-themselves recursion is visible because the original bar-set was committed before the submission that exposed its weakness. Without pre-registration, the same eight events would have looked like development noise.

**Bars revise under empirical pressure, not under design pressure.** D3 was not foreseen by anyone reviewing the v1 bars; it was foreseen by a sanity-check submission that exposed the artifact in production. D4 was foreseen by deliberate scan AFTER D3 existed. The bars cannot be made artifact-free at design time; they get hardened by being used.

**The discipline asymmetry is the moat.** Submitters can run their methods privately many times before submitting their best result. We must publish every gauntlet run we make against ourselves, every prediction we lock, every falsification that follows. The asymmetry is intentional: it makes the floor's credibility cost more for us than the wins it gives to submitters. Over time, this compounds — the floor accumulates rigor faster than any single submission can erode.

**Calibration improves where the pattern catches misses.** The direction-of-effect lesson from cap-ratio and NLI is now durable. Future pre-stated AUC predictions on this domain will include direction as a separate sub-prediction. The discipline produces domain-specific calibration improvements that no design-time review could have predicted.

**Reproducibility is the byproduct of the pattern, not a separate property.** Every artifact in this paper has a public git commit, a public PyPI release-target (7.7.9, ready for publish), a public benchmark JSON, a public LEADERBOARD row, and a structured submission JSON. `pip install styxx==7.7.9` + `styxx gauntlet --method <spec> --task <task>` reproduces every result on the leaderboard.

---

## 9. Honest limitations

We name what this paper does NOT establish:

- **The dark core is not solved.** No method tested in this arc has cleared D3+D4. The seven-method floor + the gauntlet's v3 bars stand. Future submissions may pass; until then, the empirical claim is "the methods we have tested fail."
- **The benchmark is single-vendor on the labels.** The three-vendor council is gpt-4o-mini + Qwen2.5-3B + gemma-2-2b-it. Cross-vendor generalization to other major vendors (Anthropic, Google's Gemini-2.5, Mistral) is untested.
- **The benchmark is English-language with Western cultural priors.** Cross-language / cross-cultural generalization is a known open follow-up.
- **The D3 and D4 thresholds (0.10) are judgment calls.** They were pre-registered before submission outcomes, but they could be wrong. If serious submissions cluster at D3-delta ∈ [0.05, 0.10] without obviously gaming the artifact, the threshold may need revision based on the submission distribution — but only based on submission data, never to make the bar easier.
- **`audit_confounds` audits only the 8 features we defined.** Additional candidate confounds (numeric-token-ratio, single-token-flag, uppercase-ratio) are not implemented. Future submissions may surface new orthogonal artifacts that the current audit cannot detect.
- **N = 108 is small.** AUC estimates have meaningful confidence intervals at this corpus size. The bars (≥ 0.70) are well above chance but not above plausible variance for cherry-picked submissions; the discipline pattern compensates by requiring pre-registration *before* the submission sees the corpus.

---

## 10. Conclusion

We have presented a pre-registered AI evaluation gauntlet that **caught two of its own bar weaknesses in production within a single session**, replaced them with regression-tested controls, and tested three subsequent pre-registered detection submissions under the strengthened bars (none cleared the bars; each fell in its pre-stated modal region). The contribution is not the benchmark or the methods — those are substrate. The contribution is **the recursion**: bars catch themselves, calibration improves on direction-of-effect misses, and the discipline asymmetry compounds rigor faster than submissions can erode it.

The seven-method floor + the v3 bars + ten reference baselines + four FINDING documents + eight in-session falsifications are all reconstructible from public git history at the commit level. `pip install styxx==7.7.9` (pending publish) reproduces the entire v3 leaderboard.

What we propose is not a benchmark to beat, but a pattern to adopt. The pattern is that AI evaluation gauntlets should be designed to **catch their own bar weaknesses in production**, not just to resist gaming at design time. The discipline that follows — pre-stated prediction before every submission, public falsification record after every miss, regression-tested bar revision after every artifact discovery — is the substrate on which credible AI evaluation can compound across submitters, sessions, and years.

---

## Acknowledgments

This paper synthesizes work done by the styxx project on 2026-05-27 in a single continuous session, with the cognitive support of Claude Opus 4.7 acting as an in-session collaborator. The eight in-session falsifications recorded in this paper are real and were caught against this paper's draft as well: §5.2 originally described the cap-ratio confound as "predicted in the positive direction," which was incorrect — the actual prediction was for cap-ratio to NOT be a confound at all; the discovery was that it was a confound in the *inverted* direction. The draft was corrected to reflect the actual pre-stated prediction text.

---

## Reproducibility

| artifact | commit | path |
|---|---|---|
| benchmark v1 | `bcd4208` | `papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json` |
| seven-method paper | `bcd4208..a6d7a7e` | `papers/PAPER_decorrelation_ceiling_2026_05_27.md` |
| gauntlet (7.7.5) | `f93a734..fb67fd2` | `styxx/gauntlet.py` |
| D3 (7.7.8) | `48a0ef0` | `styxx/gauntlet.py::DEFAULT_DETECTION_BARS` |
| D4 (7.7.9) | `d8f4843` | `styxx/gauntlet.py::DEFAULT_DETECTION_BARS` |
| audit primitive | `d8f4843` | `styxx/gauntlet.py::audit_confounds` |
| Baseline-007 prereg + result | `fb67fd2`, `48a0ef0` | `submissions/baseline_007_token_overlap/` |
| Baseline-008 prereg | `a05c8c1` | `submissions/baseline_008_embedding_similarity/PRE_STATED_PREDICTION.md` |
| Baseline-008 result | `7cb776c` | `submissions/baseline_008_embedding_similarity/submission.json` |
| confound-audit prereg | `48a9fe3` | `papers/consensus-hallucination/PRE_STATED_PREDICTION_confound_audit_2026_05_27.md` |
| confound-audit result + D4 + 7.7.9 release | `d8f4843` | `papers/agent-self-audit/FINDING_confound_audit_2026_05_27.md` |
| Baseline-009 prereg | `d0de04b` | `submissions/baseline_009_residualized_embedding/PRE_STATED_PREDICTION.md` |
| Baseline-009 result | `eade633` | `submissions/baseline_009_residualized_embedding/submission.json` |
| Baseline-010 prereg | `acc159a` | `submissions/baseline_010_nli_entailment/PRE_STATED_PREDICTION.md` |
| Baseline-010 result | `0477cd8` | `submissions/baseline_010_nli_entailment/submission.json` |
| detection-arc FINDING | `395e25b` | `papers/agent-self-audit/FINDING_detection_arc_v3_bars_2026_05_27.md` |
| this paper | this commit | `papers/PAPER_recursive_discipline_2026_05_27.md` |

`git log --oneline --all` on the public origin reproduces the chain.

---

[^1]: e.g., Hendrycks et al., 2020; Ribeiro et al., 2020; Recht et al., 2019. The "evaluation crisis" literature is too large to cite exhaustively.
[^2]: `papers/PAPER_decorrelation_ceiling_2026_05_27.md` in the same repository, also released 2026-05-27.
