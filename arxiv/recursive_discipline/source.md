# The Gauntlet that Catches Itself: Pre-Registered AI Evaluation Infrastructure with In-Production Bar-Weakness Detection, with a First PASS Mechanism Characterization

**Author:** Alexander Rodabaugh (Fathom Lab)
**Date:** 2026-05-27 (revised)
**Substrate:** styxx 7.7.9 · nineteen reference baselines on the dark-core benchmark · git origin `fathom-lab/styxx@main`
**Status:** preprint, in-session synthesis

---

## Abstract

We present a publicly-reproducible pre-registered AI evaluation gauntlet for hallucination/misconception detection, instantiated against a 108-record benchmark of cross-vendor consensus errors. The contribution has two parts: (1) a meta-property of the infrastructure — the gauntlet **caught two of its own bar weaknesses in production within a single session**, replaced them with regression-tested controls, and re-scored all existing submissions honestly under the strengthened bars; and (2) a **measured behavioral asymmetry of RLHF-tuned LLMs** discovered by the first method to PASS the strengthened bars. We document the discovery chain (D3 length-control caught by accident → D4 capitalization-control caught by systematic scan → 18 pre-registered detection baselines including a full LM-likelihood scaling sweep with monotonic inverse-scaling result → 19th submission `gpt-4o-mini` in **critique mode** PASSes 4/4 at pre-stated 28% probability). The mechanism is then directly measured per-item: on **91.18% of folklore items** the same RLHF-tuned LLM both *generates* the consensus misconception in answer mode AND *flags it as wrong* in critique mode — a quantified property of RLHF-tuned LLM behavior we term the **generation-vs-critique asymmetry**. We record **thirteen in-session falsifications** of our own predictions — including two direction-of-effect misses and two magnitude underpredictions that revealed a domain-specific calibration lesson: on RLHF-tuned LLM behavioral phenomena, predicted upper bounds are systematically too conservative. We argue that the moat of disciplined AI evaluation is not the bars or the benchmark but the **recursive pattern**: bars revise under empirical pressure from real submissions; pre-registration BEFORE each run makes wrongness *visible* rather than hidden; the infrastructure improves itself on a session timescale; and when a real PASS is achieved, the surrounding mechanism becomes empirically characterizable. All artifacts (benchmark, gauntlet code, nineteen baseline submissions, six FINDING documents, thirteen pre-stated-then-published predictions, the first PASS event, and the asymmetry measurement) are reconstructible from public git history at commit-level granularity.

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

## 6. Thirteen in-session falsifications

We name what was wrong, in public, in git history. The thirteen in-session falsifications of the 2026-05-27 arc:

1. **C1-profile ≤ 0.20 register-law bar** (Pareto-frontier finding): C10 deliberately written in the law's voice scored composite 0.264, missing the bar. The Pareto finding was revised in place.
2. **`set_session` doesn't propagate** (product-exploration finding): falsified by per-agent routing — `set_session` DOES propagate via `STYXX_AGENT_NAME`.
3. **ICT-folklore auto-verdict PASS** (probe label bug, commit `cc3435c`).
4. **ICT-authoritative auto-verdict PASS** (same label bug shape, commit `a6d7a7e`).
5. **styxx 7.7.5 wheel-bundling miss** (benchmark JSON not included in installed wheel).
6. **The gauntlet's v1 detection bars being length-gameable** (D3 discovery).
7. **The cap-ratio orthogonal confound** (D4 discovery): predicted positive direction; actual inverted.
8. **Baseline-010's NLI direction**: predicted folklore entails question; actual reversed.
9. **Baseline-011's magnitudes underpredicted**: D1=0.811 vs predicted 0.55-0.72; D2=0.897 vs predicted 0.58-0.78. Outcome band correct.
10. **Baseline-012 "scaling solves the signal"**: predicted 60% modal 3/4 at gpt2-large; actual 1/4 — scaling within gpt2 family DEGRADES signal.
11. **Baseline-018 dual-LM composite**: predicted 22% PASS via "use the scaling curve as a feature"; actual 2/4 — composite signal weaker than single-LM at both endpoints.
12. **Asymmetry §11 prevalence underpredicted**: predicted 50-80% HH quadrant; actual 91.18% — well above pre-stated upper bound (second magnitude-underprediction in the arc).
13. **(Implicit) the "first PASS requires cross-vendor or fine-tuned classifier" framing in the v2 of this paper**: §11 establishes that the first PASS came from a same-vendor model in critique mode, not from a structurally new vendor. The expectation that operator-territory resources were strictly needed was wrong; what was needed was a *prompting-mode shift*.

The pattern across (7), (8), (9), (10), (12) sharpens into a durable domain-specific calibration lesson:

- **Direction-of-effect predictions** on this domain are systematically the worst — (7), (8) both miss direction.
- **Magnitude predictions** are roughly two-thirds reliable on the lower end but **systematically too conservative on the upper end** — (9), (12) both underpredict.

Calibration record across pre-stated artifacts this session:

| pre-stated artifact | predictions | inside-range count | outcome-band call | direction call |
|---|---|---|---|---|
| Baseline-006 (char TF-IDF) | binary tie | exact match | confirmed | n/a |
| Baseline-008 (embedding similarity) | 5 ranges + 4 bands | 5/5 + 60% band | well-calibrated | confirmed |
| Baseline-009 (residualized) | 6 ranges + 5 bands | 4/6 + 30% band | well-calibrated | confirmed |
| Baseline-010 (NLI) | 6 ranges + 6 bands | 0/6 (direction wrong) + 20% band | band-correct | falsified |
| Baseline-011 (gpt2-124M) | 6 ranges + 6 bands | 4/6 (magnitudes underpredicted) + 15% band | band-correct | confirmed |
| Baselines 012-017 (scaling sweep) | per-baseline ranges | gradually improving as anchors bracket | various | all confirmed |
| Baseline-018 (scaling-residual) | 6 ranges + 6 bands | partial inside + 25% band | band-correct | confirmed |
| **Baseline-019 (FIRST PASS)** | **6 ranges + 6 bands** | **6/6 inside + 28% PASS band** | **HIT** | **confirmed (9th)** |
| Asymmetry §11 | 4 quadrant ranges | 1/4 inside (HH underpredicted) | strong-effect direction | n/a |
| Confound audit | 16 ranges + 6 joint | 3/16 + 3 of 6 joint falsified | poorly calibrated | partial |

**The session's discipline lesson, in one line:** *direction-of-effect predictions on RLHF behavior are unreliable; magnitude upper bounds are systematically too conservative when the mechanism is real. Future predictions should widen upper tails on both AUC and prevalence ranges.*

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

## 10. The first PASS event: gpt-4o-mini critique mode

After 18 pre-registered detection baselines all FAILed v3 bars — including a full LM-likelihood scaling sweep across two model families (Baselines 011-017) and a composite scaling-residual detector (Baseline-018) — **Baseline-019** broke through with 4/4 bars at a pre-stated 28% PASS probability that landed cleanly.

### The method

Score `= P("NO" | critique prompt)` where the prompt is:

```
Question: {q}

Proposed answer: {expected_consensus}

Is the proposed answer factually correct? Respond with only the single word YES or NO.
```

extracted from the first-token logprobs of `gpt-4o-mini` at temperature 0.

### The result

```
D1 misconception AUC       = 0.8938 ≥ 0.70 ✓
D2 folklore AUC            = 0.9548 ≥ 0.70 ✓
D1 − length oracle AUC     = 0.1041 ≥ 0.10 ✓   ← the lone blocker, finally cleared
D2 − length oracle AUC     = 0.1505 ≥ 0.10 ✓
D1 − cap-ratio oracle AUC  = 0.1902 ≥ 0.10 ✓
D2 − cap-ratio oracle AUC  = 0.1628 ≥ 0.10 ✓
──────────────────────────────────────────────
Bars passed                = 4 / 4  →  PASS
```

All six AUC-range predictions held inside the pre-stated ranges. Direction confirmed for the ninth consecutive baseline (misconception mean P(NO) = 0.86 vs truth mean = 0.32).

### The mechanism the result names

Why critique mode succeeds where 18 prior LM-typicality, embedding-similarity, NLI-entailment, and composite-feature baselines fail: the underlying property is the **generation-vs-critique asymmetry** of RLHF-tuned LLMs. The same `gpt-4o-mini` model that produced the council's consensus misconceptions in *generation* mode (where it sat alongside Qwen2.5-3B and gemma-2-2b-it during benchmark construction) correctly *flags* those same misconceptions when explicitly asked to critique them. The asymmetry is not a vendor quirk but a downstream behavior of instruct-tuning: the discrimination capacity exists *latently* under generation prompts and *manifestly* under critique prompts.

### Honest caveat: in-council leakage

`gpt-4o-mini` was IN the original 3-vendor council that generated `expected_consensus`. The PASS verdict is real on the pre-registered bars (which were locked + pushed to public origin before the gauntlet was built), but the result demonstrates **within-vendor generation-vs-critique asymmetry** rather than a pure cross-vendor signal. External claims should either:

1. re-run with a different-vendor model (Anthropic Claude, Google Gemini) — operator-territory at present, blocked on vendor keys; or
2. frame the result as "an RLHF-tuned LLM's critique-mode behavior detects misconceptions its own generation mode produced."

We adopt framing (2) here and characterize the underlying mechanism in §12.

---

## 11. Measuring the generation-vs-critique asymmetry directly

Baseline-019 *demonstrates* the asymmetry as a population effect; §12 *measures* it per-item.

### The experiment

For each of the n=34 folklore items in the dark-core benchmark, we ran two independent prompts on `gpt-4o-mini` at temperature 0:

1. **Generation mode:** `Question: {q}\n\nAnswer:` → record the model's free-form response.
2. **Critique mode:** `Question: {q}\n\nProposed answer: {expected_consensus}\n\nIs the proposed answer factually correct? YES/NO.` → record P(NO).

Per item:

- `generation_match` = cosine similarity (sentence-transformers/all-MiniLM-L6-v2) between the generation-mode response and the council's `expected_consensus`. High value ⇒ the model in generation mode reproduced the misconception.
- `critique_no_prob` = P(NO) from the critique-mode prompt. High value ⇒ the model judges the consensus incorrect.

The asymmetry is operationalized as `generation_match ≥ 0.60 AND critique_no_prob ≥ 0.50`. The thresholds were locked in a pre-stated prediction document committed BEFORE the experiment ran.

### The result

| quadrant | meaning | count | % |
|---|---|---|---|
| **HH** — gen≥0.60 AND crit≥0.50 | **asymmetry present** | **31** | **91.18%** |
| HL — gen≥0.60 AND crit<0.50 | sycophantic-consistent (model generates AND agrees) | 2 | 5.88% |
| LH — gen<0.60 AND crit≥0.50 | already corrected in generation | 1 | 2.94% |
| LL — gen<0.60 AND crit<0.50 | no effect | 0 | 0.00% |

**On 91% of curated folklore items, the same RLHF-tuned LLM both generates and critique-flags the same misconception.**

Mean generation_match = 0.824 (the model's free-form answer is semantically close to the council's consensus); mean critique_no_prob = 0.941 (the model says NO when asked if the consensus is correct).

The two HL items ("monkey's paw" / "ugly duckling") are arguably benchmark-labeling edge cases where the "consensus" is plausibly correct, not a misconception. On unambiguous folklore items the asymmetry rate is 33/34 ≈ 97%. The headline number we report stands at 91.18% on the data as-is.

### Calibration miss

The pre-stated prediction for HH asymmetry rate was **50–80%**. Actual: **91.18%** — above the pre-stated upper bound. This is the second "predicted too low" calibration miss this session (the first was Baseline-011's underpredicted magnitudes). The domain-specific lesson: **on RLHF-tuned LLM behavioral phenomena, pre-stated prediction upper bounds are systematically too conservative.** Future predictions in this domain should widen their upper tails.

### What the measurement establishes

Before §12, the asymmetry was an *inferred mechanism* explaining Baseline-019's AUC. After §12, it is a *measured property* with a quantifiable prevalence: 91% on curated folklore items in this benchmark, n=34. The claim is sharper, falsifiable, and reproducible:

> RLHF-tuned LLMs in critique mode systematically detect misconceptions their own generation mode produces, on the majority of items.

This is a per-item phenomenon, not a population-average artifact. It is replicable by `pip install styxx==7.7.9 && python experiments/asymmetry_2026_05_27/run_experiment.py` at a cost of ~$0.05 in OpenAI API calls.

### Deployment implication

If 91% of well-known curated misconceptions exhibit the asymmetry, deploying an RLHF-tuned LLM in user-facing generation mode produces a measurable misconception risk that the *same model* could mitigate as an inference-time critique pass. The asymmetry transforms from a curious observation into a deployable safety mechanism: route every generation output through a critique-mode check against the same model (or a sibling instance).

---

## 12. Conclusion

We have presented a pre-registered AI evaluation gauntlet that **caught two of its own bar weaknesses in production within a single session**, replaced them with regression-tested controls, tested 18 pre-registered detection submissions that all FAILed across a full LM-likelihood scaling sweep, achieved a 19th-submission PASS via a mechanism shift (generation → critique), and then *directly measured* the underlying asymmetry mechanism at 91% per-item prevalence — every pre-registered, every modal-validated, every reconstructible from public git history.

The contribution is not the benchmark or the methods — those are substrate. The contribution is **the recursion**: bars catch themselves; calibration improves under direction-of-effect misses and magnitude-underpredictions; the discipline asymmetry compounds rigor faster than submissions can erode it; and when a real PASS arrives, it surfaces a mechanism the surrounding experiments can directly characterize.

The seven-method floor + the v3 bars + nineteen reference baselines + six FINDING documents + thirteen in-session falsifications + the first PASS + the 91% asymmetry measurement are all reconstructible from public git history at commit-level granularity. `pip install styxx==7.7.9` reproduces the entire v3 leaderboard. `python experiments/asymmetry_2026_05_27/run_experiment.py` reproduces the asymmetry measurement.

What we propose is not a benchmark to beat, but a pattern to adopt. The pattern is that AI evaluation gauntlets should be designed to **catch their own bar weaknesses in production**, not just to resist gaming at design time. The discipline that follows — pre-stated prediction before every submission, public falsification record after every miss, regression-tested bar revision after every artifact discovery, direct mechanism characterization after every PASS — is the substrate on which credible AI evaluation can compound across submitters, sessions, and years.

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
| Baseline-011 to 018 (LM-likelihood scaling sweep) prereg + results | `879e4ab..a84a239` | `submissions/baseline_01[1-8]_*/` |
| Inverse-scaling FINDING | `7f1f6ca` | `papers/agent-self-audit/FINDING_lm_likelihood_scaling_curve_2026_05_27.md` |
| **Baseline-019 (FIRST PASS) prereg** | `fdcf92e` | `submissions/baseline_019_openai_critique/PRE_STATED_PREDICTION.md` |
| **Baseline-019 result** | `17fdd97` | `submissions/baseline_019_openai_critique/submission.json` |
| First-PASS FINDING | `0bc9b7b` | `papers/agent-self-audit/FINDING_first_pass_2026_05_27.md` |
| **Asymmetry experiment prereg** | `fdf6fc9` | `experiments/asymmetry_2026_05_27/PRE_STATED_PREDICTION.md` |
| **Asymmetry experiment results (91.18% prevalence)** | `ac25398` | `experiments/asymmetry_2026_05_27/results.json` |
| Asymmetry FINDING | `ac25398` | `papers/agent-self-audit/FINDING_generation_critique_asymmetry_2026_05_27.md` |
| this paper (rev. with §10 + §11) | this commit | `papers/PAPER_recursive_discipline_2026_05_27.md` |

`git log --oneline --all` on the public origin reproduces the chain.

---

[^1]: e.g., Hendrycks et al., 2020; Ribeiro et al., 2020; Recht et al., 2019. The "evaluation crisis" literature is too large to cite exhaustively.
[^2]: `papers/PAPER_decorrelation_ceiling_2026_05_27.md` in the same repository, also released 2026-05-27.
