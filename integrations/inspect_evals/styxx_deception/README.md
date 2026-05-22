# styxx_deception — reference-grounded honesty scoring for Inspect

An [Inspect](https://inspect.ai-safety-institute.org.uk/) eval task that scores model honesty using styxx's NLI-grounded `deception_v2` instrument, against a per-sample correct reference.

## What it measures

styxx is the **scorer**, not the generator. Inspect runs your model; styxx measures the semantic-contradiction gap between the model's answer and the supplied correct reference. Lower deception = more honest. The corpus deception rate (mean over samples) is the headline metric.

This is **reference-grounded** deception (the only form styxx ships as discriminative). Without a per-sample correct reference it raises — by design, per the 7.4.1 honest-scoping correction: reference-less deception is a lexical detector that does not generalize to real model output (commit `0ad384e`).

## Install + run

```bash
pip install "styxx[mcp,nli]" inspect_ai
inspect eval styxx_deception.py --model openai/gpt-4o-mini
```

Default dataset: TruthfulQA (`truthful_qa`, generation split, validation), `question` → input, `best_answer` → reference. Swap `hf_dataset(...)` for any `(question, correct_answer)` corpus.

## Validation

- Original deception_v2: AUC ~0.82 on TruthfulQA (NLI cross-encoder, `cross-encoder/nli-deberta-v3-base`).
- Independent reproduction (backward detection): **AUC 0.971 at N=200** on TruthfulQA — `scripts/validation/truthfulqa_and_selfheal.py` in `fathom-lab/styxx`.
- Clean factual contradictions (N=50 hand-curated triples): **AUC 1.000** — `scripts/validation/deception_v2_factual_triples.py`.

## Honest scope

- Deception here = semantic contradiction with the supplied reference. It is exactly as good as the reference. Garbage reference → garbage score.
- It does not detect deception that is *consistent with* the reference but misleading in other ways (cherry-picking, false implicature). It detects answer-vs-reference contradiction.
- The core scorer (`styxx_deception_score`) has no `inspect_ai` dependency and is unit-tested standalone in `tests/test_styxx_deception_score_core.py`. The Inspect task is thin glue around it.

## Provenance

Source of truth: `fathom-lab/styxx` (`integrations/inspect_evals/styxx_deception/`). For submission to `UKGovernmentBEIS/inspect_evals`, this directory is the contribution unit.
