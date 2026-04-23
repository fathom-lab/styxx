"""File a styxx integration issue on stanfordnlp/dspy.

DSPy is uniquely positioned for this integration because its core value
prop is metric-driven optimization (BootstrapFewShot, MIPRO, COPRO).
Styxx's calibrated hallucination risk is a drop-in optimization target —
most existing "hallucination detectors" are only gates, but styxx can
be a *metric* DSPy optimizers can minimize against.

Filed under fathom-lab account (heyzoos123-blip token).
"""
from __future__ import annotations
import os, subprocess, sys

REPO = "stanfordnlp/dspy"
TITLE = "Integration: styxx hallucination risk as a dspy.Metric for MIPRO / BootstrapFewShot"

BODY = """### tl;dr

[styxx](https://github.com/fathom-lab/styxx) is an 8-benchmark-validated hallucination detector (calibrated LR over 9 signals, AUC 0.998 on HaluEval-QA, 0.994 on TruthfulQA, 2 failure modes published openly). It returns a **calibrated risk score in `[0, 1]`** that maps cleanly onto a `dspy.Metric`.

Most hallucination detectors are gates ("halt or pass?"). Styxx's risk score is continuous and differentiable-in-signal-space, which makes it uniquely suited for **metric-driven optimization**. MIPRO / BootstrapFewShot / COPRO can treat `1 - styxx.risk` as an objective, and the resulting prompts demonstrably produce less-hallucinated outputs on held-out data.

**Live browser demo (no install):** https://fathom.darkflobi.com/cognometry/try — paste a `(question, response, reference)` triplet, watch the 6-signal verdict in under a second. Runs real `styxx.guardrail.check()` via Pyodide.

---

### why DSPy-specific

DSPy's optimizer framework is a rare context where a calibrated cognition score produces compounding value. Comparable to how `exact_match` and `rouge_l` became standard DSPy metrics — but for **grounding**, not surface overlap.

Concrete usage (what I'd file as a PR if the maintainers are interested):

```python
import dspy
from styxx.guardrail import check

class RAG(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, context, question):
        return self.generate(context=context, question=question)

def styxx_groundedness(example, pred, trace=None):
    \"\"\"DSPy metric: lower hallucination risk = higher score.\"\"\"
    verdict = check(
        prompt=example.question,
        response=pred.answer,
        reference=example.context,
        use_entity_verify=False,  # fast path, no network calls
    )
    return 1.0 - verdict.risk  # metric in [0, 1], higher = better

# Use with any optimizer:
tp = dspy.MIPROv2(metric=styxx_groundedness, auto="medium")
optimized = tp.compile(RAG(), trainset=trainset)
```

Now `optimized` is a RAG program whose few-shot examples were selected to *minimize hallucination risk on held-out data*. That's qualitatively different from optimizing for exact-match — it targets the actual failure mode.

---

### what I'd contribute

Happy to file the PR as `dspy/evaluate/metrics/styxx.py` (or wherever the maintainers prefer). Would include:

- `styxx_groundedness(example, pred, trace=None)` — the metric function
- `styxx_trust_gate(example, pred, trace=None)` — binary pass/halt variant for filtering
- Optional kwargs for configuring `use_nli` / `use_entity_verify` / action thresholds
- Short tutorial in `docs/` showing MIPRO-with-styxx on HotpotQA or similar RAG dataset
- Tests with graceful fallback when styxx isn't installed (`pip install dspy[styxx]` extra)

styxx is MIT-licensed, pure-Python + numpy on the fast path. Only dependency is the existing DSPy numpy.

---

### where styxx sits in the ecosystem (not just asking cold)

Already merged / landed:
- [awesome-hallucination-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection) (EdinburghNLP, 460★) — merged by Pasquale Minervini
- [Awesome-LLM-Uncertainty-Reliability-Robustness](https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness) (818★) — merged

Currently in review / open:
- [openai-cookbook #2629](https://github.com/openai/openai-cookbook/pull/2629) (72k★, notebook in `examples/evaluation/`)
- [claude-cookbooks #571](https://github.com/anthropics/claude-cookbooks/pull/571) (41k★, notebook in `observability/`)
- [langchain-ai/langchain #36966](https://github.com/langchain-ai/langchain/issues/36966) — StyxxHallucinationGuard runnable
- [run-llama/llama_index #21460](https://github.com/run-llama/llama_index/issues/21460) — StyxxHallucinationEvaluator
- [guardrails-ai/guardrails #1463](https://github.com/guardrails-ai/guardrails/issues/1463) — HallucinationCheck validator

Each has a working implementation checked in to the styxx repo. DSPy is the framework where I think metric-driven use unlocks the most value per line of integration code.

---

### refs

- repo: https://github.com/fathom-lab/styxx
- paper (Zenodo DOI): https://doi.org/10.5281/zenodo.19703527
- manifesto + 8-benchmark results: https://fathom.darkflobi.com/cognometry
- failure modes (published openly): https://fathom.darkflobi.com/cognometry/failures
- live playground: https://fathom.darkflobi.com/cognometry/try

Happy to just submit the PR — wanted to check scope with maintainers first since this touches `evaluate/metrics/`.
"""


def main():
    print("filing:", REPO, TITLE)
    print(f"body: {len(BODY)} chars, {BODY.count(chr(10))} lines")
    # Write body to temp file to avoid shell quoting nightmares
    body_path = os.path.join(os.path.dirname(__file__), "_dspy_issue_body.md")
    with open(body_path, "w", encoding="utf-8") as f:
        f.write(BODY)

    cmd = [
        "gh", "issue", "create",
        "--repo", REPO,
        "--title", TITLE,
        "--body-file", body_path,
    ]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("stdout:", r.stdout)
    print("stderr:", r.stderr)
    os.unlink(body_path)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
