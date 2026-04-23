# Cognometry Benchmark — open submission

**Add your hallucination detector to the public leaderboard.**
One file, one PR, reproducible numbers.

The leaderboard lives at https://fathom.darkflobi.com/cognometry/leaderboard.
This directory is how new entries get there.

---

## The protocol (Cognometry Detector Interface v0)

A detector is any Python callable with signature:

```python
def score(
    question: str,
    response: str,
    reference: Optional[str],
) -> float:
    """Return a probability in [0, 1] that `response` is a
    hallucination relative to `question` and (optional) `reference`.

    Higher = more likely hallucinated.
    """
    ...
```

Deterministic if possible. Network calls allowed but discouraged
(benchmarks run offline in CI). No external API calls that require
secrets — the CI has none. Installable from PyPI or bundled in the PR.

That is the entire contract.

---

## How to submit

1. Fork this repo.
2. Create `submissions/<your-system-name>.py` using the template
   in [_template_detector.py](_template_detector.py). Fill in:
    - `SYSTEM_NAME` — short display name for the leaderboard.
    - `AUTHOR` — lab / org / individual handle.
    - `CONTACT` — email or GitHub handle for correspondence.
    - `LICENSE` — your detector's license (MIT, Apache-2.0, etc.).
    - `REFERENCES` — arXiv / DOI / repo URLs.
    - `score(...)` — your implementation.
3. In `requirements-submission.txt`, list any dependencies your detector
   needs (will be pip-installed in CI).
4. Open a PR with title starting `[submission]`.
5. CI runs your detector against our 8 benchmarks at 3 seeds
   (n=150/dataset) and reports per-dataset AUC as a PR comment.
6. If the numbers are reproducible and the submission honors the
   protocol, we merge and the leaderboard updates automatically.

Expected CI cost: ~15–30 minutes depending on detector speed.

---

## The eight benchmarks

| Benchmark | Source | Paired? | Size |
|---|---|---|---|
| HaluEval-QA | `pminervini/HaluEval` | yes (truth/hallu) | 150 |
| HaluEval-Dialog | `pminervini/HaluEval` | yes | 150 |
| HaluEval-Summarization | `pminervini/HaluEval` | yes | 150 |
| TruthfulQA | `truthfulqa/truthful_qa` | yes | 150 |
| HaluBench-DROP | `PatronusAI/HaluBench` | unpaired (PASS/FAIL) | 150/class |
| HaluBench-PubMedQA | `PatronusAI/HaluBench` | unpaired | 150/class |
| HaluBench-FinanceBench | `PatronusAI/HaluBench` | unpaired | 150/class |
| HaluBench-RAGTruth | `PatronusAI/HaluBench` | unpaired | 150/class |

All datasets are loaded from Hugging Face Hub in the CI environment
using the loaders in `benchmarks/hallucination_test/cross_dataset_8bench.py`.
Seeds: [31, 47, 83]. Train/test split: 75/25. Metric: held-out AUC
(Mann-Whitney U, tie-averaged).

---

## Reported metrics

For each submission CI produces:

```json
{
  "system_name": "your-detector",
  "author": "you",
  "seeds": [31, 47, 83],
  "per_dataset": {
    "halueval_qa":             {"mean": 0.998, "std": 0.001, "seeds": [...]},
    "halueval_dialogue":       {"mean": 0.676, "std": 0.037, "seeds": [...]},
    "halueval_summarization":  {"mean": 0.643, "std": 0.060, "seeds": [...]},
    "truthfulqa":              {"mean": 0.994, "std": 0.006, "seeds": [...]},
    "halubench_drop":          {"mean": 0.424, "std": 0.080, "seeds": [...]},
    "halubench_pubmed":        {"mean": 0.719, "std": 0.051, "seeds": [...]},
    "halubench_finance":       {"mean": 0.492, "std": 0.026, "seeds": [...]},
    "halubench_ragtruth":      {"mean": 0.807, "std": 0.043, "seeds": [...]}
  },
  "overall_mean_auc": 0.719,
  "datasets_above_0_65": 5,
  "declared_failure_modes": ["halubench_drop", "halubench_finance"]
}
```

`declared_failure_modes` is a list of datasets your system explicitly
flags as unreliable (AUC < 0.55 expected). This is a first-class
field. Systems that declare their failure modes get a visible badge
on the leaderboard. Systems that don't, get listed with an
undeclared-below-chance warning. Honesty is load-bearing here.

---

## Ground rules

1. Numbers are held-out, not dev.
2. Seeds disclosed (default [31, 47, 83] — any override needs justification).
3. Submissions must include a reproducer OR a published paper with
   enough detail to reproduce.
4. If a benchmark beats our score, you move above us in that column.
   We don't grade ourselves.
5. Failure modes are published alongside successes. Cherry-picked
   submissions will not be merged.
6. CC-BY-4.0 on your reported numbers (so the leaderboard can
   redistribute freely).

---

## What you get

- A permanent row on https://fathom.darkflobi.com/cognometry/leaderboard.
- Your per-seed raw results archived in this repo's `submissions/_results/`.
- Citation of your submission in the cognometry v0.1 update paper
  (if your numbers are above any of ours on any benchmark, we will cite
  you specifically with the delta).
- Standing invitation to submit v2, v3, etc. as your system evolves.

---

## Questions

Open a GitHub issue on this repo with label `leaderboard`. Or email
the maintainer listed in CONTRIBUTING.md.

Disconfirmations welcome. This is the whole point.
