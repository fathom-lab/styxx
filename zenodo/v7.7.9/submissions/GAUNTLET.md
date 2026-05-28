# submissions/ — `styxx gauntlet` public-challenge protocol

This file documents the **gauntlet protocol** (added in styxx 7.7.6+), which is *separate from* the Cognometry Detector Interface v0 documented in `submissions/README.md`. The two protocols target different benchmarks:

| protocol | benchmark | what it tests |
|---|---|---|
| **Cognometry Detector Interface v0** (`submissions/README.md`) | the eight hallucination benchmarks (HaluEval-QA, TruthfulQA, etc.) | hallucination-detection AUC across 8 datasets at 3 seeds |
| **Gauntlet** (this file) | `darkcore_benchmark_2026_05_27.json` — 108 labeled records, 4 classes | classification or detection on the consensus-hallucination dark core |

Pick the protocol that matches the benchmark you want to attempt. Both are valid open submissions and both ship to the public leaderboard.

---

## Quick start — submit your gauntlet method

1. **Fork the repo and create `submissions/<your-name>/`** with at least:
   - `method.py` — your detection or classification function
   - `submission.json` — metadata + your reported gauntlet result
   - `README.md` *(optional)* — short description + citation
   - `requirements.txt` *(optional)* — extra pip deps

2. **Write `method.py`** with one of these signatures:

   ```python
   # CLASSIFICATION
   def predict(question: str) -> dict:
       """Return a class label.
       Returns: dict with key "class" ∈ {"folklore", "pseudoscience", "factual-error", "truth"}.
       """
       ...
       return {"class": "..."}

   # DETECTION
   def detect(question: str, response: str) -> dict:
       """Score whether (question, response) is misconception-like.
       Returns: dict with key "score" (float; higher = more misconception-like).
       """
       ...
       return {"score": ...}
   ```

3. **Run the gauntlet locally** and save the JSON result:

   ```bash
   pip install styxx>=7.7.6
   styxx gauntlet \
     --method submissions.<your-name>.method:predict \
     --task classification \
     --name "Your-Method-Name" \
     --format json > submissions/<your-name>/_local_result.json
   ```

4. **Write `submission.json`** with metadata + reported scores. The CI workflow `.github/workflows/gauntlet-pr.yml` re-runs the gauntlet on your method and compares; if scores don't match (float tolerance 1e-3), the PR fails verification.

   ```json
   {
     "name": "Your-Method-Name",
     "submitter": "Your Name / Affiliation",
     "method": "submissions.<your-name>.method:predict",
     "task": "classification",
     "description": "short description of the method",
     "code_url": "link to your repo or this PR",
     "reported": {
       "overall_pass": false,
       "n_passed": 0,
       "n_total_bars": 3,
       "metrics": {
         "accuracy": 0.72,
         "folklore_F1_indist": 0.55,
         "folklore_F1_crosscorpus": 0.42
       }
     }
   }
   ```

5. **Open a PR.** CI auto-verification runs:
   - Installs your `requirements.txt` if present.
   - Runs `styxx gauntlet` against the **bundled** benchmark JSON.
   - Compares the CI result to your `reported` block (1e-3 float tolerance).
   - If they match → CI green, PR mergeable. Operator review for leaderboard placement.
   - If they mismatch → CI red, mismatch report printed; submitter fixes and re-pushes.

## Bars (locked, cannot be modified by PR)

| task | bar | threshold | source |
|---|---|---|---|
| classification | K1 folklore F1 (in-distribution) | ≥ 0.70 | `preregistration_darkcore_classifier_2026_05_27.md` |
| classification | K2 accuracy (4-way) | ≥ 0.65 | same |
| classification | K3 folklore F1 (cross-corpus) | ≥ 0.60 | same — **the load-bearing bar** |
| detection | D1 misconception AUC | ≥ 0.70 | derived from JD/ICT findings |
| detection | D2 folklore-subset AUC | ≥ 0.70 | derived from JD/ICT findings |

## Existing reference baselines

- **`baseline_002_classifier/`** — the shipped dark-core classifier (sentence-transformer + balanced LR). 1/3 bars passed (K2 0.77 ✓; K1 0.42 ✗; K3 0.36 ✗).
- **`baseline_003_length/`** — a deliberately bad length-only heuristic. 0/3 bars. Anchors the bottom of the leaderboard; notably gets K3=0.56 from high recall + bad precision.

## Honor system + CI

- CI re-runs gauntlet on the **bundled** benchmark; you can't beat the floor by editing the benchmark JSON.
- CI cannot verify your training data is free of benchmark contamination. Submitters are on the honor system; independent papers may catch a cheat.
- The bars are locked at prereg commits cited in `LEADERBOARD.md`. They are not editable by PR.

## What gets rejected

- PRs that modify the benchmark JSON, the bars, or the gauntlet runner itself.
- PRs whose `submission.json` reported scores don't match the CI re-run.
- PRs without a `submission.json` or `method.py`.
- PRs whose method requires resources we can't run in CI (proprietary APIs without a free tier, GPU > 16 GB VRAM, manual interaction). Open an issue first to discuss.

## What gets accepted

- A clean, reproducible method that runs in CI and whose scores match the report. Even if it fails all bars, the row goes on the board — the floor compounds across submissions.

The discipline is the moat. Submit honestly; we merge fairly.
