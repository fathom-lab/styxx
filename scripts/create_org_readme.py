"""Create fathom-lab/.github repo + profile/README.md so
github.com/fathom-lab gets a proper landing experience.

Without this, the org page shows only a list of repos. With it, a
curated intro with the cognometry narrative, open PRs, and real
numbers is the first thing visitors see.
"""
from __future__ import annotations

import base64
import re
import sys
import time
from pathlib import Path

import requests

TOKEN_FILE = Path(r"C:\Users\heyzo\clawd\secrets\fathomlab-github.txt")
ORG = "fathom-lab"

README = """\
# Fathom Lab

*Cognitive instruments for machine cognition. Open source. Published failure modes.*

---

## What we build

We measure cognitive states of large language models at runtime — refusal,
confabulation, retrieval, reasoning, adversarial drift — from signals already
carried on the token stream and residual activations. Three of our tools are
public today:

- **[`styxx`](https://github.com/fathom-lab/styxx)** — one decorator, any LLM call,
  cross-validated hallucination detection. `pip install styxx[nli]` +
  `@trust`. Cross-validated across 8 public benchmarks with two declared
  failure modes published openly in the weights module.

- **[`fathom`](https://github.com/fathom-lab/fathom)** — SAE-based depth
  measurement for transformer internals. Fathom constant 1.0212 measured
  across two open-weight architectures.

- **Cognometry manifesto** — [fathom.darkflobi.com/cognometry](https://fathom.darkflobi.com/cognometry)
  — three falsifiable laws for cognometric measurement, each with a
  cross-validated number.

## Current numbers (styxx v4.0.2, 3-seed averaged, n=150/dataset)

| Benchmark | AUC |
|---|---|
| HaluEval-QA | **0.998 ± 0.001** |
| TruthfulQA | **0.994 ± 0.006** |
| HaluBench-RAGTruth | **0.807 ± 0.043** |
| HaluBench-PubMedQA | **0.719 ± 0.051** |
| HaluEval-Dialog | 0.676 ± 0.037 |
| HaluEval-Summarization | 0.643 ± 0.060 |
| HaluBench-FinanceBench | 0.492 ± 0.026 — *declared failure* |
| HaluBench-DROP | 0.424 ± 0.080 — *declared failure* |

> Two of the eight came in below chance. They're declared in
> `calibrated_weights_v4.CALIBRATION_NOTES.documented_failure_modes` so
> production callers know where the detector will lie. That honesty is
> load-bearing for how we run this lab.

## Cognometry leaderboard

Open submission: any lab can PR a detector following the [Cognometry
Detector Interface v0](https://github.com/fathom-lab/styxx/blob/main/submissions/README.md)
protocol and have it auto-evaluated against our 8 benchmarks. Live table:

**→ [fathom.darkflobi.com/cognometry/leaderboard](https://fathom.darkflobi.com/cognometry/leaderboard)**

## Papers

- [Cognometry v0 (Zenodo)](https://doi.org/10.5281/zenodo.19703527) — 8-benchmark
  cross-validated hallucination detection.
- [Cognitive Metrology (Zenodo)](https://doi.org/10.5281/zenodo.19504993) —
  logprob-trajectory methodology.

## Also at

- Site: [fathom.darkflobi.com](https://fathom.darkflobi.com)
- Twitter/X: [@fathom_lab](https://x.com/fathom_lab)
- PyPI: [styxx](https://pypi.org/project/styxx/)
- OSF: [osf.io/g2epj](https://osf.io/g2epj/) · parent project [osf.io/wtkzg](https://osf.io/wtkzg/)

## How to contribute

- **Disconfirmations welcome.** If a number is wrong at your favorite seed,
  open an issue or PR — we cite disconfirmations in the next paper.
- **Submit a detector** to the cognometry leaderboard (one-file PR,
  protocol above).
- **Extend the benchmark suite.** FEVER, FactCC, XSum-Faithful, and PHD-A
  are on the v4.2 track — PRs welcome.

---

*"nothing crosses unseen"*
"""


def token():
    m = re.search(r"gh[ps]_[A-Za-z0-9]+", TOKEN_FILE.read_text(encoding="utf-8"))
    return m.group(0)


def gh(s, method, path, **kw):
    r = s.request(
        method, f"https://api.github.com/{path.lstrip('/')}",
        timeout=30, **kw,
    )
    if r.status_code >= 400:
        raise requests.HTTPError(
            f"{r.status_code} {r.reason}: {r.text[:400]}", response=r,
        )
    return r.json() if r.content else {}


def main():
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token()}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "fathom-lab-bot",
    })

    # Check if .github repo exists in the org
    try:
        repo = gh(s, "GET", f"repos/{ORG}/.github")
        print(f"repo exists: {repo['html_url']}")
    except requests.HTTPError as e:
        if e.response.status_code != 404:
            raise
        # Create it
        print("creating .github repo...")
        repo = gh(s, "POST", f"orgs/{ORG}/repos", json={
            "name": ".github",
            "description": "Fathom Lab organization profile",
            "private": False,
            "auto_init": True,
        })
        print(f"  created: {repo['html_url']}")
        time.sleep(3)

    default = repo.get("default_branch", "main")

    # Upsert profile/README.md
    path = "profile/README.md"
    sha = None
    try:
        existing = gh(s, "GET",
                      f"repos/{ORG}/.github/contents/{path}?ref={default}")
        sha = existing["sha"]
        print(f"README exists (sha {sha[:8]}) — updating")
    except requests.HTTPError as e:
        if e.response.status_code != 404:
            raise
        print("README not present — creating")

    payload = {
        "message": "org: publish Fathom Lab profile README",
        "content": base64.b64encode(README.encode("utf-8")).decode("ascii"),
        "branch": default,
    }
    if sha:
        payload["sha"] = sha
    r = gh(s, "PUT", f"repos/{ORG}/.github/contents/{path}", json=payload)
    print(f"\norg profile README live:")
    print(f"  {r['content']['html_url']}")
    print(f"  rendered at: https://github.com/{ORG}")


if __name__ == "__main__":
    main()
