"""Post the HN self-comment on item 47874435.

HN comment form flow:
  GET item?id=X → extract hmac + parent
  POST /comment with text
"""
from __future__ import annotations

import re
import sys

import browser_cookie3
import requests

ITEM_ID = "47874435"

COMMENT = """Author here.

Styxx 4.0.2 is the first hallucination detector I'm aware of cross-validated across 8 public benchmarks — HaluEval QA/Dialog/Summarization, TruthfulQA, and four HaluBench subsets (DROP, PubMedQA, FinanceBench, RAGTruth). 3-seed averaged, n=150/dataset, pooled 9-signal logistic regression.

Paper (Zenodo, peer-archived): https://doi.org/10.5281/zenodo.19703527
Code: https://github.com/fathom-lab/styxx
Leaderboard: https://fathom.darkflobi.com/cognometry/leaderboard
Colab demo (2 min): https://colab.research.google.com/github/fathom-lab/styxx/blob/main/examples/cognometry_colab.ipynb

Real numbers:

    HaluEval-QA             AUC 0.998
    TruthfulQA              AUC 0.994
    HaluBench-RAGTruth      AUC 0.807   (new — RAG faithfulness)
    HaluBench-PubMedQA      AUC 0.719   (new — biomedical)
    HaluEval-Dialog         AUC 0.676
    HaluEval-Summarization  AUC 0.643
    HaluBench-FinanceBench  AUC 0.492   (below chance)
    HaluBench-DROP          AUC 0.424   (below chance)

Two below-chance results are the part I'd most like HN to react to. They are published as failure modes in the weights module itself, not hidden:

- DROP: reading-comp hallucinations are extractive-span errors — wrong span, right passage. NLI scores that as entailed; novelty signals don't fire. Tried 6 naive heuristic fixes; all null. The null probe is committed alongside the successes.
- FinanceBench: hallucinations are calculation errors on numbers copied verbatim from the source. Novelty + NLI are semantically blind to arithmetic correctness.

Both failure modes are declared in calibrated_weights_v4.CALIBRATION_NOTES.documented_failure_modes so production callers know where the detector will lie.

pip install styxx[nli] → wrap a function with @trust → get verified output on every call. Zero config: auto-detects context/reference/passage kwargs, auto-enables NLI when installed, adaptive threshold. MIT on code, CC-BY on calibrated weights.

Happy to get disconfirmations on any of the 8 benchmarks at your favorite random seed."""


def main():
    jar = browser_cookie3.firefox(domain_name="news.ycombinator.com")
    s = requests.Session()
    for c in jar:
        s.cookies.set(c.name, c.value, domain=c.domain)
    s.headers.update({"User-Agent": "Mozilla/5.0"})

    # Verify login + own the submission
    r = s.get(
        f"https://news.ycombinator.com/item?id={ITEM_ID}", timeout=15
    )
    if 'href="logout' not in r.text:
        print("NOT LOGGED IN — abort")
        sys.exit(1)

    # Extract hmac for the top-level reply form. HN comment forms
    # have a hidden hmac field; the parent field is the item id.
    hmac_m = re.search(
        r'<input[^>]+name="hmac"[^>]+value="([^"]+)"', r.text
    )
    if not hmac_m:
        print("hmac not found — HN layout may have changed")
        sys.exit(1)
    hmac = hmac_m.group(1)
    print(f"hmac: {hmac[:16]}...")

    data = {
        "parent": ITEM_ID,
        "goto": f"item?id={ITEM_ID}",
        "hmac": hmac,
        "text": COMMENT,
    }
    r = s.post(
        "https://news.ycombinator.com/comment",
        data=data,
        allow_redirects=False,
        timeout=20,
    )
    print(f"comment POST status: {r.status_code}")
    loc = r.headers.get("Location", "")
    print(f"redirect to: {loc}")
    if r.status_code in (200, 302):
        print(
            f"\nSELF-COMMENT POSTED on "
            f"https://news.ycombinator.com/item?id={ITEM_ID}"
        )
    else:
        print("FAIL body:", r.text[:500])
        sys.exit(2)


if __name__ == "__main__":
    main()
