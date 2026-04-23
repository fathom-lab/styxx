"""Fork jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness,
add cognometry entry under Reliability > Hallucination, open PR.
"""
from __future__ import annotations

import base64
import re
import sys
import time
from pathlib import Path

import requests

TOKEN_FILE = Path(r"C:\Users\heyzo\clawd\secrets\fathomlab-github.txt")
UPSTREAM = "jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness"

NEW_ENTRY = """\
**Cognometry v0: 8-Benchmark Cross-Validated Hallucination Detection in Production LLMs** \\
Introduces *cognometry* — the empirical measurement of cognitive states in LLMs. 9-signal pooled LR (text, entity, knowledge grounding, 4 response-novelty variants, NLI contradiction via DeBERTa-v3-base-mnli) cross-validated on 8 benchmarks (HaluEval QA/Dialog/Summ, TruthfulQA, HaluBench DROP/PubMedQA/FinanceBench/RAGTruth). AUC 0.998 on HaluEval-QA; two below-chance results (DROP, FinanceBench) declared as published failure modes in the weights module. \\
[[Paper](https://doi.org/10.5281/zenodo.19703527)] [[Code](https://github.com/fathom-lab/styxx)] [[Manifesto](https://fathom.darkflobi.com/cognometry)]
"""


def token():
    m = re.search(r"gh[ps]_[A-Za-z0-9]+", TOKEN_FILE.read_text(encoding="utf-8"))
    return m.group(0)


def gh(s, method, path, **kw):
    r = s.request(method, f"https://api.github.com/{path.lstrip('/')}",
                   timeout=30, **kw)
    r.raise_for_status()
    return r.json() if r.content else {}


def main():
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token()}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "styxx-launch-bot",
    })
    me = gh(s, "GET", "user")["login"]

    fork = gh(s, "POST", f"repos/{UPSTREAM}/forks")
    fork_full = fork["full_name"]
    fork_branch = fork.get("default_branch", "main")
    print(f"fork: {fork_full}")
    time.sleep(3)

    for _ in range(8):
        try:
            readme = gh(s, "GET", f"repos/{fork_full}/contents/README.md")
            break
        except Exception:
            time.sleep(2)
    content = base64.b64decode(readme["content"]).decode("utf-8")
    sha = readme["sha"]

    if "Cognometry v0" in content:
        print("already present — skipping")
        return

    # Insert as first paper entry under ### Hallucination section
    # The section starts with "### Hallucination\n> [awesome hallucination detection]..."
    marker = "### Hallucination"
    idx = content.find(marker)
    if idx < 0:
        raise SystemExit("Hallucination section not found")
    # Find the first "**" (paper) in that section — that's where to insert before
    rel = content.find("**", idx)
    if rel < 0:
        raise SystemExit("no paper entries in Hallucination section")

    new_content = content[:rel] + NEW_ENTRY + "\n" + content[rel:]

    # Branch + commit + PR
    ref = gh(s, "GET", f"repos/{fork_full}/git/ref/heads/{fork_branch}")
    branch = "add-cognometry-v0"
    try:
        gh(s, "POST", f"repos/{fork_full}/git/refs", json={
            "ref": f"refs/heads/{branch}",
            "sha": ref["object"]["sha"],
        })
    except requests.HTTPError as e:
        if e.response.status_code != 422:
            raise

    gh(s, "PUT", f"repos/{fork_full}/contents/README.md", json={
        "message": "docs: add Cognometry v0 (Reliability > Hallucination)",
        "content": base64.b64encode(new_content.encode("utf-8")).decode("ascii"),
        "sha": sha,
        "branch": branch,
        "committer": {"name": "Fathom Lab", "email": "heyzoos123@gmail.com"},
    })

    body = """\
Adding **Cognometry v0** to Reliability > Hallucination. First open-source
hallucination detector cross-validated across 8 public benchmarks with
3-seed-averaged AUCs.

Headline: AUC 0.998 on HaluEval-QA, 0.994 on TruthfulQA, 0.807 on
HaluBench-RAGTruth. **Two failure modes declared openly** in the weights
module (HaluBench-DROP 0.424, HaluBench-FinanceBench 0.492 — below chance,
with structural explanation).

Just merged by @pminervini into `EdinburghNLP/awesome-hallucination-detection`
(#55), thought it belonged here too given the Reliability framing.

Paper: https://doi.org/10.5281/zenodo.19703527
Code: https://github.com/fathom-lab/styxx (MIT + CC-BY-4.0)
"""
    pr = gh(s, "POST", f"repos/{UPSTREAM}/pulls", json={
        "title": "Add Cognometry v0 — 8-benchmark cross-validated hallucination detection",
        "head": f"{me}:{branch}",
        "base": fork_branch,
        "body": body,
    })
    print(f"\nPR opened: {pr['html_url']}")


if __name__ == "__main__":
    main()
