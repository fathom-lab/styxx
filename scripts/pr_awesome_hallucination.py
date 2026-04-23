"""Fork EdinburghNLP/awesome-hallucination-detection, add a
cognometry-v0 entry to README.md, open a PR.

Uses the fathomlab-github.txt PAT.
"""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

import requests

TOKEN_FILE = Path(r"C:\Users\heyzo\clawd\secrets\fathomlab-github.txt")
UPSTREAM = "EdinburghNLP/awesome-hallucination-detection"
DEFAULT_BRANCH = "main"

NEW_ENTRY = """\
### [Cognometry v0: 8-Benchmark Cross-Validated Hallucination Detection in Production LLMs](https://doi.org/10.5281/zenodo.19703527)
- **Metrics:** AUC (held-out test, 3-seed averaged)
- **Datasets:** HaluEval-QA, HaluEval-Dialog, HaluEval-Summarization, TruthfulQA, HaluBench-DROP, HaluBench-PubMedQA, HaluBench-FinanceBench, HaluBench-RAGTruth
- **Comments:** Introduces **cognometry** as the empirical measurement of cognitive states in LLMs. Ships a 9-signal pooled LR (text claim risk, entity verification, knowledge grounding, four response-novelty variants, NLI contradiction via DeBERTa-v3-base-mnli-fever-anli) cross-validated on 8 public benchmarks. Achieves AUC **0.998** on HaluEval-QA and **0.994** on TruthfulQA, with below-chance results on HaluBench-DROP (0.424) and FinanceBench (0.492) declared openly as **documented failure modes** in the weights module itself — DROP because extractive-span errors pass NLI entailment, FinanceBench because arithmetic errors on verbatim-copied numbers pass novelty. First open-source hallucination detector cross-validated at this breadth in the literature. Ships as `pip install styxx[nli]` + `@trust` decorator. 591 tests, 3-seed reproducer committed. MIT code, CC-BY-4.0 calibrated weights. Companion manifesto defines three falsifiable laws of cognometry (vitals exist, transfer, are causally actionable) with cross-validated numerical support. (Zenodo 2026)
"""


def token():
    m = re.search(
        r"gh[ps]_[A-Za-z0-9]+", TOKEN_FILE.read_text(encoding="utf-8")
    )
    if not m:
        raise SystemExit("token not found")
    return m.group(0)


def gh(sess: requests.Session, method: str, path: str, **kw) -> dict:
    url = f"https://api.github.com/{path.lstrip('/')}"
    r = sess.request(method, url, timeout=30, **kw)
    r.raise_for_status()
    return r.json() if r.content else {}


def main():
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token()}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "styxx-launch-bot",
    })

    # 1. Who am I
    me = gh(s, "GET", "user")
    me_login = me["login"]
    print(f"authed as: {me_login}")

    # 2. Fork (idempotent — if fork exists, returns it)
    print("forking...")
    fork = gh(s, "POST", f"repos/{UPSTREAM}/forks")
    fork_full = fork["full_name"]
    fork_branch = fork.get("default_branch", DEFAULT_BRANCH)
    print(f"  fork: {fork_full} (default: {fork_branch})")
    # Fork takes a moment to materialize
    time.sleep(3)

    # 3. Read upstream README via our fork to include all latest content
    for _ in range(10):
        try:
            readme = gh(s, "GET", f"repos/{fork_full}/contents/README.md")
            break
        except Exception:
            time.sleep(2)
    content_b64 = readme["content"]
    sha = readme["sha"]
    content = base64.b64decode(content_b64).decode("utf-8")

    # 4. Insert the new entry right after "## Papers and Summaries" heading
    marker = "## Papers and Summaries"
    if marker not in content:
        raise SystemExit("section marker not found in README")
    if "Cognometry v0" in content:
        print("entry already present — skipping edit (maybe prior PR)")
        return

    new_content = content.replace(
        marker,
        marker + "\n\n" + NEW_ENTRY.rstrip() + "\n",
        1,
    )
    # increment paper count badge if easy
    m = re.search(r"Papers-(\d+)-blue", new_content)
    if m:
        n = int(m.group(1)) + 1
        new_content = new_content.replace(
            f"Papers-{m.group(1)}-blue", f"Papers-{n}-blue"
        )
        print(f"bumped paper count badge: {m.group(1)} → {n}")

    # 5. Commit on a new branch
    branch_name = "add-cognometry-v0"
    # create branch from main
    ref = gh(s, "GET", f"repos/{fork_full}/git/ref/heads/{fork_branch}")
    sha_main = ref["object"]["sha"]
    try:
        gh(s, "POST", f"repos/{fork_full}/git/refs", json={
            "ref": f"refs/heads/{branch_name}",
            "sha": sha_main,
        })
        print(f"  created branch {branch_name}")
    except requests.HTTPError as e:
        if e.response.status_code == 422:
            print(f"  branch {branch_name} already exists — continuing")
        else:
            raise

    # 6. Commit the file change on the branch
    gh(s, "PUT", f"repos/{fork_full}/contents/README.md", json={
        "message": "docs: add Cognometry v0 — 8-benchmark cross-validated hallucination detection",
        "content": base64.b64encode(new_content.encode("utf-8")).decode("ascii"),
        "sha": sha,
        "branch": branch_name,
        "committer": {
            "name": "Fathom Lab",
            "email": "heyzoos123@gmail.com",
        },
    })
    print("  commit pushed")

    # 7. Open PR
    pr_body = """\
Adding a paper entry for **Cognometry v0** — the first open-source
hallucination detector cross-validated across 8 public benchmarks
(including the four HaluBench subsets from Patronus AI).

The submission is in the list's established format (metrics, datasets,
comments). I've placed it at the top of "Papers and Summaries" since
it was deposited on Zenodo on 2026-04-23.

Two things worth calling out beyond the headline AUCs (0.998 HaluEval-QA,
0.994 TruthfulQA):

1. **Two failure modes published openly** (HaluBench-DROP 0.424 and
   HaluBench-FinanceBench 0.492 — both below chance) with their
   structural causes characterized in the weights module itself.
2. **3-seed averaging + committed reproducer** — full benchmark harness
   is in the repo and produces the per-dataset AUCs from raw HuggingFace
   datasets.

Paper: https://doi.org/10.5281/zenodo.19703527
Code: https://github.com/fathom-lab/styxx (MIT + CC-BY-4.0)
Manifesto: https://fathom.darkflobi.com/cognometry
Leaderboard: https://fathom.darkflobi.com/cognometry/leaderboard

Happy to refine the summary or move the entry if you'd prefer a different
placement.
"""
    pr = gh(s, "POST", f"repos/{UPSTREAM}/pulls", json={
        "title": "Add Cognometry v0 — 8-benchmark cross-validated hallucination detection",
        "head": f"{me_login}:{branch_name}",
        "base": fork_branch,
        "body": pr_body,
    })
    print(f"\nPR opened: {pr['html_url']}")


if __name__ == "__main__":
    main()
