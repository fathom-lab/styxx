"""Fork Giskard-AI/awesome-ai-safety, add a cognometry entry under
Large Language Models, open a PR.
"""
from __future__ import annotations

import base64
import re
import sys
import time
from pathlib import Path

import requests

TOKEN_FILE = Path(r"C:\Users\heyzo\clawd\secrets\fathomlab-github.txt")
UPSTREAM = "Giskard-AI/awesome-ai-safety"

NEW_LINE = (
    "* [Cognometry v0: 8-Benchmark Cross-Validated Hallucination "
    "Detection in Production LLMs](https://doi.org/10.5281/zenodo.19703527) "
    "(Flobi, 2026) `#Hallucination` `#Reliability` `#Benchmarking`"
)


def token():
    m = re.search(
        r"gh[ps]_[A-Za-z0-9]+", TOKEN_FILE.read_text(encoding="utf-8")
    )
    return m.group(0)


def gh(s, method, path, **kw):
    r = s.request(
        method,
        f"https://api.github.com/{path.lstrip('/')}",
        timeout=30, **kw,
    )
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

    # Fork
    fork = gh(s, "POST", f"repos/{UPSTREAM}/forks")
    fork_full = fork["full_name"]
    fork_branch = fork.get("default_branch", "main")
    print(f"fork: {fork_full}")
    time.sleep(3)

    # Read README
    for _ in range(8):
        try:
            readme = gh(
                s, "GET", f"repos/{fork_full}/contents/README.md"
            )
            break
        except Exception:
            time.sleep(2)
    content = base64.b64decode(readme["content"]).decode("utf-8")
    sha = readme["sha"]

    if "Cognometry v0" in content:
        print("entry already present — skipping")
        return

    # Insert as the first LLM entry (right after the '### Large Language Models' header)
    marker = "### Large Language Models"
    if marker not in content:
        raise SystemExit("LLM section marker not found")

    pattern = re.compile(
        r"(### Large Language Models\s*\n\s*\n)(\* )",
        re.MULTILINE,
    )
    new_content, n = pattern.subn(
        r"\1" + NEW_LINE + r"\n\2",
        content,
        count=1,
    )
    if n != 1:
        raise SystemExit("LLM section pattern did not match")

    # Branch
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

    # Commit
    gh(s, "PUT", f"repos/{fork_full}/contents/README.md", json={
        "message": "docs: add Cognometry v0 (LLM hallucination section)",
        "content": base64.b64encode(
            new_content.encode("utf-8")
        ).decode("ascii"),
        "sha": sha,
        "branch": branch,
        "committer": {
            "name": "Fathom Lab",
            "email": "heyzoos123@gmail.com",
        },
    })

    body = """\
Adding an LLM-safety-focused entry for Cognometry v0 — the first
open-source hallucination detector cross-validated across 8 public
benchmarks.

Key numbers:
- AUC 0.998 on HaluEval-QA (held out, 3-seed averaged)
- AUC 0.994 on TruthfulQA
- 2 published failure modes (HaluBench-DROP, HaluBench-FinanceBench)
  declared openly in the weights module itself, with structural
  cause analysis

Paper: https://doi.org/10.5281/zenodo.19703527
Code: https://github.com/fathom-lab/styxx (MIT + CC-BY-4.0)
Manifesto: https://fathom.darkflobi.com/cognometry

Entry placed at the top of the Large Language Models subsection.
Hashtags follow the repo's existing convention.
"""
    pr = gh(s, "POST", f"repos/{UPSTREAM}/pulls", json={
        "title": "Add Cognometry v0 — 8-benchmark cross-validated hallucination detection (LLM Safety)",
        "head": f"{me}:{branch}",
        "base": fork_branch,
        "body": body,
    })
    print(f"\nPR opened: {pr['html_url']}")


if __name__ == "__main__":
    main()
