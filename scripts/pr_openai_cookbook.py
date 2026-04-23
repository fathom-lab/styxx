"""PR a hallucination-detection notebook to openai/openai-cookbook.

- Fork openai-cookbook
- Add examples/evaluation/hallucination_detection_with_styxx.ipynb
- Update registry.yaml with the entry
- Open PR
"""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TOKEN_FILE = Path(r"C:\Users\heyzo\clawd\secrets\fathomlab-github.txt")
UPSTREAM = "openai/openai-cookbook"
NOTEBOOK_PATH = "examples/evaluation/hallucination_detection_with_styxx.ipynb"


NOTEBOOK = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Hallucination detection for OpenAI responses with styxx\n",
                "\n",
                "This notebook shows how to wrap any OpenAI call with runtime hallucination detection using [styxx](https://github.com/fathom-lab/styxx) — a 9-signal hallucination detector cross-validated across 8 public benchmarks.\n",
                "\n",
                "**What you'll do:**\n",
                "1. `pip install styxx[nli]`\n",
                "2. Wrap any function that calls the OpenAI API with `@trust`\n",
                "3. See the detector pass-through a correct answer and intercept a hallucinated one\n",
                "4. Inspect the per-signal verdict (risk, action, signal table)\n",
                "\n",
                "**Why this matters for production OpenAI workloads:**\n",
                "- Cross-validated AUCs on public benchmarks:\n",
                "\n",
                "| Benchmark | AUC |\n",
                "|---|---|\n",
                "| HaluEval-QA | 0.998 |\n",
                "| TruthfulQA | 0.994 |\n",
                "| HaluBench-RAGTruth (RAG faithfulness) | 0.807 |\n",
                "| HaluBench-PubMedQA | 0.719 |\n",
                "| HaluEval-Dialog | 0.676 |\n",
                "| HaluEval-Summarization | 0.643 |\n",
                "| HaluBench-FinanceBench | 0.492 (declared failure mode) |\n",
                "| HaluBench-DROP | 0.424 (declared failure mode) |\n",
                "\n",
                "- Two failure modes published openly so you know where the detector **will not** help (extractive-span reading-comp errors on DROP-like tasks and financial arithmetic on FinanceBench-like tasks).\n",
                "- Local-first: runs offline on CPU (~400ms) or CUDA (~30ms). No extra API key required.\n",
                "- MIT on code, CC-BY-4.0 on calibrated weights.\n",
                "\n",
                "Full manifesto + paper: https://fathom.darkflobi.com/cognometry · DOI: [10.5281/zenodo.19703527](https://doi.org/10.5281/zenodo.19703527).\n",
                "\n",
                "---"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Install"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%pip install -q styxx[nli] openai"],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Set your OpenAI key\n",
                "\n",
                "Either export `OPENAI_API_KEY` in your shell or paste it below."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "# os.environ['OPENAI_API_KEY'] = 'sk-...'\n",
                "assert os.environ.get('OPENAI_API_KEY'), 'set OPENAI_API_KEY first'"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Wrap your OpenAI function with `@trust`\n",
                "\n",
                "Zero configuration. `@trust` inspects your function signature and auto-detects `context` (or `reference`, `passage`, `docs`, `source`, `knowledge`, `grounding`, `retrieved`) as the grounding passage. Auto-enables the NLI contradiction signal because `styxx[nli]` is installed."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from styxx import trust\n",
                "import openai\n",
                "\n",
                "client = openai.OpenAI()\n",
                "\n",
                "@trust\n",
                "def ask(question, *, context):\n",
                "    r = client.chat.completions.create(\n",
                "        model='gpt-4o-mini',\n",
                "        messages=[\n",
                "            {'role': 'user',\n",
                "             'content': f'Context: {context}\\n\\nQuestion: {question}\\n\\n'\n",
                "                        f'Answer concisely using only the context.'},\n",
                "        ],\n",
                "        temperature=0.3,\n",
                "    )\n",
                "    return r.choices[0].message.content"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Correct answer — passes through unchanged\n",
                "\n",
                "The model gets a grounded question on a grounded context. `@trust` sees low novelty + no NLI contradiction → returns the response verbatim."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "context = '''Inception is a 2010 science fiction film written and directed\n",
                "by Christopher Nolan. It stars Leonardo DiCaprio as a thief who steals\n",
                "corporate secrets through dream-sharing technology.'''\n",
                "\n",
                "ask('Who directed Inception?', context=context)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Force a hallucination — `@trust` intercepts it\n",
                "\n",
                "Same function, but now we ask a specific question whose answer is NOT in the context. The model will confabulate a plausible number. The detector fires because the response contains high-novelty tokens (made-up dollar figure) that have no support in the reference passage."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "context = '''Inception is a 2010 science fiction film written and\n",
                "directed by Christopher Nolan.'''\n",
                "\n",
                "ask('What was the exact production budget of Inception, to the dollar? '\n",
                "    'Reply with just the number.', context=context)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "If that returned the styxx fallback message instead of a fabricated number, the detector worked.\n",
                "\n",
                "## 6. Inspect the verdict — `on_halt='annotate'`\n",
                "\n",
                "Return a `TrustResult` that exposes per-signal risk, halt action, and the signal table. Same interface works across OpenAI, Anthropic, local models, and anything that returns a string."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "@trust(on_halt='annotate')\n",
                "def ask_annotated(question, *, context):\n",
                "    r = client.chat.completions.create(\n",
                "        model='gpt-4o-mini',\n",
                "        messages=[{'role': 'user',\n",
                "                   'content': f'Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer concisely.'}],\n",
                "        temperature=0.3,\n",
                "    )\n",
                "    return r.choices[0].message.content\n",
                "\n",
                "result = ask_annotated(\n",
                "    'What was the exact production budget of Inception, to the dollar?',\n",
                "    context=context,\n",
                ")\n",
                "\n",
                "print(f'response : {result.response}')\n",
                "print(f'risk     : {result.verdict.risk:.3f}')\n",
                "print(f'action   : {result.verdict.action}')\n",
                "print(f'halted   : {result.halted}')\n",
                "print(f'attempts : {result.attempts}')\n",
                "print('\\nsignals:')\n",
                "for s in result.verdict.signals:\n",
                "    print(f'  {s.name:<22s} {s.value}')"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Halt policies\n",
                "\n",
                "`@trust(on_halt=...)` supports four policies for what happens when risk exceeds threshold:\n",
                "\n",
                "- `fallback` (default): return a safe fallback string\n",
                "- `retry`: re-call the function up to `max_retries` times; best-of-N picks the lowest-risk response\n",
                "- `raise`: raise `TrustViolation` exception\n",
                "- `annotate`: return `TrustResult(response, verdict)` — let the caller decide\n",
                "\n",
                "```python\n",
                "@trust(on_halt='retry', max_retries=3, threshold=0.7)\n",
                "def safer_ask(q, *, context): ...\n",
                "```"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Honest failure modes\n",
                "\n",
                "Styxx **does not work well** on two benchmark types, published as declared failure modes in the weights module itself:\n",
                "\n",
                "- **Reading-comprehension extractive-span errors** (e.g. DROP-style tasks where the answer is a specific span of a passage). The wrong span is *entailed* by the passage at the NLI level; novelty signals don't fire because the tokens overlap.\n",
                "- **Financial arithmetic** (e.g. FinanceBench-style calculation errors on numbers copied verbatim from the source). Novelty + NLI are semantically blind to arithmetic correctness.\n",
                "\n",
                "Do not deploy `@trust` for production workloads in those two domains without additional domain-specific checks. Full deep-dive with the null-probe evidence: https://fathom.darkflobi.com/cognometry/failures\n",
                "\n",
                "## Further reading\n",
                "\n",
                "- Manifesto & methodology: https://fathom.darkflobi.com/cognometry\n",
                "- Leaderboard (open submissions): https://fathom.darkflobi.com/cognometry/leaderboard\n",
                "- Paper (Zenodo): https://doi.org/10.5281/zenodo.19703527\n",
                "- Source: https://github.com/fathom-lab/styxx (MIT + CC-BY-4.0)"
            ],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


REGISTRY_ENTRY = """\
- title: Hallucination detection for OpenAI responses with styxx
  path: examples/evaluation/hallucination_detection_with_styxx.ipynb
  slug: hallucination-detection-with-styxx
  description: Wrap any OpenAI call with `@trust` for runtime hallucination detection. Cross-validated across 8 public benchmarks (AUC 0.998 on HaluEval-QA, 0.807 on HaluBench-RAGTruth), with two declared failure modes published openly.
  date: 2026-04-23
  authors:
    - fathomlab
  tags:
    - evaluation
    - hallucination
    - reliability
    - rag

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
        "User-Agent": "fathomlab-bot",
    })
    me = gh(s, "GET", "user")["login"]

    # 1. Fork
    fork = gh(s, "POST", f"repos/{UPSTREAM}/forks")
    fork_full = fork["full_name"]
    fork_branch = fork.get("default_branch", "main")
    print(f"fork: {fork_full}")
    time.sleep(3)

    # 2. Get upstream registry.yaml content via our fork
    for _ in range(8):
        try:
            registry = gh(s, "GET",
                           f"repos/{fork_full}/contents/registry.yaml")
            break
        except Exception:
            time.sleep(2)
    content = base64.b64decode(registry["content"]).decode("utf-8")
    if "hallucination-detection-with-styxx" in content:
        print("registry already has our entry — skipping")
        return

    # Insert our entry at the top of the registry, after the header comments.
    # Find the first "- title:" line and prepend there.
    idx = content.find("\n- title:")
    if idx < 0:
        print("could not locate entries in registry.yaml")
        sys.exit(1)
    new_registry = content[: idx + 1] + REGISTRY_ENTRY + content[idx + 1:]

    # 3. Create branch
    ref = gh(s, "GET", f"repos/{fork_full}/git/ref/heads/{fork_branch}")
    branch = "add-styxx-hallucination-notebook"
    try:
        gh(s, "POST", f"repos/{fork_full}/git/refs", json={
            "ref": f"refs/heads/{branch}",
            "sha": ref["object"]["sha"],
        })
        print(f"  created branch {branch}")
    except requests.HTTPError as e:
        if e.response.status_code != 422:
            raise

    # 4. Upload notebook (new file)
    nb_b64 = base64.b64encode(
        json.dumps(NOTEBOOK, indent=1).encode("utf-8")
    ).decode("ascii")
    gh(s, "PUT", f"repos/{fork_full}/contents/{NOTEBOOK_PATH}", json={
        "message": "docs: add styxx hallucination detection notebook",
        "content": nb_b64,
        "branch": branch,
        "committer": {"name": "Fathom Lab",
                       "email": "heyzoos123@gmail.com"},
    })
    print(f"  notebook added at {NOTEBOOK_PATH}")

    # 5. Update registry.yaml
    gh(s, "PUT", f"repos/{fork_full}/contents/registry.yaml", json={
        "message": "docs: register styxx hallucination notebook",
        "content": base64.b64encode(
            new_registry.encode("utf-8")
        ).decode("ascii"),
        "sha": registry["sha"],
        "branch": branch,
        "committer": {"name": "Fathom Lab",
                       "email": "heyzoos123@gmail.com"},
    })
    print("  registry.yaml updated")

    # 6. PR
    pr_body = """\
Adds an end-to-end example notebook demonstrating runtime hallucination detection on OpenAI responses, using [styxx](https://github.com/fathom-lab/styxx) — a 9-signal detector cross-validated across 8 public benchmarks.

**Headline numbers (3-seed averaged, n=150/dataset):**

| Benchmark | AUC |
|---|---|
| HaluEval-QA | 0.998 |
| TruthfulQA | 0.994 |
| HaluBench-RAGTruth | 0.807 |
| HaluBench-PubMedQA | 0.719 |
| HaluEval-Dialog | 0.676 |
| HaluEval-Summarization | 0.643 |
| HaluBench-FinanceBench | 0.492 *(declared failure mode)* |
| HaluBench-DROP | 0.424 *(declared failure mode)* |

Two failure modes declared openly in the weights module itself — users know where NOT to deploy `@trust` (extractive-span reading-comp, financial arithmetic). Deep-dive at https://fathom.darkflobi.com/cognometry/failures.

**Why this might fit the cookbook:**

- Concrete runtime-gating pattern for the most common OpenAI production pain (hallucination). One decorator, zero config.
- Applies directly to RAG — HaluBench-RAGTruth AUC 0.807 is the strongest new number and exactly what RAG users care about.
- Zenodo-archived paper + committed reproducer. Merged into [EdinburghNLP/awesome-hallucination-detection#55](https://github.com/EdinburghNLP/awesome-hallucination-detection/pull/55) by Pasquale Minervini (HaluEval author).
- MIT on code, CC-BY-4.0 on calibrated weights. No OpenAI-incompatible licensing.

**What this PR does:**
1. Adds `examples/evaluation/hallucination_detection_with_styxx.ipynb` — install → wrap → pass-through → intercept → inspect → halt-policies → honest failure modes.
2. Adds the entry to `registry.yaml` (first in the list, tagged `evaluation`, `hallucination`, `reliability`, `rag`).

Happy to iterate on the notebook content, format, or placement per whatever editorial standard you apply. If a specific OpenAI model should be swapped in instead of `gpt-4o-mini`, I'll rerun everything.

Full cognometry manifesto: https://fathom.darkflobi.com/cognometry
Paper DOI: https://doi.org/10.5281/zenodo.19703527

Thanks,
Flobi / Fathom Lab
"""
    pr = gh(s, "POST", f"repos/{UPSTREAM}/pulls", json={
        "title": "docs: add hallucination detection notebook with styxx",
        "head": f"{me}:{branch}",
        "base": fork_branch,
        "body": pr_body,
    })
    print(f"\nPR opened: {pr['html_url']}")


if __name__ == "__main__":
    main()
