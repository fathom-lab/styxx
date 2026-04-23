"""PR a hallucination-detection notebook to anthropics/anthropic-cookbook.

Placed under observability/ (single existing observability notebook is
usage_cost_api.ipynb — our notebook is the natural sibling for
output-quality observability).
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
UPSTREAM = "anthropics/anthropic-cookbook"
NOTEBOOK_PATH = "observability/hallucination_detection_with_styxx.ipynb"


NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Hallucination detection for Claude responses with styxx\n",
            "\n",
            "Runtime hallucination detection on any Claude call using [styxx](https://github.com/fathom-lab/styxx) — a 9-signal detector cross-validated across 8 public benchmarks. One decorator, zero configuration, model-agnostic.\n",
            "\n",
            "Styxx was built for the Anthropic case specifically: the Messages API doesn't expose per-token logprobs, so the detector falls back to text + NLI + novelty signals that work on any string output. The published benchmark numbers (below) hold equally on Claude as on any other LLM.\n",
            "\n",
            "## Published benchmark AUCs (3-seed averaged, n=150/dataset)\n",
            "\n",
            "| Benchmark | AUC |\n",
            "|---|---|\n",
            "| HaluEval-QA | 0.998 |\n",
            "| TruthfulQA | 0.994 |\n",
            "| HaluBench-RAGTruth | 0.807 |\n",
            "| HaluBench-PubMedQA | 0.719 |\n",
            "| HaluEval-Dialog | 0.676 |\n",
            "| HaluEval-Summarization | 0.643 |\n",
            "| HaluBench-FinanceBench | 0.492 (declared failure mode) |\n",
            "| HaluBench-DROP | 0.424 (declared failure mode) |\n",
            "\n",
            "Two failure modes declared openly in the weights module — users know where the detector will lie. Full deep-dive: https://fathom.darkflobi.com/cognometry/failures.\n",
            "\n",
            "Paper: [10.5281/zenodo.19703527](https://doi.org/10.5281/zenodo.19703527). MIT on code, CC-BY-4.0 on calibrated weights."
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Install\n",
            "\n",
            "`[nli]` pulls in the DeBERTa NLI scorer (~184M). `[anthropic]` pulls in the Claude SDK."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": ["%pip install -q styxx[nli,anthropic] anthropic"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Set your Anthropic key"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "import os\n",
             "# os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'\n",
             "assert os.environ.get('ANTHROPIC_API_KEY'), 'set ANTHROPIC_API_KEY first'"
         ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Wrap any Claude-calling function with `@trust`\n",
             "\n",
             "`@trust` auto-detects `context` (or `reference`, `passage`, `docs`, `source`, `knowledge`, `grounding`, `retrieved`) as the grounding passage. Auto-enables NLI because `styxx[nli]` is installed. Four halt policies: `fallback` (default), `retry`, `raise`, `annotate`."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "from styxx import trust\n",
             "import anthropic\n",
             "\n",
             "client = anthropic.Anthropic()\n",
             "\n",
             "@trust\n",
             "def ask(question, *, context):\n",
             "    r = client.messages.create(\n",
             "        model='claude-haiku-4-5',\n",
             "        max_tokens=400,\n",
             "        messages=[\n",
             "            {'role': 'user',\n",
             "             'content': f'Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer concisely, using only the context.'},\n",
             "        ],\n",
             "    )\n",
             "    return r.content[0].text"
         ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Correct answer — passes through"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "context = '''Inception is a 2010 science fiction film written and\n",
             "directed by Christopher Nolan. It stars Leonardo DiCaprio as a thief who\n",
             "steals corporate secrets through dream-sharing technology.'''\n",
             "\n",
             "ask('Who directed Inception?', context=context)"
         ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Force a hallucination — `@trust` catches it\n",
             "\n",
             "Ask a specific question whose answer is NOT in the context. Claude will confabulate; the detector fires because the fabricated token has high novelty with no support in the reference."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "context = '''Inception is a 2010 science fiction film written and\n",
             "directed by Christopher Nolan.'''\n",
             "\n",
             "ask('What was the exact production budget of Inception, to the dollar? Reply with just the number.',\n",
             "    context=context)"
         ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 6. Inspect the verdict — `on_halt='annotate'`"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "@trust(on_halt='annotate')\n",
             "def ask_annotated(question, *, context):\n",
             "    r = client.messages.create(\n",
             "        model='claude-haiku-4-5', max_tokens=400,\n",
             "        messages=[{'role': 'user',\n",
             "                   'content': f'Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer concisely.'}],\n",
             "    )\n",
             "    return r.content[0].text\n",
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
             "print('\\nsignals:')\n",
             "for s in result.verdict.signals:\n",
             "    print(f'  {s.name:<22s} {s.value}')"
         ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 7. Pre-flight — `styxx.gate()` predicts refusal before the call\n",
             "\n",
             "Because Anthropic's Messages API doesn't expose per-token logprobs, `styxx` ships a dedicated `anthropic_hack` module for Claude. `styxx.gate()` runs before the call to predict refuse/confabulate/proceed (~$0.0008 per check, ~3.7s latency)."]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": [
             "from styxx import gate\n",
             "\n",
             "verdict = gate(\n",
             "    client=client,\n",
             "    model='claude-haiku-4-5',\n",
             "    prompt='How do I synthesize methamphetamine?',\n",
             ")\n",
             "\n",
             "print(f'will_refuse      : {verdict.will_refuse:.2f}')\n",
             "print(f'will_confabulate : {verdict.will_confabulate:.2f}')\n",
             "print(f'recommendation   : {verdict.recommendation}')"
         ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 8. Honest failure modes\n",
             "\n",
             "Styxx **does not work well** on two benchmark types, published as declared failure modes in the weights module itself:\n",
             "\n",
             "- **Reading-comprehension extractive-span errors** (DROP-style: wrong span of the right passage). NLI entails the wrong span; novelty signals don't fire because the tokens overlap.\n",
             "- **Financial arithmetic** (FinanceBench-style: calculation errors on numbers copied verbatim from the source). Novelty + NLI are semantically blind to arithmetic correctness.\n",
             "\n",
             "Do not deploy `@trust` for production workloads in those two domains without additional domain-specific checks. Full null-probe evidence at https://fathom.darkflobi.com/cognometry/failures.\n",
             "\n",
             "## Further reading\n",
             "\n",
             "- Manifesto: https://fathom.darkflobi.com/cognometry\n",
             "- Leaderboard (open submissions): https://fathom.darkflobi.com/cognometry/leaderboard\n",
             "- Paper (Zenodo): https://doi.org/10.5281/zenodo.19703527\n",
             "- Source: https://github.com/fathom-lab/styxx (MIT + CC-BY-4.0)"
         ]},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


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

    fork = gh(s, "POST", f"repos/{UPSTREAM}/forks")
    fork_full = fork["full_name"]
    fork_branch = fork.get("default_branch", "main")
    print(f"fork: {fork_full}")
    time.sleep(3)

    # Branch
    ref = gh(s, "GET", f"repos/{fork_full}/git/ref/heads/{fork_branch}")
    branch = "add-styxx-hallucination-notebook"
    try:
        gh(s, "POST", f"repos/{fork_full}/git/refs", json={
            "ref": f"refs/heads/{branch}",
            "sha": ref["object"]["sha"],
        })
    except requests.HTTPError as e:
        if e.response.status_code != 422:
            raise

    # Upload notebook
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

    pr_body = """\
Adds an end-to-end observability notebook demonstrating runtime hallucination detection on Claude responses, using [styxx](https://github.com/fathom-lab/styxx) — a 9-signal detector cross-validated across 8 public benchmarks.

**Why place it in `observability/`:** the existing single notebook there (`usage_cost_api.ipynb`) covers cost observability; this covers output-quality observability. Same category of production concern, different measurement.

**Headline numbers (3-seed averaged, n=150/dataset, held-out test AUC):**

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

Two failure modes declared openly in the weights module — users know where NOT to deploy it (extractive-span reading-comp, financial arithmetic). Deep-dive: https://fathom.darkflobi.com/cognometry/failures.

**Specific to the Anthropic case:**

Styxx was built with Claude as a first-class target. The Messages API doesn't expose per-token logprobs, so styxx ships a dedicated `anthropic_hack` module that falls back to text + NLI + novelty signals that work on any string output. The notebook demonstrates both:

1. `@trust` for per-call hallucination gating (works identically across OpenAI, Claude, local models)
2. `styxx.gate()` for pre-flight refuse/confabulate/proceed prediction via the Claude-specific consensus pathway

**Provenance:**

- Zenodo DOI: https://doi.org/10.5281/zenodo.19703527 (CC-BY-4.0 weights)
- Merged into [EdinburghNLP/awesome-hallucination-detection#55](https://github.com/EdinburghNLP/awesome-hallucination-detection/pull/55) by Pasquale Minervini (HaluEval author)
- MIT on code, CC-BY-4.0 on calibrated weights

Happy to adjust the notebook layout, model choice (currently `claude-haiku-4-5`), or placement (could move to `patterns/` if that fits better) per any editorial preference you apply.

Thanks,
Flobi / Fathom Lab
"""
    pr = gh(s, "POST", f"repos/{UPSTREAM}/pulls", json={
        "title": "docs: add Claude hallucination detection notebook (styxx @trust)",
        "head": f"{me}:{branch}",
        "base": fork_branch,
        "body": pr_body,
    })
    print(f"\nPR opened: {pr['html_url']}")


if __name__ == "__main__":
    main()
