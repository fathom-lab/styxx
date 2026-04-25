# Quickstart Examples

**Drop-in cognitive observability for the most common LLM stacks.**

Each script is self-contained: copy → paste → `pip install -U styxx + the framework` → run. Working code, no abstractions, no setup steps. The kind of example you'd want when you're evaluating whether to add a tool to your stack.

## What's here

| File | Stack | What it shows |
|---|---|---|
| [`openai_chat.py`](openai_chat.py) | OpenAI chat completions | The minimum-viable integration — one-line swap from `openai.OpenAI` to `styxx.OpenAI` |
| [`anthropic_text_only.py`](anthropic_text_only.py) | Anthropic Claude | Tier-3 text-only mode for closed-API substrates without logprob access |
| [`langchain_agent.py`](langchain_agent.py) | LangChain tool-using agent | Wrap the agent function with `@styxx.profile`, get a flamegraph |
| [`crewai_yes_man_finder.py`](crewai_yes_man_finder.py) | CrewAI multi-agent | Detect which agent in a 4-agent crew is sycophantically agreeing with everything |
| [`ci_drift_check.py`](ci_drift_check.py) | Any provider, CI use | Fail your build if cognometric axes drift more than threshold (e.g., after a fine-tune) |
| [`../colab_60_second_demo.ipynb`](../colab_60_second_demo.ipynb) | Browser-only via Google Colab | Try styxx without installing anything locally |
| [`../cline_proxy.py`](../cline_proxy.py) | Cline / Claude Code editor extensions | A local FastAPI proxy that profiles every Anthropic API call your editor makes |

## Five-second mental model

```python
import styxx

@styxx.profile
def my_agent(task):
    return run_anything(task)   # your existing code, unchanged

result, profile = my_agent("...")

profile.summary       # one-line verdict
profile.faults        # list of cognitive failures localized to specific steps
profile.to_html()     # self-contained flamegraph (open in browser)
profile.to_langsmith() # drop into LangSmith trace
profile.to_datadog()   # drop into Datadog APM
```

## Citation

If you publish or build on this work:

```
Fathom Lab. Cognometric Fingerprint Specification v1.0.
Zenodo, 2026-04-24. doi:10.5281/zenodo.19746215.
```

## Links

- **Spec (v20):** https://doi.org/10.5281/zenodo.19746215
- **Software (v6.2.0):** https://doi.org/10.5281/zenodo.19758619
- **Robustness audit (v22):** https://doi.org/10.5281/zenodo.19761194
- **Concept (always-latest):** https://doi.org/10.5281/zenodo.19326174
- **Project:** https://fathom.darkflobi.com
- **Browser extension:** https://fathom.darkflobi.com/scope

## License

Examples: MIT. styxx package: MIT. Atlas data: CC-BY-4.0. Specification: CC-BY-4.0.

*Nothing crosses unseen.*
