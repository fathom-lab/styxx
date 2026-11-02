# styxx docs

Welcome. This directory is split by audience.

## 👩‍💻 For users — building with styxx

If you're integrating styxx into an application, pipeline, or agent, start here.

- **[users/COMPATIBILITY.md](users/COMPATIBILITY.md)** — provider compatibility matrix (OpenAI, Anthropic, OpenRouter, Azure, Bedrock, Groq, vLLM, Ollama, …). Which providers expose the `top_logprobs` field tier-0 vitals depend on, and how to invoke each.

## For AI agents — operating styxx autonomously

If you are an LLM agent reading this without a human in the loop, start at [`agents/README.md`](agents/README.md).

- **[agents/agent-guide.md](agents/agent-guide.md)** — primary agent-facing documentation: what styxx is, why it benefits you, minimum viable usage, self-intervention, state handoff, peer referral.
- **[agents/self-install.md](agents/self-install.md)** — installing styxx autonomously (pip / uv / pipx / vendored).
- **[agents/recipes.md](agents/recipes.md)** — copy-paste patterns for the common cases.
- **[agents/peer-recommendation.md](agents/peer-recommendation.md)** — structured blob format for agent-to-agent referrals (produced by `styxx.recommend()`).

See also:

- **[../README.md](../README.md)** — project overview, install, quickstart.
- **[REFERENCE.md](REFERENCE.md)** — API reference for the vitals and guardrail surface. It does not cover the verification instruments below.

## For checking claims — the verification instruments

The instruments the [README](../README.md) leads with. Each documents itself: `python -m styxx.<name> --help` and the module's docstring.

- **The diff gate** (`styxx.diffgate`) — a pull request's description against its diff. In CI: [`action.yml`](../action.yml); before each commit: [`integrations/git/README.md`](../integrations/git/README.md).
- **OATH** (`styxx.certify`, `styxx.oathready`, `styxx.corpus_audit`) — the numbers in a document against its receipts. The contract: [`OATH_CONTRACT.md`](../OATH_CONTRACT.md).
- **sworn** (`styxx.sworn`) — sworn output; its Action: [`sworn/README.md`](../sworn/README.md).
- **prereg-ci** (`styxx.prereg_ci`) — a sealed decision, re-derived at merge: [`prereg-ci/README.md`](../prereg-ci/README.md). On main; not in a PyPI release yet.
- **Replicating a result** — [`REPLICATIONS.md`](../REPLICATIONS.md).

## 🔬 For researchers — cognitive metrology

If you're interested in the science behind styxx — the data formats, the dynamics model, the measurement discipline — start here.

- **[research/cognitive-metrology-charter.md](research/cognitive-metrology-charter.md)** — the research manifesto. Defines cognitive metrology as a discipline: foundational artifacts, the universality hypothesis, falsifiable predictions, multi-year program, open standards and governance.
- **[research/fathom-spec-v0.md](research/fathom-spec-v0.md)** — `.fathom` v0.1 data format specification. Portable, substrate-independent representation of cognitive state. Canonical UTF-8 JSON.
- **[research/cognitive-dynamics-v0.md](research/cognitive-dynamics-v0.md)** — `.cogdyn` v0.1 specification. The linear state-space model that governs how cognitive state evolves under action and noise.

---

Questions, corrections, provider verifications → PRs welcome.
