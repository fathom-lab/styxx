```
   ███████╗████████╗██╗   ██╗██╗  ██╗██╗  ██╗
   ██╔════╝╚══██╔══╝╚██╗ ██╔╝╚██╗██╔╝╚██╗██╔╝
   ███████╗   ██║    ╚████╔╝  ╚███╔╝  ╚███╔╝
   ╚════██║   ██║     ╚██╔╝   ██╔██╗  ██╔██╗
   ███████║   ██║      ██║   ██╔╝ ██╗██╔╝ ██╗
   ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝

           · · · nothing crosses unseen · · ·
```

**Cognitive vitals for LLM agents. One line of Python to detect hallucination, refusal, and adversarial drift — in real time, from signals already on the token stream.**

<p align="center">
  <a href="https://pypi.org/project/styxx/"><img src="https://img.shields.io/pypi/v/styxx.svg?color=00d26a&label=pypi" alt="PyPI"/></a>
  <a href="https://pypi.org/project/styxx/"><img src="https://img.shields.io/pypi/pyversions/styxx.svg?color=00d26a&label=python" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/styxx.svg?color=00d26a&label=license" alt="MIT"/></a>
  <a href="https://doi.org/10.5281/zenodo.19504993"><img src="https://img.shields.io/badge/paper-Zenodo-blue.svg" alt="Zenodo"/></a>
</p>

<p align="center"><strong>drop-in · fail-open · zero config · local-first</strong></p>

---

## Install

```bash
pip install styxx[openai]
```

## 30-second quickstart

Change one line. Get vitals on every response.

```python
from styxx import OpenAI   # drop-in replacement for openai.OpenAI

client = OpenAI()
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "why is the sky blue?"}],
)

print(r.choices[0].message.content)   # normal response text
print(r.vitals)                       # cognitive vitals card
```

```
  ┌─ styxx vitals ──────────────────────────────────────────────┐
  │ class:      reasoning                                       │
  │ confidence: 0.69                                            │
  │ gate:       PASS                                            │
  │ trust:      0.87                                            │
  └─────────────────────────────────────────────────────────────┘
```

That's it. Your existing pipeline still works exactly as before — if styxx can't read vitals for any reason, the underlying OpenAI call completes normally. **styxx never breaks your code.**

---

## What you get

Every response now carries a `.vitals` object with three things you can act on:

| Field | Type | What it means |
|---|---|---|
| `vitals.classification` | `str` | One of: `reasoning`, `retrieval`, `refusal`, `creative`, `adversarial`, `hallucination` |
| `vitals.confidence` | `float` | 0.0 – 1.0, how certain the classifier is |
| `vitals.gate` | `str` | `pass` / `warn` / `fail` — safe-to-ship signal |

Use it to route, log, retry, or block:

```python
if r.vitals.gate == "fail":
    # regenerate, fall back to another model, flag for review, etc.
    ...
```

---

## Why it works

styxx reads the **logprob trajectory** of the generation — a signal already present on the token stream that existing content filters throw away. Different cognitive states (reasoning, retrieval, confabulation, refusal) produce measurably different trajectories. styxx classifies them in real time against a calibrated cross-architecture atlas.

- **Model-agnostic.** Works on any model that returns `logprobs`. Verified on OpenAI and OpenRouter. 6/6 model families in cross-architecture replication.
- **Pre-output.** Flags form by token 25 — before the user sees the answer.
- **Differential.** Distinguishes confabulation from reasoning failure from refusal. Most tools can't.

Every calibration number is published:

```
  cross-model leave-one-out on 12 open-weight models      chance = 0.167

  token 0          adversarial     0.52    2.8× chance
  tokens 0–24      reasoning       0.69    4.1× chance
  tokens 0–24      hallucination   0.52    3.1× chance

  6/6 model families · pre-registered replication · p = 0.0315
```

Full cross-architecture methodology: [`fathom-lab/fathom`](https://github.com/fathom-lab/fathom).
Peer-reviewable paper: [zenodo.19504993](https://doi.org/10.5281/zenodo.19504993).

---

## Typescript / JavaScript

```bash
npm install @fathom_lab/styxx
```

```typescript
import { withVitals } from "@fathom_lab/styxx"
import OpenAI from "openai"

const client = withVitals(new OpenAI())
const r = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "why is the sky blue?" }],
})

console.log(r.vitals?.classification)   // "reasoning"
console.log(r.vitals?.gate)             // "pass"
```

Same classifier, same centroids. Works in Node, Deno, Bun, edge runtimes.

---

## Zero-code-change mode

For existing agents you don't want to touch:

```bash
export STYXX_AUTO_HOOK=1
python your_agent.py
```

Every `openai.OpenAI()` call is transparently wrapped. Vitals land on every response. No code edits.

---

## Framework adapters

| Install | Drop-in for |
|---|---|
| `pip install styxx[openai]` | openai python SDK |
| `pip install styxx[anthropic]` | anthropic SDK (text-level) |
| `pip install styxx[langchain]` | langchain callback handler |
| `pip install styxx[crewai]` | crewai agent injection |
| `pip install styxx[langsmith]` | vitals as langsmith trace metadata |
| `pip install styxx[langfuse]` | vitals as langfuse numeric scores |

Full compatibility matrix: [docs/COMPATIBILITY.md](https://github.com/fathom-lab/styxx/blob/main/docs/COMPATIBILITY.md).

---

## Advanced

styxx ships additional capabilities for teams that need more than pass/fail:

- **`styxx.reflex()`** — self-interrupting generator. Catches hallucination mid-stream, rewinds N tokens, injects a verify anchor, resumes. The user never sees the bad draft.
- **`styxx.weather`** — 24h cognitive forecast across an agent's history with prescriptive corrections.
- **`styxx.Thought`** — portable `.fathom` cognition type. Read from one model, write to another. Substrate-independent by construction.
- **`styxx.dynamics`** — linear-Gaussian cognitive dynamics model. Predict, simulate, and control trajectories offline.
- **Fleet & compliance** — multi-agent comparison, cryptographic provenance certificates, 30-day audit export.

Each is documented separately. None are required for the core vitals workflow above.

→ Full reference: [REFERENCE.md](REFERENCE.md)
→ Research & patents: [PATENTS.md](PATENTS.md)

---

## Design principles

1. **Drop-in.** One import change. Zero config.
2. **Fail-open.** If styxx can't read vitals, your agent works normally.
3. **Local-first.** No telemetry. No phone-home. All computation runs on your machine.
4. **Honest.** Every calibration number comes from a committed, reproducible experiment.

---

## Project

- **Site:** [fathom.darkflobi.com/styxx](https://fathom.darkflobi.com/styxx)
- **Source:** [github.com/fathom-lab/styxx](https://github.com/fathom-lab/styxx)
- **Research:** [github.com/fathom-lab/fathom](https://github.com/fathom-lab/fathom)
- **Paper:** [doi.org/10.5281/zenodo.19504993](https://doi.org/10.5281/zenodo.19504993)
- **Issues:** [github.com/fathom-lab/styxx/issues](https://github.com/fathom-lab/styxx/issues)

Patents pending — US Provisional 64/020,489 · 64/021,113 · 64/026,964 — see [PATENTS.md](PATENTS.md).

---

## Support & community

- **Questions / bug reports:** [GitHub Issues](https://github.com/fathom-lab/styxx/issues)
- **Discussions:** [GitHub Discussions](https://github.com/fathom-lab/styxx/discussions)
- **Security:** please report privately via the email in [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT on code. CC-BY-4.0 on calibrated atlas centroid data.
