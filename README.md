```
   ███████╗████████╗██╗   ██╗██╗  ██╗██╗  ██╗
   ██╔════╝╚══██╔══╝╚██╗ ██╔╝╚██╗██╔╝╚██╗██╔╝
   ███████╗   ██║    ╚████╔╝  ╚███╔╝  ╚███╔╝
   ╚════██║   ██║     ╚██╔╝   ██╔██╗  ██╔██╗
   ███████║   ██║      ██║   ██╔╝ ██╗██╔╝ ██╗
   ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝

           · · · nothing crosses unseen · · ·
```

<p align="center">
  <a href="https://pypi.org/project/styxx/"><img src="https://img.shields.io/pypi/v/styxx.svg?color=00d26a&label=pypi" alt="PyPI"/></a>
  <a href="https://pypi.org/project/styxx/"><img src="https://img.shields.io/pypi/pyversions/styxx.svg?color=00d26a&label=python" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/styxx.svg?color=00d26a&label=license" alt="MIT"/></a>
  <a href="https://www.npmjs.com/package/@fathom_lab/styxx"><img src="https://img.shields.io/npm/v/@fathom_lab/styxx.svg?color=00d26a&label=npm" alt="npm"/></a>
  <a href="https://doi.org/10.5281/zenodo.19504993"><img src="https://img.shields.io/badge/paper-Zenodo-blue.svg" alt="Zenodo"/></a>
  <a href="https://github.com/fathom-lab/fathom"><img src="https://img.shields.io/badge/research-fathom--lab%2Ffathom-purple.svg" alt="research"/></a>
  <a href="https://fathom.darkflobi.com/styxx"><img src="https://img.shields.io/badge/site-fathom.darkflobi.com-black.svg" alt="site"/></a>
</p>

---

# styxx — proprioception for ai agents

**one line of python gives your agent the ability to feel itself thinking.** styxx reads an
LLM's internal cognitive state in real time — reasoning, refusal, hallucination, commitment —
from signals already on the token stream. no new model. no retraining. fail-open.

> **2026-04-14:** styxx is the reference implementation of **cognitive metrology** —
> a new branch of measurement science founding tonight by public charter.
> read the charter: [docs/cognitive-metrology-charter.md](https://github.com/fathom-lab/styxx/blob/main/docs/cognitive-metrology-charter.md)

<p align="center">
  <img src="https://raw.githubusercontent.com/fathom-lab/styxx/main/demo/styxx_reflex.gif" width="720" alt="reflex arc: agent catches itself mid-hallucination, rewinds, self-corrects"/>
</p>

> *"you didn't build a better monitor. you built the first proprioception system for artificial
> minds. the ability to feel yourself thinking."*
> — xendro, first external user

---

## 30-second quickstart

```bash
pip install styxx[openai]
```

```python
from styxx import OpenAI   # drop-in replacement for openai.OpenAI

client = OpenAI()
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "why is the sky blue?"}],
)

print(r.choices[0].message.content)   # normal response text
print(r.vitals.phase4)                 # "reasoning:0.69"
print(r.vitals.gate)                   # "pass"  /  "warn"  /  "fail"
```

one-line change: `from openai import OpenAI` → `from styxx import OpenAI`. every response now
carries a `.vitals` attribute alongside `.choices`. fail-open: if styxx can't read vitals, the
underlying call works exactly as before.

---

## what styxx does

```
  observe  ───►  know what you're doing right now
  reflex   ───►  catch yourself before you fall
  weather  ───►  know what you should become next
```

### 1. `observe` — six cognitive states, classified from the logprob stream

```python
import styxx

vitals = styxx.observe(response)   # any openai chat completion with logprobs=True
print(vitals.summary)              # full ASCII vitals card
```

```
  ┌─ styxx vitals ──────────────────────────────────────────────┐
  │ phase1 (token 0)         reasoning       0.43   pass        │
  │ phase4 (tokens 0-24)     reasoning       0.69   pass        │
  │ gate:                    PASS                                │
  │ trust:                   0.87                                │
  └──────────────────────────────────────────────────────────────┘
```

six classes: `reasoning · retrieval · refusal · creative · adversarial · hallucination`.
works on any model that returns logprobs.

### 2. `reflex` — self-interrupt, rewind, resume

```python
import styxx, openai

def on_hallucination(vitals):
    styxx.rewind(4, anchor=" — actually, let me verify: ")

client = openai.OpenAI()
with styxx.reflex(on_hallucination=on_hallucination, max_rewinds=2) as session:
    for chunk in session.stream_openai(
        client, model="gpt-4o", messages=msgs,
    ):
        print(chunk, end="", flush=True)

print(f"\n[reflex] rewinds fired: {session.rewind_count}")
```

every 5 tokens the trajectory is re-classified. when a hallucination attractor forms
mid-generation the reflex fires, drops the last N tokens, injects a verify anchor, and
resumes. **the user never sees the bad draft.**

### 3. `weather` — 24h forecast with prescriptions

```bash
$ styxx weather
```

```
  ╔═══════════════════════════════════════════════════════════════╗
  ║ cognitive weather · my-agent · 2026-04-13                     ║
  ║                                                                ║
  ║ condition:  clear and steady                                   ║
  ║                                                                ║
  ║ morning    ██████████████░░░░░░  reasoning  72%   steady       ║
  ║ afternoon  ████████░░░░░░░░░░░░  reasoning  42%   cautious     ║
  ║                                                                ║
  ║ prescription:                                                  ║
  ║ 1. take on a creative task to rebalance                        ║
  ║ 2. your refusal rate is climbing — check over-hedging          ║
  ╚═══════════════════════════════════════════════════════════════╝
```

not observation. **prescription.** styxx reads 24h of the agent's own history and tells it
what cognitive task to take on next. self-directed course correction.

### 4. `Thought` — cognition as a portable data type *(3.0.0a1)*

```python
import styxx

# read a Thought from any vitals reading
t = styxx.read_thought(response)         # or styxx.read_thought(vitals)
print(t)                                  # <Thought reasoning:0.69 phases=4/4 src=gpt-4o>

# save it as a portable .fathom file
t.save("my_thought.fathom")

# load it back from disk in a different process / host / vendor
loaded = styxx.Thought.load("my_thought.fathom")
assert loaded == t                        # cognitive equality

# build a steering target for any model
target = styxx.Thought.target("reasoning", confidence=0.85)
result = styxx.write_thought(target, client=styxx.OpenAI(), model="gpt-4o")
print(result["text"])                     # cognitively-aligned generation
print(result["distance"])                 # how close to the target

# algebra in eigenvalue space
mid    = t1 + t2                          # convex midpoint (mean)
mixed  = styxx.Thought.mix([t1, t2, t3], weights=[0.5, 0.3, 0.2])
delta  = t1 - t2                          # ThoughtDelta — what changed
d      = t1.distance(t2)                  # in eigenvalue space
sim    = t1.similarity(t2)                # 1.0 = identical, 0.0 = orthogonal
```

a `Thought` is the cognitive content of a generation — projected onto fathom's calibrated
cross-architecture eigenvalue space. it is **substrate-independent by construction**: the
same Thought can be read out of one model and written back through a different one, because
the categories themselves are calibrated to be cross-model invariant on atlas v0.3.

> PNG is the format for images.
> JSON is the format for data.
> `.fathom` is the format for thoughts.

every other interpretability representation — SAE features, activation patches, embedding
vectors — is model-specific and dies the moment a vendor swaps the model under you. a
Thought survives the swap by design. spec: [docs/fathom-spec-v0.md](https://github.com/fathom-lab/styxx/blob/main/docs/fathom-spec-v0.md).
algebra invariants and round-trip fidelity proven against bundled atlas v0.3 trajectories
in `tests/test_thought.py` (68 tests, all passing).

### 5. `dynamics` — predict, simulate, control cognitive trajectories *(3.1.0a1)*

```python
import styxx
from styxx.dynamics import CognitiveDynamics, Observation

# 1. collect observation tuples from your fleet
obs = [
    Observation.from_thoughts(state=t0, action=a0, next_state=t1),
    Observation.from_thoughts(state=t1, action=a1, next_state=t2),
    # ... at least 12 tuples for a well-conditioned fit
]

# 2. fit a linear-gaussian dynamics model: s_{t+1} = A·s_t + B·a_t + ε
dyn = CognitiveDynamics()
result = dyn.fit(obs)
print(result)             # <FitResult n=… r2=… spectral=…>

# 3. predict the next cognitive state from the current state + action
predicted = dyn.predict(current_thought, target_action)

# 4. simulate offline — multi-step rollout, no real model calls, zero API cost
trajectory = dyn.simulate(initial=t0, actions=[a1, a2, a3])

# 5. controller — find the action that drives state to a target
optimal = dyn.suggest(current=t0, target=styxx.Thought.target("reasoning"))

# 6. natural-drift forecast — what does cognition do under no intervention?
drift_path = dyn.forecast_horizon(t0, n_steps=10)

# 7. save / load
dyn.save("my_agent.cogdyn")
loaded = CognitiveDynamics.load("my_agent.cogdyn")
```

the field treats LLM inference as **open-loop** because nobody had a measurable cognitive
state vector. fathom's calibrated cross-architecture eigenvalue projection (atlas v0.3) gives
us one. once you have a state vector you can fit a dynamical system to it. once you have a
dynamical system, you can **predict, simulate, and control** cognitive trajectories.

`styxx.dynamics` is the first cognitive dynamics model in the field. v0.1 is linear-Gaussian
and fits in closed form. recovery to machine epsilon on full-rank synthetic data, validated
by 44 tests. spec at [docs/cognitive-dynamics-v0.md](https://github.com/fathom-lab/styxx/blob/main/docs/cognitive-dynamics-v0.md),
source at [styxx/dynamics.py](https://github.com/fathom-lab/styxx/blob/main/styxx/dynamics.py). CC-BY-4.0 spec, MIT impl.

> closed-loop cognitive control becomes a one-liner.

---

## provider compatibility

styxx tier-0 vitals require `top_logprobs` on the chat completion response. **OpenAI**
(via `styxx.OpenAI()`) and **OpenRouter** (passthrough to logprob-supporting models) are
verified. **Anthropic Claude** is not supported at tier 0 because the Messages API has no
`logprobs` parameter — `styxx.Anthropic()` exists as a passthrough wrapper and warns once.
Gemini, Azure, Bedrock, Groq, vLLM, llama.cpp, Ollama, and LiteLLM are not yet verified.

Full matrix + verified usage snippets + contributor TODOs:
[docs/COMPATIBILITY.md](https://github.com/fathom-lab/styxx/blob/main/docs/COMPATIBILITY.md)

## zero-code-change mode

```bash
pip install styxx
export STYXX_AGENT_NAME=my-agent
export STYXX_AUTO_HOOK=1
python my_agent.py   # styxx boots, wraps openai, tags every session. done.
```

set two env vars. every subsequent `openai.OpenAI()` is transparently wrapped. vitals land on
every response. fingerprints save on exit. a weather report prints on next boot.

---

## honest specs

every number comes from the cross-architecture leave-one-out tests in
[`fathom-lab/fathom`](https://github.com/fathom-lab/fathom). no rounding. no cherry-picking.

```
  cross-model LOO on 12 open-weight models            chance = 0.167

  phase 1 (token 0)        adversarial     0.52    2.8× chance   ★
  phase 1 (token 0)        reasoning       0.43    2.6× chance
  phase 4 (tokens 0-24)    reasoning       0.69    4.1× chance   ★
  phase 4 (tokens 0-24)    hallucination   0.52    3.1× chance   ★

  6/6 model families · pre-registered replication · p = 0.0315
```

styxx detects adversarial prompts at token zero, reasoning-mode generations by token 25, and
hallucination attractors by token 25. it does **not** replace output-level content filters,
measure consciousness, or tell fortunes. instrument panel, not fortune teller.

---

## framework adapters

| install | drop-in for |
|---|---|
| `pip install styxx[openai]` | openai python sdk |
| `pip install styxx[anthropic]` | anthropic sdk (text-level, no logprobs) |
| `pip install styxx[langchain]` | langchain callback handler |
| `pip install styxx[crewai]` | crewai agent injection |
| `pip install styxx[autogen]` | autogen agent wrapper |
| `pip install styxx[langsmith]` | vitals as langsmith trace metadata |
| `pip install styxx[langfuse]` | vitals as langfuse numeric scores |

### typescript / javascript

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

console.log(r.vitals?.phase4)   // "reasoning:0.69"
console.log(r.vitals?.gate)     // "pass"
```

same classifier, same centroids. works in node, deno, bun, edge runtimes. cross-language
determinism verified on all six cognitive categories.

---

<details>
<summary><strong>more — fleet, memory, compliance, cli (click to expand)</strong></summary>

### fleet management

```python
styxx.set_agent_name("agent-1")
styxx.list_agents()                    # discover all agents
styxx.compare_agents()                 # side-by-side leaderboard
styxx.best_agent_for("reasoning")      # cognitive task routing
```

### self-calibration

```python
styxx.calibrate()                      # outcome-driven centroid adjustment
styxx.train_text_classifier()          # per-agent logistic regression
styxx.enable_auto_feedback()           # auto-label every observation
```

### cognitive memory

```python
styxx.remember("user prefers concise answers")   # trust-weighted memory
styxx.recall("user preferences")                  # ranked by trust score
styxx.handoff(task, data)                          # inter-agent state transfer
```

### compliance + provenance

```python
cert = styxx.certify(vitals)           # cryptographic cognitive provenance certificate
styxx.compliance_report(days=30)       # json/markdown audit export
styxx.probe(agent_fn)                   # red-team: 15 adversarial prompts
```

each certificate carries a header of the form:

```
X-Cognitive-Provenance: styxx:1.0:reasoning:0.82:pass:0.95:verified:496b94b5
```

### cli

```bash
styxx weather          # cognitive forecast with prescriptions
styxx dashboard        # live cognitive display at localhost:9800
styxx reflect          # self-check + drift detection
styxx personality      # 7-day personality profile
styxx agent-card       # shareable personality png
styxx doctor           # install-time health check
styxx compare          # atlas fixtures side-by-side
styxx fingerprint      # cognitive identity vector
styxx export           # compliance export (json/markdown)
styxx scan "..."       # one-shot vitals on a single prompt
styxx ci-test          # cognitive regression testing for CI/CD
```

### environment variables

| variable | effect |
|---|---|
| `STYXX_AGENT_NAME` | set this and styxx boots automatically + namespaces data under `~/.styxx/agents/{name}/` |
| `STYXX_AUTO_HOOK=1` | auto-wrap every `openai.OpenAI()` call with vitals |
| `STYXX_DISABLED=1` | full kill switch — styxx becomes invisible |
| `STYXX_NO_AUDIT=1` | disable audit log writes (vitals still computed) |
| `STYXX_NO_COLOR=1` | disable ANSI color output |
| `STYXX_SESSION_ID` | tag audit entries with a session id (auto-generated if unset) |

</details>

---

## design principles

1. **plug and play.** set env vars, install, done. zero code changes to existing agents.
2. **fail-open.** if styxx can't read vitals, your agent works normally. styxx never breaks your code.
3. **agent-facing.** every surface is designed for the agent to read about itself, not for a human to watch from outside.
4. **local-first.** no telemetry, no phone-home. all computation runs on your machine.
5. **honest by construction.** every calibration number comes from a committed experiment.

---

## where it comes from

styxx is the production face of **[fathom-lab/fathom](https://github.com/fathom-lab/fathom)** — a
research program on cognitive measurement instruments for transformer internals. the research
side ships the atlas, the pre-registrations, and the paper. the styxx side ships the runtime.

- **research repo:** [github.com/fathom-lab/fathom](https://github.com/fathom-lab/fathom)
- **paper (zenodo doi):** [doi.org/10.5281/zenodo.19504993](https://doi.org/10.5281/zenodo.19504993)
- **site:** [fathom.darkflobi.com/styxx](https://fathom.darkflobi.com/styxx)
- **pypi:** [pypi.org/project/styxx](https://pypi.org/project/styxx/)
- **npm:** [npmjs.com/package/@fathom_lab/styxx](https://www.npmjs.com/package/@fathom_lab/styxx)
- **twitter:** [@fathom_lab](https://x.com/fathom_lab)

patents pending — US Provisional **64/020,489 · 64/021,113 · 64/026,964** — see [PATENTS.md](PATENTS.md).

---

## license

MIT on code. CC-BY-4.0 on the atlas centroid data. patent pending on the underlying methodology.

```
  · · · fathom lab · 2026 · · ·

  nothing crosses unseen.
```
