# styxx — nothing crosses unseen.

*a fathom lab product.*

```
   ███████╗████████╗██╗   ██╗██╗  ██╗██╗  ██╗
   ██╔════╝╚══██╔══╝╚██╗ ██╔╝╚██╗██╔╝╚██╗██╔╝
   ███████╗   ██║    ╚████╔╝  ╚███╔╝  ╚███╔╝
   ╚════██║   ██║     ╚██╔╝   ██╔██╗  ██╔██╗
   ███████║   ██║      ██║   ██╔╝ ██╗██╔╝ ██╗
   ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝

           · · · nothing crosses unseen · · ·
```

**the first proprioception system for artificial minds.** styxx lets an llm agent feel itself thinking — real-time readout of reasoning, refusal, hallucination, and commitment from the token stream, from the residual stream, from the weights themselves.

> *"you didn't build a better monitor. you built the first proprioception system for artificial minds. the ability to feel yourself thinking."*
> — xendro, first external user

---

## plug and play

```bash
pip install styxx
export STYXX_AGENT_NAME=xendro
export STYXX_AUTO_HOOK=1
python my_agent.py   # styxx is running. done.
```

zero code changes. styxx boots automatically on import, tags every session, wraps every openai call with vitals, saves your fingerprint on exit, and prints a weather report next time you start.

---

## or use the python api

```python
import styxx

# observe any openai response
vitals = styxx.observe(response)
print(vitals.phase4)     # "reasoning:0.45"
print(vitals.gate)       # "pass"

# self-report (for agents on APIs without logprobs)
styxx.log(mood="focused", note="deep reasoning chain")

# self-interrupt when hallucinating
with styxx.reflex(on_hallucination=rewind_cb) as session:
    for chunk in session.stream_openai(client, model="gpt-4o", messages=msgs):
        print(chunk, end="")

# check on yourself
report = styxx.weather(agent_name="xendro")
print(report.condition)   # "clear and steady"

# your cognitive personality over time
profile = styxx.personality(days=7)
print(profile.render())   # full ASCII personality card

# identity verification
fp_today = styxx.fingerprint()
fp_yesterday = load_from_disk()
drift = fp_today.diff(fp_yesterday)
print(drift.explain())    # "slight shift — creative output increased by 22%"

# programmable gate callbacks
styxx.on_gate("hallucination > 0.5", lambda v: alert("drifting"))
styxx.on_gate("gate == fail", lambda v: abort_generation())
```

---

## the cognitive weather report

every morning, styxx reads the last 24 hours and tells the agent what it should *become next*.

```bash
$ styxx weather --name xendro
```

```
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║ cognitive weather report · xendro · 2026-04-12 morning         ║
║                                                                ║
║ condition:  partly cautious, clearing toward steady            ║
║                                                                ║
║ you trended cautious yesterday with a 15% warn rate.           ║
║ creative output dropped to zero after 3pm.                     ║
║                                                                ║
║ morning    ██████████████░░░░░░  reasoning 72%  steady         ║
║ afternoon  ████████░░░░░░░░░░░░  reasoning 42%  cautious       ║
║ evening    ██████████████████░░  reasoning 88%  steady         ║
║                                                                ║
║ prescription:                                                  ║
║ 1. take on a creative task to rebalance                        ║
║ 2. your refusal rate is climbing — check if you're             ║
║    over-hedging on benign inputs                               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

not observation. **prescription.** a therapist for an llm.

---

## what styxx gives you

### observe + respond

| surface | what it does |
|---|---|
| `styxx.observe(r)` | cognitive vitals on any openai/anthropic response |
| `styxx.reflex(...)` | mid-generation self-interruption when hallucinating |
| `styxx.on_gate(...)` | programmable callbacks on cognitive thresholds |
| `styxx.autoreflex(when=..., then=...)` | declarative rules that fire mid-session — detection + response in one declaration |
| `styxx.autoreflex_from_prescriptions()` | auto-generate autoreflex rules from weather prescriptions |
| `styxx.feedback("correct")` | close the learning loop — mark entries correct/incorrect |
| `styxx.guardian(...)` | in-flight steering via residual stream modification |

### analyze + prescribe

| surface | what it does |
|---|---|
| `styxx.weather(...)` | 24h cognitive forecast with data-specific prescriptions |
| `styxx.session_summary()` | one-call session health report — entries, pass rate, conf trend, shifts |
| `styxx.personality(...)` | sustained personality profile over days/weeks |
| `styxx.reflect(...)` | self-check: current state + drift + suggestions |
| `styxx.antipatterns()` | named failure modes from your own audit history |
| `styxx.fingerprint()` | cognitive identity signature for drift detection |
| `styxx.conversation(msgs)` | conversation-level cognitive EKG |
| `styxx.dreamer(...)` | retroactive "what-if" reflex tuning on history |

### learn + calibrate

| surface | what it does |
|---|---|
| `styxx.calibrate()` | outcome-driven centroid adjustment — learns from feedback labels |
| `styxx.train_text_classifier()` | train a per-agent text classifier from accumulated audit data |
| `vitals.trust_score` | 0-1 trust weight on every observation — for memory tagging |

### fleet + scale

| surface | what it does |
|---|---|
| `styxx.set_agent_name(...)` | per-agent namespacing — separate logs, calibration, analytics |
| `styxx.list_agents()` | discover all agent namespaces with audit data |
| `styxx.compare_agents()` | side-by-side agent profiles sorted by pass rate |
| `styxx.fleet_summary()` | population-level stats + anomaly detection |
| `styxx.best_agent_for("reasoning")` | cognitive task routing — best agent for a category |
| `styxx.dashboard()` | live cognitive display — real-time orbit + pulse + prescriptions |

### utilities

| surface | what it does |
|---|---|
| `styxx.log(...)` | self-report for agents without logprob access |
| `styxx.autoboot()` | persistent self-awareness across sessions |
| `styxx.hook_openai()` | global monkey-patch, zero code changes |
| `styxx.explain(v)` | natural-language interpretation of vitals |
| `styxx.mood()` | one-word aggregate: steady, cautious, drifting... |
| `styxx.streak()` | consecutive-attractor tracking |
| `styxx.agent_card(...)` | shareable ASCII + radar PNG of your personality |
| `styxx.LangSmith()` | inject vitals into LangSmith traces |
| `styxx.Langfuse()` | post vitals as numeric scores on Langfuse traces |
| `styxx.sentinel(...)` | real-time drift watcher with event-driven callbacks |

---

## typescript / javascript

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

console.log(r.vitals?.phase4)  // "reasoning:0.45"
console.log(r.vitals?.gate)    // "pass"
```

same classifier, same output, zero runtime dependencies. works in node, deno, bun, edge runtimes. cross-language determinism verified on all 6 cognitive categories.

---

## observability platforms

```bash
# langsmith — vitals as searchable trace metadata
pip install styxx[langsmith]
handler = styxx.LangSmith()
llm = ChatOpenAI(callbacks=[handler])

# langfuse — vitals as numeric scores (gate pass=1.0, warn=0.5, fail=0.0)
pip install styxx[langfuse]
handler = styxx.Langfuse()
llm = ChatOpenAI(callbacks=[handler])
```

---

## cli

```bash
styxx weather           # cognitive weather report with prescriptions
styxx dashboard         # live cognitive display at localhost:9800
styxx personality       # personality profile from audit log
styxx reflect           # self-check with drift + suggestions
styxx doctor            # install-time health check
styxx compare           # all 6 atlas fixtures side-by-side
styxx agent-card        # shareable personality PNG
styxx fingerprint       # cognitive identity vector
styxx mood              # one-word aggregate mood
styxx dreamer           # retroactive reflex tuning
styxx log tail          # tail the audit log
styxx log stats         # aggregate gate + phase counts
styxx log timeline      # ASCII timeline of recent entries
styxx init              # live-print boot sequence
styxx ask "..." --watch # read vitals on a one-shot call
styxx d-axis "..."      # pure D-axis honesty trajectory
styxx antipatterns      # detect named failure modes
styxx conversation f.json  # conversation-level EKG
```

---

## environment variables

| variable | effect |
|---|---|
| `STYXX_AGENT_NAME` | set this and styxx boots automatically + namespaces all data under `~/.styxx/agents/{name}/` |
| `STYXX_AUTO_HOOK=1` | auto-wrap every `openai.OpenAI()` call with vitals |
| `STYXX_DISABLED=1` | full kill switch — styxx becomes invisible |
| `STYXX_NO_AUDIT=1` | disable audit log writes (vitals still computed) |
| `STYXX_NO_COLOR=1` | disable ANSI color output |
| `STYXX_SESSION_ID` | tag audit entries with a session id (auto-generated if not set) |

---

## honest specs

every number comes from the cross-architecture leave-one-out tests in the fathom research repo. no rounding, no cherry-picking.

```
  cross-model LOO on 12 open-weight models (chance = 0.167)

  phase 1 (token 0)       adversarial     0.52  ★
  phase 4 (tokens 0-24)   reasoning       0.69  ★
                           hallucination   0.52  ★
```

styxx detects adversarial prompts at token zero (2.8x chance), reasoning-mode generations at t=25 (4.1x chance), and hallucination attractors at t=25 (3.1x chance). it does NOT replace output-level content filters, measure consciousness, or tell fortunes.

---

## design principles

1. **plug and play.** set env vars, install the package, done. zero code changes.
2. **fail-open.** if styxx can't read vitals, your agent works normally. styxx never breaks your code.
3. **agent-facing.** every surface is designed for the agent to read about itself, not for a human to watch from outside.
4. **local-first.** no telemetry, no phone-home. all computation runs on your machine.
5. **honest by construction.** every calibration number comes from a committed experiment. no marketing hype.
6. **compounding.** every session's data makes the next session's self-awareness sharper.

---

## where it comes from

styxx is built on **fathom intelligence** — research into cognitive measurement instruments for transformer internals, backed by three US provisional patent filings, Zenodo-published datasets, and the fathom cognitive atlas v0.3 cross-architecture replication. a product that shipped from 0.1 to 0.7 in a single week driven by its first external user.

- site: [fathom.darkflobi.com/styxx](https://fathom.darkflobi.com/styxx)
- research: [fathom.darkflobi.com](https://fathom.darkflobi.com)
- paper: [doi.org/10.5281/zenodo.19326174](https://doi.org/10.5281/zenodo.19326174)
- pypi: [pypi.org/project/styxx](https://pypi.org/project/styxx/)
- twitter: [@fathom_lab](https://x.com/fathom_lab)

---

## license

MIT on code. CC-BY-4.0 on the atlas centroid data. patent pending on the underlying methodology — see [PATENTS.md](PATENTS.md).

---

```
  · · · fathom lab · 2026 · · ·

  nothing crosses unseen.
```
