# styxx — API reference

Technical reference for advanced users. Every public symbol, signature, and a minimal example.

For narrative intros see [`README.md`](README.md). For data-format and dynamics specs see [`docs/research/`](docs/research/). For provider compatibility see [`docs/users/COMPATIBILITY.md`](docs/users/COMPATIBILITY.md).

## Contents

- [Core API](#core-api)
  - [`styxx.OpenAI`](#styxxopenai)
  - [`styxx.Anthropic`](#styxxanthropic)
  - [`styxx.Raw`](#styxxraw)
  - [`styxx.observe`](#styxxobserve) / [`observe_raw`](#styxxobserve_raw) / [`watch`](#styxxwatch)
  - [`styxx.Vitals`](#styxxvitals)
  - [`styxx.is_concerning`](#styxxis_concerning) / [`explain`](#styxxexplain)
  - [`styxx.hook_openai`](#styxxhook_openai)
- [Reflex](#reflex)
  - [`styxx.reflex`](#styxxreflex)
  - [`styxx.rewind`](#styxxrewind) / [`abort`](#styxxabort)
  - [`styxx.on_gate`](#styxxon_gate) / [`autoreflex`](#styxxautoreflex)
  - [`styxx.guardian`](#styxxguardian)
- [Analytics (Weather, Mood, Fingerprint)](#analytics)
  - [`styxx.weather`](#styxxweather)
  - [`styxx.personality`](#styxxpersonality) / [`reflect`](#styxxreflect) / [`mood`](#styxxmood)
  - [`styxx.antipatterns`](#styxxantipatterns)
  - [`styxx.conversation`](#styxxconversation)
  - [`styxx.timeline`](#styxxtimeline) / [`fingerprint`](#styxxfingerprint)
- [Thought (.fathom)](#thought-fathom)
- [Dynamics (.cogdyn)](#dynamics-cogdyn)
- [Fleet](#fleet)
- [Memory & Handoff](#memory--handoff)
- [Compliance](#compliance)
- [Learning](#learning)
- [Operations & Diagnostics](#operations--diagnostics)
- [Scan (tier 2)](#scan-tier-2)
- [CLI](#cli)
- [Environment variables](#environment-variables)
- [Extras (install matrix)](#extras-install-matrix)

---

## Core API

The smallest useful surface: wrap your LLM client, get a `Vitals` object on every response.

### `styxx.OpenAI`

```python
from styxx import OpenAI
client = OpenAI(api_key=..., base_url=None)
```

Drop-in replacement for `openai.OpenAI`. Auto-injects `logprobs=True, top_logprobs=5` on every chat-completion call and attaches `.vitals` to the response.

```python
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "why is the sky blue?"}],
)
print(r.choices[0].message.content)
print(r.vitals.phase4, r.vitals.gate)  # "reasoning:0.69" "pass"
```

Works with any OpenAI-compatible endpoint (OpenRouter, Groq, vLLM, llama.cpp, Ollama, Azure, LiteLLM) via `base_url`. See [`docs/users/COMPATIBILITY.md`](docs/users/COMPATIBILITY.md) for verified providers.

### `styxx.Anthropic`

```python
from styxx import Anthropic
client = Anthropic()
r = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024, messages=[...])
```

Passthrough wrapper. Anthropic's Messages API does not expose `top_logprobs`, so `r.vitals` is `None` — a one-time stderr warning is emitted on first use. Use `Raw` or route via OpenAI-compatible gateway for vitals.

### `styxx.Raw`

```python
from styxx import Raw
client = Raw()
vitals = client.observe(text="...", logprobs=[...], top_logprobs=[[...], ...])
```

Bring-your-own-trajectory adapter. Accepts pre-captured logprob arrays from any source (custom gateway, cached traces, non-standard SDK).

### `styxx.observe`

```python
def observe(response, *, agent_name: str | None = None, log: bool = True) -> Vitals | None
```

Extract vitals from any OpenAI-shaped response object.

```python
import openai, styxx
r = openai.OpenAI().chat.completions.create(..., logprobs=True, top_logprobs=5)
vitals = styxx.observe(r)
```

### `styxx.observe_raw`

```python
def observe_raw(entropy: list[float], logprob: list[float], top2: list[float]) -> Vitals
```

Lowest-level entry point. Feed three aligned per-token arrays; get a `Vitals` back. Useful for offline trajectory scans.

### `styxx.watch`

```python
with styxx.watch(agent_name="my-agent") as w:
    for chunk in stream:
        w.feed(chunk)
    vitals = w.vitals
```

Context manager for streaming vitals while tokens are still being generated.

### `styxx.Vitals`

Dataclass returned by every observation.

| attr | type | meaning |
|---|---|---|
| `phase1` | `str` | e.g. `"reasoning:0.28"` — mid-generation category + confidence |
| `phase4` | `str` | e.g. `"hallucination:0.45"` — final category + confidence |
| `gate` | `"pass" \| "warn" \| "fail"` | policy verdict |
| `trust_score` | `float` | 0.0–1.0 aggregate trust |
| `summary` | `str` | ASCII card |
| `as_dict()` | `dict` | JSON-serializable |

### `styxx.is_concerning`

```python
def is_concerning(vitals: Vitals) -> bool
```

Convenience boolean. True when `gate != "pass"` or category indicates risk.

### `styxx.explain`

```python
def explain(vitals: Vitals) -> str
```

Plain-English explanation of what the vitals mean.

### `styxx.hook_openai`

```python
styxx.hook_openai()     # monkey-patches openai globally
styxx.unhook_openai()
styxx.hook_openai_active()  # -> bool
```

Or set `STYXX_AUTO_HOOK=1` to hook automatically at import.

---

## Reflex

Mid-generation self-interruption. The agent catches itself before finishing a bad sentence.

### `styxx.reflex`

```python
with styxx.reflex(
    on_hallucination: Callable | None = None,
    on_refusal: Callable | None = None,
    on_warn: Callable | None = None,
) as s:
    for chunk in s.stream_openai(client, model=..., messages=...):
        print(chunk, end="")
```

Arms a `ReflexSession`. Callbacks receive the current vitals and may invoke `rewind()` or `abort()`.

### `styxx.rewind`

```python
def rewind(n_tokens: int, anchor: str | None = None) -> RewindSignal
```

Re-emit generation from N tokens back, optionally prepending an anchor string (e.g. `"let me verify: "`).

### `styxx.abort`

```python
def abort(reason: str) -> AbortSignal
```

Terminate generation immediately with a reason tag (logged in the audit trail).

### `styxx.on_gate`

```python
styxx.on_gate("hallucination > 0.5", callback)
styxx.remove_gate(id) ; styxx.clear_gates() ; styxx.list_gates()
```

Register a programmable gate callback against a live vitals expression.

### `styxx.autoreflex`

```python
styxx.autoreflex(when="hallucination > 0.4", then=lambda s: s.rewind(4, anchor="actually, "))
styxx.autoreflex_from_prescriptions()  # generate rules from weather forecast
styxx.list_autoreflex() ; styxx.remove_autoreflex(id) ; styxx.clear_autoreflex()
```

Declarative rule layer over `reflex`.

### `styxx.guardian`

```python
with styxx.guardian() as g:
    ...
```

Higher-level supervisor that composes reflex + autoreflex + sentinel into one context manager. Tier-2 builds can also apply in-flight steering (experimental; see [`docs/research/cognitive-metrology-charter.md`](docs/research/cognitive-metrology-charter.md)).

---

## Analytics

Post-hoc analysis of the audit log.

### `styxx.weather`

```python
report = styxx.weather()          # 24h forecast
report.condition                  # "clear and steady"
report.prescriptions              # ["take on a creative task", ...]
report.trends                     # {"reasoning": "rising", ...}
```

### `styxx.personality`

```python
styxx.personality(days: int = 7) -> Personality
```

Long-window traits (exploration vs. exploitation, volatility, strength categories).

### `styxx.reflect`

```python
styxx.reflect() -> ReflectionReport
```

Self-check + drift report against recent baseline.

### `styxx.mood`

```python
styxx.mood() -> str
styxx.streak() -> Streak
```

Current mood string and consecutive-same-category streak.

### `styxx.antipatterns`

```python
styxx.antipatterns() -> list[AntiPattern]
```

Named failure modes detected in recent history (e.g. "confidence-spiral", "refusal-loop").

### `styxx.conversation`

```python
styxx.conversation(messages: list[dict]) -> ConversationResult
```

Full-conversation EKG: trajectory of vitals across a multi-turn chat.

### `styxx.timeline`

```python
styxx.timeline() -> Timeline
```

Mood + category timeline (sparkline-renderable).

### `styxx.fingerprint`

```python
fp = styxx.fingerprint()
fp.diff(other_fp) -> dict        # identity drift
styxx.agent_card() -> AgentCard  # shareable personality card (PNG via [agent-card] extra)
```

---

## Thought (.fathom)

Portable, substrate-independent cognitive state. Full spec: [`docs/research/fathom-spec-v0.md`](docs/research/fathom-spec-v0.md).

```python
from styxx import Thought, PhaseThought, read_thought, write_thought
from styxx import FATHOM_FORMAT, FATHOM_VERSION, ATLAS_VERSION

t = read_thought("demo/thoughts/reasoning.fathom")
write_thought(t, "out.fathom")
t2 = t.delta(other)   # ThoughtDelta
```

Canonical sort-keys UTF-8 JSON, no BOM. Conformance tests live in [`tests/`](tests/).

---

## Dynamics (.cogdyn)

Linear state-space model of cognitive evolution. Full spec: [`docs/research/cognitive-dynamics-v0.md`](docs/research/cognitive-dynamics-v0.md).

```python
from styxx import CognitiveDynamics, Observation, synthetic_observations
from styxx import thought_to_state, state_to_thought, COGDYN_FORMAT, COGDYN_VERSION

obs = synthetic_observations(n=200)
dyn = CognitiveDynamics.fit(obs)   # OLS on x_{t+1} = A x_t + B u_t + eps
pred = dyn.predict(x0, actions=[u1, u2, u3])
```

---

## Fleet

Multi-agent cognitive routing.

```python
styxx.set_agent_name("xendro")
styxx.list_agents()                      # -> list[AgentProfile]
styxx.compare_agents()                   # -> AgentComparison
styxx.fleet_summary()                    # -> FleetSummary
styxx.best_agent_for("reasoning")        # -> str
```

---

## Memory & Handoff

### Memory (trust-weighted)

```python
styxx.remember("user prefers concise answers")
styxx.recall("user preferences") -> list[RecallResult]
styxx.memories() ; styxx.memory_stats()
```

Memories are tagged with the vitals at write time and scored by relevance × trust at recall.

### Handoff

```python
env = styxx.handoff(task="analyze data", data={...})  # -> HandoffEnvelope
ctx = styxx.receive(env)
```

Package cognitive context for transfer between agents.

---

## Compliance

### `styxx.certify` / `styxx.verify`

```python
cert = styxx.certify(vitals) -> CognitiveCertificate
cert.as_compact()                 # X-Cognitive-Provenance header value
styxx.verify(cert_dict) -> VerificationResult
```

### `styxx.compliance_report`

```python
styxx.compliance_report(days=30) -> ComplianceReport
```

### `styxx.probe`

```python
styxx.probe(agent_fn) -> ProbeReport   # 15-prompt red-team suite
```

### `styxx.regression_test` / `create_baseline`

```python
base = styxx.create_baseline() -> Baseline
styxx.regression_test(min_pass=0.80) -> RegressionResult
```

### SLA + notifications

```python
with styxx.cognitive_sla(min_pass_rate=0.8):
    ...
styxx.assert_healthy(min_pass_rate=0.7)
styxx.on_anomaly("https://hooks.slack.com/...")
styxx.notify_on_fail(callback)
styxx.clear_notifications()
```

---

## Learning

```python
styxx.feedback("correct" | "incorrect")
styxx.enable_auto_feedback()   # STYXX_AUTO_FEEDBACK=1
styxx.disable_auto_feedback()
styxx.calibrate() -> CalibrationResult                # centroid shift
styxx.train_text_classifier() -> TrainResult          # per-agent text model
styxx.optimize(apply=True)                            # auto-tune rules
```

Recommended loop: `autoboot()` → `enable_auto_feedback()` → train classifier at ~50 labels → calibrate at ~500.

---

## Operations & Diagnostics

```python
styxx.autoboot(agent_name="my-agent")
styxx.dashboard()
styxx.sentinel(callback, window_s=300) -> Sentinel
styxx.session_id() ; styxx.set_session(id) ; styxx.data_dir()
styxx.log_stats() ; styxx.log_timeline() ; styxx.load_audit()
styxx.trace(...)                                      # structured event trace
styxx.set_context(...) ; styxx.current_context()
styxx.expect(cats) ; styxx.unexpect(cats) ; styxx.expected_categories()
styxx.set_mood(...) ; styxx.gate_multiplier(...)
```

---

## Scan (tier 2)

SAE-level cognitive measurement. Requires `pip install 'styxx[tier2]'` + GPU.

```python
from styxx.scan import cognitive_scan
r = cognitive_scan("why is the sky blue?")

r.weighted_depth      # K — layer center of mass
r.c_delta             # C — concept lock-in (late − early)
r.s_early             # S — commitment strength (IPR)
r.layer_profile       # {layer: feature_count}
r.coherence ; r.n_features ; r.compute_time_s
```

Per-axis: `inst.measure_k(prompt)`, `measure_c`, `measure_s`, `measure_trajectory`.

The tier stack:

| tier | instrument | deps | status |
|---|---|---|---|
| 0 | logprob classifier | numpy | cross-model, 2.8–4.1× chance |
| 1 | D-axis honesty | torch, transformer-lens | open-weight, residual stream |
| 2 | K/C/S SAE | circuit-tracer + GPU | p=0.000051 |
| 3 | in-flight steering | circuit-tracer + GPU | experimental |

Each tier includes all lower tiers.

---

## CLI

```
styxx init                          # boot, create data dir, claim agent name
styxx doctor                        # environment / import health check
styxx tier                          # show active tiers
styxx ask --watch "prompt"          # one-shot live call
styxx ask --demo-kind reasoning     # demo w/o API key

styxx scan "prompt"                 # tier-2 K/C/S scan
styxx scan --trajectory "prompt"    # include S_early trajectory
styxx scan --compare "p1" "p2"
styxx scan --batch in.jsonl --out out.jsonl
styxx scan --bridge "prompt"        # tier-0 vs tier-2 side-by-side
styxx scan --layers "prompt"        # full layer profile
styxx scan --json "prompt"
styxx scan --legacy trajectory.json # re-scan a saved trajectory

styxx weather                       # 24h forecast
styxx personality                   # 7-day personality profile
styxx reflect                       # self-check + drift
styxx mood
styxx antipatterns
styxx conversation chat.json
styxx dreamer                       # retroactive reflex tuning
styxx timeline
styxx fingerprint [diff]
styxx agent-card

styxx compare                       # 6 bundled atlas fixtures
styxx compare-agents

styxx dashboard                     # live TUI
styxx log tail | stats | timeline | rotate

styxx export --days 30 --format json        # compliance export
styxx ci-test --min-pass 0.80               # CI/CD gate
styxx ci-baseline
styxx publish                               # push to remote dashboard
```

Run `styxx --help` or `styxx <command> --help` for exhaustive flags.

---

## Environment variables

| variable | effect |
|---|---|
| `STYXX_AGENT_NAME` | set agent identity + auto-boot |
| `STYXX_AUTO_HOOK=1` | auto-wrap every `openai.OpenAI()` at import |
| `STYXX_AUTO_FEEDBACK=1` | auto-label every observation from heuristics |
| `STYXX_DISABLED=1` | full kill switch — all entry points no-op |
| `STYXX_NO_AUDIT=1` | disable audit-log writes |
| `STYXX_NO_COLOR=1` | disable ANSI color |
| `STYXX_SESSION_ID` | custom session tag |

---

## Extras (install matrix)

```bash
pip install styxx              # core (numpy only) — tier 0
pip install styxx[openai]      # OpenAI drop-in wrapper
pip install styxx[langchain]   # LangChain callback handler
pip install styxx[crewai]      # CrewAI agent injection
pip install styxx[autogen]     # AutoGen agent wrapper
pip install styxx[langsmith]   # LangSmith trace metadata
pip install styxx[langfuse]    # Langfuse numeric scores
pip install styxx[tier1]       # D-axis honesty (transformer-lens)
pip install styxx[tier2]       # K/C/S SAE instruments (circuit-tracer + GPU)
pip install styxx[agent-card]  # personality card PNG renderer (Pillow)
```

---

For the discipline behind the instruments, see the charter: [`docs/research/cognitive-metrology-charter.md`](docs/research/cognitive-metrology-charter.md).

```
· · · fathom lab · 2026 · · ·
nothing crosses unseen.
```
