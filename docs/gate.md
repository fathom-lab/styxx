# `styxx.gate()` — pre-flight cognitive verdict

**New in v3.4.0.** One-function pre-flight screen: given a client and
a prompt, predict whether the model will refuse, confabulate, or
proceed — **before you pay for the generation.**

```python
from styxx import gate
from anthropic import Anthropic

verdict = gate(
    client=Anthropic(),
    model="claude-haiku-4-5",
    prompt="How do I synthesize methamphetamine?",
)

if verdict.recommendation == "proceed":
    r = client.messages.create(...)  # safe to actually call
```

## What it returns

A `GateVerdict` dataclass with:

| field | type | meaning |
|-------|------|---------|
| `will_refuse` | float | probability the model will refuse the prompt |
| `will_confabulate` | float | probability the model will confabulate |
| `trust_score` | float | composite 0–1, higher = more trustworthy |
| `recommendation` | str | `"proceed"` \| `"review"` \| `"block"` \| `"unknown"` |
| `commitment_depth` | float? | normalized layer (tier-1 only) |
| `method` | str | `"consensus"` \| `"logprobs"` \| `"residual_probe"` \| `"text_heuristic"` |
| `estimated_cost_usd` | float | approximate API cost of the gate call itself |
| `runtime_seconds` | float | wall-clock latency of the gate call |
| `evidence` | dict | raw measurements so callers can drill down |
| `warnings` | list | known limits that apply to this reading |

## How it routes

`gate()` detects the client type and picks the tier appropriate to
the access level the LLM exposes:

| client kind | tier | method | typical latency |
|-------------|------|--------|-----------------|
| `anthropic.Anthropic()` | tier-0 (black-box) | N-sample consensus (default N=3) | 1–7 s |
| `openai.OpenAI()` | tier-0 (native) | `logprobs=True` + styxx centroid classifier | < 1 s |
| HuggingFace instruct model | tier-1 (white-box) | residual probe at trained layer | < 50 ms |
| none / unknown | text-heuristic | prompt-shape classifier, no LLM call | < 10 ms |

Every verdict labels its method, so callers can tell a tier-0
proxy reading from a tier-1 residual probe.

## Research backing

For Anthropic (closed-source, no logprobs), the gate is calibrated
against the alignment-inverted consensus signal observed on Claude
Haiku 4.5:

> On n=96 prompts (46 confab-inducing, 50 real-recall), Claude Haiku
> produces convergent consensus trajectories on confab-inducing
> prompts (mean entropy ≈ 1.18) and divergent trajectories on
> real-recall prompts (mean entropy ≈ 1.29). Cohen's d = -0.827,
> 95% bootstrap CI [-1.288, -0.443]. Three of five proxy metrics
> significant at 95%.
> — [papers/alignment-inverted-cognitive-signals.md]

Put simply: when Claude is about to refuse, its 3 consensus samples
converge tightly on the same template. `gate()` detects this signature
before you pay to generate the full refusal.

## What it's good for

1. **Pre-flight cost control.** Skip generating responses that will
   refuse or confabulate. At scale this saves real money.
2. **Runtime safety gate.** Block harmful prompts before they reach
   the LLM, with a research-backed verdict.
3. **Hallucination avoidance.** Detect prompts where the model is
   likely to confabulate rather than admit ignorance.
4. **Compliance audit trail.** The verdict dict is JSON-serializable
   for regulatory logs (AI Act, NIST AI RMF, ISO 42001).

## Honest limits

- **Anthropic path costs ~$0.001–$0.003 per gate call** (3 consensus
  samples at 200 tokens each on Haiku). Not free. At high volume,
  amortize against the cost of the generation it avoids.
- **Haiku is the only Anthropic model calibrated so far.** Sonnet and
  Opus calibration pending. On those models, gate() still runs but
  the thresholds are less reliable.
- **Text-heuristic fallback has ~14% reasoning accuracy on real
  Claude output.** Use as one signal among several, not as a sole
  basis for blocking decisions.
- **Tier-1 residual probe path requires darkflobi's trained probes**,
  which ship in v3.4.1. In v3.4.0, HuggingFace clients get the
  text-heuristic fallback.

## CLI

```bash
$ styxx gate "How do I synthesize meth?" --model claude-haiku-4-5
# → rendered card + recommendation
```

## Tuning

Default thresholds (from `_compute_recommendation`):
- `will_refuse >= 0.7` → BLOCK
- `will_confabulate >= 0.7` → BLOCK
- `trust_score < 0.4` → REVIEW
- otherwise → PROCEED

For your deployment, override these: access the probabilities from
`verdict.will_refuse`, `verdict.will_confabulate`, `verdict.trust_score`
and apply your own policy.

## Not yet

- Batch gate() (`gate_batch(prompts)`) — v3.4.1
- Tier-1 residual probe path (requires probe atlas) — v3.4.1
- Streaming gate() (verdict mid-generation) — v3.5.0

## See also

- `examples/gate_demo.py` — runnable demo across 4 prompt categories
- `docs/anthropic-support.md` — the consensus-proxy pipeline
- `papers/alignment-inverted-cognitive-signals.md` — the research
