# Cognometry — open research agenda, 2026

**2026-04-23. The next-24-months map of ambitious bets, one paragraph each.
These are the moves that make cognometry a defensible research program
rather than a one-paper, one-package moment.**

---

## Bet 1 — Adversarial drift detection in production

Every existing detector, ours included, targets hallucination and refusal.
Nobody ships runtime detection of **prompt injection or adversarial
steering** as a first-class signal. A user-inserted hostile token
produces a measurable residual-stream deviation against a known-safe
reference trajectory. Build the diff: `trajectory(prompt + attack) −
trajectory(prompt alone)`. Calibrate the deviation against a public
jailbreak corpus (JailbreakBench, AdvBench). A production-grade
adversarial-drift signal lets `@trust` emit `ATTACKED` as a verdict
class — the same way `REFUSE` is a class today. Blocking this
universally solves a class of production vulnerabilities nobody
is attacking at runtime. **v4.2 scope.**

## Bet 2 — Meta-cognometric guardians

Have Claude Sonnet 4.6 read Opus 4.7's residual stream and flag
when Opus is about to hallucinate. This is **cross-model cognitive
observation**. Requires: API access to residuals (we have HF, Anthropic
doesn't expose them yet), but a protocol-level proxy works via
consensus signals today. Build the guardian-observer pattern: one model
watches the other's output trajectory and interrupts. First proof of
concept: Haiku-4.5 guarding Opus-4.7 through the API using consensus
disagreement as the cognitive-state signal. If this works, every
production pipeline gains a cheap "co-pilot that watches."
**v5.0 scope.**

## Bet 3 — Online calibration via user feedback

The detector's weights are frozen at ship time. Every real `@trust`
deployment has a stream of user corrections ("this response was fine
actually" / "this was hallucinated, you missed it"). Log those. Retrain
a small adapter over the existing signal features every N hours.
Users get a detector that improves during use, pinned to THEIR domain
distribution. This is the product moat — the calibration adapts;
competitors ship static weights. **v4.3 scope; one engineering
sprint.**

## Bet 4 — Cognitive fingerprint arbitrage

Every model produces a measurable cognitive fingerprint — a distribution
over its refusal/confab/retrieval/reasoning signatures on a standardized
probe set. Ship a fingerprinter that classifies "which model is this?"
from N tokens of output. **Real-world use case:** detect when an API
provider silently downgrades GPT-4o to GPT-4o-mini. Detect when an
agent framework swaps in a different model mid-pipeline. Cognitive
model-ID: a new primitive no one ships. **v4.4 scope.**

## Bet 5 — The cognometry index

Rank public hallucination benchmarks by intrinsic difficulty under the
current signal stack. We already have an 8-benchmark audit with
mean AUC 0.719 and documented failure modes. Extend to 20+ benchmarks,
publish per-benchmark difficulty scores, make the index a standing
resource the field can cite for benchmark selection. Host at
`fathom.darkflobi.com/cognometry/index`. This is the **arxiv for
cognometric difficulty** — a service to every other lab building in
this space. **v4.5 scope; continuous maintenance.**

## Bet 6 — Cognitive temperature as first-class

`styxx.temperature` already exists as a one-off module. Make it the
default output of `@trust`: every call returns not just PASS/FAIL but
a continuous cognitive-temperature reading — how confident, how
creative, how memorized the response was. This changes the API
conversation from binary safety to continuous calibrated confidence.
The product positioning shifts from "hallucination detector" to
"cognitive confidence meter" — broader surface, broader market.
**v4.6 scope.**

## Bet 7 — Token-rate streaming cognometry

`generate_safe` halts generation on rising risk. Extend to emit a
*live cognitive telemetry stream* during generation: per-token
confidence, per-token retrieval-vs-reasoning classification, per-token
adversarial drift. Subscribers (logging, UIs, monitoring dashboards)
get realtime cognitive state without polling. This is Observable but
for LLM cognition. **v4.7 scope; significant engineering.**

## Bet 8 — Memory-vs-synthesis classifier

Novel hallucination class: the model *memorized* vs *synthesized* its
output. Memorized outputs (leaked training data, verbatim-copied text)
vs synthesized outputs have different residual-stream signatures. Add
a binary classifier: given a generation, did it come from memory or
synthesis? Applications: copyright / IP attribution, training-data
leakage detection, retrieval-augmentation sanity check. This is
**cognometric provenance** — no one is building it. **v5.1 scope.**

## Bet 9 — The cognometry protocol

A standardized HTTPS protocol for cross-model cognitive-state exchange
(draft exists at `docs/cognet-protocol-v0.md`). Every compliant model
exposes a canonical `/cognet` endpoint returning normalized vital
signs. Regulators address cognitive state in common units; safety
libraries query once for any model. This is the **OpenTelemetry of
LLM cognition**. Ship the reference implementation in `styxx.cognet`
and pitch it to the big labs. **v5.2 scope.**

## Bet 10 — Cognometry for agents, not just LLMs

All current signals operate on single-turn generation. An agent is a
SEQUENCE of cognitive states. Build trajectory-level cognometry:
track an agent's cognitive-temperature over a 100-step task, detect
drift, detect reasoning failure modes that are invisible in any single
step. **Agents hallucinate differently than LLMs** — the pattern is
cumulative error, not single-call fabrication. This is the second
cognometric frontier after the one we just opened. **v5.3 scope.**

---

## Prioritization (read: what actually moves the needle)

| # | Name | Research risk | Dev cost | Headline potential |
|---|---|---|---|---|
| 1 | Adversarial drift detection | medium | 1-2 weeks | **HIGH** |
| 2 | Meta-cognometric guardians | high | 2-3 weeks | **HIGH** |
| 3 | Online calibration | low | 1 week | medium |
| 4 | Cognitive fingerprint arbitrage | medium | 2 weeks | **HIGH** |
| 5 | Cognometry index | low | ongoing | medium |
| 6 | Cognitive temperature as default | low | 3 days | medium |
| 7 | Token-rate streaming | low | 2 weeks | medium |
| 8 | Memory-vs-synthesis | high | 3 weeks | **HIGH** |
| 9 | Cognometry protocol | low | 2 weeks | **HIGH** (if adopted) |
| 10 | Agent-level cognometry | high | 4 weeks | **HIGHEST** |

**If we had to pick three to ship in Q2 2026:**

1. **Bet 1 (adversarial drift)** — under-served, defensible, ships fast.
2. **Bet 3 (online calibration)** — product moat, easy.
3. **Bet 10 (agent-level)** — where the field is going; we should lead
   it rather than follow.

---

## What this document is NOT

Not a roadmap. Not a commitment. A map of high-leverage moves so that
the next time someone asks "what's next for cognometry?" there is a
document answer, not an ad-hoc one. Disagreement, addition, subtraction
welcome — file an issue on the repo.
