# styxx Protocols

Formal, transport-agnostic specifications for agent-to-agent
interoperability. Implementable without importing styxx.

## Index

- [**handoff-v1.md**](handoff-v1.md) — `styxx-handoff/1`: the
  agent-to-agent handoff envelope. Lets a sender propagate its last
  cognitive state alongside the task it's delegating.

- [**cognitive-vocabulary.md**](cognitive-vocabulary.md) — the shared
  6-class vocabulary (`reasoning`, `retrieval`, `refusal`, `creative`,
  `adversarial`, `hallucination`) used by the handoff protocol and any
  downstream A2A tooling.

## Status

| Protocol | Version | Status |
|---|---|---|
| `styxx-handoff` | 1 | Stable |

## Non-styxx Implementers

You do **not** need to `pip install styxx` to participate. Each spec
includes a "Minimum Viable Implementation" section showing the ~40
lines required to emit and validate envelopes against the vocabulary.

## Reference Code

The canonical Python implementation lives in
[`styxx/handoff.py`](../../styxx/handoff.py) (class
`ProtocolEnvelope`). A runnable two-agent example is in
[`examples/advanced/handoff_two_agents.py`](../../examples/advanced/handoff_two_agents.py).
