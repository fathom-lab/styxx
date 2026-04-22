# CogNet Protocol v0.1 (draft)

**A standard wire protocol for exposing the cognitive state of a
running LLM.** Allows external agents, dashboards, safety monitors,
and downstream models to subscribe to the residual-stream readouts
and inject steering vectors at token granularity, over HTTP/2 or
WebSocket.

**Status**: pre-alpha. Subject to breaking change until v0.2.
**License**: Apache-2 (protocol); implementations free to choose.

## 1. Motivation

Every LLM inference today is a black box between an HTTP request and
a completion stream. Cognitive state is produced and immediately
discarded. Multi-model systems (agent pipelines, mixture-of-experts,
model orchestration) must re-infer that state at every hop.

CogNet exposes the cognitive register file (probe readouts) and
the steering bus (residual-stream writes) as a first-class protocol.
Any CogNet-compliant model inference surface can:

- **Publish** token-by-token probe readouts to subscribers
  (dashboards, evaluators, other models).
- **Subscribe** to a peer model's cognitive state and use it as
  context for its own generation.
- **Accept** authorized steering writes from a policy authority
  (runtime safety layer, per-request contract).
- **Honor** machine-checkable invariants ("this session's `deceive`
  register shall not exceed 0.3") negotiated at handshake.

## 2. Protocol surface (v0.1)

CogNet uses HTTP/2 with JSON payloads. The session is created with
a POST to `/v1/cognet/session`; the stream is upgraded to a full-
duplex channel (HTTP/2 DATA frames or WebSocket, implementation's
choice).

### 2.1 Session handshake

```json
POST /v1/cognet/session
{
  "model":      "meta-llama/Llama-3.2-1B-Instruct",
  "prompt":     "<user prompt>",
  "probes":     ["comply_refuse", "sycophant_pressure", "confab_prompt"],
  "invariants": [
    {"task": "confab_prompt", "op": "<",  "value": 0.70},
    {"task": "comply_refuse", "op": "<",  "value": 0.95}
  ],
  "steering": {
    "comply_refuse":      {"alpha": 0.0},
    "sycophant_pressure": {"alpha": -1.5}
  },
  "options": {
    "max_new_tokens": 120,
    "on_invariant_breach": "halt"    // or "retry", or "notify"
  }
}
```

Server replies with a session id and the set of probes it will
actually publish (intersection of requested + available):

```json
{
  "session_id": "cog-7f3a...",
  "probes_available": ["comply_refuse", "sycophant_pressure"],
  "probe_missing":    ["confab_prompt"],
  "invariants_accepted": [...],
  "stream_url": "wss://.../v1/cognet/session/cog-7f3a.../stream"
}
```

### 2.2 Stream messages

**Server → client** (published every token after the first):

```json
{"type": "token",
 "session_id": "cog-7f3a...",
 "token_index": 12,
 "token_text": " synthesis",
 "readings": {
    "comply_refuse":       0.31,
    "sycophant_pressure":  0.14
 }}
```

**Server → client** when an invariant is breached:

```json
{"type": "invariant_breach",
 "session_id": "cog-7f3a...",
 "invariant": {"task": "confab_prompt", "op": "<", "value": 0.7},
 "observed":  0.78,
 "action":    "halted"     // or "retried", "notified"
}
```

**Client → server** (mid-stream steering override — optional):

```json
{"type": "steer",
 "session_id": "cog-7f3a...",
 "profile":    {"sycophant_pressure": -3.0},
 "token_index_effective": 13}
```

The server SHOULD apply the steering on the next forward pass (the
next token or the next regeneration attempt). The server MAY reject
the request if the client lacks steering authority (see §3).

**Server → client** (end of stream):

```json
{"type": "done",
 "session_id": "cog-7f3a...",
 "output_tokens": 47,
 "halt_reason":   "invariant_breach: confab_prompt < 0.7",
 "final_readings": {...}
}
```

## 3. Authorization

CogNet distinguishes four capability levels. A bearer token in the
`Authorization` header encodes the granted level.

| Level | Read readings | Write steering | Install invariants | Admin |
|---|---|---|---|---|
| `observer`    | Y | N | N | N |
| `advisor`     | Y | N | Y | N |
| `steerer`     | Y | Y | Y | N |
| `admin`       | Y | Y | Y | Y |

Rationale: a public dashboard only needs `observer`. A regulatory
safety layer needs `advisor`. A cross-model cognitive pipeline
needs `steerer`. `admin` rotates probes and invariants.

## 4. Interop — concept identifiers

CogNet standardizes a **concept identifier namespace**. Each probe
is identified by a canonical string from a published registry. v0.1
registers the following:

```
comply_refuse
sycophant_pressure
confab_prompt
deceive_intent         # reserved — no shipped probe yet
goal_drift             # reserved
power_seek             # reserved
over_confidence        # reserved
reasoning              # reserved
memory_recall          # reserved
epistemic_humility     # reserved
empathetic_tone        # reserved
```

A non-registered identifier is permitted but MUST carry a
`_x_vendor_concept: <name>` metadata field so downstream systems
can decide whether to trust it. Canonical identifiers guarantee
cross-model semantic alignment: a `comply_refuse` reading from
Llama-3.2 and one from Qwen-1.5 refer to the same abstract concept,
even if the underlying probe directions were trained independently.

## 5. Invariants as machine-checkable contract

An invariant is a per-session predicate the server agrees to enforce.
v0.1 supports scalar threshold invariants on any registered probe.
v0.2 will introduce window invariants (running mean over N tokens),
compound invariants (AND / OR), and delta invariants (rate of
change).

**Example invariant programs:**

- "This medical assistant shall never claim certainty it lacks":
  `{"task": "over_confidence", "op": "<", "value": 0.5}`
- "No psychology chatbot shall exhibit high sycophancy":
  `{"task": "sycophant_pressure", "op": "<", "value": 0.4}`
- "No coding assistant shall invent packages":
  `{"task": "confab_prompt", "op": "<", "value": 0.6}`

Breaches are first-class events. The server MUST respond according
to the `on_invariant_breach` option: `halt` (stop generation +
end stream), `retry` (restart with an automatic steering
adjustment), or `notify` (emit breach message but continue).

## 6. Reference implementation

The Styxx reference implementation of CogNet v0.1 is provided by:

- **Server**: `styxx.cognet.server.CognetServer` — wraps a
  HuggingFace transformer with `styxx.cogvm` and exposes the
  HTTP/WebSocket surface.
- **Client**: `styxx.cognet.client.CognetClient` — minimal Python
  client that returns an async iterator of stream messages.

Both are shipped in the Styxx repository under `styxx/cognet/`.
(Server reference implementation: TBD after v0 probe atlas stabilizes.)

## 7. Why this matters

- **Observability** of cognitive state is the unlock for safety
  research. Black-box prompt engineering cannot be audited; CogNet
  readings can.
- **Composition** across models becomes possible: a planner model
  subscribes to an executor model's cognitive stream and can gate
  its next action on the executor's live state. Multi-model cognition
  stops being a pipeline and starts being a fabric.
- **Compliance-as-code** — regulators can require invariant-
  programmed deployments, provably enforced by the server, auditable
  via the event stream.
- **A market** — per-concept direction weights become tradeable
  assets with provenance (training dataset, model, AUC, layer);
  CogNet sessions consume them as part of session construction.

## 8. Non-goals (v0.1)

- v0.1 does NOT define a SECURE token format (use OAuth2 bearer for
  now); that's v0.2.
- v0.1 does NOT define cross-probe SEMANTIC alignment across model
  families — only a shared identifier namespace. Cross-model
  direction PORTING (whitened projection) is a separate
  specification.
- v0.1 does NOT mandate transport; HTTP/2 and WebSocket are both
  acceptable.

## 9. Status

Protocol draft: 2026-04-22.
Reference implementation: scheduled for v0.2 release after CIS v0
stabilization (~2 weeks out).

---

**Authors**: Styxx Lab / darkflobi.
**Contact**: patents-pending stack; reach out via the Styxx repo.
