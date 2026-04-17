# styxx-handoff/1 — Agent-to-Agent Handoff Protocol

**Version:** 1.0
**Status:** Stable
**Wire format:** JSON (UTF-8)
**Transport:** any (HTTP, stdio, queue, file — the protocol is transport-agnostic)

## Table of Contents

1. [Overview](#1-overview)
2. [Envelope Structure](#2-envelope-structure)
3. [Field Reference](#3-field-reference)
4. [Cognitive Vocabulary](#4-cognitive-vocabulary)
5. [Validation Rules](#5-validation-rules)
6. [Version Negotiation](#6-version-negotiation)
7. [Signing (optional)](#7-signing-optional)
8. [Rejection Reasons](#8-rejection-reasons)
9. [Backwards Compatibility Policy](#9-backwards-compatibility-policy)
10. [Minimum Viable Implementation](#10-minimum-viable-implementation)
11. [Reference Example](#11-reference-example)

---

## 1. Overview

`styxx-handoff/1` formalizes *cognitive state propagation* between
autonomous agents. When Agent A hands a task to Agent B, A's
last-known cognitive vitals (what kind of thinking it was doing, how
confident it was, whether its self-check passed) travel with the task
inside a signed JSON envelope.

Agent B can then reason about trust *before* executing:

> "This came from agent `darkflobi` in `reasoning` mode, trust 0.87,
> gate `pass`. Proceed."
>
> "This came from agent `unknown` in `hallucination` mode, trust 0.31,
> gate `fail`. Reject or verify before acting."

The protocol does **not** mandate styxx. Any agent that can emit or
parse this JSON shape is conformant.

## 2. Envelope Structure

```json
{
  "protocol": "styxx-handoff/1",
  "timestamp": 1734567890.123,
  "sender_id": "agent-a",
  "receiver_id": "agent-b",
  "last_vitals": {
    "category": "reasoning",
    "confidence": 0.87,
    "gate": "pass",
    "trust": 0.87,
    "coherence": 0.72,
    "forecast_risk": "low",
    "mood": "steady"
  },
  "thought": "Computed Q4 revenue from line items; cross-checked totals.",
  "task_context": {
    "task": "summarize q4 financials",
    "data": {"report_id": "q4-2025"}
  },
  "trust": 0.87,
  "signature": "a1b2c3…"
}
```

## 3. Field Reference

| Field | Type | Required | Notes |
|---|---|---|---|
| `protocol` | string | ✅ | MUST start with `"styxx-handoff/1"` |
| `timestamp` | number (unix seconds) | ✅ | Positive; sender clock |
| `sender_id` | string | ✅ | Non-empty agent identifier |
| `receiver_id` | string | optional | Intended recipient; `null` = broadcast |
| `last_vitals` | object | ✅ | See below; fields inside are optional |
| `task_context` | object | ✅ | Free-form JSON (task description, payload, etc.) |
| `thought` | string | optional | Sender's last emitted thought (for context) |
| `trust` | number 0..1 | optional | Aggregate trust; defaults to `last_vitals.trust` |
| `signature` | hex string | optional | Ed25519 over canonical JSON (see §7) |

### 3.1 `last_vitals`

| Field | Type | Meaning |
|---|---|---|
| `category` | string | One of the [6 cognitive classes](#4-cognitive-vocabulary), or `"unknown"` |
| `confidence` | number 0..1 | Classifier's confidence in `category` |
| `gate` | string | `"pass"` \| `"warn"` \| `"fail"` \| `"pending"` |
| `trust` | number 0..1 | Aggregate trust score |
| `coherence` | number 0..1 | Optional coherence metric |
| `forecast_risk` | string | `"low"` \| `"medium"` \| `"high"` \| `"critical"` |
| `mood` | string | Free-form descriptor (optional) |

Consumers MUST tolerate missing optional fields by treating them as
absent (not as defaults).

## 4. Cognitive Vocabulary

The `category` field is drawn from a fixed, shared 6-class vocabulary:

- `reasoning`
- `retrieval`
- `refusal`
- `creative`
- `adversarial`
- `hallucination`

Precise definitions are in [`cognitive-vocabulary.md`](cognitive-vocabulary.md).
Any agent — styxx or not — that runs a classifier mapping its output
to these labels is interoperable at the vocabulary layer.

If an implementation cannot determine the class, it MUST emit
`"unknown"` rather than a made-up label.

## 5. Validation Rules

A receiver MUST reject an envelope when any of the following fail:

1. `protocol` does not match `"styxx-handoff/1"` or `"styxx-handoff/1.*"`.
2. `sender_id` is missing or empty.
3. `timestamp` is missing or non-positive.
4. `task_context` is not a JSON object.
5. `last_vitals.category` is not in the shared vocabulary nor `"unknown"`.
6. `last_vitals.confidence` or `last_vitals.trust` is outside `[0, 1]`.
7. `last_vitals.gate` is not one of `pass|warn|fail|pending`.

Unknown top-level keys MUST be ignored (forward-compat).

## 6. Version Negotiation

- `styxx-handoff/1` is the stable major.
- Minor-versioned strings like `styxx-handoff/1.2` are backwards
  compatible: a `/1` receiver MUST accept them and ignore fields it
  doesn't recognize.
- A different major (`styxx-handoff/2`, `foo-handoff/1`, …) MUST be
  rejected with reason `protocol: version mismatch`.

## 7. Signing (optional)

1. Compute canonical JSON: `to_dict()` with `signature` omitted,
   `sort_keys=true`, no spaces, UTF-8 encoded.
2. Sign the UTF-8 bytes with Ed25519 (`cryptography` library or
   equivalent).
3. Hex-encode the 64-byte signature; put it in the `signature` field.

Verification reverses this: drop `signature`, re-serialize canonically,
Ed25519-verify. Key distribution is out of scope (agent cards,
DNS-based attestation, etc.).

## 8. Rejection Reasons

When a receiver rejects an envelope, it SHOULD return a reason from:

| Reason | Meaning |
|---|---|
| `protocol: version mismatch` | Major version not supported |
| `sender_id: required` | Missing / empty |
| `timestamp: invalid` | Missing or non-positive |
| `task_context: malformed` | Not a JSON object |
| `last_vitals: malformed` | Structural errors in vitals |
| `last_vitals.category: unknown vocabulary` | Outside shared 6 classes |
| `signature: invalid` | Signature present but failed verification |
| `trust: below threshold` | Application-level refusal |

## 9. Backwards Compatibility Policy

- **Additive changes only** within `styxx-handoff/1.*`. New optional
  fields may appear; existing ones never change meaning or type.
- Removing or repurposing a field requires a major bump to
  `styxx-handoff/2`.
- The 6-class vocabulary is append-only within major `1`. Adding a
  7th class would require major `2` *or* a minor bump plus a feature
  flag (`vocabulary: "v2"`) that receivers opt into.
- The in-process helpers `styxx.handoff()` / `styxx.receive()` and
  `styxx.handshake.HandoffEnvelope` remain stable. `ProtocolEnvelope`
  is the formal *wire* complement; bridges are provided via
  `styxx.from_handshake_envelope` / `to_handshake_envelope`.

## 10. Minimum Viable Implementation

A non-styxx agent can participate in ~40 lines of code:

```python
import json, time

PROTOCOL = "styxx-handoff/1"
CLASSES = {"reasoning","retrieval","refusal","creative","adversarial","hallucination"}

def build_envelope(sender_id, task, vitals, receiver_id=None):
    return {
        "protocol": PROTOCOL,
        "timestamp": time.time(),
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "last_vitals": vitals,   # dict with category, confidence, gate, trust, ...
        "task_context": {"task": task},
    }

def validate(env):
    if not str(env.get("protocol","")).startswith(PROTOCOL): return "protocol: version mismatch"
    if not env.get("sender_id"): return "sender_id: required"
    if not (env.get("timestamp") or 0) > 0: return "timestamp: invalid"
    if not isinstance(env.get("task_context"), dict): return "task_context: malformed"
    v = env.get("last_vitals") or {}
    c = v.get("category","unknown")
    if c != "unknown" and c not in CLASSES: return "last_vitals.category: unknown vocabulary"
    return None

def parse(s):
    env = json.loads(s)
    err = validate(env)
    if err: raise ValueError(err)
    return env
```

That is the full bar for interoperability. Signing, coherence, mood,
forecast — all optional.

## 11. Reference Example

```python
import styxx
env = styxx.ProtocolEnvelope(
    sender_id="agent-a",
    task_context={"task": "summarize q4"},
    last_vitals=styxx.Vitals(
        category="reasoning", confidence=0.87,
        gate="pass", trust=0.87,
    ),
)
env.validate()           # raises if invalid
wire = env.to_json()     # send this over the wire
got = styxx.ProtocolEnvelope.from_json(wire)
assert got.last_vitals.category == "reasoning"
```
