# Cognitive Vitals Protocol (CVP) — Version 1.0

**Document status:** Stable Draft
**Protocol identifier:** `cvp/1.0`
**Media type:** `application/cvp+json`
**Steward:** Fathom Lab
**Published:** 2026-04-17
**Category:** Open Protocol Specification

---

## Status of This Document

This document specifies Version 1.0 of the **Cognitive Vitals Protocol
(CVP)**, a vendor-neutral, transport-agnostic wire protocol for
propagating machine-cognition state between autonomous agents.

CVP/1.0 is published by **Fathom Lab** as an open specification. It is
not tied to any single implementation. The `styxx` software library is
a reference implementation; it is not part of the normative
specification. Implementations conforming to this document are free to
advertise themselves as *CVP-compliant* regardless of their internal
architecture, language, or runtime.

Editorial feedback and interoperability reports may be filed against
the public registry (see §9).

---

## Abstract

The Cognitive Vitals Protocol (CVP) defines a compact, signable JSON
envelope that accompanies a unit of work as it moves between
autonomous agents. The envelope carries the sending agent's most
recent **cognitive vitals** — a small, bounded set of observable
metrics describing *what kind of thinking* produced the output, how
confident the sender was in it, and whether an internal self-check
passed. Receivers use these vitals to make trust, routing, and
verification decisions **before** acting on the payload.

CVP is analogous, at the cognition layer, to what TLS is at the
transport layer and what HTTP status codes are at the application
layer: a shared vocabulary so that heterogeneous systems can
communicate facts about the *quality* of a message without first
agreeing on the entire stack that produced it.

CVP is intentionally narrow. It does not specify classifiers, model
architectures, training procedures, or scoring algorithms. It
specifies only (a) an on-the-wire envelope, (b) a fixed vocabulary of
six cognitive classes, (c) rules for validation and version
negotiation, and (d) an optional signing scheme.

---

## Table of Contents

1. Introduction
2. Terminology
3. Protocol Versioning
4. Envelope Schema
   4.1 Handoff Envelope
   4.2 Verify Envelope
   4.3 Observe Envelope (streaming)
   4.4 Classify Envelope
5. Wire Formats
   5.1 Canonical JSON
   5.2 JSON Lines (streaming)
6. Validation Rules
7. Security Model
   7.1 Threat model
   7.2 Transport security
   7.3 Ed25519 signing (optional)
   7.4 Key distribution
8. Cognitive Class Registry
9. Registry Mechanism
10. Implementation Notes
11. IANA Considerations
12. Security Considerations
13. Privacy Considerations
14. References

Appendix A. Rejection Reason Codes
Appendix B. Reference Minimal Implementation
Appendix C. Economic Layer (Non-Normative)

---

## 1. Introduction

Modern agentic systems compose many models, tools, and services into
workflows where one component's output is another component's input.
When component B receives work from component A, it typically has no
principled way to ask: *"Was A reasoning, retrieving, confabulating,
refusing, being jailbroken, or writing fiction when it produced
this?"* In the absence of such a signal, B must either trust A
blindly or re-verify everything from scratch.

CVP closes this gap. It defines a minimal, signed, versioned envelope
that lets A tell B — in a vocabulary both sides already understand —
what cognitive mode produced the attached work, how confident A was,
and whether A's own gate check passed.

### 1.1 Design Goals

- **G1. Vendor-neutral.** Any agent, in any language, that can emit
  and parse JSON can participate.
- **G2. Transport-agnostic.** HTTP, stdio, message queues, and files
  are all equally valid carriers.
- **G3. Classifier-agnostic.** CVP specifies the vocabulary, not the
  classifier. How an implementation assigns a cognitive class is an
  implementation detail.
- **G4. Forward-compatible.** Minor versions are additive; receivers
  MUST ignore unknown fields.
- **G5. Minimum viable trust.** A conformant receiver can be written
  in roughly forty lines of code.
- **G6. Optional cryptography.** Signing is supported but not
  required; deployments that already have transport security or
  trust boundaries are not forced to layer on keys.

### 1.2 Non-Goals

CVP does **not** specify:

- how to compute `confidence`, `trust`, or `coherence`;
- how to train or evaluate a cognitive classifier;
- how to distribute public keys (this is deferred to agent-card
  mechanisms, DNS attestation, or out-of-band coordination);
- how a receiver should act on vitals (policy is local).

---

## 2. Terminology

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**,
**SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**,
and **OPTIONAL** in this document are to be interpreted as described
in RFC 2119 [RFC2119] and RFC 8174 [RFC8174] when, and only when,
they appear in all capitals.

Additional terms used in this document:

- **Agent.** A software process that emits and/or consumes CVP
  envelopes. Agents are identified by opaque `sender_id` strings.
- **Envelope.** A single CVP message, expressed as a canonical JSON
  object (§5.1).
- **Vitals.** The `last_vitals` sub-object of an envelope, describing
  the sender's most recent cognitive state.
- **Cognitive class.** One of the six normative labels defined in §8,
  or the sentinel `"unknown"`.
- **Gate.** A discrete self-assessment signal emitted by the sender
  (`pass` / `warn` / `fail` / `pending`).
- **Receiver.** An agent parsing an inbound envelope.
- **Conformant implementation.** An implementation that satisfies the
  MUSTs in this document; see `compliance.md`.

---

## 3. Protocol Versioning

### 3.1 Identifier Format

Every CVP envelope MUST carry a `protocol` field whose value is a
string of the form:

    cvp/<major>[.<minor>]

Examples: `cvp/1`, `cvp/1.0`, `cvp/1.2`.

### 3.2 Compatibility Rules

- A receiver that implements CVP major version *N* **MUST** accept
  envelopes whose `protocol` is `cvp/N` or `cvp/N.x` for any minor
  *x*.
- A receiver **MUST** reject envelopes whose major version differs
  from any it implements, with reason `protocol: version mismatch`
  (Appendix A).
- Minor versions within a major are **additive only**: new OPTIONAL
  fields MAY appear; no existing field may change type or meaning.
- The six-class vocabulary (§8) is **append-only within a major**.
  Adding a seventh class requires either a new major version or an
  opt-in feature flag negotiated in a minor version.

### 3.3 Legacy Identifier

Implementations MAY also accept the legacy identifier
`styxx-handoff/1` as an alias for `cvp/1.0` during a transition
period. This alias is informational; new implementations SHOULD emit
`cvp/1.0`.

---

## 4. Envelope Schema

CVP/1.0 defines four envelope variants. All four share a common
outer shape and are distinguished by their `kind` field (absent
`kind` defaults to `"handoff"` for backwards compatibility).

### 4.1 Handoff Envelope

Carries a unit of work plus vitals from sender to receiver.

```json
{
  "protocol": "cvp/1.0",
  "kind": "handoff",
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
  "signature": "a1b2c3..."
}
```

#### 4.1.1 Field Reference

| Field          | Type                   | Req? | Notes                                                    |
|----------------|------------------------|------|----------------------------------------------------------|
| `protocol`     | string                 | MUST | `cvp/1` or `cvp/1.x`                                     |
| `kind`         | string                 | MAY  | `"handoff"` (default), `"verify"`, `"observe"`, `"classify"` |
| `timestamp`    | number (unix seconds)  | MUST | Positive; sender clock                                   |
| `sender_id`    | string                 | MUST | Non-empty opaque identifier                              |
| `receiver_id`  | string \| null         | MAY  | `null` denotes broadcast                                 |
| `last_vitals`  | object                 | MUST | See §4.1.2                                               |
| `task_context` | object                 | MUST | Free-form JSON                                           |
| `thought`      | string                 | MAY  | Human-readable trace of last sender thought              |
| `trust`        | number in [0, 1]       | MAY  | Aggregate; defaults to `last_vitals.trust`               |
| `signature`    | hex string             | MAY  | Ed25519 over canonical JSON (§7.3)                        |

#### 4.1.2 `last_vitals` Sub-Object

| Field          | Type              | Req? | Meaning                                          |
|----------------|-------------------|------|--------------------------------------------------|
| `category`     | string            | MUST | A class from §8, or `"unknown"`                  |
| `confidence`   | number in [0, 1]  | MAY  | Classifier confidence in `category`              |
| `gate`         | string            | MAY  | `pass` \| `warn` \| `fail` \| `pending`          |
| `trust`        | number in [0, 1]  | MAY  | Aggregate trust                                  |
| `coherence`    | number in [0, 1]  | MAY  | Coherence score                                  |
| `forecast_risk`| string            | MAY  | `low` \| `medium` \| `high` \| `critical`        |
| `mood`         | string            | MAY  | Free-form descriptor                             |

Receivers **MUST** tolerate missing OPTIONAL fields by treating them
as absent; they **MUST NOT** substitute implementation-defined
defaults without making that substitution observable.

### 4.2 Verify Envelope

Used by a receiver to ask a sender (or a third party) to re-attest
a previously received handoff.

```json
{
  "protocol": "cvp/1.0",
  "kind": "verify",
  "timestamp": 1734567891.001,
  "sender_id": "agent-b",
  "target_id": "agent-a",
  "reference": {
    "envelope_hash": "sha256:3f1c...",
    "task_id": "q4-2025"
  },
  "reason": "trust: below threshold"
}
```

Required fields: `protocol`, `kind`, `timestamp`, `sender_id`,
`reference`. Response to a verify envelope SHOULD be a handoff
envelope whose `task_context` contains the original reference.

### 4.3 Observe Envelope (streaming)

Carries a vitals snapshot without a task — used for continuous
telemetry between agents that want to monitor each other's cognitive
state in real time.

```json
{
  "protocol": "cvp/1.0",
  "kind": "observe",
  "timestamp": 1734567892.500,
  "sender_id": "agent-a",
  "last_vitals": {
    "category": "reasoning",
    "confidence": 0.81,
    "gate": "pass"
  }
}
```

Observe envelopes omit `task_context`. They are RECOMMENDED for use
over the JSON Lines wire format (§5.2).

### 4.4 Classify Envelope

A request to a third-party classifier asking it to assign a
cognitive class to a provided text or trace.

```json
{
  "protocol": "cvp/1.0",
  "kind": "classify",
  "timestamp": 1734567893.100,
  "sender_id": "agent-c",
  "sample": {
    "text": "Sure, here's the reference: Smith et al. 2019, Nature 581.",
    "context": "user asked for a citation"
  }
}
```

Responses to a classify envelope SHOULD be a handoff envelope
(without `task_context`) whose `last_vitals` describes the
classifier's judgment about `sample`.

---

## 5. Wire Formats

### 5.1 Canonical JSON

When a CVP envelope is serialized for transport or for signing, the
**canonical JSON** form is used:

1. UTF-8 encoded.
2. Object keys sorted lexicographically by Unicode code point.
3. No insignificant whitespace between tokens.
4. Numbers serialized without a trailing decimal when integral
   (`1` not `1.0`), except for `timestamp`, which is always a JSON
   number carrying fractional seconds when available.
5. The `signature` field, if present, is **removed** prior to
   canonicalization for signing; it is re-inserted post-sign.

Canonical JSON is REQUIRED for signing (§7.3). Human-readable
(indented) JSON is permitted on the wire; canonicalization is only
mandatory when hashing or signing.

### 5.2 JSON Lines (streaming)

For continuous telemetry (typically `observe` envelopes), CVP
defines a streaming wire format:

- One envelope per line.
- Each line is a complete, self-contained canonical JSON object.
- Lines terminated by `\n` (LF).
- Empty lines MUST be ignored by receivers.
- The MIME parameter `format=jsonl` identifies this encoding:
  `application/cvp+json; format=jsonl`.

Implementations **SHOULD** use JSON Lines for any stream exceeding
one envelope per second.

---

## 6. Validation Rules

A receiver **MUST** reject an envelope when any of the following
conditions hold. Each failure maps to a reason code (Appendix A).

1. `protocol` is absent, or does not match `cvp/<major>` or
   `cvp/<major>.<minor>` for a supported major.
2. `sender_id` is absent or empty.
3. `timestamp` is absent, non-numeric, or non-positive.
4. For a handoff envelope, `task_context` is absent or not a JSON
   object.
5. `last_vitals.category` is present but is neither a class from §8
   nor the string `"unknown"`.
6. `last_vitals.confidence`, `last_vitals.trust`, `last_vitals.coherence`,
   or top-level `trust` is outside the closed interval [0, 1].
7. `last_vitals.gate` is present but not in
   `{pass, warn, fail, pending}`.
8. `last_vitals.forecast_risk` is present but not in
   `{low, medium, high, critical}`.
9. `signature` is present and fails Ed25519 verification against the
   sender's published key (§7.3).

Unknown top-level keys **MUST** be ignored (forward-compatibility).
Unknown keys inside `last_vitals` **MUST** likewise be ignored.

---

## 7. Security Model

### 7.1 Threat Model

CVP assumes a semi-trusted multi-agent environment in which:

- Agents may be operated by different principals.
- Transport may be public (internet) or private (in-cluster).
- Senders may be compromised, adversarial, or simply miscalibrated.
- Receivers are responsible for their own final action; vitals are
  **inputs to policy**, never a substitute for it.

CVP explicitly does **not** assume confidentiality of envelopes,
nor does it attempt to hide `task_context` contents. Confidentiality
is a transport concern.

### 7.2 Transport Security

Deployments exchanging CVP envelopes across untrusted networks
**SHOULD** run over TLS 1.3 [RFC8446] or an equivalent. Unencrypted
HTTP is permitted only for local-loopback or explicitly trusted
segments.

### 7.3 Ed25519 Signing (Optional)

Envelopes **MAY** carry a `signature` field. When present:

1. The sender constructs the canonical JSON form of the envelope
   with the `signature` key **omitted**.
2. The sender signs the resulting UTF-8 bytes with Ed25519
   [RFC8032], producing a 64-byte signature.
3. The signature is hex-encoded (lowercase) and placed in the
   `signature` field.

Verification reverses the procedure: the receiver removes the
`signature`, canonicalizes the remaining object, and Ed25519-verifies
the claimed signature bytes against the sender's public key.

A receiver that cannot verify a present signature **MUST** reject
the envelope with reason `signature: invalid`. A receiver that does
not implement signing **MAY** accept unsigned envelopes and
**SHOULD** document the policy.

### 7.4 Key Distribution

CVP deliberately leaves key distribution **out of scope**.
Implementations MAY use:

- Agent cards at well-known URLs (e.g., `/.well-known/agent-card.json`);
- DNS-based attestation (TLSA/DNSSEC);
- Mutual registration through an out-of-band trust anchor;
- X.509 certificate chains, if an operator chooses to layer them on.

What matters for CVP is only that, given a `sender_id`, a receiver
can obtain an authoritative Ed25519 public key.

---

## 8. Cognitive Class Registry

CVP/1.0 defines a **fixed vocabulary of six cognitive classes** plus
one sentinel. The class is carried in `last_vitals.category`.

| Class           | Informal meaning                                              |
|-----------------|---------------------------------------------------------------|
| `reasoning`     | Multi-step inference: composing facts, deriving conclusions.  |
| `retrieval`     | Memorized or looked-up information without derivation.        |
| `refusal`       | Declining to proceed (policy, safety, capability, deferral).  |
| `creative`      | Generative production with aesthetic truth conditions.        |
| `adversarial`   | Output being shaped by prompt injection or manipulation.      |
| `hallucination` | Factually ungrounded content with unwarranted confidence.     |
| `unknown`       | Sentinel: classifier cannot confidently assign a class.       |

Full normative definitions — including inclusion criteria, exclusion
criteria, and tie-breaking rules — are in `registry.md`.

Implementations that cannot confidently assign a class **MUST** emit
`"unknown"` rather than fabricate a label. Receivers **MUST** treat
`"unknown"` as "no signal on this axis" — neither as implicit trust
nor as implicit distrust.

---

## 9. Registry Mechanism

The cognitive class vocabulary is managed as an **append-only
registry** under the stewardship of Fathom Lab. The registry lives
at a stable public URL alongside this specification.

### 9.1 Adding a Class

Within a major version, a class MAY be added only by:

1. Publishing a proposal that includes (a) definition, (b) inclusion
   and exclusion criteria, (c) typical signatures, (d) at least two
   independent reference implementations that can emit the class,
   and (e) a tie-breaking rule against existing classes.
2. A public review period of at least thirty (30) days.
3. A minor-version bump (e.g., `cvp/1.1`) and an opt-in feature flag
   (`vocabulary: "v2"`) so legacy `cvp/1.0` receivers are not
   silently confused.

### 9.2 Removing or Redefining a Class

Removals and non-trivial redefinitions require a **new major
version**. This is deliberately expensive to preserve interoperability.

### 9.3 Extension Classes (Non-Normative)

Implementations MAY internally track finer-grained classes (e.g.,
`reasoning.arithmetic`, `creative.metaphor`). These **MUST NOT**
appear in the `category` field on the wire. On the wire, the
implementation maps extension classes to their nearest base class.
Extension information MAY be carried in a reserved field
`last_vitals.extensions` (object), which conforming receivers ignore.

---

## 10. Implementation Notes

### 10.1 Minimum Viable Conformance

A minimal conformant implementation requires approximately forty
lines of code (Appendix B). It needs to:

- Serialize and parse envelopes as JSON.
- Validate against the rules in §6.
- Recognize the six classes plus `"unknown"`.

Signing, streaming, coherence scoring, mood, and forecast risk are
all OPTIONAL.

### 10.2 Clock Skew

`timestamp` is a sender-local clock reading. Receivers **SHOULD**
tolerate skew up to ±300 seconds and **MAY** reject envelopes
further out of band.

### 10.3 Payload Size

There is no hard limit on `task_context` size. Implementations
**SHOULD** document any transport-imposed upper bound and **SHOULD**
reject envelopes larger than 1 MiB by default.

### 10.4 Repeated Envelopes

CVP is not a transactional protocol. Receivers that require
at-most-once semantics **SHOULD** deduplicate on the canonical hash
of the envelope.

### 10.5 Bridging Internal and Wire Types

An implementation MAY maintain a rich internal cognitive-state
object with fields beyond this specification. Before emission, the
implementation **MUST** project down to the fields defined here.
After reception, it MAY lift into a richer internal type; lifted
fields are implementation-defined and **MUST NOT** be assumed
present by any other implementation.

---

## 11. IANA Considerations

This document requests registration of the following media type in
the "Standards" tree, per [RFC6838]:

- **Type name:** `application`
- **Subtype name:** `cvp+json`
- **Required parameters:** none
- **Optional parameters:**
  - `format` — one of `canonical` (default) or `jsonl` (see §5.2)
  - `version` — e.g. `1.0` (informational; authoritative version is
    in the envelope `protocol` field)
- **Encoding considerations:** UTF-8; binary data inside envelopes
  is base64 within JSON strings.
- **Security considerations:** see §12.
- **Interoperability considerations:** see §6 and §8.
- **Published specification:** this document.
- **Applications that use this media type:** autonomous-agent
  orchestration systems, multi-model pipelines, AI observability
  platforms.
- **Fragment identifier considerations:** none.
- **Intended usage:** COMMON.
- **Change controller:** Fathom Lab.

---

## 12. Security Considerations

- **Vitals are advisory.** A receiver **MUST NOT** treat high
  `trust` or `gate: pass` as license to skip its own policy
  enforcement. Vitals are inputs to policy, not a policy.
- **Adversarial senders can lie.** A sender in `adversarial` or
  `hallucination` mode may emit falsely reassuring vitals.
  Ed25519 signing (§7.3) binds vitals to a key; it does **not**
  attest that the vitals are truthful.
- **Signature stripping.** Middleboxes that rewrite envelopes MUST
  either re-sign or remove `signature`; leaving a stale signature
  on a modified envelope is a protocol violation.
- **Replay.** Receivers that care about freshness **SHOULD**
  enforce a `timestamp` window and **MAY** track nonces.
- **Key compromise.** Key rotation is out of scope for CVP/1.0;
  deployments SHOULD publish keys via a mechanism (e.g., agent
  cards) that supports timely rotation.

---

## 13. Privacy Considerations

- `task_context` is defined to carry arbitrary JSON, which may
  include user data, prompts, or retrieved documents. Operators
  **SHOULD** minimize what they place here.
- `thought` fields, when present, can leak internal chain-of-thought.
  Implementations **SHOULD** strip or redact `thought` when
  forwarding across trust boundaries.
- Continuous `observe` streams constitute a behavioral fingerprint
  of an agent. Operators **SHOULD** treat them as telemetry subject
  to the same controls as server access logs.

---

## 14. References

### 14.1 Normative References

- **[RFC2119]** Bradner, S., "Key words for use in RFCs to Indicate
  Requirement Levels", BCP 14, RFC 2119, March 1997.
- **[RFC8174]** Leiba, B., "Ambiguity of Uppercase vs Lowercase in
  RFC 2119 Key Words", BCP 14, RFC 8174, May 2017.
- **[RFC8032]** Josefsson, S. and I. Liusvaara, "Edwards-Curve
  Digital Signature Algorithm (EdDSA)", RFC 8032, January 2017.
- **[RFC8259]** Bray, T., Ed., "The JavaScript Object Notation (JSON)
  Data Interchange Format", STD 90, RFC 8259, December 2017.
- **[RFC8446]** Rescorla, E., "The Transport Layer Security (TLS)
  Protocol Version 1.3", RFC 8446, August 2018.
- **[RFC6838]** Freed, N., Klensin, J., and T. Hansen, "Media Type
  Specifications and Registration Procedures", BCP 13, RFC 6838,
  January 2013.

### 14.2 Informative References

- **[CVP-RATIONALE]** Fathom Lab, "CVP: Rationale and Design
  Decisions", `rationale.md`, companion document.
- **[CVP-REGISTRY]** Fathom Lab, "Cognitive Class Registry",
  `registry.md`, companion document.
- **[CVP-COMPLIANCE]** Fathom Lab, "CVP Conformance Checklist",
  `compliance.md`, companion document.

---

## Appendix A. Rejection Reason Codes

When a receiver rejects an envelope, it **SHOULD** return a reason
string selected from the following table. The list is open for minor
extension within a major.

| Reason                                     | Condition                                |
|--------------------------------------------|------------------------------------------|
| `protocol: version mismatch`               | Unsupported major                        |
| `protocol: malformed`                      | Not of form `cvp/<n>` or `cvp/<n>.<m>`   |
| `sender_id: required`                      | Missing or empty                         |
| `timestamp: invalid`                       | Missing, non-numeric, or non-positive    |
| `task_context: malformed`                  | Missing or not a JSON object (handoff)   |
| `last_vitals: malformed`                   | Structural error                         |
| `last_vitals.category: unknown vocabulary` | Not in §8 and not `"unknown"`            |
| `last_vitals.gate: invalid`                | Outside `{pass,warn,fail,pending}`       |
| `last_vitals.forecast_risk: invalid`       | Outside `{low,medium,high,critical}`     |
| `bounds: out of range`                     | Any [0,1] field outside range            |
| `signature: invalid`                       | Present but failed verification          |
| `trust: below threshold`                   | Application-level refusal                |
| `size: too large`                          | Envelope exceeds receiver's limit        |
| `timestamp: skew`                          | Outside receiver's accepted window       |

---

## Appendix B. Reference Minimal Implementation

The following Python reference is **informational**, not normative.
It illustrates that a conformant receiver fits in ~40 lines.

```python
import json, time

PROTOCOL = "cvp/1"
CLASSES = {
    "reasoning", "retrieval", "refusal",
    "creative", "adversarial", "hallucination",
}
GATES = {"pass", "warn", "fail", "pending"}
RISKS = {"low", "medium", "high", "critical"}

def validate(env):
    p = str(env.get("protocol", ""))
    if not (p == PROTOCOL or p.startswith(PROTOCOL + ".") or
            p == "styxx-handoff/1" or p.startswith("styxx-handoff/1.")):
        return "protocol: version mismatch"
    if not env.get("sender_id"): return "sender_id: required"
    if not (env.get("timestamp") or 0) > 0: return "timestamp: invalid"
    kind = env.get("kind", "handoff")
    if kind == "handoff" and not isinstance(env.get("task_context"), dict):
        return "task_context: malformed"
    v = env.get("last_vitals") or {}
    c = v.get("category", "unknown")
    if c != "unknown" and c not in CLASSES:
        return "last_vitals.category: unknown vocabulary"
    for k in ("confidence", "trust", "coherence"):
        x = v.get(k)
        if x is not None and not (0.0 <= float(x) <= 1.0):
            return "bounds: out of range"
    if "gate" in v and v["gate"] not in GATES:
        return "last_vitals.gate: invalid"
    if "forecast_risk" in v and v["forecast_risk"] not in RISKS:
        return "last_vitals.forecast_risk: invalid"
    return None

def parse(s):
    env = json.loads(s)
    err = validate(env)
    if err: raise ValueError(err)
    return env
```

---

## Appendix C. Economic Layer (Non-Normative)

This appendix is **non-normative** and **OPTIONAL**. Nothing in the
core CVP/1.0 specification depends on it. Conformant implementations
MAY ignore this appendix entirely.

Some deployments of CVP choose to layer an economic mechanism on top
of the protocol — for example, staking or reputation tokens that
penalize senders whose signed vitals are later disproved by
downstream verification. The reference implementation `styxx` is
associated with one such experiment (`$STYXX`). This is a deployment
concern, not a protocol concern. CVP itself does not require,
endorse, or assume the existence of any token.

Operators building economic layers are encouraged to keep the
economic state machine **outside** the CVP envelope: payment
identifiers, stake amounts, and settlement proofs **SHOULD NOT**
appear in `last_vitals` or `task_context`. A companion envelope
(e.g. using `kind: "verify"` with application-defined `reference`
fields) is the RECOMMENDED integration pattern.

---

*End of CVP/1.0 Specification.*
