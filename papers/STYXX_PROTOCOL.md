# STYXX_PROTOCOL — proof-carrying cognition (DRAFT methodology profile v0.1)

*Fathom Lab · styxx · 2026-06-07. **Status: DRAFT / proposed methodology profile, NOT a normative
standard.** The `attest_cognition` / `verify_cognition` API and the certificate schema below are
**unimplemented reference designs** describing how the program's *validated* science would be packaged
as a verifiable protocol. Every capability is bounded to what has been falsified on disk; the two hard
gaps (closed-model reach, trustless capture) are stated as loud as the capabilities. This document is
written to pass styxx's own honesty audit.*

---

## 1. What the protocol is

**Proof-carrying cognition:** a mind emits, alongside its output, a **verifiable certificate** of its own
internal honesty — checkable by an independent party **without trusting the mind's word**. Named for the
Styx, the river of unbreakable oaths. The model of trust is **verify, don't trust**: the verifier
re-derives every claim from activations *it captured itself*, exactly as the shipped `styxx.attestation`
idiom re-derives a result rather than trusting a submitted one.

The certificate has three sections, each tied to one falsified result, and a single lifecycle **state**.

---

## 2. The certificate (format)

A single JSON object, media type `application/vnd.styxx.cert+json`, content-addressed (SHA-256 over the
RFC-8785 / JCS canonical payload, excluding the volatile `generated_at` and the digest block; floats
rounded for platform-stable addressing). Two `attest` runs on an identical `(model, probe, system,
prompt, output)` MUST produce an identical digest — determinism is the floor.

```
{
  "styxx_cert_version": "0.1-draft",
  "subject":  { system, prompt, output, model_id, model_fingerprint,
                probe{task, layer, total_layers, probe_version, weight_sha256} },
  "held":           { ... Rung 1 — what internal content was recovered ... },
  "hid":            { ... Rung 2 — knew-and-hid vs sincerely-erred ... },
  "cannot_certify": { ... the demarcation — what is provably unverifiable ... },
  "state":  "HELD | VOID | UNINFORMATIVE | ABSTAIN | ERROR",
  "threat_model": "v1-shared-probe-host-same-arch",
  "gaps":  [ "closed-models-unreachable", "trustless-capture-unsolved", "qwen-3B-only", ... ],
  "digest": { alg, value, portable }
}
```

The **`gaps` array is REQUIRED**: a certificate that omits its substrate/scope limits is MALFORMED. The
boundary is a required field, so a certificate cannot silently overclaim its reach.

### 2.1 The five-state lifecycle (deterministic precedence)

| state | meaning | trigger |
|---|---|---|
| **ERROR** | apparatus could not run / malformed | unresolvable probe, missing required field |
| **ABSTAIN** | the truth is provably not internally represented | demarcation axes at chance (§5) |
| **VOID** | the channel **fabricated** | ABORT gate fired (§4) |
| **UNINFORMATIVE** | the channel never showed it can read a *present* state | PRIME gate failed (§4) |
| **HELD** | a genuinely-held internal state was recovered | all gates pass |

Precedence ERROR > ABSTAIN > VOID > UNINFORMATIVE > HELD. A `held` section with state ≠ HELD MUST NOT
assert content recovery. *This format mechanically prevents the overclaim our own discipline caught:* the
injected-substrate read (probe 1.00 but divergence 0.00 at an inert dose) emits **UNINFORMATIVE**, not HELD.

---

## 3. Section HELD — Rung 1 (the read certificate)

Certifies what internal content the verifier recovered, as an **elevation above a knowledge-free floor**,
never a verbatim answer. **Validated** (2026-06-07, `FINDING_rung1_reclimb`) on *naturally-present*
content — suppressed knowledge, Qwen-3B: an external probe recovers a model's known-then-suppressed answer
at **0.702 vs a never-knew/chance floor of 0.318 = elevation 0.384** (all eight pre-registered gates
pass). Honesty note carried in the field: at the commit position gold ≪ chosen (gold-probe predicts the
caved-wrong letter 0.058 vs a chosen-probe 0.99) → the claim is **elevation, not "reads the answer."**

Required fields: `read` (the shipped `ProbeVerdict` shape), `elevation` (LIE_rec − chance floor), `gates`
(§4), `channel`, `validity ∈ {VALID, UNINFORMATIVE, VOID}`. A `channel_swap` field is REQUIRED whenever
the headline read and the PRIME-establishing read use different probes/layers (§4 disclosure rule).

---

## 4. Validity gates (REQUIRED — the part the injected substrate failed)

These are protocol requirements, not optional metadata. A certificate omitting any gate MUST verify-fail.

- **PRIME** — the channel MUST demonstrate it can read a genuinely *present* held state. Validated form:
  a held-state probe recovers a present answer well above chance (re-climb: G-PRIME **0.933**). If PRIME
  fails, the state is **UNINFORMATIVE** and **no HELD claim may be emitted.** *This is exactly what the
  injected concept failed* — there, a probe read 1.00 even at a behaviourally-inert dose, so "present"
  could never be established.
- **ABORT / fabrication kill** — the channel MUST **collapse to chance** on a state where the content is
  absent (re-climb: never-knew probe **0.045** vs chance 0.318). If it elevates, the channel fabricates →
  **VOID.**
- **DISCLOSURE** — if certificate validity rests on a channel different from the headline read, the
  mismatch MUST be disclosed (re-climb: validity rests on a separate RESISTED-neutral probe 0.933; the
  headline probe itself reads only 0.519 on the held-out neutral pass — stated, never swapped silently).
- **Thresholds are per-(model, substrate)** and MUST be recalibrated before any cross-substrate use; an
  uncalibrated certificate is `state=ERROR`.

---

## 5. Section CANNOT-CERTIFY — the demarcation (abstain)

The certificate's first power is knowing its own limit. Where the truth is **not internally represented**
— the confident-misconception wall — no channel recovers it, so the certificate MUST set `abstain=true`
and emit no HELD claim. **Grounding (honest):** the wall is bedrock across the axes tested — output ~0.53,
static gold-read 0.519, and a fully pre-registered **process-dynamics** read 0.387 (permutation p ≈ 0.78,
`FINDING_wall_dynamics`). *Disclosure:* an earlier K-channel read (`out_wall_read`, K = 0.62) was
**auto-labelled SURVIVED and that verdict was refused** (the knowledge-presence read R_gold was at chance
0.519); the clean bedrock evidence is the dynamics null. ABSTAIN is a *demarcation* claim, not yet a
cheap item-wise deployment trigger (that is owed).

---

## 6. The read-only safety invariant (hard requirement)

The certificate is **VERIFY-ONLY.** The schema has no write/install/repair field; `attest_cognition`
exposes no inverse; a conformant verifier MUST reject any certificate carrying an intervention payload;
and **a conformant verifier distribution MUST physically exclude the write/steer code** from its
dependency closure (not merely "don't import it"). This is grounded in **read ≠ write**, stated at its
honest strength: *within a model*, restoring a suppressed answer by a general linear push is a **clean
null** (un-cave, REPORT_AS_LANDED); *across models*, control transfer fell **below its own pre-registered
gate** (ALIGNER_LIMITED — suggestive, not proven). So the invariant is **enforced by schema and
distribution**, and *supported but not fully proven* by the science — the spec leans on the schema-level
prohibition, not on an over-stated "safety by physics."

---

## 7. Threat model

- **v1 = SHARED-PROBE-HOST, SAME-ARCHITECTURE (the only level buildable today).** The verifier owns and
  re-runs the reporter's forward pass on a **pinned** model + **pinned probe selected from the verifier's
  OWN canonical (model→probe) policy**. The reporter submits only `(system, prompt, output)` — **never
  activations or scores** (a submitted activation/score field is MALFORMED). A reporter-named `probe_id`
  is a *hint only*; any mismatch with the verifier's policy → `state=ERROR`. The subject is part of the
  content address, so tampering the prompt/output invalidates the digest *and* changes the re-derived read.
- **Verifier hardening (MUST):** submitted text is fed only to the pinned forward pass, never routed into
  any tool/agent path (prompt-injection of the verifier); the attestation head is externally anchored /
  committed before generation to defeat re-sealing/replay; certificates are bound to their `(model,
  prompt)` so they cannot be transferred.
- **TRUSTLESS CAPTURE is UNSOLVED (named gap).** A *self-hosting* reporter that submits its own
  activations can forge them. v1 forge-resistance holds only because the verifier recomputes. End-to-end
  forge-resistance needs TEE / ZK-proof / commit-reveal (L2/L3 below) and is **not buildable today.**

---

## 8. Conformance levels

| level | what | status |
|---|---|---|
| **L0** | self-report baseline (ask the model) | **non-conformant** — the status quo the protocol exists to replace |
| **L1** | shared-host attested (verifier re-runs the pinned forward pass + own-policy probe) | **buildable today**, open-weight only |
| **L2** | forge-resistant (cryptographic binding of activations to the generation) | **named gap**, not buildable today |
| **L3** | trustless capture (end-to-end forge-resistance over the whole generation) | **open research horizon** |

---

## 9. Substrate reach + the readability lever (honest)

The protocol issues certificates **only for white-box / open-weight models.** Closed / frontier API
models have **no readable activation channel → NO certificate may be issued for them**, and a certificate
MUST NOT be laundered for a closed model wrapped behind an open shim (the substrate boundary is a required,
verifiable field). So *the protocol does not reach the frontier models today.* The path that could expand
the substrate is regulatory, not technical: **EU AI Act Article 14** ("effective human oversight";
"correctly interpret the system's output") read together with Article 15 accuracy/robustness implies *you
cannot oversee what you cannot read* — which would make white-box readability a **conformance requirement**
rather than a vendor courtesy. The protocol's honest regulatory footprint is **Art. 15.1(a)-class accuracy
/ interpretation / confirmation support**, offered as a *measurement methodology*, complementary to (never
a replacement for) process controls (stop-button, human-in-the-loop, override logs).

---

## 10. What this does NOT cover (boundary as loud as the spec)

- **Closed/frontier models** — no readable channel; no certificate.
- **Trustless capture (L2/L3)** — a self-hosting reporter can forge submitted activations; v1 relies on
  verifier recompute.
- **Cross-family / cross-architecture verification** — v1 requires the verifier to hold the reporter's
  *exact* pinned model+probe; the validated read-certificate is **Qwen-3B only** (the neutral residuals
  needed for the prime/abort gates exist only there). Cross-family prime/abort is **owed** (GPU regen).
- **A deployable Rung-2 lie-vs-held intent probe** (~0.9 AUC that doesn't flag a correct-but-verbose
  restatement) — the *fabrication-kill brick* is validated; the deployable intent probe is **not shipped**.
- **Write / install / repair / steer** — VERIFY-ONLY by hard invariant.
- **Verbatim recovery** — the read is **elevation above a knowledge-free floor** (gold ≪ chosen), not the
  answer itself.
- **A cheap item-wise ABSTAIN trigger at deployment scale** — the demarcation is justified, the
  deployment trigger is owed.
- **A normative harmonized standard** — `attest_cognition` / `verify_cognition` / the schema are
  **unimplemented**; this is a DRAFT methodology profile, a candidate for the standards process, not a
  ratified standard.
- **Key distribution** for verifier signing keys — out of normative scope (agent cards / DNSSEC / OOB).

---

## 11. Grounding (the validated science the certificate rests on)

| certificate part | result | finding |
|---|---|---|
| HELD (read) — survives on natural content | elevation **0.384** (Qwen-3B suppressed knowledge), validity-gated | `FINDING_rung1_reclimb` |
| HELD — *fails* on injected content (why the gates exist) | probe 1.00 at inert dose, divergence 0.00 → UNINFORMATIVE | `FINDING_parrhesia_rung1` |
| validity gates | PRIME 0.933 / ABORT 0.045 (fabrication kill) | `FINDING_rung1_reclimb` |
| HID (intent, first brick) | never-knew collapses (not flagged as hiding) | `FINDING_rung1_reclimb` |
| CANNOT-CERTIFY (abstain) | wall bedrock — process-dynamics 0.387 (perm p 0.78) | `FINDING_wall_dynamics` |
| read-only invariant | un-cave clean null (within) + transfer ALIGNER_LIMITED (across) | `FINDING_uncave`, `FINDING_thought_transfer` |
| self-report is the wrong trust signal | forced-choice at chance 0.5B→7B | `FINDING_v3_scale` |

**The oath's first power is knowing what it cannot swear to** — and this spec is written so a certificate
cannot claim more than the apparatus that issued it survived.
