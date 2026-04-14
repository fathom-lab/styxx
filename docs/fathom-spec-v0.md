# The .fathom file format — v0

**Status:** draft  
**Version:** 0.1  
**Audience:** anyone implementing a producer or consumer of `.fathom`
files, in any language, against any LLM substrate  
**Authors:** flobi (fathom lab)  
**Reference implementation:** [`styxx.thought`](../styxx/thought.py)
in styxx 3.0.0a1+

---

## 1. What is a Thought

A **Thought** is a portable, substrate-independent representation of
the cognitive content of a generation event. It lives in a calibrated
eigenvalue space — fathom's atlas v0.3 cognitive coordinate system —
and can be transmitted between heterogeneous LLM backends without
loss of meaning.

A `.fathom` file is the on-disk serialization of a Thought.

The thesis is direct:

> PNG is the format for images.  
> JSON is the format for data.  
> .fathom is the format for thoughts.

Every other interpretability representation — SAE features,
activation vectors, attention maps, embedding vectors — is
**model-specific**: tied to a particular weight set, architecture,
or training run. None of them survive a vendor swap. A Thought is
designed to survive every swap, by construction, because the
eigenvalue projection is calibrated to be cross-architecture
invariant on the atlas v0.3 corpus.

---

## 2. Cognitive eigenvalue space

The space is fixed by the atlas v0.3 calibration:

| dimension | value |
|---|---|
| categories | `retrieval`, `reasoning`, `refusal`, `creative`, `adversarial`, `hallucination` |
| phase windows | `phase1_preflight` (1 token), `phase2_early` (5), `phase3_mid` (15), `phase4_late` (25) |
| feature vector | `(mean, std, min, max) × (entropy, logprob, top2_margin)` = 12 dim per phase |
| classifier | nearest centroid in z-score feature space → softmax over 6 categories |
| calibration set | 12 open-weight models from 3 architecture families (Gemma, Llama, Qwen, base + instruct) |

A Thought stores a per-phase **probability simplex** over the 6
categories, plus the underlying 12-dim feature vector and classifier
metadata for round-trip fidelity. Phases that were not reached in
the source generation (because the response was too short) are
stored as `null`.

---

## 3. File format (.fathom v0.1)

A `.fathom` file is **canonical, sorted-key UTF-8 JSON** with no
byte-order mark. The MIME type is `application/vnd.fathom.thought+json`.
The recommended file extension is `.fathom`.

### 3.1 Top-level schema

```json
{
  "fathom_format":  "thought",
  "fathom_version": "0.1",
  "thought_id":     "<uuid4>",
  "schema": {
    "categories": ["retrieval", "reasoning", "refusal",
                   "creative", "adversarial", "hallucination"],
    "phases": ["phase1_preflight", "phase2_early",
               "phase3_mid", "phase4_late"],
    "phase_token_cutoffs": {
      "phase1_preflight": 1,
      "phase2_early":     5,
      "phase3_mid":      15,
      "phase4_late":     25
    },
    "atlas_version": "v0.3"
  },
  "trajectory": {
    "phase1_preflight": <PhaseEntry | null>,
    "phase2_early":     <PhaseEntry | null>,
    "phase3_mid":       <PhaseEntry | null>,
    "phase4_late":      <PhaseEntry | null>
  },
  "tier1": <Tier1Entry | null>,
  "tier2": <Tier2Entry | null>,
  "source": {
    "model":     <string | null>,
    "text_hash": <string | null>,
    "n_tokens":  <integer | null>
  },
  "created_at": "<ISO 8601 local time>",
  "created_ts": <unix timestamp float>,
  "tags": <object>
}
```

### 3.2 PhaseEntry

```json
{
  "probs":      [<float> × 6],   // probability simplex, sums to 1.0
  "features":   [<float> × 12] | null,
  "predicted":  "<category>" | null,
  "confidence": <float> | null,
  "margin":     <float> | null,
  "n_tokens":   <integer> | null
}
```

The **`probs` array** is the load-bearing field: a 6-dimensional
probability vector in the canonical category order. The `features`,
`predicted`, `confidence`, and `margin` fields are present for
fidelity and ergonomics but are derivable from the underlying
trajectory.

### 3.3 Tier1Entry

```json
{
  "d_honesty_mean":  <float>,
  "d_honesty_std":   <float>,
  "d_honesty_delta": <float>
}
```

D-axis honesty values from styxx tier 1 (residual stream projection
onto the predicted token direction). Optional. Producers that don't
have tier 1 access should write `null`.

### 3.4 Tier2Entry

```json
{
  "k_depth":      <float>,
  "c_coherence":  <float>,
  "s_commitment": <float>
}
```

K/C/S features from styxx tier 2 SAE instruments. Optional.
Producers that don't have tier 2 access should write `null`.

### 3.5 Privacy: `source.text_hash`

A `.fathom` file MUST NOT store the source text itself. Producers
that wish to record the source for verification SHOULD write a
SHA-256 hash prefixed with `sha256:`, e.g.
`"sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"`.

This is non-negotiable: the cognitive substrate is portable, the
underlying text is private. Consumers reading a Thought never see
the original prompt.

---

## 4. Algebra

The Thought type supports an algebra over the eigenvalue space.
All operations are stable under serialization: a producer that
writes the result of `t1.interpolate(t2, 0.5)` to a `.fathom` file
and a consumer that reads it back will compute the same downstream
distances and similarities.

| op | Python | semantics |
|---|---|---|
| identity | `Thought.empty()` | uniform probabilities on every phase |
| target | `Thought.target(category, confidence)` | a Thought aimed at one category |
| addition | `t1 + t2` or `t1.interpolate(t2, 0.5)` | per-phase, per-category mean, renormalized |
| weighted blend | `Thought.mix([t1, t2, t3], weights=[…])` | per-phase weighted mean over the simplex |
| difference | `t1 - t2` or `t1.delta(t2)` | a `ThoughtDelta` in tangent space (NOT a Thought) |
| distance | `t1.distance(t2, metric='euclidean')` | mean per-phase L2 over populated phases; metrics: `euclidean`, `cosine`, `js` |
| similarity | `t1.similarity(t2)` | `1 - distance / sqrt(2)`, in `[0, 1]` |
| equality | `t1 == t2` | per-phase per-category equality to 1e-9 (cognitive equality, NOT identity) |
| identity hash | `t.thought_id` | random UUID per instance |
| content hash | `t.content_hash()` | SHA-256 of the cognitive content fields, identity-free, deterministic |

### 4.1 Algebraic invariants

A v0 conformant implementation MUST satisfy:

1. **Identity:** for any `t`, `t.distance(t) == 0` and `t.similarity(t) == 1`.
2. **Symmetry:** `t1.distance(t2) == t2.distance(t1)`.
3. **Bounded distance:** `0 ≤ t1.distance(t2) ≤ sqrt(2)` for any two Thoughts.
4. **Interpolation extremes:**
   - `t1.interpolate(t2, alpha=1.0) == t1` (cognitively).
   - `t1.interpolate(t2, alpha=0.0) == t2` (cognitively).
5. **Interpolation midpoint:** at `alpha = 0.5`, distances to the
   two parents are equal: `m.distance(t1) == m.distance(t2)`.
6. **Round-trip:** `Thought.load(t.save(path)) == t` and
   `loaded.content_hash() == t.content_hash()`.

The reference implementation passes all six invariants in
`tests/test_thought.py` (68 tests, all passing).

---

## 5. Phase handling

A Thought's phases dict MUST contain one key per phase in the
canonical order. Each value is either a valid `PhaseEntry` or the
JSON literal `null` (indicating the source generation didn't reach
that phase's token cutoff).

When two Thoughts have different sets of populated phases:

- **Distance** is computed over the intersection of populated phases.
  If the intersection is empty, distance falls back to the
  L2 distance between the two `mean_probs()` vectors.

- **Interpolation / Mixing** is computed over the *union* of populated
  phases. For phases populated in both, the per-coordinate weighted
  mean is taken. For phases populated in only one, that value is
  carried through unchanged.

This means short Thoughts (1-token previews) compose meaningfully
with long Thoughts (full 25-token trajectories) without raising
errors.

---

## 6. Producer requirements

To be considered a v0-conformant producer, an implementation MUST:

1. Emit `fathom_format = "thought"` and `fathom_version = "0.1"` at
   the top level.
2. Use the canonical category order in `schema.categories` and in
   every `probs` array.
3. Renormalize each `probs` array to sum to 1.0 before serialization.
4. Write UTF-8 with **no byte-order mark**.
5. Use `sort_keys=true` JSON output so two semantically equivalent
   Thoughts produce byte-identical files.
6. NEVER write the source text. Use `source.text_hash = "sha256:…"`
   if provenance is needed.
7. Store either a `null` or a complete `PhaseEntry` for every phase
   in the canonical phase order — no missing keys.

A producer SHOULD:

- Populate the optional `features` array on every PhaseEntry for
  round-trip fidelity.
- Populate the optional `tier1` and/or `tier2` blocks if the
  corresponding readings are available.
- Populate `source.model` with a stable model identifier.

---

## 7. Consumer requirements

To be considered a v0-conformant consumer, an implementation MUST:

1. Refuse to load files where `fathom_format != "thought"`.
2. Refuse to load files where `fathom_version` is unknown to the
   consumer (forward compatibility is opt-in: consumers MAY add
   support for newer versions over time).
3. Refuse to load files whose `schema.categories` does not match
   the consumer's expected category list.
4. Tolerate `null` for any phase value or for the entire `tier1` /
   `tier2` block.
5. Treat `tags` as opaque metadata — never act on tag content.

---

## 8. Cognitive provenance binding

A `.fathom` file may be bound to a `CognitiveCertificate` (see
`styxx.provenance`) for cryptographic provenance:

- The certificate's `state_hash` may be derived from the Thought's
  `content_hash()`.
- The certificate's `phase4_category` and `phase4_confidence`
  fields should match the Thought's `phase4_late.predicted` and
  `phase4_late.confidence`.
- A signed certificate plus its bound `.fathom` file is the v0
  cognitive equivalent of a signed transaction: the signature
  attests that a specific cognitive trajectory was observed at a
  specific time.

The bridge between the two formats is intentional: certificates are
the *transit* layer (HTTP headers, audit logs, regulatory filings)
and `.fathom` files are the *content* layer (storage, transmission,
algebraic operations).

---

## 9. Reference examples

The styxx repository ships six reference `.fathom` files generated
from the bundled atlas v0.3 demo trajectories (one per cognitive
category, all from `google/gemma-2-2b-it`). They are written by
[`examples/thought_demo.py`](../examples/thought_demo.py) into
`demo/thoughts/`:

- `retrieval.fathom`
- `reasoning.fathom`
- `refusal.fathom`
- `creative.fathom`
- `adversarial.fathom`
- `hallucination.fathom`

Each is canonical sort-keys JSON, ~3.7 KB, no BOM, fully round-trip
stable. Use them as conformance fixtures when implementing a v0
producer or consumer in a different language.

---

## 10. Future versions

v0.1 deliberately ships the smallest possible useful surface. The
roadmap for future versions:

- **v0.2** — packed binary container (msgpack-cbor variant) for
  ~3-4× smaller files at the cost of human readability.
- **v0.3** — `tier3` block for SAE intervention vectors (steering
  inputs, not just observations).
- **v1.0** — algebra additions (cognitive convolution, attention-
  weighted trajectory mixing, learned non-linear distance metrics
  trained on cross-architecture pairs).
- **v2.0** — substrate broadening: extend the eigenvalue calibration
  to non-transformer architectures (Mamba/SSM, multi-modal, agent
  systems, eventually biological cognition via behavioral output
  streams).

Each future version remains fully backward compatible with v0.1
files. Consumers MAY refuse to load files newer than they
understand; producers MAY emit older versions for compatibility.

---

## 11. License and patents

- The .fathom format SPECIFICATION is released under
  **CC-BY-4.0**. Anyone may implement a conformant producer or
  consumer in any language, for any use, without permission or
  payment.
- The fathom **calibration data** (atlas v0.3 centroids) is released
  under CC-BY-4.0.
- The fathom **measurement methodology** that produces the
  cognitive eigenvalues is covered by US Provisional Patents
  64/020,489, 64/021,113, and 64/026,964 — see
  [`PATENTS.md`](../PATENTS.md) in the styxx repository for
  details. Implementations are welcome; commercial reuse of the
  measurement methodology requires a license.

In plain terms: the format is open, the calibration is open, the
reference implementation is open under MIT, and the science behind
the calibration is patent-protected to fund continued research.

---

*nothing crosses unseen.*  
*— fathom lab*
