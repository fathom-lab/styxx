# styxx Attestation — Content-Addressing Spec v1.0

A styxx attestation is a content-addressed JSON artifact. Its structural
integrity is defined entirely by the scheme below — small enough to reimplement
from scratch in any language. `scripts/styxx_verify_standalone.py` is an
independent, stdlib-only Python reimplementation of this spec that imports
nothing from styxx, so you can verify a styxx attestation without trusting (or
installing) styxx.

This spec covers **structural integrity** (digest + Merkle chain + head). It
does NOT cover semantic re-derivation (claim verdicts, cognometric vitals) —
those require styxx and, for claims, the pinned repository tree. A correct
verifier reports those as `NOT CHECKED`, never asserts them.

## 1. Canonical payload

Given an attestation artifact (a JSON object), the canonical payload is:

```
core = artifact without the keys "generated_at" and "digest"
canonical = JSON of core with:
    - keys sorted (recursively)
    - no whitespace: item separator ",", key/value separator ":"
    - ensure_ascii = false (UTF-8 text emitted literally)
```

In Python this is exactly:

```python
core = {k: v for k, v in artifact.items() if k not in ("generated_at", "digest")}
canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

`generated_at` (a timestamp) and `digest` (the field being computed) are
excluded so the address is a pure function of the attested content.

## 2. Attestation digest

```
digest = SHA-256( canonical.encode("utf-8") )   # lowercase hex
```

Stored in the artifact as `digest.value` (with `digest.alg = "sha256"`). An
artifact is structurally intact iff `digest.value == SHA-256(canonical_payload)`.

## 3. Chain digest (Merkle linkage)

A chain artifact has `links` (ordered) and `head_chain_digest`. Each link
carries the full `attestation`, its `attestation_digest`, a `prev_chain_digest`,
and a `chain_digest`. The linkage:

```
genesis      = "styxx-attestation-chain-v1"
link_digest(prev, att_digest) = SHA-256( f"{prev}|{att_digest}".encode("utf-8") )
```

Walking the chain from genesis:

```
prev = genesis
for link in links:
    att_digest = SHA-256(canonical_payload(link.attestation))   # recompute, don't trust
    assert link.attestation_digest == att_digest
    assert link.prev_chain_digest  == prev
    assert link.chain_digest       == link_digest(prev, att_digest)
    prev = link.chain_digest
assert head_chain_digest == prev   # (== genesis if no links)
```

The `|` (U+007C) byte separates the two hex strings; both operands are
lowercase hex, so the separator is unambiguous.

## 4. Honest boundaries

- **Re-seal.** If an attacker edits the substrate AND recomputes the digest, the
  artifact is again internally consistent and structural verification passes.
  This is inherent to content addressing. To catch a re-sealed *chain*, anchor
  the head externally: a verifier that knows the expected head (published
  out-of-band) compares `head_chain_digest` against it. The standalone verifier
  accepts `--expected-head` for this.
- **Semantic claims.** Claim verdicts depend on git + the pinned repo tree;
  vitals scores depend on styxx's scoring instruments. Neither is re-derivable
  by a stdlib-only verifier. They are reported `NOT CHECKED`.
- **Cross-language portability.** The canonical payload uses Python's
  `json.dumps` number repr (e.g. `1.0` → `"1.0"`, small floats → `e`-notation).
  An independent **Python** reimplementation agrees byte-for-byte. A
  cross-language verifier (JS/browser/Go) needs a canonical-number scheme such
  as JCS (RFC 8785); that is future work. Until then, port in Python or
  normalize numbers per RFC 8785 on both sides.

## 5. Reference reimplementation

`scripts/styxx_verify_standalone.py` — stdlib only, ~150 lines. Usage:

```
python scripts/styxx_verify_standalone.py path/to/attestation.json
python scripts/styxx_verify_standalone.py path/to/chain.json --expected-head <hex>
```

Its digests are cross-validated byte-for-byte against the styxx library in
`tests/test_standalone_verifier.py`.
