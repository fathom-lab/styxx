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
- **Cross-language portability.** The §2 canonical payload uses Python's
  `json.dumps` number repr (e.g. `1.0` → `"1.0"`, small floats → `e`-notation).
  An independent **Python** reimplementation agrees byte-for-byte. A
  cross-language verifier (JS/browser/Go) needs a canonical-number scheme such
  as JCS (RFC 8785). The **portable digest** (§6) supplies exactly this — verify
  in any language.

## 6. Portable content address (`digest.portable`) — verify in any language

The legacy `digest.value` (§2) uses Python's json number repr and is not
language-portable. `digest.portable` is an **additive, versioned** second
address (alg `sha256-jcs`) computed over an RFC 8785 / JCS canonical form, so it
reproduces byte-for-byte in any language. The legacy `digest.value` is left
unchanged — every previously issued receipt stays valid.

Portable canonical payload = the same `core` (artifact minus `generated_at` and
`digest`), serialized by these JCS rules:

- objects: keys sorted by code point (styxx keys are ASCII → identical to JS
  UTF-16 sort), `{` + `"key":value` joined by `,` + `}`, no whitespace;
- arrays: `[` + values joined by `,` + `]`;
- strings: JSON-escaped (== `json.dumps(s, ensure_ascii=False)` ==
  `JSON.stringify(s)` for the styxx domain);
- `true` / `false` / `null` literally;
- **numbers: ECMAScript `Number::toString` (RFC 8785 §3.2.2.3)** — exactly what
  JavaScript's `String(n)` produces. Integers carry no `.0`; a saturating token
  serializes as `1`, not `1.0`.

```
digest.portable.value = SHA-256( jcs(core).encode("utf-8") )   # lowercase hex
```

Chains carry an additive parallel portable address: each link has
`attestation_portable_digest`, and `head_chain_portable_digest` rolls them with
the SAME hex-only Merkle rule as §3 (already language-agnostic).

**Reference reimplementations.** `styxx.attestation._compute_portable_digest`
(Python) and `web/styxx_verify.js` (JavaScript, zero-dependency, runs in Node
and the browser — open `web/verify.html` and paste). The two agree byte-for-byte
over a fuzz corpus of finite doubles and the 4 real artifact shapes
(`tests/test_portable_attestation.py`), including the saturating token that
diverges under the legacy scheme. NaN/Inf are not permitted; keys are assumed
ASCII (the styxx artifact domain).

## 7. Cognometric Transparency Log (RFC 6962) — no silent suppression

A receipt proves what it says; it cannot prove that **nothing was suppressed**.
The transparency log closes that gap the way Certificate Transparency does: an
append-only Merkle log over attestation digests, with two checkable proofs.

**Leaves.** Each leaf entry is an attestation's `digest.portable.value` (a hex
string), so the log inherits the cross-language content address.

**Hashing (string-tagged domain separation).** A documented deviation from RFC
6962's 0x00/0x01 byte tags — ASCII string tags so the pure-JS string SHA-256
works unchanged across languages. Functionally equivalent; a styxx log is NOT
submittable to a CT log and vice versa.

```
leaf_hash(entry)      = SHA-256("styxx-tlog-leaf:" + entry)            # hex
node_hash(left,right) = SHA-256("styxx-tlog-node:" + left + ":" + right)  # left,right hex
```

**Merkle Tree Hash** (RFC 6962 §2.1): `MTH({}) = SHA-256("")`,
`MTH({d}) = leaf_hash(d)`, and for n>1 with k = largest power of two < n,
`MTH(D) = node_hash(MTH(D[0:k]), MTH(D[k:n]))`.

**Inclusion proof** — `{leaf_index, tree_size, leaf_hash, audit_path, root}`
proves "entry i is in the log whose root is R" (audit path; RFC 6962 §2.1.1).

**Consistency proof** — `{first_size, second_size, first_root, second_root,
proof}` proves the size-n log is an append-only extension of the earlier size-m
log (RFC 6962 §2.1.2): no past leaf was edited, deleted, reordered, or
truncated. Verify against a **witnessed earlier root** to detect a rewrite of
history.

**Tree head** — `{tlog_version, log_id, size, root, timestamp, digest}`,
content-addressed by `digest` (sha256 over the canonical core). NOT
cryptographically signed: signing/witnessing/gossip is the equivocation-defeating
layer and is **out of scope** (honest boundary). Append-only-ness is proven only
relative to a tree head a verifier has witnessed.

**Reference reimplementations.** `styxx.transparency` (Python) and
`web/styxx_verify.js` (`leafHash` / `nodeHash` / `merkleTreeHash` /
`verifyInclusion` / `verifyConsistency`, plus `verify()` auto-dispatch on
`kind`). The two agree byte-for-byte on the root and on every inclusion +
consistency proof over the real log (`tests/test_transparency.py`), and the
browser page (`web/verify.html`) verifies a pasted proof client-side.

## 8. Redactable Cognometric Attestation (selective disclosure) — disclose one fact, keep the rest private

Every proof above forces you to publish the underlying text to re-derive
anything. A **redactable attestation** lets an agent disclose a CHOSEN SUBSET of
attested fields and prove each is exactly the value committed into the public
`digest.redactable.root`, while the rest of the response stays private. This is
selective DISCLOSURE (like SD-JWT / redactable Merkle credentials), **not**
zero-knowledge — see the honest boundary below.

**Leaves.** Flatten the attestation core (the artifact minus `generated_at` /
`digest`) to a canonical, pointer-sorted list of `(pointer, value)` scalar leaves
(JSON-pointer paths `a/b/0/c`; values serialized with the same RFC 8785 / JCS
rule as `digest.portable`). Each leaf gets a fresh **256-bit salt**.

**Hashing (string-tagged domain separation).** Same ethos as the transparency
log — ASCII string tags so the pure-JS string SHA-256 works across languages.

```
leaf_hash(salt,ptr,val) = SHA-256("styxx-redact-leaf:" + salt + ":" + jcs(ptr) + ":" + jcs(val))   # hex
node_hash(left,right)   = SHA-256("styxx-redact-node:" + left + ":" + right)                          # left,right hex
```

The salted leaf hashes roll into the same RFC 6962-style Merkle Tree Hash (k =
largest power of two < n); the root is published as `digest.redactable.root` with
`{alg:"sha256-redact", version, tree_size}`. **The salts are secret and are never
serialized into the public artifact.**

**Disclosure** — `{kind:"disclosure", alg, version, tree_size, root, fields}`
where each field is `{pointer, value, salt, leaf_index, audit_path}`. A pointer
selects itself and every descendant. The verifier recomputes `leaf_hash` and
checks its inclusion against the root (pass the public
`digest.redactable.root`, or a transparency-log leaf, to bind to the
append-only history). Undisclosed fields appear only as opaque sibling hashes.

**The salt is load-bearing.** A low-entropy field (a verdict in
{PASS,FAIL,ERROR}, a 0–1 score) is brute-forceable from an UNSALTED leaf hash by
anyone who knows the domain; the 256-bit per-leaf salt makes that infeasible.
Two commitments to the same object differ (fresh salts), so equal underlying
values are unlinkable across receipts.

**Additivity.** Adding `digest.redactable` leaves `digest.value` (legacy) and
`digest.portable.value` byte-identical — both canonical payloads exclude the
whole `digest` key — so every previously issued receipt stays valid. Redactable
mode is opt-in (`attest(..., redactable=True)`) and, by design, non-deterministic
(the salts are the confidentiality).

**Honest boundary (refused overclaim).** This does NOT prove a *predicate*
(range / threshold) over a HIDDEN value, and it does NOT re-derive scores — a
disclosed value is trusted as the *committed* value, inheriting the commit-time /
re-seal boundary caught only via the transparency log + an external witness. A
disclosure leaks the field **count** (tree size) and the disclosed pointers +
values; it hides every undisclosed pointer and value. Calling this a ZK range
proof would be the overclaim; styxx will not.

**Reference reimplementations.** `styxx.redact` (Python:
`redactable_commit` / `disclose` / `verify_disclosure`) and `web/styxx_verify.js`
(`verifyDisclosure`, plus `verify()` auto-dispatch on `kind === "disclosure"`).
The two agree byte-for-byte on the root and on every disclosure
(`tests/test_redact.py`), and `web/verify.html` verifies a pasted disclosure
client-side.

## 5. Reference reimplementation

`scripts/styxx_verify_standalone.py` — stdlib only, ~150 lines. Usage:

```
python scripts/styxx_verify_standalone.py path/to/attestation.json
python scripts/styxx_verify_standalone.py path/to/chain.json --expected-head <hex>
```

Its digests are cross-validated byte-for-byte against the styxx library in
`tests/test_standalone_verifier.py`.
