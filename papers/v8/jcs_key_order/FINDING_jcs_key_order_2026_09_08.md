# FINDING — `styxx.attestation.jcs` sorts object keys by code point; RFC 8785 requires UTF-16 code unit

Fathom Lab · 2026-09-08 · **A finding, not a result.** One defect in one function, one differential
between two backends, one scan of one repository at one commit. Not sworn; not in the charon log; no
certificate. Every number below is read from `scan_key_order.json` (sha256
`b9e9194e7b4c6d224d02dda939f3e56a4588bcec384c86857c5aaefe54abac26`) or from
`reproduce_stdout.txt` (sha256 `3e1029d8e56122c0feef5a9b32a69d2460f53bd3cc5a89fc4d330717919b4ea7`),
produced by `scan_key_order.py` (sha256
`189441be15a0b867ac0bec2a122137bb090cf6a7ec934a8bf8a0d19a01a95de2`) and `reproduce.py` (sha256
`dd6b7ce40a05fa8235e203bd37239e15797c837c07a37d13f891f73b38f79f2c`), with `scan_stdout.txt` at sha256
`45d005c8edeb3291e789048992454a040507d0aa5772233199233e399fa1b20e`. Repository
`fathom-lab/styxx` at commit `4dba3a72601a6f2d0d1cf62a89a6a172b1797b46`, worktree
`C:\Users\heyzo\clawd\wt\v8`. Nothing in this session changed shipped code and nothing was committed.

It is **not** a claim that any receipt is wrong, that any digest is unreproducible, or that any
verifier disagreed in the field. The scan below says the opposite, with a denominator.

## The defect

`styxx/attestation.py:274-277`:

```python
        parts = (
            json.dumps(k, ensure_ascii=False) + ":" + _jcs(v)
            for k, v in sorted(obj.items(), key=lambda kv: kv[0])
        )
```

`sorted(..., key=lambda kv: kv[0])` is Python's default `str` comparison, which orders by Unicode
**code point**. RFC 8785 section 3.2.3, "Sorting of Object Properties", requires instead that
property name strings be treated, for the purpose of the comparison, as
"arrays of UTF-16 [UNICODE] code units" (RFC 8785 §3.2.3) — compared as unsigned integers,
independent of locale. The reference Python implementation of the RFC reads the section the same way
and sorts on `kv[0].encode("utf-16be")` (`rfc8785/_impl.py:236`, with its comment that the RFC's
examples imply big-endian).

The two orders are identical over the whole BMP below U+D800 and identical again from U+E000 up **within the
BMP**; they differ only when one sibling key contains a character at or above U+10000 (which encodes
as a surrogate pair whose lead unit is 0xD800-0xDBFF) and another sibling key contains a character at
or above U+E000 in the deciding position. The lead surrogate 0xD800 is *below* 0xE000 as a code unit,
while the code point U+1F600 is *above* U+E000, so the two orderings invert.

The public `jcs()` (line 282) and `_portable_canonical_payload()` (line 291) route through the same
comparison, so the portable content address `digest.portable` ("sha256-jcs") inherits it, and so does
every caller listed under *Reach* below.

## How it was found

Inside the v8 work, not in production. `styxx/v8/jcs.py` (uncommitted, this worktree) canonicalizes
through **two** backends — `rfc8785.dumps` when the package is importable, `styxx.attestation.jcs`
otherwise — and the v8 contract requires the two to agree byte-for-byte. Writing the domain
validation for that shim is what surfaced the divergence: a differential between two implementations
of the same spec, which is the only reason it was seen at all. The shim currently refuses divergent
objects on the fallback path (`_refuse_utf16_divergent_keys`) rather than emitting bytes that
disagree with the RFC. Its docstring says the boundary is "at or above U+D800"; the reachable
boundary in valid text is U+E000, because U+D800-U+DFFF are surrogates and cannot appear in a
well-formed string. That is an erratum in an uncommitted file, noted here so it is not carried
forward.

## Reproduction

`python papers/v8/jcs_key_order/reproduce.py` — python 3.12.10, `rfc8785` 0.1.4, node v24.13.0,
Windows 11. Exit 0. Full stdout in `reproduce_stdout.txt`.

Divergent object — two keys, `"\uE000"` (code point 0xE000, UTF-16 units `[0xe000]`) and
`"\U0001F600"` (code point 0x1F600, UTF-16 units `[0xd83d, 0xde00]`):

| backend | canonical bytes | sha256 of those bytes |
|---|---|---|
| `styxx.attestation.jcs` | `{"\ue000":1,"😀":2}` | `871954531859c7572c6279f90eb83a594ddc3a289e8bdc28d2a84ffb8c1a1703` |
| `rfc8785` 0.1.4 | `{"😀":2,"\ue000":1}` | `28c95d1bbb2209223307e62f489020e8f9e0cfa16adf2daf6d88127a1e8dd22a` |
| `styxx/_data/sworn_verify.js` (node) | `{"😀":2,"\ue000":1}` | `28c95d1bbb2209223307e62f489020e8f9e0cfa16adf2daf6d88127a1e8dd22a` |
| `web/styxx_verify.js` (node) | `{"😀":2,"\ue000":1}` | `28c95d1bbb2209223307e62f489020e8f9e0cfa16adf2daf6d88127a1e8dd22a` |

Control object, all keys BMP (`"\uE000"`, `"é"`, `"a"`, `"\uFFFF"`): all four backends produce
`{"a":3,"é":2,"\ue000":1,"\uffff":4}`, sha256
`a2c2c69d71d8da4f6b24ed045b7d52581f110988a3fd2e9c33e9d4225d5bcfc6`. The divergence is confined to the
condition named above; it is not a general disagreement.

### Which implementations agree with which

Three of the four agree with RFC 8785 section 3.2.3. The Python library is the outlier.

Both JavaScript canonicalizers use `Array.prototype.sort()` with no comparator
(`styxx/_data/sworn_verify.js:412,417`; `web/styxx_verify.js:106`), and ECMAScript's default
string comparison is defined on UTF-16 code units, so the JS is RFC-correct **by construction**. That
was established by loading both files in node and hashing their output (table above), not by reading
the spec and assuming.

The practical consequence of the sign of this defect: the Python library and the two JavaScript
verifiers that are meant to be independent re-derivations of the same content address **would
disagree** on such an object. The cross-language agreement the portable digest exists to provide is
what the defect would break — for objects that, as measured below, do not exist in this repository.

## Blast radius

`python papers/v8/jcs_key_order/scan_key_order.py`. Full stdout in `scan_stdout.txt`, full record in
`scan_key_order.json`.

Rule applied to every object: **affected iff** `sorted(keys) != sorted(keys, key=lambda k:
k.encode("utf-16-be"))`. Objects are walked through `json`'s `object_pairs_hook`, so duplicate keys
are measured before any collapse.

Denominator — every git-tracked `*.json` and `*.jsonl` in the repository at `4dba3a72`, plus every
base64 blob in `conformance/sworn/blobs.json` that decodes to JSON (those blobs are the payloads the
sworn conformance vectors take `core_sha256` over):

| | count |
|---|---|
| files enumerated (`git ls-files "*.json" "*.jsonl"`) | 2111 |
| files parsed | **2111** (0 unparseable, 0 unreadable; 11 vendored `data/bfcl_v3/*.json` are JSON Lines under a `.json` name and were parsed that way) |
| of which under `papers/**` | 1676 |
| of which under `conformance/sworn/**` | 39 |
| of which `*.certificate.json` | 213 |
| of which `*.sworn.json` | 38 |
| of which `*.sworn-receipt.json` | 46 |
| of which `*.jsonl` (includes `papers/charon/charon.log.jsonl`, 244 lines) | 161 |
| sworn blobs decoded and walked as JSON | 2227 of 3981 |
| blob remainder | 1745 document text (hashed as raw bytes, never through `jcs`), 8 JSON-shaped but malformed by design (fixtures the verifier must reject, so no canonical form is taken over them), 1 undecodable |

Counts:

| | count |
|---|---|
| objects walked | **531,162** |
| keys walked | **3,791,239** |
| objects with any non-ASCII key | **23** |
| objects with any non-BMP key | **0** |
| objects where the two orders differ | **0** |
| keys containing any character at or above U+E000 | **0** |
| maximum code point over all 3,791,239 keys | **U+9ED1** (`黑`) |

64 distinct non-ASCII keys appear at all. The highest any of them reaches is U+9ED1; the divergence
needs U+10000 or above on one side and U+E000 or above on the other. Neither condition is met
anywhere in the denominator, and the second is not met by a margin of 0x4B2F code points.

### The same answer, measured a second way

Counting divergent objects proves the point by argument. The scan also proves it by bytes: every
parsed document is canonicalized twice — once through the shipped `styxx.attestation._jcs`, once
through a copy of that function differing in nothing but the sort key — and the two byte strings are
compared.

| | count |
|---|---|
| documents canonicalized both ways | 65,674 |
| byte-identical | **65,659** |
| bytes differ | **0** |
| outside the JCS domain under both sorts | 15 |

The 15 are documents holding `NaN` or an infinity (`ValueError: portable digest is defined for
finite numbers only`), which `jcs()` refuses under either sort, so no canonical form and no digest
exists over them through this path either way. They are listed in `scan_key_order.json` under
`repair_equivalence.domain_error_examples`.

**Zero committed artifacts are affected, out of 2111 files, 531,162 objects and 3,791,239 keys.** No
committed digest moves under the repair: 65,659 of 65,659 canonicalizable documents produce identical
bytes under both sorts. The repair is a behaviour change that no committed artifact can observe.

### Reach of the defect in code (not affected today, but where it would land)

`styxx.attestation.jcs` / `_jcs` is the canonicalizer behind: `styxx/sworn.py:326-328` (and through
it every `core_sha256`, receipt `digest`, and `coverage_sha256` at lines 752, 1832, 1836, 1880, 1906,
1912), `styxx/charon.py:94` (entry ids at 446, header digest at 459, re-derivation at 524, drift
comparison at 657), `styxx/capsule.py:704,911` (gate binding, sworn portable core),
`styxx/redact.py:80` (redaction leaf hashes), `styxx/worklog.py:97-98`, `styxx/harness/junit.py:76`,
`conformance/sworn/differential.py:37,268,304`, and `styxx/v8/jcs.py:34` (fallback path only). All of
those inherit the ordering; none of them has an artifact that triggers it.

## Proposed repair

See `REPAIR_PROPOSED.md` beside this file. Not applied in this session.

## Limits

One repository at one commit; the scan says nothing about artifacts held anywhere else, about
receipts issued to third parties, or about any styxx release on PyPI. The scan covers **objects that
exist as committed JSON** — it cannot see an object that a future caller constructs in memory and
hands to `jcs()`, which is exactly the case the repair is for. The 1745 document blobs and the 8
malformed blobs were not walked as objects because they are not objects; that is an argument from
what they are, not a measurement of their key order. The legacy digest path
`_canonical_payload` (`styxx/attestation.py:195`, `json.dumps(..., sort_keys=True)`) carries the same
code-point ordering, but it never claimed RFC 8785 conformance — it is documented as Python-specific
and language-nonportable — so it is out of scope here and is not repaired by the proposed diff. The
JavaScript verifiers were exercised on two objects, the divergent one and the control; that
establishes their ordering on the deciding case, not their conformance across RFC 8785 as a whole.
No claim is made about `Object.keys()` ordering for integer-like keys in either JS file, which is a
separate question this finding did not examine.
