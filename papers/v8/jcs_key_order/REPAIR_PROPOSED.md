# REPAIR PROPOSED — `styxx/attestation.py`, JCS object key order

Fathom Lab · 2026-09-08 · **A proposal, not a change.** Nothing in this session edited shipped code
and nothing was committed. Companion to `FINDING_jcs_key_order_2026_09_08.md`; every number below
comes from that finding's `scan_key_order.json` and `reproduce_stdout.txt`. Repository
`fathom-lab/styxx` at `4dba3a72601a6f2d0d1cf62a89a6a172b1797b46`.

## The diff

One line. `styxx/attestation.py`, in `_jcs`:

```diff
@@ styxx/attestation.py:255 def _jcs(obj: Any) -> str:
     if isinstance(obj, dict):
         parts = (
             json.dumps(k, ensure_ascii=False) + ":" + _jcs(v)
-            for k, v in sorted(obj.items(), key=lambda kv: kv[0])
+            # RFC 8785 section 3.2.3 orders object properties by their UTF-16 code
+            # units, NOT by code point. Encoding to UTF-16 big-endian makes the
+            # bytewise comparison a code-unit comparison, which is the RFC's rule.
+            # The two orders differ only when sibling keys mix a non-BMP character
+            # (a surrogate pair, lead unit 0xD800-0xDBFF) with a BMP character at or
+            # above U+E000. papers/v8/jcs_key_order/ carries the measurement.
+            for k, v in sorted(obj.items(), key=lambda kv: kv[0].encode("utf-16-be"))
         )
         return "{" + ",".join(parts) + "}"
```

The docstring at line 256 should also lose its "ASCII keys" domain note, or keep it and say that
non-ASCII keys are now ordered by the RFC's rule rather than left undefined.

## Why that is exactly RFC 8785 section 3.2.3

Section 3.2.3 requires object properties to be sorted on the property name strings treated as
"arrays of UTF-16 [UNICODE] code units" (RFC 8785 §3.2.3), compared as unsigned integers and
independent of locale.

`s.encode("utf-16-be")` emits each UTF-16 code unit as two bytes, most significant first. Python's
`bytes` comparison is lexicographic and bytewise-unsigned, so comparing two such encodings compares
the code-unit sequences element by element as unsigned 16-bit integers, with a shorter prefix
sorting first — which is what comparing arrays of code units means. Big-endian is the endianness
that makes bytewise order equal code-unit order; little-endian would not. `utf-16-be` (rather than
`utf-16`) is required because the `utf-16` codec prepends a BOM, which would prefix every key
identically and, worse, be compared as a code unit.

Two independent confirmations that this is the right key, not an invention of this document:

- The reference Python implementation of the RFC does exactly this:
  `sorted(obj.items(), key=lambda kv: kv[0].encode("utf-16be"))` — `rfc8785/_impl.py:236` in
  `rfc8785` 0.1.4, under the comment naming RFC 8785 3.2.3 and stating that the RFC's examples imply
  big-endian.
- Both committed JavaScript canonicalizers reach the same order by a different route:
  `Array.prototype.sort()` with no comparator (`styxx/_data/sworn_verify.js:412,417`;
  `web/styxx_verify.js:106`), and ECMAScript's default string comparison is defined on UTF-16 code
  units. Run in node, both produce `{"\u{1F600}":2,"":1}`, sha256
  `28c95d1bbb2209223307e62f489020e8f9e0cfa16adf2daf6d88127a1e8dd22a`, agreeing byte-for-byte with
  `rfc8785.dumps` and disagreeing with the shipped Python.

After the repair, all four implementations produce the same bytes on the divergent object. No
JavaScript changes.

## The test that would pin it

Add to `tests/test_portable_attestation.py` (the file that already cross-validates the Python
portable digest against `web/styxx_verify.js`):

```python
def test_jcs_orders_object_keys_by_utf16_code_unit_not_code_point():
    """RFC 8785 section 3.2.3. U+1F600 is the surrogate pair D83D DE00, whose lead
    unit 0xD83D sorts BELOW the single code unit 0xE000, so the astral key comes
    FIRST — the opposite of code-point order. Regression for the defect measured in
    papers/v8/jcs_key_order/."""
    from styxx.attestation import jcs

    bmp, astral = "", "\U0001f600"
    assert jcs({bmp: 1, astral: 2}) == '{"' + astral + '":2,"' + bmp + '":1}'
    # and the control: all-BMP keys are unmoved by the repair
    assert jcs({bmp: 1, "é": 2, "a": 3, "￿": 4}) == \
        '{"a":3,"é":2,"' + bmp + '":1,"￿":4}'
```

Stronger, and cheap because the dependency is already a v8 dependency: a differential against
`rfc8785.dumps` over a hypothesis strategy that draws keys from
`{"a", "é", chr(0xE000), chr(0xFFFF), "\U00010000", "\U0001F600", "\U0010FFFF"}` and asserts
byte equality. `tests/test_v8_jcs.py` (uncommitted, this worktree) already carries that strategy at
lines 228-229 and 305-325.

Note that `tests/test_v8_jcs.py:545-552` currently **pins the defect** — it asserts the code-point
order as a recorded divergence, and `styxx/v8/jcs.py:_refuse_utf16_divergent_keys` refuses such
objects on the fallback path rather than emitting them. Both are uncommitted. Applying the repair
means:

1. flipping the assertion at `tests/test_v8_jcs.py:545-552` to the RFC order;
2. deleting `_refuse_utf16_divergent_keys` from `styxx/v8/jcs.py` and its docstring paragraph, since
   the fallback backend would no longer diverge and would no longer need to refuse;
3. correcting that docstring's "at or above U+D800" to "at or above U+E000" if the paragraph is kept
   in any form — U+D800-U+DFFF are surrogates and cannot appear in a well-formed string, so the
   reachable boundary is U+E000.

## Conformance implication

**The sworn conformance vector set does not have to be regenerated.**

`conformance/sworn/gen_vectors.py` computes every pinned number through this canonicalizer:
vector ids at line 299 (`_sha256(sworn._jcs({"mode":..., "inputs":...}))`), `core_sha256` at
lines 304-307, `sidecar_sha256` at line 320, and the set digest at line 492
(`index["set_sha256"] = _sha256(sworn._jcs(digest_body).encode("utf-8"))`); `sworn._jcs`
(`styxx/sworn.py:326-328`) is `styxx.attestation.jcs`. So the repair reaches all 3620 vectors and
the set digest `ca5e715ac66ecb4169f7d38a7702badff1fd3498a0249164f152de388a38144d`
(`conformance/sworn/index.json`).

It reaches them without moving them. The scan walked all 39 files under `conformance/sworn/**` and
decoded all 3981 blobs in `conformance/sworn/blobs.json`, canonicalizing the 2227 that are JSON. Over
the whole denominator — 2111 files, 531,162 objects, 3,791,239 keys — there are 0 objects with a
non-BMP key, 0 keys containing any character at or above U+E000, and 0 objects whose two orders
differ; the highest code point in any key anywhere is U+9ED1. Canonicalizing all 65,674 parsed
documents both ways gives 65,659 byte-identical and 0 differing (15 hold `NaN`/infinity and are
outside the JCS domain under either sort).

Regenerating would therefore reproduce the same vectors, the same `core_sha256` values, and the same
`set_sha256`. That is a reason **not** to regenerate: a committed receipt is history, and
re-deriving one in place to confirm a number nobody disputes is how a certificate gets silently
invalidated. If the set is regenerated for any other reason, `set_sha256` must come out unchanged; if
it does not, this finding is wrong and the repair should stop.

## Operator decision needed

Three, all for Flobi.

1. **Apply or defer.** The repair changes shipped behaviour on inputs that, as measured, no
   committed artifact contains. Deferring costs nothing today and costs a cross-language
   disagreement the first time an emoji or a CJK extension-B character becomes an object key
   (a model-generated label, a user-supplied field name, a filename in a manifest). Applying costs a
   one-line change plus the three uncommitted-file edits listed above.
2. **Release framing.** If applied, this is a canonicalization change in a library whose whole point
   is that its canonical form is stable. It wants a CHANGELOG entry that says plainly: the RFC 8785
   key order was wrong for keys mixing non-BMP with U+E000-and-above; no previously issued digest
   changes; the measurement is at `papers/v8/jcs_key_order/`. Whether that is a patch or a minor
   version is Flobi's call, not a mechanical one.
3. **Scope of the repair.** The legacy digest path `_canonical_payload`
   (`styxx/attestation.py:195`, `json.dumps(..., sort_keys=True)`) carries the same code-point
   ordering. It never claimed RFC 8785 conformance — it is documented as Python-specific and
   language-nonportable — so the diff above deliberately leaves it alone. Repairing it too would
   move the legacy `digest.value` for the same class of objects (none of which exist in the
   repository, by the same scan). Recommendation: leave it, and let the docstring at lines 202-209
   keep saying why the portable digest exists.

## Limits of this proposal

The diff was not applied and no test suite was run against it in this session; the equivalence claim
rests on canonicalizing every parsed document through a copy of `_jcs` that differs only in the sort
key (`scan_key_order.py::_jcs_repaired`), not on a patched tree passing `pytest`. That copy tracks
the shipped function by hand — if `_jcs` changes, the copy must be re-derived before its numbers mean
anything. The conformance conclusion is an argument from the scan plus a reading of
`gen_vectors.py`, not a regeneration: nobody ran `gen_vectors.py` under the repair, and this document
recommends against doing so.
