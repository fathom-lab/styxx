# -*- coding: utf-8 -*-
"""RECEIPT v0 (DRAFT, unsigned, nothing frozen): the projector.

One file, standard library only, nothing imported from the lab's package. Three things:

  jcs(value)                 the canonical form of RECEIPT_v0.md section 3 (no floats)
  to_v0(core, document=None) today's verdict core, or a committed /v1 or /v2 verdict receipt,
                             projected onto the v0 core with draft decisions D1, D2, D3 applied
  digest_v0(core)            sha256 over the UTF-8 bytes of jcs(core), lowercase hex

Every function is pure: its result depends on its arguments and on nothing else. A refusal is a
`Refusal` (a ValueError) carrying one code from REFUSALS.

What the projector does NOT do: it does not lex a document, resolve a receipt or compare anything.
It rewrites rows a checker already produced. Where the v0 text asks for something a row never
carried (the claim, when no document is handed over), the result is marked with the legacy
profile and says so; it is never filled with a guess.
"""
import hashlib

__all__ = ["jcs", "to_v0", "digest_v0", "receipt_v0", "Refusal",
           "SCHEMA_V0", "PROFILE_RN", "PROFILE_LEGACY", "SPAN_VERDICTS", "DOCUMENT_VERDICTS",
           "REASONS_V0", "REASONS_LEGACY_ONLY", "REFUSALS", "KINDS", "CLAIM_CAP_CODE_POINTS"]

SCHEMA_V0 = "receipt/v0-draft"
PROFILE_RN = "rn-inline/v0-draft"
PROFILE_LEGACY = "legacy-projection/v0-draft"
FORMAT = "sworn/0.1"
SOURCE_SCHEMAS = ("styxx.sworn.verdict-receipt/v1", "styxx.sworn.verdict-receipt/v2")
KINDS = ("numeric", "quote", "hash", "absent")
SPAN_VERDICTS = ("HELD", "FAILED", "UNRESOLVED", "MALFORMED")
DOCUMENT_VERDICTS = ("SWORN-HELD", "SWORN-FAILED", "SWORN-UNCHECKED", "UNSWORN")
CLAIM_CAP_CODE_POINTS = 300

# Reason codes a v0 checker can produce (RECEIPT_v0.md section 10), by verdict.
REASONS_MALFORMED = (
    "tag_syntax", "nesting", "stray_closer", "unclosed", "empty_span", "length_cap",
    "receipt_form", "kind_unknown", "kind_reserved", "number_count", "number_grammar",
    "needle_count", "needle_empty", "digest_form", "absent_over_partial", "hash_over_partial",
    "hidden_commitment", "short_needle", "directional_override",
    "pointer_unresolvable", "pointer_ambiguous", "anchor_out_of_range", "leaf_not_scalar",
    "leaf_not_numeric", "leaf_not_string", "receipt_not_json", "receipt_author_minted",
    "kind_of_source_unknown",
)
REASONS_UNRESOLVED = (
    "manifest_absent", "manifest_id_missing", "manifest_integrity", "manifest_bytes_absent",
    "manifest_no_completeness", "rung_unknown", "no_repository",
)
REASONS_FAILED = ("value_mismatch", "rounds_to_zero", "needle_missing", "needle_present",
                  "digest_mismatch")
REASONS_V0 = REASONS_MALFORMED + REASONS_UNRESOLVED + REASONS_FAILED
# MALFORMED reasons decided on the document's bytes alone (RECEIPT_v0.md 7.1a-h, section 10): the
# only MALFORMED a v0 checker can give a path: or prereg: span, which it never resolves (4.4).
REASONS_DOCUMENT_ONLY = (
    "tag_syntax", "nesting", "stray_closer", "unclosed", "empty_span", "hidden_commitment",
    "length_cap", "kind_unknown", "kind_reserved", "receipt_form", "absent_over_partial",
    "hash_over_partial", "directional_override", "number_count", "number_grammar",
    "needle_count", "needle_empty", "digest_form",
)
# Reasons only a checker holding a repository can produce. Outside v0; a row carrying one can
# only come from a receipt issued before v0, and marks the projection as legacy.
REASONS_LEGACY_ONLY = ("no_commit", "commit_absent", "path_absent", "not_a_blob",
                       "receipt_too_large", "git_unavailable", "prereg_not_in_tree")
DOCUMENT_REASONS = ("invalid_utf8", "unbalanced_fences")

REFUSALS = (
    # the canonical form (section 3)
    "float", "int_range", "non_ascii_key", "lone_surrogate", "depth", "type",
    # the projector's own
    "not_a_core", "schema_out_of_scope", "format_unknown", "row_shape", "verdict_out_of_scope",
    "reason_unknown", "counts_mismatch", "document_mismatch", "claim_unlocatable",
    "name_form", "manifest_not_canonical", "source_digest",
)

# |n| <= 2^53 - 1 (RECEIPT_v0.md 3.1): every integer a binary double rounds onto lies outside it, so a
# reader that parses numbers as doubles refuses the same values this one does.
_INT_MAX = 2 ** 53 - 1
_DEPTH_MAX = 32
_LEAF_TYPES = {"_Obj": "object", "dict": "object", "list": "array", "NoneType": "null",
               "object": "object", "array": "array", "null": "null"}
_HEX = "0123456789abcdef"
_SOURCE_OUTSIDE_DIGEST = ("digest", "timestamp", "coverage", "coverage_sha256")
_SHORT = {0x08: "\\b", 0x09: "\\t", 0x0A: "\\n", 0x0C: "\\f", 0x0D: "\\r", 0x22: '\\"',
          0x5C: "\\\\"}


class Refusal(ValueError):
    """The projector or the canonical form declined. `code` is one of REFUSALS."""

    def __init__(self, code, message=""):
        ValueError.__init__(self, "%s: %s" % (code, message) if message else code)
        self.code = code


# ---------------------------------------------------------------------------------------------
# (a) canonical form: RFC 8785 restricted to null, booleans, integers, strings, arrays, objects
# ---------------------------------------------------------------------------------------------

def _jcs_string(s):
    out = ['"']
    for ch in s:
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            raise Refusal("lone_surrogate", "U+%04X" % cp)
        if cp in _SHORT:
            out.append(_SHORT[cp])
        elif cp < 0x20:
            out.append("\\u00" + _HEX[cp >> 4] + _HEX[cp & 15])
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _jcs(v, depth):
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        if v > _INT_MAX or v < -_INT_MAX:
            raise Refusal("int_range", str(v)[:40])
        return str(v)
    if isinstance(v, float):
        raise Refusal("float", repr(v))
    if isinstance(v, str):
        return _jcs_string(v)
    if isinstance(v, (list, tuple)):
        if depth >= _DEPTH_MAX:
            raise Refusal("depth")
        return "[" + ",".join(_jcs(x, depth + 1) for x in v) + "]"
    if isinstance(v, dict):
        if depth >= _DEPTH_MAX:
            raise Refusal("depth")
        for k in v:
            if not isinstance(k, str):
                raise Refusal("type", "object key is %s" % type(k).__name__)
            if any(ord(c) > 0x7E for c in k):
                raise Refusal("non_ascii_key", ascii(k)[:60])
        # keys are U+0000..U+007E here, so code point order and UTF-16 code unit order agree
        return "{" + ",".join(_jcs_string(k) + ":" + _jcs(v[k], depth + 1)
                              for k in sorted(v)) + "}"
    raise Refusal("type", type(v).__name__)


def jcs(value):
    """The canonical text of `value` (a str). Refuses, never approximates."""
    return _jcs(value, 0)


def _sha256(b):
    return hashlib.sha256(b).hexdigest()


def digest_v0(core):
    """sha256 over the UTF-8 bytes of the canonical text of a v0 core; 64 lowercase hex."""
    return _sha256(jcs(core).encode("utf-8"))


def receipt_v0(core, verifier=None):
    """The receipt object of section 9: the core, its digest, and two things BESIDE it that the
    digest does not cover: `verifier` (provenance, free-form object) and `attestations`, the one
    named slot v0 leaves empty."""
    out = {"core": core, "digest": digest_v0(core), "attestations": []}
    if verifier is not None:
        if not isinstance(verifier, dict):
            raise Refusal("type", "verifier must be an object")
        out["verifier"] = verifier
    return out


# ---------------------------------------------------------------------------------------------
# (b) the projection
# ---------------------------------------------------------------------------------------------

def _decimal_text_is_zero(text):
    """True when a decimal string (plain or with an E exponent, optional sign) has no non-zero
    digit in its coefficient. Decided on the text; no number is ever built."""
    t = text.strip()
    if t[:1] in "+-":
        t = t[1:]
    for mark in ("E", "e"):
        if mark in t:
            t = t.split(mark, 1)[0]
    digits = t.replace(".", "")
    if not digits or any(c not in "0123456789" for c in digits):
        raise Refusal("row_shape", "not a decimal string: %r" % text[:40])
    return all(c == "0" for c in digits)


def _is_hex64(x):
    return isinstance(x, str) and len(x) == 64 and all(c in _HEX for c in x)


def _is_hex40(x):
    return isinstance(x, str) and len(x) == 40 and all(c in _HEX for c in x)


def _ref_form(ref):
    """'rn', 'path', 'prereg' or None, from the head of a REF (RECEIPT_v0.md 4.2, 4.4)."""
    if not isinstance(ref, str):
        return None
    head = ref.split("#", 1)[0]
    if head.startswith("path:"):
        return "path"
    if head.startswith("prereg:"):
        return "prereg"
    if (len(head) >= 2 and head[0] == "r" and head[1] in "123456789"
            and all(c in "0123456789" for c in head[2:])):
        return "rn"
    return None


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _detail_v0(reason, detail):
    """Section 7's detail object: today's keys, with the one Python type name replaced."""
    if not isinstance(detail, dict):
        raise Refusal("row_shape", "detail is not an object")
    out = dict(detail)
    if reason == "leaf_not_scalar" and "leaf_type" in out:
        name = out["leaf_type"]
        if name not in _LEAF_TYPES:
            raise Refusal("row_shape", "leaf_type %r" % (name,))
        out["leaf_type"] = _LEAF_TYPES[name]
    return out


def _claim(row, document):
    """(claim_sha256, claim) for one row: the inner bytes exactly as written, located from the
    row's own offsets. (None, None) when the row has no canonical offsets or no document is given."""
    start, end = row.get("start"), row.get("end")
    if document is None or start is None or end is None:
        return None, None
    at, ref, kind = row.get("at"), row.get("receipt"), row.get("kind")
    if not (_is_int(at) and _is_int(start) and _is_int(end) and isinstance(ref, str)
            and isinstance(kind, str) and 0 <= start <= end):
        raise Refusal("row_shape", "a row with offsets must carry at, receipt and kind")
    opener = ('<sworn r="%s" k="%s">' % (ref, kind)).encode("utf-8")
    inner_at = at + len(opener)
    inner_end = inner_at + (end - start)
    if document[at:inner_at] != opener or document[inner_end:inner_end + 8] != b"</sworn>":
        raise Refusal("claim_unlocatable", "no span of that shape at byte %d" % at)
    inner = document[inner_at:inner_end]
    try:
        text = inner.decode("utf-8")
    except UnicodeDecodeError:
        raise Refusal("claim_unlocatable", "inner bytes at %d are not UTF-8" % at)
    return _sha256(inner), (text if len(text) <= CLAIM_CAP_CODE_POINTS else None)


def _row_v0(row, document):
    if not isinstance(row, dict):
        raise Refusal("row_shape", "a span row is not an object")
    verdict, reason = row.get("verdict"), row.get("reason")
    if verdict not in SPAN_VERDICTS:
        raise Refusal("verdict_out_of_scope", "%r" % (verdict,))
    if verdict == "HELD":
        if reason is not None:
            raise Refusal("row_shape", "a HELD row carries a reason")
    elif reason not in REASONS_V0 and reason not in REASONS_LEGACY_ONLY:
        raise Refusal("reason_unknown", "%r" % (reason,))
    if verdict in ("HELD", "FAILED") and not (
            isinstance(row.get("provenance"), dict) and _is_hex64(row.get("resolved_sha256"))):
        raise Refusal("row_shape", "a %s row without provenance and resolved_sha256" % verdict)
    detail = _detail_v0(reason, row.get("detail") if row.get("detail") is not None else {})
    # D2 (draft): a numeric comparison that held only because a non-zero receipt rounds to zero
    # at the printed precision does not hold.
    if (verdict == "HELD" and row.get("kind") == "numeric"
            and isinstance(detail.get("receipt"), str)
            and isinstance(detail.get("receipt_rounded"), str)
            and not _decimal_text_is_zero(detail["receipt"])
            and _decimal_text_is_zero(detail["receipt_rounded"])):
        verdict, reason = "FAILED", "rounds_to_zero"
    claim_sha256, claim = _claim(row, document)                                   # D3 (draft)
    out = {"at": row.get("at"), "start": row.get("start"), "end": row.get("end"),
           "receipt": row.get("receipt"), "kind": row.get("kind"),
           "claim_sha256": claim_sha256, "claim": claim,
           "verdict": verdict, "reason": reason, "detail": detail}
    if "resolved_sha256" in row:
        out["resolved_sha256"] = row["resolved_sha256"]
    if "provenance" in row:
        out["provenance"] = row["provenance"]
    return out


def _document_verdict(document_malformed, counts, total):
    """D1 (draft). Held needs at least one HELD span and no FAILED or MALFORMED span; spans with
    nothing failed and nothing held is its own word."""
    if document_malformed is not None:
        return "SWORN-FAILED"
    if total == 0:
        return "UNSWORN"
    if counts["FAILED"] or counts["MALFORMED"]:
        return "SWORN-FAILED"
    if counts["HELD"] == 0:
        return "SWORN-UNCHECKED"
    return "SWORN-HELD"


def _is_legacy(rows):
    """True when a v0 checker could not give these rows: a row resolved through a repository, a
    legacy-only reason, a path: or prereg: span answered by anything but what 4.4 allows, or
    offsets without a bound claim."""
    for r in rows:
        prov = r.get("provenance")
        if isinstance(prov, dict) and prov.get("form") in ("path", "prereg"):
            return True
        if r["reason"] in REASONS_LEGACY_ONLY:
            return True
        if r["start"] is not None and r["claim_sha256"] is None:
            return True
        if _ref_form(r["receipt"]) in ("path", "prereg") and not (
                (r["verdict"], r["reason"]) == ("UNRESOLVED", "no_repository")
                or (r["verdict"] == "MALFORMED" and r["reason"] in REASONS_DOCUMENT_ONLY
                    and "provenance" not in r)):
            return True
    return False


def _rungs(rows):
    """RECEIPT_v0.md 9.1: only the keys that occur. A row without provenance (UNRESOLVED, and
    MALFORMED before 6.9) counts as 'unresolved'; legacy rows resolved through a repository as
    'committed'."""
    out = {}
    for r in rows:
        prov = r.get("provenance") or {}
        key = prov.get("rung") if prov.get("form") == "rn" else (
            "committed" if prov.get("form") in ("path", "prereg") else "unresolved")
        out[key] = out.get(key, 0) + 1
    return out


def _manifest_seen_without_digest(rows, manifest_digest):
    """True when a manifest was handed over (some rN row got past 6.1) but the core records no
    manifest digest: today's code reads such a manifest when it carries no digest key, and a v0
    checker answers every rN row past 6.1 with UNRESOLVED manifest_integrity (5.6, 6.2)."""
    if manifest_digest is not None:
        return False
    for r in rows:
        if _ref_form(r["receipt"]) != "rn":
            continue
        if (r["verdict"] == "MALFORMED" and r["reason"] in REASONS_DOCUMENT_ONLY
                and "provenance" not in r):
            continue
        if (r["verdict"], r["reason"]) in (("UNRESOLVED", "manifest_absent"),
                                           ("UNRESOLVED", "manifest_integrity")):
            continue
        return True
    return False


def to_v0(core, document=None):
    """Project a verdict core (what the checker returns today) or a committed /v1 or /v2 verdict
    receipt onto the v0 core.

    `document` is the inline document's bytes. When given, its sha256 must be the one the core
    names, and every row with canonical offsets gets its claim bound (D3). When not given, rows
    carry null claims and the result is marked PROFILE_LEGACY.

    Left out of the result, by design: `verifier`, `coverage`, `coverage_sha256`, `certifies`,
    `digest`, `timestamp`. The caller keeps `verifier` beside the core with receipt_v0().

    Checked before anything is rewritten: a committed receipt's own digest re-derives
    (source_digest); counts and rungs agree with the rows (counts_mismatch); commit and
    manifest_digest have their shapes (not_a_core); HELD and FAILED rows carry provenance
    (row_shape); the document name is Unicode text (name_form); rN rows read from a manifest whose
    digest is null are refused (manifest_not_canonical), because a v0 checker answers them
    UNRESOLVED manifest_integrity. Nothing else is re-derived: a projected core is what today's
    checker said, rewritten; only a v0 checker's own run re-derives one.
    """
    if not isinstance(core, dict):
        raise Refusal("not_a_core", "not an object")
    if core.get("schema") not in SOURCE_SCHEMAS:
        raise Refusal("schema_out_of_scope", "%r" % (core.get("schema"),))
    if core.get("format") != FORMAT:
        raise Refusal("format_unknown", "%r" % (core.get("format"),))
    if "digest" in core:
        # a committed /v1 or /v2 receipt: its own digest must re-derive before its rows are trusted
        body = {k: v for k, v in core.items() if k not in _SOURCE_OUTSIDE_DIGEST}
        try:
            again = _sha256(jcs(body).encode("utf-8"))
        except Refusal as e:
            raise Refusal("source_digest", "the receipt has no canonical form (%s)" % e.code)
        if again != core["digest"]:
            raise Refusal("source_digest", "the receipt's own digest does not re-derive")
    doc = core.get("document")
    if not (isinstance(doc, dict) and isinstance(doc.get("name"), str)
            and _is_hex64(doc.get("inline_sha256"))
            and (doc.get("canonical_sha256") is None or _is_hex64(doc.get("canonical_sha256")))):
        raise Refusal("not_a_core", "document block")
    if any(0xD800 <= ord(c) <= 0xDFFF for c in doc["name"]):
        raise Refusal("name_form", "the document name holds an unpaired surrogate")
    commit = core.get("commit")
    if not (commit is None or _is_hex64(commit) or _is_hex40(commit)):
        raise Refusal("not_a_core", "commit is neither null nor 40 or 64 lowercase hex digits")
    manifest_digest = core.get("manifest_digest")
    if not (manifest_digest is None or _is_hex64(manifest_digest)):
        raise Refusal("not_a_core", "manifest_digest is neither null nor 64 lowercase hex digits")
    if document is not None:
        if not isinstance(document, (bytes, bytearray)):
            raise Refusal("type", "document must be bytes")
        document = bytes(document)
        if _sha256(document) != doc["inline_sha256"]:
            raise Refusal("document_mismatch", "the bytes handed over are not the document named")
    spans_in = core.get("spans")
    counts_in = core.get("counts")
    if not isinstance(spans_in, list) or not isinstance(counts_in, dict):
        raise Refusal("not_a_core", "spans or counts")
    # the input must be consistent with itself before it is rewritten
    seen = {}
    for r in spans_in:
        v = r.get("verdict") if isinstance(r, dict) else None
        seen[v] = seen.get(v, 0) + 1
    for word in set(seen) | set(counts_in):
        if seen.get(word, 0) != counts_in.get(word, 0):
            raise Refusal("counts_mismatch", "%r: rows %r, counts %r"
                          % (word, seen.get(word, 0), counts_in.get(word, 0)))
    rows = [_row_v0(r, document) for r in spans_in]
    counts = {w: 0 for w in SPAN_VERDICTS}
    for r in rows:
        counts[r["verdict"]] += 1
    if _manifest_seen_without_digest(rows, manifest_digest):
        raise Refusal("manifest_not_canonical", "rows were read from a manifest whose digest is null")
    rungs = _rungs(rows)
    if core.get("rungs") is not None:
        jcs(core["rungs"])                    # a float or an odd key is refused, never compared
        if core["rungs"] != rungs:
            raise Refusal("counts_mismatch", "rungs %r, rows give %r" % (core["rungs"], rungs))
    document_malformed = core.get("document_malformed")
    if document_malformed is not None and not (
            isinstance(document_malformed, dict)
            and document_malformed.get("reason") in DOCUMENT_REASONS):
        raise Refusal("not_a_core", "document_malformed")
    out = {
        "schema": SCHEMA_V0,
        "profile": PROFILE_LEGACY if _is_legacy(rows) else PROFILE_RN,
        "format": FORMAT,
        "document": {"name": doc["name"], "inline_sha256": doc["inline_sha256"],
                     "canonical_sha256": doc.get("canonical_sha256")},
        "commit": commit,
        "manifest_digest": manifest_digest,
        "spans": rows,
        "counts": counts,
        "sworn_total": len(rows),
        "unresolved": counts["UNRESOLVED"],
        "document_verdict": _document_verdict(document_malformed, counts, len(rows)),
        "document_malformed": document_malformed,
        "rungs": rungs,
    }
    jcs(out)                 # a core that has no canonical form is refused here, not later
    return out
