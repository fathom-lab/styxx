# -*- coding: utf-8 -*-
"""RECEIPT v0 (DRAFT, unsigned, nothing frozen): a checker written from receipt/v0/RECEIPT_v0.md.

LAB-WRITTEN, NOT INDEPENDENT. Fathom Lab wrote this file, from a text the lab wrote by reading the
lab's own checker (styxx/sworn.py). It imports nothing from styxx, and its first draft was written
from the text alone; where the text left a choice open the writer then looked at the code. The
places where the text was silent are now settled in the text itself (5.5 and 11.3 on refusals,
7.2 on the Unicode version). Agreement between this file and the lab's other code is
agreement inside one lab. It is not a standard, nobody outside the lab emits or checks these
objects, and a HELD verdict never means a sentence is true (RECEIPT_v0.md sections 12 and 14).

One file, Python standard library only. The public functions are pure: no clock, no network, no
subprocess, no file system, no float in any verdict path, no regular expression at all.

  check(document, name, manifest=None, commit=None) -> verdict core  (sections 2 to 9, 11.1)
  receipt(core, verifier=None)                       -> the verdict receipt (9.4)
  recheck(receipt_bytes, document, manifest=None)    -> "REDERIVES" | "DOES-NOT-REDERIVE"  (11.3)
  jcs(value), digest(core)                           -> the canonical form (section 3), its sha256
  parse_json(data)                                   -> 6.13's strict reading

`document`, `manifest` and `receipt_bytes` are bytes. A refusal (section 11) raises Refusal with a
code from REFUSALS; nothing is emitted.

Command line (the only code here that touches a file, and it only reads its arguments):

  python check_v0.py check DOC [--manifest FILE] [--name NAME] [--commit HEX]
  python check_v0.py recheck RECEIPT DOC [--manifest FILE]
"""
import binascii
import hashlib

__all__ = ["check", "receipt", "recheck", "jcs", "digest", "parse_json", "read_manifest",
           "Refusal", "JSONError", "JNum", "JConst", "JObj", "REFUSALS", "REASONS",
           "SCHEMA_V0", "PROFILE_RN", "PROFILE_LEGACY", "FORMAT", "MANIFEST_SPEC"]

SCHEMA_V0 = "receipt/v0-draft"
PROFILE_RN = "rn-inline/v0-draft"
PROFILE_LEGACY = "legacy-projection/v0-draft"
FORMAT = "sworn/0.1"
MANIFEST_SPEC = "sworn/manifest/0.2"
KINDS = ("numeric", "quote", "hash", "absent")
EXTERNAL = ("tool_stdout", "tool_stderr", "file_read", "http_fetch", "harness_note",
            "test_report", "attestation")
AUTHOR_SIDE = ("agent_output", "agent_file_write", "agent_message")
SPAN_VERDICTS = ("HELD", "FAILED", "UNRESOLVED", "MALFORMED")
DOCUMENT_VERDICTS = ("SWORN-HELD", "SWORN-FAILED", "SWORN-UNCHECKED", "UNSWORN")
CAP = 300
MIN_NEEDLE = 16
INT_MAX = 2 ** 53 - 1
DEPTH_MAX = 32
ADJUSTED_MAX = 320

# section 10, the closed lists
MALFORMED_DOCUMENT = ("tag_syntax", "nesting", "stray_closer", "unclosed", "empty_span",
                      "hidden_commitment", "length_cap", "kind_unknown", "kind_reserved",
                      "receipt_form", "absent_over_partial", "hash_over_partial",
                      "directional_override", "number_count", "number_grammar", "needle_count",
                      "needle_empty", "digest_form")
MALFORMED_EVIDENCE = ("kind_of_source_unknown", "receipt_author_minted", "anchor_out_of_range",
                      "receipt_not_json", "pointer_unresolvable", "pointer_ambiguous",
                      "leaf_not_scalar", "leaf_not_numeric", "leaf_not_string", "short_needle")
UNRESOLVED_REASONS = ("manifest_absent", "manifest_integrity", "rung_unknown",
                      "manifest_id_missing", "manifest_bytes_absent", "manifest_no_completeness",
                      "no_repository")
FAILED_REASONS = ("value_mismatch", "rounds_to_zero", "needle_missing", "needle_present",
                  "digest_mismatch")
REASONS = MALFORMED_DOCUMENT + MALFORMED_EVIDENCE + UNRESOLVED_REASONS + FAILED_REASONS
LEGACY_REASONS = ("no_commit", "commit_absent", "path_absent", "not_a_blob", "receipt_too_large",
                  "git_unavailable", "prereg_not_in_tree")
DOCUMENT_REASONS = ("invalid_utf8", "unbalanced_fences")
REFUSALS = ("nothing_to_verify", "name_form", "commit_form", "manifest_spec",
            "manifest_receipts_shape", "manifest_authored_shape", "manifest_form",
            "float", "int_range", "non_ascii_key", "lone_surrogate", "depth", "type",
            "receipt_shape", "receipt_digest", "attestations_not_empty", "profile_not_checkable",
            "document_mismatch")
REDERIVES = "REDERIVES"
DOES_NOT_REDERIVE = "DOES-NOT-REDERIVE"

_HEX_LOWER = "0123456789abcdef"
_HEX_ANY = "0123456789abcdefABCDEF"
_DIGITS = "0123456789"
_ASCII_WS = b" \t\n\x0b\x0c\r"
_WORD_ASCII = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
_CANDIDATE_TAIL = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"


class Refusal(ValueError):
    """Section 11: not a verdict. `code` is one of REFUSALS."""

    def __init__(self, code, message=""):
        ValueError.__init__(self, "%s: %s" % (code, message) if message else code)
        self.code = code


class JSONError(ValueError):
    """The bytes are not JSON under 6.13."""


def _sha256(b):
    return hashlib.sha256(b).hexdigest()


def _is_hex64_lower(x):
    return isinstance(x, str) and len(x) == 64 and all(c in _HEX_LOWER for c in x)


def _is_hex40_lower(x):
    return isinstance(x, str) and len(x) == 40 and all(c in _HEX_LOWER for c in x)


def _has_surrogate(s):
    return any(0xD800 <= ord(c) <= 0xDFFF for c in s)


# ---------------------------------------------------------------------------------------------
# 6.13: JSON, read strictly. Numbers keep their digits; NaN, Infinity and -Infinity parse and are
# never numeric; a repeated key keeps its last value and is remembered as repeated.
# ---------------------------------------------------------------------------------------------

class JNum(object):
    """A JSON number exactly as written. Never converted to binary floating point."""
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text

    def is_integer_token(self):
        return not any(c in self.text for c in ".eE")

    def __repr__(self):
        return "JNum(%s)" % self.text


class JConst(object):
    """One of the bare tokens NaN, Infinity, -Infinity."""
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "JConst(%s)" % self.name


class JObj(dict):
    """A JSON object: the last value of each key, and the set of keys written more than once."""

    def __init__(self):
        dict.__init__(self)
        self.dup = set()


_JSON_WS = " \t\n\r"
_CTRL = dict.fromkeys(range(0x20))
_ESC = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _ws(s, i):
    n = len(s)
    while i < n and s[i] in _JSON_WS:
        i += 1
    return i


def _hex4(s, i):
    h = s[i:i + 4]
    if len(h) != 4 or not all(c in _HEX_ANY for c in h):
        raise JSONError("bad \\u escape at %d" % i)
    return int(h, 16)


def _string(s, i):
    """s[i] is '"'. Returns (text, index after the closing quote)."""
    i += 1
    parts = []
    n = len(s)
    while True:
        q = s.find('"', i)
        if q < 0:
            raise JSONError("unterminated string")
        bs = s.find("\\", i, q)
        stop = q if bs < 0 else bs
        chunk = s[i:stop]
        if len(chunk.translate(_CTRL)) != len(chunk):
            raise JSONError("control character in a string")
        parts.append(chunk)
        if bs < 0:
            return "".join(parts), q + 1
        i = bs + 1
        if i >= n:
            raise JSONError("unterminated escape")
        e = s[i]
        if e in _ESC:
            parts.append(_ESC[e])
            i += 1
        elif e == "u":
            cp = _hex4(s, i + 1)
            i += 5
            if 0xD800 <= cp <= 0xDBFF and s[i:i + 2] == "\\u":
                lo = _hex4(s, i + 2)
                if 0xDC00 <= lo <= 0xDFFF:
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    i += 6
            parts.append(chr(cp))
        else:
            raise JSONError("bad escape")


def _number(s, i):
    n = len(s)
    j = i
    if j < n and s[j] == "-":
        j += 1
    if j < n and s[j] == "0":
        j += 1
    elif j < n and "1" <= s[j] <= "9":
        while j < n and "0" <= s[j] <= "9":
            j += 1
    else:
        raise JSONError("bad number at %d" % i)
    if j < n and s[j] == ".":
        j += 1
        k = j
        while j < n and "0" <= s[j] <= "9":
            j += 1
        if j == k:
            raise JSONError("bad fraction at %d" % i)
    if j < n and s[j] in "eE":
        j += 1
        if j < n and s[j] in "+-":
            j += 1
        k = j
        while j < n and "0" <= s[j] <= "9":
            j += 1
        if j == k:
            raise JSONError("bad exponent at %d" % i)
    return JNum(s[i:j]), j


_LITERALS = (("true", True), ("false", False), ("null", None))


def _parse_text(s):
    """Iterative: no recursion, so no host recursion limit decides anything."""
    n = len(s)
    stack = []          # [container, pending key or None]
    i = _ws(s, 0)
    while True:
        # ---- a value starts at i
        if i >= n:
            raise JSONError("value expected at end")
        c = s[i]
        val = None
        complete = True
        if c == "{":
            i = _ws(s, i + 1)
            obj = JObj()
            if i < n and s[i] == "}":
                val, i = obj, i + 1
            else:
                if i >= n or s[i] != '"':
                    raise JSONError("key expected at %d" % i)
                key, i = _string(s, i)
                i = _ws(s, i)
                if i >= n or s[i] != ":":
                    raise JSONError("colon expected at %d" % i)
                i = _ws(s, i + 1)
                stack.append([obj, key])
                complete = False
        elif c == "[":
            i = _ws(s, i + 1)
            arr = []
            if i < n and s[i] == "]":
                val, i = arr, i + 1
            else:
                stack.append([arr, None])
                complete = False
        elif c == '"':
            val, i = _string(s, i)
        elif s.startswith("-Infinity", i):
            val, i = JConst("-Infinity"), i + 9
        elif c == "-" or "0" <= c <= "9":
            val, i = _number(s, i)
        elif s.startswith("NaN", i):
            val, i = JConst("NaN"), i + 3
        elif s.startswith("Infinity", i):
            val, i = JConst("Infinity"), i + 8
        else:
            for word, v in _LITERALS:
                if s.startswith(word, i):
                    val, i = v, i + len(word)
                    break
            else:
                raise JSONError("unexpected %r at %d" % (c, i))
        if not complete:
            continue
        # ---- a value is complete: attach it, then close as many containers as the text closes
        while True:
            if not stack:
                i = _ws(s, i)
                if i != n:
                    raise JSONError("trailing bytes at %d" % i)
                return val
            frame = stack[-1]
            cont = frame[0]
            if isinstance(cont, JObj):
                key = frame[1]
                if key in cont:
                    cont.dup.add(key)
                cont[key] = val
            else:
                cont.append(val)
            i = _ws(s, i)
            if i >= n:
                raise JSONError("unterminated container")
            c = s[i]
            if c == ",":
                i = _ws(s, i + 1)
                if isinstance(cont, JObj):
                    if i >= n or s[i] != '"':
                        raise JSONError("key expected at %d" % i)
                    key, i = _string(s, i)
                    i = _ws(s, i)
                    if i >= n or s[i] != ":":
                        raise JSONError("colon expected at %d" % i)
                    i = _ws(s, i + 1)
                    frame[1] = key
                break
            if c == ("}" if isinstance(cont, JObj) else "]"):
                stack.pop()
                val, i = cont, i + 1
                continue
            raise JSONError("unexpected %r at %d" % (c, i))


def parse_json(data):
    """6.13 over bytes: UTF-8 without a byte order mark, RFC 8259 plus NaN, Infinity, -Infinity."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("bytes expected")
    try:
        s = bytes(data).decode("utf-8")
    except UnicodeDecodeError:
        raise JSONError("not UTF-8")
    if s.startswith("\ufeff"):
        # redundant with the parse (U+FEFF is neither JSON whitespace nor a value's first
        # character); kept as stated intent, so a mutant of this line cannot be caught
        raise JSONError("byte order mark")
    return _parse_text(s)


def _has_dup(v):
    """True when any object anywhere in the value had a key written twice."""
    todo = [v]
    while todo:
        x = todo.pop()
        if isinstance(x, JObj):
            if x.dup:
                return True
            todo.extend(x.values())
        elif isinstance(x, list):
            todo.extend(x)
    return False


# ---------------------------------------------------------------------------------------------
# Section 3: the canonical form
# ---------------------------------------------------------------------------------------------

_SHORT = {0x08: "\\b", 0x09: "\\t", 0x0A: "\\n", 0x0C: "\\f", 0x0D: "\\r", 0x22: '\\"',
          0x5C: "\\\\"}


def _jcs_string(s, out):
    out.append('"')
    for ch in s:
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            raise Refusal("lone_surrogate", "U+%04X" % cp)
        if cp in _SHORT:
            out.append(_SHORT[cp])
        elif cp < 0x20:
            out.append("\\u00" + _HEX_LOWER[cp >> 4] + _HEX_LOWER[cp & 15])
        else:
            out.append(ch)
    out.append('"')


def _jcs_int(n, out):
    if n > INT_MAX or n < -INT_MAX:
        raise Refusal("int_range")
    out.append(str(n))


def _jcs(v, depth, out):
    if v is None:
        out.append("null")
    elif v is True:
        out.append("true")
    elif v is False:
        out.append("false")
    elif isinstance(v, int):
        _jcs_int(v, out)
    elif isinstance(v, JNum):
        # decided on the number as written (3.1): a fraction or an exponent is never admitted
        if not v.is_integer_token():
            raise Refusal("float", v.text[:40])
        digits = v.text.lstrip("-")
        if len(digits) > 16:
            raise Refusal("int_range", v.text[:40])
        _jcs_int(int(v.text), out)
    elif isinstance(v, JConst):
        raise Refusal("float", v.name)
    elif isinstance(v, str):
        _jcs_string(v, out)
    elif isinstance(v, (list, tuple)):
        if depth >= DEPTH_MAX:
            raise Refusal("depth")
        out.append("[")
        for k, x in enumerate(v):
            if k:
                out.append(",")
            _jcs(x, depth + 1, out)
        out.append("]")
    elif isinstance(v, dict):
        if depth >= DEPTH_MAX:
            raise Refusal("depth")
        for k in v:
            if not isinstance(k, str):
                raise Refusal("type", "object key")
            if any(ord(c) > 0x7E for c in k):
                raise Refusal("non_ascii_key")
        out.append("{")
        for n, k in enumerate(sorted(v)):
            if n:
                out.append(",")
            _jcs_string(k, out)
            out.append(":")
            _jcs(v[k], depth + 1, out)
        out.append("}")
    else:
        raise Refusal("type", type(v).__name__)


def jcs(value):
    """The canonical text of `value` (section 3). Refuses with 3.3's code, never approximates."""
    out = []
    _jcs(value, 0, out)
    return "".join(out)


def digest(value):
    """3.4: sha256 over the UTF-8 bytes of the canonical text."""
    return _sha256(jcs(value).encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# 6.10: strict base64
# ---------------------------------------------------------------------------------------------

_B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_B64_DELETE = dict.fromkeys(map(ord, _B64_ALPHABET))


def _b64_strict(s):
    """bytes, or None when `s` is not strict base64 (RFC 4648 section 4 alphabet, length a
    multiple of 4, '=' only as one or two final padding characters). Pad bits are ignored."""
    if not isinstance(s, str) or len(s) % 4:
        return None
    pad = 0
    if s.endswith("=="):
        pad = 2
    elif s.endswith("="):
        pad = 1
    body = s[:len(s) - pad]
    if body.translate(_B64_DELETE):
        return None                     # a character outside the alphabet, or '=' inside
    if pad and len(body) % 4 == 0:
        return None                     # unreachable after the length test; kept as stated intent
    try:
        return binascii.a2b_base64(body.encode("ascii") + b"=" * pad)
    except (binascii.Error, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# Section 2: the span grammar
# ---------------------------------------------------------------------------------------------

def _fences(b):
    """2.2: (regions [(start, end_exclusive)], delimiter line numbers)."""
    delims = []
    pos, line_no, n = 0, 1, len(b)
    while True:
        nl = b.find(b"\n", pos)
        end = n if nl < 0 else nl
        i = pos
        while i < end and i - pos < 3 and b[i] == 0x20:
            i += 1
        if i + 3 <= end and b[i:i + 3] == b"```":
            delims.append((line_no, pos, n if nl < 0 else nl + 1))
        if nl < 0:
            break
        pos, line_no = nl + 1, line_no + 1
    regions = [(delims[k][1], delims[k + 1][2]) for k in range(0, len(delims) - 1, 2)]
    return regions, [d[0] for d in delims]


def _backtick_runs(b):
    """Every maximal backtick run: start -> (length, start of the next run on the same line with
    the same length, or -1)."""
    runs = []
    n = len(b)
    p = b.find(b"`")
    while p >= 0:
        q = p
        while q < n and b[q] == 0x60:
            q += 1
        runs.append((p, q - p))
        p = b.find(b"`", q)
    info = {}
    later = {}          # (line start, length) -> start of the nearest later run
    line_of = {}
    line_start, lf = 0, b.find(b"\n")
    for start, _ in runs:
        while 0 <= lf < start:
            line_start, lf = lf + 1, b.find(b"\n", lf + 1)
        line_of[start] = line_start
    for start, length in reversed(runs):
        key = (line_of[start], length)
        info[start] = (length, later.get(key, -1))
        later[key] = start
    return info


class _Decl(object):
    __slots__ = ("at", "receipt", "kind", "mark", "raw", "inner_start", "inner_end", "closer_at")

    def __init__(self, at, receipt, kind, mark=None, raw=None):
        self.at, self.receipt, self.kind, self.mark, self.raw = at, receipt, kind, mark, raw
        self.inner_start = self.inner_end = self.closer_at = None


def _attr(b, i):
    """End index of an attribute value starting at i (zero or more bytes other than '"', '<',
    '>', CR, LF), at the '"' that ends it; -1 when a forbidden byte or the end comes first."""
    n = len(b)
    while i < n:
        c = b[i]
        if c == 0x22:
            return i
        if c in (0x3C, 0x3E, 0x0D, 0x0A):
            return -1
        i += 1
    return -1


def _opener(b, p):
    """(REF, KIND, end) when an opener starts at p, else None."""
    if not b.startswith(b'<sworn r="', p):
        return None
    r0 = p + 10
    r1 = _attr(b, r0)
    if r1 < 0 or not b.startswith(b'" k="', r1):
        return None
    k0 = r1 + 5
    k1 = _attr(b, k0)
    if k1 < 0 or not b.startswith(b'">', k1):
        return None
    return b[r0:r1].decode("utf-8"), b[k0:k1].decode("utf-8"), k1 + 2


def _candidate(b, p):
    q = p + 1
    if q < len(b) and b[q] == 0x2F:
        q += 1
    if b[q:q + 5].lower() != b"sworn":
        return False
    return q + 5 == len(b) or b[q + 5] not in _CANDIDATE_TAIL


def _scan(b):
    """2.3 and 2.4 over a well-formed UTF-8 document with balanced fences: the declarations."""
    regions, _ = _fences(b)
    runs = _backtick_runs(b)
    n = len(b)
    decls, stack, unmarked_open = [], [], []
    p, c, ri = 0, 0, 0
    next_lt = next_bt = -2
    while p < n:
        # a. inside a fenced region
        while ri < len(regions) and regions[ri][1] <= p:
            ri += 1
        if ri < len(regions) and regions[ri][0] <= p:
            p = regions[ri][1]
            continue
        byte = b[p]
        # b. a comment opener at or after the comment end
        if byte == 0x3C and p >= c and b.startswith(b"<!--", p):
            e = b.find(b"-->", p + 4)
            c = n if e < 0 else e + 3
            p += 4
            continue
        # c. a backtick run
        if byte == 0x60:
            if p in runs:
                length, nxt = runs[p]
            else:                                   # not reached: p is always at a run start
                q = p
                while q < n and b[q] == 0x60:
                    q += 1
                length, nxt = q - p, -1
                lf = b.find(b"\n", q)
                r = q
                end = n if lf < 0 else lf
                while r < end:
                    if b[r] == 0x60:
                        s0 = r
                        while r < end and b[r] == 0x60:
                            r += 1
                        if r - s0 == length:
                            nxt = s0
                            break
                    else:
                        r += 1
            p = nxt + length if nxt >= 0 else p + length
            continue
        # d. a candidate
        if byte == 0x3C and _candidate(b, p):
            op = _opener(b, p)
            if op is not None:
                ref, kind, end = op
                d = _Decl(p, ref, kind)
                if stack:
                    for o in unmarked_open:
                        o.mark = "nesting"
                    unmarked_open = []
                    d.mark = "nesting"
                elif p < c:
                    d.mark = "hidden_commitment"
                else:
                    unmarked_open.append(d)
                d.inner_start = end
                decls.append(d)
                stack.append(d)
                p = end
                continue
            if b.startswith(b"</sworn>", p):
                if stack:
                    d = stack.pop()
                    if unmarked_open and unmarked_open[-1] is d:
                        unmarked_open.pop()
                    d.inner_end = p
                    d.closer_at = p
                    if p == d.inner_start and d.mark in (None, "hidden_commitment"):
                        d.mark = "empty_span"
                else:
                    decls.append(_Decl(p, None, None, "stray_closer", "</sworn>"))
                p += 8
                continue
            lf = b.find(b"\n", p)
            lim = n if lf < 0 else lf
            gt = b.find(b">", p, lim)
            end = gt + 1 if gt >= 0 else lim
            decls.append(_Decl(p, None, None, "tag_syntax", b[p:end].decode("utf-8")))
            p = end
            continue
        # e. advance: jump to the next byte at which a, b, c or d can apply
        if next_lt < p:
            next_lt = b.find(b"<", p + 1)
            if next_lt < 0:
                next_lt = n
        if next_bt < p:
            next_bt = b.find(b"`", p + 1)
            if next_bt < 0:
                next_bt = n
        nxt = min(next_lt, next_bt)
        if ri < len(regions):
            nxt = min(nxt, regions[ri][0])
        p = max(nxt, p + 1)
    for d in stack:
        if d.mark in (None, "hidden_commitment"):
            d.mark = "unclosed"
    decls.sort(key=lambda d: d.at)
    return decls


_LEXICAL = ("tag_syntax", "nesting", "stray_closer", "unclosed")


def _canonical(b, decls):
    """2.5: (canonical bytes or None, {id(decl): (start, end)})."""
    if any(d.mark in _LEXICAL for d in decls):
        return None, {}
    parts, offsets, last, removed = [], {}, 0, 0
    for d in decls:
        opener_len = d.inner_start - d.at
        parts.append(b[last:d.at])
        parts.append(b[d.inner_start:d.inner_end])
        start = d.at - removed
        offsets[id(d)] = (start, start + (d.inner_end - d.inner_start))
        removed += opener_len + 8
        last = d.closer_at + 8
    parts.append(b[last:])
    return b"".join(parts), offsets


# ---------------------------------------------------------------------------------------------
# Section 4: REF
# ---------------------------------------------------------------------------------------------

def _pos_int(s):
    return bool(s) and s[0] in "123456789" and all(ch in _DIGITS for ch in s)


_PATH_BANNED = set("\\*?[]") | set(chr(x) for x in range(0x21)) | {
    "\u007f", "\u0085", "\u00a0", "\u1680", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000"} | set(
    chr(x) for x in range(0x2000, 0x200B))


def _path_target_ok(t):
    if not t or t[0] in "/:":
        return False
    if any(ch in _PATH_BANNED for ch in t):
        return False
    return all(seg not in ("", ".", "..") for seg in t.split("/"))


def _parse_ref(ref):
    """(form, head, fragment) with fragment None, ("lines", a, b) or ("pointer", tokens); None
    when the REF is receipt_form. a and b are digit strings."""
    head, sep, frag = ref.partition("#")
    if sep and frag == "":
        return None
    if len(head) >= 2 and head[0] == "r" and _pos_int(head[1:]):
        form = "rn"
    elif head.startswith("path:"):
        if not _path_target_ok(head[5:]):
            return None
        form = "path"
    elif head.startswith("prereg:"):
        h = head[7:]
        if not (len(h) == 64 and all(ch in _HEX_ANY for ch in h)):
            return None
        form = "prereg"
    else:
        return None
    fragment = None
    if sep:
        if frag[0] == "/":
            k = 0
            while True:
                k = frag.find("~", k)
                if k < 0:
                    break
                if frag[k + 1:k + 2] not in ("0", "1"):
                    return None
                k += 2
            tokens = [t.replace("~1", "/").replace("~0", "~") for t in frag[1:].split("/")]
            fragment = ("pointer", tokens)
        else:
            if frag[0] != "L":
                return None
            a, dash, rest = frag[1:].partition("-")
            if not _pos_int(a):
                return None
            if dash:
                if rest[:1] != "L" or not _pos_int(rest[1:]):
                    return None
                bb = rest[1:]
                if _digits_lt(bb, a):
                    return None
            else:
                bb = a
            fragment = ("lines", a, bb)
    return form, head, fragment


def _digits_lt(x, y):
    """x < y for digit strings without leading zeros."""
    return (len(x), x) < (len(y), y)


# ---------------------------------------------------------------------------------------------
# Section 5: the manifest
# ---------------------------------------------------------------------------------------------

class _Manifest(object):
    __slots__ = ("intact", "digest", "harness", "rung", "receipts", "authored")


_UPPER_HEX = {ord(u): ord(l) for u, l in zip("ABCDEF", "abcdef")}


def read_manifest(data):
    """5.5 and 5.6 over the manifest's bytes. Refuses (11.1) or returns what section 6 reads."""
    try:
        m = parse_json(data)
    except JSONError:
        # 5.5: manifest bytes that do not parse under 6.13 are refused as manifest_spec.
        raise Refusal("manifest_spec", "the manifest bytes are not JSON under 6.13")
    if _has_dup(m):
        # 5.5: a key written twice anywhere refuses the manifest before any field is read.
        raise Refusal("manifest_form", "a key written twice")
    if not isinstance(m, JObj) or m.get("spec") != MANIFEST_SPEC or not isinstance(m.get("spec"), str):
        raise Refusal("manifest_spec")
    receipts = m.get("receipts")
    if receipts is None:
        receipts = JObj()
    elif not isinstance(receipts, JObj):
        raise Refusal("manifest_receipts_shape")
    authored = m.get("authored_sha256")
    if authored is None:
        authored = []
    elif not (isinstance(authored, list) and all(
            isinstance(a, str) and len(a) == 64 and all(ch in _HEX_ANY for ch in a)
            for a in authored)):
        raise Refusal("manifest_authored_shape")
    for key in ("harness", "turn", "minted_at"):
        if not isinstance(m.get(key), str):
            raise Refusal("manifest_form", "%s missing or not a string" % key)
    rebuilt = sorted(a.translate(_UPPER_HEX) for a in authored)
    rung = m.get("rung")
    core = {"spec": m["spec"], "harness": m["harness"], "turn": m["turn"],
            "minted_at": m["minted_at"], "authored_sha256": rebuilt, "receipts": receipts,
            "rung": rung}
    out = _Manifest()
    try:
        out.digest = digest(core)
    except Refusal:
        out.digest = None
    declared = m.get("digest")
    out.intact = out.digest is not None and (declared is None or (
        isinstance(declared, str) and declared == out.digest))
    out.harness = m["harness"]
    out.rung = rung
    out.receipts = receipts
    out.authored = set(rebuilt)
    return out


# ---------------------------------------------------------------------------------------------
# Section 7: the kinds
# ---------------------------------------------------------------------------------------------

_DASHES = set(chr(x) for x in (
    0x00AD, 0x058A, 0x05BE, 0x1400, 0x1806, 0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015,
    0x2E17, 0x2E1A, 0x2E3A, 0x2E3B, 0x2E40, 0x2E5D, 0x301C, 0x3030, 0x30A0, 0xFE31, 0xFE32,
    0xFE58, 0xFE63, 0xFF0D, 0x10EAD))
_TOKEN_EXTRA = set("_.,+-%/:") | {"\u2212", "\u00b1"} | _DASHES


# 7.2's classes, pinned to Unicode 15.0.0 and carried here as data, so that no host's Unicode
# tables decide a verdict: General_Category L* or N* (LN) and Nd (ND), as space-separated
# ranges of code points in upper-case hex (first-last, or one code point alone). Generated from
# Python 3.12.10's unicodedata (unidata_version 15.0.0); tests/test_receipt_v0_checkers.py
# re-derives both tables from unicodedata when it is at 15.0.0, compares them with check_v0.js's
# copies, and checks the sha256 values RECEIPT_v0.md 7.2 states.
_UNICODE_15_0_0_LN = (
    "30-39 41-5A 61-7A AA B2-B3 B5 B9-BA BC-BE C0-D6 D8-F6 F8-2C1 2C6-2D1 2E0-2E4 2EC 2EE "
    "370-374 376-377 37A-37D 37F 386 388-38A 38C 38E-3A1 3A3-3F5 3F7-481 48A-52F 531-556 559 "
    "560-588 5D0-5EA 5EF-5F2 620-64A 660-669 66E-66F 671-6D3 6D5 6E5-6E6 6EE-6FC 6FF 710 "
    "712-72F 74D-7A5 7B1 7C0-7EA 7F4-7F5 7FA 800-815 81A 824 828 840-858 860-86A 870-887 "
    "889-88E 8A0-8C9 904-939 93D 950 958-961 966-96F 971-980 985-98C 98F-990 993-9A8 9AA-9B0 "
    "9B2 9B6-9B9 9BD 9CE 9DC-9DD 9DF-9E1 9E6-9F1 9F4-9F9 9FC A05-A0A A0F-A10 A13-A28 A2A-A30 "
    "A32-A33 A35-A36 A38-A39 A59-A5C A5E A66-A6F A72-A74 A85-A8D A8F-A91 A93-AA8 AAA-AB0 "
    "AB2-AB3 AB5-AB9 ABD AD0 AE0-AE1 AE6-AEF AF9 B05-B0C B0F-B10 B13-B28 B2A-B30 B32-B33 "
    "B35-B39 B3D B5C-B5D B5F-B61 B66-B6F B71-B77 B83 B85-B8A B8E-B90 B92-B95 B99-B9A B9C "
    "B9E-B9F BA3-BA4 BA8-BAA BAE-BB9 BD0 BE6-BF2 C05-C0C C0E-C10 C12-C28 C2A-C39 C3D C58-C5A "
    "C5D C60-C61 C66-C6F C78-C7E C80 C85-C8C C8E-C90 C92-CA8 CAA-CB3 CB5-CB9 CBD CDD-CDE "
    "CE0-CE1 CE6-CEF CF1-CF2 D04-D0C D0E-D10 D12-D3A D3D D4E D54-D56 D58-D61 D66-D78 D7A-D7F "
    "D85-D96 D9A-DB1 DB3-DBB DBD DC0-DC6 DE6-DEF E01-E30 E32-E33 E40-E46 E50-E59 E81-E82 E84 "
    "E86-E8A E8C-EA3 EA5 EA7-EB0 EB2-EB3 EBD EC0-EC4 EC6 ED0-ED9 EDC-EDF F00 F20-F33 F40-F47 "
    "F49-F6C F88-F8C 1000-102A 103F-1049 1050-1055 105A-105D 1061 1065-1066 106E-1070 "
    "1075-1081 108E 1090-1099 10A0-10C5 10C7 10CD 10D0-10FA 10FC-1248 124A-124D 1250-1256 "
    "1258 125A-125D 1260-1288 128A-128D 1290-12B0 12B2-12B5 12B8-12BE 12C0 12C2-12C5 "
    "12C8-12D6 12D8-1310 1312-1315 1318-135A 1369-137C 1380-138F 13A0-13F5 13F8-13FD "
    "1401-166C 166F-167F 1681-169A 16A0-16EA 16EE-16F8 1700-1711 171F-1731 1740-1751 "
    "1760-176C 176E-1770 1780-17B3 17D7 17DC 17E0-17E9 17F0-17F9 1810-1819 1820-1878 "
    "1880-1884 1887-18A8 18AA 18B0-18F5 1900-191E 1946-196D 1970-1974 1980-19AB 19B0-19C9 "
    "19D0-19DA 1A00-1A16 1A20-1A54 1A80-1A89 1A90-1A99 1AA7 1B05-1B33 1B45-1B4C 1B50-1B59 "
    "1B83-1BA0 1BAE-1BE5 1C00-1C23 1C40-1C49 1C4D-1C7D 1C80-1C88 1C90-1CBA 1CBD-1CBF "
    "1CE9-1CEC 1CEE-1CF3 1CF5-1CF6 1CFA 1D00-1DBF 1E00-1F15 1F18-1F1D 1F20-1F45 1F48-1F4D "
    "1F50-1F57 1F59 1F5B 1F5D 1F5F-1F7D 1F80-1FB4 1FB6-1FBC 1FBE 1FC2-1FC4 1FC6-1FCC "
    "1FD0-1FD3 1FD6-1FDB 1FE0-1FEC 1FF2-1FF4 1FF6-1FFC 2070-2071 2074-2079 207F-2089 "
    "2090-209C 2102 2107 210A-2113 2115 2119-211D 2124 2126 2128 212A-212D 212F-2139 "
    "213C-213F 2145-2149 214E 2150-2189 2460-249B 24EA-24FF 2776-2793 2C00-2CE4 2CEB-2CEE "
    "2CF2-2CF3 2CFD 2D00-2D25 2D27 2D2D 2D30-2D67 2D6F 2D80-2D96 2DA0-2DA6 2DA8-2DAE "
    "2DB0-2DB6 2DB8-2DBE 2DC0-2DC6 2DC8-2DCE 2DD0-2DD6 2DD8-2DDE 2E2F 3005-3007 3021-3029 "
    "3031-3035 3038-303C 3041-3096 309D-309F 30A1-30FA 30FC-30FF 3105-312F 3131-318E "
    "3192-3195 31A0-31BF 31F0-31FF 3220-3229 3248-324F 3251-325F 3280-3289 32B1-32BF "
    "3400-4DBF 4E00-A48C A4D0-A4FD A500-A60C A610-A62B A640-A66E A67F-A69D A6A0-A6EF "
    "A717-A71F A722-A788 A78B-A7CA A7D0-A7D1 A7D3 A7D5-A7D9 A7F2-A801 A803-A805 A807-A80A "
    "A80C-A822 A830-A835 A840-A873 A882-A8B3 A8D0-A8D9 A8F2-A8F7 A8FB A8FD-A8FE A900-A925 "
    "A930-A946 A960-A97C A984-A9B2 A9CF-A9D9 A9E0-A9E4 A9E6-A9FE AA00-AA28 AA40-AA42 "
    "AA44-AA4B AA50-AA59 AA60-AA76 AA7A AA7E-AAAF AAB1 AAB5-AAB6 AAB9-AABD AAC0 AAC2 "
    "AADB-AADD AAE0-AAEA AAF2-AAF4 AB01-AB06 AB09-AB0E AB11-AB16 AB20-AB26 AB28-AB2E "
    "AB30-AB5A AB5C-AB69 AB70-ABE2 ABF0-ABF9 AC00-D7A3 D7B0-D7C6 D7CB-D7FB F900-FA6D "
    "FA70-FAD9 FB00-FB06 FB13-FB17 FB1D FB1F-FB28 FB2A-FB36 FB38-FB3C FB3E FB40-FB41 "
    "FB43-FB44 FB46-FBB1 FBD3-FD3D FD50-FD8F FD92-FDC7 FDF0-FDFB FE70-FE74 FE76-FEFC "
    "FF10-FF19 FF21-FF3A FF41-FF5A FF66-FFBE FFC2-FFC7 FFCA-FFCF FFD2-FFD7 FFDA-FFDC "
    "10000-1000B 1000D-10026 10028-1003A 1003C-1003D 1003F-1004D 10050-1005D 10080-100FA "
    "10107-10133 10140-10178 1018A-1018B 10280-1029C 102A0-102D0 102E1-102FB 10300-10323 "
    "1032D-1034A 10350-10375 10380-1039D 103A0-103C3 103C8-103CF 103D1-103D5 10400-1049D "
    "104A0-104A9 104B0-104D3 104D8-104FB 10500-10527 10530-10563 10570-1057A 1057C-1058A "
    "1058C-10592 10594-10595 10597-105A1 105A3-105B1 105B3-105B9 105BB-105BC 10600-10736 "
    "10740-10755 10760-10767 10780-10785 10787-107B0 107B2-107BA 10800-10805 10808 "
    "1080A-10835 10837-10838 1083C 1083F-10855 10858-10876 10879-1089E 108A7-108AF "
    "108E0-108F2 108F4-108F5 108FB-1091B 10920-10939 10980-109B7 109BC-109CF 109D2-10A00 "
    "10A10-10A13 10A15-10A17 10A19-10A35 10A40-10A48 10A60-10A7E 10A80-10A9F 10AC0-10AC7 "
    "10AC9-10AE4 10AEB-10AEF 10B00-10B35 10B40-10B55 10B58-10B72 10B78-10B91 10BA9-10BAF "
    "10C00-10C48 10C80-10CB2 10CC0-10CF2 10CFA-10D23 10D30-10D39 10E60-10E7E 10E80-10EA9 "
    "10EB0-10EB1 10F00-10F27 10F30-10F45 10F51-10F54 10F70-10F81 10FB0-10FCB 10FE0-10FF6 "
    "11003-11037 11052-1106F 11071-11072 11075 11083-110AF 110D0-110E8 110F0-110F9 "
    "11103-11126 11136-1113F 11144 11147 11150-11172 11176 11183-111B2 111C1-111C4 "
    "111D0-111DA 111DC 111E1-111F4 11200-11211 11213-1122B 1123F-11240 11280-11286 11288 "
    "1128A-1128D 1128F-1129D 1129F-112A8 112B0-112DE 112F0-112F9 11305-1130C 1130F-11310 "
    "11313-11328 1132A-11330 11332-11333 11335-11339 1133D 11350 1135D-11361 11400-11434 "
    "11447-1144A 11450-11459 1145F-11461 11480-114AF 114C4-114C5 114C7 114D0-114D9 "
    "11580-115AE 115D8-115DB 11600-1162F 11644 11650-11659 11680-116AA 116B8 116C0-116C9 "
    "11700-1171A 11730-1173B 11740-11746 11800-1182B 118A0-118F2 118FF-11906 11909 "
    "1190C-11913 11915-11916 11918-1192F 1193F 11941 11950-11959 119A0-119A7 119AA-119D0 "
    "119E1 119E3 11A00 11A0B-11A32 11A3A 11A50 11A5C-11A89 11A9D 11AB0-11AF8 11C00-11C08 "
    "11C0A-11C2E 11C40 11C50-11C6C 11C72-11C8F 11D00-11D06 11D08-11D09 11D0B-11D30 11D46 "
    "11D50-11D59 11D60-11D65 11D67-11D68 11D6A-11D89 11D98 11DA0-11DA9 11EE0-11EF2 11F02 "
    "11F04-11F10 11F12-11F33 11F50-11F59 11FB0 11FC0-11FD4 12000-12399 12400-1246E "
    "12480-12543 12F90-12FF0 13000-1342F 13441-13446 14400-14646 16800-16A38 16A40-16A5E "
    "16A60-16A69 16A70-16ABE 16AC0-16AC9 16AD0-16AED 16B00-16B2F 16B40-16B43 16B50-16B59 "
    "16B5B-16B61 16B63-16B77 16B7D-16B8F 16E40-16E96 16F00-16F4A 16F50 16F93-16F9F "
    "16FE0-16FE1 16FE3 17000-187F7 18800-18CD5 18D00-18D08 1AFF0-1AFF3 1AFF5-1AFFB "
    "1AFFD-1AFFE 1B000-1B122 1B132 1B150-1B152 1B155 1B164-1B167 1B170-1B2FB 1BC00-1BC6A "
    "1BC70-1BC7C 1BC80-1BC88 1BC90-1BC99 1D2C0-1D2D3 1D2E0-1D2F3 1D360-1D378 1D400-1D454 "
    "1D456-1D49C 1D49E-1D49F 1D4A2 1D4A5-1D4A6 1D4A9-1D4AC 1D4AE-1D4B9 1D4BB 1D4BD-1D4C3 "
    "1D4C5-1D505 1D507-1D50A 1D50D-1D514 1D516-1D51C 1D51E-1D539 1D53B-1D53E 1D540-1D544 "
    "1D546 1D54A-1D550 1D552-1D6A5 1D6A8-1D6C0 1D6C2-1D6DA 1D6DC-1D6FA 1D6FC-1D714 "
    "1D716-1D734 1D736-1D74E 1D750-1D76E 1D770-1D788 1D78A-1D7A8 1D7AA-1D7C2 1D7C4-1D7CB "
    "1D7CE-1D7FF 1DF00-1DF1E 1DF25-1DF2A 1E030-1E06D 1E100-1E12C 1E137-1E13D 1E140-1E149 "
    "1E14E 1E290-1E2AD 1E2C0-1E2EB 1E2F0-1E2F9 1E4D0-1E4EB 1E4F0-1E4F9 1E7E0-1E7E6 "
    "1E7E8-1E7EB 1E7ED-1E7EE 1E7F0-1E7FE 1E800-1E8C4 1E8C7-1E8CF 1E900-1E943 1E94B "
    "1E950-1E959 1EC71-1ECAB 1ECAD-1ECAF 1ECB1-1ECB4 1ED01-1ED2D 1ED2F-1ED3D 1EE00-1EE03 "
    "1EE05-1EE1F 1EE21-1EE22 1EE24 1EE27 1EE29-1EE32 1EE34-1EE37 1EE39 1EE3B 1EE42 1EE47 "
    "1EE49 1EE4B 1EE4D-1EE4F 1EE51-1EE52 1EE54 1EE57 1EE59 1EE5B 1EE5D 1EE5F 1EE61-1EE62 "
    "1EE64 1EE67-1EE6A 1EE6C-1EE72 1EE74-1EE77 1EE79-1EE7C 1EE7E 1EE80-1EE89 1EE8B-1EE9B "
    "1EEA1-1EEA3 1EEA5-1EEA9 1EEAB-1EEBB 1F100-1F10C 1FBF0-1FBF9 20000-2A6DF 2A700-2B739 "
    "2B740-2B81D 2B820-2CEA1 2CEB0-2EBE0 2F800-2FA1D 30000-3134A 31350-323AF"
)

_UNICODE_15_0_0_ND = (
    "30-39 660-669 6F0-6F9 7C0-7C9 966-96F 9E6-9EF A66-A6F AE6-AEF B66-B6F BE6-BEF C66-C6F "
    "CE6-CEF D66-D6F DE6-DEF E50-E59 ED0-ED9 F20-F29 1040-1049 1090-1099 17E0-17E9 1810-1819 "
    "1946-194F 19D0-19D9 1A80-1A89 1A90-1A99 1B50-1B59 1BB0-1BB9 1C40-1C49 1C50-1C59 "
    "A620-A629 A8D0-A8D9 A900-A909 A9D0-A9D9 A9F0-A9F9 AA50-AA59 ABF0-ABF9 FF10-FF19 "
    "104A0-104A9 10D30-10D39 11066-1106F 110F0-110F9 11136-1113F 111D0-111D9 112F0-112F9 "
    "11450-11459 114D0-114D9 11650-11659 116C0-116C9 11730-11739 118E0-118E9 11950-11959 "
    "11C50-11C59 11D50-11D59 11DA0-11DA9 11F50-11F59 16A60-16A69 16AC0-16AC9 16B50-16B59 "
    "1D7CE-1D7FF 1E140-1E149 1E2F0-1E2F9 1E4F0-1E4F9 1E950-1E959 1FBF0-1FBF9"
)


def _ranges(text):
    starts, ends = [], []
    for item in text.split(" "):
        a, _, b = item.partition("-")
        starts.append(int(a, 16))
        ends.append(int(b or a, 16))
    return starts, ends


_LN_STARTS, _LN_ENDS = _ranges(_UNICODE_15_0_0_LN)
_ND_STARTS, _ND_ENDS = _ranges(_UNICODE_15_0_0_ND)


def _in_ranges(cp, starts, ends):
    lo, hi = 0, len(starts)
    while lo < hi:
        mid = (lo + hi) // 2
        if starts[mid] <= cp:
            lo = mid + 1
        else:
            hi = mid
    return lo > 0 and cp <= ends[lo - 1]


def _is_ln(ch):
    return _in_ranges(ord(ch), _LN_STARTS, _LN_ENDS)


def _is_nd(ch):
    return _in_ranges(ord(ch), _ND_STARTS, _ND_ENDS)


def _token_char(ch):
    return ch in _TOKEN_EXTRA or _is_ln(ch)


def _numeric_tokens(text):
    tokens, cur = [], []
    for ch in text:
        if _token_char(ch):
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return [t for t in tokens if any(_is_nd(ch) for ch in t)]


def _ascii_digits(s):
    return bool(s) and all(ch in _DIGITS for ch in s)


def _numeric_form(text):
    """7.2: (None, (negative, int_digits, frac_digits, stripped token)) or (reason, detail)."""
    found = _numeric_tokens(text)
    detail = {"digit_bearing_tokens": found}
    if len(found) != 1:
        return ("number_count", detail), None
    tok = found[0]
    t = tok
    while t and t[-1] in ".,:":
        t = t[:-1]
    stripped = t
    neg = False
    if t[:1] in ("-", "+", "\u2212"):
        neg = t[0] != "+"
        t = t[1:]
    if t.endswith("%"):
        t = t[:-1]
    ip, dot, fp = t.partition(".")
    ok = True
    if dot:
        ok = _ascii_digits(fp)
    if ip == "":
        ok = ok and bool(dot)
    elif "," in ip:
        groups = ip.split(",")
        ok = ok and 1 <= len(groups[0]) <= 3 and _ascii_digits(groups[0]) and all(
            len(g) == 3 and _ascii_digits(g) for g in groups[1:])
    else:
        ok = ok and _ascii_digits(ip)
    if not ok:
        return ("number_grammar", detail), None
    return None, (neg, ip.replace(",", ""), fp, stripped)


def _plain(neg, digits, f):
    """Plain notation of the integer `digits` scaled by 10^-f: exactly f fractional digits, a zero
    result unsigned."""
    digits = digits.lstrip("0") or "0"
    if f:
        digits = digits.rjust(f + 1, "0")
        text = digits[:-f] + "." + digits[-f:]
    else:
        text = digits
    if neg and digits.strip("0"):
        text = "-" + text
    return text


def _inc(digits):
    """A digit string plus one."""
    out = list(digits)
    k = len(out) - 1
    while k >= 0:
        if out[k] == "9":
            out[k] = "0"
            k -= 1
        else:
            out[k] = chr(ord(out[k]) + 1)
            return "".join(out)
    return "1" + "".join(out)


def _dec(digits):
    """A positive digit string minus one."""
    out = list(digits)
    k = len(out) - 1
    while out[k] == "0":
        out[k] = "9"
        k -= 1
    out[k] = chr(ord(out[k]) - 1)
    return "".join(out).lstrip("0") or "0"


def _str_add(a, d):
    """|a| (a digit string, possibly longer than an int can be read from) plus the int d; the
    result is non-negative by construction of the callers."""
    if len(a) <= 4000:
        return str(int(a) + d)
    head, tail = a[:-19], int(a[-19:]) + d
    base = 10 ** 19
    while tail >= base:
        head, tail = _inc(head), tail - base
    while tail < 0:
        head, tail = _dec(head), tail + base
    return (head + str(tail).rjust(19, "0")).lstrip("0") or "0"


class _Dec(object):
    """A JSON number as coefficient and exponent (7.3). exp is an int, or None when the written
    exponent has too many digits to read as an int (then exp_neg and exp_text say which way)."""
    __slots__ = ("neg", "coef", "exp", "exp_neg", "exp_text", "nfrac")


def _decimal(text):
    d = _Dec()
    d.neg = text.startswith("-")
    t = text[1:] if d.neg else text
    mant, e_mark, e_part = t, "", ""
    for mark in "eE":
        if mark in t:
            mant, e_part = t.split(mark, 1)
            e_mark = mark
    ip, _, fp = mant.partition(".")
    d.coef = (ip + fp).lstrip("0") or "0"
    d.nfrac = len(fp)
    e_neg = e_part.startswith("-")
    e_digits = (e_part.lstrip("+-").lstrip("0") or "0") if e_mark else "0"
    if len(e_digits) <= 4000:
        written = int(e_digits)
        d.exp = (-written if e_neg else written) - d.nfrac
        d.exp_neg = d.exp < 0
        d.exp_text = None
    else:
        d.exp = None
        d.exp_neg = e_neg
        d.exp_text = e_digits
    return d


def _adjusted_text(d):
    """(negative, absolute adjusted exponent as digits)."""
    shift = len(d.coef) - 1
    if d.exp is not None:
        adj = d.exp + shift
        return adj < 0, str(-adj if adj < 0 else adj)
    if d.exp_neg:                                  # -(E + nfrac) + ... : |adj| = |E| + nfrac - shift
        return True, _str_add(d.exp_text, d.nfrac - shift)
    return False, _str_add(d.exp_text, shift - d.nfrac)


def _too_large(d):
    neg, mag = _adjusted_text(d)
    return (not neg) and _digits_lt(str(ADJUSTED_MAX), mag)


def _sci(d):
    """7.3's `receipt`: the scientific string form of the General Decimal Arithmetic spec."""
    sign = "-" if d.neg else ""
    adj_neg, adj_mag = _adjusted_text(d)
    if d.exp is not None and d.exp <= 0 and (not adj_neg or not _digits_lt("6", adj_mag)):
        k = -d.exp
        if k == 0:
            return sign + d.coef
        if len(d.coef) > k:
            return sign + d.coef[:-k] + "." + d.coef[-k:]
        return sign + "0." + "0" * (k - len(d.coef)) + d.coef
    body = d.coef[0] + ("." + d.coef[1:] if len(d.coef) > 1 else "")
    return sign + body + "E" + ("-" if adj_neg else "+") + adj_mag


def _round_scaled(d, f):
    """round_half_even(|value| * 10^f) as a digit string, in exact decimal arithmetic."""
    if d.coef == "0":
        return "0"
    if d.exp is None:
        return "0"                                    # only reached for a huge negative exponent
    s = d.exp + f
    if s >= 0:
        return d.coef + "0" * s
    k = -s
    if k > len(d.coef):
        return "0"
    kept = d.coef[:len(d.coef) - k] or "0"
    dropped = d.coef[len(d.coef) - k:]
    first, rest = dropped[0], dropped[1:]
    if first > "5" or (first == "5" and rest.strip("0")):
        up = True
    elif first == "5":
        up = kept[-1] in "13579"
    else:
        up = False
    if up:
        kept = _inc(kept)
    return kept.lstrip("0") or "0"


def _leaf_text(text):
    head = text[:80]
    return "".join("?" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in head)


def _leaf_type(v):
    if isinstance(v, JObj):
        return "object"
    if isinstance(v, list):
        return "array"
    if v is None:
        return "null"
    return None


def _numeric_compare(form, leaf):
    """7.3 and 7.3a: (verdict, reason, detail)."""
    lt = _leaf_type(leaf)
    if lt is not None:
        return "MALFORMED", "leaf_not_scalar", {"leaf_type": lt}
    if isinstance(leaf, str):
        return "MALFORMED", "leaf_not_numeric", {"leaf": _leaf_text(leaf)}
    if leaf is True or leaf is False:
        return "MALFORMED", "leaf_not_numeric", {"leaf": "True" if leaf else "False"}
    if isinstance(leaf, JConst):
        return "MALFORMED", "leaf_not_numeric", {"leaf": leaf.name}
    d = _decimal(leaf.text)
    if _too_large(d):
        return "MALFORMED", "leaf_not_numeric", {"leaf": _leaf_text(_sci(d))}
    neg, int_digits, frac_digits, token = form
    f = len(frac_digits)
    printed = _plain(neg, int_digits + frac_digits, f)
    rounded = _plain(d.neg, _round_scaled(d, f), f)
    detail = {"printed_token": token, "printed": printed, "receipt": _sci(d),
              "receipt_rounded": rounded, "fractional_digits": f, "rounding": "ROUND_HALF_EVEN"}
    if printed != rounded:
        return "FAILED", "value_mismatch", detail
    if d.coef != "0" and not rounded.lstrip("-").replace(".", "").strip("0"):
        return "FAILED", "rounds_to_zero", detail                        # 7.3a (D2, draft)
    return "HELD", None, detail


def _code_spans(inner):
    """2.3c's rule over the inner bytes: the contents of every inline code span, in order."""
    out, p, n = [], 0, len(inner)
    while p < n:
        if inner[p] != 0x60:
            p += 1
            continue
        q = p
        while q < n and inner[q] == 0x60:
            q += 1
        run = q - p
        lf = inner.find(b"\n", q)
        end = n if lf < 0 else lf
        r, found = q, -1
        while r < end:
            if inner[r] == 0x60:
                s0 = r
                while r < end and inner[r] == 0x60:
                    r += 1
                if r - s0 == run:
                    found = s0
                    break
            else:
                r += 1
        if found >= 0:
            out.append(inner[q:found])
            p = found + run
        else:
            p = q
    return out


def _needle_form(inner):
    spans = _code_spans(inner)
    if len(spans) != 1:
        return ("needle_count", {}), None
    needle = spans[0]
    if not needle.strip(_ASCII_WS):
        return ("needle_empty", {}), None
    return None, needle


def _hash_form(text):
    runs, n, i = [], len(text), 0
    while i < n:
        if text[i] in _HEX_ANY:
            j = i
            while j < n and text[j] in _HEX_ANY:
                j += 1
            glued = (i > 0 and text[i - 1] in _WORD_ASCII) or (j < n and text[j] in _WORD_ASCII)
            if not glued:
                runs.append((j - i, text[i:j]))
            i = j
        else:
            i += 1
    lengths = [r[0] for r in runs]
    if lengths.count(64) != 1 or any(x in (32, 40, 96, 128) for x in lengths):
        return ("digest_form", {"hex_runs": lengths}), None
    return None, [r[1] for r in runs if r[0] == 64][0].lower()


def _lines(data):
    """6.11: the lines of the receipt bytes (LF only, CR kept)."""
    pieces = data.split(b"\n")
    if data.endswith(b"\n") or not data:
        pieces = pieces[:-1]
    return pieces


def _walk(root, tokens):
    """6.12: (leaf, None) or (None, reason)."""
    v = root
    for t in tokens:
        if isinstance(v, JObj):
            if t not in v:
                return None, "pointer_unresolvable"
            if t in v.dup:
                return None, "pointer_ambiguous"
            v = v[t]
        elif isinstance(v, list):
            if not (t == "0" or _pos_int(t)) or len(t) > 18 or int(t) >= len(v):
                return None, "pointer_unresolvable"
            v = v[int(t)]
        else:
            return None, "pointer_unresolvable"
    return v, None


# ---------------------------------------------------------------------------------------------
# Section 6 and 7.1: one declaration
# ---------------------------------------------------------------------------------------------

def _adjudicate(d, b, man, offsets):
    """(verdict, reason, detail, extra) for one declaration; extra holds resolved_sha256 and
    provenance when 6.14 says the row carries them."""
    if d.mark is not None:                                                           # 7.1a
        return "MALFORMED", d.mark, ({"raw": d.raw} if d.raw is not None else {}), {}
    inner = b[d.inner_start:d.inner_end]
    text = inner.decode("utf-8")
    if not inner.strip(_ASCII_WS):                                                   # 7.1b
        return "MALFORMED", "empty_span", {}, {}
    if len(text) > CAP:                                                              # 7.1c
        return "MALFORMED", "length_cap", {"code_points": len(text), "bytes": len(inner),
                                           "cap": CAP}, {}
    if d.kind == "exec":                                                             # 7.1d
        return "MALFORMED", "kind_reserved", {"kind": d.kind}, {}
    if d.kind not in KINDS:
        return "MALFORMED", "kind_unknown", {"kind": d.kind}, {}
    ref = _parse_ref(d.receipt)                                                      # 7.1e
    if ref is None:
        return "MALFORMED", "receipt_form", {"receipt": d.receipt}, {}
    form, head, frag = ref
    if frag is not None and d.kind == "absent":                                      # 7.1f
        return "MALFORMED", "absent_over_partial", {"receipt": d.receipt}, {}
    if frag is not None and d.kind == "hash":
        return "MALFORMED", "hash_over_partial", {"receipt": d.receipt}, {}
    for ch in text:                                                                  # 7.1g
        if ch in ("\u202d", "\u202e"):
            return "MALFORMED", "directional_override", {"code_point": "U+%04X" % ord(ch)}, {}
    form_value = None                                                                # 7.1h
    if d.kind == "numeric":
        bad, form_value = _numeric_form(text)
    elif d.kind in ("quote", "absent"):
        bad, form_value = _needle_form(inner)
    else:
        bad, form_value = _hash_form(text)
    if bad is not None:
        return "MALFORMED", bad[0], bad[1], {}
    if form != "rn":                                                                 # 7.1i, 4.4
        return "UNRESOLVED", "no_repository", {}, {}
    # section 6
    if man is None:                                                                  # 6.1
        return "UNRESOLVED", "manifest_absent", {}, {}
    if not man.intact:                                                               # 6.2
        return "UNRESOLVED", "manifest_integrity", {}, {}
    if not (man.rung is None or (isinstance(man.rung, str) and man.rung in ("L1", "L2"))):  # 6.3
        return "UNRESOLVED", "rung_unknown", {}, {}
    if head not in man.receipts:                                                     # 6.4
        return "UNRESOLVED", "manifest_id_missing", {}, {}
    entry = man.receipts[head]
    if not isinstance(entry, JObj) or not _is_hex64_lower(entry.get("sha256")):      # 6.5
        return "UNRESOLVED", "manifest_integrity", {}, {}
    kos = entry.get("kind_of_source")
    if not isinstance(kos, str):                                                     # 6.6
        return "MALFORMED", "kind_of_source_unknown", {}, {}
    if kos in AUTHOR_SIDE or entry["sha256"] in man.authored:                        # 6.7
        return "MALFORMED", "receipt_author_minted", {}, {}
    if kos not in EXTERNAL:                                                          # 6.8
        return "MALFORMED", "kind_of_source_unknown", {}, {}
    extra = {"resolved_sha256": entry["sha256"],                                     # 6.9
             "provenance": {"form": "rn", "harness": man.harness,
                            "rung": man.rung if man.rung is not None else "undeclared",
                            "kind_of_source": kos}}
    data = None                                                                      # 6.10
    if "bytes" in entry:
        data = _b64_strict(entry["bytes"])
        if data is None or _sha256(data) != entry["sha256"]:
            return "UNRESOLVED", "manifest_integrity", {}, {}
    elif d.kind != "hash":
        return "UNRESOLVED", "manifest_bytes_absent", {}, {}
    leaf, have_leaf, haystack = None, False, data
    if frag is not None and frag[0] == "lines":                                      # 6.11
        lines = _lines(data)
        a, bb = frag[1], frag[2]
        if len(bb) > 18 or int(bb) > len(lines):
            return "MALFORMED", "anchor_out_of_range", {}, {}
        haystack = b"\n".join(lines[int(a) - 1:int(bb)])
    elif frag is not None:                                                           # 6.12
        try:
            root = parse_json(data)
        except JSONError:
            return "MALFORMED", "receipt_not_json", {}, {}
        leaf, why = _walk(root, frag[1])
        if why is not None:
            return "MALFORMED", why, {}, {}
        have_leaf = True
    if d.kind == "absent":                                                           # 7.1j
        complete = entry.get("complete")
        if complete is not True and complete is not False:
            return "UNRESOLVED", "manifest_no_completeness", {}, {}
        if complete is False:
            return "MALFORMED", "absent_over_partial", {"complete": False}, extra
    # 7.1k
    if d.kind == "numeric":                                                          # 7.3
        if not have_leaf:
            try:
                leaf = parse_json(haystack)
            except JSONError:
                return "MALFORMED", "receipt_not_json", {}, extra
        v, r, det = _numeric_compare(form_value, leaf)
        return v, r, det, extra
    if d.kind == "quote":                                                            # 7.5
        needle = form_value
        if have_leaf:
            if not isinstance(leaf, str):
                return "MALFORMED", "leaf_not_string", {}, extra
            if _has_surrogate(leaf):
                return "MALFORMED", "leaf_not_string", {
                    "note": "leaf is not encodable as UTF-8"}, extra
            hay = leaf.encode("utf-8")
        else:
            hay = haystack
            if len(needle) < MIN_NEEDLE and not _narrows(frag, data, hay):
                return "MALFORMED", "short_needle", {"needle_bytes": len(needle),
                                                     "minimum_bytes": MIN_NEEDLE}, extra
        occ = hay.count(needle)
        det = {"needle_bytes": len(needle), "haystack_bytes": len(hay), "occurrences": occ}
        return ("HELD", None, det, extra) if occ else ("FAILED", "needle_missing", det, extra)
    if d.kind == "hash":                                                             # 7.7
        det = {"stated": form_value, "receipt_sha256": entry["sha256"]}
        if form_value == entry["sha256"]:
            return "HELD", None, det, extra
        return "FAILED", "digest_mismatch", det, extra
    needle = form_value                                                              # 7.8
    occ = data.count(needle)
    det = {"needle_bytes": len(needle), "haystack_bytes": len(data), "complete": True}
    if occ:
        return "FAILED", "needle_present", det, extra
    return "HELD", None, det, extra


def _narrows(frag, data, slice_bytes):
    """7.5: a line fragment narrows when its slice differs from the full-content slice, or when
    the whole receipt is under 16 bytes. Only a line fragment can narrow."""
    if frag is None or frag[0] != "lines":
        return False
    if len(data) < MIN_NEEDLE:
        return True
    full = data.rstrip(b"\n")
    if not full:
        return False                                      # k = 0: nothing narrows but the clause above
    return slice_bytes != full


# ---------------------------------------------------------------------------------------------
# Sections 8, 9 and 11.1: the core
# ---------------------------------------------------------------------------------------------

def _document_verdict(document_malformed, counts, total):
    if document_malformed is not None:
        return "SWORN-FAILED"
    if total == 0:
        return "UNSWORN"
    if counts["FAILED"] or counts["MALFORMED"]:
        return "SWORN-FAILED"
    if counts["HELD"] == 0:
        return "SWORN-UNCHECKED"
    return "SWORN-HELD"


def check(document, name, manifest=None, commit=None):
    """Sections 2 to 9 with 11.1's refusals. `document` and `manifest` are bytes (manifest may be
    None); `name` a str; `commit` None or a hex label. Returns the verdict core (a dict)."""
    if document is None:
        raise Refusal("nothing_to_verify")
    if not isinstance(document, (bytes, bytearray)):
        raise TypeError("the document is bytes")
    b = bytes(document)
    if not isinstance(name, str) or _has_surrogate(name):
        raise Refusal("name_form")
    if not (commit is None or _is_hex40_lower(commit) or _is_hex64_lower(commit)):
        raise Refusal("commit_form")
    man = None
    if manifest is not None:
        if not isinstance(manifest, (bytes, bytearray)):
            raise TypeError("the manifest is bytes")
        man = read_manifest(bytes(manifest))
    rows, document_malformed, canonical_sha256 = [], None, None
    try:
        b.decode("utf-8")
        bad_at = None
    except UnicodeDecodeError as e:
        bad_at = e.start
    if bad_at is not None:                                                          # 2.1
        document_malformed = {"reason": "invalid_utf8", "at": bad_at}
    else:
        regions, delimiter_lines = _fences(b)
        if len(delimiter_lines) % 2:                                                # 2.2
            document_malformed = {"reason": "unbalanced_fences",
                                  "delimiter_lines": delimiter_lines}
    if document_malformed is None:
        decls = _scan(b)
        canonical, offsets = _canonical(b, decls)
        if canonical is not None:
            canonical_sha256 = _sha256(canonical)
        for d in decls:
            verdict, reason, detail, extra = _adjudicate(d, b, man, offsets)
            if id(d) in offsets:
                start, end = offsets[id(d)]
                inner = b[d.inner_start:d.inner_end]
                claim_sha256 = _sha256(inner)
                text = inner.decode("utf-8")
                claim = text if len(text) <= CAP else None
            else:
                start = end = claim_sha256 = claim = None
            row = {"at": d.at, "start": start, "end": end, "receipt": d.receipt,
                   "kind": d.kind, "claim_sha256": claim_sha256, "claim": claim,
                   "verdict": verdict, "reason": reason, "detail": detail}
            row.update(extra)
            rows.append(row)
    counts = {w: 0 for w in SPAN_VERDICTS}
    rungs = {}
    for r in rows:
        counts[r["verdict"]] += 1
        key = r["provenance"]["rung"] if "provenance" in r else "unresolved"
        rungs[key] = rungs.get(key, 0) + 1
    core = {
        "schema": SCHEMA_V0,
        "profile": PROFILE_RN,
        "format": FORMAT,
        "document": {"name": name, "inline_sha256": _sha256(b),
                     "canonical_sha256": canonical_sha256},
        "commit": commit,
        "manifest_digest": man.digest if man is not None else None,
        "spans": rows,
        "counts": counts,
        "sworn_total": len(rows),
        "unresolved": counts["UNRESOLVED"],
        "document_verdict": _document_verdict(document_malformed, counts, len(rows)),
        "document_malformed": document_malformed,
        "rungs": rungs,
    }
    jcs(core)                                    # 11.2: a core with no canonical form is refused
    return core


def receipt(core, verifier=None):
    """9.4: the core beside its digest. `verifier` is provenance the digest does not cover."""
    out = {"core": core, "digest": digest(core), "attestations": []}
    if verifier is not None:
        if not isinstance(verifier, dict):
            raise TypeError("verifier is an object")
        out["verifier"] = verifier
    return out


# ---------------------------------------------------------------------------------------------
# 11.3: checking a receipt
# ---------------------------------------------------------------------------------------------

def _is_int_min(v, lo):
    if not (isinstance(v, JNum) and v.is_integer_token()):
        return False
    t = v.text
    if len(t.lstrip("-")) > 30:
        return not t.startswith("-")
    return int(t) >= lo


def _is_const_int(v, n):
    return isinstance(v, JNum) and v.is_integer_token() and len(v.text) < 30 and int(v.text) == n


def _is_hex64(v):
    return _is_hex64_lower(v)


def _keys_ok(obj, allowed, required):
    return isinstance(obj, JObj) and set(obj) <= set(allowed) and set(required) <= set(obj)


_DETAIL_TYPES = {
    "raw": "string", "code_points": "count", "bytes": "count", "cap": ("const", 300),
    "kind": "string", "receipt": "string", "complete": "boolean", "code_point": "code_point",
    "digit_bearing_tokens": "strings", "hex_runs": "counts", "printed_token": "string",
    "printed": "string", "receipt_rounded": "string", "fractional_digits": "count",
    "rounding": ("str", "ROUND_HALF_EVEN"), "needle_bytes": "count", "minimum_bytes": ("const", 16),
    "haystack_bytes": "count", "occurrences": "count", "leaf_type": "leaf_type", "leaf": "leaf",
    "note": "string", "stated": "hex64", "receipt_sha256": "hex64",
}


def _detail_ok(det):
    if not isinstance(det, JObj):
        return False
    for k, v in det.items():
        t = _DETAIL_TYPES.get(k)
        if t is None:
            return False
        if t == "string":
            ok = isinstance(v, str)
        elif t == "count":
            ok = _is_int_min(v, 0)
        elif t == "boolean":
            ok = v is True or v is False
        elif t == "code_point":
            ok = (isinstance(v, str) and v.startswith("U+") and 4 <= len(v) - 2 <= 6
                  and all(ch in "0123456789ABCDEF" for ch in v[2:]))
        elif t == "strings":
            ok = isinstance(v, list) and all(isinstance(x, str) for x in v)
        elif t == "counts":
            ok = isinstance(v, list) and all(_is_int_min(x, 0) for x in v)
        elif t == "leaf_type":
            ok = v in ("object", "array", "null") and isinstance(v, str)
        elif t == "leaf":
            ok = isinstance(v, str) and len(v) <= 80
        elif t == "hex64":
            ok = _is_hex64(v)
        elif t[0] == "const":
            ok = _is_const_int(v, t[1])
        else:
            ok = isinstance(v, str) and v == t[1]
        if not ok:
            return False
    return True


def _provenance_ok(p, rn_only):
    if _keys_ok(p, ("form", "harness", "rung", "kind_of_source"),
                ("form", "harness", "rung", "kind_of_source")):
        if (p["form"] == "rn" and isinstance(p["form"], str) and isinstance(p["harness"], str)
                and isinstance(p["rung"], str) and p["rung"] in ("L1", "L2", "undeclared")
                and isinstance(p["kind_of_source"], str) and p["kind_of_source"] in EXTERNAL):
            return True
    if rn_only:
        return False
    return (_keys_ok(p, ("form", "note"), ("form", "note")) and isinstance(p["form"], str)
            and p["form"] in ("path", "prereg") and isinstance(p["note"], str))


_ROW_KEYS = ("at", "start", "end", "receipt", "kind", "claim_sha256", "claim", "verdict",
             "reason", "detail")


def _row_ok(r, rn):
    if not _keys_ok(r, _ROW_KEYS + ("resolved_sha256", "provenance"), _ROW_KEYS):
        return False
    if not _is_int_min(r["at"], 0):
        return False
    for k in ("start", "end"):
        if not (r[k] is None or _is_int_min(r[k], 0)):
            return False
    for k in ("receipt", "kind"):
        if not (r[k] is None or isinstance(r[k], str)):
            return False
    if not (r["claim_sha256"] is None or _is_hex64(r["claim_sha256"])):
        return False
    if not (r["claim"] is None or (isinstance(r["claim"], str) and len(r["claim"]) <= CAP)):
        return False
    v = r["verdict"]
    if not (isinstance(v, str) and v in SPAN_VERDICTS):
        return False
    reason = r["reason"]
    allowed = REASONS if rn else REASONS + LEGACY_REASONS
    if not (reason is None or (isinstance(reason, str) and reason in allowed)):
        return False
    if not _detail_ok(r["detail"]):
        return False
    if "resolved_sha256" in r and not _is_hex64(r["resolved_sha256"]):
        return False
    if "provenance" in r and not _provenance_ok(r["provenance"], rn):
        return False
    if v == "HELD":
        if reason is not None or "resolved_sha256" not in r or "provenance" not in r:
            return False
    elif not isinstance(reason, str):
        return False
    if r["start"] is None:
        if not (r["end"] is None and r["claim_sha256"] is None and r["claim"] is None):
            return False
    elif not isinstance(r["end"], JNum):
        return False
    if r["claim_sha256"] is None and r["claim"] is not None:
        return False
    return True


_CORE_KEYS = ("schema", "profile", "format", "document", "commit", "manifest_digest", "spans",
              "counts", "sworn_total", "unresolved", "document_verdict", "document_malformed",
              "rungs")


def _core_ok(c):
    if not _keys_ok(c, _CORE_KEYS, _CORE_KEYS):
        return False
    if not (isinstance(c["schema"], str) and c["schema"] == SCHEMA_V0):
        return False
    if not (isinstance(c["profile"], str) and c["profile"] in (PROFILE_RN, PROFILE_LEGACY)):
        return False
    rn = c["profile"] == PROFILE_RN
    if not (isinstance(c["format"], str) and c["format"] == FORMAT):
        return False
    doc = c["document"]
    keys = ("name", "inline_sha256", "canonical_sha256")
    if not (_keys_ok(doc, keys, keys) and isinstance(doc["name"], str)
            and _is_hex64(doc["inline_sha256"])
            and (doc["canonical_sha256"] is None or _is_hex64(doc["canonical_sha256"]))):
        return False
    if not (c["commit"] is None or _is_hex40_lower(c["commit"]) or _is_hex64(c["commit"])):
        return False
    if not (c["manifest_digest"] is None or _is_hex64(c["manifest_digest"])):
        return False
    if not (isinstance(c["spans"], list) and all(_row_ok(r, rn) for r in c["spans"])):
        return False
    if not (_keys_ok(c["counts"], SPAN_VERDICTS, SPAN_VERDICTS)
            and all(_is_int_min(c["counts"][k], 0) for k in SPAN_VERDICTS)):
        return False
    if not (_is_int_min(c["sworn_total"], 0) and _is_int_min(c["unresolved"], 0)):
        return False
    if not (isinstance(c["document_verdict"], str) and c["document_verdict"] in DOCUMENT_VERDICTS):
        return False
    dm = c["document_malformed"]
    if dm is not None:
        if _keys_ok(dm, ("reason", "at"), ("reason", "at")):
            if not (dm["reason"] == "invalid_utf8" and isinstance(dm["reason"], str)
                    and _is_int_min(dm["at"], 0)):
                return False
        elif _keys_ok(dm, ("reason", "delimiter_lines"), ("reason", "delimiter_lines")):
            if not (dm["reason"] == "unbalanced_fences" and isinstance(dm["reason"], str)
                    and isinstance(dm["delimiter_lines"], list)
                    and all(_is_int_min(x, 1) for x in dm["delimiter_lines"])):
                return False
        else:
            return False
    rg = c["rungs"]
    if not (isinstance(rg, JObj) and all(_is_int_min(v, 0) for v in rg.values())):
        return False
    if rn and not set(rg) <= {"L1", "L2", "undeclared", "unresolved"}:
        return False
    return True


def _receipt_ok(r):
    """The schema's `receipt` shape, except `attestations` holding items: that is 11.3's own
    second refusal, attestations_not_empty, and is answered there."""
    if not _keys_ok(r, ("core", "digest", "verifier", "attestations"),
                    ("core", "digest", "attestations")):
        return False
    if not _is_hex64(r["digest"]):
        return False
    if "verifier" in r and not isinstance(r["verifier"], JObj):
        return False
    if not isinstance(r["attestations"], list):
        return False
    return _core_ok(r["core"])


def recheck(receipt_bytes, document, manifest=None):
    """11.3: REDERIVES or DOES-NOT-REDERIVE, or a refusal. REDERIVES says the receipt is what
    this checker computes; it does not say the document held."""
    if not isinstance(receipt_bytes, (bytes, bytearray)):
        raise TypeError("the receipt is bytes")
    try:
        r = parse_json(bytes(receipt_bytes))
    except JSONError:
        raise Refusal("receipt_shape", "not JSON under 6.13")
    if _has_dup(r):
        raise Refusal("receipt_shape", "a key written twice")
    if not _receipt_ok(r):
        raise Refusal("receipt_shape")
    if r["attestations"]:
        raise Refusal("attestations_not_empty")
    try:
        again = digest(r["core"])
    except Refusal:
        raise Refusal("receipt_digest", "the core has no canonical form")
    if again != r["digest"]:
        raise Refusal("receipt_digest")
    core = r["core"]
    if core["profile"] != PROFILE_RN:
        raise Refusal("profile_not_checkable")
    if document is None:
        raise Refusal("nothing_to_verify")
    if _sha256(bytes(document)) != core["document"]["inline_sha256"]:
        raise Refusal("document_mismatch")
    mine = check(document, core["document"]["name"], manifest, core["commit"])
    return REDERIVES if digest(mine) == r["digest"] else DOES_NOT_REDERIVE


# ---------------------------------------------------------------------------------------------
# Command line. The only function that reads a file; it reads only the paths it is handed.
# ---------------------------------------------------------------------------------------------

VERIFIER = {"name": "receipt/v0/check_v0.py", "written_by": "fathom-lab",
            "independent": False, "status": "draft"}


def _main(argv):
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="check_v0.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("doc")
    c.add_argument("--manifest")
    c.add_argument("--name")
    c.add_argument("--commit")
    rc = sub.add_parser("recheck")
    rc.add_argument("receipt")
    rc.add_argument("doc")
    rc.add_argument("--manifest")
    a = ap.parse_args(argv)

    def read(path):
        with open(path, "rb") as fh:
            return fh.read()

    try:
        man = read(a.manifest) if a.manifest is not None else None
        if a.cmd == "check":
            core = check(read(a.doc), a.name if a.name is not None else a.doc, man, a.commit)
            sys.stdout.buffer.write(jcs(receipt(core, VERIFIER)).encode("utf-8") + b"\n")
            return 0
        answer = recheck(read(a.receipt), read(a.doc), man)
        sys.stdout.write(answer + "\n")
        return 0 if answer == REDERIVES else 1
    except Refusal as e:
        sys.stdout.write("REFUSED %s\n" % e.code)
        return 2


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_main(_sys.argv[1:]))
