#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""derive.py: the checker layer of DERIVE v0 (DRAFT, unsigned). See DERIVE_v0.md beside this file.

One derive span is checked by derive_span(claim, receipt_bytes): the derive blocks of the document,
the span's REF and inner text, the resolution of its manifest entry (done by RECEIPT v0, not here)
and the entry's bytes. The answer is a span row (verdict, reason, detail), the budget units the
span spent, and a surface text that is outside every digest.

Pure: every function below main() is a function of its arguments. Standard library only; nothing
from styxx. No floating point, no evaluation of text, no regular expressions, no clock, no network,
no subprocess, no environment. The one place that touches a file is main(), which reads the files
the user names on its command line and nothing else.
"""
import base64
import hashlib
import sys
import unicodedata

PROFILE = "derive/v0-draft"

CAP_RECEIPT_BYTES = 16777216
CAP_ROWS = 100000
CAP_VALUES = 1000000
CAP_DEPTH = 32
CAP_BLOCKS = 16
CAP_BLOCK_BYTES = 65536
CAP_DERIVATIONS = 256
CAP_SPANS = 256
CAP_DERIVATION_BYTES = 4096
CAP_TESTS = 8
CAP_IN = 16
CAP_STRING_BYTES = 256
CAP_POINTER_BYTES = 1024
CAP_POINTER_TOKENS = 16
CAP_PLACES = 18
CAP_INT_DIGITS = 37
CAP_INNER_CP = 300
CAP_CENSUS = 16
CAP_SIBLINGS = 16
CAP_SHOW_CP = 64
CAP_CLAIM_BYTES = 8388608
CAP_BATCH_BYTES = 33554432
BUDGET = 67108864
SCALE = 18
TEN18 = 10 ** SCALE
INT_MAX = 9007199254740991

OPS = ("count", "share", "percent", "sum", "mean")
UNRESOLVED_REASONS = ("manifest_absent", "manifest_integrity", "rung_unknown",
                      "manifest_id_missing", "manifest_bytes_absent")
RESOLUTION_MALFORMED = ("kind_of_source_unknown", "receipt_author_minted")
RUNGS = ("L1", "L2", "undeclared")


class Mal(Exception):
    """A MALFORMED answer."""

    def __init__(self, reason, detail=None):
        Exception.__init__(self, reason)
        self.reason = reason
        self.detail = {} if detail is None else detail


class Refusal(Exception):
    """Not a verdict: the caller gets a code and nothing is emitted."""

    def __init__(self, code):
        Exception.__init__(self, code)
        self.code = code


# ------------------------------------------------------------------ strict JSON (appendix A.1)
class Num(object):
    __slots__ = ("tok",)

    def __init__(self, tok):
        self.tok = tok


class NonFinite(object):
    __slots__ = ("tok",)

    def __init__(self, tok):
        self.tok = tok


class Arr(object):
    __slots__ = ("items", "spans")

    def __init__(self):
        self.items = []
        self.spans = []

    def add(self, key, v, s, e):
        self.items.append(v)
        self.spans.append((s, e))


class Obj(object):
    __slots__ = ("pairs", "last", "repeated", "spans")

    def __init__(self):
        self.pairs = []
        self.last = {}
        self.repeated = set()
        self.spans = []

    def add(self, key, v, s, e):
        self.pairs.append((key, v))
        self.spans.append((s, e))
        if key in self.last:
            self.repeated.add(key)
        self.last[key] = v


class NotJson(Exception):
    pass


class TooManyValues(Exception):
    pass


class Counter(object):
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0


_WS = " \t\n\r"
_HEX = "0123456789abcdefABCDEF"
_CTRL = dict((c, None) for c in range(32))
_ESC = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _ws(t, i, n):
    while i < n and t[i] in _WS:
        i += 1
    return i


def _hex4(t, k, n):
    if k + 4 > n:
        raise NotJson()
    s = t[k:k + 4]
    for ch in s:
        if ch not in _HEX:
            raise NotJson()
    return int(s, 16)


def _string(t, i, n):
    j = i + 1
    parts = []
    q = -2
    while True:
        # q, the next quote at or after j, stays valid while q >= j; the backslash search is
        # bounded by q, so each byte of the text is scanned a bounded number of times
        if q < j:
            q = t.find('"', j)
            if q < 0:
                raise NotJson()
        b = t.find("\\", j, q)
        end = q if b == -1 else b
        seg = t[j:end]
        if seg and len(seg.translate(_CTRL)) != len(seg):
            raise NotJson()
        parts.append(seg)
        if end == q:
            return "".join(parts), q + 1
        if b + 1 >= n:
            raise NotJson()
        e = t[b + 1]
        if e == "u":
            cu = _hex4(t, b + 2, n)
            j = b + 6
            if 0xD800 <= cu <= 0xDBFF and t.startswith("\\u", j):
                lo = _hex4(t, j + 2, n)
                if 0xDC00 <= lo <= 0xDFFF:
                    parts.append(chr(0x10000 + ((cu - 0xD800) << 10) + (lo - 0xDC00)))
                    j += 6
                    continue
            parts.append(chr(cu))
        elif e in _ESC:
            parts.append(_ESC[e])
            j = b + 2
        else:
            raise NotJson()


def _isdigit(ch):
    return "0" <= ch <= "9"


def _number(t, i, n):
    s = i
    if t[i] == "-":
        i += 1
    if i >= n:
        raise NotJson()
    if t[i] == "0":
        i += 1
    elif "1" <= t[i] <= "9":
        while i < n and _isdigit(t[i]):
            i += 1
    else:
        raise NotJson()
    if i < n and t[i] == ".":
        i += 1
        k = i
        while i < n and _isdigit(t[i]):
            i += 1
        if i == k:
            raise NotJson()
    if i < n and (t[i] == "e" or t[i] == "E"):
        i += 1
        if i < n and (t[i] == "+" or t[i] == "-"):
            i += 1
        k = i
        while i < n and _isdigit(t[i]):
            i += 1
        if i == k:
            raise NotJson()
    return t[s:i], i


def _key(t, i, n):
    if i >= n or t[i] != '"':
        raise NotJson()
    k, i = _string(t, i, n)
    i = _ws(t, i, n)
    if i >= n or t[i] != ":":
        raise NotJson()
    return k, _ws(t, i + 1, n)


def read_json(t, ctr):
    """A str -> a strict value. Raises NotJson (syntax, or a 33rd container opened) or
    TooManyValues (the counter passed CAP_VALUES), whichever the reader meets first."""
    n = len(t)
    i = _ws(t, 0, n)
    stack = []
    while True:
        if i >= n:
            raise NotJson()
        ctr.n += 1
        if ctr.n > CAP_VALUES:
            raise TooManyValues()
        c = t[i]
        if c == "[" or c == "{":
            if len(stack) >= CAP_DEPTH:
                raise NotJson()
            node = Arr() if c == "[" else Obj()
            close = "]" if c == "[" else "}"
            i = _ws(t, i + 1, n)
            if i < n and t[i] == close:
                i += 1
                val = node
            else:
                frame = [node, c == "{", None, 0]
                if c == "{":
                    frame[2], i = _key(t, i, n)
                frame[3] = i
                stack.append(frame)
                continue
        elif c == '"':
            val, i = _string(t, i, n)
        elif t.startswith("-Infinity", i):
            val = NonFinite("-Infinity")
            i += 9
        elif c == "-" or _isdigit(c):
            tok, i = _number(t, i, n)
            val = Num(tok)
        elif t.startswith("true", i):
            val = True
            i += 4
        elif t.startswith("false", i):
            val = False
            i += 5
        elif t.startswith("null", i):
            val = None
            i += 4
        elif t.startswith("NaN", i):
            val = NonFinite("NaN")
            i += 3
        elif t.startswith("Infinity", i):
            val = NonFinite("Infinity")
            i += 8
        else:
            raise NotJson()
        while True:
            if not stack:
                i = _ws(t, i, n)
                if i != n:
                    raise NotJson()
                return val
            fr = stack[-1]
            fr[0].add(fr[2], val, fr[3], i)
            i = _ws(t, i, n)
            if i < n and t[i] == ",":
                i = _ws(t, i + 1, n)
                if fr[1]:
                    fr[2], i = _key(t, i, n)
                fr[3] = i
                break
            if i < n and t[i] == ("}" if fr[1] else "]"):
                i += 1
                stack.pop()
                val = fr[0]
                continue
            raise NotJson()


def utf8_bad(data):
    """Offset of the first byte of the first ill-formed UTF-8 sequence, or -1."""
    try:
        data.decode("utf-8")
        return -1
    except UnicodeDecodeError:
        pass
    i, n = 0, len(data)
    while i < n:
        b = data[i]
        if b < 0x80:
            i += 1
            continue
        if 0xC2 <= b <= 0xDF:
            need, lo, hi = 1, 0x80, 0xBF
        elif b == 0xE0:
            need, lo, hi = 2, 0xA0, 0xBF
        elif 0xE1 <= b <= 0xEC or 0xEE <= b <= 0xEF:
            need, lo, hi = 2, 0x80, 0xBF
        elif b == 0xED:
            need, lo, hi = 2, 0x80, 0x9F
        elif b == 0xF0:
            need, lo, hi = 3, 0x90, 0xBF
        elif 0xF1 <= b <= 0xF3:
            need, lo, hi = 3, 0x80, 0xBF
        elif b == 0xF4:
            need, lo, hi = 3, 0x80, 0x8F
        else:
            return i
        if i + need >= n:
            return i
        if not (lo <= data[i + 1] <= hi):
            return i
        for k in range(2, need + 1):
            if not (0x80 <= data[i + k] <= 0xBF):
                return i
        i += need + 1
    return -1


def jtype(v):
    if v is None:
        return "null"
    if v is True or v is False:
        return "boolean"
    if isinstance(v, str):
        return "string"
    if isinstance(v, Num):
        return "number"
    if isinstance(v, NonFinite):
        return "non_finite"
    if isinstance(v, Arr):
        return "array"
    return "object"


# ------------------------------------------------------------------ canonical form (appendix A.2)
def _has_lone(s):
    for ch in s:
        if 0xD800 <= ord(ch) <= 0xDFFF:
            return True
    return False


def _cstr(s):
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif o == 8:
            out.append("\\b")
        elif o == 9:
            out.append("\\t")
        elif o == 10:
            out.append("\\n")
        elif o == 12:
            out.append("\\f")
        elif o == 13:
            out.append("\\r")
        elif o < 0x20:
            out.append("\\u%04x" % o)
        elif 0xD800 <= o <= 0xDFFF:
            raise ValueError("lone_surrogate")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def canon(v):
    """Plain value -> canonical text (RECEIPT v0 3.1, 3.2). Only values this module built reach
    here; a value outside the form is a defect of this module and raises."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        if v < -INT_MAX or v > INT_MAX:
            raise ValueError("int_range")
        return str(v)
    if isinstance(v, str):
        return _cstr(v)
    if isinstance(v, list):
        return "[" + ",".join(canon(x) for x in v) + "]"
    if isinstance(v, dict):
        ks = sorted(v)
        for k in ks:
            for ch in k:
                if ord(ch) > 0x7E:
                    raise ValueError("non_ascii_key")
        return "{" + ",".join(_cstr(k) + ":" + canon(v[k]) for k in ks) + "}"
    raise ValueError("type")


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def utf8_len(s):
    n = 0
    for ch in s:
        o = ord(ch)
        n += 1 if o < 0x80 else 2 if o < 0x800 else 3 if o < 0x10000 else 4
    return n


def show(s):
    """At most 64 code points of s, each unpaired surrogate replaced by '?'."""
    out = []
    for ch in s[:CAP_SHOW_CP]:
        out.append("?" if 0xD800 <= ord(ch) <= 0xDFFF else ch)
    return "".join(out)


def _plain_canonical(v, depth):
    """Strict value -> plain value with a canonical form, or None when it has none."""
    if v is None or v is True or v is False:
        return (v,)
    if isinstance(v, str):
        return None if _has_lone(v) else (v,)
    if isinstance(v, Num):
        t = v.tok
        neg = t.startswith("-")
        d = t[1:] if neg else t
        if not d or not all(_isdigit(ch) for ch in d):
            return None
        if d == "0":
            return None if neg else (0,)
        if d[0] == "0" or len(d) > 16:
            return None
        x = int(d)
        if x > INT_MAX:
            return None
        return (-x if neg else x,)
    if isinstance(v, NonFinite):
        return None
    if depth >= CAP_DEPTH:
        return None
    if isinstance(v, Arr):
        out = []
        for x in v.items:
            p = _plain_canonical(x, depth + 1)
            if p is None:
                return None
            out.append(p[0])
        return (out,)
    if v.repeated:
        return None
    out = {}
    for k, x in v.pairs:
        if any(ord(ch) > 0x7E for ch in k):
            return None
        p = _plain_canonical(x, depth + 1)
        if p is None:
            return None
        out[k] = p[0]
    return (out,)


# ------------------------------------------------------------------ section 3: the derivation
def is_name(s):
    if not isinstance(s, str) or not (1 <= len(s) <= 32):
        return False
    if not ("a" <= s[0] <= "z"):
        return False
    for ch in s[1:]:
        if not ("a" <= ch <= "z" or _isdigit(ch) or ch == "_"):
            return False
    return True


def pointer_tokens(p):
    """A JSON Pointer string -> its unescaped tokens, or None when it breaks 3.2's rule."""
    if p == "":
        return []
    if p[0] != "/" or utf8_len(p) > CAP_POINTER_BYTES:
        return None
    raw = p.split("/")[1:]
    if len(raw) > CAP_POINTER_TOKENS:
        return None
    out = []
    for tok in raw:
        k = 0
        while True:
            k = tok.find("~", k)
            if k < 0:
                break
            if k + 1 >= len(tok) or tok[k + 1] not in "01":
                return None
            k += 2
        out.append(tok.replace("~1", "/").replace("~0", "~"))
    return out


def dec_scaled(s):
    """A dec literal's string -> integer count of 10^-18 units, or None when it breaks 3.4."""
    if not isinstance(s, str):
        return None
    neg = s.startswith("-")
    t = s[1:] if neg else s
    whole, dot, frac = t.partition(".")
    if not whole or not all(_isdigit(ch) for ch in whole):
        return None
    if len(whole) > 1 and whole[0] == "0":
        return None
    if len(whole) > 18:
        return None
    if dot:
        if not frac or len(frac) > 18 or not all(_isdigit(ch) for ch in frac):
            return None
    v = int(whole + frac + "0" * (SCALE - len(frac)))
    return -v if neg else v


def _lit_ok(x, ordered):
    if x is True or x is False:
        return not ordered
    if isinstance(x, int):
        return True
    if isinstance(x, str):
        return utf8_len(x) <= CAP_STRING_BYTES
    if isinstance(x, dict):
        return list(x) == ["dec"] and dec_scaled(x["dec"]) is not None
    return False


def _subject_ok(t, jsonl, for_of):
    """t: a test or an `of` object. True when it has exactly one well-formed subject key."""
    keys = [k for k in t if k in ("field", "at", "row")]
    if len(keys) != 1:
        return False
    k = keys[0]
    x = t[k]
    if k == "field":
        return isinstance(x, str) and utf8_len(x) <= CAP_STRING_BYTES
    if k == "at":
        if not isinstance(x, str) or x == "":
            return False
        return pointer_tokens(x) is not None
    if x == "value":
        return True
    if x == "key":
        return not jsonl and not for_of
    return False


def _test_ok(t, jsonl):
    if not isinstance(t, dict) or len(t) != 2 or not _subject_ok(t, jsonl, False):
        return False
    rel = [k for k in t if k not in ("field", "at", "row")]
    if len(rel) != 1:
        return False
    r = rel[0]
    x = t[r]
    if r == "eq":
        return _lit_ok(x, False)
    if r in ("ge", "le"):
        return _lit_ok(x, True)
    if r == "in":
        if not isinstance(x, list) or not (1 <= len(x) <= CAP_IN):
            return False
        if not all(_lit_ok(y, False) for y in x):
            return False
        return len(set(canon(y) for y in x)) == len(x)
    if r == "present":
        return (x is True or x is False) and "row" not in t
    return False


def check_derivation(raw):
    """Section 3.6 -> the plain derivation, or Mal derivation_form / op_unknown."""
    if not isinstance(raw, Obj):
        raise Mal("derivation_form")
    p = _plain_canonical(raw, 1)
    if p is None:
        raise Mal("derivation_form")
    d = p[0]
    if utf8_len(canon(d)) > CAP_DERIVATION_BYTES:
        raise Mal("derivation_form")
    op = d.get("op")
    if not isinstance(op, str):
        raise Mal("derivation_form")
    if op not in OPS:
        raise Mal("op_unknown", {"op": op})
    want = {"count": ["op", "rows", "where"], "share": ["match", "op", "rows", "where"],
            "percent": ["match", "op", "rows", "where"], "sum": ["of", "op", "rows", "where"],
            "mean": ["of", "op", "rows", "where"]}[op]
    have = sorted(k for k in d if k != "within")
    if have != want:
        raise Mal("derivation_form")
    if "within" in d and not is_name(d["within"]):
        raise Mal("derivation_form")
    rows = d["rows"]
    if not isinstance(rows, dict) or rows.get("format") not in ("json", "jsonl"):
        raise Mal("derivation_form")
    jsonl = rows["format"] == "jsonl"
    if jsonl:
        if sorted(rows) != ["format"]:
            raise Mal("derivation_form")
    else:
        if sorted(rows) != ["at", "format"] or not isinstance(rows["at"], str):
            raise Mal("derivation_form")
        if pointer_tokens(rows["at"]) is None:
            raise Mal("derivation_form")
    w = d["where"]
    m = d.get("match", [])
    if not isinstance(w, list) or not isinstance(m, list):
        raise Mal("derivation_form")
    if "match" in d and not m:
        raise Mal("derivation_form")
    if len(w) + len(m) > CAP_TESTS:
        raise Mal("derivation_form")
    for t in w + m:
        if not _test_ok(t, jsonl):
            raise Mal("derivation_form")
    if "of" in d:
        of = d["of"]
        if not isinstance(of, dict) or len(of) != 1 or not _subject_ok(of, jsonl, True):
            raise Mal("derivation_form")
    return d


# ------------------------------------------------------------------ v0 7.2: the printed number
_PUNCT = frozenset("_.,+-%/:") | frozenset(["−", "±"]) | frozenset(chr(c) for c in (
    0xAD, 0x58A, 0x5BE, 0x1400, 0x1806, 0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2E17,
    0x2E1A, 0x2E3A, 0x2E3B, 0x2E40, 0x2E5D, 0x301C, 0x3030, 0x30A0, 0xFE31, 0xFE32, 0xFE58,
    0xFE63, 0xFF0D, 0x10EAD))


def _tokchar(ch):
    return ch in _PUNCT or unicodedata.category(ch)[0] in ("L", "N")


def _alldig(s):
    return s != "" and all(_isdigit(ch) for ch in s)


def printed_number(inner):
    """-> (printed_token, signed integer P at f places, f, has_percent) or Mal."""
    toks, cur = [], []
    for ch in inner:
        if _tokchar(ch):
            cur.append(ch)
        else:
            if cur:
                toks.append("".join(cur))
            cur = []
    if cur:
        toks.append("".join(cur))
    bearing = [t for t in toks if any(unicodedata.category(c) == "Nd" for c in t)]
    if len(bearing) != 1:
        raise Mal("number_count", {"digit_bearing_tokens": bearing})
    tok = bearing[0].rstrip(".,:")
    bad = Mal("number_grammar", {"digit_bearing_tokens": bearing})
    i, n = 0, len(tok)
    neg = False
    if i < n and tok[i] in ("-", "+", "−"):
        neg = tok[i] != "+"
        i += 1
    pct = n > i and tok[n - 1] == "%"
    body = tok[i:n - 1] if pct else tok[i:]
    whole, dot, frac = body.partition(".")
    if dot and not _alldig(frac):
        raise bad
    if whole == "":
        if not dot:
            raise bad
    elif "," in whole:
        groups = whole.split(",")
        if not (1 <= len(groups[0]) <= 3 and _alldig(groups[0])):
            raise bad
        for g in groups[1:]:
            if len(g) != 3 or not _alldig(g):
                raise bad
        whole = whole.replace(",", "")
    elif not _alldig(whole):
        raise bad
    f = len(frac)
    if f > CAP_PLACES:
        raise Mal("places_cap", {"fractional_digits": f, "cap": CAP_PLACES})
    if len(whole) > CAP_INT_DIGITS:
        raise Mal("printed_too_long", {"integer_digits": len(whole), "cap": CAP_INT_DIGITS})
    p = int((whole or "0") + frac)
    return tok, (-p if neg else p), f, pct


# ------------------------------------------------------------------ section 7: numbers
def scaled(tok):
    """A JSON number token -> integer count of 10^-18 units, or None when outside 7.1's domain.
    Every bound is decided on the token's text before any integer is built."""
    neg = tok.startswith("-")
    t = tok[1:] if neg else tok
    e = t.find("e")
    if e < 0:
        e = t.find("E")
    mant, exp = (t[:e], t[e + 1:]) if e >= 0 else (t, "")
    esign = 1
    if exp and exp[0] in "+-":
        esign = -1 if exp[0] == "-" else 1
        exp = exp[1:]
    whole, _, frac = mant.partition(".")
    c = (whole + frac).lstrip("0")
    if not c:
        return 0
    ed = exp.lstrip("0") or "0"
    if len(ed) > 4:
        return None
    q = esign * int(ed) - len(frac)
    st = c.rstrip("0")
    q += len(c) - len(st)
    c = st
    if q < -SCALE or len(c) + q > SCALE:
        return None
    v = int(c) * 10 ** (q + SCALE)
    return -v if neg else v


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def frac_obj(p, q):
    g = gcd(abs(p), q) or 1
    return {"num": str(p // g), "den": str(q // g)}


def round_half_even(p, q, f):
    """p/q, q > 0, at f places -> (text, exact_tie, w)."""
    n = abs(p) * 10 ** f
    w, r = divmod(n, q)
    tie = 2 * r == q
    if 2 * r > q or (tie and w % 2 == 1):
        w += 1
    s = str(w)
    if len(s) < f + 1:
        s = "0" * (f + 1 - len(s)) + s
    text = (s[:len(s) - f] + "." + s[len(s) - f:]) if f else s
    if p < 0 and w != 0:
        text = "-" + text
    return text, tie, w


# ------------------------------------------------------------------ section 2: the block table
def block_table(blocks):
    """-> (table, None) or (None, block_line)."""
    table = {}
    for bi, blk in enumerate(blocks):
        line = blk["line"]
        if bi >= CAP_BLOCKS:
            return None, line
        content = blk["content"]
        if utf8_len(content) > CAP_BLOCK_BYTES:
            return None, line
        try:
            v = read_json(content, Counter())
        except (NotJson, TooManyValues):
            return None, line
        if not isinstance(v, Obj):
            return None, line
        for k, d in v.pairs:
            if not is_name(k) or k in table or len(table) >= CAP_DERIVATIONS:
                return None, line
            table[k] = d
    return table, None


# ------------------------------------------------------------------ section 4: the row source
def load_rows(data, rows):
    """-> dict(rows=[(key or None, value, text)], keyed, source fields) or Mal."""
    ctr = Counter()
    if rows["format"] == "jsonl":
        nlines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
        if nlines > CAP_ROWS:
            raise Mal("rows_too_many", {"rows": nlines, "cap": CAP_ROWS})
        bad = utf8_bad(data)
        if bad >= 0:
            raise Mal("receipt_not_jsonl", {"line": data.count(b"\n", 0, bad) + 1})
        text = data.decode("utf-8")
        lines = text.split("\n")
        if text.endswith("\n") or text == "":
            lines.pop()
        out = []
        for ln, line in enumerate(lines, 1):
            body = line.strip(" \t\r")
            if body == "":
                raise Mal("receipt_not_jsonl", {"line": ln})
            try:
                v = read_json(line, ctr)
            except NotJson:
                raise Mal("receipt_not_jsonl", {"line": ln})
            except TooManyValues:
                raise Mal("receipt_too_many_values", {"cap": CAP_VALUES})
            out.append((None, v, body))
        return {"rows": out, "keyed": False, "pointer_tokens": 0, "parent": None, "token": None}
    if utf8_bad(data) >= 0:
        raise Mal("receipt_not_json")
    text = data.decode("utf-8")
    try:
        v = read_json(text, ctr)
    except NotJson:
        raise Mal("receipt_not_json")
    except TooManyValues:
        raise Mal("receipt_too_many_values", {"cap": CAP_VALUES})
    at = rows["at"]
    toks = pointer_tokens(at)
    parent, ptok = None, None
    for tok in toks:
        if isinstance(v, Obj):
            if tok not in v.last:
                raise Mal("pointer_unresolvable", {"at": at})
            if tok in v.repeated:
                raise Mal("pointer_ambiguous", {"at": at})
            parent, ptok = v, tok
            v = v.last[tok]
        elif isinstance(v, Arr):
            if not _alldig(tok) or (len(tok) > 1 and tok[0] == "0") or len(tok) > 9 or int(tok) >= len(v.items):
                raise Mal("pointer_unresolvable", {"at": at})
            parent, ptok = v, int(tok)
            v = v.items[int(tok)]
        else:
            raise Mal("pointer_unresolvable", {"at": at})
    if isinstance(v, Arr):
        nrows = len(v.items)
    elif isinstance(v, Obj):
        nrows = len(v.pairs)
    else:
        raise Mal("rows_not_container", {"type": jtype(v)})
    if nrows > CAP_ROWS:
        raise Mal("rows_too_many", {"rows": nrows, "cap": CAP_ROWS})
    if isinstance(v, Obj):
        if v.repeated:
            raise Mal("rows_ambiguous")
        out = [(k, x, None) for k, x in v.pairs]
        keyed = True
    else:
        out = [(None, x, text[s:e]) for x, (s, e) in zip(v.items, v.spans)]
        keyed = False
    return {"rows": out, "keyed": keyed, "pointer_tokens": len(toks), "parent": parent, "token": ptok}


def _size(v):
    return len(v.items) if isinstance(v, Arr) else len(v.pairs)


def source_info(src, data, rows_spec, complete):
    parent, ptok = src["parent"], src["token"]
    sib, nsib, members = [], 0, None
    if parent is not None:
        if isinstance(parent, Obj):
            members = len(parent.pairs)
            seq = [(k, x) for k, x in parent.pairs if k != ptok]
        else:
            members = len(parent.items)
            seq = [(str(i), x) for i, x in enumerate(parent.items) if i != ptok]
        for k, x in seq:
            if isinstance(x, (Arr, Obj)):
                nsib += 1
                if len(sib) < CAP_SIBLINGS:
                    sib.append({"at": show(k), "type": jtype(x), "size": _size(x)})
    return {"format": rows_spec["format"], "receipt_bytes": len(data), "pointer_tokens": src["pointer_tokens"],
            "parent_members": members, "siblings": sib, "sibling_containers": nsib, "complete": complete}


# ------------------------------------------------------------------ sections 5, 6, 8: the visit
ABSENT = object()


class Test(object):
    __slots__ = ("subj", "skind", "sarg", "rel", "lits", "obj", "ck")


def _compile_subject(t):
    if "field" in t:
        return {"field": t["field"]}, "field", t["field"]
    if "at" in t:
        return {"at": t["at"]}, "at", pointer_tokens(t["at"])
    return {"row": t["row"]}, t["row"], None


def _lit_value(x):
    if x is True or x is False:
        return ("b", x)
    if isinstance(x, int):
        return ("n", x * TEN18)
    if isinstance(x, str):
        return ("s", x)
    return ("n", dec_scaled(x["dec"]))


def compile_test(t):
    c = Test()
    c.obj = t
    c.subj, c.skind, c.sarg = _compile_subject(t)
    c.rel = [k for k in t if k not in ("field", "at", "row")][0]
    x = t[c.rel]
    if c.rel == "in":
        c.lits = [_lit_value(y) for y in x]
    elif c.rel == "present":
        c.lits = x
    else:
        c.lits = [_lit_value(x)]
    return c


def subject_value(skind, sarg, subj, key, row, i):
    if skind == "field":
        if not isinstance(row, Obj) or sarg not in row.last:
            return ABSENT
        if sarg in row.repeated:
            raise Mal("field_ambiguous", {"row": i, "subject": subj})
        return row.last[sarg]
    if skind == "at":
        v = row
        for tok in sarg:
            if isinstance(v, Obj):
                if tok not in v.last:
                    return ABSENT
                if tok in v.repeated:
                    raise Mal("field_ambiguous", {"row": i, "subject": subj})
                v = v.last[tok]
            elif isinstance(v, Arr):
                if not _alldig(tok) or (len(tok) > 1 and tok[0] == "0") or len(tok) > 9 or int(tok) >= len(v.items):
                    return ABSENT
                v = v.items[int(tok)]
            else:
                return ABSENT
        return v
    if skind == "key":
        return key
    return row


def conv(v, i):
    s = scaled(v.tok)
    if s is None:
        raise Mal("number_out_of_range", {"row": i, "number": v.tok[:80]})
    return s


def _eq(v, lit, i):
    kind, x = lit
    if kind == "s":
        return isinstance(v, str) and v == x
    if kind == "b":
        return v is x
    return isinstance(v, Num) and conv(v, i) == x


def apply_test(c, v, i):
    """-> (truth, no_value)."""
    if c.rel == "present":
        return ((v is not ABSENT) == c.lits), False
    if v is ABSENT or v is None:
        return False, True
    if c.rel == "eq":
        return _eq(v, c.lits[0], i), False
    if c.rel == "in":
        if isinstance(v, Num) and any(k == "n" for k, _ in c.lits):
            conv(v, i)
        for lit in c.lits:
            if _eq(v, lit, i):
                return True, False
        return False, False
    kind, x = c.lits[0]
    if kind == "s":
        if not isinstance(v, str):
            raise Mal("where_type_mismatch", {"row": i, "type": jtype(v), "literal_type": "string"})
        a = v
    else:
        if not isinstance(v, Num):
            raise Mal("where_type_mismatch", {"row": i, "type": jtype(v), "literal_type": "number"})
        a = conv(v, i)
    return (a >= x if c.rel == "ge" else a <= x), False


def census_key(v):
    if v is None:
        return ("null", "null")
    if v is True:
        return ("boolean", "true")
    if v is False:
        return ("boolean", "false")
    if isinstance(v, str):
        return ("string", v)
    if isinstance(v, Num):
        return ("number", v.tok)
    if isinstance(v, NonFinite):
        return ("non_finite", v.tok)
    return (jtype(v), "")


def _lit_seen(lit, hist):
    kind, x = lit
    if kind == "s":
        return ("string", x) in hist
    if kind == "b":
        return ("boolean", "true" if x else "false") in hist
    for (ty, tok) in hist:
        if ty == "number" and scaled(tok) == x:
            return True
    return False


def visit(src, d, wd):
    """The pass of sections 5 and 6 over every row. wd: the within derivation or None."""
    op = d["op"]
    rows = src["rows"]
    W = [compile_test(t) for t in d["where"]]
    M = [compile_test(t) for t in d.get("match", [])]
    X = [compile_test(t) for t in (wd["where"] if wd is not None else [])]
    of = _compile_subject(d["of"]) if "of" in d else None
    # the subjects, each once, in order of first appearance: where, match, of, within
    subs, order = {}, []
    for s in [c.subj for c in W + M] + ([of[0]] if of else []) + [c.subj for c in X]:
        k = canon(s)
        if k not in subs:
            subs[k] = _compile_subject(s)
            order.append(k)
    for c in W + M + X:
        c.ck = canon(c.subj)
    ofk = canon(of[0]) if of else None
    guard_keys = [k for k in order if subs[k][1] in ("field", "at")]
    seen_guard = set()
    # census subjects: subjects of eq and in tests, in order of first appearance in where, match
    cen_order, cen_lits = [], {}
    for c in W + M:
        if c.rel in ("eq", "in"):
            if c.ck not in cen_lits:
                cen_order.append(c.ck)
                cen_lits[c.ck] = []
            raw = c.obj[c.rel]
            for y in (raw if c.rel == "in" else [raw]):
                if canon(y) not in [canon(z) for z in cen_lits[c.ck]]:
                    cen_lits[c.ck].append(y)
    hist = dict((k, {}) for k in cen_order)
    absent = dict((k, 0) for k in cen_order)
    nW = len(W)
    wsteps = [[0, 0] for _ in W]
    msteps = [[0, 0] for _ in M]
    wo_sel = [0] * nW
    wo_match = [0] * nW
    wo_used = [0] * nW
    wo_total = [0] * nW
    selected = matched = used = no_value = total = 0
    within_sel = outside = 0
    first_row = last_row = first_key = last_key = None
    dup = 0
    seen_text = set()
    sum_like = op in ("sum", "mean")
    share_like = op in ("share", "percent")
    for i, (key, row, text) in enumerate(rows):
        vals = {}
        for k in order:
            sk = subs[k]
            vals[k] = subject_value(sk[1], sk[2], sk[0], key, row, i)
        wres = []
        for j, c in enumerate(W):
            ok, nv = apply_test(c, vals[c.ck], i)
            wres.append(ok)
            if ok:
                wsteps[j][0] += 1
            if nv:
                wsteps[j][1] += 1
        mres = [apply_test(c, vals[c.ck], i) for c in M]
        ofnum = None
        if of is not None:
            ofv = vals[ofk]
            if ofv is not ABSENT and ofv is not None:
                if isinstance(ofv, Num):
                    ofnum = conv(ofv, i)
                else:
                    raise Mal("field_not_numeric", {"row": i, "type": jtype(ofv)})
        xres = True
        for c in X:
            ok, _ = apply_test(c, vals[c.ck], i)
            xres = xres and ok
        for k in guard_keys:
            if vals[k] is not ABSENT:
                seen_guard.add(k)
        for k in cen_order:
            v = vals[k]
            if v is ABSENT:
                absent[k] += 1
            else:
                ck = census_key(v)
                h = hist[k]
                h[ck] = h.get(ck, 0) + 1
        if text is not None:
            if text in seen_text:
                dup += 1
            else:
                seen_text.add(text)
        mall = all(ok for ok, _ in mres)
        fails = [j for j in range(nW) if not wres[j]]
        sel = not fails
        if wd is not None:
            if xres:
                within_sel += 1
            if sel and not xres:
                outside += 1
        if sel:
            selected += 1
            if first_row is None:
                first_row = i
                first_key = key
            last_row = i
            last_key = key
            for j, (ok, nv) in enumerate(mres):
                if ok:
                    msteps[j][0] += 1
                if nv:
                    msteps[j][1] += 1
            if mall:
                matched += 1
            if sum_like:
                if ofnum is None:
                    no_value += 1
                else:
                    used += 1
                    total += ofnum
        targets = range(nW) if sel else (fails if len(fails) == 1 else [])
        for j in targets:
            wo_sel[j] += 1
            if mall:
                wo_match[j] += 1
            if sum_like and ofnum is not None:
                wo_used[j] += 1
                wo_total[j] += ofnum
    N = len(rows)
    if N > 0:
        for k in guard_keys:
            if k not in seen_guard:
                raise Mal("field_never_present", {"subject": subs[k][0], "rows_total": N})
    counts = {"total": N, "selected": selected, "duplicate_rows": None if src["keyed"] else dup}
    if share_like:
        counts["used"] = selected
        counts["matched"] = matched
    elif sum_like:
        counts["used"] = used
        counts["no_value"] = no_value
    else:
        counts["used"] = selected
    if wd is not None:
        counts["within"] = within_sel
        counts["outside_within"] = outside
    without = []
    for j in range(nW):
        e = {"selected": wo_sel[j]}
        pq = value_of(op, wo_sel[j], wo_match[j], wo_used[j], wo_total[j])
        if share_like:
            e["matched"] = wo_match[j]
        if sum_like:
            e["used"] = wo_used[j]
        e["derived"] = None if pq is None else frac_obj(pq[0], pq[1])
        without.append(e)
    census = []
    for k in cen_order:
        h = hist[k]
        items = sorted(h.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        top = items[:CAP_CENSUS]
        shown = sum(c for _, c in top)
        census.append({"subject": subs[k][0], "distinct": len(h), "absent": absent[k],
                       "values": [{"type": ty, "text": show(tx), "rows": c} for (ty, tx), c in top],
                       "other": N - absent[k] - shown,
                       "literals_never_seen": [y for y in cen_lits[k] if not _lit_seen(_lit_value(y), h)]})
    span = {"first_row": first_row, "last_row": last_row,
            "first_key": None if first_key is None else show(first_key),
            "last_key": None if last_key is None else show(last_key)}
    return {"rows": counts, "where_steps": [{"passed": a, "no_value": b} for a, b in wsteps],
            "match_steps": [{"passed": a, "no_value": b} for a, b in msteps],
            "without": without, "census": census, "selected_span": span,
            "pq": value_of(op, selected, matched, used, total)}


def value_of(op, k, m, used, total):
    """The exact value P/Q of section 7.2, or None where it is undefined."""
    if op == "count":
        return (k, 1)
    if op in ("share", "percent"):
        if k == 0:
            return None
        return (m * (100 if op == "percent" else 1), k)
    if used == 0:
        return None
    return (total, TEN18) if op == "sum" else (total, used * TEN18)


# ------------------------------------------------------------------ the claim (section 13)
def _no_lone_anywhere(v):
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            if _has_lone(x):
                return False
        elif isinstance(x, Arr):
            stack.extend(x.items)
        elif isinstance(x, Obj):
            if x.repeated:
                return False
            for k, y in x.pairs:
                if _has_lone(k):
                    return False
                stack.append(y)
    return True


def _claim_int(v, minimum):
    if not isinstance(v, Num):
        raise Refusal("claim_form")
    t = v.tok
    if not _alldig(t) or (len(t) > 1 and t[0] == "0") or len(t) > 15:
        raise Refusal("claim_form")
    x = int(t)
    if x < minimum:
        raise Refusal("claim_form")
    return x


def _obj_keys(o, required, optional=()):
    if not isinstance(o, Obj):
        raise Refusal("claim_form")
    ks = set(o.last)
    if not set(required) <= ks or not ks <= set(required) | set(optional):
        raise Refusal("claim_form")
    return o.last


def read_claim(data):
    """Claim file bytes -> a plain claim, or Refusal."""
    if len(data) > CAP_CLAIM_BYTES:
        raise Refusal("claim_too_large")
    if utf8_bad(data) >= 0:
        raise Refusal("claim_form")
    try:
        v = read_json(data.decode("utf-8"), Counter())
    except (NotJson, TooManyValues):
        raise Refusal("claim_form")
    if not _no_lone_anywhere(v):
        raise Refusal("claim_form")
    top = _obj_keys(v, ("blocks", "span", "resolution"), ("budget",))
    if not isinstance(top["blocks"], Arr):
        raise Refusal("claim_form")
    blocks = []
    for b in top["blocks"].items:
        bk = _obj_keys(b, ("line", "content"))
        if not isinstance(bk["content"], str):
            raise Refusal("claim_form")
        blocks.append({"line": _claim_int(bk["line"], 1), "content": bk["content"]})
    sp = _obj_keys(top["span"], ("receipt", "inner"), ("ordinal",))
    if not isinstance(sp["receipt"], str) or not isinstance(sp["inner"], str):
        raise Refusal("claim_form")
    span = {"receipt": sp["receipt"], "inner": sp["inner"],
            "ordinal": _claim_int(sp["ordinal"], 1) if "ordinal" in sp else 1}
    rs = top["resolution"]
    if not isinstance(rs, Obj) or not isinstance(rs.last.get("verdict"), str):
        raise Refusal("claim_form")
    verdict = rs.last["verdict"]
    if verdict == "RESOLVED":
        r = _obj_keys(rs, ("verdict", "entry"))
        e = _obj_keys(r["entry"], ("kind_of_source", "complete", "harness", "rung"))
        if not isinstance(e["kind_of_source"], str) or not isinstance(e["harness"], str):
            raise Refusal("claim_form")
        if e["complete"] not in (True, False, None) or e["rung"] not in RUNGS:
            raise Refusal("claim_form")
        res = {"verdict": "RESOLVED", "entry": {"kind_of_source": e["kind_of_source"], "complete": e["complete"],
                                                "harness": e["harness"], "rung": e["rung"]}}
    elif verdict in ("UNRESOLVED", "MALFORMED"):
        r = _obj_keys(rs, ("verdict", "reason"))
        allowed = UNRESOLVED_REASONS if verdict == "UNRESOLVED" else RESOLUTION_MALFORMED
        if r["reason"] not in allowed:
            raise Refusal("claim_form")
        res = {"verdict": verdict, "reason": r["reason"]}
    else:
        raise Refusal("claim_form")
    budget = {"spent_before": 0, "entry_read_before": False}
    if "budget" in top:
        bg = _obj_keys(top["budget"], ("spent_before", "entry_read_before"))
        if bg["entry_read_before"] not in (True, False):
            raise Refusal("claim_form")
        budget = {"spent_before": _claim_int(bg["spent_before"], 0), "entry_read_before": bg["entry_read_before"]}
    return {"blocks": blocks, "span": span, "resolution": res, "budget": budget}


# ------------------------------------------------------------------ section 9: one derive span
def _ref_parts(ref):
    head, hashmark, frag = ref.partition("#")
    if not hashmark or len(head) < 2 or head[0] != "r" or head[1] == "0" or not _alldig(head[1:]):
        return None
    if not is_name(frag):
        return None
    return head, frag


def derive_span(claim, data):
    """-> (row, spent). claim: as read_claim returns it; data: the entry's bytes."""
    span = claim["span"]
    ref = span["receipt"]
    row = {"kind": "derive", "receipt": ref}
    spent = claim["budget"]["spent_before"]
    state = {"spent": spent, "visit": False, "provenance": None}
    try:
        detail, verdict, reason = _derive(claim, data, state)
    except Mal as m:
        verdict, reason, detail = "MALFORMED", m.reason, m.detail
    except _Unresolved as u:
        verdict, reason, detail = "UNRESOLVED", u.reason, {}
    row["verdict"] = verdict
    row["reason"] = reason
    row["detail"] = detail
    if state["visit"]:
        row["provenance"] = state["provenance"]
        row["resolved_sha256"] = sha256_hex(data)
    return row, state["spent"]


class _Unresolved(Exception):
    def __init__(self, reason):
        Exception.__init__(self, reason)
        self.reason = reason


def _derive(claim, data, state):
    span = claim["span"]
    ref, inner = span["receipt"], span["inner"]
    # b, c: v0 7.1b and 7.1c
    if all(ch in " \t\n\x0b\x0c\r" for ch in inner):
        raise Mal("empty_span")
    if len(inner) > CAP_INNER_CP:
        raise Mal("length_cap", {"code_points": len(inner), "bytes": utf8_len(inner), "cap": CAP_INNER_CP})
    # d
    if span["ordinal"] > CAP_SPANS:
        raise Mal("derive_span_cap", {"cap": CAP_SPANS})
    # e
    parts = _ref_parts(ref)
    if parts is None:
        raise Mal("receipt_form", {"receipt": ref})
    name = parts[1]
    for ch in inner:
        if ch in ("‭", "‮"):
            raise Mal("directional_override", {"code_point": "U+%04X" % ord(ch)})
    tok, pp, f, pct = printed_number(inner)
    table, bad_line = block_table(claim["blocks"])
    if table is None:
        raise Mal("derive_block_malformed", {"block_line": bad_line})
    if name not in table:
        raise Mal("derivation_missing", {"name": name})
    d = check_derivation(table[name])
    if pct and d["op"] != "percent":
        raise Mal("unit_mismatch", {"printed_token": tok, "op": d["op"]})
    wd = None
    if "within" in d:
        wn = d["within"]
        bad = Mal("within_form", {"within": wn})
        if wn == name or wn not in table:
            raise bad
        try:
            wd = check_derivation(table[wn])
        except Mal:
            raise bad
        if "within" in wd or canon(wd["rows"]) != canon(d["rows"]):
            raise bad
    # f: resolution, by RECEIPT v0 section 6
    res = claim["resolution"]
    if res["verdict"] == "UNRESOLVED":
        raise _Unresolved(res["reason"])
    if res["verdict"] == "MALFORMED":
        raise Mal(res["reason"])
    entry = res["entry"]
    # g: the row source
    if len(data) > CAP_RECEIPT_BYTES:
        raise Mal("rows_too_large", {"bytes": len(data), "cap": CAP_RECEIPT_BYTES})
    spent0 = claim["budget"]["spent_before"]
    cost_bytes = 0 if claim["budget"]["entry_read_before"] else len(data)
    if spent0 + cost_bytes > BUDGET:
        raise Mal("derive_budget", {"budget": BUDGET, "spent_before": spent0, "cost": cost_bytes})
    state["spent"] = spent0 + cost_bytes
    src = load_rows(data, d["rows"])
    if not src["keyed"]:
        subjects = d["where"] + d.get("match", []) + (wd["where"] if wd else [])
        if any(t.get("row") == "key" for t in subjects):
            raise Mal("row_key_unavailable")
    ntests = len(d["where"]) + len(d.get("match", [])) + (1 if "of" in d else 0) + (len(wd["where"]) if wd else 0)
    cost_rows = len(src["rows"]) * (ntests + 1)
    if spent0 + cost_bytes + cost_rows > BUDGET:
        raise Mal("derive_budget", {"budget": BUDGET, "spent_before": spent0, "cost": cost_bytes + cost_rows})
    state["spent"] = spent0 + cost_bytes + cost_rows
    # h: the visit
    state["visit"] = True
    state["provenance"] = {"form": "rn", "harness": entry["harness"], "rung": entry["rung"],
                           "kind_of_source": entry["kind_of_source"]}
    r = visit(src, d, wd)
    base = {"printed_token": tok, "printed": round_half_even(pp, 10 ** f, f)[0], "fractional_digits": f,
            "derivation": d, "derivation_sha256": sha256_hex(canon(d).encode("utf-8")),
            "rows": r["rows"], "where_steps": r["where_steps"], "without": r["without"],
            "census": r["census"], "selected_span": r["selected_span"],
            "source": source_info(src, data, d["rows"], entry["complete"])}
    if d["op"] in ("share", "percent"):
        base["match_steps"] = r["match_steps"]
    if r["pq"] is None:
        raise Mal("division_by_zero" if d["op"] in ("share", "percent") else "no_rows_used", base)
    p, q = r["pq"]
    # i: the comparison
    rounded, tie, w = round_half_even(p, q, f)
    a, b = pp * q, p * 10 ** f
    base["rounding"] = "ROUND_HALF_EVEN"
    base["derived"] = frac_obj(p, q)
    base["derived_rounded"] = rounded
    base["exact_tie"] = tie
    base["printed_vs_exact"] = "above" if a > b else "below" if a < b else "equal"
    if rounded != base["printed"]:
        return base, "FAILED", "value_mismatch"
    if p != 0 and w == 0:
        return base, "FAILED", "rounds_to_zero"
    if d["op"] in ("share", "percent"):
        top = (100 if d["op"] == "percent" else 1) * 10 ** f
        if p < (100 if d["op"] == "percent" else 1) * q and w == top:
            return base, "FAILED", "rounds_to_whole"
    if wd is not None and r["rows"]["outside_within"] > 0:
        return base, "FAILED", "not_within"
    return base, "HELD", None


# ------------------------------------------------------------------ section 8.4: the surface text
def _render_lit(x):
    if isinstance(x, dict):
        return x["dec"]
    return canon(x)


def _render_subject(t):
    if "field" in t:
        return "field " + canon(t["field"])
    if "at" in t:
        return "value at " + canon(t["at"])
    return "row key" if t["row"] == "key" else "row value"


def _render_test(t):
    s = _render_subject(t)
    if "eq" in t:
        return s + " = " + _render_lit(t["eq"])
    if "in" in t:
        return s + " in [" + ", ".join(_render_lit(y) for y in t["in"]) + "]"
    if "ge" in t:
        return s + " >= " + _render_lit(t["ge"])
    if "le" in t:
        return s + " <= " + _render_lit(t["le"])
    return s + (" is present" if t["present"] else " is absent")


def _render_where(tests):
    return " and ".join(_render_test(t) for t in tests) if tests else ""


def surface(row):
    """A fixed template over the row (section 8.4). Text, never markup; outside every digest."""
    if row["verdict"] not in ("HELD", "FAILED"):
        return ""
    det = row["detail"]
    d = det["derivation"]
    op = d["op"]
    w = _render_where(d["where"])
    among = ("rows where " + w) if w else "all rows"
    if op == "count":
        head = "count of " + among
    elif op in ("share", "percent"):
        head = op + " of rows where " + _render_where(d["match"]) + ", among " + among
    else:
        head = op + " of " + _render_subject(d["of"]) + " over " + among
    if "within" in d:
        head += ", within the selection of " + d["within"]
    rc = det["rows"]
    counts = "%d of %d rows selected" % (rc["selected"], rc["total"])
    if op in ("share", "percent"):
        counts += "; %d of %d matched" % (rc["matched"], rc["selected"])
    if op in ("sum", "mean"):
        counts += "; %d of %d used, %d without a value" % (rc["used"], rc["selected"], rc["no_value"])
    if "outside_within" in rc:
        counts += "; %d selected rows outside %s" % (rc["outside_within"], d["within"])
    if rc["duplicate_rows"]:
        counts += "; %d rows repeat an earlier row" % rc["duplicate_rows"]
    pe = det["printed_vs_exact"]
    value = "exact value %s/%s; printed %s %s" % (
        det["derived"]["num"], det["derived"]["den"], det["printed"],
        {"above": "rounds up", "below": "rounds down", "equal": "is exact"}[pe])
    pv = row["provenance"]
    so = det["source"]
    prov = "rows of entry %s: kind_of_source %s, rung %s, harness %s, complete %s; %d bytes" % (
        row["receipt"].partition("#")[0], pv["kind_of_source"], pv["rung"], canon(pv["harness"]),
        canon(so["complete"]), so["receipt_bytes"])
    if so["sibling_containers"]:
        prov += "; beside it: " + ", ".join("%s (%s of %d)" % (canon(s["at"]), s["type"], s["size"]) for s in so["siblings"])
    lines = [head, counts, value, prov]
    for c in det["census"]:
        vals = ", ".join("%s %d" % (canon(v["text"]) if v["type"] == "string" else v["text"] or v["type"], v["rows"])
                         for v in c["values"])
        line = _render_subject(c["subject"]) + ": " + (vals if vals else "no values")
        if c["other"]:
            line += ", other %d" % c["other"]
        if c["absent"]:
            line += ", absent %d" % c["absent"]
        for y in c["literals_never_seen"]:
            line += "; literal " + _render_lit(y) + " occurs in no row"
        lines.append(line)
    for j, wo in enumerate(det["without"]):
        dv = wo["derived"]
        lines.append("without test %d: %d rows selected, value %s" % (
            j + 1, wo["selected"], "undefined" if dv is None else dv["num"] + "/" + dv["den"]))
    return "\n".join(lines)


# ------------------------------------------------------------------ the result object
def check(claim_bytes, receipt_bytes):
    """-> the result object (a plain value) or {"refusal": code}."""
    try:
        claim = read_claim(claim_bytes)
    except Refusal as r:
        return {"refusal": r.code}
    row, spent = derive_span(claim, receipt_bytes)
    return {"profile": PROFILE, "row": row, "spent": spent, "surface": surface(row)}


def result_text(claim_bytes, receipt_bytes):
    return canon(check(claim_bytes, receipt_bytes))


def _expand_receipt(rv):
    """A vector's receipt: {"parts": [{"b64": ..., "repeat": n}]} -> bytes."""
    if not isinstance(rv, Obj):
        raise Refusal("vector_form")
    parts = rv.last.get("parts")
    if not isinstance(parts, Arr):
        raise Refusal("vector_form")
    out, size = [], 0
    for p in parts.items:
        if not isinstance(p, Obj) or not isinstance(p.last.get("b64"), str):
            raise Refusal("vector_form")
        chunk = base64.b64decode(p.last["b64"].encode("ascii"), validate=True)
        rep = _claim_int(p.last.get("repeat", Num("1")), 0)
        size += len(chunk) * rep
        if size > CAP_BATCH_BYTES:
            raise Refusal("vector_form")
        out.append(chunk * rep)
    return b"".join(out)


def batch_lines(vectors_bytes):
    """A vector file (bytes) -> one canonical line per vector: {"id", "out"}."""
    v = read_json(vectors_bytes.decode("utf-8"), Counter())
    cases = v.last["vectors"].items if isinstance(v, Obj) else v.items
    out = []
    for c in cases:
        cid = c.last["id"]
        if "claim_parts" in c.last:
            claim = _expand_receipt(c.last["claim_parts"])
        else:
            claim = base64.b64decode(c.last["claim_b64"].encode("ascii"), validate=True)
        receipt = _expand_receipt(c.last["receipt"])
        out.append(canon({"id": cid, "out": check(claim, receipt)}))
    return out


def main(argv):
    if len(argv) == 3 and argv[1] == "--batch":
        with open(argv[2], "rb") as fh:
            data = fh.read()
        sys.stdout.buffer.write(("\n".join(batch_lines(data)) + "\n").encode("utf-8"))
        return 0
    if len(argv) != 3:
        sys.stderr.write("usage: derive.py CLAIM_FILE RECEIPT_FILE | derive.py --batch VECTOR_FILE\n")
        return 64
    with open(argv[1], "rb") as fh:
        claim = fh.read()
    with open(argv[2], "rb") as fh:
        receipt = fh.read()
    res = check(claim, receipt)
    sys.stdout.buffer.write((canon(res) + "\n").encode("utf-8"))
    return 2 if "refusal" in res else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
