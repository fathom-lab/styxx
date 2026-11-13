# -*- coding: utf-8 -*-
"""RECEIPT v0 (DRAFT, unsigned): more hand vectors, read by build_receipt_v0_vectors.py.

Each entry carries the rows worked out by hand from RECEIPT_v0.md ([verdict, reason] per row); the
builder refuses to write a vector whose check_v0 rows differ, and computes the document verdict
from the rows by 9.2. Three groups:

  coverage   probes aimed at rules a mutation run over check_v0.py found unvectored (written by the
             lab's coverage attacker, 2026-11-13; one probe corrected: a '<' inside KIND is
             swallowed into the tag_syntax raw, so it gives tag_syntax then stray_closer);
  killers    one vector per mutant that the differential attacker's generator, vectors and probes
             let through (PM14, PM31, PM34, PM41, PM43, PM47, JM06, JM10, JM14, JM19 in its
             catalogue; JM06 is a canonical-form vector in the builder itself), and one each for
             two mutants of the coverage attacker's run that it took for equivalent (4.3, 6.5);
  pins       the rules the fixer's round added to the text: 7.2's Unicode version, 5.5's
             duplicate-key refusal before any field is read.

Lab-written. Vectors written by the lab for its own checkers are agreement inside one lab.
"""
import base64
import hashlib
import json


def sha(b):
    return hashlib.sha256(b).hexdigest()


def man(entries, rung="L2", digest=True, **extra):
    """A manifest over `entries`; an entry's "raw" becomes its bytes and default sha256. No digest
    key unless `digest` is a string."""
    rec = {}
    for k, e in entries.items():
        e = dict(e)
        if "raw" in e:
            raw = e.pop("raw")
            e.setdefault("sha256", sha(raw))
            e["bytes"] = base64.b64encode(raw).decode()
        e.setdefault("id", k)
        e.setdefault("kind_of_source", "tool_stdout")
        e.setdefault("captured_at", "2026-01-01T00:00:00Z")
        e.setdefault("complete", True)
        rec[k] = e
    m = {"spec": "sworn/manifest/0.2", "harness": "h", "turn": "t", "minted_at": "2026-01-01T00:00:00Z",
         "authored_sha256": [], "receipts": rec, "rung": rung}
    m.update(extra)
    if isinstance(digest, str):
        m["digest"] = digest
    return json.dumps(m).encode()


def doc(ref, kind, inner):
    if isinstance(inner, str):
        inner = inner.encode("utf-8")
    return b'<sworn r="' + ref.encode("utf-8") + b'" k="' + kind.encode() + b'">' + inner + b"</sworn>"


P = []


def p(label, d, expect, m=None, name="d.md", details=None, group="coverage"):
    P.append({"label": label, "doc": d, "manifest": m, "name": name, "rows": expect,
              "details": details or {}, "group": group})


MA = [["UNRESOLVED", "manifest_absent"]]
NR = [["UNRESOLVED", "no_repository"]]
RF = [["MALFORMED", "receipt_form"]]

# --- coverage: probes copied from the coverage attacker's gap_probes.py (rows worked by hand)
# 2.3d attribute bytes
p("2.3d '>' inside REF ends no opener: tag_syntax then stray_closer", b'<sworn r="a>b" k="numeric">7</sworn>', [["MALFORMED", "tag_syntax"], ["MALFORMED", "stray_closer"]])
p("2.3d CR inside REF: tag_syntax then stray_closer", b'<sworn r="r\r1" k="numeric">7</sworn>', [["MALFORMED", "tag_syntax"], ["MALFORMED", "stray_closer"]])
p("2.3d '<' inside KIND is swallowed into the tag_syntax raw", b'<sworn r="r1" k="nu<meric">7</sworn>', [["MALFORMED", "tag_syntax"], ["MALFORMED", "stray_closer"]])
# 4.1 first '#'
p("4.1 split at the FIRST '#': pointer token 'a#b' is well formed", doc("r1#/a#b", "numeric", "7"), MA)
p("4.1 first '#' resolves: key 'a#b'", doc("r1#/a#b", "numeric", "7"), [["HELD", None]], man({"r1": {"raw": b'{"a#b": 7, "a": {"b": 8}}'}}))
# 4.3
p("4.3 '~01' unescapes to '~1' (not '/')", doc("r1#/~01", "numeric", "5"), [["HELD", None]], man({"r1": {"raw": b'{"~1": 5, "/": 7}'}}))
p("4.3 L1-L1 (b == a) is well formed and resolves", doc("r1#L1-L1", "numeric", "7"), [["HELD", None]], man({"r1": {"raw": b"7\n8\n"}}))
p("4.3 '/' fragment names the key ''", doc("r1#/", "numeric", "7"), [["HELD", None]], man({"r1": {"raw": b'{"": 7}'}}))
# 4.4 banned path characters, one each
for cp in [0x00, 0x1F, 0x20, 0x7F, 0x85, 0xA0, 0x1680, 0x2000, 0x2005, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]:
    p("4.4 path: U+%04X refused" % cp, doc("path:a" + chr(cp) + "b", "numeric", "7"), RF)
for ch in "\\*?[]":
    p("4.4 path: %r refused" % ch, doc("path:a" + ch + "b", "numeric", "7"), RF)
for cp in [0x200B, 0x2030, 0x00A1, 0x21]:
    p("4.4 path: U+%04X allowed" % cp, doc("path:a" + chr(cp) + "b", "numeric", "7"), NR)
p("4.4 path: '...' segment allowed", doc("path:a/.../b", "numeric", "7"), NR)
p("4.4 path: '.x' segment allowed", doc("path:.x", "numeric", "7"), NR)
p("4.4 prereg: 63 hex refused", doc("prereg:" + "a" * 63, "numeric", "7"), RF)
p("4.4 prereg: 65 hex refused", doc("prereg:" + "a" * 65, "numeric", "7"), RF)
p("4.4 prereg: upper case 64 hex", doc("prereg:" + "A" * 64, "numeric", "7"), NR)
p("4.4 prereg: with a line fragment", doc("prereg:" + "a" * 64 + "#L1-L2", "numeric", "7"), NR)
p("4.4 prereg: with a pointer fragment", doc("prereg:" + "a" * 64 + "#/x", "quote", "`abc`"), NR)
p("4.4 prereg: hash over a fragment is hash_over_partial", doc("prereg:" + "a" * 64 + "#L1", "hash", "a" * 64), [["MALFORMED", "hash_over_partial"]])
# 6.2-6.4 order
good7 = {"raw": b"7"}
p("6.3 before 6.4: rung L3 and id missing", doc("r9", "numeric", "7"), [["UNRESOLVED", "rung_unknown"]], man({"r1": good7}, rung="L3"))
p("6.2 before 6.3: rung L3 and a wrong digest", doc("r1", "numeric", "7"), [["UNRESOLVED", "manifest_integrity"]], man({"r1": good7}, rung="L3", digest="0" * 64))
p("6.3 rung 'l2' (lower case) unknown", doc("r1", "numeric", "7"), [["UNRESOLVED", "rung_unknown"]], man({"r1": good7}, rung="l2"))
# 6.5 isolated from 6.10
p("6.5 upper-case sha256, no bytes, hash kind: manifest_integrity", doc("r1", "hash", "A" * 64), [["UNRESOLVED", "manifest_integrity"]], man({"r1": {"sha256": "A" * 64}}))
p("6.5 63-digit sha256, no bytes, numeric: manifest_integrity (not bytes_absent)", doc("r1", "numeric", "7"), [["UNRESOLVED", "manifest_integrity"]], man({"r1": {"sha256": "a" * 63}}))
p("6.5 entry an array", doc("r1", "numeric", "7"), [["UNRESOLVED", "manifest_integrity"]], json.dumps({"spec": "sworn/manifest/0.2", "harness": "h", "turn": "t", "minted_at": "x", "authored_sha256": [], "receipts": {"r1": [1]}, "rung": None}).encode())
# 6.11 CR kept, observable
p("6.11 CR kept: quote over L1 counts the CR in haystack_bytes", doc("r1#L1", "quote", "`cdefghijklmnop`"), [["HELD", None]], man({"r1": {"raw": b"abcdefghijklmnop\r\nq\n"}}))
p("6.11 CR kept: needle ending in CR found in a line slice", doc("r1#L1", "quote", "`nop\r`"), [["HELD", None]], man({"r1": {"raw": b"abcdefghijklmnop\r\nq\n"}}))
p("6.11 a range ending on the last line without LF", doc("r1#L2-L2", "numeric", "8"), [["HELD", None]], man({"r1": {"raw": b"7\n8"}}))
p("6.11 a range past the last line, receipt ending LF", doc("r1#L3", "numeric", "8"), [["MALFORMED", "anchor_out_of_range"]], man({"r1": {"raw": b"7\n8\n"}}))
p("6.11 '7 LF LF' has two lines, the second empty: L3 is anchor_out_of_range", doc("r1#L3", "numeric", "8"), [["MALFORMED", "anchor_out_of_range"]], man({"r1": {"raw": b"7\n\n"}}))
# 6.12
p("6.12 '-' is not an array index", doc("r1#/-", "numeric", "7"), [["MALFORMED", "pointer_unresolvable"]], man({"r1": {"raw": b"[7]"}}))
p("6.12 key present with a dup elsewhere at the same level is fine", doc("r1#/a", "numeric", "7"), [["HELD", None]], man({"r1": {"raw": b'{"a": 7, "b": 1, "b": 2}'}}))
# 7.1 order
p("7.1 b before c: 301 spaces is empty_span", doc("r1", "numeric", " " * 301), [["MALFORMED", "empty_span"]])
p("7.1 c before d: 301 code points with an unknown kind", doc("r1", "bogus", "x" * 301), [["MALFORMED", "length_cap"]])
p("7.1 e before f: absent over 'r1#'", doc("r1#", "absent", "`abc`"), RF)
p("7.1 f before g: absent with a fragment and U+202E", doc("r1#L1", "absent", "`abc` \u202e"), [["MALFORMED", "absent_over_partial"]])
p("7.1 g before h: override inside a bad number", doc("r1", "numeric", "\u202d1e5"), [["MALFORMED", "directional_override"]])
p("7.1 i before j: absent, no bytes, no complete", doc("r1", "absent", "`abc`"), [["UNRESOLVED", "manifest_bytes_absent"]], man({"r1": {"sha256": "b" * 64, "complete": None}}))
p("7.1b NBSP only is not empty: number_count", doc("r1", "numeric", "\u00a0"), [["MALFORMED", "number_count"]])
p("7.1b VT FF only is empty_span", doc("r1", "numeric", "\x0b\x0c"), [["MALFORMED", "empty_span"]])
# 7.2
DASHES = [0x00AD, 0x058A, 0x05BE, 0x1400, 0x1806, 0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2E17, 0x2E1A, 0x2E3A, 0x2E3B, 0x2E40, 0x2E5D, 0x301C, 0x3030, 0x30A0, 0xFE31, 0xFE32, 0xFE58, 0xFE63, 0xFF0D, 0x10EAD]
for cp in DASHES:
    p("7.2 dash U+%04X joins: number_grammar" % cp, doc("r1", "numeric", "12" + chr(cp) + "13"), [["MALFORMED", "number_grammar"]])
p("7.2 U+2043 HYPHEN BULLET is not a token char: number_count", doc("r1", "numeric", "12\u204313"), [["MALFORMED", "number_count"]])
p("7.2 a No digit (superscript two) is not digit-bearing", doc("r1", "numeric", "x\u00b2 is 5"), MA)
p("7.2 '42.,' strips both", doc("r1", "numeric", "is 42.,"), MA)
p("7.2 '1.2,3' number_grammar", doc("r1", "numeric", "1.2,3"), [["MALFORMED", "number_grammar"]])
p("7.2 '-.5' is a number", doc("r1", "numeric", "-.5"), MA)
p("7.2 '1,234,' strips the comma", doc("r1", "numeric", "1,234,"), MA)
p("7.2 '%42' number_grammar", doc("r1", "numeric", "%42"), [["MALFORMED", "number_grammar"]])
p("7.2 '42%:' strips ':'", doc("r1", "numeric", "42%:"), MA)
p("7.2 '+-4' number_grammar", doc("r1", "numeric", "+-4"), [["MALFORMED", "number_grammar"]])
p("7.2 '0,123' grouped", doc("r1", "numeric", "0,123"), MA)
# 7.4
p("7.4 a code span cannot cross a line: needle_count", doc("r1", "quote", "`abc\ndef`"), [["MALFORMED", "needle_count"]])
p("7.4 a double-backtick span holding a backtick is one needle", doc("r1", "quote", "``a`b``"), MA)
# 7.6
p("7.6 a 96-digit run: digest_form", doc("r1", "hash", "a" * 64 + " " + "b" * 96), [["MALFORMED", "digest_form"]])
p("7.6 a 128-digit run: digest_form", doc("r1", "hash", "a" * 64 + " " + "b" * 128), [["MALFORMED", "digest_form"]])
p("7.6 a 64 run beside e-acute counts", doc("r1", "hash", "\u00e9" + "a" * 64), MA)
p("7.6 a 65-digit run alone: digest_form", doc("r1", "hash", "a" * 65), [["MALFORMED", "digest_form"]])
p("7.6 a 64 run and a 63 run: passes", doc("r1", "hash", "a" * 64 + " " + "b" * 63), MA)
# 7.5 narrowing edge: receipt of exactly 16 bytes, L1 = all
p("7.5 receipt of 16 bytes, whole-content L1 does not narrow", doc("r1#L1", "quote", "`abc`"), [["MALFORMED", "short_needle"]], man({"r1": {"raw": b"abcdefghijklmnop"}}))
p("7.5 receipt of 15 bytes narrows under L1", doc("r1#L1", "quote", "`abc`"), [["HELD", None]], man({"r1": {"raw": b"abcdefghijklmno"}}))
p("7.5 a needle of exactly 16 bytes over the whole receipt", doc("r1", "quote", "`abcdefghijklmnop`"), [["HELD", None]], man({"r1": {"raw": b"xxabcdefghijklmnopxx"}}))
# 7.3
p("7.3 '+0' against 0", doc("r1", "numeric", "+0"), [["HELD", None]], man({"r1": {"raw": b"0"}}))
p("7.3 receipt 0.05 against 0.1: rounded at f=1, ties to even, it is 0.0: value_mismatch", doc("r1", "numeric", "0.1"), [["FAILED", "value_mismatch"]], man({"r1": {"raw": b"0.05"}}))
p("7.3 receipt 0.15 against 0.2 (half even up)", doc("r1", "numeric", "0.2"), [["HELD", None]], man({"r1": {"raw": b"0.15"}}))
p("7.3 receipt 0.25 against 0.2 (half even down)", doc("r1", "numeric", "0.2"), [["HELD", None]], man({"r1": {"raw": b"0.25"}}))
p("7.3 receipt -0.5 against 0: rounds_to_zero", doc("r1", "numeric", "0"), [["FAILED", "rounds_to_zero"]], man({"r1": {"raw": b"-0.5"}}))
p("7.3 receipt -0.5 against a printed -0: both are written unsigned as 0: rounds_to_zero", doc("r1", "numeric", "-0"), [["FAILED", "rounds_to_zero"]], man({"r1": {"raw": b"-0.5"}}))

# --- killers: one vector per mutant that nothing caught (the differential attacker's catalogue)
p("6.10 interior whitespace in base64 of a length that is a multiple of 4 (PM14)",
  doc("r1", "quote", "`ABC`"), [["UNRESOLVED", "manifest_integrity"]],
  man({"r1": {"sha256": sha(b"ABC"), "bytes": "QUJD\n\n\n\n"}}), group="killers")
p("6.13 a raw U+0001 inside a JSON string is not JSON (PM31)",
  doc("r1", "numeric", "7"), [["MALFORMED", "receipt_not_json"]],
  man({"r1": {"raw": b'"\x01"'}}), group="killers")
p("6.13 VT before a number is not JSON whitespace (JM14)",
  doc("r1", "numeric", "42"), [["MALFORMED", "receipt_not_json"]],
  man({"r1": {"raw": b"\x0b42"}}), group="killers")
p("7.1b VT alone is ASCII whitespace: empty_span (PM41)",
  doc("r1", "quote", "\x0b"), [["MALFORMED", "empty_span"]], man({"r1": {"raw": b"x"}}),
  group="killers")
p("7.4 a needle of VT only is needle_empty (PM41)",
  doc("r1", "quote", "`\x0b`"), [["MALFORMED", "needle_empty"]], man({"r1": {"raw": b"x"}}),
  group="killers")
p("7.5 a periodic needle counts occurrences without overlap (PM39, JM15)",
  doc("r1", "quote", "`" + "a" * 16 + "`"), [["HELD", None]], man({"r1": {"raw": b"a" * 20}}),
  details={0: {"needle_bytes": 16, "haystack_bytes": 20, "occurrences": 1}}, group="killers")
p("7.3 a receipt exponent of 4,001 digits: the receipt string keeps the adjusted exponent (PM47, JM19)",
  doc("r1", "numeric", "0"), [["FAILED", "rounds_to_zero"]],
  man({"r1": {"raw": b"15e-" + b"1" * 4001}}),
  details={0: {"receipt": "1.5E-" + "1" * 4000 + "0", "receipt_rounded": "0", "printed": "0"}},
  group="killers")
p("7.3 leaf text: the first 80 code points of a string of 81 astral characters (JM10)",
  doc("r1#/s", "numeric", "1"), [["MALFORMED", "leaf_not_numeric"]],
  man({"r1": {"raw": json.dumps({"s": "\U0001F600" * 81}, ensure_ascii=False).encode("utf-8")}}),
  details={0: {"leaf": "\U0001F600" * 80}}, group="killers")

p("7.2 the detail lists the digit-bearing tokens as found, trailing marks kept: number_grammar",
  doc("r1", "numeric", "v1.2."), [["MALFORMED", "number_grammar"]],
  details={0: {"digit_bearing_tokens": ["v1.2."]}}, group="killers")
p("7.2 the detail lists the digit-bearing tokens as found, trailing marks kept: number_count",
  doc("r1", "numeric", "1. and 2:"), [["MALFORMED", "number_count"]],
  details={0: {"digit_bearing_tokens": ["1.", "2:"]}}, group="killers")

# two mutants of the coverage attacker's run that it listed as equivalent and that are not (fixer,
# third start): a range end read by stripping every leading 'L', and an entry looked up by its id
p("4.3 a range end written 'LL3' is not L and a positive integer: receipt_form",
  doc("r1#L1-LL3", "numeric", "7"), RF, man({"r1": {"raw": b"7\n8\n9\n"}}), group="killers")
p("6.5 swapped ids: the key r1 names the entry whose own id is r2",
  doc("r1", "numeric", "7"), [["HELD", None]],
  man({"r1": {"raw": b"7", "id": "r2"}, "r2": {"raw": b"8", "id": "r1"}}), group="killers")

# --- pins: rules the text now states
SEVEN = man({"r1": {"raw": b"7"}})
p("7.2 Unicode 15.0.0: U+11BF0 (Nd only from Unicode 16.0) beside a 7 is no digit",
  doc("r1", "numeric", "n is 7 \U00011BF0"), [["HELD", None]], SEVEN,
  details={0: {"printed_token": "7"}}, group="pins")
p("7.2 Unicode 15.0.0: U+1C89 (a letter only from Unicode 16.0) glued to a 7 is no token character",
  doc("r1", "numeric", "n is 7Ᲊ"), [["HELD", None]], SEVEN,
  details={0: {"printed_token": "7"}}, group="pins")
p("7.2 Unicode 15.0.0: U+1CCF0 (Nd only from Unicode 16.0) alone is no digit: number_count",
  doc("r1", "numeric", "\U0001CCF0"), [["MALFORMED", "number_count"]], SEVEN,
  details={0: {"digit_bearing_tokens": []}}, group="pins")
p("7.2 Unicode 15.0.0: U+143E7 (assigned only from Unicode 16.0) before a 7",
  doc("r1", "numeric", "\U000143E7 7"), [["HELD", None]], SEVEN, group="pins")
p("7.2 Unicode 15.0.0: U+1C80 (a letter in 15.0.0) glued to a 7 joins the token: number_grammar",
  doc("r1", "numeric", "n is 7ᲀ"), [["MALFORMED", "number_grammar"]], SEVEN,
  details={0: {"digit_bearing_tokens": ["7ᲀ"]}}, group="pins")
p("7.2 Unicode 15.0.0: U+0663 (Nd in 15.0.0) is digit-bearing and not ASCII: number_grammar",
  doc("r1", "numeric", "n is ٣"), [["MALFORMED", "number_grammar"]], SEVEN,
  details={0: {"digit_bearing_tokens": ["٣"]}}, group="pins")
