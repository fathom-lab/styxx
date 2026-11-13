# -*- coding: utf-8 -*-
"""Writes derive_v0_vectors.json beside this file from the TEXT of DERIVE_v0.md.

It never imports derive.py or derive.js: each expectation below is written by hand from the rule it
names, with fractions.Fraction (standard library) for the arithmetic. An expectation is PARTIAL: it
names the fields the rule decides; the full objects the two implementations emit are compared with
each other by tests/test_derive_v0.py, which is agreement between two programs written by one agent
and is not evidence that either follows the text where no expectation reaches.

Matching (tests/test_derive_v0.py): a dict in an expectation is a subset of the output; a list must
have the same length and match element by element; {"$eq": v} must equal v exactly; {"$lacks": [k]}
inside a dict says those keys are absent; "surface_has" lists substrings of the surface text.

Usage: python -I make_vectors.py   (writes derive_v0_vectors.json; refuses to overwrite a file whose
content would change unless --force is given, so a regenerated vector set is a decision, not drift.)
"""
import base64
import json
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "derive_v0_vectors.json")

VECTORS = []
ENTRY = {"kind_of_source": "tool_stdout", "complete": True, "harness": "vector-harness", "rung": "L2"}
RES = {"verdict": "RESOLVED", "entry": ENTRY}
ROOT = {"format": "json", "at": ""}
BUDGET = 67108864


def B(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def claim(derivs=None, name="n", inner="1", ref=None, res=None, budget=None, ordinal=None, blocks=None):
    if blocks is None:
        blocks = [{"line": 1, "content": json.dumps(derivs, ensure_ascii=False)}]
    c = {"blocks": blocks, "span": {"receipt": ref if ref is not None else "r1#" + name, "inner": inner},
         "resolution": res if res is not None else RES}
    if ordinal is not None:
        c["span"]["ordinal"] = ordinal
    if budget is not None:
        c["budget"] = budget
    return json.dumps(c, ensure_ascii=False).encode("utf-8")


def parts(*chunks):
    """chunks: bytes, or (bytes, repeat)."""
    out = []
    for c in chunks:
        if isinstance(c, tuple):
            out.append({"b64": base64.b64encode(c[0]).decode("ascii"), "repeat": c[1]})
        else:
            out.append({"b64": base64.b64encode(c).decode("ascii"), "repeat": 1})
    return {"parts": out}


def V(rule, claim_bytes, receipt, expect, note=""):
    vid = "D%03d" % (len(VECTORS) + 1)
    if isinstance(receipt, (bytes, bytearray)):
        receipt = parts(bytes(receipt))
    rec = {"id": vid, "rule": rule, "note": note}
    if isinstance(claim_bytes, dict):
        rec["claim_parts"] = claim_bytes
    else:
        rec["claim_b64"] = base64.b64encode(claim_bytes).decode("ascii")
    rec["receipt"] = receipt
    rec["expect"] = expect
    VECTORS.append(rec)


def fr(x):
    x = Fraction(x)
    return {"num": str(x.numerator), "den": str(x.denominator)}


def held(**detail):
    return {"row": {"verdict": "HELD", "reason": None, "detail": detail}}


def failed(reason, **detail):
    return {"row": {"verdict": "FAILED", "reason": reason, "detail": detail}}


def mal(reason, detail=None, exact=True, prov=False):
    row = {"verdict": "MALFORMED", "reason": reason}
    if detail is not None:
        row["detail"] = {"$eq": detail} if exact else detail
    if not prov:
        row["$lacks"] = ["provenance", "resolved_sha256"]
    return {"row": row}


def refuse(code):
    return {"$eq": {"refusal": code}}


def cnt(where, **kw):
    d = {"rows": kw.pop("rows", ROOT), "where": where, "op": "count"}
    d.update(kw)
    return d


def shr(where, match, op="share", **kw):
    d = {"rows": kw.pop("rows", ROOT), "where": where, "match": match, "op": op}
    d.update(kw)
    return d


def sm(where, of, op="sum", **kw):
    d = {"rows": kw.pop("rows", ROOT), "where": where, "of": of, "op": op}
    d.update(kw)
    return d


# common receipts --------------------------------------------------------------------------------
SIX = B([{"g": "a", "x": 1}, {"g": "a", "x": 2}, {"g": "b", "x": 3}, {"g": "a"}, {"g": "b", "x": None}, {"g": "a", "x": 4}])
ANS = {}
for _i in range(130):
    ANS["E1-%03d" % _i] = ("CONTRADICTED" if _i % 10 in (0, 1) or _i in (3, 13, 23)
                           else ("UNSURE" if _i % 5 == 4 else "SUPPORTED"))
ANS_B = B(ANS)
ARM = [{"row": "key", "ge": "E1-000"}, {"row": "key", "le": "E1-099"}]
UP = [{"row": "value", "eq": "CONTRADICTED"}]
_arm = ["E1-%03d" % i for i in range(100)]
N_UP_ARM = sum(1 for k in _arm if ANS[k] == "CONTRADICTED")
N_UP_ALL = sum(1 for v in ANS.values() if v == "CONTRADICTED")
N_UNSURE_ARM = sum(1 for k in _arm if ANS[k] == "UNSURE")
N_SUP_ALL = sum(1 for v in ANS.values() if v == "SUPPORTED")
N_UNS_ALL = sum(1 for v in ANS.values() if v == "UNSURE")
N_UP_OUT = N_UP_ALL - N_UP_ARM

# ================================================================================ A. claim refusals
V("13.2 claim not JSON", b"{not json", b"[]", refuse("claim_form"))
V("13.2 claim lacks span", json.dumps({"blocks": [], "resolution": RES}).encode(), b"[]", refuse("claim_form"))
V("13.2 claim string with an unpaired surrogate (escape)", claim({"n": cnt([])}, inner="\\ud800 1").replace(b"\\\\ud800", b"\\ud800"), b"[]",
  refuse("claim_form"))
V("13.2 claim with a repeated key", b'{"blocks":[],"blocks":[],"span":{"receipt":"r1#n","inner":"1"},"resolution":{"verdict":"UNRESOLVED","reason":"manifest_absent"}}',
  b"[]", refuse("claim_form"))
V("13.2 claim not UTF-8", b'{"blocks":[],"span":{"receipt":"r1#n","inner":"\xff"},"resolution":{"verdict":"UNRESOLVED","reason":"manifest_absent"}}',
  b"[]", refuse("claim_form"))
V("13.2 claim with a BOM", b"\xef\xbb\xbf" + claim({"n": cnt([])}), b"[]", refuse("claim_form"))
V("13.2 unknown resolution verdict", claim({"n": cnt([])}, res={"verdict": "HELD"}), b"[]", refuse("claim_form"))
V("13.2 UNRESOLVED reason outside the list (no_repository cannot arise for rN)",
  claim({"n": cnt([])}, res={"verdict": "UNRESOLVED", "reason": "no_repository"}), b"[]", refuse("claim_form"))
V("13.2 negative budget", claim({"n": cnt([])}, budget={"spent_before": -1, "entry_read_before": False}), b"[]",
  refuse("claim_form"))
V("13.2 extra top-level key", json.dumps({"blocks": [], "span": {"receipt": "r1#n", "inner": "1"}, "resolution": RES, "x": 1}).encode(),
  b"[]", refuse("claim_form"))
V("13.2 rung L3 is reserved", claim({"n": cnt([])}, res={"verdict": "RESOLVED", "entry": dict(ENTRY, rung="L3")}), b"[]",
  refuse("claim_form"))
V("13.2 ordinal 0", claim({"n": cnt([])}, ordinal=0), b"[]", refuse("claim_form"))

# ================================================================================ B. the span, document only
V("9b empty inner", claim({"n": cnt([])}, inner=""), b"[]", mal("empty_span", {}))
V("9b inner of ASCII whitespace", claim({"n": cnt([])}, inner=" \t\n"), b"[]", mal("empty_span", {}))
V("9c 301 code points", claim({"n": cnt([])}, inner="1" + " x" * 150), b"[]",
  mal("length_cap", {"code_points": 301, "bytes": 301, "cap": 300}))
V("9c 300 code points of which some are two bytes", claim({"n": cnt([])}, inner="0" + " é" * 149 + " "), b"[]",
  held(derived=fr(0), rows={"total": 0, "selected": 0}))
V("9d the 257th derive span", claim({"n": cnt([])}, inner="0", ordinal=257), b"[]", mal("derive_span_cap", {"cap": 256}))
V("9d the 256th derive span is checked", claim({"n": cnt([])}, inner="0", ordinal=256), b"[]", held(derived=fr(0)))
V("2.1 REF without a fragment", claim({"n": cnt([])}, ref="r1"), b"[]", mal("receipt_form", {"receipt": "r1"}))
V("2.1 REF with a pointer fragment", claim({"n": cnt([])}, ref="r1#/a"), b"[]", mal("receipt_form", {"receipt": "r1#/a"}))
V("2.1 REF with a line fragment", claim({"n": cnt([])}, ref="r1#L1"), b"[]", mal("receipt_form", {"receipt": "r1#L1"}))
V("2.1 path: head is refused before any resolver (hermetic)", claim({"n": cnt([])}, ref="path:../../x#n"), b"[]",
  mal("receipt_form", {"receipt": "path:../../x#n"}))
V("2.1 prereg: head is refused", claim({"n": cnt([])}, ref="prereg:" + "a" * 64 + "#n"), b"[]",
  mal("receipt_form", {"receipt": "prereg:" + "a" * 64 + "#n"}))
V("2.1 r0", claim({"n": cnt([])}, ref="r0#n"), b"[]", mal("receipt_form", {"receipt": "r0#n"}))
V("2.1 r01", claim({"n": cnt([])}, ref="r01#n"), b"[]", mal("receipt_form", {"receipt": "r01#n"}))
V("2.1 R1", claim({"n": cnt([])}, ref="R1#n"), b"[]", mal("receipt_form", {"receipt": "R1#n"}))
V("2.1 NAME in upper case", claim({"N": cnt([])}, ref="r1#N"), b"[]", mal("receipt_form", {"receipt": "r1#N"}))
V("2.1 NAME of 33 characters", claim({"n": cnt([])}, ref="r1#a" + "b" * 32), b"[]",
  mal("receipt_form", {"receipt": "r1#a" + "b" * 32}))
V("2.1 NAME of 32 characters", claim({"a" + "b" * 31: cnt([])}, name="a" + "b" * 31, inner="0"), b"[]", held(derived=fr(0)))
V("2.1 empty fragment", claim({"n": cnt([])}, ref="r1#"), b"[]", mal("receipt_form", {"receipt": "r1#"}))
V("v0 7.1g U+202E in the inner text", claim({"n": cnt([])}, inner="‮1"), b"[]",
  mal("directional_override", {"code_point": "U+202E"}))
V("v0 7.2 two numbers", claim({"n": cnt([])}, inner="23 of 100"), b"[]",
  mal("number_count", {"digit_bearing_tokens": ["23", "100"]}))
V("v0 7.2 no number", claim({"n": cnt([])}, inner="none"), b"[]", mal("number_count", {"digit_bearing_tokens": []}))
V("v0 7.2 exponent form", claim({"n": cnt([])}, inner="1e-5"), b"[]", mal("number_grammar", {"digit_bearing_tokens": ["1e-5"]}))
V("v0 7.2 version string", claim({"n": cnt([])}, inner="v0.1"), b"[]", mal("number_grammar", {"digit_bearing_tokens": ["v0.1"]}))
V("v0 7.2 bad grouping", claim({"n": cnt([])}, inner="1,23"), b"[]", mal("number_grammar", {"digit_bearing_tokens": ["1,23"]}))
V("v0 7.2 a non-ASCII digit", claim({"n": cnt([])}, inner="٣"), b"[]", mal("number_grammar", {"digit_bearing_tokens": ["٣"]}))
V("7.3 19 places", claim({"n": cnt([])}, inner="0." + "1" * 19), b"[]", mal("places_cap", {"fractional_digits": 19, "cap": 18}))
V("7.3 38 integer digits", claim({"n": cnt([])}, inner="1" * 38), b"[]", mal("printed_too_long", {"integer_digits": 38, "cap": 37}))
V("7.3 printed 2^256 + 23 (78 digits) is refused, never wrapped", claim({"n": cnt([])}, inner=str(2 ** 256 + 23)), b"[]",
  mal("printed_too_long", {"integer_digits": 78, "cap": 37}))
V("7.3 37 integer digits are compared", claim({"n": cnt([])}, inner="1" * 37), b"[]",
  failed("value_mismatch", printed="1" * 37, derived_rounded="0"))
V("7.3 18 places are compared", claim({"n": cnt([])}, inner="0." + "0" * 18), b"[]",
  held(printed="0." + "0" * 18, derived_rounded="0." + "0" * 18, fractional_digits=18))
V("7.6 % with count is unit_mismatch", claim({"n": cnt([])}, inner="0%"), b"[]",
  mal("unit_mismatch", {"printed_token": "0%", "op": "count"}))
V("7.6 a 40% rate printed 0.4% with share (C11)", claim({"n": shr([], [{"field": "err", "eq": True}])}, inner="0.4%"),
  B([{"err": i < 40} for i in range(100)]), mal("unit_mismatch", {"printed_token": "0.4%", "op": "share"}))
V("7.6 % with percent holds", claim({"n": shr([], [{"field": "err", "eq": True}], op="percent")}, inner="40%"),
  B([{"err": i < 40} for i in range(100)]), held(derived=fr(40), printed_token="40%", printed="40"))
V("7.6 'percent' in words with op percent is allowed", claim({"n": shr([], [{"field": "err", "eq": True}], op="percent")}, inner="40 percent"),
  B([{"err": i < 40} for i in range(100)]), held(derived=fr(40), printed_token="40"))
V("7.6 % with sum is unit_mismatch", claim({"n": sm([], {"field": "x"})}, inner="6%"), SIX,
  mal("unit_mismatch", {"printed_token": "6%", "op": "sum"}))

# ================================================================================ C. the block table
_one = {"n": cnt([])}
V("2.4 the 17th block", claim(blocks=[{"line": 10 * k + 1, "content": json.dumps({"b%d" % k: cnt([])})} for k in range(17)], name="b0", inner="0"),
  b"[]", mal("derive_block_malformed", {"block_line": 161}))
V("2.4 16 blocks are read", claim(blocks=[{"line": 10 * k + 1, "content": json.dumps({"b%d" % k: cnt([])})} for k in range(16)], name="b15", inner="0"),
  b"[]", held(derived=fr(0)))
V("2.4 block content of 65,537 bytes", claim(blocks=[{"line": 3, "content": json.dumps(_one) + " " * (65537 - len(json.dumps(_one)))}], inner="0"),
  b"[]", mal("derive_block_malformed", {"block_line": 3}))
V("2.4 block content of 65,536 bytes", claim(blocks=[{"line": 3, "content": json.dumps(_one) + " " * (65536 - len(json.dumps(_one)))}], inner="0"),
  b"[]", held(derived=fr(0)))
V("2.4 a trailing comma", claim(blocks=[{"line": 5, "content": '{"n": {"rows": {"format": "json", "at": ""}, "where": [], "op": "count"},}'}], inner="0"),
  b"[]", mal("derive_block_malformed", {"block_line": 5}))
V("2.4 the block is an array", claim(blocks=[{"line": 2, "content": "[1]"}], inner="0"), b"[]", mal("derive_block_malformed", {"block_line": 2}))
V("2.4 a member name that is not a NAME", claim({"Bad": cnt([]), "n": cnt([])}, inner="0"), b"[]",
  mal("derive_block_malformed", {"block_line": 1}))
V("2.4 a NAME repeated in a later block", claim(blocks=[{"line": 1, "content": json.dumps(_one)}, {"line": 9, "content": json.dumps(_one)}], inner="0"),
  b"[]", mal("derive_block_malformed", {"block_line": 9}))
V("2.4 a NAME repeated in one block", claim(blocks=[{"line": 4, "content": '{"n": 1, "n": 2}'}], inner="0"), b"[]",
  mal("derive_block_malformed", {"block_line": 4}))
V("2.4 257 derivations", claim(blocks=[{"line": 1, "content": json.dumps(dict(("d%d" % k, 0) for k in range(257)))}], name="d0", inner="0"),
  b"[]", mal("derive_block_malformed", {"block_line": 1}))
V("2.4 256 derivations, values never examined", claim(blocks=[{"line": 1, "content": json.dumps(dict([("n", cnt([]))] + [("d%d" % k, 0) for k in range(255)]))}], inner="0"),
  b"[]", held(derived=fr(0)))
V("2.4 a block 32 containers deep reads", claim(blocks=[{"line": 1, "content": '{"n": %s, "z": %s}' % (json.dumps(cnt([])), "[" * 31 + "]" * 31)}], inner="0"),
  b"[]", held(derived=fr(0)))
V("2.4 a block 33 containers deep does not", claim(blocks=[{"line": 1, "content": '{"n": %s, "z": %s}' % (json.dumps(cnt([])), "[" * 32 + "]" * 32)}], inner="0"),
  b"[]", mal("derive_block_malformed", {"block_line": 1}))
V("2.6 NaN inside a derivation nobody names", claim(blocks=[{"line": 1, "content": '{"n": %s, "z": NaN}' % json.dumps(cnt([]))}], inner="0"),
  b"[]", held(derived=fr(0)))
V("2.5 a defective block after the named one still answers every span", claim(blocks=[{"line": 1, "content": json.dumps(_one)}, {"line": 7, "content": "{"}], inner="0"),
  b"[]", mal("derive_block_malformed", {"block_line": 7}))
V("2.4 block content starting with U+FEFF", claim(blocks=[{"line": 1, "content": "﻿" + json.dumps(_one)}], inner="0"), b"[]",
  mal("derive_block_malformed", {"block_line": 1}))
V("9e derivation_missing", claim({"m": cnt([])}, name="n", inner="3"), b"[]", mal("derivation_missing", {"name": "n"}))
V("9e no blocks at all", claim(blocks=[], inner="3"), b"[]", mal("derivation_missing", {"name": "n"}))

# ================================================================================ D. the derivation (3.6)
def D(rule, deriv_text, expect, inner="0", receipt=b"[]"):
    V(rule, claim(blocks=[{"line": 1, "content": '{"n": %s}' % deriv_text}], inner=inner), receipt, expect)


D("3.7 the derivation is an array", "[1]", mal("derivation_form", {}))
D("3.7 a float literal", '{"rows":{"format":"json","at":""},"where":[{"field":"x","eq":1.5}],"op":"count"}', mal("derivation_form", {}))
D("3.1 integer 2^53-1 is admitted", '{"rows":{"format":"json","at":""},"where":[{"field":"x","eq":9007199254740991}],"op":"count"}',
  held(derived=fr(1)), receipt=b'[{"x":9007199254740991},{"x":9007199254740990}]', inner="1")
D("3.1 integer 2^53 is not (S1)", '{"rows":{"format":"json","at":""},"where":[{"field":"x","eq":9007199254740992}],"op":"count"}', mal("derivation_form", {}))
D("3.1 integer 9007199254740993 is not", '{"rows":{"format":"json","at":""},"where":[{"field":"x","eq":9007199254740993}],"op":"count"}', mal("derivation_form", {}))
D("3.1 a 5000-digit integer is refused on its digits, no crash", '{"rows":{"format":"json","at":""},"where":[{"field":"x","eq":%s}],"op":"count"}' % ("9" * 5000),
  mal("derivation_form", {}))
D("3.1 -0 has no canonical form", '{"rows":{"format":"json","at":""},"where":[{"field":"x","eq":-0}],"op":"count"}', mal("derivation_form", {}))
D("3.1 1.0 has no canonical form", '{"rows":{"format":"json","at":""},"where":[{"field":"x","eq":1.0}],"op":"count"}', mal("derivation_form", {}))
D("3.7 size is checked before op (G07): 4,097 canonical bytes with op median",
  '{"rows":{"format":"json","at":""},"where":[],"op":"median","x":"%s"}' % ("a" * (4097 - len('{"op":"median","rows":{"at":"","format":"json"},"where":[],"x":""}'))),
  mal("derivation_form", {}))
D("3.7 4,096 canonical bytes pass the size rule; op is checked before the other keys",
  '{"rows":{"format":"json","at":""},"where":[],"op":"bogus","x":"%s"}' % ("a" * (4096 - len('{"op":"bogus","rows":{"at":"","format":"json"},"where":[],"x":""}'))),
  mal("op_unknown", {"op": "bogus"}))
D("3.7 op missing", '{"rows":{"format":"json","at":""},"where":[]}', mal("derivation_form", {}))
D("3.7 op is a number", '{"rows":{"format":"json","at":""},"where":[],"op":5}', mal("derivation_form", {}))
D("3.7 op median", '{"rows":{"format":"json","at":""},"where":[],"op":"median"}', mal("op_unknown", {"op": "median"}))
D("3.7 op constructor (no host lookup, S4)", '{"rows":{"format":"json","at":""},"where":[],"op":"constructor"}', mal("op_unknown", {"op": "constructor"}))
D("3.7 op toString", '{"rows":{"format":"json","at":""},"where":[],"op":"toString"}', mal("op_unknown", {"op": "toString"}))
D("3.7 op __proto__", '{"rows":{"format":"json","at":""},"where":[],"op":"__proto__"}', mal("op_unknown", {"op": "__proto__"}))
D("3.7 op is carried whole, never cut by UTF-16 units (S5)", '{"rows":{"format":"json","at":""},"where":[],"op":"%s"}' % ("x" * 63 + "\U0001F600" + "y"),
  mal("op_unknown", {"op": "x" * 63 + "\U0001F600" + "y"}))
D("3.7 op max is not in the vocabulary", '{"rows":{"format":"json","at":""},"where":[],"of":{"field":"x"},"op":"max"}', mal("op_unknown", {"op": "max"}))
D("3.7 an extra key round", '{"rows":{"format":"json","at":""},"where":[],"op":"count","round":2}', mal("derivation_form", {}))
D("3.7 count with match", '{"rows":{"format":"json","at":""},"where":[],"match":[{"row":"value","eq":1}],"op":"count"}', mal("derivation_form", {}))
D("3.7 share without match", '{"rows":{"format":"json","at":""},"where":[],"op":"share"}', mal("derivation_form", {}))
D("3.7 share with an empty match", '{"rows":{"format":"json","at":""},"where":[],"match":[],"op":"share"}', mal("derivation_form", {}))
D("3.3 nine tests", json.dumps(cnt([{"row": "value", "eq": k} for k in range(9)])), mal("derivation_form", {}))
D("3.3 eight tests", json.dumps(cnt([{"row": "value", "eq": k} for k in range(8)])), held(derived=fr(0), rows={"total": 0}))
D("3.4 in with 17 literals", json.dumps(cnt([{"row": "value", "in": list(range(17))}])), mal("derivation_form", {}))
D("3.4 in with 16 literals", json.dumps(cnt([{"row": "value", "in": list(range(16))}])), held(derived=fr(0)))
D("3.4 in with two equal literals", json.dumps(cnt([{"row": "value", "in": ["a", "a"]}])), mal("derivation_form", {}))
D("3.4 a string literal of 257 bytes", json.dumps(cnt([{"row": "value", "eq": "a" * 257}])), mal("derivation_form", {}))
D("3.4 a string literal of 256 bytes", json.dumps(cnt([{"row": "value", "eq": "a" * 256}])), held(derived=fr(0)))
D("3.4 a field name of 258 bytes written as 129 escaped e-acute (bytes of the decoded string, F5)",
  json.dumps(cnt([{"field": "é" * 129, "eq": 1}])), mal("derivation_form", {}))
D("3.4 a field name of 256 bytes written with escapes", json.dumps(cnt([{"field": "é" * 128, "eq": 1}])),
  held(derived=fr(0)))
D("3.4 present with a row subject", json.dumps(cnt([{"row": "value", "present": True}])), mal("derivation_form", {}))
D("3.4 ge with a boolean", json.dumps(cnt([{"row": "value", "ge": True}])), mal("derivation_form", {}))
D("3.4 a test with two subjects", json.dumps(cnt([{"row": "value", "field": "x"}])), mal("derivation_form", {}))
D("3.4 a test with three members", json.dumps(cnt([{"row": "value", "eq": 1, "le": 2}])), mal("derivation_form", {}))
D("3.4 relation ne", json.dumps(cnt([{"row": "value", "ne": 1}])), mal("derivation_form", {}))
D("3.2 format csv", json.dumps({"rows": {"format": "csv"}, "where": [], "op": "count"}), mal("derivation_form", {}))
D("3.2 json without at", json.dumps({"rows": {"format": "json"}, "where": [], "op": "count"}), mal("derivation_form", {}))
D("3.2 a pointer without a leading slash", json.dumps({"rows": {"format": "json", "at": "a"}, "where": [], "op": "count"}), mal("derivation_form", {}))
D("3.2 a pointer with ~2", json.dumps({"rows": {"format": "json", "at": "/a~2"}, "where": [], "op": "count"}), mal("derivation_form", {}))
D("3.2 a pointer of 17 tokens", json.dumps({"rows": {"format": "json", "at": "/a" * 17}, "where": [], "op": "count"}), mal("derivation_form", {}))
D("3.2 a pointer of 1,025 bytes", json.dumps({"rows": {"format": "json", "at": "/" + "a" * 1024}, "where": [], "op": "count"}), mal("derivation_form", {}))
D("3.2 jsonl with at", json.dumps({"rows": {"format": "jsonl", "at": ""}, "where": [], "op": "count"}), mal("derivation_form", {}))
D("3.4 jsonl with a row key subject is refused from the document alone (G13)",
  json.dumps({"rows": {"format": "jsonl"}, "where": [{"row": "key", "eq": "a"}], "op": "count"}), mal("derivation_form", {}),
  receipt=b"not json\n")
D("3.5 of row key", json.dumps(sm([], {"row": "key"})), mal("derivation_form", {}))
D("3.5 of with two members", json.dumps(sm([], {"field": "x", "row": "value"})), mal("derivation_form", {}))
D("3.4 dec 00.5", json.dumps(cnt([{"row": "value", "eq": {"dec": "00.5"}}])), mal("derivation_form", {}))
D("3.4 dec 1e3", json.dumps(cnt([{"row": "value", "eq": {"dec": "1e3"}}])), mal("derivation_form", {}))
D("3.4 dec with 19 places", json.dumps(cnt([{"row": "value", "eq": {"dec": "0." + "1" * 19}}])), mal("derivation_form", {}))
D("3.4 dec with 19 integer digits", json.dumps(cnt([{"row": "value", "eq": {"dec": "1" * 19}}])), mal("derivation_form", {}))
D("3.4 dec with an extra key", json.dumps(cnt([{"row": "value", "eq": {"dec": "1", "x": 1}}])), mal("derivation_form", {}))
D("3.4 dec 1. (no fraction digits)", json.dumps(cnt([{"row": "value", "eq": {"dec": "1."}}])), mal("derivation_form", {}))
D("3.6 within that is not a NAME", json.dumps(cnt([], within="N")), mal("derivation_form", {}))
D("3.1 an escaped unpaired surrogate in a literal", '{"rows":{"format":"json","at":""},"where":[{"row":"value","eq":"\\ud800"}],"op":"count"}',
  mal("derivation_form", {}))
D("3.1 a non-ASCII key", '{"rows":{"format":"json","at":""},"where":[],"op":"count","é":1}', mal("derivation_form", {}))
D("3.1 NaN in the named derivation", '{"rows":{"format":"json","at":""},"where":[{"row":"value","eq":NaN}],"op":"count"}', mal("derivation_form", {}))
D("3.1 a repeated key inside the derivation", '{"rows":{"format":"json","at":"","at":""},"where":[],"op":"count"}', mal("derivation_form", {}))
D("3.4 an empty at subject", json.dumps(cnt([{"at": "", "eq": 1}])), mal("derivation_form", {}))
D("3.4 at subject with a bad escape", json.dumps(cnt([{"at": "/a~", "eq": 1}])), mal("derivation_form", {}))
D("3.1 a key escaped as \\u0072ows is the key rows", '{"\\u0072ows":{"format":"json","at":""},"where":[],"op":"count"}', held(derived=fr(0)))

# ================================================================================ W. within
_W = {"k": cnt(UP, within="arm"), "arm": cnt(ARM)}
V("4.5 within names itself", claim({"n": cnt([], within="n")}, inner="0"), b"[]", mal("within_form", {"within": "n"}))
V("4.5 within names a missing derivation", claim({"n": cnt([], within="zz")}, inner="0"), b"[]", mal("within_form", {"within": "zz"}))
V("4.5 within names a malformed derivation", claim({"n": cnt([], within="z"), "z": {"op": "count"}}, inner="0"), b"[]", mal("within_form", {"within": "z"}))
V("4.5 within names a derivation that has within", claim({"n": cnt([], within="z"), "z": cnt([], within="y"), "y": cnt([])}, inner="0"), b"[]",
  mal("within_form", {"within": "z"}))
V("4.5 within over different rows", claim({"n": cnt([], within="z"), "z": cnt([], rows={"format": "json", "at": "/a"})}, inner="0"), b"[]",
  mal("within_form", {"within": "z"}))
V("4.5 k of n whose selections do not nest (C03)", claim(_W, name="k", inner=str(N_UP_ALL)), ANS_B,
  failed("not_within", derived=fr(N_UP_ALL), rows={"total": 130, "selected": N_UP_ALL, "within": 100, "outside_within": N_UP_OUT}),
  note="C03: the CONTRADICTED rows outside the id range are counted in k but not in n")
V("4.5 k of n that nest", claim({"k": cnt(ARM + UP, within="arm"), "arm": cnt(ARM)}, name="k", inner=str(N_UP_ARM)), ANS_B,
  held(derived=fr(N_UP_ARM), rows={"selected": N_UP_ARM, "within": 100, "outside_within": 0}))
V("4.5 within with an empty selection list selects every row", claim({"k": cnt(UP, within="all"), "all": cnt([])}, name="k", inner=str(N_UP_ALL)), ANS_B,
  held(rows={"within": 130, "outside_within": 0}))
V("4.5 within on a share", claim({"p": shr(ARM, UP, within="arm"), "arm": cnt(ARM)}, name="p", inner="0.23"), ANS_B,
  held(derived=fr(Fraction(N_UP_ARM, 100)), rows={"within": 100, "outside_within": 0}))
V("7.5 a value mismatch is reported before not_within", claim(_W, name="k", inner="1"), ANS_B, failed("value_mismatch"))

# ================================================================================ R. resolution
V("9f UNRESOLVED from v0 6.4", claim({"n": cnt([])}, inner="0", res={"verdict": "UNRESOLVED", "reason": "manifest_id_missing"}), b"[]",
  {"row": {"verdict": "UNRESOLVED", "reason": "manifest_id_missing", "detail": {"$eq": {}}, "$lacks": ["provenance", "resolved_sha256"]}})
V("9f UNRESOLVED manifest_bytes_absent", claim({"n": cnt([])}, inner="0", res={"verdict": "UNRESOLVED", "reason": "manifest_bytes_absent"}), b"[]",
  {"row": {"verdict": "UNRESOLVED", "reason": "manifest_bytes_absent"}})
V("9f MALFORMED receipt_author_minted from v0 6.7", claim({"n": cnt([])}, inner="0", res={"verdict": "MALFORMED", "reason": "receipt_author_minted"}),
  b"[]", mal("receipt_author_minted", {}))
V("9e a document-only MALFORMED comes before resolution", claim({"n": {"op": "count"}}, inner="0", res={"verdict": "UNRESOLVED", "reason": "manifest_absent"}),
  b"[]", mal("derivation_form", {}))
V("9f receipt bytes are not read when unresolved", claim({"n": cnt([])}, inner="0", res={"verdict": "UNRESOLVED", "reason": "manifest_absent"}),
  b"not json", {"row": {"verdict": "UNRESOLVED", "reason": "manifest_absent"}, "spent": 0})

# ================================================================================ S. the row source
V("4.1 16,777,217 bytes", claim({"n": cnt([])}, inner="0"), parts((b" ", 16777217)),
  mal("rows_too_large", {"bytes": 16777217, "cap": 16777216}))
V("4.1 16,777,216 bytes are read", claim({"n": cnt([])}, inner="1"), parts(b'["', (b"a", 16777212), b'"]'),
  held(derived=fr(1), source={"receipt_bytes": 16777216}))
V("4.2 not JSON", claim({"n": cnt([])}), b"[1,", mal("receipt_not_json", {}))
V("4.2 not UTF-8", claim({"n": cnt([])}), b'["\xff"]', mal("receipt_not_json", {}))
V("4.2 a byte order mark is not stripped (S4)", claim({"n": cnt([])}), b"\xef\xbb\xbf[1]", mal("receipt_not_json", {}))
V("4.2 CESU-8 surrogate bytes are not UTF-8 (S4)", claim({"n": cnt([])}), b'["\xed\xa0\x80\xed\xb0\x80"]', mal("receipt_not_json", {}))
V("4.2 an overlong encoding", claim({"n": cnt([])}), b'["\xc0\xaf"]', mal("receipt_not_json", {}))
V("4.2 33 containers deep", claim({"n": cnt([])}), b"[" * 33 + b"]" * 33, mal("receipt_not_json", {}))
V("4.2 32 containers deep reads", claim({"n": cnt([])}, inner="1"), b"[" * 32 + b"]" * 32, held(derived=fr(1)))
V("4.2 16 MiB of [ stops at the 33rd (S2)", claim({"n": cnt([])}), parts((b"[", 16777216)), mal("receipt_not_json", {}))
V("4.2 1,000,001 values (S2)", claim({"n": cnt([])}), parts(b"[", (b"0,", 1000000), b"0]"),
  mal("receipt_too_many_values", {"cap": 1000000}))
V("4.2 1,000,000 values read, then the row cap", claim({"n": cnt([])}), parts(b"[", (b"0,", 999998), b"0]"),
  mal("rows_too_many", {"rows": 999999, "cap": 100000}))
V("4.2 a large filler outside the named container counts (S2)", claim({"n": cnt([], rows={"format": "json", "at": "/rows"})}),
  parts(b'{"filler":[', (b"0,", 1000000), b'0],"rows":[1]}'), mal("receipt_too_many_values", {"cap": 1000000}))
V("4.2 a reader error before the cap is reported first", claim({"n": cnt([])}), parts(b"[01,", (b"0,", 1000000), b"0]"),
  mal("receipt_not_json", {}))
V("4.2 a missing member", claim({"n": cnt([], rows={"format": "json", "at": "/x"})}), b'{"a":[]}', mal("pointer_unresolvable", {"at": "/x"}))
V("4.2 an index past the end", claim({"n": cnt([], rows={"format": "json", "at": "/1"})}), b"[[]]", mal("pointer_unresolvable", {"at": "/1"}))
V("4.2 an index with a leading zero", claim({"n": cnt([], rows={"format": "json", "at": "/01"})}), b"[[],[]]", mal("pointer_unresolvable", {"at": "/01"}))
V("4.2 the token length on an array (no host lookup, S4)", claim({"n": cnt([], rows={"format": "json", "at": "/length"})}), b"[[]]",
  mal("pointer_unresolvable", {"at": "/length"}))
V("4.2 a walk through a scalar", claim({"n": cnt([], rows={"format": "json", "at": "/a/b"})}), b'{"a":1}', mal("pointer_unresolvable", {"at": "/a/b"}))
V("4.2 a repeated key on the path", claim({"n": cnt([], rows={"format": "json", "at": "/a"})}), b'{"a":[],"a":[]}', mal("pointer_ambiguous", {"at": "/a"}))
V("4.2 an escaped repeated key on the path (S4)", claim({"n": cnt([], rows={"format": "json", "at": "/a"})}), b'{"a":[],"\\u0061":[]}',
  mal("pointer_ambiguous", {"at": "/a"}))
V("4.2 ~1 and ~0 in a pointer", claim({"n": cnt([], rows={"format": "json", "at": "/a~1b/c~0d"})}, inner="2"), b'{"a/b":{"c~d":[1,2]}}',
  held(derived=fr(2), source={"pointer_tokens": 2, "parent_members": 1}))
V("4.2 the target is a number", claim({"n": cnt([], rows={"format": "json", "at": "/k"})}), b'{"k":182}', mal("rows_not_container", {"type": "number"}))
V("4.2 the target is a string", claim({"n": cnt([])}), b'"x"', mal("rows_not_container", {"type": "string"}))
V("4.2 the target is null", claim({"n": cnt([])}), b"null", mal("rows_not_container", {"type": "null"}))
V("4.2 the target is NaN", claim({"n": cnt([])}), b"NaN", mal("rows_not_container", {"type": "non_finite"}))
V("4.2 the target is -Infinity", claim({"n": cnt([])}), b"-Infinity", mal("rows_not_container", {"type": "non_finite"}))
V("4.2 repeated member names in the rows object", claim({"n": cnt([])}), b'{"a":"X","a":"Y","b":"X"}', mal("rows_ambiguous", {}))
V("4.2 100,001 rows", claim({"n": cnt([])}), parts(b"[", (b"0,", 100000), b"0]"), mal("rows_too_many", {"rows": 100001, "cap": 100000}))
V("4.2 100,000 rows", claim({"n": cnt([])}, inner="100,000"), parts(b"[", (b"0,", 99999), b"0]"), held(derived=fr(100000)))
V("4.2 the row cap before repeated names (G12)", claim({"n": cnt([])}),
  b"{" + b",".join(b'"k%d":0' % (k % 100000) for k in range(100001)) + b"}", mal("rows_too_many", {"rows": 100001, "cap": 100000}))
V("4.6 a row key test over an array", claim({"n": cnt([{"row": "key", "eq": "a"}])}), b"[1]", mal("row_key_unavailable", {}))
V("4.6 row_key_unavailable before the visit, whatever the rows hold", claim({"n": cnt([{"row": "key", "eq": "a"}, {"field": "x", "le": 1}])}),
  b'[{"x":"s"}]', mal("row_key_unavailable", {}))
_JL = {"rows": {"format": "jsonl"}, "where": [], "op": "count"}
V("4.3 JSON Lines with an empty second line", claim({"n": _JL}), b'{"a":1}\n\n{"a":2}\n', mal("receipt_not_jsonl", {"line": 2}))
V("4.3 a line of spaces and CR", claim({"n": _JL}), b'1\n \t\r\n2\n', mal("receipt_not_jsonl", {"line": 2}))
V("4.3 a trailing LF is not a line", claim({"n": _JL}, inner="2"), b"1\n2\n", held(derived=fr(2), source={"format": "jsonl", "pointer_tokens": 0}))
V("4.3 no trailing LF", claim({"n": _JL}, inner="2"), b"1\n2", held(derived=fr(2)))
V("4.3 CR LF line ends", claim({"n": _JL}, inner="2"), b"1\r\n2\r\n", held(derived=fr(2)))
V("4.3 zero bytes are zero rows", claim({"n": _JL}, inner="0"), b"", held(derived=fr(0), rows={"total": 0}))
V("4.3 one LF is one empty line", claim({"n": _JL}), b"\n", mal("receipt_not_jsonl", {"line": 1}))
V("4.3 a bad byte on line 3", claim({"n": _JL}), b"1\n2\n\xff\n", mal("receipt_not_jsonl", {"line": 3}))
V("4.3 a line that does not read", claim({"n": _JL}), b"1\n[1,\n", mal("receipt_not_jsonl", {"line": 2}))
V("4.3 two values on one line", claim({"n": _JL}), b"1 2\n", mal("receipt_not_jsonl", {"line": 1}))
V("4.3 100,001 lines are refused before any line is read", claim({"n": _JL}), parts((b"x\n", 100001)), mal("rows_too_many", {"rows": 100001, "cap": 100000}))
V("4.3 a BOM on the first line", claim({"n": _JL}), b"\xef\xbb\xbf1\n", mal("receipt_not_jsonl", {"line": 1}))
V("4.3 a line 33 containers deep", claim({"n": _JL}), b"1\n" + b"[" * 33 + b"]" * 33 + b"\n", mal("receipt_not_jsonl", {"line": 2}))

# ================================================================================ U. the visit
V("5.1 a row repeats the named field", claim({"n": cnt([{"field": "a", "eq": 1}])}), b'[{"a":1},{"a":1,"a":2}]',
  mal("field_ambiguous", {"row": 1, "subject": {"field": "a"}}, prov=True))
V("5.1 a repeat written with an escape (S4)", claim({"n": cnt([{"field": "a", "eq": 1}])}), b'[{"a":1,"\\u0061":2}]',
  mal("field_ambiguous", {"row": 0, "subject": {"field": "a"}}, prov=True))
V("5.1 a row repeats a field nobody names", claim({"n": cnt([{"field": "a", "eq": 1}])}, inner="1"), b'[{"a":1,"b":1,"b":2}]', held(derived=fr(1)))
V("5.2 lookups come before tests on a row (field_ambiguous on the within subject first)",
  claim({"n": cnt([{"field": "a", "le": 1}], within="w"), "w": cnt([{"field": "b", "eq": 1}])}), b'[{"a":"s","b":1,"b":2}]',
  mal("field_ambiguous", {"row": 0, "subject": {"field": "b"}}, prov=True))
V("7.1 number_out_of_range when compared with an integer", claim({"n": cnt([{"field": "x", "eq": 1}])}), b'[{"x":1},{"x":1e30}]',
  mal("number_out_of_range", {"row": 1, "number": "1e30"}, prov=True))
V("7.1 1e-19 summed", claim({"n": sm([], {"field": "x"})}), b'[{"x":1e-19}]', mal("number_out_of_range", {"row": 0, "number": "1e-19"}, prov=True))
V("7.1 a number never converted is never refused", claim({"n": cnt([{"field": "s", "eq": "a"}])}, inner="1"), b'[{"s":"a","x":1e400}]', held(derived=fr(1)))
V("7.1 1E400 compared", claim({"n": cnt([{"field": "x", "eq": 1}])}), b'[{"x":1E400}]', mal("number_out_of_range", {"row": 0, "number": "1E400"}, prov=True))
V("7.1 the number detail keeps 80 characters", claim({"n": cnt([{"field": "x", "eq": 1}])}), b'[{"x":' + b"1" * 100 + b"}]",
  mal("number_out_of_range", {"row": 0, "number": "1" * 80}, prov=True))
V("5.3 ge with a string literal on a number", claim({"n": cnt([{"field": "x", "ge": "a"}])}), b'[{"x":"b"},{"x":3}]',
  mal("where_type_mismatch", {"row": 1, "type": "number", "literal_type": "string"}, prov=True))
V("5.3 le with an integer on a string (X19)", claim({"n": cnt([{"field": "layer", "le": 5}])}), b'[{"layer":"tree"}]',
  mal("where_type_mismatch", {"row": 0, "type": "string", "literal_type": "number"}, prov=True))
V("5.3 ge with a dec on a boolean", claim({"n": cnt([{"field": "x", "ge": {"dec": "0.5"}}])}), b'[{"x":true}]',
  mal("where_type_mismatch", {"row": 0, "type": "boolean", "literal_type": "number"}, prov=True))
V("5.3 ge on NaN", claim({"n": cnt([{"field": "x", "ge": 0}])}), b'[{"x":NaN}]',
  mal("where_type_mismatch", {"row": 0, "type": "non_finite", "literal_type": "number"}, prov=True))
V("5.2 every test reads every row: a mismatch on a row the first test excludes", claim({"n": cnt([{"field": "g", "eq": "a"}, {"field": "x", "le": 1}])}),
  b'[{"g":"b","x":"s"}]', mal("where_type_mismatch", {"row": 0, "type": "string", "literal_type": "number"}, prov=True))
V("6 sum over a string (X14)", claim({"n": sm([], {"field": "x"})}), b'[{"x":3},{"x":"12"},{"x":4}]',
  mal("field_not_numeric", {"row": 1, "type": "string"}, prov=True))
V("6 sum over a boolean", claim({"n": sm([], {"field": "x"})}), b'[{"x":true}]', mal("field_not_numeric", {"row": 0, "type": "boolean"}, prov=True))
V("6 sum over NaN", claim({"n": sm([], {"field": "x"})}), b'[{"x":NaN}]', mal("field_not_numeric", {"row": 0, "type": "non_finite"}, prov=True))
V("6 of is read on rows that are not selected", claim({"n": sm([{"field": "g", "eq": "a"}], {"field": "x"})}), b'[{"g":"a","x":1},{"g":"b","x":"n/a"}]',
  mal("field_not_numeric", {"row": 1, "type": "string"}, prov=True))
V("5.5 a misspelt field (X13)", claim({"n": cnt([{"field": "verdcit", "eq": "survived"}])}), B([{"verdict": "killed"}] * 57),
  mal("field_never_present", {"subject": {"field": "verdcit"}, "rows_total": 57}, prov=True))
V("5.5 present on a misspelt field is guarded too (F8, C06b)", claim({"n": cnt([{"field": "eror", "present": True}])}),
  B([{"run": i, "error": "x"} if i % 7 == 0 else {"run": i} for i in range(50)]),
  mal("field_never_present", {"subject": {"field": "eror"}, "rows_total": 50}, prov=True))
V("5.5 present false on a misspelt field", claim({"n": cnt([{"field": "eror", "present": False}])}), b'[{"a":1}]',
  mal("field_never_present", {"subject": {"field": "eror"}, "rows_total": 1}, prov=True))
V("5.5 of names a field no row has", claim({"n": sm([], {"field": "y"})}), b'[{"x":1}]',
  mal("field_never_present", {"subject": {"field": "y"}, "rows_total": 1}, prov=True))
V("5.5 an at subject that resolves in no row", claim({"n": cnt([{"at": "/m/acc", "ge": 0}])}), b'[{"m":{"ac":1}}]',
  mal("field_never_present", {"subject": {"at": "/m/acc"}, "rows_total": 1}, prov=True))
V("5.5 the guard order is where, match, of, within", claim({"n": shr([{"field": "p", "eq": 1}], [{"field": "q", "eq": 1}])}), b'[{"z":1}]',
  mal("field_never_present", {"subject": {"field": "p"}, "rows_total": 1}, prov=True))
V("5.5 a field present in one row only passes the guard", claim({"n": cnt([{"field": "x", "eq": 1}])}, inner="1"), b'[{"x":1},{},{}]',
  held(derived=fr(1), where_steps=[{"passed": 1, "no_value": 2}]))
V("5.5 zero rows: the guard does not apply", claim({"n": cnt([{"field": "x", "eq": 1}])}, inner="0"), b"[]", held(derived=fr(0), rows={"total": 0}))
V("6 share over no rows", claim({"n": shr([], [{"field": "x", "eq": 1}])}, inner="0.5"), b"[]",
  {"row": {"verdict": "MALFORMED", "reason": "division_by_zero",
           "detail": {"rows": {"$eq": {"total": 0, "selected": 0, "used": 0, "matched": 0, "duplicate_rows": 0}},
                      "where_steps": [], "match_steps": [{"passed": 0, "no_value": 0}], "without": [],
                      "$lacks": ["derived", "derived_rounded", "exact_tie", "printed_vs_exact", "rounding"]}}})
V("6 share over an empty selection (X12)", claim({"n": shr([{"field": "g", "eq": "z"}], [{"field": "x", "eq": 1}])}, inner="0.50"), SIX,
  {"row": {"verdict": "MALFORMED", "reason": "division_by_zero",
           "detail": {"rows": {"$eq": {"total": 6, "selected": 0, "used": 0, "matched": 0, "duplicate_rows": 0}},
                      "where_steps": [{"passed": 0, "no_value": 0}], "printed": "0.50",
                      "without": [{"selected": 6, "matched": 1, "derived": fr(Fraction(1, 6))}]}}})
V("6 sum where every selected row lacks a value (X26)", claim({"n": sm([{"field": "g", "eq": "b"}], {"field": "x"})}, inner="0"),
  B([{"g": "b"}, {"g": "b", "x": None}, {"g": "a", "x": 1}]),
  {"row": {"verdict": "MALFORMED", "reason": "no_rows_used",
           "detail": {"rows": {"$eq": {"total": 3, "selected": 2, "used": 0, "no_value": 2, "duplicate_rows": 0}},
                      "without": [{"selected": 3, "used": 1, "derived": fr(1)}]}}})
V("6 mean with no row used", claim({"n": sm([], {"field": "x"}, op="mean")}, inner="0"), b'[{"x":null}]',
  {"row": {"verdict": "MALFORMED", "reason": "no_rows_used", "provenance": {"rung": "L2"}}})
V("5.3 9007199254740993 in a row against dec 9007199254740992 (S4)", claim({"n": cnt([{"field": "x", "eq": {"dec": "9007199254740992"}}])}, inner="0"),
  b'[{"x":9007199254740993}]', held(derived=fr(0), rows={"selected": 0}))
V("5.3 9007199254740993 matches its own dec", claim({"n": cnt([{"field": "x", "eq": {"dec": "9007199254740993"}}])}, inner="1"),
  b'[{"x":9007199254740993}]', held(derived=fr(1)))
V("5.3 1.0 equals 1, 2.50 equals 2.5 (F6)", claim({"n": cnt([{"field": "x", "in": [1, {"dec": "2.5"}]}])}, inner="3"),
  b'[{"x":1.0},{"x":1},{"x":2.50},{"x":"1"}]', held(derived=fr(3)))
V("5.3 -0 equals 0", claim({"n": cnt([{"field": "x", "eq": 0}])}, inner="2"), b'[{"x":-0},{"x":0.0e5}]', held(derived=fr(2)))
V("5.3 NaN is never equal and is a value", claim({"n": cnt([{"field": "x", "eq": 1}])}, inner="0"), b'[{"x":NaN}]',
  held(derived=fr(0), where_steps=[{"passed": 0, "no_value": 0}]))
V("5.3 a member named constructor is found only when present (S4)", claim({"n": cnt([{"field": "constructor", "present": True}])}, inner="1"),
  b'[{"a":1},{"constructor":1}]', held(derived=fr(1)))
V("5.3 toString is not a member of a row that lacks it (S4)", claim({"n": cnt([{"field": "toString", "present": True}])}),
  b'[{"a":1}]', mal("field_never_present", {"subject": {"field": "toString"}, "rows_total": 1}, prov=True))
V("5.3 a member named __proto__ is an ordinary member (S4)", claim({"n": cnt([{"field": "__proto__", "eq": 1}])}, inner="1"),
  b'[{"__proto__":1},{"__proto__":{"verdict":1}}]', held(derived=fr(1)))
V("5.3 an inherited name is not reached through __proto__ (S4)", claim({"n": cnt([{"field": "verdict", "eq": 1}])}),
  b'[{"__proto__":{"verdict":1}}]', mal("field_never_present", {"subject": {"field": "verdict"}, "rows_total": 1}, prov=True))
V("5.1 at into a nested value (F10)", claim({"n": cnt([{"at": "/m/acc", "ge": {"dec": "0.9"}}])}, inner="2"),
  b'[{"m":{"acc":0.91}},{"m":{"acc":0.89}},{"m":{"acc":0.9}},{"m":1}]', held(derived=fr(2), where_steps=[{"passed": 2, "no_value": 1}]))
V("5.1 at with an array index", claim({"n": cnt([{"at": "/s/0", "eq": 1}])}, inner="1"), b'[{"s":[1,2]},{"s":[]},{"s":[2]}]',
  held(derived=fr(1), where_steps=[{"passed": 1, "no_value": 1}]))
V("5.1 at through a repeated key", claim({"n": cnt([{"at": "/m/a", "eq": 1}])}), b'[{"m":{"a":1,"a":2}}]',
  mal("field_ambiguous", {"row": 0, "subject": {"at": "/m/a"}}, prov=True))
V("3.4 a decimal literal selects a hyperparameter (F10)", claim({"n": cnt([{"field": "lr", "eq": {"dec": "0.001"}}])}, inner="2"),
  b'[{"lr":0.001},{"lr":1e-3},{"lr":0.01}]', held(derived=fr(2)))
V("3.4 a negative dec threshold", claim({"n": cnt([{"field": "d", "le": {"dec": "-0.5"}}])}, inner="1"), b'[{"d":-0.5},{"d":-0.49}]', held(derived=fr(1)))
V("5.3 boolean literal", claim({"n": cnt([{"field": "ok", "eq": False}])}, inner="1"), b'[{"ok":false},{"ok":0},{"ok":"false"}]', held(derived=fr(1)))
V("5.3 null is no value", claim({"n": cnt([{"field": "x", "eq": "a"}])}, inner="1"), b'[{"x":"a"},{"x":null},{}]',
  held(derived=fr(1), where_steps=[{"passed": 1, "no_value": 2}]))
V("5.3 a string is never equal to a number", claim({"n": cnt([{"field": "x", "in": ["1", 2]}])}, inner="2"), b'[{"x":1},{"x":"1"},{"x":2},{"x":"2"}]',
  held(derived=fr(2)))
V("5.3 in converts a number even when a string literal comes first", claim({"n": cnt([{"field": "x", "in": ["a", 1]}])}), b'[{"x":1e30}]',
  mal("number_out_of_range", {"row": 0, "number": "1e30"}, prov=True))
V("5.3 eq with a string never converts a number", claim({"n": cnt([{"field": "x", "eq": "a"}])}, inner="0"), b'[{"x":1e30}]', held(derived=fr(0)))
V("5.4 code point order, not UTF-16 order (S4)", claim({"n": cnt([{"row": "value", "ge": "～"}])}, inner="2"),
  B(["\U0001F600", "～", "~"]), held(derived=fr(2)))
V("5.4 a proper prefix is smaller", claim({"n": cnt([{"row": "value", "le": "ab"}])}, inner="3"), B(["a", "ab", "abc", "", "b"]), held(derived=fr(3)))
V("5.3 present true counts no no_value", claim({"n": cnt([{"field": "x", "present": True}])}, inner="2"), b'[{"x":null},{"x":1},{}]',
  held(derived=fr(2), where_steps=[{"passed": 2, "no_value": 0}]))
V("5.3 row value subject on object rows is a value", claim({"n": cnt([{"row": "value", "eq": 1}])}, inner="1"), b'[{"a":1},1]', held(derived=fr(1)))

# ================================================================================ O. the five operations and rounding
V("6 count, X01b shape", claim({"n": cnt(ARM)}, inner="100"), ANS_B, held(derived=fr(100), rows={"total": 130, "selected": 100, "used": 100}))
V("6 share, X01c shape", claim({"n": shr(ARM, UP)}, inner="0.23"), ANS_B,
  held(derived=fr(Fraction(N_UP_ARM, 100)), derived_rounded="0.23", rows={"matched": N_UP_ARM, "selected": 100, "total": 130},
       match_steps=[{"passed": N_UP_ARM, "no_value": 0}], where_steps=[{"passed": 130, "no_value": 0}, {"passed": 100, "no_value": 0}]))
V("6 percent 28.6% (exact 200/7)", claim({"n": shr([], [{"row": "value", "eq": 1}], op="percent")}, inner="28.6%"), B([1, 1, 0, 0, 0, 0, 0]),
  held(derived=fr(Fraction(200, 7)), derived_rounded="28.6", printed_vs_exact="above"))
V("6 percent at two places", claim({"n": shr([], [{"row": "value", "eq": 1}], op="percent")}, inner="28.57%"), B([1, 1, 0, 0, 0, 0, 0]),
  held(derived_rounded="28.57", printed_vs_exact="below"))
V("6 sum, X09: 1.5, 2.50, null, missing, 25E-1", claim({"n": sm([{"field": "g", "eq": "a"}], {"field": "x"})}, inner="6.5"),
  b'[{"g":"a","x":1.5},{"g":"a","x":2.50},{"g":"a","x":null},{"g":"a"},{"g":"a","x":25E-1},{"g":"b","x":9}]',
  held(derived=fr(Fraction(13, 2)), rows={"total": 6, "selected": 5, "used": 3, "no_value": 2}))
V("6 mean over three seeds (F10)", claim({"n": sm([], {"field": "acc"}, op="mean")}, inner="0.92"), b'[{"acc":0.91},{"acc":0.93},{"acc":0.92}]',
  held(derived=fr(Fraction(23, 25)), rows={"used": 3, "no_value": 0}))
V("6 mean skips a missing value and counts it", claim({"n": sm([], {"field": "acc"}, op="mean")}, inner="0.92"), b'[{"acc":0.91},{"acc":0.93},{"x":1}]',
  held(derived=fr(Fraction(23, 25)), rows={"used": 2, "no_value": 1, "selected": 3}))
V("6 mean, GRIM: 24/7 printed 3.43 holds", claim({"n": sm([], {"row": "value"}, op="mean")}, inner="3.43"), B([3, 4, 3, 4, 3, 4, 3]),
  held(derived=fr(Fraction(24, 7))))
V("6 mean, GRIM: 3.48 is not a mean of 7 integers", claim({"n": sm([], {"row": "value"}, op="mean")}, inner="3.48"), B([3, 4, 3, 4, 3, 4, 3]),
  failed("value_mismatch", derived_rounded="3.43"))
V("6 sum of negatives", claim({"n": sm([], {"row": "value"})}, inner="-3.5"), b"[-1,-2.5]", held(derived=fr(Fraction(-7, 2)), printed="-3.5"))
V("6 a sum whose unreduced numerator exceeds 2^63 (F6)", claim({"n": sm([], {"row": "value"})}, inner="1999999999999999999"),
  b"[999999999999999999.5,999999999999999999.5]", held(derived=fr(1999999999999999999)))
V("7.1 the smallest unit: 1e-18 summed 100 times", claim({"n": sm([], {"row": "value"})}, inner="0.0000000000000001"),
  b"[" + b",".join([b"1e-18"] * 100) + b"]", held(derived=fr(Fraction(1, 10 ** 16))))
V("7.1 100e-20 is 1e-18", claim({"n": sm([], {"row": "value"})}, inner="0.000000000000000001"), b"[100e-20]", held(derived=fr(Fraction(1, 10 ** 18))))
V("7.1 999999999999999999 is in range", claim({"n": sm([], {"row": "value"})}, inner="999999999999999999"), b"[999999999999999999]", held())
V("7.1 1e18 is out", claim({"n": sm([], {"row": "value"})}, inner="0"), b"[1e18]", mal("number_out_of_range", {"row": 0, "number": "1e18"}, prov=True))
V("7.1 1000000000000000000 is out", claim({"n": sm([], {"row": "value"})}, inner="0"), b"[1000000000000000000]",
  mal("number_out_of_range", {"row": 0, "number": "1000000000000000000"}, prov=True))
V("7.1 1.0000000000000000001 is out", claim({"n": sm([], {"row": "value"})}, inner="1"), b"[1.0000000000000000001]",
  mal("number_out_of_range", {"row": 0, "number": "1.0000000000000000001"}, prov=True))
V("7.1 0e99999 is zero", claim({"n": sm([], {"row": "value"})}, inner="0"), b"[0e99999]", held(derived=fr(0)))
V("7.1 1e00005 has a 1-digit exponent", claim({"n": sm([], {"row": "value"})}, inner="100000"), b"[1e00005]", held(derived=fr(100000)))
V("7.1 an exponent of five digits", claim({"n": sm([], {"row": "value"})}, inner="0"), b"[1e10000]",
  mal("number_out_of_range", {"row": 0, "number": "1e10000"}, prov=True))
V("7.1 an exponent of 30 digits never builds an integer", claim({"n": sm([], {"row": "value"})}, inner="0"), b"[1e" + b"9" * 30 + b"]",
  mal("number_out_of_range", {"row": 0, "number": "1e" + "9" * 30}, prov=True))
V("7.1 4,000 zeros after the point then a 1 is out", claim({"n": sm([], {"row": "value"})}, inner="0"), b"[0." + b"0" * 4000 + b"1]",
  mal("number_out_of_range", {"row": 0, "number": "0." + "0" * 78}, prov=True))
V("7.1 4,000 trailing zeros of an integer reduce to range", claim({"n": sm([], {"row": "value"})}, inner="1"),
  b"[1." + b"0" * 4000 + b"]", held(derived=fr(1)))
V("6 count printed with places", claim({"n": cnt(ARM)}, inner="100.0"), ANS_B, held(derived_rounded="100.0", printed="100.0"))
V("6 count printed with a grouping comma", claim({"n": cnt([])}, inner="1,000"), parts(b"[", (b"0,", 999), b"0]"), held(printed="1000", printed_token="1,000"))
V("7.4 1/8 at two places is 0.12, a tie (X10b)", claim({"n": shr([], [{"row": "value", "eq": 1}])}, inner="0.12"), B([1, 0, 0, 0, 0, 0, 0, 0]),
  held(exact_tie=True, derived_rounded="0.12"))
V("7.4 1/8 printed 0.13 fails (X10a)", claim({"n": shr([], [{"row": "value", "eq": 1}])}, inner="0.13"), B([1, 0, 0, 0, 0, 0, 0, 0]),
  failed("value_mismatch", exact_tie=True, derived_rounded="0.12"))
V("7.4 3/8 at two places is 0.38", claim({"n": shr([], [{"row": "value", "eq": 1}])}, inner="0.38"), B([1, 1, 1, 0, 0, 0, 0, 0]),
  held(exact_tie=True, derived_rounded="0.38"))
V("7.4 5/2 at none is 2", claim({"n": sm([], {"row": "value"})}, inner="2"), b"[2.5]", held(exact_tie=True, derived_rounded="2"))
V("7.4 7/2 at none is 4", claim({"n": sm([], {"row": "value"})}, inner="4"), b"[3.5]", held(exact_tie=True, derived_rounded="4"))
V("7.4 -4/1000 at two places is 0.00 unsigned; the floor fails it", claim({"n": sm([], {"row": "value"})}, inner="0.00"), b"[-0.004]",
  failed("rounds_to_zero", derived_rounded="0.00", printed_vs_exact="above"))
V("7.4 a printed -0.00 is written 0.00", claim({"n": sm([], {"row": "value"})}, inner="-0.00"), b"[-0.004]",
  failed("rounds_to_zero", printed="0.00", printed_token="-0.00"))
V("7.5 rounds_to_zero (X11)", claim({"n": shr([], [{"row": "value", "eq": 1}])}, inner="0.00"), parts(b"[1", (b",0", 999), b"]"),
  failed("rounds_to_zero", derived=fr(Fraction(1, 1000))))
V("7.5 an exact zero holds against a printed zero", claim({"n": shr([], [{"row": "value", "eq": 1}])}, inner="0.00"), parts(b"[0", (b",0", 999), b"]"),
  held(derived=fr(0), printed_vs_exact="equal"))
V("7.5 rounds_to_whole on share: 999/1000 printed 1.00 (C07b)", claim({"n": shr([], [{"field": "ok", "eq": True}])}, inner="1.00"),
  B([{"ok": i < 999} for i in range(1000)]), failed("rounds_to_whole", derived=fr(Fraction(999, 1000)), printed_vs_exact="above"))
V("7.5 rounds_to_whole on percent: 100%", claim({"n": shr([], [{"field": "ok", "eq": True}], op="percent")}, inner="100%"),
  B([{"ok": i < 999} for i in range(1000)]), failed("rounds_to_whole", derived=fr(Fraction(999, 10))))
V("7.5 99.9% holds", claim({"n": shr([], [{"field": "ok", "eq": True}], op="percent")}, inner="99.9%"),
  B([{"ok": i < 999} for i in range(1000)]), held(printed_vs_exact="equal"))
V("7.5 1000 of 1000 printed 1.00 holds", claim({"n": shr([], [{"field": "ok", "eq": True}])}, inner="1.00"),
  B([{"ok": True} for i in range(1000)]), held(derived=fr(1)))
V("7.5 9496 of 10000 printed 0.95 holds and says it rounds up (C07)", claim({"n": shr([], [{"field": "ok", "eq": True}])}, inner="0.95"),
  B([{"ok": i < 9496} for i in range(10000)]),
  dict(held(derived=fr(Fraction(1187, 1250)), printed_vs_exact="above"), surface_has=["printed 0.95 rounds up"]))
V("7.5 a sign difference fails", claim({"n": sm([], {"row": "value"})}, inner="-3"), b"[3]", failed("value_mismatch", printed="-3", derived_rounded="3"))
V("7.5 U+2212 is a minus", claim({"n": sm([], {"row": "value"})}, inner="−3"), b"[-3]", held(printed="-3", printed_token="−3"))
V("7.5 a plus sign is dropped", claim({"n": sm([], {"row": "value"})}, inner="+5"), b"[5]", held(printed="5", printed_token="+5"))
V("7.5 a trailing full stop is not part of the number", claim({"n": sm([], {"row": "value"})}, inner="It was 42."), b"[42]", held(printed_token="42"))
V("7.5 .5 is 0.5", claim({"n": sm([], {"row": "value"})}, inner=".5"), b"[0.5]", held(printed="0.5"))
V("7.5 printed below the exact value", claim({"n": shr([], [{"row": "value", "eq": 1}])}, inner="0.33"), B([1, 0, 0]),
  held(printed_vs_exact="below", derived=fr(Fraction(1, 3))))
V("7.5 a count of zero is a value", claim({"n": cnt([{"row": "value", "eq": 9}])}, inner="0"), B([1, 2]), held(derived=fr(0)))
V("6 count of a keyed source", claim({"n": cnt([{"row": "value", "eq": "UNSURE"}])}, inner=str(N_UNS_ALL)), ANS_B, held(derived=fr(N_UNS_ALL)))
V("6 percent over a JSON Lines source", claim({"n": {"rows": {"format": "jsonl"}, "where": [], "match": [{"field": "ok", "eq": True}], "op": "percent"}}, inner="75%"),
  b'{"ok":true}\n{"ok":true}\n{"ok":false}\n{"ok":true}\n', held(derived=fr(75)))
V("6 sum with a where on a JSON Lines source", claim({"n": {"rows": {"format": "jsonl"}, "where": [{"field": "g", "eq": "a"}], "of": {"field": "x"}, "op": "sum"}}, inner="3"),
  b'{"g":"a","x":1}\n{"g":"b","x":5}\n{"g":"a","x":2}\n', held(derived=fr(3)))

# ================================================================================ CN. the census (F01)
_runs = B([{"run": i, "status": ("PASS" if i % 9 else "FAIL")} for i in range(163)])
V("8.4 census: a wrong-case literal proves a zero; the census shows it (C06)", claim({"n": cnt([{"field": "status", "eq": "FAILED"}])}, inner="0"), _runs,
  dict(held(derived=fr(0), rows={"total": 163, "selected": 0},
            census=[{"subject": {"field": "status"}, "distinct": 2, "absent": 0, "other": 0,
                     "values": [{"type": "string", "text": "PASS", "rows": 144}, {"type": "string", "text": "FAIL", "rows": 19}],
                     "literals_never_seen": ["FAILED"]}]),
       surface_has=['literal "FAILED" occurs in no row', 'field "status": "PASS" 144, "FAIL" 19']))
V("8.4 census: the honest literal is seen", claim({"n": cnt([{"field": "status", "eq": "FAIL"}])}, inner="19"), _runs,
  held(derived=fr(19), census=[{"literals_never_seen": []}]))
_many = B([{"v": "v%02d" % k} for k in range(20) for _ in range(20 - k)])
V("8.4 census: capped at 16, other counts the rest", claim({"n": cnt([{"field": "v", "eq": "v00"}])}, inner="20"), _many,
  held(census=[{"distinct": 20, "other": 4 + 3 + 2 + 1, "absent": 0,
                "values": [{"type": "string", "text": "v%02d" % k, "rows": 20 - k} for k in range(16)]}]))
V("8.4 census: ties ordered by type then code point (S4 order trap)", claim({"n": cnt([{"row": "value", "eq": "~"}])}, inner="1"),
  B(["\U0001F600", "～", "~", 1, None, True]),
  held(census=[{"distinct": 6, "values": [{"type": "boolean", "text": "true", "rows": 1}, {"type": "null", "text": "null", "rows": 1},
                                          {"type": "number", "text": "1", "rows": 1}, {"type": "string", "text": "~", "rows": 1},
                                          {"type": "string", "text": "～", "rows": 1}, {"type": "string", "text": "\U0001F600", "rows": 1}]}]))
V("8.4 census: absent and null are told apart", claim({"n": cnt([{"field": "x", "eq": 1}])}, inner="1"), b'[{"x":1},{"x":null},{}]',
  held(census=[{"absent": 1, "values": [{"type": "null", "text": "null", "rows": 1}, {"type": "number", "text": "1", "rows": 1}]}]))
V("8.4 census: 1 and 1.0 are two tokens; literal 1 is seen", claim({"n": cnt([{"field": "x", "eq": 1}])}, inner="2"), b'[{"x":1},{"x":1.0}]',
  held(census=[{"distinct": 2, "literals_never_seen": []}]))
V("5.2 the census never runs past a refused conversion", claim({"n": cnt([{"field": "x", "eq": 2}])}, inner="0"), b'[{"x":1},{"x":1e30}]',
  mal("number_out_of_range", {"row": 1, "number": "1e30"}, prov=True))
V("8.4 census: a dec literal never seen", claim({"n": cnt([{"field": "x", "in": [{"dec": "0.5"}, 1]}])}, inner="1"), b'[{"x":1},{"x":"0.5"}]',
  held(census=[{"literals_never_seen": [{"dec": "0.5"}]}]))
V("8.4 census: arrays and objects are one entry per type", claim({"n": cnt([{"row": "value", "eq": 1}])}, inner="0"), b'[[1],[2],{"a":1}]',
  held(census=[{"distinct": 2, "values": [{"type": "array", "text": "", "rows": 2}, {"type": "object", "text": "", "rows": 1}]}]))
V("8.4 census: shown text is 64 code points, unpaired surrogates as ?", claim({"n": cnt([{"row": "value", "eq": "a"}])}, inner="0"),
  b'["' + ("\U0001F600" * 70).encode("utf-8") + b'","\\ud800x"]',
  held(census=[{"values": [{"type": "string", "text": "?x", "rows": 1}, {"type": "string", "text": "\U0001F600" * 64, "rows": 1}]}]))
V("8.4 census: one entry per subject across where and match", claim({"n": shr([{"field": "s", "in": ["a", "b"]}], [{"field": "s", "eq": "c"}, {"field": "t", "eq": 1}])}, inner="0.00"),
  b'[{"s":"a","t":1},{"s":"b","t":2}]',
  held(census=[{"subject": {"field": "s"}, "literals_never_seen": ["c"]}, {"subject": {"field": "t"}, "literals_never_seen": []}]))
V("8.4 census: ge and le subjects have none", claim({"n": cnt([{"field": "x", "ge": 1}])}, inner="1"), b'[{"x":1}]', held(census=[]))
V("8.4 census: the row value subject (X01 shape)", claim({"n": cnt(UP)}, inner=str(N_UP_ALL)), ANS_B,
  held(census=[{"subject": {"row": "value"}, "distinct": 3,
                "values": [{"type": "string", "text": "SUPPORTED", "rows": N_SUP_ALL}, {"type": "string", "text": "CONTRADICTED", "rows": N_UP_ALL},
                           {"type": "string", "text": "UNSURE", "rows": N_UNS_ALL}]}]))

# ================================================================================ L. leave-one-out (F06)
V("8.3 without: dropping UNSURE lifts 0.23 to 0.29; without that test it is 0.23 (C01)",
  claim({"n": shr(ARM + [{"row": "value", "in": ["SUPPORTED", "CONTRADICTED"]}], UP)}, inner="0.29"), ANS_B,
  dict(held(derived=fr(Fraction(N_UP_ARM, 100 - N_UNSURE_ARM)), rows={"selected": 100 - N_UNSURE_ARM},
            without=[{"selected": 100 - N_UNSURE_ARM, "matched": N_UP_ARM},
                     {"selected": 130 - N_UNSURE_ARM - 0 - sum(1 for k, v in ANS.items() if k > "E1-099" and v == "UNSURE")},
                     {"selected": 100, "matched": N_UP_ARM, "derived": fr(Fraction(N_UP_ARM, 100))}]),
       surface_has=["without test 3: 100 rows selected, value %s" % ("%d/%d" % (Fraction(N_UP_ARM, 100).numerator, Fraction(N_UP_ARM, 100).denominator))]))
V("8.2 where_steps do not depend on test order (C12), order A", claim({"n": cnt(ARM + [{"row": "value", "in": ["SUPPORTED", "CONTRADICTED"]}])}, inner=str(100 - N_UNSURE_ARM)),
  ANS_B, held(where_steps=[{"passed": 130}, {"passed": 100}, {"passed": 130 - N_UNS_ALL}]))
V("8.2 where_steps do not depend on test order (C12), order B", claim({"n": cnt([{"row": "value", "in": ["SUPPORTED", "CONTRADICTED"]}] + ARM)}, inner=str(100 - N_UNSURE_ARM)),
  ANS_B, held(where_steps=[{"passed": 130 - N_UNS_ALL}, {"passed": 130}, {"passed": 100}]))
V("8.3 without on a sum carries used", claim({"n": sm([{"field": "g", "eq": "a"}], {"field": "x"})}, inner="7"), SIX,
  held(without=[{"selected": 6, "used": 4, "derived": fr(10)}], rows={"used": 3, "no_value": 1, "selected": 4}))
V("8.3 without on a count", claim({"n": cnt([{"field": "g", "eq": "a"}, {"field": "x", "ge": 2}])}, inner="2"), SIX,
  held(without=[{"selected": 3, "derived": fr(3)}, {"selected": 4, "derived": fr(4)}]))
V("8.1 no where test: without is empty", claim({"n": cnt([])}, inner="6"), SIX, held(without=[]))

# ================================================================================ DU. duplicate rows (F09)
_base = [{"t": "t%02d" % i, "ok": i < 90} for i in range(100)]
V("8.5 duplicate rows: 40 passing rows written twice (C08)", claim({"n": shr([], [{"field": "ok", "eq": True}])}, inner="0.93"),
  B(_base + _base[:40]), dict(held(derived=fr(Fraction(130, 140)), rows={"duplicate_rows": 40, "total": 140}),
                              surface_has=["40 rows repeat an earlier row"]))
V("8.5 duplicate rows: a keyed source has none to count", claim({"n": cnt([])}, inner="130"), ANS_B, held(rows={"duplicate_rows": None}))
V("8.5 duplicate rows: JSON Lines lines equal after the surrounding space", claim({"n": _JL}, inner="3"), b'{"a":1}\n {"a":1}\r\n{"a": 1}\n',
  held(rows={"duplicate_rows": 1}))
V("8.5 duplicate rows: bytes as written, so 1 and 1.0 differ", claim({"n": cnt([])}, inner="3"), b"[1,1.0, 1 ]", held(rows={"duplicate_rows": 1}))
V("8.5 duplicate rows: scalar rows repeat", claim({"n": cnt([])}, inner="3"), b'["a","a","a"]', held(rows={"duplicate_rows": 2}))

# ================================================================================ SO. the source (F05)
_full = [{"id": k, "v": v} for k, v in ANS.items() if k <= "E1-099"]
_kept = [r for r in _full if r["v"] != "UNSURE"]
V("8.7 source: a pre-filtered sibling shows (C02)", claim({"n": shr([], [{"field": "v", "eq": "CONTRADICTED"}], rows={"format": "json", "at": "/kept"})}, inner="0.29"),
  B({"all": _full, "kept": _kept}),
  dict(held(derived=fr(Fraction(N_UP_ARM, len(_kept))), rows={"total": len(_kept)},
            source={"parent_members": 2, "pointer_tokens": 1, "sibling_containers": 1,
                    "siblings": [{"at": "all", "type": "array", "size": 100}], "complete": True}),
       surface_has=['beside it: "all" (array of 100)']))
V("8.7 source: sibling list capped at 16", claim({"n": cnt([], rows={"format": "json", "at": "/x"})}, inner="0"),
  B(dict([("s%02d" % k, [0] * k) for k in range(20)] + [("x", []), ("y", 1)])),
  held(source={"parent_members": 22, "sibling_containers": 20, "siblings": [{"at": "s%02d" % k, "size": k} for k in range(16)]}))
V("8.7 source: an array parent names siblings by index", claim({"n": cnt([], rows={"format": "json", "at": "/1"})}, inner="1"), b'[[1,2],[3],{"a":1},4]',
  held(source={"parent_members": 4, "siblings": [{"at": "0", "type": "array", "size": 2}, {"at": "2", "type": "object", "size": 1}]}))
V("8.7 source: complete false is carried", claim({"n": cnt([])}, inner="0", res={"verdict": "RESOLVED", "entry": dict(ENTRY, complete=False)}), b"[]",
  held(source={"complete": False}))
V("8.7 source: complete null is carried", claim({"n": cnt([])}, inner="0", res={"verdict": "RESOLVED", "entry": dict(ENTRY, complete=None)}), b"[]",
  held(source={"complete": None}))
V("10.1 provenance is the entry's", claim({"n": cnt([])}, inner="0", res={"verdict": "RESOLVED", "entry": dict(ENTRY, rung="undeclared", kind_of_source="test_report")}),
  b"[]", {"row": {"verdict": "HELD", "provenance": {"$eq": {"form": "rn", "harness": "vector-harness", "rung": "undeclared", "kind_of_source": "test_report"}},
                  "resolved_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"}})

# ================================================================================ SS. selected rows (F11)
_up = B([{"id": "r%d" % i, "ok": True} for i in range(1, 131)])
V("8.6 selected_span: unpadded ids select 77, not 50 (C13)", claim({"n": cnt([{"field": "id", "ge": "r1"}, {"field": "id", "le": "r50"}])}, inner="77"), _up,
  held(derived=fr(77), selected_span={"first_row": 0, "last_row": 129, "first_key": None, "last_key": None}))
V("8.6 selected_span: keys of a keyed source", claim({"n": cnt(ARM)}, inner="100"), ANS_B,
  held(selected_span={"first_row": 0, "last_row": 99, "first_key": "E1-000", "last_key": "E1-099"}))
V("8.6 selected_span: nothing selected", claim({"n": cnt([{"row": "value", "eq": "Q"}])}, inner="0"), ANS_B,
  held(selected_span={"$eq": {"first_row": None, "last_row": None, "first_key": None, "last_key": None}}))

# ================================================================================ SU. the surface (F07, F12, S9)
_pr = B([{"actual": i < 80, "pred": (i < 40) or (200 - i <= 5)} for i in range(200)])
V("8.8 the template names what was computed: a precision where the prose says recall (C04)",
  claim({"recall": shr([{"field": "pred", "eq": True}], [{"field": "actual", "eq": True}])}, name="recall", inner="0.89"), _pr,
  dict(held(derived=fr(Fraction(40, 45))),
       surface_has=['share of rows where field "actual" = true, among rows where field "pred" = true', "45 of 200 rows selected; 40 of 45 matched"]))
V("8.8 author strings are text: a field name that looks like markup (S9)",
  claim({"n": cnt([{"field": "<img src=x onerror=alert(1)>", "eq": 1}])}, inner="1"), b'[{"<img src=x onerror=alert(1)>":1}]',
  dict(held(derived=fr(1)), surface_has=['field "<img src=x onerror=alert(1)>" = 1']))
V("8.8 the surface carries provenance and never says raw", claim({"n": cnt([])}, inner="0"), b"[]",
  dict(held(), surface_has=["rows of entry r1: kind_of_source tool_stdout, rung L2, harness \"vector-harness\", complete true; 2 bytes"]))
V("8.8 no surface for a MALFORMED row", claim({"n": {"op": "count"}}, inner="0"), b"[]", {"surface": ""})

# ================================================================================ BU. the document budget (S3)
V("12.2 the budget refuses the bytes of a new entry", claim({"n": cnt([])}, inner="0", budget={"spent_before": BUDGET - 1, "entry_read_before": False}),
  b"[]", dict(mal("derive_budget", {"budget": BUDGET, "spent_before": BUDGET - 1, "cost": 2}), spent=BUDGET - 1))
V("12.2 an entry read before costs no bytes", claim({"n": cnt([])}, inner="0", budget={"spent_before": BUDGET - 1, "entry_read_before": True}),
  b"[]", dict(held(derived=fr(0)), spent=BUDGET - 1))
V("12.2 the rows cost: N times (tests + 1)", claim({"n": cnt([{"row": "value", "eq": 1}])}, inner="1", budget={"spent_before": BUDGET - 10, "entry_read_before": True}),
  b"[1,2,3,4,5,6]", dict(mal("derive_budget", {"budget": BUDGET, "spent_before": BUDGET - 10, "cost": 12}), spent=BUDGET - 10))
V("12.2 spent is bytes plus rows cost", claim({"n": shr([{"field": "g", "eq": "a"}], [{"field": "x", "ge": 2}])}, inner="0.50"), SIX,
  dict(held(derived=fr(Fraction(1, 2)), match_steps=[{"passed": 2, "no_value": 1}]), spent=len(SIX) + 6 * 3))
V("12.2 a reader failure still spends the bytes", claim({"n": cnt([])}, inner="0", budget={"spent_before": 5, "entry_read_before": False}), b"[1,",
  dict(mal("receipt_not_json", {}), spent=5 + 3))
V("12.2 rows_too_large comes before the budget", claim({"n": cnt([])}, inner="0", budget={"spent_before": BUDGET, "entry_read_before": False}),
  parts((b" ", 16777217)), dict(mal("rows_too_large", {"bytes": 16777217, "cap": 16777216}), spent=BUDGET))

# ================================================================================ X. the draft's worked shapes kept
V("draft X04 shape: a count over items at a pointer", claim({"n": cnt([{"field": "kind", "in": ["path", "file"]}], rows={"format": "json", "at": "/items"})}, inner="2"),
  b'{"items":[{"kind":"path"},{"kind":"file"},{"kind":"prose"}],"n":3}', held(derived=fr(2), source={"parent_members": 2, "sibling_containers": 0}))
V("draft X05 shape: share where control is false", claim({"n": shr([{"field": "control", "eq": False}], [{"field": "verdict", "eq": "killed"}])}, inner="0.5"),
  b'[{"control":false,"verdict":"killed"},{"control":false,"verdict":"survived"},{"control":true,"verdict":"killed"}]',
  held(derived=fr(Fraction(1, 2)), rows={"selected": 2, "matched": 1}))
V("draft X08 shape: a sum of counts at a deep pointer", claim({"n": sm([], {"field": "k"}, rows={"format": "json", "at": "/gates/G_C/per_twin"})}, inner="182"),
  B({"gates": {"G_C": {"per_twin": [{"k": 20}] * 9 + [{"k": 2}], "k": 182}}}), held(derived=fr(182), rows={"used": 10}))


# ================================================================================ C. the cheating critic's other cases
_pnl = B([{"m": "jan", "pnl": 120}, {"m": "feb", "pnl": 80}, {"m": "mar", "pnl": None}, {"m": "apr", "pnl": 95}, {"m": "may"}, {"m": "jun", "pnl": 60}])
V("16.11 a total that skips null and absent months holds; the skip is counted (C05)", claim({"t": sm([], {"field": "pnl"})}, name="t", inner="355"), _pnl,
  dict(held(derived=fr(355), rows={"used": 4, "no_value": 2, "selected": 6}), surface_has=["4 of 6 used, 2 without a value"]))
V("16.1 rows synthesised from a summary hold like any rows (C09); only provenance can speak",
  claim({"r": shr([], [{"field": "ok", "eq": True}])}, name="r", inner="0.95"), B([{"ok": True}] * 95 + [{"ok": False}] * 5),
  held(derived=fr(Fraction(19, 20)), rows={"duplicate_rows": 98}))
_garden = B([{"id": "x%03d" % i, "lang": "py js go rs"[3 * (i % 4):3 * (i % 4) + 2].strip(), "size": ["s", "m", "l"][i % 3],
              "ok": (i * 37) % 100 < 52} for i in range(120)])
_gsel = [i for i in range(120) if "py js go rs"[3 * (i % 4):3 * (i % 4) + 2].strip() == "py" and ["s", "m", "l"][i % 3] == "s"]
_gok = sum(1 for i in _gsel if (i * 37) % 100 < 52)
_gw = round(Fraction(_gok, len(_gsel)) * 100)   # Fraction rounding is half to even, as 7.4
V("16.4 one derivation of a garden, published alone, holds; its selection count shows (C10)",
  claim({"r": shr([{"field": "lang", "in": ["py"]}, {"field": "size", "in": ["s"]}], [{"field": "ok", "eq": True}])}, name="r",
        inner="%d.%02d" % (_gw // 100, _gw % 100)),
  _garden, held(derived=fr(Fraction(_gok, len(_gsel))), rows={"selected": len(_gsel), "total": 120, "matched": _gok}),
  note="HELD says nothing about how many derivations were tried (15.12)")


V("9f MALFORMED kind_of_source_unknown from v0 6.6", claim({"n": cnt([])}, inner="0", res={"verdict": "MALFORMED", "reason": "kind_of_source_unknown"}),
  b"[]", mal("kind_of_source_unknown", {}))
V("9f UNRESOLVED manifest_integrity", claim({"n": cnt([])}, inner="0", res={"verdict": "UNRESOLVED", "reason": "manifest_integrity"}), b"[]",
  {"row": {"verdict": "UNRESOLVED", "reason": "manifest_integrity"}})
V("9f UNRESOLVED rung_unknown", claim({"n": cnt([])}, inner="0", res={"verdict": "UNRESOLVED", "reason": "rung_unknown"}), b"[]",
  {"row": {"verdict": "UNRESOLVED", "reason": "rung_unknown"}})
V("9f UNRESOLVED manifest_absent", claim({"n": cnt([])}, inner="0", res={"verdict": "UNRESOLVED", "reason": "manifest_absent"}), b"[]",
  {"row": {"verdict": "UNRESOLVED", "reason": "manifest_absent"}})
V("13.2 a claim file over 8 MiB", parts(b'{"blocks":[],"span":{"receipt":"r1#n","inner":"', (b"1", 8388608), b'"},"resolution":{"verdict":"UNRESOLVED","reason":"manifest_absent"}}'),
  b"[]", refuse("claim_too_large"))


# ------------------------------------------------ added after mutation run 1 (builder notes mutants.out): J02 and J05 were missed
V("A.1 a raw control character inside a receipt string", claim({"n": cnt([])}), b'["a\x01b"]', mal("receipt_not_json", {}))
_TAB_RAW = '{"n": {"rows": {"format": "json", "at": ""}, "where": [{"row": "value", "eq": "a' + chr(9) + 'b"}], "op": "count"}}'
_TAB_ESC = '{"n": {"rows": {"format": "json", "at": ""}, "where": [{"row": "value", "eq": "a' + chr(92) + 'tb"}], "op": "count"}}'
V("A.1 a raw TAB inside a block string", claim(blocks=[{"line": 2, "content": _TAB_RAW}], inner="0"), b"[]", mal("derive_block_malformed", {"block_line": 2}))
V("A.1 an escaped TAB inside a block string is fine", claim(blocks=[{"line": 2, "content": _TAB_ESC}], inner="1"),
  b'["a' + bytes([92]) + b'tb"]', held(derived=fr(1)))
V("7.1 five exponent digits are out even when the value would be 1", claim({"n": sm([], {"row": "value"})}, inner="1"),
  b"[0." + b"0" * 9999 + b"1e10000]", mal("number_out_of_range", {"row": 0, "number": "0." + "0" * 78}, prov=True))
V("7.1 four exponent digits and a long fraction reach 1", claim({"n": sm([], {"row": "value"})}, inner="1"),
  b"[0." + b"0" * 9998 + b"1e9999]", held(derived=fr(1)))


# ------------------------------------------------------------------ write
def main():
    force = "--force" in sys.argv
    ids = [v["id"] for v in VECTORS]
    assert len(set(ids)) == len(ids)
    doc = {"profile": "derive/v0-draft", "status": "DRAFT, unsigned",
           "generator": "receipt/derive/vectors/make_vectors.py",
           "count": len(VECTORS), "vectors": VECTORS}
    text = json.dumps(doc, ensure_ascii=True, indent=None, separators=(",", ":"), sort_keys=False) + "\n"
    if os.path.exists(OUT) and not force:
        with open(OUT, "r", encoding="utf-8") as fh:
            old = fh.read()
        if old != text:
            sys.stderr.write("refusing to overwrite %s with different content; pass --force\n" % OUT)
            return 3
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    sys.stdout.write("wrote %d vectors to %s\n" % (len(VECTORS), OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
