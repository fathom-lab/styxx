# -*- coding: utf-8 -*-
"""RECEIPT v0 (DRAFT, unsigned): builds receipt/v0/vectors/receipt_v0_vectors.json.

Four sources, each vector carrying its own expected answer:

  conformance  every in-scope conformance document (the 1,718 tests/test_receipt_v0.py walks).
               Expected: digest_v0(to_v0(sworn.verify(...))), i.e. today's checker projected, NOT
               check_v0's answer. Where the text is stricter than today's code on purpose
               (RECEIPT_v0.md 13.14) the vector carries the text's answer and a note.
  hand         written from the text: every reason code, every refusal of 11.1, every worked
               example. Each carries the rows the text gives ([verdict, reason] per row, plus the
               detail values a worked example states); the builder refuses to write a vector
               whose check_v0 rows differ from those, and only then records check_v0's digest.
  recheck      11.3: REDERIVES, DOES-NOT-REDERIVE and each refusal of 11.3.
  jcs          section 3: known canonical texts and each refusal code JSON can reach.

A vector's id is sha256 over the UTF-8 bytes of the canonical text (section 3) of
{"op": op, "inputs": inputs}; a name or other input string that has no canonical form (an
unpaired surrogate) is replaced inside the id object, and only there, by
{"utf16be_hex": <its UTF-16BE code units, surrogates passed through, lowercase hex>}.
No id depends on a build.

Needs styxx (for the conformance source) on PYTHONPATH. Writes nothing when the output exists;
`--check` rebuilds in memory and compares with the committed file instead.
"""
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V0 = HERE.parent
ROOT = V0.parent.parent
OUT = HERE / "receipt_v0_vectors.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("receipt_check_v0", V0 / "check_v0.py")
P = _load("receipt_to_v0", V0 / "to_v0.py")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def b64(b):
    return None if b is None else base64.b64encode(b).decode("ascii")


def _id_safe(v):
    if isinstance(v, str):
        try:
            C.jcs(v)
            return v
        except C.Refusal:
            return {"utf16be_hex": v.encode("utf-16-be", "surrogatepass").hex()}
    if isinstance(v, dict):
        return {k: _id_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_id_safe(x) for x in v]
    return v


def vector_id(op, inputs):
    return C.digest({"op": op, "inputs": _id_safe(inputs)})


# reason -> the rules that produce it
RULES = {
    "tag_syntax": ["2.3d"], "nesting": ["2.3d"], "stray_closer": ["2.3d"], "unclosed": ["2.4"],
    "empty_span": ["2.3d", "7.1b"], "hidden_commitment": ["2.3b", "2.3d"], "length_cap": ["7.1c"],
    "kind_unknown": ["7.1d"], "kind_reserved": ["7.1d"], "receipt_form": ["4", "7.1e"],
    "absent_over_partial": ["7.1f", "7.1j"], "hash_over_partial": ["7.1f"],
    "directional_override": ["7.1g"], "number_count": ["7.2"], "number_grammar": ["7.2"],
    "needle_count": ["7.4"], "needle_empty": ["7.4"], "digest_form": ["7.6"],
    "kind_of_source_unknown": ["6.6", "6.8"], "receipt_author_minted": ["6.7"],
    "anchor_out_of_range": ["6.11"], "receipt_not_json": ["6.12", "7.3"],
    "pointer_unresolvable": ["6.12"], "pointer_ambiguous": ["6.12"],
    "leaf_not_scalar": ["7.3"], "leaf_not_numeric": ["7.3"], "leaf_not_string": ["7.5"],
    "short_needle": ["7.5"], "manifest_absent": ["6.1"],
    "manifest_integrity": ["5.6", "6.2", "6.5", "6.10"], "rung_unknown": ["6.3"],
    "manifest_id_missing": ["6.4"], "manifest_bytes_absent": ["6.10"],
    "manifest_no_completeness": ["7.1j"], "no_repository": ["4.4"],
    "value_mismatch": ["7.3"], "rounds_to_zero": ["7.3a"], "needle_missing": ["7.5"],
    "needle_present": ["7.8"], "digest_mismatch": ["7.7"],
}
HELD_RULES = {"numeric": ["6.9", "7.3"], "quote": ["6.9", "7.5"], "hash": ["6.9", "7.7"],
              "absent": ["6.9", "7.8"]}


def rules_of(core, manifest_given):
    r = {"2.3", "8", "9.1", "9.2", "9.3"}
    if manifest_given:
        r |= {"5.4", "5.5"}
    if core["document_malformed"] is not None:
        r.add("2.1" if core["document_malformed"]["reason"] == "invalid_utf8" else "2.2")
    if core["document"]["canonical_sha256"] is not None:
        r.add("2.5")
    for row in core["spans"]:
        r |= set(HELD_RULES[row["kind"]] if row["verdict"] == "HELD" else RULES[row["reason"]])
    return sorted(r, key=_rule_key)


def _rule_key(rule):
    key = []
    for part in rule.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        key.append((int(digits) if digits else 0, part[len(digits):]))
    return key


# ---------------------------------------------------------------------------------------------
# (a) the conformance documents
# ---------------------------------------------------------------------------------------------

def conformance_vectors():
    sys.path.insert(0, str(ROOT))
    import styxx
    from styxx import sworn
    assert Path(styxx.__file__).resolve().parent.parent == ROOT, styxx.__file__
    out = []
    for set_dir in (ROOT / "conformance" / "sworn", ROOT / "conformance" / "sworn" / "receipt_v2"):
        blobs = json.loads((set_dir / "blobs.json").read_text(encoding="utf-8"))
        for fam in sorted((set_dir / "vectors").glob("*.json")):
            for v in json.loads(fam.read_text(encoding="utf-8"))["vectors"]:
                if not (v["mode"] == "inline" and set(v["requires"]) <= {"manifest"}
                        and v["expect"]["outcome"] == "core"):
                    continue
                i = v["inputs"]
                doc = base64.b64decode(blobs[i["document"]])
                mbytes = base64.b64decode(blobs[i["manifest"]]) if i["manifest"] else None
                inputs = {"document_b64": b64(doc), "name": i["name"],
                          "manifest_b64": b64(mbytes), "commit": i["commit"]}
                vec = {"id": vector_id("check", inputs), "source": "conformance", "op": "check",
                       "conformance": {"set": set_dir.name, "family": v["family"], "id": v["id"]},
                       "inputs": inputs, "note": None}
                mdict = json.loads(mbytes.decode("utf-8")) if mbytes is not None else None
                if isinstance(mdict, dict) and mdict.get("spec") != C.MANIFEST_SPEC:
                    vec["expect"] = {"refusal": "manifest_spec"}
                    vec["rules"] = ["5.5", "11.1"]
                    vec["note"] = ("stricter than today's code on purpose (RECEIPT_v0.md 13.14, first "
                                   "item: manifests at 0.1 are refused); today's checker reads "
                                   "this manifest (spec %r) and answers %s"
                                   % (mdict.get("spec"), v["expect"]["document_verdict"]))
                else:
                    man = sworn.Manifest.from_dict(mdict) if mdict is not None else None
                    core = sworn.verify(doc, name=i["name"], manifest=man, tree=None,
                                        commit=i["commit"])
                    proj = P.to_v0(core, doc)
                    vec["expect"] = {"core_digest": P.digest_v0(proj)}
                    vec["rules"] = rules_of(proj, mbytes is not None)
                out.append(vec)
    return out


# ---------------------------------------------------------------------------------------------
# (b) hand-written vectors, from the text
# ---------------------------------------------------------------------------------------------

def S(ref, kind, inner):
    return '<sworn r="%s" k="%s">%s</sworn>' % (ref, kind, inner)


TS = "2026-01-01T00:00:00Z"
LOG = b"line one is here\np95: 240\nresult: all tests passed ok\n"
OBJ = (b'{"p95": 240, "s": "the quick brown fox", "s42": "42", "t": true, "obj": {"k": 1}, '
       b'"a": [1, 2], "n": null, "nan": NaN, "inf": Infinity, "ninf": -Infinity, '
       b'"x": "\\ud800x", "dup": 1, "dup": 2}')
H64 = sha(b"some artifact")
H32 = "0123456789abcdef0123456789abcdef"


def entry(data=None, kos="tool_stdout", complete=True, sha256=None, drop=(), **extra):
    e = {"id": None, "sha256": sha256 if sha256 is not None else sha(data),
         "kind_of_source": kos, "captured_at": TS, "complete": complete}
    if data is not None:
        e["bytes"] = b64(data)
    e.update(extra)
    for k in drop:
        e.pop(k, None)
    return e


def manifest(receipts, rung="L2", authored=(), with_digest=True, top=None, drop_top=()):
    m = {"spec": C.MANIFEST_SPEC, "harness": "harness-x", "turn": "turn-1", "minted_at": TS,
         "authored_sha256": list(authored), "receipts": {}, "rung": rung}
    for rid, e in receipts.items():
        if isinstance(e, dict):
            e = dict(e)
            if e.get("id", 1) is None:
                e["id"] = rid
        m["receipts"][rid] = e
    for k in drop_top:
        m.pop(k, None)
    if with_digest:
        core = {k: m.get(k) for k in ("spec", "harness", "turn", "minted_at", "authored_sha256",
                                       "receipts", "rung")}
        core["authored_sha256"] = sorted(a.translate({ord(u): ord(l) for u, l in zip("ABCDEF", "abcdef")})
                                         for a in core["authored_sha256"])
        m["digest"] = C.digest(core)
    if top:
        m.update(top)
    return json.dumps(m, ensure_ascii=False, sort_keys=True).encode("utf-8")


def M1(data, **kw):
    """A manifest with the one entry r1 over `data`."""
    rung = kw.pop("rung", "L2")
    return manifest({"r1": entry(data, **kw)}, rung=rung)


HAND = []


def H(label, doc, rows, dv, manifest=None, name="d.md", commit=None, details=None, note=None,
      core_checks=None, rules=None):
    """rows: [(verdict, reason)]; details: {row index: {key: value}} that must hold exactly."""
    HAND.append({"label": label, "doc": doc, "rows": rows, "dv": dv, "manifest": manifest,
                 "name": name, "commit": commit, "details": details or {}, "note": note,
                 "core_checks": core_checks or {}, "rules": rules})


def R(label, doc, refusal, manifest=None, name="d.md", commit=None, rules=None, note=None):
    HAND.append({"label": label, "doc": doc, "refusal": refusal, "manifest": manifest,
                 "name": name, "commit": commit, "rules": rules or ["11.1"], "note": note})


U, M, F, HD = "UNRESOLVED", "MALFORMED", "FAILED", "HELD"
N7 = S("r1", "numeric", "n is 7")

# --- section 2: documents, fences, the scan
H("2.1 empty document is UNSWORN with the stated canonical sha256", b"", [], "UNSWORN",
  core_checks={"canonical_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"})
H("narrative only", "no spans here; 42 is just text", [], "UNSWORN")
H("2.1 invalid_utf8: overlong form", b"ok \xc0\xaf " + N7.encode(), [], "SWORN-FAILED",
  core_checks={"document_malformed": {"reason": "invalid_utf8", "at": 3}, "canonical_sha256": None})
H("2.1 invalid_utf8: an encoded surrogate", b"\xed\xa0\x80", [], "SWORN-FAILED",
  core_checks={"document_malformed": {"reason": "invalid_utf8", "at": 0}})
H("2.1 invalid_utf8: above U+10FFFF", b"x\xf4\x90\x80\x80", [], "SWORN-FAILED",
  core_checks={"document_malformed": {"reason": "invalid_utf8", "at": 1}})
H("2.1 invalid_utf8: truncated sequence", b"x\xe2\x82 " + N7.encode(), [], "SWORN-FAILED",
  core_checks={"document_malformed": {"reason": "invalid_utf8", "at": 1}})
H("2.2 unbalanced fences list every delimiter line", "```\ncode\n```\n" + N7 + "\n  ```py\n",
  [], "SWORN-FAILED",
  core_checks={"document_malformed": {"reason": "unbalanced_fences", "delimiter_lines": [1, 3, 5]},
               "canonical_sha256": None})
H("2.2 a fenced region hides a tag", "```\n" + N7 + "\n```\n", [], "UNSWORN")
H("2.2 three spaces still fence", "   ````x\n" + N7 + "\n```\n", [], "UNSWORN")
H("2.2 four spaces do not fence", "    ```\n" + N7 + "\n", [(U, "manifest_absent")], "SWORN-UNCHECKED")
H("2.2 tilde fences are not fences", "~~~\n" + N7 + "\n~~~\n", [(U, "manifest_absent")],
  "SWORN-UNCHECKED")
H("2.2 a blockquoted fence is not a fence", "> ```\n" + N7 + "\n", [(U, "manifest_absent")],
  "SWORN-UNCHECKED")
H("2.3c an inline code span hides a tag", "`" + N7 + "`", [], "UNSWORN")
H("2.3c a backtick run with no equal run on its line hides nothing", "`` x " + N7 + " `",
  [(U, "manifest_absent")], "SWORN-UNCHECKED")
H("2.3c the closing run must be on the same line", "`\n" + N7 + "`", [(U, "manifest_absent")],
  "SWORN-UNCHECKED")
H("2.3b hidden_commitment inside a comment", "<!-- " + N7 + " -->", [(M, "hidden_commitment")],
  "SWORN-FAILED")
H("2.3b an unterminated comment runs to the end", "<!-- " + N7, [(M, "hidden_commitment")],
  "SWORN-FAILED")
H("2.3b a closed comment hides nothing after it", "<!-- x --> " + N7, [(U, "manifest_absent")],
  "SWORN-UNCHECKED")
H("2.3b '<!-->' does not close the comment it opens", "<!--> " + N7 + " -->",
  [(M, "hidden_commitment")], "SWORN-FAILED")
H("2.3b a comment opener inside a comment does not move the comment end",
  "<!-- <!--> " + N7 + " -->", [(U, "manifest_absent")], "SWORN-UNCHECKED")
H("2.3d tag_syntax: single quotes", "x <sworn r='r1' k=\"numeric\">7 here", [(M, "tag_syntax")],
  "SWORN-FAILED", details={0: {"raw": "<sworn r='r1' k=\"numeric\">"}},
  core_checks={"canonical_sha256": None, "start": None})
H("2.3d tag_syntax: upper case opener and closer", '<SWORN r="r1" k="numeric">7</SWORN>',
  [(M, "tag_syntax"), (M, "tag_syntax")], "SWORN-FAILED",
  details={0: {"raw": '<SWORN r="r1" k="numeric">'}, 1: {"raw": "</SWORN>"}})
H("2.3d tag_syntax: raw stops before LF", '<sworn r="r1"\nk="numeric">7', [(M, "tag_syntax")],
  "SWORN-FAILED", details={0: {"raw": '<sworn r="r1"'}})
H("2.3d tag_syntax: raw runs to the end", '<sworn r="r1', [(M, "tag_syntax")], "SWORN-FAILED",
  details={0: {"raw": '<sworn r="r1'}})
H("2.3d tag_syntax: double space", '<sworn  r="r1" k="numeric">7</sworn>',
  [(M, "tag_syntax"), (M, "stray_closer")], "SWORN-FAILED")
H("2.3d not a candidate: a word character follows", "<sworns> <sworn-x> <sworn_y> <sworn9>", [],
  "UNSWORN")
H("2.3d a candidate at the end of the document", "text </sworn", [(M, "tag_syntax")],
  "SWORN-FAILED", details={0: {"raw": "</sworn"}})
H("2.3d stray_closer", "a </sworn> b", [(M, "stray_closer")], "SWORN-FAILED",
  details={0: {"raw": "</sworn>"}})
H("2.3d nesting marks both", S("r1", "numeric", "a " + S("r2", "numeric", "1") + " 2"),
  [(M, "nesting"), (M, "nesting")], "SWORN-FAILED", core_checks={"canonical_sha256": None})
H("2.4 unclosed", '<sworn r="r1" k="numeric">7', [(M, "unclosed")], "SWORN-FAILED")
H("2.3d empty_span: zero inner bytes", S("r1", "numeric", ""), [(M, "empty_span")],
  "SWORN-FAILED", details={0: {}})
H("2.3d an empty span in a comment is empty_span", "<!--" + S("r1", "numeric", "") + "-->",
  [(M, "empty_span")], "SWORN-FAILED")
H("7.1b empty_span: ASCII whitespace only", S("r1", "numeric", " \t\r\n\x0b\x0c"),
  [(M, "empty_span")], "SWORN-FAILED")
H("2.5 worked example: offsets after a two-byte character", "\u00e9" + S("r1", "numeric", "n 7"),
  [(U, "manifest_absent")], "SWORN-UNCHECKED",
  core_checks={"at": 2, "start": 2, "end": 5, "claim": "n 7"})
H("2.5 canonical text deletes openers and paired closers",
  "a " + S("r1", "numeric", "7") + " b " + S("r2", "numeric", "8") + " c",
  [(U, "manifest_absent"), (U, "manifest_absent")], "SWORN-UNCHECKED",
  core_checks={"canonical_sha256": sha(b"a 7 b 8 c")})
H("2.6 rows ordered by at: a stray closer before a span", "</sworn> " + N7,
  [(M, "stray_closer"), (U, "manifest_absent")], "SWORN-FAILED")

# --- 7.1 a to h: decided on the document's bytes
H("7.1c length_cap counts code points", S("r1", "numeric", "7" + "\u00e9" * 300),
  [(M, "length_cap")], "SWORN-FAILED",
  details={0: {"code_points": 301, "bytes": 601, "cap": 300}}, core_checks={"claim": None})
H("7.1c 300 code points is under the cap", S("r1", "numeric", "7 " + "\u00e9" * 298),
  [(U, "manifest_absent")], "SWORN-UNCHECKED")
H("7.1d kind_unknown", S("r1", "number", "7"), [(M, "kind_unknown")], "SWORN-FAILED",
  details={0: {"kind": "number"}})
H("7.1d kind_reserved", S("r1", "exec", "7"), [(M, "kind_reserved")], "SWORN-FAILED",
  details={0: {"kind": "exec"}})
H("7.1d an upper case kind is unknown", S("r1", "Numeric", "7"), [(M, "kind_unknown")],
  "SWORN-FAILED")
H("7.1d the kind is checked before the REF", S("R1", "number", "7"), [(M, "kind_unknown")],
  "SWORN-FAILED")
for ref in ("R1", "r01", "r0", "r", "r1#", "r1#L0", "r1#L01", "r1#L2-L1", "r1#L1-2", "r1#M1",
            "r1#/a~2", "r1#/a~", "path:/etc/x", "path::x", "path:a/../b", "path:a/./b",
            "path:a//b", "path:a/", "path:", "path:a b", "path:a\u00a0b", "path:a*b",
            "path:a\\b", "prereg:abc", "prereg:" + "g" * 64, "", "r1 ", "x:r1"):
    H("4 receipt_form: %r" % ref, S(ref, "numeric", "7"), [(M, "receipt_form")],
      "SWORN-FAILED", details={0: {"receipt": ref}})
H("4.3 a JSON Pointer fragment with ~0 and ~1 is well formed",
  S("r1#/a~1b/c~0d", "numeric", "7"), [(U, "manifest_absent")], "SWORN-UNCHECKED")
H("4.3 a line range is well formed", S("r1#L2-L3", "numeric", "7"), [(U, "manifest_absent")],
  "SWORN-UNCHECKED")
H("7.1f absent_over_partial: a fragment on absent", S("r1#L1", "absent", "`needle`"),
  [(M, "absent_over_partial")], "SWORN-FAILED", details={0: {"receipt": "r1#L1"}})
H("7.1f hash_over_partial", S("r1#/a", "hash", H64), [(M, "hash_over_partial")],
  "SWORN-FAILED", details={0: {"receipt": "r1#/a"}})
H("7.1f applies to path: too", S("path:a.json#/x", "absent", "`needle`"),
  [(M, "absent_over_partial")], "SWORN-FAILED")
H("7.1g directional_override", S("r1", "numeric", "7 \u202e"), [(M, "directional_override")],
  "SWORN-FAILED", details={0: {"code_point": "U+202E"}})
H("7.1g the first override is reported", S("r1", "numeric", "\u202d7\u202e"),
  [(M, "directional_override")], "SWORN-FAILED", details={0: {"code_point": "U+202D"}})
H("7.1g isolates are legal", S("r1", "numeric", "\u20667\u2069"), [(U, "manifest_absent")],
  "SWORN-UNCHECKED")
H("7.2 number_count: two numbers", S("r1", "numeric", "between 3 and 4"), [(M, "number_count")],
  "SWORN-FAILED", details={0: {"digit_bearing_tokens": ["3", "4"]}})
H("7.2 number_count: none", S("r1", "numeric", "no digits"), [(M, "number_count")],
  "SWORN-FAILED", details={0: {"digit_bearing_tokens": []}})
for tok in ("r1", "v0.1", "1e-5", "3/4", "2026-09-01", "\u0663", "1,23", "1234,567", "4%2",
            "42%%", "1.2.3", "\u00b142", "12\u201313"):
    H("7.2 number_grammar: %r" % tok, S("r1", "numeric", "value " + tok), [(M, "number_grammar")],
      "SWORN-FAILED", details={0: {"digit_bearing_tokens": [tok]}})
H("7.2 a combining mark splits a token", S("r1", "numeric", "7\u0301 apples"),
  [(U, "manifest_absent")], "SWORN-UNCHECKED")
H("7.4 needle_count: no code span", S("r1", "quote", "no code span"), [(M, "needle_count")],
  "SWORN-FAILED", details={0: {}})
H("7.4 needle_count: two code spans", S("r1", "quote", "`aaaaaaaaaaaaaaaaaa` and `b`"),
  [(M, "needle_count")], "SWORN-FAILED")
H("7.4 needle_empty", S("r1", "quote", "the ` ` here"), [(M, "needle_empty")], "SWORN-FAILED",
  details={0: {}})
H("7.4 needle_empty on absent", S("r1", "absent", "`\t`"), [(M, "needle_empty")],
  "SWORN-FAILED")
H("7.6 digest_form: a 32-digit run beside the 64", S("r1", "hash", "sha256 %s and %s" % (H64, H32)),
  [(M, "digest_form")], "SWORN-FAILED", details={0: {"hex_runs": [64, 32]}})
H("7.6 digest_form: no run", S("r1", "hash", "no digest"), [(M, "digest_form")], "SWORN-FAILED",
  details={0: {"hex_runs": []}})
H("7.6 digest_form: two 64-digit runs", S("r1", "hash", H64 + " " + H64), [(M, "digest_form")],
  "SWORN-FAILED", details={0: {"hex_runs": [64, 64]}})
H("7.6 digest_form: a 40-digit run", S("r1", "hash", H64 + " " + "a" * 40), [(M, "digest_form")],
  "SWORN-FAILED", details={0: {"hex_runs": [64, 40]}})
H("7.6 a run glued to a word character is not a run", S("r1", "hash", "sha256 %s g%s" % (H64, H32)),
  [(U, "manifest_absent")], "SWORN-UNCHECKED")
H("7.6 an upper case digest passes the form", S("r1", "hash", H64.upper()),
  [(U, "manifest_absent")], "SWORN-UNCHECKED")

# --- section 6 and the kinds, with a manifest
H("7.3 HELD at the printed precision", S("r1", "numeric", "the rate is 0.42"), [(HD, None)],
  "SWORN-HELD", manifest=M1(b"0.4211"),
  details={0: {"printed_token": "0.42", "printed": "0.42", "receipt": "0.4211",
               "receipt_rounded": "0.42", "fractional_digits": 2, "rounding": "ROUND_HALF_EVEN"}},
  core_checks={"rungs": {"L2": 1}})
H("7.3 value_mismatch", S("r1", "numeric", "n = 7"), [(F, "value_mismatch")], "SWORN-FAILED",
  manifest=M1(b"8"), details={0: {"printed": "7", "receipt": "8", "receipt_rounded": "8"}})
H("7.3 ties round to even (down)", S("r1", "numeric", "0.12"), [(HD, None)], "SWORN-HELD",
  manifest=M1(b"0.125"), details={0: {"receipt_rounded": "0.12"}})
H("7.3 ties round to even (up)", S("r1", "numeric", "0.14"), [(HD, None)], "SWORN-HELD",
  manifest=M1(b"0.135"))
H("7.3 above the tie rounds up", S("r1", "numeric", "0.13"), [(HD, None)], "SWORN-HELD",
  manifest=M1(b"0.1250001"))
H("7.3 negative values round symmetrically", S("r1", "numeric", "-2"), [(HD, None)],
  "SWORN-HELD", manifest=M1(b"-2.5"), details={0: {"receipt_rounded": "-2", "receipt": "-2.5"}})
H("7.3a rounds_to_zero: 'is 0.' against 0.4211", S("r1", "numeric", "the A-share is 0."),
  [(F, "rounds_to_zero")], "SWORN-FAILED", manifest=M1(b"0.4211"),
  details={0: {"printed_token": "0", "printed": "0", "receipt": "0.4211", "receipt_rounded": "0",
               "fractional_digits": 0}})
H("7.3a rounds_to_zero: 0.00 against 0.004", S("r1", "numeric", "p = 0.00"),
  [(F, "rounds_to_zero")], "SWORN-FAILED", manifest=M1(b"0.004"))
H("7.3a rounds_to_zero: a negative receipt", S("r1", "numeric", "p = 0.0"),
  [(F, "rounds_to_zero")], "SWORN-FAILED", manifest=M1(b"-0.04"),
  details={0: {"receipt_rounded": "0.0"}})
H("7.3a a receipt exactly zero holds against any printed zero", S("r1", "numeric", "p = 0.00"),
  [(HD, None)], "SWORN-HELD", manifest=M1(b"0E-7"), details={0: {"receipt": "0E-7"}})
H("7.3a 0.000 holds against 0", S("r1", "numeric", "zero: 0"), [(HD, None)], "SWORN-HELD",
  manifest=M1(b"0.000"), details={0: {"receipt": "0.000", "receipt_rounded": "0"}})
for printed, rec, verdict, reason, want in (
        ("1.5", b"1.50", HD, None, {"receipt": "1.50"}),
        ("1000", b"100e1", HD, None, {"receipt": "1.00E+3"}),
        ("0", b"0.001e322", F, "value_mismatch", {"receipt": "1E+319"}),
        ("0", b"0.000e323", HD, None, {"receipt": "0E+320"}),
        ("1", b"0.0001e324", F, "value_mismatch", {"receipt": "1E+320"}),
        ("0", b"0.0001e322", F, "value_mismatch", {"receipt": "1E+318"}),
        ("0.0000001", b"1e-7", HD, None, {"receipt": "1E-7"}),
        ("0", b"-0", HD, None, {"receipt": "-0", "receipt_rounded": "0"}),
        ("0.000001", b"1E-6", HD, None, {"receipt": "0.000001"}),
        ("42", b"4.2e1", HD, None, {"receipt": "42"}),
        ("0.5", b"5e-1", HD, None, {"receipt": "0.5"})):
    H("7.3 receipt number %s against %s" % (rec.decode(), printed),
      S("r1", "numeric", "value " + printed), [(verdict, reason)],
      "SWORN-HELD" if verdict == HD else "SWORN-FAILED", manifest=M1(rec), details={0: want})
for rec, leaf in ((b"1e321", "1E+321"), (b"0e400", "0E+400"),
                  (b"1e99999999999999999999", "1E+99999999999999999999")):
    H("7.3 leaf_not_numeric: adjusted exponent above 320 (%s)" % rec.decode(),
      S("r1", "numeric", "value 0"), [(M, "leaf_not_numeric")], "SWORN-FAILED",
      manifest=M1(rec), details={0: {"leaf": leaf}},
      note=("13.4: the text classes this by 7.3's bound; today's code stops with an error "
            "(decimal.InvalidOperation)") if b"9999" in rec else None)
H("7.3 leaf_not_numeric: a written exponent of 5000 digits", S("r1", "numeric", "value 0"),
  [(M, "leaf_not_numeric")], "SWORN-FAILED", manifest=M1(b"1e" + b"9" * 5000),
  details={0: {"leaf": ("1E+" + "9" * 5000)[:80]}},
  note="13.4: the text classes this by 7.3's bound; today's code stops with an error")
H("7.3 a negative exponent of 5000 digits is compared, and rounds to zero",
  S("r1", "numeric", "value 0"), [(F, "rounds_to_zero")], "SWORN-FAILED",
  manifest=M1(b"1e-" + b"9" * 5000), details={0: {"receipt_rounded": "0"}},
  note=("7.3 gives this number a coefficient and exponent like any other (adjusted exponent far "
        "below 320); today's code stops with an error"))
H("6.11 a line range written with 5000 digits is out of range", S("r1#L1-L" + "9" * 5000,
                                                                   "numeric", "7"),
  [(M, "anchor_out_of_range")], "SWORN-FAILED", manifest=M1(b"7"),
  note="today's code stops with an error converting the line number")
for token, rec, want in (("42.", b"42", {"printed_token": "42", "printed": "42"}),
                         ("+42", b"42", {"printed_token": "+42", "printed": "42"}),
                         ("42%", b"42", {"printed_token": "42%", "printed": "42"}),
                         ("\u22123", b"-3", {"printed_token": "\u22123", "printed": "-3"}),
                         (".5", b"0.5", {"printed": "0.5"}),
                         ("1,234.5", b"1234.5", {"printed_token": "1,234.5", "printed": "1234.5"}),
                         ("12,345,678", b"12345678", {"printed": "12345678"}),
                         ("007", b"7", {"printed": "7"}),
                         ("-0.00", b"0", {"printed": "0.00"}),
                         ("3:", b"3", {"printed_token": "3"})):
    H("7.3 printed token %r" % token, S("r1", "numeric", "n " + token), [(HD, None)],
      "SWORN-HELD", manifest=M1(rec), details={0: want})
H("12.1 the words around the number are not read", S("r1", "numeric", "latency fell to 240"),
  [(HD, None)], "SWORN-HELD", manifest=M1(b"240"))
H("6.11 numeric over a line slice", S("r1#L2", "numeric", "240"), [(HD, None)], "SWORN-HELD",
  manifest=M1(b"7\n240\n"))
H("6.11 CR is kept: a slice holding CR is still a JSON value", S("r1#L1", "numeric", "7"),
  [(HD, None)], "SWORN-HELD", manifest=M1(b"7\r\n8\n"))
H("7.3 receipt_not_json: a line slice is not a token search", S("r1#L2", "numeric", "latency 240"),
  [(M, "receipt_not_json")], "SWORN-FAILED", manifest=M1(LOG), core_checks={"provenance": True})
H("7.3 receipt_not_json: the whole receipt", S("r1", "numeric", "latency 240"),
  [(M, "receipt_not_json")], "SWORN-FAILED", manifest=M1(LOG), core_checks={"provenance": True})
H("6.12 receipt_not_json: a pointer into bytes that are not JSON", S("r1#/p95", "numeric", "240"),
  [(M, "receipt_not_json")], "SWORN-FAILED", manifest=M1(LOG), core_checks={"provenance": False})
H("6.13 a byte order mark is not JSON", S("r1", "numeric", "7"), [(M, "receipt_not_json")],
  "SWORN-FAILED", manifest=M1(b"\xef\xbb\xbf7"))
H("6.13 a leading zero is not JSON", S("r1", "numeric", "7"), [(M, "receipt_not_json")],
  "SWORN-FAILED", manifest=M1(b"07"))
H("6.13 a non-ASCII digit is not JSON", S("r1", "numeric", "7"), [(M, "receipt_not_json")],
  "SWORN-FAILED", manifest=M1("[\u0667]".encode("utf-8")))
H("6.12 numeric through a pointer", S("r1#/p95", "numeric", "240 ms"), [(HD, None)],
  "SWORN-HELD", manifest=M1(OBJ))
H("6.12 pointer_unresolvable: missing key", S("r1#/missing", "numeric", "7"),
  [(M, "pointer_unresolvable")], "SWORN-FAILED", manifest=M1(OBJ),
  core_checks={"provenance": False})
H("6.12 pointer_unresolvable: index past the end", S("r1#/a/2", "numeric", "7"),
  [(M, "pointer_unresolvable")], "SWORN-FAILED", manifest=M1(OBJ))
H("6.12 pointer_unresolvable: an index with a leading zero", S("r1#/a/01", "numeric", "2"),
  [(M, "pointer_unresolvable")], "SWORN-FAILED", manifest=M1(OBJ))
H("6.12 pointer_unresolvable: a scalar cannot be walked", S("r1#/p95/x", "numeric", "7"),
  [(M, "pointer_unresolvable")], "SWORN-FAILED", manifest=M1(OBJ))
H("6.12 an array index", S("r1#/a/1", "numeric", "2"), [(HD, None)], "SWORN-HELD",
  manifest=M1(OBJ))
H("6.12 pointer_ambiguous", S("r1#/dup", "numeric", "2"), [(M, "pointer_ambiguous")],
  "SWORN-FAILED", manifest=M1(OBJ))
for ptr, lt in (("/obj", "object"), ("/a", "array"), ("/n", "null")):
    H("7.3 leaf_not_scalar: %s" % lt, S("r1#" + ptr, "numeric", "7"), [(M, "leaf_not_scalar")],
      "SWORN-FAILED", manifest=M1(OBJ), details={0: {"leaf_type": lt}},
      core_checks={"provenance": True})
for ptr, leaf in (("/s42", "42"), ("/t", "True"), ("/nan", "NaN"), ("/inf", "Infinity"),
                  ("/ninf", "-Infinity"), ("/x", "?x")):
    H("7.3 leaf_not_numeric: %s" % ptr, S("r1#" + ptr, "numeric", "7"), [(M, "leaf_not_numeric")],
      "SWORN-FAILED", manifest=M1(OBJ), details={0: {"leaf": leaf}})
H("7.3 leaf_not_numeric: a long string leaf is cut at 80 code points", S("r1", "numeric", "7"),
  [(M, "leaf_not_numeric")], "SWORN-FAILED", manifest=M1(('"' + "\u00e9" * 100 + '"').encode()),
  details={0: {"leaf": "\u00e9" * 80}})
H("7.5 quote HELD over the whole receipt", S("r1", "quote", "it says `the quick brown fox`"),
  [(HD, None)], "SWORN-HELD", manifest=M1(b"the quick brown fox jumps"),
  details={0: {"needle_bytes": 19, "haystack_bytes": 25, "occurrences": 1}})
H("7.5 occurrences count without overlap", S("r1", "quote", "`aaaaaaaaaaaaaaaa`"), [(HD, None)],
  "SWORN-HELD", manifest=M1(b"a" * 40), details={0: {"occurrences": 2}})
H("7.5 needle_missing", S("r1", "quote", "`the slow brown fox`"), [(F, "needle_missing")],
  "SWORN-FAILED", manifest=M1(b"the quick brown fox jumps"),
  details={0: {"occurrences": 0}})
H("7.5 the needle is untrimmed", S("r1", "quote", "` the quick brown fox `"),
  [(F, "needle_missing")], "SWORN-FAILED", manifest=M1(b"the quick brown fox jumps"))
H("7.5 short_needle over the whole receipt", S("r1", "quote", "`ok`"), [(M, "short_needle")],
  "SWORN-FAILED", manifest=M1(LOG), details={0: {"needle_bytes": 2, "minimum_bytes": 16}},
  core_checks={"provenance": True})
H("7.5 a line fragment that narrows lifts the floor", S("r1#L2", "quote", "`240`"),
  [(HD, None)], "SWORN-HELD", manifest=M1(LOG))
H("7.5 a line fragment equal to the full content does not narrow", S("r1#L1", "quote", "`ok`"),
  [(M, "short_needle")], "SWORN-FAILED", manifest=M1(b"result: all tests passed ok\n\n"))
H("7.5 a receipt under 16 bytes narrows under a line fragment", S("r1#L1", "quote", "`ok`"),
  [(HD, None)], "SWORN-HELD", manifest=M1(b"ok\n"))
H("7.5 without a fragment a receipt under 16 bytes keeps the floor", S("r1", "quote", "`ok`"),
  [(M, "short_needle")], "SWORN-FAILED", manifest=M1(b"ok\n"))
H("7.5 a pointer leaf has no floor", S("r1#/s", "quote", "`quick`"), [(HD, None)], "SWORN-HELD",
  manifest=M1(OBJ))
H("7.5 leaf_not_string", S("r1#/p95", "quote", "`240`"), [(M, "leaf_not_string")],
  "SWORN-FAILED", manifest=M1(OBJ), details={0: {}}, core_checks={"provenance": True})
H("7.5 leaf_not_string: an unpaired surrogate", S("r1#/x", "quote", "`x`"),
  [(M, "leaf_not_string")], "SWORN-FAILED", manifest=M1(OBJ),
  details={0: {"note": "leaf is not encodable as UTF-8"}})
H("6.11 anchor_out_of_range", S("r1#L4", "quote", "`240`"), [(M, "anchor_out_of_range")],
  "SWORN-FAILED", manifest=M1(LOG), core_checks={"provenance": False})
H("6.11 the last line counts without a final LF", S("r1#L2", "numeric", "8"), [(HD, None)],
  "SWORN-HELD", manifest=M1(b"7\n8"))
H("6.11 an empty receipt has no line", S("r1#L1", "quote", "`x`"), [(M, "anchor_out_of_range")],
  "SWORN-FAILED", manifest=M1(b""))
H("7.7 hash HELD over bytes", S("r1", "hash", "sha256 " + sha(b"artifact bytes")), [(HD, None)],
  "SWORN-HELD", manifest=M1(b"artifact bytes"))
H("7.7 hash HELD on the manifest's word (12.9)", S("r1", "hash", H64.upper()), [(HD, None)],
  "SWORN-HELD", manifest=manifest({"r1": entry(None, sha256=H64)}),
  details={0: {"stated": H64, "receipt_sha256": H64}})
H("7.7 digest_mismatch", S("r1", "hash", H64), [(F, "digest_mismatch")], "SWORN-FAILED",
  manifest=M1(b"artifact bytes"))
H("7.8 absent HELD", S("r1", "absent", "`password`"), [(HD, None)], "SWORN-HELD",
  manifest=M1(b"all clear"), details={0: {"needle_bytes": 8, "haystack_bytes": 9, "complete": True}})
H("7.8 needle_present", S("r1", "absent", "`pw`"), [(F, "needle_present")], "SWORN-FAILED",
  manifest=M1(b"the pw is here"))
H("7.1j absent_over_partial: complete false", S("r1", "absent", "`pw`"),
  [(M, "absent_over_partial")], "SWORN-FAILED", manifest=M1(b"x", complete=False),
  details={0: {"complete": False}}, core_checks={"provenance": True})
H("7.1j manifest_no_completeness: complete missing", S("r1", "absent", "`pw`"),
  [(U, "manifest_no_completeness")], "SWORN-UNCHECKED", manifest=M1(b"x", drop=("complete",)))
H("7.1j manifest_no_completeness: complete not a boolean", S("r1", "absent", "`pw`"),
  [(U, "manifest_no_completeness")], "SWORN-UNCHECKED", manifest=M1(b"x", complete="yes"))
H("6.10 manifest_bytes_absent on absent", S("r1", "absent", "`pw`"),
  [(U, "manifest_bytes_absent")], "SWORN-UNCHECKED", manifest=manifest({"r1": entry(None, sha256=H64)}))
H("6.10 manifest_bytes_absent on numeric", N7, [(U, "manifest_bytes_absent")],
  "SWORN-UNCHECKED", manifest=manifest({"r1": entry(None, sha256=H64)}))
H("6.1 manifest_absent", N7, [(U, "manifest_absent")], "SWORN-UNCHECKED")
H("6.2 manifest_integrity: the declared digest does not re-derive", N7,
  [(U, "manifest_integrity")], "SWORN-UNCHECKED",
  manifest=manifest({"r1": entry(b"7")}, top={"digest": "0" * 64}))
H("6.2 manifest_integrity: a digest in upper case does not re-derive", N7,
  [(U, "manifest_integrity")], "SWORN-UNCHECKED",
  manifest=json.dumps(dict(json.loads(manifest({"r1": entry(b"7")})),
                           digest=json.loads(manifest({"r1": entry(b"7")}))["digest"].upper()),
                      sort_keys=True).encode("utf-8"))
H("5.6 a null digest is absent: the manifest is intact", N7, [(HD, None)], "SWORN-HELD",
  manifest=manifest({"r1": entry(b"7")}, with_digest=False, top={"digest": None}))
H("5.6 no digest key: intact", N7, [(HD, None)], "SWORN-HELD",
  manifest=manifest({"r1": entry(b"7")}, with_digest=False))
H("5.4 keys beyond the seven are ignored", N7, [(HD, None)], "SWORN-HELD",
  manifest=manifest({"r1": entry(b"7")}, top={"extra": {"z": 1.5}}))
for label, poison in (("a non-integer number", {"captured_at": 1.5}),
                      ("an integer outside 3.1", {"n": 9007199254740992}),
                      ("a key above U+007E", {"\u00e9": 1})):
    H("5.6 manifest_integrity: %s in an entry (13.14)" % label, N7, [(U, "manifest_integrity")],
      "SWORN-UNCHECKED", manifest=manifest({"r1": entry(b"7", **poison)}, with_digest=False),
      core_checks={"manifest_digest": None},
      note="stricter than today's code on purpose (RECEIPT_v0.md 13.14, third item)")
H("5.6 manifest_integrity: NaN with no digest key (13.14)", N7, [(U, "manifest_integrity")],
  "SWORN-UNCHECKED",
  manifest=manifest({"r1": entry(b"7")}, with_digest=False).replace(b'"complete": true',
                                                                     b'"complete": true, "w": NaN'),
  core_checks={"manifest_digest": None},
  note="stricter than today's code on purpose (RECEIPT_v0.md 13.14, fourth item)")
H("5.6 manifest_integrity: an unpaired surrogate with no digest key (13.14)", N7,
  [(U, "manifest_integrity")], "SWORN-UNCHECKED",
  manifest=manifest({"r1": entry(b"7")}, with_digest=False).replace(b'"harness-x"',
                                                                     b'"harness-\\ud800"'),
  core_checks={"manifest_digest": None},
  note="stricter than today's code on purpose (RECEIPT_v0.md 13.14, fourth item)")
H("5.6 a broken manifest still refuses nothing a span did not cite", "plain text", [],
  "UNSWORN", manifest=manifest({"r1": entry(b"7")}, top={"digest": "0" * 64}))
H("6.3 rung_unknown: L3", N7, [(U, "rung_unknown")], "SWORN-UNCHECKED", manifest=M1(b"7", rung="L3"))
H("6.3 rung_unknown: a number", N7, [(U, "rung_unknown")], "SWORN-UNCHECKED",
  manifest=M1(b"7", rung=2))
H("6.9 rung null reads undeclared", N7, [(HD, None)], "SWORN-HELD", manifest=M1(b"7", rung=None),
  core_checks={"rungs": {"undeclared": 1}})
H("6.9 a missing rung reads undeclared", N7, [(HD, None)], "SWORN-HELD",
  manifest=manifest({"r1": entry(b"7")}, drop_top=("rung",)), core_checks={"rungs": {"undeclared": 1}})
H("6.9 rung L1", N7, [(HD, None)], "SWORN-HELD", manifest=M1(b"7", rung="L1"),
  core_checks={"rungs": {"L1": 1}})
H("6.4 manifest_id_missing (D1: nothing checked is not held)", S("r9", "numeric", "n is 7"),
  [(U, "manifest_id_missing")], "SWORN-UNCHECKED", manifest=M1(b"7"))
H("9.2 one HELD beside an UNRESOLVED is SWORN-HELD (D1)", N7 + " " + S("r9", "numeric", "8"),
  [(HD, None), (U, "manifest_id_missing")], "SWORN-HELD", manifest=M1(b"7"))
H("6.5 manifest_integrity: an entry that is not an object", N7, [(U, "manifest_integrity")],
  "SWORN-UNCHECKED", manifest=manifest({"r1": "x"}))
H("6.5 manifest_integrity: an upper case entry sha256", N7, [(U, "manifest_integrity")],
  "SWORN-UNCHECKED", manifest=manifest({"r1": entry(b"7", sha256=sha(b"7").upper())}))
H("6.5 the key names the entry, not its id", N7, [(HD, None)], "SWORN-HELD",
  manifest=manifest({"r1": dict(entry(b"7"), id="r2")}))
H("6.6 kind_of_source_unknown: not a string", N7, [(M, "kind_of_source_unknown")],
  "SWORN-FAILED", manifest=M1(b"7", kos=5), core_checks={"provenance": False})
H("6.8 kind_of_source_unknown: not in the vocabulary", N7, [(M, "kind_of_source_unknown")],
  "SWORN-FAILED", manifest=M1(b"7", kos="stdout"))
H("6.7 receipt_author_minted: author-side kind", N7, [(M, "receipt_author_minted")],
  "SWORN-FAILED", manifest=M1(b"7", kos="agent_output"), core_checks={"provenance": False})
H("6.7 receipt_author_minted: listed in authored_sha256 (upper case, rebuilt)", N7,
  [(M, "receipt_author_minted")], "SWORN-FAILED",
  manifest=manifest({"r1": entry(b"7")}, authored=[sha(b"7").upper()]))
H("6.7 is checked before 6.8", N7, [(M, "receipt_author_minted")], "SWORN-FAILED",
  manifest=manifest({"r1": entry(b"7", kos="stdout")}, authored=[sha(b"7")]))
H("6.6 is checked before 6.7", N7, [(M, "kind_of_source_unknown")], "SWORN-FAILED",
  manifest=manifest({"r1": entry(b"7", kos=5)}, authored=[sha(b"7")]))
for kos in ("tool_stderr", "file_read", "http_fetch", "harness_note", "test_report", "attestation"):
    H("5.2 external kind_of_source %s" % kos, N7, [(HD, None)], "SWORN-HELD",
      manifest=M1(b"7", kos=kos))
for kos in ("agent_file_write", "agent_message"):
    H("5.2 author-side kind_of_source %s" % kos, N7, [(M, "receipt_author_minted")],
      "SWORN-FAILED", manifest=M1(b"7", kos=kos))
H("6.10 manifest_integrity: bytes that do not hash to sha256", N7, [(U, "manifest_integrity")],
  "SWORN-UNCHECKED", manifest=manifest({"r1": entry(b"7", sha256=sha(b"8"))}))
for label, value in (("null", None), ("a number", 5)):
    e = entry(b"7")
    e["bytes"] = value
    H("6.10 manifest_integrity: bytes %s" % label, N7, [(U, "manifest_integrity")],
      "SWORN-UNCHECKED", manifest=manifest({"r1": e}))
A_SHA = sha(b"A")
for enc, verdict, reason in (("QQ==", HD, None), ("QR==", HD, None), ("QQ", U, "manifest_integrity"),
                             ("QQ=", U, "manifest_integrity"), ("Q Q==", U, "manifest_integrity"),
                             ("QQ==QQ==", U, "manifest_integrity"),
                             ("QQ==\n", U, "manifest_integrity"), ("QQ-=", U, "manifest_integrity"),
                             ("=QQ=", U, "manifest_integrity")):
    e = entry(None, sha256=A_SHA)
    e["bytes"] = enc
    H("6.10 strict base64 %r" % enc, S("r1", "hash", A_SHA), [(verdict, reason)],
      "SWORN-HELD" if verdict == HD else "SWORN-UNCHECKED", manifest=manifest({"r1": e}),
      note="13.16: non-zero pad bits are ignored" if enc == "QR==" else None)
e = entry(None, sha256=sha(b""))
e["bytes"] = ""
H("6.10 the empty string decodes to zero bytes", S("r1", "hash", sha(b"")), [(HD, None)],
  "SWORN-HELD", manifest=manifest({"r1": e}))
H("4.4 no_repository: path:", S("path:docs/a.md", "quote", "`some long quoted text`"),
  [(U, "no_repository")], "SWORN-UNCHECKED", core_checks={"provenance": False})
H("4.4 no_repository: prereg:", S("prereg:" + H64.upper(), "hash", H64), [(U, "no_repository")],
  "SWORN-UNCHECKED")
H("4.4 no_repository with a manifest given", S("path:a.json#/x", "numeric", "7"),
  [(U, "no_repository")], "SWORN-UNCHECKED", manifest=M1(b"7"))
H("4.4 a path: span still meets 7.1 h", S("path:a.md", "numeric", "v0.1"), [(M, "number_grammar")],
  "SWORN-FAILED")
H("9.1 rungs worked example: one HELD on L2 and one tag_syntax",
  N7 + " <sworn>", [(HD, None), (M, "tag_syntax")], "SWORN-FAILED", manifest=M1(b"7"),
  core_checks={"rungs": {"L2": 1, "unresolved": 1}})
H("8 D3: the claim and its hash, inner bytes untrimmed", S("r1", "numeric", " n is 7 "),
  [(HD, None)], "SWORN-HELD", manifest=M1(b"7"),
  core_checks={"claim": " n is 7 ", "claim_sha256": sha(b" n is 7 ")})
H("8 D3: a hidden commitment's claim is bound", "<!--" + N7 + "-->", [(M, "hidden_commitment")],
  "SWORN-FAILED", core_checks={"claim": "n is 7"})
H("9.1 the commit label is carried, 40 digits", N7, [(U, "manifest_absent")], "SWORN-UNCHECKED",
  commit="a" * 40, core_checks={"commit": "a" * 40})
H("9.1 the commit label is carried, 64 digits", N7, [(U, "manifest_absent")], "SWORN-UNCHECKED",
  commit="b" * 64)
H("13.6 the name is inside the digest", N7, [(U, "manifest_absent")], "SWORN-UNCHECKED",
  name="other.md")
H("a name may be any Unicode text", N7, [(U, "manifest_absent")], "SWORN-UNCHECKED",
  name="r\u00e9sum\u00e9 \U0001f600.md")

AUTH_B, AUTH_A = "B" * 64, "a" * 64
H("5.4 authored_sha256 rebuilt: [BB..B, aa..a, aa..a]", N7, [(HD, None)], "SWORN-HELD",
  manifest=manifest({"r1": entry(b"7")}, authored=[AUTH_B, AUTH_A, AUTH_A], with_digest=False))
H("5.4 authored_sha256 rebuilt: [aa..a, aa..a, bb..b] gives the same digest", N7, [(HD, None)],
  "SWORN-HELD",
  manifest=manifest({"r1": entry(b"7")}, authored=[AUTH_A, AUTH_A, "b" * 64], with_digest=False))
H("5.4 authored_sha256 rebuilt: [aa..a, bb..b] gives another digest", N7, [(HD, None)],
  "SWORN-HELD", manifest=manifest({"r1": entry(b"7")}, authored=[AUTH_A, "b" * 64],
                                  with_digest=False))
H("5.5 a missing receipts reads as {}", N7, [(U, "manifest_id_missing")], "SWORN-UNCHECKED",
  manifest=manifest({}, drop_top=("receipts",), with_digest=False))
H("5.5 a null receipts reads as {}", N7, [(U, "manifest_id_missing")], "SWORN-UNCHECKED",
  manifest=manifest({}, with_digest=False, top={"receipts": None}))
H("5.5 a null authored_sha256 reads as []", N7, [(HD, None)], "SWORN-HELD",
  manifest=manifest({"r1": entry(b"7")}, with_digest=False, top={"authored_sha256": None}))

# --- 11.1 refusals
R("11.1 nothing_to_verify", None, "nothing_to_verify")
R("11.1 name_form: an unpaired surrogate", N7, "name_form", name="d\udcff.md")
R("11.1 name_form: not a string", N7, "name_form", name=None)
R("11.1 commit_form: upper case", N7, "commit_form", commit="A" * 40)
R("11.1 commit_form: 41 digits", N7, "commit_form", commit="a" * 41)
R("11.1 commit_form: empty", N7, "commit_form", commit="")
MAN_OK = json.loads(manifest({"r1": entry(b"7")}, with_digest=False))


def mbytes(**edits):
    m = dict(MAN_OK)
    for k, v in edits.items():
        if v is KeyError:
            m.pop(k)
        else:
            m[k] = v
    return json.dumps(m, sort_keys=True).encode("utf-8")


R("5.5 manifest_spec: 0.1 (13.14)", N7, "manifest_spec", manifest=mbytes(spec="sworn/manifest/0.1"),
  rules=["5.5", "11.1"], note="stricter than today's code on purpose (RECEIPT_v0.md 13.14)")
R("5.5 manifest_spec: spec missing", N7, "manifest_spec", manifest=mbytes(spec=KeyError))
R("5.5 manifest_spec: not an object", N7, "manifest_spec", manifest=b"[]")
R("5.5 manifest_spec: null", N7, "manifest_spec", manifest=b"null")
R("5.5 manifest_spec: bytes that are not JSON (text gap: no code named)", N7, "manifest_spec",
  manifest=b"{not json", note="the text names no code for manifest bytes that do not parse; "
  "this checker reads them as not an object (11.1)")
R("5.5 manifest_spec: a byte order mark", N7, "manifest_spec",
  manifest=b"\xef\xbb\xbf" + mbytes(), note="6.13: UTF-8 without a byte order mark")
R("5.5 manifest_receipts_shape", N7, "manifest_receipts_shape", manifest=mbytes(receipts=[]))
R("5.5 manifest_authored_shape: not 64 hex", N7, "manifest_authored_shape",
  manifest=mbytes(authored_sha256=["xyz"]), note="stricter than today on purpose (13.14)")
R("5.5 manifest_authored_shape: U+0130 (13.14)", N7, "manifest_authored_shape",
  manifest=mbytes(authored_sha256=["\u0130" + "a" * 63]))
R("5.5 manifest_authored_shape: a string, not an array", N7, "manifest_authored_shape",
  manifest=mbytes(authored_sha256="a" * 64))
R("5.5 manifest_form: minted_at missing (13.14)", N7, "manifest_form",
  manifest=mbytes(minted_at=KeyError), note="stricter than today's code on purpose (13.14)")
R("5.5 manifest_form: harness not a string (13.14)", N7, "manifest_form", manifest=mbytes(harness=5))
R("5.5 manifest_form: turn null", N7, "manifest_form", manifest=mbytes(turn=None))
R("5.5 worked example: a receipts key written twice (13.14)", N7, "manifest_form",
  manifest=mbytes().replace(b'"receipts": {', b'"receipts": {"r1": {}, ', 1),
  note="stricter than today's code on purpose (13.14): today the last value wins")
R("5.5 a key written twice in an ignored top-level key still refuses", N7, "manifest_form",
  manifest=mbytes()[:-1] + b', "x": {"a": 1, "a": 2}}')
R("5.5 order: manifest_spec before manifest_form", N7, "manifest_spec",
  manifest=mbytes(spec="sworn/manifest/0.1").replace(b'"receipts": {', b'"receipts": {"r1": {}, ', 1))
R("5.5 order: manifest_receipts_shape before manifest_authored_shape", N7,
  "manifest_receipts_shape", manifest=mbytes(receipts=5, authored_sha256=5))
R("5.5 order: manifest_authored_shape before manifest_form", N7, "manifest_authored_shape",
  manifest=mbytes(authored_sha256=["x"], harness=5))
R("11.1 order: name_form before commit_form", N7, "name_form", name="\ud800", commit="X")
R("11.1 order: commit_form before the manifest", N7, "commit_form", commit="X", manifest=b"[]")
R("11.1 a refused manifest refuses a document with no span", "plain", "manifest_spec",
  manifest=b"[]")


def hand_vectors():
    out = []
    for h in HAND:
        doc = h["doc"]
        if isinstance(doc, str):
            doc = doc.encode("utf-8")
        inputs = {"document_b64": b64(doc), "name": h["name"], "manifest_b64": b64(h["manifest"]),
                  "commit": h["commit"]}
        vec = {"id": vector_id("check", inputs), "source": "hand", "op": "check",
               "label": h["label"], "inputs": inputs, "note": h.get("note")}
        try:
            core = C.check(doc, h["name"], h["manifest"], h["commit"])
            got_refusal = None
        except C.Refusal as e:
            core, got_refusal = None, e.code
        if "refusal" in h:
            if got_refusal != h["refusal"]:
                raise SystemExit("hand vector %r: the text says refusal %s, check_v0 gives %s"
                                 % (h["label"], h["refusal"], got_refusal or "a core"))
            vec["expect"] = {"refusal": h["refusal"]}
            vec["rules"] = h["rules"]
        else:
            if core is None:
                raise SystemExit("hand vector %r: check_v0 refused %s" % (h["label"], got_refusal))
            rows = [[r["verdict"], r["reason"]] for r in core["spans"]]
            want = [list(x) for x in h["rows"]]
            if rows != want or core["document_verdict"] != h["dv"]:
                raise SystemExit("hand vector %r: the text says %r %s, check_v0 gives %r %s"
                                 % (h["label"], want, h["dv"], rows, core["document_verdict"]))
            for i, sub in h["details"].items():
                d = core["spans"][i]["detail"]
                if sub == {}:
                    ok = d == {}
                else:
                    ok = all(d.get(k, KeyError) == v for k, v in sub.items())
                if not ok:
                    raise SystemExit("hand vector %r row %d: detail %r, the text says %r"
                                     % (h["label"], i, d, sub))
            for k, v in h["core_checks"].items():
                if k in ("canonical_sha256",):
                    got = core["document"][k]
                elif k in ("document_malformed", "rungs", "manifest_digest", "commit"):
                    got = core[k]
                elif k == "provenance":
                    got = "provenance" in core["spans"][0]
                else:
                    got = core["spans"][0][k]
                if got != v:
                    raise SystemExit("hand vector %r: %s is %r, the text says %r"
                                     % (h["label"], k, got, v))
            vec["expect"] = {"core_digest": C.digest(core)}
            vec["expect_rows"] = want
            vec["expect_document_verdict"] = h["dv"]
            vec["rules"] = h["rules"] or rules_of(core, h["manifest"] is not None)
        out.append(vec)
    # 5.4's worked example, as a relation between three vectors
    by = {v["label"]: v for v in out}
    a = by["5.4 authored_sha256 rebuilt: [BB..B, aa..a, aa..a]"]["expect"]["core_digest"]
    b = by["5.4 authored_sha256 rebuilt: [aa..a, aa..a, bb..b] gives the same digest"]["expect"]["core_digest"]
    c = by["5.4 authored_sha256 rebuilt: [aa..a, bb..b] gives another digest"]["expect"]["core_digest"]
    if not (a == b != c):
        raise SystemExit("5.4's worked example does not hold")
    return out


# ---------------------------------------------------------------------------------------------
# (c) receipt checks, 11.3
# ---------------------------------------------------------------------------------------------

def recheck_vectors():
    out = []
    doc = (N7 + " and " + S("r2", "quote", "`the quick brown fox`")).encode("utf-8")
    man = manifest({"r1": entry(b"7"), "r2": entry(b"the quick brown fox jumps")})
    core = C.check(doc, "d.md", man, None)
    good = C.jcs(C.receipt(core)).encode("utf-8")

    def rv(label, receipt_bytes, expect, document=doc, mani=man, note=None):
        inputs = {"receipt_b64": b64(receipt_bytes), "document_b64": b64(document),
                  "manifest_b64": b64(mani)}
        try:
            got = {"answer": C.recheck(receipt_bytes, document, mani)}
        except C.Refusal as e:
            got = {"refusal": e.code}
        if got != expect:
            raise SystemExit("recheck vector %r: the text says %r, check_v0 gives %r"
                             % (label, expect, got))
        out.append({"id": vector_id("recheck", inputs), "source": "recheck", "op": "recheck",
                    "label": label, "inputs": inputs, "expect": expect, "rules": ["11.3"],
                    "note": note})

    def remint(c, **top):
        r = {"core": c, "digest": C.digest(c), "attestations": []}
        r.update(top)
        return C.jcs(r).encode("utf-8")

    def edited(fn):
        c = json.loads(good)["core"]
        fn(c)
        return c

    rv("REDERIVES", good, {"answer": "REDERIVES"})
    rv("REDERIVES: verifier is provenance, never read",
       remint(core, verifier={"name": "anything", "version": 3}), {"answer": "REDERIVES"})
    rv("REDERIVES: key order and whitespace are not the canonical form",
       json.dumps(json.loads(good), indent=2).encode("utf-8"), {"answer": "REDERIVES"})
    nm = C.check(b"plain text", "n.md", None, None)
    rv("REDERIVES: no manifest, no span", C.jcs(C.receipt(nm)).encode(), {"answer": "REDERIVES"},
       document=b"plain text", mani=None)
    cm = C.check(doc, "d.md", None, "c" * 40)
    rv("REDERIVES: the commit label the core records is used", C.jcs(C.receipt(cm)).encode(),
       {"answer": "REDERIVES"}, mani=None)
    rv("DOES-NOT-REDERIVE: a verdict edited and the digest re-minted",
       remint(edited(lambda c: c.update(document_verdict="SWORN-FAILED"))),
       {"answer": "DOES-NOT-REDERIVE"})
    rv("DOES-NOT-REDERIVE: a claim edited and the digest re-minted",
       remint(edited(lambda c: c["spans"][0].update(claim="n is 8"))),
       {"answer": "DOES-NOT-REDERIVE"})
    rv("DOES-NOT-REDERIVE: another manifest", good, {"answer": "DOES-NOT-REDERIVE"},
       mani=manifest({"r1": entry(b"7"), "r2": entry(b"the quick brown fox jumps")}, rung="L1"))
    rv("DOES-NOT-REDERIVE: no manifest handed over", good, {"answer": "DOES-NOT-REDERIVE"},
       mani=None)
    rv("receipt_shape: not JSON", b"{", {"refusal": "receipt_shape"})
    rv("receipt_shape: document_verdict written twice",
       good.replace(b'"document_verdict":', b'"document_verdict":"SWORN-HELD","document_verdict":', 1),
       {"refusal": "receipt_shape"})
    rv("receipt_shape: attestations missing",
       C.jcs({"core": core, "digest": C.digest(core)}).encode(), {"refusal": "receipt_shape"})
    rv("receipt_shape: a fourteenth core key", remint(edited(lambda c: c.update(extra=1))),
       {"refusal": "receipt_shape"})
    rv("receipt_shape: a count written 1.0 (text gap)",
       good.replace(b'"sworn_total":2', b'"sworn_total":2.0', 1), {"refusal": "receipt_shape"},
       note="JSON Schema calls 2.0 an integer; 3.1 calls it a float. This checker refuses it as "
            "receipt_shape (see the writer's report)")
    rv("receipt_shape: a reason outside the closed list",
       remint(edited(lambda c: c["spans"][0].update(verdict="FAILED", reason="WITHHELD"))),
       {"refusal": "receipt_shape"})
    rv("receipt_shape: an unknown detail key",
       remint(edited(lambda c: c["spans"][0]["detail"].update(extra="x"))),
       {"refusal": "receipt_shape"})
    rv("receipt_shape: a byte order mark", b"\xef\xbb\xbf" + good, {"refusal": "receipt_shape"})
    rv("attestations_not_empty", remint(core, attestations=[{"by": "x"}]),
       {"refusal": "attestations_not_empty"},
       note="the schema bounds attestations at zero items; this checker answers 11.3's own code")
    rv("receipt_digest: one digit changed",
       good.replace(b'"digest":"' + C.digest(core).encode(),
                    b'"digest":"' + ("0" if C.digest(core)[0] != "0" else "1").encode()
                    + C.digest(core)[1:].encode()), {"refusal": "receipt_digest"})
    rv("receipt_digest: a core with no canonical form (text gap)",
       good.replace(b'"claim":"n is 7"', b'"claim":"n is 7\\ud800"', 1), {"refusal": "receipt_digest"},
       note="the core holds an unpaired surrogate: its digest cannot be computed")
    rv("profile_not_checkable", remint(edited(lambda c: c.update(profile="legacy-projection/v0-draft"))),
       {"refusal": "profile_not_checkable"})
    rv("document_mismatch", good, {"refusal": "document_mismatch"}, document=doc + b" ")
    rv("order: receipt_digest before profile_not_checkable",
       C.jcs({"core": edited(lambda c: c.update(profile="legacy-projection/v0-draft")),
              "digest": "0" * 64, "attestations": []}).encode(), {"refusal": "receipt_digest"})
    rv("order: profile_not_checkable before document_mismatch",
       remint(edited(lambda c: c.update(profile="legacy-projection/v0-draft"))),
       {"refusal": "profile_not_checkable"}, document=b"other")
    rv("a manifest refusal while re-deriving", good, {"refusal": "manifest_spec"}, mani=b"[]")
    return out


# ---------------------------------------------------------------------------------------------
# (d) the canonical form
# ---------------------------------------------------------------------------------------------

def jcs_vectors():
    cases = [
        (b'{"b": 1, "a": [true, false, null, "x"]}', {"canonical": '{"a":[true,false,null,"x"],"b":1}'}),
        (b'"\\u0001\\u001f\\b\\t\\n\\f\\r\\"\\\\/\\u007f\\u2028\\u00e9"',
         {"canonical": '"\\u0001\\u001f\\b\\t\\n\\f\\r\\"\\\\/\x7f\u2028\u00e9"'}),
        (b"9007199254740991", {"canonical": "9007199254740991"}),
        (b"-9007199254740991", {"canonical": "-9007199254740991"}),
        (b"-0", {"canonical": "0"}),
        (b'{"\\u0000": 1, "~": 2, "A": 3, "a": 4}', {"canonical": '{"\\u0000":1,"A":3,"a":4,"~":2}'}),
        (b"9007199254740992", {"refusal": "int_range"}),
        (b"9007199254740993", {"refusal": "int_range"}),
        (b"-9007199254740992", {"refusal": "int_range"}),
        (b"1.0", {"refusal": "float"}),
        (b"1e2", {"refusal": "float"}),
        (b"0.5", {"refusal": "float"}),
        (b"NaN", {"refusal": "float"}),
        (b"-Infinity", {"refusal": "float"}),
        (b'"\\ud800"', {"refusal": "lone_surrogate"}),
        (b'"\\udc00\\ud800"', {"refusal": "lone_surrogate"}),
        (b'"\\ud83d\\ude00"', {"canonical": '"\U0001f600"'}),
        (b'{"\\u007f": 1}', {"refusal": "non_ascii_key"}),
        (b'{"\\u00e9": 1}', {"refusal": "non_ascii_key"}),
        (b"[" * 32 + b"]" * 32, {"canonical": "[" * 32 + "]" * 32}),
        (b"[" * 33 + b"]" * 33, {"refusal": "depth"}),
        (b'{"a":' * 32 + b"1" + b"}" * 32, {"canonical": '{"a":' * 32 + "1" + "}" * 32}),
        (b'{"a":' * 33 + b"1" + b"}" * 33, {"refusal": "depth"}),
    ]
    out = []
    for value, expect in cases:
        inputs = {"value_b64": b64(value)}
        v = C.parse_json(value)
        try:
            got = {"canonical": C.jcs(v)}
        except C.Refusal as e:
            got = {"refusal": e.code}
        if got != expect:
            raise SystemExit("jcs vector %r: the text says %r, check_v0 gives %r" % (value, expect, got))
        out.append({"id": vector_id("jcs", inputs), "source": "jcs", "op": "jcs", "inputs": inputs,
                    "expect": expect, "rules": ["3.1", "3.2", "3.3", "6.13"], "note": None})
    return out


def build():
    conf = conformance_vectors()
    hand = hand_vectors()
    rech = recheck_vectors()
    jv = jcs_vectors()
    # a hand vector whose inputs equal a conformance vector's is merged into it: the two sources
    # must then expect the same thing, which is itself a check of the hand reading
    seen = {}
    vectors = []
    merged = 0
    for v in conf + hand + rech + jv:
        if v["id"] in seen:
            first = seen[v["id"]]
            if first["expect"] != v["expect"]:
                raise SystemExit("vector %s: two sources expect different things" % v["id"])
            first.setdefault("also", []).append({"source": v["source"], "label": v.get("label"),
                                                 "expect_rows": v.get("expect_rows")})
            merged += 1
            continue
        seen[v["id"]] = v
        vectors.append(v)
    hand = [v for v in vectors if v["source"] == "hand"]
    produced = set()
    refusals = set()
    for v in vectors:
        if "refusal" in v["expect"]:
            refusals.add(v["expect"]["refusal"])
        for row in v.get("expect_rows", []):
            if row[1] is not None:
                produced.add(row[1])
    return {
        "schema": "receipt/v0-draft/vectors",
        "status": "DRAFT, unsigned. Lab-written vectors for a lab draft; not a standard.",
        "text": "receipt/v0/RECEIPT_v0.md",
        "id_rule": ("sha256 of the UTF-8 canonical text (section 3) of {op, inputs}; an input string "
                    "with no canonical form is replaced inside the id object by "
                    "{utf16be_hex: its UTF-16BE code units in lowercase hex}"),
        "expect_rule": {
            "conformance": ("digest_v0(to_v0(sworn.verify(...))) at the commit this file was built "
                            "from, except where a note cites RECEIPT_v0.md 13.14"),
            "hand": ("rows and refusals as the text gives them; core_digest is check_v0's, recorded "
                     "only after its rows matched expect_rows"),
            "recheck": "11.3's answer or refusal",
            "jcs": "section 3's canonical text or refusal code, the value read by 6.13",
        },
        "counts": {"conformance": len(conf), "hand": len(hand), "hand_merged_into_conformance": merged,
                   "recheck": len(rech),
                   "jcs": len(jv), "total": len(vectors)},
        "codes_produced_by_hand_vectors": sorted(produced),
        "refusal_codes_expected": sorted(refusals),
        "vectors": vectors,
    }


def serialise(obj):
    return (json.dumps(obj, ensure_ascii=True, indent=1, sort_keys=False) + "\n").encode("ascii")


if __name__ == "__main__":
    data = serialise(build())
    if "--check" in sys.argv:
        same = OUT.is_file() and OUT.read_bytes() == data
        print("vectors re-derive" if same else "VECTORS DIFFER from the committed file")
        sys.exit(0 if same else 1)
    if OUT.exists() and "--write-new" not in sys.argv:
        raise SystemExit("refusing to overwrite %s (a committed vector file is history); "
                         "use --check" % OUT)
    target = OUT if not OUT.exists() else OUT.with_suffix(".new.json")
    target.write_bytes(data)
    print("wrote %s (%d bytes)" % (target, len(data)))
