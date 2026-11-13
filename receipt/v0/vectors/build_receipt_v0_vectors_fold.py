# -*- coding: utf-8 -*-
"""RECEIPT v0 (DRAFT, unsigned): builds receipt/v0/vectors/receipt_v0_vectors_fold.json.

The fold round (2026-11-13). A lab agent built a Go checker from RECEIPT_v0.md, the schema and the
vector file (receipt/v0/cleanroom-go/), and a second lab agent measured it against the lab's two
checkers. Its gap list named the places where the text left an implementer guessing. This file
holds one or more vectors for each gap item a vector can state; the text now states each rule
(RECEIPT_v0.md, the sections each vector's `rules` name). The committed vector file
receipt_v0_vectors.json is history and is not rebuilt: these vectors live beside it, in the same
format (13.15), with one more field, `gap`, naming the item of the measurer's list.

Every expected answer below is written by hand from the text. The builder refuses to write a vector
whose check_v0 answer differs; for a check that emits a core it then records check_v0's core digest,
as the main builder does. So the detail fields not stated by hand are check_v0's own, cross-checked
by check_v0.js and the clean-room Go build in tests/test_receipt_v0_fold.py.

Lab-written. Vectors written by the lab for its own checkers are agreement inside one lab.

  python build_receipt_v0_vectors_fold.py           writes the file when it does not exist
  python build_receipt_v0_vectors_fold.py --check   rebuilds in memory and compares
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "receipt_v0_vectors_fold.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# the main builder's helpers (S, entry, manifest, vector_id, rules_of, b64) and its check_v0
B = _load("receipt_v0_build_main", HERE / "build_receipt_v0_vectors.py")
C = B.C
S, entry, manifest, b64, vector_id = B.S, B.entry, B.manifest, B.b64, B.vector_id
U, M, F, HD = "UNRESOLVED", "MALFORMED", "FAILED", "HELD"

DEEP = 100000   # nesting depth of the deep vectors: past Python's and Node's default recursion
# limits; the 3,000,000 at which the clean-room Go build dies is built in memory by the test
# instead (6 MB of brackets per input would make this file ten times its size)

JCS, CHECK, RECHECK = [], [], []


def J(gap, label, value, expect, rules=("3.2", "3.3", "6.13")):
    JCS.append({"gap": gap, "label": label, "value": value, "expect": expect, "rules": list(rules)})


def H(gap, label, doc, rows, manifest=None, name="d.md", commit=None, details=None, rules=None):
    CHECK.append({"gap": gap, "label": label, "doc": doc, "rows": rows, "manifest": manifest,
                  "name": name, "commit": commit, "details": details or {}, "rules": rules})


def R(gap, label, doc, refusal, manifest=None, name="d.md", commit=None, rules=None):
    CHECK.append({"gap": gap, "label": label, "doc": doc, "refusal": refusal, "manifest": manifest,
                  "name": name, "commit": commit, "rules": rules or ["11.1"]})


# ---------------------------------------------------------------------------------------------
# 3.3: which code a value with several faults gets (G2), a key holding a surrogate (G3)
# ---------------------------------------------------------------------------------------------
G2 = "G2 (3.3, precedence)"
J(G2, "values in the code-point order of their keys: '' before '9'",
  b'{"9":0e0,"":"\\udfff"}', {"refusal": "lone_surrogate"})
J(G2, "values in key order, depth first: the value of 'a' is walked whole before 'b'",
  b'{"b":[1.5],"a":{"x":"\\ud800"}}', {"refusal": "lone_surrogate"})
J(G2, "values in key order: 'a' before 'b' whatever the written order",
  b'{"b":9007199254740992,"a":"\\ud800"}', {"refusal": "lone_surrogate"})
J(G2, "every key of an object before any of its values",
  b'{"a":1.5,"\\u00e9":1}', {"refusal": "non_ascii_key"})
J(G2, "the keys of an inner object are checked when its value is reached, not before",
  b'{"a":1.5,"b":{"\\u00e9":1}}', {"refusal": "float"})
J(G2, "array elements in order", b'[2.5,"\\ud800"]', {"refusal": "float"})
J(G2, "depth is judged on entering a container, before anything inside it",
  b"[" * 33 + b"1.5" + b"]" * 33, {"refusal": "depth"})
J(G2, "depth before the keys of the 33rd container",
  b'{"a":' * 32 + b'{"\\u00e9":1}' + b"}" * 32, {"refusal": "depth"})
J(G2, "an element before a deeper sibling: the first fault met is the code",
  b"[1.5," + b"[" * 32 + b"]" * 32 + b"]", {"refusal": "float"})
G3 = "G3 (3.3, a key with an unpaired surrogate)"
J(G3, "a key holding an unpaired surrogate is non_ascii_key", b'{"\\udfff":1}',
  {"refusal": "non_ascii_key"})
J(G3, "a surrogate key and a surrogate value: the key is checked first", b'{"\\ud800":"\\ud800"}',
  {"refusal": "non_ascii_key"})
J("G1 (6.13, nesting depth)", "JSON nested %d deep parses; the canonical form refuses it" % DEEP,
  b"[" * DEEP + b"]" * DEEP, {"refusal": "depth"})
J("G30 (6.13, keys compared after unescaping)",
  "'a' and '\\u0061' are one key written twice: last wins under 3.2",
  b'{"a":1,"\\u0061":2}', {"canonical": '{"a":2}'})

# ---------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------
G1 = "G1 (6.13, nesting depth)"
H(G1, "receipt bytes nested %d objects deep are JSON: leaf_not_scalar" % DEEP,
  S("r1", "numeric", "n is 7"), [(M, "leaf_not_scalar")],
  manifest=B.M1(b'{"a":' * DEEP + b"1" + b"}" * DEEP),
  details={0: {"leaf_type": "object"}})
H(G1, "receipt bytes nested %d arrays deep, read through a line fragment" % DEEP,
  S("r1#L1", "numeric", "n is 7"), [(M, "leaf_not_scalar")],
  manifest=B.M1(b"[" * DEEP + b"]" * DEEP + b"\n"), details={0: {"leaf_type": "array"}})
_deep_top = json.loads(manifest({"r1": entry(b"7")}, with_digest=True))
_mdeep = json.dumps(_deep_top, sort_keys=True).encode("utf-8")[:-1] + b', "x_ignored": ' \
    + b"[" * DEEP + b"]" * DEEP + b"}"
H(G1, "a manifest with an ignored top-level key nested %d deep is read and intact" % DEEP,
  S("r1", "numeric", "n is 7"), [(HD, None)], manifest=_mdeep)

G8 = "G8 (7.5, narrowing at k = 0)"
H(G8, "a receipt of twenty LF bytes cited as r1#L1 does not narrow: short_needle",
  S("r1#L1", "quote", "`x`"), [(M, "short_needle")], manifest=B.M1(b"\n" * 20),
  details={0: {"needle_bytes": 1, "minimum_bytes": 16}})
H(G8, "a receipt of ten LF bytes is under 16 bytes, so r1#L1 narrows: the empty slice",
  S("r1#L1", "quote", "`x`"), [(F, "needle_missing")], manifest=B.M1(b"\n" * 10),
  details={0: {"needle_bytes": 1, "haystack_bytes": 0, "occurrences": 0}})

G10 = "G10 (7.4, ASCII whitespace)"
for _cp in (0x00A0, 0x3000, 0x2028, 0xFEFF, 0x0085, 0x001C, 0x001F):
    H(G10, "a needle of U+%04X only is not whitespace here: not needle_empty" % _cp,
      S("r1", "quote", "`%s`" % chr(_cp)), [(U, "manifest_absent")])
H(G10, "inner text of U+3000 only is not empty_span: number_count",
  S("r1", "numeric", "　"), [(M, "number_count")])
H(G10, "inner text of U+001C only is not empty_span: needle_count",
  S("r1", "quote", "\x1c"), [(M, "needle_count")])

G14 = "G14 (6.5, an entry without sha256)"
for _lab, _drop, _val in (("missing", True, None), ("null", False, None), ("a number", False, 5),
                          ("an array", False, [B.sha(b"7")])):
    _e = entry(b"7", drop=("sha256",) if _drop else ())
    if not _drop:
        _e["sha256"] = _val
    H(G14, "an entry whose sha256 is %s: manifest_integrity" % _lab, S("r1", "numeric", "n is 7"),
      [(U, "manifest_integrity")], manifest=manifest({"r1": _e}, with_digest=False))

G21 = "G21 (7.3, printed leading zeros)"
H(G21, "'00.5' is the number 0.5", S("r1", "numeric", "is 00.5"), [(HD, None)],
  manifest=B.M1(b"0.5"), details={0: {"printed_token": "00.5", "printed": "0.5"}})
H(G21, "'-007.50' is the number -7.50", S("r1", "numeric", "is -007.50"), [(HD, None)],
  manifest=B.M1(b"-7.5"), details={0: {"printed_token": "-007.50", "printed": "-7.50"}})
H(G21, "'00,123' is grouped with a leading group of two digits: 123",
  S("r1", "numeric", "is 00,123"), [(HD, None)], manifest=B.M1(b"123"),
  details={0: {"printed_token": "00,123", "printed": "123"}})

G27 = "G27 (6.12, array tokens)"
for _tok in ("00", "01", "+1", " 1", "1e0", "1 ", "0x1"):
    H(G27, "array token %r is pointer_unresolvable" % _tok,
      S("r1#/" + _tok, "numeric", "n is 8"), [(M, "pointer_unresolvable")],
      manifest=B.M1(b"[7, 8]"))
H(G27, "array token '1' reaches the second element", S("r1#/1", "numeric", "n is 8"),
  [(HD, None)], manifest=B.M1(b"[7, 8]"))

G23 = "G23 (11.1, name)"
R(G23, "a name that is a number is name_form", S("r1", "numeric", "n is 7"), "name_form", name=5)
R(G23, "a name that is an array is name_form", S("r1", "numeric", "n is 7"), "name_form",
  name=["d.md"])
R(G23, "a name that is an object is name_form, and comes before commit_form",
  S("r1", "numeric", "n is 7"), "name_form", name={}, commit="X")

G30 = "G30 (6.13, keys compared after unescaping)"
H(G30, "pointer_ambiguous: 'a' and '\\u0061' are one key written twice",
  S("r1#/a", "numeric", "n is 7"), [(M, "pointer_ambiguous")],
  manifest=B.M1(b'{"a": 7, "\\u0061": 8}'))
H(G30, "an escaped key is its unescaped self: '\\u0062' is reached as /b",
  S("r1#/b", "numeric", "n is 8"), [(HD, None)], manifest=B.M1(b'{"a": 7, "\\u0062": 8}'))
_m = manifest({"r1": entry(b"7")}, with_digest=False)
R(G30, "manifest_form: a receipts id written plainly and with an escape",
  S("r1", "numeric", "n is 7"), "manifest_form",
  manifest=_m.replace(b'"receipts": {', b'"receipts": {"\\u0072\\u0031": {}, ', 1),
  rules=["5.5", "6.13", "11.1"])
R(G30, "manifest_form: spec written plainly and with an escape",
  S("r1", "numeric", "n is 7"), "manifest_form",
  manifest=_m.replace(b'"spec": ', b'"\\u0073pec": "sworn/manifest/0.2", "spec": ', 1),
  rules=["5.5", "6.13", "11.1"])


# ---------------------------------------------------------------------------------------------
# receipt checks, 11.3
# ---------------------------------------------------------------------------------------------

def recheck_cases():
    doc = (S("r1", "numeric", "x" * 301) + " and " + S("r2", "quote", "`abc`")).encode("utf-8")
    man = manifest({"r2": entry(b"abcdefghijklmnopqrstuvwxyz")})
    core = C.check(doc, "d.md", man, None)
    assert [r["reason"] for r in core["spans"]] == ["length_cap", "short_needle"], core["spans"]
    good = C.jcs(C.receipt(core)).encode("utf-8")
    assert b'"cap":300' in good and b'"minimum_bytes":16' in good

    def remint(c, **top):
        r = {"core": c, "digest": C.digest(c), "attestations": []}
        r.update(top)
        return C.jcs(r).encode("utf-8")

    legacy = json.loads(good)["core"]
    legacy["profile"] = "legacy-projection/v0-draft"
    G19 = "G19 (11.3, numbers in the schema step)"
    G17 = "G17 (11.3, a receipt check without a document)"
    G16 = "G16 (11.3, attestations in the shape step)"
    out = [
        (G19, "REDERIVES: a core with a length_cap row and a short_needle row", good, doc, man,
         {"answer": "REDERIVES"}),
        (G19, "const 300 written 300.0 is receipt_shape", good.replace(b'"cap":300', b'"cap":300.0', 1),
         doc, man, {"refusal": "receipt_shape"}),
        (G19, "const 300 written 3e2 is receipt_shape", good.replace(b'"cap":300', b'"cap":3e2', 1),
         doc, man, {"refusal": "receipt_shape"}),
        (G19, "const 300 written 3E+2 is receipt_shape",
         good.replace(b'"cap":300', b'"cap":3E+2', 1), doc, man, {"refusal": "receipt_shape"}),
        (G19, "const 16 written 16.0 is receipt_shape",
         good.replace(b'"minimum_bytes":16', b'"minimum_bytes":16.0', 1), doc, man,
         {"refusal": "receipt_shape"}),
        (G19, "-0 is the integer 0 and meets minimum 0: an offset of 0 written -0",
         good.replace(b'"at":0,', b'"at":-0,', 1), doc, man, {"answer": "REDERIVES"}),
        (G17, "no document: nothing_to_verify, after every refusal that reads only the receipt",
         good, None, man, {"refusal": "nothing_to_verify"}),
        (G17, "no document and a legacy profile: profile_not_checkable", remint(legacy), None, man,
         {"refusal": "profile_not_checkable"}),
        (G17, "no document and a wrong digest: receipt_digest",
         C.jcs({"core": json.loads(good)["core"], "digest": "0" * 64,
                "attestations": []}).encode("utf-8"),
         None, man, {"refusal": "receipt_digest"}),
        (G16, "attestations an object is receipt_shape", remint(core, attestations={}), doc, man,
         {"refusal": "receipt_shape"}),
        (G16, "attestations a string is receipt_shape", remint(core, attestations="x"), doc, man,
         {"refusal": "receipt_shape"}),
        (G16, "attestations [1] is attestations_not_empty, not receipt_shape",
         remint(core, attestations=[1]), doc, man, {"refusal": "attestations_not_empty"}),
        ("G30 (6.13, keys compared after unescaping)",
         "document_verdict written plainly and with an escape is receipt_shape",
         good.replace(b'"document_verdict":', b'"\\u0064ocument_verdict":"SWORN-HELD","document_verdict":', 1),
         doc, man, {"refusal": "receipt_shape"}),
    ]
    return out


# ---------------------------------------------------------------------------------------------

def build_vectors():
    out = []
    for j in JCS:
        inputs = {"value_b64": b64(j["value"])}
        try:
            got = {"canonical": C.jcs(C.parse_json(j["value"]))}
        except C.Refusal as e:
            got = {"refusal": e.code}
        if got != j["expect"]:
            raise SystemExit("jcs vector %r: the text says %r, check_v0 gives %r"
                             % (j["label"], j["expect"], got))
        out.append({"id": vector_id("jcs", inputs), "source": "jcs", "op": "jcs", "gap": j["gap"],
                    "label": j["label"], "inputs": inputs, "expect": j["expect"],
                    "rules": j["rules"], "note": None})
    for h in CHECK:
        doc = h["doc"].encode("utf-8") if isinstance(h["doc"], str) else h["doc"]
        inputs = {"document_b64": b64(doc), "name": h["name"], "manifest_b64": b64(h["manifest"]),
                  "commit": h["commit"]}
        vec = {"id": vector_id("check", inputs), "source": "hand", "op": "check", "gap": h["gap"],
               "label": h["label"], "inputs": inputs, "note": None}
        try:
            core, got_refusal = C.check(doc, h["name"], h["manifest"], h["commit"]), None
        except C.Refusal as e:
            core, got_refusal = None, e.code
        if "refusal" in h:
            if got_refusal != h["refusal"]:
                raise SystemExit("vector %r: the text says refusal %s, check_v0 gives %s"
                                 % (h["label"], h["refusal"], got_refusal or "a core"))
            vec["expect"] = {"refusal": h["refusal"]}
            vec["rules"] = h["rules"]
        else:
            if core is None:
                raise SystemExit("vector %r: check_v0 refused %s" % (h["label"], got_refusal))
            rows = [[r["verdict"], r["reason"]] for r in core["spans"]]
            want = [list(x) for x in h["rows"]]
            dv = B._dv(want)
            if rows != want or core["document_verdict"] != dv:
                raise SystemExit("vector %r: the text says %r %s, check_v0 gives %r %s"
                                 % (h["label"], want, dv, rows, core["document_verdict"]))
            for i, sub in h["details"].items():
                d = core["spans"][i]["detail"]
                if not all(d.get(k, KeyError) == v for k, v in sub.items()):
                    raise SystemExit("vector %r row %d: detail %r, the text says %r"
                                     % (h["label"], i, d, sub))
            vec["expect"] = {"core_digest": C.digest(core)}
            vec["expect_rows"] = want
            vec["expect_document_verdict"] = dv
            vec["rules"] = h["rules"] or B.rules_of(core, h["manifest"] is not None)
        out.append(vec)
    for gap, label, rec, doc, man, expect in recheck_cases():
        inputs = {"receipt_b64": b64(rec), "document_b64": b64(doc), "manifest_b64": b64(man)}
        try:
            got = {"answer": C.recheck(rec, doc, man)}
        except C.Refusal as e:
            got = {"refusal": e.code}
        if got != expect:
            raise SystemExit("recheck vector %r: the text says %r, check_v0 gives %r"
                             % (label, expect, got))
        out.append({"id": vector_id("recheck", inputs), "source": "recheck", "op": "recheck",
                    "gap": gap, "label": label, "inputs": inputs, "expect": expect,
                    "rules": ["11.3"], "note": None})
    ids = [v["id"] for v in out]
    if len(set(ids)) != len(ids):
        raise SystemExit("two fold vectors have the same inputs")
    main = json.loads((HERE / "receipt_v0_vectors.json").read_text(encoding="utf-8"))
    clash = set(ids) & {v["id"] for v in main["vectors"]}
    if clash:
        raise SystemExit("a fold vector repeats a vector of the main file: %s" % sorted(clash)[:3])
    return out


def build():
    vectors = build_vectors()
    counts = {}
    for v in vectors:
        counts[v["source"]] = counts.get(v["source"], 0) + 1
    counts["total"] = len(vectors)
    gaps = sorted({v["gap"] for v in vectors}, key=lambda g: int(g.split()[0][1:]))
    return {
        "schema": "receipt/v0-draft/vectors",
        "status": "DRAFT, unsigned. Lab-written vectors for a lab draft; not a standard.",
        "text": "receipt/v0/RECEIPT_v0.md",
        "supplements": "receipt/v0/vectors/receipt_v0_vectors.json",
        "round": ("fold, 2026-11-13: one or more vectors per item of the clean-room measurer's gap "
                  "list that a vector can state; field `gap` names the item"),
        "id_rule": ("sha256 of the UTF-8 canonical text (section 3) of {op, inputs}; an input string "
                    "with no canonical form is replaced inside the id object by "
                    "{utf16be_hex: its UTF-16BE code units in lowercase hex}"),
        "expect_rule": {
            "hand": ("rows and refusals as the text gives them; core_digest is check_v0's, recorded "
                     "only after its rows matched expect_rows"),
            "recheck": "11.3's answer or refusal",
            "jcs": "section 3's canonical text or refusal code, the value read by 6.13",
        },
        "counts": counts,
        "gaps": gaps,
        "vectors": vectors,
    }


if __name__ == "__main__":
    data = B.serialise(build())
    if "--check" in sys.argv:
        same = OUT.is_file() and OUT.read_bytes() == data
        print("fold vectors re-derive" if same else "FOLD VECTORS DIFFER from the file")
        sys.exit(0 if same else 1)
    if OUT.exists():
        raise SystemExit("refusing to overwrite %s (a committed vector file is history); "
                         "use --check" % OUT)
    OUT.write_bytes(data)
    print("wrote %s (%d bytes, %d vectors)" % (OUT, len(data), len(json.loads(data)["vectors"])))
