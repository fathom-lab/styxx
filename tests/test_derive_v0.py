# -*- coding: utf-8 -*-
"""DERIVE v0 (DRAFT): the vectors through derive.py and derive.js, their agreement, purity, and time.

- every vector's hand-written expectation (written from DERIVE_v0.md by vectors/make_vectors.py,
  which never imports either implementation) is checked against BOTH implementations;
- the two outputs must be byte-identical canonical JSON on every vector;
- node is required: if it is missing the test FAILS, it does not skip;
- the two source files are scanned for forbidden names;
- hostile inputs within the stated limits must finish within a stated wall time.

Agreement here is between two programs one agent wrote with the text open. It shows the text was
computable the same way twice; it is not an independent implementation and not a measure of how
much of the text the vectors reach (no mutation run is part of this file).
"""
import ast
import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDIR = os.path.join(ROOT, "receipt", "derive")
PY = os.path.join(DDIR, "derive.py")
JS = os.path.join(DDIR, "derive.js")
VEC = os.path.join(DDIR, "vectors", "derive_v0_vectors.json")
GEN = os.path.join(DDIR, "vectors", "make_vectors.py")

# the closed lists of DERIVE_v0.md section 11, copied from the text by hand
REASONS = {
    "MALFORMED": ["empty_span", "length_cap", "derive_span_cap", "receipt_form", "directional_override", "number_count",
                  "number_grammar", "places_cap", "printed_too_long", "derive_block_malformed", "derivation_missing",
                  "derivation_form", "op_unknown", "unit_mismatch", "within_form", "kind_of_source_unknown",
                  "receipt_author_minted", "rows_too_large", "derive_budget", "receipt_not_json", "receipt_not_jsonl",
                  "receipt_too_many_values", "pointer_unresolvable", "pointer_ambiguous", "rows_not_container",
                  "rows_too_many", "rows_ambiguous", "row_key_unavailable", "field_ambiguous", "number_out_of_range",
                  "where_type_mismatch", "field_not_numeric", "field_never_present", "division_by_zero", "no_rows_used"],
    "FAILED": ["value_mismatch", "rounds_to_zero", "rounds_to_whole", "not_within"],
    "UNRESOLVED": ["manifest_absent", "manifest_integrity", "rung_unknown", "manifest_id_missing", "manifest_bytes_absent"],
}
REFUSALS = ["claim_form", "claim_too_large"]
OPS = ["count", "share", "percent", "sum", "mean"]


def _load_py():
    spec = importlib.util.spec_from_file_location("derive_v0_under_test", PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CACHE = {}


def vectors():
    if "vec" not in _CACHE:
        with open(VEC, "r", encoding="utf-8") as fh:
            _CACHE["vec"] = json.load(fh)["vectors"]
    return _CACHE["vec"]


def py_lines():
    if "py" not in _CACHE:
        with open(VEC, "rb") as fh:
            data = fh.read()
        t0 = time.perf_counter()
        _CACHE["py"] = _load_py().batch_lines(data)
        _CACHE["py_seconds"] = time.perf_counter() - t0
    return _CACHE["py"]


def js_lines():
    if "js" not in _CACHE:
        node = shutil.which("node")
        assert node is not None, "node is required for the derive.js half of this test; it is missing (FAIL, not skip)"
        t0 = time.perf_counter()
        r = subprocess.run([node, JS, "--batch", VEC], capture_output=True, timeout=900)
        _CACHE["js_seconds"] = time.perf_counter() - t0
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-2000:]
        text = r.stdout.decode("utf-8")
        assert text.endswith("\n")
        _CACHE["js"] = text[:-1].split("\n")
    return _CACHE["js"]


def strict_eq(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_eq(x, y) for x, y in zip(a, b))
    return a == b


def match(exp, got, path):
    errs = []
    if isinstance(exp, dict):
        if set(exp) == {"$eq"}:
            if not strict_eq(exp["$eq"], got):
                errs.append("%s: expected exactly %r, got %r" % (path, exp["$eq"], got))
            return errs
        if not isinstance(got, dict):
            return ["%s: expected an object, got %r" % (path, got)]
        for k, v in exp.items():
            if k == "$lacks":
                for kk in v:
                    if kk in got:
                        errs.append("%s: %s should be absent" % (path, kk))
            elif k == "surface_has":
                for frag in v:
                    if frag not in got.get("surface", ""):
                        errs.append("%s: surface lacks %r" % (path, frag))
            elif k not in got:
                errs.append("%s.%s: missing" % (path, k))
            else:
                errs.extend(match(v, got[k], path + "." + k))
        return errs
    if isinstance(exp, list):
        if not isinstance(got, list) or len(got) != len(exp):
            return ["%s: expected a list of %d, got %r" % (path, len(exp), got)]
        for i, (x, y) in enumerate(zip(exp, got)):
            errs.extend(match(x, y, "%s[%d]" % (path, i)))
        return errs
    if not strict_eq(exp, got):
        errs.append("%s: expected %r, got %r" % (path, exp, got))
    return errs


def _check_expectations(lines, who):
    vs = vectors()
    assert len(lines) == len(vs), (who, len(lines), len(vs))
    failures = []
    for v, line in zip(vs, lines):
        got = json.loads(line)
        assert got["id"] == v["id"]
        errs = match(v["expect"], got["out"], v["id"])
        if errs:
            failures.append("%s [%s] %s" % (v["id"], v["rule"], "; ".join(errs)))
    assert not failures, "%s: %d of %d vectors miss their expectation:\n%s" % (who, len(failures), len(vs), "\n".join(failures))


def test_vector_file_shape_and_reach():
    vs = vectors()
    assert len(vs) >= 200
    seen_reasons, seen_refusals, seen_ops = set(), set(), set()
    for v in vs:
        e = v["expect"]
        if set(e) == {"$eq"} and "refusal" in e["$eq"]:
            seen_refusals.add(e["$eq"]["refusal"])
        row = e.get("row", {})
        if "verdict" in row and row.get("reason"):
            seen_reasons.add((row["verdict"], row["reason"]))
        claim = (base64.b64decode(v["claim_b64"]) if "claim_b64" in v else b"").replace(b"\\", b"")
        for op in OPS:
            if ('"op": "%s"' % op).encode() in claim or ('"op":"%s"' % op).encode() in claim:
                seen_ops.add(op)
    missing = [(vd, r) for vd, rs in REASONS.items() for r in rs if (vd, r) not in seen_reasons]
    assert not missing, "reason codes no vector expects: %r" % missing
    assert set(REFUSALS) <= seen_refusals, seen_refusals
    assert set(OPS) <= seen_ops, seen_ops


def test_generator_never_imports_an_implementation():
    with open(GEN, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for n in names:
                assert "derive" not in n, n
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "derive.js" not in node.value or "never" in node.value


def test_python_meets_every_expectation():
    _check_expectations(py_lines(), "derive.py")


def test_js_meets_every_expectation():
    _check_expectations(js_lines(), "derive.js")


def test_python_and_js_are_byte_identical_on_every_vector():
    a, b = py_lines(), js_lines()
    assert len(a) == len(b) == len(vectors())
    differ = [vectors()[i]["id"] for i in range(len(a)) if a[i] != b[i]]
    assert not differ, "outputs differ on %d vectors: %s" % (len(differ), differ[:20])


FORBIDDEN_PY_NAMES = {"eval", "exec", "compile", "__import__", "float", "input", "globals", "locals", "getattr",
                      "setattr", "delattr", "vars", "breakpoint", "memoryview"}
ALLOWED_PY_IMPORTS = {"base64", "hashlib", "sys", "unicodedata"}


def test_purity_python():
    with open(PY, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    opens = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name in ALLOWED_PY_IMPORTS, a.name
        if isinstance(node, ast.ImportFrom):
            raise AssertionError("from-import in derive.py: %s" % node.module)
        if isinstance(node, ast.Name):
            assert node.id not in FORBIDDEN_PY_NAMES, node.id
            if node.id == "open":
                opens.append(node)
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("system", "popen", "urlopen", "time", "now", "environ", "getenv"), node.attr
        if isinstance(node, ast.Constant):
            assert not isinstance(node.value, float), "float literal at line %d" % node.lineno
        if isinstance(node, ast.BinOp):
            assert not isinstance(node.op, ast.Div), "true division at line %d" % node.lineno
        if isinstance(node, ast.AugAssign):
            assert not isinstance(node.op, ast.Div), "true division at line %d" % node.lineno
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(main) == 1
    inside = {id(n) for n in ast.walk(main[0])}
    assert opens and all(id(n) in inside for n in opens), "open() outside main()"
    # the import allowlist above already excludes re, os, time, socket, subprocess, random, datetime


FORBIDDEN_JS = ["eval(", "Function(", "new Function", "XMLHttpRequest", "fetch(", "WebSocket", "Date", "Math.random",
                "performance", "setTimeout", "setInterval", "import(", "JSON.parse", "JSON.stringify", "parseFloat",
                "toFixed", "localStorage", "sessionStorage", "indexedDB", "child_process", "Math."]


def test_purity_js():
    with open(JS, "r", encoding="utf-8") as fh:
        src = fh.read()
    begin, end = src.index("// BEGIN CLI"), src.index("// END CLI")
    body = src[:begin] + src[end:]
    cli = src[begin:end]
    for word in FORBIDDEN_JS:
        assert word not in body, word
    import re
    for pat in (r"\brequire\s*\(", r"\bprocess\b", r"\bfs\.", r"\b__dirname\b", r"\bBuffer\b", r"\bglobalThis\.(?!DeriveV0\b)"):
        assert not re.search(pat, body), pat
    assert "child_process" not in cli and "fetch" not in cli and "http" not in cli
    # no decimal-point or exponent numeric literals outside comments and strings
    code = re.sub(r"//[^\n]*", "", body)
    code = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', code)
    assert not re.search(r"(?<![\w.])\d+\.\d+(?![\w])", code), "decimal literal in derive.js"
    assert not re.search(r"(?<![\w.])\d+[eE][+-]?\d+(?![\w])", code), "exponent literal in derive.js"
    r = subprocess.run([shutil.which("node") or "node", "--check", JS], capture_output=True)
    assert r.returncode == 0, r.stderr


def _claim(deriv, inner):
    c = {"blocks": [{"line": 1, "content": json.dumps({"n": deriv})}], "span": {"receipt": "r1#n", "inner": inner},
         "resolution": {"verdict": "RESOLVED", "entry": {"kind_of_source": "tool_stdout", "complete": True,
                                                         "harness": "t", "rung": "L2"}}}
    return json.dumps(c).encode("utf-8")


def _hostile_cases():
    root = {"format": "json", "at": ""}
    lits = ["x%03d" % k + "a" * 0 for k in range(16)]
    heavy_rows = b"[" + b",".join(b'{"a":"x%03d","b":%d,"c":"y","d":%d}' % (k % 50, k, k % 7) for k in range(100000)) + b"]"
    heavy = {"rows": root, "where": [{"field": "a", "in": lits}, {"field": "b", "ge": 0}, {"field": "c", "eq": "y"},
                                     {"field": "d", "le": 5}, {"field": "a", "present": True}],
             "match": [{"field": "d", "in": list(range(16))}, {"field": "b", "le": 99999}, {"field": "c", "in": ["y", "z"]}],
             "op": "share"}
    return [
        ("16 MiB of [", _claim({"rows": root, "where": [], "op": "count"}, "0"), b"[" * 16777216),
        ("1,000,001 values", _claim({"rows": root, "where": [], "op": "count"}, "0"), b"[" + b"0," * 1000000 + b"0]"),
        ("one large string of 16 MiB", _claim({"rows": root, "where": [], "op": "count"}, "1"), b'["' + b"a" * 16777212 + b'"]'),
        ("100,000 rows and 8 tests", _claim(heavy, "0.5"), heavy_rows),
        ("4,000-digit exponent and coefficient", _claim({"rows": root, "where": [], "of": {"row": "value"}, "op": "sum"}, "0"),
         b"[" + b"1" * 4000 + b"e" + b"9" * 4000 + b"]"),
        ("escapes all the way", _claim({"rows": root, "where": [], "op": "count"}, "1"), b'["' + b"\\n" * 8000000 + b'"]'),
    ]


LIMIT_SECONDS = 120


def test_bounded_time_on_hostile_inputs(tmp_path):
    mod = _load_py()
    node = shutil.which("node")
    assert node is not None, "node is required (FAIL, not skip)"
    report = []
    for name, claim, receipt in _hostile_cases():
        t0 = time.perf_counter()
        py_out = mod.result_text(claim, receipt)
        tp = time.perf_counter() - t0
        cp, rp = tmp_path / "c.json", tmp_path / "r.bin"
        cp.write_bytes(claim)
        rp.write_bytes(receipt)
        t0 = time.perf_counter()
        r = subprocess.run([node, JS, str(cp), str(rp)], capture_output=True, timeout=LIMIT_SECONDS * 2)
        tj = time.perf_counter() - t0
        js_out = r.stdout.decode("utf-8").rstrip("\n")
        verdict = json.loads(py_out)["row"]
        report.append("%-36s py %6.2fs  js %6.2fs  %s %s" % (name, tp, tj, verdict["verdict"], verdict["reason"]))
        assert py_out == js_out, name
        assert tp < LIMIT_SECONDS and tj < LIMIT_SECONDS, (name, tp, tj)
    sys.stdout.write("\n" + "\n".join(report) + "\n")


def test_report_timings():
    py_lines()
    js_lines()
    sys.stdout.write("\nvectors %d: derive.py %.2fs in process, derive.js %.2fs including node start\n"
                     % (len(vectors()), _CACHE["py_seconds"], _CACHE["js_seconds"]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "-p", "no:cacheprovider"]))
