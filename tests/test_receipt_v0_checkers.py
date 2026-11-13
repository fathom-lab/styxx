# -*- coding: utf-8 -*-
"""RECEIPT v0 (DRAFT, unsigned): the lab-written checker receipt/v0/check_v0.py against the vector
file receipt/v0/vectors/receipt_v0_vectors.json.

  - every vector through check_v0 (check, recheck and the canonical form);
  - hand-written vectors also give the rows the text gives, and their cores and receipts pass the
    schema;
  - a purity test over check_v0's source: standard library imports only, no clock, network,
    process, file or float in the checking code, no regular expression at all;
  - bounded time on large and deeply nested inputs;
  - the JavaScript checker receipt/v0/check_v0.js: every vector through it with node (the test
    FAILS, it does not skip, when node is missing), byte-identical canonical receipts between the
    two checkers on every vector that emits a core, the same answers on the large inputs, its
    purity by the source, its command line, and a run with no Node globals at all.
  - the fixer's round: both checkers carry 7.2's Unicode 15.0.0 tables as data (checked against
    unicodedata at 15.0.0 and against the digests the text states), both sources are pure ASCII,
    the JavaScript sha256 at every padding boundary, and each mutant that nothing caught before the
    round now fails a vector.

check_v0 and check_v0.js are lab-written and not independent of the lab's checker (its docstring says so). These
tests show that it answers its own vectors; they do not show that any other party agrees.
"""
import ast
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V0 = Path(os.environ.get("RECEIPT_V0_DIR", str(ROOT / "receipt" / "v0")))
VECTORS = V0 / "vectors" / "receipt_v0_vectors.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("receipt_check_v0_under_test", V0 / "check_v0.py")
DATA = json.loads(VECTORS.read_text(encoding="utf-8"))
ALL = DATA["vectors"]


def _b(x):
    return None if x is None else base64.b64decode(x)


def _run(v):
    i = v["inputs"]
    try:
        if v["op"] == "check":
            core = C.check(_b(i["document_b64"]), i["name"], _b(i["manifest_b64"]), i["commit"])
            return {"core_digest": C.digest(core)}, core
        if v["op"] == "recheck":
            return {"answer": C.recheck(_b(i["receipt_b64"]), _b(i["document_b64"]),
                                        _b(i["manifest_b64"]))}, None
        if v["op"] == "jcs":
            return {"canonical": C.jcs(C.parse_json(_b(i["value_b64"])))}, None
    except C.Refusal as e:
        return {"refusal": e.code}, None
    raise AssertionError("unknown op %r" % v["op"])


def _id_safe(x):
    if isinstance(x, str):
        try:
            C.jcs(x)
            return x
        except C.Refusal:
            return {"utf16be_hex": x.encode("utf-16-be", "surrogatepass").hex()}
    if isinstance(x, dict):
        return {k: _id_safe(y) for k, y in x.items()}
    return x


# ---------------------------------------------------------------------------------------------
# the vectors
# ---------------------------------------------------------------------------------------------

def test_the_vector_file_is_what_it_says():
    counts = DATA["counts"]
    by = {}
    for v in ALL:
        by[v["source"]] = by.get(v["source"], 0) + 1
    assert by == {k: counts[k] for k in ("conformance", "hand", "recheck", "jcs")}
    assert counts["total"] == len(ALL)
    assert counts["conformance"] == 1718
    ids = [v["id"] for v in ALL]
    assert len(set(ids)) == len(ids)
    for v in ALL:
        assert v["id"] == C.digest({"op": v["op"], "inputs": _id_safe(v["inputs"])}), v["id"]
        assert v["rules"], v["id"]
    raw = VECTORS.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize("source", ["conformance", "hand", "recheck", "jcs"])
def test_every_vector_through_check_v0(source):
    wrong = []
    n = 0
    for v in ALL:
        if v["source"] != source:
            continue
        n += 1
        got, core = _run(v)
        if got != v["expect"]:
            wrong.append((v["id"][:16], v.get("label") or v.get("conformance"), v["expect"], got))
            continue
        if "expect_rows" in v:
            rows = [[r["verdict"], r["reason"]] for r in core["spans"]]
            if rows != v["expect_rows"] or core["document_verdict"] != v["expect_document_verdict"]:
                wrong.append((v["id"][:16], v["label"], v["expect_rows"], rows))
    assert n > 0
    assert not wrong, "%d of %d vectors differ, first: %r" % (len(wrong), n, wrong[:3])


def test_the_closed_lists_are_produced_or_named_as_not_produced():
    """Every reason in section 10 that a v0 checker can give appears in a hand vector's rows; the
    refusal codes reached by some vector are 11's list less `type`, which JSON cannot reach."""
    assert set(DATA["codes_produced_by_hand_vectors"]) == set(C.REASONS)
    assert set(DATA["refusal_codes_expected"]) == set(C.REFUSALS) - {"type"}
    verdicts, dvs = set(), set()
    for v in ALL:
        for row in v.get("expect_rows", []):
            verdicts.add(row[0])
        if "expect_document_verdict" in v:
            dvs.add(v["expect_document_verdict"])
    assert verdicts == set(C.SPAN_VERDICTS) and dvs == set(C.DOCUMENT_VERDICTS)
    answers = {v["expect"].get("answer") for v in ALL if v["op"] == "recheck"} - {None}
    assert answers == {"REDERIVES", "DOES-NOT-REDERIVE"}


def test_the_lists_in_check_v0_are_the_text_lists():
    text = (V0 / "RECEIPT_v0.md").read_text(encoding="utf-8")
    section = text.split("## 10. The closed lists", 1)[1].split("## 11.", 1)[0]
    named = set()
    for line in section.splitlines():
        for part in line.split("`")[1::2]:
            named.add(part)
    ours = (set(C.REASONS) | set(C.REFUSALS) | set(C.LEGACY_REASONS) | set(C.DOCUMENT_REASONS)
            | set(C.SPAN_VERDICTS) | set(C.DOCUMENT_VERDICTS))
    assert ours == named, (sorted(ours - named), sorted(named - ours))


def test_hand_cores_and_receipts_pass_the_schema():
    t = _load("receipt_v0_step1_tests", ROOT / "tests" / "test_receipt_v0.py")
    n = 0
    for v in ALL:
        if v["source"] != "hand" or "core_digest" not in v["expect"]:
            continue
        i = v["inputs"]
        core = C.check(_b(i["document_b64"]), i["name"], _b(i["manifest_b64"]), i["commit"])
        t.assert_valid(core, "verdictCore")
        t.assert_valid(C.receipt(core, C.VERIFIER), "receipt")
        n += 1
    assert n > 200


def test_a_receipt_check_rederives_every_hand_core():
    n = 0
    for v in ALL:
        if v["source"] != "hand" or "core_digest" not in v["expect"]:
            continue
        i = v["inputs"]
        doc, man = _b(i["document_b64"]), _b(i["manifest_b64"])
        core = C.check(doc, i["name"], man, i["commit"])
        rec = C.jcs(C.receipt(core)).encode("utf-8")
        assert C.recheck(rec, doc, man) == "REDERIVES", v["label"]
        n += 1
    assert n > 200


# ---------------------------------------------------------------------------------------------
# the text's worked examples, stated once more as plain asserts
# ---------------------------------------------------------------------------------------------

def test_worked_examples():
    core = C.check("\u00e9<sworn r=\"r1\" k=\"numeric\">n 7</sworn>".encode("utf-8"), "d.md")
    row = core["spans"][0]
    assert (row["at"], row["start"], row["end"]) == (2, 2, 5)
    assert C.check(b"", "e")["document"]["canonical_sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    for text, code in (("9007199254740993", "int_range"), ("1.0", "float")):
        with pytest.raises(C.Refusal) as e:
            C.jcs(C.parse_json(text.encode()))
        assert e.value.code == code
    for enc, want in (("QQ==", b"A"), ("QR==", b"A"), ("", b""), ("QQ", None), ("QQ=", None),
                      ("Q Q==", None), ("QQ==QQ==", None), ("QQ==\n", None)):
        assert C._b64_strict(enc) == want, enc


# ---------------------------------------------------------------------------------------------
# purity, by the source
# ---------------------------------------------------------------------------------------------

_SOURCE = (V0 / "check_v0.py").read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)
_IO_ONLY = {"_main"}          # the command line: the one function that reads the files it is handed


def _pure_nodes():
    """Every node outside _main and outside the `if __name__ == "__main__"` block."""
    skip = set()
    for node in _TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name in _IO_ONLY:
            skip.add(id(node))
        if isinstance(node, ast.If) and "__main__" in ast.dump(node.test):
            skip.add(id(node))
    out = []
    for node in _TREE.body:
        if id(node) not in skip:
            out.extend(ast.walk(node))
    return out


def test_imports_are_standard_library_only():
    names = set()
    for node in ast.walk(_TREE):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names.add(node.module.split(".")[0])
    assert names <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names)
    assert not any(n.startswith("styxx") for n in names)
    pure_imports = set()
    for node in _pure_nodes():
        if isinstance(node, ast.Import):
            pure_imports |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            pure_imports.add(node.module)
    # no unicodedata: 7.2's classes are carried as data (the fixer's round), never the host's
    assert pure_imports == {"hashlib", "binascii"}, pure_imports


def test_the_checking_code_names_nothing_impure():
    banned = {"open", "exec", "eval", "compile", "__import__", "input", "print", "globals",
              "locals", "vars", "setattr", "delattr", "float", "complex", "time", "datetime",
              "os", "sys", "subprocess", "socket", "random", "secrets", "re", "pathlib",
              "urllib", "http", "requests", "importlib", "io", "shutil", "tempfile", "Decimal",
              "decimal", "fractions", "math", "argparse", "json"}
    hits = []
    for node in _pure_nodes():
        if isinstance(node, ast.Name) and node.id in banned:
            hits.append((node.lineno, node.id))
        if isinstance(node, ast.Attribute) and node.attr in ("open", "system", "popen", "urlopen",
                                                               "write", "time", "now"):
            hits.append((node.lineno, node.attr))
    assert not hits, hits


def test_no_float_and_no_true_division_anywhere():
    for node in ast.walk(_TREE):
        assert not (isinstance(node, ast.Constant) and isinstance(node.value, float)), node.lineno
        assert not isinstance(node, (ast.Div,)), getattr(node, "lineno", "?")


# ---------------------------------------------------------------------------------------------
# bounded time
# ---------------------------------------------------------------------------------------------

def _timed(fn, limit_s):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    assert dt < limit_s, "took %.1f s (limit %d s)" % (dt, limit_s)
    return out


def _man(data):
    m = {"spec": "sworn/manifest/0.2", "harness": "h", "turn": "t", "minted_at": "x",
         "authored_sha256": [], "rung": "L2",
         "receipts": {"r1": {"id": "r1", "sha256": hashlib.sha256(data).hexdigest(),
                             "kind_of_source": "tool_stdout", "captured_at": "x",
                             "complete": True,
                             "bytes": base64.b64encode(data).decode("ascii")}}}
    return json.dumps(m).encode("utf-8")


SPAN = b'<sworn r="r1" k="numeric">n is 7</sworn> '


_BIG = [
    ("20,000 spans", SPAN * 20000, _man(b"7"), 60,
     lambda c: c["counts"]["HELD"] == 20000),
    ("20,000 nested openers", b'<sworn r="r1" k="numeric">' * 20000 + b"7" + b"</sworn>" * 20000,
     None, 30, lambda c: c["counts"]["MALFORMED"] == 20000),
    ("a 4 MB line of distinct backtick runs",
     b" ".join(b"`" * k for k in range(1, 2000)) + b" " + SPAN, None, 30,
     lambda c: c["sworn_total"] == 1),
    ("200,000 candidates with no closing bracket, one per line", b"<sworn r\n" * 200000, None, 30,
     lambda c: c["counts"]["MALFORMED"] == 200000),
    ("200,000 comment openers", b"<!-- " * 200000 + SPAN, None, 30,
     lambda c: c["spans"][0]["reason"] == "hidden_commitment"),
    ("10,000 fenced regions", b"```\nx\n```\n" * 10000 + SPAN, None, 30,
     lambda c: c["sworn_total"] == 1),
    ("8 MB of narrative", b"plain text, no tags. " * 400000, None, 30,
     lambda c: c["document_verdict"] == "UNSWORN"),
    ("a receipt nested 200,000 deep", b'<sworn r="r1" k="numeric">7</sworn>',
     _man(b"[" * 200000 + b"]" * 200000), 30,
     lambda c: c["spans"][0]["reason"] == "leaf_not_scalar"),
    ("a 4 MB receipt searched for a quote", b'<sworn r="r1" k="quote">`needle that is long enough`</sworn>',
     _man(b"x" * 4000000 + b"needle that is long enough"), 30,
     lambda c: c["spans"][0]["verdict"] == "HELD"),
    ("a receipt number with 100,000 digits", b'<sworn r="r1" k="numeric">0.5</sworn>',
     _man(b"0." + b"5" * 100000), 30, lambda c: c["spans"][0]["verdict"] == "FAILED"),
]


@pytest.mark.parametrize("label", [b[0] for b in _BIG])
def test_bounded_time(label):
    _, doc, man, limit, check = [b for b in _BIG if b[0] == label][0]
    core = _timed(lambda: C.check(doc, "big.md", man), limit)
    assert check(core), label
    _timed(lambda: C.digest(core), limit)


def test_bounded_time_receipt_check():
    doc = SPAN * 5000
    man = _man(b"7")
    rec = C.jcs(C.receipt(C.check(doc, "d.md", man))).encode("utf-8")
    assert _timed(lambda: C.recheck(rec, doc, man), 60) == "REDERIVES"


def test_a_deeply_nested_receipt_object_is_refused_by_shape_not_by_a_crash():
    deep = b'{"core":' + b"[" * 100000 + b"]" * 100000 + b',"digest":"' + b"0" * 64 + b'","attestations":[]}'
    with pytest.raises(C.Refusal) as e:
        _timed(lambda: C.recheck(deep, b"", None), 30)
    assert e.value.code == "receipt_shape"


# ---------------------------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------------------------

def test_the_command_line(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_bytes(b'n is <sworn r="r1" k="numeric">7</sworn>')
    man = tmp_path / "m.json"
    man.write_bytes(_man(b"7"))
    exe = [sys.executable, str(V0 / "check_v0.py")]
    out = subprocess.run(exe + ["check", str(doc), "--manifest", str(man), "--name", "d.md"],
                         capture_output=True, check=True).stdout
    rec = json.loads(out)
    assert rec["core"]["document_verdict"] == "SWORN-HELD"
    assert rec["verifier"]["independent"] is False
    assert out.rstrip(b"\n") == C.jcs(rec).encode("utf-8")
    rp = tmp_path / "r.json"
    rp.write_bytes(out)
    ok = subprocess.run(exe + ["recheck", str(rp), str(doc), "--manifest", str(man)],
                        capture_output=True)
    assert (ok.returncode, ok.stdout.strip()) == (0, b"REDERIVES")
    bad = subprocess.run(exe + ["recheck", str(rp), str(doc)], capture_output=True)
    assert (bad.returncode, bad.stdout.strip()) == (1, b"DOES-NOT-REDERIVE")
    ref = subprocess.run(exe + ["check", str(doc), "--commit", "XYZ"], capture_output=True)
    assert (ref.returncode, ref.stdout.strip()) == (2, b"REFUSED commit_form")


# ---------------------------------------------------------------------------------------------
# the JavaScript checker, check_v0.js
# ---------------------------------------------------------------------------------------------

JS = V0 / "check_v0.js"
JS_RUNNER = V0 / "vectors" / "run_check_v0_js.js"


def _node():
    exe = shutil.which("node")
    assert exe, "node is not on PATH: the JavaScript checker's tests fail here, they do not skip"
    return exe


def _node_run(args):
    return subprocess.run([_node()] + [str(a) for a in args], capture_output=True, timeout=600)


def _js_lines(vectors_path):
    out = _node_run([JS_RUNNER, vectors_path, JS])
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")[-2000:]
    # split on LF only: str.splitlines would also split on U+2028, U+0085 and others that
    # JSON.stringify leaves raw inside strings
    return [json.loads(line) for line in out.stdout.decode("utf-8").split("\n") if line]


def test_every_vector_through_check_v0_js_and_receipts_are_byte_identical():
    """Each vector: check_v0.js answers what the vector expects, answers what check_v0.py answers,
    and on every check that emits a core the canonical text of receipt(core) (no verifier: that
    names the checker and is outside the digest, 9.4) is the same bytes from both checkers."""
    js = _js_lines(VECTORS)
    assert len(js) == len(ALL)
    wrong, receipts = [], 0
    for v, j in zip(ALL, js):
        assert j["id"] == v["id"]
        py_got, core = _run(v)
        if j["got"] != v["expect"] or j["got"] != py_got:
            wrong.append((v["id"][:16], v.get("label"), v["expect"], py_got, j["got"]))
            continue
        if core is None:
            assert "receipt_b64" not in j, v["id"]
            continue
        py_rec = C.jcs(C.receipt(core)).encode("utf-8")
        js_rec = base64.b64decode(j["receipt_b64"])
        if py_rec != js_rec:
            wrong.append((v["id"][:16], v.get("label"), "receipt bytes differ"))
        receipts += 1
    assert not wrong, "%d of %d vectors differ, first: %r" % (len(wrong), len(ALL), wrong[:3])
    expected = sum(1 for v in ALL if "core_digest" in v["expect"])
    assert receipts == expected and receipts > 1900


def test_check_v0_js_agrees_on_the_large_inputs(tmp_path):
    """The bounded-time inputs above, through check_v0.js: the same core digest as check_v0.py,
    all of them inside the sum of their limits."""
    vecs = []
    for label, doc, man, limit, _ in _BIG:
        vecs.append({"id": label, "op": "check", "inputs": {
            "document_b64": base64.b64encode(doc).decode("ascii"), "name": "big.md",
            "manifest_b64": None if man is None else base64.b64encode(man).decode("ascii"),
            "commit": None}})
    path = tmp_path / "big.json"
    path.write_text(json.dumps({"vectors": vecs}), encoding="utf-8")
    t0 = time.perf_counter()
    js = _js_lines(path)
    assert time.perf_counter() - t0 < sum(b[3] for b in _BIG)
    assert len(js) == len(_BIG)
    for (label, doc, man, _, _), j in zip(_BIG, js):
        assert j["got"] == {"core_digest": C.digest(C.check(doc, "big.md", man))}, label


_JS_SRC = JS.read_text(encoding="utf-8")
_JS_MARK = "/* ============================================================================ command line"
_JS_CORE = _JS_SRC.split(_JS_MARK, 1)[0]


def test_check_v0_js_is_one_file_and_names_nothing_impure():
    assert _JS_MARK in _JS_SRC
    raw = JS.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    import re
    code = re.sub(r"/\*.*?\*/", " ", _JS_CORE, flags=re.S)      # block comments
    code = re.sub(r"(?m)^\s*//.*$|\s//\s.*$", " ", code)          # line comments
    banned = ["require", "import", "process", "Buffer", "Date", "performance", "fetch",
              "XMLHttpRequest", "WebSocket", "random", "crypto", "parseFloat", "eval", "Function",
              "setTimeout", "setInterval", "localStorage", "navigator",
              "Float64Array", "Float32Array"]
    hits = [b for b in banned if re.search(r"(?<![\w$.])" + b + r"(?![\w$])", code)]
    hits += [b for b in ("JSON.parse", "JSON.stringify", "Math.random", "document.", "window.",
                         ".toFixed(", ".toPrecision(", ".toExponential(")
             if b in code]
    assert not hits, hits
    assert "check" in code and "ReceiptV0" in code


def test_check_v0_js_runs_with_no_node_globals():
    """The file evaluated as a browser would: no module, no require, no process, no Buffer; only
    the standard TextEncoder and TextDecoder. It defines ReceiptV0 and checks a document."""
    script = r"""
const fs = require("fs"), vm = require("vm");
const src = fs.readFileSync(process.argv[1], "utf8");
const g = { TextEncoder, TextDecoder };
g.globalThis = g;
vm.createContext(g);
vm.runInContext(src, g);
const R = g.ReceiptV0;
const doc = new g.TextEncoder().encode('n is <sworn r="r1" k="numeric">7</sworn>');
const core = R.check(vm.runInContext("Uint8Array", g).from(doc), "d.md");
process.stdout.write(JSON.stringify([typeof g.module, typeof g.require, typeof g.process,
  typeof g.Buffer, core.document_verdict, core.spans[0].reason, R.digest(core)]));
"""
    out = _node_run(["-e", script, JS])
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")[-2000:]
    got = json.loads(out.stdout)
    want = C.check(b'n is <sworn r="r1" k="numeric">7</sworn>', "d.md")
    assert got == ["undefined", "undefined", "undefined", "undefined", "SWORN-UNCHECKED",
                   "manifest_absent", C.digest(want)]


def test_check_v0_js_command_line(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_bytes(b'n is <sworn r="r1" k="numeric">7</sworn>')
    man = tmp_path / "m.json"
    man.write_bytes(_man(b"7"))
    out = _node_run([JS, "check", doc, "--manifest", man, "--name", "d.md"])
    assert out.returncode == 0, out.stderr
    rec = json.loads(out.stdout)
    assert rec["core"]["document_verdict"] == "SWORN-HELD"
    assert rec["verifier"]["independent"] is False
    assert rec["verifier"]["name"] == "receipt/v0/check_v0.js"
    assert out.stdout.rstrip(b"\n") == C.jcs(rec).encode("utf-8")
    assert rec["digest"] == C.digest(C.check(doc.read_bytes(), "d.md", man.read_bytes()))
    rp = tmp_path / "r.json"
    rp.write_bytes(out.stdout)
    ok = _node_run([JS, "recheck", rp, doc, "--manifest", man])
    assert (ok.returncode, ok.stdout.strip()) == (0, b"REDERIVES")
    assert C.recheck(rp.read_bytes(), doc.read_bytes(), man.read_bytes()) == "REDERIVES"
    bad = _node_run([JS, "recheck", rp, doc])
    assert (bad.returncode, bad.stdout.strip()) == (1, b"DOES-NOT-REDERIVE")
    ref = _node_run([JS, "check", doc, "--commit", "XYZ"])
    assert (ref.returncode, ref.stdout.strip()) == (2, b"REFUSED commit_form")


def test_check_v0_js_lists_are_the_python_lists():
    script = r"""
const R = require(process.argv[1]);
process.stdout.write(JSON.stringify([R.REASONS, R.REFUSALS, R.LEGACY_REASONS, R.DOCUMENT_REASONS,
  R.SPAN_VERDICTS, R.DOCUMENT_VERDICTS, R.SCHEMA_V0, R.PROFILE_RN, R.PROFILE_LEGACY, R.FORMAT,
  R.MANIFEST_SPEC]));
"""
    out = _node_run(["-e", script, JS])
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [
        list(C.REASONS), list(C.REFUSALS), list(C.LEGACY_REASONS), list(C.DOCUMENT_REASONS),
        list(C.SPAN_VERDICTS), list(C.DOCUMENT_VERDICTS), C.SCHEMA_V0, C.PROFILE_RN,
        C.PROFILE_LEGACY, C.FORMAT, C.MANIFEST_SPEC]


JS_PROBES = V0 / "vectors" / "check_v0_js_probes.json"


def test_check_v0_js_and_check_v0_py_agree_on_the_javascript_pitfall_probes():
    """check_v0_js_probes.json: inputs aimed at what JavaScript gets wrong by default. No expected
    answer is stored; the two checkers must give the same answer and byte-identical canonical
    receipts on every probe. The three marked unicode_gap split the two checkers while check_v0.js
    read the host's Unicode 16.0 tables and check_v0.py Python's 15.0.0; 7.2 now pins 15.0.0 and
    both checkers carry its tables as data, so those three are compared like the rest."""
    probes = json.loads(JS_PROBES.read_text(encoding="utf-8"))["vectors"]
    js = _js_lines(JS_PROBES)
    assert len(js) == len(probes) > 80
    wrong, gap = [], 0
    for v, j in zip(probes, js):
        assert j["id"] == v["id"]
        py_got, core = _run(v)
        if v["unicode_gap"]:
            gap += 1
        if j["got"] != py_got:
            wrong.append((v["id"], v["label"], py_got, j["got"]))
        elif core is not None and base64.b64decode(j["receipt_b64"]) != C.jcs(C.receipt(core)).encode("utf-8"):
            wrong.append((v["id"], v["label"], "receipt bytes differ"))
    assert gap == 3
    assert not wrong, wrong[:3]


# ---------------------------------------------------------------------------------------------
# the fixer's round (2026-11-13): 7.2's Unicode pin, pure-ASCII sources, the JavaScript sha256 at
# every padding boundary, and the mutants that nothing caught before the round
# ---------------------------------------------------------------------------------------------

# the digests RECEIPT_v0.md 7.2 states for its two tables
_LN_SHA = "4ca0e4c9044e908691020e69f4bd19bea0dbb8f67fb7c5488aa1e70f19121bf5"
_ND_SHA = "ef10fe9b74cef6d6af8debed5957b7c2b2af8909759ddabff75d8c34f95f222c"


def _table_text(pred):
    out, s = [], None
    for cp in range(0x110000):
        ok = pred(chr(cp))
        if ok and s is None:
            s = cp
        if not ok and s is not None:
            out.append((s, cp - 1))
            s = None
    if s is not None:
        out.append((s, 0x10FFFF))
    return " ".join(("%X" % a) if a == b else ("%X-%X" % (a, b)) for a, b in out)


def test_the_unicode_tables_are_the_ones_the_text_states():
    text = (V0 / "RECEIPT_v0.md").read_text(encoding="utf-8")
    assert "Unicode 15.0.0" in text and _LN_SHA in text and _ND_SHA in text
    assert hashlib.sha256(C._UNICODE_15_0_0_LN.encode("ascii")).hexdigest() == _LN_SHA
    assert hashlib.sha256(C._UNICODE_15_0_0_ND.encode("ascii")).hexdigest() == _ND_SHA
    assert len(C._UNICODE_15_0_0_LN.split(" ")) == 747 and len(C._UNICODE_15_0_0_ND.split(" ")) == 64
    import unicodedata
    if unicodedata.unidata_version != "15.0.0":
        pytest.skip("this Python's unicodedata is %s; the tables were checked against 15.0.0 only"
                    % unicodedata.unidata_version)
    assert C._UNICODE_15_0_0_LN == _table_text(lambda c: unicodedata.category(c)[0] in "LN")
    assert C._UNICODE_15_0_0_ND == _table_text(lambda c: unicodedata.category(c) == "Nd")


def test_check_v0_reads_no_host_unicode_table():
    assert "unicodedata" not in {a.name for n in ast.walk(_TREE) if isinstance(n, ast.Import)
                                 for a in n.names}
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("isalnum", "isalpha", "isdigit",
                                                               "isdecimal", "isnumeric")
                   for n in ast.walk(_TREE))


def test_check_v0_js_carries_the_same_unicode_tables():
    script = ("const R = require(process.argv[1]); "
              "process.stdout.write(JSON.stringify(R.UNICODE_TABLES));")
    out = _node_run(["-e", script, JS])
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")[-2000:]
    assert json.loads(out.stdout) == {"version": "15.0.0", "LN": C._UNICODE_15_0_0_LN,
                                      "ND": C._UNICODE_15_0_0_ND}
    assert "\\p{" not in _JS_CORE


@pytest.mark.parametrize("path", [V0 / "check_v0.py", V0 / "check_v0.js"], ids=["py", "js"])
def test_the_checker_source_is_pure_ascii(path):
    """A browser that decodes check_v0.js under another charset must read the same program."""
    raw = path.read_bytes()
    bad = [i for i, x in enumerate(raw) if x > 0x7E or (x < 0x20 and x != 0x0A)]
    assert not bad, bad[:5]


def test_check_v0_js_sha256_at_every_padding_boundary():
    script = ("const R = require(process.argv[1]); const out = [];"
              "for (let n = 0; n <= 200; n++) { const b = new Uint8Array(n);"
              " for (let i = 0; i < n; i++) b[i] = (i * 7 + n) & 255; out.push(R.sha256(b)); }"
              "process.stdout.write(JSON.stringify(out));")
    out = _node_run(["-e", script, JS])
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")[-2000:]
    want = [hashlib.sha256(bytes((i * 7 + n) & 255 for i in range(n))).hexdigest()
            for n in range(201)]
    assert json.loads(out.stdout) == want


# (id, file, rule, old line, new line): the differential attacker's mutants of check_v0.py and
# check_v0.js that its generator, the vectors and its probes let through before the fixer's round
# (PM30 and JM08 are equivalent and left out: no input can tell them apart). Each must now fail at
# least one vector. A line that no longer occurs exactly once fails the test, so an edit to the
# checker that moves one of these lines is noticed.
_LET_THROUGH = [
    ('PM14', 'py', '6.10 no whitespace in strict base64',
     '    if body.translate(_B64_DELETE):',
     '    if body.translate(_B64_DELETE).strip():'),
    ('PM31', 'py', '6.13 RFC 8259 control characters',
     '        if len(chunk.translate(_CTRL)) != len(chunk):',
     '        if False:'),
    ('PM34', 'py', '6.5 sha256 64 LOWERCASE hex',
     '    if not isinstance(entry, JObj) or not _is_hex64_lower(entry.get("sha256")):      # 6.5',
     '    if not isinstance(entry, JObj) or not _is_hex64_lower(str(entry.get("sha256")).lower()):      # 6.5'),
    ('PM39', 'py', '7.5 occurrences without overlap',
     '        occ = hay.count(needle)',
     '        occ = sum(1 for i in range(len(hay)) if hay.startswith(needle, i))'),
    ('PM41', 'py', '7.1b VT is ASCII whitespace',
     '_ASCII_WS = b" \\t\\n\\x0b\\x0c\\r"',
     '_ASCII_WS = b" \\t\\n\\x0c\\r"'),
    ('PM43', 'py', '4.4 U+0085 banned in a path target',
     '    "\\u007f", "\\u0085", "\\u00a0", "\\u1680", "\\u2028", "\\u2029", "\\u202f", "\\u205f", "\\u3000"} | set(',
     '    "\\u007f", "\\u00a0", "\\u1680", "\\u2028", "\\u2029", "\\u202f", "\\u205f", "\\u3000"} | set('),
    ('PM47', 'py', '7.3 adjusted exponent of a huge negative exponent',
     '        return True, _str_add(d.exp_text, d.nfrac - shift)',
     '        return True, _str_add(d.exp_text, shift - d.nfrac)'),
    ('JM06', 'js', '3.1 1E2 is not an integer',
     '    isIntegerToken() { return !/[.eE]/.test(this.text); }',
     '    isIntegerToken() { return !/[.e]/.test(this.text); }'),
    ('JM10', 'js', '7.3 leaf: first 80 code points',
     '    const head = cpHead(text, 80);',
     '    const head = text.slice(0, 80);'),
    ('JM14', 'js', '6.13 JSON whitespace is four bytes',
     '      if (c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d) i++;',
     '      if (c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d || c === 0x0b) i++;'),
    ('JM15', 'js', '7.5 occurrences without overlap',
     '      i = j + needle.length;',
     '      i = j + 1;'),
    ('JM19', 'js', '7.3 adjusted exponent of a huge negative exponent',
     '    if (d.expNeg) return [true, strAddSmall(d.expText, d.nfrac - shift)];',
     '    if (d.expNeg) return [true, strAddSmall(d.expText, shift - d.nfrac)];'),
]


def _run_mod(M, v):
    i = v["inputs"]
    try:
        if v["op"] == "check":
            return {"core_digest": M.digest(M.check(_b(i["document_b64"]), i["name"],
                                                    _b(i["manifest_b64"]), i["commit"]))}
        if v["op"] == "recheck":
            return {"answer": M.recheck(_b(i["receipt_b64"]), _b(i["document_b64"]),
                                        _b(i["manifest_b64"]))}
        return {"canonical": M.jcs(M.parse_json(_b(i["value_b64"])))}
    except M.Refusal as e:
        return {"refusal": e.code}
    except Exception as e:                      # a crash is a wrong answer
        return {"error": type(e).__name__}


@pytest.mark.parametrize("mid,lang,rule,old,new", _LET_THROUGH, ids=[m[0] for m in _LET_THROUGH])
def test_a_mutant_that_nothing_caught_now_fails_a_vector(mid, lang, rule, old, new, tmp_path):
    path = V0 / ("check_v0.py" if lang == "py" else "check_v0.js")
    lines = path.read_text(encoding="utf-8").split("\n")
    assert sum(1 for x in lines if x == old) == 1, (mid, "the mutated line moved")
    src = "\n".join(new if x == old else x for x in lines)
    if lang == "py":
        import types
        M = types.ModuleType("receipt_check_v0_mutant_" + mid)
        exec(compile(src, str(tmp_path / (mid + ".py")), "exec"), M.__dict__)
        failing = [v["id"] for v in ALL if _run_mod(M, v) != v["expect"]]
    else:
        mp = tmp_path / (mid + ".js")
        mp.write_bytes(src.encode("utf-8"))
        out = _node_run([JS_RUNNER, VECTORS, mp])
        if out.returncode != 0:
            return                                  # the mutant does not even run: caught
        got = [json.loads(x) for x in out.stdout.decode("utf-8").split("\n") if x]
        failing = [v["id"] for v, j in zip(ALL, got) if j["got"] != v["expect"]]
    assert failing, "%s (%s) passes every vector" % (mid, rule)
