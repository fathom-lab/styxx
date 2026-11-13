# -*- coding: utf-8 -*-
"""RECEIPT v0 (DRAFT, unsigned): the fold round of 2026-11-13.

  - the fold vector file receipt/v0/vectors/receipt_v0_vectors_fold.json is what it says and its
    builder re-derives it;
  - every fold vector through check_v0.py and check_v0.js (the JavaScript test FAILS, it does not
    skip, when node is missing), byte-identical canonical receipts between the two;
  - JSON nested 3,000,000 deep, built in memory, through both lab checkers (6.13: no depth changes
    an answer outside 3.2, and a checker must not stop without an answer);
  - the clean-room Go checker kept as evidence in receipt/v0/cleanroom-go/: when go is on PATH it
    is built offline from the copy and run over both vector files, and the test requires exactly
    the counts its README states (2193 of 2193 on the main file, byte for byte its first run, and
    59 of 62 on the fold file, the three it fails being 3.3's order-of-faults vectors). When go is
    not on PATH the test says so loudly and skips: the Go build is evidence, not a supported
    implementation. When the Go on PATH is older than the module file names, the committed
    folder is left alone and a temporary copy whose go.mod names the running version is built
    from the same source files.

All three checkers were written inside the lab, the Go one by an agent of the same model family as
the lab's authors. Agreement among them is agreement inside one lab.
"""
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V0 = Path(os.environ.get("RECEIPT_V0_DIR", str(ROOT / "receipt" / "v0")))
MAIN = V0 / "vectors" / "receipt_v0_vectors.json"
FOLD = V0 / "vectors" / "receipt_v0_vectors_fold.json"
GO_SRC = V0 / "cleanroom-go" / "checker"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("receipt_check_v0_fold_test", V0 / "check_v0.py")
DATA = json.loads(FOLD.read_text(encoding="utf-8"))
ALL = DATA["vectors"]

# the fold vectors on which the clean-room Go build takes another reading of 3.3 (it walks an
# object's values in the order written, the text in the code-point order of their keys)
GO_FAILS_ON = {
    "values in the code-point order of their keys: '' before '9'",
    "values in key order, depth first: the value of 'a' is walked whole before 'b'",
    "values in key order: 'a' before 'b' whatever the written order",
}


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
    if isinstance(x, list):
        return [_id_safe(y) for y in x]
    return x


# ---------------------------------------------------------------------------------------------
# the fold vector file
# ---------------------------------------------------------------------------------------------

def test_the_fold_vector_file_is_what_it_says():
    raw = FOLD.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    by = {}
    for v in ALL:
        by[v["source"]] = by.get(v["source"], 0) + 1
        assert v["id"] == C.digest({"op": v["op"], "inputs": _id_safe(v["inputs"])}), v["id"]
        assert v["rules"] and v["gap"] and v["label"], v["id"]
    assert by == {k: DATA["counts"][k] for k in by} and DATA["counts"]["total"] == len(ALL)
    ids = [v["id"] for v in ALL]
    assert len(set(ids)) == len(ids)
    main_ids = {v["id"] for v in json.loads(MAIN.read_text(encoding="utf-8"))["vectors"]}
    assert not set(ids) & main_ids
    assert GO_FAILS_ON <= {v["label"] for v in ALL}


def test_the_fold_builder_re_derives_the_file():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, str(V0 / "vectors" / "build_receipt_v0_vectors_fold.py"),
                          "--check"], capture_output=True, env=env, timeout=600)
    assert out.returncode == 0, (out.stdout + out.stderr).decode("utf-8", "replace")[-2000:]


def test_every_fold_vector_through_check_v0():
    wrong = []
    for v in ALL:
        got, core = _run(v)
        if got != v["expect"]:
            wrong.append((v["label"], v["expect"], got))
        elif "expect_rows" in v:
            rows = [[r["verdict"], r["reason"]] for r in core["spans"]]
            if rows != v["expect_rows"] or core["document_verdict"] != v["expect_document_verdict"]:
                wrong.append((v["label"], v["expect_rows"], rows))
    assert not wrong, "%d of %d fold vectors differ, first: %r" % (len(wrong), len(ALL), wrong[:3])


# ---------------------------------------------------------------------------------------------
# check_v0.js
# ---------------------------------------------------------------------------------------------

def _node_lines(vectors_path):
    exe = shutil.which("node")
    assert exe, "node is not on PATH: the JavaScript checker's tests fail here, they do not skip"
    out = subprocess.run([exe, str(V0 / "vectors" / "run_check_v0_js.js"), str(vectors_path),
                          str(V0 / "check_v0.js")], capture_output=True, timeout=900)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")[-2000:]
    return [json.loads(line) for line in out.stdout.decode("utf-8").split("\n") if line]


def test_every_fold_vector_through_check_v0_js_and_receipts_are_byte_identical():
    js = _node_lines(FOLD)
    assert len(js) == len(ALL)
    wrong, receipts = [], 0
    for v, j in zip(ALL, js):
        assert j["id"] == v["id"]
        py_got, core = _run(v)
        if j["got"] != v["expect"] or j["got"] != py_got:
            wrong.append((v["label"], v["expect"], py_got, j["got"]))
            continue
        if core is not None:
            if base64.b64decode(j["receipt_b64"]) != C.jcs(C.receipt(core)).encode("utf-8"):
                wrong.append((v["label"], "receipt bytes differ"))
            receipts += 1
    assert not wrong, wrong[:3]
    assert receipts == sum(1 for v in ALL if "core_digest" in v["expect"])


# ---------------------------------------------------------------------------------------------
# 6.13: any nesting depth, built in memory (6 MB of brackets per input)
# ---------------------------------------------------------------------------------------------

DEEP = 3000000


def _man(data):
    m = {"spec": "sworn/manifest/0.2", "harness": "h", "turn": "t", "minted_at": "x",
         "authored_sha256": [], "rung": "L2",
         "receipts": {"r1": {"id": "r1", "sha256": hashlib.sha256(data).hexdigest(),
                             "kind_of_source": "tool_stdout", "captured_at": "x", "complete": True,
                             "bytes": base64.b64encode(data).decode("ascii")}}}
    return json.dumps(m).encode("utf-8")


def _deep_cases():
    span = b'<sworn r="r1" k="numeric">n is 7</sworn>'
    return [
        ("jcs", {"value_b64": base64.b64encode(b"[" * DEEP + b"]" * DEEP).decode("ascii")},
         {"refusal": "depth"}),
        ("jcs", {"value_b64": base64.b64encode(b'{"a":' * DEEP + b"1" + b"}" * DEEP).decode("ascii")},
         {"refusal": "depth"}),
        ("check", {"document_b64": base64.b64encode(span).decode("ascii"), "name": "d.md",
                   "manifest_b64": base64.b64encode(_man(b'{"a":' * DEEP + b"1" + b"}" * DEEP)
                                                    ).decode("ascii"), "commit": None}, None),
    ]


def test_json_nested_three_million_deep_through_both_lab_checkers(tmp_path):
    vecs = []
    t0 = time.perf_counter()
    for k, (op, inputs, expect) in enumerate(_deep_cases()):
        v = {"id": "deep%d" % k, "op": op, "inputs": inputs}
        got, core = _run(v)
        if expect is None:
            assert core["spans"][0]["reason"] == "leaf_not_scalar"
            assert core["spans"][0]["detail"] == {"leaf_type": "object"}
        else:
            assert got == expect
        v["py"] = got
        vecs.append(v)
    assert time.perf_counter() - t0 < 180
    path = tmp_path / "deep.json"
    path.write_text(json.dumps({"vectors": vecs}), encoding="utf-8")
    t0 = time.perf_counter()
    js = _node_lines(path)
    assert time.perf_counter() - t0 < 180
    assert [j["got"] for j in js] == [v["py"] for v in vecs]


# ---------------------------------------------------------------------------------------------
# the clean-room Go build, kept as evidence
# ---------------------------------------------------------------------------------------------

def _go_checker(tmp_path):
    go = shutil.which("go")
    if not go:
        msg = ("GO IS NOT ON PATH: the clean-room Go checker in receipt/v0/cleanroom-go was NOT "
               "built or run, so this run says nothing about it")
        warnings.warn(msg)
        sys.stderr.write("\n*** %s ***\n" % msg)
        pytest.skip(msg)
    env = dict(os.environ, GOTOOLCHAIN="local", GOPROXY="off", GOSUMDB="off", GOFLAGS="-mod=mod",
               CGO_ENABLED="0")
    exe = tmp_path / ("checker.exe" if os.name == "nt" else "checker")
    src = GO_SRC
    # The module file names the Go the clean-room build was written with. A toolchain older than
    # that refuses the module outright (GOTOOLCHAIN=local, and nothing may be downloaded here). The
    # committed folder is evidence and is never edited: for an older toolchain the same source
    # files are copied to a temporary folder whose go.mod names the running version instead, and
    # the copy is built. If the copy does not build, this run says nothing about the Go checker.
    mod_text = (GO_SRC / "go.mod").read_text(encoding="utf-8")
    go_line = [ln for ln in mod_text.split("\n") if ln.startswith("go ")]
    want = _go_minor(go_line[0]) if len(go_line) == 1 else None
    probe = subprocess.run([go, "env", "GOVERSION"], cwd=str(tmp_path), env=env,
                           capture_output=True, timeout=120)
    have = _go_minor(probe.stdout.decode("utf-8", "replace"))
    lowered = want is not None and have is not None and have < want
    if lowered:
        src = tmp_path / "cleanroom_go_copy"
        shutil.copytree(str(GO_SRC), str(src))
        # open(), not Path.write_text(newline=...): that keyword is new in Python 3.10
        with open(str(src / "go.mod"), "w", encoding="utf-8", newline="\n") as f:
            f.write(mod_text.replace(go_line[0], "go %d.%d" % have))
        sys.stderr.write("\n*** go %d.%d is older than the clean-room module names (%d.%d): building "
                         "a temporary copy whose go.mod names the running version ***\n"
                         % (have + want))
    out = subprocess.run([go, "build", "-o", str(exe), "."], cwd=str(src), env=env,
                         capture_output=True, timeout=600)
    if lowered and out.returncode != 0:
        msg = ("GO %d.%d IS OLDER THAN THE CLEAN-ROOM MODULE NAMES (%d.%d) and the copy with the "
               "version line lowered did not build, so this run says nothing about the Go checker: %s"
               % (have + want + (out.stderr.decode("utf-8", "replace")[-600:],)))
        warnings.warn(msg)
        sys.stderr.write("\n*** %s ***\n" % msg)
        pytest.skip(msg)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")[-2000:]
    return exe


def _go_minor(text):
    """(major, minor) read from 'go1.24.13', 'go 1.25' or '1.25'; None when it does not read as one."""
    t = text.strip()
    if t.startswith("go"):
        t = t[2:].strip()
    parts = t.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def test_the_clean_room_go_checker_over_both_vector_files(tmp_path):
    exe = _go_checker(tmp_path)
    run = subprocess.run([str(exe), "run", str(MAIN)], capture_output=True, timeout=600)
    assert run.returncode == 0, run.stderr.decode("utf-8", "replace")[-2000:]
    assert b"\nSUMMARY: 2193 of 2193 vectors pass\n" in run.stdout
    # the copy builds to the checker that wrote first_run.txt: the same bytes, line for line
    first = (V0 / "cleanroom-go" / "first_run.txt").read_bytes()
    assert run.stdout.replace(b"\r\n", b"\n") == first
    fold = subprocess.run([str(exe), "run", str(FOLD)], capture_output=True, timeout=600)
    text = fold.stdout.decode("utf-8").replace("\r\n", "\n")
    assert "\nSUMMARY: 59 of 62 vectors pass\n" in text, text[-600:]
    failed = {line.split(" ")[1] for line in text.split("\n") if line.startswith("FAIL ")}
    by_label = {v["label"]: v["id"] for v in ALL}
    assert failed == {by_label[label] for label in GO_FAILS_ON}
