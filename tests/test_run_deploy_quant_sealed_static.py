# -*- coding: utf-8 -*-
"""run_deploy_quant.py's sealed constants and gates, checked without importing the runner.

Verification of 2026-09-14, ci-1: the runner imports torch at module level and CI installs no torch, so
tests/test_run_deploy_quant_sealed.py skips on every CI leg, and a drift between the runner's SEALED
digests and the scorer's would leave CI green. This file never imports the runner. It reads the runner as
text with ast: SEALED and PREREGS are the dict literals in its source, compared with the BANDS of
papers/checksum/score.py (which imports only the standard library). The gates sealed_refusal and
beacon_refusal are compiled from the runner's own source (those function definitions and the module-level
assignments they read, nothing else, so no torch import runs) and exercised:

  RDQ-1  an untracked file under styxx/, or a checksum module that is not the checkout's styxx/checksum.py,
         refuses the experiment; the runner's own git query sees an untracked styxx/checksum/ package that
         `git status --untracked-files=no` hides, and Python imports that package over the module;
  RDQ-3  a `git status` that did not answer is not a clean tree;
  RDQ-2  the beacon_draw experiment runs only under the beacon of a sealed-prereg line that
         styxx.clock.check_line reads ANCHORED; the chain is the fixture from tests/test_clock.py, so no
         network is used; main() makes that check before any model loads, for that experiment only.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

from styxx import clock
from tests.test_clock import ATTACKER, BLOCKHASH_BYTES, _chain, _hist, _tx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "papers", "checksum", "run_deploy_quant.py")
GATES = ("_same_file", "sealed_refusal", "beacon_refusal")
BEACON = BLOCKHASH_BYTES.hex()


def _tree():
    with open(RUNNER, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=RUNNER)


def _assign(tree, name):
    found = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(found) == 1, f"run_deploy_quant.py assigns {name} at module level {len(found)} times, expected once"
    return found[0]


def _function(tree, name):
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, f"run_deploy_quant.py defines {name} {len(found)} times, expected once"
    return found[0]


def _callee(call):
    return call.func.id if isinstance(call.func, ast.Name) else (call.func.attr if isinstance(call.func, ast.Attribute) else None)


@pytest.fixture(scope="module")
def gates():
    tree = _tree()
    body = [_assign(tree, n) for n in ("PREREGS", "SEALED", "ANCHORS")] + [_function(tree, g) for g in GATES]
    ns = {"__name__": "run_deploy_quant_gates", "os": os, "json": json, "ROOT": ROOT}
    exec(compile(ast.Module(body=body, type_ignores=[]), RUNNER, "exec"), ns)
    return ns


def _prov(g, **over):
    p = {"git_head": "a" * 40, "prereg_blob_sha256": g["SEALED"]["beacon_draw"], "git_dirty_tracked": False,
         "styxx_untracked": [], "checksum_file": os.path.join(ROOT, "styxx", "checksum.py"), "checksum_py_sha256": "b" * 64}
    p.update(over)
    return p


# ci-1 ---------------------------------------------------------------------------------------------------------

def test_the_runner_and_the_scorer_freeze_the_same_sealed_digests_read_without_torch():
    tree = _tree()
    spec = importlib.util.spec_from_file_location("score_for_static_runner_test", os.path.join(ROOT, "papers", "checksum", "score.py"))
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    sealed = ast.literal_eval(_assign(tree, "SEALED").value)
    preregs = ast.literal_eval(_assign(tree, "PREREGS").value)
    assert sealed == {k: v["sealed_blob"] for k, v in score.BANDS.items()}
    assert preregs == {k: v["prereg"] for k, v in score.BANDS.items()}
    assert sorted(sealed) == ["beacon_draw", "deploy_quant"]


# RDQ-3 --------------------------------------------------------------------------------------------------------

def test_a_git_status_that_did_not_answer_is_not_a_clean_tree(gates):
    sr = gates["sealed_refusal"]
    assert sr(_prov(gates), "beacon_draw", True) is None
    assert "git did not answer" in sr(_prov(gates, git_dirty_tracked=None), "beacon_draw", True)
    missing = _prov(gates)
    del missing["git_dirty_tracked"]
    assert "git did not answer" in sr(missing, "beacon_draw", True)
    assert "git did not answer" in sr(_prov(gates, styxx_untracked=None), "beacon_draw", True)
    assert "tracked files differ" in sr(_prov(gates, git_dirty_tracked=True), "beacon_draw", True)
    # the older gates stand
    assert "not the sealed digest" in sr(_prov(gates, prereg_blob_sha256="0" * 64), "beacon_draw", True)
    assert "git did not answer" in sr(_prov(gates, git_head=None), "beacon_draw", True)
    # an instrument check is never refused
    assert sr(_prov(gates, git_dirty_tracked=None, styxx_untracked=None, checksum_file=None), "beacon_draw", False) is None


# RDQ-1 --------------------------------------------------------------------------------------------------------

def test_untracked_files_under_styxx_refuse_the_experiment(gates):
    sr = gates["sealed_refusal"]
    msg = sr(_prov(gates, styxx_untracked=["styxx/checksum/__init__.py"]), "beacon_draw", True)
    assert "untracked files exist under styxx/" in msg and "styxx/checksum/__init__.py" in msg
    many = [f"styxx/x{i}.py" for i in range(8)]
    assert "and 3 more" in sr(_prov(gates, styxx_untracked=many), "beacon_draw", True)


def test_the_imported_checksum_must_be_the_checkouts_checksum_py_and_must_be_hashed(gates, tmp_path):
    sr = gates["sealed_refusal"]
    shadow = os.path.join(ROOT, "styxx", "checksum", "__init__.py")
    assert "checksum module that was imported" in sr(_prov(gates, checksum_file=shadow), "beacon_draw", True)
    assert "checksum module that was imported" in sr(_prov(gates, checksum_file=None), "beacon_draw", True)
    elsewhere = str(tmp_path / "styxx" / "checksum.py")
    assert "checksum module that was imported" in sr(_prov(gates, checksum_file=elsewhere), "beacon_draw", True)
    assert "could not be read and hashed" in sr(_prov(gates, checksum_py_sha256=None), "beacon_draw", True)


def test_python_imports_an_untracked_checksum_package_over_the_module_and_the_gate_refuses_it(gates, tmp_path):
    sr = gates["sealed_refusal"]
    pkg = tmp_path / "styxx"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "checksum.py").write_text("WHO = 'module'\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}

    def imported():
        r = subprocess.run([sys.executable, "-c", "from styxx import checksum as c; print(c.WHO); print(c.__file__)"],
                           cwd=str(tmp_path), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=120)
        assert r.returncode == 0, r.stderr
        who, path = r.stdout.strip().splitlines()[-2:]
        return who, path

    who, path = imported()
    assert who == "module"
    assert sr(_prov(gates, checksum_file=path), "beacon_draw", True, root=str(tmp_path)) is None
    (pkg / "checksum").mkdir()
    (pkg / "checksum" / "__init__.py").write_text("WHO = 'shadow'\n", encoding="utf-8")
    who, path = imported()
    assert who == "shadow"
    assert "checksum module that was imported" in sr(_prov(gates, checksum_file=path), "beacon_draw", True, root=str(tmp_path))


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")
def test_the_runners_git_query_sees_an_untracked_checksum_package_that_status_untracked_no_hides(tmp_path):
    prov_fn = _function(_tree(), "provenance")
    calls = [[a.value for a in n.args] for n in ast.walk(prov_fn)
             if isinstance(n, ast.Call) and _callee(n) == "_git" and all(isinstance(a, ast.Constant) for a in n.args)]
    untracked = [c for c in calls if c[:1] == ["ls-files"]]
    tracked = [c for c in calls if c[:1] == ["status"] and "--untracked-files=no" in c]
    assert untracked == [["ls-files", "--others", "--exclude-standard", "--", "styxx"]]
    assert len(tracked) == 1

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=60, check=True)

    git("init", "-q")
    (tmp_path / "styxx").mkdir()
    (tmp_path / "styxx" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "styxx" / "checksum.py").write_text("WHO = 'module'\n", encoding="utf-8")
    git("add", "styxx")
    (tmp_path / "styxx" / "checksum").mkdir()
    (tmp_path / "styxx" / "checksum" / "__init__.py").write_text("WHO = 'shadow'\n", encoding="utf-8")
    assert "styxx/checksum/" not in git(*tracked[0]).stdout
    assert git(*untracked[0]).stdout.splitlines() == ["styxx/checksum/__init__.py"]


# RDQ-2 --------------------------------------------------------------------------------------------------------

def _anchors(tmp_path, *lines):
    p = tmp_path / "anchors.jsonl"
    p.write_text("".join((x if isinstance(x, str) else json.dumps(x)) + "\n" for x in lines), encoding="utf-8")
    return str(p)


def _seal_line(g, **over):
    d = g["SEALED"]["beacon_draw"]
    line = {"n": 6, "kind": "sealed-prereg", "digest": d, "tx": "sig1", "memo": clock.memo("sealed-prereg", d)}
    line.update(over)
    return line


def _seal_tx(g, **over):
    return _tx(clock.memo("sealed-prereg", g["SEALED"]["beacon_draw"]), **over)


def test_the_runner_reads_the_seals_from_papers_charon_anchors_jsonl(gates):
    assert gates["ANCHORS"] == os.path.join("papers", "charon", "anchors.jsonl")


def test_the_seals_beacon_is_the_only_beacon_the_beacon_draw_experiment_runs_under(gates, tmp_path):
    br, d = gates["beacon_refusal"], gates["SEALED"]["beacon_draw"]
    path = _anchors(tmp_path, "not json", {"n": 1, "kind": "sworn-receipt", "digest": "c" * 64, "tx": "other"}, _seal_line(gates))
    assert br(BEACON, path, d, _chain(_seal_tx(gates))) is None
    msg = br("0" * 64, path, d, _chain(_seal_tx(gates)))
    assert "is not the seal's beacon" in msg and BEACON in msg


def test_no_anchors_file_or_no_seal_line_for_the_digest_refuses(gates, tmp_path):
    br, d = gates["beacon_refusal"], gates["SEALED"]["beacon_draw"]
    assert "does not exist" in br(BEACON, str(tmp_path / "absent.jsonl"), d, _chain(_seal_tx(gates)))
    other = gates["SEALED"]["deploy_quant"]
    path = _anchors(tmp_path, {"n": 3, "kind": "sealed-prereg", "digest": other, "tx": "sig1"},
                    {"n": 7, "kind": "sealed-canaries", "digest": d, "tx": "sig1"})
    assert "has no sealed-prereg line" in br(BEACON, path, d, _chain(_seal_tx(gates)))


def test_a_seal_line_the_chain_does_not_read_anchored_refuses(gates, tmp_path):
    br, d = gates["beacon_refusal"], gates["SEALED"]["beacon_draw"]
    path = _anchors(tmp_path, _seal_line(gates))
    msg = br(BEACON, path, d, _chain(None))
    assert "reads ANCHORED" in msg and "NOT_FOUND" in msg
    assert "NOT_CREATOR" in br(BEACON, path, d, _chain(_seal_tx(gates, signer=ATTACKER)))
    assert "RPC_ERROR" in br(BEACON, path, d, _chain(_seal_tx(gates), tx_raises=True))

    def broken(method, params, rpcs=None):
        raise OSError("socket closed")
    assert "check failed (OSError" in br(BEACON, path, d, broken)


def test_a_seal_line_that_has_a_beacon_but_is_not_the_earliest_seal_refuses(gates, tmp_path):
    # check_line computes the slot's beacon before the earliest-memo scan, so these lines carry a beacon
    # equal to --beacon; only the ANCHORED status makes that beacon the seal's
    br, d = gates["beacon_refusal"], gates["SEALED"]["beacon_draw"]
    path = _anchors(tmp_path, _seal_line(gates))
    m = clock.memo("sealed-prereg", d)
    earlier = _hist(("sig1", 123, None, f"[{len(m)}] {m}"), ("sig0", 99, None, f"[{len(m)}] {m}"))
    msg = br(BEACON, path, d, _chain(_seal_tx(gates), history=earlier))
    assert "EARLIER_MEMO_EXISTS" in msg and "reads ANCHORED" in msg
    assert "EARLIEST_UNKNOWN" in br(BEACON, path, d, _chain(_seal_tx(gates), history_raises=True))


def test_main_checks_the_beacon_before_any_model_loads_and_only_for_the_beacon_draw_experiment():
    main = _function(_tree(), "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    br = [n for n in calls if _callee(n) == "beacon_refusal"]
    assert len(br) == 1
    loads = [n.lineno for n in calls if _callee(n) == "load"]
    draws = [n.lineno for n in calls if _callee(n) == "draw"]
    assert loads and draws and br[0].lineno < min(loads) and br[0].lineno < min(draws)
    guards = [n for n in ast.walk(main) if isinstance(n, ast.If) and any(m is br[0] for m in ast.walk(n))]
    names = {x.id for g in guards for x in ast.walk(g.test) if isinstance(x, ast.Name)}
    consts = {x.value for g in guards for x in ast.walk(g.test) if isinstance(x, ast.Constant)}
    assert "is_the_experiment" in names and "beacon_draw" in consts
