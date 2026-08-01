# -*- coding: utf-8 -*-
"""styxx.diffgate — the summary cannot lie about the diff. Catches are the product."""
import subprocess

from styxx.diffgate import gate_diff


def _repo(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "docs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def old():\n    return 1\n", encoding="utf-8")
    (tmp_path / "docs" / "readme.md").write_text("# hi\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True,
                          text=True).stdout.strip()
    # the change: edit src/app.py (add function retry + a test file), edit docs
    (tmp_path / "src" / "app.py").write_text(
        "def old():\n    return 1\n\ndef retry(n):\n    return n\n", encoding="utf-8")
    (tmp_path / "tests_new.py").write_text(
        "def test_retry():\n    assert True\n\ndef test_retry_twice():\n    assert True\n",
        encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "change")
    return base


def test_honest_summary_passes(tmp_path):
    base = _repo(tmp_path)
    g = gate_diff("Modified src/app.py and created tests_new.py. "
                  "Adds function retry. Added 2 tests. 2 files changed.",
                  tmp_path, base, "HEAD")
    assert g.verdict == "PASS"
    assert all(c.verdict == "VERIFIED" for c in g.claims)


def test_phantom_file_claim_contradicted(tmp_path):
    base = _repo(tmp_path)
    g = gate_diff("Updated docs/readme.md with the new usage section.",
                  tmp_path, base, "HEAD")
    assert g.verdict == "FAIL"
    assert g.claims[0].verdict == "CONTRADICTED"
    assert "does not appear in the diff" in g.claims[0].why


def test_phantom_symbol_contradicted(tmp_path):
    base = _repo(tmp_path)
    g = gate_diff("Adds function backoff for resilience.", tmp_path, base, "HEAD")
    assert g.verdict == "FAIL"
    assert any(c.kind == "symbol_added" and c.verdict == "CONTRADICTED" for c in g.claims)


def test_wrong_test_count_contradicted(tmp_path):
    base = _repo(tmp_path)
    g = gate_diff("Added 5 tests covering the retry path.", tmp_path, base, "HEAD")
    assert g.verdict == "FAIL"
    assert "adds 2 test functions, claim says 5" in g.claims[0].why


def test_only_touches_lie_contradicted(tmp_path):
    base = _repo(tmp_path)
    g = gate_diff("This change only touches files under docs/.", tmp_path, base, "HEAD")
    assert g.verdict == "FAIL"
    assert any(c.kind == "only_touches" and c.verdict == "CONTRADICTED" for c in g.claims)


def test_tests_pass_uncheckable_without_run_and_gate_still_passes(tmp_path):
    base = _repo(tmp_path)
    g = gate_diff("Modified src/app.py. All tests pass.", tmp_path, base, "HEAD")
    assert g.verdict == "PASS"           # uncheckable is not a lie — unless --strict
    tp = [c for c in g.claims if c.kind == "tests_pass"][0]
    assert tp.verdict == "UNCHECKABLE" and "does not take the agent's word" in tp.why


def test_strict_makes_uncheckable_fatal(tmp_path):
    base = _repo(tmp_path)
    g = gate_diff("Modified src/app.py. All tests pass.", tmp_path, base, "HEAD",
                  strict=True)
    assert g.verdict == "FAIL"


def test_tests_pass_verified_with_run(tmp_path):
    base = _repo(tmp_path)
    g = gate_diff("Modified src/app.py. Tests pass.", tmp_path, base, "HEAD",
                  run="python -c \"import sys; sys.exit(0)\"")
    assert g.verdict == "PASS"
    assert [c for c in g.claims if c.kind == "tests_pass"][0].verdict == "VERIFIED"


def test_uncovered_prose_is_counted_not_judged(tmp_path):
    base = _repo(tmp_path)
    g = gate_diff("Modified src/app.py. This work is brilliant and revolutionary.",
                  tmp_path, base, "HEAD")
    assert g.verdict == "PASS"
    assert g.uncovered_sentences >= 1
