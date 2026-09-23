# -*- coding: utf-8 -*-
"""`styxx ci-audit`, confined (SWALLOW-15): the simulated steps' shell may write only beneath a
scratch directory of its own, open no TCP connection and signal nothing outside the audit; the audit
reads the same with and without it; the CLI refuses where the kernel cannot confine it; the Action
refuses on a runner someone keeps."""
from __future__ import annotations

import errno
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from styxx.ciaudit import confine as C  # noqa: E402
from tests.test_ciaudit_frontier import FRONTIER_FIXTURE, LISTS_FIXTURE, _tree  # noqa: E402

landlock = pytest.mark.skipif(C.abi() == 0, reason="this kernel has no Landlock")


def _errno(f) -> str:
    try:
        f()
        return "allowed"
    except OSError as e:
        return errno.errorcode.get(e.errno, str(e.errno))


@landlock
def test_the_confined_process_writes_only_in_its_scratch(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    parent = os.getpid()

    def probe():
        scratch = Path(os.environ["TMPDIR"])
        keep = outside / "keep"

        def connect():
            s = socket.socket()
            s.settimeout(5)
            s.connect(("127.0.0.1", port))
            s.close()
        got = {
            "write inside": _errno(lambda: (scratch / "a").write_text("x")),
            "remove inside": _errno(lambda: (scratch / "a").unlink()),
            "create outside": _errno(lambda: (outside / "new").write_text("x")),
            "truncate outside": _errno(lambda: open(keep, "w").close()),
            "append outside": _errno(lambda: open(keep, "a").write("x")),
            "remove outside": _errno(lambda: keep.unlink()),
            "rename outside": _errno(lambda: keep.rename(outside / "moved")),
            "mkdir outside": _errno(lambda: (outside / "d").mkdir()),
            "symlink outside": _errno(lambda: (outside / "ln").symlink_to("/etc")),
            "write /dev/null": _errno(lambda: open("/dev/null", "w").write("x")),
            "read outside": _errno(lambda: keep.read_text()),
        }
        if C.abi() >= 4:
            got["tcp connect"] = _errno(connect)
        if C.abi() >= 6:
            got["signal the caller"] = _errno(lambda: os.kill(parent, 0))
        p = subprocess.run(["bash", "-c", f'rm -rf "{outside}"/*; echo x > "{outside}/from-bash"; echo "$?"'], capture_output=True, text=True)
        got["bash"] = p.stdout.strip()
        return got

    got, info = C.run(probe)
    srv.close()
    assert info["confined"] and info["mechanism"] == "landlock" and info["abi"] == C.abi()
    assert got["write inside"] == got["remove inside"] == got["write /dev/null"] == got["read outside"] == "allowed"
    for k in ("create outside", "truncate outside", "append outside", "remove outside", "rename outside", "mkdir outside", "symlink outside"):
        assert got[k] == "EACCES", k
    assert got.get("tcp connect", "EACCES") == "EACCES" and got.get("signal the caller", "EPERM") == "EPERM"
    assert got["bash"] == "1"                                   # the redirect refused; the child's shell inherits the confinement
    assert sorted(p.name for p in outside.iterdir()) == ["keep"] and (outside / "keep").read_text() == "keep"


CANARY_STEP = """on: [push]
jobs:
  t:
    runs-on: ubuntu-latest
    steps:
      - name: Clean the report dir, then test
        run: |
          D="$(find-report-dir)"
          rm -rf "$D"{canary}/*
          npm test || true
"""


@landlock
def test_a_step_that_deletes_through_an_empty_value_deletes_nothing_confined(tmp_path):
    from styxx import ciaudit
    for confined, survives in ((False, False), (True, True)):   # the unconfined control deletes the canary: the step is live
        canary = tmp_path / f"canary-{confined}"
        canary.mkdir()
        (canary / "a").write_text("keep")
        tree = _tree(tmp_path / f"t-{confined}", CANARY_STEP.format(canary=canary))
        rec = ciaudit.audit(str(tree), confined=confined)
        assert rec["confinement"]["confined"] is confined
        assert (canary / "a").exists() is survives, f"confined={confined}"


@landlock
def test_the_audit_reads_the_same_confined(tmp_path):
    from styxx import ciaudit
    for name, text in (("frontier", FRONTIER_FIXTURE), ("lists", LISTS_FIXTURE)):
        tree = _tree(tmp_path / name, text)
        a = ciaudit.audit(str(tree), repair=True, confined=False)
        b = ciaudit.audit(str(tree), repair=True, confined=True)
        for r in (a, b):
            r.pop("seconds")
            r.pop("confinement")
        assert a == b, name


@landlock
def test_the_card_says_it_was_confined(tmp_path, capsys):
    from styxx import ciaudit
    tree = _tree(tmp_path / "t", LISTS_FIXTURE)
    ciaudit.main([str(tree)])
    assert "confined (Landlock ABI" in capsys.readouterr().out


def test_the_cli_refuses_where_it_cannot_confine(tmp_path, monkeypatch, capsys):
    from styxx import ciaudit
    tree = _tree(tmp_path / "t", LISTS_FIXTURE)
    monkeypatch.setattr(C, "abi", lambda: 0)
    monkeypatch.delenv(C.UNCONFINED_ENV, raising=False)
    assert ciaudit.main([str(tree)]) == 2
    err = capsys.readouterr().err
    assert "cannot confine" in err and "--unconfined" in err
    assert ciaudit.main([str(tree), "--unconfined", "--format", "json"]) in (0, 1)
    assert '"confined": false' in capsys.readouterr().out
    monkeypatch.setenv(C.UNCONFINED_ENV, "1")
    assert ciaudit.main([str(tree), "--format", "json"]) in (0, 1)


def test_the_action_refuses_on_a_runner_someone_keeps(monkeypatch):
    from styxx.ciaudit import action as A
    monkeypatch.setattr(C, "abi", lambda: 0)
    monkeypatch.delenv(C.UNCONFINED_ENV, raising=False)
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "self-hosted")
    with pytest.raises(RuntimeError, match="cannot confine"):
        A._confined(lambda: {})
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")      # thrown away after the job
    rec = A._confined(lambda: {"fires": False})
    assert rec["confinement"]["confined"] is False and "GitHub-hosted" in rec["confinement"]["why"]
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "self-hosted")
    monkeypatch.setenv(C.UNCONFINED_ENV, "1")
    assert A._confined(lambda: {})["confinement"]["why"].startswith(C.UNCONFINED_ENV)


def test_an_exception_in_the_confined_part_is_carried_back():
    if C.abi() == 0:
        pytest.skip("this kernel has no Landlock")

    def boom():
        raise ValueError("inside")
    with pytest.raises(C.ConfinedError, match="ValueError: inside"):
        C.run(boom)
