"""styxx.undeclared v0.1 — the two-author reconciliation.

The load-bearing tests are the ones that keep it from becoming an accuser: it
must carry no verdict, it must never call UNATTRIBUTED concealment, and it must
parse the diff with the gate's own parser rather than a copy that can drift.
"""
from __future__ import annotations

import json

import pytest

from styxx.undeclared import SPEC, reconcile

DIFF = (
    "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@\n+x=1\n"
    "diff --git a/package-lock.json b/package-lock.json\n"
    "--- a/package-lock.json\n+++ b/package-lock.json\n@@\n+dep\n"
    "diff --git a/src/new.py b/src/new.py\nnew file mode 100644\n"
    "--- /dev/null\n+++ b/src/new.py\n@@\n+y=2\n"
)


def wl(*paths):
    return {"spec": "styxx-worklog/v0.1", "session": "s", "harness": "h",
            "entries": [{"seq": i + 1, "path": p, "tool": "edit",
                         "at": "2026-09-01T00:00:00Z",
                         "after_sha256": "0" * 64, "before_sha256": None}
                        for i, p in enumerate(paths)]}


def test_the_agent_wrote_some_of_it():
    r = reconcile(wl("src/app.py", "src/new.py"), DIFF)
    assert r["attributed"] == ["src/app.py", "src/new.py"]
    assert r["unattributed"] == ["package-lock.json"]


def test_lockfile_written_by_a_package_manager_is_unattributed_not_concealed():
    r = reconcile(wl("src/app.py", "src/new.py"), DIFF)
    assert "package-lock.json" in r["unattributed"]
    assert "conceal" in r["boundary"]          # the word appears only as a denial
    assert "does NOT mean concealment" in r["boundary"]


def test_written_then_reverted_is_reported_not_accused():
    r = reconcile(wl("src/app.py", "scratch/tmp.py"), DIFF)
    assert r["recorded_not_in_diff"] == ["scratch/tmp.py"]
    assert "scratch/tmp.py" not in r["unattributed"]


def test_empty_worklog_attributes_nothing_and_still_refuses_a_verdict():
    r = reconcile(wl(), DIFF)
    assert r["attributed"] == []
    assert len(r["unattributed"]) == 3
    assert r["verdict"] == "UNGATED"


def test_path_separators_are_normalised_both_sides():
    r = reconcile(wl("src\\app.py"), DIFF)
    assert "src/app.py" in r["attributed"]


# ------------------------------------------------- the contract, as assertions

@pytest.mark.parametrize("banned", ["CONTRADICTED", "VERIFIED", "FAIL", "PASS",
                                    "LIE", "concealment "])
def test_no_verdict_language_anywhere_in_the_report(banned):
    """If this fails, someone gave a record an opinion."""
    r = reconcile(wl("src/app.py"), DIFF)
    blob = json.dumps({k: v for k, v in r.items() if k != "boundary"})
    assert banned not in blob


def test_it_says_what_it_has_not_measured():
    r = reconcile(wl("src/app.py"), DIFF)
    assert "never been measured" in r["not_measured"]


def test_it_uses_the_gates_own_parser_not_a_copy():
    """A second diff parser would drift from the first. The correction on
    2026-08-31 was caused by exactly that kind of disagreement."""
    import inspect

    import styxx.undeclared as U
    src = inspect.getsource(U.reconcile)
    assert "from styxx.diffgate import parse_unified_diff" in src
    assert "@@" not in src, "reconcile appears to parse diff text itself"


def test_spec_pinned():
    assert SPEC == "styxx-undeclared/v0.1"
