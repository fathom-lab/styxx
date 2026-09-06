"""The corpus's own receipts appeal only to history it has, or say in writing why they do not.

`papers/sworn/receipt_provenance_audit.py` is the guard. This runs it, and then checks the guard
itself can fail -- a declarations file is a place to hide things, so the ways it can go wrong are
worth pinning as hard as the happy path.

The audit runs in two worlds and this test asserts a different thing in each:

  full history (a developer's clone)  -- every tree-claiming receipt must be backed or declared
  shallow history (CI's default)      -- the audit must ACCUSE NOBODY and say it cannot answer

The second is the one worth having. `actions/checkout` clones at depth 1, so in CI nearly every
commit is missing; an audit that read absence as fabrication would report the entire corpus as
fabricated on its first CI run. This project has already shipped an accusation at 0.23 precision by
making exactly that inference, so the silence is tested, not assumed.

The accusation is matched as the block header rather than the bare word: the summary table prints
"UNDECLARED" on every run, including runs where the count is zero. Matching the word alone made this
file's own shallow-clone test fail the first time it ran, which is the small version of the mistake
it exists to catch.

Watched to fail before it was kept: with no declarations file and full history, the audit reports
the five synthetic dry-run receipts as UNDECLARED and exits 1; with a fabricated receipt staged into
the tracked set, it names that receipt too.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "papers" / "sworn" / "receipt_provenance_audit.py"

ACCUSATION = "\nUNDECLARED:"          # the block header, not the summary label

if str(AUDIT.parent) not in sys.path:
    sys.path.insert(0, str(AUDIT.parent))


@pytest.fixture()
def audit():
    if not AUDIT.exists():
        pytest.skip("the audit is not present in this checkout")
    import importlib
    mod = importlib.import_module("receipt_provenance_audit")
    return importlib.reload(mod)


@pytest.fixture()
def full_history(audit, monkeypatch):
    """The accusation path only exists when history is complete; assert it wherever we run."""
    monkeypatch.setattr(audit, "_is_shallow", lambda: False)
    return audit


def _real_declarations(audit) -> dict:
    return json.loads(audit.DECLARATIONS.read_text(encoding="utf-8"))


def test_every_tree_claiming_receipt_is_backed_by_history_or_declared(audit, capsys):
    rc = audit.main([])
    out = capsys.readouterr().out
    if audit._is_shallow():
        assert rc == audit.EXIT_INDETERMINATE, out
        assert ACCUSATION not in out, "a shallow clone accused someone: %s" % out
    else:
        assert rc == audit.EXIT_OK, out


def test_a_shallow_clone_accuses_nobody(audit, monkeypatch, capsys):
    """The central property. Absence of a commit is not evidence when nothing is present."""
    monkeypatch.setattr(audit, "_is_shallow", lambda: True)
    monkeypatch.setattr(audit, "DECLARATIONS", Path("does-not-exist.json"))
    rc = audit.main([])
    out = capsys.readouterr().out
    assert rc == audit.EXIT_INDETERMINATE
    assert ACCUSATION not in out
    assert "INDETERMINATE" in out and "not evidence of a fabricated one" in out


def test_it_walks_nested_receipts_not_only_the_top_level(audit):
    """The first version saw 51 of 2118 receipt-shaped objects and would have called that all."""
    r = {"spans": [], "document_verdict": "UNSWORN"}
    doc = {"runs": [{"result": r}, {"result": {"nope": 1}}], "top": r}
    found = dict(audit._receipts_in(doc))
    assert set(found) == {"$.runs[0].result", "$.top"}, found


def test_the_conformance_vector_exclusion_is_stated_not_silent(audit, capsys):
    """Coverage that holds by accident is not coverage: say what is skipped, and what it costs."""
    audit.main([])
    out = capsys.readouterr().out
    assert "conformance vectors, not audited" in out
    assert audit.VECTORS_PREFIX in out
    assert "make a tree claim" in out


def test_it_runs_as_a_command(audit):
    """CLI output is behaviour: the audit must work when invoked the way its docstring says."""
    p = subprocess.run([sys.executable, str(AUDIT)], capture_output=True, cwd=str(ROOT))
    out = p.stdout.decode("utf-8", "replace")
    assert p.returncode in (audit.EXIT_OK, audit.EXIT_INDETERMINATE), out
    assert "receipts appealing to the tree channel" in out


def test_the_output_is_ascii_so_it_survives_a_windows_console():
    """An em-dash in this audit's output rendered as a replacement character on cp1252 once."""
    p = subprocess.run([sys.executable, str(AUDIT), "--list"], capture_output=True, cwd=str(ROOT))
    p.stdout.decode("ascii")           # raises UnicodeDecodeError if anything non-ASCII is printed


def test_dropping_a_declaration_makes_the_audit_fail(full_history, tmp_path, monkeypatch, capsys):
    """The declarations are load-bearing: remove one and the receipt it covered is undeclared."""
    d = _real_declarations(full_history)
    assert d["declared"], "there is nothing to drop; this test would pass vacuously"
    f = tmp_path / "decl.json"
    f.write_text(json.dumps(dict(d, declared=d["declared"][1:])), encoding="utf-8")
    monkeypatch.setattr(full_history, "DECLARATIONS", f)
    assert full_history.main([]) == full_history.EXIT_UNDECLARED
    assert ACCUSATION in capsys.readouterr().out


def test_no_declarations_at_all_makes_the_audit_fail(full_history, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(full_history, "DECLARATIONS", tmp_path / "absent.json")
    assert full_history.main([]) == full_history.EXIT_UNDECLARED
    out = capsys.readouterr().out
    assert ACCUSATION in out and "which is not a commit in this repository" in out


def test_a_declaration_without_a_reason_is_refused(audit, tmp_path, monkeypatch):
    """A declaration exists to record WHY. One with no reason is a silencer, and is refused."""
    e = dict(_real_declarations(audit)["declared"][0])
    e.pop("reason")
    f = tmp_path / "decl.json"
    f.write_text(json.dumps({"declared": [e]}), encoding="utf-8")
    monkeypatch.setattr(audit, "DECLARATIONS", f)
    with pytest.raises(SystemExit) as ei:
        audit.main([])
    assert "reason" in str(ei.value)


def test_a_declaration_for_a_receipt_that_is_now_backed_is_reported_stale(full_history, tmp_path,
                                                                         monkeypatch, capsys):
    """Declarations must not outlive their cause, or the file becomes a standing exemption list."""
    backed = None
    for rel in full_history._tracked_json():
        try:
            obj = json.loads((full_history.ROOT / rel).read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            continue
        if (full_history._is_receipt(obj) and full_history._tree_claim(obj)
                and full_history._commit_exists(obj.get("commit"))):
            backed = (rel.replace("\\", "/"), obj["commit"])
            break
    if backed is None:
        pytest.skip("no backed tree-claiming receipt here to build the case from (shallow clone?)")

    d = _real_declarations(full_history)
    f = tmp_path / "decl.json"
    f.write_text(json.dumps(dict(d, declared=d["declared"] + [
        {"path": backed[0], "commit": backed[1], "kind": "synthetic",
         "reason": "planted by a test"}])), encoding="utf-8")
    monkeypatch.setattr(full_history, "DECLARATIONS", f)
    assert full_history.main([]) == full_history.EXIT_UNDECLARED
    assert "STALE DECLARATION" in capsys.readouterr().out
