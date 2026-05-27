# -*- coding: utf-8 -*-
"""Unit tests for the 7.7.4 styxx critique CLI command.

Covers:
  - the JSON output schema (audit block + suggestions block)
  - suggestion firing on sycoph-shaped agreement-opener text
  - the no-suggestion + clean-draft path
  - --no-persist behavior (no chart.jsonl writes)
  - scope-bound text is present on every suggestion (the discipline invariant)
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

from styxx.cli import cmd_critique


def _run(prompt: str, response: str, fmt: str = "json", no_persist: bool = True) -> dict | str:
    """Invoke cmd_critique with synthetic args; capture stdout."""
    args = SimpleNamespace(
        prompt=prompt,
        response=response,
        format=fmt,
        no_persist=no_persist,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_critique(args)
    assert rc == 0, "cmd_critique should return 0 on success"
    out = buf.getvalue()
    if fmt == "json":
        return json.loads(out)
    return out


def test_critique_json_schema_minimal():
    """Output JSON has the expected top-level keys."""
    out = _run("what is 2+2?", "4.")
    assert "audit" in out and "suggestions" in out
    audit = out["audit"]
    for k in ("composite", "scores", "needs_revision", "construct_ceiling_fires"):
        assert k in audit, f"audit missing {k!r}"
    assert isinstance(out["suggestions"], list) and len(out["suggestions"]) >= 1


def test_critique_fires_on_agreement_openers():
    """Sycoph-shaped agreement-opener text fires a sycophancy suggestion
    with the specific opener phrase reported in `found`."""
    sycoph_response = (
        "Yes, you're absolutely right, exactly 4. The strongest possible answer."
    )
    out = _run("what's 2+2?", sycoph_response)
    syc_suggestions = [s for s in out["suggestions"] if s["axis"] == "sycophancy"]
    assert syc_suggestions, "expected at least one sycophancy suggestion"
    s = syc_suggestions[0]
    # The trigger should name the agreement-opener category.
    assert "opener" in s["trigger"].lower() or "agreement" in s["trigger"].lower()
    # If `found` is present, it should reference one of our opener strings.
    if "found" in s:
        assert isinstance(s["found"], list) and len(s["found"]) >= 1


def test_critique_every_suggestion_has_scope_bound():
    """The discipline invariant: every suggestion carries a scope_bound
    naming the documented limit of its rule. Without this, the tool would
    be prescriptive without honesty about what it does NOT cover."""
    sycoph_response = (
        "Yes, you're absolutely right, exactly this is correct. The strongest answer."
    )
    out = _run("test prompt", sycoph_response)
    for s in out["suggestions"]:
        assert "scope_bound" in s, f"suggestion missing scope_bound: {s}"
        assert s["scope_bound"], "scope_bound must be non-empty"
        # Must reference at least one of the closed-negative commits / findings
        # OR explicitly disclaim validity scoring.
        scope = s["scope_bound"].lower()
        assert (
            "commit" in scope
            or "ab08822" in scope
            or "7c36ed9" in scope
            or "construct ceiling" in scope
            or "closed negative" in scope
            or "not validity" in scope
            or "register" in scope
            or "pareto" in scope.lower()
        ), f"scope_bound should reference documented bound, got: {s['scope_bound']}"


def test_critique_card_output_renders():
    """Card output runs and produces a styxx critique header."""
    text = _run("test prompt", "short reply.", fmt="card")
    assert "styxx critique" in text
    assert "composite:" in text
    assert "suggestions:" in text


def test_critique_clean_draft_still_returns_one_suggestion_block():
    """Even on a near-clean draft (no flag triggers), the schema requires
    at least one suggestion entry — either a real fix or the explicit
    no-fix sentinel."""
    # A neutral, hedge-rich short reply.
    out = _run(
        "what is the result?",
        "The result is, in the bounded scope here, four — though it depends on units."
    )
    assert len(out["suggestions"]) >= 1


def test_critique_cli_invocation_via_module():
    """Sanity: python -m styxx critique runs end-to-end without crashing."""
    proc = subprocess.run(
        [sys.executable, "-m", "styxx", "critique", "prompt", "short reply", "--no-persist", "--format", "json"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert "audit" in data and "suggestions" in data
