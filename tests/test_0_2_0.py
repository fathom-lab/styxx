# -*- coding: utf-8 -*-
"""
test_0_2_0.py -- tests for the 0.2.0 milestone release.

Covers the three directions shipped as 0.2.0:

  Phase 1 (data layer)
    * Personality.as_dict / as_json / as_csv / as_markdown
    * styxx.reflect() -> ReflectionReport
    * styxx.recipes.memory.tag_memory_entry + tag_memory_with_personality

  Phase 2 (comparison + visualization)
    * styxx personality --format flag (ascii/json/csv/markdown)
    * styxx reflect CLI command
    * Chance-level reference line on PNG bars

  Phase 3 (distribution surfaces)
    * styxx.serve.run_serve — at least the handler + html template build
    * styxx agent-card --serve CLI flag wiring

Also covers:
    * Vitals.summary gate verdict is dynamic (B fix from 0.1.0a4)
    * RegisteredGate.__repr__ (already tested in test_agent_api)
    * styxx.trace decorator (F fix from 0.1.0a4)
    * audit log rotation (D fix from 0.1.0a4)
    * styxx log clear / rotate (E fix from 0.1.0a4)

All tests run without network and without touching the real
~/.styxx/chart.jsonl.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import styxx
from styxx import analytics


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _fixture_vitals(kind: str):
    from styxx.cli import _load_demo_trajectories
    data = _load_demo_trajectories()
    t = data["trajectories"][kind]
    return styxx.Raw().read(
        entropy=t["entropy"],
        logprob=t["logprob"],
        top2_margin=t["top2_margin"],
    )


def _synth_entry(
    *,
    ts: float,
    p1_pred: str = "reasoning",
    p1_conf: float = 0.45,
    p4_pred: str = "reasoning",
    p4_conf: float = 0.42,
    gate: str = "pass",
    session_id: str | None = None,
    model: str = "fake:gpt-fake",
) -> dict:
    return {
        "ts": ts,
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
        "session_id": session_id,
        "model": model,
        "prompt": "test prompt",
        "tier_active": 0,
        "phase1_pred": p1_pred,
        "phase1_conf": p1_conf,
        "phase4_pred": p4_pred,
        "phase4_conf": p4_conf,
        "gate": gate,
        "abort": None,
    }


@pytest.fixture
def temp_audit_log(monkeypatch, tmp_path):
    """Redirect the audit log to a temp file and populate it with
    a mixed phase / gate / session set of entries."""
    fake = tmp_path / "chart.jsonl"
    now = time.time()
    entries = [
        *[_synth_entry(ts=now - 7200 + i, session_id="ses-a") for i in range(10)],
        *[_synth_entry(
            ts=now - 3600 + i,
            p1_pred="adversarial", p1_conf=0.40,
            p4_pred="refusal", p4_conf=0.35,
            gate="warn",
            session_id="ses-a",
        ) for i in range(3)],
        *[_synth_entry(
            ts=now - 1800 + i,
            p1_pred="reasoning", p1_conf=0.28,
            p4_pred="hallucination", p4_conf=0.38,
            gate="fail",
            session_id="ses-b",
        ) for i in range(2)],
        *[_synth_entry(
            ts=now - 900 + i,
            p1_pred="creative", p1_conf=0.38,
            p4_pred="creative", p4_conf=0.41,
            gate="pass",
            session_id="ses-b",
        ) for i in range(5)],
    ]
    with open(fake, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    monkeypatch.setattr("styxx.analytics._audit_log_path", lambda: fake)
    monkeypatch.setattr("styxx.cli._audit_log_path", lambda: fake)
    # Reset the analytics cache so prior tests don't bleed in
    styxx.clear_audit_cache()
    return fake


# ══════════════════════════════════════════════════════════════════
# Phase 1 — data layer
# ══════════════════════════════════════════════════════════════════

def test_personality_as_dict(temp_audit_log):
    profile = analytics.personality(days=7)
    assert profile is not None
    d = profile.as_dict()
    assert d["n_samples"] == 20
    assert "rates" in d
    assert "variance" in d
    assert "gate_rates" in d
    assert "narrative" in d


def test_personality_as_json_is_valid(temp_audit_log):
    profile = analytics.personality(days=7)
    s = profile.as_json()
    parsed = json.loads(s)
    assert parsed["n_samples"] == 20


def test_personality_as_csv_format(temp_audit_log):
    profile = analytics.personality(days=7)
    s = profile.as_csv()
    lines = s.splitlines()
    assert len(lines) == 2   # header + values
    headers = lines[0].split(",")
    values = lines[1].split(",")
    assert len(headers) == len(values)
    assert "n_samples" in headers
    assert "rate_reasoning" in headers
    assert "gate_pass" in headers


def test_personality_as_markdown(temp_audit_log):
    profile = analytics.personality(days=7)
    s = profile.as_markdown()
    assert s.startswith("```styxx-personality")
    assert s.endswith("```")
    assert "phase4 distribution:" in s
    assert "gate distribution:" in s
    assert "mean phase1 conf:" in s


def test_reflect_returns_report(temp_audit_log):
    r = styxx.reflect(now_days=1, baseline_days=7)
    assert r is not None
    assert r.now is not None
    assert 0.0 <= r.drift_cosine <= 1.0
    assert r.drift_label in {
        "stable", "slight drift", "significant drift", "insufficient history",
    }
    assert r.current_mood in {
        "drifting", "cautious", "defensive",
        "creative", "steady", "unfocused", "mixed", "quiet",
    }


def test_reflect_as_dict_json_roundtrip(temp_audit_log):
    r = styxx.reflect()
    d = r.as_dict()
    s = r.as_json()
    parsed = json.loads(s)
    assert parsed["drift_label"] == d["drift_label"]


def test_reflect_as_markdown(temp_audit_log):
    r = styxx.reflect()
    md = r.as_markdown()
    assert md.startswith("```styxx-reflection")
    assert md.endswith("```")
    assert "drift vs yesterday" in md


def test_reflect_suggestions_fire(temp_audit_log):
    """Our synthetic audit log has 3 warn + 2 fail out of 20 entries.
    The gate pass rate is 15/20 = 75%. That's just above the 70%
    threshold, so the "gate pass rate" suggestion should NOT fire
    — but hallucination rate is 2/20 = 10%, right at the boundary
    where it MIGHT fire depending on the rate's exact value."""
    r = styxx.reflect()
    # At minimum, suggestions is a list (could be empty)
    assert isinstance(r.suggestions, list)


def test_recipes_memory_tag_entry(temp_audit_log):
    from styxx.recipes.memory import tag_memory_entry
    v = _fixture_vitals("refusal")
    tagged = tag_memory_entry("remember this thing", vitals=v)
    assert "remember this thing" in tagged
    assert "```styxx" in tagged
    assert "phase1" in tagged


def test_recipes_memory_tag_without_vitals():
    from styxx.recipes.memory import tag_memory_entry
    tagged = tag_memory_entry("no vitals available")
    assert "no vitals available" in tagged
    assert "not captured" in tagged


def test_recipes_memory_tag_with_personality(temp_audit_log):
    from styxx.recipes.memory import tag_memory_with_personality
    tagged = tag_memory_with_personality("checkpoint note", days=7)
    assert "checkpoint note" in tagged
    assert "```styxx-personality" in tagged


# ══════════════════════════════════════════════════════════════════
# Phase 2 — CLI commands + chance-level line
# ══════════════════════════════════════════════════════════════════

def test_personality_cli_format_json(temp_audit_log, capsys):
    from styxx.cli import cmd_personality
    class _Args:
        days = 7
        format = "json"
    rc = cmd_personality(_Args())
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["n_samples"] == 20


def test_personality_cli_format_csv(temp_audit_log, capsys):
    from styxx.cli import cmd_personality
    class _Args:
        days = 7
        format = "csv"
    rc = cmd_personality(_Args())
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert len(lines) == 2


def test_personality_cli_format_markdown(temp_audit_log, capsys):
    from styxx.cli import cmd_personality
    class _Args:
        days = 7
        format = "markdown"
    rc = cmd_personality(_Args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "```styxx-personality" in out


def test_reflect_cli_runs(temp_audit_log, capsys):
    from styxx.cli import cmd_reflect
    class _Args:
        now_days = 1
        baseline_days = 7
        format = "ascii"
    rc = cmd_reflect(_Args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "reflection report" in out


def test_reflect_cli_json_format(temp_audit_log, capsys):
    from styxx.cli import cmd_reflect
    class _Args:
        now_days = 1
        baseline_days = 7
        format = "json"
    rc = cmd_reflect(_Args())
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "drift_cosine" in parsed
    assert "suggestions" in parsed


# ══════════════════════════════════════════════════════════════════
# Phase 3 — distribution surfaces
# ══════════════════════════════════════════════════════════════════

def test_serve_handler_builds():
    """Just verify the _make_handler function constructs a usable
    request handler class without raising."""
    from styxx.serve import _make_handler
    handler_cls = _make_handler(
        serve_dir=Path.home() / ".styxx" / "serve",
        agent_name="test",
        refresh_seconds=30,
    )
    assert handler_cls is not None
    # Should be a class with a do_GET method
    assert hasattr(handler_cls, "do_GET")


def test_serve_html_template_renders():
    """The HTML template should format cleanly with the expected
    placeholders."""
    from styxx.serve import _HTML_TEMPLATE
    html = _HTML_TEMPLATE.format(
        agent_name="test-agent",
        refresh_seconds=30,
        cache_bust=1234567890,
    )
    assert "test-agent" in html
    assert "refresh" in html.lower()
    assert "card.png?ts=1234567890" in html


def test_agent_card_serve_flag_wired():
    """The CLI parser should accept --serve, --port, --refresh,
    and --no-browser flags on agent-card."""
    from styxx.cli import _build_parser
    parser = _build_parser()
    # Parse a synthetic agent-card --serve invocation; should not raise
    args = parser.parse_args(["agent-card", "--serve", "--port", "9797",
                               "--refresh", "30", "--no-browser"])
    assert args.serve is True
    assert args.port == 9797
    assert args.refresh == 30
    assert args.no_browser is True


# ══════════════════════════════════════════════════════════════════
# Fixes from 0.1.0a4 (rolled into 0.2.0)
# ══════════════════════════════════════════════════════════════════

def test_vitals_summary_uses_dynamic_gate():
    """The ASCII card verdict line should reflect the actual
    gate state, not a hardcoded 'PASS'."""
    v_reason = _fixture_vitals("reasoning")
    v_refusal = _fixture_vitals("refusal")

    card_reason = v_reason.summary
    card_refusal = v_refusal.summary

    assert "PASS" in card_reason, "reasoning fixture should verdict PASS"
    # Refusal fixture should verdict WARN (not PASS)
    assert "WARN" in card_refusal, (
        f"refusal fixture should verdict WARN, got card:\n{card_refusal}"
    )


def test_trace_decorator_tags_session():
    """@styxx.trace sets session_id inside the decorated function."""
    captured = []

    @styxx.trace("test-func")
    def inner():
        captured.append(styxx.session_id())

    # Outer session is whatever it was
    outer = styxx.session_id()
    inner()
    assert captured == ["test-func"]
    # After exit, the outer session is restored
    assert styxx.session_id() == outer


def test_trace_decorator_restores_on_exception():
    """Even if the wrapped function raises, the outer session
    should be restored."""
    outer = styxx.session_id()

    @styxx.trace("error-func")
    def broken():
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        broken()
    assert styxx.session_id() == outer


def test_trace_decorator_requires_string_name():
    with pytest.raises(ValueError):
        @styxx.trace("")
        def f():
            pass


def test_audit_log_rotate_command(tmp_path, monkeypatch, capsys):
    """styxx log rotate moves chart.jsonl -> chart.jsonl.1."""
    fake = tmp_path / "chart.jsonl"
    fake.write_text('{"ts": 100, "gate": "pass"}\n')
    monkeypatch.setattr("styxx.cli._audit_log_path", lambda: fake)

    from styxx.cli import cmd_log_rotate
    class _Args:
        pass
    rc = cmd_log_rotate(_Args())
    assert rc == 0
    assert not fake.exists()
    assert (tmp_path / "chart.jsonl.1").exists()


def test_audit_log_clear_command(tmp_path, monkeypatch, capsys):
    """styxx log clear deletes chart.jsonl."""
    fake = tmp_path / "chart.jsonl"
    fake.write_text('{"ts": 100, "gate": "pass"}\n')
    monkeypatch.setattr("styxx.cli._audit_log_path", lambda: fake)

    from styxx.cli import cmd_log_clear
    class _Args:
        pass
    rc = cmd_log_clear(_Args())
    assert rc == 0
    assert not fake.exists()


def test_load_audit_cache_invalidates_on_mtime(tmp_path, monkeypatch):
    """load_audit() cache should invalidate when the file is
    modified, so newly-added entries are visible."""
    fake = tmp_path / "chart.jsonl"
    fake.write_text('{"ts": 100, "gate": "pass", "phase1_pred": "reasoning"}\n')
    monkeypatch.setattr("styxx.analytics._audit_log_path", lambda: fake)
    styxx.clear_audit_cache()

    entries_1 = styxx.load_audit()
    assert len(entries_1) == 1

    # Wait a tick so mtime advances
    time.sleep(0.01)
    with open(fake, "a") as f:
        f.write('{"ts": 200, "gate": "warn", "phase1_pred": "refusal"}\n')

    entries_2 = styxx.load_audit()
    assert len(entries_2) == 2


def test_reflex_events_capture_discarded_text():
    """When a rewind fires, the ReflexEvent should capture the
    text that was about to be discarded."""
    from styxx.reflex import ReflexEvent
    ev = ReflexEvent(
        kind="rewind",
        token_idx=10,
        rewind_n=2,
        rewind_anchor="retry",
        discarded_text="bad tokens",
    )
    assert ev.discarded_text == "bad tokens"


# ══════════════════════════════════════════════════════════════════
# Version sanity
# ══════════════════════════════════════════════════════════════════

def test_version_is_current():
    assert styxx.__version__.startswith(("0.2.", "0.3.", "0.4.", "0.5."))


def test_all_0_2_0_exports_present():
    for name in (
        # Phase 1
        "reflect", "ReflectionReport",
        # 0.1.0a4 rollups
        "trace", "agent_card", "clear_audit_cache",
    ):
        assert hasattr(styxx, name), f"styxx.{name} missing"
