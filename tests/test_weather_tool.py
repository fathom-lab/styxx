# -*- coding: utf-8 -*-
"""tool_weather_report — the MCP weather tool's gate must distinguish three states.

Regression: the tool called styxx.weather(window=N) — a kwarg the engine does not
have (weather() is keyword-only agent_name/window_hours/baseline_days) — so EVERY
invocation raised TypeError, the except swallowed it, and the payload reported
gate:"pass" with no error field. A crashed engine, an empty window and a healthy
fleet were indistinguishable at the gate.
"""
import styxx
from styxx.cognometrics import tool_weather_report


def test_engine_crash_is_error_not_pass(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("corrupt audit entry")
    monkeypatch.setattr(styxx, "weather", boom)
    out = tool_weather_report({})
    assert out["gate"] == "error"
    assert "RuntimeError" in out["error"]


def test_empty_window_is_no_data_not_pass(monkeypatch):
    monkeypatch.setattr(styxx, "weather", lambda **kwargs: None)
    out = tool_weather_report({"window_hours": 6})
    assert out["gate"] == "no_data"
    assert out["window_hours"] == 6.0
    assert "error" not in out


def test_call_uses_the_engines_real_signature(monkeypatch):
    seen = {}

    def fake_weather(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(styxx, "weather", fake_weather)
    tool_weather_report({"window_hours": 12})
    assert seen == {"window_hours": 12.0}
    # deprecated alias: "window" (a count in the old schema) is read as hours
    seen.clear()
    tool_weather_report({"window": 48})
    assert seen == {"window_hours": 48.0}


def test_warning_report_is_not_stamped_pass(monkeypatch):
    class StubReport:
        def as_dict(self):
            return {"condition": "stormy", "warn_rate": 0.3, "gate_pass_rate": 0.7}

    monkeypatch.setattr(styxx, "weather", lambda **kwargs: StubReport())
    out = tool_weather_report({})
    assert out["gate"] == "warn"


def test_calm_report_passes_on_its_own_entries(monkeypatch):
    class StubReport:
        def as_dict(self):
            return {"condition": "clear", "warn_rate": 0.0, "gate_pass_rate": 1.0}

    monkeypatch.setattr(styxx, "weather", lambda **kwargs: StubReport())
    out = tool_weather_report({})
    assert out["gate"] == "pass"
