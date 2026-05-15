# -*- coding: utf-8 -*-
"""
test_cognometric_card.py — the 7.4.x luxury registry card.

Verifies:
  * CardData.from_audit_json reads all four supported JSON shapes
    (rows[].audit, rows[].{baseline_audit, healed_audit}, rows[].scores,
     results[].{baseline,reflex}.scores).
  * Composite fallback (mean of axes) when "composite" is absent.
  * _band() categorisation is correct at every threshold.
  * Serial number is deterministic for (agent, ts).
  * render_card produces a non-trivial PNG when matplotlib is installed
    (skipped otherwise).

Network: none. Tmp files only.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from styxx.cognometric_card import (
    AXES,
    CardData,
    _band,
    _serial_number,
)


# ── fixtures ───────────────────────────────────────────────────────
AXIS_VALUES = {
    "sycophancy":     0.10,
    "deception":      0.20,
    "overconfidence": 0.30,
    "refusal":        0.40,
}
# mean of axis values = 0.25 — composite fallback target


def _audit(values: dict) -> dict:
    return {**values, "composite": sum(values.values()) / len(values)}


def _shape_audit(values: dict) -> dict:
    """rows[].audit shape — self_claude_dogfood format."""
    return {"ts": "2026-05-12", "model": "test-agent", "rows": [
        {"id": "t1", "audit": _audit(values)},
        {"id": "t2", "audit": _audit({k: v + 0.05 for k, v in values.items()})},
    ]}


def _shape_inversion(values: dict) -> dict:
    """rows[].{baseline_audit, healed_audit} shape."""
    return {"ts": "2026-05-12", "rows": [{
        "id": "t1",
        "baseline_audit": _audit(values),
        "healed_audit":   _audit({k: v * 0.5 for k, v in values.items()}),
    }]}


def _shape_scores(values: dict) -> dict:
    """rows[].scores shape — horizon_scaling format (no composite)."""
    return {"ts": "2026-05-12", "model": "test-agent", "rows": [
        {"turn": 1, "scores": values},
        {"turn": 2, "scores": {k: v + 0.05 for k, v in values.items()}},
    ]}


def _shape_reflex_loop(values: dict) -> dict:
    """results[].{baseline,reflex}.scores shape — reflex_loop format."""
    return {"ts": "2026-05-12", "model": "test-agent", "results": [{
        "id": "syc_01",
        "baseline": {"text": "...", "scores": values},
        "reflex":   {"text": "...", "scores": {k: v * 0.5 for k, v in values.items()}},
    }]}


# ── tests ──────────────────────────────────────────────────────────
def test_audit_shape_loads(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_shape_audit(AXIS_VALUES)), encoding="utf-8")
    data = CardData.from_audit_json(p, agent="test-agent")
    assert data.agent == "test-agent"
    assert data.n_turns == 2
    for ax in AXES:
        assert ax in data.means
    assert 0.20 < data.composite_mean < 0.40


def test_inversion_shape_baseline_and_healed(tmp_path: Path):
    p = tmp_path / "inv.json"
    p.write_text(json.dumps(_shape_inversion(AXIS_VALUES)), encoding="utf-8")

    baseline = CardData.from_audit_json(p, agent="m", healed=False)
    healed   = CardData.from_audit_json(p, agent="m", healed=True)

    assert baseline.composite_mean > healed.composite_mean
    assert baseline.healed is False
    assert healed.healed is True


def test_scores_shape_composite_fallback(tmp_path: Path):
    p = tmp_path / "scores.json"
    p.write_text(json.dumps(_shape_scores(AXIS_VALUES)), encoding="utf-8")
    data = CardData.from_audit_json(p)
    # composite should be the mean of 4 axes
    assert abs(data.composite_series[0] - 0.25) < 1e-6


def test_reflex_loop_shape(tmp_path: Path):
    p = tmp_path / "reflex.json"
    p.write_text(json.dumps(_shape_reflex_loop(AXIS_VALUES)), encoding="utf-8")
    baseline = CardData.from_audit_json(p, healed=False)
    healed   = CardData.from_audit_json(p, healed=True)
    assert baseline.composite_mean > healed.composite_mean


def test_band_thresholds():
    # < 0.30 — pristine
    assert _band(0.00)[0] == "pristine"
    assert _band(0.29)[0] == "pristine"
    # < 0.50 — stable
    assert _band(0.30)[0] == "stable"
    assert _band(0.49)[0] == "stable"
    # < 0.75 — elevated
    assert _band(0.50)[0] == "elevated"
    assert _band(0.74)[0] == "elevated"
    # >= 0.75 — critical
    assert _band(0.75)[0] == "critical"
    assert _band(1.00)[0] == "critical"


def test_serial_number_deterministic():
    s1 = _serial_number("claude-opus-4-7", "2026-05-11")
    s2 = _serial_number("claude-opus-4-7", "2026-05-11")
    assert s1 == s2
    assert s1.startswith("STX-")
    assert len(s1) == 8
    # different agent ⇒ different serial
    assert _serial_number("gpt-5-mini", "2026-05-11") != s1


def test_above_threshold_count(tmp_path: Path):
    rows = [
        {"id": "t1", "audit": {"sycophancy": 0.1, "deception": 0.1,
                                 "overconfidence": 0.1, "refusal": 0.1,
                                 "composite": 0.10}},
        {"id": "t2", "audit": {"sycophancy": 0.5, "deception": 0.5,
                                 "overconfidence": 0.5, "refusal": 0.5,
                                 "composite": 0.55}},
        {"id": "t3", "audit": {"sycophancy": 0.8, "deception": 0.8,
                                 "overconfidence": 0.8, "refusal": 0.8,
                                 "composite": 0.80}},
    ]
    p = tmp_path / "thresh.json"
    p.write_text(json.dumps({"ts": "2026-05-12", "rows": rows}), encoding="utf-8")
    data = CardData.from_audit_json(p, agent="m")
    assert data.above_threshold == 2  # t2 and t3 are ≥ 0.5


def test_empty_audits_raises(tmp_path: Path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"ts": "2026-05-12", "rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no audits"):
        CardData.from_audit_json(p, agent="m")


def test_render_card_writes_png(tmp_path: Path):
    """Smoke-test: render to disk and verify a non-trivial PNG exists.
    Skipped if matplotlib isn't installed."""
    pytest.importorskip("matplotlib")
    from styxx.cognometric_card import render_card

    p = tmp_path / "a.json"
    p.write_text(json.dumps(_shape_audit(AXIS_VALUES)), encoding="utf-8")
    data = CardData.from_audit_json(p, agent="test-agent")

    out = tmp_path / "card.png"
    result = render_card(data, out)
    assert result == out
    assert out.exists()
    # PNG magic bytes
    head = out.read_bytes()[:8]
    assert head == b"\x89PNG\r\n\x1a\n"
    # non-trivial size — should be tens of KB for a 1200x630
    assert out.stat().st_size > 10_000
