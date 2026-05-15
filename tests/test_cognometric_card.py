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
    result = render_card(data, out, register=False)
    assert result == out
    assert out.exists()
    # PNG magic bytes
    head = out.read_bytes()[:8]
    assert head == b"\x89PNG\r\n\x1a\n"
    # non-trivial size — should be tens of KB for a 1200x630
    assert out.stat().st_size > 10_000


# ── from_single_audit + registry + heal-pair ──────────────────────
def test_from_single_audit_wraps_audit_dict():
    audit = {"sycophancy": 0.10, "deception": 0.20,
             "overconfidence": 0.30, "refusal": 0.40,
             "composite": 0.25}
    d = CardData.from_single_audit(audit, agent="m", ts="2026-05-12")
    assert d.agent == "m"
    assert d.n_turns == 1
    assert d.ts == "2026-05-12"
    assert d.composite_mean == 0.25
    for ax in AXES:
        assert d.means[ax] == audit[ax]
        assert d.series[ax] == [audit[ax]]


def test_from_single_audit_composite_fallback():
    audit = {"sycophancy": 0.1, "deception": 0.2,
             "overconfidence": 0.3, "refusal": 0.4}  # NO composite
    d = CardData.from_single_audit(audit, agent="m")
    assert abs(d.composite_mean - 0.25) < 1e-6


def test_register_card_appends_jsonl(tmp_path: Path, monkeypatch):
    """register_card writes one JSON record per call to cards.jsonl."""
    from styxx import cognometric_card as cc
    monkeypatch.setattr(cc, "_registry_dir", lambda: tmp_path)

    cc.register_card(tmp_path / "a.png",
                     serial="STX-1111", agent="m", ts="2026-05-12",
                     composite=0.34, band="stable", variant="single")
    cc.register_card(tmp_path / "b.png",
                     serial="STX-2222", agent="m", ts="2026-05-12",
                     composite=0.46, band="stable", variant="heal-pair",
                     extra={"recovery_pct": 28.3})

    records = cc.list_cards(limit=10)
    assert len(records) == 2
    assert records[0]["serial"] == "STX-1111"
    assert records[1]["serial"] == "STX-2222"
    assert records[1]["variant"] == "heal-pair"
    assert records[1]["extra"]["recovery_pct"] == 28.3


def test_render_card_auto_registers(tmp_path: Path, monkeypatch):
    """render_card with register=True (default) appends to the local log."""
    pytest.importorskip("matplotlib")
    from styxx import cognometric_card as cc
    monkeypatch.setattr(cc, "_registry_dir", lambda: tmp_path)

    audit = {"sycophancy": 0.1, "deception": 0.2,
             "overconfidence": 0.3, "refusal": 0.4, "composite": 0.25}
    d = cc.CardData.from_single_audit(audit, agent="m", ts="2026-05-12")
    cc.render_card(d, tmp_path / "a.png")  # register=True default

    records = cc.list_cards(limit=10)
    assert len(records) == 1
    assert records[0]["agent"] == "m"
    assert records[0]["variant"] == "single"


def test_render_heal_card_produces_png(tmp_path: Path, monkeypatch):
    pytest.importorskip("matplotlib")
    from styxx import cognometric_card as cc
    monkeypatch.setattr(cc, "_registry_dir", lambda: tmp_path)

    baseline_audit = {"sycophancy": 0.09, "deception": 0.93,
                      "overconfidence": 0.90, "refusal": 0.05,
                      "composite": 0.64}
    healed_audit   = {"sycophancy": 0.15, "deception": 0.64,
                      "overconfidence": 0.58, "refusal": 0.11,
                      "composite": 0.46}
    b = cc.CardData.from_single_audit(baseline_audit, agent="gpt-4o-mini",
                                       ts="2026-05-12")
    h = cc.CardData.from_single_audit(healed_audit,   agent="gpt-4o-mini",
                                       ts="2026-05-12", healed=True)

    out = tmp_path / "heal.png"
    result = cc.render_heal_card(b, h, out)
    assert result == out
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 10_000

    # registry: heal-pair record written with extra carrying delta + recovery
    records = cc.list_cards(limit=10)
    heal_records = [r for r in records if r["variant"] == "heal-pair"]
    assert len(heal_records) == 1
    extra = heal_records[0]["extra"]
    assert extra["baseline_composite"] == 0.64
    assert extra["healed_composite"] == 0.46
    assert abs(extra["recovery_pct"] - 28.1) < 0.5


def test_heal_result_card_methods(tmp_path: Path, monkeypatch):
    """HealResult.{baseline_card, healed_card, heal_card} all produce PNGs."""
    pytest.importorskip("matplotlib")
    from styxx import cognometric_card as cc
    from styxx.reflex import HealResult
    monkeypatch.setattr(cc, "_registry_dir", lambda: tmp_path)

    r = HealResult(
        text="healed text",
        audit_baseline={"sycophancy": 0.1, "deception": 0.9,
                         "overconfidence": 0.9, "refusal": 0.05,
                         "composite": 0.62},
        audit_final   ={"sycophancy": 0.15, "deception": 0.6,
                         "overconfidence": 0.55, "refusal": 0.10,
                         "composite": 0.44},
        n_audits=2,
        recovered=0.18,
        recovery_pct=29.0,
    )
    p_b = Path(r.baseline_card(tmp_path / "b.png",
                                agent="m", ts="2026-05-12"))
    p_h = Path(r.healed_card  (tmp_path / "h.png",
                                agent="m", ts="2026-05-12"))
    p_pair = Path(r.heal_card  (tmp_path / "pair.png",
                                agent="m", ts="2026-05-12"))
    for p in (p_b, p_h, p_pair):
        assert p.exists()
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_mcp_cogn_share_card_single(tmp_path: Path):
    """MCP tool produces a single card when given an audit dict."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("mcp")
    from styxx.mcp.server import tool_cogn_share_card
    res = tool_cogn_share_card({
        "agent": "m",
        "variant": "single",
        "audit": {"sycophancy": 0.1, "deception": 0.2,
                   "overconfidence": 0.3, "refusal": 0.4,
                   "composite": 0.25},
        "out_dir": str(tmp_path),
    })
    assert "registry_id" in res
    assert res["registry_id"].startswith("STX-")
    assert res["variant"] == "single"
    assert res["composite"] == 0.25
    assert res["band"] == "pristine"
    assert Path(res["card_path"]).exists()


def test_mcp_cogn_share_card_heal(tmp_path: Path):
    """MCP tool produces a paired heal card when given baseline + healed audits."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("mcp")
    from styxx.mcp.server import tool_cogn_share_card
    res = tool_cogn_share_card({
        "agent": "gpt-4o-mini",
        "variant": "heal",
        "baseline_audit": {"sycophancy": 0.09, "deception": 0.93,
                            "overconfidence": 0.90, "refusal": 0.05,
                            "composite": 0.64},
        "healed_audit":   {"sycophancy": 0.15, "deception": 0.64,
                            "overconfidence": 0.58, "refusal": 0.11,
                            "composite": 0.46},
        "out_dir": str(tmp_path),
    })
    assert res["variant"] == "heal"
    assert res["baseline_composite"] == 0.64
    assert res["healed_composite"] == 0.46
    assert abs(res["recovery_pct"] - 28.1) < 0.5
    assert Path(res["card_path"]).exists()


def test_mcp_cogn_share_card_heal_validates_inputs():
    """variant='heal' without both audits returns a clean error."""
    pytest.importorskip("mcp")
    from styxx.mcp.server import tool_cogn_share_card
    res = tool_cogn_share_card({"agent": "m", "variant": "heal"})
    assert "error" in res
    assert "baseline_audit" in res["error"]
