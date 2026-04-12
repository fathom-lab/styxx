# -*- coding: utf-8 -*-
"""
test_power_ups.py -- tests for the 0.1.0a3 power-up layer.

Covers:
  * styxx doctor          diagnostic health check
  * styxx.hook_openai()   global monkey-patch (and unhook)
  * styxx.explain()       prose interpretation
  * Vitals.as_markdown()  markdown render
  * styxx.set_session()   programmatic session tagging
  * styxx.load_audit()    audit log reader
  * styxx.log_stats()     aggregation
  * styxx.log_timeline()  ASCII timeline
  * styxx.streak()        consecutive-attractor tracking
  * styxx.mood()          one-word aggregate
  * styxx.fingerprint()   cognitive identity signature
  * styxx.personality()   full personality profile
  * styxx.dreamer()       retroactive reflex tuning

All tests run on a synthetic audit log written into a temp dir so we
never touch the real ~/.styxx/chart.jsonl during CI.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import styxx
from styxx import analytics, doctor, explain, hooks
from styxx.gates import dispatch_gates


# ══════════════════════════════════════════════════════════════════
# Synthetic audit log fixture
# ══════════════════════════════════════════════════════════════════

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
    a synthetic mix of phase / gate / session entries."""
    fake = tmp_path / "chart.jsonl"
    now = time.time()

    entries = [
        # 10 reasoning (steady state)
        *[_synth_entry(ts=now - 7200 + i, session_id="ses-a")
          for i in range(10)],
        # 3 refusal (defensive)
        *[_synth_entry(
            ts=now - 3600 + i,
            p1_pred="adversarial", p1_conf=0.40,
            p4_pred="refusal", p4_conf=0.35,
            gate="warn",
            session_id="ses-a",
        ) for i in range(3)],
        # 2 hallucination (drift)
        *[_synth_entry(
            ts=now - 1800 + i,
            p1_pred="reasoning", p1_conf=0.28,
            p4_pred="hallucination", p4_conf=0.38,
            gate="fail",
            session_id="ses-b",
        ) for i in range(2)],
        # 5 creative (different session)
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

    # Patch the audit log path resolver used by both cli and analytics
    monkeypatch.setattr("styxx.analytics._audit_log_path", lambda: fake)
    monkeypatch.setattr("styxx.cli._audit_log_path", lambda: fake)
    monkeypatch.setattr("styxx.doctor.Path", Path)
    return fake


# ══════════════════════════════════════════════════════════════════
# doctor (P1)
# ══════════════════════════════════════════════════════════════════

def test_doctor_all_checks_return_check_result():
    """Every _check_* function returns a CheckResult with a valid
    status + label."""
    checks = [
        doctor._check_python_version,
        doctor._check_numpy,
        doctor._check_styxx_version,
        doctor._check_centroids_sha,
        doctor._check_tier_0,
        doctor._check_tier_1,
        lambda: doctor._check_optional_sdk("json", "json-stdlib"),
        doctor._check_kill_switch,
    ]
    for fn in checks:
        result = fn()
        assert result.status in {"ok", "dim", "warn", "fail"}
        assert result.label
        assert result.sym  # emits a symbol


def test_doctor_runs_and_returns_int(capsys):
    rc = doctor.run_doctor(use_color=False)
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "styxx doctor" in out


# ══════════════════════════════════════════════════════════════════
# hooks (P3)
# ══════════════════════════════════════════════════════════════════

def test_hook_openai_idempotent_and_reversible():
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai SDK not installed")

    # Hook and verify
    first = styxx.hook_openai()
    assert first is True
    assert styxx.hook_openai_active() is True

    # Idempotent second call
    second = styxx.hook_openai()
    assert second is False
    assert styxx.hook_openai_active() is True

    # Unhook and verify
    assert styxx.unhook_openai() is True
    assert styxx.hook_openai_active() is False

    # Double-unhook is a no-op
    assert styxx.unhook_openai() is False


def test_hook_openai_replaces_class():
    try:
        import openai
    except ImportError:
        pytest.skip("openai SDK not installed")

    styxx.unhook_openai()  # clean state
    original = openai.OpenAI
    styxx.hook_openai()
    assert openai.OpenAI is not original
    assert getattr(openai.OpenAI, "_styxx_hooked", False) is True
    styxx.unhook_openai()
    assert openai.OpenAI is original


# ══════════════════════════════════════════════════════════════════
# explain (P4)
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


def test_explain_returns_string_for_vitals():
    v = _fixture_vitals("refusal")
    out = styxx.explain(v)
    assert isinstance(out, str)
    assert len(out) > 50
    # Should mention phase 1 and phase 4
    assert "phase 1" in out
    # Should have a verdict line
    assert "verdict" in out


def test_explain_handles_none():
    out = styxx.explain(None)
    assert isinstance(out, str)
    assert len(out) > 0
    assert "no vitals" in out.lower() or "disabled" in out.lower()


def test_explain_differs_by_gate():
    v_pass = _fixture_vitals("reasoning")
    v_warn = _fixture_vitals("refusal")
    out_pass = styxx.explain(v_pass)
    out_warn = styxx.explain(v_warn)
    # Different gates should produce different prose
    assert out_pass != out_warn
    # The warn version should mention warn or refusal
    assert "warn" in out_warn or "refusal" in out_warn


# ══════════════════════════════════════════════════════════════════
# Vitals.as_markdown (P5)
# ══════════════════════════════════════════════════════════════════

def test_vitals_as_markdown_renders_code_block():
    v = _fixture_vitals("refusal")
    md = v.as_markdown()
    assert md.startswith("```styxx")
    assert md.endswith("```")
    assert "phase1:" in md
    assert "phase4:" in md
    assert "gate:" in md


def test_vitals_as_markdown_short_trajectory():
    v = styxx.Raw().read(
        entropy=[1.0] * 5,
        logprob=[-0.5] * 5,
        top2_margin=[0.5] * 5,
    )
    md = v.as_markdown()
    assert "phase1" in md
    # phase2 may or may not appear depending on window cutoffs
    assert md.startswith("```styxx")


# ══════════════════════════════════════════════════════════════════
# config session tagging (P6)
# ══════════════════════════════════════════════════════════════════

def test_set_session_override():
    styxx.set_session("test-session-xyz")
    assert styxx.session_id() == "test-session-xyz"
    # Clear
    styxx.set_session(None)
    assert styxx.session_id() is None or "STYXX_SESSION_ID" in os.environ


def test_session_id_env_fallback(monkeypatch):
    styxx.set_session(None)
    monkeypatch.setenv("STYXX_SESSION_ID", "env-session-abc")
    assert styxx.session_id() == "env-session-abc"
    monkeypatch.delenv("STYXX_SESSION_ID", raising=False)
    assert styxx.session_id() is None


def test_session_override_beats_env(monkeypatch):
    monkeypatch.setenv("STYXX_SESSION_ID", "from-env")
    styxx.set_session("from-api")
    assert styxx.session_id() == "from-api"
    styxx.set_session(None)
    assert styxx.session_id() == "from-env"
    monkeypatch.delenv("STYXX_SESSION_ID", raising=False)


# ══════════════════════════════════════════════════════════════════
# analytics: load_audit + log_stats + log_timeline
# ══════════════════════════════════════════════════════════════════

def test_load_audit_returns_entries(temp_audit_log):
    entries = analytics.load_audit()
    assert len(entries) == 20
    # Chronological order
    assert entries[0]["ts"] < entries[-1]["ts"]


def test_load_audit_last_n_filter(temp_audit_log):
    entries = analytics.load_audit(last_n=5)
    assert len(entries) == 5


def test_load_audit_since_filter(temp_audit_log):
    # Entries from the last 1000 seconds
    entries = analytics.load_audit(since_s=1000)
    # Should include the creative run (900s ago)
    assert len(entries) >= 5


def test_load_audit_session_filter(temp_audit_log):
    entries_a = analytics.load_audit(session_id="ses-a")
    entries_b = analytics.load_audit(session_id="ses-b")
    assert len(entries_a) == 13   # 10 reasoning + 3 refusal
    assert len(entries_b) == 7    # 2 halluc + 5 creative


def test_log_stats_aggregation(temp_audit_log):
    stats = analytics.log_stats()
    assert stats.n_entries == 20
    # 15 pass (10 reasoning + 5 creative), 3 warn, 2 fail
    assert stats.gate_counts.get("pass", 0) == 15
    assert stats.gate_counts.get("warn", 0) == 3
    assert stats.gate_counts.get("fail", 0) == 2


def test_log_stats_summary_renders(temp_audit_log):
    stats = analytics.log_stats()
    out = stats.summary()
    assert "gate distribution" in out
    assert "phase1" in out
    assert "phase4" in out


def test_log_timeline_renders(temp_audit_log):
    out = analytics.log_timeline(last_n=5)
    assert "time" in out
    assert "phase1" in out
    assert "phase4" in out
    assert "gate" in out


# ══════════════════════════════════════════════════════════════════
# streak + mood (P9)
# ══════════════════════════════════════════════════════════════════

def test_streak_detects_current_run(temp_audit_log):
    # Most recent 5 entries are all "creative" -> streak of 5
    s = analytics.streak()
    assert s is not None
    assert s.category == "creative"
    assert s.length == 5


def test_mood_returns_string(temp_audit_log):
    m = analytics.mood(window_s=86400)
    assert isinstance(m, str)
    assert m in {
        "drifting", "cautious", "defensive",
        "creative", "steady", "unfocused", "mixed", "quiet",
    }


# ══════════════════════════════════════════════════════════════════
# fingerprint (P7)
# ══════════════════════════════════════════════════════════════════

def test_fingerprint_returns_full_vector(temp_audit_log):
    fp = analytics.fingerprint(last_n=500)
    assert fp is not None
    assert fp.n_samples == 20
    # 6 categories + 6 categories + 4 gate statuses
    assert len(fp.phase1_vec) == 6
    assert len(fp.phase4_vec) == 6
    assert len(fp.gate_vec) == 4
    # Vectors should sum to roughly 1.0 (they're rates)
    assert abs(sum(fp.phase1_vec) - 1.0) < 0.01 or sum(fp.phase1_vec) == 0
    assert abs(sum(fp.phase4_vec) - 1.0) < 0.01 or sum(fp.phase4_vec) == 0


def test_fingerprint_cosine_similarity_identical(temp_audit_log):
    fp = analytics.fingerprint(last_n=500)
    sim = fp.cosine_similarity(fp)
    assert abs(sim - 1.0) < 0.001


def test_fingerprint_detects_drift():
    """Two fingerprints with different distributions should have
    cosine similarity < 1.0."""
    from styxx.analytics import Fingerprint
    fp_a = Fingerprint(
        n_samples=10,
        phase1_vec=(0, 1, 0, 0, 0, 0),  # 100% reasoning
        phase4_vec=(0, 1, 0, 0, 0, 0),
        phase1_mean_conf=0.5,
        phase4_mean_conf=0.5,
        gate_vec=(1, 0, 0, 0),
    )
    fp_b = Fingerprint(
        n_samples=10,
        phase1_vec=(0, 0, 1, 0, 0, 0),  # 100% refusal
        phase4_vec=(0, 0, 1, 0, 0, 0),
        phase1_mean_conf=0.5,
        phase4_mean_conf=0.5,
        gate_vec=(0, 1, 0, 0),
    )
    sim = fp_a.cosine_similarity(fp_b)
    # Orthogonal distributions -> sim = 0
    assert sim < 0.1


# ══════════════════════════════════════════════════════════════════
# personality (P8) — the headline feature
# ══════════════════════════════════════════════════════════════════

def test_personality_returns_profile(temp_audit_log):
    # temp_audit_log has 20 entries all within the last 2 hours,
    # so days=1 window should include all of them.
    profile = analytics.personality(days=1.0)
    assert profile is not None
    assert profile.n_samples == 20
    # Category rates should sum to ~1.0
    total = sum(profile.rates.values())
    assert abs(total - 1.0) < 0.01
    # Should identify reasoning as dominant (10 of 20)
    assert profile.rates.get("reasoning", 0) > 0.4


def test_personality_narrative_present(temp_audit_log):
    profile = analytics.personality(days=1.0)
    assert profile.narrative is not None
    assert len(profile.narrative) > 0


def test_personality_renders(temp_audit_log):
    profile = analytics.personality(days=1.0)
    out = profile.render()
    assert "cognitive personality profile" in out
    assert "phase4 category distribution" in out
    assert "gate status distribution" in out


def test_personality_handles_empty_log(tmp_path, monkeypatch):
    fake = tmp_path / "chart.jsonl"
    fake.write_text("")   # empty
    monkeypatch.setattr("styxx.analytics._audit_log_path", lambda: fake)
    profile = analytics.personality(days=7)
    assert profile is None


# ══════════════════════════════════════════════════════════════════
# dreamer (P10)
# ══════════════════════════════════════════════════════════════════

def test_dreamer_counts_would_have_fired(temp_audit_log):
    # threshold=0.10 is below chance floor, should fire on every
    # load-bearing-category entry
    report = analytics.dreamer(threshold=0.10)
    assert report.n_total == 20
    # We have 3 refusal (warn) + 2 hallucination (fail) = 5 load-bearing
    assert report.n_would_have_fired >= 5


def test_dreamer_higher_threshold_fires_less(temp_audit_log):
    low = analytics.dreamer(threshold=0.10)
    high = analytics.dreamer(threshold=0.40)
    assert high.n_would_have_fired <= low.n_would_have_fired


def test_dreamer_by_category(temp_audit_log):
    report = analytics.dreamer(threshold=0.10)
    assert "refusal" in report.by_category or "hallucination" in report.by_category


# ══════════════════════════════════════════════════════════════════
# version sanity
# ══════════════════════════════════════════════════════════════════

def test_version_is_current():
    assert styxx.__version__.startswith(("0.2.", "0.3.", "0.4.", "0.5.", "0.6.", "0.7.", "0.8.", "0.9.", "1."))


def test_all_0_1_0a3_exports_present():
    for name in (
        "hook_openai", "unhook_openai", "hook_openai_active",
        "explain",
        "session_id", "set_session",
        "load_audit", "log_stats", "LogStats",
        "log_timeline",
        "streak", "Streak",
        "mood",
        "fingerprint", "Fingerprint",
        "personality", "Personality",
        "dreamer", "DreamReport",
    ):
        assert hasattr(styxx, name), f"styxx.{name} missing"
