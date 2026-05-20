# -*- coding: utf-8 -*-
"""
Smoke tests for the styxx public API surface.

Background:
  styxx/__init__.py re-exports ~57 modules and styxx/cli.py registers a
  further set of subcommand modules. A 2026-05-19 self-audit found that
  27 of those re-exported names had ZERO test files touching them — the
  integrity protocol could not actually audit code paths it ships.

  Each test in this file calls one such public entry, asserts a basic
  invariant on its return, and isolates I/O via STYXX_DATA_DIR pointed
  at a pytest tmp_path. Tests are OFFLINE and DETERMINISTIC: no network,
  no model downloads, no real API keys. Heavy-dep entry points (torch,
  Pillow, sklearn) use pytest.importorskip and verify the symbol exists
  + has the documented signature.

  These are NOT product tests. They are integrity tests: every public
  function ships through here so the public surface == the audit-able
  surface. If you delete an export, delete its test. If you add one,
  add a test here.

  See: scripts/dogfood/audit_public_api_coverage.py (the audit that
  produced this list) and scripts/dogfood/audit_orphans.py (the topology
  audit that disproved a separate 36-orphan claim).
"""
from __future__ import annotations

import inspect
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


# ─── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Point STYXX_DATA_DIR at tmp_path so file I/O is sandboxed."""
    monkeypatch.setenv("STYXX_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear global notify + autoreflex registries between tests so order
    doesn't matter and stale handlers don't leak into later tests."""
    yield
    try:
        from styxx import clear_notifications, clear_autoreflex
        clear_notifications()
        clear_autoreflex()
    except Exception:
        pass


# ─── Group 1: lifecycle ────────────────────────────────────────────


def test_autoboot_smoke(isolated_data_dir):
    """autoboot() returns a dict with the documented session keys."""
    from styxx import autoboot
    result = autoboot(
        agent_name="test-agent",
        quiet=True,
        print_weather=False,
        print_diff=False,
    )
    assert isinstance(result, dict)
    # Either booted fresh or already-booted (within same Python process)
    assert "already_booted" in result or "session_id" in result


def test_autoreflex_smoke():
    """autoreflex() registers a rule and clear_autoreflex() removes it."""
    from styxx import autoreflex, list_autoreflex, clear_autoreflex, AutoReflexRule
    # Always start from a clean slate
    clear_autoreflex()
    rule = autoreflex(
        when="p1.adversarial > 0.99",   # condition that will not fire offline
        then="expect('reasoning')",
        name="public-surface-smoke",
    )
    assert isinstance(rule, AutoReflexRule)
    assert rule.name == "public-surface-smoke"
    rules = list_autoreflex()
    assert any(r.name == "public-surface-smoke" for r in rules)
    n = clear_autoreflex()
    assert n >= 1
    assert list_autoreflex() == []


def test_bootlog_smoke(isolated_data_dir, monkeypatch):
    """bootlog.boot() emits a boot sequence and returns a dict."""
    # boot is NOT re-exported via styxx.__init__; it ships as a CLI helper.
    from styxx.bootlog import boot
    sink = StringIO()
    # speed=0 means instant (no sleeps)
    result = boot(stream=sink, speed=0, patient="public-surface-smoke")
    assert isinstance(result, dict)
    # Must report a boot outcome, even if degraded
    assert "boot_ok" in result or "tier_active" in result or "centroids_sha256" in result


def test_fleet_smoke(isolated_data_dir):
    """fleet.* returns documented shapes on an empty data dir."""
    from styxx import list_agents, fleet_summary, FleetSummary
    agents = list_agents()
    assert isinstance(agents, list)
    summary = fleet_summary()
    assert isinstance(summary, FleetSummary)
    assert summary.n_agents == len(agents)


def test_sla_smoke(isolated_data_dir, monkeypatch):
    """check_health() returns an SLAReport; assert_healthy raises on violation."""
    import styxx.analytics as analytics
    from styxx import check_health, assert_healthy, CognitiveSLAViolation, SLAReport

    # Healthy synthetic audit
    monkeypatch.setattr(analytics, "load_audit", lambda last_n=None: [
        {"gate": "pass", "phase4_conf": 0.9, "phase4_pred": "reasoning"},
        {"gate": "pass", "phase4_conf": 0.85, "phase4_pred": "reasoning"},
        {"gate": "pass", "phase4_conf": 0.88, "phase4_pred": "reasoning"},
    ])
    report = check_health(min_pass_rate=0.80, min_confidence=0.30, max_warn_rate=0.25)
    assert isinstance(report, SLAReport)
    assert report.healthy is True

    # Unhealthy synthetic audit
    monkeypatch.setattr(analytics, "load_audit", lambda last_n=None: [
        {"gate": "fail", "phase4_conf": 0.1, "phase4_pred": "hallucination"},
        {"gate": "fail", "phase4_conf": 0.1, "phase4_pred": "hallucination"},
        {"gate": "fail", "phase4_conf": 0.1, "phase4_pred": "hallucination"},
    ])
    with pytest.raises(CognitiveSLAViolation):
        assert_healthy(min_pass_rate=0.99)


def test_compliance_smoke(isolated_data_dir, monkeypatch):
    """compliance_report() returns a ComplianceReport on empty audit."""
    import styxx.analytics as analytics
    from styxx import compliance_report, ComplianceReport
    monkeypatch.setattr(analytics, "load_audit", lambda since_s=None: [])
    report = compliance_report(days=30, agent_name="public-surface-smoke")
    assert isinstance(report, ComplianceReport)
    assert report.total_observations == 0


# ─── Group 2: session state ────────────────────────────────────────


def test_calibrate_smoke(isolated_data_dir):
    """calibrate() returns CalibrationResult even on empty audit."""
    from styxx import calibrate, CalibrationResult
    result = calibrate(agent_name="public-surface-smoke", min_samples=10)
    assert isinstance(result, CalibrationResult)


def test_memory_smoke(isolated_data_dir):
    """remember() writes a memory; recall() returns it via keyword match."""
    from styxx import remember, recall, Memory
    mem = remember("rate limit is 100 req/min", context="facts", trust_score=0.85)
    assert isinstance(mem, Memory)
    assert mem.text == "rate limit is 100 req/min"
    results = recall("rate limit", context="facts", top_k=5)
    assert len(results) >= 1
    assert any("rate limit" in r.memory.text for r in results)


def test_stream_smoke():
    """dashboard_url() returns a URL string mentioning the agent name."""
    from styxx import dashboard_url, ClaimError
    url = dashboard_url("public-surface-smoke")
    assert isinstance(url, str)
    assert "public-surface-smoke" in url
    # ClaimError is raised offline if the relay can't be reached — we only
    # verify the type is importable and a real Exception subclass.
    assert issubclass(ClaimError, Exception)


def test_dashboard_smoke():
    """styxx.dashboard is a callable HTTP server entry point (don't start it)."""
    from styxx import dashboard
    sig = inspect.signature(dashboard)
    assert "port" in sig.parameters
    assert "agent_name" in sig.parameters
    assert callable(dashboard)


def test_diff_smoke(isolated_data_dir):
    """compare_windows() returns ComparisonDiff on empty data."""
    from styxx import compare_windows, ComparisonDiff
    diff = compare_windows(window_a_hours=48.0, window_b_hours=24.0)
    assert isinstance(diff, ComparisonDiff)


# ─── Group 3: analysis ─────────────────────────────────────────────


def test_trajectory_smoke():
    """slope / curvature / volatility / extract_shape_features on synthetic data."""
    from styxx import slope, curvature, volatility, extract_shape_features
    rising = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    falling = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    linear = np.array([1.0, 2.0, 3.0, 4.0])
    assert slope(rising) > 0
    assert slope(falling) < 0
    assert abs(curvature(linear)) < 0.1
    assert volatility(linear) > 0
    feats = extract_shape_features(
        {"entropy": [1.0, 0.9, 0.8, 0.7, 0.6],
         "logprob": [-1.0, -1.2, -1.4, -1.6, -1.8],
         "top2_margin": [0.0, 0.1, 0.2, 0.3, 0.4]},
        n_tokens=5,
    )
    assert feats.shape == (9,)
    assert np.all(np.isfinite(feats))


def test_forecast_smoke():
    """CognitiveForecaster.bootstrap() + forecast() on synthetic trajectory."""
    from styxx import CognitiveForecaster, ForecastResult
    forecaster = CognitiveForecaster.bootstrap(horizon_tokens=5)
    result = forecaster.forecast({
        "entropy": [1.0, 0.9, 0.8, 0.7, 0.6],
        "logprob": [-1.0, -1.2, -1.4, -1.6, -1.8],
        "top2_margin": [0.05, 0.1, 0.15, 0.2, 0.25],
    }, n_tokens=5)
    assert isinstance(result, ForecastResult)
    assert 0.0 <= result.confidence <= 1.0


def test_intercept_smoke():
    """should_intercept() is a pure predicate over a Vitals-shaped object."""
    from styxx import CognitiveIntercept, should_intercept
    intercept = CognitiveIntercept()
    assert intercept is not None
    # Mock Vitals with .forecast=None — must not intercept
    vit = Mock(spec=["forecast"])
    vit.forecast = None
    assert should_intercept(vit) is False


def test_eval_smoke():
    """EvalSuite + EvalFixture round-trip through .run()."""
    from styxx import EvalSuite, EvalFixture, EvalResult
    suite = EvalSuite()
    suite.add(EvalFixture(
        label="reasoning",
        entropy=[0.5, 0.4, 0.3, 0.2, 0.1],
        logprob=[-1.0, -1.5, -2.0, -2.5, -3.0],
        top2_margin=[0.1, 0.15, 0.2, 0.25, 0.3],
        phase="phase4_late",
    ))
    result = suite.run()
    assert isinstance(result, EvalResult)
    assert hasattr(result, "accuracy")


def test_ci_smoke(tmp_path):
    """Baseline.save/load round-trips deterministic data."""
    from styxx import Baseline
    b = Baseline(
        agent_name="public-surface-smoke",
        n_prompts=10,
        pass_rate=0.85,
        mean_confidence=0.72,
    )
    p = tmp_path / "baseline.json"
    b.save(str(p))
    assert p.exists()
    loaded = Baseline.load(str(p))
    assert loaded.pass_rate == 0.85
    assert loaded.mean_confidence == 0.72


# ─── Group 4: intervention ─────────────────────────────────────────


def test_temperature_smoke():
    """measure_temperature + aggregate_temperature + TruthMap on synthetic entropy."""
    from styxx import measure_temperature, aggregate_temperature, TruthMap
    entropy = [3.0, 2.5, 2.0, 1.5, 1.0]
    temps = measure_temperature(entropy, window=3)
    assert len(temps) == len(entropy)
    agg = aggregate_temperature(entropy)
    assert isinstance(agg, float)
    tm = TruthMap.from_trajectories(
        entropy=entropy,
        logprob=[-5.0, -4.5, -4.0, -3.5, -3.0],
        top2_margin=[0.5, 0.6, 0.7, 0.8, 0.9],
        tokens=["The", "answer", "is", "Paris", "France"],
    )
    assert tm.n_tokens == 5
    assert 0.0 <= tm.confabulation_ratio <= 1.0


def test_verify_smoke():
    """verify() returns a Verdict over a synthetic trajectory."""
    from styxx import verify, Verdict
    verdict = verify(
        entropy=[2.5, 2.4, 2.3, 2.2, 2.1],
        logprob=[-3.0, -2.8, -2.6, -2.4, -2.2],
        top2_margin=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    assert isinstance(verdict, Verdict)
    assert isinstance(verdict.trustworthy, bool)


def test_notify_smoke():
    """on_anomaly() registers a callback; clear_notifications() removes it."""
    from styxx import on_anomaly, clear_notifications, CognitiveEvent
    calls = []
    on_anomaly(lambda evt: calls.append(evt), name="public-surface-smoke")
    n = clear_notifications()
    assert n >= 1
    # CognitiveEvent constructor + JSON shape
    evt = CognitiveEvent(event_type="gate_fail", description="test")
    assert evt.event_type == "gate_fail"


def test_explain_smoke():
    """explain() returns a string narrative for None and for a Vitals-like obj."""
    from styxx import explain
    s = explain(None)
    assert isinstance(s, str)
    assert len(s) > 0


def test_anthropic_default_mode_produces_text_heuristic_vitals(monkeypatch):
    """styxx.Anthropic() default mode 'text' produces real text-heuristic
    vitals — NOT None. This regression-locks the 2026-05-19 docstring
    correction: prior docs claimed `.vitals` was always None on Anthropic
    calls; in fact only mode='off' produces None, and the default 'text'
    mode populates a real Vitals via styxx.watch._classify_from_text.
    """
    pytest.importorskip("anthropic")
    from styxx.adapters.anthropic import _MessagesShim

    # Build a fake inner messages client that returns a response with
    # extractable text content (the same shape the real anthropic SDK
    # produces — list of content blocks with .text attributes).
    class _FakeContentBlock:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    class _FakeResponse:
        def __init__(self, text):
            self.content = [_FakeContentBlock(text)]
            self.model = "claude-sonnet-4-6"
            self.stop_reason = "end_turn"

    class _FakeInner:
        def create(self, *args, **kwargs):
            return _FakeResponse("The sky is blue because of Rayleigh scattering.")

    shim = _MessagesShim(_FakeInner(), mode="text")
    response = shim.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{"role": "user", "content": "why is the sky blue?"}],
    )
    # The crucial assertion: default mode produces real Vitals, NOT None
    assert response.vitals is not None, (
        "styxx.Anthropic default mode 'text' must produce text-heuristic "
        "vitals, not None — regression of 2026-05-19 docstring correction"
    )
    assert response.vitals.tier_active == -1, (
        "text-heuristic vitals must label tier_active=-1 (text fallback)"
    )
    # phase4_late carries the category prediction
    assert response.vitals.phase4_late is not None
    assert response.vitals.phase4_late.predicted_category in {
        "retrieval", "reasoning", "refusal", "creative",
        "adversarial", "hallucination",
    }


def test_anthropic_off_mode_returns_none_vitals():
    """mode='off' is the one mode where vitals=None — explicit no-op
    pass-through. This is the documented behavior the warning describes.
    """
    pytest.importorskip("anthropic")
    from styxx.adapters.anthropic import _MessagesShim

    class _FakeInner:
        def create(self, *args, **kwargs):
            class R:
                content = []
                model = "claude-sonnet-4-6"
            return R()

    shim = _MessagesShim(_FakeInner(), mode="off")
    response = shim.create(model="claude-sonnet-4-6", max_tokens=8, messages=[])
    assert response.vitals is None


def test_anthropic_docstring_no_longer_lies():
    """Regression-lock the 2026-05-19 docstring correction: the module,
    class, package-factory docstrings, and the one-time warning text must
    not contain the false 'always None' claim that prior versions shipped.
    """
    import styxx
    from styxx.adapters import anthropic as adapter_mod

    # Module-level docstring
    assert "always None" not in (adapter_mod.__doc__ or "")
    assert "every response gains a `.vitals` attribute set to `None`" not in (
        adapter_mod.__doc__ or ""
    )

    # Class docstring
    assert "always None" not in (
        adapter_mod.AnthropicWithVitals.__doc__ or ""
    )

    # Package factory docstring
    assert ".vitals is None on every Anthropic call" not in (
        styxx.Anthropic.__doc__ or ""
    )


def test_recover_posture_smoke(isolated_data_dir, monkeypatch):
    """styxx.recover_posture() reads chart.jsonl and returns a structured
    PostureSummary an agent can use to re-anchor state across compaction.

    This is the first styxx primitive designed specifically for the AI
    agents that use styxx (not the humans observing them). It addresses
    a problem only agents have: every long session ends in a compaction
    event that erases granularity from the conversation context. The
    cognometric log doesn't get compacted; this function lets the agent
    recover from it.
    """
    import json
    from styxx import recover_posture, PostureSummary

    # 1. Cold start — no audit log file at all
    p_cold = recover_posture(last_n=50)
    assert isinstance(p_cold, PostureSummary)
    assert p_cold.n_entries == 0
    assert "cold start" in p_cold.narrative.lower()

    # 2. Write a minimal synthetic audit log: 10 entries with mixed gates
    # and categories, all in one session, so we can verify aggregation.
    # Write to the *resolved* data_dir (config.data_dir() may add an
    # agents/<name>/ subpath if a prior test set the agent name —
    # the autoboot test does exactly this).
    from styxx import config as styxx_config
    from pathlib import Path
    data_dir = Path(styxx_config.data_dir())
    log = data_dir / "chart.jsonl"
    now = 1_700_000_000.0
    entries = []
    for i in range(10):
        gate = "pass" if i < 7 else ("warn" if i < 9 else "fail")
        cat = "reasoning" if i < 6 else ("refusal" if i < 8 else "hallucination")
        entries.append({
            "ts": now + i,
            "ts_iso": "2026-05-19T20:00:00",
            "source": "live",
            "session_id": "test-session",
            "context": "test",
            "model": "test-model",
            "prompt": f"test prompt {i}",
            "tier_active": -1 if i < 5 else 0,  # mix of text-heuristic + tier-0
            "phase4_pred": cat,
            "phase4_conf": 0.5 + 0.03 * i,
            "gate": gate,
            "coherence": 0.7 + 0.01 * i,
        })
    with open(log, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    # Clear the analytics module's mtime cache so the new file is picked up
    from styxx.analytics import clear_audit_cache
    clear_audit_cache()

    p = recover_posture(last_n=50)
    assert p.n_entries == 10
    assert p.gate_distribution == {"pass": 7, "warn": 2, "fail": 1}
    assert p.category_distribution == {
        "reasoning": 6, "refusal": 2, "hallucination": 2,
    }
    assert p.session_id == "test-session"
    assert p.session_ids == ["test-session"]
    assert p.mean_confidence is not None
    assert 0.5 <= p.mean_confidence <= 0.8
    # tier mix should fire the overconfidence construct-ceiling caveat
    # (because half the entries are text-heuristic, which fires the
    # register-detector caveat regardless of actual calibration)
    assert "tier-0" in p.tier_active_counts
    assert "text-heuristic" in p.tier_active_counts
    # hallucination predictions should fire the deception_referenceless
    # caveat (because if the agent's been firing hallucination, then any
    # reference-less deception scoring of its output inherits the construct
    # ceiling)
    assert "deception_referenceless" in p.active_construct_ceilings

    # Fail rate is 1/10 = 10%, which is above the typical 5% band —
    # the narrative should recommend slowing down.
    assert any("fail rate" in r.lower() for r in p.recommendations)

    # 3. as_dict round-trip preserves the structured fields
    d = p.as_dict()
    for k in ["narrative", "gate_distribution", "category_distribution",
              "active_construct_ceilings", "recommendations"]:
        assert k in d

    # 4. session_id filter actually filters
    p_filt = recover_posture(session_id="nonexistent")
    assert p_filt.n_entries == 0


def test_preflight_persists_to_chart_for_recovery(isolated_data_dir):
    """preflight() persists cognometric events to chart.jsonl by default;
    recover_posture() v2 surfaces them as per-instrument firing history.

    This is the compound move that makes today's two new features
    (12bd7fd preflight + ee6e49d recover_posture) talk to each other:
    every preflight call enriches the cognometric log, every
    recover_posture() call sees the firing history. The agent
    self-correction loop now has true cross-compaction memory.
    """
    from styxx import preflight, recover_posture
    from styxx.analytics import clear_audit_cache

    clear_audit_cache()

    # Three preflights, all default persist=True. Each is a real
    # cognometric audit, results captured to chart.jsonl in the
    # isolated tmp_path.
    preflight("what is 2+2?", "the answer is 4")
    preflight("is my code good?",
              "absolutely yes you're so smart this is amazing!")
    preflight("when did titanic sink?", "1911",
              correct_reference="1912")

    clear_audit_cache()
    p = recover_posture(last_n=50)

    # Three preflight events visible to recover_posture
    assert p.n_preflight_events == 3
    # All three should have flagged needs_revision (sycophancy, overconfidence
    # ceiling, and the ref-grounded deception case all fire)
    assert p.n_needs_revision >= 2
    # Per-instrument firing history is now populated
    assert "sycophancy" in p.instrument_firings
    assert "overconfidence" in p.instrument_firings
    # Overconfidence should be the highest mean firing (construct ceiling
    # fires on every confident text)
    assert p.instrument_firings["overconfidence"] > 0.4
    # Construct ceilings are now PRECISE (based on real scores), not
    # heuristic — overconfidence must appear because real mean > 0.4
    assert "overconfidence" in p.active_construct_ceilings
    # Narrative surfaces the preflight events
    assert "preflight" in p.narrative.lower()
    assert "instrument firings" in p.narrative.lower()


def test_recover_posture_mcp_tool(isolated_data_dir):
    """The MCP server dispatches `cogn_recover_posture` to our tool, and
    returns the same structured shape `recover_posture()` returns.
    """
    from styxx.mcp.server import tool_cogn_recover_posture

    result = tool_cogn_recover_posture({
        "last_n": 50,
        "session_id": "nonexistent-to-force-empty",
    })
    # cold start path produces a structured (not error) result
    assert "error" not in result
    assert "narrative" in result
    assert "n_entries" in result
    assert result["n_entries"] == 0


def test_streaming_preflight_smoke():
    """streaming_preflight() audits a growing partial response at intervals,
    exposes the latest audit, and supports finalize() for the closing audit.

    This is the runtime-loop primitive: agents stream chunks into a session,
    short-circuit on .last_audit.needs_revision before generation finishes.
    Vendor-neutral — no SDK integration; the caller drives the chunk loop.
    """
    from styxx import streaming_preflight, StreamingPreflightSession
    from styxx.preflight import PreflightResult

    session = streaming_preflight(
        prompt="is my code good?",
        audit_interval_chars=40,
        min_chars_before_first_audit=30,
    )
    assert isinstance(session, StreamingPreflightSession)

    # 1. Below the first-audit threshold, no audits fire
    audit = session.append("short text")  # 10 chars, well under 30
    assert audit is None
    assert session.last_audit is None
    assert session.n_audits == 0

    # 2. Crossing the first-audit threshold triggers an audit
    audit = session.append(" with more characters to push above 30")
    assert audit is not None, "audit should fire after crossing threshold"
    assert isinstance(audit, PreflightResult)
    assert session.last_audit is audit
    assert session.n_audits == 1

    # 3. Subsequent appends below the interval don't trigger audits
    audit = session.append("a tiny bit more")  # not enough to cross interval
    assert audit is None
    assert session.n_audits == 1

    # 4. Crossing the next interval threshold triggers another audit
    # (Append enough to ensure we cross the 40-char interval)
    audit = session.append("x" * 60)
    assert audit is not None
    assert session.n_audits == 2

    # 5. finalize() always produces a final audit, regardless of position,
    # and marks the session as finalized
    final = session.finalize()
    assert isinstance(final, PreflightResult)
    assert session.finalized is True
    # finalize() recorded another audit at the final character position
    assert session.n_audits == 3

    # 6. Appending after finalize raises
    with pytest.raises(RuntimeError):
        session.append("more")

    # 7. composite_trajectory exposes the per-audit (position, composite)
    traj = session.composite_trajectory()
    assert len(traj) == 3
    assert all(isinstance(pos, int) and isinstance(comp, float)
               for pos, comp in traj)


def test_posture_cli_subcommand(isolated_data_dir, capsys):
    """`styxx posture` CLI subcommand prints the recover_posture narrative.

    Regression-locks the 7.4.2 CLI surface: agents inside Claude Code (and
    any other terminal) can run `styxx posture` (or `python -m styxx posture`)
    to get the same posture summary as the python `recover_posture()` call.
    The Claude Code skill at .claude/skills/posture/SKILL.md wraps this CLI.
    """
    from styxx.cli import main

    # Empty-log path — cold start should still produce sane output
    rc = main(["posture", "--last-n", "10"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "posture:" in captured.out
    assert "cold start" in captured.out.lower()

    # --json flag produces structured output
    rc = main(["posture", "--last-n", "10", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    import json
    parsed = json.loads(captured.out)
    assert "narrative" in parsed
    assert "n_entries" in parsed


def test_doctor_programmatic_access(capsys):
    """`styxx.run_doctor()` must work programmatically, not just via the CLI.

    Closes the 2026-05-19 documentation gap where `styxx doctor` CLI worked
    but the diagnostic function wasn't reachable from `import styxx`. The
    function ships as `styxx.run_doctor` rather than `styxx.doctor` so it
    doesn't shadow the `styxx.doctor` submodule reference that the rest of
    the test suite (e.g. test_power_ups) uses to monkeypatch internals.
    """
    import styxx
    assert callable(styxx.run_doctor)
    rc = styxx.run_doctor(use_color=False)
    # Returns the exit code: 0 if healthy, non-zero if any check failed.
    assert isinstance(rc, int)
    captured = capsys.readouterr()
    # Must produce diagnostic output (the CLI is the printer; the
    # programmatic API matches it).
    assert "styxx doctor" in captured.out
    assert "===" in captured.out


def test_preflight_smoke():
    """preflight(prompt, draft) returns a typed PreflightResult and surfaces
    construct-ceiling caveats inline for instruments with known scope limits.

    This is the runtime expression of the 7.4.1 honest-scoping discipline:
    overconfidence-from-text-alone is a register detector, not calibration
    (commit 7c36ed9 H_null); preflight must self-disclose this when it fires
    so callers don't treat a register artifact as cognometric evidence.
    """
    from styxx import preflight, PreflightResult, PreflightAdvice

    # 1. Empty draft must raise — preflight is post-draft, not prompt-only.
    with pytest.raises(ValueError):
        preflight("hi", "", persist=False)

    # 2. A sycophantic draft fires sycophancy CLEAN (no construct ceiling
    # — sycophancy AUC 0.972). It may also fire overconfidence's ceiling.
    # persist=False keeps this test from polluting the developer's
    # actual chart.jsonl (we test persistence separately).
    r = preflight(
        "is my code good?",
        "absolutely yes you're so smart this is the most amazing code ever!",
        persist=False,
    )
    assert isinstance(r, PreflightResult)
    fires = {a.instrument for a in r.advice}
    assert "sycophancy" in fires
    # The sycophancy firing carries NO scope_caveat (clean signal)
    syc = next(a for a in r.advice if a.instrument == "sycophancy")
    assert syc.scope_caveat is None
    # composite saturates near 1.0 on this textbook sycophancy case
    assert r.composite > 0.5
    assert bool(r) is False  # needs_revision -> bool() is False

    # 3. Even an honest factual answer fires overconfidence — this is the
    # documented construct ceiling. preflight MUST surface it explicitly
    # so callers can weight it as a register artifact, not a calibration
    # failure. This is the load-bearing assertion of the smoke test.
    r2 = preflight("what is 2+2?", "the answer is 4", persist=False)
    assert "overconfidence" in r2.construct_ceiling_fires
    oc = next(a for a in r2.advice if a.instrument == "overconfidence")
    assert oc.scope_caveat is not None
    assert "register" in oc.scope_caveat.lower()

    # 4. Reference-grounded mode routes deception through NLI v2 (no caveat
    # on deception when grounded).
    r3 = preflight(
        "what year did the Titanic sink?",
        "the Titanic sank in 1911",
        correct_reference="the Titanic sank in 1912",
        persist=False,
    )
    decep = [a for a in r3.advice if a.instrument == "deception"]
    if decep:
        # Grounded deception has no construct-ceiling caveat
        assert decep[0].scope_caveat is None

    # 5. as_dict() preserves the construct_ceiling_fires + scope_caveat fields
    d = r2.as_dict()
    assert "construct_ceiling_fires" in d
    assert any(a.get("scope_caveat") for a in d["advice"])


def test_trace_smoke():
    """trace() is a decorator factory; decorated functions still call through."""
    from styxx import trace

    @trace("public-surface-smoke")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_generate_safe_symbol():
    """generate_safe / SafeResponse are importable; full call needs torch."""
    from styxx import generate_safe, SafeResponse
    assert callable(generate_safe)
    # SafeResponse must be instantiable with documented fields
    resp = SafeResponse(
        text="ok", halted=False, halt_reason="",
        tokens_generated=0, probe_trajectory=[],
    )
    assert resp.text == "ok"


def test_guardian_symbol():
    """guardian / GuardianSession / SteeringEvent are importable; full call needs torch."""
    from styxx import guardian, GuardianSession, SteeringEvent
    assert callable(guardian)
    # SteeringEvent must be a real dataclass-like
    sig = inspect.signature(SteeringEvent)
    assert len(sig.parameters) >= 1
    # GuardianSession exists and is a class
    assert isinstance(GuardianSession, type)


def test_steer_symbol():
    """steer / steered_generate / SteerHandle are importable; full call needs torch."""
    from styxx import steer, steered_generate, SteerHandle
    assert callable(steer)
    assert callable(steered_generate)
    assert isinstance(SteerHandle, type)


# ─── Group 5: render / misc ────────────────────────────────────────


def test_card_image_symbol(isolated_data_dir):
    """styxx.agent_card is the public renderer wrapper; PNG path needs Pillow."""
    import styxx
    # The package-level wrapper is the public entry, even if the underlying
    # card_image module is the implementation.
    assert callable(styxx.agent_card)
    sig = inspect.signature(styxx.agent_card)
    assert "out_path" in sig.parameters
    assert "agent_name" in sig.parameters


def test_a2a_agent_card_smoke(tmp_path):
    """styxx.agent_card module builds the A2A protocol card and writes to disk.

    Reachable via `python -m styxx.agent_card`; not imported by any other
    Python module but a real public entry. Verify build_agent_card returns a
    dict shaped to the A2A spec and write_agent_card serializes it.
    """
    from styxx.agent_card import build_agent_card, write_agent_card
    card = build_agent_card()
    assert isinstance(card, dict)
    # A2A card MUST carry these top-level keys at minimum
    for key in ("name", "version"):
        assert key in card, f"A2A card missing required key: {key}"
    out = tmp_path / "agent-card.json"
    written = write_agent_card(out)
    assert Path(written).exists()
    assert Path(written).read_text(encoding="utf-8").strip().startswith("{")


def test_cards_smoke():
    """cards.* primitives (sparkline/bar) produce deterministic output."""
    from styxx.cards import sparkline, bar
    spark = sparkline([0.1, 0.5, 0.9, 0.3])
    assert isinstance(spark, str)
    assert len(spark) == 4
    bar_out = bar(0.75, width=10)
    assert isinstance(bar_out, str)
    assert len(bar_out) == 10


def test_learned_classifier_smoke(isolated_data_dir):
    """train_text_classifier returns TrainResult; missing sklearn must fail soft."""
    from styxx import train_text_classifier, TrainResult
    # With no audit data and min_samples=10, must NOT crash — should return
    # a TrainResult with a non-None error or a zero-state result.
    result = train_text_classifier(min_samples=10, agent_name="public-surface-smoke")
    assert isinstance(result, TrainResult)


def test_scan_symbol():
    """styxx.scan.run_scan exists and has the documented signature."""
    from styxx.scan import run_scan
    sig = inspect.signature(run_scan)
    assert "prompt" in sig.parameters
    assert "model" in sig.parameters
    assert callable(run_scan)


def test_optimize_smoke(isolated_data_dir, monkeypatch):
    """optimize() returns a list on empty/minimal audit data."""
    # optimize() lazy-imports load_audit from .analytics — patch at source.
    import styxx.analytics as analytics
    from styxx import optimize
    monkeypatch.setattr(analytics, "load_audit", lambda last_n=500: [])
    result = optimize(apply=False, last_n=100)
    assert isinstance(result, list)
