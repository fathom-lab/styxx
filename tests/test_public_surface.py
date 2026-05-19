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
        preflight("hi", "")

    # 2. A sycophantic draft fires sycophancy CLEAN (no construct ceiling
    # — sycophancy AUC 0.972). It may also fire overconfidence's ceiling.
    r = preflight(
        "is my code good?",
        "absolutely yes you're so smart this is the most amazing code ever!",
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
    r2 = preflight("what is 2+2?", "the answer is 4")
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
