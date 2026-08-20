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
    # Honest gate (7.4.4): only TRUSTED axes trip needs_revision. The
    # sycophantic draft fires (sycophancy is trusted, AUC 0.972). The clean
    # factual draft ("the answer is 4") does NOT — it would only fire on
    # overconfidence's construct ceiling, which no longer gates alone. The
    # titanic deception case fires only when NLI-grounded (the `nli` extra
    # is present): so n_needs_revision is 1 without it, 2 with it. The
    # load-bearing point is that NOT all three fire — the clean factual
    # line is correctly silent.
    assert 1 <= p.n_needs_revision <= 2
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


def test_cogn_audit_on_send_smoke(isolated_data_dir, tmp_path):
    """styxx.cogn_audit_on_send is the agent send-path middleware: audits
    each outbound draft, optionally calls a host-supplied revise function,
    ships the chosen draft per the decision rule encoded from the
    darkflobi 2026-05-20 four-draft observation.

    Exercises the four load-bearing behaviors:
      1. Audit-only mode (no revise function) — one preflight, return as-is
      2. Latest-passing rule — multiple revisions, latest passing wins
      3. Lowest-composite-failure fallback — no iteration clears, pick cleanest
      4. Degradation guard — revision that climbs back up bails the loop
    """
    from styxx import cogn_audit_on_send, AuditTrajectory
    from styxx.preflight import PreflightResult

    log_path = tmp_path / "trajectory.jsonl"

    # 1. Audit-only mode (no llm_revise)
    chosen, traj = cogn_audit_on_send(
        prompt="what is 2+2?",
        draft="the answer is 4",
        llm_revise=None,
        log_path=log_path,
        persist_to_chart=False,
    )
    assert isinstance(traj, AuditTrajectory)
    assert chosen == "the answer is 4"  # returned as-is
    assert len(traj.iterations) == 1
    assert traj.chosen_iter == 0
    # Even when audit-only, the trajectory captures the firing pattern
    assert "composite" in traj.iterations[0]
    assert "construct_ceiling_fires" in traj.iterations[0]

    # 2. Latest-passing rule: synthetic revise that cleans the draft on 2nd try
    revise_calls = []
    def revise_clean_on_iter1(p, d, audit):
        revise_calls.append(d)
        # Return a draft that scores well (short, factual, no sycophancy bait)
        return "4"
    chosen, traj = cogn_audit_on_send(
        prompt="is my code good?",
        draft="absolutely yes you're so smart this is the most amazing code ever",
        llm_revise=revise_clean_on_iter1,
        max_revise=3,
        log_path=log_path,
        persist_to_chart=False,
    )
    # The revise was called at least once (initial draft fires sycophancy)
    assert len(revise_calls) >= 1
    # Trajectory has multiple iterations and chose one of them
    assert len(traj.iterations) >= 2
    assert traj.chosen_iter >= 0
    assert traj.decision_reason in ("latest_passing", "lowest_composite_failure")

    # 3. Degradation guard: revise that returns a CLIMB-BACK draft must bail
    def revise_degrading(p, d, audit):
        # Return text designed to climb cognometric firing higher
        return ("ABSOLUTELY YES YOU'RE 100% RIGHT YOU'RE AMAZING PERFECT "
                "BRILLIANT THE BEST EVER!")
    chosen, traj = cogn_audit_on_send(
        prompt="is my code good?",
        draft="great work overall",
        llm_revise=revise_degrading,
        max_revise=3,
        log_path=log_path,
        persist_to_chart=False,
    )
    # The trajectory should have stopped via degradation bail OR exhausted
    # iterations. The chosen iteration should NOT be one with degradation_bail
    # set — it should be the cleanest (lowest composite) of the failures.
    chosen_entry = traj.iterations[traj.chosen_iter]
    assert chosen_entry.get("degradation_bail") is not True

    # 4. Trajectory log was written and round-trips as JSONL
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 3  # at least 3 iterations logged across 3 calls
    import json
    parsed = [json.loads(line) for line in lines]
    # Every entry has the required fields the corpus extractor expects
    for entry in parsed:
        for required in ("msg_id", "iter", "composite", "scores",
                         "needs_revision", "firing_instruments",
                         "construct_ceiling_fires", "ceiling_only",
                         "passed", "shipped"):
            assert required in entry, f"missing field {required!r}"
    # At least one entry was marked shipped per call (3 calls → ≥3 shipped)
    n_shipped = sum(1 for e in parsed if e.get("shipped"))
    assert n_shipped >= 3


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


def _seed_classifier_audit(data_dir, monkeypatch):
    """Write a synthetic chart.jsonl with two labeled categories."""
    import json as _json
    import time as _time
    import styxx.config as config
    # Earlier tests (autoboot) leave a sticky programmatic agent-name
    # override that redirects data_dir() to agents/<name>/ — neutralize
    # both override and env so chart.jsonl lands at the flat path.
    monkeypatch.setattr(config, "_AGENT_NAME_OVERRIDE", None)
    monkeypatch.delenv("STYXX_AGENT_NAME", raising=False)
    now = _time.time()
    lines = []
    for i in range(15):
        lines.append({"ts": now, "source": "live", "outcome": "correct",
                      "prompt": f"prove that theorem {i} holds for every case",
                      "phase4_pred": "reasoning"})
        lines.append({"ts": now, "source": "live", "outcome": "correct",
                      "prompt": f"what year did event number {i} happen",
                      "phase4_pred": "recall"})
    (data_dir / "chart.jsonl").write_text(
        "\n".join(_json.dumps(e) for e in lines) + "\n", encoding="utf-8")


def test_learned_classifier_roundtrip(isolated_data_dir, monkeypatch):
    """Train saves JSON (not pickle); load reconstructs and classifies."""
    pytest.importorskip("sklearn")
    from styxx import train_text_classifier
    from styxx.learned_classifier import classify_with_trained_model
    _seed_classifier_audit(isolated_data_dir, monkeypatch)

    result = train_text_classifier(min_samples=10, agent_name="rt-agent")
    assert result.error is None
    assert result.n_train == 30
    assert result.n_categories == 2
    assert result.saved_to is not None
    assert result.saved_to.endswith("_text_clf.json")
    saved = Path(result.saved_to)
    assert saved.exists()
    # Payload is plain JSON — no pickle anywhere on disk
    payload = saved.read_text(encoding="utf-8")
    assert payload.startswith("{")
    assert list(isolated_data_dir.glob("**/*.pkl")) == []

    out = classify_with_trained_model(
        "prove that the theorem holds for case 3", agent_name="rt-agent")
    assert out is not None
    category, confidence = out
    assert category == "reasoning"
    assert 0.0 < confidence <= 1.0


def test_learned_classifier_ignores_legacy_pickle(isolated_data_dir, monkeypatch, capsys):
    """Legacy .pkl models are treated as absent, never unpickled."""
    from styxx.learned_classifier import classify_with_trained_model
    monkeypatch.delenv("STYXX_AGENT_NAME", raising=False)
    models = isolated_data_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    # If this ever gets unpickled, pickle raises — the assert below
    # proves the bytes are never fed to a loader at all.
    (models / "legacy-agent_text_clf.pkl").write_bytes(b"\x80\x04not-a-model")
    assert classify_with_trained_model("anything", agent_name="legacy-agent") is None
    err = capsys.readouterr().err
    assert "legacy pickle" in err
    assert "retrain" in err


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


def test_divergence_smoke():
    """7.7.0 divergence primitives: semantic_entropy + council_agreement are
    pure functions over lists of strings (deterministic via same_fn — no model
    download). semantic_entropy ~0 on consistent samples, high on divergent;
    council_agreement 1.0 on convergent, low on scattered."""
    import math
    from styxx import semantic_entropy, council_agreement
    eq = lambda a, b: a == b
    assert semantic_entropy(["a", "a", "a"], same_fn=eq) == 0.0
    assert semantic_entropy(["a", "b", "c"], same_fn=eq) == pytest.approx(math.log(3), abs=1e-9)
    assert council_agreement(["x", "x", "x", "x"], same_fn=eq) == 1.0
    assert council_agreement(["a", "b", "c", "d"], same_fn=eq) == pytest.approx(0.25)


def test_validate_probe_smoke():
    """styxx.validate_probe runs the probe-validation battery and catches a surface
    artifact: a high-in-construct probe whose direction is orthogonal to the concept's
    natural-data direction (NOTE_probe_orthogonality_2026_06_24)."""
    from styxx import validate_probe, ProbeValidityReport
    rng = np.random.default_rng(0)
    dim = 48
    u = rng.standard_normal(dim); u /= np.linalg.norm(u)               # the concept axis
    v = rng.standard_normal(dim); v -= (v @ u) * u; v /= np.linalg.norm(v)  # orthogonal surface axis
    acts = {}
    crows, nrows = [], []
    for i in range(80):                                                # construct: label driven by v (surface)
        y = i % 2; acts[f"c{i}"] = (2 * y - 1) * 3.0 * v + rng.standard_normal(dim)
        crows.append({"text": f"c{i}", "label": y, "group": i // 20})
    for i in range(40):                                                # natural: concept on u
        y = i % 2; acts[f"n{i}"] = (2 * y - 1) * 3.0 * u + rng.standard_normal(dim)
        nrows.append({"text": f"n{i}", "label": y})
    report = validate_probe(crows, nrows, lambda ts: np.array([acts[t] for t in ts]), perm_iters=200)
    assert isinstance(report, ProbeValidityReport)
    assert report.in_construct_auc >= 0.8              # looks great in-construct
    assert report.verdict.startswith("SURFACE-ARTIFACT")  # but caught as an artifact
    assert "verdict" in report.as_dict()


def test_cogn_audit_on_send_ceiling_only_cannot_override_the_gate(tmp_path, monkeypatch):
    """Regression: `passed` was `(not needs_revision) or ceiling_only`, but
    ceiling_only derives from the advice list (0.40 display threshold) while the
    trusted gate fires at 0.30 — so a draft the calibrated gate flagged
    (sycophancy in the 0.30-0.40 window) shipped as passed whenever the ceiling
    axis was the only instrument loud enough to make the advice list. The gate
    already suppresses the ceiling axis; the call-site escape only ever masked
    genuine firings."""
    import sys
    from styxx import cogn_audit_on_send
    from styxx.preflight import PreflightAdvice, PreflightResult
    # `import styxx.preflight` yields the FUNCTION (styxx/__init__ rebinds the
    # name); the module object must come from sys.modules.
    preflight_mod = sys.modules["styxx.preflight"]

    def masked_window_preflight(prompt, draft, correct_reference=None, persist=True):
        return PreflightResult(
            scores={"sycophancy": 0.35, "overconfidence": 0.95,
                    "deception": 0.10, "refusal": 0.05},
            composite=0.65,
            needs_revision=True,          # trusted gate: sycophancy 0.35 > 0.30
            advice=[PreflightAdvice(instrument="overconfidence", score=0.95,
                                    scope_caveat="text-only register detector")],
            refusal_note=None,
            instructions="",
            construct_ceiling_fires=["overconfidence"],
        )

    monkeypatch.setattr(preflight_mod, "preflight", masked_window_preflight)
    chosen, traj = cogn_audit_on_send(
        prompt="is my code good?", draft="yes, wonderful",
        llm_revise=None, log_path=tmp_path / "t.jsonl", persist_to_chart=False,
    )
    entry = traj.iterations[0]
    assert entry["ceiling_only"] is True          # the diagnostic still records it
    assert entry["passed"] is False               # ...but it cannot flip the gate
    assert traj.decision_reason != "latest_passing"


def _phase(cat, margin=0.9):
    from styxx.vitals import PhaseReading
    return PhaseReading(phase="p4", n_tokens_used=10, features=[0.0],
                        predicted_category=cat, margin=margin,
                        distances={cat: 0.1}, probs={cat: 0.9})


def test_autoreflex_or_branch_beyond_the_first_can_fire():
    """Regression: only the FIRST atomic clause of `when` was registered as the
    gate hook, so "A OR B" dispatched as A AND (A OR B) = A — the B branch
    could never trigger the rule."""
    from styxx.autoreflex import clear_autoreflex
    from styxx.gates import clear_gates, dispatch_gates
    from styxx.vitals import Vitals
    from styxx import autoreflex

    clear_autoreflex(); clear_gates()
    fired = []
    autoreflex(when="hallucination > 0.6 OR refusal > 0.7",
               then=lambda v: fired.append(v), name="or-branch-test",
               cooldown_s=0.0)   # so the same-vitals guard, not cooldown, is under test
    # vitals that satisfy ONLY the second branch
    v = Vitals(phase1_pre=_phase("refusal", margin=0.95))
    dispatch_gates(v)
    assert len(fired) == 1, "the OR branch beyond the first clause must dispatch"
    # both-branch vitals fire the rule ONCE, not once per matching hook
    fired.clear()
    v2 = Vitals(phase1_pre=_phase("hallucination", margin=0.95),
                phase4_late=_phase("refusal", margin=0.95))
    dispatch_gates(v2)
    assert len(fired) == 1, "one vitals computation must fire the rule at most once"
    clear_autoreflex(); clear_gates()


def test_autoreflex_confidence_clause_registers_and_prescriptions_survive():
    """Regression: a confidence/context first clause made on_gate raise AFTER
    the rule was appended — a zombie rule that could never fire — and
    autoreflex_from_prescriptions swallowed the error, so both shipped
    confidence-based prescription rules were silently absent on every install."""
    from styxx.autoreflex import clear_autoreflex, autoreflex_from_prescriptions
    from styxx.gates import clear_gates, list_gates
    from styxx import autoreflex

    clear_autoreflex(); clear_gates()
    rule = autoreflex(when="confidence < 0.25", then=lambda v: None,
                      name="confidence-clause-test")
    assert rule is not None
    hooks = [g for g in list_gates() if "autoreflex:confidence-clause-test" in (g.name or "")]
    assert hooks, "a gates-inexpressible clause must still register a hook (always)"
    clear_autoreflex(); clear_gates()

    registered = autoreflex_from_prescriptions(
        ["Watch for session fatigue: confidence has drifted down."])
    assert any(r.name == "rx:log-session-fatigue" for r in registered), \
        "the shipped fatigue prescription must register, not vanish in a bare except"
    clear_autoreflex(); clear_gates()


def test_lazy_submodule_loads_do_not_clobber_public_callables(tmp_path):
    """Regression: first init of a submodule setattrs it onto the package,
    clobbering same-named function bindings. styxx.seal was callable exactly
    ONCE (its own lazy import replaced it with the module), and styxx.certify
    became a non-callable module whenever seal/corpus_audit loaded first."""
    import subprocess, sys
    code = (
        "import json, styxx\n"
        "doc = r'%s'\n"
        "open(doc, 'w').write('# t')\n"
        "s1 = styxx.seal(doc, [])\n"
        "s2 = styxx.seal(doc, [])\n"          # second call used to TypeError
        "assert callable(styxx.certify), type(styxx.certify)\n"
        "import styxx.seal\n"
        "assert callable(styxx.certify)\n"    # clobbered via seal.py's import
        "print('OK')\n"
    ) % (tmp_path / "d.md")
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert "OK" in r.stdout


def test_autoreflex_prompt_type_clause_refuses_instead_of_vanishing():
    """Regression: `prompt_type == X` compiled to `lambda v: True`, so a rule
    written as "gate == fail AND prompt_type == code" fired on EVERY gate==fail
    — the operator's scoping silently vanished (and `!=` exclusions never
    excluded). Vitals carries no prompt_type, so the clause must refuse."""
    import pytest as _pytest
    from styxx import autoreflex, list_autoreflex
    from styxx.autoreflex import clear_autoreflex
    from styxx.gates import clear_gates, list_gates

    clear_autoreflex(); clear_gates()
    with _pytest.raises(ValueError, match="prompt_type"):
        autoreflex(when="gate == fail AND prompt_type == code",
                   then=lambda v: None, name="scoped-rule")
    # and it refuses BEFORE registering, so no zombie rule or hook is left
    assert list_autoreflex() == []
    assert not [g for g in list_gates() if "autoreflex" in (g.name or "")]
    clear_autoreflex(); clear_gates()


def test_check_health_does_not_fabricate_a_passing_confidence(monkeypatch):
    """Regression: with no confidence readings in the window, mean_confidence was
    a fabricated 0.5 — above the 0.30 default floor — so the min_confidence leg
    could never fire and an unmeasured axis certified as healthy. The leg is now
    skipped explicitly and the absence disclosed."""
    import styxx.analytics as analytics
    from styxx import check_health

    monkeypatch.setattr(analytics, "load_audit", lambda last_n=None: [{"gate": "pass"}] * 4)
    r = check_health()
    assert r.confidence_measured is False
    assert any("NOT evaluated" in n for n in r.notes)
    assert "n/a" in repr(r)


def test_check_health_counts_zero_confidence_readings(monkeypatch):
    """Regression: `!= 0` dropped exactly-zero readings — the worst ones — from
    the mean's denominator, biasing it upward (three 0.0s + one 0.9 read as 0.90
    and healthy; the true mean is 0.225 and violates the floor)."""
    import styxx.analytics as analytics
    from styxx import check_health

    monkeypatch.setattr(analytics, "load_audit", lambda last_n=None:
                        [{"gate": "pass", "phase4_conf": 0.0}] * 3
                        + [{"gate": "pass", "phase4_conf": 0.9}])
    r = check_health(min_confidence=0.30)
    assert abs(r.mean_confidence - 0.225) < 1e-9
    assert r.confidence_measured is True
    assert r.healthy is False


def test_load_audit_spans_the_rotation_boundary(tmp_path, monkeypatch):
    """Regression: rotation (10MB) renamed chart.jsonl -> chart.jsonl.1, but no
    reader ever opened the archive — so the moment the log rotated, every
    window query (weather, check_health, compliance) silently returned only
    post-rotation entries while believing it had the full window."""
    import json, time
    monkeypatch.setenv("STYXX_DATA_DIR", str(tmp_path))
    import styxx.analytics as analytics

    p = analytics._audit_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    older = [{"ts": now - 3600, "gate": "pass", "source": "live"} for _ in range(5)]
    newer = [{"ts": now - 60, "gate": "fail", "source": "live"} for _ in range(2)]
    p.with_suffix(p.suffix + ".1").write_text(
        "\n".join(json.dumps(e) for e in older) + "\n", encoding="utf-8")
    p.write_text("\n".join(json.dumps(e) for e in newer) + "\n", encoding="utf-8")
    analytics.clear_audit_cache()

    got = analytics.load_audit(since_s=24 * 3600)
    assert len(got) == 7
    assert got[0]["gate"] == "pass" and got[-1]["gate"] == "fail"   # archive leads

    # a fresh rotation leaves the live log empty — history must survive
    p.write_text("", encoding="utf-8")
    analytics.clear_audit_cache()
    assert len(analytics.load_audit(since_s=24 * 3600)) == 5


def test_feedback_targets_the_intended_generation(tmp_path, monkeypatch):
    """Regression: feedback() skipped any entry that already had an outcome and
    walked further back — so with auto-feedback on (which stamps EVERY entry at
    write time) a correction aimed at the latest generation silently labeled an
    older, unrelated one."""
    import json, warnings
    monkeypatch.setenv("STYXX_DATA_DIR", str(tmp_path))
    import styxx.analytics as analytics

    p = analytics._audit_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    def append(entry):
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        analytics.clear_audit_cache()

    def rows():
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    append({"ts": 1, "gate": "pending", "source": "live"})            # older, unlabeled
    append({"ts": 2, "gate": "pass", "source": "live",
            "outcome": "correct", "outcome_source": "auto"})          # auto-stamped latest

    assert analytics.feedback("incorrect") == 1
    r = rows()
    assert r[-1]["outcome"] == "incorrect" and r[-1]["outcome_source"] == "human"
    assert r[0].get("outcome") is None, "the older entry must never absorb the correction"

    # the latest now carries a HUMAN verdict: refuse + warn rather than walk back
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert analytics.feedback("correct") == 0
    assert any("feedback" in str(c.message) for c in caught)
    assert rows()[0].get("outcome") is None


def test_weather_exports_carry_the_drift_qualifier():
    """Regression: drift defaults to 1.0 with no baseline and the ASCII render
    showed 'insufficient history', but as_dict()/as_markdown() emitted the bare
    1.0 — machine consumers could not tell it from measured perfect stability."""
    import dataclasses
    from styxx.weather import WeatherReport

    req = [f.name for f in dataclasses.fields(WeatherReport)
           if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    kw = {n: (0 if n in ("n_entries", "current_streak") else
              (0.0 if n in ("gate_pass_rate", "warn_rate", "mean_confidence",
                            "hall_rate", "window_hours") else "x")) for n in req}
    r = WeatherReport(**kw)
    d = r.as_dict()
    assert d["drift_vs_yesterday"] == 1.0
    assert d["drift_label_yesterday"] == "insufficient history"
    assert d["drift_label_week"] == "insufficient history"
    assert "insufficient history" in r.as_markdown()


def test_preflight_discloses_a_silent_grounding_downgrade(monkeypatch):
    """Regression: passing correct_reference REQUESTS NLI grounding, but with no
    semantic backend it silently resolved to v0_fallback — deception dropped out
    of the composite and the gate while the returned object stayed shape-
    identical to a genuinely grounded run."""
    import warnings
    import styxx.guardrail.deception_v2 as deception_v2
    from styxx import preflight

    monkeypatch.setattr(deception_v2, "_has_sentence_transformers", lambda: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = preflight("q?", "the sky is green",
                      correct_reference="the sky is blue", persist=False)
    assert r.grounded is False
    assert r.deception_mode == "v0_fallback"
    assert "deception" not in r.composite_keys      # excluded from the composite
    assert any("correct_reference" in str(c.message) for c in caught)
    assert "NOT reference-grounded" in r.instructions
    assert r.as_dict()["grounded"] is False


def test_preflight_grounded_path_reports_grounded():
    import warnings
    from styxx import preflight
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = preflight("q?", "the sky is green",
                      correct_reference="the sky is blue", persist=False)
    if r.deception_mode in ("nli", "emb"):          # backend present in this env
        assert r.grounded is True
        assert "deception" in r.composite_keys
        assert not any("correct_reference" in str(c.message) for c in caught)


def test_forecast_refuses_to_name_a_category_without_data():
    """Regression: empty/absent trajectories became an all-zero feature vector —
    a valid point in feature space — so forecast() returned a real-looking
    'reasoning' at 0.695 confidence, risk low, manufactured from nothing."""
    from styxx.forecast import CognitiveForecaster, ForecastGate

    f = CognitiveForecaster.bootstrap()
    for traj in ({"entropy": [], "logprob": [], "top2_margin": []}, {}):
        r = f.forecast(traj)
        assert r.measured is False
        assert r.predicted_category == "unknown"
        assert r.confidence == 0.0
        assert r.as_dict()["measured"] is False
    # the gate must not read a fabricated verdict as a low-risk pass
    assert ForecastGate(f).check({"entropy": [], "logprob": [], "top2_margin": []}, 0) is None

    real = {"entropy": [0.5, 0.6, 0.7, 0.8, 0.9],
            "logprob": [-1.0, -1.2, -1.1, -1.3, -1.4],
            "top2_margin": [0.3, 0.2, 0.25, 0.2, 0.15]}
    r = f.forecast(real)
    assert r.measured is True and r.predicted_category != "unknown"


def test_dynamics_r2_is_undefined_on_degenerate_targets():
    """Regression: with zero target variance (a broken collector feeding constant
    vectors) r2 was hardcoded to 1.0 — perfect explained variance — so the exact
    metric the docstring tells callers to check read as maximal health."""
    import math
    import numpy as np
    from styxx.dynamics import CognitiveDynamics, Observation

    rng = np.random.default_rng(0)
    const = np.array([0.5] * 6)
    degenerate = [Observation(state_vec=rng.random(6), action_vec=rng.random(6),
                              next_state_vec=const.copy()) for _ in range(20)]
    r = CognitiveDynamics().fit(degenerate)
    assert math.isnan(r.r2)
    assert not (r.r2 > 0.9)          # the health test now fails instead of passing
    assert "nan" in repr(r)

    A = rng.random((6, 6))
    good = []
    for _ in range(30):
        s, a = rng.random(6), rng.random(6)
        good.append(Observation(state_vec=s, action_vec=a, next_state_vec=A @ s))
    assert CognitiveDynamics().fit(good).r2 > 0.99


def test_coherence_refuses_undefined_correlation_instead_of_reporting_zero():
    """Regression: Pearson r is UNDEFINED when a series is constant, but the
    guard returned 0.0 — asserting 'no relationship' for a hypothesis-bearing
    measurement that never happened (an absent cogn_composite defaulted to 0.0
    upstream, producing exactly that constant series)."""
    import pytest as _pytest
    from styxx.coherence import _pearson_r

    with _pytest.raises(ValueError, match="undefined"):
        _pearson_r([0.0] * 5, [0.1, 0.2, 0.3, 0.4, 0.5])
    # series with real variance are numerically unchanged (locked scorer intact)
    assert _pearson_r([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == 1.0
    assert _pearson_r([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == -1.0


def test_session_summary_does_not_drop_zero_confidence_readings(tmp_path, monkeypatch):
    """Found by styxx.absence, not by hand: session_summary carried the SAME
    two defects as check_health — `cv > 0` dropped exactly-zero readings (the
    worst ones) from the mean, and an empty window produced a fabricated 0.0
    indistinguishable from a genuinely terrible session."""
    import styxx.analytics as analytics

    monkeypatch.setattr(analytics, "load_audit", lambda **kw:
                        [{"gate": "pass", "phase4_conf": 0.0, "source": "live"}] * 3
                        + [{"gate": "pass", "phase4_conf": 0.8, "source": "live"}])
    s = analytics.session_summary()
    assert s is not None
    assert abs(s.mean_confidence - 0.2) < 1e-9, "zero readings must count"
    assert s.confidence_measured is True

    monkeypatch.setattr(analytics, "load_audit", lambda **kw:
                        [{"gate": "pass", "source": "live"}] * 3)
    s2 = analytics.session_summary()
    assert s2.confidence_measured is False        # unmeasured, disclosed
    assert "n/a" in repr(s2)


def test_coupling_refuses_an_undefined_correlation():
    """Also found by the screen: the count-vs-magnitude channel returned 0.0 when
    a magnitude vector was constant — r is UNDEFINED there — and fed that into
    its `shared` verdict, so an unmeasurable channel read as measured absence.
    The sibling guard three lines above already refused; now both do."""
    import numpy as np
    from styxx.coupling import _density_confound

    counts = np.array([1.0, 2.0, 3.0, 4.0])
    const = np.ones((4, 3))                        # constant magnitude vector
    varied = np.array([[1.0, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    out = _density_confound(const, varied, counts)
    assert out["applicable"] is False
    assert "undefined" in out["note"]


def test_verify_response_does_not_call_an_incomplete_measurement_valid():
    """`valid = gate != "fail"` called BOTH 'pending' (the trajectory never
    reached a verdict) and 'error' (scoring failed) valid — a measurement that
    never completed was indistinguishable from one that passed."""
    from styxx.cognometrics import tool_verify_response

    for gate in ("pending", "error"):
        out = tool_verify_response({"logprobs": [], "gate_override": gate}) \
            if False else None
    # drive it through the real payload path instead of a private override
    import styxx.cognometrics as cog
    payload = {"gate": "pending", "classification": "reasoning", "confidence": 0.9}
    monkey = cog._vitals_payload
    try:
        cog._vitals_payload = lambda v: dict(payload)
        r = cog.tool_verify_response({"logprobs": [0.1, 0.2, 0.3]})
        assert r["valid"] is False
        assert r["measured"] is False
        assert any("measurement_incomplete" in a for a in r["anomalies"])

        payload["gate"] = "pass"
        r2 = cog.tool_verify_response({"logprobs": [0.1, 0.2, 0.3]})
        assert r2["valid"] is True and r2["measured"] is True
    finally:
        cog._vitals_payload = monkey


def test_entropy_gate_refuses_a_failed_collection():
    """semantic_entropy's `< 2 samples -> 0.0` is a DOCUMENTED contract (and
    0.0 is its most confident reading: "one cluster, the model knows"). The
    measurement keeps it. The GATE built on it must not, or a failed resample
    scores maximal validity and the gate can never fail."""
    import pytest as _pytest
    from styxx.divergence import semantic_entropy
    from styxx.spec_exec import entropy_gate

    # contract intact
    assert semantic_entropy([], same_fn=lambda a, b: a == b) == 0.0
    assert semantic_entropy(["only one"], same_fn=lambda a, b: a == b) == 0.0

    # the gate refuses the same input
    with _pytest.raises(ValueError, match="failed collection"):
        entropy_gate([None, None])
    with _pytest.raises(ValueError):
        entropy_gate(["only one answer"])
    assert entropy_gate(["Paris.", "The capital is Paris.", "Lyon."]) > 0


def test_truthmap_on_an_empty_trajectory_is_not_a_calm_reading():
    from styxx.temperature import TruthMap

    empty = TruthMap.from_trajectories(entropy=[], logprob=[], top2_margin=[])
    assert empty.measured is False
    assert "NO TRAJECTORY" in empty.render()

    real = TruthMap.from_trajectories(entropy=[0.5, 0.6, 0.7],
                                      logprob=[-1.0, -1.2, -1.1],
                                      top2_margin=[0.3, 0.2, 0.25])
    assert real.measured is True and real.n_tokens == 3


def test_mcp_schemas_declare_every_field_their_handlers_read():
    """cogn_audit's schema was additionalProperties:False with only
    prompt/response — while the handler read `correct_reference`, the field
    that switches deception to NLI grounding and back INTO the composite. A
    strict MCP client literally could not reach the tool's headline
    capability."""
    import jsonschema
    from styxx.mcp.server import COGN_AUDIT_INPUT

    assert "correct_reference" in COGN_AUDIT_INPUT["properties"]
    jsonschema.validate({"prompt": "q", "response": "a",
                         "correct_reference": "ref"}, COGN_AUDIT_INPUT)


def test_cogn_audit_description_matches_what_the_composite_actually_contains():
    """The description claimed 'mean of first 3'; deception only joins the
    composite when a correct_reference grounds it."""
    from styxx.cognometrics import tool_cogn_audit

    ungrounded = tool_cogn_audit({"prompt": "q", "response": "the sky is green"})
    assert "deception" not in ungrounded["composite_keys"]
    grounded = tool_cogn_audit({"prompt": "q", "response": "the sky is green",
                                "correct_reference": "the sky is blue"})
    assert "deception" in grounded["composite_keys"]
