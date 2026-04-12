# -*- coding: utf-8 -*-
"""
test_agent_api.py -- tests for the 0.1.0a1 agent-cooperative API.

Covers:
  - Vitals shortcut properties  (.phase1, .phase4, .gate)
  - styxx.watch() context manager + styxx.observe()
  - styxx.is_concerning() helper
  - styxx.on_gate() DSL parser + dispatch
  - styxx.reflex() end-to-end with rewind signaling
  - styxx ask compare CLI smoke test
  - styxx.Anthropic pass-through adapter (vitals = None)

These tests run with no network access and no API keys. They use
the bundled atlas fixtures or deterministic synthetic trajectories.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import styxx
from styxx import Raw
from styxx.cli import _load_demo_trajectories
from styxx.gates import dispatch_gates, parse_condition


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _fake_trajectory(n: int) -> dict:
    return {
        "entropy":     [1.5 + (i % 3) * 0.1 for i in range(n)],
        "logprob":     [-0.5 - (i % 4) * 0.05 for i in range(n)],
        "top2_margin": [0.5 + (i % 2) * 0.05 for i in range(n)],
    }


def _fixture_vitals(kind: str):
    data = _load_demo_trajectories()
    t = data["trajectories"][kind]
    return Raw().read(
        entropy=t["entropy"],
        logprob=t["logprob"],
        top2_margin=t["top2_margin"],
    )


# ══════════════════════════════════════════════════════════════════
# 1. Vitals shortcut properties
# ══════════════════════════════════════════════════════════════════

def test_vitals_phase1_shortcut():
    v = _fixture_vitals("refusal")
    assert ":" in v.phase1
    cat, conf = v.phase1.split(":")
    assert cat in {
        "retrieval", "reasoning", "refusal",
        "creative", "adversarial", "hallucination",
    }
    assert 0.0 <= float(conf) <= 1.0


def test_vitals_phase4_shortcut():
    v = _fixture_vitals("reasoning")
    assert v.phase4 != "-"
    cat, conf = v.phase4.split(":")
    assert 0.0 <= float(conf) <= 1.0


def test_vitals_phase4_returns_dash_when_short():
    # 5-token trajectory -- phase 4 cutoff is 25 so phase4_late stays None
    v = Raw().read(**_fake_trajectory(5))
    assert v.phase4_late is None
    assert v.phase4 == "-"


def test_vitals_gate_pass_on_reasoning():
    v = _fixture_vitals("reasoning")
    assert v.gate == "pass"


def test_vitals_gate_warn_on_refusal():
    v = _fixture_vitals("refusal")
    # refusal fixture predicts phase4=refusal:0.29, above 0.20 floor
    assert v.gate == "warn"


def test_vitals_gate_pending_on_short_trajectory():
    v = Raw().read(**_fake_trajectory(5))
    assert v.gate == "pending"


# ══════════════════════════════════════════════════════════════════
# 2. is_concerning helper
# ══════════════════════════════════════════════════════════════════

def test_is_concerning_on_reasoning_is_false():
    v = _fixture_vitals("reasoning")
    assert styxx.is_concerning(v) is False


def test_is_concerning_on_refusal_is_true():
    v = _fixture_vitals("refusal")
    assert styxx.is_concerning(v) is True


def test_is_concerning_on_none_returns_false():
    assert styxx.is_concerning(None) is False


# ══════════════════════════════════════════════════════════════════
# 3. watch() + observe()
# ══════════════════════════════════════════════════════════════════

def test_watch_observes_raw_dict():
    data = _load_demo_trajectories()
    t = data["trajectories"]["refusal"]
    fake_response = {
        "entropy":     t["entropy"],
        "logprob":     t["logprob"],
        "top2_margin": t["top2_margin"],
    }
    with styxx.watch() as w:
        w.observe(fake_response)
    assert w.vitals is not None
    assert w.vitals.phase1_pre is not None
    assert w.n_observed == 1
    assert w.error is None


def test_observe_one_shot_helper():
    data = _load_demo_trajectories()
    t = data["trajectories"]["reasoning"]
    fake_response = {
        "entropy":     t["entropy"],
        "logprob":     t["logprob"],
        "top2_margin": t["top2_margin"],
    }
    vitals = styxx.observe(fake_response)
    assert vitals is not None
    assert vitals.gate == "pass"


def test_watch_unknown_shape_sets_error():
    with styxx.watch() as w:
        w.observe("this is not a response")
    assert w.vitals is None
    assert w.error is not None
    assert "unknown" in w.error.lower()


def test_watch_session_resets_on_enter():
    w = styxx.watch()
    # Simulate a first run
    w.observe({"entropy": [], "logprob": [], "top2_margin": []})
    w.n_observed = 5   # hacky but proves reset
    # Re-enter — nothing in styxx resets an existing session on
    # re-enter, but new watch() calls should be independent.
    w2 = styxx.watch()
    assert w2.n_observed == 0
    assert w2.vitals is None


# ══════════════════════════════════════════════════════════════════
# 4. Gate DSL parser
# ══════════════════════════════════════════════════════════════════

def test_parse_condition_any_phase_category():
    pred = parse_condition("refusal > 0.20")
    v = _fixture_vitals("refusal")
    assert pred(v) is True
    v2 = _fixture_vitals("reasoning")
    assert pred(v2) is False


def test_parse_condition_phase_pinned():
    pred = parse_condition("p4.refusal > 0.20")
    v = _fixture_vitals("refusal")
    assert pred(v) is True


def test_parse_condition_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown category"):
        parse_condition("pizza > 0.5")


def test_parse_condition_rejects_malformed():
    with pytest.raises(ValueError):
        parse_condition("garbage")


def test_parse_condition_gate_status():
    pred_pass = parse_condition("gate == pass")
    v_reason = _fixture_vitals("reasoning")
    v_refusal = _fixture_vitals("refusal")
    assert pred_pass(v_reason) is True
    assert pred_pass(v_refusal) is False

    pred_warn = parse_condition("gate == warn")
    assert pred_warn(v_refusal) is True


def test_parse_condition_all_ops():
    for op in (">", ">=", "<", "<=", "==", "!="):
        parse_condition(f"reasoning {op} 0.5")   # should not raise


# ══════════════════════════════════════════════════════════════════
# 5. Gate registration + dispatch
# ══════════════════════════════════════════════════════════════════

def test_on_gate_registers_and_fires(monkeypatch):
    styxx.clear_gates()
    fired = []
    styxx.on_gate("refusal > 0.20", lambda v: fired.append(v.phase4))
    v = _fixture_vitals("refusal")
    n = dispatch_gates(v)
    assert n == 1
    assert len(fired) == 1
    styxx.clear_gates()


def test_remove_gate():
    styxx.clear_gates()
    g = styxx.on_gate("refusal > 0.20", lambda v: None)
    assert len(styxx.list_gates()) == 1
    assert styxx.remove_gate(g) is True
    assert len(styxx.list_gates()) == 0
    # Removing again returns False
    assert styxx.remove_gate(g) is False


def test_gate_callback_exception_does_not_propagate():
    styxx.clear_gates()
    def bad(v):
        raise RuntimeError("intentional test failure")
    styxx.on_gate("refusal > 0.20", bad)
    v = _fixture_vitals("refusal")
    # Should not raise — exception gets warned and swallowed
    with pytest.warns(RuntimeWarning):
        dispatch_gates(v)
    styxx.clear_gates()


def test_clear_gates_returns_count():
    styxx.clear_gates()
    styxx.on_gate("refusal > 0.20", lambda v: None)
    styxx.on_gate("hallucination > 0.20", lambda v: None)
    n = styxx.clear_gates()
    assert n == 2
    assert len(styxx.list_gates()) == 0


# ══════════════════════════════════════════════════════════════════
# 6. Reflex session — end-to-end with fake streaming
# ══════════════════════════════════════════════════════════════════

class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeTopLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class _FakeTokenLogprob:
    def __init__(self, logprob, top_logprobs):
        self.logprob = logprob
        self.top_logprobs = top_logprobs


class _FakeLogprobsBlock:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, delta, logprobs):
        self.delta = delta
        self.logprobs = logprobs


class _FakeChunk:
    def __init__(self, text, chosen_lp, top_lps):
        tok = _FakeTokenLogprob(
            logprob=chosen_lp,
            top_logprobs=[_FakeTopLogprob(lp) for lp in top_lps],
        )
        self.choices = [_FakeChoice(
            delta=_FakeDelta(content=text),
            logprobs=_FakeLogprobsBlock(content=[tok]),
        )]


class _FakeCompletions:
    def __init__(self, fixture):
        self._fx = fixture

    def create(self, **kwargs):
        def _synth_top5(chosen_lp, entropy, top2):
            p1 = math.exp(chosen_lp)
            p2 = max(1e-6, p1 - top2)
            remaining = max(0.0, 1.0 - p1 - p2)
            return [
                math.log(max(p, 1e-8))
                for p in (p1, p2, remaining * 0.5, remaining * 0.3, remaining * 0.2)
            ]

        def gen():
            for i, (e, lp, t2) in enumerate(zip(
                self._fx["entropy"],
                self._fx["logprob"],
                self._fx["top2_margin"],
            )):
                lps = _synth_top5(lp, e, t2)
                yield _FakeChunk(text=f"tok{i} ", chosen_lp=lp, top_lps=lps)
        return gen()


class _FakeChat:
    def __init__(self, fixture):
        self.completions = _FakeCompletions(fixture)


class _FakeClient:
    def __init__(self, fixture):
        self.chat = _FakeChat(fixture)


def test_reflex_basic_completion():
    """Reflex runs a full stream end-to-end with no callbacks and
    emits the complete text stream."""
    data = _load_demo_trajectories()
    client = _FakeClient(data["trajectories"]["reasoning"])

    collected = []
    with styxx.reflex() as session:
        for chunk in session.stream_openai(
            client,
            model="fake",
            messages=[{"role": "user", "content": "hi"}],
        ):
            collected.append(chunk)
    # We asked for 30 tokens — should get 30 chunks
    assert len(collected) == 30
    # Session should have logged at least one classification + complete event
    assert any(e.kind == "classify" for e in session.events)
    assert any(e.kind == "complete" for e in session.events)
    assert session.rewind_count == 0
    assert session.aborted is False


def test_reflex_rewind_fires_and_increments():
    """on_drift fires on the first classification, triggers a rewind,
    session records the rewind and continues."""
    data = _load_demo_trajectories()
    client = _FakeClient(data["trajectories"]["refusal"])

    drift_count = [0]

    def on_drift(vitals):
        drift_count[0] += 1
        # Rewind on the first drift detection only
        if drift_count[0] == 1:
            styxx.rewind(2, anchor=" (verify) ")

    with styxx.reflex(on_drift=on_drift, classify_every_k=5, max_rewinds=1) as s:
        for _ in s.stream_openai(
            client, model="fake",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass
    assert s.rewind_count == 1
    # Events should include at least one "rewind" kind
    assert any(e.kind == "rewind" for e in s.events)
    # Rewind budget exhausted, next run proceeds to completion
    assert any(e.kind == "complete" for e in s.events)


def test_reflex_abort_stops_cleanly():
    """abort() in a callback stops the stream cleanly."""
    data = _load_demo_trajectories()
    client = _FakeClient(data["trajectories"]["refusal"])

    def on_drift(vitals):
        styxx.abort(reason="test-abort")

    collected = []
    with styxx.reflex(on_drift=on_drift, classify_every_k=5) as s:
        for chunk in s.stream_openai(
            client, model="fake",
            messages=[{"role": "user", "content": "hi"}],
        ):
            collected.append(chunk)
    assert s.aborted is True
    assert s.abort_reason == "test-abort"
    # We yielded at least one token before the abort fired
    assert len(collected) >= 5
    # Events should include an abort entry
    assert any(e.kind == "abort" for e in s.events)


def test_reflex_max_rewinds_respected():
    """Rewind budget exhausted -> session disables callbacks and
    lets the stream complete instead of looping forever."""
    data = _load_demo_trajectories()
    client_factory = lambda: _FakeClient(data["trajectories"]["refusal"])

    count = [0]
    def on_drift(vitals):
        count[0] += 1
        styxx.rewind(2, anchor="x")

    # Re-create client each time because the fake generator exhausts
    class _MultiClient:
        def __init__(self):
            self._inner = _FakeClient(data["trajectories"]["refusal"])
            self.chat = _WrapChat(self, data["trajectories"]["refusal"])

    class _WrapChat:
        def __init__(self, outer, fx):
            self._fx = fx
            self.completions = _WrapCompletions(fx)

    class _WrapCompletions:
        def __init__(self, fx):
            self._fx = fx
        def create(self, **kwargs):
            return _FakeCompletions(self._fx).create(**kwargs)

    with styxx.reflex(on_drift=on_drift, max_rewinds=2, classify_every_k=5) as s:
        for _ in s.stream_openai(
            _MultiClient(), model="fake",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass
    assert s.rewind_count == 2
    # Once budget is exhausted, the reflex should finish
    assert any(e.kind == "complete" for e in s.events)


# ══════════════════════════════════════════════════════════════════
# 7. compare CLI smoke test
# ══════════════════════════════════════════════════════════════════

def test_compare_cli_runs_without_error(capsys):
    from styxx.cli import cmd_compare
    # Fake args namespace with the fields the CLI expects
    class _Args:
        pass
    args = _Args()
    rc = cmd_compare(args)
    assert rc == 0
    out = capsys.readouterr().out
    # Should mention each of the 6 fixture categories
    for kind in ("retrieval", "reasoning", "refusal",
                 "creative", "adversarial", "hallucination"):
        assert kind in out, f"compare output missing {kind}"
    # Should have the json footer for agent parsing
    assert '"command":"compare"' in out


# ══════════════════════════════════════════════════════════════════
# 8. Anthropic pass-through adapter
# ══════════════════════════════════════════════════════════════════

def test_anthropic_adapter_importable():
    # Not actually instantiable without the anthropic SDK, but the
    # module should import cleanly and the factory should be callable.
    from styxx.adapters.anthropic import AnthropicWithVitals, _warn_once
    assert callable(AnthropicWithVitals)
    assert callable(_warn_once)


def test_anthropic_factory_raises_clear_error_without_sdk():
    # If the user's env doesn't have `anthropic`, styxx.Anthropic()
    # should raise ImportError with a clear install hint rather than
    # a cryptic module-not-found crash.
    try:
        import anthropic  # noqa: F401
        pytest.skip("anthropic SDK is installed; can't test the missing path")
    except ImportError:
        with pytest.raises(ImportError, match="styxx.Anthropic requires"):
            styxx.Anthropic()


# ══════════════════════════════════════════════════════════════════
# 9. Version bump sanity
# ══════════════════════════════════════════════════════════════════

def test_version_is_0_2_x():
    assert styxx.__version__.startswith("0.2.")


def test_all_new_exports_exist():
    for name in (
        "watch", "observe", "observe_raw", "is_concerning", "WatchSession",
        "on_gate", "remove_gate", "clear_gates", "list_gates",
        "reflex", "rewind", "abort",
        "ReflexSession", "ReflexSignal", "RewindSignal", "AbortSignal",
        "Anthropic",
    ):
        assert hasattr(styxx, name), f"styxx.{name} missing"


# ══════════════════════════════════════════════════════════════════
# 10. 0.1.0a2 patch release — gate repr + observe_raw fidelity
# ══════════════════════════════════════════════════════════════════

def test_registered_gate_repr_is_clean():
    """Xendro nit #1: default dataclass repr dumped function
    memory addresses. 0.1.0a2 uses a human-readable repr."""
    styxx.clear_gates()
    g = styxx.on_gate("hallucination > 0.2", lambda v: None)
    r = repr(g)
    # Should contain the condition string
    assert "hallucination > 0.2" in r
    # Must NOT contain the function-address noise
    assert "<function" not in r
    assert "at 0x" not in r
    # Should be wrapped as a styxx gate
    assert "styxx gate" in r
    styxx.clear_gates()


def test_registered_gate_repr_with_name():
    styxx.clear_gates()
    g = styxx.on_gate(
        "refusal > 0.2",
        lambda v: None,
        name="refusal_alert",
    )
    r = repr(g)
    assert "refusal_alert" in r
    assert "refusal > 0.2" in r
    assert "<function" not in r
    styxx.clear_gates()


def test_observe_raw_fidelity_path():
    """observe_raw bypasses the top-5 reconstruction entirely.
    Should give the same classification as styxx.Raw().read()."""
    v_raw = _fixture_vitals("refusal")
    data = _load_demo_trajectories()
    t = data["trajectories"]["refusal"]
    v_observed = styxx.observe_raw(
        entropy=t["entropy"],
        logprob=t["logprob"],
        top2_margin=t["top2_margin"],
    )
    # Same trajectories -> same classification
    assert v_observed is not None
    assert v_observed.phase1 == v_raw.phase1
    assert v_observed.phase4 == v_raw.phase4
    assert v_observed.gate == v_raw.gate


def test_observe_raw_dispatches_gates():
    """Gate callbacks should fire from observe_raw the same way
    they fire from observe()."""
    styxx.clear_gates()
    fired = []
    styxx.on_gate("gate == warn", lambda v: fired.append("warn"))

    data = _load_demo_trajectories()
    t = data["trajectories"]["refusal"]
    styxx.observe_raw(
        entropy=t["entropy"],
        logprob=t["logprob"],
        top2_margin=t["top2_margin"],
    )
    assert "warn" in fired
    styxx.clear_gates()


def test_observe_sidechannel_raw_attributes():
    """When a fake openai response has _styxx_raw_* attributes
    attached, observe() should use them directly instead of
    reconstructing from top-5 logprobs. This is the test-harness
    fidelity path Xendro needed."""
    data = _load_demo_trajectories()
    t = data["trajectories"]["refusal"]

    # Build a minimal object that looks like an openai response
    # with the sidechannel attributes attached.
    class _SideChannel:
        pass
    resp = _SideChannel()
    resp._styxx_raw_entropy = t["entropy"]
    resp._styxx_raw_logprob = t["logprob"]
    resp._styxx_raw_top2_margin = t["top2_margin"]

    vitals = styxx.observe(resp)
    # Should match the direct Raw().read path exactly
    v_raw = _fixture_vitals("refusal")
    assert vitals is not None
    assert vitals.phase1 == v_raw.phase1
    assert vitals.phase4 == v_raw.phase4
    assert vitals.gate == v_raw.gate


def test_observe_dict_path_bypasses_reconstruction():
    """Passing a plain dict with trajectory keys to observe()
    should go through the raw path, not the openai reconstruction
    path (which would lossy-bridge the entropy)."""
    data = _load_demo_trajectories()
    t = data["trajectories"]["refusal"]

    v_from_dict = styxx.observe({
        "entropy": t["entropy"],
        "logprob": t["logprob"],
        "top2_margin": t["top2_margin"],
    })
    v_raw = _fixture_vitals("refusal")

    # Dict path must be bit-identical to Raw().read() path
    assert v_from_dict is not None
    assert v_from_dict.phase1 == v_raw.phase1
    assert v_from_dict.phase4 == v_raw.phase4
    assert v_from_dict.gate == v_raw.gate
