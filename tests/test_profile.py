"""Tests for styxx.profile — the flagship @styxx.profile cognometric profiler.

profile() is the first example in the README, yet its fault-injection,
phase-transition, and export machinery were almost entirely uncovered. These
tests exercise that machinery directly.

profile.py consumes vitals *structurally* (getattr of category/confidence/
trust_score/coherence + a to_dict()), not via the concrete styxx.vitals.Vitals
dataclass (which requires PhaseReading objects). So we drive it with a small
duck-typed fake that lets each fault threshold be hit precisely. The one record()
path that does `isinstance(v, Vitals)` is covered by monkeypatching profile.Vitals
to the fake.
"""

import importlib
import json

import pytest

import styxx

# `styxx.profile` the *attribute* is the decorator function (the package re-exports
# it), so reach the module object explicitly for its internals (_get_stack,
# _install_tap, Vitals, the profile() callable as a module attr, ...).
P = importlib.import_module("styxx.profile")
from styxx.profile import (
    CognitiveProfile,
    Fault,
    ProfileStep,
    _detect_faults_for_step,
    _detect_phase_transitions,
    _extract_text,
    K_DRIFT,
    K_CONFAB,
    K_REFUSAL,
    K_SYCOPHANT,
    K_PHASE_TRANSITION,
    K_LOW_TRUST,
    K_INCOHERENCE,
)


class FakeVitals:
    """Structural stand-in for Vitals as profile.py reads it.

    trust_score defaults to 1.0 so a vitals only triggers K_LOW_TRUST when a test
    explicitly asks for it — the detector reads getattr(v,'trust_score',0.0),
    which would otherwise fire low-trust on every fake.
    """

    def __init__(self, category="reasoning", confidence=0.0,
                 trust_score=1.0, coherence=None, gate="pass"):
        self.category = category
        self.confidence = confidence
        self.trust_score = trust_score
        self.coherence = coherence
        self.gate = gate

    def to_dict(self):
        return {
            "category": self.category,
            "confidence": self.confidence,
            "trust": self.trust_score,
            "coherence": self.coherence,
            "gate": self.gate,
        }


def fv(**kw):
    return FakeVitals(**kw)


# ──────────────────────────────────────────────────────────────────────
# _extract_text — all four SDK shapes + None
# ──────────────────────────────────────────────────────────────────────

def test_extract_text_none():
    assert _extract_text(None) is None


def test_extract_text_plain_string():
    assert _extract_text("hello world") == "hello world"


def test_extract_text_openai_choices_shape():
    class Msg:
        content = "chat answer"

    class Choice:
        message = Msg()

    class Resp:
        choices = [Choice()]

    assert _extract_text(Resp()) == "chat answer"


def test_extract_text_anthropic_content_list():
    class Part:
        def __init__(self, text):
            self.text = text

    class Resp:
        content = [Part("foo"), Part("bar")]

    assert _extract_text(Resp()) == "foobar"


def test_extract_text_dict_text_key():
    assert _extract_text({"text": "via text key"}) == "via text key"


def test_extract_text_dict_content_key():
    assert _extract_text({"content": "via content key"}) == "via content key"


def test_extract_text_unknown_shape_returns_none():
    assert _extract_text(object()) is None


# ──────────────────────────────────────────────────────────────────────
# _detect_faults_for_step — every kind + thresholds
# ──────────────────────────────────────────────────────────────────────

def test_no_vitals_yields_no_faults():
    assert _detect_faults_for_step(0, None) == []


def test_clean_vitals_yields_no_faults():
    faults = _detect_faults_for_step(0, fv(category="reasoning", confidence=0.9))
    assert faults == []


@pytest.mark.parametrize("category", ["tool_arg_drift", "drift", "tool_confab", "arg_swap"])
def test_drift_detected_above_half(category):
    faults = _detect_faults_for_step(3, fv(category=category, confidence=0.7))
    assert [f.kind for f in faults] == [K_DRIFT]
    assert faults[0].step_index == 3
    assert faults[0].severity == pytest.approx(0.7)


def test_drift_not_detected_at_or_below_half():
    assert _detect_faults_for_step(0, fv(category="drift", confidence=0.5)) == []


@pytest.mark.parametrize("category", ["confab", "confabulation", "hallucination", "fabrication"])
def test_confab_detected(category):
    faults = _detect_faults_for_step(0, fv(category=category, confidence=0.6))
    assert [f.kind for f in faults] == [K_CONFAB]


@pytest.mark.parametrize("category", ["sycophant", "sycophancy"])
def test_sycophant_detected(category):
    faults = _detect_faults_for_step(0, fv(category=category, confidence=0.55))
    assert [f.kind for f in faults] == [K_SYCOPHANT]


def test_refusal_needs_high_confidence():
    # Refusal fires only above 0.8 (it's often informational below that).
    assert _detect_faults_for_step(0, fv(category="refusal", confidence=0.9))[0].kind == K_REFUSAL
    assert _detect_faults_for_step(0, fv(category="refusal", confidence=0.8)) == []
    assert _detect_faults_for_step(0, fv(category="refusal", confidence=0.7)) == []


def test_low_trust_detected_and_severity():
    faults = _detect_faults_for_step(2, fv(category="reasoning", confidence=0.1, trust_score=0.2))
    assert [f.kind for f in faults] == [K_LOW_TRUST]
    assert faults[0].severity == pytest.approx(0.8)  # 1.0 - 0.2


def test_low_trust_boundary_not_fired_at_threshold():
    assert _detect_faults_for_step(0, fv(confidence=0.1, trust_score=0.3)) == []


def test_incoherence_detected_and_severity():
    faults = _detect_faults_for_step(0, fv(confidence=0.1, coherence=0.25))
    assert [f.kind for f in faults] == [K_INCOHERENCE]
    assert faults[0].severity == pytest.approx(0.75)  # 1.0 - 0.25


def test_incoherence_none_coherence_is_skipped():
    assert _detect_faults_for_step(0, fv(confidence=0.1, coherence=None)) == []


def test_incoherence_non_numeric_coherence_is_swallowed():
    assert _detect_faults_for_step(0, fv(confidence=0.1, coherence="n/a")) == []


def test_snapshot_captured_in_fault():
    f = _detect_faults_for_step(0, fv(category="drift", confidence=0.9))[0]
    assert f.vitals_snapshot is not None
    assert f.vitals_snapshot["category"] == "drift"


def test_snapshot_none_when_to_dict_raises():
    class Boom(FakeVitals):
        def to_dict(self):
            raise RuntimeError("no snapshot")

    f = _detect_faults_for_step(0, Boom(category="drift", confidence=0.9))[0]
    assert f.vitals_snapshot is None


def test_multiple_faults_one_step():
    # low trust AND incoherence on the same step.
    faults = _detect_faults_for_step(0, fv(confidence=0.1, trust_score=0.1, coherence=0.1))
    kinds = {f.kind for f in faults}
    assert kinds == {K_LOW_TRUST, K_INCOHERENCE}


# ──────────────────────────────────────────────────────────────────────
# _detect_phase_transitions
# ──────────────────────────────────────────────────────────────────────

def _step(index, category):
    return ProfileStep(index=index, label=f"s{index}", started_ts=0.0,
                       vitals=(fv(category=category) if category is not None else None))


def test_phase_transition_flagged_on_category_shift():
    steps = [_step(0, "reasoning"), _step(1, "refusal")]
    faults = _detect_phase_transitions(steps)
    assert [f.kind for f in faults] == [K_PHASE_TRANSITION]
    assert faults[0].step_index == 1
    assert faults[0].severity == 0.5


def test_no_phase_transition_when_category_stable():
    steps = [_step(0, "reasoning"), _step(1, "reasoning")]
    assert _detect_phase_transitions(steps) == []


def test_none_category_does_not_break_chain():
    # A step with no vitals/category must not be treated as a transition; the
    # previous non-None category carries across the gap.
    steps = [_step(0, "reasoning"), _step(1, None), _step(2, "reasoning")]
    assert _detect_phase_transitions(steps) == []
    steps2 = [_step(0, "reasoning"), _step(1, None), _step(2, "refusal")]
    assert [f.step_index for f in _detect_phase_transitions(steps2)] == [2]


# ──────────────────────────────────────────────────────────────────────
# Fault / ProfileStep dataclasses
# ──────────────────────────────────────────────────────────────────────

def test_fault_str_and_to_dict():
    f = Fault(1, K_DRIFT, 0.73, "because reasons", {"category": "drift"})
    s = str(f)
    assert "drift" in s and "step=1" in s and "0.73" in s
    d = f.to_dict()
    assert d["step_index"] == 1 and d["kind"] == K_DRIFT
    assert d["vitals_snapshot"] == {"category": "drift"}


def test_profilestep_to_dict_truncates_long_text():
    step = ProfileStep(index=0, label="x", started_ts=0.0,
                       prompt="p" * 1000, response_text="r" * 1000,
                       vitals=fv(category="reasoning"))
    d = step.to_dict()
    assert len(d["prompt"]) == 500
    assert len(d["response_text"]) == 500
    assert d["vitals"]["category"] == "reasoning"


def test_profilestep_to_dict_vitals_none_safe():
    d = ProfileStep(index=0, label="x", started_ts=0.0).to_dict()
    assert d["vitals"] is None
    assert d["prompt"] is None and d["response_text"] is None


# ──────────────────────────────────────────────────────────────────────
# CognitiveProfile.record / finish
# ──────────────────────────────────────────────────────────────────────

def test_record_with_explicit_vitals_creates_step_and_fault():
    p = CognitiveProfile(name="t")
    step = p.record("the response text", label="plan", vitals=fv(category="drift", confidence=0.9))
    assert isinstance(step, ProfileStep)
    assert step.label == "plan"
    assert step.response_text == "the response text"
    assert len(p) == 1
    assert [f.kind for f in p.faults] == [K_DRIFT]


def test_record_auto_labels_steps():
    p = CognitiveProfile(name="t")
    p.record(vitals=fv(), label=None)
    p.record(vitals=fv(), label=None)
    assert [s.label for s in p] == ["step_0", "step_1"]


def test_record_picks_up_vitals_attached_to_response(monkeypatch):
    # Path (1): record() reads response.vitals when it isinstance(Vitals). Point
    # profile.Vitals at the fake so the isinstance check matches.
    monkeypatch.setattr(P, "Vitals", FakeVitals)

    class Resp:
        vitals = fv(category="hallucination", confidence=0.9)
        # no .choices / .content -> _extract_text returns None, fine

    p = CognitiveProfile(name="t")
    p.record(Resp())
    assert p.steps[0].vitals is Resp.vitals
    assert [f.kind for f in p.faults] == [K_CONFAB]


def test_record_observe_fallback(monkeypatch):
    # Path (2): no explicit vitals, response has no .vitals -> observe() is tried.
    watch = importlib.import_module("styxx.watch")
    sentinel = fv(category="refusal", confidence=0.95)
    monkeypatch.setattr(watch, "observe", lambda resp, prompt=None: sentinel)
    p = CognitiveProfile(name="t")
    p.record("some text needing observation")
    assert p.steps[0].vitals is sentinel
    assert [f.kind for f in p.faults] == [K_REFUSAL]


def test_record_observe_failure_is_swallowed(monkeypatch):
    watch = importlib.import_module("styxx.watch")

    def boom(resp, prompt=None):
        raise RuntimeError("observe down")

    monkeypatch.setattr(watch, "observe", boom)
    p = CognitiveProfile(name="t")
    step = p.record("text")  # must not raise
    assert step.vitals is None
    assert p.faults == []


def test_finish_is_idempotent():
    p = CognitiveProfile(name="t")
    p.record(vitals=fv())
    p.finish()
    ts = p.finished_ts
    p.finish()
    assert p.finished_ts == ts
    assert p._finished is True


def test_finish_runs_phase_transition_detection():
    p = CognitiveProfile(name="t")
    p.record(vitals=fv(category="reasoning"))
    p.record(vitals=fv(category="refusal", confidence=0.1))  # low conf: no refusal fault
    assert not any(f.kind == K_PHASE_TRANSITION for f in p.faults)
    p.finish()
    assert any(f.kind == K_PHASE_TRANSITION for f in p.faults)


# ──────────────────────────────────────────────────────────────────────
# Views: summary, duration_s, __len__/__iter__/__repr__
# ──────────────────────────────────────────────────────────────────────

def test_summary_empty_profile():
    assert "0 steps observed" in CognitiveProfile(name="empty").summary


def test_summary_no_faults():
    p = CognitiveProfile(name="clean")
    p.record(vitals=fv(category="reasoning", confidence=0.9))
    assert "no faults detected" in p.summary


def test_summary_dedupes_by_kind_and_step():
    p = CognitiveProfile(name="t")
    p.record(vitals=fv(category="drift", confidence=0.9))
    # Manually inject a duplicate (kind, step_index) fault — summary must collapse it.
    p.faults.append(Fault(0, K_DRIFT, 0.99, "dup", None))
    s = p.summary
    assert "1 fault(s):" in s


def test_summary_caps_at_five_with_overflow_note():
    p = CognitiveProfile(name="t")
    # 6 distinct faults across 6 steps.
    for i in range(6):
        p.faults.append(Fault(i, K_DRIFT, 0.5 + i * 0.05, f"f{i}", None))
        p.steps.append(ProfileStep(index=i, label=f"s{i}", started_ts=0.0))
    s = p.summary
    assert "6 fault(s):" in s
    assert "and 1 more" in s


def test_len_and_iter():
    p = CognitiveProfile(name="t")
    p.record(vitals=fv())
    p.record(vitals=fv())
    assert len(p) == 2
    assert [s.index for s in p] == [0, 1]


def test_repr_mentions_counts():
    p = CognitiveProfile(name="agentx")
    p.record(vitals=fv(category="drift", confidence=0.9))
    r = repr(p)
    assert "agentx" in r and "steps=1" in r


def test_duration_s_nonnegative_before_and_after_finish():
    p = CognitiveProfile(name="t")
    assert p.duration_s >= 0.0
    p.finish()
    assert p.duration_s >= 0.0


# ──────────────────────────────────────────────────────────────────────
# Exports: to_dict / to_json / to_html / to_langsmith / to_datadog
# ──────────────────────────────────────────────────────────────────────

def _profile_with_one_faulted_step():
    p = CognitiveProfile(name="exp")
    p.record("resp", label="call0", prompt="do the thing",
             vitals=fv(category="drift", confidence=0.9))
    p.finish()
    return p


def test_to_dict_shape():
    p = _profile_with_one_faulted_step()
    d = p.to_dict()
    assert d["name"] == "exp"
    assert d["n_steps"] == 1
    assert d["n_faults"] >= 1
    assert isinstance(d["steps"], list) and isinstance(d["faults"], list)
    assert d["steps"][0]["label"] == "call0"


def test_to_json_returns_string_without_path():
    p = _profile_with_one_faulted_step()
    out = p.to_json()
    parsed = json.loads(out)  # must be valid JSON
    assert parsed["name"] == "exp"


def test_to_json_writes_file_and_returns_string(tmp_path):
    p = _profile_with_one_faulted_step()
    dest = tmp_path / "run.json"
    out = p.to_json(dest)
    assert dest.exists()
    assert json.loads(dest.read_text(encoding="utf-8"))["name"] == "exp"
    assert json.loads(out)["name"] == "exp"  # also returns the string


def test_to_html_returns_markup_and_writes_file(tmp_path):
    p = _profile_with_one_faulted_step()
    html = p.to_html()
    assert isinstance(html, str) and "<" in html and len(html) > 0
    dest = tmp_path / "run.html"
    p.to_html(dest)
    assert dest.exists() and dest.read_text(encoding="utf-8").strip().startswith("<")


def test_to_langsmith_structure():
    p = _profile_with_one_faulted_step()
    ls = p.to_langsmith()
    assert ls["name"] == "exp"
    assert ls["run_type"] == "chain"
    assert len(ls["child_runs"]) == 1
    child = ls["child_runs"][0]
    assert child["run_type"] == "llm"
    assert child["inputs"]["prompt"] == "do the thing"
    assert child["outputs"]["output"] == "resp"
    # the step's fault must be attached to its child span
    assert child["extra"]["styxx_faults"]
    assert child["extra"]["styxx"]["category"] == "drift"


def test_to_datadog_structure_and_nanoseconds():
    p = _profile_with_one_faulted_step()
    dd = p.to_datadog()
    assert "spans" in dd and len(dd["spans"]) == 1
    span = dd["spans"][0]
    assert span["name"] == "llm.call0"
    assert isinstance(span["trace_id"], int)
    assert isinstance(span["start"], int)  # nanoseconds
    assert span["meta"]["styxx.category"] == "drift"
    assert span["meta"]["styxx.faulted"] == "true"
    assert "styxx.confidence" in span["metrics"]


# ──────────────────────────────────────────────────────────────────────
# Public decorator / context-manager surface
# ──────────────────────────────────────────────────────────────────────

def test_bare_decorator_returns_result_and_profile():
    @P.profile
    def my_agent(task):
        return f"did:{task}"

    result, p = my_agent("summarize")
    assert result == "did:summarize"
    assert isinstance(p, CognitiveProfile)
    assert p.name == "my_agent"  # inferred from func name
    assert p._finished is True


def test_parametric_decorator_positional_name():
    @P.profile("sql_agent")
    def run(task):
        return task

    result, p = run("q")
    assert result == "q"
    assert p.name == "sql_agent"


def test_parametric_decorator_keyword_name():
    @P.profile(name="kw_agent", auto_hook=False)
    def run(task):
        return task

    _, p = run("q")
    assert p.name == "kw_agent"


def test_context_manager_form():
    with P.profile(name="ctx_agent", auto_hook=False) as p:
        assert isinstance(p, CognitiveProfile)
        p.record(vitals=fv(category="drift", confidence=0.9))
    assert p.name == "ctx_agent"
    assert p._finished is True
    assert any(f.kind == K_DRIFT for f in p.faults)


def test_noarg_context_manager_defaults_to_agent():
    with P.profile(auto_hook=False) as p:
        pass
    assert p.name == "agent"
    assert p._finished is True


def test_decorator_pops_profile_even_on_exception():
    before = len(P._get_stack())

    @P.profile(name="boom", auto_hook=False)
    def explode(_):
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        explode("x")
    # stack must be unwound — no leaked active profile.
    assert len(P._get_stack()) == before


def test_profile_session_factory_then_manual_record():
    p = styxx.profile_session(name="manual")
    assert isinstance(p, CognitiveProfile)
    p.record(vitals=fv(category="drift", confidence=0.9), label="a")
    p.record(vitals=fv(category="reasoning", confidence=0.9), label="b")
    p.finish()
    assert len(p) == 2
    assert any(f.kind == K_DRIFT for f in p.faults)


# ──────────────────────────────────────────────────────────────────────
# Tap installation
# ──────────────────────────────────────────────────────────────────────

def test_install_tap_is_idempotent():
    from styxx import analytics
    P._install_tap()
    first = analytics.write_audit
    P._install_tap()
    assert analytics.write_audit is first  # not re-patched
    assert P._tap_installed is True
