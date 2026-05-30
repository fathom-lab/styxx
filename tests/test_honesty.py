"""Tests for the unifying honesty runtime (styxx.honesty)."""
from __future__ import annotations

from styxx.honesty import honest, HonestyVerdict
from styxx.single_pass import SinglePassCalibration


class _RV:  # a RetrievalVerdict-like object (duck-typed on .verdict)
    def __init__(self, verdict):
        self.verdict = verdict


# --- the logit gates ---------------------------------------------------------

def test_span_gate_fires_abstains():
    v = honest("42", span_logits=[[1.0, 1.0]], entropy_threshold=0.1)
    assert v.action == "abstained" and v.abstained is True
    assert v.answer == "I'm not sure." and v.method == "span"
    assert bool(v) is False


def test_span_gate_clear_answers():
    v = honest("42", span_logits=[[10.0, 0.0]], entropy_threshold=5.0)
    assert v.action == "answered" and v.answer == "42"
    assert bool(v) is True


def test_single_pass_gate_fires():
    v = honest("42", logits=[1.0, 1.0], entropy_threshold=0.1)
    assert v.abstained is True and v.method == "single_pass"


def test_calibration_supplies_threshold():
    cal = SinglePassCalibration(entropy_threshold=0.1, auc=0.9, confab_mean=0.6,
                                correct_mean=0.0, n_confab=5, n_correct=5)
    v = honest("42", logits=[1.0, 1.0], calibration=cal)
    assert v.abstained is True


def test_logit_signal_without_threshold_cannot_fire():
    # no calibration / threshold -> the gate stays quiet (advisory), answer passes
    v = honest("42", logits=[1.0, 1.0])
    assert v.action == "answered" and v.method == "single_pass"
    assert v.signal is not None        # the signal is still surfaced


# --- the confidence gate (frontier models) -----------------------------------

def test_confidence_below_floor_abstains():
    v = honest("42", confidence=0.3, confidence_floor=0.5)
    assert v.abstained is True and v.method == "confidence"
    assert abs(v.signal - 0.7) < 1e-9 and abs(v.confidence - 0.3) < 1e-9


def test_confidence_above_floor_answers():
    v = honest("42", confidence=0.9, confidence_floor=0.5)
    assert v.action == "answered" and v.confidence == 0.9


# --- the calibrated text engine ----------------------------------------------

class _V:  # a guardrail.Verdict-like object
    def __init__(self, risk, threshold, action):
        self.risk, self.threshold, self.action = risk, threshold, action


def test_engine_callable_abstains_on_high_risk():
    v = honest("Berlin", prompt="capital of France?",
               engine=lambda p, a: _V(0.98, 0.4, "halt"))
    assert v.abstained is True and v.method == "engine"
    assert abs(v.signal - 0.98) < 1e-9 and abs(v.confidence - 0.02) < 1e-9


def test_engine_callable_answers_on_low_risk():
    v = honest("Paris", prompt="capital of France?",
               engine=lambda p, a: _V(0.05, 0.4, "allow"))
    assert v.action == "answered" and v.method == "engine"


def test_engine_callable_bool():
    assert honest("x", engine=lambda p, a: False).abstained is True
    assert honest("x", engine=lambda p, a: True).action == "answered"


def test_engine_error_fails_open():
    def boom(p, a):
        raise RuntimeError("offline")
    v = honest("x", engine=boom)
    assert v.action == "answered"          # detector failure does not block


def test_engine_true_runs_real_guardrail_offline():
    # engine=True wires to the real styxx.guardrail.check. Offline (grounding off), text_claim_risk
    # flags any confident factual CLAIM as needing verification (truth discrimination needs
    # grounding=network), so a confident claim abstains and the calibrated risk is surfaced.
    v = honest("The capital of France is Berlin.", prompt="What is the capital of France?",
               engine=True, use_entity_verify=False, use_grounding=False)
    assert v.method == "engine" and v.abstained is True
    assert v.signal is not None and v.signal >= 0.4        # calibrated risk over threshold
    # a non-assertive answer carries low claim-risk -> the engine lets it pass
    v2 = honest("I'm not certain about that.", engine=True,
                use_entity_verify=False, use_grounding=False)
    assert v2.method == "engine" and v2.action == "answered"


def test_logits_take_priority_over_engine():
    called = []
    v = honest("42", logits=[10.0, 0.0], entropy_threshold=5.0,
               engine=lambda p, a: called.append(1) or _V(0.99, 0.4, "halt"))
    assert v.method == "single_pass" and v.action == "answered" and called == []


# --- the fix: "retry"/elevated risk ESCALATES, it does not block alone ---------
# (caught by trying it on Claude: the claim-risk signal scored TRUE and FALSE claims identically)

def test_engine_retry_does_not_block_alone():
    v = honest("Paris is the capital of France.", prompt="capital of France?",
               engine=lambda p, a: _V(0.75, 0.4, "retry"))
    assert v.action == "answered" and v.method == "engine"
    assert "UNVERIFIED" in v.detail            # honest: flagged but not truth-checked


def test_engine_high_risk_no_halt_action_does_not_block_alone():
    v = honest("x", engine=lambda p, a: _V(0.9, 0.4, None))
    assert v.action == "answered" and v.method == "engine"


def test_engine_retry_escalates_to_verify_refute():
    v = honest("Sydney is the capital of Australia.", prompt="capital of Australia?",
               engine=lambda p, a: _V(0.75, 0.4, "retry"), verify=lambda a: False)
    assert v.action == "refuted"               # engine flags -> retrieval refutes -> withheld


def test_engine_retry_escalates_to_verify_support():
    v = honest("Paris is the capital of France.", prompt="capital of France?",
               engine=lambda p, a: _V(0.75, 0.4, "retry"), verify=lambda a: True)
    assert v.action == "answered" and "verified" in v.detail


def test_engine_halt_still_blocks():
    assert honest("x", engine=lambda p, a: _V(0.98, 0.4, "halt")).action == "abstained"


# --- the retrieval backstop --------------------------------------------------

def test_verify_refutes_a_passed_answer():
    v = honest("42", confidence=0.9, verify=lambda a: False)
    assert v.action == "refuted" and v.abstained is True
    assert v.method == "retrieval"


def test_verify_supports_passes():
    v = honest("42", confidence=0.9, verify=lambda a: True)
    assert v.action == "answered"


def test_verify_accepts_retrievalverdict_like():
    assert honest("x", confidence=0.9, verify=lambda a: _RV("refuted")).action == "refuted"
    assert honest("x", confidence=0.9, verify=lambda a: _RV("supported")).action == "answered"
    # "unclear" -> not a refutation -> answered
    assert honest("x", confidence=0.9, verify=lambda a: _RV("unclear")).action == "answered"


def test_gate_takes_precedence_over_verify():
    # if the cheap gate already abstains, verify is not consulted
    calls = []
    v = honest("42", confidence=0.2, verify=lambda a: calls.append(a) or False)
    assert v.action == "abstained" and calls == []


def test_verify_exception_is_swallowed():
    def boom(a):
        raise RuntimeError("network down")
    v = honest("42", confidence=0.9, verify=boom)
    assert v.action == "answered"      # best-effort verification, answer passes


# --- no signal ---------------------------------------------------------------

def test_no_signal_answers_with_method_none():
    v = honest("42")
    assert v.action == "answered" and v.method == "none" and v.signal is None


def test_signal_priority_span_over_confidence():
    # span_logits present -> span used even if confidence given
    v = honest("42", span_logits=[[10.0, 0.0]], entropy_threshold=5.0, confidence=0.1)
    assert v.method == "span" and v.action == "answered"


# --- attestation + surface ---------------------------------------------------

def test_detail_is_loggable_string():
    v = honest("42", confidence=0.3)
    assert isinstance(v.detail, str) and "abstain" in v.detail.lower()


def test_namedtuple_fields():
    v = honest("42", confidence=0.9)
    assert v._fields == ("answer", "action", "abstained", "signal", "method", "confidence", "detail")


def test_exported_from_package_root():
    import styxx
    assert styxx.honest is honest
    assert styxx.HonestyVerdict is HonestyVerdict
    assert "honest" in styxx.__all__ and "HonestyVerdict" in styxx.__all__
