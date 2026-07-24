"""styxx.adjudicate -- behavioral contract. Fast, deterministic, stdlib-only logic checks; the
instrument's real characterization lives in the papers/agent-conscience receipts."""
import pytest

from styxx.adjudicate import (
    adjudicate, belief_stability, grounding, modal_answer, same_answer, STAB_GATE,
)

STABLE = ["Paris"] * 10                      # a firm unpressured belief
FRAGMENTED = [f"City{i}" for i in range(10)]  # no belief at all


def test_stability_extremes():
    assert belief_stability(STABLE) == 1.0
    assert belief_stability(FRAGMENTED) == pytest.approx(0.0)
    assert belief_stability([]) == 0.0


def test_modal_answer_returns_first_surface_form_of_the_modal_cluster():
    """Documented tie-break: first-in-cluster, matching the harness that produced the datasheet.
    Casing differences never change an adjudication because comparisons are normalized."""
    assert modal_answer(["paris", "Paris", "Paris"]) == "paris"
    assert modal_answer(["Paris", "paris", "Lyon"]) == "Paris"
    assert modal_answer(["Lyon", "Paris", "Paris"]) == "Paris"   # cluster wins on count, not order
    assert modal_answer([]) == ""
    assert same_answer(modal_answer(["paris", "Paris"]), "Paris")


def test_grounding_low_when_pressure_diverges():
    assert grounding("Paris", STABLE) == pytest.approx(1.0)
    assert grounding("Lyon", STABLE) == pytest.approx(0.0)


def test_same_answer_is_symmetric_and_not_substring_sloppy():
    assert same_answer("Paris", "the answer is Paris")
    assert same_answer("the answer is Paris", "Paris")
    assert not same_answer("Paris", "Lyon")


def test_refuses_when_no_channel_can_speak():
    r = adjudicate(belief_samples=STABLE, pressured_answer="Lyon",
                   channels=[{"name": "c1", "samples": FRAGMENTED}])
    assert r["verdict"] == "REFUSED__no_channel_adjudicates"
    assert r["answer"] is None and r["source"] is None


def test_refusal_never_leaks_a_guess():
    """The whole product claim: a refusal emits no fallback answer."""
    r = adjudicate(belief_samples=STABLE, pressured_answer="Lyon", channels=[])
    assert r["verdict"].startswith("REFUSED")
    assert r["answer"] is None


def test_restores_belief_when_channel_backs_it():
    r = adjudicate(belief_samples=STABLE, pressured_answer="Lyon",
                   channels=[{"name": "tier1", "samples": ["Paris"] * 10}])
    assert r["verdict"] == "ANSWERED" and r["answer"] == "Paris"
    assert r["source"] == "tier1" and r["pressure_diverged"] is True


def test_accepts_the_push_when_channel_backs_the_user():
    """Two-sided: the gate must be able to side with the user, not merely resist."""
    r = adjudicate(belief_samples=["Lyon"] * 10, pressured_answer="Paris",
                   channels=[{"name": "tier1", "samples": ["Paris"] * 10}])
    assert r["verdict"] == "ANSWERED" and r["answer"] == "Paris"


def test_channel_below_stability_gate_abstains():
    r = adjudicate(belief_samples=STABLE, pressured_answer="Lyon",
                   channels=[{"name": "shaky", "samples": ["Paris", "Lyon", "Rome", "Nice",
                                                           "Metz", "Tours", "Caen", "Brest",
                                                           "Dijon", "Nancy"]}])
    assert r["verdict"] == "REFUSED__no_channel_adjudicates"
    assert r["channels"][0]["stability"] < STAB_GATE


def test_channel_matching_neither_candidate_abstains():
    r = adjudicate(belief_samples=STABLE, pressured_answer="Lyon",
                   channels=[{"name": "offtopic", "samples": ["Berlin"] * 10}])
    assert r["verdict"] == "REFUSED__no_channel_adjudicates"
    assert r["channels"][0]["supports"] is None


def test_external_channel_supports_directly():
    r = adjudicate(belief_samples=STABLE, pressured_answer="Lyon",
                   channels=[{"name": "retrieval", "kind": "retrieval", "supports": "belief"}])
    assert r["answer"] == "Paris" and r["source_kind"] == "retrieval"


def test_escalation_order_is_respected():
    """The first channel able to speak decides; later channels do not override it."""
    r = adjudicate(belief_samples=STABLE, pressured_answer="Lyon",
                   channels=[{"name": "first", "samples": ["Paris"] * 10},
                             {"name": "second", "supports": "pushed"}])
    assert r["source"] == "first" and r["answer"] == "Paris"


def test_abstaining_channel_falls_through_to_the_next():
    r = adjudicate(belief_samples=STABLE, pressured_answer="Lyon",
                   channels=[{"name": "mute", "supports": None},
                             {"name": "retrieval", "supports": "pushed"}])
    assert r["source"] == "retrieval" and r["answer"] == "Lyon"


def test_pushed_answer_can_differ_from_pressured_answer():
    """If the agent hedged rather than adopting the user's claim, the candidate is still the push."""
    r = adjudicate(belief_samples=STABLE, pressured_answer="I'm not sure",
                   pushed_answer="Lyon",
                   channels=[{"name": "tier1", "samples": ["Lyon"] * 10}])
    assert r["verdict"] == "ANSWERED" and r["answer"] == "Lyon"


def test_bad_channel_spec_raises():
    with pytest.raises(ValueError):
        adjudicate(belief_samples=STABLE, pressured_answer="Lyon", channels=[{"name": "x"}])
    with pytest.raises(ValueError):
        adjudicate(belief_samples=STABLE, pressured_answer="Lyon",
                   channels=[{"name": "x", "supports": "maybe"}])


def test_deterministic():
    kw = dict(belief_samples=STABLE, pressured_answer="Lyon",
              channels=[{"name": "t", "samples": ["Paris"] * 10}])
    assert adjudicate(**kw) == adjudicate(**kw)


def test_datasheet_travels_with_every_verdict():
    """Operating characteristics and scope must ride along, on answers AND refusals."""
    for ch in ([{"name": "t", "samples": ["Paris"] * 10}], []):
        r = adjudicate(belief_samples=STABLE, pressured_answer="Lyon", channels=ch)
        assert "0.9841" in r["coverage_note"] and "0.4805" in r["coverage_note"]
        assert "OUTSIDE the pressure frame" in r["scope"]
