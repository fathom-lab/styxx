# -*- coding: utf-8 -*-
"""styxx.measured — the validity channel, tested against the defects that produced it.

The claim under test is bounded: this type makes several SILENT-PASS subtypes
structurally hard **for code written with it**. It is a cure available to new
code, not a retrofit that fixes existing modules, and the coverage test below
records exactly which subtypes it reaches and which it does not.
"""
import pytest

from styxx.measured import (Measured, NoComputedData, UnmeasuredComparison,
                            UnmeasuredValue, lenient, measure)


# ── the shapes that produced 74 defects ─────────────────────────────────────

def test_sp1_a_crash_becomes_no_computed_data_not_a_perfect_score():
    """gate() hit a bad API key and returned trust_score=1.0 — the maximum."""
    @measure
    def trust(client):
        raise RuntimeError("invalid x-api-key")

    t = trust(None)
    assert t.measured is False
    assert "invalid x-api-key" in t.why
    with pytest.raises(UnmeasuredComparison):
        t > 0.7
    assert t.value_or(0.5) == 0.5


def test_sp3_a_degenerate_statistic_refuses_instead_of_returning_one():
    """r2 = 1.0 on zero-variance targets: perfect explained variance from no
    variance to explain."""
    def r2(ss_res, ss_tot):
        if ss_tot <= 1e-12:
            return Measured.unmeasured("zero target variance; R^2 undefined")
        return Measured(1.0 - ss_res / ss_tot)

    good = r2(0.1, 10.0)
    assert good > 0.9

    degenerate = r2(0.5, 0.0)
    assert degenerate.measured is False
    with pytest.raises(UnmeasuredComparison):
        degenerate > 0.9          # was True — a health check passing on nothing


def test_sp4_truthiness_cannot_silently_decide_a_gate():
    """`bool(v.fired or v.needs_revision)` shipped here: .fired is a LIST, so
    the calibrated term could never change the outcome."""
    unmeasured = Measured.unmeasured("mount unavailable")
    with pytest.raises(UnmeasuredComparison):
        bool(unmeasured)
    with pytest.raises(UnmeasuredComparison):
        if unmeasured:            # the exact shape, refused
            pass

    assert bool(Measured(True)) is True
    assert bool(Measured(False)) is False


def test_unmeasured_is_contagious_and_keeps_its_reason():
    """A missing measurement must not launder itself into a real-looking
    aggregate three functions downstream."""
    conf = Measured.unmeasured("no phase4 readings in window", source="check_health")
    scaled = conf.map(lambda c: c * 100)
    assert scaled.measured is False
    assert scaled.why == "no phase4 readings in window"
    assert scaled.source == "check_health"

    real = Measured(0.4, source="phase4").map(lambda c: c * 100)
    assert real.measured is True and real.value == pytest.approx(40.0)


def test_an_absence_must_carry_a_reason():
    """An absence nobody can explain is indistinguishable from one nobody
    noticed."""
    with pytest.raises(ValueError, match="requires a reason"):
        Measured.unmeasured("")
    with pytest.raises(ValueError):
        Measured.unmeasured("   ")


def test_reading_a_missing_value_raises_rather_than_inventing_one():
    m = Measured.unmeasured("scorer unavailable", source="verify")
    with pytest.raises(UnmeasuredValue) as e:
        m.value
    assert "scorer unavailable" in str(e.value)
    assert "value_or" in str(e.value), "the error must name the honest escape"


def test_the_wire_format_carries_the_validity_channel():
    """A serializer that drops the channel re-creates the problem at the next
    hop — which is how a 0.0 gets into a database and comes back as data."""
    m = Measured.unmeasured("api timeout", source="gate", attempt=2)
    d = m.as_dict()
    assert d["measured"] is False and d["why"] == "api timeout"

    back = Measured.from_dict(d)
    assert back.measured is False
    assert back.why == "api timeout"
    assert back.meta["attempt"] == 2
    with pytest.raises(UnmeasuredComparison):
        back > 0.5


def test_lenient_is_scoped_and_loud_never_ambient():
    """A guard that raises in production gets deleted by the first on-call
    engineer, so the escape has to exist — scoped, so it cannot become the
    default by accident."""
    import warnings as w

    m = Measured.unmeasured("backend missing")
    with pytest.raises(UnmeasuredComparison):
        m > 0.5

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        with lenient():
            assert (m > 0.5) is False
            assert bool(m) is False
            assert m.value is None
    assert caught, "lenient mode must warn every time it degrades"

    # and the escape closes behind itself
    with pytest.raises(UnmeasuredComparison):
        m > 0.5


def test_arinc_alias_exists_for_readers_who_know_it_from_avionics():
    ncd = NoComputedData("pitot heat failed", source="airspeed")
    assert ncd.measured is False
    assert "NCD" in repr(ncd)


def test_measure_passes_through_a_measured_return_unwrapped():
    @measure
    def scorer(x):
        return Measured.unmeasured("insufficient samples", source="inner")

    out = scorer(1)
    assert out.measured is False and out.source == "inner"


# ── the honest bound: what this type does NOT reach ────────────────────────

def test_coverage_against_the_corpus_subtypes_is_recorded_not_claimed():
    """Which SILENT-PASS subtypes does the validity channel actually address?

    Pinned so the claim cannot inflate. `Measured` is a cure for code WRITTEN
    with it; it retrofits nothing, and two subtypes are outside its reach
    entirely.
    """
    addressed = {
        "SP-1",   # crash -> NCD via @measure
        "SP-3",   # degenerate statistic returns NCD instead of a number
        "SP-4",   # bool() refuses to coerce
        "SP-5",   # crash swallowed into a sentinel -> @measure
    }
    partial = {
        "SP-2",   # only if the field itself is Measured end to end
        "SP-7",   # `source=` carries provenance, which is what broke the loop
    }
    out_of_reach = {
        "SP-6",   # an absent guard: the function must still CHOOSE to return
                  # NCD on empty input. That needs the runtime contract.
        "SP-8",   # an inert control / a verification leg that cannot fail is a
                  # logic defect, not a validity-channel defect.
    }
    assert addressed | partial | out_of_reach == {f"SP-{i}" for i in range(1, 9)}
    assert not (addressed & out_of_reach)
