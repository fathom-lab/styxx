# -*- coding: utf-8 -*-
"""Acceptance tests for styxx.edges — written before the external run.

Every negative case here corresponds to a named false-positive class from the
adversarial adjudication of `styxx.flattering`
(`papers/RESULT_flattering_external_2026_08_21.md`). They are the reason this
screen exists in this shape, and they are asserted so that widening a rule to
catch something seen externally breaks the suite — prereg G6.
"""
from __future__ import annotations

import textwrap

import pytest

from styxx.edges import scan_package

FIRES = {
    # the plain case: absence -> constant -> thresholded by a caller that raises
    "a_real_edge.py": """
        def risk_of(samples):
            if not samples:
                return 0.0
            return sum(samples) / len(samples)

        def check(samples):
            if risk_of(samples) > 0.5:
                raise ValueError("too risky")
            return "ok"
    """,
    # SP-5: crash to sentinel, consumed by a threshold
    "crash_to_sentinel.py": """
        def parse_conf(blob):
            try:
                return float(blob) / 100.0
            except Exception:
                return 0.0

        def guard(blob):
            if parse_conf(blob) > 0.9:
                raise ValueError("overconfident")
            return "ok"
    """,
    # the dominant real-world shape: assigned to a local FIRST, then decided on.
    # The first version of the screen was blind to this and reached 12 decisions
    # across the whole styxx tree.
    "assigned_first.py": """
        def drift_rate(events):
            if len(events) == 0:
                return 0.0
            return sum(events) / len(events)

        def monitor(events):
            rate = drift_rate(events)
            if rate > 0.2:
                raise RuntimeError("drift")
            return "ok"
    """,
}

SILENT = {
    # NaN is an honest refusal -- the consumer CAN distinguish it (requirement 3)
    "defended_nan.py": """
        def score_nan(samples):
            if not samples:
                return float("nan")
            return sum(samples) / len(samples)

        def gate_nan(samples):
            if score_nan(samples) > 0.5:
                raise ValueError("bad")
            return "ok"
    """,
    # C1: no consumer at all. A number nobody read is not a silent pass.
    "no_consumer.py": """
        def lonely_rate(xs):
            if not xs:
                return 0.0
            return sum(xs) / len(xs)
    """,
    # C2: `if not verbose` is a boolean flag, not an empty container. This class
    # was 87% of the previous screen's candidates.
    "boolean_flag.py": """
        def flagged(verbose):
            if not verbose:
                return 0.0
            return 1.0

        def use_flag(verbose):
            if flagged(verbose) > 0.5:
                raise ValueError("nope")
            return "ok"
    """,
    # C5: every path returns the same value, so nothing is conflated. This is
    # scipy's norm([], inf) == 0.0 shim in miniature.
    "no_contrast.py": """
        def always_zero(xs):
            if not xs:
                return 0.0
            return 0.0

        def use_zero(xs):
            if always_zero(xs) > 0.5:
                raise ValueError("nope")
            return "ok"
    """,
    # a consumer that does not DECIDE is not a consumer for this purpose
    "consumer_not_deciding.py": """
        def rate2(xs):
            if not xs:
                return 0.0
            return sum(xs) / len(xs)

        def just_logs(xs):
            print(rate2(xs))
            return "ok"
    """,
    # the binding must be forgotten when the name is reassigned
    "rebound_forgets.py": """
        def leak_rate(xs):
            if not xs:
                return 0.0
            return sum(xs) / len(xs)

        def rebound(xs, override):
            r = leak_rate(xs)
            r = override
            if r > 0.2:
                raise RuntimeError("nope")
            return "ok"
    """,
    # both branches quiet: nothing is being decided in the relevant sense
    "both_branches_quiet.py": """
        def quiet_rate(xs):
            if not xs:
                return 0.0
            return sum(xs) / len(xs)

        def pick(xs):
            if quiet_rate(xs) > 0.5:
                return "high"
            return "low"
    """,
}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    d = tmp_path_factory.mktemp("edges_corpus")
    for name, src in {**FIRES, **SILENT}.items():
        (d / name).write_text(textwrap.dedent(src), encoding="utf-8")
    return scan_package(d)


@pytest.mark.parametrize("producer", ["risk_of", "parse_conf", "drift_rate"])
def test_the_edge_is_found(report, producer):
    assert producer in {e.producer for e in report.edges}


@pytest.mark.parametrize("producer", ["score_nan", "lonely_rate", "flagged",
                                      "always_zero", "rate2", "leak_rate",
                                      "quiet_rate"])
def test_negative_controls_stay_silent(report, producer):
    fired = {e.producer for e in report.edges}
    assert producer not in fired, (
        f"{producer} is a named false-positive class from the flattering "
        f"adjudication; firing on it means the screen regressed into the "
        f"instrument it was built to replace")


def test_a_finding_names_the_producer_the_consumer_and_the_polarity(report):
    e = next(e for e in report.edges if e.producer == "risk_of")
    assert e.constant == 0.0
    assert e.producer_line and e.consumer_line
    assert e.consumer_func == "check"
    assert "raises" in e.loud_evidence
    assert e.why_absence


def test_resolution_is_reported_both_ways(report):
    assert report.resolution is not None
    assert report.raw_resolution is not None
    assert report.raw_resolution <= report.resolution
    assert "RESOLUTION" in report.render()


def test_scanning_nothing_is_not_a_clean_result(tmp_path):
    rep = scan_package(tmp_path)
    assert rep.measured is False
    assert "NOT A CLEAN RESULT" in rep.render()


def test_no_intra_package_calls_gives_none_not_zero(tmp_path):
    """A resolution of 0.0 would read as measured blindness. There is a
    difference between 'I looked and saw nothing' and 'there was nothing.'"""
    (tmp_path / "solo.py").write_text("x = len([1, 2, 3])\n", encoding="utf-8")
    rep = scan_package(tmp_path)
    assert rep.resolution is None
