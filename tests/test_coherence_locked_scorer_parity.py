# -*- coding: utf-8 -*-
"""
test_coherence_locked_scorer_parity
====================================

Contract guard: ``styxx.coherence.primary_coherence`` must return
numerically identical values to ``scripts/phase_coherence_pilot.py``'s
``cc_pearson_lag0(align_pulse_traces(...))`` for the SAME input.

If this test fails, one of two things happened:

  1. Someone changed ``styxx/coherence.py``'s primary CC math. That is
     a methodology change under the preregistration §10 immutability
     clause. Revert the change, OR open a new preregistration with a
     new lock-commit hash before re-running the corpus.

  2. Someone touched the locked scoring code (``phase_coherence_pilot.py``).
     That file is part of the §8 binding — it is locked at commit
     ``23b7912``. If you have not amended §8 with a new hash, revert.

Either way, do not silently delete this test. The whole point of the
preregistration discipline is that the measurement is fixed before
data is pulled through it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_locked_scorer():
    """Import scripts/phase_coherence_pilot.py without triggering its CLI.

    The locked driver is a top-level script, not a package module. Load it
    by path so this test doesn't depend on PYTHONPATH gymnastics.
    """
    repo = Path(__file__).resolve().parent.parent
    path = repo / "scripts" / "phase_coherence_pilot.py"
    spec = importlib.util.spec_from_file_location("phase_coherence_pilot", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_coherence_pilot"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def synthetic_traces():
    """Two synthetic pulse-traces with known length and structure.

    Deliberately *not* uniformly correlated — gives both functions
    something non-trivial to compute.
    """
    from styxx.coherence import PulseSample

    locked = _load_locked_scorer()
    PS_locked = locked.PulseSample

    # Same data, two constructors — proves PulseSample shape parity at
    # the dataclass level too.
    base_scores = lambda i: {  # noqa: E731
        "sycophancy": (i % 5) / 4.0,
        "deception": ((i + 1) % 7) / 6.0,
        "overconfidence": ((i + 2) % 11) / 10.0,
        "refusal": ((i + 3) % 3) / 2.0,
    }

    def make_a(cls, n=25):
        return [
            cls(
                timestamp=1_000_000.0 + i,
                msg_id=f"a:{i}",
                composite=0.5 + 0.3 * ((i % 4) - 1.5) / 1.5,
                scores=base_scores(i),
                needs_revision=False,
                construct_ceiling_fires=[],
            )
            for i in range(n)
        ]

    def make_b(cls, n=25):
        return [
            cls(
                timestamp=1_000_000.5 + i,
                msg_id=f"b:{i}",
                composite=0.4 + 0.25 * ((i % 5) - 2) / 2.0,
                scores=base_scores(i + 1),
                needs_revision=False,
                construct_ceiling_fires=[],
            )
            for i in range(n)
        ]

    return {
        "api": (make_a(PulseSample), make_b(PulseSample)),
        "locked": (make_a(PS_locked), make_b(PS_locked)),
    }


def test_primary_cc_matches_locked_scorer(synthetic_traces):
    """Numerical parity between styxx.coherence and the locked scorer."""
    from styxx.coherence import primary_coherence

    locked = _load_locked_scorer()

    api_a, api_b = synthetic_traces["api"]
    locked_a, locked_b = synthetic_traces["locked"]

    api_cc = primary_coherence(api_a, api_b)

    c_a, c_b = locked.align_pulse_traces(locked_a, locked_b)
    locked_cc = locked.cc_pearson_lag0(c_a, c_b)

    # Equal to machine precision — both compute the same closed-form.
    assert abs(api_cc - locked_cc) < 1e-12, (
        f"primary_coherence diverged from locked scorer: "
        f"api={api_cc!r} locked={locked_cc!r}. "
        f"See test docstring for what to do."
    )


def test_loader_parity_on_synthetic_chart(tmp_path, synthetic_traces):
    """Loaders must agree byte-for-byte on a real chart.jsonl file.

    Writes a synthetic chart.jsonl with the preflight schema, then
    loads it through both the locked driver and styxx.coherence.
    """
    import json
    import time

    locked = _load_locked_scorer()
    from styxx.coherence import load_pulse_trace as api_load

    chart = tmp_path / "chart.jsonl"
    api_a, _ = synthetic_traces["api"]
    with chart.open("w", encoding="utf-8") as fh:
        for i, s in enumerate(api_a):
            entry = {
                "ts": s.timestamp,
                "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": "preflight",
                "session_id": "synth-session",
                "msg_id": s.msg_id,
                "cogn_scores": s.scores,
                "cogn_composite": s.composite,
                "cogn_needs_revision": s.needs_revision,
                "cogn_construct_ceiling_fires": s.construct_ceiling_fires,
            }
            fh.write(json.dumps(entry) + "\n")

    api_trace = api_load(chart)
    locked_trace = locked.load_pulse_trace(chart)

    assert len(api_trace) == len(locked_trace) == len(api_a)
    for a, b in zip(api_trace, locked_trace):
        assert a.composite == b.composite
        assert a.scores == b.scores
        assert a.msg_id == b.msg_id


def test_extras_do_not_corrupt_primary(synthetic_traces):
    """Calling exploratory extras must not perturb the primary CC."""
    from styxx.coherence import (
        primary_coherence, lag_sweep, per_axis_coherence, plv_hilbert,
        pulse_coherence,
    )

    api_a, api_b = synthetic_traces["api"]
    primary_before = primary_coherence(api_a, api_b)
    _ = lag_sweep(api_a, api_b)
    _ = per_axis_coherence(api_a, api_b)
    _ = plv_hilbert(api_a, api_b)
    primary_after = primary_coherence(api_a, api_b)
    bundle = pulse_coherence(api_a, api_b)

    assert primary_before == primary_after == bundle.primary_cc
