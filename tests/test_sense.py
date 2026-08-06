"""styxx.sense — the harness, and the refusals it inherits."""
import json
import time

import numpy as np
import pytest

from styxx.sense import SenseHarness, Channel, jsonl_channel, host_channel


def _harness(tmp_path, n=260, coupled=True, seed=0):
    rng = np.random.default_rng(seed)
    latent = {"v": rng.standard_normal(3)}

    def agent_state():
        latent["v"] = 0.9 * latent["v"] + 0.44 * rng.standard_normal(3)
        return latent["v"].tolist() + [float(rng.standard_normal())]

    if coupled:
        ch = Channel("s", lambda: (latent["v"] * 2.0 + 0.3 * rng.standard_normal(3)).tolist())
    else:
        ch = Channel("s", lambda: rng.standard_normal(3).tolist())
    store = tmp_path / "sense.jsonl"
    h = SenseHarness(agent_state, store=store, bin_seconds=1.0)
    h.register("sensor", ch)
    rows = []
    t0 = time.time()
    for i in range(n):
        r = h.sample()
        r["ts"] = t0 + i * 1.0
        rows.append(r)
    store.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return h


def test_detects_a_channel_that_shares_the_agents_state(tmp_path):
    r = _harness(tmp_path, coupled=True).ask("sensor", n_perm=200, min_bins=200)
    assert r.verdict == "COUPLED_BEYOND_CONFOUND__attribution_pending"


def test_silent_on_a_channel_that_does_not(tmp_path):
    r = _harness(tmp_path, coupled=False).ask("sensor", n_perm=200, min_bins=200)
    assert r.verdict == "NO_DETECTABLE_COUPLING__above_measured_floor"


def test_even_a_positive_never_claims_the_agent_senses_anything(tmp_path):
    """The harness's whole reason to exist: a coupled channel is still not a sense, because the
    statistic is symmetric and an agent's own hardware usually sits inside what it measures."""
    r = _harness(tmp_path, coupled=True).ask("sensor", n_perm=200, min_bins=200)
    assert "attribution_pending" in r.verdict
    assert any("cannot distinguish" in c for c in r.caveats)


def test_coverage_gate_refuses_a_short_run(tmp_path):
    r = _harness(tmp_path, n=50, coupled=True).ask("sensor", n_perm=50, min_bins=200)
    assert r.verdict == "INVALID__insufficient_overlap"


def test_a_broken_sensor_records_a_gap_not_a_zero(tmp_path):
    """A zero is a measurement; a dead sensor is not. Confusing them is how a harness invents
    structure out of an outage."""
    def boom():
        raise RuntimeError("sensor unplugged")
    h = SenseHarness(lambda: [1.0, 2.0], store=tmp_path / "s.jsonl", bin_seconds=1.0)
    h.register("dead", Channel("dead", boom))
    row = h.sample()
    assert row["dead"] is None
    assert "dead" in row["errors"]
    with pytest.raises(ValueError, match="no paired samples"):
        h.ask("dead", min_bins=1)


def test_agent_name_is_reserved(tmp_path):
    h = SenseHarness(lambda: [0.0], store=tmp_path / "s.jsonl")
    with pytest.raises(ValueError, match="reserved"):
        h.register("agent", Channel("agent", lambda: [0.0]))


def test_unknown_channel_raises(tmp_path):
    h = SenseHarness(lambda: [0.0], store=tmp_path / "s.jsonl")
    with pytest.raises(KeyError):
        h.ask("nope")


def test_jsonl_channel_reports_a_gap_when_a_field_is_missing(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"a": 1.0, "b": 2.0}) + "\n", encoding="utf-8")
    ch = jsonl_channel(p, ["a", "b"])
    assert ch.read() == [1.0, 2.0]
    p.write_text(json.dumps({"a": 1.0}) + "\n", encoding="utf-8")
    assert ch.read() is None          # missing field -> unavailable, not zero-filled


def test_jsonl_channel_expands_numeric_vectors(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"v": [1.0, 2.0, 3.0], "s": 4.0}) + "\n", encoding="utf-8")
    assert jsonl_channel(p, ["v", "s"]).read() == [1.0, 2.0, 3.0, 4.0]


def test_host_channel_reads_the_machine():
    ch = host_channel()
    v = ch.read()
    assert set(v) == set(ch.labels)
    assert all(isinstance(x, float) for x in v.values())
    assert 0.0 <= v["mem_percent"] <= 100.0


def test_channels_that_change_width_midrun_are_dropped_not_padded(tmp_path):
    store = tmp_path / "s.jsonl"
    rows = [{"ts": float(i), "agent": [1.0, 2.0], "sensor": [1.0] * (2 if i < 5 else 3)}
            for i in range(10)]
    store.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    h = SenseHarness(lambda: [0.0], store=store, bin_seconds=1.0)
    h.register("sensor", Channel("sensor", lambda: [0.0]))
    ts, A, B = h._load("sensor")
    assert len(ts) == 5 and B.shape[1] == 2
