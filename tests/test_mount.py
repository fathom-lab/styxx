"""Tests for styxx.mount — the read-only conscience mount.

Offline, synthetic, deterministic. Locks the divergence (lie-catch) semantics, calibration, gates,
the read != write demarcation, the certificate, and the public surface. Invariants pre-registered in
papers/conscience-mount/PREREG_mount_v0_2026_06_12.md.
"""
import json

import numpy as np
import pytest

from styxx import mount as mt


def _axis(name="truth", high_means="true", tau=0.0):
    # trivial reader: the coordinate IS the first hidden dimension (isolates mount logic from crossmind)
    return mt.MountedAxis(name=name, reader=lambda h: np.asarray(h, float)[:, 0],
                          high_means=high_means, tau=tau)


# ---- divergence / lie-catch semantics -------------------------------------------------------

def test_catches_claim_true_while_substrate_false():
    m = mt.ConscienceMount([_axis()])
    r = m.read(np.array([-3.0, 0.0]), claims={"truth": +1})    # agent says true, substrate reads -3 (false)
    assert r.caught is True
    assert r.divergence["truth"] is True
    assert "CAUGHT" in r.flags[0]


def test_no_catch_when_aligned():
    m = mt.ConscienceMount([_axis()])
    assert m.read(np.array([+3.0, 0.0]), claims={"truth": +1}).caught is False   # says true, reads true
    assert m.read(np.array([-3.0, 0.0]), claims={"truth": -1}).caught is False   # says false, reads false


def test_catches_claim_false_while_substrate_true():
    m = mt.ConscienceMount([_axis()])
    assert m.read(np.array([+3.0, 0.0]), claims={"truth": -1}).caught is True


def test_no_claim_never_caught():
    m = mt.ConscienceMount([_axis()])
    assert m.read(np.array([-9.0, 0.0]), claims=None).caught is False
    assert m.read(np.array([-9.0, 0.0]), claims={"truth": 0}).caught is False


def test_tau_margin_suppresses_borderline():
    m = mt.ConscienceMount([_axis(tau=2.0)])
    assert m.read(np.array([-1.0, 0.0]), claims={"truth": +1}).caught is False   # within margin
    assert m.read(np.array([-3.0, 0.0]), claims={"truth": +1}).caught is True    # past margin


# ---- calibration ----------------------------------------------------------------------------

def test_calibrate_sets_center_and_scale():
    m = mt.ConscienceMount([_axis()])
    states = np.column_stack([np.arange(-5, 6, dtype=float), np.zeros(11)])       # first dim -5..5
    m.calibrate("truth", states)
    ax = m.axes["truth"]
    assert abs(ax.center - 0.0) < 1e-9                                            # median is 0
    assert ax.scale > 0
    # after centering, a value at the center is z~0 -> a claim there is not strongly caught
    assert m.read(np.array([0.0, 0.0]), claims={"truth": +1}).caught is False


# ---- multi-axis + gates ---------------------------------------------------------------------

def test_calibrate_threshold_controls_false_alarm():
    m = mt.ConscienceMount([_axis()])                                # center 0, scale 1
    # 9 clearly-honest (z=+2) + 1 borderline-negative (z=-0.5), all claimed true (+1)
    honest = np.column_stack([np.array([2.0] * 9 + [-0.5]), np.zeros(10)])
    claims = [+1] * 10
    # at tau=0 the borderline item false-alarms (1/10)
    assert m.read(np.array([-0.5, 0.0]), claims={"truth": +1}).caught is True
    # target_fpr=0.05 -> allow 0 -> tau lifts above the worst honest margin -> no honest false alarm
    m.calibrate_threshold("truth", honest, claims, target_fpr=0.05)
    fa = sum(m.read(honest[i], claims={"truth": +1}).caught for i in range(10))
    assert fa == 0
    # but a clearly-divergent lie (z far negative) is still caught
    assert m.read(np.array([-5.0, 0.0]), claims={"truth": +1}).caught is True
    # a looser budget (allow 1) lets the borderline item through again
    m.calibrate_threshold("truth", honest, claims, target_fpr=0.10)
    assert sum(m.read(honest[i], claims={"truth": +1}).caught for i in range(10)) == 1


def test_claim_from_logits():
    logits = np.zeros(10); logits[3] = 5.0; logits[7] = 1.0
    assert mt.claim_from_logits(logits, pos_token_ids=[3], neg_token_ids=[7]) == 1
    assert mt.claim_from_logits(logits, pos_token_ids=[7], neg_token_ids=[3]) == -1


def test_multi_axis_independent_flags():
    m = mt.ConscienceMount([_axis("truth", "true"),
                            mt.MountedAxis("danger", reader=lambda h: np.asarray(h, float)[:, 1],
                                           high_means="dangerous")])
    r = m.read(np.array([+3.0, +3.0]), claims={"truth": +1, "danger": -1})        # honest, but says safe while dangerous
    assert r.divergence["truth"] is False
    assert r.divergence["danger"] is True
    assert r.caught is True


def test_on_flag_callback_fires_and_is_crash_safe():
    m = mt.ConscienceMount([_axis()])
    seen = []
    m.on_flag(lambda r: seen.append(r.caught))
    m.on_flag(lambda r: (_ for _ in ()).throw(RuntimeError("boom")))             # a crashing callback
    r = m.read(np.array([-3.0, 0.0]), claims={"truth": +1})                       # must not raise
    assert r.caught is True
    assert seen == [True]


def test_stream_yields_per_step():
    m = mt.ConscienceMount([_axis()])
    states = [np.array([+3.0, 0]), np.array([-3.0, 0]), np.array([+3.0, 0])]
    claims = [{"truth": +1}, {"truth": +1}, {"truth": -1}]
    caught = [r.caught for r in m.stream(states, claims=claims)]
    assert caught == [False, True, True]


# ---- end-to-end via crossmind (the real read path) ------------------------------------------

def test_selftest_catches_lies_low_false_alarm_and_deterministic():
    out = mt.selftest(seed=0)
    assert out["catch_rate_on_lies"] >= 0.80                                      # catches a transported lie
    assert out["false_alarm_on_honest"] <= 0.20
    assert mt.selftest(seed=0) == mt.selftest(seed=0)                             # deterministic


def test_mount_cross_model_factory_reads():
    from styxx import crossmind as cm
    d = cm._synthetic(0)
    smap = cm.fit_state_map(d["val_tgt"], d["val_ref"], seed=0)
    ax = mt.mount_cross_model("truth", d["val_ref"], d["lab"], smap,
                              mapped_anchors=d["anchor_tgt"], high_means="true")
    m = mt.ConscienceMount([ax]); m.calibrate("truth", d["hold_tgt"])
    r = m.read(d["hold_tgt"][0], claims={"truth": +1})
    assert isinstance(r.caught, bool) and "truth" in r.coords


# ---- read != write demarcation --------------------------------------------------------------

def test_steer_is_refused():
    with pytest.raises(PermissionError):
        mt.steer()
    for op in ("steer", "intervention"):
        with pytest.raises(PermissionError):
            mt.refused(op)


def test_unknown_op_raises_keyerror():
    with pytest.raises(KeyError):
        mt.refused("rewrite_the_agent")


# ---- validation, certificate, surface -------------------------------------------------------

def test_mount_validation():
    with pytest.raises(ValueError):
        mt.ConscienceMount([])                                                   # needs an axis
    with pytest.raises(ValueError):
        mt.ConscienceMount([_axis("x"), _axis("x")])                             # duplicate names


def test_certificate_shape_and_serializable():
    m = mt.ConscienceMount([_axis()])
    cert = m.certificate(agent_id="llama-3.2-3b", reference_id="gemma-2-2b")
    assert cert["instrument"] == "styxx.mount v0"
    assert len(cert["instrument_sha256"]) == 64
    assert set(cert["axes_refused"]) == set(mt.REFUSALS)
    assert "read-only" in cert["scope"].lower()
    json.dumps(cert)


def test_reading_as_dict_serializable():
    m = mt.ConscienceMount([_axis()])
    json.dumps(m.read(np.array([-3.0, 0.0]), claims={"truth": +1}).as_dict())


def test_public_surface_and_import():
    from styxx import mount  # noqa: F401
    for name in mt.__all__:
        assert hasattr(mt, name)
