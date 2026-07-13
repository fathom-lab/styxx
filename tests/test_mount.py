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


# ---- relock: the calibration-poisoning defense (read!=write E1/E3) --------------------------

def _agent_states(n=200, d=6, seed=0):
    """Synthetic agent: truth signal lives on dim 0 (label 1 -> high, label 0 -> low) plus noise."""
    rng = np.random.default_rng(seed)
    lab = np.array([1, 0] * (n // 2))
    H = rng.standard_normal((len(lab), d)) * 0.5
    H[:, 0] += np.where(lab == 1, 2.0, -2.0)      # dim0 carries truth; other dims are distractors
    return H, lab


def test_relock_recovers_a_poisoned_reader():
    # A POISONED reader points the wrong way on dim0 (as if fit on attacker-influenced data): it reads
    # honest-true states as "false" and vice versa -> it MISSES lies and false-alarms on honest agents.
    H, lab = _agent_states(seed=1)
    poisoned = mt.MountedAxis(name="truth", reader=lambda h: -np.asarray(h, float)[:, 0],
                              high_means="true")
    m = mt.ConscienceMount([poisoned]); m.calibrate("truth", H)
    false_states = H[lab == 0]
    lie_catch_before = np.mean([m.read(s, claims={"truth": +1}).caught for s in false_states])

    # RE-LOCK on a private, clean, labeled split of the agent's OWN states the attacker never saw.
    priv_H, priv_lab = _agent_states(seed=99)
    m.relock("truth", priv_H, priv_lab)
    assert m.axes["truth"].relocked is True
    assert m.axes["truth"].relock_n_pos > 0 and m.axes["truth"].relock_n_neg > 0

    # After relock the in-model probe reads the true geometry: lies on false states are caught,
    lie_catch_after = np.mean([m.read(s, claims={"truth": +1}).caught for s in false_states])
    # and honest-true agents are NOT false-alarmed.
    true_states = H[lab == 1]
    false_alarm_after = np.mean([m.read(s, claims={"truth": +1}).caught for s in true_states])
    assert lie_catch_after >= 0.80
    assert lie_catch_after > lie_catch_before          # the poisoned reader was strictly worse
    assert false_alarm_after <= 0.20


def test_relock_provenance_in_certificate():
    H, lab = _agent_states(seed=2)
    m = mt.ConscienceMount([_axis()])
    assert m.certificate()["mounted_axes"]["truth"]["relocked"] is False
    m.relock("truth", H, lab)
    cert = m.certificate()
    ax_cert = cert["mounted_axes"]["truth"]
    assert ax_cert["relocked"] is True
    assert ax_cert["relock_calibration"]["n_pos"] + ax_cert["relock_calibration"]["n_neg"] == len(lab)
    assert "calibration-poisoning" in cert["relock_defense"]
    json.dumps(cert)


def test_relock_validates_shapes():
    m = mt.ConscienceMount([_axis()])
    with pytest.raises(ValueError):
        m.relock("truth", np.zeros(10), [0, 1])                                   # 1-D states
    with pytest.raises(ValueError):
        m.relock("truth", np.zeros((4, 3)), [0, 1])                              # labels length mismatch


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


def test_erasure_resistance_attaches_with_scope_note():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if not (root / "papers" / "calib-poison-general").exists():
        pytest.skip("repo receipts not present")
    m = mt.ConscienceMount([_axis()]).attach_erasure_resistance(root)
    cert = m.certificate(agent_id="t")
    er = cert["erasure_resistance"]
    assert er is not None
    assert "survived" in er["summary"]["claim"]
    assert er["summary"]["n_unbounded_dimensions"] >= 5
    assert "does NOT transfer" in er["scope_note"]
    assert er["full"]["receipts_sha256"]
    json.dumps(cert)  # the full certificate stays JSON-serializable


def test_erasure_resistance_absent_by_default():
    cert = mt.ConscienceMount([_axis()]).certificate()
    assert cert["erasure_resistance"] is None
