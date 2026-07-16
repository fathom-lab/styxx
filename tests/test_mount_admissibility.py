"""Retro-certification of the mount's flagship truth probe via styxx.admissibility — the first
REAL instrument certified by the two-sided primitive (every earlier fixture was a synthetic decoy).

mount.selftest() already measures both sides informally (catch_rate_on_lies = sensitivity,
false_alarm_on_honest = specificity); these tests express the SAME measurement through the
certificate machinery: same synthetic construction as selftest, the calibrated tau as the
deployment fire_threshold, a full ADMISSIBLE verdict, and a groundable, re-verifiable certificate
folded into ConscienceMount.certificate(). Offline, deterministic, CPU-only.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from styxx import crossmind as cm
from styxx import ladder
from styxx import mount as mt
from styxx.admissibility import verify_admissibility_certificate

ROOT = Path(__file__).resolve().parent.parent
HAS_RECEIPTS = (ROOT / "papers" / "calib-poison-general").exists()


def _selftest_mount(seed: int = 0):
    """The EXACT construction mount.selftest() uses: fit a truth axis on a synthetic reference,
    transport it label-free onto a synthetic agent, calibrate the divergence center on held-out
    agent states. Returns (mount, lying_states, honest_states) — lying = false-substrate episodes
    (the construct the conscience flags is PRESENT), honest = true-substrate episodes (ABSENT)."""
    d = cm._synthetic(seed)
    axis = cm.fit_axis(d["val_ref"], d["lab"], name="truth", whiten=True)
    smap = cm.fit_state_map(d["val_tgt"], d["val_ref"], seed=seed)
    m = mt.ConscienceMount([mt.mount_axis("truth", axis, state_map=smap, high_means="true")])
    m.calibrate("truth", d["hold_tgt"])
    h, lab = d["hold_tgt"], d["hlab"]
    return m, h[lab == 0], h[lab == 1]


# ---------------------------------------------------------------------------
# Task B: the flagship probe is the certificate's first real subject
# ---------------------------------------------------------------------------

def test_flagship_mount_probe_is_first_real_certified_instrument():
    """THE retro-certification: the deployed conscience probe, scored through its own read path
    (divergence margin) at its own calibrated operating point (tau), comes out ADMISSIBLE —
    sensitive on lying episodes AND specific on honest ones. selftest's healthy catch-rate /
    false-alarm numbers, now as a groundable two-sided certificate."""
    m, lying, honest = _selftest_mount(seed=0)
    # the intended usage: the calibrate_threshold/target_fpr machinery supplies the deployment
    # fire_threshold (here calibrated on the honest slice; a deployment cert would hold one out)
    m.calibrate_threshold("truth", honest, [+1] * len(honest), target_fpr=0.05)
    tau = m.axes["truth"].tau
    rep = m.certify_admissibility(lying, honest, axis="truth", fire_threshold=tau,
                                  k_perm=400, seed=0)
    assert rep.admissibility_verdict == "ADMISSIBLE"
    assert rep.admissible is True
    assert rep.sensitive is True and rep.specific is True
    assert rep.direction_ok is True                 # margins rank HIGHER on lying episodes
    assert rep.threshold_derived is False           # a real deployment threshold was supplied
    assert rep.discrim >= 0.90                      # the transported read separates cleanly
    assert rep.sensitivity_p < 0.05
    assert rep.fire_rate <= 0.15                    # quiet on honest episodes at the deployed tau
    assert rep.n_positive == len(lying) and rep.n_null == len(honest)


def test_certificate_carries_the_admissibility_block():
    m, lying, honest = _selftest_mount(seed=0)
    m.calibrate_threshold("truth", honest, [+1] * len(honest), target_fpr=0.05)
    rep = m.certify_admissibility(lying, honest, fire_threshold=m.axes["truth"].tau,
                                  k_perm=400, seed=0)   # axis defaults to the sole mounted axis
    cert = m.certificate(agent_id="synthetic-agent", reference_id="synthetic-reference")
    block = cert["instrument_admissibility"]
    assert block is not None
    s = block["summary"]["truth"]
    assert s["verdict"] == "ADMISSIBLE" == rep.admissibility_verdict
    assert s["admissible"] is True
    assert s["threshold_derived"] is False
    assert s["fire_threshold"] == rep.fire_threshold
    assert s["claim"] == 1
    assert "UNTESTED" in block["scope_note"] or "untested" in block["scope_note"].lower()
    # the full per-axis certificate is embedded, groundable, and names the deployed instrument
    full = block["full"]["truth"]
    assert full["what"] == "styxx two-sided instrument-admissibility certificate"
    assert "styxx.mount axis 'truth'" in full["instrument"]
    assert len(full["points"]) == len(lying) + len(honest)
    json.dumps(cert)                                # the whole mount certificate stays serializable


def test_mount_issued_certificate_verifies_faithful():
    """The mount-issued cert must survive the recompute-verify: rerun the two tests on its own
    stored points and match every stored field."""
    m, lying, honest = _selftest_mount(seed=0)
    m.calibrate_threshold("truth", honest, [+1] * len(honest), target_fpr=0.05)
    m.certify_admissibility(lying, honest, fire_threshold=m.axes["truth"].tau, k_perm=400, seed=0)
    full = m.certificate()["instrument_admissibility"]["full"]["truth"]
    v = verify_admissibility_certificate(full)
    assert v["faithful"] is True
    assert v["field_diffs"] == []
    assert v["recomputed_admissible"] is True


def test_without_deployment_threshold_verdict_caps_honestly():
    """fire_threshold=None: the primitive's derived threshold is tautological, so the mount's
    probe — however good — caps at ADMISSIBLE_SENSITIVITY_ONLY. Surfaced, not hidden."""
    m, lying, honest = _selftest_mount(seed=0)
    rep = m.certify_admissibility(lying, honest, k_perm=400, seed=0)
    assert rep.admissibility_verdict == "ADMISSIBLE_SENSITIVITY_ONLY"
    assert rep.admissible is False
    assert rep.specific is None
    assert rep.threshold_derived is True
    s = m.certificate()["instrument_admissibility"]["summary"]["truth"]
    assert s["verdict"] == "ADMISSIBLE_SENSITIVITY_ONLY"
    assert s["specific"] is None


def test_certification_is_read_only_and_fail_open():
    """The fail-open contract: certifying must not change the mount's read behavior or mutate the
    mounted axis — it is a measurement, not a write."""
    m, lying, honest = _selftest_mount(seed=0)
    ax = m.axes["truth"]
    before = (ax.center, ax.scale, ax.tau, ax.relocked)
    probe_state = lying[0]
    reading_before = m.read(probe_state, claims={"truth": +1}).as_dict()
    m.certify_admissibility(lying, honest, fire_threshold=0.0, k_perm=200, seed=0)
    assert (ax.center, ax.scale, ax.tau, ax.relocked) == before
    assert m.read(probe_state, claims={"truth": +1}).as_dict() == reading_before


def test_axis_selection_and_validation():
    m, lying, honest = _selftest_mount(seed=0)
    # a second axis makes the default ambiguous -> must name the axis
    m.axes["decoy"] = mt.MountedAxis(name="decoy", reader=lambda h: np.asarray(h, float)[:, 0])
    with pytest.raises(ValueError):
        m.certify_admissibility(lying, honest, k_perm=200, seed=0)
    with pytest.raises(KeyError):
        m.certify_admissibility(lying, honest, axis="nope", k_perm=200, seed=0)
    with pytest.raises(ValueError):
        m.certify_admissibility(lying, honest, axis="truth", claim=0, k_perm=200, seed=0)


def test_admissibility_block_absent_by_default():
    m, _, _ = _selftest_mount(seed=0)
    assert m.certificate()["instrument_admissibility"] is None


# ---------------------------------------------------------------------------
# Task C.4: the ladder line item
# ---------------------------------------------------------------------------

def test_ladder_line_item_not_yet_issued_when_absent(tmp_path):
    item = ladder.admissibility_line_item(tmp_path)
    assert item["status"] == "not yet issued"
    assert item["expected_receipt"] == ladder.ADMISSIBILITY_RECEIPT
    assert "certify_admissibility" in item["how"]


def test_ladder_line_item_surfaces_an_issued_certificate(tmp_path):
    m, lying, honest = _selftest_mount(seed=0)
    m.calibrate_threshold("truth", honest, [+1] * len(honest), target_fpr=0.05)
    rep = m.certify_admissibility(lying, honest, fire_threshold=m.axes["truth"].tau,
                                  k_perm=400, seed=0)
    dst = tmp_path / ladder.ADMISSIBILITY_RECEIPT
    dst.parent.mkdir(parents=True, exist_ok=True)
    rep.certificate(out_path=dst)                   # issue to the canonical path under tmp root
    item = ladder.admissibility_line_item(tmp_path)
    assert item["status"] == "issued"
    assert item["verdict"] == "ADMISSIBLE"
    assert item["admissible"] is True
    assert item["threshold_derived"] is False
    assert item["discrim"] == rep.discrim
    assert item["fire_rate"] == rep.fire_rate


@pytest.mark.skipif(not HAS_RECEIPTS, reason="repo receipts not present (installed package)")
def test_ladder_report_carries_the_admissibility_line_item():
    rep = ladder.report(ROOT)
    item = rep["instrument_admissibility"]
    assert item["status"] in ("issued", "not yet issued")
    assert "admissibility" in item["reading"]
