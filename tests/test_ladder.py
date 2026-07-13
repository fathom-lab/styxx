"""Tests for styxx.ladder -- the probe-robustness ladder over canonical receipts."""
from pathlib import Path

import pytest

from styxx import ladder

ROOT = Path(__file__).resolve().parent.parent
HAS_RECEIPTS = (ROOT / "papers" / "calib-poison-general").exists()

needs_receipts = pytest.mark.skipif(not HAS_RECEIPTS, reason="repo receipts not present (installed package)")


def test_registry_shape():
    keys = [r.key for r in ladder.RUNGS]
    assert keys == ["poisoning", "parity", "static-erasure", "adaptive-erasure"]
    for r in ladder.RUNGS:
        assert r.canonical_verdict, r.key
        assert r.script.startswith("papers/"), r.key
        assert r.receipt.endswith(".json"), r.key


def test_load_receipt_missing_is_loud(tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        ladder.load_receipt(ladder.RUNGS[0], tmp_path)
    assert "receipt missing" in str(ei.value)


@needs_receipts
def test_verify_clean_on_repo():
    assert ladder.verify(ROOT) == []


@needs_receipts
def test_report_assembles_all_rungs():
    rep = ladder.report(ROOT)
    assert len(rep["rungs"]) == 4
    assert rep["all_verdicts_canonical"] is True
    for r in rep["rungs"]:
        assert r["verdict_matches_canonical"], r["rung"]


@needs_receipts
def test_parity_attribution_is_computed_not_hardcoded():
    pa = ladder.parity_attribution(ROOT)
    assert pa["n_cells"] >= 4
    assert pa["median_capacity_share"] is not None
    # cycle 33-34 finding: the recovery is capacity-DOMINATED -- the median share must be
    # well above half; and it must be a genuine ratio, not a copied constant
    assert 0.5 < pa["median_capacity_share"] <= 1.5
    for c in pa["points"]:
        # capacity_share is stored rounded to 4 decimals
        assert abs(round(1.0 - c["parity_gap"] / c["baseline_gap"], 4) - c["capacity_share"]) < 1e-9


@needs_receipts
def test_certificate_composes_only_from_receipts():
    cert = ladder.erasure_resistance_certificate(ROOT)
    # the two 1.5B removal rungs are SURVIVES in the canonical receipts
    scope_attackers = {e["attacker"] for e in cert["claim_scope"]}
    assert "static subspace erasure" in scope_attackers
    assert "adaptive re-fit erasure" in scope_attackers
    # the 3B scale receipt is classified wherever its verdict puts it -- or pending if absent
    b7 = ROOT / "papers" / "calib-poison-general" / "b7_erasure_3b_result.json"
    if b7.exists():
        placed = [e for b in ("claim_scope", "measured_breaks", "unadjudicated")
                  for e in cert[b] if e.get("receipt", "").endswith("b7_erasure_3b_result.json")]
        assert len(placed) == 1
    else:
        assert any(p["receipt"].endswith("b7_erasure_3b_result.json") for p in cert["pending"])
    # mandatory honesty surfaces
    assert len(cert["unbounded_dimensions"]) >= 5
    assert all(len(h) == 64 for h in cert["receipts_sha256"].values())


@needs_receipts
def test_certificate_surfaces_a_break_with_equal_prominence(tmp_path):
    """Pre-committed blind behavior: an ERASED verdict lands in measured_breaks and the claim
    scope excludes that receipt -- the certificate reports breaks, never hides them."""
    import json, shutil
    for rung in ladder.RUNGS:
        src = ROOT / rung.receipt
        if src.exists():
            dst = tmp_path / rung.receipt
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    # tamper the adaptive rung into a break
    adaptive = tmp_path / "papers" / "calib-poison-general" / "b2_adaptive_erasure_result.json"
    rec = json.loads(adaptive.read_text(encoding="utf-8"))
    rec["verdict"] = "ERASED_ADAPT__read_neq_write_BROKEN_1p5B"
    adaptive.write_text(json.dumps(rec), encoding="utf-8")
    cert = ladder.erasure_resistance_certificate(tmp_path)
    assert any(e["attacker"] == "adaptive re-fit erasure" for e in cert["measured_breaks"])
    assert not any(e["attacker"] == "adaptive re-fit erasure" for e in cert["claim_scope"])
    assert "REMOVED" in cert["measured_breaks_summary"]


@needs_receipts
def test_erased_verdict_would_be_flagged(tmp_path):
    """verify() must catch a receipt whose verdict drifted from the frozen canonical string."""
    import json, shutil
    # replicate the repo layout for one rung with a tampered verdict
    rung = ladder.RUNGS[2]
    src = ROOT / rung.receipt
    dst = tmp_path / rung.receipt
    dst.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(src.read_text(encoding="utf-8"))
    rec["verdict"] = "SURVIVES__but_edited"
    dst.write_text(json.dumps(rec), encoding="utf-8")
    problems = ladder.verify(tmp_path)
    assert any("static-erasure" in p and "SURVIVES__but_edited" in p for p in problems)
