"""The audit's search population must be the corpus, not everything a glob can reach.

`.claude/worktrees/` holds agent scratch clones of this repository. Each one is a full copy, so
every certificate and every receipt in the tree has a byte-identical phantom twin inside it. The
search had no exclusions, and the cost was measured on 2026-08-27:

  * 178 of the 365 certificates the audit enumerated were phantoms — 49% of the audited corpus,
    which double-counted every finding;
  * CAPSTONE_universal_mind's one genuinely CHANGED receipt was reported `ambiguous` ("several
    candidates, no non-arbitrary choice") because of its twin, so the audit printed
    `receipt-changed 0` over a receipt it could see had drifted.

These tests pin the scope. The important one is `test_receipt_only_in_scratch_is_absent`: the fix
must make the search NARROWER, never resolve something it previously refused.
"""
import json
from pathlib import Path

from styxx.corpus_audit import (_EXCLUDED_DIRS, _resolve_receipts, _search,
                                classify_missing, discover_certificates)

REPO = Path(__file__).resolve().parents[1]


def _write(p: Path, payload) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_search_skips_every_excluded_directory(tmp_path):
    real = _write(tmp_path / "papers" / "r.json", {"a": 1})
    for d in sorted(_EXCLUDED_DIRS):
        _write(tmp_path / d / "papers" / "r.json", {"a": 1})
    assert list(_search(tmp_path, "r.json")) == [real]


def test_exclusion_matches_the_directory_at_any_depth(tmp_path):
    """`.claude/worktrees/<name>/papers/...` is three levels deep; a top-level-only check misses it."""
    _write(tmp_path / ".claude" / "worktrees" / "sharp-noether" / "papers" / "deep.json", {"a": 1})
    assert list(_search(tmp_path, "deep.json")) == []


def test_receipt_only_in_scratch_is_absent_not_resolved(tmp_path):
    """The whole point: narrowing must never turn an unresolved receipt into a resolved one."""
    body = {"metric": 0.5}
    raw = json.dumps(body).encode()
    import hashlib
    sha = hashlib.sha256(raw).hexdigest()
    _write(tmp_path / ".claude" / "worktrees" / "w" / "only_here.json", body)
    cert_path = _write(tmp_path / "papers" / "D.certificate.json",
                       {"receipts_sha256": {"only_here.json": sha}})
    cert = json.loads(cert_path.read_text(encoding="utf-8"))

    paths, missing, drift = _resolve_receipts(cert_path, cert, tmp_path)
    assert paths == [] and missing == ["only_here.json"]
    assert classify_missing(cert_path, cert, missing, tmp_path)["only_here.json"]["status"] \
        == "absent"


def test_a_matching_receipt_outside_scratch_still_resolves(tmp_path):
    """The narrowing must not break the cross-directory resolution it was built for."""
    body = {"metric": 0.5}
    import hashlib
    sha = hashlib.sha256(json.dumps(body).encode()).hexdigest()
    real = _write(tmp_path / "papers" / "other" / "r.json", body)
    _write(tmp_path / ".claude" / "worktrees" / "w" / "papers" / "other" / "r.json", body)
    cert_path = _write(tmp_path / "papers" / "D.certificate.json",
                       {"receipts_sha256": {"r.json": sha}})
    cert = json.loads(cert_path.read_text(encoding="utf-8"))

    paths, missing, _ = _resolve_receipts(cert_path, cert, tmp_path)
    assert missing == []
    assert paths == [real], "should resolve the real one, and only once"


def test_phantom_twin_no_longer_makes_a_changed_receipt_look_ambiguous(tmp_path):
    """The live CAPSTONE case, reproduced in miniature.

    One real file whose content is not what was certified, plus a byte-identical twin in scratch.
    Before the fix this classified `ambiguous`; the twin is not a second candidate, it is not a
    candidate at all.
    """
    certified = {"receipts_sha256": {"v.json": "0" * 64}}      # a sha nothing matches
    changed = {"metric": 0.99}
    _write(tmp_path / "papers" / "mind" / "v.json", changed)
    _write(tmp_path / ".claude" / "worktrees" / "w" / "papers" / "mind" / "v.json", changed)
    cert_path = _write(tmp_path / "papers" / "D.certificate.json", certified)

    detail = classify_missing(cert_path, certified, ["v.json"], tmp_path)["v.json"]
    assert detail["status"] == "changed"
    assert len(detail["candidates"]) == 1


def test_certificate_enumeration_skips_scratch(tmp_path):
    real = _write(tmp_path / "papers" / "A.certificate.json", {})
    _write(tmp_path / ".claude" / "worktrees" / "w" / "papers" / "A.certificate.json", {})
    assert discover_certificates(tmp_path) == [real]


def test_live_repo_enumeration_excludes_scratch():
    """A regression guard on the real tree, skipped when there is no scratch to exclude."""
    certs = discover_certificates(REPO)
    assert certs, "the repository should carry certificates"
    assert not [p for p in certs if not _EXCLUDED_DIRS.isdisjoint(p.parts)]


def test_the_capstone_receipt_is_reported_changed_on_the_live_tree():
    """Pins the finding the phantom was hiding, against the real corpus."""
    cert_path = (REPO / "papers" / "ancient-question-program"
                 / "CAPSTONE_universal_mind_2026_06_10.certificate.json")
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    _, missing, _ = _resolve_receipts(cert_path, cert, REPO)
    assert "mind_v0_validation.json" in missing
    detail = classify_missing(cert_path, cert, missing, REPO)["mind_v0_validation.json"]
    assert detail["status"] == "changed", detail
