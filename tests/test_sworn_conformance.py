"""The sworn conformance set replays through styxx.sworn, byte for byte, with nothing skipped.

Spec: papers/sworn/SPEC_sworn_conformance_vectors_v01_2026_09_05.md, "Tests this spec commits to".
LOAD-BEARING: test_every_vector_replays_through_styxx_sworn. A vector that does not replay is a
verifier whose behaviour moved; the answer is the generator refusing (C6), never this test skipping.
The set is loaded at import so a missing set is a collection error, not a skip.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "conformance" / "sworn"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.sworn.replay import blob_bytes, replay_vector  # noqa: E402
from styxx import sworn  # noqa: E402
from styxx.sworn import REASONS, VERDICTS  # noqa: E402


def _load(name):
    return json.loads((SET / name).read_bytes().decode("utf-8"))


INDEX = _load("index.json")
BLOBS = _load("blobs.json")
FAMILIES = {name: _load(meta["file"])["vectors"] for name, meta in INDEX["families"].items()}
ALL = [v for name in sorted(FAMILIES) for v in FAMILIES[name]]
FAMILY_NAMES = sorted(FAMILIES)


def _jcs_sha(obj):
    return sworn._sha256(sworn._jcs(obj).encode("utf-8"))


# ============================================================================ bytes and digests

def test_the_set_is_where_the_spec_says_and_wears_the_spec_s_strings():
    assert INDEX["schema"] == "styxx.sworn.conformance/v1"
    assert INDEX["format"] == sworn.SPEC and INDEX["clock"] == "2026-09-01T00:00:00Z"
    assert INDEX["label"] == "the precondition for any second verifier; no claim"
    assert INDEX["core_definition"] == ["schema", "format", "document", "commit", "manifest_digest", "spans",
                                        "counts", "sworn_total", "unresolved", "document_verdict",
                                        "document_malformed", "rungs", "certifies"]
    assert INDEX["sources"] == ["tests/test_sworn.py", "tests/test_sworn_attacks.py"]
    assert INDEX["vector_count"] == len(ALL) == sum(m["count"] for m in INDEX["families"].values())
    assert set(INDEX["requires_legend"]) == {"manifest", "tree", "git", "observer"}
    observer = _load("observer.json")
    assert set(observer["vectors"]) <= {v["id"] for v in ALL}, "observer.json keys are vector ids"
    assert "observer" not in {k for k, v in INDEX["families"].items()}, "the observer is not a family"


def test_every_blob_hashes_to_its_key():
    assert len(BLOBS) == INDEX["blobs"]["count"]
    for sha in BLOBS:
        blob_bytes(BLOBS, sha)              # raises BadBlob on a mismatch


def test_every_family_file_hashes_to_its_index_entry_and_the_set_digest_re_derives():
    for name, meta in INDEX["families"].items():
        assert sworn._sha256((SET / meta["file"]).read_bytes()) == meta["sha256"], name
        assert meta["count"] == len(FAMILIES[name])
    assert sworn._sha256((SET / "blobs.json").read_bytes()) == INDEX["blobs"]["sha256"]
    body = {k: v for k, v in INDEX.items() if k not in ("set_sha256", "provenance")}
    assert _jcs_sha(body) == INDEX["set_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", INDEX["set_sha256"])


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_every_vector_id_re_derives_from_its_mode_and_inputs_and_names_only_blobs_in_the_store(family):
    ids = []
    for v in FAMILIES[family]:
        assert v["id"] == _jcs_sha({"mode": v["mode"], "inputs": v["inputs"]})
        assert v["family"] == family
        assert v["sources"] == sorted(set(v["sources"])) and v["sources"]
        for key in ("document", "sidecar", "manifest", "receipt"):
            if v["inputs"].get(key) is not None:
                assert v["inputs"][key] in BLOBS, (v["id"], key)
        tree = v["inputs"].get("tree")
        if tree is not None:
            for path, e in tree["entries"].items():
                assert e["sha256"] is None or e["sha256"] in BLOBS, (v["id"], path)
        e = v["expect"]
        if e["outcome"] == "core":
            assert e["core_sha256"] in BLOBS
            assert json.loads(blob_bytes(BLOBS, e["core_sha256"]).decode("utf-8"))["document_verdict"] == e["document_verdict"]
        if e["outcome"] == "sidecar":
            assert e["sidecar"] in BLOBS
        ids.append(v["id"])
    assert ids == sorted(ids) and len(set(ids)) == len(ids), "vectors sorted by id, no duplicates"


# ============================================================================ the replay

@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_every_vector_replays_through_styxx_sworn(family):
    """LOAD-BEARING. Cores by core_sha256 and floor, refusals by match, sidecars by sidecar_sha256
    and a byte-exact render, manifests by digest and rung status, checks by the three fields."""
    failures, skipped = [], []
    for v in FAMILIES[family]:
        status, detail = replay_vector(v, BLOBS)
        if status == "fail":
            failures.append((v["id"][:12], v["sources"][0], detail))
        elif status == "skip":
            skipped.append(v["id"])
    assert not failures, failures[:5]
    assert not skipped, "nothing is skipped in v0.1: no vector requires git"
    assert all("git" not in v["requires"] for v in FAMILIES[family])


def test_requires_is_exactly_what_the_inputs_carry():
    for v in ALL:
        want = []
        if v["inputs"].get("manifest") is not None:
            want.append("manifest")
        if v["inputs"].get("tree") is not None:
            want.append("tree")
        assert v["requires"] == want, v["id"]
    # the fuzz family, every sidecar and load refusal and every manifest shape need no tree
    assert all("tree" not in v["requires"] for v in FAMILIES["fuzz"])
    assert all("tree" not in v["requires"] for v in ALL if v["mode"] in ("canon", "load", "manifest"))


# ============================================================================ the rule contract

_CORES = {}


def core_of(v):
    sha = v["expect"]["core_sha256"]
    if sha not in _CORES:
        _CORES[sha] = json.loads(blob_bytes(BLOBS, sha).decode("utf-8"))
    return _CORES[sha]


def spans_of(v):
    return v["expect"]["spans"] if v["expect"]["outcome"] == "core" else []


def core_spans(v):
    return core_of(v)["spans"] if v["expect"]["outcome"] == "core" else []


def rungs_of(v):
    return v["expect"]["rungs"] if v["expect"]["outcome"] == "core" else {}


def floor_of(v):
    return v["expect"]["floor"] if v["expect"]["outcome"] == "core" else None


def check_of(v):
    return v["expect"]["check"] if v["expect"]["outcome"] == "check" else None


def dv(v):
    return v["expect"].get("document_verdict")


def prov(s):
    return s.get("provenance") or {}


_FRAG = re.compile(r"^r[1-9][0-9]*#")


def rec(s):
    """A span's receipt, '' when the lexer found none (a tag_syntax MALFORMED span has no receipt)."""
    return s.get("receipt") or ""


RULES = {
    "R1": (lambda v: any(_FRAG.match(rec(s)) and s["verdict"] == "HELD" for s in spans_of(v)),
           lambda v: any(_FRAG.match(rec(s)) and s["verdict"] == "MALFORMED"
                         and s["reason"] in ("receipt_form", "absent_over_partial", "hash_over_partial")
                         for s in spans_of(v))),
    "R2": (lambda v: "R2" in v["rules"] and any(s["reason"] != "hidden_commitment" for s in spans_of(v)),
           lambda v: any(s["reason"] == "hidden_commitment" for s in spans_of(v))),
    "R3": (lambda v: any(s["kind"] == "quote" and s["verdict"] == "HELD" and "#" in rec(s) for s in spans_of(v)),
           lambda v: any(s["reason"] == "short_needle" for s in spans_of(v))),
    "R4": (lambda v: "R4" in v["rules"] and any(s["reason"] != "length_cap" for s in spans_of(v)),
           lambda v: any(s["reason"] == "length_cap" for s in spans_of(v))),
    "R5": (lambda v: "R5" in v["rules"] and any(s["verdict"] == "HELD" and prov(s).get("kind_of_source") == "attestation"
                                                for s in core_spans(v)),
           lambda v: any(s["reason"] in ("kind_of_source_unknown", "receipt_author_minted") for s in spans_of(v))),
    "R6": (lambda v: any(k in ("L1", "L2") for k in rungs_of(v)),
           lambda v: any(s["reason"] == "rung_unknown" for s in spans_of(v))),
    "R7": (lambda v: "R7" in v["rules"] and any(prov(s).get("form") in ("rn", "path", "prereg") for s in core_spans(v)),
           lambda v: "unresolved" in rungs_of(v)),
    "R8": (lambda v: floor_of(v) is not None and isinstance(floor_of(v)["sentence_share"], float),
           lambda v: floor_of(v) is not None and floor_of(v)["sentence_share"] is None),
    "R9": (lambda v: (check_of(v) or {}).get("status") == "VERIFIED",
           lambda v: (check_of(v) or {}).get("status") == "FAILED"),
}

ATTACKS = {
    "A1": lambda v: any(s["kind"] == "numeric" and s["verdict"] == "HELD" for s in spans_of(v)),
    "A2": lambda v: dv(v) == "SWORN-HELD" and floor_of(v)["narrative_sentences"] > 0 and floor_of(v)["sentence_share"] < 1,
    "A3": lambda v: dv(v) == "SWORN-HELD" and floor_of(v)["narrative_sentences"] > 0 and floor_of(v)["sentence_share"] < 1,
    "A4": lambda v: dv(v) == "SWORN-FAILED" and any(s["reason"] == "hidden_commitment" for s in spans_of(v)),
    "A5": lambda v: any(s["verdict"] == "HELD" and prov(s).get("form") == "path" for s in core_spans(v)),
    "A6": lambda v: any(s["verdict"] == "HELD" and prov(s).get("form") == "prereg" for s in core_spans(v)),
    "A7": lambda v: any(s["verdict"] == "HELD" and prov(s).get("rung") == "undeclared" for s in core_spans(v)),
    "A8": lambda v: dv(v) == "SWORN-FAILED" and any(rec(s).startswith("path:") and s["reason"] == "value_mismatch"
                                                     for s in spans_of(v)),
    "A9": lambda v: any(s["reason"] == "value_mismatch" for s in spans_of(v)),
    "A10": lambda v: any(s["reason"] == "short_needle" for s in spans_of(v)),
    "A11": lambda v: any(s["verdict"] == "HELD" for s in spans_of(v)),
    "A12": lambda v: any(s["verdict"] == "HELD" for s in spans_of(v)),
}


def test_the_contract_names_every_rule_and_every_battery_row_and_nothing_else():
    assert set(INDEX["rule_contract"]) == set(RULES) | set(ATTACKS)
    for r in RULES:
        assert set(INDEX["rule_contract"][r]) == {"positive", "negative"}, r
    for a in ATTACKS:
        assert set(INDEX["rule_contract"][a]) == {"shows"}, a


@pytest.mark.parametrize("rule", sorted(RULES, key=lambda r: int(r[1:])))
def test_every_v02_rule_has_a_positive_and_a_negative_vector(rule):
    positive, negative = RULES[rule]
    tagged = [v for v in ALL if rule in v["rules"]]
    assert tagged, "no vector is tagged %s" % rule
    assert any(positive(v) for v in ALL), "%s: no positive vector (%s)" % (rule, INDEX["rule_contract"][rule]["positive"])
    assert any(negative(v) for v in ALL), "%s: no negative vector (%s)" % (rule, INDEX["rule_contract"][rule]["negative"])


@pytest.mark.parametrize("row", sorted(ATTACKS, key=lambda r: int(r[1:])))
def test_every_battery_row_has_a_vector_that_shows_its_verdict(row):
    tagged = [v for v in ALL if row in v["rules"]]
    assert tagged, "no vector is tagged %s" % row
    assert any(ATTACKS[row](v) for v in tagged), "%s: no tagged vector shows %s" % (row, INDEX["rule_contract"][row]["shows"])


# ============================================================================ the closed sets

def test_every_reason_and_verdict_is_produced_by_a_vector_or_listed_unvectored():
    reasons, verdicts = set(), set()
    for v in ALL:
        for s in spans_of(v):
            verdicts.add(s["verdict"])
            if s["reason"] is not None:
                reasons.add(s["reason"])
        if v["expect"]["outcome"] == "core":
            dm = core_of(v).get("document_malformed")
            if isinstance(dm, dict) and dm.get("reason"):
                reasons.add(dm["reason"])
    assert reasons <= set(REASONS) and verdicts <= set(VERDICTS)
    unvectored = INDEX["unvectored"]
    assert sorted(set(REASONS) - reasons) == unvectored["reasons"]
    assert sorted(set(VERDICTS) - verdicts) == unvectored["verdicts"]
    # C5: the one reason the verifier declares and never emits, the one only a missing git binary
    # produces, and the one verdict with no producer. Nothing else is missing.
    assert unvectored["reasons"] == ["git_unavailable", "manifest_spec_unknown"]
    assert unvectored["verdicts"] == ["WITHHELD"]
    assert {"receipt_too_large", "not_a_blob", "commit_absent"} <= reasons, "C10: the snapshot carries these"


def test_nothing_unvectorable_was_dropped_silently():
    skipped = INDEX["unvectored"]["skipped"]
    assert skipped, "C5: the NaN manifest and the whole-repository tree are listed"
    whys = " ".join(s["why"] for s in skipped)
    assert "NaN" in whys or "not JSON-representable" in whys
    assert "the repository this set lives in" in whys
    for s in skipped:
        assert set(s) == {"source", "where", "why"} and s["source"].startswith("tests/")


def test_every_refusal_code_in_the_table_is_produced_or_says_where_it_was_not():
    codes = INDEX["refusal_codes"]
    seen = {}
    for v in ALL:
        if v["expect"]["outcome"] == "refused":
            r = v["expect"]["refusal"]
            assert r["code"] in codes and codes[r["code"]]["match"] == r["match"]
            seen.setdefault(r["code"], set()).add(r["where"])
    for code, meta in codes.items():
        assert sorted(seen.get(code, ())) == meta["where"], code
    assert INDEX["unvectored"]["refusal_codes"] == sorted(c for c, m in codes.items() if not m["where"])


# ============================================================================ the layout

def _tracked():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "conformance/"], capture_output=True, check=True).stdout
    return sorted(p for p in out.decode("utf-8").split("\0") if p)


def _check_attr(path):
    out = subprocess.run(["git", "-C", str(ROOT), "check-attr", "text", "--", path],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return out.strip().rsplit(": ", 1)[-1]


def test_every_tracked_conformance_file_is_eol_pinned_and_holds_no_carriage_return():
    files = _tracked()
    assert files, "the set is not tracked"
    for rel in files:
        assert _check_attr(rel) == "unset", rel
        assert b"\r" not in (ROOT / rel).read_bytes(), rel


def test_no_conformance_file_wears_a_suffix_another_sweep_claims():
    for rel in _tracked():
        name = Path(rel).name
        assert not name.endswith((".sworn.json", ".sworn-receipt.json", ".certificate.json")), rel
        assert "result" not in name.lower(), rel
    for p in SET.rglob("*"):
        if p.is_file():
            assert not p.name.endswith((".sworn.json", ".sworn-receipt.json", ".certificate.json")), p


def test_the_committed_set_regenerates_to_its_own_digest():
    """C7: gen_vectors.py --check regenerates in memory and exits 1 if set_sha256 differs."""
    r = subprocess.run([sys.executable, "conformance/sworn/gen_vectors.py", "--check"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1000:]
    assert INDEX["set_sha256"] in r.stdout and "CHECK OK" in r.stdout
