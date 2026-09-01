"""h_mapping.json is the declared answer to 'who handed the verifier this token?', and it must
cover every obligation source the verifier can emit — or a new clause silently creates a stratum
of the corpus that no handedness figure accounts for.

Companion: papers/closed-model-frontier/DECLARATION_h_mapping_2026_09_01.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from styxx.certify import _EPISTEMICS_SOURCES

ROOT = Path(__file__).resolve().parent.parent
ARC = ROOT / "papers" / "closed-model-frontier"
MAPPING = ARC / "h_mapping.json"
RECEIPT = ARC / "h_mapping_census_result.json"
DECLARATION = ARC / "DECLARATION_h_mapping_2026_09_01.md"


@pytest.fixture(scope="module")
def mapping():
    return json.loads(MAPPING.read_text(encoding="utf-8"))


def test_every_source_the_verifier_can_emit_is_declared(mapping):
    """LOAD-BEARING. An emittable source with no declared class is an unaccounted stratum."""
    missing = sorted(set(_EPISTEMICS_SOURCES) - set(mapping["declared_sources"]))
    assert not missing, (f"styxx.certify can emit obligation sources the mapping does not "
                         f"declare: {missing}. Declare them in {MAPPING.name} first.")


def test_emittable_flags_agree_with_the_verifier(mapping):
    for name, src in mapping["declared_sources"].items():
        assert src["emittable"] == (name in _EPISTEMICS_SOURCES), name


def test_every_declared_source_has_a_class_from_the_closed_vocabulary(mapping):
    classes = set(mapping["handedness_classes"])
    for name, src in mapping["declared_sources"].items():
        assert src["handed_by"] in classes, (name, src["handed_by"])
        assert src["clause"], name


def test_the_receipt_was_folded_under_this_mapping_and_this_verifier(mapping):
    rec = json.loads(RECEIPT.read_text(encoding="utf-8"))
    import hashlib
    assert rec["mapping_sha256"] == hashlib.sha256(MAPPING.read_bytes()).hexdigest(), (
        "h_mapping.json changed since the census ran — re-run h_mapping_census.py")
    assert rec["sources"]["emittable_by_this_verifier"] == sorted(_EPISTEMICS_SOURCES), (
        "the verifier's emittable sources moved since the census ran — re-run the census")


def test_the_two_populations_are_never_pooled():
    rec = json.loads(RECEIPT.read_text(encoding="utf-8"))
    printed, live = rec["population_PRINTED"], rec["population_LIVE"]
    assert printed["obligated_tokens"] > 0 and live["obligated_tokens"] > 0
    assert live["obligated_tokens"] != printed["obligated_tokens"]
    # no top-level key carries a token count: a pooled figure would have to live at the top
    for k, v in rec.items():
        assert not (isinstance(v, int) and k.endswith("tokens")), f"pooled count at top level: {k}"
    assert "never_pool" in rec


def test_both_denominators_are_named_in_the_receipt():
    rec = json.loads(RECEIPT.read_text(encoding="utf-8"))
    live = rec["population_LIVE"]
    assert "source_share_of_obligated" in live and "source_share_of_verified" in live
    voc_obl = live["source_share_of_obligated"]["vocabulary"]
    voc_ver = live["source_share_of_verified"]["vocabulary"]
    assert voc_obl > voc_ver, "the two denominators must give different shares or the point is moot"


def test_instrument_level_is_a_separate_population_with_sworn_as_external(mapping):
    rows = mapping["instrument_level"]["rows"]
    assert any("sworn" in k for k in rows)
    sworn = next(v for k, v in rows.items() if "sworn" in k)
    assert sworn["handed_by"] == ["external"]
    assert sworn["finds_its_own_target"] is False
    assert "do_not_conflate" in mapping["instrument_level"]


def test_the_declaration_cites_its_mapping_receipt_and_gate():
    txt = DECLARATION.read_text(encoding="utf-8")
    for needle in ("h_mapping.json", "h_mapping_census_result.json", "tests/test_h_mapping.py",
                   "UNVERIFIED"):
        assert needle in txt, needle
