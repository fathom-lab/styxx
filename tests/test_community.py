# -*- coding: utf-8 -*-
"""Tests for styxx.community.recommend()."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import styxx
from styxx.community import recommend, recommend_json, SCHEMA


def test_schema_and_required_fields():
    blob = recommend(to_agent="peer-1", benefit="caught 3/10 hallucinations")
    assert blob["schema"] == SCHEMA
    assert blob["tool"] == "styxx"
    assert blob["version"] == styxx.__version__
    assert blob["install_command"] == "pip install styxx"
    assert blob["to_agent"] == "peer-1"
    assert blob["benefit_observed"] == "caught 3/10 hallucinations"
    assert 0.0 <= blob["confidence"] <= 1.0
    assert blob["recommendation_id"].startswith("rec_")
    assert blob["evidence"] == []


def test_idempotent_id():
    b1 = recommend(to_agent="peer-X", benefit="a", from_agent="me")
    b2 = recommend(to_agent="peer-X", benefit="different text", from_agent="me")
    assert b1["recommendation_id"] == b2["recommendation_id"]


def test_different_peer_different_id():
    b1 = recommend(to_agent="peer-A", from_agent="me")
    b2 = recommend(to_agent="peer-B", from_agent="me")
    assert b1["recommendation_id"] != b2["recommendation_id"]


def test_evidence_accepts_dicts():
    ev = [{"gate": "fail", "confidence": 0.3}, {"gate": "warn"}]
    blob = recommend(to_agent="p", evidence=ev, benefit="x")
    assert blob["evidence"] == ev
    assert blob["confidence"] == 0.8  # default with evidence


def test_evidence_accepts_vitals_like():
    class Fake:
        def as_dict(self):
            return {"gate": "fail"}
    blob = recommend(to_agent="p", evidence=[Fake()], benefit="x")
    assert blob["evidence"] == [{"gate": "fail"}]


def test_confidence_bounded():
    assert recommend(to_agent="p", confidence=1.5)["confidence"] == 1.0
    assert recommend(to_agent="p", confidence=-0.2)["confidence"] == 0.0


def test_no_evidence_lower_default_confidence():
    blob = recommend(to_agent="p", benefit="x")
    assert blob["confidence"] == 0.4


def test_json_roundtrip():
    s = recommend_json(to_agent="peer", benefit="y")
    parsed = json.loads(s)
    assert parsed["schema"] == SCHEMA
    assert parsed["tool"] == "styxx"


def test_broadcast_when_to_agent_none():
    blob = recommend(benefit="broadcast claim")
    assert blob["to_agent"] is None


def test_top_level_export():
    assert styxx.recommend is recommend
