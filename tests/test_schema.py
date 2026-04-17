# -*- coding: utf-8 -*-
"""Tests for styxx.schema JSON Schema definitions."""

import json
import pytest

import styxx
from styxx import schema
from styxx.errors import StyxxError, StyxxConfigError


def _vitals(kind="reasoning"):
    from styxx.cli import _load_demo_trajectories
    data = _load_demo_trajectories()
    t = data["trajectories"][kind]
    return styxx.Raw().read(
        entropy=t["entropy"],
        logprob=t["logprob"],
        top2_margin=t["top2_margin"],
    )


def test_vitals_schema_exports():
    assert "classification" in schema.VITALS_SCHEMA["properties"]
    assert "gate" in schema.VITALS_SCHEMA["properties"]
    assert schema.VITALS_SCHEMA["properties"]["confidence"]["maximum"] == 1.0


def test_vitals_to_dict_valid():
    v = _vitals("reasoning")
    d = v.to_dict()
    assert d["classification"] in schema.CLASSIFICATIONS
    assert d["gate"] in schema.GATES
    assert 0.0 <= d["confidence"] <= 1.0
    assert 0.0 <= d["trust"] <= 1.0


def test_vitals_to_json_roundtrip():
    v = _vitals("reasoning")
    s = v.to_json()
    loaded = json.loads(s)
    assert loaded["classification"] == v.classification
    # jsonl is one line
    assert "\n" not in v.to_jsonl()


def test_jsonschema_validation():
    jsonschema = pytest.importorskip("jsonschema")
    v = _vitals("reasoning")
    jsonschema.validate(v.to_dict(), schema.VITALS_SCHEMA)


def test_styxx_error_schema_validation():
    jsonschema = pytest.importorskip("jsonschema")
    e = StyxxConfigError("styxx.config.missing", "nope", retry=True, reason="test")
    jsonschema.validate(e.to_dict(), schema.STYXX_ERROR_SCHEMA)
    assert e.to_dict()["retry"] is True
    assert json.loads(e.to_json())["code"] == "styxx.config.missing"


def test_styxx_error_default_severity():
    e = StyxxError(message="boom")
    assert e.severity == "error"
    assert e.retry is False


def test_public_exports():
    assert hasattr(styxx, "schema")
    assert hasattr(styxx, "StyxxError")
    assert hasattr(styxx, "is_agent_mode")
    assert "StyxxError" in styxx.__all__
    assert "schema" in styxx.__all__
    assert "is_agent_mode" in styxx.__all__
