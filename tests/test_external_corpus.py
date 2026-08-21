# -*- coding: utf-8 -*-
"""SP-EXT structural checks.

The external corpus is the artifact other people would rely on, so its
incompleteness has to be machine-visible, not just prose in a paper. These tests
fail if the status marker is dropped, if a case loses its upstream anchor, or if
the recall disclaimer disappears.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parent.parent / "benchmarks" / "silent_pass" / "external.json"


@pytest.fixture(scope="module")
def corpus():
    return json.loads(EXT.read_text(encoding="utf-8"))


def test_incompleteness_is_declared_in_the_data(corpus):
    """A partial corpus that does not say so is the defect it catalogues."""
    assert "INCOMPLETE" in corpus["status"]
    assert "no gate has been applied" in corpus["status"].lower()


def test_recall_is_declared_unknown(corpus):
    """SP-EXT is a lower bound. If this line goes, someone will quote a rate."""
    r = corpus["recall"].upper()
    assert "UNKNOWN" in r and "LOWER BOUND" in r
    assert "never be quoted as a rate" in corpus["recall"]


def test_every_case_is_anchored_upstream(corpus):
    """Ground truth is the upstream maintainers' own fix, not our opinion. A case
    without a fix commit is our opinion."""
    for c in corpus["cases"]:
        assert c["repo"] and "/" in c["repo"], c["id"]
        assert c["fix_commit"], c["id"]
        assert c["url"].startswith("https://github.com/"), c["id"]
        assert c["module"] and c["symbol"], c["id"]


def test_every_case_records_a_consumer(corpus):
    """A value nobody reads is not a silent pass. The consumer is load-bearing."""
    for c in corpus["cases"]:
        assert len(c["consumer"]) > 40, c["id"]


def test_every_case_carries_its_adjudication(corpus):
    for c in corpus["cases"]:
        a = c["adjudication"]
        assert a["reviewers"] >= 3, c["id"]
        rej = a["rejections"]
        assert rej < (a["reviewers"] + 1) // 2, (
            f"{c['id']} was accepted with a majority of rejections")


def test_the_inclusion_rule_is_the_frozen_three(corpus):
    assert len(corpus["inclusion_rule"]) == 3
    joined = " ".join(corpus["inclusion_rule"]).lower()
    assert "did not happen" in joined
    assert "indistinguishable" in joined
    assert "visible" in joined
