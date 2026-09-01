"""Every sworn document committed under papers/ must re-derive, or it is not a sworn document.

Two checks per document, in the spirit of tests/test_certificate_reproduces.py: the sidecar
renders byte-for-byte to the committed inline document, and the committed verdict receipt
re-derives at the commit the sidecar names. A receipt that stops re-deriving is the designed
drift signal — a receipt file moved under a span that swore to it, or the observer's version
changed — and it fails here rather than sitting quietly in the tree.

The commit must be reachable: on a shallow CI clone this test repairs its precondition the way
tests/test_ledger.py does, and fails in CI (never skips) if it cannot.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from styxx.sworn import GitTree, load_sidecar, render, verify_receipt
from tests.test_ledger import _in_ci, has_full_history

ROOT = Path(__file__).resolve().parent.parent


def sworn_documents():
    for side in sorted(ROOT.glob("papers/**/*.sworn.json")):
        stem = side.name[:-len(".sworn.json")]
        yield pytest.param(side, side.with_name(stem + ".md"),
                           side.with_name(stem + ".sworn-receipt.json"), id=stem)


DOCS = list(sworn_documents())


def test_there_is_at_least_one_sworn_document_in_the_tree():
    assert DOCS, "the lab swears to nothing?"


@pytest.mark.parametrize("side,doc,rec", DOCS)
def test_the_sidecar_renders_to_the_committed_document_bytes(side, doc, rec):
    obj = load_sidecar(json.loads(side.read_text(encoding="utf-8")))
    assert render(obj) == doc.read_bytes(), "%s: sidecar and document disagree" % doc.name


@pytest.mark.parametrize("side,doc,rec", DOCS)
def test_the_verdict_receipt_re_derives_at_the_commit_it_names(side, doc, rec):
    obj = load_sidecar(json.loads(side.read_text(encoding="utf-8")))
    receipt = json.loads(rec.read_text(encoding="utf-8"))
    assert receipt["commit"] == obj["commit"]
    tree = GitTree(ROOT, obj["commit"])
    if obj["commit"] is not None and tree._ready() == "commit_absent":
        has_full_history(ROOT)
        if tree._ready() == "commit_absent":
            msg = "commit %s is not in this checkout" % obj["commit"][:12]
            if _in_ci():
                pytest.fail(msg + " even after unshallowing — the receipt cannot be re-derived")
            pytest.skip(msg)
    res = verify_receipt(receipt, sidecar=obj, tree=tree)
    assert res["status"] == "VERIFIED", res
    assert receipt["document_verdict"] in ("SWORN-HELD", "SWORN-FAILED", "UNSWORN")


@pytest.mark.parametrize("side,doc,rec", DOCS)
def test_a_committed_receipt_prints_its_coverage_and_its_boundary(side, doc, rec):
    receipt = json.loads(rec.read_text(encoding="utf-8"))
    assert receipt["coverage"]["advisory"] is True
    assert "NOT a claim that the document is correct" in receipt["certifies"]
