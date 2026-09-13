# -*- coding: utf-8 -*-
"""styxx.clock — memo exactness and chain re-verification, with the chain replaced by a fixture."""
from __future__ import annotations

import json

import pytest

from styxx import clock as ledger

D = "fc8ad3a52de50d106e90c9d0a5441a6460df0f5c80cb5553dad7dbba110cd5a5"


def test_memo_is_exact_and_refuses_bad_input():
    assert ledger.memo("sworn-receipt", D) == f"styxx sworn-receipt {D}"
    assert D in ledger.command("sealed-prereg", D)
    with pytest.raises(ValueError):
        ledger.memo("sworn-receipt", "abc")
    with pytest.raises(ValueError):
        ledger.memo("tweet", D)


def _fake_chain(memo_text: str, slot: int = 123, blockhash: str = "B" * 44):
    def fetch(method, params, rpcs=None):
        if method == "getTransaction":
            return {"slot": slot, "blockTime": 1_757_700_000,
                    "transaction": {"message": {"instructions": [
                        {"program": "spl-token", "parsed": {"type": "transfer"}},
                        {"program": "spl-memo", "programId": "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr",
                         "parsed": memo_text}]}}, "meta": {"innerInstructions": []}}
        if method == "getBlock":
            return {"blockhash": blockhash}
        raise AssertionError(method)
    return fetch


def test_anchor_line_reads_anchored_and_a_seal_returns_the_slot_blockhash_as_beacon(tmp_path):
    p = tmp_path / "anchors.jsonl"
    p.write_text(json.dumps({"n": 1, "kind": "sealed-prereg", "digest": D, "memo": ledger.memo("sealed-prereg", D),
                             "tx": "sig1"}) + "\n")
    rs = ledger.verify(str(p), fetch=_fake_chain(ledger.memo("sealed-prereg", D)))
    assert rs[0]["status"] == "ANCHORED"
    assert rs[0]["beacon"] == "B" * 44
    assert rs[0]["slot"] == 123


def test_a_memo_that_differs_on_chain_is_a_mismatch_not_a_pass(tmp_path):
    p = tmp_path / "anchors.jsonl"
    p.write_text(json.dumps({"n": 1, "kind": "sworn-receipt", "digest": D, "tx": "sig1"}) + "\n")
    rs = ledger.verify(str(p), fetch=_fake_chain("styxx sworn-receipt " + "0" * 64))
    assert rs[0]["status"] == "MEMO_MISMATCH"
