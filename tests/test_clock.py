# -*- coding: utf-8 -*-
"""styxx.clock — memo exactness and chain re-verification, with the chain replaced by a fixture.

Every status the module documents is produced here from a transaction shaped like the chain's
jsonParsed encoding. The chain itself is never touched by a test.
"""
from __future__ import annotations

import json

import pytest

from styxx import beacon
from styxx import clock as ledger

D = "fc8ad3a52de50d106e90c9d0a5441a6460df0f5c80cb5553dad7dbba110cd5a5"
MEMO_PROGRAM = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ATTACKER = "ATTACKER1111111111111111111111111111111111111"
BLOCKHASH_BYTES = bytes(range(1, 33))
BLOCKHASH = ledger._b58encode(BLOCKHASH_BYTES)


def test_memo_is_exact_and_refuses_bad_input():
    assert ledger.memo("sworn-receipt", D) == f"styxx sworn-receipt {D}"
    assert D in ledger.command("sealed-prereg", D)
    with pytest.raises(ValueError):
        ledger.memo("sworn-receipt", "abc")
    with pytest.raises(ValueError):
        ledger.memo("tweet", D)
    with pytest.raises(ValueError):
        ledger.memo("sworn-receipt", D[:-1])
    with pytest.raises(ValueError):
        ledger.memo("sworn-receipt", D + "0")


def test_memo_takes_the_digest_as_written_and_never_folds_it():
    # an uppercase or padded digest is refused, not normalised: the memo on chain must be the
    # exact string a document names
    with pytest.raises(ValueError):
        ledger.memo("sealed-prereg", D.upper())
    with pytest.raises(ValueError):
        ledger.memo("sealed-prereg", " " + D)


def test_base58_round_trips_including_leading_zero_bytes():
    for raw in (BLOCKHASH_BYTES, b"\x00\x00" + bytes(range(30)), b"\x00" * 32):
        assert ledger._b58decode(ledger._b58encode(raw)) == raw
    with pytest.raises(ValueError):
        ledger._b58decode("0OIl")  # not base58 characters


def test_blockhash_to_beacon_is_the_32_bytes_as_hex_and_refuses_other_lengths():
    assert ledger.blockhash_to_beacon(BLOCKHASH) == BLOCKHASH_BYTES.hex()
    with pytest.raises(ValueError):
        ledger.blockhash_to_beacon("B" * 10)


def _tx(memo_text, *, slot=123, block_time=1_757_700_000, signer=ledger.CREATOR, err=None,
        transfer="checked", memos_before=()):
    ins = [{"program": "spl-memo", "programId": MEMO_PROGRAM, "parsed": m} for m in memos_before]
    if transfer == "checked":
        ins.append({"program": "spl-token", "programId": TOKEN_PROGRAM,
                    "parsed": {"type": "transferChecked",
                               "info": {"mint": ledger.MINT, "authority": signer, "tokenAmount": {"amount": "1"}}}})
    elif transfer == "plain":
        ins.append({"program": "spl-token", "programId": TOKEN_PROGRAM,
                    "parsed": {"type": "transfer", "info": {"authority": signer, "amount": "1"}}})
    if memo_text is not None:
        ins.append({"program": "spl-memo", "programId": MEMO_PROGRAM, "parsed": memo_text})
    meta = {"err": err, "innerInstructions": []}
    if transfer == "plain":
        meta["preTokenBalances"] = [{"mint": ledger.MINT, "owner": signer}]
        meta["postTokenBalances"] = [{"mint": ledger.MINT, "owner": signer}]
    return {"slot": slot, "blockTime": block_time,
            "transaction": {"message": {"accountKeys": [{"pubkey": signer, "signer": True, "writable": True}],
                                        "instructions": ins}},
            "meta": meta}


def _chain(tx, blk=None, block_raises=False, tx_raises=False, history=None, history_raises=False):
    """history: the wallet's signature listing, newest first, as getSignaturesForAddress returns it; by
    default the recorded transaction 'sig1' alone, carrying the memo of the given tx."""
    def fetch(method, params, rpcs=None):
        if method == "getTransaction":
            if tx_raises:
                raise RuntimeError("rpc getTransaction failed: 429")
            return tx
        if method == "getBlock":
            if block_raises:
                raise RuntimeError("rpc getBlock failed: slot was skipped")
            return blk if blk is not None else {"blockhash": BLOCKHASH}
        if method == "getSignaturesForAddress":
            if history_raises:
                raise RuntimeError("rpc getSignaturesForAddress failed")
            if history is not None:
                opts = params[1]
                page = history
                if opts.get("before"):
                    idx = [h["signature"] for h in history].index(opts["before"])
                    page = history[idx + 1:]
                return page[: opts["limit"]]
            # the chain's listing carries every memo of a transaction, "[len] text" joined with "; "
            memos = ledger.parse_memos(tx) if tx else []
            return [{"signature": "sig1", "slot": (tx or {}).get("slot", 123), "err": None,
                     "memo": "; ".join(f"[{len(m)}] {m}" for m in memos) if memos else None}]
        raise AssertionError(method)
    return fetch


def _line(kind="sealed-prereg", **extra):
    return {"n": 1, "kind": kind, "digest": D, "tx": "sig1", **extra}


def _check(line, tx, **kw):
    return ledger.check_line(line, fetch=_chain(tx, **kw))


def test_an_honest_seal_reads_anchored_and_the_beacon_is_the_blockhash_as_hex():
    r = _check(_line(memo=ledger.memo("sealed-prereg", D)), _tx(ledger.memo("sealed-prereg", D)))
    assert r["status"] == "ANCHORED", r
    assert r["slot"] == 123
    assert r["block_time"] == "2025-09-12T18:00:00Z"
    assert r["beacon_b58"] == BLOCKHASH
    assert r["beacon"] == BLOCKHASH_BYTES.hex()
    assert all(r["checks"].values())
    # and what the clock hands over, the beacon takes: the two modules compose
    assert len(beacon.select(r["beacon"], 5)) == 5


def test_a_receipt_anchor_needs_no_beacon():
    r = _check(_line("sworn-receipt"), _tx(ledger.memo("sworn-receipt", D)))
    assert r["status"] == "ANCHORED"
    assert "beacon" not in r


def test_a_failed_transaction_is_not_an_anchor():
    r = _check(_line(), _tx(ledger.memo("sealed-prereg", D), err={"InstructionError": [0, "Custom"]}))
    assert r["status"] == "FAILED_TX"
    assert r["checks"]["succeeded"] is False


def test_a_memo_from_another_wallet_is_not_the_lab_seal():
    r = _check(_line(), _tx(ledger.memo("sealed-prereg", D), signer=ATTACKER))
    assert r["status"] == "NOT_CREATOR"
    assert r["signer"] == ATTACKER


def test_a_memo_without_a_styxx_transfer_is_not_an_anchor():
    r = _check(_line(), _tx(ledger.memo("sealed-prereg", D), transfer=None))
    assert r["status"] == "NO_TRANSFER"


def test_a_plain_transfer_is_recognised_by_the_token_balance_deltas():
    r = _check(_line(), _tx(ledger.memo("sealed-prereg", D), transfer="plain"))
    assert r["status"] == "ANCHORED"


def test_a_memo_that_differs_on_chain_is_a_mismatch_not_a_pass():
    r = _check(_line("sworn-receipt"), _tx("styxx sworn-receipt " + "0" * 64))
    assert r["status"] == "MEMO_MISMATCH"
    assert r["memo_on_chain"] == "styxx sworn-receipt " + "0" * 64


def test_a_decoy_memo_before_the_real_one_does_not_hide_it():
    r = _check(_line(), _tx(ledger.memo("sealed-prereg", D), memos_before=("decoy", "styxx sealed-prereg " + "1" * 64)))
    assert r["status"] == "ANCHORED"
    assert r["memo_on_chain"] == ledger.memo("sealed-prereg", D)
    assert len(r["memos_on_chain"]) == 3


def test_an_undecodable_memo_instruction_does_not_hide_the_next_one():
    tx = _tx(ledger.memo("sealed-prereg", D))
    tx["transaction"]["message"]["instructions"].insert(0, {"programId": MEMO_PROGRAM, "data": "3yZe7d"})  # not utf-8
    r = _check(_line(), tx)
    assert r["status"] == "ANCHORED"


def test_a_line_whose_own_memo_disagrees_with_its_digest_never_reaches_the_chain():
    def never(method, params, rpcs=None):
        raise AssertionError("the chain was consulted for a line that is malformed on its face")
    r = ledger.check_line(_line(memo="styxx sealed-prereg " + "0" * 64), fetch=never)
    assert r["status"] == "MALFORMED_LINE"


def test_missing_block_time_is_not_anchored_and_is_never_reported_as_1970():
    r = _check(_line(), _tx(ledger.memo("sealed-prereg", D), block_time=None))
    assert r["status"] == "TIME_UNAVAILABLE"
    assert r["block_time"] is None


def test_a_recorded_slot_or_time_that_disagrees_with_the_chain_is_flagged():
    r = _check(_line(slot=999), _tx(ledger.memo("sealed-prereg", D)))
    assert r["status"] == "SLOT_MISMATCH"
    r = _check(_line(block_time="2026-01-01T00:00:00Z"), _tx(ledger.memo("sealed-prereg", D)))
    assert r["status"] == "TIME_MISMATCH"
    r = _check(_line(slot=123, block_time="2025-09-12T18:00:00Z"), _tx(ledger.memo("sealed-prereg", D)))
    assert r["status"] == "ANCHORED"


def test_rpc_failure_is_a_status_not_a_crash_and_the_next_line_is_still_checked(tmp_path):
    p = tmp_path / "anchors.jsonl"
    p.write_text(json.dumps(_line()) + "\n" + json.dumps({**_line("sworn-receipt"), "n": 2}) + "\n")
    rs = ledger.verify(str(p), fetch=_chain(_tx(ledger.memo("sealed-prereg", D)), tx_raises=True))
    assert [r["status"] for r in rs] == ["RPC_ERROR", "RPC_ERROR"]
    assert [r["n"] for r in rs] == [1, 2]


def test_a_malformed_line_is_a_status_not_a_crash():
    r = ledger.check_line({"n": 1, "kind": "sealed-prereg", "digest": D}, fetch=_chain(None))
    assert r["status"] == "MALFORMED_LINE"
    r = ledger.check_line(_line("tweet"), fetch=_chain(None))
    assert r["status"] == "MALFORMED_LINE"
    r = ledger.check_line({**_line(), "digest": D.upper()}, fetch=_chain(None))
    assert r["status"] == "MALFORMED_LINE"


def test_a_transaction_the_chain_does_not_have_is_not_found():
    r = _check(_line(), None)
    assert r["status"] == "NOT_FOUND"


def test_a_seal_whose_block_cannot_be_fetched_is_not_anchored():
    r = _check(_line(), _tx(ledger.memo("sealed-prereg", D)), block_raises=True)
    assert r["status"] == "BEACON_UNAVAILABLE"
    r = _check(_line(), _tx(ledger.memo("sealed-prereg", D)), blk={})
    assert r["status"] == "BEACON_UNAVAILABLE"
    r = _check(_line(), _tx(ledger.memo("sealed-prereg", D)), blk={"blockhash": "B" * 10})
    assert r["status"] == "BEACON_MALFORMED"


def _hist(*entries):
    return [{"signature": s, "slot": slot, "err": err, "memo": memo} for s, slot, err, memo in entries]


def test_an_honest_seal_is_the_earliest_memo_carrying_its_digest():
    m = ledger.memo("sealed-prereg", D)
    r = _check(_line(), _tx(m), history=_hist(("sig1", 123, None, f"[{len(m)}] {m}"), ("older", 100, None, "[5] hello")))
    assert r["status"] == "ANCHORED" and r["checks"]["earliest"] is True
    assert r["earliest_tx"] == "sig1" and r["memos_carrying_digest"] == 1


def test_a_second_anchor_of_the_same_digest_is_not_the_seal():
    # the signer anchored twice and recorded the later one, whose slot drew the canaries they liked
    m = ledger.memo("sealed-prereg", D)
    hist = _hist(("sig1", 123, None, f"[{len(m)}] {m}"), ("sig0", 99, None, f"[{len(m)}] {m}"))
    r = _check(_line(), _tx(m), history=hist)
    assert r["status"] == "EARLIER_MEMO_EXISTS"
    assert r["earliest_tx"] == "sig0" and r["earliest_slot"] == 99 and r["memos_carrying_digest"] == 2
    # a failed earlier attempt does not count: only confirmed memos seal
    hist2 = _hist(("sig1", 123, None, f"[{len(m)}] {m}"), ("sig0", 99, {"InstructionError": [0, "Custom"]}, f"[{len(m)}] {m}"))
    assert _check(_line(), _tx(m), history=hist2)["status"] == "ANCHORED"


def test_a_history_the_scan_cannot_finish_is_unknown_never_anchored():
    m = ledger.memo("sealed-prereg", D)
    hist = _hist(*[(f"s{i}", 5000 - i, None, "[1] x") for i in range(1000 * 50)]) + _hist(("sig1", 123, None, f"[{len(m)}] {m}"))
    r = _check(_line(), _tx(m), history=hist)
    assert r["status"] == "EARLIEST_UNKNOWN"
    r = _check(_line(), _tx(m), history_raises=True)
    assert r["status"] == "EARLIEST_UNKNOWN"
    r = _check(_line(), _tx(m), history=_hist(("other", 50, None, "[5] hello")))
    assert r["status"] == "EARLIEST_UNKNOWN" and "disagree" in r["detail"]


def test_a_receipt_anchor_is_not_scanned_and_scan_can_be_turned_off():
    m = ledger.memo("sealed-prereg", D)
    hist = _hist(("sig1", 123, None, f"[{len(m)}] {m}"), ("sig0", 99, None, f"[{len(m)}] {m}"))
    r = ledger.check_line(_line(), fetch=_chain(_tx(m), history=hist), scan=False)
    assert r["status"] == "ANCHORED" and "earliest" not in r["checks"]
    def no_history(method, params, rpcs=None):
        assert method != "getSignaturesForAddress"
        return _chain(_tx(ledger.memo("sworn-receipt", D)))(method, params, rpcs)
    assert ledger.check_line(_line("sworn-receipt"), fetch=no_history)["status"] == "ANCHORED"


def test_the_checks_map_records_every_check_even_when_an_earlier_one_fails():
    r = _check(_line(), _tx(ledger.memo("sealed-prereg", D), signer=ATTACKER, transfer=None, block_time=None))
    assert r["status"] == "NOT_CREATOR"
    assert r["checks"] == {"succeeded": True, "memo": True, "creator": False, "transfer": False, "time": False,
                           "slot_recorded": True, "time_recorded": True}
