# -*- coding: utf-8 -*-
"""Replay every committed numerals conformance vector through the installed reader.

`conformance/numerals/vectors_*.json`: the `reader` and `binding` families pin what the numeral reader
v2 reads under the flags each set names (`vectors_v2.json`: the earlier build session's default, bare
magnitudes and form noise on; `vectors_v2_default.json`: the shipped default of RULE_final 0.1); the
`v1_reader` and `v1_binding` families pin what the reader at `879f6be9` read (recorded there, before the
v2 code existed) and are replayed with the master flag off, so "V15_READER_V2 = False behaves as HEAD" is
a test, not a sentence. Nothing here is skipped: a vector that does not reproduce fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NUMERALS = ROOT / "conformance" / "numerals"
sys.path.insert(0, str(NUMERALS))
import gen_vectors as G  # noqa: E402

SETS = sorted(NUMERALS.glob("vectors_*.json"))


def test_there_is_a_committed_set():
    assert SETS, "conformance/numerals/vectors_*.json is missing"


@pytest.mark.parametrize("path", SETS, ids=lambda p: p.name)
def test_the_set_digest_re_derives(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    core = {k: v for k, v in doc.items() if k not in ("set_sha256", "provenance")}
    assert G._sha(G.jcs(core)) == doc["set_sha256"]
    assert doc["schema"] == G.SCHEMA
    assert set(doc["families"]) >= {"reader", "binding", "v1_reader", "v1_binding"}
    for fam in doc["families"].values():
        assert fam["n"] == len(fam["vectors"]) > 0
        ids = [v["id"] for v in fam["vectors"]]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)


@pytest.mark.parametrize("path", SETS, ids=lambda p: p.name)
def test_every_vector_replays(path):
    bad = G.replay(json.loads(path.read_text(encoding="utf-8")))
    assert bad == [], f"{len(bad)} vector(s) do not reproduce; the three shown are: {bad[:3]}"


@pytest.mark.parametrize("path", SETS, ids=lambda p: p.name)
def test_the_v1_families_were_recorded_from_head_before_the_build(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    rec = doc["provenance"]["v1_recording"]
    assert rec["head"].startswith("879f6be9")
    assert rec["certify_sha256_lf"].startswith("c448ded108df0f54")       # styxx/certify.py at 078b0a16 / 879f6be9, LF
    assert doc["families"]["v1_reader"]["n"] == rec["n_reader"]
    # two recorded cases with the same text and receipts are one vector, so the set holds no more than the recording
    assert 0 < doc["families"]["v1_binding"]["n"] <= rec["n_binding"]
    assert doc["families"]["v1_binding"]["n"] == doc["families"]["binding"]["n"]


@pytest.mark.parametrize("path", SETS, ids=lambda p: p.name)
def test_the_product_need_is_a_pinned_vector(path):
    """The vector the product kit was waiting for: a made-up `$99.9k` is one unbound row, in every set."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    by_text = {v["text"]: v for v in doc["families"]["binding"]["vectors"]}
    v = by_text["BTC is at $99.9k on Binance spot, up from yesterday"]
    assert v["expect"]["counts"] == {"VERIFIED": 0, "ABSTAIN": 1, "UNGROUNDED": 0}
    assert v["expect"]["rows"] == [{"token": "99.9k", "status": "ABSTAIN", "receipt_ref": None}]
    v1 = {v["text"]: v for v in doc["families"]["v1_binding"]["vectors"]}["BTC is at $99.9k on Binance spot, up from yesterday"]
    assert v1["expect"]["rows"] == [] and v1["expect"]["verdict"] == "OATH-HELD"      # what HEAD read: nothing


# the set that pins the shipped default; vectors_v2_default.json pinned it before the review fixes
SHIPPED_SET = NUMERALS / "vectors_v2_default_after_review.json"


def test_the_default_set_pins_the_shipped_flags():
    """RULE_final 0.1 and the review fixes: the shipped default has its own set, and that set names every
    flag as shipped; its ids are over the whole flag block, since a flag that ships off (the accounting
    forms) changes a reading without changing the list of rules that are on."""
    import importlib
    C = importlib.import_module("styxx.certify")    # not the package's `certify` function
    doc = json.loads(SHIPPED_SET.read_text(encoding="utf-8"))
    assert doc["reader"]["flags"] == {k: bool(getattr(C, k)) for k in C._V15_RULE_FLAGS}
    assert doc["reader"]["rules"] == sorted(k for k in C._V15_RULE_FLAGS if getattr(C, k))
    assert doc["id_rule"] == G.ID_RULE_FLAGS


def test_a_set_written_before_a_flag_replays_it_at_the_flag_it_was_split_from():
    """V15_ACCOUNTING_SIGN was split out of V15_SIGNED_CURRENCY by the review fix (R2), as
    V15_MAGNITUDE_BARE was split out of V15_MAGNITUDE by RULE_final 0.1. A committed set that predates the
    split replays the new flag at its parent's value: what the parent covered the day the set was written."""
    for name in ("vectors_v2.json", "vectors_v2_default.json"):
        doc = json.loads((NUMERALS / name).read_text(encoding="utf-8"))
        assert "V15_ACCOUNTING_SIGN" not in (doc["reader"].get("flags") or {}), name
        got = G.set_flags_of(doc)
        assert got["V15_ACCOUNTING_SIGN"] is True and got["V15_SIGNED_CURRENCY"] is True, name
    named = json.loads((NUMERALS / "vectors_v2_default.json").read_text(encoding="utf-8"))
    assert G.set_flags_of(named)["V15_MAGNITUDE_BARE"] is False          # a flag a set names is read as named
    assert G.set_flags_of(json.loads(SHIPPED_SET.read_text(encoding="utf-8")))["V15_ACCOUNTING_SIGN"] is False


def _magnitude_row(v):
    rows = [r for r in v["expect"]["rows"] if any(c.isalpha() for c in r["token"])]
    assert len(rows) == 1, v["expect"]["rows"]
    return rows[0]


def test_the_reviews_cases_are_pinned_in_the_set_after_the_review():
    doc = json.loads(SHIPPED_SET.read_text(encoding="utf-8"))
    by_name = {v["name"]: v for v in doc["families"]["binding"]["vectors"]}
    # R1: the timestamp leaves the review found bind no longer
    for n in ("FB01_mcap_vs_okx_ts", "FB02_mcap_vs_ws_event_time", "FB03_volume_vs_seconds_ts",
              "FB04_volume_vs_updated", "FB05_mcap_vs_kline_open_time", "F1_created", "F1_last_updated"):
        assert _magnitude_row(by_name[n])["status"] == "ABSTAIN", n
    # R1, what stays: a sibling inside the window of a whole response still binds; a narrow receipt does not
    for n, leaf in (("FB06_liquidations_vs_trade_count", "btc.count"), ("FB07_price_vs_base_volume", "btc.volume"),
                    ("FB08_wrong_price_vs_low", "btc.lowPrice"), ("FB09_wrong_price_vs_high", "btc.highPrice"),
                    ("FB10_wrong_price_vs_open", "btc.openPrice")):
        r = _magnitude_row(by_name[n])
        assert (r["status"], r["receipt_ref"]) == ("VERIFIED", f"receipt[0]:{leaf}"), n
    for n in ("FB06_narrow", "FB07_narrow", "FB08_narrow"):
        assert _magnitude_row(by_name[n])["status"] != "VERIFIED", n
    # R2: a conversion or a dash before a price is not a loss
    for n, tok in (("FM01_t8_conversion", "33,706"), ("FM02_conversion", "2,431"), ("FM03_paren_price", "67.4k"),
                   ("FM04_bold_dash", "67,412.50"), ("FM05_ticker_dash", "67,412.50"), ("FM06_nested_bullet", "67.4k"),
                   ("FM09_emoji_dash", "67.4k"), ("FM10_table_dash", "67.4k")):
        mine = [r for r in by_name[n]["expect"]["rows"] if r["token"] == tok]
        assert [r["status"] for r in mine] == ["VERIFIED"], (n, by_name[n]["expect"]["rows"])
    # the product need, unchanged
    v = by_name["k_suffix_made_up"]
    assert v["expect"]["rows"] == [{"token": "99.9k", "status": "ABSTAIN", "receipt_ref": None}]


def test_the_first_set_is_read_under_the_configuration_it_was_written_in():
    """vectors_v2.json names no flags block; it replays from its rules, with V15_MAGNITUDE then covering
    the bare arm (the flag split out on 2026-10-25), and its ids carry no rules."""
    doc = json.loads((NUMERALS / "vectors_v2.json").read_text(encoding="utf-8"))
    assert "flags" not in doc["reader"] and doc["id_rule"] == G.ID_RULE_V1
    got = G.set_flags_of(doc)
    assert got["V15_MAGNITUDE"] is True and got["V15_MAGNITUDE_BARE"] is True and got["V15_FORM_NOISE"] is True
    assert got["V15_SENTENCE_FINAL"] is False and got["V15_MAGNITUDE_RANGES"] is False
