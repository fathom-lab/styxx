# -*- coding: utf-8 -*-
"""certify's numeral reader v2, after its two reviews (2026-10-25): what the review fixes pin.

The correctness review found two defects that stop a ship, and the claims review four sentences that the
code contradicts. Each test here was written before the fix and watched failing against 73d480fa.

- R1 (correctness) / F1 (claims): a currency-led magnitude cited against a WHOLE exchange response
  verified made-up numbers on timestamp leaves the leaf-class exclusion (A-1b) did not name (`ts`, `E`,
  `updated`, a kline tuple's open time). The exclusion now names the exchange keys and the tuple cell.
  What stays: a sibling price, count or volume inside the rounding window still binds; the certificate's
  own scope text says so, and these tests pin that it does.
- R2 (correctness): `($x)` and ` - $x` read negative on common post shapes (`0.5 BTC ($33,706)`, the
  product red team's own probe t8). The accounting forms now sit behind their own flag,
  `V15_ACCOUNTING_SIGN`, shipped OFF; `-$x`, `$-x` and U+2212 stay read as the sign.
- F2 (claims): the policy text said lakh and crore are not read; they are read, as separate groups.
- F3 (claims): the symbol must be glued to the digits; the spaced, trailing and word-currency forms read
  nothing, and the certificate now says so.

Every test runs under the SHIPPED flags unless it sets others inside `flags(...)`.
"""
from __future__ import annotations

import contextlib
import importlib
import json

import pytest

# importlib, not `import styxx.certify as C`: the package attribute `styxx.certify` is the FUNCTION
# styxx/__init__ exports (from styxx.provenance)
C = importlib.import_module("styxx.certify")
certify_doc, extract_numbers, uncovered_spans = C.certify_doc, C.extract_numbers, C.uncovered_spans

_MISSING = object()
NOW_MS, NOW_S = 1792929600000, 1792929600
# the correctness review's receipts (review_correctness/adversarial.py), values synthetic
TICK = {"btc": {"lastPrice": "67412.50", "priceChangePercent": "2.31", "highPrice": "68120.00", "lowPrice": "66010.10",
                "openPrice": "66800.00", "weightedAvgPrice": "67102.33", "volume": "18234.551",
                "quoteVolume": "1229876543.21", "count": "1834221", "openTime": NOW_MS - 86400000, "closeTime": NOW_MS,
                "firstId": 3812345678, "lastId": 3814179899}}
OKX = {"btc": {"instId": "BTC-USDT", "last": "67412.5", "high24h": "68120", "low24h": "66010.1", "vol24h": "18234.5",
               "ts": str(NOW_MS)}}
WS = {"btc": {"e": "24hrTicker", "E": NOW_MS + 123, "s": "BTCUSDT", "c": "67412.50", "O": NOW_MS - 86400000,
              "C": NOW_MS, "n": 1834221}}
KLINES = {"btc": {"klines": [[NOW_MS - 3600000, "67000.0", "68120.0", "66010.1", "67412.5"]]}}


@contextlib.contextmanager
def flags(**kw):
    old = {k: getattr(C, k, _MISSING) for k in kw}
    for k, v in kw.items():
        setattr(C, k, v)
    C._v15_reset()
    try:
        yield
    finally:
        for k, v in old.items():
            if v is _MISSING:
                delattr(C, k)
            else:
                setattr(C, k, v)
        C._v15_reset()


def nums(text):
    return [(n["token"], n["value"]) for n in extract_numbers(text)]


def cert(tmp_path, text, receipt, name="d"):
    doc = tmp_path / f"{name}.md"
    doc.write_text(text + "\n", encoding="utf-8")
    rp = tmp_path / f"{name}_r.json"
    rp.write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
    return certify_doc(doc, [rp])


def rows(c):
    return [(r["token"], r["status"], r.get("receipt_ref")) for r in c["ledger"]]


# --------------------------------------------------------------------------- R1 / F1: the leaf class

@pytest.mark.parametrize("seg", [
    # the names A-1b knew before the review
    "openTime", "closeTime", "firstId", "lastId", "tradeIds", "timestamp", "epoch_ms", "updated_at", "at", "id",
    "date",
    # the review's: OKX and Bybit short keys, the update and creation names, run-together compounds
    "ts", "mts", "unix", "unix_ms", "created", "updated", "last_updated", "modified", "createdTime", "updatedTime",
    "uTime", "cTime", "createtime", "servertime", "mtime", "eventtime", "birthdate", "date_added", "ts_ms",
    # Binance WebSocket's single-letter times and trade ids, case-sensitive
    "E", "T", "O", "C", "t", "F", "L", "f",
])
def test_the_leaf_class_names_the_exchange_time_and_id_keys(seg):
    assert C._v15_leaf_excluded(f"btc.{seg}") is True, seg
    assert C._v15_leaf_excluded(f"data[0].{seg}") is True, seg


@pytest.mark.parametrize("seg", [
    "c", "o", "h", "l", "v", "q", "n", "p", "w", "lastPrice", "highPrice", "lowPrice", "openPrice", "price", "last",
    "count", "volume", "quoteVolume", "vol24h", "market_cap", "bids", "grids", "params", "width", "trades",
])
def test_a_price_count_or_volume_name_is_not_in_the_leaf_class(seg):
    assert C._v15_leaf_excluded(f"btc.{seg}") is False, seg


def test_a_cell_of_an_array_of_arrays_is_in_the_leaf_class_and_a_named_array_is_not():
    """A kline or market_chart tuple carries no name; its cell 0 is an open time inside a `$1.8T` window."""
    assert C._v15_leaf_excluded("btc.klines[0][0]") is True
    assert C._v15_leaf_excluded("market_caps[3][1]") is True
    assert C._v15_leaf_excluded("[0][4]") is True
    assert C._v15_leaf_excluded("btc.prices[0]") is False
    assert C._v15_leaf_excluded("rows[3].price") is False


@pytest.mark.parametrize("text,receipt,leaf", [
    ("BTC market cap is $1.79T today", OKX, "btc.ts"),                                   # FB01
    ("Total crypto market cap $1.8T", WS, "btc.E"),                                      # FB02
    ("ETH 24h volume $1.79B", {"eth": {"price": "2431.20", "ts": NOW_S}}, "eth.ts"),     # FB03
    ("SOL volume hit $1.79B", {"sol": {"price": "148.20", "updated": NOW_S}}, "sol.updated"),   # FB04
    ("BTC market cap $1.8T", KLINES, "btc.klines[0][0]"),                               # FB05
    ("BTC market cap is $1.8T today", {"data": [{"created": 1792345678901}]}, "data[0].created"),
    ("BTC market cap $1.7T", {"k": {"last_updated": 1700000000000}}, "k.last_updated"),
])
def test_a_magnitude_does_not_bind_a_timestamp_the_review_found(tmp_path, text, receipt, leaf):
    c = cert(tmp_path, text, receipt)
    (tok, status, ref), = rows(c)
    assert status == "ABSTAIN" and ref is None, rows(c)
    assert c["ledger"][0]["epistemics"]["path_class_excluded"] == [f"d_r.json:{leaf}"]


def test_the_true_values_of_the_same_responses_still_bind(tmp_path):
    assert rows(cert(tmp_path, "BTC is at $67.4k on OKX", OKX)) == [("67.4k", "VERIFIED", "d_r.json:btc.last")]
    assert rows(cert(tmp_path, "BTC is at $67.4k on Binance", WS)) == [("67.4k", "VERIFIED", "d_r.json:btc.c")]
    assert rows(cert(tmp_path, "BTC 24h volume $1.23B on Binance", TICK)) == [
        ("1.23B", "VERIFIED", "d_r.json:btc.quoteVolume")]


@pytest.mark.parametrize("text,leaf", [
    ("BTC liquidations were $1.83M in 24h", "btc.count"),        # FB06: the trade count
    ("BTC is at $18.2k on Binance spot", "btc.volume"),          # FB07: the base-asset volume
    ("BTC is at $66k on Binance spot", "btc.lowPrice"),          # FB08: a wrong price, the low
    ("BTC is at $68.1k on Binance spot", "btc.highPrice"),       # FB09: the high
    ("BTC is at $66.8k on Binance spot", "btc.openPrice"),       # FB10: the open
])
def test_what_stays_a_sibling_inside_the_window_still_binds(tmp_path, text, leaf):
    """The residual, pinned so it cannot be forgotten: a magnitude's window is half a unit of its last
    shown digit, so against a WHOLE response a sibling price, count or volume inside it binds. The
    mitigation is a narrow receipt (only the fields the post cites); the certificate's scope says so."""
    (_tok, status, ref), = rows(cert(tmp_path, text, TICK))
    assert (status, ref) == ("VERIFIED", f"d_r.json:{leaf}")
    pol = cert(tmp_path, text, TICK)["uncovered_policy"]
    assert "sibling" in pol["scope_limit"] and "narrow receipt" in pol["scope_limit"]
    # the same claims against a narrow receipt: the wrong ones do not bind
    assert rows(cert(tmp_path, text, {"btc": {"lastPrice": "67412.50"}}))[0][1] == "ABSTAIN"


def test_the_policy_text_prints_the_leaf_class_and_the_window_arithmetic(tmp_path):
    scope = cert(tmp_path, "BTC $67.4k", {"btc": {"price": 67412.5}})["uncovered_policy"]["scope_limit"]
    for word in ("ts", "updated", "E T O C", "array of arrays"):
        assert word in scope, word
    assert "1.75T" in scope and "1.85T" in scope          # a 2026 epoch-ms timestamp sits in a `$1.8T` window


# --------------------------------------------------------------------------- R2: the accounting forms

def test_the_accounting_forms_ship_off_and_the_minus_forms_stay_on():
    assert C.V15_ACCOUNTING_SIGN is False
    assert C.V15_SIGNED_CURRENCY is True
    assert "V15_ACCOUNTING_SIGN" in C._V15_RULE_FLAGS


@pytest.mark.parametrize("text,receipt,token", [
    ("Wallet balance: 0.5 BTC ($33,706)", {"wallet": {"btc": "0.5", "balance_usd": "33706"}}, "33,706"),   # t8, FM01
    ("Holding 1 ETH ($2,431) in the vault", {"eth": {"price": "2431"}}, "2,431"),                       # FM02
    ("BTC ($67.4k) leads the majors", {"btc": {"price": "67412.50"}}, "67.4k"),                       # FM03
    ("**BTC** - $67,412.50", {"btc": {"price": "67412.50"}}, "67,412.50"),                            # FM04
    ("Bitcoin (BTC) - $67,412.50", {"btc": {"price": "67412.50"}}, "67,412.50"),                      # FM05
    ("- prices\n    - $67.4k BTC spot", {"btc": {"price": "67412.50"}}, "67.4k"),                     # FM06
    ("BTC \U0001F680 - $67.4k", {"btc": {"price": "67412.50"}}, "67.4k"),                             # FM09
    ("| BTC | - $67.4k |", {"btc": {"price": "67412.50"}}, "67.4k"),                                   # FM10
])
def test_a_conversion_or_a_dash_before_a_price_is_not_a_loss(tmp_path, text, receipt, token):
    c = cert(tmp_path, text, receipt)
    mine = [r for r in rows(c) if r[0] == token]
    assert len(mine) == 1 and mine[0][1] == "VERIFIED", rows(c)


@pytest.mark.parametrize("text,token,value", [
    ("PnL -$1,234.56 today", "-$1,234.56", -1234.56), ("loss of −$500 on the hedge", "-$500", -500.0),
    ("$-5k", "-5k", -5000.0), ("-$5k", "-$5k", -5000.0),
    ("PnL ($1.2k) today", "1.2k", 1200.0), ("PnL - $1.2k today", "1.2k", 1200.0),
    ("($1,234.56)", "1,234.56", 1234.56),
])
def test_the_minus_forms_are_the_sign_and_the_accounting_forms_read_as_before_v2(text, token, value):
    n, = extract_numbers(text)
    assert (n["token"], n["value"]) == (token, value)
    assert "sign_form" not in n


def test_the_accounting_flag_on_restores_the_accounting_reading():
    with flags(V15_ACCOUNTING_SIGN=True):
        assert nums("PnL ($1.2k) today") == [("($1.2k)", -1200.0)]
        assert extract_numbers("PnL ($1.2k) today")[0]["sign_form"] == "parenthesis"
        assert nums("PnL - $1.2k today") == [("- $1.2k", -1200.0)]
        assert extract_numbers("PnL - $1.2k today")[0]["sign_form"] == "spaced"
        assert nums("- $5k revenue") == [("5k", 5000.0)]          # a bullet is still structure
    with flags(V15_ACCOUNTING_SIGN=True, V15_SIGNED_CURRENCY=False):
        assert nums("PnL ($1.2k) today") == [("($1.2k)", -1200.0)]   # each flag its own clause
        assert nums("PnL -$1,234.56") == [("1,234.56", 1234.56)]


def test_the_cost_of_the_accounting_flag_off_is_stated(tmp_path):
    """Off, an accounting-format loss reads as the positive amount, as it did before v2."""
    c = cert(tmp_path, "BTC PnL ($5k) today", {"btc": {"pnl": "5000"}})
    assert rows(c) == [("5k", "VERIFIED", "d_r.json:btc.pnl")]
    note = c["uncovered_policy"]["note"]
    assert "V15_ACCOUNTING_SIGN is off" in note
    with flags(V15_ACCOUNTING_SIGN=True):
        assert rows(cert(tmp_path, "BTC PnL ($5k) today", {"btc": {"pnl": "5000"}}, "e"))[0][1] == "ABSTAIN"


def test_the_certificate_names_the_accounting_rule_only_when_it_is_on(tmp_path):
    c = cert(tmp_path, "BTC $67.4k", {"btc": {"price": 67412.5}})
    assert "V15_ACCOUNTING_SIGN" not in c["numeral_reader"]["rules"]
    assert "V15_ACCOUNTING_SIGN is off" in c["uncovered_policy"]["note"]
    with flags(V15_ACCOUNTING_SIGN=True):
        c = cert(tmp_path, "BTC $67.4k", {"btc": {"price": 67412.5}}, "e")
        assert "V15_ACCOUNTING_SIGN" in c["numeral_reader"]["rules"]
        assert "V15_ACCOUNTING_SIGN is off" not in c["uncovered_policy"]["note"]


# --------------------------------------------------------------------------- F2: lakh and crore

def test_lakh_grouping_is_read_as_separate_groups_and_the_certificate_says_so(tmp_path):
    assert nums("the mean rate is 12,34,567 today") == [("12", 12.0), ("34,567", 34567.0)]
    assert nums("1,00,000 rows") == [("1", 1.0), ("00,000", 0.0)]
    scope = cert(tmp_path, "BTC $67.4k", {})["uncovered_policy"]["scope_limit"]
    unread = scope.split("not read and not counted:")[1].split(".")[0]
    assert "lakh" not in unread
    assert "12,34,567" in scope and "lakh" in scope


# --------------------------------------------------------------------------- F3: the glued symbol

@pytest.mark.parametrize("text", [
    "BTC $ 67.4k", "€ 5m raised", "US$ 1.2B", "BTC 67.4k$", "5m€ raised", "BTC USD 67.4k", "BTC 67.4k USD",
    "BTC $ 99.9k", "BTC 99.9k USD",
])
def test_a_symbol_not_glued_to_the_digits_leads_no_magnitude(text):
    assert extract_numbers(text) == [] and [u for u in uncovered_spans(text) if u["counted"]] == []


def test_a_space_before_the_suffix_reads_the_mantissa_alone():
    n, = extract_numbers("BTC $67.4 k")
    assert (n["token"], n["value"]) == ("67.4", 67.4) and "magnitude" not in n


def test_the_certificate_names_the_glued_symbol_condition(tmp_path):
    scope = cert(tmp_path, "BTC $67.4k", {})["uncovered_policy"]["scope_limit"]
    for form in ("$ 67.4k", "67.4k USD", "USD 67.4k", "67.4k$", "$67.4 k"):
        assert form in scope, form
    assert "glued" in scope


# --------------------------------------------------------------------------- m3: the cost of dotted groups

def test_a_three_decimal_euro_price_is_a_counted_dotted_group(tmp_path):
    """RULE_final 1.9 chose this: `€1.789` (fuel) is counted, never valued. Its cost, pinned."""
    c = cert(tmp_path, "Super E10 costs €1.789 per litre", {"fuel": {"e10_eur": "1.789"}})
    assert rows(c) == []
    assert [(u["token"], u["reason"], u["counted"]) for u in c["uncovered_items"]] == [("1.789", "dotted-groups", True)]
