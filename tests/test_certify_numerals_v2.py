# -*- coding: utf-8 -*-
"""certify's numeral reader v2: suffixed magnitudes, form noise, signs and separators, by a written rule.

One test per clause of the rule (`RULE_final.md` of the 2026-10-25 certify_suffix design, after its red
team), written before the code and watched failing against the reader at `879f6be9` (every test here
raised AttributeError: the flags did not exist). Every test here runs under `CLAUSES`, the configuration
the clauses were written against: every rule on except the sentence-final rule and the magnitude-range
pair. The SHIPPED default is narrower (RULE_final 0.1: a flag that moves a committed certificate, capsule
or charon line ships off, so bare magnitudes and form noise are off too) and has its own suite,
`tests/test_certify_numerals_v2_default.py`.

The product need these tests confirm: a made-up `$99.9k` against a receipt that holds 67412.50 is one
ledger row, ABSTAIN, unbound; `$67.4k` against the same receipt is VERIFIED. `_NUM` is not widened: the
v2 reader is a second path that reads `_NUM`'s shapes and the band's `_WIDE_NUM` and never copies them.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import pytest

import importlib

# importlib, not `import styxx.certify as C`: the package attribute `styxx.certify` is the FUNCTION
# styxx/__init__ exports (from styxx.provenance), and which one an import binds depends on the order
# the suite imported things in (the earlier session's band test failed in a full run on exactly that)
C = importlib.import_module("styxx.certify")
certify_doc, extract_numbers, uncovered_spans = C.certify_doc, C.extract_numbers, C.uncovered_spans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "conformance" / "numerals"))
import texts as T  # noqa: E402

THIN, NNBSP, NBSP, MINUS, THREAD = " ", " ", " ", "−", "\U0001F9F5"


_MISSING = object()

# the configuration every clause test below runs under: each rule on except the two that read the
# sentence period and the magnitude pair, which have their own tests inside `flags(...)`
CLAUSES = {"V15_READER_V2": True, "V15_MAGNITUDE": True, "V15_MAGNITUDE_BARE": True, "V15_MAGNITUDE_WORDS": True,
           "V15_FORM_NOISE": True, "V15_SIGNED_CURRENCY": True, "V15_THIN_GROUPS": True, "V15_DECIMAL_COMMA": True,
           "V15_GLUED_NONASCII": True, "V15_SENTENCE_FINAL": False, "V15_MAGNITUDE_RANGES": False}


@contextlib.contextmanager
def flags(**kw):
    """Set reader flags for one test and put them back; the regex cache is cleared both ways. A flag the
    installed module does not define is set for the test and removed after it."""
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


@pytest.fixture(autouse=True)
def _clauses():
    with flags(**CLAUSES):
        yield


def nums(text):
    return [(n["token"], n["value"]) for n in extract_numbers(text)]


def unc(text):
    return [(u["token"], u["reason"], u["counted"]) for u in uncovered_spans(text)]


def excluded(text):
    return [(u["token"], u["reason"]) for u in uncovered_spans(text) if not u["counted"]]


def counted(text):
    return [(u["token"], u["reason"]) for u in uncovered_spans(text) if u["counted"]]


def cert(tmp_path, text, receipt=None, name="d"):
    doc = tmp_path / f"{name}.md"
    doc.write_text(text + "\n", encoding="utf-8")
    recs = receipt if isinstance(receipt, list) else [receipt if receipt is not None else {}]
    rps = []
    for i, r in enumerate(recs):
        rp = tmp_path / (f"r.json" if i == 0 else f"r{i}.json")
        rp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        rps.append(rp)
    return certify_doc(doc, rps)


def rows(c):
    return [(r["token"], r["status"], r.get("receipt_ref")) for r in c["ledger"]]


# --------------------------------------------------------------------------- the flags, as shipped

def test_the_clause_configuration_is_every_rule_but_the_period_and_the_pair():
    """The shipped default is pinned in tests/test_certify_numerals_v2_default.py; here, what the clauses read."""
    assert set(C._V15_RULE_FLAGS) == set(CLAUSES) - {"V15_READER_V2"}
    for k, v in CLAUSES.items():
        assert getattr(C, k) is v, k


def test_num_itself_is_not_widened():
    """The band's premise (tests/test_uncovered_band.py) still holds: the v2 reader is a second path."""
    assert [m.group(0) for m in C._NUM.finditer("precision of 0.55.")] == []
    assert [m.group(0) for m in C._NUM.finditer("BTC $67.4k")] == []


# --------------------------------------------------------------------------- 1.2 the suffix table

@pytest.mark.parametrize("text,token,value", [
    ("BTC $67.4k", "67.4k", 67400.0), ("ETH 2.4K", "2.4K", 2400.0), ("24h volume 3.4M", "3.4M", 3.4e6),
    ("market cap $1.2B", "1.2B", 1.2e9), ("$2.5T market", "2.5T", 2.5e12), ("$1.2bn deal", "1.2bn", 1.2e9),
    ("5Bn", "5Bn", 5e9), ("5BN", "5BN", 5e9), ("5mn", "5mn", 5e6), ("5Mn", "5Mn", 5e6),
    ("$5mm fund", "5mm", 5e6), ("$5MM", "5MM", 5e6), ("$5m", "5m", 5e6), ("€5m", "5m", 5e6),
    ("A$5m", "5m", 5e6), ("US$1.2B", "1.2B", 1.2e9), ("$5b", "5b", 5e9), ("$5t", "5t", 5e12),
    ("₹5m raised", "5m", 5e6), ("₿0.5k", "0.5k", 500.0), ("1,234k", "1,234k", 1234000.0),
    (".5M", ".5M", 5e5), ("$.5M", ".5M", 5e5), ("-5k", "-5k", -5000.0), ("0.50M", "0.50M", 5e5),
    ("+12k%", "+12k", 12000.0), ("4K video", "4K", 4000.0), ("401k plan", "401k", 401000.0),
    ("3M company", "3M", 3e6), ("10K run", "10K", 10000.0), ("a 7B model", "7B", 7e9),
    ("$1K token", "1K", 1000.0), ("BTC $68k.", "68k", 68000.0), ("BTC $99.9k.", "99.9k", 99900.0),
    ("67.4k;", "67.4k", 67400.0), ("67.4k/", "67.4k", 67400.0), ("$67.4K's", "67.4K", 67400.0),
    ("**$67.4k**.", "67.4k", 67400.0), ("BTC:$67.4k", "67.4k", 67400.0),
])
def test_a_glued_suffix_scales_the_mantissa(text, token, value):
    assert nums(text) == [(token, value)]


@pytest.mark.parametrize("text", [
    "fees 5m", "5m candles", "15m chart", "5mm", "5MM", "5kg", "4MB", "4Mbps", "5TB", "5MW", "3Mt", "2Bq",
    "Llama-3.1-8B", "Qwen2.5-7B", "Mixtral 8x7B", "Mistral-7B", "$67.4k5", "$67.4kk", "$67.4k_", "$67.4k1",
    "67.4e3", "6.74e4", "5m€", "$ 5m", "CHF 5m", "1INCH token", "C98 token", "7B-parameter model",
    "7B-Instruct", "mid-60k", "3-year high", "sub-67k", "low-70k", "sixty-seven thousand",
])
def test_what_is_still_not_read_reads_nothing(text):
    """Letter-glued units, model names, lower-case m/b/t without a currency, hyphen words: silent, as today."""
    assert nums(text) == []


@pytest.mark.parametrize("text,expected", [
    ("$5 M", [("5", 5.0)]), ("5 M", [("5", 5.0)]), ("1.2 bn", [("1.2", 1.2)]), ("5 k", [("5", 5.0)]),
    ("5 mn", [("5", 5.0)]), ("5 millions", [("5", 5.0)]), ("1.5 Cr", [("1.5", 1.5)]),
])
def test_a_space_before_a_letter_suffix_loses_the_suffix(text, expected):
    assert nums(text) == expected


@pytest.mark.parametrize("text,token,value", [
    ("1.2 billion parameters", "1.2 billion", 1.2e9), ("2 trillion tokens", "2 trillion", 2e12),
    ("1 Million Dollars", "1 Million", 1e6), ("67.4 thousand", "67.4 thousand", 67400.0),
    ("a 1.2 billion-dollar deal", "1.2 billion", 1.2e9), ("It cost 1.2 billion.", "1.2 billion", 1.2e9),
])
def test_a_magnitude_word_after_a_space_scales(text, token, value):
    assert nums(text) == [(token, value)]


def test_the_magnitude_row_carries_its_reading():
    n, = extract_numbers("BTC $67.4k today")
    assert n["token"] == "67.4k" and n["value"] == 67400.0 and n["decimals"] == 1
    assert n["col"] == "BTC $67.4k today".index("67.4k")
    assert n["magnitude"] == {"suffix": "k", "multiplier": 1000, "mantissa": "67.4", "coarse": False,
                              "currency_led": True}
    w, = extract_numbers("1.2 billion")
    assert w["magnitude"]["suffix"] == "billion" and w["magnitude"]["multiplier"] == 10 ** 9
    assert "magnitude" not in extract_numbers("BTC 67,412.50")[0]


@pytest.mark.parametrize("text,coarse", [("$1B", True), (".5M", True), ("2.0B", True), ("0.5k", True),
                                         ("1.2B", False), ("67k", False), ("124M", False)])
def test_coarse_marks_a_one_significant_digit_mantissa(text, coarse):
    assert extract_numbers(text)[0]["magnitude"]["coarse"] is coarse


@pytest.mark.parametrize("text,led", [("BTC $67k", True), ("€5m", True), ("-$5k", True), ("$-5k", True),
                                      ("BTC 67k", False), ("a 7B model", False), ("$ 67k", False)])
def test_currency_led_is_the_symbol_directly_before_the_number(text, led):
    assert extract_numbers(text)[0]["magnitude"]["currency_led"] is led


# --------------------------------------------------------------------------- A-15 order independence

@pytest.mark.parametrize("text,expected", [
    ("$5m", [("5m", 5e6)]), ("5m | $5m", [("5m", 5e6)]), ("$5m | 5m", [("5m", 5e6)]),
    ("15m chart: BTC $67m vol", [("67m", 6.7e7)]), ("5m candles | mcap $1.2b", [("1.2b", 1.2e9)]),
    ("in 5m BTC did $5m volume", [("5m", 5e6)]), ("fees 5m | $5m", [("5m", 5e6)]),
    ("15m | ETH $410b mcap | SOL $70b", [("410b", 4.1e11), ("70b", 7e10)]), ("5b | $5b", [("5b", 5e9)]),
    ("5m | 3.4M | $5m", [("3.4M", 3.4e6), ("5m", 5e6)]), ("5m | $5m | 3.4M", [("5m", 5e6), ("3.4M", 3.4e6)]),
    ("5m | $5k", [("5k", 5000.0)]), ("5m | $67,412", [("67,412", 67412.0)]), ("€5m after 5m", [("5m", 5e6)]),
    ("1h | ETH $410b", [("410b", 4.1e11)]),
])
def test_a_bare_m_earlier_on_the_line_does_not_silence_a_currency_led_one(text, expected):
    assert nums(text) == expected


def _order_free(t: str) -> bool:
    s = t.lstrip()
    return "\n" not in t and bool(s) and not (s[0].isdigit() or s[0] in "-*+>#|.(" or s.startswith(THREAD))


def test_a_tokens_reading_does_not_change_when_a_token_is_added_earlier_on_its_line():
    prefix = "15m | "
    checked = 0
    for t in T.all_texts():
        if not _order_free(t):
            continue
        base = [(n["token"], n["value"], n.get("magnitude")) for n in extract_numbers(t)]
        with_prefix = [(n["token"], n["value"], n.get("magnitude")) for n in extract_numbers(prefix + t)]
        assert with_prefix == base, t
        base_u = [(u["token"], u["reason"], u["counted"]) for u in uncovered_spans(t)]
        pref_u = [(u["token"], u["reason"], u["counted"]) for u in uncovered_spans(prefix + t)]
        assert pref_u == base_u, t
        checked += 1
    assert checked > 200


# --------------------------------------------------------------------------- 1.4 binding a magnitude

@pytest.mark.parametrize("text,leaf,status", [
    ("BTC is at $67.4k", 67350, "VERIFIED"), ("BTC is at $67.4k", 67450, "VERIFIED"),
    ("BTC is at $67.4k", 67450.01, "ABSTAIN"), ("BTC is at $67.4k", 67412.5, "VERIFIED"),
    ("BTC is at $67.4k", "67412.50000000", "VERIFIED"),
    ("BTC $67k", 66500, "VERIFIED"), ("BTC $67k", 67500, "VERIFIED"), ("BTC $67k", 67500.01, "ABSTAIN"),
    ("BTC $67k", 67499, "VERIFIED"), ("BTC $67.41k", 67415, "VERIFIED"), ("BTC $67.41k", 67415.01, "ABSTAIN"),
])
def test_the_window_is_half_a_unit_of_the_last_shown_mantissa_digit(tmp_path, text, leaf, status):
    assert rows(cert(tmp_path, text, {"btc": {"price": leaf}})) == [
        (text.split("$")[1], status, "r.json:btc.price" if status == "VERIFIED" else None)]


@pytest.mark.parametrize("text,leaf,status", [
    ("market cap $1B", 510000000, "ABSTAIN"), ("market cap $1B", 1490000000, "ABSTAIN"),
    ("market cap $1B", 949000000, "ABSTAIN"), ("market cap $1B", 950000000, "VERIFIED"),
    ("market cap $1B", 1040000000, "VERIFIED"), ("market cap $1B", 1050000000, "VERIFIED"),
    ("market cap $2T", 1500000000000, "ABSTAIN"), ("market cap $2T", 2050000000000, "VERIFIED"),
    ("market cap .5M", 450000, "ABSTAIN"), ("market cap .5M", 480000, "VERIFIED"),
    ("market cap 9B", 8500000000, "ABSTAIN"), ("market cap 9B", 8600000000, "VERIFIED"),
    ("market cap $1.2B", 1234567890, "VERIFIED"), ("market cap $1.2B", 1250000000, "VERIFIED"),
    ("market cap $1.2B", 1260000000, "ABSTAIN"),
])
def test_the_window_is_never_looser_than_five_percent(tmp_path, text, leaf, status):
    """A-2: `$1B` no longer swears to $510M. The receipt path shares a stem with the line (`market_cap`)
    so the integer filter is not what decides these."""
    c = cert(tmp_path, text, {"sol": {"market_cap": leaf}})
    assert [r[1] for r in rows(c)] == [status]


def test_percent_scaling_never_applies_to_a_magnitude(tmp_path):
    c = cert(tmp_path, "rate 12.0k% today", {"r": {"rate": 120}})
    assert rows(c) == [("12.0k", "ABSTAIN", None)]


def test_a_zero_decimal_magnitude_keeps_the_integer_count_binding_filter(tmp_path):
    """A-1(a). `$67k` is a count-shaped token: it binds only a leaf whose path the line names."""
    named = cert(tmp_path, "BTC is at $67k on Binance spot", {"btc": {"price": "67412.50"}})
    assert rows(named) == [("67k", "VERIFIED", "r.json:btc.price")]
    assert named["ledger"][0]["epistemics"]["path_checked"] is True
    unnamed = cert(tmp_path, "Bitcoin is at $67k on Binance spot", {"btc": {"price": "67412.50"}})
    assert rows(unnamed) == [("67k", "ABSTAIN", None)]
    # a float mantissa is not filtered, as floats are not
    assert rows(cert(tmp_path, "Bitcoin is at $67.4k", {"x": {"y": 67412.5}})) == [("67.4k", "VERIFIED", "r.json:x.y")]


@pytest.mark.parametrize("seg,excluded", [
    ("openTime", True), ("closeTime", True), ("firstId", True), ("lastId", True), ("tradeIds", True),
    ("timestamp", True), ("epoch_ms", True), ("updated_at", True), ("at", True), ("id", True), ("ID", True),
    ("_id", True), ("date", True), ("last_updated", False), ("bids", False), ("grids", False), ("count", False),
    ("lastPrice", False), ("market_cap", False), ("params", False), ("width", False),
])
def test_the_leaf_class_exclusion_names_times_and_ids_and_nothing_else(seg, excluded):
    assert C._v15_leaf_excluded(f"btc.{seg}") is excluded
    assert C._v15_leaf_excluded(f"rows[3].{seg}[0]") is excluded


def test_a_magnitude_never_binds_a_timestamp_or_an_id_of_a_whole_response(tmp_path):
    """A-1(b): `$1.7T` against a Binance 24h row landed on openTime (ms since the epoch) in the design's
    prototype; now the hit is recorded as excluded and the row is ABSTAIN."""
    c = cert(tmp_path, "BTC mcap $1.7T", {"btc": T.BINANCE})
    assert rows(c) == [("1.7T", "ABSTAIN", None)]
    assert set(c["ledger"][0]["epistemics"]["path_class_excluded"]) == {"r.json:btc.openTime", "r.json:btc.closeTime"}
    c = cert(tmp_path, "24h vol $4.4B", {"btc": T.BINANCE})
    assert rows(c) == [("4.4B", "ABSTAIN", None)]
    assert set(c["ledger"][0]["epistemics"]["path_class_excluded"]) == {"r.json:btc.firstId", "r.json:btc.lastId"}
    # the true values of the same row still bind
    c = cert(tmp_path, "BTC $67.4k | 24h vol $1.2B | 2.4M trades", {"btc": T.BINANCE})
    assert rows(c) == [("67.4k", "VERIFIED", "r.json:btc.lastPrice"), ("1.2B", "VERIFIED", "r.json:btc.quoteVolume"),
                       ("2.4M", "VERIFIED", "r.json:btc.count")]
    assert "path_class_excluded" not in c["ledger"][0]["epistemics"]
    assert rows(cert(tmp_path, "BTC bids 4.4B", {"btc": {"bids": 4401234567}})) == [("4.4B", "VERIFIED", "r.json:btc.bids")]
    assert rows(cert(tmp_path, "BTC $1.7T", {"btc": {"updated_at": 1729814400000}})) == [("1.7T", "ABSTAIN", None)]


# --------------------------------------------------------------------------- 1.5 obligation

def test_a_bare_magnitude_is_never_obligated(tmp_path):
    c = cert(tmp_path, "a 7B model with recall 0.9", {"m": {"recall": 0.9}})
    assert rows(c) == [("7B", "ABSTAIN", None), ("0.9", "VERIFIED", "r.json:m.recall")]
    assert c["ledger"][0]["epistemics"] == {"branch": "silent", "obligated": False, "obligation_source": None}
    assert c["verdict"] == "OATH-HELD"
    # range-sanity does not fire, the precision rule does not fire, a trigger word does not fire
    assert rows(cert(tmp_path, "accuracy 12k", {"run": {"accuracy": 0.9}})) == [("12k", "ABSTAIN", None)]
    assert rows(cert(tmp_path, "trained on 12.3456789k samples", {"run": {"n": 1}})) == [("12.3456789k", "ABSTAIN", None)]
    assert rows(cert(tmp_path, "the mean rate is 99.9k", {"btc": {"price": 67412.5}})) == [("99.9k", "ABSTAIN", None)]


def test_a_currency_led_magnitude_is_obligated_like_an_unsuffixed_numeral(tmp_path):
    """A-8: a writer who suffixes a made-up number can still fail the oath."""
    c = cert(tmp_path, "the mean rate is $99.9k", {"btc": {"price": 67412.5}})
    assert rows(c) == [("99.9k", "UNGROUNDED", None)]
    assert c["ledger"][0]["epistemics"]["obligation_source"] == "vocabulary"
    assert c["verdict"] == "OATH-FAILED"
    assert rows(cert(tmp_path, "the mean rate is $67.4k", {"btc": {"price": 67412.5}})) == [("67.4k", "VERIFIED", "r.json:btc.price")]
    c = cert(tmp_path, "accuracy $12k", {"run": {"accuracy": 0.9}})
    assert rows(c) == [("12k", "UNGROUNDED", None)]      # `accuracy` obligated it and nothing grounds it
    assert c["ledger"][0]["epistemics"]["obligation_source"] == "vocabulary"
    assert rows(cert(tmp_path, "the threshold is $5k", {})) == [("5k", "ABSTAIN", "spec-or-historical")]


def test_the_product_need_a_made_up_suffixed_price_is_marked(tmp_path):
    rec = {"btc": {"price": "67412.50000000"}}
    made_up = cert(tmp_path, "BTC is at $99.9k on Binance spot, up from yesterday", rec)
    assert made_up["counts"] == {"VERIFIED": 0, "ABSTAIN": 1, "UNGROUNDED": 0}
    assert rows(made_up) == [("99.9k", "ABSTAIN", None)]
    assert made_up["uncovered"] == 0 and made_up["verdict"] == "OATH-HELD"
    true = cert(tmp_path, "BTC is at $67.4k on Binance spot, up from yesterday", rec)
    assert rows(true) == [("67.4k", "VERIFIED", "r.json:btc.price")]
    # at the end of a sentence the magnitude is still read: the suffix letter ends the token
    assert rows(cert(tmp_path, "BTC is at $99.9k.", rec)) == [("99.9k", "ABSTAIN", None)]
    assert rows(cert(tmp_path, "BTC is at $67.4k.", rec)) == [("67.4k", "VERIFIED", "r.json:btc.price")]
    # the whole builder line with narrow receipts: the made-up price is the one unbound row
    line = "BTC $99.9k (+2.3%) | ETH 3.4K | vol 1.2B | 14:00 UTC"
    c = cert(tmp_path, line, [T.NARROW3, T.ETH])
    assert [(t, s) for t, s, _ in rows(c)] == [("99.9k", "ABSTAIN"), ("+2.3", "VERIFIED"), ("3.4K", "VERIFIED"),
                                               ("1.2B", "VERIFIED")]
    assert excluded(line) == [("14:00 UTC", "clock-time")]


# --------------------------------------------------------------------------- 1.6 ranges

def test_a_bare_lower_half_before_a_magnitude_is_notation_never_verified(tmp_path):
    rec = {"btc": {"rsi": 60, "low": 60100, "high": 69900}}
    assert rows(cert(tmp_path, "BTC range 60-70k today", rec)) == [("60", "ABSTAIN", "v05-notation")]
    assert rows(cert(tmp_path, "BTC range 60k-70k today", rec)) == [("60k", "ABSTAIN", "v05-notation")]
    assert nums("2–3B params") == [("2", 2.0)]
    assert rows(cert(tmp_path, "2–3B params", {})) == [("2", "ABSTAIN", "v05-notation")]
    assert rows(cert(tmp_path, "range $1.2-1.5B", {})) == [("1.2", "ABSTAIN", "v05-notation")]
    assert rows(cert(tmp_path, "fee $60-70m", {})) == [("60", "ABSTAIN", "v05-notation")]
    assert rows(cert(tmp_path, "took 60-70m", {}))[0][2] is None     # m is minutes without a currency
    # two currency-led halves are two magnitudes, both read
    assert nums("BTC range $66k-$68k today") == [("66k", 66000.0), ("68k", 68000.0)]
    assert nums("$67.4k–$68k") == [("67.4k", 67400.0), ("68k", 68000.0)]
    assert [r[1] for r in rows(cert(tmp_path, "BTC range $60k-$70k today", rec))] == ["VERIFIED", "VERIFIED"]


def test_the_range_pair_is_read_only_under_its_own_flag():
    assert nums("66k-68k") == [("66k", 66000.0)]
    with flags(V15_MAGNITUDE_RANGES=True):
        assert nums("60-70k") == [("60", 60000.0), ("70k", 70000.0)]
        assert nums("$60-70k") == [("60", 60000.0), ("70k", 70000.0)]
        assert nums("66k-68k") == [("66k", 66000.0), ("68k", 68000.0)]
        assert extract_numbers("60-70k")[0]["magnitude"] == {"suffix": "k", "multiplier": 1000, "mantissa": "60",
                                                            "coarse": False, "currency_led": False, "range": "lower"}
        assert nums("Llama-3.2-3B") == []
        assert nums("2–3B params") == [("2", 2e9), ("3B", 3e9)]


def test_the_range_pair_binds_both_halves_under_its_flag(tmp_path):
    rec = {"btc": {"rsi": 60, "low": 60000, "high": 70000}}
    with flags(V15_MAGNITUDE_RANGES=True):
        assert rows(cert(tmp_path, "BTC range 60-70k today", rec)) == [("60", "VERIFIED", "r.json:btc.low"),
                                                                        ("70k", "VERIFIED", "r.json:btc.high")]


# --------------------------------------------------------------------------- 1.7 drops

@pytest.mark.parametrize("text,expected", [
    ("$99.9k-ish", [("99.9k", 99900.0)]), ("a $1.2B-plus raise", [("1.2B", 1.2e9)]), ("sub-$67k", [("67k", 67000.0)]),
    ("67k-ish", [("67k", 67000.0)]), ("$99.9-ish", [("99.9", 99.9)]), ("$99,900-ish", [("99,900", 99900.0)]),
    ("BTC $67.4k-plus", [("67.4k", 67400.0)]), ("about 70k-odd", [("70k", 70000.0)]),
    ("mid-60k", []), ("3-year high", []), ("7B-parameter model", []), ("7B-Instruct", []), ("1-syc", []),
])
def test_currency_and_the_hyphen_words_are_not_formula_notation(text, expected):
    assert nums(text) == expected
    assert counted(text) == []


# --------------------------------------------------------------------------- 1.8 signs

@pytest.mark.parametrize("text,token,value", [
    ("PnL -$1,234.56 today", "-$1,234.56", -1234.56), ("loss of " + MINUS + "$500 on the hedge", "-$500", -500.0),
    ("$-5k", "-5k", -5000.0), ("-$5k", "-$5k", -5000.0), ("-$.5k", "-$.5k", -500.0), ("+$5k", "5k", 5000.0),
    ("PnL -1.2k", "-1.2k", -1200.0), ("PnL ($1.2k) today", "($1.2k)", -1200.0), ("($1,234.56)", "($1,234.56)", -1234.56),
    ("PnL - $1.2k today", "- $1.2k", -1200.0), ("loss -1.2 million", "-1.2 million", -1.2e6),
])
def test_a_minus_before_the_currency_and_the_accounting_forms_are_the_sign(text, token, value):
    n, = extract_numbers(text)
    assert (n["token"], n["value"]) == (token, value)
    assert n["col"] == text.replace(MINUS, "-").index(token)


def test_the_accounting_forms_name_their_sign_form():
    assert extract_numbers("PnL ($1.2k) today")[0]["sign_form"] == "parenthesis"
    assert extract_numbers("PnL - $1.2k today")[0]["sign_form"] == "spaced"
    assert "sign_form" not in extract_numbers("PnL -$1.2k today")[0]


@pytest.mark.parametrize("text", ["- $5k revenue", "- $5k", "  - $1.2k fee", "* $5k", "+ $5k"])
def test_a_markdown_bullet_is_structure_not_a_sign(text):
    n, = extract_numbers(text)
    assert n["value"] > 0 and "sign_form" not in n


def test_a_loss_no_longer_swears_to_a_gain(tmp_path):
    assert rows(cert(tmp_path, "Closed with PnL -$1,234.56 on the trade", {"trade": {"pnl": "1234.56"}})) == [
        ("-$1,234.56", "ABSTAIN", None)]
    assert rows(cert(tmp_path, "Closed with PnL -$1,234.56 on the trade", {"trade": {"pnl": "-1234.56"}})) == [
        ("-$1,234.56", "VERIFIED", "r.json:trade.pnl")]
    assert [r[1] for r in rows(cert(tmp_path, "PnL ($1.2k) today", {"trade": {"pnl": "1234.56"}}))] == ["ABSTAIN"]
    assert [r[1] for r in rows(cert(tmp_path, "PnL ($1.2k) today", {"trade": {"pnl": "-1234.56"}}))] == ["VERIFIED"]
    assert [r[1] for r in rows(cert(tmp_path, "PnL - $1.2k today", {"trade": {"pnl": "1234.56"}}))] == ["ABSTAIN"]
    assert [r[1] for r in rows(cert(tmp_path, "PnL - $1.2k today", {"trade": {"pnl": "-1234.56"}}))] == ["VERIFIED"]
    assert [r[1] for r in rows(cert(tmp_path, "- $5k revenue", {"q": {"revenue": 5000}}))] == ["VERIFIED"]


# --------------------------------------------------------------------------- 1.9 separators and locale forms

def test_thin_and_narrow_no_break_spaces_join_three_digit_groups():
    assert nums("Price 67" + THIN + "412.50 USDT") == [("67" + THIN + "412.50", 67412.5)]
    assert nums("Price 67" + NNBSP + "412.50 USDT") == [("67" + NNBSP + "412.50", 67412.5)]
    assert nums("Price 67 412.50 USDT") == [("67", 67.0), ("412.50", 412.5)]
    assert nums("67" + NBSP + "412.50") == [("67", 67.0), ("412.50", 412.5)]
    assert nums("1 234 567") == [("1", 1.0), ("234", 234.0), ("567", 567.0)]


@pytest.mark.parametrize("text,item", [
    ("Prix 84,17 €", ("84,17", "decimal-comma")), ("€ 84,17", ("84,17", "decimal-comma")),
    ("84,17€", ("84,17", "decimal-comma")), ("$84,17", ("84,17", "decimal-comma")),
    ("Le total est de 1.234,5 unités", ("1.234,5", "decimal-comma")), ("1,5k Follower", ("1,5k", "decimal-comma")),
    ("ETH 3,4K", ("3,4K", "decimal-comma")), ("0,5k", ("0,5k", "decimal-comma")),
    ("67" + THIN + "412,50", ("67" + THIN + "412,50", "decimal-comma")),
    ("67" + NNBSP + "412,50", ("67" + NNBSP + "412,50", "decimal-comma")),
    ("Preis 67.412,50 €", ("67.412,50", "decimal-comma")),
    ("1.234.567", ("1.234.567", "dotted-groups")), ("1.234.567,89", ("1.234.567,89", "dotted-groups")),
    ("Marktkapitalisierung 1.234.567 €", ("1.234.567", "dotted-groups")),
    ("Preis 67.412 €", ("67.412", "dotted-groups")), ("€ 67.412", ("67.412", "dotted-groups")),
])
def test_a_decimal_comma_or_a_dotted_group_is_counted_never_valued(text, item):
    assert nums(text) == []
    assert counted(text) == [item]


def test_the_locale_forms_that_cannot_be_told_apart_stay_as_today():
    assert len(nums("5,00")) == 2 and counted("5,00") == []
    assert nums("$0.001") == [("0.001", 0.001)]            # a dotted group beside the euro only
    assert nums("1,234,56") == [("1,234", 1234.0), ("56", 56.0)]
    assert nums("price 67.412.50") == [] and counted("price 67.412.50") == []   # the version scrub, as today


def test_a_counted_locale_form_travels_in_the_verdict(tmp_path):
    c = cert(tmp_path, "Prix 84,17 €", {"order": {"total": 84.17}})
    assert c["counts"] == {"VERIFIED": 0, "ABSTAIN": 0, "UNGROUNDED": 0}
    assert c["uncovered"] == 1 and c["verdict"] == "OATH-HELD, 1 uncovered"
    assert c["uncovered_items"][0]["reason"] == "decimal-comma"


# --------------------------------------------------------------------------- 1.10 form noise

@pytest.mark.parametrize("text,numbers,forms", [
    ("14:00", [], [("14:00", "clock-time")]),
    ("Market update 14:00 UTC: BTC holds above 67,000", ["67,000"], [("14:00 UTC", "clock-time")]),
    ("at 9:30 am", [], [("9:30 am", "clock-time")]), ("at 2:15pm", [], [("2:15pm", "clock-time")]),
    ("meeting 14:00:05", [], [("14:00:05", "clock-time")]), ("14:00:05.123", [], [("14:00:05.123", "clock-time")]),
    ("2026-10-24T14:00:05Z", [], [("T14:00:05Z", "clock-time")]), ("00:30", [], [("00:30", "clock-time")]),
    ("BTC 67.4k | 14:00UTC", ["67.4k"], [("14:00UTC", "clock-time")]), ("BTC 67.4k | 14:00Z", ["67.4k"], [("14:00Z", "clock-time")]),
    ("BTC 67.4k at 14:00 CET", ["67.4k"], [("14:00 CET", "clock-time")]),
    ("at 14:00. BTC", [], [("14:00", "clock-time")]), ("14:00-15:00", [], [("14:00-15:00", "clock-time")]),
    ("score 21:18", [], [("21:18", "clock-time")]), ("odds 10:11", [], [("10:11", "clock-time")]),
    ("1:14:00 elapsed", [], [("1:14:00", "clock-time")]),
    ("9:30", ["9", "30"], []), ("ratio 1:10", ["1", "10"], []), ("odds 3:1", ["3", "1"], []), ("24:00", ["24", "00"], []),
    ("12:60", ["12", "60"], []), ("0:30", ["0", "30"], []), ("verse 3:16", ["3", "16"], []),
    ("BTC 67.4k at 14h00", ["67.4k"], []),
])
def test_clock_times_are_named_exclusions(text, numbers, forms):
    assert [n[0] for n in nums(text)] == numbers
    assert excluded(text) == forms
    assert counted(text) == []


@pytest.mark.parametrize("text,numbers,forms", [
    ("1/ thread on why ETH funding flipped negative", [], [("1/", "thread-marker")]),
    ("1/n of many", [], [("1/n", "thread-marker")]),
    (THREAD + " 1/7 why funding flipped", [], [("1/7", "thread-marker")]),
    (THREAD + " 3/5 trades won", [], [("3/5", "thread-marker")]),
    ("- 2/", [], [("2/", "thread-marker")]),
    (T.BASE + " | 1/5", ["67.4k", "+2.3", "3.4K", "1.2B"], [("14:00 UTC", "clock-time"), ("1/5", "thread-marker")]),
    (T.BASE + " (1/5)", ["67.4k", "+2.3", "3.4K", "1.2B"], [("14:00 UTC", "clock-time"), ("1/5", "thread-marker")]),
    (T.BASE + " [1/5]", ["67.4k", "+2.3", "3.4K", "1.2B"], [("14:00 UTC", "clock-time"), ("1/5", "thread-marker")]),
    (T.BASE + " — 1/5", ["67.4k", "+2.3", "3.4K", "1.2B"], [("14:00 UTC", "clock-time"), ("1/5", "thread-marker")]),
    ("BTC $67.4k · 2/7", ["67.4k"], [("2/7", "thread-marker")]),
    ("BTC $67.4k | 1/5.", ["67.4k"], [("1/5", "thread-marker")]),
    # stays read: line-initial, word-preceded, table cell, out of range, emoji after
    ("2/7 funding flipped", ["2", "7"], []), ("9/12 held", ["9", "12"], []), ("1/5 " + "BTC $67.4k", ["1", "5", "67.4k"], []),
    ("fired on 15/15", ["15", "15"], []), ("| 30/30 |", ["30", "30"], []), ("BTC $67.4k | 1/51", ["67.4k", "1", "51"], []),
    ("BTC $67.4k | 1 of 5", ["67.4k", "1", "5"], []), ("Part 1/5: BTC $67.4k", ["1", "5", "67.4k"], []),
    ("Fear & Greed 71/100", ["71", "100"], []), ("3/5", ["3", "5"], []),
])
def test_thread_markers_are_named_exclusions(text, numbers, forms):
    assert [n[0] for n in nums(text)] == numbers
    assert excluded(text) == forms


@pytest.mark.parametrize("text,numbers,forms", [
    ("version 2.5 shipped", [], [("2.5", "version-string")]), ("ver. 2.5", [], [("2.5", "version-string")]),
    ("ver. 99", [], [("99", "version-string")]), ("version 99.9 shipped", [], [("99.9", "version-string")]),
    ("Version 7.50.0", [], []), ("v2.5 shipped", [], []), ("styxx 7.50.0 shipped", [], []),
    ("claude opus 4.8 and gemini 3.1 pro preview", ["4.8", "3.1"], []),
])
def test_version_strings_after_the_word_are_named_exclusions(text, numbers, forms):
    assert [n[0] for n in nums(text)] == numbers
    assert excluded(text) == forms


@pytest.mark.parametrize("text,numbers,forms", [
    ("on Oct 24 the price", [], [("Oct 24", "calendar-date")]), ("on 24 Oct 2026 the price", [], [("24 Oct 2026", "calendar-date")]),
    ("on October 24, 2026 the price", [], [("October 24, 2026", "calendar-date")]),
    ("on 10/24/2026 the price", [], [("10/24/2026", "calendar-date")]), ("24/10/2026", [], [("24/10/2026", "calendar-date")]),
    ("2026/10/24", [], [("2026/10/24", "calendar-date")]), ("Oct 24, 2026", [], [("Oct 24, 2026", "calendar-date")]),
    ("Oct. 24", [], [("Oct. 24", "calendar-date")]), ("October 24th, 2026", [], [("October 24th, 2026", "calendar-date")]),
    ("24 October 2026", [], [("24 October 2026", "calendar-date")]), ("May 5", [], [("May 5", "calendar-date")]),
    ("May 5.", [], [("May 5", "calendar-date")]), ("May 5 at noon", [], [("May 5", "calendar-date")]),
    ("5 May", [], [("5 May", "calendar-date")]), ("5 May, 2026", [], [("5 May, 2026", "calendar-date")]),
    ("May 5th", [], [("May 5th", "calendar-date")]), ("on Oct 24, the price was $67.4k", ["67.4k"], [("Oct 24", "calendar-date")]),
    # A-6: a date never splits a thousands number; A-16: a day followed by a word is a count
    ("sold Mar 10,000 contracts", ["10,000"], []), ("Jun 2,431 ETH", ["2,431"], []), ("expiry Dec 1,000 contracts", ["1,000"], []),
    ("we did May 5 trades today", ["5"], []), ("bought 10 Dec puts", ["10"], []), ("Mar 10 BTC bought", ["10"], []),
    ("bought 5 Jun 70k calls", ["5", "70k"], []), ("Sep 18 cut 25 bps", ["18", "25"], []),
    ("on 10/24 the price", ["10", "24"], []), ("10/24 close", ["10", "24"], []), ("Oct 32", ["32"], []),
    ("24.10.2026", [], []), ("May 2026 outlook", [], []),     # a year `_NUM` reaches is dropped, not listed (as today)
])
def test_calendar_dates_are_named_exclusions_that_never_hide_a_count(text, numbers, forms):
    assert [n[0] for n in nums(text)] == numbers
    assert excluded(text) == forms
    assert counted(text) == []


def test_a_paren_list_marker_stays_read_and_a_period_marker_stays_the_bands():
    assert nums("1) BTC up") == [("1", 1.0)]
    assert nums("2. ETH down") == [] and excluded("2. ETH down") == [("2", "ordered-list-marker")]


def test_a_form_claim_is_refused_where_it_would_end_inside_a_number():
    # the regexes already refuse these; the backstop is exercised through the public surface
    for text in ("sold Mar 10,000 contracts", "Jun 2,431 ETH", "at 14:00.5 odd", "Mar 10.5 units"):
        for u in uncovered_spans(text):
            assert u["reason"] not in ("calendar-date", "clock-time"), (text, u)


def test_an_exclusion_is_listed_in_the_certificate_and_counted_by_form(tmp_path):
    c = cert(tmp_path, "Market update 14:00 UTC on Oct 24: version 2.5 live | 1/5", {})
    assert c["counts"] == {"VERIFIED": 0, "ABSTAIN": 0, "UNGROUNDED": 0}
    assert c["uncovered"] == 0 and c["verdict"] == "OATH-HELD"
    assert [(u["token"], u["reason"]) for u in c["uncovered_excluded_by_rule"]] == [
        ("14:00 UTC", "clock-time"), ("Oct 24", "calendar-date"), ("2.5", "version-string"), ("1/5", "thread-marker")]
    assert c["numeral_reader"]["excluded_forms"] == {"clock-time": 1, "calendar-date": 1, "thread-marker": 1,
                                                     "version-string": 1}


# --------------------------------------------------------------------------- 1.11 non-ASCII glue

@pytest.mark.parametrize("text,item", [
    ("$99.9к", ("99.9", "glued-nonascii")), ("3.4М", ("3.4", "glued-nonascii")), ("2.4К", ("2.4", "glued-nonascii")),
    ("67.4ｋ", ("67.4", "glued-nonascii")), ("67.4ĸ", ("67.4", "glued-nonascii")), ("1.2Т", ("1.2", "glued-nonascii")),
    ("67.4ᵏ", ("67.4", "glued-nonascii")), ("$99.9ḱ", ("99.9", "glued-nonascii")),
    ("$99.9k²", ("99.9k", "glued-nonascii")), ("$67.4k⁰", ("67.4k", "glued-nonascii")), ("10²", ("10", "glued-nonascii")),
    ("BTC $99.9к today", ("99.9", "glued-nonascii")),
])
def test_a_look_alike_suffix_is_counted_not_silent(text, item):
    assert nums(text) == []
    assert counted(text) == [item]


@pytest.mark.parametrize("text", ["r·e^{i·2π} t", "read very false (~-3.4σ): a", "5μ", "2α"])
def test_a_greek_letter_is_letter_glue_not_a_look_alike(text):
    """`2π`, `-3.4σ`: mathematical notation, silent as `10x` is; the census found the two in the corpus."""
    assert nums(text) == [] and counted(text) == []


def test_digits_of_any_script_are_read_as_their_value(tmp_path):
    assert nums("BTC $６７.４k") == [("６７.４k", 67400.0)]
    assert nums("BTC $٦٧.٤k") == [("٦٧.٤k", 67400.0)]
    assert rows(cert(tmp_path, "BTC $６７.４k", {"btc": {"price": 67412.5}})) == [
        ("６７.４k", "VERIFIED", "r.json:btc.price")]
    assert nums("$99.9​k") == [("99.9", 99.9)]          # a zero-width space is not glue
    assert nums("$67.4k‏") == [("67.4k", 67400.0)]


# --------------------------------------------------------------------------- 1.12 sentence-final, behind its flag

def test_sentence_final_numerals_are_the_bands_as_today():
    assert nums("precision of 0.55.") == []
    assert unc("precision of 0.55.") == [("0.55", "trailing-period", True)]
    assert nums("all 23,247.") == [("23", 23.0)]
    assert unc("all 23,247.") == [("23,247", "partial-extraction", True)]
    assert nums("Order 10293 totals $84.17.") == [("10293", 10293.0)]


def test_the_sentence_final_flag_reads_what_the_period_hid(tmp_path):
    with flags(V15_SENTENCE_FINAL=True):
        assert nums("precision of 0.55.") == [("0.55", 0.55)] and unc("precision of 0.55.") == []
        assert nums("all 23,247.") == [("23,247", 23247.0)] and unc("all 23,247.") == []
        assert nums("Order 10293 totals $84.17.") == [("10293", 10293.0), ("84.17", 84.17)]
        assert nums("## 1. A section") == [] and excluded("## 1. A section") == [("1", "heading-ordinal")]
        assert nums("1. Item one") == [] and excluded("1. Item one") == [("1", "ordered-list-marker")]
        c = cert(tmp_path, "Order 10293 totals $84.17.", {"order": {"order_number": 10293, "total_price": "84.17"}})
        assert [r[1] for r in rows(c)] == ["VERIFIED", "VERIFIED"] and c["verdict"] == "OATH-HELD"
        c = cert(tmp_path, "The gate reached a precision of 0.65.", {"precision": 0.55})
        assert rows(c) == [("0.65", "UNGROUNDED", None)] and c["verdict"] == "OATH-FAILED"
        # C2: an ellipsis is a truncation the writer marked, not a sentence end
        c = cert(tmp_path, "(rate 0.0666...)", {"misfit_flag_rate": 0.06666666666666667})
        assert rows(c) == [("0.0666", "ABSTAIN", "v15-ellipsis-truncated")]
        assert c["ledger"][0]["epistemics"]["branch"] == "notation"
        c = cert(tmp_path, "rate 0.0666… ok", {"misfit_flag_rate": 0.06666666666666667})
        assert rows(c) == [("0.0666", "ABSTAIN", "v15-ellipsis-truncated")]


def test_an_ellipsis_after_a_magnitude_is_notation_under_the_default(tmp_path):
    assert rows(cert(tmp_path, "vol 3.4M... up", {"eth": {"vol": 3412000}})) == [("3.4M", "ABSTAIN", "v15-ellipsis-truncated")]
    assert rows(cert(tmp_path, "ETH 3.4K…", {"eth": {"price": 3412}})) == [("3.4K", "ABSTAIN", "v15-ellipsis-truncated")]
    assert extract_numbers("ETH 3.4K...")[0]["ellipsis"] is True
    assert "ellipsis" not in extract_numbers("ETH 3.4K")[0]


# --------------------------------------------------------------------------- 2. the certificate

def test_the_policy_block_names_the_v2_reader(tmp_path):
    pol = cert(tmp_path, "BTC $67.4k", {})["uncovered_policy"]
    assert pol["schema"] == "styxx-oath/uncovered-band/v2"
    assert pol["counted_reasons"] == sorted(C._V15_UNCOVERED_COUNTED) == sorted(
        {"decimal-comma", "dotted-groups", "glued-nonascii", "partial-extraction", "trailing-period", "unexplained"})
    assert "does NOT mean fully covered" in pol["scope_limit"]
    assert "Unicode" in pol["scope_limit"] and "code" in pol["scope_limit"]
    for r in ("clock-time", "calendar-date", "thread-marker", "version-string"):
        assert r in pol["excluded_reasons_note"]


def test_the_certificate_names_its_reader_and_counts_its_magnitudes(tmp_path):
    c = cert(tmp_path, "BTC $67.4k | a 7B model | the mean rate is $99.9k | 14:00 UTC", {"btc": {"price": 67412.5}})
    nr = c["numeral_reader"]
    assert nr["schema"] == "styxx-oath/numerals/v2"
    assert nr["rules"] == sorted(f for f in C._V15_RULE_FLAGS if getattr(C, f))
    assert "V15_SENTENCE_FINAL" not in nr["rules"] and "V15_MAGNITUDE" in nr["rules"]
    assert nr["magnitudes"] == {"rows": 3, "verified": 1, "abstain": 1, "ungrounded": 1, "currency_led": 2}
    assert nr["excluded_forms"] == {"clock-time": 1, "calendar-date": 0, "thread-marker": 0, "version-string": 0}
    assert c["verdict"] == "OATH-FAILED"


def test_with_the_master_off_the_certificate_is_the_v1_certificate(tmp_path):
    with flags(V15_READER_V2=False):
        c = cert(tmp_path, "BTC $67.4k | 14:00 UTC | 1/5", {"btc": {"price": 67412.5}})
        assert "numeral_reader" not in c
        assert c["uncovered_policy"]["schema"] == "styxx-oath/uncovered-band/v1"
        assert rows(c) == [("14", "ABSTAIN", None), ("00", "ABSTAIN", None), ("1", "ABSTAIN", None), ("5", "ABSTAIN", None)]


def test_the_invariants_asserted_at_issuance_hold_with_the_new_rows(tmp_path):
    text = ("BTC $67.4k, all 23,247. and Prix 84,17 € and a 7B model; recall 0.91 and precision 0.55.\n"
            "| coin | price |\n|---|---|\n| BTC | $67.4k |\n| ETH | 3.4K |\n"
            "PnL ($1.2k) today, 14:00 UTC, Oct 24, 1/5")
    c = cert(tmp_path, text, {"btc": {"price": 67412.5}, "recall": 0.91})
    es = c["epistemics_summary"]
    assert sum(es["by_branch"].values()) == sum(c["counts"].values())
    rows_ = {(e["line"], e["col"], e["token"]) for e in c["ledger"]}
    spans = {(s["line"], s["col"], s["token"]) for s in c["uncovered_items"] + c["uncovered_excluded_by_rule"]}
    assert rows_ & spans == set()
    assert c["uncovered"] == 3           # 23,247 partial, 84,17 decimal-comma, 0.55 trailing-period


def test_the_cli_and_the_exit_code_do_not_move(tmp_path, capsys):
    doc = tmp_path / "d.md"
    doc.write_text("BTC is at $99.9k on Binance spot\n", encoding="utf-8")
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps({"btc": {"price": 67412.5}}), encoding="utf-8")
    code = C.main([str(doc), str(rec), "--out", str(tmp_path / "d.certificate.json")])
    header = capsys.readouterr().out.splitlines()[0]
    assert header.startswith("OATH-HELD") and "abstained=1" in header
    assert code == 0


# --------------------------------------------------------------------------- severability

def test_each_rule_flag_reverts_its_own_clause_and_nothing_else():
    with flags(V15_MAGNITUDE=False):
        assert nums("BTC $67.4k | 14:00 UTC") == [] and excluded("BTC $67.4k | 14:00 UTC") == [("14:00 UTC", "clock-time")]
        assert nums("ETH 2.4K") == []                       # the bare arm reads nothing without its rule
    with flags(V15_MAGNITUDE_BARE=False):
        # RULE_final 0.1: a bare magnitude reads as at 879f6be9; a currency-led one is still a magnitude
        assert nums("ETH 2.4K | $67.4k | 1.2 billion | $1.2 billion") == [
            ("67.4k", 67400.0), ("1.2", 1.2), ("1.2 billion", 1.2e9)]
        assert excluded("ETH 2.4K | 14:00 UTC") == [("14:00 UTC", "clock-time")]
    with flags(V15_MAGNITUDE_WORDS=False):
        assert nums("1.2 billion") == [("1.2", 1.2)] and nums("$67.4k") == [("67.4k", 67400.0)]
    with flags(V15_FORM_NOISE=False):
        assert nums("$67.4k | 14:00 UTC | Oct 24") == [("67.4k", 67400.0), ("14", 14.0), ("00", 0.0), ("24", 24.0)]
    with flags(V15_SIGNED_CURRENCY=False):
        assert nums("PnL -$1,234.56") == [("1,234.56", 1234.56)] and nums("-5k") == [("-5k", -5000.0)]
    with flags(V15_THIN_GROUPS=False):
        assert nums("67" + THIN + "412.50") == [("67", 67.0), ("412.50", 412.5)]
    with flags(V15_DECIMAL_COMMA=False):
        assert nums("Prix 84,17 €") == [("84", 84.0), ("17", 17.0)] and counted("Prix 84,17 €") == []
    with flags(V15_GLUED_NONASCII=False):
        assert nums("$99.9к") == [] and counted("$99.9к") == []


def test_with_the_master_off_the_reader_is_the_recorded_v1_reader():
    """Invariant I0 on the families recorded from HEAD before the build (vectors_v2.json, v1_reader)."""
    vec = json.loads((ROOT / "conformance" / "numerals" / "vectors_v2.json").read_text(encoding="utf-8"))
    fam = vec["families"]["v1_reader"]["vectors"]
    assert len(fam) >= 400
    with flags(V15_READER_V2=False):
        for v in fam:
            got = {"numbers": [{k: x for k, x in n.items() if k != "context"} for n in extract_numbers(v["text"])],
                   "uncovered": [{k: x for k, x in u.items() if k != "context"} for u in uncovered_spans(v["text"])]}
            assert got == v["expect"], v["text"]


# --------------------------------------------------------------------------- the capsule's layer 2 (A-13)

def test_layer_two_names_both_readers_and_the_flag_when_they_differ():
    from styxx.capsule import verify_capsule
    rep = verify_capsule(ROOT / "papers" / "closed-model-frontier" / "CORPUS_STATE_2026_08_31.capsule.html")
    assert rep["ok"] is True
    assert any("numeral_reader" in a and "V15_READER_V2" in a for a in rep["advisory"] + rep["not_checked"])
