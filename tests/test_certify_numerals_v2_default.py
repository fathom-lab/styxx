# -*- coding: utf-8 -*-
"""certify's numeral reader v2 under the SHIPPED default (RULE_final 0.1 of the certify_suffix design).

The lead's scope (2026-10-25): every flag that moves any committed certificate, capsule or charon line
ships OFF, so the tree reads exactly as today; the product need, a made-up `$99.9k` is marked, is met.
Measured by the census (papers/closed-model-frontier/oath_v15_numeral_reader_census_result.json): the
bare magnitude arm and the form-noise rule each move committed certificates and turn charon lines SKEW,
which fails the stranger's ferry-log step, so both ship off; the currency-led magnitude, the word
magnitude after a currency symbol, the signs, the separators and the non-ASCII glue move nothing and ship
on. These tests pin that default: the product need is met, a bare magnitude and every form read exactly
as the reader at `879f6be9` read them, the certificate's own text says which rules were on, and two flags
give the lead's explicit list back. The clauses themselves are tests/test_certify_numerals_v2.py.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import pytest

import styxx.certify as C
from styxx.certify import certify_doc, extract_numbers, uncovered_spans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "conformance" / "numerals"))
import texts as T  # noqa: E402

ON = {"V15_MAGNITUDE", "V15_MAGNITUDE_WORDS", "V15_SIGNED_CURRENCY", "V15_THIN_GROUPS", "V15_DECIMAL_COMMA",
      "V15_GLUED_NONASCII"}
OFF = {"V15_MAGNITUDE_BARE", "V15_FORM_NOISE", "V15_SENTENCE_FINAL", "V15_MAGNITUDE_RANGES"}
BUILDER = "BTC $99.9k (+2.3%) | ETH 3.4K | vol 1.2B | 14:00 UTC"


@contextlib.contextmanager
def flags(**kw):
    old = {k: getattr(C, k) for k in kw}
    for k, v in kw.items():
        setattr(C, k, v)
    C._v15_reset()
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(C, k, v)
        C._v15_reset()


def _strip(d):
    return {k: v for k, v in d.items() if k != "context"}


def read(text):
    return {"numbers": [_strip(n) for n in extract_numbers(text)],
            "uncovered": [_strip(u) for u in uncovered_spans(text)]}


def nums(text):
    return [(n["token"], n["value"]) for n in extract_numbers(text)]


def cert(tmp_path, text, receipt=None, name="d"):
    doc = tmp_path / f"{name}.md"
    doc.write_text(text + "\n", encoding="utf-8")
    recs = receipt if isinstance(receipt, list) else [receipt if receipt is not None else {}]
    rps = []
    for i, r in enumerate(recs):
        rp = tmp_path / (f"{name}_r.json" if i == 0 else f"{name}_r{i}.json")
        rp.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        rps.append(rp)
    return certify_doc(doc, rps)


def rows(c):
    return [(r["token"], r["status"], r.get("receipt_ref")) for r in c["ledger"]]


def test_the_shipped_default_is_the_scope_rule():
    assert C.V15_READER_V2 is True
    assert set(C._V15_RULE_FLAGS) == ON | OFF
    for f in ON:
        assert getattr(C, f) is True, f
    for f in OFF:
        assert getattr(C, f) is False, f


def test_the_product_need_is_met_by_the_shipped_default(tmp_path):
    rec = {"btc": {"price": "67412.50000000"}}
    made_up = cert(tmp_path, "BTC is at $99.9k on Binance spot, up from yesterday", rec)
    assert made_up["counts"] == {"VERIFIED": 0, "ABSTAIN": 1, "UNGROUNDED": 0}
    assert rows(made_up) == [("99.9k", "ABSTAIN", None)] and made_up["verdict"] == "OATH-HELD"
    assert rows(cert(tmp_path, "BTC is at $67.4k on Binance spot", rec)) == [("67.4k", "VERIFIED", "d_r.json:btc.price")]
    assert rows(cert(tmp_path, "BTC is at $99.9k.", rec)) == [("99.9k", "ABSTAIN", None)]
    failed = cert(tmp_path, "the mean rate is $99.9k", {"btc": {"price": 67412.5}})
    assert rows(failed) == [("99.9k", "UNGROUNDED", None)] and failed["verdict"] == "OATH-FAILED"
    with flags(V15_READER_V2=False):                     # what the reader at 879f6be9 read: nothing
        assert rows(cert(tmp_path, "BTC is at $99.9k on Binance spot, up from yesterday", rec)) == []


@pytest.mark.parametrize("text,token,value", [
    ("market cap $1.2B", "1.2B", 1.2e9), ("€5m raised", "5m", 5e6), ("US$1.2B", "1.2B", 1.2e9),
    ("$1.2 billion deal", "1.2 billion", 1.2e9), ("BTC:$67.4k", "67.4k", 67400.0), ("$5mm fund", "5mm", 5e6),
])
def test_a_currency_led_magnitude_is_read(text, token, value):
    assert nums(text) == [(token, value)]
    assert extract_numbers(text)[0]["magnitude"]["currency_led"] is True


@pytest.mark.parametrize("text", [
    "ETH 2.4K", "24h volume 3.4M", "a 7B model with recall 0.9", "trained on 10k samples", "gpt2 has 124M params",
    "1.2 billion users", "5 million rows", "BTC range 60-70k today", "BTC range 60k-70k today", "Llama-3.1-8B",
    "$ 5m with a space", "5m | 3M company", "4K video and a 401k plan", "ETH 3.4K...", "67k-ish", "12.5k followers",
])
def test_a_bare_magnitude_reads_as_the_reader_at_879f6be9(text):
    """RULE_final 0.1: V15_MAGNITUDE_BARE is off, so a magnitude no currency symbol leads is read exactly as
    before v2: a glued suffix reads nothing and is not counted, a word magnitude reads its mantissa."""
    shipped = read(text)
    with flags(V15_READER_V2=False):
        assert shipped == read(text), text


@pytest.mark.parametrize("text", [
    "14:00 UTC", "at 09:30 ET", "1/ a thread opens", "\U0001F9F5 2/5", "closing | 3/7", "version 2.5 shipped",
    "Oct 24 at noon", "October 24, 2026", "24 Oct 2026", "10/24/2026", "sold Mar 10,000 contracts", "score 21:18",
])
def test_every_form_reads_as_the_reader_at_879f6be9(text):
    """RULE_final 0.1: V15_FORM_NOISE is off; clock times, thread markers, word-led versions and calendar
    dates read as they did before v2."""
    shipped = read(text)
    with flags(V15_READER_V2=False):
        assert shipped == read(text), text


def test_the_builder_line_differs_from_head_by_the_currency_led_price_alone(tmp_path):
    recs = [T.NARROW3, T.ETH]
    live = rows(cert(tmp_path, BUILDER, recs, "live"))
    with flags(V15_READER_V2=False):
        head = rows(cert(tmp_path, BUILDER, recs, "head"))

    def norm(rs, tag):
        return [(t, s, (r or "").replace(tag + "_", "")) for t, s, r in rs]
    assert norm(live, "live") == [("99.9k", "ABSTAIN", "")] + norm(head, "head")
    assert [t for t, _s, _r in head] == ["+2.3", "14", "00"]


def test_two_flags_give_the_leads_explicit_list_back(tmp_path):
    with flags(V15_MAGNITUDE_BARE=True, V15_FORM_NOISE=True):
        c = cert(tmp_path, BUILDER, [T.NARROW3, T.ETH])
        assert [(t, s) for t, s, _ in rows(c)] == [("99.9k", "ABSTAIN"), ("+2.3", "VERIFIED"), ("3.4K", "VERIFIED"),
                                                   ("1.2B", "VERIFIED")]
        assert [(u["token"], u["reason"]) for u in c["uncovered_excluded_by_rule"]] == [("14:00 UTC", "clock-time")]
        assert "V15_MAGNITUDE_BARE" in c["numeral_reader"]["rules"] and "V15_FORM_NOISE" in c["numeral_reader"]["rules"]


def test_the_certificate_says_which_rules_read_it(tmp_path):
    c = cert(tmp_path, "BTC $67.4k | a 7B model | 14:00 UTC", {"btc": {"price": 67412.5}})
    nr = c["numeral_reader"]
    assert nr["rules"] == sorted(ON)
    assert nr["magnitudes"] == {"rows": 1, "verified": 1, "abstain": 0, "ungrounded": 0, "currency_led": 1}
    assert nr["excluded_forms"] == {"clock-time": 0, "calendar-date": 0, "thread-marker": 0, "version-string": 0}
    pol = c["uncovered_policy"]
    assert pol["schema"] == "styxx-oath/uncovered-band/v2"
    # the policy text names only what this reader read: no bare magnitude, no form exclusion
    assert "V15_MAGNITUDE_BARE is off" in pol["note"] and "V15_FORM_NOISE is off" in pol["note"]
    for r in ("clock-time", "calendar-date", "thread-marker", "version-string"):
        assert r not in pol["excluded_reasons_note"], r
    assert "does NOT mean fully covered" in pol["scope_limit"]
    with flags(V15_MAGNITUDE_BARE=True, V15_FORM_NOISE=True):
        pol = cert(tmp_path, "BTC $67.4k", {}, "e")["uncovered_policy"]
        assert "V15_MAGNITUDE_BARE is off" not in pol["note"] and "V15_FORM_NOISE is off" not in pol["note"]
        for r in ("clock-time", "calendar-date", "thread-marker", "version-string"):
            assert r in pol["excluded_reasons_note"], r


def test_the_cli_header_and_exit_code_under_the_shipped_default(tmp_path, capsys):
    doc = tmp_path / "d.md"
    doc.write_text("BTC is at $99.9k on Binance spot\n", encoding="utf-8")
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps({"btc": {"price": 67412.5}}), encoding="utf-8")
    code = C.main([str(doc), str(rec), "--out", str(tmp_path / "d.certificate.json")])
    header = capsys.readouterr().out.splitlines()[0]
    assert header.startswith("OATH-HELD") and "abstained=1" in header and code == 0


def _order_free(t: str) -> bool:
    s = t.lstrip()
    return "\n" not in t and bool(s) and not (s[0].isdigit() or s[0] in "-*+>#|.(" or s.startswith("\U0001F9F5"))


@pytest.mark.parametrize("prefix", ["15m | ", "3.4M | ", "a 7B model | "])
def test_a_tokens_reading_does_not_change_when_a_bare_magnitude_stands_earlier_on_its_line(prefix):
    """A-15 for the bare arm's re-match: what stands earlier on a line never changes how a later token is
    read under the shipped default, and a bare magnitude in front adds nothing."""
    checked = 0
    for t in T.all_texts():
        if not _order_free(t):
            continue
        base = [(n["token"], n["value"], n.get("magnitude")) for n in extract_numbers(t)]
        assert [(n["token"], n["value"], n.get("magnitude")) for n in extract_numbers(prefix + t)] == base, t
        base_u = [(u["token"], u["reason"], u["counted"]) for u in uncovered_spans(t)]
        assert [(u["token"], u["reason"], u["counted"]) for u in uncovered_spans(prefix + t)] == base_u, t
        checked += 1
    assert checked > 200
