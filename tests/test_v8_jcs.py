"""tests for styxx.v8.jcs -- RFC 8785 (JCS) canonical bytes.

Contract: styxx/v8/INTERFACES_foundations.md (frozen 2026-09-07), section jcs.

1. The RFC 8785 section 3.2.3 example, pinned as bytes.  The RFC prints the canonical output as
   text and as a hex dump; it does not print a SHA-256 of it.  The hash pinned here is therefore
   COMPUTED from the pinned bytes -- it is a receipt for ``digest``, not a quotation.
2. ES6 number formatting cases.
3. Differential: ``rfc8785.dumps(x) == styxx.attestation.jcs(x).encode("utf-8")`` over
   hypothesis-generated JSON values, then a MUTATION CHECK: six semantic mutations are applied
   in-process to a copy of ``styxx.attestation._jcs`` (the copy is a new function object sharing
   the code, with its globals monkeypatched) and the differential generator must catch each one
   within ``max_examples=300``.  A mutant the generator cannot catch goes into ``KNOWN_MISSES``.
4. Refusals: NaN, inf, ``2**53 + 1``, bytes, set, a dict with an int key.

Also pinned, because the differential found them:
* the two raw backends disagree on object-key order when sibling keys mix a non-BMP character
  with a BMP character in U+E000..U+FFFF at the same position (code-point order vs the RFC's
  UTF-16 order); the module's fallback backend refuses such objects rather than emit them;
* ``rfc8785`` refuses ``|n| == 2**53`` while ``styxx.attestation`` emits it; the module coerces
  that boundary value to the equal float so both backends yield the same bytes.

Unicode escapes of the form backslash-u are deliberately absent from this file's source (the
tooling that writes files on this box collapses them); characters are spelled with ``chr()``,
``\\x`` or eight-digit ``\\U`` escapes instead.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
import types

import pytest
from hypothesis import HealthCheck, Phase, given, settings, strategies as st

import styxx.attestation as attestation
from styxx.attestation import jcs as styxx_jcs
from styxx.v8 import jcs as J
from styxx.v8.jcs import backend, canonical_bytes, digest, sha256_hex

# Not importorskip: the contract says a missing dependency is a failure, never a skip.
import rfc8785  # noqa: E402

BS = "\\"          # one backslash, as text
BSB = b"\\"        # one backslash, as bytes
EURO = chr(0x20AC)
LONE_HIGH = chr(0xD800)   # a lone high surrogate: not UTF-8 encodable, refused everywhere
LONE_LOW = chr(0xDFFF)    # a lone low surrogate


# --------------------------------------------------------------------------- 1. RFC example

def rfc_example():
    # RFC 8785 section 3.2.3 input.  Spelled with chr()/hex escapes because this file must not
    # contain backslash-u sequences.  The RFC's JSON text for the string decodes to: euro sign,
    # $, 0x0f, LF, A, ', B, quote, backslash (from the u005c escape), backslash (from the
    # doubled backslash), quote (from the escaped quote), slash -- two backslashes in a row.
    string = EURO + "$" + "\x0f" + "\x0a" + "A'" + "B" + '"' + BS + BS + '"' + "/"
    return {
        "numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],
        "string": string,
        "literals": [None, True, False],
    }


# The RFC's canonical output:
#   {"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],
#    "string":"<euro>$<bs>u000f<bs>nA'B<bs>"<bs><bs><bs><bs><bs>"/"}
# with the euro sign as raw UTF-8 (e2 82 ac), the two escapes as six- and two-character
# sequences, and four backslashes in the middle of the string.
RFC_EXAMPLE_BYTES = (
    b'{"literals":[null,true,false],'
    b'"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
    b'"string":"' + EURO.encode("utf-8") + b"$"
    + BSB + b"u000f" + BSB + b"n"
    + b"A'B" + BSB + b'"' + BSB * 4 + BSB + b'"' + b'/"}'
)

# Same bytes as a hex dump, so the pin is readable next to the RFC's own dump.
RFC_EXAMPLE_HEX = (
    "7b226c69746572616c73223a5b6e756c6c2c747275652c66616c73655d2c"
    "226e756d62657273223a5b3333333333333333332e333333333333332c31652b33302c342e352c302e3030322c31652d32375d2c"
    "22737472696e67223a22e282ac245c75303030665c6e4127425c225c5c5c5c5c222f227d"
)

# COMPUTED as sha256(RFC_EXAMPLE_BYTES) on 2026-09-07; not quoted from the RFC.
RFC_EXAMPLE_SHA256 = "2d5e01a318d0f0879ab568c4be289c8b1f64ef8921a53c6277d5e069978baacb"


def test_rfc8785_example_bytes():
    assert bytes.fromhex(RFC_EXAMPLE_HEX) == RFC_EXAMPLE_BYTES
    out = canonical_bytes(rfc_example())
    assert out == RFC_EXAMPLE_BYTES
    # Both raw backends produce the same bytes on the RFC's own example.
    assert rfc8785.dumps(rfc_example()) == RFC_EXAMPLE_BYTES
    assert styxx_jcs(rfc_example()).encode("utf-8") == RFC_EXAMPLE_BYTES
    # The escapes are the ones section 3.2.2.2 prescribes: 0x0f as the six-character lowercase
    # hex escape, LF as the two-character escape; the slash and the euro sign are not escaped.
    assert BSB + b"u000f" in out
    assert BSB + b"n" in out
    assert BSB + b"/" not in out
    assert BSB + b"u20ac" not in out
    # no raw control byte survives into the canonical form
    assert not any(b < 0x20 for b in out)
    assert json.loads(out) == rfc_example()


def test_rfc8785_example_digest():
    assert hashlib.sha256(RFC_EXAMPLE_BYTES).hexdigest() == RFC_EXAMPLE_SHA256
    assert digest(rfc_example()) == RFC_EXAMPLE_SHA256
    assert sha256_hex(RFC_EXAMPLE_BYTES) == RFC_EXAMPLE_SHA256
    assert not digest(rfc_example()).startswith("sha256:")


def test_sha256_hex_shape_and_types():
    h = sha256_hex(b"")
    assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert len(h) == 64 and h == h.lower()
    assert sha256_hex(bytearray(b"abc")) == sha256_hex(b"abc") == sha256_hex(memoryview(b"abc"))
    with pytest.raises(TypeError):
        sha256_hex("abc")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sha256_hex(123)  # type: ignore[arg-type]


def test_backend_name():
    assert backend() == "rfc8785"
    assert backend() in ("rfc8785", "styxx.attestation")


# --------------------------------------------------------------------------- 2. ES6 numbers

ES6_CASES = [
    (1e21, b"1e+21"),
    (1e-7, b"1e-7"),
    (0.000001, b"0.000001"),
    (-0.0, b"0"),
    (1.0, b"1"),
    (100, b"100"),
    (100.0, b"100"),
    (1.5e300, b"1.5e+300"),
    (5e-324, b"5e-324"),
    (123456789012345680000.0, b"123456789012345680000"),
    (0.1 + 0.2, b"0.30000000000000004"),
    (1e30, b"1e+30"),
    (1e-27, b"1e-27"),
    (4.50, b"4.5"),
    (2e-3, b"0.002"),
    (333333333.33333329, b"333333333.3333333"),
    (1e308, b"1e+308"),
    (2.2250738585072014e-308, b"2.2250738585072014e-308"),
    (-1.5, b"-1.5"),
    (0, b"0"),
    (-1, b"-1"),
    (2**53 - 1, b"9007199254740991"),
    (-(2**53 - 1), b"-9007199254740991"),
    (2**53, b"9007199254740992"),
    (-(2**53), b"-9007199254740992"),
    (9007199254740992.0, b"9007199254740992"),
]


@pytest.mark.parametrize("value,expected", ES6_CASES, ids=[repr(v) for v, _ in ES6_CASES])
def test_es6_number_formatting(value, expected):
    assert canonical_bytes(value) == expected
    assert canonical_bytes([value]) == b"[" + expected + b"]"
    assert styxx_jcs(float(value) if isinstance(value, float) else value).encode("utf-8") == expected


def _same_json_value(parsed, original) -> bool:
    """Structural equality that reads an integer-looking literal back as the double it came from.

    ``json.loads`` parses ``42036440897448250`` as an int; Python compares int and float exactly,
    so above 2**53 the int need not equal the double whose shortest representation it is.  The
    canonical form is lossless as a DOUBLE (ES6 Number::toString guarantees round-trip), which is
    what this compares.
    """
    if isinstance(original, bool) or original is None:
        return parsed is original
    if isinstance(original, float):
        return not isinstance(parsed, bool) and float(parsed) == original
    if isinstance(original, int):
        return isinstance(parsed, int) and not isinstance(parsed, bool) and parsed == original
    if isinstance(original, str):
        return isinstance(parsed, str) and parsed == original
    if isinstance(original, list):
        return (isinstance(parsed, list) and len(parsed) == len(original)
                and all(_same_json_value(p, o) for p, o in zip(parsed, original)))
    if isinstance(original, dict):
        return (isinstance(parsed, dict) and set(parsed) == set(original)
                and all(_same_json_value(parsed[k], original[k]) for k in original))
    raise TypeError(type(original).__name__)


def test_es6_number_formatting_is_lossless():
    for value, expected in ES6_CASES:
        back = json.loads(expected)
        assert _same_json_value(back, value), (value, expected, back)
        if isinstance(value, float):
            assert float(back) == value
        else:
            assert back == value
    # the helper itself: the boundary that motivated it, and a negative
    assert _same_json_value(123456789012345680000, 123456789012345680000.0)
    assert not _same_json_value(123456789012345680001, 123456789012345680000.0 + 1e5)
    assert not _same_json_value(1, True) and not _same_json_value(True, 1)


def test_large_int_literal_from_contract_is_refused_as_int_but_accepted_as_float():
    # The contract's "123456789012345680000" is only representable as a double; as a Python int it
    # is outside |n| <= 2**53 and both the module and rfc8785 refuse it.
    with pytest.raises(ValueError):
        canonical_bytes(123456789012345680000)
    with pytest.raises(Exception):
        rfc8785.dumps(123456789012345680000)
    assert canonical_bytes(float(123456789012345680000)) == b"123456789012345680000"


# --------------------------------------------------------------------------- 3. differential

C0 = [chr(c) for c in range(0x20)]
INTERESTING_CHARS = C0 + [
    '"', BS, "/", "\x7f", " ", "a", "0", "-",
    chr(0x2028), chr(0x2029), chr(0x20AC), chr(0xFEFF), chr(0xFFFD),
    chr(0xE000), chr(0xFFFF),
    "\U00010000", "\U0001D11E", "\U0001F600", "\U0010FFFF",
]
# Object keys avoid U+D800..U+FFFF entirely so that code-point order and UTF-16 order agree
# (the divergence is pinned separately below); values use the full UTF-8 alphabet.
KEY_SAFE_CHARS = [c for c in INTERESTING_CHARS if not (0xD800 <= ord(c) <= 0xFFFF)]

value_chars = st.one_of(st.sampled_from(INTERESTING_CHARS), st.characters(codec="utf-8"))
key_chars = st.one_of(
    st.sampled_from(KEY_SAFE_CHARS),
    st.characters(codec="utf-8", max_codepoint=0xD7FF),
    st.characters(codec="utf-8", min_codepoint=0x10000),
)
value_text = st.text(alphabet=value_chars, max_size=12)
key_text = st.text(alphabet=key_chars, max_size=8)
full_key_text = st.text(alphabet=value_chars, max_size=8)

scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53 - 1), max_value=2**53 - 1),
    st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=True),
    value_text,
)


def _json_values(keys):
    return st.recursive(
        scalars,
        lambda children: st.one_of(
            st.lists(children, max_size=6),
            st.dictionaries(keys, children, max_size=6),
        ),
        max_leaves=30,
    )


json_values = _json_values(key_text)
json_values_full_keys = _json_values(full_key_text)


def _no_lone_surrogates(x) -> bool:
    if isinstance(x, str):
        x.encode("utf-8")
        return True
    if isinstance(x, list):
        return all(_no_lone_surrogates(v) for v in x)
    if isinstance(x, dict):
        return all(_no_lone_surrogates(k) and _no_lone_surrogates(v) for k, v in x.items())
    return True


@settings(max_examples=600, deadline=None, database=None)
@given(json_values)
def test_differential_rfc8785_vs_styxx_attestation(x):
    assert _no_lone_surrogates(x)
    a = rfc8785.dumps(x)
    b = styxx_jcs(x).encode("utf-8")
    assert a == b
    assert canonical_bytes(x) == a
    assert _same_json_value(json.loads(a), x)
    assert digest(x) == hashlib.sha256(a).hexdigest()


@settings(max_examples=300, deadline=None, database=None)
@given(json_values)
def test_differential_fallback_backend_matches_rfc8785(x):
    saved = J._rfc8785
    J._rfc8785 = None
    try:
        assert backend() == "styxx.attestation"
        assert canonical_bytes(x) == rfc8785.dumps(x)
    finally:
        J._rfc8785 = saved
    assert backend() == "rfc8785"


def _codepoint_vs_utf16_orders_differ(x) -> bool:
    if isinstance(x, dict):
        keys = list(x)
        if sorted(keys) != sorted(keys, key=lambda k: k.encode("utf-16-be")):
            return True
        return any(_codepoint_vs_utf16_orders_differ(v) for v in x.values())
    if isinstance(x, list):
        return any(_codepoint_vs_utf16_orders_differ(v) for v in x)
    return False


@settings(max_examples=300, deadline=None, database=None)
@given(json_values_full_keys)
def test_differential_with_unrestricted_keys_disagrees_only_on_utf16_order(x):
    a = rfc8785.dumps(x)
    b = styxx_jcs(x).encode("utf-8")
    if a == b:
        assert canonical_bytes(x) == a
        return
    # The only admitted disagreement: key order across the UTF-16 surrogate boundary.
    assert _codepoint_vs_utf16_orders_differ(x)
    assert sorted(json.loads(a).items() if isinstance(x, dict) else [json.loads(a)]) == \
        sorted(json.loads(b).items() if isinstance(x, dict) else [json.loads(b)])


# ---- the mutation check ------------------------------------------------------------------

def _copy_of_styxx_jcs(**overrides):
    """A fresh function object with ``styxx.attestation._jcs``'s code and its own globals.

    The copy recurses into itself (``_jcs`` in the copied globals is rebound to the copy), so a
    mutation applied to the globals affects every level of the serialization.  The real module
    is never touched.
    """
    original = attestation._jcs
    g = dict(attestation.__dict__)
    g.update(overrides)
    copy = types.FunctionType(original.__code__, g, "_jcs_copy", original.__defaults__,
                              original.__closure__)
    g["_jcs"] = copy
    return copy


class _JsonShim:
    """Stands in for the ``json`` module inside the copied function; only ``dumps`` is used."""

    def __init__(self, dumps):
        self.dumps = dumps


def _mutant_drop_key_sorting():
    return _copy_of_styxx_jcs(sorted=lambda iterable, key=None: list(iterable))


def _mutant_exponent_without_plus():
    orig = attestation._es_number_to_string
    return _copy_of_styxx_jcs(_es_number_to_string=lambda f: orig(f).replace("e+", "e"))


def _mutant_escape_slash():
    return _copy_of_styxx_jcs(json=_JsonShim(lambda s, **kw: json.dumps(s, **kw).replace("/", BS + "/")))


def _mutant_negative_zero():
    orig = attestation._es_number_to_string

    def fmt(f):
        if f == 0.0 and math.copysign(1.0, f) < 0:
            return "-0"
        return orig(f)

    return _copy_of_styxx_jcs(_es_number_to_string=fmt)


def _mutant_ensure_ascii():
    return _copy_of_styxx_jcs(json=_JsonShim(lambda s, **kw: json.dumps(s, ensure_ascii=True)))


_LOWER_HEX_ESCAPE = re.compile(BS + BS + "u00([0-9a-f]{2})")


def _mutant_uppercase_hex_escapes():
    def dumps(s, **kw):
        return _LOWER_HEX_ESCAPE.sub(lambda m: BS + "u00" + m.group(1).upper(), json.dumps(s, **kw))

    return _copy_of_styxx_jcs(json=_JsonShim(dumps))


# name -> (factory, a hand-picked witness value on which the mutant must differ from the original)
MUTANTS = {
    "drop_key_sorting": (_mutant_drop_key_sorting, {"b": 1, "a": 2}),
    "exponent_without_plus": (_mutant_exponent_without_plus, 1e21),
    "escape_slash": (_mutant_escape_slash, "/"),
    "negative_zero": (_mutant_negative_zero, -0.0),
    "ensure_ascii": (_mutant_ensure_ascii, EURO),
    "uppercase_hex_escapes": (_mutant_uppercase_hex_escapes, "\x1f"),
}

# Mutants the calibrated generator could not catch within max_examples=300.  Measured, not
# assumed: test_mutation_check fails if a listed mutant is caught (then it must leave the list)
# or an unlisted one is missed (then it must join it).  An empty list means every mutant was
# caught on this box with these strategies.
KNOWN_MISSES: list[str] = []

MUTATION_BUDGET = 300


def _generator_catches(mutant, max_examples=MUTATION_BUDGET) -> bool:
    @settings(
        max_examples=max_examples,
        deadline=None,
        database=None,
        derandomize=True,
        phases=[Phase.generate],
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(json_values)
    def probe(x):
        assert rfc8785.dumps(x) == mutant(x).encode("utf-8")

    try:
        probe()
    except AssertionError:
        return True
    except BaseExceptionGroup as group:
        return any(isinstance(e, AssertionError) for e in group.exceptions)
    return False


def test_copy_of_styxx_jcs_is_a_faithful_copy():
    copy = _copy_of_styxx_jcs()
    assert copy is not attestation._jcs
    assert copy.__code__ is attestation._jcs.__code__
    for value in (rfc_example(), {"b": [1, 2.5, None], "a": {"c": EURO + "/\x1f"}}, -0.0, 1e21):
        assert copy(value) == attestation._jcs(value)
    # the unmodified copy agrees with rfc8785 on the RFC example, so a disagreement below is the
    # mutation's doing and nothing else
    assert copy(rfc_example()).encode("utf-8") == RFC_EXAMPLE_BYTES


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_live(name):
    factory, witness = MUTANTS[name]
    mutant = factory()
    assert mutant(witness) != attestation._jcs(witness), f"{name} is a no-op on its witness"
    assert mutant(witness).encode("utf-8") != rfc8785.dumps(witness)
    # and the real module is untouched
    assert attestation._jcs(witness).encode("utf-8") == rfc8785.dumps(witness)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutation_check(name):
    factory, _ = MUTANTS[name]
    caught = _generator_catches(factory())
    if name in KNOWN_MISSES:
        assert not caught, f"{name} is listed in KNOWN_MISSES but the generator caught it; remove it"
    else:
        assert caught, f"the differential generator missed {name} within {MUTATION_BUDGET} examples"


def test_known_misses_list_contents():
    assert len(MUTANTS) >= 5
    assert set(KNOWN_MISSES) <= set(MUTANTS)
    assert KNOWN_MISSES == []


# --------------------------------------------------------------------------- 4. refusals

class _Custom:
    pass


@pytest.mark.parametrize("value", [
    float("nan"), float("inf"), float("-inf"),
    2**53 + 1, -(2**53 + 1), 2**64, 10**30,
    {1: 2}, {None: 1}, {True: 1}, {(1, 2): 3},
    LONE_HIGH, "a" + LONE_LOW + "b", {LONE_HIGH: 1},
    [float("nan")], {"k": [1, {"n": float("inf")}]}, {"k": 2**60},
], ids=repr)
def test_refusals_value_error(value):
    with pytest.raises(ValueError):
        canonical_bytes(value)
    with pytest.raises(ValueError):
        digest(value)


@pytest.mark.parametrize("value", [
    b"bytes", bytearray(b"x"), {1, 2}, frozenset(), (1, 2),
    _dt.datetime(2026, 9, 7, tzinfo=_dt.timezone.utc), _dt.date(2026, 9, 7),
    _Custom(), object(), complex(1, 2), {"k": b"x"}, [{1, 2}],
], ids=lambda v: type(v).__name__)
def test_refusals_type_error(value):
    with pytest.raises(TypeError):
        canonical_bytes(value)
    with pytest.raises(TypeError):
        digest(value)


def test_refusals_are_backend_independent():
    saved = J._rfc8785
    J._rfc8785 = None
    try:
        for value in (float("nan"), float("inf"), 2**53 + 1, {1: 2}, LONE_HIGH):
            with pytest.raises(ValueError):
                canonical_bytes(value)
        for value in (b"x", {1, 2}, (1,), _Custom()):
            with pytest.raises(TypeError):
                canonical_bytes(value)
    finally:
        J._rfc8785 = saved


def test_bool_is_not_an_int_here():
    assert canonical_bytes(True) == b"true"
    assert canonical_bytes([False, 0, 1, True]) == b"[false,0,1,true]"


def test_safe_int_boundary_is_coerced_to_the_equal_float():
    # rfc8785 refuses |n| == 2**53; styxx.attestation emits it; the module returns the same bytes
    # on both backends by handing the boundary value over as the float it equals.
    with pytest.raises(Exception):
        rfc8785.dumps(2**53)
    assert styxx_jcs(2**53) == "9007199254740992"
    assert canonical_bytes(2**53) == b"9007199254740992" == canonical_bytes(float(2**53))
    assert canonical_bytes(-(2**53)) == b"-9007199254740992"
    saved = J._rfc8785
    J._rfc8785 = None
    try:
        assert canonical_bytes(2**53) == b"9007199254740992"
        assert canonical_bytes({"n": [2**53, -(2**53), 2**53 - 1]}) == \
            b'{"n":[9007199254740992,-9007199254740992,9007199254740991]}'
    finally:
        J._rfc8785 = saved


# --------------------------------------------------------------------------- the divergence

UTF16_DIVERGENT = {"\U00010000": 1, chr(0xFFFF): 2}


def test_utf16_key_order_divergence_is_pinned():
    # RFC 8785 section 3.2.3 sorts keys by UTF-16 code units: U+10000 is D800 DC00, which sorts
    # before FFFF.  styxx.attestation sorts by code point, which puts U+FFFF before U+10000.
    astral = "\U00010000".encode("utf-8")
    bmp = chr(0xFFFF).encode("utf-8")
    assert rfc8785.dumps(UTF16_DIVERGENT) == b'{"' + astral + b'":1,"' + bmp + b'":2}'
    assert styxx_jcs(UTF16_DIVERGENT).encode("utf-8") == b'{"' + bmp + b'":2,"' + astral + b'":1}'
    assert _codepoint_vs_utf16_orders_differ(UTF16_DIVERGENT)
    # The module follows the RFC on the rfc8785 backend ...
    assert canonical_bytes(UTF16_DIVERGENT) == rfc8785.dumps(UTF16_DIVERGENT)
    # ... and refuses rather than disagree on the fallback backend.
    saved = J._rfc8785
    J._rfc8785 = None
    try:
        with pytest.raises(ValueError):
            canonical_bytes(UTF16_DIVERGENT)
        with pytest.raises(ValueError):
            canonical_bytes({"outer": [UTF16_DIVERGENT]})
        # keys that do not straddle the boundary are unaffected
        assert canonical_bytes({"\U00010000": 1, "a": 2}) == rfc8785.dumps({"\U00010000": 1, "a": 2})
        assert canonical_bytes({chr(0xFFFF): 1, "a": 2}) == rfc8785.dumps({chr(0xFFFF): 1, "a": 2})
    finally:
        J._rfc8785 = saved


# --------------------------------------------------------------------------- determinism

def test_canonical_bytes_is_pure_and_order_insensitive():
    a = {"z": [1, 2.0, {"y": None}], "a": "x"}
    b = {"a": "x", "z": [1, 2.0, {"y": None}]}
    assert canonical_bytes(a) == canonical_bytes(b) == b'{"a":"x","z":[1,2,{"y":null}]}'
    assert digest(a) == digest(b)
    assert canonical_bytes(a) == canonical_bytes(a)
    assert isinstance(canonical_bytes(a), bytes)
    assert canonical_bytes({}) == b"{}"
    assert canonical_bytes([]) == b"[]"
    assert canonical_bytes("") == b'""'
    assert canonical_bytes(None) == b"null"


# =========================================================================== hostile pass
# Appended 2026-09-08 by the hostile tester.  Nothing above this line was changed.
#
# A test marked ``xfail(strict=True)`` pins a DEFECT: it fails today for the stated reason and
# becomes a hard failure (XPASS) the moment the behaviour changes, so the marker must be removed
# together with the fix.  Everything else is a pinned attack that the module survives.

import collections
import contextlib
import copy
import enum
import os
import random
import shutil
import struct
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ES6_NUMBER = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]*[1-9])?(e[+-][1-9][0-9]*)?$")

ASTRAL_10000 = chr(0x10000)
ASTRAL_1F600 = chr(0x1F600)
ASTRAL_10FFFF = chr(0x10FFFF)


@contextlib.contextmanager
def _fallback():
    saved = J._rfc8785
    J._rfc8785 = None
    try:
        assert backend() == "styxx.attestation"
        yield
    finally:
        J._rfc8785 = saved


def _both_backends(value) -> bytes:
    """canonical_bytes on the rfc8785 path and on the fallback path; asserts they agree."""
    a = canonical_bytes(value)
    with _fallback():
        b = canonical_bytes(value)
    assert a == b, (value, a, b)
    return a


# ---- strings: every C0 control, DEL, line separators, astral -------------------------------

def _expected_escape(c: int) -> bytes:
    short = {0x08: b"b", 0x09: b"t", 0x0A: b"n", 0x0C: b"f", 0x0D: b"r"}
    if c in short:
        return BSB + short[c]
    return BSB + (b"u%04x" % c)


@pytest.mark.parametrize("c", list(range(0x20)), ids=lambda c: "0x%02x" % c)
def test_hostile_every_c0_control_escape_pinned_by_hand(c):
    want = b'"' + _expected_escape(c) + b'"'
    assert _both_backends(chr(c)) == want
    assert rfc8785.dumps(chr(c)) == want
    # inside a key too, and the key sorts by its RAW code unit (pinned separately below)
    assert _both_backends({chr(c): 0}) == b"{" + want + b":0}"
    # never an uppercase hex digit, never a raw control byte
    assert want == want.lower()
    assert not any(b < 0x20 for b in want)


@pytest.mark.parametrize("ch,raw", [
    (chr(0x7F), b"\x7f"),
    (chr(0x2028), b"\xe2\x80\xa8"),
    (chr(0x2029), b"\xe2\x80\xa9"),
    (chr(0xFEFF), b"\xef\xbb\xbf"),
    (chr(0xFFFD), b"\xef\xbf\xbd"),
    (chr(0xFFFF), b"\xef\xbf\xbf"),
    (chr(0xE000), b"\xee\x80\x80"),
    (chr(0xD7FF), b"\xed\x9f\xbf"),
    (chr(0xA0), b"\xc2\xa0"),
    (chr(0xAD), b"\xc2\xad"),
    (ASTRAL_10000, b"\xf0\x90\x80\x80"),
    (ASTRAL_1F600, b"\xf0\x9f\x98\x80"),
    (ASTRAL_10FFFF, b"\xf4\x8f\xbf\xbf"),
    ("/", b"/"),
    ("'", b"'"),
    (chr(0x301), b"\xcc\x81"),
], ids=lambda v: v.hex() if isinstance(v, bytes) else "U+%04X" % ord(v))
def test_hostile_unescaped_characters_are_emitted_raw(ch, raw):
    assert _both_backends(ch) == b'"' + raw + b'"'
    assert rfc8785.dumps(ch) == b'"' + raw + b'"'


def test_hostile_surrogate_pair_spelled_as_two_code_points_is_refused():
    pair = chr(0xD83D) + chr(0xDE00)          # two lone surrogates, not one astral character
    for value in (pair, [pair], {"k": pair}, {pair: 1}):
        with pytest.raises(ValueError):
            canonical_bytes(value)
        with _fallback():
            with pytest.raises(ValueError):
                canonical_bytes(value)
    # the same pair arriving as JSON text decodes to the single astral character, which is fine
    decoded = json.loads('"' + BS + "ud83d" + BS + "ude00" + '"')
    assert decoded == ASTRAL_1F600
    assert _both_backends(decoded) == b'"\xf0\x9f\x98\x80"'


def test_hostile_no_unicode_normalization_is_applied():
    decomposed = "e" + chr(0x301)
    composed = chr(0xE9)
    assert decomposed != composed
    out = _both_backends({composed: 1, decomposed: 2})
    # U+0065 sorts before U+00E9; both keys survive
    assert out == b'{"e\xcc\x81":2,"\xc3\xa9":1}'
    assert out == rfc8785.dumps({composed: 1, decomposed: 2})


# ---- key order: raw code units, not the escaped text ---------------------------------------

def test_hostile_key_order_is_raw_code_units_not_escaped_text():
    keys = ["~", "b", "ab", "aa", "a", "]", BS, "A", '"', "!", "\n", "\x01", "\x00", "", chr(0x7F)]
    obj = {k: i for i, k in enumerate(keys)}
    # ascending raw code unit order; escaping happens after sorting
    want = (
        b'{"":13,'
        + b'"' + BSB + b'u0000":12,'
        + b'"' + BSB + b'u0001":11,'
        + b'"' + BSB + b'n":10,'
        + b'"!":9,'
        + b'"' + BSB + b'"":8,'
        + b'"A":7,'
        + b'"' + BSB + BSB + b'":6,'
        + b'"]":5,'
        + b'"a":4,"aa":3,"ab":2,"b":1,"~":0,'
        + b'"\x7f":14}'
    )
    assert _both_backends(obj) == want
    assert rfc8785.dumps(obj) == want
    # an escaped-text sort would put the LF key ("\n" starts with a backslash) after "!"
    assert want.index(b'"!":') > want.index(BSB + b'n":')


def test_hostile_dunder_and_numeric_looking_keys():
    obj = {"constructor": 2, "__proto__": 1, "9": 4, "10": 3, "": 5, "toString": 6}
    want = b'{"":5,"10":3,"9":4,"__proto__":1,"constructor":2,"toString":6}'
    assert _both_backends(obj) == want
    assert rfc8785.dumps(obj) == want


# ---- the UTF-16 divergence, attacked with realistic keys -----------------------------------

@pytest.mark.parametrize("bmp", [chr(0xFF21), chr(0xFB01), chr(0xE000), chr(0xFFFD), chr(0xFFFF)],
                         ids=lambda c: "U+%04X" % ord(c))
def test_hostile_realistic_bmp_keys_above_d800_next_to_an_emoji_key(bmp):
    # fullwidth A, the fi ligature, private use, the replacement character: all sort AFTER an
    # astral key under UTF-16 and BEFORE it under code points.
    obj = {bmp: 1, ASTRAL_1F600: 2}
    want = b'{"' + ASTRAL_1F600.encode("utf-8") + b'":2,"' + bmp.encode("utf-8") + b'":1}'
    assert canonical_bytes(obj) == want == rfc8785.dumps(obj)
    assert styxx_jcs(obj).encode("utf-8") != want
    with _fallback():
        with pytest.raises(ValueError):
            canonical_bytes(obj)


def test_hostile_fallback_guard_fires_for_siblings_only():
    cousins = {"a": {ASTRAL_10000: 1}, "b": {chr(0xFFFF): 2}}
    assert _both_backends(cousins) == rfc8785.dumps(cousins)
    siblings_in_list = [{ASTRAL_10000: 1, chr(0xFFFF): 2}]
    assert canonical_bytes(siblings_in_list) == rfc8785.dumps(siblings_in_list)
    with _fallback():
        with pytest.raises(ValueError):
            canonical_bytes(siblings_in_list)
        # a shared prefix does not change the verdict
        with pytest.raises(ValueError):
            canonical_bytes({"x" + ASTRAL_10000: 1, "x" + chr(0xFFFF): 2})
        # keys below U+D800 next to astral keys agree in both orders
        assert canonical_bytes({chr(0xD7FF): 1, ASTRAL_10000: 2}) == \
            rfc8785.dumps({chr(0xD7FF): 1, ASTRAL_10000: 2})


# ---- numbers -------------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (float(2**53 + 1), b"9007199254740992"),     # rounds to even as a double
    (float(2**53 + 2), b"9007199254740994"),
    (2.0**63, b"9223372036854776000"),
    (1e20, b"100000000000000000000"),
    (9.999999999999999e20, b"999999999999999900000"),
    (1e21, b"1e+21"),
    (1e16, b"10000000000000000"),
    (1e-6, b"0.000001"),
    (9.999999999999999e-7, b"0.000001"),  # this literal IS the double 1e-6 (repr 1e-06); ES6 renders 0.000001
    (1.000000000000001e-7, b"1.000000000000001e-7"),
    (1e-5, b"0.00001"),
    (0.00001234, b"0.00001234"),
    (1.5e-10, b"1.5e-10"),
    (1e100, b"1e+100"),
    (-5e-324, b"-5e-324"),
    (-1.7976931348623157e308, b"-1.7976931348623157e+308"),
    (4.9406564584124654e-324, b"5e-324"),
    (0.1 + 0.7, b"0.7999999999999999"),
    (1 / 3, b"0.3333333333333333"),
    (2 / 3, b"0.6666666666666666"),
    (1e15 + 0.3, b"1000000000000000.2"),
], ids=repr)
def test_hostile_number_spot_checks_pinned_by_hand(value, expected):
    assert _both_backends(value) == expected
    assert rfc8785.dumps(value) == expected
    assert _both_backends([value, {"n": value}]) == b"[" + expected + b',{"n":' + expected + b"}]"
    assert float(expected) == value


def test_hostile_negative_zero_in_every_position():
    obj = {"z": -0.0, "a": [0.0, -0.0, -0, [-0.0]], "m": {"k": -0.0}}
    assert _both_backends(obj) == b'{"a":[0,0,0,[0]],"m":{"k":0},"z":0}'
    assert rfc8785.dumps(obj) == b'{"a":[0,0,0,[0]],"m":{"k":0},"z":0}'
    assert b"-0" not in canonical_bytes(obj)


def test_hostile_int_boundary_on_both_backends():
    for n in (2**53, -(2**53)):
        assert _both_backends(n) == str(n).encode()
        assert _both_backends({"n": [n]}) == b'{"n":[' + str(n).encode() + b"]}"
    for n in (2**53 + 1, -(2**53 + 1), 2**53 + 2, 2**63, -(2**63), 2**64, 10**100):
        for value in (n, [n], {"k": n}, {"k": [{"d": n}]}):
            with pytest.raises(ValueError):
                canonical_bytes(value)
            with _fallback():
                with pytest.raises(ValueError):
                    canonical_bytes(value)


def test_hostile_random_double_bit_patterns_agree_on_both_backends_and_round_trip():
    rng = random.Random(0x8785)
    seen = 0
    while seen < 3000:
        f = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if math.isnan(f) or math.isinf(f):
            continue
        seen += 1
        out = _both_backends(f)
        assert out == rfc8785.dumps(f)
        assert ES6_NUMBER.match(out.decode("ascii")), out
        assert float(out) == f
        assert b"E" not in out and b".0," not in out and not out.endswith(b".0")


# ---- an independent oracle: node's JSON.stringify + default sort (UTF-16 code units) --------

NODE_ORACLE = """
const fs = require("fs");
function canon(v) {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(canon).join(",") + "]";
  const keys = Object.keys(v).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + canon(v[k])).join(",") + "}";
}
const input = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(input.map((v) => Buffer.from(canon(v), "utf8").toString("hex")).join("\\n") + "\\n");
"""

HOSTILE_ALPHABET = C0 + [
    '"', BS, "/", chr(0x7F), " ", "a", "Z", "0", "-", ".", chr(0x2028), chr(0x2029), chr(0x20AC),
    chr(0xFEFF), chr(0xFFFD), chr(0xE000), chr(0xFFFF), chr(0xD7FF), chr(0xFF21), chr(0xFB01),
    ASTRAL_10000, chr(0x1D11E), ASTRAL_1F600, ASTRAL_10FFFF, chr(0xA0), chr(0xAD), chr(0x301),
]
HOSTILE_SPECIAL_NUMBERS = [
    0.0, -0.0, 1e21, 1e-7, 5e-324, 1.7976931348623157e308, 2.2250738585072014e-308, 1e20, 1e16,
    0.1 + 0.2, 123456789012345680000.0, 9.999999999999999e20, 1e-6, 9.999999999999999e-7,
    2**53 - 1, -(2**53 - 1), 2**53, -(2**53), 1.0, -1.0, 100, 4.5, 333333333.33333329, 1e-27,
    1e30, 1.5e300, float(2**53 + 2), 2.0**63, 1e-5, 0.00001234,
]


def _hostile_corpus(seed: int, n: int):
    rng = random.Random(seed)

    def rstr(k=6):
        return "".join(rng.choice(HOSTILE_ALPHABET) for _ in range(rng.randint(0, k)))

    def rfloat():
        while True:
            f = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
            if not (math.isnan(f) or math.isinf(f)):
                return f

    def rscalar():
        r = rng.random()
        if r < 0.15:
            return None
        if r < 0.25:
            return rng.random() < 0.5
        if r < 0.40:
            return rng.randint(-(2**53 - 1), 2**53 - 1)
        if r < 0.50:
            return rng.choice(HOSTILE_SPECIAL_NUMBERS)
        if r < 0.75:
            return rfloat()
        return rstr()

    def rvalue(depth=0):
        r = rng.random()
        if depth >= 4 or r < 0.5:
            return rscalar()
        if r < 0.75:
            return [rvalue(depth + 1) for _ in range(rng.randint(0, 5))]
        return {rstr(4): rvalue(depth + 1) for _ in range(rng.randint(0, 5))}

    return [rvalue() for _ in range(n)]


def test_hostile_node_oracle_agrees_with_the_module_on_both_backends(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.fail("node is not on PATH; the contract says fail, not skip")
    corpus = HOSTILE_SPECIAL_NUMBERS + [
        rfc_example(), [[], {}, [{}], {"": []}], {chr(0xFF21): 1, ASTRAL_1F600: 2},
        {ASTRAL_10000: 1, chr(0xFFFF): 2}, "".join(HOSTILE_ALPHABET), {"".join(C0): [-0.0]},
        {"__proto__": 1, "constructor": 2, "": 3},
    ] + _hostile_corpus(20260908, 1500)
    script = tmp_path / "jcs_oracle.js"
    script.write_bytes(NODE_ORACLE.encode("utf-8"))
    payload = json.dumps(corpus, ensure_ascii=True, allow_nan=False).encode("utf-8")
    res = subprocess.run([node, str(script)], input=payload, capture_output=True, timeout=120)
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    lines = [line for line in res.stdout.decode("ascii").split("\n") if line]
    assert len(lines) == len(corpus)
    divergent = 0
    for value, hexline in zip(corpus, lines):
        want = bytes.fromhex(hexline)
        out = canonical_bytes(value)
        assert out == want, (value, out, want)
        assert not any(b < 0x20 for b in out)
        assert _same_json_value(json.loads(out), value)
        assert digest(value) == hashlib.sha256(want).hexdigest()
        with _fallback():
            if _codepoint_vs_utf16_orders_differ(value):
                divergent += 1
                with pytest.raises(ValueError):
                    canonical_bytes(value)
            else:
                assert canonical_bytes(value) == want
    assert divergent >= 2  # the hand-planted divergent objects were exercised


# ---- determinism across processes and hash seeds -------------------------------------------

DETERMINISM_OBJECT_SRC = (
    "{chr(0x41 + i % 26) * (i % 5 + 1) + str(i): [i, i / 7.0, {'k' + str(j): None for j in range(i % 4)}]"
    " for i in range(500)}"
)
DETERMINISM_DIGEST = "086128a0698cd5d79dccba5ea0d4dfedf614e128db9baf562923bb08d2fb7641"


def test_hostile_digest_is_identical_across_processes_and_hash_seeds():
    obj = eval(DETERMINISM_OBJECT_SRC)  # noqa: S307 - a literal expression pinned above
    assert digest(obj) == DETERMINISM_DIGEST
    with _fallback():
        assert digest(obj) == DETERMINISM_DIGEST
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from styxx.v8.jcs import digest, backend;"
        " print(backend(), digest(" + DETERMINISM_OBJECT_SRC + "))"
    )
    outs = []
    for seed in ("1", "2", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run([sys.executable, "-c", code, REPO_ROOT], capture_output=True,
                             env=env, cwd=REPO_ROOT, timeout=120)
        assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
        outs.append(res.stdout.decode("ascii").split())
    assert outs == [["rfc8785", DETERMINISM_DIGEST]] * 3


@pytest.mark.parametrize("raw,hexdigest", [
    (b"{}", "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"),
    (b"[]", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"),
    (b"null", "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b"),
    (b'""', "12ae32cb1ec02d01eda3581b127c1fee3b0dc53572ed6baf239721a03d82e126"),
    (b"0", "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9"),
])
def test_hostile_known_digests_pinned(raw, hexdigest):
    value = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == hexdigest
    assert digest(value) == hexdigest
    assert sha256_hex(raw) == hexdigest
    assert HEX64.match(hexdigest)


def test_hostile_sha256_hex_and_digest_shapes():
    assert HEX64.match(digest({"a": [1, 2.5, None, "x"]}))
    for bad in (True, None, 1.5, [b"x"], "x", object()):
        with pytest.raises(TypeError):
            sha256_hex(bad)  # type: ignore[arg-type]
    import array
    assert sha256_hex(memoryview(array.array("B", [0x61, 0x62, 0x63]))) == sha256_hex(b"abc")
    assert sha256_hex(memoryview(b"abcd").cast("B", (2, 2))) == sha256_hex(b"abcd")
    buf = bytearray(b"abc")
    h = sha256_hex(buf)
    buf[0] = 0x7A
    assert h == sha256_hex(b"abc")  # the bytes were hashed at call time, not aliased


# ---- subclassed and look-alike types -------------------------------------------------------

class _List(list):
    pass


class _Dict(dict):
    pass


class _IntE(enum.IntEnum):
    ONE = 1


class _IntF(enum.IntFlag):
    A = 1
    B = 2


class _StrE(enum.StrEnum):
    X = "x"


class _StrMixinE(str, enum.Enum):
    A = "a"


class _PlainE(enum.Enum):
    RED = "red"


class _EvilStr(str):
    def __str__(self):
        return "evil"


class _EvilInt(int):
    def __int__(self):
        return 2**60

    def __index__(self):
        return 2**60


class _NaNFloat(float):
    def __float__(self):
        return float("nan")


class _CollidingKey(str):
    def __hash__(self):
        return 12345

    def __eq__(self, other):
        return False


def test_hostile_container_subclasses_are_accepted_and_sorted():
    od = collections.OrderedDict([("b", 1), ("a", 2)])
    assert _both_backends(od) == b'{"a":2,"b":1}'
    dd = collections.defaultdict(int, a=1)
    assert _both_backends(dd) == b'{"a":1}'
    assert _both_backends(_List([1, _List([2])])) == b"[1,[2]]"
    assert _both_backends(_Dict(b=_Dict(a=1))) == b'{"b":{"a":1}}'
    for bad in (collections.UserDict(a=1), collections.UserList([1]), collections.ChainMap({"a": 1}),
                types.MappingProxyType({"a": 1}), range(3), iter([1]), (x for x in [1])):
        with pytest.raises(TypeError):
            canonical_bytes(bad)
        with _fallback():
            with pytest.raises(TypeError):
                canonical_bytes(bad)


def test_hostile_int_and_str_enums_serialize_as_their_values():
    assert _both_backends(_IntE.ONE) == b"1" == rfc8785.dumps(_IntE.ONE)
    assert _both_backends([_IntF.A | _IntF.B]) == b"[3]"
    assert _both_backends(_StrE.X) == b'"x"' == rfc8785.dumps(_StrE.X)
    assert _both_backends({_StrE.X: _IntE.ONE}) == b'{"x":1}'
    for value in (_PlainE.RED, [_PlainE.RED], {"k": _PlainE.RED}):
        with pytest.raises(TypeError):
            canonical_bytes(value)
    with pytest.raises(ValueError):
        canonical_bytes({_PlainE.RED: 1})


def test_hostile_raw_backends_agree_on_a_str_mixin_enum_member():
    # both raw backends serialize the (str, Enum) member as its value, so any other output from
    # the module is the module's own doing
    assert rfc8785.dumps(_StrMixinE.A) == b'"a"'
    assert styxx_jcs(_StrMixinE.A).encode("utf-8") == b'"a"'
    assert rfc8785.dumps({_StrMixinE.A: 1}) == b'{"a":1}'
    assert rfc8785.dumps(_EvilStr("x")) == b'"x"'
    assert styxx_jcs(_EvilStr("x")).encode("utf-8") == b'"x"'


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "DEFECT: _normalize copies str subclasses with str(obj), which runs the subclass's __str__; "
    "a (str, Enum) member becomes '\"_StrMixinE.A\"' although rfc8785 and styxx.attestation both "
    "emit '\"a\"' (rfc8785 deliberately avoids str() for exactly this case)"))
def test_hostile_str_mixin_enum_member_serializes_as_its_value():
    assert canonical_bytes(_StrMixinE.A) == b'"a"'
    assert canonical_bytes({_StrMixinE.A: 1}) == b'{"a":1}'
    assert canonical_bytes(_EvilStr("x")) == b'"x"'
    with _fallback():
        assert canonical_bytes(_StrMixinE.A) == b'"a"'


def test_hostile_numeric_subclass_hooks_cannot_smuggle_bad_values():
    for value in (_EvilInt(5), [_EvilInt(5)], _NaNFloat(1.5), {"k": _NaNFloat(1.5)}):
        with pytest.raises(ValueError):
            canonical_bytes(value)
        with _fallback():
            with pytest.raises(ValueError):
                canonical_bytes(value)


# repaired 2026-09-08: colliding keys are refused with ValueError (styxx/v8/jcs.py _normalize); xfail removed
def test_hostile_colliding_keys_are_refused_not_silently_dropped():
    obj = {"a": 1, _CollidingKey("a"): 2}
    assert len(obj) == 2
    assert rfc8785.dumps(obj) == b'{"a":1,"a":2}'
    with pytest.raises(ValueError):
        canonical_bytes(obj)


def test_hostile_bool_never_becomes_a_number_on_either_backend():
    value = [True, 1, 1.0, False, 0, 0.0, -0.0, {"t": True, "f": False}]
    assert _both_backends(value) == b'[true,1,1,false,0,0,0,{"f":false,"t":true}]'
    assert rfc8785.dumps(value) == b'[true,1,1,false,0,0,0,{"f":false,"t":true}]'
    with pytest.raises(ValueError):
        canonical_bytes({True: 1})
    with pytest.raises(ValueError):
        canonical_bytes({False: 1})


def test_hostile_tuple_is_refused_by_the_module_although_rfc8785_accepts_it():
    # contract gap: tuples are not named; the module treats them as non-JSON and rfc8785 does not
    assert rfc8785.dumps((1, 2)) == b"[1,2]"
    for value in ((1, 2), (), {"k": (1,)}, [(1,)]):
        with pytest.raises(TypeError):
            canonical_bytes(value)
        with _fallback():
            with pytest.raises(TypeError):
                canonical_bytes(value)


# ---- depth, cycles, size -------------------------------------------------------------------

def _nested_list(depth: int):
    root = cur = []
    for _ in range(depth):
        nxt = []
        cur.append(nxt)
        cur = nxt
    return root


def test_hostile_depth_400_is_identical_on_both_backends():
    want = b"[" * 401 + b"]" * 401
    assert _both_backends(_nested_list(400)) == want == rfc8785.dumps(_nested_list(400))


@pytest.mark.xfail(strict=True, raises=RecursionError, reason=(
    "DEFECT (minor): the nesting ceiling depends on the backend -- the rfc8785 path canonicalizes "
    "depth 600 and the fallback path raises RecursionError near depth 500, so the same input is "
    "in the domain of one backend and outside the other's"))
def test_hostile_nesting_ceiling_is_the_same_on_both_backends():
    assert _both_backends(_nested_list(600)) == b"[" * 601 + b"]" * 601


@pytest.mark.xfail(strict=True, raises=RecursionError, reason=(
    "DEFECT: a self-referential input escapes as RecursionError, which is neither of the two "
    "documented exception classes; json.dumps raises ValueError('Circular reference detected')"))
def test_hostile_cyclic_input_raises_a_documented_exception():
    cyc = []
    cyc.append(cyc)
    with pytest.raises(ValueError):
        canonical_bytes(cyc)
    cycd = {}
    cycd["k"] = [cycd]
    with pytest.raises(ValueError):
        canonical_bytes(cycd)


def test_hostile_large_inputs():
    text = "".join(C0) * 20000            # 640k characters, every one escaped
    out = canonical_bytes(text)
    assert len(out) == 2 + 20000 * (5 * 2 + 27 * 6)
    assert json.loads(out) == text
    big_str = "x" * (2 * 1024 * 1024)
    assert canonical_bytes(big_str) == b'"' + big_str.encode() + b'"'
    keys = ["k%06d" % i for i in range(50000)]
    obj = {k: i for i, k in enumerate(reversed(keys))}
    out = canonical_bytes(obj)
    assert list(json.loads(out)) == keys
    assert canonical_bytes(list(range(100000))) == json.dumps(list(range(100000)), separators=(",", ":")).encode()


def test_hostile_input_is_not_mutated():
    obj = {"z": [1, {"y": -0.0, "x": collections.OrderedDict([("b", 1), ("a", 2)])}], "a": 2**53}
    before = copy.deepcopy(obj)
    canonical_bytes(obj)
    with _fallback():
        canonical_bytes(obj)
    assert obj == before
    assert list(obj) == ["z", "a"]
    assert list(obj["z"][1]["x"]) == ["b", "a"]
    assert isinstance(obj["a"], int) and math.copysign(1.0, obj["z"][1]["y"]) < 0


def test_hostile_only_the_two_documented_exception_classes_escape_on_flat_garbage():
    rng = random.Random(1234)
    garbage = [
        b"x", bytearray(), memoryview(b"x"), {1, 2}, frozenset(), (1,), object(), _PlainE.RED,
        float("nan"), float("inf"), -float("inf"), 2**53 + 1, -(2**60), LONE_HIGH, LONE_LOW,
        _dt.datetime.now(), _dt.timedelta(1), complex(0, 1), range(2), slice(1), Ellipsis,
        NotImplemented, type, canonical_bytes, _EvilInt(1), _NaNFloat(0.0),
    ]
    ok = [None, True, False, 0, 1, -1, 1.5, "s", "", [], {}]
    for _ in range(2000):
        pool = ok + garbage
        value = rng.choice(pool)
        if rng.random() < 0.5:
            value = [rng.choice(pool), value, {"k": rng.choice(pool)}]
        if rng.random() < 0.3:
            value = {rng.choice(["a", 1, None, b"k", 1.5, (1,), LONE_HIGH]): value}
        for run in (lambda: canonical_bytes(value), lambda: digest(value)):
            try:
                run()
            except (TypeError, ValueError):
                pass
            except BaseException as e:  # noqa: BLE001 - the point is to catch anything else
                pytest.fail(f"{type(e).__name__} escaped for {value!r}")
