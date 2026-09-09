"""The spec and the code must agree about the constants they both name.

WHY THIS FILE EXISTS. On 2026-09-09 the first complete run of this system produced a noise floor
whose ten pairwise distances were all exactly zero — a valid cert, a valid plan, a valid log, and a
number that measured nothing. The cause was not the runner. Amendment A-29 had split the recipe in
the specification so that `batch_size` lives in `execution` and is never comparability-gating,
precisely so a floor may vary it; the code never received that change and still gates on the whole
`decoding` block. So varying the batch size made runs incomparable (exit 3), holding it fixed made
every distance zero, and the system silently took the second road.

Nothing could have caught that, because no artifact checked the specification against the
implementation. This file is that check, for the part of the specification that is mechanically
checkable: the constants both documents name out loud.

WHAT IT DOES NOT DO. It cannot check prose, and most of the specification is prose. It checks
enumerations, field lists, tags and codes. A passing run means the two agree about those and says
nothing about the rest. That boundary is the point: this is a narrow instrument with its aperture
written down, not an assurance that the code implements the spec.

SILENCE IS NOT SUCCESS. Every extraction below fails loudly when it cannot find its anchor in the
spec text. A regex that silently matches nothing would turn this file into a test that passes
because it looked at nothing, which is the defect it exists to prevent, one level up.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from styxx.v8 import consts as C

SPEC = Path(__file__).resolve().parents[1] / "papers" / "v8" / "SPEC_v8_v0.2_draft.md"


@pytest.fixture(scope="module")
def spec_text() -> str:
    if not SPEC.exists():                       # a missing spec is a failure, never a skip
        pytest.fail(f"the specification is not at {SPEC}; this check cannot run without it")
    text = SPEC.read_text(encoding="utf-8")
    assert len(text) > 20000, f"specification is implausibly short ({len(text)} chars)"
    return text


def _require(pattern: str, text: str, what: str, flags: int = 0) -> re.Match:
    """Find an anchor in the spec, or fail naming what could not be found.

    The failure mode this guards against: the spec is reworded, the pattern stops matching, and a
    test that asserts nothing starts passing.
    """
    m = re.search(pattern, text, flags)
    if m is None:
        pytest.fail(
            f"cannot locate {what} in the specification. Either the spec no longer states it, or "
            f"the wording moved and this check has gone blind. Repair the pattern or the spec; do "
            f"not delete the assertion."
        )
    return m


# --------------------------------------------------------------------------- types

def test_the_eight_cert_types_agree(spec_text):
    m = _require(r'"type":\s*"([^"]+)"', spec_text, "the cert type enumeration in section 2")
    spec_types = tuple(t.strip() for t in m.group(1).split("|"))
    assert spec_types == C.TYPES, (
        f"the specification names {spec_types} and the code names {C.TYPES}"
    )


def test_the_schema_version_agrees(spec_text):
    m = _require(r'"styxx":\s*"([0-9]+\.[0-9]+)"', spec_text, "the schema version in section 2")
    assert m.group(1) == C.SCHEMA_VERSION


# --------------------------------------------------------------------------- recipe core

@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DRIFT, recorded 2026-09-09 in papers/v8/vacuous_floor_2026_09_09/. Amendment A-29 "
        "split the recipe in the specification so that batch_size sits in `execution` and is never "
        "comparability-gating; the code still gates on the whole `decoding` block. This marker is "
        "strict: when the code receives A-29 this test passes, the strict xfail turns that into a "
        "failure, and whoever lands the fix is told to delete the marker. Do not delete the "
        "assertion instead."
    ),
)
def test_recipe_core_agrees(spec_text):
    """The exact drift that produced the vacuous floor. This is the assertion that would
    have caught it."""
    m = _require(
        r"`recipe_core`\s*=\s*\(([^)]*)\)", spec_text, "the recipe_core definition in section 2.3"
    )
    spec_fields = tuple(f.strip().strip("`") for f in m.group(1).split(",") if f.strip())
    code_fields = tuple(C.RECIPE_CORE_FIELDS) if hasattr(C, "RECIPE_CORE_FIELDS") else None
    if code_fields is None:
        from styxx.v8 import cert
        code_fields = tuple(cert.RECIPE_CORE_FIELDS)
    assert spec_fields == code_fields, (
        f"recipe_core drift. The specification says {spec_fields}; the code says {code_fields}.\n"
        f"This is the defect recorded in papers/v8/vacuous_floor_2026_09_09/: when the code gates "
        f"comparability on the whole `decoding` block, two runs that differ only in batch size are "
        f"incomparable, so a floor plan that varies the batch size cannot produce a distance, so "
        f"the floor is measured with the batch size held fixed and every distance is zero."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The same known drift as test_recipe_core_agrees: the code gates on the undivided "
        "`decoding` block. Strict, so the repair announces itself. "
        "papers/v8/vacuous_floor_2026_09_09/."
    ),
)
def test_execution_is_not_comparability_gating(spec_text):
    """Section 2.3 says `execution` is never comparability-gating. If the code gates on a field
    the spec puts in `execution`, a floor that varies it cannot exist."""
    _require(
        r"`execution`.{0,400}?never comparability-gating",
        spec_text,
        "the sentence in section 2.3 stating that execution is never comparability-gating",
        re.S,
    )
    from styxx.v8 import cert
    gating = tuple(cert.RECIPE_CORE_FIELDS)
    # The weak form of this check ("is 'batch_size' one of the gating field names?") passes while
    # the defect is present, because the gating field is `decoding`, which CONTAINS batch_size.
    # Name the container.
    assert "decoding" not in gating, (
        f"the specification splits the recipe into `decoding_core` (gating) and `execution` "
        f"(nuisance, never gating), but the code gates on the undivided `decoding` block: "
        f"RECIPE_CORE_FIELDS = {gating}. Two runs differing only in batch size are then "
        f"incomparable (exit 3), so a floor plan that varies the batch size cannot produce a "
        f"distance. See papers/v8/vacuous_floor_2026_09_09/."
    )
    for nuisance_field in ("batch_size", "padding_side", "device", "execution"):
        assert nuisance_field not in gating, (
            f"the specification puts `{nuisance_field}` in `execution` and says execution is never "
            f"comparability-gating, but the code gates on it: {gating}"
        )


# --------------------------------------------------------------------------- vocabularies

def test_the_family_enum_agrees(spec_text):
    m = _require(
        r'"families":\s*\[([^\]]+)\]', spec_text,
        "the families enumeration in the battery cert body of section 4.5",
    )
    spec_families = tuple(f.strip().strip('"').strip("`") for f in m.group(1).split(",") if f.strip())
    assert spec_families == C.FAMILIES, (
        f"family enum drift: spec {spec_families} vs code {C.FAMILIES}"
    )


def test_the_domain_tags_agree(spec_text):
    """A changed tag silently invalidates every signature ever made under the old one."""
    for tag, name in ((C.CERT_TAG, "cert"), (C.STH_TAG, "tree head"), (C.SEAL_TAG, "seal")):
        assert tag in spec_text, (
            f"the code signs {name} preimages under the domain tag {tag!r}, which does not appear "
            f"anywhere in the specification"
        )


def test_the_channels_agree(spec_text):
    for channel in C.CHANNELS:
        assert re.search(rf"\*\*{channel}\*\*|`{channel}`", spec_text), (
            f"the code declares a channel `{channel}` the specification never names"
        )


def test_the_battery_kinds_agree(spec_text):
    m = _require(r'"kind":\s*"(pool-v1[^"]*)"', spec_text, "the battery kind enumeration in section 4.5")
    spec_kinds = tuple(k.strip() for k in m.group(1).split("|"))
    assert spec_kinds == C.BATTERY_KINDS


# --------------------------------------------------------------------------- exit codes

def test_every_exit_code_the_spec_tabulates_agrees(spec_text):
    """Section 6's table is the CI contract. A code that means one thing in the table and another
    in the code is a contract break that no test would otherwise see."""
    table = _require(
        r"\|\s*code\s*\|\s*meaning\s*\|(.{0,6000}?)\n\n", spec_text,
        "the exit code table in section 6", re.S,
    ).group(1)
    rows = re.findall(r"^\|\s*(\d)\s*\|\s*([^|]+?)\s*\|", table, re.M)
    assert len(rows) >= 4, f"parsed only {len(rows)} rows from the exit code table; the pattern has gone blind"
    spec_codes = {int(code) for code, _ in rows}
    code_values = set(C.EXIT.values())
    assert spec_codes <= code_values, (
        f"the specification tabulates exit codes {sorted(spec_codes)} and the code can emit "
        f"{sorted(code_values)}; the specification names one the code cannot produce"
    )
    for code, meaning in rows:
        verdicts = [v for v, c in C.EXIT.items() if c == int(code)]
        assert verdicts, f"section 6 tabulates exit {code} ({meaning.strip()}) and no verdict maps to it"


# --------------------------------------------------------------------------- the aperture itself

def test_this_check_states_what_it_cannot_see():
    """A calibration note, asserted so it cannot quietly disappear.

    The lab's standing rule is that an agreement number without its detection power is not a
    number. This file compares constants. It does not read prose, does not execute the system, and
    would not have caught: a rule stated in the spec and never implemented anywhere, a rule
    implemented differently from its description, or any of the four defects that the first real
    run surfaced by use rather than by reading.
    """
    covered = {"cert types", "schema version", "recipe_core", "execution not gating",
               "family enum", "domain tags", "channels", "battery kinds", "exit codes"}
    not_covered = {"prose rules", "algorithm behaviour", "append refusals", "verdict precedence",
                   "field presence per type", "anything stated only in Appendix B or C"}
    assert covered & not_covered == set()
    assert len(not_covered) >= 6, "the miss list must be stated, not implied"
