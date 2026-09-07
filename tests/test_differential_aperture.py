"""The generator can reach the characters that hid defects, and cannot reach the ones that would lie.

On 2026-09-06 the differential harness reported `100000 agree, 0 disagree` at a seed it had never
run, and the generator's reachable alphabet was measured at 98 code points, SEVEN of them non-ASCII.
Every character-level defect that day's audit found lay outside it:

    U+2010 and 25 other dashes   a minus sign silently dropped, -0.42 HELD against 0.42
    U+202E                       the reader sees 55.0, the verifier checked 0.55
    U+0085                       SWORN-FAILED in Python, SWORN-HELD in node
    U+10D40                      a Unicode-version skew between the two runtimes

So the agreement number was true of a 98-character alphabet and was read as a statement about the
two implementations. This pins the repair.

TWO CLAIMS, AND THE SECOND IS THE INTERESTING ONE.

  1. the alphabet REACHES the first three, so a regression of any of those repairs is generatable;
  2. the alphabet does NOT reach U+10D40, and must not.

The second is not an oversight being enshrined. CPython here is on Unicode 15.0.0 and V8's ICU on
16.0, so the two runtimes classify that code point differently and no edit to either implementation
can make them agree. A generator that emitted it would turn this harness from a defect detector into
a runtime detector -- red on this machine, green on one whose runtimes match, for something nobody
can fix. The aperture widens only where the two runtimes already agree, and every member of
APERTURE_ALPHABET was checked against conformance/sworn/class_census.py.

See papers/sworn/RESULT_aperture_widening_2026_09_07.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MISSED = {0x2010: "a dash that dropped a minus sign",
          0x202E: "the directional override",
          0x0085: "the path-segment divergence"}
SKEWED = {0x10D40: "a Garay digit, Nd in Unicode 16.0 and unassigned in 15.0.0"}

SAMPLE = 12000
SEED = 20260905


@pytest.fixture(scope="module")
def alphabet():
    try:
        import conformance.sworn.differential as D
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip("the differential harness is not importable here: %s" % exc)
    seen = set()
    for i in range(SAMPLE):
        try:
            seen.update(ord(c) for c in D.case(SEED, i)["document"].decode("utf-8"))
        except Exception:                                        # noqa: BLE001
            continue
    assert seen, "the generator produced no readable document"
    return seen


@pytest.mark.parametrize("cp", sorted(MISSED), ids=lambda c: "U+%04X" % c)
def test_the_generator_reaches_the_characters_that_hid_defects(cp, alphabet):
    assert cp in alphabet, (
        "U+%04X (%s) is outside the generator's reach again: a regression of that repair could not "
        "be generated, and an agreement number would not cover it" % (cp, MISSED[cp]))


@pytest.mark.parametrize("cp", sorted(SKEWED), ids=lambda c: "U+%04X" % c)
def test_the_generator_does_not_reach_a_runtime_skewed_character(cp, alphabet):
    assert cp not in alphabet, (
        "U+%04X (%s) became generatable. The two runtimes classify it differently, so this harness "
        "would now report a disagreement no edit to either implementation can fix — a defect "
        "detector turned into a runtime detector" % (cp, SKEWED[cp]))


def test_the_alphabet_is_materially_wider_than_it_was(alphabet):
    """98 code points, 7 non-ASCII, was the measurement that made the miss list possible."""
    non_ascii = {c for c in alphabet if c > 0x7F}
    assert len(alphabet) >= 130, "reachable alphabet shrank to %d code points" % len(alphabet)
    assert len(non_ascii) >= 50, (
        "non-ASCII reach shrank to %d code points; it was 7 when four defects hid outside it"
        % len(non_ascii))


def test_every_injected_character_is_one_both_runtimes_agree_on():
    """The safety property that lets the aperture widen at all.

    Checked against the census rather than asserted: if a future edit adds a character to
    APERTURE_ALPHABET that the two runtimes classify differently, this fails before the harness
    starts reporting unfixable disagreements.
    """
    from shutil import which
    if which("node") is None:
        pytest.skip("node is not available")
    try:
        import conformance.sworn.differential as D
        sys.path.insert(0, str(ROOT / "conformance" / "sworn"))
        import class_census as C
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip("the census is not importable here: %s" % exc)

    skew = set()
    for _label, kind, py, js in C.census():
        if kind == "property":
            skew |= (py ^ js)
    unsafe = sorted(ord(ch) for ch in D.APERTURE_ALPHABET if ord(ch) in skew)
    assert not unsafe, (
        "APERTURE_ALPHABET contains %d code point(s) the two runtimes classify differently: %s"
        % (len(unsafe), ["U+%04X" % c for c in unsafe[:8]]))
