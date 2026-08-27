"""The LEDGER's refusal classifier, asserted by hand.

`tests/test_ledger.py` regenerates `papers/LEDGER.md` and fails on a single changed character.
That guard proves the committed file agrees with its generator. It cannot prove the generator is
RIGHT — and for months it did not, while the section headed "cycles where a preregistered gate
returned `INVALID__*`" listed `SHIPPED`, `PRODUCT`, `DO`, `REWRITTEN` and `BUILT`, and counted the
cycle that built the ledger as a loss because its verdict quotes the ledger's own negatives count.

A self-consistency check will accept a wrong classifier forever. These are the assertions that
will not.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papers"))

from ledger_verdicts import is_refusal, refusal_tokens  # noqa: E402

CYCLE_LOG = ROOT / "papers" / "autopilot" / "CYCLE_LOG.jsonl"

# Adjudicated by hand against the real log on 2026-08-26. The verdict field is free prose, so
# these are judgements about what each cycle's gate actually returned, not regex output.
REFUSALS = (67, 85, 101, 110, 115, 117, 118, 132, 149)

# Every cycle whose verdict MENTIONS an INVALID__ token without having returned one. Each is a
# case the old substring rule got wrong, and each names why.
MENTIONS_ONLY = {
    133: "SHIPPED to PyPI; its prose discusses an earlier invalid",
    134: "PRODUCT RECALL — a real negative, but not a machinery refusal",
    142: "INSTRUMENT_BLIND_TO_ISC — a real negative, not an INVALID__ verdict",
    152: "DO NOT SHIP AS-IS — a real negative, not an INVALID__ verdict",
    154: "REWRITTEN AND PASSING ITS OWN KILL LIST",
    156: "BUILT the ledger; counted as a loss because it quotes the ledger's negatives count",
    157: "NO_LEGIBILITY_ISLANDS — quotes b48's INVALID__ in a parenthetical",
}


def _cycles():
    if not CYCLE_LOG.exists():
        pytest.skip("CYCLE_LOG.jsonl not present")
    return {c["cycle"]: c.get("verdict", "")
            for c in (json.loads(x) for x in
                      CYCLE_LOG.read_text(encoding="utf-8").splitlines() if x.strip())}


# ---------------------------------------------------------------- against the real record

@pytest.mark.parametrize("cycle", REFUSALS)
def test_real_machinery_refusals_are_classified_as_refusals(cycle):
    v = _cycles().get(cycle)
    if v is None:
        pytest.skip(f"cycle {cycle} not in the log")
    assert is_refusal(v), f"cycle {cycle} returned an INVALID__ verdict and must count"


@pytest.mark.parametrize("cycle,why", sorted(MENTIONS_ONLY.items()))
def test_cycles_that_merely_mention_an_invalid_are_not_refusals(cycle, why):
    v = _cycles().get(cycle)
    if v is None:
        pytest.skip(f"cycle {cycle} not in the log")
    assert not is_refusal(v), f"cycle {cycle} is not a machinery refusal: {why}"


def test_the_refusal_set_is_exactly_this_and_nothing_else():
    """Asserted in both directions. A new refusal fails here, and so does a repaired one."""
    got = {n for n, v in _cycles().items() if is_refusal(v)}
    assert got == set(REFUSALS), (
        f"refusal set changed: added {sorted(got - set(REFUSALS))}, "
        f"removed {sorted(set(REFUSALS) - got)}. Adjudicate the change and update REFUSALS "
        "with a recorded reason — do not widen the regex until this passes.")


def test_the_ledger_never_prints_a_first_word_again():
    """`cycle 110 — TWO` is what a first-word renderer produced. It must print real tokens."""
    cycles = _cycles()
    for n in REFUSALS:
        v = cycles.get(n)
        if v is None:
            continue
        rendered = refusal_tokens(v)
        assert "INVALID__" in rendered, f"cycle {n} rendered as {rendered!r}"
        first_word = v.strip().split()[0].strip("(")
        if not first_word.startswith("INVALID__"):
            assert first_word not in rendered, (
                f"cycle {n} rendered its first word {first_word!r} instead of its token")


def test_cycle_115_shows_both_of_its_invalids():
    v = _cycles().get(115)
    if v is None:
        pytest.skip("cycle 115 not in the log")
    assert refusal_tokens(v).count("INVALID__") == 2, "115 returned two independent invalids"


# ---------------------------------------------------------------- unit cases

@pytest.mark.parametrize("verdict", [
    "INVALID__underpowered (FINDING OATH-HELD 9/13/0)",
    "INVALID__null_leaks. everything after the period is commentary.",
    "TWO INVALIDS, TWO MECHANISMS, NEITHER GENERALITY CLAIM LICENSED",
    "TWO HONEST INVALIDS, FAMILY PARKED PER ITS OWN CLAUSE",
    "(INVALID__parenthesised_lead) still a refusal",
])
def test_verdicts_that_are_refusals(verdict):
    assert is_refusal(verdict)


@pytest.mark.parametrize("verdict", [
    "SHIPPED to PyPI as 7.31.0; supersedes the INVALID__pipeline_broken run",
    "BUILT AND LINKED. the ledger counts INVALID__ verdicts, of which there are several",
    "DO NOT SHIP AS-IS -> six defects fixed. cites INVALID__breaks_existing_preregs",
    "NO_LEGIBILITY_ISLANDS__does_not_generalize (b48's INVALID__null_artifact is quoted here)",
    "",
    "   ",
])
def test_verdicts_that_are_not_refusals(verdict):
    assert not is_refusal(verdict)


def test_a_verdict_with_no_invalid_token_still_renders_legibly():
    assert refusal_tokens("SOMETHING_ELSE") == "`INVALID__*`"


def test_tokens_are_deduplicated_and_sorted():
    v = "INVALID__b, INVALID__a, INVALID__b again"
    assert refusal_tokens(v) == "`INVALID__a`, `INVALID__b`"
