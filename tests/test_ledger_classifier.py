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


# ---------------------------------------------------------------- the machine-readable token

def test_an_unadjudicated_cycle_is_not_assumed_to_be_anything():
    """Unknown is a third answer. Defaulting it either way flatters, in one direction or another."""
    from ledger_verdicts import classify, token_of
    assert token_of({}) is None
    assert token_of({"verdict_token": "   "}) is None
    assert token_of({"verdict_token": "INVALID__x"}) == "INVALID__x"
    assert classify(None) is None
    assert classify("SOMETHING_NOBODY_ADJUDICATED") is None


@pytest.mark.parametrize("token", [
    "INVALID__underpowered", "REFUSED__power_basis_does_not_transfer", "NULL__below_ceiling",
    "CLOSED_NEGATIVE", "NO_LEGIBILITY_ISLANDS__x",
])
def test_negative_tokens_classify_negative(token):
    from ledger_verdicts import classify
    assert classify(token) is True


@pytest.mark.parametrize("token", ["V11_OVERREACH", "V11_BATTERY_VOID", "V11_FUSE",
                                   "V11_ORDINAL_RETRACTION_SHIPS"])
def test_cycle_scoped_tokens_are_unrecognised_rather_than_guessed(token):
    """A per-prereg outcome token is defined by ITS OWN frozen outcome table, and no general
    classifier can know that vocabulary. `V11_BATTERY_VOID` reads as a void to a human and as
    an unknown prefix to a prefix rule — and the honest answer is unknown.

    Widening the rule to match `VOID` anywhere inside the token is precisely the substring-over-
    text move that produced `SHIPPED` in the refusal list. These stay unrecognised until a
    human registers them, which is a cheap thing to do and an expensive thing to fake.
    """
    from ledger_verdicts import classify
    assert classify(token) is None


@pytest.mark.parametrize("token", [
    "SURVIVED__vs_adaptive_erasure", "LICENSED__cohort_coupling", "DOOR_OPENS", "SHIPPED",
])
def test_positive_tokens_classify_positive(token):
    from ledger_verdicts import classify
    assert classify(token) is False


def test_coverage_reports_progress_rather_than_a_number_it_cannot_compute():
    from ledger_verdicts import adjudication_coverage
    recs = [{"verdict_token": "INVALID__a"}, {"verdict_token": "SHIPPED"},
            {"verdict_token": "MYSTERY"}, {}]
    c = adjudication_coverage(recs)
    assert c == {"cycles": 4, "with_verdict_token": 3, "without_verdict_token": 1,
                 "negative": 1, "positive": 1, "token_unrecognised": 1,
                 "ratio_is_computable": False}


def test_the_ratio_is_computable_only_when_every_cycle_is_adjudicated():
    from ledger_verdicts import adjudication_coverage
    assert adjudication_coverage(
        [{"verdict_token": "INVALID__a"}, {"verdict_token": "SHIPPED"}])["ratio_is_computable"]
    assert not adjudication_coverage(
        [{"verdict_token": "INVALID__a"}, {}])["ratio_is_computable"]


def test_the_real_corpus_is_not_yet_adjudicated_and_says_so():
    """Anchored to reality: when this starts failing, the migration is underway and the ledger's
    disclosure line should be checked against it."""
    from ledger_verdicts import adjudication_coverage
    c = adjudication_coverage([json.loads(x) for x in
                               CYCLE_LOG.read_text(encoding="utf-8").splitlines() if x.strip()])
    assert c["cycles"] == 163
    assert not c["ratio_is_computable"], (
        "cycles now carry verdict_token — update papers/LEDGER.md's disclosure and this test")
