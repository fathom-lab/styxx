"""Pin the two fields that are demonstrably uncorroborated, and keep the census from going blind.

`papers/v8/class_two_empty_2026_09_09/lone_difference_census.py` builds the roster mechanically
instead of by inspection: it looks for two real certificates that differ in exactly one field, which
is a demonstration that forging that field forces no other byte to move. Applied to the thirteen
certificates in this arc it names `recipe.decoding.batch_size` and `subject.precision` -- roster
member 1, and the member no roster in `THE_BOUNDARY_2026_09_09.md` ever listed.

Two kinds of test here, and the distinction is the point:

* **Passing tests** keep the instrument honest. If the corpus moves or the census stops finding the
  two fields it is known to find, that is the census going blind, and a blind census reports a clean
  sheet. This lab has a receipt about a checker that answered when it had no evidence.
* **Strict xfails** assert the state we want and do not have. When somebody gives `precision` a
  corroborating byte, the xfail becomes an unexpected pass, the suite goes red, and the repair
  announces itself instead of being noticed a year later.
"""

from __future__ import annotations

import glob
import itertools
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARC = ROOT / "papers" / "v8" / "first_verdict_2026_09_09"
CENSUS = ROOT / "papers" / "v8" / "class_two_empty_2026_09_09" / "lone_difference_census.py"

IGNORED_PREFIXES = ("id", "sig", "created", "body", "refs", "issuer", "styxx")
INTERESTING = ("subject", "recipe")

KNOWN_UNCORROBORATED = ("subject.precision", "recipe.decoding.batch_size")


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    else:
        out[prefix] = json.dumps(obj, sort_keys=True)
    return out


def _interesting(path: str) -> bool:
    head = path.split(".")[0]
    return head not in IGNORED_PREFIXES and head in INTERESTING


def _corpus():
    out = []
    for f in sorted(glob.glob(str(ARC / "log" / "entries" / "*" / "*[0-9].json"))):
        out.append((f"log:{pathlib.Path(f).stem}", json.loads(pathlib.Path(f).read_text("utf-8"))))
    for sub in ("fp_bf16", "fp_fp16"):
        for f in sorted(glob.glob(str(ARC / sub / "*.json"))):
            out.append((sub, json.loads(pathlib.Path(f).read_text("utf-8"))))
    return out


def _lone_differences():
    certs = _corpus()
    flat = {i: {k: v for k, v in _flatten(c).items() if _interesting(k)}
            for i, (_, c) in enumerate(certs)}
    lone = {}
    for (ia, fa), (ib, fb) in itertools.combinations(flat.items(), 2):
        diff = [k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)]
        if len(diff) == 1:
            lone.setdefault(diff[0], []).append((ia, ib))
    return lone


def test_the_corpus_is_present() -> None:
    certs = _corpus()
    assert len(certs) >= 7, (
        f"only {len(certs)} certificates found under {ARC}. The census reads real published bytes; "
        f"if they moved, repair the path rather than letting this assert nothing."
    )


def test_the_census_script_is_present_and_parses() -> None:
    assert CENSUS.exists(), (
        "lone_difference_census.py is gone and THE_BOUNDARY cites it as the receipt for its "
        "mechanical roster. A finding whose receipt was deleted is an assertion."
    )
    compile(CENSUS.read_text(encoding="utf-8"), str(CENSUS), "exec")


@pytest.mark.parametrize("field", KNOWN_UNCORROBORATED)
def test_the_census_still_finds_what_it_is_known_to_find(field: str) -> None:
    """Not a claim about the design. A check that the instrument has not gone blind."""
    lone = _lone_differences()
    assert field in lone, (
        f"the census no longer demonstrates {field!r} as a lone difference. Either the corpus "
        f"changed, or this reimplementation drifted from the script. A census that finds nothing "
        f"reports a clean sheet, which is the failure mode worth catching."
    )


def test_it_finds_no_more_than_it_should() -> None:
    """A census that names everything names nothing. Record the count, do not gate on it."""
    lone = _lone_differences()
    assert set(KNOWN_UNCORROBORATED) <= set(lone)
    extra = sorted(set(lone) - set(KNOWN_UNCORROBORATED))
    assert not extra, (
        f"the census now demonstrates lone differences for {extra}, which were not known when "
        f"this pin was written. That is a FINDING, not a failure: each is a field with no "
        f"corroborating byte inside the certificate. Add it to KNOWN_UNCORROBORATED, record it in "
        f"THE_BOUNDARY's roster, and decide whether anything outside the cert constrains it."
    )


@pytest.mark.xfail(
    strict=True,
    reason="`subject.precision` has no corroborating byte anywhere in the design. The published "
           "bf16 and fp16 subjects differ in that one string and share all four Appendix A.2 "
           "hashes, the revision, the repo and the environment. Excluding it from snapshot "
           "agreement is CORRECT -- hashing it would refuse that honest pair -- so the repair is "
           "not 'add it to the hash'. Delete this marker when some logged byte would move if it "
           "were forged.",
)
def test_precision_gains_a_corroborating_byte() -> None:
    lone = _lone_differences()
    assert "subject.precision" not in lone


@pytest.mark.xfail(
    strict=True,
    reason="`recipe.decoding.batch_size` is roster member 1: the batch labels written beside the "
           "recipe meant to corroborate them. Reachable across a log via the cross-certificate "
           "floor predicate, which as of this pin is implemented in JavaScript and called by no "
           "verification path. Delete this marker when a forged batch label forces some other "
           "byte in its own certificate to move.",
)
def test_batch_size_gains_a_corroborating_byte() -> None:
    lone = _lone_differences()
    assert "recipe.decoding.batch_size" not in lone
