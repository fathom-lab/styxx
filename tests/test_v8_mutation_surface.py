"""What the mutation measurement does NOT measure, pinned so the gap is visible.

`conformance/v8/mutation_coverage.json` is this project's answer to its own standing rule that a
differential agreement number without detection power is not a number. It mutates the
implementation and reports which mutations the committed vector set would catch. It is the right
instrument and its miss list is published, which is more than most such numbers get.

This file measures the instrument's *aperture* rather than its hit rate, because a mutation score
says nothing about code no mutation was ever proposed against. Re-measured against the tree today,
the committed receipt reproduces exactly — 17 caught, 5 missed, 22 viable over 26 proposed, every
module hash and the set digest matching — so the receipt is honest about what it did. These tests
are about what it did not do.

Two gaps, both pinned:

* **The canonicalization layer is not in the measurement.** `styxx/v8/jcs.py` produces the bytes
  every certificate id, every signature preimage and every Merkle leaf is computed over. A silent
  defect there changes every artifact this system has ever produced. It carries no mutations.
  `styxx/v8/fingerprint.py`, which builds the objects being certified, carries none either.
* **`log.py` is measured thinly relative to what it now does.** Three viable mutations, one missed.
  Over one day it gained the log-binding predicate, cross-log snapshot agreement, the
  stale-versus-contradicted metadata split and the baseline-gap repair. None of that is in the
  catalogue, so the coverage figure for the module is a figure about the module as it was.

Neither is a defect in the receipt. Both are reasons not to read a mutation score as a statement
about the system.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "conformance" / "v8" / "mutation_coverage.json"

# Modules whose bytes decide what every other module's output means.
LOAD_BEARING = ("styxx/v8/jcs.py", "styxx/v8/fingerprint.py")


def _receipt() -> dict:
    if not RECEIPT.exists():
        pytest.skip("mutation_coverage.json is absent; this pin has gone blind, not passed")
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_the_receipt_still_has_the_shape_these_pins_read() -> None:
    d = _receipt()
    for key in ("implementation", "by_module", "counts", "missed"):
        assert key in d, (
            f"mutation_coverage.json no longer carries {key!r}; repair these pins deliberately "
            f"rather than letting them assert nothing."
        )


def test_the_published_aperture_is_recorded() -> None:
    """Not a threshold. A number a reader should see next to the coverage rate."""
    d = _receipt()
    counts = d["counts"]
    assert counts["viable"] > 0
    modules_with_results = len(d["by_module"])
    assert modules_with_results <= len(d["implementation"]), (
        "more modules have mutation results than are pinned, which means the receipt and the "
        "implementation map disagree about what was measured"
    )


@pytest.mark.parametrize("module", LOAD_BEARING)
@pytest.mark.xfail(
    strict=True,
    reason="The canonical-bytes layer and the fingerprint builder carry no mutations, so the "
           "coverage rate says nothing about them. jcs.py decides the bytes behind every cert id, "
           "every signature preimage and every Merkle leaf. Delete this marker when the catalogue "
           "proposes mutations there and the receipt reports them.",
)
def test_the_load_bearing_modules_are_mutated(module: str) -> None:
    d = _receipt()
    assert module in d["implementation"], f"{module} is not pinned in the implementation map"
    assert d["by_module"].get(module, {}).get("viable", 0) > 0, (
        f"{module} has no viable mutations in the receipt"
    )


@pytest.mark.xfail(
    strict=True,
    reason="log.py gained the binding predicate, cross-log snapshot agreement, the "
           "stale-versus-contradicted metadata split and the baseline-gap repair in one day, and "
           "the catalogue proposes no mutations against any of them. Its 3 viable mutations "
           "describe an earlier module. Delete this marker when the catalogue covers the new "
           "predicates.",
)
def test_log_py_is_measured_in_proportion_to_what_it_does() -> None:
    d = _receipt()
    viable = d["by_module"].get("styxx/v8/log.py", {}).get("viable", 0)
    assert viable >= 8, (
        f"log.py has {viable} viable mutations. The threshold here is a judgement, not a law: it "
        f"is set at roughly one per predicate the module now decides, and it is deliberately "
        f"visible so that raising it is a decision someone makes rather than a drift."
    )


def test_the_unmeasured_modules_are_named_not_hidden() -> None:
    """A list a reader can act on, printed as the assertion message when it grows."""
    d = _receipt()
    present = {p.as_posix().replace("\\", "/") for p in (ROOT / "styxx" / "v8").glob("*.py")}
    present = {p.split("styxx/v8/")[-1] for p in present}
    pinned = {k.split("/")[-1] for k in d["implementation"]}
    unmeasured = sorted(present - pinned - {"__init__.py", "__main__.py", "consts.py"})
    assert unmeasured, (
        "every module is now in the mutation measurement, which would be a real improvement; "
        "delete this test and the two xfails above with it."
    )
