"""Two functions in this repository hash an item order, and they disagree.

`styxx/v8/sweep.py::item_order_sha256` computes ``sha256(UTF-8("\\n".join(ids)))`` and its docstring
cites section 3.1's `nuisance`. `styxx/v8/fingerprint.py::order_sha256` computes
``sha256(UTF-8(JCS([ids])))`` over the same thing. Both are live, and nothing compares them.

Found by an adversary asked to attack a different claim entirely, which is the usual way. Pinned
here rather than repaired, for two reasons:

* **A receipt is history.** `sweep.py`'s digest is written into committed probe records and
  parameter blocks. Changing the function changes what those records mean, and this lab does not
  regenerate a committed receipt in place.
* **Which one is canonical is a decision, not a deduction.** The published certificates match the
  JCS form, which is evidence about what was minted, not about what the specification requires.
  Section 3.1 has to say which, and until it does an implementer can read either function as the
  reference.

The concrete cost, and the reason this is not cosmetic: the proposed repair for THE_BOUNDARY's
class-two member 2 is the predicate *"the runs' assignments are the plan's schedule, in order"*. A
plan whose order digest came from one function, compared against a fingerprint whose order digest
came from the other, false-refuses on its first day. A refusal predicate built on two disagreeing
digests is worse than no predicate, because it accuses honest parties.
"""

from __future__ import annotations

import glob
import hashlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "papers" / "v8" / "first_verdict_2026_09_09" / "log" / "entries"


def _newline_join(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _jcs_array(ids) -> str:
    body = "[" + ",".join(json.dumps(i, ensure_ascii=False) for i in ids) + "]"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _published_runs():
    out = []
    for f in sorted(glob.glob(str(LOG / "*" / "*[0-9].json"))):
        cert = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        if cert.get("type") != "fingerprint":
            continue
        nuisance = cert["body"].get("nuisance") or {}
        stored = nuisance.get("item_order_sha256")
        items = cert["body"].get("items") or []
        if stored and items:
            out.append((pathlib.Path(f).stem, stored, [str(i["item_id"]) for i in items]))
    return out


def test_the_published_certs_are_available() -> None:
    runs = _published_runs()
    assert len(runs) == 5, (
        f"expected the 5 published fingerprints, found {len(runs)}. This pin reads the committed "
        f"log; if that moved, repair the path rather than deleting the test."
    )


@pytest.mark.parametrize("entry,stored,ids", _published_runs(),
                         ids=[r[0] for r in _published_runs()])
def test_the_two_implementations_disagree(entry: str, stored: str, ids: list[str]) -> None:
    """Both are computed here from the cert's own item ids. They must not agree by accident."""
    a, b = _newline_join(ids), _jcs_array(ids)
    assert a != b, (
        "the two preimage conventions produced the same digest, which would mean this finding "
        "has evaporated or the reimplementation here is wrong. Check before deleting the pin."
    )


@pytest.mark.parametrize("entry,stored,ids", _published_runs(),
                         ids=[r[0] for r in _published_runs()])
def test_the_published_certs_used_the_json_form(entry: str, stored: str, ids: list[str]) -> None:
    """Evidence about what was minted, not about what the spec requires."""
    assert stored == _jcs_array(ids), (
        f"{entry}: stored {stored} matches neither reimplementation "
        f"(newline {_newline_join(ids)}, json {_jcs_array(ids)}). Either the ordering convention "
        f"is a third thing, or the recorded items are not in run order."
    )
    assert stored != _newline_join(ids)


@pytest.mark.xfail(
    strict=True,
    reason="Two live functions hash an item order with different preimages: "
           "sweep.item_order_sha256 joins with newlines, fingerprint.order_sha256 uses JCS. "
           "Section 3.1 does not say which is the reference. Delete this marker when the "
           "specification fixes one and both call sites agree -- NOT by silently changing "
           "sweep.py, whose digest is written into committed probe records.",
)
def test_one_convention_is_named_somewhere() -> None:
    sweep = (ROOT / "styxx" / "v8" / "sweep.py").read_text(encoding="utf-8")
    fingerprint = (ROOT / "styxx" / "v8" / "fingerprint.py").read_text(encoding="utf-8")
    both_newline = "\\n".join in sweep and "\\n".join in fingerprint
    both_jcs = "canonical_bytes" in sweep and "canonical_bytes" in fingerprint
    assert both_newline or both_jcs, "the two modules still use different preimages"
