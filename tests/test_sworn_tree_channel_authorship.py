"""Invariant 2 — the agent cannot swear to bytes it minted — on every form a receipt can take.

THE DEFECT THIS PINS. `_resolve` refuses a receipt whose sha256 is in `manifest.authored_sha256` on
the rN branch, at one line, with reason `receipt_author_minted`. The tree branch — `path:` and
`prereg:` — computes the resolved bytes' sha256 and never compares it to anything. So the same
bytes, which the manifest itself lists as agent-authored, are refused by id and HELD by path:

    rN        numeric  ->  MALFORMED receipt_author_minted
    path:     numeric  ->  HELD
    prereg:   numeric  ->  HELD
    path:     absent   ->  HELD        <- the strongest verdict, over the author's own bytes

Naming the same bytes by path instead of by id is the whole attack.

Spec: papers/sworn/SPEC_tree_channel_authorship_v01_2026_09_06.md, frozen before the repair. This
file exists before the repair too (T4): it fails against the shipped code on path: and prereg:,
passes on rN, and passes on all three after. It also pins the honest case — a committed file whose
digest is NOT in authored_sha256 — so a repair that refused every tree receipt would fail it (T2).
"""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

C40 = "a" * 40
AUTHORED = b'{"loss": 0}\n'
AUTHORED_SHA = hashlib.sha256(AUTHORED).hexdigest()
HONEST = b'{"loss": 4200000}\n'
HONEST_SHA = hashlib.sha256(HONEST).hexdigest()


def _manifest(authored):
    return sworn.Manifest.from_dict({
        "spec": "sworn/manifest/0.2", "harness": "ci", "turn": "t",
        "minted_at": "2026-09-01T00:00:00Z", "rung": "L2",
        "authored_sha256": list(authored),
        "receipts": {"r1": {"id": "r1", "sha256": AUTHORED_SHA, "kind_of_source": "tool_stdout",
                            "captured_at": "2026-09-01T00:00:00Z", "complete": True,
                            "bytes": base64.b64encode(AUTHORED).decode("ascii")}},
    })


TREE = sworn.SnapshotTree(
    {"audit/loss.json": {"mode": "100644", "size": len(AUTHORED), "sha256": AUTHORED_SHA,
                         "bytes": AUTHORED},
     "audit/honest.json": {"mode": "100644", "size": len(HONEST), "sha256": HONEST_SHA,
                           "bytes": HONEST}},
    C40, commit=C40)


def _span(receipt, kind, sentence, authored=(AUTHORED_SHA,)):
    doc = ('<sworn r="%s" k="%s">%s</sworn>\n' % (receipt, kind, sentence)).encode("utf-8")
    core = sworn.verify(doc, name="d.md", manifest=_manifest(authored), commit=C40, tree=TREE)
    return core["spans"][0]


# ---------------------------------------------------------------- T1: one refusal, three forms

@pytest.mark.parametrize("receipt", [
    "r1#/loss",
    "path:audit/loss.json#/loss",
    "prereg:" + AUTHORED_SHA + "#/loss",
], ids=["rN", "path", "prereg"])
def test_author_minted_bytes_are_refused_on_every_form(receipt):
    s = _span(receipt, "numeric", "the loss is 0.")
    assert (s["verdict"], s["reason"]) == ("MALFORMED", "receipt_author_minted"), (
        "%s resolved author-minted bytes as %s/%s — invariant 2 is enforced on one channel of three"
        % (receipt.split(":")[0] if ":" in receipt else "rN", s["verdict"], s["reason"]))


def test_absent_cannot_be_earned_over_the_authors_own_bytes():
    """The strongest verdict in the format, over bytes the manifest says the agent wrote."""
    s = _span("path:audit/loss.json", "absent",
              "the receipt never mentions `customer_record_exposed` anywhere.")
    assert (s["verdict"], s["reason"]) == ("MALFORMED", "receipt_author_minted"), s


# ---------------------------------------------------------------- T2: the honest case still holds

def test_a_committed_file_that_is_not_author_minted_still_resolves_by_path():
    """Without this, a repair could refuse every tree receipt and pass the tests above."""
    s = _span("path:audit/honest.json#/loss", "numeric", "the loss is 4200000.")
    assert s["verdict"] == "HELD", s


def test_absent_still_holds_over_an_honest_committed_file():
    s = _span("path:audit/honest.json", "absent",
              "the receipt never mentions `customer_record_exposed` anywhere.")
    assert s["verdict"] == "HELD", s
    assert s["detail"].get("complete") is True, (
        "a committed blob is complete — the verifier holds every byte of it (T2)")


def test_prereg_of_an_honest_digest_still_resolves():
    s = _span("prereg:" + HONEST_SHA + "#/loss", "numeric", "the loss is 4200000.")
    assert s["verdict"] == "HELD", s


# ---------------------------------------------------------------- T5: no list, no refusal

@pytest.mark.parametrize("receipt", ["path:audit/loss.json#/loss", "prereg:" + AUTHORED_SHA + "#/loss"],
                         ids=["path", "prereg"])
def test_an_empty_authored_list_refuses_nothing(receipt):
    """The repair adds a comparison, not a requirement."""
    s = _span(receipt, "numeric", "the loss is 0.", authored=())
    assert s["verdict"] == "HELD", s


# ---------------------------------------------------------------- the rN branch is unchanged

def test_rn_refusal_is_unchanged():
    s = _span("r1#/loss", "numeric", "the loss is 0.")
    assert (s["verdict"], s["reason"]) == ("MALFORMED", "receipt_author_minted")
