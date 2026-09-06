"""SWORN-HELD does not mean the spans held, and the headline now says so.

THE DEFECT THIS PINS. The document verdict ladder reads:

    elif counts["FAILED"] == 0 and counts["MALFORMED"] == 0:
        document_verdict = "SWORN-HELD"

`UNRESOLVED` is never consulted. A document in which NOTHING was checked carries the same headline
as one in which everything held — which is precisely the conflation this module's own doctrine
refuses four lines from the top of the file:

    "And a document that swore nothing is ``UNSWORN``, never 'no failures'."

The principle is applied to `sworn_total == 0` and not to `unresolved == sworn_total`.

IT IS AUTHOR-REACHABLE WITHOUT FORGING ANYTHING. A manifest rung this verifier does not know makes
every span UNRESOLVED with reason `rung_unknown`, and that fires BEFORE any receipt id is looked up.
Same document, same receipt, a sentence that contradicts it:

    rung "L2"  ->  FAILED value_mismatch   ->  SWORN-FAILED
    rung "L3"  ->  UNRESOLVED rung_unknown ->  SWORN-HELD

One string in the manifest turns a caught lie into a document that prints SWORN-HELD.

AND IT CATCHES HONEST DOCUMENTS TOO, which is how it was found. Verifying a real committed RESULT
with `--repo .` but without `--commit` resolves nothing: held=0, unresolved=10, SWORN-HELD. With the
commit its sidecar names: held=10, unresolved=0. Both print SWORN-HELD; one checked ten things and
the other checked nothing.

WHAT IS FIXED HERE AND WHAT IS NOT. Renaming the verdict is a breaking change to a published
vocabulary that every consumer of `document_verdict` depends on, and that is the operator's call.
Saying so on the line a reader actually reads is not, so the headline warns. These tests pin the
warning and the conditions under which it must stay silent.
"""
from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

PAYLOAD = b'{"loss": 4200000}'


def _manifest(rung: str) -> dict:
    return {
        "spec": "sworn/manifest/0.2", "harness": "ci", "turn": "t",
        "minted_at": "2026-09-01T00:00:00Z", "authored_sha256": [], "rung": rung,
        "receipts": {"r1": {
            "id": "r1", "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
            "kind_of_source": "tool_stdout", "captured_at": "2026-09-01T00:00:00Z",
            "complete": True, "bytes": base64.b64encode(PAYLOAD).decode("ascii"),
        }},
    }


CONTRADICTED = b'<sworn r="r1#/loss" k="numeric">the audited loss is 0.</sworn>\n'


def test_an_unknown_rung_turns_a_caught_lie_into_SWORN_HELD():
    """The exploit, stated as a test. The receipt says 4200000 and the sentence says 0."""
    caught = sworn.verify(CONTRADICTED, name="d.md",
                          manifest=sworn.Manifest.from_dict(_manifest("L2")), commit=None)
    assert caught["spans"][0]["verdict"] == "FAILED"
    assert caught["document_verdict"] == "SWORN-FAILED"

    blinded = sworn.verify(CONTRADICTED, name="d.md",
                           manifest=sworn.Manifest.from_dict(_manifest("L3")), commit=None)
    assert blinded["spans"][0]["verdict"] == "UNRESOLVED"
    assert blinded["spans"][0]["reason"] == "rung_unknown"
    # This is the defect, asserted rather than deplored: one string moves the headline.
    assert blinded["document_verdict"] == "SWORN-HELD", (
        "if this now reports something other than SWORN-HELD the ladder has been changed, which is "
        "the operator-gated repair this test's docstring describes — update this test with it")
    assert blinded["counts"]["HELD"] == 0


def test_the_headline_warns_when_nothing_was_checked():
    core = sworn.verify(CONTRADICTED, name="d.md",
                        manifest=sworn.Manifest.from_dict(_manifest("L3")), commit=None)
    line = sworn._headline(core)
    assert "SWORN-HELD" in line
    assert "WARNING" in line, (
        "a document where nothing was checked printed the same headline as one where everything "
        "held, with no warning:\n%s" % line)
    assert "nothing was checked" in line, line


def test_the_headline_warns_when_only_some_spans_resolved():
    """A partial document must warn too, and say how many — otherwise the warning teaches a reader
    that SWORN-HELD is safe whenever it is absent, which is only true at unresolved == 0."""
    # r2 is a WELL-FORMED receipt id that the manifest simply does not carry, so it resolves to
    # UNRESOLVED/manifest_id_missing. An ill-formed id would be MALFORMED instead and make the
    # document SWORN-FAILED, which is not the case under test.
    doc = (b'<sworn r="r1#/loss" k="numeric">the audited loss is 4200000.</sworn>\n'
           b'<sworn r="r2" k="numeric">and the reserve is 5.</sworn>\n')
    core = sworn.verify(doc, name="d.md",
                        manifest=sworn.Manifest.from_dict(_manifest("L2")), commit=None)
    assert core["document_verdict"] == "SWORN-HELD"
    assert core["counts"]["HELD"] == 1 and core["counts"]["UNRESOLVED"] == 1
    line = sworn._headline(core)
    assert "WARNING" in line and "1 of 2" in line, line


def test_the_headline_is_silent_when_everything_resolved():
    """Without this the tests above pass for a verifier that warns on every document, which would
    teach a reader to ignore the warning."""
    doc = b'<sworn r="r1#/loss" k="numeric">the audited loss is 4200000.</sworn>\n'
    core = sworn.verify(doc, name="d.md",
                        manifest=sworn.Manifest.from_dict(_manifest("L2")), commit=None)
    assert core["document_verdict"] == "SWORN-HELD"
    assert core["counts"] == {"HELD": 1, "FAILED": 0, "UNRESOLVED": 0, "MALFORMED": 0,
                              "WITHHELD": 0}
    assert "WARNING" not in sworn._headline(core), sworn._headline(core)


def test_a_failed_document_is_not_warned_about():
    """The warning is about SWORN-HELD hiding unresolved spans. A SWORN-FAILED document is already
    saying the thing the warning exists to say."""
    core = sworn.verify(CONTRADICTED, name="d.md",
                        manifest=sworn.Manifest.from_dict(_manifest("L2")), commit=None)
    assert core["document_verdict"] == "SWORN-FAILED"
    assert "WARNING" not in sworn._headline(core)


def test_the_cli_prints_the_warning(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_bytes(b'<sworn r="r1" k="numeric">the audited loss is 0.</sworn>\n')
    r = subprocess.run([sys.executable, "-m", "styxx.sworn", "verify", str(doc)],
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SWORN-HELD" in r.stdout and "WARNING" in r.stdout, (
        "the CLI is where a reader meets this verdict, and it did not warn:\n%s" % r.stdout)
