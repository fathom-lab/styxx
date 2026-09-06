"""An embedded manifest is not discarded because its receipts map happens to be empty.

`verify()` gated the sidecar's embedded manifest on the TRUTHINESS of its receipts map
(`if emb.get("receipts"):`), and `{}` is falsy. An empty receipts map therefore threw away the whole
manifest, `authored_sha256` included -- the list invariant 2 consults. The agent empties its receipts
map and its own committed bytes stop being refused:

    agent's own bytes, declared in authored_sha256:
      receipts {r1}   -> SWORN-FAILED  MALFORMED/receipt_author_minted
      receipts EMPTY  -> SWORN-HELD    HELD

Invariant 2 -- the agent cannot swear to bytes it minted -- is the format's central rule, and PR #72
extended it to the tree channel so it could not be evaded by choosing a receipt form. It was evaded
by choosing an empty dict.

WHAT THIS DELIBERATELY DOES NOT DO. The gate asks "does it have receipts?" where it means "is there
a manifest?". Adopting `if emb:` would be the honest rule and would set `manifest_digest` from None
to a digest on the 34 of 43 committed sidecars that carry `receipts: {}` -- the ordinary shape for a
document swearing entirely through the tree channel. That is inside the digested core: 34 committed
receipts would stop re-deriving. E-G3 below pins those 34 as unchanged, and the broader question is
left for the operator.

Spec: papers/sworn/SPEC_embedded_manifest_is_not_dropped_v01_2026_09_06.md (E1).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

AGENT_BYTES = b"the precision is 0.9900\n"
AGENT_SHA = hashlib.sha256(AGENT_BYTES).hexdigest()
COMMIT = "a" * 40
DOC = ('<sworn r="path:out.txt" k="quote">the run reports `the precision is 0.9900` here</sworn>\n'
       ).encode("utf-8")


def _tree():
    return sworn.MemoryTree({"out.txt": AGENT_BYTES}, commit=COMMIT)


def _sidecar(receipts_empty: bool, authored):
    m = sworn.Manifest(harness="ci", turn="t1", rung="L2")
    if not receipts_empty:
        m.add("r1", b"an unrelated receipt", "tool_stdout", complete=True)
    for a in authored:
        m.authored_sha256.append(a)
    side = sworn.to_sidecar(DOC, "d.md", commit=COMMIT)
    side["manifest"] = m.to_dict()
    return side


def _verify(receipts_empty, authored, manifest=None):
    return sworn.verify(sidecar=_sidecar(receipts_empty, authored), tree=_tree(), manifest=manifest)


def test_an_empty_receipts_map_does_not_disable_invariant_two():
    """E-G1, the guard that must be seen red."""
    core = _verify(receipts_empty=True, authored=[AGENT_SHA])
    span = core["spans"][0]
    assert span["verdict"] == "MALFORMED", (
        "the agent declared these bytes in authored_sha256 and swore to them anyway; with an empty "
        "receipts map the manifest was discarded and the span came back %s/%s"
        % (span["verdict"], span.get("reason")))
    assert span["reason"] == "receipt_author_minted", span
    assert core["document_verdict"] == "SWORN-FAILED"


def test_the_same_document_with_receipts_is_refused_too():
    """E-G2, the control: the refusal is not new, it was merely reachable around."""
    core = _verify(receipts_empty=False, authored=[AGENT_SHA])
    assert core["spans"][0]["reason"] == "receipt_author_minted", core["spans"][0]
    assert core["document_verdict"] == "SWORN-FAILED"


def test_a_sidecar_with_nothing_authored_is_unchanged():
    """E-G3, the guard that must NEVER go red.

    34 of the 43 committed sidecars carry `receipts: {}` with an empty `authored_sha256`. Honouring
    their manifests would move `manifest_digest` from None to a digest, inside the digested core,
    and 34 committed receipts would stop re-deriving. E1 leaves them alone by construction.
    """
    core = _verify(receipts_empty=True, authored=[])
    assert core["manifest_digest"] is None, (
        "a sidecar with an empty receipts map AND an empty authored_sha256 must still resolve with "
        "no manifest, or the 34 committed sidecars shaped like it stop re-deriving")
    assert core["spans"][0]["verdict"] == "HELD"
    assert core["document_verdict"] == "SWORN-HELD"


def test_a_supplied_manifest_that_disagrees_is_still_refused():
    """E-G4. The dropped gate also dropped the disagreement check."""
    other = sworn.Manifest(harness="other", turn="t9", rung="L1")
    other.add("r1", b"something else", "tool_stdout", complete=True)
    with pytest.raises(SystemExit) as ei:
        _verify(receipts_empty=True, authored=[AGENT_SHA], manifest=other)
    assert "disagrees with the embedded one" in str(ei.value), str(ei.value)


def test_the_committed_sidecars_shaped_like_this_still_verify():
    """The 34, asserted as a population rather than as a claim in a docstring."""
    import json
    import subprocess
    empty_both = 0
    for f in subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.sworn.json"],
                            capture_output=True, text=True).stdout.split():
        try:
            d = json.loads((ROOT / f).read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            continue
        m = d.get("manifest")
        if isinstance(m, dict) and not m.get("receipts") and not m.get("authored_sha256"):
            empty_both += 1
    assert empty_both >= 25, (
        "expected the committed corpus to still hold the population this rule protects; found %d"
        % empty_both)
