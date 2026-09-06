"""Every summary in the verdict core agrees with the span table shipped in that same core.

WHY THIS EXISTS. The core carries five summaries over its spans — `counts`, `sworn_total`,
`unresolved`, `document_verdict` and `rungs` — and every reader acts on those, not on the span table
underneath. Inside `verify` they are tallied from the internal `verdicts` list, and `spans` is set
from that same list; nothing states that they must stay the same list. A refactor that filtered,
re-ordered or re-adjudicated one of the two would produce a core whose headline described a span
table it no longer shipped, and every existing test would still pass.

WHAT THIS CHECKS, EXACTLY. That `core[field]` equals the same field recomputed from `core["spans"]`.
That is a narrower claim than it may look:

  - It DOES catch the summaries and the emitted spans being derived from different lists, a span
    dropped or added between tally and emission, and an edit to one derivation rule and not the
    other.
  - It does NOT independently verify the adjudication rules. `_recompute` below restates the rules
    in `verify`; if both are wrong in the same way, this passes. It is a change-detector over the
    summary logic and an agreement check between the two halves of the core — not a proof that
    either half is right.
  - It says nothing about whether any individual span's verdict is correct. A verifier that
    adjudicated every span wrongly and summarised those wrong spans faithfully would pass.

It runs over the differential harness's seeded grammar rather than hand-written documents, so the
inputs are ones nobody chose — the corpus that found two real defects in the JavaScript verifier.

PROVENANCE. Written after a hypothesis failed. Several findings in the sidecar-battery leg looked
like "a summary overselling its details", so the shape was tested as a property over 4000 generated
documents. Not one diverged: the hypothesis was wrong, and the class of defect it predicted does not
exist inside the core. The property is kept because nothing else pins it, not because it caught
anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402
from styxx.sworn import VERDICTS  # noqa: E402

try:
    import conformance.sworn.differential as D
except Exception as exc:                                        # noqa: BLE001
    D = None
    _WHY = str(exc)

SEED = 20260906
CASES = 1500          # the probe that produced the negative result ran 4000; this is the standing size


def _recompute(core: dict) -> dict:
    """Each summary, derived from `core["spans"]` alone, restating the rules in `verify`."""
    spans = core["spans"]
    counts = {v: 0 for v in VERDICTS}
    for s in spans:
        counts[s["verdict"]] += 1
    sworn_total = sum(counts.values())

    if core["document_malformed"]:
        verdict = "SWORN-FAILED"
    elif sworn_total == 0:
        verdict = "UNSWORN"
    elif counts["FAILED"] == 0 and counts["MALFORMED"] == 0:
        verdict = "SWORN-HELD"
    else:
        verdict = "SWORN-FAILED"

    rungs: dict = {}
    for s in spans:
        prov = s.get("provenance") or {}
        if prov.get("form") == "rn":
            key = prov.get("rung")
        elif prov.get("form") in ("path", "prereg"):
            key = "committed"
        else:
            key = "unresolved"
        rungs[key] = rungs.get(key, 0) + 1

    return {"counts": counts, "sworn_total": sworn_total, "unresolved": counts["UNRESOLVED"],
            "document_verdict": verdict, "rungs": rungs}


def _cores():
    """Verified cores over the seeded grammar. Cases the generator makes unverifiable are skipped —
    a document the verifier refuses has no summaries to check."""
    if D is None:
        pytest.skip("the differential generator is not importable here: %s" % _WHY)
    made = 0
    for i in range(CASES):
        c = D.case(SEED, i)
        try:
            man = sworn.Manifest.from_dict(c["manifest"]) if c["manifest"] is not None else None
        except BaseException:                                   # noqa: BLE001
            continue
        try:
            core = sworn.verify(c["document"], name=c["name"], manifest=man, commit=c["commit"])
        except BaseException:                                   # noqa: BLE001
            continue
        made += 1
        yield i, core
    if made == 0:
        pytest.fail("no document verified at all — this test would pass vacuously")


@pytest.mark.parametrize("field", ["counts", "sworn_total", "unresolved", "document_verdict",
                                   "rungs"])
def test_the_summary_agrees_with_the_spans_it_ships_with(field):
    checked = 0
    for i, core in _cores():
        checked += 1
        want = _recompute(core)[field]
        got = core.get(field)
        assert got == want, (
            "case %d: core[%r] is %r but the span table shipped in the same core implies %r — the "
            "headline describes spans the reader is not given" % (i, field, got, want))
    assert checked > CASES // 2, (
        "only %d of %d generated documents verified; the corpus this pins has shrunk" % (checked,
                                                                                          CASES))


def test_the_check_is_not_vacuous():
    """The grammar must reach more than one document verdict and a non-empty rungs table, or the
    parametrised tests above are pinning a single shape."""
    verdicts, rung_keys, any_spans = set(), set(), 0
    for _i, core in _cores():
        verdicts.add(core["document_verdict"])
        rung_keys |= set(core.get("rungs") or {})
        any_spans += len(core["spans"])
    assert len(verdicts) >= 2, "only reached %r" % (verdicts,)
    assert rung_keys, "no span carried provenance, so the rungs summary was never exercised"
    assert any_spans > 0
