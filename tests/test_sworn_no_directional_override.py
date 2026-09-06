"""A sworn span carries no directional override, so what is checked is what is rendered.

U+202E RIGHT-TO-LEFT OVERRIDE inside a numeric span was accepted silently. `_TOKEN` does not match
category Cf, so the control split tokens rather than joining them, `_number_token` still found one
digit-bearing token, and the span was adjudicated on the LOGICAL order while a UAX #9 renderer
displays the run reversed:

    <sworn r="r1" k="numeric">the rate was <U+202E>0.55<U+202C> on the panel</sworn>
      verdict SWORN-HELD, detail printed_token '0.55'
      Chrome 148, visual left-to-right glyph order: "55.0"

The reader sees 55.0 under a SWORN-HELD badge; the verifier checked 0.55. The field is literally
named `printed_token` and does not hold what is printed.

This applies a decision the lab has already made twice rather than making a new one: R2
`hidden_commitment` pays a MALFORMED rule for the same verify/render divergence, and capsule.py
already sanitises this exact code-point range in its viewer.

Spec: papers/sworn/SPEC_no_directional_override_in_a_span_v01_2026_09_06.md (D1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

RLO, LRO, PDF = "‮", "‭", "‬"
LRE, RLE = "‪", "‫"
LRI, RLI, FSI, PDI = "⁦", "⁧", "⁨", "⁩"

OVERRIDES = [(RLO, "U+202E RLO"), (LRO, "U+202D LRO")]
LEGAL = [(LRE, "U+202A LRE"), (RLE, "U+202B RLE"), (PDF, "U+202C PDF"),
         (LRI, "U+2066 LRI"), (RLI, "U+2067 RLI"), (FSI, "U+2068 FSI"), (PDI, "U+2069 PDI")]


def _manifest(value=b"0.55"):
    m = sworn.Manifest(harness="ci", turn="t1", rung="L2")
    m.add("r1", value, "tool_stdout", complete=True)
    return m


def _numeric(control: str) -> bytes:
    return ('<sworn r="r1" k="numeric">the held-out error rate was %s0.55%s on the panel</sworn>\n'
            % (control, PDF if control else "")).encode("utf-8")


def _quote(control: str) -> bytes:
    return ('<sworn r="r1" k="quote">the log records %s`the value is 0.55 exactly`%s here</sworn>\n'
            % (control, PDF if control else "")).encode("utf-8")


def _verify(doc: bytes, manifest=None, tree=None):
    core = sworn.verify(doc, name="D.md", manifest=manifest, tree=tree)
    return core, core["spans"][0]


@pytest.mark.parametrize("control,name", OVERRIDES, ids=[n for _c, n in OVERRIDES])
def test_a_directional_override_in_a_numeric_span_is_refused(control, name):
    """D-G1/D-G2, the guard that must be seen red."""
    core, span = _verify(_numeric(control), manifest=_manifest())
    assert span["verdict"] == "MALFORMED", (
        "%s wrapped the number and the span was %s with printed_token %r — the verifier read the "
        "logical order while a renderer shows it reversed"
        % (name, span["verdict"], (span.get("detail") or {}).get("printed_token")))
    assert span["reason"] == "directional_override", span
    assert core["document_verdict"] == "SWORN-FAILED"


@pytest.mark.parametrize("control,name", OVERRIDES, ids=[n for _c, n in OVERRIDES])
def test_a_directional_override_in_a_quote_span_is_refused(control, name):
    """D-G2. The deception is not specific to numbers; a needle reverses too."""
    _core, span = _verify(_quote(control), manifest=_manifest(b"the value is 0.55 exactly"))
    assert span["verdict"] == "MALFORMED" and span["reason"] == "directional_override", span


@pytest.mark.parametrize("control,name", LEGAL, ids=[n for _c, n in LEGAL])
def test_isolates_and_embeddings_stay_legal(control, name):
    """D-G3, the guard that must NEVER go red.

    U+2066-U+2069 are the Unicode-recommended way to embed a Latin or numeric run inside Arabic or
    Hebrew text, and none of these reorders a digit run: the audit's skeptic measured all of them
    rendering `0.55` unchanged in Chrome 148. Refusing them would penalise correct mixed-direction
    authoring to defend against an attack they cannot carry.
    """
    _core, span = _verify(_numeric(control), manifest=_manifest())
    assert span["reason"] != "directional_override", (
        "%s is not an override and must stay legal: refusing it costs RTL authors the recommended "
        "way to write a number" % name)


def test_a_span_with_no_control_is_untouched():
    """D-G4."""
    _core, held = _verify(_numeric(""), manifest=_manifest())
    assert held["verdict"] == "HELD", held
    _core2, failed = _verify(_numeric(""), manifest=_manifest(b"0.99"))
    assert failed["verdict"] == "FAILED", failed


@pytest.mark.parametrize("control,name", OVERRIDES, ids=[n for _c, n in OVERRIDES])
def test_the_refusal_is_decidable_from_the_document_bytes_alone(control, name):
    """D-G5. A MALFORMED must never depend on evidence the verifier might not have."""
    core, span = _verify(_numeric(control), manifest=None, tree=None)
    assert span["verdict"] == "MALFORMED" and span["reason"] == "directional_override", span
    assert core["document_verdict"] == "SWORN-FAILED"


def test_the_reason_is_in_the_closed_vocabulary():
    """REASONS is a closed set a consumer keys on; a reason outside it would break them."""
    assert "directional_override" in sworn.REASONS


def test_the_javascript_verifier_agrees():
    """D-G6, the parity gate, by core digest through the differential harness's node runner."""
    from shutil import which
    if which("node") is None:
        pytest.skip("node is not available")
    try:
        import conformance.sworn.differential as D
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip("the differential harness is not importable here: %s" % exc)

    docs = ([_numeric(c) for c, _n in OVERRIDES] + [_quote(c) for c, _n in OVERRIDES]
            + [_numeric(c) for c, _n in LEGAL] + [_numeric("")])
    batch = [{"index": i, "document": d, "manifest": _manifest().to_dict(),
              "name": "D.md", "commit": None} for i, d in enumerate(docs)]

    import tempfile
    rows = D.js_digests(batch, Path(tempfile.mkdtemp()))
    bad = []
    for i in range(len(docs)):
        py, py_err, _c = D.python_digest(batch[i])
        js = rows.get(i) or {}
        if py != js.get("digest"):
            bad.append("case %d python=%s(%s) node=%s(%s)"
                       % (i, (py or "-")[:12], py_err or "ok",
                          (js.get("digest") or "-")[:12], js.get("error") or "ok"))
    assert not bad, "the two implementations disagree:\n  " + "\n  ".join(bad)
