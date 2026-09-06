"""The Action discovers documents with the lexer's own test, not a case-sensitive copy of it.

The Action decided what to verify with `b"<sworn" in data`. The verifier's candidate lexer is
case-insensitive by construction -- `_CANDIDATE = rb"<(/?)[sS][wW][oO][rR][nN](?!...)"` -- and
DECISIONS["tag_grammar"] says a non-lowercase tag-shaped candidate is "MALFORMED, never narrative".
So the two disagreed about what a document even is:

    document: <SWORN r="r1" k="numeric">the rate was 0.42 on the panel</SWORN>
      the Action's test  b'<sworn' in data : False   -> skipped, "carries no <sworn tag"
      the lexer's test   _CANDIDATE.search : True
      the verifier's verdict               : SWORN-FAILED, two MALFORMED/tag_syntax

CI reported "carries no <sworn tag" for a document the verifier calls SWORN-FAILED. The skip is
silent by design -- a skipped document is not a failure -- so an uppercase tag was a way to put a
tag-shaped candidate in a pull request and have the gate say nothing.

Not a false HELD: the Action never claimed the document held. It claimed the document has no tags,
which is false, and it is the claim a reviewer reads.

Spec: papers/sworn/SPEC_action_finds_what_the_lexer_finds_v01_2026_09_06.md (C1).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.sworn import _CANDIDATE, verify  # noqa: E402

ACTION = ROOT / "sworn" / "sworn_action.py"

SPELLINGS = [
    (b'<sworn r="r1" k="numeric">the rate was 0.42</sworn>', "lowercase, the ordinary case"),
    (b'<SWORN r="r1" k="numeric">the rate was 0.42</SWORN>', "uppercase"),
    (b'<Sworn r="r1" k="numeric">the rate was 0.42</Sworn>', "title case"),
    (b'<sWoRn r="r1" k="numeric">the rate was 0.42</sWoRn>', "mixed case"),
    (b'</SWORN>', "an uppercase closer alone"),
]
NO_TAG = [
    (b"a paragraph about swearing, with no tag in it at all\n", "prose containing the word"),
    (b"<swornish>not a tag</swornish>\n", "a longer name: the lexer's own negative lookahead"),
    (b"nothing here\n", "plain prose"),
]


def _action():
    if not ACTION.exists():
        pytest.skip("the action is not present in this checkout")
    spec = importlib.util.spec_from_file_location("sworn_action_under_test", ACTION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _action_finds(mod, data: bytes) -> bool:
    """The Action's discovery predicate, read out of the shipped source.

    Lifted rather than restated: a test that spells the predicate itself passes when the Action and
    the test drift together, which is the whole defect here in miniature.
    """
    src = ACTION.read_text(encoding="utf-8")
    if "_CANDIDATE" in src:
        return _CANDIDATE.search(data) is not None
    return b"<sworn" in data                      # the pre-repair predicate


@pytest.mark.parametrize("doc,label", SPELLINGS, ids=[l for _d, l in SPELLINGS])
def test_the_action_finds_every_tag_shaped_candidate_the_lexer_finds(doc, label):
    """C-G1/C-G4, the guard that must be seen red."""
    mod = _action()
    assert _action_finds(mod, doc) is True, (
        "%s: the lexer sees a candidate here and the Action does not, so CI reports 'carries no "
        "<sworn tag' for a document the verifier would adjudicate" % label)


@pytest.mark.parametrize("doc,label", NO_TAG, ids=[l for _d, l in NO_TAG])
def test_a_document_with_no_candidate_is_still_skipped(doc, label):
    """C-G3. The repair must not start verifying prose."""
    mod = _action()
    assert _action_finds(mod, doc) is False, label
    assert _CANDIDATE.search(doc) is None, "%s: the lexer must agree it is not a candidate" % label


@pytest.mark.parametrize("doc,label", SPELLINGS[1:], ids=[l for _d, l in SPELLINGS[1:]])
def test_what_the_verifier_says_about_the_documents_the_action_was_skipping(doc, label):
    """The stakes: these are not harmless, they are SWORN-FAILED."""
    core = verify(doc, name="D.md", manifest=None, tree=None)
    assert core["document_verdict"] == "SWORN-FAILED", (label, core["document_verdict"])
    assert all(s["verdict"] == "MALFORMED" for s in core["spans"]), core["spans"]


def test_the_skip_messages_are_unchanged():
    """C-G5's half that needs no subprocess: the samples pin these strings byte for byte.

    `sworn_action_sample.py --check` compares the committed sample byte for byte, and a sample is
    history under that script's own rule -- changing the wording would need a new prefix at a new
    commit. After C1 the messages stay accurate, because they are only emitted for a document
    carrying no candidate in any case.
    """
    src = ACTION.read_text(encoding="utf-8")
    assert '"why": "carries no <sworn tag"' in src
    assert '"why": "the body carries no <sworn tag"' in src


def test_the_action_imports_the_predicate_rather_than_spelling_it():
    """C1's mechanism, not just its effect.

    A second spelling of "what a tag looks like" is the drift that the U+0085 path-segment defect
    was, one repair earlier in this same audit. The Action must ask the lexer.
    """
    src = ACTION.read_text(encoding="utf-8")
    assert "_CANDIDATE" in src, "the Action does not import the lexer's candidate pattern"
    # Only CODE. The first version of this assertion grepped the whole file and failed on the
    # comment that explains the defect, which quotes the old predicate on purpose — a guard that
    # polices prose rather than behaviour, which is the same mistake in miniature.
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    for dead in ('"<sworn" in body', 'b"<sworn" not in data'):
        assert dead not in code, "the case-sensitive predicate %r is still live" % dead
