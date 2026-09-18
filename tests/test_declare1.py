"""DECLARE-1: the declaration block, and every line of the prereg that says what it must not do.

PREREG_declare1_the_toll_2026_09_18.md, sha256 `7ffd0ba1…`, frozen before `styxx/declare.py`
existed. Most of these tests pin a refusal rather than a capability, because the prereg spends most
of its length on what this must not become.
"""
from __future__ import annotations

import pytest

from styxx.declare import BLOCK_RE, canonical_sentence, declaration_pass, parse_declaration
from styxx.diffgate import gate_diff_text


def diff(*paths: str, body: str = "+def gate_diff_text(a):") -> str:
    return "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n{body}\n" for p in paths)


def block(*lines: str) -> str:
    return "```styxx\n" + "\n".join(lines) + "\n```"


def claims(summary, d):
    return {c.kind: c for c in gate_diff_text(summary, d, run=None, strict=False).claims}


# ---- it reads a declaration ------------------------------------------------------------------

def test_a_declared_count_is_checked():
    c = claims(block("files_changed: 2"), diff("a.py", "b.py"))["files_changed_count"]
    assert c.verdict == "VERIFIED" and c.detail["declared"] is True


def test_a_declared_count_can_be_contradicted():
    c = claims(block("files_changed: 9"), diff("a.py", "b.py"))["files_changed_count"]
    assert c.verdict == "CONTRADICTED" and "claim says 9" in c.why


def test_a_declared_scope_is_checked():
    c = claims(block("only_touches: web/gate"), diff("web/gate/a.py", "src/x.py"))["only_touches"]
    assert c.verdict == "CONTRADICTED" and "src/x.py" in c.why


def test_a_trailing_glob_is_a_prefix_not_a_refusal():
    c = claims(block("only_touches: web/gate/**"), diff("web/gate/a.py"))["only_touches"]
    assert c.verdict == "VERIFIED"


def test_a_declared_symbol_is_checked():
    c = claims(block("adds_symbol: gate_diff_text"), diff("a.py"))["symbol_added"]
    assert c.verdict == "VERIFIED"


# ---- what it must not do ---------------------------------------------------------------------

def test_prose_is_untouched_when_there_is_no_block():
    """The gate must read a body with no declaration exactly as it always did."""
    summary = "This only touches web/gate and changes 2 files."
    assert claims(summary, diff("web/gate/a.py", "web/gate/b.py")).keys()
    for c in gate_diff_text(summary, diff("web/gate/a.py"), run=None, strict=False).claims:
        assert "declared" not in (c.detail or {})


def test_tests_pass_is_unverifiable_even_when_declared():
    """The one field an agent could most easily use to write the verdict it wants."""
    c = claims(block("tests_pass: true"), diff("a.py"))["tests_pass"]
    assert c.verdict == "UNCHECKABLE"
    assert "not evidence that they did" in c.why


@pytest.mark.parametrize("value", ["true", "yes", "all green", "100%"])
def test_no_declared_value_can_make_tests_pass_verified(value):
    c = claims(block(f"tests_pass: {value}"), diff("a.py"))["tests_pass"]
    assert c.verdict == "UNCHECKABLE"


def test_declaring_narrowly_is_visible_and_not_punished():
    """A declaration must never license an accusation about something undeclared."""
    d = diff("web/gate/a.py", "src/elsewhere.py")
    got = claims(block("files_changed: 2"), d)
    assert got["files_changed_count"].verdict == "VERIFIED"
    assert "only_touches" not in got          # nothing was declared about scope, so nothing is said
    assert not any(c.verdict == "CONTRADICTED" for c in got.values())


@pytest.mark.parametrize("line", ["files_changed: many", "adds_symbol: 9lives",
                                  "only_touches: ???", "tests_added: lots"])
def test_a_malformed_value_is_reported_and_never_accused(line):
    """A declaration that cannot be read is not a lie."""
    g = gate_diff_text(block(line), diff("a.py"), run=None, strict=False)
    assert g.verdict == "PASS"
    assert not any(c.verdict == "CONTRADICTED" for c in g.claims)


def test_an_unknown_key_is_reported_and_never_checked():
    g = gate_diff_text(block("ships_on_friday: yes"), diff("a.py"), run=None, strict=False)
    probs = [c for c in g.claims if c.kind == "declaration_problem"]
    assert probs and "unknown key" in probs[0].why
    assert all(c.verdict != "CONTRADICTED" for c in g.claims)


def test_two_blocks_declare_nothing():
    """Merging them would invent a declaration nobody wrote."""
    body = block("files_changed: 1") + "\n\nprose\n\n" + block("files_changed: 2")
    mapping, problems = parse_declaration(body)
    assert mapping is None
    assert problems and "declares once" in problems[0]


def test_only_a_styxx_fence_is_a_declaration():
    for fence in ("```", "```python", "```yaml", "```styxxish"):
        body = f"{fence}\nfiles_changed: 9\n```"
        assert parse_declaration(body)[0] is None, fence


# ---- structural ------------------------------------------------------------------------------

def test_the_synthesized_text_contains_no_fence_so_recursion_terminates():
    text, rep = declaration_pass(block("files_changed: 2", "only_touches: a/b",
                                       "adds_symbol: foo", "tests_added: 3"))
    assert rep["declared"] and text
    assert not BLOCK_RE.search(text)


def test_canonical_sentences_are_what_the_existing_reader_matches():
    """The whole design: a declaration becomes the sentence the prose reader already understands."""
    d = diff("a.py", "b.py")
    for key, value, kind in [("files_changed", "2", "files_changed_count"),
                             ("tests_added", "3", "tests_added"),
                             ("adds_symbol", "gate_diff_text", "symbol_added"),
                             ("only_touches", "a.py", "only_touches"),
                             ("file_touched", "a.py", "file_touched")]:
        sent, why = canonical_sentence(key, value)
        assert sent and why is None, (key, why)
        assert kind in {c.kind for c in gate_diff_text(sent, d, run=None, strict=False).claims}, key


def test_a_declaration_and_the_same_claim_in_prose_agree():
    """They cannot drift apart, because they are read by the same code."""
    d = diff("web/gate/a.py", "src/x.py")
    declared = claims(block("only_touches: web/gate"), d)["only_touches"]
    prose = claims("This only touches web/gate.", d)["only_touches"]
    assert (declared.verdict, declared.why) == (prose.verdict, prose.why)
