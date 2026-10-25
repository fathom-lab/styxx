# -*- coding: utf-8 -*-
"""prereg-ci B3: the RESULT readers (a ```result fence, a sworn span on a receipt's verdict leaf),
with the gates scanner's attack cases.

DESIGN 3.2(c): "A sworn span on the receipt's verdict leaf (as RESULT_pmnist_robustness_2026_10_23.md:4
does), or one ```result fence naming prereg, receipt and verdict, scanned with the gates fence's
rules. Both must agree when both are present." DESIGN 3.2(a) scans the ```locks fence "with the same
rules as the gates fence". PLAN K4: ``styxx.protocol._select_gates_block`` hard-codes its info
string and styxx/protocol.py does not change, so core holds the scanner with the info string as a
parameter (``core._select_fence``), and this file shows it decides as ``_select_gates_block`` does on
every gates vector and on every PREREG at REV. PLAN K11: C1 also pairs every receipt named by a
RESULT the pull request adds or changes.

The attack classes are the ones ``tests/test_protocol_v4.py`` pins for the gates fence: a decoy in an
HTML comment beside the visible fence, an unterminated comment, a tilde fence, a case variant, a
confusable (non-ASCII) info string, a tab-indented opener, two fences, an unclosed fence, and
duplicate and non-ASCII keys. Each refuses for ```result and ```locks. Beyond them, for those two
info strings only, a fence CommonMark renders and the scanner does not count (behind a byte-order
mark, in a block quote, in a list item, an info string with a second word) is refused: the B0
conformance README left that class to this step. The gates fence keeps the scanner's reading, which
sealed PREREGs depend on.

Every repository built here is a ``git init`` in tmp_path with no remote; the history tests read
styxx's own objects at REV and skip, saying why, where REV is not in the object store.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "conformance" / "prereg_ci"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci import REV  # noqa: E402
from conformance.prereg_ci.replay import blob_bytes  # noqa: E402
from styxx import protocol  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from styxx.prereg_ci import git as G  # noqa: E402
from tests.test_prereg_ci_git import Git, prereg  # noqa: E402

GateSpecError = protocol.GateSpecError
F = "```"
REFUSES, DIFFER, NOT_READ = "RESULT_REFUSES", "RESULT_FORMS_DIFFER", "RESULT_NOT_READ"
PREREG_DIFFERS, NO_RECEIPT = "RESULT_PREREG_DIFFERS", "RESULT_NAMES_NO_RECEIPT"

GATES_BODY = {"gates": {"G": {"metric": "m", "op": "<=", "value": 0.2}},
              "outcomes": [{"when": {"G": True}, "verdict": "PASS"}, {"when": {"G": False}, "verdict": "FAIL"}],
              "smoke_verdict": "SMOKE"}
BODIES = {
    "gates": GATES_BODY,
    "result": {"prereg": "PREREG_a.md", "receipt": "receipt.json", "verdict": "PASS"},
    "locks": {"scorer": "score.py", "reads": ["data.csv"], "outcomes": ["PASS", "FAIL"], "withdraws": {}},
}
# one letter of each info string swapped for a Cyrillic look-alike (U+0430, U+0435, U+043E)
CONFUSABLE = {"gates": "g\u0430tes", "result": "r\u0435sult", "locks": "l\u043ecks"}
INFOS = tuple(BODIES)


def honest(info: str) -> str:
    return json.dumps(BODIES[info])


def decoy(info: str) -> str:
    """The same block with one value moved: what a decoy fence would make the machine read."""
    body = json.loads(json.dumps(BODIES[info]))
    if info == "gates":
        body["gates"]["G"]["value"] = 99
    elif info == "result":
        body["verdict"] = "FAIL"
    else:
        body["scorer"] = "other.py"
    return json.dumps(body)


def attacks(info: str) -> dict:
    """The attack documents of tests/test_protocol_v4.py, written for *info*: name -> (text, the
    words the refusal holds)."""
    h, d = honest(info), decoy(info)
    return {
        "decoy-in-comment": (f"# d\n\n<!--\n{F}{info}\n{d}\n{F}\n-->\n\n{F}{info}\n{h}\n{F}\n", "2 " + info + "-like"),
        "tilde-fence": (f"# d\n\n<!--\n{F}{info}\n{d}\n{F}\n-->\n\n~~~{info}\n{h}\n~~~\n", info + "-like"),
        "case-variant": (f"# d\n\n{F}{info.upper()}\n{h}\n{F}\n", "not a plain unindented"),
        "fence-only-in-comment": (f"# d\n\n<!--\n{F}{info}\n{h}\n{F}\n-->\n", "HTML comment"),
        "confusable-info": (f"# d\n\n{F}{CONFUSABLE[info]}\n{h}\n{F}\n\n{F}{info}\n{d}\n{F}\n", "non-ASCII"),
        "tab-indented-opener": (f"# d\n\n<!--\n{F}{info}\n{h}\n{F}\n-->\n\n\t{F}{info}\n{d}\n{F}\n", "HTML comment"),
        "unterminated-comment": (f"# d\n\n<!--\n\n{F}{info}\n{h}\n{F}\n", "HTML comment"),
        "unclosed-fence": (f"# d\n\n{F}{info}\n{h}\n", "never closed"),
        "two-fences": (f"# d\n\n{F}{info}\n{h}\n{F}\n\n{F}{info}\n{d}\n{F}\n", "2 " + info + "-like"),
        "indented-opener": (f"# d\n\n  {F}{info}\n{h}\n{F}\n", "not a plain unindented"),
    }


def key_attacks(info: str) -> dict:
    """Duplicate and non-ASCII keys (test_protocol_v4 D2 and F2), in the one plain fence."""
    h = honest(info)
    lead = h[1:h.index(":")]                          # the block's leading key, quoted
    dup = h.replace(lead + ":", lead + ": 1, " + lead + ":", 1)
    homoglyph = h.replace(lead + ":", lead.replace(lead[1], "\u0441", 1) + ": 1, " + lead + ":", 1)
    return {"duplicate-key": (f"# d\n\n{F}{info}\n{dup}\n{F}\n", "duplicate key"),
            "non-ascii-key": (f"# d\n\n{F}{info}\n{homoglyph}\n{F}\n", "non-ASCII key")}


def shadows(info: str) -> dict:
    """Fences CommonMark renders as an <info> block that the scanner does not count."""
    h = honest(info)
    block = f"{F}{info}\n{h}\n{F}\n"
    return {
        "bom-on-line-1": "\ufeff" + block,
        "block-quote": "# d\n\n" + "".join("> " + ln + "\n" for ln in block.splitlines()),
        "list-item": "# d\n\n" + "".join(("- " if i == 0 else "  ") + ln + "\n"
                                         for i, ln in enumerate(block.splitlines())),
        "ordered-list-item": "# d\n\n" + "".join(("1. " if i == 0 else "   ") + ln + "\n"
                                                 for i, ln in enumerate(block.splitlines())),
        "quote-in-list": "# d\n\n" + "".join(("- > " if i == 0 else "  > ") + ln + "\n"
                                             for i, ln in enumerate(block.splitlines())),
        "second-word": f"# d\n\n{F}{info} json\n{h}\n{F}\n",
    }


def protocol_reading(text: str):
    """What styxx.protocol._select_gates_block decides: the block, None for "no gates block", or the
    refusal's text."""
    try:
        return ("block", protocol._select_gates_block(text))
    except GateSpecError as e:
        if str(e).startswith("prereg has no ```gates block"):
            return ("none", None)
        return ("raises", str(e))


def fence_reading(text: str, info: str):
    try:
        got = core._select_fence(text, info)
    except GateSpecError as e:
        return ("raises", str(e))
    return ("none", None) if got is None else ("block", got)


# ---- the parameterised scanner decides as _select_gates_block does ------------------------------------

def _gates_vector_texts():
    out = {}
    for name in ("vectors_score.json", "vectors_compare.json"):
        doc = json.loads((SET / name).read_text(encoding="utf-8"))
        blobs = doc.get("blobs") or json.loads((SET / "blobs.json").read_text(encoding="utf-8"))["blobs"]
        for v in doc["vectors"]:
            out[v["inputs"]["prereg"]] = blob_bytes(blobs[v["inputs"]["prereg"]])
    fence = json.loads((SET / "vectors_fence.json").read_text(encoding="utf-8"))
    for v in fence["vectors"]:
        if v["info"] == "gates":
            out[v["inputs"]["text"]] = blob_bytes(fence["blobs"][v["inputs"]["text"]])
    return out


def test_the_scanner_decides_as_select_gates_block_on_every_gates_vector():
    texts = _gates_vector_texts()
    assert len(texts) >= 60
    seen, undecodable = set(), []
    for key, data in sorted(texts.items()):
        try:
            raw = data.decode("utf-8")
        except UnicodeDecodeError:
            undecodable.append(key)                  # bytes no scanner reads (rescore refuses them before it scans)
            continue
        for text in (raw, core._decision_text(data)):
            want, got = protocol_reading(text), fence_reading(text, "gates")
            assert got == want, key
            seen.add(want[0])
    assert seen == {"block", "none", "raises"}
    assert len(undecodable) <= 2, undecodable
    print(f"gates vectors: {len(texts)} PREREG texts, {len(texts) - len(undecodable)} decoded and read alike by "
          f"both scanners, {len(undecodable)} not UTF-8")


@pytest.mark.parametrize("name", sorted(attacks("gates")) + sorted(key_attacks("gates")))
def test_every_gates_attack_case_decides_as_select_gates_block(name):
    text = {**attacks("gates"), **key_attacks("gates")}[name][0]
    want = protocol_reading(text)
    assert fence_reading(text, "gates") == want
    if name in attacks("gates"):
        assert want[0] == "raises"                       # the scanner itself refuses each of these


@pytest.mark.parametrize("info", ("result", "locks"))
@pytest.mark.parametrize("name", sorted(attacks("gates")))
def test_every_attack_case_refuses_for_result_and_locks(info, name):
    text, words = attacks(info)[name]
    with pytest.raises(GateSpecError, match=words):
        core._select_fence(text, info)
    with pytest.raises(GateSpecError, match=words):
        core._fence_json(text, info)


@pytest.mark.parametrize("info", ("result", "locks"))
@pytest.mark.parametrize("name", sorted(key_attacks("gates")))
def test_duplicate_and_non_ascii_keys_refuse_for_result_and_locks(info, name):
    text, words = key_attacks(info)[name]
    assert core._select_fence(text, info) is not None           # the scanner selects the block,
    with pytest.raises(GateSpecError, match=words):              # and the strict reader refuses it
        core._fence_json(text, info)


@pytest.mark.parametrize("name", sorted(attacks("gates")) + sorted(key_attacks("gates")))
def test_every_attack_case_refuses_a_result_document(name):
    text = {**attacks("result"), **key_attacks("result")}[name][0]
    got = core._result_reading(text.encode("utf-8"))
    assert got["code"] == REFUSES and got["kind"] == "FAIL", got
    assert got["declarations"] == []


@pytest.mark.parametrize("info", ("result", "locks"))
@pytest.mark.parametrize("name", sorted(shadows("gates")))
def test_a_fence_a_renderer_shows_and_the_scanner_does_not_count_refuses_for_result_and_locks(info, name):
    text = shadows(info)[name]
    assert core._select_fence(text, info) is None                # the scanner counts no fence here
    with pytest.raises(GateSpecError, match="renders"):
        core._fence_json(text, info)
    if info == "result":
        assert core._result_reading(text.encode("utf-8"))["code"] == REFUSES


@pytest.mark.parametrize("name", sorted(shadows("gates")))
def test_the_gates_fence_keeps_the_scanners_reading_of_those_fences(name):
    text = shadows("gates")[name]
    assert protocol_reading(text) == ("none", None) == fence_reading(text, "gates")


def test_a_shadow_beside_the_plain_fence_refuses_too():
    text = f"# d\n\n{F}result\n{honest('result')}\n{F}\n\n> {F}result\n> {decoy('result')}\n> {F}\n"
    assert core._select_fence(text, "result") == honest("result")
    with pytest.raises(GateSpecError, match="renders"):
        core._fence_json(text, "result")


def test_an_indented_code_block_is_not_a_shadow():
    """Four columns of indent is code a renderer prints as text, backticks included: no block."""
    text = f"# d\n\n    {F}result\n    {honest('result')}\n    {F}\n"
    assert core._select_fence(text, "result") is None
    assert core._fence_json(text, "result") is None
    assert core._result_reading(text.encode("utf-8"))["declarations"] == []


def test_the_info_string_is_one_lower_case_ascii_word():
    for bad in ("", "Result", "res ult", "r\u0435sult", 5, None):
        with pytest.raises((ValueError, TypeError)):
            core._select_fence("x", bad)


def test_the_scanner_decides_as_select_gates_block_on_every_prereg_at_rev():
    """Every PREREG_*.md in the tree at REV, papers/ and the rest; R8 counted papers/ alone."""
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", REV + "^{commit}"], capture_output=True).returncode:
        pytest.skip(f"{REV} is not in the object store; the PREREGs are read from it")
    with G._Repo(ROOT) as repo:
        paths = [p for p in repo.paths(REV) if p.rsplit("/", 1)[-1].startswith("PREREG_") and p.endswith(".md")]
        kinds = {"block": 0, "none": 0, "raises": 0}
        gated_in_papers = 0
        for p in paths:
            text = core._decision_text(repo.read(REV, p))
            want = protocol_reading(text)
            assert fence_reading(text, "gates") == want, p
            kinds[want[0]] += 1
            gated_in_papers += want[0] == "block" and p.startswith("papers/")
    assert sum(p.startswith("papers/") for p in paths) == 412      # R8: 412 PREREG_*.md under papers/
    assert gated_in_papers == 52                                    # R8: 52 of them gated
    print(f"PREREG_*.md at {REV[:12]}: {len(paths)} in the tree; the two scanners agree on every one; {kinds}")


# ---- the ```result fence -------------------------------------------------------------------------------

def result_doc(body, extra: str = "") -> bytes:
    text = body if isinstance(body, str) else json.dumps(body)
    return f"# RESULT\n\nThe verdict is the fence's.\n\n{F}result\n{text}\n{F}\n{extra}".encode("utf-8")


def test_a_result_fence_names_prereg_receipt_and_verdict():
    got = core._result_reading(result_doc(BODIES["result"]))
    assert got["code"] is None and got["unread"] == []
    assert got["declarations"] == [{"form": "fence", "receipt": "receipt.json", "prereg": "PREREG_a.md",
                                    "verdict": "PASS", "line": 5}]


@pytest.mark.parametrize("body, words", [
    ({"receipt": "receipt.json", "verdict": "PASS"}, "prereg"),
    ({**BODIES["result"], "gates": {"G": True}}, "gates"),
    ({**BODIES["result"], "verdict": 5}, "verdict"),
    ({**BODIES["result"], "verdict": ""}, "verdict"),
    ({**BODIES["result"], "receipt": None}, "receipt"),
    ([BODIES["result"]], "object"),
    ('{"prereg": "PREREG_a.md", "receipt": "receipt.json", "verdict": "PASS",}', "strict JSON"),
])
def test_a_result_fence_holds_those_three_strings_and_nothing_else(body, words):
    got = core._result_reading(result_doc(body))
    assert got["code"] == REFUSES and words in got["detail"], got
    assert got["declarations"] == []


def test_a_document_with_no_result_form_declares_nothing():
    got = core._result_reading(b"# notes\n\nThe verdict was PASS.\n\n```python\nprint(1)\n```\n")
    assert got == {"code": None, "kind": None, "detail": None, "declarations": [], "unread": []}


def test_a_crlf_result_reads_as_its_lf_bytes():
    lf = result_doc(BODIES["result"])
    crlf = lf.replace(b"\n", b"\r\n")
    assert core._result_reading(crlf)["declarations"] == core._result_reading(lf)["declarations"]


def test_a_result_that_is_not_utf8_refuses_when_it_holds_a_form():
    assert core._result_reading(b"\xff" + result_doc(BODIES["result"]))["code"] == REFUSES
    assert core._result_reading(b"\xff# notes\n")["code"] is None


# ---- the sworn span on a receipt's verdict leaf ---------------------------------------------------------

RP = "papers/x/receipt.json"


def span(verdict: str, ref: str = f"path:{RP}#/verdict", kind: str = "quote") -> str:
    return f'<sworn r="{ref}" k="{kind}">`{verdict}`</sworn>'


def sworn_doc(body: str) -> bytes:
    return ("# RESULT\n\n" + body + "\n").encode("utf-8")


def test_a_sworn_span_on_the_verdict_leaf_declares_the_verdict():
    got = core._result_reading(sworn_doc("Verdict: " + span("PASS") + "."))
    assert got["code"] is None
    assert got["declarations"] == [{"form": "sworn", "receipt": RP, "prereg": None, "verdict": "PASS", "line": 3}]


def test_spans_on_other_leaves_are_not_declarations():
    body = (span("PASS", ref=f"path:{RP}#/gates_sha256") + " " + span("PASS", ref=f"path:{RP}#/verdict/x")
            + " " + span("PASS", ref=f"path:{RP}#/result/verdict") + " " + span("PASS", ref=f"path:{RP}#L3"))
    assert core._result_reading(sworn_doc(body)) == {"code": None, "kind": None, "detail": None,
                                                     "declarations": [], "unread": []}


LONG = "A" * 301
SWORN_ATTACKS = {
    # name: (body, words the refusal holds)
    "hidden-in-a-comment": ("Verdict: HOLD. <!-- " + span("PASS") + " -->", "hidden_commitment"),
    "behind-an-unterminated-comment": ("Verdict: HOLD.\n<!--\n" + span("PASS"), "hidden_commitment"),
    "nested": ('<sworn r="path:' + RP + '#/verdict" k="quote">' + span("PASS") + "</sworn>", "nesting"),
    "unclosed": ('<sworn r="path:' + RP + '#/verdict" k="quote">`PASS`', "unclosed"),
    "a-stray-closer-elsewhere": (span("PASS") + " and later </sworn>", "do not lex"),
    "single-quoted-attributes": ("<sworn r='path:" + RP + "#/verdict' k='quote'>`PASS`</sworn>", "do not lex"),
    "upper-case-tag": ('<SWORN r="path:' + RP + '#/verdict" k="quote">`PASS`</SWORN>', "do not lex"),
    "kind-numeric": (span("PASS", kind="numeric"), 'k="quote"'),
    "kind-unknown": (span("PASS", kind="quot"), "kind_unknown"),
    "kind-absent": (span("PASS", kind="absent"), "absent_over_partial"),
    "two-needles": ('<sworn r="path:' + RP + '#/verdict" k="quote">`PASS` or `HOLD`</sworn>', "needle_count"),
    "blank-needle": (span(" "), "needle_empty"),
    "directional-override": (span("\u202ePASS"), "directional_override"),
    "over-the-length-cap": (span(LONG), "length_cap"),
    "a-path-sworn-cannot-name": (span("PASS", ref="path:/abs/receipt.json#/verdict"), "receipt_form"),
    "unbalanced-fences": (F + "\n" + span("PASS"), "unbalanced_fences"),
}


@pytest.mark.parametrize("name", sorted(SWORN_ATTACKS))
def test_every_sworn_attack_on_the_verdict_leaf_refuses(name):
    body, words = SWORN_ATTACKS[name]
    got = core._result_reading(sworn_doc(body))
    assert got["code"] == REFUSES and got["kind"] == "FAIL", got
    assert words in got["detail"], got["detail"]
    assert got["declarations"] == []


def test_a_document_sworn_cannot_lex_refuses_only_when_a_tag_names_a_verdict_leaf():
    """The conformance README's own case: a wrapped line that opens with a backtick run leaves sworn's
    fence count odd, and the text names the verdict leaf in an inline code span, in no tag."""
    prose = "# notes\n\n  " + F + "locks in a wrapped line\n\nThe leaf `path:<receipt>#/verdict` is named in prose.\n"
    assert core._result_reading(prose.encode("utf-8"))["code"] is None
    got = core._result_reading((prose + span("PASS") + "\n").encode("utf-8"))
    assert got["code"] == REFUSES and "unbalanced_fences" in got["detail"]
    assert core._result_reading(b"\xff" + prose.encode("utf-8"))["code"] is None


def test_invalid_utf8_around_a_verdict_span_refuses():
    got = core._result_reading(b"\xff\n" + sworn_doc(span("PASS")))
    assert got["code"] == REFUSES and "UTF-8" in got["detail"]


@pytest.mark.parametrize("body", [
    F + "\n" + span("PASS") + "\n" + F,                  # inside a fenced code block: sworn skips it
    "`` " + span("PASS") + " ``",                         # inside an inline code span: sworn skips it
])
def test_a_span_sworn_does_not_read_is_no_declaration(body):
    got = core._result_reading(sworn_doc(body))
    assert got["code"] is None and got["declarations"] == []


@pytest.mark.parametrize("ref", ["r1#/verdict", "prereg:" + "a" * 64 + "#/verdict"])
def test_a_verdict_span_on_a_receipt_with_no_committed_path_is_printed_not_read(ref):
    got = core._result_reading(sworn_doc(span("PASS", ref=ref)))
    assert got["code"] is None and got["declarations"] == []
    assert [(u["code"], u["kind"], u["receipt"]) for u in got["unread"]] == [(NOT_READ, "NOTE", ref)]


# ---- both forms, and two spans: they must agree --------------------------------------------------------

def _paired(reading, path=RP):
    return [dict(d, receipt_path=path, result="RESULT.md") for d in reading["declarations"]]


def test_two_forms_that_agree_are_one_declaration_with_both_sources():
    data = result_doc({**BODIES["result"], "receipt": RP}, "\nVerdict: " + span("PASS") + ".\n")
    got = core._declared(_paired(core._result_reading(data)))
    assert got["findings"] == []
    assert [(d["receipt_path"], d["verdict"]) for d in got["declared"]] == [(RP, "PASS")]
    assert [s["form"] for s in got["declared"][0]["sources"]] == ["fence", "sworn"]


def test_two_forms_that_disagree_are_refused():
    data = result_doc({**BODIES["result"], "receipt": RP}, "\nVerdict: " + span("HOLD") + ".\n")
    got = core._declared(_paired(core._result_reading(data)))
    assert [(f["code"], f["kind"]) for f in got["findings"]] == [(DIFFER, "FAIL")]
    assert "PASS" in got["findings"][0]["detail"] and "HOLD" in got["findings"][0]["detail"]
    assert got["declared"][0]["verdict"] is None and got["declared"][0]["verdicts"] == ["PASS", "HOLD"]


def test_two_sworn_spans_that_disagree_are_refused():
    data = sworn_doc("Verdict: " + span("PASS") + ". Again: " + span("HOLD") + ".")
    got = core._declared(_paired(core._result_reading(data)))
    assert [f["code"] for f in got["findings"]] == [DIFFER]


def test_spans_on_two_receipts_are_two_declarations():
    data = sworn_doc(span("PASS") + " " + span("HOLD", ref="path:papers/y/receipt.json#/verdict"))
    decl = [dict(d, receipt_path=d["receipt"], result="RESULT.md") for d in core._result_reading(data)["declarations"]]
    got = core._declared(decl)
    assert got["findings"] == []
    assert [(d["receipt_path"], d["verdict"]) for d in got["declared"]] == [(RP, "PASS"),
                                                                           ("papers/y/receipt.json", "HOLD")]


# ---- the live sworn-span case, and A.1 on its bytes (styxx's own objects at REV) -------------------------

LIVE = "papers/frequency-resonance/RESULT_pmnist_robustness_2026_10_23.md"
LIVE_RECEIPT = "papers/frequency-resonance/pmnist_robustness_result.json"
LIVE_PREREG = "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md"
SEAL_224 = "3598e59d3fe708ebf2dd20992f253d02d753c88b"
LANDED = "BUDGET_CONFOUND__gap_does_not_survive_3x_steps"
TYPED = "ROBUST__ROTATION_LOAD_BEARING__beyond_diversity"


@pytest.fixture(scope="module")
def repo():
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", REV + "^{commit}"], capture_output=True).returncode:
        pytest.skip(f"{REV} is not in the object store (a shallow or partial clone); the live case is read from it")
    with G._Repo(ROOT) as r:
        if r.is_shallow():
            pytest.skip("a shallow clone holds no history to read the live case from")
        yield r


def test_the_live_result_declares_its_verdict_at_line_4(repo):
    got = core._result_reading(repo.read(REV, LIVE))
    assert got["code"] is None and got["unread"] == []
    assert got["declarations"] == [{"form": "sworn", "receipt": LIVE_RECEIPT, "prereg": None, "verdict": LANDED,
                                    "line": 4}]


def _sed_once_per_line(data: bytes, old: str, new: str) -> bytes:
    """``sed 's/old/new/'``: the leftmost match on each line (DESIGN 4.A, A.1)."""
    return b"\n".join(ln.replace(old.encode(), new.encode(), 1) for ln in data.split(b"\n"))


def test_a1_a_result_that_types_a_rung_the_receipt_did_not_land_on_reads_verdict_differs(repo):
    receipt = json.loads(repo.read(REV, LIVE_RECEIPT))
    rescored = core.rescore(repo.read(SEAL_224, LIVE_PREREG), receipt)
    assert rescored["verdict"] == LANDED
    recorded = {k: receipt[k] for k in ("verdict", "gates", "gates_sha256")}
    honest_reading = core._result_reading(repo.read(REV, LIVE))
    assert core.compare(rescored, recorded, declared=honest_reading["declarations"][0]["verdict"]) == []
    typed = core._result_reading(_sed_once_per_line(repo.read(REV, LIVE), LANDED, TYPED))
    assert typed["code"] is None and [d["verdict"] for d in typed["declarations"]] == [TYPED]
    found = core.compare(rescored, recorded, declared=typed["declarations"][0]["verdict"])
    assert [(f["code"], f["kind"]) for f in found] == [("VERDICT_DIFFERS", "FAIL")]
    assert found[0]["detail"] == f"the sealed decision scores this receipt {LANDED}; the RESULT prints {TYPED}"


def test_every_markdown_file_at_rev_reads_as_a_result_without_a_refusal(repo):
    """K11 reads every markdown file a pull request adds or changes as a RESULT, so the reader must
    not refuse a document that is not one: every .md blob in the tree at REV is read, none refused."""
    paths = [p for p in repo.paths(REV) if p.lower().endswith(".md")]
    readings = {p: core._result_reading(repo.read(REV, p) or b"") for p in paths}
    refused = sorted(p for p, r in readings.items() if r["code"] is not None)
    declaring = sum(bool(r["declarations"]) for r in readings.values())
    print(f"markdown at {REV[:12]}: {len(paths)} read as RESULTs, {declaring} declare, {len(refused)} refused")
    assert refused == []
    assert (len(paths), declaring) == (1560, 13)


def test_every_sworn_verdict_span_at_rev_reads_and_agrees_with_its_sealed_decision(repo):
    """The lab's tree at REV: every markdown file that names a ``json#/verdict`` leaf, read as a
    RESULT through the git layer; every declaration on a receipt that names a PREREG re-scored under
    the PREREG's default seal and compared.

    Measured by the build (B3 report): 11 of the 17 declarations sit on six receipts that hold their
    metrics under ``/metrics``, and their sealed gates blocks name no ``receipt_pointer``, so under
    DESIGN 3.2(a)'s default (the whole receipt) the re-score refuses them; read at ``/metrics`` each
    of the 11 gives the verdict the RESULT declares. No declaration differs from its re-score."""
    paths = repo.grep(REV, "json#/verdict", ":(icase)*.md")
    rows = [G._result_declarations(repo, REV, p) for p in paths]
    assert all(r["code"] is None and r["findings"] == [] for r in rows), [r for r in rows if r["code"] or r["findings"]]
    declaring = [r for r in rows if r["declared"]]
    spans = sum(len(s["sources"]) for r in rows for s in r["declared"])
    assert (len(paths), len(declaring), spans) == (20, 13, 17)
    tally, at_metrics = Counter(), Counter()
    for r in declaring:
        for d in r["declared"]:
            if d["prereg_path"] is None:
                tally["names no PREREG"] += 1
                continue
            sealed = G._seal(repo, d["prereg_path"], REV)["bytes"]
            rescored = core.rescore(sealed, d["receipt"])
            if rescored["code"] is not None:
                tally[rescored["code"]] += 1
                again = core.rescore(sealed, d["receipt"].get("metrics"))
                at_metrics["agrees" if again["code"] is None and again["verdict"] == d["verdict"] else "other"] += 1
            else:
                tally["agrees" if rescored["verdict"] == d["verdict"] else "differs"] += 1
    print(f"at {REV[:12]}: {len(paths)} files name a json#/verdict leaf, {len(declaring)} declare through "
          f"{spans} spans; per declaration: {dict(sorted(tally.items()))}; the refused ones read at /metrics: "
          f"{dict(at_metrics)}")
    assert tally["differs"] == 0
    assert dict(tally) == {"SCORER_REFUSES": 11, "agrees": 6}
    assert dict(at_metrics) == {"agrees": 11}


# ---- K11: C1 pairs every receipt a RESULT names, when the pull request adds or changes the RESULT ---------

def receipt_json(m: float = 0.7, **extra) -> str:
    return json.dumps({"prereg": "PREREG_a.md", "m": m, **extra}, indent=1) + "\n"


def result_md(verdict: str, receipt: str = "p/receipt.json", fence: bool = False, prereg_name: str = "PREREG_a.md"):
    if fence:
        return f"# RESULT\n\n{F}result\n" + json.dumps({"prereg": prereg_name, "receipt": receipt,
                                                        "verdict": verdict}) + f"\n{F}\n"
    return "# RESULT\n\nVerdict: " + span(verdict, ref=f"path:{receipt}#/verdict") + ".\n"


@pytest.fixture()
def base_repo(tmp_path):
    """main: a PREREG (bar 0.5) and its receipt (m 0.7, so PASS), each landed; the receipt's
    commit is r."""
    g = Git(tmp_path / "r")
    g.write("p/PREREG_a.md", prereg(0.5))
    s = g.commit("seal")
    g.write("p/receipt.json", receipt_json())
    r = g.commit("receipt")
    return g, s, r


def test_k11_a_result_changed_in_the_pull_request_pairs_the_base_receipt_it_names(base_repo):
    g, s, r = base_repo
    g.write("p/RESULT_a.md", result_md("PASS"))
    base = g.commit("result")
    g("switch", "-q", "-c", "typed")
    g.write("p/RESULT_a.md", result_md("HOLD"))
    head = g.commit("type the rung")
    with G._Repo(g.root) as repo:
        assert G._receipts_written(repo, base, head)["receipts"] == []          # no receipt is written
        got = G._pairs_written(repo, base, head)
    assert got["findings"] == []
    [pair] = got["pairs"]
    assert (pair["path"], pair["status"], pair["receipt_commit"]) == ("p/receipt.json", "named", r)
    assert [(d["result"], d["verdict"]) for d in pair["declared"]] == [("p/RESULT_a.md", "HOLD")]
    rescored = core.rescore(prereg(0.5), pair["receipt"])
    found = core.compare(rescored, {}, declared=pair["declared"][0]["verdict"])
    assert "VERDICT_DIFFERS" in [f["code"] for f in found]


def test_k11_a_receipt_and_its_result_added_together_are_one_pair(base_repo):
    g, s, r = base_repo
    base = g("rev-parse", "HEAD")
    g("switch", "-q", "-c", "rerun")
    g.write("p/receipt_2.json", receipt_json(0.3))
    g.write("p/RESULT_b.md", result_md("HOLD", receipt="receipt_2.json", fence=True))
    head = g.commit("re-run")
    with G._Repo(g.root) as repo:
        got = G._pairs_written(repo, base, head)
    assert got["findings"] == []
    [pair] = got["pairs"]
    assert (pair["path"], pair["status"], pair["receipt_commit"]) == ("p/receipt_2.json", "A", head)
    assert [(d["result"], d["verdict"]) for d in pair["declared"]] == [("p/RESULT_b.md", "HOLD")]
    assert [s["form"] for s in pair["declared"][0]["sources"]] == ["fence"]


def test_k11_a_receipt_written_with_no_result_has_no_declaration(base_repo):
    g, s, r = base_repo
    base = g("rev-parse", "HEAD")
    g("switch", "-q", "-c", "rerun")
    g.write("p/receipt_2.json", receipt_json(0.3))
    head = g.commit("re-run")
    with G._Repo(g.root) as repo:
        got = G._pairs_written(repo, base, head)
    assert [(p["path"], p["declared"]) for p in got["pairs"]] == [("p/receipt_2.json", [])]


@pytest.mark.parametrize("doc, code, kind", [
    (result_md("PASS", receipt="p/missing.json"), "RECEIPT_ABSENT", "INPUT"),
    (result_md("PASS", receipt="missing.json", fence=True), "RECEIPT_ABSENT", "INPUT"),
    (result_md("PASS", fence=True, prereg_name="PREREG_other.md"), PREREG_DIFFERS, "FAIL"),
    (result_md("PASS", receipt="p/plain.json", fence=True), PREREG_DIFFERS, "FAIL"),
    (result_md("PASS", receipt="p/plain.json"), NO_RECEIPT, "NOTE"),
    (f"# RESULT\n\n<!--\n{F}result\n{{}}\n{F}\n-->\n", REFUSES, "FAIL"),
    (result_md("PASS", fence=True) + "\nVerdict: " + span("HOLD", ref="path:p/receipt.json#/verdict") + "\n",
     DIFFER, "FAIL"),
])
def test_k11_what_a_result_names_is_read_or_refused(base_repo, doc, code, kind):
    g, s, r = base_repo
    g.write("p/PREREG_other.md", prereg(0.9))
    g.write("p/plain.json", json.dumps({"verdict": "PASS"}) + "\n")
    base = g.commit("more")
    g("switch", "-q", "-c", "pr")
    g.write("p/RESULT_a.md", doc)
    head = g.commit("result")
    with G._Repo(g.root) as repo:
        got = G._pairs_written(repo, base, head)
    assert [(f["code"], f["kind"], f["path"]) for f in got["findings"]] == [(code, kind, "p/RESULT_a.md")]
    if kind == "NOTE" or code in (REFUSES, "RECEIPT_ABSENT"):
        assert got["pairs"] == []


def test_k11_a_result_deleted_or_left_alone_is_not_read(base_repo):
    """Neither is paired. Since the B3 review (N1) the deleted one, which declared a verdict at the
    merge base, is a RESULT_DECLARATION_DROPPED NOTE; the one left alone is not read at all."""
    g, s, r = base_repo
    g.write("p/RESULT_a.md", result_md("HOLD"))
    g.write("p/RESULT_b.md", result_md("HOLD"))
    base = g.commit("results")
    g("switch", "-q", "-c", "pr")
    g("rm", "-q", "p/RESULT_a.md")
    g.write("README.md", "unrelated\n")
    head = g.commit("delete one, touch another file")
    with G._Repo(g.root) as repo:
        got = G._pairs_written(repo, base, head)
    assert (got["pairs"], got["notes"], got["excluded"]) == ([], [], [])
    assert [(f["code"], f["kind"], f["path"]) for f in got["findings"]] == [
        ("RESULT_DECLARATION_DROPPED", "NOTE", "p/RESULT_a.md")]


def test_k11_an_excluded_result_is_listed_not_read(base_repo):
    g, s, r = base_repo
    base = g("rev-parse", "HEAD")
    g("switch", "-q", "-c", "pr")
    g.write("docs/RESULT_a.md", result_md("HOLD"))
    head = g.commit("result under an excluded prefix")
    with G._Repo(g.root) as repo:
        got = G._pairs_written(repo, base, head, exclude=("docs/",))
    assert got["pairs"] == [] and got["excluded"] == ["docs/RESULT_a.md"]


# ---- the conformance vectors, and the house rules --------------------------------------------------------

def test_the_fence_vectors_replay():
    doc = json.loads((SET / "vectors_fence.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "styxx.prereg-ci.conformance.fence/v0"
    seen = set()
    for v in doc["vectors"]:
        data = blob_bytes(doc["blobs"][v["inputs"]["text"]])
        text = core._decision_text(data)
        e = v["expect"]
        assert _select(text, v["info"]) == e["select"], v["name"]
        if v["info"] == "gates":
            assert _protocol(text) == e["protocol"] == e["select"], v["name"]
        else:
            assert _json_leg(text, v["info"]) == e["fence_json"], v["name"]
        if v["info"] == "result":
            assert json.loads(json.dumps(core._result_reading(data))) == e["reading"], v["name"]
        seen.add(v["info"])
    assert seen == set(INFOS)
    assert len(doc["vectors"]) >= 3 * len(attacks("gates")) + 2 * len(shadows("gates"))


def _select(text, info):
    try:
        got = core._select_fence(text, info)
    except GateSpecError as e:
        return {"raises": "GateSpecError", "message": str(e)}
    return {"none": True} if got is None else {"block": got}


def _protocol(text):
    try:
        return {"block": protocol._select_gates_block(text)}
    except GateSpecError as e:
        if str(e).startswith("prereg has no ```gates block"):
            return {"none": True}
        return {"raises": "GateSpecError", "message": str(e)}


def _json_leg(text, info):
    try:
        got = core._fence_json(text, info)
    except GateSpecError as e:
        return {"raises": "GateSpecError", "message": str(e)}
    return {"none": True} if got is None else {"line": got[0], "object": got[1]}


def test_the_fence_vectors_regenerate_byte_for_byte():
    r = subprocess.run([sys.executable, "-B", str(SET / "gen_vectors_fence.py"), "--check"], capture_output=True,
                       cwd=str(ROOT))
    assert r.returncode == 0, r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")


def test_the_result_codes_sit_in_their_own_table():
    assert core._RESULT_KIND == {REFUSES: "FAIL", DIFFER: "FAIL", NOT_READ: "NOTE"}
    assert not set(core._RESULT_KIND) & set(core._KIND)
    assert G._KIND[PREREG_DIFFERS] == "FAIL" and G._KIND[NO_RECEIPT] == "NOTE"


def test_sworn_private_names_are_used_in_core_only():
    for p in sorted((ROOT / "styxx" / "prereg_ci").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_") and not node.attr.startswith("__"):
                if isinstance(node.value, ast.Name) and node.value.id == "_sworn":
                    assert p.name == "core.py", (p.name, node.attr)


def test_the_public_surface_does_not_grow():
    import styxx.prereg_ci as P
    assert set(P.__all__) <= {"rescore", "compare", "licences"} and set(core.__all__) == set(P.__all__)
