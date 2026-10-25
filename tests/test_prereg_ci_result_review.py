# -*- coding: utf-8 -*-
"""prereg-ci B3, the review's fix round: the shapes the B3 review got past the RESULT readers.

ADV1. A ```result or ```locks fence on a list item's continuation line (4 or more columns in, or behind a tab)
renders as a fence, and neither the scanner nor the B3 shadow rule counted it, so a second result block a reader
sees went unread. The rule now: behind a container marker on the same line (a byte-order mark, a block quote, a
list item, a footnote label, each at any indent) a result-like or locks-like opener refuses at any indent; with no
marker on the line, one 4 or more columns in refuses once any line above it opens a list item or a footnote,
whether or not that container is still open. Indented code with no such line above it stays code.

ADV2. Renaming the RESULT off ``.md`` took it out of the reading. Every file the pull request writes is now read
as a RESULT, whatever its suffix, when its bytes hold the marker of a form (``core._result_candidate``), and a
``.md`` file always; a RESULT that declared a verdict at the merge base and declares none of it at HEAD (deleted,
renamed away or under ``exclude``, or edited to prose) is ``RESULT_DECLARATION_DROPPED``, a NOTE.

M1. A RESULT in UTF-16 or UTF-32 (a byte-order mark), or holding NUL bytes, is read for the forms' markers and
refused when it holds one: a RESULT is read as UTF-8.

The renderer evidence for these shapes (markdown-it-py, CommonMark, with its footnote plugin) is in the fix
round's case files; this file does not import a renderer. Every repository built here is a ``git init`` in
tmp_path with no remote.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "conformance" / "prereg_ci"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci.replay import blob_bytes  # noqa: E402
from styxx import protocol  # noqa: E402
from styxx import sworn  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from styxx.prereg_ci import git as G  # noqa: E402
from tests.test_prereg_ci_git import Git, prereg  # noqa: E402

GateSpecError = protocol.GateSpecError
F = "```"
REFUSES, DROPPED = "RESULT_REFUSES", "RESULT_DECLARATION_DROPPED"
BODIES = {
    "result": {"prereg": "PREREG_a.md", "receipt": "receipt.json", "verdict": "PASS"},
    "locks": {"scorer": "score.py", "reads": ["data.csv"], "outcomes": ["PASS", "FAIL"], "withdraws": {}},
}
INFOS = tuple(BODIES)
BODIES["gates"] = {"gates": {"G": {"metric": "m", "op": "<=", "value": 0.2}},
                   "outcomes": [{"when": {"G": True}, "verdict": "PASS"}, {"when": {"G": False}, "verdict": "FAIL"}],
                   "smoke_verdict": "SMOKE"}


def body(info: str, verdict: str = "PASS") -> str:
    b = dict(BODIES[info])
    if info == "result":
        b["verdict"] = verdict
    return json.dumps(b)


def indented(block: str, pad: str) -> str:
    return "".join(pad + ln + "\n" for ln in block.splitlines())


def block(info: str, verdict: str = "PASS") -> str:
    return f"{F}{info}\n{body(info, verdict)}\n{F}\n"


def contained(info: str) -> dict:
    """Shapes a CommonMark renderer (markdown-it-py with its footnote plugin) shows as an <info> fence and that
    the B3 rule did not count: name -> document."""
    b = block(info, "FAIL")
    return {
        "list-continuation-4": "# d\n\n-   the verdict:\n\n" + indented(b, "    "),
        "ordered-continuation-4": "# d\n\n10. the verdict:\n" + indented(b, "    "),
        "list-continuation-tab": "# d\n\n- the verdict:\n\n" + indented(b, "\t"),
        "list-continuation-7": "# d\n\n-   the verdict:\n\n" + indented(b, "       "),
        "nested-list-continuation": "# d\n\n- a\n  - b\n\n" + indented(b, "      "),
        "lazy-paragraph-then-4": "# d\n\n- a\nlazy\n" + indented(b, "    "),
        "quoted-list-continuation": "# d\n\n> - a\n>\n" + "".join(">     " + ln + "\n" for ln in b.splitlines()),
        "nested-marker-at-4": "# d\n\n-   a\n\n    - " + b.splitlines()[0] + "\n" + indented("\n".join(
            b.splitlines()[1:]), "      "),
        "footnote-continuation": "# d\n\ntext[^1]\n\n[^1]: note\n\n" + indented(b, "    "),
        "footnote-label-line": "# d\n\ntext[^1]\n\n[^1]: " + b.splitlines()[0] + "\n" + indented(
            "\n".join(b.splitlines()[1:]), "    "),
        "quote-then-tab": "# d\n\n" + "".join("> \t" + ln + "\n" for ln in b.splitlines()),
    }


@pytest.mark.parametrize("info", INFOS)
@pytest.mark.parametrize("name", sorted(contained("result")))
def test_a_fence_inside_a_container_the_b3_rule_missed_refuses(info, name):
    text = contained(info)[name]
    assert core._select_fence(text, info) is None              # the scanner counts nothing there
    assert core._shadow_fences(text, info)
    with pytest.raises(GateSpecError, match="render"):
        core._fence_json(text, info)
    if info == "result":
        got = core._result_reading(text.encode("utf-8"))
        assert got["code"] == REFUSES and got["declarations"] == [], got


def test_adv1_a_plain_fence_beside_a_list_continuation_fence_refuses():
    """The review's sharp case: the renderer shows two result blocks, PASS and FAIL; B3 read PASS."""
    text = f"# R\n\n{block('result', 'PASS')}\n-   but read this one:\n\n" + indented(block("result", "FAIL"), "    ")
    assert core._select_fence(text, "result") == body("result", "PASS")
    got = core._result_reading(text.encode("utf-8"))
    assert got["code"] == REFUSES and "line 9" in got["detail"], got
    assert got["declarations"] == []


def test_adv1_a_prereg_whose_one_locks_fence_sits_on_a_list_continuation_refuses():
    text = "# P\n\n-   the decision:\n\n" + indented(block("locks"), "    ")
    with pytest.raises(GateSpecError, match="may render"):
        core._fence_json(text, "locks")


@pytest.mark.parametrize("info", INFOS)
def test_indented_code_with_no_list_item_or_footnote_above_it_is_still_code(info):
    for text in ("# d\n\ntext\n\n" + indented(block(info), "    "), "# d\n\n" + indented(block(info), "\t"),
                 "# d\n\n> quoted\n\n" + indented(block(info), "    ")):
        assert core._shadow_fences(text, info) == []
        assert core._fence_json(text, info) is None


@pytest.mark.parametrize("info", INFOS)
def test_an_indented_opener_below_a_list_that_has_ended_refuses_too(info):
    """Over-refusal, stated: the renderer prints this as code (the paragraph ends the list), and the reader
    does not follow a container to its end, so it refuses once a list item has opened above."""
    text = "# d\n\n- a\n\npara\n\n" + indented(block(info), "    ")
    with pytest.raises(GateSpecError, match="may render"):
        core._fence_json(text, info)


@pytest.mark.parametrize("info", INFOS)
def test_the_b3_shadow_classes_keep_their_words(info):
    text = "# d\n\n" + "".join("> " + ln + "\n" for ln in block(info).splitlines())
    with pytest.raises(GateSpecError) as e:
        core._fence_json(text, info)
    assert str(e.value).startswith(f"a ```{info} fence at line 3 renders as a {info} block (behind a byte-order mark, "
                                   f"in a block quote or a list item, or with a second word in its info string)")


def test_the_gates_fence_keeps_the_scanners_reading_of_these_shapes():
    for name, text in contained("gates").items():
        try:
            want = ("block", protocol._select_gates_block(text))
        except GateSpecError as e:
            want = ("none", None) if str(e).startswith("prereg has no ```gates block") else ("raises", str(e))
        assert want == ("none", None), name
        assert core._select_fence(text, "gates") is None, name


# ---- M1: wide encodings and NUL bytes ------------------------------------------------------------------------

def result_text(verdict: str = "FAIL") -> str:
    return f"# RESULT a\n\n{block('result', verdict)}"


WIDE = {
    "utf-16-le-bom": lambda t: b"\xff\xfe" + t.encode("utf-16-le"),
    "utf-16-be-bom": lambda t: b"\xfe\xff" + t.encode("utf-16-be"),
    "utf-32-le-bom": lambda t: b"\xff\xfe\x00\x00" + t.encode("utf-32-le"),
    "utf-32-be-bom": lambda t: b"\x00\x00\xfe\xff" + t.encode("utf-32-be"),
}


@pytest.mark.parametrize("enc", sorted(WIDE))
def test_a_result_in_a_wide_encoding_refuses_when_it_holds_a_form(enc):
    data = WIDE[enc](result_text())
    assert core._result_candidate(data)
    got = core._result_reading(data)
    assert got["code"] == REFUSES and got["declarations"] == [], got
    assert enc[:6].upper() in got["detail"] and "read as UTF-8" in got["detail"]


@pytest.mark.parametrize("enc", sorted(WIDE))
def test_a_wide_document_with_no_form_declares_nothing(enc):
    data = WIDE[enc]("# notes\n\nThe verdict was FAIL.\n\n" + F + "python\nprint(1)\n" + F + "\n")
    assert not core._result_candidate(data)
    assert core._result_reading(data) == {"code": None, "kind": None, "detail": None, "declarations": [],
                                          "unread": []}


def test_bytes_with_no_byte_order_mark_are_read_as_they_stand():
    """UTF-16 with no mark holds NUL bytes, which git's diff reads as binary: it is read as the bytes it is,
    and its fence is no fence there. A UTF-8 RESULT holding a NUL byte is read as UTF-8, as at B3."""
    for enc in ("utf-16-le", "utf-16-be"):
        got = core._result_reading(result_text().encode(enc))
        assert got["code"] is None and got["declarations"] == [], (enc, got)
    got = core._result_reading(result_text().encode("utf-8") + b"\x00")
    assert got["code"] is None and [d["verdict"] for d in got["declarations"]] == ["FAIL"]


def test_binary_bytes_are_no_candidate_by_chance():
    """For bytes that are not UTF-8 the marker stays B3's narrow one: a fence run followed by a high byte
    is common in compressed data, and a wider marker holds in 43 of the 203 files in styxx's own tree that
    are not UTF-8."""
    rng = random.Random(20261025)
    blob = bytes(rng.getrandbits(8) for _ in range(1 << 20))
    assert not core._result_candidate(blob) and core._result_reading(blob)["code"] is None
    planted = b"\x89PNG\r\n\x1a\n" + blob[:4096] + b"```\xc3\xa9" + blob[4096:8192]
    assert not core._result_candidate(planted) and core._result_reading(planted)["code"] is None
    marked = b"\x89PNG\r\n\x1a\n" + blob[:4096] + b"\n```result\n" + blob[4096:8192]
    assert core._result_candidate(marked) and core._result_reading(marked)["code"] == REFUSES


def test_a_wide_document_holding_a_sworn_verdict_span_or_a_confusable_fence_refuses():
    span = '<sworn r="path:p/receipt.json#/verdict" k="quote">`FAIL`</sworn>'
    for text in ("# R\n\nVerdict: " + span + "\n", "# R\n\n" + F + "r\u0435sult\n{}\n" + F + "\n"):
        data = b"\xff\xfe" + text.encode("utf-16-le")
        assert core._result_reading(data)["code"] == REFUSES, text


# ---- the candidate test: no reading finds anything in bytes it passes over --------------------------------------

PIECES = ["# t\n", "\n", "text\n", "- item\n", "10. item\n", "> quote\n", "[^1]: note\n", "    ", "\t", "  ", ">",
          "- ", F, "~~~", "````", "result", "RESULT", "r\u0435sult", "locks", "python", " json", "\u3000", "\u00a0",
          "\x0c", "<!--", "-->", '{"prereg": "P.md", "receipt": "r.json", "verdict": "X"}', "\n" + F + "\n",
          '<sworn r="path:r.json#/verdict" k="quote">`X`</sworn>', '<sworn r="path:r.json#/verdict"',
          "</sworn>", "#/verdict", "`", "\ufeff", "caf\u00e9", "\x00"]


def test_result_candidate_is_true_wherever_a_reading_finds_anything():
    rng = random.Random(20261025)
    encs = ("utf-8", "utf-16", "utf-16-le", "utf-32", "latin-1")
    seen = {"candidate": 0, "passed over": 0}
    for _ in range(6000):
        text = "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 14)))
        enc = rng.choice(encs)
        try:
            data = text.encode(enc)
        except UnicodeEncodeError:
            data = text.encode("utf-8")
        got = core._result_reading(data)
        if core._result_candidate(data):
            seen["candidate"] += 1
            continue
        seen["passed over"] += 1
        assert got == {"code": None, "kind": None, "detail": None, "declarations": [], "unread": []}, (enc, text)
        try:
            core._fence_json(core._decision_text(data), "result")
        except UnicodeDecodeError:
            pass
        except GateSpecError as e:
            raise AssertionError((enc, text, str(e)))
    assert seen["candidate"] > 1000 and seen["passed over"] > 1000, seen


def test_result_candidate_holds_every_committed_fence_vector_that_declares_or_refuses():
    for fn in ("vectors_fence.json", "vectors_fence_containers.json"):
        doc = json.loads((SET / fn).read_text(encoding="utf-8"))
        for v in doc["vectors"]:
            if v["info"] != "result":
                continue
            data = blob_bytes(doc["blobs"][v["inputs"]["text"]])
            e = v["expect"]
            finds = (e["reading"]["code"] is not None or e["reading"]["declarations"]
                     or e["reading"]["unread"] or "raises" in e.get("fence_json", {}))
            if finds:
                assert core._result_candidate(data), (fn, v["name"])


# ---- ADV2: a RESULT under another suffix, and a declaration the pull request drops -------------------------------

def result_md(verdict: str) -> str:
    return result_text(verdict)


@pytest.fixture()
def lab(tmp_path):
    """main: a PREREG (bar 0.5), its receipt (m 0.7, so PASS) and RESULT_a.md declaring PASS by a result fence."""
    g = Git(tmp_path / "r")
    g.write("PREREG_a.md", prereg(0.5))
    g.commit("seal")
    g.write("receipt.json", json.dumps({"prereg": "PREREG_a.md", "m": 0.7, "verdict": "PASS"}) + "\n")
    r = g.commit("receipt")
    g.write("RESULT_a.md", result_md("PASS"))
    base = g.commit("result")
    g("switch", "-q", "-c", "pr")
    return g, r, base


def read_pr(g, base, head, exclude=()):
    with G._Repo(g.root) as repo:
        got = G._pairs_written(repo, base, head, exclude)
        changes = [(x["status"], x["path"]) for x in repo.changes(base, head)["changes"]]
    pairs = [(p["path"], p["status"], [(d["result"], d["verdict"]) for d in p["declared"]]) for p in got["pairs"]]
    differs = []
    for p in got["pairs"]:
        rescored = core.rescore(prereg(0.5), p["receipt"])
        for d in p["declared"]:
            differs += [f["code"] for f in core.compare(rescored, {}, declared=d["verdict"])]
    return {"changes": changes, "pairs": pairs,
            "findings": [(f["code"], f["kind"], f["path"]) for f in got["findings"]],
            "details": [f["detail"] for f in got["findings"]], "compare": differs}


@pytest.mark.parametrize("new", ["RESULT_a.markdown", "RESULT_a.mdown", "RESULT_a.txt", "RESULT_a", "RESULT_a.json",
                                 "notes/RESULT_a.rst"])
def test_adv2_a_result_renamed_off_md_is_read_and_its_typed_verdict_differs(lab, new):
    g, r, base = lab
    g.mv("RESULT_a.md", new)
    g.write(new, result_md("FAIL"))
    head = g.commit("rename the RESULT and type FAIL")
    got = read_pr(g, base, head)
    assert got["pairs"] == [("receipt.json", "named", [(new, "FAIL")])], got
    assert "VERDICT_DIFFERS" in got["compare"]
    assert [f for f in got["findings"] if f[0] == DROPPED] == []


def test_adv2_a_result_added_as_markdown_is_read(lab):
    g, r, base = lab
    g.write("RESULT_b.markdown", result_md("HOLD"))
    head = g.commit("a second RESULT under another suffix")
    got = read_pr(g, base, head)
    assert got["pairs"] == [("receipt.json", "named", [("RESULT_b.markdown", "HOLD")])], got
    assert "VERDICT_DIFFERS" in got["compare"]


def test_a_written_file_with_no_form_is_not_read_as_a_result(lab):
    g, r, base = lab
    g.write("tool.py", "print('the verdict was FAIL')\n")
    g.write("data.bin", bytes(range(256)) * 4)
    head = g.commit("code and data")
    got = read_pr(g, base, head)
    assert got["pairs"] == [] and got["findings"] == [], got


def test_m1_a_result_rewritten_as_utf16_refuses_in_the_pull_request(lab):
    g, r, base = lab
    g.write("RESULT_a.md", b"\xff\xfe" + result_md("FAIL").encode("utf-16-le"))
    head = g.commit("rewrite the RESULT as UTF-16")
    got = read_pr(g, base, head)
    assert (REFUSES, "FAIL", "RESULT_a.md") in got["findings"], got
    assert got["pairs"] == []


def test_n1_a_result_the_pull_request_deletes_is_a_note(lab):
    g, r, base = lab
    g("rm", "-q", "RESULT_a.md")
    head = g.commit("delete the RESULT")
    got = read_pr(g, base, head)
    assert got["pairs"] == [] and got["findings"] == [(DROPPED, "NOTE", "RESULT_a.md")], got
    assert "deletes it" in got["details"][0] and "receipt.json" in got["details"][0]


def test_n1_a_result_renamed_under_exclude_is_a_note(lab):
    g, r, base = lab
    g.mv("RESULT_a.md", "docs/RESULT_a.md")
    head = g.commit("move the RESULT under an excluded prefix")
    got = read_pr(g, base, head, exclude=("docs/",))
    assert got["pairs"] == [] and got["findings"] == [(DROPPED, "NOTE", "RESULT_a.md")], got
    assert "docs/RESULT_a.md" in got["details"][0] and "exclude" in got["details"][0]


def test_n1_adv4_a_result_edited_to_prose_is_a_note(lab):
    g, r, base = lab
    g.write("RESULT_a.md", "# RESULT a\n\nVerdict: **FAIL** (receipt.json, under PREREG_a.md).\n")
    head = g.commit("prose")
    got = read_pr(g, base, head)
    assert got["pairs"] == [] and got["findings"] == [(DROPPED, "NOTE", "RESULT_a.md")], got


def test_n1_a_result_that_drops_one_of_two_receipts_names_that_one(lab):
    g, r, base = lab
    g.write("receipt_2.json", json.dumps({"prereg": "PREREG_a.md", "m": 0.9, "verdict": "PASS"}) + "\n")
    g.write("RESULT_c.md", "# RESULT c\n\n"
            + '<sworn r="path:receipt.json#/verdict" k="quote">`PASS`</sworn> and '
            + '<sworn r="path:receipt_2.json#/verdict" k="quote">`PASS`</sworn>\n')
    base2 = g.commit("a RESULT naming two receipts")
    g("switch", "-q", "-c", "pr2")
    g.write("RESULT_c.md", "# RESULT c\n\n" + '<sworn r="path:receipt.json#/verdict" k="quote">`PASS`</sworn>\n')
    head = g.commit("drop one span")
    got = read_pr(g, base2, head)
    assert got["findings"] == [(DROPPED, "NOTE", "RESULT_c.md")], got
    assert "receipt_2.json" in got["details"][0] and "receipt.json," not in got["details"][0]
    assert got["pairs"] == [("receipt.json", "named", [("RESULT_c.md", "PASS")])]


def test_a_rename_that_keeps_the_declaration_is_no_note(lab):
    g, r, base = lab
    g.mv("RESULT_a.md", "RESULT_a.markdown")
    head = g.commit("rename only")
    got = read_pr(g, base, head)
    assert got["findings"] == [] and got["pairs"] == [("receipt.json", "named", [("RESULT_a.markdown", "PASS")])]
    assert "VERDICT_DIFFERS" not in got["compare"]


def test_the_drop_code_is_a_note_in_the_git_layers_table():
    assert G._KIND[DROPPED] == "NOTE"


# ---- the new vector file, and the conformance README ----------------------------------------------------------

def test_the_container_vectors_replay():
    doc = json.loads((SET / "vectors_fence_containers.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "styxx.prereg-ci.conformance.fence-containers/v0"
    names = set()
    for v in doc["vectors"]:
        data = blob_bytes(doc["blobs"][v["inputs"]["text"]])
        e = v["expect"]
        assert core._result_candidate(data) is e["candidate"], v["name"]
        if "fence_json" in e:
            try:
                got = core._fence_json(core._decision_text(data), v["info"])
                leg = {"none": True} if got is None else {"line": got[0], "object": got[1]}
            except GateSpecError as err:
                leg = {"raises": "GateSpecError", "message": str(err)}
            assert leg == e["fence_json"], v["name"]
        if "reading" in e:
            assert json.loads(json.dumps(core._result_reading(data))) == e["reading"], v["name"]
        names.add(v["name"])
    for info in INFOS:
        for name in contained(info):
            assert f"{info}/contained/{name}" in names


def test_the_container_vectors_regenerate_byte_for_byte():
    r = subprocess.run([sys.executable, "-B", str(SET / "gen_vectors_fence_containers.py"), "--check"],
                       capture_output=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")


def test_the_conformance_readme_opens_no_fence_mid_list_and_reads_as_no_result():
    raw = (SET / "README.md").read_bytes()
    assert sworn.scan(raw)["document_malformed"] is None          # sworn's fence lines balance
    for i, ln in enumerate(raw.decode("utf-8").split("\n"), 1):
        stripped = ln.lstrip(" ")
        if stripped.startswith(("`" * 3, "~" * 3)):
            assert ln == stripped, (i, ln)                         # a fence line sits at column 0, outside a list
    assert core._result_reading(raw)["code"] is None
    for info in INFOS:
        assert core._shadow_fences(raw.decode("utf-8"), info) == []
