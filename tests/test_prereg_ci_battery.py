# -*- coding: utf-8 -*-
"""prereg-ci B8: the mutation battery's misses, each pinned by a case that failed before its fix.

The B8 red team ran 139 mutation cases against prereg-ci at 9b32b76d; 16 that the tool must refuse
passed with exit 0, and the fresh red team found two majors. Each class is pinned here in the shape the
battery built it, on repositories made with ``git init`` in tmp_path (no remote, no network):

* M1: a gates fence a renderer shows and the scanner does not count (a block quote, a list item's
  continuation 4 columns in, a second word in the info string) beside the counted one, or alone;
* M2: a pull request that deletes the seals manifest, and one that re-adds it with a line restated
  after a merged deletion;
* M3: a receipt kept under a suffix other than ``.json``;
* M4: a PREREG kept under a suffix other than ``.md`` that a receipt names (C0 and the tree's licences);
* M5: a locked scorer that reads the clock (printed as the boundary, not refused: KNOWN_MISSES);
* F1: a locks PREREG that arrives in the pull request, whose scorer is the pull request's own code;
* F2: author bytes that would open a line of the printout with a workflow command.

``prereg-ci/KNOWN_MISSES.md`` names what v0 does not refuse and why; the cases at the end of this file
pin those boundaries, so a later hardening is seen as a change.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.prereg_ci import core  # noqa: E402
from tests.test_prereg_ci_cli import F, SENTENCE, TABLE, prereg, result_md, run, scored  # noqa: E402
from tests.test_prereg_ci_git import Git  # noqa: E402

HOLD = {"acc": 0.95, "ece": 0.08}
SHIP = {"acc": 0.95, "ece": 0.03}
CFG = 'surfaces = ["README*.md"]\n'
README_PLAIN = "# demo\n\nA digits model.\n"


def block(table=None) -> str:
    return json.dumps(table or TABLE, indent=1)


def moved(bar: float) -> dict:
    t = json.loads(json.dumps(TABLE))
    t["gates"]["G_ece"]["value"] = bar
    return t


def quoted(text: str) -> str:
    return "\n".join("> " + ln for ln in text.split("\n"))


def fence(table=None, info="gates") -> str:
    return f"{F}{info}\n{block(table)}\n{F}"


def sealed(tmp_path, name: str, text: str, readme: str = README_PLAIN) -> Git:
    g = Git(tmp_path / name)
    g.write("PREREG_digits.md", text)
    g.write("README_demo.md", readme)
    g.write("prereg-ci.toml", CFG)
    g.commit("seal")
    return g


# ---- M1: a gates fence the scanner does not count, beside the counted one or alone ------------------------

SHADOWS = {
    # the reader's table in a block quote, the machine's (a looser bar) in a collapsed <details> (B11q)
    "block-quote-beside-details": "# PREREG digits\n\nThe decision:\n\n" + quoted(fence()) + "\n\n"
                                  "<details><summary>appendix</summary>\n\n" + fence(moved(0.5)) + "\n\n</details>\n",
    # both visible, the reader's quoted (B11q2)
    "block-quote-beside-plain": "# PREREG digits\n\n" + quoted(fence()) + "\n\n" + fence(moved(0.5)) + "\n",
    # a fence 4 columns in under an ordered list item, rendered in the item (B11n)
    "ordered-item-4": "# PREREG digits\n\n1.  the decision:\n\n" + "\n".join("    " + ln for ln in fence().split("\n"))
                      + "\n\n" + fence(moved(0.5)) + "\n",
    # a second word in the info string, which CommonMark renders as the language gates (B11t)
    "second-word": "# PREREG digits\n\n" + fence(info="gates strict") + "\n\n" + fence(moved(0.5)) + "\n",
    # the one table, blockquoted: a reader sees a decision, the scanner none (B11z)
    "only-block-quote": "# PREREG digits\n\n" + quoted(fence()) + "\n",
}


@pytest.mark.parametrize("name", sorted(SHADOWS))
def test_a_gates_fence_a_renderer_shows_and_the_scanner_does_not_count_is_named_beside_the_scanners_reading(name):
    """``_decision`` keeps the scanner's reading (the conformance set pins it: vectors_compare.json's hidden
    fences read NOT_GATED); ``_shadowed`` names the fence the reader is shown, which check refuses."""
    why = core._shadowed(SHADOWS[name])
    assert why is not None and ("shadowing channel" in why or "inside a container" in why), why
    assert core._decision(SHADOWS[name])["code"] in (None, "NOT_GATED")


def test_a_plain_fence_alone_and_prose_quoting_another_language_name_no_shadow():
    text = "# PREREG digits\n\n> a note, quoted\n>\n> " + f"{F}json\n> {{}}\n> {F}" + "\n\n" + fence() + "\n"
    assert core._decision(text)["form"] == "gates" and core._shadowed(text) is None
    assert core._shadowed("# PREREG digits\n\n> quoted prose, no fence\n") is None
    assert core._shadowed(prereg()) is None and core._shadowed(b"\xff\xfe") is None


def test_a_blockquoted_only_table_does_not_read_as_not_gated_in_check(tmp_path, capsys):
    """B11z: with the one table blockquoted the PREREG read NOT_GATED, which is never refused, and a
    RESULT typing SHIP over HOLD numbers passed."""
    g = sealed(tmp_path, "z", SHADOWS["only-block-quote"])
    g("switch", "-q", "-c", "run")
    g.write("receipt.json", json.dumps({**scored(HOLD), "verdict": "SHIP"}, indent=1) + "\n")
    g.write("RESULT_digits.md", result_md("SHIP"))
    g.commit("run")
    code, out = run(g, "check", "--base", "main", "--head", "run", capsys=capsys)
    assert code == 1, out
    assert "FAIL FENCE_SHADOWED" in out and "renders as a gates block" in out, out


def test_a_blockquoted_decoy_added_to_a_sealed_prereg_is_decision_edited_after_seal(tmp_path, capsys):
    """B11a: the sealed fence untouched, a pull request adds a blockquoted table with another bar; C0 read
    it as prose (PROSE_AMENDED_IN_PLACE, a NOTE) and the check passed."""
    g = sealed(tmp_path, "a", prereg())
    g("switch", "-q", "-c", "decoy")
    g.write("PREREG_digits.md", prereg() + "\nThe bar, as a reader is shown it:\n\n" + quoted(fence(moved(0.01))) + "\n")
    g.commit("a decoy table")
    code, out = run(g, "check", "--base", "main", "--head", "decoy", capsys=capsys)
    assert code == 1, out
    assert "FAIL DECISION_EDITED_AFTER_SEAL" in out and "holds a second table beside it" in out, out


@pytest.mark.parametrize("name", ["block-quote-beside-details", "second-word"])
def test_a_shadowed_table_is_refused_where_the_receipt_is_checked(tmp_path, capsys, name):
    """B11q and B11t: the check printed PASS and the ladder of the looser table."""
    g = sealed(tmp_path, name, SHADOWS[name])
    g("switch", "-q", "-c", "run")
    rec = scored(HOLD, table=moved(0.5))
    g.write("receipt.json", json.dumps(rec, indent=1) + "\n")
    g.write("RESULT_digits.md", result_md(rec["verdict"]))
    g.commit("run")
    code, out = run(g, "check", "--base", "main", "--head", "run", capsys=capsys)
    assert code == 1 and "FAIL FENCE_SHADOWED" in out, out


def test_lint_refuses_a_draft_holding_a_shadowed_table(tmp_path, capsys):
    g = Git(tmp_path / "lint")
    g.write("README_demo.md", f"The model is {SENTENCE}.\n")
    g.write("prereg-ci.toml", CFG)
    g.commit("surfaces")
    g.write("PREREG_digits.md", SHADOWS["block-quote-beside-plain"])
    code, out = run(g, "lint", "PREREG_digits.md", capsys=capsys)
    assert code == 1 and "FAIL FENCE_SHADOWED" in out, out


# ---- M2: the seals manifest deleted, and re-added after a merged deletion -------------------------------------

MANIFEST = "prereg-ci.seals.jsonl"


def show(g: Git, rev: str, path: str) -> bytes:
    r = subprocess.run(["git", "--no-replace-objects", "show", f"{rev}:{path}"], cwd=str(g.root), capture_output=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def line(**kw) -> str:
    d = {"id": None, "kind": None, "path": None, "commit": None, "blob_sha256": None, "amends": None,
         "decision": "own", "seen": None, "time": []}
    d.update(kw)
    return json.dumps(d) + "\n"


def landed_with_receipt(tmp_path, name: str, metrics=None, prereg_name: str = "PREREG_digits.md",
                        readme: str = README_PLAIN) -> Git:
    """main: the PREREG sealed, then a merged pull request adding a receipt under it."""
    g = Git(tmp_path / name)
    g.write(prereg_name, prereg())
    g.write("README_demo.md", readme)
    g.write("prereg-ci.toml", CFG)
    g.commit("seal")
    g("switch", "-q", "-c", "result")
    g.write("receipt.json", json.dumps({**scored(metrics or HOLD), "prereg": prereg_name}, indent=1) + "\n")
    g.commit("result")
    g("switch", "-q", "main")
    g.merge("result", "Merge pull request #2 from result")
    return g


def with_amendment_line(tmp_path, name: str):
    """main: the PREREG and a HOLD receipt; then a merged pull request adding an own amendment, the manifest
    (its prereg line and the amendment line, seen "results seen") and a receipt under the amendment."""
    g = landed_with_receipt(tmp_path, name)
    land = g("rev-list", "--reverse", "main").split()[0]
    g("switch", "-q", "-c", "amend")
    g.write("AMENDMENT_digits.md", prereg(moved(0.10), prose="the ceiling is moved to 0.10"))
    am = g.commit("amendment")
    l1 = line(id="1", kind="prereg", path="PREREG_digits.md", commit=land,
              blob_sha256=hashlib.sha256(show(g, land, "PREREG_digits.md")).hexdigest())
    kw = dict(id="2", kind="amendment", path="AMENDMENT_digits.md", commit=am,
              blob_sha256=hashlib.sha256(show(g, am, "AMENDMENT_digits.md")).hexdigest(), amends="1")
    g.write(MANIFEST, l1 + line(seen="results seen", **kw))
    g.commit("seal the amendment, results seen")
    g.write("receipt_amended.json", json.dumps(scored(HOLD, table=moved(0.10)), indent=1) + "\n")
    g.commit("re-run under the amendment")
    g("switch", "-q", "main")
    g.merge("amend", "Merge pull request #3")
    return g, l1, kw


def test_a_pull_request_that_deletes_the_seals_manifest_is_manifest_rewritten(tmp_path, capsys):
    """O16d: with no manifest at HEAD the append-only reading never ran, and the deletion passed."""
    g, _, _ = with_amendment_line(tmp_path, "d")
    g("switch", "-q", "-c", "drop")
    (g.root / MANIFEST).unlink()
    g.commit("drop the manifest")
    code, out = run(g, "check", "--base", "main", "--head", "drop", capsys=capsys)
    assert code == 1, out
    assert "FAIL MANIFEST_REWRITTEN" in out and "is deleted in this pull request" in out, out


def test_a_manifest_re_added_after_a_merged_deletion_keeps_the_lines_it_held(tmp_path, capsys):
    """O16r: deleted in one merged pull request, re-added in the next with the amendment's seen restated
    (results seen -> nothing run); each check compared its merge base to HEAD only, and both passed."""
    g, l1, kw = with_amendment_line(tmp_path, "r")
    g("switch", "-q", "-c", "drop")
    (g.root / MANIFEST).unlink()
    g.commit("drop the manifest")
    g("switch", "-q", "main")
    g.merge("drop", "Merge pull request #4")
    g("switch", "-q", "-c", "restate")
    g.write(MANIFEST, l1 + line(seen="nothing run", **kw))
    g.commit("the manifest again, seen restated")
    code, out = run(g, "check", "--base", "main", "--head", "restate", capsys=capsys)
    assert code == 1, out
    assert "FAIL MANIFEST_REWRITTEN" in out and "held it on the base branch" in out, out
    g("switch", "-q", "-c", "restore", "main")
    g.write(MANIFEST, l1 + line(seen="results seen", **kw))
    g.commit("the manifest again, as it was")
    code, out = run(g, "check", "--base", "main", "--head", "restore", capsys=capsys)
    assert "MANIFEST_REWRITTEN" not in out, out


def test_a_manifest_added_where_the_base_never_held_one_is_not_compared(tmp_path, capsys):
    g = landed_with_receipt(tmp_path, "new")
    land = g("rev-list", "--reverse", "main").split()[0]
    g("switch", "-q", "-c", "man")
    g.write(MANIFEST, line(id="1", kind="prereg", path="PREREG_digits.md", commit=land,
                           blob_sha256=hashlib.sha256(show(g, land, "PREREG_digits.md")).hexdigest()))
    g.commit("start the manifest")
    code, out = run(g, "check", "--base", "main", "--head", "man", capsys=capsys)
    assert code == 0 and "MANIFEST_REWRITTEN" not in out, out


# ---- M3: a receipt kept under a suffix other than .json --------------------------------------------------------

def test_a_typed_receipt_under_another_suffix_is_refused_where_it_is_written(tmp_path, capsys):
    """O50: a typed SHIP over HOLD numbers in receipt2.jsonc, no RESULT: C1 read *.json only, and it passed."""
    g = sealed(tmp_path, "jsonc", prereg())
    g("switch", "-q", "-c", "run")
    g.write("receipt2.jsonc", json.dumps({**scored(HOLD), "verdict": "SHIP"}, indent=1) + "\n")
    g.commit("run")
    code, out = run(g, "check", "--base", "main", "--head", "run", capsys=capsys)
    assert code == 1, out
    assert "FAIL RECEIPT_NOT_JSON" in out and "receipt2.jsonc" in out, out


@pytest.mark.parametrize("name", ["receipt.txt", "receipt_old.jsonc"])
def test_a_committed_receipt_under_another_suffix_is_held_immutable(tmp_path, capsys, name):
    """O07, O07b: a receipt paired through its RESULT when it landed, rewritten in place later, passed."""
    g = sealed(tmp_path, "txt", prereg())
    g("switch", "-q", "-c", "result")
    rec = scored(SHIP)
    g.write(name, json.dumps(rec, indent=1) + "\n")
    g.write("RESULT_digits.md", result_md(rec["verdict"], receipt=name))
    g.commit("result")
    g("switch", "-q", "main")
    g.merge("result", "Merge pull request #2")
    g("switch", "-q", "-c", "rewrite")
    g.write(name, json.dumps(scored({"acc": 0.95, "ece": 0.04}), indent=1) + "\n")
    g.commit("rewrite")
    code, out = run(g, "check", "--base", "main", "--head", "rewrite", capsys=capsys)
    assert code == 1, out
    assert "FAIL RECEIPT_REWRITTEN" in out and f"{name}, committed at" in out, out
    g("switch", "-q", "-c", "delete", "main")
    (g.root / name).unlink()
    g.commit("delete it")
    code, out = run(g, "check", "--base", "main", "--head", "delete", capsys=capsys)
    assert code == 1 and "FAIL RECEIPT_REWRITTEN" in out and "is deleted" in out, out


def test_text_files_that_hold_no_receipt_are_left_alone(tmp_path, capsys):
    g = sealed(tmp_path, "plain", prereg())
    g("switch", "-q", "-c", "docs")
    g.write("notes.txt", "{ not json\n")
    g.write("config.jsonc", json.dumps({"name": "x", "verdict": "SHIP"}) + "\n")
    g.write("list.txt", json.dumps(["PREREG_digits.md"]) + "\n")
    g.write("doc.md", "# a\n")
    g.commit("docs")
    code, out = run(g, "check", "--base", "main", "--head", "docs", capsys=capsys)
    assert code == 0 and "RECEIPT_NOT_JSON" not in out, out


# ---- M4: a PREREG kept under a suffix other than .md, named by a receipt ---------------------------------------

@pytest.mark.parametrize("name", ["PREREG_digits.markdown", "PREREG_digits.txt"])
def test_a_bar_moved_in_a_sealed_prereg_under_another_suffix_is_decision_edited_after_seal(tmp_path, capsys, name):
    """O20, O20t: C0 read *.md only while the resolver pairs a receipt with a PREREG of any name: '0 moved', PASS."""
    g = landed_with_receipt(tmp_path, "moved", prereg_name=name)
    g("switch", "-q", "-c", "move")
    g.write(name, prereg(moved(0.10)))
    g.commit("move the bar")
    code, out = run(g, "check", "--base", "main", "--head", "move", capsys=capsys)
    assert code == 1, out
    assert "FAIL DECISION_EDITED_AFTER_SEAL" in out and name in out, out


def test_a_withdrawn_sentence_back_on_a_surface_is_refused_for_a_prereg_under_another_suffix(tmp_path, capsys):
    """O20l: the tree's licence pass grepped *.md only, so a .markdown PREREG's landed HOLD row withdrew
    nothing on later pull requests."""
    g = landed_with_receipt(tmp_path, "lic", prereg_name="PREREG_digits.markdown")
    g("switch", "-q", "-c", "claim")
    g.write("README_demo.md", f"# demo\n\nThe model is {SENTENCE}.\n")
    g.commit("the sentence again")
    code, out = run(g, "check", "--base", "main", "--head", "claim", capsys=capsys)
    assert code == 1, out
    assert "FAIL LICENCE_WITHDRAWN" in out and "README_demo.md" in out, out


def test_a_file_outside_md_that_holds_a_fence_and_is_named_by_no_receipt_is_not_a_sealed_decision(tmp_path, capsys):
    """tests/test_protocol.py on styxx holds a fence the scanner reads; a pull request editing such a file is
    not C0's, unless a receipt names it."""
    g = sealed(tmp_path, "py", prereg())
    g.write("tests/test_fence.py", 'TEXT = """\n' + fence() + '\n"""\n')
    g.commit("a test file")
    g("switch", "-q", "-c", "edit")
    g.write("tests/test_fence.py", 'TEXT = """\n' + fence(moved(0.2)) + '\n"""\n')
    g.commit("edit the test")
    code, out = run(g, "check", "--base", "main", "--head", "edit", capsys=capsys)
    assert code == 0 and "DECISION_EDITED_AFTER_SEAL" not in out, out


# ---- F1: a locks PREREG that arrives in the pull request, whose scorer is the pull request's own code ----------

LOCK_SCORER = ("import json, sys\n"
               "r = json.load(open(sys.argv[1], encoding='utf-8'))\n"
               "{pre}v = 'INVALID__accuracy_floor' if r['acc'] < 0.9 else ('SHIP' if r['ece'] <= 0.05 else "
               "'HOLD__calibration')\n"
               "print(json.dumps({{'verdict': v}}))\n")
LOCKS = {"scorer": "score.py", "reads": [], "outcomes": ["INVALID__accuracy_floor", "SHIP", "HOLD__calibration"]}


def locks_prereg() -> str:
    return f"# PREREG locks\n\nthe plan\n\n{F}locks\n{json.dumps(LOCKS, indent=1)}\n{F}\n"


def test_a_locks_prereg_that_arrives_in_the_pull_request_does_not_run_its_scorer(tmp_path, capsys):
    """The fresh review's P2: PREREG, scorer and receipt in one pull request; the scorer ran as the judge,
    so the pull request chose the code that judged it (DESIGN 3.2(f)), and the check printed PASS."""
    g = Git(tmp_path / "pr")
    g.write("README_demo.md", README_PLAIN)
    g.write("prereg-ci.toml", CFG)
    g.commit("init")
    g("switch", "-q", "-c", "pr")
    g.write("PREREG_locks.md", locks_prereg())
    g.write("score.py", LOCK_SCORER.format(pre="r['acc'] = 1.0\nr['ece'] = 0.0\n"))
    g.commit("plan and scorer")
    g.write("receipt.json", json.dumps({"prereg": "PREREG_locks.md", **HOLD, "verdict": "SHIP"}, indent=1) + "\n")
    g.commit("result")
    code, out = run(g, "check", "--base", "main", "--head", "pr", capsys=capsys)
    assert code == 3, out
    assert "UNCHECKABLE SCORER_NOT_RUN" in out and "the base" in out and "does not hold" in out, out
    assert "locked scorer score.py" not in out, out


def test_a_landed_locks_prereg_still_runs_its_sealed_scorer_and_an_edit_in_the_pull_request_does_not_judge(
        tmp_path, capsys):
    g = Git(tmp_path / "landed")
    g.write("README_demo.md", README_PLAIN)
    g.write("prereg-ci.toml", CFG)
    g.write("PREREG_locks.md", locks_prereg())
    g.write("score.py", LOCK_SCORER.format(pre=""))
    g.commit("seal")
    g("switch", "-q", "-c", "run")
    g.write("score.py", LOCK_SCORER.format(pre="r['ece'] = 0.0\n"))
    g.write("receipt.json", json.dumps({"prereg": "PREREG_locks.md", **HOLD, "verdict": "SHIP"}, indent=1) + "\n")
    g.commit("a looser judge and a typed receipt")
    code, out = run(g, "check", "--base", "main", "--head", "run", capsys=capsys)
    assert code == 1 and "FAIL VERDICT_DIFFERS" in out and "NOTE LOCKED_FILE_MOVED" in out, out


def test_the_action_says_a_locked_scorer_runs_as_code_and_where_the_action_must_not_run():
    """The fresh review's F1: the Action said "contents: read" and "opens no connection", and nothing of the
    code a locks PREREG runs, nor of the token its checkout keeps."""
    import yaml
    desc = yaml.safe_load((ROOT / "prereg-ci" / "action.yml").read_text(encoding="utf-8"))["description"]
    for words in ("runs as code", "not a sandbox", "SCORER_NOT_RUN", "pull_request_target", "no secret"):
        assert words in desc, words


# ---- F2: author bytes cannot open a line of the printout with a workflow command ---------------------------------

def test_a_receipt_verdict_carrying_a_newline_and_a_workflow_command_stays_on_its_line(tmp_path, capsys):
    """The fresh review's P1: a receipt verdict "SHIP\\n::error title=prereg-ci::...PASS" printed two lines
    that opened with ::error, which a runner reads as annotations."""
    g = sealed(tmp_path, "inj", prereg())
    g("switch", "-q", "-c", "run")
    rec = {**scored(HOLD), "verdict": "SHIP\n::error title=prereg-ci::every receipt re-derived, PASS"}
    g.write("receipt.json", json.dumps(rec, indent=1) + "\n")
    g.write("RESULT_digits.md", result_md("SHIP\r\n::notice::PASS"))
    g.commit("run")
    code, out = run(g, "check", "--base", "main", "--head", "run", capsys=capsys)
    assert code == 1, out
    for ln in out.splitlines():
        assert not ln.lstrip().startswith("::"), ln
    assert "SHIP\\n::error title=prereg-ci" in out, out


@pytest.mark.parametrize("raw, shown", [
    ("plain words", "plain words"),
    ("a\ttab stays", "a\ttab stays"),
    ("one\ntwo", "one\\ntwo"),
    ("one\r\ntwo", "one\\r\\ntwo"),
    ("  ::warning::x", "  \\x3a:warning::x"),
    ("::add-mask::y", "\\x3a:add-mask::y"),
    ("text ::notice::z", "text ::notice::z"),
    ("a ##[error]b", "a ##\\x5berror]b"),
    ("nel\x85line sep\x1b[0m", "nel\\x85line\\u2028sep\\x1b[0m"),
])
def test_every_printed_line_is_one_line_with_no_command_at_its_head(raw, shown):
    from styxx.prereg_ci import report
    got = report._printable(raw)
    assert got == shown
    assert len(got.splitlines()) == 1 and not got.lstrip().startswith("::")


def test_the_action_stops_workflow_commands_around_the_check_and_resumes_after(tmp_path):
    import yaml
    action = yaml.safe_load((ROOT / "prereg-ci" / "action.yml").read_text(encoding="utf-8"))
    (step,) = [s for s in action["runs"]["steps"] if s.get("id") == "check"]
    script = step["run"]
    stop, call, resume = (script.index('echo "::stop-commands::$token"'), script.index("python -I -X utf8 -m styxx.prereg_ci"),
                          script.index('echo "::$token::"'))
    assert stop < call < resume, script
    assert "/dev/urandom" in script and '-ne 32' in script


# ---- M5 (named, not refused in v0): what the locked scorer run does not watch -----------------------------------

def test_boundary_a_locked_scorer_whose_verdict_reads_the_clock_passes_and_the_boundary_says_so(tmp_path, capsys):
    """B10c pinned as the boundary, not the contract (KNOWN_MISSES.md, M5): a scorer that types SHIP after a
    date is not refused in v0; the printout names the clock beside the PASS."""
    g = Git(tmp_path / "clock")
    g.write("README_demo.md", README_PLAIN)
    g.write("prereg-ci.toml", CFG)
    g.write("PREREG_locks.md", locks_prereg())
    g.write("score.py", LOCK_SCORER.format(pre="import time\nr['ece'] = 0.0 if time.time() > 1.7e9 else r['ece']\n"))
    g.commit("seal")
    g("switch", "-q", "-c", "run")
    g.write("receipt.json", json.dumps({"prereg": "PREREG_locks.md", **HOLD, "verdict": "SHIP"}, indent=1) + "\n")
    g.commit("result")
    code, out = run(g, "check", "--base", "main", "--head", "run", capsys=capsys)
    assert code == 0, out
    assert "the locked scorer's clock, entropy and environment: not watched" in out, out


# ---- the list the README prints ----------------------------------------------------------------------------

KNOWN = ROOT / "prereg-ci" / "KNOWN_MISSES.md"


def test_the_known_misses_list_names_what_v0_does_not_refuse_and_the_case_that_pins_it():
    from tests.test_prereg_ci_package import HOUSE_RX, PRIORITY_RX
    raw = KNOWN.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")
    assert not HOUSE_RX.search(text) and not PRIORITY_RX.search(text)
    for word in ("M5", "FENCE_SHADOWED", "RECEIPT_NOT_JSON", "SCORER_NOT_RUN", "pull_request_target"):
        assert word in text, word
    for case in ("O03", "O11", "O32r", "O32", "O31", "O30z", "O30s", "O30n", "O50b", "O39", "O39e",
                 "O29", "O30", "O34", "O47", "O53", "O33n", "B06i"):
        assert case in text, case
    pinned = "test_boundary_a_locked_scorer_whose_verdict_reads_the_clock_passes_and_the_boundary_says_so"
    assert pinned in text and pinned in globals()
