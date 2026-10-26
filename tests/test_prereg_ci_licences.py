# -*- coding: utf-8 -*-
"""prereg-ci B5: licences (``withdraws``), the surface scan, and the lint's pattern coverage.

DESIGN 3.2(g): an outcome row (or the locks block's ``withdraws`` map) may carry ``"withdraws":
[{"id", "quote", "pattern"}]``; ``quote`` is the sentence as the PREREG writes it, ``pattern`` a
case-insensitive regular expression over whitespace-collapsed text. When a receipt lands on that row,
every later check scans the shipped surfaces (surface_lint's ``repo`` profile, or the repository's
configured globs) and refuses a match with no withdrawal within 300 characters (``LICENCE_WITHDRAWN``).
Before any receipt lands, a match prints ``NOTE LICENCE_PENDING``; a pattern that matches nothing at
seal time is a lint refusal (``LICENCE_PATTERN_FINDS_NOTHING``), the licence analogue of a vacuous gate.
PLAN K14: the profile is copied into core (``scripts/`` is not in the wheel) and pinned equal to
``scripts/surface_lint.py``'s ``PROFILES["repo"]``. PLAN K15: curly apostrophes are straightened too.

Done when (PLAN B5): at ``414db45a`` the module reads 328 surfaces, finds the PREREG's verbatim quote
on 0 and its last five words on 4, the same 4 the hand rule ``withdrawn/gate-claim`` finds; at
``3437340b^1``, 394, 0 and 3 against the hand rule's 10 (RV3; the hand rule is taken from
``scripts/surface_lint.py`` here, never from prereg-ci); RV4's verdict, gates, digest and ``withdraws``
come back through ``core.rescore`` under the RECONSTRUCTED table copied under ``conformance/``.

The history tests read styxx's own objects and skip, saying why, where the commits are not in the
object store. Every other repository here is a ``git init`` in tmp_path with no remote.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "conformance" / "prereg_ci"
TABLE = SET / "reconstructed" / "external1_reconstructed.md"
# sha256 of revision_probes/external1_reconstructed.md, the design's RV4 table, copied byte for byte
TABLE_SHA256 = "fe6454d54cb0228ecea2231d8fd558e6b62eb89dd0fba8bfd37c260b2fc07840"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.prereg_ci import core  # noqa: E402
from styxx.prereg_ci import git as G  # noqa: E402
from styxx.prereg_ci import scorer_run as S  # noqa: E402
from tests.test_prereg_ci_git import Git  # noqa: E402


def _surface_lint():
    """scripts/surface_lint.py, loaded from its file: the hand rule's source, never prereg-ci's."""
    spec = importlib.util.spec_from_file_location("surface_lint_b5", ROOT / "scripts" / "surface_lint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SL = _surface_lint()
F = "`" * 3
QUOTE = "an agent's PR summary cannot lie about its diff"           # PREREG_external1_aidev_2026_08_31.md:93-95
FIVE = "cannot lie about its diff"                                  # its last five words, the RV3 pattern
HAND_RULE = "withdrawn/gate-claim"
RESULT_COMMIT = "414db45a7914af2659c9b127a613d93b87aded84"          # the EXTERNAL-1 RESULT's add commit
BEFORE_221 = "d5f8ac7d5e6ab27eaac9133b6d738334554afddc"             # 3437340b^1, the parent of the #221 merge
RECEIPT = "papers/closed-model-frontier/external1_adjudication.json"
RV4_DIGEST = "bd6f4256dcd01ccbfdcfa9864c312d1c008e0a5fd8aebed29486cb3321311841"
FAILED = "FAILURE__gate_claim_not_yet_true"
GATE_CLAIM = [{"id": "gate-claim", "quote": QUOTE, "pattern": FIVE}]
WITHDRAWN, PENDING, NOTHING = "LICENCE_WITHDRAWN", "LICENCE_PENDING", "LICENCE_PATTERN_FINDS_NOTHING"

LIC = {"id": "ship-claim", "quote": "The model is calibrated well enough to ship.",
       "pattern": "calibrated well enough to ship"}
LIC2 = {"id": "floor-claim", "quote": "Accuracy never drops under the floor.", "pattern": r"never\s+drops"}
EXTS = core._SURFACE_EXTS if hasattr(core, "_SURFACE_EXTS") else frozenset()


def table(withdraws=None):
    spec = {"gates": {"G": {"metric": "m", "op": ">=", "value": 0.5}},
            "outcomes": [{"when": {"G": True}, "verdict": "SHIP"}, {"when": {"G": False}, "verdict": "HOLD"}],
            "smoke_verdict": "SMOKE"}
    for verdict, lics in (withdraws or {}).items():
        next(r for r in spec["outcomes"] if r["verdict"] == verdict)["withdraws"] = lics
    return spec


def prereg(spec=None, **withdraws) -> str:
    spec = spec if spec is not None else table(withdraws or None)
    return f"# PREREG\n\nprose\n\n{F}gates\n{json.dumps(spec, indent=1)}\n{F}\n"


def locks_prereg(withdraws=None) -> str:
    spec = {"scorer": "score.py", "reads": [], "outcomes": ["SHIP", "HOLD"]}
    if withdraws is not None:
        spec["withdraws"] = withdraws
    return f"# PREREG\n\nprose\n\n{F}locks\n{json.dumps(spec, indent=1)}\n{F}\n"


def specs(text):
    return core._licence_specs(core._decision(text))


def codes(reading):
    return [f["code"] for f in reading["findings"]]


def _have(repo, *revs) -> bool:
    try:
        for rev in revs:
            repo.commit(rev)
    except G._Refusal:
        return False
    return True


# ---- the public surface ----------------------------------------------------------------------------------

def test_core_and_the_package_export_licences_beside_rescore_and_compare():
    import styxx.prereg_ci as P
    assert core.__all__ == ["rescore", "compare", "licences"]
    assert set(P.__all__) == {"rescore", "compare", "licences"} and P.licences is core.licences
    for code, kind in ((WITHDRAWN, "FAIL"), (PENDING, "NOTE"), (NOTHING, "FAIL")):
        assert core._KIND[code] == kind, code


# ---- the licences a decision declares (DESIGN 3.2(g)) --------------------------------------------------

def test_a_gates_row_s_withdraws_read_as_licences_in_table_order():
    got = specs(prereg(HOLD=[LIC, LIC2]))
    assert [(s["row"], s["verdict"], s["id"], s["quote"], s["pattern"]) for s in got] == [
        (1, "HOLD", "ship-claim", LIC["quote"], LIC["pattern"]), (1, "HOLD", "floor-claim", LIC2["quote"],
                                                                 LIC2["pattern"])]
    assert all(s["rx"].flags & re.IGNORECASE for s in got)
    assert specs(prereg()) == []


def test_a_locks_block_s_withdraws_map_reads_as_licences_in_outcome_order():
    got = specs(locks_prereg({"HOLD": [LIC], "SHIP": [LIC2]}))
    assert [(s["row"], s["verdict"], s["id"]) for s in got] == [(0, "SHIP", "floor-claim"), (1, "HOLD", "ship-claim")]
    assert specs(locks_prereg()) == [] and specs(locks_prereg({})) == []


@pytest.mark.parametrize("withdraws, words", [
    ("x", "list"),
    (["x"], "object"),
    ([dict(LIC, note="n")], "keys"),
    ([{"id": "a", "quote": "q"}], "pattern"),
    ([{"quote": "q", "pattern": "p"}], "id"),
    ([{"id": "a", "pattern": "p"}], "quote"),
    ([dict(LIC, id="")], "id"),
    ([dict(LIC, id=" a")], "id"),
    ([dict(LIC, id="a\nb")], "id"),
    ([dict(LIC, id=5)], "id"),
    ([LIC, dict(LIC2, id="ship-claim")], "twice"),
    ([dict(LIC, quote="")], "quote"),
    ([dict(LIC, quote=5)], "quote"),
    ([dict(LIC, pattern="")], "pattern"),
    ([dict(LIC, pattern=5)], "pattern"),
    ([dict(LIC, pattern="(")], "compile"),
    ([dict(LIC, pattern="a*")], "empty"),
    ([dict(LIC, pattern="(?=x)")], "empty"),
    ([dict(LIC, pattern="x" * 1001)], "long"),
])
def test_a_licence_the_format_does_not_read_is_refused(withdraws, words):
    with pytest.raises(core._GateSpecError) as e:
        specs(prereg(HOLD=withdraws))
    assert words in str(e.value), str(e.value)
    with pytest.raises(core._GateSpecError) as e2:
        specs(locks_prereg({"HOLD": withdraws}))
    assert words in str(e2.value), str(e2.value)


def test_a_licence_id_repeated_across_rows_is_refused():
    with pytest.raises(core._GateSpecError) as e:
        specs(prereg(SHIP=[LIC], HOLD=[dict(LIC2, id=LIC["id"])]))
    assert "twice" in str(e.value)


def test_a_table_score_frozen_refuses_is_refused_before_its_licences_are_read():
    spec = table({"HOLD": [LIC]})
    spec["outcomes"][0]["when"] = {"Z": True}
    with pytest.raises(core._GateSpecError) as e:
        specs(prereg(spec))
    assert "does not declare" in str(e.value)


# ---- normalisation (DESIGN 3.2(g), PLAN K15) ----------------------------------------------------------

def test_collapse_straightens_curly_apostrophes_and_collapses_whitespace_runs_and_maps_back():
    c = core._collapse("a’b  c\r\nd\te")
    assert c.text == "a'b c d e"
    assert c.source == "a'b  c\nd\te"
    assert [c.at(i) for i in range(len(c.text))] == [0, 1, 2, 3, 5, 6, 7, 8, 9]
    c = core._collapse("  lead\n")
    assert c.text == " lead " and c.at(0) == 0 and c.at(1) == 2 and c.at(5) == 6
    assert core._collapse("").text == ""


def test_line_and_column_are_read_from_the_source_text():
    text = "ab\ncd\r\nef"
    src = core._collapse(text).source
    assert core._line_col(src, 0) == (1, 1) and core._line_col(src, 3) == (2, 1) and core._line_col(src, 4) == (2, 2)
    assert core._line_col(src, 6) == (3, 1)


# ---- the scan of one surface -------------------------------------------------------------------------

def test_a_match_is_case_insensitive_over_collapsed_text_and_named_by_line_and_column():
    sp = specs(prereg(HOLD=[LIC]))
    got = core._scan("# Model\n\nOur model is CALIBRATED  well\nenough to ship, the card says.\n", sp, "README.md")
    assert len(got) == 1
    m = got[0]
    assert (m["id"], m["path"], m["line"], m["col"], m["withdrawn"]) == ("ship-claim", "README.md", 3, 14, False)
    assert m["text"] == "CALIBRATED well enough to ship"


def test_a_pattern_with_its_own_whitespace_class_matches_the_collapsed_text():
    sp = specs(prereg(HOLD=[LIC2]))
    got = core._scan("Accuracy never\n\n\tdrops.\n", sp, "a.md")
    assert [(m["line"], m["col"], m["text"]) for m in got] == [(1, 10, "never drops")]


def test_curly_apostrophes_on_the_surface_match_a_straight_one_in_the_pattern():
    sp = specs(prereg(HOLD=[{"id": "q", "quote": QUOTE, "pattern": re.escape("agent's pr summary")}]))
    got = core._scan("An agent’s PR summary is read.\n", sp, "a.md")
    assert [m["text"] for m in got] == ["agent's PR summary"]
    assert core._scan("An agent‘s PR summary is read.\n", sp, "a.md")[0]["text"] == "agent's PR summary"


def test_a_withdrawal_within_the_window_marks_the_match_withdrawn_and_outside_it_does_not():
    sp = specs(prereg(HOLD=[LIC]))
    assert core._WINDOW == 300
    pad = "x" * 290
    near = f"The sentence 'calibrated well enough to ship' is withdrawn. {pad}\n"
    far = f"calibrated well enough to ship. {'y' * 300} withdrawn\n"
    # the withdrawal must lie whole inside the window: 290 filler characters, a space (the word's boundary, as
    # surface_lint's rule has it) and the 9 characters of "withdrawn" end at the edge; one more filler does not fit
    exact = "calibrated well enough to ship" + "z" * 290 + " withdrawn\n"
    over = "calibrated well enough to ship" + "z" * 291 + " withdrawn\n"
    phrase = "calibrated well enough to ship; this is not yet true of this instrument.\n"
    assert [m["withdrawn"] for m in core._scan(near, sp, "a.md")] == [True]
    assert [m["withdrawn"] for m in core._scan(far, sp, "a.md")] == [False]
    assert [m["withdrawn"] for m in core._scan(exact, sp, "a.md")] == [True]
    assert [m["withdrawn"] for m in core._scan(over, sp, "a.md")] == [False]
    assert [m["withdrawn"] for m in core._scan(phrase, sp, "a.md")] == [True]
    for word, ok in (("withdrew", True), ("withdrawal", True), ("withdrawing", True), ("withdrawals", False),
                     ("WITHDRAWN", True)):
        assert core._scan(f"calibrated well enough to ship ({word})\n", sp, "a.md")[0]["withdrawn"] is ok, word


def test_the_window_rule_and_the_withdrawal_words_are_surface_lint_s():
    assert core._WINDOW == SL.WINDOW
    assert core._WITHDRAWAL.pattern == SL._WITHDRAWAL_RX.pattern and core._WITHDRAWAL.flags == SL._WITHDRAWAL_RX.flags


def test_matches_are_sorted_by_path_line_and_column_and_every_match_is_listed():
    sp = specs(prereg(HOLD=[LIC]))
    text = "calibrated well enough to ship\nand calibrated   well enough to ship again\n"
    got = core._scan(text, sp, "a.md")
    assert [(m["line"], m["col"]) for m in got] == [(1, 1), (2, 5)]


# ---- licences: the reading (DESIGN 3.4 C8, 3.7) -----------------------------------------------------------

SURFACES = [
    ("README.md", "# Demo\n\nThe model is calibrated well enough to ship.\n"),
    ("docs/notes.md", "Nothing here.\n"),
    ("web/claim.html", "<p>Calibrated well\nenough to ship, we say; the ship-claim is withdrawn below.</p>\n"),
    ("styxx/bin.py", None),
]


def test_before_any_receipt_lands_a_match_is_pending_and_a_pattern_finding_nothing_is_refused():
    r = core.licences(prereg(HOLD=[LIC, LIC2]), SURFACES, landed=None)
    assert (r["form"], r["code"], r["landed"], r["surfaces"], r["skipped"]) == ("gates", None, None, 3, ["styxx/bin.py"])
    assert [(li["id"], li["surfaces"]) for li in r["licences"]] == [("ship-claim", ["README.md", "web/claim.html"]),
                                                                   ("floor-claim", [])]
    assert codes(r) == [PENDING, NOTHING]
    pending, nothing = r["findings"]
    assert (pending["kind"], pending["licence"], pending["path"], pending["line"], pending["col"]) == (
        "NOTE", "ship-claim", "README.md", 3, 14)
    assert "no receipt has landed" in pending["detail"] and "README.md:3:14" in pending["detail"]
    assert (nothing["kind"], nothing["licence"], nothing["path"]) == ("FAIL", "floor-claim", None)
    assert "matches none of the 3 surfaces" in nothing["detail"] and "vacuous" in nothing["detail"]
    assert r["licences"][0]["matches"][1]["withdrawn"] is True       # web/claim.html: withdrawn beside the match


def test_a_receipt_landed_on_the_licensed_row_withdraws_every_unwithdrawn_match():
    r = core.licences(prereg(HOLD=[LIC]).encode("utf-8"), SURFACES, landed="HOLD")
    assert codes(r) == [WITHDRAWN]
    f = r["findings"][0]
    assert (f["kind"], f["licence"], f["row"], f["verdict"], f["path"], f["line"], f["col"]) == (
        "FAIL", "ship-claim", 1, "HOLD", "README.md", 3, 14)
    assert "README.md:3:14" in f["detail"] and "HOLD" in f["detail"] and "300 characters" in f["detail"]
    assert "calibrated well enough to ship" in f["detail"]
    matched = r["licences"][0]["matches"]
    assert [(m["path"], m["withdrawn"]) for m in matched] == [("README.md", False), ("web/claim.html", True)]


def test_a_receipt_landed_on_another_row_or_a_smoke_run_withdraws_nothing_and_the_matches_are_still_listed():
    for landed in ("SHIP", "SMOKE"):
        r = core.licences(prereg(HOLD=[LIC]), SURFACES, landed=landed)
        assert codes(r) == [] and r["landed"] == landed, landed
        assert [m["path"] for m in r["licences"][0]["matches"]] == ["README.md", "web/claim.html"]


def test_a_landed_verdict_the_decision_does_not_declare_is_a_caller_error():
    with pytest.raises(ValueError):
        core.licences(prereg(HOLD=[LIC]), SURFACES, landed="NOPE")
    with pytest.raises(ValueError):
        core.licences(locks_prereg({"HOLD": [LIC]}), SURFACES, landed="NOPE")


def test_a_locks_decision_s_licences_read_the_same_way():
    r = core.licences(locks_prereg({"HOLD": [LIC]}), SURFACES, landed="HOLD")
    assert r["form"] == "locks" and codes(r) == [WITHDRAWN]
    r = core.licences(locks_prereg({"HOLD": [LIC]}), SURFACES, landed="SHIP")
    assert codes(r) == []


def test_a_prereg_with_no_decision_or_a_refused_one_is_its_own_finding_and_scans_nothing():
    r = core.licences("# PREREG\n\nprose only\n", SURFACES, landed=None)
    assert (r["form"], r["code"]) == (None, "NOT_GATED") and codes(r) == ["NOT_GATED"]
    assert r["licences"] == [] and r["findings"][0]["kind"] == "NOTE"
    r = core.licences(prereg(HOLD=[dict(LIC, pattern="(")]), SURFACES, landed=None)
    assert r["code"] == "SCORER_REFUSES" and codes(r) == ["SCORER_REFUSES"] and "compile" in r["detail"]
    assert r["findings"][0]["kind"] == "FAIL" and r["licences"] == []
    r = core.licences(prereg(HOLD=[LIC]) + f"\n{F}locks\n{{}}\n{F}\n", SURFACES, landed=None)
    assert r["code"] == "SCORER_REFUSES" and "one decision" in r["detail"]


def test_rescore_s_withdraws_key_is_the_landed_row_s_list():
    assert "withdraws" in core._RESCORE_KEYS
    r = core.rescore(prereg(HOLD=[LIC]), {"m": 0.2})
    assert (r["verdict"], r["row"], r["withdraws"]) == ("HOLD", 1, [LIC])
    r = core.rescore(prereg(HOLD=[LIC]), {"m": 0.9})
    assert (r["verdict"], r["row"], r["withdraws"]) == ("SHIP", 0, [])
    assert core.rescore(prereg(HOLD=[LIC]), {"m": 0.2, "smoke": True})["withdraws"] == []
    assert core.rescore(prereg(HOLD=[LIC]), {"z": 1})["withdraws"] is None
    assert core.rescore(locks_prereg({"HOLD": [LIC]}), {"m": 0.2})["withdraws"] is None
    assert core.rescore("# PREREG\n\nprose\n", {"m": 0.2})["withdraws"] is None
    # rescore returns the row's value as written; licences reads it strictly
    assert core.rescore(prereg(HOLD=[dict(LIC, pattern="(")]), {"m": 0.2})["withdraws"] == [dict(LIC, pattern="(")]


def test_the_locked_scorer_run_fills_withdraws_from_the_outcome_it_read():
    text = locks_prereg({"SHIP": [LIC]}).encode("utf-8")
    files = {"score.py": b'print(\'{"verdict": "SHIP"}\')\n'}
    r = S._rescore_locked(text, "r.json", b"{}", files)
    assert (r["code"], r["verdict"], r["withdraws"]) == (None, "SHIP", [LIC]), r
    r = S._rescore_locked(text, "r.json", b"{}", {"score.py": b'print(\'{"verdict": "HOLD"}\')\n'})
    assert (r["code"], r["verdict"], r["withdraws"]) == (None, "HOLD", []), r
    r = S._rescore_locked(text, "r.json", b"{}", {"score.py": b"raise SystemExit(3)\n"})
    assert r["code"] == "SCORER_ERROR" and r["withdraws"] is None


def test_surfaces_are_pairs_of_path_and_text_or_none():
    with pytest.raises(TypeError):
        core.licences(prereg(HOLD=[LIC]), [("a.md", b"bytes")], landed=None)
    with pytest.raises(TypeError):
        core.licences(prereg(HOLD=[LIC]), ["a.md"], landed=None)
    r = core.licences(prereg(HOLD=[LIC]), iter(SURFACES), landed="HOLD")
    assert codes(r) == [WITHDRAWN]


def test_findings_follow_licence_order_then_path_line_and_column():
    surfaces = [("b.md", "calibrated well enough to ship\n"), ("a.md", "never drops\ncalibrated well enough to ship\n")]
    r = core.licences(prereg(HOLD=[LIC, LIC2]), surfaces, landed="HOLD")
    assert [(f["licence"], f["path"], f["line"]) for f in r["findings"]] == [
        ("ship-claim", "a.md", 2), ("ship-claim", "b.md", 1), ("floor-claim", "a.md", 1)]


# ---- the surface profile (PLAN K14) ----------------------------------------------------------------------

def test_the_copied_profile_equals_surface_lint_s_repo_profile():
    repo = SL.PROFILES["repo"]
    assert core._SURFACE_PROFILE["include"] == repo["include"]
    assert core._SURFACE_PROFILE["exclude"] == repo["exclude"]
    assert core._SURFACE_PROFILE["exts"] == repo["exts"] == core._SURFACE_EXTS
    assert core._SURFACE_PRUNE == SL.PRUNE
    assert core._SURFACE_MAX_BYTES == SL.MAX_BYTES


def test_selection_agrees_with_surface_lint_on_every_tracked_path_and_the_lint_test_s_samples():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    paths = [p.decode("utf-8", "surrogateescape") for p in out.split(b"\0") if p]
    assert len(paths) > 1000
    repo = SL.PROFILES["repo"]
    disagree = [p for p in paths if core._surface_selected(p, core._SURFACE_PROFILE) != SL.selected(p, repo)]
    assert disagree == []
    for path in ("papers/closed-model-frontier/PREREG_external1_aidev_2026_08_31.md", "CHANGELOG.md",
                 "tests/test_diffgate.py", "web/gate/differential/bodies/pr98.md", "styxx/_data/x.json",
                 "styxx/__pycache__/diffgate.cpython-312.py", "styxx/a.b.c", "styxx/x.", "notes.txt"):
        assert not core._surface_selected(path, core._SURFACE_PROFILE), path
    for path in ("README.md", "action.yml", "styxx/diffgate.py", "integrations/git/commit-msg",
                 "integrations/claude-code/diffgate-hook/README.md", "web/gate/README.md", "sworn/action.yml",
                 "styxx/README.MD", "integrations/git/hooks/.hidden"):
        assert core._surface_selected(path, core._SURFACE_PROFILE), path


def test_a_configured_profile_takes_globs_where_one_star_stays_inside_a_directory():
    prof = {"include": ["README*.md", "docs/**", "src/*.py", "exact.txt"], "exclude": ["docs/old/**"], "exts": EXTS}
    yes = ("README.md", "README_demo.md", "docs/a.md", "docs/a/b.md", "src/a.py", "exact.txt")
    no = ("readme.md", "src/a/b.py", "docs/old/a.md", "docs/a.json", "other/README.md", "exact.txt.bak", "exactXtxt")
    for p in yes:
        assert core._surface_selected(p, prof), p
    for p in no:
        assert not core._surface_selected(p, prof), p


def test_surface_text_reads_utf8_with_replacement_and_no_binary():
    assert core._surface_text(b"ab\xffc") == "ab�c"
    assert core._surface_text(b"ab\x00c") is None
    assert core._surface_text("café\n".encode("utf-8")) == "café\n"
    assert core._surface_text(b"") == ""


# ---- the surfaces at a rev (git) ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def surfaces_repo(tmp_path_factory):
    g = Git(tmp_path_factory.mktemp("surfaces"))
    g.write("README.md", "The model is calibrated well enough to ship.\n")
    g.write("notes.txt", "not in the profile\n")
    g.write("styxx/x.py", "'''calibrated well enough to ship'''\n")
    g.write("styxx/data.json", '{"a": "calibrated well enough to ship"}\n')
    g.write("styxx/__pycache__/x.py", "pruned\n")
    g.write("web/gate/differential/a.md", "excluded\n")
    g.write("styxx/b.py", b"\x00\x01binary calibrated well enough to ship\n")
    g.write("docs/big.md", "x" * 64 + "\n")
    c1 = g.commit("c1")
    g.write("target.txt", "README.md")
    oid = g("hash-object", "-w", "target.txt")
    (g.root / "target.txt").unlink()
    g("update-index", "--add", "--cacheinfo", f"120000,{oid},styxx/link.py")
    g("commit", "-q", "-m", "a symlink")
    c2 = g("rev-parse", "HEAD")
    with G._Repo(g.root) as repo:
        yield repo, c1, c2


def test_the_surfaces_at_a_rev_are_the_selected_blobs_sorted_with_binary_and_links_named_unread(surfaces_repo):
    repo, c1, c2 = surfaces_repo
    got = G._surfaces(repo, c1)
    assert [p for p, _ in got] == ["README.md", "docs/big.md", "styxx/b.py", "styxx/x.py"]
    assert dict(got)["styxx/b.py"] is None and dict(got)["README.md"].startswith("The model")
    got2 = dict(G._surfaces(repo, c2))
    assert "styxx/link.py" in got2 and got2["styxx/link.py"] is None
    assert G._surfaces(repo, c1, {"include": ["notes.txt"], "exclude": [], "exts": EXTS}) == [
        ("notes.txt", "not in the profile\n")]


def test_a_surface_over_the_size_cap_is_named_and_not_read(surfaces_repo, monkeypatch):
    repo, c1, _ = surfaces_repo
    monkeypatch.setattr(core, "_SURFACE_MAX_BYTES", 60)
    got = dict(G._surfaces(repo, c1))
    assert got["docs/big.md"] is None and got["README.md"] is not None


def test_licences_over_the_surfaces_of_a_synthetic_repository(surfaces_repo):
    repo, c1, _ = surfaces_repo
    r = core.licences(prereg(HOLD=[LIC]), G._surfaces(repo, c1), landed="HOLD")
    assert [(f["path"], f["line"]) for f in r["findings"]] == [("README.md", 1), ("styxx/x.py", 1)]
    assert r["skipped"] == ["styxx/b.py"] and r["surfaces"] == 3


# ---- RV3 and RV4 through the module (the done-when) ----------------------------------------------------

@pytest.fixture(scope="module")
def styxx_repo():
    with G._Repo(ROOT) as repo:
        if not _have(repo, RESULT_COMMIT, BEFORE_221):
            pytest.skip(f"{RESULT_COMMIT[:8]} or {BEFORE_221[:8]} is not in this object store")
        yield repo


def test_the_copied_table_is_the_design_s_rv4_table_byte_for_byte():
    data = TABLE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == TABLE_SHA256
    assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf")
    assert data.startswith(b"# RECONSTRUCTION, written 2026-10-24 after the result was known")
    d = core._decision(data)
    assert d["form"] == "gates" and d["gates_sha256"] == RV4_DIGEST


def _hand_rule(surfaces):
    """The files where scripts/surface_lint.py's withdrawn/gate-claim rule reports an unwithdrawn hit."""
    return {path for path, text in surfaces if text is not None
            and any(h.rid == HAND_RULE for h in SL.lint_text(text, path)[0])}


def _quoted_table() -> bytes:
    """The RV4 table with the licence's pattern set to the PREREG's verbatim quote, escaped."""
    text = TABLE.read_text(encoding="utf-8")
    assert text.count(json.dumps(FIVE)) == 1
    return text.replace(json.dumps(FIVE), json.dumps(re.escape(QUOTE))).encode("utf-8")


@pytest.mark.parametrize("rev, read, five, hand", [(RESULT_COMMIT, 328, 4, 4), (BEFORE_221, 394, 3, 10)])
def test_rv3_the_module_reads_the_surfaces_surface_lint_reads_and_the_five_words_find_what_the_hand_rule_finds(
        styxx_repo, rev, read, five, hand):
    surfaces = G._surfaces(styxx_repo, rev)
    assert [p for p, _ in surfaces] == sorted(p for p, _ in SL.rev_files(ROOT, rev, SL.PROFILES["repo"]))
    assert sum(1 for _, t in surfaces if t is not None) == read
    r = core.licences(TABLE.read_bytes(), surfaces, landed=FAILED)
    assert r["code"] is None and len(r["licences"]) == 1
    matched = r["licences"][0]["surfaces"]
    assert len(matched) == five and set(codes(r)) == {WITHDRAWN}
    assert sorted({f["path"] for f in r["findings"]}) == matched       # RV3 counts files, not matches
    by_hand = _hand_rule(surfaces)
    assert len(by_hand) == hand and set(matched) <= by_hand
    q = core.licences(_quoted_table(), surfaces, landed=None)
    assert codes(q) == [NOTHING] and q["licences"][0]["surfaces"] == []


def test_rv3_the_four_surfaces_on_the_result_s_day_and_the_seven_the_pattern_misses_before_221(styxx_repo):
    day = core.licences(TABLE.read_bytes(), G._surfaces(styxx_repo, RESULT_COMMIT), landed=FAILED)
    assert day["licences"][0]["surfaces"] == ["README.md", "action.yml", "styxx/__init__.py", "styxx/diffgate.py"]
    before = G._surfaces(styxx_repo, BEFORE_221)
    r = core.licences(TABLE.read_bytes(), before, landed=FAILED)
    assert r["licences"][0]["surfaces"] == ["action.yml", "styxx/__init__.py", "styxx/diffgate.py"]
    missed = sorted(_hand_rule(before) - set(r["licences"][0]["surfaces"]))
    assert len(missed) == 7 and all(p.startswith("integrations/") for p in missed), missed
    pending = core.licences(TABLE.read_bytes(), before, landed=None)
    assert set(codes(pending)) == {PENDING}
    assert sorted({f["path"] for f in pending["findings"]}) == ["action.yml", "styxx/__init__.py", "styxx/diffgate.py"]


def test_rv4_the_committed_receipt_scores_under_the_reconstructed_table_through_rescore(styxx_repo):
    rec = json.loads(styxx_repo.read(RESULT_COMMIT, RECEIPT).decode("utf-8"))
    assert (rec["decoys_correct"], rec["precision"], rec["gate_G_E1_pass"]) == (30, 0.23, False)
    r = core.rescore(TABLE.read_bytes(), rec)
    assert (r["code"], r["verdict"], r["gates"], r["gates_sha256"], r["row"]) == (
        None, FAILED, {"G_rel": True, "G_E1": False}, RV4_DIGEST, 1)
    assert r["withdraws"] == GATE_CLAIM
    assert core.compare(r, {"verdict": FAILED}) == [
        {"code": "RECORDED_DIGEST_ABSENT", "kind": "NOTE", "detail": "the receipt records no gates digest, so (i) is not read"}]


def test_the_copied_rv3_and_rv4_outputs_state_the_numbers_these_tests_pin():
    """The design's RV3 and RV4 outputs, copied under conformance/ (CRLF read as LF, as the B1 copies
    were), state the counts, the verdict, the digest and the licence the tests above hold the module to."""
    text = (SET / "licence_replay_ca347730.txt").read_text(encoding="utf-8")
    rows = re.findall(r"== rev (\S+): surfaces read (\d+); files with Q (\d+), K (\d+), H (\d+)", text)
    assert rows == [("414db45a", "328", "0", "4", "4"), ("3437340b^1", "394", "0", "3", "10")]
    assert text.count("  H -K added ") == 7 and text.count("  H -- added 2026-09-16 integrations/") == 7
    text4 = (SET / "external1_rescore_ca347730.txt").read_text(encoding="utf-8")
    assert f"verdict {FAILED}\n" in text4 and f"gates_sha256 {RV4_DIGEST}\n" in text4
    assert "gates {'G_rel': True, 'G_E1': False}\n" in text4
    assert json.loads(text4.split("withdraws ", 1)[1]) == GATE_CLAIM
    for name in ("licence_replay_ca347730.txt", "external1_rescore_ca347730.txt"):
        data = (SET / name).read_bytes()
        assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf"), name


# ---- the licence vectors (PLAN K15: the normalisation pinned by a vector) ------------------------------

VECTORS = SET / "vectors_licence.json"


@pytest.fixture(scope="module")
def vectors():
    doc = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert doc["schema"] == "styxx.prereg-ci.conformance.licence/v0"
    return doc


def test_every_licence_vector_replays(vectors):
    assert len(vectors["vectors"]) >= 12
    for v in vectors["vectors"]:
        inp = v["inputs"]
        r = core.licences(inp["prereg"], [tuple(s) for s in inp["surfaces"]], landed=inp["landed"])
        got = {"code": r["code"], "findings": [[f["code"], f["licence"], f["path"], f["line"], f["col"]]
                                               for f in r["findings"]],
               "matches": {li["id"]: [[m["path"], m["line"], m["col"], m["text"], m["withdrawn"]]
                                      for m in li["matches"]] for li in r["licences"]}}
        assert got == v["expect"], v["name"]


def test_the_vectors_pin_the_normalisation_and_the_window(vectors):
    names = {v["name"] for v in vectors["vectors"]}
    for needed in ("normalise/curly-apostrophe", "normalise/case", "normalise/whitespace-across-lines",
                   "window/withdrawn-inside", "window/withdrawn-outside", "window/not-yet-true",
                   "lint/pattern-finds-nothing", "pending/before-data", "landed/other-row", "landed/smoke",
                   "locks/withdraws-map", "refusal/pattern-does-not-compile"):
        assert needed in names, needed


def test_the_licence_generator_regenerates_the_file():
    r = subprocess.run([sys.executable, "-B", str(SET / "gen_vectors_licence.py"), "--check"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert r.stdout.startswith("SAME vectors_licence.json=")
