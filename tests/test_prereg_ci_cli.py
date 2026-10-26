# -*- coding: utf-8 -*-
"""prereg-ci B7: the commands (DESIGN 3.3) on repositories made with ``git init`` in tmp_path.

``lint`` (3.7), ``check`` (3.4, C0 to C9), ``ladder`` and ``census``, with the exit codes of 3.3: 0
PASS (or nothing to check), 1 FAIL, 2 input error, 3 UNCHECKABLE. A refusal is read ahead of an
input error and an unchecked reading, so exit 1 says something was refused whatever else was not
read, and the report lists the rest. Policies come from ``prereg-ci.toml`` read at BASE (a pull
request cannot relax the policy that judges it) and from flags, which win; PLAN K8: ``tomllib`` where
the interpreter has it, else a strict reader of the subset the config uses, and both read every
example the same way.

Each repository is the shape of DESIGN 4.B: a PREREG whose HOLD row withdraws a README sentence, a
receipt, a RESULT with a result fence. No remote, no network.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx.prereg_ci import cli  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from styxx.prereg_ci import time as T  # noqa: E402
from tests.test_prereg_ci_git import Git  # noqa: E402

F = "`" * 3
SENTENCE = "calibrated well enough to ship"
TABLE = {"gates": {"G_acc": {"metric": "acc", "op": ">=", "value": 0.9, "power_basis": "a floor"},
                   "G_ece": {"metric": "ece", "op": "<=", "value": 0.05, "power_basis": "a ceiling"}},
         "outcomes": [{"when": {"G_acc": False}, "verdict": "INVALID__accuracy_floor"},
                      {"when": {"G_acc": True, "G_ece": True}, "verdict": "SHIP"},
                      {"when": {"G_acc": True, "G_ece": False}, "verdict": "HOLD__calibration",
                       "withdraws": [{"id": "calibration", "quote": f"the model is {SENTENCE}",
                                      "pattern": SENTENCE}]}],
         "smoke_verdict": "INVALID__smoke"}


def prereg(table=None, prose="the plan") -> str:
    return f"# PREREG digits\n\n{prose}\n\n{F}gates\n{json.dumps(table or TABLE, indent=1)}\n{F}\n"


def scored(metrics: dict, table=None, **extra) -> dict:
    """A receipt scored the way an Experiment records one: its metrics, the verdict, gates and digest."""
    rec = {"prereg": "PREREG_digits.md", **metrics, **extra}
    rs = core.rescore(prereg(table), rec)
    rec.update(verdict=rs["verdict"], gates=rs["gates"], gates_sha256=rs["gates_sha256"])
    return rec


def result_md(verdict: str, receipt: str = "receipt.json") -> str:
    fence = json.dumps({"prereg": "PREREG_digits.md", "receipt": receipt, "verdict": verdict})
    return f"# RESULT digits\n\nThe run reads {verdict}.\n\n{F}result\n{fence}\n{F}\n"


def run(g: Git, *argv, capsys=None):
    cwd = os.getcwd()
    try:
        os.chdir(g.root)
        code = cli.main(list(argv))
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out if capsys is not None else ""
    return code, out


def demo(tmp_path, metrics=None, readme=None, config='surfaces = ["README*.md"]\n'):
    """main: the plan sealed (PREREG, README, config), then a pull request adding receipt and RESULT."""
    g = Git(tmp_path / "demo")
    g.write("PREREG_digits.md", prereg())
    g.write("README_demo.md", readme if readme is not None else f"# demo\n\nThe model is {SENTENCE}.\n")
    if config is not None:
        g.write("prereg-ci.toml", config)
    g.commit("seal the plan")
    g("switch", "-q", "-c", "result")
    rec = scored(metrics or {"acc": 0.95, "ece": 0.03})
    g.write("receipt.json", json.dumps(rec, indent=1) + "\n")
    g.write("RESULT_digits.md", result_md(rec["verdict"]))
    g.commit("result")
    return g, rec


# ---- check: PASS and the four refusals a byte comparison does not give (DESIGN 4.B) -----------------------

def test_check_passes_a_receipt_scored_under_the_sealed_table_and_prints_its_ladder(tmp_path, capsys):
    g, rec = demo(tmp_path)
    code, out = run(g, "check", "--base", "main", "--head", "result", capsys=capsys)
    assert code == 0, out
    assert "PASS  RESULT_digits.md" in out
    assert "landed row 2 of 3: SHIP" in out
    assert "time AUTHOR_ORDER (T2 not read, T3 none)" in out
    assert "flips that move the verdict" in out and "G_acc 0.95 >= 0.9 (margin 0.05) -> INVALID__accuracy_floor" in out
    assert "G_ece 0.03 <= 0.05 (margin 0.02) -> HOLD__calibration" in out
    assert "not examined: the run itself" in out
    assert out.rstrip().endswith("PASS  exit 0")


def test_a_result_that_types_another_rung_is_verdict_differs(tmp_path, capsys):
    g, rec = demo(tmp_path)
    g("switch", "-q", "-c", "typed", "result")
    g.write("RESULT_digits.md", result_md("HOLD__calibration"))
    g.commit("type the verdict")
    code, out = run(g, "check", "--base", "result", "--head", "typed", capsys=capsys)
    assert code == 1, out
    assert "FAIL VERDICT_DIFFERS" in out and "the RESULT RESULT_digits.md prints HOLD__calibration" in out


def test_a_receipt_scored_under_an_unsealed_bar_is_digest_unsealed_and_verdict_differs(tmp_path, capsys):
    g, rec = demo(tmp_path)
    moved = json.loads(json.dumps(TABLE))
    moved["gates"]["G_ece"]["value"] = 0.01
    g("switch", "-q", "-c", "rerun", "result")
    g.write("receipt_rerun.json", json.dumps(scored({"acc": 0.95, "ece": 0.03}, table=moved)) + "\n")
    g.commit("re-run")
    code, out = run(g, "check", "--base", "result", "--head", "rerun", capsys=capsys)
    assert code == 1, out
    assert "FAIL DECISION_DIGEST_UNSEALED" in out and "FAIL VERDICT_DIFFERS" in out
    assert "the sealed decision scores this receipt SHIP; the receipt records HOLD__calibration" in out
    assert "C0 sealed decisions: 0 moved" in out


def test_a_receipt_rewritten_in_place_is_refused_and_the_note_policy_prints_it(tmp_path, capsys):
    g, rec = demo(tmp_path)
    g("switch", "-q", "main")
    g("merge", "-q", "--no-ff", "-m", "merge result", "result")
    g("switch", "-q", "-c", "rewrite")
    g.write("receipt.json", json.dumps(scored({"acc": 0.95, "ece": 0.04})) + "\n")
    g.commit("rewrite")
    code, out = run(g, "check", "--base", "main", "--head", "rewrite", capsys=capsys)
    assert code == 1 and "FAIL RECEIPT_REWRITTEN" in out, out
    code, out = run(g, "check", "--base", "main", "--head", "rewrite", "--receipt-rewrite", "note", capsys=capsys)
    assert code == 0 and "NOTE RECEIPT_REWRITTEN" in out and "receipt_rewrite note (flag)" in out, out


def test_a_bar_moved_in_the_sealed_prereg_is_decision_edited_after_seal(tmp_path, capsys):
    g, rec = demo(tmp_path)
    g("switch", "-q", "-c", "lower", "main")
    moved = json.loads(json.dumps(TABLE))
    moved["gates"]["G_acc"]["value"] = 0.8
    g.write("PREREG_digits.md", prereg(moved))
    g.commit("lower the bar")
    code, out = run(g, "check", "--base", "main", "--head", "lower", capsys=capsys)
    assert code == 1 and "FAIL DECISION_EDITED_AFTER_SEAL" in out, out
    assert "path protection" in out


def test_a_receipt_committed_with_its_prereg_is_seal_not_before_receipt(tmp_path, capsys):
    g = Git(tmp_path / "same")
    g.write("README.md", "# r\n")
    g.commit("root")
    g("switch", "-q", "-c", "both")
    g.write("PREREG_digits.md", prereg())
    g.write("receipt.json", json.dumps(scored({"acc": 0.95, "ece": 0.03})) + "\n")
    g.commit("plan and result at once")
    code, out = run(g, "check", "--base", "main", "--head", "both", capsys=capsys)
    assert code == 1, out
    assert "FAIL SEAL_NOT_BEFORE_RECEIPT" in out and "time NOT_ORDERED" in out


def test_a_prereg_that_arrives_in_the_pull_request_is_ordered_by_its_own_commits(tmp_path, capsys):
    g = Git(tmp_path / "arrives")
    g.write("README.md", "# r\n")
    g.commit("root")
    g("switch", "-q", "-c", "pr")
    g.write("PREREG_digits.md", prereg())
    g.commit("plan")
    g.write("receipt.json", json.dumps(scored({"acc": 0.95, "ece": 0.03})) + "\n")
    g.commit("result")
    code, out = run(g, "check", "--base", "main", "--head", "pr", capsys=capsys)
    assert code == 0, out
    assert "not landed: the PREREG arrives in this pull request (C2)" in out and "time AUTHOR_ORDER" in out


# ---- licences: the landed row's sentences, on this pull request and every later one ------------------------

def test_a_receipt_landing_on_hold_refuses_the_readme_sentence_by_file_and_line(tmp_path, capsys):
    g, rec = demo(tmp_path, metrics={"acc": 0.95, "ece": 0.08})
    assert rec["verdict"] == "HOLD__calibration"
    code, out = run(g, "check", "--base", "main", "--head", "result", capsys=capsys)
    assert code == 1, out
    assert "FAIL LICENCE_WITHDRAWN README_demo.md:3:14" in out


def test_before_data_a_licence_on_a_surface_is_pending_and_the_check_passes(tmp_path, capsys):
    g = Git(tmp_path / "pending")
    g.write("README.md", "# r\n")
    g.write("prereg-ci.toml", 'surfaces = ["README*.md"]\n')
    g.commit("root")
    g("switch", "-q", "-c", "plan")
    g.write("PREREG_digits.md", prereg())
    g.write("README_demo.md", f"The model is {SENTENCE}.\n")
    g.commit("seal the plan")
    code, out = run(g, "check", "--base", "main", "--head", "plan", capsys=capsys)
    assert code == 0, out
    assert "NOTE LICENCE_PENDING README_demo.md:1:14" in out


def test_a_later_pull_request_that_adds_the_withdrawn_wording_is_refused_when_it_adds_it(tmp_path, capsys):
    g, rec = demo(tmp_path, metrics={"acc": 0.95, "ece": 0.08}, readme="# demo\n\nNo claim here.\n")
    g("switch", "-q", "main")
    g("merge", "-q", "--no-ff", "-m", "merge result", "result")
    code, out = run(g, "check", "--base", "main~1", "--head", "main", capsys=capsys)
    assert code == 0, out
    g("switch", "-q", "-c", "later")
    g.write("README_more.md", f"Ours is {SENTENCE}, again.\n")
    g.commit("a later surface")
    code, out = run(g, "check", "--base", "main", "--head", "later", capsys=capsys)
    assert code == 1, out
    assert "FAIL LICENCE_WITHDRAWN README_more.md:1:9" in out and "receipt.json" in out
    g.write("README_more.md", f"Ours is {SENTENCE}, a sentence withdrawn by the HOLD row.\n")
    g.commit("withdraw it")
    code, out = run(g, "check", "--base", "main", "--head", "later", capsys=capsys)
    assert code == 0, out


# ---- policies, config and exit codes -----------------------------------------------------------------

def test_policies_are_read_at_base_so_a_pull_request_cannot_relax_its_own_judge(tmp_path, capsys):
    g, rec = demo(tmp_path)
    g("switch", "-q", "main")
    g("merge", "-q", "--no-ff", "-m", "merge result", "result")
    g("switch", "-q", "-c", "relax")
    g.write("prereg-ci.toml", 'surfaces = ["README*.md"]\nreceipt_rewrite = "note"\n')
    g.write("receipt.json", json.dumps(scored({"acc": 0.95, "ece": 0.04})) + "\n")
    g.commit("relax and rewrite")
    code, out = run(g, "check", "--base", "main", "--head", "relax", capsys=capsys)
    assert code == 1 and "FAIL RECEIPT_REWRITTEN" in out, out
    assert "NOTE CONFIG_CHANGED" in out and "read at base" in out


def test_a_config_holding_a_key_or_a_value_the_format_does_not_name_is_an_input_error(tmp_path, capsys):
    for bad in ('receipt_rewrite = "maybe"\n', "receipt_rewrite = 1\n", 'colour = "red"\n',
                '[table]\nreceipt_rewrite = "note"\n', 'surfaces = "README.md"\n'):
        g, rec = demo(tmp_path / str(abs(hash(bad))), config=bad)
        code, out = run(g, "check", "--base", "main", "--head", "result", capsys=capsys)
        assert code == 2 and "CONFIG_INVALID" in out, (bad, out)


def test_an_invalid_config_is_refused_where_it_is_written_and_a_broken_base_can_be_repaired(tmp_path, capsys):
    """The B7 review's F2, its three pull requests: (1) the one that writes a config the format refuses
    reads it at HEAD and exits 2, so under a required check it does not land; (2) if it lands anyway (a
    bypass, a direct push), an unrelated pull request still exits 2, since the policies the repository
    set do not read; (3) the pull request that replaces the broken copy with one that reads is judged
    under the defaults and flags (as a repository with no config is), prints CONFIG_REPAIRED, and
    passes, so the repository can merge its own repair."""
    g, rec = demo(tmp_path)
    g("switch", "-q", "main")
    g("merge", "-q", "--no-ff", "-m", "merge result", "result")
    main0 = g("rev-parse", "HEAD")
    g("switch", "-q", "-c", "pr1", main0)
    g.write("prereg-ci.toml", 'surfaces = ["README*.md"]\nreceipt_rewrite = "sometimes"\n')
    g.commit("tune the policy")
    code, out = run(g, "check", "--base", main0, "--head", "pr1", capsys=capsys)
    assert code == 2, out
    assert f"INPUT CONFIG_INVALID  prereg-ci.toml at {g('rev-parse', 'pr1')[:12]}: receipt_rewrite" in out, out
    assert "NOTE CONFIG_CHANGED" in out and out.rstrip().endswith("INPUT_ERROR  exit 2: CONFIG_INVALID")
    g("switch", "-q", "main")
    g("merge", "-q", "--no-ff", "-m", "merge pr1 past its check", "pr1")
    main1 = g("rev-parse", "HEAD")
    g("switch", "-q", "-c", "pr2", main1)
    g.write("notes.md", "a note\n")
    g.commit("a note")
    code, out = run(g, "check", "--base", main1, "--head", "pr2", capsys=capsys)
    assert code == 2 and f"INPUT CONFIG_INVALID  prereg-ci.toml at {main1[:12]}" in out, out
    for name, text in (("pr3", 'surfaces = ["README*.md"]\n'), ("pr3_rm", None)):
        g("switch", "-q", "-c", name, main1)
        if text is None:
            g("rm", "-q", "prereg-ci.toml")
        else:
            g.write("prereg-ci.toml", text)
        g.commit("repair the config")
        code, out = run(g, "check", "--base", main1, "--head", name, capsys=capsys)
        assert code == 0, (name, out)
        assert "NOTE CONFIG_REPAIRED" in out and "does not read, and this pull request replaces it" in out, out
        assert "receipt_rewrite fail (default)" in out and "(whose copy does not read, so the defaults govern)" in out
    g("switch", "-q", "-c", "pr4", main1)
    g.write("prereg-ci.toml", 'receipt_rewrite = "never"\n')
    g.commit("a repair that does not read either")
    code, out = run(g, "check", "--base", main1, "--head", "pr4", capsys=capsys)
    assert code == 2 and f"prereg-ci.toml at {main1[:12]}" in out and "CONFIG_REPAIRED" not in out, out


def test_a_repair_of_a_broken_base_config_is_still_judged_by_the_defaults(tmp_path, capsys):
    """The repair path does not let a pull request choose its policies: one that writes a valid config
    relaxing receipt_rewrite and rewrites a committed receipt in the same commit is refused under the
    default (fail), not under the copy it writes."""
    g, rec = demo(tmp_path, config='receipt_rewrite = "sometimes"\n')
    g("switch", "-q", "main")
    g("merge", "-q", "--no-ff", "-m", "merge result", "result")
    g("switch", "-q", "-c", "relax")
    g.write("prereg-ci.toml", 'surfaces = ["README*.md"]\nreceipt_rewrite = "note"\n')
    g.write("receipt.json", json.dumps(scored({"acc": 0.95, "ece": 0.04})) + "\n")
    g.commit("repair, relax and rewrite")
    code, out = run(g, "check", "--base", "main", "--head", "relax", capsys=capsys)
    assert code == 1 and "FAIL RECEIPT_REWRITTEN" in out and "NOTE CONFIG_REPAIRED" in out, out


CONFIGS = [
    "",
    "# nothing but a comment\n",
    'immutability = "file"\namend_after_data = "fail"\n',
    'receipt_rewrite = "note"  # a build output\nrequire_time = "T3a"\npreregistered_word = "fail"\n',
    'surfaces = ["README*.md", "docs/**"]\nsurfaces_exclude = []\nexclude = ["conformance/"]\n',
    'surfaces = [\n  "README*.md",  # the demo\n  \'docs/**\',\n]\n',
    "exclude = ['a\\\\b/', \"c\\\"d/\"]\n",
    'surfaces = ["\\u00e9t\\u00e9/**"]\n',
    "a = true\nb = false\n",
]
REFUSED = ['key = 1\n', 'key = 1.5\n', '[t]\n', 'a = "x"\na = "y"\n', 'a = [1, 2]\n', 'a = {b = "c"}\n',
           'a = """x"""\n', 'a.b = "c"\n', 'a = "x" "y"\n', 'a = [\n"x"\n', 'a = "\\q"\n', '= "x"\n',
           'a = 1979-05-27\n', 'a = [["x"]]\n']


@pytest.mark.parametrize("text", CONFIGS)
def test_the_strict_config_reader_reads_what_tomllib_reads(text):
    got = cli._toml_subset(text)
    try:
        import tomllib
    except ImportError:
        pytest.skip("tomllib is not in this interpreter (3.10 or older); the strict reader alone is read")
    assert got == tomllib.loads(text)


@pytest.mark.parametrize("text", REFUSED)
def test_the_strict_config_reader_refuses_what_the_subset_does_not_hold(text):
    with pytest.raises(ValueError):
        cli._toml_subset(text)


def test_an_absent_base_is_uncheckable_and_a_shallow_clone_is_uncheckable(tmp_path, capsys):
    g, rec = demo(tmp_path)
    code, out = run(g, "check", "--base", "0" * 40, "--head", "result", capsys=capsys)
    assert code == 3 and "COMMIT_ABSENT" in out, out
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", "result", g.root.resolve().as_uri(),
                    str(shallow)], check=True, capture_output=True)
    cwd = os.getcwd()
    try:
        os.chdir(shallow)
        code = cli.main(["check", "--base", "HEAD", "--head", "HEAD"])
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out
    assert code == 3 and "SHALLOW_CLONE" in out, out


def test_an_ambiguous_prereg_is_an_input_error_and_a_refusal_beside_it_is_exit_one(tmp_path, capsys):
    g, rec = demo(tmp_path)
    g("switch", "-q", "-c", "two", "result")
    g.write("a/PREREG_digits.md", prereg())
    g.write("b/PREREG_digits.md", prereg())
    g.write("c/receipt.json", json.dumps(scored({"acc": 0.95, "ece": 0.03})) + "\n")
    g.commit("two plans of one name")
    code, out = run(g, "check", "--base", "result", "--head", "two", capsys=capsys)
    assert code == 2 and "INPUT PREREG_AMBIGUOUS" in out, out
    g.write("RESULT_digits.md", result_md("SHIP__typed"))
    g.commit("and a typed verdict")
    code, out = run(g, "check", "--base", "result", "--head", "two", capsys=capsys)
    assert code == 1 and "INPUT PREREG_AMBIGUOUS" in out and "FAIL VERDICT_DIFFERS" in out, out


def test_check_reads_the_trust_store_at_base(tmp_path, capsys, monkeypatch):
    g, rec = demo(tmp_path)
    seen = []
    real = T._reading

    def spy(*a, **k):
        seen.append(k.get("trust_rev"))
        return real(*a, **k)

    monkeypatch.setattr(T, "_reading", spy)
    code, out = run(g, "check", "--base", "main", "--head", "result", capsys=capsys)
    assert code == 0 and seen == [g("rev-parse", "main")], (seen, out)


def test_require_time_t3a_refuses_a_seal_read_on_author_order(tmp_path, capsys):
    g, rec = demo(tmp_path)
    code, out = run(g, "check", "--base", "main", "--head", "result", "--require-time", "T3a", capsys=capsys)
    assert code == 1 and "FAIL TIME_BELOW_POLICY" in out, out


def test_a_result_titled_preregistered_on_author_order_is_a_note_and_a_fail_under_the_policy(tmp_path, capsys):
    g, rec = demo(tmp_path)
    g("switch", "-q", "-c", "titled", "result")
    g.write("RESULT_digits.md", result_md("SHIP").replace("# RESULT digits", "# RESULT digits (preregistered)"))
    g.commit("title")
    code, out = run(g, "check", "--base", "main", "--head", "titled", capsys=capsys)
    assert code == 0 and "NOTE PREREGISTERED_ON_AUTHOR_ORDER" in out, out
    code, out = run(g, "check", "--base", "main", "--head", "titled", "--preregistered-word", "fail", capsys=capsys)
    assert code == 1 and "FAIL PREREGISTERED_ON_AUTHOR_ORDER" in out, out


def test_nothing_to_check_is_pass(tmp_path, capsys):
    g = Git(tmp_path / "empty")
    g.write("README.md", "# r\n")
    g.commit("root")
    g("switch", "-q", "-c", "docs")
    g.write("docs.md", "# docs\n")
    g.commit("docs")
    code, out = run(g, "check", "--base", "main", "--head", "docs", capsys=capsys)
    assert code == 0 and "C1 pairs: 0" in out, out


# ---- the locks form inside check ------------------------------------------------------------------------

SCORER = ("import json, sys\n"
          "d = json.load(open(sys.argv[1], encoding='utf-8'))\n"
          "print(json.dumps({'verdict': 'PASS' if d['m'] >= 0.5 else 'HOLD'}))\n")


def locks_prereg() -> str:
    block = {"scorer": "score.py", "reads": [], "outcomes": ["PASS", "HOLD"]}
    return f"# PREREG locked\n\n{F}locks\n{json.dumps(block)}\n{F}\n"


def test_a_locks_decision_runs_its_sealed_scorer_and_a_locked_file_edited_after_reads_as_a_note(tmp_path, capsys):
    g = Git(tmp_path / "locks")
    g.write("PREREG_locked.md", locks_prereg())
    g.write("score.py", SCORER)
    g.commit("seal")
    g("switch", "-q", "-c", "run")
    g.write("receipt.json", json.dumps({"prereg": "PREREG_locked.md", "m": 0.7, "verdict": "PASS"}) + "\n")
    g.commit("result")
    code, out = run(g, "check", "--base", "main", "--head", "run", capsys=capsys)
    assert code == 0, out
    assert "locked scorer score.py" in out and "PASS  receipt.json" in out
    g.write("score.py", SCORER.replace("'PASS' if", "'HOLD' if"))
    g.write("receipt2.json", json.dumps({"prereg": "PREREG_locked.md", "m": 0.7, "verdict": "PASS"}) + "\n")
    g.commit("edit the judge and add a receipt")
    code, out = run(g, "check", "--base", "main", "--head", "run", capsys=capsys)
    assert code == 0, out
    assert "NOTE LOCKED_FILE_MOVED" in out


# ---- lint -----------------------------------------------------------------------------------------------

def lint(g, path, capsys):
    return run(g, "lint", path, capsys=capsys)


def test_lint_passes_a_sound_table_and_lists_where_each_pattern_matches(tmp_path, capsys):
    g = Git(tmp_path / "lint")
    g.write("README_demo.md", f"The model is {SENTENCE}.\n")
    g.write("prereg-ci.toml", 'surfaces = ["README*.md"]\n')
    g.commit("surfaces")
    g.write("PREREG_digits.md", prereg())
    code, out = lint(g, "PREREG_digits.md", capsys)
    assert code == 0, out
    assert "total over 4 assignments" in out and "NOTE LICENCE_PENDING README_demo.md:1:14" in out
    assert "licence calibration: 1 surface" in out


@pytest.mark.parametrize("mutate, code_", [
    (lambda t: t["outcomes"].pop(0), "TABLE_NOT_TOTAL"),
    (lambda t: t["outcomes"].insert(0, {"when": {}, "verdict": "ALWAYS"}), "ROW_UNREACHABLE"),
    (lambda t: t["outcomes"][2].update(verdict="SHIP"), "VERDICTS_REPEATED"),
    (lambda t: t["gates"].update(G_extra={"metric": "x", "op": ">=", "value": 1, "power_basis": "b"}), "GATE_VACUOUS"),
    (lambda t: t["outcomes"][2].update(withdraw=t["outcomes"][2].pop("withdraws")), "WITHDRAWS_KEY_MISSPELLED"),
    (lambda t: t["outcomes"][2]["withdraws"][0].update(pattern="no surface says this"),
     "LICENCE_PATTERN_FINDS_NOTHING"),
    (lambda t: t["gates"]["G_acc"].update(op="=~"), "SCORER_REFUSES"),
])
def test_lint_refuses_before_data(tmp_path, capsys, mutate, code_):
    g = Git(tmp_path / "lint")
    g.write("README_demo.md", f"The model is {SENTENCE}.\n")
    g.write("prereg-ci.toml", 'surfaces = ["README*.md"]\n')
    g.commit("surfaces")
    t = json.loads(json.dumps(TABLE))
    mutate(t)
    g.write("PREREG_digits.md", prereg(t))
    code, out = lint(g, "PREREG_digits.md", capsys)
    assert code == 1 and f"FAIL {code_}" in out, out


def test_lint_notes_an_undeclared_power_basis_and_an_unread_row_key(tmp_path, capsys):
    g = Git(tmp_path / "lint")
    g.write("README_demo.md", f"The model is {SENTENCE}.\n")
    g.write("prereg-ci.toml", 'surfaces = ["README*.md"]\n')
    g.commit("surfaces")
    t = json.loads(json.dumps(TABLE))
    del t["gates"]["G_ece"]["power_basis"]
    t["outcomes"][1]["note"] = "a reader's note"
    g.write("PREREG_digits.md", prereg(t))
    code, out = lint(g, "PREREG_digits.md", capsys)
    assert code == 0, out
    assert "NOTE POWER_BASIS_UNDECLARED" in out and "G_ece" in out and "NOTE ROW_KEY_UNREAD" in out


def test_lint_of_a_prose_prereg_is_not_gated_and_exit_zero(tmp_path, capsys):
    g = Git(tmp_path / "lint")
    g.write("README.md", "# r\n")
    g.commit("root")
    g.write("PREREG_prose.md", "# PREREG\n\nprose only\n")
    code, out = lint(g, "PREREG_prose.md", capsys)
    assert code == 0 and "NOT_GATED" in out, out


def test_lint_of_a_locks_decision_refuses_a_locked_path_that_is_not_committed(tmp_path, capsys):
    g = Git(tmp_path / "lint")
    g.write("score.py", SCORER)
    g.commit("scorer")
    g.write("PREREG_locked.md", locks_prereg().replace('"reads": []', '"reads": ["data/absent.csv"]'))
    code, out = lint(g, "PREREG_locked.md", capsys)
    assert code == 1 and "FAIL LOCKED_FILE_ABSENT" in out and "data/absent.csv" in out, out


# ---- ladder and census -----------------------------------------------------------------------------------

def test_ladder_prints_the_reading_of_one_receipt_under_a_prereg_read_from_the_file_system(tmp_path, capsys):
    g, rec = demo(tmp_path, metrics={"acc": 0.95, "ece": 0.08})
    outside = tmp_path / "table.md"
    outside.write_text(prereg(), encoding="utf-8")
    code, out = run(g, "ladder", "receipt.json", "--prereg", str(outside), "--rev", "result", capsys=capsys)
    assert code == 1, out
    assert "landed row 3 of 3: HOLD__calibration" in out and "FAIL LICENCE_WITHDRAWN README_demo.md:3:14" in out
    assert "not a seal" in out
    assert "the licence's verbatim quote matches 1 surface" in out


def test_census_rederives_every_receipt_at_a_revision_and_tallies_them(tmp_path, capsys):
    g, rec = demo(tmp_path)
    g("switch", "-q", "main")
    g("merge", "-q", "--no-ff", "-m", "merge result", "result")
    code, out = run(g, "census", "--rev", "main", capsys=capsys)
    assert code == 0, out
    assert "RESCORES receipt.json SHIP" in out and 'TALLY {"RESCORES": 1}' in out
    g("switch", "-q", "-c", "bad")
    bad = scored({"acc": 0.95, "ece": 0.03})
    bad["verdict"] = "HOLD__calibration"
    g.write("other/receipt.json", json.dumps(bad) + "\n")
    g.commit("a receipt that records another verdict")
    code, out = run(g, "census", "--rev", "bad", capsys=capsys)
    assert code == 1, out
    assert "VERDICT_DIFFERS other/receipt.json" in out and '"VERDICT_DIFFERS": 1' in out


def test_a_report_path_that_does_not_write_is_an_input_error_and_a_crash_is_no_verdict(tmp_path, capsys,
                                                                                       monkeypatch):
    """The B7 review's F5: ``--json`` naming a directory raised out of main (exit 1, which the Action
    reads as FAIL). It is now exit 2 after the reading, or 1 when the reading refused something; an
    exception no reading names exits 4, the Action's ERROR, with the exception printed."""
    g, rec = demo(tmp_path)
    d = tmp_path / "a_directory"
    d.mkdir()
    code, out = run(g, "check", "--base", "main", "--head", "result", "--json", str(d), capsys=capsys)
    assert code == 2 and f"REFUSED --json {d} does not write" in out and "PASS  exit 0" in out, out
    code, out = run(g, "check", "--base", "main", "--head", "result", "--summary", str(d), capsys=capsys)
    assert code == 2 and "REFUSED --summary" in out, out
    g("switch", "-q", "-c", "typed", "result")
    g.write("RESULT_digits.md", result_md("HOLD__calibration"))
    g.commit("type the verdict")
    code, out = run(g, "check", "--base", "result", "--head", "typed", "--json", str(d), capsys=capsys)
    assert code == 1 and "FAIL VERDICT_DIFFERS" in out and "REFUSED --json" in out, out

    def boom(*a, **k):
        raise RuntimeError("a reader that broke")

    monkeypatch.setattr(cli, "_run_check", boom)
    code, out = run(g, "check", "--base", "result", "--head", "typed", capsys=capsys)
    assert code == 4 and "ERROR  RuntimeError: a reader that broke" in out, out


@pytest.mark.parametrize("value", [7, True, None, ["PREREG_digits.md"], {"path": "PREREG_digits.md"}])
def test_a_receipt_whose_prereg_is_not_a_string_is_an_input_error_as_an_empty_one_is(tmp_path, capsys, value):
    """The B7 review's F4: a receipt recording verdict, gates and gates_sha256 whose prereg is a number,
    a boolean, null, a list or an object read PASS (it was a record, and C1 pairs receipts), while
    ``"prereg": ""`` read INPUT PREREG_ABSENT. Both now read PREREG_ABSENT, exit 2; a JSON file with no
    prereg key and no verdict stays outside the pairing, and the check says so."""
    g, rec = demo(tmp_path)
    g("switch", "-q", "-c", "odd", "result")
    g.write("r_odd.json", json.dumps(dict(scored({"acc": 0.95, "ece": 0.03}), prereg=value)) + "\n")
    g.write("settings.json", json.dumps({"prereg": True, "colour": "red"}) + "\n")
    g.commit("a receipt whose prereg is no path")
    code, out = run(g, "check", "--base", "result", "--head", "odd", capsys=capsys)
    assert code == 2, out
    assert "INPUT PREREG_ABSENT  r_odd.json: the record's prereg is" in out and "settings.json" not in out, out
    assert "names no PREREG by a top-level prereg key is not paired" in out


def test_the_python_dash_m_entry_runs_check(tmp_path):
    g, rec = demo(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-m", "styxx.prereg_ci", "check", "--base", "main", "--head", "result"],
                       cwd=str(g.root), env=env, capture_output=True)
    assert r.returncode == 0, r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    assert b"PASS  RESULT_digits.md" in r.stdout
