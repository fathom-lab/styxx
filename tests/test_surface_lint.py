# -*- coding: utf-8 -*-
"""scripts/surface_lint.py: the word lint over shipped surfaces, held in CI.

Three things are held here. No withdrawn sentence stands on a shipped surface except beside its withdrawal. No
shipped file carries more of the charter's words or of the plan's must-not-say words than
scripts/surface_lint_baseline.json records, so a count can fall and never rise. And the sentence the EXTERNAL-1
preregistration committed the lab to print is printed, in its words, in action.yml and README.md.

This is a word lint and these tests say so: a pass means no listed string was found. It is not a check that any
sentence is true (the lab's own measurements of finding claims in prose are in the script's docstring).

The charter's words are stored reversed, as in tests/test_leaderboard_no_priority_claim.py, so this file does not
spell them either.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "surface_lint.py"
BASELINE = ROOT / "scripts" / "surface_lint_baseline.json"
PREREG = ROOT / "papers" / "closed-model-frontier" / "PREREG_external1_aidev_2026_08_31.md"


def _load():
    spec = importlib.util.spec_from_file_location("surface_lint", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["surface_lint"] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load()


def _r(s: str) -> str:
    return s[::-1]


CHARTER = {
    "charter/ordinal": _r("tsrif"),
    "charter/newness-adjective": _r("levon"),
    "charter/revolution-adjective": _r("yranoitulover"),
    "charter/ground-adjective": _r("gnikaerbdnuorg"),
    "charter/break-noun": _r("hguorhtkaerb"),
    "charter/tamper-compound": _r("foorp-repmat"),
    "charter/self-compound": _r("gniyfirev-fles"),
    "charter/halluc-compound": _r("rotceted noitanicullah"),
}


def _rids(text: str):
    hits, allowed = L.lint_text(text, "x.md")
    return [h.rid for h in hits], [h.rid for h in allowed]


# ---- the rules, on text built here --------------------------------------------------------------------------

@pytest.mark.parametrize("rid,word", sorted(CHARTER.items()))
def test_each_charter_word_is_a_hit_and_no_excerpt_spells_it(rid, word):
    hits, _ = L.lint_text(f"A line with the {word.upper()} word in it.\n", "x.md")
    assert [h.rid for h in hits] == [rid]
    assert word.lower() not in hits[0].excerpt.lower() and f"[{rid}]" in hits[0].excerpt


def test_a_charter_word_inside_a_longer_word_is_not_a_hit():
    assert _rids(_r("tsrif") + "ly, and un" + _r("levon") + "ish\n") == ([], [])


@pytest.mark.parametrize("text,rid", [
    ("Your agent's PR summary cannot lie about its diff.", "withdrawn/gate-claim"),
    ("the agent can't lie about it", "withdrawn/gate-claim"),
    ("the agent cannot post a message that lies about its diff", "withdrawn/gate-claim"),
    ("a commit or PR command whose message lies about its diff", "withdrawn/gate-claim"),
    ("gate: the diff cannot be lied about", "withdrawn/gate-claim"),
    ("Every claim machine-verified against committed receipts", "withdrawn/every-claim-verified"),
    ("certify: every number checks out", "withdrawn/every-claim-verified"),
    ("Every number in a paper proves itself against committed receipts", "withdrawn/every-claim-verified"),
    ("Every number the lab reports is verified against a committed receipt", "withdrawn/every-claim-verified"),
    ("One line in CI; the lie exits 1.", "withdrawn/lie-label"),
    ('const marks = { CONTRADICTED: ["lie", "[LIE]"] };', "withdrawn/lie-label"),
    ("if we ever lie about a diff, our own product fails our own build", "withdrawn/own-build"),
    ("blocks only on kinds whose precision has been measured; today none has", "withdrawn/today-none-has"),
    ("Zero false accusations on our history", "withdrawn/zero-false-accusations"),
    ("the spans the author bound were checked against bytes the author did not write, at the commit",
     "withdrawn/certifies-v1"),
    ("The spans the author bound were checked against\nbytes the author did not write.", "withdrawn/certifies-v1"),
    # the exhibit agent's descriptions of its gate (AMENDMENT_darkflobi_gate_lint_overclaims_2026_10_25.md)
    ("total cryptographic accountability for my outbound mouth", "withdrawn/gate-cryptographic"),
    ("since they weren't cryptographically bound to a proof receipt", "withdrawn/gate-cryptographic"),
    ("found no cryptographic receipt tying them to a live execution", "withdrawn/gate-cryptographic"),
    ("gets hashed, evaluated, and permanently logged in `ledger.jsonl`", "withdrawn/gate-permanent"),
    ("ledger 0044 is the permanent receipt of that audit", "withdrawn/gate-permanent"),
    ("every single message i send goes through it. there is no bypass.", "withdrawn/gate-no-bypass"),
    ("it watches everything i say", "withdrawn/gate-no-bypass"),
    ("the canary system ensures no live secrets (keys, wallets) ever leak", "withdrawn/gate-canary-secrets"),
    ("instead of writing real keys to disk to test redaction, we plant synthetic canaries",
     "withdrawn/gate-canary-secrets"),
    ("if styxx can detect deception, it can penalize it during training", "withdrawn/gate-detects-deception"),
    ('an objective metric for "deception" to train against', "withdrawn/gate-detects-deception"),
    ("the foundation for a provably *honest* AI", "withdrawn/gate-detects-deception"),
    ("Nobody does this.", "plan/nobody"),
    ("a gap of 0.117 between", "plan/0.117"),
    ("Sworn makes agents honest", "plan/sworn-honest"),
    ("the contract every agent's claims are checked against", "plan/the-contract"),
    ("sworn is the wire format", "plan/wire-or-protocol-beside-sworn"),
    ("the sworn protocol", "plan/wire-or-protocol-beside-sworn"),
])
def test_each_listed_sentence_is_a_hit(text, rid):
    hits, allowed = _rids(text + "\n")
    assert rid in hits and not allowed, (hits, allowed)


def test_protocol_alone_is_not_a_hit_because_it_is_a_module_name():
    assert _rids("python -m styxx.protocol runs the preregistration gates\n") == ([], [])


@pytest.mark.parametrize("text", [
    "a contradicted claim is printed as [CONTRADICTED] with its reason",   # the label that replaced it
    "certify: every number vs its receipt",                                 # names what certify reads, claims nothing
    "a claim the author believed may still be a lie the diff shows",       # the word alone is not the label
    "every number in the table is from the run named beside it",           # not one of the listed wordings
    # the true sentences about the gate, and the repository's own uses of the neighbouring words
    "the ledger is hash-chained; nothing is signed and there is no key",
    "a test report not cryptographically bound to the receipt it cites",    # the evidence leg's words, not the agent's
    "the daily anchor shows the order; the weekly snapshot holds the bytes",
    "the canaries are planted falsehoods that measure the gate's catch rate",
    "It does not detect deception that is consistent with the reference",   # integrations/inspect_evals' README
    "programmed deployments, provably enforced by the server",              # docs/cognet-protocol-v0.md
    "redaction = redact.redactable_commit(core)",                            # styxx/attestation.py
    "a credential helper bypassed the hook",                                 # AUTOPILOT.md
])
def test_the_new_wordings_are_narrow(text):
    assert _rids(text + "\n") == ([], [])


def test_the_gate_wordings_are_allowed_beside_their_withdrawal():
    text = ('the sentence "there is no bypass" is withdrawn: with REQUIRED absent nothing is blocked, and the\n'
            "exec exit, the shell and GitHub writes are open.\n")
    assert _rids(text) == ([], ["withdrawn/gate-no-bypass"])


def test_an_excerpt_the_console_cannot_encode_is_escaped_not_fatal(tmp_path):
    import os
    import subprocess

    f = tmp_path / "draft.md"
    f.write_text("❓ nobody reads this\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    env["PYTHONIOENCODING"] = "cp1252"                     # what a Windows pipe gets without the variable
    r = subprocess.run([sys.executable, str(SCRIPT), "--files", str(f)], capture_output=True, env=env, timeout=120)
    assert r.returncode == 1 and b"Traceback" not in r.stderr, r.stderr.decode("utf-8", "replace")
    assert b"plan/nobody" in r.stdout and b"\\u2753" in r.stdout


def test_a_baseline_written_for_another_source_names_it(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<p>nobody</p>\n", encoding="utf-8")
    out = tmp_path / "site_baseline.json"
    assert L.main(["--profile", "site", "--root", str(site), "--write-baseline", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["profile"] == "site" and doc["source"] == site.as_posix()
    assert doc["counts"] == {"index.html": {"plan/nobody": 1}}
    assert L.main(["--profile", "site", "--root", str(site), "--baseline", str(out)]) == 0
    (site / "index.html").write_text("<p>nobody</p>\n<p>the diff cannot be lied about</p>\n", encoding="utf-8")
    assert L.main(["--profile", "site", "--root", str(site), "--baseline", str(out)]) == 1


def test_the_withdrawal_may_quote_what_it_withdraws_even_across_a_line_break():
    text = ('The claim "an agent\'s PR summary cannot lie about its diff" is not yet true of this\n'
            "instrument on real-world data.\n")
    assert _rids(text) == ([], ["withdrawn/gate-claim"])
    assert _rids("The zero-false-accusation claim that stood here is withdrawn.\n") == (
        [], ["withdrawn/zero-false-accusations"])


def test_a_withdrawal_far_away_does_not_excuse_a_claim():
    text = "the claim is withdrawn.\n" + ("filler " * 80) + "\nYour agent's summary cannot lie.\n"
    assert _rids(text) == (["withdrawn/gate-claim"], [])


def test_line_and_column_are_counted_with_crlf_read_as_lf():
    hits, _ = L.lint_text("ok\r\nok\r\n  nobody\r\n", "x.md")
    assert [(h.line, h.col) for h in hits] == [(3, 3)]


def test_the_baseline_lets_a_count_fall_and_never_rise_and_never_holds_a_withdrawn_sentence():
    hits, _ = L.lint_text("nobody\nnobody\ncannot lie\n", "a.md")
    nobody = [h for h in hits if h.rid == "plan/nobody"]
    problems = L.over_baseline(hits, {"counts": {"a.md": {"plan/nobody": 2, "withdrawn/gate-claim": 9}}})
    assert len(problems) == 1 and "withdrawn/gate-claim" in problems[0] and "never baselined" in problems[0]
    assert L.over_baseline(nobody, {"counts": {"a.md": {"plan/nobody": 3}}}) == []
    assert L.over_baseline(nobody, {"counts": {"a.md": {"plan/nobody": 1}}}) == ["a.md plan/nobody: 2 > baseline 1"]


def test_the_profile_keeps_records_and_fixtures_out():
    repo = L.PROFILES["repo"]
    for path in ("papers/closed-model-frontier/PREREG_external1_aidev_2026_08_31.md", "CHANGELOG.md",
                 "tests/test_diffgate.py", "web/gate/differential/bodies/pr98.md", "styxx/_data/x.json",
                 "styxx/__pycache__/diffgate.cpython-312.py"):
        assert not L.selected(path, repo), path
    for path in ("README.md", "action.yml", "styxx/diffgate.py", "integrations/git/commit-msg",
                 "integrations/claude-code/diffgate-hook/README.md", "web/gate/README.md", "sworn/action.yml"):
        assert L.selected(path, repo), path


# ---- the repository ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    return L.run(L.tree_files(ROOT, L.PROFILES["repo"]))


def test_the_lint_reads_the_front_door(report):
    assert report.files > 300, report.files


def test_no_withdrawn_sentence_stands_on_a_shipped_surface(report):
    bad = [f"{h.path}:{h.line}:{h.col} {h.rid} | {h.excerpt}" for h in report.hits if h.kind == "withdrawn"]
    assert not bad, "a withdrawn sentence is back on a shipped surface:\n" + "\n".join(bad)


def test_no_shipped_file_carries_more_listed_words_than_the_baseline(report):
    problems = L.over_baseline(report.hits, json.loads(BASELINE.read_text(encoding="utf-8")))
    assert not problems, ("a shipped file carries more of the charter's or the plan's listed words than "
                          "scripts/surface_lint_baseline.json allows:\n" + "\n".join(problems))


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_the_sentence_external1_committed_to_is_printed_on_the_front_door():
    """The PREREG (lines 93-95) bound the lab to say the claim 'is not yet true of this instrument on real-world
    data' in those words. The words are read from the PREREG, so this holds the printed sentence to its source."""
    prereg = _flat(PREREG.read_text(encoding="utf-8"))
    m = re.search(r'the claim "(an agent\'s PR summary cannot lie about its diff)" (is not yet true of this '
                  r'instrument on real-world data)', prereg)
    assert m, "the PREREG no longer carries the committed sentence in the form this test reads"
    sentence = f'The claim "{m.group(1)}" {m.group(2)}'
    for rel in ("action.yml", "README.md"):
        assert sentence in _flat((ROOT / rel).read_text(encoding="utf-8")), f"{rel} does not print: {sentence}"


def test_the_script_imports_only_the_standard_library_and_spells_no_charter_word():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    if hasattr(sys, "stdlib_module_names"):
        assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods
    for path in (SCRIPT, Path(__file__)):
        hits, _ = L.lint_text(path.read_text(encoding="utf-8"), path.name)
        assert not [h for h in hits if h.kind == "charter"], path.name
