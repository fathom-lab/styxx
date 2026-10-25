# -*- coding: utf-8 -*-
"""prereg-ci B4: the locks decision, the locked scorer run, and the checksum scorer as a locks decision.

DESIGN 3.2(a): a PREREG's decision is exactly one of two forms, a ```gates table or one ```locks
fence "scanned with the same rules as the gates fence", holding ``{"scorer": PATH, "reads": [PATH,
...], "outcomes": [VERDICT, ...], "withdraws": {VERDICT: [...]}}``; "the sealed blob of each is what
runs". 3.2(f): ``scorer_run.py`` "writes each locked path's sealed blob into an empty temporary
directory, at the same relative path", runs the scorer under ``python -I -S`` with a 60-second
timeout and the receipt copied in, and "a read of any unlocked file fails, so the run reads FAIL
SCORER_READS_UNLOCKED or SCORER_ERROR with the scorer's last stderr line". PLAN K2b adds two
optional keys, ``argv`` (``{receipt}`` and ``{out}`` placeholders) and ``verdict_from`` (``stdout``,
or an RFC 6901 pointer into the ``{out}`` JSON), so a scorer with its own command line is locked as
it stands; PLAN K3: a ``-c`` bootstrap sets ``sys.path`` to the run directory and the standard
library and runs the scorer with ``runpy``, and ``-B`` keeps caches off the disk. Not a sandbox
(DESIGN 7, risk 5).

Done when (DESIGN 8, B4, restated by PLAN K2b): ``papers/checksum/score.py`` at its sealed blob, run
by the scorer run over the dry-run certs, re-derives the committed dry-run card; a scorer that opens
an unlocked file reads SCORER_READS_UNLOCKED or SCORER_ERROR; the card is an instrument check, not a
result. The checksum PREREGs carry no locks fence and, sealed, cannot gain one, so the locks block is
a RECONSTRUCTION written for this step (``conformance/prereg_ci/locks/``), never sealed and not a
catch. Every committed version of the deploy-quant dry-run card is re-derived here at the commit that
wrote it, byte for byte; the beacon-draw path needs numpy through ``styxx.checksum``, which ``-S``
removes, and its scorer swallows that import failure into a K5 line, so the run reads
SCORER_READS_UNLOCKED there and the card is not re-derived.

Every repository read is this one (read only) or a ``git init`` in tmp_path; every scorer runs in a
temporary directory with no remote, and a socket call is refused before it connects.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci import REV  # noqa: E402
from styxx import protocol  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from styxx.prereg_ci import git as G  # noqa: E402
from styxx.prereg_ci import scorer_run as S  # noqa: E402

F = "`" * 3
LOCKS_FILE = ROOT / "conformance" / "prereg_ci" / "locks" / "LOCKS_checksum_deploy_quant_reconstructed.md"
GEN = ROOT / "conformance" / "prereg_ci" / "gen_locks_checksum.py"

SCORE = "papers/checksum/score.py"
CERTS = "papers/checksum/deploy_quant_certs_dryrun_qwen0.5b.json"
CARD_V1 = "papers/checksum/deploy_quant_scorecard_dryrun_qwen0.5b.json"
CARD_V2 = "papers/checksum/deploy_quant_scorecard_v2_dryrun_qwen0.5b.json"
BEACON_CERTS = "papers/checksum/beacon_draw_certs_dryrun_qwen0.5b.json"
BEACON_CARD_V1 = "papers/checksum/beacon_draw_scorecard_dryrun_qwen0.5b.json"
BEACON_CARD_V2 = "papers/checksum/beacon_draw_scorecard_v2_dryrun_qwen0.5b.json"
# every commit that wrote a deploy-quant dry-run card, with the card it wrote (git log over both cards)
DEPLOY_CARDS = (
    ("383b4155cb92f9234434f17ef23ce52a59b08211", CARD_V1),
    ("4e08cf284b9722cac5ed3d80c28f34385d74f96a", CARD_V2),
    ("a3a26ee1094d6d5b6f55c93bca43a47530359829", CARD_V2),
    ("f0ec56056fc6c903e7cfbf70a1b17c9f5ba1574e", CARD_V2),
)
# every commit that wrote a beacon-draw dry-run card
BEACON_CARDS = (
    ("383b4155cb92f9234434f17ef23ce52a59b08211", BEACON_CARD_V1),
    ("4e08cf284b9722cac5ed3d80c28f34385d74f96a", BEACON_CARD_V2),
    ("cb63928e9f941aec629c135416734c7a15472e11", BEACON_CARD_V2),
    ("a3a26ee1094d6d5b6f55c93bca43a47530359829", BEACON_CARD_V2),
)
SEALED_SCORER = "f0ec56056fc6c903e7cfbf70a1b17c9f5ba1574e"     # where score.py was last written
CERTS_COMMIT = "2b4c3dcd"                                      # where the dry-run certs were written, 2026-09-13
DRY_READING = "INSTRUMENT CHECK; gates fired: provenance; H1 FAILED, H2 FAILED, H3 HELD, H4 HELD"
BEACON_READING = ("NOT A RESULT: K5's beacon clause not evaluated (no --expect-beacon); INSTRUMENT CHECK; gates "
                  "fired: K5; H1 HELD, H2 HELD, H3 HELD, H4 HELD, H5 PENDING, H6 PENDING")
# a cert key that holds the ordinal the house does not print, built from its reversed spelling
HITS_KEY = "tsrif"[::-1] + "_token_top1_hits"
# the house words and priority phrases, reversed (as tests/test_prereg_ci_package.py holds them)
_HOUSE = [w[::-1] for w in ("tsrif", "levon", "yranoitulover", "gnikaerbdnuorg", "hguorhtkaerb",
                            "foorp-repmat", "gniyfirev-fles", "rotceted noitanicullah")]
HOUSE_RX = re.compile("|".join(re.escape(w).replace(r"\-", r"[\s\-]?").replace(r"\ ", r"[\s\-]+")
                               for w in _HOUSE), re.IGNORECASE)
PRIORITY_RX = re.compile("|".join(re.escape(w[::-1]).replace(r"\ ", r"\s+") + r"\b"
                                  for w in ("sah eno on", "ylno eht")), re.IGNORECASE)


def locks_prereg(spec=None, prose: str = "notes") -> str:
    spec = spec if spec is not None else {"scorer": "score.py", "reads": ["data.csv"], "outcomes": ["PASS", "HOLD"]}
    return f"# PREREG\n\n{prose}\n\n{F}locks\n{json.dumps(spec, indent=1)}\n{F}\n"


GATES = json.dumps({"gates": {"G": {"metric": "m", "op": ">=", "value": 0.5}},
                    "outcomes": [{"when": {"G": True}, "verdict": "PASS"}, {"when": {"G": False}, "verdict": "HOLD"}],
                    "smoke_verdict": "SMOKE"}, indent=1)


def spec(**kw):
    base = {"scorer": "score.py", "reads": [], "outcomes": ["PASS", "HOLD"]}
    base.update(kw)
    return core._locks(json.dumps(base))


# ---- the decision forms (DESIGN 3.2(a)) ------------------------------------------------------------------

def test_a_prereg_with_one_locks_fence_reads_the_locks_form_and_its_digest():
    text = locks_prereg()
    d = core._decision(text)
    body = text.split(F + "locks\n", 1)[1].split("\n" + F, 1)[0]
    assert d["form"] == "locks" and d["code"] is None and d["block"] == body
    assert d["gates_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert core._decision(text.replace("\n", "\r\n").encode("utf-8"))["gates_sha256"] == d["gates_sha256"]


def test_a_gates_prereg_reads_as_before():
    d = core._decision(f"# P\n\n{F}gates\n{GATES}\n{F}\n")
    assert d["form"] == "gates" and d["gates_sha256"] == hashlib.sha256(GATES.encode()).hexdigest()
    assert core._decision("# P\n\nprose\n")["code"] == "NOT_GATED"


@pytest.mark.parametrize("text, words", [
    (f"# P\n\n{F}gates\n{GATES}\n{F}\n\n{F}locks\n{{}}\n{F}\n", "one decision"),
    (f"# P\n\n{F}locks\n{{}}\n{F}\n\n{F}gates\n{GATES}\n{F}\n", "one decision"),
    (f"# P\n\n{F}gates\n{GATES}\n{F}\n\n> {F}locks\n> {{}}\n> {F}\n", "block quote"),
    (f"# P\n\n{F}locks\n{{}}\n{F}\n\n{F}locks\n{{}}\n{F}\n", "2 locks-like fenced blocks"),
    (f"# P\n\n<!--\n{F}locks\n{{}}\n{F}\n-->\n", "HTML comment"),
    ("# P\n\n~~~locks\n{}\n~~~\n", "not a plain unindented"),
    (f"# P\n\n{F}locks\n{{}}\n", "never closed"),
    (f"# P\n\n- item\n\n    {F}locks\n    {{}}\n    {F}\n", "inside a container"),
])
def test_a_second_form_or_a_locks_fence_the_scanner_refuses_is_a_refusal(text, words):
    d = core._decision(text)
    assert d["form"] is None and d["code"] == "SCORER_REFUSES", d
    assert words in d["detail"], d["detail"]


def test_a_gates_refusal_keeps_its_words_when_a_locks_fence_is_there_too():
    """The gates scanner reads before the locks reader, so a refused gates PREREG reads as it did."""
    text = f"# P\n\n{F}gates\n{GATES}\n{F}\n\n{F}gates\n{GATES}\n{F}\n\n{F}locks\n{{}}\n{F}\n"
    d = core._decision(text)
    with pytest.raises(protocol.GateSpecError) as e:
        protocol._select_gates_block(text)
    assert d["code"] == "SCORER_REFUSES" and d["detail"] == str(e.value)


def test_a_locks_fence_that_is_not_json_still_has_a_digest_and_rescore_refuses_it():
    text = f"# P\n\n{F}locks\n{{not json\n{F}\n"
    d = core._decision(text)
    assert d["form"] == "locks" and d["gates_sha256"] == hashlib.sha256(b"{not json").hexdigest()
    r = core.rescore(text, {})
    assert r["form"] == "locks" and r["code"] == "SCORER_REFUSES" and "strict JSON" in r["detail"]


def test_rescore_of_a_locks_prereg_says_the_scorer_was_not_run():
    r = core.rescore(locks_prereg(), {"prereg": "PREREG.md"})
    assert r["form"] == "locks" and r["code"] == "SCORER_NOT_RUN" and r["verdict"] is None
    assert "scorer_run" in r["detail"] and r["gates_sha256"] == core._decision(locks_prereg())["gates_sha256"]
    assert core.compare(r, {"verdict": "PASS"})[0]["code"] == "SCORER_NOT_RUN"
    assert core._KIND["SCORER_NOT_RUN"] == "UNCHECKABLE"
    assert set(r) == set(core.rescore(f"# P\n\n{F}gates\n{GATES}\n{F}\n", {"m": 1}))


# ---- the locks block (DESIGN 3.2(a), PLAN K2b) -----------------------------------------------------------

def test_a_minimal_locks_block_takes_the_defaults():
    s = spec()
    assert s["argv"] == ["{receipt}"] and s["verdict_from"] == "stdout" and s["withdraws"] == {}
    assert s["locked"] == ["score.py"] and s["outcomes"] == ["PASS", "HOLD"]
    s = spec(reads=["lib/a.py", "data/x.csv"], argv=["--in", "{receipt}", "--out", "{out}"],
             verdict_from="/verdict", withdraws={"HOLD": []})
    assert s["locked"] == ["score.py", "lib/a.py", "data/x.csv"] and s["verdict_from"] == "/verdict"


@pytest.mark.parametrize("change, words", [
    ({"extra": 1}, "keys the format does not name"),
    ({"scorer": None}, "scorer"),
    ({"scorer": "/abs/score.py"}, "repository-relative"),
    ({"scorer": "a\\score.py"}, "repository-relative"),
    ({"scorer": "../score.py"}, "'..'"),
    ({"scorer": "CON/score.py"}, "portable"),
    ({"reads": "data.csv"}, "list"),
    ({"reads": ["a.csv", "a.csv"]}, "twice"),
    ({"reads": ["score.py"]}, "twice"),
    ({"reads": ["d/A.csv", "d/a.csv"]}, "case"),
    ({"reads": ["d", "d/a.csv"]}, "directory"),
    ({"reads": [".prereg-ci/x.json"]}, ".prereg-ci"),
    ({"reads": [".PREREG-CI/x.json"]}, ".prereg-ci"),
    ({"outcomes": []}, "outcomes"),
    ({"outcomes": "PASS"}, "outcomes"),
    ({"outcomes": ["PASS", "PASS"]}, "twice"),
    ({"outcomes": ["PASS", ""]}, "non-empty"),
    ({"outcomes": ["PASS", "a\nb"]}, "one line"),
    ({"outcomes": ["PASS", "PASS "]}, "whitespace"),
    ({"outcomes": ["PASS", " HOLD"]}, "whitespace"),
    ({"outcomes": ["PASS", "HOLD "]}, "whitespace"),
    ({"outcomes": ["PASS", 3]}, "non-empty"),
    ({"withdraws": []}, "withdraws"),
    ({"withdraws": {"SHIP": []}}, "not a declared outcome"),
    ({"withdraws": {"PASS": {}}}, "list"),
    ({"argv": "{receipt}"}, "argv"),
    ({"argv": ["--x"]}, "{receipt}"),
    ({"argv": ["{receipt}", "{receipt}"]}, "{receipt}"),
    ({"argv": ["{receipt}", "{out}", "{out}"]}, "{out}"),
    ({"argv": ["--in={receipt}"]}, "whole"),
    ({"argv": ["{receipt}", 3]}, "argv"),
    ({"verdict_from": "verdict"}, "verdict_from"),
    ({"verdict_from": "/verdict"}, "{out}"),
])
def test_a_locks_block_the_format_does_not_read_is_refused(change, words):
    base = {"scorer": "score.py", "reads": [], "outcomes": ["PASS", "HOLD"]}
    base.update(change)
    with pytest.raises(core._GateSpecError) as e:
        core._locks(json.dumps(base))
    assert words in str(e.value), str(e.value)


@pytest.mark.parametrize("missing", ["scorer", "reads", "outcomes"])
def test_a_locks_block_without_a_required_key_is_refused(missing):
    base = {"scorer": "score.py", "reads": [], "outcomes": ["PASS"]}
    del base[missing]
    with pytest.raises(core._GateSpecError, match=missing):
        core._locks(json.dumps(base))


def test_a_locks_block_that_is_not_an_object_or_not_strict_json_is_refused():
    for bad in ("[]", '{"scorer": "a.py", "scorer": "b.py", "reads": [], "outcomes": ["P"]}', "{"):
        with pytest.raises(core._GateSpecError):
            core._locks(bad)


# ---- the reading of a run, pure (core._run_reading) ------------------------------------------------------

def run(**kw):
    base = {"exit": 0, "timed_out": False, "timeout_s": 60, "done": True, "stdout": b'{"verdict": "PASS"}\n',
            "stdout_over": False, "stderr_tail": "", "out": None, "out_problem": None, "unlocked": []}
    base.update(kw)
    return base


def test_one_json_line_on_stdout_is_the_verdict():
    r = core._run_reading(spec(), run())
    assert (r["code"], r["verdict"], r["gates"]) == (None, "PASS", None)
    r = core._run_reading(spec(), run(stdout=b'{"verdict": "HOLD", "gates": {"G": false}}'))
    assert (r["code"], r["verdict"], r["gates"]) == (None, "HOLD", {"G": False})


@pytest.mark.parametrize("stdout, words", [
    (b'{"verdict": "PASS"}\n{"verdict": "PASS"}\n', "one line"),
    (b"PASS\n", "JSON"),
    (b'["PASS"]\n', "object"),
    (b'{"verdict": 1}\n', "string"),
    (b'{"verdict": "PASS", "note": "x"}\n', "verdict and gates"),
    (b'{"verdict": "PASS", "gates": []}\n', "gates"),
    (b'{"verdict": "SHIP"}\n', "does not declare"),
    (b"", "one line"),
    (b'{"verdict": "PASS", "verdict": "HOLD"}\n', "JSON"),
    (b'\xff\n', "UTF-8"),
])
def test_a_stdout_the_contract_does_not_read_is_a_scorer_error(stdout, words):
    r = core._run_reading(spec(), run(stdout=stdout))
    assert r["code"] == "SCORER_ERROR" and r["verdict"] is None and words in r["detail"], r["detail"]


def test_a_run_that_failed_is_a_scorer_error_with_its_last_stderr_line():
    r = core._run_reading(spec(), run(exit=3, stderr_tail="ValueError: boom"))
    assert r["code"] == "SCORER_ERROR" and "exited 3" in r["detail"] and "ValueError: boom" in r["detail"]
    r = core._run_reading(spec(), run(exit=None, timed_out=True, done=False))
    assert r["code"] == "SCORER_ERROR" and "60 s" in r["detail"]
    r = core._run_reading(spec(), run(done=False))
    assert r["code"] == "SCORER_ERROR" and "closing line" in r["detail"]
    r = core._run_reading(spec(), run(stdout_over=True))
    assert r["code"] == "SCORER_ERROR" and "printed more" in r["detail"]


def test_an_unlocked_read_is_refused_whatever_the_scorer_printed_or_exited():
    for kw in ({}, {"exit": 2}, {"done": False}, {"stdout": b"junk"}, {"timed_out": True, "exit": None}):
        r = core._run_reading(spec(), run(unlocked=["opens data/other.csv to read"], **kw))
        assert r["code"] == "SCORER_READS_UNLOCKED" and "data/other.csv" in r["detail"] and r["verdict"] is None
    assert core._KIND["SCORER_READS_UNLOCKED"] == "FAIL" and core._KIND["SCORER_ERROR"] == "FAIL"


def test_a_verdict_read_from_the_out_file_by_pointer():
    s = spec(argv=["{receipt}", "--out", "{out}"], verdict_from="/card/reading")
    r = core._run_reading(s, run(stdout=b"free text\nmore\n", out=b'{"card": {"reading": "HOLD"}}\n'))
    assert (r["code"], r["verdict"], r["output"]) == (None, "HOLD", {"card": {"reading": "HOLD"}})
    for out, problem, words in ((None, "the scorer wrote no {out} file", "no {out}"), (b"{", None, "JSON"),
                                (b'{"card": {}}', None, "does not resolve"),
                                (b'{"card": {"reading": 1}}', None, "string"),
                                (b'{"card": {"reading": "SHIP"}}', None, "does not declare")):
        r = core._run_reading(s, run(out=out, out_problem=problem))
        assert r["code"] == "SCORER_ERROR" and words in r["detail"], (out, r["detail"])


# ---- the scorer run (scorer_run, DESIGN 3.2(f), PLAN K3) ---------------------------------------------------

def scorer(src: str) -> bytes:
    return textwrap.dedent(src).lstrip("\n").encode("utf-8")


def go(src: str, files=None, s=None, receipt=b'{"prereg": "PREREG.md", "m": 0.7}\n', **kw):
    s = s or spec(reads=sorted((files or {}).keys()))
    blobs = {"score.py": scorer(src), **(files or {})}
    raw = S._run(s, blobs, "runs/receipt.json", receipt, **kw)
    return raw, core._run_reading(s, raw)


def test_a_scorer_that_reads_its_locked_file_and_the_receipt_gives_its_verdict():
    raw, r = go("""
        import json, sys
        rec = json.load(open(sys.argv[1], encoding="utf-8"))
        bar = float(open("data/bar.txt", encoding="utf-8").read())
        print(json.dumps({"verdict": "PASS" if rec["m"] >= bar else "HOLD", "gates": {"G": rec["m"] >= bar}}))
        """, {"data/bar.txt": b"0.5\n"})
    assert (r["code"], r["verdict"], r["gates"]) == (None, "PASS", {"G": True}), (r, raw["stderr_tail"])
    assert raw["exit"] == 0 and raw["done"] and raw["unlocked"] == []
    assert set(raw["runtime"]) == {"implementation", "version", "unicode"}
    assert raw["runtime"]["implementation"] == sys.implementation.name


def test_the_receipt_and_the_out_file_are_named_by_relative_paths_in_the_run():
    raw, r = go("""
        import json, os, sys
        ok = sys.argv[1:] == ["runs/receipt.json", ".prereg-ci/out.json"] and os.path.isfile(sys.argv[1])
        json.dump({"v": "PASS" if ok else "HOLD"}, open(sys.argv[2], "w", encoding="utf-8"))
        print(open(sys.argv[2], encoding="utf-8").read())
        """, s=spec(argv=["{receipt}", "{out}"], verdict_from="/v"))
    assert (r["code"], r["verdict"]) == (None, "PASS"), (r, raw["stderr_tail"])


def test_the_run_holds_the_locked_files_and_the_standard_library_only():
    raw, r = go("""
        import json, os, sys
        here = os.path.realpath(os.getcwd())
        ok = (os.path.realpath(sys.path[0]) == here
              and not any("site-packages" in p or "dist-packages" in p for p in sys.path)
              and sorted(os.listdir(here)) == [".prereg-ci", "runs", "score.py"]
              and sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.flags.dont_write_bytecode == 1)
        print(json.dumps({"verdict": "PASS" if ok else "HOLD"}))
        """)
    assert (r["code"], r["verdict"]) == (None, "PASS"), (r, raw["stderr_tail"])


def test_a_locked_module_beside_the_scorer_imports():
    raw, r = go("""
        import json
        from lib.rule import verdict
        print(json.dumps({"verdict": verdict()}))
        """, {"lib/rule.py": b"def verdict():\n    return 'HOLD'\n"})
    assert (r["code"], r["verdict"]) == (None, "HOLD"), (r, raw["stderr_tail"])


def test_a_scorer_that_puts_its_own_directories_on_sys_path_does_not_alter_the_import_system():
    """The close reads the import system as the bootstrap set it; a scorer's own directories under the
    run, inserted at the head of sys.path (papers/checksum/score.py's idiom, with the repository root it
    computes from __file__), are not an alteration."""
    s = spec(scorer="tools/score.py", reads=["tools/lib/rule.py"])
    src = scorer("""
        import json, os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from rule import verdict
        print(json.dumps({"verdict": verdict()}))
        """)
    raw = S._run(s, {"tools/score.py": src, "tools/lib/rule.py": b"def verdict():\n    return 'HOLD'\n"},
                 "r.json", b"{}")
    r = core._run_reading(s, raw)
    assert (r["code"], r["verdict"], raw["unlocked"]) == (None, "HOLD", []), (r, raw["stderr_tail"])


def test_the_standard_library_s_own_optional_imports_are_not_reaches():
    """A stdlib module that tries an optional import and falls back (tarfile's pwd on Windows, the
    C accelerators elsewhere) is not the scorer reaching past its locks: only an import the locked
    code itself asks for is read."""
    raw, r = go("""
        import argparse, bisect, csv, datetime, decimal, fractions, hashlib, heapq, json, math, pickle
        import random, statistics, string, struct, tarfile, textwrap, zlib
        print(json.dumps({"verdict": "PASS"}))
        """)
    assert (r["code"], r["verdict"]) == (None, "PASS"), (r, raw)


@pytest.mark.parametrize("body, words", [
    ("open('data/other.csv').read()", "data/other.csv"),
    ("open(OUTSIDE).read()", "outside the run"),
    ("import numpy", "imports numpy"),
    ("import styxx", "imports styxx"),
    ("import importlib; importlib.import_module('not_a_module_here')", "imports not_a_module_here"),
    ("exec('import not_a_module_either', globals())", "imports not_a_module_either"),
    ("import subprocess, sys; subprocess.run([sys.executable, '-c', '0'])", "starts a process"),
    ("import os; os.system('echo x')", "starts a process"),
    ("import socket; socket.create_connection(('127.0.0.1', 9), timeout=1)", "socket"),
    ("open('elsewhere.json', 'w').write('{}')", "for writing"),
    ("open('score.py', 'a').write('#')", "for writing"),
    ("import os; os.remove('score.py')", "changes the file system"),
    ("import os; os.listdir(os.path.dirname(OUTSIDE))", "lists a directory outside the run"),
    ("import sys; sys.path.append(os.path.dirname(OUTSIDE)); import outside_mod", "outside_mod"),
    ("import sys; open(sys.executable, 'rb').read(4)", "outside the run"),
    ("exec(compile('import numpy', '<string>', 'exec'), {'__name__': 'x'})", "imports numpy"),
    ("import sys; sys.meta_path.pop(); import numpy", "altered the import system (the last import finder"),
    ("import sys; sys.path.pop(0)", "altered the import system (the run directory is off"),
    ("import sys; sys.path.remove(sys.path[0]); sys.path.append(os.getcwd())", "standard library is ahead"),
    ("import sys; sys.path.append(os.path.dirname(OUTSIDE))", "sys.path holds a directory outside"),
])
def test_a_scorer_that_reaches_past_its_locks_reads_unlocked_even_when_it_catches_the_error(body, words, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    (tmp_path / "outside_mod.py").write_text("X = 1\n", encoding="utf-8")
    raw, r = go(f"""
        import json, os
        OUTSIDE = {str(outside)!r}
        try:
            {body}
        except Exception:
            pass
        print(json.dumps({{"verdict": "PASS"}}))
        """)
    assert r["code"] == "SCORER_READS_UNLOCKED" and words in r["detail"], (r, raw)
    assert str(tmp_path) not in r["detail"] and str(outside) not in r["detail"]
    assert raw["stdout"] == b'{"verdict": "PASS"}\n' and raw["exit"] == 0 and raw["done"]


def test_a_file_in_the_interpreter_s_site_packages_is_not_standard_library():
    """-S leaves site packages off sys.path; a scorer that opens a file there by its path reaches past
    its locks even where that directory sits under the standard library's (Lib/site-packages here)."""
    import sysconfig
    purelib = Path(sysconfig.get_paths()["purelib"])
    found = sorted(p for p in purelib.iterdir() if p.is_file()) if purelib.is_dir() else []
    if not found:
        pytest.skip("this interpreter's site-packages holds no file to read")
    raw, r = go(f"""
        import json
        try:
            open({str(found[0])!r}, "rb").read()
        except OSError:
            pass
        print(json.dumps({{"verdict": "PASS"}}))
        """)
    assert r["code"] == "SCORER_READS_UNLOCKED" and "outside the run" in r["detail"], (r, raw)


def test_a_scorer_error_carries_its_last_stderr_line():
    raw, r = go("""
        raise ValueError("boom at the bar")
        """)
    assert r["code"] == "SCORER_ERROR" and "ValueError: boom at the bar" in r["detail"], r
    raw, r = go("""
        import sys
        sys.exit(4)
        """)
    assert r["code"] == "SCORER_ERROR" and "exited 4" in r["detail"], r


def test_the_last_stderr_line_names_the_run_directory_without_printing_it():
    """DESIGN 3.2(e): no absolute path in the report. A traceback naming the scorer's own file names
    the run directory as <run>."""
    raw, r = go("""
        raise ValueError("at " + __file__)
        """)
    assert r["code"] == "SCORER_ERROR" and "<run>" in r["detail"] and "prereg-ci-run-" not in r["detail"], r


def test_the_last_stderr_line_names_the_interpreter_and_the_home_directory_without_printing_them():
    """DESIGN 3.2(e): no absolute path. The interpreter's directories and the home directory are named,
    as the run's are (the B4 review's case 6: sys.executable printed to stderr reached the detail)."""
    home = os.path.expanduser("~")
    raw, r = go(f"""
        import sys
        print(sys.executable, sys.prefix, {home!r}, file=sys.stderr)
        sys.exit(3)
        """)
    d = r["detail"]
    assert r["code"] == "SCORER_ERROR" and "<python>" in d and "<home>" in d, d
    for p in (sys.executable, sys.prefix, home):
        assert p not in d and p.replace("\\", "/") not in d, d
    assert "Users" not in d and ":/" not in d and ":\\" not in d, d
    assert all(os.path.splitdrive(form)[1].strip("/\\") for form, _ in S._masks("x", "y"))


def test_a_scorer_that_leaves_without_the_closing_line_is_an_error():
    raw, r = go("""
        import json, os, sys
        print(json.dumps({"verdict": "PASS"}))
        sys.stdout.flush()
        os._exit(0)
        """)
    assert raw["exit"] == 0 and not raw["done"]
    assert r["code"] == "SCORER_ERROR" and "closing line" in r["detail"], r


def test_a_runtime_line_the_scorer_writes_to_the_record_is_a_problem_and_the_opening_line_is_the_runtime():
    """The record's descriptor is open in the scorer's process and os.write raises no audit event, so a
    scorer can write lines to it (the B4 review's case 3 forged the runtime, and the last runtime line
    won). The bootstrap writes the runtime line as the opening line, with the nonce the parent gave it
    outside the run; a runtime line anywhere else, or without the nonce, is a problem and the opening
    line is the runtime."""
    raw, r = go("""
        import json, os
        line = (json.dumps({"t": "runtime", "implementation": "forged", "version": "0.0.0", "unicode": "0"}) + "\\n").encode()
        for fd in range(3, 64):
            try:
                os.write(fd, line)
            except OSError:
                pass
        print(json.dumps({"verdict": "PASS"}))
        """)
    assert raw["runtime"]["implementation"] == sys.implementation.name and raw["runtime"]["version"] != "0.0.0", raw
    assert r["code"] == "SCORER_ERROR" and "runtime line" in r["detail"], (r, raw)


def test_a_closing_line_the_scorer_writes_itself_is_not_the_run_s():
    """A scorer that wrote a done line to the record and then ended the process itself would read as a
    scorer that returned, and whatever the bootstrap checks at the close would not run; the closing
    line carries the nonce, which the scorer does not hold (the config file's read is refused)."""
    raw, r = go("""
        import json, os, sys
        line = (json.dumps({"t": "done"}) + "\\n").encode()
        for fd in range(3, 64):
            try:
                os.write(fd, line)
            except OSError:
                pass
        print(json.dumps({"verdict": "PASS"}))
        sys.stdout.flush()
        os._exit(0)
        """)
    assert not raw["done"] and r["code"] == "SCORER_ERROR" and "closing line" in r["detail"], (r, raw)


def test_a_scorer_past_its_timeout_is_stopped():
    raw, r = go("""
        import time
        time.sleep(30)
        """, timeout=2)
    assert raw["timed_out"] and r["code"] == "SCORER_ERROR" and "2 s" in r["detail"], r


def test_the_streams_are_utf8_whatever_the_console_and_a_print_is_one_lf_line():
    raw, r = go("""
        import json, sys
        print("\\u2264 \\u0394 \\u2032", file=sys.stderr)
        print(json.dumps({"verdict": "HOLD"}))
        """)
    assert (r["code"], r["verdict"]) == (None, "HOLD") and raw["stdout"] == b'{"verdict": "HOLD"}\n', raw
    assert raw["stderr_tail"] == "\u2264 \u0394 \u2032"


def test_the_parent_environment_does_not_reach_the_run(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "planted.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(lib))
    monkeypatch.setenv("PRE_REG_CI_PROBE", "1")
    raw, r = go("""
        import json, os
        ok = "PRE_REG_CI_PROBE" not in os.environ
        try:
            import planted
            ok = False
        except ImportError:
            pass
        print(json.dumps({"verdict": "PASS" if ok else "HOLD"}))
        """)
    assert r["code"] == "SCORER_READS_UNLOCKED" and "imports planted" in r["detail"], r
    assert raw["stdout"] == b'{"verdict": "PASS"}\n'


def test_the_run_directory_is_removed_and_the_repository_is_not_written(tmp_path):
    before = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True).stdout
    raw, r = go("""
        import json, os
        print(json.dumps({"verdict": "PASS", "gates": {"dir": os.getcwd()}}))
        """)
    assert r["code"] is None and not os.path.exists(r["gates"]["dir"]) and raw["removed"]
    after = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True).stdout
    assert before == after


def test_a_receipt_on_a_locked_path_or_in_the_run_s_own_directory_is_refused():
    for path in ("score.py", ".prereg-ci/r.json", "SCORE.PY", "../r.json", "score.py/r.json"):
        r = S._rescore_locked(locks_prereg({"scorer": "score.py", "reads": [], "outcomes": ["PASS"]}).encode(),
                              path, b"{}", {"score.py": b"print('{}')\n"})
        assert r["code"] == "SCORER_REFUSES" and "receipt" in r["detail"] and r["run"] is None, (path, r)


def test_a_locked_file_the_caller_did_not_hand_over_is_refused():
    r = S._rescore_locked(locks_prereg().encode(), "r.json", b"{}", {"score.py": b"print('{}')\n"})
    assert r["code"] == "SCORER_REFUSES" and "data.csv" in r["detail"] and r["run"] is None, r


def test_rescore_locked_has_rescore_s_shape_and_names_what_ran():
    prereg = locks_prereg({"scorer": "score.py", "reads": ["data.csv"], "outcomes": ["PASS", "HOLD"]}).encode()
    src = b"import json\nopen('data.csv').read()\nprint(json.dumps({'verdict': 'HOLD'}))\n"
    r = S._rescore_locked(prereg, "r.json", b"{}", {"score.py": src, "data.csv": b"1\n"})
    assert set(core.rescore(f"# P\n\n{F}gates\n{GATES}\n{F}\n", {"m": 1})) <= set(r)
    assert (r["form"], r["code"], r["verdict"]) == ("locks", None, "HOLD")
    assert r["gates_sha256"] == core._decision(prereg)["gates_sha256"]
    assert r["run"]["locked"] == [{"path": "score.py", "sha256": hashlib.sha256(src).hexdigest()},
                                  {"path": "data.csv", "sha256": hashlib.sha256(b"1\n").hexdigest()}]
    assert r["run"]["receipt"] == {"path": "r.json", "sha256": hashlib.sha256(b"{}").hexdigest()}
    assert set(r["run"]["runtime"]) == {"implementation", "version", "unicode"}
    assert r["run"]["network"].startswith(("not isolated", "removed"))
    assert "watched" in r["run"] and "not a sandbox" in r["run"]["watched"]
    assert [f["code"] for f in core.compare(r, {"verdict": "HOLD"})] == ["RECORDED_DIGEST_ABSENT"]
    assert core.compare(r, {"verdict": "PASS"})[-1]["code"] == "VERDICT_DIFFERS"


def test_compare_returns_each_code_of_the_locks_form_with_its_kind():
    """The three codes tests/test_prereg_ci_compare_vectors.py leaves to this file: compare returns a
    locks run's refusal as its own finding, and reads no recorded field beside it."""
    prereg = locks_prereg({"scorer": "score.py", "reads": [], "outcomes": ["PASS"]})
    got = {}
    got["SCORER_NOT_RUN"] = core.rescore(prereg, {})
    got["SCORER_ERROR"] = S._rescore_locked(prereg.encode(), "r.json", b"{}", {"score.py": b"raise SystemExit(3)\n"})
    got["SCORER_READS_UNLOCKED"] = S._rescore_locked(prereg.encode(), "r.json", b"{}",
                                                     {"score.py": b"import numpy\n"})
    for code, r in got.items():
        found = core.compare(r, {"verdict": "PASS", "gates_sha256": "0" * 64})
        assert [(f["code"], f["kind"]) for f in found] == [(code, core._KIND[code])], (code, found)
    assert {core._KIND[c] for c in got} == {"UNCHECKABLE", "FAIL"}


def test_rescore_locked_on_a_gates_prereg_or_a_refused_block_does_not_run_anything():
    r = S._rescore_locked(f"# P\n\n{F}gates\n{GATES}\n{F}\n".encode(), "r.json", b"{}", {})
    assert r["code"] == "SCORER_REFUSES" and "locks" in r["detail"] and r["run"] is None
    r = S._rescore_locked(locks_prereg({"scorer": "score.py", "reads": [], "outcomes": []}).encode(), "r.json",
                          b"{}", {"score.py": b""})
    assert r["code"] == "SCORER_REFUSES" and "outcomes" in r["detail"] and r["run"] is None
    r = S._rescore_locked(b"# prose only\n", "r.json", b"{}", {})
    assert r["code"] == "NOT_GATED" and r["run"] is None


def test_the_run_module_keeps_every_name_private():
    """DESIGN 3.1: the public surface is core.rescore, .compare, .licences and the CLI."""
    tree = ast.parse((ROOT / "styxx" / "prereg_ci" / "scorer_run.py").read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    assert names and all(n.startswith("_") for n in names), sorted(n for n in names if not n.startswith("_"))


# ---- network (DESIGN 3.2(f): unshare -n where permitted, else "network: not isolated") -------------------------

def test_off_linux_the_network_is_not_isolated(monkeypatch):
    monkeypatch.setattr(S.sys, "platform", "win32")
    assert S._network() == ([], "not isolated")


def test_on_linux_unshare_is_used_when_it_runs_and_named_when_it_does_not(monkeypatch):
    monkeypatch.setattr(S.sys, "platform", "linux")
    monkeypatch.setattr(S.shutil, "which", lambda name: None)
    assert S._network() == ([], "not isolated (no unshare on PATH)")
    monkeypatch.setattr(S.shutil, "which", lambda name: "/usr/bin/unshare")
    calls = []

    def fake(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, fake.rc, b"", b"")

    fake.rc = 0
    monkeypatch.setattr(S.subprocess, "run", fake)
    assert S._network() == (["/usr/bin/unshare", "-rn"], "removed (unshare -rn)")
    assert calls == [["/usr/bin/unshare", "-rn", "true"]]
    fake.rc = 1
    assert S._network() == ([], "not isolated (unshare -rn refused here)")


# ---- the git layer reads a locks decision as it reads a gates one (C0, the seal) ------------------------------

class Git:
    def __init__(self, root: Path):
        self.root, self.t = root, 1_760_000_000
        root.mkdir(parents=True, exist_ok=True)
        self("init", "-q", "-b", "main")
        for k, v in (("user.email", "t@t"), ("user.name", "t"), ("core.autocrlf", "false"),
                     ("commit.gpgsign", "false"), ("core.hooksPath", str(root / ".no-hooks"))):
            self("config", k, v)

    def __call__(self, *args: str) -> str:
        env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
        self.t += 60
        env.update(GIT_AUTHOR_DATE=f"{self.t} +0000", GIT_COMMITTER_DATE=f"{self.t} +0000")
        r = subprocess.run(["git", *args], cwd=str(self.root), env=env, capture_output=True)
        assert r.returncode == 0, r.stderr
        return r.stdout.decode("utf-8").strip()

    def commit(self, files, msg):
        for rel, data in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
        self("add", "-A")
        self("commit", "-q", "-m", msg)
        return self("rev-parse", "HEAD")


def test_c0_and_the_seal_hold_a_locks_fence(tmp_path):
    g = Git(tmp_path / "r")
    s1 = {"scorer": "p/score.py", "reads": [], "outcomes": ["PASS", "HOLD"]}
    base = g.commit({"p/PREREG_a.md": locks_prereg(s1), "p/score.py": "print(1)\n"}, "seal")
    with G._Repo(g.root) as repo:
        seal = G._seal(repo, "p/PREREG_a.md", base)
        assert seal["form"] == "locks" and seal["gates_sha256"] == core._decision(locks_prereg(s1))["gates_sha256"]
        assert seal["landing"] == base and "locks" in G._label(seal)
    g("switch", "-q", "-c", "prose")
    g.commit({"p/PREREG_a.md": locks_prereg(s1, prose="notes, reworded")}, "prose")
    g("switch", "-q", "-c", "bar", "main")
    g.commit({"p/PREREG_a.md": locks_prereg({**s1, "outcomes": ["PASS", "HOLD", "SHIP"]})}, "move")
    with G._Repo(g.root) as repo:
        assert [f["code"] for f in G._decisions_moved(repo, "main", "prose")] == ["PROSE_AMENDED_IN_PLACE"]
        assert [f["code"] for f in G._decisions_moved(repo, "main", "bar")] == ["DECISION_EDITED_AFTER_SEAL"]


def test_the_seal_verb_names_a_locks_decision(tmp_path, monkeypatch, capsys):
    from styxx.prereg_ci import cli
    g = Git(tmp_path / "r")
    s1 = {"scorer": "p/score.py", "reads": [], "outcomes": ["PASS"]}
    g.commit({"p/PREREG_a.md": locks_prereg(s1), "p/score.py": "print(1)\n"}, "seal")
    monkeypatch.chdir(g.root)
    assert cli.main(["seal", "p/PREREG_a.md"]) == 0
    out = capsys.readouterr().out
    assert f"decision: locks {core._decision(locks_prereg(s1))['gates_sha256']}" in out, out


def test_the_locked_files_are_read_at_a_named_commit(tmp_path):
    g = Git(tmp_path / "r")
    s1 = {"scorer": "p/score.py", "reads": ["p/bar.txt"], "outcomes": ["PASS", "HOLD"]}
    c1 = g.commit({"p/PREREG_a.md": locks_prereg(s1), "p/score.py": "A\n", "p/bar.txt": "0.5\n"}, "seal")
    g.commit({"p/score.py": "B\n"}, "later")
    with G._Repo(g.root) as repo:
        sp = core._locks(core._decision(locks_prereg(s1))["block"])
        assert S._locked_files(repo, c1, sp) == {"p/score.py": b"A\n", "p/bar.txt": b"0.5\n"}
        assert S._locked_files(repo, "HEAD", sp)["p/score.py"] == b"B\n"
        with pytest.raises(G._Refusal) as e:
            S._locked_files(repo, c1, {**sp, "locked": ["p/score.py", "p/absent.txt"]})
        assert e.value.code == "LOCKED_FILE_ABSENT" and "p/absent.txt" in e.value.detail
        assert e.value.kind == "FAIL"


# ---- the checksum scorer as a locks decision (the reconstruction) -------------------------------------------

def test_the_reconstruction_reads_as_one_locks_decision():
    text = LOCKS_FILE.read_bytes()
    assert b"\r" not in text and not text.startswith(b"\xef\xbb\xbf")
    d = core._decision(text)
    assert d["form"] == "locks", d
    s = core._locks(d["block"])
    assert s["scorer"] == SCORE and s["reads"] == [] and s["withdraws"] == {}
    assert s["argv"] == ["{receipt}", "--prereg", "deploy_quant", "--out", "{out}"]
    assert s["verdict_from"] == "/run_reading"
    assert len(s["outcomes"]) == 1730 and DRY_READING in s["outcomes"] and s["outcomes"] == sorted(s["outcomes"])
    words = text.decode("utf-8")
    assert "RECONSTRUCTION" in words and "not a catch" in words and "never sealed" in words
    assert "instrument check, not a result" in words
    assert core._result_reading(text)["declarations"] == [] and core._result_reading(text)["code"] is None
    assert not HOUSE_RX.search(words) and not PRIORITY_RX.search(words)


def test_the_reconstruction_s_outcomes_are_the_deploy_quant_reading_grammar():
    """Restated here from score.py's deploy-quant path, apart from the generator: the reading is
    "INCONCLUSIVE (K1)" when K1 fired, else the gates that fired among provenance, K2, K3, K4 (in
    that order) and the four hypotheses' statuses; "INSTRUMENT CHECK; " leads exactly when the
    provenance gate fired (K1 aside, where it leads when the certs are not the experiment)."""
    s = core._locks(core._decision(LOCKS_FILE.read_bytes())["block"])
    status = r"(HELD|FAILED|NOT_EVALUABLE)"
    h1 = r"(HELD|FAILED|NOT_EVALUABLE|HELD on the floor, INCONCLUSIVE on the pair by construction)"
    rx = re.compile(rf"(INSTRUMENT CHECK; )?(?:INCONCLUSIVE \(K1\)|(?:gates fired: ((?:provenance|K2|K3|K4)"
                    rf"(?:, (?:K2|K3|K4))*)|no gate fired); H1 {h1}, H2 {status}, H3 {status}, H4 {status})")
    for v in s["outcomes"]:
        m = rx.fullmatch(v)
        assert m, v
        if "INCONCLUSIVE (K1)" not in v:
            fired = (m.group(2) or "").split(", ") if m.group(2) else []
            assert fired == sorted(set(fired), key=["provenance", "K2", "K3", "K4"].index), v
            assert bool(m.group(1)) == ("provenance" in fired), v
    assert {v for v in s["outcomes"] if "K1" in v} == {"INCONCLUSIVE (K1)", "INSTRUMENT CHECK; INCONCLUSIVE (K1)"}


def test_the_reconstruction_regenerates_from_its_generator():
    r = subprocess.run([sys.executable, "-B", str(GEN), "--check"], cwd=str(ROOT), capture_output=True,
                       timeout=300)
    out = r.stdout.decode("utf-8", "replace")
    assert r.returncode == 0 and "SAME" in out, (out, r.stderr.decode("utf-8", "replace"))


def _have(*commits: str) -> bool:
    return all(subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", c + "^{commit}"],
                              capture_output=True).returncode == 0 for c in commits)


@pytest.fixture(scope="module")
def repo():
    if not _have(REV, *{c for c, _ in DEPLOY_CARDS + BEACON_CARDS}):
        pytest.skip("the checksum commits are not in the object store (a shallow or partial clone)")
    with G._Repo(ROOT) as r:
        if r.is_shallow():
            pytest.skip("a shallow clone holds no history to read the checksum scorer from")
        yield r


@pytest.fixture(scope="module")
def locks():
    text = LOCKS_FILE.read_bytes()
    return text, core._locks(core._decision(text)["block"])


def test_the_card_list_is_every_commit_that_wrote_a_dry_run_card(repo):
    for cards, paths in ((DEPLOY_CARDS, (CARD_V1, CARD_V2)), (BEACON_CARDS, (BEACON_CARD_V1, BEACON_CARD_V2))):
        log = subprocess.run(["git", "--no-replace-objects", "-C", str(ROOT), "log", "--format=%H", "HEAD", "--",
                              *paths], capture_output=True, check=True).stdout.decode().split()
        assert sorted(log) == sorted(c for c, _ in cards)


@pytest.mark.parametrize("commit, card", DEPLOY_CARDS)
def test_the_checksum_scorer_re_derives_every_committed_deploy_quant_card_from_sealed_blobs(repo, locks, commit, card):
    text, s = locks
    files = S._locked_files(repo, commit, s)
    certs = repo.read(commit, CERTS)
    r = S._rescore_locked(text, CERTS, certs, files)
    committed = repo.read(commit, card)
    assert (r["code"], r["verdict"]) == (None, DRY_READING), r
    assert r["run"]["out_bytes"] == committed                            # the card, byte for byte
    want = json.loads(committed)
    for key in ("run_reading", "hypotheses", "gates", "notes", "counts_as_result", "is_the_experiment",
                "certs_file", "certs_sha256"):
        assert r["output"][key] == want[key], key
    assert want["counts_as_result"] is False and "INSTRUMENT CHECK" in want["run_reading"]
    assert r["run"]["locked"] == [{"path": SCORE, "sha256": hashlib.sha256(files[SCORE]).hexdigest()}]
    assert r["run"]["unlocked"] == [] and r["run"]["exit"] == 0
    assert [f["code"] for f in core.compare(r, {"verdict": want["run_reading"]})] == ["RECORDED_DIGEST_ABSENT"]


def test_the_committed_cards_still_read_at_head(repo):
    for commit, card in DEPLOY_CARDS[-1:]:
        assert repo.blob_id("HEAD", card) == repo.blob_id(commit, card)
        assert repo.blob_id("HEAD", SCORE) == repo.blob_id(SEALED_SCORER, SCORE)


def test_the_certs_came_before_any_scorer_so_a_seal_could_not_order_them(repo):
    """Why the reconstruction is not a catch: the certs were written before score.py's oldest blob,
    so a locks PREREG holding the scorer could only have been sealed after the receipt it scores."""
    certs_at = repo.commit(CERTS_COMMIT)
    oldest = DEPLOY_CARDS[0][0]
    assert repo.read(certs_at, SCORE) is None and repo.read(oldest, SCORE) is not None
    assert repo.is_strict_ancestor(certs_at, oldest)
    assert repo.blob_id(certs_at, CERTS) == repo.blob_id(SEALED_SCORER, CERTS)


def test_the_scorer_s_blob_is_what_judges(repo, locks):
    """The v1 scorer over the same certs prints the same reading and another card; a card recorded
    with another reading is VERDICT_DIFFERS."""
    text, s = locks
    v1, v2 = DEPLOY_CARDS[0][0], DEPLOY_CARDS[-1][0]
    r1 = S._rescore_locked(text, CERTS, repo.read(v2, CERTS), S._locked_files(repo, v1, s))
    assert r1["verdict"] == DRY_READING and r1["run"]["out_bytes"] != repo.read(v2, CARD_V2)
    assert r1["output"]["schema"] == "styxx.checksum/scorecard/v1"
    r2 = S._rescore_locked(text, CERTS, repo.read(v2, CERTS), S._locked_files(repo, v2, s))
    found = core.compare(r2, {"verdict": DRY_READING.replace("H1 FAILED", "H1 HELD")})
    assert found[-1]["code"] == "VERDICT_DIFFERS" and DRY_READING in found[-1]["detail"]


def test_a_moved_number_in_the_certs_moves_the_reading(repo, locks):
    text, s = locks
    certs = json.loads(repo.read(SEALED_SCORER, CERTS))
    certs["Q4"]["distance"]["mean_abs_nats"] = 0.2                       # inside H2's [0.01, 0.30]
    data = (json.dumps(certs, indent=1) + "\n").encode("utf-8")
    r = S._rescore_locked(text, CERTS, data, S._locked_files(repo, SEALED_SCORER, s))
    assert r["code"] is None and r["verdict"] == DRY_READING.replace("H2 FAILED", "H2 HELD"), r
    assert core.compare(r, {"verdict": DRY_READING})[-1]["code"] == "VERDICT_DIFFERS"


def test_a_pull_request_that_edits_the_locked_scorer_does_not_move_the_judge(repo, locks, tmp_path):
    """DESIGN 3.2(f): the scorer runs from sealed blobs on the base side. SYNTHETIC ORDER: the
    repository built here seals the reconstruction with score.py's f0ec5605 blob on main before a
    pull request adds the certs and edits the scorer, which the lab's history does not do (the certs
    came before any scorer); it shows which blobs the run reads. C0 reads the fence, so the edit of a
    locked file is not a moved decision (named in B4's report for B7)."""
    text, s = locks
    sealed_src = repo.read(SEALED_SCORER, SCORE)
    assert sealed_src.count(b'"INSTRUMENT CHECK; "') == 1
    g = Git(tmp_path / "r")
    base = g.commit({"LOCKS.md": text, SCORE: sealed_src}, "seal")
    g("switch", "-q", "-c", "pr")
    head = g.commit({SCORE: sealed_src.replace(b'"INSTRUMENT CHECK; "', b'"EDITED; "'),
                     CERTS: repo.read(SEALED_SCORER, CERTS)}, "certs and an edited scorer")
    with G._Repo(g.root) as r:
        seal = G._seal(r, "LOCKS.md", "main")
        assert seal["landing"] == base and seal["form"] == "locks"
        sp = core._locks(core._decision(seal["bytes"])["block"])
        at_seal = S._rescore_locked(seal["bytes"], CERTS, r.read(head, CERTS), S._locked_files(r, seal["landing"], sp))
        at_head = S._rescore_locked(seal["bytes"], CERTS, r.read(head, CERTS), S._locked_files(r, head, sp))
        moved = G._decisions_moved(r, "main", "pr")
    assert (at_seal["code"], at_seal["verdict"]) == (None, DRY_READING)
    assert at_seal["run"]["out_bytes"] == repo.read(SEALED_SCORER, CARD_V2)
    assert at_head["code"] == "SCORER_ERROR" and "EDITED; " in at_head["detail"], at_head
    assert moved == []


@pytest.mark.parametrize("commit, card", BEACON_CARDS)
def test_the_beacon_draw_path_needs_site_packages_and_reads_unlocked(repo, commit, card):
    block_ = {"scorer": SCORE, "reads": ["styxx/beacon.py", "styxx/checksum.py"],
              "argv": ["{receipt}", "--prereg", "beacon_draw", "--out", "{out}"], "verdict_from": "/run_reading",
              "outcomes": sorted({BEACON_READING, json.loads(repo.read(commit, card))["run_reading"]})}
    text = locks_prereg(block_).encode("utf-8")
    s = core._locks(core._decision(text)["block"])
    r = S._rescore_locked(text, BEACON_CERTS, repo.read(commit, BEACON_CERTS), S._locked_files(repo, commit, s))
    assert r["code"] == "SCORER_READS_UNLOCKED" and "imports numpy" in r["detail"], r


def test_without_the_import_watch_the_beacon_card_would_differ_by_one_swallowed_line(repo, tmp_path):
    """The reason the run watches imports: under -I -S the beacon-draw scorer catches the failed numpy
    import, adds one line to K5's detail and prints the committed reading, so a reading compared alone
    would agree with a card the run did not re-derive."""
    commit, card = BEACON_CARDS[-1]
    for rel in (SCORE, "styxx/beacon.py", "styxx/checksum.py", BEACON_CERTS):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(repo.read(commit, rel))
    subprocess.run([sys.executable, "-I", "-S", "-B", "-X", "utf8", SCORE, BEACON_CERTS, "--prereg", "beacon_draw",
                    "--out", "out.json"], cwd=str(tmp_path), capture_output=True, check=True, timeout=120)
    got, want = json.loads((tmp_path / "out.json").read_bytes()), json.loads(repo.read(commit, card))
    assert got["run_reading"] == want["run_reading"] == BEACON_READING
    extra = [x for x in got["gates"]["K5"]["detail"] if x not in want["gates"]["K5"]["detail"]]
    assert extra == ["the draw record could not be re-derived: No module named 'numpy'"]
    assert {k: v for k, v in got.items() if k != "gates"} == {k: v for k, v in want.items() if k != "gates"}


def _load_scorer(repo, tmp_path):
    """score.py at its sealed blob as a module of this process; it puts its own ROOT on sys.path when
    it loads, so sys.path is put back."""
    p = tmp_path / "papers" / "checksum" / "score_at_seal.py"
    p.parent.mkdir(parents=True)
    p.write_bytes(repo.read(SEALED_SCORER, SCORE))
    spec_ = importlib.util.spec_from_file_location("score_at_seal", str(p))
    mod = importlib.util.module_from_spec(spec_)
    saved = list(sys.path)
    try:
        spec_.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def test_every_reading_the_deploy_quant_path_prints_is_a_declared_outcome(repo, locks, tmp_path):
    """The outcomes are the scorer's reading grammar, enumerated by the generator from the code; 4000
    certs varied at seed 20261025 over every field the deploy-quant path reads each print one of them."""
    _, s = locks
    mod = _load_scorer(repo, tmp_path)
    base = json.loads(repo.read(SEALED_SCORER, CERTS))
    rng = random.Random(20261025)
    seen = set()

    def pick(*xs):
        return rng.choice(xs)

    for _ in range(4000):
        c = json.loads(json.dumps(base))
        for key in ("sanity", "provenance", "canaries"):
            c.setdefault(key, {})
        c["sanity"]["null_floor_nats"] = pick(None, 0.0, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, -1.0, float("nan"), "x")
        if rng.random() < 0.5:
            c.pop("k1", None)
        c["model"] = pick("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B")
        c["is_the_experiment"] = pick(True, False)
        c["tag"] = pick(None, "", "_dryrun")
        c["smoke"] = pick(False, True)
        c["provenance"]["git_head"] = pick("abc", "", None)
        c["provenance"]["prereg_blob_sha256"] = pick(mod.BANDS["deploy_quant"]["sealed_blob"], "00")
        c["canaries"]["n"] = pick(48, None)
        c["canaries"]["canary_sha256"] = pick(mod.HAND_CANARY_SHA256, None)
        for arm in ("A2", "Q4", "Q8", "R"):
            if rng.random() < 0.1:
                c.pop(arm, None)
                continue
            d = c.setdefault(arm, {}).setdefault("distance", {})
            d["verdict"] = pick("SAME", "DRIFT", "INCONCLUSIVE", None)
            d["mean_abs_nats"] = pick(0.0, 0.005, 0.05, 0.2, 0.5, 2.0, 6.0, None)
            d["rdm_r"] = pick(0.1, 0.5, 0.96, 0.99, None)
        c["sanity"]["top1_loss_vs_A"] = {"Q4": pick(0, 4, 5, 13, None), "Q8": pick(0, 9, None)}
        c["sanity"][HITS_KEY] = {"R": pick(0, 2, 3, None)}
        card = mod.score(c, "deploy_quant")
        assert card["run_reading"] in s["outcomes"], card["run_reading"]
        seen.add(card["run_reading"])
    print(f"distinct readings over 4000 varied certs: {len(seen)} of {len(s['outcomes'])}")
    assert len(seen) >= 100, len(seen)


# ---- the locks form moves no committed reading ---------------------------------------------------------------

def _protocol_reading(data: bytes):
    try:
        text = core._decision_text(data)
    except UnicodeDecodeError:
        return ("refuses", None)
    try:
        block = protocol._select_gates_block(text)
    except protocol.GateSpecError as e:
        return ("none", None) if str(e).startswith(core._NO_FENCE) else ("refuses", str(e))
    return ("gates", hashlib.sha256(block.encode("utf-8")).hexdigest())


def _reading(data: bytes):
    d = core._decision(data)
    if d["form"] is not None:
        return (d["form"], d["gates_sha256"])
    return ("none", None) if d["code"] == "NOT_GATED" else ("refuses", d["detail"] if d["detail"] and
                                                            not d["detail"].startswith("the PREREG is not UTF-8")
                                                            else None)


@pytest.mark.parametrize("rev", [REV, "HEAD"])
def test_every_markdown_file_reads_as_the_gates_scanner_read_it_before_the_locks_form(repo, rev):
    """Every .md blob in the tree: the decision reads as styxx.protocol's scanner reads it, except a
    file that holds a locks fence, and every such file is the reconstruction this step writes."""
    tally, locked = {}, []
    for path in repo.paths(rev):
        if not path.lower().endswith(".md"):
            continue
        try:
            data = repo.read(rev, path)
        except G._Refusal:
            continue
        mine, theirs = _reading(data), _protocol_reading(data)
        if mine[0] == "locks":
            assert theirs[0] == "none", path
            locked.append(path)
            continue
        assert mine == theirs, (path, mine, theirs)
        tally[mine[0]] = tally.get(mine[0], 0) + 1
    print(f"{rev[:12]}: {tally}, locks {locked}")
    assert all(p.startswith("conformance/prereg_ci/locks/") for p in locked), locked
    if rev == REV:
        assert locked == [] and tally.get("gates", 0) >= 52
