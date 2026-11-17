# -*- coding: utf-8 -*-
"""A blocking check fails on an unread or unmeasured diff under a claim, and on a withheld
contradiction; report mode exits 0 as it did.

With soft-fail "false" and strict off, `diffgate_action.py` returned 0 on a diff it could not fetch
(GitHub refuses a diff past its size limit), on a diff the gate did not measure, and on a claim whose
CONTRADICTED reading the PATH-2a overlay withheld, so the author of a pull request could turn a red
blocking check green while a claim the diff contradicts stayed in the description. A blocking check
now fails in exactly those three cases:

  (a) the diff could not be fetched and the description holds a claim the gate reads;
  (b) the gate did not measure the diff and the description holds a claim the gate reads;
  (c) a claim is UNCHECKABLE because the overlay withheld a CONTRADICTED reading of it.

Every other UNCHECKABLE claim passes unless strict is on, as before; a withheld VERIFIED reading
passes; strict fails what it failed; a push that creates a tag is not gated and asks nothing of the
API. Report mode (soft-fail "true", the default, or a value the script does not recognise) exits 0
on every path, and its annotations are the ones it printed before, but on a push that creates a tag, which is
no longer gated.

The table below is every path through `main()` by soft-fail and strict, run in the test's own
process with `api` replaced. The process tests at the end run the script as `action.yml` does,
`python <path>/diffgate_action.py`, against a server on 127.0.0.1, and read the real exit code;
the child names the styxx it imported, and a test that cannot show it is the tree's own skips.
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ACTION = ROOT / "diffgate_action.py"

FALSE = "Only touches files under src/."
PROSE = "Refactors the retry loop for clarity."
ACCENTED = "Only touches files under src/ été."


def _file(path, added):
    return (f"diff --git a/{path} b/{path}\nindex 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -1 +1,2 @@\n x = 0\n+{added}\n")


DIFF = _file("config/evil.yml", "b: 2") + _file("src/a.py", "y = 1")      # one file outside src/
DIFF_SRC = _file("src/a.py", "y = 1")                                     # every file under src/
DIFF_SEP = _file("config/evil.yml", "b: 2") + _file("src/a.py", "y = 1  #\u2028--- note")
# A path holding a separator character: the overlay withholds a CONTRADICTED file count.
DIFF_SEP_PATH = _file("config/evil.yml", "b: 2") + _file("src/a\u2028b.py", "y = 1")

# The workflow commands this change adds; each blocking exit 1 below prints one of them or the
# contradicted-claim and strict ones it had.
E_FETCH = "::error title=styxx diffgate - diff not read, so the check fails::"
E_UNMEASURED = "::error title=styxx diffgate - UNMEASURED, so the check fails::"
E_WITHHELD = "::error title=styxx diffgate - withheld, so the check fails::"
E_CONTRA = "::error title=styxx diffgate - contradicted claim::"
E_STRICT = "::error title=styxx diffgate - uncheckable under strict::"
NEW_ERRORS = (E_FETCH, E_UNMEASURED, E_WITHHELD)


class Refused(RuntimeError):
    """What `api` raises in place of an HTTP error: the fetch failed."""


# Each path: the event, the description, what `api` does, and the expected exit code in blocking
# mode with strict off and on (report mode exits 0 on every path). `errors` is the count of
# `::error` lines in blocking mode, strict off and on; `words` must stand in stdout in blocking mode,
# and `words_off` there with strict off.
PATHS = {
    "E1 event pull_request_target": dict(event="pull_request_target", body=FALSE, block=(0, 0), errors=(0, 0)),
    "E2 empty description": dict(body="   ", block=(0, 0), errors=(0, 0)),
    "F1 fetch refused, a claim": dict(body=FALSE, api=Refused("HTTP Error 406: Not Acceptable"),
                                      block=(1, 1), errors=(1, 1), words=(E_FETCH, "judged none of the 1 claim(s)")),
    "F2 fetch refused, no claim": dict(body=PROSE, api=Refused("HTTP Error 406: Not Acceptable"),
                                       block=(0, 1), errors=(0, 1), words_off=("there was no claim to judge",)),
    "F3 push creating a branch, a claim": dict(event="push", body=FALSE, created="branch",
                                               api=Refused("HTTP Error 404: Not Found"), block=(1, 1), errors=(1, 1),
                                               words=(E_FETCH, "has no base commit to compare with")),
    "F4 push creating a branch, no claim": dict(event="push", body=PROSE, created="branch",
                                                api=Refused("HTTP Error 404: Not Found"), block=(0, 1), errors=(0, 1),
                                                words=("has no base commit to compare with",)),
    "U1 unmeasured, a claim": dict(body=FALSE, api="", block=(1, 1), errors=(1, 1),
                                   words=(E_UNMEASURED, "judged none of the 1 claim(s)")),
    "U2 unmeasured, no claim": dict(body=PROSE, api="", block=(0, 1), errors=(0, 1),
                                    words_off=("there was no claim to judge",)),
    "C0 measured, no claim": dict(body=PROSE, block=(0, 0), errors=(0, 0)),
    "C1 every claim VERIFIED": dict(body=FALSE, api=DIFF_SRC, block=(0, 0), errors=(0, 0)),
    "C2 one CONTRADICTED": dict(body=FALSE, block=(1, 1), errors=(1, 1), words=(E_CONTRA,)),
    "C3 one UNCHECKABLE": dict(body="All tests pass.", api=DIFF_SRC, block=(0, 1), errors=(0, 1)),
    "C4 withheld CONTRADICTED, an accented word": dict(body=ACCENTED, block=(1, 1), errors=(1, 1),
                                                       words=(E_WITHHELD, "although the gate's verdict is PASS")),
    "C5 withheld VERIFIED": dict(body=ACCENTED, api=DIFF_SRC, block=(0, 1), errors=(0, 1)),
    "C6 withheld CONTRADICTED, a separator in content": dict(body=FALSE, api=DIFF_SEP, block=(1, 1), errors=(1, 1),
                                                             words=(E_WITHHELD,)),
    "C7 withheld CONTRADICTED and a VERIFIED": dict(body=ACCENTED + " Modifies config/evil.yml.", block=(1, 1),
                                                    errors=(1, 1), words=(E_WITHHELD,)),
    "C8 withheld CONTRADICTED, a file count": dict(body="5 files changed.", api=DIFF_SEP_PATH, block=(1, 1),
                                                   errors=(1, 1), words=(E_WITHHELD,)),
    "T1 push creating a tag, before is zeros": dict(event="push", body=FALSE, created="tag", api=Refused("asked"),
                                                    block=(0, 0), errors=(0, 0), words=("creates a tag is not gated",)),
    "T2 push creating a tag, created is true": dict(event="push", body=FALSE, created="tag-flag", api=Refused("asked"),
                                                    block=(0, 0), errors=(0, 0), words=("creates a tag is not gated",)),
    "T3 push moving a tag": dict(event="push", body=FALSE, created="tag-moved", block=(1, 1), errors=(1, 1),
                                 words=(E_CONTRA,)),
    "T4 push creating a tag, before only": dict(event="push", body=FALSE, created="tag-zeros", api=Refused("asked"),
                                                block=(0, 0), errors=(0, 0), words=("creates a tag is not gated",)),
}

# The pushes that create a tag, which are not gated; a push that moves a tag is.
TAG_CREATES = ("tag", "tag-flag", "tag-zeros")

SOFT = [("true", False), ("maybe", False), (None, False), ("false", True)]      # (STYXX_SOFT_FAIL, blocks)


def load(name="diffgate_action_fails_closed", path=ACTION):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def payload(event, body, created=None):
    if event == "push":
        p = {"head_commit": {"message": body}, "before": "1" * 40, "after": "a" * 40, "ref": "refs/heads/main",
             "repository": {"compare_url": "https://api.github.test/repos/o/r/compare/{base}...{head}"}}
        if created == "branch":
            p.update(before="0" * 40, created=True, ref="refs/heads/new")
        elif created == "tag":
            p.update(before="0" * 40, created=True, ref="refs/tags/v1")
        elif created == "tag-flag":
            p.update(created=True, ref="refs/tags/v1")
        elif created == "tag-zeros":
            p.update(before="0" * 40, ref="refs/tags/v1")             # no `created` key
        elif created == "tag-moved":
            p.update(created=False, ref="refs/tags/v1")
        return p
    return {"pull_request": {"number": 7, "body": body, "url": "https://api.github.test/repos/o/r/pulls/7"}}


def run_main(mod, tmp_path, monkeypatch, capsys, case, soft, strict):
    """main() on one path; returns (exit code, stdout, summary, the addresses `api` was asked)."""
    ev, summary = tmp_path / "event.json", tmp_path / "summary.md"
    ev.write_text(json.dumps(payload(case.get("event", "pull_request"), case["body"], case.get("created"))),
                  encoding="utf-8")
    if summary.exists():
        summary.unlink()
    monkeypatch.setenv("GITHUB_EVENT_NAME", case.get("event", "pull_request"))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(ev))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GH_TOKEN", "made-up")
    monkeypatch.setenv("STYXX_STRICT", "true" if strict else "false")
    if soft is None:
        monkeypatch.delenv("STYXX_SOFT_FAIL", raising=False)
    else:
        monkeypatch.setenv("STYXX_SOFT_FAIL", soft)
    asked = []
    answer = case.get("api", DIFF)

    def fake_api(url, accept):
        asked.append(url)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(mod, "api", fake_api)
    rc = mod.main()
    out = capsys.readouterr().out
    return rc, out, summary.read_bytes().decode("utf-8") if summary.exists() else "", asked


@pytest.fixture(scope="module")
def act():
    return load()


@pytest.mark.parametrize("strict", [False, True], ids=["strict-off", "strict-on"])
@pytest.mark.parametrize("soft, blocks", SOFT, ids=["soft-true", "soft-unrecognised", "soft-unset", "soft-false"])
@pytest.mark.parametrize("name", list(PATHS))
def test_every_path_by_soft_fail_and_strict(act, tmp_path, monkeypatch, capsys, name, soft, blocks, strict):
    case = PATHS[name]
    rc, out, summary, asked = run_main(act, tmp_path, monkeypatch, capsys, case, soft, strict)
    errors = [ln for ln in out.splitlines() if ln.startswith("::error")]
    if not blocks:
        assert rc == 0
        assert not any(ln.startswith(NEW_ERRORS) for ln in errors), errors
        # report mode prints the errors it printed before: a contradicted claim, and under strict an uncheckable one
        assert all(ln.startswith((E_CONTRA, E_STRICT)) for ln in errors), errors
    else:
        assert rc == case["block"][strict], out
        assert len(errors) == case["errors"][strict], out
        for w in case.get("words", ()) + (() if strict else case.get("words_off", ())):
            if w == "although the gate's verdict is PASS" and strict:
                continue                                    # under strict the gate's own verdict is FAIL
            assert w in out, (w, out)
        if rc == 1:
            assert errors, "a blocking check that fails says why in an ::error"
    if case.get("created", "") in TAG_CREATES:
        assert asked == [], "a push that creates a tag asks nothing of the API"
        assert summary == ""
    # a withheld claim is never called contradicted by the script's own words: its row stays UNCHECKABLE
    if "withheld" in name:
        assert "❌ CONTRADICTED" not in summary
        assert "| ❓ UNCHECKABLE |" in summary


def test_the_three_routes_named_in_the_change(act, tmp_path, monkeypatch, capsys):
    """The routes a reviewer found, each in blocking mode with strict off: a diff the API refuses, an
    accented word in the claim's sentence, and a separator character in one line of content (the diff
    DIFF_SEP builds). Each was 0."""
    for case in (PATHS["F1 fetch refused, a claim"], PATHS["C4 withheld CONTRADICTED, an accented word"],
                 PATHS["C6 withheld CONTRADICTED, a separator in content"]):
        assert run_main(act, tmp_path, monkeypatch, capsys, case, "false", False)[0] == 1
        assert run_main(act, tmp_path, monkeypatch, capsys, case, "true", False)[0] == 0


def test_a_withheld_contradiction_says_both_verdicts(act, tmp_path, monkeypatch, capsys):
    rc, out, summary, _ = run_main(act, tmp_path, monkeypatch, capsys, PATHS["C4 withheld CONTRADICTED, an accented word"],
                                   "false", False)
    assert rc == 1
    last = out.strip().splitlines()[-1]
    assert last.startswith("styxx diffgate: PASS - 1 claim(s), 0 contradicted; 1 not judged")
    assert "the job fails although the gate's verdict is PASS" in last
    assert "**PASS** is the gate's verdict, and the job fails" in summary
    error = [ln for ln in out.splitlines() if ln.startswith(E_WITHHELD)][0]
    assert "may be wrong" in error and "plain ASCII" in error and 'soft-fail: "true"' in error
    assert "every other UNCHECKABLE claim passes unless strict is on" in error
    assert "The overlay's reason: CONTRADICTED withheld by PATH-2a" in error
    # the reason the job fails stands above the table
    assert summary.index("**The job fails on 1 claim(s) the gate did not judge") < summary.index("| verdict |")


def test_a_pipe_in_a_withheld_claim_is_shown_as_it_is_above_the_table(act, tmp_path, monkeypatch, capsys):
    """Above the table the claims are named in a paragraph, where the table's escape for "|" is not read and
    its backslash would be shown; in the claim's row the escape stands."""
    rc, out, summary, _ = run_main(act, tmp_path, monkeypatch, capsys, dict(body=ACCENTED[:-1] + " a|b."), "false",
                                   False)
    assert rc == 1
    above, table = summary.split("| verdict |", 1)
    assert "a|b.`" in above and "a\\|b" not in above
    assert "a\\|b.`" in table


def test_a_withheld_reason_counts_only_where_the_overlay_wrote_it(act):
    """A claim counts as a withheld contradiction only where the overlay wrote its reason: a kind the overlay
    may move. The same words in the reason of a kind it never moves do not count."""
    from types import SimpleNamespace
    why = "CONTRADICTED withheld by PATH-2a (#121): x. main's reading: y"
    assert act._withheld_contradiction(SimpleNamespace(verdict="UNCHECKABLE", kind="only_touches", why=why))
    assert not act._withheld_contradiction(SimpleNamespace(verdict="UNCHECKABLE", kind="tests_pass", why=why))


def test_a_failed_fetch_names_the_remedies(act, tmp_path, monkeypatch, capsys):
    rc, out, summary, _ = run_main(act, tmp_path, monkeypatch, capsys, PATHS["F1 fetch refused, a claim"], "false", False)
    assert rc == 1
    error = [ln for ln in out.splitlines() if ln.startswith(E_FETCH)][0]
    assert "A re-run reads the diff again" in error
    assert "GitHub refuses a diff past its size limit" in error
    assert "holds no claim the gate reads passes" in error
    assert out.strip().splitlines()[-1].startswith("styxx diffgate: DID NOT RUN - the diff could not be fetched")
    assert summary.startswith("## styxx diffgate — DID NOT RUN, and the check fails")


def test_extraction_that_raises_is_taken_to_hold_a_claim(act, tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("no")
    monkeypatch.setattr(act, "gate_diff_text", boom)
    rc, out, _, _ = run_main(act, tmp_path, monkeypatch, capsys, PATHS["F2 fetch refused, no claim"], "false", False)
    assert rc == 1 and E_FETCH in out and "judged none of the claims" in out


def test_no_wording_reads_the_old_wide_rule(act, tmp_path, monkeypatch, capsys):
    """No sentence says a blocking check fails on everything it could not judge: each names its case."""
    for name in PATHS:
        for soft in ("false", "true"):
            for strict in (False, True):
                _, out, summary, _ = run_main(act, tmp_path, monkeypatch, capsys, PATHS[name], soft, strict)
                text = (out + summary).lower()
                assert "does not pass what it could not judge" not in text
                assert "does not pass claims it did not judge" not in text


def _flood(n=3000):
    many = " ".join(f"Modified src/nothere_{i:05d}_{'x' * 60}.py." for i in range(n))
    return many + " " + ACCENTED


def test_a_withheld_contradiction_keeps_its_row_when_the_table_overflows(act, tmp_path, monkeypatch, capsys):
    """Thousands of claims the gate cannot check, then one whose CONTRADICTED reading the overlay withheld.
    In blocking mode it is ranked with the contradicted, so its row is written; its ::error is never under
    the cap on warnings; the reason the job fails stands above the table."""
    case = dict(body=_flood())
    rc, out, summary, _ = run_main(act, tmp_path, monkeypatch, capsys, case, "false", False)
    assert rc == 1
    assert "more claim(s) are not in this table" in summary
    assert "withheld by PATH-2a" in summary.split("| verdict |")[1]
    assert sum(1 for ln in out.splitlines() if ln.startswith(E_WITHHELD)) == 1
    assert "written with the CONTRADICTED" in summary
    assert summary.index("**The job fails on 1 claim(s)") < summary.index("| verdict |")
    rc, out, summary, _ = run_main(act, tmp_path, monkeypatch, capsys, case, "true", False)
    assert rc == 0 and E_WITHHELD not in out


# -- report mode is as it was -------------------------------------------------------------------------------------

BASE = "8627931d"           # the commit this change was cut from


def _base_script(tmp_path):
    try:
        text = subprocess.run(["git", "show", f"{BASE}:diffgate_action.py"], cwd=ROOT, capture_output=True,
                              timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip(f"the script at {BASE} is not in this checkout (a shallow clone), so there is nothing to compare")
    path = tmp_path / "base" / "diffgate_action.py"
    path.parent.mkdir()
    path.write_bytes(text)
    return path


# Lines of the summary that state the blocking rule, the one thing report mode may now say otherwise.
_RULE_LINE = re.compile(r"^_(?:A workflow that sets `soft-fail|Set `strict: true` and `soft-fail)")
_FOOTER_TAIL = "A path claim the diff does not show"


def _without_rule(summary):
    kept = []
    for ln in summary.split("\n"):
        if _RULE_LINE.match(ln):
            continue
        if _FOOTER_TAIL in ln:
            ln = ln[ln.index(_FOOTER_TAIL):]
        kept.append(ln)
    return "\n".join(kept).rstrip("\n")


@pytest.mark.parametrize("soft", ["true", "maybe", None])
@pytest.mark.parametrize("strict", [False, True])
def test_report_mode_prints_what_it_printed(act, tmp_path, monkeypatch, capsys, soft, strict):
    """Report mode: the same exit code and the same stdout, byte for byte, as the script this change was
    cut from, and the same summary but for the sentences that state the blocking rule. Tag pushes are left
    out (they are no longer gated in any mode)."""
    old = load("diffgate_action_base", _base_script(tmp_path))
    for name, case in PATHS.items():
        if case.get("created", "") in TAG_CREATES:
            continue
        new_run = run_main(act, tmp_path, monkeypatch, capsys, case, soft, strict)
        old_run = run_main(old, tmp_path, monkeypatch, capsys, case, soft, strict)
        assert new_run[0] == old_run[0] == 0, name
        assert new_run[1] == old_run[1], name
        assert _without_rule(new_run[2]) == _without_rule(old_run[2]), name


# -- the script as a process --------------------------------------------------------------------------------------

# Run in a child as `action.yml` runs it: `python <path>/diffgate_action.py` puts the script's directory first
# on sys.path, so it imports the styxx beside it. The wrapper does the same and runs the file as __main__, so
# `sys.exit(main())` sets the exit code. A developer install of styxx (an "editable" import finder) is no part
# of a runner, so the wrapper drops it, so that it cannot be the one imported; at exit the child names the
# styxx it imported, and the test skips where that is not the tree's.
_WRAPPER = """\
import atexit, os, runpy, sys
script = sys.argv[1]
sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in type(f).__module__.lower() and "editable" not in repr(f).lower()]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
sys.argv = [script]
def _which():
    m = sys.modules.get("styxx")
    sys.stderr.write("STYXX_FILE=" + str(getattr(m, "__file__", None)) + "\\n")
atexit.register(_which)
runpy.run_path(script, run_name="__main__")
"""


class _Api(http.server.BaseHTTPRequestHandler):
    status, body = 200, ""

    def do_GET(self):
        cls = type(self)
        data = (cls.body if cls.status == 200 else '{"message": "refused"}').encode("utf-8")
        self.send_response(cls.status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def api_server():
    handler = type("Api", (_Api,), {})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield handler, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _process(tmp_path, api_server, body, *, status=200, diff=DIFF, soft="false"):
    handler, base = api_server
    handler.status, handler.body = status, diff
    ev = tmp_path / "event.json"
    ev.write_text(json.dumps({"pull_request": {"number": 7, "body": body, "url": base + "/repos/o/r/pulls/7"}}),
                  encoding="utf-8")
    env = dict(os.environ, GITHUB_EVENT_NAME="pull_request", GITHUB_EVENT_PATH=str(ev),
               GITHUB_STEP_SUMMARY=str(tmp_path / "summary.md"), GITHUB_API_URL=base, GH_TOKEN="made-up-token",
               STYXX_STRICT="false", STYXX_SOFT_FAIL=soft, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", _WRAPPER, str(ACTION)], cwd=str(tmp_path), env=env,
                       capture_output=True, timeout=300)
    err = r.stderr.decode("utf-8", "replace")
    which = re.findall(r"STYXX_FILE=(.*)", err)
    if not which or which[-1].strip() in ("None", ""):
        pytest.skip(f"the child did not name the styxx it imported: {err[-300:]!r}")
    imported = Path(which[-1].strip()).resolve()
    if ROOT.resolve() not in imported.parents:
        pytest.skip(f"the child imported a styxx outside the tree ({imported}), so its exit code is not this tree's")
    return r.returncode, r.stdout.decode("utf-8", "replace")


@pytest.mark.parametrize("body, status, diff, soft, code", [
    (FALSE, 200, DIFF, "false", 1),                 # contradicted, blocking
    (ACCENTED, 200, DIFF, "false", 1),              # withheld contradiction, blocking
    (ACCENTED, 200, DIFF, "true", 0),               # withheld contradiction, report
    (FALSE, 406, DIFF, "false", 1),                 # a diff GitHub refuses, blocking
    (FALSE, 406, DIFF, "true", 0),                  # a diff GitHub refuses, report
    (FALSE, 200, DIFF_SRC, "false", 0),             # clean
], ids=["contradicted-blocking", "withheld-blocking", "withheld-report", "refused-blocking", "refused-report",
        "clean-blocking"])
def test_the_exit_code_of_the_process(tmp_path, api_server, body, status, diff, soft, code):
    rc, out = _process(tmp_path, api_server, body, status=status, diff=diff, soft=soft)
    assert rc == code, out
    if code == 1:
        assert "::error" in out
