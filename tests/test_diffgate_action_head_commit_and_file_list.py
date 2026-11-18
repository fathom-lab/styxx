# -*- coding: utf-8 -*-
"""Two readings the Action makes where the event's own diff cannot be had.

A push that creates a branch has no commit before it, so its compare address names no base. The head
commit's one parent is looked up and the head commit is read against it: the message gated is that
commit's, and so is the diff. A commit with no parent, or a merge, is not read so.

A pull request whose diff GitHub refuses (HTTP 406, a diff past its size limit) is read against
GitHub's list of the pull request's files, as a diff of headers and no content, and only where every
claim the gate reads is about which files changed, the list is whole, every name is one git writes
into a header as it is, and the pull request did not change while the list was read. Otherwise the
gate does not run, as before, and the text says which of those did not hold.

Every test runs `main()` with the real `api()` against a server on 127.0.0.1.
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "made-up-token-0123456789"
AFTER, PARENT, OTHER = "a" * 40, "b" * 40, "c" * 40
HEAD = "d" * 40
DIFF_ACCEPT, JSON_ACCEPT = "application/vnd.github.diff", "application/vnd.github+json"
TOO_LARGE = json.dumps({"message": "Sorry, the diff exceeded the maximum number of lines (20000)",
                        "errors": [{"resource": "PullRequest", "field": "diff", "code": "too_large"}], "status": "406"})
N_LIST = "::notice title=styxx diffgate - list of files read for a refused diff::"
LIST_HEADING = "## styxx diffgate — the summary vs GitHub's list of the changed files"


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FC = _load_file("diffgate_action_fails_closed_for_fetch", Path(__file__).with_name("test_diffgate_action_fails_closed.py"))


class _Routed(http.server.BaseHTTPRequestHandler):
    """Answers each request with what `route(path, accept)` returns: (status, body). Every request is
    written down with its Authorization header."""
    route, seen = None, []

    def do_GET(self):
        cls = type(self)
        accept = self.headers.get("Accept")
        cls.seen.append((self.path, accept, self.headers.get("Authorization")))
        status, body = cls.route(self.path, accept)
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    handler = type("Routed", (_Routed,), {"seen": []})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield handler, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def act(monkeypatch):
    mod = FC.load("diffgate_action_fetch")
    monkeypatch.setattr(mod, "_sleep", lambda seconds: None)
    monkeypatch.setattr(mod, "_now", lambda: 1000.0)
    return mod


def run(act, tmp_path, monkeypatch, capsys, base, event_name, payload, soft="false", strict=False):
    """main() with the real api(); returns (exit code, stdout, summary)."""
    ev, summary = tmp_path / "event.json", tmp_path / "summary.md"
    ev.write_text(json.dumps(payload), encoding="utf-8")
    if summary.exists():
        summary.unlink()
    for name, value in (("GITHUB_EVENT_NAME", event_name), ("GITHUB_EVENT_PATH", str(ev)),
                        ("GITHUB_STEP_SUMMARY", str(summary)), ("GITHUB_API_URL", base), ("GH_TOKEN", TOKEN),
                        ("STYXX_STRICT", "true" if strict else "false"), ("STYXX_SOFT_FAIL", soft)):
        monkeypatch.setenv(name, value)
    rc = act.main()
    return rc, capsys.readouterr().out, summary.read_bytes().decode("utf-8") if summary.exists() else ""


# -- a pull request whose diff GitHub refuses ---------------------------------------------------------------------

def entry(name, status="modified", **more):
    return dict({"filename": name, "status": status, "additions": 1, "deletions": 0, "patch": "@@ -1 +1,2 @@\n x\n+y"},
                **more)


FILES = [entry("src/a.py"), entry("src/new.py", "added"), entry("docs/guide.md")]
HOLDS = "Modified src/a.py. 3 files changed."                       # true of FILES
OUTSIDE = "Only touches files under src/."                          # false of FILES: docs/guide.md
CONTENT = "Modified src/a.py. Adds 2 tests."                        # the second claim needs the diff's content


def pr_payload(base, body, files=FILES, **pr):
    fields = {"number": 7, "body": body, "url": base + "/repos/o/r/pulls/7", "changed_files": len(files),
              "head": {"sha": HEAD}}
    fields.update(pr)
    return {"pull_request": {k: v for k, v in fields.items() if v is not None}}


def refusing(files=FILES, *, list_status=200, now=None, diff_status=406, diff_body=TOO_LARGE, pages=None,
             now_status=200):
    """A route for a pull request whose diff is refused: the diff request answers 406, the list is
    served 100 files a page, and the pull request read again answers `now` (by default what the
    event said)."""
    def route(path, accept):
        if path == "/repos/o/r/pulls/7":
            if accept == DIFF_ACCEPT:
                return diff_status, diff_body
            if now_status != 200:
                return now_status, "{}"
            return 200, json.dumps(now if now is not None else {"head": {"sha": HEAD}, "changed_files": len(files)})
        if path.startswith("/repos/o/r/pulls/7/files?per_page=100&page="):
            if list_status != 200:
                return list_status, "{}"
            page = int(path.rsplit("=", 1)[1])
            if pages is not None:
                return 200, pages[page - 1]
            return 200, json.dumps(files[(page - 1) * 100:page * 100])
        return 404, "{}"
    return route


@pytest.mark.parametrize("soft", ["false", "true"])
def test_path_claims_that_hold_are_read_against_the_list(act, server, tmp_path, monkeypatch, capsys, soft):
    handler, base = server
    handler.route = staticmethod(refusing())
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, HOLDS), soft)
    assert rc == 0, out
    lines = out.splitlines()
    assert lines[0].startswith(N_LIST + "GitHub refused this pull request's diff (HTTP Error 406: Not Acceptable "
                               "(GitHub says: Sorry, the diff exceeded the maximum number of lines (20000))). No claim "
                               "the gate reads in the description needs the diff's content")
    assert lines[-1] == ("styxx diffgate: PASS - 2 claim(s), 0 contradicted; read against GitHub's list of 3 file(s), "
                         "the diff being refused")
    assert summary.startswith(LIST_HEADING + "\n\n**PASS** · 2 claim(s) checked on the PR #7 body\n\n_GitHub refused "
                              "this pull request's diff: `HTTP Error 406: Not Acceptable (GitHub says: Sorry, the diff "
                              "exceeded the maximum number of lines (20000))`.")
    assert "3 file(s), each with its path and whether it was added, removed, modified or renamed, and no line of " \
           "content._" in summary
    assert summary.count("| ✅ VERIFIED |") == 2 and "DID NOT RUN" not in summary
    # what was asked: the diff, one page of the list, and the pull request once more; the token on each
    assert [(p, a) for p, a, _ in handler.seen] == [
        ("/repos/o/r/pulls/7", DIFF_ACCEPT), ("/repos/o/r/pulls/7/files?per_page=100&page=1", JSON_ACCEPT),
        ("/repos/o/r/pulls/7", JSON_ACCEPT)]
    assert all(auth == f"Bearer {TOKEN}" for _, _, auth in handler.seen)
    assert TOKEN not in out + summary


def test_a_contradicted_path_claim_fails_a_blocking_check_from_the_list(act, server, tmp_path, monkeypatch, capsys):
    handler, base = server
    handler.route = staticmethod(refusing())
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, OUTSIDE))
    assert rc == 1 and FC.E_CONTRA in out and "| ❌ CONTRADICTED |" in summary
    assert summary.startswith(LIST_HEADING)
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, OUTSIDE), "true")
    assert rc == 0 and FC.E_CONTRA in out


def test_no_line_of_content_reaches_the_gate(act, server, tmp_path, monkeypatch, capsys):
    """The list's entries carry a patch; the gate is handed headers and nothing else."""
    handler, base = server
    handler.route = staticmethod(refusing())
    handed = []
    real = act.gate_diff_text

    def spy(summary, diff, **k):
        handed.append(diff)
        return real(summary, diff, **k)

    monkeypatch.setattr(act, "gate_diff_text", spy)
    run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, HOLDS))
    assert handed[-1] == ("diff --git a/src/a.py b/src/a.py\ndiff --git a/src/new.py b/src/new.py\nnew file mode 100644\n"
                          "diff --git a/docs/guide.md b/docs/guide.md\n")


def test_a_rename_and_a_removal_are_written_as_git_writes_them(act):
    assert act._headers_only([("renamed", "lib/a.py", "src/a.py"), ("removed", "old.txt", None)]) == (
        "diff --git a/src/a.py b/lib/a.py\nsimilarity index 100%\nrename from src/a.py\nrename to lib/a.py\n"
        "diff --git a/old.txt b/old.txt\ndeleted file mode 100644\n")


def test_a_rename_out_of_the_scope_is_contradicted_from_the_list(act, server, tmp_path, monkeypatch, capsys):
    files = [entry("src/a.py"), entry("lib/b.py", "renamed", previous_filename="src/b.py")]
    handler, base = server
    handler.route = staticmethod(refusing(files))
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, OUTSIDE, files))
    assert rc == 1 and FC.E_CONTRA in out


@pytest.mark.parametrize("soft, code", [("false", 1), ("true", 0)])
def test_a_claim_that_needs_content_is_not_judged_and_the_list_is_not_asked_for(act, server, tmp_path, monkeypatch,
                                                                              capsys, soft, code):
    handler, base = server
    handler.route = staticmethod(refusing())
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, CONTENT), soft)
    assert rc == code
    why = ("GitHub's list of the pull request's files was not asked for: it carries the files' paths and not all of "
           "their content, and 1 of the 2 claim(s) the gate reads in the description need the diff's content. A "
           "description whose claims are all about which files changed is read against that list.")
    assert why in out and why in summary
    assert "(GitHub says: Sorry, the diff exceeded the maximum number of lines (20000))" in out
    assert [p for p, _, _ in handler.seen] == ["/repos/o/r/pulls/7"]            # the diff, and nothing after it
    assert N_LIST not in out and "DID NOT RUN" in summary
    if code:
        assert FC.E_FETCH in out
        assert "A re-run asks for the diff again; while GitHub refuses it, the gate reads at most the list of files." in out
        assert "no rewording of the description" not in out


def test_report_mode_prints_these_lines_for_a_refused_diff_whose_list_is_not_used(act, server, tmp_path, monkeypatch,
                                                                                 capsys):
    handler, base = server
    handler.route = staticmethod(refusing())
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, CONTENT), "true")
    assert out == ("::warning title=styxx diffgate - could not fetch the diff::HTTP Error 406: Not Acceptable (GitHub "
                   "says: Sorry, the diff exceeded the maximum number of lines (20000)). GitHub's list of the pull "
                   "request's files was not asked for: it carries the files' paths and not all of their content, and 1 "
                   "of the 2 claim(s) the gate reads in the description need the diff's content. A description whose "
                   "claims are all about which files changed is read against that list. The gate did not run; this is "
                   "not a pass.\n")
    assert summary.startswith("## styxx diffgate — DID NOT RUN\n\nThe diff could not be fetched: `HTTP Error 406: Not "
                              "Acceptable (GitHub says: Sorry, the diff exceeded the maximum number of lines (20000))`."
                              "\n\nGitHub's list of the pull request's files was not asked for:")


def test_tests_pass_beside_a_path_claim_does_not_keep_the_list_from_being_read(act, server, tmp_path, monkeypatch,
                                                                              capsys):
    handler, base = server
    handler.route = staticmethod(refusing())
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request",
                           pr_payload(base, "Modified src/a.py. All tests pass."))
    assert rc == 0 and N_LIST in out
    assert "| ✅ VERIFIED |" in summary and "| ❓ UNCHECKABLE |" in summary


def test_a_tests_pass_claim_reads_the_same_on_a_diff_and_on_its_headers(act):
    """Why `tests_pass` may stand beside path claims: in the Action it is UNCHECKABLE with the same
    reason whatever the diff holds."""
    full = FC.DIFF
    headers = "".join(ln + "\n" for ln in full.split("\n") if ln.startswith("diff --git "))
    a, b = (act.gate_diff_text("All tests pass.", d).claims[0] for d in (full, headers))
    assert (a.kind, a.verdict, a.why) == (b.kind, b.verdict, b.why) == ("tests_pass", "UNCHECKABLE", a.why)


def test_the_kinds_read_from_the_list_are_the_gates_kinds_less_the_ones_that_read_content(act):
    """A kind the gate gains later is not read from the list until someone decides it can be."""
    from styxx import diffgate
    kinds = {template[0] for template in diffgate._TEMPLATES}
    assert kinds - act._LIST_KINDS == {"compat_claim", "symbol_added", "tests_added"}
    assert act._LIST_KINDS <= kinds


@pytest.mark.parametrize("strict, code", [(False, 0), (True, 1)])
def test_with_no_claim_the_list_is_not_asked_for_and_the_job_does_what_it_did(act, server, tmp_path, monkeypatch,
                                                                             capsys, strict, code):
    handler, base = server
    handler.route = staticmethod(refusing())
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, FC.PROSE),
                           "false", strict)
    assert rc == code
    assert [p for p, _, _ in handler.seen] == ["/repos/o/r/pulls/7"]
    assert "GitHub's list" not in out + summary
    if strict:
        assert "no rewording of the description makes it serve one" in out


NOT_ASKED = "GitHub's list of the pull request's files was not asked for: the event does not carry"
NOT_USED = "GitHub's list of the pull request's files was not used: "
NAMES = NOT_USED + "1 of its 3 file(s) have a status other than added, removed, modified or renamed, or a name"
RENAME = "or for a renamed file an earlier name that is missing or is such a name)."


def _with(i, **change):
    files = [dict(f) for f in FILES]
    files[i].update(change)
    return files


@pytest.mark.parametrize("payload_change, route_change, sentence", [
    (dict(changed_files=None), {}, NOT_ASKED),
    (dict(changed_files="3"), {}, NOT_ASKED),
    (dict(changed_files=True), {}, NOT_ASKED),
    (dict(head=None), {}, NOT_ASKED),
    (dict(changed_files=4), {}, NOT_USED + "it holds 3 file(s) and the event says 4 changed (GitHub lists at most "
                                           "3,000). A re-run reads it again."),
    ({}, dict(files=_with(0, status="copied")), NAMES),
    ({}, dict(files=_with(0, status="changed")), NAMES),
    ({}, dict(files=_with(0, filename="src/café.py")), NAMES),
    ({}, dict(files=_with(0, filename='src/a".py')), NAMES),
    ({}, dict(files=_with(0, filename="src\\a.py")), NAMES),
    ({}, dict(files=_with(0, filename=" src/a.py")), NAMES),
    ({}, dict(files=_with(0, filename="src/a.py ")), NAMES),
    ({}, dict(files=_with(0, filename="src/a\tb.py")), NAMES),
    ({}, dict(files=_with(0, filename="")), NAMES),
    ({}, dict(files=_with(0, filename=7)), NAMES),
    ({}, dict(files=_with(0, filename="src/new.py")), NAMES),                       # listed twice
    ({}, dict(files=_with(0, status="renamed")), RENAME),                           # a rename with no previous name
    ({}, dict(files=_with(0, status="renamed", previous_filename="oldé.py")), RENAME),
    ({}, dict(pages=['{"message": "x"}']), NOT_USED + "a page of it is not a list of files."),
    ({}, dict(pages=['["a", "b", "c"]']), NOT_USED + "a page of it is not a list of files."),
    ({}, dict(pages=["not json"]), "GitHub's list of the pull request's files could not be fetched: GitHub's answer "
                                   "for the list of files is not JSON."),
    ({}, dict(list_status=403), "GitHub's list of the pull request's files could not be fetched: HTTP Error 403: "
                                "Forbidden. The list needs the permission `pull-requests: read`."),
    ({}, dict(list_status=404), "The list needs the permission `pull-requests: read`."),
    ({}, dict(list_status=410), "GitHub's list of the pull request's files could not be fetched: HTTP Error 410: "
                                "Gone."),
    ({}, dict(now={"head": {"sha": OTHER}, "changed_files": 3}), NOT_USED + "the pull request changed while the list "
                                                                            "was read"),
    ({}, dict(now={"head": {"sha": HEAD}, "changed_files": 4}), NOT_USED + "the pull request changed while the list "
                                                                           "was read"),
    ({}, dict(now={"head": {"sha": HEAD}, "changed_files": 3.0}), NOT_USED + "the pull request changed while the "
                                                                             "list was read"),
    ({}, dict(now=["x"]), NOT_USED + "read again after the list, the pull request's address did not answer with a "
                                     "pull request. A re-run reads it again."),
    ({}, dict(now_status=410), "GitHub's list of the pull request's files was not used, because the pull request "
                               "could not be read again after it: HTTP Error 410: Gone. A re-run reads it again."),
    (dict(changed_files=3001), {}, NOT_USED + "it holds 3 file(s) and the event says 3001 changed (GitHub lists at "
                                              "most 3,000). A pull request of more files than that is not read from "
                                              "the list, so a re-run fails the same way."),
])
def test_a_list_that_is_not_whole_or_not_this_pull_request_is_not_used(act, server, tmp_path, monkeypatch, capsys,
                                                                      payload_change, route_change, sentence):
    """Each way the list fails to stand in for the diff: the gate does not run, a blocking check
    fails as it does for any diff that was not read under a claim, and the text says which."""
    handler, base = server
    files = route_change.pop("files", FILES) if "files" in route_change else FILES
    handler.route = staticmethod(refusing(files, **{k: v for k, v in route_change.items() if k != "files"}))
    payload = pr_payload(base, HOLDS, **payload_change)
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", payload)
    assert rc == 1 and FC.E_FETCH in out, out
    assert sentence in out, out
    assert "DID NOT RUN, and the check fails" in summary and N_LIST not in out
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", payload, "true")
    assert rc == 0 and sentence in out and "::error" not in out


def test_a_name_with_a_space_inside_it_is_read(act, server, tmp_path, monkeypatch, capsys):
    files = [entry("docs/user guide.md"), entry("src/a b/c.py", "added"), entry("src/a.py")]
    handler, base = server
    handler.route = staticmethod(refusing(files))
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request",
                           pr_payload(base, "3 files changed.", files))
    assert rc == 0 and N_LIST in out and "| ✅ VERIFIED |" in summary


def test_a_list_of_more_than_one_page(act, server, tmp_path, monkeypatch, capsys):
    files = [entry(f"src/f{i:03d}.py") for i in range(101)]
    handler, base = server
    handler.route = staticmethod(refusing(files))
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "pull_request",
                     pr_payload(base, "101 files changed. Only touches files under src/.", files))
    assert rc == 0 and N_LIST in out and "list of 101 file(s)" in out
    assert [p for p, _, _ in handler.seen][1:3] == ["/repos/o/r/pulls/7/files?per_page=100&page=1",
                                                    "/repos/o/r/pulls/7/files?per_page=100&page=2"]


def test_a_list_of_exactly_one_page_asks_for_the_next_and_stops(act, server, tmp_path, monkeypatch, capsys):
    files = [entry(f"src/f{i:03d}.py") for i in range(100)]
    handler, base = server
    handler.route = staticmethod(refusing(files))
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "pull_request",
                     pr_payload(base, "100 files changed.", files))
    assert rc == 0 and "list of 100 file(s)" in out
    assert len(handler.seen) == 4          # the diff, two pages (the second empty), the pull request again


def test_what_github_says_is_escaped_where_it_is_printed(act, server, tmp_path, monkeypatch, capsys):
    said = "big`diff|<b>x</b>\n::error::forged%0A [a](http://x) \x1b\u202e" + TOKEN
    handler, base = server
    handler.route = staticmethod(refusing(diff_body=json.dumps({"message": said})))
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, CONTENT), "true")
    assert rc == 0 and len(out.splitlines()) == 1 and TOKEN not in out + summary
    assert "(GitHub says: big`diff|<b>x</b>%0A::error::forged%250A [a](http://x) \\x1b\\u202e***)" in out
    assert ("`HTTP Error 406: Not Acceptable (GitHub says: big'diff|<b>x</b> ::error::forged%0A [a](http://x) "
            "\\x1b\\u202e***)`") in summary


@pytest.mark.parametrize("body", ["", "not json", "[1]", '{"message": 7}', '{"other": "x"}'])
def test_a_refusal_with_no_message_is_shown_without_one(act, server, tmp_path, monkeypatch, capsys, body):
    handler, base = server
    handler.route = staticmethod(refusing(diff_body=body))
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, HOLDS), "true")
    assert rc == 0 and "GitHub says" not in out and N_LIST in out


def test_another_error_than_406_does_not_ask_for_the_list(act, server, tmp_path, monkeypatch, capsys):
    handler, base = server
    handler.route = staticmethod(refusing(diff_status=404, diff_body="{}"))
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, HOLDS))
    assert rc == 1 and FC.E_FETCH in out and "GitHub's list" not in out
    assert "no rewording of the description makes it serve one" in out
    assert len(handler.seen) == 1


# -- a push that creates a branch ---------------------------------------------------------------------------------

def push_payload(base, message, *, after=AFTER, git_commits=True, ref="refs/heads/new", head_id=None, commits="one"):
    repo = {"compare_url": base + "/repos/o/r/compare/{base}...{head}"}
    if git_commits:
        repo["git_commits_url"] = base + "/repos/o/r/git/commits{/sha}"
    payload = {"head_commit": {"id": after if head_id is None else head_id, "message": message}, "before": "0" * 40,
               "after": after, "created": True, "ref": ref, "repository": repo}
    if commits is not None:                 # the commits of the push: by default the head commit and no other
        payload["commits"] = [{"id": after}] if commits == "one" else commits
    return payload


def branching(parents, *, lookup_status=200, compare_status=200, diff=FC.DIFF, lookup_body=None):
    def route(path, accept):
        if path == f"/repos/o/r/git/commits/{AFTER}":
            if lookup_status != 200:
                return lookup_status, "{}"
            return 200, lookup_body if lookup_body is not None else json.dumps(
                {"sha": AFTER, "parents": [{"sha": p} for p in parents]})
        if path == f"/repos/o/r/compare/{PARENT}...{AFTER}" and accept == DIFF_ACCEPT:
            return compare_status, diff if compare_status == 200 else "{}"
        return 404, "{}"
    return route


OWN = ("_This push created its branch with one commit, so there is no commit before it to compare with. The diff read "
       "is that commit's own, against its one parent._")
OWN_LAST = ("; the diff read is the head commit's own against its one parent (the push created its branch with that "
            "one commit)")
NO_BASE = "A push that creates a branch has no base commit to compare with"


@pytest.mark.parametrize("soft, code", [("false", 1), ("true", 0)])
def test_the_head_commit_of_a_created_branch_is_read_against_its_one_parent(act, server, tmp_path, monkeypatch, capsys,
                                                                           soft, code):
    handler, base = server
    handler.route = staticmethod(branching([PARENT]))
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "push", push_payload(base, FC.FALSE), soft)
    assert rc == code and FC.E_CONTRA in out                       # the diff holds a file outside src/
    assert out.splitlines()[1 if soft == "false" else 1].startswith("styxx diffgate: FAIL - 1 claim(s), 1 contradicted")
    assert [ln for ln in out.splitlines() if ln.startswith("styxx diffgate: ")][-1].endswith(OWN_LAST)
    assert OWN in summary and summary.startswith("## styxx diffgate — the summary vs the diff")
    # the parent is looked up, then compared with the head; the address with no base is never asked
    assert [(p, a) for p, a, _ in handler.seen] == [(f"/repos/o/r/git/commits/{AFTER}", JSON_ACCEPT),
                                                    (f"/repos/o/r/compare/{PARENT}...{AFTER}", DIFF_ACCEPT)]
    assert all(auth == f"Bearer {TOKEN}" for _, _, auth in handler.seen)


def test_a_claim_that_holds_of_the_head_commit_passes(act, server, tmp_path, monkeypatch, capsys):
    handler, base = server
    handler.route = staticmethod(branching([PARENT], diff=FC.DIFF_SRC))
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "push", push_payload(base, FC.FALSE))
    assert rc == 0 and out.strip().endswith("styxx diffgate: PASS - 1 claim(s), 0 contradicted" + OWN_LAST)
    assert OWN in summary


def test_a_head_commit_with_an_empty_diff_is_unmeasured_and_says_what_was_read(act, server, tmp_path, monkeypatch,
                                                                              capsys):
    handler, base = server
    handler.route = staticmethod(branching([PARENT], diff=""))
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "push", push_payload(base, FC.FALSE))
    assert rc == 1 and FC.E_UNMEASURED in out and OWN in summary
    assert out.strip().endswith(OWN_LAST)


@pytest.mark.parametrize("parents, said", [
    ([], "the head commit has no parent to read it against"),
    ([PARENT, OTHER], "the head commit is a merge of 2 parents, and this script reads a commit against one parent only"),
])
def test_a_head_commit_without_exactly_one_parent_is_not_read(act, server, tmp_path, monkeypatch, capsys, parents, said):
    handler, base = server
    handler.route = staticmethod(branching(parents))
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "push", push_payload(base, FC.FALSE))
    assert rc == 1 and FC.E_FETCH in out
    assert said in out and NO_BASE + ", so a re-run fails the same way." in out
    assert len(handler.seen) == 1 and OWN not in summary
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "push", push_payload(base, FC.FALSE), "true")
    assert rc == 0 and said in out and "::error" not in out


ASKS_AGAIN = NO_BASE + ", so the head commit's parent was looked up, and the lookup failed; a re-run asks again."
NOT_USABLE = NO_BASE + ", so the head commit's parent was looked up, and GitHub's answer could not be used."


@pytest.mark.parametrize("change, said, ending", [
    (dict(lookup_status=404), "HTTP Error 404: Not Found", ASKS_AGAIN),
    (dict(lookup_status=410), "HTTP Error 410: Gone", ASKS_AGAIN),
    (dict(lookup_body="not json"), "GitHub's answer for the head commit is not JSON", NOT_USABLE),
    (dict(lookup_body='["x"]'), "GitHub's answer for the head commit is not for that commit", NOT_USABLE),
    (dict(lookup_body=json.dumps({"sha": OTHER, "parents": [{"sha": PARENT}]})),
     "GitHub's answer for the head commit is not for that commit", NOT_USABLE),
    (dict(lookup_body=json.dumps({"sha": AFTER})), "holds no list of parents", NOT_USABLE),
    (dict(lookup_body=json.dumps({"sha": AFTER, "parents": [{"sha": "zz"}]})),
     "names a parent that is not a commit id", NOT_USABLE),
    (dict(lookup_body=json.dumps({"sha": AFTER, "parents": ["x"]})), "names a parent that is not a commit id",
     NOT_USABLE),
])
def test_a_lookup_that_fails_or_cannot_be_used(act, server, tmp_path, monkeypatch, capsys, change, said, ending):
    """A request that failed can be asked again and the text says so; an answer that cannot be used is
    said to be that. Neither is read, and the compare is never asked."""
    handler, base = server
    handler.route = staticmethod(branching([PARENT], **change))
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "push", push_payload(base, FC.FALSE))
    assert rc == 1 and FC.E_FETCH in out and said in out
    assert ending in out
    assert "fails the same way" not in out
    assert all("/compare/" not in p for p, _, _ in handler.seen)


NOT_LISTED = "the event does not list the push's commits as the head commit alone"


@pytest.mark.parametrize("commits, held", [
    ([{"id": OTHER}, {"id": AFTER}], "the push that created this branch holds 2 commits"),
    ([], "the push that created this branch holds 0 commits"),
    ([{"id": OTHER}], NOT_LISTED),
    ([{"id": AFTER.upper()}], NOT_LISTED),
    (None, NOT_LISTED),
])
def test_a_created_branch_whose_push_is_not_its_head_commit_alone_is_not_read(act, server, tmp_path, monkeypatch,
                                                                             capsys, commits, held):
    """The head commit's own diff is the diff of the push only where the push holds that one commit.
    A push of more commits (or one that names none, or another) is not read as its head commit: it
    fails a blocking check when the message holds a claim, as before, and nothing is asked."""
    handler, base = server
    handler.route = staticmethod(branching([PARENT], diff=FC.DIFF_SRC))        # the head commit's own diff would pass
    payload = push_payload(base, FC.FALSE, commits=commits)
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "push", payload)
    assert rc == 1 and FC.E_FETCH in out
    assert f"{held}, and a created branch is read only where its push is the head commit alone" in out
    assert NO_BASE + ", so a re-run fails the same way." in out
    assert handler.seen == [] and OWN not in summary
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "push", payload, "true")
    assert rc == 0 and "::error" not in out and held in out


def test_a_lookup_address_off_the_origin_is_refused_by_its_own_name(act, server, tmp_path, monkeypatch, capsys):
    handler, base = server
    handler.route = staticmethod(branching([PARENT]))
    payload = push_payload(base, FC.FALSE)
    payload["repository"]["git_commits_url"] = "http://127.0.0.1:9/repos/o/r/git/commits{/sha}"
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "push", payload)
    assert rc == 1 and FC.E_FETCH in out and handler.seen == []
    assert f"the head commit's address 'http://127.0.0.1:9/repos/o/r/git/commits/{AFTER}' is not on {base}" in out
    assert ASKS_AGAIN in out


def test_a_token_that_the_cut_of_githubs_message_falls_inside_is_hidden_whole(act, server, tmp_path, monkeypatch,
                                                                             capsys):
    said = "x" * 290 + TOKEN + " and more"
    handler, base = server
    handler.route = staticmethod(refusing(diff_body=json.dumps({"message": said})))
    for body in (HOLDS, CONTENT):
        rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, body), "true")
        assert rc == 0 and TOKEN[:12] not in out + summary and "x" * 290 + "***" in out


def test_a_parent_that_is_found_and_a_compare_that_fails_is_a_failed_fetch_like_any_other(act, server, tmp_path,
                                                                                         monkeypatch, capsys):
    handler, base = server
    handler.route = staticmethod(branching([PARENT], compare_status=404))
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "push", push_payload(base, FC.FALSE))
    assert rc == 1 and FC.E_FETCH in out and NO_BASE not in out
    assert "A re-run reads the diff again" in out and len(handler.seen) == 2


@pytest.mark.parametrize("payload_change", [dict(git_commits=False), dict(after="zzz"), dict(after="A" * 40),
                                            dict(after="a" * 39), dict(head_id=OTHER)])
def test_where_the_event_does_not_carry_what_the_lookup_needs_nothing_is_looked_up(act, server, tmp_path, monkeypatch,
                                                                                  capsys, payload_change):
    """As before this change: the compare address with no base is asked, the API answers 404, and a
    blocking check fails when the message holds a claim."""
    handler, base = server
    handler.route = staticmethod(branching([PARENT]))
    payload = push_payload(base, FC.FALSE, **payload_change)
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "push", payload)
    assert rc == 1 and FC.E_FETCH in out and NO_BASE + ", so a re-run fails the same way." in out
    assert all("/git/commits/" not in p for p, _, _ in handler.seen)


def test_a_push_that_creates_a_tag_still_asks_nothing(act, server, tmp_path, monkeypatch, capsys):
    handler, base = server
    handler.route = staticmethod(branching([PARENT]))
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "push",
                     push_payload(base, FC.FALSE, ref="refs/tags/v1"))
    assert rc == 0 and "creates a tag is not gated" in out and handler.seen == []


def test_a_push_to_a_branch_that_exists_is_read_as_before(act, server, tmp_path, monkeypatch, capsys):
    handler, base = server

    def route(path, accept):
        return (200, FC.DIFF) if path == f"/repos/o/r/compare/{OTHER}...{AFTER}" else (404, "{}")

    handler.route = staticmethod(route)
    payload = push_payload(base, FC.FALSE)
    payload.update(before=OTHER, created=False, ref="refs/heads/main")
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "push", payload)
    assert rc == 1 and FC.E_CONTRA in out and OWN not in summary and OWN_LAST not in out
    assert len(handler.seen) == 1


# -- three more edges, each found by a changed script the tests above let pass ------------------------------------

def test_what_github_says_is_escaped_where_the_list_is_read_too(act, server, tmp_path, monkeypatch, capsys):
    said = "big`diff|<b>x</b>\n::error::forged%0A [a](http://x) \x1b\u202e" + TOKEN
    handler, base = server
    handler.route = staticmethod(refusing(diff_body=json.dumps({"message": said})))
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "pull_request", pr_payload(base, HOLDS), "true")
    assert rc == 0 and TOKEN not in out + summary
    lines = out.splitlines()
    assert lines[0].startswith(N_LIST) and not any(ln.startswith("::error") for ln in lines)
    assert "(GitHub says: big`diff|<b>x</b>%0A::error::forged%250A [a](http://x) \\x1b\\u202e***)" in lines[0]
    assert ("`HTTP Error 406: Not Acceptable (GitHub says: big'diff|<b>x</b> ::error::forged%0A [a](http://x) "
            "\\x1b\\u202e***)`") in summary


def test_a_push_whose_compare_answers_406_is_a_failed_fetch_and_no_list_is_spoken_of(act, server, tmp_path,
                                                                                    monkeypatch, capsys):
    handler, base = server

    def route(path, accept):
        return 406, TOO_LARGE

    handler.route = staticmethod(route)
    payload = push_payload(base, "Modified src/a.py.")
    payload.update(before=OTHER, created=False, ref="refs/heads/main")
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "push", payload)
    assert rc == 1 and FC.E_FETCH in out and "GitHub's list" not in out + summary
    assert "no rewording of the head commit message makes it serve one" in out
    assert len(handler.seen) == 1


def test_a_created_ref_that_is_no_branch_is_not_looked_up(act, server, tmp_path, monkeypatch, capsys):
    handler, base = server
    handler.route = staticmethod(branching([PARENT]))
    rc, out, _ = run(act, tmp_path, monkeypatch, capsys, base, "push",
                     push_payload(base, FC.FALSE, ref="refs/notes/commits"))
    assert rc == 1 and FC.E_FETCH in out
    assert all("/git/commits/" not in p for p, _, _ in handler.seen)


def test_an_unmeasured_head_commit_with_no_claim_passes_and_says_what_was_read(act, server, tmp_path, monkeypatch,
                                                                              capsys):
    handler, base = server
    handler.route = staticmethod(branching([PARENT], diff=""))
    rc, out, summary = run(act, tmp_path, monkeypatch, capsys, base, "push", push_payload(base, FC.PROSE))
    assert rc == 0 and OWN in summary
    assert out.strip().endswith("reads, so the job passes" + OWN_LAST)
