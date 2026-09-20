# -*- coding: utf-8 -*-
"""`python -m styxx.diffgate --pr URL` — a public pull request, gated with no checkout.

The network is faked at the one seam `fetch_pr` exposes (`_open`), so these tests read
no GitHub and pass on a machine with no network. What they pin: the URL forms accepted,
the two documents fetched (JSON body, then the diff by Accept header), the verdict
being the same object `gate_diff_text` returns for that pair, the exit codes, the
refusal of `--run`, and the plain messages for 404 / 403 / too-large answers.

The 403 cases get their own test below: GitHub answers 403 for at least three unrelated
reasons and this entry point used to report all of them as an exhausted rate limit.
"""
import io
import json
import urllib.error

import pytest

from styxx.diffgate import fetch_pr, main

_DIFF = """\
--- a/src/retry.py
+++ b/src/retry.py
@@ -1,3 +1,6 @@
 def retry(n):
     return n
+
+def retry_once(n):
+    return retry(1)
--- /dev/null
+++ b/tests/test_retry.py
@@ -0,0 +1,2 @@
+def test_retry_once():
+    assert True
"""
_BODY = "Modified src/retry.py. Added 3 tests. Only touches files under src/."


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _opener(body=_BODY, diff=_DIFF, status=None, seen=None):
    def open_(req, timeout=60):
        if seen is not None:
            seen.append((req.full_url, req.get_header("Accept"), req.get_header("Authorization")))
        if status:
            raise urllib.error.HTTPError(req.full_url, status, "nope", {}, None)
        if req.get_header("Accept") == "application/vnd.github.diff":
            return _Resp(diff.encode("utf-8"))
        meta = {"title": "t", "body": body, "html_url": "https://github.com/o/r/pull/7",
                "base": {"ref": "main"}, "head": {"sha": "abcdef1234567890"}}
        return _Resp(json.dumps(meta).encode("utf-8"))
    return open_


@pytest.mark.parametrize("url", [
    "https://github.com/o/r/pull/7",
    "http://www.github.com/o/r/pull/7/files",
    "github.com/o/r/pull/7#issuecomment-1",
    "https://github.com/o/r/pull/7?diff=split",
])
def test_pr_url_forms_are_accepted(url):
    seen = []
    pr = fetch_pr(url, _open=_opener(seen=seen))
    assert pr["repo"] == "o/r" and pr["number"] == 7
    assert pr["body"] == _BODY and pr["diff"] == _DIFF
    assert pr["base"] == "main" and pr["head"].startswith("abcdef")
    assert [s[1] for s in seen] == ["application/vnd.github+json", "application/vnd.github.diff"]
    assert all(s[0] == "https://api.github.com/repos/o/r/pulls/7" for s in seen)


@pytest.mark.parametrize("bad", ["https://github.com/o/r", "https://gitlab.com/o/r/-/merge_requests/1",
                                 "o/r#7", "https://github.com/o/r/issues/7"])
def test_non_pr_urls_are_refused_before_any_fetch(bad):
    seen = []
    with pytest.raises(ValueError):
        fetch_pr(bad, _open=_opener(seen=seen))
    assert seen == []


def test_token_travels_as_bearer_and_nothing_else_changes():
    seen = []
    fetch_pr("https://github.com/o/r/pull/7", token="tok", _open=_opener(seen=seen))
    assert {s[2] for s in seen} == {"Bearer tok"}
    seen.clear()
    fetch_pr("https://github.com/o/r/pull/7", _open=_opener(seen=seen))
    assert {s[2] for s in seen} == {None}


@pytest.mark.parametrize("code, needle", [
    (404, "does not exist or is private"),
    (403, "either the request budget or credentials"),
    (429, "either the request budget or credentials"),
    (406, "too large"),
    (500, "HTTP 500"),
])
def test_github_errors_become_one_plain_sentence(code, needle):
    with pytest.raises(SystemExit) as e:
        fetch_pr("https://github.com/o/r/pull/7", _open=_opener(status=code))
    assert needle in str(e.value)


def test_cli_pr_gates_the_fetched_pair_and_exits_like_the_file_path(monkeypatch, capsys, tmp_path):
    import styxx.diffgate as dg
    monkeypatch.setattr(dg, "fetch_pr",
                        lambda url, token=None, timeout=60, _open=None: fetch_pr(url, _open=_opener()))
    out = tmp_path / "gate.json"
    rc = main(["--pr", "https://github.com/o/r/pull/7", "--out", str(out)])
    text = capsys.readouterr().out
    # the body lies twice (3 tests, only src/) and tells the truth once (retry.py modified)
    assert rc == 1
    assert "FAIL" in text and "contradicted=2" in text
    assert text.startswith("# o/r#7 — description and diff read from api.github.com, no checkout")
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["verdict"] == "FAIL" and d["base"] == "o/r#7:main" and d["head"].startswith("abcdef")
    kinds = {(c["kind"], c["verdict"]) for c in d["claims"]}
    assert ("tests_added", "CONTRADICTED") in kinds and ("only_touches", "CONTRADICTED") in kinds
    assert ("file_touched", "VERIFIED") in kinds


def test_cli_pr_with_a_summary_file_gates_that_file_instead_of_the_body(monkeypatch, capsys, tmp_path):
    import styxx.diffgate as dg
    monkeypatch.setattr(dg, "fetch_pr",
                        lambda url, token=None, timeout=60, _open=None: fetch_pr(url, _open=_opener()))
    doc = tmp_path / "BODY.md"
    doc.write_text("Modified src/retry.py. Added 1 test.", encoding="utf-8")
    rc = main([str(doc), "--pr", "https://github.com/o/r/pull/7"])
    assert rc == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_pr_refuses_run():
    with pytest.raises(SystemExit) as e:
        main(["--pr", "https://github.com/o/r/pull/7", "--run", "pytest -q"])
    assert e.value.code == 2


def test_cli_pr_uses_the_environment_token(monkeypatch, capsys):
    import styxx.diffgate as dg
    got = {}

    def fake(url, token=None, timeout=60, _open=None):
        got["token"] = token
        return fetch_pr(url, _open=_opener())
    monkeypatch.setattr(dg, "fetch_pr", fake)
    monkeypatch.setenv("GITHUB_TOKEN", "abc")
    main(["--pr", "https://github.com/o/r/pull/7"])
    assert got["token"] == "abc"


def test_cli_pr_says_when_the_description_is_empty(monkeypatch, capsys):
    import styxx.diffgate as dg
    monkeypatch.setattr(dg, "fetch_pr",
                        lambda url, token=None, timeout=60, _open=None: fetch_pr(url, _open=_opener(body="")))
    rc = main(["--pr", "https://github.com/o/r/pull/7"])
    text = capsys.readouterr().out
    assert rc == 0 and "the description is EMPTY" in text and "claims=0" in text


# --------------------------------------------------------------------------------------------- #
# 403 is not one failure
# --------------------------------------------------------------------------------------------- #

def _raiser(status, body=b"", headers=None):
    """An opener that answers with a real HTTPError, body and headers included.

    `_opener(status=...)` above raises with no body and no headers, which is the case where
    GitHub tells us nothing. These three cases are the ones where it does.
    """
    def open_(req, timeout=60):
        raise urllib.error.HTTPError(req.full_url, status, "nope", headers or {}, io.BytesIO(body))
    return open_


@pytest.mark.parametrize("name, body, headers, must_say, must_not_say", [
    (
        "credentials cannot see the repository",
        b'{"message":"GitHub access to this repository is not enabled for this session."}',
        {"X-RateLimit-Remaining": "15000"},
        ["NOT the rate limit", "Waiting will not help", "not enabled for this session"],
        ["Set GITHUB_TOKEN for a bigger budget"],
    ),
    (
        "hourly budget spent",
        b'{"message":"API rate limit exceeded for 1.2.3.4."}',
        {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1790000000"},
        ["budget is spent", "or wait"],
        ["cannot reach"],
    ),
    (
        "secondary limit",
        b'{"message":"You have exceeded a secondary rate limit."}',
        {"Retry-After": "60"},
        ["SECONDARY rate limit", "Waiting does fix this one", "60s"],
        ["cannot reach"],
    ),
])
def test_403_says_which_kind_of_403_it_is(name, body, headers, must_say, must_not_say):
    """The defect: every 403 was reported as an exhausted rate limit.

    Only one of these three is fixed by waiting. Telling the other two to wait sends a caller
    to a clock that will never help them, on the entry point the bookmarklet and the public
    "reply with a PR" offer both point at.
    """
    with pytest.raises(SystemExit) as e:
        fetch_pr("https://github.com/o/r/pull/7", _open=_raiser(403, body, headers))
    msg = str(e.value)
    for s in must_say:
        assert s in msg, f"{name}: missing {s!r} in {msg!r}"
    for s in must_not_say:
        assert s not in msg, f"{name}: should not say {s!r} in {msg!r}"


def test_403_quotes_github_rather_than_paraphrasing_it():
    """Whatever the branch, the caller gets GitHub's own sentence to search for."""
    with pytest.raises(SystemExit) as e:
        fetch_pr("https://github.com/o/r/pull/7",
                 _open=_raiser(403, b'{"message":"Resource not accessible by integration"}', {}))
    assert 'GitHub said: "Resource not accessible by integration"' in str(e.value)


def test_403_with_an_unreadable_body_still_names_both_possibilities():
    """A proxy can answer 403 with HTML. Guessing a cause from nothing is how this bug started."""
    with pytest.raises(SystemExit) as e:
        fetch_pr("https://github.com/o/r/pull/7", _open=_raiser(403, b"<html>nope</html>", {}))
    msg = str(e.value)
    assert "nope" in msg                      # the bytes are shown rather than swallowed
    assert "NOT the rate limit" in msg
