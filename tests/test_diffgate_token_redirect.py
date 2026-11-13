# -*- coding: utf-8 -*-
"""fetch_pr's token is sent to the API's origin and nowhere else, whatever a redirect names.

Through 8.0.1 `fetch_pr` put ``Authorization: Bearer <token>`` in the request's ordinary headers and
sent it through `urllib.request.urlopen`, which follows 301, 302, 303, 307 and 308 for a GET and copies
those headers to whatever address the Location names (another host, another port, https to http). The
token now travels as an unredirected header, which urllib never copies to a redirected request, and
fetch_pr's own opener follows a redirect only within the API's origin, putting the token back there.

Everything here runs against real HTTP servers on 127.0.0.1, because a test that replaces the client
cannot see redirect handling: `api` stands for the API and answers with redirects, `elsewhere` stands for
any other address and records every header it is sent. Two ways point fetch_pr at loopback, and each test
says which: a caller-supplied opener that rewrites the request's address to `api` before a stock urllib
opener sends it (the request fetch_pr built, headers and all, is what goes), or the module constant
`styxx.diffgate._API_BASE` monkeypatched to `api`'s address. The token is a made-up string.
"""
from __future__ import annotations

import http.server
import json
import threading
import urllib.request

import pytest

import styxx.diffgate as dg
from styxx.diffgate import fetch_pr, main

TOKEN = "made-up-token-0123456789abcdef"     # not a real token; checked for in every text shown
PR_URL = "https://github.com/o/r/pull/7"
API_PATH = "/repos/o/r/pulls/7"
CODES = (301, 302, 303, 307, 308)

_DIFF = """\
--- a/src/retry.py
+++ b/src/retry.py
@@ -1,3 +1,6 @@
 def retry(n):
     return n
+
+def retry_once(n):
+    return retry(1)
"""
_META = {"title": "t", "body": "Modified src/retry.py.", "html_url": "https://github.com/o/r/pull/7",
         "base": {"ref": "main"}, "head": {"sha": "abcdef1234567890"}}


class Server:
    """A real HTTP server on 127.0.0.1. `routes` maps a path to (code, Location) to answer with a
    redirect; any other path is served as the pull request (JSON or the diff, by Accept). Every request
    is recorded as (path, {header: value})."""

    def __init__(self):
        self.routes: dict[str, tuple[int, str]] = {}
        self.seen: list[tuple[str, dict]] = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
                if self.path in outer.routes:
                    code, loc = outer.routes[self.path]
                    self.send_response(code)
                    self.send_header("Location", loc)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.headers.get("Accept") == "application/vnd.github.diff":
                    body = _DIFF.encode("utf-8")
                else:
                    body = json.dumps(_META).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def auths(self) -> list:
        return [h.get("authorization") for _, h in self.seen]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def servers(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(dg, "_API_OPENER", None, raising=False)    # built afresh, with no proxy
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})
    api, elsewhere = Server(), Server()
    yield api, elsewhere
    api.close()
    elsewhere.close()


def rewriting_opener(api: Server, opener=None):
    """A caller-supplied opener: the request fetch_pr built, its address moved to `api`, sent by a
    stock urllib opener (which follows every redirect)."""
    stock = opener or urllib.request.build_opener()

    def open_(req, timeout=60):
        assert req.full_url == "https://api.github.com" + API_PATH
        req.full_url = api.base + API_PATH
        return stock.open(req, timeout=timeout)
    return open_


def point_api_at(monkeypatch, api: Server):
    """The module constant moved to `api` (it exists only with the repair; without it this fails
    here, before any request is made, rather than asking the real API)."""
    assert hasattr(dg, "_API_BASE"), "styxx.diffgate has no _API_BASE to point at loopback"
    monkeypatch.setattr(dg, "_API_BASE", api.base)


def assert_no_token(*texts):
    for t in texts:
        assert TOKEN not in str(t)


# ── the baseline: no opener copies the token to a redirected request ───────────────────────────────

@pytest.mark.parametrize("code", CODES)
def test_a_caller_opener_follows_the_redirect_without_the_token(servers, code):
    api, elsewhere = servers
    api.routes[API_PATH] = (code, elsewhere.base + API_PATH)
    pr = fetch_pr(PR_URL, token=TOKEN, _open=rewriting_opener(api))
    assert pr["diff"] == _DIFF and pr["title"] == "t"           # the redirect was followed, both times
    assert api.auths() == [f"Bearer {TOKEN}"] * 2               # the API's origin got the token
    assert len(elsewhere.seen) == 2
    assert elsewhere.auths() == [None, None]                    # the other address never did


def test_a_caller_opener_through_a_chain_of_two_redirects_never_forwards_the_token(servers):
    api, elsewhere = servers
    api.routes[API_PATH] = (302, elsewhere.base + "/hop")
    elsewhere.routes["/hop"] = (307, elsewhere.base + API_PATH)
    pr = fetch_pr(PR_URL, token=TOKEN, _open=rewriting_opener(api))
    assert pr["diff"] == _DIFF
    assert api.auths() == [f"Bearer {TOKEN}"] * 2
    assert [p for p, _ in elsewhere.seen] == ["/hop", API_PATH] * 2
    assert all(a is None for a in elsewhere.auths())


def test_a_caller_opener_back_to_the_first_server_carries_no_token_either(servers):
    """The baseline alone drops the token on every redirect, even one back to the same server: only
    fetch_pr's own opener puts it back, and only within the API's origin."""
    api, _ = servers
    api.routes[API_PATH] = (301, api.base + "/moved")
    fetch_pr(PR_URL, token=TOKEN, _open=rewriting_opener(api))
    assert [(p, h.get("authorization")) for p, h in api.seen] == [
        (API_PATH, f"Bearer {TOKEN}"), ("/moved", None)] * 2


def test_the_installed_opener_through_urlopen_follows_without_the_token(servers, monkeypatch):
    api, elsewhere = servers
    point_api_at(monkeypatch, api)
    api.routes[API_PATH] = (302, elsewhere.base + API_PATH)
    urllib.request.install_opener(urllib.request.build_opener())
    try:
        pr = fetch_pr(PR_URL, token=TOKEN, _open=urllib.request.urlopen)
    finally:
        urllib.request.install_opener(None)
    assert pr["diff"] == _DIFF
    assert api.auths() == [f"Bearer {TOKEN}"] * 2
    assert len(elsewhere.seen) == 2 and elsewhere.auths() == [None, None]


def test_fetch_pr_does_not_use_an_installed_opener(servers, monkeypatch):
    api, elsewhere = servers
    point_api_at(monkeypatch, api)
    api.routes[API_PATH] = (302, elsewhere.base + API_PATH)
    used = []

    class Spy(urllib.request.BaseHandler):
        def http_request(self, req):
            used.append(req.full_url)
            return req
    urllib.request.install_opener(urllib.request.build_opener(Spy))
    try:
        with pytest.raises(SystemExit):
            fetch_pr(PR_URL, token=TOKEN)
    finally:
        urllib.request.install_opener(None)
    assert used == [] and elsewhere.seen == []


# ── fetch_pr's own opener: refuses a redirect off the origin, keeps the token within it ────────────

@pytest.mark.parametrize("code", CODES)
def test_a_redirect_off_the_origin_is_refused_and_the_other_address_gets_nothing(servers, monkeypatch, code):
    api, elsewhere = servers
    point_api_at(monkeypatch, api)
    api.routes[API_PATH] = (code, elsewhere.base + API_PATH)
    with pytest.raises(SystemExit) as e:
        fetch_pr(PR_URL, token=TOKEN)
    msg = str(e.value)
    assert f"answers {code}, a redirect to" in msg and "not followed" in msg
    assert repr(elsewhere.base + API_PATH) in msg
    assert_no_token(msg, repr(e.value), e.value.__cause__)
    assert api.auths() == [f"Bearer {TOKEN}"]                   # asked once, with the token; then refused
    assert elsewhere.seen == []


@pytest.mark.parametrize("code", CODES)
def test_a_redirect_within_the_origin_keeps_the_token(servers, monkeypatch, code):
    api, elsewhere = servers
    point_api_at(monkeypatch, api)
    api.routes[API_PATH] = (code, "/repositories/1/pulls/7")   # relative: the same origin
    pr = fetch_pr(PR_URL, token=TOKEN)
    assert pr["diff"] == _DIFF
    assert [(p, h.get("authorization")) for p, h in api.seen] == [
        (API_PATH, f"Bearer {TOKEN}"), ("/repositories/1/pulls/7", f"Bearer {TOKEN}")] * 2
    assert elsewhere.seen == []


def test_a_chain_within_the_origin_then_off_it_is_refused_at_the_step_that_leaves(servers, monkeypatch):
    api, elsewhere = servers
    point_api_at(monkeypatch, api)
    api.routes[API_PATH] = (301, api.base + "/moved")
    api.routes["/moved"] = (307, elsewhere.base + API_PATH)
    with pytest.raises(SystemExit) as e:
        fetch_pr(PR_URL, token=TOKEN)
    assert_no_token(str(e.value))
    assert api.auths() == [f"Bearer {TOKEN}", f"Bearer {TOKEN}"]
    assert elsewhere.seen == []


@pytest.mark.parametrize("where", ["other port", "other host", "other scheme", "user part"])
def test_the_same_server_by_another_origin_is_refused(servers, monkeypatch, where):
    api, elsewhere = servers
    point_api_at(monkeypatch, api)
    target = {"other port": elsewhere.base + "/moved",
              "other host": f"http://localhost:{api.port}/moved",
              "other scheme": f"https://127.0.0.1:{api.port}/moved",
              "user part": f"http://x@127.0.0.1:{api.port}/moved"}[where]
    api.routes[API_PATH] = (302, target)
    with pytest.raises(SystemExit) as e:
        fetch_pr(PR_URL, token=TOKEN)
    assert "not followed" in str(e.value)
    assert_no_token(str(e.value))
    assert [p for p, _ in api.seen] == [API_PATH]               # nothing reached /moved on any port
    assert elsewhere.seen == []


def test_no_token_means_no_authorization_anywhere(servers, monkeypatch):
    api, elsewhere = servers
    api.routes[API_PATH] = (302, elsewhere.base + API_PATH)
    fetch_pr(PR_URL, _open=rewriting_opener(api))
    point_api_at(monkeypatch, api)
    api.routes[API_PATH] = (302, "/moved")
    fetch_pr(PR_URL)
    assert api.seen and elsewhere.seen
    assert all(a is None for a in api.auths() + elsewhere.auths())


@pytest.mark.parametrize("url, same", [
    ("https://api.github.com/repositories/1/pulls/7", True),
    ("https://API.GitHub.com/x", True),
    ("https://api.github.com:443/x", True),                     # the default port, written out
    ("http://api.github.com/x", False),
    ("https://api.github.com:8443/x", False),
    ("https://api.github.com.evil.example/x", False),
    ("https://evil.example/api.github.com/x", False),
    ("https://user@api.github.com/x", False),
    ("https://user:pw@api.github.com/x", False),
    ("https://[::1/x", False),                                  # cannot be read
    ("//api.github.com/x", False),
])
def test_the_origin_rule_against_the_real_base(url, same):
    assert dg._API_BASE == "https://api.github.com"
    assert dg._same_api_origin(url) is same


# ── the command line: the token from the environment ───────────────────────────────────────────────

@pytest.mark.parametrize("var", ["GITHUB_TOKEN", "GH_TOKEN"])
def test_cli_environment_token_reaches_the_api_and_no_text(servers, monkeypatch, capsys, var):
    api, _ = servers
    point_api_at(monkeypatch, api)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv(var, TOKEN)
    api.routes[API_PATH] = (301, "/repositories/1/pulls/7")
    rc = main(["--pr", PR_URL])
    out = capsys.readouterr()
    assert rc == 0 and "PASS" in out.out
    assert all(a == f"Bearer {TOKEN}" for a in api.auths()) and len(api.seen) == 4
    assert_no_token(out.out, out.err)


@pytest.mark.parametrize("code", CODES)
def test_cli_environment_token_is_not_sent_off_the_origin(servers, monkeypatch, capsys, code):
    api, elsewhere = servers
    point_api_at(monkeypatch, api)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    api.routes[API_PATH] = (code, elsewhere.base + API_PATH)
    with pytest.raises(SystemExit) as e:
        main(["--pr", PR_URL])
    out = capsys.readouterr()
    assert "not followed" in str(e.value)
    assert_no_token(str(e.value), out.out, out.err)
    assert elsewhere.seen == []
