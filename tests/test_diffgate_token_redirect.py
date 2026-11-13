# -*- coding: utf-8 -*-
"""fetch_pr's token is sent to the API's origin and nowhere else, whatever a redirect names.

Through 8.0.1 `fetch_pr` put ``Authorization: Bearer <token>`` in the request's ordinary headers and
sent it through `urllib.request.urlopen`, which follows 301, 302, 303 and 307 for a GET (and 308 from
Python 3.11) and copies those headers to whatever address the Location names (another host, another
port, https to http). The token now travels as an unredirected header, which urllib's own redirect
handling never copies to a redirected request, and fetch_pr's own opener follows a redirect (308
included, on every Python) only within the API's origin, putting the token back there.

Everything here runs against real HTTP servers on 127.0.0.1, because a test that replaces the client
cannot see redirect handling: `api` stands for the API and answers with redirects, `elsewhere` stands for
any other address and records every header it is sent. Two ways point fetch_pr at loopback, and each test
says which: a caller-supplied opener that rewrites the request's address to `api` before a stock urllib
opener sends it (the request fetch_pr built, headers and all, is what goes), or the module constant
`styxx.diffgate._API_BASE` monkeypatched to `api`'s address. The token is a made-up string.

The `servers` fixture refuses, for the length of each test that uses it, any connection outside
127.0.0.1 and ::1 and any name lookup other than those and `localhost`, so a regression that follows a
redirect to some other name fails here with `OffLoopback` instead of asking a resolver, under any runner.
"""
from __future__ import annotations

import http.server
import json
import socket
import threading
import urllib.request

import pytest

import styxx.diffgate as dg
from styxx.diffgate import fetch_pr, main

TOKEN = "made-up-token-0123456789abcdef"     # not a real token; checked for in every text shown
PR_URL = "https://github.com/o/r/pull/7"
API_PATH = "/repos/o/r/pulls/7"
CODES = (301, 302, 303, 307, 308)
# urllib's own handler follows 308 only from Python 3.11; fetch_pr's opener follows it on every version.
STOCK_FOLLOWS_308 = hasattr(urllib.request.HTTPRedirectHandler, "http_error_308")
STOCK_CODES = [c if c != 308 else pytest.param(308, marks=pytest.mark.skipif(
    not STOCK_FOLLOWS_308, reason="urllib before Python 3.11 does not follow 308")) for c in CODES]
_LOOPBACK_ADDR = {"127.0.0.1", "::1"}
_LOOPBACK_NAMES = _LOOPBACK_ADDR | {"localhost", None}


class OffLoopback(OSError):
    """Raised by the `servers` fixture's guard for a connection or a lookup that would leave loopback."""

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


def _guard_loopback(monkeypatch):
    real_connect, real_connect_ex, real_gai = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def host_of(address):
        host = address[0] if isinstance(address, tuple) else address
        return host.decode("ascii", "replace") if isinstance(host, bytes) else host

    def connect(self, address):
        if host_of(address) not in _LOOPBACK_ADDR:
            raise OffLoopback(f"connect to {host_of(address)!r} refused by the test's loopback guard")
        return real_connect(self, address)

    def connect_ex(self, address):
        if host_of(address) not in _LOOPBACK_ADDR:
            raise OffLoopback(f"connect to {host_of(address)!r} refused by the test's loopback guard")
        return real_connect_ex(self, address)

    def getaddrinfo(host, *a, **k):
        if host_of(host) not in _LOOPBACK_NAMES:
            raise OffLoopback(f"lookup of {host_of(host)!r} refused by the test's loopback guard")
        return real_gai(host, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.fixture
def servers(monkeypatch):
    _guard_loopback(monkeypatch)
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

@pytest.mark.parametrize("code", STOCK_CODES)
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
    assert api.auths() == [f"Bearer {TOKEN}"]


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
    assert api.auths() == [f"Bearer {TOKEN}"]
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
    ("https://api.github.com:0443/x", True),                    # the same port, a leading zero
    ("http://api.github.com/x", False),
    ("http://api.github.com:443/x", False),                     # the scheme alone differs
    ("https://api.github.com:8443/x", False),
    ("https://api.github.com.evil.example/x", False),
    ("https://evilgithub.com/x", False),                        # a host sharing a suffix with the base
    ("https://xapi.github.com/x", False),
    ("https://github.com/x", False),
    ("https://uploads.github.com/x", False),
    ("https://evil.example/api.github.com/x", False),
    ("https://user@api.github.com/x", False),
    ("https://user:pw@api.github.com/x", False),
    ("https://[::1/x", False),                                  # cannot be read
    ("//api.github.com/x", False),
])
def test_the_origin_rule_against_the_real_base(url, same):
    assert dg._API_BASE == "https://api.github.com"
    assert dg._same_api_origin(url) is same


@pytest.mark.parametrize("base, url, same", [
    ("https://127.0.0.1:8443", "https://127.0.0.1:8443/x", True),
    ("https://127.0.0.1:8443", "http://127.0.0.1:8443/x", False),   # the scheme alone differs
    ("http://localhost:8080", "http://localhost:8080/x", True),
    ("http://localhost:8080", "http://xlocalhost:8080/x", False),   # a suffix of the host
    ("http://localhost:8080", "http://localhost.:8080/x", False),
])
def test_the_origin_rule_against_other_bases(monkeypatch, base, url, same):
    """The rule compares with `_API_BASE` whatever it is; pure, no request is made."""
    monkeypatch.setattr(dg, "_API_BASE", base)
    assert dg._same_api_origin(url) is same


@pytest.mark.parametrize("code", CODES)
def test_the_handler_puts_the_token_back_as_an_unredirected_header_within_the_origin(code):
    """Called directly, no request is made: within the origin the new request carries the token as an
    unredirected header (so no later urllib redirect copies it), for 308 too on every Python; off the
    origin there is no new request."""
    assert hasattr(dg._ApiRedirect, "http_error_308")
    req = urllib.request.Request("https://api.github.com" + API_PATH, headers={"Accept": "a"})
    req.add_unredirected_header("Authorization", f"Bearer {TOKEN}")
    h = dg._ApiRedirect()
    new = h.redirect_request(req, None, code, "", {}, "https://api.github.com/repositories/1/pulls/7")
    assert new.full_url == "https://api.github.com/repositories/1/pulls/7"
    assert new.unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"
    assert "Authorization" not in new.headers and new.headers.get("Accept") == "a"
    assert h.redirect_request(req, None, code, "", {}, "https://evilgithub.com" + API_PATH) is None


def test_a_redirect_loop_within_the_origin_is_named_as_one(servers, monkeypatch):
    """urllib gives up on a loop with an HTTPError carrying the 3xx code; the sentence says the target
    was within the origin and why it stopped, not that only such a redirect is followed."""
    api, elsewhere = servers
    point_api_at(monkeypatch, api)
    api.routes[API_PATH] = (302, "/loop")
    api.routes["/loop"] = (302, "/loop")
    with pytest.raises(SystemExit) as e:
        fetch_pr(PR_URL, token=TOKEN)
    msg = str(e.value)
    assert f"a redirect to '/loop', within {api.base}, which was not followed" in msg
    assert "loop or are more than urllib follows" in msg and "only a redirect within" not in msg
    assert_no_token(msg, repr(e.value))
    assert all(a == f"Bearer {TOKEN}" for a in api.auths()) and len(api.seen) > 2
    assert elsewhere.seen == []


def test_a_caller_opener_that_does_not_follow_is_named_as_the_one_that_stopped(servers):
    api, elsewhere = servers
    api.routes[API_PATH] = (302, elsewhere.base + API_PATH)
    stock = urllib.request.build_opener(dg._ApiRedirect)       # follows nothing off the real base
    with pytest.raises(SystemExit) as e:
        fetch_pr(PR_URL, token=TOKEN, _open=rewriting_opener(api, stock))
    assert "which the opener passed as _open did not follow" in str(e.value)
    assert_no_token(str(e.value))
    assert api.auths() == [f"Bearer {TOKEN}"]
    assert elsewhere.seen == []


def test_the_fixture_refuses_to_leave_loopback(servers):
    """The guard the `servers` fixture installs: a lookup of any other name and a connection to any other
    address raise OffLoopback before anything is sent (192.0.2.1 is a documentation address)."""
    with pytest.raises(OffLoopback):
        socket.getaddrinfo("x@127.0.0.1", 80)
    with pytest.raises(OffLoopback):
        socket.getaddrinfo("example.invalid", 80)
    s = socket.socket()
    try:
        with pytest.raises(OffLoopback):
            s.connect(("192.0.2.1", 80))
    finally:
        s.close()


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
    assert api.auths() == [f"Bearer {TOKEN}"]
    assert elsewhere.seen == []
