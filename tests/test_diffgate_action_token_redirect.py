# -*- coding: utf-8 -*-
"""The Action's workflow token is sent to the API's origin and nowhere else, whatever a redirect names.

`diffgate_action.py`, the script `action.yml` runs with `GH_TOKEN` filled from `github.token`, put
``Authorization: Bearer <token>`` among the request's ordinary headers and sent it through
`urllib.request.urlopen`, which follows 301, 302, 303 and 307 for a GET (and 308 from Python 3.11) and
copies those headers to whatever address the Location names (another host, another port, https to
http). 8.0.2 repaired that pattern in `styxx.diffgate.fetch_pr` and left this script as it was. The
token now travels as an unredirected header, which urllib's own redirect handling never copies to a
redirected request; the script's own opener follows a redirect (308 included, on every Python) only
within the API's origin and puts the token back there; and an address off that origin is not asked.

The origin is the one the runner names in `GITHUB_API_URL` (https://api.github.com on github.com), so
these tests point the script at loopback by setting that variable, which is how a GitHub Enterprise
Server points it at itself. Everything that sends a request runs against real HTTP servers on 127.0.0.1,
because a test that replaces the client cannot see redirect handling: `api` stands for the API and
answers with redirects, `elsewhere` stands for any other address and records every header it is sent.
The token is a made-up string.

The `servers` fixture refuses, for the length of each test that uses it, any connection outside
127.0.0.1 and ::1 and any name lookup other than those and `localhost`, so a regression that follows a
redirect to some other name fails here with `OffLoopback` instead of asking a resolver, under any runner.
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import socket
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

import styxx.diffgate as dg

ACTION = Path(__file__).resolve().parent.parent / "diffgate_action.py"

TOKEN = "made-up-token-0123456789abcdef"     # not a real token; checked for in every text shown
API_PATH = "/repos/o/r/pulls/7"
MOVED = "/repositories/1/pulls/7"
DIFF_ACCEPT = "application/vnd.github.diff"
CODES = (301, 302, 303, 307, 308)
SUMMARY = "Modified src/retry.py."
_LOOPBACK_ADDR = {"127.0.0.1", "::1"}
_LOOPBACK_NAMES = _LOOPBACK_ADDR | {"localhost", None}

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


class OffLoopback(OSError):
    """Raised by the `servers` fixture's guard for a connection or a lookup that would leave loopback."""


class Server:
    """A real HTTP server on 127.0.0.1. `routes` maps a path to (code, Location) to answer with a
    redirect; any other path is served the diff. Every request is recorded as (path, {header: value})."""

    def __init__(self):
        self.routes: dict = {}
        self.seen: list = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
                if self.path in outer.routes:
                    code, loc = outer.routes[self.path]
                    self.send_response(code)
                    if loc is not None:
                        self.send_header("Location", loc)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = _DIFF.encode("utf-8")
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

    def paths(self) -> list:
        return [p for p, _ in self.seen]

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


def load_action():
    spec = importlib.util.spec_from_file_location("diffgate_action_token", ACTION)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diffgate_action_token"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def act():
    return load_action()


@pytest.fixture
def servers(monkeypatch):
    """Two servers on loopback, the guard, no proxy, and the script pointed at `api` the way a runner
    points it: GITHUB_API_URL names `api`, GH_TOKEN holds the token."""
    _guard_loopback(monkeypatch)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
              "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})
    api, elsewhere = Server(), Server()
    monkeypatch.setenv("GITHUB_API_URL", api.base)
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    yield api, elsewhere
    api.close()
    elsewhere.close()


def assert_no_token(*texts):
    for t in texts:
        assert TOKEN not in str(t)


# ── the token goes to the API's origin ─────────────────────────────────────────────────────────────

def test_the_api_gets_the_token_and_serves_the_diff(servers, act):
    api, elsewhere = servers
    assert act.api(api.base + API_PATH, DIFF_ACCEPT) == _DIFF
    assert api.auths() == [f"Bearer {TOKEN}"]
    assert api.seen[0][1].get("accept") == DIFF_ACCEPT
    assert api.seen[0][1].get("user-agent") == "styxx-diffgate-action"
    assert api.seen[0][1].get("x-github-api-version") == "2022-11-28"
    assert elsewhere.seen == []


# ── a redirect: refused off the origin, followed with the token within it ──────────────────────────

@pytest.mark.parametrize("code", CODES)
def test_a_redirect_off_the_origin_is_refused_and_the_other_address_gets_nothing(servers, act, code):
    api, elsewhere = servers
    api.routes[API_PATH] = (code, elsewhere.base + API_PATH)
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    msg = str(e.value)
    assert f"GitHub answers {code}, a redirect to" in msg and "which was not followed" in msg
    assert repr(elsewhere.base + API_PATH) in msg
    assert f"only a redirect within {api.base} is" in msg
    assert_no_token(msg, repr(e.value), e.value.__cause__, repr(e.value.__cause__))
    assert api.auths() == [f"Bearer {TOKEN}"]                   # asked once, with the token; then refused
    assert elsewhere.seen == []


@pytest.mark.parametrize("code", CODES)
def test_a_redirect_within_the_origin_keeps_the_token(servers, act, code):
    api, elsewhere = servers
    api.routes[API_PATH] = (code, MOVED)                        # relative: the same origin
    assert act.api(api.base + API_PATH, DIFF_ACCEPT) == _DIFF
    assert [(p, h.get("authorization")) for p, h in api.seen] == [
        (API_PATH, f"Bearer {TOKEN}"), (MOVED, f"Bearer {TOKEN}")]
    assert elsewhere.seen == []


@pytest.mark.parametrize("code", CODES)
def test_an_absolute_redirect_within_the_origin_keeps_the_token(servers, act, code):
    api, elsewhere = servers
    api.routes[API_PATH] = (code, api.base + MOVED)
    assert act.api(api.base + API_PATH, DIFF_ACCEPT) == _DIFF
    assert api.auths() == [f"Bearer {TOKEN}"] * 2 and api.paths() == [API_PATH, MOVED]
    assert elsewhere.seen == []


def test_a_chain_within_the_origin_then_off_it_is_refused_at_the_step_that_leaves(servers, act):
    api, elsewhere = servers
    api.routes[API_PATH] = (301, api.base + "/moved")
    api.routes["/moved"] = (307, elsewhere.base + API_PATH)
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert "GitHub answers 307" in str(e.value) and "which was not followed" in str(e.value)
    assert_no_token(str(e.value))
    assert api.auths() == [f"Bearer {TOKEN}", f"Bearer {TOKEN}"]
    assert elsewhere.seen == []


@pytest.mark.parametrize("where", ["other port", "other host", "other scheme", "user part"])
def test_the_same_server_by_another_origin_is_refused(servers, act, where):
    api, elsewhere = servers
    target = {"other port": elsewhere.base + "/moved",
              "other host": f"http://localhost:{api.port}/moved",
              "other scheme": f"https://127.0.0.1:{api.port}/moved",
              "user part": f"http://x@127.0.0.1:{api.port}/moved"}[where]
    api.routes[API_PATH] = (302, target)
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert "which was not followed" in str(e.value)
    assert_no_token(str(e.value))
    assert api.paths() == [API_PATH]                            # nothing reached /moved on any port
    assert api.auths() == [f"Bearer {TOKEN}"]
    assert elsewhere.seen == []


def test_a_redirect_loop_within_the_origin_is_named_as_one(servers, act):
    """urllib gives up on a loop with an HTTPError carrying the 3xx code; the sentence says the target
    was within the origin and why it stopped, not that only such a redirect is followed."""
    api, elsewhere = servers
    api.routes[API_PATH] = (302, "/loop")
    api.routes["/loop"] = (302, "/loop")
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    msg = str(e.value)
    assert f"a redirect to '/loop', within {api.base}, which was not followed" in msg
    assert "loop or are more than urllib follows" in msg and "only a redirect within" not in msg
    assert_no_token(msg, repr(e.value))
    assert all(a == f"Bearer {TOKEN}" for a in api.auths()) and len(api.seen) > 2
    assert elsewhere.seen == []


def test_a_redirect_that_names_no_address_is_refused_as_one(servers, act):
    api, elsewhere = servers
    api.routes[API_PATH] = (302, None)
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert "GitHub answers 302, a redirect to (none given), which was not followed" in str(e.value)
    assert api.auths() == [f"Bearer {TOKEN}"] and elsewhere.seen == []


def test_the_address_in_a_refusal_is_cut_at_200_characters_and_quoted(servers, act):
    api, elsewhere = servers
    long = elsewhere.base + "/" + "a" * 400 + "\r\n::error::x"
    api.routes[API_PATH] = (302, long.replace("\r\n", "%0D%0A"))
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    msg = str(e.value)
    assert repr(long.replace("\r\n", "%0D%0A")[:200]) in msg
    assert "a" * 201 not in msg and "\n" not in msg and "\r" not in msg
    assert elsewhere.seen == []


def test_an_answer_that_is_not_a_redirect_is_raised_as_urllib_raises_it(servers, act):
    api, _ = servers
    api.routes[API_PATH] = (404, None)
    with pytest.raises(urllib.error.HTTPError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert e.value.code == 404
    assert_no_token(str(e.value), repr(e.value))


# ── the first address: one off the origin is not asked at all ──────────────────────────────────────

@pytest.mark.parametrize("where", ["other port", "other host", "other scheme", "user part"])
def test_an_address_off_the_origin_is_not_asked(servers, act, where):
    api, elsewhere = servers
    url = {"other port": elsewhere.base + API_PATH,
           "other host": f"http://localhost:{api.port}{API_PATH}",
           "other scheme": f"https://127.0.0.1:{api.port}{API_PATH}",
           "user part": f"http://x@127.0.0.1:{api.port}{API_PATH}"}[where]
    with pytest.raises(RuntimeError) as e:
        act.api(url, DIFF_ACCEPT)
    msg = str(e.value)
    assert f"the diff's address {url!r} is not on {api.base}" in msg and "was not asked for" in msg
    assert_no_token(msg, repr(e.value))
    assert api.seen == [] and elsewhere.seen == []


@pytest.mark.parametrize("url", ["", "not an address", "/repos/o/r/pulls/7", "//127.0.0.1/x",
                                 "file:///etc/hostname", "http://[::1/x", None, 7])
def test_a_value_that_is_not_an_address_on_the_origin_is_not_asked(servers, act, url):
    api, elsewhere = servers
    with pytest.raises(RuntimeError) as e:
        act.api(url, DIFF_ACCEPT)
    assert "the diff's address" in str(e.value) and "was not asked for" in str(e.value)
    assert_no_token(str(e.value))
    assert api.seen == [] and elsewhere.seen == []


def test_a_long_address_off_the_origin_is_cut_at_200_characters(servers, act):
    api, elsewhere = servers
    url = elsewhere.base + "/" + "a" * 400
    with pytest.raises(RuntimeError) as e:
        act.api(url, DIFF_ACCEPT)
    assert repr(url[:200]) in str(e.value) and "a" * 201 not in str(e.value)
    assert api.seen == [] and elsewhere.seen == []


# ── which origin: the one the runner names ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, "", "   "])
def test_without_github_api_url_the_origin_is_api_github_com(monkeypatch, act, value):
    if value is None:
        monkeypatch.delenv("GITHUB_API_URL", raising=False)
    else:
        monkeypatch.setenv("GITHUB_API_URL", value)
    assert act._api_origin() == ("https", "api.github.com", 443)


@pytest.mark.parametrize("value, origin", [
    ("https://api.github.com", ("https", "api.github.com", 443)),
    ("https://API.GitHub.com/", ("https", "api.github.com", 443)),
    ("https://ghe.example/api/v3", ("https", "ghe.example", 443)),
    ("https://api.tenant.ghe.com", ("https", "api.tenant.ghe.com", 443)),
    ("http://ghe.example:8080/api/v3", ("http", "ghe.example", 8080)),
    ("  https://ghe.example/api/v3\n", ("https", "ghe.example", 443)),
])
def test_github_api_url_names_the_origin(monkeypatch, act, value, origin):
    monkeypatch.setenv("GITHUB_API_URL", value)
    assert act._api_origin() == origin


@pytest.mark.parametrize("value", ["api.github.com", "not an address", "ftp://ghe.example/api/v3",
                                   "file:///etc/hostname", "https://user@ghe.example/api/v3",
                                   "https://[::1/x", "//ghe.example/api/v3", "https://"])
def test_a_github_api_url_that_cannot_be_read_names_no_origin_and_nothing_is_asked(servers, act,
                                                                                    monkeypatch, value):
    api, elsewhere = servers
    monkeypatch.setenv("GITHUB_API_URL", value)
    assert act._api_origin() is None
    for url in (api.base + API_PATH, "https://api.github.com" + API_PATH, value):
        with pytest.raises(RuntimeError) as e:
            act.api(url, DIFF_ACCEPT)
        msg = str(e.value)
        assert f"GITHUB_API_URL is {value!r}, which is not an http or https address" in msg
        assert "the workflow token is sent nowhere" in msg
        assert_no_token(msg)
    assert api.seen == [] and elsewhere.seen == []


def test_without_github_api_url_a_loopback_address_is_not_asked(servers, act, monkeypatch):
    """The default origin is https://api.github.com, so with the variable unset the token is not sent
    to the server that held it a moment ago."""
    api, elsewhere = servers
    monkeypatch.delenv("GITHUB_API_URL")
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert "is not on https://api.github.com, the one origin" in str(e.value)
    assert api.seen == [] and elsewhere.seen == []


def test_an_enterprise_server_gets_the_token_under_its_own_path_and_no_other_origin_does(servers, act,
                                                                                         monkeypatch):
    """GITHUB_API_URL on a GitHub Enterprise Server carries a path (/api/v3). The rule is the origin,
    so any path on that server is asked with the token and a redirect off the server is refused."""
    api, elsewhere = servers
    monkeypatch.setenv("GITHUB_API_URL", api.base + "/api/v3")
    api.routes["/api/v3" + API_PATH] = (301, "/api/v3" + MOVED)
    assert act.api(api.base + "/api/v3" + API_PATH, DIFF_ACCEPT) == _DIFF
    assert api.auths() == [f"Bearer {TOKEN}"] * 2
    api.routes["/api/v3" + MOVED] = (302, elsewhere.base + API_PATH)
    with pytest.raises(RuntimeError):
        act.api(api.base + "/api/v3" + API_PATH, DIFF_ACCEPT)
    assert elsewhere.seen == []


# ── the origin rule and the handler, as pure functions ─────────────────────────────────────────────

_ORIGIN_TABLE = [
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
    ("https://api.github.com.:443/x", False),                   # a trailing dot is another name here
    ("https://evil.example\\@api.github.com/x", False),
    ("https://api.github.com\\@evil.example/x", False),
    ("https://[::1/x", False),                                  # cannot be read
    ("//api.github.com/x", False),
    ("api.github.com/x", False),
    ("", False),
]


@pytest.mark.parametrize("url, same", _ORIGIN_TABLE)
def test_the_origin_rule_against_api_github_com(monkeypatch, act, url, same):
    monkeypatch.delenv("GITHUB_API_URL", raising=False)
    assert (act._origin(url) == act._api_origin()) is same


@pytest.mark.parametrize("url", [u for u, _ in _ORIGIN_TABLE] + [
    "http://127.0.0.1:8080/x", "http://localhost.:8080/x", "https://[::1]:8443/x", "HTTPS://Host/x",
    "http://host:65536/x", "http://host:-1/x", "http://host:port/x"])
def test_the_script_and_the_package_read_an_origin_the_same_way(act, url):
    """The script writes the reading out instead of importing a private name; this holds the two copies
    to one answer on every address above."""
    assert act._origin(url) == dg._origin(url)


@pytest.mark.parametrize("value", [None, 7, b"https://api.github.com/x"])
def test_a_value_that_is_not_a_string_has_no_origin(act, value):
    assert act._origin(value) is None


@pytest.mark.parametrize("origin, text", [
    (("https", "api.github.com", 443), "https://api.github.com"),
    (("http", "ghe.example", 80), "http://ghe.example"),
    (("http", "127.0.0.1", 8080), "http://127.0.0.1:8080"),
    (("https", "ghe.example", 8443), "https://ghe.example:8443"),
    (("https", "::1", 8443), "https://[::1]:8443"),
])
def test_an_origin_is_written_as_the_address_it_is(act, origin, text):
    assert act._origin_text(origin) == text
    assert act._origin(text + "/x") == origin


@pytest.mark.parametrize("code", CODES)
def test_the_handler_puts_the_token_back_as_an_unredirected_header_within_the_origin(act, code):
    """Called directly, no request is made: within the origin the new request carries the token as an
    unredirected header (so no later urllib redirect copies it), for 308 too on every Python; off the
    origin, or to an address that cannot be read, there is no new request."""
    assert hasattr(act._ApiRedirect, "http_error_308")
    req = urllib.request.Request("https://api.github.com" + API_PATH, headers={"Accept": "a"})
    req.add_unredirected_header("Authorization", f"Bearer {TOKEN}")
    h = act._ApiRedirect(("https", "api.github.com", 443))
    new = h.redirect_request(req, None, code, "", {}, "https://api.github.com" + MOVED)
    assert new.full_url == "https://api.github.com" + MOVED
    assert new.unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"
    assert "Authorization" not in new.headers and new.headers.get("Accept") == "a"
    assert h.redirect_request(req, None, code, "", {}, "https://evilgithub.com" + API_PATH) is None
    assert h.redirect_request(req, None, code, "", {}, "http://api.github.com" + API_PATH) is None
    assert h.redirect_request(req, None, code, "", {}, "https://[::1/x") is None


def test_a_handler_with_no_origin_follows_nothing(act):
    """`api` never builds one, since it refuses first; the handler alone does not treat two unreadable
    addresses as the same origin."""
    req = urllib.request.Request("https://api.github.com" + API_PATH)
    assert act._ApiRedirect(None).redirect_request(req, None, 302, "", {}, "https://[::1/x") is None


def test_the_request_the_script_builds_keeps_the_token_out_of_the_ordinary_headers(servers, act,
                                                                                   monkeypatch):
    """What urllib copies on a redirect is `Request.headers`; the token is not there."""
    api, _ = servers
    built = []
    real = urllib.request.OpenerDirector.open

    def spy(self, req, *a, **k):
        if not built:
            built.append(req)
        return real(self, req, *a, **k)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", spy)
    act.api(api.base + API_PATH, DIFF_ACCEPT)
    req = built[0]
    assert req.unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"
    assert all(k.lower() != "authorization" for k in req.headers)
    assert_no_token(req.headers)


def test_an_installed_opener_is_not_used(servers, act):
    api, elsewhere = servers
    api.routes[API_PATH] = (302, elsewhere.base + API_PATH)
    used = []

    class Spy(urllib.request.BaseHandler):
        def http_request(self, req):
            used.append(req.full_url)
            return req
    urllib.request.install_opener(urllib.request.build_opener(Spy))
    try:
        with pytest.raises(RuntimeError):
            act.api(api.base + API_PATH, DIFF_ACCEPT)
    finally:
        urllib.request.install_opener(None)
    assert used == [] and elsewhere.seen == []
    assert api.auths() == [f"Bearer {TOKEN}"]


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


# ── through main(), as the Action runs it ──────────────────────────────────────────────────────────

def _event(tmp_path, monkeypatch, name, payload, *, strict="false", soft=None):
    ev = tmp_path / "event.json"
    ev.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(ev))
    monkeypatch.setenv("GITHUB_EVENT_NAME", name)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))
    monkeypatch.setenv("STYXX_STRICT", strict)
    if soft is None:
        monkeypatch.delenv("STYXX_SOFT_FAIL", raising=False)
    else:
        monkeypatch.setenv("STYXX_SOFT_FAIL", soft)
    return tmp_path / "sum.md"


def _pr_event(url):
    return {"pull_request": {"number": 7, "body": SUMMARY, "url": url}}


def test_main_gates_a_pull_request_and_the_token_reaches_only_the_api(servers, act, tmp_path, monkeypatch,
                                                                      capsys):
    api, elsewhere = servers
    api.routes[API_PATH] = (301, MOVED)
    summary = _event(tmp_path, monkeypatch, "pull_request", _pr_event(api.base + API_PATH))
    assert act.main() == 0
    out = capsys.readouterr()
    assert "styxx diffgate: PASS" in out.out
    assert "the summary vs the diff" in summary.read_text(encoding="utf-8")
    assert api.auths() == [f"Bearer {TOKEN}"] * 2 and elsewhere.seen == []
    assert_no_token(out.out, out.err, summary.read_text(encoding="utf-8"))


@pytest.mark.parametrize("code", CODES)
def test_main_reports_a_refused_redirect_as_a_gate_that_did_not_run(servers, act, tmp_path, monkeypatch,
                                                                    capsys, code):
    api, elsewhere = servers
    api.routes[API_PATH] = (code, elsewhere.base + API_PATH)
    summary = _event(tmp_path, monkeypatch, "pull_request", _pr_event(api.base + API_PATH))
    assert act.main() == 0                                      # the default reports; it is not a pass
    out = capsys.readouterr()
    text = summary.read_text(encoding="utf-8")
    assert "could not fetch the diff" in out.out and "The gate did not run; this is not a pass." in out.out
    assert f"GitHub answers {code}, a redirect to" in out.out and "which was not followed" in out.out
    assert "DID NOT RUN" in text and "which was not followed" in text
    assert api.auths() == [f"Bearer {TOKEN}"] and elsewhere.seen == []
    assert_no_token(out.out, out.err, text)


def test_main_fails_the_job_on_a_refused_redirect_under_strict_with_soft_fail_off(servers, act, tmp_path,
                                                                                  monkeypatch, capsys):
    api, elsewhere = servers
    api.routes[API_PATH] = (302, elsewhere.base + API_PATH)
    _event(tmp_path, monkeypatch, "pull_request", _pr_event(api.base + API_PATH), strict="true", soft="false")
    assert act.main() == 1
    assert elsewhere.seen == []
    assert_no_token(*capsys.readouterr())


def test_main_does_not_ask_a_pull_request_address_off_the_origin(servers, act, tmp_path, monkeypatch, capsys):
    api, elsewhere = servers
    summary = _event(tmp_path, monkeypatch, "pull_request", _pr_event(elsewhere.base + API_PATH))
    assert act.main() == 0
    out = capsys.readouterr()
    assert "is not on " + api.base in out.out and "was not asked for" in out.out
    assert "DID NOT RUN" in summary.read_text(encoding="utf-8")
    assert api.seen == [] and elsewhere.seen == []
    assert_no_token(out.out, out.err, summary.read_text(encoding="utf-8"))


def _push_event(template, compare="https://github.com/o/r/compare/aaa...bbb"):
    return {"head_commit": {"message": SUMMARY}, "before": "aaa", "after": "bbb", "compare": compare,
            "repository": {"compare_url": template}}


def test_main_gates_a_push_and_the_token_reaches_only_the_api(servers, act, tmp_path, monkeypatch, capsys):
    api, elsewhere = servers
    tpl = api.base + "/repos/o/r/compare/{base}...{head}"
    _event(tmp_path, monkeypatch, "push", _push_event(tpl))
    assert act.main() == 0
    out = capsys.readouterr()
    assert "styxx diffgate: PASS" in out.out
    assert api.paths() == ["/repos/o/r/compare/aaa...bbb"] and api.auths() == [f"Bearer {TOKEN}"]
    assert elsewhere.seen == []
    assert_no_token(out.out, out.err)


def test_main_does_not_ask_a_push_address_off_the_origin(servers, act, tmp_path, monkeypatch, capsys):
    api, elsewhere = servers
    tpl = elsewhere.base + "/repos/o/r/compare/{base}...{head}"
    _event(tmp_path, monkeypatch, "push", _push_event(tpl))
    assert act.main() == 0
    out = capsys.readouterr()
    assert "was not asked for" in out.out and "The gate did not run; this is not a pass." in out.out
    assert api.seen == [] and elsewhere.seen == []
    assert_no_token(out.out, out.err)


def test_on_github_com_the_addresses_a_runner_writes_are_on_the_origin(monkeypatch, act):
    """The three addresses main() can build from a github.com event, against the default origin. Pure:
    no request is made."""
    monkeypatch.delenv("GITHUB_API_URL", raising=False)
    origin = act._api_origin()
    assert act._origin("https://api.github.com/repos/o/r/pulls/7") == origin
    assert act._origin("https://api.github.com/repos/o/r/compare/{base}...{head}"
                       .replace("{base}", "a" * 40).replace("{head}", "b" * 40)) == origin
    rewritten = "https://github.com/o/r/compare/aaa...bbb".replace("github.com", "api.github.com/repos", 1)
    assert act._origin(rewritten) == origin
    # a branch or commit id in the template's place cannot move the origin: it lands after the path's first slash
    assert act._origin("https://api.github.com/repos/o/r/compare/{base}...{head}"
                       .replace("{base}", "@evil.example/").replace("{head}", "x")) == origin
