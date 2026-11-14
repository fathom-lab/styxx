# -*- coding: utf-8 -*-
"""The Action's workflow token is sent to the API's origin and nowhere else, whatever a redirect names.

`diffgate_action.py`, the script `action.yml` runs with `GH_TOKEN` filled from `github.token`, put
``Authorization: Bearer <token>`` among the request's ordinary headers and sent it through
`urllib.request.urlopen`, which follows 301, 302, 303 and 307 for a GET (and 308 from Python 3.11) and
copies those headers to whatever address the Location names (another host, another port, https to
http). 8.0.2 repaired that pattern in `styxx.diffgate.fetch_pr` and left this script as it was. The
token now travels as an unredirected header, which urllib's own redirect handling never copies to a
redirected request; the script's own opener follows a redirect (308 included, on every Python) only
within the API's origin and puts the token back there; an address off that origin is not asked; and
every request is addressed in the script's own spelling of the origin, so urllib and a proxy are never
handed the text that was checked.

The origin is the one the runner names in `GITHUB_API_URL` (https://api.github.com on github.com), so
these tests point the script at loopback by setting that variable, which is how a GitHub Enterprise
Server points it at itself. Everything that sends a request runs against real servers on 127.0.0.1,
because a test that replaces the client cannot see redirect handling: `api` stands for the API and
answers with redirects, `elsewhere` stands for any other address and records every header it is sent.
`Server` is `http.server`; `RawServer` answers with bytes written out by the test, for the answers
`http.server` cannot send (a folded header, two Location headers, a reason phrase with a carriage
return) and to stand in for a proxy. The token is a made-up string.

Not here: an https origin on the wire (every origin that is asked is plain http on loopback; https
appears in the pure-function tests and in the CONNECT a proxy is sent), and a runner (what GitHub does
with the lines the script prints is not run).

The `servers`, `raw` and `proxied` fixtures refuse, for the length of each test that uses them, any
connection outside 127.0.0.1 and ::1 and any name lookup other than those and `localhost`, so a
regression that follows a redirect to some other name fails here with `OffLoopback` instead of asking a
resolver, under any runner.
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
        # the default poll of half a second is what a shutdown waits for; a short one keeps each teardown short
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01},
                                       daemon=True)
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


def test_the_address_in_a_refusal_is_cut_at_200_characters(servers, act):
    api, elsewhere = servers
    long = elsewhere.base + "/" + "a" * 400
    api.routes[API_PATH] = (302, long)
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    msg = str(e.value)
    assert repr(long[:200]) in msg and "a" * 201 not in msg
    assert elsewhere.seen == []


def test_an_answer_that_is_not_a_redirect_is_raised_as_urllib_raises_it(servers, act):
    api, _ = servers
    api.routes[API_PATH] = (404, None)
    with pytest.raises(urllib.error.HTTPError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert e.value.code == 404
    assert_no_token(str(e.value), repr(e.value))


# ── the address the script is given: one off the origin is not asked at all ────────────────────────

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
    if where == "user part":                                    # not an address the script reads at all
        assert f"the diff's address {url!r} is not one this script can read" in msg
    else:
        assert f"the diff's address {url!r} is not on {api.base}, the one origin" in msg
    assert "was not asked for" in msg
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
    "http://host:65536/x", "http://host:-1/x", "http://host:port/x", "https://api.github.com:0/x",
    "https://@api.github.com/x", "https://:@api.github.com/x", "https://:pw@api.github.com/x"])
def test_the_script_and_the_package_read_an_origin_the_same_way(act, url):
    """The script writes the reading out instead of importing a private name; this holds the two copies
    to one answer on every address above."""
    assert act._origin(url) == dg._origin(url)


# What the script's reading refuses and the package's does not, or not on every Python: the script allows
# one shape of host and port, so that its answer does not hang on how lenient the running urlsplit is.
_STRICTER = [
    "http://[fe80::1%25eth0]:8080/x",       # an IPv6 zone identifier
    "http://%61pi.github.com/x",            # a percent sign in the host
    "https://api.github.com:+443/x",        # Python 3.9 reads these three ports as 443, 3.11 on refuses them
    "https://api.github.com:4_43/x",
    "https://api.github.com:\u0664\u0664\u0663/x",
    "https://api.github.com /x",            # a space, a tab and a line break: urlsplit drops some of them
    "https://api.github.co\tm/x",
    "https://api.github.com:443\n/x",
    "https://api.github.com/a b",
    "https://[api.github.com].evil.example/x",   # text beside a bracketed host
    "https://evil.example.[api.github.com]/x",
    "https://\u0430pi.github.com/x",        # a Cyrillic letter in the name
]


@pytest.mark.parametrize("url", _STRICTER)
def test_the_script_reads_no_origin_where_only_a_lenient_reading_would(act, url):
    assert act._origin(url) is None


@pytest.mark.parametrize("url", [u for u, _ in _ORIGIN_TABLE] + _STRICTER)
def test_where_the_script_reads_an_origin_the_package_reads_the_same_one(act, url):
    mine = act._origin(url)
    assert mine is None or mine == dg._origin(url)


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
    """`api` never builds one, since it refuses before it gets there; the handler alone does not treat two unreadable
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
    # a branch or commit id in the template's place cannot move the origin: it lands after the slash that opens the path
    assert act._origin("https://api.github.com/repos/o/r/compare/{base}...{head}"
                       .replace("{base}", "@evil.example/").replace("{head}", "x")) == origin


# ── answers http.server cannot send, and a stand-in for a proxy ────────────────────────────────────

class RawServer:
    """A listener on 127.0.0.1 that answers each request with bytes the test wrote. `answers` maps a
    request target (the second word of the request line: a path, an absolute address when the client
    takes this for a proxy, or host:port for a CONNECT) to the whole answer; any other target is served
    the diff. Each request is recorded as (method, target, {header: value}, the bytes as received)."""

    def __init__(self):
        self.answers: dict = {}
        self.seen: list = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self._closing = False
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                if self._closing:
                    return
                conn.settimeout(5)
                raw = b""
                try:
                    while b"\r\n\r\n" not in raw:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        raw += chunk
                except OSError:
                    pass
                head = raw.split(b"\r\n\r\n", 1)[0].decode("latin-1")
                lines = head.split("\r\n")
                words = lines[0].split(" ")
                method, target = words[0], (words[1] if len(words) > 1 else "")
                headers = {}
                for line in lines[1:]:
                    k, _, v = line.partition(":")
                    headers[k.strip().lower()] = v.strip()
                self.seen.append((method, target, headers, raw))
                body = _DIFF.encode("utf-8")
                default = (b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode("ascii")
                           + b"\r\nConnection: close\r\n\r\n" + body)
                try:
                    conn.sendall(self.answers.get(target, default))
                except OSError:
                    pass

    def auths(self) -> list:
        return [h.get("authorization") for _, _, h, _ in self.seen]

    def targets(self) -> list:
        return [t for _, t, _, _ in self.seen]

    def close(self):
        self._closing = True
        try:
            socket.create_connection(("127.0.0.1", self.port), timeout=1).close()   # wakes accept()
        except OSError:
            pass
        self.sock.close()


def answer(status_line: bytes, *headers: bytes) -> bytes:
    return (status_line + b"\r\n" + b"".join(h + b"\r\n" for h in headers)
            + b"Content-Length: 0\r\nConnection: close\r\n\r\n")


def _no_proxy_in_the_environment(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
              "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def raw(monkeypatch):
    """As `servers`, with a RawServer standing for the API."""
    _guard_loopback(monkeypatch)
    _no_proxy_in_the_environment(monkeypatch)
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})
    api, elsewhere = RawServer(), Server()
    monkeypatch.setenv("GITHUB_API_URL", api.base)
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    yield api, elsewhere
    api.close()
    elsewhere.close()


@pytest.fixture
def proxied(monkeypatch):
    """The environment names a proxy for http and for https, as a runner behind one does: a RawServer on
    loopback that records what it is handed and forwards nothing. `api` and `elsewhere` are real servers
    that must hear nothing while the proxy is named."""
    _guard_loopback(monkeypatch)
    _no_proxy_in_the_environment(monkeypatch)
    proxy, api, elsewhere = RawServer(), Server(), Server()
    monkeypatch.setenv("http_proxy", proxy.base)
    monkeypatch.setenv("https_proxy", proxy.base)
    monkeypatch.setenv("GITHUB_API_URL", api.base)
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    yield proxy, api, elsewhere
    proxy.close()
    api.close()
    elsewhere.close()


# ── redirects as only a raw answer can write them ──────────────────────────────────────────────────

def test_a_location_urllib_cannot_read_is_refused_in_the_scripts_own_sentence(raw, act):
    """urllib reads the address before it asks the handler and raises a ValueError that quotes it whole;
    the script says it as it says any other refusal, cut and quoted."""
    api, elsewhere = raw
    bad = "http://[::1/" + "a" * 400
    api.answers[API_PATH] = answer(b"HTTP/1.1 302 Found", b"Location: " + bad.encode("ascii"))
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    msg = str(e.value)
    assert msg == (f"GitHub answers 302, a redirect to {bad[:200]!r}, which urllib could not read as an "
                   "address, so it was not followed")
    assert isinstance(e.value.__cause__, ValueError)
    assert api.auths() == [f"Bearer {TOKEN}"] and elsewhere.seen == []


def test_a_value_error_that_is_not_about_a_redirect_is_raised_as_it_is(servers, act, monkeypatch):
    """Only a ValueError raised while urllib read a redirect's address becomes the redirect sentence."""
    api, _ = servers

    def boom(self, req, *a, **k):
        raise ValueError("not about a redirect")
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", boom)
    with pytest.raises(ValueError, match="not about a redirect"):
        act.api(api.base + API_PATH, DIFF_ACCEPT)


def test_a_uri_header_within_the_origin_is_followed_with_the_token(raw, act):
    """urllib follows a URI header where there is no Location."""
    api, elsewhere = raw
    api.answers[API_PATH] = answer(b"HTTP/1.1 302 Found", b"URI: " + MOVED.encode("ascii"))
    assert act.api(api.base + API_PATH, DIFF_ACCEPT) == _DIFF
    assert api.targets() == [API_PATH, MOVED] and api.auths() == [f"Bearer {TOKEN}"] * 2
    assert elsewhere.seen == []


def test_a_uri_header_off_the_origin_is_refused_and_named(raw, act):
    api, elsewhere = raw
    target = elsewhere.base + API_PATH
    api.answers[API_PATH] = answer(b"HTTP/1.1 302 Found", b"URI: " + target.encode("ascii"))
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert f"GitHub answers 302, a redirect to {target!r}, which was not followed" in str(e.value)
    assert api.auths() == [f"Bearer {TOKEN}"] and elsewhere.seen == []


def test_an_empty_location_is_a_redirect_to_the_same_address_and_is_named_a_loop(raw, act):
    """urllib follows an empty Location to the address it just asked, until its loop limit. The sentence
    says what happened: followed within the origin, then stopped."""
    api, elsewhere = raw
    api.answers[API_PATH] = answer(b"HTTP/1.1 302 Found", b"Location: ")
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    msg = str(e.value)
    assert f"GitHub answers 302, a redirect to '', within {api.base}, which was not followed" in msg
    assert "loop or are more than urllib follows" in msg and "only a redirect within" not in msg
    assert len(api.seen) > 2 and all(a == f"Bearer {TOKEN}" for a in api.auths())
    assert elsewhere.seen == []


def test_a_redirect_with_no_address_after_a_followed_one_is_not_called_a_loop(raw, act):
    """What became of the last answer decides the sentence, not what became of an earlier one."""
    api, elsewhere = raw
    api.answers[API_PATH] = answer(b"HTTP/1.1 301 Moved", b"Location: " + MOVED.encode("ascii"))
    api.answers[MOVED] = answer(b"HTTP/1.1 302 Found")
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    msg = str(e.value)
    assert "GitHub answers 302, a redirect to (none given), which was not followed: only a redirect" in msg
    assert "loop" not in msg
    assert api.auths() == [f"Bearer {TOKEN}"] * 2 and elsewhere.seen == []


@pytest.mark.parametrize("scheme", ["javascript:alert(1)", "file:///etc/hostname", "gopher://127.0.0.1/x"])
def test_a_redirect_to_a_kind_of_address_urllib_does_not_follow_is_refused(raw, act, scheme):
    api, elsewhere = raw
    api.answers[API_PATH] = answer(b"HTTP/1.1 302 Found", b"Location: " + scheme.encode("ascii"))
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert f"a redirect to {scheme!r}, which was not followed: only a redirect within" in str(e.value)
    assert api.auths() == [f"Bearer {TOKEN}"] and elsewhere.seen == []


def test_of_two_location_headers_the_one_urllib_reads_decides(raw, act):
    """urllib reads the one that comes before the other, and so does the sentence."""
    api, elsewhere = raw
    out = b"Location: " + (elsewhere.base + API_PATH).encode("ascii")
    within = b"Location: " + MOVED.encode("ascii")
    api.answers[API_PATH] = answer(b"HTTP/1.1 302 Found", within, out)
    assert act.api(api.base + API_PATH, DIFF_ACCEPT) == _DIFF
    assert api.targets() == [API_PATH, MOVED] and elsewhere.seen == []
    api.seen.clear()
    api.answers[API_PATH] = answer(b"HTTP/1.1 302 Found", out, within)
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert repr(elsewhere.base + API_PATH) in str(e.value) and "which was not followed" in str(e.value)
    assert api.targets() == [API_PATH] and elsewhere.seen == []


def test_a_folded_location_keeps_its_line_break_out_of_the_sentence(raw, act):
    """http.client keeps the line break of a folded header in the value. `repr` is what keeps it out of
    the sentence, and so out of the line main() prints."""
    api, elsewhere = raw
    api.answers[API_PATH] = answer(
        b"HTTP/1.1 302 Found", b"Location: " + elsewhere.base.encode("ascii") + b"/x\r\n ::error::folded")
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    msg = str(e.value)
    assert "\r" not in msg and "\n" not in msg
    assert "/x\\r\\n ::error::folded'" in msg
    assert elsewhere.seen == []


@pytest.mark.parametrize("code", [300, 304, 305, 306])
def test_a_3xx_urllib_does_not_follow_is_raised_as_urllib_raises_it(servers, act, code):
    """300, 304, 305 and 306 are not redirects urllib follows; the script does not call them one, and a
    Location on such an answer is not asked."""
    api, elsewhere = servers
    api.routes[API_PATH] = (code, elsewhere.base + API_PATH)
    with pytest.raises(urllib.error.HTTPError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert e.value.code == code
    assert_no_token(str(e.value), repr(e.value))
    assert api.auths() == [f"Bearer {TOKEN}"] and elsewhere.seen == []


def test_a_refused_redirect_carries_urllibs_error_as_its_cause(servers, act):
    api, elsewhere = servers
    api.routes[API_PATH] = (302, elsewhere.base + API_PATH)
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    cause = e.value.__cause__
    assert isinstance(cause, urllib.error.HTTPError) and cause.code == 302
    assert_no_token(cause, repr(cause), cause.headers)
    assert elsewhere.seen == []


def test_every_request_is_made_with_the_60_second_timeout(servers, act, monkeypatch):
    api, elsewhere = servers
    api.routes[API_PATH] = (302, MOVED)
    timeouts = []
    real = urllib.request.OpenerDirector.open

    def spy(self, req, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
        timeouts.append(timeout)
        return real(self, req, data, timeout)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", spy)
    assert act.api(api.base + API_PATH, DIFF_ACCEPT) == _DIFF
    assert timeouts == [60, 60]                                 # the request and the one followed
    assert elsewhere.seen == []


# ── the address is asked in the script's own spelling ──────────────────────────────────────────────

_API = ("https", "api.github.com", 443)


@pytest.mark.parametrize("url, origin, written", [
    ("https://api.github.com/repos/o/r/pulls/7", _API, "https://api.github.com/repos/o/r/pulls/7"),
    ("https://API.GitHub.com:0443/a/b?x=1&y=%20#frag", _API, "https://api.github.com/a/b?x=1&y=%20"),
    ("HTTPS://api.github.com", _API, "https://api.github.com/"),
    ("https://api.github.com/repos/o/r/compare/aaa...bbb", _API,
     "https://api.github.com/repos/o/r/compare/aaa...bbb"),
    ("https://api.github.com/?", _API, "https://api.github.com/"),
    ("http://ghe.example:08080/api/v3/x", ("http", "ghe.example", 8080), "http://ghe.example:8080/api/v3/x"),
    ("http://GHE.example:80/x", ("http", "ghe.example", 80), "http://ghe.example/x"),
    ("https://[::1]:8443/x?q", ("https", "::1", 8443), "https://[::1]:8443/x?q"),
])
def test_an_address_on_the_origin_is_written_out_again_in_the_scripts_spelling(act, url, origin, written):
    assert act._on_origin(url, origin) == written
    assert act._origin(written) == origin                       # and what is written is on the origin


@pytest.mark.parametrize("url", [u for u, same in _ORIGIN_TABLE if not same] + _STRICTER + [None, 7])
def test_an_address_off_the_origin_is_written_nowhere(act, url):
    assert act._on_origin(url, _API) is None
    assert act._on_origin("https://api.github.com/x", None) is None


class _Asked(Exception):
    """Raised by `_spy_on_open` in place of a request, so 'asked' and 'refused' can be told apart with
    no server and nothing sent."""


def _spy_on_open(monkeypatch):
    asked = []

    def spy(self, req, *a, **k):
        asked.append(req)
        raise _Asked(getattr(req, "full_url", req))
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", spy)
    return asked


@pytest.mark.parametrize("url, same", _ORIGIN_TABLE + [(u, False) for u in _STRICTER])
def test_api_asks_an_address_of_the_table_only_where_it_is_on_the_origin(monkeypatch, act, url, same):
    """The table is held above against `_origin` and `_api_origin`, compared by the test itself. This
    holds it against the check `api` makes before it asks: a check weakened to a text prefix, or to a
    host that only starts or ends with the API's, passes every loopback test. No request is made: the
    opener's `open` is replaced."""
    _guard_loopback(monkeypatch)
    monkeypatch.delenv("GITHUB_API_URL", raising=False)
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    asked = _spy_on_open(monkeypatch)
    if same:
        with pytest.raises(_Asked):
            act.api(url, DIFF_ACCEPT)
        assert len(asked) == 1 and asked[0].full_url.startswith("https://api.github.com/")
        assert asked[0].full_url == act._on_origin(url, _API)
        assert asked[0].unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"
    else:
        with pytest.raises(RuntimeError) as e:
            act.api(url, DIFF_ACCEPT)
        if act._origin(url) is None:
            assert "the diff's address" in str(e.value) and "is not one this script can read" in str(e.value)
        else:
            assert f"the diff's address {url!r} is not on https://api.github.com, the one origin" in str(e.value)
        assert "was not asked for" in str(e.value)
        assert asked == []
        assert_no_token(str(e.value))


@pytest.mark.parametrize("url, same", _ORIGIN_TABLE + [(u, False) for u in _STRICTER])
def test_the_handler_follows_an_address_of_the_table_only_where_it_is_on_the_origin(act, url, same):
    """The same table against the handler's own check, called directly: no request is made."""
    req = urllib.request.Request("https://api.github.com" + API_PATH)
    req.add_unredirected_header("Authorization", f"Bearer {TOKEN}")
    h = act._ApiRedirect(_API)
    new = h.redirect_request(req, None, 302, "", {}, url)
    if same:
        assert new is not None and new.full_url == act._on_origin(url, _API)
        assert new.full_url.startswith("https://api.github.com/")
        assert new.unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"
        assert h.verdict == "followed"
    else:
        assert new is None and h.verdict == "refused"


def test_the_handler_follows_only_to_the_origin_it_was_given(act, monkeypatch):
    """Not to the origin of the request it is redirecting (a request that is off the origin is not
    followed to its own origin), and not to an origin it reads from the environment for itself."""
    monkeypatch.setenv("GITHUB_API_URL", "https://other.example")
    h = act._ApiRedirect(_API)
    stray = urllib.request.Request("https://other.example/a")
    stray.add_unredirected_header("Authorization", f"Bearer {TOKEN}")
    assert h.redirect_request(stray, None, 302, "", {}, "https://other.example/b") is None
    on = urllib.request.Request("https://api.github.com/a")
    on.add_unredirected_header("Authorization", f"Bearer {TOKEN}")
    assert h.redirect_request(on, None, 302, "", {}, "https://other.example/b") is None
    new = h.redirect_request(on, None, 302, "", {}, "https://api.github.com/b")
    assert new is not None and new.full_url == "https://api.github.com/b"
    assert new.unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"


def test_the_handler_adds_no_authorization_where_the_request_had_none(act):
    req = urllib.request.Request("https://api.github.com" + API_PATH)
    new = act._ApiRedirect(_API).redirect_request(req, None, 302, "", {}, "https://api.github.com" + MOVED)
    assert new is not None and "Authorization" not in new.unredirected_hdrs
    assert all(k.lower() != "authorization" for k in new.headers)


def _oddly_written(api):
    """Addresses that name the API server's host and port in a spelling the script does not read."""
    port = str(api.port)
    return [f"http://127.0.0.1:{port[:-1]}_{port[-1]}{API_PATH}",       # Python 3.9 reads these two as the port
            f"http://127.0.0.1:+{port}{API_PATH}",
            f"http://127.0.0.1:{port}\t{API_PATH}",                     # urlsplit drops a tab and a line break
            f"http://127.0.0.\t1:{port}{API_PATH}",
            f"http://127.0.0.1:{port}\r\n{API_PATH}",
            f"http://127.0.0.1:{port}{API_PATH} ",
            f" http://127.0.0.1:{port}{API_PATH}",
            f"http://127.0.0.1:{port}{API_PATH}?a b",
            f"http://[127.0.0.1].evil.example:{port}{API_PATH}",
            f"http://evil.example.[127.0.0.1]:{port}{API_PATH}",
            f"http://127.0.0.1%2eevil.example:{port}{API_PATH}"]


@pytest.mark.parametrize("n", range(11))
def test_an_oddly_written_address_is_not_asked(servers, act, n):
    api, elsewhere = servers
    url = _oddly_written(api)[n]
    with pytest.raises(RuntimeError) as e:
        act.api(url, DIFF_ACCEPT)
    assert "the diff's address" in str(e.value) and "was not asked for" in str(e.value)
    assert "\r" not in str(e.value) and "\n" not in str(e.value) and "\t" not in str(e.value)
    assert api.seen == [] and elsewhere.seen == []


def test_a_port_with_leading_zeros_is_the_port_and_is_asked_in_the_scripts_spelling(servers, act):
    api, elsewhere = servers
    assert act.api(f"http://127.0.0.1:00{api.port}{API_PATH}?x=1", DIFF_ACCEPT) == _DIFF
    assert api.paths() == [API_PATH + "?x=1"] and api.auths() == [f"Bearer {TOKEN}"]
    assert api.seen[0][1].get("host") == f"127.0.0.1:{api.port}"
    assert elsewhere.seen == []


def test_a_value_that_is_not_a_string_is_shown_cut_too(servers, act):
    api, elsewhere = servers
    with pytest.raises(RuntimeError) as e:
        act.api(["x" * 50] * 1000, DIFF_ACCEPT)
    assert len(str(e.value)) < 400 and "was not asked for" in str(e.value)
    assert api.seen == [] and elsewhere.seen == []


@pytest.mark.parametrize("origin, text", [
    (("http", "ghe.example", 443), "http://ghe.example:443"),       # the other scheme's default port is written
    (("https", "ghe.example", 80), "https://ghe.example:80"),
    (("https", "::1", 443), "https://[::1]"),                       # an IPv6 literal on the default port
    (("http", "::1", 80), "http://[::1]"),
])
def test_an_origin_on_the_other_schemes_default_port_is_written_with_that_port(act, origin, text):
    assert act._origin_text(origin) == text
    assert act._origin(text + "/x") == origin


def test_a_long_github_api_url_that_cannot_be_read_is_cut_at_200_characters(servers, act, monkeypatch):
    api, elsewhere = servers
    value = "ftp://ghe.example/" + "a" * 400
    monkeypatch.setenv("GITHUB_API_URL", value)
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    msg = str(e.value)
    assert f"GITHUB_API_URL is {value[:200]!r}, which is not an http or https address" in msg
    assert "a" * 201 not in msg
    assert api.seen == [] and elsewhere.seen == []


def test_whitespace_of_any_kind_around_github_api_url_is_stripped(monkeypatch, act):
    """A no-break space and an em space, which urlsplit does not drop on any Python."""
    monkeypatch.setenv("GITHUB_API_URL", "  https://ghe.example/api/v3  ")
    assert act._api_origin() == ("https", "ghe.example", 443)


@pytest.mark.parametrize("value", ["https://ghe.example/api v3", "https://ghe.exa\tmple", "https://ghe.example:+443",
                                   "https://ghe.example:4_43", "https://[ghe.example].evil.example",
                                   "https://ghe%2eexample"])
def test_a_github_api_url_in_a_spelling_the_script_does_not_read_names_no_origin(monkeypatch, act, value):
    monkeypatch.setenv("GITHUB_API_URL", value)
    assert act._api_origin() is None


# ── a proxy the environment names ──────────────────────────────────────────────────────────────────

def test_a_proxy_in_front_of_a_plain_http_origin_is_handed_the_request_in_the_scripts_spelling(proxied, act):
    """urllib hands a plain-http request to the proxy whole, token included; that is what a proxy is, and
    it was so before. What the proxy is handed is the script's own text for the address."""
    proxy, api, elsewhere = proxied
    assert act.api(f"http://127.0.0.1:00{api.port}{API_PATH}", DIFF_ACCEPT) == _DIFF
    assert [(m, t) for m, t, _, _ in proxy.seen] == [("GET", api.base + API_PATH)]
    assert proxy.auths() == [f"Bearer {TOKEN}"]
    assert api.seen == [] and elsewhere.seen == []


@pytest.mark.parametrize("n", range(11))
def test_behind_a_proxy_an_oddly_written_address_is_handed_to_nobody(proxied, act, n):
    """On Python 3.9 urlsplit reads a port written 5008_0 or +50080 as a number, and a proxy handed that
    text may read another. Such an address is refused before anything is sent, on every Python."""
    proxy, api, elsewhere = proxied
    with pytest.raises(RuntimeError) as e:
        act.api(_oddly_written(api)[n], DIFF_ACCEPT)
    assert "was not asked for" in str(e.value)
    assert proxy.seen == [] and api.seen == [] and elsewhere.seen == []


def test_behind_a_proxy_a_redirect_off_the_origin_is_still_refused(proxied, act):
    proxy, api, elsewhere = proxied
    proxy.answers[api.base + API_PATH] = answer(
        b"HTTP/1.1 302 Found", b"Location: " + (elsewhere.base + API_PATH).encode("ascii"))
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert "which was not followed" in str(e.value)
    assert proxy.targets() == [api.base + API_PATH]             # the proxy was handed one request, and no second
    assert api.seen == [] and elsewhere.seen == []


def test_a_proxy_in_front_of_an_https_origin_sees_the_host_and_port_and_no_token(proxied, act, monkeypatch):
    """For https urllib asks the proxy for a tunnel and sends the request inside it. The proxy here
    refuses the tunnel, so nothing follows; what it was handed is the origin's host and port in the
    script's spelling, and no header of the request."""
    proxy, api, elsewhere = proxied
    monkeypatch.setenv("GITHUB_API_URL", f"https://127.0.0.1:{api.port}")
    proxy.answers[f"127.0.0.1:{api.port}"] = answer(b"HTTP/1.1 403 Forbidden")
    with pytest.raises(OSError):
        act.api(f"https://127.0.0.1:00{api.port}{API_PATH}", DIFF_ACCEPT)
    assert [(m, t) for m, t, _, _ in proxy.seen] == [("CONNECT", f"127.0.0.1:{api.port}")]
    assert_no_token(proxy.seen[0][3].decode("latin-1"))
    assert "authorization" not in proxy.seen[0][2]
    assert api.seen == [] and elsewhere.seen == []


# ── the token itself ───────────────────────────────────────────────────────────────────────────────

def test_without_gh_token_nothing_is_asked(servers, act, monkeypatch):
    api, elsewhere = servers
    monkeypatch.delenv("GH_TOKEN")
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert str(e.value) == "GH_TOKEN is not set, so the diff was not asked for"
    assert api.seen == [] and elsewhere.seen == []


@pytest.mark.parametrize("tail", ["\n", "\r\n", "\nX-Injected: 1", "\x7f", "\tx", "\x1b[0m",
                                  "\u0085", "\u00e9", "\u20ac", "\u2028"])
def test_a_token_a_header_cannot_carry_is_sent_nowhere_and_not_quoted(servers, act, monkeypatch, tail):
    """http.client refuses a control character in a header value in a sentence that quotes the value,
    and a character it cannot encode in one that names the character and its place; the script refuses
    before either, in a sentence that names the variable and quotes nothing."""
    api, elsewhere = servers
    monkeypatch.setenv("GH_TOKEN", TOKEN + tail)
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert str(e.value) == ("GH_TOKEN holds a line break, another control character or a character past "
                            "ASCII, so it was sent nowhere and the diff was not asked for")
    assert api.seen == [] and elsewhere.seen == []


def test_an_address_that_carries_the_token_back_is_shown_without_it(raw, act):
    """The API has just been sent the token and can name it in a Location; the sentence replaces it."""
    api, elsewhere = raw
    api.answers[API_PATH] = answer(
        b"HTTP/1.1 302 Found", b"Location: " + (elsewhere.base + "/?t=" + TOKEN + "&u=1").encode("ascii"))
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert repr(elsewhere.base + "/?t=***&u=1") in str(e.value)
    assert_no_token(str(e.value), repr(e.value))
    assert elsewhere.seen == []


def test_the_token_is_replaced_before_the_cut_so_no_part_of_it_is_left_at_the_edge(act):
    value = "a" * 190 + TOKEN
    assert act._shown(value, TOKEN) == repr("a" * 190 + "***")
    assert act._shown(value) == repr(value[:200])               # with no token named, the plain cut
    assert act._shown(None) == "None" and act._shown(7) == "7"


# ── what main() prints and writes when the fetch fails ─────────────────────────────────────────────

def _lines(out):
    """The lines a runner would read from stdout: it ends one at LF, at CR, and at CR LF."""
    return [ln for ln in out.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln]


def test_main_keeps_a_reason_phrase_with_a_carriage_return_on_the_warnings_line(raw, act, tmp_path,
                                                                                monkeypatch, capsys):
    """A server chooses its reason phrase, and urllib's error quotes it. Printed as it stood, a carriage
    return in it ended the ::warning line and the text after it stood at the start of a line of its own."""
    api, elsewhere = raw
    api.answers[API_PATH] = (b"HTTP/1.1 404 Not Found\r::error title=x::said by the reason phrase\r\n"
                             b"Content-Length: 0\r\nConnection: close\r\n\r\n")
    summary = _event(tmp_path, monkeypatch, "pull_request", _pr_event(api.base + API_PATH))
    assert act.main() == 0
    out = capsys.readouterr().out
    lines = _lines(out)
    assert len(lines) == 1 and lines[0].startswith("::warning title=styxx diffgate - could not fetch the diff::")
    assert "%0D::error title=x::said by the reason phrase" in lines[0]
    assert lines[0].endswith("The gate did not run; this is not a pass.")
    text = summary.read_text(encoding="utf-8")
    assert "DID NOT RUN" in text and "\r" not in text
    assert elsewhere.seen == []


def test_main_escapes_a_percent_sign_and_a_backtick_an_address_carries(raw, act, tmp_path, monkeypatch, capsys):
    """A runner decodes %0A in a command's message into a line break, and a backtick closes the summary's
    code span. Both come from the server here, in the Location."""
    api, elsewhere = raw
    api.answers[API_PATH] = answer(
        b"HTTP/1.1 302 Found", b"Location: " + elsewhere.base.encode("ascii") + b"/x%0A::error::y`[z](http://w)`")
    summary = _event(tmp_path, monkeypatch, "pull_request", _pr_event(api.base + API_PATH))
    assert act.main() == 0
    out = capsys.readouterr().out
    assert len(_lines(out)) == 1
    assert "/x%250A::error::y" in out and "/x%0A" not in out
    line = [ln for ln in summary.read_text(encoding="utf-8").split("\n") if ln.startswith("The diff could not")]
    assert len(line) == 1 and line[0].count("`") == 2           # the two that open and close the span
    assert "/x%0A::error::y'[z](http://w)'" in line[0]
    assert elsewhere.seen == []


def test_main_prints_no_token_a_server_sends_back_in_its_reason_phrase(raw, act, tmp_path, monkeypatch, capsys):
    api, elsewhere = raw
    api.answers[API_PATH] = (b"HTTP/1.1 404 " + TOKEN.encode("ascii") + b"\r\nContent-Length: 0\r\n"
                             b"Connection: close\r\n\r\n")
    summary = _event(tmp_path, monkeypatch, "pull_request", _pr_event(api.base + API_PATH))
    assert act.main() == 0
    out = capsys.readouterr()
    assert "HTTP Error 404: ***" in out.out
    assert_no_token(out.out, out.err, summary.read_text(encoding="utf-8"))


def test_main_says_so_when_gh_token_is_not_set_or_cannot_be_sent(servers, act, tmp_path, monkeypatch, capsys):
    api, elsewhere = servers
    summary = _event(tmp_path, monkeypatch, "pull_request", _pr_event(api.base + API_PATH))
    monkeypatch.setenv("GH_TOKEN", TOKEN + "\n")
    assert act.main() == 0
    out = capsys.readouterr()
    assert "GH_TOKEN holds a line break, another control character or a character past ASCII" in out.out
    assert "The gate did not run; this is not a pass." in out.out
    assert_no_token(out.out, out.err, summary.read_text(encoding="utf-8"))
    monkeypatch.delenv("GH_TOKEN")
    assert act.main() == 0
    assert "GH_TOKEN is not set, so the diff was not asked for" in capsys.readouterr().out
    assert api.seen == [] and elsewhere.seen == []


def test_the_text_of_a_failed_fetch_is_cut_at_1000_characters(act, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    assert act._failure_text(ValueError("x" * 5000)) == "x" * 1000
    assert act._failure_text(ValueError("a " + TOKEN + " b")) == "a *** b"
    monkeypatch.setenv("GH_TOKEN", "  " + TOKEN + " ")           # a token with spaces round it: both spellings go
    assert act._failure_text(ValueError("a " + TOKEN + " b")) == "a *** b"
    monkeypatch.delenv("GH_TOKEN")
    assert act._failure_text(ValueError("a b")) == "a b"


def test_the_two_escapes_as_pure_functions(act):
    assert act._command_data("50%\r\n::error::x\n") == "50%25%0D%0A::error::x%0A"
    assert act._code_span("a`b`\r\nc") == "a'b'  c"


# ── the shape of an address, the written-out address and the hidden token, change by change ────────

@pytest.mark.parametrize("ch", [chr(i) for i in range(0x21, 0x7f)])
def test_the_shape_of_a_host_character_by_character(act, ch):
    """A letter, a digit, ".", "-", "_" and "~" make a name; "/", "?" and "#" end the host; any other
    printable ASCII character leaves no origin."""
    got = act._origin(f"https://a{ch}b/x")
    if ch.isalnum() or ch in "._~-":
        assert got == ("https", f"a{ch}b".lower(), 443) == dg._origin(f"https://a{ch}b/x")
    elif ch in "/?#":
        assert got == ("https", "a", 443)
    else:
        assert got is None


def test_an_empty_port_is_the_schemes_own(act):
    assert act._origin("https://api.github.com:/x") == _API == dg._origin("https://api.github.com:/x")
    assert act._origin("https://[::1]:/x") == ("https", "::1", 443)
    assert act._on_origin("https://api.github.com:/x?q", _API) == "https://api.github.com/x?q"


@pytest.mark.parametrize("netloc, fits", [
    ("api.github.com", True), ("api.github.com:443", True), ("api.github.com:", True),
    ("[::1]", True), ("[::1]:8443", True), ("[::1]:", True),
    ("[::ffff:1.2.3.4]", True), ("[::FFFF:1.2.3.4]:8443", True), ("[FE80::1]", True),
    ("[::1].evil.example", False), ("[::1]evil.example:8443", False), ("[::1]x", False),
    ("evil.example.[::1]", False), ("evil.example.[::1]:8443", False), ("x[::1]", False),
    ("api.github.com:٤٤٣", False), ("api.github.com:+443", False),
    ("api.github.com:4_43", False), ("api.github.com:443:443", False),
    ("user@api.github.com", False), ("api.github.com\\", False), ("api.github.com%2e", False),
    ("[fe80::1%25eth0]", False), ("[v1.x]", False), ("[]", False),
    ("\u0430pi.github.com", False), ("api.github.com\u00e9", False),      # letters, and not ASCII ones
    ("[::\u0661]", False), ("", False),
])
def test_the_shape_of_host_and_port_held_apart_from_urlsplit(act, netloc, fits):
    """`_origin` asks urlsplit before it asks the shape, and this Python's urlsplit may refuse an address before the shape
    is asked (text beside a bracketed host; from 3.11 a port that is not ASCII digits). The shape is
    what stands where urlsplit is lenient, so it is held here by itself."""
    assert (act._NETLOC.fullmatch(netloc) is not None) is fits


@pytest.mark.parametrize("url", ["https://[::FFFF:1.2.3.4]:8443/x", "https://[::ffff:1.2.3.4]:8443/x"])
def test_an_ipv6_literal_with_capitals_or_a_dotted_tail_is_read(act, url):
    assert act._origin(url) == ("https", "::ffff:1.2.3.4", 8443) == dg._origin(url)
    assert act._on_origin(url, ("https", "::ffff:1.2.3.4", 8443)) == "https://[::ffff:1.2.3.4]:8443/x"


@pytest.mark.parametrize("c", ["\x00", "\x01", "\x08", "\x0b", "\x0c", "\x1b", "\x1f", " ", "\x7f",
                               "\x80", "\x85", "\x9f", "\xa0", "\xe4", "\u2028", "\u3000", "\u4e2d"])
@pytest.mark.parametrize("where", ["{c}https://api.github.com/x", "https://api.github.com/a{c}b",
                                   "https://api.github.com/x?a{c}b", "https://api.github.com/x#a{c}b",
                                   "https://api.github.com/x{c}", "https://api.github.com{c}/x"])
def test_anything_but_printable_ascii_anywhere_leaves_no_origin(act, c, where):
    """NUL, ESC and DEL as well as the space, and every character past ASCII: urlsplit drops a leading
    control character and keeps DEL in a path, and urllib drops a no-break space from the end of one."""
    assert act._origin(where.format(c=c)) is None
    assert act._on_origin(where.format(c=c), _API) is None


def test_an_address_the_script_cannot_read_is_said_to_be_that_and_not_to_be_off_the_origin(servers, act):
    """A space in a path, a character past ASCII, a line break at the end: the address names the API's
    host and port, so "is not on" the origin would not be true of it."""
    api, elsewhere = servers
    for tail in ("/a b", "/\u00e4", API_PATH + "\n", API_PATH + "\u00a0"):
        with pytest.raises(RuntimeError) as e:
            act.api(api.base + tail, DIFF_ACCEPT)
        msg = str(e.value)
        assert "is not one this script can read" in msg and "the one origin" not in msg
        assert "was not asked for" in msg and "\n" not in msg
    assert api.seen == [] and elsewhere.seen == []


def test_an_ipv6_literal_with_no_double_colon_goes_back_in_its_brackets(act):
    origin = ("https", "1:2:3:4:5:6:7:8", 8443)
    assert act._origin_text(origin) == "https://[1:2:3:4:5:6:7:8]:8443"
    assert act._origin("https://[1:2:3:4:5:6:7:8]:8443/x") == origin
    assert act._on_origin("https://[1:2:3:4:5:6:7:8]:08443/x", origin) == "https://[1:2:3:4:5:6:7:8]:8443/x"


@pytest.mark.parametrize("rest", ["/a;b/c;d?e", "/a%2Fb/c%20d", "/a%25b?c=%26&d=%3D", "/~user/a+b,c:d@e!f$g'(h)*i"])
def test_the_path_and_the_query_are_written_out_as_given(act, rest):
    """A ;parameter stays in the path, and nothing is quoted again."""
    assert act._on_origin("https://API.github.com:443" + rest, _API) == "https://api.github.com" + rest


def test_of_a_location_and_a_uri_the_sentence_names_the_location_urllib_reads(raw, act):
    api, elsewhere = raw
    out = elsewhere.base + API_PATH
    api.answers[API_PATH] = answer(b"HTTP/1.1 302 Found", b"URI: " + MOVED.encode("ascii"),
                                   b"Location: " + out.encode("ascii"))
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert f"GitHub answers 302, a redirect to {out!r}, which was not followed: only a redirect within" in str(e.value)
    assert api.targets() == [API_PATH] and elsewhere.seen == []


@pytest.mark.parametrize("code", CODES)
def test_the_sentence_for_a_location_urllib_cannot_read_names_the_code_that_was_answered(raw, act, code):
    api, elsewhere = raw
    api.answers[API_PATH] = answer(b"HTTP/1.1 %d Moved" % code, b"Location: http://[::1/x")
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert str(e.value) == (f"GitHub answers {code}, a redirect to 'http://[::1/x', which urllib could not "
                            "read as an address, so it was not followed")
    assert api.auths() == [f"Bearer {TOKEN}"] and elsewhere.seen == []


def test_a_location_urllib_cannot_read_that_carries_the_token_back_is_shown_without_it(raw, act):
    api, elsewhere = raw
    api.answers[API_PATH] = answer(b"HTTP/1.1 302 Found",
                                   b"Location: http://[::1/?t=" + TOKEN.encode("ascii") + b"&u=1")
    with pytest.raises(RuntimeError) as e:
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert "a redirect to 'http://[::1/?t=***&u=1', which urllib could not read as an address" in str(e.value)
    assert_no_token(str(e.value), repr(e.value))
    assert elsewhere.seen == []


def test_a_value_error_after_a_redirect_was_followed_is_raised_as_it_is(servers, act, monkeypatch):
    """Once the handler has agreed to a hop, a ValueError is not about reading that redirect's address."""
    api, elsewhere = servers
    api.routes[API_PATH] = (302, MOVED)
    real = urllib.request.OpenerDirector.open
    opened = []

    def fails_on_the_followed_request(self, req, *a, **k):
        opened.append(req.full_url)
        if len(opened) == 2:
            raise ValueError("not about a redirect")
        return real(self, req, *a, **k)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", fails_on_the_followed_request)
    with pytest.raises(ValueError, match="not about a redirect"):
        act.api(api.base + API_PATH, DIFF_ACCEPT)
    assert opened == [api.base + API_PATH, api.base + MOVED]
    assert api.paths() == [API_PATH] and elsewhere.seen == []


def test_a_token_with_a_space_in_it_is_sent_as_it_is(servers, act, monkeypatch):
    api, elsewhere = servers
    monkeypatch.setenv("GH_TOKEN", "made up token")
    assert act.api(api.base + API_PATH, DIFF_ACCEPT) == _DIFF
    assert api.auths() == ["Bearer made up token"] and elsewhere.seen == []


def test_an_empty_gh_token_is_not_an_unset_one(servers, act, monkeypatch):
    """What the script does today with GH_TOKEN="": it asks, with an empty bearer. If that is not what
    is wanted, `_token` is the place and this is the test to turn round."""
    api, elsewhere = servers
    monkeypatch.setenv("GH_TOKEN", "")
    assert act.api(api.base + API_PATH, DIFF_ACCEPT) == _DIFF
    assert [a.strip() for a in api.auths()] == ["Bearer"] and elsewhere.seen == []


def test_an_address_off_the_origin_is_refused_before_the_token_is_read(servers, act, monkeypatch):
    """Where two refusals apply: the origin's, then the address's, then the token's."""
    api, elsewhere = servers
    monkeypatch.delenv("GH_TOKEN")
    with pytest.raises(RuntimeError) as e:
        act.api(elsewhere.base + API_PATH, DIFF_ACCEPT)
    assert "the diff's address" in str(e.value) and "GH_TOKEN" not in str(e.value)
    with pytest.raises(RuntimeError) as e:
        act.api("not an address", DIFF_ACCEPT)
    assert "is not one this script can read" in str(e.value) and "GH_TOKEN" not in str(e.value)
    monkeypatch.setenv("GITHUB_API_URL", "not an address")
    with pytest.raises(RuntimeError) as e:
        act.api(elsewhere.base + API_PATH, DIFF_ACCEPT)
    assert "GITHUB_API_URL is 'not an address'" in str(e.value) and "GH_TOKEN" not in str(e.value)
    assert api.seen == [] and elsewhere.seen == []


def test_the_token_is_replaced_in_a_failed_fetchs_text_before_the_cut(act, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    assert act._failure_text(ValueError("x" * 990 + TOKEN)) == "x" * 990 + "***"
    assert act._failure_text(ValueError("x" * 990 + TOKEN + "y" * 20)) == ("x" * 990 + "***" + "y" * 20)[:1000]


def test_a_token_with_spaces_round_it_is_replaced_whole_where_the_text_holds_it_whole(act, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "  " + TOKEN + " ")
    assert act._failure_text(ValueError("a" + "  " + TOKEN + " " + "b")) == "a***b"
    assert act._failure_text(ValueError("a" + TOKEN + "b")) == "a***b"


def test_a_token_carried_back_more_than_once_is_replaced_every_time(act, monkeypatch):
    """Three copies: `_failure_text` replaces two spellings, which are one for a token with no spaces
    round it, so two copies would not tell a single replacement from a full one."""
    assert act._shown("a" + TOKEN + "b" + TOKEN + "c", TOKEN) == repr("a***b***c")
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    assert act._failure_text(ValueError(TOKEN + " and " + TOKEN + " and " + TOKEN)) == "*** and *** and ***"


def test_with_no_gh_token_a_failed_fetchs_text_is_left_as_it_is(act, monkeypatch):
    text = "".join(chr(i) for i in range(0x20, 0x7f))
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert act._failure_text(ValueError(text)) == text
    monkeypatch.setenv("GH_TOKEN", "")
    assert act._failure_text(ValueError(text)) == text


def test_the_two_escapes_replace_every_copy(act):
    assert act._command_data("%0A%0D %\r\r\n\n") == "%250A%250D %25%0D%0D%0A%0A"
    assert act._code_span("`a`\r\r\n\n`") == "'a'    '"


def test_main_keeps_a_reason_phrase_with_two_carriage_returns_on_the_warnings_line(raw, act, tmp_path,
                                                                                   monkeypatch, capsys):
    api, elsewhere = raw
    api.answers[API_PATH] = (b"HTTP/1.1 404 Not Found\r::error::one %0A\r::error::two %0A\r\n"
                             b"Content-Length: 0\r\nConnection: close\r\n\r\n")
    summary = _event(tmp_path, monkeypatch, "pull_request", _pr_event(api.base + API_PATH))
    assert act.main() == 0
    out = capsys.readouterr().out
    lines = _lines(out)
    assert len(lines) == 1 and "%0D::error::one %250A%0D::error::two %250A" in lines[0]
    assert "%0A" not in out.replace("%250A", "")
    text = summary.read_text(encoding="utf-8")
    assert "\r" not in text and "::error::one %0A ::error::two %0A" in text
    assert elsewhere.seen == []


def test_a_token_with_spaces_round_it_is_hidden_in_an_address_as_a_server_would_send_it_back(act):
    """A server reads the header's value without the spaces round it, and that is what it can echo."""
    assert act._shown("x" + TOKEN + "y", "  " + TOKEN + " ") == repr("x***y")
    assert act._shown("x  " + TOKEN + " y", "  " + TOKEN + " ") == repr("x***y")
    assert act._hide("a b", None) == "a b" and act._hide("a b", "") == "a b" and act._hide("a b", "   ") == "a b"


def test_a_control_character_in_a_failed_fetchs_text_is_written_as_an_escape(act, monkeypatch):
    """ESC, a vertical tab, NEL and DEL are not line ends to a runner, and they are not printed raw
    either. CR and LF are left for the place the text is printed."""
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert act._failure_text(ValueError("a\x1b[31mb\x0bc\x85d\x7fe\x00f\tg")) == \
        "a\\x1b[31mb\\x0bc\\x85d\\x7fe\\x00f\\x09g"
    assert act._failure_text(ValueError("a\r\nb é 中")) == "a\r\nb é 中"


def test_main_prints_no_raw_control_character_from_a_reason_phrase(raw, act, tmp_path, monkeypatch, capsys):
    api, elsewhere = raw
    api.answers[API_PATH] = (b"HTTP/1.1 404 Not\x1b[2J Found\x0b::error::x\r\nContent-Length: 0\r\n"
                             b"Connection: close\r\n\r\n")
    summary = _event(tmp_path, monkeypatch, "pull_request", _pr_event(api.base + API_PATH))
    assert act.main() == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x0b" not in out and len(out.splitlines()) == 1
    assert "HTTP Error 404: Not\\x1b[2J Found\\x0b::error::x" in out
    text = summary.read_text(encoding="utf-8")
    assert "\x1b" not in text and "\x0b" not in text
    assert elsewhere.seen == []


def test_api_installs_no_opener_for_the_rest_of_the_process(servers, act):
    api, _ = servers
    before = urllib.request._opener
    assert act.api(api.base + API_PATH, DIFF_ACCEPT) == _DIFF
    assert urllib.request._opener is before
