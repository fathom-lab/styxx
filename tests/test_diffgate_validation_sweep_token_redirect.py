# -*- coding: utf-8 -*-
"""The validation sweep's token is sent to the API's origin and nowhere else, whatever a redirect or a
search result names.

`scripts/diffgate_validation_sweep.py` (the script action.yml and the README send users to, to re-run the
numbers) put ``Authorization: Bearer <token>`` among a request's ordinary headers and sent it through
`urllib.request.urlopen`, which follows 301, 302, 303 and 307 for a GET (and 308 from Python 3.11) and
copies those headers to whatever address the Location names. It also sent the token to each pull
request's address as the search answer named it, wherever that pointed. The token now travels as an
unredirected header, the script's own opener follows a redirect (308 included, on every Python) only
within the API's origin and puts the token back there, and an address off that origin is not asked at all.

Everything here runs against real HTTP servers on 127.0.0.1, because a test that replaces the client
cannot see redirect handling: `api` stands for the API, `elsewhere` for any other address and records every
header it is sent. The script's `_API_BASE` is pointed at `api`, and its `token` is replaced by one that
returns a made-up string, so no file or variable holding a real token is read. Connections and name lookups
off loopback are refused for the length of each test.
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import socket
import sys
import threading
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

SWEEP = Path(__file__).resolve().parent.parent / "scripts" / "diffgate_validation_sweep.py"
TOKEN = "made-up-sweep-token-0123456789abcdef"     # not a real token; checked for in every text shown
CODES = (301, 302, 303, 307, 308)
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
    """Raised by the guard for a connection or a lookup that would leave loopback."""


class Server:
    """A real HTTP server on 127.0.0.1. `routes` maps a path (query included) to (code, Location) to
    answer with a redirect (with no Location header when it is None), or to (200, a JSON-able value) to answer with it; any other path is answered
    with the diff when the diff is asked for and with {"ok": true} otherwise. Every request is recorded
    as (path, {header: value})."""

    def __init__(self):
        self.routes = {}
        self.seen = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
                code, value = outer.routes.get(self.path, (200, None))
                if code != 200:
                    self.send_response(code)
                    if value is not None:
                        self.send_header("Location", value)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if value is None and self.headers.get("Accept") == "application/vnd.github.diff":
                    body = _DIFF.encode("utf-8")
                else:
                    body = json.dumps({"ok": True} if value is None else value).encode("utf-8")
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

    def auths(self):
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
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {})
    api, elsewhere = Server(), Server()
    try:
        yield api, elsewhere
    finally:
        api.close()
        elsewhere.close()


def load_sweep(monkeypatch, base, tok=TOKEN, keep_token_source=False):
    """The script loaded from its file (scripts/ is not a package), its API pointed at `base` and its
    token replaced, so the real token source is never read. With `keep_token_source` the script's own
    `token` stays and the test sets GH_TOKEN itself; every such test has gh_token in its name."""
    name = "diffgate_validation_sweep_under_test"
    spec = importlib.util.spec_from_file_location(name, SWEEP)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    if not keep_token_source:
        monkeypatch.setattr(mod, "token", lambda: tok)
    monkeypatch.setattr(mod, "_API_BASE", base, raising=False)
    monkeypatch.setattr(mod, "_OPENER", None, raising=False)
    return mod


def test_a_request_on_the_api_carries_the_token(servers, monkeypatch):
    api, elsewhere = servers
    mod = load_sweep(monkeypatch, api.base)
    assert mod.api(api.base + "/repos/o/r/pulls/7") == {"ok": True}
    assert api.auths() == [f"Bearer {TOKEN}"]
    assert elsewhere.seen == []


@pytest.mark.parametrize("code", CODES)
def test_a_redirect_to_another_origin_is_not_followed_and_the_token_goes_nowhere_else(servers, monkeypatch, code):
    api, elsewhere = servers
    api.routes["/moved"] = (code, elsewhere.base + "/stolen")
    mod = load_sweep(monkeypatch, api.base)
    with pytest.raises(Exception) as e:
        mod.api(api.base + "/moved")
    assert TOKEN not in str(e.value)
    assert elsewhere.seen == []
    assert api.auths() == [f"Bearer {TOKEN}"]


@pytest.mark.parametrize("code", CODES)
def test_a_redirect_within_the_api_origin_is_followed_with_the_token(servers, monkeypatch, code):
    api, elsewhere = servers
    api.routes["/moved"] = (code, "/repos/o/r/pulls/7")
    mod = load_sweep(monkeypatch, api.base)
    assert mod.api(api.base + "/moved", "application/vnd.github.diff") == _DIFF
    assert [p for p, _ in api.seen] == ["/moved", "/repos/o/r/pulls/7"]
    assert api.auths() == [f"Bearer {TOKEN}", f"Bearer {TOKEN}"]
    assert elsewhere.seen == []


@pytest.mark.parametrize("target", [
    "http://localhost:{api}/x",          # the same server under another name
    "http://127.0.0.1:{other}/x",        # another port
    "http://user@127.0.0.1:{api}/x",     # a user part
    "https://127.0.0.1:{api}/x",         # another scheme
])
def test_a_redirect_to_the_same_server_by_another_origin_is_not_followed(servers, monkeypatch, target):
    api, elsewhere = servers
    api.routes["/moved"] = (302, target.format(api=api.port, other=elsewhere.port))
    mod = load_sweep(monkeypatch, api.base)
    with pytest.raises(Exception) as e:
        mod.api(api.base + "/moved")
    assert TOKEN not in str(e.value)
    assert [p for p, _ in api.seen] == ["/moved"]
    assert elsewhere.seen == []


def test_an_address_off_the_api_origin_is_not_asked_at_all(servers, monkeypatch):
    api, elsewhere = servers
    mod = load_sweep(monkeypatch, api.base)
    with pytest.raises(RuntimeError) as e:
        mod.api(elsewhere.base + "/repos/o/r/pulls/7", "application/vnd.github.diff")
    assert TOKEN not in str(e.value)
    assert elsewhere.seen == [] and api.seen == []


def test_a_pull_request_address_from_the_search_answer_off_the_origin_gets_no_request(servers, monkeypatch,
                                                                                    capsys):
    """The search answer names two pull requests: one on the API, one elsewhere. The one on the API is
    gated; the other is skipped and its server is sent nothing."""
    api, elsewhere = servers
    q = urllib.parse.quote('"Generated with [Claude Code]" in:body is:pr is:public')
    item = {"body": "Modified src/retry.py.", "title": "t", "html_url": "https://github.com/o/r/pull/7"}
    api.routes[f"/search/issues?q={q}&per_page=30&page=1"] = (200, {"items": [
        dict(item, pull_request={"url": elsewhere.base + "/repos/o/r/pulls/7"}),
        dict(item, pull_request={"url": api.base + "/repos/o/r/pulls/7"}),
    ]})
    api.routes[f"/search/issues?q={q}&per_page=30&page=2"] = (200, {"items": []})
    mod = load_sweep(monkeypatch, api.base)
    rows = mod.sweep_market(5)
    assert len(rows) == 1
    assert elsewhere.seen == []
    assert set(api.auths()) == {f"Bearer {TOKEN}"}
    out = capsys.readouterr()
    assert TOKEN not in out.out and TOKEN not in out.err


def test_an_installed_opener_is_not_used(servers, monkeypatch):
    api, elsewhere = servers
    used = []

    class Grab(urllib.request.BaseHandler):
        handler_order = 0

        def http_open(self, req):
            used.append((req.full_url, req.get_header("Authorization"), req.unredirected_hdrs.get("Authorization")))
            raise OSError("the installed opener was used")

    monkeypatch.setattr(urllib.request, "_opener", None)
    urllib.request.install_opener(urllib.request.build_opener(Grab))
    mod = load_sweep(monkeypatch, api.base)
    assert mod.api(api.base + "/repos/o/r/pulls/7") == {"ok": True}
    assert used == []
    assert api.auths() == [f"Bearer {TOKEN}"]


def test_with_no_token_no_request_carries_an_authorization_header(servers, monkeypatch):
    api, elsewhere = servers
    api.routes["/moved"] = (302, "/repos/o/r/pulls/7")
    mod = load_sweep(monkeypatch, api.base, tok=None)
    assert mod.api(api.base + "/moved") == {"ok": True}
    assert api.auths() == [None, None]
    assert elsewhere.seen == []


def test_the_token_is_read_from_gh_token_alone(monkeypatch):
    """The script names no file of anyone's machine as a token source; it reads GH_TOKEN."""
    name = "diffgate_validation_sweep_token_source"
    spec = importlib.util.spec_from_file_location(name, SWEEP)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    assert mod.token() == TOKEN
    monkeypatch.delenv("GH_TOKEN")
    assert mod.token() is None
    assert "secrets" not in SWEEP.read_text(encoding="utf-8")


def _shown(e) -> str:
    """Everything a run would print for the exception in `e`: its text and its whole traceback."""
    return str(e.value) + "".join(traceback.format_exception(e.type, e.value, e.tb))


# -- the origin check itself, against the shipped base ----------------------------------------------------

@pytest.mark.parametrize("url, same", [
    ("https://api.github.com/search/issues", True),
    ("https://API.GitHub.com/x", True),                   # host names are compared in lower case
    ("https://api.github.com:443/x", True),               # the scheme's default port written out
    ("http://api.github.com/x", False),                   # another scheme
    ("http://api.github.com:443/x", False),               # another scheme on the same port
    ("https://api.github.com:8443/x", False),             # another port
    ("https://user@api.github.com/x", False),             # a user part, even before the right host
    ("https://api.github.com@evil.example/x", False),     # a user part that names the API
    ("https://api.github.com/x\ty", False),               # a control character
    ("https://api.github.com/x\n", False),                # a line break
    ("https://api.github.com/x y", False),                # a space
    ("https://api.github.com/\u00e9", False),             # a character past ASCII
    ("ftp://api.github.com/x", False),                    # neither http nor https
    ("https:///x", False),                                # no host
])
def test_the_origin_check_compares_scheme_host_and_port_and_refuses_a_user_part(monkeypatch, url, same):
    mod = load_sweep(monkeypatch, "https://api.github.com")
    assert mod._same_origin(url) is same


# -- the token never shows in an error -----------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["\n", "\r\nHost: 127.0.0.2", "\nX-Evil: 1"])
def test_a_token_with_a_line_break_sends_nothing_and_shows_in_no_error(servers, monkeypatch, bad):
    """http.client refuses such a header value with a ValueError that quotes the header whole; that
    error must not reach the user with the token in it."""
    api, elsewhere = servers
    mod = load_sweep(monkeypatch, api.base, tok=TOKEN + bad)
    with pytest.raises(RuntimeError) as e:
        mod.api(api.base + "/repos/o/r/pulls/7")
    assert TOKEN not in _shown(e)
    assert api.seen == [] and elsewhere.seen == []


def test_gh_token_is_stripped_of_the_white_space_around_it(monkeypatch):
    mod = load_sweep(monkeypatch, "https://api.github.com", keep_token_source=True)
    monkeypatch.setenv("GH_TOKEN", "  " + TOKEN + "\r\n")
    assert mod.token() == TOKEN
    monkeypatch.setenv("GH_TOKEN", " \n")
    assert mod.token() is None


@pytest.mark.parametrize("inside", [" ", "\n", "\r\nX-Evil: 1", "\t", "\u00e9"])
def test_gh_token_with_white_space_or_a_control_character_inside_is_refused_without_being_quoted(
        monkeypatch, inside):
    mod = load_sweep(monkeypatch, "https://api.github.com", keep_token_source=True)
    monkeypatch.setenv("GH_TOKEN", TOKEN + inside + "tail")
    with pytest.raises(RuntimeError) as e:
        mod.token()
    assert TOKEN not in _shown(e)


def test_a_market_sweep_with_a_broken_gh_token_sends_nothing_and_prints_no_token(servers, monkeypatch,
                                                                                 capsys):
    api, elsewhere = servers
    mod = load_sweep(monkeypatch, api.base, keep_token_source=True)
    monkeypatch.setenv("GH_TOKEN", TOKEN + "\nX-Evil: 1")
    assert mod.sweep_market(1) == []
    out = capsys.readouterr()
    assert TOKEN not in out.out and TOKEN not in out.err
    assert "unavailable" in out.err
    assert api.seen == [] and elsewhere.seen == []


def test_a_location_urllib_cannot_parse_is_raised_without_the_token(servers, monkeypatch):
    api, elsewhere = servers
    api.routes["/moved"] = (302, f"http://[127.0.0.2]:{elsewhere.port}/x")
    mod = load_sweep(monkeypatch, api.base)
    with pytest.raises(RuntimeError) as e:
        mod.api(api.base + "/moved")
    assert TOKEN not in _shown(e)
    assert elsewhere.seen == []


# -- what the error says ----------------------------------------------------------------------------------------

def test_the_error_says_off_the_origin_only_for_a_redirect_off_it(servers, monkeypatch):
    api, elsewhere = servers
    api.routes["/away"] = (302, elsewhere.base + "/x")
    api.routes["/loop"] = (302, "/loop")
    api.routes["/noloc"] = (302, None)
    mod = load_sweep(monkeypatch, api.base)
    with pytest.raises(RuntimeError) as away:
        mod.api(api.base + "/away")
    assert "redirect off" in str(away.value)
    for path in ("/loop", "/noloc"):
        with pytest.raises(RuntimeError) as e:
            mod.api(api.base + path)
        assert "redirect off" not in str(e.value)
        assert "not followed" in str(e.value)
        assert TOKEN not in _shown(e)
    assert elsewhere.seen == []


# -- the search -----------------------------------------------------------------------------------------------------

def test_a_search_redirected_off_the_origin_is_reported_unavailable_and_sends_nothing_elsewhere(
        servers, monkeypatch, capsys):
    api, elsewhere = servers
    q = urllib.parse.quote('"Generated with [Claude Code]" in:body is:pr is:public')
    api.routes[f"/search/issues?q={q}&per_page=30&page=1"] = (302, elsewhere.base + "/search")
    mod = load_sweep(monkeypatch, api.base)
    assert mod.sweep_market(5) == []
    out = capsys.readouterr()
    assert "unavailable" in out.err
    assert TOKEN not in out.out and TOKEN not in out.err
    assert elsewhere.seen == []


# -- how the token is carried -----------------------------------------------------------------------------------------

def test_the_token_is_carried_as_an_unredirected_header_alone(monkeypatch):
    """urllib copies a request's ordinary headers to a redirected request and never its unredirected
    ones; the token must be among the latter only, whatever the redirect handler does."""
    mod = load_sweep(monkeypatch, "http://127.0.0.1:9")
    sent = []

    class Stop(Exception):
        pass

    class Recorder:
        def open(self, req, timeout=None):
            sent.append(req)
            raise Stop()

    monkeypatch.setattr(mod, "_opener", lambda: Recorder())
    with pytest.raises(Stop):
        mod.api("http://127.0.0.1:9/repos/o/r/pulls/7")
    (req,) = sent
    assert req.unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"
    assert all(k.lower() != "authorization" for k in req.headers)
