# -*- coding: utf-8 -*-
"""The know-say example follows no redirect, so its API key goes to no address a redirect names.

`examples/knowsay_endpoint.py` (the script the README tells users to run with their own key) put
``Authorization: Bearer <key>`` among a request's ordinary headers and sent a POST through
`urllib.request.urlopen`. urllib answers a 301, 302 or 303 by sending a GET with every ordinary header
to the address the Location names, with no check of host, port or scheme, so the key went along. The key
now travels as an unredirected header, through an opener of the script's own that follows no redirect
for any of 301, 302, 303, 307 and 308; a redirect ends the run at once, without the retry waits.

Everything here runs against real HTTP servers on 127.0.0.1, because a test that replaces the client
cannot see redirect handling: `endpoint` stands for the endpoint the user names and `elsewhere` for any
other address; both record every request with its headers. The key is a made-up string. Connections
and name lookups off loopback are refused for the length of each test.
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import runpy
import socket
import sys
import threading
import traceback
import urllib.request
from pathlib import Path

import pytest

# the example imports styxx.knowsay; where that cannot be imported, there is nothing here to test
pytest.importorskip("styxx.knowsay")

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "knowsay_endpoint.py"
KEY = "made-up-knowsay-key-0123456789abcdef"     # not a real key; checked for in every text shown
CODES = (301, 302, 303, 307, 308)
_LOOPBACK_ADDR = {"127.0.0.1", "::1"}
_LOOPBACK_NAMES = _LOOPBACK_ADDR | {"localhost", None}
_REPLY = {"choices": [{"message": {"content": " Paris "}}]}


class OffLoopback(OSError):
    """Raised by the guard for a connection or a lookup that would leave loopback."""


class Server:
    """A real HTTP server on 127.0.0.1. `redirect` is (code, Location) to answer with a redirect
    (Location None: no Location header at all), or None to answer with a chat completion; the first
    `redirect_after` requests are answered with a chat completion whatever `redirect` says. Every
    request is recorded as (method, path, {header: value})."""

    def __init__(self):
        self.redirect = None
        self.redirect_after = 0
        self.seen = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def _answer(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                outer.seen.append((self.command, self.path,
                                   {k.lower(): v for k, v in self.headers.items()}))
                if outer.redirect is not None and len(outer.seen) > outer.redirect_after:
                    code, location = outer.redirect
                    self.send_response(code)
                    if location is not None:
                        self.send_header("Location", location)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps(_REPLY).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = _answer

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def auths(self):
        return [h.get("authorization") for _, _, h in self.seen]

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
    endpoint, elsewhere = Server(), Server()
    try:
        yield endpoint, elsewhere
    finally:
        endpoint.close()
        elsewhere.close()


@pytest.fixture
def sleeps(monkeypatch):
    """Every wait the retry loop asks for, recorded and not slept."""
    asked = []
    import time
    monkeypatch.setattr(time, "sleep", lambda s: asked.append(s))
    return asked


def load_example(monkeypatch):
    """The example loaded from its file (examples/ is not a package), with a fresh opener."""
    name = "knowsay_endpoint_under_test"
    spec = importlib.util.spec_from_file_location(name, EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_OPENER", None, raising=False)
    return mod


def _shown(e) -> str:
    """Everything a run would print for the exception in `e`: its text and its whole traceback."""
    return str(e.value) + "".join(traceback.format_exception(e.type, e.value, e.tb))


def _try_chat(mod, base_url):
    """chat() called once: (its answer, None), or (None, the exception it raised). The caller checks
    where the key went before it checks how the call ended, so a leak is reported as a leak."""
    try:
        return mod.chat(base_url, "m", KEY, _MSGS), None
    except Exception as exc:
        return None, exc


def _no_key_reached(server):
    """`server` received no request, so no header of it holds the key."""
    assert not any(KEY in v for _, _, h in server.seen for v in h.values()), "the key reached it"
    assert server.seen == []


_MSGS = [{"role": "user", "content": "What is the capital of France?"}]


def test_a_normal_answer_works_and_the_endpoint_receives_the_key(servers, monkeypatch, sleeps):
    endpoint, elsewhere = servers
    mod = load_example(monkeypatch)
    assert mod.chat(endpoint.base + "/v1", "m", KEY, _MSGS) == "Paris"
    assert [(m, p) for m, p, _ in endpoint.seen] == [("POST", "/v1/chat/completions")]
    assert endpoint.auths() == [f"Bearer {KEY}"]
    assert elsewhere.seen == []
    assert sleeps == []


@pytest.mark.parametrize("code", CODES)
def test_a_redirect_is_not_followed_and_the_key_goes_nowhere_else(servers, monkeypatch, sleeps, code):
    endpoint, elsewhere = servers
    endpoint.redirect = (code, elsewhere.base + "/stolen")
    mod = load_example(monkeypatch)
    answer, exc = _try_chat(mod, endpoint.base + "/v1")
    _no_key_reached(elsewhere)
    assert exc is not None, f"the redirect was followed and chat() answered {answer!r}"
    shown = str(exc) + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert KEY not in shown
    assert str(code) in str(exc)
    assert "no redirect" in str(exc)


@pytest.mark.parametrize("code", (302, 307))
def test_a_redirect_with_no_location_ends_the_call_at_once(servers, monkeypatch, sleeps, code):
    endpoint, elsewhere = servers
    endpoint.redirect = (code, None)
    mod = load_example(monkeypatch)
    answer, exc = _try_chat(mod, endpoint.base + "/v1")
    assert exc is not None, f"chat() answered {answer!r}"
    assert str(code) in str(exc) and "no redirect" in str(exc)
    assert KEY not in str(exc)
    assert sleeps == []
    assert len(endpoint.seen) == 1


def test_a_redirect_after_a_good_answer_is_not_followed(servers, monkeypatch, sleeps):
    endpoint, elsewhere = servers
    endpoint.redirect = (302, elsewhere.base + "/stolen")
    endpoint.redirect_after = 1
    mod = load_example(monkeypatch)
    assert mod.chat(endpoint.base + "/v1", "m", KEY, _MSGS) == "Paris"
    answer, exc = _try_chat(mod, endpoint.base + "/v1")
    _no_key_reached(elsewhere)
    assert exc is not None, f"the redirect was followed and chat() answered {answer!r}"
    assert "302" in str(exc) and KEY not in str(exc)
    assert endpoint.auths() == [f"Bearer {KEY}"] * 2
    assert sleeps == []


@pytest.mark.parametrize("code", CODES)
def test_a_redirect_burns_no_retry_waits(servers, monkeypatch, sleeps, code):
    endpoint, elsewhere = servers
    endpoint.redirect = (code, elsewhere.base + "/stolen")
    mod = load_example(monkeypatch)
    with pytest.raises(Exception):
        mod.chat(endpoint.base + "/v1", "m", KEY, _MSGS)
    assert sleeps == []
    assert endpoint.auths() == [f"Bearer {KEY}"]
    assert elsewhere.seen == []


@pytest.mark.parametrize("code", CODES)
def test_a_redirect_to_the_same_server_is_not_followed_either(servers, monkeypatch, sleeps, code):
    endpoint, elsewhere = servers
    endpoint.redirect = (code, "/v2/chat/completions")
    mod = load_example(monkeypatch)
    with pytest.raises(Exception) as e:
        mod.chat(endpoint.base + "/v1", "m", KEY, _MSGS)
    assert [p for _, p, _ in endpoint.seen] == ["/v1/chat/completions"]
    assert KEY not in _shown(e)
    assert elsewhere.seen == []


def test_an_installed_opener_is_not_used(servers, monkeypatch, sleeps):
    endpoint, elsewhere = servers
    used = []

    class Grab(urllib.request.BaseHandler):
        handler_order = 0

        def http_open(self, req):
            used.append(req.full_url)
            raise OSError("the installed opener was used")

    monkeypatch.setattr(urllib.request, "_opener", None)
    urllib.request.install_opener(urllib.request.build_opener(Grab))
    mod = load_example(monkeypatch)
    assert mod.chat(endpoint.base + "/v1", "m", KEY, _MSGS) == "Paris"
    assert used == []
    assert endpoint.auths() == [f"Bearer {KEY}"]


def test_the_key_is_carried_as_an_unredirected_header_alone(monkeypatch):
    """urllib copies a request's ordinary headers to a redirected request and never its unredirected
    ones; the key must be among the latter only, whatever the redirect handler does."""
    mod = load_example(monkeypatch)
    sent = []

    class Stop(Exception):
        pass

    class Recorder:
        def open(self, req, timeout=None):
            sent.append(req)
            raise Stop()

    monkeypatch.setattr(mod, "_opener", lambda: Recorder())
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    with pytest.raises(Stop):
        mod.chat("http://127.0.0.1:9/v1", "m", KEY, _MSGS, retries=1)
    (req,) = sent
    assert req.unredirected_hdrs.get("Authorization") == f"Bearer {KEY}"
    assert all(k.lower() != "authorization" for k in req.headers)


@pytest.mark.parametrize("code", (302, 308))
def test_a_run_ends_at_once_with_one_sentence_and_writes_no_output(servers, monkeypatch, sleeps, tmp_path,
                                                                   capsys, code):
    """The script run as a command: a redirect ends it with a message naming the code, no traceback, no
    datasheet written, and the key in nothing printed."""
    endpoint, elsewhere = servers
    endpoint.redirect = (code, elsewhere.base + "/stolen")
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps({"q": "Capital of France?", "gold": "Paris"}) + "\n", encoding="utf-8")
    out = tmp_path / "sheet.json"
    monkeypatch.setenv("KNOWSAY_TEST_KEY", KEY)
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE), str(questions), "--base-url", endpoint.base + "/v1",
                                      "--model", "m", "--api-key-env", "KNOWSAY_TEST_KEY", "--out", str(out)])
    monkeypatch.setattr(urllib.request, "_opener", None)
    try:
        runpy.run_path(str(EXAMPLE), run_name="__main__")
        message = None
    except SystemExit as stop:
        message = stop.code
    _no_key_reached(elsewhere)
    assert message is not None, "the run ended normally"
    assert isinstance(message, str)
    assert str(code) in message and "no redirect" in message
    assert KEY not in message
    printed = capsys.readouterr()
    assert KEY not in printed.out and KEY not in printed.err
    assert not out.exists()
    assert sleeps == []
    assert len(endpoint.seen) == 1
    assert elsewhere.seen == []
