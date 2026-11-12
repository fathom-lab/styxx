# -*- coding: utf-8 -*-
"""`styxx publish`, `styxx compare-agents`, `styxx claim` and the stream name no host by default.

Found on 2026-11-12, after styxx 8.0.0 was published: those four defaulted to hosts under
darkflobi.com, and every name under that domain answered that day from a domain-parking service,
not from a lab server. By its code, `styxx publish --name X` in 8.0.0 hands its payload to that
server and reports any 2xx reply as published, and `styxx compare-agents` reads a web page there
as a leaderboard with no agents on it (read from the code and run against stand-ins: no POST was
sent to that host from here).

What these tests pin: with no endpoint, relay or leaderboard URL named, none of the four opens a
connection; a 2xx reply that is not a JSON object is not "published"; a reply that is not a
leaderboard is not an empty one; a relay given as a bare origin is claimed on that host and no
other; no redirect is followed, so the stream key and the reply stay with the address named; and
no file under `styxx/` names that domain apart from eight listed lines.

Scope. Other commands of the package contact a named service by design and are not the subject
here: `styxx.clock` reads two public Solana RPC hosts, `styxx.guardrail.entity_verify` queries
Wikipedia, the diff gate fetches a pull request the caller names.

Most tests here open no connection. They replace `urllib.request.OpenerDirector.open`, the one
method both `urlopen` and the package's own opener go through, with a recorder that notes the URL
and answers or refuses, and they assert on the recorder: several of the functions under test
catch every exception, so a guard that only raised would be swallowed and prove nothing (a review
of an earlier version of this file put a default host or a link on the domain back eight ways,
and seven went unseen).

A recorder at that level cannot see a redirect, which urllib follows below it. A later review
showed exactly that: a relay answering 301, 302 or 303 had the stream key forwarded to the
address it named, in 8.0.0 and under these tests alike. The redirect tests at the end of this
file therefore run the real urllib against two servers on 127.0.0.1 that this file starts, with
every proxy setting switched off, and read what each server received.
"""
from __future__ import annotations

import http.server
import importlib
import inspect
import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import urllib.error
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import styxx

# each module by its name: a package attribute of the same name can be something else
cli = importlib.import_module("styxx.cli")
compare = importlib.import_module("styxx.compare")
publish_mod = importlib.import_module("styxx.publish")
stream = importlib.import_module("styxx.stream")
analytics = importlib.import_module("styxx.analytics")

ROOT = Path(__file__).resolve().parent.parent
OLD_RELAY = "https://live.old-default.invalid/ingest"
_REAL_LOAD_CREDS = stream._load_creds       # the fixture below replaces it for every test


def _reply(body: bytes, status: int = 200):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class Net:
    """Stands in for the opener: records every URL asked for, then answers `body` or refuses."""

    def __init__(self, body=None):
        self.opened, self.bodies, self.timeouts, self.body = [], [], [], body

    def __call__(self, req, *a, **k):
        self.opened.append(getattr(req, "full_url", req))
        self.bodies.append(getattr(req, "data", None))
        self.timeouts.append(k.get("timeout"))
        if self.body is None:
            raise OSError("no network in tests")
        return _reply(self.body)


@pytest.fixture
def net(monkeypatch):
    """One recorder behind every request these modules can send: `urlopen` and the opener of
    styxx.stream.open_url both end in OpenerDirector.open. (An instance set on the class is not
    bound, so it is called with the request as its leading argument, as `urlopen` was.)"""
    import urllib.request
    rec = Net()
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", rec)
    return rec


@pytest.fixture(autouse=True)
def _no_ambient_settings(monkeypatch, tmp_path):
    for name in ("STYXX_STREAM_RELAY", "STYXX_DASHBOARD_BASE", "STYXX_STREAM", "STYXX_STREAM_AGENT",
                 "STYXX_AGENT_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STYXX_DATA_DIR", str(tmp_path / "data"))
    # never the credentials file of whoever runs the tests
    monkeypatch.setattr(stream, "_load_creds", lambda: {})
    monkeypatch.setattr(stream, "_save_creds", lambda data: None)
    # the emitter's state, as a fresh process has it; and no opener built under another test's
    # proxy settings
    for name, value in (("_enabled", False), ("_agent", None), ("_key", None), ("_relay", None),
                        ("_worker", None), ("_OPENER", None)):
        monkeypatch.setattr(stream, name, value)


class _NoThread:
    started = 0

    def __init__(self, *a, **k):
        pass

    def start(self):
        type(self).started += 1

    def is_alive(self):
        return True


@pytest.fixture
def no_thread(monkeypatch):
    _NoThread.started = 0
    monkeypatch.setattr(stream.threading, "Thread", _NoThread)
    return _NoThread


def _fingerprint():
    return analytics.Fingerprint(n_samples=100, phase1_vec=(0.05, 0.7, 0.05, 0.1, 0.05, 0.05),
                                 phase4_vec=(0.05, 0.7, 0.05, 0.1, 0.05, 0.05), phase1_mean_conf=0.8,
                                 phase4_mean_conf=0.8, gate_vec=(0.9, 0.08, 0.02))


# ── publish ───────────────────────────────────────────────────────

def test_publish_has_no_default_endpoint_anywhere():
    assert publish_mod._DEFAULT_ENDPOINT is None
    assert inspect.signature(publish_mod.publish).parameters["endpoint"].default is None
    assert cli._build_parser().parse_args(["publish", "--name", "agent"]).endpoint is None


@pytest.mark.parametrize("args", [("agent",), ("agent", ""), ("agent", None)],
                         ids=["no_endpoint", "empty_endpoint", "endpoint_none"])
def test_publish_without_an_endpoint_collects_and_sends_nothing(monkeypatch, net, capsys, args):
    collected = []
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a, **k: collected.append(a) or {})
    assert publish_mod.publish(*args) is None
    assert net.opened == [] and collected == []
    assert "nothing was sent" in capsys.readouterr().err


def test_publish_command_without_an_endpoint_fails_and_sends_nothing(net, capsys):
    assert cli.main(["publish", "--name", "agent"]) == 1
    captured = capsys.readouterr()
    assert net.opened == []
    assert "nothing was sent" in captured.err
    assert "published" not in captured.out


@pytest.mark.parametrize("endpoint", ["example.invalid/api", "ftp://example.invalid/api", "https://", "//x"])
def test_publish_to_something_that_is_not_a_url_sends_nothing_and_does_not_raise(monkeypatch, net, capsys, endpoint):
    collected = []
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a, **k: collected.append(a) or {})
    assert publish_mod.publish("agent", endpoint) is None
    assert net.opened == [] and collected == []
    assert "not an http or https URL" in capsys.readouterr().err
    assert cli.main(["publish", "--name", "agent", "--endpoint", endpoint]) == 1
    assert net.opened == []
    capsys.readouterr()


@pytest.mark.parametrize("body", [
    b"<!doctype html><html><title>parked</title></html>",     # what a parked host answers
    b"",
    b"[1, 2]",
    b'"ok"',
])
def test_a_2xx_reply_that_is_not_a_json_object_is_not_published(monkeypatch, net, capsys, body):
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a, **k: {"agent_name": "agent"})
    net.body = body
    assert publish_mod.publish("agent", "https://example.invalid/api") is None
    assert net.opened == ["https://example.invalid/api"]
    err = capsys.readouterr().err
    assert "not counted as published" in err and "the payload was sent" in err


def test_a_json_object_reply_is_published(monkeypatch, net):
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a, **k: {"agent_name": "agent"})
    net.body = json.dumps({"ok": True}).encode("utf-8")
    out = publish_mod.publish("agent", "https://example.invalid/api")
    assert out is not None and out["status"] == 200 and out["response"] == {"ok": True}
    assert "published agent to https://example.invalid/api" in out["summary"]
    assert net.opened == ["https://example.invalid/api"]


# ── compare-agents ────────────────────────────────────────────────

def test_compare_agents_has_no_default_url_anywhere():
    assert inspect.signature(compare.compare_agents).parameters["api_url"].default is None
    assert cli._build_parser().parse_args(["compare-agents"]).api_url is None


def test_compare_agents_without_a_url_fetches_nothing(net):
    out = compare.compare_agents(_fingerprint())
    assert net.opened == []
    assert out.n_agents == 0 and out.status == "no_url"
    assert "no default leaderboard" in out.narrative
    assert "publish to it" not in out.render()


def test_a_leaderboard_that_cannot_be_read_is_not_an_empty_one(net):
    net.body = b"<!doctype html><html><title>parked</title></html>"
    out = compare.compare_agents(_fingerprint(), api_url="https://example.invalid/board")
    assert net.opened == ["https://example.invalid/board"]
    assert out.n_agents == 0 and out.status == "unreadable"
    assert "could not be read" in out.narrative and "example.invalid" in out.narrative
    assert "publish to it" not in out.render() and "no agents yet" not in out.render()


def test_an_empty_leaderboard_says_so_and_names_the_publish_flags(net):
    net.body = b'{"agents": []}'
    out = compare.compare_agents(_fingerprint(), api_url="https://example.invalid/board")
    assert out.status == "empty"
    text = out.render()
    assert "no agents yet" in text and "--endpoint" in text
    assert out.as_dict()["status"] == "empty"


def test_a_population_is_compared(net):
    net.body = json.dumps({"agents": [{"fingerprint": {"reasoning": 0.5, "retrieval": 0.1}},
                                      {"fingerprint": {"reasoning": 0.4}}]}).encode("utf-8")
    out = compare.compare_agents(_fingerprint(), api_url="https://example.invalid/board")
    assert out.status == "ok" and out.n_agents == 2
    assert out.percentiles["reasoning"] == 100.0


@pytest.mark.parametrize("body, want", [
    (b"<!doctype html><html>parked</html>", None),
    (b"{}", None),
    (b'{"agents": "many"}', None),
    (b"[]", None),
    (b'{"agents": []}', []),
    (b'{"agents": [{"fingerprint": {"reasoning": 0.5}}, {"fingerprint": {}}, 7]}', "one"),
])
def test_fetch_population_tells_unreadable_from_empty(net, body, want):
    net.body = body
    got = compare._fetch_population("https://example.invalid/board")
    if want == "one":
        assert isinstance(got, list) and len(got) == 1
    else:
        assert got == want
    assert net.opened == ["https://example.invalid/board"]


def test_fetch_population_is_none_when_the_host_does_not_answer(net):
    assert compare._fetch_population("https://example.invalid/board") is None
    assert net.opened == ["https://example.invalid/board"]


def test_compare_agents_without_a_fingerprint_says_so_and_fetches_nothing(monkeypatch, net):
    monkeypatch.setattr(analytics, "fingerprint", lambda *a, **k: None)
    out = compare.compare_agents(api_url="https://example.invalid/board")
    assert out.status == "no_fingerprint" and net.opened == []


@pytest.mark.parametrize("value", [b"9" * 400, b"NaN", b"Infinity", b"-Infinity", b"1e400"],
                         ids=["integer_of_400_digits", "nan", "infinity", "minus_infinity", "1e400"])
def test_a_fingerprint_value_no_finite_float_holds_is_not_a_vector(net, value):
    """A 400-digit integer ended the comparison in an OverflowError through 8.0.0, and NaN or
    Infinity, which Python's JSON reader takes, came out as a population mean."""
    net.body = b'{"agents": [{"fingerprint": {"retrieval": 0.5, "reasoning": ' + value + b"}}]}"
    assert compare._fetch_population("https://example.invalid/board") == []
    assert compare.compare_agents(_fingerprint(), api_url="https://example.invalid/board").status == "empty"


def test_compare_agents_command_end_to_end(monkeypatch, net, capsys):
    """The real command, not a stand-in for it: what a user sees with no URL and with a web page."""
    monkeypatch.setattr(analytics, "fingerprint", lambda *a, **k: _fingerprint())
    assert cli.main(["compare-agents"]) == 0
    out = capsys.readouterr().out
    assert net.opened == []
    assert "no default leaderboard" in out and "publish to it" not in out
    net.body = b"<!doctype html><html><title>parked</title></html>"
    assert cli.main(["compare-agents", "--api-url", "https://example.invalid/board"]) == 0
    out = capsys.readouterr().out
    assert net.opened == ["https://example.invalid/board"]
    assert "could not be read" in out and "publish to it" not in out
    assert cli.main(["compare-agents", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "no_url"


# ── claim and feed ────────────────────────────────────────────────

def test_stream_has_no_default_relay_or_dashboard():
    assert stream.DEFAULT_RELAY is None and stream.DEFAULT_DASHBOARD_BASE is None
    assert stream.relay_url() is None
    assert stream.dashboard_url("agent") == ""
    assert cli._build_parser().parse_args(["claim", "agent"]).relay is None


def test_relay_and_dashboard_come_from_the_caller_or_the_environment(monkeypatch):
    assert stream.relay_url("https://example.invalid/ingest") == "https://example.invalid/ingest"
    monkeypatch.setenv("STYXX_STREAM_RELAY", "https://env.invalid/ingest")
    monkeypatch.setenv("STYXX_DASHBOARD_BASE", "https://env.invalid")
    assert stream.relay_url() == "https://env.invalid/ingest"
    assert stream.dashboard_url("My Agent") == "https://env.invalid/a/my-agent"


@pytest.mark.parametrize("relay, want", [
    ("https://relay.example.invalid/ingest", "https://relay.example.invalid/claim"),
    ("https://relay.example.invalid", "https://relay.example.invalid/claim"),      # was https://claim
    ("https://relay.example.invalid/", "https://relay.example.invalid/claim"),
    ("https://relay.example.invalid/v1/ingest", "https://relay.example.invalid/v1/claim"),
    ("http://127.0.0.1:8787/ingest", "http://127.0.0.1:8787/claim"),
    ("https://relay.example.invalid/ingest?x=1#y", "https://relay.example.invalid/claim"),
])
def test_the_claim_url_is_on_the_relay_the_caller_named(relay, want):
    assert stream.claim_url(relay) == want


@pytest.mark.parametrize("relay", ["relay.example.invalid/ingest", "ftp://relay.example.invalid/x", "",
                                   "https://", "claim"])
def test_a_relay_that_is_not_a_url_is_refused(net, capsys, relay):
    with pytest.raises(stream.ClaimError):
        stream.claim_url(relay)
    if relay:
        with pytest.raises(stream.ClaimError):
            stream.claim_agent("agent", relay=relay)
        assert cli.main(["claim", "agent", "--relay", relay]) == 1
        captured = capsys.readouterr()
        assert "reserving subdomain" not in captured.out and "registered" not in captured.out
    assert net.opened == []


def test_claim_without_a_relay_raises_and_sends_nothing(net):
    with pytest.raises(stream.ClaimError) as e:
        stream.claim_agent("agent")
    assert "no default relay" in str(e.value)
    assert net.opened == []


def test_claim_command_without_a_relay_fails_before_any_step(net, capsys):
    assert cli.main(["claim", "agent"]) == 1
    captured = capsys.readouterr()
    assert net.opened == []
    assert "no default relay" in captured.err and "--relay" in captured.err
    assert "reserving subdomain" not in captured.out and "registered" not in captured.out


@pytest.mark.parametrize("relay", ["https://relay.example.invalid/ingest", "https://relay.example.invalid"])
def test_claim_posts_the_name_to_the_relay_and_to_no_other_host(net, relay):
    net.body = json.dumps({"name": "agent", "key": "k"}).encode("utf-8")
    out = stream.claim_agent("agent", relay=relay)
    assert net.opened == ["https://relay.example.invalid/claim"]
    assert json.loads(net.bodies[0]) == {"name": "agent"}
    assert out == {"name": "agent", "key": "k", "url": ""}


@pytest.mark.parametrize("body", [
    b"[]",
    b'["key", "name"]',                                  # holds both words and is no object
    b'"key name"',
    b'{"name": null, "key": "k"}',
    b'{"name": ["agent"], "key": "k"}',                  # a traceback through 8.0.0
    b'{"name": {"a": 1}, "key": "k"}',
    b'{"name": "_default", "key": "k"}',                 # would overwrite the file's own pointer
    b'{"name": "Agent\\u001b[2J", "key": "k"}',           # not a name styxx can look up or print
    b'{"name": "", "key": "k"}',
    b'{"name": "agent", "key": null}',
    b'{"name": "agent", "key": ""}',
    b'{"name": "agent", "key": {"k": 1}}',
    b"[" * 100000 + b"]" * 100000,                       # deeper than the JSON reader goes
], ids=["empty_list", "list_of_the_words", "string_of_the_words", "null_name", "list_name",
        "object_name", "default_name", "escape_in_name", "empty_name", "null_key", "empty_key",
        "object_key", "nested_deep"])
def test_claim_with_a_reply_that_is_not_a_name_and_a_key_is_refused_and_stores_nothing(monkeypatch, net, body):
    saved = []
    monkeypatch.setattr(stream, "_save_creds", saved.append)
    net.body = body
    with pytest.raises(stream.ClaimError):
        stream.claim_agent("agent", relay="https://relay.example.invalid/ingest")
    assert saved == []
    assert net.opened == ["https://relay.example.invalid/claim"]


def test_claim_stores_the_key_beside_the_relay_that_gave_it(monkeypatch, net):
    saved = []
    monkeypatch.setattr(stream, "_save_creds", saved.append)
    net.body = json.dumps({"name": "agent", "key": "k"}).encode("utf-8")
    stream.claim_agent("agent", relay="https://relay.example.invalid/ingest")
    assert len(saved) == 1 and saved[0]["_default"] == "agent"
    assert saved[0]["agent"]["key"] == "k"
    assert saved[0]["agent"]["relay"] == "https://relay.example.invalid/ingest"


@pytest.mark.parametrize("link, want", [
    ("https://dash.example.invalid/a/agent", "https://dash.example.invalid/a/agent"),
    ("javascript:alert(1)", ""),
    ("https://dash.example.invalid/\u001b[2J", ""),
    (7, ""),
])
def test_a_dashboard_address_from_the_relay_is_kept_only_if_it_is_a_web_address(net, link, want):
    net.body = json.dumps({"name": "agent", "key": "k", "url": link}).encode("utf-8")
    assert stream.claim_agent("agent", relay="https://relay.example.invalid/ingest")["url"] == want


def test_claim_does_not_fall_back_to_a_relay_an_earlier_claim_stored(monkeypatch, net, capsys):
    """A credentials file written by 8.0.0 holds the relay that was the default then. Neither the
    command nor the function reads it: with no relay named now, nothing is sent."""
    monkeypatch.setattr(stream, "_load_creds", lambda: {
        "agent": {"key": "k", "relay": OLD_RELAY, "dashboard": "https://live.old-default.invalid/a/agent"},
        "_default": "agent"})
    assert cli.main(["claim", "agent"]) == 1
    captured = capsys.readouterr()
    assert net.opened == []
    assert "old-default" not in captured.out + captured.err and "no default relay" in captured.err
    with pytest.raises(stream.ClaimError):
        stream.claim_agent("agent")
    assert net.opened == []
    # and a claim that names a relay prints that relay as the next step, not the one stored
    net.body = json.dumps({"name": "agent", "key": "k"}).encode("utf-8")
    assert cli.main(["claim", "agent", "--relay", "https://relay.example.invalid/ingest"]) == 0
    out = capsys.readouterr().out
    assert "STYXX_STREAM_RELAY=https://relay.example.invalid/ingest" in out and "old-default" not in out
    assert net.opened == ["https://relay.example.invalid/claim"]


def test_claim_command_prints_a_next_step_that_names_the_relay(net, capsys):
    """The next step used to be `autoboot(stream=True)` alone, which now leaves the feed off:
    the emitter needs the relay named, and the command says how."""
    net.body = json.dumps({"name": "agent", "key": "k"}).encode("utf-8")
    assert cli.main(["claim", "agent", "--relay", "https://relay.example.invalid/ingest"]) == 0
    out = capsys.readouterr().out
    assert "STYXX_STREAM_RELAY=https://relay.example.invalid/ingest" in out
    assert "(none: the relay gave no dashboard address that is an http or https URL)" in out


def test_feed_command_without_a_dashboard_base_fails(capsys):
    assert cli.main(["feed", "agent"]) == 1
    captured = capsys.readouterr()
    assert "no default dashboard" in captured.err and captured.out == ""


def test_feed_command_with_no_name_claimed_names_a_command_that_can_work(capsys):
    """It used to say `run: styxx claim <name>`, which exits 1 now that there is no default relay."""
    assert cli.main(["feed"]) == 1
    captured = capsys.readouterr()
    assert "styxx claim NAME --relay URL" in captured.err and captured.out == ""


def test_feed_command_with_a_dashboard_base_prints_the_url(monkeypatch, capsys):
    monkeypatch.setenv("STYXX_DASHBOARD_BASE", "https://dash.example.invalid")
    assert cli.main(["feed", "agent"]) == 0
    assert capsys.readouterr().out.strip() == "https://dash.example.invalid/a/agent"


# ── the stream emitter ────────────────────────────────────────────

def test_the_emitter_stays_off_without_a_relay(net, no_thread):
    assert stream.enable(agent="agent") is False
    assert stream.is_enabled() is False and stream._relay is None
    assert no_thread.started == 0 and net.opened == []


def test_a_relay_stored_with_a_claimed_name_is_not_used(monkeypatch, net, no_thread):
    """A credentials file written by an earlier styxx holds the relay that was the default then.
    The emitter does not read it: with no relay named now, nothing starts."""
    monkeypatch.setattr(stream, "_load_creds", lambda: {
        "agent": {"key": "k", "relay": OLD_RELAY, "dashboard": "https://live.old-default.invalid/a/agent"},
        "_default": "agent"})
    assert stream.enable() is False
    assert stream.enable(agent="agent") is False
    assert stream.is_enabled() is False and stream._relay is None
    assert no_thread.started == 0 and net.opened == []


def test_the_emitter_starts_with_a_relay_and_keeps_it(no_thread):
    assert stream.enable(agent="agent", relay="https://relay.example.invalid/ingest") is True
    assert stream._relay == "https://relay.example.invalid/ingest" and no_thread.started == 1
    # a later call that names no relay keeps the one in use and says the truth about it
    assert stream.enable(agent="agent") is True
    assert stream.is_enabled() is True and stream._relay == "https://relay.example.invalid/ingest"
    stream.disable()
    assert stream.is_enabled() is False


def test_the_emitter_refuses_a_relay_that_is_not_a_url(no_thread):
    assert stream.enable(agent="agent", relay="relay.example.invalid/ingest") is False
    assert stream.is_enabled() is False and no_thread.started == 0


def test_post_sends_nothing_without_a_relay(net):
    assert stream._post({"agent": "agent"}) is False
    assert net.opened == []


def test_post_sends_to_the_relay_named(monkeypatch, net):
    monkeypatch.setattr(stream, "_relay", "https://relay.example.invalid/ingest")
    net.body = b"{}"
    assert stream._post({"agent": "agent"}) is True
    assert net.opened == ["https://relay.example.invalid/ingest"]


_AUTOBOOT_DRIVER = r'''
import json, os, sys
sys.meta_path = [f for f in sys.meta_path if "editable" not in type(f).__module__.lower()
                 and "editable" not in repr(f).lower()]
sys.path.insert(0, sys.argv[1])
import socket, urllib.request
attempts = []
def _open(self, req, *a, **k):
    attempts.append(getattr(req, "full_url", str(req)))
    raise OSError("no network in tests")
def _connect(self, address, *a, **k):
    attempts.append(repr(address))
    raise OSError("no network in tests")
def _lookup(host, *a, **k):
    attempts.append("lookup " + repr(host))
    raise OSError("no network in tests")
urllib.request.OpenerDirector.open = _open      # urlopen and styxx.stream.open_url both end here
socket.socket.connect = _connect
socket.getaddrinfo = _lookup
import styxx
assert os.path.abspath(styxx.__file__).startswith(os.path.abspath(sys.argv[1])), styxx.__file__
import importlib
stream = importlib.import_module("styxx.stream")
autoboot = importlib.import_module("styxx.autoboot")
result = autoboot.autoboot(agent_name="agent", print_weather=False, print_diff=False)
stream._post({"agent": "agent"})
print(json.dumps({"stream_enabled": result.get("stream_enabled"), "stream_url": result.get("stream_url"),
                  "already_booted": result.get("already_booted"), "is_enabled": stream.is_enabled(),
                  "relay": stream._relay, "attempts": attempts}))
'''


def _autoboot_in_a_fresh_process(tmp_path, extra_env):
    """autoboot keeps a booted flag for the life of a process, so its stream branch is run in a
    fresh one, with a home that holds a credentials file an earlier styxx could have written."""
    home = tmp_path / "home"
    (home / ".styxx").mkdir(parents=True)
    (home / ".styxx" / "credentials.json").write_text(json.dumps({
        "agent": {"key": "k", "relay": OLD_RELAY, "dashboard": "https://live.old-default.invalid/a/agent"},
        "_default": "agent"}), encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text(_AUTOBOOT_DRIVER, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("STYXX_")}
    env.update({"HOME": str(home), "USERPROFILE": str(home), "STYXX_DATA_DIR": str(tmp_path / "data"),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}, **extra_env)
    r = subprocess.run([sys.executable, "-X", "utf8", "-B", str(driver), str(ROOT)], env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1]), r.stderr


def test_autoboot_with_the_stream_asked_for_and_no_relay_sends_nothing_and_says_so(tmp_path):
    out, err = _autoboot_in_a_fresh_process(tmp_path, {"STYXX_STREAM": "on"})
    assert out["already_booted"] is None
    assert out["stream_enabled"] is False and out["is_enabled"] is False and out["relay"] is None
    assert out["attempts"] == [], "a connection was attempted"
    assert "live feed: off" in err and "STYXX_STREAM_RELAY" in err


def test_autoboot_with_the_relay_named_in_the_environment_turns_the_stream_on(tmp_path):
    out, err = _autoboot_in_a_fresh_process(
        tmp_path, {"STYXX_STREAM": "on", "STYXX_STREAM_RELAY": "https://relay.example.invalid/ingest"})
    assert out["stream_enabled"] is True and out["relay"] == "https://relay.example.invalid/ingest"
    assert out["stream_url"] is None, "no dashboard base is set"
    # the driver posts one packet itself, so the list is not empty and the check below is not idle
    assert "https://relay.example.invalid/ingest" in out["attempts"]
    assert all("relay.example.invalid" in a for a in out["attempts"]), out["attempts"]
    assert "live feed: on (no dashboard base set)" in err


# ── links ─────────────────────────────────────────────────────────

# Every line under styxx/ that names the domain, in any spelling. Five are identifiers of
# documents (not addresses the package opens) and are left as they are; three are sentences that
# say what happened. Anything else that names it fails this test.
_ALLOWED_LINES = {
    ("styxx/provenance.py", 'SCHEMA_URI = "https://fathom.darkflobi.com/schemas/cognitive-provenance/v1"'),
    ("styxx/schema.py", '"$id": "https://fathom.darkflobi.com/styxx/schema/vitals.json",'),
    ("styxx/schema.py", '"$id": "https://fathom.darkflobi.com/styxx/schema/verification.json",'),
    ("styxx/schema.py", '"$id": "https://fathom.darkflobi.com/styxx/schema/error.json",'),
    ("styxx/schema.py", '"$id": "https://fathom.darkflobi.com/styxx/schema/thought.json",'),
    ("styxx/publish.py", "host under darkflobi.com, where the public dashboard lived. on"),
    ("styxx/compare.py", "under darkflobi.com; on 2026-11-12 that host answered from a"),
    ("styxx/stream.py", "8.0.0 both were hosts under darkflobi.com, where the public relay"),
}
_TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".html", ".js", ".yaml", ".yml", ".toml", ".cfg", ".css"}


def test_no_file_of_the_package_names_the_parked_domain_except_the_listed_lines():
    found = set()
    for path in sorted((ROOT / "styxx").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line in text.splitlines():
            if re.search(r"darkflobi\s*\.\s*com", line, re.I):
                found.add((rel, line.strip()))
    assert found == _ALLOWED_LINES, sorted(found ^ _ALLOWED_LINES)


@pytest.mark.parametrize("rel", [
    "README.md", "docs/REFERENCE.md", "openapi/styxx-v1.yaml", ".well-known/agent-card.json",
    "docs/agents/peer-recommendation.md", "packages/styxx-js/package.json", "submissions/README.md",
])
def test_the_files_corrected_outside_the_package_do_not_name_the_parked_domain(rel):
    """The links of these seven files were moved off the domain with the package's own. Other
    tracked files outside the package still name it and are listed in the CHANGELOG."""
    path = ROOT / rel
    if not path.is_file():
        pytest.skip(f"{rel} is not in this tree (a source distribution ships the tests without it)")
    hits = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if re.search(r"darkflobi\s*\.\s*com", line, re.I)]
    assert hits == []


def test_the_package_url_is_the_project_homepage(monkeypatch):
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    home = re.search(r'^Homepage\s*=\s*"([^"]+)"', text, re.M).group(1)
    assert styxx.__url__ == home == "https://github.com/fathom-lab/styxx"
    community = importlib.import_module("styxx.community")
    assert community.HOMEPAGE == home
    # `styxx.agent_card` is a public function and also the name of a submodule the package does
    # not load at import: importing the submodule would replace the function with the module for
    # every later test. Pinning the attribute has monkeypatch put the function back.
    monkeypatch.setattr(styxx, "agent_card", styxx.agent_card)
    card = importlib.import_module("styxx.agent_card").build_agent_card()
    assert card["url"] == card["documentationUrl"] == home
    assert card["provider"]["url"] == "https://github.com/fathom-lab"
    committed = json.loads((ROOT / ".well-known" / "agent-card.json").read_text(encoding="utf-8"))
    assert committed["url"] == committed["documentationUrl"] == home
    assert committed["provider"]["url"] == "https://github.com/fathom-lab"


# ── added after the second review ─────────────────────────────────

@pytest.mark.parametrize("url", [
    "https://user:secret@relay.example.invalid/ingest",       # http.client refuses the port it reads
    "https://relay.example.invalid:port/ingest",
    "https://relay.example.invalid /ingest",
    "https://relay.example.invalid/in\ngest",
    "https://relay.example.invalid/in\x7fgest",                # DEL, and a C1 control below
    "https://relay.example.invalid/in\x85gest",
])
def test_a_url_the_http_client_would_refuse_is_refused_before_it_gets_there(monkeypatch, net, capsys, url):
    """The four at the top of the list each passed the scheme-and-host check of an earlier draft
    and ended `styxx claim` in a traceback from http.client after its two step lines. (Until the third
    review this list held a URL with a space before it. The http client does not refuse that
    one: urllib drops the space, and 8.0.0 published to such an endpoint. See
    test_spaces_around_a_publish_endpoint_are_dropped.)"""
    assert not stream.is_http_url(url) and not publish_mod._is_http_url(url)
    with pytest.raises(stream.ClaimError):
        stream.claim_url(url)
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a, **k: {"agent_name": "agent"})
    assert publish_mod.publish("agent", url) is None
    assert net.opened == []
    capsys.readouterr()
    assert cli.main(["claim", "agent", "--relay", url]) == 1
    captured = capsys.readouterr()
    assert "reserving subdomain" not in captured.out and "secret" not in captured.out
    assert net.opened == []


def test_publish_and_the_stream_check_a_url_with_one_function():
    assert publish_mod._is_http_url is stream.is_http_url


@pytest.mark.parametrize("pad", [" {}", "{} ", "{}\n", "{}\r\n", "\t{}\t"],
                         ids=["space_before", "space_after", "line_feed", "cr_lf", "tabs"])
def test_spaces_around_a_publish_endpoint_are_dropped(monkeypatch, net, pad):
    """8.0.0 published to an endpoint given with a space or a line end around it (urllib drops
    them), and the stream of this change does. An earlier commit of this change refused it in
    publish alone, and its change log said such a value used to end in a traceback."""
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a, **k: {"agent_name": "agent"})
    net.body = b'{"ok": true}'
    out = publish_mod.publish("agent", pad.format("https://example.invalid/api"))
    assert net.opened == ["https://example.invalid/api"]
    assert out is not None and "published agent to https://example.invalid/api (" in out["summary"]


def test_an_endpoint_of_spaces_alone_is_no_endpoint(monkeypatch, net, capsys):
    collected = []
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a, **k: collected.append(a) or {})
    assert publish_mod.publish("agent", " \n") is None
    assert net.opened == [] and collected == []
    assert "no default publish endpoint" in capsys.readouterr().err


def test_spaces_around_a_relay_are_dropped_and_the_claim_goes_to_that_relay(net):
    net.body = json.dumps({"name": "agent", "key": "k"}).encode("utf-8")
    assert stream.relay_url("  https://relay.example.invalid \n") == "https://relay.example.invalid"
    stream.claim_agent("agent", relay="  https://relay.example.invalid \n")
    assert net.opened == ["https://relay.example.invalid/claim"]


def test_an_exception_of_the_http_client_is_a_claim_error(monkeypatch):
    import http.client
    import urllib.request

    def refuse(*a, **k):
        raise http.client.InvalidURL("nonnumeric port")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", refuse)
    with pytest.raises(stream.ClaimError):
        stream.claim_agent("agent", relay="https://relay.example.invalid/ingest")


def test_claim_command_takes_the_relay_from_the_environment(monkeypatch, net, capsys):
    monkeypatch.setenv("STYXX_STREAM_RELAY", "https://relay.example.invalid/ingest")
    net.body = json.dumps({"name": "agent", "key": "k"}).encode("utf-8")
    assert cli.main(["claim", "agent"]) == 0
    out = capsys.readouterr().out
    assert "STYXX_STREAM_RELAY=https://relay.example.invalid/ingest" in out and "None" not in out
    assert net.opened == ["https://relay.example.invalid/claim"]
    monkeypatch.setenv("STYXX_STREAM_RELAY", "relay.example.invalid/ingest")
    assert cli.main(["claim", "agent"]) == 1
    captured = capsys.readouterr()
    assert "reserving subdomain" not in captured.out and "not an http or https URL" in captured.err
    assert net.opened == ["https://relay.example.invalid/claim"], "nothing more was opened"


def test_feed_command_does_not_print_a_url_an_earlier_claim_stored(monkeypatch, capsys):
    """A credentials file written by an earlier styxx holds a dashboard URL built from the default
    base of the day. `styxx feed` builds its URL from the base named now, or prints none."""
    monkeypatch.setattr(stream, "_load_creds", lambda: {
        "agent": {"key": "k", "relay": OLD_RELAY, "dashboard": "https://live.old-default.invalid/a/agent"},
        "_default": "agent"})
    assert cli.main(["feed"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "old-default" not in captured.err


def test_a_dashboard_base_with_a_final_slash_gives_one_slash(monkeypatch):
    monkeypatch.setenv("STYXX_DASHBOARD_BASE", "https://dash.example.invalid/")
    assert stream.dashboard_url("agent") == "https://dash.example.invalid/a/agent"


@pytest.mark.parametrize("body", [
    b'{"agents": [{"fingerprint": {"retrieval": 0.5, "reasoning": null}}]}',
    b'{"agents": [{"fingerprint": {"retrieval": 0.5, "reasoning": [1]}}]}',
    b'{"agents": [{"fingerprint": {"retrieval": 0.5, "reasoning": true}}]}',
    b'{"agents": [{"fingerprint": {"retrieval": "0.5"}}]}',
])
def test_a_fingerprint_with_a_value_that_is_not_a_number_is_not_a_vector(net, body):
    """A null after a positive value used to be kept and ended the comparison in a TypeError."""
    net.body = body
    assert compare._fetch_population("https://example.invalid/board") == []
    out = compare.compare_agents(_fingerprint(), api_url="https://example.invalid/board")
    assert out.status == "empty"


@pytest.mark.parametrize("toggle", ["1", "true", "yes", "public"])
def test_autoboot_sends_nothing_under_every_spelling_of_the_toggle(tmp_path, toggle):
    out, err = _autoboot_in_a_fresh_process(tmp_path, {"STYXX_STREAM": toggle})
    assert out["stream_enabled"] is False and out["relay"] is None and out["attempts"] == []
    assert "live feed: off" in err


def test_autoboot_says_so_when_the_relay_named_is_not_a_url(tmp_path):
    out, err = _autoboot_in_a_fresh_process(
        tmp_path, {"STYXX_STREAM": "on", "STYXX_STREAM_RELAY": "relay.example.invalid/ingest"})
    assert out["stream_enabled"] is False and out["relay"] is None and out["attempts"] == []
    assert "live feed: off" in err and "not an http or https URL" in err


# ── no redirect is followed (added after the third review) ────────

class _Origin(http.server.BaseHTTPRequestHandler):
    """Answers a path that starts /r301, /r302, /r303, /r307 or /r308 with that redirect to the
    server's `target`, and anything else with 200 and the server's `reply`. Notes every request."""
    protocol_version = "HTTP/1.0"

    def _any(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        self.server.seen.append({"method": self.command, "path": self.path, "body": body,
                                 "key": self.headers.get("X-Styxx-Key")})
        head = self.path.strip("/").split("/")[0]
        if head[:1] == "r" and head[1:].isdigit():
            self.send_response(int(head[1:]))
            self.send_header("Location", self.server.target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.server.reply)))
        self.end_headers()
        self.wfile.write(self.server.reply)

    do_GET = do_POST = _any

    def log_message(self, *a):
        pass


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    connections = 0                 # every connection taken, a request read from it or not

    def get_request(self):
        self.connections += 1
        return super().get_request()

    def server_bind(self):          # the stock one asks the resolver for this machine's name
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}{path}"


_BOARD = [{"fingerprint": {"reasoning": 0.9}}]
REDIRECTS = [301, 302, 303, 307, 308]


@pytest.fixture
def origins(monkeypatch):
    """Two servers on 127.0.0.1. `a` is the address a test names; `b` is where `a` points when
    asked for a redirect. No proxy: urllib must reach 127.0.0.1 itself, whatever the machine or
    its environment is set to."""
    for name in list(os.environ):
        if name.lower() in ("http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "no_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NO_PROXY", "*")
    servers = []
    for label in ("a", "b"):
        srv = _Server(("127.0.0.1", 0), _Origin)
        srv.seen = []
        srv.reply = json.dumps({"name": "agent" if label == "a" else "from-b", "ok": True,
                                "key": f"KEY-FROM-{label.upper()}", "agents": _BOARD}).encode("utf-8")
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
    a, b = servers
    a.target = b.url("/landed")
    b.target = a.url("/back")
    try:
        yield a, b
    finally:
        for srv in servers:
            srv.shutdown()
            srv.server_close()


def test_the_two_servers_can_show_a_followed_redirect(origins):
    """What these tests would see if a redirect were followed, shown with urllib's stock opener,
    which follows one: the second server receives the request, key and all. Without this, an
    empty `b.seen` below could mean only that nothing can reach `b`."""
    import urllib.request
    a, b = origins
    req = urllib.request.Request(a.url("/r302/ingest"), data=b"{}", method="POST",
                                 headers={"X-Styxx-Key": "KEY", "Content-Type": "application/json"})
    with urllib.request.build_opener().open(req, timeout=10) as resp:      # as urlopen opens
        assert resp.status == 200
    assert [(r["method"], r["key"]) for r in a.seen] == [("POST", "KEY")]
    assert [(r["method"], r["path"], r["key"], r["body"]) for r in b.seen] == [("GET", "/landed", "KEY", b"")]


@pytest.mark.parametrize("code", REDIRECTS)
def test_a_redirect_from_the_relay_is_a_failed_post_and_the_key_goes_nowhere_else(origins, monkeypatch, code):
    a, b = origins
    monkeypatch.setattr(stream, "_relay", a.url(f"/r{code}/ingest"))
    monkeypatch.setattr(stream, "_key", "KEY-OF-THE-USER")
    with pytest.warns(RuntimeWarning) as caught:
        assert stream._post({"agent": "agent", "v": 1}) is False
    assert b.seen == [], "the address the relay pointed to received a request"
    assert [(r["method"], r["path"], r["key"]) for r in a.seen] == [("POST", f"/r{code}/ingest", "KEY-OF-THE-USER")]
    assert json.loads(a.seen[0]["body"]) == {"agent": "agent", "v": 1}
    text = " ".join(str(w.message) for w in caught)
    assert f"HTTP {code}" in text and b.url("/landed") in text and "does not follow a redirect" in text
    assert "KEY-OF-THE-USER" not in text and b.connections == 0


def test_a_post_the_relay_takes_is_delivered_to_it_alone(origins, monkeypatch):
    a, b = origins
    monkeypatch.setattr(stream, "_relay", a.url("/ok/ingest"))
    monkeypatch.setattr(stream, "_key", "KEY-OF-THE-USER")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert stream._post({"agent": "agent"}) is True
    assert [(r["method"], r["key"]) for r in a.seen] == [("POST", "KEY-OF-THE-USER")] and b.seen == []


def test_a_redirect_to_the_relay_itself_is_not_followed_either(origins, monkeypatch):
    """urllib would re-send the POST as a GET with no body and count the 200 as delivered."""
    a, b = origins
    a.target = a.url("/ok/ingest")
    monkeypatch.setattr(stream, "_relay", a.url("/r302/ingest"))
    with pytest.warns(RuntimeWarning):
        assert stream._post({"agent": "agent"}) is False
    assert [(r["method"], r["path"]) for r in a.seen] == [("POST", "/r302/ingest")] and b.seen == []


def test_a_redirect_does_not_end_the_worker_where_warnings_are_errors(origins, monkeypatch):
    a, b = origins
    monkeypatch.setattr(stream, "_relay", a.url("/r302/ingest"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert stream._post({"agent": "agent"}) is False
    assert b.seen == []


@pytest.mark.parametrize("code", REDIRECTS)
def test_a_redirect_from_the_relay_fails_the_claim_and_stores_nothing(origins, monkeypatch, capsys, code):
    """Through 8.0.0 the name and the key a second address answered with were stored beside the
    relay the caller named."""
    a, b = origins
    saved = []
    monkeypatch.setattr(stream, "_save_creds", saved.append)
    with pytest.raises(stream.ClaimError) as e:
        stream.claim_agent("agent", relay=a.url(f"/r{code}/ingest"))
    assert f"HTTP {code}" in str(e.value) and b.url("/landed") in str(e.value)
    assert b.seen == [] and saved == []
    assert [(r["method"], r["path"]) for r in a.seen] == [("POST", f"/r{code}/claim")]
    assert json.loads(a.seen[0]["body"]) == {"name": "agent"}
    assert cli.main(["claim", "agent", "--relay", a.url(f"/r{code}/ingest")]) == 1
    captured = capsys.readouterr()
    assert "redirect" in captured.err and "registered" not in captured.out
    assert b.seen == [] and saved == []


def test_a_claim_the_relay_takes_is_stored(origins, monkeypatch):
    a, b = origins
    saved = []
    monkeypatch.setattr(stream, "_save_creds", saved.append)
    out = stream.claim_agent("agent", relay=a.url("/ok/ingest"))
    assert out["name"] == "agent" and out["key"] == "KEY-FROM-A"
    assert len(saved) == 1 and b.seen == []


@pytest.mark.parametrize("code", REDIRECTS)
def test_a_redirect_from_the_endpoint_is_not_published(origins, monkeypatch, capsys, code):
    """Through 8.0.0 the reply of the second address was reported as the endpoint's: `published
    NAME to ENDPOINT (HTTP 200)`."""
    a, b = origins
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a_, **k: {"agent_name": "agent"})
    assert publish_mod.publish("agent", a.url(f"/r{code}/submit")) is None
    err = capsys.readouterr().err
    assert f"HTTP {code}" in err and b.url("/landed") in err and "not counted as published" in err
    assert b.seen == []
    assert [(r["method"], r["path"]) for r in a.seen] == [("POST", f"/r{code}/submit")]
    assert cli.main(["publish", "--name", "agent", "--endpoint", a.url(f"/r{code}/submit")]) == 1
    assert b.seen == []


def test_a_payload_the_endpoint_takes_is_published(origins, monkeypatch):
    a, b = origins
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a_, **k: {"agent_name": "agent"})
    out = publish_mod.publish("agent", a.url("/ok/submit"))
    assert out is not None and out["status"] == 200 and out["response"]["ok"] is True
    assert [(r["method"], r["path"]) for r in a.seen] == [("POST", "/ok/submit")] and b.seen == []


@pytest.mark.parametrize("code", REDIRECTS)
def test_a_redirect_from_the_leaderboard_is_not_read_as_its_answer(origins, code):
    a, b = origins
    out = compare.compare_agents(_fingerprint(), api_url=a.url(f"/r{code}/board"))
    assert out.status == "unreadable" and out.n_agents == 0
    assert f"HTTP {code}" in out.narrative and b.url("/landed") in out.narrative
    assert compare._fetch_population(a.url(f"/r{code}/board")) is None
    assert b.seen == [] and [r["method"] for r in a.seen] == ["GET", "GET"]


def test_a_leaderboard_that_answers_is_read(origins):
    a, b = origins
    out = compare.compare_agents(_fingerprint(), api_url=a.url("/ok/board"))
    assert out.status == "ok" and out.n_agents == 1 and b.seen == []


def _http_error(code, location):
    import email.message
    headers = email.message.Message()
    if location is not None:
        headers["Location"] = location
    return urllib.error.HTTPError("http://relay.example.invalid/ingest", code, "x", headers, None)


def test_the_address_of_a_redirect_is_printed_quoted_and_cut():
    assert stream.redirect_target(_http_error(404, "http://other.example.invalid/")) is None
    assert stream.redirect_target(_http_error(500, None)) is None
    assert stream.redirect_target(OSError("no")) is None
    assert stream.redirect_target(_http_error(302, None)) == "(none given)"
    assert stream.redirect_target(_http_error(302, "http://other.example.invalid/x")) == "'http://other.example.invalid/x'"
    shown = stream.redirect_target(_http_error(301, "http://other.example.invalid/\x1b[2J" + "a" * 500))
    assert "\x1b" not in shown and "\\x1b[2J" in shown, "an escape sequence is shown, not sent to the terminal"
    assert len(shown) < stream._LOCATION_MAX + 20 and shown.endswith("...'")


# ── added after the review of the redirect repair ─────────────────

def _post_under(monkeypatch, origins, path="/r302/ingest"):
    a, b = origins
    monkeypatch.setattr(stream, "_relay", a.url(path))
    monkeypatch.setattr(stream, "_key", "KEY-OF-THE-USER")
    return a, b


@pytest.mark.parametrize("target", ["other_name", "https", "relative", "unparsable"])
def test_no_form_of_address_is_followed(origins, monkeypatch, target):
    """A handler that followed only to another host name, only to an https address or only a
    relative address passed every redirect test of an earlier commit (a review made each). And
    for an address urllib cannot parse, the stock handler raised ValueError before it was asked
    anything, so the stream said nothing at all."""
    a, b = _post_under(monkeypatch, origins)
    a.target = {"other_name": f"http://localhost:{b.server_address[1]}/landed",
                "https": f"https://127.0.0.1:{b.server_address[1]}/landed",
                "relative": "/ok/ingest",
                "unparsable": "http://[::1/landed"}[target]
    with pytest.warns(RuntimeWarning) as caught:
        assert stream._post({"agent": "agent"}) is False
    assert b.connections == 0, "the address the relay pointed to was connected to"
    assert [(r["method"], r["path"]) for r in a.seen] == [("POST", "/r302/ingest")]
    text = " ".join(str(w.message) for w in caught)
    assert a.target in text and "KEY-OF-THE-USER" not in text


def test_the_two_servers_can_show_a_redirect_followed_to_another_name(origins):
    """The control for the test above: urllib's stock opener follows to `localhost`, so an empty
    connection count there is not for want of a road."""
    import urllib.request
    a, b = origins
    a.target = f"http://localhost:{b.server_address[1]}/landed"
    req = urllib.request.Request(a.url("/r302/ingest"), data=b"{}", method="POST",
                                 headers={"X-Styxx-Key": "KEY"})
    with urllib.request.build_opener().open(req, timeout=10):
        pass
    assert b.connections >= 1 and [r["key"] for r in b.seen] == ["KEY"]


@pytest.mark.parametrize("code", [300, 304, 305, 306])
def test_a_3xx_that_is_no_redirect_is_an_http_error_and_is_not_called_one(origins, monkeypatch, capsys, code):
    """An earlier commit called a 304 "a redirect to (none given)"."""
    a, b = _post_under(monkeypatch, origins, f"/r{code}/ingest")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert stream._post({"agent": "agent"}) is False
    with pytest.raises(stream.ClaimError) as e:
        stream.claim_agent("agent", relay=a.url(f"/r{code}/ingest"))
    assert "redirect" not in str(e.value)
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a_, **k: {"agent_name": "agent"})
    assert publish_mod.publish("agent", a.url(f"/r{code}/submit")) is None
    err = capsys.readouterr().err
    assert f"HTTP {code} from" in err and "redirect" not in err
    assert b.connections == 0


def test_the_redirect_messages_do_not_say_what_the_relay_did(origins, monkeypatch, capsys):
    """An earlier commit said "nothing is claimed" and "nothing is delivered". A relay that takes
    the POST and then redirects (post, redirect, get) holds the name, or the packet. styxx knows
    only that it read no reply."""
    a, b = _post_under(monkeypatch, origins, "/r303/ingest")
    with pytest.warns(RuntimeWarning) as caught:
        stream._post({"agent": "agent"})
    text = " ".join(str(w.message) for w in caught)
    assert "counted as not delivered" in text and "nothing is delivered" not in text
    with pytest.raises(stream.ClaimError) as e:
        stream.claim_agent("agent", relay=a.url("/r303/ingest"))
    assert "stored no key" in str(e.value) and "nothing is claimed" not in str(e.value)
    assert e.value.answered is True
    assert cli.main(["claim", "agent", "--relay", a.url("/r303/ingest")]) == 1
    assert "may be offline" not in capsys.readouterr().err, "the relay answered"


def test_the_offline_hint_is_printed_only_where_nothing_answered(net, capsys):
    assert cli.main(["claim", "agent", "--relay", "https://relay.example.invalid/ingest"]) == 1
    assert "may be offline" in capsys.readouterr().err
    for body in (b"<!doctype html><html>a page</html>", b"\xef\xbb\xbf{}", b'{"name": "agent"}'):
        net.body = body
        assert cli.main(["claim", "agent", "--relay", "https://relay.example.invalid/ingest"]) == 1
        err = capsys.readouterr().err
        assert "may be offline" not in err and "name is taken" not in err, body
    net.body = b"<!doctype html><html>a page</html>"
    with pytest.raises(stream.ClaimError) as e:
        stream.claim_agent("agent", relay="https://relay.example.invalid/ingest")
    assert "its reply is not JSON" in str(e.value) and "unreachable" not in str(e.value)


@pytest.mark.parametrize("name", ["a" * 31 + "-x", "b" * 31 + "_tail", "plain"])
def test_the_relays_echo_of_the_name_that_was_sent_is_taken(monkeypatch, net, name):
    """_sanitize_name strips hyphens and underscores and then cuts to 32 characters, so what it
    sends can end in one. An earlier commit refused the relay's echo of exactly that name."""
    sent = stream._sanitize_name(name)
    saved = []
    monkeypatch.setattr(stream, "_save_creds", saved.append)
    net.body = json.dumps({"name": sent, "key": "k"}).encode("utf-8")
    out = stream.claim_agent(name, relay="https://relay.example.invalid/ingest")
    assert json.loads(net.bodies[0]) == {"name": sent}
    assert out["name"] == sent and list(saved[0]) == [sent, "_default"]


@pytest.mark.parametrize("body", [
    b'{"name": "someone-else", "key": "k"}',             # would overwrite that agent's key
    b'{"name": "Agent", "key": "k"}',
    b'{"name": "agent", "key": "' + b"k" * 513 + b'"}',
], ids=["another_agent", "another_case", "a_key_of_513_characters"])
def test_a_reply_that_is_not_for_the_name_sent_is_refused(monkeypatch, net, body):
    saved = []
    monkeypatch.setattr(stream, "_save_creds", saved.append)
    net.body = body
    with pytest.raises(stream.ClaimError) as e:
        stream.claim_agent("agent", relay="https://relay.example.invalid/ingest")
    assert saved == [] and e.value.answered is True


@pytest.mark.parametrize("link", ["https://dash.example.invalid/\u202egnp.exe", "https://dash.example.invalid/" + "a" * 2048,
                                  "<https://dash.example.invalid/a>"])
def test_a_dashboard_address_with_a_format_character_or_of_great_length_is_dropped(net, capsys, link):
    net.body = json.dumps({"name": "agent", "key": "k", "url": link}).encode("utf-8")
    assert stream.claim_agent("agent", relay="https://relay.example.invalid/ingest")["url"] == ""
    assert cli.main(["claim", "agent", "--relay", "https://relay.example.invalid/ingest"]) == 0
    out = capsys.readouterr().out
    # "the relay gave no dashboard URL" was false where it gave one that was dropped
    assert "(none: the relay gave no dashboard address that is an http or https URL)" in out


def test_text_a_relay_chose_is_printed_with_its_control_characters_as_escapes(net, capsys):
    import urllib.error

    def error_reply(self, req, *a, **k):
        import io
        raise urllib.error.HTTPError(req.full_url, 409, "Conflict\x1b[2J", None,
                                     io.BytesIO(b'{"error": "taken\\u001b[2J\\u202e' + b"x" * 5000 + b'"}'))

    import urllib.request
    urllib_open = urllib.request.OpenerDirector.open
    try:
        urllib.request.OpenerDirector.open = error_reply
        assert cli.main(["claim", "agent", "--relay", "https://relay.example.invalid/ingest"]) == 1
    finally:
        urllib.request.OpenerDirector.open = urllib_open
    err = capsys.readouterr().err
    assert "\x1b" not in err and "\u202e" not in err and "taken\\x1b[2J" in err
    assert len(err) < 1000
    assert stream.show("a\x1b\u202e\nb" + "c" * 400, 10) == "a\\x1b\\u202e\\nb" + "c" * 5 + "..."


def test_a_c1_control_and_a_format_character_are_no_part_of_a_url():
    """U+0085 is white space to Python, so the C1 half of the rule had no test of its own."""
    for ch in ("\x9f", "\x80", "\u202e", "\u200b"):
        assert not ch.isspace()
        assert not stream.is_http_url("https://relay.example.invalid/in" + ch + "gest")
    assert stream.is_http_url("https://relay.example.invalid/ingest")


def test_every_request_carries_its_timeout(monkeypatch, net):
    monkeypatch.setattr(stream, "_relay", "https://relay.example.invalid/ingest")
    stream._post({"agent": "agent"})
    with pytest.raises(stream.ClaimError):
        stream.claim_agent("agent", relay="https://relay.example.invalid/ingest", timeout=7.5)
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a, **k: {"agent_name": "agent"})
    publish_mod.publish("agent", "https://example.invalid/api")
    compare._fetch_population("https://example.invalid/board")
    assert net.timeouts == [stream._POST_TIMEOUT, 7.5, 15, 10]


def test_the_opener_keeps_proxies_and_certificate_checks_and_is_built_once(monkeypatch, net):
    """What open_url changes from urlopen is the redirect handler and nothing else."""
    import urllib.request
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setattr(stream, "_relay", "https://relay.example.invalid/ingest")
    stream._post({"agent": "agent"})
    opener = stream._OPENER
    assert opener is not None
    kinds = [type(h) for h in opener.handlers]
    redirectors = [k for k in kinds if issubclass(k, urllib.request.HTTPRedirectHandler)]
    assert redirectors == [stream._NoRedirect]
    (https,) = [h for h in opener.handlers if isinstance(h, urllib.request.HTTPSHandler)]
    context = getattr(https, "_context", None)
    if context is not None:     # Python 3.12 builds the default context in the handler; before it, http.client does
        import ssl
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True
    (proxy,) = [h for h in opener.handlers if isinstance(h, urllib.request.ProxyHandler)]
    assert proxy.proxies.get("http") == "http://127.0.0.1:9"
    stream._post({"agent": "agent"})
    assert stream._OPENER is opener, "one opener for the process"


def test_an_opener_a_program_installed_is_not_used(monkeypatch, net):
    """Declared in the change log: through 8.0.0 these calls went through urlopen and so through
    an installed opener, which follows redirects."""
    import urllib.request

    class Installed:
        calls = 0

        def open(self, *a, **k):
            type(self).calls += 1
            raise OSError("the installed opener was used")

    monkeypatch.setattr(urllib.request, "_opener", Installed())
    monkeypatch.setattr(stream, "_relay", "https://relay.example.invalid/ingest")
    stream._post({"agent": "agent"})
    assert Installed.calls == 0 and net.opened == ["https://relay.example.invalid/ingest"]


def test_standing_in_for_open_url_stands_in_for_all_four_requests(monkeypatch):
    """The one patch point. publish and compare call it as an attribute of styxx.stream, so a
    stand-in there is on the road of each."""
    calls = []

    def stand_in(req, timeout):
        calls.append(req.full_url)
        raise OSError("stand-in")

    monkeypatch.setattr(stream, "open_url", stand_in)
    monkeypatch.setattr(stream, "_relay", "https://relay.example.invalid/ingest")
    stream._post({"agent": "agent"})
    with pytest.raises(stream.ClaimError):
        stream.claim_agent("agent", relay="https://relay.example.invalid/ingest")
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a, **k: {"agent_name": "agent"})
    publish_mod.publish("agent", "https://example.invalid/api")
    compare._fetch_population("https://example.invalid/board")
    assert calls == ["https://relay.example.invalid/ingest", "https://relay.example.invalid/claim",
                     "https://example.invalid/api", "https://example.invalid/board"]


def test_compare_agents_calls_the_fetch_function_with_the_url_alone(monkeypatch, net):
    """tests/test_0_6_0.py and a caller's own tests replace compare._fetch_population with a
    function of one argument. An earlier commit moved compare_agents to another function, so the
    replacement was called by nothing and the real request went out."""
    monkeypatch.setattr(compare, "_fetch_population", lambda url: [(0.1, 0.5, 0.1, 0.2, 0.05, 0.05)])
    out = compare.compare_agents(_fingerprint(), api_url="https://example.invalid/board")
    assert out.status == "ok" and out.n_agents == 1 and net.opened == []


def test_values_too_large_to_sum_are_not_a_vector(net):
    net.body = b'{"agents": [{"fingerprint": {"reasoning": 1e308}}, {"fingerprint": {"reasoning": 1e308}}]}'
    assert compare._fetch_population("https://example.invalid/board") == []


def test_a_credentials_file_that_is_not_an_object_holds_no_credentials(monkeypatch, tmp_path):
    """Through 8.0.0 it ended enable(), a claim and `styxx feed` in a traceback."""
    path = tmp_path / "credentials.json"
    monkeypatch.setattr(stream, "_creds_path", lambda: path)
    for text in ("[1, 2]", '"a string"', "7", "null", "{not json"):
        path.write_text(text, encoding="utf-8")
        assert _REAL_LOAD_CREDS() == {}
    path.write_text('{"agent": {"key": "k"}}', encoding="utf-8")
    assert _REAL_LOAD_CREDS() == {"agent": {"key": "k"}}
