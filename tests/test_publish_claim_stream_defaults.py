# -*- coding: utf-8 -*-
"""`styxx publish`, `styxx compare-agents`, `styxx claim` and the stream name no host by default.

Found on 2026-11-12, after styxx 8.0.0 was published: those four defaulted to hosts under
darkflobi.com, and every name under that domain answered that day from a domain-parking service,
not from a lab server. `styxx publish --name X` handed its payload to that server and reported
any 2xx reply as published; `styxx compare-agents` read the parking page as a leaderboard with no
agents on it.

What these tests pin: with no endpoint, relay or leaderboard URL named, none of the four opens a
connection; a 2xx reply that is not a JSON object is not "published"; a reply that is not a
leaderboard is not an empty one; a relay given as a bare origin is claimed on that host and no
other; and no file under `styxx/` names that domain apart from eight listed lines.

Scope. Other commands of the package contact a named service by design and are not the subject
here: `styxx.clock` reads two public Solana RPC hosts, `styxx.guardrail.entity_verify` queries
Wikipedia, the diff gate fetches a pull request the caller names.

No test here opens a connection. Each replaces urllib's `urlopen` with a recorder that notes the
URL and answers or refuses, and asserts on the recorder: several of the functions under test catch
every exception, so a guard that only raised would be swallowed and prove nothing (a review of
an earlier version of this file put a default host or a link on the domain back eight ways, and
seven went unseen).
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import subprocess
import sys
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


def _reply(body: bytes, status: int = 200):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class Net:
    """Stands in for urlopen: records every URL asked for, then answers `body` or refuses."""

    def __init__(self, body=None):
        self.opened, self.bodies, self.body = [], [], body

    def __call__(self, req, *a, **k):
        self.opened.append(getattr(req, "full_url", req))
        self.bodies.append(getattr(req, "data", None))
        if self.body is None:
            raise OSError("no network in tests")
        return _reply(self.body)


@pytest.fixture
def net(monkeypatch):
    """One recorder behind every urlopen these modules can reach (they share urllib.request)."""
    import urllib.request
    rec = Net()
    monkeypatch.setattr(urllib.request, "urlopen", rec)
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
    # the emitter's state, as a fresh process has it
    for name, value in (("_enabled", False), ("_agent", None), ("_key", None), ("_relay", None),
                        ("_worker", None)):
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


def test_claim_with_a_reply_that_is_not_an_object_is_malformed(net):
    net.body = b"[]"
    with pytest.raises(stream.ClaimError):
        stream.claim_agent("agent", relay="https://relay.example.invalid/ingest")


def test_claim_command_prints_a_next_step_that_names_the_relay(net, capsys):
    """The next step used to be `autoboot(stream=True)` alone, which now leaves the feed off:
    the emitter needs the relay named, and the command says how."""
    net.body = json.dumps({"name": "agent", "key": "k"}).encode("utf-8")
    assert cli.main(["claim", "agent", "--relay", "https://relay.example.invalid/ingest"]) == 0
    out = capsys.readouterr().out
    assert "STYXX_STREAM_RELAY=https://relay.example.invalid/ingest" in out
    assert "(the relay gave no dashboard URL)" in out


def test_feed_command_without_a_dashboard_base_fails(capsys):
    assert cli.main(["feed", "agent"]) == 1
    captured = capsys.readouterr()
    assert "no default dashboard" in captured.err and captured.out == ""


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
def _urlopen(req, *a, **k):
    attempts.append(getattr(req, "full_url", str(req)))
    raise OSError("no network in tests")
def _connect(self, address, *a, **k):
    attempts.append(repr(address))
    raise OSError("no network in tests")
urllib.request.urlopen = _urlopen
socket.socket.connect = _connect
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
    ("styxx/publish.py", "under darkflobi.com, where the public dashboard lived. on that day"),
    ("styxx/compare.py", "under darkflobi.com; on that day the host answered from a"),
    ("styxx/stream.py", "both were hosts under darkflobi.com, where the public relay lived. On"),
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
    " https://relay.example.invalid/ingest",
])
def test_a_url_the_http_client_would_refuse_is_refused_before_it_gets_there(monkeypatch, net, capsys, url):
    """Each of these passed the scheme-and-host check of an earlier draft and ended `styxx claim`
    in a traceback from http.client after its two step lines."""
    assert not stream.is_http_url(url) and not publish_mod._is_http_url(url)
    with pytest.raises(stream.ClaimError):
        stream.claim_url(url)
    monkeypatch.setattr(publish_mod, "prepare_payload", lambda *a, **k: {"agent_name": "agent"})
    assert publish_mod.publish("agent", url) is None
    assert net.opened == []
    capsys.readouterr()
    if url == url.strip():           # relay_url drops outer spaces; the rest reach the command as typed
        assert cli.main(["claim", "agent", "--relay", url]) == 1
        captured = capsys.readouterr()
        assert "reserving subdomain" not in captured.out and "secret" not in captured.out
        assert net.opened == []


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

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
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
