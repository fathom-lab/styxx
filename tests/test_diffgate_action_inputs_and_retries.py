# -*- coding: utf-8 -*-
"""How the Action reads `strict`, what `action.yml` hands the script, what a refused GITHUB_API_URL
shows, and when a request is asked again.

`strict` is read the way `soft-fail` is (surrounding whitespace stripped, case ignored), and a value
that is neither "true" nor "false" is named in a warning and means not strict. `action.yml`'s `runs:`
block is held line for line, since nothing else holds that a workflow's `soft-fail` and `strict` reach
the script and that the script's exit code is the step's. A GITHUB_API_URL the script refuses is shown
without anything that stands before its last "@". A request answered 429, 500 to 599, or 403 with a
rate limit's headers is asked again, to the same address, within fixed bounds.

The retry tests run `api()` against servers on 127.0.0.1 with the clock and the sleep replaced; no
test waits.
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import sys
import threading
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ACTION = ROOT / "diffgate_action.py"
TOKEN = "made-up-token-0123456789"
DIFF_ACCEPT = "application/vnd.github.diff"


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FC = _load_file("diffgate_action_fails_closed_for_inputs", Path(__file__).with_name("test_diffgate_action_fails_closed.py"))
UNCHECKABLE = dict(body="All tests pass.", api=FC.DIFF_SRC)          # one claim, UNCHECKABLE on any diff
W_STRICT = "::warning title=styxx diffgate - strict value not recognised::"
W_SOFT = "::warning title=styxx diffgate - soft-fail value not recognised::"


@pytest.fixture
def act():
    """The script, loaded anew for each test, so the seconds one test waited are not another's."""
    return FC.load("diffgate_action_inputs")


def _run(act, tmp_path, monkeypatch, capsys, case, soft, strict_raw):
    """main() on one path with `api` replaced, STYXX_STRICT set to `strict_raw` as it is (None: unset).
    Returns (exit code, stdout, summary)."""
    ev, summary = tmp_path / "event.json", tmp_path / "summary.md"
    event = case.get("event", "pull_request")
    ev.write_text(json.dumps(FC.payload(event, case["body"], case.get("created"))), encoding="utf-8")
    if summary.exists():
        summary.unlink()
    for name, value in (("GITHUB_EVENT_NAME", event), ("GITHUB_EVENT_PATH", str(ev)),
                        ("GITHUB_STEP_SUMMARY", str(summary)), ("GH_TOKEN", "made-up"), ("STYXX_SOFT_FAIL", soft)):
        monkeypatch.setenv(name, value)
    if strict_raw is None:
        monkeypatch.delenv("STYXX_STRICT", raising=False)
    else:
        monkeypatch.setenv("STYXX_STRICT", strict_raw)
    answer = case.get("api", FC.DIFF)
    monkeypatch.setattr(act, "api", lambda url, accept: answer)
    rc = act.main()
    return rc, capsys.readouterr().out, summary.read_bytes().decode("utf-8") if summary.exists() else ""


# -- strict -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, strict, unrecognised", [
    (None, False, False), ("false", False, False), ("FALSE", False, False), (" false ", False, False),
    ("true", True, False), ("TRUE", True, False), ("True", True, False),
    (" true", True, False), ("true ", True, False), ("\ttrue\n", True, False), (" true", True, False),
    ("", False, True), ("yes", False, True), ("1", False, True), ("on", False, True), ("tru e", False, True),
    ("truefalse", False, True),
])
def test_how_the_strict_value_is_read(act, raw, strict, unrecognised):
    assert act._strict(raw) == (strict, raw if unrecognised else None)


@pytest.mark.parametrize("value, strict, warned", [
    ("true", True, False), (" true", True, False), ("TRUE ", True, False), ("false", False, False),
    ("yes", False, True), ("", False, True), ("1", False, True),
])
def test_strict_in_a_blocking_job(act, tmp_path, monkeypatch, capsys, value, strict, warned):
    """An UNCHECKABLE claim in a blocking job: it fails the job where strict is on and passes where it
    is not, and a value that is neither "true" nor "false" is named in a warning."""
    rc, out, summary = _run(act, tmp_path, monkeypatch, capsys, UNCHECKABLE, "false", value)
    assert rc == (1 if strict else 0), out
    assert (W_STRICT in out) is warned
    assert (FC.E_STRICT in out) is strict
    if warned:
        assert out.startswith(W_STRICT + f"strict is {value!r}, which is neither")
        assert 'Set strict: "true" to turn it on.' in out.splitlines()[0]


def test_strict_unset_is_not_strict_and_is_no_value_to_warn_of(act, tmp_path, monkeypatch, capsys):
    rc, out, _ = _run(act, tmp_path, monkeypatch, capsys, UNCHECKABLE, "false", None)
    assert rc == 0 and W_STRICT not in out


def test_the_strict_warning_comes_after_the_soft_fail_one_and_escapes_a_percent(act, tmp_path, monkeypatch, capsys):
    rc, out, _ = _run(act, tmp_path, monkeypatch, capsys, UNCHECKABLE, "maybe", "50%\nyes")
    lines = out.splitlines()
    assert rc == 0
    assert lines[0].startswith(W_SOFT) and lines[1].startswith(W_STRICT)
    assert "strict is '50%25\\nyes', which is neither" in lines[1]          # one line: repr wrote the line break


@pytest.mark.parametrize("strict, word", [(False, "PASS"), (True, "FAIL")])
@pytest.mark.parametrize("soft", ["false", "true"])
def test_the_verdict_word_under_strict(act, tmp_path, monkeypatch, capsys, soft, strict, word):
    """The word in the summary's heading and on the last line the gate prints is the gate's verdict
    under the same strict the exit code is worked out with: an UNCHECKABLE claim is FAIL under strict
    and PASS without it, in both modes."""
    rc, out, summary, _ = FC.run_main(act, tmp_path, monkeypatch, capsys, UNCHECKABLE, soft, strict)
    assert f"**{word}** · 1 claim(s) checked" in summary
    assert [ln for ln in out.splitlines() if ln.startswith("styxx diffgate: ")][-1] == (
        f"styxx diffgate: {word} - 1 claim(s), 0 contradicted")
    assert rc == (1 if strict and soft == "false" else 0)


# -- action.yml ---------------------------------------------------------------------------------------------------

def test_action_yml_hands_the_script_its_inputs_and_keeps_its_exit_code():
    """The `runs:` block line for line, comments and blank lines aside. A dropped or swapped `env:` line
    would turn a blocking workflow into a reporting one with no other sign, and `|| true` or
    `continue-on-error` would discard the exit code."""
    yml = (ROOT / "action.yml").read_text(encoding="utf-8").replace("\r\n", "\n")
    runs = yml[yml.index("\nruns:\n") + 1:]
    lines = [ln.rstrip() for ln in runs.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    assert lines == [
        "runs:",
        '  using: "composite"',
        "  steps:",
        "    - uses: actions/setup-python@v5",
        "      with:",
        '        python-version: "3.11"',
        "    - shell: bash",
        '      run: pip install --quiet "${{ inputs.styxx-version }}"',
        "    - shell: bash",
        "      env:",
        "        STYXX_STRICT: ${{ inputs.strict }}",
        "        STYXX_SOFT_FAIL: ${{ inputs.soft-fail }}",
        "        GH_TOKEN: ${{ github.token }}",
        '      run: python "${{ github.action_path }}/diffgate_action.py"',
    ]
    source = ACTION.read_text(encoding="utf-8")
    for name in ("STYXX_STRICT", "STYXX_SOFT_FAIL", "GH_TOKEN"):
        assert f'os.environ.get("{name}")' in source, name
    assert source.rstrip().endswith('if __name__ == "__main__":\n    sys.exit(main())')


# -- a GITHUB_API_URL that is refused -----------------------------------------------------------------------------

@pytest.mark.parametrize("value, shown, cut", [
    ("https://user@ghe.example/api/v3", "https://ghe.example/api/v3", True),
    ("https://user:S3CRETPW@ghe.example/api/v3", "https://ghe.example/api/v3", True),
    ("user:S3CRETPW@ghe.example/api/v3", "ghe.example/api/v3", True),
    ("https://user:S3C/RET@ghe.example", "https://ghe.example", True),
    ("https://a@b@ghe.example/x", "https://ghe.example/x", True),
    ("ftp://ghe.example/api/v3", "ftp://ghe.example/api/v3", False),
    ("not an address", "not an address", False),
    ("", "", False),
])
def test_what_stands_before_the_last_at_is_left_out(act, monkeypatch, value, shown, cut):
    assert act._without_user_part(value) == (shown, cut)
    monkeypatch.setenv("GITHUB_API_URL", value)
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    if act._api_origin() is not None:               # an empty value is the default origin, and is not refused
        return
    with pytest.raises(RuntimeError) as e:
        act.api("https://api.github.com/repos/o/r/pulls/7", DIFF_ACCEPT)
    msg = str(e.value)
    assert msg.startswith(f"GITHUB_API_URL is {shown!r}")
    assert ('(what stood before its last "@" is left out here' in msg) is cut
    assert "S3C" not in msg and "user:" not in msg and "user@" not in msg and "a@b" not in msg
    assert "the workflow token is sent nowhere" in msg


def test_a_token_in_a_refused_address_is_hidden_where_repr_doubles_its_backslash(act, monkeypatch):
    """`_shown` gets the token for the addresses refused before the token is checked, so a token that
    `repr` would write otherwise than it was set is replaced before the quoting."""
    token = "tok\\en-0123456789"
    monkeypatch.setenv("GH_TOKEN", token)
    monkeypatch.delenv("GITHUB_API_URL", raising=False)
    with pytest.raises(RuntimeError) as e:
        act.api("https://elsewhere.example/" + token, DIFF_ACCEPT)
    assert "0123456789" not in str(e.value) and "***" in str(e.value)
    with pytest.raises(RuntimeError) as e:
        act.api("gopher://x/" + token, DIFF_ACCEPT)
    assert "0123456789" not in str(e.value) and "***" in str(e.value)


# -- asking again -------------------------------------------------------------------------------------------------

class _Scripted(http.server.BaseHTTPRequestHandler):
    """Answers in the order given; the last answer is given again to any further request. An answer is
    (status, headers, body); a status of None drops the connection without an answer."""
    answers, seen = [], []

    def do_GET(self):
        cls = type(self)
        cls.seen.append((self.path, self.headers.get("Authorization")))
        status, headers, body = cls.answers[min(len(cls.seen), len(cls.answers)) - 1]
        if status is None:
            self.close_connection = True
            return
        data = body.encode("utf-8")
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def servers():
    made = []

    def make(*answers):
        handler = type("Scripted", (_Scripted,), {"answers": list(answers), "seen": []})
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        made.append(srv)
        return handler, f"http://127.0.0.1:{srv.server_address[1]}"

    yield make
    for srv in made:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def waits(act, monkeypatch):
    """The clock stands at 1000 and a sleep is written down, not slept."""
    slept = []
    monkeypatch.setattr(act, "_sleep", slept.append)
    monkeypatch.setattr(act, "_now", lambda: 1000.0)
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    return slept


def _ask(act, monkeypatch, base, path="/repos/o/r/pulls/7"):
    monkeypatch.setenv("GITHUB_API_URL", base)
    return act.api(base + path, DIFF_ACCEPT)


OK = (200, {}, "the diff")


@pytest.mark.parametrize("answers, slept", [
    ([(503, {}, ""), OK], [2]),
    ([(500, {}, ""), (502, {}, ""), OK], [2, 8]),
    ([(599, {}, ""), OK], [2]),
    ([(429, {"Retry-After": "5"}, ""), OK], [5]),
    ([(429, {"Retry-After": "0"}, ""), OK], [0]),
    ([(429, {}, ""), OK], [60]),
    ([(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, ""), OK], [60]),
    ([(429, {"Retry-After": "-5"}, ""), OK], [60]),
    ([(429, {"Retry-After": "+5"}, ""), OK], [60]),
    ([(429, {"Retry-After": "5_0"}, ""), OK], [60]),
    ([(403, {"Retry-After": "3"}, ""), OK], [3]),
    ([(403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1007"}, ""), OK], [7]),
    ([(403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "990"}, ""), OK], [0]),
    ([(429, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1030"}, ""), OK], [30]),
    ([(429, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "soon"}, ""), OK], [60]),
    ([(429, {"X-RateLimit-Remaining": "12", "X-RateLimit-Reset": "1007"}, ""), OK], [60]),
])
def test_an_answer_that_is_asked_again(act, servers, waits, monkeypatch, capsys, answers, slept):
    handler, base = servers(*answers)
    assert _ask(act, monkeypatch, base) == "the diff"
    assert waits == slept
    assert len(handler.seen) == len(slept) + 1
    assert all(seen == ("/repos/o/r/pulls/7", f"Bearer {TOKEN}") for seen in handler.seen)
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"styxx diffgate: GitHub answered {answers[i][0]}; asking again in {s} s (request {i + 2} of 3)"
                     for i, s in enumerate(slept)]


@pytest.mark.parametrize("status, headers", [
    (403, {}), (403, {"X-RateLimit-Remaining": "17"}), (403, {"X-RateLimit-Remaining": "0"}),
    (403, {"Retry-After": "soon"}), (404, {}), (404, {"Retry-After": "1"}), (406, {}), (401, {}),
    (422, {"Retry-After": "1"}),
])
def test_an_answer_that_is_not_asked_again(act, servers, waits, monkeypatch, capsys, status, headers):
    handler, base = servers((status, headers, "{}"))
    with pytest.raises(urllib.error.HTTPError) as e:
        _ask(act, monkeypatch, base)
    assert e.value.code == status
    assert waits == [] and len(handler.seen) == 1
    assert capsys.readouterr().out == ""


def test_three_requests_are_the_most(act, servers, waits, monkeypatch):
    handler, base = servers((503, {}, ""))
    with pytest.raises(urllib.error.HTTPError) as e:
        _ask(act, monkeypatch, base)
    assert e.value.code == 503 and waits == [2, 8] and len(handler.seen) == 3


@pytest.mark.parametrize("headers, asked", [
    ({"Retry-After": "61"}, 61),
    ({"Retry-After": "999999"}, 999999),
    ({"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "3400"}, 2400),
])
def test_a_wait_longer_than_a_minute_is_not_made(act, servers, waits, monkeypatch, headers, asked):
    handler, base = servers((429, headers, ""), OK)
    with pytest.raises(RuntimeError) as e:
        _ask(act, monkeypatch, base)
    assert f"GitHub answers 429, and the wait before the next request would be {asked} seconds" in str(e.value)
    assert "so it was not asked again" in str(e.value)
    assert isinstance(e.value.__cause__, urllib.error.HTTPError)
    assert waits == [] and len(handler.seen) == 1


def test_the_waits_of_one_run_add_up_to_two_minutes_at_most(act, servers, waits, monkeypatch):
    """Three calls: 60 seconds, 60 seconds, and then no wait is left, so the third is not asked again."""
    for _ in range(2):
        handler, base = servers((429, {}, ""), OK)
        assert _ask(act, monkeypatch, base) == "the diff"
    assert waits == [60, 60] and act._waited == 120
    handler, base = servers((503, {}, ""), OK)
    with pytest.raises(RuntimeError) as e:
        _ask(act, monkeypatch, base)
    assert "GitHub answers 503, and the wait before the next request would be 2 seconds" in str(e.value)
    assert "120 in one run (it has waited 120)" in str(e.value)
    assert waits == [60, 60] and len(handler.seen) == 1


@pytest.mark.parametrize("value", [" 5", "5 ", "+5", "-5", "5_0", "5.0", "٥", "5" * 13, "", "0x5", None])
def test_only_ascii_digits_are_a_number_in_a_header(act, value):
    assert act._header_number({"Retry-After": value}, "Retry-After", 12) is None


def test_a_header_of_digits_is_its_number(act):
    assert act._header_number({"Retry-After": "007"}, "Retry-After", 6) == 7
    assert act._header_number({"Retry-After": "1234567"}, "Retry-After", 6) is None
    assert act._header_number(None, "Retry-After", 6) is None


def test_a_request_asked_again_follows_a_redirect_within_the_origin_only(act, servers, waits, monkeypatch):
    other, other_base = servers(OK)
    handler, base = servers((503, {}, ""), (302, {"Location": "/moved"}, ""), OK)
    assert _ask(act, monkeypatch, base) == "the diff"
    assert [seen[0] for seen in handler.seen] == ["/repos/o/r/pulls/7", "/repos/o/r/pulls/7", "/moved"]
    assert all(seen[1] == f"Bearer {TOKEN}" for seen in handler.seen)
    handler, base = servers((503, {}, ""), (302, {"Location": other_base + "/elsewhere"}, ""), OK)
    with pytest.raises(RuntimeError) as e:
        _ask(act, monkeypatch, base)
    assert "a redirect to" in str(e.value) and "which was not followed" in str(e.value)
    assert other.seen == []
    assert TOKEN not in str(e.value)


def test_a_dropped_connection_is_not_asked_again(act, servers, waits, monkeypatch):
    handler, base = servers((None, {}, ""), OK)
    with pytest.raises(Exception) as e:
        _ask(act, monkeypatch, base)
    assert not isinstance(e.value, urllib.error.HTTPError)
    assert waits == [] and len(handler.seen) == 1


def test_a_diff_served_on_the_second_request_is_gated_as_any_other(act, servers, waits, tmp_path, monkeypatch, capsys):
    """main() with the real api(): a 503, then the diff. The claim is contradicted and a blocking job fails."""
    handler, base = servers((503, {}, ""), (200, {}, FC.DIFF))
    ev = tmp_path / "event.json"
    ev.write_text(json.dumps({"pull_request": {"number": 7, "body": FC.FALSE, "url": base + "/repos/o/r/pulls/7"}}),
                  encoding="utf-8")
    for name, value in (("GITHUB_EVENT_NAME", "pull_request"), ("GITHUB_EVENT_PATH", str(ev)),
                        ("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md")), ("GITHUB_API_URL", base),
                        ("STYXX_STRICT", "false"), ("STYXX_SOFT_FAIL", "false")):
        monkeypatch.setenv(name, value)
    assert act.main() == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "styxx diffgate: GitHub answered 503; asking again in 2 s (request 2 of 3)"
    assert any(ln.startswith(FC.E_CONTRA) for ln in out)
    assert waits == [2] and len(handler.seen) == 2
