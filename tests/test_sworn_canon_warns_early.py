"""`canon` warns about a numeric span that verify will call MALFORMED, at the step where it is free.

WHY THIS EXISTS. A numeric span must carry exactly ONE digit-bearing token. Two makes it MALFORMED
`number_count`. That rule is purely lexical — `_number_token` reads the span's own inner text and
needs no manifest, no repository and no receipt — and `canon` already holds the span table and the
canonical text.

It cost three documents in one night, all by the same author, all the same shape:

    "712 of 150000 disagreed"           tokens 712, 150000
    "0 of 150000 disagreed"             tokens 0, 150000
    "9 of the 25 kills rest on one"     tokens 9, 25

Every one was canonised without complaint, committed, sworn, and only then reported MALFORMED — by
which point the sidecar names a commit, so the repair costs a re-canon and a re-swear. The
information existed at the first step and was not printed.

The warning does NOT change the exit code and does not refuse. `canon`'s job is a faithful round
trip; `verify`'s job is the verdict; this CLI reports and never gates. These tests pin both halves:
that the warning appears, and that it changes nothing else.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _canon(tmp_path, body: bytes):
    doc = tmp_path / "d.md"
    doc.write_bytes(body)
    out = tmp_path / "d.sworn.json"
    r = subprocess.run([sys.executable, "-m", "styxx.sworn", "canon", str(doc), "--out", str(out)],
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    return r, out


def test_canon_warns_when_a_numeric_span_carries_two_numbers(tmp_path):
    r, out = _canon(tmp_path, b'<sworn r="r1" k="numeric">712 of 150000 disagreed.</sworn>\n')
    assert r.returncode == 0, r.stdout + r.stderr
    assert "number_count" in r.stdout, (
        "canon saw a span it knows verify will refuse and said nothing:\n%s" % r.stdout)
    assert "712" in r.stdout and "150000" in r.stdout, (
        "the warning must name the tokens it found, or the author cannot see which two:\n%s"
        % r.stdout)


def test_the_warning_does_not_refuse_and_does_not_change_the_sidecar(tmp_path):
    """Reports, never gates. A doomed span still round-trips: the sidecar is written, the exit code
    is 0, and the bytes are what they would have been without the warning."""
    body = b'<sworn r="r1" k="numeric">712 of 150000 disagreed.</sworn>\n'
    r, out = _canon(tmp_path, body)
    assert r.returncode == 0
    assert out.exists(), "canon refused to write a sidecar over a warning"
    side = json.loads(out.read_text(encoding="utf-8"))
    assert len(side["spans"]) == 1 and side["spans"][0]["kind"] == "numeric"

    r2 = subprocess.run([sys.executable, "-m", "styxx.sworn", "render", str(out)],
                        cwd=str(ROOT), capture_output=True, timeout=300)
    assert r2.returncode == 0 and r2.stdout == body, (
        "the round trip must be byte-exact whether or not a span was warned about")


def test_a_well_formed_numeric_span_is_not_warned_about(tmp_path):
    """Without this, the test above passes for a verifier that warns on everything."""
    r, _ = _canon(tmp_path, b'<sworn r="r1" k="numeric">the count is 712.</sworn>\n')
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING" not in r.stdout, (
        "a span carrying exactly one digit-bearing token is fine and must not be warned "
        "about:\n%s" % r.stdout)


def test_a_numeric_span_with_no_number_is_warned_about_too(tmp_path):
    """The other half of number_count: zero tokens is as doomed as two."""
    r, _ = _canon(tmp_path, b'<sworn r="r1" k="numeric">no digits appear here.</sworn>\n')
    assert r.returncode == 0, r.stdout + r.stderr
    assert "number_count" in r.stdout, r.stdout


def test_non_numeric_spans_are_left_alone(tmp_path):
    """A quote span may carry as many numbers as it likes; the rule is numeric-only."""
    r, _ = _canon(
        tmp_path,
        b'<sworn r="r1" k="quote">712 of 150000 is a phrase to find.</sworn>\n')
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING" not in r.stdout, r.stdout
