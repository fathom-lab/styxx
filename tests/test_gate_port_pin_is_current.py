"""The browser port's pin must name the instrument that is actually in this checkout.

`web/gate/diffgate.js` is a hand transliteration of one specific file, `styxx/diffgate.py`.
`web/gate/differential/py_side.py` pins that file's sha256 and refuses to run against anything
else, so "0 disagreements" always means "against the file the port claims to be". That is the
right design. The problem is when it runs.

## This is SP-1, and this repository has already been bitten by it once

`py_side.py` says so itself, in a comment above the pin:

    ... which is why this script has been REFUSING TO RUN on main ever since -- the check that
    guards the two-implementation claim was itself disabled, which is how the port fell a whole
    cycle behind without anything failing.

The port fell a cycle behind, the public bookmarklet served a stale reading, and every check
stayed green, because the differential is run by hand and `web/gate/` appears in no workflow.
That instance was found and repaired; the class was not. The next time `styxx/diffgate.py`
moves, the same silence returns.

`benchmarks/silent_pass/CORPUS.md` catalogues this shape as SP-1 — an absent measurement
surfacing as a passing check — and `tests/test_ledger.py` makes exactly this argument about the
ledger's regeneration guarantee. This file does the same job for the browser door.

## What this proves, and what it does not

It proves the **pin is current**: `py_side.py` would run against this checkout's instrument
rather than refusing, and `web/gate/README.md` names the same file. So a stale port can no
longer hide behind a guard that has quietly stopped firing.

It does **not** prove the port agrees with the Python. Only the differential run proves that,
it needs node, and it is not what this test is. When this test fails, running the differential
is the next step, not editing the pin.

There is no skip path. A test that skips when its precondition is missing is the defect this
file exists to prevent, and `tests/test_ledger.py` has the same rule for the same reason.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT = ROOT / "styxx" / "diffgate.py"
PY_SIDE = ROOT / "web" / "gate" / "differential" / "py_side.py"
GATE_README = ROOT / "web" / "gate" / "README.md"

_PIN = re.compile(r'^PINNED\s*=\s*"([0-9a-f]{64})"', re.M)


def _lf_sha256(path: Path) -> str:
    """The convention py_side.py uses: CRLF -> LF, then sha256.

    A wheel built on Windows carries CRLF and the same file hashes differently, so the
    normalisation is part of the claim rather than a convenience.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_the_pin_names_this_checkouts_instrument():
    assert INSTRUMENT.is_file(), f"{INSTRUMENT} is missing; the pin has nothing to name"
    assert PY_SIDE.is_file(), f"{PY_SIDE} is missing; the port's guard is gone"

    m = _PIN.search(PY_SIDE.read_text(encoding="utf-8"))
    assert m, "py_side.py no longer declares a PINNED sha256 at the start of a line"
    pinned, actual = m.group(1), _lf_sha256(INSTRUMENT)

    assert pinned == actual, (
        "web/gate is pinned to an instrument this checkout does not contain.\n"
        f"  pinned  {pinned}\n"
        f"  actual  {actual}\n"
        "py_side.py will refuse to run, so the differential that backs the two-implementation "
        "claim is not running, and nothing else would have said so. Re-cut "
        "web/gate/diffgate.js against the current styxx/diffgate.py and run the differential; "
        "move the pin because the port moved, never to make this test pass."
    )


def test_the_gate_readme_names_the_same_file_as_the_pin():
    """Prose and pin drift apart silently otherwise, and the README is what a reader trusts."""
    m = _PIN.search(PY_SIDE.read_text(encoding="utf-8"))
    assert m
    pinned = m.group(1)
    readme = GATE_README.read_text(encoding="utf-8")

    full = set(re.findall(r"\b([0-9a-f]{64})\b", readme))
    assert pinned in full, (
        "web/gate/README.md does not state the pinned sha256 in full.\n"
        f"  pinned            {pinned}\n"
        f"  README states     {sorted(full) or '(none)'}\n"
        "The README is the receipt for the port; if it names a different file than the guard "
        "does, one of them is lying to a reader who cannot run either."
    )

    # Abbreviated mentions (`9b620e00…`) must belong to the pinned file too -- but only where
    # the sentence is about the instrument. The README also lists earlier *bookmarklet build*
    # hashes, correctly labelled as earlier builds, and those are not instrument claims. The
    # first draft of this check flagged them, which is the reason the rule is written narrowly
    # rather than over every hex string on the page.
    bad = []
    for i, line in enumerate(readme.splitlines(), 1):
        if not re.search(r"diffgate\.py|\binstrument\b|hashes to", line):
            continue
        for s in re.findall(r"`([0-9a-f]{8})…`", line):
            if not pinned.startswith(s):
                bad.append((i, s, line.strip()[:90]))
    assert not bad, (
        "web/gate/README.md abbreviates an instrument sha that is not the pinned file:\n"
        + "\n".join(f"  line {i}: {s} -- {l}" for i, s, l in bad)
        + f"\n  pinned: {pinned[:8]}…\n"
        "Either the sentence is about an older instrument and should say so, or it is stale."
    )
