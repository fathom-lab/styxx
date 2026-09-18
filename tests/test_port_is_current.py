"""The port's differential is a guard only if something runs it. Nothing did.

`web/gate/diffgate.js` is a transliteration of `styxx/diffgate.py`, and
`web/gate/differential/` holds the machinery that proves the two agree. On 2026-09-18 that
machinery was found to have been inert for a cycle, in three separate ways at once:

1. **Nothing in CI ran it.** No workflow invoked `py_side.py`, `js_side.js`, `differential.py` or
   `check_pairs.js`. The README's "0 disagreements" was a number somebody once typed. #124 then
   merged its Python half without its JavaScript half and the two implementations genuinely
   diverged on compat claims, with every test green.
2. **The pin had gone stale.** `py_side.py` refuses to run unless `styxx/diffgate.py` matches
   `PINNED`. An unrelated feature (`fetch_pr`) had moved the hash, so the script exited 1 on sight
   — correctly, and to nobody, because of (1).
3. **A new pinned-pairs file was one `.gitignore` line from never existing.** That directory
   ignores `*.json` and whitelists the pinned files by name. `check_pairs.js` skips a missing file
   through an `exists()` guard, so an un-whitelisted corpus shrinks the pinned set while the run
   still prints success.

`tests/test_differential_agreement.py` already says the principle, about a different pair of
implementations: *"A differential test's value is not the run that was published — it is that every
future edit to either implementation gets differentially tested before it lands."* It was right and
it had not been applied here. These tests apply it.

The full 3,220-pair differential cannot run in CI — `corpus_real.json` is gitignored and 73 MB.
What runs here is what is committed: the pinned pairs, both implementations, and the three
bookkeeping facts whose staleness is what let the divergence through.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from styxx.diffgate import gate_diff_text

ROOT = Path(__file__).resolve().parent.parent
INSTRUMENT = ROOT / "styxx" / "diffgate.py"
DIFFERENTIAL = ROOT / "web" / "gate" / "differential"
PY_SIDE = DIFFERENTIAL / "py_side.py"
CHECK_PAIRS = DIFFERENTIAL / "check_pairs.js"
GATE_README = ROOT / "web" / "gate" / "README.md"
PORT = ROOT / "web" / "gate" / "diffgate.js"


def instrument_sha() -> str:
    """LF-normalised, because a checkout on Windows carries CRLF and hashes differently."""
    return hashlib.sha256(INSTRUMENT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pairs_files_named_by_check_pairs() -> list[str]:
    m = re.search(r"for \(const name of \[(.*?)\]", CHECK_PAIRS.read_text(encoding="utf-8"), re.S)
    assert m, "check_pairs.js no longer lists its corpora in a form this test can read"
    return re.findall(r'"([^"]+)"', m.group(1))


# ---- (2): the pin ----------------------------------------------------------------------------

def test_the_differential_pin_matches_the_instrument():
    """The exact failure of 2026-09-18. A stale pin does not fail loudly; it fails to no one."""
    m = re.search(r'^PINNED\s*=\s*"([0-9a-f]{64})"', PY_SIDE.read_text(encoding="utf-8"), re.M)
    assert m, "py_side.py has no readable PINNED"
    assert m.group(1) == instrument_sha(), (
        "web/gate/differential/py_side.py pins a different styxx/diffgate.py than this checkout "
        "has. The differential will refuse to run, and refuse silently as far as CI is concerned. "
        "Move the pin in the same commit that moves the instrument."
    )


def test_the_gate_readme_names_the_same_instrument():
    """The README carries the hash too, and went stale beside the pin."""
    body = GATE_README.read_text(encoding="utf-8")
    found = set(re.findall(r"\b([0-9a-f]{64})\b", body))
    assert instrument_sha() in found, (
        "web/gate/README.md does not name this checkout's styxx/diffgate.py. It tells readers which "
        f"file the port was made from; it currently names {sorted(found) or 'nothing'}."
    )


# ---- (3): the corpus cannot be silently swallowed ---------------------------------------------

@pytest.mark.parametrize("name", pairs_files_named_by_check_pairs())
def test_every_named_pairs_file_exists(name):
    """check_pairs.js skips a missing corpus through exists() and still prints success."""
    p = DIFFERENTIAL / name
    assert p.is_file(), (
        f"check_pairs.js reads {name} but it is not in the checkout. If it was added recently, the "
        f"`.gitignore` in {DIFFERENTIAL.name}/ ignores *.json and must whitelist it by name, or the "
        f"file never lands and the pinned set silently shrinks."
    )


def test_gitignore_whitelists_every_named_pairs_file():
    ignore = (DIFFERENTIAL / ".gitignore").read_text(encoding="utf-8")
    allowed = {l[1:].strip() for l in ignore.splitlines() if l.startswith("!")}
    missing = [n for n in pairs_files_named_by_check_pairs() if n not in allowed]
    assert not missing, f"{missing} are read by check_pairs.js but not whitelisted in .gitignore"


# ---- (1): the guard actually runs, on both implementations ------------------------------------

def _pairs() -> list[tuple[str, dict]]:
    out = []
    for name in pairs_files_named_by_check_pairs():
        p = DIFFERENTIAL / name
        if p.is_file():
            for pair in json.loads(p.read_text(encoding="utf-8")):
                out.append((name, pair))
    return out


def test_there_are_pinned_pairs_to_check():
    """A guard that reaches nothing keeps passing while measuring nothing."""
    assert len(_pairs()) >= 36, f"only {len(_pairs())} pinned pairs found"


@pytest.mark.parametrize("name,pair", _pairs(), ids=lambda v: v if isinstance(v, str) else v.get("id", "?"))
def test_python_matches_each_pinned_expect_block(name, pair):
    g = gate_diff_text(pair["summary"], pair["diff"], run=None, strict=False)
    with_why = bool(pair["expect"]["claims"]) and len(pair["expect"]["claims"][0]) == 3
    got = [[c.kind, c.verdict, str(c.why)] if with_why else [c.kind, c.verdict] for c in g.claims]
    assert got == [list(c) for c in pair["expect"]["claims"]], f"{name}:{pair['id']}"
    assert g.verdict == pair["expect"]["verdict"]
    assert g.uncovered_sentences == pair["expect"]["uncovered_sentences"]


def test_the_javascript_port_agrees_on_every_pinned_pair():
    """check_pairs.js, run here rather than left as a command in a README."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; the port cannot be held to the pinned pairs here")
    r = subprocess.run([node, str(CHECK_PAIRS)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0, f"check_pairs.js reported disagreements:\n{r.stdout[-3000:]}"
    assert "0 disagreement(s)" in r.stdout, r.stdout[-500:]


def test_the_port_names_the_instrument_it_transliterates():
    found = set(re.findall(r"\b([0-9a-f]{64})\b", PORT.read_text(encoding="utf-8")))
    assert instrument_sha() in found, (
        "web/gate/diffgate.js states which styxx/diffgate.py it is a transliteration of, and it is "
        f"not this one. It names {sorted(found) or 'nothing'}."
    )
