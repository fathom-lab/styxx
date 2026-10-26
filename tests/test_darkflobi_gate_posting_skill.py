"""The posting skill's committed text, papers/darkflobi-gate/styxx-post_SKILL.md, beside its AMENDMENT
(AMENDMENT_darkflobi_gate_posting_skill_2026_10_25.md), on CPU, offline, with nothing of the gate.

What is pinned here: the text a stranger reads is the text the agent is told. Its LF-normalised sha256 is the lock row
the AMENDMENT writes for `W:skills/styxx-post/SKILL.md` (the row parsed by `lock_hashes.parse_lock_rows`, so the row
regex reads it); it opens with clawdbot's skill frontmatter (`name: styxx-post`, one `description:` line); it names the
pin's 40 hex and tells the agent to cite the pin or an ancestor of it, never `origin/main` (PREREG 15.11); it names the
guard's three readings; it is UTF-8 with no byte-order mark and no CR in the blob; this checkout's
`scripts/surface_lint.py --files` reads 0 hits over it and over the AMENDMENT (its withdrawn wordings stand inside
their withdrawal context, which the lint reads as allowed); and none of the three files prints a word of the lab's
BANNED tuple. When a caller names the box (`STYXX_GATE_W_ROOT`), the placed file equals the committed one.
What this does not check: whether the agent follows the skill, whether the gate's guard works (the gate's own tests
under `W:gate/tests/` do), or anything a sentence of the skill says.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "papers" / "darkflobi-gate"
SKILL = FOLDER / "styxx-post_SKILL.md"
AMENDMENT = FOLDER / "AMENDMENT_darkflobi_gate_posting_skill_2026_10_25.md"
LOCK_PY = FOLDER / "lock_hashes.py"
TEST_FILE = Path(__file__).resolve()
PIN = "3437340b9e944644372e41d9fcab735d47e20fa0"
SKILL_ROW = "W:skills/styxx-post/SKILL.md"


def _lf(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


@pytest.fixture(scope="module")
def lock_mod():
    spec = importlib.util.spec_from_file_location("lock_hashes_for_skill", LOCK_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_committed_skill_is_the_amendments_row(lock_mod):
    rows = lock_mod.parse_lock_rows(AMENDMENT.read_text(encoding="utf-8"))
    assert SKILL_ROW in rows, sorted(rows)
    want, label = rows[SKILL_ROW]                     # parse_lock_rows: path -> (digest, label)
    assert label == "the posting skill"
    assert re.fullmatch(r"[0-9a-f]{64}", want), want
    assert hashlib.sha256(_lf(SKILL)).hexdigest() == want


def test_the_amendment_parses_to_its_seven_rows(lock_mod):
    rows = lock_mod.parse_lock_rows(AMENDMENT.read_text(encoding="utf-8"))
    assert sorted(rows) == sorted([
        "W:gate/gate.py", "W:gate/gate_verify.py", "W:gate/README_exhibit.md", "W:gate/tests/test_prepared_guard.py",
        "W:gate/tests/test_gate.py", "W:gate/tests/test_gate_round2.py", SKILL_ROW])


def test_the_skill_opens_with_clawdbots_frontmatter():
    text = _lf(SKILL).decode("utf-8")
    lines = text.split("\n")
    assert lines[0] == "---"
    end = lines.index("---", 1)
    front = lines[1:end]
    assert front[0] == "name: styxx-post"
    assert len(front) == 2 and front[1].startswith("description: ") and len(front[1]) > 80


def test_the_skill_names_the_pin_and_the_guard():
    text = _lf(SKILL).decode("utf-8")
    assert PIN in text
    assert "or one exact ancestor of it" in text
    assert "Never\n  `origin/main`" in text or "Never `origin/main`" in text
    assert "copied as the object writes it" in text
    for code in ("record_dry_run", "record_url_not_live", "record_receipts_moved"):
        assert code in text, code
    assert "never `clawdbot send` from a\nshell" in text or "Never `clawdbot send` from a\nshell" in text


@pytest.mark.parametrize("path", [SKILL, AMENDMENT, TEST_FILE], ids=lambda p: p.name)
def test_utf8_without_a_bom_or_a_cr_in_the_blob(path):
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in _lf(path)
    raw.decode("utf-8")


def test_the_surface_lint_reads_no_hit():
    lint = ROOT / "scripts" / "surface_lint.py"
    if not lint.exists():
        pytest.skip("scripts/surface_lint.py is not in this checkout")
    r = subprocess.run([sys.executable, str(lint), "--files", str(SKILL), str(AMENDMENT), "--json"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    report = json.loads(r.stdout)
    assert report["files_read"] == 2
    assert report["hits"] == [], report["hits"]


def test_the_lab_never_prints_these_words():
    src = ROOT / "scripts" / "zenodo_publish_v7_48_0.py"
    if not src.exists():
        pytest.skip("the lab's BANNED tuple is not in this checkout")
    m = re.search(r"^BANNED = \((.*?)\)$", src.read_text(encoding="utf-8"), re.S | re.M)
    assert m, "BANNED tuple not found"
    words = re.findall(r'"([^"]+)"', m.group(1))
    assert len(words) >= 8
    banned = re.compile(r"\b(" + "|".join(re.escape(w) + "s?" for w in words) + r")\b", re.I)
    for p in (SKILL, AMENDMENT, TEST_FILE):
        hits = [h.group(0) for h in banned.finditer(p.read_bytes().decode("utf-8"))]
        assert not hits, f"{p.name}: {hits}"


def test_the_placed_skill_is_the_committed_one_when_the_box_is_named():
    w_root = os.environ.get("STYXX_GATE_W_ROOT")
    if not w_root:
        pytest.skip("STYXX_GATE_W_ROOT names no box")
    placed = Path(w_root) / "skills" / "styxx-post" / "SKILL.md"
    if not placed.exists():
        pytest.skip("the skill is not placed on the named box")
    assert _lf(placed) == _lf(SKILL)
