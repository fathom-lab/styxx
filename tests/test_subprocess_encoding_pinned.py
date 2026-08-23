"""Every text-mode subprocess call in the shipped package pins its encoding.

`subprocess.run(..., text=True)` with no `encoding=` decodes using the platform preference —
cp1252 on Windows, UTF-8 on Linux — so the same code returns different strings on different
machines. Two failure modes, both quiet:

  * a byte undefined in cp1252 (0x81, 0x8D, 0x8F, 0x90, 0x9D — and 0x90 occurs inside ordinary
    UTF-8 sequences such as U+2010) raises UnicodeDecodeError inside subprocess's reader THREAD,
    so `communicate` returns with an empty stdout and the caller sees "no output";
  * everything else mojibakes, so substring tests silently miss.

This bit twice in this repo. `benchmarks/silent_pass` lost 7 of its 20 cases to the first mode and
reported them as "unavailable", which then surfaced as a detector miss. `styxx.agent_audit` reads
git DIFFS, and `git_show_diff_contains` returned MATCH=False for a substring genuinely present in
the diff — an auditor calling a truthful claim unsupported.

The AST test below is the one that matters: it guards the whole class rather than the three call
sites that happened to be found.
"""
import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_TEXT_KWARGS = {"text", "universal_newlines"}
_PINS = {"encoding", "errors"}


def _unpinned_calls(package: Path):
    out = []
    for f in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "subprocess"):
                continue
            if fn.attr not in ("run", "Popen", "check_output", "check_call", "call"):
                continue
            kw = {k.arg for k in node.keywords if k.arg}
            text_mode = bool(kw & _TEXT_KWARGS) or fn.attr == "check_output"
            if text_mode and not (kw & _PINS):
                out.append(f"{f.relative_to(ROOT).as_posix()}:{node.lineno}")
    return out


def test_shipped_package_pins_subprocess_encoding():
    offenders = _unpinned_calls(ROOT / "styxx")
    assert not offenders, (
        "text-mode subprocess call(s) without encoding=/errors= in the shipped package: "
        f"{offenders}. Pass encoding='utf-8', errors='replace' — otherwise the platform locale "
        "decides what this code reads, and on Windows it either mojibakes or empties stdout."
    )


def test_git_diff_substring_survives_a_non_ascii_needle(tmp_path):
    """The functional shape of the defect: a non-ASCII needle that IS in the diff must be found."""
    from styxx.agent_audit import _Checkers

    repo = tmp_path / "r"
    repo.mkdir()

    def git(*args, **kw):
        return subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", **kw)

    git("init", "-q")
    git("config", "user.email", "t@t.t")
    git("config", "user.name", "t")
    # U+2212 MINUS SIGN and U+2010 HYPHEN; the latter's UTF-8 encoding contains byte 0x90,
    # which is undefined in cp1252 and is what crashed the silent-pass loader.
    (repo / "m.py").write_text('SIGN = "−"  # ‐ hyphen\n', encoding="utf-8")
    git("add", "m.py")
    r = git("commit", "-qm", "add sign")
    if r.returncode != 0:
        pytest.skip(f"git unavailable in this environment: {r.stderr[:120]}")

    head = git("rev-parse", "HEAD").stdout.strip()
    ev = _Checkers()

    present, why = ev.git_show_diff_contains(repo, commit=head, file="m.py",
                                             substring="−")
    assert present, f"U+2212 is in the diff but was reported missing: {why[:200]}"

    present_ascii, _ = ev.git_show_diff_contains(repo, commit=head, file="m.py",
                                                 substring="SIGN")
    assert present_ascii, "the ASCII control needle must also be found"

    absent, _ = ev.git_show_diff_contains(repo, commit=head, file="m.py",
                                          substring="✓ not here")
    assert not absent, "a needle that is genuinely absent must still report no match"
