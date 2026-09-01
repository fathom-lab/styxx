"""A tag that ships the previous version is a SILENT PASS, and it has happened here.

From the release-path commit that has been sitting unpushed since 2026-08-19, blocked on a
personal access token without `workflow` scope:

    skip-existing:true made a mistagged release a silent green no-op (tag vX.Y.Z on a tree still
    versioned X.Y.(Z-1) builds the old version, PyPI skips the duplicate, everything is green,
    nothing shipped).

That is SP-1 from `benchmarks/silent_pass/CORPUS.md` wearing a release badge: the outcome the
pipeline exists to produce did not happen, and every check was green.

The full repair lives in `.github/workflows/publish.yml` and cannot be pushed from here. **This is
the half that does not need it.** The workflow fix gates publishing on the tag matching the
version *before* building; this test catches the same mismatch from ordinary CI, on the next run
after a bad tag, without any workflow change at all.

It is strictly weaker than the workflow fix — it fires after the fact rather than before the
upload — and it is not a substitute for landing that patch. It is what can be enforced today.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_SEMVER = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


def _declared_version() -> str:
    text = (ROOT / "styxx" / "_version.py").read_text(encoding="utf-8")
    m = re.search(r"__version__\s*=\s*[\"']([^\"']+)", text)
    assert m, "styxx/_version.py does not declare __version__"
    return m.group(1)


def _parse(v: str):
    m = _SEMVER.match(v.strip())
    return tuple(int(g) for g in m.groups()) if m else None


def _newest_tag():
    """The highest v* tag visible to this checkout, or None."""
    try:
        r = subprocess.run(["git", "tag", "--list", "v*", "--sort=-v:refname"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if _parse(line):
            return line.strip()
    return None


def test_declared_version_is_parseable():
    assert _parse(_declared_version()), f"unparseable version {_declared_version()!r}"


def test_version_is_never_behind_the_newest_tag():
    """The mistag catcher.

    If someone tags v7.46.0 on a tree whose `_version.py` still says 7.45.0, the build produces
    7.45.0, PyPI skips it as a duplicate, and the release is green and empty. After such a tag the
    newest visible tag is AHEAD of the declared version, and that is what this asserts against.
    """
    tag = _newest_tag()
    if tag is None:
        pytest.skip("no v* tags visible in this checkout (shallow clone or fresh fork)")
    declared, tagged = _parse(_declared_version()), _parse(tag)
    assert declared >= tagged, (
        f"styxx/_version.py declares {_declared_version()} but the newest tag is {tag}. "
        f"A tag ahead of the declared version means the release built the PREVIOUS version; "
        f"with skip-existing the upload is then a silent green no-op and nothing shipped. "
        f"Bump styxx/_version.py and re-tag.")


def test_a_version_behind_its_tag_is_detected():
    """The guard must be able to fail, or it is a leg that cannot fail.

    Pins the comparison itself rather than the repository's current state, which is expected to
    be clean and therefore proves nothing about whether the check works.
    """
    assert _parse("7.45.0") < _parse("v7.46.0"), "a mistag must compare as behind"
    assert _parse("7.46.0") >= _parse("v7.46.0"), "an exact release tag must pass"
    assert _parse("7.46.1") >= _parse("v7.46.0"), "a version ahead of the tag must pass"


def test_tag_parsing_tolerates_the_v_prefix_and_rejects_junk():
    assert _parse("v1.2.3") == (1, 2, 3)
    assert _parse("1.2.3") == (1, 2, 3)
    assert _parse("nightly") is None
