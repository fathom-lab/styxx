"""Every registry manifest parses, carries its required fields, and claims no certification.

`registry/SUBMIT.md` tells a submitter that CI checks the file parses and the required fields are
present. This is that check. It exists because writing the promise without the test would be the
exact defect this repository publishes papers about: a claim in prose with nothing behind it.

The last test is the one worth reading. No conformance suite for spec v1.0.0 exists, so no
`registry_token` can mean anything, so a manifest that carries one is asserting a certification
nobody can check. That is refused here rather than in review, because review is a person and this
is not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:                                    # 3.11+
    import tomllib
except ModuleNotFoundError:             # 3.9 / 3.10
    try:
        import tomli as tomllib         # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None                  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "registry" / "manifests"

REQUIRED = {
    "impl": ("name", "language", "maintainer", "url", "contact"),
    "conformance": ("spec_version",),
    "scope": ("modes",),
}
KNOWN_MODES = {"direct", "proxy"}


def manifests() -> list[Path]:
    return sorted(MANIFESTS.glob("*.toml"))


def load(p: Path) -> dict:
    with p.open("rb") as fh:
        return tomllib.load(fh)


def test_the_manifests_directory_exists():
    """It is empty today. It still has to exist, or SUBMIT.md tells people to write into nothing."""
    assert MANIFESTS.is_dir(), f"{MANIFESTS} is missing; registry/SUBMIT.md points submitters at it"


@pytest.mark.skipif(tomllib is None, reason="no TOML parser on this interpreter")
@pytest.mark.parametrize("path", manifests(), ids=lambda p: p.name)
def test_a_manifest_parses_and_has_its_required_fields(path: Path):
    data = load(path)
    for table, keys in REQUIRED.items():
        assert table in data, f"{path.name}: missing [{table}]"
        for key in keys:
            assert data[table].get(key) not in (None, "", []), \
                f"{path.name}: [{table}] {key} is required and must not be empty"


@pytest.mark.skipif(tomllib is None, reason="no TOML parser on this interpreter")
@pytest.mark.parametrize("path", manifests(), ids=lambda p: p.name)
def test_the_filename_matches_the_declared_name(path: Path):
    """So a reader scanning the directory sees the same names the manifests claim."""
    assert load(path)["impl"]["name"] == path.stem, \
        f"{path.name}: [impl] name must equal the filename stem"


@pytest.mark.skipif(tomllib is None, reason="no TOML parser on this interpreter")
@pytest.mark.parametrize("path", manifests(), ids=lambda p: p.name)
def test_modes_are_drawn_from_the_closed_set(path: Path):
    modes = set(load(path)["scope"]["modes"])
    assert modes and modes <= KNOWN_MODES, \
        f"{path.name}: [scope] modes must be a non-empty subset of {sorted(KNOWN_MODES)}"


@pytest.mark.skipif(tomllib is None, reason="no TOML parser on this interpreter")
@pytest.mark.parametrize("path", manifests(), ids=lambda p: p.name)
def test_no_manifest_claims_a_certification_that_cannot_be_checked(path: Path):
    """The registry cannot certify anything until a conformance suite for the spec exists.

    Deleting this test is the cheap way to let a certification claim through, so if it ever goes,
    the suite it is waiting on had better have arrived first. See registry/README.md.
    """
    conformance = load(path).get("conformance", {})
    assert "registry_token" not in conformance, (
        f"{path.name}: [conformance] registry_token is not issued and not recognised — "
        "no conformance suite exists for spec v1.0.0 (registry/README.md)"
    )


def test_the_submission_instructions_still_disclaim_certification():
    """SUBMIT.md's honesty about the missing gate is load-bearing; a silent edit should fail here."""
    text = (ROOT / "registry" / "SUBMIT.md").read_text(encoding="utf-8")
    assert "does not exist yet" in text, \
        "registry/SUBMIT.md must keep stating that the conformance suite does not exist"
