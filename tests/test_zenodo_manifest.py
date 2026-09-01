"""The mechanism that was missing.

Zenodo deposits live outside this repo's release path. Nothing failed when a
deposited file was later edited here, so the permanent record and the repo drifted
apart in silence -- twice, in the same commit, and only one of the two was noticed.

This test binds the two together:

  1. Every file `zenodo/MANIFEST.json` records as deposited must still hash to the
     value recorded there. Edit a deposited paper and this goes red until either
     the deposit is re-cut or the manifest records the divergence deliberately.
  2. Every Zenodo DOI named in CITATION.cff or README.md must appear in the
     manifest. A new citation cannot be introduced without a manifest entry saying
     what kind of DOI it is and what it points at.

On hashing. `core.autocrlf=true` in this repo and only `styxx/centroids/*.json` is
marked `-text`, so a text file's raw bytes are CRLF in a Windows checkout and LF in
a Linux CI checkout. Hashing raw bytes would make this test pass on one machine and
fail on the other -- which is exactly the failure .gitattributes already documents
for the centroid pins. Text entries are therefore compared on LF-normalized bytes;
binary entries (pdf, png, zip) are compared on raw bytes, where normalization would
be meaningless and destructive.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "zenodo" / "MANIFEST.json"

DOI_RE = re.compile(r"10\.5281/zenodo\.\d+")

# Files whose DOI citations are load-bearing: they are what a reader or a
# "Cite this repository" widget follows.
CITATION_SURFACES = ("CITATION.cff", "README.md")

VALID_KINDS = {"CONCEPT", "VERSION", "UNCHECKABLE"}
VALID_STATUSES = {
    "TRACKED",
    "DIVERGED",
    "NO_LOCAL_COUNTERPART",
    "UNCHECKABLE",
    "DRAFT_OR_RESERVED",
    "THIRD_PARTY",
}
VALID_CONTENT_KINDS = {"text", "binary"}


def load_manifest() -> dict:
    assert MANIFEST_PATH.exists(), (
        f"{MANIFEST_PATH} is missing. It is the only record of which local files "
        f"were deposited to Zenodo; without it nothing checks deposit drift."
    )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def tracked_files(manifest: dict):
    """Yield (doi, file_entry) for every file the manifest claims was deposited."""
    for deposit in manifest["deposits"]:
        for f in deposit.get("deposited_files") or []:
            yield deposit["doi"], f


def digest(path: Path, content_kind: str) -> str:
    raw = path.read_bytes()
    if content_kind == "text":
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def recorded_digest(f: dict) -> str:
    if f["content_kind"] == "text":
        return f["sha256_lf_normalized"]
    return f["sha256_local_bytes"]


# --------------------------------------------------------------------------
# 1. drift of deposited files
# --------------------------------------------------------------------------


def test_manifest_exists_and_parses():
    manifest = load_manifest()
    assert manifest["schema"] == "fathom/zenodo-manifest/v1"
    assert manifest["deposits"], "manifest records no deposits at all"


def test_every_tracked_file_still_exists():
    manifest = load_manifest()
    missing = [
        (doi, f["local_path"])
        for doi, f in tracked_files(manifest)
        if not (ROOT / f["local_path"]).exists()
    ]
    assert not missing, (
        "manifest tracks local counterparts of deposited files that are no longer "
        f"in the repo (moved or deleted): {missing}. A deposited file cannot simply "
        "vanish from the repo -- either restore it or record the move in the manifest."
    )


def test_tracked_files_match_recorded_hashes():
    """The drift guard. Red means repo and deposit no longer agree."""
    manifest = load_manifest()
    drifted = []
    for doi, f in tracked_files(manifest):
        path = ROOT / f["local_path"]
        if not path.exists():
            continue  # reported by test_every_tracked_file_still_exists
        actual = digest(path, f["content_kind"])
        expected = recorded_digest(f)
        if actual != expected:
            drifted.append(
                f"{f['local_path']} (deposited in {doi}): "
                f"recorded {expected[:16]}... actual {actual[:16]}..."
            )
    assert not drifted, (
        "Local files have drifted from the bytes the manifest records as deposited:\n  "
        + "\n  ".join(drifted)
        + "\n\nThis is the condition that went unnoticed twice: a deposited paper was "
        "edited in the repo and the permanent Zenodo record kept asserting the old "
        "text. Either cut a new deposit version and update the manifest, or record "
        "the divergence deliberately (status DIVERGED with evidence) and update the "
        "recorded hash to the new repo bytes."
    )


def test_recorded_hashes_are_wellformed():
    manifest = load_manifest()
    bad = []
    for doi, f in tracked_files(manifest):
        assert f["content_kind"] in VALID_CONTENT_KINDS, (
            f"{f['local_path']}: content_kind {f['content_kind']!r} not in "
            f"{sorted(VALID_CONTENT_KINDS)}"
        )
        d = recorded_digest(f)
        if not (isinstance(d, str) and re.fullmatch(r"[0-9a-f]{64}", d)):
            bad.append((doi, f["local_path"], d))
        if f["content_kind"] == "binary" and f["sha256_lf_normalized"] is not None:
            bad.append(
                (doi, f["local_path"], "binary entry must not carry an LF-normalized hash")
            )
    assert not bad, f"malformed hash records: {bad}"


# --------------------------------------------------------------------------
# 2. citation surfaces may not name a DOI the manifest does not know
# --------------------------------------------------------------------------


def dois_in(relpath: str) -> set[str]:
    text = (ROOT / relpath).read_text(encoding="utf-8", errors="replace")
    return set(DOI_RE.findall(text))


@pytest.mark.parametrize("surface", CITATION_SURFACES)
def test_citation_surface_dois_are_all_in_manifest(surface):
    manifest = load_manifest()
    known = {d["doi"] for d in manifest["deposits"]}
    cited = dois_in(surface)
    assert cited, f"{surface} names no Zenodo DOI at all -- did the file move?"
    unknown = sorted(cited - known)
    assert not unknown, (
        f"{surface} names Zenodo DOIs that {MANIFEST_PATH.name} does not record: "
        f"{unknown}\n\nA DOI on a citation surface is what strangers follow. Every one "
        "of them needs a manifest entry saying whether it is a CONCEPT or a VERSION "
        "DOI, what it points at, and whether the deposited content still matches this "
        "repo. Add the entry -- do not delete the citation to make this pass."
    )


def test_manifest_records_which_citation_lines_are_wrong():
    """The manifest is also the record of known-bad citation lines.

    The agent that built it was not permitted to edit CITATION.cff or README --
    rewriting a citation identifier is the operator's call -- so the defects are
    recorded instead. This test asserts the record is still populated and still
    points at real files, so the defects cannot be quietly dropped.
    """
    manifest = load_manifest()
    defects = manifest["citation_surface_defects"]
    assert defects, (
        "citation_surface_defects is empty. If the defects were actually fixed, delete "
        "this test in the same commit as the fix, so the deletion is reviewable."
    )
    for d in defects:
        assert (ROOT / d["file"]).exists(), f"defect {d['id']} names a missing file {d['file']}"
        assert d["defect"], f"defect {d['id']} has no description"
        assert d["evidence"], f"defect {d['id']} has no evidence"


# --------------------------------------------------------------------------
# 3. manifest hygiene
# --------------------------------------------------------------------------


def test_no_duplicate_dois():
    manifest = load_manifest()
    dois = [d["doi"] for d in manifest["deposits"]]
    dupes = sorted({d for d in dois if dois.count(d) > 1})
    assert not dupes, f"manifest records the same DOI more than once: {dupes}"


def test_kinds_and_statuses_are_from_the_vocabulary():
    manifest = load_manifest()
    for d in manifest["deposits"]:
        assert d["kind"] in VALID_KINDS, f"{d['doi']}: kind {d['kind']!r}"
        assert d["status"] in VALID_STATUSES, f"{d['doi']}: status {d['status']!r}"


def test_every_deposit_says_where_its_kind_came_from():
    """No DOI kind may be asserted without naming the in-repo evidence for it."""
    manifest = load_manifest()
    unsourced = [
        d["doi"] for d in manifest["deposits"] if not (d.get("kind_evidence") or "").strip()
    ]
    assert not unsourced, (
        f"deposits assert a DOI kind with no stated evidence: {unsourced}. "
        "Guessing CONCEPT vs VERSION from the number is how the current mislabels "
        "happened."
    )


def test_diverged_deposits_explain_themselves():
    manifest = load_manifest()
    for d in manifest["deposits"]:
        if d["status"] != "DIVERGED":
            continue
        files = d.get("deposited_files") or []
        assert any("DIVERGED" in (f.get("note") or "") for f in files), (
            f"{d['doi']} is marked DIVERGED but no file entry says which file diverged "
            "or on what evidence"
        )
