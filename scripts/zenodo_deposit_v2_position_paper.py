"""Prepare v2 of the Every Mind Leaves Vitals position paper deposit.

WHY THIS EXISTS
---------------
10.5281/zenodo.19777921 (version label `position-paper-v1`, deposited
2026-04-26) is named as the preferred citation in CITATION.cff line 46. Its
content predates the 2026-06-21 scope erratum that now heads the repo copy of
`papers/every-mind-leaves-vitals.md`. The erratum bounds exactly the two claims
the deposit's own description asserts most loudly -- that the result is "a
property of cognition rather than of language models", and that "the detectors
transfer across model families". A reader who follows the preferred citation
today gets the unbounded claims and no erratum.

This script prepares a new version under the same concept record so the erratum
travels with the paper, and repeats the erratum text in the deposit description
so a reader of the record sees it without opening the file.

WHAT IT WILL NOT DO
-------------------
It never publishes. There is no flag that publishes. `actions/publish` is not
called anywhere in this file, by design: deposits are permanent, made under a
real name, and the repo's own doctrine
(`papers/DEPOSIT_frame_locality_2026_07_28.md`) is that the agent prepares and
the operator fires. The most this script will do is leave an unpublished draft
in the operator's Zenodo account with the files uploaded and the metadata set,
and print the edit URL.

It also does not touch `papers/every-mind-leaves-vitals.md`. The paper goes up
as-is, erratum block and all -- that block is the point.

USAGE
-----
    ZENODO_TOKEN=... python scripts/zenodo_deposit_v2_position_paper.py
        Dry run. No network calls. Prints the manifest, the file hashes, the
        extracted erratum, and the exact metadata that would be sent.

    ZENODO_TOKEN=... python scripts/zenodo_deposit_v2_position_paper.py --prepare-draft
        Creates the new-version draft, uploads the files, sets the metadata,
        and STOPS. The operator reviews the draft in the browser and presses
        Publish themselves, or discards it.

ZENODO_TOKEN is required in both modes. The dry run refuses without it on
purpose: a dry run that works without credentials is a dry run of a different
program than the one that will actually run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# The manifest and the paper both contain characters (the erratum's warning sign,
# "read≠write", em dashes) that a cp1252 Windows console cannot encode. Printing is
# the whole point of the dry run, so make stdout able to carry them rather than
# letting a UnicodeEncodeError abort the report half-written.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # non-reconfigurable stream (pipe, capture)
        pass

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "zenodo" / "MANIFEST.json"

ZENODO_API = "https://zenodo.org/api"

# The concept record whose chain holds the manifesto and the position paper.
# Evidence: release/zenodo-deposit-receipt-emlv-versioned.json (concept_recid).
CONCEPT_RECID = "19703526"

# The latest published version in that chain, and the one CITATION.cff points at.
# A new version is created from the latest published deposition id.
PARENT_DEP_ID = 19777921
PARENT_DOI = "10.5281/zenodo.19777921"

# The orphan that 19777921 superseded. Recorded, not touched.
ORPHAN_DOI = "10.5281/zenodo.19777361"

PAPER = ROOT / "papers" / "every-mind-leaves-vitals.md"
CHART = ROOT / "release" / "phase-transition-chart.png"
FILES_TO_UPLOAD = [PAPER, CHART]

NEW_VERSION_LABEL = "position-paper-v2-erratum"


# ---------------------------------------------------------------------------
# hashing, kept identical to tests/test_zenodo_manifest.py
# ---------------------------------------------------------------------------

def digest(path: Path, content_kind: str) -> str:
    raw = path.read_bytes()
    if content_kind == "text":
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        die(f"{MANIFEST_PATH} is missing. Refusing to deposit without the manifest: "
            "it is the only in-repo record of what was already deposited.")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def die(msg: str, code: int = 2) -> None:
    print(f"\nREFUSING: {msg}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# the erratum
# ---------------------------------------------------------------------------

def extract_erratum(md_path: Path) -> list[str]:
    """Return the erratum blockquote from the paper, as lines with '>' stripped.

    The block is identified by content, not by line number, so it survives edits
    above it. If it is not found the script refuses: depositing this paper
    without the erratum is the exact failure this script exists to repair.
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(">") and re.search(
            r"correction|erratum", ln, re.IGNORECASE
        ):
            start = i
            break
    if start is None:
        die(f"no erratum blockquote found in {md_path.name}. This script only exists "
            "to carry that block into the deposit; if the block is gone, stop and "
            "find out why before depositing anything.")
    out: list[str] = []
    for ln in lines[start:]:
        s = ln.rstrip()
        if not s.lstrip().startswith(">"):
            break
        out.append(re.sub(r"^\s*>\s?", "", s))
    if len(out) < 3:
        die("the erratum block found is suspiciously short "
            f"({len(out)} lines); refusing to deposit a truncated correction.")
    return out


def md_inline_to_html(s: str) -> str:
    """Minimal, deliberately dumb markdown->HTML for the erratum lines.

    Handles only what the erratum actually uses: bold, italic, inline code, and
    markdown links. Anything it does not understand passes through as text, which
    is the safe direction -- a stray asterisk in a Zenodo description is a
    cosmetic defect; a swallowed clause is a content defect.
    """
    s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def erratum_to_html(erratum_lines: list[str]) -> str:
    """Render the erratum as an HTML block for the deposit description."""
    html: list[str] = []
    in_list = False
    for raw in erratum_lines:
        line = raw.strip()
        if not line:
            if in_list:
                html.append("</ul>")
                in_list = False
            continue
        if line.startswith("- "):
            if not in_list:
                html.append("<ul>")
                in_list = True
            html.append(f"<li>{md_inline_to_html(line[2:])}</li>")
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<p>{md_inline_to_html(line)}</p>")
    if in_list:
        html.append("</ul>")
    return "\n".join(html)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def build_description(erratum_html: str) -> str:
    return f"""\
<p><strong>This version exists to carry a correction. Read it first.</strong></p>

<div style="border-left:4px solid #999;padding-left:1em">
{erratum_html}
</div>

<p><em>The text of the paper below is unchanged from
<a href="https://doi.org/{PARENT_DOI}">{PARENT_DOI}</a> apart from the
correction block, which now heads the file. Nothing was deleted to make the
paper look better; the original claims stand in place, bounded.</em></p>

<hr>

<p><strong>Every mind leaves vitals.</strong></p>

<p>This paper extends the Cognometry Manifesto beyond LLMs. It reports that
three calibrated cognometric instruments &mdash; hallucination, refusal,
tool-call drift &mdash; exhibit phase-transition structure under feature-count
ablation: detection does not scale smoothly with classifier capacity but jumps
discretely at a critical feature, replicated across three independent feature
bases.</p>

<p>The paper's central leap &mdash; that this is a property of cognition rather
than of language models &mdash; and its claim of cross-family transfer are the
two things the correction block above bounds. They were bounded by this
program's own pre-registered experiments, not by outside criticism. The
phase-transition result and the constitutional commitments are unaffected.</p>

<p>The paper records six constitutional commitments for cognometric instruments
shipped under the Fathom name &mdash; MIT license, weights and reproducers
in-tree, failure modes declared in-weights, calibration fingerprint required,
CPU and browser-runnable, no private detectors under the Fathom name &mdash;
written so that if they are ever broken, the paper makes the breaking
visible.</p>

<p><strong>Reproduce.</strong>
<a href="https://github.com/fathom-lab/styxx">github.com/fathom-lab/styxx</a></p>
"""


def build_metadata(erratum_html: str) -> dict:
    return {
        "metadata": {
            "title": (
                "Every Mind Leaves Vitals: On the Cognometric Layer, "
                "Substrate-Independence, and the One-Time Choice We Have "
                "[corrected edition]"
            ),
            "version": NEW_VERSION_LABEL,
            "upload_type": "publication",
            "publication_type": "workingpaper",
            "description": build_description(erratum_html),
            "creators": [
                {"name": "Rodabaugh, Alexander", "affiliation": "Fathom Lab"},
            ],
            "keywords": [
                "cognometry",
                "cognitive observability",
                "calibration fingerprint",
                "phase transitions",
                "hallucination detection",
                "refusal detection",
                "tool-call drift",
                "AI safety",
                "interpretability",
                "measurement standard",
                "open science",
                "position paper",
                "erratum",
                "styxx",
            ],
            "language": "eng",
            "access_right": "open",
            "license": "cc-by-4.0",
            "notes": (
                "Corrected edition. Supersedes "
                f"{PARENT_DOI}, whose text asserts substrate-independence and "
                "cross-family transfer without the 2026-06-21 scope erratum. The "
                "erratum is reproduced in this record's description so it is visible "
                "without opening the file. Reproducers at github.com/fathom-lab/styxx."
            ),
            "related_identifiers": [
                {
                    "identifier": PARENT_DOI,
                    "relation": "isNewVersionOf",
                    "resource_type": "publication-workingpaper",
                    "scheme": "doi",
                },
                {
                    "identifier": ORPHAN_DOI,
                    "relation": "isAlternateIdentifier",
                    "resource_type": "publication-workingpaper",
                    "scheme": "doi",
                },
                {
                    "identifier": "https://github.com/fathom-lab/styxx",
                    "relation": "isSupplementedBy",
                    "resource_type": "software",
                    "scheme": "url",
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_manifest(manifest: dict) -> None:
    """Print the FULL manifest with hashes before doing anything at all."""
    rule("ZENODO MANIFEST — every deposit this repo cites")
    print(f"source:  {MANIFEST_PATH}")
    print(f"schema:  {manifest['schema']}   generated: {manifest['generated']}")
    print(f"commit:  {manifest.get('generated_from_commit')}")
    print(f"deposits: {len(manifest['deposits'])}")
    print()
    for d in manifest["deposits"]:
        print(f"  {d['doi']:32}  {d['kind']:12} {d['status']}")
        print(f"      {d['points_at']}")
        for f in d.get("deposited_files") or []:
            path = ROOT / f["local_path"]
            live = digest(path, f["content_kind"]) if path.exists() else None
            recorded = (
                f["sha256_lf_normalized"] if f["content_kind"] == "text"
                else f["sha256_local_bytes"]
            )
            if live is None:
                mark = "MISSING "
            elif live == recorded:
                mark = "match   "
            else:
                mark = "DRIFTED "
            print(f"      [{mark}] {recorded}  {f['local_path']}")
        print()

    rule("KNOWN-BAD CITATION LINES (recorded, not fixed — operator's call)")
    for x in manifest["citation_surface_defects"]:
        line = f":{x['line']}" if x.get("line") else ""
        print(f"  {x['id']}  {x['file']}{line}")
        print(f"      defect:     {x['defect']}")
        if x.get("should_say"):
            print(f"      should say: {x['should_say']}")
        print()


def check_upload_files_against_manifest(manifest: dict) -> None:
    """Every file we are about to upload must be one the manifest knows about."""
    by_path = {}
    for d in manifest["deposits"]:
        for f in d.get("deposited_files") or []:
            by_path.setdefault(f["local_path"], []).append((d["doi"], f))

    rule("FILES THIS SCRIPT WOULD UPLOAD")
    for p in FILES_TO_UPLOAD:
        rel = p.relative_to(ROOT).as_posix()
        if not p.exists():
            die(f"{rel} does not exist")
        recs = by_path.get(rel)
        if not recs:
            die(f"{rel} is not recorded in the manifest as a deposited file. "
                "Add it to the manifest first: uploading a file the manifest does not "
                "track recreates the exact blind spot this work is repairing.")
        doi, f = recs[0]
        live_raw = hashlib.sha256(p.read_bytes()).hexdigest()
        live = digest(p, f["content_kind"])
        recorded = (
            f["sha256_lf_normalized"] if f["content_kind"] == "text"
            else f["sha256_local_bytes"]
        )
        state = "matches manifest" if live == recorded else "DRIFTED FROM MANIFEST"
        print(f"  {rel}")
        print(f"      {p.stat().st_size} bytes, content_kind={f['content_kind']}")
        print(f"      sha256 (raw bytes as they will be uploaded): {live_raw}")
        print(f"      sha256 (manifest identity):                  {live}  [{state}]")
        print(f"      recorded in manifest under {doi}")
        if live != recorded:
            die(f"{rel} no longer matches the hash the manifest records. Either the "
                "manifest is stale or the file changed unexpectedly. Resolve that "
                "before depositing -- do not deposit bytes nobody has accounted for.")
        print()


# ---------------------------------------------------------------------------
# the (never-publishing) network path
# ---------------------------------------------------------------------------

def prepare_draft(token: str, metadata: dict) -> None:
    try:
        import requests
    except ImportError:
        die("requests is not installed; cannot prepare the draft.")

    params = {"access_token": token}

    rule("PREPARING DRAFT (this script never publishes)")
    print(f"creating a new version under deposition {PARENT_DEP_ID} "
          f"(concept recid {CONCEPT_RECID})...")
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions/{PARENT_DEP_ID}/actions/newversion",
        params=params, timeout=30,
    )
    if r.status_code >= 400:
        die(f"newversion failed: {r.status_code} {r.text[:400]}", code=1)

    draft_url = r.json().get("links", {}).get("latest_draft")
    if not draft_url:
        die("no latest_draft link in the newversion response", code=1)

    draft = requests.get(draft_url, params=params, timeout=30).json()
    dep_id = draft["id"]
    bucket = draft["links"]["bucket"]
    print(f"  draft deposition id: {dep_id}")

    print("\nclearing files inherited from the previous version...")
    for f in draft.get("files", []):
        rd = requests.delete(f["links"]["self"], params=params, timeout=30)
        print(f"  deleted {f.get('filename', '?')}: {rd.status_code}")

    print("\nuploading...")
    for p in FILES_TO_UPLOAD:
        with open(p, "rb") as fh:
            ru = requests.put(f"{bucket}/{p.name}", data=fh, params=params, timeout=300)
        if ru.status_code >= 400:
            die(f"upload {p.name} failed: {ru.status_code} {ru.text[:300]}", code=1)
        print(f"  uploaded {p.name} ({p.stat().st_size} bytes)")

    print("\nsetting metadata...")
    rm = requests.put(
        f"{ZENODO_API}/deposit/depositions/{dep_id}", params=params,
        json=metadata, timeout=60,
    )
    if rm.status_code >= 400:
        die(f"metadata failed: {rm.status_code} {rm.text[:800]}", code=1)
    print("  metadata attached")

    rule("DRAFT PREPARED — NOT PUBLISHED")
    print(f"  review it:  https://zenodo.org/deposit/{dep_id}")
    print()
    print("  This script does not call actions/publish and has no flag that does.")
    print("  Read the draft, then press Publish yourself, or discard it.")
    print()
    print("  After publishing, the version DOI Zenodo assigns must be:")
    print("    - added to zenodo/MANIFEST.json as a new VERSION entry under")
    print("      concept 10.5281/zenodo.19703526, with the deposited file hashes;")
    print("    - the old entry 10.5281/zenodo.19777921 left in place and its")
    print("      DIVERGED status updated to say which version superseded it;")
    print("    - decided-on for CITATION.cff line 46 (defect D2) — the new version")
    print("      DOI, or the concept DOI, is your call, not the agent's.")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare (never publish) v2 of the position-paper deposit.",
    )
    ap.add_argument(
        "--prepare-draft", action="store_true",
        help="actually create the unpublished draft on Zenodo (still never publishes)",
    )
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        die("ZENODO_TOKEN is not set. Required in dry-run mode too: a dry run that "
            "works without credentials is a dry run of a different program than the "
            "one that will actually run.")

    manifest = load_manifest()

    # Everything below prints before anything is sent anywhere.
    print_manifest(manifest)
    check_upload_files_against_manifest(manifest)

    erratum_lines = extract_erratum(PAPER)
    rule("ERRATUM EXTRACTED FROM THE PAPER (will be carried in the description)")
    for ln in erratum_lines:
        print(f"  {ln}")

    erratum_html = erratum_to_html(erratum_lines)
    metadata = build_metadata(erratum_html)

    rule("METADATA THAT WOULD BE SENT")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    if not args.prepare_draft:
        rule("DRY RUN — nothing was sent")
        print("  No network call was made.")
        print("  Re-run with --prepare-draft to create the unpublished draft.")
        print("  There is no flag that publishes. That step is the operator's, in the")
        print("  browser, on a draft they have read.")
        return

    prepare_draft(token, metadata)


if __name__ == "__main__":
    main()
