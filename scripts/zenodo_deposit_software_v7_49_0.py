"""Prepare styxx v7.49.0 as the next version of the styxx SOFTWARE concept record on Zenodo.

Concept record: 10.5281/zenodo.19758618. Its latest version on the lab's receipts is styxx 7.48.1,
10.5281/zenodo.23200977 (release/zenodo-deposit-receipt-software-v7.48.1.json, published
2026-10-07T02:12Z, 2026-10-06 at UTC-4). Adapted from scripts/zenodo_deposit_software_v7_48_1.py.

WHAT IT DOES
------------
  1. Checks everything it can offline, and stops on any failure: the three files against their
     pinned sizes, sha256 and md5; the source bundle against the v7.49.0 tree blob by blob; the
     wheel and sdist contents against the tag's blobs; metadata.json against the 7.48.1 record as
     published (and as read back), and every claim of its description against the tag
     (CHANGELOG.md's [7.49.0] section, the [7.48.1] section for the EXTERNAL-1 line, the files the
     description names, and the facts it states about them); the resource_type of the spec, the
     series and the isNewVersionOf rows against DataCite's record (datacite-relations-7.49.0.json).
  2. Asks Zenodo which record is the LATEST version of concept 19758618, and proceeds only if it
     is 23200977 (7.48.1): metadata.json names that record as the version before this one
     (isNewVersionOf, the description and the notes), so a draft made from any other record would
     carry a false relation. Any other latest version stops it; rebuild the metadata before a run.
  3. Refuses if an unpublished draft already exists in that concept, unless --resume-draft names
     that draft explicitly.
  4. POSTs actions/newversion on the latest version, deletes the files the new draft inherited,
     uploads the three files, PUTs metadata.json, reads the draft back and compares every field and
     Zenodo's md5 and size for every file with the local bytes; any metadata key it did not send
     (other than doi, prereserve_doi and imprint_publisher, which Zenodo adds) stops it.
  5. Writes zenodo-draft-receipt-software-v7.49.0.json beside this file and STOPS.

WHAT IT WILL NOT DO
-------------------
It never publishes. Every request goes through an allowlist of method + path, a publish, edit or
discard action is not on it, and the mutating requests are further pinned to the one record they
may touch (newversion on the latest version only; DELETE and PUT on the new draft and its bucket
only, and no DELETE whose file id is . or .., which requests would send as a DELETE of the draft).
zenodo_publish_software_v7_49_0.py publishes, and only the draft this script's receipt names.

The token is read only from --token-file, in the lab's sectioned format: a [ZENODO] section with a
`zenodo_token: ...` line. It is sent only in an Authorization header, never in a URL, never
printed, never logged, never written to the receipt, and scrubbed from any error text. The file's
path is not printed or recorded either: a file that cannot be read (missing, locked, not UTF-8) is
a refusal that names the kind of error only.

USAGE
-----
    python zenodo_deposit_software_v7_49_0.py --dry-run
        Offline. Reads no token and makes no network call (a socket guard refuses any connection
        attempt). Validates the files and the metadata and prints every request the real run would
        send, with the token shown as <token>.

    python zenodo_deposit_software_v7_49_0.py --token-file PATH
        Creates the unpublished draft, fills it, verifies it, writes the receipt, stops.

    ... --resume-draft DRAFT_ID
        Take over an existing UNPUBLISHED draft in concept 19758618 (for example one left by an
        interrupted run): files that already match a local file by name, md5 and size are kept,
        every other file is deleted, the missing ones are uploaded and the metadata is overwritten.
        Only ever with an id the operator has looked at.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import difflib
import hashlib
import html
import json
import re
import subprocess
import sys
import tarfile
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"
METADATA_PATH = HERE / "metadata.json"
MARKERS_PATH = HERE / "excluded_markers.json"    # local only, never committed
PYPI_SOURCES_PATH = HERE / "pypi-sources-7.49.0.json"
RECEIPT_PATH = HERE / "zenodo-draft-receipt-software-v7.49.0.json"
PUBLISH_SCRIPT = "zenodo_publish_software_v7_49_0.py"
DEFAULT_REPO = "C:/Users/heyzo/clawd/styxx"

ZENODO_HOST = "zenodo.org"
ZENODO_API = f"https://{ZENODO_HOST}/api"
UA = "fathom-lab-styxx-research/1 (+https://github.com/fathom-lab/styxx)"

CONCEPT_RECID = "19758618"
CONCEPT_DOI = "10.5281/zenodo.19758618"
PREDECESSOR_RECID = 23200977            # styxx 7.48.1, release/zenodo-deposit-receipt-software-v7.48.1.json
PREDECESSOR_VERSION = "7.48.1"
PREDECESSOR_DOI = "10.5281/zenodo.23200977"
TAG = "v7.49.0"
TAG_COMMIT = "c6e00da02673ec6a2e84919f17f56a2e4435a8c9"
BRANCH_POINT = "d390ddcb"                # main when the release branched (CHANGELOG [7.49.0])
PR201_COMMIT = "506758b3"                # #201 (CONTRADICTED label): diffgate.py 4cded2e3… -> 09867056…
PATH2A_MERGE = "56e6c5f1"                # #187, PATH-2a merged to main
AMENDMENT_COMMIT = "e08a41b4"            # the operator's decision on G-P1, committed alone before #187
AMENDMENT = "papers/closed-model-frontier/AMENDMENT_path2_withholding_2026_10_06.md"
P2A_REF = "tests/_p2a_ref.py"            # pins the reader the PATH-2a tests rebuild (MAIN_PY_SHA)
# main's reader under the overlay since #201: 7.48.x's 9b620e00… with the demo's CONTRADICTED label
MAIN_READER_SHA = "68873068c0a665c2f5a2c40e2ca4d1e2cfccf28318f6f8d6fc03087d51fcdcc1"
LEADERBOARD = "styxx/_data/LEADERBOARD.md"   # what `styxx leaderboard` prints, in the wheel
PRIOR_ART_NOTE = "papers/NOTE_prior_art_credit_2026_09_29.md"
VERSION = "7.49.0"
RELEASE_DATE = "2026-10-07"
ADVISORY = "GHSA-3g8h-qcfm-25xw"
ADVISORY_URL = f"https://github.com/fathom-lab/styxx/security/advisories/{ADVISORY}"
PREV_ADVISORY = "GHSA-h5xv-4344-f62r"    # 7.48.1's, which 7.49.0 keeps
AFFECTED = ("7.47.0", "7.48.0", "7.48.1")
# 7.48.1 on Zenodo, as published and as read back, committed at the tag (PR #198)
BASE_METADATA = "release/zenodo-metadata-software-v7.48.1-as-published.json"
BASE_READBACK = "release/zenodo-record-software-v7.48.1-readback.json"
BASE_RECEIPT = "release/zenodo-deposit-receipt-software-v7.48.1.json"
PREFIX = "styxx-7.49.0/"

# (filename, bytes, sha256, md5). The wheel and sdist sha256 are PyPI's; the bundle is
# `git -c core.autocrlf=false archive --format=zip --prefix=styxx-7.49.0/ v7.49.0` (git 2.52.0.windows.1).
EXPECTED_FILES = [
    ("styxx-v7.49.0-source-bundle.zip", 276179193,
     "aef9945a67204ea89ece4e69cb8074a75abbeb7fa9385fba646ec7e55437485b",
     "b6f174e73d8b83182edf0f2b5e4745d0"),
    ("styxx-7.49.0-py3-none-any.whl", 8101703,
     "272f5cb384088888ede45044a4d51f24e81001278148f4555e3045440c22d01b",
     "3eafa8d101253cfe304eab200d03ae8e"),
    ("styxx-7.49.0.tar.gz", 8761703,
     "a45e74c20766d952830c6f7eda7d5d4f220c8626632e58b4def29d742f13e9e2",
     "60dbcf0d480220da3de29ed2810759cd"),
]
SDIST_GENERATED = {"PKG-INFO", "setup.cfg", "styxx.egg-info/PKG-INFO", "styxx.egg-info/SOURCES.txt",
                   "styxx.egg-info/dependency_links.txt", "styxx.egg-info/entry_points.txt",
                   "styxx.egg-info/requires.txt", "styxx.egg-info/top_level.txt"}

# Every Zenodo DOI the record names, with the committed file at the tag that backs it.
DOI_EVIDENCE = {
    "10.5281/zenodo.19758618": "release/zenodo-deposit-receipt-software-v7.48.1.json",
    "10.5281/zenodo.23200977": "release/zenodo-deposit-receipt-software-v7.48.1.json",
    "10.5281/zenodo.23042251": "release/zenodo-deposit-receipt-software-v7.48.0.json",
    "10.5281/zenodo.19758619": "release/zenodo-deposit-receipt-software-v6.2.0.json",
    "10.5281/zenodo.19746215": "release/zenodo-deposit-receipt-spec-v1.0.json",
    "10.5281/zenodo.19326174": "release/zenodo-deposit-receipt-spec-v1.0.json",
    "10.5281/zenodo.19777921": "CITATION.cff",
}

# exactly the related identifiers this version carries: (relation, identifier, scheme, resource_type)
SPEC_DOI = "10.5281/zenodo.19746215"
SERIES_DOI = "10.5281/zenodo.19326174"
EXPECTED_RELATED = [
    ("isSupplementTo", f"https://github.com/fathom-lab/styxx/releases/tag/{TAG}", "url", "software"),
    ("isDerivedFrom", f"https://github.com/fathom-lab/styxx/tree/{TAG_COMMIT}", "url", "software"),
    ("isSupplementTo", f"https://pypi.org/project/styxx/{VERSION}/", "url", "software"),
    ("isDocumentedBy", ADVISORY_URL, "url", None),
    ("isNewVersionOf", PREDECESSOR_DOI, "doi", "software"),
    ("isSupplementTo", SPEC_DOI, "doi", "publication-workingpaper"),
    ("isPartOf", SERIES_DOI, "doi", "publication-preprint"),
]
EXPECTED_SPEC_ROW = EXPECTED_RELATED[5]     # as 7.48.1 has it, its resource_type re-checked (DataCite)
EXPECTED_SERIES_ROW = EXPECTED_RELATED[6]   # as 7.48.1 has it, its resource_type re-checked (DataCite)

# What DataCite records for the Zenodo DOIs whose resource_type comes from outside the tag
# (fetch_datacite_7490.py writes the receipt), and the Zenodo resource_type registered as each:
# Text is told apart by resourceType, the others by resourceTypeGeneral.
DATACITE_RECEIPT_PATH = HERE / "datacite-relations-7.49.0.json"
DATACITE_TO_ZENODO = {"Preprint": "publication-preprint", "Software": "software"}
DATACITE_TEXT_TO_ZENODO = {"Working paper": "publication-workingpaper"}


def datacite_to_zenodo(rec: dict) -> str | None:
    if rec.get("resourceTypeGeneral") == "Text":
        return DATACITE_TEXT_TO_ZENODO.get(str(rec.get("resourceType") or ""))
    return DATACITE_TO_ZENODO.get(str(rec.get("resourceTypeGeneral") or ""))


# Keys Zenodo adds to a deposition's metadata on its own; any other key that metadata.json does
# not send (a community, contributor, reference, subject or grant added in the browser) stops a run.
# 7.48.0's and 7.48.1's draft and as-published metadata carry exactly the sent keys plus these.
ZENODO_ADDED_KEYS = {"doi", "prereserve_doi", "imprint_publisher"}

# Zenodo's documented relation types for related_identifiers (legacy deposit API).
ZENODO_RELATIONS = {
    "isCitedBy", "cites", "isSupplementTo", "isSupplementedBy", "isContinuedBy", "continues",
    "isDescribedBy", "describes", "hasMetadata", "isMetadataFor", "isNewVersionOf",
    "isPreviousVersionOf", "isPartOf", "hasPart", "isReferencedBy", "references",
    "isDocumentedBy", "documents", "isCompiledBy", "compiles", "isVariantFormOf",
    "isOriginalFormof", "isIdenticalTo", "isAlternateIdentifier", "isReviewedBy", "reviews",
    "isDerivedFrom", "isSourceOf", "requires", "isRequiredBy", "isObsoletedBy", "obsoletes",
}

# the lab's charter: words its public text never carries (checked, never printed). Stored
# reversed, so that this file, which goes to the repository, does not spell them either.
CHARTER_FORBIDDEN = [w[::-1] for w in (
    "tsrif", "levon", "yranoitulover", "gnikaerbdnuorg", "hguorhtkaerb", "foorp-repmat",
    "gniyfirev-fles", "rotceted noitanicullah",
)]

# Zenodo stores the licence "mit" as "mit-license" (7.48.0's and 7.48.1's receipts)
LICENSE_READBACK = {"mit": {"mit", "mit-license"}}


# ---------------------------------------------------------------------------
# output, secrets and the network guard
# ---------------------------------------------------------------------------

_SECRETS: list[str] = []


def redact(text: object) -> str:
    s = str(text)
    for t in _SECRETS:
        if t:
            s = s.replace(t, "[REDACTED]")
    s = re.sub(r"(access_token=)[^&\s'\"]+", r"\1[REDACTED]", s)
    s = re.sub(r"(Bearer\s+)(?!<token>)[^\s'\"]+", r"\1[REDACTED]", s)
    return s


def say(*parts: object) -> None:
    print(redact(" ".join(str(p) for p in parts)), flush=True)


def rule(title: str) -> None:
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


class Refuse(Exception):
    """A check failed; the message says which. Never carries the token."""


class HttpStatus(Refuse):
    """Zenodo answered, with a status the call did not expect. `.status` is the HTTP status."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class Transport(Refuse):
    """No answer was read: a timeout, a dropped connection or another transport error."""


class NetworkForbidden(RuntimeError):
    """Raised by the dry-run socket guard on any connection or name lookup."""


def forbid_network() -> None:
    """From here on this process cannot open a connection or resolve a name."""
    import socket

    def _blocked(*_a, **_k):
        raise NetworkForbidden("network access attempted during a dry run")

    socket.socket.connect = _blocked          # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked       # type: ignore[method-assign]
    socket.create_connection = _blocked       # type: ignore[assignment]
    socket.getaddrinfo = _blocked             # type: ignore[assignment]


def network_guard_holds() -> bool:
    import socket
    try:
        socket.create_connection(("127.0.0.1", 9), timeout=1)
    except NetworkForbidden:
        pass
    except OSError:
        return False
    else:
        return False
    try:
        socket.getaddrinfo(ZENODO_HOST, 443)
    except NetworkForbidden:
        return True
    except OSError:
        return False
    return False


TOKEN_SOURCE = "--token-file ([ZENODO] zenodo_token)"


def load_token(token_file: str | None) -> tuple[str, str]:
    """The token, from the lab's sectioned token file only. Neither the value nor the path is
    ever printed or recorded; errors say what is missing, not what the file holds."""
    if not token_file:
        raise Refuse("no token: pass --token-file PATH (a file with a [ZENODO] section and a "
                     "'zenodo_token: ...' line)")
    p = Path(token_file)
    try:
        if not p.is_file():
            raise Refuse("--token-file does not name a readable file (path not shown)")
        text = p.read_text(encoding="utf-8-sig")
    except Refuse:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as e:
        # an OSError's text names the file: refuse with the kind of error only, never the path
        raise Refuse(f"--token-file could not be read ({type(e).__name__}; path not shown)") from None
    lines = [ln.strip() for ln in text.splitlines()]
    in_zenodo = False
    for s in lines:
        if not s or s.startswith("#") or s.startswith(";"):
            continue
        if s.startswith("[") and s.endswith("]"):
            in_zenodo = s.upper() == "[ZENODO]"
            continue
        if in_zenodo and re.match(r"(?i)^zenodo_token\s*[:=]", s):
            value = re.split(r"[:=]", s, maxsplit=1)[1].strip().strip("'\"")
            if value and not re.search(r"\s", value):
                return value, TOKEN_SOURCE
    raise Refuse("no token found in --token-file: expected a [ZENODO] section with a "
                 "'zenodo_token: ...' line (file contents and path not shown)")


# ---------------------------------------------------------------------------
# offline checks
# ---------------------------------------------------------------------------

_HASH_CACHE: dict[tuple[str, int, int], tuple[int, str, str]] = {}


def hash_file(path: Path) -> tuple[int, str, str]:
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
            n += len(chunk)
    _HASH_CACHE[key] = (n, sha.hexdigest(), md5.hexdigest())
    return _HASH_CACHE[key]


def git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], check=check, capture_output=True, timeout=120)


def git_show(repo: str, spec: str) -> bytes:
    return git(repo, "show", spec).stdout


def git_text(repo: str, spec: str) -> str:
    return git_show(repo, spec).decode("utf-8")


def git_exists(repo: str, spec: str) -> bool:
    return git(repo, "cat-file", "-e", spec, check=False).returncode == 0


def repo_available(repo: str) -> bool:
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", TAG + "^{commit}"],
                             check=True, capture_output=True, text=True, timeout=30).stdout
        return out.strip() == TAG_COMMIT
    except (OSError, subprocess.SubprocessError):
        return False


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, ok: bool | None, name: str, detail: str = "") -> None:
        self.rows.append(("PASS" if ok else ("SKIP" if ok is None else "FAIL"), name, detail))

    @property
    def failed(self) -> list[tuple[str, str, str]]:
        return [r for r in self.rows if r[0] == "FAIL"]

    @property
    def skipped(self) -> list[tuple[str, str, str]]:
        return [r for r in self.rows if r[0] == "SKIP"]

    def show(self) -> None:
        for status, name, detail in self.rows:
            say(f"  [{status}] {name}" + (f": {detail}" if detail else ""))


def check_files(c: Checks) -> list[dict]:
    facts = []
    for name, size, sha, md5 in EXPECTED_FILES:
        p = BUNDLE / name
        if not p.is_file():
            c.add(False, f"file {name}", "missing")
            continue
        n, got_sha, got_md5 = hash_file(p)
        ok = (n, got_sha, got_md5) == (size, sha, md5)
        c.add(ok, f"file {name}", f"{n} bytes, sha256 {got_sha}, md5 {got_md5}"
              + ("" if ok else f" (pinned {size} / {sha} / {md5})"))
        facts.append({"name": name, "path": str(p), "bytes": n, "sha256": got_sha, "md5": got_md5})
    try:
        src = json.loads(PYPI_SOURCES_PATH.read_text(encoding="utf-8"))["files"]
        pins = {(v["filename"], v["bytes"], v["sha256"]) for v in src.values()}
        want = {(n, b, s) for n, b, s, _m in EXPECTED_FILES[1:]}
        c.add(pins == want, "wheel and sdist are the files PyPI lists for 7.49.0",
              f"{PYPI_SOURCES_PATH.name}: {sorted(p[0] for p in pins)}")
    except (OSError, KeyError, ValueError) as e:
        c.add(False, "wheel and sdist are the files PyPI lists for 7.49.0", f"{type(e).__name__}")
    return facts


def git_blob_id(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def tag_tree(repo: str) -> dict[str, tuple[str, str]]:
    out = git(repo, "ls-tree", "-r", "-z", TAG).stdout
    tree = {}
    for rec in out.split(b"\0"):
        if rec:
            meta, path = rec.split(b"\t", 1)
            _mode, kind, oid = meta.decode().split()
            tree[path.decode("utf-8")] = (kind, oid)
    return tree


def check_bundle_tree(c: Checks, repo: str | None, tree: dict | None) -> None:
    if not repo or tree is None:
        c.add(None, "source bundle is the v7.49.0 tree", "repository not available here")
        return
    problems, seen = [], set()
    with zipfile.ZipFile(BUNDLE / EXPECTED_FILES[0][0]) as z:
        if z.comment.decode("ascii", "replace").strip() != TAG_COMMIT:
            problems.append("zip comment is not the tagged commit id")
        for info in z.infolist():
            if info.filename.endswith("/"):
                continue
            if not info.filename.startswith(PREFIX):
                problems.append(f"entry outside prefix: {info.filename}")
                continue
            rel = info.filename[len(PREFIX):]
            if rel not in tree or tree[rel][0] != "blob":
                problems.append(f"not a blob in the tree: {rel}")
                continue
            if rel in seen:
                problems.append(f"duplicate entry: {rel}")
                continue
            seen.add(rel)
            if git_blob_id(z.read(info)) != tree[rel][1]:
                problems.append(f"bytes differ from the blob: {rel}")
    problems += [f"missing: {m}" for m in sorted(set(tree) - seen)]
    c.add(not problems, "source bundle is the v7.49.0 tree",
          f"{len(seen)} of {len(tree)} blobs byte-identical" if not problems
          else f"{len(problems)} problem(s), e.g. {problems[:3]}")


def check_packages_against_tree(c: Checks, tree: dict | None) -> None:
    if tree is None:
        c.add(None, "wheel and sdist contents are the tag's blobs", "repository not available here")
        return
    whl = BUNDLE / EXPECTED_FILES[1][0]
    sdist = BUNDLE / EXPECTED_FILES[2][0]
    same, bad, version_ok = 0, [], False
    with zipfile.ZipFile(whl) as z:
        for info in z.infolist():
            n = info.filename
            if n.endswith("/"):
                continue
            if ".dist-info/" in n:
                if n.endswith(".dist-info/METADATA"):
                    version_ok = bool(re.search(rb"(?m)^Version: 7\.49\.0\r?$", z.read(info)))
                continue
            if n not in tree or git_blob_id(z.read(info)) != tree[n][1]:
                bad.append(n)
            else:
                same += 1
    c.add(not bad and same > 0 and version_ok,
          "every wheel file outside .dist-info equals the tag's blob; METADATA says 7.49.0",
          f"{same} files" if not bad else f"{len(bad)} differ, e.g. {bad[:3]}")
    same_s, bad_s, generated = 0, [], set()
    with tarfile.open(sdist) as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            top, _, rel = m.name.partition("/")
            if top != "styxx-7.49.0":
                bad_s.append(m.name)
                continue
            if rel in SDIST_GENERATED:
                generated.add(rel)
                continue
            data = t.extractfile(m).read()  # type: ignore[union-attr]
            if rel not in tree or git_blob_id(data) != tree[rel][1]:
                bad_s.append(rel)
            else:
                same_s += 1
    c.add(not bad_s and same_s > 0 and generated == SDIST_GENERATED,
          "every sdist file but the build's generated metadata equals the tag's blob",
          f"{same_s} files, plus {len(generated)} generated" if not bad_s
          else f"{len(bad_s)} differ, e.g. {bad_s[:3]}")


class _TagBalance(HTMLParser):
    VOID = {"br", "hr", "img", "meta", "link", "input", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.text: list[str] = []
        self.code: list[str] = []
        self._in_code = 0

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)
        if tag == "code":
            self._in_code += 1
            self.code.append("")

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if tag == "code" and self._in_code:
            self._in_code -= 1
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.stack[-1] if self.stack else 'nothing'}")
        else:
            self.stack.pop()

    def handle_data(self, data):
        self.text.append(data)
        if self._in_code:
            self.code[-1] += data


def parse_html(fragment: str) -> _TagBalance:
    p = _TagBalance()
    p.feed(fragment)
    p.close()
    return p


def word_hits(text: str, word: str) -> int:
    return len(re.findall(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text.lower()))


NUM_RX = r"(?<![0-9A-Za-z_.\-/])\d+(?:[.,]\d+)*(?![0-9A-Za-z_])"


def has_token(token: str, corpus: str) -> bool:
    return re.search(r"(?<![0-9A-Za-z_.\-/])" + re.escape(token) + r"(?![0-9A-Za-z_])",
                     corpus) is not None


def load_markers(c: Checks) -> list[str] | None:
    if not MARKERS_PATH.is_file():
        c.add(None, "unreleased-content markers", f"{MARKERS_PATH.name} not here")
        return None
    try:
        markers = json.loads(MARKERS_PATH.read_text(encoding="utf-8"))["markers"]
        assert isinstance(markers, list) and all(isinstance(m, str) and m for m in markers)
        return markers
    except (OSError, ValueError, KeyError, AssertionError):
        c.add(False, "unreleased-content markers", f"{MARKERS_PATH.name} unreadable")
        return None


def markers_fingerprint() -> dict:
    try:
        raw = MARKERS_PATH.read_bytes()
        return {"file": MARKERS_PATH.name, "sha256": hashlib.sha256(raw).hexdigest(),
                "count": len(json.loads(raw.decode("utf-8"))["markers"])}
    except (OSError, ValueError, KeyError):
        return {"file": MARKERS_PATH.name, "present": False}


def changelog_sections(repo: str) -> tuple[str, str, str]:
    """CHANGELOG.md at the tag, its [7.49.0] section and its [7.48.1] section."""
    changelog = git_text(repo, f"{TAG}:CHANGELOG.md")
    start = changelog.index("## [7.49.0]")
    mid = changelog.index("\n## [7.48.1]", start)
    end = changelog.index("\n## [7.48.0]", mid)
    return changelog, changelog[start:mid], changelog[mid + 1:end]


def md_plain(text: str) -> str:
    """CHANGELOG markdown as the description's plain text reads it: no backticks, no bold."""
    return re.sub(r"\s+", " ", text.replace("`", "").replace("**", ""))


_P2A_MARK = "# === PATH-2a abstain-only overlay: "


def reconstruct_reader(text: str) -> str | None:
    """`main`'s reader inside the tag's styxx/diffgate.py, by tests/_p2a_ref.py's rule: cut the
    PATH-2a block, turn the two `g = _gate(` back into `return _gate(`, drop the two hook lines.
    None when the file does not have exactly that shape."""
    begin, end = _P2A_MARK + "BEGIN ===", _P2A_MARK + "END ==="
    if text.count(begin) != 1 or text.count(end) != 1:
        return None
    out, n = re.subn(re.escape(begin) + r"\n.*?" + re.escape(end) + r"\n\n\n", "", text, count=1, flags=re.S)
    if n != 1 or out.count("    g = _gate(summary_text,") != 2:
        return None
    lines = out.replace("    g = _gate(summary_text,", "    return _gate(summary_text,").split("\n")
    if sum(1 for x in lines if x.endswith("  # PATH-2a")) != 2:
        return None
    return "\n".join(x for x in lines if not x.endswith("  # PATH-2a"))


def demo_only_diff(old: str, new: str) -> bool:
    """True when every line that differs between two versions of diffgate.py sits inside `_demo`
    in both, and the new one prints CONTRADICTED where the old printed LIE."""
    a, b = old.split("\n"), new.split("\n")

    def owner(lines: list[str], i: int) -> str | None:
        for j in range(min(i, len(lines) - 1), -1, -1):
            m = re.match(r"(?:def|class) (\w+)", lines[j])
            if m:
                return m.group(1)
        return None

    ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    removed = "\n".join(x for _t, i1, i2, _j1, _j2 in ops for x in a[i1:i2])
    added = "\n".join(x for _t, _i1, _i2, j1, j2 in ops for x in b[j1:j2])
    return (bool(ops) and all(owner(a, i1) == "_demo" and owner(b, j1) == "_demo"
                              for _t, i1, _i2, j1, _j2 in ops)
            and '"CONTRADICTED": "LIE"' in removed and '"CONTRADICTED": "CONTRADICTED"' in added)


def check_metadata(c: Checks, facts: list[dict], repo: str | None,
                   tree: dict | None) -> dict | None:
    try:
        raw = METADATA_PATH.read_bytes()
    except OSError as e:
        c.add(False, "metadata.json readable", str(e))
        return None
    c.add(not raw.startswith(b"\xef\xbb\xbf"), "metadata.json has no BOM")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        c.add(False, "metadata.json parses", str(e))
        return None
    c.add(set(doc) == {"metadata"}, "top level is exactly {'metadata': ...}", str(sorted(doc)))
    m = doc.get("metadata", {})

    c.add(m.get("upload_type") == "software", "upload_type software", repr(m.get("upload_type")))
    c.add(m.get("version") == VERSION, f"version {VERSION}", repr(m.get("version")))
    try:
        _dt.date.fromisoformat(m.get("publication_date", ""))
        date_ok = m.get("publication_date") == RELEASE_DATE
    except (TypeError, ValueError):
        date_ok = False
    c.add(date_ok, f"publication_date {RELEASE_DATE}", repr(m.get("publication_date")))
    c.add(m.get("access_right") == "open", "access_right open", repr(m.get("access_right")))
    c.add("doi" not in m and "prereserve_doi" not in m,
          "no DOI set by hand (Zenodo mints the version DOI)")
    c.add(m.get("license") == "mit", "license sent as 'mit' (Zenodo stores it as mit-license)",
          repr(m.get("license")))

    title = m.get("title", "")
    c.add(title.startswith(f"styxx v{VERSION} — "), f"title starts 'styxx v{VERSION} — '", title[:40])

    kws = m.get("keywords", [])
    c.add(bool(kws) and all(isinstance(k, str) and k for k in kws), "keywords present",
          f"{len(kws)} keywords")

    desc = m.get("description", "")
    notes = m.get("notes", "")
    p = parse_html(desc)
    c.add(not p.errors and not p.stack, "description HTML tags balance",
          "; ".join(p.errors[:3]) + (f" unclosed {p.stack}" if p.stack else ""))
    plain = re.sub(r"\s+", " ", "".join(p.text))
    files_at = desc.find("<p><strong>Files in this record</strong></p>")
    c.add(files_at > 0 and desc.count("<p><strong>Files in this record</strong></p>") == 1,
          "description ends with one 'Files in this record' section")
    files_html = desc[files_at:] if files_at > 0 else ""
    for f in facts:
        ok = (f["sha256"] in files_html and f["md5"] in files_html
              and f">{f['name']}</code>: " in files_html and f"{f['bytes']} bytes" in files_html)
        c.add(ok, f"description lists bytes, sha256 and md5 of {f['name']}")
    c.add(CONCEPT_DOI in desc and CONCEPT_DOI in notes, "description and notes name the concept DOI")
    c.add(ADVISORY_URL in desc and ADVISORY in notes and ADVISORY in title + plain
          and f'<a href="{ADVISORY_URL}">{ADVISORY}</a>' in desc,
          "description links the advisory; notes name it")
    c.add(f"pip install -U styxx=={VERSION}" in plain, "description says how to upgrade")
    c.add(all(v in plain for v in AFFECTED), "description names all three affected releases",
          ", ".join(AFFECTED))
    c.add(TAG_COMMIT in desc and TAG_COMMIT in notes, "description and notes name the tagged commit")
    c.add(PREDECESSOR_DOI in desc and PREDECESSOR_DOI in notes,
          "description and notes name 7.48.1's version DOI")
    try:
        up = sorted(v["upload_time_iso_8601"][:16] + "Z" for v in
                    json.loads(PYPI_SOURCES_PATH.read_text(encoding="utf-8"))["files"].values())
    except (OSError, KeyError, ValueError, TypeError):
        up = []
    c.add(bool(up) and f"PyPI records the upload at {up[0]}, UTC" in notes,
          f"notes give PyPI's upload time as {PYPI_SOURCES_PATH.name} records it", str(up[:1]))
    c.add(not desc.endswith(("\n", " ")), "description has no trailing whitespace (Zenodo strips it)")

    markers = load_markers(c)
    public_text = " ".join([title, plain, notes, " ".join(kws)])
    hits = {w: word_hits(public_text, w) for w in CHARTER_FORBIDDEN}
    hits = {w: n for w, n in hits.items() if n}
    c.add(not hits, "charter: no forbidden word in title/description/notes/keywords",
          f"{sum(hits.values())} hit(s)" if hits else "")
    if markers is not None:
        ex = [i for i, w in enumerate(markers, 1) if w.lower() in public_text.lower()
              or w.lower() in desc.lower()]
        c.add(not ex, "no unreleased-content marker in the public text",
              f"marker(s) #{ex} of {MARKERS_PATH.name} matched" if ex else f"{len(markers)} markers")

    creators = m.get("creators", [])
    c.add(bool(creators) and all(re.match(r"^[^,]+, [^,]+$", cr.get("name", "")) for cr in creators),
          "creators present, 'Family, Given'", str([cr.get("name") for cr in creators]))

    rels = m.get("related_identifiers", [])
    bad_rel = [r for r in rels if r.get("relation") not in ZENODO_RELATIONS
               or not r.get("identifier") or r.get("scheme") not in {"url", "doi"}]
    c.add(not bad_rel, "related_identifiers use Zenodo relation types", str(bad_rel or ""))
    got_rel = [(r.get("relation"), r.get("identifier"), r.get("scheme"), r.get("resource_type"))
               for r in rels]
    c.add(got_rel == EXPECTED_RELATED,
          "related identifiers: release, tree at c6e00da0, PyPI 7.49.0, the advisory, 7.48.1 "
          "(isNewVersionOf), the spec, the research series",
          "" if got_rel == EXPECTED_RELATED else f"got {got_rel}")
    check_datacite_types(c, rels)
    dois_named = set(re.findall(r"10\.5281/zenodo\.\d+", desc + " " + notes + " "
                                + " ".join(r.get("identifier", "") for r in rels)))
    unbacked = sorted(d for d in dois_named if d not in DOI_EVIDENCE)
    c.add(not unbacked, "every Zenodo DOI named has a committed file that backs it",
          str(unbacked or f"{len(dois_named)} DOIs"))

    if not repo or tree is None:
        c.add(None, "claims of the description against the tag", "repository not available here")
        return doc
    check_against_tag(c, m, desc, plain, notes, files_at, dois_named, repo, tree)
    return doc


def check_datacite_types(c: Checks, rels: list) -> None:
    """The resource_type of the spec, series and isNewVersionOf rows is what DataCite records."""
    name = ("resource_type of 10.5281/zenodo.19746215, 10.5281/zenodo.19326174 and "
            f"10.5281/zenodo.23200977 = what DataCite records ({DATACITE_RECEIPT_PATH.name})")
    try:
        recs = json.loads(DATACITE_RECEIPT_PATH.read_text(encoding="utf-8"))["records"]
        spec, series, prev = recs[SPEC_DOI], recs[SERIES_DOI], recs[PREDECESSOR_DOI]
    except (OSError, ValueError, KeyError, TypeError) as e:
        c.add(False, name, f"{DATACITE_RECEIPT_PATH.name} unreadable ({type(e).__name__})")
        return
    sent = {r.get("identifier"): r.get("resource_type") for r in rels if r.get("scheme") == "doi"}
    want = {d: datacite_to_zenodo(rec) for d, rec in
            ((SPEC_DOI, spec), (SERIES_DOI, series), (PREDECESSOR_DOI, prev))}
    ok = (all(r.get("http_status") == 200 for r in (spec, series, prev))
          and all(want.values()) and all(sent.get(d) == t for d, t in want.items())
          and prev.get("version") == PREDECESSOR_VERSION
          and "IsVersionOf" in (prev.get("relations_to_concept") or []))
    c.add(ok, name, f"DataCite: 19746215 {spec.get('resourceTypeGeneral')}/{spec.get('resourceType')}, "
          f"19326174 {series.get('resourceTypeGeneral')}, 23200977 {prev.get('resourceTypeGeneral')} "
          f"{prev.get('version')}; sent {sent.get(SPEC_DOI)}, {sent.get(SERIES_DOI)}, "
          f"{sent.get(PREDECESSOR_DOI)}")


# Each statement of the description's condensed sections, held to the [7.49.0] section: the
# description's plain text must hold every fragment of the description list, and the section
# (backticks and bold dropped) every fragment of the CHANGELOG list (the description list again
# when that is None).
SOURCED: list[tuple[str, list[str], list[str] | None]] = [
    ("what the release carries",
     ["The release carries a security repair to how python -m styxx.capsule verify checks a v0.1 "
      "OATH capsule, the diff gate's PATH-2a overlay, the GitHub Action's new default, and what "
      "else merged on main after 7.48.1."],
     ["a security repair to how python -m styxx.capsule verify checks a v0.1 OATH capsule, the "
      "diff gate's PATH-2a overlay, the GitHub Action's new default, and what else merged on main "
      "after 7.48.1."]),
    ("layer 2 re-runs certify",
     ["Layer 2 of a v0.1 OATH capsule, python -m styxx.capsule verify, re-runs certify on the "
      "embedded document and receipts."],
     ["Layer 2 of a v0.1 capsule, python -m styxx.capsule verify, re-runs certify on the embedded "
      "document and receipts."]),
    ("what was wrong",
     ["Before this release it checked the document and receipt hashes and compared only the "
      "verdict (from 7.48.0 by class, its , N uncovered suffix stripped), the counts and the "
      "status of each ledger row the certificate carries, and it read the payload by text without "
      "checking the page around it, so a capsule whose certificate was edited to say what its "
      "bytes do not verified."], None),
    ("the five forgeries",
     ["from the v7.47.0, v7.48.0 and v7.48.1 tag trees, each over capsules it minted itself: a "
      "certificate edited to report 0 uncovered (the D1 forgery), a deleted ledger row, a row's "
      "receipt_ref pointed at another receipt, a row's epistemics flag flipped, and a genuine "
      "payload hidden in an HTML comment ahead of one whose document reads otherwise all verify "
      "under 7.48.0 and 7.48.1.",
      "7.47.0's certify writes no uncovered band, so D1 has nothing to edit in its own capsules; "
      "7.47.0 verifies the other four, and verifies D1 applied to a capsule a later styxx minted, "
      "whose honest form it fails.", "7.49.0 fails every one of them."], None),
    ("affected releases", ["Affected releases. 7.47.0, 7.48.0 and 7.48.1."],
     ["Affected: 7.47.0, 7.48.0 and 7.48.1."]),
    ("upgrading is the fix",
     ["For anyone who verifies capsules made by someone else, upgrading is the fix."], None),
    ("the whole-certificate comparison",
     ["compares every field certify_doc writes, type for type: the verdict string, the counts, "
      "the uncovered band, the epistemics summary, receipts_sha256, the receipt-binding digests, "
      "and the ledger, ungrounded and abstained in both directions, in order, every field of "
      "every row."],
     ["It now compares every field certify_doc writes, type for type",
      "the verdict string, the counts, the uncovered band, the epistemics summary, "
      "receipts_sha256, the receipt-binding digests, and the ledger, ungrounded and abstained in "
      "both directions, in order, every field of every row."]),
    ("the page is held to the payload",
     ["The page around the payload must be the page a styxx renders for exactly that payload: the "
      "current page, or the one every styxx rendered before this change, kept verbatim in "
      "styxx/_capsule_page_v01_legacy.py.", "An edited page now fails."],
     ["the page a styxx renders for exactly that payload: the current page, or the one every "
      "styxx rendered before this change, kept verbatim in styxx/_capsule_page_v01_legacy.py",
      "An edited page now fails."]),
    ("NOT CHECKED and the advisories",
     ["A field an older certificate lacks is printed NOT CHECKED by name, and every advisory is "
      "printed."],
     ["prints by name each field an older certificate lacks (NOT CHECKED)", "and every advisory",
      "It now prints every advisory, NOT CHECKED field and stated field"]),
    ("renamed copy, minted page, committed capsules",
     ["A capsule of a renamed copy now fails",
      "A newly minted page shows a verdict only after its hashes match, and its install line names "
      "styxx>=7.49.0, not the minter's version.",
      "All ten committed v0.1 capsules still verify, each with an advisory that it states a styxx "
      "below 7.49.0."], None),
    ("7.48.1's bare-name repair kept",
     ["7.48.1's repair for GHSA-h5xv-4344-f62r, is kept as released."],
     ["7.48.1 shipped that repair (GHSA-h5xv-4344-f62r), and this change keeps its bare-name rule "
      "and its messages as released."]),
    ("what to do until you can upgrade",
     ["an exit 0 from 7.47.0, 7.48.0 or 7.48.1 says only that the payload it read, which need not "
      "be the one a browser draws, carries bytes matching its certificate's hashes, and that the "
      "verdict (by class in 7.48.0 and 7.48.1), the counts and the status of each row the "
      "certificate carries reproduce.",
      "To rely on a document's numbers, certify the document and receipts you mean to rely on "
      "yourself (python -m styxx.certify DOC RECEIPTS...) and read that certificate, not the "
      "capsule's or its page."], None),
    ("what 7.49.0 does not close",
     ["7.49.0 still accepts the page every styxx rendered before it, since honest capsules carry "
      "it.", "the operator's decision",
      "Around that page, the D1 forgery posed as older than the uncovered band (the band fields "
      "and the receipt binding deleted, another issuer's hash) exits 0, and 7.49.0 prints the "
      "installed verifier's verdict beside the embedded one, six NOT CHECKED lines and three "
      "advisories, one naming the number nothing checked and one saying every styxx below 7.49.0 "
      "passes forgeries 7.49.0 fails.",
      "Around the page 7.49.0 mints, the same pose fails.",
      "The older page tells its reader to pip install the payload's verifier.pip, which the "
      "minter chooses, and 7.47.0, 7.48.0 and 7.48.1 each pass this pose around either page.",
      "The v0.2 and sworn pages' install lines are not repaired.",
      "classes a review found still end verify in a traceback, so #196 stays open",
      "Nothing in a capsule is signed."], None),
    ("PATH-2a",
     ["gate_diff_text, gate_diff, the commit-msg and agent hooks",
      "where the #97, #121 or #101 mechanism can have made a VERIFIED or CONTRADICTED verdict "
      "wrong, the claim is now UNCHECKABLE, and its reason names the verdict withheld, the defect "
      "and the reading without the overlay.",
      "The overlay never adds an accusation and never makes a verdict VERIFIED; the three defects "
      "are not repaired.",
      "On the lab's committed corpora it withholds 80 of 2,231 decided claims (3.6%",
      "--strict fails on each new abstention, as on any UNCHECKABLE.",
      # the clause after it (whether PATH-2a stands in for G-P1 is the operator's decision) is
      # stale at the tag; the stale lines say so (the AMENDMENT of 2026-10-06)
      "PREREG_path2's G-P1 expects VERIFIED on the reproductions, so G-P1 is not met.",
      "The path accusation stays withheld."],
     ["gate_diff_text, gate_diff, the commit-msg and agent hooks",
      "where the #97, #121 or #101 mechanism can have made a VERIFIED or CONTRADICTED verdict "
      "wrong, the claim is now UNCHECKABLE, and its reason names the verdict withheld, the defect "
      "and the reading without the overlay.",
      "The overlay never adds an accusation and never makes a verdict VERIFIED; the three defects "
      "are not repaired.",
      "On the lab's committed corpora it withholds 80 of 2,231 decided claims (3.6%",
      "--strict fails on each new abstention, as on any UNCHECKABLE.",
      "PREREG_path2's G-P1 expects VERIFIED on the reproductions, so G-P1 is not met",
      "the path accusation stays withheld"]),
    ("CONTRADICTED, not LIE",
     ["A contradicted claim is printed as [CONTRADICTED] with its reason, not as [LIE], in the "
      "demo, the hooks and the bookmarklet, and the demo's closing line says how many claims the "
      "diff contradicts.", "Records, verdicts and exit codes are unchanged."], None),
    ("the Action's default",
     ["soft-fail defaults to \"true\": every verdict goes in the job summary, each contradicted "
      "claim is named in an annotation, and the gate's verdicts never fail the job.",
      "Only an explicit soft-fail: \"false\" blocks.",
      "This changes behaviour for anyone who relied on the default, which was \"false\"."],
     ["soft-fail defaults to \"true\": every verdict goes in the job summary, each contradicted "
      "claim is named in an annotation, and the gate's verdicts never fail the job.",
      "Only an explicit soft-fail: \"false\" blocks.",
      "This changes behaviour for anyone who relied on the default.",
      "action.yml's soft-fail input defaulted to \"false\""]),
    ("why the Action reports",
     ["kind of accusation the instrument still makes has been measured clearing the 0.95 precision "
      "floor the lab set for accusing", "Path claims are reported UNCHECKABLE",
      "EXTERNAL-1 (2026-08-31) measured precision 0.23 against a preregistered floor of 0.95, a "
      "blind three-seat panel upholding 23 of 100 sampled accusations",
      "V14 (2026-09-01) measured 0.16 on held-out path accusations",
      "only_touches still accuses, at precision 0.25 (PATH-1, 2026-09-17: 2 of 8 accusations "
      "correct).", "No receipt re-measures it with the PATH-2a overlay on.",
      "tests_added, symbol_added and files_changed_count still accuse, and no committed RESULT "
      "states a precision for them."],
     ["kind of accusation the instrument still makes has been measured clearing the 0.95 precision "
      "floor the lab set for accusing", "Path claims no longer accuse.",
      "file_created, file_deleted and file_touched are reported UNCHECKABLE",
      "EXTERNAL-1 (2026-08-31): a blind three-seat panel upheld 23 of 100 sampled accusations, "
      "precision 0.23, against a preregistered floor of 0.95",
      "V14 (2026-09-01), the later measurement",
      "upheld 16 of 100 held-out path accusations, precision 0.16",
      "only_touches still accuses, at precision 0.25.",
      "PATH-1 (2026-09-17) took it from 0.18 to 0.25: 2 of 8 accusations correct",
      "No receipt re-measures it with the PATH-2a overlay on.",
      "tests_added, symbol_added and files_changed_count still accuse, and no committed RESULT "
      "states a precision for them."]),
    ("the Action imports the styxx beside it",
     ["The Action imports the styxx package beside its script, at the ref the workflow names, not "
      "the one pip installs, so @main and a workflow pinned to this release's tag both run "
      "PATH-2a.", "so the lab's own check keeps blocking."], None),
    ("verify, charon and create beyond the repair",
     ["escapes control characters in what it prints",
      "a sweep of 600 malformed single-field mutations now ends in a problem, not a traceback",
      "in an UNRESOLVED or not-reproduced line",
      "capsule create refuses some certificates older styxx issued (re-certify, then mint)."],
     None),
    ("priority sentences withdrawn",
     ["Priority sentences that no survey priced are withdrawn from docstrings and printed strings "
      "in the package and from the docs, and the prior art is credited with dates"], None),
    ("in the repository, not the wheel",
     ["SECURITY.md sends reports through GitHub private vulnerability reporting instead of an email "
      "address the lab cannot confirm receives mail", "7.48.1's Zenodo record (10.5281/zenodo.23200977)",
      "web/gate/README.md's run-book pin, held to py_side.py by a test"],
     ["SECURITY.md sends reports through GitHub private vulnerability reporting instead of an email "
      "address the lab cannot confirm receives mail", "7.48.1's Zenodo record (10.5281/zenodo.23200977)",
      "web/gate/README.md's run-book pin, held to py_side.py by a test",
      "scripts/zenodo_deposit_software_v7_48_1.py makes the draft"]),
    ("what did not change",
     ["styxx/certify.py, styxx/sworn.py and styxx/corpus_audit.py are the files 7.48.1 shipped, so "
      "certify, sworn and corpus_audit read as they did.",
      "Outside the PATH-2a block and the printed label the diff gate reads as 7.48.1 did"], None),
    ("cutting: version, floor, citation",
     ["styxx/_version.py is 7.49.0 (3e8722cb), the floor the capsule page names for layer 2 "
      "(_LAYER2_FLOOR and const FLOOR in styxx/capsule.py), and CITATION.cff gives version 7.49.0 "
      "and date-released 2026-10-07 (25d35543)."], None),
    ("cutting: conformance/sworn",
     ["conformance/sworn/ was regenerated for the version stamp (3e8722cb)",
      "15 vectors took new ids, 0 moved, and no expected outcome changed",
      "3620 vectors, 20 families and 3981 blobs",
      "That CI passes on the regenerated set was not observed at the cut."],
     ["conformance/sworn/ was regenerated for the version stamp (3e8722cb",
      "15 vectors took new ids, 0 moved, and no expected outcome changed",
      "Still 3620 vectors, 20 families and 3981 blobs.",
      "that CI passes on the regenerated set (not observed here)"]),
    ("cutting: README and web/gate/README.md",
     ["README.md, which becomes the PyPI page, was audited at this cut against the tree it ships "
      "with", "89 tag-pinned links now name v7.49.0 instead of v7.48.0 (0a4c479e), on the same lines",
      "no other line changed", "PATH-2a section now says 7.49.0 ships it (7a418e2b)"],
     ["README.md, which becomes the PyPI page, was audited at this cut against the tree it ships "
      "with", "89 tag-pinned links now name v7.49.0 instead of v7.48.0 (0a4c479e), on the same lines",
      "no other line changed", "said PATH-2a was not released; it now says 7.49.0 ships it (7a418e2b)"]),
]


# What the condensed sections say beyond the SOURCED fragments, piece by piece (the text between
# fragments, trimmed of " .,;:()"): headings, labels and joins, and five statements a fact check
# below holds (the heading against the title and the advisory, the upgrade command, diffgate.yml,
# the prior-art sentence, 7.48.1's receipts and scripts, web/gate/README.md's opening paragraph).
# Any other text in those sections fails "every statement of the condensed sections is sourced".
GLUE = {
    "Security: capsule verify passed v0.1 capsules whose certificate was edited to say what their "
    "bytes do not (GHSA-3g8h-qcfm-25xw",
    "What was wrong", "Run at the cut", "The repair. verify now",
    "The bare-name rule for a capsule's embedded names",
    "What to do. Upgrade: pip install -U styxx==7.49.0. Until you can",
    "What 7.49.0 does not close. By", "Two",
    "The diff gate: PATH-2a withholds the verdicts its three known defects can make wrong",
    "For users of python -m styxx.diffgate", "The GitHub Action reports by default", "Why: no",
    "and", "This repository's own .github/workflows/diffgate.yml sets soft-fail: \"false\"",
    "Also in this release", "For users who verify capsules, beyond the security repair: verify",
    "in styxx.charon",
    "papers/NOTE_prior_art_credit_2026_09_29.md). With them go the priority claims and charter "
    "words that 7.48.1's record listed as stale, in the docstrings of styxx/__init__.py, "
    "forecast.py, intercept.py, critique.py and hallucination.py, in "
    "attack/universal_suffixes_v0.json, adapters/guardrails.py and admissibility.py, and in "
    "README.md",
    "In the repository, not the wheel",
    "with its receipts in release/ and the two scripts that made it in scripts/; and",
    "What did not change", "Cutting this release", "still", "its", "web/gate/README.md's",
    "its opening paragraph still says otherwise (below",
}

# The stale lines, exactly; each is held by a fact check in check_against_tag ("stale: ...").
STALE_LINES = [
    # the cut made only "not released" stale; #201 (506758b3, before the branch point) had already
    # made the Action entry's 4cded2e3… stale
    "Two entries of the ## [7.49.0] section, kept whole from [Unreleased], carry wording that does "
    "not hold at the tag. The PATH-2a entry says it is not released, which the cut made stale. The "
    "Action entry gives the whole of styxx/diffgate.py as sha256 4cded2e3…, which #201 made stale "
    "before the cut. At the tag that file is 09867056… (LF), as web/gate/differential/py_side.py "
    "pins it. Both entries, and the description in action.yml, also give 9b620e00… as the reader "
    "the overlay runs over; since #201 that reader is 68873068… (9b620e00… with the demo's "
    "CONTRADICTED label), as tests/_p2a_ref.py pins it. Where the PATH-2a entry says pip install "
    "styxx (7.48.0), 7.48.1 was already on PyPI when PATH-2a merged to main (by the upload time in "
    "7.48.1's record under release/).",
    # the G-P1 line also covers web/gate/README.md's PATH-2a section and the entry's "Open for the
    # operator" paragraph, which list as open what the amendment decides (decisions 1, 3 and 4)
    "The PATH-2a entry calls whether PATH-2a stands in for G-P1 the operator's decision; the "
    "operator took it on 2026-10-06 (papers/closed-model-frontier/AMENDMENT_path2_withholding_"
    "2026_10_06.md): G-P1 is not met, withholding is accepted as what lands now and not as the "
    "repair, and #97, #101 and #121 stay open. web/gate/README.md's PATH-2a section still calls G-P1 the operator's decision, "
    "and says the operator confirms the switch's removal at merge. The entry's \"Open for the "
    "operator\" paragraph still lists G-P1, the switch's removal and bar A's boundary, which the "
    "amendment decides.",
    "web/gate/README.md's opening paragraph calls the PATH-2a block not yet released, and its drift "
    "section, web/gate/differential/py_side.py and the headers of web/gate/diffgate.js and "
    "web/gate/bookmarklet_src.js still speak of 7.48.0 as not yet shipped. The bookmarklet's panel "
    "text (web/gate/bookmarklet_ui.js) still calls its port the 7.48.0 one, as web/gate/README.md "
    "says.",
    "The header comment of .github/workflows/diffgate.yml says the instrument is the released "
    "package; the Action imports the styxx package beside its script, at the ref the workflow names "
    "(above).",
    "sworn/README.md and sworn/action.yml still give styxx==7.48.0 as the way to install the "
    "released styxx.sworn; GHSA-h5xv-4344-f62r and GHSA-3g8h-qcfm-25xw both name 7.48.0 as "
    "affected.",
    # the "priority sentences withdrawn" bullet holds for the sentences the 2026-09-29 note records;
    # the leaderboard the CLI prints and two comments still carry the claim critique.py withdrew
    "styxx/_data/LEADERBOARD.md, which the styxx leaderboard command prints (the root "
    "LEADERBOARD.md is the same file), and two comments in styxx/__init__.py still carry the "
    "ordinal priority claim for Baseline-019's pass of the gauntlet's v3 bars that "
    "styxx/critique.py's docstring withdrew. The bullet above on withdrawn priority sentences holds "
    "for the sentences papers/NOTE_prior_art_credit_2026_09_29.md records, not for every string the "
    "package prints.",
    "README.md labels 10.5281/zenodo.19326174, the concept record of the Fathom research-paper "
    "series, as the always-latest DOI in its badge row and link table (open defects D3 and D4 in "
    "zenodo/MANIFEST.json). Its citation row names the software concept DOI, "
    "10.5281/zenodo.19758618.",
    "CITATION.cff names the software concept DOI, but GitHub's \"Cite this repository\" prompt "
    "renders its preferred-citation, the position paper 10.5281/zenodo.19777921, which predates "
    "that paper's 2026-06-21 scope erratum (D2 in zenodo/MANIFEST.json). Whether the prompt should "
    "cite the software is left to the maintainer.",
    "The committed EXTERNAL-1 packet, papers/closed-model-frontier/external1_packet.json, still "
    "carries the id leak. Since #125 its builder, external1_packet.py, numbers items by their "
    "shuffled position by default, and its build --as-published mode writes the pre-repair "
    "numbering on purpose, leak included, because the committed packet, key and digest are "
    "receipts; that mode has been shown only on a synthetic ledger and shelf, not run on the real "
    "ones. Whether any adjudicator used the leak cannot be re-tested.",
    "zenodo/README.md still describes the 7.48.0 deposit flow only.",
    "zenodo/MANIFEST.json names 7.48.1 (10.5281/zenodo.23200977) as the latest version of this "
    "concept; this record supersedes that line.",
]

def check_against_tag(c: Checks, m: dict, desc: str, plain: str, notes: str, files_at: int,
                      dois_named: set, repo: str, tree: dict) -> None:
    """Every statement of the description, held to the tag's own bytes."""
    changelog, section, section_7481 = changelog_sections(repo)
    norm_section = re.sub(r"\s+", " ", section)
    norm_7481 = re.sub(r"\s+", " ", section_7481)
    md_section, md_7481 = md_plain(section), md_plain(section_7481)
    head = re.search(r"^## \[7\.49\.0\] — 2026-10-07 — (.+)$", changelog, re.M)
    c.add(bool(head) and m.get("title") == f"styxx v{VERSION} — {head.group(1).strip().replace('`', '')}",
          "title subtitle = the [7.49.0] headline at the tag (markdown backticks dropped)")
    tag_msg = git(repo, "cat-file", "-p", TAG).stdout.decode("utf-8", "replace")
    c.add(ADVISORY in norm_section and ADVISORY in tag_msg,
          "the advisory id is the one the tag's CHANGELOG and tag message name")
    c.add(f"pip install -U styxx=={VERSION}" in norm_section, "the upgrade command is the CHANGELOG's")

    # -- what 7.49.0 keeps from 7.48.1 as published ---------------------------------------------
    base = json.loads(git_text(repo, f"{TAG}:{BASE_METADATA}"))
    readback = json.loads(git_text(repo, f"{TAG}:{BASE_READBACK}"))
    rm = readback.get("metadata") or {}
    kept = ("creators", "keywords", "language", "access_right", "related_identifiers", "description")
    c.add(base.get("version") == PREDECESSOR_VERSION and base.get("doi") == PREDECESSOR_DOI
          and readback.get("id") == PREDECESSOR_RECID and str(readback.get("conceptrecid")) == CONCEPT_RECID
          and all(rm.get(k) == base.get(k) for k in kept)
          and (rm.get("license") or {}).get("id") == base.get("license"),
          "7.48.1 as published (the deposition) and as read back (the public record) agree on every "
          "field kept here")
    c.add(m.get("creators") == base.get("creators"), "creators = 7.48.1 as published")
    c.add(m.get("keywords") == base.get("keywords"), "keywords = 7.48.1 as published")
    c.add(m.get("language") == base.get("language") and m.get("access_right") == base.get("access_right")
          and m.get("upload_type") == base.get("upload_type"),
          "language, access_right, upload_type = 7.48.1 as published")
    c.add(m.get("license") in {k for k, v in LICENSE_READBACK.items() if base.get("license") in v},
          "license = 7.48.1 as published", f"{m.get('license')!r} -> stored as {base.get('license')!r}")
    opening = re.match(r"(<p>This version continues the styxx software record .*?</p>)",
                       base.get("description", ""), re.S)
    desc_open = re.match(r"(<p>This version continues the styxx software record .*?</p>)", desc, re.S)
    c.add(bool(opening) and desc.startswith(opening.group(1) + "\n\n")
          and bool(desc_open) and desc_open.group(1) == opening.group(1),
          "opening paragraph (spec and research series) = 7.48.1 as published")
    base_rel = [(x.get("relation"), x.get("identifier"), x.get("scheme"), x.get("resource_type"))
                for x in base.get("related_identifiers", []) if x.get("scheme") == "doi"]
    c.add(len(base_rel) == 3 and base_rel[1] == EXPECTED_SPEC_ROW and base_rel[2] == EXPECTED_SERIES_ROW
          and base_rel[0][:3] == ("isNewVersionOf", "10.5281/zenodo.23042251", "doi"),
          "spec and series relations = 7.48.1 as published, resource types included", str(base_rel))

    cff = git_text(repo, f"{TAG}:CITATION.cff")
    authors_block = cff.split("authors:", 1)[1].split("\ntype:", 1)[0] if "authors:" in cff else ""
    fam = re.findall(r"family-names:\s*\"?([^\"\n]+)", authors_block)
    giv = re.findall(r"given-names:\s*\"?([^\"\n]+)", authors_block)
    aff = re.findall(r"affiliation:\s*\"?([^\"\n]+)", authors_block)
    orc = re.findall(r"orcid:\s*\"?([^\"\n]+)", authors_block)
    creators = m.get("creators", [])
    c.add([cr.get("name") for cr in creators] == [f"{f.strip()}, {g.strip()}" for f, g in zip(fam, giv)]
          and [cr.get("affiliation") for cr in creators] == [a.strip() for a in aff]
          and [cr["orcid"] for cr in creators if cr.get("orcid")] == [o.strip().rsplit("/", 1)[-1] for o in orc],
          "creators = CITATION.cff authors at the tag (no ORCID recorded there, none sent)")
    c.add(re.search(r'(?m)^version: "7\.49\.0"$', cff) is not None
          and re.search(r'(?m)^date-released: "2026-10-07"$', cff) is not None,
          "CITATION.cff: version 7.49.0, date-released 2026-10-07")
    c.add(git_text(repo, f"{TAG}:LICENSE").startswith("MIT License"), "LICENSE at the tag is MIT")
    pyproject = git_text(repo, f"{TAG}:pyproject.toml")
    try:
        import tomllib
        py_kws = list(tomllib.loads(pyproject)["project"]["keywords"])
    except ModuleNotFoundError:
        blk = re.search(r"^keywords\s*=\s*\[(.*?)\]", pyproject, re.S | re.M)
        py_kws = re.findall(r'"([^"]+)"', blk.group(1)) if blk else []
    c.add(m.get("keywords") == py_kws, "keywords = pyproject.toml at the tag", f"{len(py_kws)} there")
    c.add('__version__ = "7.49.0"' in git_text(repo, f"{TAG}:styxx/_version.py"),
          "styxx/_version.py at the tag is 7.49.0")

    # -- DOIs ----------------------------------------------------------------------------------
    unbacked = [d for d in sorted(dois_named) if d in DOI_EVIDENCE
                and d not in git_text(repo, f"{TAG}:{DOI_EVIDENCE[d]}")]
    c.add(not unbacked, "each DOI named appears in the committed file that backs it", str(unbacked or ""))
    prev = json.loads(git_text(repo, f"{TAG}:{BASE_RECEIPT}"))
    c.add(prev.get("software_doi") == PREDECESSOR_DOI and prev.get("version") == PREDECESSOR_VERSION
          and str(prev.get("concept_recid")) == CONCEPT_RECID and prev.get("deposit_id") == PREDECESSOR_RECID
          and prev.get("status") == "published_and_verified"
          and prev.get("concept_latest_after_publish") == PREDECESSOR_RECID,
          "7.48.1 is 10.5281/zenodo.23200977 in concept 19758618, the latest after its publish "
          "(its committed receipt)")
    c.add(re.search(r"^## \[7\.48\.1\] — 2026-10-06 — ", changelog, re.M) is not None,
          "7.48.1 was released 2026-10-06 (its CHANGELOG heading)")

    # -- the regions of the body ------------------------------------------------------------------
    open_end = desc.find("</p>") + 4
    sec_at = desc.find("<p><strong>Security: ")
    stale_at = desc.find("<p><strong>Known stale or wrong lines")
    notin_at = desc.find("<p><strong>Not in this deposit</strong></p>")
    regions_ok = 0 < open_end < sec_at < stale_at < notin_at < files_at
    c.add(regions_ok, "the body's regions are in order: release, Security .. Cutting, stale lines, "
          "Not in this deposit, Files")
    if not regions_ok:
        return

    def region_plain(a: int, b: int) -> tuple[str, list[str]]:
        rp = parse_html(desc[a:b])
        return re.sub(r"\s+", " ", "".join(rp.text)), rp.code

    intro_plain, intro_code = region_plain(open_end, sec_at)
    cond_plain, cond_code = region_plain(sec_at, stale_at)
    stale_plain, stale_code = region_plain(stale_at, notin_at)
    notin_plain, notin_code = region_plain(notin_at, files_at)
    body_plain = " ".join([intro_plain, cond_plain, stale_plain, notin_plain])

    manifest = json.loads(git_text(repo, f"{TAG}:zenodo/MANIFEST.json"))
    defects = {d["id"]: d for d in manifest.get("citation_surface_defects", [])}
    concept_entry = next((d for d in manifest.get("deposits", []) if d.get("doi") == CONCEPT_DOI), {})
    wide = " ".join([norm_section, norm_7481,
                     json.dumps(manifest.get("citation_surface_defects"), ensure_ascii=False),
                     json.dumps(concept_entry, ensure_ascii=False), cff])

    # numbers and dates: the release paragraphs and the condensed sections from [7.49.0] alone;
    # the stale lines from [7.49.0], [7.48.1], the manifest's defects and 19758618 entry, CITATION.cff
    def nums_dates_missing(text: str, corpus: str) -> list[str]:
        nums = sorted(set(re.findall(NUM_RX, text)))
        dates = sorted(set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)))
        return [n for n in nums if not has_token(n, corpus)] + [d for d in dates if d not in corpus]

    for label, text, corpus, src in [
        ("the release paragraphs", intro_plain + " " + notin_plain, norm_section, "CHANGELOG [7.49.0]"),
        ("the Security, diff gate, Action, Also and Cutting sections", cond_plain, norm_section,
         "CHANGELOG [7.49.0] alone"),
        ("the stale lines", stale_plain, wide, "CHANGELOG [7.49.0] or [7.48.1], the manifest's "
         "defect list or its 19758618 entry, or CITATION.cff"),
    ]:
        miss = nums_dates_missing(text, corpus)
        n_nums = len(set(re.findall(NUM_RX, text)))
        c.add(not miss, f"every number and date of {label} is in {src} at the tag",
              f"{n_nums} numbers" if not miss else f"missing {miss}")

    # -- the sentence that says where each section comes from, held to what it says ------------
    headings = [re.sub(r"<[^>]+>", "", h).strip()
                for h in re.findall(r"<p><strong>((?:(?!</p>).)*?)</strong></p>", desc[open_end:], re.S)]
    want_heads = ["Security: ", "The diff gate: ", "The GitHub Action reports by default",
                  "Also in this release", "Cutting this release",
                  "Known stale or wrong lines in the deposited tree", "Not in this deposit",
                  "Files in this record"]
    norm_desc = re.sub(r"\s+", " ", desc)
    c.add(len(headings) == len(want_heads) and all(h.startswith(w) for h, w in zip(headings, want_heads))
          and "The next five sections, from the security repair to cutting this release, are "
              "condensed from the <code>## [7.49.0]</code> section" in norm_desc
          and "The last two sections describe this deposit" in norm_desc,
          "the body's sections are the five the CHANGELOG sentence covers, the stale lines, and the "
          "two that describe the deposit, in that order", str(headings))

    # -- each condensed statement, held to the [7.49.0] section ----------------------------------
    for label, in_desc, in_log in SOURCED:
        miss_d = [f for f in in_desc if f not in plain]
        miss_c = [f for f in (in_log if in_log is not None else in_desc) if f not in md_section]
        c.add(not miss_d and not miss_c, f"stated as CHANGELOG [7.49.0] states it: {label}",
              "" if not (miss_d or miss_c) else
              f"not in the description: {[x[:60] for x in miss_d]}; not in the CHANGELOG: "
              f"{[x[:60] for x in miss_c]}")

    # -- nothing in the condensed sections beyond what the checks hold ---------------------------
    frags = sorted({f for _l, ins, _c in SOURCED for f in ins}, key=len, reverse=True)
    unsourced = []
    for _tag, inner in re.findall(r"<(li|p)>(.*?)</\1>", desc[sec_at:stale_at], re.S):
        t = re.sub(r"\s+", " ", "".join(parse_html(inner).text)).strip()
        for f in frags:
            t = t.replace(f, "|")
        unsourced += [x for x in (piece.strip(" .,;:()") for piece in t.split("|"))
                      if x and x not in GLUE]
    c.add(not unsourced, "every statement of the condensed sections is a sourced statement above, "
          "or a heading, label, join or fact-checked statement GLUE names",
          "" if not unsourced else f"not sourced: {[x[:70] for x in unsourced[:5]]}")
    stale_items = [re.sub(r"\s+", " ", "".join(parse_html(inner).text)).strip()
                   for inner in re.findall(r"<li>(.*?)</li>", desc[stale_at:notin_at], re.S)]
    c.add(stale_items == STALE_LINES and len(STALE_LINES) == 11,
          "the stale lines are exactly the eleven the stale checks below hold",
          f"{len(stale_items)} lines" if stale_items == STALE_LINES else
          f"differ: {[x[:70] for x in stale_items if x not in STALE_LINES]}")
    # the statements GLUE carries, held to the description as GLUE states them
    c.add("For users who verify capsules, beyond the security repair:" in md_section,
          "'For users who verify capsules, beyond the security repair' is the CHANGELOG's heading")

    # -- <code> spans, commit ids, hash prefixes and #numbers ---------------------------------------
    paths = set(tree)
    dirs = {q.rsplit("/", 1)[0] + "/" for q in paths if "/" in q}
    dirs |= {d for q in list(dirs) for d in ["/".join(q.split("/")[:i]) + "/"
                                              for i in range(1, q.count("/"))]}

    def code_ok(span: str, corpora: tuple[str, ...]) -> bool:
        s = re.sub(r"\s+", " ", html.unescape(span)).strip()
        if not s:
            return False
        if s == TAG or s in paths or s in dirs or any(q.endswith("/" + s) for q in paths):
            return True
        if re.fullmatch(r"[0-9a-f]{8,40}", s):
            return git(repo, "merge-base", "--is-ancestor", s, TAG, check=False).returncode == 0
        return any(s in x for x in corpora)

    # the release paragraphs name the [7.48.1] section the stale lines draw on, as the stale lines do
    bad_code = ([s for s in notin_code + cond_code if not code_ok(s, (norm_section,))]
                + [s for s in intro_code + stale_code if not code_ok(s, (norm_section, norm_7481))])
    n_code = len(intro_code + cond_code + stale_code + notin_code)
    c.add(not bad_code, "every <code> span of the body is a path in the tree, an ancestor commit, "
          "or text of CHANGELOG [7.49.0] (the release paragraphs and the stale lines: or [7.48.1])",
          f"{n_code} spans" if not bad_code else f"{bad_code}")
    hex_ids = re.findall(r"(?<![0-9a-z./])([0-9a-f]{8}(?:[0-9a-f]{32})?)(?![0-9a-z])(…)?", desc + " " + notes)
    commits = sorted({x for x, ell in hex_ids if not ell})
    prefixes = sorted({x for x, ell in hex_ids if ell})
    not_anc = [x for x in commits
               if git(repo, "merge-base", "--is-ancestor", x, TAG, check=False).returncode != 0]
    c.add(not not_anc and TAG_COMMIT in commits,
          "every commit id named is the tag's commit or an ancestor of it", str(not_anc or f"{len(commits)} ids"))
    c.add(all(f"`{x}…`" in norm_section for x in prefixes),
          "every hash prefix named (an id followed by …) is one CHANGELOG [7.49.0] prints",
          str(prefixes))
    prs_c = sorted(set(re.findall(r"#\d+", intro_plain + " " + cond_plain + " " + notin_plain)))
    prs_s = sorted(set(re.findall(r"#\d+", stale_plain)))
    c.add(all(x in norm_section for x in prs_c) and all(x in norm_section or x in norm_7481 for x in prs_s),
          "every #number of the body is in CHANGELOG [7.49.0] (the stale lines: or [7.48.1])",
          str([x for x in prs_c if x not in norm_section]
              + [x for x in prs_s if x not in norm_section and x not in norm_7481] or prs_c + prs_s))

    # -- the facts the description states about the tree ---------------------------------------
    # affected releases and the repair
    caps = {v: git_text(repo, f"{v}:styxx/capsule.py") if git_exists(repo, f"{v}:styxx/capsule.py") else ""
            for v in ("v7.47.0", "v7.48.0", "v7.48.1", TAG)}
    c.add(all("def verify_capsule" in caps[v] and "_LAYER2_FLOOR" not in caps[v]
              for v in ("v7.47.0", "v7.48.0", "v7.48.1"))
          and not git_exists(repo, "v7.48.1:styxx/_capsule_page_v01_legacy.py"),
          "affected releases: verify_capsule is in 7.47.0, 7.48.0 and 7.48.1, none with the repair "
          "(no layer-2 floor, no legacy-page module)")
    c.add(git_exists(repo, f"{TAG}:styxx/_capsule_page_v01_legacy.py")
          and re.search(r'(?m)^_LAYER2_FLOOR = "7\.49\.0"$', caps[TAG]) is not None
          and "const FLOOR = '7.49.0';" in caps[TAG],
          "at the tag: styxx/_capsule_page_v01_legacy.py, and _LAYER2_FLOOR and const FLOOR are 7.49.0")
    caps_html = [q for q in paths if q.startswith("papers/") and q.endswith(".capsule.html")]
    caps_v01 = [q for q in caps_html if b"styxx-oath/capsule/v0.1" in git_show(repo, f"{TAG}:{q}")]
    c.add(len(caps_v01) == 10, "ten v0.1 capsules are committed under papers/",
          f"{len(caps_v01)} v0.1 among {len(caps_html)} capsule files")
    c.add("unsafe_embedded_name" in git_text(repo, f"{TAG}:styxx/charon.py")
          and "tests/test_capsule_bare_names.py" in paths
          and git(repo, "merge-base", "--is-ancestor", "76e9dcc5", TAG, check=False).returncode == 0,
          "7.48.1's bare-name repair (76e9dcc5) is in the tag: charon's unsafe_embedded_name and "
          "its test")
    unchanged = git(repo, "diff", "--quiet", "v7.48.1", TAG, "--", "styxx/certify.py", "styxx/sworn.py",
                    "styxx/corpus_audit.py", check=False).returncode == 0
    c.add(unchanged, "styxx/certify.py, sworn.py and corpus_audit.py are 7.48.1's files")
    # the Action
    act = git_text(repo, f"{TAG}:action.yml")
    sf = re.search(r"(?ms)^  soft-fail:\n(.*?)(?=^  \S)", act)
    c.add(bool(sf) and re.search(r'(?m)^    default: "true"\s*$', sf.group(1)) is not None,
          "action.yml at the tag: soft-fail defaults to \"true\"")
    dg = git_text(repo, f"{TAG}:.github/workflows/diffgate.yml")
    step = re.search(r"(?ms)^\s+uses: \./\n(.*?)(?=^\s*-\s|\Z)", dg)
    c.add(bool(step) and re.search(r'(?m)^\s+soft-fail: "false"\s*$', step.group(1)) is not None,
          "the repository's diffgate workflow sets soft-fail: \"false\" on its uses: ./ step")
    c.add(re.search(r"(?m)^from styxx\.diffgate import", git_text(repo, f"{TAG}:diffgate_action.py"))
          is not None, "diffgate_action.py imports styxx.diffgate (the package beside it)")
    # Also
    errata_files = ["styxx/__init__.py", "styxx/forecast.py", "styxx/intercept.py", "styxx/critique.py",
                    "styxx/hallucination.py", "styxx/attack/universal_suffixes_v0.json",
                    "styxx/adapters/guardrails.py", "styxx/admissibility.py"]
    changed = [f for f in errata_files
               if git(repo, "diff", "--quiet", "v7.48.1", TAG, "--", f, check=False).returncode == 1]
    base_stale = re.sub(r"\s+", " ", base.get("description", ""))
    readme = git_text(repo, f"{TAG}:README.md")
    readme_7481 = git_text(repo, "v7.48.1:README.md")
    # what 7.48.1's record listed were priority claims AND charter words: each of the eight files
    # carries fewer charter-word hits at the tag than at v7.48.1 (the hits left are ordinary
    # technical uses), counted by word, never printed
    fewer = [f for f in errata_files
             if sum(word_hits(git_text(repo, f"{TAG}:{f}"), w) for w in CHARTER_FORBIDDEN)
             < sum(word_hits(git_text(repo, f"v7.48.1:{f}"), w) for w in CHARTER_FORBIDDEN)]
    c.add(changed == errata_files and fewer == errata_files
          and all(f"`{f}`" in section for f in errata_files)
          and "the control almost nobody runs" in readme_7481
          and "the control almost nobody runs" not in readme
          and "the control almost nobody runs" in md_section
          and "drop charter words about the lab's own work" in md_section
          and "still carry the priority claims and charter words that the errata to [7.48.0] list, "
              "in the docstrings of" in base_stale
          and all(f.split("styxx/", 1)[1] in base_stale for f in errata_files)
          and "and on README line 422" in base_stale,
          "the priority claims and charter words 7.48.1's record listed: each of its eight files "
          "changed since 7.48.1, carries fewer charter-word hits and is named in the prior-art "
          "entry, and README no longer carries its sentence",
          "" if changed == fewer == errata_files else
          f"unchanged: {sorted(set(errata_files) - set(changed))}; "
          f"no fewer charter-word hits: {sorted(set(errata_files) - set(fewer))}")
    c.add("papers/NOTE_prior_art_credit_2026_09_29.md" in paths, "the prior-art record is in the tree")
    sec_md = git_text(repo, f"{TAG}:SECURITY.md")
    c.add("https://github.com/fathom-lab/styxx/security/advisories/new" in sec_md
          and re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z.]+", sec_md) is None,
          "SECURITY.md sends reports to GitHub private reporting and names no email address")
    z7481 = ["release/zenodo-draft-receipt-software-v7.48.1.json",
             "release/zenodo-deposit-receipt-software-v7.48.1.json",
             "release/zenodo-metadata-software-v7.48.1.json",
             "release/zenodo-metadata-software-v7.48.1-as-published.json",
             "release/zenodo-record-software-v7.48.1-readback.json",
             "release/NOTE_zenodo_software_v7_48_1_provenance.md",
             "scripts/zenodo_deposit_software_v7_48_1.py", "scripts/zenodo_publish_software_v7_48_1.py"]
    c.add(all(f in paths for f in z7481), "7.48.1's Zenodo receipts and its two scripts are in the tree")
    c.add("test_port_is_current" in " ".join(paths) and "PINNED" in git_text(repo, f"{TAG}:tests/test_port_is_current.py"),
          "tests/test_port_is_current.py holds the run book's pin to py_side.py's PINNED")
    # Cutting
    def touched(commit: str) -> set[str]:
        return set(git(repo, "show", "--name-only", "--format=", commit).stdout.decode().split())

    t1, t2, t3, t4 = touched("3e8722cb"), touched("25d35543"), touched("0a4c479e"), touched("7a418e2b")
    c.add("styxx/_version.py" in t1 and any(x.startswith("conformance/sworn/") for x in t1)
          and t2 == {"CITATION.cff"} and t3 == {"README.md"} and t4 == {"web/gate/README.md"},
          "3e8722cb touches _version.py and conformance/sworn/, 25d35543 CITATION.cff, 0a4c479e "
          "README.md, 7a418e2b web/gate/README.md")
    idx = json.loads(git_text(repo, f"{TAG}:conformance/sworn/index.json"))
    c.add(idx.get("vector_count") == 3620 and idx.get("family_count") == 20
          and (idx.get("blobs") or {}).get("count") == 3981
          and (idx.get("provenance") or {}).get("styxx_version") == VERSION,
          "conformance/sworn at the tag: 3620 vectors, 20 families, 3981 blobs, stamped 7.49.0")
    pinned = re.findall(r"(?:github\.com/fathom-lab/styxx/(?:blob|tree|raw)|"
                        r"raw\.githubusercontent\.com/fathom-lab/styxx)/(v\d+\.\d+\.\d+)/", readme)
    c.add(len(pinned) == 89 and set(pinned) == {"v7.49.0"},
          "README's tag-pinned links: 89, all at the v7.49.0 tag", f"{len(pinned)}, tags {sorted(set(pinned))}")
    d3 = git(repo, "diff", "-U0", "0a4c479e^", "0a4c479e", "--", "README.md").stdout.decode("utf-8")
    minus = [ln[1:] for ln in d3.splitlines() if ln.startswith("-") and not ln.startswith("---")]
    plus = [ln[1:] for ln in d3.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    hunks = re.findall(r"(?m)^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", d3)
    same_lines = all(a == c_ and (b or "1") == (d or "1") for a, b, c_, d in hunks)
    c.add(bool(minus) and len(minus) == len(plus) and same_lines
          and all(x.replace("/v7.48.0/", "/v7.49.0/") == y and x != y for x, y in zip(minus, plus)),
          "0a4c479e changes README.md's tag-pinned links from v7.48.0 to v7.49.0 on the same lines "
          "and nothing else", f"{len(minus)} lines")
    wg = git_text(repo, f"{TAG}:web/gate/README.md")
    wg_open = wg.split("\n\n", 2)[:2]
    c.add("**Released in 7.49.0.**" in wg and "not yet released; see *PATH-2a* below" in " ".join(wg_open),
          "web/gate/README.md: its PATH-2a section says Released in 7.49.0; its opening paragraph "
          "still says not yet released")

    # -- the stale lines -----------------------------------------------------------------------------
    dg_blob = git_show(repo, f"{TAG}:styxx/diffgate.py")
    dg_sha = hashlib.sha256(dg_blob).hexdigest()
    py_side = git_text(repo, f"{TAG}:web/gate/differential/py_side.py")
    def sha_at(rev: str, path: str) -> str:
        return hashlib.sha256(git_show(repo, f"{rev}:{path}")).hexdigest()

    def is_ancestor(a: str, b: str) -> bool:
        return git(repo, "merge-base", "--is-ancestor", a, b, check=False).returncode == 0

    def subject(commit: str) -> str:
        return git(repo, "log", "-1", "--format=%s", commit).stdout.decode("utf-8").strip()

    # "kept whole from [Unreleased]": both sentences stand under [Unreleased] at the branch point,
    # the parent of the cut's earliest commit
    pre = git_text(repo, f"{BRANCH_POINT}:CHANGELOG.md")
    pre_unrel = re.sub(r"\s+", " ", pre[pre.index("## [Unreleased]"):pre.index("\n## [7.48.1]")]) \
        if "## [Unreleased]" in pre and "\n## [7.48.1]" in pre else ""
    branch_ok = git(repo, "rev-parse", "3e8722cb^").stdout.decode().strip().startswith(BRANCH_POINT)
    c.add(branch_ok and "It is not released: `pip install styxx` (7.48.0) and this branch disagree" in pre_unrel
          and "(the whole file is sha256 `4cded2e3…`)" in pre_unrel
          and "The entries headed `[Unreleased]` until this cut are kept whole below" in norm_section,
          f"both sentences stood under [Unreleased] at {BRANCH_POINT}, the main the cut (3e8722cb) "
          "branched from, and [7.49.0] keeps those entries whole")
    c.add("It is not released: `pip install styxx` (7.48.0) and this branch disagree" in norm_section
          and "(the whole file is sha256 `4cded2e3…`)" in norm_section
          and "The whole file is `09867056…` (LF)" in norm_section
          and b"\r" not in dg_blob and dg_sha.startswith("09867056") and not dg_sha.startswith("4cded2e3")
          and f'PINNED = "{dg_sha}"' in py_side,
          "stale: the PATH-2a entry says it is not released, the Action entry gives diffgate.py as "
          "4cded2e3…; at the tag it is 09867056… (LF), py_side.py's PINNED",
          dg_sha[:16])
    # which change made each one stale. "Not released": PATH-2a is in no release before this one
    # (main at the branch point is 7.48.1, whose diffgate.py is 9b620e00… and has no PATH-2a), so
    # the cut made it stale. "4cded2e3…": #201 had already moved the file to the tag's bytes, at a
    # commit that is an ancestor of the branch point, so it was stale before the cut.
    c.add('__version__ = "7.48.1"' in git_text(repo, f"{BRANCH_POINT}:styxx/_version.py")
          and sha_at("v7.48.1", "styxx/diffgate.py").startswith("9b620e00")
          and b"PATH-2a" not in git_show(repo, "v7.48.1:styxx/diffgate.py") and b"PATH-2a" in dg_blob
          and "`9b620e00…`, the file in the `v7.48.0` and `v7.48.1` tags" in norm_section,
          f"stale: PATH-2a is in no release before 7.49.0 (main at {BRANCH_POINT} is 7.48.1, whose "
          "diffgate.py is 9b620e00… without PATH-2a): the cut made 'not released' stale")
    c.add(is_ancestor(PR201_COMMIT, BRANCH_POINT) and subject(PR201_COMMIT).endswith("(#201)")
          and sha_at(f"{PR201_COMMIT}^", "styxx/diffgate.py").startswith("4cded2e3")
          and sha_at(PR201_COMMIT, "styxx/diffgate.py") == dg_sha
          and sha_at(BRANCH_POINT, "styxx/diffgate.py") == dg_sha
          and "(`4cded2e3…` when this was measured, `09867056…` after #201)" in norm_section,
          f"stale: #201 ({PR201_COMMIT}, an ancestor of {BRANCH_POINT}) took diffgate.py from "
          f"4cded2e3… to 09867056…, and at {BRANCH_POINT} it is already the tag's file: the "
          "Action entry's 4cded2e3… was stale before the cut")

    def entry(heading: str) -> str:
        """One ### entry of [7.49.0], whitespace collapsed ('' when the heading is not there)."""
        at = section.find("\n### " + heading)
        if at < 0:
            return ""
        nxt = section.find("\n### ", at + 1)
        return re.sub(r"\s+", " ", section[at:nxt if nxt > 0 else len(section)])

    p2a_entry = entry("PATH-2a: the diff gate withholds a verdict")
    action_entry = entry("the GitHub Action reports by default")

    # the reader the overlay runs over: the Action entry, the PATH-2a entry and action.yml's
    # description give 9b620e00… (7.48.x's file); since #201 it is 68873068…, that file with the
    # demo's CONTRADICTED label, as tests/_p2a_ref.py pins it and as the tag's own file rebuilds
    def main_py_sha(text: str) -> str | None:
        mm = re.search(r'(?m)^MAIN_PY_SHA = "([0-9a-f]{64})"', text)
        return mm.group(1) if mm else None

    reader_7481 = git_show(repo, "v7.48.1:styxx/diffgate.py")
    sha_7481 = hashlib.sha256(reader_7481).hexdigest()
    rebuilt = reconstruct_reader(dg_blob.decode("utf-8"))
    act_n = re.sub(r"\s+", " ", act)
    reader_facts = {
        "the Action entry: @main runs 9b620e00… under the overlay": (
            "Its reader is `styxx/diffgate.py` at sha256 `9b620e00…`, the file in the `v7.48.0` and "
            "`v7.48.1` tags. At `@main` that reader runs under the PATH-2a overlay" in action_entry),
        "the PATH-2a entry: the test gets main's reader back as 9b620e00…": (
            "gets `main`'s two files back byte for byte (`9b620e00…`, `06688702…`), then uses them as "
            "the reference for every differential" in p2a_entry),
        "action.yml's description: @main runs 9b620e00… under the overlay": (
            "styxx/diffgate.py at sha256 9b620e00…, the reader in 7.48.0 and 7.48.1; @main runs that "
            "reader under the PATH-2a overlay" in act_n),
        "9b620e00… is 7.48.1's diffgate.py": sha_7481.startswith("9b620e00"),
        f"{P2A_REF} pinned 9b620e00… before #201 and 68873068… from #201 to the tag": (
            main_py_sha(git_text(repo, f"{PR201_COMMIT}^:{P2A_REF}")) == sha_7481
            and main_py_sha(git_text(repo, f"{PR201_COMMIT}:{P2A_REF}")) == MAIN_READER_SHA
            and main_py_sha(git_text(repo, f"{TAG}:{P2A_REF}")) == MAIN_READER_SHA
            and MAIN_READER_SHA.startswith("68873068")),
        "the tag's diffgate.py without the PATH-2a block is 68873068…": (
            rebuilt is not None and hashlib.sha256(rebuilt.encode("utf-8")).hexdigest() == MAIN_READER_SHA),
        "68873068… is 9b620e00… with the demo's CONTRADICTED label, nothing else": (
            rebuilt is not None and demo_only_diff(reader_7481.decode("utf-8"), rebuilt)),
        "[7.49.0] and py_side.py name 68873068… as that reader": (
            "`tests/_p2a_ref.py` pins it at `68873068…` (that file with the demo's label)" in norm_section
            and f"main's {MAIN_READER_SHA[:8]} reader (9b620e00 + the demo's CONTRADICTED label)" in py_side),
    }
    c.add(all(reader_facts.values()),
          "stale: both [7.49.0] entries and action.yml give 9b620e00… as the reader the overlay runs "
          f"over; since #201 it is 68873068… (9b620e00… with the demo's label): {P2A_REF}'s "
          "MAIN_PY_SHA, and the tag's diffgate.py rebuilt without the block",
          "" if all(reader_facts.values()) else f"not as stated: {[k for k, v in reader_facts.items() if not v]}")
    # "(7.48.0)": what the PATH-2a entry says pip install styxx gets; 7.48.1's own record, committed
    # at the tag, gives its PyPI upload before PATH-2a (#187) merged
    up = re.search(r"PyPI records the upload at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2})Z, UTC", base.get("notes") or "")
    merged_at = git(repo, "log", "-1", "--format=%cI", PATH2A_MERGE).stdout.decode().strip()
    try:
        up_t = _dt.datetime.fromisoformat(up.group(1) + ":00+00:00") if up else None
        merged_t = _dt.datetime.fromisoformat(merged_at)
    except ValueError:
        up_t = merged_t = None
    not_rel = "It is not released: `pip install styxx` (7.48.0)"
    pypi_facts = {
        "the PATH-2a entry gives pip install styxx as 7.48.0": not_rel in p2a_entry,
        "it did when PATH-2a merged (#187)": (
            subject(PATH2A_MERGE).endswith("(#187)")
            and not_rel in re.sub(r"\s+", " ", git_text(repo, f"{PATH2A_MERGE}:CHANGELOG.md"))),
        "7.48.1's record gives its PyPI upload before that merge": (
            base.get("version") == PREDECESSOR_VERSION and up_t is not None and merged_t is not None
            and up_t < merged_t),
    }
    c.add(all(pypi_facts.values()),
          "stale: the PATH-2a entry's pip install styxx (7.48.0): 7.48.1 was on PyPI before PATH-2a "
          f"merged ({PATH2A_MERGE}, {merged_at}; 7.48.1's record: "
          f"{up.group(1) + 'Z' if up else 'no upload time'})",
          "" if all(pypi_facts.values()) else f"not as stated: {[k for k, v in pypi_facts.items() if not v]}")
    # the G-P1 line: the PATH-2a entry's open decision, taken in the AMENDMENT, which was committed
    # alone, after the sentence was written, and before #187 merged
    amend = re.sub(r"\s+", " ", git_text(repo, f"{TAG}:{AMENDMENT}")) if AMENDMENT in paths else ""
    gp1 = ("PREREG_path2's G-P1 expects VERIFIED on the reproductions, so G-P1 is not met; whether "
           "PATH-2a stands in for it is the operator's decision.")
    before_amend = md_plain(git_text(repo, f"{AMENDMENT_COMMIT}^:CHANGELOG.md"))
    wg_p2a = (re.sub(r"\s+", " ", wg.split("\n## PATH-2a: ", 1)[1].split("\n## ", 1)[0])
              if "\n## PATH-2a: " in wg else "")
    gp1_facts = {
        "the sentence is in [7.49.0] and stood under [Unreleased]": gp1 in md_section and gp1 in md_plain(pre_unrel),
        "it was written before the amendment": gp1 in before_amend,
        "the amendment's commit adds only it": touched(AMENDMENT_COMMIT) == {AMENDMENT},
        "the amendment is in #187 and on main before the cut": (
            is_ancestor(AMENDMENT_COMMIT, PATH2A_MERGE) and subject(PATH2A_MERGE).endswith("(#187)")
            and is_ancestor(AMENDMENT_COMMIT, BRANCH_POINT)),
        "the amendment says what the line says": all(s in amend for s in (
            "# AMENDMENT — PATH-2's repair is not what lands; withholding is. The operator's decision.",
            "Fathom Lab · 2026-10-06", "on the operator's instruction of 2026-10-06",
            "**G-P1 is not met, and this branch does not claim it.**",
            "**Withholding is accepted as what lands now.**",
            "#97, #101 and #121 stay open",
            "It is not the answer PREREG_path2 asked for. Repair remains the open goal.")),
        # web/gate/README.md's PATH-2a section, and the entry's "Open for the operator" paragraph,
        # leave open what the amendment decides (its decisions 1, 3 and 4)
        "web/gate/README.md's PATH-2a section leaves G-P1, and the switch's removal, to the operator": (
            "so PATH-2a does not meet G-P1: whether it stands in for it is the operator's decision" in wg_p2a
            and "The ninth pass removed it by the lead's decision of 2026-10-04, which the operator "
                "confirms at merge." in wg_p2a),
        "the entry's 'Open for the operator' lists G-P1, the switch's removal and bar A's boundary": (
            "**Open for the operator.** G-P1 (not met). The lead's decision to remove the switch" in p2a_entry
            and "The boundary of bar A as the lead restated it on 2026-10-05" in p2a_entry),
        "the amendment decides all three (decisions 1, 3 and 4)": all(s in amend for s in (
            "1. **G-P1 is not met, and this branch does not claim it.**",
            "3. **The removal of the gate-agreement switch is confirmed.**",
            "4. **The boundary of bar A is confirmed as the lead restated it at pass eleven.**")),
    }
    c.add(all(gp1_facts.values()),
          "stale: the PATH-2a entry leaves G-P1 to the operator; the operator decided on 2026-10-06 "
          f"({AMENDMENT.rsplit('/', 1)[-1]}, {AMENDMENT_COMMIT}, before #187 merged): G-P1 not met, "
          "withholding accepted as what lands and not as the repair, #97, #101 and #121 open; "
          "web/gate/README.md's PATH-2a section leaves G-P1 and the switch's removal to the operator, "
          "and the entry's 'Open for the operator' lists G-P1, the switch and bar A, which the "
          "amendment decides (1, 3, 4)",
          "" if all(gp1_facts.values()) else f"not as stated: {[k for k, v in gp1_facts.items() if not v]}")
    gate_js = git_text(repo, f"{TAG}:web/gate/diffgate.js")
    bm_src = git_text(repo, f"{TAG}:web/gate/bookmarklet_src.js")
    bm_ui = git_text(repo, f"{TAG}:web/gate/bookmarklet_ui.js")
    drift = wg.split("## Drift: the port is ahead of the release", 1)
    c.add(len(drift) == 2 and "Until 7.48.0 ships" in drift[1].split("\n## ", 1)[0]
          and "until 7.48.0 ships" in py_side
          and "that 7.48.0 ships once they merge" in gate_js
          and "that 7.48.0 ships once they merge" in bm_src
          and "COMPAT-2 checkout (7.48.0)" in bm_ui
          and "still calls it the 7.48.0 port" in wg,
          "stale: web/gate's README drift section, py_side.py, the diffgate.js and bookmarklet_src.js "
          "headers speak of 7.48.0 as not shipped; the panel text calls its port 7.48.0's")
    dg_comment = re.sub(r"\s+", " ", " ".join(ln.strip().lstrip("#").strip() for ln in dg.splitlines()
                                               if ln.strip().startswith("#")))
    c.add("the instrument itself is the released package" in dg_comment
          and "imports the `styxx` package beside its script, at the ref the workflow names, not the "
              "one pip installs" in norm_section,
          "stale: diffgate.yml's header comment says the instrument is the released package")
    sworn_readme = git_text(repo, f"{TAG}:sworn/README.md")
    sworn_action = git_text(repo, f"{TAG}:sworn/action.yml")
    c.add('`styxx-source: "styxx==7.48.0"`' in sworn_readme and "`styxx==7.48.0` installs" in sworn_action
          and "Affected: 7.47.0 and 7.48.0 through the capsule verifier" in norm_7481
          and PREV_ADVISORY in norm_7481 and "Affected: 7.47.0, 7.48.0 and 7.48.1." in norm_section,
          "stale: sworn's README and action.yml name styxx==7.48.0, which both advisories name as affected")
    # the leaderboard `styxx leaderboard` prints and two comments in styxx/__init__.py still carry
    # the ordinal priority claim for Baseline-019 that critique.py's docstring withdrew; the
    # 2026-09-29 note and [7.49.0] record that withdrawal and name neither. The word is taken from
    # CHARTER_FORBIDDEN and never spelled here.
    ordinal = CHARTER_FORBIDDEN[0]
    ord_rx = r"(?<![a-z0-9])" + re.escape(ordinal) + r"(?![a-z0-9])"
    lb_rows = [ln for ln in git_text(repo, f"{TAG}:{LEADERBOARD}").splitlines() if "**Baseline-019**" in ln]
    init_hits = [ln for ln in git_text(repo, f"{TAG}:styxx/__init__.py").splitlines()
                 if "#" in ln and re.search(ord_rx + r"-pass\b", ln.split("#", 1)[1].lower())]
    cli = git_text(repo, f"{TAG}:styxx/cli.py")
    cmd_lb = cli.split("\ndef cmd_leaderboard(", 1)[1].split("\ndef ", 1)[0] if "\ndef cmd_leaderboard(" in cli else ""
    try:
        with zipfile.ZipFile(BUNDLE / EXPECTED_FILES[1][0]) as z:
            in_wheel = LEADERBOARD in z.namelist()
    except (OSError, zipfile.BadZipFile):
        in_wheel = False

    def module_doc(rev: str) -> str:
        try:
            doc = ast.get_docstring(ast.parse(git_text(repo, f"{rev}:styxx/critique.py"))) or ""
        except (SyntaxError, ValueError):
            doc = ""
        return re.sub(r"\s+", " ", doc).lower()

    crit_claim = f"the {ordinal} method to pass the styxx gauntlet's v3 detection bars"
    note_n = re.sub(r"\s+", " ", git_text(repo, f"{TAG}:{PRIOR_ART_NOTE}"))
    lb_facts = {
        "LEADERBOARD.md's Baseline-019 row makes the ordinal claim": (
            len(lb_rows) == 1 and re.search(ord_rx + r" real pass on the leaderboard", lb_rows[0].lower()) is not None),
        "the root LEADERBOARD.md is the same file, and neither changed since 7.48.1": (
            LEADERBOARD in tree and tree.get("LEADERBOARD.md") == tree.get(LEADERBOARD)
            and git(repo, "diff", "--quiet", "v7.48.1", TAG, "--", LEADERBOARD, "LEADERBOARD.md",
                    check=False).returncode == 0),
        "styxx leaderboard prints it, and the wheel carries it": (
            'pkg_data = _Path(__file__).resolve().parent / "_data" / "LEADERBOARD.md"' in cmd_lb
            and "print(text)" in cmd_lb and "p_leaderboard.set_defaults(func=cmd_leaderboard)" in cli
            and in_wheel),
        "two comments in styxx/__init__.py make it, both on the critique detector": (
            len(init_hits) == 2 and all("critique" in ln.lower() for ln in init_hits)),
        "critique.py's docstring withdrew it": (
            crit_claim in module_doc("v7.48.1") and crit_claim not in module_doc(TAG)
            and "it passed the styxx gauntlet's v3 detection bars" in module_doc(TAG)),
        "the 2026-09-29 note and [7.49.0] record that withdrawal and name neither": (
            "| `styxx/critique.py` docstring" in note_n
            and f"\"the {ordinal} method to PASS the styxx gauntlet's v3 detection bars\"" in note_n
            and "\"It passed the styxx gauntlet's v3 detection bars\"" in note_n
            and f"`styxx/critique.py` (\"the {ordinal} method to PASS\" the gauntlet's v3 bars)" in norm_section
            and "LEADERBOARD" not in note_n and "Baseline-019" not in note_n
            and "LEADERBOARD" not in section and "Baseline-019" not in section),
    }
    c.add(all(lb_facts.values()),
          f"stale: {LEADERBOARD} (what styxx leaderboard prints, in the wheel; the root copy is the same "
          "file) and two comments in styxx/__init__.py carry the ordinal priority claim for Baseline-019 "
          "that critique.py's docstring withdrew; the 2026-09-29 note and [7.49.0] name neither",
          "" if all(lb_facts.values()) else f"not as stated: {[k for k, v in lb_facts.items() if not v]}")
    c.add("| DOI (concept, always-latest) | [10.5281/zenodo.19326174]" in readme
          and "concept_DOI-always--latest" in readme and "doi.org/10.5281/zenodo.19326174)" in readme
          and "software concept DOI [10.5281/zenodo.19758618]" in readme,
          "stale: README labels 19326174 always-latest (badge row and link table); its citation row "
          "names 19758618")
    c.add(all(k in defects and "resolved" not in defects[k] for k in ("D2", "D3", "D4", "D5")),
          "D2, D3, D4 and D5 are open in zenodo/MANIFEST.json at the tag")
    c.add('value: "10.5281/zenodo.19758618"' in cff and 'doi: "10.5281/zenodo.19777921"' in
          cff.split("preferred-citation:", 1)[-1].split("\n\n", 1)[0],
          "CITATION.cff names the concept DOI; its preferred-citation is 10.5281/zenodo.19777921")
    c.add("2026-06-21" in defects.get("D2", {}).get("defect", ""),
          "D2 says the position paper predates its 2026-06-21 scope erratum")
    # the EXTERNAL-1 line, held to CHANGELOG [7.48.1], the builder and the committed packet
    cmf = "papers/closed-model-frontier"
    packet_py = git_text(repo, f"{TAG}:{cmf}/external1_packet.py")
    packet = json.loads(git_text(repo, f"{TAG}:{cmf}/external1_packet.json"))
    zz_ids = sorted(str(it.get("id")) for it in packet.get("items", [])
                    if str((it.get("claim_detail") or {}).get("path") or "")
                    .rsplit("/", 1)[-1].startswith("zz_"))
    ext1 = {
        "description": all(s in stale_plain for s in (
            f"The committed EXTERNAL-1 packet, {cmf}/external1_packet.json, still carries the id leak.",
            "Since #125 its builder, external1_packet.py, numbers items by their shuffled position by "
            "default, and its build --as-published mode writes the pre-repair numbering on purpose, "
            "leak included, because the committed packet, key and digest are receipts; that mode has "
            "been shown only on a synthetic ledger and shelf, not run on the real ones.",
            "Whether any adjudicator used the leak cannot be re-tested.")),
        "CHANGELOG [7.48.1]": ("`build` now numbers items by their shuffled position" in norm_7481
                               and "`build --as-published` writes, on a synthetic ledger and shelf, "
                                   "exactly the bytes the pre-repair builder wrote (arm-ordered "
                                   "numbering, key and digest, leak included), because EXTERNAL-1's "
                                   "committed packet, key and digest are receipts" in norm_7481
                               and "It has not been run on the real shelf and ledger" in norm_7481
                               and "`--as-published` has not been run against the real shelf and the "
                                   "pre-correction ledger" in norm_7481
                               and "The `#125` paragraph below now says `--as-published` was shown on "
                                   "a synthetic ledger only." in norm_7481
                               and "the committed EXTERNAL-1 packet still carries the id leak, and "
                                   "whether any adjudicator used it cannot be re-tested" in norm_7481),
        # the limit 7.48.1's own record carried, after the lab's erratum added it
        "7.48.1 as published": ("that mode has been shown only on a synthetic ledger and shelf, not "
                                "run on the real ones" in base_stale),
        "builder": ("def build(as_published: bool = False)" in packet_py
                    and 'iid = f"E1-{(arm_pos if as_published else pos):03d}"' in packet_py
                    and "leak included" in packet_py),
        "committed packet": zz_ids == [f"E1-{i:03d}" for i in range(115, 130)],
        "unchanged since 7.48.1": git(repo, "diff", "--quiet", "v7.48.1", TAG, "--",
                                      f"{cmf}/external1_packet.py", f"{cmf}/external1_packet.json",
                                      check=False).returncode == 0,
    }
    c.add(all(ext1.values()),
          "stale: the EXTERNAL-1 packet still carries the id leak; the builder numbers by shuffled "
          "position by default and build --as-published writes the leak on purpose, shown only on "
          "a synthetic ledger and shelf (CHANGELOG [7.48.1] with its erratum, 7.48.1 as published, "
          "external1_packet.py and external1_packet.json at the tag)",
          "the 15 zz_ items are E1-115..E1-129" if all(ext1.values())
          else f"not as stated: {[k for k, v in ext1.items() if not v]}")
    zr = git_text(repo, f"{TAG}:zenodo/README.md")
    c.add("the styxx 7.48.0 software version" in zr and "7.48.1" not in zr and "7.49.0" not in zr
          and "`zenodo/README.md` still describes the 7.48.0 flow only." in norm_section,
          "stale: zenodo/README.md describes the 7.48.0 flow only")
    c.add("Latest version: 7.48.1, 10.5281/zenodo.23200977" in (concept_entry.get("notes") or ""),
          "stale: zenodo/MANIFEST.json's 19758618 entry names 7.48.1 (23200977) as the latest version")

    # -- the deposit's own sections ---------------------------------------------------------------
    pub_yml = git_text(repo, f"{TAG}:.github/workflows/publish.yml")
    c.add(re.search(r"(?m)^\s+attestations: false\s*$", pub_yml) is not None
          and "attestations: false" in sec_md and "SHA-256" in sec_md
          and "uploads with <code>attestations: false</code>" in desc[files_at:],
          "publish.yml sets attestations: false and SECURITY.md says to check by SHA-256")
    c.add(f"each of its {len(tree)} files is byte-identical" in desc,
          "the bundle's file count in the description = the tree's blob count", str(len(tree)))


def guard_selftest(c: Checks) -> None:
    ctx = AllowContext(latest_id=PREDECESSOR_RECID, draft_id=2, bucket="0" * 32)
    refused = []
    for action in ("publish", "edit", "discard"):
        try:
            check_allowed("POST", f"{ZENODO_API}/deposit/depositions/2/actions/{action}", ctx)
        except Refuse:
            refused.append(action)
    c.add(refused == ["publish", "edit", "discard"],
          "allowlist refuses POST .../actions/{publish,edit,discard}", str(refused))
    probes = [
        ("PUT", "https://example.org/api/files/x/y", "a host other than zenodo.org"),
        ("GET", f"http://{ZENODO_HOST}/api/records/{CONCEPT_RECID}", "plain http"),
        ("POST", f"{ZENODO_API}/deposit/depositions/{PREDECESSOR_RECID + 1}/actions/newversion",
         "newversion on a record other than the latest"),
        ("DELETE", f"{ZENODO_API}/deposit/depositions/3/files/abc", "a DELETE on another draft"),
        ("DELETE", f"{ZENODO_API}/deposit/depositions/2", "a DELETE of the draft itself"),
        ("DELETE", f"{ZENODO_API}/deposit/depositions/2/files/..",
         "a DELETE with file id '..' (sent as a DELETE of the draft itself)"),
        ("DELETE", f"{ZENODO_API}/deposit/depositions/2/files/.",
         "a DELETE with file id '.' (sent as a DELETE of the draft's file list)"),
        ("PUT", f"{ZENODO_API}/deposit/depositions/{PREDECESSOR_RECID}", "a PUT on the published record"),
        ("PUT", f"{ZENODO_API}/files/{'1' * 32}/x.zip", "a PUT into another bucket"),
    ]
    bad = []
    for method, url, what in probes:
        try:
            check_allowed(method, url, ctx)
            bad.append(what)
        except Refuse:
            pass
    try:   # the rule still lets a plain file id through, and the run's own check agrees with it
        check_allowed("DELETE", f"{ZENODO_API}/deposit/depositions/2/files/2-0", ctx)
    except Refuse:
        bad.append("a DELETE of a plain file id of the draft was refused")
    if safe_file_id(".") or safe_file_id("..") or not safe_file_id("2-0"):
        bad.append("the run's file id check takes '.' or '..', or refuses a plain id")
    c.add(not bad, "allowlist pins every mutating request to the one record it may touch "
          "(no DELETE with a file id of . or ..)",
          f"allowed: {bad}" if bad else f"{len(probes)} probes refused")


def preflight(repo_arg: str, skip_tree: bool) -> tuple[Checks, list[dict], dict | None]:
    c = Checks()
    repo = repo_arg if repo_available(repo_arg) else None
    c.add(True if repo else None, f"repository at {repo_arg} holds {TAG} -> {TAG_COMMIT[:12]}",
          "" if repo else "not found; repository checks are skipped")
    tree = tag_tree(repo) if repo else None
    facts = check_files(c)
    if skip_tree:
        c.add(None, "source bundle is the v7.49.0 tree", "--skip-tree-check")
    elif not c.failed:
        check_bundle_tree(c, repo, tree)
    if len(facts) == len(EXPECTED_FILES) and not c.failed:
        check_packages_against_tree(c, tree)
    doc = check_metadata(c, facts, repo, tree) if len(facts) == len(EXPECTED_FILES) else None
    guard_selftest(c)
    return c, facts, doc


# ---------------------------------------------------------------------------
# the network path (never publishes)
# ---------------------------------------------------------------------------

# A draft file id as one plain path segment. Never "." or "..": requests removes dot segments
# before it sends, so DELETE .../depositions/{d}/files/.. goes out as a DELETE of the draft itself,
# and .../files/. as a DELETE of its file list.
FILE_ID_RX = r"(?!\.{1,2}$)[A-Za-z0-9._-]+"


def safe_file_id(fid: object) -> bool:
    return isinstance(fid, str) and re.fullmatch(FILE_ID_RX, fid) is not None


class AllowContext:
    """What the mutating requests may touch, filled in as the run learns it."""

    def __init__(self, latest_id: int | None = None, draft_id: int | None = None,
                 bucket: str | None = None) -> None:
        self.latest_id = latest_id
        self.draft_id = draft_id
        self.bucket = bucket


def check_allowed(method: str, url: str, ctx: AllowContext) -> None:
    u = urlparse(url)
    if u.scheme != "https" or u.hostname != ZENODO_HOST or u.port not in (None, 443):
        raise Refuse(f"request to {u.scheme}://{u.hostname} is not allowed")
    path = u.path
    d, lt, b = ctx.draft_id, ctx.latest_id, ctx.bucket
    allowed = [
        ("GET", rf"^/api/records/{CONCEPT_RECID}/versions/latest$"),
        ("GET", rf"^/api/records/{CONCEPT_RECID}$"),
        ("GET", r"^/api/deposit/depositions$"),
        ("GET", r"^/api/deposit/depositions/\d+$"),
        ("GET", r"^/api/deposit/depositions/\d+/files$"),
    ]
    if lt is not None:
        allowed.append(("POST", rf"^/api/deposit/depositions/{lt}/actions/newversion$"))
    if d is not None:
        allowed += [
            ("DELETE", rf"^/api/deposit/depositions/{d}/files/{FILE_ID_RX}$"),
            ("PUT", rf"^/api/deposit/depositions/{d}$"),
        ]
    if b:
        allowed.append(("PUT", rf"^/api/files/{re.escape(b)}/[A-Za-z0-9._+-]+$"))
    if not any(m == method and re.match(rx, path) for m, rx in allowed):
        raise Refuse(f"{method} {path} is not on this script's allowlist")


class Zenodo:
    def __init__(self, token: str, ctx: AllowContext, allow=check_allowed) -> None:
        try:
            import requests
        except ImportError as e:
            raise Refuse("the 'requests' package is required for the real run "
                         "(pip install requests)") from e
        self._requests = requests
        self._s = requests.Session()
        self._token = token
        self.ctx = ctx
        self._allow = allow
        self.log: list[dict] = []

    def call(self, method: str, url: str, *, params=None, json_body=None, data=None,
             timeout=(30, 60), expect=(200,)):
        self._allow(method, url, self.ctx)
        headers = {"Authorization": f"Bearer {self._token}", "User-Agent": UA,
                   "Accept": "application/json"}
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        if data is not None:
            headers["Content-Type"] = "application/octet-stream"
        t0 = time.time()
        try:
            r = self._s.request(method, url, params=params, json=json_body, data=data,
                                headers=headers, timeout=timeout,
                                allow_redirects=(method == "GET"))
        except self._requests.RequestException as e:
            self.log.append({"method": method, "path": urlparse(url).path, "status": None,
                             "error": type(e).__name__, "seconds": round(time.time() - t0, 2)})
            raise Transport(f"{method} {urlparse(url).path}: {type(e).__name__}: {redact(e)}") from None
        if urlparse(r.url).hostname != ZENODO_HOST:
            raise Refuse(f"{method} {urlparse(url).path} was redirected off {ZENODO_HOST}")
        self.log.append({"method": method, "path": urlparse(url).path, "status": r.status_code,
                         "seconds": round(time.time() - t0, 2)})
        if r.status_code not in expect:
            raise HttpStatus(f"{method} {urlparse(url).path} -> HTTP {r.status_code}: "
                             f"{redact(r.text)[:1500]}", r.status_code)
        return r


def md5_of(checksum: str | None) -> str | None:
    if not checksum:
        return None
    return checksum.split(":", 1)[1] if checksum.startswith("md5:") else checksum


def license_id(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("id")
    return str(value or "").lower()


def compare_metadata(sent: dict, got: dict) -> dict[str, bool]:
    """Field by field: what Zenodo holds against what metadata.json says."""
    def rel_rows(rows):
        return sorted((x.get("relation"), x.get("identifier"), x.get("scheme"),
                       x.get("resource_type") or "") for x in rows or [])

    def people(rows):
        return [(x.get("name"), x.get("affiliation") or "", x.get("orcid") or "") for x in rows or []]

    return {
        "title": got.get("title") == sent["title"],
        "version": got.get("version") == sent["version"],
        "publication_date": got.get("publication_date") == sent["publication_date"],
        "upload_type": got.get("upload_type") == sent["upload_type"],
        "access_right": got.get("access_right") == sent["access_right"],
        "language": got.get("language") == sent["language"],
        "license": license_id(got.get("license")) in LICENSE_READBACK.get(sent["license"], {sent["license"]}),
        "creators": people(got.get("creators")) == people(sent["creators"]),
        "keywords": sorted(got.get("keywords") or []) == sorted(sent["keywords"]),
        "related_identifiers": rel_rows(got.get("related_identifiers")) == rel_rows(sent["related_identifiers"]),
        "description": (got.get("description") or "").strip() == sent["description"].strip(),
        "notes": (got.get("notes") or "").strip() == (sent.get("notes") or "").strip(),
    }


def unexpected_metadata_keys(sent: dict, got: dict) -> list[str]:
    """Keys Zenodo holds that metadata.json does not send and Zenodo does not add on its own.
    compare_metadata compares the sent fields only; this catches what was added beside them."""
    return sorted(set(got or {}) - set(sent) - ZENODO_ADDED_KEYS)


def write_json(path: Path, obj: dict) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    blob = redact(text).encode("utf-8")
    path.write_bytes(blob)
    if path.read_bytes() != blob:
        raise Refuse(f"{path.name} did not read back identical")


def write_receipt(receipt: dict) -> None:
    receipt["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(RECEIPT_PATH, receipt)
    say(f"  receipt written: {RECEIPT_PATH.name} (status {receipt['status']})")


def run(token: str, token_source: str, doc: dict, facts: list[dict],
        resume_draft: int | None) -> int:
    ctx = AllowContext()
    z = Zenodo(token, ctx)
    receipt: dict = {
        "schema": "fathom/zenodo-draft-receipt/v2",
        "status": "started",
        "published": False,
        # what the next step is, not a record of who takes it: 7.48.1's provenance NOTE found its
        # receipt saying the operator would read the draft, and the publish followed in 21 seconds
        "publish_step": f"{PUBLISH_SCRIPT} publishes only the draft_id below, only if this status is "
                        "draft_ready_unpublished and --confirm repeats that id. Read the draft at "
                        "draft_url before running it; this receipt does not record whether anyone "
                        "did",
        "concept_recid": CONCEPT_RECID,
        "concept_doi": CONCEPT_DOI,
        "version": VERSION,
        "tag": TAG,
        "tag_commit": TAG_COMMIT,
        "token_source": token_source,
        "metadata_json_sha256": hashlib.sha256(METADATA_PATH.read_bytes()).hexdigest(),
        "description_sha256": hashlib.sha256(doc["metadata"]["description"].encode("utf-8")).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "excluded_markers": markers_fingerprint(),
        "local_files": [{k: f[k] for k in ("name", "bytes", "sha256", "md5")} for f in facts],
    }

    def fail(step: str, msg: str) -> int:
        receipt["status"] = f"stopped_at_{step}"
        receipt["stopped_because"] = redact(msg)
        receipt["requests"] = z.log
        if "draft_id" in receipt:
            write_receipt(receipt)
        say(f"\nSTOPPED at {step}: {redact(msg)}")
        if "draft_id" in receipt:
            say(f"  An unpublished draft exists: {receipt.get('draft_url')}")
            say("  Re-run with --resume-draft "
                f"{receipt['draft_id']} after reading it, or discard it in the Zenodo UI. "
                "Nothing was published.")
        else:
            say("  No draft was created or changed. Nothing was published.")
        return 1

    rule(f"1. latest version of concept {CONCEPT_RECID} (resolved, not assumed)")
    try:
        try:
            rec = z.call("GET", f"{ZENODO_API}/records/{CONCEPT_RECID}/versions/latest").json()
        except Refuse as e:
            say(f"  versions/latest failed ({e}); asking the concept record directly")
            rec = z.call("GET", f"{ZENODO_API}/records/{CONCEPT_RECID}").json()
    except (Refuse, ValueError) as e:
        return fail("resolve_latest", str(e))
    if str(rec.get("conceptrecid")) != CONCEPT_RECID:
        return fail("resolve_latest", f"record {rec.get('id')} has conceptrecid "
                    f"{rec.get('conceptrecid')}, not {CONCEPT_RECID}")
    try:
        latest_id = int(rec["id"])
    except (KeyError, TypeError, ValueError):
        return fail("resolve_latest", f"no record id in the latest-version response: {rec.get('id')!r}")
    latest_version = (rec.get("metadata") or {}).get("version")
    latest_files = {(f.get("key") or f.get("filename"), md5_of(f.get("checksum")))
                    for f in rec.get("files") or []}
    receipt["predecessor"] = {"id": latest_id, "doi": rec.get("doi"), "version": latest_version,
                              "title": (rec.get("metadata") or {}).get("title"),
                              "files": sorted(k for k, _ in latest_files if k)}
    say(f"  latest: id {latest_id}  doi {rec.get('doi')}  version {latest_version}")
    if latest_version == VERSION:
        return fail("resolve_latest", f"the latest version is already {VERSION} "
                    f"(record {latest_id}); nothing to do")
    # metadata.json names 23200977 (7.48.1) as the version before this one: isNewVersionOf, the
    # description and the notes. A draft made from any other record would carry a false relation
    # and a false sentence, so any other latest version stops the run.
    if latest_id != PREDECESSOR_RECID:
        return fail("resolve_latest", f"the latest version is record {latest_id} "
                    f"({latest_version!r}), not {PREDECESSOR_RECID} ({PREDECESSOR_VERSION}). "
                    f"metadata.json names {PREDECESSOR_RECID} as the release before this one "
                    "(isNewVersionOf, the description and the notes), so a draft made from "
                    f"{latest_id} would carry a false relation. Decide by hand, and rebuild the "
                    "metadata from the record that is the latest before any run")
    if latest_version != PREDECESSOR_VERSION:
        return fail("resolve_latest", f"record {latest_id} reports version {latest_version!r}, "
                    f"not {PREDECESSOR_VERSION}")
    say(f"  the latest version is {PREDECESSOR_RECID} ({PREDECESSOR_VERSION}), as the lab's receipts "
        "and metadata.json say")

    if resume_draft is None:
        rule("2. refuse if an unpublished draft already exists in this concept")
        try:
            listed = z.call("GET", f"{ZENODO_API}/deposit/depositions",
                            params={"status": "draft", "size": 100}).json()
            open_drafts = [d.get("id") for d in listed
                           if str(d.get("conceptrecid")) == CONCEPT_RECID
                           and not d.get("submitted")]
        except (Refuse, ValueError, AttributeError) as e:
            open_drafts = None
            say(f"  could not list drafts ({redact(e)}); the fresh-copy guard after newversion "
                "still applies")
        if open_drafts:
            return fail("existing_draft", f"unpublished draft(s) {open_drafts} already exist in "
                        f"concept {CONCEPT_RECID}. Read them at https://zenodo.org/uploads/<id>; "
                        "re-run with --resume-draft <id> to take one over, or discard it")
        if open_drafts == []:
            say("  none")

        rule(f"3. new version from record {latest_id}")
        try:
            dep = z.call("GET", f"{ZENODO_API}/deposit/depositions/{latest_id}").json()
            if not dep.get("submitted"):
                return fail("new_version", f"deposition {latest_id} is not published")
            ctx.latest_id = latest_id
            nv = z.call("POST", f"{ZENODO_API}/deposit/depositions/{latest_id}/actions/newversion",
                        expect=(200, 201)).json()
            ctx.latest_id = None
        except Refuse as e:
            return fail("new_version", str(e))
        link = (nv.get("links") or {}).get("latest_draft") or ""
        mm = re.search(r"/deposit/depositions/(\d+)$", link)
        if not mm:
            return fail("new_version", f"no latest_draft link in the newversion response: {link!r}")
        draft_id = int(mm.group(1))
        if draft_id == latest_id:
            return fail("new_version", "newversion returned the published record, not a draft")
    else:
        draft_id = int(resume_draft)
        rule(f"2-3. resuming draft {draft_id} named by --resume-draft")

    try:
        draft = z.call("GET", f"{ZENODO_API}/deposit/depositions/{draft_id}").json()
    except Refuse as e:
        return fail("fetch_draft", str(e))
    if str(draft.get("conceptrecid")) != CONCEPT_RECID:
        return fail("fetch_draft", f"draft {draft_id} has conceptrecid "
                    f"{draft.get('conceptrecid')}, not {CONCEPT_RECID}; nothing was changed in it")
    if draft.get("submitted") or draft.get("state") == "done":
        return fail("fetch_draft", f"deposition {draft_id} is published; refusing to touch it")
    receipt["draft_id"] = draft_id
    receipt["draft_url"] = (draft.get("links") or {}).get("html") or f"https://zenodo.org/uploads/{draft_id}"
    receipt["draft_api_url"] = f"{ZENODO_API}/deposit/depositions/{draft_id}"
    existing = draft.get("files") or []
    dv = (draft.get("metadata") or {}).get("version")
    if resume_draft is None:
        extra = [(f.get("filename"), md5_of(f.get("checksum"))) for f in existing
                 if (f.get("filename"), md5_of(f.get("checksum"))) not in latest_files]
        if dv not in (None, "", latest_version) or extra:
            receipt.pop("draft_id")
            return fail("fresh_copy_guard", f"draft {draft_id} is not a fresh copy of record "
                        f"{latest_id} (version {dv!r}, files not in the predecessor {extra}); it "
                        "looks like a draft someone already worked on. Nothing was changed in it. "
                        f"Read it at https://zenodo.org/uploads/{draft_id}")
    elif dv not in (None, "", latest_version, VERSION):
        receipt.pop("draft_id")
        return fail("fetch_draft", f"draft {draft_id} holds version {dv!r}, neither "
                    f"{latest_version} nor {VERSION}; nothing was changed in it")
    ctx.draft_id = draft_id
    bucket_url = (draft.get("links") or {}).get("bucket", "")
    mb = re.fullmatch(rf"https://{re.escape(ZENODO_HOST)}/api/files/([0-9a-fA-F-]{{32,36}})", bucket_url)
    if not mb:
        return fail("fetch_draft", f"draft {draft_id} has no usable bucket link: {bucket_url!r}")
    ctx.bucket = mb.group(1)
    say(f"  draft {draft_id}: {receipt['draft_url']}")
    receipt["status"] = "draft_created" if resume_draft is None else "draft_resumed"
    write_receipt(receipt)

    rule("4. clear the draft of every file that is not one of the three")
    wanted = {(f["name"], f["md5"], f["bytes"]) for f in facts}
    keep, deleted = set(), []
    try:
        for f in existing:
            key = (f.get("filename"), md5_of(f.get("checksum")), f.get("filesize"))
            if key in wanted and key[0] not in keep:
                keep.add(key[0])
                say(f"  kept {key[0]} (name, md5 and size match the local file)")
                continue
            if not safe_file_id(f.get("id")):
                raise Refuse(f"the draft's file {f.get('filename')!r} has id {f.get('id')!r}, which "
                             "is not one plain path segment (or is . or ..); no DELETE was sent for it")
            z.call("DELETE", f"{ZENODO_API}/deposit/depositions/{draft_id}/files/{f['id']}",
                   expect=(200, 204))
            deleted.append(f.get("filename"))
            say(f"  deleted {f.get('filename')}")
        left = z.call("GET", f"{ZENODO_API}/deposit/depositions/{draft_id}/files").json()
    except (Refuse, ValueError, KeyError) as e:
        return fail("clear_files", str(e))
    left_names = sorted(x.get("filename") for x in left)
    if left_names != sorted(keep):
        return fail("clear_files", f"draft lists {left_names} after clearing, expected {sorted(keep)}")
    receipt["files_deleted"] = deleted
    receipt["files_kept"] = sorted(keep)
    say(f"  {len(deleted)} file(s) deleted, {len(keep)} kept")

    rule("5. upload the files the draft does not hold")
    uploaded = []
    for f in facts:
        if f["name"] in keep:
            continue
        try:
            with open(f["path"], "rb") as fh:
                r = z.call("PUT", f"{bucket_url}/{f['name']}", data=fh, timeout=(30, 3600),
                           expect=(200, 201))
            body = r.json()
        except (Refuse, ValueError, OSError) as e:
            return fail("upload", f"{f['name']}: {e}")
        remote_md5, remote_size = md5_of(body.get("checksum")), body.get("size")
        uploaded.append({"name": f["name"], "put_md5": remote_md5, "put_size": remote_size})
        if remote_md5 != f["md5"] or remote_size != f["bytes"]:
            return fail("upload", f"{f['name']}: Zenodo stored md5 {remote_md5} / {remote_size} "
                        f"bytes, local {f['md5']} / {f['bytes']}")
        say(f"  uploaded {f['name']}  {remote_size} bytes  md5 {remote_md5}  (matches local)")
    receipt["uploads"] = uploaded

    rule("6. set metadata from metadata.json and read it back")
    try:
        z.call("PUT", f"{ZENODO_API}/deposit/depositions/{draft_id}", json_body=doc,
               timeout=(30, 120))
        back = z.call("GET", f"{ZENODO_API}/deposit/depositions/{draft_id}").json()
    except (Refuse, ValueError) as e:
        return fail("metadata", str(e))
    got = back.get("metadata") or {}
    comparisons = compare_metadata(doc["metadata"], got)
    receipt["metadata_readback"] = comparisons
    receipt["license_as_stored"] = license_id(got.get("license"))
    reserved = (got.get("prereserve_doi") or {}).get("doi")
    if reserved:
        receipt["reserved_doi_not_registered_until_publish"] = reserved
    for k, ok in comparisons.items():
        say(f"  [{'same' if ok else 'DIFFERS'}] {k}")
    differs = [k for k, ok in comparisons.items() if not ok]
    extra = unexpected_metadata_keys(doc["metadata"], got)
    receipt["metadata_keys_beyond_sent"] = extra
    say(f"  [{'none' if not extra else 'EXTRA'}] keys beyond those sent "
        f"(+ {', '.join(sorted(ZENODO_ADDED_KEYS))})" + (f": {extra}" if extra else ""))
    if extra:
        differs.append(f"keys metadata.json does not send: {extra}")
    if reserved and reserved != f"10.5281/zenodo.{draft_id}":
        differs.append(f"reserved DOI {reserved} is not 10.5281/zenodo.{draft_id}")

    rule("7. verify the draft's files: Zenodo md5 and size against local")
    try:
        remote = z.call("GET", f"{ZENODO_API}/deposit/depositions/{draft_id}/files").json()
    except (Refuse, ValueError) as e:
        return fail("verify_files", str(e))
    by_name = {x.get("filename"): x for x in remote}
    checks, ok_all = [], (sorted(by_name) == sorted(f["name"] for f in facts) and len(remote) == len(facts))
    for f in facts:
        x = by_name.get(f["name"], {})
        rmd5, rsize = md5_of(x.get("checksum")), x.get("filesize")
        ok = rmd5 == f["md5"] and rsize == f["bytes"]
        ok_all &= ok
        checks.append({"name": f["name"], "bytes": f["bytes"], "sha256": f["sha256"],
                       "local_md5": f["md5"], "zenodo_md5": rmd5, "zenodo_filesize": rsize,
                       "match": ok})
        say(f"  [{'match' if ok else 'MISMATCH'}] {f['name']}  local md5 {f['md5']}  "
            f"zenodo md5 {rmd5}  size {rsize}")
    receipt["file_checksums"] = checks
    if not ok_all:
        return fail("verify_files", f"the draft's files do not match: {sorted(by_name)}")
    if differs:
        return fail("metadata_readback", f"files verified, but Zenodo read back these metadata "
                    f"fields differently from metadata.json: {differs}. Compare them in the "
                    "draft before anyone publishes")

    receipt["status"] = "draft_ready_unpublished"
    receipt["requests"] = z.log
    write_receipt(receipt)
    rule("DRAFT READY. NOT PUBLISHED.")
    say(f"  read it:  {receipt['draft_url']}")
    if reserved:
        say(f"  reserved DOI (registered only if published): {reserved}")
    say("  This script sent no publish request and has no way to. After reading the draft,")
    say(f"  publish it with:  python {PUBLISH_SCRIPT} --token-file PATH --confirm {draft_id}")
    say("  or discard it in the Zenodo UI.")
    return 0


# ---------------------------------------------------------------------------

def print_plan(doc: dict, facts: list[dict]) -> None:
    body = METADATA_PATH.read_bytes()
    rule("PLANNED REQUESTS (the real run, in order; nothing below was sent)")
    say(f"  every request: https only, host {ZENODO_HOST}, headers "
        f"'Authorization: Bearer <token>' + 'User-Agent: {UA}'")
    say(f"  token at run time: {TOKEN_SOURCE}; neither the value nor the path is ever printed")
    plan = [
        ("GET", f"{ZENODO_API}/records/{CONCEPT_RECID}/versions/latest",
         f"LATEST_ID / LATEST_VERSION; stop unless conceptrecid is {CONCEPT_RECID}, LATEST_ID is "
         f"{PREDECESSOR_RECID} and LATEST_VERSION is {PREDECESSOR_VERSION} (the release metadata.json "
         "names as the one before this)"),
        ("GET", f"{ZENODO_API}/deposit/depositions?status=draft&size=100",
         f"stop if any unpublished draft in concept {CONCEPT_RECID} exists"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{LATEST_ID}}", "must be published"),
        ("POST", f"{ZENODO_API}/deposit/depositions/{{LATEST_ID}}/actions/newversion",
         "the only POST this script can send; DRAFT_ID from links.latest_draft"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}",
         f"guard: conceptrecid {CONCEPT_RECID}, unsubmitted, a fresh copy of LATEST_ID (else stop, "
         "touching nothing); then write the receipt with status draft_created"),
        ("DELETE", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}/files/{{FILE_ID}}",
         "once per inherited file (the 7.48.1 files); allowed on DRAFT_ID only, and never with a "
         "FILE_ID of . or .. (requests would send /files/.. as a DELETE of the draft)"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}/files", "must return []"),
    ]
    for f in facts:
        plan.append(("PUT", f"{ZENODO_API}/files/{{DRAFT_BUCKET}}/{f['name']}",
                     f"body: {f['bytes']} bytes, sha256 {f['sha256']}; response md5 must be "
                     f"{f['md5']}"))
    plan += [
        ("PUT", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}",
         f"body: metadata.json, {len(body)} bytes, sha256 {hashlib.sha256(body).hexdigest()}"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}",
         "read back; title, version, date, type, access, language, licence (mit or mit-license), "
         "creators, keywords, related identifiers, description and notes must equal metadata.json, "
         "and no other key may be there but doi, prereserve_doi and imprint_publisher"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}/files",
         "exactly the three files; Zenodo md5 + filesize must equal local"),
    ]
    for i, (method, url, why) in enumerate(plan, 1):
        say(f"  {i:2}. {method:6} {url}")
        say(f"        {why}")
    say(f"  then: write {RECEIPT_PATH.name} (status draft_ready_unpublished) and STOP.")
    say(f"  No publish request exists in this plan or in this script; {PUBLISH_SCRIPT} sends it.")

    rule("METADATA THAT WOULD BE SENT (description shortened here; full text in metadata.json)")
    m = dict(doc["metadata"])
    d = m.pop("description")
    say(json.dumps(m, indent=2, ensure_ascii=False))
    say(f"  description: {len(d)} characters of HTML, sha256 "
        f"{hashlib.sha256(d.encode('utf-8')).hexdigest()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prepare (never publish) styxx v7.49.0 as the next Zenodo software version.")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline: no token, no network; validate files and metadata, print the plan")
    ap.add_argument("--token-file", help="file with a [ZENODO] section and a zenodo_token line")
    ap.add_argument("--resume-draft", type=int, metavar="DRAFT_ID",
                    help="take over this existing unpublished draft in concept 19758618")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="styxx checkout holding tag v7.49.0 (for the offline checks)")
    ap.add_argument("--skip-tree-check", action="store_true",
                    help="skip the blob-by-blob check of the source bundle (dry run only)")
    a = ap.parse_args(argv)

    if a.dry_run:
        forbid_network()

    rule("PREFLIGHT (offline)")
    c, facts, doc = preflight(a.repo, a.skip_tree_check)
    if a.dry_run:
        c.add(network_guard_holds(), "dry run: the socket guard refuses connections and name lookups")
    c.show()
    if c.failed or doc is None:
        say(f"\nREFUSED: {len(c.failed)} check(s) failed. Nothing was sent.")
        return 2

    if a.dry_run:
        print_plan(doc, facts)
        rule("DRY RUN: no token was read and no network call was made")
        passed = sum(1 for r in c.rows if r[0] == "PASS")
        say(f"  checks: {passed} passed, {len(c.skipped)} skipped, 0 failed")
        return 0

    if c.skipped:
        say("REFUSED: a check was skipped; the real run needs every check to pass "
            f"({[r[1] for r in c.skipped]})")
        return 2
    try:
        token, source = load_token(a.token_file)
    except Refuse as e:
        say(f"REFUSED: {e}")
        return 2
    _SECRETS.append(token)
    say(f"\ntoken: read from {source} (value not shown)")
    try:
        return run(token, source, doc, facts, a.resume_draft)
    except Refuse as e:
        say(f"REFUSED: {e}")
        return 1
    except Exception as e:  # noqa: BLE001 -- last line of defence: redact, never re-raise raw
        say(f"ERROR: {type(e).__name__}: {redact(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
