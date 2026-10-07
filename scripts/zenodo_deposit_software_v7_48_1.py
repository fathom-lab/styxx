"""Prepare styxx v7.48.1 as the next version of the styxx SOFTWARE concept record on Zenodo.

Concept record: 10.5281/zenodo.19758618. Its latest version on the lab's receipts is styxx 7.48.0,
10.5281/zenodo.23042251 (release/zenodo-deposit-receipt-software-v7.48.0.json, published
2026-09-29). Adapted from scripts/zenodo_deposit_software_v7_48_0.py.

WHAT IT DOES
------------
  1. Checks everything it can offline, and stops on any failure: the three files against their
     pinned sizes, sha256 and md5; the source bundle against the v7.48.1 tree blob by blob; the
     wheel and sdist contents against the tag's blobs; metadata.json against the 7.48.0 record as
     published and corrected, and every claim of its description against the tag (CHANGELOG.md's
     [7.48.1] section, the files it names, and the facts it states about them); the resource_type
     of the series and isNewVersionOf rows against DataCite's record (datacite-relations-7.48.1.json).
  2. Asks Zenodo which record is the LATEST version of concept 19758618, and proceeds only if it
     is 23042251 (7.48.0) or a later published version older than 7.48.1.
  3. Refuses if an unpublished draft already exists in that concept, unless --resume-draft names
     that draft explicitly.
  4. POSTs actions/newversion on the latest version, deletes the files the new draft inherited,
     uploads the three files, PUTs metadata.json, reads the draft back and compares every field and
     Zenodo's md5 and size for every file with the local bytes; any metadata key it did not send
     (other than doi, prereserve_doi and imprint_publisher, which Zenodo adds) stops it.
  5. Writes zenodo-draft-receipt-software-v7.48.1.json beside this file and STOPS.

WHAT IT WILL NOT DO
-------------------
It never publishes. Every request goes through an allowlist of method + path, a publish, edit or
discard action is not on it, and the mutating requests are further pinned to the one record they
may touch (newversion on the latest version only; DELETE and PUT on the new draft and its bucket
only, and no DELETE whose file id is . or .., which requests would send as a DELETE of the draft).
zenodo_publish_software_v7_48_1.py publishes, and only the draft this script's receipt names.

The token is read only from --token-file, in the lab's sectioned format: a [ZENODO] section with a
`zenodo_token: ...` line. It is sent only in an Authorization header, never in a URL, never
printed, never logged, never written to the receipt, and scrubbed from any error text. The file's
path is not printed or recorded either: a file that cannot be read (missing, locked, not UTF-8) is
a refusal that names the kind of error only.

USAGE
-----
    python zenodo_deposit_software_v7_48_1.py --dry-run
        Offline. Reads no token and makes no network call (a socket guard refuses any connection
        attempt). Validates the files and the metadata and prints every request the real run would
        send, with the token shown as <token>.

    python zenodo_deposit_software_v7_48_1.py --token-file PATH
        Creates the unpublished draft, fills it, verifies it, writes the receipt, stops.

    ... --resume-draft DRAFT_ID
        Take over an existing UNPUBLISHED draft in concept 19758618 (for example one left by an
        interrupted run): files that already match a local file by name, md5 and size are kept,
        every other file is deleted, the missing ones are uploaded and the metadata is overwritten.
        Only ever with an id the operator has looked at.
"""
from __future__ import annotations

import argparse
import datetime as _dt
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
PYPI_SOURCES_PATH = HERE / "pypi-sources-7.48.1.json"
RECEIPT_PATH = HERE / "zenodo-draft-receipt-software-v7.48.1.json"
PUBLISH_SCRIPT = "zenodo_publish_software_v7_48_1.py"
DEFAULT_REPO = "C:/Users/heyzo/clawd/styxx"

ZENODO_HOST = "zenodo.org"
ZENODO_API = f"https://{ZENODO_HOST}/api"
UA = "fathom-lab-styxx-research/1 (+https://github.com/fathom-lab/styxx)"

CONCEPT_RECID = "19758618"
CONCEPT_DOI = "10.5281/zenodo.19758618"
PREDECESSOR_RECID = 23042251            # styxx 7.48.0, release/zenodo-deposit-receipt-software-v7.48.0.json
PREDECESSOR_VERSION = "7.48.0"
PREDECESSOR_DOI = "10.5281/zenodo.23042251"
TAG = "v7.48.1"
TAG_COMMIT = "b42942186015848daa32e40b95970f4e370a3016"
BRANCH_POINT = "43b3b608"                # the commit the release branched from (CHANGELOG [7.48.1])
VERSION = "7.48.1"
RELEASE_DATE = "2026-10-06"
ADVISORY = "GHSA-h5xv-4344-f62r"
ADVISORY_URL = f"https://github.com/fathom-lab/styxx/security/advisories/{ADVISORY}"
BASE_METADATA = "release/zenodo-metadata-software-v7.48.0-edit-2026-09-29.json"
PREFIX = "styxx-7.48.1/"

# (filename, bytes, sha256, md5). The wheel and sdist sha256 are PyPI's; the bundle is
# `git -c core.autocrlf=false archive --format=zip --prefix=styxx-7.48.1/ v7.48.1`.
EXPECTED_FILES = [
    ("styxx-v7.48.1-source-bundle.zip", 275578927,
     "d50a7933fb668285fb21dfc34d2671cd2483b7d97e2068c2f4a20b07e140103c",
     "438ed1bd56ffb8cda4e95c82ec7fe20c"),
    ("styxx-7.48.1-py3-none-any.whl", 8043925,
     "a5d8a6b3ba1e0070cda1cb69c5452cd68f22f8ecec286af7157594b5a47b9a31",
     "52cf80c432493b64ddff70cf9f513cb5"),
    ("styxx-7.48.1.tar.gz", 8593254,
     "fde727642f4e6140c69d8c6805c810d26b5f6159deb7065c629999f74bce3da6",
     "e418930c1defb173dffbf37ba077db6a"),
]
SDIST_GENERATED = {"PKG-INFO", "setup.cfg", "styxx.egg-info/PKG-INFO", "styxx.egg-info/SOURCES.txt",
                   "styxx.egg-info/dependency_links.txt", "styxx.egg-info/entry_points.txt",
                   "styxx.egg-info/requires.txt", "styxx.egg-info/top_level.txt"}

# Every Zenodo DOI the record names, with the committed file at the tag that backs it.
DOI_EVIDENCE = {
    "10.5281/zenodo.19758618": "release/zenodo-deposit-receipt-software-v7.48.0.json",
    "10.5281/zenodo.23042251": "release/zenodo-deposit-receipt-software-v7.48.0.json",
    "10.5281/zenodo.19758619": "release/zenodo-deposit-receipt-software-v6.2.0.json",
    "10.5281/zenodo.19746215": "release/zenodo-deposit-receipt-spec-v1.0.json",
    "10.5281/zenodo.19326174": "release/zenodo-deposit-receipt-spec-v1.0.json",
    "10.5281/zenodo.19777921": "CITATION.cff",
}

# exactly the related identifiers this version carries: (relation, identifier, scheme, resource_type)
SERIES_DOI = "10.5281/zenodo.19326174"
EXPECTED_RELATED = [
    ("isSupplementTo", f"https://github.com/fathom-lab/styxx/releases/tag/{TAG}", "url", "software"),
    ("isDerivedFrom", f"https://github.com/fathom-lab/styxx/tree/{TAG_COMMIT}", "url", "software"),
    ("isSupplementTo", f"https://pypi.org/project/styxx/{VERSION}/", "url", "software"),
    ("isDocumentedBy", ADVISORY_URL, "url", None),
    ("isNewVersionOf", PREDECESSOR_DOI, "doi", "software"),
    ("isSupplementTo", "10.5281/zenodo.19746215", "doi", "publication-workingpaper"),
    ("isPartOf", SERIES_DOI, "doi", "publication-preprint"),
]
EXPECTED_SPEC_ROW = EXPECTED_RELATED[5]     # as 7.48.0 has it after its 2026-09-29 corrections
EXPECTED_SERIES_ROW = EXPECTED_RELATED[6]   # as 7.48.0 has it, but for resource_type (DataCite's)

# What DataCite records for the Zenodo DOIs whose resource_type comes from outside the tag
# (fetch_datacite_7481.py writes the receipt), and the Zenodo resource_type registered as each.
DATACITE_RECEIPT_PATH = HERE / "datacite-relations-7.48.1.json"
DATACITE_TO_ZENODO = {"Preprint": "publication-preprint", "Software": "software"}

# Keys Zenodo adds to a deposition's metadata on its own; any other key that metadata.json does
# not send (a community, contributor, reference, subject or grant added in the browser) stops a run.
# 7.48.0's draft, as-published and corrected metadata carry exactly the sent keys plus these.
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

# the lab's charter: words its public text never carries (checked, never printed)
CHARTER_FORBIDDEN = [
    "first", "novel", "revolutionary", "groundbreaking", "breakthrough", "tamper-proof",
    "self-verifying", "hallucination detector",
]

# Zenodo stores the licence "mit" as "mit-license" (7.48.0's draft receipt and provenance NOTE)
LICENSE_READBACK = {"mit": {"mit", "mit-license"}}


def vtuple(v: object) -> tuple[int, int, int] | None:
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", str(v or ""))
    return tuple(int(x) for x in m.groups()) if m else None  # type: ignore[return-value]


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
        c.add(pins == want, "wheel and sdist are the files PyPI lists for 7.48.1",
              f"{PYPI_SOURCES_PATH.name}: {sorted(p[0] for p in pins)}")
    except (OSError, KeyError, ValueError) as e:
        c.add(False, "wheel and sdist are the files PyPI lists for 7.48.1", f"{type(e).__name__}")
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
        c.add(None, "source bundle is the v7.48.1 tree", "repository not available here")
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
    c.add(not problems, "source bundle is the v7.48.1 tree",
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
                    version_ok = bool(re.search(rb"(?m)^Version: 7\.48\.1\r?$", z.read(info)))
                continue
            if n not in tree or git_blob_id(z.read(info)) != tree[n][1]:
                bad.append(n)
            else:
                same += 1
    c.add(not bad and same > 0 and version_ok,
          "every wheel file outside .dist-info equals the tag's blob; METADATA says 7.48.1",
          f"{same} files" if not bad else f"{len(bad)} differ, e.g. {bad[:3]}")
    same_s, bad_s, generated = 0, [], set()
    with tarfile.open(sdist) as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            top, _, rel = m.name.partition("/")
            if top != "styxx-7.48.1":
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


def changelog_section(repo: str) -> tuple[str, str]:
    changelog = git_text(repo, f"{TAG}:CHANGELOG.md")
    start = changelog.index("## [7.48.1]")
    end = changelog.index("\n## [7.48.0]", start)
    return changelog, changelog[start:end]


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
    c.add(m.get("version") == VERSION, "version 7.48.1", repr(m.get("version")))
    try:
        _dt.date.fromisoformat(m.get("publication_date", ""))
        date_ok = m.get("publication_date") == RELEASE_DATE
    except (TypeError, ValueError):
        date_ok = False
    c.add(date_ok, "publication_date 2026-10-06", repr(m.get("publication_date")))
    c.add(m.get("access_right") == "open", "access_right open", repr(m.get("access_right")))
    c.add("doi" not in m and "prereserve_doi" not in m,
          "no DOI set by hand (Zenodo mints the version DOI)")
    c.add(m.get("license") == "mit", "license sent as 'mit' (Zenodo stores it as mit-license)",
          repr(m.get("license")))

    title = m.get("title", "")
    c.add(title.startswith(f"styxx v{VERSION} — "), "title starts 'styxx v7.48.1 — '", title[:40])

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
    c.add(ADVISORY_URL in desc and ADVISORY in notes and ADVISORY in title + plain,
          "description links the advisory; notes name it")
    c.add(f"pip install -U styxx=={VERSION}" in plain, "description says how to upgrade")
    c.add(all(v in plain for v in ("7.47.0", "7.48.0")), "description names both affected releases")
    c.add(TAG_COMMIT in desc and TAG_COMMIT in notes, "description and notes name the tagged commit")
    try:
        up = sorted(v["upload_time_iso_8601"][:16] + "Z" for v in
                    json.loads(PYPI_SOURCES_PATH.read_text(encoding="utf-8"))["files"].values())
    except (OSError, KeyError, ValueError, TypeError):
        up = []
    c.add(bool(up) and f"PyPI records the upload at {up[0]}, UTC" in notes,
          "notes give PyPI's upload time as pypi-sources-7.48.1.json records it", str(up[:1]))
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
          "related identifiers: release, tree at b4294218, PyPI 7.48.1, the advisory, 7.48.0 "
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
    """The resource_type of the series row and the isNewVersionOf row is what DataCite records."""
    name = ("resource_type of 10.5281/zenodo.19326174 and 10.5281/zenodo.23042251 = what DataCite "
            f"records ({DATACITE_RECEIPT_PATH.name})")
    try:
        recs = json.loads(DATACITE_RECEIPT_PATH.read_text(encoding="utf-8"))["records"]
        series, prev = recs[SERIES_DOI], recs[PREDECESSOR_DOI]
    except (OSError, ValueError, KeyError, TypeError) as e:
        c.add(False, name, f"{DATACITE_RECEIPT_PATH.name} unreadable ({type(e).__name__})")
        return
    sent = {r.get("identifier"): r.get("resource_type") for r in rels if r.get("scheme") == "doi"}
    want = {d: DATACITE_TO_ZENODO.get(rec.get("resourceTypeGeneral"))
            for d, rec in ((SERIES_DOI, series), (PREDECESSOR_DOI, prev))}
    ok = (series.get("http_status") == 200 and prev.get("http_status") == 200
          and all(want.values()) and all(sent.get(d) == t for d, t in want.items())
          and prev.get("version") == PREDECESSOR_VERSION
          and "IsVersionOf" in (prev.get("relations_to_concept") or []))
    c.add(ok, name, f"DataCite: 19326174 {series.get('resourceTypeGeneral')}, 23042251 "
          f"{prev.get('resourceTypeGeneral')} {prev.get('version')}; sent "
          f"{sent.get(SERIES_DOI)}, {sent.get(PREDECESSOR_DOI)}")


def check_against_tag(c: Checks, m: dict, desc: str, plain: str, notes: str, files_at: int,
                      dois_named: set, repo: str, tree: dict) -> None:
    """Every statement of the description, held to the tag's own bytes."""
    changelog, section = changelog_section(repo)
    norm_section = re.sub(r"\s+", " ", section)
    head = re.search(r"^## \[7\.48\.1\] — 2026-10-06 — (.+)$", changelog, re.M)
    c.add(bool(head) and m.get("title") == f"styxx v{VERSION} — {head.group(1).strip()}",
          "title subtitle = the [7.48.1] headline at the tag")
    tag_msg = git(repo, "cat-file", "-p", TAG).stdout.decode("utf-8", "replace")
    c.add(ADVISORY in norm_section and ADVISORY in tag_msg,
          "the advisory id is the one the tag's CHANGELOG and tag message name")
    c.add("pip install -U styxx==7.48.1" in norm_section, "the upgrade command is the CHANGELOG's")

    # -- what 7.48.1 keeps from 7.48.0 as published and corrected --------------------------------
    base = json.loads(git_text(repo, f"{TAG}:{BASE_METADATA}"))
    c.add(m.get("creators") == base.get("creators"), "creators = 7.48.0 as published")
    c.add(m.get("keywords") == base.get("keywords"), "keywords = 7.48.0 as published")
    c.add(m.get("language") == base.get("language") and m.get("access_right") == base.get("access_right")
          and m.get("upload_type") == base.get("upload_type"),
          "language, access_right, upload_type = 7.48.0 as published")
    c.add(m.get("license") in {k for k, v in LICENSE_READBACK.items() if base.get("license") in v},
          "license = 7.48.0 as published", f"{m.get('license')!r} -> stored as {base.get('license')!r}")
    opening = re.match(r"(<p>This version continues the styxx software record .*?</p>)",
                       base.get("description", ""), re.S)
    c.add(bool(opening) and desc.startswith(opening.group(1) + "\n\n"),
          "opening paragraph (spec and research series) = 7.48.0 as corrected")
    base_rel = [(x.get("relation"), x.get("identifier"), x.get("scheme"), x.get("resource_type"))
                for x in base.get("related_identifiers", []) if x.get("scheme") == "doi"]
    c.add(len(base_rel) == 2 and base_rel[0] == EXPECTED_SPEC_ROW
          and base_rel[1][:3] == EXPECTED_SERIES_ROW[:3],
          "spec relation = 7.48.0 as corrected; series relation = 7.48.0's but for its "
          "resource_type, which is DataCite's", str(base_rel))

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
    c.add(re.search(r'(?m)^version: "7\.48\.1"$', cff) is not None
          and re.search(r'(?m)^date-released: "2026-10-06"$', cff) is not None,
          "CITATION.cff: version 7.48.1, date-released 2026-10-06")
    c.add(git_text(repo, f"{TAG}:LICENSE").startswith("MIT License"), "LICENSE at the tag is MIT")
    pyproject = git_text(repo, f"{TAG}:pyproject.toml")
    try:
        import tomllib
        py_kws = list(tomllib.loads(pyproject)["project"]["keywords"])
    except ModuleNotFoundError:
        blk = re.search(r"^keywords\s*=\s*\[(.*?)\]", pyproject, re.S | re.M)
        py_kws = re.findall(r'"([^"]+)"', blk.group(1)) if blk else []
    c.add(m.get("keywords") == py_kws, "keywords = pyproject.toml at the tag", f"{len(py_kws)} there")
    c.add('__version__ = "7.48.1"' in git_text(repo, f"{TAG}:styxx/_version.py"),
          "styxx/_version.py at the tag is 7.48.1")

    # -- DOIs ----------------------------------------------------------------------------------
    unbacked = [d for d in sorted(dois_named) if d in DOI_EVIDENCE
                and d not in git_text(repo, f"{TAG}:{DOI_EVIDENCE[d]}")]
    c.add(not unbacked, "each DOI named appears in the committed file that backs it", str(unbacked or ""))
    prev = json.loads(git_text(repo, f"{TAG}:release/zenodo-deposit-receipt-software-v7.48.0.json"))
    c.add(prev.get("software_doi") == PREDECESSOR_DOI and prev.get("version") == PREDECESSOR_VERSION
          and str(prev.get("concept_recid")) == CONCEPT_RECID and prev.get("deposit_id") == PREDECESSOR_RECID,
          "7.48.0 is 10.5281/zenodo.23042251 in concept 19758618 (its committed receipt)")
    c.add(re.search(r"^## \[7\.48\.0\] — 2026-09-25 — ", changelog, re.M) is not None,
          "7.48.0 was released 2026-09-25 (its CHANGELOG heading)")

    # -- every number, code span, commit id and pull-request number of the body ------------------
    open_end = desc.find("</p>") + 4
    body_html = desc[open_end:files_at]
    bp = parse_html(body_html)
    body_plain = re.sub(r"\s+", " ", "".join(bp.text))
    manifest = json.loads(git_text(repo, f"{TAG}:zenodo/MANIFEST.json"))
    defects = {d["id"]: d for d in manifest.get("citation_surface_defects", [])}
    concept_entry = next((d for d in manifest.get("deposits", []) if d.get("doi") == CONCEPT_DOI), {})
    corpus = " ".join([norm_section, json.dumps(manifest.get("citation_surface_defects"), ensure_ascii=False),
                       json.dumps(concept_entry, ensure_ascii=False), cff])
    nums = sorted(set(re.findall(NUM_RX, body_plain)))
    missing = [n for n in nums if not has_token(n, corpus)]
    c.add(not missing, "every number in the description body is in CHANGELOG [7.48.1] (or the "
          "manifest's defect list or its 19758618 entry, or CITATION.cff) at the tag",
          f"{len(nums)} numbers" if not missing else f"missing {missing}")
    dates = sorted(set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", body_plain)))
    c.add(all(d in corpus for d in dates), "every date in the description body is in that text",
          f"{len(dates)} dates" if all(d in corpus for d in dates)
          else f"missing {[d for d in dates if d not in corpus]}")

    # -- the sentence that says where each section comes from, held to what it says ------------
    # a section heading is a paragraph that is all <strong> (the release paragraph only opens with one)
    headings = [re.sub(r"<[^>]+>", "", h).strip()
                for h in re.findall(r"<p><strong>((?:(?!</p>).)*?)</strong></p>", desc[open_end:], re.S)]
    want_heads = ["Security: ", "Also in this release", "Known stale or wrong lines in the deposited tree",
                  "Not in this deposit", "Files in this record"]
    norm_desc = re.sub(r"\s+", " ", desc)
    c.add(len(headings) == len(want_heads) and all(h.startswith(w) for h, w in zip(headings, want_heads))
          and "The next two sections, the security repair and what else the release carries, are "
              "condensed from" in norm_desc
          and "The last two sections describe this deposit" in norm_desc,
          "the body's sections are the two the CHANGELOG sentence covers, the stale lines, and the "
          "two that describe the deposit, in that order", str(headings))
    sec_at = desc.find("<p><strong>Security: ")
    stale_at = desc.find("<p><strong>Known stale or wrong lines")
    if 0 < sec_at < stale_at:
        cp = parse_html(desc[sec_at:stale_at])
        cond_plain = re.sub(r"\s+", " ", "".join(cp.text))
        cond_nums = sorted(set(re.findall(NUM_RX, cond_plain)))
        cond_dates = sorted(set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", cond_plain)))
        miss = ([n for n in cond_nums if not has_token(n, norm_section)]
                + [d for d in cond_dates if d not in norm_section])
        c.add(not miss, "every number and date of the Security and Also-in-this-release sections is "
              "in CHANGELOG [7.48.1] alone", f"{len(cond_nums)} numbers, {len(cond_dates)} dates"
              if not miss else f"missing {miss}")
    else:
        c.add(False, "every number and date of the Security and Also-in-this-release sections is "
              "in CHANGELOG [7.48.1] alone", "sections not found")

    paths = set(tree)
    dirs = {q.rsplit("/", 1)[0] + "/" for q in paths if "/" in q}
    dirs |= {d for q in list(dirs) for d in ["/".join(q.split("/")[:i]) + "/"
                                              for i in range(1, q.count("/"))]}

    def code_ok(span: str) -> bool:
        s = re.sub(r"\s+", " ", html.unescape(span)).strip()
        if not s:
            return False
        if s == TAG or s in paths or s in dirs or any(q.endswith("/" + s) for q in paths):
            return True
        if re.fullmatch(r"[0-9a-f]{8,40}", s):
            return git(repo, "merge-base", "--is-ancestor", s, TAG, check=False).returncode == 0
        return s in norm_section

    bad_code = [s for s in bp.code if not code_ok(s)]
    c.add(not bad_code, "every <code> span of the body is a path in the tree, an ancestor commit, "
          "or text of CHANGELOG [7.48.1]", f"{len(bp.code)} spans" if not bad_code else f"{bad_code}")
    commits = sorted(set(re.findall(r"(?<![0-9a-z./])[0-9a-f]{8}(?:[0-9a-f]{32})?(?![0-9a-z])",
                                    desc + " " + notes)))
    not_anc = [x for x in commits
               if git(repo, "merge-base", "--is-ancestor", x, TAG, check=False).returncode != 0]
    c.add(not not_anc and TAG_COMMIT in commits,
          "every commit id named is the tag's commit or an ancestor of it", str(not_anc or f"{len(commits)} ids"))
    prs = sorted(set(re.findall(r"#\d+", body_plain)))
    c.add(all(x in norm_section for x in prs), "every #number of the body is in CHANGELOG [7.48.1]",
          str([x for x in prs if x not in norm_section] or prs))

    # -- the facts the description states about the tree ---------------------------------------
    p_parent = git(repo, "rev-parse", "76e9dcc5^").stdout.decode().strip()
    c.add(p_parent.startswith(BRANCH_POINT), "the capsule repair 76e9dcc5 branched from 43b3b608")
    cap47 = git_exists(repo, "v7.47.0:styxx/capsule.py") and \
        "def verify_capsule" in git_text(repo, "v7.47.0:styxx/capsule.py")
    cap48 = git_exists(repo, "v7.48.0:styxx/capsule.py") and \
        "def verify_capsule" in git_text(repo, "v7.48.0:styxx/capsule.py")
    ch47 = git_exists(repo, "v7.47.0:styxx/charon.py")
    ch48 = git_exists(repo, "v7.48.0:styxx/charon.py")
    c.add(cap47 and cap48 and ch48 and not ch47,
          "affected releases: verify_capsule is in 7.47.0 and 7.48.0, charon only in 7.48.0")
    ca_47 = git_exists(repo, "v7.47.0:styxx/corpus_audit.py")
    ca_47_history = ca_47 and "--history" in git_text(repo, "v7.47.0:styxx/corpus_audit.py")
    c.add(git_exists(repo, "v7.48.0:styxx/corpus_audit.py") and not ca_47_history,
          "corpus_audit's history re-derivation is in 7.48.0 and not in 7.47.0")
    added = git(repo, "log", "--diff-filter=A", "--format=%ad", "--date=short", TAG, "--",
                "styxx/capsule.py").stdout.decode().split()
    c.add(bool(added) and added[-1] == "2026-08-31", "styxx/capsule.py has been in the tree since 2026-08-31",
          str(added[-1:] or ""))
    charon = git_text(repo, f"{TAG}:styxx/charon.py")
    audit = git_text(repo, f"{TAG}:styxx/corpus_audit.py")
    c.add("unsafe_embedded_name" in charon, "charon records live_error unsafe_embedded_name")
    c.add("stands_reason" in audit and re.search(r'"--history".*"off"', audit) is not None,
          "corpus_audit says why in stands_reason and takes --history off")
    caps = [q for q in paths if q.startswith("papers/") and q.endswith(".capsule.html")]
    caps_v01 = [q for q in caps if b"styxx-oath/capsule/v0.1" in git_show(repo, f"{TAG}:{q}")]
    c.add(len(caps_v01) == 10, "ten v0.1 capsules are committed under papers/",
          f"{len(caps_v01)} v0.1 among {len(caps)} capsule files")
    c.add("--island-z" in git_text(repo, f"{TAG}:styxx/islands.py"), "styxx.islands takes --island-z")
    pub_yml = git_text(repo, f"{TAG}:.github/workflows/publish.yml")
    security = git_text(repo, f"{TAG}:SECURITY.md")
    c.add(re.search(r"(?m)^\s+attestations: false\s*$", pub_yml) is not None
          and "attestations: false" in security and "SHA-256" in security,
          "publish.yml sets attestations: false and SECURITY.md says to check by SHA-256")
    idx = json.loads(git_text(repo, f"{TAG}:conformance/sworn/index.json"))
    c.add(idx.get("vector_count") == 3620 and idx.get("family_count") == 20
          and (idx.get("blobs") or {}).get("count") == 3981
          and (idx.get("provenance") or {}).get("styxx_version") == VERSION,
          "conformance/sworn at the tag: 3620 vectors, 20 families, 3981 blobs, stamped 7.48.1")
    errata_files = ["styxx/__init__.py", "styxx/forecast.py", "styxx/intercept.py", "styxx/critique.py",
                    "styxx/hallucination.py", "styxx/attack/universal_suffixes_v0.json",
                    "styxx/adapters/guardrails.py", "styxx/admissibility.py"]
    unchanged = git(repo, "diff", "--quiet", "v7.48.0", TAG, "--", *errata_files, check=False).returncode == 0
    readme = git_text(repo, f"{TAG}:README.md")
    readme_old = git_text(repo, "v7.48.0:README.md")
    c.add(unchanged and all(f in paths for f in errata_files)
          and readme.splitlines()[421] == readme_old.splitlines()[421],
          "the files the [7.48.0] errata name, and README line 422, did not change in 7.48.1")
    pinned = re.findall(r"(?:github\.com/fathom-lab/styxx/(?:blob|tree|raw)|"
                        r"raw\.githubusercontent\.com/fathom-lab/styxx)/(v\d+\.\d+\.\d+)/", readme)
    c.add(bool(pinned) and set(pinned) == {"v7.48.0"},
          "README's tag-pinned links all point at the v7.48.0 tag",
          f"{len(pinned)} tag-pinned links, tags {sorted(set(pinned))}")
    m_concept = (concept_entry.get("notes") or "")
    c.add("Latest version: 7.48.0, 10.5281/zenodo.23042251" in m_concept,
          "zenodo/MANIFEST.json's 19758618 entry names 7.48.0 (23042251) as the latest version")
    c.add("| DOI (concept, always-latest) | [10.5281/zenodo.19326174]" in readme
          and "software concept DOI [10.5281/zenodo.19758618]" in readme,
          "README labels 19326174 always-latest; its citation row names 19758618")
    c.add(all(k in defects and "resolved" not in defects[k] for k in ("D2", "D3", "D4", "D5")),
          "D2, D3, D4 and D5 are open in zenodo/MANIFEST.json at the tag")
    c.add('value: "10.5281/zenodo.19758618"' in cff and 'doi: "10.5281/zenodo.19777921"' in
          cff.split("preferred-citation:", 1)[-1].split("\n\n", 1)[0],
          "CITATION.cff names the concept DOI; its preferred-citation is 10.5281/zenodo.19777921")
    c.add("2026-06-21" in defects.get("D2", {}).get("defect", ""),
          "D2 says the position paper predates its 2026-06-21 scope erratum")
    c.add("Until 7.48.0 ships" in git_text(repo, f"{TAG}:web/gate/README.md")
          and "until 7.48.0 ships" in git_text(repo, f"{TAG}:web/gate/differential/py_side.py"),
          "web/gate/README.md and py_side.py still say 7.48.0 has not shipped")
    # the EXTERNAL-1 line, held to CHANGELOG [7.48.1], the builder and the committed packet
    cmf = "papers/closed-model-frontier"
    packet_py = git_text(repo, f"{TAG}:{cmf}/external1_packet.py")
    packet = json.loads(git_text(repo, f"{TAG}:{cmf}/external1_packet.json"))
    zz_ids = sorted(str(it.get("id")) for it in packet.get("items", [])
                    if str((it.get("claim_detail") or {}).get("path") or "")
                    .rsplit("/", 1)[-1].startswith("zz_"))
    ext1 = {
        "description": all(s in body_plain for s in (
            "The #125 repair makes the blind-packet builder, "
            f"{cmf}/external1_packet.py, number items by their shuffled position by default.",
            "Its build --as-published mode reproduces the pre-repair bytes on purpose, leak included,",
            "shown only on a synthetic ledger and shelf, not run on the real ones.",
            "The committed packet still carries the id leak, and whether any adjudicator used it "
            "cannot be re-tested.")),
        "CHANGELOG": ("`build` now numbers items by their shuffled position" in norm_section
                      and "`build --as-published` writes, on a synthetic ledger and shelf, exactly "
                          "the bytes the pre-repair builder wrote (arm-ordered numbering, key and "
                          "digest, leak included)" in norm_section
                      and "It has not been run on the real shelf and ledger" in norm_section
                      and "the committed EXTERNAL-1 packet still carries the id leak, and whether "
                          "any adjudicator used it cannot be re-tested" in norm_section),
        "builder": ("def build(as_published: bool = False)" in packet_py
                    and 'iid = f"E1-{(arm_pos if as_published else pos):03d}"' in packet_py
                    and "leak included" in packet_py),
        "committed packet": zz_ids == [f"E1-{i:03d}" for i in range(115, 130)],
    }
    c.add(all(ext1.values()),
          "EXTERNAL-1 line: the builder numbers by shuffled position by default, build "
          "--as-published writes the leak on purpose, and the committed packet still carries it "
          "(CHANGELOG [7.48.1], external1_packet.py and external1_packet.json at the tag)",
          "the 15 zz_ items are E1-115..E1-129" if all(ext1.values())
          else f"not as stated: {[k for k, v in ext1.items() if not v]}")
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
        c.add(None, "source bundle is the v7.48.1 tree", "--skip-tree-check")
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
        "publish_step": f"{PUBLISH_SCRIPT}, run by the operator after reading the draft; it "
                        "publishes only the draft_id below, and only if this status is "
                        "draft_ready_unpublished",
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
    if latest_id == PREDECESSOR_RECID:
        if latest_version != PREDECESSOR_VERSION:
            return fail("resolve_latest", f"record {latest_id} reports version {latest_version!r}, "
                        f"not {PREDECESSOR_VERSION}")
        say(f"  the latest version is {PREDECESSOR_RECID} (7.48.0), as the lab's receipts say")
    elif latest_id > PREDECESSOR_RECID:
        lv = vtuple(latest_version)
        if lv is None or lv >= vtuple(VERSION):  # type: ignore[operator]
            return fail("resolve_latest", f"the latest version, record {latest_id}, is "
                        f"{latest_version!r}: not older than {VERSION}, so {VERSION} would not "
                        "follow it. Decide by hand")
        say(f"  note: the latest version is {latest_id} ({latest_version}), published after "
            f"{PREDECESSOR_RECID}; the lab's receipts do not record it. Proceeding from it.")
    else:
        return fail("resolve_latest", f"the latest version is record {latest_id}, older than "
                    f"{PREDECESSOR_RECID} (7.48.0, which the lab's receipts record as published "
                    "in this concept). Something is not as recorded; decide by hand")

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
         f"LATEST_ID / LATEST_VERSION; stop unless conceptrecid is {CONCEPT_RECID} and LATEST_ID is "
         f"{PREDECESSOR_RECID} (7.48.0) or a later published version older than {VERSION}"),
        ("GET", f"{ZENODO_API}/deposit/depositions?status=draft&size=100",
         f"stop if any unpublished draft in concept {CONCEPT_RECID} exists"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{LATEST_ID}}", "must be published"),
        ("POST", f"{ZENODO_API}/deposit/depositions/{{LATEST_ID}}/actions/newversion",
         "the only POST this script can send; DRAFT_ID from links.latest_draft"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}",
         f"guard: conceptrecid {CONCEPT_RECID}, unsubmitted, a fresh copy of LATEST_ID (else stop, "
         "touching nothing); then write the receipt with status draft_created"),
        ("DELETE", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}/files/{{FILE_ID}}",
         "once per inherited file (the 7.48.0 files); allowed on DRAFT_ID only, and never with a "
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
        description="Prepare (never publish) styxx v7.48.1 as the next Zenodo software version.")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline: no token, no network; validate files and metadata, print the plan")
    ap.add_argument("--token-file", help="file with a [ZENODO] section and a zenodo_token line")
    ap.add_argument("--resume-draft", type=int, metavar="DRAFT_ID",
                    help="take over this existing unpublished draft in concept 19758618")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="styxx checkout holding tag v7.48.1 (for the offline checks)")
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
