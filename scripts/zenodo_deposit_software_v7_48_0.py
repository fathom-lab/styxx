"""Prepare styxx v7.48.0 as a new version of the styxx SOFTWARE concept record on Zenodo.

Concept record: 10.5281/zenodo.19758618 (evidence: release/zenodo-deposit-receipt-software-v6.2.0.json,
which records concept_doi 10.5281/zenodo.19758618 for the v6.2.0 deposit 19758619).

WHAT IT DOES
------------
  1. Checks everything it can offline: the three files against their pinned sizes and sha256,
     the source bundle against the v7.48.0 tree blob by blob, and metadata.json against the
     tagged CHANGELOG, CITATION.cff, LICENSE and pyproject.toml. Any failure stops it here.
  2. Asks Zenodo which record is the LATEST version of concept 19758618. It does not assume
     19758619 is still the latest.
  3. Refuses if an unpublished draft already exists in that concept (the operator's work is
     never overwritten), unless --resume-draft names that draft explicitly.
  4. POSTs actions/newversion on the latest version, deletes the files the new draft inherited,
     uploads the three files, PUTs metadata.json, reads the draft back and compares Zenodo's md5
     and size for every file with the local bytes.
  5. Writes zenodo-draft-receipt-v7.48.0.json beside this file and STOPS.

WHAT IT WILL NOT DO
-------------------
It never publishes. There is no publish request anywhere in this file and no flag that adds one:
every request goes through an allowlist of method + path, and a publish, edit or discard action is
not on it. Publishing mints a permanent DOI under a real name; the operator reads the draft in the
browser and presses Publish, or discards it.

The token is read from the ZENODO_TOKEN environment variable, or from the file given as
--token-file (either one bare token line, or the lab's sectioned format: a [ZENODO] section with a
`zenodo_token: ...` line). It is sent only in an Authorization header, never in a URL, never
printed, never logged, never written to the receipt, and scrubbed from any error text.

USAGE
-----
    python zenodo_deposit_software_v7_48_0.py --dry-run
        Offline. Reads no token, makes no network call. Validates files and metadata and prints
        every request the real run would send, with the token shown as <token>.

    ZENODO_TOKEN=... python zenodo_deposit_software_v7_48_0.py
    python zenodo_deposit_software_v7_48_0.py --token-file PATH
        Creates the unpublished draft, fills it, verifies it, writes the receipt, stops.

    ... --resume-draft DRAFT_ID
        Take over an existing UNPUBLISHED draft in concept 19758618 (for example one left behind
        by an interrupted run): its files are deleted and replaced, its metadata overwritten.
        Only ever with an id the operator has looked at.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
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
RECEIPT_PATH = HERE / "zenodo-draft-receipt-v7.48.0.json"
DEFAULT_REPO = "C:/Users/heyzo/clawd/styxx"

ZENODO_HOST = "zenodo.org"
ZENODO_API = f"https://{ZENODO_HOST}/api"
UA = "fathom-lab-styxx-research/1 (+https://github.com/fathom-lab/styxx)"

CONCEPT_RECID = "19758618"
CONCEPT_DOI = "10.5281/zenodo.19758618"
TAG = "v7.48.0"
TAG_COMMIT = "1218dbad4fca61b01179f3af2b83dbea67d34f82"
VERSION = "7.48.0"
RELEASE_DATE = "2026-09-25"

# (filename, bytes, sha256, md5). The wheel and sdist sha256 are PyPI's; the bundle is
# `git -c core.autocrlf=false archive --format=zip --prefix=styxx-7.48.0/ v7.48.0`.
EXPECTED_FILES = [
    ("styxx-v7.48.0-source-bundle.zip", 275463019,
     "4e886abe45691e56d1046ad41fa1f9d1cb59ab4594366e30a6faa2dcd90d1336",
     "1145aff5e0fe5f532ee2f7e8ab01f9fe"),
    ("styxx-7.48.0-py3-none-any.whl", 8040992,
     "624e0715e1f7e23dc17fd2aabf21b942caebec8f094d2a620f8f29e8811be1f0",
     "4304824ee5e3f0335586ba82a1ddd771"),
    ("styxx-7.48.0.tar.gz", 8576261,
     "48208cdbe60dd80285110b2d845f52c52fa0c653cf1bb6115560ae937acb7b0d",
     "0fc7b2fcf57762d633d6fffc1415939b"),
]

# DOIs a related identifier may carry: only ones a committed receipt backs. Anything else is
# an invented identifier and is refused.
RECEIPT_BACKED_DOIS = {"10.5281/zenodo.19758618", "10.5281/zenodo.19758619",
                       # release/zenodo-deposit-receipt-spec-v1.0.json: the spec version DOI and its concept
                       "10.5281/zenodo.19746215", "10.5281/zenodo.19326174"}

# Zenodo's documented relation types for related_identifiers (legacy deposit API).
ZENODO_RELATIONS = {
    "isCitedBy", "cites", "isSupplementTo", "isSupplementedBy", "isContinuedBy", "continues",
    "isDescribedBy", "describes", "hasMetadata", "isMetadataFor", "isNewVersionOf",
    "isPreviousVersionOf", "isPartOf", "hasPart", "isReferencedBy", "references",
    "isDocumentedBy", "documents", "isCompiledBy", "compiles", "isVariantFormOf",
    "isOriginalFormof", "isIdenticalTo", "isAlternateIdentifier", "isReviewedBy", "reviews",
    "isDerivedFrom", "isSourceOf", "requires", "isRequiredBy", "isObsoletedBy", "obsoletes",
}

CHARTER_FORBIDDEN = [
    "first", "novel", "revolutionary", "groundbreaking", "breakthrough", "tamper-proof",
    "self-verifying", "hallucination detector",
]
# markers of content that is not released and must not appear in the record
EXCLUDED_MARKERS = [
    "third-party-precision", "rivals-on-adjudicated", "landscape", "outreach", "#161", "#165",
]

# method + path patterns this script may send. Nothing else leaves the process.
ALLOWED_REQUESTS = [
    ("GET", re.compile(rf"^/api/records/{CONCEPT_RECID}/versions/latest$")),
    ("GET", re.compile(rf"^/api/records/{CONCEPT_RECID}$")),
    ("GET", re.compile(r"^/api/deposit/depositions$")),
    ("GET", re.compile(r"^/api/deposit/depositions/\d+$")),
    ("GET", re.compile(r"^/api/deposit/depositions/\d+/files$")),
    ("POST", re.compile(r"^/api/deposit/depositions/\d+/actions/newversion$")),
    ("DELETE", re.compile(r"^/api/deposit/depositions/\d+/files/[A-Za-z0-9._-]+$")),
    ("PUT", re.compile(r"^/api/deposit/depositions/\d+$")),
    ("PUT", re.compile(r"^/api/files/[0-9a-fA-F-]{32,36}/[A-Za-z0-9._+-]+$")),
]

# ---------------------------------------------------------------------------
# output and secrets
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
    print(redact(" ".join(str(p) for p in parts)))


def rule(title: str) -> None:
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


class Refuse(Exception):
    """A check failed; the message says which. Never carries the token."""


def load_token(token_file: str | None) -> tuple[str, str]:
    if token_file:
        p = Path(token_file)
        if not p.is_file():
            raise Refuse(f"--token-file {p} does not exist or is not a file")
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8-sig").splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("#")]
        if len(lines) == 1 and not re.search(r"[\s:=\[\]]", lines[0]):
            return lines[0], f"--token-file {p} (bare token line)"
        in_zenodo = False
        for s in lines:
            if s.startswith("[") and s.endswith("]"):
                in_zenodo = s.upper() == "[ZENODO]"
                continue
            if in_zenodo and s.lower().startswith("zenodo_token"):
                parts = re.split(r"[:=]", s, maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    return parts[1].strip(), f"--token-file {p} ([ZENODO] zenodo_token)"
        raise Refuse(f"no token found in {p}: expected one bare token line, or a [ZENODO] "
                     "section with a 'zenodo_token: ...' line (file contents not shown)")
    tok = os.environ.get("ZENODO_TOKEN", "").strip()
    if not tok:
        raise Refuse("no token: set ZENODO_TOKEN or pass --token-file PATH")
    return tok, "ZENODO_TOKEN environment variable"


# ---------------------------------------------------------------------------
# offline checks
# ---------------------------------------------------------------------------

def hash_file(path: Path) -> tuple[int, str, str]:
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
            n += len(chunk)
    return n, sha.hexdigest(), md5.hexdigest()


def git_show(repo: str, spec: str) -> bytes:
    return subprocess.run(["git", "-C", repo, "show", spec],
                          check=True, capture_output=True, timeout=60).stdout


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
    return facts


def git_blob_id(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def check_bundle_tree(c: Checks, repo: str | None) -> None:
    if not repo:
        c.add(None, "source bundle is the v7.48.0 tree", "repository not available here")
        return
    out = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "-z", TAG],
                         check=True, capture_output=True, timeout=120).stdout
    tree = {}
    for rec in out.split(b"\0"):
        if rec:
            meta, path = rec.split(b"\t", 1)
            _mode, kind, oid = meta.decode().split()
            tree[path.decode("utf-8")] = (kind, oid)
    problems, seen = [], set()
    with zipfile.ZipFile(BUNDLE / EXPECTED_FILES[0][0]) as z:
        if z.comment.decode("ascii", "replace").strip() != TAG_COMMIT:
            problems.append("zip comment is not the tagged commit id")
        for info in z.infolist():
            if info.filename.endswith("/"):
                continue
            if not info.filename.startswith("styxx-7.48.0/"):
                problems.append(f"entry outside prefix: {info.filename}")
                continue
            rel = info.filename[len("styxx-7.48.0/"):]
            if rel not in tree or tree[rel][0] != "blob":
                problems.append(f"not a blob in the tree: {rel}")
                continue
            seen.add(rel)
            if git_blob_id(z.read(info)) != tree[rel][1]:
                problems.append(f"bytes differ from the blob: {rel}")
    problems += [f"missing: {m}" for m in sorted(set(tree) - seen)]
    c.add(not problems, "source bundle is the v7.48.0 tree",
          f"{len(seen)} of {len(tree)} blobs byte-identical" if not problems
          else f"{len(problems)} problem(s), e.g. {problems[:3]}")


class _TagBalance(HTMLParser):
    VOID = {"br", "hr", "img", "meta", "link", "input", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.stack[-1] if self.stack else 'nothing'}")
        else:
            self.stack.pop()

    def handle_data(self, data):
        self.text.append(data)


def word_hits(text: str, word: str) -> int:
    return len(re.findall(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text.lower()))


def check_metadata(c: Checks, facts: list[dict], repo: str | None) -> dict | None:
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
    c.add(m.get("version") == VERSION, "version 7.48.0", repr(m.get("version")))
    try:
        _dt.date.fromisoformat(m.get("publication_date", ""))
        date_ok = m.get("publication_date") == RELEASE_DATE
    except (TypeError, ValueError):
        date_ok = False
    c.add(date_ok, "publication_date 2026-09-25", repr(m.get("publication_date")))
    c.add(m.get("access_right") == "open", "access_right open", repr(m.get("access_right")))
    c.add("doi" not in m and "prereserve_doi" not in m,
          "no DOI set by hand (Zenodo mints the version DOI)")

    title = m.get("title", "")
    c.add(title.startswith(f"styxx v{VERSION} — "), "title starts 'styxx v7.48.0 — '", title[:40])

    kws = m.get("keywords", [])
    c.add(bool(kws) and all(isinstance(k, str) and k for k in kws), "keywords present",
          f"{len(kws)} keywords")
    c.add(not any("hallucination" in k.lower() for k in kws),
          "keywords omit the one pyproject dropped at the tag")

    desc = m.get("description", "")
    p = _TagBalance()
    p.feed(desc)
    p.close()
    c.add(not p.errors and not p.stack, "description HTML tags balance",
          "; ".join(p.errors[:3]) + (f" unclosed {p.stack}" if p.stack else ""))
    plain = re.sub(r"\s+", " ", "".join(p.text))
    for f in facts:
        ok = f["sha256"] in desc and f["md5"] in desc and str(f["bytes"]) in plain
        c.add(ok, f"description carries bytes, sha256 and md5 of {f['name']}")
    c.add(CONCEPT_DOI in desc, "description names the concept DOI")
    for label in ("INVALID", "WITHDRAWN", "ABANDONED", "Erratum"):
        c.add(label in plain, f"description names {label} results as such")
    c.add("owed" in plain and "Staged for 7.48.0" in plain,
          "description carries the erratum on the staged 'owed' note")

    public_text = " ".join([title, plain, m.get("notes", ""), " ".join(kws)])
    hits = {w: word_hits(public_text, w) for w in CHARTER_FORBIDDEN}
    hits = {w: n for w, n in hits.items() if n}
    c.add(not hits, "charter: no forbidden word in title/description/notes/keywords", str(hits or ""))
    ex = [w for w in EXCLUDED_MARKERS if w.lower() in public_text.lower()]
    c.add(not ex, "no unreleased-content marker in the public text", str(ex or ""))

    creators = m.get("creators", [])
    c.add(bool(creators) and all(re.match(r"^[^,]+, [^,]+$", cr.get("name", "")) for cr in creators),
          "creators present, 'Family, Given'", str([cr.get("name") for cr in creators]))

    rels = m.get("related_identifiers", [])
    bad_rel = [r for r in rels if r.get("relation") not in ZENODO_RELATIONS
               or not r.get("identifier") or r.get("scheme") not in {"url", "doi"}]
    c.add(not bad_rel, "related_identifiers use Zenodo relation types", str(bad_rel or ""))
    dois = [r["identifier"] for r in rels if "10.5281/zenodo." in r.get("identifier", "")]
    invented = [d for d in dois if d.split("doi.org/")[-1] not in RECEIPT_BACKED_DOIS]
    c.add(not invented, "no Zenodo DOI in related_identifiers without a receipt", str(invented or ""))
    by = {(r.get("relation"), r.get("identifier")) for r in rels}
    c.add(("isSupplementTo", "https://github.com/fathom-lab/styxx/releases/tag/v7.48.0") in by,
          "GitHub release, isSupplementTo")
    c.add(any(rel in ("isDerivedFrom", "isIdenticalTo") and TAG_COMMIT in ident
              for rel, ident in by), "tagged source tree at the tag's commit")
    c.add(any(ident == "https://pypi.org/project/styxx/7.48.0/" for _rel, ident in by),
          "PyPI 7.48.0 release page")

    c.add(m.get("license") == "mit", "license mit", repr(m.get("license")))

    if not repo:
        for name in ("title subtitle = release headline", "creators = CITATION.cff",
                     "license = LICENSE", "keywords = pyproject.toml at the tag",
                     "every number of the release-notes summary is in the description"):
            c.add(None, name, "repository not available here")
        return doc

    changelog = git_show(repo, f"{TAG}:CHANGELOG.md").decode("utf-8")
    head = re.search(r"^## \[7\.48\.0\] — 2026-09-25 — (.+)$", changelog, re.M)
    c.add(bool(head) and title == f"styxx v{VERSION} — {head.group(1).strip()}",
          "title subtitle = release headline at the tag")

    cff = git_show(repo, f"{TAG}:CITATION.cff").decode("utf-8")
    authors_block = cff.split("authors:", 1)[1].split("\n\n", 1)[0] if "authors:" in cff else ""
    fam = re.findall(r"family-names:\s*\"?([^\"\n]+)", authors_block)
    giv = re.findall(r"given-names:\s*\"?([^\"\n]+)", authors_block)
    aff = re.findall(r"affiliation:\s*\"?([^\"\n]+)", authors_block)
    orc = re.findall(r"orcid:\s*\"?([^\"\n]+)", authors_block)
    want = [f"{f.strip()}, {g.strip()}" for f, g in zip(fam, giv)]
    got = [cr.get("name") for cr in creators]
    c.add(got == want, "creators = CITATION.cff authors", f"{got} vs {want}")
    c.add([cr.get("affiliation") for cr in creators] == [a.strip() for a in aff],
          "affiliations = CITATION.cff")
    cff_orcids = [o.strip().rsplit("/", 1)[-1] for o in orc]
    sent_orcids = [cr["orcid"] for cr in creators if cr.get("orcid")]
    c.add(sent_orcids == cff_orcids, "ORCIDs = CITATION.cff (none recorded there, none sent)"
          if not cff_orcids else "ORCIDs = CITATION.cff", str(sent_orcids))

    lic = git_show(repo, f"{TAG}:LICENSE").decode("utf-8")
    c.add(lic.startswith("MIT License"), "LICENSE at the tag is MIT")

    pyproject = git_show(repo, f"{TAG}:pyproject.toml").decode("utf-8")
    try:
        import tomllib
        py_kws = list(tomllib.loads(pyproject)["project"]["keywords"])
    except ModuleNotFoundError:
        blk = re.search(r"^keywords\s*=\s*\[(.*?)\]", pyproject, re.S | re.M)
        py_kws = re.findall(r'"([^"]+)"', blk.group(1)) if blk else []
    c.add(kws == py_kws, "keywords = pyproject.toml at the tag", f"{len(py_kws)} there")

    section = changelog.split("## [7.48.0]", 1)[1]
    summary = section.split("\n### ", 1)[0]
    nums = {t.rstrip(".,") for t in re.findall(r"\d[\d,.]*\d%?|\d%?", summary)}
    missing = sorted(n for n in nums if n not in plain)
    c.add(not missing, "every number of the release-notes summary is in the description",
          f"{len(nums)} numbers" if not missing else f"missing {missing}")
    return doc


def guard_selftest(c: Checks) -> None:
    refused = []
    for action in ("publish", "edit", "discard"):
        try:
            check_allowed("POST", f"{ZENODO_API}/deposit/depositions/1/actions/{action}")
        except Refuse:
            refused.append(action)
    c.add(refused == ["publish", "edit", "discard"],
          "allowlist refuses POST .../actions/{publish,edit,discard}", str(refused))
    try:
        check_allowed("PUT", "https://example.org/api/files/x/y")
        c.add(False, "allowlist refuses a host other than zenodo.org")
    except Refuse:
        c.add(True, "allowlist refuses a host other than zenodo.org")


def preflight(repo_arg: str, skip_tree: bool) -> tuple[Checks, list[dict], dict | None]:
    c = Checks()
    repo = repo_arg if repo_available(repo_arg) else None
    c.add(True if repo else None, f"repository at {repo_arg} holds {TAG} -> {TAG_COMMIT[:12]}",
          "" if repo else "not found; repository checks are skipped")
    facts = check_files(c)
    if skip_tree:
        c.add(None, "source bundle is the v7.48.0 tree", "--skip-tree-check")
    elif not c.failed:
        check_bundle_tree(c, repo)
    doc = check_metadata(c, facts, repo) if len(facts) == len(EXPECTED_FILES) else None
    guard_selftest(c)
    return c, facts, doc


# ---------------------------------------------------------------------------
# the network path (never publishes)
# ---------------------------------------------------------------------------

def check_allowed(method: str, url: str) -> None:
    u = urlparse(url)
    if u.scheme != "https" or u.hostname != ZENODO_HOST:
        raise Refuse(f"request to {u.scheme}://{u.hostname} is not allowed")
    if not any(m == method and rx.match(u.path) for m, rx in ALLOWED_REQUESTS):
        raise Refuse(f"{method} {u.path} is not on this script's allowlist")


class Zenodo:
    def __init__(self, token: str) -> None:
        try:
            import requests
        except ImportError as e:
            raise Refuse("the 'requests' package is required for the real run "
                         "(pip install requests)") from e
        self._requests = requests
        self._s = requests.Session()
        self._token = token
        self.log: list[dict] = []

    def call(self, method: str, url: str, *, params=None, json_body=None, data=None,
             timeout=(30, 60), expect=(200,)):
        check_allowed(method, url)
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
            raise Refuse(f"{method} {urlparse(url).path}: {type(e).__name__}: {redact(e)}") from None
        if urlparse(r.url).hostname != ZENODO_HOST:
            raise Refuse(f"{method} {urlparse(url).path} was redirected off {ZENODO_HOST}")
        self.log.append({"method": method, "path": urlparse(url).path, "status": r.status_code,
                         "seconds": round(time.time() - t0, 2)})
        if r.status_code not in expect:
            raise Refuse(f"{method} {urlparse(url).path} -> HTTP {r.status_code}: "
                         f"{redact(r.text)[:1500]}")
        return r


def md5_of(checksum: str | None) -> str | None:
    if not checksum:
        return None
    return checksum.split(":", 1)[1] if checksum.startswith("md5:") else checksum


def write_receipt(receipt: dict) -> None:
    receipt["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    blob = redact(text).encode("utf-8")
    RECEIPT_PATH.write_bytes(blob)
    if RECEIPT_PATH.read_bytes() != blob:
        raise Refuse(f"receipt {RECEIPT_PATH} did not read back identical")
    say(f"  receipt written: {RECEIPT_PATH} (status {receipt['status']})")


def run(token: str, token_source: str, doc: dict, facts: list[dict],
        resume_draft: int | None) -> int:
    z = Zenodo(token)
    receipt: dict = {
        "schema": "fathom/zenodo-draft-receipt/v1",
        "status": "started",
        "published": False,
        "publish_step": "the operator, in the browser, after reading the draft",
        "concept_recid": CONCEPT_RECID,
        "concept_doi": CONCEPT_DOI,
        "version": VERSION,
        "token_source": token_source,
        "metadata_json_sha256": hashlib.sha256(METADATA_PATH.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
            say("  Finish it by hand (MANUAL_UPLOAD.md), re-run with --resume-draft "
                f"{receipt['draft_id']}, or discard it in the Zenodo UI. Nothing was published.")
        return 1

    rule(f"1. latest version of concept {CONCEPT_RECID} (resolved, not assumed)")
    try:
        try:
            rec = z.call("GET", f"{ZENODO_API}/records/{CONCEPT_RECID}/versions/latest").json()
        except Refuse as e:
            # the concept id itself redirects to the latest version on Zenodo
            say(f"  versions/latest failed ({e}); asking the concept record directly")
            rec = z.call("GET", f"{ZENODO_API}/records/{CONCEPT_RECID}").json()
    except (Refuse, ValueError) as e:
        return fail("resolve_latest", str(e))
    if str(rec.get("conceptrecid")) != CONCEPT_RECID:
        return fail("resolve_latest", f"record {rec.get('id')} has conceptrecid "
                    f"{rec.get('conceptrecid')}, not {CONCEPT_RECID}")
    latest_id = int(rec["id"])
    latest_version = (rec.get("metadata") or {}).get("version")
    latest_files = {(f.get("key") or f.get("filename"), md5_of(f.get("checksum")))
                    for f in rec.get("files") or []}
    receipt["predecessor"] = {"id": latest_id, "doi": rec.get("doi"), "version": latest_version,
                              "title": (rec.get("metadata") or {}).get("title"),
                              "files": sorted(k for k, _ in latest_files if k)}
    say(f"  latest: id {latest_id}  doi {rec.get('doi')}  version {latest_version}")
    if latest_id != 19758619:
        say("  note: the latest version is not 19758619 (v6.2.0); the lab's receipts do not "
            "record it. Proceeding from what Zenodo reports.")
    if latest_version == VERSION:
        return fail("resolve_latest", f"the latest version is already {VERSION} "
                    f"(record {latest_id}); nothing to do")

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
            say(f"  could not list drafts ({redact(e)}); the post-newversion guard still applies")
        if open_drafts:
            return fail("existing_draft", f"unpublished draft(s) {open_drafts} already exist in "
                        f"concept {CONCEPT_RECID}. Review them at https://zenodo.org/uploads/"
                        f"<id>; re-run with --resume-draft <id> to take one over, or discard it")
        if open_drafts == []:
            say("  none")

        rule(f"3. new version from record {latest_id}")
        try:
            dep = z.call("GET", f"{ZENODO_API}/deposit/depositions/{latest_id}").json()
            if not dep.get("submitted"):
                return fail("new_version", f"deposition {latest_id} is not published")
            nv = z.call("POST", f"{ZENODO_API}/deposit/depositions/{latest_id}/actions/newversion",
                        expect=(200, 201)).json()
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
    receipt["draft_id"] = draft_id
    receipt["draft_url"] = (draft.get("links") or {}).get("html") or f"https://zenodo.org/uploads/{draft_id}"
    receipt["draft_api_url"] = f"{ZENODO_API}/deposit/depositions/{draft_id}"
    if str(draft.get("conceptrecid")) != CONCEPT_RECID:
        return fail("fetch_draft", f"draft {draft_id} has conceptrecid "
                    f"{draft.get('conceptrecid')}, not {CONCEPT_RECID}")
    if draft.get("submitted") or draft.get("state") == "done":
        return fail("fetch_draft", f"deposition {draft_id} is published; refusing to touch it")
    inherited = draft.get("files") or []
    if resume_draft is None:
        dv = (draft.get("metadata") or {}).get("version")
        extra = [(f.get("filename"), md5_of(f.get("checksum"))) for f in inherited
                 if (f.get("filename"), md5_of(f.get("checksum"))) not in latest_files]
        if dv not in (None, "", latest_version) or extra:
            return fail("fresh_copy_guard", f"draft {draft_id} is not a fresh copy of record "
                        f"{latest_id} (version {dv!r}, files not in the predecessor {extra}); it "
                        "looks like a draft someone already worked on. Nothing was changed in it")
    say(f"  draft {draft_id}: {receipt['draft_url']}")
    receipt["status"] = "draft_created"
    write_receipt(receipt)

    rule("4. delete the files the draft inherited")
    try:
        for f in inherited:
            z.call("DELETE", f"{ZENODO_API}/deposit/depositions/{draft_id}/files/{f['id']}",
                   expect=(200, 204))
            say(f"  deleted {f.get('filename')}")
        left = z.call("GET", f"{ZENODO_API}/deposit/depositions/{draft_id}/files").json()
    except Refuse as e:
        return fail("delete_inherited", str(e))
    if left:
        return fail("delete_inherited", f"draft still lists files after deletion: "
                    f"{[x.get('filename') for x in left]}")
    receipt["inherited_files_deleted"] = [f.get("filename") for f in inherited]
    say(f"  {len(inherited)} inherited file(s) deleted; the draft holds none")

    rule("5. upload the three files")
    bucket = (draft.get("links") or {}).get("bucket", "")
    uploaded = []
    for f in facts:
        try:
            with open(f["path"], "rb") as fh:
                r = z.call("PUT", f"{bucket}/{f['name']}", data=fh, timeout=(30, 1800),
                           expect=(200, 201))
            body = r.json()
        except (Refuse, ValueError) as e:
            return fail("upload", f"{f['name']}: {e}")
        remote_md5, remote_size = md5_of(body.get("checksum")), body.get("size")
        uploaded.append({"name": f["name"], "put_md5": remote_md5, "put_size": remote_size})
        if remote_md5 != f["md5"] or remote_size != f["bytes"]:
            return fail("upload", f"{f['name']}: Zenodo stored md5 {remote_md5} / {remote_size} "
                        f"bytes, local {f['md5']} / {f['bytes']}")
        say(f"  uploaded {f['name']}  {remote_size} bytes  md5 {remote_md5}  (matches local)")
    receipt["uploads"] = uploaded

    rule("6. set metadata from metadata.json")
    try:
        z.call("PUT", f"{ZENODO_API}/deposit/depositions/{draft_id}", json_body=doc,
               timeout=(30, 120))
        back = z.call("GET", f"{ZENODO_API}/deposit/depositions/{draft_id}").json()
    except Refuse as e:
        return fail("metadata", str(e))
    sent, got = doc["metadata"], back.get("metadata") or {}
    lic = got.get("license")
    lic = (lic.get("id") if isinstance(lic, dict) else lic) or ""
    comparisons = {
        "title": got.get("title") == sent["title"],
        "version": got.get("version") == sent["version"],
        "publication_date": got.get("publication_date") == sent["publication_date"],
        "upload_type": got.get("upload_type") == sent["upload_type"],
        "license": lic.lower() == sent["license"],
        "creators": [x.get("name") for x in got.get("creators") or []]
                    == [x["name"] for x in sent["creators"]],
        "keywords": sorted(got.get("keywords") or []) == sorted(sent["keywords"]),
        "related_identifiers": sorted((x.get("relation"), x.get("identifier"))
                                      for x in got.get("related_identifiers") or [])
                               == sorted((x["relation"], x["identifier"])
                                         for x in sent["related_identifiers"]),
        "description_nonempty": bool(got.get("description")),
        "notes": (got.get("notes") or "") == (sent.get("notes") or ""),
        "access_right": got.get("access_right") == sent.get("access_right"),
    }
    receipt["metadata_readback"] = comparisons
    reserved = (got.get("prereserve_doi") or {}).get("doi")
    if reserved:
        receipt["reserved_doi_not_registered_until_publish"] = reserved
    for k, ok in comparisons.items():
        say(f"  [{'same' if ok else 'DIFFERS'}] {k}")
    differs = [k for k, ok in comparisons.items() if not ok]

    rule("7. verify the draft's files: Zenodo md5 and size against local")
    try:
        remote = z.call("GET", f"{ZENODO_API}/deposit/depositions/{draft_id}/files").json()
    except Refuse as e:
        return fail("verify_files", str(e))
    by_name = {x.get("filename"): x for x in remote}
    checks, ok_all = [], set(by_name) == {f["name"] for f in facts}
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
    say(f"  review it:  {receipt['draft_url']}")
    if reserved:
        say(f"  reserved DOI (registered only if published): {reserved}")
    say("  This script sent no publish request and has no way to. Read the draft, then press")
    say("  Publish yourself, or discard it.")
    return 0


# ---------------------------------------------------------------------------

def print_plan(doc: dict, facts: list[dict], token_hint: str) -> None:
    body = METADATA_PATH.read_bytes()
    rule("PLANNED REQUESTS (the real run, in order; nothing below was sent)")
    say(f"  every request: https only, host {ZENODO_HOST}, headers "
        f"'Authorization: Bearer <token>' + 'User-Agent: {UA}'")
    say(f"  token source at run time: {token_hint}; the value is never printed")
    plan = [
        ("GET", f"{ZENODO_API}/records/{CONCEPT_RECID}/versions/latest",
         "resolve LATEST_ID / LATEST_DOI / LATEST_VERSION; stop if conceptrecid differs or "
         "LATEST_VERSION is already 7.48.0"),
        ("GET", f"{ZENODO_API}/deposit/depositions?status=draft&size=100",
         "stop if any unpublished draft in concept 19758618 exists"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{LATEST_ID}}", "must be published"),
        ("POST", f"{ZENODO_API}/deposit/depositions/{{LATEST_ID}}/actions/newversion",
         "DRAFT_ID from links.latest_draft; stop if it equals LATEST_ID"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}",
         "guard: conceptrecid 19758618, unsubmitted, a fresh copy (else stop, touching nothing); "
         "then write the receipt with status draft_created"),
        ("DELETE", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}/files/{{FILE_ID}}",
         "once per inherited file"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}/files", "must return []"),
    ]
    for f in facts:
        plan.append(("PUT", f"{{BUCKET}}/{f['name']}",
                     f"body: {f['bytes']} bytes, sha256 {f['sha256']}; response md5 must be "
                     f"{f['md5']}"))
    plan += [
        ("PUT", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}",
         f"body: metadata.json, {len(body)} bytes, sha256 {hashlib.sha256(body).hexdigest()}"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}",
         "read the metadata back; title/version/publication_date/upload_type must match"),
        ("GET", f"{ZENODO_API}/deposit/depositions/{{DRAFT_ID}}/files",
         "exactly the three files; Zenodo md5 + filesize must equal local"),
    ]
    for i, (method, url, why) in enumerate(plan, 1):
        say(f"  {i:2}. {method:6} {url}")
        say(f"        {why}")
    say(f"  then: write {RECEIPT_PATH.name} (status draft_ready_unpublished) and STOP.")
    say("  No publish request exists in this plan or in this script.")

    rule("METADATA THAT WOULD BE SENT (description shortened here; full text in metadata.json)")
    m = dict(doc["metadata"])
    d = m.pop("description")
    say(json.dumps(m, indent=2, ensure_ascii=False))
    say(f"  description: {len(d)} characters of HTML, sha256 "
        f"{hashlib.sha256(d.encode('utf-8')).hexdigest()}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepare (never publish) styxx v7.48.0 as a new Zenodo software version.")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline: validate files and metadata, print the planned requests")
    ap.add_argument("--token-file", help="file holding the Zenodo token (never printed)")
    ap.add_argument("--resume-draft", type=int, metavar="DRAFT_ID",
                    help="take over this existing unpublished draft in concept 19758618")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="styxx checkout holding tag v7.48.0 (for the offline checks)")
    ap.add_argument("--skip-tree-check", action="store_true",
                    help="skip the blob-by-blob check of the source bundle against the tag")
    a = ap.parse_args()

    rule("PREFLIGHT (offline)")
    c, facts, doc = preflight(a.repo, a.skip_tree_check)
    c.show()
    if c.failed or doc is None:
        say(f"\nREFUSED: {len(c.failed)} check(s) failed. Nothing was sent.")
        return 2

    if a.dry_run:
        hint = (f"--token-file {a.token_file}" if a.token_file
                else "ZENODO_TOKEN environment variable (or --token-file PATH)")
        print_plan(doc, facts, hint)
        rule("DRY RUN: no token was read and no network call was made")
        passed = sum(1 for r in c.rows if r[0] == "PASS")
        skipped = sum(1 for r in c.rows if r[0] == "SKIP")
        say(f"  checks: {passed} passed, {skipped} skipped, 0 failed")
        return 0

    if any(r[0] == 'SKIP' for r in c.rows):
        say('REFUSED: a check was skipped; the real run needs every check to pass (pass --repo)')
        return 2
    try:
        token, source = load_token(a.token_file)
    except Refuse as e:
        say(f"REFUSED: {e}")
        return 2
    _SECRETS.append(token)
    say(f"\ntoken: from {source} (value not shown)")
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
