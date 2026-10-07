"""Publish the styxx 7.49.0 Zenodo draft, only after re-checking it, then read the record back.

Adapted from scripts/zenodo_publish_software_v7_48_1.py (which published 7.48.1 as
10.5281/zenodo.23200977). It publishes only the draft that
zenodo-draft-receipt-software-v7.49.0.json names (there is no --draft-id override), it checks the
licence and every other field, and it reads the published record and its DOI back and writes the
deposit receipt from what Zenodo returns. New here: a doi.org answer that redirects to
zenodo.org/doi/<the DOI> (what 7.48.1's DOI answered just after its publish) is recorded as
pointing at Zenodo's DOI page, apart from one that points at the record itself.

BEFORE IT PUBLISHES (any failure stops it, and nothing is published)
  offline
    - every offline check of zenodo_deposit_software_v7_49_0.py passes again: the three files,
      the bundle against the v7.49.0 tree blob by blob, the wheel and sdist against the tag,
      metadata.json and every claim of its description against the tag
    - the draft receipt says draft_ready_unpublished, names this concept, version and commit,
      pins the same metadata.json, the same three files and the same deposit script, and
      recorded every file and metadata field as matching
    - --confirm repeats the draft id the receipt names
    - zenodo-deposit-receipt-software-v7.49.0.json does not exist yet (no second publish)
  from Zenodo
    - the draft is unpublished, in concept 19758618, and its reserved DOI is its own
    - every metadata field equals metadata.json (the licence may read back as mit-license), and
      the draft's metadata holds no key metadata.json does not send, other than doi,
      prereserve_doi and imprint_publisher (a community, contributor, reference, subject or grant
      added in the browser after the deposit stops it)
    - it holds exactly the three files, each with the local md5 and size
    - the concept's latest published version is still the one the draft was made from

THEN: one POST .../actions/publish on that draft id. It is never retried. A 2xx counts as
accepted only when its body is this deposition with submitted true. Only an HTTP 4xx from
Zenodo counts as a refusal (exit 1), and only once a read of the draft succeeds and shows it still
unpublished. If that read fails (no answer, an HTTP error, or a body that is not the deposition),
the outcome is not known, as it is with no answer to the publish (a timeout or a dropped
connection), a 5xx, or a 2xx whose body is not this deposition marked submitted (not JSON, another
object, or the draft still unsubmitted): it re-reads the draft up to READBACK_TRIES times,
READBACK_PAUSE apart, and goes on to the read-back if the draft reads as published, or stops with
exit 4 and a deposit receipt whose status is publish_outcome_unknown.

The draft receipt must name 23200977 (7.48.1) as the version the draft was made from: metadata.json
names that record as the release before this one (isNewVersionOf), and a draft made from another
would publish a false relation.

AFTER: reads the deposition and the public record back (DOI 10.5281/zenodo.<id>, concept DOI,
version, title, files and md5), checks that the concept's latest version is now this record, asks
doi.org (without the token) where the DOI points, and writes
zenodo-deposit-receipt-software-v7.49.0.json, zenodo-metadata-software-v7.49.0-as-published.json
and zenodo-record-software-v7.49.0-readback.json.

The token is read only from --token-file ([ZENODO] section, zenodo_token line), sent only in an
Authorization header to zenodo.org, and never printed, logged or written; the file's path is not
printed or recorded either.

USAGE
    python zenodo_publish_software_v7_49_0.py --dry-run
        Offline: no token, no network (a socket guard refuses any connection). Runs every offline
        check, validates the draft receipt if it exists, and prints the requests a real run sends.
    python zenodo_publish_software_v7_49_0.py --token-file PATH --confirm DRAFT_ID
        Publishes. DRAFT_ID must be the one the draft receipt names.

Exit codes: 0 published and read back; 2 refused offline; 1 refused after reading the draft, or
Zenodo answered the publish with an HTTP 4xx and a read then showed the draft still unpublished
(nothing published); 3 published, but a read-back
check did not pass (the DOI exists; read the receipt); 4 the publish request's outcome could not
be determined (read the draft in the browser; the receipt it leaves blocks a second run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zenodo_deposit_software_v7_49_0 as dep  # noqa: E402  (same checks, loader, allowlist style)

DRAFT_RECEIPT = dep.RECEIPT_PATH
DEPOSIT_RECEIPT = HERE / "zenodo-deposit-receipt-software-v7.49.0.json"
AS_PUBLISHED = HERE / "zenodo-metadata-software-v7.49.0-as-published.json"
RECORD_READBACK = HERE / "zenodo-record-software-v7.49.0-readback.json"
DEPOSIT_SCRIPT = HERE / "zenodo_deposit_software_v7_49_0.py"
API = dep.ZENODO_API
READBACK_TRIES = 8
READBACK_PAUSE = 5.0

say, rule, Refuse, redact = dep.say, dep.rule, dep.Refuse, dep.redact


class Ctx:
    def __init__(self, draft_id: int) -> None:
        self.draft_id = draft_id


def check_allowed(method: str, url: str, ctx: Ctx) -> None:
    """zenodo.org only, and only the one draft the receipt names."""
    u = urlparse(url)
    if u.scheme != "https" or u.hostname != dep.ZENODO_HOST or u.port not in (None, 443):
        raise Refuse(f"request to {u.scheme}://{u.hostname} is not allowed")
    d = ctx.draft_id
    allowed = [
        ("GET", rf"^/api/deposit/depositions/{d}$"),
        ("GET", rf"^/api/deposit/depositions/{d}/files$"),
        ("POST", rf"^/api/deposit/depositions/{d}/actions/publish$"),
        ("GET", rf"^/api/records/{d}$"),
        ("GET", rf"^/api/records/{dep.CONCEPT_RECID}/versions/latest$"),
    ]
    if not any(m == method and re.match(rx, u.path) for m, rx in allowed):
        raise Refuse(f"{method} {u.path} is not on this script's allowlist")


def check_doi_allowed(url: str, draft_id: int) -> None:
    u = urlparse(url)
    if u.scheme != "https" or u.hostname != "doi.org" or \
            not re.fullmatch(rf"/10\.5281/zenodo\.{draft_id}", u.path):
        raise Refuse(f"DOI lookup {url} is not allowed")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_draft_receipt(c: dep.Checks, facts: list[dict], rec: dict) -> int | None:
    """Offline checks of the draft receipt; returns the draft id it names, or None."""
    did = rec.get("draft_id")
    ok_id = isinstance(did, int) and did > 0
    c.add(ok_id, "draft receipt names a draft id", repr(did))
    c.add(rec.get("schema") == "fathom/zenodo-draft-receipt/v2", "draft receipt schema v2",
          repr(rec.get("schema")))
    c.add(rec.get("status") == "draft_ready_unpublished" and rec.get("published") is False,
          "draft receipt: draft_ready_unpublished, not published", repr(rec.get("status")))
    c.add(str(rec.get("concept_recid")) == dep.CONCEPT_RECID and rec.get("version") == dep.VERSION
          and rec.get("tag_commit") == dep.TAG_COMMIT,
          f"draft receipt: concept {dep.CONCEPT_RECID}, version {dep.VERSION}, commit {dep.TAG_COMMIT[:8]}")
    c.add(rec.get("metadata_json_sha256") == sha256_file(dep.METADATA_PATH),
          "metadata.json is the one the draft was filled from")
    c.add(rec.get("script_sha256") == sha256_file(DEPOSIT_SCRIPT),
          "the deposit script is the one that made the draft")
    local = [{k: f[k] for k in ("name", "bytes", "sha256", "md5")} for f in facts]
    c.add(rec.get("local_files") == local, "the three files are the ones the draft was given")
    fc = rec.get("file_checksums") or []
    c.add(len(fc) == len(facts) and all(x.get("match") for x in fc),
          "draft receipt: every file matched on Zenodo")
    mr = rec.get("metadata_readback") or {}
    c.add(bool(mr) and all(mr.values()), "draft receipt: every metadata field read back the same",
          str(sorted(k for k, v in mr.items() if not v)) if mr and not all(mr.values()) else "")
    # metadata.json names 23200977 (7.48.1) as the version before this one; a draft made from any
    # other record would publish a false isNewVersionOf, so only that predecessor is accepted
    pred = rec.get("predecessor") or {}
    c.add(pred.get("id") == dep.PREDECESSOR_RECID and pred.get("version") == dep.PREDECESSOR_VERSION,
          f"draft receipt: made from {dep.PREDECESSOR_RECID} ({dep.PREDECESSOR_VERSION}), the "
          "version metadata.json names as the one before this",
          f"{pred.get('id')!r} ({pred.get('version')!r})")
    reserved = rec.get("reserved_doi_not_registered_until_publish")
    c.add(reserved in (None, f"10.5281/zenodo.{did}"), "reserved DOI is the draft's own", repr(reserved))
    return did if ok_id else None


def offline(repo: str, dry_run: bool) -> tuple[dep.Checks, list[dict], dict | None, dict | None, int | None]:
    c, facts, doc = dep.preflight(repo, skip_tree=False)
    c.add(not DEPOSIT_RECEIPT.exists(), "no deposit receipt yet (this draft was not published by this script)",
          DEPOSIT_RECEIPT.name if DEPOSIT_RECEIPT.exists() else "")
    rec, did = None, None
    if not DRAFT_RECEIPT.is_file():
        c.add(None if dry_run else False, "draft receipt",
              f"{DRAFT_RECEIPT.name} not written yet: the deposit has not run")
    else:
        try:
            rec = json.loads(DRAFT_RECEIPT.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            c.add(False, "draft receipt parses", type(e).__name__)
        else:
            if len(facts) == len(dep.EXPECTED_FILES):
                did = validate_draft_receipt(c, facts, rec)
    guard_selftest(c, did or 2)
    return c, facts, doc, rec, did


def guard_selftest(c: dep.Checks, did: int) -> None:
    ctx = Ctx(did)
    bad = []
    probes = [("POST", f"{API}/deposit/depositions/{did + 1}/actions/publish"),
              ("POST", f"{API}/deposit/depositions/{dep.PREDECESSOR_RECID}/actions/publish"),
              ("POST", f"{API}/deposit/depositions/{did}/actions/edit"),
              ("POST", f"{API}/deposit/depositions/{did}/actions/discard"),
              ("POST", f"{API}/deposit/depositions/{did}/actions/newversion"),
              ("PUT", f"{API}/deposit/depositions/{did}"),
              ("DELETE", f"{API}/deposit/depositions/{did}"),
              ("POST", f"https://example.org/api/deposit/depositions/{did}/actions/publish")]
    for method, url in probes:
        try:
            check_allowed(method, url, ctx)
            bad.append(f"{method} {urlparse(url).path}")
        except Refuse:
            pass
    try:
        check_allowed("POST", f"{API}/deposit/depositions/{did}/actions/publish", ctx)
        only = True
    except Refuse:
        only = False
    c.add(not bad and only, "allowlist: publish on the receipt's draft only; no edit, discard, "
          "newversion, PUT or DELETE", f"allowed: {bad}" if bad else f"{len(probes)} probes refused")
    try:
        check_doi_allowed(f"https://doi.org/10.5281/zenodo.{did + 1}", did)
        c.add(False, "DOI lookup only for the draft's own DOI")
    except Refuse:
        c.add(True, "DOI lookup only for the draft's own DOI")


def remote_checks(dpos: dict, files: list, latest: dict, doc: dict, facts: list[dict],
                  rec: dict, did: int) -> list[tuple[str, bool, str]]:
    m = dpos.get("metadata") or {}
    rows: list[tuple[str, bool, str]] = []
    rows.append(("unpublished draft", dpos.get("submitted") is False
                 and dpos.get("state") in ("unsubmitted", "inprogress"),
                 f"submitted={dpos.get('submitted')} state={dpos.get('state')}"))
    rows.append(("in concept 19758618", str(dpos.get("conceptrecid")) == dep.CONCEPT_RECID,
                 str(dpos.get("conceptrecid"))))
    rows.append(("draft id is the receipt's", dpos.get("id") == did, str(dpos.get("id"))))
    reserved = (m.get("prereserve_doi") or {}).get("doi")
    rows.append(("reserved DOI is the draft's own", reserved in (None, f"10.5281/zenodo.{did}"),
                 str(reserved)))
    cmp = dep.compare_metadata(doc["metadata"], m)
    for k, ok in cmp.items():
        rows.append((f"metadata {k} = metadata.json", ok, ""))
    extra = dep.unexpected_metadata_keys(doc["metadata"], m)
    rows.append(("no metadata key beyond those sent (+ doi, prereserve_doi, imprint_publisher)",
                 not extra, f"extra: {extra}" if extra else ""))
    by = {x.get("filename"): x for x in files}
    rows.append(("exactly the three files", sorted(by) == sorted(f["name"] for f in facts)
                 and len(files) == len(facts), str(sorted(by))))
    for f in facts:
        x = by.get(f["name"], {})
        rows.append((f"{f['name']}: md5 and size = local",
                     dep.md5_of(x.get("checksum")) == f["md5"] and x.get("filesize") == f["bytes"],
                     f"{dep.md5_of(x.get('checksum'))} / {x.get('filesize')}"))
    pred = (rec.get("predecessor") or {}).get("id")
    rows.append(("the concept's latest version is still the draft's predecessor, 23200977",
                 str(latest.get("id")) == str(pred) == str(dep.PREDECESSOR_RECID)
                 and str(latest.get("conceptrecid")) == dep.CONCEPT_RECID,
                 f"latest {latest.get('id')} ({(latest.get('metadata') or {}).get('version')}), "
                 f"predecessor {pred}"))
    return rows


def record_file_rows(files, facts: list[dict]) -> tuple[bool, list]:
    if isinstance(files, dict):   # an InvenioRDM-style {"entries": ...} instead of the legacy list
        files = files.get("entries") or []
        files = list(files.values()) if isinstance(files, dict) else files
    got = {}
    for x in files or []:
        if not isinstance(x, dict):
            continue
        name = x.get("key") or x.get("filename")
        got[name] = (dep.md5_of(x.get("checksum")), x.get("size") if x.get("size") is not None
                     else x.get("filesize"))
    rows = []
    ok = sorted(got) == sorted(f["name"] for f in facts)
    for f in facts:
        md5, size = got.get(f["name"], (None, None))
        good = md5 == f["md5"] and size is not None and int(size) == f["bytes"]
        ok &= good
        rows.append({"name": f["name"], "md5": md5, "size": size, "match": good})
    return ok, rows


def read_deposition(z, did: int) -> dict | None:
    """One GET of the deposition: this deposition as a JSON object with a boolean `submitted`, or
    None when the read failed (no answer, an HTTP error, a body that is not a JSON object, or one
    that is not this deposition). None says nothing about whether the draft was published."""
    try:
        body = z.call("GET", f"{API}/deposit/depositions/{did}").json()
    except (Refuse, ValueError) as e:
        say(f"  draft {did} not readable: {e}")
        return None
    if not isinstance(body, dict) or body.get("id") != did or not isinstance(body.get("submitted"), bool):
        say(f"  draft {did} not readable: the answer is not this deposition with a submitted flag")
        return None
    return body


def reread_until_submitted(z, did: int) -> tuple[dict | None, list[str]]:
    """Up to READBACK_TRIES reads, READBACK_PAUSE apart: the deposition once it reads as
    submitted, else None, with what each read showed."""
    seen: list[str] = []
    for i in range(READBACK_TRIES):
        if i:
            time.sleep(READBACK_PAUSE)
        after = read_deposition(z, did)
        if after is not None and after["submitted"] is True:
            return after, seen + ["submitted"]
        seen.append(f"state {after.get('state')}" if after is not None else "unreadable")
    return None, seen


def publish(token: str, source: str, doc: dict, facts: list[dict], rec: dict, did: int) -> int:
    ctx = Ctx(did)
    z = dep.Zenodo(token, ctx, allow=check_allowed)
    out: dict = {
        "schema": "fathom/zenodo-deposit-receipt/v2",
        "action": "software_deposit_published",
        "status": "started",
        "version": dep.VERSION,
        "tag": dep.TAG,
        "tag_commit": dep.TAG_COMMIT,
        "deposit_id": did,
        "concept_recid": dep.CONCEPT_RECID,
        "token_source": source,
        "draft_receipt_sha256": sha256_file(DRAFT_RECEIPT),
        "metadata_json_sha256": sha256_file(dep.METADATA_PATH),
        "deposit_script_sha256": sha256_file(DEPOSIT_SCRIPT),
        "publish_script_sha256": sha256_file(Path(__file__)),
        "predecessor": rec.get("predecessor"),
        "files": {f["name"]: {"bytes": f["bytes"], "sha256": f["sha256"], "md5": f["md5"]} for f in facts},
    }

    def finish(status: str, code: int, why: str = "") -> int:
        out["status"] = status
        if why:
            out["because"] = redact(why)
        out["requests"] = z.log
        out["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if status.startswith("published") or status == "publish_outcome_unknown":
            dep.write_json(DEPOSIT_RECEIPT, out)
            say(f"  receipt written: {DEPOSIT_RECEIPT.name} (status {status})")
        else:
            say(f"  no deposit receipt written (status {status})")
        return code

    rule(f"1. re-read draft {did} from Zenodo")
    try:
        dpos = z.call("GET", f"{API}/deposit/depositions/{did}").json()
        files = z.call("GET", f"{API}/deposit/depositions/{did}/files").json()
        latest = z.call("GET", f"{API}/records/{dep.CONCEPT_RECID}/versions/latest").json()
    except (Refuse, ValueError) as e:
        say(f"REFUSED: {e}")
        return finish("refused_before_publish", 1, str(e))
    rows = remote_checks(dpos, files, latest, doc, facts, rec, did)
    for name, ok, detail in rows:
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    out["checks_before_publish"] = {name: ok for name, ok, _ in rows}
    if not all(ok for _, ok, _ in rows):
        say("REFUSED: a check failed; nothing was published")
        return finish("refused_before_publish", 1, "a check of the draft failed")

    rule(f"2. publish draft {did} (one request, never retried)")
    pub: dict | None = None
    unknown_why = ""
    try:
        r = z.call("POST", f"{API}/deposit/depositions/{did}/actions/publish",
                   expect=(200, 201, 202), timeout=(30, 300))
    except dep.HttpStatus as e:
        if 400 <= e.status < 500:
            # Zenodo answered and refused. Only a read that succeeds and shows the draft still
            # unpublished lets this say so. If it reads as published after all, go on to the
            # read-back; if the read fails, the outcome is not known: re-read below.
            say(f"  Zenodo refused the publish: HTTP {e.status}")
            after = read_deposition(z, did)
            if after is None:
                say("  the read that would confirm the draft is still unpublished failed")
                unknown_why = (f"Zenodo answered HTTP {e.status}, but the read that would confirm "
                               "the draft is still unpublished failed")
            elif after["submitted"] is not True:
                say("  nothing was published")
                return finish("publish_refused", 1, str(e))
            else:
                say("  Zenodo answered 4xx, but the draft reads back as published; continuing to "
                    "the read-back")
                pub = after
        else:
            unknown_why = f"Zenodo answered HTTP {e.status}"
    except dep.Transport as e:
        unknown_why = f"no answer was read ({e})"
    except Refuse as e:   # not expected for this request; never read as a refusal by Zenodo
        unknown_why = f"the request did not complete ({e})"
    else:
        try:
            body = r.json()
        except ValueError:
            body = None
        # a 2xx says the publish was accepted only when its body is this deposition, submitted;
        # any other body (not JSON, another object, this draft still unsubmitted) leaves the
        # outcome to the re-reads below
        if isinstance(body, dict) and body.get("id") == did and body.get("submitted") is True:
            pub = body
            say(f"  Zenodo accepted the publish: doi {pub.get('doi')}")
        elif not isinstance(body, dict):
            unknown_why = f"HTTP {r.status_code}, but the answer is not a JSON object"
        elif body.get("id") != did:
            unknown_why = (f"HTTP {r.status_code}, but the answer is not deposition {did} "
                           f"(id {body.get('id')!r})")
        else:
            unknown_why = (f"HTTP {r.status_code}, but the answer shows deposition {did} with "
                           f"submitted {body.get('submitted')!r}")
    if pub is None:
        say(f"  the publish request's outcome is not known: {unknown_why}")
        say(f"  re-reading draft {did} (up to {READBACK_TRIES} times, {READBACK_PAUSE:.0f} s apart)")
        after, seen = reread_until_submitted(z, did)
        if after is None:
            say("  the draft never read back as published; the publish may still complete")
            return finish("publish_outcome_unknown", 4,
                          f"publish: {unknown_why}; then {READBACK_TRIES} reads of the draft gave "
                          f"{seen}. Zenodo may still complete the publish. Read "
                          f"https://zenodo.org/uploads/{did} in the browser before anything else; "
                          f"this script refuses to run again while {DEPOSIT_RECEIPT.name} exists")
        say("  the draft reads back as published; continuing to the read-back")
        pub = after
    out["published_at_local_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        return readback(z, out, finish, pub, doc, facts, did)
    except Exception as e:  # noqa: BLE001 -- published already: always leave a receipt
        say(f"  read-back raised {type(e).__name__}: {redact(e)}")
        return finish("published_readback_incomplete", 3, f"read-back raised {type(e).__name__}")


def readback(z, out: dict, finish, pub: dict, doc: dict, facts: list[dict], did: int) -> int:
    rule("3. read the published record and its DOI back")
    want_doi = f"10.5281/zenodo.{did}"
    post: dict[str, bool] = {}
    post["publish response: this deposition, submitted, DOI = 10.5281/zenodo.<id>"] = (
        pub.get("id") == did and pub.get("submitted") is True and pub.get("doi") == want_doi)
    after, record = {}, {}
    try:
        after = z.call("GET", f"{API}/deposit/depositions/{did}").json()
    except (Refuse, ValueError) as e:
        say(f"  deposition read-back failed: {e}")
    am = after.get("metadata") or {}
    post["deposition: published (state done)"] = after.get("submitted") is True and after.get("state") == "done"
    post["deposition: DOI = 10.5281/zenodo.<id>"] = (after.get("doi") or am.get("doi")) == want_doi
    post["deposition: concept DOI 10.5281/zenodo.19758618"] = (
        str(after.get("conceptrecid")) == dep.CONCEPT_RECID and after.get("conceptdoi") == dep.CONCEPT_DOI)
    cmp = dep.compare_metadata(doc["metadata"], am) if am else {}
    post["deposition: every metadata field = metadata.json"] = bool(cmp) and all(cmp.values())
    if am:
        dep.write_json(AS_PUBLISHED, am)
        say(f"  wrote {AS_PUBLISHED.name}")
    for i in range(READBACK_TRIES):
        try:
            record = z.call("GET", f"{API}/records/{did}").json()
            if record.get("doi") == want_doi:
                break
        except (Refuse, ValueError) as e:
            say(f"  public record not readable yet ({e})")
        time.sleep(READBACK_PAUSE)
    rm = record.get("metadata") or {}
    post["public record: DOI = 10.5281/zenodo.<id>"] = record.get("doi") == want_doi
    post["public record: concept 19758618"] = (str(record.get("conceptrecid")) == dep.CONCEPT_RECID
                                               and record.get("conceptdoi") in (None, dep.CONCEPT_DOI))
    post[f"public record: version {dep.VERSION}, title = metadata.json"] = (
        rm.get("version") == dep.VERSION and rm.get("title") == doc["metadata"]["title"])
    files_ok, file_rows = record_file_rows(record.get("files") or after.get("files") or [], facts)
    post["public record: the three files, md5 and size = local"] = files_ok
    if record:
        dep.write_json(RECORD_READBACK, record)
        say(f"  wrote {RECORD_READBACK.name}")
    latest_id = None
    for i in range(READBACK_TRIES):
        try:
            latest_id = z.call("GET", f"{API}/records/{dep.CONCEPT_RECID}/versions/latest").json().get("id")
            if str(latest_id) == str(did):
                break
        except (Refuse, ValueError) as e:
            say(f"  versions/latest not readable ({e})")
        time.sleep(READBACK_PAUSE)
    latest_ok = str(latest_id) == str(did)
    out["concept_latest_after_publish"] = latest_id

    doi_check = resolve_doi(want_doi, did)
    out["doi_resolver"] = doi_check

    out.update({
        "software_doi": after.get("doi") or record.get("doi") or pub.get("doi"),
        "software_doi_url": f"https://doi.org/{want_doi}",
        "concept_doi": after.get("conceptdoi") or record.get("conceptdoi"),
        "record_url": ((record.get("links") or {}).get("self_html") or (record.get("links") or {}).get("html")
                       or f"https://zenodo.org/records/{did}"),
        "published_at_zenodo": {"deposition_modified": after.get("modified"),
                                "record_created": record.get("created")},
        "file_checksums_after_publish": file_rows,
        "related_identifiers": sorted([x.get("relation"), x.get("identifier")]
                                      for x in am.get("related_identifiers") or []),
        "license_as_stored": dep.license_id(am.get("license")),
        "checks_after_publish": post,
    })
    for name, ok in post.items():
        say(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    say(f"  [{'PASS' if latest_ok else 'LATER'}] concept 19758618 now resolves to {latest_id}"
        + ("" if latest_ok else " (Zenodo's index may lag; re-check the concept page)"))
    say(f"  [{'PASS' if doi_check.get('points_at_record') else 'LATER'}] doi.org: "
        f"{doi_check.get('status')} -> {doi_check.get('location')}"
        + ("" if doi_check.get("points_at_record") else
           " (Zenodo's DOI page, which redirects to the record; 7.48.1's DOI answered the same)"
           if doi_check.get("points_at_zenodo_doi_page") else " (DataCite registration may lag)"))

    if not all(post.values()):
        rule("PUBLISHED, BUT A READ-BACK CHECK DID NOT PASS")
        say(f"  The DOI {want_doi} exists. Read {DEPOSIT_RECEIPT.name} and the record page.")
        return finish("published_readback_incomplete", 3, "a read-back check failed")
    rule("PUBLISHED AND READ BACK")
    say(f"  DOI: https://doi.org/{want_doi}")
    say(f"  record: {out['record_url']}")
    return finish("published_and_verified", 0)


def resolve_doi(doi: str, did: int) -> dict:
    """Ask doi.org where the DOI points. No token is sent; redirects are not followed."""
    url = f"https://doi.org/{doi}"
    check_doi_allowed(url, did)
    try:
        import requests
        r = requests.get(url, headers={"User-Agent": dep.UA}, allow_redirects=False, timeout=30)
    except Exception as e:  # noqa: BLE001 -- recorded, never fatal
        return {"url": url, "status": None, "error": type(e).__name__, "points_at_record": False}
    loc = r.headers.get("Location", "")
    hit = bool(re.search(rf"^https://zenodo\.org/(records?|record)/{did}/?$", loc))
    # just after its publish, 7.48.1's DOI answered with zenodo.org/doi/<doi>, which redirects on
    # to the record
    doi_page = bool(re.search(rf"^https://zenodo\.org/doi/10\.5281/zenodo\.{did}/?$", loc))
    return {"url": url, "status": r.status_code, "location": loc, "points_at_record": hit,
            "points_at_zenodo_doi_page": doi_page}


def print_plan(did: int | None) -> None:
    d = str(did) if did else "{DRAFT_ID}"
    rule("PLANNED REQUESTS (the real run, in order; nothing below was sent)")
    say(f"  zenodo.org requests carry 'Authorization: Bearer <token>' + 'User-Agent: {dep.UA}'")
    say(f"  token at run time: {dep.TOKEN_SOURCE}; neither the value nor the path is ever printed")
    say(f"  the real run also needs --confirm {d}")
    plan = [
        ("GET", f"{API}/deposit/depositions/{d}", "unpublished, concept 19758618, reserved DOI its "
         "own, every metadata field = metadata.json (licence: mit or mit-license), no other key "
         "but doi, prereserve_doi and imprint_publisher"),
        ("GET", f"{API}/deposit/depositions/{d}/files", "exactly the three files, md5 + size = local"),
        ("GET", f"{API}/records/{dep.CONCEPT_RECID}/versions/latest",
         f"still the version the draft was made from, {dep.PREDECESSOR_RECID} ({dep.PREDECESSOR_VERSION})"),
        ("POST", f"{API}/deposit/depositions/{d}/actions/publish",
         "the only POST; sent once, only if every check above passed; never retried. A 2xx is "
         "accepted only as this deposition, submitted. An HTTP 4xx "
         "is a refusal (exit 1) once a read shows the draft still unpublished; no answer, a 5xx, "
         "a 2xx whose body is not this deposition marked submitted, or a 4xx whose confirming "
         "read fails re-reads the draft up to "
         f"{READBACK_TRIES} times: published -> read-back, else exit 4 with a receipt"),
        ("GET", f"{API}/deposit/depositions/{d}", "state done, DOI 10.5281/zenodo." + d
         + ", concept DOI 10.5281/zenodo.19758618, metadata unchanged; saved as the as-published file"),
        ("GET", f"{API}/records/{d}", "public record: DOI, concept, version, title, the three files "
         f"and their md5 (up to {READBACK_TRIES} tries, {READBACK_PAUSE:.0f} s apart)"),
        ("GET", f"{API}/records/{dep.CONCEPT_RECID}/versions/latest", "now this record"),
        ("GET", f"https://doi.org/10.5281/zenodo.{d}", "NO token; redirects not followed; the "
         "Location should be zenodo.org/records/" + d),
    ]
    for i, (method, url, why) in enumerate(plan, 1):
        say(f"  {i}. {method:5} {url}")
        say(f"       {why}")
    say(f"  then: write {DEPOSIT_RECEIPT.name}, {AS_PUBLISHED.name} and {RECORD_READBACK.name}.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Publish the styxx 7.49.0 Zenodo draft named by the "
                                 "draft receipt, after re-checking it, and read it back.")
    ap.add_argument("--dry-run", action="store_true", help="offline: no token, no network")
    ap.add_argument("--token-file", help="file with a [ZENODO] section and a zenodo_token line")
    ap.add_argument("--confirm", type=int, metavar="DRAFT_ID",
                    help="the draft id the draft receipt names (required for the real run)")
    ap.add_argument("--repo", default=dep.DEFAULT_REPO)
    a = ap.parse_args(argv)

    if a.dry_run:
        dep.forbid_network()
    rule("PREFLIGHT (offline)")
    c, facts, doc, rec, did = offline(a.repo, a.dry_run)
    if a.dry_run:
        c.add(dep.network_guard_holds(), "dry run: the socket guard refuses connections and name lookups")
    else:
        c.add(a.confirm is not None and did is not None and a.confirm == did,
              "--confirm repeats the draft id the receipt names", f"--confirm {a.confirm}, receipt {did}")
    c.show()
    if c.failed or doc is None:
        say(f"\nREFUSED: {len(c.failed)} check(s) failed. Nothing was sent.")
        return 2
    if a.dry_run:
        print_plan(did)
        rule("DRY RUN: no token was read and no network call was made")
        passed = sum(1 for r in c.rows if r[0] == "PASS")
        say(f"  checks: {passed} passed, {len(c.skipped)} skipped, 0 failed")
        if did is None:
            say("  the draft receipt does not exist yet; run zenodo_deposit_software_v7_49_0.py before this one")
        return 0
    if c.skipped or rec is None or did is None:
        say(f"REFUSED: a check was skipped ({[r[1] for r in c.skipped]}); nothing was sent")
        return 2
    try:
        token, source = dep.load_token(a.token_file)
    except Refuse as e:
        say(f"REFUSED: {e}")
        return 2
    dep._SECRETS.append(token)
    say(f"\ntoken: read from {source} (value not shown)")
    try:
        return publish(token, source, doc, facts, rec, did)
    except Refuse as e:
        say(f"REFUSED: {e}")
        return 1
    except Exception as e:  # noqa: BLE001 -- last line of defence: redact, never re-raise raw
        say(f"ERROR: {type(e).__name__}: {redact(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
