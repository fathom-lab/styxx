# -*- coding: utf-8 -*-
"""styxx.charon — the ferry log: every verdict this lab issued, re-derived from bytes, chained.

Spec: ``papers/charon/SPEC_charon_v01_2026_09_02.md`` (blob sha256
88b3fa3b1762730b49beaa0fc612cf73e3bb28f54623ea030526b2ed32bec55b), committed before this module
was, and its dated ERRATA section, written after the adversarial pass of the same day
(``papers/charon/ATTACKS_charon_v01_battery_2026_09_02.md``). Log schema ``styxx.charon/log/v1``;
entry schema ``styxx.charon/entry/v1``. The v0 log written before the pass remains in history at
a5cf9ec and is not a v1 log.

WHAT IT IS. An append-only, hash-chained log in which every line is a verdict RE-DERIVED from
bytes when the line was written, over the three verdict-bearing artifact kinds this lab produces:
a sworn document at the commit its sidecar names (``styxx.sworn``), a capsule (``styxx.capsule``
and the pure functions it wraps), an OATH certificate with its receipts (``styxx.corpus_audit`` →
``styxx.certify``). Charon calls those verifiers and writes what they return. It adjudicates
nothing, accuses no one, fetches nothing, signs nothing.

WHAT THE LINE CARRIES.
  * ``receipts.n`` and every receipt digest — the RESOLVED set the verdict was reproduced against;
    for an OATH line the certificate's CITED set travels beside it. The 2026-09-01 dogfood showed
    an OATH verdict on fixed bytes moving FAILED → HELD as receipts were added; the number is on
    the line so a HELD bought with volume is visible. For OATH lines a larger set makes HELD
    strictly easier; for sworn lines ``n`` is the count of distinct receipts the spans named; for
    diffgate lines it is the count of bindings (summary, diff, gate).
  * ``verifier.modules`` — every module on the derivation path, Charon itself included, each with
    its bytes' sha256, digested into ``verifier.digest``. ``verify`` can then tell SKEW (the line
    moved and the instrument's bytes moved) from DRIFT (the line moved under the same build).
    SKEW detection is bounded by the hashed set; a change outside it reads as DRIFT.
  * ``prev`` — the previous line's ``entry_id``; the line after the header names the header's
    domain-separated digest, so the header is chained too.

WHAT THE CHAIN DOES AND DOES NOT DO. An interior edit, an interior removal, a duplicated line or a
reordered line without a rebuild is TAMPER. A forger who REBUILDS every later ``entry_id`` gets a
different HEAD; a truncated tail is a shorter log with a different HEAD. The chain therefore binds
order and content to the head and nothing more: the head must be pinned OUTSIDE the log (a sworn
span, a commit) and handed to ``verify --expect-head`` for the log to be identified. Without an
expected head, ``verify`` establishes only that the file is internally consistent.

DECISIONS the spec left open or the errata changed, stated once here:
  * ``subject.path`` is the DOCUMENT's repository-relative POSIX path; an artifact outside
    ``--repo`` is refused, so no absolute path is ever written into a line.
  * A sworn line's ``subject.sha256`` is over the render of the sidecar as presented at ingest;
    ``at.commit`` is where its receipts resolve; ``at.document_at_commit`` says whether the
    document bytes at that commit are the same bytes (null when no commit).
  * ``reproduced`` is ``true``/``false``/``null``: for a sworn line, ``styxx.sworn.verify_receipt``
    over the committed receipt (null when none exists); for a capsule, the capsule verifier's own
    ``ok``; for a certificate, no verdict class moved, no receipt missing, changed or drifted.
  * ``verdict`` for a capsule is the LIVE re-derivation (``certify_doc`` over the embedded document
    and receipts; ``gate_diff_text`` over the embedded summary and diff); the embedded verdict is
    ``counts.recorded_verdict``. A capsule whose bindings do not hold is UNRESOLVED.
  * ``verdict_class``: sworn → HELD/FAILED/UNSWORN; OATH → ``corpus_audit.verdict_class``
    (``OATH-HELD``/``OATH-FAILED``, coverage suffix stripped); diffgate → PASS/FAIL; UNRESOLVED.
  * ``receipts.vacuous`` marks a sworn HELD that resolved nothing; such lines are excluded from
    every held count.
  * A missing document or receipt is a line (UNRESOLVED, reason in ``counts.reason``), never a
    skip — a population defined by what survived is the defect this lane catalogued nine times.
    The header carries ``population``: how the artifacts were enumerated, so a reader can re-run
    the enumeration and compare counts.
  * Duplicates are permitted — a line is a crossing, not an artifact — and every summary counts
    lines; ``distinct_subjects`` is reported beside ``entries``.
  * Timestamps are outside every digest. JCS is ``styxx.attestation.jcs``: ASCII keys, finite
    doubles; a count key outside ASCII is refused at ingest.
  * Exit status: 0 for every verify outcome except TAMPER and HEAD_MISMATCH; a missing log or an
    unreadable artifact is REFUSED (exit 2), never a clean zero.

WHAT IT DOES NOT SAY, printed on the page and in every report: that any verdict is true; who
wrote any line; that a receipt set was not chosen to pass; that anything is immutable,
tamper-proof or self-verifying — the log is append-only by contract and re-derivable by
construction, and a chain of hashes is exactly that.

CLI::

    python -m styxx.charon ingest --log L --repo R [--name N] [--population TEXT] PATH...
    python -m styxx.charon verify --log L --repo R [--out REPORT.json] [--expect-head HEX]
    python -m styxx.charon derive --repo R PATH            # the core one line would carry, no log touched
    python -m styxx.charon page   --log L --out index.html [--report REPORT.json]
    python -m styxx.charon status --log L [--expect-head HEX]
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import html as _html
import importlib
import json
import re as _re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from styxx.attestation import jcs

__all__ = ["LOG_SCHEMA", "ENTRY_SCHEMA", "KINDS", "STATUSES", "derive", "make_entry",
           "read_log", "check_chain", "header_digest", "ingest", "verify_log", "render_page", "main"]

LOG_SCHEMA = "styxx.charon/log/v1"
ENTRY_SCHEMA = "styxx.charon/entry/v1"
VERIFY_SCHEMA = "styxx.charon/verify/v1"
KINDS = ("sworn", "capsule-oath", "capsule-diffgate", "oath-certificate")
STATUSES = ("SAME_LINE", "MOVED_VERIFIER", "SKEW", "DRIFT", "UNRESOLVED", "HEAD_MISMATCH", "TAMPER")
HELD_CLASSES = ("HELD", "OATH-HELD", "PASS")
_OUTSIDE_DIGEST = ("entry_id", "timestamp", "note")
_CORE_KEYS = ("kind", "subject", "at", "receipts", "verifier", "verdict", "verdict_class", "counts",
              "floor", "rungs", "reproduced")
_COMPARED_KEYS = ("subject", "at", "receipts", "verdict", "verdict_class", "counts", "floor", "rungs",
                  "reproduced")
_REQUIRED_ENTRY_KEYS = ("schema", "seq", "prev", "entry_id") + _CORE_KEYS
_HEADER_DOMAIN = b"styxx.charon/log/v1\n"
# every module on the derivation path, per kind, Charon itself first: a change to any of them is a
# change to the instrument that wrote the line
_MODULES = {
    "sworn": ("styxx.charon", "styxx.sworn", "styxx.attestation", "styxx.claimdetect"),
    "capsule-oath": ("styxx.charon", "styxx.capsule", "styxx.certify"),
    "capsule-diffgate": ("styxx.charon", "styxx.capsule", "styxx.diffgate", "styxx.evidence", "styxx.claimdetect"),
    "oath-certificate": ("styxx.charon", "styxx.corpus_audit", "styxx.certify"),
}
_CERTIFIES = ("every line is a verdict re-derived from bytes by the named verifier modules — at the "
              "named commit where one is named, over the working tree otherwise — chained to the line "
              "before it, with the size and digests of the receipt set it was reproduced against; NOT "
              "a claim that any verdict is true, NOT a record of who wrote a line, NOT a check that a "
              "receipt set was not chosen to pass, and nothing signed; the chain binds order and "
              "content to the head, and only a head pinned outside the log identifies the log")


# ---------------------------------------------------------------------------------------- utilities

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _content_sha256(b: bytes) -> str:
    """Content identity modulo newlines, for bytes read from a WORKING TREE.

    `styxx.corpus_audit._receipt_sha_matches` already states the doctrine this follows: a hash
    that depends on line endings is pinning the wrong thing, because it makes the record
    platform-dependent and defeats the promise that anyone can re-run it. A repository checked
    out on Windows holds CRLF where Linux holds LF, so a raw hash of a working-tree document or
    receipt differs between two strangers reading the same commit — and every OATH line would
    read as moved on the other platform.

    Applied to: the document an OATH certificate certifies, the receipt bytes handed to that
    certification, and a capsule file. NOT applied to a sworn document: the sworn format refuses
    newline normalisation by design, its bytes are pinned by `.gitattributes`, and its digest is
    an identity claim about exact bytes. Digests that come from git plumbing or from embedded
    base64 are already platform-stable and are used as they are.

    DISCLOSED, as the auditor discloses it: two files differing only in line endings hash the
    same here, so this certifies content identity, not byte identity.
    """
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Path, root: Path) -> str:
    """Repository-relative POSIX path, or a refusal: no absolute path is ever written into a line."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        raise SystemExit("REFUSED: %s is outside --repo %s" % (path, root))
    return str(rel).replace("\\", "/")


def _module_pair(name: str) -> Tuple[str, Optional[str]]:
    try:
        mod = importlib.import_module(name)
        return name, _sha256(Path(mod.__file__).read_bytes())
    except Exception:                                  # noqa: BLE001 - an absent optional module is recorded as such
        return name, None


def _verifier(kind: str) -> dict:
    from styxx._version import __version__
    modules = [list(_module_pair(m)) for m in _MODULES[kind]]
    return {"styxx_version": __version__, "modules": modules,
            "digest": _sha256(jcs(modules).encode("utf-8"))}


def _ascii_keys(obj) -> bool:
    if isinstance(obj, dict):
        return all(isinstance(k, str) and k.isascii() and _ascii_keys(v) for k, v in obj.items())
    if isinstance(obj, list):
        return all(_ascii_keys(x) for x in obj)
    return True


def _write_lines_lf(path: Path, lines: List[str], append: bool) -> None:
    with open(path, "a" if append else "w", encoding="utf-8", newline="\n") as fh:
        for ln in lines:
            fh.write(ln + "\n")


# ------------------------------------------------------------------------------ the three derivers

def _unresolved(kind: str, name: str, rel: str, reason: str, sha: Optional[str] = None,
                cited: Optional[dict] = None, recorded: Optional[str] = None) -> dict:
    return {"kind": kind, "subject": {"name": name, "path": rel, "sha256": sha},
            "at": {"commit": None, "manifest_digest": None, "document_at_commit": None},
            "receipts": {"n": 0, "sha256": [], "cited": cited, "vacuous": False},
            "verifier": _verifier(kind),
            "verdict": "UNRESOLVED", "verdict_class": "UNRESOLVED",
            "counts": {"reason": reason, "recorded_verdict": recorded}, "floor": None, "rungs": None,
            "reproduced": None}


def derive_sworn(sidecar_path: Path, repo: Path) -> dict:
    """A sworn document at the commit its sidecar names. Never raises on content."""
    from styxx import sworn as _sworn
    stem = sidecar_path.name[: -len(".sworn.json")]
    doc_path = sidecar_path.with_name(stem + ".md")
    doc_rel = _rel(doc_path, repo)
    try:
        side = _sworn.load_sidecar(json.loads(sidecar_path.read_text(encoding="utf-8")))
    except (SystemExit, OSError, ValueError) as e:
        return _unresolved("sworn", stem + ".md", doc_rel, "sidecar_refused: %s" % getattr(e, "code", e))
    inline = _sworn.render(side)
    commit = side["commit"]
    tree = None
    doc_at_commit: Optional[bool] = None
    if commit is not None:
        tree = _sworn.GitTree(repo, commit)
        why = tree._ready()
        if why:
            return _unresolved("sworn", stem + ".md", doc_rel, why, _sha256(inline))
        blob, _ = tree.blob(doc_rel)
        doc_at_commit = (blob is not None and _sha256(blob) == _sha256(inline))
    try:
        core = _sworn.verify(sidecar=side, name=stem + ".md", tree=tree, commit=commit)
    except SystemExit as e:
        return _unresolved("sworn", stem + ".md", doc_rel, "verify_refused: %s" % e.code, _sha256(inline))
    shas = set()
    for s in core["spans"]:
        if s.get("resolved_sha256"):
            shas.add(s["resolved_sha256"])
    for entry in (side.get("manifest") or {}).get("receipts", {}).values():
        if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
            shas.add(entry["sha256"])
    verdict = core["document_verdict"]
    klass = {"SWORN-HELD": "HELD", "SWORN-FAILED": "FAILED", "UNSWORN": "UNSWORN"}.get(verdict, verdict)
    counts = dict(core["counts"])
    counts["unresolved_spans"] = core["unresolved"]
    counts["recorded_verdict"] = None
    reproduced: Optional[bool] = None
    rec_path = sidecar_path.with_name(stem + ".sworn-receipt.json")
    if rec_path.exists():
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            counts["recorded_verdict"] = rec.get("document_verdict")
            res = _sworn.verify_receipt(rec, sidecar=side, tree=tree)
            counts["receipt_check"] = {k: res.get(k) for k in ("status", "digest_match", "verdict_reproduces",
                                                                 "same_verifier_build", "schema")}
            reproduced = res.get("status") == "VERIFIED"
        except (OSError, ValueError):
            counts["receipt_check"] = {"status": "unreadable"}
            reproduced = False
    vacuous = (verdict == "SWORN-HELD" and not shas)
    return {"kind": "sworn", "subject": {"name": stem + ".md", "path": doc_rel, "sha256": _sha256(inline)},
            "at": {"commit": commit, "manifest_digest": core.get("manifest_digest"),
                   "document_at_commit": doc_at_commit},
            "receipts": {"n": len(shas), "sha256": sorted(shas), "cited": None, "vacuous": vacuous},
            "verifier": _verifier("sworn"),
            "verdict": verdict, "verdict_class": klass, "counts": counts,
            "floor": core["coverage"].get("sentence_share"), "rungs": core.get("rungs"),
            "reproduced": reproduced}


def _capsule_payload(path: Path) -> Optional[dict]:
    from styxx import capsule as _cap
    html = path.read_text(encoding="utf-8")
    try:
        i = html.index(_cap._BEGIN) + len(_cap._BEGIN)
        j = html.index(_cap._END, i)
        return json.loads(html[i:j])
    except (ValueError, json.JSONDecodeError):
        return None


def _capsule_live_v01(payload: dict) -> Tuple[Optional[str], Optional[dict]]:
    """The same pure function the capsule verifier runs: certify_doc over the embedded bytes."""
    from styxx.certify import certify_doc
    try:
        doc_bytes = base64.b64decode(payload["document"]["b64"])
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / payload["document"]["name"]
            d.write_bytes(doc_bytes)
            rps = []
            for r in payload.get("receipts") or []:
                rp = Path(td) / r["name"]
                rp.write_bytes(base64.b64decode(r["b64"]))
                rps.append(rp)
            live = certify_doc(d, rps)
        return live["verdict"], dict(live.get("counts") or {})
    except Exception as e:                             # noqa: BLE001 - recorded on the line, never raised
        return None, {"live_error": type(e).__name__}


def _capsule_live_v02(payload: dict) -> Tuple[Optional[str], Optional[dict]]:
    from styxx.diffgate import gate_diff_text
    try:
        summary = base64.b64decode(payload["summary"]["b64"]).decode("utf-8")
        diff = base64.b64decode(payload["diff"]["b64"]).decode("utf-8")
        g = gate_diff_text(summary, diff, run=None, strict=False).to_dict()
        counts = {"claims": len(g.get("claims") or []),
                  "contradicted": sum(1 for c in g.get("claims") or [] if c.get("verdict") == "CONTRADICTED"),
                  "uncheckable": sum(1 for c in g.get("claims") or [] if c.get("verdict") == "UNCHECKABLE")}
        return str(g.get("verdict")), counts
    except Exception as e:                             # noqa: BLE001
        return None, {"live_error": type(e).__name__}


def derive_capsule(path: Path, repo: Path) -> dict:
    """A capsule is a pure function of its own bytes; the repository supplies only the relative path."""
    from styxx import capsule as _cap
    from styxx.corpus_audit import verdict_class
    rel = _rel(path, repo)
    raw = path.read_bytes()
    raw_id = _content_sha256(raw)          # platform-stable identity of the file
    payload = _capsule_payload(path)
    if payload is None:
        return _unresolved("capsule-oath", path.name, rel, "no_capsule_payload", raw_id)
    spec = payload.get("spec")
    if spec not in (_cap.SPEC, _cap.SPEC_V02):
        return _unresolved("capsule-oath", path.name, rel, "unknown_capsule_spec: %r" % spec, raw_id)
    rep = _cap.verify_capsule(path)
    kind = "capsule-diffgate" if spec == _cap.SPEC_V02 else "capsule-oath"
    problems = [str(x) for x in (rep.get("problems") or [])]
    binding_broken = any(("!=" in p and "binding" in p) or "bytes !=" in p or "ambiguous" in p for p in problems)
    if spec == _cap.SPEC_V02:
        gate = payload.get("gate") or {}
        binding = payload.get("binding") or {}
        shas = sorted({v.get("value") for v in binding.values() if isinstance(v, dict) and v.get("value")})
        recorded = str(gate.get("verdict"))
    else:
        cert = payload.get("certificate") or {}
        shas = sorted({_sha256(base64.b64decode(r["b64"])) for r in payload.get("receipts") or [] if r.get("b64")})
        recorded = str(cert.get("verdict"))
    if binding_broken:
        out = _unresolved(kind, path.name, rel, "capsule_binding: " + problems[0][:160], raw_id,
                          recorded=recorded)
        out["receipts"] = {"n": len(shas), "sha256": shas, "cited": None, "vacuous": False}
        return out
    live, live_counts = (_capsule_live_v02(payload) if spec == _cap.SPEC_V02 else _capsule_live_v01(payload))
    if live is None:
        out = _unresolved(kind, path.name, rel, "capsule_live_failed: %s" % (live_counts or {}).get("live_error"),
                          raw_id, recorded=recorded)
        out["receipts"] = {"n": len(shas), "sha256": shas, "cited": None, "vacuous": False}
        return out
    counts = dict(live_counts or {})
    counts["recorded_verdict"] = recorded
    counts["problems"] = len(problems)
    counts["problems_detail"] = [p[:160] for p in problems[:3]]
    counts["pinned_verifier_version"] = (payload.get("verifier") or {}).get("styxx_version")
    klass = live if spec == _cap.SPEC_V02 else verdict_class(live)
    if spec == _cap.SPEC_V02 and live not in ("PASS", "FAIL"):
        klass = "UNRESOLVED"
    return {"kind": kind, "subject": {"name": path.name, "path": rel, "sha256": raw_id},
            "at": {"commit": None, "manifest_digest": None, "document_at_commit": None},
            "receipts": {"n": len(shas), "sha256": shas, "cited": None, "vacuous": False},
            "verifier": _verifier(kind),
            "verdict": live, "verdict_class": klass, "counts": counts, "floor": None, "rungs": None,
            "reproduced": bool(rep.get("ok"))}


def derive_certificate(cert_path: Path, repo: Path) -> dict:
    """An OATH certificate re-certified over the working-tree receipts, drift flagged, exactly as
    ``styxx.corpus_audit`` does it; the RESOLVED receipt bytes are hashed onto the line beside the
    CITED digests. Digest resolution at the issuing commit is a separate leg of the plan."""
    from styxx.corpus_audit import _doc_for, _resolve_receipts, audit_document
    doc = _doc_for(cert_path)
    rel = _rel(doc, repo)
    try:
        cert = json.loads(cert_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return _unresolved("oath-certificate", doc.name, rel, "certificate_unreadable: %s" % e)
    cited_list = sorted({v for v in (cert.get("receipts_sha256") or {}).values() if isinstance(v, str)})
    cited = {"n": len(cited_list), "sha256": cited_list}
    doc_sha = _content_sha256(doc.read_bytes()) if doc.exists() else None
    rec = audit_document(cert_path, search_root=repo)
    if rec.get("status") != "OK":
        return _unresolved("oath-certificate", doc.name, rel, str(rec.get("status")), doc_sha, cited=cited,
                           recorded=cert.get("verdict"))
    resolved_paths, _missing, _drift = _resolve_receipts(cert_path, cert, repo)
    resolved = sorted({_content_sha256(Path(p).read_bytes()) for p in resolved_paths if Path(p).exists()})
    counts = dict(rec.get("counts") or {})
    counts["recorded_verdict"] = cert.get("verdict")
    counts["pinned_verifier_sha256"] = cert.get("verifier_sha256")
    counts["receipt_drift"] = rec.get("receipt_drift")
    counts["receipt_changed"] = list(rec.get("receipt_changed") or [])
    counts["missing_receipts"] = list(rec.get("missing_receipts") or [])
    counts["incomplete_receipts"] = bool(rec.get("incomplete_receipts"))
    counts["verdict_changed"] = bool(rec.get("verdict_changed"))
    counts["uncovered"] = rec.get("uncovered")
    reproduced = (not rec.get("verdict_changed")) and not rec.get("incomplete_receipts") \
        and not rec.get("receipt_changed") and not rec.get("receipt_drift") and not rec.get("missing_receipts")
    return {"kind": "oath-certificate", "subject": {"name": doc.name, "path": rel, "sha256": doc_sha},
            "at": {"commit": None, "manifest_digest": None, "document_at_commit": None},
            "receipts": {"n": len(resolved), "sha256": resolved, "cited": cited, "vacuous": False},
            "verifier": _verifier("oath-certificate"),
            "verdict": rec["live_verdict"], "verdict_class": rec["live_verdict_class"],
            "counts": counts, "floor": None, "rungs": None, "reproduced": bool(reproduced)}


def derive(path: Path, repo: Path) -> dict:
    """Dispatch on the artifact's suffix. Anything else is refused by name."""
    path = Path(path)
    if not path.exists():
        raise SystemExit("REFUSED: no such artifact %s" % path)
    name = path.name
    if name.endswith(".sworn.json"):
        d = derive_sworn(path, repo)
    elif name.endswith(".capsule.html"):
        d = derive_capsule(path, repo)
    elif name.endswith(".certificate.json"):
        d = derive_certificate(path, repo)
    else:
        raise SystemExit("REFUSED: %s is not a sworn sidecar, a capsule or a certificate" % name)
    if not _ascii_keys(d["counts"]):
        raise SystemExit("REFUSED: a count key outside ASCII cannot be canonicalised (%s)" % name)
    return d


def _artifact_for(entry: dict, repo: Path) -> Path:
    """The artifact a logged entry re-derives from, by kind and suffix, never by guess."""
    rel = entry["subject"]["path"]
    p = repo / rel
    if entry["kind"] == "sworn" and p.name.endswith(".md"):
        return p.with_name(p.name[:-3] + ".sworn.json")
    if entry["kind"] == "oath-certificate" and p.name.endswith(".md"):
        return p.with_name(p.name[:-3] + ".certificate.json")
    return p


# ------------------------------------------------------------------------------------ the entry

def make_entry(derived: dict, seq: int, prev: str, timestamp: Optional[str] = None,
               note: Optional[str] = None) -> dict:
    core = {"schema": ENTRY_SCHEMA, "seq": seq, "prev": prev}
    for k in _CORE_KEYS:
        core[k] = derived[k]
    entry = dict(core)
    entry["entry_id"] = _sha256(jcs(core).encode("utf-8"))
    entry["timestamp"] = timestamp or _now()
    if note:
        entry["note"] = note
    return entry


def _core_of(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k not in _OUTSIDE_DIGEST}


def header_digest(header: dict) -> str:
    """Domain-separated so a header can never masquerade as an entry, and the line after it names it."""
    return _sha256(_HEADER_DOMAIN + jcs(header).encode("utf-8"))


# -------------------------------------------------------------------------------------- the log

def read_log(path: Path) -> Tuple[Optional[dict], List[dict], List[dict], dict]:
    """(header, entries, problems, file_facts). Never raises on content: an unparsable line is a
    problem keyed by its line number, and a BOM or CRLF is a fact reported, not a crash."""
    if not path.exists():
        raise SystemExit("REFUSED: no such log %s" % path)
    raw = path.read_bytes()
    facts = {"file_sha256": _sha256(raw), "bytes": len(raw),
             "eol": ("mixed" if (b"\r\n" in raw and raw.replace(b"\r\n", b"").count(b"\n")) else
                     "CRLF" if b"\r\n" in raw else "LF"),
             "bom": raw.startswith(b"\xef\xbb\xbf")}
    problems: List[dict] = []
    header = None
    entries: List[dict] = []
    if facts["bom"]:
        problems.append({"line": 1, "seq": None, "problem": "line 1 begins with a BOM"})
        raw = raw[3:]
    for i, line in enumerate(raw.decode("utf-8", errors="replace").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            problems.append({"line": i, "seq": None, "problem": "not JSON"})
            continue
        if not isinstance(obj, dict):
            problems.append({"line": i, "seq": None, "problem": "not a JSON object"})
            continue
        if header is None and not entries and obj.get("schema") == LOG_SCHEMA:
            header = obj
            continue
        obj["_line"] = i
        entries.append(obj)
    if header is None and entries:
        problems.append({"line": 1, "seq": None, "problem": "no header on line 1"})
    return header, entries, problems, facts


def check_chain(header: Optional[dict], entries: List[dict]) -> List[dict]:
    """Every TAMPER a log can carry, keyed by line: a missing key, an entry_id that does not
    re-derive, a prev that does not point at the line before (or at the header), a seq that is not
    dense from 1. Every line after the first break is UNVERIFIABLE_AFTER_BREAK."""
    problems: List[dict] = []
    prev = header_digest(header) if header is not None else None
    broken = False
    for i, e in enumerate(entries, start=1):
        ln = e.get("_line")
        if broken:
            problems.append({"line": ln, "seq": e.get("seq"), "problem": "UNVERIFIABLE_AFTER_BREAK"})
            continue
        missing = [k for k in _REQUIRED_ENTRY_KEYS if k not in e]
        if missing:
            problems.append({"line": ln, "seq": e.get("seq"), "problem": "missing keys %s" % missing})
            broken = True
            continue
        if e.get("schema") != ENTRY_SCHEMA:
            problems.append({"line": ln, "seq": e.get("seq"), "problem": "unknown entry schema %r" % e.get("schema")})
            broken = True
            continue
        core = {k: v for k, v in e.items() if k not in _OUTSIDE_DIGEST and k != "_line"}
        try:
            want = _sha256(jcs(core).encode("utf-8"))
        except (TypeError, ValueError):
            problems.append({"line": ln, "seq": e.get("seq"), "problem": "core cannot be canonicalised"})
            broken = True
            continue
        bad = []
        if e.get("entry_id") != want:
            bad.append("entry_id does not re-derive from the core")
        if e.get("seq") != i:
            bad.append("seq is not dense (expected %d)" % i)
        if e.get("prev") != prev:
            bad.append("prev does not name the previous line" if i > 1 else "prev does not name the header")
        if bad:
            problems.append({"line": ln, "seq": e.get("seq"), "problem": "; ".join(bad)})
            broken = True
            continue
        prev = e.get("entry_id")
    return problems


def head_of(entries: List[dict]) -> Optional[str]:
    """The last line's entry_id, or None. A malformed last line has no head to report and must
    not raise: a log that is not what it says it is is TAMPER, never a traceback."""
    return entries[-1].get("entry_id") if entries else None


def ingest(paths: List[Path], log: Path, repo: Path, name: str = "charon", population: Optional[str] = None,
           timestamp: Optional[str] = None) -> List[dict]:
    """Append one entry per artifact. Refuses a log that does not chain, a log with entries but no
    header, and any artifact outside the repository."""
    repo = Path(repo)
    if log.exists():
        header, entries, problems, _facts = read_log(log)
        problems += check_chain(header, entries)
        if problems:
            raise SystemExit("REFUSED: the log does not chain; %d problem(s), first: line %s — %s"
                             % (len(problems), problems[0]["line"], problems[0]["problem"]))
    else:
        header, entries = None, []
    new_lines: List[str] = []
    added: List[dict] = []
    if header is None:
        header = {"schema": LOG_SCHEMA, "name": name, "created": timestamp or _now(),
                  "population": population, "certifies": _CERTIFIES}
        new_lines.append(json.dumps(header, ensure_ascii=False))
    seq = len(entries)
    prev = head_of(entries) or header_digest(header)
    for p in paths:
        d = derive(Path(p), repo)
        seq += 1
        e = make_entry(d, seq, prev, timestamp=timestamp)
        prev = e["entry_id"]
        added.append(e)
        new_lines.append(json.dumps(e, ensure_ascii=False))
    _write_lines_lf(log, new_lines, append=log.exists())
    return added


def _well_formed(entry: dict) -> bool:
    """A line every summary and every renderer can read. A line that is not is counted as
    malformed and never indexed into — a log that is not what it says it is is TAMPER, and
    TAMPER must not be a traceback."""
    if not all(k in entry for k in _REQUIRED_ENTRY_KEYS):
        return False
    return (isinstance(entry.get("subject"), dict) and isinstance(entry.get("receipts"), dict)
            and isinstance(entry.get("verifier"), dict) and isinstance(entry.get("counts"), dict)
            and isinstance(entry.get("kind"), str) and isinstance(entry.get("seq"), int))


def _summaries(all_entries: List[dict]) -> dict:
    by_kind: Dict[str, int] = {}
    rep_by_kind: Dict[str, Dict[str, int]] = {}
    held_by_kind: Dict[str, Dict[str, int]] = {}
    subjects = set()
    ns = []
    entries = [e for e in all_entries if _well_formed(e)]
    malformed = len(all_entries) - len(entries)
    for e in entries:
        k = e["kind"]
        by_kind[k] = by_kind.get(k, 0) + 1
        d = rep_by_kind.setdefault(k, {"true": 0, "false": 0, "null": 0})
        d[{True: "true", False: "false"}.get(e["reproduced"], "null")] += 1
        subjects.add((k, e["subject"]["path"]))
        ns.append(e["receipts"]["n"])
        h = held_by_kind.setdefault(k, {"held": 0, "held_10_or_more": 0, "vacuous_excluded": 0})
        if e["verdict_class"] in HELD_CLASSES:
            if e["receipts"].get("vacuous"):
                h["vacuous_excluded"] += 1
            else:
                h["held"] += 1
                if e["receipts"]["n"] >= 10:
                    h["held_10_or_more"] += 1
    return {"entries": len(all_entries), "malformed_lines": malformed,
            "distinct_subjects": len(subjects), "by_kind": by_kind,
            "reproduced_at_ingest": rep_by_kind,
            "receipts_n": {"min": min(ns) if ns else None, "max": max(ns) if ns else None,
                           "entries_with_10_or_more": sum(1 for x in ns if x >= 10),
                           "held_total": sum(h["held"] for h in held_by_kind.values()),
                           "held_with_10_or_more": sum(h["held_10_or_more"] for h in held_by_kind.values()),
                           "by_kind": held_by_kind}}


def verify_log(log: Path, repo: Path, expect_head: Optional[str] = None) -> dict:
    """Re-derive every line under the installed verifier and name what happened to it."""
    repo = Path(repo)
    header, entries, problems, facts = read_log(log)
    chain = check_chain(header, entries)
    tampered_lines = {p["line"] for p in problems + chain}
    lines = []
    by_status: Dict[str, int] = {s: 0 for s in STATUSES}
    builds: Dict[str, set] = {}
    for e in entries:
        row = {"line": e.get("_line"), "seq": e.get("seq"), "kind": e.get("kind"),
               "name": (e.get("subject") or {}).get("name") if isinstance(e.get("subject"), dict) else None,
               "recorded_class": e.get("verdict_class"),
               "fresh_class": None, "status": None, "fields_changed": [], "subject_moved": None,
               "receipts_moved": None,
               "verifier_recorded": ((e.get("verifier") or {}).get("digest") or "")[:12]
                                    if isinstance(e.get("verifier"), dict) else "",
               "verifier_now": None}
        if e.get("_line") in tampered_lines or not _well_formed(e):
            row["status"] = "TAMPER"
        else:
            art = _artifact_for(e, repo)
            try:
                if not art.exists():
                    raise SystemExit("artifact_missing")
                fresh = derive(art, repo)
            except SystemExit as ex:
                fresh = _unresolved(e["kind"], e["subject"]["name"], e["subject"]["path"], str(ex.code))
            row["fresh_class"] = fresh["verdict_class"]
            row["verifier_now"] = fresh["verifier"]["digest"][:12]
            same_build = fresh["verifier"]["digest"] == e["verifier"]["digest"]
            changed = [k for k in _COMPARED_KEYS if jcs(fresh[k]) != jcs(e[k])]
            row["fields_changed"] = changed
            row["subject_moved"] = fresh["subject"]["sha256"] != e["subject"]["sha256"]
            row["receipts_moved"] = fresh["receipts"]["sha256"] != e["receipts"]["sha256"]
            if fresh["verdict_class"] == "UNRESOLVED" and e["verdict_class"] != "UNRESOLVED":
                row["status"] = "UNRESOLVED"
                row["reason"] = fresh["counts"].get("reason")
            elif not changed and same_build:
                row["status"] = "SAME_LINE"
            elif not changed:
                row["status"] = "MOVED_VERIFIER"
            elif not same_build:
                row["status"] = "SKEW"
            else:
                row["status"] = "DRIFT"
        by_status[row["status"]] += 1
        if isinstance(e.get("verifier"), dict):
            for pair in e["verifier"].get("modules") or []:
                if isinstance(pair, list) and len(pair) == 2:
                    builds.setdefault(pair[0], set()).add((pair[1] or "absent")[:12])
        lines.append(row)
    head = head_of(entries)
    head_matches: Optional[bool] = None
    if expect_head is not None:
        head_matches = (head == expect_head.lower())
        if not head_matches:
            by_status["HEAD_MISMATCH"] += 1
    rep = {"schema": VERIFY_SCHEMA, "log": log.name, "head": head, "head_expected": expect_head,
           "head_matches": head_matches, "file": facts, "header": header,
           "chain_broken_at_line": (min(p["line"] for p in chain if p["line"] is not None) if chain else None),
           "chain_problems": problems + chain, "by_status": by_status,
           "verifier_builds_recorded": {k: sorted(v) for k, v in builds.items()},
           "lines": lines, "certifies": _CERTIFIES,
           "note": ("SAME_LINE = the fresh core equals the recorded core under the same verifier build; "
                    "MOVED_VERIFIER = same core, the instrument's bytes moved; SKEW = the core moved and the "
                    "instrument's bytes moved; DRIFT = the core moved under the same build (the bytes changed); "
                    "UNRESOLVED = bytes unavailable now, never an accusation; HEAD_MISMATCH = the head is not "
                    "the one you expected — a truncated or rebuilt log; TAMPER = the chain broke. Without "
                    "--expect-head this report establishes internal consistency only.")}
    rep.update(_summaries(entries))
    return rep


# ------------------------------------------------------------------------------------- the page

_CSS = """
:root{--bg:#000;--fg:#00ff00;--cy:#00ffff;--dim:#0a5;--bad:#ff3355;--warn:#ffcc00;--mute:#6a6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:ui-monospace,Consolas,"Courier New",monospace;font-size:13px;line-height:1.5}
header{padding:20px 24px;border-bottom:1px solid var(--dim)}h1{margin:0;font-size:18px;letter-spacing:.08em;color:var(--cy)}
.head{color:var(--mute);font-size:12px;margin-top:6px;word-break:break-all}
main{max-width:1240px;margin:0 auto;padding:20px 24px}
.tot{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}.card{border:1px solid var(--dim);padding:10px 14px;min-width:120px}
.card b{display:block;font-size:20px;color:var(--cy)}.card span{color:var(--mute);font-size:11px;text-transform:uppercase;letter-spacing:.1em}
table{border-collapse:collapse;width:100%;margin-top:14px}th,td{border-bottom:1px solid #063;padding:5px 8px;text-align:left;vertical-align:top;font-size:12px}
th{color:var(--mute);font-weight:normal;letter-spacing:.1em;text-transform:uppercase;font-size:11px}
.v-HELD,.v-OATH-HELD,.v-PASS{color:var(--fg)}.v-FAILED,.v-OATH-FAILED,.v-FAIL{color:var(--bad)}.v-UNSWORN,.v-UNRESOLVED{color:var(--warn)}
.id{color:var(--mute);font-size:11px}.cmd{color:var(--cy);font-size:11px;white-space:nowrap}
.no{color:var(--bad)}.yes{color:var(--fg)}.null{color:var(--warn)}
.legend{color:var(--mute);font-size:12px;margin:8px 0;white-space:pre-wrap}
footer{border-top:1px solid var(--dim);margin-top:30px;padding:18px 24px;color:var(--mute);font-size:12px;white-space:pre-wrap;max-width:1240px;margin-left:auto;margin-right:auto}
"""


def render_page(log: Path, out: Path, verify_report: Optional[dict] = None, repo: Optional[Path] = None) -> Path:
    """A static page with no script and no external request: the log, shown. It proves nothing."""
    header, entries, problems, facts = read_log(log)
    esc = _html.escape
    log_rel = _rel(log, repo) if repo is not None else log.name
    status_by_line = {r["line"]: r for r in (verify_report or {}).get("lines", [])}
    s = _summaries(entries)
    cards = [("lines", s["entries"]), ("distinct subjects", s["distinct_subjects"])]
    rep = s["reproduced_at_ingest"]
    cards += [("reproduced at ingest", sum(v["true"] for v in rep.values())),
              ("not reproduced at ingest", sum(v["false"] for v in rep.values())),
              ("no recorded verdict", sum(v["null"] for v in rep.values()))]
    cards += [(k, v) for k, v in sorted(s["by_kind"].items())]
    cards += [("held lines", s["receipts_n"]["held_total"]), ("held on 10+ receipts", s["receipts_n"]["held_with_10_or_more"])]
    if verify_report:
        cards += [("verify " + k, v) for k, v in verify_report["by_status"].items() if v]
    rows = []
    for e in entries:
        if not _well_formed(e):
            rows.append("<tr><td>%s</td><td colspan=\"10\">MALFORMED LINE — not a charon entry; "
                        "see the verify report</td></tr>" % esc(str(e.get("seq"))))
            continue
        st = status_by_line.get(e.get("_line"), {}).get("status", "")
        rungs = ", ".join("%s=%d" % kv for kv in sorted((e.get("rungs") or {}).items())) or "—"
        art = _artifact_for(e, Path(".")).as_posix()
        cmd = "python -m styxx.charon derive --repo . %s" % esc(art)
        rp = {True: ("yes", "yes"), False: ("no", "no")}.get(e["reproduced"], ("null", "none recorded"))
        n_txt = "%d" % e["receipts"]["n"]
        if e["receipts"].get("cited"):
            n_txt += " (cited %d)" % e["receipts"]["cited"]["n"]
        if e["receipts"].get("vacuous"):
            n_txt += " vacuous"
        rows.append(
            "<tr><td>%d</td><td>%s</td><td>%s</td><td class=\"v-%s\">%s</td><td>%s</td><td>%s</td>"
            "<td class=\"%s\">%s</td><td class=\"id\">%s</td><td class=\"id\">%s</td><td>%s</td>"
            "<td class=\"cmd\">%s</td></tr>"
            % (e["seq"], esc(e["kind"]), esc(e["subject"]["name"]), esc(e["verdict_class"]),
               esc(str(e["verdict"])), n_txt, esc(rungs), rp[0], rp[1],
               esc((e["verifier"].get("digest") or "")[:12]), esc(e["entry_id"][:16]), esc(st), cmd))
    hdr = header or {}
    page = ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
            "<title>charon — %s</title><style>%s</style></head><body>"
            "<header><h1>charon · the ferry log</h1><div class=\"head\">log %s · name %s · created %s · head %s · %d lines · "
            "rendered offline, no script, no request</div><div class=\"head\">population: %s</div>"
            "<div class=\"head\">this log's header says: %s</div></header><main>"
            "<div class=\"tot\">%s</div>"
            "<div class=\"legend\">receipts: for an OATH line, the RESOLVED receipts the verdict was reproduced against "
            "(cited count in parentheses) — a larger set makes HELD strictly easier there; for a sworn line, the distinct "
            "receipts its spans named; for a diffgate line, the bindings (summary, diff, gate). reproduced at ingest: whether "
            "the artifact's own recorded verdict re-derived when the line was written. verify now: from the report this "
            "page was rendered with, if any.</div>"
            "<table><tr><th>seq</th><th>kind</th><th>subject</th><th>verdict</th><th>receipts</th>"
            "<th>rungs</th><th>reproduced at ingest</th><th>verifier</th><th>entry id</th>"
            "<th>verify now</th><th>re-derive this line</th></tr>%s</table></main>"
            "<footer>what this page shows: the lines of this log as rendered, with the first 16 hex of each entry id. "
            "it proves nothing. what it does not show: that any verdict is true; who wrote any line; that a receipt "
            "set was not chosen to pass. nothing here is signed. a rebuilt or truncated log is a different log with a "
            "different head, so re-derive from the repository root against the head you were given:\n\n"
            "  python -m styxx.charon verify --log %s --repo . --expect-head %s\n\n%s</footer></body></html>\n"
            % (esc(log.name), _CSS, esc(log_rel), esc(str(hdr.get("name"))), esc(str(hdr.get("created"))),
               esc(head_of(entries) or "none"), len(entries), esc(str(hdr.get("population"))),
               esc(str(hdr.get("certifies"))),
               "".join("<div class=\"card\"><b>%d</b><span>%s</span></div>" % (v, esc(k)) for k, v in cards),
               "".join(rows), esc(log_rel), esc(head_of(entries) or "none"), esc(_CERTIFIES)))
    low = page.lower()
    # no script, no inline handler, no javascript: URL, no external reference — asserted on the
    # bytes about to be written, because "no script, no request" is printed in the header
    if ("<script" in low or "javascript:" in low or _re.search(r"\son[a-z]+\s*=", low)
            or _re.search(r"\b(href|src)\s*=|url\(|@import|https?://", low)):
        raise SystemExit("REFUSED: the page would carry a script, a handler or a request")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    return out


# -------------------------------------------------------------------------------------- the CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.charon", description="the ferry log — every verdict re-derived "
                                 "from bytes, chained; reproduces, never adjudicates")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="append one line per artifact (*.sworn.json, *.capsule.html, *.certificate.json)")
    i.add_argument("paths", nargs="+")
    i.add_argument("--log", required=True)
    i.add_argument("--repo", default=".", help="repository root; every artifact must be inside it")
    i.add_argument("--name", default="charon")
    i.add_argument("--population", default=None, help="how the artifacts were enumerated (written into the header)")
    v = sub.add_parser("verify", help="re-derive every line under the installed verifier")
    v.add_argument("--log", required=True)
    v.add_argument("--repo", default=".")
    v.add_argument("--out", default=None, help="write the verify report JSON here")
    v.add_argument("--expect-head", default=None, help="the head you were given outside the log")
    d = sub.add_parser("derive", help="print the core one line would carry for an artifact; touches no log")
    d.add_argument("path")
    d.add_argument("--repo", default=".")
    p = sub.add_parser("page", help="render the log as one static page")
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=".")
    p.add_argument("--report", default=None, help="a verify report to print beside each line")
    s = sub.add_parser("status", help="head and totals")
    s.add_argument("--log", required=True)
    s.add_argument("--expect-head", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "derive":
        core = derive(Path(a.path), Path(a.repo))
        print(json.dumps(core, indent=1, ensure_ascii=False))
        return 0
    log = Path(a.log)
    if a.cmd == "ingest":
        added = ingest([Path(x) for x in a.paths], log, Path(a.repo), name=a.name, population=a.population)
        for e in added:
            print("%4d %-17s %-58s %-12s receipts=%-3d reproduced=%-5s %s"
                  % (e["seq"], e["kind"], e["subject"]["name"][:58], e["verdict_class"], e["receipts"]["n"],
                     {True: "yes", False: "no"}.get(e["reproduced"], "null"), e["entry_id"][:12]))
        _h, entries, _p, _f = read_log(log)
        print("head %s  (%d lines)" % (head_of(entries), len(entries)))
        return 0
    if a.cmd == "verify":
        rep = verify_log(log, Path(a.repo), expect_head=a.expect_head)
        print("charon verify %s: %d lines (%d distinct subjects)  head %s%s"
              % (log.name, rep["entries"], rep["distinct_subjects"], (rep["head"] or "none")[:16],
                 "" if rep["head_matches"] is None else ("  expected head %s" % ("MATCHES" if rep["head_matches"] else "MISMATCH"))))
        print("  " + "  ".join("%s=%d" % kv for kv in rep["by_status"].items()))
        print("  " + "  ".join("%s=%d" % kv for kv in sorted(rep["by_kind"].items())))
        print("  eol=%s file_sha256=%s" % (rep["file"]["eol"], rep["file"]["file_sha256"][:16]))
        for r in rep["lines"]:
            if r["status"] != "SAME_LINE":
                print("  %-14s line %-4s seq %-4s %-17s %-46s recorded=%s now=%s changed=%s %s"
                      % (r["status"], r["line"], r["seq"], r["kind"], (r["name"] or "")[:46], r["recorded_class"],
                         r["fresh_class"], ",".join(r["fields_changed"]) or "-", r.get("reason", "")))
        if a.out:
            with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
            print("report -> %s" % a.out)
        return 1 if (rep["by_status"]["TAMPER"] or rep["by_status"]["HEAD_MISMATCH"]) else 0
    if a.cmd == "page":
        rep = json.loads(Path(a.report).read_text(encoding="utf-8")) if a.report else None
        out = render_page(log, Path(a.out), rep, repo=Path(a.repo))
        print("page -> %s" % out)
        return 0
    if a.cmd == "status":
        header, entries, problems, facts = read_log(log)
        problems = problems + check_chain(header, entries)
        head = head_of(entries)
        mism = a.expect_head is not None and head != a.expect_head.lower()
        print("%s: %d lines, head %s, chain %s%s, eol %s" % (log.name, len(entries), (head or "none")[:16],
                                                            "ok" if not problems else "%d problem(s)" % len(problems),
                                                            ", HEAD MISMATCH" if mism else "", facts["eol"]))
        return 1 if (problems or mism) else 0
    return 2                                                   # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
