# -*- coding: utf-8 -*-
"""styxx.charon — the ferry log: every verdict this lab issued, re-derived from bytes, chained.

Spec: ``papers/charon/SPEC_charon_v01_2026_09_02.md``, frozen before this file existed. Log
schema ``styxx.charon/log/v0``; entry schema ``styxx.charon/entry/v0``.

WHAT IT IS. An append-only, hash-chained log in which every line is a verdict that was
RE-DERIVED from bytes when the line was written, over the three verdict-bearing artifacts this
lab produces: a sworn document at a named commit (``styxx.sworn``), a capsule (``styxx.capsule``),
an OATH certificate with its receipts (``styxx.corpus_audit`` → ``styxx.certify``). Charon calls
those verifiers and writes what they return. It adjudicates nothing, accuses nobody, fetches
nothing, signs nothing.

THREE THINGS THE LINE CARRIES THAT NO CORPUS AUDIT CARRIED BEFORE.
  * ``receipts.n`` and every receipt digest — so a HELD bought by supplying a larger receipt set
    (the 2026-09-01 dogfood: FAILED → HELD on fixed document bytes as receipts were added) is
    visible in the log rather than hidden in it.
  * ``verifier.module_sha256`` — so ``verify`` can tell SKEW (the verdict moved AND the verifier's
    bytes moved: the instrument changed) from DRIFT (the verdict moved under the same verifier
    build: the bytes changed). Every corpus audit before this called both "drift".
  * ``prev`` — the previous line's ``entry_id``, so a line cannot be removed or reordered without
    the chain saying so (TAMPER).

DECISIONS the spec left open, stated once here:
  * ``subject.path`` is the DOCUMENT's repository-relative POSIX path (the .md for sworn and OATH
    entries; the .capsule.html for capsules); re-derivation maps it back to the sidecar or the
    certificate by suffix.
  * ``counts`` is the verifier's own counts dict plus ``recorded_verdict`` (what the artifact
    itself claimed) and, for OATH entries, the auditor's drift flags. ``reproduced`` is the
    boolean that compares them.
  * A sworn entry whose commit is not in the repository, a certificate whose document or
    receipts are gone: written as ``verdict UNRESOLVED`` with the reason in
    ``counts.reason`` — never skipped.
  * ``verdict_class``: sworn → HELD/FAILED/UNSWORN; OATH → ``corpus_audit.verdict_class``;
    diffgate capsule → PASS/FAIL; UNRESOLVED stays UNRESOLVED.
  * Timestamps are outside every digest. Two ingests of one artifact at the same ``seq``/``prev``
    produce one ``entry_id``.
  * Exit status: 0 for every verify outcome except TAMPER (the log is not what it says it is).

WHAT IT DOES NOT SAY, printed on the page and in every report: that any verdict is true; who
wrote any line; that a receipt set was not chosen to pass; that anything is "immutable",
"tamper-proof" or "self-verifying" — the log is append-only by contract and re-derivable by
construction, and a chain of hashes is exactly that.

CLI::

    python -m styxx.charon ingest --log L [--repo R] PATH...   # *.sworn.json, *.capsule.html, *.certificate.json
    python -m styxx.charon verify --log L [--repo R] [--out REPORT.json]
    python -m styxx.charon page   --log L --out index.html
    python -m styxx.charon status --log L
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import html as _html
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from styxx.attestation import jcs

__all__ = ["LOG_SCHEMA", "ENTRY_SCHEMA", "KINDS", "STATUSES", "derive", "make_entry",
           "read_log", "check_chain", "ingest", "verify_log", "render_page", "main"]

LOG_SCHEMA = "styxx.charon/log/v0"
ENTRY_SCHEMA = "styxx.charon/entry/v0"
VERIFY_SCHEMA = "styxx.charon/verify/v0"
KINDS = ("sworn", "capsule-oath", "capsule-diffgate", "oath-certificate")
STATUSES = ("REPRODUCED", "MOVED_VERIFIER", "SKEW", "DRIFT", "UNRESOLVED", "TAMPER")
_OUTSIDE_DIGEST = ("entry_id", "timestamp", "note")
_CERTIFIES = ("every line is a verdict re-derived from bytes by the named verifier at the named "
              "commit, chained to the line before it, with the size of the receipt set it was "
              "reproduced against — NOT a claim that any verdict is true, NOT a record of who wrote "
              "a line, NOT a check that a receipt set was not chosen to pass, and nothing signed")


# ---------------------------------------------------------------------------------------- utilities

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _posix(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def _module_sha256(module_name: str) -> str:
    """The verifier's own bytes, so a moved verifier is a fact the log can state."""
    import importlib
    mod = importlib.import_module(module_name)
    return _sha256(Path(mod.__file__).read_bytes())


def _version() -> str:
    from styxx._version import __version__
    return __version__


def _write_lines_lf(path: Path, lines: List[str], append: bool) -> None:
    with open(path, "a" if append else "w", encoding="utf-8", newline="\n") as fh:
        for ln in lines:
            fh.write(ln + "\n")


# ------------------------------------------------------------------------------ the three derivers

def _unresolved(kind: str, name: str, rel: str, reason: str, module: str, sha: Optional[str] = None) -> dict:
    return {"kind": kind, "subject": {"name": name, "path": rel, "sha256": sha},
            "at": {"commit": None, "manifest_digest": None},
            "receipts": {"n": 0, "sha256": []},
            "verifier": {"styxx_version": _version(), "module": module,
                         "module_sha256": _module_sha256(module)},
            "verdict": "UNRESOLVED", "verdict_class": "UNRESOLVED",
            "counts": {"reason": reason}, "floor": None, "rungs": None, "reproduced": False}


def derive_sworn(sidecar_path: Path, repo: Optional[Path]) -> dict:
    """A sworn document at the commit its sidecar names. Never raises on content."""
    from styxx import sworn as _sworn
    root = repo or sidecar_path.parent
    stem = sidecar_path.name[: -len(".sworn.json")]
    doc_rel = _posix(sidecar_path.with_name(stem + ".md"), root)
    try:
        side = _sworn.load_sidecar(json.loads(sidecar_path.read_text(encoding="utf-8")))
    except (SystemExit, OSError, ValueError) as e:
        return _unresolved("sworn", stem + ".md", doc_rel, "sidecar_refused: %s" % getattr(e, "code", e),
                           "styxx.sworn")
    inline = _sworn.render(side)
    commit = side["commit"]
    tree = None
    if commit is not None:
        if repo is None:
            return _unresolved("sworn", stem + ".md", doc_rel, "no_repository", "styxx.sworn", _sha256(inline))
        tree = _sworn.GitTree(repo, commit)
        why = tree._ready()
        if why:
            return _unresolved("sworn", stem + ".md", doc_rel, why, "styxx.sworn", _sha256(inline))
    try:
        core = _sworn.verify(sidecar=side, name=stem + ".md", tree=tree, commit=commit)
    except SystemExit as e:
        return _unresolved("sworn", stem + ".md", doc_rel, "verify_refused: %s" % e.code, "styxx.sworn",
                           _sha256(inline))
    shas = set()
    for s in core["spans"]:
        if s.get("resolved_sha256"):
            shas.add(s["resolved_sha256"])
    for entry in (side.get("manifest") or {}).get("receipts", {}).values():
        if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
            shas.add(entry["sha256"])
    recorded = None
    rec_path = sidecar_path.with_name(stem + ".sworn-receipt.json")
    if rec_path.exists():
        try:
            recorded = json.loads(rec_path.read_text(encoding="utf-8")).get("document_verdict")
        except (OSError, ValueError):
            recorded = None
    verdict = core["document_verdict"]
    klass = {"SWORN-HELD": "HELD", "SWORN-FAILED": "FAILED", "UNSWORN": "UNSWORN"}.get(verdict, verdict)
    counts = dict(core["counts"])
    counts["recorded_verdict"] = recorded
    counts["unresolved_spans"] = core["unresolved"]
    return {"kind": "sworn", "subject": {"name": stem + ".md", "path": doc_rel, "sha256": _sha256(inline)},
            "at": {"commit": commit, "manifest_digest": core.get("manifest_digest")},
            "receipts": {"n": len(shas), "sha256": sorted(shas)},
            "verifier": {"styxx_version": _version(), "module": "styxx.sworn",
                         "module_sha256": _module_sha256("styxx.sworn")},
            "verdict": verdict, "verdict_class": klass, "counts": counts,
            "floor": core["coverage"].get("sentence_share"), "rungs": core.get("rungs"),
            "reproduced": (recorded is not None and recorded == verdict)}


def _capsule_payload(path: Path) -> Optional[dict]:
    from styxx import capsule as _cap
    html = path.read_text(encoding="utf-8")
    try:
        i = html.index(_cap._BEGIN) + len(_cap._BEGIN)
        j = html.index(_cap._END, i)
        return json.loads(html[i:j])
    except (ValueError, json.JSONDecodeError):
        return None


def derive_capsule(path: Path, repo: Optional[Path]) -> dict:
    """A capsule is a pure function of its own bytes; the repository is not consulted."""
    from styxx import capsule as _cap
    from styxx.corpus_audit import verdict_class
    root = repo or path.parent
    rel = _posix(path, root)
    raw = path.read_bytes()
    payload = _capsule_payload(path)
    if payload is None:
        return _unresolved("capsule-oath", path.name, rel, "no_capsule_payload", "styxx.capsule", _sha256(raw))
    rep = _cap.verify_capsule(path)
    spec = payload.get("spec")
    if spec == _cap.SPEC_V02:
        gate = payload.get("gate") or {}
        binding = payload.get("binding") or {}
        shas = sorted({v.get("value") for v in binding.values() if isinstance(v, dict) and v.get("value")})
        verdict = str(gate.get("verdict"))
        counts = {"claims": len(gate.get("claims") or []),
                  "contradicted": sum(1 for c in gate.get("claims") or [] if c.get("verdict") == "CONTRADICTED"),
                  "uncheckable": sum(1 for c in gate.get("claims") or [] if c.get("verdict") == "UNCHECKABLE"),
                  "recorded_verdict": verdict, "problems": len(rep.get("problems") or []),
                  "pinned_verifier": (payload.get("verifier") or {}).get("styxx_version"),
                  "problems_detail": [str(x)[:160] for x in (rep.get("problems") or [])[:3]]}
        return {"kind": "capsule-diffgate", "subject": {"name": path.name, "path": rel, "sha256": _sha256(raw)},
                "at": {"commit": None, "manifest_digest": None},
                "receipts": {"n": len(shas), "sha256": shas},
                "verifier": {"styxx_version": _version(), "module": "styxx.capsule",
                             "module_sha256": _module_sha256("styxx.capsule")},
                "verdict": verdict, "verdict_class": verdict if verdict in ("PASS", "FAIL") else "UNRESOLVED",
                "counts": counts, "floor": None, "rungs": None, "reproduced": bool(rep.get("ok"))}
    cert = payload.get("certificate") or {}
    shas = sorted({_sha256(base64.b64decode(r["b64"])) for r in payload.get("receipts") or [] if r.get("b64")})
    verdict = str(cert.get("verdict"))
    counts = dict(cert.get("counts") or {})
    counts["recorded_verdict"] = verdict
    counts["problems"] = len(rep.get("problems") or [])
    # the verifier build the capsule itself pins, and the first reasons it did not reproduce under
    # the installed one — so a capsule minted under an older verifier reads as what it is
    counts["pinned_verifier"] = (payload.get("verifier") or {}).get("styxx_version")
    counts["problems_detail"] = [str(x)[:160] for x in (rep.get("problems") or [])[:3]]
    return {"kind": "capsule-oath", "subject": {"name": path.name, "path": rel, "sha256": _sha256(raw)},
            "at": {"commit": None, "manifest_digest": None},
            "receipts": {"n": len(shas), "sha256": shas},
            "verifier": {"styxx_version": _version(), "module": "styxx.capsule",
                         "module_sha256": _module_sha256("styxx.capsule")},
            "verdict": verdict, "verdict_class": verdict_class(verdict) if spec == _cap.SPEC else "UNRESOLVED",
            "counts": counts, "floor": None, "rungs": None, "reproduced": bool(rep.get("ok"))}


def derive_certificate(cert_path: Path, repo: Optional[Path]) -> dict:
    """An OATH certificate re-certified over the working-tree receipts, drift flagged, exactly as
    ``styxx.corpus_audit`` does it. Digest resolution at the issuing commit is a separate leg."""
    from styxx.corpus_audit import _doc_for, audit_document
    root = repo or cert_path.parent
    doc = _doc_for(cert_path)
    rel = _posix(doc, root)
    try:
        cert = json.loads(cert_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return _unresolved("oath-certificate", doc.name, rel, "certificate_unreadable: %s" % e, "styxx.certify")
    rec = audit_document(cert_path, search_root=repo)
    doc_sha = _sha256(doc.read_bytes()) if doc.exists() else None
    shas = sorted({v for v in (cert.get("receipts_sha256") or {}).values() if isinstance(v, str)})
    if rec.get("status") != "OK":
        out = _unresolved("oath-certificate", doc.name, rel, str(rec.get("status")), "styxx.certify", doc_sha)
        out["receipts"] = {"n": len(shas), "sha256": shas}
        out["counts"]["recorded_verdict"] = cert.get("verdict")
        return out
    counts = dict(rec.get("counts") or {})
    counts["recorded_verdict"] = cert.get("verdict")
    counts["receipt_drift"] = rec.get("receipt_drift")
    counts["receipt_changed"] = list(rec.get("receipt_changed") or [])
    counts["incomplete_receipts"] = bool(rec.get("incomplete_receipts"))
    counts["uncovered"] = rec.get("uncovered")
    reproduced = (not rec.get("verdict_changed")) and not rec.get("incomplete_receipts") \
        and not rec.get("receipt_changed") and not rec.get("receipt_drift")
    return {"kind": "oath-certificate", "subject": {"name": doc.name, "path": rel, "sha256": doc_sha},
            "at": {"commit": None, "manifest_digest": None},
            "receipts": {"n": len(shas), "sha256": shas},
            "verifier": {"styxx_version": _version(), "module": "styxx.certify",
                         "module_sha256": _module_sha256("styxx.certify")},
            "verdict": rec["live_verdict"], "verdict_class": rec["live_verdict_class"],
            "counts": counts, "floor": None, "rungs": None, "reproduced": bool(reproduced)}


def derive(path: Path, repo: Optional[Path]) -> dict:
    """Dispatch on the artifact's suffix. Anything else is refused by name."""
    name = path.name
    if name.endswith(".sworn.json"):
        return derive_sworn(path, repo)
    if name.endswith(".capsule.html"):
        return derive_capsule(path, repo)
    if name.endswith(".certificate.json"):
        return derive_certificate(path, repo)
    raise SystemExit("REFUSED: %s is not a sworn sidecar, a capsule or a certificate" % name)


def _artifact_for(entry: dict, repo: Path) -> Path:
    """The artifact a logged entry re-derives from, by kind and suffix, never by guess."""
    rel = entry["subject"]["path"]
    kind = entry["kind"]
    p = repo / rel
    if kind == "sworn":
        return p.with_name(p.name[:-3] + ".sworn.json") if p.name.endswith(".md") else p
    if kind == "oath-certificate":
        return p.with_name(p.name[:-3] + ".certificate.json") if p.name.endswith(".md") else p
    return p


# ------------------------------------------------------------------------------------ the entry

def make_entry(derived: dict, seq: int, prev: Optional[str], timestamp: Optional[str] = None,
               note: Optional[str] = None) -> dict:
    core = {"schema": ENTRY_SCHEMA, "seq": seq, "prev": prev}
    for k in ("kind", "subject", "at", "receipts", "verifier", "verdict", "verdict_class", "counts",
              "floor", "rungs", "reproduced"):
        core[k] = derived[k]
    entry = dict(core)
    entry["entry_id"] = _sha256(jcs(core).encode("utf-8"))
    entry["timestamp"] = timestamp or _now()
    if note:
        entry["note"] = note
    return entry


def _core_of(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k not in _OUTSIDE_DIGEST}


# -------------------------------------------------------------------------------------- the log

def read_log(path: Path) -> Tuple[Optional[dict], List[dict]]:
    if not path.exists():
        return None, []
    header = None
    entries: List[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        obj = json.loads(line)
        if i == 0 and obj.get("schema") == LOG_SCHEMA:
            header = obj
            continue
        entries.append(obj)
    return header, entries


def check_chain(entries: List[dict]) -> List[dict]:
    """Every TAMPER a log can carry: an entry_id that does not re-derive, a prev that does not point
    at the line before, a seq that is not dense from 1."""
    problems = []
    prev = None
    for i, e in enumerate(entries, start=1):
        want = _sha256(jcs(_core_of(e)).encode("utf-8"))
        if e.get("entry_id") != want:
            problems.append({"seq": e.get("seq"), "problem": "entry_id does not re-derive from the core"})
        if e.get("seq") != i:
            problems.append({"seq": e.get("seq"), "problem": "seq is not dense (expected %d)" % i})
        if e.get("prev") != prev:
            problems.append({"seq": e.get("seq"), "problem": "prev does not name the previous entry"})
        if e.get("schema") != ENTRY_SCHEMA:
            problems.append({"seq": e.get("seq"), "problem": "unknown entry schema %r" % e.get("schema")})
        prev = e.get("entry_id")
    return problems


def head_of(entries: List[dict]) -> Optional[str]:
    return entries[-1]["entry_id"] if entries else None


def ingest(paths: List[Path], log: Path, repo: Optional[Path], name: str = "charon",
           timestamp: Optional[str] = None) -> List[dict]:
    """Append one entry per artifact. Refuses to touch a log whose existing lines do not chain."""
    header, entries = read_log(log)
    problems = check_chain(entries)
    if problems:
        raise SystemExit("REFUSED: the log does not chain; %d problem(s), first: seq %s — %s"
                         % (len(problems), problems[0]["seq"], problems[0]["problem"]))
    new_lines: List[str] = []
    added: List[dict] = []
    if header is None:
        header = {"schema": LOG_SCHEMA, "name": name, "created": timestamp or _now(),
                  "certifies": _CERTIFIES}
        new_lines.append(json.dumps(header, ensure_ascii=False))
    seq = len(entries)
    prev = head_of(entries)
    for p in paths:
        d = derive(Path(p), repo)
        seq += 1
        e = make_entry(d, seq, prev, timestamp=timestamp)
        prev = e["entry_id"]
        added.append(e)
        new_lines.append(json.dumps(e, ensure_ascii=False))
    _write_lines_lf(log, new_lines, append=log.exists())
    return added


def verify_log(log: Path, repo: Optional[Path]) -> dict:
    """Re-derive every line under the installed verifier and name what happened to it."""
    header, entries = read_log(log)
    tamper = check_chain(entries)
    tampered = {p["seq"] for p in tamper}
    lines = []
    by_status: Dict[str, int] = {s: 0 for s in STATUSES}
    by_kind: Dict[str, int] = {}
    builds: Dict[str, set] = {}
    root = repo or log.parent
    for e in entries:
        kind = e["kind"]
        by_kind[kind] = by_kind.get(kind, 0) + 1
        row = {"seq": e["seq"], "kind": kind, "name": e["subject"]["name"],
               "recorded_class": e["verdict_class"], "fresh_class": None, "status": None,
               "verifier_recorded": e["verifier"]["module_sha256"][:12], "verifier_now": None}
        if e["seq"] in tampered:
            row["status"] = "TAMPER"
        else:
            art = _artifact_for(e, root)
            if not art.exists():
                fresh = _unresolved(kind, e["subject"]["name"], e["subject"]["path"], "artifact_missing",
                                    e["verifier"]["module"])
            else:
                try:
                    fresh = derive(art, repo)
                except SystemExit as ex:
                    fresh = _unresolved(kind, e["subject"]["name"], e["subject"]["path"],
                                        "refused: %s" % ex.code, e["verifier"]["module"])
            row["fresh_class"] = fresh["verdict_class"]
            row["verifier_now"] = fresh["verifier"]["module_sha256"][:12]
            same_build = fresh["verifier"]["module_sha256"] == e["verifier"]["module_sha256"]
            # the line reproduces only if BOTH the verdict class and the artifact's own
            # reproducibility (reproduced at ingest) come back the same: a capsule that verified
            # when it was logged and does not verify now has moved, whatever its badge says
            same_class = (fresh["verdict_class"] == e["verdict_class"]
                          and bool(fresh["reproduced"]) == bool(e["reproduced"]))
            row["fresh_reproduced"] = bool(fresh["reproduced"])
            if fresh["verdict_class"] == "UNRESOLVED":
                row["status"] = "UNRESOLVED"
                row["reason"] = fresh["counts"].get("reason")
            elif same_class and same_build:
                row["status"] = "REPRODUCED"
            elif same_class:
                row["status"] = "MOVED_VERIFIER"
            elif not same_build:
                row["status"] = "SKEW"
            else:
                row["status"] = "DRIFT"
        by_status[row["status"]] += 1
        builds.setdefault(e["verifier"]["module"], set()).add(e["verifier"]["module_sha256"][:12])
        lines.append(row)
    return {"schema": VERIFY_SCHEMA, "log": log.name, "head": head_of(entries), "entries": len(entries),
            "chain_problems": tamper, "by_status": by_status, "by_kind": by_kind,
            "verifier_builds_recorded": {k: sorted(v) for k, v in builds.items()},
            "lines": lines, "certifies": _CERTIFIES,
            "note": ("SKEW = the verdict moved and the verifier's bytes moved (the instrument changed); "
                     "DRIFT = the verdict moved under the same verifier build (the bytes changed); "
                     "neither is an accusation and neither changes any line")}


# ------------------------------------------------------------------------------------- the page

_CSS = """
:root{--bg:#000;--fg:#00ff00;--cy:#00ffff;--dim:#0a5;--bad:#ff3355;--warn:#ffcc00;--mute:#6a6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:ui-monospace,Consolas,"Courier New",monospace;font-size:13px;line-height:1.5}
header{padding:20px 24px;border-bottom:1px solid var(--dim)}h1{margin:0;font-size:18px;letter-spacing:.08em;color:var(--cy)}
.head{color:var(--mute);font-size:12px;margin-top:6px;word-break:break-all}
main{max-width:1200px;margin:0 auto;padding:20px 24px}
.tot{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}.card{border:1px solid var(--dim);padding:10px 14px;min-width:120px}
.card b{display:block;font-size:20px;color:var(--cy)}.card span{color:var(--mute);font-size:11px;text-transform:uppercase;letter-spacing:.1em}
table{border-collapse:collapse;width:100%;margin-top:14px}th,td{border-bottom:1px solid #063;padding:5px 8px;text-align:left;vertical-align:top;font-size:12px}
th{color:var(--mute);font-weight:normal;letter-spacing:.1em;text-transform:uppercase;font-size:11px}
.v-HELD,.v-PASS{color:var(--fg)}.v-FAILED,.v-FAIL{color:var(--bad)}.v-UNSWORN,.v-UNRESOLVED{color:var(--warn)}
.id{color:var(--mute);font-size:11px}.cmd{color:var(--cy);font-size:11px;white-space:nowrap}
.no{color:var(--bad)}.yes{color:var(--fg)}
footer{border-top:1px solid var(--dim);margin-top:30px;padding:18px 24px;color:var(--mute);font-size:12px;white-space:pre-wrap;max-width:1200px;margin-left:auto;margin-right:auto}
"""


def render_page(log: Path, out: Path, verify_report: Optional[dict] = None) -> Path:
    """A static page with no script and no external request: the log, printed."""
    header, entries = read_log(log)
    esc = _html.escape
    status_by_seq = {r["seq"]: r for r in (verify_report or {}).get("lines", [])}
    by_class: Dict[str, int] = {}
    by_kind: Dict[str, int] = {}
    reproduced = 0
    for e in entries:
        by_class[e["verdict_class"]] = by_class.get(e["verdict_class"], 0) + 1
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        reproduced += 1 if e["reproduced"] else 0
    cards = [("entries", len(entries)), ("reproduced at ingest", reproduced)]
    cards += [("class " + k, v) for k, v in sorted(by_class.items())]
    cards += [(k, v) for k, v in sorted(by_kind.items())]
    if verify_report:
        cards += [("verify " + k, v) for k, v in verify_report["by_status"].items() if v]
    rows = []
    for e in entries:
        st = status_by_seq.get(e["seq"], {}).get("status", "")
        rungs = ", ".join("%s=%d" % kv for kv in sorted((e.get("rungs") or {}).items())) or "—"
        if e["kind"] == "sworn":
            cmd = "python -m styxx.sworn verify %s --repo . " % esc(e["subject"]["path"][:-3] + ".sworn.json")
        elif e["kind"].startswith("capsule"):
            cmd = "python -m styxx.capsule verify %s" % esc(e["subject"]["path"])
        else:
            cmd = "python -m styxx.corpus_audit %s" % esc(str(Path(e["subject"]["path"]).parent).replace("\\", "/"))
        rows.append(
            "<tr><td>%d</td><td>%s</td><td>%s</td><td class=\"v-%s\">%s</td><td>%d</td><td>%s</td>"
            "<td class=\"%s\">%s</td><td class=\"id\">%s</td><td class=\"id\">%s</td><td>%s</td>"
            "<td class=\"cmd\">%s</td></tr>"
            % (e["seq"], esc(e["kind"]), esc(e["subject"]["name"]), esc(e["verdict_class"]),
               esc(str(e["verdict"])), e["receipts"]["n"], esc(rungs),
               "yes" if e["reproduced"] else "no", "yes" if e["reproduced"] else "no",
               esc(e["verifier"]["module_sha256"][:12]), esc(e["entry_id"][:16]), esc(st), cmd))
    page = ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
            "<title>charon — %s</title><style>%s</style></head><body>"
            "<header><h1>charon · the ferry log</h1><div class=\"head\">log %s · head %s · %d entries · "
            "rendered offline, no script, no request</div></header><main>"
            "<div class=\"tot\">%s</div>"
            "<table><tr><th>seq</th><th>kind</th><th>subject</th><th>verdict</th><th>receipts</th>"
            "<th>rungs</th><th>reproduced at ingest</th><th>verifier</th><th>entry id</th>"
            "<th>verify now</th><th>re-derive it yourself</th></tr>%s</table></main>"
            "<footer>%s\n\nwhat this page proves: that these lines are in this log, and that each line's "
            "entry id re-derives from its own content. what it does not prove: that any verdict is true; "
            "who wrote any line; that a receipt set was not chosen to pass — a larger receipt set makes "
            "HELD strictly easier, which is why receipts.n is a column. nothing here is signed. "
            "re-derive the whole log: python -m styxx.charon verify --log %s --repo .</footer></body></html>\n"
            % (esc(log.name), _CSS, esc(log.name), esc(head_of(entries) or "none"), len(entries),
               "".join("<div class=\"card\"><b>%d</b><span>%s</span></div>" % (v, esc(k)) for k, v in cards),
               "".join(rows), esc(_CERTIFIES), esc(log.name)))
    assert "<script" not in page.lower()
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    return out


# -------------------------------------------------------------------------------------- the CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.charon", description="the ferry log — every verdict re-derived "
                                 "from bytes, chained; reproduces, never adjudicates")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="append one entry per artifact (*.sworn.json, *.capsule.html, *.certificate.json)")
    i.add_argument("paths", nargs="+")
    i.add_argument("--log", required=True)
    i.add_argument("--repo", default=None, help="repository whose commits sworn sidecars name")
    i.add_argument("--name", default="charon")
    v = sub.add_parser("verify", help="re-derive every line under the installed verifier")
    v.add_argument("--log", required=True)
    v.add_argument("--repo", default=None)
    v.add_argument("--out", default=None, help="write the verify report JSON here")
    p = sub.add_parser("page", help="render the log as one static page")
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", default=None, help="a verify report to print beside each line")
    s = sub.add_parser("status", help="head and totals")
    s.add_argument("--log", required=True)
    a = ap.parse_args(argv)

    repo = Path(a.repo) if getattr(a, "repo", None) else None
    log = Path(a.log)
    if a.cmd == "ingest":
        added = ingest([Path(x) for x in a.paths], log, repo, name=a.name)
        for e in added:
            print("%4d %-17s %-58s %-12s receipts=%-3d reproduced=%s  %s"
                  % (e["seq"], e["kind"], e["subject"]["name"][:58], e["verdict_class"], e["receipts"]["n"],
                     "yes" if e["reproduced"] else "no", e["entry_id"][:12]))
        print("head %s  (%d entries)" % (head_of(read_log(log)[1]), len(read_log(log)[1])))
        return 0
    if a.cmd == "verify":
        rep = verify_log(log, repo)
        print("charon verify %s: %d entries  head %s" % (log.name, rep["entries"], (rep["head"] or "none")[:16]))
        print("  " + "  ".join("%s=%d" % kv for kv in rep["by_status"].items()))
        print("  " + "  ".join("%s=%d" % kv for kv in sorted(rep["by_kind"].items())))
        for r in rep["lines"]:
            if r["status"] not in ("REPRODUCED",):
                print("  %-14s seq %-4d %-17s %-50s recorded=%s now=%s %s"
                      % (r["status"], r["seq"], r["kind"], r["name"][:50], r["recorded_class"],
                         r["fresh_class"], r.get("reason", "")))
        if a.out:
            with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
            print("report -> %s" % a.out)
        return 1 if rep["by_status"]["TAMPER"] else 0
    if a.cmd == "page":
        rep = json.loads(Path(a.report).read_text(encoding="utf-8")) if a.report else None
        out = render_page(log, Path(a.out), rep)
        print("page -> %s" % out)
        return 0
    if a.cmd == "status":
        header, entries = read_log(log)
        problems = check_chain(entries)
        print("%s: %d entries, head %s, chain %s" % (log.name, len(entries), (head_of(entries) or "none")[:16],
                                                     "ok" if not problems else "%d problem(s)" % len(problems)))
        return 1 if problems else 0
    return 2                                                   # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
