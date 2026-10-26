# -*- coding: utf-8 -*-
"""styxx.prereg_ci.report: the report of a check, a ladder or a census (DESIGN 3.2(e), step B7).

``styxx.prereg-ci/report/v0`` is JSON plus a markdown job summary. Per pair it carries the seal (the
blob's sha256, the decision digest, the landing commit L, the add commit, the order commit, the
manifest line), the amendments in effect with ``seen``, the time ladder, the PREREG path's blob history
from the order commit to the receipt's commit (a print: an edit and revert on a branch never reaches a
per-pull-request check, RED_TEAM m8), the decision form, the re-score, the recorded and declared
verdicts, the outcome ladder with each gate's metric, op, bar and margin, the single-gate flip table,
the licence readings, every check's findings, and ``not_examined`` (C9, printed beside every PASS). The
scorer's identity is the sha256 of each module file that judges, with CRLF read as LF (the rule of
``styxx.charon``'s verifier scheme v1, re-read here from the module files rather than through charon's
private reader; a test pins the two equal), and the runtime's three compared fields
(``styxx.instrument.RUNTIME_COMPARED``). No absolute path, host name or ``sys.version`` string is
written: paths are repository-relative, a file named outside the work tree is named by its basename.

Exit codes (DESIGN 3.3, the audit-claims contract plus one): 0 PASS or nothing to check, 1 FAIL, 2 an
input error, 3 UNCHECKABLE. A refusal is read ahead of the other two: exit 1 says that something the
check read was refused, whatever else it could not read, and the report lists the rest.

Nothing here is public API (DESIGN 3.1); the CLI is this module's caller.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from styxx.prereg_ci import core as _core
from styxx.prereg_ci import git as _git
from styxx.prereg_ci import time as _time

_SCHEMA = "styxx.prereg-ci/report/v0"
_VERDICT = {0: "PASS", 1: "FAIL", 2: "INPUT_ERROR", 3: "UNCHECKABLE"}
_ORDER = ("FAIL", "INPUT", "UNCHECKABLE", "NOTE")
# the codes the commands add (step B7); the kinds of the rest are the tables of core, git and time
_CHECK_KIND = {
    "SEAL_NOT_BEFORE_RECEIPT": "FAIL",          # C3: the order commit is not a strict ancestor of R
    "LOCKED_FILE_MOVED": "NOTE",                # a locked file's blob at HEAD is not the seal's; the run read the seal's
    "CONFIG_INVALID": "INPUT",                  # prereg-ci.toml holds what its format does not name
    "CONFIG_CHANGED": "NOTE",                   # the pull request changes prereg-ci.toml; the base's governs it
    "CONFIG_REPAIRED": "NOTE",                  # the base's copy does not read; the pull request replaces it
    "CLOSE_AFTER_UNREAD": "NOTE",               # a sealed close_after the time reader does not read (check prints it)
    # a gates-like fence a renderer shows that the scanner does not count (core._shadowed; the B8 battery, M1):
    # read in the decision in effect (check, census) and in a draft (lint)
    "FENCE_SHADOWED": "FAIL",
    # lint (DESIGN 3.7), before data
    "TABLE_NOT_TOTAL": "FAIL", "ROW_UNREACHABLE": "FAIL", "VERDICTS_REPEATED": "FAIL", "GATE_VACUOUS": "FAIL",
    "POWER_BASIS_UNDECLARED": "NOTE", "ROW_KEY_UNREAD": "NOTE", "WITHDRAWS_KEY_MISSPELLED": "FAIL",
    "CLOSE_AFTER_INVALID": "FAIL",
}
# the modules whose bytes judge a pair: the package and the code it reuses from styxx
_SCORER_MODULES = ("styxx.prereg_ci.core", "styxx.prereg_ci.git", "styxx.prereg_ci.scorer_run",
                   "styxx.prereg_ci.time", "styxx.prereg_ci.report", "styxx.prereg_ci.cli", "styxx.protocol",
                   "styxx.research_capsule", "styxx.sworn")
_NOTES_SHOWN = 20                               # NOTE lines of one code printed before the rest are counted
_DROP = ("bytes", "out_bytes")                  # bytes a report never carries (their digests are beside them)


def _kinds() -> Dict[str, str]:
    """Every code's kind as the tables give it; a policy may still make a finding a FAIL or a NOTE
    (``receipt_rewrite``, ``immutability``, ``amend_after_data``, ``preregistered_word``,
    ``require_time``), and each finding carries the kind it was given."""
    out: Dict[str, str] = {}
    for table in (_core._KIND, _core._SEAL_KIND, _core._RESULT_KIND, _git._KIND, _time._KIND, _CHECK_KIND):
        for code, kind in table.items():
            if out.get(code, kind) != kind:
                raise AssertionError(f"{code} is a {out[code]} in one table and a {kind} in another")
            out[code] = kind
    return out


def _finding(code: str, detail: str, **extra: Any) -> Dict[str, Any]:
    return {"code": code, "kind": _kinds()[code], "detail": detail, **extra}


def _exit(findings: Iterable[Dict[str, Any]]) -> int:
    kinds = {f.get("kind") for f in findings}
    if "FAIL" in kinds:
        return 1
    if "INPUT" in kinds:
        return 2
    if "UNCHECKABLE" in kinds:
        return 3
    return 0


def _module_pair(name: str) -> Tuple[str, Optional[str]]:
    """The module's name and the sha256 of its file with CRLF read as LF; None when it does not import
    or its file does not read (charon's rule, ``styxx.charon``'s verifier scheme v1)."""
    try:
        mod = importlib.import_module(name)
        with open(mod.__file__, "rb") as fh:
            data = fh.read()
        return name, hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    except Exception:                               # noqa: BLE001 - an absent module is recorded as such
        return name, None


def _module_pairs(names: Sequence[str] = _SCORER_MODULES) -> List[Tuple[str, Optional[str]]]:
    return [_module_pair(n) for n in names]


def _runtime() -> Dict[str, Optional[str]]:
    from styxx import instrument
    rt = instrument.runtime()
    return {k: rt.get(k) for k in instrument.RUNTIME_COMPARED}


def _tool() -> Dict[str, Any]:
    from styxx._version import __version__
    from styxx.prereg_ci import VERSION
    return {"name": "prereg-ci", "version": VERSION, "styxx": __version__,
            "modules": [list(p) for p in _module_pairs()], "runtime": _runtime()}


def _jsonable(obj: Any) -> Any:
    """*obj* as JSON holds it: tuples as lists, sets as sorted lists, the ``bytes`` keys dropped (their
    digests sit beside them), any other bytes as their sha256 and size."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k not in _DROP}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_jsonable(v) for v in obj)
    if isinstance(obj, (bytes, bytearray)):
        return {"sha256": hashlib.sha256(bytes(obj)).hexdigest(), "size": len(obj)}
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    return obj


def _dumps(report: Dict[str, Any]) -> bytes:
    return (json.dumps(_jsonable(report), indent=1, ensure_ascii=True) + "\n").encode("ascii")


def _write(path: str, data: bytes, append: bool = False) -> None:
    """Write *data* (LF, no byte-order mark) and read it back; a file that does not read back is an
    OSError, never a silent pass."""
    before = b""
    if append:
        try:
            with open(path, "rb") as fh:
                before = fh.read()
        except FileNotFoundError:
            before = b""
    with open(path, "ab" if append else "wb") as fh:
        fh.write(data)
    with open(path, "rb") as fh:
        back = fh.read()
    if back != before + data:
        raise OSError(f"{path} did not read back as written")


def _num(x: Any) -> str:
    """A value as the printout shows it: four decimal places at most, ints as ints, JSON's words for
    true, false and null."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if x is None:
        return "null"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        r = round(x, 4)
        return repr(abs(r) if r == 0 else r)
    return str(x)


def _short(oid: Optional[str], n: int = 8) -> str:
    return oid[:n] if isinstance(oid, str) else "none"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ---- the printout ---------------------------------------------------------------------------------------

def _loc(f: Dict[str, Any]) -> Optional[str]:
    if f.get("line") is not None and f.get("path"):
        return f"{f['path']}:{f['line']}:{f['col']}"
    return None


def _finding_line(f: Dict[str, Any]) -> str:
    """``KIND CODE  detail``; a finding at a place in a file is ``KIND CODE path:line:col  detail``."""
    loc = _loc(f)
    detail = f.get("detail") or ""
    if loc is not None:
        if detail.startswith(loc + " "):
            detail = detail[len(loc) + 1:]
        return f"{f['kind']} {f['code']} {loc}  {detail}"
    return f"{f['kind']} {f['code']}  {detail}"


def _finding_lines(findings: Sequence[Dict[str, Any]], indent: str = "  ") -> List[str]:
    """Every finding but the NOTE lines past ``_NOTES_SHOWN`` of one code, which are counted (every one
    is in the JSON report)."""
    out: List[str] = []
    shown: Dict[str, int] = {}
    hidden: Dict[str, int] = {}
    for f in sorted(findings, key=lambda f: _ORDER.index(f["kind"]) if f["kind"] in _ORDER else len(_ORDER)):
        if f["kind"] == "NOTE":
            shown[f["code"]] = shown.get(f["code"], 0) + 1
            if shown[f["code"]] > _NOTES_SHOWN:
                hidden[f["code"]] = hidden.get(f["code"], 0) + 1
                continue
        out.append(indent + _finding_line(f))
    for code, n in hidden.items():
        out.append(f"{indent}NOTE {code}  and {n} more, in the JSON report")
    return out


def _gate_line(g: Dict[str, Any]) -> str:
    return f"{g['name']} {_num(g['observed'])} {g['op']} {_num(g['bar'])} (margin {_num(g['margin'])})"


def _ladder_lines(lad: Optional[Dict[str, Any]], indent: str = "  ") -> List[str]:
    if lad is None:
        return []
    out = [f"{indent}landed row {lad['landed'] + 1} of {len(lad['rows'])}: {lad['verdict']}"]
    out.append(f"{indent}gates:")
    for g in lad["gates"]:
        out.append(f"{indent}  {_gate_line(g)}: {'fired' if g['fired'] else 'not fired'}"
                   + (", read by the landed row" if g["read_by_landed"] else ""))
    out.append(f"{indent}rows:")
    for r in lad["rows"]:
        when = ", ".join(f"{k}={_num(v)}" for k, v in r["when"].items()) or "always"
        out.append(f"{indent}  {'*' if r['landed'] else ' '} {r['row'] + 1}. {r['verdict']} when {when}")
    by = {g["name"]: g for g in lad["gates"]}
    moving = [f for f in lad["flips"] if f["moves"]]
    still = [f["gate"] for f in lad["flips"] if not f["moves"]]
    out.append(f"{indent}flips that move the verdict ({len(moving)} of {_plural(len(lad['flips']), 'gate')}):")
    for f in moving:
        out.append(f"{indent}  {_gate_line(by[f['gate']])} -> {f['verdict'] or 'no row (a hole in the table)'}")
    out.append(f"{indent}flips that leave it: {', '.join(still) if still else 'none'}")
    return out


def _licence_lines(lic: Optional[Dict[str, Any]], indent: str = "  ") -> List[str]:
    if lic is None:
        return []
    out = [f"{indent}licences: surfaces read {lic['surfaces']}, not read {len(lic['skipped'])}"]
    for entry in lic["licences"]:
        n = len(entry["surfaces"])
        out.append(f"{indent}  licence {entry['id']}: {_plural(n, 'surface')}"
                   + (f" ({', '.join(entry['surfaces'])})" if n else "")
                   + f", pattern {json.dumps(entry['pattern'])}, row {entry['row'] + 1} ({entry['verdict']})")
    landed = lic.get("landed")
    if landed is not None:
        standing: Dict[str, List[str]] = {}
        for f in lic["findings"]:
            if f["code"] == "LICENCE_WITHDRAWN":
                pat = next(e["pattern"] for e in lic["licences"] if e["id"] == f["licence"])
                if f["path"] not in standing.setdefault(pat, []):
                    standing[pat].append(f["path"])
        for pat, paths in standing.items():
            out.append(f"{indent}  withdrawn by the landed row, still printed: pattern {json.dumps(pat)} in "
                       f"{', '.join(paths)} -> FAIL LICENCE_WITHDRAWN")
    return out


def _coverage_lines(cov: Optional[List[Dict[str, Any]]], read: int, indent: str = "  ") -> List[str]:
    out = []
    for c in cov or ():
        n = len(c["surfaces"])
        line = (f"{indent}  the licence's verbatim quote matches {_plural(n, 'surface')} (of {read} read)"
                + (f": {', '.join(c['surfaces'])}" if n else ""))
        if n == 0:
            line += (": registered as the pattern, lint would have refused it before data "
                     "(LICENCE_PATTERN_FINDS_NOTHING)")
        out.append(line + f" [{c['id']}]")
    return out


def _policy_line(pol: Dict[str, Any]) -> str:
    src = pol.get("source", {})
    parts = [f"{k} {pol[k]} ({src.get(k, 'default')})" for k in
             ("immutability", "amend_after_data", "receipt_rewrite", "require_time", "preregistered_word")]
    ex = pol.get("exclude") or []
    parts.append("exclude " + (", ".join(ex) if ex else "none") + (f" ({src['exclude']})" if ex else ""))
    surf = pol.get("surfaces")
    parts.append("surfaces " + (", ".join(surf["include"]) if surf else "the repo profile of scripts/surface_lint.py"))
    return "policies: " + ", ".join(parts)


def _pair_label(pair: Dict[str, Any]) -> str:
    results = [d["result"] for d in pair.get("declared") or () if d.get("result")]
    if results:
        return ", ".join(dict.fromkeys(r.rsplit("/", 1)[-1] for r in results))
    return pair["receipt"]


def _pair_lines(pair: Dict[str, Any]) -> List[str]:
    out = [f"{pair['reading']}  {_pair_label(pair)}"]
    out.append(f"  receipt {pair['receipt']} ({pair['status']}), R {_short(pair['receipt_commit'], 12)}; "
               f"PREREG {pair['prereg'] or 'not resolved'}")
    seal = pair.get("seal")
    if seal is not None:
        dec = pair.get("decision") or {}
        out.append(f"  decision {dec.get('form') or 'none'} "
                   f"{dec.get('gates_sha256') or dec.get('code') or ''}".rstrip())
        landed = (f"landed at {_short(seal['landing'])}" + (f" ({seal['pr']})" if seal.get("pr") else "")
                  if seal.get("landing") else "not landed")
        time_line = pair["time"]["line"] if pair.get("time") else "time not read (the PREREG holds no decision)"
        out.append(f"  seal bytes {_short(seal['blob_sha256'])}... {landed}; order commit {_short(seal['order'])}; "
                   f"{time_line}")
        if not seal.get("landed"):
            out.append("  not landed: the PREREG arrives in this pull request (C2), so its seal is read from its own "
                       "commits and the time rung cannot exceed AUTHOR_ORDER")
        man = seal.get("manifest_id")
        seen = seal.get("seen")
        out.append(f"  add commit {_short(seal.get('add'))}; manifest line "
                   + (f"{man} (seen: {seen if seen is not None else 'not stated'})" if man else "none")
                   + f"; blob history from the order commit to R: {_plural(len(pair['blob_history']), 'commit')}, "
                     f"{_plural(len({b for _, b in pair['blob_history']}), 'blob')}")
        for a in (pair.get("effect") or {}).get("amendments", ()):
            out.append(f"  amendment {a['id']} ({a['path']} at {_short(a['commit'], 12)}, {a['decision']}, seen: "
                       f"{a['seen'] if a['seen'] is not None else 'not stated'}): "
                       + ("in effect" if a["in_effect"] else "not in effect")
                       + (", decision-identical" if a.get("decision_identical") else ""))
    run = pair.get("run")
    if run is not None:
        out.append(f"  locked scorer {run['argv'][0]}: {_plural(len(run['locked']), 'locked file')} at the seal's "
                   f"blobs, exit {run['exit']}, network: {run['network']}")
    rs = pair.get("rescore")
    if rs is not None and rs.get("code") is None:
        said = [f"the sealed decision scores this receipt {rs['verdict']}"]
        rec = pair.get("recorded") or {}
        said.append(f"the receipt records {rec['verdict']}" if "verdict" in rec else "the receipt records no verdict")
        for d in pair.get("declared") or ():
            said.append(f"the RESULT {d['result']} prints {d['verdict'] if d['verdict'] is not None else ' and '.join(d['verdicts'])}")
        out.append("  " + "; ".join(said))
    out += _ladder_lines(pair.get("ladder"))
    out += _licence_lines(pair.get("licences"))
    out += _finding_lines(pair["findings"])
    out.append("  not examined: " + "; ".join(pair["not_examined"]))
    return out


def _check_lines(report: Dict[str, Any]) -> List[str]:
    out = [f"prereg-ci check {report['tool']['version']}: base {_short(report['base'], 12)}, head "
           f"{_short(report['head'], 12)}, read from merge base {_short(report['merge_base'], 12)}"]
    out.append(_policy_line(report["policies"]))
    cfg = report["policies"].get("config") or {}
    out.append(f"config: {cfg.get('path')} read at base {_short(cfg.get('read_at'), 12)}: "
               + ("does not read, and this pull request replaces it, so the defaults and flags govern"
                  if cfg.get("reads") is False else
                  "read" if cfg.get("present") else "absent, so the defaults and flags govern"))
    out += _finding_lines(report.get("config_findings", ()))
    moved = [f for f in report["decisions"] if f["code"] == "DECISION_EDITED_AFTER_SEAL"]
    out.append(f"C0 sealed decisions: {len(moved)} moved"
               + (" (the refusal path protection gives too: CODEOWNERS with required review, honest-signal's "
                  "registration rule, reproducible-science's CHANGED; DESIGN 3.4 C0)" if moved else ""))
    out += _finding_lines(report["decisions"])
    man = report["manifest"]
    out.append(f"manifest: {man['path']} at HEAD, {_plural(man['lines'], 'line')}" if man["present"]
               else f"manifest: none ({man['path']} is not in HEAD's tree), so every seal is the landing rule's")
    out += _finding_lines(man["findings"])
    pairs = report["pairs"]
    named = sum(1 for p in pairs if p["status"] == "named")
    out.append(f"C1 pairs: {len(pairs)} ({_plural(len(pairs) - named, 'receipt')} written, {named} named by a RESULT "
               f"only); "
               f"{_plural(len([f for f in report['rewritten'] if f['code'] == 'RECEIPT_REWRITTEN']), 'receipt')} "
               f"rewritten")
    out += _finding_lines(report["rewritten"])
    out += _finding_lines(report["results"]["findings"])
    for pair in pairs:
        out += _pair_lines(pair)
    for entry in report["tree_licences"]:
        if not entry["findings"]:
            continue
        who = (f"under receipt {entry['receipt']} (landed {entry['landed']}, {entry['how']})" if entry["receipt"]
               else "before any receipt has landed")
        out.append(f"C8 licences of {entry['prereg']} {who}:")
        out += _finding_lines(entry["findings"])
    out.append("not examined (this check): " + "; ".join(report["not_examined"]))
    out.append(_closing(report))
    return out


def _closing(report: Dict[str, Any]) -> str:
    refused = [f["code"] for f in report["findings"] if f["kind"] == "FAIL"]
    tail = ""
    if refused:
        tail = f": {_plural(len(refused), 'refusal')} ({', '.join(dict.fromkeys(refused))})"
    elif report["exit"] in (2, 3):
        other = [f["code"] for f in report["findings"] if f["kind"] in ("INPUT", "UNCHECKABLE")]
        tail = f": {', '.join(dict.fromkeys(other))}"
    return f"{report['verdict']}  exit {report['exit']}{tail}"


# ---- the markdown job summary ----------------------------------------------------------------------------

def _md(text: Any) -> str:
    """*text* as plain text in a summary cell or list item. Author text (a RESULT's sentences, a path, a
    finding's detail) reaches the summary, so HTML is written as entities and brackets are escaped: a tag
    or a markdown link in a RESULT prints as its characters, not as an image or a link (B7 review N3)."""
    s = str(text).replace("\r", " ").replace("\n", " ").replace("\\", "\\\\")
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.replace("[", "\\[").replace("]", "\\]").replace("|", "\\|")


def _code(text: Any) -> str:
    """*text* as a code span (a path, a licence id): nothing inside one renders, so only what would end
    it is changed, a backtick (as ') and a line break, and a pipe, which a table cell splits on."""
    s = str(text).replace("\r", " ").replace("\n", " ").replace("`", "'").replace("|", "\\|")
    return f"`{s}`"


def _licence_rows(entries: Iterable[Dict[str, Any]]) -> List[str]:
    """One row per licence and file: how many matches, how many with a withdrawal in window (the B5
    review's N3: a pattern may match thousands of places, and the summary stays one row per file)."""
    rows: Dict[Tuple[str, str], List[int]] = {}
    for lic in entries:
        for m in lic.get("matches", ()):
            c = rows.setdefault((lic["id"], m["path"]), [0, 0])
            c[0] += 1
            c[1] += 1 if m.get("withdrawn") else 0
    return [f"| {_code(lid)} | {_code(path)} | {n} | {w} |" for (lid, path), (n, w) in rows.items()]


def _markdown(report: Dict[str, Any]) -> str:
    out = [f"## prereg-ci {report.get('command', 'check')}: {report.get('verdict')}", ""]
    out.append(f"exit {report.get('exit')}; base `{_short(report.get('base'), 12)}`, head "
               f"`{_short(report.get('head'), 12)}`. Every FAIL is a function of git objects at named commits; "
               f"the time ladder is printed, and decides a FAIL only under require_time T3a.")
    out.append("")
    pairs = report.get("pairs") or []
    if pairs:
        out += ["| reading | receipt | PREREG | verdict | landed row | time |", "|---|---|---|---|---|---|"]
        for p in pairs:
            rs = p.get("rescore") or {}
            lad = p.get("ladder")
            out.append(f"| {p.get('reading')} | {_code(p.get('receipt'))} | {_code(p.get('prereg'))} | "
                       f"{_md(rs.get('verdict') or rs.get('code') or '')} | "
                       f"{(str(lad['landed'] + 1) + ' of ' + str(len(lad['rows']))) if lad else ''} | "
                       f"{_md((p.get('time') or {}).get('name') or '')} |")
        out.append("")
    else:
        out += ["No receipt to re-derive in this range.", ""]
    findings = [f for f in report.get("findings") or () if not f["code"].startswith("LICENCE_")]
    if findings:
        out += ["| kind | code | where | detail |", "|---|---|---|---|"]
        for f in sorted(findings, key=lambda f: _ORDER.index(f["kind"]) if f["kind"] in _ORDER else 9)[:200]:
            out.append(f"| {f['kind']} | `{f['code']}` | {_code(f.get('pair') or f.get('path') or '')} | "
                       f"{_md((f.get('detail') or '')[:400])} |")
        out.append("")
    licence_entries = [lic for p in pairs for lic in ((p.get("licences") or {}).get("licences") or ())]
    licence_entries += [lic for e in report.get("tree_licences") or () for lic in e.get("licences") or ()]
    rows = _licence_rows(licence_entries)
    if rows:
        out += ["Licences (matches per licence and file; each match is in the JSON report):", "",
                "| licence | file | matches | withdrawn in window |", "|---|---|---|---|", *rows, ""]
    lic_findings = [f for f in report.get("findings") or () if f["code"].startswith("LICENCE_")]
    if lic_findings:
        counts: Dict[Tuple[str, str], int] = {}
        for f in lic_findings:
            counts[(f["kind"], f["code"])] = counts.get((f["kind"], f["code"]), 0) + 1
        out.append("Licence findings: " + "; ".join(f"{k} {c} x{n}" for (k, c), n in counts.items()))
        out.append("")
    ne = [_md(x) for x in report.get("not_examined") or []]
    for p in pairs:
        ne += [f"{_code(p.get('receipt'))}: " + _md("; ".join(p.get("not_examined") or []))]
    if ne:
        out.append("**not examined**")
        out.append("")
        out += [f"- {x}" for x in ne]
        out.append("")
    return "\n".join(out) + "\n"


# ---- what prose a RESULT says about the seal, and whether it calls itself preregistered ------------------

_SWORN_TAG = re.compile(r"</?sworn\b[^>]*>", re.IGNORECASE)
_SEAL_WORD = re.compile(r"\bseal", re.IGNORECASE)
_ORDER_WORD = re.compile(r"\border(?:s|ed|ing)?\b", re.IGNORECASE)
_PREREGISTERED = re.compile(r"\bpre-?registered\b", re.IGNORECASE)
_STATUS = re.compile(r"^[\s>*_#-]*status\b", re.IGNORECASE)
_QUOTED = 5
_QUOTE_CHARS = 300


def _seal_sentences(text: str) -> List[str]:
    """The RESULT's own sentences that speak of the seal's order (a sentence holding "seal" and a form
    of "order"), quoted for the boundary print (C9): what the author said about the order of data and
    seal, which no check here reads. A pattern over prose, capped; nothing is decided by it."""
    text = _SWORN_TAG.sub("", text.replace("\r\n", "\n").replace("\r", "\n")).replace("**", "")
    blocks = re.split(r"\n\s*\n|\n(?=\s*(?:#|[-*+] |\d+[.)] ))", text)
    out: List[str] = []
    for block in blocks:
        for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", block).strip()):
            s = s.lstrip("#-*+ ").strip()
            if s and _SEAL_WORD.search(s) and _ORDER_WORD.search(s):
                out.append(s if len(s) <= _QUOTE_CHARS else s[:_QUOTE_CHARS - 3] + "...")
            if len(out) >= _QUOTED:
                return out
    return out


def _calls_itself_preregistered(text: str) -> bool:
    """Whether a RESULT's title (its opening heading) or a status line calls it preregistered (DESIGN
    3.5, ``preregistered_word``)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    title = next((ln for ln in lines if ln.lstrip().startswith("#")), "")
    status = [ln for ln in lines if _STATUS.match(ln)]
    return any(_PREREGISTERED.search(ln) for ln in [title, *status])
