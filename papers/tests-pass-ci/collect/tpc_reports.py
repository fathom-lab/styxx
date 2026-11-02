# -*- coding: utf-8 -*-
"""Machine-readable test reports (PREREG P18).

At read, the collector downloads every artifact at the head commit whose name matches
tpc_const.REPORT_ARTIFACT_NAME, listed as not expired and at most ARTIFACT_MAX bytes, for the rows
that get logs. This module looks inside a downloaded zip, without extracting anything it does not
need: a file counts when it is an .xml whose root element is testsuite or testsuites, a .trx file,
or a JSON in-toto statement whose predicateType contains "test-result". Each counted file is
written beside the zip and, when styxx is importable, read by `styxx.evidence` (pinned in the
PREREG's lock table), whose verdict and reason are recorded.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional

from tpc_const import REPORT_ARTIFACT_NAME

NAME_RE = re.compile(REPORT_ARTIFACT_NAME)
_PROLOG = re.compile(rb"\A(?:\s|<\?.*?\?>|<!--.*?-->|<!DOCTYPE[^>]*>)*", re.S)
_FIRST = re.compile(rb"<([A-Za-z_][\w.\-]*:)?([A-Za-z_][\w.\-]*)")
MAX_MEMBER = 50 * 1024 * 1024


def report_like_name(name: str) -> bool:
    return bool(NAME_RE.search(name or ""))


def xml_root(data: bytes) -> Optional[str]:
    """The local name of an XML document's root element, read by pattern (no parser, no entities)."""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    head = data[:65536]
    m = _PROLOG.match(head)
    rest = head[m.end():] if m else head
    m2 = _FIRST.match(rest.lstrip())
    return m2.group(2).decode("ascii", "replace") if m2 else None


def classify_member(name: str, data: bytes) -> Optional[str]:
    low = name.lower()
    if low.endswith(".xml"):
        root = xml_root(data)
        if root in ("testsuite", "testsuites"):
            return "junit_xml"
        return None
    if low.endswith(".trx"):
        return "trx"
    if low.endswith(".json") or low.endswith(".jsonl") or low.endswith(".intoto.jsonl"):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        for chunk in (text.splitlines() if low.endswith("jsonl") else [text]):
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue
            if isinstance(obj, dict) and "test-result" in str(obj.get("predicateType") or ""):
                return "intoto_test_result"
    return None


def inspect_zip(zip_bytes: bytes) -> dict:
    """{'members': n, 'counted': [{'name', 'kind', 'bytes', '_data'}], 'bad_zip': bool}"""
    out: dict = {"members": 0, "counted": [], "bad_zip": False}
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        out["bad_zip"] = True
        return out
    infos = zf.infolist()
    out["members"] = len(infos)
    for info in infos:
        if info.is_dir() or info.file_size > MAX_MEMBER:
            continue
        low = info.filename.lower()
        if not (low.endswith(".xml") or low.endswith(".trx") or low.endswith(".json") or low.endswith(".jsonl")):
            continue
        data = zf.read(info)
        kind = classify_member(info.filename, data)
        if kind:
            out["counted"].append({"name": info.filename, "kind": kind, "bytes": len(data), "_data": data})
    return out


def evidence_reader(styxx_root: Optional[Path]) -> Optional[Callable]:
    """styxx.evidence's load + adjudicate, or None when styxx cannot be imported."""
    import sys
    if styxx_root is not None and str(styxx_root) not in sys.path:
        sys.path.insert(0, str(styxx_root))
    try:
        from styxx import evidence  # type: ignore
    except Exception:  # noqa: BLE001
        return None

    def read(paths: list, commit: Optional[str]) -> dict:
        ev = evidence.load_evidence([str(p) for p in paths])
        verdict, reason = evidence.adjudicate_tests_pass(ev, commit)
        return {"verdict": verdict, "reason": str(reason)[:500]}

    return read
