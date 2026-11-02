# -*- coding: utf-8 -*-
"""The raw store: every file the collector writes, with its sha256 in an append-only manifest.

Layout under the root (never in git; P2.4, P4.3):

    manifest.jsonl          one row per file written: path, sha256 and bytes of the file on disk,
                            and for gzipped files the sha256 and bytes of what was compressed
    readlogs/               every API call (tpc_gh), one file per stage and UTC day
    runs.jsonl              one row per collector run: mode, code digests, counts, seal state
    frames/                 frame_<D>.jsonl and frame_digests.jsonl (P2.4)
    draw/                   beacon.json, draw_<D>.jsonl, draw_digests.jsonl (P3)
    a2/raw/<pr_id>/<attempt>/   every response of one read, gzipped; logs; artifacts (P4.2)
    a2/rows/<pr_id>.json    the row record, written last (its presence means the read finished)
    a2/rows_index.jsonl     one line per finished row record
    dryrun/                 counts of --dry-run runs; never rows

Files are written to a temporary name and renamed, so a crash never leaves a half file under the
final name. Nothing is ever overwritten: a second read of a row goes to a new attempt directory.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from tpc_const import UTC, iso


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dumps(obj: Any, indent: Optional[int] = None) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=False) + "\n").encode("utf-8")


class Store:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---- paths ---------------------------------------------------------------------------------
    def path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        root = self.root.resolve()
        if root != p and root not in p.parents:
            raise ValueError(f"path outside the store: {rel}")
        return p

    def rel(self, p: Path) -> str:
        return Path(p).resolve().relative_to(self.root.resolve()).as_posix()

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    # ---- writes --------------------------------------------------------------------------------
    def _atomic(self, p: Path, data: bytes) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        if p.stat().st_size != len(data):  # verify the write landed
            raise OSError(f"short write: {p}")

    def write(self, rel: str, data: bytes, *, overwrite: bool = False, raw: Optional[bytes] = None,
              note: Optional[dict] = None) -> dict:
        """Write bytes under rel and record them in the manifest. Refuses to overwrite unless asked
        (only digest logs and state files are rewritten, and those say so)."""
        p = self.path(rel)
        if p.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {rel}")
        self._atomic(p, data)
        row = {"at": iso(dt.datetime.now(UTC)), "path": self.rel(p), "sha256": sha256_bytes(data),
               "bytes": len(data)}
        if raw is not None:
            row["raw_sha256"] = sha256_bytes(raw)
            row["raw_bytes"] = len(raw)
        if overwrite:
            row["rewritten"] = True
        if note:
            row.update(note)
        self._manifest(row)
        return row

    def write_gz(self, rel: str, raw: bytes, **kw: Any) -> dict:
        if not rel.endswith(".gz"):
            rel += ".gz"
        # mtime=0: the same bytes always gzip to the same file
        return self.write(rel, gzip.compress(raw, compresslevel=6, mtime=0), raw=raw, **kw)

    def write_json(self, rel: str, obj: Any, **kw: Any) -> dict:
        return self.write(rel, dumps(obj, indent=1), **kw)

    def append_jsonl(self, rel: str, row: dict, *, manifest: bool = False) -> None:
        """Append one line. Logs (read logs, digest logs, indexes) are append-only; the manifest
        records their digest when a run closes (`seal_log`)."""
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = dumps(row)
        with self._lock:
            with p.open("ab") as fh:
                fh.write(line)
        if manifest:
            self.seal_log(rel)

    def seal_log(self, rel: str) -> dict:
        """Record an append-only file's current sha256 and size in the manifest."""
        p = self.path(rel)
        data = p.read_bytes()
        row = {"at": iso(dt.datetime.now(UTC)), "path": self.rel(p), "sha256": sha256_bytes(data),
               "bytes": len(data), "append_only": True}
        self._manifest(row)
        return row

    def _manifest(self, row: dict) -> None:
        line = dumps(row)
        with self._lock:
            mp = self.root / "manifest.jsonl"
            with mp.open("ab") as fh:
                fh.write(line)

    # ---- reads ---------------------------------------------------------------------------------
    def read_json(self, rel: str) -> Any:
        return json.loads(self.path(rel).read_bytes().decode("utf-8"))

    def read_jsonl(self, rel: str) -> list:
        p = self.path(rel)
        if not p.exists():
            return []
        out = []
        for line in p.read_bytes().decode("utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def verify_manifest(self) -> dict:
        """Re-hash every file the manifest names (latest row per path). Returns counts and the
        paths whose bytes differ or are missing."""
        latest: dict = {}
        for row in self.read_jsonl("manifest.jsonl"):
            latest[row["path"]] = row
        bad, missing, ok = [], [], 0
        for rel, row in latest.items():
            p = self.root / rel
            if not p.exists():
                missing.append(rel)
                continue
            if row.get("append_only"):
                data = p.read_bytes()[: row["bytes"]]  # a log may have grown since; its prefix may not change
            else:
                data = p.read_bytes()
            if sha256_bytes(data) != row["sha256"]:
                bad.append(rel)
            else:
                ok += 1
        return {"files": len(latest), "ok": ok, "differ": bad, "missing": missing}
