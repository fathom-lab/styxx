# -*- coding: utf-8 -*-
"""The lock-hash list of the darkflobi gate: what the PREREG froze, re-derived from the files on the box.

Standard library only. Three verbs:

    py -3.12 papers/darkflobi-gate/lock_hashes.py table [--w-root C:/Users/heyzo/clawd]
        prints the markdown rows of the lock table for the gate's files as they stand on the box:
        every regular file under W:gate/ (less the live state the lock never covers: the ledger, the heads,
        the salt's bytes, sources/, refused/, prepared/, pending/, baseline/, the wheel, scratch, caches), the
        snapshot folder as one tree digest, the salt as salt_sha256, and the three paths outside W:gate the
        plan names (the X script, the mint script, the posting skill). A path that does not exist prints
        `not built at the freeze`.

    py -3.12 papers/darkflobi-gate/lock_hashes.py tree DIR
        prints the tree digest of DIR: sha256 over the sorted lines "<sha256 of bytes>  <relative path>\\n",
        relative paths with "/" separators, every regular file, no normalisation of bytes.

    py -3.12 papers/darkflobi-gate/lock_hashes.py check PREREG.md [--w-root ...] [--home ...]
        reads every lock row of the PREREG (its section 11 and any later row under section 15; the last row
        for a path wins), re-derives each hash from the box and prints SAME / CHANGED / ABSENT per row.
        Exit 1 when any row reads CHANGED. ABSENT is not a failure of the check (a row that was never built
        is said so in the PREREG); it is printed so a reader sees it.

What this does not check: that the files do what the plan says (their own tests do), that the hashes in
the PREREG were computed before anything else (the pull request's server timestamp does), or anything about
the ledger's content. A hash the agent computed is the agent's statement; the operator's own `sha256sum`
at each weekly merge (OD-10) is the one beside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

NOT_BUILT = "not built at the freeze"

# the live state under W:gate/ the lock never covers (names relative to W:gate/); `salt` and `config.json` get
# their own rows below; `exhibit_out/` is the local mirror of the `exhibit` branch, agent-published content that
# grows with every post and is pinned by its own charon log, not by this table
EXCLUDED_NAMES = {"ledger.jsonl", "head.txt", "heads.txt", "salt", "config.json", "REQUIRED"}
EXCLUDED_DIRS = {"sources", "refused", "prepared", "pending", "baseline", "wheel", "_scratch", "exhibit_out",
                 "canary", "__pycache__", "node_modules", ".git"}
# a sealed folder (the canary seed, a panel key) is never opened by this script, not even to hash it
SEALED_MARKERS = ("sealed", "secret")


def _excluded_dir(name: str) -> bool:
    low = name.lower()
    return name in EXCLUDED_DIRS or name.startswith(SNAPSHOT_PREFIX) or any(m in low for m in SEALED_MARKERS)
EXCLUDED_SUFFIXES = {".pyc", ".log", ".tmp"}
SNAPSHOT_PREFIX = "snapshot_"
PIN = "3437340b9e944644372e41d9fcab735d47e20fa0"

# the files under W:gate/ the plan names (P0.1, P0.2, P0.3, P0.7); a missing one prints NOT_BUILT rather than nothing
PLAN_NAMED = ("gate.py", "gate_verify.py", "page.py", "patch_dist.js", "selector.py", "canary_shadow.py",
              "reconcile.py")
# paths outside W:gate/ the plan names as lock-hashed (P0.4, P0.5), relative to W:
OUTSIDE_W = (
    ("the X check", "W:ops/darkflobi-tweet.js"),
    ("the mint script", "W:ops/exhibit_mint.py"),
    ("the posting skill", "W:skills/styxx-post/SKILL.md"),
)
# the plugin as installed (OD-4); hashed where it exists, else NOT_BUILT; relative to the home folder
PLUGIN_HOME = ("~/.clawdbot/extensions/styxx-gate/clawdbot.plugin.json",
               "~/.clawdbot/extensions/styxx-gate/index.js",
               "~/.clawdbot/extensions/styxx-gate/gate.js")

# a lock row: | label | `path` | `sha256 (64 hex), or the pin (40 hex), or the not-built note` | note |
ROW_RE = re.compile(r"^\|\s*(?P<label>[^|]*?)\s*\|\s*`(?P<path>[^`]+)`\s*\|\s*`(?P<hash>[0-9a-f]{64}|[0-9a-f]{40}|"
                    + re.escape(NOT_BUILT) + r")`\s*\|\s*(?P<note>[^|]*?)\s*\|\s*$")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(root: Path) -> tuple[str, int]:
    """sha256 over sorted "<sha256>  <relpath>\\n" lines; every regular file under root."""
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if not p.is_file() or p.is_symlink():
                continue
            rel = p.relative_to(root).as_posix()
            rows.append(f"{sha256_file(p)}  {rel}\n")
    rows.sort(key=lambda r: r.split("  ", 1)[1])
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest(), len(rows)


def canonical_json_sha256(value) -> tuple[str, int]:
    canon = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canon).hexdigest(), len(canon)


def resolve(path: str, w_root: Path, home: Path) -> Path:
    if path.startswith("W:"):
        return w_root / path[2:]
    if path.startswith("~/"):
        return home / path[2:]
    return Path(path)


def gate_files(gate: Path):
    """Regular files under W:gate/ the lock covers, as (relpath, Path), sorted."""
    out = []
    if not gate.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(gate):
        rel_dir = Path(dirpath).relative_to(gate)
        parts = rel_dir.parts
        dirnames[:] = sorted(d for d in dirnames if not _excluded_dir(d))
        if any(_excluded_dir(p) for p in parts):
            continue
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if not parts and name in EXCLUDED_NAMES:
                continue
            if name.startswith("_"):            # a scratch file by the folder's own convention (`_scratch/`, `_test_*.txt`)
                continue
            if p.suffix in EXCLUDED_SUFFIXES or not p.is_file():
                continue
            out.append((p.relative_to(gate).as_posix(), p))
    return out


def rows_for_box(w_root: Path, home: Path) -> list[tuple[str, str, str, str]]:
    """(label, path, hash-or-NOT_BUILT, note) for the whole lock set as the box stands."""
    gate = w_root / "gate"
    rows: list[tuple[str, str, str, str]] = []
    rows.append(("the pin", "pin:3437340b", PIN, "the commit the legs run from; moves only by AMENDMENT"))
    snaps = sorted(p for p in gate.glob(SNAPSHOT_PREFIX + "*") if p.is_dir()) if gate.is_dir() else []
    if snaps:
        for s in snaps:
            d, n = tree_digest(s)
            rows.append(("the snapshot", f"W:gate/{s.name}/", d, f"tree digest over {n} files (`lock_hashes.py tree`)"))
    else:
        rows.append(("the snapshot", f"W:gate/{SNAPSHOT_PREFIX}{PIN[:8]}/", NOT_BUILT, "tree digest"))
    present = set()
    for rel, p in gate_files(gate):
        present.add(rel)
        rows.append(("gate file", f"W:gate/{rel}", sha256_file(p), f"{p.stat().st_size} bytes"))
    for name in PLAN_NAMED:
        if name not in present:
            rows.append(("gate file", f"W:gate/{name}", NOT_BUILT, "named by the plan's section 3.1; locked when the piece lands"))
    salt = gate / "salt"
    if salt.is_file():
        rows.append(("salt_sha256", "W:gate/salt", sha256_file(salt), f"sha256 of the {salt.stat().st_size} raw bytes; the salt itself never leaves the box"))
    else:
        rows.append(("salt_sha256", "W:gate/salt", NOT_BUILT, "written at the sitting (OD-6); appended under section 15"))
    cfg = gate / "config.json"
    cfg_note = ("written once at the sitting from OD-6 and OD-11 (`operator_dm_hash`, `anchor_channel_hash`, "
                "`canary_chat_hash`); `config_sha256` on every line")
    if cfg.is_file():
        rows.append(("config_sha256", "W:gate/config.json", sha256_file(cfg), f"{cfg.stat().st_size} bytes; " + cfg_note))
    else:
        rows.append(("config_sha256", "W:gate/config.json", NOT_BUILT, cfg_note + "; appended under section 15 before the merge"))
    for label, rel in OUTSIDE_W:
        p = resolve(rel, w_root, home)
        rows.append((label, rel, sha256_file(p) if p.is_file() else NOT_BUILT,
                     f"{p.stat().st_size} bytes" if p.is_file() else "locked when the piece lands"))
    for rel in PLUGIN_HOME:
        p = resolve(rel, w_root, home)
        rows.append(("plugin as installed", rel, sha256_file(p) if p.is_file() else NOT_BUILT,
                     f"{p.stat().st_size} bytes" if p.is_file() else "installed at the sitting (OD-4)"))
    return rows


def markdown_rows(rows) -> str:
    return "\n".join(f"| {label} | `{path}` | `{h}` | {note} |" for label, path, h, note in rows)


def parse_lock_rows(text: str) -> dict[str, tuple[str, str]]:
    """path -> (hash, label); the last row for a path wins (section 15 overrides section 11)."""
    out: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m:
            out[m.group("path")] = (m.group("hash"), m.group("label"))
    return out


def check(prereg: Path, w_root: Path, home: Path) -> int:
    text = prereg.read_text(encoding="utf-8")
    rows = parse_lock_rows(text)
    if not rows:
        print("no lock rows found", file=sys.stderr)
        return 2
    changed = 0
    for path, (h, label) in rows.items():
        if path.startswith("pin:"):
            status = "SAME" if h == PIN else "CHANGED"
            now = PIN
        elif path.startswith("plugins-subtree:"):
            status, now = "SKIPPED (read the config with the digest script; never printed here)", "-"
        elif path.endswith("/") and SNAPSHOT_PREFIX in path:
            p = resolve(path.rstrip("/"), w_root, home)
            if p.is_dir():
                now, _ = tree_digest(p)
                status = "SAME" if now == h else "CHANGED"
            else:
                now, status = "-", "ABSENT"
        else:
            p = resolve(path, w_root, home)
            if p.is_file():
                now = sha256_file(p)
                status = "SAME" if now == h else ("BUILT SINCE" if h == NOT_BUILT else "CHANGED")
            else:
                now, status = "-", ("ABSENT" if h != NOT_BUILT else "NOT BUILT")
        if status == "CHANGED":
            changed += 1
        print(f"{status:12s} {path}  locked={h[:16]}  now={now[:16] if now != '-' else '-'}")
    print(f"rows {len(rows)}, changed {changed}")
    return 1 if changed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    t = sub.add_parser("table")
    t.add_argument("--w-root", default="C:/Users/heyzo/clawd")
    t.add_argument("--home", default=str(Path.home()))
    t.add_argument("--json", action="store_true")
    tr = sub.add_parser("tree")
    tr.add_argument("dir")
    c = sub.add_parser("check")
    c.add_argument("prereg")
    c.add_argument("--w-root", default="C:/Users/heyzo/clawd")
    c.add_argument("--home", default=str(Path.home()))
    a = ap.parse_args(argv)
    if a.verb == "tree":
        d, n = tree_digest(Path(a.dir))
        print(f"{d}  {n} files  {Path(a.dir).as_posix()}")
        return 0
    if a.verb == "table":
        rows = rows_for_box(Path(a.w_root), Path(a.home))
        if a.json:
            print(json.dumps([{"label": l, "path": p, "sha256": h, "note": n} for l, p, h, n in rows],
                             indent=1, ensure_ascii=False))
        else:
            print(markdown_rows(rows))
        return 0
    return check(Path(a.prereg), Path(a.w_root), Path(a.home))


if __name__ == "__main__":
    sys.exit(main())
