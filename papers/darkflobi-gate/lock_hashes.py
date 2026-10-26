# -*- coding: utf-8 -*-
"""The lock-hash list of the darkflobi gate: what the PREREG froze, re-derived from the files on the box.

Standard library only (and `git` for `pin-tree`). Four verbs:

    py -3.12 papers/darkflobi-gate/lock_hashes.py table [--w-root C:/Users/heyzo/clawd] [--json]
        prints the markdown rows of the lock table for the gate's files as they stand on the box:
        every regular file under W:gate/ the lock covers (below), each snapshot folder as one tree digest,
        the salt as salt_sha256, config.json, the two public canary files, and the paths outside W:gate/
        the plan names (the X script, the posting skill, the plugin as installed). A path that does not
        exist prints `not built at the freeze`.

    py -3.12 papers/darkflobi-gate/lock_hashes.py tree DIR
        prints the tree digest of DIR: sha256 over the sorted lines "<sha256 of bytes>  <relative path>\\n",
        relative paths with "/" separators, every regular file, no normalisation of bytes, and the file count.
        A directory the walker cannot list, a path it cannot stat, or an entry that is not a regular file (a
        symlink, a junction, a device) is an error, never a silent skip; Windows paths past 260 characters are
        read through the `\\\\?\\` prefix.

    py -3.12 papers/darkflobi-gate/lock_hashes.py pin-tree [--repo DIR] [--commit 3437340b...]
        prints the same tree digest computed from the commit's own blobs (`git ls-tree -r -z`, `git cat-file
        --batch`), with no checkout and no archive: the digest the snapshot row must equal, re-derivable by
        anyone holding the repository.

    py -3.12 papers/darkflobi-gate/lock_hashes.py check PREREG.md [--w-root ...] [--home ...] [--strict]
                                                     [--amendment AMENDMENT.md ...]
        reads every lock row of the PREREG (its section 11 and any later row under section 15; the last row
        for a path wins), then the lock rows of each `--amendment` file in the order given (a file written
        under the PREREG's section 12 after the merge; its rows are applied after the PREREG's, so an
        amendment's row for a path replaces the PREREG's), re-derives each hash from the box and prints one
        status per row:
          SAME         the bytes are the locked bytes
          CHANGED      the bytes moved (a failure)
          BUILT SINCE  the row reads `not built at the freeze` and the file now exists with no later row (a failure:
                       the hash must be restated under section 15 before the merge)
          NOT BUILT    the row reads `not built at the freeze` and the file is still absent
          ABSENT       a locked file is gone (printed; a failure under --strict)
          UNREADABLE   the walker could not read a tree row's folder whole (a failure)
          SKIPPED      the plugins-subtree row (the config is read by its own digest script, never here)
        and then one `UNLOCKED <path>` line per file the lock covers on the box that no row names (a file added
        under W:gate/ after the fill), and one `BYTECODE <path>` line per `__pycache__` folder and `.pyc` file where
        the lock covers W:gate/ (Python runs a cache whose header carries the source's size and time in place of the source, so a
        forged cache beside a locked `.py` would run with every row SAME; the snapshot's caches are inside its tree
        digest). Exit 1 on any CHANGED, BUILT SINCE or UNREADABLE; with --strict also on any ABSENT, UNLOCKED or
        BYTECODE, which is the form the merge-time run and the weekly check use.
        Each amendment file read prints one `AMENDMENT <name> rows <n> sha256 <16 hex>` line (the sha256 of its LF
        bytes, so the reading names the text it read), and a row an amendment supplied carries `via=<name>`. A
        path two amendment files, or one file twice, name with two digests is a CLASH: both names and both digests
        are printed and `check` exits 2 before any row is read, since no reading of the box can say which row is
        the freeze. An amendment with no rows is read and counted (`rows 0`); a missing file exits 2. A `#gates`
        row is read against the PREREG's own text, as before.

What the lock covers under W:gate/, and what it never covers. Never entered, at the top level of W:gate/: the live
state folders `sources/`, `refused/`, `prepared/`, the mint's dry-run records `prepared_dryrun/`, `pending/`,
`baseline/`, the plugin's scratch `tmp/`, the wheel
`wheel/`, `exhibit_out/` (the
local mirror of the `exhibit` branch, pinned by its own charon log), the canary harness's working folder `canary/`
(its two public files, `schedule.json` and the sealed seed's salted digest under `keys/`, get rows of their own), and
the `snapshot_*` folders (one tree-digest row each). Never entered, at any depth: a folder whose name opens with `_`
(scratch by the folder's own convention: `_scratch/`, `_tmp/`, `_out/`), `__pycache__`, `.pytest_cache`,
`node_modules`, `.git`, and any folder whose name holds `sealed` or `secret`, which this script never lists, never
opens and never hashes. Never a row: at the top level the live files `ledger.jsonl`, its lock `ledger.jsonl.lock` and the lock's breaker
`ledger.jsonl.lock.break`, `head.txt`, `heads.txt`, `tamper.jsonl`, `ledger_failures.jsonl`, `REQUIRED`,
`plugin_sent_index.jsonl`, `plugin_failures.jsonl`, the plugin's `anchor_state.json` and the snapshot manifest cache
`.manifest_snapshot_<pin8>.json` (written by gate.py beside the snapshot it caches; the snapshot row binds the bytes
it caches); at any depth a file whose name opens with `_` (`_test_*.txt`) or ends `.pyc`, `.log` or `.tmp`. `salt` and
`config.json` get rows of their own. Everything else under W:gate/ is a row whether or not the plan named it.

What this does not check: that the files do what the plan says (their own tests do), that the hashes in the PREREG
were computed before anything else (the pull request's server timestamp does), or anything about the ledger's content.
A hash the agent computed is the agent's statement; the operator's own `sha256sum` at each weekly merge (OD-10) is the
one beside it.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import stat as _stat
import subprocess
import sys
from pathlib import Path

NOT_BUILT = "not built at the freeze"

# never a row, at the top level of W:gate/ (the live state; `salt` and `config.json` get their own rows below)
EXCLUDED_NAMES = {"ledger.jsonl", "ledger.jsonl.lock", "ledger.jsonl.lock.break", "head.txt", "heads.txt",
                  "tamper.jsonl", "ledger_failures.jsonl", "salt", "config.json", "REQUIRED",
                  "plugin_sent_index.jsonl", "plugin_failures.jsonl", "anchor_state.json"}
EXCLUDED_TOP_GLOBS = (".manifest_snapshot_*.json",)
# never entered, at the top level of W:gate/
EXCLUDED_TOP_DIRS = {"sources", "refused", "prepared", "prepared_dryrun", "pending", "baseline", "tmp", "wheel",
                     "exhibit_out", "canary"}
# never entered, at any depth
EXCLUDED_ANY_DIRS = {"__pycache__", ".pytest_cache", "node_modules", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".log", ".tmp"}
# a sealed folder (the canary seed, a panel key) is never listed, opened or hashed by this script
SEALED_MARKERS = ("sealed", "secret")
SNAPSHOT_PREFIX = "snapshot_"
PIN = "3437340b9e944644372e41d9fcab735d47e20fa0"

# the files under W:gate/ the plan names (P0.1, P0.2, P0.3, P0.4, P0.7); a missing one prints NOT_BUILT rather than nothing
PLAN_NAMED = ("gate.py", "gate_verify.py", "page.py", "patch_dist.js", "selector.py", "canary_shadow.py",
              "reconcile.py", "exhibit_mint.py", "plugin/index.js", "plugin/clawdbot.plugin.json", "gate.js")
# the two public files of the canary harness (its working folder is otherwise not entered): the schedule (item ids and
# times, no construction letters) and the sealed seed's salted digest, which the committed copy under
# papers/darkflobi-gate/keys/ equals byte for byte
CANARY_PUBLIC = (
    ("canary schedule", "W:gate/canary/schedule.json"),
    ("canary seed digest", "W:gate/canary/keys/sworn_measurement_darkflobi_gate_canary_seed.sha256"),
)
# paths outside W:gate/ the plan names as lock-hashed (P0.4, P0.5), relative to W:
OUTSIDE_W = (
    ("the X script (P0.5 not built: no check in it)", "W:ops/darkflobi-tweet.js"),
    ("the posting skill", "W:skills/styxx-post/SKILL.md"),
)
# the plugin as installed (OD-4; W:gate/plugin/README.md step 1 copies these six); hashed where it exists, else
# NOT_BUILT; relative to the home folder
PLUGIN_HOME = ("~/.clawdbot/extensions/styxx-gate/clawdbot.plugin.json",
               "~/.clawdbot/extensions/styxx-gate/index.js",
               "~/.clawdbot/extensions/styxx-gate/package.json",
               "~/.clawdbot/extensions/styxx-gate/README.md",
               "~/.clawdbot/extensions/styxx-gate/gate.js",
               "~/.clawdbot/extensions/styxx-gate/patch_dist.js")

# a lock row: | label | `path` | `sha256 (64 hex), or the pin (40 hex), or the not-built note` | note |
ROW_RE = re.compile(r"^\|\s*(?P<label>[^|]*?)\s*\|\s*`(?P<path>[^`]+)`\s*\|\s*`(?P<hash>[0-9a-f]{64}|[0-9a-f]{40}|"
                    + re.escape(NOT_BUILT) + r")`\s*\|\s*(?P<note>[^|]*?)\s*\|\s*$")


class WalkError(RuntimeError):
    """A path the walker could not read as a regular file; no digest is computed over a partial tree."""


def _long(p) -> str:
    """The OS path string for `p`, long-path safe on Windows (`\\\\?\\` + the absolute path)."""
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        if s.startswith("\\\\"):
            return "\\\\?\\UNC\\" + s[2:]
        return "\\\\?\\" + s
    return s


def _sealed(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in SEALED_MARKERS)


def _is_link(full: str) -> bool:
    if os.path.islink(full):
        return True
    isj = getattr(os.path, "isjunction", None)
    return bool(isj and isj(full))


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(_long(p), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_regular_files(root, prune=None):
    """[(relative posix path, os path string, size)] for every regular file under root, sorted by path.

    `prune(rel_dir, name)` -> True keeps the walker out of the sub-folder `name` of `rel_dir` (posix, "" at the
    root): a pruned folder is never listed. Raises WalkError on a folder that cannot be listed, an entry that
    cannot be stat'ed, a symlinked or junction folder that is not pruned, and an entry that is not a regular file:
    a digest over a tree that silently lost a file is the defect the round-1 review named (its 2.7).
    """
    base = _long(root)
    if not os.path.isdir(base):
        raise WalkError("not a folder: %s" % root)

    def onerror(err):
        raise WalkError("cannot list %s: %s" % (getattr(err, "filename", "?"), err))

    out = []
    for dirpath, dirnames, filenames in os.walk(base, onerror=onerror):
        rel_dir = os.path.relpath(dirpath, base).replace("\\", "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        keep = []
        for d in sorted(dirnames):
            if prune is not None and prune(rel_dir, d):
                continue
            full = os.path.join(dirpath, d)
            if _is_link(full):
                raise WalkError("a linked folder, not walked: %s" % full)
            keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            try:
                st = os.lstat(full)
            except OSError as e:
                raise WalkError("cannot stat %s: %s" % (full, e)) from e
            if _stat.S_ISLNK(st.st_mode) or _is_link(full):
                raise WalkError("not a regular file (a link): %s" % full)
            if not _stat.S_ISREG(st.st_mode):
                raise WalkError("not a regular file: %s" % full)
            rel = (rel_dir + "/" + name) if rel_dir else name
            out.append((rel, full, st.st_size))
    out.sort(key=lambda t: t[0])
    return out


def tree_digest(root) -> tuple[str, int]:
    """sha256 over sorted "<sha256>  <relpath>\\n" lines; every regular file under root; no silent skip."""
    rows = [f"{sha256_file(full)}  {rel}\n" for rel, full, _ in walk_regular_files(root)]
    return hashlib.sha256("".join(rows).encode("utf-8")).hexdigest(), len(rows)


def pin_tree_digest(repo, commit: str = PIN) -> tuple[str, int]:
    """The tree digest of `commit` computed from its own blobs: `git ls-tree -r -z` then `git cat-file --batch`.

    The same rule as tree_digest; the bytes are the blobs' (the snapshot is `git -c core.autocrlf=false archive`,
    and the pin's tree carries no `export-subst` or `ident` attribute, so the archive's bytes are the blobs').
    Raises ValueError on an entry that is not a plain or executable blob (a symlink or a submodule).
    """
    ls = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", commit], capture_output=True, timeout=120)
    if ls.returncode != 0:
        raise ValueError("git ls-tree failed: " + ls.stderr.decode("utf-8", "replace")[:200])
    entries = []
    for rec in ls.stdout.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, obj = meta.split(b" ")
        if typ != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError("not a plain blob in the tree: %s %s %s" % (mode, typ, path))
        entries.append((path.decode("utf-8"), obj.decode("ascii")))
    proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    rows = []
    try:
        for path, obj in entries:
            proc.stdin.write(obj.encode("ascii") + b"\n")
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            size = int(header[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)
            rows.append((path, hashlib.sha256(data).hexdigest()))
    finally:
        proc.stdin.close()
        proc.wait(timeout=60)
    rows.sort(key=lambda t: t[0])
    text = "".join(f"{h}  {p}\n" for p, h in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(rows)


def canonical_json_sha256(value) -> tuple[str, int]:
    canon = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canon).hexdigest(), len(canon)


def resolve(path: str, w_root: Path, home: Path) -> Path:
    if path.startswith("W:"):
        return w_root / path[2:]
    if path.startswith("~/"):
        return home / path[2:]
    return Path(path)


def _gate_prune(rel_dir: str, name: str) -> bool:
    if name in EXCLUDED_ANY_DIRS or name.startswith("_") or _sealed(name):
        return True
    if rel_dir == "" and (name in EXCLUDED_TOP_DIRS or name.startswith(SNAPSHOT_PREFIX)):
        return True
    return False


def gate_files(gate: Path):
    """Regular files under W:gate/ the lock covers, as (relpath, os path, size), sorted."""
    if not gate.is_dir():
        return []
    out = []
    for rel, full, size in walk_regular_files(gate, prune=_gate_prune):
        top = "/" not in rel
        name = rel.rsplit("/", 1)[-1]
        if top and (name in EXCLUDED_NAMES or any(fnmatch.fnmatchcase(name, g) for g in EXCLUDED_TOP_GLOBS)):
            continue
        if name.startswith("_"):            # scratch by the folder's own convention (`_test_*.txt`)
            continue
        if os.path.splitext(name)[1] in EXCLUDED_SUFFIXES:
            continue
        out.append((rel, full, size))
    return out


def bytecode_paths(gate: Path) -> list[str]:
    """Every `__pycache__` folder (as `<rel>/`) and every `.pyc` file where the lock covers W:gate/, sorted posix paths.

    The walk is the lock's own (`_gate_prune`): never a sealed folder (never listed), `node_modules`, `.git`, `_`
    scratch, the live-state and harness folders at the top, or the `snapshot_*` folders (their caches are inside the
    tree digest); a `__pycache__` folder is named and not listed. Raises WalkError as the walker does."""
    if not gate.is_dir():
        return []
    found: list[str] = []

    def prune(rel_dir: str, name: str) -> bool:
        if name == "__pycache__":
            found.append((rel_dir + "/" if rel_dir else "") + name + "/")
            return True
        return _gate_prune(rel_dir, name)

    for rel, _full, _size in walk_regular_files(gate, prune=prune):
        if rel.endswith(".pyc"):
            found.append(rel)
    return sorted(found)


def rows_for_box(w_root: Path, home: Path, hashed: bool = True) -> list[tuple[str, str, str, str]]:
    """(label, path, hash-or-NOT_BUILT, note) for the whole lock set as the box stands.

    With hashed=False nothing is read but names and sizes: an existing path's hash column reads "present" (the
    form `check` uses to find UNLOCKED paths without hashing the snapshot twice)."""
    gate = w_root / "gate"
    rows: list[tuple[str, str, str, str]] = []
    rows.append(("the pin", "pin:3437340b", PIN, "the commit the legs run from; moves only by AMENDMENT"))
    snaps = sorted(p for p in gate.iterdir() if p.is_dir() and p.name.startswith(SNAPSHOT_PREFIX)) \
        if gate.is_dir() else []
    if snaps:
        for s in snaps:
            if hashed:
                d, n = tree_digest(s)
                rows.append(("the snapshot", f"W:gate/{s.name}/", d, f"tree digest over {n} files (`lock_hashes.py tree`)"))
            else:
                rows.append(("the snapshot", f"W:gate/{s.name}/", "present", "tree digest"))
    else:
        rows.append(("the snapshot", f"W:gate/{SNAPSHOT_PREFIX}{PIN[:8]}/", NOT_BUILT, "tree digest"))

    def file_row(label, rel, p, absent_note):
        if p.is_file():
            return (label, rel, sha256_file(p) if hashed else "present", f"{p.stat().st_size} bytes")
        return (label, rel, NOT_BUILT, absent_note)

    present = set()
    for rel, full, size in gate_files(gate):
        present.add(rel)
        rows.append(("gate file", f"W:gate/{rel}", sha256_file(full) if hashed else "present", f"{size} bytes"))
    for name in PLAN_NAMED:
        if name not in present:
            rows.append(("gate file", f"W:gate/{name}", NOT_BUILT, "named by the plan's section 3.1; locked when the piece lands"))
    for label, rel in CANARY_PUBLIC:
        rows.append(file_row(label, rel, resolve(rel, w_root, home), "written by `canary_shadow.py seal-seed` / `plan`"))
    salt = gate / "salt"
    if salt.is_file():
        rows.append(("salt_sha256", "W:gate/salt", sha256_file(salt) if hashed else "present",
                     f"sha256 of the {salt.stat().st_size} raw bytes; the salt itself never leaves the box"))
    else:
        rows.append(("salt_sha256", "W:gate/salt", NOT_BUILT, "written at the sitting (OD-6); appended under section 15"))
    cfg_note = ("written once at the sitting from OD-6 and OD-11 (`operator_dm_hash`, `anchor_channel_hash`, "
                "`canary_chat_hash`); `config_sha256` on every line")
    cfg = gate / "config.json"
    if cfg.is_file():
        rows.append(("config_sha256", "W:gate/config.json", sha256_file(cfg) if hashed else "present",
                     f"{cfg.stat().st_size} bytes; " + cfg_note))
    else:
        rows.append(("config_sha256", "W:gate/config.json", NOT_BUILT, cfg_note + "; appended under section 15 before the merge"))
    for label, rel in OUTSIDE_W:
        rows.append(file_row(label, rel, resolve(rel, w_root, home), "locked when the piece lands"))
    for rel in PLUGIN_HOME:
        rows.append(file_row("plugin as installed", rel, resolve(rel, w_root, home), "installed at the sitting (OD-4)"))
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


def amendment_rows(paths) -> tuple[dict[str, tuple[str, str]], dict[str, str], list[tuple[str, int, str]], list[str]]:
    """The lock rows of the amendment files, read in the order given.

    Returns (rows, via, files, clashes): rows maps path -> (hash, label) as `parse_lock_rows` does; via maps each
    path to the file name(s) that named it ("a.md + b.md" when two files name one digest); files lists
    (name, row count, sha256 of the LF bytes) per file; clashes lists one sentence per path named with two digests,
    across files or inside one file. The caller refuses to read the box while clashes is not empty."""
    rows: dict[str, tuple[str, str]] = {}
    via: dict[str, str] = {}
    files: list[tuple[str, int, str]] = []
    clashes: list[str] = []
    for f in paths:
        f = Path(f)
        raw = f.read_bytes()
        text = raw.decode("utf-8")
        n = 0
        for line in text.splitlines():
            m = ROW_RE.match(line)
            if not m:
                continue
            n += 1
            p, h, label = m.group("path"), m.group("hash"), m.group("label")
            if p in rows and rows[p][0] != h:
                clashes.append(f"{p}: {via[p]} reads {rows[p][0]}, {f.name} reads {h}")
                continue
            if p in rows:
                if f.name not in via[p].split(" + "):
                    via[p] = via[p] + " + " + f.name
            else:
                via[p] = f.name
            rows[p] = (h, label)
        files.append((f.name, n, hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()))
    return rows, via, files, clashes


def gates_block_sha256(text: str) -> str:
    """The sha256 of the text between the plain ```gates fence and its closer, as the scanner reads a plain file.

    `styxx.protocol._select_gates_block` is the authority (it refuses hidden, indented, tilde and doubled fences);
    this is the standard-library reading of the one plain form, enough to re-derive the row without styxx on the path.
    """
    lines = text.split("\n")
    opens = [i for i, ln in enumerate(lines) if ln == "```gates"]
    if len(opens) != 1:
        raise ValueError(f"expected exactly one plain ```gates fence, found {len(opens)}")
    i = opens[0]
    for j in range(i + 1, len(lines)):
        if lines[j].rstrip() == "```":
            return hashlib.sha256("\n".join(lines[i + 1:j]).encode("utf-8")).hexdigest()
    raise ValueError("the ```gates fence is never closed")


def check(prereg: Path, w_root: Path, home: Path, strict: bool = False, amendments=()) -> int:
    text = prereg.read_text(encoding="utf-8")
    rows = parse_lock_rows(text)
    if not rows:
        print("no lock rows found", file=sys.stderr)
        return 2
    via: dict[str, str] = {}
    if amendments:
        try:
            arows, via, files, clashes = amendment_rows(amendments)
        except (OSError, UnicodeDecodeError) as e:
            print(f"REFUSED: an amendment file cannot be read: {e}", file=sys.stderr)
            return 2
        for name, n, digest in files:
            print(f"{'AMENDMENT':12s} {name}  rows {n}  sha256 {digest[:16]}")
        if clashes:
            for c in clashes:
                print(f"{'CLASH':12s} {c}")
                print(f"CLASH: {c}", file=sys.stderr)
            print(f"clash {len(clashes)}: no row was read; each path must carry one digest across the amendment files",
                  file=sys.stderr)
            return 2
        rows.update(arows)
    repo_root = prereg.resolve().parent.parent.parent      # papers/darkflobi-gate/PREREG -> the repository
    counts = {"CHANGED": 0, "BUILT SINCE": 0, "ABSENT": 0, "UNREADABLE": 0}
    for path, (h, label) in rows.items():
        extra = ""
        if path.startswith("pin:"):
            status = "SAME" if h == PIN else "CHANGED"
            now = PIN
        elif path.endswith("#gates"):
            now = gates_block_sha256(text)
            status = "SAME" if now == h else "CHANGED"
        elif path.startswith("plugins-subtree:"):
            status, now = "SKIPPED", "-"
            extra = "  (read the config with the digest script; never printed here)"
        elif not (path.startswith("W:") or path.startswith("~/")):
            # a repository file, locked as its git blob: LF bytes, whatever the checkout's line ends
            p = repo_root / path
            if p.is_file():
                now = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                status = "SAME" if now == h else "CHANGED"
            else:
                now, status = "-", "ABSENT"
        elif path.endswith("/"):
            p = resolve(path.rstrip("/"), w_root, home)
            if p.is_dir():
                try:
                    now, n = tree_digest(p)
                    status = "SAME" if now == h else ("BUILT SINCE" if h == NOT_BUILT else "CHANGED")
                    extra = f"  files={n}"
                except WalkError as e:
                    now, status, extra = "-", "UNREADABLE", f"  ({e})"
            else:
                now, status = "-", ("ABSENT" if h != NOT_BUILT else "NOT BUILT")
        else:
            p = resolve(path, w_root, home)
            if p.is_file():
                now = sha256_file(p)
                status = "SAME" if now == h else ("BUILT SINCE" if h == NOT_BUILT else "CHANGED")
            else:
                now, status = "-", ("ABSENT" if h != NOT_BUILT else "NOT BUILT")
        if status in counts:
            counts[status] += 1
        if path in via:
            extra += f"  via={via[path]}"
        print(f"{status:12s} {path}  locked={h[:16]}  now={now[:16] if now != '-' else '-'}{extra}")
    # files the lock covers on the box that no row names: added after the fill, seen by no hash the agent did not compute
    unlocked = 0
    try:
        box = rows_for_box(w_root, home, hashed=False)
    except WalkError as e:
        print(f"{'UNREADABLE':12s} W:gate/  ({e})")
        counts["UNREADABLE"] += 1
        box = []
    for _label, path, h, _note in box:
        if path not in rows and h != NOT_BUILT:
            unlocked += 1
            print(f"{'UNLOCKED':12s} {path}  locked=-  now=present")
    # caches Python would run in place of a locked source
    try:
        caches = bytecode_paths(w_root / "gate")
    except WalkError as e:
        print(f"{'UNREADABLE':12s} W:gate/ (bytecode scan)  ({e})")
        counts["UNREADABLE"] += 1
        caches = []
    for rel in caches:
        print(f"{'BYTECODE':12s} W:gate/{rel}")
    print(f"rows {len(rows)}, changed {counts['CHANGED']}, built since {counts['BUILT SINCE']}, "
          f"absent {counts['ABSENT']}, unreadable {counts['UNREADABLE']}, unlocked {unlocked}, bytecode {len(caches)}"
          + (" (strict)" if strict else ""))
    failed = counts["CHANGED"] + counts["BUILT SINCE"] + counts["UNREADABLE"]
    if strict:
        failed += counts["ABSENT"] + unlocked + len(caches)
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    t = sub.add_parser("table")
    t.add_argument("--w-root", default="C:/Users/heyzo/clawd")
    t.add_argument("--home", default=str(Path.home()))
    t.add_argument("--json", action="store_true")
    tr = sub.add_parser("tree")
    tr.add_argument("dir")
    pt = sub.add_parser("pin-tree")
    pt.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent.parent))
    pt.add_argument("--commit", default=PIN)
    c = sub.add_parser("check")
    c.add_argument("prereg")
    c.add_argument("--w-root", default="C:/Users/heyzo/clawd")
    c.add_argument("--home", default=str(Path.home()))
    c.add_argument("--strict", action="store_true",
                   help="an ABSENT or UNLOCKED path is a failure too (the merge-time and weekly form)")
    c.add_argument("--amendment", action="append", default=[], metavar="FILE",
                   help="an AMENDMENT file whose lock rows are applied after the PREREG's (repeatable, in order; "
                        "a path two files name with two digests exits 2)")
    a = ap.parse_args(argv)
    try:
        if a.verb == "tree":
            d, n = tree_digest(Path(a.dir))
            print(f"{d}  {n} files  {Path(a.dir).as_posix()}")
            return 0
        if a.verb == "pin-tree":
            d, n = pin_tree_digest(a.repo, a.commit)
            print(f"{d}  {n} files  {a.commit}")
            return 0
        if a.verb == "table":
            rows = rows_for_box(Path(a.w_root), Path(a.home))
            if a.json:
                print(json.dumps([{"label": l, "path": p, "sha256": h, "note": n} for l, p, h, n in rows],
                                 indent=1, ensure_ascii=False))
            else:
                print(markdown_rows(rows))
            return 0
    except (WalkError, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return check(Path(a.prereg), Path(a.w_root), Path(a.home), strict=a.strict,
                 amendments=[Path(x) for x in a.amendment])


if __name__ == "__main__":
    sys.exit(main())
