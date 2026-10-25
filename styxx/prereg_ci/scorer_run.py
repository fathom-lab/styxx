# -*- coding: utf-8 -*-
"""styxx.prereg_ci.scorer_run: run a locked scorer at its sealed blobs (DESIGN 3.2(f), step B4).

A locks decision names a scorer program and every file it reads (``core._locks``). The run writes
each locked file's sealed blob into an empty temporary directory at its repository-relative path,
copies the receipt in at its own path, keeps ``.prereg-ci/`` for the ``{out}`` file, and runs

    <python> -I -S -B -X utf8 -c <bootstrap> <config>

with that directory as the working directory and a 60-second timeout. ``-I`` ignores every
``PYTHON*`` variable and leaves the script's directory off ``sys.path``; ``-S`` leaves site packages
off it; ``-B`` writes no bytecode; ``-X utf8`` makes the streams UTF-8 on every console (the
scorer's v1 blob dies on cp1252 otherwise). The bootstrap (PLAN K3) puts the run directory ahead of
the standard library on ``sys.path``, sets ``sys.argv`` to the scorer's path and its arguments, and
runs the scorer with ``runpy`` as ``__main__``, as ``python <scorer> <args>`` would. The child's
environment holds nothing of the caller's but ``SYSTEMROOT`` on Windows, which the interpreter needs
to start there.

A read of an unlocked file fails (3.2(f)). The bootstrap installs an audit hook
(``sys.addaudithook``) and a last finder on ``sys.meta_path`` before the scorer runs, and they record
each reach past the locks to a file outside the run directory, written through a descriptor opened
before the hook:
- opening a file to read that is neither locked, the receipt, the ``{out}`` file, nor under the
  standard library's directories as ``-I -S`` sets them (a ``site-packages`` or ``dist-packages``
  directory below them is not standard library);
- opening any file to write but the ``{out}`` file, and every event that changes the file system;
- listing a directory outside the run and the standard library;
- an import the locked code asks for that no finder holds (a standard library module's own optional
  import, such as tarfile's ``pwd`` on Windows, is not the scorer's ask and is not recorded);
- starting a process, a socket or another network client, a native library through ctypes, the
  Windows registry, a temporary file.
Each is refused by raising ``PermissionError`` from the hook, and recorded whether or not the scorer
catches it, so ``core._run_reading`` reads ``SCORER_READS_UNLOCKED`` whatever the scorer printed. The
run's closing line is written when the scorer returns or raises ``SystemExit``; a scorer that ends
the process itself leaves none. This is not a sandbox (DESIGN 7, risk 5): native code the scorer
already holds, ``stat`` calls and the interpreter itself are not watched, the interpreter is not
locked (it is printed as the runtime's three fields), and on Linux the network is removed with
``unshare -rn`` only where the runner permits it; elsewhere the run prints ``not isolated``.

Nothing here is public API (DESIGN 3.1); the CLI is this module's caller.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from styxx.prereg_ci import core as _core
from styxx.prereg_ci import git as _git
from styxx.sworn import MAX_RECEIPT_BYTES as _OUT_CAP

_TIMEOUT_S = 60
_STDOUT_CAP = 1 << 20                     # bytes of stdout read; the stdout contract is one line of JSON
_STDERR_TAIL = 1 << 16                    # bytes from the end of stderr searched for its last line
_TAIL_CHARS = 400
_OUT = _core._RUN_DIR + "/out.json"
_WATCHED = ("the run refuses and records, through an audit hook and a last import finder installed before the "
            "scorer runs: a read of any file the locks block does not lock (the receipt, the {out} file and the "
            "standard library aside), a write but the {out} file, a listing outside the run, an import the locked "
            "code asks for that the run does not hold, a process, a socket, a native library and the registry. It "
            "is not a sandbox: native code, stat calls and the interpreter are not watched (DESIGN 7, risk 5)")

# The bootstrap the child runs under -I -S -B -X utf8. argv[1] is a JSON config file outside the run
# directory: watch (the record), scorer, args, read and write (run-relative paths).
_BOOTSTRAP = r'''
import json, os, sys
with open(sys.argv[1], "rb") as _f:
    _cfg = json.loads(_f.read().decode("utf-8"))
_fd = os.open(_cfg["watch"], os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o600)


def _say(rec):
    os.write(_fd, (json.dumps(rec, sort_keys=True) + "\n").encode("ascii"))


import platform, pkgutil, runpy, threading, unicodedata
for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(newline="\n")
_say({"t": "runtime", "implementation": sys.implementation.name, "version": platform.python_version(),
      "unicode": unicodedata.unidata_version})
_nc = os.path.normcase
_raw = os.path.abspath(os.getcwd())
_here = _nc(_raw)
_sep = os.sep
_std = [_nc(os.path.abspath(p)) for p in sys.path if p]


def _under(p, root):
    return p == root or p.startswith(root if root.endswith(_sep) else root + _sep)


def _site(p):
    parts = p.replace("\\", "/").split("/")
    return "site-packages" in parts or "dist-packages" in parts


def _stdlib(p):
    return not _site(p) and any(_under(p, r) for r in _std)


def _full(rel):
    return _nc(os.path.join(_raw, rel.replace("/", _sep)))


_reads = {_full(r) for r in _cfg["read"]}
_writes = {_full(r) for r in _cfg["write"]}
# the import system looks for a locked module's cached bytecode before it reads the source; nothing can
# write one there (-B, and every write but {out} is refused), so the look finds nothing and is not a reach
import importlib.util
_reads |= {_nc(importlib.util.cache_from_source(os.path.join(_raw, r.replace("/", _sep))))
           for r in _cfg["read"] if r.endswith(".py")}
_seen = set()


def _note(what):
    if what not in _seen:
        _seen.add(what)
        _say({"t": "unlocked", "what": what})


def _path(x):
    if x is None:
        x = "."
    if isinstance(x, int):
        return None, None
    try:
        x = os.fspath(x)
    except TypeError:
        return None, None
    if isinstance(x, bytes):
        x = os.fsdecode(x)
    a = os.path.abspath(x)
    return _nc(a), a


def _name(p, a):
    return os.path.relpath(a, _raw).replace(_sep, "/") if _under(p, _here) else None


_WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | getattr(os, "O_EXCL", 0)
_PROCESS = {"subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.startfile", "os.fork",
            "os.forkpty", "pty.spawn", "os.kill", "os.killpg", "_winapi.CreateProcess"}
_CHANGE = {"os.chmod", "os.chown", "os.chflags", "os.lchflags", "os.lchmod", "os.lchown", "os.link", "os.mkdir",
           "os.mkfifo", "os.mknod", "os.remove", "os.removexattr", "os.rename", "os.rmdir", "os.setxattr",
           "os.symlink", "os.truncate", "os.utime", "shutil.chown", "shutil.copyfile", "shutil.copymode",
           "shutil.copystat", "shutil.copytree", "shutil.make_archive", "shutil.move", "shutil.rmtree",
           "shutil.unpack_archive"}
_NET = ("socket.", "http.client.", "ftplib.", "imaplib.", "nntplib.", "poplib.", "smtplib.", "telnetlib.",
        "urllib.", "webbrowser.")
_OTHER = {"ctypes.dlopen": "loads a native library", "sqlite3.connect": "opens a database",
          "tempfile.mkstemp": "makes a temporary file", "tempfile.mkdtemp": "makes a temporary directory"}


def _judge(event, args):
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        p, a = _path(path)
        if p is None:
            return None
        write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & _WFLAGS))
        if write:
            if p in _writes:
                return None
            n = _name(p, a)
            return f"opens {n} for writing" if n else "opens a file outside the run for writing"
        if p in _reads or p in _writes:
            return None
        n = _name(p, a)
        if n is not None:
            return f"opens {n} to read"
        if _stdlib(p):
            return None
        return "opens a file outside the run to read"
    if event in ("os.listdir", "os.scandir"):
        p, a = _path(args[0] if args else None)
        if p is None or _under(p, _here) or _stdlib(p):
            return None
        return "lists a directory outside the run"
    if event in _PROCESS:
        return f"starts a process ({event})"
    if event in _CHANGE:
        return f"changes the file system ({event})"
    if event.startswith(_NET):
        return f"uses the network ({event})" if not event.startswith("socket.") else f"uses a socket ({event})"
    if event.startswith("winreg."):
        return f"reads the Windows registry ({event})"
    if event in _OTHER:
        return f"{_OTHER[event]} ({event})"
    return None


_busy = threading.local()


def _hook(event, args):
    if getattr(_busy, "on", False):
        return
    _busy.on = True
    try:
        why = _judge(event, args)
    finally:
        _busy.on = False
    if why is not None:
        _note(why)
        raise PermissionError(f"prereg-ci: the scorer run refuses this (the scorer {why}); only locked files are read")


def _locked_code(frame):
    fn = frame.f_code.co_filename
    if fn.startswith("<"):
        fn = frame.f_globals.get("__file__")
        if not isinstance(fn, str):
            return False
    return _under(_nc(os.path.abspath(os.path.join(_raw, fn))), _here)


class _Miss:
    """The last finder: an import nothing else holds. Recorded when the locked code asked for it."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        f = sys._getframe(1)
        while f is not None and (f.f_globals.get("__name__") or "").split(".")[0] == "importlib":
            f = f.f_back
        if f is not None and _locked_code(f):
            _note(f"imports {name}")
        return None

    @staticmethod
    def invalidate_caches():
        return None


sys.argv = [_cfg["scorer"], *_cfg["args"]]
sys.path.insert(0, _raw)
sys.meta_path.append(_Miss)
sys.addaudithook(_hook)
try:
    runpy.run_path(os.path.join(_raw, _cfg["scorer"].replace("/", _sep)), run_name="__main__")
except SystemExit:
    _say({"t": "done"})
    raise
_say({"t": "done"})
'''


def _network() -> Tuple[List[str], str]:
    """``(argv prefix, label)``: on Linux, ``unshare -rn`` when it runs here (a user and network
    namespace, so the scorer has a loopback device only), else nothing and the reason it is not
    isolated."""
    if not sys.platform.startswith("linux"):
        return [], "not isolated"
    exe = shutil.which("unshare")
    if exe is None:
        return [], "not isolated (no unshare on PATH)"
    try:
        probe = subprocess.run([exe, "-rn", "true"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return [], "not isolated (unshare -rn refused here)"
    if probe.returncode != 0:
        return [], "not isolated (unshare -rn refused here)"
    return [exe, "-rn"], "removed (unshare -rn)"


def _child_env() -> Dict[str, str]:
    """The child's environment: nothing of the caller's, but SYSTEMROOT on Windows."""
    env: Dict[str, str] = {}
    if os.name == "nt":
        for key in ("SYSTEMROOT", "SystemRoot"):
            if os.environ.get(key):
                env["SYSTEMROOT"] = os.environ[key]
                break
    return env


def _place(run: str, rel: str, data: bytes) -> None:
    """Write *data* at the run-relative POSIX path *rel* (already held to ``_locked_path_problem``)."""
    target = os.path.join(run, *rel.split("/"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "xb") as fh:
        fh.write(data)


def _head(path: str, cap: int) -> Tuple[bytes, bool]:
    """The bytes of *path* up to *cap*, and whether it holds more."""
    with open(path, "rb") as fh:
        data = fh.read(cap + 1)
    return data[:cap], len(data) > cap


def _masks(run: str, root: str) -> List[Tuple[str, str]]:
    """The directories a stderr line names rather than prints (DESIGN 3.2(e): no absolute path), longest
    form ahead: the run and its parent; the interpreter's (``sys.prefix`` and its kin, and the directory
    of ``sys.executable``) as ``<python>``; the home directory as ``<home>``. Each in its own, real and
    normalised spellings, with backslashes and with slashes. A drive or file-system root is no mask."""
    named = [(run, "<run>"), (root, "<run's parent>")]
    named += [(d, "<python>") for d in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
                                        os.path.dirname(sys.executable or "")) if d]
    home = os.path.expanduser("~")
    if home and home != "~":
        named.append((home, "<home>"))
    out = set()
    for d, said in named:
        for spelt in (d, os.path.realpath(d), os.path.normcase(d)):
            for form in (spelt, spelt.replace("\\", "/")):
                if os.path.splitdrive(form)[1].strip("/\\"):
                    out.add((form, said))
    return sorted(out, key=lambda pair: (-len(pair[0]), pair[0]))


def _tail(path: str, run: str, root: str) -> str:
    """The last non-empty line of the stderr file, with the directories of ``_masks`` named, not printed."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.seek(max(0, size - _STDERR_TAIL))
        text = fh.read().decode("utf-8", "replace")
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
    line = lines[-1] if lines else ""
    for form, said in _masks(run, root):
        line = line.replace(form, said)
    return line[:_TAIL_CHARS]


def _watch(path: str) -> Tuple[Dict[str, Any], List[str], bool, Optional[str]]:
    """``(runtime, unlocked, done, problem)`` from the run's record. A line that is not one the
    bootstrap writes is a problem: the record is not read as clean."""
    runtime: Dict[str, Any] = {"implementation": None, "version": None, "unicode": None}
    unlocked: List[str] = []
    done, problem = False, None
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return runtime, unlocked, done, "the run left no record (its bootstrap did not start)"
    for raw in data.split(b"\n")[:-1]:
        try:
            rec = json.loads(raw.decode("ascii"))
            kind = rec["t"]
        except (ValueError, UnicodeDecodeError, KeyError, TypeError):
            problem = "the run's record holds a line its bootstrap does not write"
            continue
        if kind == "runtime":
            runtime = {k: rec.get(k) if isinstance(rec.get(k), str) else None for k in runtime}
        elif kind == "unlocked" and isinstance(rec.get("what"), str):
            unlocked.append(rec["what"])
        elif kind == "done":
            done = True
        else:
            problem = "the run's record holds a line its bootstrap does not write"
    return runtime, list(dict.fromkeys(unlocked)), done, problem


def _remove(root: str) -> bool:
    """Remove the run's directory; a file the scorer left read-only is made writable. True when gone."""
    for _ in range(10):
        shutil.rmtree(root, ignore_errors=True)
        if not os.path.exists(root):
            return True
        for top, dirs, files in os.walk(root):
            for name in dirs + files:
                try:
                    os.chmod(os.path.join(top, name), 0o700)
                except OSError:
                    pass
        time.sleep(0.05)
    return not os.path.exists(root)


def _run(spec: Dict[str, Any], blobs: Dict[str, bytes], receipt_path: str, receipt: bytes, *,
         timeout: float = _TIMEOUT_S) -> Dict[str, Any]:
    """Run the locked scorer of *spec* (``core._locks``) over one receipt, and record what it did.

    *blobs* holds the bytes of every locked path (read at the seal's commit by ``_locked_files``);
    *receipt_path* is the receipt's repository-relative path, at which its bytes are copied in. The
    paths are not checked here (``_rescore_locked`` does). Returns what ``core._run_reading`` reads:
    ``exit``, ``timed_out``, ``timeout_s``, ``done``, ``stdout``, ``stdout_over``, ``stderr_tail``,
    ``out``, ``out_problem``, ``unlocked``, with ``runtime``, ``network``, ``argv`` (as the scorer
    saw it, run-relative), ``watch_problem`` and ``removed`` (the temporary directory is gone)."""
    root = tempfile.mkdtemp(prefix="prereg-ci-run-")
    try:
        run = os.path.join(root, "run")
        os.mkdir(run)
        os.mkdir(os.path.join(run, _core._RUN_DIR))
        for rel in spec["locked"]:
            _place(run, rel, blobs[rel])
        _place(run, receipt_path, receipt)
        out_rel = _OUT if _core._OUT_ARG in spec["argv"] else None
        args = [receipt_path if a == _core._RECEIPT_ARG else out_rel if a == _core._OUT_ARG else a
                for a in spec["argv"]]
        watch, cfg = os.path.join(root, "watch.jsonl"), os.path.join(root, "config.json")
        conf = {"watch": watch, "scorer": spec["scorer"], "args": args,
                "read": [*spec["locked"], receipt_path], "write": [out_rel] if out_rel else []}
        with open(cfg, "wb") as fh:
            fh.write(json.dumps(conf, sort_keys=True).encode("utf-8"))
        prefix, network = _network()
        argv = [*prefix, sys.executable, "-I", "-S", "-B", "-X", "utf8", "-c", _BOOTSTRAP, cfg]
        so_path, se_path = os.path.join(root, "stdout"), os.path.join(root, "stderr")
        with open(so_path, "wb") as so, open(se_path, "wb") as se:
            try:
                done_ = subprocess.run(argv, cwd=run, env=_child_env(), stdin=subprocess.DEVNULL, stdout=so,
                                       stderr=se, timeout=timeout)
                code: Optional[int] = done_.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        stdout, over = _head(so_path, _STDOUT_CAP)
        tail = _tail(se_path, run, root)
        runtime, unlocked, done, problem = _watch(watch)
        out, out_problem = None, None
        if out_rel is not None:
            target = os.path.join(run, *out_rel.split("/"))
            if os.path.islink(target) or (os.path.exists(target) and not os.path.isfile(target)):
                out_problem = f"the scorer left something other than a file at {_core._OUT_ARG}"
            elif not os.path.exists(target):
                out_problem = f"the scorer wrote no {_core._OUT_ARG} file"
            else:
                out, big = _head(target, _OUT_CAP)
                if big:
                    out, out_problem = None, f"the {_core._OUT_ARG} file is over {_OUT_CAP} bytes"
    finally:
        removed = _remove(root)
    return {"exit": code, "timed_out": timed_out, "timeout_s": timeout, "done": done, "stdout": stdout,
            "stdout_over": over, "stderr_tail": tail, "out": out, "out_problem": out_problem, "unlocked": unlocked,
            "runtime": runtime, "network": network, "argv": [spec["scorer"], *args], "watch_problem": problem,
            "removed": removed}


def _receipt_problem(path: Any, spec: Dict[str, Any]) -> Optional[str]:
    """Why the receipt cannot sit at *path* in the run, or None: the locked paths' rules, and not on,
    above or below a locked file."""
    why = _core._locked_path_problem(path)
    if why is not None:
        return why
    fold = path.casefold()
    for p in spec["locked"]:
        q = p.casefold()
        if q == fold:
            return f"is the locked file {p!r}"
        if q.startswith(fold + "/") or fold.startswith(q + "/"):
            return f"and the locked file {p!r} would be a file and a directory at once"
    return None


def _locked_files(repo: Any, commit: str, spec: Dict[str, Any]) -> Dict[str, bytes]:
    """The bytes of every locked path at *commit* (a ``git._Repo``): the blobs the run materialises.
    A path that is not a file there is ``LOCKED_FILE_ABSENT`` (a FAIL: the sealed decision cannot
    run); a symlink, a submodule or a blob over the cap refuse as the git layer refuses them."""
    c = repo.commit(commit)
    out: Dict[str, bytes] = {}
    for p in spec["locked"]:
        data = repo.read(c, p)
        if data is None:
            raise _git._Refusal("LOCKED_FILE_ABSENT", f"{p} is locked by the decision and is not a file at {c[:12]}")
        out[p] = data
    return out


def _rescore_locked(prereg: Any, receipt_path: str, receipt: Any, files: Dict[str, bytes], *,
                    timeout: float = _TIMEOUT_S) -> Dict[str, Any]:
    """``core.rescore``'s reading for a locks decision, by running its scorer (DESIGN 3.4 C5).

    *prereg* is the PREREG's bytes (the decision in effect), *receipt_path* and *receipt* the
    receipt's repository-relative path and bytes, *files* the locked files' bytes at the seal
    (``_locked_files``). Returns every key ``rescore`` returns (form "locks", the decision digest,
    ``code``, ``detail``, ``verdict``, ``gates``; the cover keys stay None) plus ``output`` (the JSON
    the verdict was read from) and ``run``: the locked files and the receipt by path and sha256 of
    the raw bytes, the scorer's argv, the runtime's three fields, the network reading, what is
    watched, the exit, the reaches past the locks, the last stderr line and the ``{out}`` bytes.
    A PREREG with no locks decision, a block ``_locks`` refuses, a receipt path the run cannot hold,
    or a locked file not handed over runs nothing (``run`` None)."""
    out: Dict[str, Any] = dict.fromkeys(_core._RESCORE_KEYS)
    out.update(output=None, run=None)

    def refuse(detail: str) -> Dict[str, Any]:
        out.update(code=_core._SCORER_REFUSES, detail=detail)
        return out

    decision = _core._decision(prereg)
    if decision["form"] is None:
        out.update(code=decision["code"], detail=decision["detail"])
        return out
    out.update(form=decision["form"], gates_sha256=decision["gates_sha256"])
    if decision["form"] != "locks":
        return refuse(f"the PREREG's decision is a ```{decision['form']} table, which core.rescore scores; the "
                      f"locked scorer run reads a ```locks decision")
    try:
        spec = _core._locks(decision["block"])
    except _core._GateSpecError as e:
        return refuse(str(e))
    why = _receipt_problem(receipt_path, spec)
    if why is not None:
        return refuse(f"the receipt path {receipt_path!r} {why}, so the run cannot hold the receipt there")
    missing = [p for p in spec["locked"] if p not in files]
    if missing:
        return refuse(f"the locked files {missing} were not handed to the run; each is read at the seal's commit")
    if not isinstance(receipt, (bytes, bytearray)):
        raise TypeError(f"a receipt is bytes (a blob), not {type(receipt).__name__}")
    receipt = bytes(receipt)
    raw = _run(spec, files, receipt_path, receipt, timeout=timeout)
    reading = _core._run_reading(spec, raw)
    out.update(code=reading["code"], detail=reading["detail"], verdict=reading["verdict"], gates=reading["gates"],
               output=reading["output"])
    out["run"] = {
        "locked": [{"path": p, "sha256": hashlib.sha256(files[p]).hexdigest()} for p in spec["locked"]],
        "receipt": {"path": receipt_path, "sha256": hashlib.sha256(receipt).hexdigest()},
        "argv": raw["argv"], "runtime": raw["runtime"], "network": raw["network"], "watched": _WATCHED,
        "timeout_s": raw["timeout_s"], "exit": raw["exit"], "timed_out": raw["timed_out"], "done": raw["done"],
        "unlocked": raw["unlocked"], "stderr_tail": raw["stderr_tail"],
        "stdout_sha256": hashlib.sha256(raw["stdout"]).hexdigest(),
        "out_bytes": raw["out"], "out_sha256": hashlib.sha256(raw["out"]).hexdigest() if raw["out"] is not None
        else None, "removed": raw["removed"]}
    return out
