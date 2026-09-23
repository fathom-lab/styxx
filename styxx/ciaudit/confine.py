# -*- coding: utf-8 -*-
"""`styxx ci-audit`, confined (SWALLOW-15).

The audit reads a workflow by running each step's shell with its tools stubbed, in a temporary
directory, with an empty environment. The shell itself is real, and so is whatever the stubs do not
cover -- `rm`, `mkdir`, a redirect -- so a step that names a path outside its temporary directory
acts on the machine, and an empty value can make a scoped path the root: `actions/setup-node`'s
`rm -rf $RUNNER_TOOL_CACHE/*` is `rm -rf /*` in the simulation (SWALLOW-14 §0).

`run(fn)` runs the part of the audit that simulates in a child process that confines itself first,
with Linux Landlock (5.13+; no root, no namespaces, nothing to install), then calls `fn` and hands its
result back. From then on, that process and every process it starts -- every simulated step --

  - may write, create, rename, truncate or delete only beneath a scratch directory of its own
    (made for it, removed after), and write to /dev/null and its kin -- the rest of the machine
    is read-only to it;
  - may not open or accept a TCP connection (Landlock ABI 4+);
  - may not signal, or connect to an abstract socket of, any process outside the audit (ABI 6+).

What it does not cover, said here so the card can say it: reads (a step may read what its user can
read; with no TCP it has no channel to send it but UDP); a file's mode, owner, timestamps and
extended attributes, which Landlock does not control (`chmod -R 777 /` still applies -- as root, to
the machine; as a user, to that user's files); and kernels without Landlock, where `run` raises
`Unconfinable` and the caller decides.
"""
from __future__ import annotations

import ctypes
import os
import pickle
import shutil
import signal
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Iterable

SCHEMA = "styxx.ci-audit-confinement/v1"

# <linux/landlock.h>; the three syscalls have the same numbers on every architecture
_CREATE_RULESET, _ADD_RULE, _RESTRICT_SELF = 444, 445, 446
_CREATE_RULESET_VERSION = 1 << 0
_RULE_PATH_BENEATH = 1
FS_WRITE_FILE = 1 << 1
FS_REMOVE_DIR, FS_REMOVE_FILE = 1 << 4, 1 << 5
FS_MAKE_CHAR, FS_MAKE_DIR, FS_MAKE_REG, FS_MAKE_SOCK = 1 << 6, 1 << 7, 1 << 8, 1 << 9
FS_MAKE_FIFO, FS_MAKE_BLOCK, FS_MAKE_SYM = 1 << 10, 1 << 11, 1 << 12
FS_REFER, FS_TRUNCATE = 1 << 13, 1 << 14                      # ABI 2, ABI 3
NET_BIND_TCP, NET_CONNECT_TCP = 1 << 0, 1 << 1                # ABI 4
SCOPE_ABSTRACT_UNIX_SOCKET, SCOPE_SIGNAL = 1 << 0, 1 << 1     # ABI 6
_PR_SET_NO_NEW_PRIVS, _PR_SET_PDEATHSIG = 38, 1

# written to by ordinary scripts (`> /dev/null`, a terminal); never created, removed or renamed
DEVICES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/tty", "/dev/ptmx", "/dev/pts")
UNCONFINED_ENV = "STYXX_CIAUDIT_UNCONFINED"


class Unconfinable(RuntimeError):
    """This machine cannot confine the simulation (no Landlock)."""


class ConfinedError(RuntimeError):
    """The confined part of the audit raised; the message carries its type, text and traceback."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64), ("scoped", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def _sys(libc, *args) -> int:
    """syscall(2) with every integer passed as a long (it is variadic)."""
    return libc.syscall(*[ctypes.c_long(a) if isinstance(a, int) else a for a in args])


def abi() -> int:
    """The Landlock ABI this kernel offers, 0 when it offers none (or this is not Linux)."""
    if not sys.platform.startswith("linux"):
        return 0
    try:
        v = _sys(_libc(), _CREATE_RULESET, None, 0, _CREATE_RULESET_VERSION)
    except (OSError, AttributeError):
        return 0
    return v if v > 0 else 0


def handled(v: int) -> dict:
    """What a ruleset for ABI `v` denies unless a rule allows it."""
    fs = (FS_WRITE_FILE | FS_REMOVE_DIR | FS_REMOVE_FILE | FS_MAKE_CHAR | FS_MAKE_DIR | FS_MAKE_REG | FS_MAKE_SOCK | FS_MAKE_FIFO
          | FS_MAKE_BLOCK | FS_MAKE_SYM)
    if v >= 2:
        fs |= FS_REFER
    if v >= 3:
        fs |= FS_TRUNCATE
    net = (NET_BIND_TCP | NET_CONNECT_TCP) if v >= 4 else 0
    scoped = (SCOPE_ABSTRACT_UNIX_SOCKET | SCOPE_SIGNAL) if v >= 6 else 0
    return {"fs": fs, "net": net, "scoped": scoped}


def restrict(writable: Iterable[str | os.PathLike], devices: Iterable[str] = DEVICES) -> dict:
    """Confine this process, irreversibly, and everything it starts: writes only beneath `writable`
    (and to `devices`), no TCP, no signal or abstract socket outside the domain (as the ABI allows).
    Returns what was applied."""
    v = abi()
    if not v:
        raise Unconfinable("this kernel has no Landlock")
    libc = _libc()
    h = handled(v)
    attr = _RulesetAttr(h["fs"], h["net"], h["scoped"])
    size = 8 if v < 4 else (16 if v < 6 else 24)
    fd = _sys(libc, _CREATE_RULESET, ctypes.byref(attr), size, 0)
    if fd < 0:
        e = ctypes.get_errno()
        raise OSError(e, f"landlock_create_ruleset: {os.strerror(e)}")
    dev_rights = FS_WRITE_FILE | (FS_TRUNCATE if v >= 3 else 0)
    applied: dict = {"schema": SCHEMA, "confined": True, "mechanism": "landlock", "abi": v, "writable": [], "devices": [],
                     "network": "no TCP" if h["net"] else "not restricted (Landlock ABI < 4)",
                     "signals": "none outside the audit" if h["scoped"] else "not scoped (Landlock ABI < 6)"}
    try:
        for p, rights, key in [(p, h["fs"], "writable") for p in writable] + [(p, dev_rights, "devices") for p in devices]:
            try:
                pfd = os.open(str(p), os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue                                        # a device this machine lacks
            try:
                r = rights if os.path.isdir(str(p)) else (rights & dev_rights)
                rule = _PathBeneathAttr(r, pfd)
                if _sys(libc, _ADD_RULE, fd, _RULE_PATH_BENEATH, ctypes.byref(rule), 0) != 0:
                    e = ctypes.get_errno()
                    raise OSError(e, f"landlock_add_rule {p}: {os.strerror(e)}")
                applied[key].append(str(p))
            finally:
                os.close(pfd)
        if libc.prctl(ctypes.c_int(_PR_SET_NO_NEW_PRIVS), ctypes.c_ulong(1), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
            e = ctypes.get_errno()
            raise OSError(e, f"prctl(PR_SET_NO_NEW_PRIVS): {os.strerror(e)}")
        if _sys(libc, _RESTRICT_SELF, fd, 0) != 0:
            e = ctypes.get_errno()
            raise OSError(e, f"landlock_restrict_self: {os.strerror(e)}")
    finally:
        os.close(fd)
    return applied


def unconfined_allowed(env=None) -> bool:
    return ((env if env is not None else os.environ).get(UNCONFINED_ENV) or "").strip().lower() in ("1", "true", "yes")


def run(fn: Callable, *args, **kwargs):
    """`fn(*args, **kwargs)` in a child process confined by `restrict([its scratch directory])`,
    with TMPDIR and HOME pointed at that directory. Returns `(result, confinement)`. The result
    must pickle. Raises `Unconfinable` without Landlock, `ConfinedError` when `fn` raises, and
    RuntimeError when the child dies without an answer. Call it from a single-threaded process."""
    if not abi():
        raise Unconfinable("this kernel has no Landlock")
    scratch = tempfile.mkdtemp(prefix="ciaudit-confined-")
    parent = os.getpid()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                                  # the child: confine, run, answer, leave
        code = 0
        try:
            os.close(r)
            libc = _libc()
            libc.prctl(ctypes.c_int(_PR_SET_PDEATHSIG), ctypes.c_ulong(signal.SIGKILL), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0))
            if os.getppid() != parent:
                os._exit(3)
            os.environ["TMPDIR"] = scratch
            os.environ["HOME"] = scratch
            tempfile.tempdir = scratch
            info = restrict([scratch])
            out = ("ok", fn(*args, **kwargs), info)
        except BaseException as e:  # noqa: BLE001 -- carried to the parent, never swallowed
            out = ("error", f"{type(e).__name__}: {e}", traceback.format_exc()[-6000:])
            code = 1
        try:
            data = pickle.dumps(out)
        except Exception as e:  # noqa: BLE001
            data = pickle.dumps(("error", f"the result does not pickle: {type(e).__name__}: {e}", ""))
            code = 1
        try:
            with os.fdopen(w, "wb") as f:
                f.write(data)
        finally:
            os._exit(code)
    os.close(w)
    try:
        with os.fdopen(r, "rb") as f:
            data = f.read()
        _, status = os.waitpid(pid, 0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if not data:
        raise RuntimeError(f"the confined audit ended without an answer (wait status {status})")
    kind, payload, extra = pickle.loads(data)
    if kind == "error":
        raise ConfinedError(f"{payload}\n{extra}".rstrip())
    return payload, extra


def describe(info: dict | None) -> str:
    """One line for the card."""
    if not info or not info.get("confined"):
        why = (info or {}).get("why") or "run with --unconfined"
        return f"UNCONFINED ({why}): the steps' shell ran on this machine"
    return (f"confined (Landlock ABI {info['abi']}): the steps' shell could write only in its own scratch directory; "
            f"network: {info['network']}; signals: {info['signals']}")
