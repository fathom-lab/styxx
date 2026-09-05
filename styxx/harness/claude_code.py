# -*- coding: utf-8 -*-
"""Claude Code hook adapter: a PostToolUse stager and a Stop finaliser, at L1.

This adapter is blind, permanently, to files written by shell commands. ``cat > file``, a
heredoc, ``python -c 'open(...).write(...)'``, ``git apply``: the hook sees a Bash tool call and
its stdout, never the file. Those bytes never enter ``authored_sha256``, and a later ``Read`` of
that file mints a ``file_read`` receipt the verifier will accept. The hook is not a filesystem
monitor and must not pretend to be one; the gap is printed into every manifest's harness string
beside the rung. A second, smaller gap is printed with it: a tool's response can differ from the
bytes on disk (a trimmed trailing newline, a preserved CRLF, an edit matched after whitespace
normalisation), so set membership over exact bytes can miss a file the agent did write. Reading
the file from disk right after Write and Edit narrows that gap and does not close it.

Two events, two costs, one manifest. PostToolUse fires once per tool call, for parallel tool
calls in one assistant message and inside subagents, all under one ``session_id``; this module
is therefore stdlib-only at import (the entry script loads it by file path so ``styxx/__init__``
never runs on that path) and ``stage_event`` writes one JSON file per event, atomically, named by
``tool_use_id``. ``finalise`` (Stop) imports ``styxx.sworn`` lazily and folds the staged events
into ``<dir>/<session_id>.manifest.json`` deterministically: receipt ids in order of (capture
time, tool_use_id), ``minted_at`` the capture time of the last event, so a re-fold of the same
events is byte-identical. Stop fires every turn; the fold is idempotent.

Where the files go: a directory that must lie OUTSIDE the workspace — under neither the payload's
``cwd`` nor ``CLAUDE_PROJECT_DIR`` — or nothing is written and stderr says why. A manifest the
agent can Read into a receipt of itself is L0 wearing an L1 label. The rule is hygiene, not
trust: the agent's shell can write there too, which is what L1 means.

What is read, tolerantly (the documentation and this box's transcripts disagree on shapes):

  Bash      tool_response.stdout / .stderr (or a string response) -> tool_stdout / tool_stderr;
            complete unless interrupted or carrying a truncation marker or persisted-output stub
  Read      tool_response.file.content, else .content, else a string -> file_read; complete only
            when the window is the whole file and tool_input set no offset/limit; images: nothing
  WebFetch  tool_response.result, else .content, else a string -> http_fetch; never complete
  Write     nothing minted; authored_sha256 <- tool_input.content and the file on disk
  Edit      nothing minted; authored_sha256 <- tool_input.new_string, the post-edit text
            reconstructed from tool_response.originalFile (replace_all honoured), the file on disk
  Write/Edit with a string response: the tool failed; nothing recorded
  ignored:  Grep, Glob, Task, TodoWrite, MultiEdit, NotebookEdit, LS, WebSearch, every mcp__* tool

Every entry point exits zero on every input. A hook that blocks the agent is a gate without a
measured precision. Stop's stdout is parsed for a decision, so nothing is printed there.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "HARNESS", "RUNG", "RECEIPT_BYTE_CAP", "BLINDNESS", "IGNORED_TOOLS", "TRUNCATION_MARKERS",
    "valid_session_id", "native_abs", "is_inside", "manifest_dir_refusal", "default_dir",
    "read_response", "stage_event", "load_events", "finalise", "main",
]

HARNESS_VERSION = "styxx.harness.claude_code/0.1"
RUNG = "L1"
RECEIPT_BYTE_CAP = 4 * 1024 * 1024        # over this, a receipt is staged and folded hash-only

BLINDNESS = ("blind, permanently, to files written by shell commands: only Write and Edit enter "
             "authored_sha256, and a later Read of a shell-written file mints a receipt the "
             "verifier will accept")
L1_WEAK = ("L1 (weak: a local hook sharing a filesystem and a shell with the agent; the agent's "
           "shell can write what the hook later reads, and can write to the manifest directory)")
DISK_GAP = ("a tool response can differ from the bytes on disk, so set membership over exact "
            "bytes can miss a file the agent did write")
LABEL = ("adapters, never a recorder: bytes a harness already holds, turned into a "
         "sworn/manifest/0.2 file; nothing signed, fetched, observed or verified")
HARNESS = "%s · %s · %s · %s · %s" % (HARNESS_VERSION, L1_WEAK, BLINDNESS, DISK_GAP, LABEL)

IGNORED_TOOLS = ("Grep", "Glob", "Task", "TodoWrite", "MultiEdit", "NotebookEdit", "LS",
                 "WebSearch", "AskUserQuestion", "ExitPlanMode", "EnterPlanMode", "KillShell",
                 "BashOutput", "Skill", "SendUserFile")
# Substrings a truncated or spilled Bash response carries; a match makes the receipt incomplete.
TRUNCATION_MARKERS = ("lines truncated]", "characters truncated]", "persisted-output")

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DEVICE_NAMES = frozenset({"CON", "PRN", "AUX", "NUL"}
                          | {"COM%d" % i for i in range(1, 10)}
                          | {"LPT%d" % i for i in range(1, 10)})
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _say(msg: str) -> None:
    """One stderr line; never raises (a console that cannot encode a path must not become a
    traceback in a hook)."""
    try:
        sys.stderr.write("sworn-hooks: %s\n" % msg)
    except UnicodeEncodeError:
        sys.stderr.write(("sworn-hooks: %s\n" % msg).encode("ascii", "backslashreplace").decode("ascii"))
    except Exception:                                                   # pragma: no cover
        pass


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _text_bytes(x: Any) -> bytes:
    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        return x.encode("utf-8", errors="replace")
    return (json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8", errors="replace")


def _truncated(text: str) -> bool:
    return any(marker in text for marker in TRUNCATION_MARKERS)


# ----------------------------------------------------------------------------- names and paths

def valid_session_id(s: Any) -> bool:
    """A string that may become a path component: the strict pattern and no Windows device name."""
    if not isinstance(s, str) or not _SAFE_ID.fullmatch(s):
        return False
    return s.split(".")[0].upper() not in _DEVICE_NAMES


def native_abs(path: Any) -> Optional[str]:
    """normcase(realpath(path)) for a native absolute path; None for anything else — a ``\\\\?\\``
    or ``//?/`` prefix, a relative path, or (on Windows) a path with no drive such as the MSYS
    spelling ``/c/Users/...``, which realpath would silently root on the current drive."""
    s = str(path) if path is not None else ""
    if not s or s.startswith("\\\\?\\") or s.startswith("//?/") or s.startswith("\\\\.\\"):
        return None
    if os.name == "nt" and not os.path.splitdrive(s)[0]:
        return None
    if not os.path.isabs(s):
        return None
    return os.path.normcase(os.path.realpath(s))


def is_inside(candidate: Any, roots: Sequence[Any]) -> bool:
    """True when ``candidate`` resolves to one of ``roots`` or below it, by path components —
    never by string prefix, which would put ``ws-manifests`` inside ``ws``."""
    c = native_abs(candidate)
    if c is None:
        return False
    cp = Path(c)
    for r in roots:
        rn = native_abs(r)
        if rn is None:
            continue
        rp = Path(rn)
        if cp == rp or rp in cp.parents:
            return True
    return False


def manifest_dir_refusal(manifest_dir: Any, roots: Sequence[Any]) -> Optional[str]:
    """None when the directory may be used; otherwise the reason it may not."""
    if native_abs(manifest_dir) is None:
        return ("manifest directory %r is not a native absolute path (no \\\\?\\ prefix, no "
                "relative path, no drive-less spelling); nothing written" % (manifest_dir,))
    usable = [r for r in roots if native_abs(r) is not None]
    if not usable:
        return "no usable workspace root to check the manifest directory against (payload cwd absent); nothing written"
    if is_inside(manifest_dir, usable):
        return ("manifest directory %s lies inside the workspace (%s); a manifest the agent can "
                "Read into a receipt of itself is L0 wearing an L1 label; nothing written"
                % (manifest_dir, ", ".join(str(r) for r in usable)))
    return None


def default_dir(env: Mapping[str, str]) -> Optional[str]:
    """STYXX_SWORN_MANIFEST_DIR, else a per-user data directory. Takes the environment as an
    argument so that only ``main`` touches ``os.environ``."""
    chosen = env.get("STYXX_SWORN_MANIFEST_DIR")
    if chosen:
        return chosen
    if os.name == "nt":
        base = env.get("LOCALAPPDATA")
        return os.path.join(base, "styxx", "sworn-manifests") if base else None
    home = env.get("HOME")
    return os.path.join(home, ".local", "share", "styxx", "sworn-manifests") if home else None


# ----------------------------------------------------------------------------- the tool table

def _disk_sha256(path: Any) -> Optional[str]:
    if not isinstance(path, str) or not path:
        return None
    try:
        return _sha256(Path(path).read_bytes())
    except (OSError, ValueError):
        return None


def read_response(tool_name: str, tool_input: Any, tool_response: Any) -> Tuple[List[dict], List[str]]:
    """The table in the docstring. Returns (receipts, authored_sha256) where each receipt is
    ``{kind, data, complete, note}`` with ``data`` in bytes."""
    ti: Dict[str, Any] = tool_input if isinstance(tool_input, dict) else {}
    receipts: List[dict] = []
    authored: List[str] = []

    if tool_name == "Bash":
        command = ti.get("command")
        note = command if isinstance(command, str) else ""
        if isinstance(tool_response, dict):
            out, err = tool_response.get("stdout"), tool_response.get("stderr")
            out_s = out if isinstance(out, str) else ("" if out is None else _text_bytes(out).decode("utf-8", "replace"))
            err_s = err if isinstance(err, str) else ""
            interrupted = bool(tool_response.get("interrupted"))
        elif isinstance(tool_response, str):
            out_s, err_s, interrupted = tool_response, "", False
        else:
            out_s = _text_bytes(tool_response).decode("utf-8", "replace")
            err_s, interrupted = "", False
            note = (note + " (response shape not recognised; serialised whole)").strip()
        receipts.append({"kind": "tool_stdout", "data": out_s.encode("utf-8", "replace"),
                         "complete": not interrupted and not _truncated(out_s), "note": note})
        if err_s:
            receipts.append({"kind": "tool_stderr", "data": err_s.encode("utf-8", "replace"),
                             "complete": not interrupted and not _truncated(err_s), "note": note})
        return receipts, authored

    if tool_name == "Read":
        path = ti.get("file_path")
        note = path if isinstance(path, str) else ""
        windowed = ("offset" in ti and ti.get("offset") is not None) or \
                   ("limit" in ti and ti.get("limit") is not None)
        whole = True
        content: Any = None
        if isinstance(tool_response, dict):
            f = tool_response.get("file")
            if isinstance(f, dict):
                if "base64" in f or f.get("type") == "image":
                    return receipts, authored                          # an image: nothing
                content = f.get("content")
                start, num, total = f.get("startLine"), f.get("numLines"), f.get("totalLines")
                if f.get("truncatedByTokenCap"):
                    whole = False
                if isinstance(start, int) and start not in (0, 1):
                    whole = False
                if isinstance(num, int) and isinstance(total, int) and num != total:
                    whole = False
            elif isinstance(tool_response.get("content"), str):
                content = tool_response.get("content")                 # the documented shape
        elif isinstance(tool_response, str):
            content = tool_response
        if not isinstance(content, str):
            content = _text_bytes(tool_response).decode("utf-8", "replace")
            note = (note + " (response shape not recognised; serialised whole)").strip()
        receipts.append({"kind": "file_read", "data": content.encode("utf-8", "replace"),
                         "complete": whole and not windowed and not _truncated(content), "note": note})
        return receipts, authored

    if tool_name == "WebFetch":
        url = ti.get("url")
        note = url if isinstance(url, str) else ""
        content = None
        if isinstance(tool_response, dict):
            for key in ("result", "content"):
                if isinstance(tool_response.get(key), str):
                    content = tool_response[key]
                    break
        elif isinstance(tool_response, str):
            content = tool_response
        if content is None:
            content = _text_bytes(tool_response).decode("utf-8", "replace")
            note = (note + " (response shape not recognised; serialised whole)").strip()
        receipts.append({"kind": "http_fetch", "data": content.encode("utf-8", "replace"),
                         "complete": False, "note": note})            # a rendering, never the page
        return receipts, authored

    if tool_name == "Write":
        if isinstance(tool_response, str):
            return receipts, authored                                  # the tool failed
        content = ti.get("content")
        if isinstance(content, str):
            authored.append(_sha256(content.encode("utf-8", "replace")))
        h = _disk_sha256(ti.get("file_path"))
        if h:
            authored.append(h)
        return receipts, authored

    if tool_name == "Edit":
        if isinstance(tool_response, str):
            return receipts, authored
        old, new = ti.get("old_string"), ti.get("new_string")
        if isinstance(new, str):
            authored.append(_sha256(new.encode("utf-8", "replace")))
        if isinstance(tool_response, dict) and isinstance(old, str) and isinstance(new, str):
            original = tool_response.get("originalFile")
            if isinstance(original, str):
                replace_all = bool(ti.get("replace_all") or tool_response.get("replaceAll"))
                rebuilt = original.replace(old, new) if replace_all else original.replace(old, new, 1)
                authored.append(_sha256(rebuilt.encode("utf-8", "replace")))
        h = _disk_sha256(ti.get("file_path"))
        if h:
            authored.append(h)
        return receipts, authored

    return receipts, authored                                          # ignored, by name


# ----------------------------------------------------------------------------- staging (PostToolUse)

def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp-%d" % os.getpid())
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def stage_event(payload: Any, manifest_dir: Any, extra_roots: Sequence[Any] = (),
                now: Optional[_dt.datetime] = None) -> Optional[Path]:
    """Stage one PostToolUse payload as ``<dir>/<session_id>/events/<stamp>-<tool_use_id>.json``.
    Returns the file written, or None (with the reason on stderr, or silently when the tool is
    one this adapter ignores)."""
    if not isinstance(payload, dict):
        _say("payload is not a JSON object; nothing written")
        return None
    sid = payload.get("session_id")
    if not valid_session_id(sid):
        _say("session_id %r is not a safe path component; nothing written" % (sid,))
        return None
    why = manifest_dir_refusal(manifest_dir, [payload.get("cwd")] + list(extra_roots))
    if why:
        _say(why)
        return None
    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str):
        _say("payload carries no tool_name; nothing written")
        return None
    receipts, authored = read_response(tool_name, payload.get("tool_input"), payload.get("tool_response"))
    if not receipts and not authored:
        return None
    tuid = payload.get("tool_use_id")
    if not valid_session_id(tuid):
        tuid = "evt-" + _sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))[:24]
    t = now or _dt.datetime.now(_dt.timezone.utc)
    staged = []
    for r in receipts:
        data = r["data"]
        staged.append({"kind": r["kind"], "complete": bool(r["complete"]), "note": r["note"],
                       "sha256": _sha256(data), "bytes_len": len(data),
                       "b64": base64.b64encode(data).decode("ascii") if len(data) <= RECEIPT_BYTE_CAP else None})
    event = {
        "tool_use_id": tuid,
        "captured_at": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "captured_at_us": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "tool_name": tool_name,
        "agent_id": payload.get("agent_id"),
        "prompt_id": payload.get("prompt_id"),
        "receipts": staged,
        "authored_sha256": authored,
    }
    events_dir = Path(manifest_dir) / sid / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    out = events_dir / ("%s-%s.json" % (t.strftime("%Y%m%dT%H%M%S%fZ"), tuid))
    _write_json_atomic(out, event)
    return out


def load_events(events_dir: Path) -> Tuple[List[dict], List[str]]:
    """Every staged event under ``events_dir`` sorted by (captured_at_us, tool_use_id), plus the
    names of files that could not be read as events (a gap the caller reports)."""
    events: List[dict] = []
    skipped: List[str] = []
    if not events_dir.is_dir():
        return events, skipped
    for p in sorted(events_dir.glob("*.json")):
        try:
            ev = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(ev, dict) or not isinstance(ev.get("receipts"), list):
                raise ValueError("not an event object")
            events.append(ev)
        except (OSError, ValueError):
            skipped.append(p.name)
    events.sort(key=lambda e: (str(e.get("captured_at_us") or ""), str(e.get("tool_use_id") or "")))
    return events, skipped


# ----------------------------------------------------------------------------- folding (Stop)

def finalise(payload: Any, manifest_dir: Any, extra_roots: Sequence[Any] = ()) -> Optional[Path]:
    """Fold the session's staged events into ``<dir>/<session_id>.manifest.json`` at L1. Imports
    styxx.sworn here and nowhere earlier. Idempotent: the same staged set yields the same bytes."""
    if not isinstance(payload, dict):
        _say("payload is not a JSON object; nothing written")
        return None
    sid = payload.get("session_id")
    if not valid_session_id(sid):
        _say("session_id %r is not a safe path component; nothing written" % (sid,))
        return None
    why = manifest_dir_refusal(manifest_dir, [payload.get("cwd")] + list(extra_roots))
    if why:
        _say(why)
        return None
    from styxx.sworn import Manifest                                    # the one styxx import

    events, skipped = load_events(Path(manifest_dir) / sid / "events")
    for name in skipped:
        _say("staged event %s could not be read and is not in the manifest (a gap, not a receipt)" % name)
    minted_at = events[-1].get("captured_at") if events else None
    m = Manifest(harness=HARNESS, turn=sid, rung=RUNG,
                 minted_at=minted_at if isinstance(minted_at, str) else None)
    n = 0
    for ev in events:
        tag = " · prompt_id=%s · agent_id=%s · tool_use_id=%s" % (
            ev.get("prompt_id") or "-", ev.get("agent_id") or "main", ev.get("tool_use_id"))
        for r in ev["receipts"]:
            if not isinstance(r, dict):
                continue
            n += 1
            data = None
            if isinstance(r.get("b64"), str):
                try:
                    data = base64.b64decode(r["b64"], validate=True)
                except (ValueError, TypeError):
                    data = None
            try:
                m.add("r%d" % n, data, str(r.get("kind")), complete=bool(r.get("complete")),
                      captured_at=ev.get("captured_at"), sha256=r.get("sha256"),
                      note="%s %s%s" % (ev.get("tool_name"), r.get("note", ""), tag))
            except ValueError as exc:
                n -= 1
                _say("staged receipt in %s skipped: %s" % (ev.get("tool_use_id"), exc))
        for h in ev.get("authored_sha256") or []:
            if isinstance(h, str) and _HEX64.fullmatch(h) and h not in m.authored_sha256:
                m.authored_sha256.append(h)
    message = payload.get("last_assistant_message")
    if isinstance(message, str) and message:
        m.record_authored(message.encode("utf-8", "replace"))
    out = Path(manifest_dir) / ("%s.manifest.json" % sid)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp-%d" % os.getpid())
    m.write(tmp)
    os.replace(tmp, out)
    _say("manifest %s: %d receipts, %d authored hashes, rung %s" % (out, n, len(m.authored_sha256), RUNG))
    return out


# ----------------------------------------------------------------------------- the CLI layer

def main(argv: Optional[Sequence[str]] = None) -> int:
    """``post-tool`` stages, ``stop`` folds. Reads the hook payload from stdin. Returns zero for
    every payload, including one that cannot be read; only argparse's own usage exit differs."""
    ap = argparse.ArgumentParser(
        prog="styxx.harness claude-code",
        description="Claude Code hook adapter at L1: stage a PostToolUse event, or fold the "
                    "session's events into a sworn manifest on Stop. Exits zero on every input.")
    ap.add_argument("event", choices=["post-tool", "stop"])
    ap.add_argument("--dir", default=None,
                    help="manifest directory, outside the workspace (default: "
                         "$STYXX_SWORN_MANIFEST_DIR, else a per-user data directory)")
    a = ap.parse_args(argv)
    try:
        raw = sys.stdin.buffer.read()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("payload is not a JSON object")
    except Exception as exc:
        _say("cannot read the hook payload: %s: %s" % (type(exc).__name__, exc))
        return 0
    env = os.environ
    manifest_dir = a.dir or default_dir(env)
    if not manifest_dir:
        _say("no manifest directory: pass --dir or set STYXX_SWORN_MANIFEST_DIR; nothing written")
        return 0
    extra = [env["CLAUDE_PROJECT_DIR"]] if env.get("CLAUDE_PROJECT_DIR") else []
    try:
        if a.event == "post-tool":
            stage_event(payload, manifest_dir, extra_roots=extra)
        else:
            finalise(payload, manifest_dir, extra_roots=extra)
    except SystemExit as exc:                                           # a Manifest refusal
        _say("refused: %s" % exc)
    except Exception as exc:
        _say("%s: %s" % (type(exc).__name__, exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
