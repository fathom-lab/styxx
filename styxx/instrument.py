"""styxx.instrument: the identity of the instrument that wrote, or re-ran, a diffgate record.

AMENDMENT_capsule_v02_instrument_identity_2026_10_11 (papers/closed-model-frontier). A diffgate record
keeps ``"diffgate": "v0"`` as the format key of its content and may carry two identity keys beside its
eleven content keys:

* ``build``: the bytes that wrote it. ``schema``, ``reader`` (``styxx.diffgate``), ``styxx`` (the version
  string, recorded and never compared: one version string has named two different readers, see the
  HANDOFF note of 2026-10-11) and ``modules``, ordered ``[name, sha256]`` pairs over RECORD_MODULES, each
  sha over the module file's bytes with CRLF read as LF (a lone CR is kept), the rule
  ``web/gate/differential/py_side.py`` and ``styxx.charon._content_sha256`` follow. No path is written.
* ``runtime``: what ran it. ``implementation``, ``version``, ``unicode`` (``unicodedata.unidata_version``;
  one Unicode database feeds ``str``, ``re`` and ``int()`` in CPython) and ``case_floor``, a label from
  CASE_PROBES that a browser can compute too, recorded and never compared. ``sys.version`` is not
  recorded anywhere (the operator's decision O9, 2026-10-11).

The comparison (``delta``) reads the compared fields only: ``build.reader``, ``build.modules`` by name,
``runtime.implementation``, ``runtime.version`` and ``runtime.unicode``. A null equals nothing, so no
"same" is claimed from an unnamed field and no skew is claimed from one. Anything this runtime cannot
know is null, never a guess. Every reader here answers hostile bytes with a reading, never a traceback.

``python -m styxx.instrument`` prints the identity this install would verify under.

This module imports nothing else from styxx at import time (``styxx._version`` lazily), so it has no
import cycle with ``styxx.diffgate``, ``styxx.capsule`` or ``styxx.charon``, which all read it.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import platform
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

IDENTITY_KEYS = ("build", "runtime")
FORMAT_KEY = "diffgate"
BUILD_SCHEMA = "styxx.diffgate/instrument/v1"
READER = "styxx.diffgate"
BUILD_COMPARED = ("reader", "modules")
RUNTIME_COMPARED = ("implementation", "version", "unicode")
# every module whose bytes decide what a record says: the reader, this module, DECLARE-1's reading, the
# evidence leg, and claimdetect (it fills only unparsed_claims, and charon already hashes it)
RECORD_MODULES = ("styxx.diffgate", "styxx.instrument", "styxx.declare", "styxx.evidence", "styxx.claimdetect")
# case pairs (upper -> lower) by the Unicode version that introduced them; the floor is the highest label
# whose pairs and every lower label's pairs hold under this runtime's case tables, "<13.0" otherwise
CASE_PROBES = (
    ("13.0", ((0xA7C7, 0xA7C8),)),
    ("14.0", ((0x2C2F, 0x2C5F), (0x10570, 0x10597))),
    ("16.0", ((0x1C89, 0x1C8A), (0x10D50, 0x10D70), (0xA7CB, 0x0264))),
)

_MAX_VALUE = 128          # a compared value is a string of at most this many characters, or null
_MAX_MODULES = 16
_MODULE_NAME = re.compile(r"^[a-z_.]{1,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUOTE = 80               # any quoted value is cut to this many characters
_LABEL = re.compile(r"[^A-Za-z0-9._+-]")


# ------------------------------------------------------------------------------------- hashing

def lf_sha256(path) -> Optional[str]:
    """sha256 of the file's bytes with every CRLF read as LF; a lone CR is kept. None when the path
    cannot be read (absent, a directory, no path at all): null, never a guess."""
    try:
        with open(path, "rb") as fh:
            b = fh.read()
    except Exception:  # noqa: BLE001 - whatever stops the read, the answer is "not known"
        return None
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _py_file_sha(path) -> Optional[str]:
    """lf_sha256 of a ``.py`` source file; a compiled-only module, or no file, is null."""
    if not isinstance(path, str) or not path.lower().endswith(".py"):
        return None
    return lf_sha256(path)


def module_sha(name: str) -> Optional[str]:
    """The LF sha256 of the module's source file, found through ``importlib.util.find_spec``: the imported
    module's spec when there is one, else the finder's, which executes nothing. None when the module is
    absent, has no readable ``.py`` origin, or the read fails."""
    try:
        spec = importlib.util.find_spec(name)
    except Exception:  # noqa: BLE001 - absent parent, a None in sys.modules, a spec that is None
        return None
    origin = getattr(spec, "origin", None) if spec is not None else None
    return _py_file_sha(origin)


# ------------------------------------------------------------------------------------- runtime

def case_floor() -> str:
    floor = "<13.0"
    for label, pairs in CASE_PROBES:
        if all(chr(a).lower() == chr(b) for a, b in pairs):
            floor = label
        else:
            break
    return floor


def _text_or_none(fn) -> Optional[str]:
    try:
        v = fn()
    except Exception:  # noqa: BLE001 - what the runtime cannot say is null
        return None
    return v if isinstance(v, str) else None


def runtime() -> dict:
    """What runs this process. Every value a string or null; never ``sys.version``."""
    return {"implementation": _text_or_none(lambda: sys.implementation.name),
            "version": _text_or_none(platform.python_version),
            "unicode": _text_or_none(lambda: unicodedata.unidata_version),
            "case_floor": _text_or_none(case_floor)}


def _styxx_version() -> Optional[str]:
    try:
        from styxx._version import __version__
    except Exception:  # noqa: BLE001 - a staged copy of the modules without the version file
        return None
    return __version__ if isinstance(__version__, str) else None


# ------------------------------------------------------------------------------- the identity

_cache: Dict[Optional[str], dict] = {}


def record_identity(self_file) -> dict:
    """``{"build", "runtime"}`` for the reader whose source file is ``self_file`` (the reader's own
    ``__file__``, never a module looked up by name: a reader built from a string names a null sha, not
    another file's bytes). Computed once per path and copied on every call, so no caller edits another
    record's identity."""
    try:
        key = str(self_file) if self_file else None
    except Exception:  # noqa: BLE001
        key = None
    if key not in _cache:
        modules: List[list] = []
        for name in RECORD_MODULES:
            if name == READER:
                sha = _py_file_sha(key)
            elif name == __name__:
                sha = _py_file_sha(globals().get("__file__"))
            else:
                sha = module_sha(name)
            modules.append([name, sha])
        _cache[key] = {"build": {"schema": BUILD_SCHEMA, "reader": READER, "styxx": _styxx_version(),
                                 "modules": modules},
                       "runtime": runtime()}
    return copy.deepcopy(_cache[key])


def content_keys(record) -> list:
    """The record's keys that are neither the format key nor an identity key, in the record's order."""
    if not isinstance(record, dict):
        return []
    return [k for k in record if k != FORMAT_KEY and k not in IDENTITY_KEYS]


# --------------------------------------------------------------------------------- reading one

def _cut(value) -> str:
    """A value quoted in ASCII and cut to _QUOTE characters: hostile bytes reach no terminal raw."""
    try:
        s = json.dumps(value, ensure_ascii=True, default=repr)
    except (ValueError, TypeError, RecursionError):
        s = "(a %s this reader does not print)" % type(value).__name__
    s = s.encode("ascii", "backslashreplace").decode("ascii")
    return s if len(s) <= _QUOTE else s[:_QUOTE] + "..."


def _text_field(obj: dict, field: str) -> Tuple[bool, Optional[str]]:
    v = obj.get(field)
    if v is None:
        return True, None
    if isinstance(v, str) and len(v) <= _MAX_VALUE:
        return True, v
    return False, None


def _read_modules(value) -> Optional[List[list]]:
    """At most _MAX_MODULES ``[name, sha]`` pairs with distinct names, or None when unreadable."""
    if not isinstance(value, list) or len(value) > _MAX_MODULES:
        return None
    out, seen = [], set()
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        name, sha = pair
        if not isinstance(name, str) or not _MODULE_NAME.match(name) or name in seen:
            return None
        if sha is not None and not (isinstance(sha, str) and _SHA256.match(sha)):
            return None
        seen.add(name)
        out.append([name, sha])
    return out


def read_identity(record) -> Tuple[Optional[dict], List[str]]:
    """The identity a record carries, normalised, and what made it unreadable.

    ``(None, [])`` when the record carries no identity key. ``(None, [why, ...])`` when an identity key
    is not an object, names an unknown schema, holds a compared value that is neither a string of at most
    128 characters nor null, or lists modules otherwise than as at most 16 ``[name, sha]`` pairs (name
    ``^[a-z_.]{1,64}$``, sha 64 lowercase hex or null): read as unnamed, with each ``why`` naming the key
    and quoting the value cut short. Otherwise ``({"build": {...} | None, "runtime": {...} | None}, [])``
    with every known field present and null where the record says nothing."""
    if not isinstance(record, dict):
        return None, []
    present = [k for k in IDENTITY_KEYS if k in record]
    if not present:
        return None, []
    unreadable: List[str] = []
    identity: dict = {"build": None, "runtime": None}

    def bad(key, why, value):
        unreadable.append("%s is not an identity this verifier reads (%s: %s)" % (key, why, _cut(value)))

    if "build" in record:
        b = record["build"]
        if not isinstance(b, dict):
            bad("build", "not an object", b)
        elif b.get("schema") != BUILD_SCHEMA:
            bad("build", "schema is not %s" % BUILD_SCHEMA, b.get("schema"))
        else:
            ok, reader = _text_field(b, "reader")
            if not ok:
                bad("build", "reader is not a string of at most %d characters or null" % _MAX_VALUE,
                    b.get("reader"))
            modules = _read_modules(b.get("modules"))
            if modules is None:
                bad("build", "modules is not a list of at most %d [name, sha] pairs" % _MAX_MODULES,
                    b.get("modules"))
            if ok and modules is not None:
                styxx = b.get("styxx")
                identity["build"] = {"schema": BUILD_SCHEMA, "reader": reader,
                                     "styxx": styxx if isinstance(styxx, str) else None,
                                     "modules": modules}
    if "runtime" in record:
        r = record["runtime"]
        if not isinstance(r, dict):
            bad("runtime", "not an object", r)
        else:
            fields = {}
            for f in RUNTIME_COMPARED:
                ok, v = _text_field(r, f)
                if not ok:
                    bad("runtime", "%s is not a string of at most %d characters or null" % (f, _MAX_VALUE),
                        r.get(f))
                    break
                fields[f] = v
            else:
                floor = r.get("case_floor")
                fields["case_floor"] = floor if isinstance(floor, str) else None
                identity["runtime"] = fields
    if unreadable:
        return None, unreadable
    return identity, []


# ------------------------------------------------------------------------------- comparing two

def _part(identity, key: str) -> dict:
    v = identity.get(key) if isinstance(identity, dict) else None
    return v if isinstance(v, dict) else {}


def _modules_by_name(part: dict) -> Dict[str, Optional[str]]:
    mods = part.get("modules")
    out: Dict[str, Optional[str]] = {}
    if isinstance(mods, list):
        for pair in mods:
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and isinstance(pair[0], str):
                out[pair[0]] = pair[1] if isinstance(pair[1], str) else None
    return out


_ABSENT = object()


def delta(recorded, live) -> dict:
    """How the recorded identity stands to the live one, over the compared fields only.

    ``state``: ``unnamed`` (the record names nothing readable); ``differs`` (a compared value is non-null
    on both sides and differs, or a module is named on one side only); ``incomplete`` (nothing differs,
    but a compared value is null on either side); ``same``. ``differs`` and ``unknown`` list the fields
    (``build.reader``, ``build.modules[<name>]``, ``runtime.version``, ...)."""
    if recorded is None:
        return {"state": "unnamed", "differs": [], "unknown": []}
    differs: List[str] = []
    unknown: List[str] = []

    def compare(field, a, b):
        if a is None or b is None:
            unknown.append(field)
        elif a != b:
            differs.append(field)

    rb, lb = _part(recorded, "build"), _part(live, "build")
    for f in BUILD_COMPARED:
        if f == "modules":
            continue
        compare("build." + f, rb.get(f), lb.get(f))
    rm, lm = _modules_by_name(rb), _modules_by_name(lb)
    # a name on one side only differs, when both sides list their modules; a side that lists none (the
    # build part absent, or its list null) leaves every name unknown
    both_listed = isinstance(rb.get("modules"), list) and isinstance(lb.get("modules"), list)
    for name in list(rm) + [n for n in lm if n not in rm]:
        a, b = rm.get(name, _ABSENT), lm.get(name, _ABSENT)
        if a is _ABSENT or b is _ABSENT:
            (differs if both_listed else unknown).append("build.modules[%s]" % name)
        else:
            compare("build.modules[%s]" % name, a, b)
    rr, lr = _part(recorded, "runtime"), _part(live, "runtime")
    for f in RUNTIME_COMPARED:
        compare("runtime." + f, rr.get(f), lr.get(f))
    state = "differs" if differs else ("incomplete" if unknown else "same")
    return {"state": state, "differs": differs, "unknown": unknown}


# ------------------------------------------------------------------------------------ one line

def _label(v) -> str:
    """A recorded value as one printable ASCII token: cut to _QUOTE characters, every character outside
    letters, digits, ``.``, ``_``, ``+`` and ``-`` written as ``?``. The result can never hold ``!=`` or a
    quote, so a problem line that carries it is never read by charon as a broken binding."""
    if not isinstance(v, str) or not v:
        return "unknown"
    return _LABEL.sub("?", v[:_QUOTE])


def describe(identity) -> str:
    """``styxx.diffgate 0986705695 (styxx 7.49.0) on cpython 3.12.10, Unicode 15.0.0``: ASCII only, because
    a Windows console pipe is cp1252. ``unknown`` stands for every null; ``not named`` for no identity."""
    if not isinstance(identity, dict):
        return "not named"
    b, r = _part(identity, "build"), _part(identity, "runtime")
    reader = _label(b.get("reader"))
    sha = _modules_by_name(b).get(b.get("reader")) if isinstance(b.get("reader"), str) else None
    sha10 = _label(sha)[:10] if isinstance(sha, str) else "sha unknown"
    return "%s %s (styxx %s) on %s %s, Unicode %s" % (
        reader, sha10, _label(b.get("styxx")), _label(r.get("implementation")), _label(r.get("version")),
        _label(r.get("unicode")))


# ---------------------------------------------------------------------------------------- the CLI

def _loaded_from() -> str:
    """``installed package``, ``source tree`` or ``unknown``: printed, never compared."""
    try:
        from pathlib import Path
        pkg = Path(sys.modules["styxx"].__file__).resolve().parent
        if pkg.parent.name in ("site-packages", "dist-packages"):
            return "installed package"
        if (pkg.parent / "pyproject.toml").is_file():
            return "source tree"
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def main(argv=None) -> int:
    """Print the identity a diffgate record would carry, and a capsule verify would re-run under, from
    this interpreter and this install: the reader's file found as the import system finds it (nothing is
    executed), the module shas, the runtime, and where styxx was loaded from."""
    try:
        spec = importlib.util.find_spec(READER)
        self_file = getattr(spec, "origin", None) if spec is not None else None
    except Exception:  # noqa: BLE001
        self_file = None
    ident = record_identity(self_file)
    out = {"schema": BUILD_SCHEMA, "styxx_version": _styxx_version(), "build": ident["build"],
           "runtime": ident["runtime"], "loaded_from": _loaded_from()}
    print(json.dumps(out, indent=1, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
