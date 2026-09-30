"""PATH-2a test helpers (not collected; NOTE_path2a_abstain_overlay_2026_09_30).

`main` here is not a vendored copy. It is this checkout's own `styxx/diffgate.py` and `web/gate/diffgate.js` with
the PATH-2a block cut out and the door hooks reverted -- the reconstruction -- asserted to hash to the files on
`origin/main` 1cde8b82. Every differential in the PATH-2a tests runs against that reconstruction, so the reference
cannot drift from the reader the branch actually carries.
"""
from __future__ import annotations

import functools
import hashlib
import json
import random
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTRUMENT = ROOT / "styxx" / "diffgate.py"
PORT = ROOT / "web" / "gate" / "diffgate.js"
DIFFERENTIAL = ROOT / "web" / "gate" / "differential"
FIXTURE = ROOT / "tests" / "fixtures" / "path2a_repros.json"

MAIN_PY_SHA = "9b620e00a19464589308a987819894ae7cc3c111c66a5f8a457a84b8a6c604eb"   # origin/main 1cde8b82, LF
MAIN_JS_SHA = "06688702999cdabe763265722a0ac14d4b9ffb40d0efcbb32339eba89f00c141"
FUZZ_SHA = "2e80cd1d5a42e867d2a2581619976580a1f492820cbfc6d446618cbb8202b91f"      # corpus_fuzz.json, never committed

_MARK = "=== PATH-2a abstain-only overlay: "
PY_BEGIN, PY_END = "# " + _MARK + "BEGIN ===", "# " + _MARK + "END ==="
JS_BEGIN, JS_END = "// " + _MARK + "BEGIN ===", "// " + _MARK + "END ==="
PY_HOOK_SUFFIX = "  # PATH-2a"

REACH = frozenset({
    ("file_created", "VERIFIED"), ("file_deleted", "VERIFIED"), ("file_touched", "VERIFIED"),
    ("files_changed_count", "VERIFIED"), ("files_changed_count", "CONTRADICTED"),
    ("only_touches", "VERIFIED"), ("only_touches", "CONTRADICTED"),
    ("tests_added", "VERIFIED"), ("tests_added", "CONTRADICTED"), ("symbol_added", "VERIFIED")})
DEFECTS = {"file_created": ("#97", "#121", "#97, #121"), "file_deleted": ("#97", "#121", "#97, #121"),
           "file_touched": ("#97", "#121", "#97, #121"), "files_changed_count": ("#121",),
           "only_touches": ("#121",), "tests_added": ("#101",), "symbol_added": ("#101",)}


def lf(path: Path) -> str:
    return path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def py_block(text: str | None = None) -> str:
    text = lf(INSTRUMENT) if text is None else text
    assert text.count(PY_BEGIN) == 1 and text.count(PY_END) == 1, "the Python block's markers"
    return text[text.index(PY_BEGIN):text.index(PY_END) + len(PY_END) + 1]


def js_block(text: str | None = None) -> str:
    text = lf(PORT) if text is None else text
    assert text.count(JS_BEGIN) == 1 and text.count(JS_END) == 1, "the port block's markers"
    return text[text.index(JS_BEGIN):text.index(JS_END) + len(JS_END) + 1]


def reconstruct_py(text: str) -> str:
    """NOTE section 2: cut the block, turn the two `g = _gate(` back into `return _gate(`, drop the hook lines."""
    assert text.count(PY_BEGIN) == 1 and text.count(PY_END) == 1
    out, n = re.subn(re.escape(PY_BEGIN) + r"\n.*?" + re.escape(PY_END) + r"\n\n\n", "", text, count=1, flags=re.S)
    assert n == 1, "the Python block is not followed by exactly two blank lines"
    assert out.count("    g = _gate(summary_text,") == 2, "the two door hooks"
    out = out.replace("    g = _gate(summary_text,", "    return _gate(summary_text,")
    lines = out.split("\n")
    assert sum(1 for x in lines if x.endswith(PY_HOOK_SUFFIX)) == 2, "the two hook lines"
    return "\n".join(x for x in lines if not x.endswith(PY_HOOK_SUFFIX))


def reconstruct_js(text: str) -> str:
    assert text.count(JS_BEGIN) == 1 and text.count(JS_END) == 1
    out, n = re.subn(re.escape(JS_BEGIN) + r"\n.*?" + re.escape(JS_END) + r"\n\n", "", text, count=1, flags=re.S)
    assert n == 1, "the port block is not followed by exactly one blank line"
    assert out.count("_gateDiffTextMain(") == 2, "the two renamed lines"
    return out.replace("_gateDiffTextMain(", "gateDiffText(")


@functools.lru_cache(maxsize=None)
def main_source() -> str:
    text = reconstruct_py(lf(INSTRUMENT))
    assert sha(text) == MAIN_PY_SHA, "the reconstructed reader is not origin/main's styxx/diffgate.py"
    return text


@functools.lru_cache(maxsize=None)
def main_port_source() -> str:
    text = reconstruct_js(lf(PORT))
    assert sha(text) == MAIN_JS_SHA, "the reconstructed port is not origin/main's web/gate/diffgate.js"
    return text


def module_from(source: str, name: str):
    """Execute `source` as the module styxx.<name>, so `from .declare import ...` resolves in the package."""
    import styxx  # noqa: F401  (the package the relative import resolves in)
    full = "styxx." + name
    mod = types.ModuleType(full)
    mod.__package__ = "styxx"
    mod.__file__ = str(INSTRUMENT.parent / (name + ".py"))
    sys.modules[full] = mod
    try:
        exec(compile(source, mod.__file__, "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(full, None)
        raise
    return mod


@functools.lru_cache(maxsize=None)
def main_module():
    return module_from(main_source(), "_p2a_main_reference")


def main_port_path(tmp: Path) -> Path:
    """The reconstructed port, written beside nothing it could import by accident."""
    p = Path(tmp) / "diffgate_main_reference.js"
    p.write_bytes(main_port_source().encode("utf-8"))
    return p


# ---- inputs -------------------------------------------------------------------------------------------------------

def pinned_files() -> list[str]:
    m = re.search(r"for \(const name of \[(.*?)\]", lf(DIFFERENTIAL / "check_pairs.js"), re.S)
    assert m, "check_pairs.js no longer lists its corpora in a form this helper can read"
    return re.findall(r'"([^"]+)"', m.group(1))


@functools.lru_cache(maxsize=None)
def fuzz_corpus() -> tuple:
    """fuzz_corpus.py's 3,000 pairs, regenerated in memory (corpus_fuzz.json is gitignored), sha-pinned."""
    p = DIFFERENTIAL / "fuzz_corpus.py"
    src = lf(p)
    cut = src.index('\n(HERE / "corpus_fuzz.json")') + 1
    ns = {"__file__": str(p), "__name__": "_p2a_fuzz_corpus"}
    state = random.getstate()
    try:
        exec(compile(src[:cut], str(p), "exec"), ns)
    finally:
        random.setstate(state)
    items = ns["items"]
    assert sha(json.dumps(items, ensure_ascii=False)) == FUZZ_SHA, "fuzz_corpus.py no longer yields corpus_fuzz.json"
    return tuple(items)


@functools.lru_cache(maxsize=None)
def repro_cases() -> tuple:
    return tuple(json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"])


def inputs(fuzz: bool = True) -> list[tuple[str, str, str, str]]:
    """(set, id, summary, diff) for every committed input: the pinned pairs, the regenerated fuzz corpus, the PATH-2
    reproductions and #161's pinned-pair inputs, and the seeded PATH-2a fuzz."""
    out = []
    for name in pinned_files():
        p = DIFFERENTIAL / name
        if p.is_file():
            out += [(name, x["id"], x["summary"], x["diff"]) for x in json.loads(p.read_text(encoding="utf-8"))]
    if fuzz:
        out += [("corpus_fuzz", x["id"], x["summary"], x["diff"]) for x in fuzz_corpus()]
    out += [("path2a_repros", x["id"], x["summary"], x["diff"]) for x in repro_cases()]
    if fuzz:
        try:
            from ._p2a_cases import fuzz_pairs
        except ImportError:                       # imported as a top-level module by a scratch script
            from _p2a_cases import fuzz_pairs
        out += [("path2a_fuzz", x["id"], x["summary"], x["diff"]) for x in fuzz_pairs()]
    return out


def uid(i: int, row: tuple) -> str:
    """A unique id for input i (the sets reuse ids across files)."""
    return f"{row[0]}::{row[1]}::{i}"


# ---- the abstain-only relation (A) ----------------------------------------------------------------------------------

def reason(verdict: str, defect: str, phrase: str, main_why: str) -> str:
    return f"{verdict} withheld by PATH-2a ({defect}): {phrase}. main's reading: {main_why}"


def relation(a: dict, b: dict, strict: bool, phrases: dict, allow_error: bool = False) -> list[str]:
    """What (A) requires of the branch's record `b` against main's record `a` for the same input; [] when it holds."""
    bad = []
    for k in a:
        if k not in ("verdict", "claims") and a[k] != b.get(k):
            bad.append(f"field {k} moved")
    if set(a) != set(b):
        bad.append("record keys differ")
    if len(a["claims"]) != len(b["claims"]):
        return bad + ["claim count differs"]
    for i, (x, y) in enumerate(zip(a["claims"], b["claims"])):
        if set(x) != set(y) or (x["kind"], x["text"], x["detail"]) != (y["kind"], y["text"], y["detail"]):
            bad.append(f"claim {i}: kind, text or detail moved")
        if x["verdict"] == y["verdict"]:
            if x["why"] != y["why"]:
                bad.append(f"claim {i}: reason moved without a verdict move")
            continue
        if (x["kind"], x["verdict"]) not in REACH or y["verdict"] != "UNCHECKABLE":
            bad.append(f"claim {i}: {x['kind']} {x['verdict']} -> {y['verdict']} is not an abstention in reach")
            continue
        allowed = {reason(x["verdict"], d, p, x["why"]) for d in DEFECTS[x["kind"]] for p in phrases.values()}
        if y["why"] not in allowed:
            bad.append(f"claim {i}: reason is not the overlay's form")
        elif not allow_error and y["why"].startswith(f"{x['verdict']} withheld by PATH-2a (") and \
                y["why"].split("): ", 1)[1].startswith(phrases["error"] + "."):
            bad.append(f"claim {i}: the overlay's error fallback fired")
    cl = b["claims"]
    want = "FAIL" if (any(c["verdict"] == "CONTRADICTED" for c in cl)
                      or (strict and any(c["verdict"] == "UNCHECKABLE" for c in cl))) else "PASS"
    if b["verdict"] != want:
        bad.append("gate verdict is not main's formula over the claims")
    return bad


def phrase_key(why: str, phrases: dict) -> str | None:
    """The overlay's phrase key in a reason it wrote, else None."""
    m = re.match(r"(?:VERIFIED|CONTRADICTED) withheld by PATH-2a \((?:#97|#121|#97, #121|#101)\): ", why)
    if not m:
        return None
    rest = why[m.end():]
    for k, p in phrases.items():
        if rest.startswith(p + ". main's reading: "):
            return k
    return None
