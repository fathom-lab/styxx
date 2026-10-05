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
    reproductions and #161's pinned-pair inputs, the seeded PATH-2a fuzz, and the seeded text-seam set (claims whose
    text main's two ports build differently, NOTE_path2a_third_pass_2026_09_30)."""
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
            from ._p2a_cases import fuzz_pairs, text_seam_pairs
        except ImportError:                       # imported as a top-level module by a scratch script
            from _p2a_cases import fuzz_pairs, text_seam_pairs
        out += [("path2a_fuzz", x["id"], x["summary"], x["diff"]) for x in fuzz_pairs()]
        out += [("path2a_seam", x["id"], x["summary"], x["diff"]) for x in text_seam_pairs()]
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
        elif not allow_error and y["why"].startswith(f"{x['verdict']} withheld by PATH-2a ("):
            said = y["why"].split("): ", 1)[1]
            for k in ("error", "malformed"):      # APPLY's two fallbacks: DECIDE raised, or returned no list
                if k in phrases and said.startswith(phrases[k] + "."):
                    bad.append(f"claim {i}: the overlay's {k} fallback fired")
    cl = b["claims"]
    want = "FAIL" if (any(c["verdict"] == "CONTRADICTED" for c in cl)
                      or (strict and any(c["verdict"] == "UNCHECKABLE" for c in cl))) else "PASS"
    if b["verdict"] != want:
        bad.append("gate verdict is not main's formula over the claims")
    return bad


def strict_alike(off: dict, on: dict) -> list[str]:
    """--strict may move the gate verdict and nothing else: the same input's claims and every other field read the
    same with it as without it (NOTE_path2a_second_pass_2026_09_30, a plant that skipped the overlay under --strict
    passed every check of pass 1)."""
    return [f"under --strict, {k} differs" for k in off if k != "verdict" and off[k] != on.get(k)]


def fake_git(name_status: str, diff: str):
    """A stand-in for a module's `_git` that answers gate_diff's two calls from a case's own recorded output."""
    def _git(repo, *args):
        if args[:2] == ("diff", "--name-status"):
            return name_status
        if args[:1] == ("diff",):
            return diff
        raise AssertionError(f"unexpected git call {args!r}")
    return _git


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


# ---- the cross-port bar (C), as NOTE_path2a_ninth_pass_2026_10_04 restates it --------------------------------------

# phrases for a reading the overlay failed to reproduce or make (`malformed`: NOTE_path2a_tenth_pass_2026_10_05)
FALLBACKS = ("unreproduced", "unparsed", "error", "malformed")


def final(c: dict, phrases: dict) -> tuple:
    """A claim as bar C compares two final lists: kind, verdict and detail, and, where the overlay wrote the reason,
    its phrase key and defect tag. Not the text and not main's own reason: main's two ports cut and strip a claim's
    text differently and print some reasons differently, and the overlay copies main's reason verbatim."""
    key = phrase_key(c["why"], phrases)
    tag = c["why"].split(" withheld by PATH-2a (", 1)[1].split("): ", 1)[0] if key else None
    return c["kind"], c["verdict"], json.dumps(c["detail"], sort_keys=True), key, tag


def how_lists_differ(a: list, x: list):
    """None where main's two claim lists are the same (one length; the same kind, verdict and detail at each
    position). Else the side that parts them: "description" (another length, or a kind or a detail apart: the two
    ports read different claims from the description) or "diff" (the same kinds and details, a verdict apart: the two
    ports decide one claim apart on the diff)."""
    def kd(c):
        return c["kind"], json.dumps(c["detail"], sort_keys=True)
    if len(a) != len(x) or any(kd(c) != kd(u) for c, u in zip(a, x)):
        return "description"
    return None if all(c["verdict"] == u["verdict"] for c, u in zip(a, x)) else "diff"


def bar_c(rows: list, phrases: dict) -> tuple:
    """Bar C over rows {id, a, b, ja, jb, strict}: main's and the branch's Python records, the same in the port, and
    under --strict the four gate verdicts in that order; a row with "raises" is one where a main raises, and is only
    counted. Returns (counts, broken, ids).
    C(ii), `broken`: an input where main's two lists are the same and the two final lists, or the two gate verdicts in
    either strict mode, are not; and any claim either port withheld with a fallback phrase.
    C(iii), `counts`: the inputs where main's lists differ, by side, and on them how often the two gate verdicts differ
    under main and under the overlay, in each strict mode; `ids` names the inputs where main's gates agree and the
    overlay's do not."""
    import collections
    c = collections.Counter({k: 0 for k in ("inputs", "a main raises", "lists equal", "lists equal, a claim withheld",
                                            "lists differ, description side", "lists differ, diff side")})
    for mode in ("no --strict", "--strict"):
        for k in ("gates differ under main", "gates differ under the overlay", "main agrees, the overlay differs",
                  "main differs, the overlay agrees"):
            c[mode + ": " + k] = 0
    broken, ids = [], {"no --strict": [], "--strict": []}
    for r in rows:
        c["inputs"] += 1
        if r.get("raises"):
            c["a main raises"] += 1
            continue
        A, B, X, Y = (r[k]["claims"] for k in ("a", "b", "ja", "jb"))
        for port, news in (("python", B), ("port", Y)):
            if any(phrase_key(y["why"], phrases) in FALLBACKS for y in news):
                broken.append((r["id"], port + ": a fallback phrase"))
        gates = {"no --strict": (r["a"]["verdict"], r["ja"]["verdict"], r["b"]["verdict"], r["jb"]["verdict"]),
                 "--strict": tuple(r["strict"])}
        how = how_lists_differ(A, X)
        if how is None:
            c["lists equal"] += 1
            c["lists equal, a claim withheld"] += any(phrase_key(y["why"], phrases) for y in B)
            if [final(y, phrases) for y in B] != [final(y, phrases) for y in Y]:
                broken.append((r["id"], "the final lists differ where main's two lists are the same"))
            for mode, (ma, mj, na, nj) in gates.items():
                if ma != mj or na != nj:
                    broken.append((r["id"], mode + ": the gate verdicts differ where main's two lists are the same"))
            continue
        c["lists differ, " + how + " side"] += 1
        for mode, (ma, mj, na, nj) in gates.items():
            c[mode + ": gates differ under main"] += ma != mj
            c[mode + ": gates differ under the overlay"] += na != nj
            c[mode + ": main differs, the overlay agrees"] += ma != mj and na == nj
            if ma == mj and na != nj:
                c[mode + ": main agrees, the overlay differs"] += 1
                ids[mode].append(r["id"])
    return dict(c), broken, ids


def bar_c_rows(main_mod, new_mod, items: list, js: dict) -> list:
    """The rows `bar_c` reads, for items {id, summary, diff} and the port's --decisions output keyed by id."""
    rows = []
    for it in items:
        j = js[it["id"]]
        try:
            a = main_mod.gate_diff_text(it["summary"], it["diff"]).to_dict()
            sa = main_mod.gate_diff_text(it["summary"], it["diff"], strict=True).verdict
        except Exception:
            a = None
        if a is None or "error" in j["main"]:
            rows.append({"id": it["id"], "raises": True})
            continue
        b = new_mod.gate_diff_text(it["summary"], it["diff"]).to_dict()
        sb = new_mod.gate_diff_text(it["summary"], it["diff"], strict=True).verdict
        rows.append({"id": it["id"], "a": a, "b": b, "ja": j["main"], "jb": j["new"],
                     "strict": (sa, j["strict"]["main"], sb, j["strict"]["new"])})
    return rows
