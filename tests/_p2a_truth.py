"""PATH-2a truth oracle and attribution variants (not collected; NOTE_path2a_abstain_overlay_2026_09_30).

Truth is read from a case's file model, {"base": {path: content}, "head": {...}}, never from the diff. A content is a
str, a list of blocks, {"x": text} (executable), {"link": target}, {"gitlink": sha} or {"b64": data}. Every claim gets a
set of readings; the judgement is T when every reading says true, F when every reading says false, `?` otherwise, and
None when the claim is not judged. Paths compare as written (case kept, dots kept: `\\` to `/`, leading `./` and `/`
dropped); definitions come from CPython's `ast` of the running interpreter.

"Because of #97, #121 or #101" is defined by counterfactual variants of main, each built from the reconstructed main
with anchored replacements asserted to occur exactly once: V97 (find_path in tiers, base name only for a bare claim),
V121 (a path key that keeps its leading dots), V101 (a `def` the removed lines also define is not counted as added)
and all three together. A decided claim is attributable when main's verdict is false by truth and some variant's
verdict for the same claim is not false.
"""
from __future__ import annotations

import ast
import unicodedata

# ---- truth ------------------------------------------------------------------------------------------------------------


def _entry(value):
    if value is None:
        return None
    if isinstance(value, list):
        return ("file", "100644", "".join(value))
    if isinstance(value, str):
        return ("file", "100644", value)
    if "x" in value:
        return ("file", "100755", value["x"])
    if "b64" in value:
        return ("file", "100644", "b64:" + value["b64"])
    if "link" in value:
        return ("symlink", "120000", value["link"])
    return ("gitlink", "160000", value["gitlink"])


def changed(model: dict) -> dict:
    """{path: A | D | T | M} for every path whose entry differs between base and head."""
    base, head = model["base"], model["head"]
    out = {}
    for p in sorted(set(base) | set(head)):
        b, h = _entry(base.get(p)), _entry(head.get(p))
        if b != h:
            out[p] = "A" if b is None else "D" if h is None else "T" if b[0] != h[0] else "M"
    return out


def written(path: str) -> str:
    path = path.replace("\\", "/")
    while path.startswith(("./", "/")):
        path = path[2:] if path.startswith("./") else path[1:]
    return path


def _judge(values: set) -> str:
    return "T" if values == {True} else "F" if values == {False} else "?"


def path_truth(kind: str, claimed: str, status: dict) -> str:
    c = written(claimed)
    want = {"file_created": "A", "file_deleted": "D"}.get(kind)
    readings = [[p for p in status if p == c], [p for p in status if p == c or p.endswith("/" + c)]]
    if "/" not in c:
        readings.append([p for p in status if p.rsplit("/", 1)[-1] == c])
    return _judge({bool(h) and (want is None or any(status[p] == want for p in h)) for h in readings})


def only_truth(detail: dict, status: dict):
    prefs = [written(detail[k]).rstrip("/.") for k in ("prefix", "prefix2") if detail.get(k)]
    if not prefs or any(not x or x.startswith("..") for x in prefs):
        return None

    def bare_rule(p, x):
        if "/" not in x and "." in x:
            return p == x or p.endswith("/" + x)
        return p == x or p.startswith(x + "/")

    by_dir = all(any(p == x or p.startswith(x + "/") for x in prefs) for p in status)
    by_file = all(any(bare_rule(p, x) for x in prefs) for p in status)
    return _judge({by_dir, by_file})


def _source(value):
    e = _entry(value)
    return e[2] if e is not None and e[0] == "file" and not e[2].startswith("b64:") else ""


def _defs(text: str):
    """(tests, names) of one Python file, or None when CPython refuses it. tests: {(qualified name, is_async)} of
    functions named test_* at module or class level; names: every def, async def and class name, as a list."""
    try:
        tree = ast.parse(text.encode("utf-8", "surrogatepass"))
    except (SyntaxError, ValueError):
        return None
    tests, names = set(), []

    def walk(node, prefix, in_function):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(ch.name)
                if isinstance(ch, ast.ClassDef):
                    walk(ch, prefix + ch.name + ".", in_function)
                    continue
                if not in_function and ch.name.startswith("test_"):
                    tests.add((prefix + ch.name, isinstance(ch, ast.AsyncFunctionDef)))
                walk(ch, prefix + ch.name + ".", True)
            else:
                walk(ch, prefix, in_function)
    walk(tree, "", False)
    return tests, names


def _py(model, status):
    """{path: (base defs, head defs)} for every .py path in the model, or None when a changed one does not parse."""
    out = {}
    for p in sorted(set(model["base"]) | set(model["head"])):
        if not p.endswith(".py"):
            continue
        b, h = _defs(_source(model["base"].get(p))), _defs(_source(model["head"].get(p)))
        if b is None or h is None:
            if p in status:
                return None
            b, h = b or (set(), []), h or (set(), [])
        out[p] = (b, h)
    return out


def tests_truth(detail: dict, model: dict, status: dict):
    py = _py(model, status)
    if py is None:
        return None
    n = int(detail["n"])
    per_file_sync = sum(len({t for t in h[0] if not t[1]} - {t for t in b[0] if not t[1]})
                        for p, (b, h) in py.items() if p in status)
    per_file_all = sum(len({t[0] for t in h[0]} - {t[0] for t in b[0]}) for p, (b, h) in py.items() if p in status)
    glob = len({t for _p, (_b, h) in py.items() for t in h[0] if not t[1]}
               - {t for _p, (b, _h) in py.items() for t in b[0] if not t[1]})
    verdict = _judge({x == n for x in (per_file_sync, per_file_all, glob)})
    noun = (detail.get("noun") or "").lower()
    if noun and noun not in ("function", "functions", "method", "methods") and verdict != "T":
        return "?"
    return verdict


def symbol_truth(detail: dict, model: dict, status: dict):
    py = _py(model, status)
    if py is None:
        return None
    name = unicodedata.normalize("NFKC", detail["name"])
    per_set = any(name in h[1] and name not in b[1] for p, (b, h) in py.items() if p in status)
    per_count = any(h[1].count(name) > b[1].count(name) for p, (b, h) in py.items() if p in status)
    glob = any(name in h[1] for _b, h in py.values()) and not any(name in b[1] for b, _h in py.values())
    return _judge({per_set, per_count, glob})


def truth(model: dict, kind: str, detail: dict):
    status = changed(model)
    if kind in ("file_created", "file_deleted", "file_touched"):
        return path_truth(kind, detail["path"], status)
    if kind == "files_changed_count":
        return _judge({len(status) == int(detail["n"])})
    if kind == "only_touches":
        return only_truth(detail, status)
    if kind == "tests_added":
        return tests_truth(detail, model, status)
    if kind == "symbol_added":
        return symbol_truth(detail, model, status)
    return None


def wrong(verdict: str, t) -> bool:
    return (verdict == "VERIFIED" and t == "F") or (verdict == "CONTRADICTED" and t == "T")


def right(verdict: str, t) -> bool:
    return (verdict == "VERIFIED" and t == "T") or (verdict == "CONTRADICTED" and t == "F")


# ---- the counterfactual variants of main ------------------------------------------------------------------------------

_FIND_PATH = """        for p, st in status.items():
            if p == c or p.endswith("/" + c) or Path(p).name == Path(c).name:
                return p, st
        return None, None
"""
_FIND_PATH_V97 = """        for _t in (0, 1, 2):
            if _t == 2 and "/" in c:
                break
            for p, st in status.items():
                if (p == c) if _t == 0 else p.endswith("/" + c) if _t == 1 else Path(p).name == Path(c).name:
                    return p, st
        return None, None
"""
_NORM = '    return p.replace("\\\\", "/").lstrip("./").lower()\n'
_NORM_V121 = '    return re.sub(r"^(?:\\.?/)+", "", p.replace("\\\\", "/")).lower()\n'
_TESTS = '                        got = len(re.findall(r"^\\s*def test_", added_blob, re.M))\n'
_TESTS_V101 = ('                        _rem = {x for _a, _r in (sides or {}).values() for _l in _r\n'
               '                                for x in re.findall(r"\\s*(?:async\\s+)?def\\s+(test_\\w*)", _l)}\n'
               '                        got = len([x for x in re.findall(r"^\\s*def (test_\\w*)", added_blob, re.M)\n'
               '                                   if x not in _rem])\n')
_SYMBOL = '                        hit = bool(re.search(pat, added_blob, re.M))\n'
_SYMBOL_V101 = (_SYMBOL + '                        hit = hit and not any(\n'
                '                            re.search(r"^\\s*(?:async\\s+)?(?:def|class)\\s+" + re.escape(d["name"])\n'
                '                                      + r"\\b", _l) for _a, _r in (sides or {}).values() for _l in _r)\n')

VARIANTS = {
    "v97": [(_FIND_PATH, _FIND_PATH_V97)],
    "v121": [(_NORM, _NORM_V121)],
    "v101": [(_TESTS, _TESTS_V101), (_SYMBOL, _SYMBOL_V101)],
}
VARIANTS["vall"] = VARIANTS["v97"] + VARIANTS["v121"] + VARIANTS["v101"]


def variant_source(main_text: str, name: str) -> str:
    out = main_text
    for old, new in VARIANTS[name]:
        assert out.count(old) == 1, f"{name}: the anchor is not in main exactly once:\n{old}"
        out = out.replace(old, new)
    return out


# ---- the same four variants of main's port ----------------------------------------------------------------------------
# NOTE_path2a_second_pass_2026_09_30 (B-2): the port's own false verdicts are judged in the port's own terms, by the same
# counterfactuals built from the reconstructed port (tests/_p2a_ref.main_port_source) with anchored edits.

_JS_FIND_PATH = """    for (const [p, st] of status) {
      if (p === c || p.endsWith("/" + c) || _basename(p) === _basename(c)) return [p, st];
    }
    return [null, null];
"""
_JS_FIND_PATH_V97 = """    for (const _t of [0, 1, 2]) {
      if (_t === 2 && c.includes("/")) break;
      for (const [p, st] of status) {
        if (_t === 0 ? p === c : _t === 1 ? p.endsWith("/" + c) : _basename(p) === _basename(c)) return [p, st];
      }
    }
    return [null, null];
"""
_JS_NORM = """  let s = p.replace(/BSBS/g, "/");
  let i = 0;
  while (i < s.length && (s[i] === "." || s[i] === "/")) i++;   // str.lstrip("./")
  return s.slice(i).toLowerCase();
""".replace("BS", chr(92))
_JS_NORM_V121 = """  return p.replace(/BSBS/g, "/").replace(/^(?:BS.?BS/)+/, "").toLowerCase();
""".replace("BS", chr(92))
_JS_TESTS = "            const got = (addedBlob.match(/^BSs*def test_/gm) || []).length;\n".replace("BS", chr(92))
_JS_TESTS_V101 = ("            const _rem = new Set();\n"
                  "            for (const [, _r] of sides.values()) for (const _l of _r) "
                  "for (const _m of _l.matchAll(/BSs*(?:asyncBSs+)?defBSs+(test_BSw*)/g)) _rem.add(_m[1]);\n"
                  "            const got = [...addedBlob.matchAll(/^BSs*def (test_BSw*)/gm)]"
                  ".filter(_m => !_rem.has(_m[1])).length;\n").replace("BS", chr(92))
_JS_SYMBOL = "            const hit = pat.test(addedBlob);\n"
_JS_SYMBOL_V101 = ("            const hit = pat.test(addedBlob) && ![...sides.values()].some(([, _r]) => _r.some(_l => "
                   "new RegExp(\"^BSBSs*(?:asyncBSBSs+)?(?:def|class)BSBSs+\" + _reEscape(d.name) + \"BSBSb\")"
                   ".test(_l)));\n").replace("BS", chr(92))

PORT_VARIANTS = {
    "v97": [(_JS_FIND_PATH, _JS_FIND_PATH_V97)],
    "v121": [(_JS_NORM, _JS_NORM_V121)],
    "v101": [(_JS_TESTS, _JS_TESTS_V101), (_JS_SYMBOL, _JS_SYMBOL_V101)],
}
PORT_VARIANTS["vall"] = PORT_VARIANTS["v97"] + PORT_VARIANTS["v121"] + PORT_VARIANTS["v101"]


def port_variant_source(main_port_text: str, name: str) -> str:
    out = main_port_text
    for old, new in PORT_VARIANTS[name]:
        assert out.count(old) == 1, f"port {name}: the anchor is not in main's port exactly once:\n{old}"
        out = out.replace(old, new)
    return out
