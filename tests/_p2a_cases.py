"""PATH-2a generated inputs (not collected; NOTE_path2a_abstain_overlay_2026_09_30).

`families(seed, n)` yields truth-judged cases -- a base/head file model, a diff rendered from it in one of several
styles, and a summary of claims -- for the three defects: #97 (same base names in several directories), #121 (dotfile
twins, dotted directories and prefixes, with and without case outside ASCII) and #101 (tests and functions changed,
renamed, moved, made async, added beside changed ones). `fuzz_pairs()` is the seeded fuzz the relation (A) and the
cross-port check (C) run on: the same families plus renderings no generator of real diffs writes -- characters the
two ports split or strip differently in headers and in content, lone CRs, `def` beside odd separators, drive-like and
astral names, headers with no hunks.

Everything is seeded, and uses only `random.Random` methods whose results are stable across CPython 3.9 to 3.14.
"""
from __future__ import annotations

import difflib
import functools
import random

TS = "2024-05-06 07:08:09.000000000 +0000"
EPOCH = "1970-01-01 00:00:00.000000000 +0000"

# ---- rendering ----------------------------------------------------------------------------------------------------


def _text(c):
    if c is None:
        return None
    if isinstance(c, list):
        return "".join(c)
    if isinstance(c, dict):
        return c.get("x", "\x00binary:" + str(c.get("b64", c.get("link", c.get("gitlink", "")))))
    return c


def _binary(t):
    return t is not None and "\x00" in t


def _lines(t):
    if not t:
        return []
    out = t.split("\n")
    res = [x + "\n" for x in out[:-1]]
    if out[-1]:
        res.append(out[-1])
    return res


def _hunks(a, b, ctx):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for group in sm.get_grouped_opcodes(ctx):
        i1, i2, j1, j2 = group[0][1], group[-1][2], group[0][3], group[-1][4]
        body = []
        for tag, a1, a2, b1, b2 in group:
            if tag == "equal":
                body += [" " + x for x in a[a1:a2]]
                continue
            if tag in ("replace", "delete"):
                body += ["-" + x for x in a[a1:a2]]
            if tag in ("replace", "insert"):
                body += ["+" + x for x in b[b1:b2]]
        al, bl = i2 - i1, j2 - j1
        out.append((i1 + 1 if al else i1, al, j1 + 1 if bl else j1, bl, body))
    return out


STYLES = ("git", "git", "git", "noprefix", "mnemonic", "plain", "gnu", "index")


def render(base: dict, head: dict, style: dict, rng: random.Random) -> str:
    """A unified diff of base -> head. style: hdr (git | noprefix | mnemonic | plain | gnu | index), order (sorted |
    shuffled), ctx (0..3), crlf, devnull (plain | ts | epoch, non-git styles)."""
    hdr, ctx = style.get("hdr", "git"), style.get("ctx", 3)
    ps = sorted(set(base) | set(head))
    if style.get("order") == "shuffled":
        rng.shuffle(ps)
    out = []
    for p in ps:
        a, b = _text(base.get(p)), _text(head.get(p))
        if a == b:
            continue
        created, deleted = a is None, b is None
        la, lb = {"git": ("a/", "b/"), "noprefix": ("", ""), "mnemonic": ("c/", "w/"), "gnu": ("a/", "b/")}.get(
            hdr, ("", ""))
        if hdr in ("git", "noprefix", "mnemonic"):
            out.append(f"diff --git {la}{p} {lb}{p}")
            if created:
                out.append("new file mode 100644")
            elif deleted:
                out.append("deleted file mode 100644")
            out.append("index 1111111..2222222" + ("" if created or deleted else " 100644"))
            if _binary(a) or _binary(b):
                out.append(f"Binary files {'/dev/null' if created else la + p} and "
                           f"{'/dev/null' if deleted else lb + p} differ")
                continue
            out.append("--- " + ("/dev/null" if created else la + p))
            out.append("+++ " + ("/dev/null" if deleted else lb + p))
        else:
            if _binary(a) or _binary(b):
                out.append(f"Binary files {'/dev/null' if created else la + p} and "
                           f"{'/dev/null' if deleted else lb + p} differ")
                continue
            if hdr == "index":
                out += [f"Index: {p}", "=" * 67]
            if hdr == "gnu":
                out.append(f"diff -{'N' if created or deleted else ''}u a/{p} b/{p}")
            ts = "\t" + TS if hdr == "gnu" else ""
            dv = style.get("devnull", "plain")
            missing = {"plain": "/dev/null", "ts": "/dev/null\t" + EPOCH}.get(dv)
            out.append("--- " + ((missing or la + p + "\t" + EPOCH) if created else la + p + ts))
            out.append("+++ " + ((missing or lb + p + "\t" + EPOCH) if deleted else lb + p + ts))
        for a1, al, b1, bl, body in _hunks(_lines(a), _lines(b), ctx):
            out.append(f"@@ -{a1},{al} +{b1},{bl} @@")
            for line in body:
                if line.endswith("\n"):
                    out.append(line[:-1])
                else:
                    out += [line, "\\ No newline at end of file"]
    eol = "\r\n" if style.get("crlf") else "\n"
    return eol.join(out) + (eol if out else "")


# ---- families -------------------------------------------------------------------------------------------------------

P97 = ["README.md", "docs/README.md", "integrations/git/README.md", "git/README.md", "src/utils.py", "lib/utils.py",
       "utils.py", "a/b/c.py", "b/c.py", "c.py", "src/node/glob.ts", "glob.ts", "pkg/a/__init__.py",
       "pkg/__init__.py", "x/src/utils.py", "b/utils.py", "a/README.md"]
P121 = [".pr_agent.toml", "pr_agent.toml", ".github/workflows/ci.yml", "github/workflows/ci.yml", ".env.json",
        "env.json", "cfg/x.json", ".cfg/x.json", "..cfg/x.json", "src/app.py", "src/.hidden.py", "src/hidden.py",
        ".eslintrc.json", "eslintrc.json", "docs/.nojekyll.txt", "docs/nojekyll.txt", "logo.png", ".logo.png"]
TWINS = [(".pr_agent.toml", "pr_agent.toml"), (".github/workflows/ci.yml", "github/workflows/ci.yml"),
         (".env.json", "env.json"), (".cfg/x.json", "cfg/x.json"), ("..cfg/x.json", "cfg/x.json"),
         (".eslintrc.json", "eslintrc.json"), (".logo.png", "logo.png"), ("src/.hidden.py", "src/hidden.py")]
PREF121 = [".github/", "github/", ".github", "github", "src", "src/", ".env.json", "env.json", "cfg", ".cfg", "docs",
           "./src", ".pr_agent.toml", "pr_agent.toml", "workflows/ci.yml"]
E, U, K = chr(0xE9), chr(0xFC), chr(0x212A)
P121U = ["." + "caf" + E + ".toml", "caf" + E + ".toml", "." + E + "nv.json", E + "nv.json", ".k.json", "k.json",
         K + ".json", "docs/" + E + "t" + E + ".md", ".docs/x.md", "docs/x.md", "src/" + U + ".py", ".src/" + U + ".py",
         chr(0xC9) + "/x.md", E + "/x.md"]


def _txt(rng, k=2):
    return "".join(f"v{rng.randint(0, 999)} = {rng.randint(0, 9)}\n" for _ in range(k))


def _put(rng, base, head, p, st, binary=False):
    c0 = {"b64": "AAE" + str(rng.randint(0, 99))} if binary else _txt(rng)
    c1 = {"b64": "AAF" + str(rng.randint(0, 99))} if binary else c0 + _txt(rng, 1)
    if st == "A":
        head[p] = c0
    elif st == "D":
        base[p] = c0
    else:
        base[p], head[p] = c0, c1


def _changed(base, head):
    return [p for p in sorted(set(base) | set(head)) if _text(base.get(p)) != _text(head.get(p))]


def _path_claims(rng, pool, n_changed):
    s = [f"{d} files changed." for d in sorted({n_changed - 1, n_changed, n_changed + 1})
         if d >= 0 and rng.random() < 0.7]
    forms = ["Created {}.", "Deleted {}.", "Modified {}.", "- {} — new.", "Updated `{}`.", "Removes {}.",
             "Edited ./{}.", "Creating file {}.", "Deleted the file {}."]
    for p in rng.sample(pool, min(len(pool), rng.randint(3, 7))):
        s.append(rng.choice(forms).format(p))
    return s


def fam97(rng):
    base, head = {}, {}
    for p in rng.sample(P97, rng.randint(2, 6)):
        _put(rng, base, head, p, rng.choice("AMD"))
    return base, head, _path_claims(rng, P97, len(_changed(base, head)))


def fam121(rng, pool=None):
    pool = pool or P121
    base, head = {}, {}
    ps = rng.sample(pool, rng.randint(2, 6))
    if pool is P121 and rng.random() < 0.7:
        for p in rng.choice(TWINS):
            if p not in ps:
                ps.append(p)
    for p in ps:
        _put(rng, base, head, p, rng.choice("AMD"), binary=p.endswith(".png") and rng.random() < 0.7)
    s = _path_claims(rng, pool, len(_changed(base, head)))
    prefs = PREF121 if pool is P121 else ["docs", ".docs", "src", ".src", ".k.json", "k.json", "docs/"]
    for _ in range(rng.randint(1, 3)):
        a = rng.choice(prefs)
        s.append(f"Only touches {a} and {rng.choice(prefs)}." if rng.random() < 0.25 else f"Only touches {a}.")
    return base, head, s


def fam121u(rng):
    return fam121(rng, P121U)


TESTS = ["test_a", "test_b", "test_c", "test_d"]


def _pyfile(tests, funcs=(), classes=(), klass=(), asyncs=(), sigs=None, extra=""):
    sigs = sigs or {}
    out = ["import os\n", "\n"]
    for f in funcs:
        out.append(f"def {f}({sigs.get(f, 'n')}):\n    return n\n\n")
    for c in classes:
        out.append(f"class {c}({sigs.get(c, 'object')}):\n    pass\n\n")
    for t in tests:
        out.append(f"{'async def' if t in asyncs else 'def'} {t}({sigs.get(t, '')}):\n    assert os\n\n")
    if klass:
        out.append("class TestK:\n")
        for t in klass:
            out.append(f"    def {t}(self{sigs.get(t, '')}):\n        assert os\n\n")
    return "".join(out) + extra


def fam101(rng, hazards=False):
    files = ["tests/test_x.py", "tests/test_y.py", "src/lib.py"]
    b_tests = {f: rng.sample(TESTS, rng.randint(0, 3)) for f in files[:2]}
    h_tests = {f: list(v) for f, v in b_tests.items()}
    b_funcs = rng.sample(["backoff", "retry_once", "helper"], rng.randint(0, 2))
    b_classes = rng.sample(["Retry", "Helper"], rng.randint(0, 1))
    h_funcs, h_classes = list(b_funcs), list(b_classes)
    b_kt, h_kt = {f: [] for f in files[:2]}, {f: [] for f in files[:2]}
    b_sigs, h_sigs, h_async = {}, {}, set()
    extra = {f: "" for f in files}
    ops = ["add", "sig", "rename", "move", "class", "async", "fsig", "fadd", "csig", "cadd", "dup", "del", "string"]
    for op in rng.sample(ops, rng.randint(1, 4)):
        f = rng.choice(files[:2])
        g = files[1] if f == files[0] else files[0]
        if op == "add":
            t = rng.choice([x for x in TESTS + ["test_e", "test_f"] if x not in h_tests[f]] or ["test_g"])
            h_tests[f].append(t)
        elif op == "sig" and h_tests[f]:
            h_sigs[rng.choice(h_tests[f])] = "tmp_path"
        elif op == "rename" and h_tests[f]:
            t = rng.choice(h_tests[f])
            h_tests[f][h_tests[f].index(t)] = t + "2"
        elif op == "move" and h_tests[f]:
            t = rng.choice(h_tests[f])
            h_tests[f].remove(t)
            if t not in h_tests[g]:
                h_tests[g].append(t)
        elif op == "class" and h_tests[f]:
            t = rng.choice(h_tests[f])
            h_tests[f].remove(t)
            h_kt[f].append(t)
        elif op == "async" and h_tests[f]:
            h_async.add(rng.choice(h_tests[f]))
        elif op == "fsig" and h_funcs:
            h_sigs[rng.choice(h_funcs)] = "n, jitter=0"
        elif op == "fadd":
            nf = rng.choice(["backoff", "retry_once", "helper", "fresh"])
            if nf not in h_funcs:
                h_funcs.append(nf)
        elif op == "csig" and h_classes:
            h_sigs[rng.choice(h_classes)] = "Base"
        elif op == "cadd":
            nc = rng.choice(["Retry", "Helper", "Fresh"])
            if nc not in h_classes:
                h_classes.append(nc)
        elif op == "dup" and h_tests[f]:
            h_tests[f].append(rng.choice(h_tests[f]))
        elif op == "del" and h_tests[f]:
            h_tests[f].remove(rng.choice(h_tests[f]))
        elif op == "string":
            extra[f] = rng.choice(['DOC = """\ndef test_a():\n"""\n', "# def test_b(): moved\n",
                                   'X = "def backoff(n):"\n'])
    base, head = {}, {}
    for f in files[:2]:
        bt = _pyfile(b_tests[f], klass=b_kt[f], sigs=b_sigs)
        ht = _pyfile(h_tests[f], klass=h_kt[f], asyncs=h_async, sigs=h_sigs, extra=extra[f])
        if b_tests[f] or b_kt[f] or rng.random() < 0.5:
            base[f] = bt
        if h_tests[f] or h_kt[f] or f in base:
            head[f] = ht
    base[files[2]] = _pyfile([], b_funcs, b_classes, sigs=b_sigs)
    head[files[2]] = _pyfile([], h_funcs, h_classes, sigs=h_sigs)
    if hazards:
        sep = rng.choice([chr(0xA0), "\x0c", "\x0b", "\ufeff", "  ", "\t", chr(0x3000)])
        for side in (base, head):
            for f in list(side):
                if rng.random() < 0.4:
                    side[f] = side[f].replace("def ", "def" + sep, rng.randint(1, 2))
                if rng.random() < 0.2:
                    side[f] = side[f].replace("\ndef ", "\n" + rng.choice(["\ufeff", "\x0c", " "]) + "def ", 1)
    s = [f"Added {n} tests." for n in range(0, 5) if rng.random() < 0.5]
    for nm in rng.sample(["backoff", "retry_once", "helper", "fresh", "Retry", "Helper", "Fresh", "test_a", "test_e"], 3):
        s.append(rng.choice([f"Adds function {nm}.", f"Added class {nm}.", f"Introduces the method {nm}."]))
    return base, head, s


FAMILIES = {"97": fam97, "121": fam121, "121u": fam121u, "101": fam101}


def _style(rng):
    return {"hdr": rng.choice(STYLES), "order": rng.choice(["sorted", "sorted", "shuffled"]),
            "ctx": rng.choice([3, 3, 1, 0]), "crlf": rng.random() < 0.1,
            "devnull": rng.choice(["plain", "plain", "ts", "epoch"])}


def families(seed: int, n: int) -> list[dict]:
    """n cases per family, truth-judged by tests/_p2a_truth.py through their models."""
    rng = random.Random(seed)
    out = []
    for fam, make in FAMILIES.items():
        for i in range(n):
            base, head, sents = make(rng)
            style = _style(rng)
            diff = render(base, head, style, rng)
            rng.shuffle(sents)
            summary = rng.choice([" ", "\n"]).join(sents)
            if rng.random() < 0.15:
                pool = P97 + P121
                decl = [f"file_touched: {rng.choice(pool)}", f"file_created: {rng.choice(pool)}",
                        f"file_deleted: {rng.choice(pool)}", f"files_changed: {rng.randint(1, 6)}",
                        f"tests_added: {rng.randint(0, 3)}", f"adds_symbol: {rng.choice(['backoff', 'Retry', 'fresh'])}",
                        f"only_touches: {rng.choice(PREF121)}"]
                summary += "\n\n```styxx\n" + "\n".join(rng.sample(decl, 3)) + "\n```\n"
            out.append({"id": f"p2a-{fam}:{seed}:{i}", "family": fam, "style": style, "summary": summary,
                        "diff": diff, "model": {"base": base, "head": head}})
    return out


# ---- the fuzz for (A) and (C) ----------------------------------------------------------------------------------------

HAZARDS = ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85", "\u2028", "\u2029", "\ufeff", "\xa0", "\u3000",
           "\r", "\t", " ", "\u0130", "\u212a", "\U0001F600", "\u03a3", ":", ".", "/", "\\"]
ODD_PATHS = ["c:x.py", "C:/src/x.py", "\U0001F600:/x.py", "./x.py", "/x.py", ".//x.py", "../x.py", "..", ".",
             "a/.", "src/./x.py", "\u0130.py", "i\u0307.py", "\u212aey.py", "key.py", "\u03a3\u03a3.md", "\u03c3\u03c2.md",
             "\U0001F600/x.py", "x/\U0001F600.py", "b/a/x.py", "a/x.py", "x.py\t2024-01-01", "sp ace.py", "tab\t.py",
             "\"q.py\"", "dir/"]


def _mutate(rng, diff: str) -> str:
    lines = diff.split("\n")
    for _ in range(rng.randint(1, 3)):
        if not lines:
            break
        i = rng.randrange(len(lines))
        h = rng.choice(HAZARDS)
        k = rng.randint(0, len(lines[i]))
        how = rng.random()
        if how < 0.5:
            lines[i] = lines[i][:k] + h + lines[i][k:]
        elif how < 0.7:
            lines[i] = lines[i] + h
        elif how < 0.85:
            lines[i] = h + lines[i]
        else:
            lines.insert(i, rng.choice(["diff --git a/z.py b/z.py", "new file mode 100644", "deleted file mode 100644",
                                        "rename from x.py", "rename to .x.py", "Binary files /dev/null and b/.z differ",
                                        "+def test_z():", "-def test_z(x):", "--- a/y.py", "+++ b/.y.py"]))
    return "\n".join(lines)


def _odd_diff(rng) -> str:
    out = []
    for p in rng.sample(ODD_PATHS, rng.randint(1, 4)):
        st = rng.choice("AMDB")
        if st == "A":
            out.append(f"diff --git a/{p} b/{p}\nnew file mode 100644\n--- /dev/null\n+++ b/{p}\n@@ -0,0 +1 @@\n+x\n")
        elif st == "D":
            out.append(f"diff --git a/{p} b/{p}\ndeleted file mode 100644\n--- a/{p}\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n")
        elif st == "M":
            out.append(f"--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-def test_a():\n+def test_a(x):\n")
        else:
            out.append(f"diff --git a/{p} b/{p}\nnew file mode 100644\nBinary files /dev/null and b/{p} differ\n")
    return "".join(out)


def _odd_summary(rng) -> str:
    s = []
    for _ in range(rng.randint(1, 5)):
        p = rng.choice(ODD_PATHS + P97 + P121).rstrip("/")
        s.append(rng.choice([f"Created {p}.", f"Deleted {p}.", f"Modified {p}.", f"{rng.randint(0, 5)} files changed.",
                             f"Only touches {rng.choice(PREF121 + ['x', '.', '..cfg/', 'C:/src'])}.",
                             f"Added {rng.randint(0, 3)} tests.", "Adds function test_a.", "Added class Retry."]))
    return " ".join(s)


@functools.lru_cache(maxsize=None)
def fuzz_pairs(seed: int = 20260930, n: int = 2000) -> tuple:
    """The seeded PATH-2a fuzz: family renderings, some mutated with hazard characters, and odd-path diffs."""
    rng = random.Random(seed)
    fams = list(FAMILIES.items()) + [("101h", lambda r: fam101(r, hazards=True))]
    out = []
    for i in range(n):
        r = rng.random()
        if r < 0.15:
            diff, summary = _odd_diff(rng), _odd_summary(rng)
        else:
            _fam, make = rng.choice(fams)
            base, head, sents = make(rng)
            diff = render(base, head, _style(rng), rng)
            rng.shuffle(sents)
            summary = " ".join(sents)
            if r < 0.55:
                diff = _mutate(rng, diff)
        if rng.random() < 0.05:
            diff = diff.replace("\n", "\r")
        out.append({"id": f"p2a-fuzz:{seed}:{i}", "summary": summary, "diff": diff})
    return tuple(out)
