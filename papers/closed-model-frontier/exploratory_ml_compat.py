"""EXPLORATORY (no verdicts, no prereg): two questions asked of the EXTERNAL-1 corpus before any
rule is frozen.

  1. Test counts outside Python.  For every PR whose description says "added N tests" (the
     tests_added template, counted noun captured), which languages does the diff touch, and how
     many test definitions per language do the ADDED lines carry, by a per-language pattern?
     How often does the claimed N equal the count in the dominant test language?  (A proxy for
     "would VERIFIED", not a precision measurement: agreement is not truth.)
  2. Compatibility claims.  How many descriptions say "no breaking changes", "backward
     compatible", "zero/no behavior change", "non-breaking"?  Of those, how many diffs REMOVE a
     public top-level name (per language) that is not re-defined anywhere in the added lines?

Counts only; no PR named.  Reads the EXTERNAL-2 shelf.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import styxx.diffgate as dg  # noqa: E402  (the wheel; only parse_unified_diff and _norm are used)
from external1_harness import reconstruct  # noqa: E402

DB = HERE / "external2_shelf.sqlite"
OUT = HERE / "exploratory_ml_compat.json"

TESTS_ADDED = re.compile(r"\b(?:add\w+|creat\w+)\s+(?P<n>\d+)\s+(?:new\s+)?tests?\b(?:\s+(?P<noun>cases?|files?|scenarios?|suites?|class(?:es)?|functions?|methods?)\b)?", re.I)
COMPAT = re.compile(r"\b(?:no\s+breaking\s+changes?|non[- ]breaking|backwards?[- ]compatib(?:le|ility)|"
                    r"(?:zero|no)\s+(?:behaviou?r(?:al)?|functional)\s+changes?|fully\s+compatible|"
                    r"does\s+not\s+(?:break|change)\s+(?:any\s+)?(?:existing\s+)?(?:behaviou?r|api|public\s+api))\b", re.I)

# per-language test-definition patterns on ADDED lines (line-anchored, indentation allowed)
TEST_PATTERNS = {
    "python": (re.compile(r"^\s*(?:async\s+)?def test_\w+", re.M), (".py",)),
    "js/ts": (re.compile(r"^\s*(?:it|test)(?:\.(?:each|only|skip|concurrent))?\s*\(\s*(?:[`'\"]|\[)", re.M),
              (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")),
    "go": (re.compile(r"^\s*func\s+Test\w*\s*\(", re.M), (".go",)),
    "rust": (re.compile(r"^\s*#\[(?:tokio::|async_std::)?test", re.M), (".rs",)),
    "java/kotlin": (re.compile(r"^\s*@(?:Test|ParameterizedTest|RepeatedTest)\b", re.M), (".java", ".kt", ".kts")),
    "c#": (re.compile(r"^\s*\[(?:Fact|Theory|Test|TestMethod|TestCase)\b", re.M), (".cs",)),
    "ruby": (re.compile(r"^\s*(?:it|test|specify)\s+['\"]", re.M), (".rb",)),
    "php": (re.compile(r"^\s*(?:public\s+)?function\s+test\w+|^\s*#\[Test\]|^\s*/\*\*\s*@test", re.M), (".php",)),
    "swift": (re.compile(r"^\s*func\s+test\w+\s*\(", re.M), (".swift",)),
    "dart": (re.compile(r"^\s*(?:test|testWidgets)\s*\(\s*['\"]", re.M), (".dart",)),
}

# per-language PUBLIC top-level definitions (for compat claims): (removed-line pattern, name group)
PUBLIC_DEFS = {
    "python": (re.compile(r"^(?:def|class)\s+(?P<name>[A-Za-z]\w*)", re.M), (".py",)),
    "js/ts": (re.compile(r"^export\s+(?:default\s+)?(?:async\s+)?(?:function\*?|class|const|let|var|interface|type|enum)\s+(?P<name>[A-Za-z_$]\w*)", re.M),
              (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")),
    "go": (re.compile(r"^func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Z]\w*)\s*\(|^type\s+(?P<name2>[A-Z]\w*)\b", re.M), (".go",)),
    "rust": (re.compile(r"^\s*pub\s+(?:async\s+)?(?:fn|struct|enum|trait|type|const|static)\s+(?P<name>[A-Za-z_]\w*)", re.M), (".rs",)),
    "java/kotlin": (re.compile(r"^\s*public\s+(?:static\s+|final\s+|abstract\s+)*(?:[\w<>\[\],\s]+?)\s+(?P<name>[a-zA-Z_]\w*)\s*\(", re.M), (".java", ".kt")),
}


def side(diff: str, sign: str, exts: tuple) -> dict:
    """Lines of one sign, bucketed by file suffix group, walking file headers."""
    out: dict = defaultdict(list)
    cur = None
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:]
            if p not in ("/dev/null",):
                cur = Path(p[2:] if p[1:2] == "/" else p).suffix.lower()
            continue
        if line.startswith("diff --git"):
            cur = None
            continue
        if line.startswith(sign) and not line.startswith(sign * 3):
            out[cur].append(line[1:])
    return out


def main() -> int:
    con = sqlite3.connect(DB)
    n_seen = n_elig = 0
    ta = Counter(); ta_lang = Counter(); ta_agree = Counter(); ta_python_only_agree = Counter()
    compat = Counter(); compat_lang = Counter(); removed_hist = Counter()
    compat_examples_shape = Counter()
    for pid, agent, title, body in con.execute("SELECT id, agent, title, body FROM pr"):
        n_seen += 1
        if not body or not body.strip():
            continue
        files = con.execute("SELECT filename, status, patch FROM f WHERE pr_id=?", (pid,)).fetchall()
        if not files:
            continue
        diff, implied = reconstruct(files)
        parsed, _ = dg.parse_unified_diff(diff)
        if parsed != implied:
            continue
        n_elig += 1
        text = f"{title or ''}\n\n{body}"
        exts = {Path(fn).suffix.lower() for fn, _s, _p in files if fn}
        has_py = bool(exts & {".py", ".pyi"})
        # ---- 1. test counts by language
        m_all = list(TESTS_ADDED.finditer(text))
        if m_all:
            added = None
            for m in m_all:
                n = int(m.group("n")); noun = (m.group("noun") or "").lower()
                if noun in ("cases", "case", "files", "file", "scenarios", "suites", "classes", "class"):
                    ta["claims_counting_non_functions"] += 1
                    continue
                ta["claims"] += 1
                if added is None:
                    added = side(diff, "+", ())
                counts = {}
                for lang, (rx, sufs) in TEST_PATTERNS.items():
                    blob = "\n".join(l for s, ls in added.items() if s in sufs for l in ls)
                    if blob:
                        counts[lang] = len(rx.findall(blob))
                langs_present = [l for l, (_r, sufs) in TEST_PATTERNS.items() if exts & set(sufs)]
                key = "python" if has_py else ("+".join(sorted(langs_present)) if langs_present else "other/none")
                ta_lang[key] += 1
                if counts:
                    dom = max(counts, key=lambda k: counts[k])
                    if counts[dom] == n:
                        ta_agree[key] += 1
                    if not has_py and counts[dom] == n:
                        ta_agree["non_python_agree_total"] += 1
                if not has_py:
                    ta["claims_non_python"] += 1
        # ---- 2. compatibility claims
        cm = COMPAT.search(text)
        if cm:
            compat["claims"] += 1
            compat_examples_shape[cm.group(0).lower()[:40]] += 1
            removed = side(diff, "-", ()); added2 = side(diff, "+", ())
            dropped = []
            for lang, (rx, sufs) in PUBLIC_DEFS.items():
                rblob = "\n".join(l for s, ls in removed.items() if s in sufs for l in ls)
                ablob = "\n".join(l for s, ls in added2.items() if s in sufs for l in ls)
                if not rblob:
                    continue
                for mm in rx.finditer(rblob):
                    name = mm.group("name") or mm.groupdict().get("name2")
                    if not name or name.startswith("_"):
                        continue
                    # re-defined anywhere in added lines (same language)?  then it is a change/move, not a removal
                    if re.search(r"\b" + re.escape(name) + r"\b", ablob):
                        continue
                    dropped.append((lang, name))
            langs_present = sorted({l for l, (_r, sufs) in PUBLIC_DEFS.items() if exts & set(sufs)})
            compat_lang["+".join(langs_present) or "other/none"] += 1
            if dropped:
                compat["claims_with_a_dropped_public_name"] += 1
                removed_hist[min(len(dropped), 10)] += 1
                for lang, _n in {(l, None) for l, _ in dropped}:
                    compat[f"dropped_in_{lang}"] += 1
        if n_seen % 10000 == 0:
            print(f"  {n_seen} seen, {n_elig} eligible; tests_added claims {ta['claims']}; compat claims {compat['claims']}", flush=True)
    payload = {"prs_seen": n_seen, "eligible": n_elig,
               "tests_added": dict(ta), "tests_added_by_language": dict(ta_lang),
               "tests_added_claimed_equals_counted_by_language": dict(ta_agree),
               "compat": dict(compat), "compat_by_language": dict(compat_lang),
               "compat_dropped_name_histogram": dict(removed_hist),
               "compat_phrase_shapes": compat_examples_shape.most_common(12)}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
