"""BENCH-1 oracle: ground truth from the live diff, in three rules, independent of the instrument.

Prereg: PREREG_bench1_pr_claim_benchmark_2026_09_17.md.  This file deliberately does NOT import
styxx.diffgate.  It is written to be read in full by anyone checking the benchmark, and it is the
only thing that assigns a ground-truth label.

The three rules, verbatim from the prereg:

  files changed = the number of distinct `diff --git` header lines
  paths changed = the `b/` path of each header, or the `a/` path where the file is deleted
  added lines   = lines beginning `+` that are not `+++`

Nothing else is derived.  `tests_added` gets no label here: deciding what counts as a test is a
judgement and the prereg excludes it from scoring.
"""
from __future__ import annotations

import re

# `diff --git a/X b/Y`, with git's quoting for paths that need it.
_HEADER = re.compile(r'^diff --git (?:"a/(?P<qa>(?:[^"\\]|\\.)*)"|a/(?P<a>.*?)) (?:"b/(?P<qb>(?:[^"\\]|\\.)*)"|b/(?P<b>.*))$')


def _unquote(s: str) -> str:
    """git C-quotes paths containing specials; undo the escapes we can see."""
    return s.replace('\\"', '"').replace("\\\\", "\\")


def headers(diff: str) -> list[dict]:
    """Every `diff --git` header, with its a/ and b/ paths and whether the file was deleted."""
    out: list[dict] = []
    cur: dict | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = _HEADER.match(line)
            if not m:                      # a path we cannot parse is recorded, never guessed
                cur = {"a": "", "b": "", "deleted": False, "unparsed": line[:200]}
            else:
                a = _unquote(m.group("qa")) if m.group("qa") is not None else (m.group("a") or "")
                b = _unquote(m.group("qb")) if m.group("qb") is not None else (m.group("b") or "")
                cur = {"a": a, "b": b, "deleted": False}
            out.append(cur)
        elif cur is not None and line.startswith("deleted file mode"):
            cur["deleted"] = True
    return out


def files_changed(diff: str) -> int:
    return len(headers(diff))


def paths_changed(diff: str) -> list[str]:
    return [(h["a"] if h["deleted"] else h["b"]) for h in headers(diff)]


def added_lines(diff: str) -> list[str]:
    return [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]


# ---- labels ---------------------------------------------------------------------------------
# Each returns (label, facts).  label is "CONTRADICTED", "SUPPORTED" or "UNDECIDABLE";
# UNDECIDABLE means the claim's own text does not give the oracle something to check, and such
# items are reported, never scored as either.

def _norm_prefix(p: str) -> str:
    return p.replace("\\", "/").strip().strip("'\"`").lstrip("./").rstrip("/").lower()


def label_files_changed_count(claim_detail: dict, diff: str):
    raw = str(claim_detail.get("n", "")).replace(",", "").strip()
    if not raw.isdigit():
        return "UNDECIDABLE", {"reason": "claim carries no integer"}
    stated, actual = int(raw), files_changed(diff)
    return ("SUPPORTED" if stated == actual else "CONTRADICTED"), {"stated": stated, "actual": actual}


def label_only_touches(claim_detail: dict, diff: str):
    prefixes = [_norm_prefix(claim_detail.get(k, "")) for k in ("prefix", "prefix2")]
    prefixes = [p for p in prefixes if p]
    if not prefixes:
        return "UNDECIDABLE", {"reason": "claim carries no prefix"}
    paths = [_norm_prefix(p) for p in paths_changed(diff)]
    if not paths:
        return "UNDECIDABLE", {"reason": "diff has no parsable paths"}
    outside = [p for p in paths
               if not any(p == pre or p.startswith(pre + "/") for pre in prefixes)]
    return ("CONTRADICTED" if outside else "SUPPORTED"), {"prefixes": prefixes,
                                                          "paths": len(paths),
                                                          "outside": outside[:5],
                                                          "n_outside": len(outside)}


def label_symbol_added(claim_detail: dict, diff: str):
    name = (claim_detail.get("symbol") or claim_detail.get("name") or "").strip()
    if not name or not re.fullmatch(r"[A-Za-z_]\w*", name):
        return "UNDECIDABLE", {"reason": "claim carries no plain symbol name"}
    blob = "\n".join(added_lines(diff))
    # a definition of `name` in any of the corpus's common languages
    pat = re.compile(
        r"(?:^|\s)(?:def|class|function|fn|func|interface|type|enum|struct|trait|const|let|var)\s+"
        + re.escape(name) + r"\b"
        r"|(?:^|\s)" + re.escape(name) + r"\s*(?:=\s*(?:function|async|\(|\[)|\()",
        re.M)
    return ("SUPPORTED" if pat.search(blob) else "CONTRADICTED"), {"symbol": name,
                                                                   "added_lines": len(added_lines(diff))}


LABELLERS = {
    "files_changed_count": label_files_changed_count,
    "only_touches": label_only_touches,
    "symbol_added": label_symbol_added,
    # tests_added: excluded by the prereg.  No labeller exists on purpose.
}


def label(kind: str, claim_detail: dict, diff: str):
    fn = LABELLERS.get(kind)
    if fn is None:
        return "EXCLUDED", {"reason": f"{kind} is not scored by this benchmark"}
    return fn(claim_detail or {}, diff)
