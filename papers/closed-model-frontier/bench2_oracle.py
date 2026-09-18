"""BENCH-2 oracle: BENCH-1's three rules, with the two admissibility tests its audit forced.

Prereg: PREREG_bench2_pr_claim_benchmark_2026_09_17.md (sha256 5a796aa7...).  Like BENCH-1's, this
file deliberately does NOT import styxx.diffgate, and is written to be read in full.

BENCH-1 failed its own blocking audit gate: 30 of 50 hand-audited items disagreed.  The cause was
that it accepted whatever token the extractor pulled out of a sentence and treated it as a fact
about code.  "this only touches the internal evaluation logic" yielded the prefix `the`; "Added
method returning the file descriptor" yielded the symbol `returning`.  Both were then scored
CONTRADICTED.

The three ground-truth rules are unchanged:

  files changed = the number of distinct `diff --git` header lines
  paths changed = the `b/` path of each header, or the `a/` path where the file is deleted
  added lines   = lines beginning `+` that are not `+++`

What is new sits in FRONT of them: a claim whose named thing is not the kind of thing the rule can
check is UNDECIDABLE -- counted, published, never scored as either outcome.
"""
from __future__ import annotations

import re

_HEADER = re.compile(r'^diff --git (?:"a/(?P<qa>(?:[^"\\]|\\.)*)"|a/(?P<a>.*?)) (?:"b/(?P<qb>(?:[^"\\]|\\.)*)"|b/(?P<b>.*))$')


def _unquote(s: str) -> str:
    return s.replace('\\"', '"').replace("\\\\", "\\")


def headers(diff: str) -> list[dict]:
    out: list[dict] = []
    cur: dict | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = _HEADER.match(line)
            if not m:
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


def _norm(p: str) -> str:
    return _norm_keep(p).rstrip("/")


def _norm_keep(p: str) -> str:
    """As _norm, but keeps a trailing slash.

    The prereg's clause (a) is "p contains /".  A stated prefix of "src/" contains one; stripping
    it first would demote that claim to a bare name and lose clause (a) entirely.  So clause (a) is
    tested against this form, before the trailing slash is normalised away.
    """
    return p.replace("\\", "/").strip().strip("'\"`").lstrip("./").lower()


# ---- admissibility --------------------------------------------------------------------------
# Prereg PATH-SHAPED, verbatim: p is admissible iff (a) it contains "/", or (b) it is a filename
# with an extension, or (c) it is a >=2-char token that occurs as a complete slash-delimited
# segment of at least one path in the diff.

_FILENAME = re.compile(r"^[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8}$")
_BARE = re.compile(r"^[A-Za-z0-9_.-]{2,}$")


def path_shaped(prefix: str, paths: list[str], raw: str = "") -> tuple[bool, str]:
    if "/" in (raw or prefix):
        return True, "a"
    if _FILENAME.match(prefix):
        return True, "b"
    if _BARE.match(prefix):
        for p in paths:
            if prefix in p.split("/"):
                return True, "c"
        return False, "not_a_segment_of_any_changed_path"
    return False, "not_token_shaped"


def inside(path: str, prefix: str, clause: str) -> bool:
    """Prereg containment: strict for (a)/(b); segment membership for a bare name admitted by (c)."""
    if clause in ("a", "b"):
        return path == prefix or path.startswith(prefix + "/")
    return prefix in path.split("/")


# Prereg CODE-SHAPED, verbatim: backticked in the sentence, or followed by "(", or contains "_"
# or a digit, or carries an internal uppercase letter.

def code_shaped(name: str, claim_text: str) -> tuple[bool, str]:
    if f"`{name}`" in claim_text:
        return True, "backticked"
    if re.search(re.escape(name) + r"\s*\(", claim_text):
        return True, "called"
    if "_" in name or any(c.isdigit() for c in name):
        return True, "underscore_or_digit"
    if any(c.isupper() for c in name[1:]):
        return True, "internal_uppercase"
    return False, "bare_lowercase_word"


# ---- labels ---------------------------------------------------------------------------------

def label_files_changed_count(claim_detail: dict, diff: str, claim_text: str = ""):
    """Carried forward from BENCH-1 unchanged -- it passed its audit."""
    raw = str(claim_detail.get("n", "")).replace(",", "").strip()
    if not raw.isdigit():
        return "UNDECIDABLE", {"reason": "claim carries no integer"}
    stated, actual = int(raw), files_changed(diff)
    return ("SUPPORTED" if stated == actual else "CONTRADICTED"), {"stated": stated, "actual": actual}


def label_only_touches(claim_detail: dict, diff: str, claim_text: str = ""):
    kept = [_norm_keep(claim_detail.get(k, "")) for k in ("prefix", "prefix2")]
    kept = [p for p in kept if p.strip("/")]
    raw = [p.rstrip("/") for p in kept]
    if not raw:
        return "UNDECIDABLE", {"reason": "claim carries no prefix"}
    paths = [_norm(p) for p in paths_changed(diff)]
    if not paths:
        return "UNDECIDABLE", {"reason": "diff has no parsable paths"}

    admitted, rejected = [], []
    for p, k in zip(raw, kept):
        ok, why = path_shaped(p, paths, raw=k)
        (admitted if ok else rejected).append((p, why))
    if not admitted:
        return "UNDECIDABLE", {"reason": "no stated prefix is path-shaped",
                               "prefixes": raw, "rejected": rejected}

    outside = [p for p in paths if not any(inside(p, pre, cl) for pre, cl in admitted)]
    return ("CONTRADICTED" if outside else "SUPPORTED"), {
        "prefixes_admitted": admitted, "prefixes_rejected": rejected,
        "paths": len(paths), "outside": outside[:5], "n_outside": len(outside)}


def label_symbol_added(claim_detail: dict, diff: str, claim_text: str = ""):
    name = (claim_detail.get("symbol") or claim_detail.get("name") or "").strip()
    if not name or not re.fullmatch(r"[A-Za-z_]\w*", name):
        return "UNDECIDABLE", {"reason": "claim carries no plain symbol name"}
    ok, why = code_shaped(name, claim_text or "")
    if not ok:
        return "UNDECIDABLE", {"reason": "symbol is not code-shaped", "symbol": name, "test": why}
    blob = "\n".join(added_lines(diff))
    pat = re.compile(
        r"(?:^|\s)(?:def|class|function|fn|func|interface|type|enum|struct|trait|const|let|var)\s+"
        + re.escape(name) + r"\b"
        r"|(?:^|\s)" + re.escape(name) + r"\s*(?:=\s*(?:function|async|\(|\[)|\()",
        re.M)
    return ("SUPPORTED" if pat.search(blob) else "CONTRADICTED"), {
        "symbol": name, "admitted_by": why, "added_lines": len(added_lines(diff))}


LABELLERS = {
    "files_changed_count": label_files_changed_count,
    "only_touches": label_only_touches,
    "symbol_added": label_symbol_added,
    # tests_added: excluded by the prereg.  No labeller exists on purpose.
}


def label(kind: str, claim_detail: dict, diff: str, claim_text: str = ""):
    fn = LABELLERS.get(kind)
    if fn is None:
        return "EXCLUDED", {"reason": f"{kind} is not scored by this benchmark"}
    return fn(claim_detail or {}, diff, claim_text)
