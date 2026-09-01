"""What is left after V13? The residual false-accusation classes, on DEVELOPMENT only.

V13 removed 34.6% of path accusations against a 66.7% bar and failed. The other
two thirds were never characterised. This reads the surviving accusations with
the repairs ON, groups them by the gate's own reason plus the shape of the
sentence that produced them, and reports the classes — so the next repair, if
there is one, is designed against evidence rather than intuition.

DEVELOPMENT bucket only (SPLIT_external_corpus_2026_08_31.md). No held-out
prose is read here; that set decides the successor and must stay unseen.

  python papers/closed-model-frontier/v14_residual_analysis.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import styxx.diffgate as DG                                   # noqa: E402
from external1_harness import reconstruct                     # noqa: E402

DB = HERE / "external1_shelf.sqlite"
OUT = HERE / "v14_residual.json"
SAMPLE_PER_CLASS = 4


def bucket(repo_url: str) -> int:
    n = (repo_url or "").strip().rstrip("/").lower()
    return int(hashlib.sha256(n.encode("utf-8")).hexdigest()[:8], 16) % 10


# ── candidate residual shapes, named before counting ───────────────────────
CODE_FENCE = re.compile(r"```|~~~")
URL_LIKE = re.compile(r"https?://|www\.|\]\(")
CHECKBOX = re.compile(r"^\s*[-*]?\s*\[[ xX]\]")
QUOTED = re.compile(r"^\s*>")
FUTURE = re.compile(r"\b(will|should|would|todo|next|plan|intend|going to)\b", re.I)
CONDITIONAL = re.compile(r"\b(if|when|unless|in case|optional|consider)\b", re.I)
GENERIC_NAME = re.compile(r"^(index|main|test|setup|config|utils?|types?|app|api)\."
                          r"(js|ts|py|json|md|yml|yaml)$", re.I)
PLURAL_GLOB = re.compile(r"\*|\.\.\.|\betc\b", re.I)


def shape(sentence: str, path: str) -> str:
    """One label per accusation, first match wins. Named before the counts."""
    if CODE_FENCE.search(sentence):
        return "inside-or-near-a-code-fence"
    if URL_LIKE.search(sentence):
        return "path-inside-a-url-or-markdown-link"
    if QUOTED.match(sentence):
        return "quoted-block (reporting someone else's text)"
    if CHECKBOX.match(sentence):
        return "checklist-item (template boilerplate, not a claim)"
    if FUTURE.search(sentence):
        return "future-or-intended, not asserted-done"
    if CONDITIONAL.search(sentence):
        return "conditional or optional"
    if PLURAL_GLOB.search(sentence):
        return "glob/ellipsis (names a set, not a file)"
    if GENERIC_NAME.match((path or "").rsplit("/", 1)[-1]):
        return "generic basename (index.ts, main.py, ...)"
    return "unclassified"


def main() -> int:
    con = sqlite3.connect(DB)
    DG.WITHHOLD_PATH_ACCUSATION = False        # measure the branch as it would accuse

    classes = Counter()
    by_reason = Counter()
    samples = defaultdict(list)
    n_pr = n_acc = 0

    for pid, title, body, url in con.execute(
            "SELECT id, title, body, html_url FROM pr "
            "WHERE body IS NOT NULL AND body != ''"):
        if bucket("/".join((url or "").split("/")[:5])) >= 3:
            continue                            # HELD-OUT — not read
        rows = con.execute("SELECT filename, status, patch FROM f WHERE pr_id=?",
                           (pid,)).fetchall()
        if not rows:
            continue
        diff, implied = reconstruct(rows)
        parsed, _ = DG.parse_unified_diff(diff)
        if parsed != implied:
            continue
        n_pr += 1
        summary = f"{title or ''}\n\n{body}"
        g = DG.gate_diff_text(summary, diff, run=None, strict=False)
        for c in g.claims:
            if c.verdict != "CONTRADICTED" or not c.kind.startswith("file_"):
                continue
            n_acc += 1
            path = (c.detail or {}).get("path", "")
            lab = shape(c.text, path)
            classes[lab] += 1
            by_reason["status-mismatch" if "status" in (c.why or "")
                      else "absent-from-diff"] += 1
            if len(samples[lab]) < SAMPLE_PER_CLASS:
                samples[lab].append({"url": url, "kind": c.kind, "path": path,
                                     "text": c.text[:150], "why": (c.why or "")[:110]})

    con.close()
    tot = n_acc or 1
    payload = {
        "split": "DEVELOPMENT only (held-out unread)",
        "prs_scored": n_pr,
        "path_accusations_after_v13": n_acc,
        "by_gate_reason": dict(by_reason),
        "residual_classes": {k: {"n": v, "share": round(v / tot, 4)}
                             for k, v in classes.most_common()},
        "samples": {k: v for k, v in samples.items()},
        "note": ("classes named before counting; 'unclassified' is the honest "
                 "remainder and is not evidence of anything"),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    print(f"DEVELOPMENT PRs scored: {n_pr}")
    print(f"path accusations surviving V13: {n_acc}")
    print(f"by gate reason: {dict(by_reason)}\n")
    print("residual classes (named before counting):")
    for k, v in classes.most_common():
        print(f"  {v:5d}  {v/tot:6.2%}  {k}")
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
