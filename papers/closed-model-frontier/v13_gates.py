"""V13 gates G-R1 (subset invariant) and G-R2 (development recovery).

G-R3 (held-out precision) runs as a separate fresh blind panel; only it licenses
re-enabling the accusation. Prereg: PREREG_v13_repair_2026_08_31.md (amended).

  python papers/closed-model-frontier/v13_gates.py
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import styxx.diffgate as DG                                    # noqa: E402
from external1_harness import reconstruct                      # noqa: E402

DB = HERE / "external1_shelf.sqlite"
OUT = HERE / "v13_gates.json"


def bucket(repo_url: str) -> int:
    n = (repo_url or "").strip().rstrip("/").lower()
    return int(hashlib.sha256(n.encode("utf-8")).hexdigest()[:8], 16) % 10


def accusations(summary: str, diff: str) -> set:
    """Accusation identity: (kind, the path/prefix it names)."""
    g = DG.gate_diff_text(summary, diff, run=None, strict=False)
    out = set()
    for c in g.claims:
        if c.verdict == "CONTRADICTED":
            d = c.detail or {}
            out.add((c.kind, d.get("path") or d.get("prefix") or d.get("name") or d.get("n")))
    return out


def main() -> int:
    con = sqlite3.connect(DB)
    before_n = after_n = 0
    grew = []
    dev_before = dev_after = 0
    kinds_before, kinds_after = Counter(), Counter()
    seen = dev_seen = 0

    for pid, agent, title, body, url in con.execute(
            "SELECT id, agent, title, body, html_url FROM pr "
            "WHERE body IS NOT NULL AND body != ''"):
        files = con.execute(
            "SELECT filename, status, patch FROM f WHERE pr_id=?", (pid,)).fetchall()
        if not files:
            continue
        diff, implied = reconstruct(files)
        parsed, _ = DG.parse_unified_diff(diff)
        if parsed != implied:
            continue
        seen += 1
        summary = f"{title or ''}\n\n{body}"

        DG.WITHHOLD_PATH_ACCUSATION = False
        before = _with_v13(summary, diff, off=True)
        after = _with_v13(summary, diff, off=False)

        before_n += len(before)
        after_n += len(after)
        for k in before:
            kinds_before[k[0]] += 1
        for k in after:
            kinds_after[k[0]] += 1
        new = after - before
        if new:
            grew.append({"pr": pid, "url": url, "new": sorted(map(str, new))[:3]})

        # DEVELOPMENT bucket only, per the frozen split
        # the shelf keeps html_url, not repo_url; the repo is its first five parts
        repo = "/".join((url or "").split("/")[:5])
        if bucket(repo) < 3:
            dev_seen += 1
            dev_before += len(before)
            dev_after += len(after)

    con.close()
    recovered = (dev_before - dev_after) / dev_before if dev_before else None
    payload = {
        "prereg": "PREREG_v13_repair_2026_08_31.md (amended)",
        "prs_scored": seen,
        "accusations_before": before_n,
        "accusations_after": after_n,
        "removed": before_n - after_n,
        "by_kind_before": dict(kinds_before),
        "by_kind_after": dict(kinds_after),
        "G_R1_subset_invariant": {"prs_gaining_an_accusation": len(grew),
                                  "pass": len(grew) == 0,
                                  "examples": grew[:5]},
        "development_bucket": {"prs": dev_seen, "before": dev_before,
                               "after": dev_after,
                               "fraction_removed": None if recovered is None
                               else round(recovered, 4)},
        "G_R2_recovery": {"threshold": 0.6667,
                          "observed": None if recovered is None else round(recovered, 4),
                          "pass": bool(recovered is not None and recovered >= 0.6667)},
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("prs_scored", "accusations_before", "accusations_after",
                       "removed", "G_R1_subset_invariant", "development_bucket",
                       "G_R2_recovery")}, indent=1)[:1600])
    return 0


def _with_v13(summary: str, diff: str, off: bool) -> set:
    """Accusations with the V13 repairs disabled (off=True) or enabled."""
    saved = (DG._NON_FILE_NOUNS, DG._REFERENTIAL, DG._CONTAINMENT)
    if off:
        import re as _re
        DG._NON_FILE_NOUNS = frozenset()
        DG._REFERENTIAL = tuple(k for k in DG._REFERENTIAL if k not in _V13_CUES)
        DG._CONTAINMENT = _re.compile(r"(?!x)x")      # matches nothing
    try:
        return accusations(summary, diff)
    finally:
        DG._NON_FILE_NOUNS, DG._REFERENTIAL, DG._CONTAINMENT = saved


_V13_CUES = frozenset({
    "avoid", "avoids", "without modif", "without chang", "without touch",
    "without altering", "no need to", "does not modify", "does not change",
    "does not touch", "doesn't modify", "doesn't change", "doesn't touch",
    "not modified", "not changed", "not touched", "no changes to",
    "unchanged", "untouched", "preserves",
})


if __name__ == "__main__":
    sys.exit(main())
