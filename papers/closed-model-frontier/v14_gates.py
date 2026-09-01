"""V14 gates G-S1 (subset invariant, path-keyed) and G-S2 (cumulative recovery).

Prereg: PREREG_v14_repair_2026_08_31.md. G-S2 measures V13 AND V14 together
against the same two-thirds bar V13 failed alone. G-S3 (held-out precision)
is a separate blind panel and is the only thing that re-enables the accusation.

  python papers/closed-model-frontier/v14_gates.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import styxx.diffgate as DG                                    # noqa: E402
from external1_harness import reconstruct                      # noqa: E402

DB = HERE / "external1_shelf.sqlite"
OUT = HERE / "v14_gates.json"

_V13_CUES = frozenset({
    "avoid", "avoids", "without modif", "without chang", "without touch",
    "without altering", "no need to", "does not modify", "does not change",
    "does not touch", "doesn't modify", "doesn't change", "doesn't touch",
    "not modified", "not changed", "not touched", "no changes to",
    "unchanged", "untouched", "preserves",
})


def bucket(repo_url: str) -> int:
    n = (repo_url or "").strip().rstrip("/").lower()
    return int(hashlib.sha256(n.encode("utf-8")).hexdigest()[:8], 16) % 10


def accused_paths(summary: str, diff: str, repairs: bool) -> set:
    """Paths the gate accuses, with all repairs on or all off.

    Identity is the PATH, never (kind, path): the repairs deliberately change
    the kind, and keying on it mis-reports a demotion as a new accusation —
    the defect that made V13's first G-R1 measurement wrong.
    """
    saved = (DG._NON_FILE_NOUNS, DG._REFERENTIAL, DG._CONTAINMENT,
             DG.V14_CONTAINMENT_TOUCH, DG.V14_BARE_NAME_ABSTAIN)
    if not repairs:
        DG._NON_FILE_NOUNS = frozenset()
        DG._REFERENTIAL = tuple(k for k in DG._REFERENTIAL if k not in _V13_CUES)
        DG._CONTAINMENT = re.compile(r"(?!x)x")        # matches nothing
        DG.V14_CONTAINMENT_TOUCH = False
        DG.V14_BARE_NAME_ABSTAIN = False
    try:
        g = DG.gate_diff_text(summary, diff, run=None, strict=False)
        return {(c.detail or {}).get("path") for c in g.claims
                if c.verdict == "CONTRADICTED" and c.kind.startswith("file_")}
    finally:
        (DG._NON_FILE_NOUNS, DG._REFERENTIAL, DG._CONTAINMENT,
         DG.V14_CONTAINMENT_TOUCH, DG.V14_BARE_NAME_ABSTAIN) = saved


def main() -> int:
    DG.WITHHOLD_PATH_ACCUSATION = False       # measure the branch as it would accuse
    con = sqlite3.connect(DB)

    grew, n_pr, dev_pr = [], 0, 0
    dev_off = dev_on = all_off = all_on = 0

    for pid, title, body, url in con.execute(
            "SELECT id, title, body, html_url FROM pr "
            "WHERE body IS NOT NULL AND body != ''"):
        rows = con.execute("SELECT filename, status, patch FROM f WHERE pr_id=?",
                           (pid,)).fetchall()
        if not rows:
            continue
        diff, implied = reconstruct(rows)
        parsed, _ = DG.parse_unified_diff(diff)
        if parsed != implied:
            continue
        n_pr += 1
        s = f"{title or ''}\n\n{body}"
        off = accused_paths(s, diff, repairs=False)
        on = accused_paths(s, diff, repairs=True)
        all_off += len(off)
        all_on += len(on)
        if on - off:
            grew.append({"pr": pid, "url": url,
                         "new": sorted(str(x) for x in (on - off))[:3]})
        if bucket("/".join((url or "").split("/")[:5])) < 3:
            dev_pr += 1
            dev_off += len(off)
            dev_on += len(on)

    con.close()
    rec = (dev_off - dev_on) / dev_off if dev_off else None
    payload = {
        "prereg": "PREREG_v14_repair_2026_08_31.md",
        "identity": "accusation keyed on PATH (kind is deliberately changed by the repairs)",
        "prs_scored": n_pr,
        "corpus_wide": {"accusations_unrepaired": all_off,
                        "accusations_after_v13_v14": all_on,
                        "removed": all_off - all_on},
        "G_S1_subset_invariant": {"paths_gaining_an_accusation": len(grew),
                                  "pass": len(grew) == 0, "examples": grew[:5]},
        "development_bucket": {"prs": dev_pr, "unrepaired": dev_off,
                               "after_v13_v14": dev_on,
                               "fraction_removed": None if rec is None else round(rec, 4)},
        "G_S2_cumulative_recovery": {
            "threshold": 0.6667,
            "observed": None if rec is None else round(rec, 4),
            "pass": bool(rec is not None and rec >= 0.6667),
            "note": "V13 alone reached 0.3462 against this same bar and failed",
        },
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print(f"PRs scored: {n_pr}")
    print(f"corpus-wide path accusations: {all_off} -> {all_on} "
          f"(removed {all_off - all_on})")
    print(f"G-S1 subset invariant: paths gaining an accusation = {len(grew)}  "
          f"{'PASS' if not grew else 'FAIL'}")
    print(f"development: {dev_off} -> {dev_on}   removed {rec:.2%}" if rec is not None
          else "development: no accusations")
    print(f"G-S2 cumulative recovery: {rec:.4f} vs 0.6667 -> "
          f"{'PASS' if payload['G_S2_cumulative_recovery']['pass'] else 'FAIL'}")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
