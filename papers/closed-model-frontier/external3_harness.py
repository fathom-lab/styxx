"""EXTERNAL-3 harness: the BC-1 checkout over the EXTERNAL-1 corpus, for the preregistered gates.

Corpus: HuggingFace ``hao-li/AIDev`` (CC-BY-4.0; Zenodo 10.5281/zenodo.16919272), the same two
tables EXTERNAL-1 streamed, read here from the published parquet files (``pull_request.parquet``,
``pr_commit_details.parquet``) with pyarrow.  Reconstruction: EXTERNAL-1's own ``reconstruct`` and
``_fold_statuses``, imported unchanged from ``external1_harness.py`` in this directory, so every
PR is gated against exactly the diff the published measurement used.  Eligibility and exclusions:
EXTERNAL-1's (empty body, no file records, reconstruction mismatch — counted, never scored).
Instrument: THIS CHECKOUT's ``styxx.diffgate`` (the BC-1 repair, PREREG_bc1_by_construction_2026_09_16),
not the wheel; the import order below pins it, and the run refuses to start if ``styxx.diffgate``
resolves to an installed package instead.  Everything else is EXTERNAL-2's harness verbatim, so the
two ledgers differ only by the instrument.

The ledger has EXTERNAL-2's shape (kind, verdict, detail, reason per claim; the diff's file
extensions; ``def test_`` names on both sides).  ``external2_census.py --ledger external3`` counts
it; ``external3_gates.py`` compares it with EXTERNAL-2's ledger for G-B1 and G-B3.  The ledger names
third-party PRs and is gitignored; only the aggregate receipts are committed.

    python external3_harness.py shelf --corpus DIR   # or reuse external2_shelf.sqlite
    python external3_harness.py gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

sys.path.insert(0, str(ROOT))  # the checkout FIRST: this run measures the repair, not the wheel
import styxx  # noqa: E402
import styxx.diffgate as dg  # noqa: E402
if not Path(dg.__file__).resolve().is_relative_to(ROOT):
    sys.exit("external3: styxx.diffgate resolved to an installed package, not this checkout")
BC1 = bool(getattr(dg, "BC1_BY_CONSTRUCTION", False))
# `--tag external3_base` runs the same harness on the checkout WITHOUT the repair (the prereg
# commit's tree) so the gates compare main-with-BC-1 against main-without-BC-1, not against the
# wheel: main already differs from the wheel (V14) in what it reads, and that difference is not
# this repair's to claim or to answer for.
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "external3"
if TAG == "external3" and not BC1:
    sys.exit("external3: this checkout's diffgate does not carry BC-1 (pass --tag external3_base for the baseline)")
sys.path.insert(0, str(HERE))
from external1_harness import reconstruct  # noqa: E402  (its styxx imports resolve to the wheel)
assert sys.modules["styxx.diffgate"] is dg

DB = HERE / "external2_shelf.sqlite"   # the same shelf: the corpus did not change
LEDGER = HERE / f"{TAG}_ledger.jsonl"
GATE_SUMMARY = HERE / f"{TAG}_gate_summary.json"
DEF_TEST = re.compile(r"^\s*def (test_\w+)", re.M)


def stage_shelf(corpus: Path) -> int:
    import pyarrow.parquet as pq

    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE pr (id INTEGER PRIMARY KEY, agent TEXT, title TEXT, body TEXT, "
                "html_url TEXT, state TEXT, merged_at TEXT)")
    con.execute("CREATE TABLE f (pr_id INTEGER, filename TEXT, status TEXT, patch TEXT)")
    rows = pq.read_table(corpus / "pull_request.parquet",
                         columns=["id", "agent", "title", "body", "html_url", "state", "merged_at"]).to_pylist()
    con.executemany("INSERT OR REPLACE INTO pr VALUES (?,?,?,?,?,?,?)",
                    [(r["id"], r["agent"], r["title"], r["body"], r["html_url"], r["state"], r["merged_at"])
                     for r in rows])
    con.commit()
    ids = {r["id"] for r in rows}
    print(f"PRs shelved: {len(rows)}", flush=True)
    m = kept = 0
    pf = pq.ParquetFile(corpus / "pr_commit_details.parquet")
    for batch in pf.iter_batches(batch_size=50000, columns=["pr_id", "filename", "status", "patch"]):
        d = batch.to_pydict()
        out = []
        for pid, fn, st, patch in zip(d["pr_id"], d["filename"], d["status"], d["patch"]):
            m += 1
            if pid in ids:
                out.append((pid, fn, st, patch))
                kept += 1
        con.executemany("INSERT INTO f VALUES (?,?,?,?)", out)
        con.commit()
        print(f"  commit rows scanned: {m}  kept: {kept}", flush=True)
    con.execute("CREATE INDEX ix_f ON f(pr_id)")
    con.commit()
    con.close()
    print(f"commit rows scanned: {m}  kept: {kept}", flush=True)
    return 0


def _side(diff: str, sign: str) -> str:
    skip = sign * 3
    return "\n".join(l[1:] for l in diff.splitlines() if l.startswith(sign) and not l.startswith(skip))


def stage_gate() -> int:
    con = sqlite3.connect(DB)
    excl: Counter = Counter()
    agents: dict = {}
    tot: Counter = Counter()
    LEDGER.write_text("", encoding="utf-8")
    out = LEDGER.open("a", encoding="utf-8")
    seen = 0
    t0 = time.time()
    for pid, agent, title, body, url, state, merged in con.execute(
            "SELECT id, agent, title, body, html_url, state, merged_at FROM pr"):
        seen += 1
        if seen % 5000 == 0:
            print(f"  gated: {seen}  eligible: {tot['eligible']}  covered: {tot['covered']}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        if not body or not body.strip():
            excl["empty_body"] += 1
            continue
        files = con.execute("SELECT filename, status, patch FROM f WHERE pr_id=?", (pid,)).fetchall()
        if not files:
            excl["no_file_records"] += 1
            continue
        diff, implied = reconstruct(files)
        parsed, _blob = dg.parse_unified_diff(diff)
        if parsed != implied:
            excl["reconstruction_mismatch"] += 1
            continue
        tot["eligible"] += 1
        summary = f"{title or ''}\n\n{body}"
        try:
            g = dg.gate_diff_text(summary, diff, run=None, strict=False)
        except Exception as e:  # noqa: BLE001  -- never silently pass
            excl[f"gate_error:{type(e).__name__}"] += 1
            continue
        a = agents.setdefault(agent or "?", Counter())
        a["eligible"] += 1
        if g.claims:
            tot["covered"] += 1
            a["covered"] += 1
        for c in g.claims:
            tot[f"claim_{c.verdict}"] += 1
            a[f"claim_{c.verdict}"] += 1
        if any(c.verdict == "CONTRADICTED" for c in g.claims):
            tot["prs_with_contradiction"] += 1
            a["contradicted_prs"] += 1
        rec = {"pr_id": pid, "agent": agent, "html_url": url, "state": state, "merged": bool(merged),
               "verdict": g.verdict, "n_claims": len(g.claims),
               "exts": sorted({Path(fn).suffix.lower() for fn, _s, _p in files if fn}),
               "claims": [{"kind": c.kind, "text": c.text, "detail": c.detail, "verdict": c.verdict,
                           "why": c.why} for c in g.claims]}
        if any(c.kind == "tests_added" for c in g.claims):
            rec["def_test_added"] = DEF_TEST.findall(_side(diff, "+"))
            rec["def_test_removed"] = DEF_TEST.findall(_side(diff, "-"))
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    con.close()
    payload = {
        "corpus": {"dataset": "hao-li/AIDev", "tables": ["pull_request", "pr_commit_details"],
                   "license": "CC-BY-4.0", "zenodo": "10.5281/zenodo.16919272"},
        "instrument": {"styxx_version": styxx.__version__,
                       "diffgate_sha256": hashlib.sha256(Path(dg.__file__).read_bytes()).hexdigest(),
                       "run": None, "strict": False},
        "reconstruction": "external1_harness.reconstruct / _fold_statuses, unchanged",
        "instrument_is": ("this checkout with BC-1" if BC1 else "this checkout without BC-1 (baseline)")
                         + ", see PREREG_bc1_by_construction_2026_09_16.md",
        "bc1": BC1,
        "prs_seen": seen, "excluded": dict(excl),
        "eligible": tot["eligible"], "covered_prs": tot["covered"],
        "coverage": round(tot["covered"] / tot["eligible"], 4) if tot["eligible"] else None,
        "prs_with_contradiction": tot["prs_with_contradiction"],
        "claims_by_verdict": {k[6:]: v for k, v in tot.items() if k.startswith("claim_")},
        "per_agent": {k: dict(v) for k, v in sorted(agents.items())},
        "seconds": round(time.time() - t0),
    }
    GATE_SUMMARY.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("prs_seen", "excluded", "eligible", "covered_prs", "coverage",
                                                "prs_with_contradiction", "claims_by_verdict")}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["shelf", "gate"])
    ap.add_argument("--corpus", type=Path, default=HERE / "aidev",
                    help="directory holding pull_request.parquet and pr_commit_details.parquet")
    ap.add_argument("--tag", default="external3", help="ledger/summary name: external3 or external3_base")
    a = ap.parse_args()
    return stage_shelf(a.corpus) if a.stage == "shelf" else stage_gate()


if __name__ == "__main__":
    sys.exit(main())
