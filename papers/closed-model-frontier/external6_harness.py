"""HARNESS-1 harness: the EXTERNAL-1 reconstruction re-folded without merge traffic; the row cap marked.

Corpus: HuggingFace ``hao-li/AIDev`` (CC-BY-4.0; Zenodo 10.5281/zenodo.16919272), the same two
tables every EXTERNAL run read, from the published parquet files (``pull_request.parquet``,
``pr_commit_details.parquet``) with pyarrow.  What changes here is the HARNESS, per
``PREREG_harness1_merge_fold_2026_09_16.md``; the instrument (``styxx/diffgate.py``, this
checkout) is untouched:

  * the shelf keeps commit identity — every file row carries its commit ``sha``, and a commit
    table carries each commit's ``message`` and its row count;
  * the fold takes every row whose commit message does not begin with ``Merge `` (after leading
    whitespace).  A PR whose rows are all merge commits keeps them all (there is nothing else to
    read) and is counted as ``all_merge``;
  * a PR with any commit at the dataset's 300-row per-commit cap is marked ``capped``.  Its
    ``files_changed_count`` claims are re-read as UNCHECKABLE with the reason
    ``CAP_REASON`` — applied to the ledger record AFTER the instrument gated the diff, kept
    beside the instrument's own reading under ``pre_cap``, and counted;
  * everything else is EXTERNAL-3's harness verbatim: ``reconstruct`` imported unchanged from
    ``external1_harness.py`` (one header pair per file, net status, patches appended), the same
    eligibility and exclusions (empty body, no file records, reconstruction mismatch), the same
    ledger shape plus the fold fields (``n_files``, ``commits``, ``merge_commits``,
    ``rows_total``, ``rows_folded``, ``all_merge``, ``capped``, ``max_rows_per_commit``,
    ``n_files_all_rows`` — the count the old fold would have given — and ``cap_rewrote``).

``harness1_gates.py`` scores the ledger against ``external4_ledger.jsonl`` (the fold as it was,
same instrument on reconstructed diffs — G-BIN-3) and against EXTERNAL-5's live counts.  The
shelf and the ledger name third-party PRs and are gitignored; only the aggregate receipts are
committed.

    python external6_harness.py shelf --corpus DIR   # rebuilds the file table WITH commit identity
    python external6_harness.py gate
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

sys.path.insert(0, str(ROOT))  # the checkout FIRST: this run measures the harness under the current file
import styxx  # noqa: E402
import styxx.diffgate as dg  # noqa: E402
if not Path(dg.__file__).resolve().is_relative_to(ROOT):
    sys.exit("external6: styxx.diffgate resolved to an installed package, not this checkout")
sys.path.insert(0, str(HERE))
from external1_harness import reconstruct  # noqa: E402  (unchanged: one header pair per file, net status)
assert sys.modules["styxx.diffgate"] is dg

DB = HERE / "external6_shelf.sqlite"
# `--tag NAME` writes NAME_ledger.jsonl / NAME_gate_summary.json from the same shelf and the same
# fold, so a later instrument change (COMPAT-2 runs `--tag compat2`) is measured ledger to ledger
# against external6 with one variable: the instrument.
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "external6"
LEDGER = HERE / f"{TAG}_ledger.jsonl"
SUMMARY = HERE / f"{TAG}_gate_summary.json"
DEF_TEST = re.compile(r"^\s*def (test_\w+)", re.M)
ROW_CAP = 300
CAP_REASON = "corpus rows capped at 300 per commit; the count cannot be reconstructed"


def is_merge_message(message: str | None) -> bool:
    """The prereg's rule: the commit message begins ``Merge `` after leading whitespace."""
    return (message or "").lstrip().startswith("Merge ")


def stage_shelf(corpus: Path, db: Path = DB) -> int:
    import pyarrow.parquet as pq

    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE pr (id INTEGER PRIMARY KEY, agent TEXT, title TEXT, body TEXT, "
                "html_url TEXT, state TEXT, merged_at TEXT)")
    con.execute("CREATE TABLE c (pr_id INTEGER, sha TEXT, message TEXT, rows INTEGER)")
    con.execute("CREATE TABLE f (pr_id INTEGER, sha TEXT, filename TEXT, status TEXT, patch TEXT)")
    rows = pq.read_table(corpus / "pull_request.parquet",
                         columns=["id", "agent", "title", "body", "html_url", "state", "merged_at"]).to_pylist()
    con.executemany("INSERT OR REPLACE INTO pr VALUES (?,?,?,?,?,?,?)",
                    [(r["id"], r["agent"], r["title"], r["body"], r["html_url"], r["state"], r["merged_at"])
                     for r in rows])
    con.commit()
    ids = {r["id"] for r in rows}
    print(f"PRs shelved: {len(rows)}", flush=True)
    m = kept = 0
    commits: dict[tuple, list] = {}   # (pr_id, sha) -> [message, rows]
    pf = pq.ParquetFile(corpus / "pr_commit_details.parquet")
    for batch in pf.iter_batches(batch_size=50000, columns=["pr_id", "sha", "message", "filename", "status", "patch"]):
        d = batch.to_pydict()
        out = []
        for pid, sha, msg, fn, st, patch in zip(d["pr_id"], d["sha"], d["message"], d["filename"], d["status"], d["patch"]):
            m += 1
            if pid in ids:
                out.append((pid, sha, fn, st, patch))
                kept += 1
                c = commits.setdefault((pid, sha), [msg, 0])
                c[1] += 1
        con.executemany("INSERT INTO f VALUES (?,?,?,?,?)", out)
        con.commit()
        print(f"  commit rows scanned: {m}  kept: {kept}", flush=True)
    con.executemany("INSERT INTO c VALUES (?,?,?,?)",
                    [(pid, sha, msg, n) for (pid, sha), (msg, n) in commits.items()])
    con.execute("CREATE INDEX ix_f ON f(pr_id)")
    con.execute("CREATE INDEX ix_c ON c(pr_id)")
    con.commit()
    n_merge = sum(1 for (msg, _n) in commits.values() if is_merge_message(msg))
    n_cap = sum(1 for (_msg, n) in commits.values() if n >= ROW_CAP)
    con.close()
    print(f"commit rows scanned: {m}  kept: {kept}  commits: {len(commits)}  "
          f"merge commits: {n_merge}  commits at the row cap: {n_cap}", flush=True)
    return 0


def _side(diff: str, sign: str) -> str:
    skip = sign * 3
    return "\n".join(l[1:] for l in diff.splitlines() if l.startswith(sign) and not l.startswith(skip))


def fold_rows(commits: list[tuple], rows: list[tuple]) -> tuple[list[tuple], dict]:
    """The prereg's fold: drop rows from merge commits unless every commit is one; mark the cap.

    ``commits``: (sha, message, n_rows) per commit of the PR.  ``rows``: (sha, filename, status,
    patch) per file row.  Returns the rows to reconstruct from, as (filename, status, patch),
    and the fold's facts for the ledger.
    """
    merge = {sha for sha, msg, _n in commits if is_merge_message(msg)}
    max_rows = max((n for _s, _m, n in commits), default=0)
    kept = [(fn, st, patch) for sha, fn, st, patch in rows if sha not in merge]
    all_merge = bool(rows) and not kept
    if all_merge:
        kept = [(fn, st, patch) for _sha, fn, st, patch in rows]
    facts = {"commits": len(commits), "merge_commits": len(merge), "rows_total": len(rows),
             "rows_folded": len(kept), "all_merge": all_merge,
             "capped": max_rows >= ROW_CAP, "max_rows_per_commit": max_rows,
             "n_files_all_rows": len({fn for _sha, fn, _st, _p in rows if fn})}
    return kept, facts


def stage_gate(db: Path = DB) -> int:
    con = sqlite3.connect(db)
    excl: Counter = Counter()
    agents: dict = {}
    tot: Counter = Counter()
    fold: Counter = Counter()
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
        rows = con.execute("SELECT sha, filename, status, patch FROM f WHERE pr_id=?", (pid,)).fetchall()
        if not rows:
            excl["no_file_records"] += 1
            continue
        commits = con.execute("SELECT sha, message, rows FROM c WHERE pr_id=?", (pid,)).fetchall()
        files, facts = fold_rows(commits, rows)
        diff, implied = reconstruct(files)
        parsed, _blob = dg.parse_unified_diff(diff)
        if parsed != implied:
            excl["reconstruction_mismatch"] += 1
            continue
        tot["eligible"] += 1
        if facts["merge_commits"]:
            fold["prs_with_a_merge_commit"] += 1
            fold["rows_dropped"] += facts["rows_total"] - facts["rows_folded"]
        if facts["all_merge"]:
            fold["prs_all_merge_commits_kept"] += 1
        if facts["capped"]:
            fold["prs_capped"] += 1
        if len(implied) != facts["n_files_all_rows"]:
            fold["prs_whose_file_count_changed"] += 1
        summary = f"{title or ''}\n\n{body}"
        try:
            g = dg.gate_diff_text(summary, diff, run=None, strict=False)
        except Exception as e:  # noqa: BLE001  -- never silently pass
            excl[f"gate_error:{type(e).__name__}"] += 1
            continue
        claims = []
        cap_rewrote = 0
        for c in g.claims:
            rec_c = {"kind": c.kind, "text": c.text, "detail": c.detail, "verdict": c.verdict, "why": c.why}
            if facts["capped"] and c.kind == "files_changed_count":
                rec_c["pre_cap"] = {"verdict": c.verdict, "why": c.why}
                rec_c["verdict"] = "UNCHECKABLE"
                rec_c["why"] = CAP_REASON
                cap_rewrote += 1
            claims.append(rec_c)
        fold["claims_rewritten_by_the_cap"] += cap_rewrote
        a = agents.setdefault(agent or "?", Counter())
        a["eligible"] += 1
        if claims:
            tot["covered"] += 1
            a["covered"] += 1
        for c in claims:
            tot[f"claim_{c['verdict']}"] += 1
            a[f"claim_{c['verdict']}"] += 1
        if any(c["verdict"] == "CONTRADICTED" for c in claims):
            tot["prs_with_contradiction"] += 1
            a["contradicted_prs"] += 1
        rec = {"pr_id": pid, "agent": agent, "html_url": url, "state": state, "merged": bool(merged),
               "verdict": g.verdict, "n_claims": len(claims),
               "exts": sorted({Path(fn).suffix.lower() for fn, _s, _p in files if fn}),
               "n_files": len(implied), "cap_rewrote": cap_rewrote, **facts,
               "claims": claims}
        if any(c["kind"] == "tests_added" for c in claims):
            rec["def_test_added"] = DEF_TEST.findall(_side(diff, "+"))
            rec["def_test_removed"] = DEF_TEST.findall(_side(diff, "-"))
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    con.close()
    payload = {
        "prereg": "PREREG_harness1_merge_fold_2026_09_16.md" if TAG == "external6" else f"{TAG}: the HARNESS-1 fold under this checkout's instrument",
        "tag": TAG,
        "corpus": {"dataset": "hao-li/AIDev", "tables": ["pull_request", "pr_commit_details"],
                   "license": "CC-BY-4.0", "zenodo": "10.5281/zenodo.16919272"},
        "instrument": {"styxx_version": styxx.__version__,
                       "diffgate_sha256": hashlib.sha256(Path(dg.__file__).read_bytes()).hexdigest(),
                       "run": None, "strict": False, "changed_here": False},
        "reconstruction": "external1_harness.reconstruct, unchanged; rows folded by external6_harness.fold_rows: "
                          "rows from commits whose message begins 'Merge ' dropped unless every commit is one; "
                          f"a commit with >= {ROW_CAP} rows marks the PR capped and its files_changed_count "
                          "claims UNCHECKABLE after gating",
        "cap_reason": CAP_REASON,
        "prs_seen": seen, "excluded": dict(excl),
        "eligible": tot["eligible"], "covered_prs": tot["covered"],
        "coverage": round(tot["covered"] / tot["eligible"], 4) if tot["eligible"] else None,
        "prs_with_contradiction": tot["prs_with_contradiction"],
        "claims_by_verdict": {k[6:]: v for k, v in tot.items() if k.startswith("claim_")},
        "fold": dict(fold),
        "per_agent": {k: dict(v) for k, v in sorted(agents.items())},
        "seconds": round(time.time() - t0),
    }
    SUMMARY.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("prs_seen", "excluded", "eligible", "covered_prs", "coverage",
                                                "prs_with_contradiction", "claims_by_verdict", "fold")}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["shelf", "gate"])
    ap.add_argument("--corpus", type=Path, default=HERE / "aidev",
                    help="directory holding pull_request.parquet and pr_commit_details.parquet")
    ap.add_argument("--shelf", type=Path, default=DB, help="where the shelf lives (5 GB; gitignored)")
    ap.add_argument("--tag", default="external6", help="ledger/summary name: external6, or compat2 for the COMPAT-2 run")
    a = ap.parse_args()
    return stage_shelf(a.corpus, a.shelf) if a.stage == "shelf" else stage_gate(a.shelf)


if __name__ == "__main__":
    sys.exit(main())
