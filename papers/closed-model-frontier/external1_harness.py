"""EXTERNAL-1 harness: run the shipped gate over AIDev agent-authored PRs.

Prereg: PREREG_external1_aidev_2026_08_31.md (frozen before this ran).
Corpus: HuggingFace `hao-li/AIDev` (CC-BY-4.0; Zenodo 10.5281/zenodo.16919272),
the MSR 2026 Mining Challenge dataset. Not collected, curated, or chosen by us.

The instrument runs exactly as it ships: `gate_diff_text` on the public API, no
template added, removed, or tuned for this corpus. The only thing this file does
is RECONSTRUCT a unified diff from the corpus's per-file records — and it checks
its own reconstruction: if `parse_unified_diff` does not return the file→status
map the corpus rows imply, the PR is EXCLUDED and COUNTED, never scored.

  python papers/closed-model-frontier/external1_harness.py --stage shelf
  python papers/closed-model-frontier/external1_harness.py --stage gate
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.diffgate import gate_diff_text, parse_unified_diff, _norm   # noqa: E402
from styxx._version import __version__ as STYXX_VERSION                # noqa: E402

DB = HERE / "external1_shelf.sqlite"
LEDGER = HERE / "external1_ledger.jsonl"
SUMMARY = HERE / "external1_summary.json"
DATASET = "hao-li/AIDev"

# corpus status -> our reconstruction intent
ADDED = {"added"}
REMOVED = {"removed"}


def stage_shelf() -> int:
    """Stream the corpus into a local shelf: curated PRs + their file records."""
    from datasets import load_dataset

    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE pr (id INTEGER PRIMARY KEY, agent TEXT, title TEXT, "
                "body TEXT, html_url TEXT, state TEXT, merged_at TEXT)")
    con.execute("CREATE TABLE f (pr_id INTEGER, filename TEXT, status TEXT, patch TEXT)")

    n = 0
    for row in load_dataset(DATASET, "pull_request", split="train", streaming=True):
        con.execute("INSERT OR REPLACE INTO pr VALUES (?,?,?,?,?,?,?)",
                    (row["id"], row.get("agent"), row.get("title"), row.get("body"),
                     row.get("html_url"), row.get("state"), row.get("merged_at")))
        n += 1
        if n % 10000 == 0:
            con.commit()
            print(f"  PRs shelved: {n}", flush=True)
    con.commit()
    print(f"PRs shelved: {n}", flush=True)

    ids = {r[0] for r in con.execute("SELECT id FROM pr")}
    m = kept = 0
    batch = []
    for row in load_dataset(DATASET, "pr_commit_details", split="train", streaming=True):
        m += 1
        pid = row.get("pr_id")
        if pid in ids:
            batch.append((pid, row.get("filename"), row.get("status"), row.get("patch")))
            kept += 1
            if len(batch) >= 5000:
                con.executemany("INSERT INTO f VALUES (?,?,?,?)", batch)
                con.commit()
                batch.clear()
        if m % 200000 == 0:
            print(f"  commit rows scanned: {m}  kept: {kept}", flush=True)
    if batch:
        con.executemany("INSERT INTO f VALUES (?,?,?,?)", batch)
    con.commit()
    con.execute("CREATE INDEX ix_f ON f(pr_id)")
    con.commit()
    print(f"commit rows scanned: {m}  kept: {kept}", flush=True)
    con.close()
    return 0


def _fold_statuses(files: list[tuple]) -> dict:
    """Net per-file status across ALL commits of one PR.

    CORRECTION (2026-08-31, before EXTERNAL-1's cause analysis was published):
    the corpus carries one row per file PER COMMIT, and 6.1% of (PR, file)
    pairs disagree across commits — typically `added` in one commit and
    `modified` in a later one. The first version of this function emitted a
    hunk per row, and `parse_unified_diff` kept the LAST header it saw, so a
    file genuinely created by the PR was reported as merely modified. That
    manufactured status mismatches, which the gate then accused. The defect was
    the harness's, not the instrument's.

    Net rule, order-free because the shelf carries no commit ordering: a file
    added and never removed is ADDED for the PR; removed and never added is
    DELETED; anything else — including the rare add-then-remove within one PR,
    which nets to absent and is deliberately not claimed as either — is
    MODIFIED.
    """
    seen: dict[str, set] = {}
    for filename, status, _patch in files:
        if filename:
            seen.setdefault(filename, set()).add((status or "").lower())
    out = {}
    for filename, sts in seen.items():
        if (sts & ADDED) and not (sts & REMOVED):
            out[filename] = "added"
        elif (sts & REMOVED) and not (sts & ADDED):
            out[filename] = "removed"
        else:
            out[filename] = "modified"
    return out


def reconstruct(files: list[tuple]) -> tuple[str, dict]:
    """Per-file corpus records -> unified diff text + the status map they imply.

    One hunk group per FILE (not per commit row), carrying the net status, with
    every commit's patch text appended under it so added-line evidence survives.
    """
    net = _fold_statuses(files)
    patches: dict[str, list] = {}
    for filename, _status, patch in files:
        if filename and patch:
            patches.setdefault(filename, []).append(patch)

    parts, implied = [], {}
    for filename, status in net.items():
        st = status
        parts.append(f"diff --git a/{filename} b/{filename}")
        if st in ADDED:
            parts += ["new file mode 100644", "--- /dev/null", f"+++ b/{filename}"]
            implied[_norm(filename)] = "A"
        elif st in REMOVED:
            parts += ["deleted file mode 100644", f"--- a/{filename}", "+++ /dev/null"]
            implied[_norm(filename)] = "D"
        else:
            parts += [f"--- a/{filename}", f"+++ b/{filename}"]
            implied[_norm(filename)] = "M"
        for patch in patches.get(filename, []):
            parts.append(patch)
    return "\n".join(parts) + "\n", implied


def stage_gate() -> int:
    con = sqlite3.connect(DB)
    excl = Counter()
    agents = {}
    n_elig = n_covered = n_contra = 0
    claims_total = Counter()
    LEDGER.write_text("", encoding="utf-8")
    out = LEDGER.open("a", encoding="utf-8")

    seen = 0
    for pid, agent, title, body, url, state, merged in con.execute(
            "SELECT id, agent, title, body, html_url, state, merged_at FROM pr"):
        seen += 1
        if seen % 5000 == 0:
            print(f"  gated: {seen}  eligible: {n_elig}  covered: {n_covered}", flush=True)
        if not body or not body.strip():
            excl["empty_body"] += 1
            continue
        files = con.execute(
            "SELECT filename, status, patch FROM f WHERE pr_id=?", (pid,)).fetchall()
        if not files:
            # the corpus's own disclosed boundary: absent patch/file records
            excl["no_file_records"] += 1
            continue
        diff, implied = reconstruct(files)
        parsed, _blob = parse_unified_diff(diff)
        if parsed != implied:
            # our reconstruction, not the instrument, is unfaithful here
            excl["reconstruction_mismatch"] += 1
            continue

        n_elig += 1
        summary = f"{title or ''}\n\n{body}"
        try:
            g = gate_diff_text(summary, diff, run=None, strict=False)
        except Exception as e:                                   # never silently pass
            excl[f"gate_error:{type(e).__name__}"] += 1
            continue

        a = agents.setdefault(agent or "?", Counter())
        a["eligible"] += 1
        for c in g.claims:
            claims_total[c.verdict] += 1
            a[f"claim_{c.verdict}"] += 1
        if g.claims:
            n_covered += 1
            a["covered"] += 1
        contra = [c for c in g.claims if c.verdict == "CONTRADICTED"]
        if contra:
            n_contra += 1
            a["contradicted_prs"] += 1
        out.write(json.dumps({
            "pr_id": pid, "agent": agent, "html_url": url, "state": state,
            "merged": bool(merged), "verdict": g.verdict,
            "n_claims": len(g.claims),
            "sentences_total": g.sentences_total,
            "uncovered_sentences": g.uncovered_sentences,
            "claims": [{"kind": c.kind, "text": c.text, "detail": c.detail,
                        "verdict": c.verdict, "why": c.why} for c in g.claims],
        }, ensure_ascii=False) + "\n")
    out.close()
    con.close()

    payload = {
        "prereg": "PREREG_external1_aidev_2026_08_31.md",
        "corpus": {"dataset": DATASET, "config": "pull_request",
                   "license": "CC-BY-4.0", "zenodo": "10.5281/zenodo.16919272"},
        "instrument": {"styxx_version": STYXX_VERSION, "diffgate": "v0",
                       "run": None, "strict": False},
        "prs_seen": seen,
        "excluded": dict(excl),
        "eligible": n_elig,
        "covered_prs": n_covered,
        "coverage": round(n_covered / n_elig, 4) if n_elig else None,
        "prs_with_contradiction": n_contra,
        "contradiction_rate_of_covered": round(n_contra / n_covered, 4) if n_covered else None,
        "claims_by_verdict": dict(claims_total),
        "per_agent": {k: dict(v) for k, v in sorted(agents.items())},
    }
    SUMMARY.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("prs_seen", "excluded", "eligible", "covered_prs", "coverage",
                       "prs_with_contradiction", "contradiction_rate_of_covered",
                       "claims_by_verdict")}, indent=1))
    print(f"-> {SUMMARY.name}, {LEDGER.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("shelf", "gate"), required=True)
    a = ap.parse_args()
    return stage_shelf() if a.stage == "shelf" else stage_gate()


if __name__ == "__main__":
    sys.exit(main())
