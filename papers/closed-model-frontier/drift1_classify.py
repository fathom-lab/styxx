"""DRIFT-1: was a false file count false when it was written, or only by the time we read it?

Preregistration `PREREG_drift1_stale_or_false_2026_09_18.md`, sha256 `9e5ac4e2…` at freeze and
`1f5829c7…` after Amendment A, both appended before any claim below was classified.

Fully local. The AIDev tables (CC-BY-4.0, Zenodo 10.5281/zenodo.16919272) supply commit order and
per-commit file lists; `_diffs/` supplies the pull request as it stands today, each file's sha256
already checked against the published dataset row by `bench_reproduce.py --fetch`.

    python papers/closed-model-frontier/drift1_classify.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict, Counter
from math import sqrt
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
# The AIDev parquet tables are not vendored (CC-BY-4.0, fetched from HuggingFace hao-li/AIDev).
# Pass the directory holding pr_commits.parquet and pr_commit_details.parquet as argv[1].
AIDEV = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "_aidev")
DIFFS = HERE / "_diffs"
DATASET = HERE / "bench2_dataset.jsonl"
OUT = HERE / "drift1_classification.json"

DIFF_HEADER = re.compile(r"^diff --git ", re.M)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * max(0.0, c - h), 1), round(100 * min(1.0, c + h), 1)


def rows() -> list[dict]:
    out = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith('{"_header"'):
            continue
        out.append(json.loads(line))
    return out


def diff_file_count(pr_id: int) -> int | None:
    p = DIFFS / f"{pr_id}.diff"
    if not p.exists():
        return None
    return len(DIFF_HEADER.findall(p.read_text(encoding="utf-8", errors="replace")))


def main() -> int:
    claims = [r for r in rows() if r["kind"] == "files_changed_count"]
    wanted = {r["pr_id"] for r in claims}

    # commit order, in stored row order — validated as a prefix of GitHub's order, 10 of 10 (G-DR1-1)
    order: "OrderedDict[int, list[str]]" = OrderedDict()
    pf = pq.ParquetFile(AIDEV / "pr_commits.parquet")
    for batch in pf.iter_batches(batch_size=200_000, columns=["sha", "pr_id"]):
        for r in batch.to_pylist():
            if r["pr_id"] in wanted:
                order.setdefault(r["pr_id"], []).append(r["sha"])

    # files per (pr, commit)
    files: dict[tuple[int, str], set[str]] = {}
    pd = pq.ParquetFile(AIDEV / "pr_commit_details.parquet")
    for batch in pd.iter_batches(batch_size=200_000, columns=["pr_id", "sha", "filename"]):
        for r in batch.to_pylist():
            if r["pr_id"] in wanted and r["filename"]:
                files.setdefault((r["pr_id"], r["sha"]), set()).add(r["filename"])

    # prefix counts per pull request
    prefixes: dict[int, list[int]] = {}
    for pid, shas in order.items():
        seen: set[str] = set()
        counts = []
        for sha in shas:
            seen |= files.get((pid, sha), set())
            counts.append(len(seen))
        prefixes[pid] = counts

    excluded, items = [], []
    for c in claims:
        pid = c["pr_id"]
        pre = prefixes.get(pid)
        fetched = diff_file_count(pid)
        if not pre or fetched is None:
            excluded.append({"url": c["url"], "reason": "no commit rows" if not pre else "no cached diff"})
            continue
        head = pre[-1]
        # G-DR1-2: the reconstruction must agree with the diff we fetched from source.
        if head != fetched:
            excluded.append({"url": c["url"], "reason": "corpus head != fetched diff",
                             "corpus_head": head, "fetched": fetched, "commits": len(pre)})
            continue
        stated = int(c["claim_detail"]["n"])
        distinct = sorted(set(pre))
        if stated == head:
            klass, first_k = "MATCHES_CORPUS_HEAD", None
        elif stated in pre[:-1]:
            klass, first_k = "STALE", pre.index(stated)
        else:
            klass, first_k = "NEITHER", None
        items.append({
            "url": c["url"], "claim": c["claim_text"].strip()[:120], "stated": stated,
            "head": head, "commits": len(pre), "prefix_counts": pre,
            "distinct_prefix_counts": len(distinct), "class": klass, "true_at_commit": first_k,
            "oracle": c["truth"],
        })

    by = Counter(i["class"] for i in items)
    disagree = [i for i in items if i["class"] != "MATCHES_CORPUS_HEAD"]
    stale = [i for i in disagree if i["class"] == "STALE"]
    weak = [i for i in stale if i["distinct_prefix_counts"] <= 2]

    report = {
        "prereg": "PREREG_drift1_stale_or_false_2026_09_18.md",
        "prereg_sha256_frozen": "9e5ac4e27bf64e205d8056187bdfc34888e6577c58aa510a8e1457de48de1e43",
        "prereg_sha256_amended": "1f5829c749ce5d3aa9bb397d1a93fe4826df50cea9bd84d98d796df429f8a580",
        "claims_in_population": len(claims),
        "excluded": excluded,
        "classified": len(items),
        "by_class": dict(by),
        "disagree_with_head": len(disagree),
        "stale_share_of_disagreements": round(len(stale) / len(disagree), 4) if disagree else None,
        "stale_share_wilson": wilson(len(stale), len(disagree)) if disagree else None,
        "stale_with_only_two_distinct_prefix_counts": len(weak),
        "oracle_supported": sum(1 for c in claims if c["truth"] == "SUPPORTED"),
        "items": items,
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"population (file-count claims)   {len(claims)}")
    print(f"excluded by G-DR1-2 / missing    {len(excluded)}")
    print(f"classified                       {len(items)}")
    for k in ("MATCHES_CORPUS_HEAD", "STALE", "NEITHER"):
        print(f"   {k:22s} {by.get(k,0)}")
    if disagree:
        print(f"\ndisagree with head               {len(disagree)}")
        print(f"   of those, STALE               {len(stale)}  "
              f"= {100*len(stale)/len(disagree):.1f}%  95% Wilson {wilson(len(stale), len(disagree))}")
        print(f"   STALE on only 2 distinct counts {len(weak)}   (weakest evidence)")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
