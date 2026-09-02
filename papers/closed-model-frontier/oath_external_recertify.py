# -*- coding: utf-8 -*-
"""oath_external_recertify.py -- rebuild the external OATH corpus at its pinned shas and re-certify
it with the CURRENT verifier, so every token carries the verifier's own epistemics.

WHY. `oath_external_corpus.json` pins every document and receipt of the 2026-08-27 external corpus
to a commit and a sha256, but the bytes lived in a temporary fetch cache that is gone. Anything
that needs the verifier's per-token judgement on that corpus -- the obligation source above all --
has had to be RE-DERIVED from the harness ledger, and PREREG_handedness_accusations_2026_09_02
showed that re-derivation diverging on half the accusations. This script ends the re-derivation:
fetch, hash-verify, re-certify, record what the verifier says.

WHAT IS COMMITTED. Only the derived ledger and a summary: one row per token with status, branch,
obligation and source, plus whether the obligation came through a table HEADER rather than the
line itself (the binding_context the verifier reads). The third-party bytes are cached outside the
tree (`$OATH_EXT_CACHE`), hash-verified on every read, and never committed: they are other people's
READMEs, pinned and reproducible from the manifest by anyone.

  python oath_external_recertify.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.certify import _TRIGGERS, certify_doc, extract_numbers   # noqa: E402

MANIFEST = HERE / "oath_external_corpus.json"
LEDGER_OUT = HERE / "oath_external_epistemics_ledger.jsonl"
SUMMARY_OUT = HERE / "oath_external_recertify_summary.json"
CACHE = Path(os.environ.get("OATH_EXT_CACHE", Path(os.environ.get("TEMP", "/tmp")) / "oath_ext_corpus_cache"))
RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
CONTEXT_CHARS = 200


def fetch(repo: str, sha: str, path: str, want_sha256: str):
    """Bytes at a pinned sha, cached on disk under the harness's own key, hash-verified always."""
    key = hashlib.sha256(f"{repo}@{sha}/{path}".encode()).hexdigest()[:32]
    blob = CACHE / key
    if blob.exists():
        data = blob.read_bytes()
        if hashlib.sha256(data).hexdigest() == want_sha256:
            return data, "cache"
    try:
        with urllib.request.urlopen(RAW.format(repo=repo, sha=sha, path=path), timeout=90) as r:
            data = r.read()
    except Exception as e:                                  # noqa: BLE001 - recorded, never raised
        return None, "fetch_failed:%s" % type(e).__name__
    if hashlib.sha256(data).hexdigest() != want_sha256:
        return None, "hash_mismatch"
    CACHE.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    return data, "fetched"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows_out = []
    summary = {"what": "external OATH corpus re-certified at the current verifier, from pinned shas",
               "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
               "repos_certified_in_manifest": 0, "repos_recertified": 0, "repos_skipped": {},
               "files": collections.Counter(), "status_counts": collections.Counter(),
               "obligation_sources": collections.Counter(), "header_bound_obligations": 0,
               "verdict_class_drift": [], "per_repo": {}}
    for rec in manifest["per_repo"]:
        if rec.get("status") != "CERTIFIED":
            continue
        summary["repos_certified_in_manifest"] += 1
        repo, sha = rec["repo"], rec["sha"]
        doc_bytes, receipts, bad = None, [], []
        for f in rec["files"]:
            data, how = fetch(repo, sha, f["path"], f["sha256"])
            summary["files"][how.split(":")[0]] += 1
            if data is None:
                bad.append((f["path"], how))
                continue
            if f["role"] == "document":
                doc_bytes = data
            else:
                receipts.append((f["path"], data))
        if doc_bytes is None or bad:
            summary["repos_skipped"][repo] = bad or "no document"
            continue
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "README.md"
            d.write_bytes(doc_bytes)
            rps = []
            for p, data in receipts:
                rp = Path(td) / (p.replace("/", "__"))
                rp.write_bytes(data)
                rps.append(rp)
            try:
                cert = certify_doc(d, rps)
            except Exception as e:                          # noqa: BLE001
                summary["repos_skipped"][repo] = "certify_failed:%s" % type(e).__name__
                continue
            text = d.read_text(encoding="utf-8")
        lines = text.splitlines()
        bctx_at = {(n["line"], n.get("col")): n.get("binding_context") for n in extract_numbers(text)}
        summary["repos_recertified"] += 1
        old_class = str(rec.get("verdict") or "").split(",")[0]
        new_class = str(cert["verdict"]).split(",")[0]
        if old_class and old_class != new_class:
            summary["verdict_class_drift"].append({"repo": repo, "was": old_class, "now": new_class})
        summary["per_repo"][repo] = {"verdict": cert["verdict"], "counts": cert["counts"], "uncovered": cert.get("uncovered")}
        for e in cert["ledger"]:
            ep = e.get("epistemics", {})
            line = lines[e["line"] - 1] if 0 < e["line"] <= len(lines) else ""
            bctx = bctx_at.get((e["line"], e.get("col")))
            line_has = bool(_TRIGGERS.search(line))
            header_bound = bool(bctx) and bool(_TRIGGERS.search(bctx)) and not line_has
            row = {"repo": repo, "sha": sha, "line": e["line"], "col": e.get("col"), "token": e["token"],
                   "value": e["value"], "status": e["status"], "branch": ep.get("branch"),
                   "obligated": ep.get("obligated"), "obligation_source": ep.get("obligation_source"),
                   "line_has_trigger": line_has, "header_bound": header_bound,
                   "context": line.strip()[:CONTEXT_CHARS]}
            rows_out.append(row)
            summary["status_counts"][e["status"]] += 1
            if ep.get("obligated"):
                summary["obligation_sources"][ep.get("obligation_source")] += 1
                if header_bound:
                    summary["header_bound_obligations"] += 1
        print(f"  {repo:40s} {cert['verdict']:28s} tokens={len(cert['ledger']):4d}", flush=True)
    LEDGER_OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out) + "\n", encoding="utf-8")
    for k in ("files", "status_counts", "obligation_sources"):
        summary[k] = dict(summary[k])
    summary["tokens"] = len(rows_out)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nrecertified {summary['repos_recertified']}/{summary['repos_certified_in_manifest']} repos, "
          f"{summary['tokens']} tokens, skipped {len(summary['repos_skipped'])}, files {summary['files']}, "
          f"verdict-class drift {len(summary['verdict_class_drift'])}, header-bound obligations {summary['header_bound_obligations']}")
    return 0 if summary["repos_recertified"] else 1


if __name__ == "__main__":
    sys.exit(main())
