"""OATH-EXT RECON — point the shipped verifier at claim documents this lab did not write.

**This is a RECON. It licenses no claim.** Its job is to size a class and to find out whether
the instrument produces signal or noise outside the corpus it grew up in. Numbers it produces
are inputs to a future preregistration's bars, not results. Nothing here is a finding.

## Why this exists

All 178 committed OATH certificates are on documents written by this lab. Twelve cycles of
instrument work — v0.1 through v0.11 — have been scored entirely against a corpus the
instrument's authors also wrote. That is the single largest unexamined assumption in the OATH
lane, and a hostile reader finds it immediately: an instrument tuned on its own authors' prose
may be measuring their idiom rather than anything about claims.

A different lane already went outside (`papers/RESULT_sp_ext_2026_08_21.md`, silent-pass defects
in third-party code) and imported the right standard from it, which this RECON inherits verbatim:

> **styxx does not adjudicate whether the external authors are right.** The only thing measured
> is whether a number in a repository's own claim document grounds in that repository's own
> committed receipt. A disagreement is an internal inconsistency between two artifacts the same
> authors published — not this lab's opinion about their work.

## Selection rule, pre-committed before any repository was read

GitHub code search for `filename:all_results.json` and `filename:eval_results.json` — the summary
metric files the HuggingFace Trainer writes, which are among the few receipt shapes in the wild
that are genuinely SUMMARY-shaped rather than per-item bulk. The first `MAX_REPOS` DISTINCT
repositories in the API's own returned order are taken, with no inspection and no substitution.

Honest disclosure, stated before the numbers: GitHub's default ordering is "best match", not
random. This is a CONVENIENCE SAMPLE of a mechanically-defined population, exactly as SP-EXT's
two cases were, and it supports no inference about base rates in any wider population. It is
reported because a mechanically-selected sample is auditable and a hand-picked one is not.

## Safety

External repositories are CLONED AND READ. No code from them is imported, executed, or
evaluated — this harness opens files and nothing else. Clones are shallow, pinned by commit sha
in the output, and confined to a temp directory.

  python papers/closed-model-frontier/oath_ext_recon.py [--max-repos N]
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import certify_doc                                       # noqa: E402

C = importlib.import_module("styxx.certify")

OUT = HERE / "oath_ext_recon.json"

# --- pre-committed selection parameters (frozen here before any repository was read) -----------
SEARCH_QUERIES = ("filename:all_results.json", "filename:eval_results.json")
MAX_REPOS = 14
RECEIPT_NAMES = ("all_results.json", "eval_results.json", "test_results.json",
                 "train_results.json", "metrics.json")
MAX_RECEIPTS_PER_DOC = 12          # keeps `receipt_values` from flattening a whole model zoo
MAX_RECEIPT_BYTES = 2_000_000
MAX_DOC_BYTES = 400_000
DOC_GLOBS = ("README.md", "readme.md", "README.MD")


def gh_search_repos() -> list[dict]:
    """Distinct repositories, in the API's own returned order, no inspection, no substitution."""
    seen, repos = set(), []
    for q in SEARCH_QUERIES:
        for page in (1, 2):
            cmd = ["gh", "api", "-X", "GET", "search/code",
                   "-f", f"q={q}", "-f", "per_page=100", "-f", f"page={page}"]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=90,
                                   encoding="utf-8", errors="replace")
            except Exception:
                break
            if r.returncode != 0:
                break
            try:
                items = json.loads(r.stdout).get("items", [])
            except Exception:
                break
            for it in items:
                full = it.get("repository", {}).get("full_name")
                if full and full not in seen:
                    seen.add(full)
                    repos.append({"repo": full,
                                  "clone_url": it["repository"]["html_url"] + ".git"})
                if len(repos) >= MAX_REPOS:
                    return repos
            if not items:
                break
    return repos


def clone(entry: dict, dest: Path) -> str | None:
    try:
        r = subprocess.run(["git", "clone", "--depth", "1", "-q", entry["clone_url"], str(dest)],
                           capture_output=True, text=True, timeout=300,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            return None
        sha = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=60,
                             encoding="utf-8", errors="replace")
        return sha.stdout.strip() or None
    except Exception:
        return None


def receipts_in(repo_dir: Path) -> list[Path]:
    out = []
    for name in RECEIPT_NAMES:
        for p in repo_dir.rglob(name):
            if ".git" in p.parts:
                continue
            try:
                if p.stat().st_size > MAX_RECEIPT_BYTES:
                    continue
                json.loads(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            out.append(p)
            if len(out) >= MAX_RECEIPTS_PER_DOC:
                return out
    return out


def docs_in(repo_dir: Path) -> list[Path]:
    for g in DOC_GLOBS:
        p = repo_dir / g
        if p.exists() and p.stat().st_size <= MAX_DOC_BYTES:
            return [p]
    return []


_INDEX_NAMES = frozenset({"line", "col", "seed", "token", "case", "i", "index", "idx",
                          "step", "epoch", "num", "count", "size", "id"})
_SUBSCRIPT = re.compile(r"\[\d+\]$")


def _why_obligated(line: str) -> dict:
    """WHICH trigger bound this token. An accusation nobody can trace to a trigger cannot be
    adjudicated, and an accusation whose trigger is a config word rather than a result word is
    the v0.11 defect class wearing different clothes."""
    return {
        "triggers": sorted({m.group(0).lower() for m in C._TRIGGERS.finditer(line)}),
        "triggers_corr": sorted({m.group(0).lower() for m in C._TRIGGERS_CORR.finditer(line)}),
        "n_equals": bool(re.search(r"n\s*=", line, re.I)),
    }


def summarise_accusation(cert: dict, doc_rel: str, repo: str, sha: str,
                         doc_lines: list) -> list[dict]:
    """Every UNGROUNDED token, by coordinate, with the TRIGGER that bound it.

    The roster is the POINT of this RECON. A count of accusations means nothing until someone has
    read them and said which are real; the v0.11 cycle exists because four of this lab's own
    accusations turned out not to be claims at all.
    """
    out = []
    for e in cert["ledger"]:
        if e["status"] != "UNGROUNDED":
            continue
        line = doc_lines[e["line"] - 1] if e["line"] - 1 < len(doc_lines) else e["context"]
        out.append({"repo": repo, "sha": sha, "doc": doc_rel, "line": e["line"],
                    "token": e["token"], "decimals": e["decimals"],
                    "full_line": line.strip()[:400],
                    "obligated_by": _why_obligated(line)})
    return out


def summarise_verification(cert: dict, doc_rel: str, repo: str, sha: str) -> list[dict]:
    """Every VERIFIED token with the leaf it was sworn to, graded by the dogfood definition.

    A verification against an index-like leaf is a coincidence, not a measurement — the channel
    v0.8 closed NEGATIVE and v0.11 retracted four accusations over. If the external VERIFIED
    column is mostly coincidence, the 'it verifies things' half of the instrument is as empty
    outside this lab as the accusing half."""
    out = []
    for e in cert["ledger"]:
        if e["status"] != "VERIFIED" or not e["receipt_ref"]:
            continue
        _r, _, path = e["receipt_ref"].partition(":")
        term = _SUBSCRIPT.sub("", path.rsplit(".", 1)[-1]).lower()
        out.append({"repo": repo, "sha": sha, "doc": doc_rel, "line": e["line"],
                    "token": e["token"], "receipt_ref": e["receipt_ref"],
                    "terminal": term,
                    "structurally_coincident": bool(_SUBSCRIPT.search(path))
                                               or term in _INDEX_NAMES,
                    "context": e["context"][:160]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-repos", type=int, default=MAX_REPOS)
    a = ap.parse_args()

    t0 = time.time()
    repos = gh_search_repos()[:a.max_repos]
    if not repos:
        print("no repositories returned by code search (gh auth? rate limit?)")
        return 2
    print(f"selected {len(repos)} repositories mechanically")

    work = Path(tempfile.mkdtemp(prefix="oath_ext_"))
    rows, accusations, verifications = [], [], []
    totals = {"VERIFIED": 0, "ABSTAIN": 0, "UNGROUNDED": 0}
    try:
        for i, entry in enumerate(repos, 1):
            dest = work / f"r{i}"
            sha = clone(entry, dest)
            if not sha:
                rows.append({"repo": entry["repo"], "status": "CLONE_FAILED"})
                print(f"  [{i}/{len(repos)}] {entry['repo']}: clone failed")
                continue
            docs, receipts = docs_in(dest), receipts_in(dest)
            if not docs or not receipts:
                rows.append({"repo": entry["repo"], "sha": sha, "status": "NO_PAIR",
                             "docs": len(docs), "receipts": len(receipts)})
                print(f"  [{i}/{len(repos)}] {entry['repo']}: no (doc, receipt) pair")
                shutil.rmtree(dest, ignore_errors=True)
                continue
            doc = docs[0]
            try:
                cert = certify_doc(doc, receipts)
            except Exception as exc:
                rows.append({"repo": entry["repo"], "sha": sha, "status": "CERTIFY_ERROR",
                             "error": str(exc)[:200]})
                print(f"  [{i}/{len(repos)}] {entry['repo']}: certify error")
                shutil.rmtree(dest, ignore_errors=True)
                continue
            c = cert["counts"]
            for k in totals:
                totals[k] += c[k]
            rel = doc.relative_to(dest).as_posix()
            rows.append({"repo": entry["repo"], "sha": sha, "status": "CERTIFIED",
                         "document": rel, "receipts": [p.relative_to(dest).as_posix()
                                                       for p in receipts],
                         "tokens": len(cert["ledger"]), "counts": c, "verdict": cert["verdict"]})
            dl = doc.read_text(encoding="utf-8", errors="replace").splitlines()
            accusations.extend(summarise_accusation(cert, rel, entry["repo"], sha, dl))
            verifications.extend(summarise_verification(cert, rel, entry["repo"], sha))
            print(f"  [{i}/{len(repos)}] {entry['repo']}: {cert['verdict']}  "
                  f"V {c['VERIFIED']} / A {c['ABSTAIN']} / U {c['UNGROUNDED']}")
            shutil.rmtree(dest, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    certified = [r for r in rows if r["status"] == "CERTIFIED"]
    tok = sum(totals.values())
    payload = {
        "recon": "OATH-EXT — the shipped verifier on claim documents this lab did not write",
        "status": "RECON. LICENSES NO CLAIM. Sizes a class; its numbers are inputs to a future "
                  "preregistration's bars, not results.",
        "ground_truth_standard":
            "styxx does not adjudicate whether the external authors are right. The only thing "
            "measured is whether a number in a repository's own claim document grounds in that "
            "repository's own committed receipt. Inherited verbatim from SP-EXT.",
        "selection": {
            "queries": list(SEARCH_QUERIES),
            "rule": "first N DISTINCT repositories in the GitHub code-search API's own returned "
                    "order; no inspection, no substitution",
            "max_repos": a.max_repos,
            "disclosure": "GitHub's default ordering is 'best match', not random. This is a "
                          "CONVENIENCE SAMPLE of a mechanically-defined population and supports "
                          "no inference about base rates anywhere. Reported because a "
                          "mechanically-selected sample is auditable and a hand-picked one is not.",
        },
        "safety": "External repositories were cloned and READ. No external code was imported or "
                  "executed; this harness opens files and nothing else.",
        "verifier_sha256": cert_verifier_sha(),
        "repos_selected": len(repos),
        "repos_certified": len(certified),
        "repos_no_pair": sum(1 for r in rows if r["status"] == "NO_PAIR"),
        "repos_failed": sum(1 for r in rows if r["status"] in ("CLONE_FAILED", "CERTIFY_ERROR")),
        "documents_held": sum(1 for r in certified if r["verdict"] == "OATH-HELD"),
        "documents_failed": sum(1 for r in certified if r["verdict"] != "OATH-HELD"),
        "tokens_total": tok,
        "status_counts": totals,
        "status_share": {k: (round(v / tok, 4) if tok else None) for k, v in totals.items()},
        "accusations_total": len(accusations),
        "verifications_total": len(verifications),
        "verifications_structurally_coincident":
            sum(1 for v in verifications if v["structurally_coincident"]),
        "verification_roster": verifications,
        "accusation_roster": accusations,
        "per_repo": rows,
        "elapsed_s": round(time.time() - t0, 1),
        "what_this_does_not_show": [
            "Whether any accusation is CORRECT. Every UNGROUNDED token here is unadjudicated, "
            "and the v0.11 cycle exists precisely because four of this lab's own accusations "
            "turned out not to be claims at all. The roster is for hand adjudication.",
            "Any base rate. The sample is a convenience sample of one file-naming idiom.",
            "Whether OATH is deployable on external documents. That needs a frozen prereg with "
            "a false-accusation bar, and this RECON exists to tell that prereg what to freeze.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\ncertified {len(certified)}/{len(repos)} repos  {tok} tokens  "
          f"V {totals['VERIFIED']} / A {totals['ABSTAIN']} / U {totals['UNGROUNDED']}  "
          f"accusations {len(accusations)} -> {OUT.name}")
    return 0


def cert_verifier_sha() -> str:
    import hashlib
    return hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
