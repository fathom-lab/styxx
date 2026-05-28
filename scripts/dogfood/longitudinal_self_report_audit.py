# -*- coding: utf-8 -*-
"""Longitudinal self-report audit — fact-check an AI agent's authored history.

Runs styxx.agent_audit over every AI-co-authored commit in this repo, checking
each extracted claim against the substrate AS IT EXISTED AT THAT COMMIT
(`git archive <sha>`), not against HEAD. Pre-registered at
``longitudinal_self_report_PREREG.md`` BEFORE this runner existed.

Read-only. Materializes each commit's tree into a throwaway temp dir; mutates
nothing in the repo. Run from anywhere:

    python scripts/dogfood/longitudinal_self_report_audit.py [--repo PATH] [--json]
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# Import styxx from the repo under audit (editable install or source tree).
REPO_DEFAULT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DEFAULT))

from styxx.agent_audit import AgentClaimAuditor, checkers, extract_claims  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, check=False,
    )
    return r.stdout.decode("utf-8", errors="replace")


def ai_commits(repo: Path) -> list[str]:
    out = _git(repo, "log", "--grep=Co-Authored-By", "--format=%H")
    return [h for h in out.splitlines() if h.strip()]


def commit_message(repo: Path, sha: str) -> str:
    return _git(repo, "log", "-1", "--format=%B", sha)


def materialize_tree(repo: Path, sha: str, dest: Path) -> bool:
    """Extract the tree at ``sha`` into ``dest``. Returns False on failure."""
    r = subprocess.run(
        ["git", "archive", "--format=tar", sha],
        cwd=str(repo), capture_output=True, check=False,
    )
    if r.returncode != 0 or not r.stdout:
        return False
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        tf.extractall(dest, filter="data")
    return True


def is_tag_claim(claim) -> bool:
    return getattr(claim.checker, "__name__", "") == "git_tag_exists"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap commits (debug)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    shas = ai_commits(repo)
    if args.limit:
        shas = shas[: args.limit]

    live_auditor = AgentClaimAuditor(repo)  # HEAD / working tree + .git for tags

    total_commits = len(shas)
    commits_with_claims = 0
    claims_total = 0
    passed = failed = errored = 0
    temporal_divergences: list[dict] = []
    contradictions: list[dict] = []
    error_examples: list[dict] = []

    for sha in shas:
        msg = commit_message(repo, sha)
        rep = extract_claims(msg, id_prefix="C")
        if not rep.claims:
            continue
        commits_with_claims += 1

        # Materialize the contemporaneous tree only if a non-tag claim needs it.
        needs_tree = any(not is_tag_claim(c) for c in rep.claims)
        tmp = Path(tempfile.mkdtemp(prefix="styxx_hist_"))
        hist_repo = tmp
        tree_ok = True
        try:
            if needs_tree:
                tree_ok = materialize_tree(repo, sha, tmp)
            hist_auditor = AgentClaimAuditor(hist_repo) if tree_ok else None

            for c in rep.claims:
                claims_total += 1
                # tag claims -> live repo; file/version/pdf -> tree-at-commit.
                if is_tag_claim(c) or not tree_ok:
                    (hist_res,) = live_auditor.run([c])
                else:
                    (hist_res,) = hist_auditor.run([c])
                (head_res,) = live_auditor.run([c])

                v = hist_res.verdict
                if v == "PASS":
                    passed += 1
                elif v == "FAIL":
                    failed += 1
                    contradictions.append({
                        "sha": sha[:10], "claim": c.text[:120],
                        "evidence": hist_res.evidence[:200],
                    })
                else:
                    errored += 1
                    if len(error_examples) < 15:
                        error_examples.append({
                            "sha": sha[:10], "claim": c.text[:120],
                            "error": hist_res.error[:160],
                        })

                # P3: temporal distinctness — verdict differs vs HEAD.
                if (not is_tag_claim(c)) and tree_ok and hist_res.verdict != head_res.verdict:
                    temporal_divergences.append({
                        "sha": sha[:10], "claim": c.text[:120],
                        "at_commit": hist_res.verdict, "at_head": head_res.verdict,
                        "commit_evidence": hist_res.evidence[:160],
                        "head_evidence": head_res.evidence[:160],
                    })
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    coverage = commits_with_claims / total_commits if total_commits else 0.0
    contra_rate = (
        len({c["sha"] for c in contradictions}) / commits_with_claims
        if commits_with_claims else 0.0
    )

    summary = {
        "corpus_total_commits_ai": total_commits,
        "commits_with_extractable_claims": commits_with_claims,
        "coverage": round(coverage, 4),
        "claims_total": claims_total,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "commit_contradiction_rate": round(contra_rate, 4),
        "temporal_divergence_count": len(temporal_divergences),
        "error_rate": round(errored / claims_total, 4) if claims_total else 0.0,
    }

    # Kill-gate evaluation (pre-registered).
    k1 = coverage < 0.005
    k2 = (len(temporal_divergences) == 0) and (failed == 0)
    summary["kill_gate"] = {
        "K1_no_coverage": k1,
        "K2_no_divergence_and_no_contradiction": k2,
        "thesis_killed": bool(k1 or k2),
    }

    payload = {
        "longitudinal_self_report_audit": summary,
        "contradictions": contradictions[:25],
        "temporal_divergences": temporal_divergences[:25],
        "error_examples": error_examples,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    s = summary
    print("=== Longitudinal self-report audit ===")
    print(f"AI-co-authored commits ............ {s['corpus_total_commits_ai']}")
    print(f"commits w/ extractable claims ..... {s['commits_with_extractable_claims']} "
          f"(coverage {s['coverage']:.1%})")
    print(f"claims audited .................... {s['claims_total']}")
    print(f"  PASS ............................ {s['passed']}")
    print(f"  FAIL (contradiction) ............ {s['failed']}")
    print(f"  ERROR ........................... {s['errored']} "
          f"(error rate {s['error_rate']:.1%})")
    print(f"commit contradiction rate ......... {s['commit_contradiction_rate']:.1%}")
    print(f"temporal divergences (commit≠HEAD)  {s['temporal_divergence_count']}")
    print()
    kg = s["kill_gate"]
    print(f"KILL-GATE: K1(no-coverage)={kg['K1_no_coverage']} "
          f"K2(no-divergence&no-contradiction)={kg['K2_no_divergence_and_no_contradiction']} "
          f"-> thesis_killed={kg['thesis_killed']}")
    if contradictions:
        print("\n--- genuine contradictions (claim FAILs against its own commit) ---")
        for c in contradictions[:10]:
            print(f"  [{c['sha']}] {c['claim']}\n     {c['evidence']}")
    if temporal_divergences:
        print("\n--- temporal divergences (verdict at commit != at HEAD) ---")
        for d in temporal_divergences[:10]:
            print(f"  [{d['sha']}] {d['claim']}")
            print(f"     at_commit={d['at_commit']}  at_head={d['at_head']}")
    if error_examples:
        print("\n--- error examples ---")
        for e in error_examples[:8]:
            print(f"  [{e['sha']}] {e['claim']}\n     {e['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
