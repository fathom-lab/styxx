# -*- coding: utf-8 -*-
"""Run the 12 pre-registered agent-claim audits against the repo at HEAD.

See ../../papers/agent-self-audit/PRE_STATED_PREDICTION_agent_claim_audit_2026_05_28.md
for the pre-stated predictions. The kill-gate is >= 3 real failures.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from styxx.agent_audit import AgentClaimAuditor, Claim, checkers  # noqa: E402


CLAIMS = [
    Claim(
        id="C1",
        text="After commit 0e97598, pyproject.toml [project] version equals '7.7.10'",
        checker=checkers.package_version_equals,
        args={"path": "pyproject.toml", "version": "7.7.10"},
        expected=True,
    ),
    Claim(
        id="C2",
        text="After commit 0e97598, 'critique_detector' is in styxx.__all__",
        checker=checkers.python_attr_in_iterable,
        args={"module": "styxx", "attr": "critique_detector", "iterable": "__all__"},
        expected=True,
    ),
    Claim(
        id="C3",
        text="After commit 0e97598, 'CritiqueDetector' is in styxx.__all__",
        checker=checkers.python_attr_in_iterable,
        args={"module": "styxx", "attr": "CritiqueDetector", "iterable": "__all__"},
        expected=True,
    ),
    Claim(
        id="C4",
        text="styxx/critique.py module docstring contains 'out-of-context critique'",
        checker=checkers.file_at_path_contains,
        args={"path": "styxx/critique.py", "substring": "out-of-context critique"},
        expected=True,
    ),
    Claim(
        id="C5",
        text="styxx/critique.py module docstring does NOT contain v1 phrase 'Measured prevalence: 91.18%'",
        checker=checkers.file_at_path_contains,
        args={"path": "styxx/critique.py", "substring": "Measured prevalence: 91.18%"},
        expected=False,  # claim of absence
    ),
    Claim(
        id="C6a",
        text="Commit 0e97598 changes pyproject.toml version from 7.7.9 (removes that line)",
        checker=checkers.git_show_diff_contains,
        args={"commit": "0e97598", "file": "pyproject.toml", "substring": '-version = "7.7.9"'},
        expected=True,
    ),
    Claim(
        id="C6b",
        text="Commit 0e97598 changes pyproject.toml version to 7.7.10 (adds that line)",
        checker=checkers.git_show_diff_contains,
        args={"commit": "0e97598", "file": "pyproject.toml", "substring": '+version = "7.7.10"'},
        expected=True,
    ),
    Claim(
        id="C7",
        text="Commits ed663ca, c75cab4, 0e97598, 87ca52d are all on origin/main in that order",
        checker=checkers.git_branch_contains_commit_chain,
        args={"branch": "origin/main", "commits": ["ed663ca", "c75cab4", "0e97598", "87ca52d"]},
        expected=True,
    ),
    Claim(
        id="C8",
        text="arxiv/recursive_discipline/main.pdf has exactly 14 pages",
        checker=checkers.pdf_page_count_equals,
        args={"path": "arxiv/recursive_discipline/main.pdf", "n": 14},
        expected=True,
    ),
    Claim(
        id="C9",
        text="arxiv/recursive_discipline/main.pdf contains the section '13. The paper catches itself'",
        checker=checkers.pdf_contains_section,
        args={"path": "arxiv/recursive_discipline/main.pdf", "section_title": "13. The paper catches itself"},
        expected=True,
    ),
    Claim(
        id="C10",
        text="git tag v7.7.10 does NOT yet exist (operator-territory pending)",
        checker=checkers.git_tag_exists,
        args={"tag": "v7.7.10"},
        expected=False,  # claim of absence
    ),
    Claim(
        id="C11",
        text="Paper v5 contains 'sixteen in-session falsifications' (no longer 'eight')",
        checker=checkers.file_at_path_contains,
        args={"path": "papers/PAPER_recursive_discipline_2026_05_27.md", "substring": "sixteen in-session falsifications"},
        expected=True,
    ),
    Claim(
        id="C12",
        text="papers/PAPER...md and arxiv/recursive_discipline/source.md are byte-identical (no drift)",
        checker=checkers.file_byte_equals,
        args={
            "path_a": "papers/PAPER_recursive_discipline_2026_05_27.md",
            "path_b": "arxiv/recursive_discipline/source.md",
        },
        expected=True,
    ),
]


def main() -> int:
    auditor = AgentClaimAuditor(repo_path=REPO)
    results = auditor.run(CLAIMS)
    out_path = Path(__file__).parent / "results.json"
    summary = {
        "n_claims": len(results),
        "n_pass": sum(1 for r in results if r.verdict == "PASS"),
        "n_fail": sum(1 for r in results if r.verdict == "FAIL"),
        "n_error": sum(1 for r in results if r.verdict == "ERROR"),
        "verdicts_by_id": {r.id: r.verdict for r in results},
        "results": [r.to_dict() for r in results],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"=== agent_claim_audit_2026_05_28 ===")
    print(f"claims: {summary['n_claims']}")
    print(f"PASS:  {summary['n_pass']}")
    print(f"FAIL:  {summary['n_fail']}")
    print(f"ERROR: {summary['n_error']}")
    print()
    for r in results:
        marker = {"PASS": "[OK ]", "FAIL": "[XX ]", "ERROR": "[ERR]"}[r.verdict]
        evidence_line = (r.evidence or r.error).split("\n")[0][:140]
        print(f"  {marker} {r.id}: {r.text[:80]}")
        print(f"         -> {evidence_line}")
    print()
    print(f"results written to {out_path}")

    # exit non-zero only if killgate fires (>= 3 real FAILs, excluding ERRORs)
    if summary["n_fail"] >= 3:
        print("KILLGATE: >=3 FAILs — Layer 5 self-falsification of session outputs")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
