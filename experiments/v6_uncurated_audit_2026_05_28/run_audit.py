# -*- coding: utf-8 -*-
"""Run the pre-registered L7 uncurated audit of v6.

Pre-registration:
  papers/agent-self-audit/PRE_STATED_PREDICTION_v6_uncurated_audit_2026_05_28.md
  (commit b18ce93, public on origin/main BEFORE this runner exists)
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from styxx.agent_audit import AgentClaimAuditor, Claim, checkers  # noqa: E402


PAPER = "papers/PAPER_recursive_discipline_2026_05_27.md"
PDF = "arxiv/recursive_discipline/main.pdf"
L5_RESULTS = "experiments/agent_claim_audit_2026_05_28/results.json"
L6_RESULTS = "experiments/critique_detector_on_paper_2026_05_28/results.json"

CLAIMS = [
    # --- Numerical counts (the suspected-FAIL zone) ---
    Claim("N1", "v6 abstract contains 'ten FINDING documents'",
          checkers.file_at_path_contains,
          {"path": PAPER, "substring": "ten FINDING documents"}, expected=True),
    Claim("N2", "actual FINDING_*.md count equals 10 (predicted FAIL — actual 13)",
          checkers.directory_file_count_equals,
          {"glob": "papers/agent-self-audit/FINDING_*.md", "n": 10}, expected=True),
    Claim("N3", "v6 abstract contains 'nineteen reference baselines'",
          checkers.file_at_path_contains,
          {"path": PAPER, "substring": "nineteen reference baselines"}, expected=True),
    Claim("N4", "actual submissions/baseline_*/ count equals 19 (predicted FAIL — actual 18)",
          checkers.directory_file_count_equals,
          {"glob": "submissions/baseline_*", "n": 19}, expected=True),
    Claim("N5", "actual submissions/baseline_*/ count equals 18 (ground-truth check)",
          checkers.directory_file_count_equals,
          {"glob": "submissions/baseline_*", "n": 18}, expected=True),
    Claim("N6", "v6 abstract contains '108-record benchmark'",
          checkers.file_at_path_contains,
          {"path": PAPER, "substring": "108-record benchmark"}, expected=True),
    Claim("N7", "darkcore_benchmark records list length is 108",
          checkers.json_path_equals,
          {"path": "papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json",
           "key_path": "n_records", "expected": 108}, expected=True),
    Claim("N8", "v6 §13 contains '1086 passed, 8 skipped' verbatim",
          checkers.file_at_path_contains,
          {"path": PAPER, "substring": "1086 passed, 8 skipped"}, expected=True),
    Claim("N9", "v6 paper contains '42/42'",
          checkers.file_at_path_contains,
          {"path": PAPER, "substring": "42/42"}, expected=True),

    # --- Per-claim numerical facts (asymmetry table) ---
    Claim("N10", "v6 contains '5.88%' (dark-core asymmetry rate)",
          checkers.file_at_path_contains,
          {"path": PAPER, "substring": "5.88%"}, expected=True),
    Claim("N11", "v6 contains '17.00%' (TruthfulQA asymmetry rate)",
          checkers.file_at_path_contains,
          {"path": PAPER, "substring": "17.00%"}, expected=True),
    Claim("N12a", "L5 results.json n_pass equals 13",
          checkers.json_path_equals,
          {"path": L5_RESULTS, "key_path": "n_pass", "expected": 13}, expected=True),
    Claim("N12b", "L5 results.json n_fail equals 0",
          checkers.json_path_equals,
          {"path": L5_RESULTS, "key_path": "n_fail", "expected": 0}, expected=True),
    Claim("N13a", "L6 results.json n_pass equals 18",
          checkers.json_path_equals,
          {"path": L6_RESULTS, "key_path": "n_pass", "expected": 18}, expected=True),
    Claim("N13b", "L6 results.json n_fail equals 0",
          checkers.json_path_equals,
          {"path": L6_RESULTS, "key_path": "n_fail", "expected": 0}, expected=True),
    Claim("N14", "L6 results.json killgate_paper_grade_fired is False",
          checkers.json_path_equals,
          {"path": L6_RESULTS, "key_path": "killgate_paper_grade_fired", "expected": False},
          expected=True),
    Claim("N15", "L6 results.json killgate_instrument_grade_fired is False",
          checkers.json_path_equals,
          {"path": L6_RESULTS, "key_path": "killgate_instrument_grade_fired", "expected": False},
          expected=True),
    Claim("N16", "L6 results.json: TRUE claim T1 observed_p_no == 0.0",
          checkers.json_path_equals,
          {"path": L6_RESULTS, "key_path": "results.[0].observed_p_no", "expected": 0.0},
          expected=True),
    Claim("N17", "L6 results.json: FALSE control F1 observed_p_no == 1.0",
          checkers.json_path_equals,
          {"path": L6_RESULTS, "key_path": "results.[13].observed_p_no", "expected": 1.0},
          expected=True),

    # --- Commit-reference checks ---
    Claim("C1", "commit ed663ca exists on origin/main",
          checkers.git_branch_contains_commit_chain,
          {"branch": "origin/main", "commits": ["ed663ca"]}, expected=True),
    Claim("C2", "commit 0e97598 exists on origin/main",
          checkers.git_branch_contains_commit_chain,
          {"branch": "origin/main", "commits": ["0e97598"]}, expected=True),
    Claim("C3", "commit 87ca52d exists on origin/main",
          checkers.git_branch_contains_commit_chain,
          {"branch": "origin/main", "commits": ["87ca52d"]}, expected=True),
    Claim("C4", "commit 1b82e27 exists on origin/main",
          checkers.git_branch_contains_commit_chain,
          {"branch": "origin/main", "commits": ["1b82e27"]}, expected=True),
    Claim("C5", "commit 3c24b5e exists on origin/main",
          checkers.git_branch_contains_commit_chain,
          {"branch": "origin/main", "commits": ["3c24b5e"]}, expected=True),
    Claim("C6", "commit a8fb1f3 exists on origin/main",
          checkers.git_branch_contains_commit_chain,
          {"branch": "origin/main", "commits": ["a8fb1f3"]}, expected=True),
    Claim("C7", "commit 05adebf exists on origin/main",
          checkers.git_branch_contains_commit_chain,
          {"branch": "origin/main", "commits": ["05adebf"]}, expected=True),
    Claim("C8", "commit 1ab0e22 (critique_detector public API) exists on origin/main",
          checkers.git_branch_contains_commit_chain,
          {"branch": "origin/main", "commits": ["1ab0e22"]}, expected=True),

    # --- Structural / API-surface checks ---
    Claim("S1", "styxx.__version__ equals '7.7.10'",
          checkers.python_attr_equals,
          {"module": "styxx", "attr": "__version__", "expected": "7.7.10"}, expected=True),
    Claim("S2", "'critique_detector' in styxx.__all__",
          checkers.python_attr_in_iterable,
          {"module": "styxx", "attr": "critique_detector", "iterable": "__all__"}, expected=True),
    Claim("S3", "'agent_audit' in styxx.__all__",
          checkers.python_attr_in_iterable,
          {"module": "styxx", "attr": "agent_audit", "iterable": "__all__"}, expected=True),
    Claim("S4", "'AgentClaimAuditor' in styxx.__all__",
          checkers.python_attr_in_iterable,
          {"module": "styxx", "attr": "AgentClaimAuditor", "iterable": "__all__"}, expected=True),
    Claim("S5", "git tag v7.7.10 does NOT exist (operator-territory)",
          checkers.git_tag_exists,
          {"tag": "v7.7.10"}, expected=False),

    # --- Section-position checks against rendered PDF ---
    Claim("P1", "v6 PDF main.pdf has 17 pages",
          checkers.pdf_page_count_equals,
          {"path": PDF, "n": 17}, expected=True),
    Claim("P2", "v6 PDF contains '13. The paper catches itself' section header",
          checkers.pdf_contains_section,
          {"path": PDF, "section_title": "13. The paper catches itself"}, expected=True),
    Claim("P3", "v6 PDF contains '14. The instrumented recursion frame' section header",
          checkers.pdf_contains_section,
          {"path": PDF, "section_title": "14. The instrumented recursion frame"}, expected=True),
]


def main() -> int:
    auditor = AgentClaimAuditor(repo_path=REPO)
    results = auditor.run(CLAIMS)
    out_path = Path(__file__).parent / "results.json"

    n_pass = sum(1 for r in results if r.verdict == "PASS")
    n_fail = sum(1 for r in results if r.verdict == "FAIL")
    n_error = sum(1 for r in results if r.verdict == "ERROR")

    summary = {
        "n_claims": len(results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_error": n_error,
        "fail_ids": [r.id for r in results if r.verdict == "FAIL"],
        "error_ids": [r.id for r in results if r.verdict == "ERROR"],
        "results": [r.to_dict() for r in results],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== L7 v6_uncurated_audit_2026_05_28 ===")
    print(f"claims: {len(results)}")
    print(f"PASS:  {n_pass}")
    print(f"FAIL:  {n_fail}  -> {summary['fail_ids']}")
    print(f"ERROR: {n_error} -> {summary['error_ids']}")
    print()
    for r in results:
        marker = {"PASS": "[OK ]", "FAIL": "[XX ]", "ERROR": "[ERR]"}[r.verdict]
        evidence_first = (r.evidence or r.error).split("\n")[0][:140]
        print(f"  {marker} {r.id}: {r.text[:80]}")
        print(f"         -> {evidence_first}")

    print()
    if n_fail == 2 and set(summary["fail_ids"]) == {"N2", "N4"}:
        print("RESULT: Exactly the 2 pre-disclosed FAILs (N2, N4). Manual eyeballing matched instrument. Predicted modal outcome.")
    elif n_fail < 2:
        print(f"RESULT: {n_fail} FAILs — fewer than the 2 pre-disclosed. Manual eyeballing was WRONG on at least one; instrument > manual.")
    elif n_fail > 2:
        unexpected = [i for i in summary["fail_ids"] if i not in ("N2", "N4")]
        print(f"RESULT: {n_fail} FAILs — INSTRUMENT CAUGHT DRIFT BEYOND MANUAL REVIEW. Unexpected FAILs: {unexpected}")
    print(f"results: {out_path}")

    return 1 if (n_fail + n_error) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
