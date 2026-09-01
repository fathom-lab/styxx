"""Our own FLAG RATE on the AIDev corpus, split DEVELOPMENT / HELD-OUT.

How often does this instrument open its mouth? Precision has been measured three
times (EXTERNAL-1 0.23, V14 held-out 0.16). What has never been reported next to
those figures is the base rate: on what fraction of pull requests does the gate
actually raise an accusation. A low-precision accuser that almost never fires is
a different object from a low-precision accuser that fires constantly, and the
two have been conflated in every write-up so far.

Method, deliberately parasitic on the published measurement so the populations
match exactly:

  * the diff reconstructor is `external1_harness.reconstruct` — imported, never
    reimplemented — and the same `parsed != implied` round-trip skip is applied,
    so the denominator is the same 71,016 eligible pull requests `v14_gates.py`
    reports.
  * the DEVELOPMENT / HELD-OUT split is `v14_gates.bucket` on the first five
    URL segments, `< 3` DEVELOPMENT, imported from the same module.
  * `DG.WITHHOLD_PATH_ACCUSATION = False`, so the path branch is measured AS IT
    WOULD ACCUSE rather than in its shipped, disabled state. The V13 and V14
    repairs are left ON at their shipped defaults: this is the flag rate of the
    repaired accuser, the one whose held-out precision is 0.16.
  * accusation identity is the PATH, never (kind, path). The repairs deliberately
    rewrite the kind, and keying on it mis-reports a demotion as a new
    accusation — the defect that made V13's first G-R1 measurement wrong.

`tests_pass` is reported alongside because it is what the evidence leg targets.
With no `--run` command the gate does not take the agent's word for test results
and every such claim is UNCHECKABLE, so its accusation count is expected to be
zero by construction. That zero is the measurement, not a bug: it is the size of
the hole the evidence leg exists to fill.

  python papers/closed-model-frontier/flag_rate.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import styxx.diffgate as DG                                    # noqa: E402
from external1_harness import reconstruct                      # noqa: E402
from v14_gates import bucket                                   # noqa: E402

DB = HERE / "external1_shelf.sqlite"
OUT = HERE / "flag_rate.json"

EXPECTED_ELIGIBLE = 71016      # v14_gates.json, prs_scored


def blank() -> dict:
    return {
        "prs_scored": 0,
        "prs_with_path_claim_any_verdict": 0,
        "prs_with_path_accusation": 0,
        "path_accusations": 0,              # distinct accused PATHS, summed over PRs
        "prs_with_tests_pass_claim": 0,
        "prs_with_tests_pass_accusation": 0,
        "tests_pass_claims": 0,
        "tests_pass_by_verdict": {},
    }


def rate(num: int, den: int):
    return round(num / den, 6) if den else None


def main() -> int:
    # Measure the branch as it would accuse. The repairs stay ON (shipped
    # defaults): this is the flag rate of the accuser that scored 0.16.
    DG.WITHHOLD_PATH_ACCUSATION = False

    con = sqlite3.connect(DB)
    splits = {"development": blank(), "held_out": blank()}
    skipped_roundtrip = 0
    gate_errors = 0

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
            skipped_roundtrip += 1
            continue

        summary = f"{title or ''}\n\n{body}"
        try:
            g = DG.gate_diff_text(summary, diff, run=None, strict=False)
        except Exception:
            # never silently pass: counted, and excluded from every numerator
            # and denominator alike
            gate_errors += 1
            continue

        key = ("development"
               if bucket("/".join((url or "").split("/")[:5])) < 3
               else "held_out")
        s = splits[key]
        s["prs_scored"] += 1

        path_claims = [c for c in g.claims if c.kind.startswith("file_")]
        # PATH identity, never (kind, path)
        accused = {(c.detail or {}).get("path") for c in path_claims
                   if c.verdict == "CONTRADICTED"}
        if path_claims:
            s["prs_with_path_claim_any_verdict"] += 1
        if accused:
            s["prs_with_path_accusation"] += 1
            s["path_accusations"] += len(accused)

        tp = [c for c in g.claims if c.kind == "tests_pass"]
        if tp:
            s["prs_with_tests_pass_claim"] += 1
            s["tests_pass_claims"] += len(tp)
            for c in tp:
                s["tests_pass_by_verdict"][c.verdict] = \
                    s["tests_pass_by_verdict"].get(c.verdict, 0) + 1
            if any(c.verdict == "CONTRADICTED" for c in tp):
                s["prs_with_tests_pass_accusation"] += 1

    con.close()

    for s in splits.values():
        n = s["prs_scored"]
        s["path_flag_rate"] = rate(s["prs_with_path_accusation"], n)
        s["path_flag_rate_of_claiming_prs"] = rate(
            s["prs_with_path_accusation"], s["prs_with_path_claim_any_verdict"])
        s["tests_pass_flag_rate"] = rate(s["prs_with_tests_pass_accusation"], n)
        s["tests_pass_claim_rate"] = rate(s["prs_with_tests_pass_claim"], n)

    total = sum(s["prs_scored"] for s in splits.values())
    both = blank()
    for s in splits.values():
        for k, v in s.items():
            if isinstance(v, int):
                both[k] += v
            elif isinstance(v, dict):
                for vk, vv in v.items():
                    both[k][vk] = both[k].get(vk, 0) + vv
    both["path_flag_rate"] = rate(both["prs_with_path_accusation"], total)
    both["path_flag_rate_of_claiming_prs"] = rate(
        both["prs_with_path_accusation"], both["prs_with_path_claim_any_verdict"])
    both["tests_pass_flag_rate"] = rate(both["prs_with_tests_pass_accusation"], total)
    both["tests_pass_claim_rate"] = rate(both["prs_with_tests_pass_claim"], total)

    payload = {
        "what": "flag rate of the repaired path accuser on AIDev, by split",
        "instrument": {
            "WITHHOLD_PATH_ACCUSATION": False,
            "V14_CONTAINMENT_TOUCH": DG.V14_CONTAINMENT_TOUCH,
            "V14_BARE_NAME_ABSTAIN": DG.V14_BARE_NAME_ABSTAIN,
            "run": None,
            "strict": False,
        },
        "identity": "accusation keyed on PATH, never (kind, path)",
        "population": {
            "expected_eligible": EXPECTED_ELIGIBLE,
            "observed_eligible": total,
            "matches_v14_gates": total == EXPECTED_ELIGIBLE,
            "skipped_reconstruction_roundtrip": skipped_roundtrip,
            "gate_errors": gate_errors,
        },
        "development": splits["development"],
        "held_out": splits["held_out"],
        "corpus_wide": both,
        "tests_pass_note": (
            "with run=None the gate does not take the agent's word for test "
            "results, so every tests_pass claim is UNCHECKABLE and the "
            "accusation count is zero by construction — the hole the evidence "
            "leg exists to fill, measured"),
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    def show(name: str, s: dict) -> None:
        print(f"\n{name}")
        print(f"  PRs scored                          {s['prs_scored']}")
        print(f"  PRs with >=1 path claim (any verdict){s['prs_with_path_claim_any_verdict']:>7}")
        print(f"  PRs with >=1 path ACCUSATION        {s['prs_with_path_accusation']}")
        print(f"  distinct accused paths              {s['path_accusations']}")
        print(f"  FLAG RATE (accused / scored)        {s['prs_with_path_accusation']}/"
              f"{s['prs_scored']} = {s['path_flag_rate']}")
        if s.get("path_flag_rate_of_claiming_prs") is not None:
            print(f"  of PRs that make a path claim       "
                  f"{s['path_flag_rate_of_claiming_prs']}")
        print(f"  PRs with >=1 tests_pass claim       {s['prs_with_tests_pass_claim']}")
        print(f"  PRs with >=1 tests_pass ACCUSATION  {s['prs_with_tests_pass_accusation']}")
        print(f"  tests_pass FLAG RATE                "
              f"{s['prs_with_tests_pass_accusation']}/{s['prs_scored']} = "
              f"{s['tests_pass_flag_rate']}")
        print(f"  tests_pass verdicts                 {s['tests_pass_by_verdict']}")

    print(f"eligible: {total} (v14_gates reports {EXPECTED_ELIGIBLE}) "
          f"{'MATCH' if total == EXPECTED_ELIGIBLE else 'MISMATCH — investigate'}")
    print(f"round-trip skips: {skipped_roundtrip}   gate errors: {gate_errors}")
    show("DEVELOPMENT", splits["development"])
    show("HELD-OUT", splits["held_out"])
    show("CORPUS-WIDE", both)
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
