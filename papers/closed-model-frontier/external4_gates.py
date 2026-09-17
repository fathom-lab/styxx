"""EXTERNAL-4 gates: COMPAT-1 measured against its preregistration (PREREG_compat1_2026_09_16).

Reads the BC-2 ledger (`external3_ledger.jsonl`, the checkout at the COMPAT-1 prereg commit,
without the compat reading) and the COMPAT-1 ledger (`external4_ledger.jsonl`, the same checkout
with it), same corpus, same reconstruction, same exclusions, and scores:

  G-C1  never accuses: zero compat_claim claims with a verdict other than UNCHECKABLE.
  G-C2  every other kind untouched: the set of (pr_id, kind, text, verdict) for all
        non-compat_claim claims is identical between the two ledgers.
  G-C3  the census reproduced: compat_claim count against the exploratory 8,467 (±2%); claims
        whose reason names at least one removed public definition, per language, with the
        dropped-name histogram, reported beside the exploratory 531.

Counts only; no PR named.  Writes external4_gates.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEFORE = HERE / "external3_ledger.jsonl"
AFTER = HERE / "external4_ledger.jsonl"
EXPLORATORY = HERE / "exploratory_ml_compat.json"
OUT = HERE / "external4_gates.json"


def _load(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out[r["pr_id"]] = r
    return out


def main() -> int:
    before, after = _load(BEFORE), _load(AFTER)
    assert set(before) == set(after), "the two ledgers cover different PRs"
    other_b = {(pid, c["kind"], c["text"], c["verdict"]) for pid, r in before.items() for c in r["claims"]}
    other_a = {(pid, c["kind"], c["text"], c["verdict"]) for pid, r in after.items() for c in r["claims"]
               if c["kind"] != "compat_claim"}
    verdicts = Counter(); prs_with_claim = 0; with_removed = 0
    per_lang = Counter(); hist = Counter(); langs_seen = Counter(); no_lang = 0; nothing_removed = 0
    for pid, r in after.items():
        cs = [c for c in r["claims"] if c["kind"] == "compat_claim"]
        if not cs:
            continue
        prs_with_claim += 1
        for c in cs:
            verdicts[c["verdict"]] += 1
        c = cs[0]                                   # one reading per PR: the diff is the same
        removed = c["detail"].get("removed", [])
        langs = c["detail"].get("languages", [])
        if not langs:
            no_lang += 1
        elif not removed:
            nothing_removed += 1
        else:
            with_removed += 1
            hist[min(len(removed), 10)] += 1
            for lang in {x["language"] for x in removed}:
                per_lang[lang] += 1
        for l in langs:
            langs_seen[l] += 1
    exp = json.loads(EXPLORATORY.read_text(encoding="utf-8"))
    exp_claims = exp["compat"]["claims"]; exp_removed = exp["compat"]["claims_with_a_dropped_public_name"]
    n_claims = sum(verdicts.values())
    gates = {
        "G-C1_never_accuses": {"verdicts": dict(verdicts),
                               "pass": set(verdicts) <= {"UNCHECKABLE"} and n_claims > 0},
        "G-C2_every_other_kind_untouched": {"before": len(other_b), "after_non_compat": len(other_a),
                                            "differences": len(other_b ^ other_a),
                                            "pass": other_b == other_a},
        "G-C3_census_reproduced": {"prs_with_a_compat_claim": prs_with_claim, "compat_claims": n_claims,
                                   "exploratory_claims": exp_claims,
                                   "within_2pct": abs(prs_with_claim - exp_claims) <= 0.02 * exp_claims,
                                   "prs_with_a_removed_public_definition": with_removed,
                                   "exploratory_with_dropped": exp_removed,
                                   "prs_nothing_removed": nothing_removed, "prs_no_covered_language": no_lang,
                                   "removed_by_language": dict(per_lang), "removed_count_histogram": dict(hist),
                                   "languages_read": dict(langs_seen), "pass": "reported, not scored"},
    }
    cov_b = sum(1 for r in before.values() if r["claims"]); cov_a = sum(1 for r in after.values() if r["claims"])
    payload = {"prereg": "PREREG_compat1_2026_09_16.md",
               "before": {"ledger": BEFORE.name, "instrument": "BC-2 checkout (patch-15), no compat reading",
                          "covered": cov_b},
               "after": {"ledger": AFTER.name, "instrument": "COMPAT-1 checkout", "covered": cov_a},
               "gates": gates,
               "all_blocking_gates_pass": gates["G-C1_never_accuses"]["pass"] and gates["G-C2_every_other_kind_untouched"]["pass"]}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0 if payload["all_blocking_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
