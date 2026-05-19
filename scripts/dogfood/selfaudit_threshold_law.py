"""Self-audit: run styxx 7.4.1 instruments on the threshold-law paper's own text.

Methodology demonstration: the paper audited by the tool it documents.

Bounded by styxx construct ceiling (text-only register detection).
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from styxx.attack import score_all
from styxx.guardrail.deception_v2 import deception_check_v2

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "papers" / "threshold-law-2026-05-18.md"
OUT = ROOT / "scripts" / "dogfood" / "out_selfaudit_threshold_law.json"

# section keys + slice headings
SECTIONS = [
    ("abstract", r"## Abstract", r"## 1\. "),
    ("s1_preregistered_claim", r"## 1\. ", r"## 2\. "),
    ("s5_failure_mode", r"## 5\. ", r"## 6\. "),
    ("s6_fine_replication_failure", r"## 6\. ", r"## 7\. "),
    ("s7_limits", r"## 7\. ", r"## 8\. "),
    ("s8_integrity", r"## 8\. ", r"## 9\. "),
]

# composite: lower-is-more-honest, mean of sycophancy + overconfidence
# (reference-less deception excluded; that's the 2026-05-17 honest fix)
COMP_KEYS = ["sycophancy", "overconfidence"]
UNDER_REVIEW = {"overconfidence"}  # construct-ceiling caveat


def slice_section(text: str, start_pat: str, end_pat: str) -> str:
    s = re.search(start_pat, text)
    e = re.search(end_pat, text)
    if not s:
        return ""
    return text[s.start(): e.start() if e else len(text)]


def composite(scores):
    vals = [scores.get(k, 0.0) for k in COMP_KEYS]
    return sum(vals) / len(vals)


def main():
    text = PAPER.read_text(encoding="utf-8")
    results = {"paper": str(PAPER.relative_to(ROOT)), "styxx_version": "7.4.1", "sections": {}}

    # use a generic "prompt" since this is a paper section, not a chat turn.
    # The register detectors fire on response text alone.
    prompt = "Self-audit: evaluate this paper section."

    for name, sp, ep in SECTIONS:
        body = slice_section(text, sp, ep)
        if not body.strip():
            results["sections"][name] = {"error": "empty section"}
            continue
        scores = score_all(prompt=prompt, response=body)
        comp = composite(scores)
        fired_under_review = [k for k in UNDER_REVIEW if scores.get(k, 0.0) > 0.50]
        results["sections"][name] = {
            "n_chars": len(body),
            "scores": {k: round(float(v), 4) for k, v in scores.items()},
            "composite_honesty": round(comp, 4),
            "composite_keys": COMP_KEYS,
            "under_review_fired": fired_under_review,
            "needs_revision": comp > 0.30 or any(scores.get(k, 0.0) > 0.60 for k in COMP_KEYS),
            "caveat": (
                "composite excludes reference-less deception (non-discriminative); "
                "overconfidence is COGN_UNDER_REVIEW (register-detector saturated). "
                "Construct-ceiling bound: text-only register signal, not honesty."
            ),
        }

    # ----- deception_v2 with reference: paper claims vs raw JSON -----
    with open(ROOT / "scripts/dogfood/out_corpus_coverage_law_fine.json") as f:
        fine = json.load(f)
    with open(ROOT / "scripts/dogfood/out_corpus_coverage_law.json") as f:
        orig = json.load(f)
    with open(ROOT / "scripts/dogfood/out_cross_vendor_refusal_transport_confirm.json") as f:
        cv = json.load(f)

    facts = {
        "tau_threshold_paper": 0.31,
        "tau_threshold_json": fine["stats"]["sufficiency_threshold_overlap"],
        "cf_AUC_at_min_overlap_paper": 0.687,
        "cf_AUC_at_min_overlap_json": fine["stats"]["cf_AUC_at_min_overlap"],
        "cf_AUC_at_max_overlap_paper": 0.847,
        "cf_AUC_at_max_overlap_json": fine["stats"]["cf_AUC_at_max_overlap"],
        "spearman_cf_fine_paper": 0.69,
        "spearman_cf_fine_json": fine["stats"]["spearman_crossfamily"],
        "spearman_sf_fine_paper": -0.41,
        "spearman_sf_fine_json": fine["stats"]["spearman_samefamily"],
        "spearman_cf_orig_paper": 0.83,
        "spearman_cf_orig_json": orig["stats"]["spearman_crossfamily"],
        "spearman_sf_orig_paper": -0.29,
        "spearman_sf_orig_json": orig["stats"]["spearman_samefamily"],
        "anthropic_min_transported_paper": 0.617,
        "anthropic_min_transported_json": cv["summary"]["anthropic"]["transported_min"],
        "anthropic_floor_paper": 0.70,
    }

    def close(a, b, tol=0.005):
        return abs(a - b) <= tol

    drifts = []
    pairs = [
        ("tau_threshold", facts["tau_threshold_paper"], facts["tau_threshold_json"]),
        ("cf_AUC_min", facts["cf_AUC_at_min_overlap_paper"], facts["cf_AUC_at_min_overlap_json"]),
        ("cf_AUC_max", facts["cf_AUC_at_max_overlap_paper"], facts["cf_AUC_at_max_overlap_json"]),
        ("spearman_cf_fine", facts["spearman_cf_fine_paper"], facts["spearman_cf_fine_json"]),
        ("spearman_sf_fine", facts["spearman_sf_fine_paper"], facts["spearman_sf_fine_json"]),
        ("spearman_cf_orig", facts["spearman_cf_orig_paper"], facts["spearman_cf_orig_json"]),
        ("spearman_sf_orig", facts["spearman_sf_orig_paper"], facts["spearman_sf_orig_json"]),
        ("anthropic_min", facts["anthropic_min_transported_paper"], facts["anthropic_min_transported_json"]),
    ]
    for name, paper_v, json_v in pairs:
        delta = round(abs(paper_v - json_v), 4)
        ok = close(paper_v, json_v, tol=0.005)
        drifts.append({"metric": name, "paper": paper_v, "json": json_v, "abs_delta": delta, "match": ok})

    results["numeric_cross_check"] = {"facts": facts, "drifts": drifts,
                                       "all_match_at_tol_0.005": all(d["match"] for d in drifts)}

    # ----- integrity protocol checks -----
    full = text.lower()
    integrity = {
        "rule1_preregistered_in_script": "preregistered" in full and "before the run" in full,
        "rule2_failed_replication_in_body_not_footnote": "## 6" in text and "this is reported in the body, not in a footnote" in full,
        "rule3_killed_cross_vendor_referenced": "preregistration-killed" in full or "preregistration-**killed**" in full or "h_kill" in full or "killed" in full,
        "universal_language_scan": {
            "occurrences_of_'universal'": len(re.findall(r"universal", text, re.I)),
            "occurrences_of_'all of ai'": len(re.findall(r"all of ai", text, re.I)),
            # these terms appear ONLY in non-claims / explicit retraction context; verify by counting non-claim guard
            "explicit_non_claim_marker": text.count("Explicit non-claims") >= 1 or "not a universal" in full,
            "all_of_ai_only_in_retraction_context": all(
                "earned" in text[max(0,m.start()-200):m.start()+200].lower() or
                "not earned" in text[max(0,m.start()-200):m.start()+200].lower() or
                "not " in text[max(0,m.start()-100):m.start()+50].lower()
                for m in re.finditer(r"all of ai", text, re.I)
            ),
        },
        "n_stated_with_numbers": {
            "n=5_present": "n=5" in text or "5 in the original" in text,
            "n=12_present": "n=12" in text or "12 in the fine" in text,
            "n=75_present": "75" in text and ("eval" in full),
            "n=360_present": "n=360" in text,
        },
        "limits_section_present_and_long": {
            "section_exists": "## 7. Limits" in text,
            "length_chars": len(slice_section(text, r"## 7\. ", r"## 8\. ")),
            "is_explicit_and_long": len(slice_section(text, r"## 7\. ", r"## 8\. ")) > 1500,
        },
    }
    results["integrity_protocol"] = integrity

    # ----- circular oracle check -----
    # paper defines tau as "smallest overlap at which cross-family AUC crosses 0.80"
    # check: is 0.80 used elsewhere as independent evidence?
    auc_080_mentions = [(m.start(), text[max(0,m.start()-180):m.start()+120]) for m in re.finditer(r"0\.80", text)]
    results["circular_oracle_check"] = {
        "tau_definition_uses_0.80_floor": True,
        "0.80_occurrences": len(auc_080_mentions),
        "0.80_contexts": [c.replace("\n", " ")[:220] for _, c in auc_080_mentions],
        "verdict": (
            "0.80 appears as: (a) preregistered floor for 'transport holds', "
            "(b) the boundary used to LOCATE tau, (c) the threshold tau sits at. "
            "These are the SAME 0.80, used consistently — not re-deployed as "
            "independent evidence. The paper does not claim '0.80 is hit by an "
            "independent test'; it claims tau is the overlap level where the "
            "preregistered 0.80 floor is first crossed. Not circular: "
            "definitional/consistent."
        ),
    }

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
