"""HARNESS-1 gates: the re-fold measured against its preregistration (PREREG_harness1_merge_fold_2026_09_16).

Reads the fold as it was (``external4_ledger.jsonl``: every commit's rows folded into the PR,
the #113 + #115 checkout) and the re-fold (``external6_ledger.jsonl``: merge-commit rows dropped,
the 300-row cap marked, this checkout — the same instrument on reconstructed diffs, G-BIN-3), and
EXTERNAL-5's per-item live facts (``external5_items.jsonl``, gitignored: the live file count of
each of the 91 pull requests, completed from its ``diff --git`` headers).  Scores:

  G-H1  the 85 live PRs: file-count items whose reconstructed count equals the live count, before
        and after; at least 14 more after, and none that agreed before disagrees after.  Blocking.
  G-H2  accusations can only fall where the fold could only inflate: ``only_touches`` CONTRADICTED
        does not rise; ``compat_claim`` readings naming a removed public definition do not rise;
        ``files_changed_count`` reported by direction.  Blocking.
  G-H3  BC-2's invariants: the three by-construction counters at 0; no ``tests_added`` /
        ``symbol_added`` verdict changes on a PR without a merge commit or a cap.  Blocking.
  G-H4  the numbers that get re-quoted, before and after: accusations by kind (the EXTERNAL-2
        census line and BC-2's survivors), COMPAT-1's "PRs whose diff drops a public name",
        coverage.  Reported.
  G-H5  no precision number; how many of the live PRs still disagree with the live count.

Counts only; no PR named.  Writes harness1_gates.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEFORE = HERE / "external4_ledger.jsonl"
AFTER = HERE / "external6_ledger.jsonl"
ITEMS = HERE / "external5_items.jsonl"
OUT = HERE / "harness1_gates.json"
PY = {".py", ".pyi"}
COUNT_WHY = re.compile(r"diff changes (\d+) files, claim says (\d+)")
G_H1_MIN_GAIN = 14


def _load(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out[r["pr_id"]] = r
    return out


def _accusations(ledger: dict, kind: str | None = None) -> set:
    return {(pid, c["kind"], c["text"]) for pid, r in ledger.items() for c in r["claims"]
            if c["verdict"] == "CONTRADICTED" and (kind is None or c["kind"] == kind)}


def _compat_drops(ledger: dict) -> int:
    """PRs whose (single) compat reading names at least one removed public definition — external4_gates' rule."""
    n = 0
    for r in ledger.values():
        cs = [c for c in r["claims"] if c["kind"] == "compat_claim"]
        if cs and cs[0]["detail"].get("languages") and cs[0]["detail"].get("removed"):
            n += 1
    return n


def _direction(why: str) -> str:
    m = COUNT_WHY.search(why)
    if not m:
        return "other"
    got, said = int(m.group(1)), int(m.group(2))
    return "corpus_above_claim" if got > said else "corpus_below_claim" if got < said else "equal"


def main() -> int:
    sys.path.insert(0, str(HERE))
    from external2_census import looks_like_path  # the census's own rule, unchanged

    before, after = _load(BEFORE), _load(AFTER)
    only_before, only_after = sorted(set(before) - set(after)), sorted(set(after) - set(before))
    common = set(before) & set(after)
    b = {p: before[p] for p in common}
    a = {p: after[p] for p in common}

    # ---- G-H1: the live PRs ------------------------------------------------------------------
    items = [json.loads(l) for l in ITEMS.open(encoding="utf-8")]
    fcc = [it for it in items if it["kind"] == "files_changed_count"
           and it["reading"]["outcome"] in ("UPHELD", "OVERTURNED")]
    agree_b = agree_a = 0
    newly = stopped = 0
    by_outcome = {"UPHELD": Counter(), "OVERTURNED": Counter()}
    capped_items = 0
    disagree_prs_after = set()
    live_prs = set()
    for it in fcc:
        pid = it["pr_id"]
        live = it["source"]["live_files"]
        m = re.search(r"diff changes (\d+) files", it["corpus_why"])
        n_before = int(m.group(1))
        rec = after.get(pid)
        n_after = rec["n_files"] if rec else None
        ab, aa = n_before == live, n_after == live
        live_prs.add(pid)
        if rec and rec["capped"]:
            capped_items += 1
        agree_b += ab
        agree_a += aa
        newly += (aa and not ab)
        stopped += (ab and not aa)
        by_outcome[it["reading"]["outcome"]][f"agree_before={ab},agree_after={aa}"] += 1
        if not aa:
            disagree_prs_after.add(pid)
    ot = [it for it in items if it["kind"] == "only_touches" and it["reading"]["outcome"] in ("UPHELD", "OVERTURNED")]
    ot_match_b = ot_match_a = 0
    for it in ot:
        rec = after.get(it["pr_id"])
        live_v = it["reading"]["verdict"]
        ot_match_b += (("CONTRADICTED" == live_v))          # every corpus item here was CONTRADICTED
        av = next((c["verdict"] for c in (rec["claims"] if rec else []) if c["kind"] == "only_touches"
                   and c["text"] == it["text"]), None)
        ot_match_a += (av == live_v)
    g_h1 = {"file_count_items": len(fcc), "agree_before": agree_b, "agree_after": agree_a,
            "gained": newly, "lost": stopped, "min_gain": G_H1_MIN_GAIN,
            "by_outcome": {k: dict(v) for k, v in by_outcome.items()},
            "items_on_capped_prs": capped_items,
            "only_touches_items_matching_the_live_verdict": {"before": ot_match_b, "after": ot_match_a, "of": len(ot)},
            "pass": (agree_a - agree_b) >= G_H1_MIN_GAIN and stopped == 0}

    # ---- G-H2: accusations can only fall where the fold could only inflate --------------------
    ot_b, ot_a = _accusations(b, "only_touches"), _accusations(a, "only_touches")
    cd_b, cd_a = _compat_drops(b), _compat_drops(a)
    fc_b, fc_a = _accusations(b, "files_changed_count"), _accusations(a, "files_changed_count")
    fc_dir_a = Counter(_direction(c["why"]) for r in a.values() for c in r["claims"]
                       if c["kind"] == "files_changed_count" and c["verdict"] == "CONTRADICTED")
    fc_dir_b = Counter(_direction(c["why"]) for r in b.values() for c in r["claims"]
                       if c["kind"] == "files_changed_count" and c["verdict"] == "CONTRADICTED")
    cap_rewrote = sum(r.get("cap_rewrote", 0) for r in a.values())
    cap_rewrote_from_contradicted = sum(1 for r in a.values() for c in r["claims"]
                                        if c.get("pre_cap", {}).get("verdict") == "CONTRADICTED")
    g_h2 = {"only_touches_contradicted": {"before": len(ot_b), "after": len(ot_a),
                                          "gained": len(ot_a - ot_b), "lost": len(ot_b - ot_a)},
            "compat_prs_naming_a_removed_public_definition": {"before": cd_b, "after": cd_a},
            "files_changed_count_contradicted": {"before": len(fc_b), "after": len(fc_a),
                                                 "gained": len(fc_a - fc_b), "lost": len(fc_b - fc_a),
                                                 "direction_before": dict(fc_dir_b), "direction_after": dict(fc_dir_a),
                                                 "rewritten_by_the_cap": cap_rewrote,
                                                 "rewritten_by_the_cap_from_CONTRADICTED": cap_rewrote_from_contradicted},
            "pass": len(ot_a) <= len(ot_b) and cd_a <= cd_b}

    # ---- G-H3: BC-2's invariants --------------------------------------------------------------
    by_construction = Counter()
    for pid, r in a.items():
        has_py = any(e in PY for e in r["exts"])
        for c in r["claims"]:
            if c["kind"] in ("tests_added", "symbol_added") and c["verdict"] == "CONTRADICTED" and not has_py:
                by_construction[f"{c['kind']}.contradicted_no_python_in_diff"] += 1
            if c["kind"] == "only_touches" and c["verdict"] == "CONTRADICTED" \
                    and not looks_like_path(c["detail"].get("prefix", "")):
                by_construction["only_touches.contradicted_prefix_is_not_a_path"] += 1
    for k in ("tests_added.contradicted_no_python_in_diff", "symbol_added.contradicted_no_python_in_diff",
              "only_touches.contradicted_prefix_is_not_a_path"):
        by_construction.setdefault(k, 0)
    untouched_prs = [pid for pid, r in a.items() if r["merge_commits"] == 0 and not r["capped"]]
    changed = 0
    checked = 0
    for pid in untouched_prs:
        vb = {(c["kind"], c["text"]): c["verdict"] for c in b[pid]["claims"] if c["kind"] in ("tests_added", "symbol_added")}
        va = {(c["kind"], c["text"]): c["verdict"] for c in a[pid]["claims"] if c["kind"] in ("tests_added", "symbol_added")}
        checked += len(vb)
        if vb != va:
            changed += 1
    g_h3 = {"counters": dict(by_construction), "prs_without_merge_or_cap": len(untouched_prs),
            "tests_or_symbol_verdicts_checked": checked, "prs_whose_verdicts_changed": changed,
            "pass": all(v == 0 for v in by_construction.values()) and changed == 0}

    # ---- G-H4: the numbers that get re-quoted ---------------------------------------------------
    acc_b, acc_a = _accusations(b), _accusations(a)
    g_h4 = {"accusations_total": {"before": len(acc_b), "after": len(acc_a)},
            "accusations_by_kind": {"before": dict(Counter(k for _, k, _ in acc_b)),
                                    "after": dict(Counter(k for _, k, _ in acc_a))},
            "accusations_gained": len(acc_a - acc_b), "accusations_lost": len(acc_b - acc_a),
            "prs_with_a_contradiction": {"before": sum(1 for r in b.values() if any(c["verdict"] == "CONTRADICTED" for c in r["claims"])),
                                         "after": sum(1 for r in a.values() if any(c["verdict"] == "CONTRADICTED" for c in r["claims"]))},
            "compat_prs_naming_a_removed_public_definition": {"before": cd_b, "after": cd_a},
            "covered_prs": {"before": sum(1 for r in b.values() if r["claims"]),
                            "after": sum(1 for r in a.values() if r["claims"])},
            "claims_by_verdict": {"before": dict(Counter(c["verdict"] for r in b.values() for c in r["claims"])),
                                  "after": dict(Counter(c["verdict"] for r in a.values() for c in r["claims"]))},
            "fold": {"prs_with_a_merge_commit": sum(1 for r in a.values() if r["merge_commits"]),
                     "prs_all_merge_commits_kept": sum(1 for r in a.values() if r["all_merge"]),
                     "prs_capped": sum(1 for r in a.values() if r["capped"]),
                     "prs_with_rows_dropped": sum(1 for r in a.values() if r["rows_folded"] != r["rows_total"]),
                     "prs_whose_file_count_changed": sum(1 for r in a.values() if r["n_files"] != r["n_files_all_rows"])},
            }

    # every claim's verdict, before -> after (the claim texts are the descriptions', so the claim
    # lists are identical by construction; only verdicts and reasons can move)
    trans = Counter()
    claim_lists_differ = 0
    for pid in common:
        cb, ca = b[pid]["claims"], a[pid]["claims"]
        if [(c["kind"], c["text"]) for c in cb] != [(c["kind"], c["text"]) for c in ca]:
            claim_lists_differ += 1
            continue
        for x, y in zip(cb, ca):
            if x["verdict"] != y["verdict"]:
                trans[f"{x['kind']}: {x['verdict']} -> {y['verdict']}" + (" (cap)" if y.get("pre_cap") else "")] += 1
    compat_trans = Counter()
    for pid in common:
        xb = [c for c in b[pid]["claims"] if c["kind"] == "compat_claim"]
        xa = [c for c in a[pid]["claims"] if c["kind"] == "compat_claim"]
        if not xb:
            continue
        rb = bool(xb[0]["detail"].get("languages") and xb[0]["detail"].get("removed"))
        ra = bool(xa[0]["detail"].get("languages") and xa[0]["detail"].get("removed"))
        if rb != ra:
            compat_trans["stopped_naming_a_removed_definition" if rb else "started_naming_one"] += 1
    per_lang = {}
    for tag, led in (("before", b), ("after", a)):
        per = Counter()
        for r in led.values():
            cs = [c for c in r["claims"] if c["kind"] == "compat_claim"]
            if cs and cs[0]["detail"].get("languages") and cs[0]["detail"].get("removed"):
                for lang in {x["language"] for x in cs[0]["detail"]["removed"]}:
                    per[lang] += 1
        per_lang[tag] = dict(per)
    g_h4["verdict_transitions"] = dict(sorted(trans.items()))
    g_h4["claim_lists_differ"] = claim_lists_differ
    g_h4["compat_transitions"] = dict(compat_trans)
    g_h4["compat_prs_naming_a_removed_public_definition_by_language"] = per_lang

    # ---- G-H5: what is not claimed -------------------------------------------------------------
    g_h5 = {"precision": "not measured; see prereg",
            "live_prs_with_a_file_count_item": len(live_prs),
            "live_prs_still_disagreeing_after": len(disagree_prs_after),
            "note": "a merge commit is identified by its message; a merge with a rewritten message is not caught"}

    gates = {"G-H1_live_prs": g_h1, "G-H2_only_falls_where_inflated": g_h2, "G-H3_bc2_invariants": g_h3,
             "G-H4_requoted_numbers": g_h4, "G-H5_not_claimed": g_h5}
    payload = {
        "prereg": "PREREG_harness1_merge_fold_2026_09_16.md",
        "before": {"ledger": BEFORE.name, "fold": "every commit's rows", "eligible": len(before)},
        "after": {"ledger": AFTER.name, "fold": "merge-commit rows dropped; 300-row cap marked", "eligible": len(after)},
        "eligibility_difference": {"only_before": len(only_before), "only_after": len(only_after)},
        "gates": gates,
        "all_blocking_gates_pass": all(gates[g]["pass"] for g in ("G-H1_live_prs", "G-H2_only_falls_where_inflated", "G-H3_bc2_invariants")),
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0 if payload["all_blocking_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
