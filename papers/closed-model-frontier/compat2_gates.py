"""COMPAT-2 gates: the sharpened reading measured against its preregistration
(PREREG_compat2_surface_and_panel_2026_09_16).

Reads the HARNESS-1 ledger (`external6_ledger.jsonl`: the re-fold, the #120 checkout) and the
COMPAT-2 ledger (`compat2_ledger.jsonl`: the same shelf, the same fold, this checkout —
`external6_harness.py gate --tag compat2`), and scores:

  G-C2-1  still never accuses: zero `compat_claim` verdicts other than UNCHECKABLE.  Blocking.
  G-C2-2  every other kind untouched: the multiset of (pr_id, kind, text, verdict) for all
          non-compat_claim claims is identical.  Blocking.
  G-C2-3  the partition is a partition: per PR the multiset of removed (path, name) is identical
          to HARNESS-1's, and surface + scaffolding = all.  Blocking.  Reported beside it: the
          candidate count, per language, the all-scaffolding count, the signature-change count.
  G-C2-6  what is not claimed.

G-C2-4 (the port) is the differential's receipt (web/gate/README.md); G-C2-5 (suite, demo, the
packet build) is the suite's and compat2_packet.py's.  Counts only; no PR named.  Writes
compat2_gates.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEFORE = HERE / "external6_ledger.jsonl"
AFTER = HERE / "compat2_ledger.jsonl"
OUT = HERE / "compat2_gates.json"


def _load(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out[r["pr_id"]] = r
    return out


def _compat(r: dict) -> dict | None:
    cs = [c for c in r["claims"] if c["kind"] == "compat_claim"]
    return cs[0] if cs else None


def main() -> int:
    before, after = _load(BEFORE), _load(AFTER)
    assert set(before) == set(after), "the two ledgers cover different PRs"

    # G-C2-1
    verdicts = Counter(c["verdict"] for r in after.values() for c in r["claims"] if c["kind"] == "compat_claim")
    g1 = {"verdicts": dict(verdicts), "pass": set(verdicts) <= {"UNCHECKABLE"} and sum(verdicts.values()) > 0}

    # G-C2-2
    other_b = Counter((pid, c["kind"], c["text"], c["verdict"]) for pid, r in before.items()
                      for c in r["claims"] if c["kind"] != "compat_claim")
    other_a = Counter((pid, c["kind"], c["text"], c["verdict"]) for pid, r in after.items()
                      for c in r["claims"] if c["kind"] != "compat_claim")
    g2 = {"before": sum(other_b.values()), "after": sum(other_a.values()),
          "differences": sum((other_b - other_a).values()) + sum((other_a - other_b).values()),
          "pass": other_b == other_a}

    # G-C2-3
    partition_bad = 0
    names_differ = 0
    # A PR whose diff does not parse at all short-circuits EVERY claim before the per-kind reading
    # ("the diff carries no file statuses and no added lines"), so its compat claim carries no
    # reading and no detail.  That is the instrument's pre-existing behaviour, untouched by
    # COMPAT-2; the prereg's condition is about the removed names, which such a PR has none of.
    # They are counted here, checked byte-identical against the HARNESS-1 ledger, and reported.
    short_circuit = 0
    short_circuit_identical = 0
    candidates = 0
    cand_lang = Counter()
    all_scaffold = 0
    with_removed = 0
    sig_prs = 0
    sig_total = 0
    surface_names = scaffold_names = 0
    claims_read = 0
    prs_with_claim = 0
    for pid in after:
        cb, ca = _compat(before[pid]), _compat(after[pid])
        if ca is None:
            assert cb is None
            continue
        prs_with_claim += 1
        claims_read += sum(1 for c in after[pid]["claims"] if c["kind"] == "compat_claim")
        db, da = cb["detail"], ca["detail"]
        if "removed" not in da:
            short_circuit += 1
            cbs = [(c["verdict"], c["why"], c["detail"]) for c in before[pid]["claims"] if c["kind"] == "compat_claim"]
            cas = [(c["verdict"], c["why"], c["detail"]) for c in after[pid]["claims"] if c["kind"] == "compat_claim"]
            short_circuit_identical += (cbs == cas)
            continue
        mb = Counter((x["path"], x["name"]) for x in db.get("removed", []))
        ma = Counter((x["path"], x["name"]) for x in da.get("removed", []))
        if mb != ma:
            names_differ += 1
        s = sum(1 for x in da.get("removed", []) if x.get("surface"))
        t = sum(1 for x in da.get("removed", []) if not x.get("surface"))
        if s != da.get("surface_removed", -1) or s + t != len(da.get("removed", [])):
            partition_bad += 1
        surface_names += s
        scaffold_names += t
        if da.get("languages") and da.get("removed"):
            with_removed += 1
            if da.get("compat2_candidate"):
                candidates += 1
                for lang in {x["language"] for x in da["removed"] if x.get("surface")}:
                    cand_lang[lang] += 1
            else:
                all_scaffold += 1
        if da.get("signature_changed"):
            sig_prs += 1
            sig_total += len(da["signature_changed"])
        if bool(da.get("compat2_candidate")) != (s > 0):
            partition_bad += 1
    g3 = {"prs_whose_removed_names_differ": names_differ, "prs_failing_the_partition": partition_bad,
          "prs_whose_diff_does_not_parse": {"count": short_circuit, "identical_to_harness1": short_circuit_identical,
                                            "note": "every claim short-circuits before the per-kind reading; "
                                                    "pre-existing, untouched by COMPAT-2, and outside the prereg's "
                                                    "condition, which is about removed names"},
          "prs_with_a_compat_claim": prs_with_claim, "compat_claims": claims_read,
          "prs_with_a_removed_public_name": with_removed,
          "candidates": candidates, "candidates_by_language": dict(cand_lang),
          "prs_all_scaffolding": all_scaffold,
          "removed_names": {"surface": surface_names, "scaffolding": scaffold_names},
          "prs_with_a_signature_change": sig_prs, "signature_changes": sig_total,
          "exploratory_expectation": {"candidates": 211, "all_scaffolding": 82,
                                      "note": "counted with this rule over the HARNESS-1 ledger before the freeze; a count, not a result"},
          "pass": names_differ == 0 and partition_bad == 0 and short_circuit == short_circuit_identical}

    g6 = {"precision": "not measured; the panel's cycle", "agent_comparison": "none",
          "note": "a candidate is a candidate; COMPAT2_LICENSED is false"}

    gates = {"G-C2-1_still_never_accuses": g1, "G-C2-2_every_other_kind_untouched": g2,
             "G-C2-3_partition": g3, "G-C2-6_not_claimed": g6}
    payload = {"prereg": "PREREG_compat2_surface_and_panel_2026_09_16.md",
               "before": {"ledger": BEFORE.name, "instrument": "the #120 checkout (HARNESS-1 fold)", "eligible": len(before)},
               "after": {"ledger": AFTER.name, "instrument": "this checkout, COMPAT-2 reading (same fold)", "eligible": len(after)},
               "gates": gates,
               "all_blocking_gates_pass": all(gates[g]["pass"] for g in
                                              ("G-C2-1_still_never_accuses", "G-C2-2_every_other_kind_untouched", "G-C2-3_partition"))}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0 if payload["all_blocking_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
