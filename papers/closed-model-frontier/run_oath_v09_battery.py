"""OATH v0.9 battery — G1..G7 per PREREG_oath_v09_is_spec_json_idiom_2026_08_23.

Under test: two severable `is_spec` recall clauses in ``styxx.certify``.

  V09_IS_SPEC_JSON_IDIOM  a token in JSON value position whose object also carries a
                          comparison-operator field is a SPECIFICATION -> ABSTAIN.  (primary)
  V09_IS_SPEC_BAR_NOUN    a token immediately followed by a bar noun ("0.10 floor") is a
                          SPECIFICATION -> ABSTAIN.  (control, shipped OFF)

Every gate runs in BOTH arms, flags OFF and flags ON, on the identical sample with the identical
mutation seed. **If the ON arm does not exceed the OFF arm on G1 the run is VOID** — a battery
reporting the same number in both arms is not measuring the clause.

G6 is the control and is EXPECTED TO FAIL. It is here for two reasons. It is the disciplined test
of the second clause, which is otherwise an untested opinion. And it is the POSITIVE CONTROL for
the safety legs: G3 and G4 assert that the primary clause destroys nothing, and a screen with
unknown recall reporting zero on a corpus is indistinguishable from a screen that cannot see. G6
is what proves this battery detects catch destruction when catch destruction is present.

Non-destructive: mutants live in temp files, corpus passes are in-memory, and the only file
written is this battery's own result JSON.

  python papers/closed-model-frontier/run_oath_v09_battery.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import styxx.certify as C                                                  # noqa: E402
from styxx.certify import certify_doc                                      # noqa: E402
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

CENSUS = HERE / "oath_v09_isspec_census.json"
BASELINE = HERE / "oath_v09_baseline_ledger.json"
OUT = HERE / "oath_v09_battery_result.json"

MUT_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
G1_BAR_ON, G1_BAR_OFF = 140, 2
G2_BAR = 24
G3_BAR = 2
G4_BAR = 0
G5_BAR = 0

# ---- hand adjudication of `frozen_adjudication_sample` (n=25, seed 9) against the prereg's frozen
# definition: a BAR is a value the document fixes IN ADVANCE as a condition on some other quantity
# and whose authority is a preregistration; a MEASUREMENT is produced by running something. Ties
# resolve MEASUREMENT — against the clause. Every one of the 25 is the `value` of a named gate in
# a prereg's frozen gates block ("G1_stability", "G2_null_clean", ...), i.e. the bar the run must
# clear, recorded before the run. The single case worth naming is
# PREREG_oath_v07_...:L76 `0.00648`, which is a bar QUOTED inside prose as an example of the class;
# it is still a bar, and its receipt is PREREG_b35c_open_vocab_2026_08_03.md.
ADJUDICATION = {
    ("PREDICTION_h1_human_islands_2026_08_06.md", 58, "0.50"): "BAR",
    ("PREREG_b34v3_labelfree_read_2026_08_03.md", 43, "0.0286"): "BAR",
    ("PREREG_b35_seed_stability_2026_08_03.md", 36, "0.143"): "BAR",
    ("PREREG_b38_legibility_cliff_2026_08_04.md", 37, "0.30"): "BAR",
    ("PREREG_b40_anisotropy_signature_2026_08_05.md", 35, "0.0"): "BAR",
    ("PREREG_b41_bridge_2026_08_05.md", 42, "0.30"): "BAR",
    ("PREREG_b42_bridge_dose_2026_08_05.md", 28, "0.15"): "BAR",
    ("PREREG_b47_eight_minds_2026_08_06.md", 35, "0.0"): "BAR",
    ("PREREG_b48_legibility_matrix_ten_2026_08_06.md", 36, "0.0208"): "BAR",
    ("PREREG_b49_amplitude_reaudit_2026_08_07.md", 25, "1"): "BAR",
    ("PREREG_c4_signed_statistic_2026_08_07.md", 14, "0.80"): "BAR",
    ("PREREG_c4_signed_statistic_2026_08_07.md", 15, "0.10"): "BAR",
    ("PREREG_c6_derived_bar_2026_08_13.md", 86, "0"): "BAR",
    ("PREREG_e1_effective_n_bakeoff_2026_08_08.md", 38, "0.20"): "BAR",
    ("PREREG_m1_magnetochemistry_2026_08_05.md", 59, "5.0"): "BAR",
    ("PREREG_oath_v07_precision_obligation_2026_08_22.md", 76, "0.00648"): "BAR",
    ("PREREG_p1_power_refusal_2026_08_08.md", 46, "3"): "BAR",
    ("PREREG_protocol_power_basis_2026_08_07.md", 32, "33"): "BAR",
    ("PREREG_protocol_v4_composition_2026_08_09.md", 44, "1.0"): "BAR",
    ("PREREG_protocol_v4_composition_2026_08_09.md", 53, "0"): "BAR",
    ("PREREG_r1_room_legibility_2026_08_05.md", 54, "0.10"): "BAR",
    ("PREREG_r1v2_room_coupling_2026_08_06.md", 29, "0.01"): "BAR",
    ("PREREG_voice_lora_honesty_2026_08_11.md", 74, "0"): "BAR",
    ("PREREG_w1v2_water_imprint_2026_08_05.md", 70, "9"): "BAR",
    ("PREREG_w1v3_water_arms_2026_08_05.md", 54, "1"): "BAR",
}


def set_flags(json_idiom: bool, bar_noun: bool) -> None:
    C.V09_IS_SPEC_JSON_IDIOM = json_idiom
    C.V09_IS_SPEC_BAR_NOUN = bar_noun


def resolvable_docs():
    out = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec)
        if receipts and not missing:
            out.append((doc, receipts))
    return out


def spec_hit(rel_path: str, line_no: int, token: str) -> bool:
    """Does `is_spec` fire on this token, at the CURRENT flag settings?

    Evaluated by certifying the token's own line against an empty receipt set: with no receipts
    nothing can VERIFY, so ABSTAIN with the `spec-or-historical` reason is exactly `is_spec` (or
    `is_hist`, which no token in these frames triggers). Reads the verifier itself rather than a
    copy of its regexes, which is the point of a battery."""
    doc = ROOT / rel_path
    lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
    lo = max(0, line_no - 3)
    stub = lines[lo:line_no]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "d.md"
        d.write_text("# t\n\npreamble sentence for the line-start filter.\n\n"
                     + "\n".join(stub) + "\n", encoding="utf-8")
        r = Path(td) / "r.json"
        r.write_text("{}", encoding="utf-8")
        cert = certify_doc(d, [r])
    for e in cert["ledger"]:
        if e["token"] == token and e["status"] == "ABSTAIN" \
                and e.get("receipt_ref") == "spec-or-historical":
            return True
    return False


def frame_recall(frame, json_idiom: bool, bar_noun: bool) -> dict:
    set_flags(json_idiom, bar_noun)
    hit = [f for f in frame if spec_hit(f["rel"], f["line"], f["token"])]
    return {"json_idiom": json_idiom, "bar_noun": bar_noun,
            "rescued": len(hit), "n": len(frame),
            "keys": [f"{f['doc']}|L{f['line']}|{f['token']}" for f in hit]}


def corpus_pass(docs, json_idiom: bool, bar_noun: bool) -> dict:
    set_flags(json_idiom, bar_noun)
    ledger, verdicts = {}, {}
    for doc, receipts in docs:
        try:
            cert = certify_doc(doc, receipts)
        except Exception:                                        # pragma: no cover - defensive
            continue
        rel = doc.relative_to(ROOT).as_posix()
        verdicts[rel] = cert["verdict"]
        for e in cert["ledger"]:
            ledger[f"{rel}|L{e['line']}|{e['token']}"] = e["status"]
    return {"ledger": ledger, "verdicts": verdicts}


def mutate_sig(tok: str, rng: random.Random) -> str:
    if "." in tok:
        frac_at = tok.index(".") + 1
        frac = tok[frac_at:]
        sig = [i for i in range(min(6, len(frac)))
               if not (frac[i] == "0" and set(frac[:i]) <= {"0"})]
        pos = frac_at + rng.choice(sig or [0])
    else:
        pos = rng.choice([i for i, ch in enumerate(tok) if ch.isdigit()])
    old = int(tok[pos])
    return tok[:pos] + str(rng.choice([d for d in range(10) if d != old])) + tok[pos + 1:]


def substitute(line: str, tok: str, mut: str):
    """Sign-aware substitution — extraction normalizes U+2212 to ASCII while the document holds
    U+2212, so a bare `line.replace` silently no-ops on every signed claim and the harness miss
    scores as a verifier miss. Inherited from `run_oath_v07_battery.py`, which owns the defect."""
    if tok in line:
        return line.replace(tok, mut, 1), True
    if tok.startswith("-"):
        alt, alt_mut = tok.replace("-", "−", 1), mut.replace("-", "−", 1)
        if alt in line:
            return line.replace(alt, alt_mut, 1), True
    return line, False


def mutation_arm(roster, doc_by_name, seed: int, json_idiom: bool, bar_noun: bool) -> dict:
    set_flags(json_idiom, bar_noun)
    rng = random.Random(seed)
    counts, unlanded = collections.Counter(), 0
    for r in roster:
        doc, receipts = doc_by_name[r["doc"]]
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        mut = mutate_sig(r["token"], rng)
        ml = list(lines)
        ml[r["line"] - 1], landed = substitute(ml[r["line"] - 1], r["token"], mut)
        unlanded += not landed
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                         encoding="utf-8") as tf:
            tf.write("\n".join(ml))
            tmp = Path(tf.name)
        try:
            cert = certify_doc(tmp, receipts)
        finally:
            tmp.unlink(missing_ok=True)
        e = next((x for x in cert["ledger"]
                  if x["line"] == r["line"] and x["token"] == mut), None)
        counts[e["status"] if e else "NOT_EXTRACTED"] += 1
    return {"seed": seed, "caught": counts["UNGROUNDED"], "false_attested": counts["VERIFIED"],
            "abstained": counts["ABSTAIN"], "did_not_land": unlanded,
            "outcomes": dict(counts)}


def main() -> int:                                                   # noqa: C901 - one report
    t0 = time.time()
    for flag in ("V09_IS_SPEC_JSON_IDIOM", "V09_IS_SPEC_BAR_NOUN"):
        if not hasattr(C, flag):
            print(f"FATAL: styxx.certify has no {flag} — nothing to test.")
            return 2
    for p in (CENSUS, BASELINE):
        if not p.exists():
            print(f"FATAL: {p.name} missing — the pre-fix measurement cannot be re-read.")
            return 2
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    original = (C.V09_IS_SPEC_JSON_IDIOM, C.V09_IS_SPEC_BAR_NOUN)

    frame = census["adjudication_frame"]["roster"]
    bar_roster = census["bar_noun"]["roster"]
    docs = resolvable_docs()
    doc_by_name = {d.name: (d, rc) for d, rc in docs}
    print(f"documents with resolvable receipts: {len(docs)}   "
          f"adjudication frame: {len(frame)}   bar-noun roster: {len(bar_roster)}\n")

    # ---- G1 recall, two-armed -------------------------------------------------------------
    g1_off = frame_recall(frame, False, False)
    g1_on = frame_recall(frame, True, False)
    g1_exceeds = g1_on["rescued"] > g1_off["rescued"]
    print(f"G1 recall  ON  : {g1_on['rescued']}/{len(frame)} (bar >= {G1_BAR_ON})")
    print(f"G1 control OFF : {g1_off['rescued']}/{len(frame)} (bar <= {G1_BAR_OFF}; ON must exceed)")

    # ---- G2 adjudicated precision ---------------------------------------------------------
    set_flags(True, False)
    sample = census["frozen_adjudication_sample"]
    adjudged, missing_verdicts = [], []
    for s in sample:
        key = (s["doc"], s["line"], s["token"])
        verdict = ADJUDICATION.get(key)
        if verdict is None:
            missing_verdicts.append("|".join(map(str, key)))
            continue
        abstained = spec_hit(s["rel"], s["line"], s["token"])
        adjudged.append({"key": "|".join(map(str, key)), "adjudication": verdict,
                         "abstained_by_clause": abstained})
    bars_abstained = sum(1 for a in adjudged if a["adjudication"] == "BAR"
                         and a["abstained_by_clause"])
    false_abstentions = [a for a in adjudged if a["adjudication"] != "BAR"
                         and a["abstained_by_clause"]]
    print(f"G2 adjudicated : {bars_abstained}/{len(sample)} abstained tokens adjudicate BAR "
          f"(bar >= {G2_BAR}); false abstentions {len(false_abstentions)}")

    # ---- G3/G4 corpus safety, ON vs baseline ----------------------------------------------
    print(f"\nG3/G4 clean corpus pass: {len(docs)} documents", flush=True)
    c_on = corpus_pass(docs, True, False)
    moved_v2a = [k for k, s in base["ledger"].items()
                 if s == "VERIFIED" and c_on["ledger"].get(k) == "ABSTAIN"]
    silenced = [k for k, s in base["ledger"].items()
                if s == "UNGROUNDED" and c_on["ledger"].get(k) == "ABSTAIN"]
    failed_before = {d for d, v in base["verdicts"].items() if v != "OATH-HELD"}
    held_after = {d for d, v in c_on["verdicts"].items() if v == "OATH-HELD"}
    unfailed = sorted(failed_before & held_after)
    new_ung = [k for k, s in c_on["ledger"].items()
               if s == "UNGROUNDED" and base["ledger"].get(k) != "UNGROUNDED"]
    print(f"G3 VERIFIED -> ABSTAIN : {len(moved_v2a)} (bar <= {G3_BAR})")
    print(f"G4 UNGROUNDED silenced : {len(silenced)} (bar {G4_BAR}) | "
          f"FAILED -> HELD flips: {len(unfailed)} (bar {G4_BAR}) | "
          f"new UNGROUNDED (I2 must be 0): {len(new_ung)}")

    # ---- G5 severability -------------------------------------------------------------------
    c_off = corpus_pass(docs, False, False)
    diffs = [k for k in set(c_off["ledger"]) | set(base["ledger"])
             if c_off["ledger"].get(k) != base["ledger"].get(k)]
    print(f"G5 severability : {len(diffs)} ledger differences with both flags OFF (bar {G5_BAR})")
    for k in diffs[:10]:
        print(f"  [DIFF] {k}: baseline={base['ledger'].get(k)} off={c_off['ledger'].get(k)}")

    # ---- G6 catch preservation — the control, and the positive control for G3/G4 -----------
    print(f"\nG6 catch preservation on the {len(bar_roster)}-token bar-noun roster, "
          f"seeds {MUT_SEEDS}:")
    g6 = []
    for seed in MUT_SEEDS:
        off = mutation_arm(bar_roster, doc_by_name, seed, True, False)
        on = mutation_arm(bar_roster, doc_by_name, seed, True, True)
        ok = on["caught"] >= off["caught"]
        g6.append({"seed": seed, "off": off, "on": on, "pass": ok})
        print(f"  seed {seed:<3d} caught OFF {off['caught']:3d} -> ON {on['caught']:3d}   "
              f"false-attested OFF {off['false_attested']:3d} -> ON {on['false_attested']:3d}   "
              f"{'PASS' if ok else 'FAIL'}")
    g6_pass = all(s["pass"] for s in g6)
    catches_lost = [s["off"]["caught"] - s["on"]["caught"] for s in g6]
    print(f"G6 {'PASS' if g6_pass else 'FAIL'} — catches lost per seed: "
          f"{min(catches_lost)}-{max(catches_lost)} "
          f"(mean {sum(catches_lost) / len(catches_lost):.1f})")

    C.V09_IS_SPEC_JSON_IDIOM, C.V09_IS_SPEC_BAR_NOUN = original

    gates = {
        "G1_recall": {"role": "RECALL (gated, two-armed)", "on": g1_on["rescued"],
                      "off": g1_off["rescued"], "bar_on": G1_BAR_ON, "bar_off": G1_BAR_OFF,
                      "on_exceeds_off": g1_exceeds,
                      "pass": (g1_on["rescued"] >= G1_BAR_ON
                               and g1_off["rescued"] <= G1_BAR_OFF and g1_exceeds)},
        "G2_adjudicated_precision": {
            "role": "PRECISION (gated, hand-adjudicated, ties against the clause)",
            "bars_abstained": bars_abstained, "n": len(sample), "bar": G2_BAR,
            "false_abstentions": [a["key"] for a in false_abstentions],
            "unadjudicated": missing_verdicts, "rows": adjudged,
            "pass": (bars_abstained >= G2_BAR and not false_abstentions
                     and not missing_verdicts)},
        "G3_coverage_bound": {"role": "COVERAGE (gated) — refuses the metric that rewards silence",
                              "verified_to_abstain": len(moved_v2a), "bar": G3_BAR,
                              "keys": moved_v2a[:50],
                              "pass": len(moved_v2a) <= G3_BAR},
        "G4_no_silenced_accusation": {
            "role": "SAFETY (gated)", "ungrounded_silenced": len(silenced),
            "failed_to_held_flips": unfailed, "new_ungrounded_I2": new_ung, "bar": G4_BAR,
            "pass": (len(silenced) == G4_BAR and not unfailed and not new_ung)},
        "G5_severability": {"role": "SEVERABILITY (gated)", "ledger_differences": len(diffs),
                            "bar": G5_BAR, "examples": diffs[:20],
                            "pass": len(diffs) == G5_BAR},
        "G6_catch_preservation": {
            "role": "CONTROL (gated; decides V09_IS_SPEC_BAR_NOUN) + POSITIVE CONTROL for G3/G4",
            "seeds": list(MUT_SEEDS), "per_seed": g6,
            "catches_lost_range": [min(catches_lost), max(catches_lost)],
            "mean_catches_lost": round(sum(catches_lost) / len(catches_lost), 2),
            "expected_to_fail": True, "pass": g6_pass},
    }
    report = {
        "prereg": "PREREG_oath_v09_is_spec_json_idiom_2026_08_23.md",
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "baseline_verifier_sha256": base["verifier_sha256"],
        "census_verifier_sha256": census["generated_at_verifier_sha256"],
        "shipped_flags": {"V09_IS_SPEC_JSON_IDIOM": original[0],
                          "V09_IS_SPEC_BAR_NOUN": original[1]},
        "documents": len(docs),
        "adjudication_frame_n": len(frame),
        "bar_noun_roster_n": len(bar_roster),
        "gates": gates,
        "asserted_invariants_not_gated": {
            "I1": "an abstained token stays abstained under one-digit mutation — the predicate "
                  "reads context and the substitution preserves token length. 0 false "
                  "attestations on the reached class is an IDENTITY, not a measurement.",
            "I2": "is_spec yields only ABSTAIN, so no clause creates an UNGROUNDED token and no "
                  "certificate flips OATH-HELD -> OATH-FAILED. Measured above as "
                  f"new_ungrounded_I2 = {len(new_ung)}.",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    primary = {k: v["pass"] for k, v in gates.items() if k != "G6_catch_preservation"}
    print("\npositive control (G1 ON must exceed OFF): "
          f"{'OK' if g1_exceeds else 'VOID — the battery is not measuring the clause'}")
    print("gates: " + "  ".join(f"{k.split('_')[0]}={'PASS' if v['pass'] else 'FAIL'}"
                                for k, v in gates.items()))
    print(f"primary clause (G1-G5): {'ALL PASS' if all(primary.values()) else 'MISSED A BAR'}")
    print(f"elapsed {report['elapsed_s']}s -> {OUT.name}")
    return 0 if all(primary.values()) and g1_exceeds else 1


if __name__ == "__main__":
    sys.exit(main())
