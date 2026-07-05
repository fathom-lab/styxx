"""Verdict driver — wires the frozen §2 tests (analysis.py) to the frozen
results JSONs (results/main_{id,ood1,ood2}.jsonl) with NO free parameters
(PREREG_v2 §11: "The code path from results JSONs to verdict table contains no
free parameters not fixed above").

What it does, in order:
  1. Load the three result arms; verify the row contract.
  2. Apply §5 complete-case exclusion PER ARM (a row is usable iff
     excluded_flag is None AND `correct` is a real bool AND every numeric signal
     {SE, depth, LP_mean, LP_norm} is finite). Report the exclusion breakdown
     per §5 ("counted per dataset in every table; never silently dropped").
  3. ID arm  -> H1 (analysis.h1) + H2 (analysis.h2_full, SE primary; LP_* Holm).
  4. H3      -> analysis.h3_ood: fit logistic on ID complete-case, freeze,
                score OOD-1 (PopQA-rare) complete-case.
  5. OOD-2 (TruthfulQA) -> NO hypothesis test: it is the SECONDARY, KG3-gated
     arm (PREREG §3/§8: 50-item human audit by flobi BEFORE any TruthfulQA
     claim). Human audit is not present in this autopilot cycle, so this arm is
     reported as ATTEMPTED/PENDING with its mechanical grade counts only.
  6. Apply the §8 kill-gate logic (KG2) and §10 falsification map to produce a
     verdict string, and dump results/verdict.json + print a table.

Deterministic given the pinned bootstrap seed 7 (analysis.py). No GPU, no
network, no model loading.

Usage:  python run_analysis.py            (from papers/depth-truth/harness/)
        python -m harness.run_analysis    (from papers/depth-truth/)
"""

from __future__ import annotations

import json
import math
import os
import sys

# Allow both "python run_analysis.py" (cwd=harness) and package import.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import analysis  # noqa: E402  (frozen §2 tests)

_RESULTS = os.path.normpath(os.path.join(_HERE, "..", "results"))
_ARMS = {
    "id": "main_id.jsonl",
    "ood1": "main_ood1.jsonl",
    "ood2": "main_ood2.jsonl",
}
_NUMERIC = ("SE", "depth", "LP_mean", "LP_norm")


def _load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def complete_case(rows):
    """PREREG §5 complete-case subset + a full exclusion breakdown.

    A row is usable iff:
      * excluded_flag is None (not nonanswer / depth_undefined / grade_ambiguous),
      * `correct` is a genuine bool (TruthfulQA rows carry correct=None until the
        neither-list is resolved -> excluded here),
      * every numeric signal in _NUMERIC is a finite number.
    """
    kept, breakdown = [], {}
    n_flagged = n_no_label = n_bad_signal = 0
    for r in rows:
        flag = r.get("excluded_flag")
        if flag is not None:
            breakdown[flag] = breakdown.get(flag, 0) + 1
            n_flagged += 1
            continue
        if not isinstance(r.get("correct"), bool):
            n_no_label += 1
            continue
        if not all(_is_finite_number(r.get(k)) for k in _NUMERIC):
            n_bad_signal += 1
            continue
        kept.append(r)
    return kept, {
        "n_total": len(rows),
        "n_kept": len(kept),
        "n_excluded_flagged": n_flagged,
        "flag_breakdown": breakdown,
        "n_excluded_no_label": n_no_label,
        "n_excluded_bad_signal": n_bad_signal,
    }


def _col(rows, key):
    return [r[key] for r in rows]


def _class_balance(rows):
    y = [1 if r["correct"] else 0 for r in rows]
    return {"n": len(y), "n_correct": int(sum(y)), "n_wrong": int(len(y) - sum(y))}


def _depth_desc(rows):
    import numpy as np

    d = np.asarray(_col(rows, "depth"), dtype=float)
    return {
        "min": float(d.min()),
        "max": float(d.max()),
        "mean": float(d.mean()),
        "std": float(d.std(ddof=1)) if d.size > 1 else 0.0,
    }


def main():
    raw = {arm: _load(os.path.join(_RESULTS, fn)) for arm, fn in _ARMS.items()}
    cc = {arm: complete_case(rows) for arm, rows in raw.items()}

    id_rows = cc["id"][0]
    ood1_rows = cc["ood1"][0]

    report = {
        "experiment": "depth-truth keystone (PREREG_v2)",
        "results_dir": _RESULTS,
        "exclusions": {arm: cc[arm][1] for arm in _ARMS},
        "class_balance": {
            "id": _class_balance(id_rows),
            "ood1": _class_balance(ood1_rows),
        },
        "depth_describe": {
            "id": _depth_desc(id_rows),
            "ood1": _depth_desc(ood1_rows),
        },
    }

    # ---- Guardrails before any test (both classes required). ----
    y_id = _col(id_rows, "correct")
    both_id = len(set(bool(v) for v in y_id)) == 2
    y_ood1 = _col(ood1_rows, "correct")
    both_ood1 = len(set(bool(v) for v in y_ood1)) == 2

    # ---- H1 + H2 on ID ----
    if both_id:
        depth_id = _col(id_rows, "depth")
        SE_id = _col(id_rows, "SE")
        LPm_id = _col(id_rows, "LP_mean")
        LPn_id = _col(id_rows, "LP_norm")
        report["H1"] = analysis.h1(depth_id, y_id)
        report["H2"] = analysis.h2_full(SE_id, depth_id, y_id, LPm_id, LPn_id)
    else:
        report["H1"] = {"error": "ID arm single-class; AUROC undefined"}
        report["H2"] = {"error": "ID arm single-class; additivity undefined"}

    # ---- H3: fit on ID, freeze, score OOD-1 ----
    if both_id and both_ood1:
        report["H3"] = analysis.h3_ood(
            _col(id_rows, "SE"), _col(id_rows, "depth"), y_id,
            _col(ood1_rows, "SE"), _col(ood1_rows, "depth"), y_ood1,
        )
    else:
        report["H3"] = {
            "error": "single-class arm(s); OOD retention undefined",
            "id_both_classes": both_id,
            "ood1_both_classes": both_ood1,
        }

    # ---- OOD-2 (TruthfulQA): KG3-gated, NO hypothesis test in autopilot ----
    ood2_raw = raw["ood2"]
    tq_true = sum(1 for r in ood2_raw if r.get("correct") is True)
    tq_false = sum(1 for r in ood2_raw if r.get("correct") is False)
    tq_none = sum(1 for r in ood2_raw if r.get("correct") is None)
    report["OOD2_truthfulqa"] = {
        "status": "ATTEMPTED_PENDING_KG3_HUMAN_AUDIT",
        "note": (
            "PREREG §3/§8: TruthfulQA is the SECONDARY, disclosed-noisy arm; no "
            "claim until flobi grades the 50-item human_audit_sample.jsonl (KG3, "
            ">10% disagreement drops the arm). Human audit absent this cycle."
        ),
        "n_total": len(ood2_raw),
        "mechanical_grade": {"correct": tq_true, "incorrect": tq_false,
                             "none_or_ambiguous": tq_none},
    }

    # ---- Verdict per §8 (KG2) + §10 falsification map ----
    report["verdict"] = _verdict(report)

    out_path = os.path.join(_RESULTS, "verdict.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    _print_table(report)
    print(f"\nverdict.json -> {out_path}")
    return report


def _verdict(report):
    """§10 falsification map, applied to the frozen §2 pass/fail booleans.
    H1 pass = excludes_half. H2 pass (primary SE) = excludes_zero AND lrt_pass.
    H3 pass = excludes_zero. KG2: H1 null does not block H2/H3."""
    h1 = report.get("H1", {})
    h2 = report.get("H2", {})
    h3 = report.get("H3", {})

    h1_pass = bool(h1.get("excludes_half"))
    prim = h2.get("primary", {}) if isinstance(h2, dict) else {}
    h2_pass = bool(prim.get("excludes_zero") and prim.get("lrt_pass"))
    h3_pass = bool(h3.get("excludes_zero"))

    if h1.get("error") or (isinstance(h2, dict) and h2.get("error")):
        label = "UNDEFINED_SINGLE_CLASS"
        meaning = "An arm was single-class; the frozen tests are undefined."
    elif h1_pass and h2_pass and h3_pass:
        label = "KEYSTONE_STANDS"
        meaning = ("§10: H1+H2+H3 hold. Depth carries a readable truth signal "
                   "that adds to confidence in AND out of distribution. Sized to "
                   "one 2B model, one metric.")
    elif h2_pass and h3_pass and not h1_pass:
        label = "COMPLEMENTARY_OOD"
        meaning = ("§10: H1 null, H2+H3 hold. Depth is complementary-only "
                   "(no solo signal) but adds information incl. OOD.")
    elif h2_pass and not h3_pass:
        label = "COMPLEMENTARY_ID_ONLY"
        meaning = ("§10: H2 holds, H3 null. Depth adds in-distribution but not "
                   "where confidence fails OOD; the OOD claim dies.")
    elif h1_pass and not h2_pass:
        label = "SOLO_NO_ADDITIVITY"
        meaning = ("H1 holds but H2 null: depth has solo AUROC but does not add "
                   "over confidence. The additivity keystone does not stand.")
    elif not h1_pass and not h2_pass:
        label = "CLOSED_NEGATIVE_NO_TRUTH_SIGNAL"
        meaning = ("§10: H1 null AND H2 null. Depth carries no truth signal in "
                   "this regime; 'we measure thought, not words' reverts to "
                   "hypothesis and a README truth-in-advertising ticket opens.")
    else:
        label = "MIXED_SEE_TABLE"
        meaning = "Non-canonical combination; read the per-hypothesis rows."

    return {
        "label": label,
        "meaning": meaning,
        "H1_pass": h1_pass,
        "H2_primary_pass": h2_pass,
        "H3_pass": h3_pass,
        "note": "KG2: H1 null does not block H2/H3 (both still run).",
    }


def _fmt(v, nd=4):
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{nd}f}"
    return str(v)


def _print_table(report):
    print("=" * 72)
    print("DEPTH-TRUTH KEYSTONE — VERDICT (PREREG_v2 §2/§8/§10)")
    print("=" * 72)
    for arm in ("id", "ood1", "ood2"):
        ex = report["exclusions"][arm]
        print(f"[{arm}] total={ex['n_total']} kept(complete-case)={ex['n_kept']} "
              f"flagged={ex['n_excluded_flagged']} {ex['flag_breakdown']} "
              f"no_label={ex['n_excluded_no_label']} bad_signal={ex['n_excluded_bad_signal']}")
    print("-" * 72)
    for arm in ("id", "ood1"):
        cb = report["class_balance"][arm]
        dd = report["depth_describe"][arm]
        print(f"[{arm}] n={cb['n']} correct={cb['n_correct']} wrong={cb['n_wrong']} "
              f"| depth mean={_fmt(dd['mean'])} std={_fmt(dd['std'])} "
              f"[{_fmt(dd['min'])},{_fmt(dd['max'])}]")
    print("-" * 72)

    h1 = report["H1"]
    if "error" in h1:
        print(f"H1  ERROR: {h1['error']}")
    else:
        print(f"H1  AUROC(depth->correct)={_fmt(h1['auroc'])} "
              f"CI95=[{_fmt(h1['ci_lo'])},{_fmt(h1['ci_hi'])}] "
              f"excludes_0.5={h1['excludes_half']}  (PASS={h1['excludes_half']})")

    h2 = report["H2"]
    if "error" in h2:
        print(f"H2  ERROR: {h2['error']}")
    else:
        p = h2["primary"]
        print(f"H2  [SE primary] dAUC={_fmt(p['dAUC'])} "
              f"CI95=[{_fmt(p['ci_lo'])},{_fmt(p['ci_hi'])}] "
              f"excl_0={p['excludes_zero']} LRT_p={_fmt(p['lrt_p'],5)} "
              f"(pass<.01={p['lrt_pass']}) DeLong_p={_fmt(p['delong_p'],5)} "
              f"=> PASS={bool(p['excludes_zero'] and p['lrt_pass'])}")
        for name, s in h2.get("secondary", {}).items():
            hp = h2.get("secondary_holm_lrt_p", {}).get(name, float("nan"))
            print(f"    [{name} sec] dAUC={_fmt(s['dAUC'])} "
                  f"CI95=[{_fmt(s['ci_lo'])},{_fmt(s['ci_hi'])}] excl_0={s['excludes_zero']} "
                  f"LRT_p={_fmt(s['lrt_p'],5)} Holm={_fmt(hp,5)} DeLong_p={_fmt(s['delong_p'],5)}")

    h3 = report["H3"]
    if "error" in h3:
        print(f"H3  ERROR: {h3['error']}")
    else:
        print(f"H3  [fit ID, score OOD-1] dAUC_ood={_fmt(h3['dAUC_ood'])} "
              f"CI95=[{_fmt(h3['ci_lo'])},{_fmt(h3['ci_hi'])}] "
              f"excl_0={h3['excludes_zero']} DeLong_p={_fmt(h3['delong_p'],5)} "
              f"(PASS={h3['excludes_zero']})")

    o2 = report["OOD2_truthfulqa"]
    print("-" * 72)
    print(f"OOD-2 TruthfulQA: {o2['status']} "
          f"(mech grade {o2['mechanical_grade']}) — {o2['n_total']} items")
    print("-" * 72)
    v = report["verdict"]
    print(f"VERDICT: {v['label']}")
    print(f"  {v['meaning']}")
    print(f"  H1={v['H1_pass']} H2(SE)={v['H2_primary_pass']} H3={v['H3_pass']}")


if __name__ == "__main__":
    main()
