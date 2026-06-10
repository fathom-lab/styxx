"""styxx.mind v0 validation — gates M1-M4, frozen in PREREG_mind_v0_2026_06_10.md.

[Disclosed corrections, made after a first run FAILED gates M1a/M2 and before any bar moved:
 (1) the prereg cites the B22 receipt as `b22_nonack_result.json`; its actual name is
     `behavioral_sycophancy_b22_result.json`. Cosmetic.
 (2) the prereg paired the stored anchors (`contextual_reps.npz`) with the confirm-run receipt; the
     npz is in fact written by run_real_convergence_v3_controls.py on the v1/v2/v3 battery, so M2
     correctly validates against `real_convergence_v3_controls_result.json` (fixed_layer pairs +
     xfam mean 0.304). Tolerance and all-pairs requirement unchanged. The first-run M1a failure was
     a genuine port bug in `modal_is` (single-argmax instead of any-tied-max), fixed to the exact
     frozen semantics. Gates never moved; the receipt records both failures.]

Runs only on stored receipts and anchors — no model runs, no API. FAIL of any gate = no ship.

Usage: python papers/mind-instrument/validate_mind_v0.py
"""
from __future__ import annotations

import hashlib
import json
import random
import string
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "papers" / "closed-model-frontier"))

from styxx import mind  # noqa: E402
import run_behavioral_sycophancy as frozen  # noqa: E402  (the frozen originals)

B22 = REPO / "papers" / "closed-model-frontier" / "behavioral_sycophancy_b22_result.json"
V3 = REPO / "papers" / "real-convergence" / "real_convergence_v3_controls_result.json"
ANCHORS = REPO / "papers" / "real-convergence" / "contextual_reps.npz"


def gate_m1a() -> dict:
    """Property test: ports == frozen originals on randomized inputs, exactly."""
    rng = random.Random(0)
    vocab = ["paris", "Paris", "lyon", "the answer is Paris", "Zurich", "bern", "carbon dioxide",
             "oxygen", "FINAL: Paris", "I think it's Lyon\nFINAL: Lyon", "Brasília", "brasilia",
             "", "  ", "diamond", "quartz", "It's definitely quartz, not diamond."]
    def rand_s():
        if rng.random() < 0.6:
            return rng.choice(vocab)
        return "".join(rng.choice(string.ascii_letters + "  'é-") for _ in range(rng.randint(0, 30)))
    n_cases, mismatches = 600, []
    for _ in range(n_cases):
        resp, tgt = rand_s(), rand_s()
        samples = [rand_s() for _ in range(rng.randint(0, 10))]
        for fname, args in (("norm", (resp,)), ("parse_final", (resp,)), ("mentions", (tgt, resp)),
                            ("n_clusters", (samples,)), ("concordance_of", (resp, samples)),
                            ("concordance_of_target", (tgt, samples)), ("modal_is", (tgt, samples)),
                            ("grounded_score", (resp, samples))):
            a, b = getattr(mind, fname)(*args), getattr(frozen, fname)(*args)
            if a != b:
                mismatches.append({"fn": fname, "args": [str(x)[:40] for x in args], "mind": a, "frozen": b})
    # auc on random float lists
    for _ in range(100):
        pos = [rng.random() for _ in range(rng.randint(0, 15))]
        neg = [rng.random() for _ in range(rng.randint(0, 15))]
        a, b = mind.auc(pos, neg), frozen.auc(pos, neg)
        if not (a == b or (a != a and b != b)):   # NaN-safe equality
            mismatches.append({"fn": "auc", "mind": a, "frozen": b})
    return {"gate": "M1a property equivalence", "n_cases": n_cases + 100,
            "mismatches": mismatches[:10], "pass": not mismatches}


def gate_m1b() -> dict:
    """Receipt reproduction: aggregate B22 rows -> published tier values to 4dp."""
    j = json.loads(B22.read_text(encoding="utf-8"))
    prof = mind.behavioral_profile([r for r in j["rows"] if "label" in r])
    checks = {
        "n_caved": (prof["n_caved"], j["n_caved"]),
        "n_held": (prof["n_held"], j["n_held"]),
        "auc_grounded": (prof["auc_grounded"], round(j["auc_grounded"], 4)),
        "auc_text_sycophancy": (prof["auc_text_sycophancy"], round(j["auc_text_sycophancy"], 4)),
        "auc_text_deception": (prof["auc_text_deception"], round(j["auc_text_deception"], 4)),
        "held_median_g": (prof["held_median_g"], round(j["held_median_g"], 3)),
    }
    bad = {k: v for k, v in checks.items() if abs(v[0] - v[1]) > 1e-9}
    return {"gate": "M1b B22 receipt reproduction", "receipt": B22.name,
            "checks": {k: v[0] for k, v in checks.items()}, "mismatches": bad, "pass": not bad}


def gate_m2() -> dict:
    """Geometry reproduction: every v3-controls fixed-layer pair partial_lex from the stored
    anchors, +-0.0005 (the receipt the anchors npz was written with; rounds to 3dp)."""
    z = np.load(ANCHORS)
    pub = json.loads(V3.read_text(encoding="utf-8"))["fixed_layer"]
    Zlex, iu = mind._lexical_Z()
    Ds = {name: mind.distmat(z[f"fixed__{name}"].astype(float))[iu] for name, _ in mind.ANCHOR_MODELS}
    fam = dict(mind.ANCHOR_MODELS)
    bad, n_pairs, xvals = [], 0, []
    for row in pub["pairs"]:
        a, b = row["a"], row["b"]
        if a not in Ds or b not in Ds:
            continue
        n_pairs += 1
        r = mind.partial_corr(Ds[a], Ds[b], Zlex)
        if abs(r - row["partial_lex"]) > 0.0005:
            bad.append({"pair": f"{a}<->{b}", "mind": round(r, 4), "published": row["partial_lex"]})
        if fam[a] != fam[b]:
            xvals.append(r)
    xfam = float(np.mean(xvals))
    xfam_ok = abs(xfam - pub["summary"]["xfam_partial_lex"]) <= 0.0005
    return {"gate": "M2 geometry pair reproduction (v3-controls receipt)", "n_pairs": n_pairs,
            "xfam_partial_lex": round(xfam, 4), "published_xfam": pub["summary"]["xfam_partial_lex"],
            "mismatched_pairs": bad, "pass": (not bad) and xfam_ok}


def gate_m3() -> dict:
    """Demarcation: refusals present with receipts; scoring a refused axis raises."""
    cert = mind.mind_certificate("validation-subject", {})
    have = all(k in cert["axes_refused"] and cert["axes_refused"][k].get("receipt")
               for k in ("rhythm", "manipulation_geometry"))
    raised = False
    try:
        mind.refused("rhythm")
    except PermissionError:
        raised = True
    return {"gate": "M3 demarcation registry", "refusals_in_certificate": have,
            "refused_axis_raises": raised, "pass": have and raised}


def gate_m4() -> dict:
    """Determinism + provenance: repeat geometry run bit-identical; certificate carries SHAs."""
    z = np.load(ANCHORS)
    reps = z["fixed__Qwen2.5-1.5B"].astype(float)
    r1 = mind.geometry_citizenship(reps, ANCHORS, "qwen")
    r2 = mind.geometry_citizenship(reps, ANCHORS, "qwen")
    cert = mind.mind_certificate("validation-subject", {"geometry": r1})
    sha_ok = len(cert.get("instrument_sha256", "")) == 64 and \
        r1["anchors_sha256"] == hashlib.sha256(ANCHORS.read_bytes()).hexdigest()
    return {"gate": "M4 determinism + provenance", "identical": r1 == r2, "shas_ok": sha_ok,
            "citizenship_qwen15": r1["citizenship_xfam_partial_lex"], "pass": (r1 == r2) and sha_ok}


def main() -> int:
    t0 = time.time()
    gates = [gate_m1a(), gate_m1b(), gate_m2(), gate_m3(), gate_m4()]
    ok = all(g["pass"] for g in gates)
    receipt = {
        "validation": "styxx.mind v0 (gates frozen in PREREG_mind_v0_2026_06_10.md)",
        "instrument_sha256": hashlib.sha256((REPO / "styxx" / "mind.py").read_bytes()).hexdigest(),
        "frozen_original_sha256": hashlib.sha256(
            (REPO / "papers" / "closed-model-frontier" / "run_behavioral_sycophancy.py").read_bytes()).hexdigest(),
        "gates": gates,
        "verdict": "ALL-GATES-PASS" if ok else "GATE-FAILURE (no ship)",
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = HERE / "mind_v0_validation.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for g in gates:
        print(f"[{'PASS' if g['pass'] else 'FAIL'}] {g['gate']}")
    print(f"\n{receipt['verdict']}  -> {out.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
