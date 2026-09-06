"""Mutation coverage for the differential harness: what divergence could it actually see?

Spec: papers/sworn/SPEC_mutation_coverage_v01_2026_09_05.md, frozen before this file ran.

The differential run reported 150000 agreements and zero disagreements. That sentence has two
readings — the implementations agree, or the generator cannot reach where they differ — and the run
cannot tell them apart. This tells them apart, for a catalogue of specific behaviours: it applies
one localised edit to a scratch copy of ONE implementation, runs the standing guard's own
comparison against the mutant, and asks whether the harness raised an alarm.

Caught means a divergence in that place would be visible. Missed means it would not, and the missed
set is the result (M5). Nothing here writes to the tree: every mutant is a temporary file, and the
substitution happens in this process only.

Usage:
    python conformance/sworn/mutation_coverage.py --catalogue <in.json> --out <receipt.json>

The catalogue is a JSON list of {name, side, old, new, why, control, region}. `old` must be an
exact substring of the named implementation; a mutation whose anchor does not match is recorded as
`anchor_missing` and excluded from the denominator, as are mutants that will not load (M1).
"""
from __future__ import annotations

import argparse

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import conformance.sworn.differential as D  # noqa: E402

JS_REAL = ROOT / "styxx" / "_data" / "sworn_verify.js"
PY_REAL = ROOT / "styxx" / "sworn.py"

# The guard's own size and seed. Deliberately NOT the recorded run's seed: the question is what the
# guard that runs on every change can see, and that is the number a reader cares about.
SEED = 20260906
CASES = 5000
BATCH = 2500

# A mutant that raises on more than this share of cases is not measuring detection — the harness
# would "catch" it for a reason that has nothing to do with the format (M1).
DEGENERATE_SHARE = 0.5


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load_python(path: Path):
    """styxx/sworn.py imports nothing from its own package, so a scratch copy loads standalone."""
    spec = importlib.util.spec_from_file_location("sworn_mutant", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(js_path: Path, py_module, n=CASES, batch=BATCH):
    """The guard's own comparison. Returns (disagreements, python_errors, js_errors, compared).

    Raises SystemExit if the node side will not run at all — that is a non-viable mutant, not a
    detection.
    """
    real_js, real_sworn = D.JS, D.sworn
    D.JS, D.sworn = Path(js_path), py_module
    try:
        dis = pyerr = jserr = compared = 0
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            for off in range(0, n, batch):
                b = [D.case(SEED, i) for i in range(off, min(off + batch, n))]
                js = D.js_digests(b, work)
                for c in b:
                    pd, pe, _cen = D.python_digest(c)
                    row = js[c["index"]]
                    compared += 1
                    pyerr += bool(pe)
                    jserr += bool(row.get("error"))
                    if (pd, bool(pe)) != (row.get("digest"), bool(row.get("error"))):
                        dis += 1
        return dis, pyerr, jserr, compared
    finally:
        D.JS, D.sworn = real_js, real_sworn


def run_mutation(m: dict, tmp: Path, baseline_err: dict) -> dict:
    """Apply one mutation to a scratch copy and report what the guard saw."""
    side = m["side"]
    real = JS_REAL if side == "js" else PY_REAL
    original = real.read_bytes().decode("utf-8")
    out = dict(m)

    hits = original.count(m["old"])
    out["anchor_occurrences"] = hits
    if hits == 0:
        out["verdict"] = "anchor_missing"
        return out
    if m["old"] == m["new"]:
        out["verdict"] = "no_op"
        return out

    mutant = tmp / ("mutant.js" if side == "js" else "mutant_sworn.py")
    mutant.write_bytes(original.replace(m["old"], m["new"], 1).encode("utf-8"))

    js_path, py_mod = JS_REAL, D.sworn
    if side == "js":
        js_path = mutant
    else:
        try:
            py_mod = _load_python(mutant)
        except Exception as e:                                    # noqa: BLE001
            out["verdict"] = "non_viable"
            out["detail"] = "the mutant will not import: %s: %s" % (type(e).__name__, str(e)[:200])
            return out

    try:
        dis, pyerr, jserr, compared = compare(js_path, py_mod)
    except SystemExit as e:
        out["verdict"] = "non_viable"
        out["detail"] = "the node side will not run: %s" % str(e)[:200]
        return out
    except Exception as e:                                        # noqa: BLE001
        out["verdict"] = "non_viable"
        out["detail"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        return out

    out["disagreements"] = dis
    out["compared"] = compared
    out["python_errors"] = pyerr
    out["javascript_errors"] = jserr

    # Errors ABOVE the unmutated baseline are what this mutation caused. A mutant that raises on
    # most cases is not a detection of the format, it is a broken file the harness noticed (M1).
    added = (pyerr - baseline_err["python"]) if side == "python" else (jserr - baseline_err["javascript"])
    if added > DEGENERATE_SHARE * compared:
        out["verdict"] = "degenerate"
        out["detail"] = ("the mutant raised on %d of %d cases (%.0f%% above baseline); excluded "
                         "from the denominator" % (added, compared, 100.0 * added / compared))
        return out

    out["verdict"] = "caught" if dis > 0 else "missed"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--out", default=str(HERE / "mutation_coverage.json"))
    ap.add_argument("--cases", type=int, default=CASES)
    a = ap.parse_args(argv)

    out_path = Path(a.out).resolve()
    if out_path.exists():
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(out_path)],
                           capture_output=True)
        if r.returncode == 0:
            print("REFUSED: %s is tracked; a run is history — write a new file (M6)" % out_path.name,
                  file=sys.stderr)
            return 2

    catalogue = json.loads(Path(a.catalogue).read_text(encoding="utf-8"))
    if isinstance(catalogue, dict):
        catalogue = catalogue.get("mutations", [])

    # The unmutated pair first. If the shipped implementations disagree, nothing below means
    # anything and the run says so rather than reporting a detection rate over a broken baseline.
    base_dis, base_pyerr, base_jserr, base_compared = compare(JS_REAL, D.sworn, n=a.cases)
    print("baseline: %d disagreements in %d cases (python raised %d, js raised %d)"
          % (base_dis, base_compared, base_pyerr, base_jserr), flush=True)
    if base_dis != 0:
        print("REFUSED: the shipped pair already disagrees; a detection rate over this baseline "
              "would be meaningless", file=sys.stderr)
        return 2
    baseline_err = {"python": base_pyerr, "javascript": base_jserr}

    results = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, m in enumerate(catalogue, 1):
            res = run_mutation(m, tmp, baseline_err)
            results.append(res)
            print("  [%3d/%3d] %-10s %-9s %-52s %s"
                  % (i, len(catalogue), res["verdict"], res.get("side", "?"),
                     res.get("name", "")[:52],
                     ("%d disagreements" % res["disagreements"]) if "disagreements" in res
                     else res.get("detail", "")[:60]), flush=True)

    verdicts = Counter(r["verdict"] for r in results)
    controls = [r for r in results if r.get("control")]
    mutants = [r for r in results if not r.get("control")]
    viable = [r for r in mutants if r["verdict"] in ("caught", "missed")]
    caught = [r for r in viable if r["verdict"] == "caught"]
    missed = [r for r in viable if r["verdict"] == "missed"]
    controls_viable = [r for r in controls if r["verdict"] in ("caught", "missed")]
    controls_caught = [r for r in controls_viable if r["verdict"] == "caught"]

    by_side, by_region = {}, {}
    for r in viable:
        for bucket, key in ((by_side, r.get("side", "?")), (by_region, r.get("region", "?"))):
            d = bucket.setdefault(key, {"viable": 0, "caught": 0, "missed": 0})
            d["viable"] += 1
            d["caught" if r["verdict"] == "caught" else "missed"] += 1

    gates = {
        "G-M": {"quantity": "viable mutants measured", "value": len(viable), "bar": ">= 25",
                "pass": len(viable) >= 25},
        "G-K": {"quantity": "controls caught", "value": len(controls_caught),
                "controls_viable": len(controls_viable), "bar": "== 0",
                "pass": len(controls_caught) == 0,
                "note": "a caught control VOIDS the run: the guard would be detecting editing, "
                        "not divergence"},
        "G-S": {"quantity": "viable mutants per side",
                "value": {k: v["viable"] for k, v in sorted(by_side.items())}, "bar": ">= 5 each",
                "pass": all(by_side.get(s, {}).get("viable", 0) >= 5 for s in ("js", "python"))},
        "G-R": {"quantity": "regions with at least one viable mutant", "value": len(by_region),
                "bar": ">= 5", "pass": len(by_region) >= 5},
        "G-D": {"quantity": "detection rate, caught / viable",
                "value": {"caught": len(caught), "viable": len(viable),
                          "rate": (round(len(caught) / len(viable), 4) if viable else None)},
                "bar": "none — reported, never passed or failed (M5)", "pass": None},
    }
    void = not gates["G-K"]["pass"]

    receipt = {
        "schema": "styxx.sworn.mutation-coverage/v1",
        "spec": "papers/sworn/SPEC_mutation_coverage_v01_2026_09_05.md",
        "seed": SEED,
        "cases": a.cases,
        "void": void,
        "baseline": {"disagreements": base_dis, "compared": base_compared,
                     "python_errors": base_pyerr, "javascript_errors": base_jserr},
        "implementations": {
            "python": {"module": "styxx/sworn.py", "sha256": _sha(PY_REAL)},
            "javascript": {"module": "styxx/_data/sworn_verify.js", "sha256": _sha(JS_REAL)},
            "note": "content identity modulo newlines, the corpus doctrine",
        },
        "counts": {
            "proposed": len(catalogue),
            "controls": len(controls),
            "controls_caught": len(controls_caught),
            "viable": len(viable),
            "caught": len(caught),
            "missed": len(missed),
            "anchor_missing": verdicts.get("anchor_missing", 0),
            "non_viable": verdicts.get("non_viable", 0),
            "degenerate": verdicts.get("degenerate", 0),
            "no_op": verdicts.get("no_op", 0),
        },
        "gates": gates,
        "by_side": by_side,
        "by_region": by_region,
        # M5: the misses ARE the result. Named in full, never summarised into a rate.
        "missed": [{"name": r["name"], "side": r["side"], "region": r.get("region"),
                    "old": r["old"], "new": r["new"], "why": r.get("why")} for r in missed],
        "mutations": results,
        "reading": ("caught means a divergence in that place would be visible to the standing "
                    "guard at this seed and size; missed means it would not, and a real "
                    "difference between the two implementations could sit there today with every "
                    "agreement number this lab has published looking exactly the same."),
    }
    out_path.write_bytes((json.dumps(receipt, indent=1, sort_keys=True,
                                     ensure_ascii=False) + "\n").encode("utf-8"))
    print("\nviable %d  caught %d  missed %d  (controls %d, caught %d)  -> %s"
          % (len(viable), len(caught), len(missed), len(controls), len(controls_caught),
             out_path.name))
    if void:
        print("VOID: a control was caught", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
