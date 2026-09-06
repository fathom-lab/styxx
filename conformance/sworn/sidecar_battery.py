"""Run an attack battery against the sidecar layer and record what each attack reached.

Spec: papers/sworn/SPEC_sidecar_battery_v01_2026_09_06.md, frozen before the battery was written.

THE THREE STOPS. They carry different guarantees and this corpus had not distinguished them:

    stop 1   load_sidecar(obj)                  the recipient validates and trusts the object
    stop 2   load_sidecar then render(obj)      the recipient renders a document to read or publish
    stop 3   render then verify(bytes)          the recipient re-verifies from scratch

Stop 3 is the honest path and is expected to hold. An attack that only reaches stop 2 is still a
finding, because the capsule story sells a *validated sidecar* and a reader who renders one is at
stop 2 — a document with sworn sentences nobody declared is a document a reader will read.

WHAT "SUCCEEDS" MEANS, decided here rather than per attack. An attack succeeds when the sidecar is
ACCEPTED and the document it renders to disagrees with the sidecar that produced it — i.e. when
re-canonising the rendered bytes yields a different span table than the one validated. That is the
round-trip property `to_sidecar` asserts for the honest direction and `load_sidecar` does not assert
for this one. It is a property of the pair, not a judgement about any individual attack's cleverness.

B3: nothing here repairs anything. The receipt names styxx/sworn.py by content digest at the run,
and a repaired run is a second receipt.
"""
from __future__ import annotations

import argparse
import base64  # noqa: F401  (available to attack builders)
import hashlib
import json
import subprocess
import sys
import textwrap
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from styxx import sworn  # noqa: E402

TARGET = ROOT / "styxx" / "sworn.py"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _build(attack: dict):
    """Compile the attack's builder body into a function and call it."""
    body = attack["builder"]
    # textwrap.dedent + indent, NOT a per-line "add four spaces unless it already has them". The
    # latter collapses a nested `def` onto its own body whenever that body was already indented,
    # which made ten perfectly good attacks fail to parse and be recorded as unrunnable. A harness
    # corrupting its own input is the same defect class as a classifier reading the working tree
    # instead of the bytes a receipt names, and it would have silently shrunk the denominator.
    src = "def _mk():\n" + textwrap.indent(textwrap.dedent(body), "    ")
    ns = {"sworn": sworn, "json": json, "base64": base64, "hashlib": hashlib}
    exec(compile(src, "<attack:%s>" % attack.get("name", "?")[:40], "exec"), ns)  # noqa: S102
    return ns["_mk"]()


def run_attack(attack: dict) -> dict:
    """Execute one attack against all three stops and record the furthest it reached."""
    out = {k: attack.get(k) for k in
           ("name", "surface", "goal", "predicted", "predicted_stop", "why")}

    try:
        side = _build(attack)
    except BaseException as e:                                    # noqa: BLE001
        out["observed"] = "builder_failed"
        out["detail"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        out["traceback"] = traceback.format_exc()[-400:]
        return out
    if not isinstance(side, dict):
        out["observed"] = "builder_failed"
        out["detail"] = "the builder returned %s, not a sidecar dict" % type(side).__name__
        return out

    # ---- stop 1: does load_sidecar accept it? ------------------------------------------------
    try:
        obj = sworn.load_sidecar(side)
        out["stop1_load"] = "accepted"
    except SystemExit as e:
        out["stop1_load"] = "refused"
        out["refusal"] = str(e)[:220]
        out["observed"] = "refused"
        out["reached"] = "none"
        return out
    except BaseException as e:                                    # noqa: BLE001
        # A crash is NOT a refusal. load_sidecar's own docstring promises every check raises the
        # REFUSED SystemExit "and nothing else", so anything else is a defect in its own right.
        out["stop1_load"] = "crashed"
        out["observed"] = "crashed_not_refused"
        out["detail"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        out["reached"] = "1_load"
        return out

    # ---- stop 2: what does render produce? ----------------------------------------------------
    try:
        rendered = sworn.render(obj)
        out["stop2_render"] = "rendered"
        out["rendered_bytes"] = len(rendered)
        out["rendered_b64"] = base64.b64encode(rendered[:2048]).decode("ascii")
    except BaseException as e:                                    # noqa: BLE001
        out["stop2_render"] = "crashed"
        out["observed"] = "crashed_not_refused"
        out["detail"] = "render: %s: %s" % (type(e).__name__, str(e)[:200])
        out["reached"] = "2_render"
        return out

    # The round-trip property: does the rendered document describe the sidecar that made it?
    declared = [dict(s) for s in obj["spans"]]
    try:
        back = sworn.to_sidecar(rendered, obj["document"]["name"], obj["commit"], None)
        out["recanon_spans"] = len(back["spans"])
        out["round_trip"] = (back["spans"] == declared)
    except SystemExit as e:
        out["recanon_spans"] = None
        out["round_trip"] = False
        out["recanon_refused"] = str(e)[:200]
    except BaseException as e:                                    # noqa: BLE001
        out["recanon_spans"] = None
        out["round_trip"] = False
        out["recanon_crashed"] = "%s: %s" % (type(e).__name__, str(e)[:160])
    out["declared_spans"] = len(declared)

    # ---- stop 3: does an independent verify agree? --------------------------------------------
    try:
        man = None
        if isinstance(obj.get("manifest"), dict) and obj["manifest"].get("receipts"):
            try:
                man = sworn.Manifest.from_dict(obj["manifest"])
            except BaseException:                                 # noqa: BLE001
                man = None
        core = sworn.verify(rendered, name=obj["document"]["name"], manifest=man,
                            commit=obj["commit"])
        out["stop3_verify"] = core["document_verdict"]
        out["stop3_counts"] = dict(core["counts"])
        out["stop3_spans"] = len(core["spans"])
    except SystemExit as e:
        out["stop3_verify"] = "REFUSED"
        out["stop3_refusal"] = str(e)[:200]
    except BaseException as e:                                    # noqa: BLE001
        out["stop3_verify"] = "CRASHED"
        out["detail"] = "verify: %s: %s" % (type(e).__name__, str(e)[:200])
        out["observed"] = "crashed_not_refused"
        out["reached"] = "3_verify"
        return out

    # ---- the verdict ---------------------------------------------------------------------------
    # Accepted, and the document it renders disagrees with the sidecar that was validated.
    out["reached"] = "3_verify"
    out["observed"] = "succeeds" if out.get("round_trip") is False else "accepted_and_faithful"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--out", default=str(HERE / "sidecar_battery.json"))
    a = ap.parse_args(argv)

    out_path = Path(a.out).resolve()
    if out_path.exists():
        r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(out_path)],
                           capture_output=True)
        if r.returncode == 0:
            print("REFUSED: %s is tracked; a run is history — write a new file (B3)" % out_path.name,
                  file=sys.stderr)
            return 2

    doc = json.loads(Path(a.battery).read_text(encoding="utf-8"))
    attacks = doc["attacks"] if isinstance(doc, dict) else doc

    # G-P: an entry with no prediction recorded before execution is void.
    void = [x.get("name") for x in attacks if not x.get("predicted")]
    if void:
        print("REFUSED (G-P): %d attacks carry no prediction: %r" % (len(void), void[:5]),
              file=sys.stderr)
        return 2

    results = []
    for i, atk in enumerate(attacks, 1):
        r = run_attack(atk)
        results.append(r)
        agree = "  " if r.get("predicted") == r.get("observed") else "!="
        print("  [%3d/%3d] %s %-22s pred=%-20s %-16s %s"
              % (i, len(attacks), agree, r.get("observed", "?"), r.get("predicted"),
                 r.get("surface", "?")[:16], r.get("name", "")[:44]), flush=True)

    observed = Counter(r["observed"] for r in results)
    succeeded = [r for r in results if r["observed"] == "succeeds"]
    crashed = [r for r in results if r["observed"] == "crashed_not_refused"]
    unrunnable = [r for r in results if r["observed"] == "builder_failed"]
    runnable = [r for r in results if r["observed"] != "builder_failed"]
    surfaces = sorted({r.get("surface") for r in runnable})
    correct = sum(1 for r in runnable if r.get("predicted") == r.get("observed"))

    gates = {
        "G-A": {"quantity": "attacks executed", "value": len(runnable), "bar": ">= 20",
                "pass": len(runnable) >= 20},
        "G-S": {"quantity": "surfaces covered", "value": len(surfaces), "surfaces": surfaces,
                "bar": ">= 4", "pass": len(surfaces) >= 4},
        "G-P": {"quantity": "predictions recorded before execution", "value": len(attacks),
                "bar": "all", "pass": True},
        "G-C": {"quantity": "attacks reaching stop 3 with a broken round trip",
                "value": len(succeeded),
                "bar": "reported; each is a defect of the first order", "pass": None},
        "G-R": {"quantity": "successes repaired with a test watched to fail",
                "value": "deferred to the repair commit (B3 forbids repairing during the run)",
                "bar": "all, or named", "pass": None},
    }

    receipt = {
        "schema": "styxx.sworn.sidecar-battery/v1",
        "spec": "papers/sworn/SPEC_sidecar_battery_v01_2026_09_06.md",
        "target": {"path": "styxx/sworn.py", "sha256": _sha(TARGET.read_bytes())},
        "stops": {
            "1_load": "load_sidecar(obj) alone — the recipient validates and trusts the object",
            "2_render": "load_sidecar then render — the recipient renders a document to read",
            "3_verify": "render then verify — the recipient re-verifies from scratch",
        },
        "success_criterion": ("accepted by load_sidecar AND the rendered document does not "
                              "re-canonise to the span table that was validated"),
        "counts": {
            "proposed": len(attacks), "runnable": len(runnable),
            "builder_failed": len(unrunnable),
            "succeeded": len(succeeded), "crashed_not_refused": len(crashed),
            "refused": observed.get("refused", 0),
            "accepted_and_faithful": observed.get("accepted_and_faithful", 0),
            "predictions_correct": correct,
        },
        "gates": gates,
        "by_surface": {s: {
            "runnable": sum(1 for r in runnable if r.get("surface") == s),
            "succeeded": sum(1 for r in succeeded if r.get("surface") == s),
        } for s in surfaces},
        "succeeded": succeeded,
        "crashed_not_refused": crashed,
        "attacks": results,
        "reading": ("succeeds means load_sidecar accepted a sidecar whose rendered document does "
                    "not describe it — the round-trip property to_sidecar asserts in the honest "
                    "direction and load_sidecar does not assert in this one. Nothing is repaired "
                    "in this run (B3)."),
    }
    out_path.write_bytes((json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=False)
                          + "\n").encode("utf-8"))
    print("\nrunnable %d  succeeded %d  crashed-not-refused %d  refused %d  (predictions right %d/%d)"
          % (len(runnable), len(succeeded), len(crashed), observed.get("refused", 0),
             correct, len(runnable)))
    print("-> %s" % out_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
