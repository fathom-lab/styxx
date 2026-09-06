"""Classify the recorded disagreements by the first core field on which the two verifiers part.

A count of disagreements is not a finding. What matters is how many DISTINCT ways the two
implementations differ, and which of those changes a verdict rather than a label — a document one
side calls MALFORMED and the other calls HELD is a different order of problem from two spellings of
the same refusal.

Every case is re-derived from the bytes the receipt stores, not from the generator, so this runs
against a receipt alone and would run in a checkout that had never seen `differential.py`.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from styxx import sworn  # noqa: E402

JS = ROOT / "styxx" / "_data" / "sworn_verify.js"
OUTSIDE_CORE = ("verifier", "coverage")

RUNNER = r"""
const fs = require('fs');
const api = require(process.argv[2]);
const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = [];
for (const c of cases) {
  try {
    const doc = new Uint8Array(Buffer.from(c.document_b64, 'base64'));
    const man = c.manifest === null ? null : api.jsonPlain(JSON.stringify(c.manifest));
    const core = api.swornVerify(doc, man, { name: c.name, commit: c.commit });
    out.push({ index: c.index, core: core });
  } catch (e) {
    out.push({ index: c.index, error: String(e && e.message ? e.message : e).slice(0, 200) });
  }
}
fs.writeFileSync(process.argv[4], JSON.stringify(out));
"""


def strip(core):
    return {k: v for k, v in core.items() if k not in OUTSIDE_CORE}


def first_difference(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return path + "/" + k, "<absent>", repr(b[k])[:100]
            if k not in b:
                return path + "/" + k, repr(a[k])[:100], "<absent>"
            d = first_difference(a[k], b[k], path + "/" + k)
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return path, "len %d" % len(a), "len %d" % len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, path + "/%d" % i)
            if d:
                return d
        return None
    if a != b:
        return path, repr(a)[:100], repr(b)[:100]
    return None


def generalise(path: str) -> str:
    """/spans/3/detail/leaf_type -> /spans/*/detail/leaf_type"""
    return "/".join("*" if p.isdigit() else p for p in path.split("/"))


def main(argv=None):
    receipt = Path(argv[0]) if argv else ROOT / "conformance/sworn/differential_agreement_2.json"
    R = json.loads(receipt.read_text(encoding="utf-8"))
    records = R["disagreements"]
    print("receipt: %s" % receipt.name)
    print("disagreements total %d, recorded in full %d\n" % (R["disagree"], len(records)))

    payload = [{"index": r["index"], "name": r["name"], "commit": r["commit"],
                "document_b64": r["document_b64"], "manifest": r["manifest"]} for r in records]

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        runner = work / "r.js"
        runner.write_text(RUNNER, encoding="utf-8")
        inp, outp = work / "in.json", work / "out.json"
        inp.write_bytes(json.dumps(payload).encode("utf-8"))
        res = subprocess.run(["node", str(runner), str(JS), str(inp), str(outp)],
                             capture_output=True, timeout=900)
        if res.returncode != 0 or not outp.exists():
            raise SystemExit("node failed: " + res.stderr.decode("utf-8", "replace")[-400:])
        jrows = {row["index"]: row for row in json.loads(outp.read_text(encoding="utf-8"))}

    classes, verdict_changing, rows = Counter(), Counter(), []
    for r in records:
        doc = base64.b64decode(r["document_b64"])
        man = sworn.Manifest.from_dict(r["manifest"]) if r["manifest"] is not None else None
        try:
            p = strip(sworn.verify(doc, name=r["name"], manifest=man, commit=r["commit"]))
            perr = None
        except BaseException as e:                                # noqa: BLE001
            p, perr = None, type(e).__name__
        j = jrows[r["index"]]
        if perr or j.get("error"):
            classes["one side raised"] += 1
            continue
        jc = strip(j["core"])
        diff = first_difference(p, jc)
        if diff is None:
            classes["cores equal, digests differ (serialisation)"] += 1
            continue
        key = generalise(diff[0])
        classes[key] += 1
        changes_verdict = p["counts"] != jc["counts"] or p["document_verdict"] != jc["document_verdict"]
        if changes_verdict:
            verdict_changing[key] += 1
        rows.append({"index": r["index"], "field": key, "python": diff[1], "javascript": diff[2],
                     "changes_a_verdict": changes_verdict,
                     "python_counts": p["counts"], "javascript_counts": jc["counts"]})

    print("BY FIELD (the first place the cores part):")
    for k, v in classes.most_common():
        vc = verdict_changing.get(k, 0)
        print("  %3d  %-34s  %s" % (v, k, "%d change a VERDICT" % vc if vc else "label only"))
    print()
    print("verdict-changing disagreements: %d of %d recorded"
          % (sum(verdict_changing.values()), len(records)))
    print()
    for row in rows:
        if row["changes_a_verdict"]:
            print("  index %-7s %s" % (row["index"], row["field"]))
            print("     python=%s" % row["python"])
            print("     js    =%s" % row["javascript"])
            print("     counts py=%s" % row["python_counts"])
            print("            js=%s" % row["javascript_counts"])

    out = {
        "schema": "styxx.sworn.disagreement-classes/v1",
        "receipt": str(receipt.relative_to(ROOT)).replace("\\", "/"),
        "disagreements_total": R["disagree"],
        "recorded_in_full": len(records),
        "by_field": dict(classes),
        "verdict_changing_by_field": dict(verdict_changing),
        "verdict_changing_total": sum(verdict_changing.values()),
        "cases": rows,
    }
    dest = ROOT / "papers" / "sworn" / "disagreement_classes.json"
    dest.write_bytes((json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False)
                      + "\n").encode("utf-8"))
    print("\nwrote %s" % dest.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
