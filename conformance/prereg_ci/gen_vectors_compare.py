# -*- coding: utf-8 -*-
"""Generate vectors_compare.json: the compare vectors of the prereg-ci conformance set.

  python -B conformance/prereg_ci/gen_vectors_compare.py           write the file where none is committed
  python -B conformance/prereg_ci/gen_vectors_compare.py --check   regenerate in memory; exit 1 if a byte differs

A new file beside the committed set, which is never written again (a receipt is history too). The
B0 review found compare thinly pinned (the 36 compare vectors of vectors_score.json reach two
codes) and one reading against DESIGN 3.4 C6(i): a receipt recording a gates digest against a
PREREG that reads NOT_GATED had no finding, so a re-run under a moved bar passed when the sealed
PREREG's fence was one the scanner does not count (a BOM before the opener on line 1, a block
quote, a list item) and styxx.protocol.Experiment refuses. These vectors pin every code compare
returns, the three hidden fences beside the visible one, and the readings the review asked for.

Every case is synthetic and states its outcome in advance: whether score_frozen and
Experiment.score agree on it, the part of rescore's reading it turns on, and compare's codes (or
the error compare raises). A case that does not come out as stated is a refusal: printed, exit 1,
nothing written. Reads no history; the bytes of every vector are in the file itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci.gen_vectors import _WORD_RX, _dump  # noqa: E402
from conformance.prereg_ci.replay import (  # noqa: E402
    agree, blob_entry, build_experiment_repo, content_id, experiment_leg, frozen_leg, normal, parse_receipt,
    sha256)
from styxx.prereg_ci import core  # noqa: E402

OUT = ROOT / "conformance" / "prereg_ci" / "vectors_compare.json"
SCHEMA = "styxx.prereg-ci.conformance.compare/v0"
LABEL = "conformance vectors for prereg-ci's compare; no claim"
RESCORE_KEYS = ["form", "code", "verdict", "gates", "row", "gates_sha256", "smoke", "smoke_verdict", "total",
                "holes", "unreachable_rows"]

BOM = chr(0xFEFF)
PROSE = "# PREREG (synthetic)\n\nThe decision is the one fence below.\n\n"
SPEC = {"gates": {"G0": {"metric": "a", "op": ">=", "value": 0.5}},
        "outcomes": [{"when": {"G0": True}, "verdict": "PASS__a"}, {"when": {}, "verdict": "FAIL__a"}],
        "smoke_verdict": "INVALID__smoke"}
EDITED = json.loads(json.dumps(SPEC))
EDITED["gates"]["G0"]["value"] = 0.1                           # the bar moved only in a working copy


def fence(spec: Any) -> str:
    return "```gates\n" + json.dumps(spec, indent=1) + "\n```\n"


def digest(spec: Any) -> str:
    """sha256 of the fence's body, the text the scanner selects and Experiment records."""
    return hashlib.sha256(json.dumps(spec, indent=1).encode("utf-8")).hexdigest()


def rec(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1) + "\n").encode("utf-8")


D, D_EDITED = digest(SPEC), digest(EDITED)
SEALED_TEXT = (PROSE + fence(SPEC)).encode("utf-8")
WORKING_TEXT = (PROSE + fence(EDITED)).encode("utf-8")
POINTER_TEXT = (PROSE + fence(dict(SPEC, receipt_pointer=5))).encode("utf-8")
NO_FENCE = b"# PREREG (synthetic)\n\nNo decision is sealed here; the plan is prose.\n"
HIDDEN = {
    "bom-on-line-1": BOM + fence(SPEC) + "\nThe decision is the fence above.\n",
    "block-quote": PROSE + "".join("> " + ln + "\n" for ln in fence(SPEC).splitlines()),
    "list-item": PROSE + "".join(("- " if i == 0 else "  ") + ln + "\n"
                                 for i, ln in enumerate(fence(SPEC).splitlines())),
}
# a re-run scored under the working-copy bar (what Experiment records for WORKING_TEXT and {"a": 0.3}),
# written to a new receipt that carries its own record, as DESIGN 4.A's A.2 writes it
RERUN = {"a": 0.3, "verdict": "PASS__a", "gates": {"G0": True}, "gates_sha256": D_EDITED}
RERUN_RECORD = {k: RERUN[k] for k in ("verdict", "gates", "gates_sha256")}
OWN = {"verdict": "PASS__a", "gates": {"G0": True}, "gates_sha256": D}

UNSEALED, DIFFERS, NOT_GATED = "DECISION_DIGEST_UNSEALED", "VERDICT_DIFFERS", "NOT_GATED"
DIGEST_ABSENT, VERDICT_ABSENT = "RECORDED_DIGEST_ABSENT", "RECORDED_VERDICT_ABSENT"
REFUSES, AMBIGUOUS = "SCORER_REFUSES", "SMOKE_AMBIGUOUS"


def case(name: str, why: str, prereg: bytes, receipt: Any, agrees: bool, want: dict, compare: Any,
         recorded: Optional[dict] = None, sealed: Any = None, declared: Optional[str] = None,
         experiment: Optional[dict] = None) -> dict:
    c = {"name": name, "why": why, "prereg": prereg, "receipt": rec(receipt), "agree": agrees, "want": want,
         "compare_want": compare, "experiment_want": experiment}
    if recorded is not None:
        c["recorded"] = recorded
    if sealed is not None:
        c["sealed"] = sealed
    if declared is not None:
        c["declared"] = declared
    return c


def cases() -> List[dict]:
    scored = {"code": None, "verdict": "PASS__a", "gates_sha256": D}
    out = [
        # -- a PREREG the scanner reads as holding no decision (DESIGN 3.2(a), 3.4 C6(i))
        case("not-gated/recorded-digest-and-verdict", "a receipt that records a gates digest names a PREREG with "
             "no gates fence: no seal holds a decision digest for it, so C6(i) refuses it",
             NO_FENCE, {"a": 0.3}, True, {"code": NOT_GATED}, [UNSEALED],
             recorded={"verdict": "PASS__a", "gates": {"G0": True}, "gates_sha256": D_EDITED}),
        case("not-gated/recorded-digest-a-sealed-list-names", "a NOT_GATED re-score holds no decision; a sealed "
             "list handed in beside it does not make the recorded digest the PREREG's",
             NO_FENCE, {"a": 0.3}, True, {"code": NOT_GATED}, [UNSEALED],
             recorded={"verdict": "PASS__a", "gates_sha256": D_EDITED}, sealed=[D_EDITED]),
        case("not-gated/recorded-verdict-only", "a prose PREREG's receipt that records a verdict and no digest: "
             "nothing is compared, and a NOTE says the PREREG holds no decision",
             NO_FENCE, {"a": 0.3}, True, {"code": NOT_GATED}, [NOT_GATED], recorded={"verdict": "PASS__a"}),
        case("not-gated/nothing-recorded", "no record and no decision: the NOTE alone",
             NO_FENCE, {"a": 0.3}, True, {"code": NOT_GATED}, [NOT_GATED], recorded={}),
        # -- DESIGN 4.A, A.2 through a fence the scanner does not count and CommonMark renders
        case("a2/the-working-copy-bar", "the working copy holds the bar moved from 0.5 to 0.1; Experiment scores "
             "the receipt PASS__a under its digest, which is the record the re-run carries",
             WORKING_TEXT, {"a": 0.3}, True, {"code": None, "verdict": "PASS__a", "gates_sha256": D_EDITED}, [],
             recorded=RERUN_RECORD, experiment=RERUN_RECORD),
        case("a2/visible-fence", "the sealed PREREG's fence is the plain form: the re-run's digest is unsealed and "
             "the sealed bar scores the receipt FAIL__a",
             SEALED_TEXT, RERUN, True, {"code": None, "verdict": "FAIL__a", "gates_sha256": D},
             [UNSEALED, DIFFERS], recorded=RERUN_RECORD),
    ]
    for label, text in HIDDEN.items():
        out.append(case(
            "a2/hidden-fence/" + label, "the sealed PREREG's fence sits " + label.replace("-", " ") + ": the "
            "scanner counts no fence, so the re-score reads NOT_GATED, and Experiment (styxx.seal's reading) "
            "refuses the PREREG; the re-run's recorded digest is unsealed",
            text.encode("utf-8"), RERUN, True, {"code": NOT_GATED}, [UNSEALED], recorded=RERUN_RECORD))
    out += [
        # -- a re-score that refuses is its own finding
        case("refusal/scorer-refuses", "a metric the receipt does not hold: the refusal passes through",
             SEALED_TEXT, {"b": 1.0}, True, {"code": REFUSES}, [REFUSES], recorded=OWN),
        case("refusal/smoke-ambiguous", "the two smoke conventions disagree: an input error passes through",
             SEALED_TEXT, {"a": 0.9, "smoke": False, "config": {"smoke": True}}, True, {"code": AMBIGUOUS},
             [AMBIGUOUS], recorded=OWN),
        # -- what the record holds (DESIGN 3.2(d), 3.4 C6(i)-(ii))
        case("recorded/all-agree", "verdict, gates and digest equal the re-score", SEALED_TEXT, {"a": 0.9}, True,
             scored, [], recorded=OWN),
        case("recorded/digest-absent", "no recorded digest: (i) is not read, and a NOTE says so", SEALED_TEXT,
             {"a": 0.9}, True, scored, [DIGEST_ABSENT], recorded={"verdict": "PASS__a", "gates": {"G0": True}}),
        case("recorded/verdict-absent-gates-agree", "no recorded verdict: (ii) reads the recorded gates, which "
             "agree", SEALED_TEXT, {"a": 0.9}, True, scored, [VERDICT_ABSENT],
             recorded={"gates": {"G0": True}, "gates_sha256": D}),
        case("recorded/verdict-absent-gates-differ", "no recorded verdict, and the recorded gates are not the "
             "ones the sealed decision fires", SEALED_TEXT, {"a": 0.9}, True, scored, [VERDICT_ABSENT, DIFFERS],
             recorded={"gates": {"G0": False}, "gates_sha256": D}),
        case("recorded/gates-differ-verdict-agrees", "the recorded verdict agrees and the recorded gates do not",
             SEALED_TEXT, {"a": 0.9}, True, scored, [DIFFERS],
             recorded={"verdict": "PASS__a", "gates": {"G0": False}, "gates_sha256": D}),
        case("recorded/verdict-differs", "the recorded verdict is not the sealed decision's", SEALED_TEXT,
             {"a": 0.9}, True, scored, [DIFFERS], recorded={"verdict": "FAIL__a", "gates_sha256": D}),
        case("recorded/nothing", "an empty record: both NOTEs", SEALED_TEXT, {"a": 0.9}, True, scored,
             [DIGEST_ABSENT, VERDICT_ABSENT], recorded={}),
        # -- the sealed digests handed in
        case("sealed/empty-list", "a seal that holds no digest holds none the receipt records", SEALED_TEXT,
             {"a": 0.9}, True, scored, [UNSEALED], recorded=OWN, sealed=[]),
        case("sealed/two-digests-one-held", "the recorded digest is one of the sealed ones", SEALED_TEXT,
             {"a": 0.9}, True, scored, [], recorded=OWN, sealed=[D_EDITED, D]),
        case("sealed/one-digest-as-a-string", "one digest passed as a string would be read character by "
             "character; compare refuses the type", SEALED_TEXT, {"a": 0.9}, True, scored, {"raises": "TypeError"},
             recorded=OWN, sealed=D),
        # -- a RESULT's declared verdict (C6(iii))
        case("declared/agrees", "the RESULT prints the sealed decision's verdict", SEALED_TEXT, {"a": 0.9}, True,
             scored, [], recorded=OWN, declared="PASS__a"),
        case("declared/differs", "the RESULT prints another verdict", SEALED_TEXT, {"a": 0.9}, True, scored,
             [DIFFERS], recorded=OWN, declared="FAIL__a"),
        # -- smoke (DESIGN 3.2(d)) and the table shape a smoke receipt is read under
        case("smoke/recorded-smoke-verdict", "a smoke receipt recording the smoke verdict by type agrees; "
             "score_frozen has no smoke branch and finds no metric",
             SEALED_TEXT, {"smoke": True}, False, {"code": None, "verdict": "INVALID__smoke", "smoke": True},
             [], recorded={"verdict": "INVALID__smoke", "gates": {}, "gates_sha256": D}),
        case("smoke/pointer-not-a-string", "a table whose receipt_pointer is not a string is refused for a smoke "
             "receipt as for a scored one; Experiment reads only the smoke verdict, score_frozen finds no metric",
             POINTER_TEXT, {"smoke": True}, False, {"code": REFUSES}, [REFUSES],
             recorded={"verdict": "INVALID__smoke", "gates": {}}),
        case("smoke/pointer-not-a-string-scored-receipt", "the same table under a scored receipt: the scorers "
             "ignore the key and agree, rescore refuses it", POINTER_TEXT, {"a": 0.9}, True, {"code": REFUSES},
             [REFUSES], recorded={"verdict": "PASS__a"}),
    ]
    return out


def _compare(c: dict) -> Any:
    try:
        got = core.compare(core.rescore(c["prereg"], parse_receipt(c["receipt"])), c.get("recorded", {}),
                           sealed=c.get("sealed"), declared=c.get("declared"))
    except Exception as e:                                  # noqa: BLE001 - the class is the outcome
        return {"raises": type(e).__name__}
    return [{"code": f["code"], "kind": f["kind"]} for f in normal(got)]


def build() -> dict:
    problems: List[str] = []
    cs = cases()
    if len({c["name"] for c in cs}) != len(cs):
        problems.append("two cases share a name")
    for c in cs:
        receipt = parse_receipt(c["receipt"])
        live = normal(core.rescore(c["prereg"], receipt))
        c["expect"] = {"frozen": frozen_leg(c["prereg"], receipt),
                       "rescore": {k: live[k] for k in RESCORE_KEYS}, "compare": _compare(c)}
        for k, want in c["want"].items():
            if live.get(k) != want:
                problems.append(f"{c['name']}: rescore {k} is {live.get(k)!r}, stated {want!r}")
        got = c["expect"]["compare"]
        codes = got if isinstance(got, dict) else [f["code"] for f in got]
        if codes != c["compare_want"]:
            problems.append(f"{c['name']}: compare gives {codes}, stated {c['compare_want']}")
    with tempfile.TemporaryDirectory(prefix="prereg_ci_compare_") as tmp:
        paths = build_experiment_repo(Path(tmp) / "repo", [("PREREG_synthetic.md", c["prereg"]) for c in cs])
        for c, p in zip(cs, paths):
            c["expect"]["experiment"] = experiment_leg(p, parse_receipt(c["receipt"]))
    for c in cs:
        e = c["expect"]
        if agree(e["frozen"], e["experiment"]) != c["agree"]:
            problems.append(f"{c['name']}: stated agree={c['agree']}, the scorers give the other")
        if c["experiment_want"] is not None and e["experiment"] != c["experiment_want"]:
            problems.append(f"{c['name']}: Experiment gives {e['experiment']}, stated {c['experiment_want']}")
        if c["name"].startswith("a2/hidden-fence/") and not (
                e["experiment"].get("at") == "init" and e["frozen"].get("at") == "select"
                and e["frozen"]["message"].startswith("prereg has no ```gates block")):
            problems.append(f"{c['name']}: the scanner and Experiment do not both refuse the hidden fence")
    if problems:
        raise SystemExit("REFUSED\n" + "\n".join(problems))

    blobs: Dict[str, dict] = {}
    vectors = []
    for c in cs:
        inputs: Dict[str, Any] = {}
        for part in ("prereg", "receipt"):
            key = sha256(c[part])
            blobs.setdefault(key, blob_entry(c[part]))
            inputs[part] = key
        for k in ("recorded", "sealed", "declared"):
            if k in c:
                inputs[k] = c[k]
        vectors.append({"id": content_id(c["name"], inputs), "name": c["name"], "why": c["why"],
                        "file": "PREREG_synthetic.md", "inputs": inputs, "agree": c["agree"],
                        "expect": {k: c["expect"][k] for k in ("frozen", "experiment", "rescore", "compare")}})
    return {
        "schema": SCHEMA, "label": LABEL,
        "rules": {
            "id": "sha256 of the sorted, compact UTF-8 JSON of {name, inputs}",
            "blobs": "inputs.prereg and inputs.receipt name bytes in this file's blobs by sha256",
            "receipt": "json.loads over the receipt bytes decoded as UTF-8",
            "frozen": "as vectors_score.json: score_frozen over the selected fence, no newline translation, no "
                      "smoke branch",
            "experiment": "as vectors_score.json: Experiment(<the PREREG committed to a repository>).score("
                          "receipt, smoke=bool(receipt.get('smoke')))",
            "rescore": "styxx.prereg_ci.core.rescore(prereg, receipt), compared on the keys the vector holds",
            "compare": "styxx.prereg_ci.core.compare(rescore, inputs.recorded or {}, sealed=inputs.sealed, "
                       "declared=inputs.declared): a list of {code, kind}, or {raises: <class>}",
            "agree": "frozen and experiment both refuse, or both give the same verdict, gates and digest",
        },
        "vectors": vectors,
        "blobs": dict(sorted(blobs.items())),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="regenerate in memory; exit 1 if a byte differs")
    args = ap.parse_args(argv)
    doc = build()
    # every byte of this file is the generator's own (no carried history), so all of it is scanned
    hits = sorted({m.group(0) for m in _WORD_RX.finditer(json.dumps(doc, ensure_ascii=False))})
    if hits:
        print("REFUSED the file prints a word the lab does not: " + ", ".join(hits))
        return 1
    raw = _dump(doc)
    if args.check or OUT.exists():
        if OUT.exists() and OUT.read_bytes() == raw:
            print(f"SAME {OUT.name}={hashlib.sha256(raw).hexdigest()[:16]}; nothing written")
            return 0
        print(f"DIFFERS {OUT.name}: the run does not regenerate the committed bytes; nothing written. New "
              f"vectors go in new files.")
        return 1
    OUT.write_bytes(raw)
    if OUT.read_bytes() != raw:
        print(f"REFUSED {OUT.name} did not read back as written")
        return 1
    print(f"WROTE {OUT.name} {len(doc['vectors'])} vectors, {len(doc['blobs'])} blobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
