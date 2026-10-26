# -*- coding: utf-8 -*-
"""Generate vectors_licence.json: the licence vectors of the prereg-ci conformance set (step B5).

  python -B conformance/prereg_ci/gen_vectors_licence.py           write the file where none is committed
  python -B conformance/prereg_ci/gen_vectors_licence.py --check   regenerate in memory; exit 1 if a byte differs

A new file beside the committed set, which is never written again (a receipt is history too). Each
vector is one call of styxx.prereg_ci.core.licences over a synthetic PREREG, a list of surfaces (a
repository-relative path and the surface's text, or null for one not read) and the verdict a receipt
landed on (null before data); its outcome is the code, every finding as [code, licence, path, line,
col], and every match per licence as [path, line, col, text, withdrawn]. They pin the normalisation
DESIGN 3.2(g) and PLAN K15 name (case, whitespace runs, the curly apostrophes), surface_lint's window
rule over compiled patterns with the withdrawal lying whole inside the window, the three codes
(LICENCE_WITHDRAWN, LICENCE_PENDING, LICENCE_PATTERN_FINDS_NOTHING), the rows that did not land, the
smoke verdict, the locks block's withdraws map, the refusals of the licence format and the order of
findings.

Every case states its outcome in advance: the code, the finding codes in order, and the paths each
licence matches. A case that does not come out as stated is a refusal: printed, exit 1, nothing
written. Reads no history; every byte of a vector is in the file itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci.gen_vectors import _WORD_RX, _dump  # noqa: E402
from conformance.prereg_ci.replay import content_id  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402

OUT = ROOT / "conformance" / "prereg_ci" / "vectors_licence.json"
SCHEMA = "styxx.prereg-ci.conformance.licence/v0"
LABEL = "conformance vectors for prereg-ci's licences; no claim"

F = "`" * 3
LIC = {"id": "ship-claim", "quote": "The model is calibrated well enough to ship.",
       "pattern": "calibrated well enough to ship"}
LIC2 = {"id": "floor-claim", "quote": "Accuracy never drops under the floor.", "pattern": r"never\s+drops"}
APOS = {"id": "apostrophe", "quote": "An agent's PR summary is read before it ships.",
        "pattern": re.escape("agent's pr summary")}
README = ("README.md", "# Demo\n\nThe model is calibrated well enough to ship.\n")
NOTES = ("docs/notes.md", "Nothing here.\n")
SENTENCE = "calibrated well enough to ship"
WITHDRAWN, PENDING, NOTHING = "LICENCE_WITHDRAWN", "LICENCE_PENDING", "LICENCE_PATTERN_FINDS_NOTHING"
REFUSES, NOT_GATED = "SCORER_REFUSES", "NOT_GATED"
PROSE = "# PREREG (synthetic)\n\nThe decision is the one fence below.\n\n"


def table(withdraws: Optional[Dict[str, list]] = None, rows: Any = None) -> dict:
    spec = {"gates": {"G": {"metric": "m", "op": ">=", "value": 0.5}},
            "outcomes": rows or [{"when": {"G": True}, "verdict": "SHIP"}, {"when": {"G": False}, "verdict": "HOLD"}],
            "smoke_verdict": "SMOKE"}
    for verdict, lics in (withdraws or {}).items():
        next(r for r in spec["outcomes"] if r["verdict"] == verdict)["withdraws"] = lics
    return spec


def gates(spec: dict) -> str:
    return f"{PROSE}{F}gates\n{json.dumps(spec, indent=1)}\n{F}\n"


def locks(withdraws: Optional[dict] = None) -> str:
    spec: Dict[str, Any] = {"scorer": "score.py", "reads": [], "outcomes": ["SHIP", "HOLD"]}
    if withdraws is not None:
        spec["withdraws"] = withdraws
    return f"{PROSE}{F}locks\n{json.dumps(spec, indent=1)}\n{F}\n"


def case(name: str, why: str, prereg: str, surfaces: list, landed: Optional[str], code: Optional[str],
         codes: List[str], matched: Dict[str, List[str]]) -> dict:
    return {"name": name, "why": why, "prereg": prereg, "surfaces": [list(s) for s in surfaces], "landed": landed,
            "want": {"code": code, "codes": codes, "matched": matched}}


HOLD = gates(table({"HOLD": [LIC]}))
BOTH = gates(table({"HOLD": [LIC, LIC2]}))


def cases() -> List[dict]:
    return [
        # -- the normalisation (DESIGN 3.2(g), PLAN K15)
        case("normalise/curly-apostrophe", "the surfaces write the apostrophe as U+2019 and as U+2018; the pattern "
             "writes it straight, and the three are read as one character (PLAN K15)",
             gates(table({"HOLD": [APOS]})),
             [("a.md", "An agent’s PR summary is read.\n"), ("b.md", "An agent‘s PR summary is read.\n")],
             None, None, [PENDING, PENDING], {"apostrophe": ["a.md", "b.md"]}),
        case("normalise/case", "the pattern is case-insensitive", HOLD,
             [("a.md", "CALIBRATED WELL ENOUGH TO SHIP\n")], "HOLD", None, [WITHDRAWN], {"ship-claim": ["a.md"]}),
        case("normalise/whitespace-across-lines", "a run of blank lines, spaces and a tab inside the sentence reads "
             "as one space; the match is named at the line and column where it starts", HOLD,
             [("a.md", "calibrated\n\n  well\tenough to ship\n")], "HOLD", None, [WITHDRAWN],
             {"ship-claim": ["a.md"]}),
        # -- the window (surface_lint's rule over compiled patterns)
        case("window/withdrawn-inside", "the sentence that withdraws the claim quotes it: a withdrawal within the "
             "window marks the match withdrawn, so the landed row refuses nothing", HOLD,
             [("a.md", "The sentence 'calibrated well enough to ship' is withdrawn.\n")], "HOLD", None, [],
             {"ship-claim": ["a.md"]}),
        case("window/withdrawn-outside", "a withdrawal more than 300 characters past the match does not reach it",
             HOLD, [("a.md", SENTENCE + ". " + "y" * 300 + " withdrawn\n")], "HOLD", None, [WITHDRAWN],
             {"ship-claim": ["a.md"]}),
        case("window/whole-inside-at-the-edge", "the withdrawal lies whole inside the 300 characters after the "
             "match's end (290 filler characters, a space and the 9 characters of 'withdrawn' end at the edge), "
             "and not when one more filler character moves its end past the edge", HOLD,
             [("edge.md", SENTENCE + "z" * 290 + " withdrawn\n"), ("over.md", SENTENCE + "z" * 291 + " withdrawn\n")],
             "HOLD", None, [WITHDRAWN], {"ship-claim": ["edge.md", "over.md"]}),
        case("window/not-yet-true", "the lab's own withdrawal phrase counts as a withdrawal", HOLD,
             [("a.md", SENTENCE + "; this is not yet true of this instrument.\n")], "HOLD", None, [],
             {"ship-claim": ["a.md"]}),
        # -- the three codes, and the rows that did not land
        case("lint/pattern-finds-nothing", "before data, a licence whose pattern matches no surface is refused (the "
             "analogue of a vacuous gate), beside the pending match of the other licence", BOTH, [README, NOTES],
             None, None, [PENDING, NOTHING], {"ship-claim": ["README.md"], "floor-claim": []}),
        case("pending/before-data", "before any receipt lands, a match is a NOTE: the sentence stands on a verdict "
             "not yet given", HOLD, [README, NOTES], None, None, [PENDING], {"ship-claim": ["README.md"]}),
        case("landed/the-row", "a receipt landed on the row that withdraws the sentence: the match is refused",
             HOLD, [README, NOTES], "HOLD", None, [WITHDRAWN], {"ship-claim": ["README.md"]}),
        case("landed/other-row", "a receipt landed on another row: nothing is refused, and the match is still "
             "listed", HOLD, [README, NOTES], "SHIP", None, [], {"ship-claim": ["README.md"]}),
        case("landed/smoke", "a smoke run lands on no row", HOLD, [README, NOTES], "SMOKE", None, [],
             {"ship-claim": ["README.md"]}),
        # -- the locks form
        case("locks/withdraws-map", "the locks block's withdraws map, read per declared outcome",
             locks({"HOLD": [LIC]}), [README], "HOLD", None, [WITHDRAWN], {"ship-claim": ["README.md"]}),
        case("locks/other-outcome", "the locks block's map for an outcome the scorer did not print",
             locks({"HOLD": [LIC]}), [README], "SHIP", None, [], {"ship-claim": ["README.md"]}),
        # -- surfaces not read, and the order of findings
        case("surfaces/skipped-and-ordered", "a surface not read (binary, a symlink, over the cap) is listed and "
             "scanned on nothing; findings follow licence order, then path, line and column, whatever order the "
             "surfaces came in", BOTH,
             [("b.md", SENTENCE + "\n"), ("a.md", "never drops\n" + SENTENCE + "\n"), ("styxx/bin.py", None)],
             "HOLD", None, [WITHDRAWN, WITHDRAWN, WITHDRAWN],
             {"ship-claim": ["a.md", "b.md"], "floor-claim": ["a.md"]}),
        # -- the refusals
        case("refusal/pattern-does-not-compile", "a pattern that does not compile is the decision's refusal",
             gates(table({"HOLD": [dict(LIC, pattern="(")]})), [README], None, REFUSES, [REFUSES], {}),
        case("refusal/pattern-matches-empty", "a pattern that can match the empty string would match every "
             "surface and license nothing", gates(table({"HOLD": [dict(LIC, pattern="(?=x)")]})), [README], None,
             REFUSES, [REFUSES], {}),
        case("refusal/table-before-licences", "a table rescore refuses (a row names an undeclared gate) is refused "
             "before its licences are read",
             gates(table({"HOLD": [LIC]}, rows=[{"when": {"Z": True}, "verdict": "SHIP"},
                                                   {"when": {"G": False}, "verdict": "HOLD"}])),
             [README], None, REFUSES, [REFUSES], {}),
        case("refusal/no-decision", "a prose PREREG holds no licences; its reading is the NOT_GATED note",
             "# PREREG (synthetic)\n\nNo decision is sealed here; the plan is prose.\n", [README], None, NOT_GATED,
             [NOT_GATED], {}),
    ]


def expect(c: dict) -> dict:
    r = core.licences(c["prereg"], [tuple(s) for s in c["surfaces"]], landed=c["landed"])
    return {"code": r["code"],
            "findings": [[f["code"], f["licence"], f["path"], f["line"], f["col"]] for f in r["findings"]],
            "matches": {li["id"]: [[m["path"], m["line"], m["col"], m["text"], m["withdrawn"]] for m in li["matches"]]
                        for li in r["licences"]}}


def build() -> dict:
    problems: List[str] = []
    cs = cases()
    if len({c["name"] for c in cs}) != len(cs):
        problems.append("two cases share a name")
    vectors = []
    for c in cs:
        e = expect(c)
        want = c["want"]
        if e["code"] != want["code"]:
            problems.append(f"{c['name']}: code is {e['code']!r}, stated {want['code']!r}")
        if [f[0] for f in e["findings"]] != want["codes"]:
            problems.append(f"{c['name']}: findings are {[f[0] for f in e['findings']]}, stated {want['codes']}")
        matched = {k: sorted({m[0] for m in v}) for k, v in e["matches"].items()}
        if matched != want["matched"]:
            problems.append(f"{c['name']}: matched {matched}, stated {want['matched']}")
        inputs = {"prereg": c["prereg"], "surfaces": c["surfaces"], "landed": c["landed"]}
        vectors.append({"id": content_id(c["name"], inputs), "name": c["name"], "why": c["why"], "inputs": inputs,
                        "expect": e})
    if problems:
        raise SystemExit("REFUSED\n" + "\n".join(problems))
    return {
        "schema": SCHEMA, "label": LABEL,
        "rules": {
            "id": "sha256 of the sorted, compact UTF-8 JSON of {name, inputs}",
            "inputs": "prereg is the PREREG's text; surfaces is a list of [path, text], text null for a surface not "
                      "read; landed is the verdict the receipt landed on, null before data",
            "licences": "styxx.prereg_ci.core.licences(prereg, [tuple(s) for s in surfaces], landed=landed)",
            "code": "its code (null where the licences were read)",
            "findings": "its findings in order, each [code, licence, path, line, col]",
            "matches": "per licence id in table order, every match as [path, line, col, text, withdrawn], sorted by "
                       "path, line and column, the withdrawn ones included",
        },
        "vectors": vectors,
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
    print(f"WROTE {OUT.name} {len(doc['vectors'])} vectors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
