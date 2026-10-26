# -*- coding: utf-8 -*-
"""Generate the prereg-ci conformance set: vectors_score.json, vectors_cover.json, blobs.json.

  python -B conformance/prereg_ci/gen_vectors.py           write the set where none is committed
  python -B conformance/prereg_ci/gen_vectors.py --check   regenerate in memory; exit 1 if a byte differs

Reads styxx's own history at REV (``conformance/prereg_ci/__init__.py``) through git plumbing with
replace refs off, read-only, and writes only the three files above. A missing REV (a shallow or
partial clone) exits 3. Before anything is written, every vector is checked:

* the census against the design's receipts: R1 (33 records naming a ``gates_sha256``, 32
  distinct pairs, every record re-scored by ``rescore`` to its recorded verdict and digest), R2
  (the #224 receipt under the blob sealed at 3598e59d, and under the one-bar edit), R8 and R8b
  (52 gated tables, b41 the one that is not total, no row unreachable, the 15-gate leg-2 table
  total where ``research_capsule._table_total`` returns None);
* each synthetic case against what it states in advance: whether ``score_frozen`` and
  ``Experiment.score`` agree on it, and what ``rescore`` reads;
* the cover against brute force on every table up to 15 gates, and against a closed form on the
  tables past it.

A case that does not come out as stated is a refusal: printed, exit 1, nothing written. Once the
set is committed this script never writes it again: a run whose bytes equal the committed files
exits 0, any other exits 1 and names the vectors whose outcome moved. A moved outcome is a finding
about the code, never a reason to rewrite the set, and new vectors go in new files (a receipt is
history too).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci import REV  # noqa: E402
from conformance.prereg_ci.replay import (  # noqa: E402
    agree, blob_entry, brute_force, build_experiment_repo, content_id, experiment_leg, frozen_leg,
    normal, parse_receipt, sha256)
from styxx import protocol  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from styxx.research_capsule import _strict_json, _table_total, score_frozen  # noqa: E402

SET = ROOT / "conformance" / "prereg_ci"
OUT_SCORE, OUT_COVER, OUT_BLOBS = SET / "vectors_score.json", SET / "vectors_cover.json", SET / "blobs.json"
LABEL = "conformance vectors for the gates scorers and the table cover; no claim"
RESCORE_KEYS = ["form", "code", "verdict", "gates", "row", "gates_sha256", "smoke", "smoke_verdict", "total",
                "holes", "unreachable_rows"]
COMPARE_KEYS = ["code", "kind"]

# the design's receipts this set reproduces
R1_RECORDS, R1_PAIRS = 33, 32
R2_PREREG = "papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md"
R2_RECEIPT = "papers/frequency-resonance/pmnist_robustness_result.json"
R2_SEAL = "3598e59d3fe708ebf2dd20992f253d02d753c88b"
R2_SEALED = ("BUDGET_CONFOUND__gap_does_not_survive_3x_steps",
             "578924f8e776573cc82db8fbbd8bb9cf84106b81171a5ba6d5783e422c96e854")
R2_EDITED = ("ROBUST__ROTATION_LOAD_BEARING__beyond_diversity",
             "e513a1da1f8c068ea6ba60138392a031bd13edd2b2539a0f4f0fbe4ab9c61c82")
R2_BAR = '"G_B_long": {"metric": "result.gap_long", "op": ">=", "value": 0.15'
R8_TABLES = 52
R8_NOT_TOTAL = ["papers/disjoint-worlds/PREREG_b41_bridge_2026_08_05.md"]
R8B_LEG2 = ("papers/sworn/PREREG_sworn_measurement_2026_10_23.md", 15, 70)
SMOKE_RECORD = "papers/frequency-resonance/realmodel_bridge_v3_smoke.json"
CENSUS_DIVERGENT = {
    SMOKE_RECORD: ("K1: the receipt is a smoke run. Only Experiment.score (through styxx.seal's reading, "
                   "bool(receipt.get('smoke'))) has a smoke branch; score_frozen reads the receipt's numbers. "
                   "rescore reads it by type, as the record does (R1 scored it so too)."),
}

# words the lab does not print, reversed (as research_capsule._HOUSE_WORDS holds them), and two priority
# phrases; checked over everything this generator writes in its own words
_HOUSE = [w[::-1] for w in ("tsrif", "levon", "yranoitulover", "gnikaerbdnuorg", "hguorhtkaerb", "foorp-repmat",
                            "gniyfirev-fles", "rotceted noitanicullah", "sah eno on", "ylno eht")]
_WORD_RX = re.compile(r"(?<![a-z0-9_])(?:" + "|".join(re.escape(w).replace(r"\-", r"[\s\-]?").replace(r"\ ", r"[\s\-]+")
                                                      for w in _HOUSE) + r")(?![a-z0-9_])", re.IGNORECASE)


class Refusal(Exception):
    pass


# ------------------------------------------------------------------------------------------------ git

def _git(*args: str, data: Optional[bytes] = None) -> bytes:
    env = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1")
    r = subprocess.run(["git", "--no-replace-objects", "-C", str(ROOT), *args], input=data,
                       capture_output=True, env=env, timeout=300)
    if r.returncode != 0:
        raise Refusal(f"git {' '.join(args[:3])} failed: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout


def _blobs(specs: List[str]) -> Dict[str, bytes]:
    """Read many ``<rev>:<path>`` objects through one ``git cat-file --batch``."""
    if not specs:
        return {}
    raw = _git("cat-file", "--batch", data=("\n".join(specs) + "\n").encode("utf-8"))
    out, pos = {}, 0
    for spec in specs:
        nl = raw.index(b"\n", pos)
        header = raw[pos:nl].decode("utf-8").split(" ")
        if len(header) != 3 or header[1] != "blob":
            raise Refusal(f"{spec} is not a blob at this revision ({' '.join(header)})")
        size = int(header[2])
        out[spec] = raw[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
    return out


def _added_at(path: str) -> str:
    """The oldest commit reachable from REV that adds *path* (R1's seal rule, ``--diff-filter=A``)."""
    hs = [h for h in _git("log", REV, "--diff-filter=A", "--format=%H", "--", path).decode().split() if h]
    if not hs:
        raise Refusal(f"no commit reachable from {REV[:8]} adds {path}")
    return hs[-1]


def _gates_digest(blob: bytes) -> str:
    return hashlib.sha256(protocol._select_gates_block(blob.decode("utf-8")).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ census

def census_records() -> List[dict]:
    """R1: every committed record at REV naming a gates_sha256, with its PREREG and receipt."""
    hits = _git("grep", "-l", '"gates_sha256"', REV, "--", "*.json").decode("utf-8").split("\n")
    paths = sorted(h.split(":", 1)[1] for h in hits if h)
    tree = [p for p in _git("ls-tree", "-r", "--name-only", REV).decode("utf-8").split("\n") if p]
    by_base: Dict[str, List[str]] = {}
    for p in tree:
        by_base.setdefault(PurePosixPath(p).name, []).append(p)
    docs = _blobs([f"{REV}:{p}" for p in paths])
    records = []
    for path in paths:
        d = json.loads(docs[f"{REV}:{path}"].decode("utf-8"))
        here = str(PurePosixPath(path).parent)
        if isinstance(d, dict) and isinstance(d.get("protocol"), list):
            for k, pr in enumerate(d["protocol"]):
                if isinstance(pr, dict) and "gates_sha256" in pr:
                    records.append({"record": path, "pointer": f"/protocol/{k}", "near": here,
                                    "prereg": pr.get("prereg"), "receipt": pr.get("result"),
                                    "verdict": pr.get("frozen_verdict"), "gates": pr.get("gates"),
                                    "gates_sha256": pr.get("gates_sha256")})
        elif isinstance(d, dict) and "gates_sha256" in d:
            records.append({"record": path, "pointer": "", "near": here, "prereg": d.get("prereg"),
                            "receipt": path, "verdict": d.get("verdict"), "gates": d.get("gates"),
                            "gates_sha256": d.get("gates_sha256")})

    def locate(name: Any, near: str) -> str:
        if not isinstance(name, str):
            raise Refusal(f"a record names {name!r}, not a path")
        base = PurePosixPath(name.replace("\\", "/")).name
        cands = by_base.get(base, [])
        same = [c for c in cands if str(PurePosixPath(c).parent) == near]
        if len(same) == 1:
            return same[0]
        if len(cands) == 1:
            return cands[0]
        raise Refusal(f"{name!r} near {near} resolves to {len(cands)} files in the tree")

    for r in records:
        r["prereg_path"] = locate(r["prereg"], r["near"])
        r["receipt_path"] = locate(r["receipt"], r["near"])
        r["prereg_commit"] = _added_at(r["prereg_path"])
    blobs = _blobs(sorted({f"{r['prereg_commit']}:{r['prereg_path']}" for r in records}
                          | {f"{REV}:{r['receipt_path']}" for r in records}))
    for r in records:
        r["prereg_bytes"] = blobs[f"{r['prereg_commit']}:{r['prereg_path']}"]
        r["receipt_bytes"] = blobs[f"{REV}:{r['receipt_path']}"]
    return records


def census_tables() -> List[dict]:
    """R8: every PREREG_*.md under papers/ at REV whose gates fence selects and parses."""
    names = [p for p in _git("ls-tree", "-r", "--name-only", REV, "papers").decode("utf-8").split("\n")
             if p.rsplit("/", 1)[-1].startswith("PREREG_") and p.endswith(".md")]
    blobs = _blobs([f"{REV}:{p}" for p in names])
    out = []
    for p in names:
        try:
            block = protocol._select_gates_block(blobs[f"{REV}:{p}"].decode("utf-8", "replace"))
            spec = _strict_json(block)
        except (protocol.GateSpecError, ValueError):
            continue
        out.append({"path": p, "gates_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
                    "gates": list(spec["gates"]), "rows": [row["when"] for row in spec["outcomes"]]})
    return out


# ------------------------------------------------------------------------------------------------ synthetic

PROSE = "# PREREG (synthetic)\n\nThe decision is the one fence below.\n\n"
G = {"G0": {"metric": "a", "op": ">=", "value": 0.5}, "G1": {"metric": "b.c", "op": "<", "value": 0.2}}
ROWS = [{"when": {"G0": False}, "verdict": "INVALID__a_below_floor"},
        {"when": {"G0": True, "G1": True}, "verdict": "PASS"},
        {"when": {"G0": True, "G1": False}, "verdict": "FAIL"}]
SMOKE_V = "INVALID__smoke_plumbing_only"
PASSING = {"a": 0.9, "b": {"c": 0.1}}


def spec(gates: Any = None, outcomes: Any = None, smoke: Any = SMOKE_V, **extra: Any) -> dict:
    out = {"gates": G if gates is None else gates, "outcomes": ROWS if outcomes is None else outcomes,
           "smoke_verdict": smoke}
    out.update(extra)
    return out


def doc(body: Any, prose: str = PROSE, newline: str = "\n", bom: str = "") -> bytes:
    text = body if isinstance(body, str) else json.dumps(body, indent=1)
    out = bom + prose + "```gates\n" + text + "\n```\n"
    return out.replace("\n", newline).encode("utf-8")


def gate(**over: Any) -> dict:
    g = {"metric": "a", "op": ">=", "value": 0.5}
    g.update(over)
    return g


def rec(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1) + "\n").encode("utf-8")


def V(verdict: Any, **more: Any) -> dict:
    out = {"code": None, "verdict": verdict}
    out.update(more)
    return out


def C(code: str) -> dict:
    return {"code": code}


def synthetic_cases() -> List[dict]:
    """Each case: name, why, PREREG bytes, receipt bytes, whether the two scorers agree, and the
    subset of rescore's reading it must give."""
    base = doc(spec())
    fifteen = ["G%02d" % i for i in range(15)]
    cases = [
        # -- the two scorers agree
        ("both/lands-row-0", "the gate on the floor fails; the earliest row lands", base,
         rec({"a": 0.1, "b": {"c": 0.1}}), True, V("INVALID__a_below_floor", row=0)),
        ("both/lands-row-1", "both gates fire", base, rec(PASSING), True, V("PASS", row=1, total=True)),
        ("both/lands-row-2", "G1 does not fire", base, rec({"a": 0.9, "b": {"c": 0.3}}), True, V("FAIL", row=2)),
        ("both/wildcard-last-row", "an empty when matches every assignment",
         doc(spec(outcomes=[{"when": {"G0": True, "G1": True}, "verdict": "PASS"}, {"when": {}, "verdict": "REST"}])),
         rec({"a": 0.1, "b": {"c": 0.1}}), True, V("REST", row=1)),
        ("both/int-metric-int-bar", "integers compare as numbers",
         doc(spec(gates={"G0": gate(metric="n", value=2)},
                  outcomes=[{"when": {"G0": True}, "verdict": "PASS"}, {"when": {"G0": False}, "verdict": "FAIL"}])),
         rec({"n": 3}), True, V("PASS")),
        ("both/when-literal-one-and-zero", "the scorers compare a fired bool with ==, so 1 matches true",
         doc(spec(gates={"G0": gate()}, outcomes=[{"when": {"G0": 1}, "verdict": "ONE"},
                                                  {"when": {"G0": 0}, "verdict": "ZERO"}])),
         rec({"a": 0.9}), True, V("ONE", total=True)),
        ("both/when-literal-string-never-matches", "no bool equals the string 'true'",
         doc(spec(gates={"G0": gate()}, outcomes=[{"when": {"G0": "true"}, "verdict": "NEVER"},
                                                  {"when": {}, "verdict": "REST"}])),
         rec({"a": 0.9}), True, V("REST", unreachable_rows=[0])),
        ("both/extra-keys-ignored", "keys the scorers do not read change no verdict (RV4)",
         doc(spec(gates={"G0": gate(power_basis="none: exploratory", metric_means="the floor"), "G1": G["G1"]},
                  outcomes=[dict(ROWS[0]), dict(ROWS[1], withdraws=[{"id": "x", "quote": "q", "pattern": "p"}]),
                            dict(ROWS[2])], close_after="2026-12-01T00:00:00Z")),
         rec(PASSING), True, V("PASS")),
        ("both/composition-holds", "a declared minimum equals the recomputed one",
         doc(spec(gates={"G0": gate(metric="worst", agg="min", over="per")},
                  outcomes=[{"when": {"G0": True}, "verdict": "PASS"}, {"when": {"G0": False}, "verdict": "FAIL"}])),
         rec({"worst": 0.6, "per": {"x": 0.6, "y": 0.9}}), True, V("PASS")),
        ("both/composition-violation", "a quoted minimum the population does not give is refused",
         doc(spec(gates={"G0": gate(metric="worst", agg="min", over="per")},
                  outcomes=[{"when": {"G0": True}, "verdict": "PASS"}, {"when": {"G0": False}, "verdict": "FAIL"}])),
         rec({"worst": 0.9, "per": {"x": 0.6, "y": 0.9}}), True, C("SCORER_REFUSES")),
        ("both/composition-half-declared", "agg without over is refused (Experiment at init, score_frozen at score)",
         doc(spec(gates={"G0": gate(metric="worst", agg="min")},
                  outcomes=[{"when": {"G0": True}, "verdict": "PASS"}, {"when": {"G0": False}, "verdict": "FAIL"}])),
         rec({"worst": 0.9}), True, C("SCORER_REFUSES")),
        ("both/metric-nan", "a NaN metric is refused", base, rec({"a": float("nan"), "b": {"c": 0.1}}), True,
         C("SCORER_REFUSES")),
        ("both/metric-bool", "a bool metric is refused", base, rec({"a": True, "b": {"c": 0.1}}), True,
         C("SCORER_REFUSES")),
        ("both/metric-string", "a string metric is refused", base, rec({"a": "0.9", "b": {"c": 0.1}}), True,
         C("SCORER_REFUSES")),
        ("both/metric-missing", "a metric path absent from the receipt is refused", base, rec({"b": {"c": 0.1}}),
         True, C("SCORER_REFUSES")),
        ("both/metric-huge-int", "an integer past a float's range raises OverflowError in both scorers",
         base, b'{"a": 1' + b"0" * 400 + b', "b": {"c": 0.1}}\n', True, C("SCORER_REFUSES")),
        ("both/unknown-op", "an op outside >=, <=, >, <, == is refused", doc(spec(gates={"G0": gate(op="=>")},
         outcomes=[{"when": {}, "verdict": "X"}])), rec({"a": 0.9}), True, C("SCORER_REFUSES")),
        ("both/not-total", "no row matches the assignment", doc(spec(gates={"G0": gate()},
         outcomes=[{"when": {"G0": True}, "verdict": "PASS"}])), rec({"a": 0.1}), True, C("SCORER_REFUSES")),
        ("both/empty-outcomes", "an empty table is refused (score_frozen by shape, Experiment as not total)",
         doc(spec(outcomes=[])), rec(PASSING), True, C("SCORER_REFUSES")),
        ("both/duplicate-key", "a key declared twice is refused",
         doc('{"gates": {"G0": {"metric": "a", "op": ">=", "value": 0.5}}, "gates": {"G0": {"metric": "a", '
             '"op": ">=", "value": 0.1}}, "outcomes": [{"when": {}, "verdict": "X"}], "smoke_verdict": "S"}'),
         rec({"a": 0.9}), True, C("SCORER_REFUSES")),
        ("both/non-ascii-key", "a gate name with a Cyrillic letter is refused",
         doc(spec(gates={"G\u04300": gate()}, outcomes=[{"when": {}, "verdict": "X"}])), rec({"a": 0.9}), True,
         C("SCORER_REFUSES")),
        ("both/invalid-json", "a fence that is not JSON is refused", doc('{"gates": }'), rec({"a": 0.9}), True,
         C("SCORER_REFUSES")),
        ("both/missing-smoke-verdict", "the block must name its smoke verdict",
         doc({"gates": G, "outcomes": ROWS}), rec(PASSING), True, C("SCORER_REFUSES")),
        ("both/spec-is-a-list", "the block must be an object", doc("[1, 2]"), rec(PASSING), True,
         C("SCORER_REFUSES")),
        ("both/metric-path-not-a-string", "a gate's metric must be a path",
         doc(spec(gates={"G0": gate(metric=5)}, outcomes=[{"when": {}, "verdict": "X"}])), rec({"a": 0.9}), True,
         C("SCORER_REFUSES")),
        ("both/receipt-not-an-object", "a receipt that is a list is refused (score_frozen finds no metric; "
         "styxx.seal's smoke reading calls .get on it)", base, rec([0.9, 0.1]), True, C("SCORER_REFUSES")),
        ("both/prereg-not-utf8", "PREREG bytes that are not UTF-8 are refused", b"\xff\xfe" + base, rec(PASSING),
         True, C("SCORER_REFUSES")),
        ("both/fifteen-gates", "both scorers read a 15-gate table; score_frozen's own total gives up past 12, "
         "rescore's cover counts it",
         doc(spec(gates={g: gate(metric=g) for g in fifteen},
                  outcomes=[{"when": {g: True}, "verdict": "R" + g} for g in fifteen] + [{"when": {}, "verdict": "REST"}])),
         rec({g: 0.0 for g in fifteen}), True, V("REST", row=15, total=True, holes=0)),
        # -- the fence scanner (styxx.protocol._select_gates_block), shared by both scorers
        ("fence/none", "a document with no gates-like fence holds no decision",
         b"# PREREG\n\nNo decision is sealed here.\n", rec(PASSING), True, C("NOT_GATED")),
        ("fence/two-gates-fences", "two gates-like fences are refused", base + doc(spec(), prose="\n"),
         rec(PASSING), True, C("SCORER_REFUSES")),
        ("fence/tilde", "a tilde fence is gates-like and not the plain form",
         doc(spec()).replace(b"```", b"~~~"), rec(PASSING), True, C("SCORER_REFUSES")),
        ("fence/inside-html-comment", "a fence a renderer hides is refused",
         PROSE.encode() + b"<!--\n```gates\n" + json.dumps(spec()).encode() + b"\n```\n-->\n", rec(PASSING), True,
         C("SCORER_REFUSES")),
        ("fence/after-unterminated-comment", "an unterminated comment hides everything after it",
         PROSE.encode() + b"<!-- \n\n```gates\n" + json.dumps(spec()).encode() + b"\n```\n", rec(PASSING), True,
         C("SCORER_REFUSES")),
        ("fence/never-closed", "an unclosed fence freezes nothing",
         PROSE.encode() + b"```gates\n" + json.dumps(spec()).encode() + b"\n", rec(PASSING), True,
         C("SCORER_REFUSES")),
        ("fence/non-ascii-info", "a confusable info string is refused outright",
         doc(spec()).replace(b"```gates", "```g\u0430tes".encode("utf-8")), rec(PASSING), True,
         C("SCORER_REFUSES")),
        ("fence/three-space-indent", "a rendered fence indented three spaces is gates-like and not plain",
         doc(spec()).replace(b"```gates", b"   ```gates"), rec(PASSING), True, C("SCORER_REFUSES")),
        ("fence/uppercase-info", "GATES casefolds to gates and is not the plain form",
         doc(spec()).replace(b"```gates", b"```GATES"), rec(PASSING), True, C("SCORER_REFUSES")),
        ("fence/tab-indented-opener", "a tab-indented opener is four columns, an indented code block, not a fence; "
         "neither a renderer nor the scanner reads a fence", doc(spec()).replace(b"```gates", b"\t```gates"),
         rec(PASSING), True, C("NOT_GATED")),
        # -- how bytes become text
        ("newline/lf-blob", "the reference for the three readings below", base, rec(PASSING), True, V("PASS")),
        ("newline/crlf-blob", "Experiment reads the working copy with universal newlines; score_frozen fed the "
         "raw decode finds no fence line; rescore translates as Experiment reads, so the digest is the LF one",
         doc(spec(), newline="\r\n"), rec(PASSING), False, V("PASS")),
        ("newline/lone-cr-blob", "a lone CR is a newline to Path.read_text and to rescore, not to a raw decode",
         doc(spec(), newline="\r"), rec(PASSING), False, V("PASS")),
        ("newline/bom-then-prose", "a BOM before the prose moves no fence line", doc(spec(), bom="\ufeff"),
         rec(PASSING), True, V("PASS")),
        ("newline/bom-on-the-fence-line", "a BOM before an opener on line 1 makes that line no fence for the "
         "scanner; Experiment and rescore keep the BOM, so this reads NOT_GATED",
         doc(spec(), prose="", bom="\ufeff"), rec(PASSING), True, C("NOT_GATED")),
        # -- DESIGN K1: where score_frozen and Experiment.score part ways
        ("k1/bar-nan", "score_frozen refuses a bar that is not a finite number; Experiment compares with NaN, "
         "every comparison is false, and the false branch lands",
         doc(spec(gates={"G0": gate(value=float("nan")), "G1": G["G1"]})), rec(PASSING), False,
         C("SCORER_REFUSES")),
        ("k1/bar-infinite", "score_frozen refuses an infinite bar; Experiment compares with it",
         doc(spec(gates={"G0": G["G0"], "G1": gate(metric="b.c", op="<", value=float("inf"))})),
         rec({"a": 0.9, "b": {"c": 0.3}}), False, C("SCORER_REFUSES")),
        ("k1/bar-string-ge", "score_frozen refuses a string bar; Experiment raises TypeError comparing with it "
         "(styxx.seal catches that as a refusal): both refuse",
         doc(spec(gates={"G0": gate(value="0.5"), "G1": G["G1"]})), rec(PASSING), True, C("SCORER_REFUSES")),
        ("k1/bar-string-eq", "score_frozen refuses a string bar; under == Experiment compares a float with a "
         "string, which is false, and lands on the false branch",
         doc(spec(gates={"G0": gate(op="==", value="0.9"), "G1": G["G1"]})), rec(PASSING), False,
         C("SCORER_REFUSES")),
        ("k1/bar-bool", "score_frozen refuses a bool bar; Experiment compares with it as the integer 1",
         doc(spec(gates={"G0": gate(value=True), "G1": G["G1"]})), rec({"a": 1.0, "b": {"c": 0.1}}), False,
         C("SCORER_REFUSES")),
        ("k1/bar-null", "score_frozen refuses a null bar; Experiment raises TypeError: both refuse",
         doc(spec(gates={"G0": gate(value=None), "G1": G["G1"]})), rec(PASSING), True, C("SCORER_REFUSES")),
        ("k1/bar-missing", "score_frozen refuses a gate with no bar; Experiment raises KeyError: both refuse",
         doc(spec(gates={"G0": {"metric": "a", "op": ">="}, "G1": G["G1"]})), rec(PASSING), True,
         C("SCORER_REFUSES")),
        ("k1/bar-huge-int", "score_frozen raises OverflowError turning the bar into a float; Experiment compares "
         "the float with the integer exactly",
         doc('{"gates": {"G0": {"metric": "a", "op": ">=", "value": 1' + "0" * 400 + '}}, "outcomes": '
             '[{"when": {"G0": true}, "verdict": "PASS"}, {"when": {"G0": false}, "verdict": "FAIL"}], '
             '"smoke_verdict": "S"}'), rec({"a": 0.9}), False, C("SCORER_REFUSES")),
        ("k1/row-names-undeclared-gate", "score_frozen refuses a row naming an undeclared gate; Experiment reads "
         "the gate as None, which no true equals, and walks past the row",
         doc(spec(outcomes=[{"when": {"G9": True}, "verdict": "GHOST"}] + ROWS)), rec(PASSING), False,
         C("SCORER_REFUSES")),
        ("k1/row-names-undeclared-gate-null", "Experiment reads the undeclared gate as None, and None == null "
         "lands the row; score_frozen refuses it",
         doc(spec(outcomes=[{"when": {"G9": None}, "verdict": "GHOST"}] + ROWS)), rec(PASSING), False,
         C("SCORER_REFUSES")),
        ("k1/malformed-row-after-the-landed-row", "score_frozen checks every row; Experiment stops at the "
         "landed row and never reads the row with no when", doc(spec(outcomes=ROWS + [{"verdict": "NO_WHEN"}])),
         rec({"a": 0.1, "b": {"c": 0.1}}), False, C("SCORER_REFUSES")),
        ("k1/verdict-not-a-string", "score_frozen refuses a verdict that is not a string; Experiment returns it",
         doc(spec(gates={"G0": gate()}, outcomes=[{"when": {"G0": True}, "verdict": 7},
                                                  {"when": {"G0": False}, "verdict": "FAIL"}])),
         rec({"a": 0.9}), False, C("SCORER_REFUSES")),
        ("k1/empty-gates-wildcard-row", "score_frozen refuses an empty gates object; Experiment lands the "
         "wildcard row", doc(spec(gates={}, outcomes=[{"when": {}, "verdict": "EVERYTHING"}])), rec({}), False,
         C("SCORER_REFUSES")),
        ("k1/non-object-gate", "score_frozen refuses a gate that is not an object; Experiment raises "
         "AttributeError at init: both refuse",
         doc(spec(gates={"G0": 5}, outcomes=[{"when": {}, "verdict": "X"}])), rec({"a": 0.9}), True,
         C("SCORER_REFUSES")),
        ("k1/spec-is-a-string", "a block that is a JSON string passes Experiment's key test by substring and "
         "raises TypeError; score_frozen refuses it as not an object: both refuse",
         doc('"gates outcomes smoke_verdict"'), rec(PASSING), True, C("SCORER_REFUSES")),
        ("k1/when-not-an-object", "score_frozen refuses a when that is a list; Experiment raises AttributeError: "
         "both refuse", doc(spec(gates={"G0": gate()}, outcomes=[{"when": [], "verdict": "X"}])),
         rec({"a": 0.9}), True, C("SCORER_REFUSES")),
        ("k1/row-without-when", "score_frozen refuses a row with no when; Experiment raises KeyError: both refuse",
         doc(spec(gates={"G0": gate()}, outcomes=[{"verdict": "X"}])), rec({"a": 0.9}), True,
         C("SCORER_REFUSES")),
        ("k1/smoke-receipt-with-numbers", "score_frozen has no smoke branch and scores the numbers; Experiment "
         "(styxx.seal's reading) returns the smoke verdict by type; rescore reads it by type",
         base, rec({"smoke": True, "a": 0.9, "b": {"c": 0.1}}), False, V(SMOKE_V, smoke=True, gates={}, row=None)),
        ("k1/smoke-receipt-without-metrics", "score_frozen finds no metric in a smoke receipt; Experiment never "
         "reads one", base, rec({"smoke": True}), False, V(SMOKE_V, smoke=True)),
        # -- prereg-ci's own readings (DESIGN 3.2(a), 3.2(d))
        ("prereg-ci/receipt-pointer", "the sealed pointer names where the metrics are; the existing scorers "
         "ignore the key and read the top level",
         doc(spec(receipt_pointer="/metrics")), rec({"a": 0.1, "b": {"c": 0.1}, "metrics": PASSING}), True,
         V("PASS")),
        ("prereg-ci/receipt-pointer-empty", "the empty pointer is the whole receipt",
         doc(spec(receipt_pointer="")), rec(PASSING), True, V("PASS")),
        ("prereg-ci/receipt-pointer-unresolved", "a sealed pointer the receipt does not hold is refused",
         doc(spec(receipt_pointer="/nope")), rec(PASSING), True, C("SCORER_REFUSES")),
        ("prereg-ci/receipt-pointer-not-a-string", "a pointer must be a string",
         doc(spec(receipt_pointer=5)), rec(PASSING), True, C("SCORER_REFUSES")),
        ("prereg-ci/smoke-false-explicit", "smoke false is not smoke", base, rec(dict(PASSING, smoke=False)),
         True, V("PASS", smoke=False)),
        ("prereg-ci/smoke-config-false-alone", "/config/smoke false agrees with an absent top-level smoke",
         base, rec(dict(PASSING, config={"smoke": False})), True, V("PASS", smoke=False)),
        ("prereg-ci/smoke-config-disagrees", "top-level smoke false, /config/smoke true: one convention decides "
         "and they disagree", base, rec(dict(PASSING, smoke=False, config={"smoke": True})), True,
         C("SMOKE_AMBIGUOUS")),
        ("prereg-ci/smoke-config-without-top-level", "an absent top-level smoke reads false (DESIGN K19), and "
         "/config/smoke true disagrees with it", base, rec(dict(PASSING, config={"smoke": True})), True,
         C("SMOKE_AMBIGUOUS")),
        ("prereg-ci/smoke-both-conventions-true", "both conventions say smoke; score_frozen has no smoke branch",
         base, rec(dict(PASSING, smoke=True, config={"smoke": True})), False, V(SMOKE_V, smoke=True)),
        ("prereg-ci/smoke-truthy-string", "styxx.seal reads smoke with bool(), so the string 'no' is smoke; "
         "rescore reads it so too, to agree with the verdicts seal recorded", base,
         rec(dict(PASSING, smoke="no")), False, V(SMOKE_V, smoke=True)),
        ("prereg-ci/smoke-table-invalid", "rescore checks the table's shape before it reads a smoke receipt by "
         "type; Experiment.score(smoke=True) reads only smoke_verdict; score_frozen refuses the NaN bar",
         doc(spec(gates={"G0": gate(value=float("nan")), "G1": G["G1"]})), rec(dict(PASSING, smoke=True)), False,
         C("SCORER_REFUSES")),
        ("prereg-ci/smoke-verdict-not-a-string", "a smoke receipt's verdict is the smoke_verdict, so it must be a "
         "string; Experiment returns the 5", doc(spec(smoke=5)), rec({"smoke": True}), False,
         C("SCORER_REFUSES")),
    ]
    out = []
    for name, why, prereg, receipt, agrees, want in cases:
        out.append({"name": name, "family": "synthetic", "why": why, "file": "PREREG_synthetic.md",
                    "prereg": prereg, "receipt": receipt, "agree": agrees, "want": want})
    return out


# ------------------------------------------------------------------------------------------------ R2 replays

def r2_cases(census: List[dict]) -> List[dict]:
    seal_blob = _blobs([f"{R2_SEAL}:{R2_PREREG}"])[f"{R2_SEAL}:{R2_PREREG}"]
    record = next(r for r in census if r["receipt_path"] == R2_RECEIPT and r["pointer"] == "")
    if record["prereg_bytes"] != seal_blob:
        raise Refusal("R2: the census reads the #224 PREREG at another blob than the one sealed at 3598e59d")
    text = seal_blob.decode("utf-8")
    if text.count(R2_BAR) != 1:
        raise Refusal("R2: the G_B_long bar is not in the sealed blob once")
    edited = text.replace(R2_BAR, R2_BAR[:-4] + "0.10").encode("utf-8")
    receipt = parse_receipt(record["receipt_bytes"])
    scored = score_frozen(protocol._select_gates_block(edited.decode("utf-8")), receipt)
    rerun = dict(receipt, verdict=scored["verdict"], gates=scored["gates"], gates_sha256=scored["gates_sha256"])
    rerun_bytes = (json.dumps(rerun, indent=1) + "\n").encode("utf-8")
    sealed = [_gates_digest(seal_blob)]
    file = PurePosixPath(R2_PREREG).name
    own = {k: receipt[k] for k in ("verdict", "gates", "gates_sha256")}
    return [
        {"name": "r2/working-copy-bar", "family": "replay", "file": file, "b64": True, "prereg": edited,
         "receipt": record["receipt_bytes"], "agree": True, "want": V(R2_EDITED[0], gates_sha256=R2_EDITED[1]),
         "why": "DESIGN 4.A, A.2: the sealed #224 blob with G_B_long's bar moved from 0.15 to 0.10, as a working "
                "copy would hold it; every leg reads the moved bar (R2)",
         "source": {"rev": REV, "prereg_path": R2_PREREG, "prereg_commit": R2_SEAL, "receipt_path": R2_RECEIPT,
                    "edit": "G_B_long value 0.15 -> 0.10"}},
        {"name": "r2/a2-re-run-under-the-working-copy-bar", "family": "replay", "file": file, "b64": True, "prereg": seal_blob,
         "receipt": rerun_bytes, "agree": True, "want": V(R2_SEALED[0], gates_sha256=R2_SEALED[1]),
         "recorded": {"verdict": rerun["verdict"], "gates": rerun["gates"], "gates_sha256": rerun["gates_sha256"]},
         "sealed": sealed,
         "why": "DESIGN 4.A, A.2: a re-run scored under the working-copy bar and written to a new receipt; the "
                "sealed bytes score it otherwise and hold another digest",
         "source": {"rev": REV, "prereg_path": R2_PREREG, "prereg_commit": R2_SEAL, "receipt_path": R2_RECEIPT,
                    "edit": "the receipt's verdict, gates and gates_sha256 set from the moved bar, written with "
                            "json.dumps(indent=1) as DESIGN 4.A writes it"}},
        {"name": "r2/a1-the-result-types-another-rung", "family": "replay", "file": file, "b64": True, "prereg": seal_blob,
         "receipt": record["receipt_bytes"], "agree": True, "want": V(R2_SEALED[0], gates_sha256=R2_SEALED[1]),
         "recorded": own, "sealed": sealed, "declared": R2_EDITED[0],
         "why": "DESIGN 4.A, A.1: a RESULT prints a rung the receipt did not land on",
         "source": {"rev": REV, "prereg_path": R2_PREREG, "prereg_commit": R2_SEAL, "receipt_path": R2_RECEIPT}},
    ]


# ------------------------------------------------------------------------------------------------ cover

def _pick(rng: random.Random, seq: List[Any]) -> Any:
    return seq[int(rng.random() * len(seq))]


def synthetic_tables() -> List[dict]:
    rng = random.Random(20261024)
    out = [
        ("synthetic/hand/wildcard-only", ["G0"], [{}]),
        ("synthetic/hand/both-polarities", ["G0"], [{"G0": True}, {"G0": False}]),
        ("synthetic/hand/shadowed-row", ["G0", "G1"], [{"G0": True}, {"G0": True, "G1": False}, {}]),
        ("synthetic/hand/dead-literals", ["G0", "G1"], [{"G0": "true"}, {"G1": None}, {"G0": 2}, {"G1": 1}]),
        ("synthetic/hand/b41-shape", ["G0", "G1", "G2", "G3"],
         [{"G0": False}, {"G0": True, "G1": False}, {"G0": True, "G1": True, "G2": True, "G3": True},
          {"G0": True, "G1": True, "G2": False}, {"G0": True, "G1": True, "G2": True, "G3": True}]),
    ]
    for n in range(1, 16):
        gates = ["G%02d" % i for i in range(n)]
        for j in range(6 if n <= 12 else 3):
            style = j % 3
            rows: List[dict] = []
            if style == 0:                                   # random cubes, sparse
                for _ in range(1 + int(rng.random() * min(3 * n, 24))):
                    rows.append({g: rng.random() < 0.5 for g in gates if rng.random() < 0.35})
            elif style == 1:                                 # a decision chain with gaps
                for g in gates:
                    if rng.random() < 0.85:
                        row = {g: rng.random() < 0.5}
                        if rng.random() < 0.3:
                            row[_pick(rng, gates)] = rng.random() < 0.5
                        rows.append(row)
            else:                                            # dense rows, odd literals, repeats
                for _ in range(1 + int(rng.random() * min(2 * n, 20))):
                    if rows and rng.random() < 0.15:
                        rows.append(dict(_pick(rng, rows)))
                        continue
                    row = {}
                    for g in gates:
                        if rng.random() < 0.6:
                            row[g] = _pick(rng, [True, False, True, False, 1, 0, "true", None, 2])
                    rows.append(row)
            if rng.random() < 0.35:
                rows.append({})
            out.append(("synthetic/n%02d/%d" % (n, j), gates, rows))
    return [{"name": n, "family": "synthetic", "gates": g, "rows": r} for n, g, r in out]


def known_tables() -> List[dict]:
    """Tables past brute force whose counts follow from their shape, stated here as closed forms."""
    out = []
    g40 = ["G%02d" % i for i in range(40)]
    chain = [{g: True} for g in g40]
    out.append(("known/chain-40-with-rest", g40, chain + [{}],
                {"holes": 0, "landings": [2 ** (39 - i) for i in range(40)] + [1]}))
    out.append(("known/chain-40-no-rest", g40, chain, {"holes": 1, "landings": [2 ** (39 - i) for i in range(40)]}))
    g64 = ["G%02d" % i for i in range(64)]
    out.append(("known/64-gates-one-literal", g64, [{"G00": True}], {"holes": 2 ** 63, "landings": [2 ** 63]}))
    out.append(("known/64-gates-two-literal-cover", g64, [{"G00": True, "G01": True}, {"G00": False}, {"G01": False}],
                {"holes": 0, "landings": [2 ** 62, 2 ** 63, 2 ** 62]}))
    g100 = ["G%03d" % i for i in range(100)]
    out.append(("known/100-gates-shadowed", g100, [{}, {"G000": True}], {"holes": 0, "landings": [2 ** 100, 0]}))
    tables = []
    for name, gates, rows, form in out:
        expect = {"total": form["holes"] == 0, "holes": form["holes"], "landings": form["landings"],
                  "unreachable_rows": [i for i, c in enumerate(form["landings"]) if c == 0]}
        tables.append({"name": name, "family": "known", "gates": gates, "rows": rows, "expect": expect})
    return tables


# ------------------------------------------------------------------------------------------------ build

def _legs(cases: List[dict]) -> None:
    """Run frozen, rescore and compare on every case, then Experiment in one repository of all."""
    for c in cases:
        receipt = parse_receipt(c["receipt"])
        c["expect"] = {"frozen": frozen_leg(c["prereg"], receipt)}
        live = normal(core.rescore(c["prereg"], receipt))
        c["live_rescore"] = live
        c["expect"]["rescore"] = {k: live[k] for k in RESCORE_KEYS}
        if "recorded" in c or "declared" in c:
            got = core.compare(core.rescore(c["prereg"], receipt), c.get("recorded", {}), sealed=c.get("sealed"),
                               declared=c.get("declared"))
            c["expect"]["compare"] = [{k: f[k] for k in COMPARE_KEYS} for f in normal(got)]
    with tempfile.TemporaryDirectory(prefix="prereg_ci_vectors_") as tmp:
        paths = build_experiment_repo(Path(tmp) / "repo", [(c["file"], c["prereg"]) for c in cases])
        for c, p in zip(cases, paths):
            c["expect"]["experiment"] = experiment_leg(p, parse_receipt(c["receipt"]))


def _check_case(c: dict, problems: List[str]) -> None:
    e = c["expect"]
    got_agree = agree(e["frozen"], e["experiment"])
    if got_agree != c["agree"]:
        problems.append(f"{c['name']}: stated agree={c['agree']}, the scorers give {got_agree}")
    for k, want in c["want"].items():
        if c["live_rescore"].get(k) != want:
            problems.append(f"{c['name']}: rescore {k} is {c['live_rescore'].get(k)!r}, stated {want!r}")
    if c["live_rescore"]["code"] is None and "raises" not in e["experiment"] \
            and c["live_rescore"]["gates_sha256"] != e["experiment"]["gates_sha256"]:
        problems.append(f"{c['name']}: rescore's digest is not the one Experiment records")


def build() -> Tuple[dict, dict, dict]:
    problems: List[str] = []
    census = census_records()
    if len(census) != R1_RECORDS:
        problems.append(f"R1: {len(census)} records, the receipt says {R1_RECORDS}")
    if len({(r["prereg_path"], r["receipt_path"]) for r in census}) != R1_PAIRS:
        problems.append(f"R1: the records are not {R1_PAIRS} distinct pairs")
    cases: List[dict] = []
    for r in census:
        recorded = {"verdict": r["verdict"], "gates_sha256": r["gates_sha256"]}
        if isinstance(r["gates"], dict):
            recorded["gates"] = r["gates"]
        name = "census/" + r["record"] + (f"#{r['pointer']}" if r["pointer"] else "")
        divergent = CENSUS_DIVERGENT.get(r["record"]) if not r["pointer"] else None
        cases.append({"name": name, "family": "census", "file": PurePosixPath(r["prereg_path"]).name,
                      "prereg": r["prereg_bytes"], "receipt": r["receipt_bytes"], "b64": True,
                      "recorded": recorded, "sealed": [_gates_digest(r["prereg_bytes"])],
                      "agree": divergent is None,
                      "want": V(r["verdict"], gates_sha256=r["gates_sha256"]),
                      "why": divergent or "R1: a committed record re-scored from the PREREG blob at the commit "
                                          "that added it",
                      "source": {"rev": REV, "record": r["record"], "pointer": r["pointer"],
                                 "prereg_path": r["prereg_path"], "prereg_commit": r["prereg_commit"],
                                 "receipt_path": r["receipt_path"], "receipt_rev": REV}})
    cases += r2_cases(census)
    cases += synthetic_cases()
    names = [c["name"] for c in cases]
    if len(set(names)) != len(names):
        problems.append("two cases share a name")
    _legs(cases)
    for c in cases:
        _check_case(c, problems)
        if c["family"] in ("census", "replay") and any(f["kind"] == "FAIL" for f in c["expect"].get("compare", [])) \
                and not c["name"].startswith("r2/a"):
            problems.append(f"{c['name']}: compare refuses a record the census re-scores")
    r2 = {c["name"]: c for c in cases}
    a2 = [f["code"] for f in r2["r2/a2-re-run-under-the-working-copy-bar"]["expect"]["compare"]]
    if a2 != ["DECISION_DIGEST_UNSEALED", "VERDICT_DIFFERS"]:
        problems.append(f"R2/A.2: compare reads {a2}")
    a1 = [f["code"] for f in r2["r2/a1-the-result-types-another-rung"]["expect"]["compare"]]
    if a1 != ["VERDICT_DIFFERS"]:
        problems.append(f"R2/A.1: compare reads {a1}")

    tables = [dict(t, brute=True) for t in synthetic_tables()]
    for t in census_tables():
        tables.append({"name": "census/" + t["path"], "family": "census", "gates": t["gates"], "rows": t["rows"],
                       "brute": True, "source": {"rev": REV, "path": t["path"], "gates_sha256": t["gates_sha256"]}})
    for t in tables:
        t["expect"] = brute_force(t["gates"], t["rows"])
    tables += [dict(t, brute=False) for t in known_tables()]
    for t in tables:
        got = core._cover(t["gates"], t["rows"])
        if {k: got[k] for k in t["expect"]} != t["expect"]:
            problems.append(f"{t['name']}: the cover gives {got}, brute force or the closed form {t['expect']}")
        if len(t["gates"]) <= 12 and _table_total(t["gates"], [{"when": w} for w in t["rows"]]) != t["expect"]["total"]:
            problems.append(f"{t['name']}: _table_total disagrees with brute force")
    census_t = [t for t in tables if t["family"] == "census"]
    if len(census_t) != R8_TABLES:
        problems.append(f"R8: {len(census_t)} gated tables, the receipt says {R8_TABLES}")
    if sorted(t["source"]["path"] for t in census_t if not t["expect"]["total"]) != R8_NOT_TOTAL:
        problems.append("R8: the tables that are not total are not b41 alone")
    if any(t["expect"]["unreachable_rows"] for t in census_t):
        problems.append("R8: a census row is unreachable")
    leg2 = [t for t in census_t if t["source"]["path"] == R8B_LEG2[0]]
    if not leg2 or (len(leg2[0]["gates"]), len(leg2[0]["rows"])) != R8B_LEG2[1:] or not leg2[0]["expect"]["total"] \
            or _table_total(leg2[0]["gates"], [{"when": w} for w in leg2[0]["rows"]]) is not None:
        problems.append("R8b: the leg-2 table is not 15 gates, 70 rows, total, with _table_total None")
    if problems:
        raise Refusal("\n".join(problems))

    blobs: Dict[str, dict] = {}
    vectors = []
    for c in cases:
        ids = {}
        for part in ("prereg", "receipt"):
            data = c[part]
            key = sha256(data)
            entry = {"b64": base64.b64encode(data).decode("ascii")} if c.get("b64") else blob_entry(data)
            if key in blobs and blobs[key] != entry:
                entry = blobs[key]
            blobs.setdefault(key, entry)
            ids[part] = key
        inputs = dict(ids)
        for k in ("recorded", "sealed", "declared"):
            if k in c:
                inputs[k] = c[k]
        v = {"id": content_id(c["name"], inputs), "name": c["name"], "family": c["family"], "why": c["why"],
             "file": c["file"], "inputs": inputs}
        if "source" in c:
            v["source"] = c["source"]
        v["agree"] = c["agree"]
        v["expect"] = {"frozen": c["expect"]["frozen"], "experiment": c["expect"]["experiment"],
                       "rescore": c["expect"]["rescore"]}
        if "compare" in c["expect"]:
            v["expect"]["compare"] = c["expect"]["compare"]
        vectors.append(v)
    score_doc = {
        "schema": "styxx.prereg-ci.conformance.score/v0", "label": LABEL, "rev": REV,
        "rules": {
            "id": "sha256 of the sorted, compact UTF-8 JSON of {name, inputs}",
            "blobs": "inputs.prereg and inputs.receipt name bytes in blobs.json by sha256",
            "receipt": "json.loads over the receipt bytes decoded as UTF-8",
            "frozen": "research_capsule.score_frozen(protocol._select_gates_block(prereg.decode('utf-8')), receipt); "
                      "no newline translation, no smoke branch",
            "experiment": "protocol.Experiment(<the PREREG written to disk and committed>).score(receipt, "
                          "smoke=bool(receipt.get('smoke'))), the reading of styxx.seal",
            "rescore": "styxx.prereg_ci.core.rescore(prereg, receipt), compared on the keys the vector holds",
            "compare": "styxx.prereg_ci.core.compare(rescore, inputs.recorded, sealed=inputs.sealed, "
                       "declared=inputs.declared), compared on code and kind",
            "outcome": "{verdict, gates, gates_sha256}, or {raises, at, message}: the class always, the message "
                       "for GateSpecError only",
            "agree": "frozen and experiment both refuse, or both give the same verdict, gates and digest",
        },
        "vectors": vectors,
    }
    cover_doc = {
        "schema": "styxx.prereg-ci.conformance.cover/v0", "label": LABEL, "rev": REV,
        "rules": {
            "match": "a row matches an assignment when every (gate, value) of its when has fired[gate] == value, "
                     "the scorers' expression; the earliest matching row lands",
            "expect": "holes: assignments no row matches; landings[i]: assignments landing on row i; total: holes "
                      "== 0; unreachable_rows: rows with no landing; over 2 ** len(gates) assignments",
            "brute": "true: expect is brute force over every assignment (replay.brute_force); false: expect is the "
                     "closed form the generator states for the table's shape",
        },
        "vectors": [{k: t[k] for k in ("name", "family", "brute", "gates", "rows", "expect", "source") if k in t}
                    for t in tables],
    }
    blobs_doc = {"schema": "styxx.prereg-ci.conformance.blobs/v0",
                 "rule": "every blob hashes to its key; utf8 holds the bytes as UTF-8 text, b64 as base64 (the "
                         "lab's committed census bytes are carried as base64)",
                 "blobs": dict(sorted(blobs.items()))}
    return score_doc, cover_doc, blobs_doc


def _dump(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1, ensure_ascii=True) + "\n").encode("ascii")


def _authored(docs: Tuple[dict, dict, dict]) -> dict:
    """What this generator writes in its own words: every rule and label, every ``why``, and every
    field of a synthetic or closed-form vector. The census and the R2 replays carry the lab's
    history as is (paths, verdicts, gate names, base64 blobs), which research_capsule's carried-
    history rule scans for leaks and never for the house words; one of the lab's directories is
    named with the ordinal."""
    score, cover, blobs = docs
    own = lambda v: v["family"] in ("synthetic", "known")                 # noqa: E731
    return {
        "score": {k: score[k] for k in score if k != "vectors"},
        "cover": {k: cover[k] for k in cover if k != "vectors"},
        "blobs_rule": blobs["rule"],
        "why": [v["why"] for v in score["vectors"]],
        "score_vectors": [v for v in score["vectors"] if own(v)],
        "cover_vectors": [v for v in cover["vectors"] if own(v)],
        "text_blobs": [e["utf8"] for e in blobs["blobs"].values() if "utf8" in e],
    }


def _words(docs: Tuple[dict, dict, dict]) -> List[str]:
    return [m.group(0) for m in _WORD_RX.finditer(json.dumps(_authored(docs), ensure_ascii=False))]


def _moved(old_path: Path, new: dict) -> List[str]:
    """Vectors already committed whose outcome the run no longer gives."""
    if not old_path.exists():
        return []
    old = json.loads(old_path.read_text(encoding="utf-8"))
    now = {v.get("id", v["name"]): v for v in new["vectors"]}
    moved = []
    for v in old["vectors"]:
        key = v.get("id", v["name"])
        if key in now and now[key]["expect"] != v["expect"]:
            moved.append(v["name"])
        if key not in now:
            moved.append(v["name"] + " (no longer produced)")
    return moved


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="regenerate in memory; exit 1 if a byte differs")
    args = ap.parse_args(argv)
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", REV + "^{commit}"], capture_output=True).returncode:
        print(f"UNCHECKABLE {REV} is not in the object store (a shallow or partial clone)")
        return 3
    try:
        docs = build()
    except Refusal as e:
        print("REFUSED\n" + str(e))
        return 1
    raw = [_dump(d) for d in docs]
    hits = _words(docs)
    if hits:
        print("REFUSED the set prints a word the lab does not: " + ", ".join(sorted(set(hits))))
        return 1
    paths = [OUT_SCORE, OUT_COVER, OUT_BLOBS]
    if args.check:
        differ = [p.name for p, b in zip(paths, raw) if not p.exists() or p.read_bytes() != b]
        print(("DIFFERS " + ", ".join(differ)) if differ else "SAME " + " ".join(
            f"{p.name}={hashlib.sha256(b).hexdigest()[:16]}" for p, b in zip(paths, raw)))
        return 1 if differ else 0
    if any(p.exists() for p in paths):
        if all(p.exists() and p.read_bytes() == b for p, b in zip(paths, raw)):
            print("SAME the committed set regenerates byte for byte; nothing written")
            return 0
        moved = _moved(OUT_SCORE, docs[0]) + _moved(OUT_COVER, docs[1])
        print("REFUSED the set is committed and this run differs from it; nothing written. New vectors go in "
              "new files." + ("\nmoved:\n" + "\n".join(moved) if moved else ""))
        return 1
    for p, b in zip(paths, raw):
        p.write_bytes(b)
        if p.read_bytes() != b:
            print(f"REFUSED {p.name} did not read back as written")
            return 1
    s, c = docs[0]["vectors"], docs[1]["vectors"]
    print(f"WROTE vectors_score.json {len(s)} vectors ({sum(1 for v in s if not v['agree'])} divergent), "
          f"vectors_cover.json {len(c)} tables, blobs.json {len(docs[2]['blobs'])} blobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
