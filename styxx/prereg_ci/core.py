# -*- coding: utf-8 -*-
"""styxx.prereg_ci.core: the pure half of prereg-ci.

A sealed decision re-derives a verdict from a receipt, and a recorded verdict is read against
it. Nothing in this module reads a disk, a clock, a process or a network; every function is a
function of the bytes and objects it is handed, so a FAIL built on it is a function of the git
objects those bytes came from.

Public: ``rescore`` and ``compare``. The private names of ``styxx.protocol`` and
``styxx.research_capsule`` it reuses are used here and nowhere else in the package; the
conformance set under ``conformance/prereg_ci/`` pins what they do, vector by vector.

The gates form. A PREREG's decision is its one ```gates fence, selected by
``styxx.protocol._select_gates_block`` (the single scanner and its refusals), over the PREREG
bytes decoded as UTF-8 with newlines read as ``Path.read_text`` reads them: CRLF and a lone CR
become LF. prereg-ci reads blobs and ``styxx.protocol.Experiment`` reads a working copy, so this
is what makes the digest of a fence the one an Experiment recorded for the same file on a box that
checks text out with CRLF. The fence is scored by ``styxx.research_capsule.score_frozen``; any
refusal of the scanner or the scorer reads ``SCORER_REFUSES`` with its text.

Where ``rescore`` departs from ``score_frozen``, and why:

* smoke: ``score_frozen`` has no smoke branch. A receipt whose top-level ``smoke`` is true (read
  as ``styxx.seal`` reads it, ``bool(receipt.get("smoke"))``) is scored by type: the verdict is
  the block's ``smoke_verdict``, gates ``{}``, as ``Experiment.score(smoke=True)`` returns. The
  table's shape is checked before that, which ``Experiment.score(smoke=True)`` does not do. A
  receipt whose ``/config/smoke`` (the capsule's convention) holds another truth value than its
  top-level ``smoke`` (absent reads false) is ``SMOKE_AMBIGUOUS``, an input error.
* the receipt pointer: an optional top-level ``"receipt_pointer"`` key of the gates block, an RFC
  6901 pointer to the object the metrics are read from (default: the whole receipt). The
  existing scorers ignore the key, as they ignore every key they do not read.
* totality: ``score_frozen`` enumerates assignments and gives up past 12 gates (its ``total``
  is None there); ``_cover`` splits cubes instead, has no gate cap, and counts holes and
  landings exactly.
"""
from __future__ import annotations

import hashlib
import math
import numbers
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from styxx import protocol as _protocol
from styxx import research_capsule as _capsule

__all__ = ["rescore", "compare"]

_GateSpecError = _protocol.GateSpecError

# codes; the kind is what a check does with one: FAIL refuses, INPUT is exit 2, NOTE is printed
_NOT_GATED = "NOT_GATED"
_SCORER_REFUSES = "SCORER_REFUSES"
_SMOKE_AMBIGUOUS = "SMOKE_AMBIGUOUS"
_DECISION_DIGEST_UNSEALED = "DECISION_DIGEST_UNSEALED"
_VERDICT_DIFFERS = "VERDICT_DIFFERS"
_RECORDED_DIGEST_ABSENT = "RECORDED_DIGEST_ABSENT"
_RECORDED_VERDICT_ABSENT = "RECORDED_VERDICT_ABSENT"
_KIND = {_NOT_GATED: "NOTE", _SCORER_REFUSES: "FAIL", _SMOKE_AMBIGUOUS: "INPUT",
         _DECISION_DIGEST_UNSEALED: "FAIL", _VERDICT_DIFFERS: "FAIL",
         _RECORDED_DIGEST_ABSENT: "NOTE", _RECORDED_VERDICT_ABSENT: "NOTE"}

# the scanner's own words for a document holding no gates-like fence at all; every other refusal
# of the scanner is a fence that is there and is not read, which is a refusal, never NOT_GATED
_NO_FENCE = "prereg has no ```gates block"
_COMPOSITION_KEYS = ("agg", "over", "excluding")


def _decision_text(prereg: Any) -> str:
    """The PREREG as the scanner reads it: UTF-8, with CRLF and a lone CR read as LF."""
    if isinstance(prereg, (bytes, bytearray)):
        text = bytes(prereg).decode("utf-8")
    elif isinstance(prereg, str):
        text = prereg
    else:
        raise TypeError(f"a PREREG is bytes or str, not {type(prereg).__name__}")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _literal(value: Any) -> Optional[bool]:
    """What a ``when`` literal matches under the scorers' rule ``fired.get(k) == v``, where a fired
    gate is a bool: True, False, or None for a literal no bool equals (the row can never match)."""
    for b in (True, False):
        if b == value:
            return b
    return None


def _cover(gates: Sequence[str], rows: Sequence[Any]) -> Dict[str, Any]:
    """Totality and row reachability of an outcome table by cube splitting, with no gate cap.

    *rows* are the ``when`` objects in table order. A row matches an assignment when every literal
    equals the fired value; the earliest matching row lands. The search keeps the rows still
    compatible with the gates fixed so far; when none is left, every assignment of the subcube is
    a hole; when the earliest compatible row has all its gates fixed, every assignment of the
    subcube lands on it (each earlier row is incompatible); otherwise it splits on that row's
    next unfixed gate. Counts are exact, over ``2 ** len(gates)`` assignments.

    Returns ``{total, holes, hole, landings, unreachable_rows}``: ``hole`` is one assignment of
    every gate that no row matches, or None. Refuses a row that is not an object or that names a
    gate the table does not declare, as ``score_frozen`` does.
    """
    names = list(gates)
    declared = set(names)
    cubes: List[Optional[Dict[str, bool]]] = []
    for when in rows:
        if not isinstance(when, dict):
            raise _GateSpecError("an outcome row's 'when' is not an object")
        unknown = sorted(k for k in when if k not in declared)
        if unknown:
            raise _GateSpecError(f"an outcome row names gates the block does not declare: {unknown}")
        cube: Optional[Dict[str, bool]] = {}
        for k, v in when.items():
            lit = _literal(v)
            if lit is None:
                cube = None
                break
            cube[k] = lit
        cubes.append(cube)
    n = len(names)
    landings = [0] * len(cubes)
    holes = 0
    hole: Optional[Dict[str, bool]] = None
    stack: List[Tuple[Dict[str, bool], List[int]]] = [({}, [i for i, c in enumerate(cubes) if c is not None])]
    while stack:
        fixed, live = stack.pop()
        live = [i for i in live if all(fixed.get(k, v) == v for k, v in cubes[i].items())]
        size = 1 << (n - len(fixed))
        if not live:
            holes += size
            if hole is None:
                hole = {g: fixed.get(g, False) for g in names}
            continue
        lead = cubes[live[0]]
        free = [k for k in lead if k not in fixed]
        if not free:
            landings[live[0]] += size
            continue
        g = free[0]
        stack.append(({**fixed, g: False}, live))
        stack.append(({**fixed, g: True}, live))
    return {"total": holes == 0, "holes": holes, "hole": hole, "landings": landings,
            "unreachable_rows": [i for i, c in enumerate(landings) if c == 0]}


def _finite_bar(value: Any) -> bool:
    """score_frozen's test of a bar; an int too large for a float is not finite here, where
    score_frozen raises OverflowError on it."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _table(spec: Any, smoke: bool) -> Tuple[List[str], List[Any]]:
    """The checks ``score_frozen`` makes before it reads a metric, without a result: the gate
    names and the ``when`` rows, or a refusal in score_frozen's words. With *smoke*, the
    ``smoke_verdict`` must be a string, since it is then the verdict."""
    if not isinstance(spec, dict):
        raise _GateSpecError("gates block is not a JSON object")
    for key in ("gates", "outcomes", "smoke_verdict"):
        if key not in spec:
            raise _GateSpecError(f"gates block missing {key!r}")
    gates, outcomes = spec["gates"], spec["outcomes"]
    if not isinstance(gates, dict) or not gates:
        raise _GateSpecError("gates block 'gates' must be a non-empty object")
    if not isinstance(outcomes, list) or not outcomes:
        raise _GateSpecError("gates block 'outcomes' must be a non-empty list")
    for name, g in gates.items():
        if not isinstance(g, dict):
            raise _GateSpecError(f"gate {name!r} is not an object")
        if g.get("op") not in _protocol._OPS:
            raise _GateSpecError(f"gate {name!r}: unknown op {g.get('op')!r}")
        if not isinstance(g.get("metric"), str) or not g["metric"]:
            raise _GateSpecError(f"gate {name!r}: missing or non-string 'metric' path")
        if not _finite_bar(g.get("value")):
            raise _GateSpecError(f"gate {name!r}: 'value' {g.get('value')!r} is not a finite number")
        if any(k in g for k in _COMPOSITION_KEYS):
            agg, over, exc = g.get("agg"), g.get("over"), g.get("excluding")
            if agg not in ("min", "max") or not isinstance(over, str) or not over:
                raise _GateSpecError(f"gate {name!r}: a composition declaration needs 'agg' min|max and a "
                                     f"non-empty 'over' path")
            if exc is not None and (not isinstance(exc, str) or not exc):
                raise _GateSpecError(f"gate {name!r}: 'excluding' must be a non-empty result path")
    for row in outcomes:
        if (not isinstance(row, dict) or not isinstance(row.get("when"), dict)
                or not isinstance(row.get("verdict"), str)):
            raise _GateSpecError("an outcome row is not {when: {...}, verdict: str}")
        unknown = sorted(k for k in row["when"] if k not in gates)
        if unknown:
            raise _GateSpecError(f"an outcome row names gates the block does not declare: {unknown}")
    if smoke and not isinstance(spec["smoke_verdict"], str):
        raise _GateSpecError(f"'smoke_verdict' {spec['smoke_verdict']!r} is not a string, and a smoke "
                             f"receipt's verdict is the smoke_verdict")
    return list(gates), [row["when"] for row in outcomes]


def _smoke(receipt: Dict[str, Any]) -> Tuple[Optional[bool], Optional[str]]:
    """(smoke, None), or (None, why) when the two conventions disagree."""
    top = bool(receipt.get("smoke"))
    cfg, missing = _capsule._pointer(receipt, "/config/smoke")
    if missing is None and bool(cfg) != top:
        return None, (f"the receipt's top-level smoke reads {top} and its /config/smoke reads {bool(cfg)}: "
                      f"one convention decides, and these two disagree")
    return top, None


def _landed(rows: Sequence[Dict[str, Any]], fired: Dict[str, bool]) -> Optional[int]:
    for i, when in enumerate(rows):
        if all(fired.get(k) == v for k, v in when.items()):
            return i
    return None


def rescore(prereg: Any, receipt: Any) -> Dict[str, Any]:
    """Re-derive the verdict a sealed decision gives a receipt (DESIGN 3.4, C5).

    *prereg* is the PREREG's bytes (a blob) or text; *receipt* is the parsed receipt. Returns a
    dict with every key present: ``form`` ("gates", or None when the PREREG holds no decision),
    ``code`` (None when scored; ``NOT_GATED``, ``SCORER_REFUSES`` or ``SMOKE_AMBIGUOUS``),
    ``detail`` (the refusal's text), ``verdict``, ``gates`` (gate name to fired), ``row`` (the
    index of the landed outcome row; None for a smoke receipt), ``gates_sha256`` (sha256 of the
    selected fence's text, the digest ``Experiment`` and ``score_frozen`` record), ``smoke``,
    ``smoke_verdict``, and the cover of the table: ``total``, ``holes``, ``unreachable_rows``.
    """
    out: Dict[str, Any] = {"form": None, "code": None, "detail": None, "verdict": None, "gates": None,
                           "row": None, "gates_sha256": None, "smoke": None, "smoke_verdict": None,
                           "total": None, "holes": None, "unreachable_rows": None}

    def refuse(detail: str) -> Dict[str, Any]:
        out.update(code=_SCORER_REFUSES, detail=detail)
        return out

    try:
        text = _decision_text(prereg)
    except UnicodeDecodeError as e:
        return refuse(f"the PREREG is not UTF-8: {e}")
    try:
        block = _protocol._select_gates_block(text)
    except _GateSpecError as e:
        if str(e).startswith(_NO_FENCE):
            out.update(code=_NOT_GATED, detail=str(e))
            return out
        return refuse(str(e))
    out["form"] = "gates"
    out["gates_sha256"] = _sha256(block)
    try:
        spec = _capsule._strict_json(block)
    except (ValueError, UnicodeDecodeError) as e:
        return refuse(f"gates block is not valid strict JSON: {e}")
    if not isinstance(receipt, dict):
        return refuse(f"the receipt is not a JSON object (it is a {type(receipt).__name__})")
    smoke, why = _smoke(receipt)
    if smoke is None:
        out.update(code=_SMOKE_AMBIGUOUS, detail=why)
        return out
    out["smoke"] = smoke
    if smoke:
        try:
            names, rows = _table(spec, smoke=True)
        except _GateSpecError as e:
            return refuse(str(e))
        out.update(verdict=spec["smoke_verdict"], gates={}, smoke_verdict=spec["smoke_verdict"])
    else:
        pointer = spec.get("receipt_pointer", "") if isinstance(spec, dict) else ""
        if not isinstance(pointer, str):
            return refuse(f"'receipt_pointer' {pointer!r} is not a string")
        metrics, missing = _capsule._pointer(receipt, pointer)
        if missing is not None:
            return refuse(f"'receipt_pointer' {pointer!r} does not resolve in the receipt: {missing}")
        try:
            scored = _capsule.score_frozen(block, metrics)
        except _GateSpecError as e:
            return refuse(str(e))
        except Exception as e:                      # a scorer error is a refusal with its text (C5)
            return refuse(f"{type(e).__name__}: {e}")
        names = list(spec["gates"])
        rows = [row["when"] for row in spec["outcomes"]]
        out.update(verdict=scored["verdict"], gates=scored["gates"], smoke_verdict=scored["smoke_verdict"],
                   row=_landed(rows, scored["gates"]))
    cover = _cover(names, rows)
    out.update(total=cover["total"], holes=cover["holes"], unreachable_rows=cover["unreachable_rows"])
    return out


def _finding(code: str, detail: str) -> Dict[str, str]:
    return {"code": code, "kind": _KIND[code], "detail": detail}


def compare(rescored: Dict[str, Any], recorded: Optional[Dict[str, Any]] = None,
            sealed: Optional[Iterable[str]] = None, declared: Any = None) -> List[Dict[str, str]]:
    """Read what a receipt records against what its sealed decision gives it (DESIGN 3.4, C6).

    *rescored* is ``rescore``'s output. *recorded* holds what the receipt (or the record that
    carries it) states: ``verdict``, optionally ``gates`` and ``gates_sha256``. *sealed* is every
    decision digest a seal holds for the PREREG (default: the digest ``rescore`` scored under).
    *declared* is the verdict a RESULT prints, when one does. Returns findings in C6's order, each
    ``{code, kind, detail}``; an empty list is agreement on everything read:

    (i) a recorded digest no seal holds is ``DECISION_DIGEST_UNSEALED``; (ii) a recorded verdict,
    or recorded gates, other than the re-score's is ``VERDICT_DIFFERS``; (iii) a RESULT's verdict
    other than the re-score's is ``VERDICT_DIFFERS``. A recorded field that is absent is not read,
    and a NOTE says so. A re-score that refused is returned as its own code.

    A ``NOT_GATED`` re-score holds no decision, so no seal holds a decision digest for it: a
    recorded digest there is ``DECISION_DIGEST_UNSEALED`` (C6(i); the receipt says it was scored
    under a fence the scanner does not read, such as a fence behind a BOM on line 1, in a block
    quote or in a list item, which ``styxx.protocol.Experiment`` refuses too). With no recorded
    digest it is its own NOTE, and a recorded verdict is not read (a prose PREREG's receipt).
    """
    code = rescored.get("code")
    if code == _NOT_GATED:
        recorded = recorded or {}
        if recorded.get("gates_sha256") is not None:
            return [_finding(_DECISION_DIGEST_UNSEALED,
                             f"the receipt records gates {recorded['gates_sha256']}; the PREREG holds no gates "
                             f"fence the scanner reads, so no seal holds a decision digest for it")]
        return [_finding(_NOT_GATED, (rescored.get("detail") or "the PREREG holds no gates fence")
                         + "; nothing recorded is read")]
    if code is not None:
        return [_finding(code, rescored.get("detail") or "")]
    recorded = recorded or {}
    verdict, digest = rescored["verdict"], rescored["gates_sha256"]
    held = list(sealed) if sealed is not None else [digest]
    found: List[Dict[str, str]] = []
    if recorded.get("gates_sha256") is None:
        found.append(_finding(_RECORDED_DIGEST_ABSENT, "the receipt records no gates digest, so (i) is not read"))
    elif recorded["gates_sha256"] not in held:
        found.append(_finding(_DECISION_DIGEST_UNSEALED,
                              f"the receipt records gates {recorded['gates_sha256']}; the sealed PREREG holds "
                              f"{', '.join(held) if held else 'no digest'}"))
    if "verdict" not in recorded:
        found.append(_finding(_RECORDED_VERDICT_ABSENT, "the receipt records no verdict, so (ii) is not read"))
    else:
        parts = []
        if recorded["verdict"] != verdict:
            parts.append(f"the sealed decision scores this receipt {verdict}; the receipt records "
                         f"{recorded['verdict']}")
        if "gates" in recorded and recorded["gates"] != rescored["gates"]:
            parts.append(f"the sealed decision fires {rescored['gates']}; the receipt records {recorded['gates']}")
        if parts:
            found.append(_finding(_VERDICT_DIFFERS, "; ".join(parts)))
    if declared is not None and declared != verdict:
        found.append(_finding(_VERDICT_DIFFERS,
                              f"the sealed decision scores this receipt {verdict}; the RESULT prints {declared}"))
    return found
