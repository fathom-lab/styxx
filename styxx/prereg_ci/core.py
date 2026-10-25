# -*- coding: utf-8 -*-
"""styxx.prereg_ci.core: the pure half of prereg-ci.

A sealed decision re-derives a verdict from a receipt, and a recorded verdict is read against
it. Nothing in this module reads a disk, a clock, a process or a network; every function is a
function of the bytes and objects it is handed, so a FAIL built on it is a function of the git
objects those bytes came from.

Public: ``rescore`` and ``compare``. The private names of ``styxx.protocol``,
``styxx.research_capsule`` and ``styxx.sworn`` it reuses are used here and nowhere else in the
package; the conformance set under ``conformance/prereg_ci/`` pins what they do, vector by vector.

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
  landings exactly. Counting the assignments a set of rows matches is a #DNF count, which is
  #P-complete, so some ordinary-looking tables need exponential work (k rows of the form
  "G2i and G2i+1" visit about 2 ** (k + 2) subcubes). The verdict does not need the cover, so
  ``rescore`` gives the cover a work budget: past it the verdict still stands, ``total``,
  ``holes`` and ``unreachable_rows`` stay None, and ``cover_code`` reads ``COVER_UNCHECKABLE``
  (DESIGN 3.3: UNCHECKABLE, exit 3).

The seals manifest (DESIGN 3.2(b), step B2). ``prereg-ci.seals.jsonl`` holds one JSON line per sealed
file: ``id``, ``kind`` (``prereg``, ``amendment`` or ``withdrawn``), ``path``, ``commit`` (a full id),
``blob_sha256`` (``git show <commit>:<path> | sha256sum``), ``amends``, ``decision`` (``own``: the
line's file carries the fence that then governs; ``inherits``: the decision digest in effect is
unchanged), ``seen`` (reproducible-science's four access levels, or null where a record states
none) and ``time`` (token files, read at step B6). ``_manifest`` parses it strictly and
``_manifest_bytes`` renders it canonically, so a generated manifest round-trips byte for byte.
``_amendment_readings``, ``_in_effect`` and ``_amended_after_data`` read the decision in effect for
a receipt (DESIGN 3.4 C4) from what the git layer hands them: each line's decision, and whether its
commit is a strict ancestor of the receipt's.

The RESULT readers (DESIGN 3.2(c), step B3). ``_fence_at`` is the gates scanner with its info string
as a parameter (PLAN K4: ``styxx.protocol`` hard-codes "gates" and does not change); for "gates" it
decides as ``_select_gates_block`` does, which the conformance set pins on every gates vector.
``_fence_json`` reads a ```result or ```locks fence through it, then refuses a fence CommonMark
renders that the scanner does not count (``_shadow_fences``: behind a byte-order mark, in a block
quote or a list item, or with a second word in the info string), then reads strict JSON.
``_result_reading`` reads a RESULT's ```result fence and its sworn spans on receipts' verdict leaves
(through ``styxx.sworn.scan``, held to the checks sworn makes from bytes alone); ``_declared`` groups
what one RESULT declares per receipt and refuses two forms that disagree.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from styxx import protocol as _protocol
from styxx import research_capsule as _capsule
from styxx import sworn as _sworn

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
_COVER_UNCHECKABLE = "COVER_UNCHECKABLE"
_KIND = {_NOT_GATED: "NOTE", _SCORER_REFUSES: "FAIL", _SMOKE_AMBIGUOUS: "INPUT",
         _DECISION_DIGEST_UNSEALED: "FAIL", _VERDICT_DIFFERS: "FAIL",
         _RECORDED_DIGEST_ABSENT: "NOTE", _RECORDED_VERDICT_ABSENT: "NOTE",
         _COVER_UNCHECKABLE: "UNCHECKABLE"}
# the seals manifest's codes (step B2), in a table of their own: _KIND is compare's and the cover's
_MANIFEST_INVALID = "MANIFEST_INVALID"
_MANIFEST_SEEN_MISSING = "MANIFEST_SEEN_MISSING"
_MANIFEST_DECISION_DIFFERS = "MANIFEST_DECISION_DIFFERS"
_AMENDED_AFTER_RECEIPT = "AMENDED_AFTER_RECEIPT"
_AMENDED_AFTER_DATA = "AMENDED_AFTER_DATA"
_SEAL_KIND = {_MANIFEST_INVALID: "INPUT", _MANIFEST_SEEN_MISSING: "FAIL", _MANIFEST_DECISION_DIFFERS: "FAIL",
              _AMENDED_AFTER_RECEIPT: "NOTE", _AMENDED_AFTER_DATA: "NOTE"}

# rescore's default work budget for the cover, in subcubes visited. The largest committed cover
# vector (synthetic/n15/0: 15 gates, 23 rows) visits 2,591 and the 15-gate leg-2 table fewer; the
# pair-row tables above reach 2 ** 20 near 36 gates, about 2 s on the build box.
_COVER_BUDGET = 1 << 20


class _CoverBudget(Exception):
    """The cover visited more subcubes than its budget allows; its counts are not read."""

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


def _decision(prereg: Any) -> Dict[str, Any]:
    """The decision a PREREG's bytes hold, read as ``rescore`` reads it, without a receipt.

    Returns ``{form, block, gates_sha256, code, detail}``: form "gates" with the selected fence's
    text and its digest; or form None with ``code`` ``NOT_GATED`` (no gates-like fence at all) or
    ``SCORER_REFUSES`` (a fence the scanner refuses, or bytes that are not UTF-8) and the refusal's
    text. The git layer compares these digests along a path's history (the order commit of DESIGN
    3.2(b), and C0); ``rescore`` scores the same block, so the two never read a PREREG two ways.
    """
    out: Dict[str, Any] = {"form": None, "block": None, "gates_sha256": None, "code": None, "detail": None}
    try:
        text = _decision_text(prereg)
    except UnicodeDecodeError as e:
        out.update(code=_SCORER_REFUSES, detail=f"the PREREG is not UTF-8: {e}")
        return out
    try:
        block = _protocol._select_gates_block(text)
    except _GateSpecError as e:
        out.update(code=_NOT_GATED if str(e).startswith(_NO_FENCE) else _SCORER_REFUSES, detail=str(e))
        return out
    out.update(form="gates", block=block, gates_sha256=_sha256(block))
    return out


def _literal(value: Any) -> Optional[bool]:
    """What a ``when`` literal matches under the scorers' rule ``fired.get(k) == v``, where a fired
    gate is a bool: True, False, or None for a literal no bool equals (the row can never match)."""
    for b in (True, False):
        if b == value:
            return b
    return None


def _cover(gates: Sequence[str], rows: Sequence[Any], budget: Optional[int] = None) -> Dict[str, Any]:
    """Totality and row reachability of an outcome table by cube splitting, with no gate cap.

    *rows* are the ``when`` objects in table order. A row matches an assignment when every literal
    equals the fired value; the earliest matching row lands. The search keeps the rows still
    compatible with the gates fixed so far; when none is left, every assignment of the subcube is
    a hole; when the earliest compatible row has all its gates fixed, every assignment of the
    subcube lands on it (each earlier row is incompatible); otherwise it splits on that row's
    next unfixed gate. Counts are exact, over ``2 ** len(gates)`` assignments.

    Returns ``{total, holes, hole, landings, unreachable_rows}``: ``hole`` is one assignment of
    every gate that no row matches, or None. Refuses a row that is not an object or that names a
    gate the table does not declare, as ``score_frozen`` does. With *budget*, raises
    ``_CoverBudget`` once more than *budget* subcubes have been visited (None: no budget).
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
    visited = 0
    while stack:
        fixed, live = stack.pop()
        visited += 1
        if budget is not None and visited > budget:
            raise _CoverBudget(f"the cover visited more than {budget} subcubes of {n} gates and {len(cubes)} "
                               f"rows; totality and reachability are not read")
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
    names and the ``when`` rows, or a refusal in score_frozen's words. prereg-ci's own key,
    ``receipt_pointer``, must be a string when present, so one table reads one way for a smoke
    receipt and for a scored one. With *smoke*, the ``smoke_verdict`` must be a string, since it
    is then the verdict."""
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
    if "receipt_pointer" in spec and not isinstance(spec["receipt_pointer"], str):
        raise _GateSpecError(f"'receipt_pointer' {spec['receipt_pointer']!r} is not a string")
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


def rescore(prereg: Any, receipt: Any, *, cover_budget: Optional[int] = _COVER_BUDGET) -> Dict[str, Any]:
    """Re-derive the verdict a sealed decision gives a receipt (DESIGN 3.4, C5).

    *prereg* is the PREREG's bytes (a blob) or text; *receipt* is the parsed receipt. Returns a
    dict with every key present: ``form`` ("gates", or None when the PREREG holds no decision),
    ``code`` (None when scored; ``NOT_GATED``, ``SCORER_REFUSES`` or ``SMOKE_AMBIGUOUS``),
    ``detail`` (the refusal's text), ``verdict``, ``gates`` (gate name to fired), ``row`` (the
    index of the landed outcome row; None for a smoke receipt), ``gates_sha256`` (sha256 of the
    selected fence's text, the digest ``Experiment`` and ``score_frozen`` record), ``smoke``,
    ``smoke_verdict``, and the cover of the table: ``total``, ``holes``, ``unreachable_rows``.
    ``cover_code`` is None, or ``COVER_UNCHECKABLE`` (with ``cover_detail``) when the cover ran
    past *cover_budget* subcubes; the verdict is read either way. ``cover_budget=None`` runs the
    cover with no budget.
    """
    if cover_budget is not None and (isinstance(cover_budget, bool) or not isinstance(cover_budget, int)
                                     or cover_budget < 1):
        raise ValueError(f"cover_budget is a positive int or None, not {cover_budget!r}")
    out: Dict[str, Any] = {"form": None, "code": None, "detail": None, "verdict": None, "gates": None,
                           "row": None, "gates_sha256": None, "smoke": None, "smoke_verdict": None,
                           "total": None, "holes": None, "unreachable_rows": None, "cover_code": None,
                           "cover_detail": None}

    def refuse(detail: str) -> Dict[str, Any]:
        out.update(code=_SCORER_REFUSES, detail=detail)
        return out

    decision = _decision(prereg)
    if decision["form"] is None:
        out.update(code=decision["code"], detail=decision["detail"])
        return out
    block = decision["block"]
    out["form"] = "gates"
    out["gates_sha256"] = decision["gates_sha256"]
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
    try:
        cover = _cover(names, rows, budget=cover_budget)
    except _CoverBudget as e:
        out.update(cover_code=_COVER_UNCHECKABLE, cover_detail=str(e))
        return out
    out.update(total=cover["total"], holes=cover["holes"], unreachable_rows=cover["unreachable_rows"])
    return out


def _finding(code: str, detail: str) -> Dict[str, str]:
    return {"code": code, "kind": _KIND[code], "detail": detail}


def compare(rescored: Dict[str, Any], recorded: Optional[Dict[str, Any]] = None,
            sealed: Optional[Iterable[str]] = None, declared: Any = None) -> List[Dict[str, str]]:
    """Read what a receipt records against what its sealed decision gives it (DESIGN 3.4, C6).

    *rescored* is ``rescore``'s output. *recorded* holds what the receipt (or the record that
    carries it) states: ``verdict``, optionally ``gates`` and ``gates_sha256``. *sealed* is every
    decision digest a seal holds for the PREREG (default: the digest ``rescore`` scored under), an
    iterable of digests; a single str or bytes is refused with TypeError, since iterating it would
    read its characters as digests.
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
    if isinstance(sealed, (str, bytes, bytearray)):
        raise TypeError(f"sealed is an iterable of digests, not one {type(sealed).__name__}; pass [digest]")
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
                              f"{', '.join(map(str, held)) if held else 'no digest'}"))
    parts = []
    if "verdict" not in recorded:
        found.append(_finding(_RECORDED_VERDICT_ABSENT,
                              "the receipt records no verdict, so (ii) reads its gates only" if "gates" in recorded
                              else "the receipt records no verdict and no gates, so (ii) is not read"))
    elif recorded["verdict"] != verdict:
        parts.append(f"the sealed decision scores this receipt {verdict}; the receipt records {recorded['verdict']}")
    if "gates" in recorded and recorded["gates"] != rescored["gates"]:
        parts.append(f"the sealed decision fires {rescored['gates']}; the receipt records {recorded['gates']}")
    if parts:
        found.append(_finding(_VERDICT_DIFFERS, "; ".join(parts)))
    if declared is not None and declared != verdict:
        found.append(_finding(_VERDICT_DIFFERS,
                              f"the sealed decision scores this receipt {verdict}; the RESULT prints {declared}"))
    return found


# ---- the seals manifest (DESIGN 3.2(b)) ---------------------------------------------------------------

_MANIFEST_KEYS = ("id", "kind", "path", "commit", "blob_sha256", "amends", "decision", "seen", "time")
_MANIFEST_KINDS = ("prereg", "amendment", "withdrawn")
_DECISIONS = ("own", "inherits")
# reproducible-science's four access levels (Tower, 2026), which the design credits to it
_SEEN = ("nothing run", "no results seen", "results not opened", "results seen")
_LINE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_AMEND_AFTER_DATA = ("report", "fail")


class _ManifestError(ValueError):
    """A manifest this module does not read: ``code`` (``MANIFEST_INVALID`` or
    ``MANIFEST_SEEN_MISSING``), its ``kind``, ``detail``, and the 1-based ``line`` (None where the
    file as a whole is refused)."""

    def __init__(self, code: str, detail: str, line: Optional[int] = None):
        super().__init__(f"{code}: " + (f"line {line}: " if line is not None else "") + detail)
        self.code, self.kind, self.detail, self.line = code, _SEAL_KIND[code], detail, line


def _line_finding(code: str, detail: str, line_id: Optional[str]) -> Dict[str, Any]:
    return {"code": code, "kind": _SEAL_KIND[code], "detail": detail, "id": line_id}


def _path_problem(path: Any) -> Optional[str]:
    """Why *path* is not a repository-relative POSIX path with a portable file name, or None
    (``research_capsule``'s rule, which every manifest path and token path is held to)."""
    return _capsule._posix_path_problem(path)


def _full_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) in (40, 64) and _capsule._is_hex(value, len(value))


def _manifest_entry(obj: Any, n: int, ids: Dict[str, Tuple[int, str]], sealed: Dict[str, str]) -> Dict[str, Any]:
    def bad(why: str) -> _ManifestError:
        return _ManifestError(_MANIFEST_INVALID, why, n)

    if not isinstance(obj, dict):
        raise bad(f"is not a JSON object (it is a {type(obj).__name__})")
    unknown = sorted(set(obj) - set(_MANIFEST_KEYS))
    if unknown:
        raise bad(f"holds keys the format does not name: {unknown}")
    kind = obj.get("kind")
    missing = [k for k in _MANIFEST_KEYS if k not in obj]
    if missing == ["seen"] and kind in ("amendment", "withdrawn"):
        raise _ManifestError(_MANIFEST_SEEN_MISSING, "an amendment line states what had been seen (DESIGN 3.2(b)); "
                                                     "this one has no 'seen' key", n)
    if missing:
        raise bad(f"lacks the keys {missing}; every line carries all of {list(_MANIFEST_KEYS)}")
    line_id = obj["id"]
    if not isinstance(line_id, str) or not _LINE_ID.fullmatch(line_id):
        raise bad(f"id {line_id!r} is not 1 to 64 of A-Z, a-z, 0-9, '.', '_', '-' (led by a letter or digit)")
    if line_id in ids:
        raise bad(f"id {line_id!r} is the id of line {ids[line_id][0]}")
    if kind not in _MANIFEST_KINDS:
        raise bad(f"kind {kind!r} is not one of {list(_MANIFEST_KINDS)}")
    problem = _path_problem(obj["path"])
    if problem:
        raise bad(f"path {obj['path']!r} {problem}")
    if not _full_id(obj["commit"]):
        raise bad(f"commit {obj['commit']!r} is not a full commit id (40 or 64 lower-case hex)")
    if not (isinstance(obj["blob_sha256"], str) and _capsule._is_hex(obj["blob_sha256"], 64)):
        raise bad(f"blob_sha256 {obj['blob_sha256']!r} is not 64 lower-case hex")
    amends, decision = obj["amends"], obj["decision"]
    if kind == "prereg":
        if amends is not None:
            raise bad(f"a prereg line amends nothing, and this one names {amends!r}")
        if decision != "own":
            raise bad(f"a prereg line's decision is its own ('own'), not {decision!r}")
        if obj["path"] in sealed:
            raise bad(f"{obj['path']} is sealed by line id {sealed[obj['path']]!r} already; a later seal of a "
                      f"path is an amendment line")
    else:
        if not isinstance(amends, str) or amends not in ids:
            raise bad(f"an {kind} line amends a line above it, and {amends!r} names none")
        if ids[amends][1] == "withdrawn":
            raise bad(f"amends {amends!r}, a withdrawn line, which nothing amends")
        if decision not in _DECISIONS:
            raise bad(f"decision {decision!r} is not one of {list(_DECISIONS)}")
    if obj["seen"] is not None and obj["seen"] not in _SEEN:
        raise bad(f"seen {obj['seen']!r} is not one of {list(_SEEN)} (or null)")
    time = obj["time"]
    if not isinstance(time, list):
        raise bad(f"time is a list of token paths, not {type(time).__name__}")
    for token in time:
        problem = _path_problem(token)
        if problem:
            raise bad(f"time token {token!r} {problem}")
    if len(set(time)) != len(time):
        raise bad("time lists a token twice")
    ids[line_id] = (n, kind)
    if kind == "prereg":
        sealed[obj["path"]] = line_id
    return {k: (list(obj[k]) if k == "time" else obj[k]) for k in _MANIFEST_KEYS}


def _manifest(data: Any) -> List[Dict[str, Any]]:
    """The lines of a seals manifest's bytes, each a dict in the format's key order, or
    ``_ManifestError`` naming the line refused.

    The file is UTF-8 with no byte-order mark, LF only (prereg-ci reads blobs; a carriage return
    is refused, not read past), one non-blank strict JSON object a line (``research_capsule``'s
    reader: duplicate and non-ASCII keys refused), ending in a newline. A line holds every key of
    ``_MANIFEST_KEYS`` and no other; ``id`` is unique; ``path`` and every ``time`` token are
    repository-relative POSIX paths; ``commit`` is a full id; ``amends`` names a line above it that
    is not withdrawn (null on a prereg line, whose ``decision`` is ``own``); a path has at most one
    prereg line; ``seen`` is null or one of ``_SEEN``. An amendment or withdrawn line with no
    ``seen`` key is ``MANIFEST_SEEN_MISSING``; whether its null is the imported record's is the git
    layer's reading, since it turns on the commit that added the manifest (PLAN K10)."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"a manifest is bytes (a blob), not {type(data).__name__}")
    data = bytes(data)
    if data.startswith(b"\xef\xbb\xbf"):
        raise _ManifestError(_MANIFEST_INVALID, "begins with a byte-order mark", 1)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _ManifestError(_MANIFEST_INVALID, f"is not UTF-8: {e}")
    if not text:
        return []
    if not text.endswith("\n"):
        raise _ManifestError(_MANIFEST_INVALID, "the file does not end with a newline, so the next line appended "
                                                "would join this one", text.count("\n") + 1)
    ids: Dict[str, Tuple[int, str]] = {}
    sealed: Dict[str, str] = {}
    out = []
    for n, raw in enumerate(text[:-1].split("\n"), 1):
        if "\r" in raw:
            raise _ManifestError(_MANIFEST_INVALID, "holds a carriage return; the manifest is LF", n)
        if not raw.strip():
            raise _ManifestError(_MANIFEST_INVALID, "is blank", n)
        try:
            obj = _capsule._strict_json(raw)
        except ValueError as e:
            raise _ManifestError(_MANIFEST_INVALID, f"is not strict JSON: {e}", n)
        out.append(_manifest_entry(obj, n, ids, sealed))
    return out


def _manifest_bytes(entries: Iterable[Dict[str, Any]]) -> bytes:
    """The canonical bytes of manifest lines: the format's key order, ``json.dumps``'s default
    spacing, ASCII, one line each with LF. ``_manifest(_manifest_bytes(es)) == es`` for every list
    ``_manifest`` returns."""
    return "".join(json.dumps({k: e[k] for k in _MANIFEST_KEYS}, ensure_ascii=True) + "\n"
                   for e in entries).encode("ascii")


def _lineage(entries: Sequence[Dict[str, Any]], root: str) -> List[Dict[str, Any]]:
    """The amendment and withdrawn lines that descend from the line *root* (through ``amends``), in
    file order. A line only amends one above it, so one pass reads the whole family."""
    family = {root}
    out = []
    for e in entries:
        if e["kind"] != "prereg" and e["amends"] in family:
            family.add(e["id"])
            out.append(e)
    return out


def _canonical(block: Optional[str]) -> Optional[str]:
    """A fence's decision tuple (DESIGN 3.4 C0: each gate's metric, op, value and composition,
    the rows in order, the smoke verdict and the receipt pointer) as canonical JSON, or None where
    the block does not parse. Prose keys such as ``power_basis`` are not in it."""
    if block is None:
        return None
    try:
        spec = _capsule._strict_json(block)
    except ValueError:
        return None
    if (not isinstance(spec, dict) or not isinstance(spec.get("gates"), dict)
            or not isinstance(spec.get("outcomes"), list)):
        return None
    gates = {name: ({k: g[k] for k in ("metric", "op", "value", *_COMPOSITION_KEYS) if k in g}
                    if isinstance(g, dict) else g) for name, g in spec["gates"].items()}
    rows = [[r.get("when"), r.get("verdict")] if isinstance(r, dict) else r for r in spec["outcomes"]]
    return json.dumps({"gates": gates, "outcomes": rows, "smoke_verdict": spec.get("smoke_verdict"),
                       "receipt_pointer": spec.get("receipt_pointer")}, sort_keys=True)


def _amendment_readings(seal: Dict[str, Any], chain: Sequence[Tuple[Dict[str, Any], Dict[str, Any], bytes]]
                        ) -> List[Dict[str, Any]]:
    """Each lineage line read against its own bytes, in file order.

    *seal* is ``_decision`` of the seal's bytes; *chain* is ``(line, _decision(bytes), bytes)`` per
    line of ``_lineage``. A line's ``decision`` must be borne out by its bytes, else
    ``MANIFEST_DECISION_DIFFERS`` (a FAIL; the line is then never in effect): ``own`` needs a fence
    the scanner reads; ``inherits`` needs no fence, or the fence in effect at that point of the file
    (the seal's, replaced by each ``own`` amendment line above it), as SEALS row 8's blob holds row
    7's fence; a fence the scanner refuses bears out neither. ``decision_identical`` marks an ``own``
    line whose decision tuple (``_canonical``) equals the one it replaces, and every ``inherits``
    line."""
    digest, block = seal.get("gates_sha256"), seal.get("block")
    out = []
    for line, reading, data in chain:
        r = {"id": line["id"], "kind": line["kind"], "path": line["path"], "commit": line["commit"],
             "decision": line["decision"], "seen": line["seen"], "form": reading["form"],
             "gates_sha256": reading["gates_sha256"], "block": reading["block"], "bytes": data,
             "decision_identical": None, "finding": None}
        why = None
        if reading["form"] is None and reading["code"] != _NOT_GATED:
            why = f"holds a fence the scanner refuses: {reading['detail']}"
        elif line["decision"] == "own" and reading["form"] is None:
            why = "holds no fence, so it carries no decision of its own"
        elif line["decision"] == "inherits" and reading["form"] is not None and reading["gates_sha256"] != digest:
            why = (f"holds the fence {reading['gates_sha256']}, not the one in effect above it "
                   f"({digest or 'none'}), so it does not inherit")
        if why is not None:
            r["finding"] = _line_finding(_MANIFEST_DECISION_DIFFERS, f"line {line['id']} ({line['path']} at "
                                         f"{line['commit'][:12]}) says {line['decision']!r} and {why}", line["id"])
        elif line["decision"] == "own":
            old = _canonical(block)
            r["decision_identical"] = old is not None and _canonical(reading["block"]) == old
            if line["kind"] == "amendment":
                digest, block = reading["gates_sha256"], reading["block"]
        else:
            r["decision_identical"] = True
        out.append(r)
    return out


def _in_effect(seal: Dict[str, Any], readings: Sequence[Dict[str, Any]], before: Dict[str, bool]) -> Dict[str, Any]:
    """The decision in effect for one receipt (DESIGN 3.4 C4).

    *seal* is ``{id, gates_sha256, bytes}`` (``id`` None for the landing rule's default seal);
    *readings* is ``_amendment_readings``; *before* maps a line id to whether its commit is a strict
    ancestor of R, the receipt's commit. The seal's fence governs, replaced by the latest ``own``
    amendment line (file order: the manifest is append-only) that is before R and bears its
    decision out. A withdrawn line is never in effect and prints nothing; an amendment line not
    before R prints ``AMENDED_AFTER_RECEIPT``. ``sealed`` is every decision digest a seal holds for
    the PREREG at R (the seal's and each own amendment's in effect), which C6(i) reads.

    Returns ``{seal, source, gates_sha256, bytes, sealed, amendments, findings}``: ``source`` is
    the governing line's id (the seal's, or None for a default seal), ``bytes`` the governing file's
    bytes (what ``rescore`` scores), ``amendments`` one entry per reading with ``before_receipt`` and
    ``in_effect``."""
    source, digest, data = seal.get("id"), seal.get("gates_sha256"), seal.get("bytes")
    sealed = [digest] if digest is not None else []
    amendments, findings = [], []
    for r in readings:
        is_before = bool(before.get(r["id"], False))
        live = r["kind"] == "amendment" and r["finding"] is None and is_before
        if r["finding"] is not None:
            findings.append(r["finding"])
        elif r["kind"] == "amendment" and not is_before:
            seen = r["seen"] if r["seen"] is not None else "not stated"
            findings.append(_line_finding(_AMENDED_AFTER_RECEIPT, (
                f"amendment {r['id']} ({r['path']} at {r['commit'][:12]}, {r['decision']}, seen: {seen}) is not "
                f"strictly before the receipt's commit, so the receipt is read under the decision before it"),
                r["id"]))
        if live and r["decision"] == "own":
            source, digest, data = r["id"], r["gates_sha256"], r["bytes"]
            if digest not in sealed:
                sealed.append(digest)
        amendments.append({"id": r["id"], "kind": r["kind"], "path": r["path"], "commit": r["commit"],
                           "decision": r["decision"], "seen": r["seen"], "gates_sha256": r["gates_sha256"],
                           "decision_identical": r["decision_identical"], "before_receipt": is_before,
                           "in_effect": live})
    return {"seal": seal.get("id"), "source": source, "gates_sha256": digest, "bytes": data, "sealed": sealed,
            "amendments": amendments, "findings": findings}


def _amended_after_data(effect: Dict[str, Any], earlier: Iterable[Dict[str, Any]],
                        policy: str = "report") -> List[Dict[str, Any]]:
    """``AMENDED_AFTER_DATA`` (DESIGN 3.4 C4): when an own amendment governs this receipt, every
    earlier receipt of the same PREREG whose commit is strictly before that amendment's commit,
    read under its own decision in effect, with its verdict and the amendment's ``seen`` (null
    printed as "not stated"). A NOTE under ``amend_after_data = "report"`` (the default), a FAIL
    under ``"fail"``. *effect* is ``_in_effect``'s output; each of *earlier* is ``{path, commit,
    before_amendment, source, verdict, code}`` (``code`` set where the re-score refused), with
    ``in_tree`` False for a receipt read from history that the trees read no longer hold."""
    if policy not in _AMEND_AFTER_DATA:
        raise ValueError(f"amend_after_data is one of {_AMEND_AFTER_DATA}, not {policy!r}")
    gov = effect["source"]
    if gov is None or gov == effect["seal"]:
        return []
    line = next(a for a in effect["amendments"] if a["id"] == gov)
    seen = line["seen"] if line["seen"] is not None else "not stated"
    out = []
    for e in earlier:
        if not e["before_amendment"] or e["source"] == gov:
            continue
        under = "the sealed table" if e["source"] == effect["seal"] else f"amendment {e['source']}"
        read = e["verdict"] if e.get("code") is None else e["code"]
        if e.get("in_tree") is False:
            under += " (in history only: it was deleted since)"
        f = _line_finding(_AMENDED_AFTER_DATA, (
            f"{e['path']}@{e['commit'][:12]} read {read} under {under}, before amendment {gov} ({line['path']} at "
            f"{line['commit'][:12]}) came to govern this receipt; amendment seen: {seen}"), gov)
        if policy == "fail":
            f["kind"] = "FAIL"
        out.append(f)
    return out


# ---- the fence scanner with its info string as a parameter (PLAN K4), and the RESULT readers (step B3) ----

# an info string this scanner is asked for: one lower-case ASCII word
_INFO = re.compile(r"[a-z][a-z0-9_-]*")
_EM = "\u2014"
# a CommonMark container prefix: a block quote marker, or a list item marker and the spacing after it
_CONTAINER = re.compile(r" {0,3}(?:>[ \t]?|(?:[-+*]|[0-9]{1,9}[.)])(?:[ \t]+|$))")
# a line CommonMark may open a fence on: its indent, then a backtick run (whose info string holds no
# backtick) or a tilde run, then the info string
_LOOSE_FENCE = re.compile(r"([ \t]*)(?:`{3,}([^`]*)|~{3,}(.*))")
_RESULT_REFUSES = "RESULT_REFUSES"
_RESULT_FORMS_DIFFER = "RESULT_FORMS_DIFFER"
_RESULT_NOT_READ = "RESULT_NOT_READ"
# the RESULT readers' codes, in a table of their own (as _SEAL_KIND is): _KIND is compare's and the cover's
_RESULT_KIND = {_RESULT_REFUSES: "FAIL", _RESULT_FORMS_DIFFER: "FAIL", _RESULT_NOT_READ: "NOTE"}
_RESULT_KEYS = ("prereg", "receipt", "verdict")
_VERDICT_LEAF = "/verdict"
# the lexer's reasons under which a document's tags have no canonical text (styxx.sworn's own list)
_LEXICAL = ("tag_syntax", "nesting", "stray_closer", "unclosed")
# a tag the sworn lexer would meet (its own candidate pattern) that names a verdict leaf, wherever it sits: what a
# document the lexer cannot read is searched for, so a mention of the leaf in prose or in code is not one
_VERDICT_TAG = re.compile(rb"</?[sS][wW][oO][rR][nN](?![A-Za-z0-9_\-])[^>\n]*#/verdict")
# bytes that are not UTF-8 are read as a RESULT only when they hold such a tag or a result fence's marker
_BYTES_FORM = re.compile(rb"(?im:^[^\n]*?(?:`{3,}|~{3,})[ \t]*result)")


class _ResultError(ValueError):
    """A RESULT whose declared verdict is not read: the reason, which ``_result_reading`` reports as
    ``RESULT_REFUSES``."""


def _cols(indent: str) -> int:
    """The columns an indent spans, a tab to the next multiple of 4 (the scanner's own count)."""
    cols = 0
    for ch in indent:
        cols = cols + 4 - cols % 4 if ch == "\t" else cols + 1
    return cols


def _fence_at(text: str, info: str) -> Optional[Tuple[int, str]]:
    """``(line index, block text)`` of the one ```<info> fence of *text*, None where the text holds
    no <info>-like fence, or a ``GateSpecError``.

    The scanner of ``styxx.protocol._select_gates_block`` (``styxx/protocol.py:66-147``) with the
    info string as a parameter, rule for rule and in the same order: every fence line is tokenised
    once with that module's own pattern; an info string holding any non-ASCII character refuses,
    on any fence line; an <info>-like block is a fence a renderer shows (indent of at most 3
    columns, a tab to the next multiple of 4) whose casefolded info string is *info*, counted
    inside HTML comments too; exactly one may exist, in the plain unindented lower-case backtick
    form, outside every HTML comment (an unterminated one runs to the end), and closed. For
    "gates" the refusals' words are that module's, so the two read a PREREG one way; the one
    difference is the reading of a text with no gates-like fence, which that module refuses and
    this returns as None."""
    if not isinstance(info, str):
        raise TypeError(f"an info string is a str, not {type(info).__name__}")
    if not _INFO.fullmatch(info):
        raise ValueError(f"an info string here is one lower-case ASCII word, not {info!r}")
    doc = "prereg" if info in ("gates", "locks") else "the document"
    lines = text.split("\n")
    spans, pos = [], 0
    while True:
        s = text.find("<!--", pos)
        if s < 0:
            break
        e = text.find("-->", s + 4)
        spans.append((s, len(text) if e < 0 else e + 3))
        pos = s + 4 if e < 0 else e + 3
    offsets, off = [], 0
    for ln in lines:
        offsets.append(off)
        off += len(ln) + 1
    like = []
    for i, ln in enumerate(lines):
        m = _protocol._FENCE_LINE_RE.match(ln)
        if not m:
            continue
        indent, marker, found = m.groups()
        if not found.isascii():
            raise _GateSpecError(
                f"fence info string {found!r} (line {i + 1}) contains non-ASCII characters {_EM} a "
                f"confusable info string is a shadowing channel, and this parser refuses the "
                f"class rather than trusting a normalization table to enumerate it")
        if _cols(indent) <= 3 and found.casefold() == info:
            like.append((i, indent, marker, found))
    if not like:
        return None
    if len(like) > 1:
        where = [f"line {i + 1} ({marker}{found})" for i, _, marker, found in like]
        raise _GateSpecError(
            f"{doc} contains {len(like)} {info}-like fenced blocks ({', '.join(where)}) "
            f"{_EM} the frozen block must be unambiguous. A document that can show a reader one "
            f"block and score against another is not frozen; remove or rename all but one.")
    i, indent, marker, found = like[0]
    if indent or marker != "```" or found != info:
        raise _GateSpecError(
            f"the single {info}-like block (line {i + 1}, {indent!r}{marker}{found}) is not a "
            f"plain unindented ```{info} fence {_EM} the one form a renderer and this parser read "
            f"identically. A block only one of them recognises is a shadowing channel, not a "
            f"style choice.")
    if any(s <= offsets[i] < e for s, e in spans):
        raise _GateSpecError(
            f"the ```{info} fence at line {i + 1} sits inside an HTML comment (possibly "
            f"unterminated) {_EM} a renderer hides it, so scoring it would divorce the machine's "
            f"authority from the reader's. De-comment it.")
    for j in range(i + 1, len(lines)):
        if _protocol._CLOSER_RE.match(lines[j]):
            return i, "\n".join(lines[i + 1:j])
    raise _GateSpecError(
        f"the ```{info} fence at line {i + 1} is never closed {_EM} an unterminated block renders "
        f"as everything-is-code and freezes nothing")


def _select_fence(text: str, info: str) -> Optional[str]:
    """The text of the one ```<info> fence (``_fence_at``), None where there is none, or a
    ``GateSpecError``. For "gates" it decides as ``styxx.protocol._select_gates_block`` does
    (``tests/test_prereg_ci_result.py`` and ``conformance/prereg_ci/vectors_fence.json``)."""
    found = _fence_at(text, info)
    return None if found is None else found[1]


def _shadow_fences(text: str, info: str) -> List[int]:
    """The 1-based lines on which CommonMark opens an <info> block that the scanner does not count:
    a fence line behind a byte-order mark, a block quote marker or a list item marker, or one
    whose info string's leading word is *info* (any case) with more words after it. A leading word
    holding a non-ASCII character counts too, the scanner's own class. A line the scanner tokenises
    as it stands is left to ``_fence_at``, and so is indented code (4 columns or more)."""
    out = []
    for i, ln in enumerate(text.split("\n")):
        rest, wrapped = ln, False
        if rest.startswith("\ufeff"):
            rest, wrapped = rest.lstrip("\ufeff"), True
        while True:
            m = _CONTAINER.match(rest)
            if m is None:
                break
            rest, wrapped = rest[m.end():], True
        m = _LOOSE_FENCE.fullmatch(rest)
        if m is None or _cols(m.group(1)) > 3:
            continue
        if not wrapped and _protocol._FENCE_LINE_RE.match(ln):
            continue
        words = (m.group(2) if m.group(2) is not None else m.group(3)).split()
        if words and (not words[0].isascii() or words[0].casefold() == info):
            out.append(i + 1)
    return out


def _fence_json(text: str, info: str) -> Optional[Tuple[int, Any]]:
    """``(1-based line, object)`` of the one ```<info> fence read as strict JSON (duplicate and
    non-ASCII keys refused, ``research_capsule``'s reader), None where the text holds none, or a
    ``GateSpecError``: every refusal of ``_fence_at``, then a fence ``_shadow_fences`` finds, then
    bytes that are not strict JSON. The reader of the ```result fence (step B3) and of the ```locks
    fence (step B4)."""
    found = _fence_at(text, info)
    shadow = _shadow_fences(text, info)
    if shadow:
        raise _GateSpecError(
            f"a ```{info} fence at line {shadow[0]} renders as a {info} block (behind a byte-order mark, in a "
            f"block quote or a list item, or with a second word in its info string) and the scanner counts no "
            f"fence there {_EM} a block only one of them recognises is a shadowing channel; write the one plain "
            f"unindented ```{info} fence")
    if found is None:
        return None
    i, block = found
    try:
        return i + 1, _capsule._strict_json(block)
    except ValueError as e:
        raise _GateSpecError(f"the ```{info} fence at line {i + 1} is not valid strict JSON: {e}")


def _result_fence(text: str) -> Optional[Dict[str, Any]]:
    """The declaration of a RESULT's one ```result fence (DESIGN 3.2(c)): a JSON object naming
    ``prereg``, ``receipt`` and ``verdict``, three non-empty strings, and nothing else, so no key a
    reader sees beside them is one the machine passes over. None where there is no fence."""
    got = _fence_json(text, "result")
    if got is None:
        return None
    line, obj = got
    if not isinstance(obj, dict):
        raise _ResultError(f"the ```result fence at line {line} is not a JSON object (it is a {type(obj).__name__})")
    missing = [k for k in _RESULT_KEYS if k not in obj]
    extra = sorted(k for k in obj if k not in _RESULT_KEYS)
    if missing or extra:
        raise _ResultError(f"the ```result fence at line {line} names prereg, receipt and verdict and nothing else; "
                           + "; ".join(([f"it lacks {missing}"] if missing else [])
                                       + ([f"it holds {extra}"] if extra else [])))
    for k in _RESULT_KEYS:
        v = obj[k]
        if not isinstance(v, str) or not v.strip() or any(ord(c) < 32 for c in v):
            raise _ResultError(f"the ```result fence at line {line}: {k} {v!r} is not a non-empty string of one line")
    return {"form": "fence", "receipt": obj["receipt"], "prereg": obj["prereg"], "verdict": obj["verdict"],
            "line": line}


def _on_verdict_leaf(d: Dict[str, Any]) -> bool:
    """Whether a declaration the sworn lexer met names a receipt's verdict leaf (``#/verdict``), or,
    for a tag the lexer could not read, whether its bytes name one."""
    ref = d.get("receipt")
    if isinstance(ref, str):
        return "#" in ref and ref.split("#", 1)[1] == _VERDICT_LEAF
    return "#" + _VERDICT_LEAF in (d.get("raw") or "")


def _verdict_spans(raw: bytes) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(declarations, unread)``: every sworn span on a receipt's verdict leaf (DESIGN 3.2(c)), read
    through ``styxx.sworn.scan``, or a ``_ResultError``.

    A span declares the verdict its needle quotes: ``k="quote"``, the content of its one inline
    code span. Each is held to the checks ``styxx.sworn`` makes from the bytes alone, in its order
    (the lexer's MALFORMED reasons, a blank span, the length cap, the kind, the receipt grammar, a
    directional override, the needle), and any of them refuses the reading; so does a document
    sworn reads as MALFORMED (unbalanced fences) when it holds a tag naming a verdict leaf
    (``_VERDICT_TAG``), and a document whose tags do not lex when one of them names a verdict leaf. Spans on other leaves are not read here. A span naming its receipt by the ``rN`` or
    ``prereg:`` form, which no committed path holds, is listed in *unread* (``RESULT_NOT_READ``)."""
    sc = _sworn.scan(raw)
    if sc["document_malformed"] is not None:
        if _VERDICT_TAG.search(raw):
            raise _ResultError(f"styxx.sworn reads the document as MALFORMED {sc['document_malformed']['reason']}, "
                               f"so no verdict sworn in it is read")
        return [], []
    decls = sc["declarations"]
    mine = [d for d in decls if _on_verdict_leaf(d)]
    if not mine:
        return [], []

    def line_of(d: Dict[str, Any]) -> int:
        return raw.count(b"\n", 0, d["at"]) + 1

    if not sc["lexical_ok"]:
        bad = next(d for d in decls if d["malformed"] in _LEXICAL)
        raise _ResultError(f"styxx.sworn reads the tag at line {line_of(bad)} as MALFORMED {bad['malformed']}: the "
                           f"document's tags do not lex, so no verdict sworn in it is read")
    out, unread = [], []
    for d in mine:
        line, ref, kind = line_of(d), d["receipt"], d["kind"]

        def bad(reason: str) -> _ResultError:
            return _ResultError(f"the span at line {line} on {ref} is MALFORMED {reason} under styxx.sworn, so the "
                                f"verdict it declares is not read")

        if d["malformed"]:
            raise bad(d["malformed"])
        inner = d["inner"]
        if not inner.strip(b" \t\r\n\f\v"):
            raise bad("empty_span")
        if len(inner.decode("utf-8")) > _sworn.SPAN_CAP_CODEPOINTS:
            raise bad("length_cap")
        if kind in _sworn.RESERVED_KINDS:
            raise bad("kind_reserved")
        if kind not in _sworn.KINDS:
            raise bad("kind_unknown")
        parsed, why = _sworn._parse_receipt(ref)
        if parsed is None:
            raise bad(why)
        if kind in ("absent", "hash") and parsed["partial"]:
            raise bad(f"{kind}_over_partial")
        if kind != "quote":
            raise _ResultError(f"the span at line {line} on {ref} has k=\"{kind}\"; a verdict leaf is declared with "
                               f"k=\"quote\", whose needle is the verdict")
        if _sworn._DIRECTIONAL_OVERRIDE.search(inner.decode("utf-8")):
            raise bad("directional_override")
        needle, why = _sworn._needle_in(inner)
        if needle is None:
            raise bad(why)
        if parsed["form"] != "path":
            unread.append({"code": _RESULT_NOT_READ, "kind": _RESULT_KIND[_RESULT_NOT_READ], "receipt": ref,
                           "line": line, "detail": f"the span at line {line} declares the verdict of {ref}, a "
                                                   f"{parsed['form']} receipt no committed path holds, so it is "
                                                   f"not paired (styxx.sworn checks the span itself)"})
            continue
        out.append({"form": "sworn", "receipt": parsed["target"], "prereg": None, "verdict": needle.decode("utf-8"),
                    "line": line})
    return out, unread


def _result_reading(data: Any) -> Dict[str, Any]:
    """What a RESULT's bytes declare (DESIGN 3.2(c)): ``{code, kind, detail, declarations, unread}``.

    *data* is a blob. The ```result fence is read from the bytes decoded as UTF-8 with CRLF and a
    lone CR read as LF (as a PREREG is), the sworn spans from the bytes as ``styxx.sworn`` lexes
    them. Each declaration is ``{form, receipt, prereg, verdict, line}``: ``form`` "fence" or
    "sworn"; ``receipt`` the name the RESULT gives (a sworn path is repository-relative, a fence
    name is resolved as DESIGN 3.2(d) resolves names); ``prereg`` the fence's PREREG name (None for
    a span); ``line`` 1-based. A refusal of either form (the scanner's, a shadow fence's, strict
    JSON's, the fence's keys, or a span's) is ``RESULT_REFUSES``, a FAIL, and then no declaration
    is read. A document holding neither form declares nothing, which is no finding: most markdown
    is not a RESULT. Bytes that are not UTF-8 are refused only when they hold a marker of a form."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"a RESULT is bytes (a blob), not {type(data).__name__}")
    data = bytes(data)
    out: Dict[str, Any] = {"code": None, "kind": None, "detail": None, "declarations": [], "unread": []}

    def refuse(detail: str) -> Dict[str, Any]:
        out.update(code=_RESULT_REFUSES, kind=_RESULT_KIND[_RESULT_REFUSES], detail=detail, declarations=[],
                   unread=[])
        return out

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        if _VERDICT_TAG.search(data) or _BYTES_FORM.search(data):
            return refuse(f"the document is not UTF-8 ({e}), and it holds the marker of a ```result fence or of a "
                          f"span on a verdict leaf")
        return out
    try:
        fence = _result_fence(text.replace("\r\n", "\n").replace("\r", "\n"))
        spans, unread = _verdict_spans(data)
    except (_GateSpecError, _ResultError) as e:
        return refuse(str(e))
    out["declarations"] = ([fence] if fence is not None else []) + spans
    out["unread"] = unread
    return out


def _declared(declarations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """The verdict each RESULT declares for each receipt (DESIGN 3.4 C6(iii)): ``{declared, findings}``.

    Each of *declarations* is a ``_result_reading`` declaration with ``result`` (the RESULT's path)
    and ``receipt_path`` (the receipt its name resolves to, the git layer's reading) added. They are
    grouped by RESULT and receipt in the order met; each group is ``{result, receipt_path, verdict,
    verdicts, sources}``, ``sources`` one ``{form, line, verdict}`` per declaration. Two verdicts
    for one receipt in one RESULT (the fence and a span, or two spans) are ``RESULT_FORMS_DIFFER``,
    a FAIL, and the group's ``verdict`` is None: DESIGN 3.2(c), both must agree."""
    groups: Dict[Tuple[Any, str], List[Dict[str, Any]]] = {}
    for d in declarations:
        groups.setdefault((d.get("result"), d["receipt_path"]), []).append(d)
    declared, findings = [], []
    for (result, path), ds in groups.items():
        verdicts = list(dict.fromkeys(d["verdict"] for d in ds))
        sources = [{"form": d["form"], "line": d["line"], "verdict": d["verdict"]} for d in ds]
        declared.append({"result": result, "receipt_path": path, "verdict": verdicts[0] if len(verdicts) == 1 else None,
                         "verdicts": verdicts, "sources": sources})
        if len(verdicts) > 1:
            findings.append({"code": _RESULT_FORMS_DIFFER, "kind": _RESULT_KIND[_RESULT_FORMS_DIFFER], "path": result,
                             "detail": f"{result or 'the RESULT'} declares {len(verdicts)} verdicts for {path}: "
                                       + "; ".join(f"the {s['form']} at line {s['line']} prints {s['verdict']}"
                                                   for s in sources)
                                       + "; the forms must agree (DESIGN 3.2(c))"})
    return {"declared": declared, "findings": findings}
