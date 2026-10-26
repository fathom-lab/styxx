# -*- coding: utf-8 -*-
"""styxx.prereg_ci.core: the pure half of prereg-ci.

A sealed decision re-derives a verdict from a receipt, and a recorded verdict is read against
it. Nothing in this module reads a disk, a clock, a process or a network; every function is a
function of the bytes and objects it is handed, so a FAIL built on it is a function of the git
objects those bytes came from.

Public: ``rescore``, ``compare`` and ``licences``. The private names of ``styxx.protocol``,
``styxx.research_capsule`` and ``styxx.sworn`` it reuses are used here and nowhere else in the
package; the conformance set under ``conformance/prereg_ci/`` pins what they do, vector by vector.

The gates form. A PREREG's decision is its one ```gates fence, selected by
``styxx.protocol._select_gates_block`` (the single scanner and its refusals), over the PREREG
bytes decoded as UTF-8 with newlines read as ``Path.read_text`` reads them: CRLF and a lone CR
become LF. prereg-ci reads blobs and ``styxx.protocol.Experiment`` reads a working copy, so this
is what makes the digest of a fence the one an Experiment recorded for the same file on a box that
checks text out with CRLF. The fence is scored by ``styxx.research_capsule.score_frozen``; any
refusal of the scanner or the scorer reads ``SCORER_REFUSES`` with its text. ``_decision`` and
``rescore`` read the gates form as that scanner reads it, which the conformance set pins (a fence
hidden from it reads ``NOT_GATED``, ``vectors_compare.json``); ``_shadowed`` names a gates-like fence a
renderer shows that the scanner does not count (in a block quote, a list item or a footnote, or with a
second word in its info string: ``_shadow_lines``, the rule the locks and result readers refuse), and
the check, C0 and lint refuse a PREREG holding one as ``FENCE_SHADOWED``, so a PREREG cannot show a
reader one table while another is scored (the B8 battery, M1).

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
quote, a list item or a footnote, on a list item's or a footnote's continuation lines, or with a
second word in the info string), then reads strict JSON.
``_result_reading`` reads a RESULT's ```result fence and its sworn spans on receipts' verdict leaves
(through ``styxx.sworn.scan``, held to the checks sworn makes from bytes alone), and refuses one in
UTF-16 or UTF-32 that holds a form's marker; ``_result_candidate`` says whether a reading of some
bytes could find anything, which is how the git layer reads any file a pull request writes;
``_declared`` groups what one RESULT declares per receipt and refuses two forms that disagree.

The locks form (DESIGN 3.2(a) and (f), step B4). A PREREG's decision is exactly one of two forms: the
gates fence above, or one ```locks fence, read by ``_fence_block`` with the result fence's rules (the
scanner's refusals, then a fence a renderer may show inside a container). A PREREG holding both is
refused, and so is a gates PREREG holding a locks-like fence the locks reader refuses. ``_decision``
gives the form and the digest of the fence's text for both, so the git layer's seal, order and C0
readings hold a locks decision as they hold a gates one. ``_locks`` reads the block: ``scorer`` and
``reads`` (the locked files, repository-relative and portable, each named once, none under the run's
own ``.prereg-ci/``), ``outcomes`` (the verdicts the scorer may print), ``withdraws`` (a map from a
declared outcome to its licences, read at step B5), and two optional keys (PLAN K2b): ``argv``, the
scorer's arguments with ``{receipt}`` once and ``{out}`` at most once, each a whole element (default
``["{receipt}"]``), and ``verdict_from``, ``"stdout"`` (one line of JSON ``{"verdict": V}``,
optionally ``"gates"``) or an RFC 6901 pointer into the JSON the scorer writes at ``{out}``.
``rescore`` runs nothing, so a locks decision reads ``SCORER_NOT_RUN`` there;
``styxx.prereg_ci.scorer_run`` runs the scorer at its sealed blobs and ``_run_reading`` reads what the
run left: ``SCORER_READS_UNLOCKED`` when the run recorded a reach past the locks, whatever the scorer
did next, else ``SCORER_ERROR`` for a timeout, a nonzero exit, a record the run cannot read, a run that
ended without its closing line, or output the contract does not read, else the verdict, which must be
a declared outcome.

Licences (DESIGN 3.2(g), 3.4 C8, 3.7; step B5). An outcome row of the gates form may carry
``"withdraws": [{"id", "quote", "pattern"}]``, and the locks block's ``withdraws`` map holds the same
lists per declared outcome: ``quote`` is the sentence as the PREREG writes it, ``pattern`` a
case-insensitive regular expression over whitespace-collapsed text (every run of whitespace one
space, with the curly apostrophes U+2018 and U+2019 read as the straight one, PLAN K15) that the
author expects to match the wordings on shipped surfaces. ``_licence_specs`` reads them strictly, in
table order, after the table itself reads as ``rescore`` reads it; ``rescore`` returns the landed
row's list as the row writes it under ``withdraws``. ``licences`` scans the surfaces it is handed,
pairs of a repository-relative path and the surface's text (None for one not read), with ``_scan``:
a match is named by line and column in the text as ``Path.read_text`` would read it, and is
withdrawn when a withdrawal (a form of "withdraw", or "not yet true of this instrument") lies whole
within ``_WINDOW`` characters of it on either side, ``scripts/surface_lint.py``'s window rule
re-implemented over compiled patterns, so the sentence that withdraws a claim may quote it. Once a
receipt has landed on a row (``landed`` names its verdict), every match of that row's licences that
is not withdrawn is ``LICENCE_WITHDRAWN`` (FAIL); before any receipt lands (``landed`` None) it is
``LICENCE_PENDING`` (NOTE: the sentence stands on a verdict not yet given), and a licence whose
pattern matches no surface at all is ``LICENCE_PATTERN_FINDS_NOTHING`` (FAIL, the lint's refusal: a
licence that matches nothing before data licenses nothing, the analogue of a vacuous gate). Rows
that did not land withdraw nothing, and every match is listed either way, so the author sees a
pattern's coverage. Which files are surfaces is a profile: ``_SURFACE_PROFILE`` is a copy of
surface_lint's ``repo`` profile (PLAN K14: ``scripts/`` is not in the wheel), a configured profile
names globs (``*`` stays inside one directory, ``**`` crosses them), ``_surface_selected`` applies
either and ``_surface_text`` decodes a blob as surface_lint does; the git layer's ``_surfaces``
reads them at a commit. Licences match strings (DESIGN 7, risk 3): a pattern registered before
data does not reach a wording written after it. The match is Python's ``re`` with no step limit, so
a pattern is capped at ``_PATTERN_MAX_CHARS`` characters, refused when it can match the empty
string, and nothing else guards its cost. ``licences`` names the landed row by its index when given
``row`` (step B7), so two rows that share a verdict are told apart; ``_quote_coverage`` reads each
licence's quote as a literal pattern, which is what the report prints beside the pattern's coverage.

The ladder (DESIGN 3.4 C7, step B7). ``_ladder`` reads a scored receipt's outcome rows in order with
the landed one marked, each gate's metric, op, bar, observed value and margin (positive where the gate
fired with room), and the single-gate flip table: where the receipt would land with one gate's reading
flipped. Every flip that moves the verdict is printed with its margin; none is called decisive.
"""
from __future__ import annotations

import bisect
import functools
import hashlib
import json
import math
import numbers
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from styxx import protocol as _protocol
from styxx import research_capsule as _capsule
from styxx import sworn as _sworn

__all__ = ["rescore", "compare", "licences"]

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
# the locks form (step B4): rescore runs no scorer; the scorer run's two readings
_SCORER_NOT_RUN = "SCORER_NOT_RUN"
_SCORER_ERROR = "SCORER_ERROR"
_SCORER_READS_UNLOCKED = "SCORER_READS_UNLOCKED"
# licences (step B5): a landed row's sentence still on a surface; a sentence on a verdict not yet given; a
# pattern that matches no surface before data (the lint's refusal)
_LICENCE_WITHDRAWN = "LICENCE_WITHDRAWN"
_LICENCE_PENDING = "LICENCE_PENDING"
_LICENCE_PATTERN_FINDS_NOTHING = "LICENCE_PATTERN_FINDS_NOTHING"
_KIND = {_NOT_GATED: "NOTE", _SCORER_REFUSES: "FAIL", _SMOKE_AMBIGUOUS: "INPUT",
         _DECISION_DIGEST_UNSEALED: "FAIL", _VERDICT_DIFFERS: "FAIL",
         _RECORDED_DIGEST_ABSENT: "NOTE", _RECORDED_VERDICT_ABSENT: "NOTE",
         _COVER_UNCHECKABLE: "UNCHECKABLE",
         _SCORER_NOT_RUN: "UNCHECKABLE", _SCORER_ERROR: "FAIL", _SCORER_READS_UNLOCKED: "FAIL",
         _LICENCE_WITHDRAWN: "FAIL", _LICENCE_PENDING: "NOTE", _LICENCE_PATTERN_FINDS_NOTHING: "FAIL"}
# every key of rescore's output, which the locked scorer run's output holds too; "withdraws" (step B5) is the
# landed row's licences as the row writes them, [] where the landed row (or a smoke run) withdraws nothing
_RESCORE_KEYS = ("form", "code", "detail", "verdict", "gates", "row", "gates_sha256", "smoke", "smoke_verdict",
                 "total", "holes", "unreachable_rows", "cover_code", "cover_detail", "withdraws")
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

    Returns ``{form, block, gates_sha256, code, detail}``: form "gates" or "locks" with the selected
    fence's text and the sha256 of that text (the decision digest; the key keeps the gates name the
    receipts record); or form None with ``code`` ``NOT_GATED`` (no gates-like and no locks-like
    fence at all) or ``SCORER_REFUSES`` (a fence a scanner refuses, both forms in one PREREG, or
    bytes that are not UTF-8) and the refusal's text. The gates scanner reads before the locks
    reader, so a gates PREREG reads as it read before step B4 unless it also holds a locks-like
    fence. The git layer compares these digests along a path's history (the order commit of DESIGN
    3.2(b), and C0); ``rescore`` scores the same block, so the two never read a PREREG two ways.
    """
    out: Dict[str, Any] = {"form": None, "block": None, "gates_sha256": None, "code": None, "detail": None}
    try:
        text = _decision_text(prereg)
    except UnicodeDecodeError as e:
        out.update(code=_SCORER_REFUSES, detail=f"the PREREG is not UTF-8: {e}")
        return out
    none = None
    try:
        block = _protocol._select_gates_block(text)
    except _GateSpecError as e:
        if not str(e).startswith(_NO_FENCE):
            out.update(code=_SCORER_REFUSES, detail=str(e))
            return out
        block, none = None, str(e)
    try:
        locks = _fence_block(text, "locks")
    except _GateSpecError as e:
        out.update(code=_SCORER_REFUSES, detail=str(e))
        return out
    if block is not None and locks is not None:
        line = _fence_at(text, "gates")[0] + 1
        out.update(code=_SCORER_REFUSES, detail=(
            f"the PREREG holds a ```gates fence (line {line}) and a ```locks fence (line {locks[0] + 1}) {_EM} "
            f"a PREREG carries one decision, in one of the two forms (DESIGN 3.2(a)); a reader who sees both "
            f"cannot tell which one judges"))
        return out
    if block is not None:
        out.update(form="gates", block=block, gates_sha256=_sha256(block))
    elif locks is not None:
        out.update(form="locks", block=locks[1], gates_sha256=_sha256(locks[1]))
    else:
        out.update(code=_NOT_GATED, detail=none)
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


def _gates_spec(block: str) -> Any:
    """A gates fence's text as strict JSON (``research_capsule``'s reader: duplicate and non-ASCII keys
    refused), or a ``GateSpecError`` in ``rescore``'s words; the reading ``lint`` starts from."""
    try:
        return _capsule._strict_json(block)
    except (ValueError, UnicodeDecodeError) as e:
        raise _GateSpecError(f"gates block is not valid strict JSON: {e}")


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
    ``smoke_verdict``, ``withdraws`` (the landed row's ``withdraws`` list as the row writes it,
    DESIGN 3.2(g); ``[]`` where the row has none or the receipt is a smoke run, which lands on no
    row; None where no verdict was read; ``licences`` reads the list strictly, this function does
    not), and the cover of the table: ``total``, ``holes``, ``unreachable_rows``.
    ``cover_code`` is None, or ``COVER_UNCHECKABLE`` (with ``cover_detail``) when the cover ran
    past *cover_budget* subcubes; the verdict is read either way. ``cover_budget=None`` runs the
    cover with no budget.

    A locks decision is scored by running its scorer, which this pure function does not do: it
    reads form "locks", the decision digest, and ``SCORER_NOT_RUN`` (UNCHECKABLE) once the block
    reads (a block ``_locks`` refuses is ``SCORER_REFUSES``); ``styxx.prereg_ci.scorer_run`` gives
    the same keys for a run.
    """
    if cover_budget is not None and (isinstance(cover_budget, bool) or not isinstance(cover_budget, int)
                                     or cover_budget < 1):
        raise ValueError(f"cover_budget is a positive int or None, not {cover_budget!r}")
    out: Dict[str, Any] = dict.fromkeys(_RESCORE_KEYS)

    def refuse(detail: str) -> Dict[str, Any]:
        out.update(code=_SCORER_REFUSES, detail=detail)
        return out

    decision = _decision(prereg)
    if decision["form"] is None:
        out.update(code=decision["code"], detail=decision["detail"])
        return out
    if decision["form"] == "locks":
        out.update(form="locks", gates_sha256=decision["gates_sha256"])
        try:
            _locks(decision["block"])
        except _GateSpecError as e:
            return refuse(str(e))
        out.update(code=_SCORER_NOT_RUN, detail=(
            "a locks decision is scored by running its scorer at its sealed blobs (styxx.prereg_ci.scorer_run); "
            "rescore is pure and runs nothing"))
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
        out.update(verdict=spec["smoke_verdict"], gates={}, smoke_verdict=spec["smoke_verdict"], withdraws=[])
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
        landed = _landed(rows, scored["gates"])
        out.update(verdict=scored["verdict"], gates=scored["gates"], smoke_verdict=scored["smoke_verdict"],
                   row=landed, withdraws=spec["outcomes"][landed].get("withdraws", []))
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
# a CommonMark container prefix: a block quote marker, or a list item marker and the spacing after it (the B3 rule)
_CONTAINER = re.compile(r" {0,3}(?:>[ \t]?|(?:[-+*]|[0-9]{1,9}[.)])(?:[ \t]+|$))")
# a container marker on a line at any indent, since a nested container sits 4 columns or more in (the B3 review,
# ADV1): a block quote marker, a list item marker and the spacing after it, or a footnote label (GFM) and the
# spacing after it
_PREFIX = re.compile(r"[ \t]*(?:>[ \t]?|(?:[-+*]|[0-9]{1,9}[.)])(?:[ \t]+|$)|\[\^[^\]\n]*\]:(?:[ \t]+|$))")
# a line's block quote markers and indent, read past to see whether the line opens a list item or a footnote
_QUOTES = re.compile(r"(?:[ \t]*>)*[ \t]*")
# a line that opens a list item or a footnote: a container whose content a renderer reads on the lines below it,
# indented by the width of its marker
_OPENS_ITEM = re.compile(r"(?:[-+*]|[0-9]{1,9}[.)])(?:[ \t]|$)|\[\^[^\]\n]*\]:")
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
# how a finding names a declaration's form
_FORM_NAME = {"fence": "result fence", "sworn": "sworn span"}
# the lexer's reasons under which a document's tags have no canonical text (styxx.sworn's own list)
_LEXICAL = ("tag_syntax", "nesting", "stray_closer", "unclosed")
# a tag the sworn lexer would meet (its own candidate pattern) that names a verdict leaf, wherever it sits: what a
# document the lexer cannot read is searched for, so a mention of the leaf in prose or in code is not one
_VERDICT_TAG = re.compile(rb"</?[sS][wW][oO][rR][nN](?![A-Za-z0-9_\-])[^>\n]*#/verdict")
# bytes that are not UTF-8 are read as a RESULT only when they hold such a tag or a result fence's marker (B3); a
# wider marker would refuse binary files by chance
_BYTES_FORM = re.compile(rb"(?im:^[^\n]*?(?:`{3,}|~{3,})[ \t]*result)")
# the marker of a fence a RESULT reading of UTF-8 text could count or refuse, in its bytes: a backtick or tilde
# run, the ASCII whitespace str.split() reads past, then "result" in any case, or a word holding a non-ASCII byte
# (a confusable info string, or a non-ASCII space before the word). With _VERDICT_TAG it is what
# ``_result_candidate`` looks for in UTF-8 text (the B3 review, ADV2 and M1).
_FORM_MARK = re.compile(rb"(?i:(?:`{3,}|~{3,})[\t\x0b\x0c\r\x1c-\x1f ]*"
                        rb"(?:result|[^\t\n\x0b\x0c\r\x1c-\x1f ]*?[\x80-\xff]))")
# byte-order marks of the wide encodings, the UTF-32 ones before the UTF-16 ones they begin with
_WIDE_BOMS = ((b"\x00\x00\xfe\xff", "UTF-32-BE"), (b"\xff\xfe\x00\x00", "UTF-32-LE"),
              (b"\xfe\xff", "UTF-16-BE"), (b"\xff\xfe", "UTF-16-LE"))


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


def _names(m: "re.Match[str]", info: str) -> bool:
    """Whether a ``_LOOSE_FENCE`` match opens an <info> block: its info string's leading word is *info*
    (any case), or holds a non-ASCII character (the scanner's own class)."""
    words = (m.group(2) if m.group(2) is not None else m.group(3)).split()
    return bool(words) and (not words[0].isascii() or words[0].casefold() == info)


def _b3_shadow(ln: str, info: str) -> bool:
    """Whether the B3 rule names the line: a fence line behind a byte-order mark, or behind a block
    quote or list item marker at most 3 columns in, opening at most 3 columns after it; or a line at
    most 3 columns in that the scanner does not tokenise. Kept so a refusal of that class keeps its
    words (``vectors_fence.json``)."""
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
        return False
    if not wrapped and _protocol._FENCE_LINE_RE.match(ln):
        return False
    return _names(m, info)


def _shadow_lines(text: str, info: str) -> List[Tuple[int, bool]]:
    """``(1-based line, whether the B3 rule names it)`` for each line ``_shadow_fences`` finds."""
    out = []
    item_above = False
    for i, ln in enumerate(text.split("\n")):
        rest, wrapped = ln, False
        if rest.startswith("\ufeff"):
            rest, wrapped = rest.lstrip("\ufeff"), True
        while True:
            m = _PREFIX.match(rest)
            if m is None:
                break
            rest, wrapped = rest[m.end():], True
        m = _LOOSE_FENCE.fullmatch(rest)
        if m is not None and _names(m, info):
            cols = _cols(m.group(1))
            if wrapped or (cols <= 3 and not _protocol._FENCE_LINE_RE.match(ln)) or (cols > 3 and item_above):
                out.append((i + 1, _b3_shadow(ln, info)))
        bare = ln.lstrip("\ufeff")
        if _OPENS_ITEM.match(bare, _QUOTES.match(bare).end()):
            item_above = True
    return out


def _shadow_fences(text: str, info: str) -> List[int]:
    """The 1-based lines on which a renderer may open an <info> block that the scanner does not count,
    a fence whose info string's leading word is *info* (any case) or holds a non-ASCII character (the
    scanner's own class):
    - behind a container marker on its own line, the marker at any indent and the fence at any indent
      after it: a byte-order mark, a block quote marker, a list item marker, a footnote label;
    - with no marker on the line and at most 3 columns in, a line the scanner does not tokenise as it
      stands (a second word in the info string);
    - with no marker on the line and 4 or more columns in (a tab to the next multiple of 4), once any
      line above it opens a list item or a footnote, its block quote markers read past. A renderer
      reads such a line as a fence in that item's content when it sits within 3 columns of the item's
      content column; this reader does not follow a container to its end, so it refuses the line
      whether or not the container is still open (the B3 review, ADV1).
    A line the scanner tokenises at most 3 columns in is left to ``_fence_at``. Indented code (4 columns
    or more) with no list item or footnote opened above it is code, and is passed over."""
    return [n for n, _ in _shadow_lines(text, info)]


def _fence_json(text: str, info: str) -> Optional[Tuple[int, Any]]:
    """``(1-based line, object)`` of the one ```<info> fence read as strict JSON (duplicate and
    non-ASCII keys refused, ``research_capsule``'s reader), None where the text holds none, or a
    ``GateSpecError``: every refusal of ``_fence_block``, then bytes that are not strict JSON. The
    reader of the ```result fence (step B3); the ```locks fence (step B4) is read through
    ``_fence_block`` by ``_decision`` and parsed by ``_locks``, so a locks fence that is not JSON
    still has a digest, as a gates fence does."""
    found = _fence_block(text, info)
    if found is None:
        return None
    i, block = found
    try:
        return i + 1, _capsule._strict_json(block)
    except ValueError as e:
        raise _GateSpecError(f"the ```{info} fence at line {i + 1} is not valid strict JSON: {e}")


def _fence_block(text: str, info: str) -> Optional[Tuple[int, str]]:
    """``(line index, block text)`` of the one ```<info> fence, None where the text holds none, or a
    ``GateSpecError``: every refusal of ``_fence_at``, then a fence ``_shadow_fences`` finds."""
    found = _fence_at(text, info)
    shadow = _shadow_lines(text, info)
    if shadow:
        raise _GateSpecError(_shadow_refusal(info, shadow))
    return found


def _shadowed(prereg: Any) -> Optional[str]:
    """The refusal for a gates-like fence a renderer shows that the scanner does not count, or None:
    ``_shadow_lines`` over the PREREG as ``_decision`` reads it (None for bytes that are not UTF-8, which
    ``_decision`` refuses itself). ``_decision`` keeps the scanner's reading, pinned by the conformance
    set; the check reads this beside it (``FENCE_SHADOWED``), so a table quoted for the reader beside the
    one the scanner scores, or a PREREG whose one table is quoted and so reads ``NOT_GATED``, is refused
    (the B8 battery, M1)."""
    try:
        text = _decision_text(prereg)
    except UnicodeDecodeError:
        return None
    shadow = _shadow_lines(text, "gates")
    return _shadow_refusal("gates", shadow) if shadow else None


def _shadow_refusal(info: str, shadow: Sequence[Tuple[int, bool]]) -> str:
    """The refusal's words for the earliest line ``_shadow_lines`` finds: the B3 rule's class (a fence a
    renderer shows that the scanner counts nowhere) or the container class past it (the B3 review)."""
    line, b3 = shadow[0]
    if b3:
        return (f"a ```{info} fence at line {line} renders as a {info} block (behind a byte-order mark, in a "
                f"block quote or a list item, or with a second word in its info string) and the scanner counts no "
                f"fence there {_EM} a block only one of them recognises is a shadowing channel; write the one plain "
                f"unindented ```{info} fence")
    return (f"a ```{info} fence at line {line} may render as a {info} block inside a container (behind a "
            f"block quote marker, a list item marker or a footnote label on its line, or 4 or more columns in "
            f"below a line that opens a list item or a footnote) and the scanner counts no fence there {_EM} this "
            f"reader does not follow a container to its end, so the class is refused; write the one plain "
            f"unindented ```{info} fence outside every container")


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
    (``_VERDICT_TAG``), and a document whose tags do not lex when one of them names a verdict leaf.
    Spans on other leaves are not read here. A span naming its receipt by the ``rN`` or ``prereg:``
    form, which no committed path holds, is listed in *unread* (``RESULT_NOT_READ``)."""
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


def _marked(data: bytes) -> bool:
    """Whether the bytes of UTF-8 text hold the marker of a form: a fence a RESULT reading could count
    or refuse (``_FORM_MARK``), or a sworn tag naming a verdict leaf (``_VERDICT_TAG``)."""
    return bool(_FORM_MARK.search(data) or _VERDICT_TAG.search(data))


def _bytes_marked(data: bytes) -> bool:
    """Whether bytes that are not UTF-8 hold a result fence's marker or a sworn tag naming a verdict
    leaf (B3's rule for them, kept narrow so a binary file does not hold one by chance)."""
    return bool(_VERDICT_TAG.search(data) or _BYTES_FORM.search(data))


def _wide(data: bytes) -> Optional[Tuple[str, bytes]]:
    """``(encoding, text as UTF-8)`` for bytes that start with a UTF-16 or UTF-32 byte-order mark and
    decode strictly under it; None for other bytes (the B3 review, M1). Bytes with no mark are read as
    they stand: UTF-16 text with no mark holds NUL bytes, which git's diff reads as binary."""
    for bom, enc in _WIDE_BOMS:
        if data.startswith(bom):
            try:
                return enc, data[len(bom):].decode(enc.lower()).encode("utf-8")
            except UnicodeDecodeError:
                continue
    return None


def _result_candidate(data: Any) -> bool:
    """Whether a RESULT reading of *data* (``_result_reading``) could find anything: a declaration, a
    refusal, or a span it prints. UTF-8 bytes are a candidate when they hold the marker of a form
    (``_marked``); bytes in a wide encoding (``_wide``), when their text does; other bytes, when they
    hold the narrow marker the reading refuses them on (``_bytes_marked``). The git layer reads a file
    the pull request writes as a RESULT when this holds, whatever the file's suffix (the B3 review,
    ADV2); for bytes it passes over the reading is empty, which
    ``tests/test_prereg_ci_result_review.py`` checks over generated documents."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"a RESULT is bytes (a blob), not {type(data).__name__}")
    data = bytes(data)
    wide = _wide(data)
    if wide is not None:
        return _marked(wide[1])
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return _bytes_marked(data)
    return _marked(data)


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
    is not a RESULT. A RESULT is read as UTF-8: bytes that start with a UTF-16 or UTF-32 byte-order
    mark and decode under it (``_wide``) are refused when their text holds the marker of a form
    (``_marked``), and declare nothing otherwise (the B3 review, M1); other bytes that are not UTF-8
    are refused only when they hold a result fence's marker or a span's (``_bytes_marked``)."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"a RESULT is bytes (a blob), not {type(data).__name__}")
    data = bytes(data)
    out: Dict[str, Any] = {"code": None, "kind": None, "detail": None, "declarations": [], "unread": []}

    def refuse(detail: str) -> Dict[str, Any]:
        out.update(code=_RESULT_REFUSES, kind=_RESULT_KIND[_RESULT_REFUSES], detail=detail, declarations=[],
                   unread=[])
        return out

    wide = _wide(data)
    if wide is not None:
        if _marked(wide[1]):
            return refuse(f"the document is {wide[0]} (it starts with that byte-order mark), and read that way it "
                          f"holds the marker of a form (a fence whose info string reads result or holds a non-ASCII "
                          f"character, or a sworn tag naming a verdict leaf); a RESULT is read as UTF-8, so nothing "
                          f"in it is read")
        return out
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        if _bytes_marked(data):
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
                                       + "; ".join(f"the {_FORM_NAME[s['form']]} at line {s['line']} prints "
                                                   f"{s['verdict']}"
                                                   for s in sources)
                                       + "; the forms must agree (DESIGN 3.2(c))"})
    return {"declared": declared, "findings": findings}


# ---- the locks form (DESIGN 3.2(a) and (f), PLAN K2b; step B4) ----------------------------------------

# the directory the scorer run keeps for itself inside the run: the {out} file lives there, so no locked
# file and no receipt may
_RUN_DIR = ".prereg-ci"
_RECEIPT_ARG = "{receipt}"
_OUT_ARG = "{out}"
_LOCKS_KEYS = ("scorer", "reads", "outcomes", "withdraws", "argv", "verdict_from")
_LOCKS_REQUIRED = ("scorer", "reads", "outcomes")
# a reading names at most this many reaches past the locks, then counts the rest
_SHOWN = 12


def _locked_path_problem(path: Any) -> Optional[str]:
    """Why *path* cannot name a locked file or a receipt in the scorer run, or None: a
    repository-relative POSIX path (``research_capsule``'s rule, as the manifest holds paths to),
    every segment a portable name on every file system (a Windows device name such as ``CON`` is
    a file on Linux and no file on Windows), and not under the run's own ``.prereg-ci/``."""
    if not isinstance(path, str) or not path:
        return "is not a repository-relative path (a non-empty string)"
    why = _path_problem(path)
    if why is not None:
        return why
    for seg in path.split("/"):
        if not _capsule._bare_name(seg):
            return f"holds the segment {seg!r}, which is not a portable name on every file system"
    if path.split("/")[0].casefold() == _RUN_DIR:
        return f"sits under {_RUN_DIR}/, which the scorer run keeps for itself"
    return None


def _locks(block: str) -> Dict[str, Any]:
    """The ```locks block (DESIGN 3.2(a), PLAN K2b) read strictly, or a ``GateSpecError``.

    Returns ``{scorer, reads, outcomes, withdraws, argv, verdict_from, locked}``: ``locked`` is the
    scorer then the reads, the files the run materialises at their sealed blobs. The block names
    ``scorer``, ``reads`` and ``outcomes``, and may name ``withdraws`` (default ``{}``: each key a
    declared outcome, each value a list of licences, read at step B5), ``argv`` (default
    ``["{receipt}"]``) and ``verdict_from`` (default ``"stdout"``); a key the format does not name is
    refused, as a result fence's is, since a key a reader sees and the machine passes over is a
    shadowing channel. No two locked paths are one file on a case-insensitive file system, and no
    locked path is a directory of another."""
    try:
        spec = _capsule._strict_json(block)
    except (ValueError, UnicodeDecodeError) as e:
        raise _GateSpecError(f"the ```locks block is not valid strict JSON: {e}")
    if not isinstance(spec, dict):
        raise _GateSpecError(f"the ```locks block is not a JSON object (it is a {type(spec).__name__})")
    extra = sorted(k for k in spec if k not in _LOCKS_KEYS)
    if extra:
        raise _GateSpecError(f"the ```locks block holds keys the format does not name: {extra}; it names "
                             f"{', '.join(_LOCKS_KEYS)} and nothing else")
    for key in _LOCKS_REQUIRED:
        if key not in spec:
            raise _GateSpecError(f"the ```locks block lacks {key!r}; it names scorer, reads and outcomes")
    scorer, reads = spec["scorer"], spec["reads"]
    why = _locked_path_problem(scorer)
    if why is not None:
        raise _GateSpecError(f"'scorer' {scorer!r} {why}")
    if not isinstance(reads, list):
        raise _GateSpecError(f"'reads' is a list of paths, not a {type(reads).__name__}")
    for p in reads:
        why = _locked_path_problem(p)
        if why is not None:
            raise _GateSpecError(f"a path in 'reads', {p!r}, {why}")
    locked = [scorer, *reads]
    folded: Dict[str, str] = {}
    for p in locked:
        other = folded.get(p.casefold())
        if other == p:
            raise _GateSpecError(f"{p!r} is locked twice; 'scorer' and 'reads' name each file once")
        if other is not None:
            raise _GateSpecError(f"{other!r} and {p!r} differ only in case, and a checkout on a case-insensitive "
                                 f"file system holds one file for both")
        folded[p.casefold()] = p
    for a in locked:
        for b in locked:
            if b.casefold().startswith(a.casefold() + "/"):
                raise _GateSpecError(f"{a!r} is locked as a file and is a directory of {b!r}")
    outcomes = spec["outcomes"]
    if not isinstance(outcomes, list) or not outcomes:
        raise _GateSpecError("'outcomes' is a non-empty list of the verdicts the scorer may print")
    seen = set()
    for v in outcomes:
        if not isinstance(v, str) or not v.strip():
            raise _GateSpecError(f"each outcome is a non-empty string; {v!r} is not")
        if any(ord(c) < 32 or c in "\x7f\u2028\u2029" for c in v):
            raise _GateSpecError(f"each outcome is a string of one line; {v!r} is not")
        if v != v.strip():
            raise _GateSpecError(f"each outcome is a string with no leading or trailing whitespace; {v!r} is not "
                                 f"(a verdict is read against it exactly, and a reader of a report cannot see the "
                                 f"difference)")
        if v in seen:
            raise _GateSpecError(f"the outcome {v!r} is declared twice")
        seen.add(v)
    withdraws = spec.get("withdraws", {})
    if not isinstance(withdraws, dict):
        raise _GateSpecError(f"'withdraws' maps a declared outcome to a list of licences, not a "
                             f"{type(withdraws).__name__}")
    for k, v in withdraws.items():
        if k not in seen:
            raise _GateSpecError(f"'withdraws' names {k!r}, which is not a declared outcome")
        if not isinstance(v, list):
            raise _GateSpecError(f"'withdraws' maps {k!r} to a {type(v).__name__}, not a list of licences")
    argv = spec.get("argv", [_RECEIPT_ARG])
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise _GateSpecError(f"'argv' is a list of strings, not {argv!r}")
    for a in argv:
        if a not in (_RECEIPT_ARG, _OUT_ARG) and (_RECEIPT_ARG in a or _OUT_ARG in a):
            raise _GateSpecError(f"'argv' element {a!r} holds a placeholder inside other text; {_RECEIPT_ARG} and "
                                 f"{_OUT_ARG} each stand as a whole element")
    if argv.count(_RECEIPT_ARG) != 1:
        raise _GateSpecError(f"'argv' names {_RECEIPT_ARG} {argv.count(_RECEIPT_ARG)} times; it names it once")
    if argv.count(_OUT_ARG) > 1:
        raise _GateSpecError(f"'argv' names {_OUT_ARG} {argv.count(_OUT_ARG)} times; it names it at most once")
    verdict_from = spec.get("verdict_from", "stdout")
    if not isinstance(verdict_from, str) or (verdict_from != "stdout" and not verdict_from.startswith("/")):
        raise _GateSpecError(f"'verdict_from' is \"stdout\" or an RFC 6901 pointer into the {_OUT_ARG} JSON, "
                             f"beginning with '/', not {verdict_from!r}")
    if verdict_from != "stdout" and _OUT_ARG not in argv:
        raise _GateSpecError(f"'verdict_from' {verdict_from!r} points into the {_OUT_ARG} file, and 'argv' names no "
                             f"{_OUT_ARG}")
    return {"scorer": scorer, "reads": list(reads), "outcomes": list(outcomes), "withdraws": dict(withdraws),
            "argv": list(argv), "verdict_from": verdict_from, "locked": locked}


def _run_reading(spec: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    """What a locked scorer's run gives (DESIGN 3.2(f)): ``{code, detail, verdict, gates, output}``.

    *spec* is ``_locks``'s reading; *raw* is what ``styxx.prereg_ci.scorer_run`` recorded: ``exit``
    (None when stopped), ``timed_out``, ``timeout_s``, ``done`` (the run's closing line was written,
    so the scorer returned or raised SystemExit), ``stdout`` (bytes) and ``stdout_over``,
    ``stderr_tail`` (its last non-empty line), ``out`` (the bytes at ``{out}``, or None with
    ``out_problem``), ``unlocked`` (each reach past the locks the run's watch refused and recorded;
    the watch is Python in the scorer's process, and a scorer written against it can leave this
    empty, as ``scorer_run``'s docstring names, so an empty list is what the watch saw, not what the
    scorer did) and ``watch_problem``
    (the run's record holds a line its bootstrap does not write, or is missing). In this order: any
    reach past the locks is ``SCORER_READS_UNLOCKED``, whatever the scorer printed or how it exited,
    since a scorer may catch the refusal and go on; a timeout, a nonzero exit, a record the run
    cannot read, or a run that ended without its closing line is ``SCORER_ERROR`` with the last
    stderr line; then the verdict is
    read from stdout (one line of JSON naming ``verdict`` and optionally ``gates``, an object) or at
    the ``verdict_from`` pointer of the ``{out}`` JSON, and must be a string the block declares
    among its outcomes, else ``SCORER_ERROR``. ``output`` is the JSON the verdict was read from."""
    out: Dict[str, Any] = {"code": None, "detail": None, "verdict": None, "gates": None, "output": None}

    def err(code: str, detail: str) -> Dict[str, Any]:
        out.update(code=code, detail=detail)
        return out

    unlocked = list(dict.fromkeys(raw.get("unlocked") or ()))
    if unlocked:
        more = f"; and {len(unlocked) - _SHOWN} more" if len(unlocked) > _SHOWN else ""
        return err(_SCORER_READS_UNLOCKED, (
            f"the scorer reaches past its locks: it {'; '.join(unlocked[:_SHOWN])}{more}. The run refuses each "
            f"reach and records it whatever the scorer does next, so nothing it printed is read (DESIGN 3.2(f): a "
            f"read of any unlocked file fails)"))
    tail = raw.get("stderr_tail") or ""
    said = f"; its last stderr line: {tail}" if tail else ""
    if raw.get("timed_out"):
        return err(_SCORER_ERROR, f"the scorer ran past the run's {raw.get('timeout_s')} s timeout and was "
                                  f"stopped{said}")
    if raw.get("exit") != 0:
        return err(_SCORER_ERROR, f"the scorer exited {raw.get('exit')}{said}")
    if raw.get("watch_problem"):
        return err(_SCORER_ERROR, f"{raw['watch_problem']}, so nothing the scorer printed is read{said}")
    if not raw.get("done"):
        return err(_SCORER_ERROR, f"the scorer exited 0 without the run's closing line (it ended the process "
                                  f"itself, as os._exit does), so nothing it printed is read{said}")
    declared = set(spec["outcomes"])

    def verdict(v: Any, where: str) -> Optional[Dict[str, Any]]:
        if not isinstance(v, str):
            return err(_SCORER_ERROR, f"the verdict at {where} is not a string (it is {v!r})")
        if v not in declared:
            return err(_SCORER_ERROR, f"the scorer prints the verdict {v!r} at {where}, which the locks block does "
                                      f"not declare among its {len(declared)} outcomes")
        return None

    if spec["verdict_from"] == "stdout":
        if raw.get("stdout_over"):
            return err(_SCORER_ERROR, "the scorer printed more on stdout than the run reads; the contract is one "
                                      "line of JSON {\"verdict\": V}")
        data = raw.get("stdout") or b""
        try:
            text = bytes(data).decode("utf-8")
        except UnicodeDecodeError as e:
            return err(_SCORER_ERROR, f"the scorer's stdout is not UTF-8 ({e})")
        body = text[:-1] if text.endswith("\n") else text
        if not body.strip() or "\n" in body:
            return err(_SCORER_ERROR, f"the scorer's stdout is not one line of JSON {{\"verdict\": V}} (it holds "
                                      f"{len(text.splitlines())} lines)")
        try:
            obj = _capsule._strict_json(body)
        except ValueError as e:
            return err(_SCORER_ERROR, f"the scorer's stdout line is not valid strict JSON: {e}")
        if not isinstance(obj, dict):
            return err(_SCORER_ERROR, f"the scorer's stdout line is not a JSON object (it is a {type(obj).__name__})")
        if "verdict" not in obj or any(k not in ("verdict", "gates") for k in obj):
            return err(_SCORER_ERROR, f"the scorer's stdout object names verdict and gates and nothing else; it "
                                      f"holds {sorted(obj)}")
        bad = verdict(obj["verdict"], "stdout")
        if bad is not None:
            return bad
        if "gates" in obj and not isinstance(obj["gates"], dict):
            return err(_SCORER_ERROR, f"the scorer's gates {obj['gates']!r} is not an object")
        out.update(verdict=obj["verdict"], gates=obj.get("gates"), output=obj)
        return out
    pointer = spec["verdict_from"]
    if raw.get("out") is None:
        return err(_SCORER_ERROR, raw.get("out_problem") or f"the scorer wrote no {_OUT_ARG} file")
    try:
        obj = _capsule._strict_json(bytes(raw["out"]))
    except (ValueError, UnicodeDecodeError) as e:
        return err(_SCORER_ERROR, f"the {_OUT_ARG} file is not valid strict JSON: {e}")
    value, missing = _capsule._pointer(obj, pointer)
    if missing is not None:
        return err(_SCORER_ERROR, f"'verdict_from' {pointer!r} does not resolve in the {_OUT_ARG} file: {missing}")
    bad = verdict(value, f"{pointer} of the {_OUT_ARG} file")
    if bad is not None:
        return bad
    out.update(verdict=value, output=obj)
    return out


# ---- licences and the shipped surfaces (DESIGN 3.2(g), 3.4 C8, 3.7; step B5) ---------------------------

try:                                      # the regular expression parser: 3.11 and later name it re._parser
    from re import _parser as _rx_parser
except ImportError:                       # 3.9 and 3.10
    import sre_parse as _rx_parser

_LICENCE_KEYS = ("id", "quote", "pattern")
_WINDOW = 300                             # characters either side of a match searched for a withdrawal
# surface_lint's withdrawal words, pattern for pattern (tests/test_prereg_ci_licences.py pins the two equal)
_WITHDRAWAL = re.compile(r"not\s+yet\s+true\s+of\s+this\s+instrument|(?<![a-z0-9_])withdr(?:aw|awn|aws|awal|awing|ew)"
                         r"(?![a-z0-9_])", re.IGNORECASE)
_PATTERN_MAX_CHARS = 1000
_WS = re.compile(r"\s+")
_CURLY = str.maketrans({"‘": "'", "’": "'"})
# the shipped surfaces (DESIGN 3.2(g), PLAN K14): scripts/surface_lint.py's ``repo`` profile, copied, since
# scripts/ is not in the wheel; tests/test_prereg_ci_licences.py pins the copy equal to the script's. A
# profile is {include, exclude, exts}: globs over repository-relative POSIX paths, where ``*`` and ``?`` stay
# inside one directory and ``**`` crosses them, and the suffixes a surface may carry.
_SURFACE_EXTS = frozenset({".py", ".md", ".rst", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".cff", ".js", ".mjs",
                           ".cjs", ".ts", ".jsx", ".tsx", ".html", ".htm"})
_SURFACE_PROFILE: Dict[str, Any] = {
    "include": ["README.md", "action.yml", "pyproject.toml", "diffgate_action.py", ".pre-commit-hooks.yaml",
                "SECURITY.md", "CONTRIBUTING.md", "CITATION.cff", "CODE_OF_CONDUCT.md", "POSITIONING.md",
                "OATH_CONTRACT.md", "BOUNTY.md", "LEADERBOARD.md", "REPLICATIONS.md", "AUTOPILOT.md",
                "PATENTS.md", "styxx/**", "sworn/**", "integrations/**", "hooks/**", "docs/**", "web/**",
                "packages/**"],
    "exclude": ["web/gate/differential/**"],
    "exts": _SURFACE_EXTS,
}
_SURFACE_PRUNE = frozenset({".git", "node_modules", "__pycache__", "dist", "build", ".netlify", ".venv", "venv"})
_SURFACE_MAX_BYTES = 5_000_000            # a larger surface is listed as skipped, never read (surface_lint's cap)


def _matches_empty(pattern: str, rx: "re.Pattern[str]") -> bool:
    """Whether *pattern* can match the empty string: its parsed minimum width is 0 (a lookaround or an
    optional piece standing alone), which ``rx.fullmatch("")`` alone misses for a lookahead."""
    try:
        return _rx_parser.parse(pattern, rx.flags).getwidth()[0] == 0
    except Exception:                         # noqa: BLE001 - the parser's own refusal falls back to a probe
        return rx.fullmatch("") is not None


def _licence(lic: Any, row: int, verdict: str, n: int, ids: Dict[str, int]) -> Dict[str, Any]:
    """One licence read strictly (``_licence_specs``), or a ``GateSpecError``."""
    where = f"licence {n + 1} of outcome row {row + 1} ({verdict})"
    if not isinstance(lic, dict):
        raise _GateSpecError(f"{where} is not a JSON object (it is a {type(lic).__name__}); a licence is "
                             f"{{\"id\", \"quote\", \"pattern\"}}")
    extra = sorted(k for k in lic if k not in _LICENCE_KEYS)
    if extra:
        raise _GateSpecError(f"{where} holds keys the format does not name: {extra}; a licence names id, quote and "
                             f"pattern and nothing else")
    for key in _LICENCE_KEYS:
        if key not in lic:
            raise _GateSpecError(f"{where} lacks {key!r}; a licence names id, quote and pattern")
    lid, quote, pattern = lic["id"], lic["quote"], lic["pattern"]
    if not isinstance(lid, str) or not _LINE_ID.fullmatch(lid):
        raise _GateSpecError(f"{where}: 'id' {lid!r} is not an id (a letter or digit, then up to 63 letters, digits, "
                             f"'.', '_' or '-'), the name a report prints for the licence")
    if lid in ids:
        raise _GateSpecError(f"the licence id {lid!r} is declared twice (outcome rows {ids[lid] + 1} and {row + 1}); "
                             f"each licence is named once in a decision")
    if not isinstance(quote, str) or not quote.strip():
        raise _GateSpecError(f"{where}: 'quote' {quote!r} is not a non-empty string; it is the sentence as the "
                             f"PREREG writes it")
    if not isinstance(pattern, str) or not pattern.strip():
        raise _GateSpecError(f"{where}: 'pattern' {pattern!r} is not a non-empty string; it is a case-insensitive "
                             f"regular expression over whitespace-collapsed text")
    if len(pattern) > _PATTERN_MAX_CHARS:
        raise _GateSpecError(f"{where}: 'pattern' is {len(pattern)} characters long, over the {_PATTERN_MAX_CHARS} "
                             f"the scan reads")
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise _GateSpecError(f"{where}: 'pattern' {pattern!r} does not compile: {e}")
    if _matches_empty(pattern, rx):
        raise _GateSpecError(f"{where}: 'pattern' {pattern!r} can match the empty string, so it would match every "
                             f"surface and license nothing")
    ids[lid] = row
    return {"id": lid, "row": row, "verdict": verdict, "quote": quote, "pattern": pattern, "rx": rx}


def _licence_table(decision: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """``_licence_specs``'s list and the verdicts the decision declares (every row's, and a gates
    table's ``smoke_verdict``), or ``([], set())`` for a PREREG with no decision."""
    form = decision["form"]
    if form is None:
        return [], set()
    if form == "locks":
        spec = _locks(decision["block"])
        rows = [(i, v, spec["withdraws"].get(v, [])) for i, v in enumerate(spec["outcomes"])]
        declared = set(spec["outcomes"])
    else:
        try:
            spec = _capsule._strict_json(decision["block"])
        except (ValueError, UnicodeDecodeError) as e:
            raise _GateSpecError(f"gates block is not valid strict JSON: {e}")
        _table(spec, smoke=False)
        rows = [(i, r["verdict"], r.get("withdraws", [])) for i, r in enumerate(spec["outcomes"])]
        declared = {v for _, v, _ in rows}
        if isinstance(spec["smoke_verdict"], str):
            declared.add(spec["smoke_verdict"])
    specs: List[Dict[str, Any]] = []
    ids: Dict[str, int] = {}
    for i, verdict, lics in rows:
        if not isinstance(lics, list):
            raise _GateSpecError(f"outcome row {i + 1} ({verdict}): 'withdraws' is a list of licences, not a "
                                 f"{type(lics).__name__}")
        for n, lic in enumerate(lics):
            specs.append(_licence(lic, i, verdict, n, ids))
    return specs, declared


def _licence_specs(decision: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The licences a decision (``_decision``'s reading) declares, DESIGN 3.2(g), read strictly in
    table order: each ``{id, row, verdict, quote, pattern, rx}``, ``rx`` the pattern compiled
    case-insensitively; a gates row's ``withdraws`` list, or the locks block's map read per declared
    outcome. The table reads before any licence does, as ``rescore`` reads it (a refusal of ``_table``
    or ``_locks`` comes ahead of a licence's). A licence is refused when it is not an object of exactly id, quote
    and pattern, its id is not an id or is declared twice, its quote or pattern is empty or not a
    string, or its pattern is over ``_PATTERN_MAX_CHARS``, does not compile, or can match the empty
    string (``GateSpecError``). ``[]`` for a PREREG with no decision."""
    return _licence_table(decision)[0]


class _Collapsed:
    """A surface's text two ways: ``source`` is the text as ``Path.read_text`` would read it (CRLF and
    a lone CR as LF) with the curly apostrophes U+2018 and U+2019 read as the straight one; ``text`` is
    ``source`` with every run of whitespace one space, the string the patterns run over; ``at(i)`` is
    the source position of collapsed position *i* (a run's start for its one space; ``len(source)`` for
    ``len(text)``)."""
    __slots__ = ("text", "source", "_c", "_s")

    def __init__(self, raw: str):
        source = raw.replace("\r\n", "\n").replace("\r", "\n").translate(_CURLY)
        pieces: List[str] = []
        c: List[int] = []
        s: List[int] = []
        pos = n = 0
        for m in _WS.finditer(source):
            if m.start() > pos:
                pieces.append(source[pos:m.start()])
                c.append(n)
                s.append(pos)
                n += m.start() - pos
            pieces.append(" ")
            c.append(n)
            s.append(m.start())
            n += 1
            pos = m.end()
        if pos < len(source):
            pieces.append(source[pos:])
            c.append(n)
            s.append(pos)
            n += len(source) - pos
        c.append(n)
        s.append(len(source))
        self.text, self.source, self._c, self._s = "".join(pieces), source, c, s

    def at(self, i: int) -> int:
        k = bisect.bisect_right(self._c, i) - 1
        return self._s[k] + (i - self._c[k])


def _collapse(raw: str) -> _Collapsed:
    return _Collapsed(raw)


def _line_col(source: str, pos: int) -> Tuple[int, int]:
    """The 1-based line and column of *pos* in *source*."""
    return source.count("\n", 0, pos) + 1, pos - source.rfind("\n", 0, pos)


def _scan(text: str, specs: Sequence[Dict[str, Any]], path: str) -> List[Dict[str, Any]]:
    """Every match of every licence in *specs* on one surface, sorted by line and column (then
    licence order): ``{id, path, line, col, text, withdrawn}``. ``text`` is the matched text as the
    collapsed string holds it; line and column name the match's start in the source text. A match
    is ``withdrawn`` when a withdrawal (``_WITHDRAWAL``: "not yet true of this instrument", or a form
    of "withdraw") lies whole within ``_WINDOW`` characters of it on either side in the source text,
    ``scripts/surface_lint.py``'s window rule re-implemented over compiled patterns, so the sentence
    that withdraws a claim may quote it (surface_lint searches the window with ``pos`` and
    ``endpos``, which reads a withdrawal cut by the window's edge as one; here it lies whole inside).
    A match of no characters, which a pattern may give at some positions and not at the empty
    string, is passed over."""
    c = _collapse(text)
    spans = [(m.start(), m.end()) for m in _WITHDRAWAL.finditer(c.source)]
    starts = [s for s, _ in spans]
    out: List[Tuple[Tuple[int, int, int], Dict[str, Any]]] = []
    for k, spec in enumerate(specs):
        for m in spec["rx"].finditer(c.text):
            if m.end() == m.start():
                continue
            s0, s1 = c.at(m.start()), c.at(m.end())
            lo, hi = s0 - _WINDOW, s1 + _WINDOW
            withdrawn = False
            i = bisect.bisect_left(starts, lo)
            while i < len(spans) and spans[i][0] <= hi:
                if spans[i][1] <= hi:
                    withdrawn = True
                    break
                i += 1
            line, col = _line_col(c.source, s0)
            out.append(((line, col, k), {"id": spec["id"], "path": path, "line": line, "col": col,
                                          "text": m.group(0), "withdrawn": withdrawn}))
    return [m for _, m in sorted(out, key=lambda x: x[0])]


def _licence_finding(code: str, detail: str, spec: Optional[Dict[str, Any]] = None,
                     match: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"code": code, "kind": _KIND[code], "detail": detail,
            "licence": spec["id"] if spec else None, "row": spec["row"] if spec else None,
            "verdict": spec["verdict"] if spec else None,
            "path": match["path"] if match else None, "line": match["line"] if match else None,
            "col": match["col"] if match else None}


def _row_verdicts(decision: Dict[str, Any]) -> List[str]:
    """The verdict of each outcome row of a decision that ``_licence_table`` has read, by index: a gates
    table's rows, or a locks block's declared outcomes."""
    if decision["form"] == "locks":
        return list(_locks(decision["block"])["outcomes"])
    return [r["verdict"] for r in _capsule._strict_json(decision["block"])["outcomes"]]


def licences(prereg: Any, surfaces: Iterable[Any], *, landed: Optional[str] = None,
             row: Optional[int] = None) -> Dict[str, Any]:
    """Read the sentences a decision's rows withdraw against the shipped surfaces (DESIGN 3.2(g),
    3.4 C8, 3.7).

    *prereg* is the PREREG's bytes (a blob) or text. *surfaces* is an iterable of ``(path, text)``
    pairs, each a repository-relative path and the surface's text (``str``; the git layer's
    ``_surfaces`` decodes a blob as surface_lint does), or None for a surface not read (binary, a
    symlink, over the size cap), which is listed under ``skipped``; another shape is a TypeError.
    *landed* is the verdict the receipt landed on (``rescore``'s ``verdict``; the smoke verdict for a
    smoke run), or None before any receipt has landed (the reading ``lint`` makes before data); a
    verdict the decision does not declare is a ValueError. *row* names the landed outcome row by its
    index (``rescore``'s ``row``; a locks block's outcome by its place in ``outcomes``): only that row's
    licences are then withdrawn, so two rows that share a verdict are told apart (the B5 review). A
    row with *landed* None, a row out of range, or a row whose verdict is not *landed* is a
    ValueError.

    Returns a dict with every key present: ``form``, ``gates_sha256``, ``code`` and ``detail`` (the
    decision's own reading where there are no licences to read: ``NOT_GATED``, or ``SCORER_REFUSES``
    for a fence the scanner refuses, a table ``rescore`` refuses or a licence the format does not
    read, each as its one finding), ``landed``, ``surfaces`` (how many were read), ``skipped``,
    ``licences`` (one entry per licence in table order: ``id``, ``row``, ``verdict``, ``quote``,
    ``pattern``, ``surfaces``, the paths with a match, and ``matches``, each as ``_scan`` gives it,
    sorted by path, line and column, the withdrawn ones included, so the author sees a pattern's
    coverage), and ``findings``, each ``{code, kind, detail, licence, row, verdict, path, line, col}``
    in licence order, then by path, line and column:

    * once a receipt has landed on a row, every match of that row's licences with no withdrawal in
      window is ``LICENCE_WITHDRAWN`` (FAIL), naming the file, line and column and the row; the
      other rows' licences withdraw nothing, whatever they match;
    * before any receipt lands, such a match is ``LICENCE_PENDING`` (NOTE: the sentence stands on a
      verdict not yet given), and a licence whose pattern matches no surface at all is
      ``LICENCE_PATTERN_FINDS_NOTHING`` (FAIL: a licence that matches nothing before data licenses
      nothing, the analogue of a vacuous gate).

    Licences match strings (DESIGN 7, risk 3): a pattern registered before data does not reach a
    wording written after it, and the report prints that as the boundary (C9), beside this reading.
    """
    out: Dict[str, Any] = {"form": None, "gates_sha256": None, "code": None, "detail": None, "landed": landed,
                           "surfaces": 0, "skipped": [], "licences": [], "findings": []}
    decision = _decision(prereg)
    out.update(form=decision["form"], gates_sha256=decision["gates_sha256"])
    if decision["form"] is None:
        out.update(code=decision["code"], detail=decision["detail"])
        out["findings"] = [_licence_finding(decision["code"], decision["detail"] or "")]
        return out
    try:
        specs, declared = _licence_table(decision)
    except _GateSpecError as e:
        out.update(code=_SCORER_REFUSES, detail=str(e))
        out["findings"] = [_licence_finding(_SCORER_REFUSES, str(e))]
        return out
    if landed is not None and landed not in declared:
        raise ValueError(f"landed {landed!r} is not a verdict the decision declares ({sorted(declared)})")
    if row is not None:
        verdicts = _row_verdicts(decision)
        if landed is None:
            raise ValueError("row names the row a receipt landed on, and landed is None (no receipt has landed)")
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < len(verdicts):
            raise ValueError(f"row {row!r} is not the index of one of the decision's {len(verdicts)} outcome rows")
        if verdicts[row] != landed:
            raise ValueError(f"row {row} reads {verdicts[row]!r}, not the landed verdict {landed!r}")
    by_id: Dict[str, List[Dict[str, Any]]] = {s["id"]: [] for s in specs}
    skipped: List[str] = []
    read = 0
    for item in surfaces:
        if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[0], str):
            raise TypeError(f"a surface is a (path, text) pair with a str path, not {item!r:.80}")
        path, text = item
        if text is None:
            skipped.append(path)
            continue
        if not isinstance(text, str):
            raise TypeError(f"a surface's text is str (a blob decoded as the git layer decodes it) or None, not "
                            f"{type(text).__name__}: {path}")
        read += 1
        if specs:
            for m in _scan(text, specs, path):
                by_id[m["id"]].append(m)
    out.update(surfaces=read, skipped=skipped)
    for spec in specs:
        matches = sorted(by_id[spec["id"]], key=lambda m: (m["path"], m["line"], m["col"]))
        out["licences"].append({"id": spec["id"], "row": spec["row"], "verdict": spec["verdict"],
                                "quote": spec["quote"], "pattern": spec["pattern"],
                                "surfaces": sorted({m["path"] for m in matches}), "matches": matches})
        standing = [m for m in matches if not m["withdrawn"]]
        if landed is None:
            for m in standing:
                out["findings"].append(_licence_finding(_LICENCE_PENDING, (
                    f"{m['path']}:{m['line']}:{m['col']} reads {m['text']!r}, a wording of licence {spec['id']} "
                    f"({spec['quote']!r}), which outcome row {spec['row'] + 1} ({spec['verdict']}) withdraws once a "
                    f"receipt lands on it; no receipt has landed, so the sentence stands on a verdict not yet "
                    f"given"), spec, m))
            if not matches:
                out["findings"].append(_licence_finding(_LICENCE_PATTERN_FINDS_NOTHING, (
                    f"licence {spec['id']} ({spec['quote']!r}): its pattern {spec['pattern']!r} matches none of the "
                    f"{read} surfaces read; a licence that matches nothing before data licenses nothing, the "
                    f"analogue of a vacuous gate (DESIGN 3.7)"), spec))
        elif (spec["row"] == row) if row is not None else (spec["verdict"] == landed):
            for m in standing:
                out["findings"].append(_licence_finding(_LICENCE_WITHDRAWN, (
                    f"{m['path']}:{m['line']}:{m['col']} reads {m['text']!r}, a wording of licence {spec['id']} "
                    f"({spec['quote']!r}), which the landed outcome row {spec['row'] + 1} ({spec['verdict']}) "
                    f"withdraws, and no withdrawal lies within {_WINDOW} characters of it (DESIGN 3.4 C8)"),
                    spec, m))
    return out


def _quote_coverage(prereg: Any, surfaces: Iterable[Any]) -> List[Dict[str, Any]]:
    """Each licence's ``quote`` read as a literal pattern over the same surfaces (DESIGN 4.A, A.3; the
    design's RV3 counts it as Q): ``{id, quote, surfaces}`` in licence order, ``surfaces`` the paths it
    matches. The quote is read as the scan reads text (whitespace runs collapsed, curly apostrophes
    straightened) and matched case-insensitively. A quote that matches nothing is the pattern
    ``lint`` would have refused (``LICENCE_PATTERN_FINDS_NOTHING``) had it been registered as the
    pattern; the report prints it beside the pattern's own coverage. A licence the format does not read
    is a ``GateSpecError``, as in ``_licence_specs``."""
    specs = _licence_specs(_decision(prereg))
    items = [(path, text) for path, text in surfaces if text is not None]
    out = []
    for spec in specs:
        literal = _collapse(spec["quote"]).text.strip()
        probe = {"id": spec["id"], "rx": re.compile(re.escape(literal), re.IGNORECASE)}
        hit = sorted({path for path, text in items if literal and _scan(text, [probe], path)})
        out.append({"id": spec["id"], "quote": spec["quote"], "surfaces": hit})
    return out


# ---- the outcome ladder and the single-gate flip table (DESIGN 3.4 C7, step B7) ---------------------------

# a gate's margin: how far its metric sits past the bar in the direction that fires it, so a gate fires
# with a positive margin (">=" and "<=" also at 0, "==" only at 0) and does not fire with a negative one
_MARGIN = {">=": lambda v, b: v - b, ">": lambda v, b: v - b, "<=": lambda v, b: b - v, "<": lambda v, b: b - v,
           "==": lambda v, b: -abs(v - b)}


def _ladder(rescored: Dict[str, Any], prereg: Any, receipt: Any) -> Optional[Dict[str, Any]]:
    """The outcome ladder of one scored receipt (DESIGN 3.4 C7), or None where there is none to print: a
    re-score that refused, a smoke receipt (scored by type, on no row), or a locks decision (its scorer
    prints the verdict, and no gate table is read).

    *rescored* is ``rescore``'s output for *prereg* (the decision in effect) and *receipt*. Returns
    ``{landed, verdict, rows, gates, flips}``: ``rows`` every outcome row in order (``row``, ``verdict``,
    ``when``, ``landed``); ``gates`` each gate in table order with ``metric``, ``op``, ``bar``,
    ``observed`` (the value the scorer read at the receipt pointer), ``fired``, ``margin`` (``_MARGIN``)
    and ``read_by_landed`` (whether the landed row's ``when`` names it); ``flips`` the single-gate flip
    table, one entry per gate: the row and verdict the receipt would land on with that gate's reading
    flipped (None where no row matches: a hole), and ``moves``. Every gate whose flip moves the
    verdict is listed with its margin, and no single bar is called the decisive one (RED_TEAM m3)."""
    if (rescored.get("form") != "gates" or rescored.get("code") is not None or rescored.get("smoke")
            or rescored.get("row") is None):
        return None
    decision = _decision(prereg)
    if decision["form"] != "gates" or decision["gates_sha256"] != rescored.get("gates_sha256"):
        raise ValueError("the PREREG is not the decision this re-score read")
    spec = _capsule._strict_json(decision["block"])
    metrics, missing = _capsule._pointer(receipt, spec.get("receipt_pointer", ""))
    if missing is not None:
        raise ValueError(f"the receipt pointer does not resolve: {missing}")
    fired = rescored["gates"]
    whens = [r["when"] for r in spec["outcomes"]]
    landed = rescored["row"]
    gates = []
    for name, g in spec["gates"].items():
        observed = _protocol._resolve(metrics, g["metric"])
        margin = _MARGIN[g["op"]](observed, g["value"])
        if margin == 0:
            margin = abs(margin)                  # no negative zero in a report
        gates.append({"name": name, "metric": g["metric"], "op": g["op"], "bar": g["value"], "observed": observed,
                      "fired": fired[name], "margin": margin, "read_by_landed": name in whens[landed]})
    flips = []
    for g in gates:
        moved = dict(fired)
        moved[g["name"]] = not moved[g["name"]]
        i = _landed(whens, moved)
        verdict = spec["outcomes"][i]["verdict"] if i is not None else None
        flips.append({"gate": g["name"], "fired": g["fired"], "row": i, "verdict": verdict,
                      "moves": verdict != rescored["verdict"], "margin": g["margin"]})
    rows = [{"row": i, "verdict": r["verdict"], "when": r["when"], "landed": i == landed}
            for i, r in enumerate(spec["outcomes"])]
    return {"landed": landed, "verdict": rescored["verdict"], "rows": rows, "gates": gates, "flips": flips}


@functools.lru_cache(maxsize=256)
def _glob(pattern: str) -> "re.Pattern[str]":
    """*pattern* as a regular expression over a POSIX path: ``**`` matches across ``/`` (``/**/`` also
    matches no directory at all), ``*`` and ``?`` within one segment, every other character itself."""
    out: List[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("/**/", i):
            out.append("/(?:.*/)?")
            i += 4
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))


def _surface_selected(path: str, profile: Dict[str, Any]) -> bool:
    """Whether *path* (repository-relative, POSIX) is a surface under *profile*, surface_lint's rule:
    an include glob matches, no exclude glob does, no directory of it is pruned (a build or cache
    directory), and its suffix, whatever its case, is one the profile reads, or it has none (an
    extensionless script such as a git hook; a name whose only dots lead, like ``.hidden``, has none)."""
    if not any(_glob(p).fullmatch(path) for p in profile["include"]):
        return False
    if any(_glob(p).fullmatch(path) for p in profile["exclude"]):
        return False
    parts = path.split("/")
    if any(part in _SURFACE_PRUNE for part in parts[:-1]):
        return False
    stem = parts[-1].lstrip(".")
    if "." in stem:
        return ("." + stem.rsplit(".", 1)[1]).lower() in profile["exts"]
    return True


def _surface_text(data: bytes) -> Optional[str]:
    """A blob as surface_lint reads it: None for a binary (a NUL byte in its head), else UTF-8 with
    the bytes it cannot decode replaced."""
    data = bytes(data)
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")
