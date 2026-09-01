# -*- coding: utf-8 -*-
"""styxx.sworn — sworn output v0.1: the author declares, the receipt disposes.

Spec: ``papers/sworn/SPEC_sworn_output_v01_2026_09_01.md``, frozen before this file existed.
Format ``sworn/0.1``. Manifest ``sworn/manifest/0.1``. Verdict receipt
``styxx.sworn.verdict-receipt/v0``.

THE INVARIANT.  *The author chooses what to swear; the author cannot choose what the receipt
says.*  A ``<sworn r="RECEIPT" k="KIND">…</sworn>`` span binds one sentence, at write time, to
bytes the author could not have written. Everything outside a span is NARRATIVE by definition —
not by a verifier's judgement — and narrative is never accused. The verifier is never handed a
target; it is handed commitments, and it checks them.

WHY THIS SHAPE.  Every instrument this lab built to *find* claims in prose was measured against
readers who did not write that prose and did not survive it (0.23 in the wild, 0.16 held-out
after repair, 0.4211 at the best structural attempt — receipts cited in the spec). Those
instruments were handed their target by the text they judged. This one is handed nothing.

WHAT THIS MODULE IS, MECHANICALLY.
  * a byte-exact lexer: tags are recognised only outside fenced code regions and inline backtick
    spans, by one exact byte pattern; anything tag-shaped that is not the pattern is MALFORMED,
    never narrative (silent downgrade is how a format gets gamed);
  * a canonicalizer whose round trip is ASSERTED: tags deleted, byte offsets recorded, tags
    re-inserted, bytes compared — and a sidecar is refused rather than emitted if they differ;
  * a resolver for exactly three receipt forms, each a pure function of bytes: ``rN`` from a
    harness-minted manifest, ``path:`` at the commit the document names (never a working tree),
    ``prereg:SHA256`` by content address;
  * four check kinds (numeric, quote, hash, absent) with no percent conversion, no float
    comparison, no unicode normalisation, no search over leaves: the author named the leaf, and
    the binding is right or wrong with no coincidence available to it;
  * a verdict receipt that is content-addressed and re-derivable (the parrhesia discipline), and
    that prints its coverage and its UNRESOLVED count at the same prominence as its verdict.

WHAT IT REFUSES TO BE.  No function here reads plain text and proposes spans (invariant 1). No
verdict here blocks anything — sworn reports, it does not gate. No ``exec`` kind (v0.2, capsule
layer). And a document that swore nothing is ``UNSWORN``, never "no failures".

IMPLEMENTATION DECISIONS the frozen spec left open are named in :data:`DECISIONS`, carried in
every verdict receipt, and pinned by ``tests/test_sworn.py``. They are decisions, not spec.

THE ARC.  The spec records its INDEX declaration as owed because ``papers/INDEX.md`` did not
exist when it was frozen. It exists now; ``papers/sworn/`` carries a row, and the two sworn
documents in the tree (``DECLARATION_h_mapping_2026_09_01.md`` and this module's own RESULT) are
re-derived by ``tests/test_sworn_dogfood.py``. Nothing here is a measurement of sworn output:
that is owed item 3 of the spec and it is still owed.

CLI::

    python -m styxx.sworn canon    DOC.md  [--commit SHA] [--manifest M.json] [--out DOC.sworn.json]
    python -m styxx.sworn render   DOC.sworn.json [--out DOC.md]
    python -m styxx.sworn verify   DOC.md|DOC.sworn.json [--repo .] [--commit SHA] [--manifest M.json] [--out RECEIPT.json]
    python -m styxx.sworn check    RECEIPT.json DOC.md|DOC.sworn.json [--repo .] [--manifest M.json]
    python -m styxx.sworn manifest new M.json --harness NAME --turn ID
    python -m styxx.sworn manifest add M.json --id r1 --file F --kind tool_stdout [--complete]
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "SPEC", "MANIFEST_SPEC", "RECEIPT_SCHEMA", "SPAN_CAP_BYTES", "KINDS", "VERDICTS",
    "DECISIONS", "REASONS",
    "scan", "Scan", "Declaration",
    "to_sidecar", "render", "load_sidecar",
    "Manifest", "MemoryTree", "GitTree",
    "verify", "issue_receipt", "verify_receipt",
    "main",
]

SPEC = "sworn/0.1"
MANIFEST_SPEC = "sworn/manifest/0.1"
RECEIPT_SCHEMA = "styxx.sworn.verdict-receipt/v0"
SPAN_CAP_BYTES = 300
KINDS = ("numeric", "quote", "hash", "absent")
RESERVED_KINDS = ("exec",)
VERDICTS = ("HELD", "FAILED", "UNRESOLVED", "MALFORMED", "WITHHELD")
ROUNDING = "ROUND_HALF_EVEN"
MAX_RECEIPT_BYTES = 64 * 1024 * 1024      # a blob over this is UNRESOLVED receipt_too_large, unread

# The manifest's `kind_of_source` vocabulary, closed. Invariant 2: a receipt the agent wrote in the
# same turn is the agent swearing to itself. The verifier cannot see the turn; it can see what the
# harness RECORDED about where the bytes came from, and it refuses author-side kinds by name. An
# unknown kind is refused too — a kind nobody defined is not a kind the harness attested.
SOURCE_KINDS_EXTERNAL = frozenset({
    "tool_stdout", "tool_stderr", "file_read", "http_fetch", "harness_note", "test_report",
})
SOURCE_KINDS_AUTHOR = frozenset({"agent_output", "agent_file_write", "agent_message"})

# Every non-HELD verdict carries one of these. A closed set, so a consumer can key on it.
REASONS = (
    # decidable from the document bytes alone (MALFORMED)
    "tag_syntax", "nesting", "stray_closer", "unclosed", "empty_span", "length_cap",
    "receipt_form", "kind_unknown", "kind_reserved", "number_count", "number_grammar",
    "needle_count", "needle_empty", "digest_form", "absent_over_partial", "hash_over_partial",
    # decidable from the declaration plus the object the author named (MALFORMED: the author
    # had those exact bytes when it wrote the fragment)
    "pointer_unresolvable", "pointer_ambiguous", "anchor_out_of_range", "leaf_not_scalar",
    "leaf_not_numeric", "leaf_not_string", "receipt_not_json", "receipt_author_minted",
    "kind_of_source_unknown",
    # the verifier could not see the evidence (UNRESOLVED — never an accusation)
    "manifest_absent", "manifest_spec_unknown", "manifest_id_missing", "manifest_integrity",
    "manifest_bytes_absent", "manifest_no_completeness", "no_repository", "no_commit",
    "commit_absent", "path_absent", "not_a_blob", "receipt_too_large", "git_unavailable",
    "prereg_not_in_tree",
    # the check ran and did not pass (FAILED)
    "value_mismatch", "needle_missing", "needle_present", "digest_mismatch",
    # document level
    "unbalanced_fences", "invalid_utf8",
)

# The frozen spec leaves these to the implementation. Each is a decision, stated once, carried in
# every receipt under `verifier.decisions`, and pinned by a test. None of them changes the spec.
DECISIONS = {
    "fence": ("a line whose first bytes are 0-3 ASCII spaces then three or more backticks is a "
              "fence delimiter; every delimiter toggles; an odd count is document-level MALFORMED. "
              "Tilde fences, blockquoted fences and 4-space indented code are NOT fences (spec-literal; "
              "not a markdown parse)."),
    "code_span": ("inline backtick spans match runs of equal length on the same line; an unmatched "
                  "run is literal"),
    "tag_grammar": ("exactly `<sworn r=\"…\" k=\"…\">` with single spaces, double quotes, lowercase, "
                    "and `</sworn>`; any other tag-shaped candidate (`<sworn`/`</sworn` followed by a "
                    "non-name byte) is MALFORMED, never narrative"),
    "lexical_malformed_refuses_sidecar": ("a document with tag_syntax/nesting/stray_closer/unclosed "
                                          "has no canonical text, and a zero-byte span has no "
                                          "representable offsets; both can be verified inline but a "
                                          "sidecar is refused, and a sidecar carrying either shape "
                                          "is refused on load"),
    "sworn_total_counts_malformed": ("sworn_total includes MALFORMED; UNSWORN is reserved for a "
                                     "document with no tag-shaped candidate at all"),
    "document_malformed_is_failed": ("unbalanced fences or undecodable UTF-8 yield SWORN-FAILED with "
                                     "document_malformed set, never UNSWORN"),
    "empty_span": "an inner text of zero bytes, or only ASCII whitespace, is MALFORMED for every kind",
    "number_grammar": ("the span is cut into MAXIMAL tokens of [\\w.,+-−%/±:]; exactly one token may "
                       "carry a digit and, after stripping trailing . , : it must be entirely one "
                       "number: [-+−]?(digits with optional ,thousands)(.digits)?%? or .digits%?. Any "
                       "other digit-bearing token (r1, v0.1, STRUCT-1, sha256, 1e-5, 3/4, 2026-09-01, "
                       "0.55-0.60, ±0.02, ٣) is MALFORMED — no identifier whitelist, no date/sha/version "
                       "scrub, no partial extraction possible by construction. `42%` is the number 42 "
                       "and no conversion happens; `0.55.` is 0.55; `23,247.` is 23247"),
    "rounding": "the receipt scalar is a Decimal from the JSON text, quantized to the printed "
                "fractional digits with ROUND_HALF_EVEN, compared as Decimal (never a float)",
    "numeric_leaf": ("the addressed value must be a JSON number (read as Decimal from its own "
                     "digits); a string, bool, null, NaN, Infinity, array or object is MALFORMED — a "
                     "string that happens to spell a number is not parsed, that would be a guess"),
    "numeric_address": ("numeric reads the receipt as JSON with numbers as Decimal; bytes that are "
                        "not UTF-8 JSON are MALFORMED receipt_not_json (the author named a fragment "
                        "or scalar inside bytes it could read); a duplicated key ON the pointer path "
                        "is MALFORMED pointer_ambiguous, a duplicate off the path is irrelevant; the "
                        "address is the JSON pointer, or the root when there is none; a line-anchored "
                        "slice must itself be one JSON number"),
    "needle": ("for quote and absent, the stated string is the content of exactly ONE inline "
               "backtick code span inside the sworn text, bytes verbatim, no trimming; a needle of "
               "zero bytes or only ASCII whitespace is MALFORMED"),
    "quote_pointer": "quote against a JSON pointer compares the needle with a string leaf's UTF-8 bytes",
    "hash_token": ("the stated hash is exactly one run of 64 hex characters in the span; any run of "
                   "32/40/96/128 hex, or two 64-runs, is MALFORMED"),
    "line_anchor": ("#L13 / #L13-L20 are 1-based, inclusive, split on LF only, CR retained, "
                    "interior newlines kept, the final line's terminating LF excluded; a line past "
                    "EOF is MALFORMED; a trailing LF does not begin an empty last line"),
    "rn_grammar": "^r[1-9][0-9]*$, no fragment",
    "path_grammar": ("relative, /-separated, no empty/./.. segment, no backslash, no whitespace; "
                     "split at the FIRST #; fragment is `/`-pointer (RFC 6901, ~0 ~1 only) or Ln[-Lm]"),
    "prereg_search": "the tree at the sidecar's commit, blobs only, memoised per commit",
    "manifest_bytes": "standard base64; an entry whose bytes do not hash to its sha256 is UNRESOLVED",
    "author_minted": ("kind_of_source in {agent_output, agent_file_write, agent_message}, or a "
                      "receipt sha256 listed in the manifest's authored_sha256 (every byte-object the "
                      "agent produced this turn, recorded by the harness), is MALFORMED "
                      "receipt_author_minted; a kind outside the closed vocabulary is MALFORMED "
                      "kind_of_source_unknown; complete missing from an rN used with absent is "
                      "UNRESOLVED manifest_no_completeness, complete:false is MALFORMED"),
    "coverage": ("numerator sworn_total; denominator adds sentences of the narrative (canonical text "
                 "minus sworn spans minus fenced regions) that styxx.claimdetect reads as claims; the "
                 "splitter is diffgate's `(?<=[.!?])\\s+|\\n+`; 0/0 is null; ALWAYS advisory"),
    "exit_codes": ("verify exits 0 for EVERY document verdict — SWORN-HELD, SWORN-FAILED, UNSWORN, "
                   "document-level MALFORMED — because sworn reports and never gates; a refusal "
                   "(undecodable document, a sidecar that cannot round-trip or carries an unknown "
                   "shape, a manifest that disagrees with the embedded one) is SystemExit('REFUSED: "
                   "…'), exit status 1, nothing written; check exits 1 when a receipt does not "
                   "re-derive"),
    "html_comments": ("a tag inside an HTML comment is recognised like any other (the spec's lexical "
                      "rules are closed); a hidden commitment inflating coverage is an owed v0.2 item"),
}

_CERTIFIES = (
    "the spans the author bound were checked against bytes the author did not write, at the commit "
    "or manifest the document names — NOT a claim that the document is correct, NOT a claim that the "
    "right sentences were bound, NOT a check that the tags were written at write time, and only as "
    "trustworthy as the harness that minted the manifest and the history that holds the commit"
)
_COVERAGE_CEILING = ("advisory: the denominator is counted by styxx.claimdetect (STRUCT-1), measured "
                     "at precision 0.4211 on n=38 with two known recall misses; false flags bias "
                     "coverage low, misses bias it high; never a gate, never a measurement")


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jcs(obj: Any) -> str:
    from styxx.attestation import jcs
    return jcs(obj)


# =============================================================================================
# 1. the lexer — bytes in, declarations out
# =============================================================================================

_CANDIDATE = re.compile(rb"<(/?)[sS][wW][oO][rR][nN](?![A-Za-z0-9_\-])")
_OPENER = re.compile(rb'<sworn r="([^"<>\r\n]*)" k="([^"<>\r\n]*)">')
_CLOSER = b"</sworn>"
_FENCE = re.compile(rb"^ {0,3}`{3,}")
_TICKS = re.compile(rb"`+")


class Declaration(dict):
    """One tag-shaped thing the lexer met. A dict so it serialises as-is.

    Keys: ``at`` (byte offset of the declaration in the INLINE document), ``receipt``, ``kind``
    (strings, verbatim, or None), ``inner`` (bytes of the sworn text, or None), ``start``/``end``
    (canonical byte offsets, or None while canonical text is undefined), ``malformed`` (a reason
    from :data:`REASONS`, or None), ``raw`` (the offending bytes for a syntax error).
    """


class Scan(dict):
    """What the lexer saw: ``declarations``, ``fenced`` regions, ``document_malformed``,
    ``canonical`` bytes (None when a lexical MALFORMED makes it undefined), ``lexical_ok``."""


def _fenced_regions(raw: bytes) -> Tuple[List[Tuple[int, int]], List[int], bool]:
    """(regions, delimiter line numbers, balanced). Spec-literal: a delimiter line toggles."""
    regions: List[Tuple[int, int]] = []
    delims: List[int] = []
    open_at: Optional[int] = None
    pos = 0
    ln = 0
    n = len(raw)
    while pos < n:
        ln += 1
        nl = raw.find(b"\n", pos)
        end = n if nl < 0 else nl + 1
        line = raw[pos:end]
        if _FENCE.match(line):
            delims.append(ln)
            if open_at is None:
                open_at = pos
            else:
                regions.append((open_at, end))
                open_at = None
        pos = end
    if open_at is not None:
        return regions, delims, False
    return regions, delims, True


def _in_regions(p: int, regions: List[Tuple[int, int]]) -> Optional[int]:
    for a, b in regions:
        if a <= p < b:
            return b
    return None


def scan(raw: bytes) -> Scan:
    """Lex an inline sworn document. Pure: bytes in, declarations out. Never raises on content."""
    out = Scan(declarations=[], fenced=[], document_malformed=None, canonical=None,
               lexical_ok=True, candidates=0)
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        out["document_malformed"] = {"reason": "invalid_utf8", "at": e.start}
        out["lexical_ok"] = False
        return out
    regions, delims, balanced = _fenced_regions(raw)
    out["fenced"] = regions
    if not balanced:
        out["document_malformed"] = {"reason": "unbalanced_fences", "delimiter_lines": delims}
        out["lexical_ok"] = False
        return out

    decls: List[Declaration] = []
    stack: List[Declaration] = []          # open declarations, innermost last
    p = 0
    n = len(raw)
    while p < n:
        skip_to = _in_regions(p, regions)
        if skip_to is not None:
            p = skip_to
            continue
        c = raw[p:p + 1]
        if c == b"`":
            run = _TICKS.match(raw, p).end() - p
            nl = raw.find(b"\n", p + run)
            line_end = n if nl < 0 else nl
            q = p + run
            closed = None
            while q < line_end:
                m = _TICKS.search(raw, q, line_end)
                if not m:
                    break
                if m.end() - m.start() == run:
                    closed = m.end()
                    break
                q = m.end()
            p = closed if closed is not None else p + run
            continue
        if c == b"<":
            cm = _CANDIDATE.match(raw, p)
            if cm:
                out["candidates"] += 1
                om = _OPENER.match(raw, p)
                if om:
                    d = Declaration(at=p, opener_end=om.end(),
                                    receipt=om.group(1).decode("utf-8"),
                                    kind=om.group(2).decode("utf-8"),
                                    closer_at=None, closer_end=None, inner=None,
                                    start=None, end=None, malformed=None)
                    if stack:
                        # nested: BOTH spans are MALFORMED (spec), and the new one still pushes
                        # so later closers pair naturally and are reported once each.
                        for o in stack:
                            o["malformed"] = o["malformed"] or "nesting"
                        d["malformed"] = "nesting"
                    stack.append(d)
                    decls.append(d)
                    p = om.end()
                    continue
                if raw.startswith(_CLOSER, p):
                    if stack:
                        d = stack.pop()
                        d["closer_at"] = p
                        d["closer_end"] = p + len(_CLOSER)
                        d["inner"] = raw[d["opener_end"]:p]
                        if not d["inner"]:
                            # zero bytes sworn: MALFORMED from the bytes, and unrepresentable in
                            # the sidecar (start == end cannot be ordered against a neighbour)
                            d["malformed"] = d["malformed"] or "empty_span"
                    else:
                        decls.append(Declaration(at=p, receipt=None, kind=None, inner=None,
                                                 start=None, end=None, malformed="stray_closer",
                                                 raw=_CLOSER.decode("ascii")))
                    p += len(_CLOSER)
                    continue
                # tag-shaped, not the pattern: MALFORMED, never narrative
                gt = raw.find(b">", p)
                nl = raw.find(b"\n", p)
                stop = min(x for x in (gt + 1 if gt >= 0 else n, nl if nl >= 0 else n, n))
                decls.append(Declaration(at=p, receipt=None, kind=None, inner=None, start=None,
                                         end=None, malformed="tag_syntax",
                                         raw=raw[p:stop].decode("utf-8", errors="replace")))
                p = max(stop, p + 1)
                continue
        p += 1
    for d in stack:
        d["malformed"] = d["malformed"] or "unclosed"

    out["declarations"] = decls
    lexical_bad = [d for d in decls
                   if d["malformed"] in ("tag_syntax", "nesting", "stray_closer", "unclosed")]
    out["lexical_ok"] = not lexical_bad
    if out["lexical_ok"]:
        # canonical text: every recognised tag deleted, nothing else changed.
        cuts = []
        for d in decls:
            cuts.append((d["at"], d["opener_end"]))
            cuts.append((d["closer_at"], d["closer_end"]))
        cuts.sort()
        pieces = []
        last = 0
        removed = 0
        boundaries = {}          # inline offset -> canonical offset, at each cut edge
        for a, b in cuts:
            pieces.append(raw[last:a])
            boundaries[a] = a - removed
            removed += b - a
            boundaries[b] = b - removed
            last = b
        pieces.append(raw[last:])
        canonical = b"".join(pieces)
        for d in decls:
            d["start"] = boundaries[d["opener_end"]]
            d["end"] = boundaries[d["closer_at"]]
        out["canonical"] = canonical
    return out


# =============================================================================================
# 2. canonical form — sidecar out, inline back, byte-exact or refused
# =============================================================================================

def _opener_bytes(receipt: str, kind: str) -> bytes:
    return ('<sworn r="%s" k="%s">' % (receipt, kind)).encode("utf-8")


def render(sidecar: dict) -> bytes:
    """Sidecar → inline bytes. Events in ascending order, applied in reverse: on a shared offset an
    earlier span's closer lands before a later span's opener, which is the only order that
    reproduces `</sworn><sworn …>` adjacency byte-for-byte."""
    text = sidecar["text"].encode("utf-8")
    events: List[Tuple[int, bytes]] = []
    for s in sidecar["spans"]:
        events.append((s["start"], _opener_bytes(s["receipt"], s["kind"])))
        events.append((s["end"], _CLOSER))
    out = text
    for off, tag in reversed(events):
        out = out[:off] + tag + out[off:]
    return out


def to_sidecar(raw: bytes, name: str, commit: Optional[str] = None,
               manifest: Optional["Manifest"] = None) -> dict:
    """Inline bytes → the sidecar object. REFUSES (SystemExit) rather than emit a sidecar that
    cannot round-trip — the capsule discipline: a record that does not reproduce is not kept."""
    sc = scan(raw)
    if sc["document_malformed"]:
        raise SystemExit("REFUSED: document-level MALFORMED (%s) — no canonical text exists"
                         % sc["document_malformed"]["reason"])
    if commit is not None and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(commit)):
        raise SystemExit("REFUSED: commit must be a full lowercase hex object id or None, not %r"
                         % (commit,))
    empties = [d for d in sc["declarations"] if d["malformed"] == "empty_span"]
    if not sc["lexical_ok"] or empties:
        bad = [d for d in sc["declarations"]
               if d["malformed"] in ("tag_syntax", "nesting", "stray_closer", "unclosed",
                                     "empty_span")]
        lines = "\n".join("  - byte %d: %s %s" % (d["at"], d["malformed"], d.get("raw") or "")
                          for d in bad)
        raise SystemExit("REFUSED: the sidecar form cannot carry a lexically MALFORMED declaration; "
                         "verify the inline document instead. Offending declarations:\n" + lines)
    spans = [{"start": d["start"], "end": d["end"], "receipt": d["receipt"], "kind": d["kind"]}
             for d in sc["declarations"]]
    spans.sort(key=lambda s: s["start"])
    side = {
        "spec": SPEC,
        "commit": commit,
        "document": {"name": name, "sha256": _sha256(sc["canonical"])},
        "text": sc["canonical"].decode("utf-8"),
        "spans": spans,
        "manifest": manifest.to_dict() if manifest is not None else {"spec": MANIFEST_SPEC,
                                                                    "receipts": {}},
    }
    back = render(side)
    if back != raw:
        raise SystemExit("REFUSED: canonical round trip does not reproduce the document bytes "
                         "(%d vs %d bytes) — no sidecar written" % (len(back), len(raw)))
    return side


_ATTR_VALUE = re.compile(r'[^"<>\r\n]*')


def _refuse(msg: str):
    raise SystemExit("REFUSED: " + msg)


def load_sidecar(obj: dict) -> dict:
    """Validate a sidecar strictly. A shape this verifier does not know is refused, never guessed
    and never crashed on: every check below raises the REFUSED SystemExit and nothing else."""
    if not isinstance(obj, dict) or obj.get("spec") != SPEC:
        _refuse("unknown sidecar spec %r (this verifier knows %s)"
                % (obj.get("spec") if isinstance(obj, dict) else None, SPEC))
    required = ("spec", "commit", "document", "text", "spans", "manifest")
    missing = [k for k in required if k not in obj]
    extra = sorted(set(obj) - set(required))
    if missing or extra:
        _refuse("sidecar keys — missing %s, unknown %s" % (missing, extra))
    commit = obj["commit"]
    if commit is not None and not (isinstance(commit, str)
                                   and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)):
        _refuse("sidecar commit must be a full lowercase hex object id or null")
    if not isinstance(obj["text"], str):
        _refuse("sidecar text must be a string")
    try:
        text = obj["text"].encode("utf-8")
    except UnicodeEncodeError as e:
        _refuse("sidecar text is not encodable as UTF-8 at character %d" % e.start)
    doc = obj["document"]
    if not (isinstance(doc, dict) and isinstance(doc.get("name"), str)
            and isinstance(doc.get("sha256"), str)):
        _refuse("sidecar document must carry a string name and a string sha256")
    if _sha256(text) != doc["sha256"]:
        _refuse("sidecar text does not hash to document.sha256")
    if not isinstance(obj["spans"], list):
        _refuse("sidecar spans must be a list")
    last_end = 0
    for i, s in enumerate(obj["spans"]):
        if not isinstance(s, dict) or set(s) != {"start", "end", "receipt", "kind"}:
            _refuse("span %d is not an object with exactly start/end/receipt/kind" % i)
        a, b = s["start"], s["end"]
        if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int)
                or not isinstance(b, int) or not (0 <= a < b <= len(text))):
            _refuse("span %d offsets %r are not a non-empty range in the text" % (i, (a, b)))
        if a < last_end:
            _refuse("spans are not ordered and non-overlapping at span %d" % i)
        for off in (a, b):
            if off < len(text) and (text[off] & 0xC0) == 0x80:
                _refuse("span %d offset %d is not on a UTF-8 character boundary" % (i, off))
        for key in ("receipt", "kind"):
            v = s[key]
            if not isinstance(v, str) or not _ATTR_VALUE.fullmatch(v):
                _refuse("span %d %s %r cannot be carried by the inline tag grammar" % (i, key, v))
        last_end = b
    man = obj["manifest"]
    if not (isinstance(man, dict) and man.get("spec") == MANIFEST_SPEC
            and isinstance(man.get("receipts"), dict)
            and all(isinstance(k, str) and isinstance(v, dict) for k, v in man["receipts"].items())
            and isinstance(man.get("authored_sha256", []), list)
            and all(isinstance(x, str) for x in man.get("authored_sha256", []))):
        _refuse("sidecar manifest is not a %s object" % MANIFEST_SPEC)
    return obj


# =============================================================================================
# 3. receipts — the manifest, the tree, the three forms
# =============================================================================================

class Manifest:
    """The turn manifest the HARNESS mints (``sworn/manifest/0.1``). Never the agent.

    This class cannot enforce who calls it; what it can do is record, per receipt, a
    ``kind_of_source`` from a closed vocabulary and refuse author-side kinds at verification.
    The manifest is only as trustworthy as the harness that wrote it, and every receipt says so.
    """

    def __init__(self, harness: str = "", turn: str = "", minted_at: Optional[str] = None,
                 receipts: Optional[Dict[str, dict]] = None,
                 authored_sha256: Optional[List[str]] = None):
        self.harness = harness
        self.turn = turn
        self.minted_at = minted_at or _now()
        self.receipts: Dict[str, dict] = dict(receipts or {})
        # sha256 of every byte-object the agent produced this turn, as the harness saw them: files
        # written, messages emitted, stdin fed to tools. Invariant 2 becomes set membership.
        self.authored_sha256: List[str] = [x.lower() for x in (authored_sha256 or [])]

    def record_authored(self, data: bytes) -> str:
        h = _sha256(data)
        if h not in self.authored_sha256:
            self.authored_sha256.append(h)
        return h

    def add(self, rid: str, data: Optional[bytes], kind_of_source: str, complete: bool = False,
            captured_at: Optional[str] = None, sha256: Optional[str] = None) -> dict:
        if not re.fullmatch(r"r[1-9][0-9]*", rid):
            raise ValueError("receipt id must match r[1-9][0-9]*: %r" % rid)
        if data is None and not sha256:
            raise ValueError("a receipt needs bytes or at least a sha256")
        entry = {
            "id": rid,
            "sha256": sha256 or _sha256(data),
            "kind_of_source": kind_of_source,
            "captured_at": captured_at or _now(),
            "complete": bool(complete),
        }
        if data is not None:
            entry["bytes"] = _b64(data)
        self.receipts[rid] = entry
        return entry

    def core(self) -> dict:
        return {"spec": MANIFEST_SPEC, "harness": self.harness, "turn": self.turn,
                "minted_at": self.minted_at, "authored_sha256": sorted(self.authored_sha256),
                "receipts": self.receipts}

    def digest(self) -> str:
        return _sha256(_jcs(self.core()).encode("utf-8"))

    def to_dict(self) -> dict:
        d = self.core()
        d["digest"] = self.digest()
        return d

    def write(self, path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n",
                     encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        if not isinstance(d, dict) or d.get("spec") != MANIFEST_SPEC:
            raise SystemExit("REFUSED: unknown manifest spec %r (this verifier knows %s)"
                             % (d.get("spec") if isinstance(d, dict) else None, MANIFEST_SPEC))
        m = cls(d.get("harness", ""), d.get("turn", ""), d.get("minted_at"),
                d.get("receipts") or {}, d.get("authored_sha256") or [])
        # the digest the harness wrote, if any. It is checked at resolution time: a manifest that
        # does not re-derive makes every rN UNRESOLVED — the verifier failing to see, never the
        # author lying.
        m.declared_digest = d.get("digest")
        return m

    @classmethod
    def load(cls, path) -> "Manifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    declared_digest: Optional[str] = None

    def intact(self) -> bool:
        return self.declared_digest is None or self.declared_digest == self.digest()


class MemoryTree:
    """A tree snapshot for tests and for embedded use: ``{path: bytes}`` at a nominal commit."""

    def __init__(self, files: Dict[str, bytes], commit: Optional[str] = None):
        self.files = dict(files)
        self.commit = commit

    def blob(self, path: str) -> Tuple[Optional[bytes], str]:
        if self.commit is None:
            return None, "no_commit"
        if path not in self.files:
            return None, "path_absent"
        return self.files[path], "ok"

    def find_sha256(self, digest: str) -> Tuple[Optional[bytes], str]:
        if self.commit is None:
            return None, "no_commit"
        for b in self.files.values():
            if _sha256(b) == digest:
                return b, "ok"
        return None, "prereg_not_in_tree"


_PREREG_INDEX: Dict[Tuple[str, str], Dict[str, str]] = {}


class GitTree:
    """The repository tree AT ONE COMMIT, read with git plumbing. Never the working tree.

    A verdict must be a function of bytes, not of somebody's checkout; resolving against a
    working tree would make it one. Every failure is a reason string, never an exception:
    UNRESOLVED is the verifier saying it could not see, and it is never an accusation.
    """

    def __init__(self, repo, commit: Optional[str]):
        self.repo = Path(repo) if repo is not None else None
        self.commit = commit

    def _git(self, *args: str, stdin: Optional[bytes] = None) -> Tuple[int, bytes, bytes]:
        try:
            r = subprocess.run(["git", "-C", str(self.repo), *args], input=stdin,
                               capture_output=True, check=False)
        except (OSError, ValueError):
            return 127, b"", b"git unavailable"
        return r.returncode, r.stdout, r.stderr

    def _ready(self) -> Optional[str]:
        if self.repo is None:
            return "no_repository"
        if self.commit is None:
            return "no_commit"
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.commit):
            return "commit_absent"
        # `cat-file -t` names the object's type without peeling: a tag or tree the document named
        # is not the commit it named, and the resolver does not guess which commit it meant.
        rc, out, err = self._git("cat-file", "-t", self.commit)
        if rc == 127:
            return "git_unavailable"
        if rc != 0 or out.strip() != b"commit":
            return "commit_absent"
        return None

    def blob(self, path: str) -> Tuple[Optional[bytes], str]:
        why = self._ready()
        if why:
            return None, why
        rc, out, _ = self._git("ls-tree", "-z", "-l", "--full-tree", self.commit, "--", path)
        if rc != 0 or not out.strip(b"\0"):
            return None, "path_absent"
        meta = out.split(b"\0")[0].split(b"\t")[0].split()
        if len(meta) < 4 or meta[1] != b"blob" or meta[0] == b"120000":
            return None, "not_a_blob"
        if meta[3].isdigit() and int(meta[3]) > MAX_RECEIPT_BYTES:
            return None, "receipt_too_large"
        rc, data, _ = self._git("cat-file", "blob", meta[2].decode("ascii"))
        if rc != 0:
            return None, "git_unavailable"
        return data, "ok"

    def _index(self) -> Optional[Dict[str, str]]:
        key = (str(self.repo.resolve()), self.commit)
        if key in _PREREG_INDEX:
            return _PREREG_INDEX[key]
        rc, out, _ = self._git("ls-tree", "-r", "-z", "--full-tree", self.commit)
        if rc != 0:
            return None
        shas = []
        for ent in out.split(b"\0"):
            if not ent:
                continue
            meta = ent.split(b"\t")[0].split()
            if len(meta) >= 3 and meta[1] == b"blob" and meta[0] != b"120000":
                shas.append(meta[2].decode("ascii"))
        idx: Dict[str, str] = {}
        rc, out, _ = self._git("cat-file", "--batch", stdin=("\n".join(shas) + "\n").encode("ascii"))
        if rc != 0:
            return None
        pos = 0
        while pos < len(out):
            nl = out.find(b"\n", pos)
            if nl < 0:
                break
            header = out[pos:nl].split()
            pos = nl + 1
            if len(header) < 3:
                continue
            size = int(header[2])
            body = out[pos:pos + size]
            pos += size + 1
            if size <= MAX_RECEIPT_BYTES:
                idx.setdefault(_sha256(body), header[0].decode("ascii"))
        _PREREG_INDEX[key] = idx
        return idx

    def find_sha256(self, digest: str) -> Tuple[Optional[bytes], str]:
        why = self._ready()
        if why:
            return None, why
        idx = self._index()
        if idx is None:
            return None, "git_unavailable"
        sha = idx.get(digest.lower())
        if sha is None:
            return None, "prereg_not_in_tree"
        rc, data, _ = self._git("cat-file", "blob", sha)
        if rc != 0:
            return None, "git_unavailable"
        return data, "ok"


_RN = re.compile(r"r[1-9][0-9]*")
_ANCHOR = re.compile(r"L([1-9][0-9]*)(?:-L([1-9][0-9]*))?")
_PATH_SEG_BAD = re.compile(r"[\\\s\x00-\x1f\x7f]")


def _parse_receipt(ref: str) -> Tuple[Optional[dict], Optional[str]]:
    """The receipt grammar, decidable from bytes. Returns (parsed, malformed_reason)."""
    if ref is None:
        return None, "receipt_form"
    if _RN.fullmatch(ref):
        return {"form": "rn", "id": ref, "fragment": None, "partial": False}, None
    if ref.startswith("path:"):
        body = ref[5:]
        form = "path"
        target: Any
        if "#" in body:
            target, frag = body.split("#", 1)
        else:
            target, frag = body, None
        if not target or target.startswith("/") or _PATH_SEG_BAD.search(target):
            return None, "receipt_form"
        if any(seg in ("", ".", "..") for seg in target.split("/")):
            return None, "receipt_form"
    elif ref.startswith("prereg:"):
        body = ref[7:]
        form = "prereg"
        if "#" in body:
            target, frag = body.split("#", 1)
        else:
            target, frag = body, None
        if not re.fullmatch(r"[0-9a-fA-F]{64}", target):
            return None, "receipt_form"
        target = target.lower()
    else:
        return None, "receipt_form"
    fragment: Optional[dict] = None
    if frag is not None:
        if frag == "":
            return None, "receipt_form"
        if frag.startswith("/"):
            if re.search(r"~(?![01])", frag):
                return None, "receipt_form"          # a `~` not followed by 0 or 1 has no meaning
            toks = [t.replace("~1", "/").replace("~0", "~") for t in frag.split("/")[1:]]
            fragment = {"type": "pointer", "tokens": toks, "raw": frag}
        else:
            m = _ANCHOR.fullmatch(frag)
            if not m:
                return None, "receipt_form"
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            if b < a:
                return None, "receipt_form"
            fragment = {"type": "lines", "first": a, "last": b, "raw": frag}
    return {"form": form, "target": target, "fragment": fragment,
            "partial": fragment is not None}, None


def _line_slice(data: bytes, first: int, last: int) -> Optional[bytes]:
    """1-based, inclusive, LF-split, CR retained; interior newlines kept, the last selected line's
    terminating LF excluded. None when a line is past EOF. Only the 0x0A byte ever splits."""
    positions = [i for i, b in enumerate(data) if b == 0x0A]
    n_lines = len(positions) + (1 if data and not data.endswith(b"\n") else 0)
    if first > n_lines or last > n_lines:
        return None
    begin = 0 if first == 1 else positions[first - 2] + 1
    end = positions[last - 1] if last - 1 < len(positions) else len(data)
    return data[begin:end]


class _Obj(dict):
    """A JSON object that remembers which of its keys were duplicated in the source."""
    dups: frozenset = frozenset()


def _json_strict(data: bytes):
    """JSON with every number as a Decimal of its own digits (never a float), NaN/Infinity kept
    as Decimal so one bad leaf cannot hide the rest of the file, and duplicate keys remembered."""
    def pairs(items):
        d = _Obj()
        seen = set()
        dups = set()
        for k, v in items:
            if k in seen:
                dups.add(k)
            seen.add(k)
            d[k] = v
        d.dups = frozenset(dups)
        return d

    text = data.decode("utf-8")
    if text.startswith("\ufeff"):
        raise ValueError("BOM-prefixed JSON")
    return json.loads(text, parse_float=Decimal, parse_int=Decimal, parse_constant=Decimal,
                      object_pairs_hook=pairs)


def _walk_pointer(obj: Any, tokens: List[str]) -> Tuple[Any, str]:
    """Returns (value, 'ok') or (None, reason). A duplicated key ON the path is ambiguous."""
    for t in tokens:
        if isinstance(obj, dict):
            if t not in obj:
                return None, "pointer_unresolvable"
            if t in getattr(obj, "dups", ()):
                return None, "pointer_ambiguous"
            obj = obj[t]
        elif isinstance(obj, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", t) or int(t) >= len(obj):
                return None, "pointer_unresolvable"
            obj = obj[int(t)]
        else:
            return None, "pointer_unresolvable"
    return obj, "ok"


class _Resolved(dict):
    """``status`` ok|unresolved|malformed, ``reason``, ``bytes`` (the whole object), ``sha256``,
    ``complete``, ``leaf`` (when a pointer addressed one), ``slice`` (when a line anchor did)."""


def _resolve(parsed: dict, kind: str, manifest: Optional[Manifest], tree) -> _Resolved:
    data: Optional[bytes] = None
    sha: Optional[str] = None
    complete = False
    if parsed["form"] == "rn":
        if manifest is None:
            return _Resolved(status="unresolved", reason="manifest_absent")
        if not manifest.intact():
            return _Resolved(status="unresolved", reason="manifest_integrity")
        entry = manifest.receipts.get(parsed["id"])
        if entry is None:
            return _Resolved(status="unresolved", reason="manifest_id_missing")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            return _Resolved(status="unresolved", reason="manifest_integrity")
        kos = entry.get("kind_of_source")
        if kos in SOURCE_KINDS_AUTHOR or sha in manifest.authored_sha256:
            return _Resolved(status="malformed", reason="receipt_author_minted")
        if kos not in SOURCE_KINDS_EXTERNAL:
            return _Resolved(status="malformed", reason="kind_of_source_unknown")
        complete = entry.get("complete") if isinstance(entry.get("complete"), bool) else None
        if "bytes" in entry:
            try:
                data = base64.b64decode(entry["bytes"], validate=True)
            except (ValueError, TypeError):
                return _Resolved(status="unresolved", reason="manifest_integrity")
            if _sha256(data) != sha:
                return _Resolved(status="unresolved", reason="manifest_integrity")
        elif kind != "hash":
            return _Resolved(status="unresolved", reason="manifest_bytes_absent")
    else:
        if tree is None:
            return _Resolved(status="unresolved", reason="no_repository")
        if parsed["form"] == "path":
            data, why = tree.blob(parsed["target"])
        else:
            data, why = tree.find_sha256(parsed["target"])
        if data is None:
            return _Resolved(status="unresolved", reason=why)
        sha = _sha256(data)
        complete = True
    res = _Resolved(status="ok", reason=None, bytes=data, sha256=sha, complete=complete,
                    leaf=None, has_leaf=False, slice=None)
    frag = parsed.get("fragment")
    if frag is not None and data is not None:
        if frag["type"] == "lines":
            sl = _line_slice(data, frag["first"], frag["last"])
            if sl is None:
                return _Resolved(status="malformed", reason="anchor_out_of_range")
            res["slice"] = sl
        else:
            try:
                obj = _json_strict(data)
            except (ValueError, UnicodeDecodeError, RecursionError):
                return _Resolved(status="malformed", reason="receipt_not_json")
            leaf, why = _walk_pointer(obj, frag["tokens"])
            if why != "ok":
                return _Resolved(status="malformed", reason=why)
            res["leaf"] = leaf
            res["has_leaf"] = True
    return res


# =============================================================================================
# 4. the kinds — what a span states, and whether the receipt disposes of it
# =============================================================================================

# The span is cut into MAXIMAL tokens; a token that carries any digit must be, whole, one number.
# There is no guard, no whitelist and no scrub, so a fragment of a number can never be extracted:
# `23,247.` is 23247 and `1e-5` is a MALFORMED span, never the number 5. Trailing sentence
# punctuation is the one thing stripped, because it is the one thing OATH's `_NUM` refused and
# then could not see past (`precision of 0.55.` certified with zero tokens examined).
_TOKEN = re.compile(r"[\w.,+\-−%/±:]+")
_DIGIT = re.compile(r"\d")
_GRAM = re.compile(r"[-+−]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)%?")
_HEXRUN = re.compile(r"(?<![A-Za-z0-9_])[0-9A-Fa-f]+(?![A-Za-z0-9_])")
_DIGEST_LENGTHS = {32, 40, 96, 128}


def _number_token(text: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """(reason, token, digit_bearing_tokens): reason None iff exactly one token is one number."""
    digit_bearing = [t for t in _TOKEN.findall(text) if _DIGIT.search(t)]
    if len(digit_bearing) != 1:
        return "number_count", None, digit_bearing
    tok = digit_bearing[0].rstrip(".,:")
    if not _GRAM.fullmatch(tok):
        return "number_grammar", None, digit_bearing
    return None, tok, digit_bearing


def _needle_in(inner: bytes) -> Tuple[Optional[bytes], str]:
    """The ONE inline backtick code span inside the sworn text, bytes verbatim."""
    spans: List[bytes] = []
    p = 0
    n = len(inner)
    while p < n:
        m = _TICKS.search(inner, p)
        if not m:
            break
        run = m.end() - m.start()
        nl = inner.find(b"\n", m.end())
        line_end = n if nl < 0 else nl
        q = m.end()
        closed = None
        while q < line_end:
            m2 = _TICKS.search(inner, q, line_end)
            if not m2:
                break
            if m2.end() - m2.start() == run:
                closed = m2
                break
            q = m2.end()
        if closed is None:
            p = m.end()
            continue
        spans.append(inner[m.end():closed.start()])
        p = closed.end()
    if len(spans) != 1:
        return None, "needle_count"
    if not spans[0].strip(b" \t\r\n"):
        return None, "needle_empty"          # a blank needle would HELD against almost anything
    return spans[0], "ok"


def _printed_decimal(tok: str) -> Tuple[Decimal, int]:
    """The printed token as a Decimal and the count of fractional digits it prints."""
    t = tok.replace(",", "").replace("−", "-").rstrip("%")
    d = Decimal(t)
    frac = len(t.split(".", 1)[1]) if "." in t else 0
    return d, frac


def _canon(x: Decimal, frac: int) -> str:
    """One canonical decimal string for both sides: quantized half-even, signed zero folded."""
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec = max(28, x.adjusted() + frac + 2)
        q = x.quantize(Decimal(1).scaleb(-frac), rounding=ROUND_HALF_EVEN)
    if q.is_zero():
        q = q.copy_abs()
    return format(q, "f")


def _check_numeric(inner_text: str, res: _Resolved, parsed: dict) -> Tuple[str, Optional[str], dict]:
    why, tok, seen = _number_token(inner_text)
    if why:
        return "MALFORMED", why, {"digit_bearing_tokens": seen}
    printed, frac = _printed_decimal(tok)
    if res["has_leaf"]:
        leaf = res["leaf"]
    else:
        source = res["slice"] if res["slice"] is not None else res["bytes"]
        try:
            leaf = _json_strict(source)
        except (ValueError, UnicodeDecodeError, RecursionError):
            return "MALFORMED", "receipt_not_json", {}
    if isinstance(leaf, (dict, list)) or leaf is None:
        return "MALFORMED", "leaf_not_scalar", {"leaf_type": type(leaf).__name__}
    if not isinstance(leaf, Decimal) or isinstance(leaf, bool):
        return "MALFORMED", "leaf_not_numeric", {"leaf": str(leaf)[:80]}
    if not leaf.is_finite() or leaf.adjusted() > 320:
        return "MALFORMED", "leaf_not_numeric", {"leaf": str(leaf)[:80]}
    try:
        lhs, rhs = _canon(leaf, frac), _canon(printed, frac)
    except (InvalidOperation, ValueError):
        return "MALFORMED", "leaf_not_numeric", {"leaf": str(leaf)[:80]}
    detail = {"printed_token": tok, "printed": rhs, "receipt": str(leaf), "receipt_rounded": lhs,
              "fractional_digits": frac, "rounding": ROUNDING}
    if lhs == rhs:
        return "HELD", None, detail
    return "FAILED", "value_mismatch", detail


def _check_quote(inner: bytes, res: _Resolved) -> Tuple[str, Optional[str], dict]:
    needle, why = _needle_in(inner)
    if needle is None:
        return "MALFORMED", why, {}
    if res["has_leaf"]:
        if not isinstance(res["leaf"], str):
            return "MALFORMED", "leaf_not_string", {}
        hay = res["leaf"].encode("utf-8")
    else:
        hay = res["slice"] if res["slice"] is not None else res["bytes"]
    detail = {"needle_bytes": len(needle), "haystack_bytes": len(hay)}
    if needle in hay:
        return "HELD", None, detail
    return "FAILED", "needle_missing", detail


def _check_absent(inner: bytes, res: _Resolved) -> Tuple[str, Optional[str], dict]:
    needle, why = _needle_in(inner)
    if needle is None:
        return "MALFORMED", why, {}
    hay = res["bytes"]
    detail = {"needle_bytes": len(needle), "haystack_bytes": len(hay), "complete": True}
    if needle in hay:
        return "FAILED", "needle_present", detail
    return "HELD", None, detail


def _check_hash(inner_text: str, res: _Resolved) -> Tuple[str, Optional[str], dict]:
    runs = [m.group(0) for m in _HEXRUN.finditer(inner_text)]
    sixty_four = [r for r in runs if len(r) == 64]
    others = [r for r in runs if len(r) in _DIGEST_LENGTHS]
    if len(sixty_four) != 1 or others:
        return "MALFORMED", "digest_form", {"hex_runs": [len(r) for r in runs]}
    stated = sixty_four[0].lower()
    detail = {"stated": stated, "receipt_sha256": res["sha256"]}
    if stated == res["sha256"]:
        return "HELD", None, detail
    return "FAILED", "digest_mismatch", detail


def _adjudicate(d: Declaration, manifest: Optional[Manifest], tree) -> dict:
    """One declaration → one verdict. A pure function of (bytes, manifest, tree)."""
    verdict: Dict[str, Any] = {"at": d["at"], "start": d.get("start"), "end": d.get("end"),
                               "receipt": d.get("receipt"), "kind": d.get("kind"),
                               "verdict": None, "reason": None, "detail": {}}

    def out(v, reason=None, detail=None, res=None):
        verdict["verdict"] = v
        verdict["reason"] = reason
        verdict["detail"] = detail or {}
        if res is not None and res.get("sha256"):
            verdict["resolved_sha256"] = res["sha256"]
        return verdict

    if d["malformed"]:
        return out("MALFORMED", d["malformed"], {"raw": d.get("raw")} if d.get("raw") else {})
    inner: bytes = d["inner"]
    if not inner.strip(b" \t\r\n\f\v"):
        return out("MALFORMED", "empty_span")
    if len(inner) > SPAN_CAP_BYTES:
        return out("MALFORMED", "length_cap", {"bytes": len(inner), "cap": SPAN_CAP_BYTES})
    kind = d["kind"]
    if kind in RESERVED_KINDS:
        return out("MALFORMED", "kind_reserved", {"kind": kind})
    if kind not in KINDS:
        return out("MALFORMED", "kind_unknown", {"kind": kind})
    parsed, why = _parse_receipt(d["receipt"])
    if parsed is None:
        return out("MALFORMED", why, {"receipt": d["receipt"]})
    if kind == "absent" and parsed["partial"]:
        return out("MALFORMED", "absent_over_partial", {"receipt": d["receipt"]})
    if kind == "hash" and parsed["partial"]:
        return out("MALFORMED", "hash_over_partial", {"receipt": d["receipt"]})
    inner_text = inner.decode("utf-8")
    # bytes-only form checks happen BEFORE any receipt is opened, so a MALFORMED never depends
    # on evidence the verifier might not have.
    if kind == "numeric":
        nwhy, _tok, seen = _number_token(inner_text)
        if nwhy:
            return out("MALFORMED", nwhy, {"digit_bearing_tokens": seen})
    elif kind in ("quote", "absent"):
        needle, nwhy = _needle_in(inner)
        if needle is None:
            return out("MALFORMED", nwhy)
    else:
        runs = [m.group(0) for m in _HEXRUN.finditer(inner_text)]
        if len([r for r in runs if len(r) == 64]) != 1 or any(len(r) in _DIGEST_LENGTHS for r in runs):
            return out("MALFORMED", "digest_form", {"hex_runs": [len(r) for r in runs]})

    res = _resolve(parsed, kind, manifest, tree)
    if res["status"] == "unresolved":
        return out("UNRESOLVED", res["reason"])
    if res["status"] == "malformed":
        return out("MALFORMED", res["reason"], res=res)
    if kind == "absent":
        if res["complete"] is None:
            return out("UNRESOLVED", "manifest_no_completeness")
        if res["complete"] is not True:
            return out("MALFORMED", "absent_over_partial", {"complete": False}, res=res)
    if kind == "numeric":
        v, r, det = _check_numeric(inner_text, res, parsed)
    elif kind == "quote":
        v, r, det = _check_quote(inner, res)
    elif kind == "absent":
        v, r, det = _check_absent(inner, res)
    else:
        v, r, det = _check_hash(inner_text, res)
    return out(v, r, det, res=res)


# =============================================================================================
# 5. the document — verdict, coverage, receipt
# =============================================================================================

# diffgate.py's splitter, byte for byte (tests assert the literal against diffgate's source): the
# one the spec names as its largest false-flag source, reused rather than re-invented so the
# coverage denominator drifts with diffgate and not on its own.
_SENTENCE_SPLIT = re.compile(rb"(?<=[.!?])\s+|\n+")


def _coverage(canonical: Optional[bytes], spans: List[dict], fenced: List[Tuple[int, int]],
              sworn_total: int) -> dict:
    """Advisory, always. The denominator is counted by an instrument with a documented ceiling."""
    cov: Dict[str, Any] = {"estimate": None, "unsworn_claims_estimate": None,
                           "unsworn_claims": [], "advisory": True, "ceiling_note": _COVERAGE_CEILING,
                           "splitter": "diffgate:(?<=[.!?])\\s+|\\n+", "claimdetect_version": None}
    if canonical is None:
        cov["note"] = "no canonical text: the document is lexically MALFORMED"
        return cov
    try:
        from styxx.claimdetect import STRUCT1_VERSION, detect
    except Exception:                                        # pragma: no cover - observer optional
        cov["note"] = "styxx.claimdetect unavailable; denominator not counted"
        return cov
    cov["claimdetect_version"] = STRUCT1_VERSION
    # mask sworn regions and fenced regions with spaces so offsets stay canonical
    buf = bytearray(canonical)
    for s in spans:
        if s.get("start") is not None:
            for i in range(s["start"], s["end"]):
                buf[i] = 0x20
    # fenced regions are recorded in INLINE coordinates; recompute them on the canonical text
    for a, b in _fenced_regions(bytes(buf))[0]:
        for i in range(a, b):
            buf[i] = 0x20
    narrative = bytes(buf)
    claims = []
    pos = 0
    pieces = []
    for m in _SENTENCE_SPLIT.finditer(narrative):
        pieces.append((pos, m.start()))
        pos = m.end()
    pieces.append((pos, len(narrative)))
    for a, b in pieces:
        seg = narrative[a:b]
        if not seg.strip():
            continue
        lead = len(seg) - len(seg.lstrip())
        text = seg.strip().decode("utf-8", errors="replace")
        try:
            is_claim = bool(detect(text).is_claim)
        except Exception:                               # the observer failing is not a verdict
            cov["note"] = "styxx.claimdetect raised; denominator not counted"
            cov["unsworn_claims"] = []
            return cov
        if is_claim:
            claims.append({"start": a + lead, "end": a + lead + len(seg.strip()), "text": text[:200]})
    n_claims = len(claims)
    denom = sworn_total + n_claims
    cov["unsworn_claims_estimate"] = n_claims
    cov["unsworn_claims"] = claims
    cov["estimate"] = (round(sworn_total / denom, 4) if denom else None)
    return cov


def verify(raw: Optional[bytes] = None, sidecar: Optional[dict] = None, *, name: str = "",
           manifest: Optional[Manifest] = None, tree=None, commit: Optional[str] = None) -> dict:
    """Verify an inline document (``raw``) or a sidecar. Returns the verdict receipt core.

    Sidecar path: render → re-scan → the re-scan must reproduce the sidecar's spans exactly, or
    the sidecar is refused. One lexer, one truth — the capsule's "must be reproducible at the
    live verifier" rule applied to the canonical form.
    """
    if sidecar is not None:
        sidecar = load_sidecar(sidecar)
        raw = render(sidecar)
        name = name or sidecar["document"]["name"]
        if commit is not None and sidecar["commit"] not in (None, commit):
            raise SystemExit("REFUSED: the sidecar names commit %s and --commit says %s"
                             % (sidecar["commit"], commit))
        commit = sidecar["commit"]
        emb = sidecar.get("manifest") or {}
        if emb.get("receipts"):
            embedded = Manifest.from_dict(emb)
            if manifest is not None and manifest.digest() != embedded.digest():
                raise SystemExit("REFUSED: the supplied manifest disagrees with the embedded one")
            manifest = manifest or embedded
    if raw is None:
        raise SystemExit("REFUSED: nothing to verify")
    if tree is not None and getattr(tree, "commit", None) is None and commit is not None:
        tree.commit = commit
    sc = scan(raw)
    if sidecar is not None:
        # One lexer, one truth: the sidecar must render into a document the lexer reads as
        # exactly those spans. A text with no canonical form, or a span table the re-scan does
        # not reproduce, is not a sidecar and is refused rather than verified on a guess.
        if sc["document_malformed"]:
            raise SystemExit("REFUSED: the sidecar text has no canonical form (%s)"
                             % sc["document_malformed"]["reason"])
        if not sc["lexical_ok"] or any(d["malformed"] == "empty_span" for d in sc["declarations"]):
            raise SystemExit("REFUSED: the rendered sidecar carries a declaration the sidecar form "
                             "cannot represent")
        seen = [(d["start"], d["end"], d["receipt"], d["kind"]) for d in sc["declarations"]]
        want = [(s["start"], s["end"], s["receipt"], s["kind"]) for s in sidecar["spans"]]
        if sorted(seen) != want:
            raise SystemExit("REFUSED: re-scanning the rendered sidecar does not reproduce its spans")
    verdicts = [_adjudicate(d, manifest, tree) for d in sc["declarations"]]
    verdicts.sort(key=lambda v: v["at"])
    counts = {v: 0 for v in VERDICTS}
    for v in verdicts:
        counts[v["verdict"]] += 1
    sworn_total = sum(counts.values())
    doc_malformed = sc["document_malformed"]
    if doc_malformed:
        document_verdict = "SWORN-FAILED"
    elif sworn_total == 0:
        document_verdict = "UNSWORN"
    elif counts["FAILED"] == 0 and counts["MALFORMED"] == 0:
        document_verdict = "SWORN-HELD"
    else:
        document_verdict = "SWORN-FAILED"
    coverage = _coverage(sc["canonical"], verdicts, sc["fenced"], sworn_total)
    from styxx._version import __version__
    core = {
        "schema": RECEIPT_SCHEMA,
        "format": SPEC,
        "document": {"name": name, "inline_sha256": _sha256(raw),
                     "canonical_sha256": _sha256(sc["canonical"]) if sc["canonical"] is not None else None},
        "commit": commit,
        "manifest_digest": manifest.digest() if manifest is not None else None,
        "spans": verdicts,
        "counts": counts,
        "sworn_total": sworn_total,
        "unresolved": counts["UNRESOLVED"],
        "document_verdict": document_verdict,
        "document_malformed": doc_malformed,
        "coverage": coverage,
        "verifier": {"styxx_version": __version__,
                     "sworn_sha256": _sha256(Path(__file__).read_bytes()),
                     "rounding": ROUNDING, "decisions": DECISIONS},
        "certifies": _CERTIFIES,
    }
    return core


def issue_receipt(core: dict, timestamp: Optional[str] = None) -> dict:
    """Content-address a verdict core: ``digest`` over the JCS form of everything but itself and
    the timestamp. Re-derivable by anyone with the document, manifest and tree."""
    rec = dict(core)
    rec.pop("digest", None)
    rec.pop("timestamp", None)
    rec["digest"] = _sha256(_jcs(rec).encode("utf-8"))
    rec["timestamp"] = timestamp or _now()
    return rec


def verify_receipt(receipt: dict, raw: Optional[bytes] = None, sidecar: Optional[dict] = None, *,
                   manifest: Optional[Manifest] = None, tree=None) -> dict:
    """Re-derive a receipt against the presented document. Trust neither the author (bytes are
    hashed) nor the verifier that issued it (the verdict is re-run)."""
    core = {k: v for k, v in receipt.items() if k not in ("digest", "timestamp")}
    digest_ok = receipt.get("digest") == _sha256(_jcs(core).encode("utf-8"))
    fresh = verify(raw, sidecar, name=core.get("document", {}).get("name", ""),
                   manifest=manifest, tree=tree, commit=core.get("commit"))
    # the verifier block names the build; a different build is reported, not hidden
    same_build = fresh["verifier"]["sworn_sha256"] == core.get("verifier", {}).get("sworn_sha256")
    cmp_fresh = {k: v for k, v in fresh.items() if k != "verifier"}
    cmp_core = {k: v for k, v in core.items() if k != "verifier"}
    reproduces = _jcs(cmp_fresh) == _jcs(cmp_core)
    return {"status": "VERIFIED" if (digest_ok and reproduces) else "FAILED",
            "digest_match": digest_ok, "verdict_reproduces": reproduces,
            "same_verifier_build": same_build,
            "note": "verify-by-re-derivation: the document is hashed and the verdict is re-run"}


# =============================================================================================
# 6. CLI — reports, never gates
# =============================================================================================

def _headline(core: dict) -> str:
    c = core["counts"]
    cov = core["coverage"]
    est = "n/a" if cov["estimate"] is None else "%.2f" % cov["estimate"]
    unsworn = cov["unsworn_claims_estimate"]
    line = ("%s  held=%d failed=%d unresolved=%d malformed=%d  "
            "coverage≈%s (advisory)  unsworn-claims≈%s"
            % (core["document_verdict"], c["HELD"], c["FAILED"], c["UNRESOLVED"], c["MALFORMED"],
               est, "n/a" if unsworn is None else unsworn))
    if core["document_malformed"]:
        line += "  document-MALFORMED: %s" % core["document_malformed"]["reason"]
    return line


def _load_tree(repo: Optional[str], commit: Optional[str]):
    if repo is None:
        return None
    return GitTree(repo, commit)


def _read_input(path: str) -> Tuple[Optional[bytes], Optional[dict]]:
    p = Path(path)
    raw = p.read_bytes()
    if p.suffix == ".json":
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            obj = None
        if isinstance(obj, dict) and obj.get("spec") == SPEC:
            return None, obj
    return raw, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.sworn",
                                 description="sworn output v0.1 — the author declares, the receipt "
                                             "disposes. Reports; never gates.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("canon", help="inline document -> sidecar (refuses if it cannot round-trip)")
    c.add_argument("doc")
    c.add_argument("--commit", default=None, help="40-hex commit the path:/prereg: receipts resolve at")
    c.add_argument("--manifest", default=None, help="harness-minted manifest to embed")
    c.add_argument("--out", default=None)

    r = sub.add_parser("render", help="sidecar -> inline document, byte-exact")
    r.add_argument("sidecar")
    r.add_argument("--out", default=None)

    v = sub.add_parser("verify", help="verify an inline document or a sidecar; write the receipt")
    v.add_argument("target")
    v.add_argument("--repo", default=None, help="repository whose tree resolves path:/prereg: receipts")
    v.add_argument("--commit", default=None)
    v.add_argument("--manifest", default=None)
    v.add_argument("--out", default=None, help="write the verdict receipt JSON here")

    k = sub.add_parser("check", help="re-derive a verdict receipt against the document it names")
    k.add_argument("receipt")
    k.add_argument("target")
    k.add_argument("--repo", default=None)
    k.add_argument("--manifest", default=None)

    m = sub.add_parser("manifest", help="harness-side: mint or extend a turn manifest")
    msub = m.add_subparsers(dest="mcmd", required=True)
    mn = msub.add_parser("new")
    mn.add_argument("manifest")
    mn.add_argument("--harness", required=True)
    mn.add_argument("--turn", required=True)
    ma = msub.add_parser("add")
    ma.add_argument("manifest")
    ma.add_argument("--id", required=True)
    ma.add_argument("--file", required=True)
    ma.add_argument("--kind", required=True, choices=sorted(SOURCE_KINDS_EXTERNAL | SOURCE_KINDS_AUTHOR))
    ma.add_argument("--complete", action="store_true")

    a = ap.parse_args(argv)

    if a.cmd == "canon":
        raw = Path(a.doc).read_bytes()
        mf = Manifest.load(a.manifest) if a.manifest else None
        side = to_sidecar(raw, Path(a.doc).name, a.commit, mf)
        out = Path(a.out) if a.out else Path(a.doc).with_suffix(".sworn.json")
        out.write_text(json.dumps(side, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("canonical: %d spans, text sha256 %s -> %s"
              % (len(side["spans"]), side["document"]["sha256"][:12], out.name))
        return 0

    if a.cmd == "render":
        side = load_sidecar(json.loads(Path(a.sidecar).read_text(encoding="utf-8")))
        raw = render(side)
        if a.out:
            Path(a.out).write_bytes(raw)
            print("rendered %d bytes -> %s" % (len(raw), a.out))
        else:
            sys.stdout.buffer.write(raw)
        return 0

    if a.cmd in ("verify", "check"):
        raw, side = _read_input(a.target)
        mf = Manifest.load(a.manifest) if a.manifest else None
        commit = getattr(a, "commit", None)
        if side is not None:
            commit = side["commit"]
        tree = _load_tree(a.repo, commit)
        if a.cmd == "verify":
            core = verify(raw, side, name=Path(a.target).name, manifest=mf, tree=tree, commit=commit)
            rec = issue_receipt(core)
            print(_headline(core))
            for s in core["spans"]:
                if s["verdict"] != "HELD":
                    print("  %-10s %-24s %s %s" % (s["verdict"], s["reason"] or "", s["receipt"] or "",
                                                  ("@%d" % s["at"])))
            for cl in core["coverage"]["unsworn_claims"][:20]:
                print("  UNSWORN-CLAIM? @%d: %s" % (cl["start"], cl["text"][:100]))
            if a.out:
                Path(a.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
                print("receipt %s -> %s" % (rec["digest"][:12], a.out))
            return 0
        rec = json.loads(Path(a.receipt).read_text(encoding="utf-8"))
        res = verify_receipt(rec, raw, side, manifest=mf, tree=tree)
        print("%s  digest=%s verdict-reproduces=%s same-build=%s"
              % (res["status"], res["digest_match"], res["verdict_reproduces"], res["same_verifier_build"]))
        return 0 if res["status"] == "VERIFIED" else 1

    if a.cmd == "manifest":
        p = Path(a.manifest)
        if a.mcmd == "new":
            Manifest(a.harness, a.turn).write(p)
            print("minted %s by %s for turn %s" % (p.name, a.harness, a.turn))
            return 0
        mf = Manifest.load(p)
        e = mf.add(a.id, Path(a.file).read_bytes(), a.kind, a.complete)
        mf.write(p)
        print("added %s %s sha256=%s complete=%s" % (e["id"], a.kind, e["sha256"][:12], e["complete"]))
        return 0
    return 2                                                   # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
