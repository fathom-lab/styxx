# -*- coding: utf-8 -*-
"""styxx.sworn — sworn output: the author declares, the receipt disposes.

Spec: ``papers/sworn/SPEC_sworn_output_v01_2026_09_01.md``, frozen before this file existed, and
``papers/sworn/SPEC_sworn_output_v02_2026_09_02.md``, frozen before the v0.2 changes below.
Format ``sworn/0.1`` (the grammar an author writes did not change). Manifest
``sworn/manifest/0.2`` (0.1 still loads). Verdict receipt ``styxx.sworn.verdict-receipt/v1``
(v0 still checks).

V0.2, IN ONE PARAGRAPH. The format was attacked twelve ways before any sentence about it left
the tree (``papers/sworn/ATTACKS_sworn_v01_battery_2026_09_02.md``, pinned by
``tests/test_sworn_attacks.py``). Four rules were the price: a tag hidden in an HTML comment is
MALFORMED ``hidden_commitment`` (R2); a short ``quote`` needle over a whole receipt is MALFORMED
``short_needle`` (R3); the cap counts code points (R4); ``rN#/pointer`` is legal (R1). The manifest
declares a trust rung and every span prints its provenance (R5–R7). And the coverage ESTIMATE is
withdrawn (R8): its denominator was counted by a diff-claim detector that never reads a measured
rate as a claim, so beside every result-shaped document it printed a number near 1.0 that meant
nothing; two counts that cannot flatter stand in its place. The verdict receipt digests its core
without coverage (R9), so a receipt re-derives wherever the observer differs.

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
exist when it was frozen. It exists now; ``papers/sworn/`` carries a row, and every sworn document
in the tree (the h declaration, this module's own RESULT, the measurement DESIGN, the
prose-claimhood CENSUS) is re-derived by ``tests/test_sworn_dogfood.py``. Nothing here is a
measurement of sworn output: that is owed item 3 of the spec, designed in
``papers/sworn/DESIGN_sworn_measurement_2026_09_01.md`` and still owed.

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
    "Manifest", "MemoryTree", "SnapshotTree", "GitTree",
    "verify", "issue_receipt", "verify_receipt",
    "main",
]

SPEC = "sworn/0.1"
# v0.2 (SPEC_sworn_output_v02_2026_09_02.md): the document grammar an author writes is unchanged,
# so the format string stays. The manifest and the verdict receipt moved; the old strings are
# still LOADED (M7: absence of a field is never a zero, and a v0 receipt in history must still
# check) and are never EMITTED.
MANIFEST_SPEC = "sworn/manifest/0.2"
MANIFEST_SPECS = ("sworn/manifest/0.1", "sworn/manifest/0.2")
RECEIPT_SCHEMA = "styxx.sworn.verdict-receipt/v1"
RECEIPT_SCHEMAS = ("styxx.sworn.verdict-receipt/v0", "styxx.sworn.verdict-receipt/v1")
COVERAGE_SCHEMA = "sworn/coverage/1"
# v0.2 R4: the cap counts Unicode code points (v0.1 counted bytes and disclosed the inequity).
# Still a pure function of bytes — invalid UTF-8 is document-level MALFORMED before any span is
# measured. The v0.1 name is kept for imports; both names hold the same number.
SPAN_CAP_CODEPOINTS = 300
SPAN_CAP_BYTES = SPAN_CAP_CODEPOINTS
# v0.2 R3: a `quote` needle under this many bytes, searched over a whole receipt or a line slice,
# is MALFORMED short_needle. A decision, not a measurement; pointer-leaf quotes and `absent` are
# exempt (the author named the leaf; a short absent needle is the conservative direction).
SHORT_NEEDLE_BYTES = 16
# v0.2 R6: the trust ladder a manifest declares. L0 (author-minted) is MALFORMED and has no rung
# string; L3 is reserved — nothing in v0.2 verifies a signature, so nothing may declare it.
RUNGS = ("L1", "L2")
RUNGS_RESERVED = ("L3",)
RUNG_UNDECLARED = "undeclared"
KINDS = ("numeric", "quote", "hash", "absent")
RESERVED_KINDS = ("exec",)
VERDICTS = ("HELD", "FAILED", "UNRESOLVED", "MALFORMED", "WITHHELD")
ROUNDING = "ROUND_HALF_EVEN"
MAX_RECEIPT_BYTES = 64 * 1024 * 1024      # a blob over this is UNRESOLVED receipt_too_large, unread

# The manifest's `kind_of_source` vocabulary, closed. Invariant 2: a receipt the agent wrote in the
# same turn is the agent swearing to itself. The verifier cannot see the turn; it can see what the
# harness RECORDED about where the bytes came from, and it refuses author-side kinds by name. An
# unknown kind is refused too — a kind nobody defined is not a kind the harness attested.
# v0.2 R5 adds `attestation`: DSSE / in-toto envelope BYTES a harness recorded. The kind names the
# shape of the bytes. No signature is verified, no key material is consulted, no trust root is
# pinned — the boundary styxx.evidence states under "WHAT `binding` IS NOT".
SOURCE_KINDS_EXTERNAL = frozenset({
    "tool_stdout", "tool_stderr", "file_read", "http_fetch", "harness_note", "test_report",
    "attestation",
})
SOURCE_KINDS_AUTHOR = frozenset({"agent_output", "agent_file_write", "agent_message"})

# Every non-HELD verdict carries one of these. A closed set, so a consumer can key on it.
REASONS = (
    # decidable from the document bytes alone (MALFORMED)
    "tag_syntax", "nesting", "stray_closer", "unclosed", "empty_span", "length_cap",
    "receipt_form", "kind_unknown", "kind_reserved", "number_count", "number_grammar",
    "needle_count", "needle_empty", "digest_form", "absent_over_partial", "hash_over_partial",
    "hidden_commitment", "short_needle",
    # decidable from the declaration plus the object the author named (MALFORMED: the author
    # had those exact bytes when it wrote the fragment)
    "pointer_unresolvable", "pointer_ambiguous", "anchor_out_of_range", "leaf_not_scalar",
    "leaf_not_numeric", "leaf_not_string", "receipt_not_json", "receipt_author_minted",
    "kind_of_source_unknown",
    # the verifier could not see the evidence (UNRESOLVED — never an accusation)
    "manifest_absent", "manifest_spec_unknown", "manifest_id_missing", "manifest_integrity",
    "manifest_bytes_absent", "manifest_no_completeness", "rung_unknown", "no_repository", "no_commit",
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
    "rn_grammar": ("^r[1-9][0-9]*$, optionally followed by the same `#` fragment grammar path: "
                   "takes — `/`-pointer (RFC 6901) or Ln[-Lm] — since v0.2 R1; absent and hash "
                   "over a fragment are MALFORMED exactly as over a partial path:"),
    "hidden_commitment": ("since v0.2 R2: a tag-shaped candidate inside an HTML comment "
                          "(`<!--` … `-->`, scanned outside fences and code spans, comments never "
                          "nesting, an unterminated comment running to end of document) is "
                          "MALFORMED hidden_commitment — never HELD, never narrative; it counts in "
                          "sworn_total and makes the document SWORN-FAILED"),
    "short_needle": ("since v0.2 R3: a quote needle under %d bytes searched over a WHOLE receipt "
                     "is MALFORMED short_needle; a quote against a JSON-pointer string leaf or a "
                     "line anchor is exempt (the author narrowed the haystack by naming it), and "
                     "absent is exempt (a short absent needle over a complete object is the "
                     "stronger oath); occurrences of a quote needle in its haystack are counted "
                     "into detail, report-only" % SHORT_NEEDLE_BYTES),
    "length_cap": ("since v0.2 R4: the inner text is capped at %d Unicode code points; a pure "
                   "function of bytes because invalid UTF-8 is document-MALFORMED first"
                   % SPAN_CAP_CODEPOINTS),
    "rung": ("since v0.2 R6: a manifest/0.2 declares rung L1 (a local hook sharing a filesystem "
             "and a shell with the agent — weak) or L2 (a runner that minted after the turn ended "
             "and the agent could not write to); L3 (signed, verified) is reserved and refused; a "
             "manifest/0.1, or a 0.2 with no rung, resolves at rung `undeclared`, never at L2; an "
             "unknown or reserved rung string makes every rN span UNRESOLVED rung_unknown — the "
             "verifier declining to see a manifest that claims what it cannot check, never an "
             "accusation against the author"),
    "provenance": ("since v0.2 R7: every span verdict carries `provenance` — for rN "
                   "{harness, rung, kind_of_source}; for path: and prereg: the literal `committed "
                   "object at <commit>; authorship unchecked` — and the receipt carries a `rungs` "
                   "count; nothing verifies a rung, the verifier prints what the manifest declared"),
    "receipt_v1": ("since v0.2 R9: the receipt digest covers the core WITHOUT coverage; coverage "
                   "travels beside it under coverage_sha256; verify_receipt re-derives the core and "
                   "reports coverage_reproduces separately (advisory); a /v0 receipt is compared on "
                   "its core minus coverage minus verifier, its digest still checked over its full "
                   "body, and the note says so"),
    "path_grammar": ("relative, /-separated, no empty/./.. segment, no backslash, no whitespace, no "
                     "glob metacharacter (* ? [ ]), no leading ':' (git pathspec magic) — a path "
                     "names ONE committed file, never a set the verifier would pick from; split at "
                     "the FIRST #; fragment is `/`-pointer (RFC 6901, ~0 ~1 only) or Ln[-Lm]"),
    "prereg_search": "the tree at the sidecar's commit, blobs only, memoised per commit",
    "manifest_bytes": "standard base64; an entry whose bytes do not hash to its sha256 is UNRESOLVED",
    "author_minted": ("kind_of_source in {agent_output, agent_file_write, agent_message}, or a "
                      "receipt sha256 listed in the manifest's authored_sha256 (every byte-object the "
                      "agent produced this turn, recorded by the harness), is MALFORMED "
                      "receipt_author_minted; a kind outside the closed vocabulary is MALFORMED "
                      "kind_of_source_unknown; complete missing from an rN used with absent is "
                      "UNRESOLVED manifest_no_completeness, complete:false is MALFORMED"),
    "coverage": ("since v0.2 R8 the ESTIMATE IS WITHDRAWN: the denominator instrument "
                 "(styxx.claimdetect, STRUCT-1) is a diff-claim detector measured on agent "
                 "pull-request prose and by its own docstring never reads result-shaped sentences "
                 "as claims, so across the twelve committed sworn receipts at 320b303 it printed "
                 "0.6667-1.0 while counting 0-8 of tens of narrative sentences; the block is now "
                 "schema sworn/coverage/1 carrying sworn_total, narrative_sentences (diffgate's "
                 "splitter `(?<=[.!?])\\s+|\\n+` over canonical text minus sworn spans minus "
                 "fenced regions, non-empty), sentence_share = sworn_total / (sworn_total + "
                 "narrative_sentences) — a FLOOR that treats every narrative sentence as "
                 "load-bearing and so cannot flatter — and diff_claim_sentences / "
                 "diff_claim_share from STRUCT-1 labelled with its idiom and ceiling; 0/0 is null; "
                 "ALWAYS advisory; never a measurement of bound recall"),
    "exit_codes": ("verify exits 0 for EVERY document verdict — SWORN-HELD, SWORN-FAILED, UNSWORN, "
                   "document-level MALFORMED — because sworn reports and never gates; a refusal "
                   "(undecodable document, a sidecar that cannot round-trip or carries an unknown "
                   "shape, a manifest that disagrees with the embedded one) is SystemExit('REFUSED: "
                   "…'), exit status 1, nothing written; check exits 1 when a receipt does not "
                   "re-derive"),
    "html_comments": ("v0.1 recognised a tag inside an HTML comment like any other and named the "
                      "hidden commitment as owed; v0.2 R2 closes it — see `hidden_commitment`"),
}

_CERTIFIES = (
    "the spans the author bound were checked against bytes the author did not write, at the commit "
    "or manifest the document names and at the rung the manifest declares — NOT a claim that the "
    "document is correct, NOT a claim that the right sentences were bound, NOT a check that the tags "
    "were written at write time, NOT a check of any signature, and only as trustworthy as the harness "
    "that minted the manifest and the history that holds the commit"
)
_COVERAGE_CEILING = ("advisory, always: sentence_share is a floor computed by diffgate's sentence "
                     "splitter (the lane's largest false-flag source) that treats every narrative "
                     "sentence as load-bearing; diff_claim_share is counted by styxx.claimdetect "
                     "(STRUCT-1), a diff-claim detector for agent pull-request prose measured at "
                     "precision 0.4211 on n=38 by one model family in-house, which does not read "
                     "result-shaped sentences (measured rates, test totals) as claims; neither number "
                     "is bound recall — that is a blind panel's to measure, and it has not")
_PROVENANCE_COMMITTED = "committed object at %s; authorship unchecked"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_json_lf(path, obj) -> Path:
    """Write a byte-pinned JSON artifact with LF line endings on every platform.

    A text-mode write on Windows translates LF to CRLF; a sidecar or receipt written that way
    hashes differently from the same artifact written on Linux, and the CRLF lesson has already
    cost this repository once (styxx/centroids, then papers/sworn/**). Every JSON this module
    writes goes through here.
    """
    p = Path(path)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    return p


def _safe_text(x: Any, limit: int = 80) -> str:
    """A detail string that can always be serialised: lone surrogates from JSON escapes are
    replaced, never allowed to crash a receipt."""
    return str(x)[:limit].encode("utf-8", errors="replace").decode("utf-8")


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
    """What the lexer saw: ``declarations``, ``fenced`` regions, ``comments`` (HTML comment
    regions in inline coordinates, v0.2 R2), ``document_malformed``, ``canonical`` bytes (None
    when a lexical MALFORMED makes it undefined), ``lexical_ok``."""


_COMMENT_OPEN = b"<!--"
_COMMENT_CLOSE = b"-->"


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
    out = Scan(declarations=[], fenced=[], comments=[], document_malformed=None, canonical=None,
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
    comment_end = -1                       # v0.2 R2: end of the HTML comment p is inside, if any
    while p < n:
        skip_to = _in_regions(p, regions)
        if skip_to is not None:
            p = skip_to
            continue
        c = raw[p:p + 1]
        if c == b"<" and raw.startswith(_COMMENT_OPEN, p) and p >= comment_end:
            # An HTML comment, met outside fences and code spans. Comments never nest; an
            # unterminated one runs to the end of the document (HTML semantics). A tag met
            # before comment_end is a hidden commitment: MALFORMED, never HELD, never narrative.
            close = raw.find(_COMMENT_CLOSE, p + len(_COMMENT_OPEN))
            comment_end = n if close < 0 else close + len(_COMMENT_CLOSE)
            out["comments"].append((p, comment_end))
            p += len(_COMMENT_OPEN)
            continue
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
                    if p < comment_end:
                        d["malformed"] = d["malformed"] or "hidden_commitment"
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
                        if not d["inner"] and d["malformed"] in (None, "hidden_commitment"):
                            # zero bytes sworn: MALFORMED from the bytes, and unrepresentable in
                            # the sidecar (start == end cannot be ordered against a neighbour) —
                            # so it outranks hidden_commitment, which the sidecar CAN carry
                            d["malformed"] = "empty_span"
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
        # a declaration still open at end of document is unclosed whatever else it was — a
        # hidden commitment with no closer has no offsets and cannot be canonicalised
        if d["malformed"] in (None, "hidden_commitment"):
            d["malformed"] = "unclosed"

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
    if not (isinstance(man, dict) and man.get("spec") in MANIFEST_SPECS
            and isinstance(man.get("receipts"), dict)
            and all(isinstance(k, str) and isinstance(v, dict) for k, v in man["receipts"].items())
            and isinstance(man.get("authored_sha256", []), list)
            and all(isinstance(x, str) for x in man.get("authored_sha256", []))):
        _refuse("sidecar manifest is not a %s object" % " / ".join(MANIFEST_SPECS))
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
                 authored_sha256: Optional[List[str]] = None,
                 rung: Optional[str] = None, spec: str = MANIFEST_SPEC):
        self.harness = harness
        self.turn = turn
        self.minted_at = minted_at or _now()
        self.receipts: Dict[str, dict] = dict(receipts or {})
        # sha256 of every byte-object the agent produced this turn, as the harness saw them: files
        # written, messages emitted, stdin fed to tools. Invariant 2 becomes set membership.
        self.authored_sha256: List[str] = [x.lower() for x in (authored_sha256 or [])]
        # v0.2 R6: the rung the harness DECLARES. Stored as given, validated at resolution — a
        # manifest that loads must never crash the verifier, and a rung it cannot check makes
        # every rN UNRESOLVED rung_unknown rather than raising here.
        self.rung: Optional[str] = rung
        # The spec string this manifest was written under; core() reproduces exactly that shape so
        # a loaded 0.1 manifest re-derives its own digest.
        self.spec: str = spec if spec in MANIFEST_SPECS else MANIFEST_SPEC

    def rung_status(self) -> Tuple[str, Optional[str]]:
        """("ok", rung) | ("undeclared", None) | ("unknown", rung)."""
        if self.rung is None:
            return "undeclared", None
        if self.rung in RUNGS:
            return "ok", self.rung
        return "unknown", str(self.rung)

    def record_authored(self, data: bytes) -> str:
        h = _sha256(data)
        if h not in self.authored_sha256:
            self.authored_sha256.append(h)
        return h

    def add(self, rid: str, data: Optional[bytes], kind_of_source: str, complete: bool = False,
            captured_at: Optional[str] = None, sha256: Optional[str] = None,
            note: Optional[str] = None) -> dict:
        if not re.fullmatch(r"r[1-9][0-9]*", rid):
            raise ValueError("receipt id must match r[1-9][0-9]*: %r" % rid)
        if data is None and not sha256:
            raise ValueError("a receipt needs bytes or at least a sha256")
        if sha256 is not None:
            sha256 = str(sha256).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise ValueError("sha256 must be 64 hex characters")
            if data is not None and _sha256(data) != sha256:
                raise ValueError("sha256 does not match the bytes")
        entry = {
            "id": rid,
            "sha256": sha256 or _sha256(data),
            "kind_of_source": kind_of_source,
            "captured_at": captured_at or _now(),
            "complete": bool(complete),
        }
        if data is not None:
            entry["bytes"] = _b64(data)
        if note is not None:
            entry["harness_note"] = str(note)          # v0.2 R6: e.g. the command that printed it
        self.receipts[rid] = entry
        return entry

    def core(self) -> dict:
        d = {"spec": self.spec, "harness": self.harness, "turn": self.turn,
             "minted_at": self.minted_at, "authored_sha256": sorted(self.authored_sha256),
             "receipts": self.receipts}
        if self.spec == "sworn/manifest/0.2":
            d["rung"] = self.rung
        return d

    def digest(self) -> str:
        return _sha256(_jcs(self.core()).encode("utf-8"))

    def to_dict(self) -> dict:
        d = self.core()
        d["digest"] = self.digest()
        return d

    def write(self, path) -> Path:
        return _write_json_lf(path, self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        if not isinstance(d, dict) or d.get("spec") not in MANIFEST_SPECS:
            raise SystemExit("REFUSED: unknown manifest spec %r (this verifier knows %s)"
                             % (d.get("spec") if isinstance(d, dict) else None,
                                ", ".join(MANIFEST_SPECS)))
        receipts = d.get("receipts")
        if receipts is None:
            receipts = {}
        authored = d.get("authored_sha256")
        if authored is None:
            authored = []
        if not isinstance(receipts, dict) or not all(isinstance(k, str) for k in receipts):
            raise SystemExit("REFUSED: manifest receipts must be an object keyed by receipt id")
        if not isinstance(authored, list) or not all(isinstance(x, str) for x in authored):
            raise SystemExit("REFUSED: manifest authored_sha256 must be a list of hex strings")
        m = cls(d.get("harness", ""), d.get("turn", ""), d.get("minted_at"), receipts, authored,
                rung=d.get("rung"), spec=d.get("spec"))
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
        """False when the harness's digest does not re-derive — or cannot even be computed,
        because the manifest carries something no canonical serialisation can hold (a NaN, a
        non-JSON value). Either way the verifier could not see a sound manifest."""
        try:
            return self.declared_digest is None or self.declared_digest == self.digest()
        except (TypeError, ValueError):
            return False

    def digest_or_none(self) -> Optional[str]:
        try:
            return self.digest()
        except (TypeError, ValueError):
            return None


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


class SnapshotTree:
    """A tree snapshot WITH MODES at one commit: ``{path: {mode, size, sha256, bytes}}``.

    ``MemoryTree`` carries bytes and no modes, so it can never say ``not_a_blob``,
    ``receipt_too_large`` or ``commit_absent``; ``GitTree`` says them but needs a git binary.
    This handle reproduces every reason ``GitTree`` can return except ``git_unavailable`` from a
    dict alone, so a recorded tree (SPEC_sworn_conformance_vectors_v01_2026_09_05.md, C10)
    replays anywhere. Whatever reads git to build one lives outside this module.

    ``snapshot_commit`` is the commit the entries were read at. ``commit`` is the handle's own
    commit and ``None`` when it names none, exactly ``MemoryTree``'s rule; ``verify()``
    overwrites it with the document's commit as it does for every handle. A handle whose commit
    is not the snapshot's, or is not a full lowercase hex id, sees ``commit_absent``: the snapshot
    was not taken there. An entry whose bytes were not embedded (``bytes`` is ``None``, or
    ``size`` is over ``MAX_RECEIPT_BYTES``) is ``receipt_too_large``: the bytes are not here.
    Modes ``040000`` (a tree), ``120000`` (a symlink) and ``160000`` (a gitlink) are
    ``not_a_blob`` whatever bytes they carry, as ``GitTree`` rules.
    """

    BLOB_MODES = ("100644", "100755")

    def __init__(self, entries: Dict[str, dict], snapshot_commit: Optional[str],
                 commit: Optional[str] = None):
        self.entries: Dict[str, dict] = {p: dict(e) for p, e in entries.items()}
        self.snapshot_commit = snapshot_commit
        self.commit = commit

    def _ready(self) -> Optional[str]:
        if self.commit is None:
            return "no_commit"
        if not (isinstance(self.commit, str)
                and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.commit)):
            return "commit_absent"
        if self.commit != self.snapshot_commit:
            return "commit_absent"
        return None

    def blob(self, path: str) -> Tuple[Optional[bytes], str]:
        why = self._ready()
        if why:
            return None, why
        entry = self.entries.get(path)
        if entry is None:
            return None, "path_absent"
        if entry.get("mode") not in self.BLOB_MODES:
            return None, "not_a_blob"
        data = entry.get("bytes")
        size = entry.get("size")
        if size is None and data is not None:
            size = len(data)
        if data is None or (size is not None and size > MAX_RECEIPT_BYTES):
            return None, "receipt_too_large"
        return data, "ok"

    def find_sha256(self, digest: str) -> Tuple[Optional[bytes], str]:
        why = self._ready()
        if why:
            return None, why
        for path in sorted(self.entries):
            entry = self.entries[path]
            data = entry.get("bytes")
            if entry.get("mode") in self.BLOB_MODES and data is not None and _sha256(data) == digest:
                return data, "ok"
        return None, "prereg_not_in_tree"

    @classmethod
    def from_memory(cls, tree: MemoryTree) -> SnapshotTree:
        """The snapshot a ``MemoryTree`` is: every file a regular blob, at the handle's commit."""
        entries = {p: {"mode": "100644", "size": len(b), "sha256": _sha256(b), "bytes": b}
                   for p, b in tree.files.items()}
        return cls(entries, tree.commit, commit=tree.commit)


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
        # Size first, bytes second, and never the whole tree in memory at once: `--batch-check`
        # names every blob's size without its body, blobs over the receipt cap are never read,
        # and the bodies that are read stream through one hasher a chunk at a time. Reading the
        # whole tree into one buffer at every commit a document names once grew a re-derivation
        # to 13 GB and died (the charon red team, 2026-09-02).
        rc, out, _ = self._git("cat-file", "--batch-check", stdin=("\n".join(shas) + "\n").encode("ascii"))
        if rc != 0:
            return None
        wanted = []
        for line in out.split(b"\n"):
            parts = line.split()
            if len(parts) >= 3 and parts[1] == b"blob" and parts[2].isdigit() and int(parts[2]) <= MAX_RECEIPT_BYTES:
                wanted.append(parts[0].decode("ascii"))
        idx: Dict[str, str] = {}
        if not wanted:
            _PREREG_INDEX[key] = idx
            return idx
        try:
            proc = subprocess.Popen(["git", "-C", str(self.repo), "cat-file", "--batch"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except (OSError, ValueError):
            return None
        assert proc.stdin is not None and proc.stdout is not None
        # feed stdin from its own thread: writing thousands of ids into a full pipe while git
        # blocks on a stdout nobody is reading yet is a deadlock, not a slow run
        import threading
        payload = ("\n".join(wanted) + "\n").encode("ascii")

        def _feed():
            try:
                proc.stdin.write(payload)
            except (OSError, ValueError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except (OSError, ValueError):
                    pass
        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()
        try:
            stream = proc.stdout
            while True:
                header = stream.readline()
                if not header:
                    break
                parts = header.split()
                if len(parts) < 3 or parts[1] != b"blob":
                    continue                                     # "missing" lines carry no body
                size = int(parts[2])
                h = hashlib.sha256()
                remaining = size
                while remaining > 0:
                    chunk = stream.read(min(remaining, 1 << 20))
                    if not chunk:
                        break
                    h.update(chunk)
                    remaining -= len(chunk)
                stream.read(1)                                   # the trailing LF after each body
                if remaining == 0:
                    idx.setdefault(h.hexdigest(), parts[0].decode("ascii"))
        finally:
            proc.stdout.close()
            proc.wait()
            feeder.join(timeout=5)
        if proc.returncode != 0:
            return None
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
_PATH_SEG_BAD = re.compile(r"[\\\s\x00-\x1f\x7f*?\[\]]")   # no whitespace, control, glob


def _parse_receipt(ref: str) -> Tuple[Optional[dict], Optional[str]]:
    """The receipt grammar, decidable from bytes. Returns (parsed, malformed_reason)."""
    if ref is None:
        return None, "receipt_form"
    if _RN.fullmatch(ref.split("#", 1)[0]):
        # v0.2 R1: an rN may carry the same fragment grammar path: takes, so a numeric span can
        # name a leaf inside a harness capture instead of needing a one-number capture.
        form = "rn"
        if "#" in ref:
            target, frag = ref.split("#", 1)
        else:
            target, frag = ref, None
    elif ref.startswith("path:"):
        body = ref[5:]
        form = "path"
        target: Any
        if "#" in body:
            target, frag = body.split("#", 1)
        else:
            target, frag = body, None
        # a leading ':' is git pathspec magic (':/', ':(top)', ':!', ':^') and would let a path
        # mean something other than the committed file it names; refused at the grammar
        if not target or target.startswith(("/", ":")) or _PATH_SEG_BAD.search(target):
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
    return {"form": form, "target": target, "id": target if form == "rn" else None,
            "fragment": fragment, "partial": fragment is not None}, None


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
    provenance: Dict[str, Any] = {}
    if parsed["form"] == "rn":
        if manifest is None:
            return _Resolved(status="unresolved", reason="manifest_absent")
        if not manifest.intact():
            return _Resolved(status="unresolved", reason="manifest_integrity")
        rung_state, rung = manifest.rung_status()
        if rung_state == "unknown":
            # v0.2 R6: a manifest claiming a rung this verifier cannot check (L3, or a string
            # nobody defined). The verifier declines to see it; it accuses nobody.
            return _Resolved(status="unresolved", reason="rung_unknown")
        if parsed["id"] not in manifest.receipts:
            return _Resolved(status="unresolved", reason="manifest_id_missing")
        entry = manifest.receipts[parsed["id"]]
        if not isinstance(entry, dict):                  # present, but not a receipt entry
            return _Resolved(status="unresolved", reason="manifest_integrity")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            return _Resolved(status="unresolved", reason="manifest_integrity")
        kos = entry.get("kind_of_source")
        if not isinstance(kos, str):
            return _Resolved(status="malformed", reason="kind_of_source_unknown")
        if kos in SOURCE_KINDS_AUTHOR or sha in manifest.authored_sha256:
            return _Resolved(status="malformed", reason="receipt_author_minted")
        if kos not in SOURCE_KINDS_EXTERNAL:
            return _Resolved(status="malformed", reason="kind_of_source_unknown")
        provenance = {"form": "rn", "harness": manifest.harness,
                      "rung": rung if rung_state == "ok" else RUNG_UNDECLARED,
                      "kind_of_source": kos}
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
        provenance = {"form": parsed["form"],
                      "note": _PROVENANCE_COMMITTED % (getattr(tree, "commit", None) or "?")}
    res = _Resolved(status="ok", reason=None, bytes=data, sha256=sha, complete=complete,
                    leaf=None, has_leaf=False, slice=None, provenance=provenance)
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
    if not spans[0].strip(b" \t\r\n\f\v"):
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
        return "MALFORMED", "leaf_not_numeric", {"leaf": _safe_text(leaf)}
    if not leaf.is_finite() or leaf.adjusted() > 320:
        return "MALFORMED", "leaf_not_numeric", {"leaf": _safe_text(leaf)}
    try:
        lhs, rhs = _canon(leaf, frac), _canon(printed, frac)
    except (InvalidOperation, ValueError):
        return "MALFORMED", "leaf_not_numeric", {"leaf": _safe_text(leaf)}
    detail = {"printed_token": tok, "printed": rhs, "receipt": str(leaf), "receipt_rounded": lhs,
              "fractional_digits": frac, "rounding": ROUNDING}
    # DECISIONS["rounding"] is deliberate and it is right: an author writing 0.42 against a receipt
    # of 0.4211 is honestly rounding, and demanding an exact match would FAIL every rounded figure
    # in the corpus. But the rule quantizes to the AUTHOR'S printed precision and has no floor, so
    # at zero fractional digits it stops rounding and starts erasing: a receipt of 0.4211 against
    # the sentence "the A-share is 0." is HELD, with a genuine harness-minted receipt and nothing
    # malformed anywhere.
    #
    # The line drawn here is not a threshold anyone has to argue about. It is the case where the
    # printed figure carries NO information about the receipt at all: a non-zero receipt that
    # rounds to zero. The verdict does not change — that would break the honest-rounding rule this
    # format needs — but the span says so, so a reader of the receipt is not left to notice that
    # `receipt` and `receipt_rounded` disagree by everything.
    # NOTE: nothing is added to `detail` here, and the first version of this signal did. `detail`
    # is INSIDE the digested core, so a field added to it moves the core digest of every affected
    # span — the conformance generator refused the regeneration and said so: "a moved core is a
    # finding about the verifier, never a reason to rewrite the set". It would also have put this
    # side out of agreement with styxx/_data/sworn_verify.js, which knows nothing about it.
    #
    # `receipt` and `receipt_rounded` are already here and already say it. The signal belongs where
    # a reader meets the verdict — the headline, which is CLI output and outside the digest.
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
        try:
            hay = res["leaf"].encode("utf-8")
        except UnicodeEncodeError:                  # a lone surrogate is not text this can compare
            return "MALFORMED", "leaf_not_string", {"note": "leaf is not encodable as UTF-8"}
    else:
        hay = res["slice"] if res["slice"] is not None else res["bytes"]
        # v0.2 R3: a short needle over a WHOLE receipt HELDs against almost anything; the author
        # must quote enough bytes to mean something. A pointer leaf (above) and a line slice are
        # exempt — the author narrowed the haystack by naming it, and the comparison is against
        # that alone.
        #
        # "Narrowed" is the operative word, and the check used to ask only whether a slice was
        # PRESENT. `#L1-L3` over a three-line receipt is the whole receipt with a line anchor on
        # it: nothing narrowed, floor gone, a two-byte needle HELD.
        #
        # The exemption is earned when the slice is NOT what a full-range slice of the receipt
        # would be. Not a raw byte comparison — _line_slice excludes the last selected line's
        # terminating LF by design, so a full-range slice of a newline-terminated receipt is one
        # byte short of the receipt and a length test calls it narrowed. That was the first repair,
        # and it left the trailing-newline case open. Asking the slicer itself what "everything"
        # looks like is exact, and the JavaScript side asks the same question of the same function.
        #
        # One more clause, and it comes from a decision this lab had already written down in its
        # own tests: "a nine-byte receipt cannot hold a sixteen-byte needle; the author narrows the
        # haystack with a line anchor and the short needle is then exempt." A receipt BELOW the
        # floor cannot be the danger the floor targets — two bytes over nine do not hold against
        # almost anything — so an anchor over such a receipt keeps its exemption even though it
        # narrows nothing. The strict reading ("a one-line file is its own whole") is right for a
        # one-line file at or above the floor, where #L1 over a 10 KB minified blob would narrow
        # nothing and mean nothing, and wrong for the tiny fixture the prior decision was about.
        if res["slice"] is not None:
            whole = res["bytes"]
            n_lines = whole.count(b"\n") + (1 if whole and not whole.endswith(b"\n") else 0)
            narrowed = (res["slice"] != _line_slice(whole, 1, n_lines)
                        or len(whole) < SHORT_NEEDLE_BYTES)
        else:
            narrowed = False
        if not narrowed and len(needle) < SHORT_NEEDLE_BYTES:
            return "MALFORMED", "short_needle", {"needle_bytes": len(needle),
                                                 "minimum_bytes": SHORT_NEEDLE_BYTES}
    detail = {"needle_bytes": len(needle), "haystack_bytes": len(hay),
              "occurrences": hay.count(needle)}
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
        if res is not None and res.get("provenance"):
            verdict["provenance"] = res["provenance"]          # v0.2 R7
        return verdict

    if d["malformed"]:
        return out("MALFORMED", d["malformed"], {"raw": d.get("raw")} if d.get("raw") else {})
    inner: bytes = d["inner"]
    if not inner.strip(b" \t\r\n\f\v"):
        return out("MALFORMED", "empty_span")
    code_points = len(inner.decode("utf-8"))              # the document decoded strictly already
    if code_points > SPAN_CAP_CODEPOINTS:                   # v0.2 R4
        return out("MALFORMED", "length_cap", {"code_points": code_points, "bytes": len(inner),
                                               "cap": SPAN_CAP_CODEPOINTS})
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
    """Advisory, always — and since v0.2 R8 an ESTIMATE no longer exists here.

    Two counts are printed instead. ``narrative_sentences`` is the splitter's count over the
    narrative, and ``sentence_share`` treats every one of them as load-bearing: a floor that cannot
    flatter a document. ``diff_claim_sentences`` is STRUCT-1's count, labelled with the idiom it
    was measured on, because it does not read a measured rate as a claim and printing its ratio
    unlabelled beside a result-shaped document is exactly how v0.1 came to print 0.94.
    """
    cov: Dict[str, Any] = {
        "schema": COVERAGE_SCHEMA, "advisory": True, "ceiling_note": _COVERAGE_CEILING,
        "sworn_total": sworn_total, "narrative_sentences": None, "sentence_share": None,
        "diff_claim_sentences": None, "diff_claim_share": None,
        "diff_claim_idiom": "agent pull-request prose (styxx.claimdetect STRUCT-1, precision "
                            "0.4211 on n=38, one model family, in-house); result-shaped sentences "
                            "are never counted by it",
        "unsworn_claims": [], "splitter": "diffgate:(?<=[.!?])\\s+|\\n+", "claimdetect_version": None,
    }
    if canonical is None:
        cov["note"] = "no canonical text: the document is lexically MALFORMED"
        return cov
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
    pos = 0
    pieces = []
    for m in _SENTENCE_SPLIT.finditer(narrative):
        pieces.append((pos, m.start()))
        pos = m.end()
    pieces.append((pos, len(narrative)))
    sentences = []
    for a, b in pieces:
        seg = narrative[a:b]
        if not seg.strip():
            continue
        lead = len(seg) - len(seg.lstrip())
        sentences.append((a + lead, a + lead + len(seg.strip()), seg.strip()))
    n_sent = len(sentences)
    cov["narrative_sentences"] = n_sent
    denom = sworn_total + n_sent
    cov["sentence_share"] = (round(sworn_total / denom, 4) if denom else None)
    try:
        from styxx.claimdetect import STRUCT1_VERSION, detect
    except Exception:                                        # pragma: no cover - observer optional
        cov["note"] = "styxx.claimdetect unavailable; diff-claim count not taken"
        return cov
    cov["claimdetect_version"] = STRUCT1_VERSION
    claims = []
    for a, b, seg in sentences:
        text = seg.decode("utf-8", errors="replace")
        try:
            is_claim = bool(detect(text).is_claim)
        except Exception:                               # the observer failing is not a verdict
            cov["note"] = "styxx.claimdetect raised; diff-claim count not taken"
            cov["unsworn_claims"] = []
            cov["diff_claim_sentences"] = None
            cov["diff_claim_share"] = None
            return cov
        if is_claim:
            claims.append({"start": a, "end": b, "text": text[:200]})
    n_claims = len(claims)
    cov["unsworn_claims"] = claims
    cov["diff_claim_sentences"] = n_claims
    d2 = sworn_total + n_claims
    cov["diff_claim_share"] = (round(sworn_total / d2, 4) if d2 else None)
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
            if manifest is not None and manifest.digest_or_none() != embedded.digest_or_none():
                raise SystemExit("REFUSED: the supplied manifest disagrees with the embedded one")
            manifest = manifest or embedded
    if raw is None:
        raise SystemExit("REFUSED: nothing to verify")
    if commit is not None and not (isinstance(commit, str)
                                   and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)):
        raise SystemExit("REFUSED: commit must be a full lowercase hex object id or None, not %r"
                         % (commit,))
    if tree is not None:
        # The receipts resolve AT THE COMMIT THE DOCUMENT NAMES. A tree handle is a repository,
        # not a choice of commit: whatever it was built with, the sidecar's commit (or the
        # caller's explicit one for an inline document) is the one that counts, and a document
        # that names none gets no commit however the handle was built.
        if commit is None and sidecar is None:
            commit = getattr(tree, "commit", None)
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
    # v0.2 R7: how many spans stood on which rung. Nothing here verifies a rung; it is what the
    # manifest declared, counted so a reader sees it at the same prominence as the verdict.
    rungs: Dict[str, int] = {}
    for v in verdicts:
        prov = v.get("provenance") or {}
        key = prov.get("rung") if prov.get("form") == "rn" else (
            "committed" if prov.get("form") in ("path", "prereg") else "unresolved")
        rungs[key] = rungs.get(key, 0) + 1
    from styxx._version import __version__
    core = {
        "schema": RECEIPT_SCHEMA,
        "format": SPEC,
        "document": {"name": name, "inline_sha256": _sha256(raw),
                     "canonical_sha256": _sha256(sc["canonical"]) if sc["canonical"] is not None else None},
        "commit": commit,
        "manifest_digest": manifest.digest_or_none() if manifest is not None else None,
        "spans": verdicts,
        "counts": counts,
        "sworn_total": sworn_total,
        "unresolved": counts["UNRESOLVED"],
        "document_verdict": document_verdict,
        "document_malformed": doc_malformed,
        "rungs": rungs,
        "coverage": coverage,
        "verifier": {"styxx_version": __version__,
                     "sworn_sha256": _sha256(Path(__file__).read_bytes()),
                     "rounding": ROUNDING, "decisions": DECISIONS},
        "certifies": _CERTIFIES,
    }
    return core


_RECEIPT_OUTSIDE_DIGEST = ("digest", "timestamp", "coverage", "coverage_sha256")


def issue_receipt(core: dict, timestamp: Optional[str] = None) -> dict:
    """Content-address a verdict core: ``digest`` over the JCS form of the core WITHOUT
    coverage (v0.2 R9); coverage travels beside it under its own ``coverage_sha256``. Re-derivable
    by anyone with the document, manifest and tree — including a verifier with no observer."""
    rec = {k: v for k, v in core.items() if k not in _RECEIPT_OUTSIDE_DIGEST}
    rec["digest"] = _sha256(_jcs(rec).encode("utf-8"))
    cov = core.get("coverage")
    if cov is not None:
        rec["coverage"] = cov
        rec["coverage_sha256"] = _sha256(_jcs(cov).encode("utf-8"))
    rec["timestamp"] = timestamp or _now()
    return rec


def _v0_shape(fresh: dict) -> dict:
    """Project a v0.2 core onto the fields a /v0 receipt carried, so a receipt in history is
    compared on what it said and not on what a later verifier learned to print."""
    # `certifies` is the verifier's own boundary sentence and moves with the verifier build, so
    # it is excluded here exactly as the `verifier` block is.
    out = {k: v for k, v in fresh.items()
           if k not in ("verifier", "coverage", "rungs", "certifies", "schema")}
    spans = []
    for s in out.get("spans", []):
        s2 = {k: v for k, v in s.items() if k != "provenance"}
        s2["detail"] = {k: v for k, v in (s.get("detail") or {}).items() if k != "occurrences"}
        spans.append(s2)
    out["spans"] = spans
    return out


def verify_receipt(receipt: dict, raw: Optional[bytes] = None, sidecar: Optional[dict] = None, *,
                   manifest: Optional[Manifest] = None, tree=None) -> dict:
    """Re-derive a receipt against the presented document. Trust neither the author (bytes are
    hashed) nor the verifier that issued it (the verdict is re-run).

    Schema-aware (M7: key on the schema string, never on key presence). A /v1 receipt digests
    its core without coverage and is compared on that core; coverage reproduction is reported
    beside it, advisory. A /v0 receipt digested everything, so its digest is checked over its
    full body, and it is compared on its core minus coverage minus verifier, projected onto the
    shape v0 carried — its coverage block cannot reproduce under this verifier and the note says so.
    """
    fail = {"status": "FAILED", "digest_match": False, "verdict_reproduces": False,
            "coverage_reproduces": None, "same_verifier_build": False, "schema": None}
    if not isinstance(receipt, dict):
        return dict(fail, note="not a receipt object")
    schema = receipt.get("schema")
    if schema not in RECEIPT_SCHEMAS:
        return dict(fail, schema=schema, note="unknown receipt schema %r (this verifier knows %s)"
                    % (schema, ", ".join(RECEIPT_SCHEMAS)))
    v0 = schema == "styxx.sworn.verdict-receipt/v0"
    digest_body = ({k: v for k, v in receipt.items() if k not in ("digest", "timestamp")} if v0
                   else {k: v for k, v in receipt.items() if k not in _RECEIPT_OUTSIDE_DIGEST})
    try:
        digest_ok = receipt.get("digest") == _sha256(_jcs(digest_body).encode("utf-8"))
    except (TypeError, ValueError):
        digest_ok = False
    core = {k: v for k, v in receipt.items() if k not in _RECEIPT_OUTSIDE_DIGEST}
    doc = core.get("document") if isinstance(core.get("document"), dict) else {}
    ver = core.get("verifier") if isinstance(core.get("verifier"), dict) else {}
    note = "verify-by-re-derivation: the document is hashed and the verdict is re-run"
    if v0:
        note += (" — /v0 receipt: compared on its core minus coverage minus verifier, projected "
                 "onto the v0 shape; its coverage block predates sworn/coverage/1 and is not compared")
    try:
        fresh = verify(raw, sidecar, name=doc.get("name", ""), manifest=manifest, tree=tree,
                       commit=core.get("commit"))
    except SystemExit as e:
        # a receipt whose fields no longer describe a verifiable document does not re-derive;
        # that is a FAILED re-derivation, not a refusal of the caller
        return dict(fail, digest_match=digest_ok, schema=schema, note=note + " — " + str(e.code))
    # the verifier block names the build; a different build is reported, not hidden
    same_build = fresh["verifier"]["sworn_sha256"] == ver.get("sworn_sha256")
    if v0:
        cmp_fresh = _v0_shape(fresh)
        cmp_core = {k: v for k, v in core.items() if k not in ("verifier", "schema", "certifies")}
    else:
        cmp_fresh = {k: v for k, v in fresh.items() if k not in ("verifier", "coverage")}
        cmp_core = {k: v for k, v in core.items() if k != "verifier"}
    try:
        reproduces = _jcs(cmp_fresh) == _jcs(cmp_core)
    except (TypeError, ValueError):
        reproduces = False
    coverage_ok: Optional[bool] = None
    if not v0 and isinstance(receipt.get("coverage"), dict):
        try:
            coverage_ok = _jcs(fresh["coverage"]) == _jcs(receipt["coverage"])
        except (TypeError, ValueError):
            coverage_ok = False
    return {"status": "VERIFIED" if (digest_ok and reproduces) else "FAILED",
            "digest_match": digest_ok, "verdict_reproduces": reproduces,
            "coverage_reproduces": coverage_ok, "same_verifier_build": same_build,
            "schema": schema, "note": note}


# =============================================================================================
# 6. CLI — reports, never gates
# =============================================================================================

def _headline(core: dict) -> str:
    c = core["counts"]
    cov = core["coverage"]
    share = "n/a" if cov.get("sentence_share") is None else "%.2f" % cov["sentence_share"]
    nsent = cov.get("narrative_sentences")
    ndiff = cov.get("diff_claim_sentences")
    rungs = ",".join("%s=%d" % kv for kv in sorted(core.get("rungs", {}).items())) or "none"
    line = ("%s  held=%d failed=%d unresolved=%d malformed=%d  "
            "coverage-floor≈%s (sworn %d / narrative-sentences %s; advisory)  diff-claims≈%s  rungs %s"
            % (core["document_verdict"], c["HELD"], c["FAILED"], c["UNRESOLVED"], c["MALFORMED"],
               share, core["sworn_total"], "n/a" if nsent is None else nsent,
               "n/a" if ndiff is None else ndiff, rungs))
    if core["document_malformed"]:
        line += "  document-MALFORMED: %s" % core["document_malformed"]["reason"]
    # SWORN-HELD is decided by `FAILED == 0 and MALFORMED == 0` and does not consult UNRESOLVED, so
    # a document in which NOTHING was checked carries the same headline as one in which everything
    # held. That conflation is the one this module's own doctrine refuses four lines from the top of
    # the file — "a document that swore nothing is UNSWORN, never 'no failures'" — applied to
    # sworn_total == 0 and not to unresolved == sworn_total.
    #
    # It is author-reachable without forging anything: a manifest rung this verifier does not know
    # makes every span UNRESOLVED with reason `rung_unknown`, BEFORE any receipt id is looked up, so
    # a sentence contradicting its own receipt goes from SWORN-FAILED to SWORN-HELD by changing one
    # string. Naming the verdict differently is a breaking change to a published vocabulary and is
    # the operator's call; saying so out loud on the line a reader actually reads is not.
    # A HELD span whose comparison was against zero says nothing about its receipt. It is honest
    # rounding taken past the point where anything survives, and it is invisible on a line that
    # only counts verdicts.
    # Derived from what the detail already carries — `receipt` and `receipt_rounded` — rather than
    # from a field added to it, because detail is inside the digested core and this must not move it.
    def _erased_span(s):
        if s.get("verdict") != "HELD":
            return False
        d = s.get("detail") or {}
        if "receipt" not in d or "receipt_rounded" not in d:
            return False
        try:
            return not Decimal(d["receipt"]).is_zero() and Decimal(d["receipt_rounded"]).is_zero()
        except (InvalidOperation, ValueError, TypeError):
            return False

    _erased = [s for s in core["spans"] if _erased_span(s)]
    if _erased:
        line += ("\n  WARNING: %d HELD span(s) compared against 0 because the sentence printed no "
                 "fractional digits; they say nothing about the receipt's value" % len(_erased))
    if core["document_verdict"] == "SWORN-HELD" and c["UNRESOLVED"]:
        if c["HELD"] == 0:
            line += ("\n  WARNING: nothing was checked — all %d sworn spans are UNRESOLVED, and "
                     "SWORN-HELD does not mean they held" % c["UNRESOLVED"])
        else:
            line += ("\n  WARNING: %d of %d sworn spans are UNRESOLVED and did not hold; "
                     "SWORN-HELD reports only that none FAILED" % (c["UNRESOLVED"], core["sworn_total"]))
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
    mn.add_argument("--rung", required=True, choices=list(RUNGS),
                    help="L1: a local hook sharing a filesystem with the agent (weak); "
                         "L2: a runner that minted after the turn, which the agent could not write to")
    ma = msub.add_parser("add")
    ma.add_argument("manifest")
    ma.add_argument("--id", required=True)
    ma.add_argument("--file", required=True)
    ma.add_argument("--kind", required=True, choices=sorted(SOURCE_KINDS_EXTERNAL | SOURCE_KINDS_AUTHOR))
    ma.add_argument("--complete", action="store_true")
    ma.add_argument("--note", default=None, help="harness note, e.g. the command whose stdout this is")

    a = ap.parse_args(argv)

    if a.cmd == "canon":
        raw = Path(a.doc).read_bytes()
        mf = Manifest.load(a.manifest) if a.manifest else None
        side = to_sidecar(raw, Path(a.doc).name, a.commit, mf)
        out = Path(a.out) if a.out else Path(a.doc).with_suffix(".sworn.json")
        _write_json_lf(out, side)
        print("canonical: %d spans, text sha256 %s -> %s"
              % (len(side["spans"]), side["document"]["sha256"][:12], out.name))
        # A numeric span carrying two digit-bearing tokens is MALFORMED number_count at verify
        # time, and that is decidable HERE: _number_token reads the span's own inner text and
        # needs no manifest, no repository and no receipt. Saying nothing costs the author a
        # commit — the sidecar names a commit, so a late repair means re-canon and re-swear.
        # Reported, never refused: canon's job is a faithful round trip and verify's is the
        # verdict.
        for _sp in side["spans"]:
            if _sp.get("kind") != "numeric":
                continue
            # Span offsets are BYTE offsets into the UTF-8 encoding — load_sidecar bounds them
            # against len(text.encode("utf-8")) and render splices into the encoded bytes. Slicing
            # the str by them lines up only while the document stays ASCII, and silently shifts as
            # soon as it does not: the first version of this warning read the wrong span on this
            # very file, reporting "no digit-bearing token" for a span carrying two.
            _enc = side["text"].encode("utf-8")
            _inner = _enc[_sp["start"]:_sp["end"]].decode("utf-8", "replace")
            _why, _tok, _seen = _number_token(_inner)
            if _why:
                print("  WARNING @%d: numeric span will be MALFORMED %s at verify — %s"
                      % (_sp.get("start", 0), _why,
                         ("digit-bearing tokens %r" % (_seen,)) if _seen
                         else "no digit-bearing token"))
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
                _write_json_lf(a.out, rec)
                print("receipt %s -> %s" % (rec["digest"][:12], a.out))
            return 0
        rec = json.loads(Path(a.receipt).read_text(encoding="utf-8"))
        res = verify_receipt(rec, raw, side, manifest=mf, tree=tree)
        # VERIFIED answers "does this RECEIPT re-derive", never "did the DOCUMENT hold". Those are
        # one word apart and the second is the one a reader usually wants; printing only the first
        # has already been read as the second in this repository, by its own author. The document's
        # verdict rides on the same line so the two cannot be confused.
        print("%s  digest=%s verdict-reproduces=%s same-build=%s  document=%s"
              % (res["status"], res["digest_match"], res["verdict_reproduces"],
                 res["same_verifier_build"], rec.get("document_verdict", "?")))
        # The exit code stays a function of the RECEIPT alone. A document that honestly reports
        # SWORN-FAILED is a working document; `check` reports and never gates.
        return 0 if res["status"] == "VERIFIED" else 1

    if a.cmd == "manifest":
        p = Path(a.manifest)
        if a.mcmd == "new":
            Manifest(a.harness, a.turn, rung=a.rung).write(p)
            print("minted %s by %s for turn %s at rung %s" % (p.name, a.harness, a.turn, a.rung))
            return 0
        mf = Manifest.load(p)
        e = mf.add(a.id, Path(a.file).read_bytes(), a.kind, a.complete, note=a.note)
        mf.write(p)
        print("added %s %s sha256=%s complete=%s" % (e["id"], a.kind, e["sha256"][:12], e["complete"]))
        return 0
    return 2                                                   # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
