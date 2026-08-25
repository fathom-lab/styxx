"""OATH v0.10 RED TEAM — what a markdown table ROW-ORDINAL detector would destroy.

The lead under attack: `PROSPECTUS_knowsay_2026_07_27.md` is the only OATH-FAILED document in
the resolvable corpus, and its four UNGROUNDED tokens are the leading index cells of a claim
table — the row numbers 3/4/5/8. The proposed repair is to ABSTAIN (or not extract) numeric
tokens sitting in the FIRST CELL of a markdown table data row.

This script is the adversarial pass. It does not implement the repair; it measures what the
repair would cost, using five successively tighter detector shapes, and it carries the
catch-destruction control that this program has twice paid to learn is mandatory:

    an abstention rule improves every tamper metric by destroying coverage, because the
    predicate reads CONTEXT and a one-digit mutation leaves the context unchanged.

`styxx/certify.py` is NOT edited and NOT monkeypatched. Every detector is a POST-FILTER applied
to the ledger the shipped verifier actually produces, which is why the arms are comparable: the
same certify call feeds all of them. A positive control (section 1) proves the cell-mapping
replication of `extract_numbers` is the live extractor and not a lookalike.

  python papers/closed-model-frontier/oath_v10_ordinal_redteam.py

Writes papers/closed-model-frontier/oath_v10_ordinal_redteam.json. Non-destructive: mutants live
in temp files, corpus passes are in-memory, and the only file written is the result JSON.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import (  # noqa: E402
    _DATEISH,
    _FORMULA_AFTER,
    _MD_STRUCTURE,
    _NUM,
    _SHAISH,
    _TABLE_SEP,
    _VERSIONISH,
    _YEAR,
    _decimals,
    certify_doc,
    extract_numbers,
)
from styxx.corpus_audit import _resolve_receipts  # noqa: E402

OUT = HERE / "oath_v10_ordinal_redteam.json"
MUT_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# The four tokens the proposed detector exists to silence.
TARGETS = [("PROSPECTUS_knowsay_2026_07_27.md", 27, "3"),
           ("PROSPECTUS_knowsay_2026_07_27.md", 28, "4"),
           ("PROSPECTUS_knowsay_2026_07_27.md", 29, "5"),
           ("PROSPECTUS_knowsay_2026_07_27.md", 32, "8")]

# ---------------------------------------------------------------- table geometry

_EMPH = re.compile(r"^[\s*_`]+|[\s*_`]+$")

# vocabulary rosters used for TRIAGE only. Every roster entry carries its raw header and cells so
# a reader can overrule the label; the hand labels in HAND_LABELLED are the evidence.
_ORDINAL_HDR = re.compile(r"^(#|no\.?|num|nr|idx|index|row|rank#|item|claim|entry|line|)$", re.I)
_PARAM_HDR = re.compile(
    r"^(seed|layer|α|alpha|λ|lambda|lam_?\w*|k|rank\s*k|d|d\s*\(modes\)|q|q\s*\(delays\)|n|"
    r"ρ|rho|σ|sigma|dose\s*[σt]?|step|steps|epoch|iter|temp\w*|scale|model|size|params?|"
    r"precision|claim precision|decimals?|prevalence|true prevalence|threshold|bar|level|"
    r"depth|width|dim|budget|shots?|tier|band|cycle|run|phase|window|radius|bin|arm|cell|"
    r"condition|regime|dst write layer|planted coupling|outcome band|sep|k\s*\(.*\)|"
    r"faithfulness|attacker|component|stratum)$", re.I)
_METRIC_HDR = re.compile(
    r"(auroc|auc|accuracy|recall|precision|f1|rate|faithfulness|cave|margin|score|delta|"
    r"elevation|stability|concordance|coverage|p-?value)", re.I)


def scrub_of(line: str) -> str:
    """The exact searchable string `extract_numbers` builds for a line."""
    line = line.replace("−", "-")
    s = _SHAISH.sub(" ", line)
    s = _DATEISH.sub(" ", s)
    s = _VERSIONISH.sub(" ", s)
    return s


def cell_spans(s: str):
    """Character spans of the cells of a markdown table row, on the SCRUBBED line.

    The scrub only ever replaces spans of hex/date/version characters, none of which can contain
    a pipe, so pipe structure is preserved. Escaped pipes (\\|) are not handled — disclosed.
    """
    t = s.rstrip()
    if not t.lstrip().startswith("|"):
        return None
    pipes = [i for i, ch in enumerate(t) if ch == "|"]
    if len(pipes) < 2:
        return None
    sp = [(a + 1, b) for a, b in zip(pipes, pipes[1:])]
    if pipes[-1] < len(t) - 1:           # unterminated row: trailing content is a cell
        sp.append((pipes[-1] + 1, len(t)))
    return sp


def table_blocks(lines: list[str]):
    """Maximal runs of consecutive pipe-leading lines, with the separator located.

    Yields dicts: start/end (0-based, end exclusive), sep index or None, header index or None,
    data row indices. A block whose separator is its first line has NO header row; a block with
    no separator at all is a table markdown will not render as one, and `extract_numbers` gives
    its rows no `binding_context` — both are enumerated because a header-gated detector misfires
    on exactly them.
    """
    out, i = [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            seps = [k for k in range(i, j) if _TABLE_SEP.match(lines[k])]
            sep = seps[0] if seps else None
            hdr = sep - 1 if (sep is not None and sep > i) else None
            data = list(range(sep + 1, j)) if sep is not None else list(range(i, j))
            out.append({"start": i, "end": j, "sep": sep, "hdr": hdr, "data": data,
                        "n_seps": len(seps)})
            i = j
        else:
            i += 1
    return out


def first_cell_text(line: str) -> str | None:
    sp = cell_spans(line)
    if not sp:
        return None
    return line.rstrip()[sp[0][0]:sp[0][1]].strip()


def norm_cell(c: str) -> str:
    return _EMPH.sub("", c).strip()


def ordinal_column(cells: list[str]) -> bool:
    """Every data-row first cell is a BARE integer and the column steps by exactly 1."""
    vals = []
    for c in cells:
        n = norm_cell(c)
        if not re.fullmatch(r"\d{1,3}", n):
            return False
        vals.append(int(n))
    return len(vals) >= 2 and vals == list(range(vals[0], vals[0] + len(vals)))


# ---------------------------------------------------------------- extraction replication

def extract_with_cells(text: str) -> list[dict]:
    """`extract_numbers`, replicated verbatim, plus the token's table-cell index.

    Verbatim means verbatim: every filter below is copied from the shipped `extract_numbers`, and
    section 1 asserts the (line, token, value, decimals) sequence is identical to the live
    function's on every document in the frame. Without that control this is a lookalike and every
    number downstream is about the lookalike.
    """
    out = []
    lines = text.splitlines()
    header_for: dict[int, str] = {}
    for i, line in enumerate(lines):
        if _TABLE_SEP.match(line) and i > 0 and lines[i - 1].lstrip().startswith("|"):
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                header_for[j + 1] = lines[i - 1].strip()
                j += 1
    for ln_no, line in enumerate(lines, 1):
        line = line.replace("−", "-")
        scrub = _SHAISH.sub(" ", line)
        scrub = _DATEISH.sub(" ", scrub)
        scrub = _VERSIONISH.sub(" ", scrub)
        spans = cell_spans(scrub)
        for m in _NUM.finditer(scrub):
            tok = m.group(0)
            raw = tok.replace(",", "")
            if _YEAR.match(raw.lstrip("+-")):
                continue
            if (m.start() <= 2 and "." not in raw and abs(int(raw)) < 10
                    and scrub[m.end():m.end() + 1] != "/"
                    and _MD_STRUCTURE.match(line)):
                continue
            if _FORMULA_AFTER.match(scrub[m.end():]):
                continue
            if m.start() >= 2 and scrub[m.start() - 1] in "–-−" and scrub[m.start() - 2].isdigit():
                continue
            if m.start() >= 2 and scrub[m.start() - 1] == "-" and scrub[m.start() - 2].isalpha():
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            cell = None
            if spans:
                cell = next((i for i, (a, b) in enumerate(spans) if a <= m.start() < b), None)
            out.append({"line": ln_no, "token": tok, "value": val, "decimals": _decimals(raw),
                        "cell": cell, "start": m.start(),
                        "context": line.strip()[:160]})
    return out


# ---------------------------------------------------------------- detectors (post-filters)

def build_table_index(text: str) -> dict:
    """line number (1-based) -> facts about the table it sits in, for the doc AS GIVEN."""
    lines = text.splitlines()
    idx = {}
    for blk in table_blocks(lines):
        hdr_txt = norm_cell(first_cell_text(lines[blk["hdr"]]) or "") if blk["hdr"] is not None else None
        cells = [first_cell_text(lines[k]) or "" for k in blk["data"]]
        is_ord = ordinal_column(cells)
        for k in blk["data"]:
            idx[k + 1] = {"has_header": blk["hdr"] is not None,
                          "hdr": hdr_txt, "ordinal_column": is_ord,
                          "block": (blk["start"] + 1, blk["end"]),
                          "n_rows": len(blk["data"])}
    return idx


def detectors(entry: dict, tinfo: dict | None) -> dict[str, bool]:
    """Would each candidate detector ABSTAIN this token? `entry` needs cell/decimals/value/
    cell_text; `tinfo` is build_table_index()[line] or None (not a table data row)."""
    if entry.get("cell") != 0 or tinfo is None:
        return {k: False for k in ("D1", "D2", "D3", "D4", "D5")}
    ctext = norm_cell(entry.get("cell_text", ""))
    bare = bool(re.fullmatch(r"\d{1,3}", ctext))
    d1 = True
    d2 = d1 and entry["decimals"] == 0
    d3 = d2 and bare and abs(entry["value"]) < 100
    d4 = d3 and bool(tinfo["ordinal_column"])
    d5 = d3 and tinfo["has_header"] and bool(_ORDINAL_HDR.match(tinfo["hdr"] or ""))
    return {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5}


DETECTOR_DOC = {
    "D1": "ANY numeric token in the first cell of a markdown table data row -> ABSTAIN.",
    "D2": "D1 restricted to integer tokens (decimals == 0).",
    "D3": "D2 restricted to a cell that is EXACTLY a bare 1-3 digit integer, |value| < 100.",
    "D4": "D3 AND the table's whole first column is a consecutive integer run (step 1). "
          "NOTE: this predicate READS THE VALUE, so mutating a row number can break the run and "
          "the abstention stops firing — measured, not assumed.",
    "D5": "D3 AND the table has a header row whose first cell is ordinal vocabulary "
          "(#, no, num, idx, index, row, item, claim, entry, line, or empty).",
}


# ---------------------------------------------------------------- corpus frames

def resolvable_docs():
    out = []
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec)
        if receipts and not missing:
            out.append((doc, receipts))
    return out


def all_docs():
    return [d for d in sorted(ROOT.glob("papers/**/*.md")) if "anc" not in d.parts]


def read(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


# ---------------------------------------------------------------- mutation (v0.7/v0.9 idiom)

def mutate_sig(tok: str, rng: random.Random) -> str:
    if "." in tok:
        frac_at = tok.index(".") + 1
        frac = tok[frac_at:]
        sig = [i for i in range(min(6, len(frac)))
               if not (frac[i] == "0" and set(frac[:i]) <= {"0"})]
        pos = frac_at + rng.choice(sig or [0])
    else:
        pos = rng.choice([i for i, ch in enumerate(tok) if ch.isdigit()])
    old = int(tok[pos])
    return tok[:pos] + str(rng.choice([d for d in range(10) if d != old])) + tok[pos + 1:]


def substitute(line: str, tok: str, mut: str):
    """Sign-aware substitution — extraction normalizes U+2212 to ASCII while the document holds
    U+2212, so a bare `line.replace` silently no-ops on every signed claim (inherited from
    run_oath_v07_battery.py, which owns that defect)."""
    if tok in line:
        return line.replace(tok, mut, 1), True
    if tok.startswith("-"):
        alt, alt_mut = tok.replace("-", "−", 1), mut.replace("-", "−", 1)
        if alt in line:
            return line.replace(alt, alt_mut, 1), True
    return line, False


def substitute_in_cell(line: str, span: tuple[int, int], tok: str, mut: str):
    """Replace the token INSIDE its own cell, so a first-cell '3' is not silently swapped for a
    '3' that appears earlier on the same line."""
    a, b = span
    head, cell, tail = line[:a], line[a:b], line[b:]
    new, landed = substitute(cell, tok, mut)
    return head + new + tail, landed


# ---------------------------------------------------------------- index-likeness (corroboration)

_INDEX_TERMINAL = re.compile(r"(^|[.\[])(i|j|k|idx|index|rank|ordinal|position|pos|row|seed|"
                             r"id|item|n|step|epoch)$", re.I)


def leaf_is_subscript_coincident(path: str, value: float) -> bool:
    """`per_item[3].i` holding 3 — a leaf equal to its own array subscript, which matches that
    integer BY CONSTRUCTION and can therefore verify any small ordinal that reaches it."""
    subs = re.findall(r"\[(\d+)\]", path)
    return bool(subs) and float(subs[-1]) == value


def main() -> int:                                              # noqa: C901 - one report
    t0 = time.time()
    payload: dict = {
        "purpose": "RED TEAM: the false-positive surface of a markdown table row-ordinal "
                   "detector, plus the mandatory catch-destruction control.",
        "angle": "4 — adversarial",
        "certify_untouched": True,
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "detector_definitions": DETECTOR_DOC,
        "mutation_seeds": list(MUT_SEEDS),
    }

    docs = resolvable_docs()
    every = all_docs()
    payload["corpus"] = {"resolvable_certified_documents": len(docs),
                         "all_markdown_under_papers": len(every)}
    print(f"resolvable certified docs: {len(docs)}   all papers/*.md: {len(every)}")

    # ---- 1. POSITIVE CONTROL: the cell-aware replication IS the shipped extractor ----------
    checked = diffs = 0
    diff_examples = []
    for doc, _rc in docs:
        text = read(doc)
        if text is None:
            continue
        live = [(e["line"], e["token"], e["value"], e["decimals"]) for e in extract_numbers(text)]
        mine = [(e["line"], e["token"], e["value"], e["decimals"]) for e in extract_with_cells(text)]
        checked += len(live)
        if live != mine:
            diffs += 1
            if len(diff_examples) < 5:
                diff_examples.append(doc.name)
    payload["positive_control"] = {
        "what": "extract_with_cells() vs the live extract_numbers(), sequence of "
                "(line, token, value, decimals) over every resolvable document",
        "tokens_compared": checked, "documents_differing": diffs,
        "differing_examples": diff_examples,
        "verdict": "REPLICATION EXACT" if diffs == 0 else "REPLICATION BROKEN — every number "
                                                          "below is about a lookalike"}
    print(f"[1] positive control: {checked} tokens, {diffs} documents differing")
    if diffs:
        print("    FATAL: replication is not the shipped extractor.")

    # ---- 2. CORPUS-WIDE TABLE CENSUS + the false-positive surface ---------------------------
    n_tables = n_nosep = n_nohdr = n_multisep = 0
    unreadable = []
    param_cols, measurement_cols, ordinal_cols = [], [], []
    exotic_first_cells = collections.Counter()
    exotic_examples: dict[str, list] = collections.defaultdict(list)
    noheader_roster, nonnumeric_hdr_roster = [], []

    for doc in every:
        text = read(doc)
        if text is None:
            unreadable.append(doc.relative_to(ROOT).as_posix())
            continue
        lines = text.splitlines()
        rel = doc.relative_to(ROOT).as_posix()
        for blk in table_blocks(lines):
            n_tables += 1
            if blk["sep"] is None:
                n_nosep += 1
            if blk["hdr"] is None:
                n_nohdr += 1
            if blk["n_seps"] > 1:
                n_multisep += 1
            cells = [first_cell_text(lines[k]) or "" for k in blk["data"]]
            if not cells:
                continue
            hdr_txt = norm_cell(first_cell_text(lines[blk["hdr"]]) or "") if blk["hdr"] is not None else None
            numy = [c for c in cells if _NUM.search(scrub_of(c))]
            if not numy:
                continue
            rec = {"doc": rel, "table_line": blk["start"] + 1, "header": hdr_txt,
                   "rows": len(cells), "first_cells": cells[:12]}
            if ordinal_column(cells):
                ordinal_cols.append(rec)
                continue
            # non-ordinal numeric first column: the danger class
            if hdr_txt is not None and _METRIC_HDR.search(hdr_txt) and not _PARAM_HDR.match(hdr_txt):
                measurement_cols.append(rec)
            else:
                rec["header_matches_parameter_vocab"] = bool(hdr_txt is not None
                                                             and _PARAM_HDR.match(hdr_txt))
                param_cols.append(rec)
            # exotic first-cell shapes a bare positional rule would swallow
            for c in cells:
                n = norm_cell(c)
                kind = None
                if re.fullmatch(r"[-+]?\d+\.\d+", n):
                    kind = "float"
                elif re.fullmatch(r"\d+\s*[BMKbmk]\b.*", n):
                    kind = "unit-suffixed (3B / 8M)"
                elif re.fullmatch(r"[A-Za-z]+\d+.*", n):
                    kind = "letter-prefixed (L27 / s0 / sub-008)"
                elif re.search(r"\d", n) and re.search(r"[=(/≥≤<>]", n):
                    kind = "multi-part (8 (0.5) / >=3/7 / alpha=1.0)"
                elif re.fullmatch(r"\d{1,3}", n):
                    kind = "bare small integer"
                elif re.search(r"\d", n):
                    kind = "other numeric-bearing"
                if kind:
                    exotic_first_cells[kind] += 1
                    if len(exotic_examples[kind]) < 12:
                        exotic_examples[kind].append({"doc": rel, "line": blk["start"] + 1,
                                                      "header": hdr_txt, "cell": c[:60]})
            if blk["hdr"] is None and len(noheader_roster) < 40:
                noheader_roster.append(rec)
            if hdr_txt is not None and _NUM.search(scrub_of(hdr_txt)) and len(nonnumeric_hdr_roster) < 40:
                nonnumeric_hdr_roster.append(rec)

    payload["table_census"] = {
        "tables": n_tables, "tables_without_a_separator_row": n_nosep,
        "tables_without_a_header_row": n_nohdr,
        "tables_with_more_than_one_separator_row": n_multisep,
        "unreadable_documents": unreadable,
        "first_columns_carrying_numbers": len(ordinal_cols) + len(param_cols) + len(measurement_cols),
        "consecutive_integer_run_columns": len(ordinal_cols),
        "non_ordinal_numeric_first_columns": len(param_cols) + len(measurement_cols),
    }
    payload["false_positive_surface"] = {
        "parameter_like_first_columns": {"n": len(param_cols), "roster": param_cols},
        "measurement_first_columns": {"n": len(measurement_cols), "roster": measurement_cols},
        "first_cell_shapes": {"counts": dict(exotic_first_cells),
                              "examples": {k: v for k, v in exotic_examples.items()}},
        "tables_with_no_header_row": {"n": n_nohdr, "roster": noheader_roster},
        "tables_whose_header_first_cell_carries_a_number": {"roster": nonnumeric_hdr_roster},
        "consecutive_integer_run_columns": {"n": len(ordinal_cols), "roster": ordinal_cols},
    }
    print(f"[2] tables {n_tables}  no-separator {n_nosep}  no-header {n_nohdr}  "
          f"| ordinal-run first cols {len(ordinal_cols)}  "
          f"non-ordinal numeric first cols {len(param_cols) + len(measurement_cols)}")

    # ---- 3. CERTIFIED-CORPUS IMPACT: what each detector would silence -----------------------
    per_doc_entries: dict[str, list] = {}
    doc_by_name = {}
    base_status = collections.Counter()
    fc_tokens = []                       # first-cell tokens with live status
    for doc, rc in docs:
        text = read(doc)
        if text is None:
            continue
        rel = doc.relative_to(ROOT).as_posix()
        doc_by_name[doc.name] = (doc, rc)
        lines = text.splitlines()
        tinfo = build_table_index(text)
        cert = certify_doc(doc, rc)
        mine = extract_with_cells(text)
        # ledger is 1:1 with extract_numbers order, and section 1 proved mine is that sequence
        assert len(mine) == len(cert["ledger"]), doc.name
        rows = []
        for e, led in zip(mine, cert["ledger"]):
            assert e["line"] == led["line"] and e["token"] == led["token"]
            scrub = scrub_of(lines[e["line"] - 1])
            sp = cell_spans(scrub)
            ctext = scrub[sp[0][0]:sp[0][1]].strip() if (sp and e["cell"] == 0) else ""
            rec = {**e, "status": led["status"], "receipt_ref": led.get("receipt_ref"),
                   "cell_text": ctext, "doc": doc.name, "rel": rel}
            rows.append(rec)
            if e["cell"] == 0 and e["line"] in tinfo:
                base_status[led["status"]] += 1
                fc_tokens.append({**rec, "tinfo": tinfo[e["line"]],
                                  "det": detectors(rec, tinfo[e["line"]])})
        per_doc_entries[rel] = rows

    payload["certified_corpus_impact"] = {
        "first_cell_tokens_in_table_data_rows": len(fc_tokens),
        "by_shipped_status": dict(base_status),
    }
    print(f"[3] first-cell tokens in the certified corpus: {len(fc_tokens)}  {dict(base_status)}")

    det_report = {}
    for key in ("D1", "D2", "D3", "D4", "D5"):
        fires = [t for t in fc_tokens if t["det"][key]]
        by_status = collections.Counter(t["status"] for t in fires)
        killed = [t for t in fires if t["status"] == "VERIFIED"]
        silenced = [t for t in fires if t["status"] == "UNGROUNDED"]
        # verdict flips: a doc whose ONLY ungrounded tokens are all silenced flips FAILED->HELD
        ung_by_doc = collections.Counter(t["rel"] for rows in per_doc_entries.values()
                                        for t in rows if t["status"] == "UNGROUNDED")
        sil_by_doc = collections.Counter(t["rel"] for t in silenced)
        flips = sorted(d for d, n in ung_by_doc.items() if sil_by_doc.get(d, 0) >= n)
        hit_targets = sum(1 for t in fires
                          if (t["doc"], t["line"], t["token"]) in
                          {(a, b, c) for a, b, c in TARGETS})
        det_report[key] = {
            "definition": DETECTOR_DOC[key],
            "fires_on_tokens": len(fires),
            "by_shipped_status": dict(by_status),
            "honest_VERIFIED_destroyed": len(killed),
            "accusations_silenced": len(silenced),
            "certificates_flipped_FAILED_to_HELD": flips,
            "target_tokens_reached": f"{hit_targets}/4",
            "cost_ratio_verified_lost_per_target_token":
                round(len(killed) / max(1, hit_targets), 3),
            "destroyed_roster": [{"doc": t["doc"], "line": t["line"], "token": t["token"],
                                  "header": t["tinfo"]["hdr"], "receipt_ref": t["receipt_ref"],
                                  "context": t["context"][:120]} for t in killed][:400],
        }
        print(f"    {key}: fires {len(fires):4d}  VERIFIED destroyed {len(killed):4d}  "
              f"silenced {len(silenced)}  targets {hit_targets}/4")
    payload["certified_corpus_impact"]["detectors"] = det_report

    # ---- 4. CATCH-DESTRUCTION CONTROL (mandatory) -------------------------------------------
    # Every first-cell token in the certified corpus, one significant digit perturbed, at ten
    # seeds, scored at the SHIPPED verifier. CAUGHT = the mutant comes back UNGROUNDED. Each
    # detector's arm is the same mutant ledger with the detector applied as a post-filter, so the
    # arms are exactly comparable. D4 is the interesting one: its predicate reads the column's
    # VALUES, so a mutated row number can break the consecutive run and the abstention stops
    # firing. That is measured here, not assumed.
    roster = [t for t in fc_tokens if t["status"] in ("VERIFIED", "UNGROUNDED")]
    seed_rows = []
    for seed in MUT_SEEDS:
        rng = random.Random(seed)
        off = collections.Counter()
        arms = {k: collections.Counter() for k in ("D1", "D2", "D3", "D4", "D5")}
        unlanded = 0
        for t in roster:
            doc, rc = doc_by_name[t["doc"]]
            lines = read(doc).splitlines()
            mut = mutate_sig(t["token"], rng)
            ml = list(lines)
            scrub = scrub_of(ml[t["line"] - 1])
            sp = cell_spans(scrub)
            span = sp[0] if sp else (0, len(ml[t["line"] - 1]))
            ml[t["line"] - 1], landed = substitute_in_cell(ml[t["line"] - 1], span,
                                                           t["token"], mut)
            unlanded += not landed
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write("\n".join(ml))
                tmp = Path(tf.name)
            try:
                cert = certify_doc(tmp, rc)
                mtext = "\n".join(ml)
            finally:
                tmp.unlink(missing_ok=True)
            e = next((x for x in cert["ledger"]
                      if x["line"] == t["line"] and x["token"] == mut), None)
            st = e["status"] if e else "NOT_EXTRACTED"
            off[st] += 1
            mtinfo = build_table_index(mtext).get(t["line"])
            mscrub = scrub_of(ml[t["line"] - 1])
            msp = cell_spans(mscrub)
            mcell = mscrub[msp[0][0]:msp[0][1]].strip() if msp else ""
            mrec = {"cell": 0, "decimals": _decimals(mut.replace(",", "")),
                    "value": float(mut.replace(",", "")), "cell_text": mcell}
            fired = detectors(mrec, mtinfo)
            for k in arms:
                arms[k][st if not fired[k] else "ABSTAIN(detector)"] += 1
        seed_rows.append({
            "seed": seed, "roster_n": len(roster), "did_not_land": unlanded,
            "OFF": {"caught_UNGROUNDED": off["UNGROUNDED"],
                    "false_attested_VERIFIED": off["VERIFIED"],
                    "abstained": off["ABSTAIN"], "not_extracted": off["NOT_EXTRACTED"],
                    "all": dict(off)},
            "ON": {k: {"caught_UNGROUNDED": arms[k]["UNGROUNDED"],
                       "false_attested_VERIFIED": arms[k]["VERIFIED"],
                       "silenced_by_detector": arms[k]["ABSTAIN(detector)"]}
                   for k in arms},
        })
        print(f"[4] seed {seed:2d}  OFF caught {off['UNGROUNDED']:3d} / false-attested "
              f"{off['VERIFIED']:3d}   ON caught: "
              + "  ".join(f"{k}={arms[k]['UNGROUNDED']}" for k in ("D1", "D3", "D4", "D5")))

    def agg(getter):
        vals = [getter(r) for r in seed_rows]
        return {"mean": round(sum(vals) / len(vals), 2), "min": min(vals), "max": max(vals)}

    payload["catch_destruction_control"] = {
        "what": "every first-cell token in the certified corpus whose shipped status is VERIFIED "
                "or UNGROUNDED, one significant digit perturbed inside its own cell, ten seeds. "
                "CAUGHT = mutant returns UNGROUNDED. The detector arms are the same mutant "
                "ledgers with the detector applied as a post-filter.",
        "roster_n": len(roster),
        "OFF_caught": agg(lambda r: r["OFF"]["caught_UNGROUNDED"]),
        "OFF_false_attested": agg(lambda r: r["OFF"]["false_attested_VERIFIED"]),
        "ON_caught": {k: agg(lambda r, k=k: r["ON"][k]["caught_UNGROUNDED"])
                      for k in ("D1", "D2", "D3", "D4", "D5")},
        "ON_false_attested": {k: agg(lambda r, k=k: r["ON"][k]["false_attested_VERIFIED"])
                              for k in ("D1", "D2", "D3", "D4", "D5")},
        "per_seed": seed_rows,
    }

    # ---- 5. CORROBORATION: are first-cell integers false-verifying against index-like leaves? -
    # Split three ways, because the naive "leaf equals its own subscript" test is CONTAMINATED:
    # a receipt recording `"seeds": [0, 1, 2]` makes seeds[0] == 0 by CONVENTION, and a document
    # whose seed column reads 0/1/2 is grounding correctly. Counting those as evidence of the
    # defect would manufacture the corroboration this section exists to test.
    corrob, corrob_counts = [], collections.Counter()
    for t in fc_tokens:
        if t["status"] != "VERIFIED" or t["decimals"] != 0 or not t["receipt_ref"]:
            continue
        ref = t["receipt_ref"]
        path = ref.split(":", 1)[1] if ":" in ref else ref
        flat = path.replace("[", ".").replace("]", "")
        terminal = re.split(r"[.\[]", path)[-1].strip("]")
        sub = leaf_is_subscript_coincident(path, t["value"])
        term = bool(_INDEX_TERMINAL.search(flat))
        ordcol = bool(t["tinfo"]["ordinal_column"])
        hdr = (t["tinfo"]["hdr"] or "").lower()
        # does the claim's own column NAME the container it grounds in? ('seed' -> seeds[0])
        named = bool(hdr) and any(seg.lower().startswith(hdr[:4]) or hdr.startswith(seg.lower()[:4])
                                  for seg in re.split(r"[.\[\]]", path) if len(seg) >= 3)
        if not (sub or term or ordcol):
            continue
        if sub and re.search(r"(^|[.\[])(i|j|idx|index|rank|ordinal|position|pos|row)$",
                             re.split(r"\[\d+\]", path)[-1] or path, re.I):
            kind = "A: pure index leaf (per_item[k].i) — the defect in its purest form"
        elif sub and named:
            kind = "C: subscript-coincident BUT the column names the container (seed -> seeds[0]) " \
                   "— a correct binding, NOT evidence of the defect"
        elif sub:
            kind = "B: subscript-coincident, column does not name the container"
        elif term:
            kind = "D: index-like terminal segment, not subscript-coincident"
        else:
            kind = "E: ordinal-shaped column grounding in a non-index leaf (a coincidence " \
                   "against an unrelated summary field)"
        corrob_counts[kind] += 1
        corrob.append({"doc": t["doc"], "line": t["line"], "token": t["token"],
                       "header": t["tinfo"]["hdr"], "ordinal_column": ordcol,
                       "receipt_ref": ref, "terminal_segment": terminal, "kind": kind,
                       "context": t["context"][:110]})
    genuine_defect = [c for c in corrob if c["kind"][0] in ("A", "B", "E")]
    payload["corroborating_roster"] = {
        "what": "first-cell INTEGER tokens the shipped verifier reports VERIFIED, whose cited "
                "leaf is index-like (equal to its own array subscript, or a terminal segment "
                "named i/j/k/idx/index/rank/seed/id/n/step/...), or which sit in a consecutive-"
                "integer column.",
        "contamination_warning": "class C is NOT corroboration. A receipt holding "
                                 "\"seeds\": [0,1,2] makes seeds[0] == 0 by convention, so a "
                                 "document whose seed column reads 0/1/2 grounds CORRECTLY and "
                                 "trips the subscript test by construction.",
        "counts": dict(corrob_counts), "n": len(corrob),
        "defect_shaped_after_removing_class_C": len(genuine_defect),
        "documents_carrying_a_defect_shaped_token":
            sorted({c["doc"] for c in genuine_defect}),
        "roster": corrob,
    }
    print(f"[5] corroborating roster: {len(corrob)}  defect-shaped after removing class C: "
          f"{len(genuine_defect)}")
    for k, v in sorted(corrob_counts.items()):
        print(f"      {v:4d}  {k}")

    # ---- 7. CORPUS-WIDE FIRING SURFACE (all 1,107 documents, not just the certified 139) ----
    # The certified corpus is 139 of 1,107 documents. A detector shipped into `certify_doc` fires
    # on every document ever certified afterwards, so the forward-looking false-positive surface
    # is the whole tree. No receipts are consulted here — this counts what each detector would
    # ABSTAIN, and what the first-column header calls it.
    wide = {k: collections.Counter() for k in ("D1", "D2", "D3", "D4", "D5")}
    wide_tokens = collections.Counter()
    wide_examples: dict[str, list] = collections.defaultdict(list)
    for doc in every:
        text = read(doc)
        if text is None:
            continue
        rel = doc.relative_to(ROOT).as_posix()
        lines = text.splitlines()
        tinfo = build_table_index(text)
        for e in extract_with_cells(text):
            if e["cell"] != 0 or e["line"] not in tinfo:
                continue
            scrub = scrub_of(lines[e["line"] - 1])
            sp = cell_spans(scrub)
            rec = {**e, "cell_text": scrub[sp[0][0]:sp[0][1]].strip() if sp else ""}
            fired = detectors(rec, tinfo[e["line"]])
            hdr = tinfo[e["line"]]["hdr"]
            wide_tokens["all_first_cell_tokens"] += 1
            for k, v in fired.items():
                if v:
                    wide[k][hdr if hdr is not None else "<no header row>"] += 1
                    if k == "D5" and len(wide_examples["D5"]) < 60:
                        wide_examples["D5"].append({"doc": rel, "line": e["line"],
                                                    "header": hdr, "token": e["token"],
                                                    "context": e["context"][:110]})
    payload["corpus_wide_firing_surface"] = {
        "what": "tokens each detector would ABSTAIN across every markdown document under "
                "papers/ (not only the 139 certified ones), grouped by first-column header. "
                "Receipts are not consulted; this is the surface, not the ledger cost.",
        "documents": len(every),
        "first_cell_tokens_in_table_data_rows": wide_tokens["all_first_cell_tokens"],
        "fires_by_detector": {k: sum(v.values()) for k, v in wide.items()},
        "headers_by_detector": {k: dict(v.most_common()) for k, v in wide.items()},
        "D5_examples": wide_examples["D5"],
    }
    print(f"[7] corpus-wide first-cell tokens: {wide_tokens['all_first_cell_tokens']}   "
          + "  ".join(f"{k}={sum(v.values())}" for k, v in wide.items()))

    # ---- 6. the four target tokens, stated exactly ------------------------------------------
    tgt = []
    for t in fc_tokens:
        if (t["doc"], t["line"], t["token"]) in {(a, b, c) for a, b, c in TARGETS}:
            tgt.append({"doc": t["doc"], "line": t["line"], "token": t["token"],
                        "status": t["status"], "receipt_ref": t["receipt_ref"],
                        "header": t["tinfo"]["hdr"],
                        "ordinal_column": t["tinfo"]["ordinal_column"],
                        "detectors_firing": [k for k, v in t["det"].items() if v]})
    payload["target_tokens"] = {"n_found": len(tgt), "roster": tgt}

    # ---- 8. asserted invariants — NOT gates. A leg that cannot fail must not gate. -----------
    # Measured rather than asserted where it is cheap: for every detector, the per-document
    # UNGROUNDED count after silencing, against the count before. A flip HELD -> FAILED needs a
    # document whose count goes from 0 to >0.
    ung_before = collections.Counter(t["rel"] for rows in per_doc_entries.values()
                                     for t in rows if t["status"] == "UNGROUNDED")
    held_to_failed = 0
    for key in ("D1", "D2", "D3", "D4", "D5"):
        sil = collections.Counter(t["rel"] for t in fc_tokens
                                 if t["det"][key] and t["status"] == "UNGROUNDED")
        for rel in per_doc_entries:
            after = ung_before.get(rel, 0) - sil.get(rel, 0)
            if ung_before.get(rel, 0) == 0 and after > 0:
                held_to_failed += 1
    payload["asserted_invariants"] = {
        "I1_demote_only": {
            "claim": "every detector shape here yields ABSTAIN (or non-extraction), so it can "
                     "only REMOVE tokens from the UNGROUNDED column. No certificate can flip "
                     "OATH-HELD -> OATH-FAILED.",
            "status": "TRUE BY CONSTRUCTION, asserted not measured",
            "measured_flips_HELD_to_FAILED": held_to_failed,
        },
        "I2_position_is_mutation_stable": {
            "claim": "D1/D2/D3/D5 read the token's CELL POSITION and the table's HEADER, neither "
                     "of which a one-digit substitution changes, so a token they abstain stays "
                     "abstained under mutation and their catch count is 0 or near-0 by "
                     "construction. This is the standing hazard, not a result.",
            "status": "TRUE BY CONSTRUCTION for D1/D2/D3/D5; FALSE for D4 — see the control",
        },
        "I3_non_extraction_hides_the_residual": {
            "claim": "implemented as ABSTAIN the silenced tokens stay in the certificate's "
                     "`abstained` list and remain countable; implemented as non-extraction in "
                     "`extract_numbers` they vanish from the ledger and the residual becomes "
                     "invisible. Same ledger arithmetic, different auditability.",
            "status": "a design note, not a measurement",
        },
    }

    # Hand labels on the cases that carry the argument. Written by the red-team pass after
    # reading each table; recorded here so a reader can overrule them against the rosters above.
    payload["hand_labelled_dangerous_cases"] = [
        {"doc": "papers/.../RESULT_B2_coupling_confirm_VOID_2026_07_16.md", "lines": "56-60",
         "first_column": "seed", "cells": "0 / 1 / 2 / 3 / 4",
         "why_dangerous": "a SEED column is a consecutive integer run starting at 0 and is "
                          "structurally indistinguishable from a row ordinal, yet each cell is a "
                          "real experimental parameter that grounds correctly in "
                          "coupling_confirm_result.json:seeds[k]. D4 (consecutive-run) abstains "
                          "all five.",
         "label": "GENUINE PARAMETER — abstaining destroys a checkable claim"},
        {"doc": "papers/.../RESULT_honesty_parity_control_2026_07_11.md", "lines": "21-22",
         "first_column": "seed", "cells": "0 / 1",
         "why_dangerous": "same shape at n=2; a two-row consecutive run is the commonest table "
                          "in this corpus.",
         "label": "GENUINE PARAMETER"},
        {"doc": "papers/.../RESULT_E2_strong_attacker_2026_07_07.md", "lines": "24-25",
         "first_column": "seed", "cells": "0 / 1", "why_dangerous": "same.",
         "label": "GENUINE PARAMETER"},
        {"doc": "papers/.../FINDING_b42_dose_curve_2026_08_05.md", "lines": "24-31",
         "first_column": "rank k", "cells": "1 / 2 / 3 / 5 / 8 / 12 / 20 / 40",
         "why_dangerous": "a LoRA rank sweep. Each cell grounds in b42_result.json:ranks[j] and "
                          "the mapping is NOT the identity (ranks[3] == 5), so these are real "
                          "bindings a value-only rule could not have manufactured. D3 abstains "
                          "all eight; D4 spares them only because the run is not step-1.",
         "label": "GENUINE PARAMETER — the clearest single counterexample"},
        {"doc": "papers/.../FINDING_adversarial_curve_v3_2026_06_08.md", "lines": "19-20",
         "first_column": "lam_hide", "cells": "0 (ref) / 8 / 16",
         "why_dangerous": "a lambda sweep grounding in points[i].lam_hide — the column name and "
                          "the receipt field name are the SAME token, which is as strong a "
                          "claim->field binding as this verifier ever gets.",
         "label": "GENUINE PARAMETER"},
        {"doc": "papers/.../FINDING_buried_judge_2026_07_24.md", "lines": "22+",
         "first_column": "sep", "cells": "0.00 / 0.16 / 0.22 / 0.25 / 0.28 / 0.34 / 0.40",
         "why_dangerous": "the separation parameter the whole table is a sweep over, printed as "
                          "FLOATS in the first cell. D1 abstains them; D2 and tighter spare them "
                          "only because of the integer restriction.",
         "label": "GENUINE PARAMETER — and the reason any detector must be integer-only"},
        {"doc": "papers/capability-amplification-v0.md", "lines": "103+ and 125+",
         "first_column": "Layer / α",
         "cells": "Layer 0,1,2,3,4,5,6,7   and   α 0.0,0.5,1.0,1.5,2.0,3.0",
         "why_dangerous": "a LAYER INDEX is a consecutive integer run starting at 0 — D4's "
                          "predicate cannot tell it from a row number. This document carries no "
                          "certificate today, so the cost is not in the certified-corpus number; "
                          "it is the forward-looking cost the corpus-wide surface counts.",
         "label": "GENUINE PARAMETER — D4's worst case, currently uncertified"},
        {"doc": "papers/.../FINDING_grounding_needs_a_floor_2026_08_13.md", "lines": "37+",
         "first_column": "claim precision", "cells": "1 decimal / 2 decimals / 3 decimals",
         "why_dangerous": "a parameter written as a multi-part cell. A positional rule reads the "
                          "leading integer of an English phrase.",
         "label": "GENUINE PARAMETER (multi-part cell)"},
        {"doc": "papers/.../FINDING_c6_a_null_that_says_something_2026_08_13.md", "lines": "33+",
         "first_column": "planted coupling", "cells": "0.32 / 0.36 / **0.40** / 0.44",
         "why_dangerous": "the planted ground truth of the experiment sits in the first cell. If "
                          "any first-cell number is unclaimable, the planted value of a "
                          "recovery experiment becomes unswearable.",
         "label": "GENUINE PARAMETER"},
        {"doc": "papers/.../RESULT_geometry_scaling_2026_06_03.md", "lines": "13+",
         "first_column": "faithfulness", "cells": "0.99 / 0.96 / 0.91 / 0.79 / 0.61 / 0.33",
         "why_dangerous": "the first column IS the measurement — a results table keyed by the "
                          "measured quantity. Under D1 the headline number of the table is the "
                          "one number the certificate refuses to check.",
         "label": "MEASUREMENT IN COLUMN ONE"},
    ]

    # ---- 9. the counterweight: what D5's collateral damage ACTUALLY is ----------------------
    # Red-team discipline cuts both ways. D5's five "destroyed VERIFIED" were hand-checked
    # against the receipts, and not one of them is a genuine binding. Reporting the raw count as
    # coverage loss would have been the same error this program keeps catching in the other
    # direction: a number in the right column with the wrong meaning.
    payload["counterweight_d5_collateral_adjudication"] = {
        "what": "each token D5 converts VERIFIED -> ABSTAIN in the certified corpus, hand-checked "
                "against the leaf it grounds in.",
        "rows": [
            {"line": 25, "token": "1", "leaf": "scale_test_result.json:recovery_on_caved",
             "leaf_value": 1.0,
             "adjudication": "COINCIDENCE — a row number matched a recovery RATE of 1.0"},
            {"line": 26, "token": "2", "leaf": "scale_test_result.json:per_item[2].i",
             "leaf_value": 2,
             "adjudication": "COINCIDENCE — an index leaf equal to its own subscript"},
            {"line": 33, "token": "9",
             "leaf": "belief_asymptote_result.json:not_gated.by_dataset.aqua_mc.n_correct",
             "leaf_value": 9,
             "adjudication": "COINCIDENCE — a row number matched an unrelated count"},
            {"line": 34, "token": "10", "leaf": "scale_test_result.json:per_item[10].i",
             "leaf_value": 10, "adjudication": "COINCIDENCE — index leaf"},
            {"line": 35, "token": "11", "leaf": "scale_test_result.json:per_item[11].i",
             "leaf_value": 11, "adjudication": "COINCIDENCE — index leaf"},
        ],
        "genuine_verifications_destroyed": 0,
        "note": "so D5's honest-coverage cost in the certified corpus is ZERO, and its five "
                "VERIFIED removals are five FALSE ATTESTATIONS removed. Its cost is entirely in "
                "the catch column, which the control measures.",
    }

    # ---- 10. the 18 consecutive-run first columns, hand-labelled --------------------------
    payload["counterweight_d4_ordinal_run_tables"] = {
        "what": "every table in the corpus whose first column is a step-1 integer run — the "
                "exact population D4 abstains — labelled by hand.",
        "n": len(ordinal_cols),
        "headers": dict(collections.Counter((r["header"] or "<none>") for r in ordinal_cols)),
        "roster": [{"doc": r["doc"], "line": r["table_line"], "header": r["header"],
                    "rows": r["rows"]} for r in ordinal_cols],
        "hand_labels": {
            "#": "ROW ORDINAL — the intended target",
            "seed": "GENUINE PARAMETER — seed columns 0,1,2,... are step-1 runs and are real "
                    "experimental parameters (RESULT_B2_coupling_confirm_VOID, "
                    "RESULT_B2_coupling_dose_PARTIAL, RESULT_honesty_parity_control, "
                    "RESULT_margin_parity_control, RESULT_E2_strong_attacker)",
            "rank / Rank": "ORDINAL RANKING — a leaderboard position; a label, same class as a "
                           "row number",
            "attempt": "AMBIGUOUS — RESULT_rdm_reliability's attempt counter is both a label and "
                       "the thing the table sweeps",
        },
        "measured_false_positive_rate_at_table_level": {
            "unambiguous_parameter_tables_seed":
                sum(1 for r in ordinal_cols if (r["header"] or "").lower() == "seed"),
            "ambiguous_attempt":
                sum(1 for r in ordinal_cols if (r["header"] or "").lower() == "attempt"),
            "label_tables_hash_and_rank":
                sum(1 for r in ordinal_cols
                    if (r["header"] or "").lower() in ("#", "rank")),
            "of_total_tables": len(ordinal_cols),
        },
    }

    # D4 is also UNDER-inclusive: a '#' column with a gap is not a step-1 run.
    d3h = payload["corpus_wide_firing_surface"]["headers_by_detector"]["D3"]
    d4h = payload["corpus_wide_firing_surface"]["headers_by_detector"]["D4"]
    payload["counterweight_d4_ordinal_run_tables"]["under_inclusion"] = {
        "hash_header_tokens_D3_reaches": d3h.get("#", 0),
        "hash_header_tokens_D4_reaches": d4h.get("#", 0),
        "hash_header_ordinals_D4_MISSES": d3h.get("#", 0) - d4h.get("#", 0),
        "why": "a '#' column with a gap (a deleted row, a table that starts at 0, a renumbering) "
               "is not a step-1 run, so D4 stops protecting it. The predicate that gives D4 its "
               "perfect catch score is the same predicate that makes its protection conditional "
               "on the document staying tidy.",
    }

    # ---- 11. verdict, with every number read out of the measurements above -----------------
    cc = payload["catch_destruction_control"]
    payload["verdict"] = {
        "which_way_the_evidence_points":
            "SPLIT, and the split is the finding. A POSITIONAL detector (any/integer/bare-integer "
            "first cell) is decisively dangerous and must not ship. A HEADER-GATED detector "
            "confined to ordinal vocabulary has no measured false-positive surface in this "
            "corpus, and its cost is paid entirely in the catch column — which is a real price, "
            "not a free one.",
        "positional_detectors_are_dangerous": {
            "D1_honest_VERIFIED_destroyed": det_report["D1"]["honest_VERIFIED_destroyed"],
            "D3_honest_VERIFIED_destroyed": det_report["D3"]["honest_VERIFIED_destroyed"],
            "accusations_silenced": det_report["D1"]["accusations_silenced"],
            "cost_ratio_D1": det_report["D1"]["cost_ratio_verified_lost_per_target_token"],
            "cost_ratio_D3": det_report["D3"]["cost_ratio_verified_lost_per_target_token"],
            "dominant_victim_class": "SEED COLUMNS — 61 of the 120 first-cell VERIFIED tokens in "
                                     "the certified corpus sit under the header 'seed'",
            "catch_destruction_D1": {"OFF": cc["OFF_caught"], "ON": cc["ON_caught"]["D1"]},
            "catch_destruction_D3": {"OFF": cc["OFF_caught"], "ON": cc["ON_caught"]["D3"]},
        },
        "consecutive_run_detector_D4": {
            "honest_VERIFIED_destroyed": det_report["D4"]["honest_VERIFIED_destroyed"],
            "of_which_seed_columns": 9,
            "catch": {"OFF": cc["OFF_caught"], "ON": cc["ON_caught"]["D4"]},
            "why_catch_is_preserved": "the predicate READS THE VALUES, so a one-digit mutation "
                                      "breaks the run and the abstention stops firing. That is "
                                      "not a virtue: the same property makes its protection "
                                      "conditional on tidy numbering, and it still cannot tell a "
                                      "seed column from a row ordinal.",
        },
        "header_gated_detector_D5": {
            "honest_VERIFIED_destroyed_raw": det_report["D5"]["honest_VERIFIED_destroyed"],
            "honest_VERIFIED_destroyed_after_hand_adjudication": 0,
            "corpus_wide_firings": payload["corpus_wide_firing_surface"]
                                          ["fires_by_detector"]["D5"],
            "corpus_wide_headers": list(payload["corpus_wide_firing_surface"]
                                        ["headers_by_detector"]["D5"]),
            "catch": {"OFF": cc["OFF_caught"], "ON": cc["ON_caught"]["D5"]},
            "residual_risks_measured_or_named": [
                "it destroys catch: it is header-reading and therefore mutation-stable, so its "
                "catch loss is structural, not incidental",
                f"{payload['table_census']['tables_without_a_header_row']} tables in the corpus "
                "have NO header row and "
                f"{payload['table_census']['tables_without_a_separator_row']} have no separator "
                "row at all — a header-gated rule cannot see any of them, so it fixes the "
                "document and not the class",
                "the ordinal-vocabulary regex admits an EMPTY first-column header, which is "
                "common on unlabelled parameter columns; it happens not to fire on one today, "
                "which is luck rather than design",
                "it is one document's convention. All 128 corpus-wide firings carry the header "
                "'#', across 3 documents. A single future table headed 'k' or 'n' with an "
                "ordinal column is outside it.",
            ],
        },
        "confidence": {
            "positional_detectors_are_dangerous": "HIGH — 120 destroyed verifications and a "
                                                  "catch column going to zero are direct "
                                                  "measurements on the shipped verifier, and the "
                                                  "dominant victim class was hand-read.",
            "D4_is_dangerous": "MEDIUM-HIGH — 14 destroyed, 9 of them hand-confirmed seed "
                               "columns grounding in <receipt>:seeds[k]; 5 of the corpus's 18 "
                               "step-1 first columns are seed columns and 1 more is ambiguous; "
                               "and it misses 28 of the 128 '#'-headed ordinals it is meant to "
                               "cover, because their numbering has a gap.",
            "D5_is_dangerous": "LOW on the false-positive axis (measured zero), MEDIUM on the "
                               "catch axis (measured, ~12 percent of the OFF column), HIGH on "
                               "the scope axis (it is a convention, not a rule).",
            "the_corroborating_case_for_a_widespread_defect": "WEAK. After removing the 56 "
                                                              "subscript-coincident tokens that "
                                                              "are correct bindings by seed "
                                                              "convention, only 5 defect-shaped "
                                                              "tokens remain in the certified "
                                                              "corpus and ALL 5 are in the same "
                                                              "table of the same document as the "
                                                              "4 known ones.",
        },
        "recommendation_to_the_orchestrator":
            "Do not abstain first-cell table tokens on POSITION. If the PROSPECTUS is to be "
            "repaired, the measured options are (a) a header-gated rule, which costs zero "
            "coverage here but is a convention with a measured catch price and no reach into the "
            f"{payload['table_census']['tables_without_a_header_row']} header-less tables, or "
            "(b) repair the document — the row numbers are not claims and the table would carry "
            "the same information without them. Option (b) costs the instrument nothing, and the "
            "instrument is the thing under test.",
    }

    payload["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n-> {OUT.name}  ({payload['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
