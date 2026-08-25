"""OATH v0.10 pre-fix census — how big is the MARKDOWN TABLE ROW ORDINAL class, and what does
the shipped verifier do with it?

Run at the SHIPPED verifier. ``styxx/certify.py`` is NOT touched: everything structural here is
replicated from the shipped source and checked against the live extractor claim-for-claim (a
``replication_mismatches`` of 0 is the licence to read anything else in this file).

The lead. ``PROSPECTUS_knowsay_2026_07_27.md`` is the only OATH-FAILED document among the
resolvable corpus, and its four UNGROUNDED tokens are the leading index cells of a claim table --
``| 3 | Persists at 7B ... |``, ``| 4 | Pressure reaches the output ... |``. They are row NUMBERS.
They are obligated because the row's TEXT carries trigger vocabulary and the whole row is one
binding context, and they are accused because the v0.3 count-binding filter empties their hits.
Under value-only matching they instead FALSE-VERIFY against leaves like ``per_item[3].i`` -- a leaf
equal to its own array subscript, which matches that integer by construction. Both regimes are
wrong; ABSTAIN (or non-extraction) is the correct status.

Six measurements, all mechanical, none of them a proposal:

  1. CLASS SIZE. Every first-cell numeric token of every markdown table data row, repo-wide under
     ``papers/**`` (excluding ``anc/``), and the certified-corpus subset.

  2. CURRENT STATUS. VERIFIED / ABSTAIN / UNGROUNDED for the certified-corpus subset at the live
     verifier, plus how many sit on lines the shipped OBLIGATION predicate binds -- the number that
     decides whether a token has a catch that an abstention rule could destroy.

  3. FALSE VERIFICATION against an INDEX-LIKE leaf: a cited leaf whose terminal path segment is an
     index name (``i`` / ``idx`` / ``index`` / ``n`` / ``j`` / ``k`` / ``ix``), or whose value
     equals its own array subscript (or that subscript + 1, the 1-based ordinal form).

  4. FIRST-COLUMN HEADER frequency. THE design question: some first columns hold real claims -- a
     lambda, a layer index that is a genuine experimental parameter, a seed. An ordinal column and
     a parameter column are structurally the same shape.

  5. MONOTONIC 1..N. Whether a table's first-cell values are exactly ``1..N`` in row order (the
     strong ordinal signal), how many tables satisfy it, and -- the separability read -- the
     cross-tabulation of that signal against the header vocabulary, in BOTH directions.

  6. MUTATION LEDGER (added leg, not requested but mandatory doctrine here). One significant digit
     perturbed per certified-corpus candidate, seeds 1-5, at the SHIPPED verifier. A rule that
     abstains or un-extracts this class destroys every CATCH in this column, and improves every
     tamper metric by doing so. The census cannot measure a clause it does not build, but it CAN
     price what such a clause would cost, and refusing to look is the defect this repo studies.

  python papers/closed-model-frontier/oath_v10_ordinal_census.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import (_DATEISH, _FORMULA_AFTER, _MD_STRUCTURE,        # noqa: E402
                           _NUM, _SHAISH, _TABLE_SEP, _TRIGGERS, _TRIGGERS_CORR,
                           _VERSIONISH, _YEAR,
                           _decimals, certify_doc, extract_numbers, receipt_values)
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v10_ordinal_census.json"
MUT_SEEDS = (1, 2, 3, 4, 5)

# Terminal path segments that name a POSITION rather than a measurement. Frozen here; the full
# terminal-segment frequency table is also written out so a reader can re-slice without re-running.
INDEX_SEGS = frozenset({"i", "idx", "index", "n", "j", "k", "ix"})

# First-column header vocabulary that READS as an ordinal / label column rather than as an
# experimental parameter. Frozen before the counts were seen; the raw header frequency table is
# written out unnormalised so this partition can be disputed against the data.
ORDINAL_HEADERS = frozenset({"#", "no", "no.", "num", "nr", "row", "id", "idx", "index", "item",
                             "n", "", "-", "claim", "line", "rank"})


# ---------------------------------------------------------------- extraction, replicated with cols
#
# certify.py's `extract_numbers` does not record a column, and the first-cell test needs one.
# This is that function with `col` (the match start IN SCRUB COORDINATES, which is the coordinate
# system the shipped positional filter itself uses) added and nothing else changed. The
# replication is checked against the live function for every document scanned.

def extract_numbers_pos(text: str) -> list[dict]:
    out = []
    lines = text.splitlines()
    header_for: dict[int, str] = {}
    for i, line in enumerate(lines):
        if _TABLE_SEP.match(line) and i > 0 and lines[i - 1].lstrip().startswith("|"):
            hdr = lines[i - 1].strip()
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                header_for[j + 1] = hdr
                j += 1
    for ln_no, line in enumerate(lines, 1):
        line = line.replace("−", "-")
        scrub = _SHAISH.sub(" ", line)
        scrub = _DATEISH.sub(" ", scrub)
        scrub = _VERSIONISH.sub(" ", scrub)
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
            entry = {"line": ln_no, "token": tok, "value": val, "decimals": _decimals(raw),
                     "context": line.strip()[:160], "col": m.start()}
            if ln_no in header_for:
                entry["binding_context"] = (header_for[ln_no] + " " + line.strip())[:320]
            out.append(entry)
    return out


def replication_diff(text: str, mine: list[dict]) -> int:
    """0 iff the replicated extractor is the shipped one. Anything else voids this file."""
    live = extract_numbers(text)
    if len(live) != len(mine):
        return abs(len(live) - len(mine)) or 1
    bad = 0
    for a, b in zip(live, mine):
        if (a["line"], a["token"], a["value"], a["decimals"]) != \
           (b["line"], b["token"], b["value"], b["decimals"]):
            bad += 1
    return bad


# ---------------------------------------------------------------- table structure, replicated

def tables_of(text: str):
    """Every markdown table certify.py's header machinery sees: (sep_index, header_line, rows).

    Same predicate as `extract_numbers`: a `_TABLE_SEP` line preceded by a line starting with '|',
    then every following line that starts with '|'."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if _TABLE_SEP.match(line) and i > 0 and lines[i - 1].lstrip().startswith("|"):
            rows = []
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(j + 1)          # 1-based
                j += 1
            out.append({"sep": i + 1, "header_line": i, "header": lines[i - 1].strip(),
                        "rows": rows})
    return out


def first_cell_span(s: str):
    """(start, end) of the FIRST cell of a pipe-delimited row, in the coordinates of *s*.

    None when the line carries no leading pipe or no closing pipe for cell one."""
    lead = len(s) - len(s.lstrip())
    if not s[lead:lead + 1] == "|":
        return None
    p1 = s.find("|", lead + 1)
    if p1 < 0:
        return None
    return (lead + 1, p1)


def first_cell_text(line: str) -> str:
    span = first_cell_span(line)
    if span is None:
        return ""
    return line[span[0]:span[1]]


def norm_header(cell: str) -> str:
    return re.sub(r"[*`_\s]+", " ", cell).strip().lower()


def sole_token(cell: str, tok: str) -> bool:
    """The first cell holds THIS token and nothing else (bold/backticks/whitespace stripped).

    The structural class 'first-cell numeric token' is broader than 'row ordinal': a cell reading
    `p95 latency` or `AUROC (k=3)` contributes a first-cell token that is plainly part of a LABEL,
    not an index. This flag is the cheapest separator available and it is reported alongside every
    other count so the two populations are never averaged."""
    return re.sub(r"[*`_\s]+", "", cell) == re.sub(r"[*`_\s]+", "", tok)


def table_shapes(text: str, nums: list[dict]) -> dict:
    """Per-table first-column shape, keyed by the table's separator line (1-based).

    ONE implementation, used by both the repo-wide leg and the certified-corpus leg, so the two
    can never diverge on what 'exactly 1..N' means."""
    lines = text.splitlines()
    by_line = collections.defaultdict(list)
    for e in nums:
        by_line[e["line"]].append(e)
    out = {}
    for t in tables_of(text):
        seq, n_numeric, n_sole, cells = [], 0, 0, {}
        for r in t["rows"]:
            raw = lines[r - 1].replace("−", "-")
            scrub = _VERSIONISH.sub(" ", _DATEISH.sub(" ", _SHAISH.sub(" ", raw)))
            span = first_cell_span(scrub)
            if span is None:
                continue
            toks = [e for e in by_line.get(r, []) if span[0] <= e["col"] < span[1]]
            cells[r] = (span, scrub[span[0]:span[1]], toks)
            if len(toks) == 1:
                n_numeric += 1
                seq.append(toks[0]["value"])
                if sole_token(scrub[span[0]:span[1]], toks[0]["token"]):
                    n_sole += 1
        n_data = len(t["rows"])
        ints = [v for v in seq if float(v).is_integer()]
        all_num = n_numeric == n_data and n_data >= 2 and len(ints) == len(seq)
        out[t["sep"]] = {
            "header": t["header"],
            "first_col_header": norm_header(first_cell_text(t["header"])),
            "first_col_header_raw": first_cell_text(t["header"]).strip(),
            "n_data_rows": n_data, "n_rows_first_cell_numeric": n_numeric,
            "n_rows_first_cell_sole_token": n_sole,
            "all_rows_numeric": n_numeric == n_data and n_data >= 1,
            "all_rows_sole_token": n_sole == n_data and n_data >= 1,
            "strict_1_to_n": all_num and seq == [float(i) for i in range(1, n_data + 1)],
            "strict_increasing_int": all_num and all(seq[i] < seq[i + 1]
                                                     for i in range(len(seq) - 1)),
            "sequence": seq[:40], "cells": cells, "rows": t["rows"],
        }
    return out


def is_bound_shipped(bctx: str, pre: str, value: float, decimals: int) -> bool:
    """The shipped obligation predicate. Row ordinals are integers, so the fractional-correlation
    register and the v0.7 precision clause can never fire on them; both are kept for fidelity."""
    if _TRIGGERS.search(bctx):
        return True
    if re.search(r"\bn\s*=\s*$", pre, re.I):
        return True
    if decimals > 0 and -1.0 <= value <= 1.0 and _TRIGGERS_CORR.search(bctx):
        return True
    return decimals >= 7


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


_SUBSCRIPT = re.compile(r"\[(\d+)\](?!.*\[\d+\])")


def leaf_class(path: str, leaf_val, tok_val: float) -> dict:
    """Is the cited leaf an INDEX rather than a measurement?"""
    segs = [s for s in re.split(r"[.\[\]]", path) if s]
    terminal = segs[-1] if segs else ""
    sub = _SUBSCRIPT.search(path)
    sub_i = int(sub.group(1)) if sub else None
    return {
        "terminal_segment": terminal,
        "terminal_is_index_name": terminal.lower() in INDEX_SEGS,
        # a bare array element (`seeds[1]`) -- the path names NO field, only a position. A leaf
        # here equalling its own subscript is AMBIGUOUS: `per_item[3].i` matches by construction,
        # but `seeds[1] = 1` is a real seed that happens to sit at position 1, and grounding a
        # "seed 1" claim in it is CORRECT. The two are not separable from the path alone and are
        # therefore counted apart, never summed into one false-verification number.
        "terminal_is_bare_subscript": terminal.isdigit(),
        "subscript": sub_i,
        "value_equals_subscript": (sub_i is not None and leaf_val is not None
                                   and float(leaf_val) == float(sub_i)),
        "value_equals_subscript_plus_1": (sub_i is not None and leaf_val is not None
                                          and float(leaf_val) == float(sub_i) + 1.0),
        "token_equals_subscript": sub_i is not None and tok_val == float(sub_i),
        "token_equals_subscript_plus_1": sub_i is not None and tok_val == float(sub_i) + 1.0,
    }


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


def substitute_at(line: str, tok: str, mut: str, col: int):
    """Land *mut* in place of the token AT ITS OWN COLUMN, not the first occurrence.

    A row ordinal is a small integer and small integers repeat on their own line ('| 3 | ... 3B
    ...'), so `run_oath_v09_battery`'s first-occurrence `replace` would doctor the WRONG token and
    score a harness miss as a verifier miss. The typographic-minus fallback that harness owns is
    kept: extraction normalises U+2212 to ASCII '-', so a negative token is reported in ASCII while
    the document holds U+2212."""
    if line[col:col + len(tok)] == tok:
        return line[:col] + mut + line[col + len(tok):], True
    if tok in line:
        return line.replace(tok, mut, 1), True
    if tok.startswith("-"):
        alt, alt_mut = tok.replace("-", "−", 1), mut.replace("-", "−", 1)
        if alt in line:
            return line.replace(alt, alt_mut, 1), True
    return line, False


def main() -> int:
    repl_bad, repl_docs = 0, 0

    # ------------------------------------------------------------ 1. class size, repo-wide
    repo_rows, repo_tables, md_scanned = [], [], 0
    for md in sorted(ROOT.glob("papers/**/*.md")):
        if "anc" in md.parts:
            continue
        md_scanned += 1
        text = md.read_text(encoding="utf-8", errors="replace")
        nums = extract_numbers_pos(text)
        repl_bad += replication_diff(text, nums)
        repl_docs += 1
        rel = md.relative_to(ROOT).as_posix()
        for sep, sh in table_shapes(text, nums).items():
            n_cand = 0
            for r, (span, cell_txt, toks) in sh["cells"].items():
                for e in toks:
                    n_cand += 1
                    repo_rows.append({
                        "rel": rel, "doc": md.name, "line": r, "token": e["token"],
                        "value": e["value"], "decimals": e["decimals"],
                        "table_sep": sep, "first_col_header": sh["first_col_header"],
                        "first_col_header_raw": sh["first_col_header_raw"],
                        "sole_token_in_cell": (len(toks) == 1
                                               and sole_token(cell_txt, e["token"])),
                        "ordinal_shaped": (len(toks) == 1 and sh["strict_1_to_n"]
                                           and sole_token(cell_txt, e["token"])),
                        "context": e["context"][:160],
                    })
            repo_tables.append({k: v for k, v in sh.items() if k != "cells"}
                               | {"rel": rel, "table_sep": sep, "n_candidate_tokens": n_cand,
                                  "header_is_ordinal_vocab":
                                      sh["first_col_header"] in ORDINAL_HEADERS})

    repo_docs = sorted({r["rel"] for r in repo_rows})
    print(f"markdown documents scanned (papers/**, no anc/): {md_scanned}")
    print(f"[1] first-cell table tokens repo-wide: {len(repo_rows)} "
          f"in {len(repo_docs)} documents, over {len(repo_tables)} tables")
    print(f"    replication check: {repl_docs} documents, {repl_bad} extractor mismatches")

    # ------------------------------------------------------------ 2/3. status in certified corpus
    docs = resolvable_docs()
    print(f"\n[2] documents with fully-resolvable receipts: {len(docs)}")
    roster, corpus_tokens = [], 0
    corpus_status = collections.Counter()
    for doc, receipts in docs:
        text = doc.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        nums = extract_numbers_pos(text)
        repl_bad += replication_diff(text, nums)
        repl_docs += 1
        try:
            cert = certify_doc(doc, receipts)
        except Exception as exc:                                  # pragma: no cover - defensive
            print(f"    SKIP {doc.name}: {exc}")
            continue
        corpus_tokens += len(cert["ledger"])
        corpus_status.update(e["status"] for e in cert["ledger"])
        # the ledger is built by iterating extract_numbers in order, so index alignment is exact
        aligned = len(cert["ledger"]) == len(nums) and all(
            a["line"] == b["line"] and a["token"] == b["token"]
            for a, b in zip(cert["ledger"], nums))
        if not aligned:
            repl_bad += 1
            continue
        rvals = {}
        for rp in receipts:
            j = json.loads(rp.read_text(encoding="utf-8"))
            for path, v in receipt_values(j):
                rvals[(rp.name, path)] = v
        cand_by_line = {}
        for sep, sh in table_shapes(text, nums).items():
            for r, (span, cell_txt, toks) in sh["cells"].items():
                cand_by_line[r] = (span, (sep, sh["first_col_header"],
                                          sh["first_col_header_raw"], cell_txt,
                                          sh["strict_1_to_n"], sh["strict_increasing_int"],
                                          len(toks)))
        rel = doc.relative_to(ROOT).as_posix()
        for e, led in zip(nums, cert["ledger"]):
            entry = cand_by_line.get(e["line"])
            if entry is None:
                continue
            span, hit = entry
            if not span[0] <= e["col"] < span[1]:
                continue
            ctx = lines[e["line"] - 1].strip().replace("−", "-")
            at = ctx.find(e["token"])
            pre = ctx[max(0, at - 18):at] if at >= 0 else ""
            bctx = e.get("binding_context", e["context"])
            ref = led["receipt_ref"]
            lc = None
            if led["status"] == "VERIFIED" and isinstance(ref, str) and ":" in ref:
                rn, path = ref.split(":", 1)
                lc = leaf_class(path, rvals.get((rn, path)), e["value"])
                lc["receipt"] = rn
                lc["path"] = path
                lc["leaf_value"] = rvals.get((rn, path))
            roster.append({
                "rel": rel, "doc": doc.name, "line": e["line"], "token": e["token"],
                "value": e["value"], "decimals": e["decimals"], "col": e["col"],
                "table_sep": hit[0], "first_col_header": hit[1],
                "first_col_header_raw": hit[2],
                "sole_token_in_cell": hit[6] == 1 and sole_token(hit[3], e["token"]),
                "table_strict_1_to_n": hit[4],
                "table_strict_increasing_int": hit[5],
                "ordinal_shaped": (hit[6] == 1 and hit[4]
                                   and sole_token(hit[3], e["token"])),
                "status": led["status"], "receipt_ref": ref,
                "bound_shipped": is_bound_shipped(bctx, pre, e["value"], e["decimals"]),
                "leaf": lc, "context": e["context"][:160],
            })

    st = collections.Counter(r["status"] for r in roster)
    repo_sole = [r for r in repo_rows if r["sole_token_in_cell"]]
    sole = [r for r in roster if r["sole_token_in_cell"]]
    sole_st = collections.Counter(r["status"] for r in sole)
    label_st = collections.Counter(r["status"] for r in roster if not r["sole_token_in_cell"])
    print(f"    corpus tokens {corpus_tokens}  {dict(corpus_status)}")
    print(f"    first-cell candidates in the certified corpus: {len(roster)} "
          f"in {len({r['rel'] for r in roster})} documents  {dict(st)}")
    print(f"    on lines the shipped OBLIGATION predicate binds: "
          f"{sum(r['bound_shipped'] for r in roster)}")
    print(f"    SOLE-TOKEN cells (the cell is the number and nothing else): "
          f"repo-wide {len(repo_sole)} of {len(repo_rows)}, "
          f"corpus {len(sole)} of {len(roster)}  {dict(sole_st)}")
    print(f"    the rest -- a number inside a LABEL cell ('p95 latency'): {dict(label_st)}")
    repo_ord = [r for r in repo_rows if r["ordinal_shaped"]]
    ordinal = [r for r in roster if r["ordinal_shaped"]]
    ord_st = collections.Counter(r["status"] for r in ordinal)
    print(f"    ORDINAL-SHAPED (sole-token cell in an exactly-1..N table): "
          f"repo-wide {len(repo_ord)}, corpus {len(ordinal)}  {dict(ord_st)}")
    print(f"    ordinal-shaped headers, repo-wide: "
          f"{dict(collections.Counter(r['first_col_header'] for r in repo_ord).most_common())}")

    ver = [r for r in roster if r["status"] == "VERIFIED"]
    idx_name = [r for r in ver if r["leaf"] and r["leaf"]["terminal_is_index_name"]]
    eq_sub = [r for r in ver if r["leaf"] and r["leaf"]["value_equals_subscript"]]
    eq_sub1 = [r for r in ver if r["leaf"] and r["leaf"]["value_equals_subscript_plus_1"]]
    idx_any = [r for r in ver if r["leaf"] and (r["leaf"]["terminal_is_index_name"]
                                                or r["leaf"]["value_equals_subscript"]
                                                or r["leaf"]["value_equals_subscript_plus_1"])]
    bare_pos = [r for r in ver if r["leaf"] and r["leaf"]["terminal_is_bare_subscript"]
                and r["leaf"]["value_equals_subscript"]]
    unambiguous = [r for r in idx_any if r["leaf"]["terminal_is_index_name"]]
    print(f"\n[3] VERIFIED first-cell tokens: {len(ver)}")
    print(f"    cited leaf's terminal segment is an index name : {len(idx_name)}")
    print(f"    cited leaf's value equals its own subscript    : {len(eq_sub)}")
    print(f"    ... equals subscript + 1 (1-based ordinal form): {len(eq_sub1)}")
    print(f"    union -- cited leaf is INDEX-LIKE (upper bound): {len(idx_any)}")
    print(f"    of which UNAMBIGUOUS (an explicit index FIELD, '...[k].i'): {len(unambiguous)}")
    print(f"    of which a BARE ARRAY POSITION ('seeds[1]' = 1) -- AMBIGUOUS, a genuine "
          f"0..N-1 parameter sweep is indistinguishable: {len(bare_pos)}")
    idx_hdr = collections.Counter(r["first_col_header"] for r in idx_any)
    print(f"    index-like by first-column header: {dict(idx_hdr.most_common())}")
    term_freq = collections.Counter(r["leaf"]["terminal_segment"] for r in ver if r["leaf"])

    # ------------------------------------------------------------ 4. first-column headers
    hdr_repo = collections.Counter(r["first_col_header"] for r in repo_rows)
    hdr_corpus = collections.Counter(r["first_col_header"] for r in roster)
    hdr_status = collections.defaultdict(collections.Counter)
    for r in roster:
        hdr_status[r["first_col_header"]][r["status"]] += 1
    hdr_repo_sole = collections.Counter(r["first_col_header"] for r in repo_sole)
    hdr_sole_status = collections.defaultdict(collections.Counter)
    for r in sole:
        hdr_sole_status[r["first_col_header"]][r["status"]] += 1
    print("\n[4] first-column header frequency (repo-wide, top 30) -- all / sole-token:")
    for h, n in hdr_repo.most_common(30):
        print(f"    {n:5d} / {hdr_repo_sole.get(h, 0):4d}  {h!r}")

    # ------------------------------------------------------------ 5. monotonic 1..N
    numeric_tables = [t for t in repo_tables if t["n_candidate_tokens"] > 0]
    n_1n = [t for t in numeric_tables if t["strict_1_to_n"]]
    n_inc = [t for t in numeric_tables if t["strict_increasing_int"]]
    n_allnum = [t for t in numeric_tables if t["all_rows_numeric"]]
    # the separability cross-tab, BOTH directions
    xt = {
        "ordinal_header_and_1_to_n": sum(1 for t in numeric_tables
                                         if t["header_is_ordinal_vocab"] and t["strict_1_to_n"]),
        "ordinal_header_not_1_to_n": sum(1 for t in numeric_tables
                                         if t["header_is_ordinal_vocab"]
                                         and not t["strict_1_to_n"]),
        "other_header_and_1_to_n": sum(1 for t in numeric_tables
                                       if not t["header_is_ordinal_vocab"]
                                       and t["strict_1_to_n"]),
        "other_header_not_1_to_n": sum(1 for t in numeric_tables
                                       if not t["header_is_ordinal_vocab"]
                                       and not t["strict_1_to_n"]),
    }
    tok_1n = sum(t["n_candidate_tokens"] for t in n_1n)
    tok_inc = sum(t["n_candidate_tokens"] for t in n_inc)
    print(f"\n[5] tables carrying >=1 first-cell numeric token: {len(numeric_tables)}")
    print(f"    every data row's first cell numeric        : {len(n_allnum)}")
    print(f"    strictly increasing integers               : {len(n_inc)} "
          f"({tok_inc} tokens)")
    print(f"    EXACTLY 1..N in row order                  : {len(n_1n)} "
          f"({tok_1n} tokens)")
    print(f"    cross-tab (header ordinal-vocab x 1..N)    : {xt}")
    n_sole_tab = [t for t in numeric_tables if t["all_rows_sole_token"]]
    both = [t for t in numeric_tables if t["all_rows_sole_token"] and t["strict_1_to_n"]]
    both_hdr = collections.Counter(t["first_col_header"] for t in both)
    tok_both = sum(t["n_candidate_tokens"] for t in both)
    print(f"    every data row a SOLE-TOKEN first cell      : {len(n_sole_tab)}")
    print(f"    sole-token AND exactly 1..N                 : {len(both)} "
          f"({tok_both} tokens)  headers {dict(both_hdr.most_common())}")
    # what a header-only rule and a shape-only rule would each miss / over-take
    hdr_1n_examples = collections.Counter(t["first_col_header"] for t in n_1n)
    other_hdr_1n = sorted({t["first_col_header"] for t in n_1n
                           if not t["header_is_ordinal_vocab"]})
    ord_hdr_not1n = [{"rel": t["rel"], "header": t["first_col_header"],
                      "sequence": t["sequence"][:12]}
                     for t in numeric_tables
                     if t["header_is_ordinal_vocab"] and not t["strict_1_to_n"]][:40]

    # ------------------------------------------------------------ 6. mutation ledger
    doc_by_rel = {d.relative_to(ROOT).as_posix(): (d, rc) for d, rc in docs}
    per_seed, mut_rows, unlanded = [], [], 0
    for seed in MUT_SEEDS:
        rng = random.Random(seed)
        rows = []
        for r in roster:
            doc, receipts = doc_by_rel[r["rel"]]
            lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
            mut = mutate_sig(r["token"], rng)
            ml = list(lines)
            # the recorded col is a SCRUB coordinate; on an unscrubbed line they coincide, and
            # substitute_at falls back when they do not.
            ml[r["line"] - 1], landed = substitute_at(ml[r["line"] - 1], r["token"], mut, r["col"])
            unlanded += not landed
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write("\n".join(ml))
                tmp = Path(tf.name)
            try:
                cert = certify_doc(tmp, receipts)
            finally:
                tmp.unlink(missing_ok=True)
            e = next((x for x in cert["ledger"]
                      if x["line"] == r["line"] and x["token"] == mut), None)
            rows.append({"seed": seed, "rel": r["rel"], "line": r["line"],
                         "token": r["token"], "mutant": mut, "landed": landed,
                         "clean_status": r["status"], "bound_shipped": r["bound_shipped"],
                         "first_col_header": r["first_col_header"],
                         "sole_token_in_cell": r["sole_token_in_cell"],
                         "ordinal_shaped": r["ordinal_shaped"],
                         "mutant_status": e["status"] if e else "NOT_EXTRACTED",
                         "mutant_ref": (e or {}).get("receipt_ref")})
        c = collections.Counter(x["mutant_status"] for x in rows)
        cs = collections.Counter(x["mutant_status"] for x in rows if x["sole_token_in_cell"])
        co = collections.Counter(x["mutant_status"] for x in rows if x["ordinal_shaped"])
        # A TRUE catch requires the clean token to have been VERIFIED: a token already accused on
        # the unmutated document carries no information about tamper detection, and counting it as
        # a catch would credit a standing FALSE ACCUSATION as coverage. The four PROSPECTUS row
        # ordinals are exactly that case, so the distinction is not academic here.
        per_seed.append({"seed": seed, "caught_ungrounded": c["UNGROUNDED"],
                         "false_attested_verified": c["VERIFIED"], "abstained": c["ABSTAIN"],
                         "not_extracted": c["NOT_EXTRACTED"],
                         "true_catch": sum(1 for x in rows if x["clean_status"] == "VERIFIED"
                                           and x["mutant_status"] == "UNGROUNDED"),
                         "already_accused_when_clean":
                             sum(1 for x in rows if x["clean_status"] == "UNGROUNDED"),
                         "sole_token_caught": cs["UNGROUNDED"],
                         "sole_token_false_attested": cs["VERIFIED"],
                         "ordinal_shaped_caught": co["UNGROUNDED"],
                         "ordinal_shaped_true_catch":
                             sum(1 for x in rows if x["ordinal_shaped"]
                                 and x["clean_status"] == "VERIFIED"
                                 and x["mutant_status"] == "UNGROUNDED"),
                         "ordinal_shaped_false_attested": co["VERIFIED"]})
        mut_rows.extend(rows)
    ns = len(MUT_SEEDS)
    mean_caught = sum(s["caught_ungrounded"] for s in per_seed) / ns
    mean_false = sum(s["false_attested_verified"] for s in per_seed) / ns
    print(f"\n[6] one-digit mutation of every candidate, SHIPPED verifier, {ns} seeds "
          f"({unlanded} did not land):")
    for s in per_seed:
        print(f"    seed {s['seed']:<3d} caught {s['caught_ungrounded']:4d}   "
              f"false-attested {s['false_attested_verified']:4d}   "
              f"abstained {s['abstained']:4d}   not-extracted {s['not_extracted']:4d}"
              f"   | sole-token caught {s['sole_token_caught']:4d} "
              f"false-attested {s['sole_token_false_attested']:4d}")
    print(f"    mean CATCHES a non-extraction / abstention rule would destroy : {mean_caught:.1f}")
    print(f"    mean FALSE ATTESTATIONS such a rule would remove              : {mean_false:.1f}")
    m_sole_c = sum(s["sole_token_caught"] for s in per_seed) / ns
    m_sole_f = sum(s["sole_token_false_attested"] for s in per_seed) / ns
    m_true = sum(s["true_catch"] for s in per_seed) / ns
    m_ord_c = sum(s["ordinal_shaped_caught"] for s in per_seed) / ns
    m_ord_tc = sum(s["ordinal_shaped_true_catch"] for s in per_seed) / ns
    m_ord_f = sum(s["ordinal_shaped_false_attested"] for s in per_seed) / ns
    print(f"    TRUE catches (clean VERIFIED -> mutant UNGROUNDED)            : {m_true:.1f}")
    print(f"    restricted to SOLE-TOKEN cells: catches {m_sole_c:.1f}, "
          f"false attestations {m_sole_f:.1f}")
    print(f"    restricted to ORDINAL-SHAPED  : catches {m_ord_c:.1f} "
          f"(true catches {m_ord_tc:.1f}), false attestations {m_ord_f:.1f}")

    report = {
        "purpose": "OATH v0.10 angle 1 — size of the markdown-table row-ordinal class and the "
                   "shipped verifier's treatment of it. MEASUREMENT ONLY; styxx/certify.py "
                   "untouched.",
        "generated_at_verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "replication_control": {
            "documents_checked": repl_docs,
            "extractor_mismatches_vs_live_extract_numbers": repl_bad,
            "note": "0 is the licence to read the rest of this file",
        },
        "frame": {
            "repo_wide": "papers/**/*.md excluding anc/",
            "markdown_documents_scanned": md_scanned,
            "certified": "papers/** documents carrying a *.certificate.json (no anc/) whose "
                         "recorded receipts ALL resolve",
            "resolvable_documents": len(docs),
            "corpus_tokens": corpus_tokens,
            "corpus_status_counts": dict(corpus_status),
        },
        "class_size": {
            "definition": "a numeric token of extract_numbers whose column falls inside the FIRST "
                          "cell of a markdown table DATA row, where 'data row' is exactly what "
                          "certify.py's header machinery calls one (a _TABLE_SEP line preceded by "
                          "a '|' line, then every following '|' line)",
            "repo_wide_tokens": len(repo_rows),
            "repo_wide_documents": len(repo_docs),
            "repo_wide_tables": len(repo_tables),
            "repo_wide_tables_with_candidates": len(numeric_tables),
            "certified_corpus_tokens": len(roster),
            "certified_corpus_documents": len({r["rel"] for r in roster}),
            "sole_token_cells": {
                "definition": "the first cell contains this number and nothing else once "
                              "whitespace / bold markers / backticks are removed -- the shape a "
                              "row ordinal actually has. The complement is a number embedded in a "
                              "LABEL cell ('p95 latency', 'AUROC (k=3)'), which is a different "
                              "population and is never averaged with this one here.",
                "repo_wide_tokens": len(repo_sole),
                "repo_wide_documents": len({r["rel"] for r in repo_sole}),
                "certified_corpus_tokens": len(sole),
                "certified_corpus_status": dict(sole_st),
                "label_cell_certified_corpus_status": dict(label_st),
            },
            "ordinal_shaped": {
                "definition": "a sole-token first cell in a table whose first column is EXACTLY "
                              "1..N in row order -- the conjunction of the two structural signals "
                              "measured here. Reported because it is the only operating point at "
                              "which no PARAMETER column survives in this repository.",
                "repo_wide_tokens": len(repo_ord),
                "repo_wide_documents": len({r["rel"] for r in repo_ord}),
                "repo_wide_headers": dict(collections.Counter(
                    r["first_col_header"] for r in repo_ord).most_common()),
                "certified_corpus_tokens": len(ordinal),
                "certified_corpus_status": dict(ord_st),
                "certified_corpus_headers": dict(collections.Counter(
                    r["first_col_header"] for r in ordinal).most_common()),
                "rows_certified_corpus": ordinal,
            },
        },
        "current_status": {
            "counts": dict(st),
            "on_bound_lines": sum(r["bound_shipped"] for r in roster),
            "on_bound_lines_by_status": {
                s: sum(1 for r in roster if r["status"] == s and r["bound_shipped"])
                for s in ("VERIFIED", "ABSTAIN", "UNGROUNDED")},
            "share_of_corpus_tokens": (round(len(roster) / corpus_tokens, 6)
                                       if corpus_tokens else None),
            "ungrounded_rows": [r for r in roster if r["status"] == "UNGROUNDED"],
        },
        "false_verification": {
            "verified": len(ver),
            "terminal_is_index_name": len(idx_name),
            "leaf_value_equals_subscript": len(eq_sub),
            "leaf_value_equals_subscript_plus_1": len(eq_sub1),
            "index_like_union_UPPER_BOUND": len(idx_any),
            "unambiguous_explicit_index_field": len(unambiguous),
            "ambiguous_bare_array_position": len(bare_pos),
            "index_like_by_first_column_header": dict(idx_hdr.most_common()),
            "disclosed_limitation":
                "the union is an UPPER BOUND on false verification, not a measurement of it. "
                "`per_item[3].i = 3` matches by construction and is a false grounding; "
                "`seeds[1] = 1` is a real seed value sitting at position 1 and grounding a "
                "'seed 1' claim in it is CORRECT. Nothing in the receipt path separates the two, "
                "so they are reported apart and never summed into one number.",
            "index_seg_vocabulary": sorted(INDEX_SEGS),
            "terminal_segment_frequency": dict(term_freq.most_common()),
            "rows": idx_any,
        },
        "first_column_headers": {
            "repo_wide_frequency": dict(hdr_repo.most_common()),
            "repo_wide_frequency_sole_token_cells": dict(hdr_repo_sole.most_common()),
            "certified_corpus_frequency": dict(hdr_corpus.most_common()),
            "certified_corpus_status_by_header": {h: dict(c) for h, c in hdr_status.items()},
            "certified_corpus_status_by_header_sole_token": {h: dict(c) for h, c
                                                             in hdr_sole_status.items()},
            "ordinal_vocabulary_frozen": sorted(ORDINAL_HEADERS),
        },
        "shape": {
            "tables_with_candidates": len(numeric_tables),
            "tables_all_rows_first_cell_numeric": len(n_allnum),
            "tables_strictly_increasing_int": len(n_inc),
            "tables_exactly_1_to_n": len(n_1n),
            "tables_all_rows_sole_token": len(n_sole_tab),
            "tables_sole_token_and_1_to_n": len(both),
            "tokens_in_sole_token_and_1_to_n_tables": tok_both,
            "headers_of_sole_token_and_1_to_n_tables": dict(both_hdr.most_common()),
            "tokens_in_1_to_n_tables": tok_1n,
            "tokens_in_strictly_increasing_tables": tok_inc,
            "header_x_shape_crosstab": xt,
            "headers_of_1_to_n_tables": dict(hdr_1n_examples.most_common()),
            "non_ordinal_headers_that_are_1_to_n": other_hdr_1n,
            "ordinal_headers_that_are_NOT_1_to_n": ord_hdr_not1n,
            "tables": repo_tables,
        },
        "mutation_ledger": {
            "seeds": list(MUT_SEEDS),
            "tokens_per_seed": len(roster),
            "did_not_land": unlanded,
            "per_seed": per_seed,
            "mean_catches_destroyed": round(mean_caught, 2),
            "mean_false_attestations_removed": round(mean_false, 2),
            "mean_true_catches_destroyed": round(m_true, 2),
            "sole_token_mean_catches_destroyed": round(m_sole_c, 2),
            "sole_token_mean_false_attestations_removed": round(m_sole_f, 2),
            "ordinal_shaped_mean_catches_destroyed": round(m_ord_c, 2),
            "ordinal_shaped_mean_true_catches_destroyed": round(m_ord_tc, 2),
            "ordinal_shaped_mean_false_attestations_removed": round(m_ord_f, 2),
            "catches_range": [min(s["caught_ungrounded"] for s in per_seed),
                              max(s["caught_ungrounded"] for s in per_seed)],
            "false_attestation_range": [min(s["false_attested_verified"] for s in per_seed),
                                        max(s["false_attested_verified"] for s in per_seed)],
            "rows": mut_rows,
        },
        "roster": roster,
        "repo_wide_roster": repo_rows,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n-> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
