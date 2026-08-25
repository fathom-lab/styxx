"""OATH v0.10 — candidate detectors for "this token is a MARKDOWN TABLE ROW ORDINAL".

MEASUREMENT ONLY. `styxx/certify.py` is never imported-and-mutated here: the detectors are
evaluated as a post-hoc overlay on the SHIPPED ledger, so nothing in this file can change what the
verifier does. Every regex the extractor uses is imported FROM the verifier rather than copied,
which is the only way a sweep can claim to be measuring the shipped instrument.

The defect this sweeps: `PROSPECTUS_knowsay_2026_07_27.md` is the only OATH-FAILED document in the
139-document fully-resolvable frame, and its four UNGROUNDED tokens are the leading index cells of
a claim table (`| 3 | Persists at 7B ... |`). They are not claims about anything. They are
obligated because the ROW's text carries trigger vocabulary and v0.3 binds a table row through its
header, so the whole row is one binding context. Value-only matching FALSE-VERIFIES them against
leaves like `per_item[3].i`; the shipped v0.3 count-binding filter ACCUSES them. The correct
status is ABSTAIN.

WHAT THIS FILE MEASURES, AND THE BAR IT HOLDS ITSELF TO
-------------------------------------------------------
An ordinal detector is an ABSTENTION RULE, and this program has already bought the lesson that an
abstention rule improves every tamper metric by destroying coverage (v0.9's `V09_IS_SPEC_BAR_NOUN`
took the CATCH column to zero at every seed and was DROPPED for it). So a precision sweep alone
cannot recommend anything, and two controls run alongside it:

  * FALSE-POSITIVE ROSTER (the precision bar). Every currently-VERIFIED token a detector fires on
    is enumerated. It is scored COINCIDENTAL only if it grounds in an index-like leaf under a
    mechanical, frozen test; **everything else is scored REAL CLAIM, and ties resolve REAL CLAIM —
    against the detector.** A detector that silences a real claim cannot ship, so (b) is the bar.
  * CATCH DESTRUCTION (the coverage bar). For every live token a detector fires on, one significant
    digit is perturbed at seeds 1-3, the document is re-certified at the shipped verifier, and the
    detector is re-evaluated ON THE MUTATED DOCUMENT. A catch is DESTROYED when the shipped
    verifier answers UNGROUNDED and the detector would have abstained it. This is mandatory, not
    optional: it is what separates a detector that sees an ordinal from a detector that stops
    looking.

  python papers/closed-model-frontier/oath_v10_ordinal_detectors.py
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

from styxx.certify import certify_doc  # noqa: E402
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
)
from styxx.corpus_audit import _resolve_receipts  # noqa: E402

OUT = HERE / "oath_v10_ordinal_detectors.json"
MUT_SEEDS = (1, 2, 3)
D3_CAP = 100                 # frozen default for D3/D7; swept separately below
D3_CAP_SWEEP = (9, 20, 50, 100, 1000)
D4_MIN_ROWS = 3              # a 2-long "sequence" 1,2 is not evidence of an ordinal column

TARGET_DOC = "papers/agent-conscience/PROSPECTUS_knowsay_2026_07_27.md"
TARGET_TOKENS = {("3", 27), ("4", 28), ("5", 29), ("8", 32)}

# ---------------------------------------------------------------- extraction with positions

def extract_with_pos(text: str) -> list[dict]:
    """`styxx.certify.extract_numbers`, replicated token-for-token, plus the token's OFFSET.

    The one deliberate difference: the sha/date/version scrub replaces each span with spaces of
    EQUAL LENGTH instead of a single space. `_NUM` cannot match inside a run of spaces and the
    boundary lookarounds see a space either way, so the extracted token sequence is identical --
    asserted against the shipped ledger for every document in `align_check` below -- while offsets
    now map 1:1 onto the (sign-normalized) source line. That is what makes a COLUMN INDEX and an
    exact in-place mutation computable at all; `ctx.find(token)` cannot tell two equal tokens on
    one line apart, and a table row is exactly where equal tokens collide.
    """
    out = []
    lines = text.splitlines()
    header_for: dict[int, str] = {}
    header_ln: dict[int, int] = {}
    block_of: dict[int, int] = {}
    blocks: dict[int, list[int]] = {}
    for i, line in enumerate(lines):
        if _TABLE_SEP.match(line) and i > 0 and lines[i - 1].lstrip().startswith("|"):
            hdr = lines[i - 1].strip()
            j = i + 1
            rows = []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                header_for[j + 1] = hdr
                header_ln[j + 1] = i          # 1-based header line number == i
                block_of[j + 1] = i
                rows.append(j + 1)
                j += 1
            if rows:
                blocks[i] = rows
    for ln_no, line in enumerate(lines, 1):
        line = line.replace("−", "-")
        scrub = _SHAISH.sub(lambda m: " " * len(m.group(0)), line)
        scrub = _DATEISH.sub(lambda m: " " * len(m.group(0)), scrub)
        scrub = _VERSIONISH.sub(lambda m: " " * len(m.group(0)), scrub)
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
            if m.start() >= 2 and scrub[m.start() - 1] in "–-−" \
                    and scrub[m.start() - 2].isdigit():
                continue
            if m.start() >= 2 and scrub[m.start() - 1] == "-" and scrub[m.start() - 2].isalpha():
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            out.append({"line": ln_no, "token": tok, "value": val, "decimals": _decimals(raw),
                        "context": line.strip()[:160], "start": m.start(),
                        "norm_line": line, "scrub": scrub,
                        "header": header_for.get(ln_no), "header_ln": header_ln.get(ln_no),
                        "block": block_of.get(ln_no)})
    return out, blocks, lines


def split_cells(line: str) -> list[str]:
    return line.split("|")


def annotate(tok: dict, lines: list[str]) -> None:
    """Attach the table-structure features every detector reads."""
    tok["in_table_row"] = tok["header"] is not None
    tok["first_cell"] = False
    tok["pipe_index"] = None
    tok["column_header_raw"] = None
    tok["column_header"] = None
    tok["cell_text"] = None
    tok["cell_sole_number"] = False
    if not tok["in_table_row"]:
        return
    row = tok["norm_line"]
    npipe = tok["scrub"][:tok["start"]].count("|")
    tok["pipe_index"] = npipe
    lead = row.lstrip().startswith("|")
    tok["first_cell"] = (npipe == 1) if lead else (npipe == 0)
    cells = split_cells(row)
    if npipe < len(cells):
        tok["cell_text"] = cells[npipe].strip()
        bare = re.sub(r"[*_`\s]", "", cells[npipe])
        tok["cell_sole_number"] = (bare == tok["token"])
    hcells = split_cells(tok["header"])
    if npipe < len(hcells):
        tok["column_header_raw"] = hcells[npipe].strip()
        h = re.sub(r"[*_`]", "", hcells[npipe]).strip().lower().rstrip(".:").strip()
        tok["column_header"] = h


ORDINAL_HEADERS = {"#", "", "no", "no.", "nº", "№", "row", "row #", "id", "idx",
                   "index", "rank", "claim #", "#.", "num", "nr"}


def block_sequence_ok(blocks, by_line, min_rows: int) -> dict[int, bool]:
    """Does the FIRST COLUMN of this table block hold a contiguous monotonic run from 0 or 1?

    Every data row must contribute a first-cell numeric token; a table where some rows have a
    numeric index and some do not is not an index column."""
    ok = {}
    for bid, rows in blocks.items():
        seq = []
        complete = True
        for ln in rows:
            firsts = [t for t in by_line.get(ln, []) if t["first_cell"]]
            if not firsts:
                complete = False
                break
            seq.append(firsts[0]["value"])
        if not complete or len(seq) < min_rows:
            ok[bid] = False
            continue
        if any(v != int(v) for v in seq):
            ok[bid] = False
            continue
        ints = [int(v) for v in seq]
        ok[bid] = ints in (list(range(1, len(ints) + 1)), list(range(0, len(ints))))
    return ok


# ---------------------------------------------------------------- the candidate detectors

def d1(t):
    return bool(t["first_cell"])


def d2(t):
    return d1(t) and t["column_header"] in ORDINAL_HEADERS


def d3(t, cap=D3_CAP):
    return d1(t) and t["decimals"] == 0 and abs(t["value"]) <= cap


def d4(t):
    return d1(t) and bool(t.get("seq_ok"))


def d5(t):
    return d2(t) or (d3(t) and d4(t))


def d6(t):
    return d3(t) and d4(t)


def d7(t):
    return d3(t) and t["cell_sole_number"]


def d8(t):
    return d7(t) and d4(t)


def d9(t):
    return d2(t) and d7(t)


def d10(t):
    return d2(t) and d4(t)


DETECTORS = {
    "D1": (d1, "first cell of a table data row, any content"),
    "D2": (d2, f"D1 AND column header in ordinal vocabulary {sorted(ORDINAL_HEADERS)} "
                "(empty header included)"),
    "D3": (d3, f"D1 AND bare integer with |value| <= {D3_CAP}"),
    "D4": (d4, f"D1 AND the first column of the block is a contiguous monotonic run from 0 or 1 "
                f"(>= {D4_MIN_ROWS} rows, every row contributing)"),
    "D5": (d5, "D2 OR (D3 AND D4)"),
    "D6": (d6, "D3 AND D4 -- the header-free core of D5, isolating the header's contribution"),
    "D7": (d7, "D3 AND the cell's ENTIRE content is the integer (markdown emphasis stripped)"),
    "D8": (d8, "D7 AND D4 -- tightest purely structural candidate"),
    "D9": (d9, "D2 AND D7 -- header vocabulary AND sole-content cell"),
    "D10": (d10, "D2 AND D4 -- header vocabulary AND contiguous column; the conjunction that "
                 "buys mutation-sensitivity on top of the header test"),
}

# ---------------------------------------------------------------- verified-token adjudication

# ---- the MECHANICAL proxy, kept only so its failure is on the record ------------------------
# First pass scored a fired VERIFIED token COINCIDENTAL when (i) the receipt path carries a
# subscript equal to the token's own value (`per_item[3].i` is a leaf equal to its own array index
# and matches that integer BY CONSTRUCTION) or (ii) the path's terminal segment is an index noun.
#
# IT IS WRONG ON THIS CORPUS AND IS NOT THE CLASSIFIER. The dominant first-column idiom here is a
# per-seed results table whose seeds are literally 0,1,2,... recorded in order as `seeds[k]`, so
# `seeds[k] == k` by CONTENT, not by construction. The proxy scored 56 of those as coincidences.
# They are real, correctly-bound claims -- the document states which seeds were run and the
# receipt records exactly those numbers -- so the proxy understated every detector's
# false-positive count by 56 tokens and flattered the two candidates that fire on them. Both
# splits are reported below; the HAND split is the bar.
_INDEX_LEAF = {"i", "idx", "index", "rank", "row", "order", "position", "pos"}


def mech_coincidental(receipt_ref: str | None, value: float) -> bool:
    if not receipt_ref or ":" not in receipt_ref:
        return False
    path = receipt_ref.split(":", 1)[1]
    if value == int(value):
        for sub in re.findall(r"\[(\d+)\]", path):
            if int(sub) == int(value):
                return True
    segs = [s for s in re.split(r"[.\[\]]", path) if s]
    return bool(segs) and segs[-1].lower() in _INDEX_LEAF


# ---- the HAND adjudication (authoritative), frozen after the first pass ----------------------
# Definition. A token is a NON-CLAIM iff it is a positional label -- a row number whose only job is
# to let prose say "row 4" -- so that changing it changes nothing the document asserts. A token is
# a REAL CLAIM iff the document asserts it: a grid value, a dose, a rank, a seed, a rate, a count,
# or a number quoted inside the row's text. **Ties resolve REAL CLAIM -- against the detector.**
#
# Every VERIFIED token that any candidate fires on was read by hand. Exactly FIVE are non-claims,
# and all five are in the target document's own claim table; they are named here and EVERYTHING
# ELSE defaults to REAL CLAIM. Three of the five (2, 10, 11) additionally satisfy the mechanical
# proxy -- they ground in `per_item[k].i` -- and two do not: `1` grounds in
# `scale_test_result.json:recovery_on_caved` (which happens to be 1.0) and `9` in
# `belief_asymptote_result.json:not_gated.by_dataset.aqua_mc.n_correct` (which happens to be 9).
# Those two are the false attestation this defect produces, sworn to in a committed certificate.
HAND_NON_CLAIM = {
    f"{TARGET_DOC}|L25|1",
    f"{TARGET_DOC}|L26|2",
    f"{TARGET_DOC}|L33|9",
    f"{TARGET_DOC}|L34|10",
    f"{TARGET_DOC}|L35|11",
}


def hand_non_claim(t: dict) -> bool:
    return f"{t['doc']}|L{t['line']}|{t['token']}" in HAND_NON_CLAIM


# ---------------------------------------------------------------- frame

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


def build_tokens(doc: Path, receipts: list[Path]):
    """Shipped ledger + structure features, index-aligned. Returns (tokens, verdict, aligned)."""
    text = doc.read_text(encoding="utf-8")
    toks, blocks, lines = extract_with_pos(text)
    by_line = collections.defaultdict(list)
    for t in toks:
        annotate(t, lines)
        by_line[t["line"]].append(t)
    seq = block_sequence_ok(blocks, by_line, D4_MIN_ROWS)
    for t in toks:
        t["seq_ok"] = seq.get(t["block"], False)
    cert = certify_doc(doc, receipts)
    led = cert["ledger"]
    aligned = len(led) == len(toks) and all(
        led[i]["line"] == toks[i]["line"] and led[i]["token"] == toks[i]["token"]
        for i in range(len(toks)))
    if aligned:
        for i, t in enumerate(toks):
            t["status"] = led[i]["status"]
            t["receipt_ref"] = led[i]["receipt_ref"]
    return toks, cert, aligned, lines


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


def mutate_doc(lines: list[str], t: dict, mut: str):
    """In-place substitution at the token's OWN offset -- no `line.replace`, which silently hits
    the wrong occurrence when a row's ordinal repeats inside the row (exactly this defect's
    population) and no-ops entirely on U+2212 tokens (the v0.7 harness defect)."""
    raw = lines[t["line"] - 1]
    norm = raw.replace("−", "-")
    s, e = t["start"], t["start"] + len(t["token"])
    if norm[s:e] != t["token"]:
        return None
    if len(mut) != len(t["token"]):                              # pragma: no cover - defensive
        return None
    new_norm = norm[:s] + mut + norm[e:]
    # `mutate_sig` swaps one digit for another, so lengths match and index i of the mutated line
    # corresponds to index i of the source line. Restore every U+2212 outside the mutated span so
    # the document is byte-identical apart from the doctored digit.
    ml = list(lines)
    ml[t["line"] - 1] = "".join(
        "−" if (i < s or i >= e) and raw[i] == "−" else ch
        for i, ch in enumerate(new_norm))
    return ml


def main() -> int:                                              # noqa: C901 - one report
    t0 = time.time()
    docs = resolvable_docs()
    print(f"frame: {len(docs)} documents with fully-resolvable receipts")

    all_tokens, verdicts, misaligned = [], {}, []
    doc_lines: dict[str, list[str]] = {}
    doc_receipts: dict[str, list[Path]] = {}
    doc_ntokens: dict[str, int] = {}
    fired_pre: dict[tuple[str, str], set[int]] = {}
    for doc, receipts in docs:
        rel = doc.relative_to(ROOT).as_posix()
        toks, cert, aligned, lines = build_tokens(doc, receipts)
        verdicts[rel] = cert["verdict"]
        doc_lines[rel] = lines
        doc_receipts[rel] = receipts
        if not aligned:
            misaligned.append(rel)
            continue
        doc_ntokens[rel] = len(toks)
        for i, t in enumerate(toks):
            t["doc"] = rel
            t["idx"] = i
        for name, (fn, _) in DETECTORS.items():
            fired_pre[(rel, name)] = {t["idx"] for t in toks if fn(t)}
        all_tokens.extend(toks)
    failed = sorted(d for d, v in verdicts.items() if v != "OATH-HELD")
    print(f"tokens {len(all_tokens)}   OATH-FAILED docs {len(failed)}   "
          f"extraction misalignments {len(misaligned)}")
    if misaligned:
        print("  MISALIGNED (excluded, reported):", misaligned)

    tables = [t for t in all_tokens if t["in_table_row"]]
    firsts = [t for t in tables if t["first_cell"]]
    print(f"tokens on table data rows: {len(tables)}   in the first cell: {len(firsts)}")

    ung = [t for t in all_tokens if t.get("status") == "UNGROUNDED"]
    print(f"UNGROUNDED tokens corpus-wide: {len(ung)}")
    target = [t for t in ung if t["doc"] == TARGET_DOC]
    print(f"  target ({Path(TARGET_DOC).name}): "
          + ", ".join(f"L{t['line']}:{t['token']}" for t in target))

    # ---- mutation control: one pass over the union of live fired tokens ------------------
    live_fired = {}
    for t in all_tokens:
        if t.get("status") not in ("VERIFIED", "UNGROUNDED"):
            continue
        if any(fn(t) for fn, _ in DETECTORS.values()):
            live_fired[(t["doc"], t["line"], t["start"])] = t
    print(f"\nmutation control: {len(live_fired)} live tokens x {len(MUT_SEEDS)} seeds", flush=True)

    mut_rows = []
    for seed in MUT_SEEDS:
        rng = random.Random(seed)
        for key, t in sorted(live_fired.items()):
            mut = mutate_sig(t["token"], rng)
            ml = mutate_doc(doc_lines[t["doc"]], t, mut)
            if ml is None:
                mut_rows.append({"seed": seed, "key": key, "landed": False})
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write("\n".join(ml))
                tmp = Path(tf.name)
            try:
                cert = certify_doc(tmp, doc_receipts[t["doc"]])
                mtoks, mblocks, mlines = extract_with_pos("\n".join(ml))
                mby = collections.defaultdict(list)
                for x in mtoks:
                    annotate(x, mlines)
                    mby[x["line"]].append(x)
                mseq = block_sequence_ok(mblocks, mby, D4_MIN_ROWS)
                for x in mtoks:
                    x["seq_ok"] = mseq.get(x["block"], False)
            finally:
                tmp.unlink(missing_ok=True)
            idx = next((i for i, x in enumerate(mtoks)
                        if x["line"] == t["line"] and x["start"] == t["start"]
                        and x["token"] == mut), None)
            if idx is None or idx >= len(cert["ledger"]):
                mut_rows.append({"seed": seed, "key": key, "landed": True,
                                 "status": "NOT_EXTRACTED", "fires": [], "collateral": {}})
                continue
            e = cert["ledger"][idx]
            if e["line"] != t["line"] or e["token"] != mut:
                mut_rows.append({"seed": seed, "key": key, "landed": True,
                                 "status": "LEDGER_DESYNC", "fires": [], "collateral": {}})
                continue
            mt = mtoks[idx]
            # COLLATERAL: tokens ELSEWHERE in the same document that the detector abstained before
            # the mutation and no longer abstains after it. A detector reading a whole-column
            # property loses the whole column when any one cell moves, so one doctored digit can
            # re-arm every other ordinal in the table.
            collateral = {}
            if len(mtoks) == doc_ntokens.get(t["doc"], -1):
                for n2, (fn2, _) in DETECTORS.items():
                    post = {i for i, x in enumerate(mtoks) if fn2(x)}
                    collateral[n2] = len(fired_pre[(t["doc"], n2)] - post - {idx})
            mut_rows.append({"seed": seed, "key": key, "landed": True, "status": e["status"],
                             "fires": [n for n, (fn, _) in DETECTORS.items() if fn(mt)],
                             "collateral": collateral})
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    # ---- per-detector report --------------------------------------------------------------
    report_dets = {}
    for name, (fn, desc) in DETECTORS.items():
        fired = [t for t in all_tokens if fn(t)]
        by_status = collections.Counter(t.get("status", "?") for t in fired)
        f_ung = [t for t in fired if t.get("status") == "UNGROUNDED"]
        f_ver = [t for t in fired if t.get("status") == "VERIFIED"]
        mech_a = [t for t in f_ver if mech_coincidental(t.get("receipt_ref"), t["value"])]
        hand_a = [t for t in f_ver if hand_non_claim(t)]
        hand_b = [t for t in f_ver if not hand_non_claim(t)]

        def row(t):
            return {"doc": t["doc"], "line": t["line"], "token": t["token"],
                    "column_header": t["column_header_raw"], "cell": t["cell_text"],
                    "receipt_ref": t.get("receipt_ref"), "context": t["context"],
                    "mechanical_proxy": ("coincidental"
                                         if mech_coincidental(t.get("receipt_ref"), t["value"])
                                         else "real-claim")}

        tgt = sum(1 for t in f_ung if t["doc"] == TARGET_DOC
                  and (t["token"], t["line"]) in TARGET_TOKENS)
        keys = {(t["doc"], t["line"], t["start"]) for t in fired}
        per_seed = []
        for seed in MUT_SEEDS:
            caught = destroyed = collat = 0
            for r in mut_rows:
                if r["seed"] != seed or r["key"] not in keys or not r.get("landed"):
                    continue
                collat += r.get("collateral", {}).get(name, 0)
                if r.get("status") == "UNGROUNDED":
                    caught += 1
                    if name in r.get("fires", []):
                        destroyed += 1
            per_seed.append({"seed": seed, "caught_by_shipped": caught,
                             "catches_destroyed": destroyed,
                             "collateral_abstentions_lost": collat})
        live = len(f_ung) + len(f_ver)
        report_dets[name] = {
            "definition": desc,
            "needs_column_header": name in ("D2", "D5", "D9", "D10"),
            "fires_on_tokens": len(fired),
            "fires_in_documents": len({t["doc"] for t in fired}),
            "documents": sorted({t["doc"] for t in fired}),
            "by_status": dict(by_status),
            "ungrounded_reached": len(f_ung),
            "target_ungrounded_reached": tgt,
            "target_ungrounded_total": len(TARGET_TOKENS),
            "ungrounded_outside_target": len(f_ung) - tgt,
            "ungrounded_roster": [row(t) for t in f_ung],
            "verified_fired": len(f_ver),
            "non_claim_a": len(hand_a),
            "false_positive_b": len(hand_b),
            "false_positive_roster_b": [row(t) for t in hand_b],
            "non_claim_roster_a": [row(t) for t in hand_a],
            "mechanical_proxy_would_have_said_a": len(mech_a),
            "mechanical_proxy_understated_b_by": len(mech_a) - len(hand_a),
            "live_tokens_fired": live,
            "precision_live_hand": round(1.0 - len(hand_b) / live, 4) if live else None,
            "catch_destruction": per_seed,
        }
        print(f"{name:<3} fires {len(fired):5d}  UNG {len(f_ung):3d} (target {tgt}/4)  "
              f"VER {len(f_ver):4d} = a{len(hand_a):3d} + b{len(hand_b):4d}   "
              f"caught/seed {[p['caught_by_shipped'] for p in per_seed]}  "
              f"destroyed {[p['catches_destroyed'] for p in per_seed]}  "
              f"collateral {[p['collateral_abstentions_lost'] for p in per_seed]}")

    # ---- D3 cap sweep ----------------------------------------------------------------------
    cap_sweep = {}
    for cap in D3_CAP_SWEEP:
        fired = [t for t in all_tokens if d3(t, cap)]
        f_ver = [t for t in fired if t.get("status") == "VERIFIED"]
        cap_sweep[str(cap)] = {
            "fires": len(fired),
            "ungrounded_reached": sum(1 for t in fired if t.get("status") == "UNGROUNDED"),
            "verified": len(f_ver),
            "false_positive_b": sum(1 for t in f_ver if not hand_non_claim(t))}
    print("\nD3 cap sweep:", json.dumps(cap_sweep))

    # every VERIFIED token any candidate fires on was read by hand; this is the population that
    # `HAND_NON_CLAIM` adjudicates, reported in full so the default-to-CLAIM is auditable.
    adjudicated_population = sorted({f"{t['doc']}|L{t['line']}|{t['token']}"
                                     for _n, (fn, _d) in DETECTORS.items() for t in all_tokens
                                     if fn(t) and t.get("status") == "VERIFIED"})

    # ---- exposure census over ALL papers/**/*.md ------------------------------------------
    # The certified frame is 139 documents; the repository holds far more markdown. A detector
    # that fires on 11 tokens in the frame is either a one-document special case or the visible
    # tip of a real class, and only the wider corpus can say which. No statuses here -- these
    # documents carry no certificate, so nothing about VERIFIED/UNGROUNDED is claimable.
    exposure = {n: {"tokens": 0, "documents": set()} for n in DETECTORS}
    exposure_docs = exposure_lossy = 0
    exposure_headers: collections.Counter = collections.Counter()
    exposure_headers_d9: collections.Counter = collections.Counter()
    d2_not_d9 = []          # the population the sole-content requirement strips
    for md in sorted(ROOT.glob("papers/**/*.md")):
        if "anc" in md.parts:
            continue
        exposure_docs += 1
        raw = md.read_bytes()
        try:
            body = raw.decode("utf-8")
        except UnicodeDecodeError:
            # counted and named, never silently dropped: skipping it would UNDERSTATE exposure,
            # which is the direction that flatters a detector.
            exposure_lossy += 1
            body = raw.decode("utf-8", errors="replace")
        toks, blocks, lines = extract_with_pos(body)
        by_line = collections.defaultdict(list)
        for t in toks:
            annotate(t, lines)
            by_line[t["line"]].append(t)
        seq = block_sequence_ok(blocks, by_line, D4_MIN_ROWS)
        rel = md.relative_to(ROOT).as_posix()
        for t in toks:
            t["seq_ok"] = seq.get(t["block"], False)
            for n, (fn, _) in DETECTORS.items():
                if fn(t):
                    exposure[n]["tokens"] += 1
                    exposure[n]["documents"].add(rel)
                    if n == "D2":
                        exposure_headers[t["column_header"] or "(empty)"] += 1
                        if not d9(t):
                            d2_not_d9.append({"doc": rel, "line": t["line"],
                                              "token": t["token"],
                                              "column_header": t["column_header_raw"],
                                              "cell": t["cell_text"],
                                              "context": t["context"][:140]})
                    if n == "D9":
                        exposure_headers_d9[t["column_header"] or "(empty)"] += 1
    exposure_out = {n: {"tokens": v["tokens"], "documents": len(v["documents"]),
                        "document_list": sorted(v["documents"])[:40]}
                    for n, v in exposure.items()}
    exposure_out["D2"]["column_headers_matched"] = dict(exposure_headers.most_common())
    exposure_out["D9"]["column_headers_matched"] = dict(exposure_headers_d9.most_common())
    exposure_out["D2_minus_D9"] = {
        "tokens": len(d2_not_d9),
        "documents": len({r["doc"] for r in d2_not_d9}),
        "roster": d2_not_d9,
        "note": "tokens D2 reaches and the sole-content requirement (D9) strips. These are "
                "first cells holding PROSE with a number in it under an ordinal-vocabulary "
                "header -- gate ids, question ids, run labels -- and the numbers inside them "
                "are real claims. NONE of these documents is in the 139-document certified "
                "frame, so this hazard is invisible to the in-frame precision table and is "
                "reported here as the reason the sole-content requirement is not optional.",
    }
    print(f"\nexposure census over {exposure_docs} markdown documents under papers/ "
          f"({exposure_lossy} needed lossy decode):")
    print(f"  D2 headers matched: {dict(exposure_headers.most_common())}")
    print(f"  D9 headers matched: {dict(exposure_headers_d9.most_common())}")
    print(f"  D2 \\ D9 (prose first cells the sole-content test strips): {len(d2_not_d9)} tokens "
          f"in {len({r['doc'] for r in d2_not_d9})} documents")
    for n in DETECTORS:
        print(f"  {n:<3} {exposure_out[n]['tokens']:5d} tokens in "
              f"{exposure_out[n]['documents']:4d} documents")

    ranked = sorted(report_dets.items(),
                    key=lambda kv: (kv[1]["false_positive_b"],
                                    -kv[1]["target_ungrounded_reached"],
                                    max(p["catches_destroyed"] for p in kv[1]["catch_destruction"]),
                                    kv[1]["fires_on_tokens"]))
    best = ranked[0][0]

    payload = {
        "purpose": "OATH v0.10 — precision/coverage sweep of table-row-ordinal detectors "
                   "(measurement only; styxx/certify.py untouched)",
        "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "frame": {"documents": len(docs), "tokens_scored": len(all_tokens),
                  "oath_failed_documents": failed,
                  "extraction_misaligned_documents": misaligned,
                  "tokens_on_table_data_rows": len(tables),
                  "tokens_in_first_cell": len(firsts),
                  "ungrounded_corpus_wide": len(ung)},
        "target": {"doc": TARGET_DOC,
                   "ungrounded": [{"line": t["line"], "token": t["token"],
                                   "column_header": t["column_header_raw"],
                                   "context": t["context"]} for t in target]},
        "adjudication_rule": {
            "non_claim_a": "a positional label -- a row number whose only job is to let prose say "
                           "'row 4' -- so changing it changes nothing the document asserts",
            "false_positive_b": "everything else: a grid value, dose, rank, seed, rate, count, or "
                                "a number quoted in the row's text. TIES RESOLVE FALSE POSITIVE, "
                                "against the detector. (b) IS THE BAR.",
            "hand_non_claim_roster": sorted(HAND_NON_CLAIM),
            "adjudicated_population": adjudicated_population,
            "mechanical_proxy_retired": {
                "rule": "subscript == token value, OR terminal path segment in "
                        + repr(sorted(_INDEX_LEAF)),
                "why_retired": "the dominant first-column idiom in this corpus is a per-seed "
                               "results table whose seeds ARE 0,1,2,... recorded in order as "
                               "seeds[k], so seeds[k] == k by CONTENT not by construction. The "
                               "proxy scored 56 real, correctly-bound seed claims as coincidences "
                               "and understated every detector's false-positive count by that "
                               "much -- flattering exactly the candidates that fire on them.",
            },
        },
        "catch_destruction_control": {
            "method": "one significant digit perturbed per live fired token at seeds "
                      f"{list(MUT_SEEDS)}; the document is re-certified at the SHIPPED verifier and "
                      "the detector re-evaluated ON THE MUTATED DOCUMENT. A catch is DESTROYED "
                      "when the shipped verifier answers UNGROUNDED and the detector fires. "
                      "COLLATERAL counts tokens elsewhere in the same document that the detector "
                      "abstained before the mutation and stops abstaining after it.",
            "live_tokens": len(live_fired),
            "mutations_that_did_not_land": sum(1 for r in mut_rows if not r.get("landed")),
        },
        "detectors": report_dets,
        "d3_cap_sweep": cap_sweep,
        "exposure_census_all_papers_md": {
            "documents_scanned": exposure_docs,
            "documents_needing_lossy_decode": exposure_lossy,
            "note": "no statuses here -- these documents carry no certificate, so nothing about "
                    "VERIFIED/UNGROUNDED is claimable. This sizes the CLASS a clause would reach, "
                    "nothing more.",
            "per_detector": exposure_out},
        "mechanical_ranking": {
            "order": [k for k, _ in ranked],
            "key": "false positives ASC, target coverage DESC, worst-seed catch destruction ASC, "
                   "fires ASC",
            "top": best,
            "why_it_is_not_the_recommendation": "the sort rewards D10's zero catch destruction, "
                                                "which is an artifact of the clause SELF-DISABLING "
                                                "under mutation, not of it detecting anything.",
        },
        "asserted_not_gated": {
            "A1": "ungrounded_outside_target == 0 for EVERY candidate is true BY CONSTRUCTION OF "
                  "THE FRAME: the 139 documents hold 4 UNGROUNDED tokens in total and all 4 are "
                  "the target. This frame therefore supplies ZERO evidence about whether an "
                  "ordinal detector would silence an accusation somewhere else. It is asserted, "
                  "not counted as a passed gate.",
            "A2": "target_ungrounded_reached == 4/4 for EVERY candidate, so target coverage does "
                  "not discriminate between candidates and cannot rank them. The whole ranking "
                  "rests on the false-positive column and on the mutation control.",
            "A3": "the in-frame false-positive count cannot see a hazard that lives only in "
                  "uncertified documents; `exposure_census_all_papers_md.per_detector."
                  "D2_minus_D9` is where that hazard was found, and it is a census, not a "
                  "certified measurement.",
        },
        "recommendation": {
            "candidate": "D9",
            "definition": "first cell of a markdown table data row, whose column header is in the "
                          "ordinal vocabulary, AND whose cell content is ENTIRELY a bare integer "
                          f"with |value| <= {D3_CAP} (markdown emphasis stripped)",
            "reasons": [
                "ZERO false positives on the 139-document frame, the only bar that matters. The "
                "first-cell position alone (D1) silences 115 real claims; adding the small-integer "
                "test (D3) still silences 74; adding the sole-content test (D7) still silences 70. "
                "Only the column header removes the last of them. The header is NOT optional.",
                "The contiguity family (D4/D6/D8, header-free) reaches 5 false positives -- the "
                "seed column 0,1,2,3,4 of RESULT_B2_coupling_confirm_VOID_2026_07_16.md, which is "
                "a contiguous run from 0 and is NOT an ordinal: those are the seeds the run used, "
                "recorded in the receipt as seeds[k]. Contiguity cannot tell an index from a "
                "0-based seed list.",
                "D9 == D2 in-frame but strips 43 tokens in 18 documents that D2 reaches in the "
                "wider corpus -- every '(empty)' and every 'id' header hit -- and those are prose "
                "first cells ('all 4 model arms complete', 'beats semantic_entropy TriviaQA 0.785 "
                "band', 'restricted to the 935 claims') whose numbers are real claims. The "
                "in-frame table cannot see this; the census can.",
                "It is LOCAL and IDEMPOTENT: it reads the token's own cell and its column header, "
                "nothing else. Every D4-family candidate reads a whole-COLUMN property, so one "
                "doctored digit costs 110 sibling abstentions per seed (90 for D10) -- doctor row "
                "3 and rows 4, 5 and 8 become accusations again. A clause that self-disables when "
                "a neighbouring cell is edited is a fuse, not a detector.",
            ],
            "the_number_that_argues_against_it": (
                "D9 destroys 5/6/4 of the 5/6/4 catches the shipped verifier lands on its class, "
                "i.e. 100%, exactly the shape that killed V09_IS_SPEC_BAR_NOUN. What differs is "
                "measured and stated: that class is 11 tokens whose entire cell content is a row "
                "number in ONE table, and the destroyed 'catch' is the verifier noticing a row "
                "LABEL changed, not a measurement. That last step is an ADJUDICATION, not a "
                "measurement, and it is the load-bearing assumption of this recommendation."),
            "residual": "D9 reaches 155 tokens in 14 documents across all 1107 papers/ markdown "
                        "files, of which only 11 tokens in 1 document sit in the certified frame "
                        "today. The other 144 gain nothing measurable until those documents carry "
                        "certificates.",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nhand-adjudicated VERIFIED population: {len(adjudicated_population)} tokens; "
          f"{len(HAND_NON_CLAIM)} scored NON-CLAIM, the rest REAL CLAIM")
    print(f"mechanical ranking: {' > '.join(k for k, _ in ranked)}  (top {best})")
    print("RECOMMENDED: D9 — 0 false positives in-frame, and the sole-content test strips the "
          "43-token prose-cell hazard D2 carries in the wider corpus. See "
          "payload['recommendation'] for the number that argues against it.")
    print(f"\nelapsed {payload['elapsed_s']}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
