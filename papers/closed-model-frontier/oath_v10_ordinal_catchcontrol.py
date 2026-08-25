"""OATH v1.0 — the CATCH-DESTRUCTION CONTROL for a table-row-ordinal abstention rule.

This is a measurement, not a change. `styxx/certify.py` is not touched: the candidate rule is
replicated as a post-ladder status override in this file, which is exactly equivalent to the real
clause because `is_spec` is FIRST in `certify_doc`'s status ladder (`if is_spec or is_hist:
status = "ABSTAIN"`) and every token's status is decided independently of every other token's.

## Why this runs BEFORE anyone writes the clause

`PREREG_oath_v09_is_spec_json_idiom_2026_08_23.md` shipped a JSON-idiom `is_spec` clause and
DROPPED its sibling, the prose bar-noun clause, for one measured reason: it took the CATCH column
to zero at every seed. An `is_spec` predicate reads the characters around a token, and a one-digit
substitution leaves them unchanged, so the predicate fires identically on the honest token and on
the doctored one. Such a clause does not detect tamper; it stops looking. Every tamper metric
improves because coverage is destroyed.

A table-row-ordinal rule is an abstention rule of the same family and is under the same suspicion.
The question this file answers, with numbers, is whether abstaining the class destroys REAL
catches, or whether the catch column is already ~zero because a row ordinal has no receipt to
disagree with.

## The two classes

  BROAD          the token is the ENTIRE first cell of a markdown table data row.
  HEADER-GATED   BROAD, and the table's first HEADER cell names an ordinal (`#`, `no.`, `idx`,
                 `row`, `rank`, `id`, `item`, `step`, `k`, ...).
  ORDINAL-SEQ    BROAD, and every data row of that table has an integer first cell and the column
                 is a consecutive +1 run (a third, orthogonal diagnostic; reported, not proposed).

Identification is POSITION-EXACT and does not replicate the verifier's regexes. Everything to the
left of a row's first cell is `|` plus whitespace, and the BROAD predicate requires the first cell
to hold exactly one number, so a first-cell token is necessarily the FIRST `extract_numbers` token
on its line. The roster therefore keys on (document, line, first-entry-on-line), never on the
token string, which is what keeps a repeated token elsewhere on the same row from aliasing in.

## Arms

  OFF  the SHIPPED verifier, unmodified.
  ON   the shipped verifier with the class overridden to ABSTAIN, re-evaluated on the MUTANT
       document's own structure — not assumed. If the predicate failed to fire on a mutant the ON
       arm would inherit the OFF status and that would show up here as a measurement.

Both arms are derived from ONE certification per (seed, token), because the override is a
post-ladder relabel of an independent per-token decision. This is an identity, not an
approximation, and it is stated so that the halved cost is not mistaken for a shortcut.

## What is an identity and what is a measurement

The ON arm's caught/false-attested columns on the mutated class token are **0 by construction**
(invariant I1 of the v0.9 prereg): the predicate reads document structure, the substitution
preserves token length and cell shape, so an abstained token stays abstained. That number is
reported because the instruction asked for both arms, and it is labelled an IDENTITY. The
load-bearing measurement in this file is the OFF arm — the size of what abstention would destroy —
together with the clean-corpus collateral and the discriminating/non-discriminating split of the
catches.

Non-destructive: mutants live in temp files, every corpus pass is in memory, and the only file
written is this script's own result JSON.

  python papers/closed-model-frontier/oath_v10_ordinal_catchcontrol.py
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
sys.path.insert(0, str(HERE))

from styxx.certify import certify_doc                                      # noqa: E402

# The mutation operator and the sign-aware substitution are IMPORTED from the v0.9 harness rather
# than copied. Copying is how a battery silently drifts from the instrument it claims to reuse --
# `substitute` exists because a bare `line.replace` no-ops on U+2212 tokens and the harness miss
# then scores as a verifier miss (the defect `run_oath_v07_battery.py` owns). Import is read-only;
# the module is not modified and its `main()` is import-guarded.
import run_oath_v09_battery as V09                                         # noqa: E402

OUT = HERE / "oath_v10_ordinal_catchcontrol.json"
MUT_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|?\s*$")     # same shape extract_numbers scans for
_CELL_DECOR = re.compile(r"[*`\s]")                   # bold/code/space decoration inside a cell
_ORDINAL_HDR = re.compile(r"^(#|no\.?|num(?:ber)?|nr|idx|index|row|rank|id|item|step|k)$", re.I)


# ------------------------------------------------------------------ the class predicate

def _first_cell(line: str):
    """Text of the first cell of a markdown table row, or None if the line is not one."""
    s = line.strip()
    if not s.startswith("|"):
        return None
    parts = s.split("|")
    return parts[1] if len(parts) >= 3 else None


def _norm(cell):
    return _CELL_DECOR.sub("", cell) if cell is not None else None


def ordinal_index(lines: list) -> dict:
    """1-based line number -> class metadata, for every markdown table DATA row.

    The table scan mirrors `extract_numbers`' own `header_for` construction (a separator row
    preceded by a `|` line opens a table; consecutive `|` lines are its data rows), so the header
    a row binds to here is the header the verifier binds it to.
    """
    rows_of, header_of = collections.defaultdict(list), {}
    tid = 0
    for i, line in enumerate(lines):
        if _TABLE_SEP.match(line) and i > 0 and lines[i - 1].lstrip().startswith("|"):
            tid += 1
            header_of[tid] = lines[i - 1].strip()
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows_of[tid].append(j + 1)
                j += 1
    seq_ok = {}
    for t, rows in rows_of.items():
        vals, ok = [], True
        for ln in rows:
            c = _norm(_first_cell(lines[ln - 1]))
            if c is None or not re.fullmatch(r"\d+", c):
                ok = False
                break
            vals.append(int(c))
        seq_ok[t] = bool(ok and len(vals) >= 2
                         and all(b - a == 1 for a, b in zip(vals, vals[1:])))
    out = {}
    for t, rows in rows_of.items():
        hdr_first = _norm(_first_cell(header_of[t])) or ""
        for ln in rows:
            c = _norm(_first_cell(lines[ln - 1]))
            if c is None:
                continue
            out[ln] = {"cell": c, "table_id": t, "header_first": hdr_first,
                       "header_gated": bool(_ORDINAL_HDR.fullmatch(hdr_first)),
                       "ordinal_sequence": seq_ok[t]}
    return out


def class_entry_ids(lines: list, ledger: list) -> tuple:
    """Indices into `ledger` that are first-cell-of-table-row tokens, per class.

    Only the FIRST ledger entry on a line can be the first cell (nothing but `|` and whitespace
    precedes it), so this is position-exact without re-implementing `_NUM`.
    """
    idx = ordinal_index(lines)
    seen, broad, gated, seqc = set(), set(), set(), set()
    for i, e in enumerate(ledger):
        if e["line"] in seen:
            continue
        seen.add(e["line"])
        m = idx.get(e["line"])
        if m and m["cell"] == e["token"]:
            broad.add(i)
            if m["header_gated"]:
                gated.add(i)
            if m["ordinal_sequence"]:
                seqc.add(i)
    return {"broad": broad, "header_gated": gated, "ordinal_sequence": seqc}, idx


CLASSES = ("broad", "header_gated", "ordinal_sequence")


# ------------------------------------------------------------------ clean-corpus baseline

def build_baseline(docs):
    roster, per_doc, lines_of = [], {}, {}
    total_tokens = 0
    for doc, receipts in docs:
        rel = doc.relative_to(ROOT).as_posix()
        lines = doc.read_text(encoding="utf-8").splitlines()
        lines_of[rel] = lines
        cert = certify_doc(doc, receipts)
        total_tokens += len(cert["ledger"])
        ids, idx = class_entry_ids(lines, cert["ledger"])
        per_doc[rel] = {"doc": doc, "receipts": receipts, "verdict": cert["verdict"],
                        "counts": cert["counts"],
                        "ungrounded_ids": {i for i, e in enumerate(cert["ledger"])
                                           if e["status"] == "UNGROUNDED"},
                        "class_ids": ids}
        on_line = collections.Counter(e["line"] for e in cert["ledger"])
        for i in sorted(ids["broad"]):
            e = cert["ledger"][i]
            m = idx[e["line"]]
            ref = e.get("receipt_ref") or ""
            roster.append({
                "rel": rel, "doc": doc.name, "line": e["line"], "token": e["token"],
                "value": e["value"], "decimals": e["decimals"],
                "ledger_index": i,
                "header_first": m["header_first"],
                "header_gated": m["header_gated"],
                "ordinal_sequence": m["ordinal_sequence"],
                "baseline_status": e["status"],
                "baseline_ref": ref,
                # a leaf whose own array subscript equals the claimed integer matches it BY
                # CONSTRUCTION (`per_item[3].i`); this flags that signature explicitly.
                "ref_self_subscript": bool(e["decimals"] == 0
                                           and f"[{int(e['value'])}]" in ref),
                "doc_baseline_verdict": cert["verdict"],
                "tokens_on_line": on_line[e["line"]],
                "context": e["context"][:140],
            })
    return roster, per_doc, lines_of, total_tokens


def clean_collateral(roster, per_doc, cls):
    """What abstaining `cls` costs and silences on the HONEST corpus."""
    sub = [r for r in roster if cls == "broad" or r[cls]]
    st = collections.Counter(r["baseline_status"] for r in sub)
    flips = []
    for rel, d in per_doc.items():
        if d["verdict"] == "OATH-HELD":
            continue
        if d["ungrounded_ids"] and d["ungrounded_ids"] <= d["class_ids"][cls]:
            flips.append(rel)
    return {
        "n": len(sub),
        "baseline_status": dict(st),
        "coverage_cost_VERIFIED_to_ABSTAIN": st["VERIFIED"],
        "accusations_silenced_UNGROUNDED_to_ABSTAIN": st["UNGROUNDED"],
        "already_ABSTAIN": st["ABSTAIN"],
        "documents_flipping_FAILED_to_HELD": sorted(flips),
        "verified_grounded_by_self_subscript": sum(1 for r in sub if r["ref_self_subscript"]),
    }


# ------------------------------------------------------------------ mutation

def run_seed(roster, per_doc, lines_of, seed):
    """Mutate one significant digit of each roster token, one token per temp document.

    Returns one row per roster token carrying the OFF status (shipped verifier) and the ON status
    for each class (shipped verifier + that class overridden to ABSTAIN, the override re-evaluated
    on the mutant's own structure).
    """
    rng = random.Random(seed)
    rows = []
    for r in roster:
        d = per_doc[r["rel"]]
        lines = lines_of[r["rel"]]
        mut = V09.mutate_sig(r["token"], rng)
        ml = list(lines)
        ml[r["line"] - 1], landed = V09.substitute(ml[r["line"] - 1], r["token"], mut)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                         encoding="utf-8") as tf:
            tf.write("\n".join(ml))
            tmp = Path(tf.name)
        try:
            cert = certify_doc(tmp, d["receipts"])
        finally:
            tmp.unlink(missing_ok=True)
        ids, _ = class_entry_ids(ml, cert["ledger"])
        on_line = [i for i, e in enumerate(cert["ledger"]) if e["line"] == r["line"]]
        i0 = on_line[0] if on_line else None
        e0 = cert["ledger"][i0] if i0 is not None else None
        if e0 is None or e0["token"] != mut:
            status_off = "NOT_EXTRACTED"
        else:
            status_off = e0["status"]
        ung = {i for i, e in enumerate(cert["ledger"]) if e["status"] == "UNGROUNDED"}
        row = {"key": f"{r['rel']}|L{r['line']}|{r['token']}", "mutant": mut,
               "landed": bool(landed), "baseline_status": r["baseline_status"],
               "doc_baseline_verdict": r["doc_baseline_verdict"],
               "status_off": status_off,
               "verdict_off": cert["verdict"],
               "header_gated": r["header_gated"], "ordinal_sequence": r["ordinal_sequence"]}
        for cls in CLASSES:
            in_class = i0 is not None and i0 in ids[cls]
            row[f"in_class_{cls}"] = in_class
            row[f"status_on_{cls}"] = "ABSTAIN" if in_class else status_off
            row[f"verdict_on_{cls}"] = ("OATH-FAILED" if (ung - ids[cls]) else "OATH-HELD")
        rows.append(row)
    return rows


def tally(rows, cls):
    sub = [x for x in rows if cls == "broad" or x[cls]]
    off = collections.Counter(x["status_off"] for x in sub)
    on = collections.Counter(x[f"status_on_{cls}"] for x in sub)
    caught = [x for x in sub if x["status_off"] == "UNGROUNDED"]
    return {
        "n": len(sub),
        "off": {"caught_UNGROUNDED": off["UNGROUNDED"], "false_attested_VERIFIED": off["VERIFIED"],
                "abstained": off["ABSTAIN"], "not_extracted": off["NOT_EXTRACTED"]},
        "on": {"caught_UNGROUNDED": on["UNGROUNDED"], "false_attested_VERIFIED": on["VERIFIED"],
               "abstained": on["ABSTAIN"], "not_extracted": on["NOT_EXTRACTED"]},
        # a catch is DISCRIMINATING only if the honest token was NOT already accused: if the
        # shipped verifier calls the true value UNGROUNDED too, accusing the doctored one
        # separates nothing and carries no information about the tamper.
        "catch_discriminating": sum(1 for x in caught if x["baseline_status"] == "VERIFIED"),
        "catch_non_discriminating": sum(1 for x in caught
                                        if x["baseline_status"] == "UNGROUNDED"),
        "catch_from_abstain": sum(1 for x in caught if x["baseline_status"] == "ABSTAIN"),
        # a catch only reaches a reader if it changes the certificate's verdict
        "catch_surfacing_in_verdict": sum(1 for x in caught
                                          if x["doc_baseline_verdict"] == "OATH-HELD"
                                          and x["verdict_off"] == "OATH-FAILED"),
        "did_not_land": sum(1 for x in sub if not x["landed"]),
        "override_missed_mutant": sum(1 for x in sub if not x[f"in_class_{cls}"]),
    }


def _mmr(vals):
    return {"mean": round(sum(vals) / len(vals), 2), "min": min(vals), "max": max(vals)}


# ------------------------------------------------------------------ seed-free exhaustive leg

def operator_support(tok: str, draws: int = 5000):
    """Every distinct mutant the SHIPPED operator can emit for `tok`, found by driving it.

    The operator's position/digit choice is not re-implemented here — re-implementing it is how a
    battery starts measuring its own copy instead of the instrument. Instead `mutate_sig` is
    driven `draws` times from deterministic seeds and the distinct outputs collected.
    `last_new_at` is reported so a reader can see the enumeration saturated rather than take it on
    trust.
    """
    seen, last_new = {}, -1
    for k in range(draws):
        m = V09.mutate_sig(tok, random.Random(k))
        if m not in seen:
            seen[m] = k
            last_new = k
    return sorted(seen), last_new


def exhaustive_leg(roster, per_doc, lines_of):
    """Certify EVERY single-digit substitution of every roster token — no seed at all.

    Ten seeds sample this space; enumerating it removes the sampling noise from the decisive
    question, which is what the shipped verifier does to a doctored row ordinal in general.
    """
    rows = []
    for r in roster:
        d = per_doc[r["rel"]]
        lines = lines_of[r["rel"]]
        variants, last_new = operator_support(r["token"])
        counts = collections.Counter()
        for mut in variants:
            ml = list(lines)
            ml[r["line"] - 1], landed = V09.substitute(ml[r["line"] - 1], r["token"], mut)
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write("\n".join(ml))
                tmp = Path(tf.name)
            try:
                cert = certify_doc(tmp, d["receipts"])
            finally:
                tmp.unlink(missing_ok=True)
            on_line = [e for e in cert["ledger"] if e["line"] == r["line"]]
            e0 = on_line[0] if on_line else None
            counts[e0["status"] if (e0 and e0["token"] == mut and landed)
                   else "NOT_EXTRACTED"] += 1
        rows.append({"key": f"{r['rel']}|L{r['line']}|{r['token']}", "token": r["token"],
                     "header_gated": r["header_gated"],
                     "ordinal_sequence": r["ordinal_sequence"],
                     "baseline_status": r["baseline_status"],
                     "baseline_ref": r["baseline_ref"],
                     "ref_self_subscript": r["ref_self_subscript"],
                     "n_variants": len(variants), "support_last_new_at": last_new,
                     "outcomes": dict(counts)})
    return rows


def exhaustive_summary(rows, cls):
    sub = [x for x in rows if cls == "broad" or x[cls]]
    agg = collections.Counter()
    for x in sub:
        agg.update(x["outcomes"])
    n = sum(agg.values())
    return {"n_tokens": len(sub), "n_mutants": n,
            "UNGROUNDED": agg["UNGROUNDED"], "VERIFIED": agg["VERIFIED"],
            "ABSTAIN": agg["ABSTAIN"], "NOT_EXTRACTED": agg["NOT_EXTRACTED"],
            "share_UNGROUNDED": round(agg["UNGROUNDED"] / n, 4) if n else None,
            "share_VERIFIED": round(agg["VERIFIED"] / n, 4) if n else None,
            # a doctored token whose HONEST value the verifier already accuses: the accusation
            # separates nothing, so it is not detection whatever the mutant's status.
            "mutants_of_already_accused_tokens": sum(
                sum(x["outcomes"].values()) for x in sub
                if x["baseline_status"] == "UNGROUNDED")}


def main() -> int:
    t0 = time.time()
    docs = V09.resolvable_docs()
    roster, per_doc, lines_of, total_tokens = build_baseline(docs)
    print(f"documents with fully-resolvable receipts: {len(docs)}   "
          f"ledger tokens: {total_tokens}")
    print(f"roster (first-cell-of-table-row): broad {len(roster)}   "
          f"header-gated {sum(1 for r in roster if r['header_gated'])}   "
          f"ordinal-sequence {sum(1 for r in roster if r['ordinal_sequence'])}\n")

    collateral = {cls: clean_collateral(roster, per_doc, cls) for cls in CLASSES}
    for cls in CLASSES:
        c = collateral[cls]
        print(f"CLEAN {cls:<17s} n={c['n']:<4d} VERIFIED {c['coverage_cost_VERIFIED_to_ABSTAIN']:<4d}"
              f" UNGROUNDED {c['accusations_silenced_UNGROUNDED_to_ABSTAIN']:<3d}"
              f" ABSTAIN {c['already_ABSTAIN']:<3d}"
              f" | FAILED->HELD docs {len(c['documents_flipping_FAILED_to_HELD'])}"
              f" | self-subscript grounds {c['verified_grounded_by_self_subscript']}")

    print(f"\nmutation, seeds {MUT_SEEDS} (one significant digit per token, one token per "
          f"mutant document)")
    per_seed = {cls: [] for cls in CLASSES}
    all_rows = {}
    for seed in MUT_SEEDS:
        rows = run_seed(roster, per_doc, lines_of, seed)
        all_rows[seed] = rows
        for cls in CLASSES:
            t = tally(rows, cls)
            t["seed"] = seed
            per_seed[cls].append(t)
        b = per_seed["broad"][-1]
        g = per_seed["header_gated"][-1]
        print(f"  seed {seed:<3d} BROAD caught OFF {b['off']['caught_UNGROUNDED']:3d} -> ON "
              f"{b['on']['caught_UNGROUNDED']:3d}  false-attested OFF "
              f"{b['off']['false_attested_VERIFIED']:3d} -> ON "
              f"{b['on']['false_attested_VERIFIED']:3d}   ||  GATED caught OFF "
              f"{g['off']['caught_UNGROUNDED']:2d} -> ON {g['on']['caught_UNGROUNDED']:2d}  "
              f"false-attested OFF {g['off']['false_attested_VERIFIED']:2d} -> ON "
              f"{g['on']['false_attested_VERIFIED']:2d}", flush=True)

    summary = {}
    for cls in CLASSES:
        ts = per_seed[cls]
        summary[cls] = {
            "n_tokens": ts[0]["n"],
            "OFF_caught": _mmr([t["off"]["caught_UNGROUNDED"] for t in ts]),
            "OFF_false_attested": _mmr([t["off"]["false_attested_VERIFIED"] for t in ts]),
            "ON_caught": _mmr([t["on"]["caught_UNGROUNDED"] for t in ts]),
            "ON_false_attested": _mmr([t["on"]["false_attested_VERIFIED"] for t in ts]),
            "catches_lost_by_abstaining": _mmr([t["off"]["caught_UNGROUNDED"]
                                                - t["on"]["caught_UNGROUNDED"] for t in ts]),
            "catch_discriminating": _mmr([t["catch_discriminating"] for t in ts]),
            "catch_non_discriminating": _mmr([t["catch_non_discriminating"] for t in ts]),
            "catch_from_abstain": _mmr([t["catch_from_abstain"] for t in ts]),
            "catch_surfacing_in_verdict": _mmr([t["catch_surfacing_in_verdict"] for t in ts]),
            "did_not_land": _mmr([t["did_not_land"] for t in ts]),
            "override_missed_mutant": _mmr([t["override_missed_mutant"] for t in ts]),
            "not_extracted": _mmr([t["off"]["not_extracted"] for t in ts]),
        }
        s = summary[cls]
        print(f"\n{cls}: OFF caught {s['OFF_caught']['mean']} "
              f"[{s['OFF_caught']['min']}-{s['OFF_caught']['max']}]   "
              f"OFF false-attested {s['OFF_false_attested']['mean']} "
              f"[{s['OFF_false_attested']['min']}-{s['OFF_false_attested']['max']}]   "
              f"ON caught {s['ON_caught']['mean']}   ON false-attested "
              f"{s['ON_false_attested']['mean']}")
        print(f"    of the OFF catches: discriminating "
              f"{s['catch_discriminating']['mean']} "
              f"[{s['catch_discriminating']['min']}-{s['catch_discriminating']['max']}], "
              f"non-discriminating {s['catch_non_discriminating']['mean']}, "
              f"from-abstain {s['catch_from_abstain']['mean']}, "
              f"surfacing in a verdict {s['catch_surfacing_in_verdict']['mean']}")

    print("\nseed-free exhaustive leg: every single-digit substitution the shipped operator "
          "can emit,\nfor every roster token", flush=True)
    ex_rows = exhaustive_leg(roster, per_doc, lines_of)
    ex = {cls: exhaustive_summary(ex_rows, cls) for cls in CLASSES}
    for cls in CLASSES:
        e = ex[cls]
        print(f"  {cls:<17s} {e['n_tokens']:>4d} tokens -> {e['n_mutants']:>5d} mutants   "
              f"UNGROUNDED {e['UNGROUNDED']:>5d} ({e['share_UNGROUNDED']:.3f})   "
              f"VERIFIED {e['VERIFIED']:>5d} ({e['share_VERIFIED']:.3f})   "
              f"ABSTAIN {e['ABSTAIN']:>5d}   NOT_EXTRACTED {e['NOT_EXTRACTED']}")

    report = {
        "purpose": "ANGLE 3 — catch-destruction control for a markdown table-row-ordinal "
                   "abstention rule, measured BEFORE the clause is written.",
        "doctrine": "PREREG_oath_v09_is_spec_json_idiom_2026_08_23.md dropped the prose bar-noun "
                    "clause because it took the CATCH column to zero at every seed. Any "
                    "abstention rule is under the same suspicion until its catch cost is "
                    "measured.",
        "verifier_untouched": True,
        "verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "mutation_operator": "run_oath_v09_battery.mutate_sig (imported, not copied)",
        "substitution": "run_oath_v09_battery.substitute (sign-aware, imported, not copied)",
        "seeds": list(MUT_SEEDS),
        "frame": {
            "documents_with_resolvable_receipts": len(docs),
            "ledger_tokens_total": total_tokens,
            "roster_broad": len(roster),
            "roster_header_gated": sum(1 for r in roster if r["header_gated"]),
            "roster_ordinal_sequence": sum(1 for r in roster if r["ordinal_sequence"]),
            "first_header_cells": dict(collections.Counter(
                r["header_first"] for r in roster).most_common()),
        },
        "clean_corpus_collateral": collateral,
        "mutation_summary": summary,
        "mutation_per_seed": per_seed,
        "exhaustive_summary": ex,
        "exhaustive_per_token": ex_rows,
        "roster": roster,
        "caught_examples": {
            cls: [{k: x[k] for k in ("key", "mutant", "baseline_status", "status_off",
                                     "doc_baseline_verdict", "verdict_off")}
                  for x in all_rows[MUT_SEEDS[0]]
                  if (cls == "broad" or x[cls]) and x["status_off"] == "UNGROUNDED"][:40]
            for cls in CLASSES
        },
        "measured_notes": {
            "ordinal_sequence_is_value_reading":
                "`override_missed_mutant` for the ordinal_sequence class is the count of mutants "
                "on which the predicate FAILED to fire. It reads the token's value (a consecutive "
                "+1 run across the column), so doctoring any ordinal breaks the run and the "
                "predicate switches off on exactly the input it exists to handle. Its ON arm is "
                "therefore identical to its OFF arm, and its clean-corpus behaviour is not "
                "evidence about its behaviour under tamper. Same shape as the standing rule that "
                "an obligation consulting `hits` cannot gate.",
            "catch_surfacing_in_verdict":
                "A ledger-level UNGROUNDED reaches a reader only through the certificate verdict. "
                "This column counts catches on documents that were OATH-HELD before the mutation "
                "and OATH-FAILED after; a catch inside an already-failing document changes "
                "nothing a reader sees.",
        },
        "asserted_identity_not_a_measurement": {
            "I1": "The ON arm's caught/false-attested columns on the mutated class token are 0 BY "
                  "CONSTRUCTION: the override predicate reads document structure only, and a "
                  "one-significant-digit substitution preserves both the token's length and the "
                  "cell's shape, so an abstained token stays abstained. `override_missed_mutant` "
                  "is the audit of that identity, and it is 0 where the identity holds. Reporting "
                  "ON=0 as a tamper result would launder an identity as a finding.",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nelapsed {report['elapsed_s']}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
