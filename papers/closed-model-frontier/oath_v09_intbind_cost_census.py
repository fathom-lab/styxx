"""OATH v0.9 — what does the SHIPPED v0.3 COUNT-BINDING rule destroy? (cost census, MEASUREMENT ONLY)

`certify_doc` filters an INTEGER claim's value-matches to leaves whose receipt PATH shares a 4-char
word stem with the claim's binding context. It has been live since v0.3. Its BENEFIT has been
argued; its COST has never been measured. The sibling v0.8 cycle built the same stem test for FLOAT
claims and killed it (`V08_COVERAGE_DESTRUCTIVE`): 30 of 40 hand-scored demotions destroyed a
GENUINE binding, because scientific prose names a measurement narratively while the receipt field
holding it is structural. Nothing about that structural argument is specific to floats.

This measures the integer half against the obvious counterfactual: **integers keep value-only
matching, exactly as floats do today.** Two arms on the identical frame:

  ON   the shipped verifier, byte-for-byte (the count-binding filter runs).
  OFF  the same verifier with the guard on the count-binding block gated to False, so an integer
       claim's `hits` are never filtered. Everything else — extraction, is_spec/is_hist/is_notation,
       the trigger registers, range-sanity, derived-percent, the v0.6.2 float preference, the v0.8
       clause (shipped OFF) and the whole status ladder — is untouched.

`styxx/certify.py` IS NOT EDITED. The OFF arm is produced by loading the verifier's own source into
a second in-memory module with ONE textual substitution on the guard line, asserted to apply exactly
once. The ON arm of that same in-memory copy is then compared, ledger entry by ledger entry, against
the LIVE `styxx.certify.certify_doc` over all resolvable documents; any difference voids the run.
That comparison is the positive control: it proves the OFF arm differs from the shipped verifier in
the count-binding filter and in nothing else.

Reported transitions are written OFF -> ON, i.e. counterfactual -> live:

  VERIFIED -> UNGROUNDED   the filter ACCUSES a claim that would otherwise verify. These are live
                           accusations the shipped verifier makes today that a value-only regime
                           would not make. Enumerated completely.
  VERIFIED -> ABSTAIN      the filter SILENCES a claim that would otherwise verify.

and the decisive verdict question: how many documents carry an OATH-FAILED verdict caused SOLELY by
a count-binding accusation (FAILED with the filter on, HELD with it off).

  python papers/closed-model-frontier/oath_v09_intbind_cost_census.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import styxx.certify as live                                              # noqa: E402
from styxx.certify import _TRIGGERS, _match, receipt_values               # noqa: E402
from styxx.corpus_audit import _resolve_receipts                          # noqa: E402

OUT = HERE / "oath_v09_intbind_cost_census.json"
CERTIFY_SRC = ROOT / "styxx" / "certify.py"

# the v0.3 count-binding guard, verbatim. The substitution below is asserted to apply exactly once;
# if this line ever moves, the script dies loudly rather than measuring the wrong thing.
GUARD = '        if num["decimals"] == 0 and hits:'
GUARD_PATCHED = '        if V09_COUNT_BINDING and num["decimals"] == 0 and hits:'


def load_variant():
    """The verifier's own source, with the count-binding guard gated by a module-level flag."""
    src = CERTIFY_SRC.read_text(encoding="utf-8")
    n = src.count(GUARD)
    if n != 1:
        raise SystemExit(f"VOID: count-binding guard found {n} times in certify.py, expected 1")
    patched = src.replace(GUARD, GUARD_PATCHED)
    mod = types.ModuleType("certify_v09_variant")
    mod.__file__ = str(CERTIFY_SRC)
    mod.__dict__["V09_COUNT_BINDING"] = True
    exec(compile(patched, str(CERTIFY_SRC), "exec"), mod.__dict__)
    return mod


def resolvable_docs():
    """Documents under papers/** with a certificate whose recorded receipts ALL resolve."""
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
            out.append((doc, receipts, rec.get("verdict")))
    return out


# A leaf whose path's trailing subscript EQUALS its own value: `per_item[3].i` == 3.0. Such a leaf
# value-matches the integer 3 by construction and carries no information about any claim; a corpus
# with per-item arrays therefore offers a free coincidental match to every small integer. This is a
# purely mechanical test on the leaf, decided without reading the document.
_SELF_IDX = re.compile(r"\[(\d+)\](?:\.(?:i|idx|index|item|rank|pos|position))?$", re.I)


def is_self_index_leaf(path: str, value: float) -> bool:
    m = _SELF_IDX.search(path)
    return bool(m) and float(m.group(1)) == value


# the verifier's OWN count-field vocabulary, lifted verbatim from the slash-pair branch of the v0.3
# rule — the fields it already accepts as "a count's plausible home". Not a new vocabulary.
_COUNT_LIKE = re.compile(r"(^|[._\[])n_|n_held|n_caved|^n(\.|$)|count", re.I)


def claim_context(doc_lines, entry):
    """`ctx` / `bctx` / `pre` / `post` exactly as `certify_doc` builds them (v0.3 + v0.6.2)."""
    ctx = doc_lines[entry["line"] - 1].strip().replace("−", "-")
    bctx = entry.get("binding_context", ctx)
    tok_at = ctx.find(entry["token"])
    pre = ctx[max(0, tok_at - 18):tok_at] if tok_at >= 0 else ""
    if 0 <= tok_at < 18 and entry["line"] >= 2:
        pre = (doc_lines[entry["line"] - 2].strip().replace("−", "-")[-(18 - tok_at):]
               + " " + pre).strip()[-24:]
    post = ctx[tok_at + len(entry["token"]):] if tok_at >= 0 else ""
    return ctx, bctx, pre, post


def make_path_ok(bctx: str, pre: str):
    """The SHIPPED v0.3 `path_ok` predicate, replicated verbatim from `certify_doc`."""
    words = {w.lower().strip("'’") for w in re.findall(r"[A-Za-z][A-Za-z_-]{2,}", bctx)}
    stems = {w[:4] for w in words} | {s[:4] for w in words
                                      for s in re.split(r"[-_]", w) if len(s) >= 3}
    is_n_eq = bool(re.search(r"\bn\s*=\s*$", pre, re.I))
    stems |= {d for d in re.findall(r"\d{2,}", bctx)}

    def path_ok(p):
        segs = {s.lower() for seg in re.split(r"[.\[\]]", p) for s in re.split(r"[-_]", seg) if s}
        pst = {s[:4] for s in segs if len(s) >= 3} | {m for s in segs
                                                      for m in re.findall(r"\d{2,}", s)}
        return bool(pst & stems) or (is_n_eq and any(s == "n" or s.startswith("n_") for s in segs))

    return path_ok


def rvals_for(receipts):
    rv = []
    for rp in receipts:
        j = json.loads(rp.read_text(encoding="utf-8"))
        for path, v in receipt_values(j):
            rv.append((rp.name, path, v))
    return rv


def magnitude_bucket(value: float) -> str:
    a = abs(value)
    if a < 10:
        return "0-9"
    if a < 100:
        return "10-99"
    if a < 1000:
        return "100-999"
    if a < 10000:
        return "1000-9999"
    return ">=10000"


def key_of(e):
    return (e["line"], e["token"], e["value"])


def ledger_pairs(off_led, on_led):
    """Positional pairing. Extraction is identical in both arms; assert it, do not assume it."""
    if len(off_led) != len(on_led):
        raise SystemExit("VOID: ledger lengths differ between arms")
    for a, b in zip(off_led, on_led):
        if key_of(a) != key_of(b):
            raise SystemExit(f"VOID: ledger entries misaligned: {key_of(a)} vs {key_of(b)}")
    return list(zip(off_led, on_led))


def main() -> int:
    t0 = time.time()
    var = load_variant()
    docs = resolvable_docs()
    print(f"docs with fully-resolvable receipts: {len(docs)}", flush=True)

    control_entries = 0
    control_status_diffs = []
    control_ref_diffs = []

    transitions = Counter()
    ref_only_changes = 0
    int_claims = 0
    int_claims_verified_off = 0
    accused_rows = []      # VERIFIED -> UNGROUNDED, the alarming set
    silenced_rows = []     # VERIFIED -> ABSTAIN
    other_rows = []        # any transition neither of the above (recorded so none can hide)
    verdict_rows = []

    for i, (doc, receipts, committed_verdict) in enumerate(docs, 1):
        var.V09_COUNT_BINDING = True
        cert_on = var.certify_doc(doc, receipts)
        var.V09_COUNT_BINDING = False
        cert_off = var.certify_doc(doc, receipts)
        var.V09_COUNT_BINDING = True

        # ---- positive control: the ON arm must be the LIVE verifier, entry for entry
        cert_live = live.certify_doc(doc, receipts)
        if len(cert_live["ledger"]) != len(cert_on["ledger"]):
            raise SystemExit(f"VOID: live/replica ledger length differs on {doc.name}")
        for a, b in zip(cert_live["ledger"], cert_on["ledger"]):
            control_entries += 1
            if key_of(a) != key_of(b):
                raise SystemExit(f"VOID: live/replica misaligned on {doc.name}")
            if a["status"] != b["status"]:
                control_status_diffs.append({"doc": doc.name, "line": a["line"],
                                             "token": a["token"], "live": a["status"],
                                             "replica": b["status"]})
            if a["receipt_ref"] != b["receipt_ref"]:
                control_ref_diffs.append({"doc": doc.name, "line": a["line"],
                                          "token": a["token"], "live": a["receipt_ref"],
                                          "replica": b["receipt_ref"]})

        verdict_rows.append({"doc": doc.name, "verdict_on": cert_on["verdict"],
                             "verdict_off": cert_off["verdict"],
                             "ungrounded_on": cert_on["counts"]["UNGROUNDED"],
                             "ungrounded_off": cert_off["counts"]["UNGROUNDED"],
                             "committed_verdict": committed_verdict})

        doc_lines = doc.read_text(encoding="utf-8").splitlines()
        rvals = None
        leaf_value = None

        for off_e, on_e in ledger_pairs(cert_off["ledger"], cert_on["ledger"]):
            if off_e["decimals"] != 0:
                continue
            int_claims += 1
            if off_e["status"] == "VERIFIED":
                int_claims_verified_off += 1
            if off_e["status"] == on_e["status"]:
                if off_e["receipt_ref"] != on_e["receipt_ref"]:
                    ref_only_changes += 1
                continue

            trans = f'{off_e["status"]}->{on_e["status"]}'
            transitions[trans] += 1

            if rvals is None:
                rvals = rvals_for(receipts)
                leaf_value = {}
                for rn, pth, rv in rvals:
                    leaf_value.setdefault((rn, pth), rv)

            ctx, bctx, pre, _post = claim_context(doc_lines, off_e)
            allow_scaling = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
            hits = [(rn, pth) for rn, pth, rv in rvals
                    if _match(off_e["value"], 0, rv, allow_scaling)]
            # NAMEABLE (the v0.8 gate, applied to the integer rule): does the CITED RECEIPT SET
            # contain ANY leaf at all whose path this context names? If not, the filter could never
            # have kept a hit for this claim whatever its value, so the withdrawal is not a
            # judgement about this claim — it is the filter having no reachable home to bind to.
            _ok = make_path_ok(bctx, pre)
            nameable = any(_ok(pth) for _rn, pth, _rv in rvals)
            ref = off_e["receipt_ref"] or ""
            rn0, _, pth0 = ref.partition(":")
            n_self_idx = sum(1 for _rn, pth in hits
                             if is_self_index_leaf(pth, off_e["value"]))
            n_array = sum(1 for _rn, pth in hits if "[" in pth)
            count_like = [f"{rn}:{pth}" for rn, pth in hits if _COUNT_LIKE.search(pth)]
            row = {
                "doc": doc.name,
                "line": off_e["line"],
                "token": off_e["token"],
                "value": off_e["value"],
                "transition": trans,
                "context": ctx[:200],
                "binding_context": bctx[:320] if bctx != ctx else None,
                "would_have_matched": ref or None,
                "would_have_matched_value": leaf_value.get((rn0, pth0)),
                "n_hits_value_only": len(hits),
                "n_self_index_leaves": n_self_idx,
                "all_matches_self_index": bool(hits) and n_self_idx == len(hits),
                "all_matches_array_indexed": bool(hits) and n_array == len(hits),
                "nameable_leaf_exists_in_receipts": nameable,
                "count_like_leaves_matched": count_like[:6],
                "has_count_like_match": bool(count_like),
                "all_value_only_leaves": [f"{rn}:{pth}" for rn, pth in hits[:12]],
                "trigger_line": bool(_TRIGGERS.search(bctx)),
                "n_eq_pairing": bool(re.search(r"\bn\s*=\s*\d", bctx, re.I)),
                "magnitude": magnitude_bucket(off_e["value"]),
                "status_on": on_e["status"],
                "ref_on": on_e["receipt_ref"],
            }
            if trans == "VERIFIED->UNGROUNDED":
                accused_rows.append(row)
            elif trans == "VERIFIED->ABSTAIN":
                silenced_rows.append(row)
            else:
                other_rows.append(row)

        if i % 40 == 0:
            print(f"  [{i}/{len(docs)}] accused {len(accused_rows)} "
                  f"silenced {len(silenced_rows)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- verdicts that depend on the rule
    failed_on = [r for r in verdict_rows if r["verdict_on"] == "OATH-FAILED"]
    failed_off = [r for r in verdict_rows if r["verdict_off"] == "OATH-FAILED"]
    solely = [r for r in verdict_rows
              if r["verdict_on"] == "OATH-FAILED" and r["verdict_off"] == "OATH-HELD"]
    # a document the rule does not flip, but whose accusation COUNT it changes
    count_changed = [r for r in verdict_rows if r["ungrounded_on"] != r["ungrounded_off"]]

    def split(rows):
        return {
            "n": len(rows),
            "docs": len({r["doc"] for r in rows}),
            "by_magnitude": {k: v for k, v in
                             sorted(Counter(r["magnitude"] for r in rows).items())},
            "by_trigger_line": {"on_trigger_line": sum(1 for r in rows if r["trigger_line"]),
                                "off_trigger_line": sum(1 for r in rows if not r["trigger_line"])},
            "by_n_eq_pairing": {"n_eq_present": sum(1 for r in rows if r["n_eq_pairing"]),
                                "no_n_eq": sum(1 for r in rows if not r["n_eq_pairing"])},
            # what the value-only regime would have grounded these claims IN. A claim whose every
            # value-only match is a self-indexing array leaf was never earned under either regime:
            # the leaf equals its own subscript and matches that integer by construction.
            "grounded_only_in_self_index_leaves": sum(1 for r in rows
                                                      if r["all_matches_self_index"]),
            "grounded_only_in_array_indexed_leaves": sum(1 for r in rows
                                                         if r["all_matches_array_indexed"]),
            "grounded_in_some_named_scalar_leaf":
                sum(1 for r in rows if not r["all_matches_array_indexed"]),
            # a count claim whose value-only match sits on a leaf the verifier's OWN count-field
            # vocabulary recognizes, withdrawn anyway because the prose word and the path segment
            # do not share a 4-char stem.
            "withdrawn_from_a_count_like_leaf": sum(1 for r in rows if r["has_count_like_match"]),
            "nameable_leaf_exists": sum(1 for r in rows if r["nameable_leaf_exists_in_receipts"]),
            "unbindable_in_principle": sum(1 for r in rows
                                           if not r["nameable_leaf_exists_in_receipts"]),
            "top_docs": dict(Counter(r["doc"] for r in rows).most_common(12)),
        }

    report = {
        "note": "OATH v0.9 ANGLE 2 — measured COST of the SHIPPED v0.3 count-binding filter for "
                "INTEGER claims. Arms: ON = live verifier; OFF = same verifier with the "
                "count-binding guard gated False (integers keep value-only matching, like floats). "
                "styxx/certify.py is NOT edited; the OFF arm is an in-memory copy of its own source "
                "with one asserted substitution. Transitions read OFF->ON (counterfactual->live).",
        "verifier_sha256": hashlib.sha256(CERTIFY_SRC.read_bytes()).hexdigest(),
        "script": Path(__file__).name,
        "docs": len(docs),
        "positive_control": {
            "purpose": "the ON arm of the in-memory copy must be the LIVE verifier, entry for entry",
            "ledger_entries_compared": control_entries,
            "status_differences": len(control_status_diffs),
            "receipt_ref_differences": len(control_ref_diffs),
            "status_difference_rows": control_status_diffs[:20],
            "receipt_ref_difference_rows": control_ref_diffs[:20],
            "VALID": len(control_status_diffs) == 0 and len(control_ref_diffs) == 0,
        },
        "integer_claims_total": int_claims,
        "integer_claims_VERIFIED_without_the_filter": int_claims_verified_off,
        "changed_by_the_filter": sum(transitions.values()),
        "changed_share_of_integer_claims": round(sum(transitions.values())
                                                 / max(int_claims, 1), 4),
        "changed_share_of_value_only_verifications": round(sum(transitions.values())
                                                           / max(int_claims_verified_off, 1), 4),
        "transitions": dict(sorted(transitions.items())),
        "receipt_ref_only_changes": ref_only_changes,
        "verdicts": {
            "docs": len(verdict_rows),
            "OATH_FAILED_with_filter_ON_live": len(failed_on),
            "OATH_FAILED_with_filter_OFF": len(failed_off),
            "FAILED_solely_because_of_count_binding": len(solely),
            "docs_whose_UNGROUNDED_count_changes": len(count_changed),
            "solely_rows": solely,
            "ungrounded_count_changed_rows": count_changed,
        },
        "splits": {
            "VERIFIED->UNGROUNDED": split(accused_rows),
            "VERIFIED->ABSTAIN": split(silenced_rows),
            "other_transitions": split(other_rows),
        },
        "verified_to_ungrounded_rows": accused_rows,
        "verified_to_abstain_rows": silenced_rows,
        "other_transition_rows": other_rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    pc = report["positive_control"]
    print(f"\npositive control: {pc['ledger_entries_compared']} entries, "
          f"{pc['status_differences']} status diffs, {pc['receipt_ref_differences']} ref diffs "
          f"-> {'VALID' if pc['VALID'] else 'VOID'}")
    print(f"integer claims: {int_claims}  (VERIFIED without the filter: "
          f"{int_claims_verified_off})")
    print(f"changed by the filter: {sum(transitions.values())}")
    for k, v in sorted(transitions.items()):
        print(f"  {k:<26} {v}")
    print(f"receipt_ref-only changes (status identical): {ref_only_changes}")
    v = report["verdicts"]
    print(f"OATH-FAILED live {v['OATH_FAILED_with_filter_ON_live']} | "
          f"without the filter {v['OATH_FAILED_with_filter_OFF']} | "
          f"FAILED SOLELY because of count-binding: "
          f"{v['FAILED_solely_because_of_count_binding']}")
    for name in ("VERIFIED->UNGROUNDED", "VERIFIED->ABSTAIN"):
        s = report["splits"][name]
        print(f"{name}: n={s['n']} across {s['docs']} docs  "
              f"mag={s['by_magnitude']}  trigger={s['by_trigger_line']}")
        print(f"    grounded ONLY in self-index leaves: "
              f"{s['grounded_only_in_self_index_leaves']} | only in array leaves: "
              f"{s['grounded_only_in_array_indexed_leaves']} | some named scalar leaf: "
              f"{s['grounded_in_some_named_scalar_leaf']}")
        print(f"    NAMEABLE (a leaf the context names exists): {s['nameable_leaf_exists']} | "
              f"unbindable in principle: {s['unbindable_in_principle']}")
        print(f"    withdrawn from a COUNT-LIKE leaf (verifier's own n_*/count vocabulary): "
              f"{s['withdrawn_from_a_count_like_leaf']}")
    print(f"\nelapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
