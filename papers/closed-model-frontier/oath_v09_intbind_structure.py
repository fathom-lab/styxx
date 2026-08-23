"""OATH v0.9 — STRUCTURE: is the SHIPPED v0.3 count-binding stem test discriminating or decorative?

v0.8 killed status-level claim->field binding for FLOAT claims on `V08_COVERAGE_DESTRUCTIVE`: 30 of
40 hand-scored demotions destroyed a GENUINE binding, because scientific prose names a measurement
narratively ("whole-stack r=16: 0.616-0.626") while the receipt field holding it is structural
(`points[2].naive_relock_auroc`). The stated general lesson was that lexical binding between prose
and receipt field names is exhausted in this codebase.

The INTEGER version of that same test has been SHIPPED since v0.3 and its structure has never been
measured. This script asks whether integers differ STRUCTURALLY, or whether the same limitation
applies and the shipped filter only LOOKS like it works because almost everything passes it.

MEASUREMENT ONLY. `styxx/certify.py` is not touched: the count-binding predicate is replicated here
from the shipped source, on top of the module-level `_ctx_stems` / `_path_stems` helpers the inline
rule uses, and the replication is checked against the real ledger claim-for-claim (a
`replication_mismatches` of 0 is the licence to read anything else in this file).

What is measured, over every document under papers/** with a certificate whose receipts all resolve:

  BASE RATE       for each integer claim, the fraction of ALL receipt leaves in its own cited receipt
                  set that pass `path_ok` for that claim's binding context. If that fraction is high
                  the filter is nearly a no-op and any apparent success is a base-rate artifact.

  SURVIVAL        hits before the filter, hits after, and whether the filter was a NO-OP because
                  every value-match already passed.

  ATTRIBUTION     whether the filter changed WHICH leaf the certificate cites. A filter that removes
                  leaves but never changes the cited leaf or the status is decorative even when it
                  fires.

  CONTAINER       whether the surviving cited path is a specific measurement field or a generic
                  container (n / count / total / len / size / index / i / idx / seed / step).

  CAUSE           which stem actually produced the match — a topic word, a bare digit run (array
                  indices and 2+-digit numbers in prose are stems in the shipped rule), or the
                  `n=` -> `n_*` escape.

  BANDS           the same statistics for the small-integer band (|value| <= 20) against larger
                  integers, and — as the v0.8 structural control — the identical predicate applied
                  hypothetically to the FLOAT claims the shipped verifier certifies without it.

  python papers/closed-model-frontier/oath_v09_intbind_structure.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import (_ctx_stems, _match, _path_stems,   # noqa: E402
                           certify_doc, extract_numbers, receipt_values)
from styxx.corpus_audit import _resolve_receipts              # noqa: E402

OUT = HERE / "oath_v09_intbind_structure.json"

# the generic-container vocabulary named in the audit brief
GENERIC_SEGS = {"n", "count", "total", "len", "size", "index", "i", "idx", "seed", "step"}

_DIGIT_RUN = re.compile(r"\d{2,}")
_SEG_SPLIT = re.compile(r"[.\[\]]")


# ---------------------------------------------------------------- the shipped predicate, replicated
#
# certify.py lines ~427-445, verbatim in behaviour. `_ctx_stems` / `_path_stems` are IMPORTED (they
# are the module-level lift of the inline copies); the two things the inline rule adds on top of
# them are replicated here and nowhere else:
#   * `stems |= {d for d in re.findall(r"\d{2,}", bctx)}`      -- digit runs are context stems
#   * `pst |= {m for s in segs for m in re.findall(r"\d{2,}", s)}` -- and path stems
#   * the `is_n_eq` escape: a context ending in "n=" accepts any leaf whose path has an `n`/`n_*` seg

def ctx_stems_aug(bctx: str) -> tuple[frozenset, frozenset]:
    """(word stems, digit-run stems) of a binding context. Their union is the shipped `stems`."""
    return frozenset(_ctx_stems(bctx)), frozenset(_DIGIT_RUN.findall(bctx))


def path_feats(p: str) -> tuple[frozenset, frozenset, bool, str]:
    """(word stems, digit-run stems, has an n/n_* segment, terminal dot-level segment)."""
    segs = {s.lower() for seg in _SEG_SPLIT.split(p) for s in re.split(r"[-_]", seg) if s}
    digits = frozenset(m for s in segs for m in _DIGIT_RUN.findall(s))
    has_n = any(s == "n" or s.startswith("n_") for s in segs)
    dot_segs = [s for s in _SEG_SPLIT.split(p) if s and not s.isdigit()]
    terminal = dot_segs[-1].lower() if dot_segs else ""
    return frozenset(_path_stems(p)), digits, has_n, terminal


def path_ok(feat, cw: frozenset, cd: frozenset, is_n_eq: bool) -> bool:
    pw, pd, has_n, _t = feat
    return bool((pw & cw) or (pd & cd)) or (is_n_eq and has_n)


def match_cause(feat, cw: frozenset, cd: frozenset, is_n_eq: bool) -> tuple[str, list]:
    """Why this leaf survived: 'word' | 'digit' | 'n_eq' | 'none'. Ties report the word stems."""
    pw, pd, has_n, _t = feat
    w, d = sorted(pw & cw), sorted(pd & cd)
    if w:
        return ("word", w)
    if d:
        return ("digit", d)
    if is_n_eq and has_n:
        return ("n_eq", [])
    return ("none", [])


def container_class(terminal: str) -> str:
    """'generic' if the leaf's terminal key IS a container word, 'generic-prefixed' if its first
    underscore piece is one (n_held, count_total), else 'specific'."""
    if terminal in GENERIC_SEGS:
        return "generic"
    head = re.split(r"[-_]", terminal)[0] if terminal else ""
    if head in GENERIC_SEGS:
        return "generic-prefixed"
    return "specific"


# ---------------------------------------------------------------- corpus

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


def pct(xs: list[float], q: float) -> float:
    """Nearest-rank percentile; 0.0 on an empty sample (reported, never silently imputed)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[k])


def thinning_test(rows: list[dict], label: str) -> dict:
    """THE discriminative-power test, unconditioned on survival.

    Over every claim where the filter actually applies (not a slash-pair), compare the number of
    value-matches that survive `path_ok` against the number expected if `path_ok` were a RANDOM draw
    over the receipt set at that claim's own measured base rate. lift = observed / expected. A lift
    near 1.0 means the filter removes exactly what a coin weighted by the base rate would remove and
    carries no information about which leaf is the claim's home."""
    app = [r for r in rows if not r["slash_pair"]]
    if not app:
        return {"population": label, "n_claims": 0}
    obs = sum(r["n_kept"] for r in app)
    exp = sum(r["base_rate_leaves"] * r["n_hits_pre"] for r in app)
    above = sum(1 for r in app if r["n_kept"] > r["base_rate_leaves"] * r["n_hits_pre"])
    return {
        "population": label,
        "n_claims_where_filter_applies": len(app),
        "value_matches_before": sum(r["n_hits_pre"] for r in app),
        "observed_surviving": obs,
        "expected_surviving_at_base_rate": round(exp, 2),
        "lift_observed_over_expected": round(obs / exp, 4) if exp else None,
        "share_of_claims_above_their_own_expectation": round(above / len(app), 4),
    }


def summarize(rows: list[dict], label: str) -> dict:
    """Every headline statistic for one population."""
    n = len(rows)
    if not n:
        return {"population": label, "n": 0}
    base = [r["base_rate_leaves"] for r in rows]
    based = [r["base_rate_paths"] for r in rows]
    kept = [r["n_kept"] for r in rows]
    pre = [r["n_hits_pre"] for r in rows]
    noop = [r for r in rows if r["n_kept"] == r["n_hits_pre"]]
    changed = [r for r in rows if r["attribution_changed"]]
    reduced_to_one = [r for r in rows if r["n_kept"] == 1 and r["n_hits_pre"] > 1]
    return {
        "population": label,
        "n": n,
        "base_rate_over_all_leaves": {
            "mean": round(sum(base) / n, 4),
            "median": round(pct(base, 0.5), 4),
            "p10": round(pct(base, 0.10), 4), "p25": round(pct(base, 0.25), 4),
            "p75": round(pct(base, 0.75), 4), "p90": round(pct(base, 0.90), 4),
            "share_above_0.5": round(sum(1 for b in base if b > 0.5) / n, 4),
            "share_above_0.9": round(sum(1 for b in base if b > 0.9) / n, 4),
            "share_below_0.1": round(sum(1 for b in base if b < 0.1) / n, 4),
        },
        "base_rate_over_distinct_paths": {
            "mean": round(sum(based) / n, 4), "median": round(pct(based, 0.5), 4),
        },
        "hits": {
            "mean_hits_before_filter": round(sum(pre) / n, 4),
            "median_hits_before_filter": pct(pre, 0.5),
            "mean_surviving_hits": round(sum(kept) / n, 4),
            "median_surviving_hits": pct(kept, 0.5),
            "p90_surviving_hits": pct(kept, 0.90),
            "max_surviving_hits": max(kept),
        },
        "discrimination": {
            "noop_share_filter_removed_nothing": round(len(noop) / n, 4),
            "noop_count": len(noop),
            "attribution_changed_share": round(len(changed) / n, 4),
            "attribution_changed_count": len(changed),
            "reduced_to_single_leaf_share": round(len(reduced_to_one) / n, 4),
            "surviving_hits_gt_1_share": round(sum(1 for k in kept if k > 1) / n, 4),
        },
        "container_class_of_cited_leaf": dict(Counter(r["container"] for r in rows).most_common()),
        "match_cause": dict(Counter(r["cause"] for r in rows).most_common()),
        "slash_pair_share": round(sum(1 for r in rows if r["slash_pair"]) / n, 4),
        "is_n_eq_share": round(sum(1 for r in rows if r["is_n_eq"]) / n, 4),
        "ctx_stem_set_size_median": pct([r["n_ctx_stems"] for r in rows], 0.5),
    }


def claim_rows(doc: Path, receipts: list[Path]) -> tuple[list[dict], int, int]:
    """Every integer and float claim in *doc* with its pre/post-filter hit structure.

    Returns (rows, n_ledger_entries, n_replication_mismatches)."""
    text = doc.read_text(encoding="utf-8")
    doc_lines = text.splitlines()
    rvals: list[tuple[str, str, float]] = []
    for rp in receipts:
        j = json.loads(rp.read_text(encoding="utf-8"))
        for path, v in receipt_values(j):
            rvals.append((rp.name, path, v))
    if not rvals:
        return [], 0, 0

    feats: dict[str, tuple] = {}
    for _rn, pth, _v in rvals:
        if pth not in feats:
            feats[pth] = path_feats(pth)
    leaf_feats = [feats[pth] for _rn, pth, _v in rvals]        # base-rate denominator = ALL leaves
    distinct_feats = list(feats.values())

    cert = certify_doc(doc, receipts)
    nums = extract_numbers(text)
    ledger = cert["ledger"]
    if len(ledger) != len(nums):                                # positional 1:1 is the whole method
        return [], len(ledger), len(ledger)

    base_cache: dict[tuple, tuple[float, float]] = {}
    rows, mismatches = [], 0
    for num, entry in zip(nums, ledger):
        # ---- ctx / bctx / pre / post, replicated from certify_doc verbatim
        ctx = doc_lines[num["line"] - 1].strip().replace("−", "-")
        bctx = num.get("binding_context", ctx)
        tok_at = ctx.find(num["token"])
        pre = ctx[max(0, tok_at - 18):tok_at] if tok_at >= 0 else ""
        if 0 <= tok_at < 18 and num["line"] >= 2:
            pre = (doc_lines[num["line"] - 2].strip().replace("−", "-")[-(18 - tok_at):]
                   + " " + pre).strip()[-24:]
        post = ctx[tok_at + len(num["token"]):] if tok_at >= 0 else ""
        allow_scaling = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None

        hits_pre = [(rn, pth) for rn, pth, rv in rvals
                    if _match(num["value"], num["decimals"], rv, allow_scaling)]
        if not hits_pre:
            continue

        cw, cd = ctx_stems_aug(bctx)
        is_n_eq = bool(re.search(r"\bn\s*=\s*$", pre, re.I))
        slash_pair = bool(re.search(r"/\s*$", pre)) or bool(re.match(r"\s*/", post))

        # ---- the shipped filter (integers) / the same predicate hypothetically (floats)
        passing = [(rn, pth) for rn, pth in hits_pre if path_ok(feats[pth], cw, cd, is_n_eq)]
        if not slash_pair:
            hits_post = passing
        elif passing:
            hits_post = hits_pre                                # slash-pair keeps value-only matching
        else:
            hits_post = [(rn, pth) for rn, pth in hits_pre
                         if re.search(r"(^|[._\[])n_|n_held|n_caved|^n(\.|$)|count", pth, re.I)]

        is_int = num["decimals"] == 0
        if is_int and entry["status"] == "VERIFIED" \
                and not str(entry.get("receipt_ref") or "").startswith("derived:"):
            want = f"{hits_post[0][0]}:{hits_post[0][1]}" if hits_post else None
            if want != entry.get("receipt_ref"):
                mismatches += 1

        # ---- base rate: what share of the cited receipt set would pass path_ok for THIS context
        key = (cw, cd, is_n_eq)
        if key not in base_cache:
            nl = sum(1 for f in leaf_feats if path_ok(f, cw, cd, is_n_eq))
            npth = sum(1 for f in distinct_feats if path_ok(f, cw, cd, is_n_eq))
            base_cache[key] = (nl / len(leaf_feats), npth / len(distinct_feats))
        br_leaves, br_paths = base_cache[key]

        cited = hits_post[0][1] if hits_post else None
        cause, stems_hit = ("none", [])
        container = "none"
        if cited is not None:
            cause, stems_hit = match_cause(feats[cited], cw, cd, is_n_eq)
            container = container_class(feats[cited][3])

        rows.append({
            "doc": doc.name, "line": num["line"], "token": num["token"],
            "value": num["value"], "decimals": num["decimals"],
            "abs_le_20": abs(num["value"]) <= 20,
            "status": entry["status"],
            "n_leaves": len(leaf_feats), "n_distinct_paths": len(distinct_feats),
            "n_hits_pre": len(hits_pre), "n_kept": len(hits_post),
            "emptied": not hits_post,
            "attribution_changed": bool(hits_post) and hits_post[0] != hits_pre[0],
            "base_rate_leaves": round(br_leaves, 6),
            "base_rate_paths": round(br_paths, 6),
            "n_ctx_stems": len(cw) + len(cd),
            "is_n_eq": is_n_eq, "slash_pair": slash_pair,
            "cause": cause, "stems_hit": stems_hit[:4],
            "cited_path": cited, "container": container,
            "terminal": feats[cited][3] if cited is not None else None,
            "context": ctx[:120],
        })
    return rows, len(ledger), mismatches


def main() -> int:
    t0 = time.time()
    docs = resolvable_docs()
    print(f"docs with fully-resolvable receipts: {len(docs)}", flush=True)

    all_rows: list[dict] = []
    mismatches, failed_docs = 0, []
    for i, (doc, receipts) in enumerate(docs, 1):
        try:
            rows, _n, mm = claim_rows(doc, receipts)
        except Exception as exc:                                # noqa: BLE001 — reported, not hidden
            failed_docs.append(f"{doc.name}: {type(exc).__name__}: {exc}")
            continue
        all_rows.extend(rows)
        mismatches += mm
        if i % 25 == 0:
            print(f"  [{i}/{len(docs)}] claims with value-matches: {len(all_rows)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ints = [r for r in all_rows if r["decimals"] == 0]
    floats = [r for r in all_rows if r["decimals"] > 0]
    int_ver = [r for r in ints if r["status"] == "VERIFIED"]
    flt_ver = [r for r in floats if r["status"] == "VERIFIED"]

    # per-document base rate, so the headline is not carried by one pathological receipt set
    per_doc = {}
    for r in int_ver:
        per_doc.setdefault(r["doc"], []).append(r["base_rate_leaves"])
    doc_medians = sorted(round(pct(v, 0.5), 4) for v in per_doc.values())

    int_base_mean = round(sum(r["base_rate_leaves"] for r in int_ver) / max(len(int_ver), 1), 4)
    flt_base_mean = round(sum(r["base_rate_leaves"] for r in flt_ver) / max(len(flt_ver), 1), 4)

    report = {
        "note": "OATH v0.9 ANGLE 3 (structure) — is the shipped v0.3 integer count-binding stem "
                "test discriminating or decorative? MEASUREMENT ONLY; styxx/certify.py untouched.",
        "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "docs_resolvable": len(docs),
        "docs_failed": failed_docs,
        "claims_with_value_matches": len(all_rows),

        "replication_fidelity": {
            "definition": "for every INTEGER claim the real ledger calls VERIFIED (non-derived), "
                          "this script's post-filter hits[0] must equal the ledger's receipt_ref",
            "integer_verified_checked": len(int_ver),
            "mismatches": mismatches,
            "faithful": mismatches == 0,
        },

        "populations": {
            "integer_VERIFIED": summarize(int_ver, "integer claims the verifier certifies VERIFIED"),
            "integer_VERIFIED_small_band_abs_le_20": summarize(
                [r for r in int_ver if r["abs_le_20"]], "integer VERIFIED, |value| <= 20"),
            "integer_VERIFIED_large_band_abs_gt_20": summarize(
                [r for r in int_ver if not r["abs_le_20"]], "integer VERIFIED, |value| > 20"),
            "float_VERIFIED_CONTROL": summarize(
                flt_ver, "float claims VERIFIED — the SAME predicate applied hypothetically; the "
                         "shipped verifier does NOT filter floats (v0.8 CLOSED_NEGATIVE)"),
        },

        "filter_reach_over_ALL_integer_claims_with_value_matches": {
            "n": len(ints),
            "emptied_by_filter": sum(1 for r in ints if r["emptied"]),
            "emptied_share": round(sum(1 for r in ints if r["emptied"]) / max(len(ints), 1), 4),
            "emptied_landing": dict(Counter(r["status"] for r in ints if r["emptied"]).most_common()),
            "removed_nothing": sum(1 for r in ints if r["n_kept"] == r["n_hits_pre"]),
            "removed_nothing_share": round(
                sum(1 for r in ints if r["n_kept"] == r["n_hits_pre"]) / max(len(ints), 1), 4),
        },

        "random_thinning_test": {
            "integer_all_with_value_matches": thinning_test(ints, "all integer claims"),
            "integer_small_band_abs_le_20": thinning_test(
                [r for r in ints if r["abs_le_20"]], "integer, |value| <= 20"),
            "integer_large_band_abs_gt_20": thinning_test(
                [r for r in ints if not r["abs_le_20"]], "integer, |value| > 20"),
            "float_CONTROL_all_with_value_matches": thinning_test(
                floats, "all float claims — predicate NOT applied by the shipped verifier"),
        },

        "base_rate_integer_vs_float_CONTROL": {
            "note": "the same predicate, the same corpus, the two populations v0.8 separated. If "
                    "these are equal, integers do not differ STRUCTURALLY from the floats the v0.8 "
                    "adjudication found the test could not bind.",
            "integer_VERIFIED_mean": int_base_mean,
            "float_VERIFIED_mean": flt_base_mean,
            "absolute_difference": round(abs(int_base_mean - flt_base_mean), 4),
        },

        "subrule_reach_over_ALL_integer_claims_with_value_matches": {
            "note": "the v0.3 rule has two named sub-rules besides the bare stem test. This counts "
                    "what each actually did on this corpus.",
            "is_n_eq": {
                "definition": "context ends in 'n=' -> any leaf with an n / n_* segment passes",
                "claims_where_it_fires": sum(1 for r in ints if r["is_n_eq"]),
                "of_those_VERIFIED": sum(1 for r in ints
                                         if r["is_n_eq"] and r["status"] == "VERIFIED"),
                "of_those_emptied": sum(1 for r in ints if r["is_n_eq"] and r["emptied"]),
                "claims_VERIFIED_via_the_n_eq_escape_alone": sum(
                    1 for r in int_ver if r["cause"] == "n_eq"),
                "status_breakdown": dict(Counter(r["status"] for r in ints
                                                 if r["is_n_eq"]).most_common()),
                "DEAD_BRANCH": "measured 0 VERIFIED of 63 firings, and the branch is unreachable "
                               "as an outcome-changer BY CONSTRUCTION, not by corpus accident: "
                               "is_n_eq requires `pre` to match r'\\bn\\s*=\\s*$', which ends `pre` "
                               "with '=' plus optional space -- and that is exactly what is_spec's "
                               "operator class r'[>=<...]\\s*\\+?$' matches. is_n_eq IMPLIES "
                               "is_spec, and is_spec wins the ladder before `hits` is consulted "
                               "(ABSTAIN, ref 'spec-or-historical', overwriting any receipt_ref). "
                               "No `pre` exists for which the branch can change an outcome.",
            },
            "slash_pair": {
                "definition": "token adjacent to '/' -> value-only matching kept if ANY hit passes, "
                              "else fall back to count-like fields only",
                "claims_where_it_fires": sum(1 for r in ints if r["slash_pair"]),
                "of_those_VERIFIED": sum(1 for r in ints
                                         if r["slash_pair"] and r["status"] == "VERIFIED"),
                "of_those_emptied": sum(1 for r in ints if r["slash_pair"] and r["emptied"]),
            },
            "digit_run_stems": {
                "definition": "2+-digit runs in prose and in path segments (array indices) are stems",
                "claims_VERIFIED_where_a_digit_run_was_the_only_cause": sum(
                    1 for r in int_ver if r["cause"] == "digit"),
            },
        },

        "cause_none_breakdown_integer_VERIFIED": {
            "note": "cited leaf carries no stem the context names — only reachable via the "
                    "slash-pair branch, which keeps value-only matching when ANY hit passes",
            "n": sum(1 for r in int_ver if r["cause"] == "none"),
            "of_which_slash_pair": sum(1 for r in int_ver
                                       if r["cause"] == "none" and r["slash_pair"]),
        },

        "per_document_base_rate_integer_VERIFIED": {
            "docs": len(doc_medians),
            "median_of_document_medians": round(pct(doc_medians, 0.5), 4),
            "p10": round(pct(doc_medians, 0.10), 4), "p25": round(pct(doc_medians, 0.25), 4),
            "p75": round(pct(doc_medians, 0.75), 4), "p90": round(pct(doc_medians, 0.90), 4),
            "docs_above_0.5": sum(1 for d in doc_medians if d > 0.5),
            "docs_above_0.9": sum(1 for d in doc_medians if d > 0.9),
            "document_medians": doc_medians,
        },

        "top_terminal_segments_of_cited_leaf_integer_VERIFIED": dict(
            Counter(r["terminal"] for r in int_ver).most_common(25)),
        "top_matching_stems_integer_VERIFIED": dict(
            Counter(s for r in int_ver for s in r["stems_hit"]).most_common(30)),
        "top_matching_stems_float_VERIFIED_CONTROL": dict(
            Counter(s for r in flt_ver for s in r["stems_hit"]).most_common(30)),

        "rows_integer_VERIFIED": int_ver,
        "rows_integer_emptied_by_filter": [r for r in ints if r["emptied"]],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    p = report["populations"]["integer_VERIFIED"]
    pf = report["populations"]["float_VERIFIED_CONTROL"]
    ps = report["populations"]["integer_VERIFIED_small_band_abs_le_20"]
    pl = report["populations"]["integer_VERIFIED_large_band_abs_gt_20"]
    print(f"\nreplication mismatches: {mismatches} (must be 0)")
    print(f"integer VERIFIED claims: {p['n']}   float VERIFIED control: {pf['n']}")
    print("BASE RATE (share of ALL leaves passing path_ok for the claim's own context)")
    print(f"  integer  mean {p['base_rate_over_all_leaves']['mean']}  "
          f"median {p['base_rate_over_all_leaves']['median']}  "
          f">0.5 in {p['base_rate_over_all_leaves']['share_above_0.5']} of claims")
    print(f"  float    mean {pf['base_rate_over_all_leaves']['mean']}  "
          f"median {pf['base_rate_over_all_leaves']['median']}")
    print(f"  per-document median of medians: "
          f"{report['per_document_base_rate_integer_VERIFIED']['median_of_document_medians']}")
    print(f"SURVIVAL  mean hits before {p['hits']['mean_hits_before_filter']}  "
          f"after {p['hits']['mean_surviving_hits']}")
    print(f"NO-OP     filter removed nothing in {p['discrimination']['noop_share_filter_removed_nothing']} "
          f"of integer VERIFIED claims ({p['discrimination']['noop_count']}/{p['n']})")
    print(f"ATTRIB    cited leaf changed in {p['discrimination']['attribution_changed_share']}")
    print(f"CONTAINER {p['container_class_of_cited_leaf']}")
    print(f"CAUSE     {p['match_cause']}")
    print(f"BANDS     |v|<=20 n={ps['n']} base {ps['base_rate_over_all_leaves']['mean']} "
          f"noop {ps['discrimination']['noop_share_filter_removed_nothing']} "
          f"hits {ps['hits']['mean_hits_before_filter']}->{ps['hits']['mean_surviving_hits']}")
    print(f"          |v|> 20 n={pl['n']} base {pl['base_rate_over_all_leaves']['mean']} "
          f"noop {pl['discrimination']['noop_share_filter_removed_nothing']} "
          f"hits {pl['hits']['mean_hits_before_filter']}->{pl['hits']['mean_surviving_hits']}")
    fr = report["filter_reach_over_ALL_integer_claims_with_value_matches"]
    print(f"REACH     {fr['emptied_by_filter']}/{fr['n']} integer claims with value-matches "
          f"were emptied ({fr['emptied_share']}) -> {fr['emptied_landing']}")
    sr = report["subrule_reach_over_ALL_integer_claims_with_value_matches"]
    print(f"SUBRULES  n=  fires {sr['is_n_eq']['claims_where_it_fires']}  "
          f"VERIFIED {sr['is_n_eq']['of_those_VERIFIED']}  "
          f"via the escape alone {sr['is_n_eq']['claims_VERIFIED_via_the_n_eq_escape_alone']}")
    print(f"          /   fires {sr['slash_pair']['claims_where_it_fires']}  "
          f"VERIFIED {sr['slash_pair']['of_those_VERIFIED']}  "
          f"emptied {sr['slash_pair']['of_those_emptied']}")
    print("THINNING  observed surviving vs expected under a random draw at the same base rate")
    for k, v in report["random_thinning_test"].items():
        if v.get("n_claims_where_filter_applies"):
            print(f"  {k:38s} n={v['n_claims_where_filter_applies']:5d}  "
                  f"obs {v['observed_surviving']:6d}  exp {v['expected_surviving_at_base_rate']:9.1f}"
                  f"  lift {v['lift_observed_over_expected']}")
    print(f"\nelapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
