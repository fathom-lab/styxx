"""OATH v0.9 ANGLE-1 census — why does the SHIPPED v0.3 count-binding filter fail to discriminate?

`oath_v07_silentpass_census.json` mutates one significant digit of every claim the shipped verifier
certifies VERIFIED and re-certifies. 604 mutants come back VERIFIED — affirmative false attestation,
the channel `PREREG_oath_v08_float_field_binding_2026_08_23.md` attacked for FLOATS and closed
CLOSED_NEGATIVE (`V08_COVERAGE_DESTRUCTIVE`). **274 of the 604 are bare integers**, and every one of
them already passed through the v0.3 count-binding stem filter, which has shipped since v0.3 and
whose recall has never been measured.

This script characterises HOW they got through. For each of the 274 it reconstructs the mutated
line, recomputes `hits` and replays the count-binding block from `certify_doc` verbatim, recording:

  * how many hits survived `path_ok`, and the surviving receipt paths;
  * WHICH sub-rule admitted the cited hit —
      `stem_overlap`         ordinary 4-char path/context stem collision;
      `n_eq_pairing`         the `n=` pairing (`is_n_eq` + an `n` / `n_*` path segment);
      `slash_pair_bypass`    a token adjacent to '/' where SOME hit passes `path_ok`, so the filter
                             is not applied AT ALL and every hit is kept, path_ok or not;
      `slash_pair_countlike` a slash pair where no hit passes `path_ok`, kept by the n_*/count regex;
  * whether the filter was a NO-OP for this claim (`n_kept == n_hits`) — the filter was never
    discriminating here — and whether the surviving leaf is the same leaf that grounds the ORIGINAL
    unmutated token;
  * the colliding stem(s): which context word and which path segment shared a 4-char prefix;
  * the magnitude of the mutated token, since small integers collide trivially.

**MEASUREMENT ONLY.** `styxx/certify.py` is not touched, imported flags are not overridden, and the
count-binding logic is REPLAYED here rather than edited there. `_ctx_stems` / `_path_stems` are
imported from the verifier rather than reimplemented (they are the same vocabulary the inline rule
uses); the inline rule's two additions over them — the `\\d{2,}` digit-run stems and the `is_n_eq`
n-segment escape — are replicated explicitly and marked as such.

Fidelity is self-checked, not assumed: every reconstructed mutant is also re-certified through the
real `certify_doc`, and the reconstructed cited reference is compared byte-for-byte against the
verifier's own `receipt_ref`. A reconstruction that does not reproduce the verifier is reported as
a miss, not silently dropped.

Non-destructive: mutants live in temp files; the only file written is this script's result JSON.

  python papers/closed-model-frontier/oath_v09_intbind_recall_census.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import (_ctx_stems, _match, _path_stems, certify_doc,   # noqa: E402
                           extract_numbers, receipt_values)
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v09_intbind_recall_census.json"
V07_CENSUS = HERE / "oath_v07_silentpass_census.json"

# the count-like fallback of the slash_pair branch, verbatim from certify_doc
_COUNTLIKE = re.compile(r"(^|[._\[])n_|n_held|n_caved|^n(\.|$)|count", re.I)
_WORDS = re.compile(r"[A-Za-z][A-Za-z_-]{2,}")


# ---------------------------------------------------------------- the v0.3 rule, replayed

def ctx_stems_v03(bctx: str) -> set[str]:
    """The inline v0.3 context stem set: `_ctx_stems` PLUS the digit-runs the inline rule adds
    (`stems |= {d for d in re.findall(r'\\d{2,}', bctx)}`)."""
    return _ctx_stems(bctx) | set(re.findall(r"\d{2,}", bctx))


def path_stems_v03(path: str) -> set[str]:
    """The inline v0.3 path stem set: `_path_stems` PLUS digit-runs glued to path segments
    (`{m for s in segs for m in re.findall(r'\\d{2,}', s)}`)."""
    segs = {s.lower() for seg in re.split(r"[.\[\]]", path) for s in re.split(r"[-_]", seg) if s}
    return _path_stems(path) | {m for s in segs for m in re.findall(r"\d{2,}", s)}


def path_segments(path: str) -> set[str]:
    return {s.lower() for seg in re.split(r"[.\[\]]", path) for s in re.split(r"[-_]", seg) if s}


def n_seg_ok(path: str) -> bool:
    """The `is_n_eq` escape of `path_ok`: any path segment equal to 'n' or starting 'n_'."""
    segs = path_segments(path)
    return any(s == "n" or s.startswith("n_") for s in segs)


def apply_count_binding(hits, bctx: str, pre: str, post: str):
    """Replay the v0.3 COUNT-BINDING block of `certify_doc` and report HOW it decided.

    Returns (kept_hits, detail). `hits` is a list of (receipt_name, path)."""
    stems = ctx_stems_v03(bctx)
    is_n_eq = bool(re.search(r"\bn\s*=\s*$", pre, re.I))

    def path_ok(p: str) -> bool:
        return bool(path_stems_v03(p) & stems) or (is_n_eq and n_seg_ok(p))

    slash_pair = bool(re.search(r"/\s*$", pre)) or bool(re.match(r"\s*/", post))
    any_ok = any(path_ok(p) for _, p in hits)

    if not slash_pair:
        kept = [(rn, pth) for rn, pth in hits if path_ok(pth)]
        branch = "filtered"
    elif not any_ok:
        kept = [(rn, pth) for rn, pth in hits if _COUNTLIKE.search(pth)]
        branch = "slash_pair_countlike"
    else:
        kept = list(hits)          # NOT filtered at all — the slash-pair bypass
        branch = "slash_pair_bypass"

    detail = {"stems": stems, "is_n_eq": is_n_eq, "slash_pair": slash_pair,
              "branch": branch, "path_ok": path_ok}
    return kept, detail


def subrule_for(path: str, detail) -> str:
    """Which sub-rule admitted this surviving hit."""
    if detail["branch"] == "slash_pair_bypass":
        return "slash_pair_bypass"
    if detail["branch"] == "slash_pair_countlike":
        return "slash_pair_countlike"
    if path_stems_v03(path) & detail["stems"]:
        return "stem_overlap"
    if detail["is_n_eq"] and n_seg_ok(path):
        return "n_eq_pairing"
    return "unclassified"          # cannot happen for a survivor of the `filtered` branch


def collision_detail(path: str, bctx: str, stems: set[str]):
    """Which context word and which path segment shared a 4-char prefix."""
    coll = sorted(path_stems_v03(path) & stems)
    words = sorted({w.lower().strip("'’") for w in _WORDS.findall(bctx)})
    segs = sorted(path_segments(path))
    rows = []
    for s in coll:
        cw = sorted({w for w in words if w[:4] == s or any(
            p[:4] == s for p in re.split(r"[-_]", w) if len(p) >= 3)})
        if not cw and re.fullmatch(r"\d{2,}", s):
            cw = [s]               # a digit-run stem, not a word
        ps = sorted({g for g in segs if g[:4] == s or s in re.findall(r"\d{2,}", g)})
        rows.append({"stem": s, "is_digit_run": bool(re.fullmatch(r"\d{2,}", s)),
                     "context_words": cw[:6], "path_segments": ps[:6]})
    return coll, rows


# ---------------------------------------------------------------- corpus plumbing

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


def rvals_for(receipts) -> list[tuple[str, str, float]]:
    rv = []
    for rp in receipts:
        j = json.loads(rp.read_text(encoding="utf-8"))
        for path, v in receipt_values(j):
            rv.append((rp.name, path, v))
    return rv


def claim_frame(doc_lines, num):
    """Reproduce `certify_doc`'s per-token ctx / bctx / pre / post / allow_scaling, verbatim."""
    ctx = doc_lines[num["line"] - 1].strip().replace("−", "-")
    bctx = num.get("binding_context", ctx)
    tok_at = ctx.find(num["token"])
    pre = ctx[max(0, tok_at - 18):tok_at] if tok_at >= 0 else ""
    if 0 <= tok_at < 18 and num["line"] >= 2:
        pre = (doc_lines[num["line"] - 2].strip().replace("−", "-")[-(18 - tok_at):]
               + " " + pre).strip()[-24:]
    post = ctx[tok_at + len(num["token"]):] if tok_at >= 0 else ""
    allow_scaling = "%" in ctx or re.search(r"\bpercent", ctx, re.I) is not None
    return ctx, bctx, pre, post, allow_scaling


def sibling_family(path: str):
    """The path with its LAST array index replaced by '*' — `per_item[45].i` -> `per_item[*].i`.

    A count claim that lands on such a leaf lands on one member of an ENUMERATION. This is how the
    dominant failure is measured rather than asserted: if the family holds a dense run of integers,
    every small integer in its span value-matches something, and value-matching carries no
    information at all."""
    m = None
    for m in re.finditer(r"\[(\d+)\]", path):
        pass
    if m is None:
        return None
    return path[:m.start()] + "[*]" + path[m.end():]


def family_index(rvals):
    fams: dict[tuple[str, str], list[float]] = {}
    for rn, pth, rv in rvals:
        fam = sibling_family(pth)
        if fam is not None:
            fams.setdefault((rn, fam), []).append(rv)
    return fams


def cover_profile(fams, rn: str, path: str):
    """Is the cited leaf a member of a dense integer enumeration? Measured, not guessed."""
    fam = sibling_family(path)
    if fam is None:
        return {"family": None, "family_size": 0, "all_integers": False,
                "distinct_values": 0, "span": 0, "density": 0.0, "dense_cover": False}
    vals = fams.get((rn, fam), [])
    ints = [v for v in vals if float(v).is_integer()]
    all_int = bool(vals) and len(ints) == len(vals)
    span = int(max(ints) - min(ints)) + 1 if ints else 0
    distinct = len({int(v) for v in ints})
    density = round(distinct / span, 4) if span > 0 else 0.0
    return {"family": fam, "family_size": len(vals), "all_integers": all_int,
            "distinct_values": distinct, "span": span, "density": density,
            "dense_cover": bool(all_int and len(vals) >= 5 and density >= 0.9)}


def magnitude_bucket(v: float) -> str:
    a = abs(v)
    if a <= 20:
        return "0-20"
    if a <= 100:
        return "21-100"
    if a <= 1000:
        return "101-1000"
    return ">1000"


# ---------------------------------------------------------------- main

def main() -> int:
    t0 = time.time()
    v07 = json.loads(V07_CENSUS.read_text(encoding="utf-8"))
    fv = [r for r in v07["rows"] if r["status"] == "VERIFIED"]
    targets = [r for r in fv if r["decimals"] == 0]
    print(f"v0.7 false-VERIFIED: {len(fv)}  of which INTEGER: {len(targets)}", flush=True)

    docs = resolvable_docs()
    doc_index = {d.name: (d, rc) for d, rc in docs}
    print(f"docs with fully-resolvable receipts: {len(docs)}", flush=True)

    by_doc: dict[str, list] = {}
    for r in targets:
        by_doc.setdefault(r["doc"], []).append(r)

    rows, unreplayed = [], []
    for dname, drows in sorted(by_doc.items()):
        if dname not in doc_index:
            unreplayed += [{"doc": dname, "line": r["line"], "token": r["token"],
                            "why": "document not in resolvable frame"} for r in drows]
            continue
        doc, receipts = doc_index[dname]
        rvals = rvals_for(receipts)
        fams = family_index(rvals)
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()

        # the ORIGINAL claim's grounding, from the shipped verifier itself
        base = certify_doc(doc, receipts)
        base_ref = {(e["line"], e["token"]): e.get("receipt_ref")
                    for e in base["ledger"] if e["status"] == "VERIFIED"}

        for r in drows:
            ln = r["line"] - 1
            if ln >= len(lines) or r["token"] not in lines[ln]:
                unreplayed.append({"doc": dname, "line": r["line"], "token": r["token"],
                                   "why": "token no longer on its line"})
                continue
            ml = list(lines)
            ml[ln] = ml[ln].replace(r["token"], r["mutant"], 1)          # v0.7 census operator
            mtext = "\n".join(ml)
            nums = extract_numbers(mtext)
            num = next((e for e in nums
                        if e["line"] == r["line"] and e["token"] == r["mutant"]), None)
            if num is None:
                unreplayed.append({"doc": dname, "line": r["line"], "token": r["token"],
                                   "why": "mutant not extracted"})
                continue

            ctx, bctx, pre, post, allow_scaling = claim_frame(ml, num)
            hits = [(rn, pth) for rn, pth, rv in rvals
                    if _match(num["value"], num["decimals"], rv, allow_scaling)]
            kept, det = apply_count_binding(hits, bctx, pre, post)

            # ---- fidelity: re-certify the mutant through the real verifier
            with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                             encoding="utf-8") as tf:
                tf.write(mtext)
                tmp = Path(tf.name)
            try:
                mc = certify_doc(tmp, receipts)
            finally:
                tmp.unlink(missing_ok=True)
            mentry = next((e for e in mc["ledger"]
                           if e["line"] == r["line"] and e["token"] == r["mutant"]), None)
            v_status = mentry["status"] if mentry else "NOT_EXTRACTED"
            v_ref = mentry.get("receipt_ref") if mentry else None
            my_ref = f"{kept[0][0]}:{kept[0][1]}" if kept else None
            faithful = (v_status == "VERIFIED") and (my_ref == v_ref)

            # ---- the ORIGINAL, unmutated token: same frame, same filter
            onum = next((e for e in extract_numbers("\n".join(lines))
                         if e["line"] == r["line"] and e["token"] == r["token"]), None)
            okept = []
            if onum is not None:
                octx, obctx, opre, opost, oscale = claim_frame(lines, onum)
                ohits = [(rn, pth) for rn, pth, rv in rvals
                         if _match(onum["value"], onum["decimals"], rv, oscale)]
                okept, _odet = apply_count_binding(ohits, obctx, opre, opost)
            opaths = {p for _, p in okept}
            kpaths = {p for _, p in kept}
            orig_ref = base_ref.get((r["line"], r["token"]))

            sub = subrule_for(kept[0][1], det) if kept else "no_surviving_hit"
            coll, coll_rows = ((collision_detail(kept[0][1], bctx, det["stems"]))
                               if kept else ([], []))
            sub_all = sorted({subrule_for(p, det) for _, p in kept})
            cov = (cover_profile(fams, kept[0][0], kept[0][1]) if kept
                   else cover_profile(fams, "", ""))
            # WHERE in the path did the stem collide: the array CONTAINER name (everything left of
            # the first '['), or the terminal field that actually names a quantity?
            cpath = kept[0][1] if kept else ""
            head = cpath.split("[")[0]
            head_segs = path_segments(head)
            coll_in_container = any(g in head_segs for cr in coll_rows for g in cr["path_segments"])
            cited_fam = (kept[0][0], sibling_family(kept[0][1])) if kept else None
            hits_in_family = (sum(1 for rn, pth in hits
                                  if (rn, sibling_family(pth)) == cited_fam)
                              if cited_fam and cited_fam[1] else 0)

            rows.append({
                "doc": dname, "line": r["line"], "token": r["token"], "mutant": r["mutant"],
                "mutant_value": num["value"], "token_value": float(r["token"].replace(",", "")),
                "magnitude_bucket": magnitude_bucket(num["value"]),
                "n_hits": len(hits), "n_kept": len(kept),
                "filter_noop": len(kept) == len(hits),
                "surviving_paths": [f"{rn}:{pth}" for rn, pth in kept][:12],
                "subrule": sub, "subrules_all_survivors": sub_all,
                "branch": det["branch"], "is_n_eq": det["is_n_eq"],
                "slash_pair": det["slash_pair"],
                "colliding_stems": coll,
                "collisions": coll_rows[:6],
                "cited_is_array_indexed": bool(kept and "[" in kept[0][1]),
                "collision_in_container_segment": bool(coll_rows) and coll_in_container,
                "cover": cov, "hits_in_cited_family": hits_in_family,
                "original_ref": orig_ref,
                "original_kept_paths": sorted(opaths)[:12],
                "same_leaf_as_original": bool(orig_ref) and my_ref == orig_ref,
                "shares_leaf_with_original": bool(kpaths & opaths),
                "verifier_status": v_status, "verifier_ref": v_ref,
                "reconstruction_faithful": faithful,
                "context": ctx[:180],
            })
        print(f"  {dname}: {len(drows)} rows ({time.time()-t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- aggregate
    n = len(rows)
    faithful = [r for r in rows if r["reconstruction_faithful"]]
    # every sub-rule is reported, including the ones that fired ZERO times — an absent measurement
    # printed as an absent row is the defect this repo studies.
    sub_counts = Counter({k: 0 for k in ("stem_overlap", "n_eq_pairing",
                                         "slash_pair_bypass", "slash_pair_countlike")})
    sub_counts.update(r["subrule"] for r in rows)
    branch_counts = Counter(r["branch"] for r in rows)
    mag_counts = Counter(r["magnitude_bucket"] for r in rows)
    mag_counts_orig = Counter(magnitude_bucket(r["token_value"]) for r in rows)

    stem_freq = Counter()
    for r in rows:
        for s in r["colliding_stems"]:
            stem_freq[s] += 1
    digit_stems = {s for s in stem_freq if re.fullmatch(r"\d{2,}", s)}

    noop = [r for r in rows if r["filter_noop"]]
    same_leaf = [r for r in rows if r["same_leaf_as_original"]]
    shares_leaf = [r for r in rows if r["shares_leaf_with_original"]]
    one_hit = [r for r in rows if r["n_hits"] == 1]
    top5 = {e for e, _ in stem_freq.most_common(5)}
    coll_width = Counter(len(r["colliding_stems"]) for r in rows)

    report = {
        "note": "OATH v0.9 angle 1 — recall census of the SHIPPED v0.3 count-binding filter: how "
                "the 274 INTEGER false attestations of oath_v07_silentpass_census.json survive it. "
                "MEASUREMENT ONLY; styxx/certify.py unmodified.",
        "verifier_sha256": hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "source_census": V07_CENSUS.name,
        "source_census_seed": v07.get("seed"),
        "docs_in_frame": len(docs),
        "v07_false_verified_total": len(fv),
        "v07_false_verified_integer": len(targets),
        "replayed": n,
        "unreplayed": len(unreplayed),
        "reconstruction_faithful": len(faithful),
        "reconstruction_fidelity": round(len(faithful) / max(n, 1), 4),
        "subrule_counts": dict(sub_counts.most_common()),
        "branch_counts": dict(branch_counts.most_common()),
        "rows_with_is_n_eq": sum(1 for r in rows if r["is_n_eq"]),
        "rows_with_slash_pair": sum(1 for r in rows if r["slash_pair"]),
        "filter_noop": len(noop),
        "filter_noop_share": round(len(noop) / max(n, 1), 4),
        "filter_removed_some_but_survivors_remained": n - len(noop),
        "single_hit_claims": len(one_hit),
        "colliding_stem_count_per_row": {str(k): v for k, v in sorted(coll_width.items())},
        "rows_surviving_on_exactly_one_stem": coll_width.get(1, 0),
        "rows_touching_a_top5_stem": sum(1 for r in rows
                                         if set(r["colliding_stems"]) & top5),
        "docs_touched": len({r["doc"] for r in rows}),
        "top_docs": dict(Counter(r["doc"] for r in rows).most_common(10)),
        "cited_leaf_array_indexed": sum(1 for r in rows if r["cited_is_array_indexed"]),
        "collision_on_container_not_field": sum(
            1 for r in rows if r["collision_in_container_segment"]),
        "dense_cover_OR_container_collision": sum(
            1 for r in rows
            if r["cover"]["dense_cover"] or r["collision_in_container_segment"]),
        "cited_leaf_dense_integer_cover": sum(1 for r in rows if r["cover"]["dense_cover"]),
        "cited_leaf_dense_cover_share": round(
            sum(1 for r in rows if r["cover"]["dense_cover"]) / max(n, 1), 4),
        "cited_leaf_last_segment": dict(Counter(
            re.sub(r"[^A-Za-z0-9_]", "",
                   re.split(r"[.\[\]]", r["surviving_paths"][0].split(":", 1)[1].rstrip("]"))[-1])
            for r in rows if r["surviving_paths"]).most_common(15)),
        "same_leaf_as_original": len(same_leaf),
        "shares_leaf_with_original": len(shares_leaf),
        "magnitude_of_mutant": dict(mag_counts.most_common()),
        "magnitude_of_original_token": dict(mag_counts_orig.most_common()),
        "small_int_share_mutant": round(mag_counts["0-20"] / max(n, 1), 4),
        "distinct_colliding_stems": len(stem_freq),
        "digit_run_stems": sorted(digit_stems),
        "top_colliding_stems": [{"stem": s, "rows": c,
                                 "share": round(c / max(n, 1), 4),
                                 "is_digit_run": bool(re.fullmatch(r"\d{2,}", s))}
                                for s, c in stem_freq.most_common(20)],
        "hits_per_claim": {"min": min((r["n_hits"] for r in rows), default=0),
                           "median": sorted(r["n_hits"] for r in rows)[n // 2] if n else 0,
                           "max": max((r["n_hits"] for r in rows), default=0)},
        "kept_per_claim": {"min": min((r["n_kept"] for r in rows), default=0),
                           "median": sorted(r["n_kept"] for r in rows)[n // 2] if n else 0,
                           "max": max((r["n_kept"] for r in rows), default=0)},
        "unreplayed_rows": unreplayed,
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nreplayed {n}/{len(targets)}  (unreplayed {len(unreplayed)})")
    print(f"reconstruction fidelity vs the real verifier: {len(faithful)}/{n} "
          f"= {report['reconstruction_fidelity']}")
    print("\nsub-rule that admitted the cited hit:")
    for s, c in sub_counts.most_common():
        print(f"  {s:22s} {c:4d}  ({c/max(n,1):.4f})")
    print(f"\nfilter was a NO-OP (kept == hits): {len(noop)}/{n} = {report['filter_noop_share']}")
    print(f"claims with exactly ONE value-match: {len(one_hit)}/{n}")
    print(f"survivors resting on exactly ONE colliding stem: {coll_width.get(1, 0)}/{n}")
    print(f"rows whose collision uses a top-5 stem: {report['rows_touching_a_top5_stem']}/{n}")
    print(f"rows where the n= pairing was even live (is_n_eq): "
          f"{report['rows_with_is_n_eq']}/{n}")
    print(f"mutant grounds in the SAME leaf as the original claim: {len(same_leaf)}/{n}")
    print(f"mutant shares ANY surviving leaf with the original:    {len(shares_leaf)}/{n}")
    print(f"\ncited leaf is ARRAY-INDEXED: {report['cited_leaf_array_indexed']}/{n}")
    print(f"cited leaf sits in a DENSE INTEGER ENUMERATION: "
          f"{report['cited_leaf_dense_integer_cover']}/{n} "
          f"= {report['cited_leaf_dense_cover_share']}")
    print(f"stem collided on the array CONTAINER, not the field: "
          f"{report['collision_on_container_not_field']}/{n}")
    print(f"dense enumeration OR container collision: "
          f"{report['dense_cover_OR_container_collision']}/{n}")
    print(f"cited-leaf last path segment: {report['cited_leaf_last_segment']}")
    print(f"\nmagnitude of the mutated token: {report['magnitude_of_mutant']}")
    print(f"\ntop colliding stems ({len(stem_freq)} distinct):")
    for e in report["top_colliding_stems"]:
        print(f"  {e['stem']:>8s}  {e['rows']:4d}  ({e['share']:.4f})"
              f"{'  [digit-run]' if e['is_digit_run'] else ''}")
    print(f"\nelapsed {time.time()-t0:.1f}s -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
