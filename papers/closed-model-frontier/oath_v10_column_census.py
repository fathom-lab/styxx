"""OATH v0.10 pre-fix census — how far ``ctx.find(token)`` sits from the token it should anchor.

Runs at the SHIPPED verifier, before ``styxx/certify.py`` is touched, and imports its regexes
rather than copying them, so what it measures is the instrument and not a paraphrase of it.

``certify_doc`` locates each extracted token on its line with ``ctx.find(num["token"])`` — the
FIRST occurrence of the token STRING, which is not necessarily the occurrence ``extract_numbers``
actually extracted. Every downstream window (``pre``, ``post``) and therefore ``is_spec``,
``is_notation``, ``is_hist``, the range-sanity ``unit_kw``/``sign_kw`` tests, the slash-pair test
and the v0.5 derived-percent parse can be computed against a different token's neighbourhood.

The census reports four things:

  1. the misplacement rate repo-wide and inside the certified corpus;
  2. how many misplaced tokens have a downstream predicate that actually DIVERGES between the two
     anchors — the ones where the defect is not merely cosmetic;
  3. the scrub-offset check the repair depends on. ``extract_numbers`` blanks sha/date/version
     spans with ``re.sub(pat, " ", ...)``, which collapses each match to ONE space and is therefore
     NOT length-preserving: a raw ``m.start()`` is not the source column on any line carrying such
     a span, so carrying the match offset is only sound once the scrub preserves length. The census
     measures whether restoring length preservation changes what is extracted;
  4. the two named live instances, re-derived here rather than asserted.

  python papers/closed-model-frontier/oath_v10_column_census.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import styxx.certify as C                                                  # noqa: E402
from styxx.corpus_audit import _resolve_receipts                           # noqa: E402

OUT = HERE / "oath_v10_column_census.json"

# The two instances named in the defect report, re-derived here rather than quoted. The second is
# a MUTANT line: the escape it demonstrates only exists once the bar has been doctored, which is
# the state a tamper battery puts the document in.
NAMED_INSTANCES = [
    {"rel": "papers/disjoint-worlds/PREREG_b49_amplitude_reaudit_2026_08_07.md",
     "line": 23, "token": "5", "mutate": None,
     "claim": "the bar in the JSON value position anchors on the 5 inside b45, so pre cannot "
              "reach the value key and the v0.9 JSON-idiom spec clause never fires"},
    {"rel": "papers/agent-conscience/FINDING_cot_inward_powered_2026_07_30.md",
     "line": 37, "token": "0.9", "mutate": ("0.8 floor", "0.9 floor"),
     "claim": "with the 0.8 bar doctored to 0.9 the token anchors inside 0.9833333333333333, "
              "so post never reaches the bar noun"},
]


def _lp_scrub(line: str) -> str:
    """The shipped scrub made length-preserving. Offsets into the result ARE source columns."""
    def rep(m):
        return " " * (m.end() - m.start())
    s = C._SHAISH.sub(rep, line)
    s = C._DATEISH.sub(rep, s)
    return C._VERSIONISH.sub(rep, s)


def _shipped_scrub(line: str) -> str:
    s = C._SHAISH.sub(" ", line)
    s = C._DATEISH.sub(" ", s)
    return C._VERSIONISH.sub(" ", s)


def line_tokens(line: str, scrubber) -> list[tuple[str, int]]:
    """The shipped ``extract_numbers`` per-line loop, verbatim in its filters, keeping offsets."""
    line = line.replace("−", "-")
    scrub = scrubber(line)
    out = []
    for m in C._NUM.finditer(scrub):
        tok = m.group(0)
        raw = tok.replace(",", "")
        if C._YEAR.match(raw.lstrip("+-")):
            continue
        if (m.start() <= 2 and "." not in raw and abs(int(raw)) < 10
                and scrub[m.end():m.end() + 1] != "/"
                and C._MD_STRUCTURE.match(line)):
            continue
        if C._FORMULA_AFTER.match(scrub[m.end():]):
            continue
        if m.start() >= 2 and scrub[m.start() - 1] in "–-−" \
                and scrub[m.start() - 2].isdigit():
            continue
        if m.start() >= 2 and scrub[m.start() - 1] == "-" and scrub[m.start() - 2].isalpha():
            continue
        try:
            float(raw)
        except ValueError:
            continue
        out.append((tok, m.start()))
    return out


def windows(ctx: str, at: int, tok: str) -> tuple[str, str]:
    pre = ctx[max(0, at - 18):at] if at >= 0 else ""
    post = ctx[at + len(tok):] if at >= 0 else ""
    return pre, post


def predicates(pre: str, post: str) -> dict:
    """Every ``certify_doc`` predicate that reads a window, at the shipped flag settings."""
    spec_ops = "≥≤<>=≈~∼" if C.V05_APPROX_NOTATION else "≥≤<>="
    return {
        "is_spec_core": bool(re.search(r"[" + spec_ops + r"]\s*\+?$|\b(bar|gate|threshold|"
                                       r"requires?|must|pre-?registered)\b[^.]{0,16}$", pre))
                        or bool(re.match(r"\s*%?\s*(CI|confidence)", post))
                        or bool(re.match(r"[^.\d]{0,12}\b(bar|threshold|gate)\b", post)),
        "json_bar_key": bool(C._JSON_BAR_KEY.search(pre)),
        "bar_noun_post": bool(C._BAR_NOUN_POST.match(post)),
        "unit_range": bool(re.match(r"\s*[–-]\s*\d+(\.\d+)?\s*[BMK]\b", post))
                      or (bool(re.search(r"\d\s*[–-]\s*$", pre))
                          and bool(re.match(r"\s*[BMK]\b", post))),
        "at_param": bool(re.search(r"@\s*$", pre)),
        "unit_kw": bool(re.search(r"\b(aurocs?|aucs?|recall|precision|accuracy|fpr|fnr|"
                                  r"concordance|stability|rates?|p)\s*[(=:≈~\s]*$",
                                  pre, re.I)),
        "sign_kw": bool(re.search(r"\b(margins?|deltas?|elevation)\s*[(=:≈~\s]*$", pre, re.I)),
        "n_eq": bool(re.search(r"\bn\s*=\s*$", pre, re.I)),
        "slash_pair": bool(re.search(r"/\s*$", pre)) or bool(re.match(r"\s*/", post)),
        "derived_pct": bool(re.match(r"\s*%\s*\(\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", post)),
    }


def scan(paths):
    """One record per extracted token: its shipped anchor, its true column, its divergences."""
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:                                         # pragma: no cover - defensive
            continue
        rel = p.relative_to(ROOT).as_posix()
        for ln_no, raw_line in enumerate(text.splitlines(), 1):
            ship = line_tokens(raw_line, _shipped_scrub)
            lp = line_tokens(raw_line, _lp_scrub)
            if not ship:
                continue
            if [t for t, _ in ship] != [t for t, _ in lp]:
                yield {"rel": rel, "line": ln_no, "scrub_sequence_divergent": True}
                continue
            norm = raw_line.replace("−", "-")
            lead = len(norm) - len(norm.lstrip())
            ctx = norm.strip()
            for (tok, _), (_, col) in zip(ship, lp):
                at_find, at_true = ctx.find(tok), col - lead
                rec = {"rel": rel, "line": ln_no, "token": tok,
                       "find_at": at_find, "true_at": at_true,
                       "misplaced": at_find != at_true,
                       "anchored": ctx[at_true:at_true + len(tok)] == tok}
                if rec["misplaced"]:
                    a = predicates(*windows(ctx, at_find, tok))
                    b = predicates(*windows(ctx, at_true, tok))
                    rec["divergent_predicates"] = sorted(k for k in a if a[k] != b[k])
                yield rec


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
        except Exception:                                         # pragma: no cover - defensive
            continue
        receipts, missing, _ = _resolve_receipts(cp, rec)
        if receipts and not missing:
            out.append((doc, receipts))
    return out


def named_instances() -> list[dict]:
    out = []
    for spec in NAMED_INSTANCES:
        p = ROOT / spec["rel"]
        line = p.read_text(encoding="utf-8").splitlines()[spec["line"] - 1]
        if spec["mutate"]:
            line = line.replace(*spec["mutate"])
        norm = line.replace("−", "-")
        ctx, lead = norm.strip(), len(norm) - len(norm.lstrip())
        hit = next(((t, c) for t, c in line_tokens(line, _lp_scrub) if t == spec["token"]), None)
        if hit is None:
            out.append({"rel": spec["rel"], "line": spec["line"], "token": spec["token"],
                        "reproduced": False, "reason": "token not extracted"})
            continue
        at_true, at_find = hit[1] - lead, ctx.find(spec["token"])
        pf, sf = windows(ctx, at_find, spec["token"])
        pt, st = windows(ctx, at_true, spec["token"])
        a, b = predicates(pf, sf), predicates(pt, st)
        out.append({"rel": spec["rel"], "line": spec["line"], "token": spec["token"],
                    "mutated_for_this_probe": spec["mutate"], "claim": spec["claim"],
                    "find_at": at_find, "true_at": at_true, "reproduced": at_find != at_true,
                    "pre_at_find": pf, "pre_at_true": pt,
                    "post_at_find": sf[:48], "post_at_true": st[:48],
                    "divergent_predicates": sorted(k for k in a if a[k] != b[k])})
    return out


def main() -> int:
    md = [p for p in sorted(ROOT.glob("papers/**/*.md")) if "anc" not in p.parts]
    tot = mis = seq_div = unanchored = 0
    div_hist, per_doc = collections.Counter(), collections.Counter()
    for r in scan(md):
        if r.get("scrub_sequence_divergent"):
            seq_div += 1
            continue
        tot += 1
        unanchored += not r["anchored"]
        if r["misplaced"]:
            mis += 1
            per_doc[r["rel"]] += 1
            for k in r["divergent_predicates"]:
                div_hist[k] += 1

    docs = resolvable_docs()
    doc_rels = sorted(d.relative_to(ROOT).as_posix() for d, _ in docs)
    live = {}
    for d, rc in docs:
        for i, e in enumerate(C.certify_doc(d, rc)["ledger"]):
            live[(d.relative_to(ROOT).as_posix(), i)] = e["status"]

    roster, corpus_tokens, corpus_div = [], 0, 0
    idx = collections.Counter()
    for r in scan([ROOT / rel for rel in doc_rels]):
        if r.get("scrub_sequence_divergent"):
            continue
        i = idx[r["rel"]]
        idx[r["rel"]] += 1
        corpus_tokens += 1
        if not r["misplaced"]:
            continue
        corpus_div += bool(r["divergent_predicates"])
        roster.append({"rel": r["rel"], "line": r["line"], "token": r["token"], "ledger_index": i,
                       "find_at": r["find_at"], "true_at": r["true_at"],
                       "divergent_predicates": r["divergent_predicates"],
                       "shipped_status": live.get((r["rel"], i))})

    payload = {
        "purpose": "v0.10 pre-fix census (PREREG_oath_v10_token_column_2026_08_23)",
        "generated_at_verifier_sha256":
            hashlib.sha256((ROOT / "styxx" / "certify.py").read_bytes()).hexdigest(),
        "repo_wide": {
            "documents": len(md), "tokens": tot, "misplaced": mis,
            "misplaced_pct": round(100.0 * mis / max(1, tot), 3),
            "documents_affected": len(per_doc),
            "tokens_not_anchored_at_true_column": unanchored,
            "divergent_predicate_histogram": dict(div_hist.most_common()),
            "worst_documents": per_doc.most_common(10),
        },
        "scrub_offset_check": {
            "shipped_substitution": "re.sub(pat, ' ', line) — collapses each match to ONE space",
            "length_preserving_substitution":
                "re.sub(pat, lambda m: ' ' * len(m.group(0)), line)",
            "shipped_is_length_preserving": False,
            "lines_where_length_preservation_changes_the_extracted_sequence": seq_div,
            "note": "0 means the length-preserving scrub is inert for extraction across the whole "
                    "papers/ corpus, so the offset repair buys its correct columns without "
                    "changing which tokens are extracted",
        },
        "certified_corpus": {
            "documents": len(docs), "tokens": corpus_tokens, "misplaced": len(roster),
            "misplaced_with_a_divergent_predicate": corpus_div,
            "shipped_status_of_roster":
                dict(collections.Counter(r["shipped_status"] for r in roster)),
            "roster": roster,
        },
        "named_instances": named_instances(),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rw = payload["repo_wide"]
    print(f"repo-wide: {rw['tokens']} tokens in {rw['documents']} docs, MISPLACED "
          f"{rw['misplaced']} ({rw['misplaced_pct']}%) across {rw['documents_affected']} docs")
    print(f"length-preserving scrub changes extraction on {seq_div} lines")
    print(f"certified corpus: {len(docs)} docs, {corpus_tokens} tokens, {len(roster)} misplaced, "
          f"{corpus_div} with a divergent predicate")
    print(f"shipped status of roster: {payload['certified_corpus']['shipped_status_of_roster']}")
    print(f"divergent predicates: {dict(div_hist.most_common())}")
    for ni in payload["named_instances"]:
        print(f"  instance {ni['rel'].split('/')[-1]}:L{ni['line']} tok={ni['token']!r} "
              f"find={ni.get('find_at')} true={ni.get('true_at')} "
              f"reproduced={ni['reproduced']} diverges={ni.get('divergent_predicates')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
