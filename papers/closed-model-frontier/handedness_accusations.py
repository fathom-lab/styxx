# -*- coding: utf-8 -*-
"""handedness_accusations.py -- frozen by PREREG_handedness_accusations_2026_09_02.

Join the 366 panel-judged accusations of the external OATH corpus to the obligation source the
verifier's clause order assigns each token, re-derived from the corpus ledger (context, recorded
trigger words, column, value). No new judgement anywhere.

  python handedness_accusations.py [--smoke]
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.certify import V07_PRECISION_DIGITS, _TRIGGERS_CORR   # noqa: E402
from styxx.protocol import Experiment                               # noqa: E402

SMOKE = "--smoke" in sys.argv
PREREG = HERE / "PREREG_handedness_accusations_2026_09_02.md"
MAPPING = json.loads((HERE / "h_mapping.json").read_text(encoding="utf-8"))["declared_sources"]

# the verifier's own regexes, copied verbatim (styxx/certify.py, the range-sanity rule and its
# v0.10 slash-pair guard), so the re-derivation reads what the verifier read
_UNIT_KW = re.compile(r"\b(aurocs?|aucs?|recall|precision|accuracy|fpr|fnr|concordance|stability|rates?|p)\s*[(=:≈~\s]*$", re.I)
_SIGN_KW = re.compile(r"\b(margins?|deltas?|elevation)\s*[(=:≈~\s]*$", re.I)
_N_GLUED = re.compile(r"\bn\s*=\s*$", re.I)


def _decimals(tok: str) -> int:
    t = tok.replace(",", "")
    return len(t.split(".", 1)[1]) if "." in t else 0


def _locate(context: str, token: str, col):
    """Where the token sits in the stripped context: the recorded column adjusted for stripped
    leading whitespace, else the first occurrence. Returns (index, method)."""
    if isinstance(col, int):
        for k in range(0, 9):
            i = col - k
            if i >= 0 and context[i:i + len(token)] == token:
                return i, "column"
    i = context.find(token)
    return (i, "first_occurrence") if i >= 0 else (-1, "not_found")


def source_of(row: dict) -> tuple:
    """(source, locate_method) in the verifier's clause order, first-writer."""
    ctx = (row.get("context") or "").replace("−", "-")
    tok = row["token"]
    val = float(row["value"])
    dec = _decimals(tok)
    at, how = _locate(ctx, tok, row.get("col"))
    pre = ctx[max(0, at - 18):at] if at >= 0 else ""
    post = ctx[at + len(tok):] if at >= 0 else ""
    if row.get("obligating_words"):
        return "vocabulary", how
    if _N_GLUED.search(pre):
        return "n-glued", how
    if dec > 0 and -1.0 <= val <= 1.0 and _TRIGGERS_CORR.search(ctx):
        return "range-correlation", how
    if dec >= V07_PRECISION_DIGITS:
        return "precision", how
    slash_pair = bool(re.search(r"/\s*$", pre)) or bool(re.match(r"\s*/", post))
    out_of_range = (_UNIT_KW.search(pre) and not 0.0 <= val <= 1.0) or (_SIGN_KW.search(pre) and not -1.0 <= val <= 1.0)
    if out_of_range and not slash_pair:
        return "range-sanity", how
    return "unknown", how


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def main() -> int:
    adj = json.loads((HERE / "oath_adjudication_result.json").read_text(encoding="utf-8"))
    ledger = [json.loads(l) for l in (HERE / "oath_external_corpus_ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_key = collections.defaultdict(list)
    for r in ledger:
        by_key[(r["repo"], r["line"], str(r["token"]))].append(r)
    accused = adj["per_arm_detail"]["UNGROUNDED"]
    if SMOKE:
        accused = accused[:40]
    rows, ambiguous, fallback = [], 0, collections.Counter()
    for it in accused:
        cands = [r for r in by_key.get((it["repo"], it["line"], str(it["token"])), []) if r["status"] == "UNGROUNDED"]
        if len(cands) != 1:
            ambiguous += 1
            rows.append({"id": it["id"], "repo": it["repo"], "line": it["line"], "token": it["token"],
                         "panel": it["verdict"], "source": "ambiguous", "class": None, "locate": None})
            continue
        src, how = source_of(cands[0])
        fallback[how] += 1
        rows.append({"id": it["id"], "repo": it["repo"], "line": it["line"], "token": it["token"],
                     "panel": it["verdict"], "source": src,
                     "class": MAPPING[src]["handed_by"] if src in MAPPING else None, "locate": how})
    n = len(rows)
    unknown = sum(1 for r in rows if r["source"] in ("unknown", "ambiguous"))
    def cell(pred):
        sel = [r for r in rows if pred(r)]
        k = sum(1 for r in sel if r["panel"] == "CLAIM")
        return {"n": len(sel), "genuine": k, "genuine_share": round(k / len(sel), 4) if sel else None,
                "wilson95": wilson(k, len(sel))}
    by_class = {c: cell(lambda r, c=c: r["class"] == c) for c in ("object_text", "object_form")}
    by_source = {s: cell(lambda r, s=s: r["source"] == s) for s in ("vocabulary", "n-glued", "range-correlation", "precision", "range-sanity", "unknown", "ambiguous")}
    ft, fm = by_class["object_form"]["genuine_share"], by_class["object_text"]["genuine_share"]
    delta = (ft - fm) if (ft is not None and fm is not None) else None
    conc = {c: collections.Counter(r["repo"] for r in rows if r["class"] == c).most_common(3) for c in ("object_text", "object_form")}
    metrics = {
        "unknown_or_ambiguous_share": round(unknown / n, 4) if n else 1.0,
        "min_cell_n": min(by_class["object_text"]["n"], by_class["object_form"]["n"]),
        "delta_form_minus_text": round(delta, 4) if delta is not None else -1.0,
        "delta_text_minus_form": round(-delta, 4) if delta is not None else -1.0,
    }
    verdict = Experiment(PREREG, repo_root=ROOT).score(metrics, smoke=SMOKE)
    res = {"prereg": PREREG.name, "smoke": SMOKE, "accusations_joined": n, "unknown": unknown, "ambiguous": ambiguous,
           "locate_methods": dict(fallback), "by_class": by_class, "by_source": by_source,
           "concentration_top3_repos": conc, "metrics": metrics, "verdict": verdict.verdict, "gates": verdict.gates,
           "rows": rows, "what_this_is_not": "no token was re-judged; the panel verdicts are the 2026-08-27 receipt's, and the sources are the verifier's clause order re-derived from the harness ledger, not from the documents"}
    out = HERE / ("handedness_accusations_smoke.json" if SMOKE else "handedness_accusations_result.json")
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("by class:", {c: (v["n"], v["genuine_share"], v["wilson95"]) for c, v in by_class.items()})
    print("by source:", {s: (v["n"], v["genuine_share"]) for s, v in by_source.items() if v["n"]})
    print("locate:", dict(fallback), "| metrics:", metrics)
    print(f"\n===== VERDICT: {res['verdict']} =====\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
