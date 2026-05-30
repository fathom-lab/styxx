"""Defying the wall: does EXTERNAL RETRIEVAL catch the confident factual misconceptions that every
model-internal / model-vs-model signal is structurally blind to?

The cross-model finding (FINDING_truthfulqa_crossmodel_2026_05_30.md) proved self-consistency,
cross-model disagreement, and LLM-judging all fail on confident factual misconceptions, because the
survivors are SHARED across models and the judge. The only lever it named was external ground truth.
This tests that lever with the cleanest possible contrast: the SAME model with vs without web
retrieval -- gpt-4o-mini (relies on its prior) vs gpt-4o-mini-search-preview (grounded in retrieved
web content) -- on the hard SHARED misconceptions that defeat model-internal methods + true controls.
Ground-truth labels committed pre-scoring (retrieval_grounding_data.json, commit 83fde91).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
PLAIN, SEARCH = "gpt-4o-mini", "gpt-4o-mini-search-preview"
_cl = OpenAI()
ASK = ("Is the following claim factually TRUE or FALSE? Begin your answer with exactly one word, "
       "TRUE or FALSE, then one sentence of justification.\n\nClaim: {claim}")


def _verdict(model, claim, use_temp):
    for attempt in range(5):
        try:
            kw = {"temperature": 0} if use_temp else {}
            r = _cl.chat.completions.create(
                model=model, max_tokens=90,
                messages=[{"role": "user", "content": ASK.format(claim=claim)}], **kw)
            t = (r.choices[0].message.content or "").strip()
            head = t[:40].upper()
            v = "FALSE" if ("FALSE" in head and head.find("FALSE") <= head.find("TRUE") + 1
                           or ("FALSE" in head and "TRUE" not in head)) else ("TRUE" if "TRUE" in head else None)
            # robust: first standalone token
            m = re.search(r"\b(TRUE|FALSE)\b", t.upper())
            v = m.group(1) if m else v
            return v, t[:160]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))


def main():
    data = json.loads((HERE / "retrieval_grounding_data.json").read_text(encoding="utf-8"))
    items = data["items"]
    rows = []
    for it in items:
        truth = "TRUE" if it["label"] else "FALSE"
        pv, pt = _verdict(PLAIN, it["claim"], use_temp=True)
        sv, st = _verdict(SEARCH, it["claim"], use_temp=False)
        rows.append({"kind": it["kind"], "truth": truth, "plain": pv, "search": sv,
                     "plain_ok": pv == truth, "search_ok": sv == truth,
                     "claim": it["claim"][:60], "plain_txt": pt, "search_txt": st})
        flip = "  <== RETRIEVAL FIXED" if (pv != truth and sv == truth) else (
            "  <== retrieval broke" if (pv == truth and sv != truth) else "")
        print(f"[{it['kind'][:18]:18}] truth={truth:5} plain={str(pv):5} search={str(sv):5}{flip}  {it['claim'][:52]}")

    def acc(rs, key):
        return sum(r[key] for r in rs) / len(rs) if rs else 0.0

    misc = [r for r in rows if r["kind"] == "shared_misconception"]
    myth = [r for r in rows if r["kind"] == "classic_myth"]
    true = [r for r in rows if r["kind"] == "true_control"]
    fixed = [r for r in rows if not r["plain_ok"] and r["search_ok"]]
    broke = [r for r in rows if r["plain_ok"] and not r["search_ok"]]

    print(f"\nRetrieval grounding (n={len(rows)}): plain gpt-4o-mini vs gpt-4o-mini-search-preview")
    print(f"  OVERALL accuracy : plain {acc(rows,'plain_ok'):.3f}  ->  retrieval {acc(rows,'search_ok'):.3f}")
    print(f"  SHARED misconceptions (the wall): plain {acc(misc,'plain_ok'):.3f}  ->  retrieval {acc(misc,'search_ok'):.3f}  (n={len(misc)})")
    print(f"  classic myths    : plain {acc(myth,'plain_ok'):.3f}  ->  retrieval {acc(myth,'search_ok'):.3f}  (n={len(myth)})")
    print(f"  true controls    : plain {acc(true,'plain_ok'):.3f}  ->  retrieval {acc(true,'search_ok'):.3f}  (n={len(true)})")
    print(f"  retrieval FIXED {len(fixed)} plain errors; BROKE {len(broke)} plain-correct items")
    for r in fixed:
        print(f"    FIXED [{r['kind'][:18]}] {r['claim']}")
    (HERE / "retrieval_grounding_result.json").write_text(json.dumps({
        "plain": PLAIN, "search": SEARCH, "n": len(rows),
        "plain_accuracy": round(acc(rows, "plain_ok"), 4), "retrieval_accuracy": round(acc(rows, "search_ok"), 4),
        "misconception_plain": round(acc(misc, "plain_ok"), 4), "misconception_retrieval": round(acc(misc, "search_ok"), 4),
        "n_fixed": len(fixed), "n_broke": len(broke), "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
