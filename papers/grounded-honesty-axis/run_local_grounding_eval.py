"""Local grounding eval — does the $0 retrieval+NLI stack catch CONFIDENT misconceptions?
PREREG_local_grounding_2026_05_31.md. No API.

Reuses residuals_meta.json (Qwen2.5-3B answers + first-token entropy + correctness on TriviaQA) as the
test items. Pools the TriviaQA evidence for those questions into ONE reference corpus — so retrieval
must DISCRIMINATE the right passage among all questions' evidence (not open-book). Grounds each
CONFIDENT claim with local_ground.LocalGrounder. CATCH = P(refuted|confident-wrong);
PRESERVE = P(not refuted|confident-correct). White-box CATCH on this subset is ~0 (the wall).

  python run_local_grounding_eval.py --limit 150   # pilot (scan first 150 questions for evidence)
  python run_local_grounding_eval.py               # full
"""
from __future__ import annotations
import argparse, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from local_ground import LocalGrounder


def passages_from(text, maxlen=400, cap=8):
    text = (text or "").replace("\n", " ").strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) < maxlen:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return [p for p in out if len(p) > 20][:cap]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)   # 0 = scan until all questions found
    ap.add_argument("--quantile", type=float, default=0.5)  # confident = entropy below this quantile
    args = ap.parse_args(argv)

    meta = json.load(open(os.path.join(HERE, "residuals_meta.json"), encoding="utf-8"))
    rows = meta["rows"]
    ent = np.array([r["entropy"] for r in rows])
    thr = float(np.quantile(ent, args.quantile))
    qset = {r["q"].strip() for r in rows}

    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
    evid = {}
    scanned = 0
    for ex in ds:
        scanned += 1
        q = ex["question"].strip()
        if q in qset and q not in evid:
            ps = []
            ep = ex.get("entity_pages", {}) or {}
            for t in (ep.get("wiki_context") or []):
                ps += passages_from(t)
            sr = ex.get("search_results", {}) or {}
            for t in (sr.get("search_context") or []):
                ps += passages_from(t)
            if ps:
                evid[q] = ps
        if (args.limit and scanned >= args.limit) or len(evid) >= len(qset):
            break

    corpus = list(dict.fromkeys(p for ps in evid.values() for p in ps))
    print(f"scanned {scanned} rc examples; questions w/ evidence {len(evid)}; pooled corpus {len(corpus)} passages")
    if not corpus:
        print("no corpus — abort"); return

    g = LocalGrounder()
    g.index(corpus)

    results = []
    for r in rows:
        if r["entropy"] >= thr or r["q"].strip() not in evid:
            continue
        claim = f"{r['q'].rstrip('?').strip()}? The answer is {r['answer']}."
        v = g.ground(claim, k=5)
        results.append({"q": r["q"], "answer": r["answer"], "correct": bool(r["correct"]),
                        "verdict": v["verdict"], "conf": v.get("confidence")})

    wrong = [x for x in results if not x["correct"]]
    right = [x for x in results if x["correct"]]
    catch = (sum(1 for x in wrong if x["verdict"] == "refuted") / len(wrong)) if wrong else None
    preserve = (sum(1 for x in right if x["verdict"] != "refuted") / len(right)) if right else None
    powered = len(wrong) >= 20 and len(right) >= 20
    catch_pass = catch is not None and catch >= 0.50
    preserve_pass = preserve is not None and preserve >= 0.80
    result = "SURVIVED" if (catch_pass and preserve_pass and powered) else "REPORT_AS_LANDED"

    print(f"\nconfident grounded: {len(results)}  (wrong {len(wrong)}, right {len(right)})  powered={powered}")
    print(f"CATCH    P(refuted | confident-WRONG)   = {catch}  (>=0.50 -> {catch_pass})")
    print(f"PRESERVE P(not-refuted | confident-RIGHT)= {preserve}  (>=0.80 -> {preserve_pass})")
    print(f"RESULT = {result}")
    print("--- sample verdicts ---")
    for x in results[:12]:
        print(f"  [{'OK ' if x['correct'] else 'WRONG'}] {x['verdict']:9} {str(x['answer'])[:22]:24} | {x['q'][:46]}")

    summary = {"experiment": "local retrieval+NLI grounding on confident misconceptions ($0, no API)",
               "prereg": "papers/grounded-honesty-axis/PREREG_local_grounding_2026_05_31.md",
               "n_corpus": len(corpus), "n_questions_evid": len(evid), "n_grounded": len(results),
               "n_wrong": len(wrong), "n_right": len(right), "powered": powered,
               "CATCH": catch, "PRESERVE": preserve,
               "bars": {"CATCH>=0.50": catch_pass, "PRESERVE>=0.80": preserve_pass, "powered": powered},
               "RESULT": result, "results": results}
    json.dump(summary, open(os.path.join(HERE, "local_grounding_result.json"), "w"), indent=2)
    print("wrote local_grounding_result.json")


if __name__ == "__main__":
    main()
