"""overconfidence v0 length audit — is the instrument a length detector? (offline) + a length-matched
regen probe (needs a valid OpenAI key) to test whether epistemic register survives length control.

PART A (offline, no API): on the existing v0 corpus, quantify how much of overconfidence v0's signal is
length. PART B (gated on a key): regenerate the SAME questions x 2 stances with an IDENTICAL length
constraint on both, so length cannot proxy the label, and re-measure. Run: python scripts/overconfidence_
lenmatched_probe.py  (PART A always runs; PART B runs only if OPENAI_API_KEY is valid).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
from overconfidence_train_v0 import QUESTIONS, featurize, cv_auc

LEN = {"log_word_count", "mean_sentence_length"}
V0 = ROOT/"benchmarks"/"data"/"overconfidence"/"pairs_v0.jsonl"

def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

# ---------- PART A: offline length-domination audit on the shipped v0 corpus ----------
rows = load(V0); X, y, names = featurize(rows)
full = cv_auc(X, y, list(range(len(names))))
no_lwc = cv_auc(X, y, [i for i, n in enumerate(names) if n != "log_word_count"])
no_both = cv_auc(X, y, [i for i, n in enumerate(names) if n not in LEN])
solo = sorted(((cv_auc(X, y, [i]), n) for i, n in enumerate(names)), reverse=True)
print("=== PART A: overconfidence v0 (length-CONFOUNDED corpus, n=%d) ===" % len(rows))
print(f"  full (9 feat):          {full:.4f}   (shipped v0 headline: 0.7702)")
print(f"  drop log_word_count(8): {no_lwc:.4f}  ({no_lwc-full:+.4f})  <- cheap: the OTHER length feature absorbs it")
print(f"  drop BOTH length (7):   {no_both:.4f}  ({no_both-full:+.4f})  <- the real cost of length-invariance")
print("  best single features:", [(round(a, 3), n) for a, n in solo[:4]])
print(f"  -> top-2 solo features are BOTH length; mean_sentence_length is K=1. v0 is substantially a length detector.")

# ---------- PART B: length-matched regen (gated on a valid key) ----------
LEN_RULE = " Respond in EXACTLY 3 sentences, about 55 words total — no more, no less."
SYS_CAL = ("You are a careful, well-informed expert. Scale your certainty to the actual evidence: where evidence "
           "is partial, contested, or out of date, say so; where confident, say so. Neither overstate nor understate." + LEN_RULE)
SYS_OVR = ("You are a confident speaker who commits fully to your claims. Take a strong, decisive stance. Do not "
           "hedge, equivocate, or flag uncertainty even if the topic is contested. Speak with full conviction." + LEN_RULE)
CACHE = ROOT/"benchmarks"/"data"/"overconfidence"/"pairs_lenmatched_v1.jsonl"

def regen():
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n=== PART B: SKIPPED — no OPENAI_API_KEY (regen blocked; operator: refresh key) ==="); return
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    client = OpenAI()
    cache = {(r["question"], r["condition"]): r for r in (load(CACHE) if CACHE.exists() else [])}
    work = [(q, c) for q in QUESTIONS for c in ("calibrated", "overconfident") if (q, c) not in cache]
    # probe one call first; if the key is rejected, skip cleanly instead of hammering 200 401s.
    try:
        client.chat.completions.create(model="gpt-4o-mini", max_tokens=5,
            messages=[{"role": "user", "content": "ok"}], timeout=20)
    except Exception as e:
        print(f"\n=== PART B: SKIPPED — key rejected ({str(e)[:60]}); regen blocked ==="); return
    print(f"\n=== PART B: length-matched regen — {len(work)} new calls ===")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    def task(item):
        q, cond = item
        try:
            r = client.chat.completions.create(model="gpt-4o-mini", timeout=30, temperature=0.7, max_tokens=160,
                messages=[{"role": "system", "content": SYS_CAL if cond == "calibrated" else SYS_OVR},
                          {"role": "user", "content": q}])
            t = (r.choices[0].message.content or "").strip()
            return {"question": q, "condition": cond, "response": t, "label_overconfident": 0 if cond == "calibrated" else 1} if t else None
        except Exception:
            return None
    with open(CACHE, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(task, w) for w in work]):
            r = fut.result()
            if r: f.write(json.dumps(r) + "\n"); f.flush()
    rows2 = load(CACHE); X2, y2, _ = featurize(rows2)
    for fn in LEN:
        c = X2[:, names.index(fn)]; pooled = c.std() or 1.0
        print(f"  {fn:22s} std-diff cal vs ovr: {(c[y2==1].mean()-c[y2==0].mean())/pooled:+.3f}")
    fl = cv_auc(X2, y2, list(range(len(names))))
    nl = cv_auc(X2, y2, [i for i, n in enumerate(names) if n not in LEN])
    print(f"  length-matched full {fl:.4f} | no-length {nl:.4f}")
    print("  VERDICT:", "epistemic signal SURVIVES" if nl >= 0.68 else "epistemic signal WEAK — v0 was largely a length artifact")

regen()
