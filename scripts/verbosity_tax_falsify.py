"""Falsify 'calibration is intrinsically wordier' — register vs prompt (PREREG_verbosity_tax_falsify).

Generates {calibrated, overconfident} under a HARD ONE-SENTENCE constraint on BOTH stances. If calibrated is
STILL longer under maximal brevity pressure, the verbosity is register-intrinsic; if equal brevity collapses
the gap, the morning claim was a prompt artifact. Also reports whether the hedge register survives brevity.

  python scripts/verbosity_tax_falsify.py --generate --model {gemma,qwen}
  python scripts/verbosity_tax_falsify.py            # analyze both
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from overconfidence_train_v0 import QUESTIONS, SYSTEM_CALIBRATED, SYSTEM_OVERCONFIDENT, featurize

CONCISE = " Answer in ONE sentence, as briefly as you possibly can."
REPOS = {"qwen": "Qwen/Qwen2.5-3B-Instruct", "gemma": "google/gemma-2-2b-it"}
OCD = ROOT / "benchmarks" / "data" / "overconfidence"
R_INTRINSIC, R_ARTIFACT = 1.08, 1.03
def cpath(tag): return OCD / f"pairs_concise_{tag}.jsonl"
def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def generate(tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = REPOS[tag]; out = cpath(tag)
    done = {(r["question"], r["condition"]) for r in load(out)}
    work = [(q, c) for q in QUESTIONS for c in ("calibrated", "overconfident") if (q, c) not in done]
    if not work:
        print(f"[gen] {tag} complete ({len(done)})"); return
    print(f"[gen] {repo}: {len(work)}")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="cuda").eval()
    no_sys = "gemma" in repo.lower(); OCD.mkdir(parents=True, exist_ok=True); n = 0
    with open(out, "a", encoding="utf-8") as f:
        for q, cond in work:
            sysp = (SYSTEM_CALIBRATED if cond == "calibrated" else SYSTEM_OVERCONFIDENT) + CONCISE
            msgs = ([{"role": "user", "content": sysp + "\n\nQuestion: " + q}] if no_sys
                    else [{"role": "system", "content": sysp}, {"role": "user", "content": q}])
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=96, do_sample=False, temperature=None, top_p=None,
                                   top_k=None, pad_token_id=tok.eos_token_id)
            t = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True).strip()
            if t:
                f.write(json.dumps({"question": q, "condition": cond, "response": t,
                                    "label_overconfident": 0 if cond == "calibrated" else 1}) + "\n"); f.flush(); n += 1
    print(f"[gen] {tag} wrote {n}"); del model; torch.cuda.empty_cache()


def analyze(tag):
    rows = load(cpath(tag))
    if not rows: return None
    X, y, names = featurize(rows)
    wc = np.array([len(r["response"].split()) for r in rows], float)
    cal, ovr = wc[y == 0].mean(), wc[y == 1].mean()
    r = cal / ovr
    hd = X[:, names.index("hedge_density")]
    hd_sd = (hd[y == 1].mean() - hd[y == 0].mean()) / (hd.std() or 1)
    print(f"  {tag:6s}: calib {cal:5.1f}w | overconf {ovr:5.1f}w | ratio {r:.3f} | hedge std-diff {hd_sd:+.3f}")
    return {"model": tag, "n": len(y), "calib_words": float(cal), "overconf_words": float(ovr),
            "ratio": float(r), "hedge_stddiff": float(hd_sd)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--generate", action="store_true")
    ap.add_argument("--model", choices=list(REPOS), default="gemma"); a = ap.parse_args()
    if a.generate:
        generate(a.model); return
    print("=== verbosity-tax falsification: ONE-sentence constraint on BOTH stances ===")
    res = [r for r in (analyze(t) for t in REPOS) if r]
    if len(res) >= 1:
        ratios = [r["ratio"] for r in res]
        if all(x >= R_INTRINSIC for x in ratios): verdict = "REGISTER-INTRINSIC (claim holds, hardened)"
        elif all(x <= R_ARTIFACT for x in ratios): verdict = "PROMPT-ARTIFACT (soften the public claim)"
        else: verdict = "PARTIAL / model-dependent (soften 'intrinsic' wording)"
        print(f"\n  ratios under hard brevity: {[round(x,3) for x in ratios]}")
        print(f"  >>> VERDICT (frozen prereg): {verdict}")
        (OCD / "_verbosity_tax_falsify_result.json").write_text(json.dumps({"results": res, "verdict": verdict,
            "thresholds": {"intrinsic>=": R_INTRINSIC, "artifact<=": R_ARTIFACT}}, indent=2))
        print("  wrote benchmarks/data/overconfidence/_verbosity_tax_falsify_result.json")


if __name__ == "__main__":
    main()
