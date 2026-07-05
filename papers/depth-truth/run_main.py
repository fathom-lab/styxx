"""Keystone MAIN RUN (PREREG_v2, post-A0/A1). The hypothesis-data run: ID (TriviaQA n=250), OOD-1
(PopQA-rare n=133), OOD-2 (TruthfulQA-gen n=250 + 50-item stratified human-audit sample).

Per item (§4, identical to the passed v2 pilot): greedy decode (hardened 5-shot, max 16 tokens, newline stop)
-> hardened extraction (Appendix B, frozen as A1) -> mechanical grading (§3; TruthfulQA tri-state) -> LP_mean /
LP_norm -> discrete SE (K=5, temp 0.7) -> depth via get_mean_depth (VERBATIM instrument) on the first content
token. NO analysis here — §2 statistics run afterwards via harness/analysis.py on the results JSONs.

RESUMABLE: appends one JSONL row per completed item; on restart, done ids are skipped. Run:
  python papers/depth-truth/run_main.py
"""
import gc
import hashlib
import json
import os
import re
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "harness"))
sys.path.insert(0, r"C:\Users\heyzo\clawd\research\circuit-tracer")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import qa_data as D
import signals as S
from run_pilot import (HARD_PROMPT, MAX_NEW_TOKENS, SE_K, SE_TEMP, GEN_SEED, MODEL_NAME,
                       hardened_extract, first_content_token, make_get_mean_depth, _gen, log)

N_ID, N_OOD1, N_OOD2 = 250, 133, 250        # AMENDMENT A0 (ratified)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def _phash(p):
    return hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]


def _done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}


def grade_tri(ans, correct_answers, incorrect_answers):
    """TruthfulQA tri-state (§3): correct-list match -> True; incorrect-list -> False; neither -> ambiguous."""
    if D.grade(ans, correct_answers):
        return True, None
    if D.grade(ans, incorrect_answers):
        return False, None
    return None, "grade_ambiguous"


def run_arm(model, torch, get_mean_depth, tok, arm, items, out_path):
    done = _done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    log(f"[{arm}] {len(items)} items, {len(done)} already done, {len(todo)} to run")
    f = open(out_path, "a", encoding="utf-8")
    for i, it in enumerate(todo):
        t0 = time.time()
        prompt = HARD_PROMPT.format(question=it["question"])
        row = {"id": it["id"], "arm": arm, "prompt_hash": _phash(prompt)}
        span, exflag = "", None
        try:
            gen, lps = _gen(model, torch, prompt, greedy=True)
            span, exflag = hardened_extract(gen)
            if exflag is None and S.is_refusal(D.normalize(span)):
                exflag = "nonanswer"
            row["answer"], row["excluded_flag"] = span, exflag
            row["LP_mean"] = S.lp_mean(lps) if lps else None
            row["LP_norm"] = S.lp_norm(sum(lps), len(lps)) if lps else None
            if exflag is None:
                if arm == "ood2":
                    row["correct"], amb = grade_tri(span, it["correct_answers"], it["incorrect_answers"])
                    if amb:
                        row["excluded_flag"] = amb
                else:
                    row["correct"] = bool(D.grade(span, it.get("gold", [])))
            else:
                row["correct"] = None
        except Exception as e:
            row["gen_error"] = f"{e}\n{traceback.format_exc()[-300:]}"
            log(f"  [{arm}] {it['id']} generate ERROR: {e}")
        try:
            samples = [D.normalize(hardened_extract(_gen(model, torch, prompt, greedy=False,
                        temp=SE_TEMP, seed=GEN_SEED + k)[0])[0]) for k in range(SE_K)]
            row["SE"] = S.semantic_entropy(samples)
        except Exception as e:
            row["se_error"] = str(e)
        try:
            target = first_content_token(span, tok) if span else ""
            row["depth_target"] = target
            d, nfeat = get_mean_depth(prompt, target) if target else (None, 0)
            row["depth"], row["depth_n_features"] = d, nfeat
            if d is None and not row.get("excluded_flag"):
                row["excluded_flag"] = "depth_undefined"
        except Exception as e:
            row["depth_error"] = str(e)
            row["depth"] = None
        row["item_seconds"] = round(time.time() - t0, 1)
        f.write(json.dumps(row) + "\n")
        f.flush()
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            log(f"  [{arm}] {len(done)+i+1}/{len(items)} (last: {row.get('answer','?')!r} "
                f"correct={row.get('correct')} depth={row.get('depth')})")
    f.close()


def main():
    # fail fast: validate ALL loaders before touching the GPU (§3: mismatch => stop and report)
    log("validating loaders (fail-fast, pre-GPU)...")
    id_items = D.load_id_triviaqa(n=N_ID, seed=GEN_SEED, skip=0)
    ood1_items = D.load_ood1_popqa_rare(n=N_OOD1, seed=GEN_SEED)
    ood2_items = D.load_ood2_truthfulqa(n=N_OOD2)
    assert len(id_items) == N_ID and len(ood1_items) == N_OOD1 and len(ood2_items) == N_OOD2
    log(f"loaders OK: ID={len(id_items)} OOD1={len(ood1_items)} OOD2={len(ood2_items)}")

    import torch
    from circuit_tracer import ReplacementModel, attribute
    log("loading gemma-2-2b + gemmascope transcoders...")
    t0 = time.time()
    model = ReplacementModel.from_pretrained(MODEL_NAME, "gemma", dtype=torch.bfloat16,
                                             backend="transformerlens")
    log(f"model loaded in {time.time()-t0:.1f}s")
    tok = model.tokenizer
    gmd = make_get_mean_depth(model, attribute, torch)
    # freeze the resolved SAE manifest (prereg Phase-0 note)
    try:
        man = {"model": MODEL_NAME, "n_layers": int(model.cfg.n_layers),
               "transcoder_preset": "gemma", "dtype": "bfloat16"}
        json.dump(man, open(os.path.join(RES, "sae_manifest.json"), "w"), indent=1)
    except Exception:
        pass

    run_arm(model, torch, gmd, tok, "id",   id_items,   os.path.join(RES, "main_id.jsonl"))
    run_arm(model, torch, gmd, tok, "ood1", ood1_items, os.path.join(RES, "main_ood1.jsonl"))
    run_arm(model, torch, gmd, tok, "ood2", ood2_items, os.path.join(RES, "main_ood2.jsonl"))

    # 50-item stratified human-audit sample from OOD2 (§3), stratified by graded outcome
    rows = [json.loads(l) for l in open(os.path.join(RES, "main_ood2.jsonl"), encoding="utf-8")]
    strata = {"True": [], "False": [], "None": []}
    for r in rows:
        strata[str(r.get("correct"))].append(r)
    rng = np.random.default_rng(GEN_SEED)
    sample = []
    quota = {"True": 17, "False": 17, "None": 16}
    for k, q in quota.items():
        pool = strata[k]
        take = min(q, len(pool))
        sample += [pool[j] for j in rng.choice(len(pool), size=take, replace=False)] if pool else []
    with open(os.path.join(RES, "human_audit_sample.jsonl"), "w", encoding="utf-8") as f:
        for r in sample[:50]:
            f.write(json.dumps(r) + "\n")
    log(f"human_audit_sample.jsonl written ({len(sample[:50])} rows, stratified by grade)")

    open(os.path.join(RES, ".main_complete"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("MAIN RUN COMPLETE. Next (§11): analysis via harness/analysis.py -> verdict table. "
        "KG3 gate: flobi grades the 50-item audit sample before TruthfulQA claims.")


if __name__ == "__main__":
    main()
