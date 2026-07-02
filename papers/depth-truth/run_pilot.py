"""Keystone pilot runner — PREREG v2 (§6). KG0 pipeline-validity + timing + KG1 degeneracy + A1-SPAN evidence.

GPU glue. Reuses the depth instrument VERBATIM from research/experiment_12_power.py@fc6f2c3 (get_mean_depth).
v2 deltas vs v1 (per PREREG_v2 changes table): hardened 5-shot prompt (Appendix A), max_new_tokens 16,
hardened extraction (strip HTML/list-num/quotes; Appendix B), A1 target = first CONTENT TOKEN of the extracted
span, KG0 gate in the report, FRESH 20 pilot items (seed 11, disjoint from v1 pilot ids + the seed-7 n_ID window).
Computes NO hypothesis statistic. Writes *_v2 outputs; v1 receipts (pilot_results.jsonl) are left untouched.

Every signal is guarded so a wrong-path error records and the rest proceed. Run:
  python papers/depth-truth/run_pilot.py            # 20 items
  python papers/depth-truth/run_pilot.py --n 3      # quick smoke
"""
import argparse
import gc
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

import qa_data as D   # harness/qa_data.py  (loaders + normalize + grade)
import signals as S   # harness/signals.py  (LP_*, SE, refusal)

# ---- frozen config (PREREG_v2 §1 + Appendix A) ------------------------------
MODEL_NAME = "google/gemma-2-2b"          # BASE, revision main (branch A won; substrate unchanged)
GEN_SEED = 7                              # generation/SE seeds
PILOT_SHUFFLE_SEED = 11                   # §1: pilot draws from a seed-11 shuffle (fresh)
MAIN_NID_MAX = 1000                       # §3: exclude the maximal seed-7 n_ID window
MAX_NEW_TOKENS = 16                       # v2 delta #2 (was 32)
SE_K = 5
SE_TEMP = 0.7
PILOT_N = 20

# Appendix A v2 — hardened 5-shot prompt (branch-A winner). {question} substituted per item.
HARD_PROMPT = (
    "Answer each question with only the answer. No numbering, no formatting, no explanation.\n\n"
    "Q: What is the chemical symbol for gold?\nA: Au\n\n"
    "Q: How many sides does a hexagon have?\nA: six\n\n"
    "Q: In what year did the first man walk on the Moon?\nA: 1969\n\n"
    "Q: What is the largest planet in the Solar System?\nA: Jupiter\n\n"
    "Q: Who wrote the play Romeo and Juliet?\nA: William Shakespeare\n\n"
    "Q: {question}\nA:"
)

# Appendix B v2 — hardened extraction + first-content-token
_HTML = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\s*/?>")
_LIST = re.compile(r"^\s*\d+\s*[.)]\s*")
_LEAD_ART = re.compile(r"^(?:a|an|the)\b\s*", re.IGNORECASE)


def hardened_extract(gen: str):
    """Appendix B steps 1-5: first line -> strip HTML -> strip list-number -> strip quotes.
    Returns (span, excluded_flag). excluded_flag: 'nonanswer' if empty or >12 words, else None."""
    line = gen.split("\n", 1)[0]
    line = _HTML.sub("", line)
    line = _LIST.sub("", line)
    line = line.strip().strip('"').strip("'").strip()
    if not line:
        return "", "nonanswer"
    if len(line.split()) > 12:
        return line, "nonanswer"
    return line, None


def first_content_token(span: str, tok):
    """Appendix B step 6: strip a leading article + punctuation, tokenize, return the first token id
    decoded to its single-token string (single-token by construction for circuit_tracer). '' if none."""
    s = _LEAD_ART.sub("", span.strip())
    s = re.sub(r"^[^\w]+", "", s)
    if not s:
        return ""
    ids = tok.encode(s)
    if ids and getattr(tok, "bos_token_id", None) is not None and ids[0] == tok.bos_token_id:
        ids = ids[1:]
    if not ids:
        return ""
    return tok.decode([ids[0]])


OUT_DIR = os.path.join(HERE, "pilot")
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- depth instrument, VERBATIM from research/experiment_12_power.py@fc6f2c3 ----
def make_get_mean_depth(model, attribute, torch):
    def get_mean_depth(prompt, target):
        try:
            graph = attribute(prompt, model, attribution_targets=[target],
                              batch_size=16, max_feature_nodes=500, offload="cpu")
            af = graph.active_features
            result = (None, 0)
            if af is not None and af.shape[0] > 0:
                layers = af[:, 0].float().cpu().numpy()
                result = (float(np.mean(layers)), len(layers))
            del graph, af
            gc.collect(); torch.cuda.empty_cache()
            return result
        except Exception as e:
            log(f"  depth ERROR: {e}")
            gc.collect(); torch.cuda.empty_cache()
            return None, 0
    return get_mean_depth


def _gen(model, torch, prompt, greedy=True, temp=1.0, seed=0):
    tokens = model.to_tokens(prompt)
    tok = model.tokenizer
    nl = set(tok.encode("\n")) | set(tok.encode("\nQ"))
    if not greedy:
        torch.manual_seed(seed)
    lps, ids = [], []
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            last = model(tokens, return_type="logits")[0, -1].float()
            if greedy:
                nxt = int(last.argmax().item())
                lps.append(float(torch.log_softmax(last, dim=-1)[nxt].item()))
            else:
                nxt = int(torch.multinomial(torch.softmax(last / temp, dim=-1), 1).item())
            ids.append(nxt)
            tokens = torch.cat([tokens, torch.tensor([[nxt]], device=tokens.device)], dim=1)
            if nxt in nl:
                break
    return tok.decode(ids), lps


def pick_pilot_items(n):
    """§3 disjointness: seed-11 shuffle, first n whose id is in NEITHER the v1 pilot ids NOR the seed-7
    first-1000 (max n_ID) window. Writes pilot/v2_pilot_ids.json (exclusion sets + chosen)."""
    v1_ids = set()
    v1_path = os.path.join(OUT_DIR, "pilot_results.jsonl")
    if os.path.exists(v1_path):
        v1_ids = {json.loads(l)["id"] for l in open(v1_path, encoding="utf-8")}
    seed7_ids = {it["id"] for it in D.load_id_triviaqa(n=MAIN_NID_MAX, seed=GEN_SEED, skip=0)}
    excluded = v1_ids | seed7_ids
    pool = D.load_id_triviaqa(n=n + 100, seed=PILOT_SHUFFLE_SEED, skip=0)
    chosen = [it for it in pool if it["id"] not in excluded][:n]
    with open(os.path.join(OUT_DIR, "v2_pilot_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"pilot_shuffle_seed": PILOT_SHUFFLE_SEED, "n": len(chosen),
                   "excluded_v1_pilot": sorted(v1_ids), "excluded_seed7_nID_window": len(seed7_ids),
                   "chosen_ids": [it["id"] for it in chosen]}, f, indent=2)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=PILOT_N)
    args = ap.parse_args()

    import torch
    from circuit_tracer import ReplacementModel, attribute

    log("loading gemma-2-2b + gemmascope transcoders (circuit_tracer 'gemma' preset)...")
    t0 = time.time()
    model = ReplacementModel.from_pretrained(MODEL_NAME, "gemma", dtype=torch.bfloat16, backend="transformerlens")
    log(f"model loaded in {time.time()-t0:.1f}s")
    tok = model.tokenizer
    get_mean_depth = make_get_mean_depth(model, attribute, torch)

    log(f"selecting {args.n} FRESH pilot items (seed {PILOT_SHUFFLE_SEED}, disjoint from v1 pilot + n_ID window)")
    items = pick_pilot_items(args.n)

    rows, item_times = [], []
    for i, it in enumerate(items):
        t_item = time.time()
        prompt = HARD_PROMPT.format(question=it["question"])
        row = {"id": it["id"], "question": it["question"], "gold": it.get("gold", [])}
        stage = {}
        span, exflag = "", None
        try:
            t = time.time()
            gen, lps = _gen(model, torch, prompt, greedy=True)
            span, exflag = hardened_extract(gen)
            if exflag is None and S.is_refusal(D.normalize(span)):
                exflag = "nonanswer"
            row["raw_gen"], row["answer"], row["excluded_flag"] = gen, span, exflag
            row["LP_mean"] = S.lp_mean(lps) if lps else None
            row["LP_norm"] = S.lp_norm(sum(lps), len(lps)) if lps else None
            row["correct"] = bool(D.grade(span, it.get("gold", []))) if exflag is None else None
            stage["generate"] = time.time() - t
        except Exception as e:
            row["gen_error"] = f"{e}\n{traceback.format_exc()}"
            log(f"  item {i} generate ERROR: {e}")
        try:
            t = time.time()
            samples = [D.normalize(hardened_extract(_gen(model, torch, prompt, greedy=False, temp=SE_TEMP,
                                                          seed=GEN_SEED + k)[0])[0]) for k in range(SE_K)]
            row["SE"] = S.semantic_entropy(samples)
            stage["SE"] = time.time() - t
        except Exception as e:
            row["se_error"] = str(e); log(f"  item {i} SE ERROR: {e}")
        # depth — A1 target = first CONTENT TOKEN of the extracted span (v2)
        try:
            t = time.time()
            target = first_content_token(span, tok) if span else ""
            row["depth_target"] = target
            d, nfeat = get_mean_depth(prompt, target) if target else (None, 0)
            row["depth"], row["depth_n_features"] = d, nfeat
            if d is None and not row.get("excluded_flag"):
                row["excluded_flag"] = "depth_undefined"
            stage["depth"] = time.time() - t
        except Exception as e:
            row["depth_error"] = str(e); row["depth"] = None
            log(f"  item {i} depth ERROR: {e}")
        row["stage_seconds"] = stage
        row["item_seconds"] = time.time() - t_item
        item_times.append(row["item_seconds"]); rows.append(row)
        log(f"item {i+1}/{len(items)}: ans={row.get('answer','?')!r} tgt={row.get('depth_target','?')!r} "
            f"correct={row.get('correct')} depth={row.get('depth')} ({row['item_seconds']:.1f}s)")
        with open(os.path.join(OUT_DIR, "pilot_results_v2.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # ---- KG0 (pipeline validity) + KG1 (degeneracy) + timing — NO hypothesis stats ----
    n = len(rows)
    clean = sum(1 for r in rows if r.get("excluded_flag") is None and r.get("answer"))
    clean_rate = clean / max(1, n)
    depths = [r["depth"] for r in rows if r.get("depth") is not None]
    n_undef = sum(1 for r in rows if r.get("depth") is None)
    kg1_zero_var = (len(depths) > 1 and float(np.std(depths)) == 0.0)
    kg1_undef_frac = n_undef / max(1, n)
    # eyeball table for flobi (KG0 human half)
    with open(os.path.join(OUT_DIR, "pilot_grading_eyeball_v2.md"), "w", encoding="utf-8") as f:
        f.write("# KG0 grading eyeball (flobi) — 20/20, flag any disagreement on an unambiguous item\n\n")
        f.write("| id | question | model answer | gold (first) | auto |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['id']} | {r['question'][:70]} | {r.get('answer','')!r} | "
                    f"{(r.get('gold') or ['?'])[0]!r} | {r.get('correct')} |\n")
    report = {
        "prereg": "PREREG_v2.md", "n_items": n,
        "KG0_extraction_clean_rate": clean_rate,
        "KG0_extraction_pass": bool(clean_rate >= 0.90),
        "KG0_human_eyeball": "PENDING flobi — pilot/pilot_grading_eyeball_v2.md (need 20/20, zero disagreements)",
        "KG0_note": ("KG0 gates BEFORE KG1: extraction-clean >=90% AND flobi's 20/20 eyeball. "
                     "If KG0 fails -> STOP, iterate plumbing, new pilot (no hypothesis claim)."),
        "median_item_seconds": float(np.median(item_times)) if item_times else None,
        "mean_item_seconds": float(np.mean(item_times)) if item_times else None,
        "depth_n_valid": len(depths),
        "depth_mean": float(np.mean(depths)) if depths else None,
        "depth_std": float(np.std(depths)) if depths else None,
        "depth_min": float(np.min(depths)) if depths else None,
        "depth_max": float(np.max(depths)) if depths else None,
        "depth_undefined_frac": kg1_undef_frac,
        "KG1_zero_variance": kg1_zero_var,
        "KG1_triggered": bool(kg1_zero_var or kg1_undef_frac > 0.30),
        "correct_rate": sum(1 for r in rows if r.get("correct") is True) / max(1, n),
        "n_gen_errors": sum(1 for r in rows if "gen_error" in r),
        "n_depth_errors": sum(1 for r in rows if "depth_error" in r),
        "A0_hint": "median_item_seconds * n gives wall-clock; pick n_ID/n_OOD1 under ~8h (§7)",
        "HALT_boundary": ("PILOT COMPLETE — the machine STOPS here (PREREG_v2 §9). Human reads KG0 + KG1 + "
                          "timing, does the 20/20 grading eyeball, then commits A1 (SPAN freeze) + A0 (sizes) "
                          "as SEPARATE ratified commits, and only THEN fires the main run."),
    }
    with open(os.path.join(OUT_DIR, "pilot_timing_report_v2.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"\n{'='*54}\nPILOT v2 DONE — extraction-clean {clean}/{n} ({100*clean_rate:.0f}%), "
        f"depth {len(depths)}/{n} valid, correct {report['correct_rate']*100:.0f}%, median "
        f"{report['median_item_seconds']}s/item")
    log(f"KG0_extraction_pass={report['KG0_extraction_pass']} (>=90%) | KG1_triggered={report['KG1_triggered']}")
    log("HALT: pilot boundary (§9). KG0 needs flobi's 20/20 eyeball; then A1 span + A0 sizes before main.")


if __name__ == "__main__":
    main()
