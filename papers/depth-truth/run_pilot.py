"""Keystone pilot runner (PREREG §6) — timing + KG1 degeneracy + A1 adaptation evidence ONLY.

This is the GPU glue. It reuses the depth instrument VERBATIM from research/experiment_12_power.py@fc6f2c3
(see get_mean_depth below) and drives 20 TriviaQA items OUTSIDE the main-run window (quarantined in pilot/).
It computes NO hypothesis statistic (§6). Its outputs: per-item wall-clock, the KG1 degeneracy check (depth
variance / undefined-rate across the 20), and the depth/signal values that inform the A1 adaptation freeze.

UNTESTED ON GPU at commit time (the card was held by a live probe). Every signal is wrapped so a wrong API
detail on one path still lets the others — especially depth, the KG1-critical one — run and be recorded, and
so the first real run yields a precise diagnostic trace rather than a bare crash. Per PREREG, a failure paste
is as useful as a timing paste.

Run:  python papers/depth-truth/run_pilot.py            # 20 items
      python papers/depth-truth/run_pilot.py --n 3      # quick smoke
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "harness"))
sys.path.insert(0, r"C:\Users\heyzo\clawd\research\circuit-tracer")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import qa_data as D   # harness/qa_data.py  (loaders + normalize + grade; NOT named 'datasets' — avoids shadowing HF)
import signals as S   # harness/signals.py   (extract_answer, LP_*, SE, refusal)

# ---- frozen config (PREREG §1 + Appendix A) ---------------------------------
MODEL_NAME = "google/gemma-2-2b"          # BASE, revision main
GEN_SEED = 7
MAX_NEW_TOKENS = 32
SE_K = 5
SE_TEMP = 0.7
PILOT_N = 20
PILOT_SKIP = 5000                         # well outside any n_ID<=1000 main-run window (§6)

# Appendix A — verbatim fixed 5-shot prompt (BASE model). {question} is substituted per item.
FIVE_SHOT = (
    "Answer each question with a short factual answer.\n\n"
    "Q: What is the chemical symbol for gold?\nA: Au\n\n"
    "Q: How many sides does a hexagon have?\nA: Six\n\n"
    "Q: In what year did the first human walk on the Moon?\nA: 1969\n\n"
    "Q: What is the largest planet in our solar system?\nA: Jupiter\n\n"
    "Q: Who wrote the play \"Romeo and Juliet\"?\nA: William Shakespeare\n\n"
    "Q: {question}\nA: "
)

OUT_DIR = os.path.join(HERE, "pilot")
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- the depth instrument, VERBATIM from research/experiment_12_power.py@fc6f2c3 -------------
# (model is bound in main(); attribute/torch imported there so import-time stays GPU-free.)
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
            gc.collect()
            torch.cuda.empty_cache()
            return result
        except Exception as e:
            log(f"  depth ERROR: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            return None, 0
    return get_mean_depth


def greedy_generate(model, torch, prompt):
    """Greedy decode (temp 0), max 32 new tokens, stop at first newline (§4/Appendix B).
    Returns (generation_text, token_logprobs:list[float]). HookedTransformer API (ReplacementModel
    is a transformerlens-backed model). Defensive: any API mismatch raises up to the per-item guard."""
    tokens = model.to_tokens(prompt)
    tok = model.tokenizer
    newline_ids = set(tok.encode("\n")) | set(tok.encode("\nQ"))  # newline or start of next Q
    logprobs, new_ids = [], []
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(tokens, return_type="logits")
            last = logits[0, -1].float()
            lp = torch.log_softmax(last, dim=-1)
            nxt = int(last.argmax().item())
            logprobs.append(float(lp[nxt].item()))
            new_ids.append(nxt)
            tokens = torch.cat([tokens, torch.tensor([[nxt]], device=tokens.device)], dim=1)
            if nxt in newline_ids:
                break
    gen = tok.decode(new_ids)
    return gen, logprobs


def sample_answer(model, torch, prompt, temp, seed):
    """One temp-sampled answer string (for discrete SE). Newline-stopped, extracted."""
    torch.manual_seed(seed)
    tokens = model.to_tokens(prompt)
    tok = model.tokenizer
    newline_ids = set(tok.encode("\n")) | set(tok.encode("\nQ"))
    new_ids = []
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(tokens, return_type="logits")
            last = logits[0, -1].float() / temp
            probs = torch.softmax(last, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            new_ids.append(nxt)
            tokens = torch.cat([tokens, torch.tensor([[nxt]], device=tokens.device)], dim=1)
            if nxt in newline_ids:
                break
    ans, _ = S.extract_answer(tok.decode(new_ids))
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=PILOT_N)
    args = ap.parse_args()

    import torch
    from circuit_tracer import ReplacementModel, attribute

    log("loading gemma-2-2b + gemmascope transcoders (circuit_tracer 'gemma' preset)...")
    t0 = time.time()
    model = ReplacementModel.from_pretrained(MODEL_NAME, "gemma", dtype=torch.bfloat16,
                                             backend="transformerlens")
    log(f"model loaded in {time.time()-t0:.1f}s")
    get_mean_depth = make_get_mean_depth(model, attribute, torch)

    log(f"loading {args.n} TriviaQA pilot items (skip={PILOT_SKIP}, seed={GEN_SEED}) — OUTSIDE main window")
    items = D.load_id_triviaqa(n=args.n, seed=GEN_SEED, skip=PILOT_SKIP)

    rows, item_times = [], []
    for i, it in enumerate(items):
        t_item = time.time()
        prompt = FIVE_SHOT.format(question=it["question"])
        row = {"id": it["id"], "question": it["question"], "gold": it.get("gold", [])}
        stage = {}
        # greedy answer + logprobs
        try:
            t = time.time()
            gen, lps = greedy_generate(model, torch, prompt)
            ans, exflag = S.extract_answer(gen)
            row["answer"], row["excluded_flag"] = ans, exflag
            row["LP_mean"] = S.lp_mean(lps) if lps else None
            row["LP_norm"] = S.lp_norm(sum(lps), len(lps)) if lps else None
            row["correct"] = bool(D.grade(ans, it.get("gold", []))) if exflag is None else None
            stage["generate"] = time.time() - t
        except Exception as e:
            row["gen_error"] = f"{e}\n{traceback.format_exc()}"
            log(f"  item {i} generate ERROR: {e}")
            ans = row.get("answer", "")
        # semantic entropy (K samples)
        try:
            t = time.time()
            samples = [D.normalize(sample_answer(model, torch, prompt, SE_TEMP, GEN_SEED + k))
                       for k in range(SE_K)]
            row["SE"] = S.semantic_entropy(samples)
            stage["SE"] = time.time() - t
        except Exception as e:
            row["se_error"] = str(e)
            log(f"  item {i} SE ERROR: {e}")
        # depth — the KG1-critical signal (verbatim get_mean_depth); target = first answer token
        try:
            t = time.time()
            target = (ans.split() or [ans])[0] if ans else ans
            d, nfeat = get_mean_depth(prompt, target)
            row["depth"], row["depth_n_features"] = d, nfeat
            if d is None:
                row["excluded_flag"] = row.get("excluded_flag") or "depth_undefined"
            stage["depth"] = time.time() - t
        except Exception as e:
            row["depth_error"] = str(e)
            row["depth"] = None
            log(f"  item {i} depth ERROR: {e}")
        row["stage_seconds"] = stage
        row["item_seconds"] = time.time() - t_item
        item_times.append(row["item_seconds"])
        rows.append(row)
        dstr = f"{row.get('depth')}" if row.get("depth") is not None else "None"
        log(f"item {i+1}/{len(items)}: ans={row.get('answer','?')!r} correct={row.get('correct')} "
            f"depth={dstr} ({row['item_seconds']:.1f}s)")
        with open(os.path.join(OUT_DIR, "pilot_results.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # KG1 degeneracy check + timing report (§6, §8 KG1) — NO hypothesis stats.
    depths = [r["depth"] for r in rows if r.get("depth") is not None]
    n_undef = sum(1 for r in rows if r.get("depth") is None)
    kg1_zero_var = (len(depths) > 1 and float(np.std(depths)) == 0.0)
    kg1_undef_frac = n_undef / max(1, len(rows))
    kg1_triggered = bool(kg1_zero_var or kg1_undef_frac > 0.30)
    report = {
        "n_items": len(rows),
        "median_item_seconds": float(np.median(item_times)) if item_times else None,
        "mean_item_seconds": float(np.mean(item_times)) if item_times else None,
        "depth_n_valid": len(depths),
        "depth_mean": float(np.mean(depths)) if depths else None,
        "depth_std": float(np.std(depths)) if depths else None,
        "depth_min": float(np.min(depths)) if depths else None,
        "depth_max": float(np.max(depths)) if depths else None,
        "depth_undefined_frac": kg1_undef_frac,
        "KG1_zero_variance": kg1_zero_var,
        "KG1_triggered": kg1_triggered,
        "KG1_note": ("DEGENERATE: STOP hypothesis run, publish the domain result (§8 KG1)"
                     if kg1_triggered else "depth varies on short QA answers; hypothesis run may proceed"),
        "n_gen_errors": sum(1 for r in rows if "gen_error" in r),
        "n_depth_errors": sum(1 for r in rows if "depth_error" in r),
        "A0_hint": "median_item_seconds * n gives wall-clock; pick n_ID/n_OOD1 under ~8h (§7)",
        "HALT_boundary": ("PILOT COMPLETE — the machine STOPS here. No main run chains (PREREG §9). "
                          "Human reads this timing table + KG1 line, then commits A1 (adaptation freeze, "
                          "a judgment call on pilot data) and A0 (sample sizes) as SEPARATE commits, and "
                          "only THEN fires the main run."),
    }
    with open(os.path.join(OUT_DIR, "pilot_timing_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"\n{'='*54}\nPILOT DONE — {report['depth_n_valid']}/{report['n_items']} depths valid, "
        f"median {report['median_item_seconds']}s/item")
    log(f"KG1_triggered={report['KG1_triggered']} | {report['KG1_note']}")
    log(f"report -> {os.path.join(OUT_DIR, 'pilot_timing_report.json')}")
    log("HALT: pilot boundary reached. Nothing chains to the main run — A1 (adaptation) + A0 (sample "
        "sizes) are the human checkpoint before main fires (PREREG §9).")


if __name__ == "__main__":
    main()
