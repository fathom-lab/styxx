"""Does an unreported knob move a benchmark score?

Runs MMLU under the standard letter-scoring rule while varying only nuisance factors that
evaluation configurations do not report: inference batch size, the order items are batched in,
and (reported separately, as a declared change rather than a nuisance) numerical precision.

Preregistered at papers/v8/batch_nuisance_mmlu/PREREG_batch_nuisance_scores_2026_09_08.md, which
was written to disk before this file was pointed at any benchmark item. The hypotheses, the 0.5
percentage-point threshold, the kill gates and the analysis are fixed there and are not adjusted
here.

Scoring is one forward pass per question: the prompt ends with "Answer:" and the four options are
read off the log-softmax at the final position for the tokens " A", " B", " C", " D". The
prediction is the argmax. This is the common MMLU convention and it makes the score a direct
function of the final-position logits, which is precisely the quantity today's probes measured
moving under batch size.

Writes receipt.json; FAILED.txt on an exception. Nothing is skipped silently: a variant that
cannot run is recorded with its exception and voids the run per kill gate 3.
"""
from __future__ import annotations

import glob
import json
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "receipt.json"
FAILED = HERE / "FAILED.txt"
T0 = time.time()

MMLU_GLOB = r"C:/Users/heyzo/.cache/huggingface/hub/datasets--cais--mmlu/**/test-*.parquet"
MODELS = [
    ("meta-llama/Llama-3.2-1B-Instruct",
     r"C:\Users\heyzo\.cache\huggingface\hub\models--meta-llama--Llama-3.2-1B-Instruct\snapshots"),
    ("google/gemma-2-2b-it",
     r"C:\Users\heyzo\.cache\huggingface\hub\models--google--gemma-2-2b-it\snapshots"),
]
# Fixed before any run, per the preregistration.
SUBJECTS = ["high_school_mathematics", "philosophy", "professional_law",
            "college_computer_science", "moral_scenarios"]
PER_SUBJECT = 200
CAP = 1000
SEED = 7
LETTERS = ["A", "B", "C", "D"]


def load_items():
    files = sorted(glob.glob(MMLU_GLOB, recursive=True))
    if not files:
        raise SystemExit(f"no MMLU parquet under {MMLU_GLOB}")
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    items, excluded = [], []
    for subject in SUBJECTS:
        sub = df[df["subject"] == subject]
        taken = 0
        for i, row in sub.iterrows():
            if taken >= PER_SUBJECT:
                break
            choices = list(row["choices"])
            q = str(row["question"]).strip()
            if len(choices) != 4:
                excluded.append({"row": int(i), "subject": subject, "reason": "not four options"})
                continue
            if not q:
                excluded.append({"row": int(i), "subject": subject, "reason": "empty question"})
                continue
            items.append({
                "item_id": f"{subject}-{taken:04d}",
                "subject": subject,
                "question": q,
                "choices": [str(c) for c in choices],
                "answer": int(row["answer"]),
            })
            taken += 1
    items = items[:CAP]
    return items, excluded


def prompt_for(it):
    lines = [f"The following are multiple choice questions (with answers) about "
             f"{it['subject'].replace('_', ' ')}.", "", it["question"]]
    for letter, choice in zip(LETTERS, it["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def letter_token_ids(tok):
    """Token id for each option letter as it appears after 'Answer:' (with a leading space)."""
    ids, detail = [], {}
    for letter in LETTERS:
        cand = tok.encode(" " + letter, add_special_tokens=False)
        used = None
        if len(cand) == 1:
            used = cand[0]
        else:                                   # fall back to the bare letter's first token
            bare = tok.encode(letter, add_special_tokens=False)
            used = bare[0]
        ids.append(used)
        detail[letter] = {"with_space": cand, "chosen": used,
                          "decoded": tok.decode([used])}
    if len(set(ids)) != 4:
        raise SystemExit(f"letter tokens are not distinct: {detail}")
    return ids, detail


def score_variant(model, tok, items, order, batch_size, letter_ids, device):
    """One forward pass per item; return {item_id: {lps, pred, margin}}."""
    import torch
    by_id = {it["item_id"]: it for it in items}
    out = {}
    ids_in_order = list(order)
    for start in range(0, len(ids_in_order), batch_size):
        chunk = ids_in_order[start:start + batch_size]
        texts = [prompt_for(by_id[i]) for i in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        with torch.no_grad():
            # logits_to_keep=1 computes the head for the FINAL position only. Without it the
            # (batch, seq, vocab) tensor is ~6.5 GB for gemma-2 at batch 32, which OOMs an 8 GB
            # card. This changes memory, not arithmetic: the final-position row is identical.
            fwd = model(**enc, logits_to_keep=1)
        last = fwd.logits[:, -1, :].float()         # left padding => final position is the prompt end
        lsm = torch.log_softmax(last, dim=-1)
        for row, item_id in enumerate(chunk):
            lps = [float(lsm[row, t]) for t in letter_ids]
            order_desc = sorted(range(4), key=lambda k: lps[k], reverse=True)
            out[item_id] = {
                "lps": lps,
                "pred": order_desc[0],
                "margin": lps[order_desc[0]] - lps[order_desc[1]],
            }
        del enc, fwd, last, lsm
    return out


def load_model(snapshot_root, dtype_name):
    import torch
    from transformers import AutoModelForCausalLM
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(snapshot_root, torch_dtype=dtype,
                                             attn_implementation="eager")
    m.to("cuda").eval()
    print(f"    [load] {dtype_name} in {time.time() - t:.1f}s", flush=True)
    return m


def resolve(root):
    p = Path(root)
    revs = sorted(x.name for x in p.iterdir() if x.is_dir())
    if not revs:
        raise SystemExit(f"no snapshot under {root}")
    return str(p / revs[-1]), revs[-1]


def run_model(label, snapshot_root, items):
    import torch
    from transformers import AutoTokenizer

    snapshot, revision = resolve(snapshot_root)
    tok = AutoTokenizer.from_pretrained(snapshot)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    letter_ids, letter_detail = letter_token_ids(tok)
    print(f"  [{label}] rev {revision[:12]} letters {letter_ids}", flush=True)

    canonical = [it["item_id"] for it in items]

    def perm(seed):
        import random
        r = random.Random(seed)
        p = list(canonical)
        r.shuffle(p)
        return p

    variants = [
        ("ref_bs1", "bfloat16", 1, canonical),
        ("repeat_bs1", "bfloat16", 1, canonical),
        ("bs8", "bfloat16", 8, canonical),
        ("bs32", "bfloat16", 32, canonical),
        ("bs8_perm11", "bfloat16", 8, perm(11)),
        ("bs32_perm12", "bfloat16", 32, perm(12)),
        ("fp16_bs1", "float16", 1, canonical),
    ]

    runs, failures = {}, {}
    model, current = None, None
    for name, dtype_name, bs, order in variants:
        if dtype_name != current:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            model = load_model(snapshot, dtype_name)
            current = dtype_name
        t = time.time()
        try:
            torch.manual_seed(SEED)
            runs[name] = score_variant(model, tok, items, order, bs, letter_ids, "cuda")
            acc = sum(1 for it in items if runs[name][it["item_id"]]["pred"] == it["answer"]) / len(items)
            print(f"    [run] {name:12s} bs={bs:<3d} acc={acc:.4f} {time.time() - t:6.1f}s", flush=True)
        except Exception as e:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            failures[name] = f"{type(e).__name__}: {' '.join(str(e).split())[:300]}"
            print(f"    [FAIL] {name}: {failures[name]}", flush=True)
    del model
    torch.cuda.empty_cache()

    gold = {it["item_id"]: it["answer"] for it in items}
    per_variant = {}
    for name, res in runs.items():
        correct = sum(1 for i in canonical if res[i]["pred"] == gold[i])
        per_variant[name] = {"n": len(canonical), "correct": correct,
                             "accuracy": correct / len(canonical)}
    ref = runs.get("ref_bs1")
    if ref is not None:
        for name, res in runs.items():
            flips = [i for i in canonical if res[i]["pred"] != ref[i]["pred"]]
            per_variant[name]["flips_vs_ref"] = len(flips)
            per_variant[name]["flipped_item_ids"] = flips
            per_variant[name]["n_identical_margin"] = sum(
                1 for i in canonical if res[i]["margin"] == ref[i]["margin"])
            per_variant[name]["mean_ref_margin_of_flipped"] = (
                sum(ref[i]["margin"] for i in flips) / len(flips)) if flips else None

    nuis = [n for n in ("ref_bs1", "repeat_bs1", "bs8", "bs32", "bs8_perm11", "bs32_perm12")
            if n in per_variant]
    accs = [per_variant[n]["accuracy"] for n in nuis]
    spread_pp = (max(accs) - min(accs)) * 100.0 if accs else None

    union_flips = sorted({i for n in nuis if n != "ref_bs1"
                          for i in per_variant[n]["flipped_item_ids"]})
    unflipped = [i for i in canonical if i not in set(union_flips)]
    gates = {
        "determinism_control_passed": ("repeat_bs1" in per_variant
                                       and per_variant["repeat_bs1"]["flips_vs_ref"] == 0),
        "sanity_floor_passed": (per_variant.get("ref_bs1", {}).get("accuracy", 0) >= 0.25),
        "completeness_passed": len(failures) == 0,
    }
    return {
        "model": {"label": label, "revision": revision, "snapshot": snapshot},
        "letters": letter_detail,
        "per_variant": per_variant,
        "failures": failures,
        "kill_gates": gates,
        "nuisance_variants": nuis,
        "accuracy_spread_pp": spread_pp,
        "union_nuisance_flips": union_flips,
        "n_union_nuisance_flips": len(union_flips),
        "mean_ref_margin_flipped": (sum(ref[i]["margin"] for i in union_flips) / len(union_flips))
        if (ref and union_flips) else None,
        "mean_ref_margin_unflipped": (sum(ref[i]["margin"] for i in unflipped) / len(unflipped))
        if (ref and unflipped) else None,
        "precision_arm": {
            "accuracy": per_variant.get("fp16_bs1", {}).get("accuracy"),
            "flips_vs_ref": per_variant.get("fp16_bs1", {}).get("flips_vs_ref"),
            "delta_pp": ((per_variant["fp16_bs1"]["accuracy"] - per_variant["ref_bs1"]["accuracy"]) * 100.0)
            if ("fp16_bs1" in per_variant and "ref_bs1" in per_variant) else None,
        },
        "per_item": {i: {"subject": next(x["subject"] for x in items if x["item_id"] == i),
                         "gold": gold[i],
                         "ref_pred": ref[i]["pred"] if ref else None,
                         "ref_margin": ref[i]["margin"] if ref else None,
                         "preds": {n: runs[n][i]["pred"] for n in runs}}
                     for i in canonical},
    }


def main():
    import torch
    items, excluded = load_items()
    print(f"[data] {len(items)} items, {len(excluded)} excluded, "
          f"subjects={sorted({i['subject'] for i in items})}", flush=True)
    print(f"[env] torch {torch.__version__} gpu {torch.cuda.get_device_name(0)}", flush=True)

    results = {}
    for label, root in MODELS:
        print(f"[model] {label}", flush=True)
        try:
            results[label] = run_model(label, root, items)
        except Exception as e:
            results[label] = {"model": {"label": label}, "harness_failure":
                              f"{type(e).__name__}: {' '.join(str(e).split())[:400]}"}
            print(f"  [MODEL FAILED] {results[label]['harness_failure']}", flush=True)
        (HERE / "receipt_partial.json").write_text(
            json.dumps({"models": results}, indent=2, sort_keys=True),
            encoding="utf-8", newline="\n")

    receipt = {
        "study": "batch-size and order as an unreported nuisance on a scored benchmark",
        "prereg": "papers/v8/batch_nuisance_mmlu/PREREG_batch_nuisance_scores_2026_09_08.md",
        "benchmark": {"name": "cais/mmlu", "split": "test", "subjects": SUBJECTS,
                      "per_subject_cap": PER_SUBJECT, "total_items": len(items),
                      "excluded": excluded, "scoring": "letter log-probability at the final position, argmax"},
        "runtime": {"torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "python": sys.version.split()[0], "platform": platform.platform(),
                    "gpu": torch.cuda.get_device_name(0)},
        "seed": SEED,
        "models": results,
        "wallclock_s": round(time.time() - T0, 1),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    print("\n================ RESULT ================")
    for label, r in results.items():
        print(f"\n{label}   (kill gates: {r['kill_gates']})")
        print(f"  {'variant':14s} {'acc':>8s} {'flips':>6s} {'identical margins':>18s}")
        for n, v in r["per_variant"].items():
            print(f"  {n:14s} {v['accuracy']:8.4f} {v.get('flips_vs_ref', 0):6d} "
                  f"{v.get('n_identical_margin', 0):18d}")
        print(f"  accuracy spread across nuisance variants: {r['accuracy_spread_pp']:.3f} pp"
              f"   (prereg H4 threshold 0.5 pp)")
        print(f"  union of nuisance flips: {r['n_union_nuisance_flips']} of {len(items)}")
        print(f"  mean reference margin  flipped: {r['mean_ref_margin_flipped']}")
        print(f"                       unflipped: {r['mean_ref_margin_unflipped']}")
        print(f"  precision arm (fp16, declared not nuisance): {r['precision_arm']}")
    print(f"\nreceipt -> {RECEIPT}")


if __name__ == "__main__":
    try:
        if FAILED.exists():
            FAILED.unlink()
        main()
    except BaseException:
        FAILED.write_text(f"FAILED after {time.time() - T0:.1f}s\n\n{traceback.format_exc()}",
                          encoding="utf-8", newline="\n")
        print(traceback.format_exc())
        sys.exit(1)
